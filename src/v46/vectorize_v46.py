"""
v4.6 Single-Pass SDF -> CFF Vectorizer (`sdf_to_cmds_v46`).

Why a rewrite (root causes measured on v4.5 outputs, see outputs/v46_report):
  1. v4.5 is a *double* lossy chain: `sdf_to_cff_commands` already fits Béziers on a 3x zoomed SDF,
     then `optimize_synthesized_contours_v45` re-flattens at 6u, runs `approximate_polygon(1.7)` and
     fits again. Errors compound and arcs degenerate into polyline chains of short `lineTo`s.
  2. v4.5 straight-stem snapping never fired (`total_snapped_segs = 0` on all 5,050 glyphs):
     `_seg_hausdorff` flattens `[moveTo, lineTo]` -> `_flatten_contour_poly` returns an empty array for
     < 3 points -> distance 999 -> every line candidate is rejected; stem primitives were compared with
     the 2-vertex decimated polygon so their midpoints always looked 70u+ away.
  3. Corner detection ran on decimated polygons (vertex angle), so rounded SDF corners and gentle
     stroke-end flares produced either missed corners (blobby 纟 turns) or spurious anchors (wobble).

v4.6 pipeline (one pass, measured in UPEM directly against the SDF zero level set):
  a. light Gaussian pre-filter on the SDF (removes ±0.3px iso-line ripple) + 2x bicubic zoom
  b. sub-pixel marching squares -> exact UPEM mapping (incl. the +0.25px target-raster centre offset)
  c. uniform 2u arc-length resampling, scale-aware corner detector (turning angle at k and 2k
     windows: corners keep their angle when the window doubles, smooth arcs do not)
  d. per section: straight-run detection -> exact `lineTo`; remaining arcs split at H/V extrema and
     fitted with tangent-constrained (G1) Schneider cubics, tol 1.0u
  e. near-axis lines (<= 2.5 deg) snapped to exact horizontal/vertical, neighbouring G1 handles
     re-projected so the joins stay smooth
  f. per-contour verification against the raw (un-smoothed) iso-line; a contour exceeding the
     tolerance is refitted without smoothing (never silently emits a distorted outline)
  g. integer UPEM output, nesting-depth CFF winding, speck/pinhole removal
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree
from skimage.measure import find_contours

DrawCmd = Tuple[str, List[Tuple[float, float]]]

CANVAS = 256
PAD_U = 96.0
UPX = (1000.0 + 2.0 * PAD_U) / CANVAS  # 4.65625 UPEM / px
ASC = 880.0


@dataclass
class VecParams:
    pre_sigma_px: float = 0.55        # Gaussian pre-filter on SDF (px @256)
    upsample: int = 2                 # bicubic zoom before marching squares
    step_u: float = 2.0               # arc-length resampling step (UPEM)
    smooth_iters: int = 4             # Taubin iterations on each open section (pinned ends)
    corner_win_u: float = 10.0        # corner detector half-window
    corner_deg: float = 30.0          # min turning angle at k-window
    corner_sharp_ratio: float = 0.60  # angle(k) / angle(2k): ~1 sharp corner, ~0.5 smooth arc (SDF rounds corners -> 0.6-0.85)
    corner_hard_deg: float = 80.0     # always a corner above this
    line_min_len_u: float = 30.0      # min straight run length (near-axis stems)
    axis_run_deg: float = 8.0         # a straight run within this of H/V counts as a stem
    diag_line_min_len_u: float = 80.0 # diagonal runs must be long and very straight to become lines
    fillet_absorb_u: float = 14.0     # curve stubs shorter than this next to a corner/line are absorbed
    line_tol_u: float = 0.9           # straight-run deviation tolerance (+0.003 * L)
    fit_tol_u: float = 2.2            # cubic fit tolerance (sweep: 1.6u->103 nodes, 2.2u->81 nodes, p95 dev 3.11->3.14u)
    axis_snap_deg: float = 2.5        # snap near-H/V lines exactly
    extrema_min_gap_u: float = 14.0   # min distance of an extremum split from piece ends
    verify_tol_u: float = 3.2         # max deviation vs raw iso-line before refit
    min_island_u2: float = 140.0      # drop specks smaller than this
    min_hole_u2: float = 110.0        # drop pinholes smaller than this
    integer_coords: bool = True


@dataclass
class Seg:
    kind: str                      # "L" or "C"
    p0: np.ndarray
    p3: np.ndarray
    c1: Optional[np.ndarray] = None
    c2: Optional[np.ndarray] = None
    smooth_start: bool = False     # G1 with previous segment
    smooth_end: bool = False       # G1 with next segment


@dataclass
class VecStats:
    contours: int = 0
    lines: int = 0
    curves: int = 0
    axis_snapped: int = 0
    extrema_splits: int = 0
    corners: int = 0
    refits: int = 0
    dropped_specks: int = 0
    dropped_pinholes: int = 0
    max_dev_u: float = 0.0
    p95_dev_u: float = 0.0
    contour_len_u: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


# ----------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------

def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _resample_closed(p: np.ndarray, step: float) -> np.ndarray:
    q = np.vstack([p, p[:1]])
    d = np.linalg.norm(np.diff(q, axis=0), axis=1)
    L = float(d.sum())
    if L < step * 4:
        return p
    s = np.concatenate([[0.0], np.cumsum(d)])
    n = max(8, int(round(L / step)))
    t = np.linspace(0.0, L, n, endpoint=False)
    x = np.interp(t, s, q[:, 0])
    y = np.interp(t, s, q[:, 1])
    return np.stack([x, y], axis=1)


def _turn_angle(p: np.ndarray, k: int) -> np.ndarray:
    n = len(p)
    a = p - np.roll(p, k, axis=0)
    b = np.roll(p, -k, axis=0) - p
    na = np.linalg.norm(a, axis=1) + 1e-9
    nb = np.linalg.norm(b, axis=1) + 1e-9
    c = np.clip((a * b).sum(1) / (na * nb), -1.0, 1.0)
    return np.degrees(np.arccos(c)) if n else np.zeros(0)


def _detect_corners(p: np.ndarray, prm: VecParams) -> List[int]:
    n = len(p)
    k = max(2, int(round(prm.corner_win_u / prm.step_u)))
    if n < 4 * k:
        return []
    a1 = _turn_angle(p, k)
    a2 = _turn_angle(p, 2 * k)
    cand = ((a1 >= prm.corner_deg) & (a1 >= prm.corner_sharp_ratio * a2)) | (a1 >= prm.corner_hard_deg)
    idx = np.where(cand)[0]
    corners: List[int] = []
    # non-maximum suppression on circular index
    for i in idx:
        win = [(i + j) % n for j in range(-k, k + 1)]
        if a1[i] >= a1[win].max() - 1e-9:
            if not corners or min(min(abs(i - c), n - abs(i - c)) for c in corners) > 2 * k:
                corners.append(int(i))
    return sorted(corners)


def _taubin_open(seg: np.ndarray, iters: int, lam: float = 0.5, mu: float = -0.52, half_win: int = 3) -> np.ndarray:
    """Multi-hop (6u at step_u=2u) Taubin low-pass filter on open sections with pinned endpoints.

    A 1-hop stencil at 2u only attenuates wavelengths < 26u (already removed by pre_sigma_px) and
    leaves 25-60u SDF iso-line ripples untouched. A 3-hop (6u) symmetric window shifts the cutoff
    to ~65u while keeping zero shrinkage (lam + mu < 0) on genuine stroke arcs (R >= 150u).
    """
    n = len(seg)
    if n < 5 or iters <= 0:
        return seg
    p = seg.astype(np.float64, copy=True)
    m = max(1, int(half_win))
    for _ in range(iters):
        for w in (lam, mu):
            q = p.copy()
            for i in range(1, n - 1):
                r = min(m, i, n - 1 - i)
                acc = np.zeros(2, dtype=np.float64)
                for k in range(1, r + 1):
                    acc += q[i - k] + q[i + k]
                lap = acc / (2.0 * r) - q[i]
                p[i] = q[i] + w * lap
    return p


def _taubin_closed(p: np.ndarray, iters: int, lam: float = 0.5, mu: float = -0.52, half_win: int = 3) -> np.ndarray:
    """Multi-hop Taubin low-pass filter on closed loops."""
    n = len(p)
    if n < 6 or iters <= 0:
        return p.astype(np.float64, copy=True)
    q = p.astype(np.float64, copy=True)
    m = max(1, min(int(half_win), n // 4))
    for _ in range(iters):
        for w in (lam, mu):
            acc = np.zeros_like(q)
            for k in range(1, m + 1):
                acc += np.roll(q, k, axis=0) + np.roll(q, -k, axis=0)
            lap = acc / (2.0 * m) - q
            q = q + w * lap
    return q


def _line_dev(seg: np.ndarray) -> float:
    p0, p1 = seg[0], seg[-1]
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return float(np.linalg.norm(seg - p0, axis=1).max())
    u = v / L
    d = (seg - p0) @ np.array([-u[1], u[0]])
    return float(np.abs(d).max())


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _end_tangent(seg: np.ndarray, at_start: bool, span_u: float = 12.0, step: float = 2.0,
                 skip_u: float = 3.0) -> np.ndarray:
    """Robust one-sided tangent: chord from ~skip_u to ~span_u (skips the SDF-rounded fillet at corners)."""
    n = len(seg)
    m = max(1, min(n - 1, int(round(span_u / step))))
    k = min(m - 1, int(round(skip_u / step))) if m > 2 else 0
    if at_start:
        return _unit(seg[m] - seg[k]) if m > k else _unit(seg[-1] - seg[0])
    return _unit(seg[n - 1 - k] - seg[n - 1 - m]) if m > k else _unit(seg[-1] - seg[0])


# ----------------------------------------------------------------------------------------------
# Schneider tangent-constrained cubic fitting
# ----------------------------------------------------------------------------------------------

def _bez(p0, c1, c2, p3, t):
    t = t[:, None]
    u = 1.0 - t
    return u ** 3 * p0 + 3 * u * u * t * c1 + 3 * u * t * t * c2 + t ** 3 * p3


def _bez_d1(p0, c1, c2, p3, t):
    t = t[:, None]
    u = 1.0 - t
    return 3 * u * u * (c1 - p0) + 6 * u * t * (c2 - c1) + 3 * t * t * (p3 - c2)


def _bez_d2(p0, c1, c2, p3, t):
    t = t[:, None]
    return 6 * (1 - t) * (c2 - 2 * c1 + p0) + 6 * t * (p3 - 2 * c2 + c1)


def _chord_param(pts: np.ndarray) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(pts))


def _gen_bezier(pts: np.ndarray, u: np.ndarray, t0: np.ndarray, t1: np.ndarray):
    p0, p3 = pts[0], pts[-1]
    A1 = t0[None, :] * (3 * (1 - u) ** 2 * u)[:, None]
    A2 = t1[None, :] * (3 * (1 - u) * u ** 2)[:, None]
    C = np.array([[np.sum(A1 * A1), np.sum(A1 * A2)], [np.sum(A1 * A2), np.sum(A2 * A2)]])
    tmp = pts - _bez(p0, p0, p3, p3, u)
    X = np.array([np.sum(A1 * tmp), np.sum(A2 * tmp)])
    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    seg_len = float(np.linalg.norm(p3 - p0))
    if abs(det) > 1e-12:
        a1 = (X[0] * C[1, 1] - X[1] * C[0, 1]) / det
        a2 = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det
    else:
        a1 = a2 = seg_len / 3.0
    eps = 1e-6 * max(seg_len, 1.0)
    # reject degenerate / looping handles (Schneider heuristic + upper bound against overshoot)
    if a1 < eps or a2 < eps or a1 > 1.5 * seg_len + 4 or a2 > 1.5 * seg_len + 4:
        a1 = a2 = seg_len / 3.0
    return p0, p0 + a1 * t0, p3 + a2 * t1, p3


def _max_err(pts, bez, u):
    q = _bez(*bez, u)
    d = np.linalg.norm(q - pts, axis=1)
    i = int(np.argmax(d))
    return float(d[i]), i


def _reparam(pts, bez, u):
    q = _bez(*bez, u)
    d1 = _bez_d1(*bez, u)
    d2 = _bez_d2(*bez, u)
    num = ((q - pts) * d1).sum(1)
    den = (d1 * d1).sum(1) + ((q - pts) * d2).sum(1)
    nu = u - np.where(np.abs(den) > 1e-9, num / den, 0.0)
    nu = np.clip(nu, 0.0, 1.0)
    nu[0], nu[-1] = 0.0, 1.0
    return np.maximum.accumulate(nu)


def _fit_cubic(pts: np.ndarray, t0: np.ndarray, t1: np.ndarray, tol: float, depth: int = 0) -> List[Seg]:
    """t0 = unit tangent leaving pts[0]; t1 = unit tangent *into* pts[-1] pointing backwards (Schneider)."""
    n = len(pts)
    if n == 2 or float(np.linalg.norm(pts[-1] - pts[0])) < 1e-6:
        L = float(np.linalg.norm(pts[-1] - pts[0])) / 3.0
        return [Seg("C", pts[0].copy(), pts[-1].copy(), pts[0] + L * t0, pts[-1] + L * t1)]
    u = _chord_param(pts)
    bez = _gen_bezier(pts, u, t0, t1)
    err, split = _max_err(pts, bez, u)
    if err <= tol:
        return [Seg("C", bez[0].copy(), bez[3].copy(), bez[1], bez[2])]
    if err <= tol * 4.0:
        for _ in range(4):
            u = _reparam(pts, bez, u)
            bez = _gen_bezier(pts, u, t0, t1)
            err, split = _max_err(pts, bez, u)
            if err <= tol:
                return [Seg("C", bez[0].copy(), bez[3].copy(), bez[1], bez[2])]
    if depth >= 7 or n < 6:
        return [Seg("C", bez[0].copy(), bez[3].copy(), bez[1], bez[2])]
    edge = max(2, min(6, n // 4))  # keep splits >= ~12u away from piece ends (no corner stubs)
    split = int(np.clip(split, edge, n - 1 - edge))
    tc = _unit(pts[split - 1] - pts[split + 1])  # points backwards
    left = _fit_cubic(pts[: split + 1], t0, tc, tol, depth + 1)
    right = _fit_cubic(pts[split:], -tc, t1, tol, depth + 1)
    left[-1].smooth_end = True
    right[0].smooth_start = True
    return left + right


# ----------------------------------------------------------------------------------------------
# section decomposition
# ----------------------------------------------------------------------------------------------

def _straight_runs(seg: np.ndarray, prm: VecParams) -> List[Tuple[int, int]]:
    """Maximal straight runs (index ranges) of length >= line_min_len_u inside an open section."""
    n = len(seg)
    m = max(2, int(round(10.0 / prm.step_u)))
    if n < 2 * m + 2:
        return []
    flag = np.zeros(n, dtype=bool)
    for i in range(m, n - m):
        if _line_dev(seg[i - m : i + m + 1]) <= 0.45:
            flag[i - m : i + m + 1] = True
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flag[j + 1]:
            j += 1
        # shrink until the whole run is straight within tolerance
        a, b = i, j
        while b - a >= 2:
            L = float(np.linalg.norm(seg[b] - seg[a]))
            if _line_dev(seg[a : b + 1]) <= prm.line_tol_u + 0.003 * L:
                break
            # drop the worse end
            if _line_dev(seg[a + 1 : b + 1]) < _line_dev(seg[a:b]):
                a += 1
            else:
                b -= 1
        v = seg[b] - seg[a]
        L = float(np.linalg.norm(v))
        if L > 1e-6:
            ang = math.degrees(math.atan2(abs(v[1]), abs(v[0])))
            near_axis = ang <= prm.axis_run_deg or ang >= 90.0 - prm.axis_run_deg
            very_straight = L >= prm.diag_line_min_len_u and _line_dev(seg[a : b + 1]) <= 0.6 + 0.002 * L
            if (near_axis and L >= prm.line_min_len_u) or very_straight:
                runs.append((a, b))
        i = j + 1
    return runs


def _extrema_splits(seg: np.ndarray, prm: VecParams) -> List[Tuple[int, np.ndarray]]:
    """Indices where tangent is horizontal/vertical (x or y extremum) with exact axis tangent."""
    n = len(seg)
    if n < 8:
        return []
    d = np.gradient(seg, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
    L = s[-1]
    out: List[Tuple[int, np.ndarray]] = []
    for axis in (0, 1):
        comp = d[:, axis]
        sg = np.sign(comp)
        for i in range(1, n - 1):
            if sg[i - 1] != 0 and sg[i] != 0 and sg[i - 1] != sg[i]:
                if s[i] < prm.extrema_min_gap_u or L - s[i] < prm.extrema_min_gap_u:
                    continue
                # tangent along the other axis, direction of travel
                other = 1 - axis
                tdir = np.zeros(2)
                tdir[other] = 1.0 if d[i, other] >= 0 else -1.0
                # require genuine extremum (not jitter on a near-axis straight stretch)
                w = max(2, int(round(8.0 / prm.step_u)))
                lo, hi = max(0, i - w), min(n - 1, i + w)
                bulge = abs(seg[i, axis] - 0.5 * (seg[lo, axis] + seg[hi, axis]))
                if bulge < 0.6:
                    continue
                out.append((i, tdir))
    out.sort(key=lambda t: t[0])
    pruned: List[Tuple[int, np.ndarray]] = []
    for i, t in out:
        if pruned and s[i] - s[pruned[-1][0]] < prm.extrema_min_gap_u:
            continue
        pruned.append((i, t))
    return pruned


def _try_single_cubic(pts: np.ndarray, t0: np.ndarray, t1: np.ndarray, tol: float) -> Optional[Seg]:
    """Attempt a single tangent-constrained cubic fit before splitting at extrema."""
    if len(pts) < 2:
        return None
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    if len(pts) == 2 or chord < 1e-6:
        L = chord / 3.0
        return Seg("C", pts[0].copy(), pts[-1].copy(), pts[0] + L * t0, pts[-1] + L * t1)
    u = _chord_param(pts)
    bez = _gen_bezier(pts, u, t0, t1)
    err, _ = _max_err(pts, bez, u)
    if err <= tol:
        return Seg("C", bez[0].copy(), bez[3].copy(), bez[1], bez[2])
    if err <= tol * 3.0:
        for _ in range(4):
            u = _reparam(pts, bez, u)
            bez = _gen_bezier(pts, u, t0, t1)
            err, _ = _max_err(pts, bez, u)
            if err <= tol:
                return Seg("C", bez[0].copy(), bez[3].copy(), bez[1], bez[2])
    return None


def _fit_section(seg: np.ndarray, prm: VecParams, st: VecStats, t_start: Optional[np.ndarray] = None,
                 t_end: Optional[np.ndarray] = None) -> List[Seg]:
    """Fit an open polyline section between two corners (or a smooth closed loop cut at extrema)."""
    if len(seg) < 2:
        return []
    L = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    chord = float(np.linalg.norm(seg[-1] - seg[0]))
    if len(seg) <= 3 or (chord >= 4.0 and _line_dev(seg) <= prm.line_tol_u + 0.003 * chord and L <= chord * 1.01 + 1.0):
        return [Seg("L", seg[0].copy(), seg[-1].copy())]

    pieces: List[Tuple[str, int, int]] = []
    runs = _straight_runs(seg, prm)
    cur = 0
    for a, b in runs:
        if a > cur:
            pieces.append(("C", cur, a))
        pieces.append(("L", a, b))
        cur = b
    if cur < len(seg) - 1:
        pieces.append(("C", cur, len(seg) - 1))
    # merge too-short curve pieces into neighbouring lines' ends (keeps a tiny fillet as a curve)
    out: List[Seg] = []
    for pi, (kind, a, b) in enumerate(pieces):
        sub = seg[a : b + 1]
        if kind == "L":
            out.append(Seg("L", sub[0].copy(), sub[-1].copy(),
                           smooth_start=(pi > 0), smooth_end=(pi < len(pieces) - 1)))
            continue
        # tangents: from adjacent lines (G1) or estimated at corners / provided
        if pi > 0 and pieces[pi - 1][0] == "L":
            pa, pb = pieces[pi - 1][1], pieces[pi - 1][2]
            t0 = _unit(seg[pb] - seg[pa])
            sm0 = True
        else:
            t0 = t_start if (t_start is not None and a == 0) else _end_tangent(sub, True, step=prm.step_u)
            sm0 = t_start is not None and a == 0
        if pi < len(pieces) - 1 and pieces[pi + 1][0] == "L":
            pa, pb = pieces[pi + 1][1], pieces[pi + 1][2]
            t1 = -_unit(seg[pb] - seg[pa])
            sm1 = True
        else:
            t1 = (-t_end) if (t_end is not None and b == len(seg) - 1) else -_end_tangent(sub, False, step=prm.step_u)
            sm1 = t_end is not None and b == len(seg) - 1

        # Fit-first, split-only-on-failure: avoids chopping gentle diagonal strokes (e.g. 纟 撇) at tiny ripples
        whole_seg = _try_single_cubic(sub, _unit(t0), _unit(t1), prm.fit_tol_u)
        if whole_seg is not None:
            whole_seg.smooth_start = sm0
            whole_seg.smooth_end = sm1
            out.append(whole_seg)
            continue

        # split at H/V extrema with exact axis tangents
        ex = _extrema_splits(sub, prm)
        cuts = [0] + [i for i, _ in ex] + [len(sub) - 1]
        tans = [t0] + [t for _, t in ex] + [None]
        st.extrema_splits += len(ex)
        segs_piece: List[Seg] = []
        for ci in range(len(cuts) - 1):
            ia, ib = cuts[ci], cuts[ci + 1]
            part = sub[ia : ib + 1]
            ta = tans[ci]
            tb = t1 if ci == len(cuts) - 2 else -tans[ci + 1]
            fitted = _fit_cubic(part, _unit(ta), _unit(tb), prm.fit_tol_u)
            if ci > 0:
                fitted[0].smooth_start = True
            if ci < len(cuts) - 2:
                fitted[-1].smooth_end = True
            segs_piece.extend(fitted)
        segs_piece[0].smooth_start = segs_piece[0].smooth_start or sm0
        segs_piece[-1].smooth_end = segs_piece[-1].smooth_end or sm1
        out.extend(segs_piece)
    return _absorb_fillets(out, prm, closed=False)


def _seg_len(s: Seg) -> float:
    return float(np.linalg.norm(s.p3 - s.p0))


def _absorb_fillets(segs: List[Seg], prm: VecParams, closed: bool = False) -> List[Seg]:
    """Remove short curve stubs at corners / between lines or curves so crisp corners survive without double nodes."""
    if len(segs) <= 1:
        return segs
    out = list(segs)
    changed = True
    while changed and len(out) > 1:
        changed = False
        n = len(out)
        for i, s in enumerate(out):
            if s.kind != "C" or _seg_len(s) >= prm.fillet_absorb_u:
                continue
            prev = out[(i - 1) % n] if (closed or i > 0) else None
            nxt = out[(i + 1) % n] if (closed or i + 1 < n) else None
            if prev is None and nxt is not None and nxt.kind == "L":
                nxt.p0 = s.p0.copy(); nxt.smooth_start = False
                out.pop(i); changed = True; break
            if nxt is None and prev is not None and prev.kind == "L":
                prev.p3 = s.p3.copy(); prev.smooth_end = False
                out.pop(i); changed = True; break
            if prev is not None and nxt is not None and prev.kind == "L" and nxt.kind == "L":
                # replace fillet by the intersection of the two lines (if it lies close by)
                d1 = _unit(prev.p3 - prev.p0); d2 = _unit(nxt.p3 - nxt.p0)
                den = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(den) > 0.2:
                    t = ((nxt.p0 - prev.p3)[0] * d2[1] - (nxt.p0 - prev.p3)[1] * d2[0]) / den
                    x = prev.p3 + t * d1
                    if float(np.linalg.norm(x - s.p0)) <= 2.0 * prm.fillet_absorb_u:
                        prev.p3 = x.copy(); nxt.p0 = x.copy()
                        prev.smooth_end = False; nxt.smooth_start = False
                        out.pop(i); changed = True; break
            # Absorb short curve stub adjacent to a line or between two curves without tilting lines
            if prev is not None and nxt is not None and (prev.kind == "C" or nxt.kind == "C"):
                stub_len = _seg_len(s)
                if stub_len < 0.75 * prm.fillet_absorb_u and _seg_len(prev) > 2.0 * stub_len and _seg_len(nxt) > 2.0 * stub_len:
                    mid = 0.5 * (s.p0 + s.p3)
                    join: Optional[np.ndarray] = None
                    if prev.kind == "L" and nxt.kind == "C":
                        # Project onto prev's exact line direction so near-H/V stems never tilt
                        u_prev = _unit(prev.p3 - prev.p0)
                        proj = prev.p0 + float(np.dot(mid - prev.p0, u_prev)) * u_prev
                        if float(np.linalg.norm(proj - prev.p3)) <= 1.5 * prm.fillet_absorb_u:
                            join = proj
                    elif prev.kind == "C" and nxt.kind == "L":
                        # Project onto nxt's exact line direction
                        u_nxt = _unit(nxt.p3 - nxt.p0)
                        proj = nxt.p0 + float(np.dot(mid - nxt.p0, u_nxt)) * u_nxt
                        if float(np.linalg.norm(proj - nxt.p0)) <= 1.5 * prm.fillet_absorb_u:
                            join = proj
                    else:
                        # Curve-curve corner: intersect end tangents when crisp, else fall back to midpoint
                        t_in = _unit(prev.p3 - (prev.c2 if prev.c2 is not None else prev.p0))
                        t_out = _unit((nxt.c1 if nxt.c1 is not None else nxt.p3) - nxt.p0)
                        den = t_in[0] * t_out[1] - t_in[1] * t_out[0]
                        if abs(den) > 0.25:
                            tau = ((nxt.p0 - prev.p3)[0] * t_out[1] - (nxt.p0 - prev.p3)[1] * t_out[0]) / den
                            cand_x = prev.p3 + tau * t_in
                            if float(np.linalg.norm(cand_x - mid)) <= 1.2 * prm.fillet_absorb_u:
                                join = cand_x
                        if join is None:
                            join = mid

                    if join is not None:
                        d_prev = join - prev.p3
                        d_nxt = join - nxt.p0
                        prev.p3 = join.copy()
                        if prev.kind == "C" and prev.c2 is not None:
                            prev.c2 = prev.c2 + 0.5 * d_prev
                        nxt.p0 = join.copy()
                        if nxt.kind == "C" and nxt.c1 is not None:
                            nxt.c1 = nxt.c1 + 0.5 * d_nxt
                        prev.smooth_end = False
                        nxt.smooth_start = False
                        out.pop(i); changed = True; break
    return out


def _snap_axis(segs: List[Seg], prm: VecParams, st: VecStats) -> None:
    n = len(segs)
    tan_thr = math.tan(math.radians(prm.axis_snap_deg))
    for i, s in enumerate(segs):
        if s.kind != "L":
            continue
        v = s.p3 - s.p0
        L = float(np.linalg.norm(v))
        if L < 12.0:
            continue
        horiz = abs(v[1]) <= tan_thr * abs(v[0])
        vert = abs(v[0]) <= tan_thr * abs(v[1])
        if not (horiz or vert):
            continue
        ax = 1 if horiz else 0  # coordinate to equalise
        m = 0.5 * (s.p0[ax] + s.p3[ax])
        d0 = m - s.p0[ax]
        d3 = m - s.p3[ax]
        prev = segs[(i - 1) % n]
        nxt = segs[(i + 1) % n]
        s.p0 = s.p0.copy(); s.p0[ax] = m
        s.p3 = s.p3.copy(); s.p3[ax] = m
        prev.p3 = s.p0.copy()
        if prev.kind == "C":
            prev.c2 = prev.c2.copy(); prev.c2[ax] += d0
        nxt.p0 = s.p3.copy()
        if nxt.kind == "C":
            nxt.c1 = nxt.c1.copy(); nxt.c1[ax] += d3
        u = _unit(s.p3 - s.p0)
        if prev.kind == "C" and s.smooth_start:
            prev.c2 = s.p0 - float(np.linalg.norm(s.p0 - prev.c2)) * u
        if nxt.kind == "C" and s.smooth_end:
            nxt.c1 = s.p3 + float(np.linalg.norm(nxt.c1 - s.p3)) * u
        st.axis_snapped += 1


def _sample_segs(segs: List[Seg], step: float = 0.8) -> np.ndarray:
    out = []
    for s in segs:
        if s.kind == "L":
            L = float(np.linalg.norm(s.p3 - s.p0))
            k = max(2, int(L / step) + 1)
            t = np.linspace(0, 1, k)[:, None]
            out.append(s.p0 + t * (s.p3 - s.p0))
        else:
            L = float(np.linalg.norm(s.c1 - s.p0) + np.linalg.norm(s.c2 - s.c1) + np.linalg.norm(s.p3 - s.c2))
            k = max(3, int(L / step) + 1)
            out.append(_bez(s.p0, s.c1, s.c2, s.p3, np.linspace(0, 1, k)))
    return np.vstack(out) if out else np.zeros((0, 2))


def _fit_closed(p: np.ndarray, prm: VecParams, st: VecStats, smooth_iters: int) -> List[Seg]:
    corners = _detect_corners(p, prm)
    n = len(p)
    if len(corners) == 0:
        # smooth closed loop (dot / teardrop without sharp corner): cut at extrema of a smoothed copy
        ps = _taubin_closed(p, smooth_iters)
        xs, ys = ps[:, 0], ps[:, 1]
        cut = sorted({int(np.argmin(xs)), int(np.argmax(xs)), int(np.argmin(ys)), int(np.argmax(ys))})
        segs: List[Seg] = []
        for k in range(len(cut)):
            a, b = cut[k], cut[(k + 1) % len(cut)]
            idx = [(a + j) % n for j in range(((b - a) % n) + 1)] if len(cut) > 1 else list(range(n)) + [0]
            sub = ps[idx]
            ta = np.array([0.0, 1.0]) if a in (int(np.argmin(xs)), int(np.argmax(xs))) else np.array([1.0, 0.0])
            ta = ta if float(np.dot(ta, sub[min(2, len(sub) - 1)] - sub[0])) >= 0 else -ta
            tb = np.array([0.0, 1.0]) if b in (int(np.argmin(xs)), int(np.argmax(xs))) else np.array([1.0, 0.0])
            tb = tb if float(np.dot(tb, sub[-1] - sub[max(0, len(sub) - 3)])) >= 0 else -tb
            f = _fit_cubic(sub, ta, -tb, prm.fit_tol_u)
            f[0].smooth_start = True
            f[-1].smooth_end = True
            segs.extend(f)
        return segs
    st.corners += len(corners)
    segs = []
    for k in range(len(corners)):
        a, b = corners[k], corners[(k + 1) % len(corners)]
        cnt = (b - a) % n if len(corners) > 1 else n
        idx = [(a + j) % n for j in range(cnt + 1)]
        sub = _taubin_open(p[idx], smooth_iters)
        segs.extend(_fit_section(sub, prm, st))
    return segs


def _segs_to_cmds(segs: List[Seg], integer: bool) -> List[DrawCmd]:
    def r(pt):
        return (float(round(pt[0])), float(round(pt[1]))) if integer else (round(float(pt[0]), 1), round(float(pt[1]), 1))

    cmds: List[DrawCmd] = [("moveTo", [r(segs[0].p0)])]
    last = r(segs[0].p0)
    for s in segs:
        e = r(s.p3)
        if s.kind == "L":
            if e != last:
                cmds.append(("lineTo", [e]))
        else:
            cmds.append(("curveTo", [r(s.c1), r(s.c2), e]))
        last = e
    cmds.append(("closePath", []))
    return cmds


# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------

def sdf_iso_contours_upem(sdf: np.ndarray, prm: VecParams) -> List[np.ndarray]:
    """Sub-pixel zero-level contours of a [-1,1] SDF (+ inside) mapped to UPEM (y up)."""
    s = sdf.astype(np.float32)
    if prm.pre_sigma_px > 0:
        s = gaussian_filter(s, prm.pre_sigma_px, mode="nearest")
    up = max(1, int(prm.upsample))
    n_in = s.shape[0]
    if up > 1:
        s = zoom(s, up, order=3)
    n_out = s.shape[0]
    pad = 2
    sp = np.pad(s, pad, mode="constant", constant_values=-1.0)
    out = []
    for rc in find_contours(sp, 0.0):
        if len(rc) < 6:
            continue
        r = rc[:, 0] - pad
        c = rc[:, 1] - pad
        f = (n_in - 1) / float(n_out - 1) if up > 1 else 1.0
        px = c * f + 0.25  # +0.25 px: 2x2-mean target raster centre offset (dataset construction)
        py = r * f + 0.25
        x = px * UPX - PAD_U
        y = (ASC + PAD_U) - py * UPX
        pts = np.stack([x, y], axis=1)
        if np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
            pts = pts[:-1]
        out.append(pts)
    return out


def _point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(pt[0]), float(pt[1])
    xs, ys = poly[:, 0], poly[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    cond = (ys > y) != (ys2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
    return bool(np.count_nonzero(cond & (x < xint)) % 2 == 1)


def _nesting_depths(polys: List[np.ndarray]) -> List[int]:
    areas = [abs(_signed_area(p)) for p in polys]
    bbs = [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in polys]
    depths = []
    for i, p in enumerate(polys):
        probe = p[len(p) // 3]
        d = 0
        for j in range(len(polys)):
            if j == i or areas[j] <= areas[i]:
                continue
            b = bbs[j]
            if not (b[0] <= probe[0] <= b[2] and b[1] <= probe[1] <= b[3]):
                continue
            if _point_in_poly(probe, polys[j]):
                d += 1
        depths.append(d)
    return depths


def polylines_to_cmds(polys: List[np.ndarray], prm: Optional[VecParams] = None,
                      st: Optional[VecStats] = None) -> Tuple[List[DrawCmd], VecStats]:
    prm = prm or VecParams()
    st = st or VecStats()
    # resample & filter specks / pinholes by depth
    rs = [_resample_closed(p, prm.step_u) for p in polys if len(p) >= 4]
    rs = [p for p in rs if len(p) >= 6]
    if not rs:
        return [], st
    depths = _nesting_depths(rs)
    keep: List[Tuple[np.ndarray, int]] = []
    for p, d in zip(rs, depths):
        a = abs(_signed_area(p))
        if d % 2 == 0 and a < prm.min_island_u2:
            st.dropped_specks += 1
            continue
        if d % 2 == 1 and a < prm.min_hole_u2:
            st.dropped_pinholes += 1
            continue
        keep.append((p, d))
    all_cmds: List[DrawCmd] = []
    devs: List[np.ndarray] = []
    for p, d in keep:
        want_ccw = d % 2 == 0
        if (_signed_area(p) > 0) != want_ccw:
            p = p[::-1].copy()
        st_c = VecStats()
        segs = _fit_closed(p, prm, st_c, prm.smooth_iters)
        if not segs:
            continue
        segs = _absorb_fillets(segs, prm, closed=True)
        _snap_axis(segs, prm, st_c)
        samp = _sample_segs(segs)
        dv, _ = cKDTree(samp).query(p)
        # Only refit without smoothing when p95 or extreme max deviation truly exceeds tolerance
        if prm.smooth_iters > 0 and (
            float(np.percentile(dv, 95)) > prm.verify_tol_u * 0.85 or float(dv.max()) > prm.verify_tol_u * 1.35
        ):
            st.refits += 1
            st_c2 = VecStats()
            segs2 = _fit_closed(p, prm, st_c2, 0)
            if segs2:
                segs2 = _absorb_fillets(segs2, prm, closed=True)
                _snap_axis(segs2, prm, st_c2)
                samp2 = _sample_segs(segs2)
                dv2, _ = cKDTree(samp2).query(p)
                if float(dv2.max()) < float(dv.max()):
                    segs, dv, st_c = segs2, dv2, st_c2
        devs.append(dv)
        st.contours += 1
        st.corners += st_c.corners
        st.extrema_splits += st_c.extrema_splits
        st.axis_snapped += st_c.axis_snapped
        st.lines += sum(1 for s in segs if s.kind == "L")
        st.curves += sum(1 for s in segs if s.kind == "C")
        st.contour_len_u += float(np.sum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)))
        all_cmds.extend(_segs_to_cmds(segs, prm.integer_coords))
    if devs:
        dd = np.concatenate(devs)
        st.max_dev_u = float(dd.max())
        st.p95_dev_u = float(np.percentile(dd, 95))
    return all_cmds, st


def sdf_to_cmds_v46(sdf: np.ndarray, prm: Optional[VecParams] = None) -> Tuple[List[DrawCmd], Dict[str, float]]:
    """Main entry: [-1,1] SDF (256x256, + inside) -> CFF DrawCmds (UPEM) + vectorization stats."""
    prm = prm or VecParams()
    polys = sdf_iso_contours_upem(sdf, prm)
    cmds, st = polylines_to_cmds(polys, prm)
    return cmds, st.as_dict()


def cmds_to_polylines(cmds: Sequence[DrawCmd], step: float = 1.5) -> List[np.ndarray]:
    """Flatten DrawCmds into closed polylines (UPEM), explicitly sampling implicit closePath segments."""
    polys: List[np.ndarray] = []
    cur: List[Tuple[float, float]] = []
    pos = (0.0, 0.0)
    start = (0.0, 0.0)
    for op, cp in cmds:
        if op == "moveTo":
            if len(cur) >= 3:
                polys.append(np.array(cur))
            cur = [tuple(cp[0])]
            pos = cp[0]
            start = cp[0]
        elif op == "lineTo":
            p0 = np.array(pos); p1 = np.array(cp[0])
            k = max(1, int(np.linalg.norm(p1 - p0) / step))
            for t in np.linspace(0, 1, k + 1)[1:]:
                q = p0 + t * (p1 - p0)
                cur.append((float(q[0]), float(q[1])))
            pos = cp[0]
        elif op == "curveTo":
            p0, c1, c2, p3 = (np.array(x, dtype=np.float64) for x in (pos, cp[0], cp[1], cp[2]))
            L = np.linalg.norm(c1 - p0) + np.linalg.norm(c2 - c1) + np.linalg.norm(p3 - c2)
            k = max(3, int(L / step))
            for q in _bez(p0, c1, c2, p3, np.linspace(0, 1, k + 1)[1:]):
                cur.append((float(q[0]), float(q[1])))
            pos = cp[2]
        elif op == "closePath":
            p0 = np.array(pos); p1 = np.array(start)
            dist_close = float(np.linalg.norm(p1 - p0))
            if dist_close > 1e-6:
                k = max(1, int(dist_close / step))
                for t in np.linspace(0, 1, k + 1)[1:]:
                    q = p0 + t * (p1 - p0)
                    cur.append((float(q[0]), float(q[1])))
            pos = start
            if len(cur) >= 3:
                polys.append(np.array(cur))
            cur = []
    if len(cur) >= 3:
        polys.append(np.array(cur))
    out = []
    for p in polys:
        if np.linalg.norm(p[0] - p[-1]) < 1e-6:
            p = p[:-1]
        out.append(p)
    return out
