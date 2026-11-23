"""
v3.0 Structure-First Acceptance Gate (`StructuralGateV3`, Plan 3.0 §2.1).

Replaces v2.0's gameable whole-glyph IoU + synthetic straightness metric with:
  1. BBox-aligned medial skeleton directional diff-zone test (`D_sc` coverage & `D_jp` non-retention).
  2. Whole-glyph bidirectional skeleton Chamfer + tail-miss caps + 2x2 quadrant balance.
  3. Hole/topology count check & per-replaced-component cropped skeleton check.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.v2.hb_outlines import BBox, DrawCmd
from src.v3.hb_light import (
    CleanSkeleton,
    SansLightExtractor,
    align_cmds_to_bbox,
    extract_clean_skeleton,
    upem_box_to_mask,
)


class StructuralGateV3:
    """
    Structure-first glyph verification gate with zero reliance on Noto Serif or straight-line snapping.
    """

    def __init__(
        self,
        sn_jp: SansLightExtractor,
        sn_sc: SansLightExtractor,
        canvas_size: int = 160,
        padding: float = 80.0,
        tau_diff_upem: float = 22.0,
    ) -> None:
        self.sn_jp = sn_jp
        self.sn_sc = sn_sc
        self.canvas_size = canvas_size
        self.padding = padding
        self.tau_diff_upem = tau_diff_upem
        self._ref_cache: Dict[str, Tuple[List[DrawCmd], CleanSkeleton, Optional[CleanSkeleton]]] = {}

    def get_ref_bundle(self, ch: str) -> Optional[Tuple[List[DrawCmd], CleanSkeleton, Optional[CleanSkeleton]]]:
        """Return `(cs_cmds, skel_sc, skel_jp_aligned)` cached per character."""
        if ch in self._ref_cache:
            return self._ref_cache[ch]
        cs = self.sn_sc.get_cmds(ch, unioned=False)
        if not cs:
            return None
        skel_sc = extract_clean_skeleton(cs, self.canvas_size, self.padding)
        cj = self.sn_jp.get_cmds(ch, unioned=False)
        skel_jp: Optional[CleanSkeleton] = None
        if cj:
            cj_aligned = align_cmds_to_bbox(cj, cs)
            skel_jp = extract_clean_skeleton(cj_aligned, self.canvas_size, self.padding)
        bundle = (cs, skel_sc, skel_jp)
        self._ref_cache[ch] = bundle
        return bundle

    @staticmethod
    def _quadrant_fractions(skel: np.ndarray) -> np.ndarray:
        """Compute fraction of skeleton pixels falling in TL, TR, BL, BR quadrants of ink bbox."""
        ys, xs = np.where(skel)
        if len(ys) < 4:
            return np.zeros(4, dtype=np.float32)
        ym = 0.5 * (int(ys.min()) + int(ys.max()))
        xm = 0.5 * (int(xs.min()) + int(xs.max()))
        tot = float(len(ys))
        return np.array(
            [
                ((ys <= ym) & (xs <= xm)).sum() / tot,
                ((ys <= ym) & (xs > xm)).sum() / tot,
                ((ys > ym) & (xs <= xm)).sum() / tot,
                ((ys > ym) & (xs > xm)).sum() / tot,
            ],
            dtype=np.float32,
        )

    def analyze_jp_sc_diff(
        self,
        ch: str,
        exempt_boxes: Optional[Sequence[BBox]] = None,
    ) -> dict:
        """Compute `D_sc` and `D_jp` masks + diff-zone statistics for `ch`."""
        bundle = self.get_ref_bundle(ch)
        if bundle is None:
            return {"has_ref": False, "dsc_px": 0, "djp_px": 0, "diff_ratio": 0.0}
        _, skel_sc, skel_jp = bundle
        if skel_jp is None or not skel_sc.skel.any():
            return {"has_ref": False, "dsc_px": 0, "djp_px": 0, "diff_ratio": 0.0}

        tau = self.tau_diff_upem
        d_sc = skel_sc.skel & (skel_jp.dt_upem > tau)
        d_jp = skel_jp.skel & (skel_sc.dt_upem > tau)

        if exempt_boxes:
            ex_mask = np.zeros_like(d_sc, dtype=bool)
            for box in exempt_boxes:
                ex_mask |= upem_box_to_mask(box, self.canvas_size, self.padding, margin_u=22.0)
            d_sc = d_sc & (~ex_mask)
            d_jp = d_jp & (~ex_mask)

        total_len = max(int(skel_sc.skel.sum()), 1)
        return {
            "has_ref": True,
            "skel_sc": skel_sc,
            "skel_jp": skel_jp,
            "d_sc": d_sc,
            "d_jp": d_jp,
            "dsc_px": int(d_sc.sum()),
            "djp_px": int(d_jp.sum()),
            "diff_ratio": float((d_sc.sum() + d_jp.sum()) / float(total_len)),
        }

    def check(
        self,
        cand_cmds: List[DrawCmd],
        ch: str,
        exempt_boxes: Optional[Sequence[BBox]] = None,
        replaced_boxes: Optional[Sequence[BBox]] = None,
        check_jp_diff: bool = True,
    ) -> Tuple[bool, dict]:
        """
        Verify candidate CFF outlines against bbox-aligned Noto Sans Light `zh-Hans` & `ja`.
        Never uses Noto Serif contours and never credits straight-line snapping.
        """
        if not cand_cmds:
            return False, {"passed": False, "reason": "empty_cmds"}

        bundle = self.get_ref_bundle(ch)
        if bundle is None:
            return False, {"passed": False, "reason": "no_sc_ref"}
        cs_cmds, skel_sc, _ = bundle
        if not skel_sc.skel.any():
            return False, {"passed": False, "reason": "empty_sc_ref"}

        cand_aligned = align_cmds_to_bbox(cand_cmds, cs_cmds)
        skel_c = extract_clean_skeleton(cand_aligned, self.canvas_size, self.padding)
        if not skel_c.skel.any():
            return False, {"passed": False, "reason": "empty_cand_skel"}

        # 1. Whole-glyph bidirectional skeleton Chamfer, p95 & tail miss ratios
        d_c2sc = skel_sc.dt_upem[skel_c.skel]
        d_sc2c = skel_c.dt_upem[skel_sc.skel]
        chamfer = 0.5 * (float(d_c2sc.mean()) + float(d_sc2c.mean()))
        p95_ch = float(np.percentile(np.concatenate([d_c2sc, d_sc2c]), 95))
        miss_sc = float((d_sc2c > 50.0).mean())
        miss_cand = float((d_c2sc > 54.0).mean())

        # Quadrant, 3x3 cosine & 4-side outer strip Chamfer checks (skip on tiny single-stroke thin shapes like `一`)
        ys_s, xs_s = np.where(skel_sc.skel)
        ys_c, xs_c = np.where(skel_c.skel)
        non_thin = len(ys_s) >= 8 and (int(ys_s.max()) - int(ys_s.min())) >= 22 and (int(xs_s.max()) - int(xs_s.min())) >= 22
        if non_thin and len(ys_c) >= 4:
            q_c = self._quadrant_fractions(skel_c.skel)
            q_s = self._quadrant_fractions(skel_sc.skel)
            quad_diff = float(np.max(np.abs(q_c - q_s)))

            y0 = min(int(ys_s.min()), int(ys_c.min()))
            y1 = max(int(ys_s.max()), int(ys_c.max())) + 1
            x0 = min(int(xs_s.min()), int(xs_c.min()))
            x1 = max(int(xs_s.max()), int(xs_c.max())) + 1
            gs = np.array(
                [
                    skel_sc.skel[
                        y0 + int(i * (y1 - y0) / 3) : y0 + int((i + 1) * (y1 - y0) / 3),
                        x0 + int(j * (x1 - x0) / 3) : x0 + int((j + 1) * (x1 - x0) / 3),
                    ].sum()
                    for i in range(3)
                    for j in range(3)
                ],
                dtype=np.float64,
            )
            gc = np.array(
                [
                    skel_c.skel[
                        y0 + int(i * (y1 - y0) / 3) : y0 + int((i + 1) * (y1 - y0) / 3),
                        x0 + int(j * (x1 - x0) / 3) : x0 + int((j + 1) * (x1 - x0) / 3),
                    ].sum()
                    for i in range(3)
                    for j in range(3)
                ],
                dtype=np.float64,
            )
            cos3x3 = float(np.dot(gs, gc) / max(float(np.linalg.norm(gs) * np.linalg.norm(gc)), 1e-6))

            sw = max(1, int(round(0.38 * (x1 - x0))))
            sh = max(1, int(round(0.38 * (y1 - y0))))
            strip_slices = (
                (slice(y0, y1), slice(x0, x0 + sw)),
                (slice(y0, y1), slice(x1 - sw, x1)),
                (slice(y0, y0 + sh), slice(x0, x1)),
                (slice(y1 - sh, y1), slice(x0, x1)),
            )
            strip_chs: List[float] = []
            for sy, sx in strip_slices:
                sc_strip = skel_sc.skel[sy, sx]
                if int(sc_strip.sum()) < 10:
                    continue
                ca_strip = skel_c.skel[sy, sx]
                if int(ca_strip.sum()) < 3:
                    strip_chs.append(999.0)
                else:
                    strip_ch = 0.5 * (
                        float(skel_sc.dt_upem[sy, sx][ca_strip].mean())
                        + float(skel_c.dt_upem[sy, sx][sc_strip].mean())
                    )
                    strip_chs.append(strip_ch)
            strip_max_ch = max(strip_chs) if strip_chs else chamfer
        else:
            quad_diff = 0.0
            cos3x3 = 1.0
            strip_max_ch = chamfer

        # Topology checks: holes, islands, and (when chamfer > 20.5) endpoints & junctions (Plan 3.0 §2.1 #3)
        hole_low = max(4, int(round(0.55 * max(skel_sc.hole_count, 1))))
        hole_high = max(7, int(round(1.30 * max(skel_sc.hole_count, 1)))) if chamfer <= 21.8 else hole_low
        hole_ok = (skel_c.hole_count >= skel_sc.hole_count - hole_low) and (skel_c.hole_count <= skel_sc.hole_count + hole_high)
        island_ok = abs(skel_c.island_count - skel_sc.island_count) <= max(3, skel_sc.island_count)
        if chamfer > 20.5:
            ep_tol = max(10, int(round(1.00 * max(skel_sc.endpoint_count, 1))))
            jc_tol = max(10, int(round(1.30 * max(skel_sc.junction_count, 1))))
            ep_ok = abs(skel_c.endpoint_count - skel_sc.endpoint_count) <= ep_tol
            jc_ok = abs(skel_c.junction_count - skel_sc.junction_count) <= jc_tol
        else:
            ep_ok = True
            jc_ok = True

        # Allow SkipPro calligraphic bow (e.g. 子 rightward curve-hook) when overall Chamfer & miss ratios are low
        p95_limit = 92.0 if (chamfer <= 28.0 and miss_sc <= 0.15 and miss_cand <= 0.15) else 72.3

        shape_ok = (
            (chamfer <= 37.5)
            and (p95_ch <= p95_limit)
            and (strip_max_ch <= 48.0)
            and (cos3x3 >= 0.80)
            and (miss_sc <= 0.245)
            and (miss_cand <= 0.235)
            and (quad_diff <= 0.20)
            and hole_ok
            and island_ok
            and ep_ok
            and jc_ok
        )

        # 2. Directional JP-vs-SC diff-zone discrimination (§2.1.1; disabled for Missing glyphs without JP counterpart)
        diff_info = self.analyze_jp_sc_diff(ch, exempt_boxes=exempt_boxes) if check_jp_diff else {"has_ref": False, "dsc_px": 0, "djp_px": 0}
        cov_dsc = 1.0
        res_djp = 0.0
        d2sc = 0.0
        d2jp = 60.0
        diff_ok = True
        if check_jp_diff and diff_info["has_ref"] and (diff_info["dsc_px"] >= 8 or diff_info["djp_px"] >= 8):
            d_sc = diff_info["d_sc"]
            d_jp = diff_info["d_jp"]
            if diff_info["dsc_px"] >= 8:
                d2sc = float(skel_c.dt_upem[d_sc].mean())
                cov_dsc = float((skel_c.dt_upem[d_sc] <= 26.0).mean())
            if diff_info["djp_px"] >= 8:
                d2jp = float(skel_c.dt_upem[d_jp].mean())
                res_djp = float((skel_c.dt_upem[d_jp] <= 16.0).mean())
                if res_djp > 0.35:
                    diff_ok = False
            if diff_info["dsc_px"] >= 8:
                clean_sc_shift = (diff_info["djp_px"] >= 8) and (res_djp <= 0.08) and (d2sc <= 28.0) and (d2jp >= d2sc + 14.0)
                if cov_dsc < 0.62 and not clean_sc_shift:
                    diff_ok = False
            if diff_info["dsc_px"] >= 6 and diff_info["djp_px"] >= 6:
                if d2sc + 2.5 >= d2jp:
                    diff_ok = False

        # 3. Per-replaced-component Chamfer verification (§2.1.2)
        comp_chamfers: List[float] = []
        comp_ok = True
        if replaced_boxes:
            for box in replaced_boxes:
                m_box = upem_box_to_mask(box, self.canvas_size, self.padding, margin_u=16.0)
                sc_crop = skel_sc.skel & m_box
                ca_crop = skel_c.skel & m_box
                if int(sc_crop.sum()) < 6:
                    continue
                if int(ca_crop.sum()) < 3:
                    comp_ok = False
                    comp_chamfers.append(999.0)
                    continue
                c_ch = 0.5 * (float(skel_sc.dt_upem[ca_crop].mean()) + float(skel_c.dt_upem[sc_crop].mean()))
                comp_chamfers.append(round(c_ch, 2))
                if c_ch > 32.0:
                    comp_ok = False

        passed = bool(shape_ok and diff_ok and comp_ok)
        fail_reasons: List[str] = []
        if chamfer > 37.5:
            fail_reasons.append(f"chamfer={chamfer:.1f}")
        if p95_ch > p95_limit:
            fail_reasons.append(f"p95={p95_ch:.1f}")
        if strip_max_ch > 48.0:
            fail_reasons.append(f"strip_max_ch={strip_max_ch:.1f}")
        if cos3x3 < 0.80:
            fail_reasons.append(f"cos3x3={cos3x3:.3f}")
        if miss_sc > 0.245:
            fail_reasons.append(f"miss_sc={miss_sc:.2f}")
        if miss_cand > 0.235:
            fail_reasons.append(f"miss_cand={miss_cand:.2f}")
        if quad_diff > 0.20:
            fail_reasons.append(f"quad_diff={quad_diff:.2f}")
        if not hole_ok:
            fail_reasons.append(f"holes={skel_c.hole_count}!={skel_sc.hole_count}")
        if not island_ok:
            fail_reasons.append(f"islands={skel_c.island_count}!={skel_sc.island_count}")
        if not ep_ok:
            fail_reasons.append(f"ep={skel_c.endpoint_count}!={skel_sc.endpoint_count}")
        if not jc_ok:
            fail_reasons.append(f"jc={skel_c.junction_count}!={skel_sc.junction_count}")
        if not diff_ok:
            fail_reasons.append(f"diff_zone(cov={cov_dsc:.2f},res={res_djp:.2f},d2sc={d2sc:.1f},d2jp={d2jp:.1f})")
        if not comp_ok:
            fail_reasons.append(f"comp_chamfer={comp_chamfers}")

        return passed, {
            "passed": passed,
            "chamfer": round(chamfer, 2),
            "p95": round(p95_ch, 2),
            "strip_max_ch": round(strip_max_ch, 2),
            "miss_sc": round(miss_sc, 4),
            "miss_cand": round(miss_cand, 4),
            "quad_diff": round(quad_diff, 4),
            "holes_cand": skel_c.hole_count,
            "holes_sc": skel_sc.hole_count,
            "ep_cand": skel_c.endpoint_count,
            "ep_sc": skel_sc.endpoint_count,
            "jc_cand": skel_c.junction_count,
            "jc_sc": skel_sc.junction_count,
            "cov_dsc": round(cov_dsc, 4),
            "res_djp": round(res_djp, 4),
            "d2sc": round(d2sc, 2),
            "d2jp": round(d2jp, 2),
            "comp_chamfers": comp_chamfers,
            "reasons": fail_reasons,
        }
