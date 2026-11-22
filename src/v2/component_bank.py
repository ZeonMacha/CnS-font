"""
Tier 0 Delta Surgery & Tier 1 Positional Donor Component Composer (§3.1 & §3.2).

Key Engineering Guarantees:
  1. Uses cjkvi-ids [G] decompositions ONLY on valid CJK codepoints (excludes
     circled-number placeholders U+2460..U+2473 and excludes `⿻` crossings).
  2. Groups contour islands (outer CCW + enclosed CW holes via Shapely) so dots
     with overlapping X-ranges (e.g. `糸` right dot vs `工` in `紅`) never mis-split.
  3. Filters out donor characters whose donated component has JP != SC differences
     unless explicitly replaced by a curated Master Bank component.
  4. Preserves stroke weight on scaled donor components via axis-wise stroke-width
     compensation (`fit_component_with_stroke_compensation`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from fontTools.misc.transform import Transform
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.transformPen import TransformPen
from shapely.geometry import Polygon

from src.v2.curve_optimizer import (
    _flatten_contour_poly,
    boolean_union_cmds,
    ensure_cff_winding,
    optimize_synthesized_contours,
)
from src.v2.hb_outlines import (
    MAINLAND_VARIANT_MARKERS,
    BBox,
    DrawCmd,
    HBOutlineExtractor,
    SkipProOutlineExtractor,
    cmds_iou,
    cmds_to_bbox,
    split_cmds_into_contours,
)

IDC_CHARS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")
IDC_ARITY = {c: (3 if c in "⿲⿳" else 2) for c in IDC_CHARS}


def is_valid_cjk_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    cp = ord(ch)
    return (0x2E80 <= cp <= 0x9FFF) or (0xF900 <= cp <= 0xFAFF) or (cp >= 0x20000)


def parse_cjkvi_ids(ids_path: str | Path) -> Dict[str, str]:
    """Parse cjkvi-ids selecting Mainland [G] decomposition whenever multiple variants exist."""
    ids_map: Dict[str, str] = {}
    for line in Path(ids_path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "\t" not in line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ch, seqs = parts[1], parts[2:]
        chosen = None
        for s in seqs:
            m = re.search(r"\[([A-Z]+)\]", s)
            if not m or "G" in m.group(1):
                chosen = re.sub(r"\[.*?\]", "", s)
                break
        if chosen is None:
            chosen = re.sub(r"\[.*?\]", "", seqs[0])
        ids_map[ch] = chosen
    return ids_map


def top_split(seq: str) -> Tuple[Optional[str], List[str]]:
    """Split top-level IDS operator into its sub-sequences."""
    if not seq or len(seq) == 1 or seq[0] not in IDC_CHARS:
        return None, [seq]
    idc = seq[0]
    i = 1
    comps: List[str] = []
    for _ in range(IDC_ARITY[idc]):
        if i >= len(seq):
            return None, [seq]
        st, need = i, 1
        while need and i < len(seq):
            if seq[i] in IDC_CHARS:
                need += IDC_ARITY[seq[i]]
            need -= 1
            i += 1
        if need != 0:
            return None, [seq]
        comps.append(seq[st:i])
    return idc, comps


def seq_has_invalid_entity(seq: str) -> bool:
    """Reject IDS strings containing circled placeholders ①..⑳ or unencoded ASCII markers."""
    for ch in seq:
        if ch not in IDC_CHARS and not is_valid_cjk_char(ch):
            return True
    return False


@dataclass
class ContourIsland:
    """Connected ink component: 1 outer CCW contour + all enclosed CW hole contours."""

    cmds: List[DrawCmd]
    bbox: BBox
    poly: Polygon
    cx: float
    cy: float
    area: float


def cmds_to_islands(cmds: List[DrawCmd]) -> List[ContourIsland]:
    """Group CFF contours into topological islands (outer + enclosed holes)."""
    contours = ensure_cff_winding(split_cmds_into_contours(cmds))
    if not contours:
        return []

    polys: List[Polygon] = []
    areas: List[float] = []
    for c in contours:
        pts, a = _flatten_contour_poly(c, step_upem=16.0)
        poly = Polygon(pts) if len(pts) >= 3 else Polygon()
        if not poly.is_valid:
            poly = poly.buffer(0)
        polys.append(poly)
        areas.append(a)

    # Outer contours have area > 0; holes (area < 0) attach to smallest enclosing outer contour
    outer_idx = [i for i, a in enumerate(areas) if a > 0]
    hole_idx = [i for i, a in enumerate(areas) if a <= 0]
    groups: Dict[int, List[int]] = {oi: [oi] for oi in outer_idx}

    for hi in hole_idx:
        rep = polys[hi].representative_point() if not polys[hi].is_empty else None
        best_oi = None
        best_area = float("inf")
        if rep is not None:
            for oi in outer_idx:
                if abs(areas[oi]) > abs(areas[hi]) and abs(areas[oi]) < best_area:
                    if polys[oi].contains(rep):
                        best_area = abs(areas[oi])
                        best_oi = oi
        if best_oi is not None:
            groups[best_oi].append(hi)
        elif outer_idx:
            # Fallback to nearest outer contour
            groups[outer_idx[0]].append(hi)

    islands: List[ContourIsland] = []
    for oi, idxs in groups.items():
        merged_cmds: List[DrawCmd] = []
        merged_poly = polys[oi]
        for idx in idxs:
            merged_cmds.extend(contours[idx])
            if idx != oi and not polys[idx].is_empty:
                try:
                    merged_poly = merged_poly.difference(polys[idx])
                except Exception:
                    pass
        bb = cmds_to_bbox(merged_cmds) or (0.0, 0.0, 100.0, 100.0)
        cx = 0.5 * (bb[0] + bb[2])
        cy = 0.5 * (bb[1] + bb[3])
        islands.append(
            ContourIsland(
                cmds=merged_cmds,
                bbox=bb,
                poly=merged_poly,
                cx=cx,
                cy=cy,
                area=float(merged_poly.area),
            )
        )
    return islands


def transform_cmds(cmds: List[DrawCmd], tx: Transform) -> List[DrawCmd]:
    rec = RecordingPen()
    tpen = TransformPen(rec, tx)
    for op, pts in cmds:
        getattr(tpen, op)(*pts)
    return [(op, [(round(float(x), 1), round(float(y), 1)) for x, y in pts]) for op, pts in rec.value]


def _offset_polygon_commands(cmds: List[DrawCmd], dx_offset: float, dy_offset: float) -> List[DrawCmd]:
    """
    Approximate axis-dependent stroke-width restoration offset via outward normal shift
    on dense vertices followed by clean PathOps union + curve cleanup (§3.2 #4).
    """
    if abs(dx_offset) < 1.5 and abs(dy_offset) < 1.5:
        return cmds

    contours = ensure_cff_winding(split_cmds_into_contours(cmds))
    shifted_contours: List[List[DrawCmd]] = []
    for c_cmds in contours:
        pts, _ = _flatten_contour_poly(c_cmds, step_upem=10.0)
        if len(pts) < 3:
            continue
        n = len(pts)
        out_pts = np.zeros_like(pts)
        for i in range(n):
            v1 = pts[i] - pts[(i - 1) % n]
            v2 = pts[(i + 1) % n] - pts[i]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            t = (v1 / max(n1, 1e-4)) + (v2 / max(n2, 1e-4))
            nt = float(np.linalg.norm(t))
            if nt < 1e-4:
                t = v1 / max(n1, 1e-4)
            else:
                t /= nt
            # Since CFF outer is CCW (+area) & hole is CW (-area), RIGHT normal (t_y, -t_x)
            # always points OUTWARD from ink! Expanding outward thickens both stems and shrinks holes!
            nx, ny = t[1], -t[0]
            out_pts[i, 0] = pts[i, 0] + nx * dx_offset
            out_pts[i, 1] = pts[i, 1] + ny * dy_offset

        c_new: List[DrawCmd] = [("moveTo", [(float(out_pts[0, 0]), float(out_pts[0, 1]))])]
        for k in range(1, n):
            c_new.append(("lineTo", [(float(out_pts[k, 0]), float(out_pts[k, 1]))]))
        c_new.append(("closePath", []))
        shifted_contours.append(c_new)

    unioned = boolean_union_cmds(shifted_contours)
    return optimize_synthesized_contours(unioned, stem_concave_sagitta=0.85)


def fit_component_with_stroke_compensation(
    cmds: List[DrawCmd],
    src_bbox: BBox,
    dst_bbox: BBox,
) -> List[DrawCmd]:
    """
    Transform a donor component from `src_bbox` to `dst_bbox`.
      - When aspect ratio scaling is mild (0.86..1.15), apply EXACT pure affine
        `TransformPen` so 100% of original Fontworks CFF control points stay untouched!
      - When compressed more strongly (< 0.86), apply axis-wise stroke-width restoration.
    """
    sx0, sy0, sx1, sy1 = src_bbox
    dx0, dy0, dx1, dy1 = dst_bbox
    sw = max(sx1 - sx0, 1.0)
    sh = max(sy1 - sy0, 1.0)
    dw = max(dx1 - dx0, 1.0)
    dh = max(dy1 - dy0, 1.0)

    sx = dw / sw
    sy = dh / sh
    tx_val = dx0 - sx0 * sx
    ty_val = dy0 - sy0 * sy

    scaled = transform_cmds(cmds, Transform(sx, 0.0, 0.0, sy, tx_val, ty_val))
    if 0.62 <= sx <= 1.35 and 0.62 <= sy <= 1.35:
        return scaled

    # Compensate stem thinning along compressed axes (typical SkipPro-B vertical=76u, horiz=56u)
    dx_comp = max(0.0, (1.0 - sx) * 38.0 * 0.65) - max(0.0, (sx - 1.15) * 28.0 * 0.45)
    dy_comp = max(0.0, (1.0 - sy) * 28.0 * 0.65) - max(0.0, (sy - 1.15) * 22.0 * 0.45)
    if abs(dx_comp) >= 2.5 or abs(dy_comp) >= 2.5:
        compensated = _offset_polygon_commands(scaled, dx_comp, dy_comp)
        # Re-snap final outer bounds precisely into dst_bbox
        new_bb = cmds_to_bbox(compensated)
        if new_bb is not None:
            nx0, ny0, nx1, ny1 = new_bb
            fx = dw / max(nx1 - nx0, 1.0)
            fy = dh / max(ny1 - ny0, 1.0)
            return transform_cmds(compensated, Transform(fx, 0, 0, fy, dx0 - nx0 * fx, dy0 - ny0 * fy))
    return scaled


def _assign_islands_to_slots(
    islands: List[ContourIsland],
    idc: str,
    n_slots: int,
    slot_boxes_norm: Optional[List[BBox]] = None,
) -> List[List[ContourIsland]]:
    """Assign glyph ContourIslands to IDS slots using spatial geometry / Noto reference boxes."""
    buckets: List[List[ContourIsland]] = [[] for _ in range(n_slots)]
    if not islands:
        return buckets

    if slot_boxes_norm is not None and len(slot_boxes_norm) == n_slots:
        for isl in islands:
            best_s = 0
            best_score = -1e9
            ix0, iy0, ix1, iy1 = isl.bbox
            for s, (bx0, by0, bx1, by1) in enumerate(slot_boxes_norm):
                # Intersection over island area minus center distance
                iw = max(0.0, min(ix1, bx1) - max(ix0, bx0))
                ih = max(0.0, min(iy1, by1) - max(iy0, by0))
                ov = (iw * ih) / max((ix1 - ix0) * (iy1 - iy0), 1.0)
                dcx = abs(isl.cx - 0.5 * (bx0 + bx1)) / 1000.0
                dcy = abs(isl.cy - 0.5 * (by0 + by1)) / 1000.0
                sc = ov * 2.0 - (dcx + dcy)
                if sc > best_score:
                    best_score = sc
                    best_s = s
            buckets[best_s].append(isl)
        return buckets

    # Heuristic fallback by IDC direction
    xs0 = [isl.bbox[0] for isl in islands]
    xs1 = [isl.bbox[2] for isl in islands]
    ys0 = [isl.bbox[1] for isl in islands]
    ys1 = [isl.bbox[3] for isl in islands]
    gx0, gy0, gx1, gy1 = min(xs0), min(ys0), max(xs1), max(ys1)
    gw = max(gx1 - gx0, 1.0)
    gh = max(gy1 - gy0, 1.0)

    for isl in islands:
        rx = (isl.cx - gx0) / gw
        ry = (isl.cy - gy0) / gh
        if idc == "⿰":
            slot = 0 if rx < 0.44 else 1
        elif idc == "⿲":
            slot = 0 if rx < 0.34 else (1 if rx < 0.67 else 2)
        elif idc == "⿱":
            slot = 0 if ry > 0.56 else 1
        elif idc == "⿳":
            slot = 0 if ry > 0.66 else (1 if ry > 0.33 else 2)
        elif idc in ("⿺", "⿲"):
            # Surrounded bottom-left (e.g. 辶, 廴): leftmost or bottommost edge islands -> slot 0
            if isl.bbox[0] < gx0 + 0.20 * gw and isl.bbox[1] < gy0 + 0.32 * gh:
                slot = 0
            elif isl.bbox[2] < gx0 + 0.36 * gw or isl.bbox[3] < gy0 + 0.25 * gh:
                slot = 0
            else:
                slot = 1
        elif idc == "⿸":
            slot = 0 if (isl.bbox[0] < gx0 + 0.26 * gw and isl.bbox[3] > gy1 - 0.35 * gh) or isl.bbox[3] > gy1 - 0.16 * gh else 1
        elif idc == "⿹":
            slot = 0 if (isl.bbox[2] > gx1 - 0.26 * gw and isl.bbox[3] > gy1 - 0.35 * gh) or isl.bbox[3] > gy1 - 0.16 * gh else 1
        elif idc == "⿴":
            touches_border = (isl.bbox[0] < gx0 + 50 and isl.bbox[2] > gx1 - 50 and isl.bbox[3] > gy1 - 50)
            slot = 0 if touches_border else 1
        elif idc == "⿵":
            touches_top = isl.bbox[3] > gy1 - 80 and (isl.bbox[2] - isl.bbox[0]) > 0.65 * gw
            slot = 0 if touches_top else 1
        else:
            slot = 0 if rx < 0.5 else 1
        buckets[min(slot, n_slots - 1)].append(isl)
    return buckets


@dataclass
class ComponentEntry:
    cmds: List[DrawCmd]
    bbox: BBox
    donor_char: str
    is_master: bool = False


class PositionalDonorBank:
    """
    Indexes native CFF components from:
      1) `manual_review/CnS-Manual-Master-Bank.otf` (curated Mainland masters, highest priority!)
      2) `FOT-SkipPro-B.OTF` positional donors filtered to prevent JP-variant leakage.
    """

    def __init__(
        self,
        skip_ext: SkipProOutlineExtractor,
        ids_map: Dict[str, str],
        jp_diff_chars: Set[str],
        sans_sc: Optional[HBOutlineExtractor] = None,
    ) -> None:
        self.skip_ext = skip_ext
        self.ids_map = ids_map
        self.jp_diff_chars = jp_diff_chars
        self.sans_sc = sans_sc
        self.positional: Dict[Tuple[str, int, str], List[ComponentEntry]] = defaultdict(list)
        self.master_full: Dict[str, ComponentEntry] = {}
        self.master_slot: Dict[Tuple[str, int, str], ComponentEntry] = {}
        self._verified_donors: Dict[Tuple[str, int, str], List[ComponentEntry]] = {}

    def build_from_skippro(self, candidate_chars: Optional[List[str]] = None) -> int:
        """Extract clean positional CFF components from SkipPro-B."""
        pool = candidate_chars if candidate_chars is not None else [
            chr(cp) for cp in self.skip_ext.cmap if is_valid_cjk_char(chr(cp))
        ]
        added = 0
        for ch in pool:
            seq = self.ids_map.get(ch)
            if not seq or seq_has_invalid_entity(seq):
                continue
            idc, comps = top_split(seq)
            if idc not in ("⿰", "⿱", "⿲", "⿳") or comps == [ch]:
                continue

            cmds = self.skip_ext.get_cmds(ch)
            if not cmds:
                continue
            islands = cmds_to_islands(cmds)
            if len(islands) < len(comps):
                continue

            buckets = _assign_islands_to_slots(islands, idc, len(comps))
            if any(not b for b in buckets):
                continue

            slot_cmds: List[List[DrawCmd]] = []
            slot_bbs: List[BBox] = []
            valid_split = True
            for b in buckets:
                c_cmds = [cmd for isl in b for cmd in isl.cmds]
                bb = cmds_to_bbox(c_cmds)
                if bb is None:
                    valid_split = False
                    break
                slot_cmds.append(c_cmds)
                slot_bbs.append(bb)
            if not valid_split:
                continue

            glyph_bb = cmds_to_bbox(cmds)
            if glyph_bb is None:
                continue
            gw = max(glyph_bb[2] - glyph_bb[0], 1.0)
            gh = max(glyph_bb[3] - glyph_bb[1], 1.0)

            if idc in ("⿰", "⿲"):
                for i in range(len(slot_bbs) - 1):
                    if slot_bbs[i][2] > slot_bbs[i + 1][0] + 20.0:
                        valid_split = False
                        break
                for bb in slot_bbs:
                    w, h = bb[2] - bb[0], bb[3] - bb[1]
                    if w < 0.15 * gw or w > 0.80 * gw or h < 0.35 * gh:
                        valid_split = False
                        break
            elif idc in ("⿱", "⿳"):
                for i in range(len(slot_bbs) - 1):
                    if slot_bbs[i][1] < slot_bbs[i + 1][3] - 25.0:
                        valid_split = False
                        break
                for bb in slot_bbs:
                    w, h = bb[2] - bb[0], bb[3] - bb[1]
                    if w < 0.30 * gw or h < 0.08 * gh or h > 0.82 * gh:
                        valid_split = False
                        break
            if not valid_split:
                continue

            for k, comp in enumerate(comps):
                # Do NOT donate a component if `comp` is a known JP!=SC glyph (unless replaced by Master Bank)
                if len(comp) == 1 and comp in self.jp_diff_chars:
                    continue
                if ch in self.jp_diff_chars and len(comp) > 1:
                    # Conservative: multi-char sub-expressions inside JP!=SC donor might contain a JP stroke
                    continue

                self.positional[(idc, k, comp)].append(
                    ComponentEntry(cmds=slot_cmds[k], bbox=slot_bbs[k], donor_char=ch, is_master=False)
                )
                added += 1
        return added

    def register_master_glyph(
        self,
        char: str,
        cmds: List[DrawCmd],
        also_register_as_radical_slots: bool = True,
        overwrite_slots: bool = False,
    ) -> None:
        """Register a curated Master Bank glyph and canonical BOUND_RADICALS slots only."""
        bb = cmds_to_bbox(cmds)
        if bb is None:
            return
        entry = ComponentEntry(cmds=cmds, bbox=bb, donor_char=char, is_master=True)

        is_bound = char in ("讠", "钅", "纟", "饣", "辶", "亠", "礻", "衤", "忄", "艹", "冫", "阝")
        # Bound radicals occupy specific canonical slots rather than arbitrary sliced sub-slots
        if also_register_as_radical_slots and is_bound:
            if char == "辶":
                self.master_slot[("⿺", 0, "辶")] = entry
                self.master_full[char] = entry
            elif char in ("亠", "艹"):
                self.master_slot[("⿱", 0, char)] = entry
                self.master_slot[("⿳", 0, char)] = entry
                self.master_full[char] = entry
            elif char == "阝":
                # U+961D in Noto SC is left-ear `阝`; keep right-ear `阝` (`邦/都`) sourced from donors
                self.master_slot[("⿰", 0, "阝")] = entry
            else:
                self.master_slot[("⿰", 0, char)] = entry
                self.master_slot[("⿲", 0, char)] = entry
                self.master_full[char] = entry
        else:
            self.master_full[char] = entry

    def load_master_otf(
        self,
        master_otf_path: str | Path,
        edited_chars: Optional[Set[str]] = None,
    ) -> int:
        """Load curated glyphs from `manual_review/CnS-Manual-Master-Bank.otf`."""
        ext = SkipProOutlineExtractor(master_otf_path)
        edited = edited_chars or set()
        bound = set("讠钅纟饣辶亠礻衤忄艹冫阝")
        count = 0
        # Register non-composite / bound radicals first, then remaining unedited, then edited last
        char_cmds: List[Tuple[str, List[DrawCmd]]] = []
        for cp in sorted(ext.cmap.keys()):
            ch = chr(cp)
            if is_valid_cjk_char(ch):
                cmds = ext.get_cmds(ch)
                if cmds:
                    char_cmds.append((ch, cmds))
        for ch, cmds in char_cmds:
            if ch in bound:
                self.register_master_glyph(ch, cmds, overwrite_slots=True)
                count += 1
        for ch, cmds in char_cmds:
            if ch not in bound and ch not in edited:
                self.register_master_glyph(ch, cmds, overwrite_slots=False)
                count += 1
        for ch, cmds in char_cmds:
            if ch in edited and ch not in bound:
                self.register_master_glyph(ch, cmds, overwrite_slots=True)
                count += 1
        ext.close()
        return count

    def _get_verified_pool(self, idc: str, slot_k: int, comp: str) -> List[ComponentEntry]:
        key = (idc, slot_k, comp)
        if key in self._verified_donors:
            return self._verified_donors[key]

        raw_pool = self.positional.get(key, [])[:15]
        if not raw_pool:
            self._verified_donors[key] = []
            return []

        if self.sans_sc is not None and len(comp) == 1:
            sc_cmds = self.sans_sc.get_unioned_cmds(comp, "zh-Hans")
            sc_bb = cmds_to_bbox(sc_cmds) if sc_cmds else None
            if sc_cmds and sc_bb is not None:
                sc_w = max(sc_bb[2] - sc_bb[0], 1.0)
                sc_h = max(sc_bb[3] - sc_bb[1], 1.0)
                verified: List[ComponentEntry] = []
                for e in raw_pool:
                    ex0, ey0, ex1, ey1 = e.bbox
                    sx = sc_w / max(ex1 - ex0, 1.0)
                    sy = sc_h / max(ey1 - ey0, 1.0)
                    tx = sc_bb[0] - ex0 * sx
                    ty = sc_bb[1] - ey0 * sy
                    cand_fit = transform_cmds(e.cmds, Transform(sx, 0.0, 0.0, sy, tx, ty))
                    if cmds_iou(cand_fit, sc_cmds, size=56) >= 0.48:
                        verified.append(e)
                self._verified_donors[key] = verified
                return verified

        self._verified_donors[key] = raw_pool
        return raw_pool

    @staticmethod
    def _pick_best_for_bbox(pool: List[ComponentEntry], dst_bbox: Optional[BBox]) -> Optional[ComponentEntry]:
        if not pool:
            return None
        if dst_bbox is None or len(pool) == 1:
            return pool[0]
        dx0, dy0, dx1, dy1 = dst_bbox
        dw = max(dx1 - dx0, 1.0)
        dh = max(dy1 - dy0, 1.0)
        best_e = pool[0]
        best_cost = 1e9
        for e in pool:
            sx0, sy0, sx1, sy1 = e.bbox
            sw = max(sx1 - sx0, 1.0)
            sh = max(sy1 - sy0, 1.0)
            cost = (
                abs(math.log(sw / dw))
                + abs(math.log(sh / dh))
                + 0.25 * (abs(sx0 - dx0) + abs(sy0 - dy0)) / 1000.0
            )
            if cost < best_cost:
                best_cost = cost
                best_e = e
        return best_e

    def lookup_component(
        self,
        idc: Optional[str],
        slot_k: int,
        comp: str,
        dst_bbox: Optional[BBox] = None,
        allow_standalone: bool = True,
    ) -> Optional[ComponentEntry]:
        """Priority: 1) Master Slot -> 2) Master Full -> 3) Verified Positional SkipPro Donor -> 4) Verified Orientation Donor -> 5) Standalone."""
        if idc is not None and (idc, slot_k, comp) in self.master_slot:
            return self.master_slot[(idc, slot_k, comp)]
        if comp in self.master_full:
            return self.master_full[comp]
        if idc is not None:
            pool = self._get_verified_pool(idc, slot_k, comp)
            picked = self._pick_best_for_bbox(pool, dst_bbox)
            if picked is not None:
                return picked

        # Check other donors for same component in same structural orientation
        for (d_idc, d_k, d_comp) in list(self.positional.keys()):
            if d_comp == comp:
                if (idc == "⿰" and d_idc == "⿰") or (idc == "⿱" and d_idc == "⿱"):
                    pool = self._get_verified_pool(d_idc, d_k, d_comp)
                    picked = self._pick_best_for_bbox(pool, dst_bbox)
                    if picked is not None:
                        return picked

        if allow_standalone and len(comp) == 1 and comp not in self.jp_diff_chars and self.skip_ext.has_char(comp):
            cmds = self.skip_ext.get_cmds(comp)
            bb = cmds_to_bbox(cmds)
            if bb is not None:
                return ComponentEntry(cmds=cmds, bbox=bb, donor_char=comp, is_master=False)
        return None


def _normalize_cmds_hash(cmds: List[DrawCmd]) -> str:
    rounded = [(op, [(round(float(x), 1), round(float(y), 1)) for x, y in pts]) for op, pts in cmds]
    return hashlib.sha1(repr(rounded).encode("utf-8")).hexdigest()


def compute_otf_glyph_hashes(otf_path: str | Path) -> Dict[str, str]:
    """Compute per-character outline hashes for an exported subset Master Bank OTF."""
    ext = SkipProOutlineExtractor(otf_path)
    hashes: Dict[str, str] = {}
    for cp in sorted(ext.cmap.keys()):
        ch = chr(cp)
        if is_valid_cjk_char(ch):
            cmds = ext.get_cmds(ch)
            if cmds:
                hashes[ch] = _normalize_cmds_hash(cmds)
    ext.close()
    return hashes


def load_master_metadata(master_otf_path: str | Path) -> Tuple[Set[str], Dict[str, str]]:
    """
    Compare current Master Bank OTF outlines against `master_bank_hashes.json`
    to detect manually edited glyphs and read original per-glyph provenance.
    """
    otf_p = Path(master_otf_path)
    meta_p = otf_p.parent / "master_bank_hashes.json"
    if not otf_p.exists() or not meta_p.exists():
        return set(), {}

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    old_hashes: Dict[str, str] = meta.get("hashes", {})
    provenance: Dict[str, str] = meta.get("provenance", {})
    cur_hashes = compute_otf_glyph_hashes(otf_p)

    edited = {
        ch for ch, h in cur_hashes.items() if ch not in old_hashes or old_hashes[ch] != h
    }
    return edited, provenance


def estimate_target_slot_bboxes(
    ch: str,
    idc: str,
    comps: List[str],
    sans_sc: HBOutlineExtractor,
) -> Optional[List[BBox]]:
    """
    Measure exact `zh-Hans` sub-component bounding boxes from `Noto Sans SC` contours.
    """
    sc_cmds = sans_sc.get_unioned_cmds(ch, "zh-Hans")
    if not sc_cmds:
        return None
    full_bb = cmds_to_bbox(sc_cmds)
    if full_bb is None:
        return None
    gx0, gy0, gx1, gy1 = full_bb
    gw = max(gx1 - gx0, 1.0)
    gh = max(gy1 - gy0, 1.0)

    sc_islands = cmds_to_islands(sc_cmds)
    # Try island grouping if enough islands exist, with strict geometric validation
    if len(sc_islands) >= len(comps):
        buckets = _assign_islands_to_slots(sc_islands, idc, len(comps))
        if all(len(b) > 0 for b in buckets):
            boxes: List[BBox] = []
            for b in buckets:
                xs0 = [isl.bbox[0] for isl in b]
                ys0 = [isl.bbox[1] for isl in b]
                xs1 = [isl.bbox[2] for isl in b]
                ys1 = [isl.bbox[3] for isl in b]
                boxes.append((min(xs0), min(ys0), max(xs1), max(ys1)))
            valid = True
            if idc in ("⿰", "⿲"):
                for i in range(len(boxes) - 1):
                    cxi = 0.5 * (boxes[i][0] + boxes[i][2])
                    cxj = 0.5 * (boxes[i + 1][0] + boxes[i + 1][2])
                    # Allow genuine sweeping-撇 overhang up to 16% of width when components clearly advance L->R
                    if (
                        cxi >= cxj
                        or boxes[i][0] >= boxes[i + 1][0] - 0.08 * gw
                        or boxes[i][2] >= boxes[i + 1][2] - 0.10 * gw
                        or boxes[i][2] > boxes[i + 1][0] + 0.16 * gw
                    ):
                        valid = False
                        break
                for bb in boxes:
                    if (bb[2] - bb[0]) < 0.16 * gw or (bb[3] - bb[1]) < 0.35 * gh:
                        valid = False
                        break
            elif idc in ("⿱", "⿳"):
                for i in range(len(boxes) - 1):
                    if boxes[i][1] < boxes[i + 1][3] - 45.0:
                        valid = False
                        break
                for idx_b, bb in enumerate(boxes):
                    bw_b, bh_b = bb[2] - bb[0], bb[3] - bb[1]
                    if bw_b < 0.28 * gw or (bh_b < 0.16 * gh and comps[idx_b] != "一"):
                        valid = False
                        break
            elif idc == "⿺" and len(comps) == 2:
                if (
                    comps[0] not in ("辶", "廴")
                    or (boxes[1][2] - boxes[1][0]) < 0.45 * gw
                    or (boxes[1][3] - boxes[1][1]) < 0.45 * gh
                ):
                    valid = False
            elif idc in ("⿸", "⿹", "⿴", "⿵") and len(comps) == 2:
                b0_area = (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1])
                if (
                    (boxes[1][2] - boxes[1][0]) < 0.42 * gw
                    or (boxes[1][3] - boxes[1][1]) < 0.42 * gh
                    or b0_area < 0.45 * gw * gh
                ):
                    valid = False

            if valid:
                if idc == "⿰" and len(comps) == 2:
                    if boxes[0][2] > boxes[1][0] - 12.0 and boxes[0][2] > gx0 + 0.40 * gw:
                        boxes[0] = (
                            boxes[0][0],
                            boxes[0][1],
                            min(boxes[0][2], max(boxes[1][0] - 14.0, gx0 + 0.38 * gw)),
                            boxes[0][3],
                        )
                return boxes

    # Fallback proportional partitioning (never synthesize blind boxes for enclosing frames!)
    if idc == "⿰" and len(comps) == 2:
        split = gx0 + 0.37 * gw
        return [(gx0, gy0, split - 18.0, gy1), (split + 10.0, gy0, gx1, gy1)]
    if idc == "⿲" and len(comps) == 3:
        w3 = gw / 3.0
        return [(gx0, gy0, gx0 + w3 - 10, gy1), (gx0 + w3 + 5, gy0, gx0 + 2 * w3 - 5, gy1), (gx0 + 2 * w3 + 10, gy0, gx1, gy1)]
    if idc == "⿱" and len(comps) == 2:
        split = gy1 - 0.44 * gh
        return [(gx0, split + 10.0, gx1, gy1), (gx0, gy0, gx1, split - 12.0)]
    if idc == "⿳" and len(comps) == 3:
        h3 = gh / 3.0
        return [(gx0, gy1 - h3 + 8, gx1, gy1), (gx0, gy0 + h3 + 5, gx1, gy1 - h3 - 5), (gx0, gy0, gx1, gy0 + h3 - 8)]
    if idc == "⿺" and len(comps) == 2 and comps[0] in ("辶", "廴"):
        return [(gx0, gy0, gx1, gy1), (gx0 + 0.25 * gw, gy0 + 0.20 * gh, gx1 - 0.03 * gw, gy1)]
    return None


def compose_from_ids(
    ch: str,
    bank: PositionalDonorBank,
    sans_sc: HBOutlineExtractor,
    depth: int = 0,
) -> Optional[Tuple[List[DrawCmd], str]]:
    """
    Tier 1 Recursive Pure-Vector Composition:
      Assembles `ch` from Master Bank & Positional SkipPro Donors in 1000-UPEM space.
      Returns `(cmds, provenance_tag)` or `None` if an atomic/unresolvable component is needed.
    """
    if ch in bank.master_full and depth > 0:
        e = bank.master_full[ch]
        return e.cmds, f"master({e.donor_char})"

    seq = bank.ids_map.get(ch)
    if not seq or seq_has_invalid_entity(seq):
        return None
    idc, comps = top_split(seq)
    if idc is None or idc == "⿻" or comps == [ch]:
        return None

    slot_boxes = estimate_target_slot_bboxes(ch, idc, comps, sans_sc)
    if slot_boxes is None:
        return None

    part_cmds_list: List[List[DrawCmd]] = []
    tags: List[str] = []
    for k, comp in enumerate(comps):
        if comp in set("丶丨丿乀乙乚亅𠃌乛"):
            return None
        if comp == "一" and (idc not in ("⿱", "⿳") or (slot_boxes[k][3] - slot_boxes[k][1]) > 140.0):
            return None
        entry = bank.lookup_component(idc, k, comp, dst_bbox=slot_boxes[k], allow_standalone=True)
        if entry is not None:
            fitted = fit_component_with_stroke_compensation(entry.cmds, entry.bbox, slot_boxes[k])
            part_cmds_list.append(fitted)
            tags.append(f"{'M' if entry.is_master else 'D'}:{entry.donor_char}")
        elif depth < 2 and len(comp) >= 1 and comp in bank.ids_map:
            # Recursive sub-composition for multi-level IDS components!
            sub = _compose_sub_seq(comp, slot_boxes[k], bank, sans_sc, depth + 1)
            if sub is None:
                return None
            part_cmds_list.append(sub[0])
            tags.append(sub[1])
        else:
            return None

    merged: List[DrawCmd] = []
    for pc in part_cmds_list:
        merged.extend(pc)
    # Check if parts overlap; only boolean-union if bounding boxes overlap significantly,
    # otherwise direct concatenation preserves 100% untouched original CFF nodes!
    overlap = False
    for i in range(len(slot_boxes)):
        for j in range(i + 1, len(slot_boxes)):
            ax0, ay0, ax1, ay1 = slot_boxes[i]
            bx0, by0, bx1, by1 = slot_boxes[j]
            if min(ax1, bx1) > max(ax0, bx0) + 25.0 and min(ay1, by1) > max(ay0, by0) + 25.0:
                overlap = True
    if overlap:
        merged = boolean_union_cmds(part_cmds_list)

    if depth == 0:
        sc_cmds = sans_sc.get_unioned_cmds(ch, "zh-Hans")
        if sc_cmds and cmds_iou(merged, sc_cmds, size=80) < 0.58:
            return None

    return merged, f"Tier1[{','.join(tags)}]"


def _compose_sub_seq(
    sub_seq: str,
    target_bbox: BBox,
    bank: PositionalDonorBank,
    sans_sc: HBOutlineExtractor,
    depth: int,
) -> Optional[Tuple[List[DrawCmd], str]]:
    """Recursively compose an intermediate IDS sub-expression (e.g. `⿱士口`) into `target_bbox`."""
    if len(sub_seq) == 1:
        e = bank.lookup_component(None, 0, sub_seq, dst_bbox=target_bbox, allow_standalone=True)
        if e is None:
            return None
        return fit_component_with_stroke_compensation(e.cmds, e.bbox, target_bbox), f"{'M' if e.is_master else 'D'}:{e.donor_char}"

    idc, subs = top_split(sub_seq)
    if idc is None or idc == "⿻" or subs == [sub_seq]:
        return None
    tx0, ty0, tx1, ty1 = target_bbox
    tw, th = tx1 - tx0, ty1 - ty0
    if idc == "⿰" and len(subs) == 2:
        sub_boxes = [(tx0, ty0, tx0 + 0.48 * tw, ty1), (tx0 + 0.52 * tw, ty0, tx1, ty1)]
    elif idc == "⿱" and len(subs) == 2:
        sub_boxes = [(tx0, ty0 + 0.52 * th, tx1, ty1), (tx0, ty0, tx1, ty0 + 0.48 * th)]
    else:
        return None

    parts: List[List[DrawCmd]] = []
    tags: List[str] = []
    for k, s in enumerate(subs):
        e = bank.lookup_component(idc, k, s, dst_bbox=sub_boxes[k], allow_standalone=True)
        if e is not None:
            parts.append(fit_component_with_stroke_compensation(e.cmds, e.bbox, sub_boxes[k]))
            tags.append(e.donor_char)
        else:
            r = _compose_sub_seq(s, sub_boxes[k], bank, sans_sc, depth + 1)
            if r is None:
                return None
            parts.append(r[0])
            tags.append(r[1])
    out: List[DrawCmd] = []
    for p in parts:
        out.extend(p)
    return out, "+".join(tags)


def delta_island_surgery(
    ch: str,
    skip_ext: SkipProOutlineExtractor,
    serif_jp: HBOutlineExtractor,
    serif_sc: HBOutlineExtractor,
    sans_sc: HBOutlineExtractor,
    bank: PositionalDonorBank,
) -> Optional[Tuple[List[DrawCmd], str]]:
    """
    Tier 0 Delta Island Surgery (§3.1 & §3.2 #4) for JP != SC characters:
      - Matches `Noto Serif JP` islands against `Noto Serif SC` islands.
      - Keeps unchanged (`IoU >= 0.90`) `FOT-SkipPro-B` CFF islands 100% verbatim!
      - Replaces ONLY the changed Mainland islands from `Master Bank` / `PositionalDonorBank`
        (e.g., `言` keeps original `一+二+口` and swaps only the top horizontal bar for
        an original `SkipPro-B` teardrop dot; `微` keeps `彳+山+攵` and replaces only center!).
    """
    skip_cmds = skip_ext.get_cmds(ch)
    if not skip_cmds:
        return None

    jp_cmds = serif_jp.get_unioned_cmds(ch, "ja")
    sc_cmds = serif_sc.get_unioned_cmds(ch, "zh-Hans")
    if not jp_cmds or not sc_cmds:
        return None

    skip_isls = cmds_to_islands(skip_cmds)
    jp_isls = cmds_to_islands(jp_cmds)
    sc_isls = cmds_to_islands(sc_cmds)
    if not skip_isls or not jp_isls or not sc_isls:
        return None

    # Special Rule A: Pure top horizontal bar -> Mainland slanted teardrop dot (e.g. 言, 亩/etc.)
    if len(skip_isls) >= 4 and len(sc_isls) == len(skip_isls):
        top_skip = max(skip_isls, key=lambda x: x.cy)
        top_sc = max(sc_isls, key=lambda x: x.cy)
        sw = top_skip.bbox[2] - top_skip.bbox[0]
        sh = top_skip.bbox[3] - top_skip.bbox[1]
        cw = top_sc.bbox[2] - top_sc.bbox[0]
        ch_h = top_sc.bbox[3] - top_sc.bbox[1]
        if sw > 2.4 * max(sh, 1.0) and cw < 1.8 * max(ch_h, 1.0):
            # Borrow crisp teardrop dot from original SkipPro-B `立` or `永`
            dot_donor = bank.skip_ext.get_cmds("立") or bank.skip_ext.get_cmds("永")
            dot_isls = sorted(cmds_to_islands(dot_donor), key=lambda isl: -isl.cy)
            if dot_isls:
                kept = [isl for isl in skip_isls if isl is not top_skip]
                # Ensure clearance between top_dot bottom and next bar top
                next_top_y = max(isl.bbox[3] for isl in kept)
                dx0, dy0, dx1, dy1 = top_sc.bbox
                dot_target = (dx0, max(dy0, next_top_y + 22.0), dx1, max(dy1, next_top_y + 165.0))
                fitted_dot = fit_component_with_stroke_compensation(dot_isls[0].cmds, dot_isls[0].bbox, dot_target)
                out_cmds: List[DrawCmd] = []
                for isl in kept:
                    out_cmds.extend(isl.cmds)
                out_cmds.extend(fitted_dot)
                return out_cmds, f"Tier0_TopDotSwap({ch})"

    # Special Rule B: High-precision IDS slot replacement when char is composite & 1+ component changed
    seq = bank.ids_map.get(ch)
    if seq and not seq_has_invalid_entity(seq):
        idc, comps = top_split(seq)
        if idc is not None and idc != "⿻" and comps != [ch] and len(skip_isls) >= len(comps):
            skip_buckets = _assign_islands_to_slots(skip_isls, idc, len(comps))
            sc_boxes = estimate_target_slot_bboxes(ch, idc, comps, sans_sc)
            if sc_boxes is not None and all(len(b) > 0 for b in skip_buckets):
                rebuilt: List[List[DrawCmd]] = []
                replaced_any = False
                can_build = True
                for k, comp in enumerate(comps):
                    orig_slot_cmds = [cmd for isl in skip_buckets[k] for cmd in isl.cmds]
                    comp_has_diff = any(
                        (c in bank.jp_diff_chars or c in MAINLAND_VARIANT_MARKERS)
                        for c in comp
                        if is_valid_cjk_char(c)
                    )
                    if not comp_has_diff:
                        rebuilt.append(orig_slot_cmds)
                    else:
                        donor = bank.lookup_component(idc, k, comp, dst_bbox=sc_boxes[k], allow_standalone=True)
                        if donor is not None:
                            rebuilt.append(fit_component_with_stroke_compensation(donor.cmds, donor.bbox, sc_boxes[k]))
                            replaced_any = True
                        else:
                            can_build = False
                            break
                if can_build and replaced_any:
                    out: List[DrawCmd] = []
                    for blk in rebuilt:
                        out.extend(blk)
                    return out, f"Tier0_IDS({ch})"

    # Rule C: Match SkipPro-B -> Noto Serif JP (IoU >= 0.50) AND Serif JP -> Serif SC (IoU >= 0.88).
    # Never return KeepAll on JP!=SC targets; only perform DotSwap when all other islands
    # are provably identical between JP and SC and exactly one small island (< 24000) differs.
    kept_cmds: List[DrawCmd] = []
    matched_sc: Set[int] = set()
    kept_cnt = 0
    for s_isl in skip_isls:
        # Step 1: match original SkipPro island to Noto Serif JP island
        best_jp_idx = -1
        best_jp_iou = 0.0
        for ji, jp_isl in enumerate(jp_isls):
            uni_jp = s_isl.poly.union(jp_isl.poly).area
            iou_jp = s_isl.poly.intersection(jp_isl.poly).area / max(uni_jp, 1.0)
            if iou_jp > best_jp_iou:
                best_jp_iou = iou_jp
                best_jp_idx = ji
        if best_jp_idx < 0 or best_jp_iou < 0.50:
            continue

        # Step 2: check same-family Noto Serif JP vs Noto Serif SC agreement >= 0.88
        jp_isl = jp_isls[best_jp_idx]
        best_sc_idx = -1
        best_sc_iou = 0.0
        for si, sc_isl in enumerate(sc_isls):
            if si in matched_sc:
                continue
            uni_sc = jp_isl.poly.union(sc_isl.poly).area
            iou_sc = jp_isl.poly.intersection(sc_isl.poly).area / max(uni_sc, 1.0)
            if iou_sc > best_sc_iou:
                best_sc_iou = iou_sc
                best_sc_idx = si
        if best_sc_idx >= 0 and best_sc_iou >= 0.88:
            kept_cmds.extend(s_isl.cmds)
            matched_sc.add(best_sc_idx)
            kept_cnt += 1

    unmatched = [sc_isls[j] for j in range(len(sc_isls)) if j not in matched_sc]
    if kept_cnt >= 1 and len(unmatched) == 1 and unmatched[0].area < 24000.0:
        dot_donor = bank.skip_ext.get_cmds("立") or bank.skip_ext.get_cmds("永")
        dot_isls = sorted(cmds_to_islands(dot_donor), key=lambda isl: -isl.cy)
        if dot_isls:
            top_dot = dot_isls[0]
            fitted_dot = fit_component_with_stroke_compensation(top_dot.cmds, top_dot.bbox, unmatched[0].bbox)
            return kept_cmds + fitted_dot, f"Tier0_DotSwap({ch})"

    return None
