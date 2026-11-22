"""
Tier 2 Native Stroke-Primitive & Isolated-Subcomponent Assembler (§3.3)
and Clean Subsetted OpenType CFF Font Builder (`cffsubr` + `fontTools`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.ttLib import TTFont

from src.v2.component_bank import (
    ContourIsland,
    PositionalDonorBank,
    cmds_to_islands,
    fit_component_with_stroke_compensation,
)
from src.v2.curve_optimizer import (
    boolean_union_cmds,
    ensure_cff_winding,
    optimize_synthesized_contours,
)
from src.v2.hb_outlines import (
    BBox,
    DrawCmd,
    HBOutlineExtractor,
    SkipProOutlineExtractor,
    cmds_iou,
    cmds_to_bbox,
    rasterize_cmds,
    split_cmds_into_contours,
)


def _select_orig_islands_in_box(
    skip_ext: SkipProOutlineExtractor,
    donor_ch: str,
    x_min: float = -1e9,
    x_max: float = 1e9,
    y_min: float = -1e9,
    y_max: float = 1e9,
) -> Optional[Tuple[List[DrawCmd], BBox]]:
    """Select original SkipPro-B islands whose centroids fall inside the specified UPEM region."""
    cmds = skip_ext.get_cmds(donor_ch)
    if not cmds:
        return None
    chosen: List[DrawCmd] = []
    for isl in cmds_to_islands(cmds):
        if x_min <= isl.cx <= x_max and y_min <= isl.cy <= y_max:
            chosen.extend(isl.cmds)
    if not chosen:
        return None
    bb = cmds_to_bbox(chosen)
    return (chosen, bb) if bb is not None else None


class AtomicStrokeAssembler:
    """
    Retrieves & welds native CFF stroke/sub-island primitives from `1,381` JP==SC
    `SkipPro-B` glyphs for key atomic / crossing characters (`伞, 丌, 习, 乡, 卫, ...`).
    """

    def __init__(
        self,
        skip_ext: SkipProOutlineExtractor,
        sans_sc: HBOutlineExtractor,
        serif_sc: HBOutlineExtractor,
        jp_diff_chars: set[str],
    ) -> None:
        self.skip_ext = skip_ext
        self.sans_sc = sans_sc
        self.serif_sc = serif_sc
        self.jp_diff_chars = jp_diff_chars

        # Build isolated stroke bank from JP==SC glyphs (single-contour islands with small/medium area)
        self.stroke_pool: List[Tuple[str, ContourIsland, np.ndarray]] = []
        self._index_isolated_strokes()

    def _index_isolated_strokes(self) -> None:
        for cp in sorted(self.skip_ext.cmap.keys()):
            ch = chr(cp)
            if not (0x4E00 <= cp <= 0x9FFF) or ch in self.jp_diff_chars:
                continue
            cmds = self.skip_ext.get_cmds(ch)
            contours = split_cmds_into_contours(cmds)
            if len(contours) < 2:
                continue
            for c_cmds in contours:
                bb = cmds_to_bbox(c_cmds)
                if bb is None:
                    continue
                w = max(bb[2] - bb[0], 1.0)
                h = max(bb[3] - bb[1], 1.0)
                if w * h > 60000.0:
                    continue
                for isl in cmds_to_islands(c_cmds):
                    if isl.area < 45000.0:
                        feat = np.array([w, h, w / h, isl.area / (w * h)], dtype=np.float32)
                        self.stroke_pool.append((ch, isl, feat))
            if len(self.stroke_pool) >= 800:
                break

    @staticmethod
    def _measure_pi_columns(sc_cmds: List[DrawCmd], sc_bb: BBox) -> Tuple[BBox, BBox, BBox]:
        """Measure left sweep & right stem x-intervals of `丌` from rasterized Noto SC contours."""
        gx0, gy0, gx1, gy1 = sc_bb
        gw, gh = max(gx1 - gx0, 1.0), max(gy1 - gy0, 1.0)
        bar_dst = (gx0, gy1 - 0.14 * gh, gx1, gy1)
        sw_dst = (gx0 + 0.08 * gw, gy0, gx0 + 0.34 * gw, gy1 - 0.06 * gh)
        st_dst = (gx0 + 0.65 * gw, gy0, gx0 + 0.77 * gw, gy1 - 0.06 * gh)

        size = 128
        pad = 100.0
        scale = size / (1000.0 + 2.0 * pad)
        mask = rasterize_cmds(sc_cmds, canvas_size=size, padding=pad)
        # Analyze lower-middle band (35%..65% down from top bar) where two stems are separate
        y_top_px = int((880.0 + pad - (gy1 - 0.25 * gh)) * scale)
        y_bot_px = int((880.0 + pad - (gy0 + 0.28 * gh)) * scale)
        band = mask[max(0, y_top_px) : min(size, max(y_top_px + 1, y_bot_px)), :]
        if band.size > 0:
            col_hits = band.mean(axis=0) > 0.35
            spans: List[Tuple[int, int]] = []
            in_span = False
            s0 = 0
            for x, val in enumerate(col_hits):
                if val and not in_span:
                    s0 = x
                    in_span = True
                elif not val and in_span:
                    spans.append((s0, x))
                    in_span = False
            if in_span:
                spans.append((s0, len(col_hits)))
            if len(spans) >= 2:
                rx0 = spans[-1][0] / scale - pad
                rx1 = spans[-1][1] / scale - pad
                lx1 = spans[0][1] / scale - pad
                sw_dst = (gx0 + 0.04 * gw, gy0, max(lx1, gx0 + 0.28 * gw), gy1 - 0.05 * gh)
                st_dst = (rx0, gy0, max(rx1, rx0 + 75.0), gy1 - 0.05 * gh)
        return bar_dst, sw_dst, st_dst

    def try_assemble_curated_atomic(self, ch: str) -> Optional[Tuple[List[DrawCmd], str]]:
        """
        Assemble key atomic / irreducible characters using native `SkipPro-B` CFF primitives
        aligned precisely to unioned `Noto Serif/Sans SC` outlines, verifying IoU >= 0.58!
        """
        sc_sr = self.serif_sc.get_unioned_cmds(ch, "zh-Hans")
        sc_sn = self.sans_sc.get_unioned_cmds(ch, "zh-Hans")
        if not sc_sr or not sc_sn:
            return None
        sc_bb = cmds_to_bbox(sc_sr) or (40.0, -70.0, 960.0, 820.0)
        sc_isls_sr = cmds_to_islands(sc_sr)
        cand: Optional[Tuple[List[DrawCmd], str]] = None

        # 1. 伞 (U+4F1E): Top 人 from 会 + Middle 丷 dots + Lower 十
        if ch == "伞" and len(sc_isls_sr) >= 3:
            top_p = _select_orig_islands_in_box(self.skip_ext, "会", y_min=460)
            cross_p = _select_orig_islands_in_box(self.skip_ext, "十")
            if top_p and cross_p:
                top_dst = max(sc_isls_sr, key=lambda x: x.cy).bbox
                cross_dst = min(sc_isls_sr, key=lambda x: x.bbox[1]).bbox
                parts = [
                    fit_component_with_stroke_compensation(top_p[0], top_p[1], top_dst),
                    fit_component_with_stroke_compensation(cross_p[0], cross_p[1], cross_dst),
                ]
                mid_dots = [isl for isl in sc_isls_sr if isl.bbox != top_dst and isl.bbox != cross_dst]
                dot_src_l = _select_orig_islands_in_box(self.skip_ext, "米", y_min=520, x_max=450)
                dot_src_r = _select_orig_islands_in_box(self.skip_ext, "米", y_min=520, x_min=550)
                if len(mid_dots) >= 2 and dot_src_l and dot_src_r:
                    mid_dots.sort(key=lambda x: x.cx)
                    parts.append(fit_component_with_stroke_compensation(dot_src_l[0], dot_src_l[1], mid_dots[0].bbox))
                    parts.append(fit_component_with_stroke_compensation(dot_src_r[0], dot_src_r[1], mid_dots[-1].bbox))
                cand = (boolean_union_cmds(parts), "Tier2_Atomic(全+十+米)")

        # 2. 丌 (U+4E0C): Unioned Noto `丌` has 1 island; use measured stems + 二 top bar + 川 stems
        elif ch == "丌":
            left_sweep = _select_orig_islands_in_box(self.skip_ext, "川", x_max=320)
            right_stem = _select_orig_islands_in_box(self.skip_ext, "川", x_min=650)
            top_bar = _select_orig_islands_in_box(self.skip_ext, "二", y_min=450)
            if left_sweep and right_stem and top_bar:
                bar_dst, sw_dst, st_dst = self._measure_pi_columns(sc_sr, sc_bb)
                parts = [
                    fit_component_with_stroke_compensation(top_bar[0], top_bar[1], bar_dst),
                    fit_component_with_stroke_compensation(left_sweep[0], left_sweep[1], sw_dst),
                    fit_component_with_stroke_compensation(right_stem[0], right_stem[1], st_dst),
                ]
                cand = (boolean_union_cmds(parts), "Tier2_Atomic(二+川)")

        # 3. 习 (U+4E60): Outer 横折钩 borrowed from `司` + inner `冫` from `次`
        elif ch == "习" and len(sc_isls_sr) >= 2:
            cmds_si = self.skip_ext.get_cmds("司")
            si_isls = cmds_to_islands(cmds_si)
            si_hook = max(si_isls, key=lambda x: x.area) if si_isls else None
            ice = _select_orig_islands_in_box(self.skip_ext, "次", x_max=350)
            if si_hook is not None and ice is not None:
                sorted_sc = sorted(sc_isls_sr, key=lambda x: -x.area)
                hook_dst = sorted_sc[0].bbox
                rest = sorted_sc[1:]
                rx0 = min(r.bbox[0] for r in rest)
                ry0 = min(r.bbox[1] for r in rest)
                rx1 = max(r.bbox[2] for r in rest)
                ry1 = max(r.bbox[3] for r in rest)
                parts = [
                    fit_component_with_stroke_compensation(si_hook.cmds, si_hook.bbox, hook_dst),
                    fit_component_with_stroke_compensation(ice[0], ice[1], (rx0, ry0, rx1, ry1)),
                ]
                cand = (boolean_union_cmds(parts), "Tier2_Atomic(司+次)")

        # General multi-island isolated matching against 1,381 JP==SC SkipPro-B islands
        elif len(sc_isls_sr) >= 2 and all(len(split_cmds_into_contours(isl.cmds)) == 1 for isl in sc_isls_sr):
            matched_parts: List[List[DrawCmd]] = []
            for sc_isl in sc_isls_sr:
                w = max(sc_isl.bbox[2] - sc_isl.bbox[0], 1.0)
                h = max(sc_isl.bbox[3] - sc_isl.bbox[1], 1.0)
                q = np.array([w, h, w / h, sc_isl.area / (w * h)], dtype=np.float32)
                best_d = 1e9
                best_isl: Optional[ContourIsland] = None
                for _, cand_isl, feat in self.stroke_pool:
                    d = (
                        abs(np.log((feat[2] + 1e-4) / (q[2] + 1e-4))) * 3.0
                        + abs(feat[0] - q[0]) / 400.0
                        + abs(feat[1] - q[1]) / 400.0
                        + abs(feat[3] - q[3]) * 2.0
                    )
                    if d < best_d:
                        best_d = d
                        best_isl = cand_isl
                if best_isl is None or best_d > 0.65:
                    matched_parts = []
                    break
                matched_parts.append(fit_component_with_stroke_compensation(best_isl.cmds, best_isl.bbox, sc_isl.bbox))
            if matched_parts:
                cand = (boolean_union_cmds(matched_parts), "Tier2_StrokePool")

        if cand is not None:
            best_iou = max(cmds_iou(cand[0], sc_sr, size=96), cmds_iou(cand[0], sc_sn, size=96))
            if best_iou >= 0.58:
                return cand
        return None


def _draw_cmds_to_pen(cmds: List[DrawCmd], pen) -> None:
    for op, pts in cmds:
        if op == "moveTo":
            pen.moveTo(pts[0])
        elif op == "lineTo":
            pen.lineTo(pts[0])
        elif op == "curveTo":
            pen.curveTo(*pts)
        elif op == "closePath":
            pen.closePath()


def build_subset_master_otf(
    base_otf_path: str | Path,
    out_otf_path: str | Path,
    glyph_cmds_map: Dict[int, List[DrawCmd]],
    kanji_fd_index: int = 11,
) -> int:
    """
    Build a lightweight standalone OpenType CFF font containing ONLY `.notdef` plus
    the provided `{unicode_codepoint: DrawCmd}` master glyphs (1000 UPEM, matching
    FOT-SkipPro-B metrics & Private dict) for fast manual editing in FontCreator/FontLab!
    """
    tt = TTFont(str(base_otf_path))
    cff = tt["CFF "].cff
    top = cff[0]
    top.ROS = ("Adobe", "Identity", 0)
    priv = top.FDArray[kanji_fd_index].Private

    # Keep only .notdef from original font so FontCreator shows ONLY the ~108 master glyphs
    notdef_cs = top.CharStrings[".notdef"]
    notdef_hm = tt["hmtx"].metrics[".notdef"]
    notdef_vm = tt["vmtx"].metrics[".notdef"] if "vmtx" in tt else (1000, 120)

    new_order = [".notdef"]
    new_charset = [".notdef"]
    new_fdselect = [0]
    new_cs_dict = {".notdef": notdef_cs}
    new_hmtx = {".notdef": notdef_hm}
    new_vmtx = {".notdef": notdef_vm}
    new_cmap: Dict[int, str] = {}

    cid_num = 1
    for cp in sorted(glyph_cmds_map.keys()):
        cmds = glyph_cmds_map[cp]
        if not cmds:
            continue
        pen = T2CharStringPen(width=1000, glyphSet=None)
        _draw_cmds_to_pen(cmds, pen)
        cs = pen.getCharString(private=priv, globalSubrs=cff.GlobalSubrs)

        bp = BoundsPen(None)
        _draw_cmds_to_pen(cmds, bp)
        bounds = bp.bounds or (50, -50, 950, 830)
        lsb = int(round(bounds[0]))
        tsb = int(round(880 - bounds[3]))

        gname = f"cid{cid_num:05d}"
        cid_num += 1
        new_order.append(gname)
        new_charset.append(gname)
        new_fdselect.append(kanji_fd_index)
        new_cs_dict[gname] = cs
        new_hmtx[gname] = (1000, lsb)
        new_vmtx[gname] = (1000, tsb)
        new_cmap[cp] = gname

    # Rebuild CFF CharStrings index cleanly
    top.CharStrings.charStrings = {}
    top.CharStrings.charStringsIndex.items = []
    for idx, gname in enumerate(new_order):
        top.CharStrings.charStringsIndex.items.append(new_cs_dict[gname])
        top.CharStrings.charStrings[gname] = idx

    top.charset = new_charset
    top.FDSelect.format = 3
    top.FDSelect.gidArray = new_fdselect
    top.CIDCount = len(new_order) + 1
    tt.setGlyphOrder(new_order)
    tt["hmtx"].metrics = new_hmtx
    if "vmtx" in tt:
        tt["vmtx"].metrics = new_vmtx
    if "VORG" in tt:
        del tt["VORG"]
    for st in tt["cmap"].tables:
        if st.isUnicode():
            if st.format == 4:
                st.cmap = {cp: gn for cp, gn in new_cmap.items() if cp <= 0xFFFF}
            else:
                st.cmap = dict(new_cmap)

    # Clear GSUB/GPOS references to stripped glyphs so FontCreator opens cleanly without warnings
    for tbl in ("GSUB", "GPOS", "BASE", "kern"):
        if tbl in tt:
            del tt[tbl]

    out_p = Path(out_otf_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tt.save(str(out_p))
    tt.close()
    return len(new_cmap)


def pack_v2_tg8105_otf(
    base_otf_path: str | Path,
    out_otf_path: str | Path,
    replaced_and_new_cmds: Dict[int, List[DrawCmd]],
    kanji_fd_index: int = 11,
    subroutinize: bool = True,
) -> int:
    """
    Pack ALL new & JP->SC replaced glyphs into `FOT-SkipPro-B.OTF`, preserving
    every untouched original glyph, kana, Latin, GPOS, GSUB, and vertical metrics,
    and optionally compressing CFF charstrings via `cffsubr`.
    """
    tt = TTFont(str(base_otf_path))
    cff = tt["CFF "].cff
    top = cff[0]
    top.ROS = ("Adobe", "Identity", 0)
    priv = top.FDArray[kanji_fd_index].Private

    glyph_order = list(tt.getGlyphOrder())
    next_cid = max(top.CIDCount, 23060)
    hmtx = tt["hmtx"].metrics
    vmtx = tt["vmtx"].metrics if "vmtx" in tt else None
    added_map: Dict[int, str] = {}

    for cp, cmds in sorted(replaced_and_new_cmds.items()):
        if not cmds:
            continue
        pen = T2CharStringPen(width=1000, glyphSet=None)
        _draw_cmds_to_pen(cmds, pen)
        cs = pen.getCharString(private=priv, globalSubrs=cff.GlobalSubrs)

        bp = BoundsPen(None)
        _draw_cmds_to_pen(cmds, bp)
        bounds = bp.bounds or (50, -50, 950, 830)
        lsb = int(round(bounds[0]))
        tsb = int(round(880 - bounds[3]))

        gname = f"cid{next_cid:05d}"
        next_cid += 1
        new_gid = len(top.CharStrings.charStringsIndex)
        top.CharStrings.charStringsIndex.append(cs)
        top.CharStrings.charStrings[gname] = new_gid
        glyph_order.append(gname)
        top.charset.append(gname)
        top.FDSelect.append(kanji_fd_index)
        hmtx[gname] = (1000, lsb)
        if vmtx is not None:
            vmtx[gname] = (1000, tsb)
        added_map[cp] = gname

    top.CIDCount = next_cid + 1
    tt.setGlyphOrder(glyph_order)

    for st in tt["cmap"].tables:
        if st.isUnicode():
            for cp, gname in added_map.items():
                if st.format == 4 and cp > 0xFFFF:
                    continue
                st.cmap[cp] = gname

    out_p = Path(out_otf_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tt.save(str(out_p))
    tt.close()

    if subroutinize:
        try:
            import cffsubr

            sub_tt = TTFont(str(out_p))
            cffsubr.subroutinize(sub_tt)
            sub_tt.save(str(out_p))
            sub_tt.close()
        except Exception as e:
            print(f"[Warning] cffsubr subroutinize skipped: {e}", flush=True)

    return len(added_map)
