"""
v3.0 Local Surgery, Masked Local SDF Inpainting, Donor Bank, & SDF Synthesizer (`Plan3.0.md §1–§3`).

Zero Noto Serif contours or Sans-620 width inheritance anywhere in this module:
  - Uses `SansLightExtractor` (`wght=300`) + calibrated `SkipPro-B` half-widths `(h_half, v_half)`.
  - Protects all 12 `HARD_KEEP_12` chars as untouched `SkipPro-B` native donors.
  - Runs Stage 0a (pure-vector local surgery) -> Stage 0b (batched GPU masked local SDF inpainting)
    keeping untouched `SkipPro-B` contour islands 100% verbatim.
  - Enforces `StructuralGateV3` on every synthesized/modified candidate before acceptance.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from fontTools.misc.transform import Transform
from scipy.ndimage import distance_transform_edt as edt
from skimage.morphology import skeletonize

from src.glyph_renderer import CanvasSpec
from src.sdf_utils import mask_to_normalized_sdf
from src.v2.component_bank import (
    ComponentEntry,
    ContourIsland,
    PositionalDonorBank,
    _assign_islands_to_slots,
    _compose_sub_seq,
    cmds_to_islands,
    fit_component_with_stroke_compensation,
    is_valid_cjk_char,
    seq_has_invalid_entity,
    top_split,
    transform_cmds,
)
from src.v2.curve_optimizer import boolean_union_cmds
from src.v2.hb_outlines import (
    BBox,
    DrawCmd,
    SkipProOutlineExtractor,
    cmds_iou,
    cmds_to_bbox,
    rasterize_cmds,
)
from src.v2.stroke_assembler import AtomicStrokeAssembler, _select_orig_islands_in_box
from src.v3 import HARD_KEEP_CHARS
from src.v3.curve_constrain import optimize_synthesized_contours_v3
from src.v3.hb_light import (
    SansLightExtractor,
    align_cmds_to_bbox,
    build_v3_deskinned_channels,
    pack_v3_input_tensor,
    upem_box_to_mask,
)
from src.v3.model_v3 import LocalStrokeSDFNetV3
from src.v3.structural_gate import StructuralGateV3
from src.vectorizer import sdf_to_cff_commands


def _boxes_overlap(a: BBox, b: BBox, min_margin: float = 0.0) -> bool:
    return (min(a[2], b[2]) > max(a[0], b[0]) + min_margin) and (min(a[3], b[3]) > max(a[1], b[1]) + min_margin)


def _expand_slot_boxes_for_skippro(
    boxes: List[BBox],
    idc: str,
    full_bb: BBox,
    dx: float,
    dy: float,
) -> List[BBox]:
    """Expand Light-reference slot boxes outward by calibrated SkipPro half-width delta without colliding slots."""
    gx0, gy0, gx1, gy1 = full_bb
    max_x0, max_y0 = max(-20.0, gx0 - dx), max(-90.0, gy0 - dy)
    max_x1, max_y1 = min(1020.0, gx1 + dx), min(900.0, gy1 + dy)
    expanded = [
        (
            max(max_x0, bb[0] - dx),
            max(max_y0, bb[1] - dy),
            min(max_x1, bb[2] + dx),
            min(max_y1, bb[3] + dy),
        )
        for bb in boxes
    ]
    # Keep non-overlapping horizontal/vertical neighbors separated by >= 10 UPEM when raw boxes didn't cross
    if idc in ("⿰", "⿲"):
        for i in range(len(expanded) - 1):
            if boxes[i][2] <= boxes[i + 1][0] + 6.0 and expanded[i][2] > expanded[i + 1][0] - 8.0:
                mid = 0.5 * (boxes[i][2] + boxes[i + 1][0])
                expanded[i] = (expanded[i][0], expanded[i][1], min(expanded[i][2], mid - 4.0), expanded[i][3])
                expanded[i + 1] = (max(expanded[i + 1][0], mid + 4.0), expanded[i + 1][1], expanded[i + 1][2], expanded[i + 1][3])
    elif idc in ("⿱", "⿳"):
        for i in range(len(expanded) - 1):
            if boxes[i][1] >= boxes[i + 1][3] - 6.0 and expanded[i][1] < expanded[i + 1][3] + 8.0:
                mid = 0.5 * (boxes[i][1] + boxes[i + 1][3])
                expanded[i] = (expanded[i][0], max(expanded[i][1], mid + 4.0), expanded[i][2], expanded[i][3])
                expanded[i + 1] = (expanded[i + 1][0], expanded[i + 1][1], expanded[i + 1][2], min(expanded[i + 1][3], mid - 4.0))
    return expanded


def estimate_target_slot_bboxes_v3(
    ch: str,
    idc: str,
    comps: List[str],
    sn_sc: SansLightExtractor,
    h_half: float,
    v_half: float,
) -> Optional[List[BBox]]:
    """
    Measure `zh-Hans` sub-component bounding boxes from `Noto Sans Light` (`wght=300`),
    expanded outward to calibrated `SkipPro-B` stroke half-widths (Plan 3.0 §1.3).
    """
    sc_cmds = sn_sc.get_cmds(ch, unioned=True)
    if not sc_cmds:
        return None
    full_bb = cmds_to_bbox(sc_cmds)
    if full_bb is None:
        return None
    gx0, gy0, gx1, gy1 = full_bb
    gw = max(gx1 - gx0, 1.0)
    gh = max(gy1 - gy0, 1.0)
    dx = float(np.clip(v_half - 23.0, 0.0, 18.0))
    dy = float(np.clip(h_half - 22.0, 0.0, 14.0))

    sc_islands = cmds_to_islands(sc_cmds)
    if len(sc_islands) >= len(comps):
        buckets = _assign_islands_to_slots(sc_islands, idc, len(comps))
        if all(len(b) > 0 for b in buckets):
            boxes: List[BBox] = [
                (
                    min(isl.bbox[0] for isl in b),
                    min(isl.bbox[1] for isl in b),
                    max(isl.bbox[2] for isl in b),
                    max(isl.bbox[3] for isl in b),
                )
                for b in buckets
            ]
            valid = True
            if idc in ("⿰", "⿲"):
                for i in range(len(boxes) - 1):
                    cxi = 0.5 * (boxes[i][0] + boxes[i][2])
                    cxj = 0.5 * (boxes[i + 1][0] + boxes[i + 1][2])
                    if (
                        cxi >= cxj
                        or boxes[i][0] >= boxes[i + 1][0] - 0.08 * gw
                        or boxes[i][2] >= boxes[i + 1][2] - 0.10 * gw
                        or boxes[i][2] > boxes[i + 1][0] + 0.16 * gw
                    ):
                        valid = False
                        break
                for bb in boxes:
                    if (bb[2] - bb[0]) < 0.15 * gw or (bb[3] - bb[1]) < 0.35 * gh:
                        valid = False
                        break
            elif idc in ("⿱", "⿳"):
                for i in range(len(boxes) - 1):
                    if boxes[i][1] < boxes[i + 1][3] - 45.0:
                        valid = False
                        break
                for idx_b, bb in enumerate(boxes):
                    bw_b, bh_b = bb[2] - bb[0], bb[3] - bb[1]
                    if bw_b < 0.28 * gw or (bh_b < 0.14 * gh and comps[idx_b] != "一"):
                        valid = False
                        break
            elif idc == "⿺" and len(comps) == 2:
                if (
                    comps[0] not in ("辶", "廴")
                    or (boxes[1][2] - boxes[1][0]) < 0.44 * gw
                    or (boxes[1][3] - boxes[1][1]) < 0.44 * gh
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
                return _expand_slot_boxes_for_skippro(boxes, idc, full_bb, dx, dy)

    ex0, ey0, ex1, ey1 = gx0 - 0.5 * dx, gy0 - 0.5 * dy, gx1 + 0.5 * dx, gy1 + 0.5 * dy
    ew, eh = max(ex1 - ex0, 1.0), max(ey1 - ey0, 1.0)
    if idc == "⿰" and len(comps) == 2:
        split = ex0 + 0.37 * ew
        return [(ex0, ey0, split - 14.0, ey1), (split + 8.0, ey0, ex1, ey1)]
    if idc == "⿲" and len(comps) == 3:
        w3 = ew / 3.0
        return [(ex0, ey0, ex0 + w3 - 8, ey1), (ex0 + w3 + 4, ey0, ex0 + 2 * w3 - 4, ey1), (ex0 + 2 * w3 + 8, ey0, ex1, ey1)]
    if idc == "⿱" and len(comps) == 2:
        split = ey1 - 0.44 * eh
        return [(ex0, split + 8.0, ex1, ey1), (ex0, ey0, ex1, split - 10.0)]
    if idc == "⿳" and len(comps) == 3:
        h3 = eh / 3.0
        return [(ex0, ey1 - h3 + 6, ex1, ey1), (ex0, ey0 + h3 + 4, ex1, ey1 - h3 - 4), (ex0, ey0, ex1, ey0 + h3 - 6)]
    if idc == "⿺" and len(comps) == 2 and comps[0] in ("辶", "廴"):
        return [(ex0, ey0, ex1, ey1), (ex0 + 0.25 * ew, ey0 + 0.20 * eh, ex1 - 0.03 * ew, ey1)]
    return None


class DonorBankV3(PositionalDonorBank):
    """
    v3.0 Donor Bank:
      - Filters donors strictly by `cls.sc_variant` (preserving all `KEEP_*` & `V1/V2` chars as donors).
      - Uses `SansLightExtractor` (`wght=300`) instead of `Sans 620`.
      - Locks all 12 `HARD_KEEP_12` characters strictly to untouched original `SkipPro-B` CFF contours.
    """

    def __init__(
        self,
        skip_ext: SkipProOutlineExtractor,
        ids_map: Dict[str, str],
        sc_variant_chars: Set[str],
        sn_sc: SansLightExtractor,
        h_half: float,
        v_half: float,
    ) -> None:
        super().__init__(skip_ext=skip_ext, ids_map=ids_map, jp_diff_chars=set(sc_variant_chars), sans_sc=sn_sc)
        self.sn_sc = sn_sc
        self.h_half = float(h_half)
        self.v_half = float(v_half)
        self.native_keep_fallbacks: Dict[str, ComponentEntry] = {}
        self.register_hard_keep_donors()

    def register_hard_keep_donors(self) -> None:
        """Lock all present `HARD_KEEP_12` glyphs verbatim into native donor entries."""
        bound_keep = set("礻衤冫艹")
        for ch in sorted(HARD_KEEP_CHARS):
            if not self.skip_ext.has_char(ch):
                continue
            cmds = self.skip_ext.get_cmds(ch)
            bb = cmds_to_bbox(cmds) if cmds else None
            if not cmds or bb is None:
                continue
            entry = ComponentEntry(cmds=cmds, bbox=bb, donor_char=ch, is_master=False)
            self.native_keep_fallbacks[ch] = entry
            if ch not in bound_keep:
                self.master_full[ch] = entry

    def register_master_glyph(
        self,
        char: str,
        cmds: List[DrawCmd],
        also_register_as_radical_slots: bool = True,
        overwrite_slots: bool = False,
    ) -> None:
        if char in HARD_KEEP_CHARS:
            return
        super().register_master_glyph(
            char,
            cmds,
            also_register_as_radical_slots=also_register_as_radical_slots,
            overwrite_slots=overwrite_slots,
        )

    def lookup_component(
        self,
        idc: Optional[str],
        slot_k: int,
        comp: str,
        dst_bbox: Optional[BBox] = None,
        allow_standalone: bool = True,
    ) -> Optional[ComponentEntry]:
        entry = super().lookup_component(idc, slot_k, comp, dst_bbox=dst_bbox, allow_standalone=allow_standalone)
        if entry is not None:
            return entry
        if comp in self.native_keep_fallbacks:
            return self.native_keep_fallbacks[comp]
        return None


def compose_from_ids_v3(
    ch: str,
    bank: DonorBankV3,
    sn_sc: SansLightExtractor,
    depth: int = 0,
) -> Optional[Tuple[List[DrawCmd], str, List[BBox]]]:
    """
    Tier 1 Recursive Pure-Vector Composition using `Noto Sans Light` + `SkipPro` half-width expansion.
    Returns `(cmds, provenance_tag, slot_boxes)` or `None`.
    """
    if ch in bank.master_full and depth > 0:
        e = bank.master_full[ch]
        return e.cmds, f"master({e.donor_char})", [e.bbox]

    seq = bank.ids_map.get(ch)
    if not seq or seq_has_invalid_entity(seq):
        return None
    idc, comps = top_split(seq)
    if idc is None or idc == "⿻" or comps == [ch]:
        return None

    slot_boxes = estimate_target_slot_bboxes_v3(ch, idc, comps, sn_sc, bank.h_half, bank.v_half)
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
            sub = _compose_sub_seq(comp, slot_boxes[k], bank, sn_sc, depth + 1)
            if sub is None:
                return None
            part_cmds_list.append(sub[0])
            tags.append(sub[1])
        else:
            return None

    overlap = False
    for i in range(len(slot_boxes)):
        for j in range(i + 1, len(slot_boxes)):
            if _boxes_overlap(slot_boxes[i], slot_boxes[j], min_margin=25.0):
                overlap = True
                break
    if overlap:
        merged = boolean_union_cmds(part_cmds_list)
    else:
        merged = [cmd for pc in part_cmds_list for cmd in pc]

    if depth == 0:
        sc_cmds = sn_sc.get_cmds(ch, unioned=True)
        if sc_cmds and cmds_iou(merged, sc_cmds, size=80) < 0.42:
            return None

    return merged, f"Tier1[{','.join(tags)}]", slot_boxes


class AtomicStrokeAssemblerV3(AtomicStrokeAssembler):
    """Deskinned Tier 2 Assembler using `Noto Sans Light` (`wght=300`) — zero `Noto Serif`."""

    def __init__(
        self,
        skip_ext: SkipProOutlineExtractor,
        sn_sc: SansLightExtractor,
        sc_variant_chars: Set[str],
        h_half: float,
        v_half: float,
    ) -> None:
        super().__init__(skip_ext=skip_ext, sans_sc=sn_sc, serif_sc=sn_sc, jp_diff_chars=set(sc_variant_chars))
        self.h_half = float(h_half)
        self.v_half = float(v_half)

    def try_assemble_curated_atomic(self, ch: str) -> Optional[Tuple[List[DrawCmd], str]]:
        # For `丌`, expand right stem width measured on Light outline to match SkipPro-B vertical thickness
        if ch == "丌":
            sc_l = self.sans_sc.get_unioned_cmds(ch, "zh-Hans")
            sc_bb = cmds_to_bbox(sc_l) if sc_l else None
            left_sweep = _select_orig_islands_in_box(self.skip_ext, "川", x_max=320)
            right_stem = _select_orig_islands_in_box(self.skip_ext, "川", x_min=650)
            top_bar = _select_orig_islands_in_box(self.skip_ext, "二", y_min=450)
            if sc_l and sc_bb and left_sweep and right_stem and top_bar:
                bar_dst, sw_dst, st_dst = self._measure_pi_columns(sc_l, sc_bb)
                dx = max(0.0, self.v_half - 23.0)
                dy = max(0.0, self.h_half - 22.0)
                bar_dst = (bar_dst[0] - dx, bar_dst[1] - dy, bar_dst[2] + dx, bar_dst[3] + dy)
                sw_dst = (sw_dst[0] - dx, sw_dst[1] - dy, sw_dst[2] + dx, sw_dst[3])
                st_dst = (st_dst[0] - dx, st_dst[1] - dy, st_dst[2] + dx, st_dst[3])
                parts = [
                    fit_component_with_stroke_compensation(top_bar[0], top_bar[1], bar_dst),
                    fit_component_with_stroke_compensation(left_sweep[0], left_sweep[1], sw_dst),
                    fit_component_with_stroke_compensation(right_stem[0], right_stem[1], st_dst),
                ]
                cand = (boolean_union_cmds(parts), "Tier2_Atomic(二+川)")
                if cmds_iou(cand[0], sc_l, size=96) >= 0.45:
                    return cand
                return None

        res = super().try_assemble_curated_atomic(ch)
        if res is not None:
            return res
        # Lower Light-vs-SkipPro IoU threshold for `伞` / `习` / multi-island matches so StructuralGateV3 decides
        sc_l = self.sans_sc.get_unioned_cmds(ch, "zh-Hans")
        if not sc_l:
            return None
        sc_isls = cmds_to_islands(sc_l)
        cand: Optional[Tuple[List[DrawCmd], str]] = None
        if ch == "伞" and len(sc_isls) >= 3:
            top_p = _select_orig_islands_in_box(self.skip_ext, "会", y_min=460)
            cross_p = _select_orig_islands_in_box(self.skip_ext, "十")
            if top_p and cross_p:
                dx = max(0.0, self.v_half - 23.0)
                dy = max(0.0, self.h_half - 22.0)
                top_dst = max(sc_isls, key=lambda x: x.cy).bbox
                cross_dst = min(sc_isls, key=lambda x: x.bbox[1]).bbox
                t_exp = (top_dst[0] - dx, top_dst[1] - dy, top_dst[2] + dx, top_dst[3] + dy)
                c_exp = (cross_dst[0] - dx, cross_dst[1] - dy, cross_dst[2] + dx, cross_dst[3] + dy)
                parts = [
                    fit_component_with_stroke_compensation(top_p[0], top_p[1], t_exp),
                    fit_component_with_stroke_compensation(cross_p[0], cross_p[1], c_exp),
                ]
                mid_dots = [isl for isl in sc_isls if isl.bbox != top_dst and isl.bbox != cross_dst]
                dot_l = _select_orig_islands_in_box(self.skip_ext, "米", y_min=520, x_max=450)
                dot_r = _select_orig_islands_in_box(self.skip_ext, "米", y_min=520, x_min=550)
                if len(mid_dots) >= 2 and dot_l and dot_r:
                    mid_dots.sort(key=lambda x: x.cx)
                    d0, d1 = mid_dots[0].bbox, mid_dots[-1].bbox
                    parts.append(fit_component_with_stroke_compensation(dot_l[0], dot_l[1], (d0[0] - dx, d0[1] - dy, d0[2] + dx, d0[3] + dy)))
                    parts.append(fit_component_with_stroke_compensation(dot_r[0], dot_r[1], (d1[0] - dx, d1[1] - dy, d1[2] + dx, d1[3] + dy)))
                cand = (boolean_union_cmds(parts), "Tier2_Atomic(全+十+米)")
        if cand is not None and cmds_iou(cand[0], sc_l, size=96) >= 0.45:
            return cand
        return None


class ZHSDFSynthesizerV3:
    """
    Deskinned Sans Light Neural SDF Fallback Generator (`Plan3.0.md §1–§2`):
      - Drives `LocalStrokeSDFNetV3` with uniform skeleton SDF & tangent orientation field.
      - Uses curvature-masked, Hausdorff-guarded `optimize_synthesized_contours_v3`.
      - Enforces `StructuralGateV3` with up to 3 jittered `(dw, r_u)` retries.
    """

    def __init__(
        self,
        ckpt_path: Path,
        canvas_size: int = 256,
        base_channels: int = 64,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canvas_size = canvas_size
        self.trunc_px = 16.0 * (canvas_size / 256.0)
        self.model = LocalStrokeSDFNetV3(4, base_channels=base_channels, max_shift_px=8.0, delta_scale=0.95).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.upem_spec = CanvasSpec(
            canvas_size=canvas_size,
            em_ratio=1000.0 / 1192.0,
            upem=1000,
            cjk_ascender=880,
        )

    def predict_sdf_batch(
        self,
        cmds_list: List[List[DrawCmd]],
        dw: float = 0.0,
        r_u: float = 40.0,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Predict `[canvas_size, canvas_size]` float32 SDF arrays for aligned Light contours."""
        if not cmds_list:
            return []
        out: List[np.ndarray] = []
        amp_dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.no_grad():
            for st in range(0, len(cmds_list), batch_size):
                sub = cmds_list[st : st + batch_size]
                tensors = []
                for cmds in sub:
                    sl, dtu, c2, s2 = build_v3_deskinned_channels(
                        cmds,
                        canvas_size=self.canvas_size,
                        padding=96.0,
                        trunc_px=self.trunc_px,
                        uniform_half_width_u=r_u,
                    )
                    tensors.append(
                        pack_v3_input_tensor(
                            sl,
                            dtu,
                            c2,
                            s2,
                            uniform_half_width_u=r_u,
                            canvas_size=self.canvas_size,
                            padding=96.0,
                            trunc_px=self.trunc_px,
                            sdf_shift=dw,
                        )
                    )
                xt = torch.from_numpy(np.stack(tensors, axis=0)).to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=self.device.type == "cuda"):
                    pred_t, _ = self.model(xt)
                pred_np = pred_t[:, 0].float().cpu().numpy()
                for i in range(len(sub)):
                    out.append(pred_np[i])
        return out

    def vectorize_and_constrain(
        self,
        sdf_2d: np.ndarray,
        ref_sc_cmds: Optional[List[DrawCmd]],
        upsample_factor: int = 3,
    ) -> Tuple[List[DrawCmd], List[DrawCmd], dict]:
        """Convert predicted SDF into CFF outlines and apply curvature-constrained cleanup."""
        raw_cmds = sdf_to_cff_commands(sdf_2d, self.upem_spec, upsample_factor=upsample_factor)
        if not raw_cmds:
            return [], [], {"snapped_contours": 0, "reverted_contours": 0}
        clean_cmds, opt_stats = optimize_synthesized_contours_v3(
            raw_cmds,
            ref_sc_cmds=ref_sc_cmds,
            stem_concave_sagitta=0.85,
            max_hausdorff_upem=4.0,
            max_area_rel=0.015,
        )
        return clean_cmds, raw_cmds, opt_stats

    def synthesize_with_gate_retries(
        self,
        chars: List[str],
        sn_sc: SansLightExtractor,
        gate: StructuralGateV3,
        exempt_boxes_map: Optional[Dict[str, List[BBox]]] = None,
        batch_size: int = 32,
        upsample_factor: int = 3,
        check_jp_diff: bool = True,
    ) -> Tuple[Dict[str, Tuple[List[DrawCmd], str, dict]], Dict[str, dict]]:
        """
        Synthesize `chars` with 3 progressive `(dw, r_u)` jitter retries under `StructuralGateV3`.
        Never admits failed characters into `accepted_map`!
        """
        accepted: Dict[str, Tuple[List[DrawCmd], str, dict]] = {}
        last_diag: Dict[str, dict] = {}
        if not chars:
            return accepted, last_diag

        # Extract HarfBuzz outlines & warm gate cache sequentially before parallel checking
        ref_cmds_map: Dict[str, List[DrawCmd]] = {}
        remaining: List[str] = []
        for ch in chars:
            gate.get_ref_bundle(ch)
            sc_c = sn_sc.get_cmds(ch, unioned=False)
            if sc_c:
                ref_cmds_map[ch] = sc_c
                remaining.append(ch)
            else:
                last_diag[ch] = {"passed": False, "reasons": ["no_sc_ref"]}

        attempts = [(0.0, 40.0), (-0.04, 43.0), (0.05, 37.0)]
        workers = min(8, max(2, (os.cpu_count() or 4)))

        for att_idx, (dw, r_u) in enumerate(attempts):
            if not remaining:
                break
            preds = self.predict_sdf_batch([ref_cmds_map[c] for c in remaining], dw=dw, r_u=r_u, batch_size=batch_size)

            def _eval_one(item: Tuple[str, np.ndarray]) -> Tuple[str, bool, List[DrawCmd], dict]:
                ch, sdf_arr = item
                sc_c = ref_cmds_map[ch]
                clean_cmds, raw_cmds, opt_st = self.vectorize_and_constrain(sdf_arr, sc_c, upsample_factor=upsample_factor)
                ex_boxes = (exempt_boxes_map or {}).get(ch) or None
                ok, gm = gate.check(clean_cmds, ch, exempt_boxes=ex_boxes, check_jp_diff=check_jp_diff)
                if ok:
                    return ch, True, clean_cmds, {**gm, **opt_st, "attempt": att_idx}
                # Fallback check without snapping if constrained snapping hurt gate score
                ok_raw, gm_raw = gate.check(raw_cmds, ch, exempt_boxes=ex_boxes, check_jp_diff=check_jp_diff)
                if ok_raw:
                    return ch, True, raw_cmds, {**gm_raw, "snapped_contours": 0, "reverted_contours": 0, "attempt": att_idx}
                return ch, False, clean_cmds, {**gm, **opt_st, "attempt": att_idx}

            next_remaining: List[str] = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for ch, ok, cmds, meta in ex.map(_eval_one, zip(remaining, preds)):
                    if ok:
                        accepted[ch] = (cmds, f"Tier3_DeskinnedSDF(att={att_idx})", meta)
                        last_diag.pop(ch, None)
                    else:
                        meta["fallback_cmds"] = cmds
                        last_diag[ch] = meta
                        next_remaining.append(ch)
            remaining = next_remaining

        return accepted, last_diag


def align_pair_via_jp_to_orig(
    sc_cmds: List[DrawCmd],
    jp_cmds: List[DrawCmd],
    orig_cmds: List[DrawCmd],
) -> Tuple[List[DrawCmd], List[DrawCmd]]:
    """
    Affinely map BOTH `sc_cmds` and `jp_cmds` using the single `jp -> orig` transform so
    shared unchanged strokes align seam-free without bbox squashing when SC heights differ.
    """
    bj = cmds_to_bbox(jp_cmds)
    bo = cmds_to_bbox(orig_cmds)
    if bj is None or bo is None:
        return align_cmds_to_bbox(sc_cmds, orig_cmds), align_cmds_to_bbox(jp_cmds, orig_cmds)
    wj, hj = max(bj[2] - bj[0], 1.0), max(bj[3] - bj[1], 1.0)
    wo, ho = max(bo[2] - bo[0], 1.0), max(bo[3] - bo[1], 1.0)
    sx, sy = wo / wj, ho / hj
    tx = Transform(sx, 0.0, 0.0, sy, bo[0] - bj[0] * sx, bo[1] - bj[1] * sy)
    return transform_cmds(sc_cmds, tx), transform_cmds(jp_cmds, tx)


def tier0_stage0a_vector_surgery(
    ch: str,
    skip_ext: SkipProOutlineExtractor,
    sn_jp: SansLightExtractor,
    sn_sc: SansLightExtractor,
    bank: DonorBankV3,
    gate: StructuralGateV3,
    exempt_boxes: Optional[List[BBox]] = None,
) -> Optional[Tuple[List[DrawCmd], str, dict]]:
    """
    Stage 0a Pure-Vector Local Surgery (CPU, zero Noto Serif):
      1. `⿺辶X` composite surgery preserving untouched inner `X` islands verbatim.
      2. Top bar -> native teardrop dot swap (`言, 亠, 高, ...`).
      3. Composite IDS slot replacement (changed component slots only).
      4. Single small-island add / delete / swap using bbox-aligned Light references.
    """
    orig_cmds = skip_ext.get_cmds(ch)
    if not orig_cmds:
        return None
    skip_isls = cmds_to_islands(orig_cmds)
    sc_u = sn_sc.get_cmds(ch, unioned=True)
    jp_u = sn_jp.get_cmds(ch, unioned=True)
    if not skip_isls or not sc_u or not jp_u:
        return None

    sc_aligned, jp_aligned = align_pair_via_jp_to_orig(sc_u, jp_u, orig_cmds)
    sc_isls = cmds_to_islands(sc_aligned)
    jp_isls = cmds_to_islands(jp_aligned)
    seq = bank.ids_map.get(ch)

    # Rule 1: `⿺辶X` preserving native SkipPro-B inner `X` verbatim
    if seq and not seq_has_invalid_entity(seq):
        idc, comps = top_split(seq)
        if idc == "⿺" and len(comps) == 2 and comps[0] == "辶" and ("辶" in bank.master_full or ("⿺", 0, "辶") in bank.master_slot):
            inner = comps[1]
            inner_has_diff = any(c in bank.jp_diff_chars for c in inner if is_valid_cjk_char(c))
            if not inner_has_diff and len(skip_isls) >= 2:
                buckets = _assign_islands_to_slots(skip_isls, "⿺", 2)
                if buckets[0] and buckets[1]:
                    zou_cmds = [c for isl in buckets[0] for c in isl.cmds]
                    in_cmds = [c for isl in buckets[1] for c in isl.cmds]
                    zou_bb = cmds_to_bbox(zou_cmds)
                    zou_entry = bank.master_slot.get(("⿺", 0, "辶")) or bank.master_full.get("辶")
                    if zou_bb is not None and zou_entry is not None:
                        fitted_zou = fit_component_with_stroke_compensation(zou_entry.cmds, zou_entry.bbox, zou_bb)
                        cand = fitted_zou + in_cmds
                        ok, gm = gate.check(cand, ch, exempt_boxes=exempt_boxes, replaced_boxes=[zou_bb])
                        if ok:
                            return cand, f"Tier0_ZouInnerNative({ch})", gm

    # Rule 2: Top horizontal bar -> native teardrop dot swap (`言, 亠, ...`)
    if len(skip_isls) >= 4 and len(sc_isls) == len(skip_isls):
        top_skip = max(skip_isls, key=lambda x: x.cy)
        top_sc = max(sc_isls, key=lambda x: x.cy)
        sw, sh = top_skip.bbox[2] - top_skip.bbox[0], top_skip.bbox[3] - top_skip.bbox[1]
        cw, ch_h = top_sc.bbox[2] - top_sc.bbox[0], top_sc.bbox[3] - top_sc.bbox[1]
        if sw > 2.2 * max(sh, 1.0) and cw < 1.9 * max(ch_h, 1.0):
            dot_donor = skip_ext.get_cmds("立") or skip_ext.get_cmds("永")
            dot_isls = sorted(cmds_to_islands(dot_donor), key=lambda isl: -isl.cy)
            if dot_isls:
                kept = [isl for isl in skip_isls if isl is not top_skip]
                next_top_y = max(isl.bbox[3] for isl in kept)
                dx = max(0.0, bank.v_half - 23.0)
                dx0, dy0, dx1, dy1 = top_sc.bbox
                dot_target = (dx0 - dx, max(dy0 - 12.0, next_top_y + 20.0), dx1 + dx, max(dy1 + 14.0, next_top_y + 160.0))
                fitted_dot = fit_component_with_stroke_compensation(dot_isls[0].cmds, dot_isls[0].bbox, dot_target)
                cand = [c for isl in kept for c in isl.cmds] + fitted_dot
                ok, gm = gate.check(cand, ch, exempt_boxes=exempt_boxes, replaced_boxes=[dot_target])
                if ok:
                    return cand, f"Tier0_TopDotSwap({ch})", gm

    # Rule 3: Composite IDS slot replacement keeping unchanged SkipPro-B slots verbatim
    if seq and not seq_has_invalid_entity(seq):
        idc, comps = top_split(seq)
        if idc is not None and idc != "⿻" and comps != [ch] and len(skip_isls) >= len(comps):
            skip_buckets = _assign_islands_to_slots(skip_isls, idc, len(comps))
            sc_boxes = estimate_target_slot_bboxes_v3(ch, idc, comps, sn_sc, bank.h_half, bank.v_half)
            if sc_boxes is not None and all(len(b) > 0 for b in skip_buckets):
                rebuilt: List[List[DrawCmd]] = []
                replaced_boxes: List[BBox] = []
                can_build = True
                for k, comp in enumerate(comps):
                    orig_slot_cmds = [cmd for isl in skip_buckets[k] for cmd in isl.cmds]
                    comp_has_diff = any(c in bank.jp_diff_chars for c in comp if is_valid_cjk_char(c))
                    if not comp_has_diff:
                        rebuilt.append(orig_slot_cmds)
                    else:
                        donor = bank.lookup_component(idc, k, comp, dst_bbox=sc_boxes[k], allow_standalone=True)
                        if donor is not None:
                            rebuilt.append(fit_component_with_stroke_compensation(donor.cmds, donor.bbox, sc_boxes[k]))
                            replaced_boxes.append(sc_boxes[k])
                        elif len(comp) >= 1 and comp in bank.ids_map:
                            sub = _compose_sub_seq(comp, sc_boxes[k], bank, sn_sc, 1)
                            if sub is not None:
                                rebuilt.append(sub[0])
                                replaced_boxes.append(sc_boxes[k])
                            else:
                                can_build = False
                                break
                        else:
                            can_build = False
                            break
                if can_build and replaced_boxes:
                    cand = [c for blk in rebuilt for c in blk]
                    ok, gm = gate.check(cand, ch, exempt_boxes=exempt_boxes, replaced_boxes=replaced_boxes)
                    if ok:
                        return cand, f"Tier0_IDS({ch})", gm

    # Rule 4: Single small-island add / delete / swap (e.g. extra or missing dot) using Light refs
    if len(skip_isls) >= 2 and jp_isls and sc_isls:
        kept_cmds: List[DrawCmd] = []
        matched_sc: Set[int] = set()
        kept_cnt = 0
        for s_isl in skip_isls:
            best_ji, best_j_iou = -1, 0.0
            for ji, j_isl in enumerate(jp_isls):
                iou = s_isl.poly.intersection(j_isl.poly).area / max(s_isl.poly.union(j_isl.poly).area, 1.0)
                if iou > best_j_iou:
                    best_j_iou, best_ji = iou, ji
            if best_ji < 0 or best_j_iou < 0.45:
                continue
            j_isl = jp_isls[best_ji]
            best_si, best_s_iou = -1, 0.0
            for si, c_isl in enumerate(sc_isls):
                if si in matched_sc:
                    continue
                iou = j_isl.poly.intersection(c_isl.poly).area / max(j_isl.poly.union(c_isl.poly).area, 1.0)
                if iou > best_s_iou:
                    best_s_iou, best_si = iou, si
            if best_si >= 0 and best_s_iou >= 0.72:
                kept_cmds.extend(s_isl.cmds)
                matched_sc.add(best_si)
                kept_cnt += 1

        unmatched = [sc_isls[j] for j in range(len(sc_isls)) if j not in matched_sc]
        if kept_cnt >= 1 and len(unmatched) == 0 and kept_cnt < len(skip_isls):
            ok, gm = gate.check(kept_cmds, ch, exempt_boxes=exempt_boxes)
            if ok:
                return kept_cmds, f"Tier0_IslandDelete({ch})", gm
        if kept_cnt >= 1 and len(unmatched) == 1 and unmatched[0].area < 24000.0:
            dot_donor = skip_ext.get_cmds("立") or skip_ext.get_cmds("永")
            dot_isls = sorted(cmds_to_islands(dot_donor), key=lambda isl: -isl.cy)
            if dot_isls:
                ux0, uy0, ux1, uy1 = unmatched[0].bbox
                dx = max(0.0, bank.v_half - 23.0)
                dst = (ux0 - dx, uy0 - dx, ux1 + dx, uy1 + dx)
                fitted = fit_component_with_stroke_compensation(dot_isls[0].cmds, dot_isls[0].bbox, dst)
                cand = kept_cmds + fitted
                ok, gm = gate.check(cand, ch, exempt_boxes=exempt_boxes, replaced_boxes=[dst])
                if ok:
                    return cand, f"Tier0_DotSwap({ch})", gm

    return None


def _feather_box_mask(
    ys: np.ndarray,
    xs: np.ndarray,
    pad_u: float,
    upem_px: float,
    exempt_boxes: Optional[List[BBox]],
    canvas_size: int,
    padding: float,
    sigma_x: float,
) -> np.ndarray:
    pad_px = max(3, int(round(pad_u / upem_px)))
    y0, y1 = max(0, int(ys.min()) - pad_px), min(canvas_size, int(ys.max()) + pad_px + 1)
    x0, x1 = max(0, int(xs.min()) - pad_px), min(canvas_size, int(xs.max()) + pad_px + 1)
    mb = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    mb[y0:y1, x0:x1] = 1.0
    if exempt_boxes:
        scale = canvas_size / (1000.0 + 2.0 * padding)
        for bx0, by0, bx1, by1 in exempt_boxes:
            ex0 = int(np.clip(np.floor((bx0 - 16.0 + padding) * scale), 0, canvas_size - 1))
            ex1 = int(np.clip(np.ceil((bx1 + 16.0 + padding) * scale), 0, canvas_size))
            ey0 = int(np.clip(np.floor((880.0 + padding - (by1 + 16.0)) * scale), 0, canvas_size - 1))
            ey1 = int(np.clip(np.ceil((880.0 + padding - (by0 - 16.0)) * scale), 0, canvas_size))
            if ex1 > ex0 and ey1 > ey0:
                mb[ey0:ey1, ex0:ex1] = 0.0
    return np.clip(cv2.GaussianBlur(mb, (0, 0), sigmaX=sigma_x), 0.0, 1.0).astype(np.float32)


def _build_local_inpaint_masks(
    sc_cmds: List[DrawCmd],
    jp_cmds: List[DrawCmd],
    orig_cmds: List[DrawCmd],
    exempt_boxes: Optional[List[BBox]],
    canvas_size: int = 256,
    padding: float = 96.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build progressive feathered local inpaint masks M1 (62u), M2 (78u box), and M3 (115u box)."""
    s_u = rasterize_cmds(sc_cmds, canvas_size=canvas_size, padding=padding)
    j_u = rasterize_cmds(jp_cmds, canvas_size=canvas_size, padding=padding)
    o_u = rasterize_cmds(orig_cmds, canvas_size=canvas_size, padding=padding)
    skel_sc = skeletonize(s_u)
    skel_jp = skeletonize(j_u)
    upem_px = (1000.0 + 2.0 * padding) / float(canvas_size)
    dt_sc = (edt(~skel_sc) * upem_px).astype(np.float32) if skel_sc.any() else np.full((canvas_size, canvas_size), 999.0, dtype=np.float32)
    dt_jp = (edt(~skel_jp) * upem_px).astype(np.float32) if skel_jp.any() else np.full((canvas_size, canvas_size), 999.0, dtype=np.float32)

    d_local = (skel_sc & (dt_jp > 20.0)) | (skel_jp & (dt_sc > 20.0))
    if int(d_local.sum()) < 4:
        d_local = (skel_sc & (dt_jp > 12.0)) | (skel_jp & (dt_sc > 12.0))
    if int(d_local.sum()) < 4:
        d_local = s_u ^ o_u

    # Include Bold SkipPro overhang extending >52u beyond Sans Light SC so JP tails aren't orphaned
    r_sc = max(2, int(round(52.0 / upem_px)))
    sc_dil = cv2.dilate(s_u.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_sc + 1, 2 * r_sc + 1))) > 0
    d_local = d_local | (o_u & (~sc_dil))

    if exempt_boxes:
        for box in exempt_boxes:
            d_local &= ~upem_box_to_mask(box, canvas_size=canvas_size, padding=padding, margin_u=22.0)

    if not d_local.any():
        full = np.ones((canvas_size, canvas_size), dtype=np.float32)
        return full, full, full

    r1 = max(2, int(round(62.0 / upem_px)))
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r1 + 1, 2 * r1 + 1))
    m1_bin = cv2.dilate(d_local.astype(np.uint8) * 255, k1)
    m1 = cv2.GaussianBlur(m1_bin.astype(np.float32) / 255.0, (0, 0), sigmaX=3.5)
    m1 = np.clip(np.maximum(m1, (m1_bin > 127).astype(np.float32) * 0.92), 0.0, 1.0).astype(np.float32)

    ys, xs = np.where(d_local)
    m2 = _feather_box_mask(ys, xs, 78.0, upem_px, exempt_boxes, canvas_size, padding, sigma_x=4.0)
    m3 = _feather_box_mask(ys, xs, 115.0, upem_px, exempt_boxes, canvas_size, padding, sigma_x=4.5)
    return m1, m2, m3


def _assemble_blended_islands(
    blended_sdf: np.ndarray,
    mask_m: np.ndarray,
    orig_isls: List[ContourIsland],
    sc_cmds: List[DrawCmd],
    synth: ZHSDFSynthesizerV3,
    upsample_factor: int = 3,
) -> Tuple[List[DrawCmd], dict]:
    """Preserve untouched original `SkipPro-B` islands verbatim and optimize ONLY inpainted islands."""
    S = synth.canvas_size
    m_edit_bin = mask_m >= 0.08
    kept_cmds: List[DrawCmd] = []
    kept_boxes: List[BBox] = []
    for isl in orig_isls:
        m_isl = rasterize_cmds(isl.cmds, canvas_size=S, padding=96.0)
        if not (m_isl & m_edit_bin).any():
            kept_cmds.extend(isl.cmds)
            kept_boxes.append(isl.bbox)

    raw_blend = sdf_to_cff_commands(blended_sdf, synth.upem_spec, upsample_factor=upsample_factor)
    if not raw_blend:
        return kept_cmds, {"snapped_contours": 0, "reverted_contours": 0}

    blend_isls = cmds_to_islands(raw_blend) if kept_boxes else []
    if kept_boxes and blend_isls:
        edited_raw: List[DrawCmd] = []
        for b_isl in blend_isls:
            m_b = rasterize_cmds(b_isl.cmds, canvas_size=S, padding=96.0)
            if (m_b & m_edit_bin).any():
                edited_raw.extend(b_isl.cmds)
    else:
        kept_cmds = []
        kept_boxes = []
        edited_raw = raw_blend

    if not edited_raw:
        return kept_cmds, {"snapped_contours": 0, "reverted_contours": 0}

    opt_edited, opt_stats = optimize_synthesized_contours_v3(
        edited_raw,
        ref_sc_cmds=sc_cmds,
        stem_concave_sagitta=0.85,
        max_hausdorff_upem=4.0,
        max_area_rel=0.015,
    )
    if not kept_cmds:
        return opt_edited, opt_stats

    ed_bb = cmds_to_bbox(opt_edited)
    if ed_bb is not None and any(_boxes_overlap(kb, ed_bb, min_margin=0.0) for kb in kept_boxes):
        return boolean_union_cmds([kept_cmds, opt_edited]), opt_stats
    return kept_cmds + opt_edited, opt_stats


def tier0_stage0b_local_inpaint_batch(
    chars: List[str],
    skip_ext: SkipProOutlineExtractor,
    sn_jp: SansLightExtractor,
    sn_sc: SansLightExtractor,
    synth: ZHSDFSynthesizerV3,
    gate: StructuralGateV3,
    exempt_boxes_map: Optional[Dict[str, List[BBox]]] = None,
    batch_size: int = 32,
    upsample_factor: int = 3,
) -> Tuple[Dict[str, Tuple[List[DrawCmd], str, dict]], Dict[str, dict]]:
    """
    Stage 0b Batched GPU Masked Local SDF Inpainting (`Plan3.0.md §2.5`):
      Blends `(1 - M) * orig_sdf + M * pred_v3_sdf` locally around `D_sc ∪ D_jp`, preserving
      all untouched original `SkipPro-B` islands verbatim!
    """
    accepted: Dict[str, Tuple[List[DrawCmd], str, dict]] = {}
    failed: Dict[str, dict] = {}
    if not chars:
        return accepted, failed

    items: List[Tuple[str, List[DrawCmd], List[DrawCmd], List[DrawCmd], List[ContourIsland]]] = []
    for ch in chars:
        gate.get_ref_bundle(ch)
        orig = skip_ext.get_cmds(ch)
        sc_raw = sn_sc.get_cmds(ch, unioned=False)
        jp_raw = sn_jp.get_cmds(ch, unioned=False)
        if not orig or not sc_raw or not jp_raw:
            failed[ch] = {"passed": False, "reasons": ["missing_outlines"]}
            continue
        sc_al, jp_al = align_pair_via_jp_to_orig(sc_raw, jp_raw, orig)
        orig_isls = cmds_to_islands(orig)
        items.append((ch, orig, sc_al, jp_al, orig_isls))

    if not items:
        return accepted, failed

    pred_sdfs = synth.predict_sdf_batch([it[2] for it in items], dw=0.0, r_u=40.0, batch_size=batch_size)
    S = synth.canvas_size
    s_hi = S * 2
    workers = min(8, max(2, (os.cpu_count() or 4)))

    def _inpaint_one(payload: Tuple[tuple, np.ndarray]) -> Tuple[str, bool, Optional[List[DrawCmd]], str, dict]:
        (ch, orig_c, sc_al, jp_al, orig_isls), pred_sdf = payload
        ex_boxes = (exempt_boxes_map or {}).get(ch) or None
        m_orig = rasterize_cmds(orig_c, canvas_size=s_hi, padding=96.0)
        orig_sdf = mask_to_normalized_sdf(m_orig, S, trunc_px=synth.trunc_px)
        m1, m2, m3 = _build_local_inpaint_masks(sc_al, jp_al, orig_c, ex_boxes, canvas_size=S, padding=96.0)

        last_cmds: Optional[List[DrawCmd]] = None
        last_meta: dict = {}
        for tag, mask_m in (
            ("Tier0_LocalInpaint", m1),
            ("Tier0_RegionInpaint", m2),
            ("Tier0_WideLocalInpaint", m3),
        ):
            blended = (1.0 - mask_m) * orig_sdf + mask_m * pred_sdf
            cand, opt_st = _assemble_blended_islands(blended, mask_m, orig_isls, sc_al, synth, upsample_factor=upsample_factor)
            ok, gm = gate.check(cand, ch, exempt_boxes=ex_boxes, check_jp_diff=True)
            last_cmds = cand
            last_meta = {**gm, **opt_st}
            if ok:
                return ch, True, cand, tag, last_meta

        # Pass 4: Whole-island inpainting preserving untouched native islands & exempt boxes verbatim
        m4 = np.ones((S, S), dtype=np.float32)
        if ex_boxes:
            for box in ex_boxes:
                m4[upem_box_to_mask(box, canvas_size=S, padding=96.0, margin_u=16.0)] = 0.0
        m2_edit = m2 >= 0.08
        upem_px = (1000.0 + 2.0 * 96.0) / float(S)
        r_isl = max(1, int(round(8.0 / upem_px)))
        k_isl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_isl + 1, 2 * r_isl + 1))
        for isl in orig_isls:
            m_isl = rasterize_cmds(isl.cmds, canvas_size=S, padding=96.0)
            if not (m_isl & m2_edit).any():
                m_dil = cv2.dilate(m_isl.astype(np.uint8) * 255, k_isl) > 0
                m4[m_dil] = 0.0
        if (m4 < 0.25).any():
            m4 = np.clip(cv2.GaussianBlur(m4, (0, 0), sigmaX=2.5), 0.0, 1.0).astype(np.float32)
            blended4 = (1.0 - m4) * orig_sdf + m4 * pred_sdf
            cand4, opt_st4 = _assemble_blended_islands(blended4, m4, orig_isls, sc_al, synth, upsample_factor=upsample_factor)
            ok4, gm4 = gate.check(cand4, ch, exempt_boxes=ex_boxes, check_jp_diff=True)
            last_cmds = cand4
            last_meta = {**gm4, **opt_st4}
            if ok4:
                return ch, True, cand4, "Tier0_IslandInpaint", last_meta

        return ch, False, last_cmds, "Tier0_Failed", last_meta

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ch, ok, cmds, tag, meta in ex.map(_inpaint_one, zip(items, pred_sdfs)):
            if ok and cmds:
                accepted[ch] = (cmds, tag, meta)
            else:
                meta["fallback_cmds"] = cmds or []
                failed[ch] = meta

    return accepted, failed
