"""
v4.5 Global-Context Direct-SDF Synthesizer (`ZHSDFSynthesizerV45`).

Fixes over v4.0-v4.3 `ZHSDFSynthesizerV4` (from 审计v4.md §2.3 #4, #6 & Phase A #2):
  1. Uses `GlobalGlyphTransformerV45` (direct SDF head, max_shift_px=14.0).
  2. Uses `optimize_synthesized_contours_v45` (segment-level snapping + orphan/floating speckle filter).
  3. Fallback `ok_raw` path always passes through `smooth_unconstrained_cmds_v45` + orphan filter
     so unsmoothed jagged polygons or floating specks never enter the final OTF.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.glyph_renderer import CanvasSpec
from src.v2.hb_outlines import BBox, DrawCmd
from src.v3.structural_gate import StructuralGateV3
from src.v45.curve_constrain_v45 import (
    filter_orphan_islands_v45,
    optimize_synthesized_contours_v45,
    smooth_unconstrained_cmds_v45,
)
from src.v45.dataset_v45 import build_deskinned_channels, pack_input_tensor
from src.v45.model_v45 import GlobalGlyphTransformerV45
from src.vectorizer import sdf_to_cff_commands


class ZHSDFSynthesizerV45:
    """Global-Context Direct-SDF Generator (v4.5) with orphan speckle filtering & segment-level optimization."""

    def __init__(
        self,
        ckpt_path: Path,
        canvas_size: int = 256,
        d_model: int = 768,
        n_transformer_layers: int = 4,
        n_heads: int = 12,
        max_shift_px: float = 14.0,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canvas_size = canvas_size
        self.trunc_px = 16.0 * (canvas_size / 256.0)

        self.model = GlobalGlyphTransformerV45(
            in_channels=4,
            canvas_size=canvas_size,
            d_model=d_model,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            max_shift_px=max_shift_px,
        ).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        self.upem_spec = CanvasSpec(
            canvas_size=canvas_size,
            em_ratio=1000.0 / 1192.0,
            upem=1000,
            cjk_ascender=880,
        )
        self.n_params = self.model.count_parameters()
        print(f"[v45-Synth] Loaded GlobalGlyphTransformerV45: {self.n_params:,} params ({self.n_params/1e6:.1f}M)", flush=True)

    def predict_sdf_batch(
        self,
        cmds_list: List[List[DrawCmd]],
        dw: float = 0.0,
        r_u: float = 40.0,
        batch_size: int = 16,
    ) -> List[np.ndarray]:
        if not cmds_list:
            return []
        out: List[np.ndarray] = []
        amp_dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.no_grad():
            for st in range(0, len(cmds_list), batch_size):
                sub = cmds_list[st : st + batch_size]
                tensors = []
                for cmds in sub:
                    sl, dtu, c2, s2 = build_deskinned_channels(
                        cmds, canvas_size=self.canvas_size, padding=96.0, trunc_px=self.trunc_px,
                    )
                    tensors.append(
                        pack_input_tensor(
                            sl, dtu, c2, s2,
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
        raw_cmds = sdf_to_cff_commands(sdf_2d, self.upem_spec, upsample_factor=upsample_factor)
        if not raw_cmds:
            return [], [], {"snapped_segs": 0, "curved_segs": 0, "reverted_contours": 0, "dropped_speckles": 0}
        clean_cmds, opt_stats = optimize_synthesized_contours_v45(
            raw_cmds,
            ref_sc_cmds=ref_sc_cmds,
            stem_concave_sagitta=0.85,
            max_seg_hausdorff_upem=5.8,
            max_area_rel=0.025,
        )
        # Smoothed + orphan-filtered fallback (never emits raw unsmoothed polygons)
        filtered_raw, _ = filter_orphan_islands_v45(raw_cmds, ref_sc_cmds)
        smooth_fallback = smooth_unconstrained_cmds_v45(filtered_raw)
        return clean_cmds, smooth_fallback, opt_stats

    def synthesize_with_gate_retries(
        self,
        chars: List[str],
        sn_sc_cmds_map: Dict[str, List[DrawCmd]],
        gate: StructuralGateV3,
        exempt_boxes_map: Optional[Dict[str, List[BBox]]] = None,
        batch_size: int = 16,
        upsample_factor: int = 3,
        check_jp_diff: bool = True,
    ) -> Tuple[Dict[str, Tuple[List[DrawCmd], str, dict]], Dict[str, dict]]:
        accepted: Dict[str, Tuple[List[DrawCmd], str, dict]] = {}
        last_diag: Dict[str, dict] = {}
        if not chars:
            return accepted, last_diag

        ref_cmds_map: Dict[str, List[DrawCmd]] = {}
        remaining: List[str] = []
        for ch in chars:
            sc_c = sn_sc_cmds_map.get(ch)
            if sc_c:
                ref_cmds_map[ch] = sc_c
                remaining.append(ch)
            else:
                last_diag[ch] = {"passed": False, "reasons": ["no_sc_ref"]}

        attempts = [(0.0, 40.0), (-0.04, 43.0), (0.05, 37.0)]
        workers = min(4, max(2, os.cpu_count() or 4))

        for att_idx, (dw, r_u) in enumerate(attempts):
            if not remaining:
                break
            preds = self.predict_sdf_batch([ref_cmds_map[c] for c in remaining], dw=dw, r_u=r_u, batch_size=batch_size)

            def _eval_one(item: Tuple[str, np.ndarray]) -> Tuple[str, bool, List[DrawCmd], dict]:
                ch, sdf_arr = item
                sc_c = ref_cmds_map[ch]
                clean_cmds, smooth_fb, opt_st = self.vectorize_and_constrain(sdf_arr, sc_c, upsample_factor=upsample_factor)
                ex_boxes = (exempt_boxes_map or {}).get(ch) or None
                ok, gm = gate.check(clean_cmds, ch, exempt_boxes=ex_boxes, check_jp_diff=check_jp_diff)
                if ok:
                    return ch, True, clean_cmds, {**gm, **opt_st, "attempt": att_idx}
                ok_fb, gm_fb = gate.check(smooth_fb, ch, exempt_boxes=ex_boxes, check_jp_diff=check_jp_diff)
                if ok_fb:
                    return ch, True, smooth_fb, {**gm_fb, **opt_st, "used_smooth_fallback": True, "attempt": att_idx}
                return ch, False, clean_cmds, {**gm, **opt_st, "attempt": att_idx}

            next_remaining: List[str] = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for ch, ok, cmds, meta in ex.map(_eval_one, zip(remaining, preds)):
                    if ok:
                        accepted[ch] = (cmds, f"Tier3_GlobalSDF_v45(att={att_idx})", meta)
                        last_diag.pop(ch, None)
                    else:
                        meta["fallback_cmds"] = cmds
                        last_diag[ch] = meta
                        next_remaining.append(ch)
            remaining = next_remaining

        return accepted, last_diag
