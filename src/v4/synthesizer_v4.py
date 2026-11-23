"""
v4.0 Global-Context SDF Synthesizer — drop-in replacement for ZHSDFSynthesizerV3.

Uses GlobalGlyphTransformer (CNN + Transformer + CNN) for Tier 3 SDF generation.
Same interface: predict_sdf_batch + synthesize_with_gate_retries.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as edt
from skimage.morphology import skeletonize

from src.glyph_renderer import CanvasSpec
from src.sdf_utils import mask_to_normalized_sdf
from src.v2.curve_optimizer import count_outline_stats
from src.v2.hb_outlines import BBox, DrawCmd, rasterize_cmds
from src.v2.stroke_assembler import _select_orig_islands_in_box
from src.v3.curve_constrain import optimize_synthesized_contours_v3
from src.v3.local_inpainter import _assemble_blended_islands, _build_local_inpaint_masks, align_pair_via_jp_to_orig
from src.v3.structural_gate import StructuralGateV3
from src.v4.dataset_v4 import build_deskinned_channels, pack_input_tensor
from src.v4.model_v4 import GlobalGlyphTransformer
from src.vectorizer import sdf_to_cff_commands


class ZHSDFSynthesizerV4:
    """
    Global-Context Neural SDF Fallback Generator (v4.0):
      - Drives GlobalGlyphTransformer with deskinned Light channels.
      - Uses curvature-masked, Hausdorff-guarded optimize_synthesized_contours_v3.
      - Enforces StructuralGateV3 with up to 3 jittered (dw, r_u) retries.
    """

    def __init__(
        self,
        ckpt_path: Path,
        canvas_size: int = 256,
        d_model: int = 768,
        n_transformer_layers: int = 4,
        n_heads: int = 12,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canvas_size = canvas_size
        self.trunc_px = 16.0 * (canvas_size / 256.0)

        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        # Auto-detect v4.0 (legacy stride-2 up0) vs v4.1 (128x128 skip up0) from checkpoint keys
        decoder_version = "v40" if "up0.0.weight" in state else "v41"

        self.model = GlobalGlyphTransformer(
            in_channels=4, canvas_size=canvas_size,
            d_model=d_model, n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            decoder_version=decoder_version,
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.upem_spec = CanvasSpec(
            canvas_size=canvas_size,
            em_ratio=1000.0 / 1192.0,
            upem=1000,
            cjk_ascender=880,
        )
        self.n_params = self.model.count_parameters()
        print(f"[v4-Synth] Loaded GlobalGlyphTransformer ({decoder_version}): {self.n_params:,} params ({self.n_params/1e6:.1f}M)", flush=True)

    def predict_sdf_batch(
        self,
        cmds_list: List[List[DrawCmd]],
        dw: float = 0.0,
        r_u: float = 40.0,
        batch_size: int = 8,
    ) -> List[np.ndarray]:
        """Predict [canvas_size, canvas_size] float32 SDF arrays for aligned Light contours."""
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
            return [], [], {"snapped_contours": 0, "reverted_contours": 0}
        clean_cmds, opt_stats = optimize_synthesized_contours_v3(
            raw_cmds,
            ref_sc_cmds=ref_sc_cmds,
            stem_concave_sagitta=0.85,
            max_hausdorff_upem=4.5,
            max_area_rel=0.018,
        )
        return clean_cmds, raw_cmds, opt_stats

    def synthesize_with_gate_retries(
        self,
        chars: List[str],
        sn_sc_cmds_map: Dict[str, List[DrawCmd]],
        gate: StructuralGateV3,
        exempt_boxes_map: Optional[Dict[str, List[BBox]]] = None,
        batch_size: int = 8,
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
                clean_cmds, raw_cmds, opt_st = self.vectorize_and_constrain(sdf_arr, sc_c, upsample_factor=upsample_factor)
                ex_boxes = (exempt_boxes_map or {}).get(ch) or None
                ok, gm = gate.check(clean_cmds, ch, exempt_boxes=ex_boxes, check_jp_diff=check_jp_diff)
                if ok:
                    return ch, True, clean_cmds, {**gm, **opt_st, "attempt": att_idx}
                ok_raw, gm_raw = gate.check(raw_cmds, ch, exempt_boxes=ex_boxes, check_jp_diff=check_jp_diff)
                if ok_raw:
                    return ch, True, raw_cmds, {**gm_raw, "snapped_contours": 0, "reverted_contours": 0, "attempt": att_idx}
                return ch, False, clean_cmds, {**gm, **opt_st, "attempt": att_idx}

            next_remaining: List[str] = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for ch, ok, cmds, meta in ex.map(_eval_one, zip(remaining, preds)):
                    if ok:
                        accepted[ch] = (cmds, f"Tier3_GlobalSDF(att={att_idx})", meta)
                        last_diag.pop(ch, None)
                    else:
                        meta["fallback_cmds"] = cmds
                        last_diag[ch] = meta
                        next_remaining.append(ch)
            remaining = next_remaining

        return accepted, last_diag
