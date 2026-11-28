"""
v4.6 Global-Context + Local-Window Direct-SDF Synthesizer (`ZHSDFSynthesizerV46`).

Differs from `ZHSDFSynthesizerV45` in three ways (per 审计v46_进度.md §3 & R5):
  1. Loads `GlobalGlyphTransformerV46` (5-channel: + radical_mask, 32x32 local-window
     attention, heteroscedastic uncertainty head, MC-dropout).
  2. `predict_sdf_batch` builds the 5th conditioning channel (radical_mask) per character
     via `build_radical_mask_channel` (needs the IDS map).
  3. Uses the new single-pass v4.6 vectorizer `sdf_to_cmds_v46` (sub-pixel iso-contours,
     corner detection, G1 tangent-constrained cubic fit, axis snapping, fillet absorb,
     integer coords + nesting depth) instead of the legacy `sdf_to_cff_commands` +
     `optimize_synthesized_contours_v45` double-lossy chain.

The `synthesize_with_gate_retries` contract is kept identical to v4.5 so the full
TG-8105 build script works unchanged.
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
from src.v45.dataset_v45 import build_deskinned_channels, pack_input_tensor
from src.v46.dataset_v46 import build_radical_mask_channel
from src.v46.glyph_qa_v46 import GlyphQAGateV46, repair_phantom_islands_v46
from src.v46.model_v46 import GlobalGlyphTransformerV46
from src.v46.vectorize_v46 import VecParams, sdf_to_cmds_v46


class ZHSDFSynthesizerV46:
    """Global-Context + Local-Window Direct-SDF Generator (v4.6) with the new single-pass vectorizer."""

    def __init__(
        self,
        ckpt_path: Path,
        canvas_size: int = 256,
        d_model: int = 768,
        n_transformer_layers: int = 4,
        n_heads: int = 12,
        max_shift_px: float = 14.0,
        use_radical_mask: bool = True,
        use_local_attn_32: bool = True,
        mc_dropout: float = 0.08,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canvas_size = canvas_size
        self.trunc_px = 16.0 * (canvas_size / 256.0)
        self.use_radical_mask = use_radical_mask

        in_ch = 5 if use_radical_mask else 4
        self.model = GlobalGlyphTransformerV46(
            in_channels=in_ch,
            use_radical_mask=use_radical_mask,
            canvas_size=canvas_size,
            d_model=d_model,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=0.1,
            mc_dropout=mc_dropout,
            max_shift_px=max_shift_px,
            use_local_attn_32=use_local_attn_32,
        ).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"v46 checkpoint missing {len(missing)} tensors: {missing[:8]}")
        self.model.eval()

        self.upem_spec = CanvasSpec(
            canvas_size=canvas_size,
            em_ratio=1000.0 / 1192.0,
            upem=1000,
            cjk_ascender=880,
        )
        self.vec_params = VecParams()
        self.n_params = self.model.count_parameters()
        print(
            f"[v46-Synth] Loaded GlobalGlyphTransformerV46: {self.n_params:,} params "
            f"({self.n_params/1e6:.1f}M) | in_ch={in_ch} radical_mask={use_radical_mask} "
            f"local_attn_32={use_local_attn_32}",
            flush=True,
        )

    def predict_sdf_batch(
        self,
        cmds_list: List[List[DrawCmd]],
        chars: Optional[List[str]] = None,
        ids_map: Optional[Dict[str, str]] = None,
        dw: float = 0.0,
        r_u: float = 40.0,
        batch_size: int = 16,
    ) -> List[np.ndarray]:
        """Predict SDF for a batch of Noto SC reference cmds.

        `chars` + `ids_map` are required when `use_radical_mask` is on (to build the 5th
        conditioning channel). If omitted, the 5th channel is filled with zeros.
        """
        if not cmds_list:
            return []
        out: List[np.ndarray] = []
        ids_map = ids_map or {}
        amp_dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.no_grad():
            for st in range(0, len(cmds_list), batch_size):
                sub = cmds_list[st : st + batch_size]
                sub_chars = chars[st : st + batch_size] if chars else [None] * len(sub)
                tensors = []
                for i, cmds in enumerate(sub):
                    sl, dtu, c2, s2 = build_deskinned_channels(
                        cmds, canvas_size=self.canvas_size, padding=96.0, trunc_px=self.trunc_px,
                    )
                    x4 = pack_input_tensor(
                        sl, dtu, c2, s2,
                        uniform_half_width_u=r_u,
                        canvas_size=self.canvas_size,
                        padding=96.0,
                        trunc_px=self.trunc_px,
                        sdf_shift=dw,
                    )
                    if self.use_radical_mask:
                        ch = sub_chars[i]
                        # Keep in [0.0, 1.0] to match DualGlyphDatasetV46 & NeuralSDFSamplerV46 exactly
                        rmask = build_radical_mask_channel(ch or "", ids_map, self.canvas_size)
                        x5 = np.concatenate([x4, rmask[None, :, :]], axis=0)
                    else:
                        x5 = x4
                    tensors.append(x5)
                xt = torch.from_numpy(np.stack(tensors, axis=0)).to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=self.device.type == "cuda"):
                    pred_t, _ = self.model(xt, return_uncertainty=False)
                pred_np = pred_t[:, 0].float().cpu().numpy()
                for i in range(len(sub)):
                    out.append(pred_np[i])
        return out

    def vectorize_and_constrain(
        self,
        sdf_2d: np.ndarray,
        ref_sc_cmds: Optional[List[DrawCmd]],
        upsample_factor: int = 3,
        repair_phantoms: bool = True,
    ) -> Tuple[List[DrawCmd], List[DrawCmd], dict]:
        """Single-pass v4.6 vectorization + optional 384px phantom-island repair.

        Returns `(clean_cmds, fallback_cmds, stats)` with accurate `axis_snapped`, `lines`,
        `refits`, `dropped_speckles`, and `repaired_phantoms` telemetry.
        """
        raw_cmds, stats = sdf_to_cmds_v46(sdf_2d, self.vec_params)
        if not raw_cmds:
            return [], [], {
                "snapped_segs": 0,
                "line_segs": 0,
                "curved_segs": 0,
                "reverted_contours": 0,
                "dropped_speckles": 0,
                "repaired_phantoms": 0,
                "extra_large_strokes": 0,
            }
        if repair_phantoms and ref_sc_cmds:
            clean_cmds, n_rep, n_extra = repair_phantom_islands_v46(raw_cmds, ref_sc_cmds)
        else:
            clean_cmds, n_rep, n_extra = raw_cmds, 0, 0

        dropped_vec = int(stats.get("dropped_specks", 0)) + int(stats.get("dropped_pinholes", 0))
        return clean_cmds, list(clean_cmds), {
            "snapped_segs": int(stats.get("axis_snapped", 0)),
            "line_segs": int(stats.get("lines", 0)),
            "curved_segs": int(stats.get("curves", 0)),
            "reverted_contours": int(stats.get("refits", 0)),
            "dropped_speckles": dropped_vec + int(n_rep),
            "repaired_phantoms": int(n_rep),
            "extra_large_strokes": int(n_extra),
            "v46_contours": int(stats.get("contours", 0)),
            "v46_max_dev_u": float(stats.get("max_dev_u", 0.0)),
            "v46_p95_dev_u": float(stats.get("p95_dev_u", 0.0)),
        }

    def synthesize_with_gate_retries(
        self,
        chars: List[str],
        sn_sc_cmds_map: Dict[str, List[DrawCmd]],
        gate: StructuralGateV3,
        ids_map: Optional[Dict[str, str]] = None,
        exempt_boxes_map: Optional[Dict[str, List[BBox]]] = None,
        batch_size: int = 16,
        upsample_factor: int = 3,
        check_jp_diff: bool = True,
        qa_gate: Optional[GlyphQAGateV46] = None,
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
            preds = self.predict_sdf_batch(
                [ref_cmds_map[c] for c in remaining],
                chars=remaining,
                ids_map=ids_map,
                dw=dw,
                r_u=r_u,
                batch_size=batch_size,
            )

            def _eval_one(item: Tuple[str, np.ndarray]) -> Tuple[str, bool, List[DrawCmd], dict]:
                ch, sdf_arr = item
                sc_c = ref_cmds_map[ch]
                ex_boxes = (exempt_boxes_map or {}).get(ch) or None
                if qa_gate is not None:
                    # QA gate performs phantom repair + StructuralGateV45 + 384px detail checks in one pass
                    raw_cmds, _, opt_st = self.vectorize_and_constrain(
                        sdf_arr, sc_c, upsample_factor=upsample_factor, repair_phantoms=False
                    )
                    qa_res = qa_gate.evaluate(
                        raw_cmds,
                        ch,
                        sc_c,
                        sdf_2d=sdf_arr,
                        exempt_boxes=ex_boxes,
                        check_jp_diff=check_jp_diff,
                        repair_phantoms=True,
                    )
                    opt_st["repaired_phantoms"] = int(qa_res.repaired_phantoms)
                    opt_st["dropped_speckles"] = int(opt_st["dropped_speckles"]) + int(qa_res.repaired_phantoms)
                    diag = {
                        **qa_res.metrics,
                        **opt_st,
                        "passed": qa_res.passed,
                        "qa_score": qa_res.score,
                        "reasons": qa_res.hard_reasons,
                        "attempt": att_idx,
                    }
                    return ch, qa_res.passed, qa_res.repaired_cmds, diag

                clean_cmds, _, opt_st = self.vectorize_and_constrain(
                    sdf_arr, sc_c, upsample_factor=upsample_factor, repair_phantoms=True
                )
                if int(opt_st.get("extra_large_strokes", 0)) > 0:
                    return ch, False, clean_cmds, {
                        "passed": False,
                        "reasons": [f"extra_stroke={opt_st['extra_large_strokes']}"],
                        **opt_st,
                        "attempt": att_idx,
                    }
                ok, gm = gate.check(clean_cmds, ch, exempt_boxes=ex_boxes, check_jp_diff=check_jp_diff)
                return ch, ok, clean_cmds, {**gm, **opt_st, "attempt": att_idx}

            next_remaining: List[str] = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for ch, ok, cmds, meta in ex.map(_eval_one, zip(remaining, preds)):
                    if ok:
                        accepted[ch] = (cmds, f"Tier3_GlobalSDF_v46(att={att_idx})", meta)
                        last_diag.pop(ch, None)
                    else:
                        meta["fallback_cmds"] = cmds
                        prev_diag = last_diag.get(ch)
                        # Retain the highest-scoring failed candidate for manual review OTF
                        if prev_diag is None or float(meta.get("qa_score", -999.0)) >= float(prev_diag.get("qa_score", -999.0)):
                            last_diag[ch] = meta
                        next_remaining.append(ch)
            remaining = next_remaining

        return accepted, last_diag
