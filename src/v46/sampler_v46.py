"""
v4.6 Multi-Sample ("抽卡") Candidate Pool, Consensus Scoring & Selection Engine (`src/v46/sampler_v46.py`).

Implements 审计v46_进度.md §5.2:
  1. Unified candidate pool per character:
       - Existing OTF glyphs (as-is + optional v4.6 re-vectorization via `cmds_to_polylines` -> `polylines_to_cmds`)
       - Neural SDF samples (supports both `GlobalGlyphTransformerV45` and `GlobalGlyphTransformerV46`):
           * (dw, r_u) jitter x5
           * +/-1 px input translation TTA x3 (shifted back on output SDF)
           * MC-dropout xK
           * Consensus SDF (mean of all neural SDF predictions)
  2. Majority-vote consensus raster & per-candidate `consensus_iou`.
  3. Per-character instability (pairwise raster IoU variance across samples) + IDS radical-family aggregation.
  4. Repair -> Hard Gate -> Soft Score selection via `GlyphQAGateV46`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.v2.hb_outlines import BBox, DrawCmd, rasterize_cmds
from src.v46.glyph_qa_v46 import GlyphQAGateV46, QAResult
from src.v46.vectorize_v46 import VecParams, cmds_to_polylines, polylines_to_cmds, sdf_to_cmds_v46

DEFAULT_JITTER_PAIRS: Tuple[Tuple[float, float], ...] = (
    (0.0, 40.0),
    (-0.04, 43.0),
    (0.04, 37.0),
    (-0.02, 38.0),
    (0.02, 42.0),
)

DEFAULT_TTA_SHIFTS: Tuple[Tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
)


@dataclass
class CandidateItem:
    source: str
    cmds: List[DrawCmd]
    sdf_2d: Optional[np.ndarray] = None
    logvar_2d: Optional[np.ndarray] = None
    qa: Optional[QAResult] = None


@dataclass
class SelectionOutcome:
    char: str
    selected_source: str
    gate_passed: bool
    score: float
    selected_cmds: List[DrawCmd]
    selected_sdf: Optional[np.ndarray]
    instability_var: float
    mean_pairwise_iou: float
    candidates: List[Dict[str, object]] = field(default_factory=list)


def compute_consensus_and_instability(
    masks: Sequence[np.ndarray],
) -> Tuple[List[float], float, float]:
    """
    Given a list of boolean 2D candidate rasters:
      - builds the pixel-wise majority-vote mask (`votes >= ceil(N/2)`),
      - returns `(per_candidate_consensus_iou, pairwise_iou_variance, mean_pairwise_iou)`.
    """
    n = len(masks)
    if n == 0:
        return [], 0.0, 1.0
    if n == 1:
        return [1.0], 0.0, 1.0

    stack = np.stack([m.astype(np.uint8) for m in masks], axis=0)  # [N, H, W]
    thresh = (n + 1) // 2
    consensus = stack.sum(axis=0) >= thresh

    cons_ious: List[float] = []
    for i in range(n):
        mi = stack[i] > 0
        inter = float((mi & consensus).sum())
        union = float((mi | consensus).sum())
        cons_ious.append(inter / max(union, 1.0))

    pair_ious: List[float] = []
    for i in range(n):
        mi = stack[i] > 0
        for j in range(i + 1, n):
            mj = stack[j] > 0
            inter = float((mi & mj).sum())
            union = float((mi | mj).sum())
            pair_ious.append(inter / max(union, 1.0))

    if pair_ious:
        arr = np.array(pair_ious, dtype=np.float64)
        return cons_ious, float(np.var(arr)), float(np.mean(arr))
    return cons_ious, 0.0, 1.0


class NeuralSDFSamplerV46:
    """
    Lazy-torch neural SDF sampler supporting both `GlobalGlyphTransformerV45` and
    `GlobalGlyphTransformerV46` checkpoints.
    """

    def __init__(
        self,
        ckpt_path: Path,
        *,
        canvas_size: int = 256,
        max_shift_px: float = 14.0,
        ids_map: Optional[Dict[str, str]] = None,
    ) -> None:
        import torch
        from src.v45.model_v45 import GlobalGlyphTransformerV45
        from src.v46.model_v46 import GlobalGlyphTransformerV46

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canvas_size = canvas_size
        self.trunc_px = 16.0 * (canvas_size / 256.0)
        self.ids_map = ids_map or {}

        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        is_v46 = any(k.startswith(("uncertainty_head.", "local_attn_32.")) for k in state.keys())
        stem_w = state.get("stem.0.net.0.weight")
        in_ch = int(stem_w.shape[1]) if stem_w is not None else 4
        self.in_channels = in_ch
        self.is_v46 = is_v46

        if is_v46:
            has_local_attn = any(k.startswith("local_attn_32.") for k in state.keys())
            self.model = GlobalGlyphTransformerV46(
                in_channels=in_ch,
                use_radical_mask=(in_ch >= 5),
                canvas_size=canvas_size,
                max_shift_px=max_shift_px,
                use_local_attn_32=has_local_attn,
            ).to(self.device)
        else:
            self.model = GlobalGlyphTransformerV45(
                in_channels=in_ch,
                canvas_size=canvas_size,
                max_shift_px=max_shift_px,
            ).to(self.device)

        self.model.load_state_dict(state)
        self.model.eval()

    def _run_batch(
        self,
        xs: List[np.ndarray],
        mc_dropout: bool = False,
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        import torch

        if not xs:
            return [], []
        if mc_dropout and hasattr(self.model, "set_mc_dropout"):
            self.model.set_mc_dropout(True)
        elif mc_dropout:
            for m in self.model.modules():
                if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.MultiheadAttention)):
                    m.train(True)

        amp_dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        xt = torch.from_numpy(np.stack(xs, axis=0)).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=(self.device.type == "cuda")):
                if self.is_v46:
                    pred_t, _, lv_t = self.model(xt, return_uncertainty=True)
                    lv_np = lv_t[:, 0].float().cpu().numpy()
                else:
                    pred_t, _ = self.model(xt)
                    lv_np = None
        pred_np = pred_t[:, 0].float().cpu().numpy()

        if mc_dropout:
            self.model.eval()

        sdfs = [pred_np[i] for i in range(len(xs))]
        lvs = [lv_np[i] if lv_np is not None else None for i in range(len(xs))]
        return sdfs, lvs

    @staticmethod
    def _shift_2d(arr: np.ndarray, dx: int, dy: int, fill: float) -> np.ndarray:
        out = np.full_like(arr, fill)
        h, w = arr.shape[-2], arr.shape[-1]
        src_x0, src_x1 = max(0, -dx), min(w, w - dx)
        dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
        src_y0, src_y1 = max(0, -dy), min(h, h - dy)
        dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
        out[..., dst_y0:dst_y1, dst_x0:dst_x1] = arr[..., src_y0:src_y1, src_x0:src_x1]
        return out

    def sample_glyph(
        self,
        ch: str,
        ref_sc_cmds: List[DrawCmd],
        *,
        jitter_pairs: Sequence[Tuple[float, float]] = DEFAULT_JITTER_PAIRS,
        tta_shifts: Sequence[Tuple[int, int]] = DEFAULT_TTA_SHIFTS,
        mc_samples: int = 3,
        vec_params: Optional[VecParams] = None,
        tag_prefix: str = "sdf",
    ) -> List[CandidateItem]:
        from src.v45.dataset_v45 import build_deskinned_channels, pack_input_tensor
        from src.v46.dataset_v46 import build_radical_mask_channel

        sl, dtu, c2, s2 = build_deskinned_channels(
            ref_sc_cmds, canvas_size=self.canvas_size, padding=96.0, trunc_px=self.trunc_px
        )
        rmask = (
            build_radical_mask_channel(ch, self.ids_map, self.canvas_size)[None, :, :]
            if self.in_channels >= 5
            else None
        )

        def _pack(dw: float, r_u: float) -> np.ndarray:
            x4 = pack_input_tensor(
                sl, dtu, c2, s2,
                uniform_half_width_u=r_u,
                canvas_size=self.canvas_size,
                padding=96.0,
                trunc_px=self.trunc_px,
                sdf_shift=dw,
            )
            return np.concatenate([x4, rmask], axis=0) if rmask is not None else x4

        # 1. Deterministic jitter + TTA batch
        det_xs: List[np.ndarray] = []
        det_meta: List[Tuple[str, int, int]] = []
        for idx, (dw, r_u) in enumerate(jitter_pairs):
            det_xs.append(_pack(dw, r_u))
            det_meta.append((f"{tag_prefix}_jit{idx}(dw={dw:+.2f},r={r_u:.0f})", 0, 0))

        base_x = det_xs[0]
        for dx, dy in tta_shifts:
            shifted = base_x.copy()
            shifted[0:2] = self._shift_2d(base_x[0:2], dx, dy, fill=-1.0)
            shifted[2:] = self._shift_2d(base_x[2:], dx, dy, fill=0.0)
            det_xs.append(shifted)
            det_meta.append((f"{tag_prefix}_tta({dx:+d},{dy:+d})", dx, dy))

        det_sdfs, det_lvs = self._run_batch(det_xs, mc_dropout=False)

        all_sdfs: List[np.ndarray] = []
        all_lvs: List[Optional[np.ndarray]] = []
        all_names: List[str] = []

        for (name, dx, dy), s_arr, lv_arr in zip(det_meta, det_sdfs, det_lvs):
            if dx != 0 or dy != 0:
                s_arr = self._shift_2d(s_arr, -dx, -dy, fill=-1.0)
                if lv_arr is not None:
                    lv_arr = self._shift_2d(lv_arr, -dx, -dy, fill=0.0)
            all_sdfs.append(s_arr)
            all_lvs.append(lv_arr)
            all_names.append(name)

        # 2. MC-dropout batch
        if mc_samples > 0:
            mc_xs = [base_x for _ in range(mc_samples)]
            mc_sdfs, mc_lvs = self._run_batch(mc_xs, mc_dropout=True)
            for k, (s_arr, lv_arr) in enumerate(zip(mc_sdfs, mc_lvs)):
                all_sdfs.append(s_arr)
                all_lvs.append(lv_arr)
                all_names.append(f"{tag_prefix}_mc{k}")

        # 3. Consensus SDF (pixel-wise mean across all samples)
        if len(all_sdfs) >= 2:
            cons_sdf = np.mean(np.stack(all_sdfs, axis=0), axis=0).astype(np.float32)
            valid_lvs = [lv for lv in all_lvs if lv is not None]
            cons_lv = np.mean(np.stack(valid_lvs, axis=0), axis=0).astype(np.float32) if valid_lvs else None
            all_sdfs.append(cons_sdf)
            all_lvs.append(cons_lv)
            all_names.append(f"{tag_prefix}_consensus")

        # 4. Vectorize each SDF with v4.6 single-pass vectorizer
        cands: List[CandidateItem] = []
        for name, s_arr, lv_arr in zip(all_names, all_sdfs, all_lvs):
            cmds, _ = sdf_to_cmds_v46(s_arr, vec_params)
            if cmds:
                cands.append(CandidateItem(source=name, cmds=cmds, sdf_2d=s_arr, logvar_2d=lv_arr))
        return cands


def select_best_candidate_v46(
    ch: str,
    ref_sc_cmds: List[DrawCmd],
    candidates: List[CandidateItem],
    qa_gate: GlyphQAGateV46,
    *,
    exempt_boxes: Optional[Sequence[BBox]] = None,
    check_jp_diff: bool = True,
    revec_otf_candidates: bool = False,
    vec_params: Optional[VecParams] = None,
) -> SelectionOutcome:
    """
    Evaluate all candidates for `ch`:
      1. Optionally add v4.6 re-vectorized copies of non-SDF (OTF) candidates.
      2. Compute majority-vote consensus IoU & pairwise instability across the pool.
      3. Run `GlyphQAGateV46` (phantom repair -> hard gate -> soft score) on each candidate.
      4. Select highest-scoring passing candidate (or best failing fallback).
    """
    pool: List[CandidateItem] = list(candidates)
    if revec_otf_candidates:
        for c in list(candidates):
            if c.sdf_2d is None and c.cmds:
                polys = cmds_to_polylines(c.cmds, step=1.5)
                if polys:
                    re_cmds, _ = polylines_to_cmds(polys, vec_params)
                    if re_cmds:
                        pool.append(CandidateItem(source=f"{c.source}_revec46", cmds=re_cmds))

    if not pool:
        return SelectionOutcome(
            char=ch,
            selected_source="none",
            gate_passed=False,
            score=-100.0,
            selected_cmds=[],
            selected_sdf=None,
            instability_var=0.0,
            mean_pairwise_iou=0.0,
            candidates=[],
        )

    # Rasterize all candidates at 256 px for consensus IoU & instability
    masks = [rasterize_cmds(c.cmds, canvas_size=256, padding=96.0) for c in pool]
    cons_ious, instab_var, mean_pw_iou = compute_consensus_and_instability(masks)

    scored: List[Tuple[CandidateItem, QAResult]] = []
    cand_records: List[Dict[str, object]] = []
    for item, c_iou in zip(pool, cons_ious):
        qa = qa_gate.evaluate(
            item.cmds,
            ch,
            ref_sc_cmds,
            sdf_2d=item.sdf_2d,
            pred_logvar_2d=item.logvar_2d,
            consensus_iou=c_iou,
            exempt_boxes=exempt_boxes,
            check_jp_diff=check_jp_diff,
            repair_phantoms=True,
        )
        item.qa = qa
        scored.append((item, qa))
        cand_records.append(
            {
                "source": item.source,
                "passed": qa.passed,
                "score": qa.score,
                "hard_reasons": qa.hard_reasons,
                "repaired_phantoms": qa.repaired_phantoms,
                "metrics": qa.metrics,
            }
        )

    # Prefer passing candidates, ranked by highest soft score
    passing = [(it, qa) for it, qa in scored if qa.passed]
    ranked = sorted(passing if passing else scored, key=lambda t: t[1].score, reverse=True)
    best_item, best_qa = ranked[0]

    return SelectionOutcome(
        char=ch,
        selected_source=best_item.source,
        gate_passed=best_qa.passed,
        score=best_qa.score,
        selected_cmds=best_qa.repaired_cmds,
        selected_sdf=best_item.sdf_2d,
        instability_var=round(instab_var, 5),
        mean_pairwise_iou=round(mean_pw_iou, 4),
        candidates=cand_records,
    )


def summarize_instability_by_radical(
    outcomes: Sequence[SelectionOutcome],
    ids_map: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Aggregate pass rate, score, instability variance, and fog ratio by IDS radical family."""
    from src.v46.dataset_v46 import classify_ids_radical_group

    groups: Dict[str, List[SelectionOutcome]] = defaultdict(list)
    for oc in outcomes:
        grp = classify_ids_radical_group(oc.char, ids_map)
        groups[grp].append(oc)

    summary: Dict[str, Dict[str, float]] = {}
    for grp, items in sorted(groups.items()):
        n = len(items)
        n_pass = sum(1 for x in items if x.gate_passed)
        summary[grp] = {
            "n": n,
            "pass_rate": round(n_pass / max(n, 1), 4),
            "mean_score": round(float(np.mean([x.score for x in items])), 2),
            "mean_instability_var": round(float(np.mean([x.instability_var for x in items])), 5),
            "mean_pairwise_iou": round(float(np.mean([x.mean_pairwise_iou for x in items])), 4),
        }
    return summary
