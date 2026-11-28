#!/usr/bin/env python3
"""
v4.6 Multi-Sample ("抽卡") Selection & Self-Distillation Export CLI (`scripts/v46_select.py`).

Implements 审计v46_进度.md §5.2:
  - Pools candidates from existing OTFs (`--otf_src v44=... --otf_src v45=...`, optional `--revec_otf`)
    and/or neural checkpoints (`--ckpt checkpoints/v45_zh_sdf_global.pt`).
  - Runs phantom-island repair, 384 px `GlyphQAGateV46` hard gate + soft scoring + consensus IoU.
  - Emits:
      1. Per-character selection manifest JSON (`v46_selection_manifest.json`),
      2. IDS radical-family instability report JSON (`v46_instability_by_radical.json`),
      3. Optional high-scoring pseudo-label NPZ (`--export_pseudo_npz`) ready to feed directly
         into `scripts/v46_train.py --extra_npz` as the optional simplified-radical dataset!
      4. Optional standalone OTF (`--emit_otf`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sdf_utils import mask_to_normalized_sdf
from src.v2.component_bank import parse_cjkvi_ids
from src.v2.hb_outlines import DrawCmd, SkipProOutlineExtractor, load_tg8105_table, rasterize_cmds
from src.v3.hb_light import SansLightExtractor
from src.v3.otf_packager import build_standalone_cns_otf
from src.v45.dataset_v45 import build_deskinned_channels, pack_input_tensor
from src.v45.gate_v45 import StructuralGateV45
from src.v46.glyph_qa_v46 import GlyphQAGateV46, QAConfig
from src.v46.sampler_v46 import (
    CandidateItem,
    NeuralSDFSamplerV46,
    SelectionOutcome,
    select_best_candidate_v46,
    summarize_instability_by_radical,
)


def _parse_otf_sources(raw_specs: List[str]) -> List[Tuple[str, SkipProOutlineExtractor]]:
    out: List[Tuple[str, SkipProOutlineExtractor]] = []
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"Expected name=path.otf for --otf_src, got: {spec}")
        name, path_s = spec.split("=", 1)
        p = Path(path_s)
        if not p.exists():
            raise FileNotFoundError(f"OTF source not found: {p}")
        out.append((name.strip(), SkipProOutlineExtractor(p)))
    return out


def _select_target_chars(
    args: argparse.Namespace,
    ids_map: Dict[str, str],
) -> List[str]:
    if args.chars:
        return list(dict.fromkeys(ch for ch in args.chars if ch.strip()))
    tg_levels = load_tg8105_table(args.tg_dir)
    all_tg = sorted(tg_levels.keys())
    if args.radicals:
        rads = [r.strip() for r in args.radicals.split(",") if r.strip()]
        return [ch for ch in all_tg if any(r in ids_map.get(ch, "") for r in rads)]
    return all_tg


def _export_pseudo_npz(
    outcomes: List[SelectionOutcome],
    sc_cmds_map: Dict[str, List[DrawCmd]],
    out_npz: Path,
    *,
    min_score: float = 82.0,
    canvas_size: int = 256,
) -> int:
    """Export passing high-score glyphs to NPZ format consumable by `scripts/v46_train.py --extra_npz`."""
    chosen = [oc for oc in outcomes if oc.gate_passed and oc.score >= min_score and oc.selected_cmds]
    if not chosen:
        print(f"[v46-Select] No glyphs met pseudo-label criteria (passed & score >= {min_score}).", flush=True)
        return 0

    trunc_px = 16.0
    chars_out: List[str] = []
    inputs_i8 = np.empty((len(chosen), 4, canvas_size, canvas_size), dtype=np.int8)
    targets_i8 = np.empty((len(chosen), 1, canvas_size, canvas_size), dtype=np.int8)
    weights = np.empty((len(chosen),), dtype=np.float32)

    for i, oc in enumerate(chosen):
        chars_out.append(oc.char)
        sl, dtu, c2, s2 = build_deskinned_channels(
            sc_cmds_map[oc.char], canvas_size=canvas_size, padding=96.0, trunc_px=trunc_px
        )
        x_f = pack_input_tensor(
            sl, dtu, c2, s2, uniform_half_width_u=40.0, canvas_size=canvas_size, padding=96.0, trunc_px=trunc_px
        )
        if oc.selected_sdf is not None and oc.selected_sdf.shape == (canvas_size, canvas_size):
            y_f = oc.selected_sdf[None, :, :].astype(np.float32)
        else:
            m_tg = rasterize_cmds(oc.selected_cmds, canvas_size=canvas_size * 2, padding=96.0)
            y_f = mask_to_normalized_sdf(m_tg, canvas_size, trunc_px=trunc_px)[None, :, :]
        inputs_i8[i] = np.clip(np.round(x_f * 127.0), -127, 127).astype(np.int8)
        targets_i8[i] = np.clip(np.round(y_f * 127.0), -127, 127).astype(np.int8)
        # Map score [min_score, 115] -> confidence weight [0.6, 1.0]
        conf = 0.6 + 0.4 * float(np.clip((oc.score - min_score) / max(115.0 - min_score, 1.0), 0.0, 1.0))
        weights[i] = conf

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        chars=np.array(chars_out),
        inputs_i8=inputs_i8,
        targets_i8=targets_i8,
        weights=weights,
    )
    print(f"[v46-Select] Exported {len(chosen)} pseudo-labels -> {out_npz}", flush=True)
    return len(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description="v4.6 Multi-Sample Candidate Selection Pipeline")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--noto_sans", type=Path, default=Path(os.environ.get("CNS_NOTO_SANS", "")))
    parser.add_argument("--noto_weight", type=int, default=300)
    parser.add_argument("--tg_dir", type=Path, default=ROOT / "data_v2" / "tg8105")
    parser.add_argument("--ids_path", type=Path, default=ROOT / "data_v2" / "ids.txt")

    parser.add_argument("--chars", type=str, default="", help="Explicit characters to run (e.g. '红级练计认议')")
    parser.add_argument("--radicals", type=str, default="纟,讠", help="IDS radical filter when --chars is omitted")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of characters")

    parser.add_argument("--otf_src", action="append", default=[], help="Candidate OTF as name=path.otf (repeatable)")
    parser.add_argument("--revec_otf", action="store_true", help="Also include v4.6 re-vectorized copies of OTF candidates")
    parser.add_argument("--ckpt", type=Path, default=None, help="Optional v4.5 or v4.6 checkpoint for neural SDF sampling")
    parser.add_argument("--mc_samples", type=int, default=3, help="Number of MC-dropout samples per char")
    parser.add_argument("--no_structural_gate", action="store_true", help="Disable StructuralGateV45 wrapper (384px QA only)")

    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs" / "v46_select")
    parser.add_argument("--export_pseudo_npz", type=Path, default=None, help="Optional NPZ export for v46_train --extra_npz")
    parser.add_argument("--pseudo_min_score", type=float, default=82.0)
    parser.add_argument("--emit_otf", action="store_true", help="Emit standalone OTF with selected passing glyphs")
    args = parser.parse_args()

    if not str(args.noto_sans) or not args.noto_sans.exists():
        raise SystemExit(f"Provide valid --noto_sans path (or set CNS_NOTO_SANS). Got: {args.noto_sans}")
    if not args.otf_src and args.ckpt is None:
        raise SystemExit("Provide at least one --otf_src or --ckpt candidate source.")

    t0 = time.time()
    ids_map = parse_cjkvi_ids(args.ids_path) if args.ids_path.exists() else {}
    target_chars = _select_target_chars(args, ids_map)
    if args.limit > 0:
        target_chars = target_chars[: args.limit]
    print(f"[v46-Select] Target chars: {len(target_chars)}", flush=True)

    sn_jp = SansLightExtractor(args.noto_sans, language="ja", weight=args.noto_weight)
    sn_sc = SansLightExtractor(args.noto_sans, language="zh-Hans", weight=args.noto_weight)
    sg_gate = None if args.no_structural_gate else StructuralGateV45(sn_jp, sn_sc)
    qa_gate = GlyphQAGateV46(structural_gate=sg_gate, cfg=QAConfig())

    otf_extractors = _parse_otf_sources(args.otf_src)
    neural_sampler: Optional[NeuralSDFSamplerV46] = None
    if args.ckpt is not None:
        neural_sampler = NeuralSDFSamplerV46(args.ckpt, ids_map=ids_map)

    outcomes: List[SelectionOutcome] = []
    sc_cmds_map: Dict[str, List[DrawCmd]] = {}
    selected_otf_cmds: Dict[int, List[DrawCmd]] = {}

    for ch in tqdm(target_chars, desc="v46 Select", mininterval=3.0):
        sc_cmds = sn_sc.get_gid_and_cmds(ch, "zh-Hans")[1]
        if not sc_cmds:
            continue
        sc_cmds_map[ch] = sc_cmds

        pool: List[CandidateItem] = []
        for src_name, ext in otf_extractors:
            c_otf = ext.get_cmds(ch)
            if c_otf:
                pool.append(CandidateItem(source=src_name, cmds=c_otf))

        if neural_sampler is not None:
            pool.extend(
                neural_sampler.sample_glyph(
                    ch, sc_cmds, mc_samples=args.mc_samples, tag_prefix="neural"
                )
            )

        oc = select_best_candidate_v46(
            ch,
            sc_cmds,
            pool,
            qa_gate,
            check_jp_diff=True,
            revec_otf_candidates=args.revec_otf,
        )
        outcomes.append(oc)
        if oc.gate_passed and oc.selected_cmds:
            selected_otf_cmds[ord(ch)] = oc.selected_cmds

    for _, ext in otf_extractors:
        ext.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        oc.char: {
            "codepoint": f"U+{ord(oc.char):04X}",
            "selected_source": oc.selected_source,
            "gate_passed": oc.gate_passed,
            "score": oc.score,
            "instability_var": oc.instability_var,
            "mean_pairwise_iou": oc.mean_pairwise_iou,
            "candidates": oc.candidates,
        }
        for oc in outcomes
    }
    man_p = args.out_dir / "v46_selection_manifest.json"
    man_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    instab = summarize_instability_by_radical(outcomes, ids_map)
    ins_p = args.out_dir / "v46_instability_by_radical.json"
    ins_p.write_text(json.dumps(instab, ensure_ascii=False, indent=2), encoding="utf-8")

    n_pass = sum(1 for oc in outcomes if oc.gate_passed)
    print(
        f"[v46-Select] Done in {time.time() - t0:.1f}s | Passed={n_pass}/{len(outcomes)} | "
        f"Manifest -> {man_p} | Instability -> {ins_p}",
        flush=True,
    )

    if args.export_pseudo_npz is not None:
        _export_pseudo_npz(outcomes, sc_cmds_map, args.export_pseudo_npz, min_score=args.pseudo_min_score)

    if args.emit_otf and selected_otf_cmds:
        otf_p = args.out_dir / "CnS-SkipPro-B-v46-Selected.otf"
        build_standalone_cns_otf(
            args.skip_otf,
            otf_p,
            selected_otf_cmds,
            family_name="CnS SkipPro B v46 Selected",
        )
        print(f"[v46-Select] OTF ({len(selected_otf_cmds)} glyphs) -> {otf_p}", flush=True)


if __name__ == "__main__":
    main()
