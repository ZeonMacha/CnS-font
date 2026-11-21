#!/usr/bin/env python3
"""
Batch Simplified Chinese Character Synthesis, QA Gate & OpenType CFF Font Compiler.

Generates missing Simplified Chinese characters (GB2312 Level-1 / Full GB2312 / Custom)
using the trained Restricted-Receptive-Field `LocalStrokeSDFNet`, validates every glyph
against the 4-stage QA pipeline, and compiles `CnS-SkipPro-B-SC.otf`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glyph_renderer import (
    AlignedFontRenderer,
    CanvasSpec,
    get_font_cjk_codepoints,
    get_gb2312_sets,
)
from src.model import LocalStrokeSDFNet
from src.otf_builder import evaluate_topology_qa, pack_glyphs_into_otf
from src.sdf_utils import build_dual_skeleton_input
from src.vectorizer import DrawCmd, sdf_to_cff_commands


def resolve_target_codepoints(
    charset_mode: str,
    custom_chars: str,
    existing_skip_cjk: set[int],
    avail_sc_cjk: set[int],
) -> List[int]:
    gb_l1, gb_all = get_gb2312_sets()
    if charset_mode == "gb2312_l1":
        candidates = gb_l1 - existing_skip_cjk
    elif charset_mode == "gb2312":
        candidates = gb_all - existing_skip_cjk
    elif charset_mode == "custom":
        candidates = {ord(c) for c in custom_chars if not c.isspace()}
    else:
        raise ValueError(f"Unknown charset mode: {charset_mode}")
    return sorted(candidates & avail_sc_cjk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate missing SC glyphs and pack into OTF")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument(
        "--noto_serif",
        type=Path,
        default=Path("/usr/share/fonts/google-noto-serif-cjk-vf-fonts/NotoSerifCJK-VF.ttc"),
    )
    parser.add_argument(
        "--noto_sans",
        type=Path,
        default=Path("/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc"),
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "m1_local_sdf_net.pt")
    parser.add_argument("--charset", choices=["gb2312_l1", "gb2312", "custom"], default="gb2312_l1")
    parser.add_argument("--chars", type=str, default="", help="Custom characters when --charset=custom")
    parser.add_argument("--canvas_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--upsample_factor", type=int, default=4)
    parser.add_argument("--out_otf", type=Path, default=ROOT / "outputs" / "CnS-SkipPro-B-SC.otf")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunc_px = 16.0 * (args.canvas_size / 256.0)
    max_shift_px = 6.0 * (args.canvas_size / 256.0)

    model = LocalStrokeSDFNet(
        in_channels=4,
        base_channels=args.base_channels,
        max_shift_px=max_shift_px,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    existing_cjk = get_font_cjk_codepoints(args.skip_otf, ttc_index=0)
    avail_sc = get_font_cjk_codepoints(args.noto_serif, ttc_index=2) & get_font_cjk_codepoints(
        args.noto_sans, ttc_index=2
    )
    target_cps = resolve_target_codepoints(args.charset, args.chars, existing_cjk, avail_sc)
    print(f"[BatchGen] Target missing characters to synthesize: {len(target_cps)}")

    spec = CanvasSpec(canvas_size=args.canvas_size)
    serif_sc = AlignedFontRenderer(args.noto_serif, spec, ttc_index=2, variation_weight=700)
    sans_sc = AlignedFontRenderer(args.noto_sans, spec, ttc_index=2, variation_weight=620)

    glyph_cmds: Dict[int, List[DrawCmd]] = {}
    qa_outliers: Dict[str, dict] = {}
    t0 = time.time()

    with torch.no_grad():
        for start in tqdm(range(0, len(target_cps), args.batch_size), desc="Synthesizing & Vectorizing"):
            batch_cps = target_cps[start : start + args.batch_size]
            batch_x = [
                build_dual_skeleton_input(chr(cp), serif_sc, sans_sc, trunc_px=trunc_px)
                for cp in batch_cps
            ]
            xt = torch.from_numpy(np.stack(batch_x, axis=0)).to(device)
            pred_sdfs, _ = model(xt)
            pred_np = pred_sdfs[:, 0].cpu().numpy()

            for i, cp in enumerate(batch_cps):
                qa = evaluate_topology_qa(pred_np[i], batch_x[i], trunc_px=trunc_px)
                cmds = sdf_to_cff_commands(pred_np[i], spec, upsample_factor=args.upsample_factor)
                anchor_cnt = sum(len(pts) for _, pts in cmds)
                if (
                    qa["phantom_ink_rate"] > 0.015
                    or qa["skeleton_recall"] < 0.86
                    or anchor_cnt > 280
                ):
                    qa_outliers[chr(cp)] = {**qa, "anchor_count": anchor_cnt}
                glyph_cmds[cp] = cmds

    packed = pack_glyphs_into_otf(args.skip_otf, args.out_otf, glyph_cmds)
    qa_log_path = args.out_otf.with_suffix(".qa_outliers.json")
    qa_log_path.write_text(json.dumps(qa_outliers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[BatchGen] Done in {time.time() - t0:.1f}s | Packed {packed} glyphs -> {args.out_otf} | "
        f"QA flagged outliers: {len(qa_outliers)} ({qa_log_path})"
    )


if __name__ == "__main__":
    main()
