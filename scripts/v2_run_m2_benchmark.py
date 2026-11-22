#!/usr/bin/env python3
"""
Step 4 M2 Benchmark Verification Script:
  Tests node cleanliness, zero-mid-stem straight edges, 100% original donor preservation,
  and Mainland `zh-Hans` correctness across the 3 targeted benchmark groups:
    - Group 1 (Composites): 红, 语, 银, 饭 (side-by-side vs original 紅 & v1 红!)
    - Group 2 (Atomics):    伞, 丌, 习, 乡, 卫, 专, 书, 东
    - Group 3 (JP!=SC):     言, 直, 令, 骨, 微, 道
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.component_bank import (
    PositionalDonorBank,
    compose_from_ids,
    delta_island_surgery,
    load_master_metadata,
    parse_cjkvi_ids,
)
from src.v2.curve_optimizer import (
    count_outline_stats,
    optimize_synthesized_contours,
    render_wireframe_cell,
)
from src.v2.hb_outlines import (
    DrawCmd,
    HBOutlineExtractor,
    SkipProOutlineExtractor,
    classify_tg8105,
)
from src.v2.stats_and_qa import check_glyph_qa_v2, measure_skippro_stats
from src.v2.stroke_assembler import AtomicStrokeAssembler
from src.v2.zh_sdf_fallback import ZHSDFSynthesizer

M2_COMPOSITES = list("红语银饭")
M2_ATOMICS = list("伞丌习乡卫专书东")
M2_JP_SC = list("言直令骨微道")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M2 Targeted Node-Level Benchmark Verification")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--v1_otf", type=Path, default=ROOT / "outputs" / "CnS-SkipPro-B-GB2312-SC.otf")
    parser.add_argument("--master_otf", type=Path, default=ROOT / "manual_review" / "CnS-Manual-Master-Bank.otf")
    parser.add_argument("--noto_serif", type=Path, required=True)
    parser.add_argument("--noto_sans", type=Path, required=True)
    parser.add_argument("--tg_dir", type=Path, default=ROOT / "data_v2" / "tg8105")
    parser.add_argument("--ids_path", type=Path, default=ROOT / "data_v2" / "ids.txt")
    parser.add_argument("--ckpt_path", type=Path, default=ROOT / "checkpoints" / "v2_zh_sdf_s256.pt")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    skip_ext = SkipProOutlineExtractor(args.skip_otf)
    v1_ext = SkipProOutlineExtractor(args.v1_otf) if args.v1_otf.exists() else None
    serif_jp = HBOutlineExtractor(args.noto_serif, ttc_index=0, weight=700)
    serif_sc = HBOutlineExtractor(args.noto_serif, ttc_index=2, weight=700)
    sans_jp = HBOutlineExtractor(args.noto_sans, ttc_index=0, weight=620)
    sans_sc = HBOutlineExtractor(args.noto_sans, ttc_index=2, weight=620)
    ids_map = parse_cjkvi_ids(args.ids_path)

    cls_res = classify_tg8105(args.tg_dir, skip_ext, serif_jp, serif_sc, sans_jp, sans_sc, ids_map=ids_map)
    jp_diff = set(cls_res.diff_all_2623)

    stats = measure_skippro_stats(skip_ext, sans_sc, cls_res.same_1381, sample_n=300)
    print(
        f"[M2] Calibrated SkipProStats: concave p10/p50/p90=({stats.concave_sagitta_p10:.2f}, "
        f"{stats.concave_sagitta_p50:.2f}, {stats.concave_sagitta_p90:.2f})u | "
        f"calib_false_alarm={stats.calib_false_alarm_rate*100:.1f}%",
        flush=True,
    )

    bank = PositionalDonorBank(skip_ext, ids_map, jp_diff, sans_sc=sans_sc)
    bank.build_from_skippro()
    edited_masters: set[str] = set()
    master_provenance: Dict[str, str] = {}
    if args.master_otf.exists():
        edited_masters, master_provenance = load_master_metadata(args.master_otf)
        loaded_m = bank.load_master_otf(args.master_otf, edited_chars=edited_masters)
        print(
            f"[M2] Loaded {loaded_m} curated masters from {args.master_otf} (edited={len(edited_masters)})",
            flush=True,
        )

    atomic_asm = AtomicStrokeAssembler(skip_ext, sans_sc, serif_sc, jp_diff)
    sdf_synth = ZHSDFSynthesizer(args.ckpt_path, serif_sc=serif_sc, sans_sc=sans_sc)

    all_targets = M2_COMPOSITES + M2_ATOMICS + M2_JP_SC
    results: Dict[str, tuple[List[DrawCmd], str, dict]] = {}

    for ch in all_targets:
        _, ref_sc = sans_sc.get_gid_and_cmds(ch, "zh-Hans")
        cmds, tag = None, ""
        is_synth = True

        # 1. Human-edited Master Bank override always wins
        if ch in edited_masters and ch in bank.master_full:
            cmds = bank.master_full[ch].cmds
            tag = "MasterEditedDirect"
            is_synth = False

        # 2. Tier 0 Delta Surgery for JP!=SC if already in SkipPro-B (is_synthesized=False!)
        if cmds is None and skip_ext.has_char(ch):
            r0 = delta_island_surgery(ch, skip_ext, serif_jp, serif_sc, sans_sc, bank)
            if r0 is not None:
                ok0, _ = check_glyph_qa_v2(r0[0], ref_sc, stats, is_synthesized=False)
                if ok0:
                    cmds, tag = r0
                    is_synth = False

        # 3. Tier 1 Recursive Pure-Vector Composition (is_synthesized=False!)
        if cmds is None:
            r1 = compose_from_ids(ch, bank, sans_sc)
            if r1 is not None:
                ok1, _ = check_glyph_qa_v2(r1[0], ref_sc, stats, is_synthesized=False)
                if ok1:
                    cmds, tag = r1
                    is_synth = False

        # 4. Direct Master Bank fallback
        if cmds is None and ch in bank.master_full:
            cmds = bank.master_full[ch].cmds
            tag = "MasterDirect"
            is_synth = master_provenance.get(ch, "").startswith(("Tier2", "Tier3"))

        # 5. Tier 2 Atomic or Tier 3 Hessian-SDF
        if cmds is None:
            r2 = atomic_asm.try_assemble_curated_atomic(ch)
            if r2 is not None:
                cand2 = optimize_synthesized_contours(r2[0], stem_concave_sagitta=0.85)
                ok2, _ = check_glyph_qa_v2(cand2, ref_sc, stats, is_synthesized=True)
                if ok2:
                    cmds, tag = cand2, r2[1]
                    is_synth = True
        if cmds is None:
            cmds = sdf_synth.synthesize_batch([ch])[ch]
            tag = "Tier3_SDF"
            is_synth = True

        _, qa = check_glyph_qa_v2(cmds, ref_sc, stats, is_synthesized=is_synth)
        results[ch] = (cmds, tag, qa)

    # Render node-level comparison sheet:
    # Row 1: Original SkipPro-B equivalents (紅 語 銀 飯 + 傘/丌... Noto SC + 原厂 言 直 令 骨 微 道)
    # Row 2: v1 Output (for direct visual comparison against Figure 1/2/3!)
    # Row 3: v2.0 Output (Node Wireframe!)
    cell = 260
    cols = len(all_targets)
    sheet = Image.new("RGB", (cols * cell, 3 * cell), (12, 12, 16))

    orig_equiv = {"红": "紅", "语": "語", "银": "銀", "饭": "飯"}
    for i, ch in enumerate(all_targets):
        # Row 0: Original SkipPro-B (or Noto Serif SC reference for missing chars)
        eq = orig_equiv.get(ch, ch)
        if skip_ext.has_char(eq):
            c_ref = skip_ext.get_cmds(eq)
            t_ref = f"ORIG:{eq}"
        else:
            _, c_ref = serif_sc.get_gid_and_cmds(ch, "zh-Hans")
            t_ref = f"SC_REF:{ch}"
        sheet.paste(render_wireframe_cell(c_ref, cell, t_ref), (i * cell, 0))

        # Row 1: v1 defective output (if available)
        if v1_ext is not None and v1_ext.has_char(ch):
            c_v1 = v1_ext.get_cmds(ch)
            sheet.paste(render_wireframe_cell(c_v1, cell, f"v1:{ch}"), (i * cell, cell))
        else:
            _, c_sn = sans_sc.get_gid_and_cmds(ch, "zh-Hans")
            sheet.paste(render_wireframe_cell(c_sn, cell, f"SansSC:{ch}"), (i * cell, cell))

        # Row 2: v2.0 output!
        c_v2, tag, _ = results[ch]
        short_tag = tag.split("[")[0][:12]
        sheet.paste(render_wireframe_cell(c_v2, cell, f"v2:{ch} {short_tag}"), (i * cell, 2 * cell))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wire_out = args.out_dir / "v2_m2_node_wireframes.png"
    sheet.save(wire_out)

    report = {
        "calibrated_stats": {
            "concave_sagitta_p10": round(stats.concave_sagitta_p10, 2),
            "concave_sagitta_p50": round(stats.concave_sagitta_p50, 2),
            "concave_sagitta_p90": round(stats.concave_sagitta_p90, 2),
            "calib_false_alarm_rate": round(stats.calib_false_alarm_rate, 4),
        },
        "benchmarks": {
            ch: {"provenance": results[ch][1], **results[ch][2]} for ch in all_targets
        },
    }
    (args.out_dir / "v2_m2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    skip_ext.close()
    if v1_ext is not None:
        v1_ext.close()
    print(f"[M2] Saved wireframe comparison -> {wire_out}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
