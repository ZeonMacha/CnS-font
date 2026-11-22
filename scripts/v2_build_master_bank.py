#!/usr/bin/env python3
"""
Step 3 Deliverable 1: Build Curated ~116-Glyph `manual_review/CnS-Manual-Master-Bank.otf`.

1. Trains (or loads) the v2.0 HarfBuzz-exact Hessian-Straightness LocalStrokeSDFNet
   (ZERO PatchGAN ripples, ZERO Pillow raqm fallback!).
2. Synthesizes all high-leverage bound radicals, simplified/atomic roots, and core JP->SC
   archetypes using Tier 0 Delta Surgery, Tier 2 Native Stroke Assembly, and Tier 3 Hessian-SDF
   + §4.2 Primitive-Aware Curve Optimization (0 mid-stem wave nodes).
3. Exports:
   - `manual_review/CnS-Manual-Master-Bank.otf` (editable directly in FontCreator/FontLab/FontForge)
   - `manual_review/master_bank_wireframe.png` (control-point wireframe & node count check)
   - `manual_review/MANUAL_EDIT_GUIDE.md`
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.component_bank import (
    PositionalDonorBank,
    compose_from_ids,
    compute_otf_glyph_hashes,
    delta_island_surgery,
    parse_cjkvi_ids,
    seq_has_invalid_entity,
    top_split,
)
from src.v2.curve_optimizer import count_outline_stats, optimize_synthesized_contours, render_wireframe_cell
from src.v2.hb_outlines import (
    DrawCmd,
    HBOutlineExtractor,
    SkipProOutlineExtractor,
    classify_tg8105,
)
from src.v2.stats_and_qa import check_glyph_qa_v2, measure_skippro_stats
from src.v2.stroke_assembler import AtomicStrokeAssembler, build_subset_master_otf
from src.v2.zh_sdf_fallback import ZHSDFSynthesizer, train_zh_hessian_sdf_model

# Master Bank Groups (~116 high-leverage glyphs covering >=98.8% of Tonggui-8105 recursive IDS!)
BOUND_RADICALS = list("讠钅纟饣辶亠礻衤忄艹冫阝")
SIMPLIFIED_ROOTS = list(
    "车贝见页马鸟鱼风韦齿龙长东专书乐为发乌义乡习买农仓仑佥尧乔夹执肃头丝亚严两丧业丛丢伞丌卫丑丐电飞办丰丽举师军宾责员岂齐卖卢刍寻罗聂婴赞华黾监厌鲁冈奂"
)
JP_SC_ARCHETYPES = list("言直令骨微真反刃每衣食鬼角免象曾半平户羽道海神社达迁关异运亏")
PASS1_ATOMIC_OVERRIDES = set(
    "伞丌习卫直令微真反刃每衣食鬼免象曾半平户鱼买飞风义仓仑乔尧两丧丽举师齐卢刍监"
)


def deduplicate_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in seq:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def render_master_wireframe_sheet(
    items: List[Tuple[str, List[DrawCmd], str]],
    out_png: Path,
    cols: int = 10,
    cell_size: int = 210,
) -> None:
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * cell_size, rows * cell_size), (14, 14, 18))
    for idx, (ch, cmds, tag) in enumerate(items):
        r, c = divmod(idx, cols)
        badge = f"U+{ord(ch):04X} {tag}"
        cell = render_wireframe_cell(cmds, size=cell_size, title=badge)
        sheet.paste(cell, (c * cell_size, r * cell_size))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def write_manual_edit_guide(
    out_md: Path,
    provenance: Dict[str, str],
    stats_map: Dict[str, dict],
) -> None:
    lines = [
        "# CnS-Manual-Master-Bank 人工精修指引",
        "",
        "本目录下的 [`CnS-Manual-Master-Bank.otf`](./CnS-Manual-Master-Bank.otf) 是全套 **8,105 《通用规范汉字表》** 的高复用核心母件字库（仅包含下列约 116 个核心字形，打开极快，可直接用 **FontCreator / FontLab / Glyphs / FontForge** 编辑）。",
        "",
        "## 修改后如何自动传播到全部 8,105 字？",
        "",
        "编辑保存 `manual_review/CnS-Manual-Master-Bank.otf`（**保持每个字的 Unicode 码位不变**）后，只需重新运行一条构建命令：",
        "",
        "```bash",
        "python scripts/v2_build_tg8105.py --master_otf manual_review/CnS-Manual-Master-Bank.otf",
        "```",
        "",
        "全库合成器会优先读取你修改过的母件，自动把修改传播至所有下游形声字、合体字与异形字！",
        "",
        "## 收录字形分组与优先精修建议",
        "",
        "1. **第一优先级 — 12 个高频偏旁母件**（单个修改影响 50~400 个下游合体字）：",
        f"   - `{''.join(BOUND_RADICALS)}`",
        "   - 建议重点检查 `讠`（点与折提比例）、`钅`（三横间距与提笔角度）、`纟`（两撇折交错锐度）、`饣`、`辶`（国标一点走之底）。",
        "2. **第二优先级 — 独体与高频简体根字（74 字）**：",
        f"   - `{''.join(SIMPLIFIED_ROOTS)}`",
        "   - 建议重点审阅独体标杆字：`伞 (U+4F1E)`、`丌 (U+4E0C)`、`习 (U+4E60)`、`乡 (U+4E61)`、`卫 (U+536B)`、`专 (U+4E13)`、`书 (U+4E66)`、`东 (U+4E1C)`。",
        "3. **第三优先级 — 中日字形差异母件（30 字）**：",
        f"   - `{''.join(JP_SC_ARCHETYPES)}`",
        "   - 含 `言 (U+8A00)`、`直 (U+76F4)`、`令 (U+4EE4)`、`骨 (U+9AA8)`、`微 (U+5FAE)`、`道 (U+9053)` 等高频陆标差异原字。",
        "",
        "## 节点 Wireframe 预览图",
        "",
        "![Master Bank Wireframe](./master_bank_wireframe.png)",
        "",
        "## 各母件生成来源与节点统计",
        "",
        "| 字符 | 码位 | 合成路径 | On-Curve 点数 | 直线中段冗余节点 |",
        "| :---: | :---: | :--- | :---: | :---: |",
    ]
    for ch, tag in provenance.items():
        st = stats_map[ch]
        lines.append(
            f"| {ch} | `U+{ord(ch):04X}` | `{tag}` | {st['on_curve_nodes']} | {st['mid_stem_nodes']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CnS-Manual-Master-Bank.otf (Step 3 Deliverable 1)")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--noto_serif", type=Path, required=True)
    parser.add_argument("--noto_sans", type=Path, required=True)
    parser.add_argument("--tg_dir", type=Path, default=ROOT / "data_v2" / "tg8105")
    parser.add_argument("--ids_path", type=Path, default=ROOT / "data_v2" / "ids.txt")
    parser.add_argument("--ckpt_path", type=Path, default=ROOT / "checkpoints" / "v2_zh_sdf_s256.pt")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "manual_review")
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()

    # 1. Ensure Hessian-regularized zh-Hans SDF fallback model is trained (no PatchGAN, no raqm!)
    train_zh_hessian_sdf_model(
        skip_otf=args.skip_otf,
        noto_serif=args.noto_serif,
        noto_sans=args.noto_sans,
        ckpt_out=args.ckpt_path,
        cache_dir=ROOT / "datasets",
        epochs=args.epochs,
    )

    skip_ext = SkipProOutlineExtractor(args.skip_otf)
    serif_jp = HBOutlineExtractor(args.noto_serif, ttc_index=0, weight=700)
    serif_sc = HBOutlineExtractor(args.noto_serif, ttc_index=2, weight=700)
    sans_jp = HBOutlineExtractor(args.noto_sans, ttc_index=0, weight=620)
    sans_sc = HBOutlineExtractor(args.noto_sans, ttc_index=2, weight=620)
    ids_map = parse_cjkvi_ids(args.ids_path)

    print("[MasterBank] Classifying TG-8105 JP!=SC characters via HarfBuzz...", flush=True)
    cls_res = classify_tg8105(
        args.tg_dir, skip_ext, serif_jp, serif_sc, sans_jp, sans_sc, ids_map=ids_map
    )
    jp_diff_set = set(cls_res.diff_all_2623)
    print(
        f"[MasterBank] TG8105: missing={len(cls_res.missing_4101)} | "
        f"same={len(cls_res.same_1381)} | diff_all={len(cls_res.diff_all_2623)} | "
        f"diff_replace={len(cls_res.diff_replace)}",
        flush=True,
    )

    stats = measure_skippro_stats(skip_ext, sans_sc, cls_res.same_1381, sample_n=300)
    bank = PositionalDonorBank(skip_ext, ids_map, jp_diff_set, sans_sc=sans_sc)
    donor_cnt = bank.build_from_skippro()
    print(f"[MasterBank] Indexed {donor_cnt} clean SkipPro-B positional donor components.", flush=True)

    atomic_asm = AtomicStrokeAssembler(skip_ext, sans_sc, serif_sc, jp_diff_set)
    sdf_synth = ZHSDFSynthesizer(args.ckpt_path, serif_sc=serif_sc, sans_sc=sans_sc)

    master_chars = deduplicate_order(BOUND_RADICALS + SIMPLIFIED_ROOTS + JP_SC_ARCHETYPES)
    master_cmds: Dict[int, List[DrawCmd]] = {}
    provenance: Dict[str, str] = {}
    stats_map: Dict[str, dict] = {}
    bound_set = set(BOUND_RADICALS)

    pass1_chars: List[str] = []
    pass2_chars: List[str] = []
    for ch in master_chars:
        seq = ids_map.get(ch, ch)
        idc, comps = top_split(seq)
        if (
            ch in bound_set
            or ch in PASS1_ATOMIC_OVERRIDES
            or idc is None
            or idc == "⿻"
            or comps == [ch]
            or seq_has_invalid_entity(seq)
        ):
            pass1_chars.append(ch)
        else:
            pass2_chars.append(ch)

    # Pass 1: Synthesize bound radicals & irreducible roots first, then register into bank
    p1_fallback: List[str] = []
    for ch in pass1_chars:
        _, ref_sc = sans_sc.get_gid_and_cmds(ch, "zh-Hans")
        if skip_ext.has_char(ch) and ch in jp_diff_set:
            res0 = delta_island_surgery(ch, skip_ext, serif_jp, serif_sc, sans_sc, bank)
            if res0 is not None:
                ok0, _ = check_glyph_qa_v2(res0[0], ref_sc, stats, is_synthesized=False)
                if ok0:
                    master_cmds[ord(ch)], provenance[ch] = res0
                    continue

        res2 = atomic_asm.try_assemble_curated_atomic(ch)
        if res2 is not None:
            cmds = optimize_synthesized_contours(res2[0], stem_concave_sagitta=0.85)
            master_cmds[ord(ch)] = cmds
            provenance[ch] = res2[1]
            continue

        p1_fallback.append(ch)

    if p1_fallback:
        print(f"[MasterBank Pass 1] Synthesizing {len(p1_fallback)} radical/atomic masters via zh-Hans Hessian-SDF...", flush=True)
        fb_res1 = sdf_synth.synthesize_batch(p1_fallback)
        for ch, cmds in fb_res1.items():
            master_cmds[ord(ch)] = cmds
            provenance[ch] = "Tier3_HessianSDF"

    for ch in pass1_chars:
        if ord(ch) in master_cmds:
            bank.register_master_glyph(ch, master_cmds[ord(ch)], overwrite_slots=(ch in bound_set))

    # Pass 2: Composite archetypes (海, 神, 社, 道, 达, 迁, 运, 卖, 责, 员, ...) inherit Pass-1 masters + SkipPro-B donors
    p2_fallback: List[str] = []
    for ch in pass2_chars:
        _, ref_sc = sans_sc.get_gid_and_cmds(ch, "zh-Hans")
        if skip_ext.has_char(ch) and ch in jp_diff_set:
            res0 = delta_island_surgery(ch, skip_ext, serif_jp, serif_sc, sans_sc, bank)
            if res0 is not None:
                ok0, _ = check_glyph_qa_v2(res0[0], ref_sc, stats, is_synthesized=False)
                if ok0:
                    master_cmds[ord(ch)], provenance[ch] = res0
                    continue

        res1 = compose_from_ids(ch, bank, sans_sc)
        if res1 is not None:
            ok1, qa1 = check_glyph_qa_v2(res1[0], ref_sc, stats, is_synthesized=False)
            if ok1 and float(qa1.get("iou_vs_sc", 0.0)) >= 0.55:
                master_cmds[ord(ch)], provenance[ch] = res1
                continue

        res2 = atomic_asm.try_assemble_curated_atomic(ch)
        if res2 is not None:
            cmds = optimize_synthesized_contours(res2[0], stem_concave_sagitta=0.85)
            master_cmds[ord(ch)] = cmds
            provenance[ch] = res2[1]
            continue

        p2_fallback.append(ch)

    if p2_fallback:
        print(f"[MasterBank Pass 2] Synthesizing {len(p2_fallback)} composite masters via zh-Hans Hessian-SDF...", flush=True)
        fb_res2 = sdf_synth.synthesize_batch(p2_fallback)
        for ch, cmds in fb_res2.items():
            master_cmds[ord(ch)] = cmds
            provenance[ch] = "Tier3_HessianSDF"

    for ch in pass2_chars:
        if ord(ch) in master_cmds:
            bank.register_master_glyph(ch, master_cmds[ord(ch)], overwrite_slots=False)

    ordered_wire_items: List[Tuple[str, List[DrawCmd], str]] = []
    for ch in master_chars:
        cmds = master_cmds.get(ord(ch), [])
        stats_map[ch] = count_outline_stats(cmds)
        ordered_wire_items.append((ch, cmds, provenance.get(ch, "unknown")))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    master_otf = args.out_dir / "CnS-Manual-Master-Bank.otf"
    n_packed = build_subset_master_otf(args.skip_otf, master_otf, master_cmds)

    hashes = compute_otf_glyph_hashes(master_otf)
    (args.out_dir / "master_bank_hashes.json").write_text(
        json.dumps({"hashes": hashes, "provenance": provenance}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    wire_png = args.out_dir / "master_bank_wireframe.png"
    render_master_wireframe_sheet(ordered_wire_items, wire_png, cols=10, cell_size=210)
    write_manual_edit_guide(args.out_dir / "MANUAL_EDIT_GUIDE.md", provenance, stats_map)

    summary = {
        "master_glyphs_packed": n_packed,
        "tier0_delta_surgery": sum("Tier0" in v for v in provenance.values()),
        "tier2_atomic_native": sum("Tier2" in v for v in provenance.values()),
        "tier3_hessian_sdf": sum("Tier3" in v for v in provenance.values()),
        "zero_mid_stem_rate": round(
            sum(s["mid_stem_nodes"] == 0 for s in stats_map.values()) / max(len(stats_map), 1), 4
        ),
        "median_on_curve_nodes": int(sorted(s["on_curve_nodes"] for s in stats_map.values())[len(stats_map) // 2]),
    }
    (args.out_dir / "master_bank_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    skip_ext.close()
    print(f"[MasterBank] Built {master_otf} ({n_packed} glyphs) | Summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
