#!/usr/bin/env python3
"""
Milestone M3: Build 104-Glyph `manual_review/CnS-Manual-Master-Bank.otf` (`Plan3.0.md §3.5`).

Strict Guarantees:
  1. Removes all 12 `HARD_KEEP_12` chars (`/uni4E10/uni4E30/uni514D/uni51AB/uni534A/uni6BCF/
     uni793B/uni793E/uni795E/uni8279/uni8863/uni8864`) -> exact 104 glyphs.
  2. Preserves native `SkipPro-B` CFF contours (`Tier0_Native`) for non-variant bank chars
     (`忄, 阝, 夹, 平, 羽, 海, ...`).
  3. Uses Stage 0a/0b local rewrite (`辶, 亠, 言, 道, 直, 骨, 微, ...`) preserving untouched
     native `SkipPro-B` islands verbatim.
  4. Uses deskinned `Noto Sans Light` (`wght=300`) + `StructuralGateV3` + `optimize_synthesized_contours_v3`
     (Hausdorff <= 4.0u, zero Noto Serif dependency!).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.component_bank import (
    compute_otf_glyph_hashes,
    parse_cjkvi_ids,
    seq_has_invalid_entity,
    top_split,
)
from src.v2.curve_optimizer import count_outline_stats, render_wireframe_cell
from src.v2.hb_outlines import DrawCmd, SkipProOutlineExtractor, load_tg8105_table
from src.v3 import HARD_KEEP_12, HARD_KEEP_CHARS
from src.v3.curve_constrain import optimize_synthesized_contours_v3
from src.v3.hb_light import SansLightExtractor, measure_skippro_hv_halfwidths
from src.v3.local_inpainter import (
    AtomicStrokeAssemblerV3,
    DonorBankV3,
    ZHSDFSynthesizerV3,
    compose_from_ids_v3,
    tier0_stage0a_vector_surgery,
    tier0_stage0b_local_inpaint_batch,
)
from src.v3.otf_packager import build_standalone_cns_otf
from src.v3.structural_gate import StructuralGateV3
from src.v3.variant_classifier import classify_tg8105_v3, load_variant_policy

BOUND_RADICALS = [c for c in "讠钅纟饣辶亠礻衤忄艹冫阝" if c not in HARD_KEEP_CHARS]
SIMPLIFIED_ROOTS = [
    c
    for c in "车贝见页马鸟鱼风韦齿龙长东专书乐为发乌义乡习买农仓仑佥尧乔夹执肃头丝亚严两丧业丛丢伞丌卫丑丐电飞办丰丽举师军宾责员岂齐卖卢刍寻罗聂婴赞华黾监厌鲁冈奂"
    if c not in HARD_KEEP_CHARS
]
JP_SC_ARCHETYPES = [
    c
    for c in "言直令骨微真反刃每衣食鬼角免象曾半平户羽道海神社达迁关异运亏"
    if c not in HARD_KEEP_CHARS
]
PASS1_ATOMIC_OVERRIDES = (
    set("伞丌习卫直令微真反刃每衣食鬼免象曾半平户鱼买飞风义仓仑乔尧两丧丽举师齐卢刍监")
    - HARD_KEEP_CHARS
)


def deduplicate_order(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for c in seq:
        if c not in seen and c not in HARD_KEEP_CHARS:
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
    gate_passed: Dict[str, bool],
    stats_map: Dict[str, dict],
) -> None:
    excluded = " ".join(f"`{ch}(U+{cp:04X})`" for cp, ch in HARD_KEEP_12.items())
    lines = [
        "# CnS-Manual-Master-Bank (v3.0) 人工精修指引",
        "",
        "本目录下的 [`CnS-Manual-Master-Bank.otf`](./CnS-Manual-Master-Bank.otf) 是 v3.0 的精简高复用核心母件字库，共包含 **104 个字符**，可用 **FontCreator / FontLab / Glyphs / FontForge** 直接打开编辑。",
        "",
        "> **已强制移除以下 12 个保留原厂字形的码位（共 116 → 104 字）**：  ",
        f"> {excluded}",
        "",
        "## 修改后如何自动传播到独立分卷字库？",
        "",
        "编辑保存 `manual_review/CnS-Manual-Master-Bank.otf`（**保持每个字的 Unicode 码位不变**）后，运行：",
        "",
        "```bash",
        "python scripts/v3_build_tg8105.py --master_otf manual_review/CnS-Manual-Master-Bank.otf --noto_sans <NotoSansCJK-VF.ttc>",
        "```",
        "",
        "全库合成器会优先读取你手动编辑过的母件，并将更新分别输出到两个独立字库：",
        "- `outputs/CnS-SkipPro-B-v3-Missing.otf`（原字体缺失字补全）",
        "- `outputs/CnS-SkipPro-B-v3-SCVariants.otf`（非豁免中日异形重绘字）",
        "",
        "## 收录字形分组",
        "",
        f"1. **高频偏旁母件 ({len(BOUND_RADICALS)} 字)**：`{''.join(BOUND_RADICALS)}`",
        f"2. **高频简体与独体根字 ({len(SIMPLIFIED_ROOTS)} 字)**：`{''.join(SIMPLIFIED_ROOTS)}`",
        f"3. **陆标异形母件 ({len(JP_SC_ARCHETYPES)} 字)**：`{''.join(JP_SC_ARCHETYPES)}`",
        "",
        "## 节点 Wireframe 预览图",
        "",
        "![Master Bank Wireframe](./master_bank_wireframe.png)",
        "",
        "## 各母件生成来源、门控与节点统计",
        "",
        "| 字符 | 码位 | 合成路径 | Gate | On-Curve 点数 | 直线中段冗余节点 |",
        "| :---: | :---: | :--- | :---: | :---: | :---: |",
    ]
    for ch, tag in provenance.items():
        st = stats_map[ch]
        gp = "PASS" if gate_passed.get(ch, False) else "REVIEW"
        lines.append(
            f"| {ch} | `U+{ord(ch):04X}` | `{tag}` | {gp} | {st['on_curve_nodes']} | {st['mid_stem_nodes']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 104-glyph v3.0 CnS-Manual-Master-Bank.otf")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--noto_sans", type=Path, required=True)
    parser.add_argument("--tg_dir", type=Path, default=ROOT / "data_v2" / "tg8105")
    parser.add_argument("--ids_path", type=Path, default=ROOT / "data_v2" / "ids.txt")
    parser.add_argument("--policy_tsv", type=Path, default=ROOT / "data_v3" / "variant_policy.tsv")
    parser.add_argument("--ckpt_path", type=Path, default=ROOT / "checkpoints" / "v3_zh_sdf_light.pt")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "manual_review")
    args = parser.parse_args()

    master_chars = deduplicate_order(BOUND_RADICALS + SIMPLIFIED_ROOTS + JP_SC_ARCHETYPES)
    assert len(master_chars) == 104, f"Expected exact 104 master chars, got {len(master_chars)}"
    assert set(master_chars).isdisjoint(HARD_KEEP_CHARS), "HARD_KEEP_12 leaked into Master Bank!"

    skip_ext = SkipProOutlineExtractor(args.skip_otf)
    sn_jp = SansLightExtractor(args.noto_sans, language="ja", weight=300)
    sn_sc = SansLightExtractor(args.noto_sans, language="zh-Hans", weight=300)
    gate = StructuralGateV3(sn_jp, sn_sc)
    tg = load_tg8105_table(args.tg_dir)
    ids_map = parse_cjkvi_ids(args.ids_path)
    policy_map = load_variant_policy(args.policy_tsv)

    print("[v3-MasterBank] Running M1 v3 classifier...", flush=True)
    cls_res = classify_tg8105_v3(tg, skip_ext, sn_jp, sn_sc, gate, ids_map, policy_map)
    sc_variant_set = set(cls_res.sc_variant)
    for ch in master_chars:
        if ch not in tg and skip_ext.has_char(ch):
            if sn_jp.shape_gid(ch, "ja") != sn_sc.shape_gid(ch, "zh-Hans"):
                if not gate.check(skip_ext.get_cmds(ch), ch)[0]:
                    sc_variant_set.add(ch)
    h_half, v_half = measure_skippro_hv_halfwidths(skip_ext, cls_res.keep_same_gid, sample_n=150)
    print(f"[v3-MasterBank] SkipPro-B half-widths: h={h_half:.1f}u, v={v_half:.1f}u | SC_VARIANT={len(sc_variant_set)}", flush=True)

    bank = DonorBankV3(skip_ext, ids_map, sc_variant_set, sn_sc, h_half, v_half)
    donor_cnt = bank.build_from_skippro()
    print(f"[v3-MasterBank] Indexed {donor_cnt} clean SkipPro-B positional donor components.", flush=True)

    atomic_asm = AtomicStrokeAssemblerV3(skip_ext, sn_sc, sc_variant_set, h_half, v_half)
    synth = ZHSDFSynthesizerV3(args.ckpt_path)

    master_cmds: Dict[int, List[DrawCmd]] = {}
    provenance: Dict[str, str] = {}
    gate_passed: Dict[str, bool] = {}
    bound_set = set(BOUND_RADICALS)

    pass1_chars: List[str] = []
    pass2_chars: List[str] = []
    for ch in master_chars:
        if skip_ext.has_char(ch) and ch not in sc_variant_set:
            master_cmds[ord(ch)] = skip_ext.get_cmds(ch)
            provenance[ch] = "Tier0_Native"
            gate_passed[ch] = True
            continue
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

    for ch in master_chars:
        if ord(ch) in master_cmds:
            bank.register_master_glyph(ch, master_cmds[ord(ch)], overwrite_slots=False)

    # Pass 1: Stage 0a -> Tier 2 -> Stage 0b batched local inpainting -> Tier 3 Deskinned SDF
    p1_stage0b: List[str] = []
    p1_sdf: List[str] = []
    for ch in pass1_chars:
        ex_boxes = cls_res.exempt_boxes_map.get(ch)
        if skip_ext.has_char(ch) and ch in sc_variant_set:
            r0 = tier0_stage0a_vector_surgery(ch, skip_ext, sn_jp, sn_sc, bank, gate, exempt_boxes=ex_boxes)
            if r0 is not None:
                master_cmds[ord(ch)], provenance[ch] = r0[0], r0[1].split("(")[0]
                gate_passed[ch] = True
                continue
            p1_stage0b.append(ch)
            continue

        r2 = atomic_asm.try_assemble_curated_atomic(ch)
        if r2 is not None:
            sc_c = sn_sc.get_cmds(ch, unioned=False)
            c2, _ = optimize_synthesized_contours_v3(r2[0], ref_sc_cmds=sc_c, stem_concave_sagitta=0.85)
            ok2, _ = gate.check(c2, ch, exempt_boxes=ex_boxes, check_jp_diff=False)
            if ok2:
                master_cmds[ord(ch)], provenance[ch], gate_passed[ch] = c2, "Tier2_Atomic", True
                continue
        p1_sdf.append(ch)

    if p1_stage0b:
        print(f"[v3-MasterBank Pass 1] Stage 0b local inpainting for {len(p1_stage0b)} SC_VARIANT roots...", flush=True)
        acc0b, fail0b = tier0_stage0b_local_inpaint_batch(
            p1_stage0b, skip_ext, sn_jp, sn_sc, synth, gate, exempt_boxes_map=cls_res.exempt_boxes_map
        )
        for ch, (cmds, tag, _) in acc0b.items():
            master_cmds[ord(ch)], provenance[ch], gate_passed[ch] = cmds, tag, True
        for ch in fail0b:
            p1_sdf.append(ch)

    def _run_sdf_split(chars_queue: List[str], label: str) -> None:
        miss_q = [c for c in chars_queue if not skip_ext.has_char(c)]
        var_q = [c for c in chars_queue if skip_ext.has_char(c)]
        for sub_q, chk_jp in ((miss_q, False), (var_q, True)):
            if not sub_q:
                continue
            print(f"[v3-MasterBank {label}] Synthesizing {len(sub_q)} chars (check_jp_diff={chk_jp}) via Deskinned Light SDF...", flush=True)
            acc3, fail3 = synth.synthesize_with_gate_retries(
                sub_q, sn_sc, gate, exempt_boxes_map=cls_res.exempt_boxes_map, check_jp_diff=chk_jp
            )
            for ch, (cmds, tag, _) in acc3.items():
                master_cmds[ord(ch)], provenance[ch], gate_passed[ch] = cmds, tag.split("(")[0], True
            for ch, meta in fail3.items():
                fb = meta.get("fallback_cmds", [])
                if fb:
                    master_cmds[ord(ch)] = fb
                provenance[ch] = "Tier3_NeedsReview"
                gate_passed[ch] = False

    if p1_sdf:
        _run_sdf_split(p1_sdf, "Pass 1")

    for ch in pass1_chars:
        if ord(ch) in master_cmds and gate_passed.get(ch, False):
            bank.register_master_glyph(ch, master_cmds[ord(ch)], overwrite_slots=(ch in bound_set))

    # Pass 2: Composite archetypes
    p2_stage0b: List[str] = []
    p2_sdf: List[str] = []
    for ch in pass2_chars:
        ex_boxes = cls_res.exempt_boxes_map.get(ch)
        if skip_ext.has_char(ch) and ch in sc_variant_set:
            r0 = tier0_stage0a_vector_surgery(ch, skip_ext, sn_jp, sn_sc, bank, gate, exempt_boxes=ex_boxes)
            if r0 is not None:
                master_cmds[ord(ch)], provenance[ch], gate_passed[ch] = r0[0], r0[1].split("(")[0], True
                continue
            r1 = compose_from_ids_v3(ch, bank, sn_sc)
            if r1 is not None:
                ok1, _ = gate.check(r1[0], ch, exempt_boxes=ex_boxes, check_jp_diff=True)
                if ok1:
                    master_cmds[ord(ch)], provenance[ch], gate_passed[ch] = r1[0], "Tier1_IDS", True
                    continue
            p2_stage0b.append(ch)
            continue

        r1 = compose_from_ids_v3(ch, bank, sn_sc)
        if r1 is not None:
            ok1, _ = gate.check(r1[0], ch, exempt_boxes=ex_boxes, check_jp_diff=False)
            if ok1:
                master_cmds[ord(ch)], provenance[ch], gate_passed[ch] = r1[0], "Tier1_IDS", True
                continue

        r2 = atomic_asm.try_assemble_curated_atomic(ch)
        if r2 is not None:
            sc_c = sn_sc.get_cmds(ch, unioned=False)
            c2, _ = optimize_synthesized_contours_v3(r2[0], ref_sc_cmds=sc_c, stem_concave_sagitta=0.85)
            ok2, _ = gate.check(c2, ch, exempt_boxes=ex_boxes, check_jp_diff=False)
            if ok2:
                master_cmds[ord(ch)], provenance[ch], gate_passed[ch] = c2, "Tier2_Atomic", True
                continue
        p2_sdf.append(ch)

    if p2_stage0b:
        print(f"[v3-MasterBank Pass 2] Stage 0b local inpainting for {len(p2_stage0b)} SC_VARIANT composites...", flush=True)
        acc0b2, fail0b2 = tier0_stage0b_local_inpaint_batch(
            p2_stage0b, skip_ext, sn_jp, sn_sc, synth, gate, exempt_boxes_map=cls_res.exempt_boxes_map
        )
        for ch, (cmds, tag, _) in acc0b2.items():
            master_cmds[ord(ch)], provenance[ch], gate_passed[ch] = cmds, tag, True
        for ch in fail0b2:
            p2_sdf.append(ch)

    if p2_sdf:
        _run_sdf_split(p2_sdf, "Pass 2")

    ordered_items: List[Tuple[str, List[DrawCmd], str]] = []
    stats_map: Dict[str, dict] = {}
    for ch in master_chars:
        cmds = master_cmds.get(ord(ch), [])
        stats_map[ch] = count_outline_stats(cmds)
        ordered_items.append((ch, cmds, provenance.get(ch, "unknown")))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    master_otf = args.out_dir / "CnS-Manual-Master-Bank.otf"
    n_packed = build_standalone_cns_otf(
        args.skip_otf,
        master_otf,
        master_cmds,
        family_name="CnS Manual Master Bank",
        style_name="Bold",
        ps_name="CnSManualMasterBank-Bold",
        subroutinize=True,
    )
    assert n_packed == 104, f"Master Bank OTF must contain exact 104 glyphs, got {n_packed}"

    hashes = compute_otf_glyph_hashes(master_otf)
    (args.out_dir / "master_bank_hashes.json").write_text(
        json.dumps(
            {
                "hashes": hashes,
                "provenance": provenance,
                "gate_passed": gate_passed,
                "excluded_hard_keep_12": {f"U+{cp:04X}": ch for cp, ch in HARD_KEEP_12.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    wire_png = args.out_dir / "master_bank_wireframe.png"
    render_master_wireframe_sheet(ordered_items, wire_png, cols=10, cell_size=210)
    write_manual_edit_guide(args.out_dir / "MANUAL_EDIT_GUIDE.md", provenance, gate_passed, stats_map)

    summary = {
        "master_glyphs_packed": n_packed,
        "excluded_hard_keep_12": len(HARD_KEEP_12),
        "tier0_native": sum(v == "Tier0_Native" for v in provenance.values()),
        "tier0_local_rewrite": sum(v.startswith("Tier0_") and v != "Tier0_Native" for v in provenance.values()),
        "tier1_ids": sum(v.startswith("Tier1") for v in provenance.values()),
        "tier2_atomic": sum(v.startswith("Tier2") for v in provenance.values()),
        "tier3_deskinned_sdf": sum(v.startswith("Tier3") for v in provenance.values()),
        "gate_pass_count": sum(1 for v in gate_passed.values() if v),
        "gate_pass_rate": round(sum(1 for v in gate_passed.values() if v) / max(len(gate_passed), 1), 4),
        "zero_mid_stem_rate": round(
            sum(s["mid_stem_nodes"] == 0 for s in stats_map.values()) / max(len(stats_map), 1), 4
        ),
        "median_on_curve_nodes": int(sorted(s["on_curve_nodes"] for s in stats_map.values())[len(stats_map) // 2]),
    }
    (args.out_dir / "master_bank_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    skip_ext.close()
    print(f"[v3-MasterBank] Built {master_otf} ({n_packed} glyphs) | Summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
