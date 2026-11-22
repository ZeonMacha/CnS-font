#!/usr/bin/env python3
"""
Milestone M4: Full Tonggui-8105 v3.0 Synthesis, Hard Gate Interception, Split Standalone OTF
Packaging, Invariant Verification, & v3-vs-v2 Showcase Generator (`Plan3.0.md §2–§5`).

Outputs:
  - `outputs/CnS-SkipPro-B-v3-Missing.otf`      (standalone missing TG chars only)
  - `outputs/CnS-SkipPro-B-v3-SCVariants.otf`   (standalone non-exempt SC variants only)
  - `manual_review/v3_needs_review.otf/.json`   (any glyph failing StructuralGateV3, blocked from output)
  - `outputs/v3_glyph_manifest.json`            (per-glyph provenance, Hausdorff/area & gate metrics)
  - `outputs/v3_metrics.json`
  - `outputs/v3_vs_v2_showcase.png`
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v3_merge_fonts import merge_split_fonts
from src.v2.component_bank import is_valid_cjk_char, parse_cjkvi_ids
from src.v2.curve_optimizer import count_outline_stats
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


def _load_master_state(master_otf: Path) -> Tuple[Dict[str, List[DrawCmd]], Set[str], Dict[str, str], Dict[str, bool]]:
    cmds_map: Dict[str, List[DrawCmd]] = {}
    edited: Set[str] = set()
    prov: Dict[str, str] = {}
    gate_ok: Dict[str, bool] = {}
    if not master_otf.exists():
        return cmds_map, edited, prov, gate_ok

    from src.v2.component_bank import load_master_metadata

    edited, prov = load_master_metadata(master_otf)
    meta_p = master_otf.parent / "master_bank_hashes.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        gate_ok = meta.get("gate_passed", {})

    ext = SkipProOutlineExtractor(master_otf)
    try:
        for cp in sorted(ext.cmap.keys()):
            ch = chr(cp)
            if is_valid_cjk_char(ch) and ch not in HARD_KEEP_CHARS:
                c = ext.get_cmds(ch)
                if c:
                    cmds_map[ch] = c
    finally:
        ext.close()
    return cmds_map, edited, prov, gate_ok


def render_v3_vs_v2_showcase(
    v3_merged_otf: Path,
    v2_otf: Path,
    orig_skip_otf: Path,
    noto_sans: Path,
    out_png: Path,
) -> None:
    """Render side-by-side comparison across Problems 1, 2, and 3 (`Plan3.0.md §5`)."""
    im = Image.new("RGB", (1960, 820), (252, 252, 250))
    d = ImageDraw.Draw(im)
    f_ui_title = ImageFont.truetype(str(noto_sans), 40, index=2)
    f_ui_hdr = ImageFont.truetype(str(noto_sans), 28, index=2)
    f_v3_body = ImageFont.truetype(str(v3_merged_otf), 31)
    has_v2 = v2_otf.exists()
    f_v2_body = ImageFont.truetype(str(v2_otf), 31) if has_v2 else None
    f_orig = ImageFont.truetype(str(orig_skip_otf), 31)

    d.text((36, 26), "CnS v3.0 优化实测对比 (Noto Sans Light 去皮 + 结构优先门控 + 微差异保留)", fill=(18, 22, 32), font=f_ui_title)

    sections = [
        (
            "【问题1 去皮轻骨架与新造补全】",
            "伞(4F1E) | 丌(4E0C) | 习习 | 乡音 | 卫 | 专 | 书 | 东 | 镕 | 喆 | 红绸 | 银河 | 饭香",
            True,
        ),
        (
            "【问题2 原字局部手术与弧线保护】",
            "言之有理 | 直截了当 | 令行禁止 | 脱胎换骨 | 微风细雨 | 道法自然 | 真假 | 反覆 | 饮食 | 鬼神",
            True,
        ),
        (
            "【问题3 保留原厂字形 + 12字硬锁定】",
            "衣(8863) | 衤(8864) | 礻(793B) | 社(793E) | 神(795E) | 免 | 每 | 半 | 丰 | 丐 | 艹 | 冫 | 礼初依祈",
            False,
        ),
    ]

    y = 104
    for header, sample, show_v2_diff in sections:
        d.rectangle([36, y, 1924, y + 40], fill=(42, 48, 64))
        d.text((48, y + 4), header, fill=(246, 248, 255), font=f_ui_hdr)
        y += 50
        d.text((40, y), f"v3.0: {sample}", fill=(16, 18, 24), font=f_v3_body)
        y += 52
        if show_v2_diff and f_v2_body is not None:
            d.text((40, y), f"v2.0: {sample}", fill=(95, 95, 108), font=f_v2_body)
            y += 58
        else:
            d.text((40, y), f"Orig: {sample}", fill=(45, 95, 65), font=f_orig)
            y += 58
        y += 18

    d.text(
        (36, 756),
        "输出架构: CnS-SkipPro-B-v3-Missing.otf + CnS-SkipPro-B-v3-SCVariants.otf (互不相交、原厂零破坏)",
        fill=(60, 68, 86),
        font=f_ui_hdr,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Full Tonggui-8105 v3.0 Split OTF Fonts")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--master_otf", type=Path, default=ROOT / "manual_review" / "CnS-Manual-Master-Bank.otf")
    parser.add_argument("--noto_sans", type=Path, required=True)
    parser.add_argument("--tg_dir", type=Path, default=ROOT / "data_v2" / "tg8105")
    parser.add_argument("--ids_path", type=Path, default=ROOT / "data_v2" / "ids.txt")
    parser.add_argument("--policy_tsv", type=Path, default=ROOT / "data_v3" / "variant_policy.tsv")
    parser.add_argument("--ckpt_path", type=Path, default=ROOT / "checkpoints" / "v3_zh_sdf_light.pt")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--v2_otf", type=Path, default=ROOT / "outputs" / "CnS-SkipPro-B-TG8105-SC.otf")
    args = parser.parse_args()

    t0 = time.time()
    skip_ext = SkipProOutlineExtractor(args.skip_otf)
    sn_jp = SansLightExtractor(args.noto_sans, language="ja", weight=300)
    sn_sc = SansLightExtractor(args.noto_sans, language="zh-Hans", weight=300)
    gate = StructuralGateV3(sn_jp, sn_sc)
    tg_levels = load_tg8105_table(args.tg_dir)
    ids_map = parse_cjkvi_ids(args.ids_path)
    policy_map = load_variant_policy(args.policy_tsv)

    print("[v3-Build] Classifying TG-8105 under Plan 3.0 rules...", flush=True)
    cls_res = classify_tg8105_v3(tg_levels, skip_ext, sn_jp, sn_sc, gate, ids_map, policy_map)
    sc_variant_set = set(cls_res.sc_variant)
    h_half, v_half = measure_skippro_hv_halfwidths(skip_ext, cls_res.keep_same_gid, sample_n=150)

    bank = DonorBankV3(skip_ext, ids_map, sc_variant_set, sn_sc, h_half, v_half)
    bank.build_from_skippro()

    master_cmds_map, edited_masters, master_prov, master_gate_ok = _load_master_state(args.master_otf)
    if args.master_otf.exists():
        n_m = bank.load_master_otf(args.master_otf, edited_chars=edited_masters)
        print(
            f"[v3-Build] Loaded {n_m} Master Bank glyphs from {args.master_otf} (human_edited={len(edited_masters)})",
            flush=True,
        )

    atomic_asm = AtomicStrokeAssemblerV3(skip_ext, sn_sc, sc_variant_set, h_half, v_half)
    synth = ZHSDFSynthesizerV3(args.ckpt_path)

    # Disjoint target sets for the two standalone output fonts (Plan 3.0 §4)
    seen_m: Set[str] = set()
    missing_targets: List[str] = []
    for ch in cls_res.missing + sorted(c for c in master_cmds_map if not skip_ext.has_char(c)):
        if ch not in seen_m and ch not in HARD_KEEP_CHARS:
            seen_m.add(ch)
            missing_targets.append(ch)

    seen_v: Set[str] = set()
    variant_targets: List[str] = []
    for ch in cls_res.sc_variant + sorted(
        c for c in edited_masters if skip_ext.has_char(c) and c not in HARD_KEEP_CHARS
    ):
        if ch not in seen_v and ch not in HARD_KEEP_CHARS:
            seen_v.add(ch)
            variant_targets.append(ch)

    print(
        f"[v3-Build] Missing targets: {len(missing_targets)} | SCVariant targets: {len(variant_targets)} | "
        f"Keep native: {len(cls_res.keep_policy) + len(cls_res.keep_same_gid) + len(cls_res.keep_v1_v2) + len(cls_res.keep_already_sc)}",
        flush=True,
    )

    missing_cmds: Dict[int, List[DrawCmd]] = {}
    variant_cmds: Dict[int, List[DrawCmd]] = {}
    needs_review_cmds: Dict[int, List[DrawCmd]] = {}
    needs_review_info: Dict[str, dict] = {}
    manifest: Dict[str, dict] = {}
    tier_counts_m: Counter = Counter()
    tier_counts_v: Counter = Counter()

    def _record_accept(ch: str, cmds: List[DrawCmd], tag: str, gm: dict, is_variant: bool) -> None:
        st = count_outline_stats(cmds)
        if is_variant:
            variant_cmds[ord(ch)] = cmds
            tier_counts_v[tag.split("[")[0].split("(")[0]] += 1
        else:
            missing_cmds[ord(ch)] = cmds
            tier_counts_m[tag.split("[")[0].split("(")[0]] += 1
        manifest[ch] = {
            "codepoint": f"U+{ord(ch):04X}",
            "target_font": "SCVariants" if is_variant else "Missing",
            "tier": tag,
            "gate_passed": True,
            "on_curve_nodes": st["on_curve_nodes"],
            "mid_stem_nodes": st["mid_stem_nodes"],
            **{k: v for k, v in gm.items() if k != "fallback_cmds"},
        }

    def _record_reject(ch: str, meta: dict, is_variant: bool) -> None:
        fb = meta.pop("fallback_cmds", [])
        if fb:
            needs_review_cmds[ord(ch)] = fb
        needs_review_info[ch] = {"codepoint": f"U+{ord(ch):04X}", "target_font": "SCVariants" if is_variant else "Missing", **meta}
        manifest[ch] = {
            "codepoint": f"U+{ord(ch):04X}",
            "target_font": "BlockedToReview",
            "intended_font": "SCVariants" if is_variant else "Missing",
            "gate_passed": False,
            **meta,
        }

    # --- 1. Process SCVariants (`FOT-SkipPro-B` present glyphs with non-exempt SC differences) ---
    v_stage0b: List[str] = []
    v_sdf_queue: List[str] = []
    for ch in tqdm(variant_targets, desc="SCVariants Vector/Stage0a", mininterval=5.0):
        ex_boxes = cls_res.exempt_boxes_map.get(ch)
        if ch in edited_masters and ch in master_cmds_map:
            _record_accept(ch, master_cmds_map[ch], "Tier0_MasterEdited", {"note": "manual_edit"}, is_variant=True)
            continue

        if ch in master_cmds_map and master_prov.get(ch, "").startswith("Tier0_") and master_prov.get(ch) != "Tier0_Native":
            ok_m, gm_m = gate.check(master_cmds_map[ch], ch, exempt_boxes=ex_boxes)
            if ok_m:
                _record_accept(ch, master_cmds_map[ch], master_prov[ch], gm_m, is_variant=True)
                continue

        r0 = tier0_stage0a_vector_surgery(ch, skip_ext, sn_jp, sn_sc, bank, gate, exempt_boxes=ex_boxes)
        if r0 is not None:
            _record_accept(ch, r0[0], r0[1], r0[2], is_variant=True)
            continue

        v_stage0b.append(ch)

    if v_stage0b:
        print(f"[v3-Build] Running Stage 0b batched local SDF inpainting for {len(v_stage0b)} SCVariants...", flush=True)
        acc0b, fail0b = tier0_stage0b_local_inpaint_batch(
            v_stage0b, skip_ext, sn_jp, sn_sc, synth, gate, exempt_boxes_map=cls_res.exempt_boxes_map, batch_size=32
        )
        for ch, (cmds, tag, meta) in acc0b.items():
            _record_accept(ch, cmds, tag, meta, is_variant=True)
        for ch in fail0b:
            r1 = compose_from_ids_v3(ch, bank, sn_sc)
            if r1 is not None:
                ok1, gm1 = gate.check(r1[0], ch, exempt_boxes=cls_res.exempt_boxes_map.get(ch))
                if ok1:
                    _record_accept(ch, r1[0], r1[1], gm1, is_variant=True)
                    continue
            v_sdf_queue.append(ch)

    if v_sdf_queue:
        print(f"[v3-Build] Running Tier 3 Deskinned SDF fallback on {len(v_sdf_queue)} SCVariants...", flush=True)
        acc3v, fail3v = synth.synthesize_with_gate_retries(
            v_sdf_queue, sn_sc, gate, exempt_boxes_map=cls_res.exempt_boxes_map, batch_size=32
        )
        for ch, (cmds, tag, meta) in acc3v.items():
            _record_accept(ch, cmds, tag, meta, is_variant=True)
        for ch, meta in fail3v.items():
            _record_reject(ch, meta, is_variant=True)

    # --- 2. Process Missing (`~4,101` Tonggui chars absent from `FOT-SkipPro-B`) ---
    m_sdf_queue: List[str] = []
    for ch in tqdm(missing_targets, desc="Missing Tier1/2 Vector Assembly", mininterval=5.0):
        ex_boxes = cls_res.exempt_boxes_map.get(ch)
        if ch in edited_masters and ch in master_cmds_map:
            _record_accept(ch, master_cmds_map[ch], "MasterEditedDirect", {"note": "manual_edit"}, is_variant=False)
            continue

        if ch in master_cmds_map and master_gate_ok.get(ch, False):
            ok_m, gm_m = gate.check(master_cmds_map[ch], ch, exempt_boxes=ex_boxes, check_jp_diff=False)
            if ok_m:
                _record_accept(ch, master_cmds_map[ch], f"MasterDirect({master_prov.get(ch, '')})", gm_m, is_variant=False)
                continue

        r1 = compose_from_ids_v3(ch, bank, sn_sc)
        if r1 is not None:
            ok1, gm1 = gate.check(r1[0], ch, exempt_boxes=ex_boxes, replaced_boxes=r1[2], check_jp_diff=False)
            if ok1:
                _record_accept(ch, r1[0], r1[1], gm1, is_variant=False)
                continue

        r2 = atomic_asm.try_assemble_curated_atomic(ch)
        if r2 is not None:
            sc_c = sn_sc.get_cmds(ch, unioned=False)
            c2, opt2 = optimize_synthesized_contours_v3(r2[0], ref_sc_cmds=sc_c, stem_concave_sagitta=0.85)
            ok2, gm2 = gate.check(c2, ch, exempt_boxes=ex_boxes, check_jp_diff=False)
            if ok2:
                _record_accept(ch, c2, "Tier2_Atomic", {**gm2, **opt2}, is_variant=False)
                continue

        m_sdf_queue.append(ch)

    if m_sdf_queue:
        print(f"[v3-Build] Synthesizing {len(m_sdf_queue)} Missing chars via Tier 3 Deskinned SDF...", flush=True)
        acc3m, fail3m = synth.synthesize_with_gate_retries(
            m_sdf_queue, sn_sc, gate, exempt_boxes_map=cls_res.exempt_boxes_map, batch_size=32, check_jp_diff=False
        )
        for ch, (cmds, tag, meta) in acc3m.items():
            _record_accept(ch, cmds, tag, meta, is_variant=False)
        for ch, meta in fail3m.items():
            _record_reject(ch, meta, is_variant=False)

    # --- 3. Package Standalone Split OTFs & Needs-Review OTF ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing_otf = args.out_dir / "CnS-SkipPro-B-v3-Missing.otf"
    variants_otf = args.out_dir / "CnS-SkipPro-B-v3-SCVariants.otf"

    n_miss = build_standalone_cns_otf(
        args.skip_otf,
        missing_otf,
        missing_cmds,
        family_name="CnS SkipPro B Missing",
        style_name="Bold",
        ps_name="CnSSkipProB-Missing",
        subroutinize=True,
    )
    n_vars = build_standalone_cns_otf(
        args.skip_otf,
        variants_otf,
        variant_cmds,
        family_name="CnS SkipPro B SC Variants",
        style_name="Bold",
        ps_name="CnSSkipProB-SCVariants",
        subroutinize=True,
    )

    review_dir = ROOT / "manual_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_otf = review_dir / "v3_needs_review.otf"
    review_json = review_dir / "v3_needs_review.json"
    review_json.write_text(json.dumps(needs_review_info, ensure_ascii=False, indent=2), encoding="utf-8")
    if needs_review_cmds:
        build_standalone_cns_otf(
            args.skip_otf,
            review_otf,
            needs_review_cmds,
            family_name="CnS v3 Needs Review",
            style_name="Bold",
            ps_name="CnSv3NeedsReview-Bold",
            subroutinize=False,
        )

    # --- 4. Verify Hard Invariants on On-Disk OTFs (`Plan3.0.md §3.5, §4.2, §5`) ---
    ext_m = SkipProOutlineExtractor(missing_otf)
    ext_v = SkipProOutlineExtractor(variants_otf)
    ext_master = SkipProOutlineExtractor(args.master_otf) if args.master_otf.exists() else None
    try:
        cmap_m = {cp for cp in ext_m.cmap if cp != 0}
        cmap_v = {cp for cp in ext_v.cmap if cp != 0}
        cmap_base = {cp for cp in skip_ext.cmap if cp != 0}
        cmap_master = {cp for cp in ext_master.cmap if cp != 0} if ext_master is not None else set()

        assert cmap_m.isdisjoint(cmap_v), f"Missing and SCVariants overlap on {len(cmap_m & cmap_v)} codepoints"
        assert cmap_m.isdisjoint(cmap_base), f"Missing overlaps original SkipPro-B on {len(cmap_m & cmap_base)} codepoints"
        assert cmap_v.issubset(cmap_base), f"SCVariants has {len(cmap_v - cmap_base)} codepoints absent from SkipPro-B"
        assert set(HARD_KEEP_12).isdisjoint(cmap_m | cmap_v | cmap_master), "HARD_KEEP_12 leaked into output/master font!"
        assert len(cmap_master) == 104 or ext_master is None, f"Master Bank expected 104 glyphs, got {len(cmap_master)}"

        # Temporary merge verification + showcase rendering
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_merged = Path(tmpd) / "tmp_merged.otf"
            merge_split_fonts(args.skip_otf, missing_otf, variants_otf, tmp_merged, subroutinize=False)
            render_v3_vs_v2_showcase(tmp_merged, args.v2_otf, args.skip_otf, args.noto_sans, args.out_dir / "v3_vs_v2_showcase.png")

            ext_merged = SkipProOutlineExtractor(tmp_merged)
            try:
                covered_by_level = {
                    lv: sum(1 for c, clv in tg_levels.items() if clv == lv and ext_merged.has_char(c))
                    for lv in (1, 2, 3)
                }
            finally:
                ext_merged.close()
    finally:
        ext_m.close()
        ext_v.close()
        if ext_master is not None:
            ext_master.close()
        skip_ext.close()

    tier0_v_cnt = sum(cnt for k, cnt in tier_counts_v.items() if k.startswith("Tier0_"))
    tier0_v_share = tier0_v_cnt / float(max(len(variant_targets), 1))
    total_targets = len(missing_targets) + len(variant_targets)
    total_passed = n_miss + n_vars
    gate_pass_rate = total_passed / float(max(total_targets, 1))

    metrics = {
        "elapsed_sec": round(time.time() - t0, 1),
        "tonggui_total": len(tg_levels),
        "missing_targets": len(missing_targets),
        "missing_packed": n_miss,
        "variant_targets": len(variant_targets),
        "variants_packed": n_vars,
        "blocked_to_review": len(needs_review_info),
        "gate_pass_rate": round(gate_pass_rate, 4),
        "sc_variant_tier0_share": round(tier0_v_share, 4),
        "tier_breakdown_missing": dict(tier_counts_m),
        "tier_breakdown_variants": dict(tier_counts_v),
        "keep_policy_12_tg": len(cls_res.keep_policy),
        "keep_same_gid": len(cls_res.keep_same_gid),
        "keep_v1_v2_exempt": len(cls_res.keep_v1_v2),
        "keep_already_sc": len(cls_res.keep_already_sc),
        "merged_tg_coverage_by_level": covered_by_level,
        "merged_tg_total_covered": sum(covered_by_level.values()),
        "invariants": {
            "missing_disjoint_variants": True,
            "missing_disjoint_base": True,
            "variants_subset_base": True,
            "hard_keep_12_point_exact": True,
            "master_bank_glyph_count": 104,
        },
    }
    (args.out_dir / "v3_glyph_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "v3_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[v3-Build] SUMMARY:", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    assert tier0_v_share >= 0.75, f"SC_VARIANT Tier 0 share {tier0_v_share:.3f} < 0.75"
    assert gate_pass_rate >= 0.95, f"StructuralGateV3 pass rate {gate_pass_rate:.4f} < 0.95"


if __name__ == "__main__":
    main()
