#!/usr/bin/env python3
"""
v4.0 Full Tonggui-8105 Build using Global-Context Transformer SDF.

Same pipeline as v3 (Tier0-Native → Tier0 → Tier1 → Tier2 → Tier3)
but Tier 3 uses GlobalGlyphTransformer instead of LocalStrokeSDFNetV3.

Outputs:
  - outputs/CnS-SkipPro-B-v4-Missing.otf
  - outputs/CnS-SkipPro-B-v4-SCVariants.otf
  - manual_review/v4_needs_review.otf/.json
  - outputs/v4_glyph_manifest.json
  - outputs/v4_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

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
    compose_from_ids_v3,
    tier0_stage0a_vector_surgery,
    tier0_stage0b_local_inpaint_batch,
)
from src.v3.otf_packager import build_standalone_cns_otf
from src.v3.structural_gate import StructuralGateV3
from src.v3.variant_classifier import classify_tg8105_v3, load_variant_policy
from src.v4.synthesizer_v4 import ZHSDFSynthesizerV4


def _load_master_state(master_otf: Path):
    from src.v2.component_bank import load_master_metadata
    cmds_map, edited, prov, gate_ok = {}, set(), {}, {}
    if not master_otf.exists():
        return cmds_map, edited, prov, gate_ok
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


def main():
    parser = argparse.ArgumentParser(description="v4.0 Full TG8105 Build")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--master_otf", type=Path, default=ROOT / "manual_review" / "CnS-Manual-Master-Bank.otf")
    parser.add_argument("--noto_sans", type=Path, required=True)
    parser.add_argument("--tg_dir", type=Path, default=ROOT / "data_v2" / "tg8105")
    parser.add_argument("--ids_path", type=Path, default=ROOT / "data_v2" / "ids.txt")
    parser.add_argument("--policy_tsv", type=Path, default=ROOT / "data_v3" / "variant_policy.tsv")
    parser.add_argument("--ckpt_path", type=Path, default=ROOT / "checkpoints" / "v4_zh_sdf_global.pt")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--noto_weight", type=int, default=None, help="Variable font wght axis (None for static TTC)")
    parser.add_argument("--version_tag", type=str, default="v4", help="Output filename tag (e.g. v4 or v41)")
    parser.add_argument("--allow_empty_ids", action="store_true", help="Allow running without full cjkvi-ids ids.txt")
    parser.add_argument("--use_legacy_v3_tiers", action="store_true", help="Enable legacy v3 MasterDirect & Tier1 splicing (not recommended for v4.1)")
    args = parser.parse_args()

    t0 = time.time()
    skip_ext = SkipProOutlineExtractor(args.skip_otf)
    sn_jp = SansLightExtractor(args.noto_sans, language="ja", weight=args.noto_weight)
    sn_sc = SansLightExtractor(args.noto_sans, language="zh-Hans", weight=args.noto_weight)
    gate = StructuralGateV3(sn_jp, sn_sc)
    tg_levels = load_tg8105_table(args.tg_dir)
    ids_map = parse_cjkvi_ids(args.ids_path)
    if not args.allow_empty_ids and len(ids_map) < 10000:
        raise RuntimeError(
            f"IDS table too small ({len(ids_map)} entries from {args.ids_path}). "
            "Download cjkvi-ids ids.txt or pass --allow_empty_ids."
        )
    print(f"[v4-Build] Loaded {len(ids_map)} IDS entries from {args.ids_path}", flush=True)
    policy_map = load_variant_policy(args.policy_tsv)

    print("[v4-Build] Classifying TG-8105...", flush=True)
    cls_res = classify_tg8105_v3(tg_levels, skip_ext, sn_jp, sn_sc, gate, ids_map, policy_map)
    print(
        f"[v4-Build] Classification: missing={len(cls_res.missing)} | sc_variant={len(cls_res.sc_variant)} | "
        f"keep_policy={len(cls_res.keep_policy)} | keep_same_gid={len(cls_res.keep_same_gid)} | "
        f"keep_v1_v2={len(cls_res.keep_v1_v2)} | keep_already_sc={len(cls_res.keep_already_sc)}",
        flush=True,
    )
    assert len(cls_res.sc_variant) > 0, (
        "Classifier produced 0 SC variants! Check sn_jp/sn_sc language binding and Noto Sans CJK TTC."
    )
    sc_variant_set = set(cls_res.sc_variant)
    h_half, v_half = measure_skippro_hv_halfwidths(skip_ext, cls_res.keep_same_gid, sample_n=150)

    bank = DonorBankV3(skip_ext, ids_map, sc_variant_set, sn_sc, h_half, v_half)
    bank.build_from_skippro()

    master_cmds_map, edited_masters, master_prov, master_gate_ok = _load_master_state(args.master_otf)
    if args.use_legacy_v3_tiers and args.master_otf.exists():
        n_m = bank.load_master_otf(args.master_otf, edited_chars=edited_masters)
        print(f"[v4-Build] Loaded {n_m} legacy Master Bank glyphs", flush=True)
    else:
        print(f"[v4-Build] Pure v4 mode: skipping un-edited v3 Master Bank (edited_masters={len(edited_masters)})", flush=True)

    atomic_asm = AtomicStrokeAssemblerV3(skip_ext, sn_sc, sc_variant_set, h_half, v_half)
    synth = ZHSDFSynthesizerV4(
        args.ckpt_path,
        d_model=args.d_model,
        n_transformer_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    # Build SC cmds map for synthesizer
    sc_cmds_map: Dict[str, List[DrawCmd]] = {}
    for ch in tqdm(cls_res.sc_variant + cls_res.missing, desc="Extracting SC refs", mininterval=10):
        cmds = sn_sc.get_gid_and_cmds(ch, "zh-Hans")[1]
        if cmds:
            sc_cmds_map[ch] = cmds

    # Disjoint target sets
    seen_m: Set[str] = set()
    missing_targets: List[str] = []
    extra_m = sorted(c for c in (master_cmds_map if args.use_legacy_v3_tiers else edited_masters) if not skip_ext.has_char(c))
    for ch in cls_res.missing + extra_m:
        if ch not in seen_m and ch not in HARD_KEEP_CHARS:
            seen_m.add(ch)
            missing_targets.append(ch)

    seen_v: Set[str] = set()
    variant_targets: List[str] = []
    for ch in cls_res.sc_variant + sorted(c for c in edited_masters if skip_ext.has_char(c) and c not in HARD_KEEP_CHARS):
        if ch not in seen_v and ch not in HARD_KEEP_CHARS:
            seen_v.add(ch)
            variant_targets.append(ch)

    print(f"[v4-Build] Missing: {len(missing_targets)} | SCVariant: {len(variant_targets)}", flush=True)

    missing_cmds: Dict[int, List[DrawCmd]] = {}
    variant_cmds: Dict[int, List[DrawCmd]] = {}
    needs_review_cmds: Dict[int, List[DrawCmd]] = {}
    needs_review_info: Dict[str, dict] = {}
    manifest: Dict[str, dict] = {}
    tier_counts_m: Counter = Counter()
    tier_counts_v: Counter = Counter()

    def _record_accept(ch, cmds, tag, gm, is_variant):
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

    def _record_reject(ch, meta, is_variant):
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

    # --- 1. SCVariants ---
    v_stage0b: List[str] = []
    v_sdf_queue: List[str] = []
    for ch in tqdm(variant_targets, desc="SCVariants Dispatch", mininterval=5):
        ex_boxes = cls_res.exempt_boxes_map.get(ch)
        if ch in edited_masters and ch in master_cmds_map:
            _record_accept(ch, master_cmds_map[ch], "Tier0_MasterEdited", {"note": "manual_edit"}, True)
            continue
        if args.use_legacy_v3_tiers:
            if ch in master_cmds_map and master_prov.get(ch, "").startswith("Tier0_") and master_prov.get(ch) != "Tier0_Native":
                ok_m, gm_m = gate.check(master_cmds_map[ch], ch, exempt_boxes=ex_boxes)
                if ok_m:
                    _record_accept(ch, master_cmds_map[ch], master_prov[ch], gm_m, True)
                    continue
            r0 = tier0_stage0a_vector_surgery(ch, skip_ext, sn_jp, sn_sc, bank, gate, exempt_boxes=ex_boxes)
            if r0 is not None:
                _record_accept(ch, r0[0], r0[1], r0[2], True)
                continue
            v_stage0b.append(ch)
        else:
            # Pure v4.3 mode: route SCVariants directly to full-glyph Tier 3 Global SDF
            # (eliminates Tier0_LocalInpaint mask-boundary seams & shattered fragments)
            v_sdf_queue.append(ch)

    if v_stage0b:
        print(f"[v4-Build] Stage 0b local inpainting for {len(v_stage0b)} SCVariants...", flush=True)
        acc0b, fail0b = tier0_stage0b_local_inpaint_batch(
            v_stage0b, skip_ext, sn_jp, sn_sc, synth, gate,
            exempt_boxes_map=cls_res.exempt_boxes_map, batch_size=16,
        )
        for ch, (cmds, tag, meta) in acc0b.items():
            _record_accept(ch, cmds, tag, meta, True)
        for ch in fail0b:
            if args.use_legacy_v3_tiers:
                r1 = compose_from_ids_v3(ch, bank, sn_sc)
                if r1 is not None:
                    ok1, gm1 = gate.check(r1[0], ch, exempt_boxes=cls_res.exempt_boxes_map.get(ch))
                    if ok1:
                        _record_accept(ch, r1[0], r1[1], gm1, True)
                        continue
            v_sdf_queue.append(ch)

    if v_sdf_queue:
        print(f"[v4-Build] Tier 3 Global SDF on {len(v_sdf_queue)} SCVariants...", flush=True)
        acc3v, fail3v = synth.synthesize_with_gate_retries(
            v_sdf_queue, sc_cmds_map, gate,
            exempt_boxes_map=cls_res.exempt_boxes_map, batch_size=16,
        )
        for ch, (cmds, tag, meta) in acc3v.items():
            _record_accept(ch, cmds, tag, meta, True)
        for ch, meta in fail3v.items():
            _record_reject(ch, meta, True)

    # --- 2. Missing ---
    m_sdf_queue: List[str] = []
    for ch in tqdm(missing_targets, desc="Missing Dispatch", mininterval=5):
        ex_boxes = cls_res.exempt_boxes_map.get(ch)
        if ch in edited_masters and ch in master_cmds_map:
            _record_accept(ch, master_cmds_map[ch], "MasterEditedDirect", {"note": "manual_edit"}, False)
            continue
        if args.use_legacy_v3_tiers:
            if ch in master_cmds_map and master_gate_ok.get(ch, False):
                ok_m, gm_m = gate.check(master_cmds_map[ch], ch, exempt_boxes=ex_boxes, check_jp_diff=False)
                if ok_m:
                    _record_accept(ch, master_cmds_map[ch], f"MasterDirect({master_prov.get(ch, '')})", gm_m, False)
                    continue
            r1 = compose_from_ids_v3(ch, bank, sn_sc)
            if r1 is not None:
                ok1, gm1 = gate.check(r1[0], ch, exempt_boxes=ex_boxes, replaced_boxes=r1[2], check_jp_diff=False)
                if ok1:
                    _record_accept(ch, r1[0], r1[1], gm1, False)
                    continue
            r2 = atomic_asm.try_assemble_curated_atomic(ch)
            if r2 is not None:
                sc_c = sc_cmds_map.get(ch, [])
                c2, opt2 = optimize_synthesized_contours_v3(r2[0], ref_sc_cmds=sc_c or None, stem_concave_sagitta=0.85)
                ok2, gm2 = gate.check(c2, ch, exempt_boxes=ex_boxes, check_jp_diff=False)
                if ok2:
                    _record_accept(ch, c2, "Tier2_Atomic", {**gm2, **opt2}, False)
                    continue
        m_sdf_queue.append(ch)

    if m_sdf_queue:
        print(f"[v4-Build] Tier 3 Global SDF on {len(m_sdf_queue)} Missing...", flush=True)
        acc3m, fail3m = synth.synthesize_with_gate_retries(
            m_sdf_queue, sc_cmds_map, gate,
            exempt_boxes_map=cls_res.exempt_boxes_map, batch_size=16, check_jp_diff=False,
        )
        for ch, (cmds, tag, meta) in acc3m.items():
            _record_accept(ch, cmds, tag, meta, False)
        for ch, meta in fail3m.items():
            _record_reject(ch, meta, False)

    # --- 3. Package ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vtag = args.version_tag
    missing_otf = args.out_dir / f"CnS-SkipPro-B-{vtag}-Missing.otf"
    variants_otf = args.out_dir / f"CnS-SkipPro-B-{vtag}-SCVariants.otf"

    n_miss = build_standalone_cns_otf(args.skip_otf, missing_otf, missing_cmds,
                                       family_name="CnS SkipPro B Missing", style_name="Bold",
                                       ps_name="CnSSkipProB-Missing", subroutinize=True)
    n_vars = build_standalone_cns_otf(args.skip_otf, variants_otf, variant_cmds,
                                       family_name="CnS SkipPro B SC Variants", style_name="Bold",
                                       ps_name="CnSSkipProB-SCVariants", subroutinize=True)

    review_dir = ROOT / "manual_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_json = review_dir / f"{vtag}_needs_review.json"
    review_json.write_text(json.dumps(needs_review_info, ensure_ascii=False, indent=2), encoding="utf-8")
    if needs_review_cmds:
        review_otf = review_dir / f"{vtag}_needs_review.otf"
        build_standalone_cns_otf(args.skip_otf, review_otf, needs_review_cmds,
                                  family_name=f"CnS {vtag} Needs Review", style_name="Bold",
                                  ps_name=f"CnS{vtag}NeedsReview-Bold", subroutinize=False)

    # --- 4. Verify invariants ---
    ext_m = SkipProOutlineExtractor(missing_otf)
    ext_v = SkipProOutlineExtractor(variants_otf)
    try:
        cmap_m = {cp for cp in ext_m.cmap if cp != 0}
        cmap_v = {cp for cp in ext_v.cmap if cp != 0}
        cmap_base = {cp for cp in skip_ext.cmap if cp != 0}
        assert cmap_m.isdisjoint(cmap_v), f"Missing/SCVariants overlap: {len(cmap_m & cmap_v)}"
        assert cmap_m.isdisjoint(cmap_base), f"Missing overlaps base: {len(cmap_m & cmap_base)}"
        assert cmap_v.issubset(cmap_base), f"SCVariants has {len(cmap_v - cmap_base)} new chars"
        assert set(HARD_KEEP_12).isdisjoint(cmap_m | cmap_v), "HARD_KEEP_12 leaked!"
        assert len(cmap_v) > 0, "SCVariants OTF is empty!"

        with tempfile.TemporaryDirectory() as tmpd:
            tmp_merged = Path(tmpd) / "tmp_merged_v4.otf"
            merge_split_fonts(args.skip_otf, missing_otf, variants_otf, tmp_merged, subroutinize=False)
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
        "model": f"GlobalGlyphTransformer d={args.d_model} L={args.n_layers} H={args.n_heads}",
        "invariants": {
            "missing_disjoint_variants": True,
            "missing_disjoint_base": True,
            "variants_subset_base": True,
            "hard_keep_12_point_exact": True,
        },
    }
    (args.out_dir / f"{vtag}_glyph_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / f"{vtag}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[v4-Build] SUMMARY:", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
