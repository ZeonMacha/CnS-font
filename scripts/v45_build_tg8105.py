#!/usr/bin/env python3
"""
v4.5 Full Tonggui-8105 Build using GlobalGlyphTransformerV45 (Clean Standalone).

Fixes over v4_build_tg8105.py (from 审计v4.md):
  1. Decoupled gates: standard `StructuralGateV3` for variant classification (zero `KEEP_ALREADY_SC` drift)
     + `StructuralGateV45` (smooth `p95` ramp, no `chamfer==28.0` cliff) for synthesis acceptance.
  2. Pure v4.5 `ZHSDFSynthesizerV45` (direct SDF + segment-level curve snapping + orphan/floating speckle filter)
     for both Missing and SCVariants; skips wasteful legacy `DonorBankV3`/`AtomicStrokeAssemblerV3` init.
  3. Reports `zero_mid_share`, `total_dropped_speckles`, `snapped_segs`, `reverted_contours` in metrics.
  4. Optional `--emit_full` directly merges & verifies `CnS-SkipPro-B-<tag>-Full.otf`.
"""

from __future__ import annotations

import argparse
import json
import os
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
from src.v2.component_bank import parse_cjkvi_ids
from src.v2.curve_optimizer import count_outline_stats
from src.v2.hb_outlines import DrawCmd, SkipProOutlineExtractor, load_tg8105_table
from src.v3 import HARD_KEEP_12, HARD_KEEP_CHARS
from src.v3.hb_light import SansLightExtractor
from src.v3.otf_packager import build_standalone_cns_otf
from src.v3.structural_gate import StructuralGateV3
from src.v3.variant_classifier import classify_tg8105_v3, load_variant_policy
from src.v45.gate_v45 import StructuralGateV45
from src.v45.synthesizer_v45 import ZHSDFSynthesizerV45


def main() -> None:
    parser = argparse.ArgumentParser(description="v4.5 Full TG-8105 Build")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--noto_sans", type=Path, default=Path(os.environ.get("CNS_NOTO_SANS", "")))
    parser.add_argument("--noto_weight", type=int, default=None, help="VF wght axis (None for static TTC, 300/400 for VF)")
    parser.add_argument("--tg_dir", type=Path, default=ROOT / "data_v2" / "tg8105")
    parser.add_argument("--ids_path", type=Path, default=ROOT / "data_v2" / "ids.txt")
    parser.add_argument("--policy_tsv", type=Path, default=ROOT / "data_v3" / "variant_policy.tsv")
    parser.add_argument("--ckpt_path", type=Path, default=ROOT / "checkpoints" / "v45_zh_sdf_global.pt")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--version_tag", type=str, default="v45")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_shift_px", type=float, default=14.0)
    parser.add_argument("--emit_full", action="store_true", help="Also emit merged CnS-SkipPro-B-<tag>-Full.otf")
    args = parser.parse_args()

    if not str(args.noto_sans) or not args.noto_sans.exists():
        raise SystemExit(f"Provide valid --noto_sans path (or set CNS_NOTO_SANS). Got: {args.noto_sans}")

    t0 = time.time()
    skip_ext = SkipProOutlineExtractor(args.skip_otf)
    sn_jp = SansLightExtractor(args.noto_sans, language="ja", weight=args.noto_weight)
    sn_sc = SansLightExtractor(args.noto_sans, language="zh-Hans", weight=args.noto_weight)

    # Decoupled gates: standard V3 gate for stable classification, smooth V45 gate for synthesis acceptance
    cls_gate = StructuralGateV3(sn_jp, sn_sc)
    accept_gate = StructuralGateV45(sn_jp, sn_sc)

    tg_levels = load_tg8105_table(args.tg_dir)
    ids_map = parse_cjkvi_ids(args.ids_path)
    if len(ids_map) < 10000:
        raise RuntimeError(f"IDS table too small ({len(ids_map)} entries from {args.ids_path}).")
    policy_map = load_variant_policy(args.policy_tsv)

    print(f"[v45-Build] Loaded {len(ids_map)} IDS entries | Classifying TG-8105...", flush=True)
    cls_res = classify_tg8105_v3(tg_levels, skip_ext, sn_jp, sn_sc, cls_gate, ids_map, policy_map)
    print(
        f"[v45-Build] Classification: missing={len(cls_res.missing)} | sc_variant={len(cls_res.sc_variant)} | "
        f"keep_policy={len(cls_res.keep_policy)} | keep_same_gid={len(cls_res.keep_same_gid)} | "
        f"keep_v1_v2={len(cls_res.keep_v1_v2)} | keep_already_sc={len(cls_res.keep_already_sc)}",
        flush=True,
    )
    assert len(cls_res.sc_variant) > 0, "Classifier produced 0 SC variants!"

    synth = ZHSDFSynthesizerV45(args.ckpt_path, max_shift_px=args.max_shift_px)

    missing_targets = [c for c in cls_res.missing if c not in HARD_KEEP_CHARS]
    variant_targets = [c for c in cls_res.sc_variant if c not in HARD_KEEP_CHARS]

    sc_cmds_map: Dict[str, List[DrawCmd]] = {}
    for ch in tqdm(variant_targets + missing_targets, desc="Extracting SC refs", mininterval=10):
        cmds = sn_sc.get_gid_and_cmds(ch, "zh-Hans")[1]
        if cmds:
            sc_cmds_map[ch] = cmds

    missing_cmds: Dict[int, List[DrawCmd]] = {}
    variant_cmds: Dict[int, List[DrawCmd]] = {}
    needs_review_cmds: Dict[int, List[DrawCmd]] = {}
    needs_review_info: Dict[str, dict] = {}
    manifest: Dict[str, dict] = {}
    tier_counts_m: Counter = Counter()
    tier_counts_v: Counter = Counter()
    zero_mid_cnt = 0
    total_dropped_speckles = 0
    total_snapped_segs = 0
    total_reverted_contours = 0

    def _record_accept(ch: str, cmds: List[DrawCmd], tag: str, gm: dict, is_variant: bool) -> None:
        nonlocal zero_mid_cnt, total_dropped_speckles, total_snapped_segs, total_reverted_contours
        st = count_outline_stats(cmds)
        if st["mid_stem_nodes"] == 0:
            zero_mid_cnt += 1
        total_dropped_speckles += int(gm.get("dropped_speckles", 0))
        total_snapped_segs += int(gm.get("snapped_segs", 0))
        total_reverted_contours += int(gm.get("reverted_contours", 0))
        if is_variant:
            variant_cmds[ord(ch)] = cmds
            tier_counts_v[tag.split("(")[0]] += 1
        else:
            missing_cmds[ord(ch)] = cmds
            tier_counts_m[tag.split("(")[0]] += 1
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
        needs_review_info[ch] = {
            "codepoint": f"U+{ord(ch):04X}",
            "target_font": "SCVariants" if is_variant else "Missing",
            **meta,
        }
        manifest[ch] = {
            "codepoint": f"U+{ord(ch):04X}",
            "target_font": "BlockedToReview",
            "intended_font": "SCVariants" if is_variant else "Missing",
            "gate_passed": False,
            **meta,
        }

    # --- 1. SCVariants (pure v4.5 Global SDF) ---
    print(f"[v45-Build] Synthesizing {len(variant_targets)} SCVariants...", flush=True)
    acc_v, fail_v = synth.synthesize_with_gate_retries(
        variant_targets, sc_cmds_map, accept_gate,
        exempt_boxes_map=cls_res.exempt_boxes_map, batch_size=args.batch_size, check_jp_diff=True,
    )
    for ch, (cmds, tag, meta) in acc_v.items():
        _record_accept(ch, cmds, tag, meta, True)
    for ch, meta in fail_v.items():
        _record_reject(ch, meta, True)

    # --- 2. Missing (pure v4.5 Global SDF) ---
    print(f"[v45-Build] Synthesizing {len(missing_targets)} Missing...", flush=True)
    acc_m, fail_m = synth.synthesize_with_gate_retries(
        missing_targets, sc_cmds_map, accept_gate,
        exempt_boxes_map=cls_res.exempt_boxes_map, batch_size=args.batch_size, check_jp_diff=False,
    )
    for ch, (cmds, tag, meta) in acc_m.items():
        _record_accept(ch, cmds, tag, meta, False)
    for ch, meta in fail_m.items():
        _record_reject(ch, meta, False)

    # --- 3. Package ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vtag = args.version_tag
    missing_otf = args.out_dir / f"CnS-SkipPro-B-{vtag}-Missing.otf"
    variants_otf = args.out_dir / f"CnS-SkipPro-B-{vtag}-SCVariants.otf"

    n_miss = build_standalone_cns_otf(
        args.skip_otf, missing_otf, missing_cmds,
        family_name="CnS SkipPro B Missing", style_name="Bold",
        ps_name="CnSSkipProB-Missing", subroutinize=True,
    )
    n_vars = build_standalone_cns_otf(
        args.skip_otf, variants_otf, variant_cmds,
        family_name="CnS SkipPro B SC Variants", style_name="Bold",
        ps_name="CnSSkipProB-SCVariants", subroutinize=True,
    )

    review_dir = ROOT / "manual_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / f"{vtag}_needs_review.json").write_text(
        json.dumps(needs_review_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if needs_review_cmds:
        build_standalone_cns_otf(
            args.skip_otf, review_dir / f"{vtag}_needs_review.otf", needs_review_cmds,
            family_name=f"CnS {vtag} Needs Review", style_name="Bold",
            ps_name=f"CnS{vtag}NeedsReview-Bold", subroutinize=False,
        )

    # --- 4. Verify invariants & optional full merge ---
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

        if args.emit_full:
            full_otf = args.out_dir / f"CnS-SkipPro-B-{vtag}-Full.otf"
            merge_split_fonts(args.skip_otf, missing_otf, variants_otf, full_otf, subroutinize=True)
            ext_merged = SkipProOutlineExtractor(full_otf)
            try:
                covered_by_level = {
                    lv: sum(1 for c, clv in tg_levels.items() if clv == lv and ext_merged.has_char(c))
                    for lv in (1, 2, 3)
                }
            finally:
                ext_merged.close()
        else:
            with tempfile.TemporaryDirectory() as tmpd:
                tmp_merged = Path(tmpd) / f"tmp_merged_{vtag}.otf"
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

    total_targets = len(missing_targets) + len(variant_targets)
    total_passed = n_miss + n_vars
    metrics = {
        "elapsed_sec": round(time.time() - t0, 1),
        "tonggui_total": len(tg_levels),
        "missing_targets": len(missing_targets),
        "missing_packed": n_miss,
        "variant_targets": len(variant_targets),
        "variants_packed": n_vars,
        "blocked_to_review": len(needs_review_info),
        "gate_pass_rate": round(total_passed / float(max(total_targets, 1)), 4),
        "zero_mid_share": round(zero_mid_cnt / float(max(total_passed, 1)), 4),
        "total_snapped_segs": total_snapped_segs,
        "total_reverted_contours": total_reverted_contours,
        "total_dropped_speckles": total_dropped_speckles,
        "tier_breakdown_missing": dict(tier_counts_m),
        "tier_breakdown_variants": dict(tier_counts_v),
        "keep_policy_12_tg": len(cls_res.keep_policy),
        "keep_same_gid": len(cls_res.keep_same_gid),
        "keep_v1_v2_exempt": len(cls_res.keep_v1_v2),
        "keep_already_sc": len(cls_res.keep_already_sc),
        "merged_tg_coverage_by_level": covered_by_level,
        "merged_tg_total_covered": sum(covered_by_level.values()),
        "model": f"GlobalGlyphTransformerV45 d=768 L=4 H=12 max_shift={args.max_shift_px}",
        "invariants": {
            "missing_disjoint_variants": True,
            "missing_disjoint_base": True,
            "variants_subset_base": True,
            "hard_keep_12_point_exact": True,
        },
    }
    (args.out_dir / f"{vtag}_glyph_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / f"{vtag}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[v45-Build] SUMMARY:", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
