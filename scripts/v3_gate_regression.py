#!/usr/bin/env python3
"""
Milestone M0 Regression Benchmark for StructuralGateV3 (`Plan3.0.md §2.6`).

Verifies all 5 mandatory gate assertions:
  - POS : >= 97.0% on 1,000 JP==SC original SkipPro-B glyphs
  - NEG1: <= 1.0% on 300 genuine Japanese-variant SkipPro-B glyphs
  - NEG2: <= 1.0% on near-miss wrong character swaps
  - NEG3: <= 1.0% on Traditional-passed-as-Simplified pairs
  - NEG4: == 0.0% on 300 random unrelated SkipPro-B glyphs
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.component_bank import parse_cjkvi_ids
from src.v2.hb_outlines import SkipProOutlineExtractor, load_tg8105_table
from src.v3.hb_light import SansLightExtractor
from src.v3.structural_gate import StructuralGateV3

NEAR_MISS_SWAPS = [
    ("直", "真"), ("真", "直"), ("微", "徽"), ("徽", "微"),
    ("令", "今"), ("今", "令"), ("言", "信"), ("骨", "滑"),
    ("道", "首"), ("海", "梅"), ("专", "传"), ("书", "韦"),
    ("东", "车"), ("习", "羽"),
]

TRAD_AS_SIMP = [
    ("红", "紅"), ("语", "語"), ("银", "銀"), ("饭", "飯"),
    ("车", "車"), ("贝", "貝"), ("见", "見"), ("页", "頁"),
    ("马", "馬"), ("鸟", "鳥"), ("鱼", "魚"), ("风", "風"),
    ("门", "門"), ("东", "東"), ("书", "書"), ("长", "長"),
    ("专", "專"), ("发", "發"), ("买", "買"), ("卖", "賣"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M0 StructuralGateV3 Regression Benchmark")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--noto_sans", type=Path, required=True)
    parser.add_argument("--tg_dir", type=Path, default=ROOT / "data_v2" / "tg8105")
    parser.add_argument("--ids_path", type=Path, default=ROOT / "data_v2" / "ids.txt")
    parser.add_argument("--out_json", type=Path, default=ROOT / "outputs" / "v3_m0_gate_regression.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sk = SkipProOutlineExtractor(args.skip_otf)
    sn_jp = SansLightExtractor(args.noto_sans, language="ja", weight=300)
    sn_sc = SansLightExtractor(args.noto_sans, language="zh-Hans", weight=300)
    gate = StructuralGateV3(sn_jp, sn_sc)
    tg = load_tg8105_table(args.tg_dir)
    ids_map = parse_cjkvi_ids(args.ids_path)

    same_pool = [
        c for c in tg
        if sk.has_char(c) and sn_jp.shape_gid(c, "ja") == sn_sc.shape_gid(c, "zh-Hans")
    ]
    rng = random.Random(args.seed)
    pos_sample = rng.sample(same_pool, min(1000, len(same_pool)))

    pos_ok = 0
    pos_fails: List[dict] = []
    for c in pos_sample:
        ok, met = gate.check(sk.get_cmds(c), c)
        if ok:
            pos_ok += 1
        elif len(pos_fails) < 25:
            pos_fails.append({"char": c, **met})
    pos_rate = pos_ok / float(max(len(pos_sample), 1))

    # NEG1: genuine Japanese variants in SkipPro-B (SkipPro matches JP closer than SC in diff zone)
    neg1_chars: List[str] = []
    for c in tg:
        if not sk.has_char(c) or sn_jp.shape_gid(c, "ja") == sn_sc.shape_gid(c, "zh-Hans"):
            continue
        seq = ids_map.get(c, c) + c
        if any(x in seq for x in "衣衤示礻"):
            continue
        di = gate.analyze_jp_sc_diff(c)
        if not di["has_ref"] or (di["dsc_px"] + di["djp_px"]) < 20 or di["djp_px"] < 8:
            continue
        from src.v3.hb_light import align_cmds_to_bbox, extract_clean_skeleton

        cs_cmds = gate.get_ref_bundle(c)[0]
        sk_cand = extract_clean_skeleton(align_cmds_to_bbox(sk.get_cmds(c), cs_cmds), gate.canvas_size, gate.padding)
        if float((sk_cand.dt_upem[di["d_jp"]] <= 18.0).mean()) >= 0.60:
            neg1_chars.append(c)
        if len(neg1_chars) >= 300:
            break

    neg1_pass: List[str] = []
    for c in neg1_chars:
        ok, _ = gate.check(sk.get_cmds(c), c)
        if ok:
            neg1_pass.append(c)
    neg1_rate = len(neg1_pass) / float(max(len(neg1_chars), 1))

    # NEG2: near-miss swaps
    neg2_pairs = [(t, w) for t, w in NEAR_MISS_SWAPS if sk.has_char(w)]
    neg2_pass = [f"{w}->{t}" for t, w in neg2_pairs if gate.check(sk.get_cmds(w), t)[0]]
    neg2_rate = len(neg2_pass) / float(max(len(neg2_pairs), 1))

    # NEG3: Trad-as-Simp
    neg3_pairs = [(t, w) for t, w in TRAD_AS_SIMP if sk.has_char(w)]
    neg3_pass = [f"{w}->{t}" for t, w in neg3_pairs if gate.check(sk.get_cmds(w), t)[0]]
    neg3_rate = len(neg3_pass) / float(max(len(neg3_pairs), 1))

    # NEG4: random unrelated chars
    tg_list = list(tg.keys())
    neg4_pairs: List[Tuple[str, str]] = []
    for t in rng.sample(tg_list, 300):
        w = rng.choice(same_pool)
        while w == t:
            w = rng.choice(same_pool)
        neg4_pairs.append((t, w))
    neg4_pass = [f"{w}->{t}" for t, w in neg4_pairs if gate.check(sk.get_cmds(w), t)[0]]
    neg4_rate = len(neg4_pass) / float(len(neg4_pairs))

    sk.close()
    report = {
        "pos_pass": pos_ok,
        "pos_total": len(pos_sample),
        "pos_rate": round(pos_rate, 4),
        "pos_sample_fails": pos_fails[:10],
        "neg1_jp_as_sc_pass": len(neg1_pass),
        "neg1_total": len(neg1_chars),
        "neg1_rate": round(neg1_rate, 4),
        "neg1_leaks": neg1_pass[:15],
        "neg2_near_miss_pass": len(neg2_pass),
        "neg2_total": len(neg2_pairs),
        "neg2_rate": round(neg2_rate, 4),
        "neg2_leaks": neg2_pass,
        "neg3_trad_as_simp_pass": len(neg3_pass),
        "neg3_total": len(neg3_pairs),
        "neg3_rate": round(neg3_rate, 4),
        "neg3_leaks": neg3_pass,
        "neg4_random_pass": len(neg4_pass),
        "neg4_total": len(neg4_pairs),
        "neg4_rate": round(neg4_rate, 4),
        "neg4_leaks": neg4_pass,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    all_ok = (
        (pos_rate >= 0.97)
        and (neg1_rate <= 0.01)
        and (neg2_rate <= 0.01)
        and (neg3_rate <= 0.01)
        and (neg4_rate == 0.0)
    )
    if not all_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
