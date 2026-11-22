#!/usr/bin/env python3
"""
Milestone M1 Classification & Visual Inspection Preview (`Plan3.0.md §3.4 & §3.5`).

Outputs:
  - `outputs/v3_m1_classification.json`
  - `manual_review/v3_keep_vs_replace.png` (side-by-side SkipPro | SansLight-JP | SansLight-SC)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.component_bank import parse_cjkvi_ids
from src.v2.hb_outlines import SkipProOutlineExtractor, load_tg8105_table, rasterize_cmds
from src.v3.hb_light import SansLightExtractor
from src.v3.structural_gate import StructuralGateV3
from src.v3.variant_classifier import classify_tg8105_v3, load_variant_policy


def _render_triplet(ch: str, tag: str, sk: SkipProOutlineExtractor, sn_jp: SansLightExtractor, sn_sc: SansLightExtractor, h: int = 86) -> Image.Image:
    w_one = 64
    tile = Image.new("RGB", (w_one * 3 + 12, h), (250, 250, 248))
    d = ImageDraw.Draw(tile)
    d.rectangle([0, 0, tile.width - 1, h - 1], outline=(215, 215, 220))
    d.text((6, 4), f"{ch} {ord(ch):04X} {tag}", fill=(40, 40, 50))
    cmds_list = [
        sk.get_cmds(ch),
        sn_jp.get_cmds(ch, unioned=False),
        sn_sc.get_cmds(ch, unioned=False),
    ]
    for idx, cmds in enumerate(cmds_list):
        m = rasterize_cmds(cmds, canvas_size=w_one - 4, padding=96.0)
        rgb = Image.fromarray(((~m).astype("uint8") * 255)).convert("RGB")
        tile.paste(rgb, (4 + idx * w_one, 20))
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M1 v3.0 Variant Classification & Preview")
    parser.add_argument("--skip_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--noto_sans", type=Path, required=True)
    parser.add_argument("--tg_dir", type=Path, default=ROOT / "data_v2" / "tg8105")
    parser.add_argument("--ids_path", type=Path, default=ROOT / "data_v2" / "ids.txt")
    parser.add_argument("--policy_tsv", type=Path, default=ROOT / "data_v3" / "variant_policy.tsv")
    parser.add_argument("--out_json", type=Path, default=ROOT / "outputs" / "v3_m1_classification.json")
    parser.add_argument("--out_png", type=Path, default=ROOT / "manual_review" / "v3_keep_vs_replace.png")
    args = parser.parse_args()

    sk = SkipProOutlineExtractor(args.skip_otf)
    sn_jp = SansLightExtractor(args.noto_sans, language="ja", weight=300)
    sn_sc = SansLightExtractor(args.noto_sans, language="zh-Hans", weight=300)
    gate = StructuralGateV3(sn_jp, sn_sc)
    tg = load_tg8105_table(args.tg_dir)
    ids_map = parse_cjkvi_ids(args.ids_path)
    policy_map = load_variant_policy(args.policy_tsv)

    res = classify_tg8105_v3(tg, sk, sn_jp, sn_sc, gate, ids_map, policy_map)

    summary = {
        "tonggui_total": len(tg),
        "missing_count": len(res.missing),
        "sc_variant_count": len(res.sc_variant),
        "keep_policy_12_count": len(res.keep_policy),
        "keep_same_gid_count": len(res.keep_same_gid),
        "keep_v1_v2_count": len(res.keep_v1_v2),
        "keep_already_sc_count": len(res.keep_already_sc),
        "keep_policy_chars": "".join(res.keep_policy),
        "keep_v1_v2_sample": "".join(res.keep_v1_v2[:60]),
        "keep_already_sc_sample": "".join(res.keep_already_sc[:60]),
        "sc_variant_sample": "".join(res.sc_variant[:60]),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    # Render visual sheet with 4 labeled rows
    groups: List[Tuple[str, List[str]]] = [
        ("KEEP_POLICY (SkipPro | JP | SC)", [c for c in "丐丰免冫半每礻社神艹衣衤" if sk.has_char(c)]),
        ("KEEP_V1_V2 (衣/衤/礻 minor diffs)", res.keep_v1_v2[:16]),
        ("KEEP_ALREADY_SC (SkipPro passes v3 gate)", res.keep_already_sc[:16]),
        ("SC_VARIANT (Genuine variants to rewrite)", list("言直令骨微道真反刃食鬼角象曾画户" if False else "言直令骨微道真反刃食鬼角象曾") + [c for c in res.sc_variant if c not in "言直令骨微道真反刃食鬼角象曾"][:8]),
    ]
    tw, th = 204, 86
    cols = 8
    total_rows = sum((len(chars) + cols - 1) // cols for _, chars in groups)
    header_h = 26
    sheet_h = total_rows * th + len(groups) * header_h + 16
    sheet = Image.new("RGB", (cols * tw + 16, sheet_h), (242, 243, 246))
    dr = ImageDraw.Draw(sheet)

    y = 8
    for title, chars in groups:
        dr.rectangle([8, y, cols * tw + 8, y + header_h - 2], fill=(45, 52, 68))
        dr.text((16, y + 5), title, fill=(245, 248, 255))
        y += header_h
        for i, ch in enumerate(chars):
            r, c = divmod(i, cols)
            if not sk.has_char(ch):
                continue
            tile = _render_triplet(ch, res.labels.get(ch, "")[:6], sk, sn_jp, sn_sc, h=th)
            sheet.paste(tile, (8 + c * tw, y + r * th))
        y += ((len(chars) + cols - 1) // cols) * th

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out_png)
    sk.close()
    print(f"[M1] Saved preview -> {args.out_png}", flush=True)


if __name__ == "__main__":
    main()
