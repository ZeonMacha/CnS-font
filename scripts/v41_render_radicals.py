#!/usr/bin/env python3
"""Render high-res verification sheet for 18 simplified-radical & variant chars."""
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v3_merge_fonts import merge_split_fonts

NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CHARS = "语饮汤绸坚劳览识银学驿乌袅仪轻车变锅"
TEN_CHARS = "举领义护飞丝绸伞龙论"

with tempfile.TemporaryDirectory() as tmpd:
    m41 = Path(tmpd) / "merged_v41.otf"
    m40 = Path(tmpd) / "merged_v40.otf"
    merge_split_fonts(
        ROOT / "FOT-SkipPro-B.OTF",
        ROOT / "outputs/CnS-SkipPro-B-v41-Missing.otf",
        ROOT / "outputs/CnS-SkipPro-B-v41-SCVariants.otf",
        m41,
        subroutinize=False,
    )
    merge_split_fonts(
        ROOT / "FOT-SkipPro-B.OTF",
        ROOT / "outputs/CnS-SkipPro-B-v4-Missing.otf",
        ROOT / "outputs/CnS-SkipPro-B-v4-SCVariants.otf",
        m40,
        subroutinize=False,
    )

    f_ui = ImageFont.truetype(NOTO, 26, index=2)
    f_lbl = ImageFont.truetype(NOTO, 22, index=2)
    f_ref = ImageFont.truetype(NOTO, 72, index=2)
    f_v40 = ImageFont.truetype(str(m40), 72)
    f_v41 = ImageFont.truetype(str(m41), 72)

    im = Image.new("RGB", (1700, 980), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((30, 18), "CnS v4.1 Verification: Simplified Radicals & Variants (Merged with Orig SkipPro-B)", fill=(20, 20, 30), font=f_ui)

    groups = [
        ("18字 简体偏旁/异体 (前9)", CHARS[:9]),
        ("18字 简体偏旁/异体 (后9)", CHARS[9:]),
        ("10字 回归验证", TEN_CHARS),
    ]

    y = 70
    for title, txt in groups:
        d.rectangle([28, y, 1670, y + 34], fill=(40, 46, 62))
        d.text((40, y + 4), title, fill=(248, 250, 255), font=f_lbl)
        y += 42
        for lbl, font, col in [
            ("Noto SC Ref", f_ref, (90, 95, 110)),
            ("v4.0 Merged", f_v40, (105, 35, 35)),
            ("v4.1 Merged", f_v41, (15, 20, 30)),
        ]:
            d.text((36, y + 22), lbl, fill=(70, 75, 90), font=f_lbl)
            d.text((230, y), txt, fill=col, font=font)
            y += 86
        y += 12

    out = ROOT / "outputs/v41_radicals_verify.png"
    im.save(out)
    print(f"Saved {out} ({out.stat().st_size} bytes)")
