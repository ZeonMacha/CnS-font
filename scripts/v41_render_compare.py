#!/usr/bin/env python3
"""Render side-by-side v4.0 vs v4.1 visual comparison for Missing + SCVariants."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

f_ui = ImageFont.truetype(NOTO, 28, index=2)
f_lbl = ImageFont.truetype(NOTO, 24, index=2)
f_orig = ImageFont.truetype(str(ROOT / "FOT-SkipPro-B.OTF"), 72)
f_v40_m = ImageFont.truetype(str(ROOT / "outputs/CnS-SkipPro-B-v4-Missing.otf"), 72)
f_v41_m = ImageFont.truetype(str(ROOT / "outputs/CnS-SkipPro-B-v41-Missing.otf"), 72)
f_v40_v = ImageFont.truetype(str(ROOT / "outputs/CnS-SkipPro-B-v4-SCVariants.otf"), 72)
f_v41_v = ImageFont.truetype(str(ROOT / "outputs/CnS-SkipPro-B-v41-SCVariants.otf"), 72)

im = Image.new("RGB", (1520, 1020), (255, 255, 255))
d = ImageDraw.Draw(im)

d.text((30, 20), "CnS v4.0 vs v4.1 Visual Comparison (Missing & SCVariants)", fill=(20, 20, 30), font=f_ui)

rows = [
    ("v4.0 Missing (row1)", "实举领啡义护侧倾", f_v40_m, (80, 20, 20)),
    ("v4.1 Missing (row1)", "实举领啡义护侧倾", f_v41_m, (15, 20, 30)),
    ("v4.0 Missing (row2)", "货飞丝绸伞笼龙弹嗓兴", f_v40_m, (80, 20, 20)),
    ("v4.1 Missing (row2)", "货飞丝绸伞笼龙弹嗓兴", f_v41_m, (15, 20, 30)),
    ("Orig SkipPro-B (JP)", "言直令骨微道真反食鬼", f_orig, (40, 110, 60)),
    ("v4.0 SCVariants",     "言直令骨微道真反食鬼", f_v40_v, (80, 20, 20)),
    ("v4.1 SCVariants",     "言直令骨微道真反食鬼", f_v41_v, (15, 20, 30)),
]

y = 85
for label, text, font, color in rows:
    d.text((30, y + 24), label, fill=(60, 65, 80), font=f_lbl)
    d.text((310, y), text, fill=color, font=font)
    y += 128

out = ROOT / "outputs/v41_vs_v40_compare.png"
im.save(out)
print(f"Saved {out} ({out.stat().st_size} bytes)")
