#!/usr/bin/env python3
"""Check if the system Noto Sans CJK font supports variable weights."""
from fontTools.ttLib import TTFont

for path in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]:
    try:
        f = TTFont(path, fontNumber=0)
        tables = list(f.keys())
        has_fvar = "fvar" in tables
        print(f"\n{path}:")
        print(f"  Has fvar (variable): {has_fvar}")
        if has_fvar:
            for axis in f["fvar"].axes:
                print(f"    {axis.axisTag}: min={axis.minValue}, default={axis.defaultValue}, max={axis.maxValue}")
        print(f"  Key tables: {[t for t in tables if t in ('fvar', 'glyf', 'CFF ', 'cmap', 'name')]}")
        # Get font name
        if "name" in tables:
            for name_rec in f["name"].names:
                if name_rec.nameID == 4 and name_rec.platformID == 3:
                    print(f"  Font name: {name_rec.toUnicode()}")
                    break
        f.close()
    except Exception as e:
        print(f"\n{path}: ERROR - {e}")
