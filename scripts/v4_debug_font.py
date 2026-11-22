#!/usr/bin/env python3
"""Debug font glyph extraction."""
import sys
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen

def main():
    font_path = sys.argv[1] if len(sys.argv) > 1 else "FOT-SkipPro-B.OTF"
    tt = TTFont(font_path)
    cmap = tt.getBestCmap()
    print(f"Font: {font_path}")
    print(f"Tables: {list(tt.keys())[:15]}")
    print(f"Has glyf: {'glyf' in tt}")
    print(f"Has CFF: {'CFF ' in tt or 'CFF2' in tt}")
    print(f"Cmap entries: {len(cmap)}")

    # Test specific codepoints
    test_cps = [0x4E00, 0x7EA2, 0x8BED, 0x94F6]  # yi, hong, yu, yin
    for cp in test_cps:
        if cp in cmap:
            gname = cmap[cp]
            print(f"  U+{cp:04X}: glyph={gname}")
            glyf = tt.getGlyphSet()
            glyph = glyf[gname]
            pen = RecordingPen()
            glyph.draw(pen)
            n_ops = len(pen.value)
            print(f"    ops={n_ops}, first 3: {pen.value[:3]}")
        else:
            print(f"  U+{cp:04X}: NOT in cmap")

    # List first 5 codepoints
    cps = sorted(cmap.keys())[:5]
    print(f"First 5 codepoints: {[f'U+{c:04X}' for c in cps]}")
    tt.close()

if __name__ == "__main__":
    main()
