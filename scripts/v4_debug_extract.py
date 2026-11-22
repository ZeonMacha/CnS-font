#!/usr/bin/env python3
"""Debug the extract_glyph_cmds function in detail."""
import sys
import traceback
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen

def extract_glyph_cmds_debug(font_path, char):
    """Extract with full error reporting."""
    try:
        tt = TTFont(str(font_path))
        cmap = tt.getBestCmap()
        cp = ord(char)
        if cp not in cmap:
            print(f"  U+{cp:04X}: NOT in cmap")
            tt.close()
            return None
        glyph_name = cmap[cp]
        print(f"  U+{cp:04X}: glyph_name={glyph_name}")
        glyf = tt.getGlyphSet()
        glyph = glyf[glyph_name]
        print(f"  glyph type: {type(glyph)}")
        pen = RecordingPen()
        glyph.draw(pen)
        tt.close()
        n_ops = len(pen.value)
        print(f"  n_ops={n_ops}")
        if n_ops == 0:
            print("  WARNING: 0 ops from RecordingPen")
            return None

        from typing import List, Tuple
        DrawCmd = Tuple[str, List[Tuple[float, float]]]
        cmds = []
        for op, args in pen.value:
            if op == "moveTo":
                cmds.append(("moveTo", [(float(args[0]), float(args[1]))]))
            elif op == "lineTo":
                cmds.append(("lineTo", [(float(args[0]), float(args[1]))]))
            elif op == "curveTo":
                cmds.append(("curveTo", [
                    (float(args[0]), float(args[1])),
                    (float(args[2]), float(args[3])),
                    (float(args[4]), float(args[5])),
                ]))
            elif op in ("closePath", "endPath"):
                cmds.append(("closePath", []))
            elif op == "qCurveTo":
                if len(args) == 2:
                    cmds.append(("lineTo", [(float(args[0]), float(args[1]))]))
                elif len(args) >= 6:
                    for i in range(0, len(args), 2):
                        if i + 1 < len(args):
                            cmds.append(("lineTo", [(float(args[i]), float(args[i+1]))]))
            else:
                print(f"  Unknown op: {op} args={args}")
        print(f"  Total cmds: {len(cmds)}")
        return cmds if cmds else None
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        traceback.print_exc()
        return None

# Test with v4-Missing font
print("Testing v4-Missing.otf with U+7EA2 (hong):")
result = extract_glyph_cmds_debug("outputs/CnS-SkipPro-B-v4-Missing.otf", "\u7ea2")
if result:
    print(f"  SUCCESS: {len(result)} cmds")
    print(f"  First 3: {result[:3]}")
else:
    print("  FAILED")

# Test with SkipPro-B font using traditional char
print("\nTesting FOT-SkipPro-B.OTF with U+7D0D (hong traditional):")
result2 = extract_glyph_cmds_debug("FOT-SkipPro-B.OTF", "\u7d0d")
if result2:
    print(f"  SUCCESS: {len(result2)} cmds")
    print(f"  First 3: {result2[:3]}")
else:
    print("  FAILED")
