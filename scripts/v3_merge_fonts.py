#!/usr/bin/env python3
"""
v3.0 Optional Standalone Split-Font Merger (`Plan3.0.md §4.3`).

Combines `FOT-SkipPro-B.OTF` + `CnS-SkipPro-B-v3-Missing.otf` + `CnS-SkipPro-B-v3-SCVariants.otf`
into a single temporary or user-requested OTF (requires `--emit_merged` to write inside `outputs/`).
Asserts point-for-point CFF outline equality on all 12 `HARD_KEEP_12` codepoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.hb_outlines import DrawCmd, SkipProOutlineExtractor
from src.v2.stroke_assembler import _draw_cmds_to_pen
from src.v3 import HARD_KEEP_12
from src.v3.otf_packager import _rewrite_name_table


def verify_hard_keep_point_for_point(base_otf: Path, target_otf: Path) -> None:
    """Assert every `HARD_KEEP_12` codepoint has byte/point-exact CFF DrawCmds matching `base_otf`."""
    ext_base = SkipProOutlineExtractor(base_otf)
    ext_tgt = SkipProOutlineExtractor(target_otf)
    try:
        for cp, ch in HARD_KEEP_12.items():
            assert ext_base.has_char(ch), f"Expected U+{cp:04X} ({ch}) in base font {base_otf}"
            assert ext_tgt.has_char(ch), f"Missing U+{cp:04X} ({ch}) in merged font {target_otf}"
            c_base = ext_base.get_cmds(ch)
            c_tgt = ext_tgt.get_cmds(ch)
            if c_base != c_tgt:
                raise AssertionError(
                    f"HARD_KEEP_12 mismatch on U+{cp:04X} ({ch}): len {len(c_base)} vs {len(c_tgt)}"
                )
    finally:
        ext_base.close()
        ext_tgt.close()


def merge_split_fonts(
    base_otf: Path,
    missing_otf: Path,
    variants_otf: Path,
    out_otf: Path,
    kanji_fd_index: int = 11,
    subroutinize: bool = True,
) -> Dict[str, int]:
    """Merge `missing_otf` and `variants_otf` onto `base_otf` while preserving all native glyphs."""
    m_ext = SkipProOutlineExtractor(missing_otf)
    v_ext = SkipProOutlineExtractor(variants_otf)
    try:
        missing_cmds: Dict[int, List[DrawCmd]] = {
            cp: m_ext.get_cmds(chr(cp)) for cp in sorted(m_ext.cmap.keys()) if cp != 0
        }
        variant_cmds: Dict[int, List[DrawCmd]] = {
            cp: v_ext.get_cmds(chr(cp)) for cp in sorted(v_ext.cmap.keys()) if cp != 0
        }
    finally:
        m_ext.close()
        v_ext.close()

    overlap = set(missing_cmds).intersection(variant_cmds)
    if overlap:
        raise AssertionError(f"Disjoint invariant violated between Missing & SCVariants: {len(overlap)} overlaps")

    hard_leak = set(HARD_KEEP_12).intersection(set(missing_cmds) | set(variant_cmds))
    if hard_leak:
        bad = ", ".join(f"U+{cp:04X}" for cp in sorted(hard_leak))
        raise AssertionError(f"HARD_KEEP_12 invariant violated in split fonts: {bad}")

    all_inject = {**missing_cmds, **variant_cmds}
    tt = TTFont(str(base_otf))
    cff = tt["CFF "].cff
    ps_name = "CnSSkipProB-TG8105SC"
    family_name = "CnS SkipPro B TG8105 SC"
    style_name = "Bold"
    cff.fontNames = [ps_name]
    top = cff[0]
    top.ROS = ("Adobe", "Identity", 0)
    top.FamilyName = family_name
    top.FullName = f"{family_name} {style_name}"
    priv = top.FDArray[kanji_fd_index].Private

    glyph_order = list(tt.getGlyphOrder())
    next_cid = max(top.CIDCount, 23060)
    hmtx = tt["hmtx"].metrics
    vmtx = tt["vmtx"].metrics if "vmtx" in tt else None
    added_map: Dict[int, str] = {}

    for cp, cmds in sorted(all_inject.items()):
        if not cmds:
            continue
        pen = T2CharStringPen(width=1000, glyphSet=None)
        _draw_cmds_to_pen(cmds, pen)
        cs = pen.getCharString(private=priv, globalSubrs=cff.GlobalSubrs)

        bp = BoundsPen(None)
        _draw_cmds_to_pen(cmds, bp)
        bounds = bp.bounds or (50, -50, 950, 830)
        lsb = int(round(bounds[0]))
        tsb = int(round(880 - bounds[3]))

        gname = f"cid{next_cid:05d}"
        next_cid += 1
        new_gid = len(top.CharStrings.charStringsIndex)
        top.CharStrings.charStringsIndex.append(cs)
        top.CharStrings.charStrings[gname] = new_gid
        glyph_order.append(gname)
        top.charset.append(gname)
        top.FDSelect.append(kanji_fd_index)
        hmtx[gname] = (1000, lsb)
        if vmtx is not None:
            vmtx[gname] = (1000, tsb)
        added_map[cp] = gname

    top.CIDCount = next_cid + 1
    tt.setGlyphOrder(glyph_order)

    for st in tt["cmap"].tables:
        if st.isUnicode():
            for cp, gname in added_map.items():
                if st.format == 4 and cp > 0xFFFF:
                    continue
                st.cmap[cp] = gname

    _rewrite_name_table(tt, family_name=family_name, style_name=style_name, ps_name=ps_name)

    out_otf.parent.mkdir(parents=True, exist_ok=True)
    tt.save(str(out_otf))
    tt.close()

    if subroutinize and added_map:
        try:
            import cffsubr

            sub_tt = TTFont(str(out_otf))
            cffsubr.subroutinize(sub_tt)
            sub_tt.save(str(out_otf))
            sub_tt.close()
        except Exception as e:
            print(f"[Warning] cffsubr subroutinize skipped for merged font: {e}", flush=True)

    verify_hard_keep_point_for_point(base_otf, out_otf)
    return {
        "missing_injected": len(missing_cmds),
        "variants_replaced": len(variant_cmds),
        "total_injected": len(added_map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge v3 split Missing & SCVariants OTFs onto SkipPro-B")
    parser.add_argument("--base_otf", type=Path, default=ROOT / "FOT-SkipPro-B.OTF")
    parser.add_argument("--missing_otf", type=Path, default=ROOT / "outputs" / "CnS-SkipPro-B-v3-Missing.otf")
    parser.add_argument("--variants_otf", type=Path, default=ROOT / "outputs" / "CnS-SkipPro-B-v3-SCVariants.otf")
    parser.add_argument("--out_otf", type=Path, default=ROOT / "outputs" / "CnS-SkipPro-B-TG8105-SC-v3-Merged.otf")
    parser.add_argument(
        "--emit_merged",
        action="store_true",
        help="Required to emit a merged font inside outputs/ (v3 defaults to standalone split OTFs)",
    )
    parser.add_argument("--no_subroutinize", action="store_true")
    args = parser.parse_args()

    out_resolved = args.out_otf.resolve()
    outputs_dir = (ROOT / "outputs").resolve()
    if (out_resolved == outputs_dir or outputs_dir in out_resolved.parents) and not args.emit_merged:
        raise SystemExit(
            f"Refusing to write merged font inside {outputs_dir} without --emit_merged "
            "(Plan3.0.md §4 keeps Missing.otf and SCVariants.otf separate by default)."
        )

    res = merge_split_fonts(
        base_otf=args.base_otf,
        missing_otf=args.missing_otf,
        variants_otf=args.variants_otf,
        out_otf=args.out_otf,
        subroutinize=not args.no_subroutinize,
    )
    print(f"[v3-Merge] Merged -> {args.out_otf} | {res} | HARD_KEEP_12 point-exact verified!", flush=True)


if __name__ == "__main__":
    main()
