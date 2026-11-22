"""
v3.0 Standalone Split-Font OpenType CFF Packager (`Plan3.0.md §4.2`).

Builds independent installable CFF fonts (`Missing.otf`, `SCVariants.otf`,
104-char `CnS-Manual-Master-Bank.otf`, and `v3_needs_review.otf`) with isolated
`name` tables and CFF `fontNames` so all fonts can coexist cleanly in any OS/font tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.ttLib import TTFont

from src.v2.hb_outlines import DrawCmd
from src.v2.stroke_assembler import _draw_cmds_to_pen


def _rewrite_name_table(tt: TTFont, family_name: str, style_name: str, ps_name: str) -> None:
    name_tbl = tt["name"]
    new_names = []
    # Preserve non-identity metadata (copyright, vendor, license); drop conflicting JP CJK duplicates
    for rec in name_tbl.names:
        if rec.nameID in (1, 2, 3, 4, 6, 16, 17, 20):
            continue
        if rec.platformID == 1 and rec.langID == 11:
            continue
        if rec.platformID == 3 and rec.langID == 1041:
            continue
        new_names.append(rec)
    name_tbl.names = new_names

    full_name = f"{family_name} {style_name}".strip()
    entries = {
        1: family_name,
        2: style_name,
        3: f"3.000;CnS;{ps_name}",
        4: full_name,
        6: ps_name,
        16: family_name,
        17: style_name,
    }
    for nid, text in entries.items():
        name_tbl.setName(text, nid, 3, 1, 1033)
        name_tbl.setName(text, nid, 1, 0, 0)


def build_standalone_cns_otf(
    base_otf_path: str | Path,
    out_otf_path: str | Path,
    glyph_cmds_map: Dict[int, List[DrawCmd]],
    family_name: str,
    style_name: str = "Bold",
    ps_name: str = "CnSSkipProB-Subset",
    kanji_fd_index: int = 11,
    subroutinize: bool = True,
) -> int:
    """
    Create a standalone OpenType CFF font containing ONLY `.notdef` + `glyph_cmds_map`,
    with `1000 UPEM`, matching `FOT-SkipPro-B` metrics/Private dict and distinct family names.
    """
    tt = TTFont(str(base_otf_path))
    cff = tt["CFF "].cff
    cff.fontNames = [ps_name]
    top = cff[0]
    top.ROS = ("Adobe", "Identity", 0)
    top.FamilyName = family_name
    top.FullName = f"{family_name} {style_name}".strip()
    priv = top.FDArray[kanji_fd_index].Private

    notdef_cs = top.CharStrings[".notdef"]
    notdef_hm = tt["hmtx"].metrics[".notdef"]
    notdef_vm = tt["vmtx"].metrics[".notdef"] if "vmtx" in tt else (1000, 120)

    new_order = [".notdef"]
    new_charset = [".notdef"]
    new_fdselect = [0]
    new_cs_dict = {".notdef": notdef_cs}
    new_hmtx = {".notdef": notdef_hm}
    new_vmtx = {".notdef": notdef_vm}
    new_cmap: Dict[int, str] = {}

    cid_num = 1
    for cp in sorted(glyph_cmds_map.keys()):
        cmds = glyph_cmds_map[cp]
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

        gname = f"cid{cid_num:05d}"
        cid_num += 1
        new_order.append(gname)
        new_charset.append(gname)
        new_fdselect.append(kanji_fd_index)
        new_cs_dict[gname] = cs
        new_hmtx[gname] = (1000, lsb)
        new_vmtx[gname] = (1000, tsb)
        new_cmap[cp] = gname

    top.CharStrings.charStrings = {}
    top.CharStrings.charStringsIndex.items = []
    for idx, gname in enumerate(new_order):
        top.CharStrings.charStringsIndex.items.append(new_cs_dict[gname])
        top.CharStrings.charStrings[gname] = idx

    top.charset = new_charset
    top.FDSelect.format = 3
    top.FDSelect.gidArray = new_fdselect
    top.CIDCount = len(new_order) + 1
    tt.setGlyphOrder(new_order)
    tt["hmtx"].metrics = new_hmtx
    if "vmtx" in tt:
        tt["vmtx"].metrics = new_vmtx
    if "VORG" in tt:
        del tt["VORG"]

    for st in tt["cmap"].tables:
        if st.isUnicode():
            if st.format == 4:
                st.cmap = {cp: gn for cp, gn in new_cmap.items() if cp <= 0xFFFF}
            else:
                st.cmap = dict(new_cmap)

    _rewrite_name_table(tt, family_name=family_name, style_name=style_name, ps_name=ps_name)

    for tbl in ("GSUB", "GPOS", "BASE", "kern"):
        if tbl in tt:
            del tt[tbl]

    out_p = Path(out_otf_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tt.save(str(out_p))
    tt.close()

    if subroutinize and len(new_cmap) > 0:
        try:
            import cffsubr

            sub_tt = TTFont(str(out_p))
            cffsubr.subroutinize(sub_tt)
            sub_tt.save(str(out_p))
            sub_tt.close()
        except Exception as e:
            print(f"[Warning] cffsubr subroutinize skipped for {out_p.name}: {e}", flush=True)

    return len(new_cmap)
