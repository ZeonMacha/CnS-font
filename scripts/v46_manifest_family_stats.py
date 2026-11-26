import json, collections
from fontTools.ttLib import TTFont
ids={}
for line in open("data_v2/ids.txt",encoding="utf-8"):
    p=line.rstrip("\n").split("\t")
    if len(p)>=3 and not line.startswith("#"): ids[p[1]]=p[2]
sk=TTFont("FOT-SkipPro-B.OTF"); cmap=sk.getBestCmap()
skc=[chr(c) for c in cmap if 0x4e00<=c<=0x9fff]
def has(ch,comps): s=ids.get(ch,""); return any(c in s for c in comps)
print("SkipPro with 纟:",[c for c in skc if has(c,"纟")][:20], " 讠:",[c for c in skc if has(c,"讠")][:20])
print("SkipPro with 糸 left:",len([c for c in skc if ids.get(c,"").startswith("⿰糹") or ids.get(c,"").startswith("⿰糸")]))
print("SkipPro with 言 left:",len([c for c in skc if ids.get(c,"").startswith("⿰言")]))
for v in ["v44","v45"]:
    m=json.load(open(f"outputs/{v}_glyph_manifest.json"))
    print("====",v,len(m))
    for comp in ["纟","讠","钅","饣","贝","车","门","马","鸟","页","见","子"]:
        cs=[c for c in m if ids.get(c,"").startswith("⿰"+comp)]
        if comp=="子": cs=[c for c in m if "子" in ids.get(c,"")]
        n=len(cs); p=sum(1 for c in cs if m[c].get("gate_passed"))
        att=collections.Counter(str(m[c].get("attempt")) for c in cs if m[c].get("gate_passed"))
        fb=sum(1 for c in cs if m[c].get("used_smooth_fallback"))
        print(comp,n,"pass",p, "att",dict(att),"smoothfb",fb)
    reasons=collections.Counter()
    for c,e in m.items():
        if not e.get("gate_passed"):
            for r in e.get("reasons",[]): reasons[r.split("=")[0].split("(")[0]]+=1
    print("reasons",reasons.most_common(15))
    keys=collections.Counter(k for e in m.values() for k in e)
    print(keys.most_common(60))
