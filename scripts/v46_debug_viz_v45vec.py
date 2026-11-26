import sys, numpy as np, cv2
sys.path.insert(0,".")
from skimage.measure import find_contours
from src.v45.synthesizer_v45 import ZHSDFSynthesizerV45
from src.glyph_renderer import CanvasSpec
from src.vectorizer import sdf_to_cff_commands
from src.v45.curve_constrain_v45 import optimize_synthesized_contours_v45
from src.v3.hb_light import SansLightExtractor
from src.v2.hb_outlines import split_cmds_into_contours
d=np.load(sys.argv[1]); out=sys.argv[2]
sn=SansLightExtractor("/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc",language="zh-Hans",weight=300)
spec=CanvasSpec(canvas_size=256, em_ratio=1000.0/1192.0, upem=1000, cjk_ascender=880)
Z=4; cells=[]
for ch,s in zip(d["chars"],d["sdf"]):
    crop=(slice(10,246),slice(10,130))  # left radical approx (px)
    g=((s+1)*127.5).astype(np.uint8)
    img=cv2.cvtColor(cv2.resize(g,(256*Z,256*Z),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
    for c in find_contours(s,0.0):
        pts=np.round(c[:,::-1]*Z+Z/2).astype(np.int32)
        cv2.polylines(img,[pts],True,(0,0,255),1)
    raw=sdf_to_cff_commands(s,spec,upsample_factor=3)
    clean,st=optimize_synthesized_contours_v45(raw,sn.get_cmds(str(ch)))
    upx=1192/256
    for cc in split_cmds_into_contours(clean):
        for op,cp in cc:
            if op in("lineTo","curveTo","moveTo"):
                x,y=cp[-1]; px=(x+96)/upx*Z; py=(880+96-y)/upx*Z
                cv2.circle(img,(int(px),int(py)),3,(0,180,0) if op=="lineTo" else (255,0,0),-1)
    img=img[crop[0].start*Z:crop[0].stop*Z, crop[1].start*Z:crop[1].stop*Z]
    cv2.putText(img,f"{ord(ch):X} {st}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,140,255),1)
    cells.append(img)
rows=[np.hstack(cells[i:i+4]) for i in range(0,len(cells),4)]
w=max(r.shape[1] for r in rows); rows=[np.pad(r,((0,0),(0,w-r.shape[1]),(0,0)),constant_values=255) for r in rows]
cv2.imwrite(out,np.vstack(rows))
