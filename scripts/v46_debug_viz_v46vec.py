import sys, time, numpy as np, cv2
sys.path.insert(0,".")
from skimage.measure import find_contours
from src.v46.vectorize_v46 import sdf_to_cmds_v46, _bez
from src.v2.hb_outlines import split_cmds_into_contours
d=np.load(sys.argv[1]); out=sys.argv[2]; xr=(10,130) if len(sys.argv)<4 else tuple(map(int,sys.argv[3].split(",")))
Z=4; cells=[]; upx=1192/256
def to_px(x,y): return ((x+96)/upx-0.25)*Z, ((976-y)/upx-0.25)*Z
for ch,s in zip(d["chars"],d["sdf"]):
    g=((s+1)*127.5).astype(np.uint8)//2+60
    img=cv2.cvtColor(cv2.resize(g,(256*Z,256*Z),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
    for c in find_contours(s,0.0):
        cv2.polylines(img,[np.round(c[:,::-1]*Z).astype(np.int32)],True,(0,0,255),1)
    t=time.time(); cmds,st=sdf_to_cmds_v46(s); dt=time.time()-t
    for cc in split_cmds_into_contours(cmds):
        pos=None; poly=[]
        for op,cp in cc:
            if op=="moveTo": pos=np.array(cp[0]); poly.append(pos)
            elif op=="lineTo": p=np.array(cp[0]); poly+= [pos+(p-pos)*t for t in np.linspace(0,1,10)]; pos=p
            elif op=="curveTo":
                q=_bez(pos,np.array(cp[0]),np.array(cp[1]),np.array(cp[2]),np.linspace(0,1,20)); poly+=list(q); pos=np.array(cp[2])
        pp=np.array([to_px(*p) for p in poly]).astype(np.int32); cv2.polylines(img,[pp],True,(0,255,255),1)
        for op,cp in cc:
            if op=="curveTo":
                for h in cp[:2]: cv2.circle(img,tuple(int(v) for v in to_px(*h)),2,(255,160,0),-1)
            if op in("lineTo","curveTo","moveTo"):
                x,y=cp[-1]; cv2.circle(img,tuple(int(v) for v in to_px(x,y)),4,(0,200,0) if op=="lineTo" else (255,0,0),-1)
    img=img[10*Z:246*Z, xr[0]*Z:xr[1]*Z].copy()
    cv2.putText(img,f"{ord(ch):X} L{st['lines']} C{st['curves']} ax{st['axis_snapped']} dev{st['max_dev_u']:.1f}/{st['p95_dev_u']:.1f} {dt*1000:.0f}ms",(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,140,255),1)
    cells.append(img)
rows=[np.hstack(cells[i:i+4]) for i in range(0,len(cells),4)]
w=max(r.shape[1] for r in rows); rows=[np.pad(r,((0,0),(0,w-r.shape[1]),(0,0)),constant_values=255) for r in rows]
cv2.imwrite(out,np.vstack(rows))
