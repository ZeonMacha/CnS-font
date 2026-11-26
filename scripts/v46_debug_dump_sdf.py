import sys, time, numpy as np, torch
sys.path.insert(0,".")
torch.set_num_threads(8)
from pathlib import Path
from src.v45.synthesizer_v45 import ZHSDFSynthesizerV45
from src.v3.hb_light import SansLightExtractor
NOTO="/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc"
sn=SansLightExtractor(NOTO,language="zh-Hans",weight=300)
syn=ZHSDFSynthesizerV45(Path("checkpoints/v45_zh_sdf_global.pt"))
chars=sys.argv[1]
cm=[sn.get_cmds(c) for c in chars]
t=time.time(); p=syn.predict_sdf_batch(cm,batch_size=8); dt=time.time()-t
print("sec/glyph",dt/len(chars))
np.savez_compressed(sys.argv[2], chars=np.array(list(chars)), sdf=np.stack(p))
