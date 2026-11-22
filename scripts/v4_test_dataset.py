#!/usr/bin/env python3
"""Quick test: build v4 dataset cache and verify a sample."""
import sys
from pathlib import Path
sys.path.insert(0, ".")

from src.v4.dataset_v4 import FullGlyphDatasetV4

ds = FullGlyphDatasetV4(
    skip_otf=Path("FOT-SkipPro-B.OTF"),
    noto_sans=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    cache_dir=Path("datasets"),
    canvas_size=256,
    holdout_chars=set(),
    use_cache=False,
)
print("Dataset built:", len(ds), "chars")

# Test a few samples
for i in [0, 1, 2]:
    x, y, ch = ds[i]
    print(f"  [{i}] char='{ch}' x={tuple(x.shape)} range=[{x.min().item():.3f},{x.max().item():.3f}] "
          f"y={tuple(y.shape)} range=[{y.min().item():.3f},{y.max().item():.3f}]")

print("DATASET TEST: OK")
