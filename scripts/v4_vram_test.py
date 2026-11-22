#!/usr/bin/env python3
"""
VRAM test for v4.0 GlobalGlyphTransformer.
Runs on the remote 4090 via: python scripts/v4_vram_test.py
"""

import sys
import torch

sys.path.insert(0, ".")
from src.v4.model_v4 import GlobalGlyphTransformer

print("=" * 60)
print("v4.0 VRAM Test")
print("=" * 60)

model = GlobalGlyphTransformer(
    in_channels=4, canvas_size=256, d_model=768,
    n_heads=12, n_transformer_layers=4,
)
n = model.count_parameters()
print(f"Params: {n:,} ({n/1e6:.1f}M)")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

if not torch.cuda.is_available():
    print("CUDA not available, skipping VRAM test.")
    sys.exit(0)

device = torch.device("cuda")
model = model.to(device)
model.eval()

# Test with increasing batch sizes
for bs in [1, 2, 4, 8, 12, 16]:
    try:
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(bs, 4, 256, 256, device=device)
        with torch.no_grad():
            pred, flow = model(x)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"batch={bs:3d}: peak={peak_mb:6.0f} MB ({peak_mb/1024:.1f} GB)  "
              f"pred_range=[{pred.min().item():.3f}, {pred.max().item():.3f}]")
        del x, pred, flow
    except RuntimeError as e:
        print(f"batch={bs:3d}: OOM - {str(e)[:60]}")
        break

# Test training memory (forward + backward)
print("\n--- Training Memory (forward + backward) ---")
for bs in [1, 2, 4, 8]:
    try:
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(bs, 4, 256, 256, device=device)
        y = torch.randn(bs, 1, 256, 256, device=device)

        from src.v4.losses_v4 import GlobalGlyphLossV4
        criterion = GlobalGlyphLossV4().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        pred, flow = model(x)
        loss, metrics = criterion(pred, y, flow)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"batch={bs:3d}: peak={peak_mb:6.0f} MB ({peak_mb/1024:.1f} GB)  loss={loss.item():.4f}")
        del x, y, pred, flow, loss
    except RuntimeError as e:
        print(f"batch={bs:3d}: OOM - {str(e)[:60]}")
        break

print("\nVRAM TEST COMPLETE")
