#!/usr/bin/env python3
"""Test v4 model at different scales (4, 6, 8 Transformer layers) on the 4090."""
import sys
import torch

sys.path.insert(0, ".")
from src.v4.model_v4 import GlobalGlyphTransformer
from src.v4.losses_v4 import GlobalGlyphLossV4

print("=" * 60)
print("v4.0 Scale Test: 4 / 6 / 8 Transformer Layers")
print("=" * 60)

for n_layers in [4, 6, 8]:
    model = GlobalGlyphTransformer(
        in_channels=4, canvas_size=256, d_model=768,
        n_heads=12, n_transformer_layers=n_layers,
    )
    n_params = model.count_parameters()
    model = model.to("cuda")
    model.eval()

    # Inference
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(8, 4, 256, 256, device="cuda")
    with torch.no_grad():
        pred, flow = model(x)
    peak_inf = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"layers={n_layers}: params={n_params/1e6:.1f}M | batch=8 inference peak={peak_inf:.0f} MB ({peak_inf/1024:.1f} GB)")
    del x, pred, flow
    torch.cuda.empty_cache()

    # Training (fwd + bwd)
    criterion = GlobalGlyphLossV4().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for bs in [4, 8, 12, 16]:
        try:
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, 4, 256, 256, device="cuda")
            y = torch.randn(bs, 1, 256, 256, device="cuda")
            model.train()
            pred, flow = model(x)
            loss, _ = criterion(pred, y, flow)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            peak_tr = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"  batch={bs:3d}: training peak={peak_tr:6.0f} MB ({peak_tr/1024:.1f} GB)  loss={loss.item():.4f}")
            del x, y, pred, flow, loss
        except RuntimeError:
            print(f"  batch={bs:3d}: OOM")
            break

    del model, criterion, optimizer
    torch.cuda.empty_cache()
    print()

print("SCALE TEST COMPLETE")
