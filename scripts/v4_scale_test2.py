#!/usr/bin/env python3
"""Test v4 at larger scales to maximize 24GB VRAM utilization."""
import sys
import torch

sys.path.insert(0, ".")
from src.v4.model_v4 import GlobalGlyphTransformer
from src.v4.losses_v4 import GlobalGlyphLossV4

print("=" * 60)
print("v4.0 Extended Scale Test (targeting 16-20 GB VRAM)")
print("=" * 60)

configs = [
    ("8 layers, d=768",  dict(n_transformer_layers=8, d_model=768)),
    ("8 layers, d=1024", dict(n_transformer_layers=8, d_model=1024)),
    ("12 layers, d=768", dict(n_transformer_layers=12, d_model=768)),
    ("12 layers, d=1024", dict(n_transformer_layers=12, d_model=1024)),
    ("16 layers, d=1024", dict(n_transformer_layers=16, d_model=1024)),
]

for name, cfg in configs:
    n_heads = cfg["d_model"] // 64
    model = GlobalGlyphTransformer(
        in_channels=4, canvas_size=256,
        d_model=cfg["d_model"], n_heads=n_heads,
        n_transformer_layers=cfg["n_transformer_layers"],
    )
    n_params = model.count_parameters()
    model = model.to("cuda")
    model.eval()

    # Inference at batch=8
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(8, 4, 256, 256, device="cuda")
    try:
        with torch.no_grad():
            pred, flow = model(x)
        peak_inf = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"{name:25s}: params={n_params/1e6:.1f}M | batch=8 inf={peak_inf:.0f} MB")
        del x, pred, flow
    except RuntimeError:
        print(f"{name:25s}: params={n_params/1e6:.1f}M | batch=8 inf=OOM")
        del x
    torch.cuda.empty_cache()

    # Training: find max batch size
    criterion = GlobalGlyphLossV4().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for bs in [8, 12, 16, 24, 32]:
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
            pct = peak_tr / (24 * 1024) * 100
            print(f"  batch={bs:3d}: train={peak_tr:6.0f} MB ({peak_tr/1024:.1f} GB, {pct:.0f}%)")
            del x, y, pred, flow, loss
        except RuntimeError:
            print(f"  batch={bs:3d}: OOM")
            break

    del model, criterion, optimizer
    torch.cuda.empty_cache()
    print()

print("EXTENDED SCALE TEST COMPLETE")
