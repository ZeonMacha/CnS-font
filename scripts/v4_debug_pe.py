#!/usr/bin/env python3
"""Debug the SincosPositionalEncoding buffer issue."""
import sys
import torch
sys.path.insert(0, ".")
from src.v4.model_v4 import SincosPositionalEncoding

pe = SincosPositionalEncoding(16, 16, 768)
print("pos_embed shape:", pe.pos_embed.shape)
print("pos_embed dtype:", pe.pos_embed.dtype)
print("pos_embed device:", pe.pos_embed.device)
print("numel:", pe.pos_embed.numel())

# Try a forward pass
x = torch.randn(2, 256, 768)
try:
    out = pe(x)
    print("Forward OK, output shape:", out.shape)
except Exception as e:
    print("Forward FAILED:", e)

# Check if the buffer survived
print("\nBuffer after init:")
for name, buf in pe.named_buffers():
    print(f"  {name}: shape={buf.shape}, numel={buf.numel()}")

# Try creating the tensor directly without register_buffer
import math
half = 768 // 4
freqs = torch.exp(torch.arange(half).float() * (-math.log(10000.0) / max(half, 1)))
rows = torch.arange(16).float()
cols = torch.arange(16).float()
gr, gc = torch.meshgrid(rows, cols, indexing="ij")
r_flat = gr.flatten()
c_flat = gc.flatten()
r_sin = torch.sin(r_flat.unsqueeze(1) * freqs.unsqueeze(0))
r_cos = torch.cos(r_flat.unsqueeze(1) * freqs.unsqueeze(0))
c_sin = torch.sin(c_flat.unsqueeze(1) * freqs.unsqueeze(0))
c_cos = torch.cos(c_flat.unsqueeze(1) * freqs.unsqueeze(0))
pe2 = torch.cat([r_sin, r_cos, c_sin, c_cos], dim=-1).unsqueeze(0)
print("\nDirect tensor shape:", pe2.shape)
print("Direct tensor numel:", pe2.numel())
