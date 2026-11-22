#!/usr/bin/env python3
"""Debug full model forward pass step by step."""
import sys
import torch
sys.path.insert(0, ".")
from src.v4.model_v4 import GlobalGlyphTransformer

m = GlobalGlyphTransformer(4, 256, 768, 12, 4)
print("Params:", m.count_parameters())

# Check buffers
print("\nBuffers:")
for name, buf in m.named_buffers():
    print(f"  {name}: shape={tuple(buf.shape)}, numel={buf.numel()}")

# Move to CUDA
m = m.to("cuda")
m.eval()

# Check buffers after .to()
print("\nBuffers after .to(cuda):")
for name, buf in m.named_buffers():
    print(f"  {name}: shape={tuple(buf.shape)}, numel={buf.numel()}, device={buf.device}")

# Manual forward step by step
x = torch.randn(2, 4, 256, 256, device="cuda")
ch0 = x[:, 0:1]
ch1 = x[:, 1:2]

f0 = m.stem(x)
print(f"\nstem: {tuple(f0.shape)}")
f1 = m.down1(f0)
print(f"down1: {tuple(f1.shape)}")
f2 = m.down2(f1)
print(f"down2: {tuple(f2.shape)}")
f3 = m.down3(f2)
print(f"down3: {tuple(f3.shape)}")

f3_proj = m.patch_conv(f3)
print(f"patch_conv: {tuple(f3_proj.shape)}")

tokens = m._to_tokens(f3_proj)
print(f"tokens (before PE): {tuple(tokens.shape)}")

print(f"\npos_enc.pos_embed: shape={tuple(m.pos_enc.pos_embed.shape)}, device={m.pos_enc.pos_embed.device}")

tokens_pe = m.pos_enc(tokens)
print(f"tokens (after PE): {tuple(tokens_pe.shape)}")

tokens_t = m.transformer(tokens_pe)
print(f"tokens (after transformer): {tuple(tokens_t.shape)}")

spatial = m._from_tokens(tokens_t)
print(f"spatial (after unpatch): {tuple(spatial.shape)}")

spatial_c = m.unpatch_conv(spatial)
print(f"unpatch_conv: {tuple(spatial_c.shape)}")

u3 = m.up3(torch.cat([spatial_c, f3], dim=1))
print(f"up3: {tuple(u3.shape)}")
u2 = m.up2(torch.cat([u3, f2], dim=1))
print(f"up2: {tuple(u2.shape)}")
u1 = m.up1(torch.cat([u2, f1], dim=1))
print(f"up1: {tuple(u1.shape)}")
u0 = m.up0(u1)
print(f"up0: {tuple(u0.shape)}")

flow = m.max_shift_px * torch.tanh(m.flow_head(u0))
print(f"flow: {tuple(flow.shape)}")

warped = m._warp_tensor(torch.cat([ch0, ch1], dim=1), flow)
print(f"warped: {tuple(warped.shape)}")

head_out = m.refine_head(torch.cat([u0, warped], dim=1))
print(f"head_out: {tuple(head_out.shape)}")

alpha = torch.sigmoid(head_out[:, 0:1])
delta = m.delta_scale * torch.tanh(head_out[:, 1:2])
base_blend = (1.0 - alpha) * warped[:, 0:1] + alpha * warped[:, 1:2]
union_sdf = torch.maximum(ch0, ch1)
gate = torch.sigmoid((union_sdf + 0.96) * 25.0)
pred = torch.clamp(base_blend + gate * delta, -1.0, 1.0)
print(f"pred_sdf: {tuple(pred.shape)} range=[{pred.min().item():.3f}, {pred.max().item():.3f}]")

print("\nFULL FORWARD PASS: OK")
