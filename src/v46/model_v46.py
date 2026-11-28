"""
v4.6 Global-Context + 32x32 Local-Window Attention Direct-SDF Generator (`GlobalGlyphTransformerV46`).

Architectural improvements over `GlobalGlyphTransformerV45` (per 审计v46_进度.md §3 & R5):
  1. Optional 5th conditioning channel (`use_radical_mask=True` -> `in_channels=5`) for
     explicit component / simplified-radical spatial masks (`纟`, `讠`, `钅`, `饣`, etc.).
  2. 32x32 Local Window Multi-Head Attention (`LocalWindowAttention2D`, window=8x8) in the
     high-res decoder (`u3` stage), giving radical-scale spatial context beyond 3x3 convs
     while keeping parameter overhead < 0.5M.
  3. Heteroscedastic Uncertainty Head (`uncertainty_head` -> `pred_logvar` [B, 1, 256, 256])
     predicting pixel-level log-scale uncertainty for direct use by the v4.6 QA gate & sampler.
  4. MC-Dropout (`SpatialDropout2d` in decoder + Transformer dropout) with `set_mc_dropout(True)`
     to enable calibrated structural diversity during multi-sample ("抽卡") inference.
  5. Seamless v4.5 -> v4.6 weight transfer helper (`load_v45_weights_into_v46`) supporting both
     4-channel and 5-channel stem expansion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.v45.model_v45 import CNNBlock, ResBlock, SincosPositionalEncoding


class LocalWindowAttention2D(nn.Module):
    """
    2D non-overlapping local-window Pre-Norm MHSA + FFN block for 32x32 feature maps.

    Shape flow:
      x: [B, C, H, W] (H=W=32, window_size=8 -> 4x4=16 windows of 64 tokens each)
      -> windows: [B * nW, W_s * W_s, C]
      -> MHSA + residual + FFN + residual
      -> out: [B, C, H, W]
    Zero-initialized residual gates (`gamma_attn`, `gamma_ffn`) allow identity-preserving
    fine-tuning when initialized from a v4.5 checkpoint.
    """

    def __init__(
        self,
        channels: int = 192,
        window_size: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )
        # Learnable LayerScale initialized small so v4.5 fine-tuning starts near-identity
        self.gamma_attn = nn.Parameter(torch.full((1, 1, channels), 1e-2))
        self.gamma_ffn = nn.Parameter(torch.full((1, 1, channels), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_pad = x
        hp, wp = x_pad.shape[2], x_pad.shape[3]
        nh, nw = hp // ws, wp // ws

        # [B, C, hp, wp] -> [B, nh, ws, nw, ws, C] -> [B*nh*nw, ws*ws, C]
        tok = (
            x_pad.permute(0, 2, 3, 1)
            .reshape(b, nh, ws, nw, ws, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b * nh * nw, ws * ws, c)
        )
        t_norm = self.norm1(tok)
        attn_out, _ = self.attn(t_norm, t_norm, t_norm, need_weights=False)
        tok = tok + self.gamma_attn * attn_out
        tok = tok + self.gamma_ffn * self.ffn(self.norm2(tok))

        # [B*nh*nw, ws*ws, C] -> [B, C, hp, wp] -> crop back to [B, C, H, W]
        out = (
            tok.reshape(b, nh, nw, ws, ws, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, hp, wp, c)
            .permute(0, 3, 1, 2)
        )
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
        return out


class GlobalGlyphTransformerV46(nn.Module):
    """
    v4.6 Hybrid CNN-Encoder + 16x16 Global Transformer + 32x32 Local-Window Attention
    + High-Res CNN-Decoder + Direct SDF & Heteroscedastic Uncertainty Heads.

    Input:  x            [B, C_in, 256, 256]  (sdf_ref, skel_sdf, cos2t, sin2t[, radical_mask])
    Output: pred_sdf     [B, 1, 256, 256]     in [-1, 1]
            flow_px      [B, 2, 256, 256]     in [-max_shift_px, +max_shift_px]
            pred_logvar  [B, 1, 256, 256]     clamped to [-6.0, 2.0] (Laplace log-scale)
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        use_radical_mask: bool = False,
        canvas_size: int = 256,
        patch_size: int = 16,
        d_model: int = 768,
        n_heads: int = 12,
        n_transformer_layers: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        mc_dropout: float = 0.08,
        max_shift_px: float = 14.0,
        use_local_attn_32: bool = True,
        local_window_size: int = 8,
    ) -> None:
        super().__init__()
        if use_radical_mask and in_channels == 4:
            in_channels = 5
        self.in_channels = in_channels
        self.use_radical_mask = use_radical_mask or (in_channels >= 5)
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.max_shift_px = max_shift_px
        self.use_local_attn_32 = use_local_attn_32

        self.grid_h = canvas_size // patch_size
        self.grid_w = canvas_size // patch_size

        c1, c2, c3, c4 = 48, 96, 192, 384

        # CNN Encoder: 256 -> 128 -> 64 -> 32 -> 16
        self.stem = nn.Sequential(CNNBlock(in_channels, c1, kernel_size=7, stride=2), ResBlock(c1))
        self.down1 = nn.Sequential(CNNBlock(c1, c2, kernel_size=3, stride=2), ResBlock(c2))
        self.down2 = nn.Sequential(CNNBlock(c2, c3, kernel_size=3, stride=2), ResBlock(c3))
        self.down3 = nn.Sequential(CNNBlock(c3, c4, kernel_size=3, stride=2), ResBlock(c4))

        # Patchify & 16x16 Global Transformer (256 tokens @ d_model=768)
        self.patch_conv = nn.Conv2d(c4, d_model, kernel_size=1, bias=False)
        self.pos_enc = SincosPositionalEncoding(self.grid_h, self.grid_w, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers, enable_nested_tensor=False
        )
        self.unpatch_conv = nn.Conv2d(d_model, c4, kernel_size=1, bias=False)

        # High-Res CNN Decoder with skip connections + MC-Dropout + 32x32 Local Window Attention
        self.up3 = nn.Sequential(
            CNNBlock(c4 * 2, c3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResBlock(c3),
        )
        self.local_attn_32 = (
            LocalWindowAttention2D(
                channels=c3,
                window_size=local_window_size,
                num_heads=6,
                mlp_ratio=2.0,
                dropout=dropout,
            )
            if use_local_attn_32
            else nn.Identity()
        )
        self.mc_drop3 = nn.Dropout2d(p=mc_dropout) if mc_dropout > 0 else nn.Identity()

        self.up2 = nn.Sequential(
            CNNBlock(c3 * 2, c2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResBlock(c2),
        )
        self.mc_drop2 = nn.Dropout2d(p=mc_dropout * 0.75) if mc_dropout > 0 else nn.Identity()

        self.up1 = nn.Sequential(
            CNNBlock(c2 * 2, c1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResBlock(c1),
        )
        self.mc_drop1 = nn.Dropout2d(p=mc_dropout * 0.5) if mc_dropout > 0 else nn.Identity()

        self.up0 = nn.Sequential(
            CNNBlock(c1 * 2, c1 // 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResBlock(c1 // 2),
        )

        # Flow, Direct SDF, and Heteroscedastic Uncertainty Heads
        c_final = c1 // 2
        self.flow_head = nn.Conv2d(c_final, 2, kernel_size=3, padding=1)
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.flow_head.bias)

        self.sdf_head = nn.Sequential(
            CNNBlock(c_final + 2, c1),
            ResBlock(c1),
            nn.Conv2d(c1, 1, kernel_size=3, padding=1),
        )

        self.uncertainty_head = nn.Sequential(
            CNNBlock(c_final + 2, c_final),
            nn.Conv2d(c_final, 1, kernel_size=3, padding=1),
        )
        # Initialize logvar around -2.5 (scale ~0.08) for stable initial NLL
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.constant_(self.uncertainty_head[-1].bias, -2.5)

    def set_mc_dropout(self, enabled: bool) -> None:
        """Toggle Dropout, Dropout2d, and MultiheadAttention dropout for MC-dropout inference sampling."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.MultiheadAttention)):
                m.train(enabled)

    @staticmethod
    def _warp_tensor(x: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        dx_norm = flow_px[:, 0] * (2.0 / max(w - 1, 1))
        dy_norm = flow_px[:, 1] * (2.0 / max(h - 1, 1))
        grid = torch.stack([xx.unsqueeze(0) + dx_norm, yy.unsqueeze(0) + dy_norm], dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # Pad 4-channel input with zeros if model expects 5 channels (radical_mask omitted)
        if x.size(1) < self.in_channels:
            pad_c = self.in_channels - x.size(1)
            x = torch.cat([x, torch.zeros((x.size(0), pad_c, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)], dim=1)

        ch0 = x[:, 0:1]  # [B, 1, 256, 256]
        ch1 = x[:, 1:2]  # [B, 1, 256, 256]

        # CNN Encoder
        f0 = self.stem(x)     # [B,  48, 128, 128]
        f1 = self.down1(f0)   # [B,  96,  64,  64]
        f2 = self.down2(f1)   # [B, 192,  32,  32]
        f3 = self.down3(f2)   # [B, 384,  16,  16]

        # Patchify -> 16x16 Global Transformer -> Unpatchify
        b = x.size(0)
        f3_proj = self.patch_conv(f3)                                            # [B, 768, 16, 16]
        tokens = f3_proj.permute(0, 2, 3, 1).reshape(b, self.grid_h * self.grid_w, self.d_model)
        tokens = self.transformer(self.pos_enc(tokens))                          # [B, 256, 768]
        spatial = tokens.reshape(b, self.grid_h, self.grid_w, self.d_model).permute(0, 3, 1, 2)
        spatial = self.unpatch_conv(spatial)                                     # [B, 384, 16, 16]

        # High-Res CNN Decoder + 32x32 Local Window Attention + MC-Dropout
        u3 = self.up3(torch.cat([spatial, f3], dim=1))   # [B, 192,  32,  32]
        u3 = self.mc_drop3(self.local_attn_32(u3))       # [B, 192,  32,  32]
        u2 = self.mc_drop2(self.up2(torch.cat([u3, f2], dim=1)))  # [B, 96, 64, 64]
        u1 = self.mc_drop1(self.up1(torch.cat([u2, f1], dim=1)))  # [B, 48, 128, 128]
        u0 = self.up0(torch.cat([u1, f0], dim=1))        # [B,  24, 256, 256]

        # Smooth Flow Warp + Direct SDF + Heteroscedastic Uncertainty
        flow_px = self.max_shift_px * torch.tanh(self.flow_head(u0))             # [B, 2, 256, 256]
        warped_bases = self._warp_tensor(torch.cat([ch0, ch1], dim=1), flow_px)  # [B, 2, 256, 256]
        head_in = torch.cat([u0, warped_bases], dim=1)                           # [B, 26, 256, 256]
        pred_sdf = torch.tanh(self.sdf_head(head_in))                            # [B, 1, 256, 256]

        if not return_uncertainty:
            return pred_sdf, flow_px

        pred_logvar = self.uncertainty_head(head_in).clamp(-6.0, 2.0)            # [B, 1, 256, 256]
        return pred_sdf, flow_px, pred_logvar

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_v45_weights_into_v46(
    model: GlobalGlyphTransformerV46,
    ckpt_path: Path,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, int]:
    """
    Load a v4.5 (or v4.6) state_dict into `GlobalGlyphTransformerV46`.
    Handles 4ch -> 5ch `stem.0.net.0.weight` expansion automatically by zero-initializing
    the extra radical-mask channel column so fine-tuning starts identically to v4.5.
    """
    raw_state = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if isinstance(raw_state, dict) and "model_state_dict" in raw_state:
        raw_state = raw_state["model_state_dict"]

    dst_state = model.state_dict()
    loaded, expanded, skipped = 0, 0, 0
    for k, v in raw_state.items():
        if k not in dst_state:
            skipped += 1
            continue
        if dst_state[k].shape == v.shape:
            dst_state[k].copy_(v)
            loaded += 1
        elif k == "stem.0.net.0.weight" and v.ndim == 4 and dst_state[k].ndim == 4:
            # [C_out, C_in_dst, kH, kW] <- [C_out, C_in_src, kH, kW]
            c_copy = min(dst_state[k].shape[1], v.shape[1])
            dst_state[k].zero_()
            dst_state[k][:, :c_copy].copy_(v[:, :c_copy])
            expanded += 1
        else:
            skipped += 1

    model.load_state_dict(dst_state)
    return {"loaded": loaded, "expanded": expanded, "skipped": skipped, "total_dst": len(dst_state)}
