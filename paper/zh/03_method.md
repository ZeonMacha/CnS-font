# 3. 方法：GlobalGlyphTransformer v4.4 架构

## 3.1 设计哲学

GlobalGlyphTransformer 是一个**混合 CNN–Transformer–CNN** 编码器-解码器：给定四通道脱皮输入，为 CJK 字形预测完整的 256×256 SDF。设计遵循三条原则：

1. **全局上下文**：CJK 字形的风格由**整字结构**决定（部首排布、比例平衡、留白分布），而非仅仅是局部笔画邻域。作用于 256 个 patch token 的 4 层 Transformer 能捕捉到纯 CNN（3×3 卷积核）无法建模的部首间关系。
2. **多尺度特征融合**：CNN 编码器在四个分辨率上产生特征（256→128→64→32→16）。Transformer 在最粗粒度（16×16 = 256 个 token，每个 768 维）上工作。CNN 解码器上采样时带有**来自所有编码器层级的跳跃连接**，包括 v4.1 新增的关键 **f0（128×128）跳跃**——它保留了 16×16 瓶颈中丢失的精细笔画细节。
3. **直接 SDF 预测（v4.4）**：解码器的空间特征与变形后的输入通道被联合送入一个**两层卷积 SDF 头**，直接输出预测 SDF。这替代了 v4.3 的分段式合成：
   ```
   v4.3: pred = (1-α)·warped_light + α·warped_skel + gate·δ_sdf
   v4.4: pred = tanh(sdf_head([spatial, warped_bases]))
   ```
   移除 `α`、`gate` 与 `δ_sdf`，即消除了截断边界处的硬截断不连续，以及截断极端曲率弯钩的 `active_gate` 死区。

## 3.2 CNN 编码器

编码器将 256×256 输入经四个阶段下采样：

```
输入:  (B, 4, 256, 256)
  │
  ├─ Stem:    Conv(4→48, 7×7, s2) + GN + SiLU + ResBlock(48)   → 48 @ 128×128
  ├─ Down1:   Conv(48→96, 3×3, s2) + GN + SiLU + ResBlock(96)  → 96 @ 64×64
  ├─ Down2:   Conv(96→192, 3×3, s2) + GN + SiLU + ResBlock(192)→ 192 @ 32×32
  └─ Down3:   Conv(192→384, 3×3, s2) + GN + SiLU + ResBlock(384)→ 384 @ 16×16
```

每个 **ResBlock** 是标准的平移等变 3×3 残差块：
```
ResBlock(x) = SiLU(x + Conv3×3(GN(SiLU(Conv3×3(x)))))
```
使用 GroupNorm（G=8）而非 BatchNorm，以保证小批量（B=16）与混合精度（BF16）下的训练稳定性。

**编码器总参数**：约 4.2M

## 3.3 Patchify 与 Transformer 编码器

在 16×16 空间层级上，特征被投影到 Transformer 隐层维度：

```
f3: (B, 384, 16, 16)
  → Conv1×1(384→768):  (B, 768, 16, 16)
  → Reshape:            (B, 256, 768)     [256 个 token，每个 768 维]
  → + Sincos PE:        (B, 256, 768)     [固定 2D 位置编码]
  → Transformer × 4:    (B, 256, 768)
```

### 位置编码

使用**固定 2D Sincos 位置编码**（非可学习）。对网格位置 $(r, c)$：
$$\text{PE}(r,c) = [\sin(rf_0), \cos(rf_0), \ldots, \sin(rf_{d/4-1}), \cos(rf_{d/4-1}), \sin(cf_0), \ldots, \sin(cf_{d/4-1}), \cos(cf_{d/4-1})]$$
频率 $f_k = 10000^{-2k/d}$。固定编码为空间局部性提供归纳偏置，且不消耗参数。

### Transformer 块

4 层均使用 **Pre-Norm**（注意力/FFN 前置 LayerNorm）：
```
PreNorm MHA:  x → LN → MultiHeadAttention(12 头, d=768) → +残差
FFN:          x → LN → Linear(768→3072) → GELU → Linear(3072→768) → +残差
```
Dropout 率 0.1；激活 GELU；`batch_first=True`，`enable_nested_tensor=False`。

**Transformer 总参数**：约 23.3M（4 层 ×（MHA ~2.2M + FFN ~3.6M + LN ~0.01M））

## 3.4 带跳跃连接的 CNN 解码器

Transformer 输出经 unpatchify 后，通过四个阶段上采样，并带有**来自所有编码器层级的跳跃连接**：

```
spatial: (B, 384, 16, 16)  [来自 unpatch_conv]
  │
  ├─ Up3: Conv(384+384→192, 3×3) + Upsample2× + ResBlock(192)  → 192 @ 32×32
  │       └── 跳跃: f3 (384 @ 16×16)，经 concat 投影
  ├─ Up2: Conv(192+192→96, 3×3) + Upsample2× + ResBlock(96)    → 96 @ 64×64
  │       └── 跳跃: f2 (96 @ 32×32)
  ├─ Up1: Conv(96+96→48, 3×3) + Upsample2× + ResBlock(48)      → 48 @ 128×128
  │       └── 跳跃: f1 (96 @ 64×64)
  └─ Up0 (v4.1+): Conv(48+48→24, 3×3) + Upsample2× + ResBlock(24) → 24 @ 256×256
          └── 跳跃: f0 (48 @ 128×128)  ← 关键：保留精细细节
```

### f0 跳跃连接（v4.1 的关键修复）

在 v4.0 中，`up0` 是步长 2 的下采样后接 4× 上采样：
```python
# v4.0（旧版）:
up0 = Conv(48→24, 3×3, s2) + GN + SiLU + Upsample4×
```
这使**空间分辨率在最终上采样前被减半**，形成 128→256 瓶颈，丢失笔画级细节。v4.1 将其替换为：
```python
# v4.1+:
up0 = Conv(48+48→24, 3×3) + GN + SiLU + Upsample2× + ResBlock(24)
#      └── cat(u1, f0) ──┘
```
**f0 跳跃**（128×128 处的 48 通道）承载了 stem 对输入的"直接视野"，保留了 16×16 Transformer 瓶颈无法分辨的亚像素笔画边缘。

## 3.5 光流头与张量变形

**光流头**预测一个 2 通道位移场：
$$\mathbf{f}(\mathbf{p}) = \tau_{\max} \cdot \tanh(\text{Conv}_{3\times3}^{24\to2}(\mathbf{u}_0(\mathbf{p})))$$
其中 $\tau_{\max} = 8.0$ px（$\approx 37.25$ UPEM）。光流场通过双线性网格采样施加于**前两个输入通道**（SDF 与骨架 SDF）：
$$\text{warped}(\mathbf{p}) = \text{interp}([\phi_{\text{light}}, \phi_{\text{skel}}],\ \mathbf{p} + \mathbf{f}(\mathbf{p}))$$

该变形使网络能够在 SDF 头做最终预测**之前**重新定位输入轮廓（例如把"子"部首的弯钩向右平移）。光流在所有版本中均**零初始化**，因此初始化时变形输入等于原始输入。

> **物理可达极限**：8 px 的光流上限意味着，距最近输入轮廓特征超过 8 px（$\approx$37 UPEM）的任何目标轮廓特征**都无法仅靠变形到达**，SDF 头必须仅凭空间特征 $\mathbf{u}_0$ 生成它们。对"子"族的极端弯钩（p99 位移 112–209 UPEM），这是重大约束——详见第 7 章。

## 3.6 直接 SDF 头（v4.4）

v4.4 的 SDF 头是一个两层卷积网络：
```python
sdf_head = Sequential(
    CNNBlock(24 + 2 → 48),           # Conv(26→48, 3×3) + GN + SiLU
    Conv2d(48 → 1, 3×3, padding=1),  # 无激活（线性输出）
)
pred_sdf = tanh(sdf_head(cat([u0, warped_bases], dim=1)))
#         └── (B, 1, 256, 256) ──┘
```

**输入**：解码器空间特征 $\mathbf{u}_0$（24 通道）与变形后的输入通道（2 通道：$\phi_{\text{light}}^{\text{warped}}$、$\phi_{\text{skel}}^{\text{warped}}$）的拼接，共 26 通道。

**输出**：单通道 SDF，取值 $[-1, +1]$（tanh 激活保证有界输出，与 SDF 截断约定兼容）。

### 为什么选择直接预测？

v4.3 的分段式合成存在三个结构性问题：

| 问题 | v4.3 机制 | v4.4 修复 |
|:---|:---|:---|
| **截断处硬截断** | `pred = clamp(base_blend + gate·δ, -1, +1)` 在 $\pm T_{\text{trunc}}$ 处产生 90° 跳变 | `tanh` 处处光滑，无硬 clamp |
| **active gate 死区** | `active_gate = sigmoid((union_dil + 0.96)·15)` → 约 107 UPEM 半径外为 0，迫使 `pred → base_blend` | 无门控；SDF 头可访问整幅画布 |
| **Blend 插值** | `base_blend = (1-α)·light + α·skel` 是两个 SDF 间的线性插值，不是合法的 SDF 运算 | 头从特征 + 变形输入学习非线性映射到 SDF |

## 3.7 参数量分解

| 组件 | 参数量 |
|:---|:---:|
| CNN 编码器（Stem + Down1–3） | ~4.2M |
| Patch 卷积（1×1） | 0.295M |
| Transformer（4 层） | ~23.3M |
| Unpatch 卷积（1×1） | 0.295M |
| CNN 解码器（Up3–Up0） | ~3.5M |
| 光流头（3×3 conv） | 0.173M |
| **SDF 头（v4.4）** | **~0.71M** |
| **总计** | **36,009,267** |

v4.4 的 SDF 头（0.71M）略大于 v4.3 的 refine 头（0.68M），原因是输出为 1 通道而内部宽度不同（输入通道数相同，均为 26）。

## 3.8 前向传播摘要

```
x (B, 4, 256, 256)
  │
  ├─ f0 = stem(x)          (B, 48, 128, 128)
  ├─ f1 = down1(f0)        (B, 96, 64, 64)
  ├─ f2 = down2(f1)        (B, 192, 32, 32)
  ├─ f3 = down3(f2)        (B, 384, 16, 16)
  │
  ├─ tokens = patch_conv(f3).reshape(B, 256, 768) + sincos_pe
  ├─ tokens = transformer(tokens)              (B, 256, 768)
  ├─ spatial = unpatch_conv(tokens.reshape(B, 768, 16, 16))  (B, 384, 16, 16)
  │
  ├─ u3 = up3(cat([spatial, f3]))   (B, 192, 32, 32)
  ├─ u2 = up2(cat([u3, f2]))        (B, 96, 64, 64)
  ├─ u1 = up1(cat([u2, f1]))        (B, 48, 128, 128)
  ├─ u0 = up0(cat([u1, f0]))        (B, 24, 256, 256)
  │
  ├─ flow = 8.0 · tanh(flow_head(u0))          (B, 2, 256, 256)
  ├─ warped = grid_sample(cat([x0, x1]), flow)  (B, 2, 256, 256)
  │
  └─ pred_sdf = tanh(sdf_head(cat([u0, warped])))  (B, 1, 256, 256)

返回: (pred_sdf, flow)
```

**推理延迟**：RTX 5070 Ti 上约 12 ms/字形（BF16，batch=16）。全库（5,421 字形）推理：约 65 秒。
