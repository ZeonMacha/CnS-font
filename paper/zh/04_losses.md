# 4. 方法：复合损失函数

## 4.1 设计原则

损失函数 $\mathcal{L}$ 是九项的加权和，每一项针对预测 SDF 的某一特定几何属性。设计遵循三条原则：

1. **多尺度监督**：损失作用于完整 256×256 SDF（为梯度计算裁去 4 px 边框），将逐像素项（SDF L1、梯度）、区域项（IoU、clDice）与积分项（曲率、光流 TV）结合。
2. **拓扑感知**：软 clDice 项显式惩罚逐像素 L1 损失无法检测的拓扑错误（笔画缺失、伪孔洞）。
3. **物理一致性**：Eikonal 与曲率项强制预测 SDF 是合法的距函数（单位梯度幅值）且轮廓光滑，避免朴素 SDF 回归中常见的多拐点抖动边界。

## 4.2 损失项

所有项都在裁剪后的 SDF（$S-8 = 248$ px）上计算，以避免基于卷积的梯度估计在边界处产生伪影。

### 项 1：边界加权 SDF L1

$$\mathcal{L}_{\text{sdf}} = \frac{1}{N} \sum_{\mathbf{p}} w(\mathbf{p}) \cdot |\hat{\phi}(\mathbf{p}) - \phi^*(\mathbf{p})|$$

其中 $w(\mathbf{p}) = 1 + \lambda_b \cdot \mathbb{1}[|\phi^*(\mathbf{p})| < 0.35]$，$\lambda_b = 8.0$。边界带（$|\phi^*| < 0.35$，约 ±0.35 × $T_{\text{trunc}}$ ≈ ±26 UPEM）获得 9 倍权重，把监督集中在几何精度最关键的轮廓附近。

**v4.4 最终值**：0.1313

### 项 2：表面法向梯度一致性

$$\mathcal{L}_{\text{grad}} = \frac{1}{N} \sum_{\mathbf{p}} m_g(\mathbf{p}) \cdot \left(|\nabla_x \hat{\phi} - \nabla_x \phi^*| + |\nabla_y \hat{\phi} - \nabla_y \phi^*|\right)$$

其中 $\nabla$ 为 Sobel 算子（3×3 核、/8 归一化），$m_g(\mathbf{p}) = \mathbb{1}[|\phi^*(\mathbf{p})| < 0.45] + 0.1$。边缘带（$|\phi^*| < 0.45$）被加权；0.1 底值保证背景区域仍有弱监督。

权重：$\lambda_g = 2.0$。

**v4.4 最终值**：0.00507

### 项 3：直笔杆法向对齐

$$\mathcal{L}_{\text{stem}} = \frac{\sum_{\mathbf{p} \in \mathcal{M}_{\text{stem}}} \left(|\hat{n}_x - n_x^*| + |\hat{n}_y - n_y^*|\right)}{|\mathcal{M}_{\text{stem}}|}$$

其中 $\mathcal{M}_{\text{stem}} = \{\mathbf{p} : |\phi^*(\mathbf{p})| < 0.30 \text{ 且 } |\nabla^2 \phi^*(\mathbf{p})| < 0.16\}$ 为近轮廓且低曲率（即直笔杆）的像素集合。$\hat{n} = \nabla\hat{\phi}/|\nabla\hat{\phi}|$ 与 $n^* = \nabla\phi^*/|\nabla\phi^*|$ 为单位表面法向。

该项强制**直笔画保持笔直**——对定义 CJK 书法风格的横/竖笔画几何至关重要。

权重：$\lambda_s = 1.6$。

**v4.4 最终值**：0.1017

### 项 4：软 clDice 拓扑保真

**软 clDice**（corner-loss Dice）损失度量预测骨架与目标骨架间的拓扑重叠：

$$\mathcal{L}_{\text{cldice}} = 1 - \frac{2 \cdot \mathcal{P} \cdot \mathcal{S}}{\mathcal{P} + \mathcal{S}}$$

其中：
- $\hat{I} = \sigma(16\hat{\phi})$，$I^* = \sigma(16\phi^*)$ 为软二值墨迹掩码（sigmoid 锐度 16）
- $\hat{S} = \text{Skel}(\hat{I})$，$S^* = \text{Skel}(I^*)$ 为**可微软骨架**（经软腐蚀/膨胀的 5 迭代形态学细化）
- $\mathcal{P} = \frac{\sum \hat{S} \cdot I^*}{\sum \hat{S}}$（精确率：预测骨架落在目标墨迹上的比例）
- $\mathcal{S} = \frac{\sum S^* \cdot \hat{I}}{\sum S^*}$（敏感度：目标骨架落在预测墨迹上的比例）

软骨架计算如下：
```
skel = 0
img1 = soft_open(img)
skel = ReLU(img - img1)
for i in 1..4:
    img = soft_erode(img)
    img1 = soft_open(img)
    delta = ReLU(img - img1)
    skel = skel + ReLU(delta - skel * delta)
```
所有运算均可微（软 max/min 通过对取反输入做 `F.max_pool2d` 实现）。

**FP32 计算**：clDice 项在 FP32（而非 BF16）下计算，以避免迭代细化中的数值不稳定。

权重：$\lambda_c = 0.8$。

**v4.4 最终值**：0.02109

### 项 5：软 IoU

$$\mathcal{L}_{\text{iou}} = 1 - \frac{\sum \hat{I} \cdot I^*}{\sum (\hat{I} + I^* - \hat{I} \cdot I^*)}$$

一种全局重叠度量，与 clDice 互补：clDice 检测**局部**拓扑错误（缺一笔），IoU 检测**全局**形状失配（整体过粗/过细）。

权重：$\lambda_{\text{iou}} = 0.6$。

**v4.4 最终值（IoU）**：0.7858

### 项 6：光流全变分 + L2

$$\mathcal{L}_{\text{flow}} = \lambda_{\text{tv}} \cdot \left(\frac{1}{N}\sum_{\mathbf{p}} |\mathbf{f}(\mathbf{p}+\hat{x}) - \mathbf{f}(\mathbf{p})| + \frac{1}{N}\sum_{\mathbf{p}} |\mathbf{f}(\mathbf{p}+\hat{y}) - \mathbf{f}(\mathbf{p})|\right) + \lambda_{\text{l2}} \cdot \frac{1}{N}\sum_{\mathbf{p}} |\mathbf{f}(\mathbf{p})|^2$$

其中 $\lambda_{\text{tv}} = 0.02$，$\lambda_{\text{l2}} = 0.002$。

TV 项鼓励**光滑的形变场**（无高频变形伪影）。L2 项正则化光流幅度，防止在 SDF 头本可处理形变时网络过度依赖变形。

**v4.4 最终 TV 值**：0.00001（对比 v4.3 的 0.04457——缩小 4,457 倍，说明直接 SDF 头需要的形变更少）

### 项 7：窄带 Eikonal

$$\mathcal{L}_{\text{eik}} = \frac{\sum_{\mathbf{p} \in \mathcal{M}_{\text{eik}}} \left(\frac{|\nabla\hat{\phi}(\mathbf{p})| - 1/T_{\text{trunc}}}{1/T_{\text{trunc}}}\right)^2}{|\mathcal{M}_{\text{eik}}|}$$

其中 $\mathcal{M}_{\text{eik}} = \{\mathbf{p} : |\phi^*(\mathbf{p})| < 0.35 \text{ 且 } |\nabla\phi^*(\mathbf{p})| > 0.5/T_{\text{trunc}}\}$。

**Eikonal 方程** $|\nabla\phi| = 1$（在归一化 SDF 空间中为 $|\nabla\phi| = 1/T_{\text{trunc}}$）是距函数的定义性性质。违背它意味着预测 SDF 不是合法的距场，会导致等值线提取伪影。

掩码排除了中轴脊线——在那里 $|\nabla\phi^*|$ 自然降为零。

**FP32 计算**。

权重：$\lambda_e = 0.25$。

**v4.4 最终值**：0.01455

### 项 8：平均曲率对齐

$$\mathcal{L}_{\text{curv}} = \frac{\sum_{\mathbf{p} \in \mathcal{M}_{\text{curv}}} |\kappa_{\hat{\phi}}(\mathbf{p}) - \kappa_{\phi^*}(\mathbf{p})|}{|\mathcal{M}_{\text{curv}}|}$$

平均曲率按 $\kappa = \nabla \cdot (\nabla\phi/|\nabla\phi|)$（单位法向的散度）计算；$\mathcal{M}_{\text{curv}} = \{\mathbf{p} : |\phi^*| < 0.30,\ |\nabla\phi^*| > 0.5/T_{\text{trunc}},\ 0.04 \leq |\nabla^2\phi^*| < 0.55\}$ 选取光滑的弯曲笔画（同时排除直笔杆与尖锐 90° 拐角）。

曲率值被截断到 $[-2, +2]$ 以保证数值稳定。

**FP32 计算**。

权重：$\lambda_k = 0.50$。

**v4.4 最终值**：0.1805

### 项 9：曲率全变分

$$\mathcal{L}_{\text{curv\_tv}} = \frac{\sum_{\text{smooth band}} |\kappa(\mathbf{p}+\hat{x}) - \kappa(\mathbf{p})| + |\kappa(\mathbf{p}+\hat{y}) - \kappa(\mathbf{p})|}{|\text{smooth band}|}$$

该项惩罚光滑弧线与直笔杆上的**曲率快速变化**，抑制 SDF 头过拟合噪声时出现的多拐点抖动。smooth band 掩码为 $|\phi^*| < 0.30$ 且 $|\nabla^2\phi^*| < 0.55$。

权重：$\lambda_{kt} = 0.15$。

**v4.4 最终值**：0.07431

## 4.3 总损失

$$\mathcal{L} = \mathcal{L}_{\text{sdf}} + 2.0\mathcal{L}_{\text{grad}} + 1.6\mathcal{L}_{\text{stem}} + 0.8\mathcal{L}_{\text{cldice}} + 0.6\mathcal{L}_{\text{iou}} + \mathcal{L}_{\text{flow}} + 0.25\mathcal{L}_{\text{eik}} + 0.50\mathcal{L}_{\text{curv}} + 0.15\mathcal{L}_{\text{curv\_tv}}$$

## 4.4 收敛行为

### Stage 1（5 epoch，冻结 CNN，训练 Transformer + SDF 头）

| Epoch | SDF L1 | IoU | Flow TV | Eikonal | Curv L1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.9651 | 0.4120 | 0.000 | 0.4381 | 0.7198 |
| 2 | 0.8282 | 0.5190 | 0.000 | 0.2960 | 0.6673 |
| 3 | 0.7956 | 0.5400 | 0.000 | 0.2607 | 0.6499 |
| 4 | 0.7718 | 0.5547 | 0.000 | 0.2436 | 0.6357 |
| 5 | 0.7527 | 0.5658 | 0.000 | 0.2329 | 0.6261 |

**关键观察**：与 v4.1/v4.3 中 Stage 1 是**零梯度死区**（SDF L1 卡在 0.807，Flow TV = 0.0）不同，v4.4 的 Stage 1 呈现**SDF L1 单调下降**（0.965 → 0.753），因为直接 `sdf_head` 非零初始化，Transformer 从第 1 个 epoch 起就收到有效梯度。

### Stage 2（35 epoch，完全端到端）

| Epoch | SDF L1 | IoU | Flow TV | Eikonal | Curv L1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.5998 | 0.6281 | 0.0035 | 0.1542 | 0.5493 |
| 5 | 0.4031 | 0.7210 | 0.0019 | 0.0803 | 0.4584 |
| 10 | 0.3479 | 0.7385 | 0.0018 | 0.0539 | 0.4115 |
| 15 | 0.2827 | 0.7565 | 0.0011 | 0.0370 | 0.3525 |
| 20 | 0.2203 | 0.7711 | 0.0008 | 0.0259 | 0.2886 |
| 25 | 0.1716 | 0.7801 | 0.0004 | 0.0191 | 0.2324 |
| 30 | 0.1408 | 0.7846 | 0.0001 | 0.0156 | 0.1933 |
| 35 | **0.1313** | **0.7858** | **0.00001** | **0.0146** | **0.1805** |

**收敛曲线**：SDF L1 单调下降，呈特征性的"先快后慢"曲线。前 10 个 epoch 贡献了总改进量的 74%（0.60 → 0.35）。Flow TV 在 Stage 2 中下降 350 倍，说明随着 SDF 头承担更多生成任务，形变场越来越平滑。

**训练耗时**：RTX 5070 Ti，BF16 混合精度，batch=16，共 3,274 秒（约 55 分钟）。

## 4.5 与 v4.3 损失轨迹对比

| 指标 | v4.3 Ep1 (S2) | v4.3 Ep35 (S2) | v4.4 Ep1 (S2) | v4.4 Ep35 (S2) | v4.4 vs v4.3 |
|:---|:---:|:---:|:---:|:---:|:---:|
| SDF L1 | 0.6214 | 0.1309 | 0.5998 | 0.1313 | 基本持平 |
| Soft IoU | 0.6582 | 0.7852 | 0.6281 | 0.7858 | v4.4 +0.0006 |
| Flow TV | 0.0233 | **0.0446** | 0.0035 | **0.00001** | **v4.4 低 4,460 倍** |
| Eikonal | 0.1187 | 0.0162 | 0.1542 | 0.0146 | v4.4 −10% |
| Curv L1 | 0.5047 | 0.1825 | 0.5493 | 0.1805 | 基本持平 |

最显著的差异是 **Flow TV 终点**：v4.3 的 flow TV 在 Stage 2 中**上升**（0.023 → 0.045），因为网络越来越依赖变形来补偿 `active_gate` 死区；v4.4 的 flow TV **单调下降**（0.0035 → 0.00001），因为直接 SDF 头无需变形即可生成新几何。
