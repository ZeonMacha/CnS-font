# 基于全局上下文符号距离场生成的 CJK 字体字形风格迁移 —— 面向日语字体的简体中文本地化

**CnS v4.4 — 技术报告（中文版）**

> **状态**：草稿 — arXiv 风格技术报告
> **代码**：[`CnS-font`](../) 仓库，分支 `new-achi-v4-t`
> **模型**：`GlobalGlyphTransformer`（36,009,267 参数），`decoder_version = v44`
> **检查点**：[`checkpoints/v44_zh_sdf_global.pt`](../../checkpoints/v44_zh_sdf_global.pt)

> 英文版见 [paper/00_overview.md](../00_overview.md)。

---

## 文档结构

本报告由 8 个可独立撰写的章节文件组成。每个文件既可独立阅读，也可合并为一份完整的 arXiv 格式论文。

| 文件 | 章节 | 内容 |
|:---|:---|:---|
| `00_overview.md` | — | 标题、摘要、文档结构、符号约定 |
| `01_intro.md` | 1 | 引言、动机（游戏字体本地化）、相关工作 |
| `02_problem.md` | 2 | 问题定义：CJK 字形覆盖、变体分类、数据管线 |
| `03_method.md` | 3 | 方法 —— `GlobalGlyphTransformer` 架构（v4.4 直接 SDF 头） |
| `04_losses.md` | 4 | 方法 —— 复合损失函数设计 |
| `05_pipeline.md` | 5 | 方法 —— 推理管线、结构门控、矢量化与后处理 |
| `06_experiments.md` | 6 | 实验：训练协议、全库结果、版本间对比 |
| `07_analysis.md` | 7 | 分析："子"族案例研究、消融实验、失效分类 |
| `08_conclusion.md` | 8 | 讨论、局限性、未来工作、结论 |

---

## 摘要

我们提出了一套面向字形批量风格迁移的神经网络系统，用于将一款日语 CJK 字体（FOT-SkipPro-B）本地化为其对应的简体中文版本，同时保留原有的艺术设计语言。核心组件 **GlobalGlyphTransformer** 是一个 3600 万参数的 CNN–Transformer–CNN 混合架构：给定由几何中性参考字体（Noto Sans CJK Light）构造的四通道"脱皮（deskinned）"输入，它直接预测目标字形的完整 256×256 符号距离场（SDF）。与先前基于局部修复（local inpainting）的方案不同——后者在部首衔接处容易产生接缝伪影与轮廓截断——v4.4 采用**直接 SDF 预测头**，彻底消除了分段式 `base_blend` 合成及其伴随的硬截断（hard-clip）不连续。模型以复合目标端到端训练，该目标包含：边界加权 SDF L1、表面法向梯度一致性、直笔杆法向对齐、软 clDice 拓扑保真、软 IoU、光流全变分正则、窄带 Eikonal 约束、曲率对齐以及带曲率 TV 平滑的曲率项。最终训练 SDF L1 达到 0.131，软 IoU 达到 0.786。在《通用规范汉字表》（Tongui-8105）全部 8,105 个目标字上，系统将 4,091 个缺失字形与 952 个简体变体重绘封装进两个独立的 OpenType CFF 字体文件，达到 8,095/8,105（99.88%）的覆盖率。一个包含 Chamfer 距离、p95 偏移、差异区域覆盖与拓扑不变量检查的结构门控，将 378 个（6.97%）低置信度候选拦截至人工复核队列。我们针对最具挑战性的"子"（zi）部首族做了详细案例研究——原字体中该部首的右向大弧度弯钩超出了模型的物理形变预算——并给出系统化的失效分类，为 v4.5 的设计路线图提供依据。

---

## 符号约定

| 符号 | 含义 |
|:---|:---|
| $S = 256$ | 画布边长（像素） |
| $\Delta = 96$ | UPEM 单位下的内边距（画布覆盖 1192 UPEM） |
| $\text{u/px} = (1000 + 2\Delta)/S = 4.65625$ | 每像素对应的 UPEM 数 |
| $T_{\text{trunc}} = 16$ px $\approx 74.5$ UPEM | SDF 截断距离 |
| $\mathbf{x} \in \mathbb{R}^{4 \times S \times S}$ | 输入：$[\phi_{\text{light}}, \phi_{\text{skel}}, \cos 2\theta, \sin 2\theta]$ |
| $\mathbf{y} \in \mathbb{R}^{1 \times S \times S}$ | 目标：SkipPro 字形的归一化 SDF |
| $\phi(\mathbf{p})$ | 点 $\mathbf{p}$ 到字形轮廓的符号距离（外正内负），在 $\pm T_{\text{trunc}}$ 处截断 |
| $\mathbf{f} \in \mathbb{R}^{2 \times S \times S}$ | 像素单位的光流位移场，上限 $\pm 8.0$ px |
| $d = 768$ | Transformer 隐层维度 |
| $L = 4$ | Transformer 编码器层数 |
| $H = 12$ | 注意力头数 |
| $P = 16 \times 16 = 256$ | patch token 数量 |
| $r_u$ | 骨架 SDF 通道的均匀半宽参数（UPEM） |
| $\text{UPEM}$ | Units Per EM（OpenType 设计单位，本字体为 1000） |
