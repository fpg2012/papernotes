---
title: "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images"
paper_date: 2024-03-21
date: 2026-06-20
arxiv: https://arxiv.org/abs/2403.14627
description: "提出 MVSplat，一个基于 cost volume 的 feed-forward 3DGS 模型。用 plane-sweep 构造多视角 cost volume 存储跨视图特征相似度作为几何线索来估计深度，反投影为 3D Gaussian 中心。仅用光度监督端到端训练。在 RealEstate10K 和 ACID 上以 10× 更少参数、2× 更快速度超越 pixelSplat 达到 SOTA，且跨数据集泛化更强。"
tldr: "feed-forward 3DGS 做稀疏视图重建的关键在于准确定位 Gaussian 中心。MVSplat 不用数据驱动的深度回归，而是用 MVS 的 plane-sweep cost volume 做特征匹配来推断深度——从几何对应而非特征映射中学习。结果：参数仅为 pixelSplat 的 1/10、速度 2× 更快、质量更高、而且跨数据集泛化显著更强。"
authors: ["Yuedong Chen", "Haofei Xu", "Chuanxia Zheng", "Bohan Zhuang", "Marc Pollefeys", "Andreas Geiger", "Tat-Jen Cham", "Jianfei Cai"]
affiliations: ["Monash University", "ETH Zurich", "University of Tübingen", "University of Oxford", "Nanyang Technological University", "Microsoft"]
cat_path: "cv/3d"
tags: [3dgs, novel-view-synthesis, sparse-views, feed-forward, multi-view-stereo, cost-volume]
slug: "mvsplat-efficient-3d-gaussian-splatting"
---

## 1. TLDR

MVSplat 提出一个**基于 cost volume 的 feed-forward 3DGS 模型**，从稀疏多视角图像（2 张起）中一次性预测 3D Gaussian 原语，实现新视角合成 **(P.1)**。核心思路是用 plane-sweep 构造**多视角 cost volume**，存储跨视图特征相似度作为几何线索来估计深度，再反投影为 3D Gaussian 中心。相比 pixelSplat [1]（基于概率密度深度预测），MVSplat 用 **10× 更少参数**、**2× 更快推理速度**，在 RealEstate10K 和 ACID 上达到 SOTA，且跨数据集泛化更强 **(P.2-3)**。

## 2. 作者背景

作者来自 **Monash University**（一作 Yuedong Chen, 通讯 Jianfei Cai）、**ETH Zurich**（Haofei Xu, Marc Pollefeys）、**University of Tübingen**（Andreas Geiger）、**VGG Oxford**（Chuanxia Zheng）、**NTU Singapore**（Tat-Jen Cham）以及 **Microsoft**（Marc Pollefeys）**(P.1)**。团队涵盖 3D 视觉、MVS、NeRF/3DGS 等多个子领域，Haofei Xu 和 Andreas Geiger 在 MVS/深度估计方向有深厚积累（UniMatch, MVSFormer 等）。

## 3. 相关工作背景

本文站在三条研究线的交叉点上：

1. **Feed-Forward NeRF**：pixelNeRF [49], MuRF [44], AttnRend [10] 等从稀疏视角预测辐射场。**问题**：NeRF 的逐像素体素采样导致渲染慢，且 3D 体素表示计算开销大 **(P.3)**。

2. **Feed-Forward 3DGS**：Splatter Image [37]（单视图物体级）、pixelSplat [1]（双视图，概率密度深度预测）等。**关键问题**：pixelSplat 直接用 Transformer 特征通过 MLP 回归概率密度分布再采样深度，这种"从特征到深度的映射"本质上是数据驱动的，几何先验弱，导致深度预测不可靠、大量漂浮高斯伪影，且需要 50K 步 depth regularization fine-tune 才能得到合理几何 **(P.4, P.9)**。

3. **Multi-View Stereo (MVS)**：MVSNet [48], Cost Volume Pipeline 等经典方法用 plane-sweep 构造 cost volume 来估计深度。**关键差异**：典型 MVS 需要**分离的**深度估计 + 点云融合阶段，且通常需要 GT 深度监督。MVSplat 将 cost volume 融入端到端可微的 3DGS 框架，仅用光度监督训练 **(P.4)**。

**关键 insight**：pixelSplat 失败的根源在于它学习的是一种**从图像特征到深度分布的数据驱动映射**，缺乏显式的几何对应信号。而 MVSplat 用 cost volume 存储**特征间的相对相似度**，从几何匹配中推断深度——这不仅更可靠，而且 cost volume 的相似度值**与特征绝对值尺度无关**，使得跨数据集泛化显著更强 **(P.10-11)**。

## 4. 核心方法

### 4.1 问题形式化

输入：$K$ 张稀疏视角图像 $\mathcal{I} = \{I^i\}_{i=1}^K$，$I^i \in \mathbb{R}^{H \times W \times 3}$，和对应的相机投影矩阵 $\mathcal{P} = \{P^i\}_{i=1}^K$，$P^i = K^i[\mathbf{R}^i | \mathbf{t}^i]$。

输出：3D Gaussian 参数集 $\{(\mu_j, \alpha_j, \Sigma_j, \mathbf{c}_j)\}_{j=1}^{H \times W \times K}$，其中 $\mu_j$ 为位置、$\alpha_j$ 为不透明度、$\Sigma_j$ 为协方差矩阵、$\mathbf{c}_j$ 为颜色（球谐系数）**(P.5)**。

学习目标是学习映射 $f_\theta$：

$$f_\theta: \{(I^i, P^i)\}_{i=1}^K \mapsto \{(\mu_j, \alpha_j, \Sigma_j, \mathbf{c}_j)\}_{j=1}^{H \times W \times K}$$

与 pixelSplat 每像素预测 3 个 Gaussian 不同，MVSplat **每像素只预测 1 个**，Gaussian 总数 = $H \times W \times K$。

### 4.2 模型架构

整体架构如 **Figure 2 (P.4)** 所示，包含两大分支：

#### 分支一：多视角深度估计（核心）

这是 MVSplat 的关键创新，目标是预测每张输入视图的深度图，再反投影为 3D Gaussian 中心。

**① 多视角特征提取** **(P.5)**：
- 浅层 ResNet-like CNN（6 个残差块）提取 4× 下采样的 per-view 特征
- 6 层 Swin Transformer（local window attention），每层含 self-attention + cross-attention（K>2 时与所有其他视图做 cross-attention，参数数不变）

输出：跨视角感知的 Transformer 特征 $\{F^i\}_{i=1}^K$，$F^i \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times C}$。

**② Cost Volume 构建 (Plane Sweep)** **(P.5-6, Eq. 2-4)**：

在逆深度空间均匀采样 $D=128$ 个深度候选 $\{d_m\}_{m=1}^D$。对视点 $i$，将视点 $j$ 的特征 $F^j$ 通过 $P^i, P^j$ 和深度 $d_m$ 变形到视点 $i$：

$$F_{d_m}^{j \to i} = \mathcal{W}(F^j, P^i, P^j, d_m) \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times C}$$

计算点积相关性：

$$C_{d_m}^i = \frac{F^i \cdot F_{d_m}^{j \to i}}{\sqrt{C}} \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4}}$$

K>2 时对所有其他视图计算相关性并 pixel-wise 平均，最终得到视点 $i$ 的 cost volume：

$$C^i = [C_{d_1}^i, C_{d_2}^i, \cdots, C_{d_D}^i] \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times D}$$

**③ Cost Volume Refinement (U-Net)** **(P.6, Eq. 5)**：

将 $F^i$ 和 $C^i$ 拼接后输入 2D U-Net，输出残差 $\Delta C^i$。U-Net 最低分辨率处插入 3 层 cross-view attention，使模型支持任意视图数：

$$\tilde{C}^i = C^i + \Delta C^i$$

最后 CNN upsampler 上采样到全分辨率 $\tilde{C}^i \in \mathbb{R}^{H \times W \times D}$。

**④ Depth Estimation** **(P.6, Eq. 6)**：

对 refined cost volume 在深度维度做 softmax 归一化，然后对深度候选 $G = [d_1, \cdots, d_D]$ 做加权平均：

$$V^i = \text{softmax}(\tilde{C}^i) G \in \mathbb{R}^{H \times W}$$

**⑤ Depth Refinement** **(P.6-7)**：

另一个轻量 2D U-Net，以多视图图像、特征和当前深度预测为输入，输出残差深度，相加得到最终深度。

#### 分支二：其他 Gaussian 参数预测 **(P.7)**

- **Opacity $\alpha$**：从 matching confidence（softmax 输出的最大值）用两层卷积预测——因为高匹配置信度的点更可能在表面上，与 opacity 物理意义相似
- **Covariance $\Sigma$** 和 **Color $\mathbf{c}$**：以拼接的图像特征、refined cost volume 和原始图像为输入，用两层卷积预测。$\Sigma = R(\theta)^T \text{diag}(s) R(\theta)$，由缩放 $s$ 和旋转四元数 $\theta$ 组成

#### 3D Gaussian Centers 生成 **(P.7)**

将每个视角预测的深度图 $V^i$ 用相机参数反投影到 3D，得到 K 组点云。直接取并集作为所有 3D Gaussian 中心。**无需复杂融合**，简单 union 即可——因为 multi-view 一致的深度预测自然对齐。

### 4.3 训练策略

**损失函数** **(P.7)**：只用光度监督，不依赖 GT 深度。

$$\mathcal{L} = \mathcal{L}_2 + 0.05 \cdot \mathcal{L}_{\text{LPIPS}}$$

**学习率调度**：Adam 优化器，单张 A100 GPU，batch size=14（每个 batch 含 1 个场景的 2 张输入 + 4 张目标视图），300K iterations **(P.7)**。

**数据增强**：两输入视图的帧间距随训练逐步增大（与 pixelSplat [1] 相同）**(P.23)**。

**初始化**：backbone 使用 UniMatch [46] 预训练权重初始化（与测试数据集无重叠），但**随机初始化也能超越 pixelSplat**（需更多轮次 450K）**(P.19, Table D)**。

## 5. 实验

### 5.1 实验设置

**数据集** **(P.7-8)**：

| 数据集 | 场景类型 | Training / Test | 分辨率 |
|--------|---------|----------------|--------|
| RealEstate10K [54] | 室内房源视频 | 67,477 / 7,289 | 256×256 |
| ACID [21] | 室外无人机航拍 | 11,075 / 1,972 | 256×256 |
| DTU [17] | 物体居中（多视图基准） | 16 个验证场景 | 256×256 |

**评估指标**：PSNR, SSIM, LPIPS + 推理时间 + 参数量

**Baselines** **(P.8)**：pixelNeRF [49], GPNR [35], AttnRend [10], MuRF [44], **pixelSplat [1]**（最核心 baseline）

**实现细节** **(P.8)**：PyTorch + CUDA 3DGS renderer，Swin Transformer（6 层，window 2×2），128 深度候选，depth range [1, 100]。单 A100 GPU 训练。

### 5.2 主要结果

**Table 1 (P.8)**：RealEstate10K 和 ACID 上的量化对比：

| 方法 | Params ↓ | Time ↓ | RE10K PSNR↑ / SSIM↑ / LPIPS↓ | ACID PSNR↑ / SSIM↑ / LPIPS↓ |
|------|----------|--------|------|------|
| pixelSplat [1] | 125.4M | 0.104s | 25.89 / 0.858 / 0.142 | 28.14 / 0.839 / 0.150 |
| **MVSplat** | **12.0M** | **0.044s** | **26.39 / 0.869 / 0.128** | **28.25 / 0.843 / 0.144** |

MVSplat 在所有指标上超越 pixelSplat，且参数仅为 1/10，速度 2× 以上。与 NeRF-based MuRF（0.186s, 26.10 PSNR）相比也显著更快更好。

**几何重建质量** **(P.9, Figure 4)**：pixelSplat 的 3D Gaussian 中存在大量漂浮伪影（floating Gaussians），需额外 50K 步 depth-regularized fine-tune。MVSplat 仅用光度监督就生成清晰、紧致的 3D Gaussian 分布，深度图更平滑。

**跨数据集泛化** **(P.10-11, Table 2)**：在 RE10K（室内）上训练的模型直接零样本测试 ACID（室外）和 DTU（物体居中）：

| 训练→测试 | 方法 | PSNR | SSIM | LPIPS |
|----------|------|------|------|-------|
| RE10K→ACID | pixelSplat | 27.64 | 0.830 | 0.160 |
| RE10K→ACID | **MVSplat** | **28.15** | **0.841** | **0.147** |
| RE10K→DTU | pixelSplat | 12.89 | 0.382 | 0.560 |
| RE10K→DTU | **MVSplat** | **13.94** | **0.473** | **0.385** |

MVSplat 泛化显著更强，尤其 LPIPS 提升了 0.175（DTU 约 31% 相对提升）。原因是 cost volume 的相似度值与特征绝对值尺度无关，pixelSplat 基于数据驱动的特征聚合对特征分布偏移敏感。

**更多输入视图** **(P.11)**：MVSplat 支持测试时任意视图数输入，增加视图后 PSNR 从 13.94→14.30。pixelSplat 则因特征分布偏移反而变差。

### 5.3 消融实验

**Table 3 (P.12)** 和 **Figure 6 (P.12)** 提供了详细的消融分析：

| 消融 | PSNR | SSIM | LPIPS | 核心结论 |
|------|------|------|-------|---------|
| **base + refine (full)** | **26.39** | **0.869** | **0.128** | — |
| base | 26.12 | 0.864 | 0.133 | depth refinement 带来 ~0.3dB 提升 |
| **w/o cost volume** | **22.83** | **0.753** | **0.197** | **最关键组件**，去掉后 PSNR 暴跌 >3dB，两视图直接叠加无法对齐 |
| w/o cross-view attention | 25.19 | 0.852 | 0.152 | 严重过拟合（见 Figure A），~1dB 下降 |
| w/o U-Net | 25.45 | 0.847 | 0.150 | 对仅单视图可见的区域影响大（~0.7dB） |

**更多消融 (Appendix A, P.18-19)**：
- **Cost volume 移植到 pixelSplat (Table A)**：将 pixelSplat 的概率密度深度预测替换为 cost volume 方案后，PSNR 从 25.89 提升至 26.63，验证了 cost volume 的通用有效性
- **Swin vs Epipolar Transformer (Table B)**：Swin 更高效（0.038s vs 0.055s），质量相当
- **Gaussian 数量 (Table C)**：每像素 3 个 Gaussian 提升 PSNR 至 26.54，但渲染速度下降 1.5×
- **随机初始化 (Table D)**：随机初始化 + 450K 达 26.29 PSNR，仍超越 pixelSplat

### 5.4 实验可能的不足

1. **反射表面**：作者指出在玻璃、窗户等非朗伯/反射表面效果不佳 **(P.20, Figure B)**
2. **训练数据多样性有限**：仅在 RealEstate10K（室内房源视频）上训练，尽管跨数据集泛化好，但直接应用到 in-the-wild 场景仍不可靠 **(P.13)**
3. **与 per-scene optimization 方法的比较缺失**：只对比了 feed-forward 方法，未与 DSNeRF、RegNeRF、InstantSplat 等 per-scene sparse-view 方法比较
4. **DTU 上的 LPIPS 仍偏高** (0.385)，尽管超越 pixelSplat (0.560)，但室内→物体居中的域偏移仍大
5. 在关键应用场景（自动驾驶 safety-critical 系统）中的可靠性未经评估 **(P.20)**

## 6. 一句话总结

MVSplat 将 MVS 中的 **cost volume (plane-sweep)** 引入 feed-forward 3DGS 框架，用特征匹配替代数据驱动的深度回归，以 **1/10 的参数量和 2× 速度** 超越 pixelSplat，在稀疏视图场景重建中建立了新的效率-质量帕累托前沿。

---

## 附录：论文图表清单

- **Figure 1** (P.1): MVSplat vs pixelSplat 的视觉对比 + 10× 更少参数、2× 更快
- **Figure 2** (P.4): 整体架构图：多视图特征提取 → Cost Volume → U-Net refinement → Depth → 反投影为 3DGS centers + 其他参数 → Rasterization
- **Figure 3** (P.9): 各方法定性对比（复杂区域如重复纹理、单视图可见、大尺度室外）
- **Figure 4** (P.10): 3D Gaussian 和深度图几何质量对比（MVSplat 无漂浮伪影）
- **Figure 5** (P.11): 跨数据集泛化定性对比（RE10K→ACID/DTU zero-shot）
- **Figure 6** (P.12): 消融实验的误差图可视化
- **Figure A** (P.20): 各消融的验证曲线（w/o cross-attn 过拟合）
- **Figure B** (P.20): 失败案例（反射表面）
- **Figures C-E** (P.21-22): 更多视觉比较（定性扩展）
- **Table 1** (P.8): 各方法在 RE10K/ACID 上的量化比较（+速度 + 参数量）
- **Table 2** (P.11): 跨数据集泛化量化（RE10K→ACID / DTU）
- **Table 3** (P.12): 消融实验量化结果
- **Table A** (P.18): Cost volume 替换 pixelSplat 深度分支的效果
- **Table B** (P.18): Swin vs Epipolar Transformer 对比
- **Table C** (P.19): Gaussian 数量 vs 质量/速度 trade-off
- **Table D** (P.19): 不同初始化策略对比
