---
title: "教程：卡尔曼滤波（Tony Lacey）"
paper_date: 2000-01-01
date: 2026-06-09
link: https://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf
description: "MIT 卡尔曼滤波经典教程（Tony Lacey）。从 MSE 和最大似然估计两个视角推导标准卡尔曼滤波，涵盖状态空间推导、卡尔曼增益、协方差更新、预测-更新递归循环，以及卡方视角和信息滤波形式。"
tldr: "标准卡尔曼滤波教程。从 MSE 最小化出发推导状态空间形式的线性卡尔曼滤波器，给出完整的滤波-预测递归算法，并用最大似然/卡方视角做交叉验证。附带信息滤波形式的协方差更新推导。"
authors: ["Tony Lacey"]
affiliations: ["MIT"]
cat_path: "math/stat-prob"
tags: [kalman-filter, state-estimation, tutorial, mit, recursive-estimation, tracking, math/stat-prob]
slug: "mit-kalman-filter-tutorial-lacey"
---

> **完整中文翻译**。原文：Tony Lacey, "Chapter 11 - Tutorial: The Kalman Filter", MIT.

---

# 教程：卡尔曼滤波

## 11.1 引言

卡尔曼滤波 [1] 长期以来被认为是许多跟踪和数据预测任务的最优解决方案 [2]。它在视觉运动分析中的应用已有大量文献记载。本文给出标准卡尔曼滤波的推导，作为在实际中使用统计技术的教程练习。该滤波器被构建为均方误差（MSE）最小化器，同时也提供了另一种推导方式，展示滤波器如何与最大似然统计相关。记录这些推导可以让读者更深入了解滤波器内部的统计结构。

滤波的目的是从信号中提取所需信息，忽略其他一切。滤波器执行此任务的好坏可以用代价函数或损失函数来衡量。事实上，我们可以将滤波器的目标定义为最小化这个损失函数。

## 11.2 均方误差 (MSE)

许多信号可以用以下方式描述：

$$ y_k = a_k x_k + n_k \tag{11.1} $$

其中 $y_k$ 是时间相关的观测信号，$a_k$ 是增益项，$x_k$ 是承载信息的信号，$n_k$ 是加性噪声。

总体目标是估计 $x_k$。估计值 $\hat{x}_k$ 与 $x_k$ 之差称为误差：

$$ f(e_k) = f(x_k - \hat{x}_k) \tag{11.2} $$

$f(e_k)$ 的具体形状取决于应用，但显然该函数应该是正的且单调递增 [3]。具有这些特性的误差函数是平方误差函数：

$$ f(e_k) = (x_k - \hat{x}_k)^2 \tag{11.3} $$

由于需要考虑滤波器在一段时间内预测许多数据的能力，更有意义的度量是误差函数的期望值：

$$ \text{loss function} = E[f(e_k)] \tag{11.4} $$

这产生了均方误差（MSE）函数：

$$ \xi(t) = E[e_k^2] \tag{11.5} $$

## 11.3 最大似然估计

上述均方误差的推导虽然直观，但有些启发式。可以使用最大似然统计进行更严格的推导。这通过重新定义滤波器的目标为寻找最大化 $y$ 概率或似然的 $\hat{x}$ 来实现：

$$ \max [P(y|\hat{x})] \tag{11.6} $$

假设加性随机噪声为高斯分布，标准差为 $\sigma_k$，则：

$$ P(y_k|\hat{x}_k) = K_k \exp\left(-\frac{(y_k - a_k \hat{x}_k)^2}{2\sigma_k^2}\right) \tag{11.7} $$

其中 $K_k$ 是归一化常数。最大似然函数为：

$$ P(y|\hat{x}) = \prod_k K_k \exp\left(-\frac{(y_k - a_k \hat{x}_k)^2}{2\sigma_k^2}\right) \tag{11.8} $$

取对数得：

$$ \log P(y|\hat{x}) = -\frac{1}{2} \sum_k \frac{(y_k - a_k \hat{x}_k)^2}{\sigma_k^2} + \text{constant} \tag{11.9} $$

方程 (11.9) 的驱动函数就是均方误差，可以通过变动 $\hat{x}_k$ 来最大化。因此，当 $y_k$ 的预期变化最好建模为高斯分布时，均方误差函数是适用的。在这种情况下，MSE 提供了使信号 $y_k$ 似然最大化的 $\hat{x}_k$ 值。

在下文推导中，最优滤波器被定义为从所有可能滤波器的集合中使均方误差最小的那个滤波器。

## 11.4 卡尔曼滤波推导

在讨论卡尔曼滤波之前，应首先提及 Norbert Wiener [4] 的工作。Wiener 描述了均方误差意义下的最优有限脉冲响应（FIR）滤波器。尽管他的解与卡尔曼滤波有很多共同之处，但这里不作讨论。只需知道他的解同时使用了接收信号的自相关和与原始数据的互相关来推导滤波器的脉冲响应。

Kalman 也给出了最优 MSE 滤波器的一个方案。但 Kalman 的方案比 Wiener 有一些优势：它避开了确定滤波器脉冲响应的需要，而这项工作不适合数值计算。Kalman 使用状态空间技术描述他的滤波器，与 Wiener 的方案不同，这使得滤波器既可以作为平滑器、滤波器，也可以作为预测器使用。这三个中的最后一项——即卡尔曼滤波器预测数据的能力——已被证明是非常有用的功能，使卡尔曼滤波器被广泛应用于各种跟踪和导航问题。用状态空间方法定义滤波器也简化了离散域中的实现，这是其广泛吸引力的另一个原因。

## 11.5 状态空间推导

假设我们想知道以下形式过程中某个变量的值：

$$ \mathbf{x}_{k+1} = \Phi \mathbf{x}_k + \mathbf{w}_k \tag{11.10} $$

其中 $\mathbf{x}_k$ 是时刻 $k$ 的过程状态向量（$n \times 1$）；$\Phi$ 是从时刻 $k$ 到时刻 $k+1$ 的状态转移矩阵，假设随时间固定不变（$n \times n$）；$\mathbf{w}_k$ 是相关的白噪声过程，具有已知协方差（$n \times 1$）。

对该变量的观测可以建模为以下形式：

$$ \mathbf{z}_k = H \mathbf{x}_k + \mathbf{v}_k \tag{11.11} $$

其中 $\mathbf{z}_k$ 是时刻 $k$ 对 $\mathbf{x}$ 的实际测量值（$m \times 1$）；$H$ 是状态向量与观测向量之间的无噪声连接，假设随时间固定（$m \times n$）；$\mathbf{v}_k$ 是相关的测量误差，同样假设为具有已知协方差的白噪声过程，且与过程噪声互不相关（$m \times 1$）。

如前面章节所示，要使 MSE 最小化产生最优滤波器，必须能用高斯分布正确建模系统误差。两个噪声模型的协方差假设随时间固定，由下式给出：

$$ Q = E[\mathbf{w}_k \mathbf{w}_k^T] \tag{11.12} $$
$$ R = E[\mathbf{v}_k \mathbf{v}_k^T] \tag{11.13} $$

均方误差由 (11.5) 给出，等价于：

$$ E[\mathbf{e}_k \mathbf{e}_k^T] = P_k \tag{11.14} $$

其中 $P_k$ 是时刻 $k$ 的误差协方差矩阵（$n \times n$）。

展开 (11.14)：

$$ P_k = E[\mathbf{e}_k \mathbf{e}_k^T] = E[(\mathbf{x}_k - \hat{\mathbf{x}}_k)(\mathbf{x}_k - \hat{\mathbf{x}}_k)^T] \tag{11.15} $$

假设 $\hat{\mathbf{x}}_k$ 的先验估计称为 $\hat{\mathbf{x}}_k'$，由系统知识获得。可以写出新估计的更新方程，将旧估计与观测数据结合：

$$ \hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k' + K_k (\mathbf{z}_k - H \hat{\mathbf{x}}_k') \tag{11.16} $$

其中 $K_k$ 是卡尔曼增益，将在稍后推导。式 (11.16) 中的 $\mathbf{z}_k - H \hat{\mathbf{x}}_k'$ 称为新息（innovation）或测量残差：

$$ \mathbf{i}_k = \mathbf{z}_k - H \hat{\mathbf{x}}_k' \tag{11.17} $$

将 (11.11) 代入 (11.16) 得：

$$ \hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k' + K_k (H\mathbf{x}_k + \mathbf{v}_k - H\hat{\mathbf{x}}_k') \tag{11.18} $$

将 (11.18) 代入 (11.15) 得：

$$ P_k = E\left[ [(I - K_k H)(\mathbf{x}_k - \hat{\mathbf{x}}_k') - K_k \mathbf{v}_k] [(I - K_k H)(\mathbf{x}_k - \hat{\mathbf{x}}_k') - K_k \mathbf{v}_k]^T \right] \tag{11.19} $$

注意到 $\mathbf{x}_k - \hat{\mathbf{x}}_k'$ 是先验估计的误差，它与测量噪声不相关，因此期望可以重写为：

$$ P_k = (I - K_k H)E[(\mathbf{x}_k - \hat{\mathbf{x}}_k')(\mathbf{x}_k - \hat{\mathbf{x}}_k')^T](I - K_k H)^T + K_k E[\mathbf{v}_k \mathbf{v}_k^T] K_k^T \tag{11.20} $$

代入 (11.13) 和 (11.15) 得：

$$ P_k = (I - K_k H)P_k'(I - K_k H)^T + K_k R K_k^T \tag{11.21} $$

其中 $P_k'$ 是 $P_k$ 的先验估计。方程 (11.21) 是误差协方差更新方程。协方差矩阵的对角线包含均方误差。

协方差矩阵的迹是对角线元素之和。对于误差协方差矩阵，迹就是均方误差的总和。因此，均方误差可以通过最小化 $P_k$ 的迹来最小化。

对 $P_k$ 的迹关于 $K_k$ 求导并令结果为零，以找到最小化条件。展开 (11.21) 得：

$$ P_k = P_k' - K_k H P_k' - P_k' H^T K_k^T + K_k (H P_k' H^T + R) K_k^T \tag{11.23} $$

利用矩阵的迹等于其转置的迹这一性质：

$$ \text{Tr}[P_k] = \text{Tr}[P_k'] - 2\text{Tr}[K_k H P_k'] + \text{Tr}[K_k (H P_k' H^T + R) K_k^T] \tag{11.24} $$

对 $K_k$ 求导：

$$ \frac{d\text{Tr}[P_k]}{dK_k} = -2(H P_k')^T + 2K_k(H P_k' H^T + R) \tag{11.25} $$

令导数为零并整理：

$$ (H P_k')^T = K_k(H P_k' H^T + R) \tag{11.26} $$

解出 $K_k$：

$$ K_k = P_k' H^T (H P_k' H^T + R)^{-1} \tag{11.27} $$

方程 (11.27) 是卡尔曼增益方程。新息 $\mathbf{i}_k$ 有一个关联的测量预测协方差：

$$ S_k = H P_k' H^T + R \tag{11.28} $$

最后，将 (11.27) 代入 (11.23) 得：

$$ P_k = P_k' - P_k' H^T (H P_k' H^T + R)^{-1} H P_k' = P_k' - K_k H P_k' = (I - K_k H) P_k' \tag{11.29} $$

方程 (11.29) 是最优增益下的误差协方差更新方程。

状态预测通过下式实现：

$$ \hat{\mathbf{x}}_{k+1}' = \Phi \hat{\mathbf{x}}_k \tag{11.30} $$

为完成递归，需要找到将误差协方差矩阵投影到下一个时间间隔 $k+1$ 的方程。首先构造先验误差的表达式：

$$ \mathbf{e}_{k+1}' = \mathbf{x}_{k+1} - \hat{\mathbf{x}}_{k+1}' = (\Phi \mathbf{x}_k + \mathbf{w}_k) - \Phi \hat{\mathbf{x}}_k = \Phi \mathbf{e}_k + \mathbf{w}_k \tag{11.31} $$

将 (11.15) 扩展到时刻 $k+1$：

$$ P_{k+1}' = E[\mathbf{e}_{k+1}' (\mathbf{e}_{k+1}')^T] = E[(\Phi \mathbf{e}_k + \mathbf{w}_k)(\Phi \mathbf{e}_k + \mathbf{w}_k)^T] \tag{11.32} $$

注意 $\mathbf{e}_k$ 和 $\mathbf{w}_k$ 具有零互相关，因为噪声 $\mathbf{w}_k$ 实际上是在 $k$ 和 $k+1$ 之间累积的，而误差 $\mathbf{e}_k$ 是截止到时刻 $k$ 的误差。因此：

$$ P_{k+1}' = \Phi P_k \Phi^T + Q \tag{11.33} $$

这就完成了递归滤波器。

### 卡尔曼滤波递归算法总结

| 描述 | 方程 |
|------|------|
| **卡尔曼增益** | $K_k = P_k' H^T (H P_k' H^T + R)^{-1}$ |
| **更新估计** | $\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k' + K_k(\mathbf{z}_k - H \hat{\mathbf{x}}_k')$ |
| **更新协方差** | $P_k = (I - K_k H) P_k'$ |
| **投影到 $k+1$** | $\hat{\mathbf{x}}_{k+1}' = \Phi \hat{\mathbf{x}}_k$ |
| **协方差投影** | $P_{k+1}' = \Phi P_k \Phi^T + Q$ |

## 11.6 卡尔曼滤波的卡方视角

卡尔曼滤波的目标是最小化实际数据与估计数据之间的均方误差。因此，它在均方误差意义上提供了数据的最佳估计。既然是这种情况，应该可以证明卡尔曼滤波与卡方有很多共同之处。

卡方函数是一个最大似然函数，由前面的方程 (11.9) 导出，通常用作拟合一组模型参数的准则——即所谓的最小二乘拟合。卡尔曼滤波通常被称为递归最小二乘（RLS）拟合器。与卡方函数划等号将为理解卡尔曼滤波正在做什么提供一个不同的视角。

卡方函数为：

$$ \chi^2 = \sum_{i=1}^k \frac{[z_i - h(a_i; \mathbf{x})]^2}{\sigma_i^2} \tag{11.34} $$

其中 $z_i$ 是测量值；$h_i$ 是具有参数 $\mathbf{x}$（假设对于 $a$ 是线性的）的数据模型；$\sigma_i^2$ 是与测量值相关的方差。最优参数集就是最小化上述函数的参数。

用向量形式表示并使用早前卡尔曼推导中的记号：

$$ \chi_k^2 = [\mathbf{z}_k - h(a; \mathbf{x}_k)] R^{-1} [\mathbf{z}_k - h(a; \mathbf{x}_k)]^T \tag{11.36} $$

其中 $R^{-1}$ 是逆方差矩阵（即 $1/\sigma_i\sigma_i$）。

已知逆模型协方差矩阵到时刻 $k$ 为止的信息，截止到时刻 $k$ 的卡方函数可以重写为：

$$ \chi_{k-1}^2 = (\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1}) P_{k-1}^{'-1} (\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1})^T \tag{11.37} $$

为了将新数据与之前的数据合并，通过最小化整体卡方函数来拟合模型参数，卡方函数变为两者之和：

$$ \chi^2 = (\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1}) P_{k-1}^{'-1} (\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1})^T + [\mathbf{z}_k - h(a; \mathbf{x}_k)] R^{-1} [\mathbf{z}_k - h(a; \mathbf{x}_k)]^T \tag{11.38} $$

通过推导（详见原文 11.39-11.48），最终得到参数更新公式：

$$ \mathbf{x}_k = \hat{\mathbf{x}}_k + K_k[\mathbf{z}_k - h(a; \hat{\mathbf{x}}_k)] \tag{11.49} $$

其中卡尔曼增益可以识别为：

$$ K_k = (P_k^{'-1} + H^T R^{-1} H)^{-1} H^T R^{-1} \tag{11.48} $$

方程 (11.49) 与 (11.16) 完全相同，描述了使用测量值与模型投影值之间的误差来改进参数估计。

## 11.7 模型协方差更新（信息滤波形式）

模型参数协方差在其逆形式下被考虑，此时称为信息矩阵。可以使用标准误差传播来制定协方差矩阵的另一种更新形式：

$$ P_k^{-1} = P_k^{'-1} + H R^{-1} H^T \tag{11.50} $$

可以证明方程 (11.50) 与标准形式 (11.29) 等价，利用恒等式 $P_k \cdot P_k^{-1} = I$ 即可。

卡尔曼滤波的两种协方差更新形式：
- **标准形式**：$P_k = (I - K_k H) P_k'$
- **信息滤波形式**：$P_k^{-1} = P_k^{'-1} + H R^{-1} H^T$

当卡尔曼滤波建立在信息矩阵上时，它被称为**信息滤波器**（Information Filter）。

---

## 翻译说明

本文是对 Tony Lacey "Chapter 11 - Tutorial: The Kalman Filter"（MIT）的完整中文翻译。原文是卡尔曼滤波的经典入门教程，从均方误差最小化和最大似然估计两个视角推导了标准线性卡尔曼滤波器的全部方程，内容紧凑、推导完整，特别适合作为系统学习卡尔曼滤波的起点。
