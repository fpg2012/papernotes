---
title: "Linear Cosine Palettes：用四条余弦波生成无限调色板"
date: 2025-09-14
description: "i = a + b·cos(2π(c·t + d)) — 四个 RGB 向量，一条公式，无限色彩变化。Inigo Quilez 的极简调色板方案，来自一个游戏图形程序员的奇思妙想。"
cat_path: "cv/gen"
tags: [color, generative-art, graphics, cosine, palette, shader]
type: "trivia"
slug: "cosine-palettes-inigo-quilez"
---

> **TLDR**: Inigo Quilez 提出用四条余弦波生成连续调色板，每条波控制一个 RGB 通道，用一个三通道频率向量控制色彩变化节奏。四个三维向量（a, b, c, d）定义一整条调色板——这是为 GPU shader 优化的极简方案，在现代生成艺术中被广泛采用。

---

## 核心公式

对于给定的 $t \in [0, 1]$，颜色的 RGB 三通道同时由余弦波生成：

$$f(t) = \mathbf{a} + \mathbf{b} \cdot \cos\big(2\pi(\mathbf{c} \cdot t + \mathbf{d})\big)$$

其中 $\mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d} \in \mathbb{R}^3$ 分别对应 RGB 三个通道的参数：

- **$\mathbf{a}$** — 偏移量（offset），决定颜色的基准位置
- **$\mathbf{b}$** — 振幅（amplitude），决定颜色偏离基准的幅度
- **$\mathbf{c}$** — 频率（frequency），决定每个通道颜色变化的快慢（最关键的参数）
- **$\mathbf{d}$** — 相位（phase），决定颜色变化的起始位置

输出被截断到 $[0, 1]$ 作为 RGB 值。

## 一些例子

三个手调参数产生的调色板：

{{< figure src="/figures/cosine-palette-examples.png" alt="余弦调色板示例" width="100%" >}}

八组随机参数生成的调色板（固定 $\mathbf{a} = [0.5, 0.5, 0.5]$）：

{{< figure src="/figures/cosine-palette-random-strips.png" alt="随机余弦调色板" width="100%" >}}

RGB 三通道的分解视图——可以看到每条通道的余弦波独立振荡，合成出丰富的色彩：

{{< figure src="/figures/cosine-palette-rgb-decomp.png" alt="RGB 通道分解" width="100%" >}}

## 直观理解

固定 $\mathbf{a} = [0.5, 0.5, 0.5]$ 时，每个通道的值在 $[0, 1]$ 之间振荡（因为余弦值域 $[-1, 1]$，乘以 $\mathbf{b}$ 再加 $\mathbf{a}$）。

**关键是 $\mathbf{c}$：** 如果 $\mathbf{c} = [1, 1, 1]$，R、G、B 会在同一频率上同步振荡，得到的调色板只是灰度渐变。频率之间的微小差异（比如 $\mathbf{c} = [1.0, 0.6, 0.3]$）才是产生丰富色彩的秘诀——三个通道以不同速度在色环上"追赶"彼此。

## Python 实现

```python
import numpy as np

def cosine_palette(a, b, c, d, n=256):
    """a, b, c, d 都是长度为 3 的 numpy 数组 (R, G, B)
    返回 n×3 的 RGB 数组，值域 [0, 1]"""
    t = np.linspace(0, 1, n)
    pal = a + b * np.cos(2 * np.pi * (c * t[:, None] + d))
    return np.clip(pal, 0, 1)

# 示例
a = np.array([0.5, 0.5, 0.5])
b = np.array([0.5, 0.4, 0.3])
c = np.array([1.0, 0.6, 0.3])
d = np.array([0.0, 0.2, 0.5])

palette = cosine_palette(a, b, c, d, n=256)
```

## 为什么它很有趣

这条公式是 Inigo Quilez 为 **GPU shader** 设计的——在 fragment shader 里，不需要纹理查找表，不需要任何内存访问，四条 `cos` 指令就能生成一整条连续调色板。对于 GPU 来说，`cos` 是有专门硬件指令的，比纹理采样还快。

后来被 Danielle Navarro 推广到 R 语言的生成艺术社区，用随机选定的基础颜色来生成 $\mathbf{b}, \mathbf{c}, \mathbf{d}$，然后用简单的 subdivision 或 Lissajous 系统就能做出风格多变的生成艺术作品。

**最妙的地方：** 16 个浮点数定义一整条无限连续的调色板。$(a_r, a_g, a_b, b_r, b_g, b_b, c_r, c_g, c_b, d_r, d_g, d_b)$ — 就这么 12 个值，穷尽所有可能的连续色板。

---

## 原文

- **Inigo Quilez**: [Simple Procedural Palettes](https://iquilezles.org/articles/palettes/) (原始灵感)
- **Danielle Navarro**: [Linear Cosine Palettes](https://blog.djnavarro.net/posts/2025-09-14_cosine-palettes/)（R 实现与生成艺术应用）
- **Mike Cheng**: [Mastodon 帖子](https://fosstodon.org/@coolbutuseless/115173701685084866)（将方案引入 R 社区）
