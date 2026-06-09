---
title: "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"
paper_date: 2026-06-02
date: 2026-06-02
arxiv: https://doi.org/10.1145/3315508.3329973
description: "提出 Triton，一种基于 tile 概念的 C-like 语言和 LLVM-based 编译器，通过 tile 级优化（层次化 tiling、memory coalescing、shared memory allocation）实现与 cuBLAS/cuDNN 相当的矩阵乘法和卷积性能，同时支持灵活的自定义 kernel 编写。"
tldr: "提出 Triton，一种基于 tile 概念的 C-like 语言和 LLVM-based 编译器，通过 tile 级优化（层次化 tiling、memory coalescing、shared memory allocation）实现与 cuBLAS/cuDNN 相当的矩阵乘法和卷积性能，同时支持灵活的自定义 kernel 编写。"
authors: ["Philippe Tillet", "H. T. Kung", "David Cox"]
cat_path: "compiler"
tags: [compiler, efficient-inference, gpu, deep-learning]
slug: "triton"
---

> **TLDR**: Triton 是一种基于 tile 的 C-like DSL + LLVM 编译器，用 statically shaped multi-dimensional tile 作为核心抽象，通过 tile 级优化 pass 实现与 cuBLAS/cuDNN 相当的 GPU kernel 性能，比 AutoTVM/TC/PlaidML 等 DSL 快 2-3×。

## 1. TLDR

- **领域**: GPU 深度学习编译器
- **任务**: 让非专家也能写出媲美手写 cuBLAS/cuDNN 的自定义 GPU kernel
- **问题**: 现有 DSL（Tensor Comprehensions、TVM、PlaidML）要么只支持 affine array indices（无法表达 shifted conv、structured sparsity），要么性能远低于 vendor library。手写 GPU kernel 则需要 CUDA 专家
- **核心贡献**: 以 **tile**（statically shaped multi-dimentional sub-array）为中心的 C-like 语言 + LLVM-based IR + JIT 编译器，通过 tile 级优化 pass（层次化 tiling、memory coalescing、shared memory allocation、自动 prefetching）在多种 workload 上达到与 cuBLAS/cuDNN 相当的性能，且能高效实现非标准算子（shift convolution）

## 2. 作者背景

Harvard University（Philippe Tillet, H. T. Kung 教授, David Cox）。H. T. Kung 是计算机体系结构领域的先驱（ systolic array 发明者之一）。David Cox 后来创立了 Modular AI（Mojo 语言）。

## 3. 相关工作背景

三条技术路线 (P.3)：

- **Tensor-level IR**（XLA, Glow）：模式匹配 tensor contraction → 预定义模板。灵活度低，难以表达新算子
- **Polyhedral model**（Tensor Comprehensions, Diesel）：用多面体模型参数化循环编译。但**只能处理 affine array indices**，无法表达 shifted conv、sparse patterns 等非仿射内存访问
- **Loop synthesizers**（Halide, TVM）：把 tensor 计算转为循环嵌套，用户手动/参数化指定 schedule。但需要专家调 schedule，且性能仍远低于 vendor library

**Triton 的差异化**：引入 **tile 级操作和优化到传统编译管线**中——比 XLA/Glow 更灵活，支持非仿射索引（如 pointer lookup table），自动推导执行 schedule（不像 Halide/TVM 需手动指定）。

## 4. 核心方法

### 4.1 整体架构 (Figure 2, P.2)

```
Triton-C (C-like DSL) → Triton-IR (LLVM-based) → Triton-JIT → GPU code
                              ↕                    ↕
                        (可被高级 DSL 生成)      Auto-Tuner
```

### 4.2 Triton-C 语言 (P.3)

C-like 语法 + Numpy-like tile 语义 + SPMD 编程模型。

**关键特性**：

**Tile 声明**：用 `int tile[16, 16]` 声明多维 tile（区别于 C 的 `int tile[16][16]`）。形状必须常量化，但可用 `tunable` 关键字参数化。

**Broadcasting 语义**：与 Numpy 一致的 broadcasting 规则——短 shape 左补 1 → 沿 size=1 的维度复制。这与 PyTorch/Numpy 用户直觉一致，隐藏了 intra-tile memory coalescing、cache 管理等底层细节。

**编程模型**：与 CUDA 的 SPMD 不同，Triton 的每个 kernel 实例是**单线程**的（但自动并行化）。没有 shared memory sync、inter-thread communication 等 CUDA 并发原语。每个实例通过 `get_global_range(axis)` 查询自己的索引范围（Figure 3, P.4）。

**Predication**：用 `@` 前缀做谓词化控制流——替代 CUDA 中的分支，因为 tile 元素不能被单独访问（Listing 6）。

**Matmul 示例**（Listing 1, P.2）：

```c
const tunable int TM = {16, 32, 64, 128};  // 可调优参数
const tunable int TN = {16, 32, 64, 128};
const tunable int TK = {8, 16};

kernel void matmul_nt(float *a, float *b, float *c, int M, int N, int K) {
    int rm[TM] = get_global_range(0);  // 行索引 tile
    int rn[TN] = get_global_range(1);  // 列索引 tile
    int rk[TK] = 0 ... TK;             // 归约维度范围

    float C[TM, TN] = 0;               // 累加器 tile
    float *pa[TM, TK] = a + rm[:, newaxis] + rk * M;  // A 的指针 tile
    float *pb[TN, TK] = b + rn[:, newaxis] + rk * K;  // B 的指针 tile

    for (int k = K; k >= 0; k -= TK) {
        float A[TM, TK] = *pa;          // load tile
        float B[TN, TK] = *pb;
        C += dot(A, trans(B));          // tile 级 matmul
        pa += TK * M;                   // 更新指针
        pb += TK * N;
    }
    float *pc[TM, TN] = c + rm[:, newaxis] + rn * M;
    *pc = C;                            // write-back
}
```

### 4.3 Triton-IR (P.4-5)

LLVM-based IR，扩展了 tile 级 data-flow 和 control-flow 支持：

**Tile 数据类型**：`i32<8, 8>` 表示 8×8 32-bit integer tile。无 `tunable` 关键字——参数化形状必须在 JIT 编译时由 auto-tuner 实例化。

**Retiling 指令**：
- `reshape`：改变 tile 的 shape（如 [8] → [8,1] 准备 broadcast）
- `broadcast`：沿 size=1 维度复制数据

**谓词 SSA（PSSA）**：`cmpp` 指令返回一对相反 predicate + `psi` 指令合并不同谓词流的值，解决 tile 级 divergent control-flow 问题。

### 4.4 Triton-JIT 编译器 (P.5-6)

Triton-JIT 的核心目标：把 Triton-IR 程序编译成高效的 GPU machine code。它通过一组**机器无关**和**机器相关**的优化 pass 来完成，最后由 auto-tuner 搜索最优参数。

---

#### Machine-Independent Passes（机器无关）

与目标硬件无关，只对 Triton-IR 本身的结构做变换。

**1. Pre-fetching（预取）**

循环中的 tile 级 load 操作会引发严重 latency——当没有足够多的独立指令来 hiding latency 时问题更突出。Triton-JIT 通过检测循环体中的 load + pointer increment 模式，自动插入 prefetch：

```
// 优化前                         // 优化后
B0:                                B0:
  %p0 = getelementptr %1, %2         %p0 = getelementptr %1, %2
  %x0 = load %p0                    %x0 = load %p0        ← 第一次load
B1:                                B1:
  %p = phi [%p0, B0], [%p1, B1]     %x = phi [%x0, B0], [%x1, B1]
  %x = load %p                      %p = phi [%p0, B0], [%p1, B1]
  %p1 = getelementptr %p, %3        %p1 = getelementptr %p, %3
                                     %x1 = load %p1       ← 提前load下一次
```

核心思路：在循环迭代 i 计算的同时，发出迭代 i+1 的 load 请求，让内存访问和计算重叠。

**2. Tile-Level Peephole（窥孔优化）**

tile 级操作的存在提供了新的 peephole 机会。例如：
- 转置链简化：$(X^T)^T = X$，`trans(trans(X))` → `X`
- 对角 tile 的代数性质（如单位矩阵乘法）

这些是在标量 IR 中无法表达的优化。

---

#### Machine-Dependent Passes（机器相关）

针对 GPU 的层次化内存模型（Figure 5, P.5）：

```
DRAM (large, slow)
  ↑↓
Shared Memory (medium, fast, per-block)
  ↑↓
Register File (small, fastest, per-thread)
```

**1. Hierarchical Tiling（层次化 Tiling）**

这是最核心的 pass。Triton 把一个 tile 分解为三层，每层对应 GPU 的一个内存层级：

| 层级 | 名称 | 大小范围 | 所在内存 | 含义 |
|------|------|----------|----------|------|
| 1 | Tile | 32-128 元素 | DRAM | 一次全局内存传输的粒度 |
| 2 | Micro-Tile | 8-32 元素 | Shared Memory | 线程块内协作计算的粒度 |
| 3 | Nano-Tile | 1-4 元素 | Register File | 单个 SIMD 单元处理的粒度 |

编译器从 Triton-IR 中提取 tile 形状后，自动枚举合法的层次化分解配置——为每一维的每一层指定分解因子——而不需要多面体模型（polyhedral model）的支持。这比 Tensor Comprehensions 的 polyhedral 方法更灵活，因为 Triton 可以处理**非仿射**的 tile 索引（pointer LUT）。

**2. Memory Coalescing（内存合并）**

GPU 的性能关键是 DRAM 访问的合并（coalescing）：当相邻线程访问连续的全局内存地址时，硬件可以把多次请求合并为一次宽位传输。否则每个线程独立请求，带宽利用率急剧下降。

Triton 的优势在这里体现：因为 Triton-IR 程序是**单线程**的，编译器拥有完全的调度自由——它可以在 micro-tile 内部排序线程的执行顺序，保证相邻线程访问相邻地址。

具体来说：
- 无合并（uncoalesced）：线程按列方向遍历 tile → 同一 warp 的线程访问不相邻的行 → 多次内存事务
- 有合并（coalesced）：编译器重排线程顺序为行优先遍历 → 同一 warp 的线程访问连续地址 → 一次内存事务

这种优化在 CUDA 中需要程序员手动保证（通过合理的 thread index 到 memory address 的映射），在 Triton 中由编译器**自动完成**。

**3. Shared Memory Allocation（共享内存分配）**

当 tile 操作具有高计算强度（如 `dot`）时，把 operand 暂存到 shared memory 可以大幅降低 DRAM 访问延迟。

分配策略分两步：
1. **Live range 分析**：对每个 tile 变量，计算它从定义到最后一次使用之间的跨度（live interval）
2. **线性时间分配**：使用 Gergov (1999) 的算法，在变量 live range 不重叠时复用同一块 shared memory——类似于寄存器分配的 graph coloring 策略，但简化版的线性扫描

**4. Shared Memory Synchronization（同步插入）**

Shared memory 的读写是异步的。Triton-JIT 需要自动插入 `__syncthreads()` barrier 来保证正确性。它使用经典的数据流方程检测两种 hazard：

- **RAW (Read-After-Write)**：先写后读 → 写完之后需要 barrier 确保其他线程能读到
- **WAR (Write-After-Read)**：先读后写 → 读完之后需要 barrier 确保旧值不被覆盖

数据流方程（s 为基本块，$read(s)$ 和 $write(s)$ 分别表示读和写的变量集合）：

RAW hazard 检测：
$$ins(RAW) = \bigcap_{p \in pred(s)} out_p(RAW)$$
$$out_s(RAW) = \begin{cases} \emptyset & \text{if } ins(RAW) \cap read(s) \neq \emptyset \text{ (需插入 barrier)} \\ins(RAW) \cup write(s) & \text{otherwise} \end{cases}$$

WAR hazard 检测：
$$ins(WAR) = \bigcap_{p \in pred(s)} out_p(WAR)$$
$$out_s(WAR) = \begin{cases} \emptyset & \text{if } ins(WAR) \cap write(s) \neq \emptyset \text{ (需插入 barrier)} \\ins(WAR) \cup read(s) & \text{otherwise} \end{cases}$$

简单来说：进入一个基本块时，如果**前驱承诺会写入的变量**（$ins(RAW)$）与**当前块要读的变量**有交集，说明可能发生 RAW hazard → 插入 barrier 并清空 $out$（因为 barrier 后所有写入都对其他线程可见了）。WAR 同理。

---

#### Auto-Tuner（自动调优）

传统 auto-tuner 依赖手写参数化模板（如 cuBLAS 对不同 shape 的手写 kernel 变体）。Triton-JIT 的 auto-tuner 直接从 Triton-IR 提取优化空间：

1.  解析 Hierarchical Tiling pass 的元参数（每维度每层级的 tile/micro-tile/nano-tile 大小）
2.  对每个元参数枚举 2 的幂范围内的合法值：
    - Tile 大小：32 ~ 128
    - Micro-tile 大小：8 ~ 32
    - Nano-tile 大小：1 ~ 4
3.  穷举搜索所有组合，对每种配置编译并 benchmark，选最快的

由于优化参数空间不大（每维度最多 3 个参数 × 几个候选值），穷举在可接受时间内完成。

![Hierarchical Tiling and GPU Memory Model](/papers/figures/triton-hierarchical-tiling.svg)

*层次化 Tiling 与 GPU 内存模型对应关系：64×64 的 tile 先分解为 4 个 16×16 的 micro-tile（Shared Memory），每个 micro-tile 再分解为 64 个 2×2 的 nano-tile（Register File）。Auto-tuner 穷举搜索各层级的最优分解参数。*

![Memory Coalescing](/papers/figures/triton-coalescing.svg)

*GPU 内存合并对比：列优先遍历（左）导致 warp 内 16 个线程访问不连续的 DRAM 地址，需要 N 次内存事务；行优先遍历（右）合并为单次宽位传输。Triton 编译器自动排序 micro-tile 内的线程访存顺序来实现 coalescing。*

### 4.5 与现有 DSL 的关键区别 (P.3)

| 维度 | Tensor Comprehensions | TVM | Triton |
|------|---------------------|-----|--------|
| 索引表达式 | 只支持 affine | 支持任意 | 支持任意 |
| Schedule | 自动 | 用户手动 | 自动 |
| Tile 级优化 | 无 | 无 | 层次化 tiling / coalescing |
| 与 cuBLAS 差距 | 2-3× 慢 | <2× 慢 | 基本持平 |

## 5. 实验

### 5.1 实验设置

- **GPU**: NVIDIA GeForce GTX1070
- **Baselines**: cuBLAS 10.0, cuDNN 7.0, AutoTVM, Tensor Comprehensions, PlaidML
- **Workloads**: DeepSpeech2 RNN、Transformer、ResNet CNN、shift convolution

### 5.2 主要结果

**矩阵乘法**（Figure 8, P.6）：

- 方形矩阵（M=N=K）：Triton 与 cuBLAS 相当，>90% 峰值性能
- DeepSpeech2 RNN：Triton 与 cuBLAS 相当；AutoTVM/TC/PlaidML 2-3× 慢
- Transformer：cuBLAS 在 shallow transformer 上略快（受益于 3D 算法），Triton 次之。TVM <2× 慢，其他 DSL 2-3× 慢

**卷积**（Figure 10, P.7）：

- ResNet dense conv：Triton **超越** cuDNN 的 IMPLICIT_GEMM（cuDNN 资源集中在 Winograd 3×3 优化，其他 kernel 投入不足）
- DeepSpeech2 conv：cuDNN 与 Triton 持平

**Shift Convolution**（Figure 11, P.7）：

- Triton fuse shift conv 实现几乎完全隐藏了 shifting 开销（与不 shift 的 1×1 conv 性能相当）
- 这是现有 DSL（TC/TVM）无法表达的非仿射索引场景——因为每个通道的位移量存储在一个 pointer LUT 中

### 5.3 可能的不足

- 编程复杂度高于 TVM/TC（相比于 Listing 2 的一行 Python，Triton-C 需要约 30 行）
- 未评估 tensor core 支持
- 未评估量化 kernel
- 未与最新 DSL（如 Halide/TVM 的后续版本）比较
- 仅在一块 GPU（GTX1070）上测试，未验证跨平台可移植性

## 6. 一句话总结

Triton 通过将 tile 作为一等公民引入 LLVM-based 编译管线，在保持灵活性的同时实现了媲美手写 vendor library 的 GPU kernel 性能，是后续 PyTorch Triton backend、OpenAI Triton 等项目的直接前身。

## 🧠 伪代码：Triton-JIT 四个核心 Pass 的实现

---

### Pass 1: Prefetching（自动预取）

**输入**：Triton-IR 的 CFG（Control Flow Graph）
**输出**：插入 prefetch load 后的 CFG

```
function PREFETCHING(cfg):
    for each loop L in cfg:
        // 1. 识别循环体中的 load → gep 链
        //    模式：%val = load %ptr
        //          %next = getelementptr %ptr, %stride
        //          (loops back to phi)
        
        for each basic_block B in L.body:
            load_insts = find_pattern(B, [load, gep, phi]) 
            
            for each load_l in load_insts:
                %ptr_l  = operand(l)          // 当前指针
                %stride = extract_stride(l)   // 每次迭代的步长
                
                // 2. 检查该 load 的 latency 能否被隐藏
                def_use_chain = compute_dependency_chain(l)
                if latency_of(load) > cycles_in_between(l, end_of_loop):
                    // latency 无法被计算隐藏 → 需要 prefetch
                    
                    // 3. 在 load 之后立即插入下一次的 prefetch
                    //    但保持语义等价性
                    %next_ptr = gep(%ptr_l, %stride)
                    %prefetch = load %next_ptr  // 提前加载下一次
                    
                    // 4. 修改 phi 节点：现在 phi 可以选 prefetched 值
                    modify_phi_node(L.header, l, %prefetch)
                    
                    // 效果：循环迭代 i 计算时，
                    //      迭代 i+1 的内存访问已提前发出
```

**关键技巧**：Triton-IR 的 tile 级 load 操作是粗粒度的（一次加载整个 tile），这使得编译器可以精确计算每次 load 的 latency，并判断是否值得 prefetch。如果 tile 较小（如 nano-tile），register file 直接命中，prefetch 没必要。

---

### Pass 2: Hierarchical Tiling（层次化 Tiling）

**输入**：Triton-IR 中的 tile shape 元数据
**输出**：最优的 tile / micro-tile / nano-tile 分解配置

```
function HIERARCHICAL_TILING(ir_program, device_spec):
    // 1. 从 IR 中提取所有 tile 变量及其 shape
    tiles = extract_tile_shapes(ir_program)
    // tiles = [{name: "A", shape: (128, 64), dims: 2}, ...]
    
    // 2. 为每个 tile 的每一维枚举分解因子
    //    分解因子 = tile → micro-tile → nano-tile 的三级分解
    //    每级大小必须是 2 的幂，且在合法范围内
    
    param_space = []
    for each tile in tiles:
        for each dim d in tile.dims:
            // 合法值：
            //   tile_size:    32 ~ 128   (DRAM 传输粒度)
            //   micro_size:   8  ~ 32    (Shared Mem 粒度)
            //   nano_size:    1  ~ 4     (Register 粒度)
            for ts in powers_of_two_in(32, 128):
                for ms in powers_of_two_in(8, 32):
                    for ns in powers_of_two_in(1, 4):
                        if ms * ns == ts:   // 必须能完整分解
                            param_space.append({tile, d, ts, ms, ns})
    
    // 3. 合法性检查：分解必须适配硬件限制
    valid_configs = []
    for cfg in param_space:
        if fits_in_shared_memory(cfg, device_spec.shared_mem_size):
            if fits_in_registers(cfg, device_spec.register_count):
                if num_blocks_fits_device(cfg, device_spec.max_blocks):
                    valid_configs.append(cfg)
    
    // 4. 返回搜索空间（由 auto-tuner 枚举 benchmark）
    return valid_configs
```

**为什么不需要 polyhedral model？** Triton-IR 的 tile 形状和 decomposition 信息是**显式的**（在 IR 中直接编码为 tile 数据类型和 reshape/broadcast 指令），不需要像 Tensor Comprehensions 那样从嵌套循环的 affine 表达式中**推断**出来。这是 Triton 能处理非仿射索引（shift conv 的 pointer LUT）的关键。

---

### Pass 3: Memory Coalescing（内存合并）

**输入**：micro-tile 及其线程-to-地址映射
**输出**：重排后的线程遍历顺序

```
function MEMORY_COALESCING(micro_tile, thread_layout):
    // micro_tile 是 8x8 或 16x16 的 2D tile
    // thread_layout 描述了线程如何映射到 tile 元素
    //
    // 核心思路：因为 Triton-IR 是单线程的，
    // 编译器可以自由决定线程的执行顺序
    
    let WARP_SIZE = 32  // NVIDIA GPU 的 warp 大小
    
    // 1. 获取该 micro_tile 所有 load 指令的访存模式
    access_patterns = analyze_memory_accesses(micro_tile)
    
    for each load_op in access_patterns:
        // 2. 计算当前线程顺序下的合并效率
        current_thread_order = get_current_order(thread_layout)
        // 例如：列优先 [T0→col0, T1→col1, ...]
        
        coalescing_ratio = compute_coalescing_ratio(
            current_thread_order, load_op, WARP_SIZE)
        // 如果 ratio < 1，说明有未合并的访问
        
        if coalescing_ratio < 1.0:
            // 3. 尝试重排线程顺序
            //    目标：同一 warp 内连续线程访问连续地址
            
            // 对 2D tile 来说，最优顺序通常是行优先
            // （row 在连续地址空间中的步长为 1）
            new_order = optimize_thread_order(thread_layout, load_op)
            // 例如：行优先 [T0→row0_col0, T1→row0_col1, ...]
            
            // 4. 验证新顺序下同一 warp 的地址是否连续
            for warp in partition_into_warps(new_order, WARP_SIZE):
                addresses = compute_addresses(warp, load_op)
                if not are_contiguous(addresses):
                    // 如果仍不连续，尝试其他排序策略
                    // 或者标记为无法合并（避免生成次优代码）
                    ...
            
            // 5. 应用新的线程顺序
            apply_thread_reordering(micro_tile, new_order)
```

**关键洞察**：在标准 CUDA 中，线程到数据的映射由程序员通过 `threadIdx.x / blockIdx.x` 手动控制，一旦写死就很难改。Triton 的 `get_global_range()` 返回的是逻辑索引而非硬件线程 ID——编译器可以自由地将逻辑索引映射到硬件线程，只要保证语义正确。这种**间接层**是 Triton 能自动 coalescing 的根本原因。

---

### Pass 4: Shared Memory Allocation（共享内存分配与同步）

**输入**：Triton-IR 函数体中所有 tile 变量的 live range
**输出**：shared memory 偏移量 + 同步 barrier 插入点

```
function SHARED_MEMORY_ALLOCATION(func):
    // ——Phase 1: Live Range 分析——
    // 对每个 tile 变量 v，计算它的 live interval [first_use, last_use]
    
    intervals = []
    for each variable v in func.variables:
        if not is_tile_candidate(v):  
            continue  // 跳过非 tile 变量或计算强度低的变量
            // 判断标准：是否为 dot() 等计算密集型操作的 operand
        
        first = find_first_use(v, func)
        last  = find_last_use(v, func)
        intervals.append({var: v, start: first, end: last, size: sizeof(v)})
    
    // 按 start 排序（线性扫描）
    sort_by_start(intervals)
    
    // ——Phase 2: 线性时间分配（Gergov 1999 算法）——
    //    类似寄存器分配中的 linear scan，但针对 shared memory
    
    active = []       // 当前活跃的 live intervals（按 end 排序）
    offsets = {}      // var → offset_in_shared_memory
    total_size = 0
    
    for each interval in intervals:
        // 2a. 释放已结束的活跃区间
        for each act in active:
            if act.end < interval.start:
                free(act)  // 回收该区间占用的 shared memory
                remove(active, act)
        
        // 2b. 分配当前区间的偏移量
        if has_free_slot(active, interval.size):
            // 在空闲区域中分配一个合适的 slot
            offset = allocate_slot(interval.size)
        else:
            // 无空闲空间，在已用空间之后追加
            offset = total_size
            total_size += interval.size
        
        offsets[interval.var] = offset
        active.append(interval)
        sort_by_end(active)
    
    // ——Phase 3: 自动插入同步 barrier——
    //    检测 RAW / WAR hazard，插入 __syncthreads()
    
    barriers = []
    for each basic_block B in postorder(func.cfg):
        // 前驱承诺的写入和读取
        ins_raw = intersect(pred(B).outs_raw)
        ins_war = intersect(pred(B).outs_war)
        
        // RAW: 如果前驱写入的变量在当前块中被读取
        if (ins_raw ∩ read_set(B)) ≠ ∅:
            insert_barrier(before(B))
            outs_raw[B] = ∅       // barrier 后所有写入对其他线程可见
            outs_war[B] = ∅
        else:
            outs_raw[B] = ins_raw ∪ write_set(B)
            outs_war[B] = ins_war ∪ read_set(B)
        
        // WAR: 如果前驱读取的变量在当前块中被写入
        //      （检测方法和 RAW 对称，用 write_set 和 read_set 对调）
        if (ins_war ∩ write_set(B)) ≠ ∅:
            insert_barrier(before(B))
            // ... 同上
    
    return offsets, barriers
```

**为什么线性扫描就够了？** Shared memory 的分配比寄存器分配简单得多——shared memory 没有固定数量的限制（只有总容量上限），且访问模式更规整。线性扫描（相对于完整的 graph coloring）已足够找到近乎最优的分配，且时间开销 O(n log n) 极低。

**数据流方程的直觉理解**：想象维护一个"跨基本块必须可见"的集合。如果一个基本块承诺了写入变量 A，所有后续执行路径都"知道"A 被写了。如果某个路径上的基本块要读 A，说明可能发生 RAW——插入 barrier 后清空这个集合，因为 barrier 之后所有写入都变得对其他线程可见了。

---

### 四个 Pass 的协作流程

```
Triton-IR Program
    │
    ▼
┌─────────────────────┐
│ Machine-Independent  │
│  ├─ Prefetching      │    ← 优化内存访问延迟
│  └─ Tile Peephole    │    ← 化简 tile 级代数表达式
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Machine-Dependent    │
│  ├─ Hierarchical     │    ← 确定 tile 分解参数
│  │   Tiling          │       （与 auto-tuner 协作）
│  ├─ Memory           │    ← 重排线程顺序实现合并
│  │   Coalescing      │
│  ├─ Shared Mem       │    ← 分配共享内存空间
│  │   Allocation      │
│  └─ Shared Mem       │    ← 插入同步 barrier
│      Synchronization │
└────────┬────────────┘
         ▼
    GPU Machine Code
```

## 关键参考文献

1. **cuBLAS / cuDNN** — NVIDIA vendor library baseline
2. **Tensor Comprehensions** — Polyhedral DSL baseline
3. **TVM / AutoTVM** — Loop synthesis DSL baseline
4. **PlaidML** — Alternative DSL baseline
5. **Halide** — Loop synthesis 开创性工作
6. **XLA / Glow** — Tensor-level IR baseline
7. **DeepSpeech2** — RNN workload
8. **ResNet** — CNN workload, 残差网络
9. **Winograd** — Fast convolution algorithm
10. **Shift convolution** — Novel operator enabled by Triton

## 附录：论文图表清单

- **Figure 1** (P.1): Roofline 模型对比各 DSL 与 cuBLAS/Triton 性能
- **Figure 2** (P.2): Triton 系统架构
- **Figure 3** (P.4): CUDA vs Triton 编程模型对比
- **Figure 4** (P.4): broadcast 指令示意图
- **Figure 5** (P.5): Triton-IR 层次化内存模型
- **Figure 6** (P.5): coalesced vs uncoalesced DRAM 访问
- **Figure 7** (P.5): Shared Memory 分配算法
- **Figure 8** (P.6): 矩阵乘法性能对比（Square/DeepSpeech2/Transformer）
- **Figure 9** (P.6): Dense conv 和 shift conv 的架构对比
- **Figure 10** (P.7): 卷积性能对比
- **Figure 11** (P.7): Shift conv 性能（naive vs fused vs max）
- **Listing 1** (P.2): Matmul 在 Triton-C 中的完整实现
- **Listing 5** (P.4): ReLU 在 Triton-IR 中的表示
- **Listing 8** (P.7): Shift conv 在 Triton-C 中的实现
