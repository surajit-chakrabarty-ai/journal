---
date: 2026-10-13 15:43
---
# The GPU Kernel Zoo: A Deep Dive into the Engines Powering Modern AI

**TL;DR:** Every time you run `model.forward()`, thousands of specialized GPU kernels spring into action—each one a carefully crafted piece of code optimized for a specific operation. This post dissects the major kernel types, from the humble element-wise operations to the mighty matrix multiplication engines that consume 90%+ of your training FLOPs.

---

## Why Should You Care About GPU Kernels?

Here's a number that should make you sit up: **A single H100 GPU costs ~$30,000 and can deliver 1,979 TFLOP/s of FP8 compute.** But here's the dirty secret—most deep learning workloads achieve only 30-50% of that theoretical peak.

The gap between "what the hardware can do" and "what your model actually achieves" is called **MFU (Model FLOP Utilization)**. And the primary determinant of MFU? The quality of your GPU kernels.

```
MFU = (Actual FLOP/s achieved) / (Theoretical peak FLOP/s)

Typical values:
- Naive PyTorch: 15-30%
- Well-optimized training: 40-55%
- SOTA implementations: 55-70%
```

Every percentage point of MFU you recover at scale translates to millions of dollars saved. At 30,000 H100s running for months, a 1% improvement is worth more than most startups raise in their Series A.

---

## The Taxonomy of GPU Kernels

GPU kernels can be broadly categorized by their **computational intensity**—the ratio of compute operations to memory operations. This single metric determines whether a kernel is:

- **Memory-bound:** Waiting on data movement (low arithmetic intensity)
- **Compute-bound:** Saturating the ALUs (high arithmetic intensity)

```
Arithmetic Intensity (AI) = FLOPs / Bytes loaded from memory

Memory-bound: AI < ridge point (~410 for H100)
Compute-bound: AI > ridge point
```

Let's walk through the kernel zoo, from simplest to most complex.

---

## 1. Element-wise Kernels: The Workhorses

**Examples:** ReLU, GELU, SiLU, LayerNorm, Dropout, Add, Multiply

**Arithmetic Intensity:** Very low (typically 1-10 FLOPs/byte)

**Characteristic:** One output element depends on one (or a few) input elements.

```python
# Conceptually simple
def relu_kernel(x):
    return max(0, x)  # 1 comparison per element

def gelu_kernel(x):
    # ~10 FLOPs per element
    return 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
```

### Why They're Tricky Despite Being "Simple"

Element-wise kernels are almost always **memory-bound**. The GPU can compute faster than it can fetch data. This leads to a counterintuitive optimization strategy: **kernel fusion**.

```
Without fusion:
  Load X → Compute ReLU → Store Y → Load Y → Compute Dropout → Store Z
  Memory ops: 4 × N elements

With fusion:
  Load X → Compute ReLU → Compute Dropout → Store Z
  Memory ops: 2 × N elements
  
Speedup: ~2x (memory-bound, so halving memory ops ≈ halving time)
```

**FlashAttention's secret sauce** isn't just about attention—it's about fusing the entire softmax-matmul-dropout sequence to minimize HBM round-trips.

### The Bandwidth Reality

| Memory Type | H100 Bandwidth | Latency |
|-------------|----------------|---------|
| HBM3 (GMEM) | 3.35 TB/s | ~400 cycles |
| L2 Cache | ~12 TB/s | ~200 cycles |
| L1/SMEM | ~33 TB/s | ~30 cycles |
| Registers | ~100+ TB/s | 1 cycle |

For a simple ReLU on a 1B parameter model (4GB in FP32):
- Time to load from HBM: 4GB / 3.35 TB/s ≈ 1.2ms
- Time to compute: negligible
- **You're paying for memory, not compute**

---

## 2. Reduction Kernels: The Aggregators

**Examples:** Sum, Mean, Max, Softmax, LayerNorm (the normalization part), Loss functions

**Arithmetic Intensity:** Low to medium

**Characteristic:** Many input elements contribute to fewer output elements.

```
Input:  [x₁, x₂, x₃, ..., xₙ]
Output: [sum(x)]  or  [mean(x)]  or  [softmax(x)]
```

### The Parallel Reduction Pattern

Reductions seem inherently sequential—you need to see all elements to compute the sum. But GPUs parallelize this through **tree reduction**:

```
Step 1: [a₀+a₁] [a₂+a₃] [a₄+a₅] [a₆+a₇]  (N/2 parallel adds)
Step 2: [s₀+s₁] [s₂+s₃]                    (N/4 parallel adds)
Step 3: [t₀+t₁]                            (N/8 parallel adds)
...
Total steps: log₂(N)
```

**Warp-level primitives** make this even faster. On NVIDIA GPUs, threads within a warp (32 threads) can directly exchange values without going through shared memory:

```cuda
// Warp-level reduction using shuffle instructions
float val = /* thread's value */;
for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// Thread 0 now has the sum of all 32 values
```

### Softmax: The Reduction That Ate Deep Learning

Softmax is everywhere in transformers, and it's surprisingly tricky to implement efficiently:

```
softmax(x)ᵢ = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
```

This requires **three passes** over the data:
1. Find max(x) — reduction
2. Compute Σ exp(xᵢ - max) — reduction  
3. Divide each element — element-wise

**Online softmax** (used in FlashAttention) computes this in a single pass by maintaining running statistics—a key innovation that enables attention fusion.

---

## 3. Matrix Multiplication Kernels: The Heavy Lifters

**Examples:** Linear layers, Attention (QKᵀ and Attention×V), Convolutions (via im2col)

**Arithmetic Intensity:** High (can reach 100s of FLOPs/byte with proper tiling)

**Characteristic:** Output elements depend on entire rows/columns of inputs.

```
C[m,n] = Σₖ A[m,k] × B[k,n]

FLOPs: 2 × M × N × K
Bytes loaded (naive): (M×K + K×N + M×N) × sizeof(dtype)
```

### Why Matmul Dominates Everything

In a transformer:
- **QKV projections:** 3 matmuls per attention head
- **Attention scores:** 1 matmul (Q @ Kᵀ)
- **Attention output:** 1 matmul (Scores @ V)
- **Output projection:** 1 matmul
- **MLP:** 2 matmuls (up-projection, down-projection)

For a typical LLM, **>90% of FLOPs are matmuls**. This is why NVIDIA keeps making tensor cores bigger and faster.

### The Tiling Hierarchy

Modern matmul kernels use a multi-level tiling strategy that maps perfectly onto GPU hardware:

```
Grid Level (Thread Blocks):
┌─────────────────────────────────────┐
│  Block    Block    Block    Block   │
│  (0,0)    (0,1)    (0,2)    (0,3)   │
│                                      │
│  Block    Block    Block    Block   │
│  (1,0)    (1,1)    (1,2)    (1,3)   │
└─────────────────────────────────────┘
Each block computes a BM×BN tile of output C

Block Level (Warps):
┌─────────────────┐
│ Warp0  │ Warp1  │
│────────│────────│
│ Warp2  │ Warp3  │
└─────────────────┘
Each warp computes a WM×WN sub-tile

Warp Level (Threads):
Each thread computes a TM×TN micro-tile
```

### The Memory Dance

The key insight: **reuse data as much as possible before evicting it from fast memory**.

```
Naive matmul:
- Load A[m,k] from GMEM: N×K times (once per output column)
- Load B[k,n] from GMEM: M×K times (once per output row)
- Total GMEM loads: O(M×N×K)

Tiled matmul:
- Load A tile to SMEM: once per K-block
- Load B tile to SMEM: once per K-block
- Reuse within SMEM: BM×BN times
- Total GMEM loads: O(M×N×K / min(BM,BN))

Arithmetic Intensity improvement: ~BM or ~BN (typically 64-256x)
```

### Tensor Cores: The Matmul Accelerators

Tensor cores are specialized hardware units that compute small matrix multiplies in a single instruction:

| Generation | Instruction | Shape | Throughput (FP16) |
|------------|-------------|-------|-------------------|
| Volta | HMMA | 16×16×16 | 125 TFLOP/s |
| Ampere | MMA | 16×8×16 | 312 TFLOP/s |
| Hopper | WGMMA | 64×N×16 | 989 TFLOP/s |
| Blackwell | WGMMA | 64×N×16 | 2.25 PFLOP/s |

The `wgmma` (warp-group matrix multiply-accumulate) instruction on Hopper computes a 64×256×16 matmul across 4 warps (128 threads) in one shot. That's **524,288 FLOPs per instruction**.

```cuda
// Hopper tensor core usage (simplified)
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
    "{%0, %1, %2, ...}, "  // accumulator registers
    "{%32, %33}, "          // A descriptor
    "{%34, %35};"           // B descriptor
    : "+f"(d[0]), "+f"(d[1]), ...
    : "l"(desc_a), "l"(desc_b)
);
```

---

## 4. Attention Kernels: The Memory Monsters

**The Problem:** Standard attention has O(N²) memory complexity for sequence length N.

```python
# Standard attention
Q, K, V = x @ Wq, x @ Wk, x @ Wv  # [B, N, D]
scores = Q @ K.T / sqrt(d)        # [B, N, N] ← O(N²) memory!
attn = softmax(scores, dim=-1)    # [B, N, N]
out = attn @ V                    # [B, N, D]
```

For a 128K context window with batch size 1 and FP16:
- Attention matrix size: 128K × 128K × 2 bytes = **32 GB**
- That's more than an entire H100's memory (80GB) just for one attention layer!

### FlashAttention: The Tiling Revolution

FlashAttention's key insight: **never materialize the full N×N attention matrix**.

Instead, compute attention in tiles, maintaining running statistics for the softmax:

```
For each query tile Qᵢ:
    For each key-value tile (Kⱼ, Vⱼ):
        Compute local scores: Sᵢⱼ = Qᵢ @ Kⱼᵀ
        Update running max and sum for online softmax
        Accumulate: Oᵢ += softmax(Sᵢⱼ) @ Vⱼ (with rescaling)
```

**Memory complexity:** O(N) instead of O(N²)
**Speed improvement:** 2-4x faster than standard attention (fewer HBM accesses)

### FlashAttention-2 and Beyond

FlashAttention-2 improved on the original by:
1. **Better work partitioning:** Parallelize over sequence length, not just batch/heads
2. **Reduced non-matmul FLOPs:** Fewer redundant computations in the inner loop
3. **Better occupancy:** More thread blocks active simultaneously

FlashAttention-3 (Hopper) adds:
1. **Asynchronous operations:** Overlap compute with memory transfers via TMA
2. **FP8 support:** 2x throughput with quantized attention
3. **Warp specialization:** Producer/consumer pattern for better pipelining

---

## 5. Convolution Kernels: The Spatial Specialists

**The Transformation:** Convolutions can be expressed as matrix multiplications via **im2col**.

```
Input: [B, C_in, H, W]
Kernel: [C_out, C_in, kH, kW]

im2col transforms input into: [B × H_out × W_out, C_in × kH × kW]
Kernel reshaped to: [C_out, C_in × kH × kW]

Convolution = im2col(input) @ kernel.T
```

This lets convolutions leverage the same highly-optimized matmul kernels.

### Winograd: Trading Multiplies for Adds

For small kernels (3×3), **Winograd convolution** reduces multiplications at the cost of more additions:

```
Standard 3×3 conv: 9 multiplies per output
Winograd F(2×2, 3×3): 4 multiplies per output (2.25x fewer)
Winograd F(4×4, 3×3): 2.25 multiplies per output (4x fewer)
```

The catch: Winograd requires transforming inputs/outputs, which adds overhead. It's only worth it when the kernel is small and the feature maps are large.

### FFT Convolution: The Frequency Domain Trick

For very large kernels, **FFT convolution** wins:

```
Convolution theorem: conv(a, b) = IFFT(FFT(a) × FFT(b))

Standard conv: O(N × K) where K = kernel size
FFT conv: O(N log N) regardless of kernel size
```

Crossover point: typically K > 64 for FFT to be faster.

---

## 6. Embedding and Gather Kernels: The Lookup Tables

**Examples:** Token embeddings, Positional embeddings, Sparse operations

**Characteristic:** Essentially fancy array indexing with potential for irregular access patterns.

```python
# Embedding lookup
def embedding(indices, weight):
    # indices: [B, N] integers
    # weight: [V, D] embedding table
    return weight[indices]  # [B, N, D]
```

### The Memory Access Challenge

Embedding lookups are **pure memory bandwidth**—zero compute. The challenge is that access patterns can be highly irregular:

```
Coalesced access (good):
Thread 0 reads weight[0]
Thread 1 reads weight[1]
Thread 2 reads weight[2]
...
→ Single 128-byte transaction serves 32 threads

Scattered access (bad):
Thread 0 reads weight[42]
Thread 1 reads weight[1337]
Thread 2 reads weight[7]
...
→ Up to 32 separate transactions!
```

For vocabulary embeddings with random token sequences, access is essentially random. Mitigation strategies:
1. **Sorting:** Group similar indices together
2. **Caching:** Keep hot embeddings in L2
3. **Quantization:** Smaller embeddings = more fit in cache

---

## 7. Communication Kernels: The Multi-GPU Glue

**Examples:** AllReduce, AllGather, ReduceScatter, Point-to-Point

When training spans multiple GPUs, **communication becomes a kernel**.

### AllReduce: The Gradient Synchronizer

```
Before AllReduce:
GPU 0: [g₀]    GPU 1: [g₁]    GPU 2: [g₂]    GPU 3: [g₃]

After AllReduce:
GPU 0: [Σgᵢ]   GPU 1: [Σgᵢ]   GPU 2: [Σgᵢ]   GPU 3: [Σgᵢ]
```

**Ring AllReduce** is the standard algorithm:
1. **ReduceScatter:** Each GPU ends up with 1/N of the sum
2. **AllGather:** Broadcast so everyone has the full sum

```
Ring AllReduce bandwidth utilization:
Theoretical: 2 × (N-1)/N × data_size / time
For N=8 GPUs: 87.5% of peak bidirectional bandwidth
```

### NVLink vs PCIe vs Network

| Interconnect | H100 Bandwidth | Latency |
|--------------|----------------|---------|
| NVLink 4.0 | 900 GB/s | ~1 μs |
| PCIe 5.0 | 128 GB/s | ~2 μs |
| InfiniBand NDR | 400 Gb/s | ~1 μs |
| Ethernet (RoCE) | 400 Gb/s | ~2 μs |

The **7x bandwidth gap** between NVLink and PCIe is why DGX systems exist.

---

## The Kernel Fusion Imperative

Modern deep learning frameworks aggressively fuse kernels to minimize memory traffic:

### PyTorch 2.0's torch.compile

```python
@torch.compile
def fused_attention_block(x, w_qkv, w_o):
    qkv = F.linear(x, w_qkv)
    q, k, v = qkv.chunk(3, dim=-1)
    attn = F.scaled_dot_product_attention(q, k, v)
    return F.linear(attn, w_o)

# torch.compile will:
# 1. Trace the computation graph
# 2. Identify fusion opportunities
# 3. Generate optimized CUDA kernels via Triton
```

### What Gets Fused

```
Commonly fused patterns:
- Linear + Bias + Activation (GEMM + elementwise)
- LayerNorm + Linear (reduction + GEMM)
- Attention (Q@K, softmax, @V all in one kernel)
- Residual + LayerNorm (add + reduction)

Fusion blockers:
- Dynamic shapes (can't compile static kernel)
- Data-dependent control flow
- Operations requiring global synchronization
```

---

## Profiling: Finding Your Bottlenecks

### The Roofline Model

The roofline model tells you whether a kernel is memory-bound or compute-bound:

```
Performance = min(Peak_Compute, Peak_Bandwidth × Arithmetic_Intensity)

For H100 SXM:
- Peak FP16 Compute: 1,979 TFLOP/s
- Peak HBM Bandwidth: 3.35 TB/s
- Ridge Point: 1979/3.35 ≈ 590 FLOPs/byte

If your kernel's AI < 590: memory-bound (optimize data movement)
If your kernel's AI > 590: compute-bound (optimize ALU usage)
```

### Nsight Compute Metrics to Watch

```
Memory-bound indicators:
- sm__throughput.avg.pct_of_peak_sustained_elapsed < 50%
- dram__throughput.avg.pct_of_peak_sustained_elapsed > 70%

Compute-bound indicators:
- sm__throughput.avg.pct_of_peak_sustained_elapsed > 70%
- sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed > 50%

Latency-bound indicators:
- sm__warps_active.avg.per_cycle_active < 4
- High stall_* metrics
```

---

## The Future: What's Coming

### FP4 and Microscaling

Blackwell introduces FP4 with microscaling—4-bit floats with per-block scale factors:

```
FP4 with microscaling:
- 4 bits per element (vs 16 for FP16)
- 8-bit scale factor per 32 elements
- Effective: 4.25 bits/element average
- 4x memory bandwidth improvement
- 2x compute throughput improvement
```

### Sparsity

Structured sparsity (2:4 pattern) doubles tensor core throughput:

```
Dense:     [a b c d e f g h]
2:4 Sparse: [a 0 c 0 e 0 g 0]  (keep 2 of every 4)

Tensor cores can skip the zeros, doubling effective throughput.
Ampere+: Native 2:4 sparsity support in tensor cores.
```

### Speculative Decoding Kernels

For LLM inference, speculative decoding requires new kernel patterns:

```
1. Draft model generates K tokens speculatively
2. Target model verifies all K tokens in parallel
3. Accept prefix of correct tokens, reject rest

Requires: Efficient batched verification kernels
         Tree attention for multiple speculation paths
```

---

## Conclusion: The Kernel is the Message

Every deep learning operation ultimately becomes a GPU kernel. Understanding this layer of the stack—even at a high level—transforms how you think about model efficiency:

1. **Memory is the bottleneck** for most operations. Fusion is your friend.
2. **Matmul is special** because it's the only operation that can truly saturate modern GPUs.
3. **Attention is memory-bound** despite being compute-heavy. FlashAttention exists because of this.
4. **Hardware shapes software.** Tensor cores, TMA, and NVLink aren't just specs—they're the reason certain algorithms exist.

The next time your training run is slower than expected, don't just throw more GPUs at it. Profile your kernels. Find the bottleneck. Understand whether you're memory-bound or compute-bound. Then optimize accordingly.

The difference between a 30% MFU and a 60% MFU is the difference between burning money and building the future.

---

## Further Reading

1. **[Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul)** - Aleksa Gordić's deep dive into matmul optimization
2. **[FlashAttention Paper](https://arxiv.org/abs/2205.14135)** - The original memory-efficient attention
3. **[CUTLASS Documentation](https://github.com/NVIDIA/cutlass)** - NVIDIA's template library for GEMM
4. **[Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)** - Write custom GPU kernels in Python
5. **[NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)** - The profiler you need to master

---

*The GPU doesn't care about your elegant PyTorch code. It only sees kernels. Learn to see what it sees.*
