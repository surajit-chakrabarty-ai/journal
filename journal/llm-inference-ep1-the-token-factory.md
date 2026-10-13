---
date: 2026-10-13 18:35
---
# LLM Inference Deep Dive: Episode 1 — The Inference Bottleneck

**TL;DR:** LLM inference isn't just "run the model" — it's a complex dance between compute, memory bandwidth, and latency constraints. This episode builds the mental model you need to understand every optimization that follows.

---

## Series Overview

This 3-episode series takes you from the fundamentals of how LLMs generate tokens to the cutting-edge systems that serve millions of requests:

- **Episode 1** (this post): The Inference Bottleneck — Understanding why LLM serving is hard
- **Episode 2**: The Optimization Playbook — Continuous batching, PagedAttention, and vLLM
- **Episode 3**: Production-Grade Serving — Speculative decoding, distributed systems, and fleet operations

---

## The Fundamental Problem: Why Can't We Just Call `model.generate()`?

When you chat with Claude or GPT, something remarkable happens behind the scenes. Your prompt flows through billions of parameters, and tokens emerge one by one. But here's the dirty secret: **a naive implementation would leave a $30,000 GPU sitting 90% idle**.

To understand why, we need to understand what actually happens during inference.

### The Two Phases of LLM Inference

Every LLM inference request has two distinct phases with radically different computational profiles:

#### Phase 1: Prefill (Processing Your Prompt)

When you send a prompt like "Explain quantum computing", the model processes all tokens in parallel:

```
Input: [<BOS>, Explain, quantum, computing]
       ↓ (parallel forward pass)
Output: probability distribution for next token
```

**Characteristics:**
- **Compute-bound**: Many tokens processed simultaneously
- **High arithmetic intensity**: Matrix multiplications dominate
- **GPU utilization**: Can reach 70-90% on well-optimized systems
- **Latency**: Proportional to prompt length

#### Phase 2: Decode (Generating the Response)

After prefill, the model generates tokens one at a time:

```
Step 1: [<BOS>, Explain, quantum, computing] → "in"
Step 2: [<BOS>, Explain, quantum, computing, in] → "simple"
Step 3: [<BOS>, Explain, quantum, computing, in, simple] → "terms"
...
```

**Characteristics:**
- **Memory-bandwidth-bound**: Only 1 token computed, but ALL weights must be loaded
- **Low arithmetic intensity**: ~1-2 FLOPs per byte loaded
- **GPU utilization**: Often 5-15% without optimization
- **Latency**: Roughly constant per token (weight-loading dominates)

---

## The Math That Explains Everything

Let's make this concrete with a 7B parameter model on an H100:

```
Model size: 7B parameters × 2 bytes (FP16) = 14 GB
H100 HBM bandwidth: 3.35 TB/s
H100 peak compute: ~990 TFLOP/s (FP16)

Time to load all weights: 14 GB / 3.35 TB/s ≈ 4.2 ms
FLOPs for batch-1 decode: ~14 billion (2 FLOPs per parameter)
Time if compute-bound: 14 TFLOP / 990 TFLOP/s ≈ 0.014 ms

Reality: We're waiting 4.2ms for memory while compute 
        could finish in 0.014ms — a 300× gap!
```

This is the **memory bandwidth wall**. For batch-1 decode, you're paying the full cost of loading every weight just to compute a single token.

### The Roofline Model: Visualizing the Problem

The roofline model plots achievable performance against arithmetic intensity:

```
Performance (TFLOP/s)
       ↑
  990  |                    _______________  ← Compute ceiling
       |                   /
       |                  /
       |                 /
       |                /  ← Memory bandwidth slope
       |               /
       |              /
       |_____________/________________________→ Arithmetic Intensity
                    ~70                         (FLOPs/byte)
                     ↑
              Ridge point (H100)
```

- **Below the ridge point**: Memory-bound (decode phase)
- **Above the ridge point**: Compute-bound (prefill phase)

For decode at batch=1, arithmetic intensity ≈ 1-2 FLOPs/byte — far below the ridge point. **This is why batching matters.**

---

## The Latency Budget: A Real-World Constraint

In production systems, you're not optimizing for raw throughput alone. Real applications have latency constraints:

### Key Metrics

| Metric | Definition | Typical Target |
|--------|------------|----------------|
| **TTFT** (Time to First Token) | Time from request to first output token | < 200ms |
| **ITL** (Inter-Token Latency) | Time between consecutive tokens | < 50ms |
| **E2E Latency** | Total request time | < 5-10s |
| **Throughput** | Tokens/second across all requests | Maximize |

### The Fundamental Tradeoff

Here's the core tension in LLM serving:

```
↑ Batch Size → ↑ Throughput but ↑ Latency
↓ Batch Size → ↓ Latency but ↓ Throughput
```

With batch=1, latency is minimal but GPU utilization is ~10%. With batch=64, GPU utilization might hit 80%, but each request waits longer.

**The art of LLM serving is navigating this tradeoff dynamically.**

---

## The KV Cache: Trading Memory for Compute

Here's a crucial insight: during decode, we don't need to recompute everything. The attention mechanism creates key (K) and value (V) vectors for each token, and these don't change once computed.

### How Attention Works (Simplified)

For each token, attention computes:
```
Q = input × W_q  (query: "what am I looking for?")
K = input × W_k  (key: "what do I contain?")
V = input × W_v  (value: "what information do I have?")

Attention(Q, K, V) = softmax(QK^T / √d) × V
```

The key insight: **K and V for previous tokens don't change when generating new tokens.**

### The KV Cache Trade-off

Instead of recomputing K and V for all previous tokens at each step:

```
Without KV Cache (naive):
Step N: Compute K,V for tokens 1...N, then attention
Cost: O(N²) compute per step, O(1) memory

With KV Cache:
Step N: Load K,V for tokens 1...N-1 from cache, compute only for token N
Cost: O(N) compute per step, O(N) memory
```

**KV cache size per token** (for a model like Llama-2-7B):
```
2 (K and V) × 32 (layers) × 32 (heads) × 128 (head dim) × 2 (FP16 bytes)
= 524,288 bytes = 512 KB per token
```

For a 4096-token context: **2 GB of KV cache per request!**

This is why memory becomes the constraint for long-context models. The KV cache grows linearly with sequence length, and you need to store it for every concurrent request.

---

## Model Architectures: What You're Serving Matters

Different architectures have different serving characteristics:

### Encoder Models (BERT, RoBERTa, DeBERTa)
- **Single forward pass**: No autoregressive generation
- **Compute-bound**: High arithmetic intensity
- **Batching-friendly**: Dynamic batching with 1-5ms queue delay works great
- **Use case**: Classification, embeddings, NLU tasks

### Decoder Models (GPT, Llama, Mistral, Qwen)
- **Autoregressive**: Generate tokens one at a time
- **Memory-bound during decode**: Low arithmetic intensity
- **KV cache hungry**: Memory scales with context length
- **Use case**: Text generation, chat, code completion

### Encoder-Decoder (T5, BART)
- **Two-phase**: Encode once, decode many
- **Hybrid profile**: Encoder is compute-bound, decoder is memory-bound
- **Use case**: Translation, summarization

### Embedding Models (CTE, Sentence Transformers)
- **Single forward pass**: Like encoders, but output is a dense vector
- **Large models**: Often 1B+ parameters
- **Batching nuanced**: Large models may saturate GPU without batching
- **Use case**: Semantic search, retrieval

### Vision-Language Models (LLaVA, GPT-4V)
- **Image tokens dominate prefill**: A single image can be 500+ tokens
- **Memory intensive**: Image embeddings + text KV cache
- **Use case**: Multimodal understanding

---

## The Bottleneck Taxonomy

When optimizing inference, first identify your bottleneck:

### 1. Compute-Bound
**Symptoms:**
- GPU compute utilization near peak (70%+)
- Increasing batch size increases latency proportionally
- Prefill-dominated workloads

**Solutions:**
- Quantization (INT8, FP8) — reduces FLOPs
- Kernel fusion — reduces overhead
- TensorRT / torch.compile — optimized kernels
- Better hardware

### 2. Memory-Bandwidth-Bound
**Symptoms:**
- Low GPU compute utilization (10-30%)
- Decode-dominated workloads
- Latency doesn't improve much with smaller batches

**Solutions:**
- Increase batch size — amortize weight loading
- Speculative decoding — generate multiple tokens per weight load
- Quantization — reduces bytes to load
- Continuous batching — keep GPU fed

### 3. Memory-Capacity-Bound
**Symptoms:**
- OOM errors when increasing batch size
- Long context requests fail
- High GPU memory utilization with low compute

**Solutions:**
- KV cache quantization (FP8, INT4)
- PagedAttention — eliminate fragmentation
- Model parallelism (TP/PP)
- Offloading to CPU/NVMe

### 4. I/O-Bound (Overhead-Bound)
**Symptoms:**
- Low GPU utilization despite small model
- High CPU time between GPU calls
- Python/framework overhead dominates

**Solutions:**
- CUDA graphs — eliminate launch overhead
- torch.compile — fuse operations
- Move to C++ server (Triton Inference Server)
- Reduce Python in hot path

---

## Reading Utilization Metrics Correctly

A critical skill: **GPU utilization can be misleading**.

```
GPU Utilization = % of time at least one kernel was running
```

This does NOT tell you how much of the GPU's capability that kernel used!

**Example:**
- Batch-1 decode: A tiny kernel runs continuously → 95% "utilization"
- But that kernel uses 2% of the GPU's compute capability
- Real efficiency: ~2%, not 95%

**Better metrics:**
- `SM Occupancy`: % of streaming multiprocessors active
- `Achieved FLOPs / Peak FLOPs`: True compute efficiency
- `InferenceCount / ExecutionCount`: Actual batch size achieved

---

## Key Takeaways from Episode 1

1. **LLM inference has two phases** with opposite characteristics: compute-bound prefill and memory-bound decode

2. **The memory bandwidth wall** is why batch-1 decode achieves <15% GPU utilization — you load 14GB of weights to compute one token

3. **KV cache** trades memory for compute, but scales linearly with context length × batch size (512KB per token for 7B models)

4. **Identify your bottleneck first**: The right optimization depends on whether you're compute-bound, memory-bound, or overhead-bound

5. **Latency and throughput are in tension**: Every optimization navigates this tradeoff — batching converts latency headroom into throughput

6. **GPU utilization lies**: A kernel running 100% of the time might use 2% of the GPU's capability

---

**Next Episode:** We'll dive into the techniques that solve these problems — continuous batching, PagedAttention, FlashAttention, and the vLLM architecture that puts them together.

---

*References:*
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [HuggingFace Blog: Continuous Batching](https://huggingface.co/blog/continuous_batching)
