---
date: 2026-10-13 18:36
---
# LLM Inference Deep Dive: Episode 2 — The Optimization Playbook

**TL;DR:** This episode covers the key techniques that transform LLM serving from "expensive and slow" to "efficient at scale" — continuous batching eliminates padding waste, PagedAttention solves KV cache fragmentation, and together they enable 10-20× throughput improvements.

---

## The Problem with Naive Batching

In Episode 1, we established that batching is essential for GPU efficiency. But naive batching has a fatal flaw: **requests have different lengths**.

### Static Batching: The Padding Tax

Consider three requests arriving together:

```
Request A: "What is AI?"           → generates 50 tokens
Request B: "Explain in detail..." → generates 200 tokens  
Request C: "Hi"                    → generates 10 tokens
```

With static batching, all requests must wait for the longest one:

```
Time →
A: [prefill][====decode====][PADDING PADDING PADDING PADDING...]
B: [prefill][==================decode========================]
C: [prefill][==][PADDING PADDING PADDING PADDING PADDING......]
              ↑
              C finished but stuck waiting for B
```

**Problems:**
1. **Wasted compute**: Padding tokens still consume GPU cycles
2. **Wasted memory**: KV cache allocated for max length
3. **Increased latency**: Short requests wait for long ones
4. **Poor throughput**: Can't add new requests until batch completes

### The Quantified Waste

If batch size B=8 and a new request needs N=100 prefill tokens:
```
Padding tokens needed: (N-1) × (B-1) = 99 × 7 = 693 padding tokens!
```

With `torch.compile` or CUDA graphs requiring fixed shapes, you'd pad to `max_seq_len` (often 2048-4096), making waste even worse.

---

## Continuous Batching: The Paradigm Shift

Continuous batching (introduced in the Orca paper) fundamentally changes how we think about batching.

### The Core Insight

Instead of batching at the request level, batch at the **token level**:

```
Traditional: Batch = [Request1, Request2, Request3]
             → All requests padded to same length

Continuous:  Batch = [Token1_from_A, Token1_from_B, Token2_from_C, ...]
             → No padding, mix prefill and decode
```

### How It Works

1. **Remove the batch dimension**: Concatenate all sequences into one long sequence
2. **Use attention masks**: Control which tokens can attend to which
3. **Dynamic scheduling**: Add new requests as old ones finish

```
Step 1: [A_prefill_tokens...][B_prefill_tokens...][C_prefill_tokens...]
        ↓ attention mask ensures A doesn't see B or C
        
Step 2: [A_decode_1][B_decode_1][C_decode_1]  # All decoding

Step 3: [A_decode_2][B_decode_2][C generates EOS, done!]
        [D_prefill_tokens...]  # D joins immediately!
        
Step 4: [A_decode_3][B_decode_3][D_decode_1]
```

### The Attention Mask: Enabling Sequence Isolation

The attention mask ensures sequences don't cross-attend:

```
Tokens: [A1, A2, A3, B1, B2, C1]

Attention Mask (1 = can attend, 0 = blocked):
        A1  A2  A3  B1  B2  C1
    A1  [1   0   0   0   0   0]
    A2  [1   1   0   0   0   0]  
    A3  [1   1   1   0   0   0]  ← A tokens only see A tokens (causal)
    B1  [0   0   0   1   0   0]
    B2  [0   0   0   1   1   0]  ← B tokens only see B tokens
    C1  [0   0   0   0   0   1]
```

This block-diagonal structure maintains causal isolation between requests while allowing them to share a GPU batch.

### Ragged Batching

The term "ragged batching" refers to this technique of concatenating variable-length sequences without padding. The sequences are "ragged" (uneven lengths), but we handle them efficiently through attention masking rather than padding.

---

## Chunked Prefill: Preventing Long-Prompt Starvation

Long prefills can block decode requests. If one request has a 10,000-token prompt, all other requests wait during that prefill. Solution: **chunk prefills**.

```
Long prompt (128 tokens) → Process in chunks of 32

Step 1: [chunk1_32_tokens][decode_tokens_from_other_requests]
Step 2: [chunk2_32_tokens][decode_tokens_from_other_requests]
Step 3: [chunk3_32_tokens][decode_tokens_from_other_requests]
Step 4: [chunk4_32_tokens][decode_tokens_from_other_requests]
        → Now this request joins decode pool
```

**Benefits:**
- Decode requests aren't blocked by long prefills
- Better latency distribution (no request starves)
- Memory usage more predictable
- Prefill chunks make forward pass compute-bound, so decode tokens "piggyback" at nearly zero marginal cost

**Implementation:** Cap the number of new tokens per step. If requested tokens exceed `long_prefill_token_threshold`, truncate to that value. The KV cache handles the rest.

---

## PagedAttention: Solving KV Cache Fragmentation

Continuous batching solves the padding problem, but introduces a new one: **memory fragmentation**.

### The Fragmentation Problem

With dynamic request lifetimes, KV cache memory looks like this:

```
GPU Memory over time:

t=0: [A_KV][B_KV][C_KV][free][free][free]
t=1: [A_KV][____][C_KV][D_KV][free][free]  ← B finished, hole created
t=2: [A_KV][____][____][D_KV][E_KV][free]  ← C finished, another hole
t=3: [A_KV][____][____][D_KV][E_KV][F_KV]
     
     ↑ Can't fit new request G (needs 2 contiguous blocks)
       even though 2 blocks are free!
```

**External fragmentation**: Free memory exists but isn't contiguous.
**Internal fragmentation**: Allocated more than needed (reserving for max length).

### PagedAttention: Virtual Memory for KV Cache

PagedAttention (from the vLLM paper) applies OS virtual memory concepts to KV cache:

```
Physical Memory (fixed-size blocks):
[Block0][Block1][Block2][Block3][Block4][Block5][Block6][Block7]

Request A's Block Table:    Request B's Block Table:
Logical → Physical          Logical → Physical
  0    →    2                 0    →    0
  1    →    5                 1    →    3
  2    →    7                 2    →    1
```

**Key innovations:**

1. **Fixed-size blocks**: KV cache divided into blocks (default: 16 tokens each)
2. **Non-contiguous allocation**: Blocks can be anywhere in physical memory
3. **On-demand allocation**: Only allocate blocks as tokens are generated
4. **Block tables**: Map logical sequence positions to physical blocks

### How PagedAttention Eliminates Waste

```
Without PagedAttention:
- Reserve max_seq_len × batch_size of KV cache upfront
- A 2048-token allocation used by 100-token request = 95% waste

With PagedAttention:
- Allocate blocks as needed (16 tokens at a time)
- 100-token request uses ceil(100/16) = 7 blocks
- No reservation for unused future tokens
```

**Memory savings**: 55-70% reduction in KV cache memory usage (from vLLM paper)

### The Block Lifecycle

```
Free Pool: [B0, B1, B2, B3, B4, B5, ...]

New request arrives:
1. Allocate block from free pool → B0 assigned to Request A
2. Fill block during decode (16 tokens)
3. Allocate next block when full → B3 assigned to Request A
4. Continue until EOS

Request completes:
1. Return all blocks to free pool
2. Optionally: keep for prefix caching (if another request might reuse)
```

---

## Prefix Caching: Reusing Computation

Many requests share common prefixes (system prompts, few-shot examples):

```
Request 1: "[System: You are a helpful assistant...] What is 2+2?"
Request 2: "[System: You are a helpful assistant...] Explain gravity"
Request 3: "[System: You are a helpful assistant...] Write a poem"
                ↑
                Same 500-token prefix!
```

### How Prefix Caching Works

1. **Hash the prefix**: Create a hash of token IDs for each block
2. **Store mapping**: `block_hash → physical_block_id`
3. **On new request**: Check if prefix blocks already exist
4. **Cache hit**: Reuse existing KV blocks, skip prefill for those tokens

```
First request with prefix:
[compute prefix KV] → [store in cache] → [compute suffix]

Subsequent requests:
[cache hit!] → [skip prefix compute] → [compute only suffix]

Speedup: If prefix is 500 tokens and suffix is 50 tokens,
         prefill is ~10× faster on cache hit
```

### Block-Level Caching Mechanics

```
Prefix: "You are a helpful assistant. Answer questions clearly."
        [Block 0: tokens 0-15] [Block 1: tokens 16-31] [Block 2: tokens 32-47]
        
Hash computation (chained for security):
Block 0 hash: hash(prev_hash=None, tokens[0:16], metadata)
Block 1 hash: hash(prev_hash=block0_hash, tokens[16:32], metadata)
Block 2 hash: hash(prev_hash=block1_hash, tokens[32:48], metadata)

Cache lookup on new request:
1. Compute block hashes for incoming tokens
2. Check cached_block_hash_to_block dictionary
3. For each hit: reuse physical block, skip KV computation
4. For first miss: compute from there forward
```

---

## FlashAttention: Memory-Efficient Attention

Standard attention has a memory problem:

```
Standard Attention:
Q × K^T → [N × N] attention matrix → softmax → × V

For N = 8192 tokens:
Attention matrix = 8192 × 8192 × 4 bytes = 256 MB per layer per head!
```

### FlashAttention's Tiling Approach

Instead of materializing the full attention matrix:

```
Standard:
1. Compute full QK^T (N×N matrix in HBM)
2. Apply softmax (read/write N×N from HBM)
3. Multiply by V (read N×N from HBM)
Total HBM access: O(N²)

FlashAttention:
1. Tile Q, K, V into blocks that fit in SRAM
2. For each tile:
   - Load Q_tile, K_tile, V_tile to SRAM
   - Compute attention for that tile
   - Accumulate result with online softmax
3. Never materialize full N×N matrix

Memory: O(N) instead of O(N²)
Speed: 2-4× faster (fewer HBM round trips)
```

### FlashAttention + PagedAttention

vLLM combines both:
- **PagedAttention**: Efficient KV cache memory management
- **FlashAttention kernels**: Efficient attention computation

The attention kernel reads from non-contiguous physical blocks using the block table:

```python
# Conceptual PagedAttention kernel logic
for q_idx in range(num_query_tokens):
    output = 0
    for block_idx in block_table[sequence_id]:
        k_block = kv_cache[block_idx].keys
        v_block = kv_cache[block_idx].values
        # Compute attention for this block
        attn_weights = softmax(q @ k_block.T)
        output += attn_weights @ v_block
```

---

## CUDA Graphs: Eliminating Launch Overhead

Each GPU operation has CPU-side launch overhead:

```
Without CUDA Graphs:
CPU: [launch kernel 1] → [launch kernel 2] → [launch kernel 3]
GPU:      [kernel 1]          [kernel 2]          [kernel 3]
                    ↑ idle time between launches (~10-50μs each)
```

For decode (many small kernels), launch overhead can dominate!

### How CUDA Graphs Help

```
Capture phase (once at warmup):
1. Run forward pass with dummy data
2. Record all GPU operations into a graph

Replay phase (every inference):
1. Update input tensors in-place
2. Launch entire graph with single CPU call
3. All kernels execute back-to-back

Speedup: 5-15% latency reduction for decode-heavy workloads
```

**Constraint**: CUDA graphs require static shapes. This is why vLLM:
- Captures graphs at warmup for common batch sizes
- Falls back to eager mode for uncommon sizes
- Uses `enforce_eager: false` by default to enable graphs

---

## The vLLM Architecture

vLLM combines all these techniques into a cohesive system:

```
┌─────────────────────────────────────────────────────────────┐
│                      vLLM Engine                             │
├─────────────────────────────────────────────────────────────┤
│  Scheduler                                                   │
│  ├── Waiting Queue (new requests)                           │
│  ├── Running Queue (active generation)                      │
│  └── Policy: FCFS or Priority                               │
├─────────────────────────────────────────────────────────────┤
│  KV Cache Manager                                            │
│  ├── Block Allocator (PagedAttention)                       │
│  ├── Free Block Queue                                        │
│  ├── Prefix Cache (block_hash → block_id)                   │
│  └── Block Tables (request_id → block_list)                 │
├─────────────────────────────────────────────────────────────┤
│  Model Executor                                              │
│  ├── Worker(s) - one per GPU                                │
│  ├── Model Runner (forward pass)                            │
│  ├── Sampler (token selection)                              │
│  └── CUDA Graph Cache                                        │
└─────────────────────────────────────────────────────────────┘
```

### The Engine Step Loop

```python
while has_pending_requests():
    # 1. Schedule: Select requests for this step
    scheduled = scheduler.schedule()
    # - Prioritize decode (running) over prefill (waiting)
    # - Respect token budget (max_num_batched_tokens)
    # - Allocate KV cache blocks via PagedAttention
    
    # 2. Forward Pass
    # - Prepare inputs (flatten sequences, build attention metadata)
    # - Execute model (use CUDA graph if available)
    # - Sample next tokens
    
    # 3. Postprocess
    # - Append tokens to sequences
    # - Check stop conditions (EOS, max_tokens, stop strings)
    # - Free KV blocks for completed requests
    # - Return outputs (streaming or batch)
```

### Key Configuration Parameters

| Parameter | What it controls | Tuning guidance |
|-----------|------------------|-----------------|
| `max_num_seqs` | Max concurrent requests | Higher = more throughput, more memory |
| `max_num_batched_tokens` | Token budget per step | Balance prefill vs decode |
| `max_model_len` | Max context length | Affects KV cache reservation |
| `gpu_memory_utilization` | % of GPU for KV cache | 0.9 typical, lower if OOM |
| `block_size` | Tokens per KV block | 16 default, rarely changed |
| `enable_prefix_caching` | Prefix cache on/off | True for shared prompts |

---

## Quantization: Reducing Memory and Compute

### Weight Quantization

```
FP16 weights: 7B params × 2 bytes = 14 GB
INT8 weights: 7B params × 1 byte = 7 GB
INT4 weights: 7B params × 0.5 bytes = 3.5 GB
```

**Trade-offs:**
- Lower precision → smaller memory footprint
- Faster memory loading → higher tokens/second
- Potential accuracy degradation (model-dependent)

### KV Cache Quantization

```
FP16 KV cache: 512 KB per token
FP8 KV cache: 256 KB per token  (2× more concurrent requests!)
INT4 KV cache: 128 KB per token (4× more concurrent requests!)
```

### When to Quantize

| Bottleneck | Weight Quant Helps? | KV Quant Helps? |
|------------|---------------------|-----------------|
| Compute-bound | ✅ (less compute) | ❌ |
| Memory-bandwidth-bound | ✅ (less data to load) | ❌ |
| Memory-capacity-bound | ✅ | ✅ (fit more requests) |

---

## Key Takeaways from Episode 2

1. **Continuous batching** eliminates padding waste by batching at the token level with attention masks — requests join and leave dynamically

2. **Chunked prefill** prevents long prompts from blocking decode requests — better latency distribution for all

3. **PagedAttention** solves KV cache fragmentation using OS-style virtual memory (fixed blocks + block tables) — 55-70% memory savings

4. **Prefix caching** reuses KV computations for shared prefixes — 10× prefill speedup for repeated system prompts

5. **FlashAttention** reduces attention memory from O(N²) to O(N) through tiling — 2-4× faster attention

6. **CUDA graphs** eliminate kernel launch overhead — 5-15% latency reduction for decode

7. **vLLM** combines all these into a production system achieving 2-24× throughput vs naive serving

---

**Next Episode:** Advanced techniques — speculative decoding, guided decoding, disaggregated prefill/decode, distributed serving, and production deployment patterns.

---

*References:*
- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [HuggingFace Blog: Continuous Batching](https://huggingface.co/blog/continuous_batching)
- [Aleksa Gordić: Inside vLLM](https://www.aleksagordic.com/blog/vllm)
