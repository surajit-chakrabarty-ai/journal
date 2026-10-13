---
date: 2026-10-13 18:38
---
# LLM Inference Deep Dive: Episode 3 — Production-Grade Serving

**TL;DR:** Moving from a single vLLM instance to a production fleet requires understanding speculative decoding, model parallelism, disaggregated architectures, and the serving infrastructure that ties it all together.

---

## Speculative Decoding: Breaking the Sequential Barrier

Decode is inherently sequential — each token depends on the previous one. But what if we could *speculate* about future tokens?

### The Core Idea

Use a small, fast "draft" model to propose K tokens, then verify them with the large "target" model in parallel:

```
Traditional (7B model, 5 tokens):
Step 1: Generate token 1  (~4ms)
Step 2: Generate token 2  (~4ms)
Step 3: Generate token 3  (~4ms)
Step 4: Generate token 4  (~4ms)
Step 5: Generate token 5  (~4ms)
Total: ~20ms for 5 tokens

Speculative (draft + 7B verify):
Draft:  Generate 5 tokens with 0.5B model  (~2ms)
Verify: Check all 5 with 7B model in ONE pass (~5ms)
Accept: If all match, done in ~7ms for 5 tokens!
        If mismatch at token 3, still got 3 tokens in ~7ms
```

### The Mathematics of Acceptance

The draft model's token might differ from what the target would generate. The acceptance algorithm ensures we sample from the *exact same distribution* as the target model:

```
For each draft token (left to right):
  p_target = target_model_probability(token)
  p_draft = draft_model_probability(token)
  
  if p_target >= p_draft:
      ACCEPT (draft was conservative)
  else:
      ACCEPT with probability p_target / p_draft
      REJECT otherwise → stop, resample from adjusted distribution
```

**Key guarantee**: The output distribution is *identical* to sampling from the target model alone. Speculative decoding is a speedup technique, not an approximation.

### Speculative Decoding Variants in vLLM

**1. Draft Model (Original)**
- Separate smaller model (e.g., 0.5B draft for 7B target)
- Best when draft model is well-aligned with target
- Overhead: Must maintain two models in memory

**2. N-gram Speculation**
- Look for repeated patterns in the sequence so far
- If "the cat sat on the" appeared before followed by "mat", propose "mat"
- Zero overhead, works for repetitive content

```python
# vLLM n-gram config
speculative_config = {
    "method": "ngram",
    "prompt_lookup_max": 5,  # Look back up to 5 tokens
    "prompt_lookup_min": 3,
    "num_speculative_tokens": 3,  # Propose 3 tokens
}
```

**3. Medusa**
- Add lightweight "speculation heads" to the target model
- Each head predicts a future token position in parallel
- Single model, lower memory overhead
- Requires fine-tuning the heads

**4. EAGLE**
- Keep embeddings and LM head from target model
- Replace transformer stack with lightweight MLP
- Fine-tune as cheap draft model
- Better alignment than separate draft model

### When Speculative Decoding Helps

```
Speedup ≈ (avg_tokens_accepted + 1) / (1 + draft_time/verify_time)

High acceptance rate (>70%) + cheap draft → 2-3× speedup
Low acceptance rate (<40%) + expensive draft → might be slower!
```

**Best for:**
- Long-form generation (more tokens to speculate)
- Predictable content (code completion, structured output)
- Memory-bound scenarios (draft amortizes weight loading)

**Not helpful for:**
- Short responses (overhead dominates)
- Highly creative/unpredictable content (low acceptance)
- Already compute-saturated systems

---

## Guided Decoding: Constraining the Output Space

Sometimes you need structured output (JSON, specific choices, regex patterns). Guided decoding constrains which tokens can be sampled.

### Finite State Machine (FSM) Approach

```
Grammar: output must be "Positive" or "Negative"

FSM States:
           ┌─"P"─→ [o] ─→ [s] ─→ [i] ─→ [t] ─→ [i] ─→ [v] ─→ [e] ─→ ACCEPT
[START] ───┤
           └─"N"─→ [e] ─→ [g] ─→ [a] ─→ [t] ─→ [i] ─→ [v] ─→ [e] ─→ ACCEPT

At each decode step:
1. Determine valid next tokens from current FSM state
2. Create bitmask: 1 for valid tokens, 0 for invalid
3. Apply mask to logits: invalid tokens → -∞
4. Sample from masked distribution
5. Advance FSM state
```

### Implementation in vLLM

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Choice constraint
guided = GuidedDecodingParams(choice=["Positive", "Negative"])
params = SamplingParams(guided_decoding=guided)

# JSON schema constraint
guided = GuidedDecodingParams(json_schema={
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
})

# Regex constraint
guided = GuidedDecodingParams(regex=r"\d{3}-\d{4}")
```

### The Bitmask Mechanism

```
Vocabulary size: 32,000 tokens
Bitmask: 32,000 bits = 1,000 integers (32-bit each)

At each step:
1. FSM determines valid tokens (e.g., only "P" and "N" valid)
2. Backend (xgrammar) produces bitmask
3. Expand bitmask to vocab-size tensor
4. logits[invalid] = -∞
5. Sample from remaining distribution
```

**Supported grammars:**
- Regular expressions (Chomsky type-3)
- Context-free grammars (type-2) — covers JSON, most programming languages
- JSON schemas (converted to CFG internally)

---

## Disaggregated Prefill/Decode

Prefill and decode have fundamentally different characteristics:

| Aspect | Prefill | Decode |
|--------|---------|--------|
| Compute profile | Compute-bound | Memory-bandwidth-bound |
| Batch behavior | Benefits from large batches | Benefits from continuous batching |
| Latency sensitivity | Affects TTFT | Affects ITL |
| Resource usage | Bursty, high intensity | Steady, lower intensity |

### The Disaggregation Architecture

Run separate instance pools for prefill and decode:

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
    ┌─────────────────┐            ┌─────────────────┐
    │  Prefill Pool   │            │  Decode Pool    │
    │  (N instances)  │            │  (M instances)  │
    │  Compute-opt    │            │  Memory-opt     │
    └────────┬────────┘            └────────┬────────┘
             │                              ▲
             │      ┌──────────────┐        │
             └─────►│  KV Cache    │────────┘
                    │  Service     │
                    └──────────────┘
```

**Benefits:**
- Scale prefill and decode independently based on traffic mix
- Optimize hardware for each workload type
- Isolate bursty prefill from latency-sensitive decode
- Better resource utilization overall

### KV Cache Transfer

After prefill completes, KV cache must move to decode workers:

```python
# Conceptual flow
# Prefill worker
kv_cache = prefill_model.forward(prompt_tokens)
kv_service.store(request_id, kv_cache)
notify_decode_pool(request_id)

# Decode worker
kv_cache = kv_service.load(request_id)
while not done:
    token = decode_model.forward(prev_token, kv_cache)
    kv_cache.append(token_kv)
```

**Transfer mechanisms:**
- Shared storage (for debugging/simple setups)
- RDMA/NVLink (for production, low latency)
- LMCache with NIXL backend (fastest production option)

---

## Scaling Up: Multi-GPU Execution

When a model doesn't fit on one GPU, or you need more throughput:

### Tensor Parallelism (TP)

Split each layer's weights across GPUs:

```
Single GPU:
[Full Weight Matrix W]
        ↓
    [Output]

TP=4 (4 GPUs):
GPU0: [W_0] ─┐
GPU1: [W_1] ─┼─► AllReduce ─► [Output]
GPU2: [W_2] ─┤
GPU3: [W_3] ─┘

Each GPU holds 1/4 of weights, computes partial result,
then AllReduce combines them.
```

**Characteristics:**
- High communication (AllReduce after every layer)
- Best within a node (fast NVLink)
- Linear memory scaling
- Sub-linear compute scaling (communication overhead)

### Pipeline Parallelism (PP)

Split layers across GPUs:

```
PP=4 (4 GPUs):
GPU0: [Layers 0-7]   ─► GPU1: [Layers 8-15]  ─►
GPU2: [Layers 16-23] ─► GPU3: [Layers 24-31]

Micro-batching fills the pipeline:
Time →
GPU0: [B1][B2][B3][B4]
GPU1:    [B1][B2][B3][B4]
GPU2:       [B1][B2][B3][B4]
GPU3:          [B1][B2][B3][B4]
```

**Characteristics:**
- Lower communication (only between pipeline stages)
- Works across nodes
- Pipeline bubbles reduce efficiency
- Requires micro-batching to fill pipeline

### Combining TP and PP

For very large models:

```
8 nodes × 8 GPUs = 64 GPUs total

Configuration: TP=8, PP=8
- Within each node: TP=8 (all 8 GPUs share each layer)
- Across nodes: PP=8 (each node handles different layers)

Model capacity: 64× single GPU
```

### vLLM Multi-GPU Configuration

```python
# Single node, 4 GPUs with tensor parallelism
vllm serve model-name --tensor-parallel-size 4

# Multi-node with both TP and PP
vllm serve model-name \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray
```

---

## The Distributed Serving Stack

For production deployment at scale:

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway                               │
│                    (Rate limiting, auth)                         │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Load Balancer                               │
│              (Route to least-loaded replica)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  API Server   │       │  API Server   │       │  API Server   │
│  (FastAPI)    │       │  (FastAPI)    │       │  (FastAPI)    │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Engine Core   │       │ Engine Core   │       │ Engine Core   │
│ (vLLM/TGI)    │       │ (vLLM/TGI)    │       │ (vLLM/TGI)    │
│ [GPU 0-3]     │       │ [GPU 0-3]     │       │ [GPU 0-3]     │
└───────────────┘       └───────────────┘       └───────────────┘
```

### Data Parallelism for Throughput

With DP > 1, replicate the full model across GPU groups:

```
DP=4, TP=2 on 8 GPUs:

Replica 0: GPU 0-1 (TP=2)  ─┐
Replica 1: GPU 2-3 (TP=2)  ─┼─► Load balancer routes requests
Replica 2: GPU 4-5 (TP=2)  ─┤
Replica 3: GPU 6-7 (TP=2)  ─┘

Each replica handles independent requests.
4× throughput vs single replica.
```

### Load Balancing Strategy

vLLM's DP coordinator uses a simple but effective metric:

```python
score = len(waiting_queue) * 4 + len(running_queue)

# Route new request to replica with lowest score
# Waiting requests weighted higher (they're blocked)
```

---

## Inference Backends: The Server vs Engine Distinction

Understanding the stack:

```
┌─────────────────────────────────────────┐
│           Inference Server              │  ← Handles HTTP, batching, scheduling
│  (Triton, vLLM server, TGI)            │
├─────────────────────────────────────────┤
│           Inference Engine              │  ← Executes the model
│  (TensorRT, ONNX Runtime, vLLM engine) │
├─────────────────────────────────────────┤
│           Model Format                  │  ← The weights
│  (.plan, .onnx, .safetensors)          │
└─────────────────────────────────────────┘
```

### Common Stacks by Model Type

| Model Type | Recommended Stack | Why |
|------------|-------------------|-----|
| Encoder (BERT) | Triton + TensorRT | Max throughput, batching |
| Embeddings (CTE) | Triton + ONNX/TRT | Flexibility + speed |
| Decoder LLM | Triton + vLLM backend | Continuous batching, PagedAttention |
| VLM | Triton + vLLM | Same, plus image handling |

### Triton Inference Server

Triton provides the serving infrastructure:

```
# config.pbtxt for a TensorRT model
name: "bert_classifier"
platform: "tensorrt_plan"
max_batch_size: 32

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 1000  # 1ms queue delay
}

instance_group [
  { count: 2, kind: KIND_GPU, gpus: [0] }
]
```

**Key features:**
- Dynamic batching with configurable queue delay
- Ensemble pipelines (preprocessing → model → postprocessing)
- Multiple model instances per GPU
- Health checks and metrics

---

## Optimization Techniques Summary

### Kernel-Level Optimizations

| Technique | What it does | When to use |
|-----------|--------------|-------------|
| **Kernel Fusion** | Combine multiple ops into one kernel | Always (reduces memory traffic) |
| **torch.compile** | JIT compile PyTorch to optimized kernels | When TensorRT won't work |
| **TensorRT** | Compile to highly optimized GPU kernels | Encoder models, static shapes |
| **CUDA Graphs** | Capture and replay kernel sequences | Decode-heavy workloads |

### Memory Optimizations

| Technique | What it does | When to use |
|-----------|--------------|-------------|
| **PagedAttention** | Virtual memory for KV cache | Always for LLMs |
| **KV Cache Quantization** | FP8/INT4 KV cache | Memory-capacity-bound |
| **Weight Quantization** | INT8/INT4 weights | Memory or compute bound |
| **FlashAttention** | O(N) memory attention | Long contexts |

### Throughput Optimizations

| Technique | What it does | When to use |
|-----------|--------------|-------------|
| **Continuous Batching** | Dynamic request scheduling | Always for LLMs |
| **Chunked Prefill** | Break long prefills into chunks | Mixed workloads |
| **Prefix Caching** | Reuse KV for shared prefixes | Repeated system prompts |
| **Speculative Decoding** | Parallel token verification | Predictable outputs |

### Latency Optimizations

| Technique | What it does | When to use |
|-----------|--------------|-------------|
| **CUDA Graphs** | Eliminate launch overhead | Decode phase |
| **Disaggregated P/D** | Separate prefill from decode | Strict latency SLAs |
| **Smaller batch sizes** | Trade throughput for latency | Real-time applications |

---

## Benchmarking: Measuring What Matters

### Key Metrics to Track

```
Latency Metrics:
- TTFT (Time to First Token): Prefill latency
- ITL (Inter-Token Latency): Decode latency per token
- E2E Latency: Total request time
- P50/P90/P99: Distribution matters more than mean

Throughput Metrics:
- Tokens/second (input + output)
- Requests/second
- Goodput: Throughput meeting SLO constraints

Efficiency Metrics:
- GPU Utilization (but interpret carefully!)
- GPU Memory Utilization
- Actual batch size (InferenceCount/ExecutionCount)
```

### vLLM Benchmarking Tools

```bash
# Latency benchmark (small batch, measure per-request)
vllm bench latency \
    --model model-name \
    --input-tokens 32 \
    --output-tokens 128 \
    --batch-size 8

# Throughput benchmark (flood with requests)
vllm bench throughput \
    --model model-name \
    --dataset sharegpt \
    --num-prompts 1000

# Server benchmark (realistic traffic patterns)
vllm bench serve \
    --model model-name \
    --request-rate 10 \  # requests per second
    --duration 300       # 5 minutes
```

### The Latency-Throughput Curve

Always measure both together:

```
Throughput
    ↑
    │                    ╭────── Saturation
    │                 ╭──╯
    │              ╭──╯
    │           ╭──╯
    │        ╭──╯
    │     ╭──╯
    │  ╭──╯
    │──╯
    └────────────────────────────→ Latency (P99)
    
    Operating point = knee of the curve
    (maximize throughput while meeting latency SLO)
```

---

## Production Checklist

### Before Launch

- [ ] **Classify your model**: Encoder, decoder, embedding, VLM?
- [ ] **Identify bottleneck**: Compute, memory-bandwidth, memory-capacity, or overhead?
- [ ] **Choose stack**: Triton + TensorRT, Triton + vLLM, standalone vLLM?
- [ ] **Set latency SLO**: What P99 latency is acceptable?
- [ ] **Measure baseline**: Current throughput and latency
- [ ] **Size the fleet**: Instances needed = Peak TPS / Per-instance TPS × safety factor

### Configuration Tuning

- [ ] **Batching**: Enable dynamic batching, tune queue delay (1-5ms typical)
- [ ] **Instance count**: Profile to find GPU vs CPU bottleneck
- [ ] **KV cache**: Right-size `max_model_len`, enable prefix caching if applicable
- [ ] **Quantization**: Test accuracy impact before deploying

### Monitoring

- [ ] **Latency alarms**: P50, P90, P99 against SLO
- [ ] **Throughput tracking**: Requests/sec, tokens/sec
- [ ] **GPU metrics**: Utilization, memory, queue depth
- [ ] **Error rates**: Timeouts, OOMs, model errors

---

## Key Takeaways from Episode 3

1. **Speculative decoding** breaks the sequential decode barrier — 2-3× speedup for predictable content while maintaining exact output distribution

2. **Guided decoding** constrains outputs to valid grammars (JSON, regex, choices) using FSM-based logit masking

3. **Disaggregated P/D** separates compute-bound prefill from memory-bound decode for independent scaling and better resource utilization

4. **Tensor parallelism** splits layers across GPUs (best within node), **pipeline parallelism** splits layers across stages (works across nodes)

5. **The serving stack** has layers: Server (Triton) → Engine (vLLM/TensorRT) → Model — choose the right combination for your model type

6. **Benchmark both latency AND throughput** — the operating point is the knee of the curve where you maximize throughput while meeting latency SLO

7. **Production requires**: Right-sized configuration, proper monitoring, and understanding your specific bottleneck

---

## Series Conclusion

Across these three episodes, we've journeyed from the fundamental physics of LLM inference to production-scale serving:

**Episode 1** established *why* LLM serving is hard: the memory bandwidth wall, the prefill/decode dichotomy, and the KV cache trade-off.

**Episode 2** covered *how* modern systems solve these problems: continuous batching eliminates padding, PagedAttention solves fragmentation, and FlashAttention makes attention tractable.

**Episode 3** showed *what* it takes to go to production: speculative decoding for speed, parallelism for scale, and the right serving stack for your workload.

The field moves fast — new techniques emerge monthly. But the fundamentals remain: understand your bottleneck, measure before optimizing, and remember that the goal isn't maximum GPU utilization — it's meeting your latency SLO at minimum cost.

---

*References:*
- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [Medusa: Simple LLM Inference Acceleration Framework](https://arxiv.org/abs/2401.10774)
- [XGrammar: Flexible and Efficient Structured Generation](https://arxiv.org/abs/2411.15100)
- [Aleksa Gordić: Inside vLLM](https://www.aleksagordic.com/blog/vllm)
- [HuggingFace Blog: Continuous Batching](https://huggingface.co/blog/continuous_batching)
