# Episode 4: The Capacity Planner — GPU Sizing, Fleet Math, and the Art of Not Going Broke

**TL;DR:** Before you spin up a single GPU, you need to answer: "How many GPUs do I actually need?" This episode teaches you to estimate capacity from first principles — starting with requests per second, working through the physics of your workload, and arriving at a defensible fleet size that won't bankrupt you or leave users waiting.

---

## The Hook: A $2 Million Conversation

Every AI startup has this conversation exactly once:

> **PM:** "We're launching next month. How many GPUs do we need?"
> **Engineer:** "Uh... 50?"
> **PM:** "Why 50?"
> **Engineer:** "It's a nice round number?"

Three months later, one of two things happens. Either those 50 GPUs sit at 15% utilization, burning $1.5 million per year on idle silicon. Or they hit 200% of capacity, users churn, and revenue evaporates. Both scenarios end the same way: the engineer updates their LinkedIn.

Capacity planning isn't guessing. It's physics, economics, and a bit of queueing theory. This episode gives you the systematic approach to get it right.

---

## The Journey: From Request to Fleet Size

We'll build up the capacity planning methodology in five stages, each building on the last:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE CAPACITY PLANNING JOURNEY                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Stage 1: Know Your Traffic                                        │
│      ↓                                                               │
│   Stage 2: Classify Your Workload                                   │
│      ↓                                                               │
│   Stage 3: Understand the Bottlenecks                               │
│      ↓                                                               │
│   Stage 4: Measure (Don't Estimate) Per-GPU Capacity                │
│      ↓                                                               │
│   Stage 5: Size the Fleet with Safety Margins                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Let's walk through each stage.

---

## Stage 1: Know Your Traffic

Before touching any GPU, you must answer fundamental questions about your traffic. Without these numbers, every subsequent calculation is fiction.

### The Essential Metrics

**Requests Per Second (RPS)** is your primary input. But you need three variants:

- **Peak RPS**: The maximum load during your busiest moment. This is what you size for.
- **Average RPS**: Typical steady-state load. This determines your baseline cost.
- **Burst Factor**: The ratio of peak to average. A burst factor of 5× means your peaks are five times your average.

**Tokens Per Second (TPS)** translates requests into actual GPU work:

```
TPS = RPS × (Average Input Tokens + Average Output Tokens)
```

A service handling 100 RPS with 500-token inputs and 200-token outputs processes 70,000 tokens per second. That's the real workload.

**Concurrency** tells you how many requests are in-flight simultaneously. Little's Law gives us this directly:

```
Concurrency = RPS × Average Latency (in seconds)
```

If you handle 100 RPS with 2-second average latency, you have 200 concurrent requests at any moment. This number determines your memory requirements.

### The Traffic Distribution Pyramid

Not all requests are equal. Understanding your distribution is critical:

```
                              ╱╲
                             ╱  ╲
                            ╱ P99╲        1% of requests
                           ╱ Long ╲       2000 input tokens, 500 output
                          ╱────────╲      These set your MAXIMUM capacity
                         ╱   P90    ╲     10% of requests
                        ╱  Medium    ╲    500 input tokens, 200 output
                       ╱──────────────╲   
                      ╱      P50       ╲  50% of requests
                     ╱     Typical      ╲ 200 input tokens, 100 output
                    ╱────────────────────╲ This is your AVERAGE case
                   ╱        Short         ╲ Many requests
                  ╱    Quick responses     ╲ 50 input tokens, 20 output
                 ╱──────────────────────────╲ These are CHEAP
                ╱____________________________╲
```

**The key insight:** You must size for P99 (to avoid tail latency disasters) but optimize for P50 (where most of your cost lives). A system that handles average cases beautifully but falls over on long requests will have terrible P99 latency.

### Traffic Patterns Matter

Traffic rarely arrives uniformly. Common patterns include:

**Time-of-day variation**: Consumer apps peak during evening hours. B2B services peak during business hours. A 3× difference between peak and trough is common.

**Weekly cycles**: Many services see 20-30% lower traffic on weekends.

**Burst events**: Product launches, viral moments, or marketing campaigns can spike traffic 10× or more with little warning.

**Geographic distribution**: A global service sees traffic follow the sun. Peak in Asia, then Europe, then Americas.

Understanding these patterns enables predictive scaling — spinning up capacity before you need it, not after users start seeing errors.

---

## Stage 2: Classify Your Workload

The capacity math differs dramatically based on what your model actually does. There are two fundamentally different workload types, and confusing them will lead you astray.

### Classification Workloads

Classification tasks process an input and emit a small, fixed output — a label, a score, a single token. Examples include:

- Sentiment analysis
- Content moderation  
- Topic classification
- Spam detection
- LLM-as-judge (outputting a single letter grade)

The compute profile of classification is **prefill-dominated**:

```
┌────────────────────────────────────────────────────────────────┐
│                  CLASSIFICATION COMPUTE PROFILE                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: "This product is terrible, worst purchase ever"        │
│  Output: "negative" (1 token)                                  │
│                                                                 │
│  [████████████████████████████████████████████████] [█]        │
│   ↑ Prefill: 95% of compute                         ↑          │
│                                              Decode: 5%        │
│                                                                 │
│  Characteristics:                                               │
│  • Compute-bound (GPU arithmetic units are the bottleneck)     │
│  • Batching provides LINEAR speedup                            │
│  • Latency is predictable (no variable-length generation)      │
│  • Memory pressure is LOW (KV cache doesn't grow)              │
│  • Typical throughput: 100-1000+ RPS per GPU                   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

For classification, **batching is your best friend**. Processing 32 requests together takes only slightly longer than processing 1, because the GPU's parallel compute units are finally being utilized. A well-optimized classification service can achieve 10-20× higher throughput than a naive implementation simply by batching effectively.

### Generation Workloads

Generation tasks produce variable-length outputs token by token. Examples include:

- Chatbots and conversational AI
- Code generation and completion
- Summarization
- Creative writing
- Any task where output length varies

The compute profile of generation is **decode-dominated**:

```
┌────────────────────────────────────────────────────────────────┐
│                   GENERATION COMPUTE PROFILE                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: "Write a poem about machine learning"                  │
│  Output: "In silicon dreams where neurons play..." (200 tok)   │
│                                                                 │
│  [████] [█][█][█][█][█][█][█][█][█][█][█][█][█][█][█]...       │
│   ↑      ↑                                                      │
│  Prefill Decode steps (one forward pass per output token)      │
│  (20%)   (80% of total time)                                   │
│                                                                 │
│  Characteristics:                                               │
│  • Memory-bandwidth-bound (loading weights dominates)          │
│  • Batching helps but with DIMINISHING returns                 │
│  • Latency is VARIABLE (depends on output length)              │
│  • Memory pressure is HIGH (KV cache grows with each token)    │
│  • Typical throughput: 5-50 RPS per GPU                        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

For generation, **memory is usually the binding constraint**, not compute. Each concurrent request needs KV cache memory that grows with sequence length. You might have plenty of compute headroom but run out of memory after 10 concurrent requests.

### The Comparison

| Aspect | Classification | Generation |
|--------|---------------|------------|
| Bottleneck | Compute (prefill) | Memory bandwidth (decode) |
| RPS per GPU | 100-1000+ | 5-50 |
| Batching benefit | Linear (huge wins) | Sub-linear (diminishing) |
| Memory scaling | Constant | Grows with concurrency |
| Latency variance | Low | High (output-dependent) |
| Primary metric | Throughput (TPS) | TTFT + ITL |
| Optimization focus | Maximize batch size | Maximize concurrency |

**Critical insight:** A single GPU that handles 500 classification requests per second might only handle 10 generation requests per second. The 50× difference comes from the fundamental physics of the workloads, not from poor optimization.


---

## Stage 3: Understand the Bottlenecks

Every inference workload hits one of three bottlenecks. Identifying which one you face determines your optimization strategy.

### Bottleneck 1: Compute-Bound

A workload is compute-bound when the GPU's arithmetic units are the limiting factor. The GPU is doing math as fast as it can, and adding more memory bandwidth wouldn't help.

**Symptoms:**
- GPU utilization is high (70-95%)
- GPU memory utilization is moderate
- Latency scales linearly with input size
- Batching provides near-linear throughput gains

**Typical workloads:** Classification, embedding generation, short-output generation, prefill phase of any model.

**Optimization levers:**
- Use faster GPUs (more TFLOPS)
- Quantization (INT8/INT4 uses tensor cores more efficiently)
- Kernel fusion (reduce memory round-trips)
- Larger batches (amortize overhead)

### Bottleneck 2: Memory-Bandwidth-Bound

A workload is memory-bandwidth-bound when the GPU spends most of its time waiting for data to arrive from HBM (High Bandwidth Memory). The arithmetic units are idle, waiting for weights to load.

**Symptoms:**
- GPU utilization appears high but actual compute is low
- Memory bandwidth is saturated (check with profiler)
- Batching helps but with diminishing returns
- Latency doesn't improve much with faster compute

**Typical workloads:** Decode phase of generation, especially at low batch sizes.

**Why this happens:** During decode, you load the entire model's weights (14GB for a 7B model) just to compute a single token. The arithmetic intensity is extremely low — a few operations per byte loaded.

**Optimization levers:**
- Use GPUs with higher memory bandwidth (H100 > A100 > L4)
- Quantization (smaller weights = less to load)
- Speculative decoding (generate multiple tokens per weight load)
- Increase batch size (amortize weight loading across more tokens)

### Bottleneck 3: Memory-Capacity-Bound

A workload is memory-capacity-bound when you simply run out of GPU memory. You can't fit more concurrent requests because the KV cache would exceed available VRAM.

**Symptoms:**
- GPU memory utilization is very high (85-95%)
- GPU compute utilization is moderate or low
- Adding more concurrent requests causes OOM errors
- Throughput plateaus despite available compute

**Typical workloads:** Long-context generation, high-concurrency serving, large models on small GPUs.

**The KV cache math:** For a typical 7B model with 4K context, each concurrent request needs approximately 1GB of KV cache memory. On a 24GB GPU with 14GB of weights, you have ~8GB for KV cache — meaning only 8 concurrent requests maximum.

**Optimization levers:**
- Use GPUs with more memory
- Quantize KV cache (FP8 instead of FP16)
- Reduce max context length if possible
- Use PagedAttention (eliminates fragmentation)
- Consider model parallelism (spread across GPUs)

### Identifying Your Bottleneck

The diagnostic process:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BOTTLENECK DIAGNOSIS FLOWCHART                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Start: Run your workload at target concurrency                     │
│                    │                                                 │
│                    ▼                                                 │
│         ┌─────────────────────┐                                     │
│         │ GPU Memory > 85%?   │                                     │
│         └─────────────────────┘                                     │
│            │ Yes          │ No                                      │
│            ▼              ▼                                         │
│    ┌──────────────┐  ┌─────────────────────┐                       │
│    │ MEMORY       │  │ GPU Compute > 70%?  │                       │
│    │ CAPACITY     │  └─────────────────────┘                       │
│    │ BOUND        │     │ Yes         │ No                         │
│    └──────────────┘     ▼             ▼                            │
│                    ┌──────────┐  ┌──────────────┐                  │
│                    │ COMPUTE  │  │ MEMORY       │                  │
│                    │ BOUND    │  │ BANDWIDTH    │                  │
│                    └──────────┘  │ BOUND        │                  │
│                                  └──────────────┘                  │
│                                                                      │
│  Note: "GPU Compute" here means actual SM utilization, not the     │
│  misleading "GPU Utilization" metric that just shows if any        │
│  kernel is running.                                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 4: Measure Per-GPU Capacity

This is the most critical stage, and the one most often done wrong. **You must measure your actual throughput on your actual hardware with your actual workload.** Theoretical calculations are useful for intuition but useless for production sizing.

### Why Measurement Beats Calculation

Theoretical capacity calculations fail because they ignore:

- Kernel launch overhead
- Memory fragmentation
- Framework inefficiencies
- Tokenization and pre/post-processing time
- Network and serialization overhead
- The specific operator mix of your model
- Your particular batch size and sequence length distribution

A model that "should" do 100 tokens/second based on FLOP counting might actually do 40 tokens/second in practice. Or it might do 150 tokens/second because the framework has optimizations you didn't account for. **The only way to know is to measure.**

### The Benchmarking Protocol

Follow this protocol to get reliable per-GPU capacity numbers:

**Step 1: Prepare a Representative Workload**

Sample at least 1,000 requests from production (or create synthetic data matching your expected distribution). Include:
- P50, P90, and P99 input lengths
- P50, P90, and P99 output lengths (for generation)
- Representative prompt templates and content types

**Step 2: Find the Latency-Throughput Curve**

Run load tests at increasing concurrency levels:

```
For concurrency in [1, 2, 4, 8, 16, 32, 64, 128]:
    Run load test for 60 seconds
    Record: throughput (RPS), P50 latency, P99 latency
    Plot the results
```

This produces the most important graph in capacity planning:

```
                    THE LATENCY-THROUGHPUT CURVE
    
    P99 Latency │
    (ms)        │                                    ╱
                │                                  ╱
         2000   │ - - - - - - - - - - - - - - - -╱- - - YOUR SLA
                │                              ╱│
                │                            ╱  │
         1000   │                          ╱    │
                │                        ╱      │
                │            ┌─────────╱        │
          500   │            │ "KNEE" ──────────│─── Operating Point
                │            │       ╱          │
                │      ╱─────┘     ╱            │
          200   │    ╱           ╱              │
                │  ╱           ╱                │
          100   │╱           ╱                  │
                └────────────┬──────────────────┴────────────────▶
                             │                  │        Throughput
                            50                 120       (RPS)
                                                │
                                    ◄───────────┘
                              This is your RPS_per_GPU
```

**Step 3: Find Your Operating Point**

Your operating point is the throughput where P99 latency equals your SLA. Read this value from the curve — this is your **RPS_per_GPU**.

**Critical warning:** Do not use maximum throughput. Use throughput at your latency SLA. A GPU might achieve 200 RPS with 5-second P99 latency, but if your SLA is 500ms, your actual capacity might be only 80 RPS.

**Step 4: Validate at Scale**

Before trusting your numbers:
- Test with 2-3 GPUs to verify linear scaling
- Test failure scenarios (kill a GPU mid-test)
- Test cold start (first request after deployment)
- Test sustained load (run for 30+ minutes to catch memory leaks)

### Reading the Curve: Three Zones

The latency-throughput curve has three distinct zones:

**Zone 1: Underutilized** — Latency is flat regardless of load. The system has excess capacity. You're paying for GPUs you don't need.

**Zone 2: Optimal** — Latency rises gradually with load. The system is well-utilized. This is where you want to operate.

**Zone 3: Overloaded** — Latency explodes. Queues are building up. Users are unhappy. You need more capacity immediately.

The "knee" of the curve — where Zone 1 transitions to Zone 2 — is your optimal operating point. Running below the knee wastes money. Running above it risks SLA violations.


---

## Stage 5: Size the Fleet

With measured per-GPU capacity in hand, fleet sizing becomes straightforward arithmetic — with a few important adjustments for production reality.

### The Master Formula

```
                            Peak_RPS
    GPUs_needed = ─────────────────────── × Safety_Factor
                    RPS_per_GPU @ SLA
```

This formula has three components, each critical:

**Peak_RPS** — Your maximum expected traffic. Use actual measurements if you have them, or conservative estimates based on user projections. When in doubt, round up.

**RPS_per_GPU @ SLA** — The throughput you measured in Stage 4, at your latency SLA. This must be a measured value, not a theoretical one.

**Safety_Factor** — A multiplier that accounts for real-world variance. Typical values range from 1.2 to 2.0 depending on your risk tolerance.

### Understanding the Safety Factor

The safety factor isn't padding for laziness — it accounts for specific, quantifiable risks:

**Traffic variance (1.1-1.3×):** Your "peak" is an estimate. Actual peaks often exceed estimates by 10-30%, especially during unexpected viral moments or marketing campaigns.

**GPU failures (1.02-1.05×):** At any given time, 2-5% of GPUs may be unavailable due to hardware failures, maintenance, or deployment rollouts.

**Performance variance (1.1-1.2×):** Measured throughput varies based on request mix. Your benchmark might have been optimistic.

**Deployment headroom (1.1-1.2×):** During deployments, you temporarily lose capacity as instances restart.

Combining these factors:

```
Conservative (production-critical):  1.3 × 1.05 × 1.2 × 1.2 ≈ 2.0×
Moderate (typical production):       1.2 × 1.03 × 1.1 × 1.1 ≈ 1.5×
Aggressive (cost-sensitive):         1.1 × 1.02 × 1.05 × 1.05 ≈ 1.25×
```

### Worked Example: Chatbot Service

Let's walk through a complete sizing exercise for a customer support chatbot.

**Given:**
- Peak concurrent users: 500
- Average conversation rate: 0.5 requests/second per active user
- Peak RPS: 500 × 0.5 = 250 RPS
- Model: Llama-2-7B
- Average input: 400 tokens (conversation history)
- Average output: 150 tokens
- Latency SLA: P99 < 2 seconds

**Step 1: Memory Analysis**

First, check if the model fits and estimate max concurrency:

```
Model weights (FP16):     14 GB
GPU memory (L4):          24 GB
Overhead (activations):    2 GB
Available for KV cache:    8 GB

KV cache per request:
  = 2 × 32 layers × 32 heads × 128 dim × 550 tokens × 2 bytes
  = 0.45 GB per request

Max concurrent requests: 8 GB / 0.45 GB ≈ 17 requests per GPU
```

**Step 2: Throughput Analysis**

Estimate request duration:

```
Prefill time (400 tokens @ 2000 tok/s): 200 ms
Decode time (150 tokens @ 40 tok/s):    3,750 ms
Total request time:                      ~4,000 ms
```

With 17 concurrent requests and 4-second average duration:

```
Theoretical RPS = 17 / 4.0 = 4.25 RPS per GPU
```

**Step 3: Benchmark Validation**

Run actual load test. Measured result: **6 RPS per GPU at P99 < 2s**

(Higher than theoretical because continuous batching improves decode efficiency)

**Step 4: Fleet Sizing**

```
Base GPUs = 250 RPS / 6 RPS = 41.7 → 42 GPUs
With safety factor (1.5×): 42 × 1.5 = 63 GPUs
```

**Step 5: Scaling Bounds**

```
Minimum (for average load): 
  Average RPS ≈ 100 (assuming 40% of peak)
  Min GPUs = 100 / 6 × 1.2 = 20 GPUs

Maximum (for peak + headroom):
  Max GPUs = 63 GPUs (from calculation above)
```

**Final Answer:** Deploy with auto-scaling from 20 to 65 L4 GPUs.

### Memory Capacity Planning Deep Dive

For generation workloads, memory often limits before compute. Here's how to calculate memory requirements precisely.

**KV Cache Memory Formula:**

```
KV_bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes

Total_KV_per_request = KV_bytes_per_token × max_sequence_length
```

**Reference values for common models:**

| Model | Layers | KV Heads | Head Dim | KV per Token | KV @ 4K Context |
|-------|--------|----------|----------|--------------|-----------------|
| Llama-2-7B | 32 | 32 | 128 | 512 KB | 2.0 GB |
| Llama-2-70B | 80 | 8 | 128 | 160 KB | 0.6 GB |
| Mistral-7B | 32 | 8 | 128 | 128 KB | 0.5 GB |
| Llama-3-8B | 32 | 8 | 128 | 128 KB | 0.5 GB |

Note: Models with Grouped Query Attention (GQA) like Llama-2-70B and Mistral have fewer KV heads, dramatically reducing memory requirements.

**Maximum Concurrency Calculation:**

```
Available_memory = GPU_memory × utilization_factor - model_weights - overhead
Max_concurrent = Available_memory / KV_per_request
```

**Example for Llama-2-7B on different GPUs:**

| GPU | Memory | Available for KV | Max Concurrent @ 4K |
|-----|--------|------------------|---------------------|
| L4 | 24 GB | 8 GB | 4 requests |
| A10G | 24 GB | 8 GB | 4 requests |
| A100-40GB | 40 GB | 24 GB | 12 requests |
| A100-80GB | 80 GB | 64 GB | 32 requests |
| H100-80GB | 80 GB | 64 GB | 32 requests |

**The context length trap:** Long context dramatically reduces concurrency. A model that handles 32 concurrent requests at 4K context might only handle 4 concurrent requests at 32K context. Plan accordingly.


---

## Routing: Distributing Load Across GPUs

Once you have multiple GPUs, you need to route requests intelligently. Poor routing can create hotspots that degrade performance even when aggregate capacity is sufficient.

### The Evolution of Routing Strategies

**Level 1: Round Robin**

The simplest approach — send requests to GPUs in rotation.

```
Request 1 → GPU 0
Request 2 → GPU 1
Request 3 → GPU 2
Request 4 → GPU 0 (cycle repeats)
```

*Problem:* Ignores request complexity. A 10,000-token request and a 100-token request get equal treatment, creating imbalanced load.

**Level 2: Least Connections**

Route to the GPU with the fewest active requests.

```
GPU 0: 3 active requests  ← New request goes here
GPU 1: 5 active requests
GPU 2: 4 active requests
```

*Improvement:* Better load distribution.
*Remaining problem:* Still ignores request complexity and queue depth.

**Level 3: Weighted Queue Depth**

Account for both running requests and queued requests, with higher weight on queued (since queuing hurts latency more).

```
Score = (waiting_requests × 4) + running_requests
Route to GPU with lowest score

GPU 0: 2 waiting, 5 running → Score = 13
GPU 1: 0 waiting, 8 running → Score = 8  ← Winner
GPU 2: 3 waiting, 3 running → Score = 15
```

*Why this works:* Queued requests represent latency debt. A GPU with 0 queue and 8 running requests will clear faster than one with 3 queued and 3 running.

**Level 4: Token-Aware Routing**

Estimate request cost in tokens and balance token load rather than request count.

```
GPU 0: 50,000 tokens in flight
GPU 1: 30,000 tokens in flight  ← Route 20K token request here
GPU 2: 45,000 tokens in flight
```

*Best for:* Highly variable request sizes where token count varies 10× or more.
*Limitation:* Requires knowing request size upfront (input tokens are known, output tokens must be estimated).

### Prefix Cache-Aware Routing

When using prefix caching, routing decisions should consider cache locality. If GPU 0 has already cached your system prompt, routing there saves prefill compute.

```
System prompt hash → Consistent routing to same GPU
                     ↓
Benefits:
• 50-90% prefill savings for repeated system prompts
• Lower TTFT for returning users
• Better GPU utilization (less redundant compute)
```

Implementation approaches:
- **Hash-based routing:** Hash the system prompt, route to GPU = hash % num_GPUs
- **Cache-aware routing:** Track which prefixes are cached where, prefer cache hits
- **Hybrid:** Use hash-based for initial routing, track actual cache state for optimization

### Routing Recommendation

For most LLM serving scenarios, **weighted queue depth** provides the best balance of simplicity and effectiveness. Add prefix cache awareness if you have significant prompt reuse (common system prompts, few-shot examples, etc.).

---

## Auto-Scaling: Dynamic Capacity Management

Static fleets waste money during low-traffic periods and risk overload during peaks. Auto-scaling adjusts capacity to match demand.

### Scaling Triggers

**Scale-up triggers** (any one should trigger):

| Metric | Threshold | Duration | Rationale |
|--------|-----------|----------|-----------|
| Queue depth | > 10 requests | 30 seconds | Requests are waiting |
| P99 latency | > SLA | 60 seconds | Users are experiencing delays |
| GPU utilization | > 80% | 2 minutes | Approaching saturation |
| Goodput | < 95% | 60 seconds | Requests missing SLA |

**Scale-down triggers** (all must be true):

| Metric | Threshold | Duration | Rationale |
|--------|-----------|----------|-----------|
| GPU utilization | < 30% | 10 minutes | Significant overcapacity |
| Queue depth | = 0 | 5 minutes | No waiting requests |
| Time since scale-up | > 15 minutes | - | Prevent flapping |
| Current replicas | > minimum | - | Maintain HA baseline |

### Scaling Behavior

**Scale-up:** Be aggressive. Add 2+ GPUs at a time. Users are waiting.

**Scale-down:** Be conservative. Remove 1 GPU at a time. Verify stability before removing more.

**Cooldown periods:** After scaling up, wait 3-5 minutes before scaling again (to let new capacity warm up). After scaling down, wait 10-15 minutes (to ensure stability).

### The Cold Start Problem

GPU instances take 1-5 minutes to become ready:
- Instance provisioning: 30-60 seconds
- Container startup: 30-60 seconds  
- Model loading: 1-3 minutes
- Warmup (CUDA graphs, JIT): 30-60 seconds

This delay means **reactive scaling is often too slow**. By the time new capacity is ready, the traffic spike may have passed (or users may have churned).

### Predictive Scaling

Better than reactive scaling is predictive scaling — adding capacity before you need it.

**Time-based patterns:** If traffic peaks at 9 AM every weekday, scale up at 8:45 AM.

**Event-based triggers:** If marketing is sending a newsletter at 2 PM, scale up at 1:55 PM.

**ML-based prediction:** Train a model on historical traffic patterns to predict load 15-30 minutes ahead.

```
Predictive scaling timeline:

Traffic:     ___╱╲___
            ╱      ╲
Reactive:  ___╱╲___     ← Scale-up happens AFTER spike starts
              ↑ Users wait here
              
Predictive: __╱╲____    ← Scale-up happens BEFORE spike
           ↑ Capacity ready when needed
```

### Scaling Boundaries

Always define minimum and maximum replica counts:

**Minimum replicas:** The baseline for high availability. Usually 2-3 GPUs minimum, even at zero traffic, to handle sudden load and provide redundancy.

**Maximum replicas:** The cost ceiling. Prevents runaway scaling from bankrupting you during traffic anomalies or attacks.


---

## Case Studies: Complete Sizing Examples

Let's apply the full methodology to three realistic scenarios.

### Case Study 1: Content Moderation Service

**Scenario:** A social media platform needs to moderate all posts in real-time.

**Requirements:**
- 10,000 posts/second at peak (viral events)
- 500 posts/second average
- P99 latency < 100ms
- Model: Fine-tuned RoBERTa-base (125M parameters)
- Input: Up to 280 characters (~100 tokens)

**Analysis:**

*Workload classification:* This is a **classification** workload — small model, fixed output (label), latency-sensitive. Expect to be **compute-bound** with high throughput potential.

*Memory check:* RoBERTa-base is ~250MB. Even the smallest GPU has abundant memory. Memory is not a constraint.

*Benchmark results:* On an L4 GPU with TensorRT optimization and batch size 32:
- Measured: 800 RPS at P99 = 45ms ✓

*Fleet sizing:*
```
Base GPUs = 10,000 / 800 = 12.5 → 13 GPUs
Safety factor (1.5×): 13 × 1.5 = 20 GPUs
Failure allowance (2%): 20 / 0.98 = 21 GPUs
```

*Scaling configuration:*
```
Minimum: 3 GPUs (handles average load with headroom)
Maximum: 25 GPUs (peak + buffer)
Scale-up: Queue > 50 for 15 seconds
Scale-down: Utilization < 20% for 10 minutes
```

**Final recommendation:** 3-25 L4 GPUs with auto-scaling

**Cost estimate:** $7,500 - $62,500/month (cloud pricing)

---

### Case Study 2: Customer Support Chatbot

**Scenario:** An e-commerce company deploys an AI chatbot for customer support.

**Requirements:**
- 100 concurrent conversations at peak
- 20 concurrent conversations average
- TTFT < 500ms, ITL < 50ms, E2E < 5 seconds
- Model: Llama-2-7B
- Input: ~500 tokens (conversation history + query)
- Output: ~150 tokens average response

**Analysis:**

*Workload classification:* This is a **generation** workload — variable output length, streaming responses. Expect to be **memory-capacity-bound** due to KV cache requirements.

*Memory analysis:*
```
Model weights (FP16): 14 GB
GPU: L4 (24 GB)
Available for KV cache: 24 - 14 - 2 (overhead) = 8 GB

KV cache per request (650 tokens): ~0.5 GB
Max concurrent per GPU: 8 / 0.5 = 16 requests
```

*Throughput analysis:*
```
Prefill (500 tokens): ~250ms
Decode (150 tokens @ 40 tok/s): ~3,750ms
Total request time: ~4,000ms

With continuous batching at 16 concurrent:
Theoretical RPS = 16 / 4.0 = 4 RPS
```

*Benchmark results:* Measured 6 RPS per GPU at SLA (continuous batching helps)

*Fleet sizing:*
```
Peak concurrent = 100 users
GPUs for concurrency = 100 / 16 = 6.25 → 7 GPUs
Safety factor (1.3×): 7 × 1.3 = 9.1 → 10 GPUs
```

*Scaling configuration:*
```
Minimum: 2 GPUs (HA baseline)
Maximum: 12 GPUs (peak + headroom)
Scale-up: P99 latency > 4 seconds for 30 seconds
Scale-down: Utilization < 25% for 15 minutes
```

**Final recommendation:** 2-12 L4 GPUs with auto-scaling

**Cost estimate:** $5,000 - $30,000/month

---

### Case Study 3: Code Completion Service

**Scenario:** A developer tools company offers AI-powered code completion in their IDE.

**Requirements:**
- 50,000 developers, ~10% concurrent at peak
- 5,000 concurrent completion requests
- TTFT < 200ms (typing must feel instant)
- Model: CodeLlama-13B
- Input: ~1,000 tokens (code context)
- Output: ~50 tokens (completion)

**Analysis:**

*Workload classification:* This is a **generation** workload but with short outputs. The tight TTFT requirement makes this **prefill-latency-sensitive**.

*Memory analysis:*
```
Model weights (FP16): 26 GB → Need A100-40GB minimum
GPU: A100-40GB
Available for KV cache: 40 - 26 - 3 = 11 GB

KV cache per request (1,050 tokens): ~0.8 GB
Max concurrent per GPU: 11 / 0.8 = 13 requests
```

*Latency analysis:*
```
Prefill (1,000 tokens): ~200ms ← Just at TTFT SLA!
Decode (50 tokens @ 60 tok/s): ~830ms
Total: ~1,030ms
```

*Throughput:*
```
RPS per GPU = 13 / 1.03 = 12.6 RPS theoretical
Measured: 10 RPS at TTFT SLA
```

*Fleet sizing:*
```
Peak RPS estimate: 5,000 users × 0.5 req/sec = 2,500 RPS
(Developers don't request continuously)

Base GPUs = 2,500 / 10 = 250 GPUs
Safety factor (1.3×): 250 × 1.3 = 325 GPUs
```

**This is expensive.** Consider optimizations:
- INT8 quantization: 2× memory efficiency → ~162 GPUs
- Speculative decoding: 2× decode speed → helps throughput
- Prefix caching: Common imports/boilerplate cached → helps TTFT

*With optimizations:*
```
Optimized RPS per GPU: ~18 RPS
Optimized fleet: 2,500 / 18 × 1.3 = 180 GPUs
```

**Final recommendation:** 50-200 A100-40GB GPUs with aggressive optimization

**Cost estimate:** $150,000 - $600,000/month

**Alternative consideration:** Evaluate smaller models (7B) or more aggressive quantization if cost is prohibitive.


---

## The Capacity Planning Checklist

Before launching any ML inference service, verify each item:

### Traffic Analysis
- [ ] Peak RPS documented with data source
- [ ] Average RPS documented
- [ ] Traffic pattern identified (steady / bursty / time-of-day)
- [ ] Input token distribution measured (P50, P90, P99)
- [ ] Output token distribution measured (P50, P90, P99)

### Latency Requirements  
- [ ] TTFT SLA defined with percentile (e.g., P99 < 500ms)
- [ ] ITL/TPOT SLA defined with percentile
- [ ] End-to-end latency SLA defined
- [ ] Availability target defined (e.g., 99.9%)

### Benchmarking
- [ ] Single-GPU throughput **measured** (not estimated)
- [ ] Latency-throughput curve generated
- [ ] Memory utilization at max concurrency verified
- [ ] Cold start time measured
- [ ] Multi-GPU scaling verified (if applicable)

### Fleet Sizing
- [ ] Base GPU count calculated from measurements
- [ ] Safety factor applied (1.2-2.0×)
- [ ] Failure allowance included
- [ ] Min/max scaling bounds defined
- [ ] Cost estimate reviewed and approved

### Operational Readiness
- [ ] Auto-scaling triggers configured
- [ ] Monitoring dashboards created
- [ ] Alerting thresholds set on P99 latency
- [ ] Runbook for capacity issues written
- [ ] Load test at 1.2× peak completed successfully

---

## Quick Reference: Formulas and Rules of Thumb

### Fleet Sizing Formula

```
GPUs = (Peak_RPS / RPS_per_GPU) × Safety_Factor / (1 - Failure_Rate)
```

### Little's Law (Concurrency)

```
Concurrent_Requests = RPS × Average_Latency_Seconds
```

### KV Cache Memory

```
GB_per_sequence = 2 × Layers × KV_Heads × Head_Dim × Seq_Len × 2 / 10⁹
```

### Maximum Concurrency

```
Max_Concurrent = (GPU_Memory - Model_Weights - Overhead) / KV_per_Sequence
```

### Generation Throughput

```
RPS ≈ Max_Concurrent / (Prefill_Seconds + Output_Tokens / Decode_TPS)
```

### Classification Throughput

```
RPS ≈ (Prefill_TPS / Avg_Input_Tokens) × Batch_Efficiency
```

### Cost Per Request

```
$/request = (GPU_Cost_per_Hour / 3600) / RPS_per_GPU
```

### Rules of Thumb

| Rule | Value |
|------|-------|
| Safety factor (typical) | 1.5× |
| GPU failure rate | 2-5% |
| Classification RPS per GPU | 100-1000+ |
| Generation RPS per GPU | 5-50 |
| KV cache per token (7B model) | ~0.25 MB |
| Model weights (FP16) | 2 bytes × parameters |
| Cold start time | 1-5 minutes |
| Scale-up cooldown | 3-5 minutes |
| Scale-down cooldown | 10-15 minutes |


---

## Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────┐
│                   EPISODE 4: THE CAPACITY PLANNER                    │
│                          Key Takeaways                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  THE FIVE-STAGE PROCESS                                             │
│  ───────────────────────                                            │
│  1. Know your traffic (RPS, token distributions, patterns)          │
│  2. Classify your workload (classification vs generation)           │
│  3. Identify your bottleneck (compute, memory-BW, memory-capacity)  │
│  4. MEASURE per-GPU capacity (never estimate!)                      │
│  5. Size the fleet with safety margins                              │
│                                                                      │
│  THE GOLDEN RULES                                                   │
│  ───────────────                                                    │
│  • Measure, don't estimate. Benchmarks beat calculations.           │
│  • Size for P99, optimize for P50.                                  │
│  • Memory limits before compute (for generation workloads).         │
│  • Classification and generation have opposite profiles.            │
│  • Include safety factor (1.3-2×) for production.                   │
│                                                                      │
│  CLASSIFICATION vs GENERATION                                       │
│  ────────────────────────────                                       │
│  Classification: Compute-bound, batch heavily, 100-1000 RPS/GPU     │
│  Generation: Memory-bound, maximize concurrency, 5-50 RPS/GPU       │
│                                                                      │
│  ROUTING                                                            │
│  ───────                                                            │
│  Best general strategy: Weighted queue depth                        │
│  Score = (waiting × 4) + running → route to lowest                  │
│                                                                      │
│  SCALING                                                            │
│  ───────                                                            │
│  Scale UP aggressively (users are waiting)                          │
│  Scale DOWN conservatively (verify stability first)                 │
│  Predictive > Reactive (cold start is slow)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Series Conclusion

Over four episodes, we've built a complete understanding of LLM inference:

| Episode | Title | Core Lesson |
|---------|-------|-------------|
| 1 | The Token Factory | Prefill vs decode, KV caching fundamentals |
| 2 | The Batch Whisperer | Continuous batching, PagedAttention, memory efficiency |
| 3 | The Distributed Orchestra | Multi-GPU scaling, speculative decoding, production systems |
| 4 | The Capacity Planner | GPU sizing, routing, auto-scaling |

**The meta-lesson:** LLM inference is not magic. It's physics (compute vs memory bandwidth), economics (cost per token), and systems engineering (routing, scaling, reliability). Master these fundamentals, and you can reason about any new technique or system that emerges.

The field moves fast — new models, new hardware, new optimization techniques appear constantly. But the underlying principles remain stable: understand your bottleneck, measure don't estimate, and always leave room for the unexpected.

Now go size your fleet. And remember: **measure twice, deploy once.**

---

## Appendix: Reference Implementations

The following Python implementations provide working code for the concepts discussed. Use these as starting points for your own capacity planning tools.

### Traffic Analysis

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class TrafficProfile:
    """Characterize your traffic for capacity planning."""
    
    # Volume metrics
    peak_rps: float
    average_rps: float
    daily_requests: int
    
    # Request shape
    input_tokens_p50: int
    input_tokens_p90: int
    input_tokens_p99: int
    output_tokens_p50: int
    output_tokens_p90: int
    output_tokens_p99: int
    
    # Patterns
    burst_factor: float  # peak / average
    time_of_day_variance: bool
    
    # SLAs
    ttft_target_ms: float
    tpot_target_ms: float
    e2e_latency_target_ms: float
    availability_target: float  # e.g., 0.999
    
    @property
    def avg_input_tokens(self) -> int:
        """Estimate average from percentiles."""
        return int(self.input_tokens_p50 * 0.7 + self.input_tokens_p90 * 0.25 + self.input_tokens_p99 * 0.05)
    
    @property
    def avg_output_tokens(self) -> int:
        return int(self.output_tokens_p50 * 0.7 + self.output_tokens_p90 * 0.25 + self.output_tokens_p99 * 0.05)
    
    @property
    def tokens_per_second(self) -> float:
        """Total token throughput at peak."""
        return self.peak_rps * (self.avg_input_tokens + self.avg_output_tokens)
```

### KV Cache Memory Calculator

```python
def calculate_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    dtype_bytes: int = 2,  # FP16 = 2, FP32 = 4, FP8 = 1
) -> float:
    """
    Calculate KV cache memory per sequence in GB.
    
    Args:
        num_layers: Number of transformer layers
        num_kv_heads: Number of key-value heads (may differ from query heads in GQA)
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length to support
        dtype_bytes: Bytes per value (2 for FP16, 1 for FP8)
    
    Returns:
        Memory in GB for one sequence's KV cache
    """
    # KV cache stores both keys and values
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    total_bytes = bytes_per_token * max_seq_len
    return total_bytes / (1024 ** 3)


def calculate_max_concurrency(
    gpu_memory_gb: float,
    model_weights_gb: float,
    kv_cache_per_seq_gb: float,
    memory_overhead_gb: float = 2.0,
    gpu_memory_utilization: float = 0.9,
) -> int:
    """
    Calculate maximum concurrent requests per GPU.
    
    Args:
        gpu_memory_gb: Total GPU memory
        model_weights_gb: Memory for model weights
        kv_cache_per_seq_gb: KV cache memory per sequence
        memory_overhead_gb: Additional overhead for activations, etc.
        gpu_memory_utilization: Fraction of memory to use (leave headroom)
    
    Returns:
        Maximum number of concurrent sequences
    """
    usable_memory = gpu_memory_gb * gpu_memory_utilization
    memory_for_kv = usable_memory - model_weights_gb - memory_overhead_gb
    
    if memory_for_kv <= 0:
        raise ValueError("Model doesn't fit in GPU memory!")
    
    return max(1, int(memory_for_kv / kv_cache_per_seq_gb))


# Common model configurations
MODEL_CONFIGS = {
    "llama-2-7b": {"layers": 32, "kv_heads": 32, "head_dim": 128, "weights_gb": 14},
    "llama-2-13b": {"layers": 40, "kv_heads": 40, "head_dim": 128, "weights_gb": 26},
    "llama-2-70b": {"layers": 80, "kv_heads": 8, "head_dim": 128, "weights_gb": 140},  # GQA
    "mistral-7b": {"layers": 32, "kv_heads": 8, "head_dim": 128, "weights_gb": 14},    # GQA
    "llama-3-8b": {"layers": 32, "kv_heads": 8, "head_dim": 128, "weights_gb": 16},    # GQA
    "llama-3-70b": {"layers": 80, "kv_heads": 8, "head_dim": 128, "weights_gb": 140},  # GQA
}

GPU_CONFIGS = {
    "l4": {"memory_gb": 24, "bandwidth_tb_s": 0.3},
    "a10g": {"memory_gb": 24, "bandwidth_tb_s": 0.6},
    "a100-40gb": {"memory_gb": 40, "bandwidth_tb_s": 1.5},
    "a100-80gb": {"memory_gb": 80, "bandwidth_tb_s": 2.0},
    "h100-80gb": {"memory_gb": 80, "bandwidth_tb_s": 3.35},
}
```

### Capacity Estimation

```python
def estimate_classification_capacity(
    prefill_tokens_per_second: float,
    avg_input_tokens: int,
    batch_size: int,
    target_latency_ms: float,
    overhead_ms: float = 5.0,
) -> dict:
    """
    Estimate capacity for classification workloads.
    
    Classification = prefill-dominated, fixed small output.
    """
    # Time to process one batch
    compute_time_ms = (avg_input_tokens * batch_size / prefill_tokens_per_second) * 1000
    total_batch_time_ms = compute_time_ms + overhead_ms
    
    # Check if we can meet latency with this batch size
    if total_batch_time_ms > target_latency_ms:
        # Reduce effective batch size to meet latency
        available_compute_ms = target_latency_ms - overhead_ms
        max_tokens = prefill_tokens_per_second * (available_compute_ms / 1000)
        effective_batch = max(1, int(max_tokens / avg_input_tokens))
        actual_latency_ms = target_latency_ms
    else:
        effective_batch = batch_size
        actual_latency_ms = total_batch_time_ms
    
    # Calculate throughput
    batches_per_second = 1000 / actual_latency_ms
    rps = batches_per_second * effective_batch
    
    return {
        "requests_per_second": rps,
        "effective_batch_size": effective_batch,
        "latency_ms": actual_latency_ms,
        "tokens_per_second": rps * avg_input_tokens,
    }


def estimate_generation_capacity(
    prefill_tokens_per_second: float,
    decode_tokens_per_second: float,
    avg_input_tokens: int,
    avg_output_tokens: int,
    max_concurrent_requests: int,
) -> dict:
    """
    Estimate capacity for generation workloads.
    
    Generation = decode-dominated, variable output length.
    """
    # Prefill time
    prefill_time_s = avg_input_tokens / prefill_tokens_per_second
    
    # Decode time (with batching benefit)
    # Batching provides sub-linear speedup for decode
    batch_efficiency = max_concurrent_requests ** 0.3  # Empirical factor
    effective_decode_tps = decode_tokens_per_second * batch_efficiency
    decode_time_s = avg_output_tokens / effective_decode_tps
    
    # Total request time
    total_time_s = prefill_time_s + decode_time_s
    
    # Throughput via Little's Law
    rps = max_concurrent_requests / total_time_s
    
    return {
        "requests_per_second": rps,
        "output_tokens_per_second": rps * avg_output_tokens,
        "ttft_ms": prefill_time_s * 1000,
        "avg_latency_ms": total_time_s * 1000,
        "time_per_output_token_ms": (decode_time_s / avg_output_tokens) * 1000,
    }
```

### Fleet Sizing

```python
import math

def size_gpu_fleet(
    peak_rps: float,
    measured_rps_per_gpu: float,
    safety_factor: float = 1.5,
    min_gpus_for_ha: int = 2,
    gpu_failure_rate: float = 0.02,
) -> dict:
    """
    Calculate GPU fleet size with production considerations.
    
    Args:
        peak_rps: Maximum expected requests per second
        measured_rps_per_gpu: MEASURED throughput per GPU at your latency SLA
        safety_factor: Multiplier for traffic variance (1.2-2.0)
        min_gpus_for_ha: Minimum GPUs for high availability
        gpu_failure_rate: Expected fraction of GPUs unavailable (0.02-0.05)
    
    Returns:
        Dictionary with fleet sizing details
    """
    # Base calculation
    base_gpus = peak_rps / measured_rps_per_gpu
    
    # Apply safety factor
    with_safety = base_gpus * safety_factor
    
    # Account for failures
    with_failures = with_safety / (1 - gpu_failure_rate)
    
    # Round up and ensure minimum
    final_gpus = max(math.ceil(with_failures), min_gpus_for_ha)
    
    # Calculate actual capacity
    effective_capacity = final_gpus * measured_rps_per_gpu * (1 - gpu_failure_rate)
    headroom_percent = ((effective_capacity - peak_rps) / peak_rps) * 100
    
    return {
        "gpus_needed": final_gpus,
        "base_gpus": math.ceil(base_gpus),
        "theoretical_capacity_rps": final_gpus * measured_rps_per_gpu,
        "effective_capacity_rps": effective_capacity,
        "headroom_percent": headroom_percent,
    }
```

### Load Balancer

```python
from typing import Dict, List
import random

class WeightedQueueDepthBalancer:
    """
    Load balancer using weighted queue depth strategy.
    
    Score = (waiting_requests × waiting_weight) + running_requests
    Route to GPU with lowest score.
    """
    
    def __init__(self, gpu_endpoints: List[str], waiting_weight: float = 4.0):
        self.endpoints = gpu_endpoints
        self.waiting_weight = waiting_weight
        self.stats: Dict[str, Dict[str, int]] = {
            ep: {"waiting": 0, "running": 0} for ep in gpu_endpoints
        }
    
    def get_endpoint(self) -> str:
        """Select the best endpoint for a new request."""
        scores = []
        for endpoint in self.endpoints:
            stats = self.stats[endpoint]
            score = (stats["waiting"] * self.waiting_weight) + stats["running"]
            scores.append((endpoint, score))
        
        # Find minimum score (with random tiebreaker)
        min_score = min(s[1] for s in scores)
        candidates = [ep for ep, score in scores if score == min_score]
        
        return random.choice(candidates)
    
    def request_started(self, endpoint: str):
        """Called when a request begins processing."""
        self.stats[endpoint]["waiting"] -= 1
        self.stats[endpoint]["running"] += 1
    
    def request_queued(self, endpoint: str):
        """Called when a request is queued."""
        self.stats[endpoint]["waiting"] += 1
    
    def request_completed(self, endpoint: str):
        """Called when a request finishes."""
        self.stats[endpoint]["running"] -= 1
```

### Auto-Scaling Logic

```python
from dataclasses import dataclass
from enum import Enum
import time

class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"

@dataclass
class ScalingConfig:
    min_replicas: int = 2
    max_replicas: int = 100
    
    # Scale-up triggers
    queue_depth_threshold: int = 10
    queue_depth_duration_sec: int = 30
    p99_latency_threshold_ms: float = 1000
    p99_latency_duration_sec: int = 60
    utilization_high_threshold: float = 0.80
    utilization_high_duration_sec: int = 120
    
    # Scale-down triggers
    utilization_low_threshold: float = 0.30
    utilization_low_duration_sec: int = 600
    cooldown_after_scale_up_sec: int = 900
    
    # Scaling increments
    scale_up_increment: int = 2
    scale_down_increment: int = 1

@dataclass
class CurrentMetrics:
    queue_depth: int
    queue_depth_duration_sec: int
    p99_latency_ms: float
    p99_latency_duration_sec: int
    gpu_utilization: float
    utilization_duration_sec: int
    current_replicas: int
    time_since_last_scale_up_sec: int

def determine_scaling_action(
    metrics: CurrentMetrics,
    config: ScalingConfig
) -> ScalingAction:
    """
    Determine whether to scale up, down, or maintain current capacity.
    """
    # Check scale-up conditions (any trigger)
    if (metrics.queue_depth > config.queue_depth_threshold and
        metrics.queue_depth_duration_sec > config.queue_depth_duration_sec):
        return ScalingAction.SCALE_UP
    
    if (metrics.p99_latency_ms > config.p99_latency_threshold_ms and
        metrics.p99_latency_duration_sec > config.p99_latency_duration_sec):
        return ScalingAction.SCALE_UP
    
    if (metrics.gpu_utilization > config.utilization_high_threshold and
        metrics.utilization_duration_sec > config.utilization_high_duration_sec):
        return ScalingAction.SCALE_UP
    
    # Check scale-down conditions (all must be true)
    if (metrics.gpu_utilization < config.utilization_low_threshold and
        metrics.utilization_duration_sec > config.utilization_low_duration_sec and
        metrics.time_since_last_scale_up_sec > config.cooldown_after_scale_up_sec and
        metrics.current_replicas > config.min_replicas):
        return ScalingAction.SCALE_DOWN
    
    return ScalingAction.NO_CHANGE
```

---

*This concludes the LLM Inference Deep Dive series. Go build something fast.*
