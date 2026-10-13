---
date: 2026-10-13 15:26
---
# The Geometry of Failure: Why Your Embedding Model Will Never Be Good Enough

**TL;DR:** A new ICLR 2026 paper proves mathematically that single-vector embeddings have a fundamental ceiling—no matter how much data you throw at them or how big your model gets, there exist simple retrieval tasks they *cannot* solve. The culprit? Geometry.

---

## The Uncomfortable Truth About Embeddings

Here's a question that should keep ML engineers up at night: **What if the architecture you're betting your RAG pipeline on is fundamentally broken?**

Not "needs more training data" broken. Not "needs a bigger model" broken. *Mathematically impossible to fix* broken.

That's the bombshell dropped by Weller et al. from Google DeepMind and Johns Hopkins in their ICLR 2026 paper. They prove that single-vector embeddings—the backbone of virtually every modern retrieval system—have an inescapable theoretical ceiling. And worse? They show you can hit that ceiling with queries a five-year-old could understand.

---

## The Setup: What Are We Even Asking Embeddings To Do?

Let's be precise. When you use an embedding model for retrieval, you're asking it to:

1. Map every document to a single vector in \(\mathbb{R}^d\)
2. Map every query to a single vector in \(\mathbb{R}^d\)  
3. Return the top-\(k\) documents whose vectors are closest to the query vector

Simple enough. But here's the catch: **the number of possible top-\(k\) sets grows combinatorially**.

For a corpus of \(n\) documents, the number of distinct top-\(k\) sets is:

\[
\binom{n}{k} = \frac{n!}{k!(n-k)!}
\]

For \(n = 325,000\) documents and \(k = 20\) (the QUEST dataset's setup), that's:

\[
\binom{325000}{20} \approx 7.1 \times 10^{91}
\]

That's more than the estimated number of atoms in the observable universe (\(10^{82}\)).

The question becomes: **Can a \(d\)-dimensional embedding space represent all these possible retrieval outcomes?**

Spoiler: No. Not even close.

---

## The Math That Breaks Everything

### Theorem 1: The Dimension Lower Bound

The paper's core theoretical contribution is elegant and devastating. Here's the setup:

Let \(v_1, \ldots, v_n \in \mathbb{R}^d\) be unit document vectors, and let queries be unit vectors \(u \in \mathbb{R}^d\). A \(k\)-subset \(S \subseteq [n]\) is **realized with margin \(\omega\)** if there exists a unit query \(u_S\) such that:

\[
\min_{i \in S} \langle u_S, v_i \rangle \geq \max_{j \notin S} \langle u_S, v_j \rangle + 2\omega
\]

In plain English: the worst relevant document must score at least \(2\omega\) higher than the best irrelevant document.

**Theorem 1:** If every \(k\)-subset can be realized with margin \(\omega\), then:

\[
\binom{n}{k} \leq \left(1 + \frac{1}{\omega}\right)^d
\]

Rearranging for the required dimension:

\[
d \geq \frac{\log \binom{n}{k}}{\log(1 + 1/\omega)}
\]

### The Proof: Sphere Packing in High Dimensions

The proof is a beautiful application of sphere packing. Here's the intuition:

1. **Distinct subsets require distinct queries.** If \(S \neq T\) are two different \(k\)-subsets, pick \(i \in S \setminus T\) and \(j \in T \setminus S\). The margin constraints force:
   
   \[
   \langle u_S, v_i - v_j \rangle \geq 2\omega \quad \text{and} \quad \langle u_T, v_j - v_i \rangle \geq 2\omega
   \]
   
   Adding these: \(\langle u_S - u_T, v_i - v_j \rangle \geq 4\omega\)

2. **Queries must be well-separated.** By Cauchy-Schwarz and the fact that \(\|v_i - v_j\| \leq 2\):
   
   \[
   \|u_S - u_T\| \geq 2\omega
   \]

3. **Packing argument.** The \(M = \binom{n}{k}\) unit query vectors are pairwise \(2\omega\)-separated. So the open balls \(B(u_S, \omega)\) are disjoint and all contained in \(B(0, 1+\omega)\).

4. **Volume comparison.** Using \(\text{vol}(B_d(r)) = C_d \cdot r^d\):
   
   \[
   M \cdot \omega^d \leq (1+\omega)^d \implies M \leq \left(\frac{1+\omega}{\omega}\right)^d
   \]

QED. The geometry of high-dimensional spheres imposes a hard limit on representational capacity.

### What Do These Numbers Actually Look Like?

Using a realistic margin of \(\omega = 0.1\) (score gap of 0.2), here are the minimum embedding dimensions required:

| Corpus Size \(n\) | \(k=2\) | \(k=10\) | \(k=100\) | \(k=1000\) |
|-------------------|---------|----------|-----------|------------|
| \(10^3\)          | 6       | 23       | 135       | trivial    |
| \(10^4\)          | 8       | 33       | 233       | 1,354      |
| \(10^5\)          | 10      | 42       | 329       | 2,334      |
| \(10^6\)          | 12      | 52       | 425       | 3,296      |
| \(10^9\)          | 17      | 81       | 713       | 6,177      |
| \(10^{11}\)       | 21      | 100      | 905       | 8,098      |

**Reality check:** Most production embeddings use 768-4096 dimensions. For web-scale search with billions of documents and \(k=100\), you'd need dimensions in the thousands just to *theoretically* represent all combinations—before accounting for any practical constraints.

---

## "But That's Just Theory!" — The Empirical Hammer

The authors anticipated the "theory vs. practice" objection and designed experiments to crush it.

### The Free Embedding Experiment

Here's the most damning setup: **What if we gave embeddings every possible advantage?**

- **No natural language constraints:** Vectors are directly optimizable parameters
- **No generalization required:** Optimize directly on the test set
- **Perfect optimization:** Full-batch gradient descent with InfoNCE loss
- **No tokenization:** Each query/document is its own learnable vector

This is the *best possible case* for embeddings. If free embeddings can't solve a task, no real model ever will.

**Results:** The relationship between embedding dimension \(d\) and the maximum solvable corpus size \(n\) (for \(k=2\)) follows:

\[
n_{\text{critical}} = -10.53 + 4.03d + 0.052d^2 + 0.0037d^3 \quad (R^2 = 0.999)
\]

Extrapolating:
- **512 dimensions:** ~500K documents max
- **768 dimensions:** ~1.7M documents max  
- **1024 dimensions:** ~4M documents max
- **4096 dimensions:** ~250M documents max

And remember: this is the *theoretical ceiling* with perfect optimization. Real models perform far worse.

### The Multiplier Problem

The theoretical lower bound from Theorem 1 says \(d \geq 4\) for \(n=100, k=2\). But free embeddings actually need \(d > 18\) to solve this case—a **4.5× multiplier** even without any real-world constraints.

Scale this up to web search, and the required dimensions become absurd.

---

## LIMIT: A Dataset That Shouldn't Be Hard (But Is)

To make this concrete, the authors created **LIMIT** (Limitations of Instruction-following Models in Information reTreival)—a dataset designed to be trivially simple yet theoretically hard.

### Construction

The setup is almost insultingly simple:

- **Documents:** "Jon Durben likes Quokkas and Apples. Ovid Rahm likes Quokkas and Rabbits..."
- **Queries:** "Who likes Quokkas?"
- **Task:** Return the 2 people who like the queried item

That's it. No reasoning. No complex operators. Just pattern matching that a child could do.

The trick is in the **qrel matrix design**: they use 46 documents with all \(\binom{46}{2} = 1035\) possible pairs as queries, ensuring maximum combinatorial coverage.

### Results: State-of-the-Art Models Fail Catastrophically

| Model | Embedding Dim | Recall@10 (LIMIT-small) |
|-------|---------------|-------------------------|
| Qwen3 Embeddings | 4096 | ~55% |
| Gemini Embeddings | 3072 | ~50% |
| GritLM | 4096 | ~45% |
| Promptriever | 4096 | ~60% |
| E5-Mistral | 4096 | ~40% |

**On a 46-document corpus with trivial queries, the best models can't even hit 60% Recall@10.**

Even more damning: models can't solve the task at Recall@20 (returning 20 documents from a 46-document corpus). The theoretical limit isn't some far-off concern—it's happening *right now* on toy problems.

### Is It Just Domain Shift?

No. The authors fine-tuned models on a training set with the same structure. Performance barely budged (from ~0% to ~2.8% Recall@10). But when they trained directly on the test set? Models could overfit and solve it.

This confirms: **the failure is architectural, not data-related.**

---

## What Actually Works?

### BM25: The Cockroach of IR

BM25 nearly aces LIMIT. Why? Because sparse models have **much higher effective dimensionality**—one dimension per vocabulary term. With vocabularies in the tens of thousands, they can represent far more combinations.

But BM25 has its own Achilles heel: **lexical matching**. When the authors created LIMIT-synonym (replacing "glasses" with "spectacles", etc.), BM25's performance dropped 89%. Neural models only dropped ~40%.

### Multi-Vector Models (ColBERT)

GTE-ModernColBERT significantly outperforms single-vector models on LIMIT, though it still doesn't solve the task. The MaxSim operator over multiple vectors per document provides more expressiveness.

### Cross-Encoders: The Nuclear Option

Gemini 2.5 Pro, used as a reranker with all 46 documents in context, achieves **100% accuracy** on LIMIT-small in a single forward pass.

Cross-encoders don't have the same geometric constraints because they compute relevance through full attention over query-document pairs, not through vector similarity.

---

## The Implications Are Uncomfortable

### For RAG Systems

If your RAG pipeline uses single-vector retrieval as the first stage, you have a fundamental ceiling on the complexity of queries you can handle. As users issue more sophisticated queries (especially with logical operators like "X AND Y" or "X OR Y"), you'll hit combinations that are geometrically impossible to retrieve correctly.

### For Instruction-Following Retrieval

The paper specifically calls out the trend toward "any query, any relevance definition" benchmarks. These implicitly assume embeddings can represent arbitrary relevance relationships. They can't.

### For Benchmark Design

Most retrieval benchmarks test a tiny fraction of possible query-document combinations. QUEST's 3,357 queries cover an infinitesimal slice of its \(7.1 \times 10^{91}\) possible top-20 sets. Benchmarks that don't stress-test combinatorial coverage are hiding these limitations.

### For Architecture Research

The single-vector paradigm has hit a wall. The paper suggests several directions:
- **Multi-vector models** (ColBERT-style)
- **Hybrid sparse-dense systems**
- **More expressive similarity functions**
- **Hyperencoders** and other emerging architectures

---

## The Experimental Playbook

For researchers wanting to build on this work, here's the methodology:

### 1. Free Embedding Optimization
```
Setup:
- Random unit vectors for n documents
- All C(n,k) combinations as queries
- Adam optimizer with InfoNCE loss
- Full-batch training with all negatives
- Projected gradient descent (unit norm)
- Early stopping at 1000 iterations without improvement

Metric: Find critical-n where 100% accuracy becomes impossible
```

### 2. LIMIT Dataset Construction
```
1. Generate 1850 unique "likeable" attributes via LLM
2. Create 50K synthetic people with random names
3. Choose qrel matrix: 46 docs with all C(46,2)=1035 pairs
4. Assign attributes to create query-document relevance
5. Pad documents to equal length with random attributes
```

### 3. Synonym Robustness Test
```
1. Generate synonyms for all attributes (no overlap)
2. Replace corpus attributes with synonyms
3. Keep queries unchanged
4. Measure performance drop across model types
```

---

## Final Thoughts: The Geometry Always Wins

There's something almost poetic about this result. We've spent years scaling embedding models—more parameters, more data, more compute—only to discover that the fundamental architecture has a geometric ceiling that no amount of scaling can breach.

The sphere packing argument is ancient mathematics meeting modern ML, and the sphere packing argument wins.

This doesn't mean embeddings are useless. For many practical applications, the combinatorial space of queries users actually issue is far smaller than the theoretical maximum. But as we push toward more sophisticated retrieval—instruction-following, reasoning, complex logical queries—we're going to keep running into this wall.

The future of retrieval isn't bigger embeddings. It's different architectures entirely.

---

**Paper:** [On the Theoretical Limitations of Embedding-Based Retrieval](https://arxiv.org/abs/2508.21038)  
**Authors:** Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee  
**Venue:** ICLR 2026  
**Code & Data:** [github.com/google-deepmind/limit](https://github.com/google-deepmind/limit)
