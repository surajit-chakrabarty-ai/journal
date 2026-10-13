---
date: 2026-10-13 15:57
---
# The Architecture of Minds: A Deep Dive into Multi-Agent AI Systems

## A Story of Three Paradigms

*It's 2026, and the question is no longer "Can AI think?" but "How should AI systems think together?"*

Imagine you're an architect tasked with designing not a building, but a mind. Not just any mind—a collective intelligence that can research complex topics, solve mathematical proofs, and navigate the messy, unpredictable terrain of real-world problems. Where do you begin?

This is the challenge that has consumed AI researchers over the past two years. And remarkably, three distinct schools of thought have emerged, each offering a radically different blueprint for building intelligent systems.

The first school looks backward—to the cognitive architectures of the 1980s, to Soar and ACT-R, to the symbolic AI that once promised artificial general intelligence. They ask: *What if we could give language models the same modular structure that cognitive scientists believe underlies human thought?*

The second school looks sideways—to neural networks themselves. They ask: *What if we treated each AI agent not as an autonomous entity, but as a neuron in a larger computational graph? What if we could backpropagate through conversations?*

The third school looks forward—to the messy reality of production systems. They ask: *How do we actually build multi-agent systems that work reliably at scale? What does it take to move from prototype to product?*

This blog synthesizes insights from three landmark works that represent these paradigms:

1. **CoALA (Cognitive Architectures for Language Agents)** — A theoretical framework drawing from cognitive science
2. **AgenticNet** — A neural network-inspired architecture with "textual backpropagation"
3. **Anthropic's Multi-Agent Research System** — Battle-tested lessons from production deployment

Together, they paint a picture of where agentic AI is heading—and the hard problems that remain unsolved.

---

## Key Takeaways

> **For the time-pressed reader:**

1. **Three architectural paradigms dominate:** Cognitive (modular memory + structured actions), Neural (layered agents + textual gradients), and Orchestrator-Worker (lead agent + parallel subagents).

2. **Memory is the bottleneck.** All three approaches grapple with how agents store, retrieve, and update information across long-horizon tasks.

3. **Multi-agent systems excel at parallelizable, breadth-first problems.** Anthropic reports 90.2% improvement over single-agent baselines on research tasks—but at 15× the token cost.

4. **"Textual backpropagation" is real.** AgenticNet demonstrates that natural language feedback can systematically improve agent prompts, achieving state-of-the-art results on coding and math benchmarks.

5. **Evaluation remains the hardest problem.** Traditional metrics fail for agents. New approaches combine code-based graders, LLM judges, and human review—but no silver bullet exists.

6. **The gap between prototype and production is vast.** Minor bugs cascade into catastrophic failures. Agents are stateful, non-deterministic, and require entirely new debugging paradigms.

---

## Part I: The Cognitive Architecture Paradigm (CoALA)

### The Core Insight: Agents as Cognitive Systems

The CoALA framework begins with a provocative observation: **language models are probabilistic production systems**.

In the 1970s, cognitive scientists like Allen Newell proposed that human cognition could be modeled as a set of "production rules"—if-then statements that fire when their conditions are met. A thermostat, for instance, might have rules like:

```
(temperature > 72°) ∧ (furnace on) → turn off furnace
(temperature < 70°) ∧ (furnace off) → turn on furnace
```

LLMs, the CoALA authors argue, do something remarkably similar. Given an input string X, they define a probability distribution over possible continuations Y. This is essentially a probabilistic production: `X ∼→ X Y`.

But raw LLMs are stateless—they don't remember anything between calls. To build agents, we need to wrap them in a **cognitive architecture** that provides:

### The Four Pillars of CoALA

#### 1. Memory Modules

CoALA defines four types of memory, each serving a distinct function:

| Memory Type | Function | Example |
|-------------|----------|---------|
| **Working Memory** | Active, readily available information for the current decision cycle | Current goal, recent observations, intermediate reasoning |
| **Episodic Memory** | Sequences of past experiences | Previous conversation turns, game trajectories, failed attempts |
| **Semantic Memory** | Facts about the world | Retrieved documents, learned rules, world knowledge |
| **Procedural Memory** | How to do things | LLM weights (implicit) + agent code (explicit) |

The key insight is that **working memory is the central hub**. It's where information from all other sources converges before being fed to the LLM, and where LLM outputs are parsed before being routed elsewhere.

#### 2. Structured Action Space

CoALA divides actions into **internal** and **external**:

```
                    ┌─────────────────────────────────────────┐
                    │              ACTION SPACE               │
                    ├─────────────────┬───────────────────────┤
                    │    INTERNAL     │       EXTERNAL        │
                    ├─────────────────┼───────────────────────┤
                    │ • Reasoning     │ • Physical (robots)   │
                    │ • Retrieval     │ • Digital (APIs/web)  │
                    │ • Learning      │ • Dialogue (humans)   │
                    └─────────────────┴───────────────────────┘
```

**Reasoning** reads from and writes to working memory—it's how agents "think." **Retrieval** pulls information from long-term memory into working memory. **Learning** writes information from working memory into long-term storage.

This structure explains why some agents feel more "intelligent" than others: they have richer internal action spaces that allow for more sophisticated self-modification.

#### 3. The Decision Cycle

Every CoALA agent follows a repeated loop:

```
┌──────────────────────────────────────────────────────────────┐
│                      DECISION CYCLE                          │
│                                                              │
│   ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌──────┐  │
│   │ Observe │ → │  Plan    │ → │  Execute  │ → │ Learn │  │
│   └─────────┘    └──────────┘    └───────────┘    └──────┘  │
│        ↑                                              │      │
│        └──────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

During **planning**, the agent uses reasoning and retrieval to propose and evaluate candidate actions. The best action is **selected** and **executed**. If the action produces an observation, the cycle begins again.

#### 4. Grounding

Grounding is how agents connect to the external world. CoALA identifies three grounding environments:

- **Physical**: Robots processing visual/audio input and executing motor commands
- **Digital**: APIs, websites, code execution, games
- **Dialogue**: Conversations with humans or other agents

The elegance of CoALA is that it treats all external interaction as a "text game"—observations come in as text, actions go out as text. This simplification enables the same architecture to work across radically different domains.

### What CoALA Gets Right

1. **Theoretical grounding**: By connecting to cognitive science, CoALA provides a principled vocabulary for discussing agent design.

2. **Modularity**: The separation of memory types and action types makes it easier to reason about what an agent can and cannot do.

3. **Generality**: The framework accommodates everything from simple chatbots to complex multi-agent simulations.

### What CoALA Leaves Open

1. **How do you actually implement adaptive retrieval?** The paper notes this is "understudied."

2. **How do agents learn new procedures safely?** Writing to procedural memory "can easily introduce bugs or allow an agent to subvert its designers' intentions."

3. **How do multiple agents coordinate?** CoALA focuses on single-agent architectures.

---

## Part II: The Neural Network Paradigm (AgenticNet)

### The Core Insight: Agents as Neurons

While CoALA looks to cognitive science, AgenticNet looks to deep learning itself. The central question: **What if we organized LLM agents the same way we organize neurons in a neural network?**

The result is a "multilayer multi-agent architecture that mirrors the computational structure of a multilayer perceptron (MLP) while operating in text space."

### The Architecture

AgenticNet organizes agents into **layers** with specialized roles:

```
                    ┌─────────────────────────────────────┐
                    │           SYNTHESIZER               │
                    │   (Verify consistency, final output)│
                    └─────────────────────────────────────┘
                                      ↑
                    ┌─────────────────────────────────────┐
                    │            EXECUTORS                │
                    │   (Advance reasoning step by step)  │
                    └─────────────────────────────────────┘
                                      ↑
                    ┌─────────────────────────────────────┐
                    │            PLANNERS                 │
                    │      (Decompose problems)           │
                    └─────────────────────────────────────┘
                                      ↑
                    ┌─────────────────────────────────────┐
                    │              INPUT                  │
                    └─────────────────────────────────────┘
```

**Lower layers** act as planners that decompose tasks. **Intermediate layers** function as executors that advance reasoning. **Upper layers** serve as synthesizers that verify consistency and produce final outputs.

### Forward Propagation

Information flows through the network like activations through an MLP:

1. Each agent in layer `l` receives outputs from layer `l-1`
2. The agent **ranks** upstream messages by preference
3. It selects the **top-k** most relevant messages
4. It generates its output conditioned on this context

This selective attention mechanism is crucial—it prevents context explosion as the network deepens.

### Textual Backpropagation: The Key Innovation

Here's where AgenticNet gets genuinely novel. When the final output is wrong, the system doesn't just fail—it **learns**.

The process works like this:

1. A **Judge** compares the predicted output `ŷ` to ground truth `y*`
2. It produces a **loss signal** `L` (e.g., reasoning validity, arithmetic correctness)
3. A **Layer Prompt Updater** inspects the forward pass and generates **textual gradients**—natural language feedback specifying how to correct the reasoning
4. A **Rewrite** operator applies these gradients to update agent prompts

The mathematical formulation is elegant:

```
∇̂_{p_i^{(l)}} L = "textual gradient estimate"

p_i^{(l)} ← p_i^{(l)} - ∇̂_{p_i^{(l)}} L
```

The subtraction symbol here denotes "applying the suggested modification"—it's gradient descent in prompt space.

### A Concrete Example

Consider a math problem: *"The vertex of a parabola is (2,3) and it contains point (4,4). Find y when x=6."*

**Forward pass**: The network correctly derives `y = (1/4)(x-2)² + 3` and `a = 1/4`. But in the final calculation, an agent incorrectly evaluates `4² = 4` instead of `16`, producing `ŷ = 4` instead of `y* = 7`.

**Backward pass**: The Layer Prompt Updater identifies the error source and generates feedback:

> "The expression (6-2)² simplifies to 4², which equals 16. Your calculation incorrectly evaluated this as 4. The correct calculation is y = (1/4)(16) + 3 = 7."

**Prompt update**: The agent's prompt is modified to include:

> "CRITICAL: Verify all exponentiation (e.g., 4² = 16) before simplifying."

### Experimental Results

AgenticNet achieves impressive results across five benchmarks:

| Benchmark | AgenticNet | Best Baseline | Improvement |
|-----------|------------|---------------|-------------|
| GSM8K | 91.8% | 92.3% (MaAS) | Competitive |
| MATH500 | 49.0% | 55.4% (MermaidFlow) | -6.4% |
| HumanEval | **98.8%** | 92.9% (MermaidFlow) | +5.9% |
| MBPP | **98.7%** | 82.3% (MermaidFlow) | +16.4% |
| **Average** | **85.6%** | 82.8% | +2.8% |

The standout results are in code generation (HumanEval, MBPP), where AgenticNet dramatically outperforms all baselines.

### Key Findings from Ablations

1. **Training matters**: Removing textual backpropagation drops AIME performance from 50% to 43.3%.

2. **Depth has diminishing returns**: For GPT-4o-mini, performance peaks at L=4 layers, then drops at L=5. Stronger models (DeepSeek-V3) continue improving with depth.

3. **Width helps, but non-monotonically**: Performance dips at W=2 before improving at W=4, suggesting a "transition phase" where agent interaction introduces noise before achieving synergy.

4. **Convergence is fast**: Most gains happen in the first 2-3 epochs of textual backpropagation.

### What AgenticNet Gets Right

1. **Principled architecture**: The neural network analogy provides clear design principles.

2. **Systematic adaptation**: Textual backpropagation offers a structured way to improve agents from feedback.

3. **Strong empirical results**: State-of-the-art on code generation benchmarks.

### What AgenticNet Leaves Open

1. **Scalability**: The paper tests up to 5 layers and 4 agents per layer. How does this scale to hundreds of agents?

2. **Dynamic tasks**: All experiments use static benchmarks. How does textual backpropagation work for open-ended, real-world tasks?

3. **Coordination overhead**: The fixed layered structure may not suit all problems.

---

## Part III: The Production Paradigm (Anthropic's Multi-Agent Research System)

### The Core Insight: Orchestration at Scale

While CoALA theorizes and AgenticNet experiments, Anthropic's engineering team has been building multi-agent systems that serve real users. Their Research feature—which allows Claude to search across the web, Google Workspace, and integrations—taught them hard lessons about what actually works.

### The Architecture: Orchestrator-Worker Pattern

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      LEAD RESEARCHER                             │
│  • Analyzes query                                                │
│  • Develops strategy                                             │
│  • Saves plan to Memory (context > 200k tokens → truncation)     │
│  • Spawns subagents                                              │
│  • Synthesizes results                                           │
└──────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Subagent │   │ Subagent │   │ Subagent │
        │    1     │   │    2     │   │    N     │
        └──────────┘   └──────────┘   └──────────┘
              │               │               │
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Search  │   │  Search  │   │  Search  │
        │  Tools   │   │  Tools   │   │  Tools   │
        └──────────┘   └──────────┘   └──────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      CITATION AGENT                              │
│  • Processes documents and research report                       │
│  • Identifies specific locations for citations                   │
│  • Ensures all claims are properly attributed                    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FINAL RESULTS                               │
└──────────────────────────────────────────────────────────────────┘
```

### Why Multi-Agent?

The Anthropic team provides a compelling analysis of when multi-agent systems make sense:

**The Token Hypothesis**: In their analysis, **token usage alone explains 80% of performance variance** on the BrowseComp evaluation. The number of tool calls and model choice explain most of the remaining 15%.

Multi-agent systems are essentially a way to **scale token usage** for tasks that exceed single-agent limits. Each subagent gets its own context window, enabling parallel exploration of different aspects of a problem.

**The Compression Insight**: "The essence of search is compression: distilling insights from a vast corpus." Subagents act as intelligent filters, exploring different aspects simultaneously before condensing the most important tokens for the lead agent.

### The Economics

Multi-agent systems are expensive:

| Interaction Type | Token Usage (relative to chat) |
|------------------|-------------------------------|
| Single chat | 1× |
| Single agent | ~4× |
| Multi-agent | ~15× |

This means multi-agent systems only make economic sense for **high-value tasks** where the benefit justifies the cost.

### Eight Principles for Prompting Multi-Agent Systems

Anthropic distills their learnings into eight principles:

#### 1. Think Like Your Agents

> "To iterate on prompts, you must understand their effects."

The team built simulations using their Console with exact production prompts and tools, then watched agents work step-by-step. This revealed failure modes: agents continuing when they had sufficient results, using overly verbose queries, selecting incorrect tools.

#### 2. Teach the Orchestrator How to Delegate

Early versions allowed the lead agent to give simple instructions like "research the semiconductor shortage." This was too vague—subagents misinterpreted tasks or duplicated work.

**Solution**: Each subagent needs:
- An objective
- An output format
- Guidance on tools and sources
- Clear task boundaries

#### 3. Scale Effort to Query Complexity

Agents struggle to judge appropriate effort. The team embedded explicit scaling rules:

| Query Type | Subagents | Tool Calls per Agent |
|------------|-----------|---------------------|
| Simple fact-finding | 1 | 3-10 |
| Direct comparisons | 2-4 | 10-15 each |
| Complex research | 10+ | Clearly divided responsibilities |

#### 4. Tool Design and Selection Are Critical

> "Agent-tool interfaces are as critical as human-computer interfaces."

Bad tool descriptions send agents down completely wrong paths. Each tool needs:
- A distinct purpose
- A clear description
- Explicit heuristics for when to use it

The team even created a **tool-testing agent** that attempts to use flawed MCP tools, then rewrites their descriptions to avoid failures. This resulted in a **40% decrease in task completion time**.

#### 5. Let Agents Improve Themselves

Claude 4 models can be excellent prompt engineers. When given a prompt and a failure mode, they diagnose why the agent is failing and suggest improvements.

#### 6. Start Wide, Then Narrow Down

Agents often default to overly specific queries that return few results. The team prompts agents to:
1. Start with short, broad queries
2. Evaluate what's available
3. Progressively narrow focus

#### 7. Guide the Thinking Process

Extended thinking mode serves as a controllable scratchpad. The lead agent uses thinking to:
- Plan its approach
- Assess which tools fit the task
- Determine query complexity and subagent count
- Define each subagent's role

#### 8. Parallel Tool Calling Transforms Speed

Early agents executed sequential searches—painfully slow. Two parallelization strategies cut research time by up to 90%:
1. Lead agent spins up 3-5 subagents in parallel
2. Each subagent uses 3+ tools in parallel

### Production Reliability: The Hard Lessons

#### Agents Are Stateful and Errors Compound

> "Without effective mitigations, minor system failures can be catastrophic for agents."

Unlike traditional software where you can restart from the beginning, agent restarts are expensive and frustrating. The team built systems that:
- Resume from where errors occurred
- Let agents know when tools are failing so they can adapt
- Combine AI adaptability with deterministic safeguards (retry logic, checkpoints)

#### Debugging Requires New Approaches

Agents are non-deterministic between runs. Users would report agents "not finding obvious information," but the team couldn't see why.

**Solution**: Full production tracing that monitors:
- Agent decision patterns
- Interaction structures
- Tool usage patterns

All without monitoring conversation contents (for privacy).

#### Deployment Needs Careful Coordination

Agent systems are "highly stateful webs of prompts, tools, and execution logic that run almost continuously." Updates can break running agents.

**Solution**: Rainbow deployments that gradually shift traffic from old to new versions while keeping both running simultaneously.

#### Synchronous Execution Creates Bottlenecks

Currently, lead agents execute subagents synchronously—waiting for each set to complete before proceeding. This simplifies coordination but creates bottlenecks:
- Lead agent can't steer subagents mid-execution
- Subagents can't coordinate with each other
- Entire system blocks on slowest subagent

Asynchronous execution would enable more parallelism but adds challenges in result coordination, state consistency, and error propagation.

### Evaluation: The Unsolved Problem

Anthropic's evaluation approach combines multiple methods:

#### LLM-as-Judge

A single LLM call with a rubric evaluates:
- Factual accuracy (do claims match sources?)
- Citation accuracy (do cited sources match claims?)
- Completeness (are all requested aspects covered?)
- Source quality (primary vs. secondary sources?)
- Tool efficiency (right tools, reasonable number of times?)

#### Human Evaluation

> "People testing agents find edge cases that evals miss."

Human testers noticed early agents consistently chose SEO-optimized content farms over authoritative sources like academic PDFs. Adding source quality heuristics to prompts helped resolve this.

#### The Key Insight on Evaluation

> "Start evaluating immediately with small samples."

In early development, changes have dramatic impacts—a prompt tweak might boost success from 30% to 80%. With effect sizes this large, you can spot changes with just 20 test cases.

---

## Part IV: Synthesis—The Emerging Consensus

Despite their different starting points, these three approaches converge on several key insights:

### 1. Memory Architecture Is Fundamental

All three frameworks emphasize structured memory:

| Framework | Memory Approach |
|-----------|-----------------|
| CoALA | Working + Episodic + Semantic + Procedural |
| AgenticNet | Layer-wise context selection + prompt storage |
| Anthropic | Lead agent Memory tool + subagent context windows |

The consensus: **agents need both short-term working memory and long-term storage**, with explicit mechanisms for moving information between them.

### 2. Specialization Enables Scale

All three use specialized roles:

| Framework | Specialization |
|-----------|----------------|
| CoALA | Reasoning vs. Retrieval vs. Learning vs. Grounding |
| AgenticNet | Planners vs. Executors vs. Synthesizers |
| Anthropic | Lead Researcher vs. Subagents vs. Citation Agent |

The consensus: **general-purpose agents don't scale**. Decomposing tasks into specialized subtasks enables parallelism and reduces coordination overhead.

### 3. Feedback Loops Are Essential

All three incorporate mechanisms for agents to improve:

| Framework | Feedback Mechanism |
|-----------|-------------------|
| CoALA | Learning actions that write to long-term memory |
| AgenticNet | Textual backpropagation from task loss |
| Anthropic | Prompt iteration based on failure analysis |

The consensus: **static agents plateau quickly**. The ability to learn from experience—whether through memory updates, prompt refinement, or parameter tuning—is what separates capable agents from toys.

### 4. Evaluation Remains the Bottleneck

All three acknowledge evaluation challenges:

| Framework | Evaluation Approach |
|-----------|---------------------|
| CoALA | Identifies evaluation as an open problem |
| AgenticNet | Uses benchmark accuracy + ablation studies |
| Anthropic | Combines LLM judges + human review + production monitoring |

The consensus: **we don't yet have good ways to evaluate agents**. Traditional metrics fail because agents can find valid solutions that evaluators didn't anticipate.

---

## Part V: Open Questions and Future Directions

### 1. How Do We Evaluate Open-Ended Agent Behavior?

AgenticNet's benchmarks have clear right answers. Anthropic's research tasks often don't. How do we evaluate an agent that's asked to "find interesting connections in this dataset"?

**Promising directions**:
- Rubric-based LLM judges with human calibration
- End-state evaluation (did the agent achieve the goal?) rather than trajectory evaluation
- Capability evals that start at low pass rates and "graduate" to regression suites

### 2. How Do Agents Coordinate in Real-Time?

Current multi-agent systems are largely synchronous—agents complete their tasks before results are aggregated. But many real-world problems require dynamic coordination:
- Agent A discovers information that changes Agent B's task
- Multiple agents need to avoid duplicating work
- The lead agent needs to steer subagents based on intermediate results

**Promising directions**:
- Asynchronous execution with message passing
- Shared blackboard architectures
- Hierarchical goal decomposition with dynamic replanning

### 3. How Do We Make Agents Safe?

CoALA notes that writing to procedural memory "can easily introduce bugs or allow an agent to subvert its designers' intentions." AgenticNet's textual backpropagation could theoretically be exploited to inject malicious prompt modifications.

**Promising directions**:
- Sandboxed execution environments
- Formal verification of agent behaviors
- Constitutional AI principles embedded in agent architectures

### 4. How Do We Handle Long-Horizon Tasks?

Anthropic notes that "production agents often engage in conversations spanning hundreds of turns." Context windows fill up, information gets lost, and agents lose track of their goals.

**Promising directions**:
- Intelligent compression and summarization
- External memory systems with retrieval
- Hierarchical task decomposition with checkpointing

### 5. What's the Right Level of Abstraction?

CoALA operates at the level of cognitive modules. AgenticNet operates at the level of neural network layers. Anthropic operates at the level of engineering systems.

Which abstraction is most useful depends on your goal:
- **Theoretical understanding**: CoALA
- **Systematic optimization**: AgenticNet
- **Production deployment**: Anthropic's patterns

The field may need **multiple levels of abstraction** that can be translated between each other.

---

## Conclusion: The Architecture of Tomorrow's Minds

We stand at an inflection point in AI development. The question is no longer whether language models can be useful—it's how to organize them into systems that can tackle problems beyond any single model's capabilities.

The three paradigms we've explored offer complementary insights:

**CoALA** gives us a vocabulary rooted in cognitive science—a way to think about what agents need (memory, actions, decision cycles) without getting lost in implementation details.

**AgenticNet** gives us a systematic approach to optimization—a way to improve agents through structured feedback rather than ad-hoc prompt engineering.

**Anthropic's production system** gives us hard-won lessons about what actually works—the gap between prototype and production, the importance of observability, the economics of token usage.

The agents of tomorrow will likely draw from all three traditions. They'll have modular memory systems inspired by cognitive architectures. They'll use layered processing with feedback mechanisms inspired by neural networks. And they'll be built with the engineering rigor that production systems demand.

The architecture of minds is still being written. But the blueprints are becoming clearer.

---

## References

1. **Cognitive Architectures for Language Agents (CoALA)**. Sumers, T.R., et al. *Transactions on Machine Learning Research*, 2024. [OpenReview](https://openreview.net/forum?id=1i6ZCvflQJ)

2. **AgenticNet: Rethinking Multi-agent System Architectures with LLM-based Networks**. Anonymous authors. *Under review*, 2025. [OpenReview](https://openreview.net/forum?id=axZSkXYUX1)

3. **How we built our multi-agent research system**. Hadfield, J., Zhang, B., Lien, K., et al. *Anthropic Engineering Blog*, June 2025. [Link](https://www.anthropic.com/engineering/multi-agent-research-system)

4. **Demystifying evals for AI agents**. Grace, M., Hadfield, J., Olivares, R., De Jonghe, J. *Anthropic Engineering Blog*, January 2026. [Link](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

---

## Appendix: Evaluation Framework for Agents

Based on Anthropic's "Demystifying Evals" guide, here's a practical framework for evaluating your own agents:

### Grader Types

| Type | Methods | Best For |
|------|---------|----------|
| **Code-based** | String match, unit tests, static analysis, tool call verification | Deterministic outcomes, code generation |
| **Model-based** | Rubric scoring, natural language assertions, pairwise comparison | Open-ended tasks, subjective quality |
| **Human** | Expert review, crowdsourced judgment, A/B testing | Calibrating model graders, catching edge cases |

### Key Metrics

**pass@k**: Probability of at least one success in k attempts. Use when one success matters.

**pass^k**: Probability of all k attempts succeeding. Use when consistency is essential.

### Evaluation Roadmap

1. **Start with 20-50 tasks** from real failures and common use cases
2. **Write unambiguous tasks** with reference solutions
3. **Build balanced problem sets** (test both positive and negative cases)
4. **Design graders thoughtfully** (grade outcomes, not paths)
5. **Check transcripts regularly** (failures should seem fair)
6. **Monitor for saturation** (100% pass rate = no signal for improvement)

### The Swiss Cheese Model

No single evaluation layer catches every issue. Combine:
- Automated evals (pre-launch, CI/CD)
- Production monitoring (post-launch)
- A/B testing (significant changes)
- User feedback (ongoing)
- Transcript review (weekly sampling)
- Human studies (calibrating LLM graders)

---

*This blog post synthesizes research from multiple sources to provide a comprehensive overview of multi-agent AI architectures. The field is evolving rapidly—check the original sources for the latest developments.*
