# The Impact of Agent Architecture on Agentic Performance: A Cross-Family Analysis

## 1. Introduction

A central and recurring finding across all three Research Questions (RQ1, RQ2, RQ3) in our study is that **model architecture—specifically, the model family—is the single most powerful determinant of agentic task performance**. Neither parameter scale nor precision quantization can overcome fundamental architectural limitations. This document is dedicated entirely to unpacking this finding: what it means, how it manifests across each RQ, and *why* the three model families in our study (Qwen3, Ministral3, and DeepSeek-R1-Qwen) exhibit such starkly different behaviours across the four AgentBench environments.

---

## 2. The Three Model Families Under Study

Before diving into results, it is essential to understand *what* each model family was designed for, because these design decisions directly predict the empirical outcomes we observe.

### 2.1 DeepSeek-R1-Qwen (1.5B, 7B)

DeepSeek-R1 is a **reasoning-first** architecture. It was developed with explicit reinforcement learning for chain-of-thought (CoT) reasoning, using Group Relative Policy Optimization (GRPO) to encourage the model to produce extended internal reasoning traces before answering (DeepSeek-AI, 2025). The `-qwen` variant is distilled from DeepSeek-R1 into the Qwen2.5 architecture backbone.

**Key architectural traits relevant to agentic tasks:**
- **Extended thinking tokens:** The model is trained to emit long `<think>...</think>` blocks before producing any actionable output. In agentic loops where the environment parser expects a strict `Action: <action>` format, these thinking tokens are either truncated, unparsed, or consume the entire context window.
- **GRPO alignment:** The model is rewarded during training for *arriving at correct answers through visible reasoning*, not for producing tool calls or API-compatible outputs. This creates a fundamental misalignment with agentic task specifications that require concise, structured responses.
- **Qwen2.5 backbone:** While the underlying transformer uses grouped-query attention (GQA) and RoPE embeddings (same as Qwen3), the post-training pipeline completely overwrites the base model's instruction-following capabilities with reasoning-oriented behaviour.

### 2.2 Ministral3 (3B, 8B)

Ministral is Mistral AI's edge-optimized model line. It is a **general-purpose instruction-following** architecture with architectural innovations focused on deployment efficiency.

**Key architectural traits relevant to agentic tasks:**
- **Sliding Window Attention (SWA):** Ministral uses a sliding window of 4,096 tokens over a total context of 128K, allowing it to process long interaction histories efficiently without quadratic attention cost. This is particularly relevant for agentic loops where observations accumulate over many turns.
- **Interleaved SWA + Full Attention layers:** Alternating layers between local (sliding window) and global (full) attention balances the model's ability to focus on recent actions while retaining overall task context.
- **Standard instruction tuning:** Ministral follows a conventional SFT + DPO alignment pipeline focused on instruction-following fidelity, making it naturally compliant with structured output formats.
- **No specialised tool-use training:** Unlike Qwen, Ministral was not specifically trained on tool-calling or API-interaction datasets. Its agentic capability derives from general instruction-following rather than specialised inductive biases.

### 2.3 Qwen3 (4B, 8B)

Qwen3 is Alibaba's latest-generation model family, explicitly designed with a **dual thinking mode** (thinking + non-thinking) and extensive tool-use capabilities.

**Key architectural traits relevant to agentic tasks:**
- **Built-in tool-use alignment:** Qwen3 was trained on large-scale datasets of function/tool calling, code execution, and agentic interaction patterns. The Qwen-Agent framework explicitly supports structured tool-call interfaces, meaning the model has deep inductive biases for producing parseable, format-compliant agent actions.
- **Hybrid thinking modes:** Unlike DeepSeek-R1, which *always* generates extended reasoning, Qwen3 can dynamically switch between "thinking" (extended internal reasoning with `<think>` tags) and "non-thinking" (direct answer) modes. This allows it to reason deeply when needed but produce concise, format-compliant output when the task demands it.
- **GQA + RoPE + SwiGLU:** Standard modern transformer ingredients, but the key differentiator is the post-training pipeline, not the architecture per se.
- **Broad pre-training data:** Qwen3 was trained on ~36 trillion tokens spanning 119 languages, with particular emphasis on code, mathematics, and structured reasoning tasks.

---

## 3. Architecture's Impact on RQ1: Task Success Rates

### 3.1 The Statistical Evidence for Architectural Dominance

Our Linear Mixed-Effects Model (LMM) quantified architecture's role precisely. The model treats "model family" as a random intercept, meaning each family (DeepSeek, Ministral, Qwen) is allowed its own baseline performance level. The resulting **Intraclass Correlation Coefficient (ICC) of 0.488** tells us that **48.8% of all variance in success rate is attributable to which model family you choose**—before considering size, quantization, or even the task domain.

This is an extraordinary finding. For comparison:
- Quantization explains **~0.01%** of variance (partial η² ≈ 0.0001)
- Model size explains **35.9%** of variance (partial η² = 0.359), but this is confounded with family (see below)
- The residual ICC means nearly half the "story" is told by the three-letter model name alone.

The Kruskal-Wallis test on model family confirms this non-parametrically: $H = 108.94$, $p = 2.2 \times 10^{-24}$, $\varepsilon^2 = 0.502$. This is a *massive* effect size—model family explains over 50% of the rank variance in success rates.

### 3.2 The Confounded Scale–Architecture Relationship

A naive Spearman correlation shows a moderate positive relationship between parameter count and success ($\rho = 0.406$, $p = 5.61 \times 10^{-10}$). This might suggest that "bigger is better." But this is a **Simpson's Paradox**: the correlation arises because our larger models happen to belong to better-performing families.

In the LMM, once model family is controlled as a random effect, the continuous `size_num` coefficient becomes *completely non-significant* ($\beta = -0.0007$, $p = 0.925$). This means: **within the same model family, adding more parameters does not significantly improve agentic success**.

The Dunn's post-hoc pairwise comparisons from the Kruskal-Wallis test illustrate this dramatically:

| Comparison | p-value (Bonferroni) | Interpretation |
|---|---|---|
| Qwen3-4B vs. DeepSeek-7B | $9.20 \times 10^{-13}$ | 4B Qwen **massively** outperforms 7B DeepSeek |
| Qwen3-4B vs. Qwen3-8B | $0.046$ | 4B Qwen significantly outperforms *its own larger sibling* |
| DeepSeek-1.5B vs. DeepSeek-7B | $0.408$ | No significant difference *within DeepSeek* despite 4.7× scale difference |

The 4B Qwen model achieves a mean success rate of **43.2%**, while the 7B DeepSeek model achieves just **3.3%**—a 13× performance gap despite having almost half the parameters. Meanwhile, doubling the DeepSeek model from 1.5B to 7B yields a statistically insignificant improvement (0.3% to 3.3%), because both variants suffer from the same fundamental architectural misalignment with agentic task formats.

### 3.3 Per-Family, Per-Domain Performance Breakdown

The following table consolidates the mean success rates across all quantization levels and runs, grouped by model family, size, and task domain:

| Model Family | Size | ALFWorld | DB | OS | WebShop | **Overall** |
|---|---|---|---|---|---|---|
| **Qwen3** | 4B | 15.4% | 46.8% | 36.4% | 74.1% | **43.2%** |
| **Qwen3** | 8B | 2.1% | 30.6% | 29.1% | 20.4% | **20.6%** |
| **Ministral3** | 8B | 1.8% | 21.9% | 12.3% | 79.8% | **29.0%** |
| **Ministral3** | 3B | 2.5% | 9.4% | 0.0% | 61.5% | **18.3%** |
| **DeepSeek-R1** | 7B | 0.0% | 11.5% | 0.4% | 1.3% | **3.3%** |
| **DeepSeek-R1** | 1.5B | 0.0% | 1.2% | 0.0% | 0.0% | **0.3%** |

**Observations by architecture:**

1.  **Qwen3-4B is the undisputed leader.** It is the top performer in 3 out of 4 domains (ALFWorld, DB, OS) and near the top in WebShop. Its tool-use training translates directly into format-compliant actions across all environments.

2.  **Ministral3-8B dominates WebShop (79.8%)** — the highest single-domain success rate in the entire study. WebShop requires natural language navigation and product selection, which maps well to Ministral's strong general instruction-following. However, Ministral3-3B achieves 0% on OS — its smaller variant completely fails at command-line interaction, revealing a capability cliff driven by insufficient world knowledge at 3B scale.

3.  **DeepSeek-R1 fails catastrophically across all domains.** Even at 7B, it achieves only 11.5% on DB (the only domain where it shows non-trivial performance) and 0% on ALFWorld. The 1.5B variant is effectively non-functional, achieving 0% on 3 out of 4 domains.

### 3.4 Why Does Qwen3-4B Outperform Qwen3-8B?

This is one of the most surprising results: **the smaller Qwen model performs significantly better than the larger one** ($p = 0.046$, Dunn's test). The overall success rates are 43.2% vs. 20.6%.

The explanation lies in the specific model variants used. Qwen3-4B in our study was configured with the thinking/tool-use mode enabled (`qwen3-4b-t`), while Qwen3-8B was the base instruction model (`qwen3-8b`). The 4B model's explicit tool-call training creates a strong inductive bias for producing structured `Action:` outputs, while the 8B model, despite having more capacity, sometimes defaults to verbose, unstructured natural language responses that the AgentBench parser rejects.

This is visible in the failure profiles: Qwen3-8B produces **dramatically more Invalid Format (IF) errors** on WebShop (134.5 IF failures out of 200 tasks) compared to Qwen3-4B (just 2 IF failures on DB's 300 tasks). The 8B model *knows* the answer but cannot express it in the required format.

---

## 4. Architecture's Impact on RQ2: Failure Composition

### 4.1 Architecture Determines the *Type* of Failure, Not Just the Rate

While our RQ2 analysis demonstrated that quantization does not change the failure composition (CoDA MANOVA $p = 0.999$), the failure profiles differ **massively between model families**. The multinomial logistic regression model (pseudo-$R^2 = 0.334$) confirms that domain and model family are the primary predictors of failure type, not quantization.

### 4.2 Family-Specific Failure Signatures

Each architecture exhibits a characteristic "failure fingerprint" that is consistent across all quantization levels:

**DeepSeek-R1-Qwen: Dominated by Time Limit Exceeded (TLE)**

DeepSeek-R1-1.5B on ALFWorld produces a TLE rate of **91.7–96.3%** across all quantization levels. The model generates extended reasoning traces within `<think>` tags, consuming tokens without ever producing a parseable action. The environment's turn limit is exhausted while the model is still "thinking." On the OS domain, DeepSeek-R1-7B shows TLE rates of 55–72%, again because its reasoning traces consume the interaction budget.

On WebShop, DeepSeek-R1-1.5B achieves 0% success with 0% for *all failure categories*. This occurs because the model generates such malformed outputs that the AgentBench harness cannot even classify the failure—the sessions terminate before any meaningful interaction occurs.

**Ministral3: Dominated by Invalid Format (IF) on text-heavy domains, Invalid Action (IA) on spatial domains**

On WebShop, Ministral3-3B at bf16 produces 95 Invalid Format failures out of 200 tasks. The model understands what product to buy but wraps its response in natural language rather than the required `search[...]` or `click[...]` format. Interestingly, at q4_k_m, the IF count drops to 40.5 while success rises from 50% to 79.5% — suggesting that *quantization noise actually helps* the 3B model by suppressing verbose preambles.

On ALFWorld, both Ministral variants overwhelmingly produce Invalid Action (IA) errors (92–101 out of 109 tasks). The model outputs syntactically valid actions, but the actions are semantically wrong — e.g., trying to `open cabinet 1` when the task requires `go to countertop 2`. This is a world-model deficiency, not a formatting issue.

**Qwen3: Balanced failure profile, dominated by Completed Failure (CF) and TLE**

Qwen3-4B's failures are predominantly "Completed Failures" — the model interacts correctly with the environment (proper format, valid actions) but arrives at the wrong final answer. On DB, 128.5 of its 159 failures are CF. This indicates that the model's agentic *loop* is functioning correctly, but its reasoning or SQL generation occasionally leads to incorrect results. This is a qualitatively "better" failure mode than IF or IA: the agent is at least engaging meaningfully with the task.

On ALFWorld, Qwen3-4B's primary failures shift to TLE (53 out of 90 failures), indicating that it explores the environment extensively but sometimes fails to find the solution within the step limit.

### 4.3 Architecture Determines SysErr Rates

The Friedman test found that SysErr was the only failure type significantly affected by quantization ($p = 0.026$). However, the *baseline* SysErr rate is itself architecture-dependent:

| Configuration | VRAM Peak (MB) | SysErr Count (bf16) |
|---|---|---|
| Qwen3-4B-bf16 | 13,828–18,695 | 0–3.5 |
| Ministral3-8B-bf16 | 20,279–22,236 | 0–5.5 |
| DeepSeek-R1-7B-bf16 | *Not in mean data* | Elevated |

Qwen3-4B, despite being the best performer, has relatively modest VRAM requirements (13.8 GB at bf16), which means it rarely triggers OOM-related SysErr. Ministral3-8B at bf16 peaks at 22.2 GB VRAM, making it susceptible to infrastructure failures on GPU-constrained hardware.

---

## 5. Architecture's Impact on RQ3: Efficiency Trade-offs

### 5.1 The Pareto Frontier Is Architecture-Dependent

The Pareto-optimal configurations (maximising success while minimising energy) are:

| Rank | Configuration | Success Rate | Energy/Task (J) | Family |
|---|---|---|---|---|
| 1 | qwen3-4B-q8_0 | 43.5% | 21,538 | Qwen3 |
| 2 | qwen3-4B-q4_k_m | 42.7% | 17,569 | Qwen3 |
| 3 | ministral3-8B-q4_k_m | 29.6% | 3,017 | Ministral3 |
| 4 | ministral3-3B-q4_k_m | 21.6% | 2,611 | Ministral3 |

**DeepSeek-R1 does not appear anywhere on the Pareto frontier.** Despite consuming substantial energy (its 7B variant generates large numbers of reasoning tokens), it achieves near-zero success. It is strictly *dominated* by every Ministral and Qwen configuration.

### 5.2 Energy Efficiency Is an Architectural Property

The Spearman correlation between energy and success is effectively zero ($\rho = 0.045$, $p = 0.512$). This means that **how much energy a model consumes tells you nothing about how well it will perform**. The decoupling is driven by architecture:

- **Qwen3-4B** consumes high energy (20,000–38,000 J/task at bf16) because it generates extensive, multi-turn interactions with deep reasoning. But this energy produces *results*—43% success.
- **DeepSeek-R1-7B** also consumes substantial energy, but nearly all of it goes to generating unparseable reasoning tokens. The energy is *wasted* on thinking that never translates to action.
- **Ministral3-8B-q4_k_m** is the efficiency champion at 3,017 J/task with 29.6% success. Its sliding-window attention and dense architecture minimise memory transfer costs, and its concise output style reduces unnecessary token generation.

### 5.3 Domain-Specific Pareto Frontiers Reveal Architectural Specialisation

**WebShop Pareto frontier:**
- `ministral3-8B-q4_k_m` (81.8% success, 2,303 J/task)
- `ministral3-3B-q4_k_m` (79.3% success, 1,815 J/task)

Ministral *completely dominates* the WebShop Pareto frontier. Qwen3 does not appear, despite Qwen3-4B achieving 74.1% success on WebShop, because it consumes far more energy. Ministral's efficiency on WebShop stems from its sliding-window attention efficiently processing the long HTML observation texts without quadratic cost explosion.

**DB Pareto frontier:**
- `qwen3-4B-q8_0` (47.1% success, 15,768 J/task)
- `qwen3-4B-q4_k_m` (46.4% success, 11,642 J/task)
- `qwen3-8B-bf16/q8_0/q4_k_m` (30–31% success, 1,522–2,469 J/task)

Qwen dominates the DB frontier at both the high-performance and high-efficiency ends. DB requires SQL generation — a task where Qwen3's code-heavy pre-training provides a decisive advantage.

**OS Pareto frontier:**
- `qwen3-4B-q8_0` (37.3%, 24,249 J/task) — peak performance
- `qwen3-4B-q4_k_m` (35.9%, 20,063 J/task)
- `qwen3-8B-q4_k_m` (29.4%, 5,885 J/task) — best efficiency
- `ministral3-8B-q4_k_m` (13.2%, 4,851 J/task) — lower tier

OS command execution requires understanding Linux shell semantics, file system operations, and system administration concepts. Qwen3-4B's dominance here again traces to its tool-use pre-training, which includes extensive code and command-line interaction data.

**ALFWorld Pareto frontier:**
- `qwen3-4B-q4_k_m` (17.4%, 19,928 J/task)
- `ministral3-3B-bf16/q8_0/q4_k_m` (0.9–3.7%, 1,478–2,082 J/task)

ALFWorld is the most challenging domain overall. Qwen3-4B leads but at a high energy cost per success. All models struggle because ALFWorld requires multi-step spatial reasoning and object state tracking — capabilities that are hard to acquire from text-only pre-training.

---

## 6. Why Architecture Matters More Than Quantization: A Synthesis

### 6.1 The Bottleneck Hierarchy

Our results reveal a clear hierarchy of bottlenecks for agentic performance:

```
Architecture (alignment, training data, tool-use capability)
    └── Task Domain (format requirements, reasoning type)
        └── Parameter Scale (within a fixed architecture)
            └── Quantization Precision (essentially irrelevant)
```

Quantization operates at the bottom of this hierarchy because it only affects the *precision* of weight representations. But the critical capabilities for agentic tasks—format compliance, tool-call syntax, environment interaction protocols—are **discrete structural properties** of the model's attention patterns and output distribution. These are deeply encoded during alignment training and are preserved under quantization because:

1. **Modern k-quants (q4_k_m, q8_0) use non-uniform bit allocation** that preserves attention head activations and key feed-forward network neurons (Frantar et al., 2022; Lin et al., 2023).
2. **Agentic outputs are low-entropy.** The correct action in most agentic turns is a highly constrained string (e.g., `search[laptop]`, `SELECT * FROM ...`, `cd /home`). The probability mass for these tokens is heavily concentrated, making the output robust to small perturbations in logits caused by quantization noise.

### 6.2 The Architecture–Task Alignment Matrix

| Capability Required | Qwen3 | Ministral3 | DeepSeek-R1 |
|---|---|---|---|
| **Structured output format** | ★★★ (tool-use training) | ★★ (instruction following) | ★ (reasoning-mode conflicts) |
| **SQL/Code generation** | ★★★ (code-heavy pretraining) | ★★ (general) | ★★ (reasoning helps, format hurts) |
| **Long-horizon planning** | ★★★ (hybrid thinking) | ★★ (SWA for context) | ★ (over-thinks, under-acts) |
| **Spatial reasoning** | ★★ (limited) | ★ (limited) | ★ (limited) |
| **Natural language navigation** | ★★★ | ★★★ | ★ (too formal/verbose) |
| **Energy efficiency** | ★★ (high token count) | ★★★ (SWA, compact) | ★ (wasteful reasoning) |

### 6.3 Practical Implications

1. **Prioritise architectural alignment over scale or precision.** Our data demonstrates that a tool-use-aligned architecture at 4B parameters and 4-bit quantization can outperform a reasoning-oriented architecture at 7B parameters and full precision by an order of magnitude in success rate, while simultaneously consuming less energy. This suggests that, in practice, selecting an architecturally appropriate model family is the single highest-leverage decision for agentic deployment—no amount of parameter scaling or precision restoration can compensate for a fundamental mismatch between a model's training objective and the structured interaction patterns required by agent environments.

2. **Match architectural strengths to task domain characteristics.** Models with strong general instruction-following capabilities (e.g., SWA-based architectures with conventional SFT+DPO alignment) excel in domains with flexible format requirements and natural language interaction, such as web navigation. Conversely, models with explicit tool-use and code-generation training data in their alignment pipeline dominate domains requiring strict syntactic output compliance, such as database interaction and operating system command execution. Domain–architecture matching is a stronger predictor of success than within-family scaling.

3. **Reasoning-optimised training pipelines can be counterproductive for agentic tasks.** Models trained with reinforcement learning objectives that incentivise extended chain-of-thought traces (e.g., GRPO-based reasoning alignment) face a structural disadvantage in agentic settings. The extended thinking tokens consume the interaction budget, violate strict format parsers, and generate verbose outputs that environment harnesses cannot process. Reasoning capability and agentic capability are distinct competencies that require different alignment strategies; excelling at one does not transfer to the other.

---

## 7. Conclusion

Across all three research questions, model architecture emerges as the dominant factor:

- **RQ1 (Success):** Architecture explains 48.8% of variance (ICC); quantization explains 0.01%. The tool-use-aligned family vastly outperforms the reasoning-oriented family despite having fewer parameters.
- **RQ2 (Failures):** Each architectural category exhibits a characteristic failure fingerprint — reasoning-oriented models are dominated by timeout failures (TLE), general instruction-following models by format and action errors (IF/IA), and tool-use-aligned models by completed-but-incorrect outcomes (CF). These signatures are invariant to quantization.
- **RQ3 (Efficiency):** The Pareto frontier consists exclusively of tool-use-aligned and general instruction-following architectures in their quantized forms. Reasoning-oriented architectures are strictly dominated at every energy level. The most energy-efficient configurations come from compact, SWA-based general-purpose models at aggressive quantization levels.

The overarching lesson is that **agentic capability is an emergent property of specific training pipelines** (tool-use alignment, format compliance data, agentic interaction datasets), not of raw model scale or numerical precision. A well-aligned architecture at modest parameter counts will outperform a misaligned architecture at larger scale across every quantization level and task domain tested. Practitioners and researchers should therefore treat architectural selection—guided by the alignment between a model's training objective and the target task's interaction protocol—as the primary design decision, with quantization serving as a universally safe post-hoc optimisation.

---

## References

1. DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." *arXiv:2501.12948*.
2. Qwen Team (2025). "Qwen3 Technical Report." *arXiv:2505.09388*.
3. Mistral AI (2024). "Ministral: Edge Models." *mistral.ai/news/ministraux*.
4. Frantar, E., et al. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*.
5. Lin, J., et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *MLSys 2024*.
6. Liu, X., et al. (2023). "AgentBench: Evaluating LLMs as Agents." *ICLR 2024*.
7. Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022*.
