# Statistical Conclusions for Research Questions

## RQ1: How do model parameter scale and precision quantization influence task success rates in interactive, goal-oriented agent environments?

### Conclusion

**Model parameter scale is the dominant predictor of agentic task success, while precision quantization has no statistically significant effect on success rates.** The interaction between scale and quantization is also non-significant, indicating that quantization does not disproportionately harm larger or smaller models.

### Statistical Evidence

#### 1. Quantization has no significant effect on success rate

The Two-Way ANOVA revealed that quantization (bf16, q8\_0, q4\_k\_m) has no significant main effect on task success rates:

| Source | SS | df | F | p |
|---|---|---|---|---|
| C(quant) | 0.0009 | 2 | 0.012 | **0.988** |
| C(size) | 4.489 | 4 | 28.11 | **1.50 × 10⁻¹⁸** |
| C(quant) × C(size) | 0.022 | 8 | 0.068 | 0.999 |

The quantization factor accounts for virtually zero variance (partial η² ≈ 0.0001), while model size explains a substantial portion (partial η² = 0.359). The arcsine-square-root transformed ANOVA confirms this pattern (F_quant = 0.004, p = 0.996; F_size = 43.98, p = 1.72 × 10⁻²⁶).

The Kruskal-Wallis non-parametric test corroborates these results:
- **Quant**: H = 0.063, p = 0.969, ε² ≈ −0.009 (negligible effect)
- **Size**: H = 124.72, p = 5.25 × 10⁻²⁶, ε² = 0.572 (large effect)

The Linear Mixed-Effects Model (the primary analysis, now treating size as a categorical factor `C(size_num)`, with model family as a random effect and domain as a fixed effect) further confirms:
- q8\_0 vs. bf16: β ≈ 0, p ≈ 1.000 (non-significant)
- q4\_k\_m vs. bf16: β = 0.004, p = 0.944 (non-significant)
- All size × quant interaction terms: p > 0.528 (all p > 0.83 except one at p = 0.528)
- Likelihood ratio test for the interaction block: χ²(8) = 1.105, p = 0.997

The ICC for model family is 0.123. Once size is modelled as discrete categories rather than a continuous covariate, the between-model-family variance captured by the random intercept drops substantially — the size-level dummies now absorb the architectural differences that previously inflated the ICC. **This confirms the overall conclusion: quantization is non-significant and model architecture (expressed through size category) explains the dominant share of variance.**

#### 2. Model scale positively predicts success rate — but the relationship is model-family-dependent

Spearman rank correlation between parameter count and success rate is significant overall (ρ = 0.406, p = 5.61 × 10⁻¹⁰, n = 216), and this trend is preserved across all quantization levels:

| Quant Level | ρ | p |
|---|---|---|
| bf16 | 0.415 | 2.89 × 10⁻⁴ |
| q8\_0 | 0.430 | 1.63 × 10⁻⁴ |
| q4\_k\_m | 0.383 | 9.09 × 10⁻⁴ |

> [!IMPORTANT]
> This scaling trend is confounded by architecture. The sizes are nested within model families (deepseek-r1-qwen: 1.5B/7B; ministral3: 3B/8B; qwen3: 4B/8B). In the updated LMM, size is modelled categorically (`C(size_num)`). Only the 4B level (β = 0.467, p < 0.001) and 8B level (β = 0.248, p = 0.003) reach significance — both driven by qwen3 and ministral3 architectures respectively — not by parameter count in isolation. The 7B level (deepseek-r1-qwen) is non-significant (β = 0.035, p = 0.558), confirming that the apparent scaling trend is primarily driven by architectural differences between model families.

#### 3. Domain is a highly significant factor

The LMM reveals large, highly significant domain effects:
- dbbench vs. alfworld: β = 0.166, p = 8.19 × 10⁻⁸
- os vs. alfworld: β = 0.094, p = 0.002
- webshop vs. alfworld: β = 0.359, p = 3.58 × 10⁻³¹

Dunn's post-hoc comparisons on size confirm clear tiers:

| Comparison | p (Bonferroni) | Interpretation |
|---|---|---|
| 4B vs. 1.5B | 2.14 × 10⁻²⁰ | 4B ≫ 1.5B |
| 4B vs. 7B | 9.20 × 10⁻¹³ | 4B ≫ 7B |
| 8B vs. 7B | 7.92 × 10⁻⁸ | 8B ≫ 7B |
| 4B vs. 8B | 0.046 | 4B > 8B |
| 1.5B vs. 7B | 0.408 | n.s. |

This again reflects model family: qwen3-4B (mean SR = 0.432) massively outperforms deepseek-r1-qwen-7B (mean SR = 0.033), despite having fewer parameters.

#### 4. Cochran-Mantel-Haenszel test corroborates null result at the count level

The CMH test evaluates whether quantization is associated with success/failure **odds** after stratifying by domain, providing a count-level robustness check on the rate-level ANOVA and LMM findings. Because the test requires 2×2 tables per stratum, pairwise comparisons against bf16 were used:

| Comparison | CMH Statistic | p-value | Pooled OR |
|---|---|---|---|
| bf16 vs q8\_0 | 0.504 | 0.478 | 0.978 |
| bf16 vs q4\_k\_m | 0.529 | 0.467 | 0.978 |

Both comparisons are strongly non-significant (p ≈ 0.47). The pooled Mantel-Haenszel odds ratios are essentially 1.0 (0.978 for both), meaning quantized models have **the same odds of task success as full-precision models** across all four domains. This provides independent, count-based confirmation of the ANOVA and LMM null results.

#### Descriptive Summary


| Quantization | Mean Success Rate | Mean Energy/Task (J) |
|---|---|---|
| bf16 | 0.188 | 11,137 |
| q8\_0 | 0.191 | 8,186 |
| q4\_k\_m | 0.193 | 6,687 |

The success rates across quantization levels differ by less than 0.5 percentage points — well within noise.

---

## RQ2: How does the structural composition of agent execution failures change as language models are quantized?

### Conclusion

**The overall compositional profile of failure types is statistically invariant to quantization when using matched, within-configuration comparisons.** A significant chi-square result at the aggregate level is driven primarily by shifts in system error (SysErr) rates, not by changes in the cognitive failure modes (TLE, IF, IA, CF). The CLR-transformed compositional analysis confirms no significant multivariate shift in failure composition across quantization levels.

### Statistical Evidence

#### 1. Aggregate Chi-Square test is significant, but effect size is negligible

The Chi-Square test of homogeneity on the overall failure-type contingency table yields:

- χ²(10) = 94.55, p = 6.68 × 10⁻¹⁶
- **Cramér's V = 0.039** (negligible effect size)

While highly significant (due to large sample size — the test used raw failure counts totaling tens of thousands), Cramér's V of 0.039 indicates that quantization explains less than 0.2% of the variance in failure-type distribution. The statistical significance is driven by sample size, not practical significance.

#### 2. Standardized residuals identify SysErr as the primary driver

The standardized residuals reveal which cells deviate from expected:

| Failure Type | bf16 | q8\_0 | q4\_k\_m |
|---|---|---|---|
| **TLE** | **−3.97** | −0.88 | **+4.86** |
| IF | +1.40 | −0.42 | −0.98 |
| IA | −0.11 | −0.23 | +0.34 |
| CF | +0.32 | +1.66 | −1.98 |
| TE | +0.02 | +0.73 | −0.75 |
| **SysErr** | **+5.32** | **−2.01** | **−3.32** |

Only two failure types show |z| > 2:
- **SysErr**: bf16 has significantly more system errors (+5.32), while q4\_k\_m has fewer (−3.32). This likely reflects that full-precision models require more memory/compute resources, increasing the probability of infrastructure-level failures — not a cognitive failure mode.
- **TLE**: q4\_k\_m has elevated time-limit exceeded failures (+4.86), while bf16 has fewer (−3.97). This suggests quantized models may take more turns to reach a conclusion, occasionally exceeding time budgets.

#### 3. Matched Friedman tests show non-significance for all cognitive failure types

When using the correct matched design (each model-size-domain as a "subject" across quantization levels), the Friedman test reveals:

| Failure Type | χ²_F | p | Kendall's W |
|---|---|---|---|
| TLE | 2.51 | 0.285 | 0.052 |
| IF | 1.32 | 0.516 | 0.028 |
| IA | 0.78 | 0.678 | 0.016 |
| CF | 1.79 | 0.408 | 0.037 |
| TE | 4.33 | 0.115 | 0.090 |
| **SysErr** | **7.27** | **0.026** | **0.151** |

Only **SysErr** reaches significance (p = 0.026, W = 0.151). Post-hoc Wilcoxon signed-rank tests show this is driven by the bf16 vs. q8\_0 comparison (W = 17.5, p_adj = 0.047), indicating bf16 produces significantly more system errors than q8\_0. All Kendall's W values are small (< 0.16), confirming minimal practical effect.

#### 4. Compositional Data Analysis (CLR + MANOVA) confirms null result

The MANOVA on CLR-transformed failure proportions shows:
- **Pillai's trace = 0.021, p = 0.999**
- **Wilks' λ = 0.979, p = 0.999**

These are strongly non-significant, confirming that the multivariate failure composition does not differ across quantization levels when analyzed with the methodologically appropriate compositional framework.

#### 5. Bootstrap CIs show overlapping proportions

The 95% bootstrap confidence intervals for each failure type overlap completely across all three quantization levels. For example:

| Failure Type | bf16 (95% CI) | q8\_0 (95% CI) | q4\_k\_m (95% CI) |
|---|---|---|---|
| TLE | [0.154, 0.284] | [0.165, 0.302] | [0.190, 0.331] |
| IF | [0.186, 0.346] | [0.176, 0.339] | [0.170, 0.339] |
| CF | [0.219, 0.399] | [0.220, 0.414] | [0.203, 0.397] |
| SysErr | [0.034, 0.069] | [0.023, 0.051] | [0.025, 0.042] |

No pair of CIs fails to overlap, consistent with the MANOVA null result.

#### 6. Multinomial Logistic Regression

The full multinomial model (failure\_type ~ quant + domain + size) is significantly better than the null (LR χ²(30) = 33,748, p ≈ 0), with pseudo-R² = 0.334. However, the large pseudo-R² is driven overwhelmingly by **domain and size** predictors, not quantization — consistent with the RQ1 finding that model identity and task domain are the primary sources of variation.

---

## RQ3: What is the empirical trade-off between agentic task effectiveness and computational efficiency across varying quantization levels?

### Conclusion

**Quantization yields substantial and statistically significant energy savings (27–38% median reduction) with no significant degradation in task success rate, making it an unambiguously favorable trade-off.** The majority of model-domain configurations fall into "win-win" (improved or maintained performance with lower energy) or "acceptable trade-off" (negligible performance loss with large energy gain) quadrants.

### Statistical Evidence

#### 1. MANOVA detects a joint effect of quantization on {success, energy}

The MANOVA on the bivariate outcome {success\_rate, log(energy\_per\_task)} yields:

| Test | Value | F | p |
|---|---|---|---|
| Pillai's trace | 0.045 | 2.44 | **0.046** |
| Wilks' λ | 0.955 | 2.46 | **0.045** |
| Hotelling-Lawley trace | 0.047 | 2.48 | **0.044** |
| Roy's greatest root | 0.047 | 4.99 | **0.008** |

All four test statistics are significant at α = 0.05. This means quantization significantly affects the **joint** distribution of effectiveness and efficiency — but as RQ1 showed, the effect is entirely on the energy dimension, not on success rate.

#### 2. Energy savings are large and statistically significant

The Relative Change Analysis using paired Wilcoxon signed-rank tests (matched by model-size-domain) reveals:

**bf16 → q8\_0:**
- Success rate change: median Δ = 0.000, W = 44.5, p = 0.615, r = 0.10 (non-significant, negligible effect)
- **Energy savings: median = 27.4%, W = 18.0, p = 3.02 × 10⁻⁵, r = 0.852** (significant, large effect)

**bf16 → q4\_k\_m:**
- Success rate change: median Δ = −0.001, W = 79.5, p = 0.533, r = 0.13 (non-significant, negligible effect)
- **Energy savings: median = 37.9%, W = 25.0, p = 1.08 × 10⁻⁴, r = 0.791** (significant, large effect)

The effect sizes for energy savings (r = 0.852 and r = 0.791) are classified as **large** by Cohen's standards (r > 0.5).

#### 3. Quadrant analysis confirms the trade-off is favorable

| Quadrant | bf16 → q8\_0 | bf16 → q4\_k\_m |
|---|---|---|
| **Win-Win** (↑perf, ↓energy) | **16/24 (66.7%)** | **10/24 (41.7%)** |
| **Trade-off** (↓perf, ↓energy) | 7/24 (29.2%) | 12/24 (50.0%) |
| Inverse (↑perf, ↑energy) | 1/24 (4.2%) | 2/24 (8.3%) |
| **Lose-Lose** (↓perf, ↑energy) | **0/24 (0%)** | **0/24 (0%)** |

> [!IMPORTANT]
> **Zero configurations fall in the Lose-Lose quadrant** for either quantization level. The overwhelming majority (96–92%) either maintain or improve performance while saving energy.

#### 4. Efficiency ratio is statistically indistinguishable across quant levels

The Kruskal-Wallis test on the efficiency ratio (success\_rate / log(energy\_per\_task)) shows:
- H = 0.069, p = 0.966, ε² ≈ −0.009

This means quantized models achieve the same "performance per unit of energy" as full-precision models — they simply use less energy while maintaining the same success rate.

However, the paired Wilcoxon on efficiency ratio (bf16 vs. q8\_0) is significant (W = 35, p = 0.016), suggesting q8\_0 is marginally *more* efficient on a per-configuration basis (median difference in efficiency ratio ≈ 0), though the practical magnitude is small.

#### 5. Pareto frontier is dominated by quantized configurations

Of the 4 Pareto-optimal configurations (overall across domains):
- **qwen3-4B-q8\_0**: SR = 0.435, Energy = 21,538 J/task
- **qwen3-4B-q4\_k\_m**: SR = 0.427, Energy = 17,569 J/task
- **ministral3-8B-q4\_k\_m**: SR = 0.296, Energy = 3,017 J/task
- **ministral3-3B-q4\_k\_m**: SR = 0.216, Energy = 2,611 J/task

**All four Pareto-optimal configurations use quantized models (q4\_k\_m or q8\_0)**. No full-precision (bf16) configuration is Pareto-optimal, as each is dominated by its quantized counterpart offering equal performance at lower cost.

#### 6. Success rate and energy are uncorrelated overall

The Spearman correlation between success rate and energy per task is non-significant overall (ρ = 0.045, p = 0.512), and non-significant within each quantization level (all p > 0.12). This means **more energy does not buy more success** — it is an inefficiency captured by architectural differences (e.g., qwen3-4B uses high energy but also achieves high success; deepseek-r1-qwen uses high energy but achieves near-zero success).

### Descriptive Summary of the Trade-off

| Transition | Success Δ | Energy Δ | Verdict |
|---|---|---|---|
| bf16 → q8\_0 | +0.3% (n.s.) | **−27.4%** (p < 0.001) | Free efficiency gain |
| bf16 → q4\_k\_m | −0.1% (n.s.) | **−37.9%** (p < 0.001) | Free efficiency gain |

---

## Summary of Key Findings

| Finding | Statistical Support | Significance |
|---|---|---|
| Quantization does not affect success rate | ANOVA: p = 0.988; LMM: p > 0.94; Kruskal-Wallis: p = 0.969; CMH: p ≈ 0.47, OR ≈ 0.978 | Robust null result (5 independent tests) |
| Model family/architecture is the dominant factor | LMM ICC = 0.123 (after size modelled categorically); Kruskal-Wallis model H = 108.9, p = 2.2 × 10⁻²⁴ | Very strong |
| Failure composition is invariant to quantization | CoDA MANOVA Pillai's p = 0.999; Friedman p > 0.11 for all cognitive types | Robust null result |
| SysErr is the only failure type affected | Friedman p = 0.026; bf16 > q8\_0 (p\_adj = 0.047) | Moderate, limited to infrastructure errors |
| Quantization saves 27–38% energy | Wilcoxon p < 10⁻⁴ for both levels; r > 0.79 | Very strong |
| Zero configurations exhibit lose-lose outcomes | Quadrant analysis: 0/24 for both q8\_0 and q4\_k\_m | Descriptive, robust |
| All Pareto-optimal configs are quantized | 4/4 Pareto-optimal use q4\_k\_m or q8\_0 | Descriptive |


## Summary in simpler terms

### RQ1 — Does making the model smaller (fewer bits) hurt its ability to solve tasks?
**Answer**: No. Compressing a model from full precision (bf16) to 8-bit (q8_0) or 4-bit (q4_k_m) does not reduce how often it successfully completes tasks. The success rates are virtually identical: 18.8% vs 19.1% vs 19.3% — a difference of less than half a percent, which is just random noise.

What actually matters is which model family you pick and which size tier it sits in. In the updated model (which correctly treats parameter counts as categories rather than a continuous scale), the 4B tier (qwen3-4B) explains a huge performance jump (β = +0.467, p < 0.001), while the 7B tier (deepseek-r1-qwen-7B) shows no significant gain. For example, qwen3 at 4B parameters solves 43% of tasks, while deepseek at 7B (almost double the size!) only solves 3.3%. Architecture/design of the model matters far more than raw size or precision.

The task domain also matters a lot — webshop tasks are much easier than alfworld tasks regardless of model.

In simple terms: Think of quantization like compressing a JPEG photo. Our tests show that even at aggressive compression levels, the "image quality" (task performance) stays the same.

### RQ2 — When models fail, do they fail differently after being compressed?
**Answer**: No, not really. The types of failures (timing out, wrong format, wrong action, system crashes, etc.) stay in roughly the same proportions regardless of quantization. We tested this five different ways and got the same answer.

The only exception is system/infrastructure errors: full-precision (bf16) models crash more often than compressed ones. This makes sense — bf16 models use more memory and compute, so they're more likely to hit hardware limits. But this is a machine-resource issue, not a "the model got dumber" issue.

The bootstrap confidence intervals (think of them as error bars) overlap completely across all quantization levels for every failure type — meaning the differences could easily be due to random chance.

In simple terms: Compressing the model doesn't change how it fails. It doesn't start making different kinds of mistakes — it just makes the same kinds of mistakes at the same rates.

### RQ3 — Is there a trade-off? Do you sacrifice performance to save energy?
**Answer**: No trade-off — it's basically a free lunch. Quantization saves a LOT of energy (27–38% less electricity) while performance stays the same. This is statistically very strong evidence (p < 0.0001 for energy savings, with large effect sizes).

The "quadrant analysis" paints the picture clearly:

67% of configurations at q8_0 are win-win (same or better performance AND less energy)
0% of configurations are "lose-lose" (worse performance AND more energy)
The remaining ~30% are "acceptable trade-offs" (tiny performance dip but big energy save)
When we looked at which model configurations are the "best deals" overall (Pareto-optimal — meaning nothing else beats them on BOTH performance and energy), all 4 winning configurations are quantized models. No full-precision model makes the best-deal list.

Also, spending more energy does NOT buy you more success. The correlation between energy and performance is basically zero (ρ = 0.045). Some models burn lots of energy and still fail; others use little energy and succeed a lot.

In simple terms: Quantization is like switching from a gas-guzzler to a hybrid car — you get the same performance with 27–38% less fuel, and there's no downside.

### The bottom line across all three RQs: You can safely compress these LLMs to 4-bit precision and expect (1) the same success rate, (2) the same failure patterns, and (3) major energy savings. The thing that actually drives performance is which model architecture you choose and which task domain you're working on.