# Statistical Testing Plan for Agent-Bench Quantization Study

## Data Overview

Before selecting tests, here's a summary of the experimental design as observed from the data:

| Factor | Levels |
|---|---|
| **Model family** | `deepseek-r1-qwen`, `ministral3`, `qwen3` |
| **Parameter size** | 1.5B, 3B, 4B, 7B, 8B (nested within model) |
| **Quantization** | `bf16` (baseline), `q8_0`, `q4_k_m` |
| **Domain (task)** | `alfworld`, `dbbench`, `os`, `webshop` |
| **Runs (replicates)** | 3 per configuration |

**Key dependent variables available:**

| Variable | Type | Notes |
|---|---|---|
| `success_rate` | Continuous [0,1] | Proportion of tasks solved |
| `failure_rate` | Continuous [0,1] | Proportion of tasks failed |
| `error_rate` | Continuous [0,1] | Runtime/system errors |
| `energy_per_task` (J) | Continuous | Computational cost proxy |
| `tle_rate` | Continuous [0,1] | Time-limit exceeded failures |
| `if_rate` | Continuous [0,1] | Incorrect-format failures |
| `ia_rate` | Continuous [0,1] | Incorrect-action failures |
| `completed_failure_rate` | Continuous [0,1] | Completed but wrong |
| `avg_turns` | Continuous | Interaction complexity |
| `gpu_energy_joules` | Continuous | GPU energy consumption |
| `cpu_energy_joules` | Continuous | CPU energy consumption |

> [!IMPORTANT]
> With only **3 replicates** per cell, normality assumptions are hard to verify. Non-parametric alternatives should be run alongside parametric tests as robustness checks throughout.

---

## RQ1: How do model parameter scale and precision quantization influence task success rates in interactive, goal-oriented agent environments?

**Core question:** Do `size` and `quant` (and their interaction) significantly affect `success_rate`?

---

### Test 1 · Two-Way Factorial ANOVA (Size × Quantization)

| Aspect | Detail |
|---|---|
| **Design** | 2-factor between-subjects (Size: 6 levels; Quant: 3 levels) on `success_rate` |
| **Why** | Tests main effects of model scale and quantization independently, plus their **interaction** (does quantization hurt larger models less?) |
| **DV** | `success_rate` (mean across runs per cell) |
| **Assumptions** | Normality (Shapiro-Wilk per cell), homogeneity of variance (Levene's test). Use arcsine-square-root transform on proportions if needed. |
| **Post-hoc** | Tukey HSD for pairwise comparisons across size levels and quant levels |
| **Effect size** | Partial η² (eta-squared) for each factor and the interaction |

> [!NOTE]
> Since sizes are nested within model families (e.g., 1.5B & 7B are only in deepseek-r1-qwen), a **nested ANOVA** or **linear mixed-effects model** (see Test 3) is more appropriate if you want to generalize across model families rather than treat each size as independent.

---

### Test 2 · Kruskal-Wallis H Test (Non-parametric alternative)

| Aspect | Detail |
|---|---|
| **Design** | One-factor test on `success_rate`, run separately for each factor |
| **Why** | With n=3 per cell and bounded proportional data, normality is unlikely to hold. This is the recommended robustness check. |
| **Post-hoc** | Dunn's test with Bonferroni correction for pairwise group comparisons |
| **Effect size** | Epsilon-squared (ε²) |
| **Usage** | Run once grouping by `quant` (3 groups), once grouping by `size` (6 groups) |

---

### Test 3 · Linear Mixed-Effects Model (LMM) — **Recommended Primary Analysis**

| Aspect | Detail |
|---|---|
| **Model** | `success_rate ~ size * quant + (1 | model) + (1 | domain)` |
| **Why** | Properly handles the **nested/crossed** structure: sizes are nested within model families, and domains are a crossed random effect. Accounts for repeated runs as replicates. This is the gold-standard approach for this design. |
| **Implementation** | R: `lme4::lmer()`, Python: `statsmodels.MixedLM` |
| **Inference** | Satterthwaite or Kenward-Roger degrees of freedom for F-tests (via `lmerTest` in R) |
| **Effect size** | Marginal R² (fixed effects only) and Conditional R² (fixed + random) |
| **Post-hoc** | Estimated marginal means (EMMs) with Tukey adjustment via `emmeans` package |

> [!TIP]
> The LMM is the strongest test here because it properly models the hierarchical structure (runs within configurations, sizes within model families, tasks across domains) and does not require balanced cells.

---

### Test 4 · Spearman Rank Correlation (Size → Success Rate)

| Aspect | Detail |
|---|---|
| **Design** | Correlation between parameter count (1.5B, 3B, 4B, 7B, 8B as ordinal/numeric) and `success_rate` |
| **Why** | Tests for a **monotonic** trend — does performance increase with scale? Spearman is preferred because the relationship may not be linear and the data is non-normal. |
| **Computed** | Overall, and stratified by `quant` level to see if the scaling trend holds under quantization |

---

### Test 5 · Cochran-Mantel-Haenszel (CMH) Test

| Aspect | Detail |
|---|---|
| **Design** | Tests association between quantization level and success/failure outcome, stratified by domain |
| **Why** | Uses the raw **count data** (successes vs. failures per configuration) rather than aggregated rates. Controls for domain as a stratification variable. More powerful than working with proportions when you have the underlying counts. |
| **Data** | Construct 2×3 contingency tables (outcome × quant) for each domain stratum |
| **Assumption** | Sufficient expected cell frequencies (≥5); pool across model sizes if cells are sparse |

---

## RQ2: How does the structural composition of agent execution failures change as language models are quantized?

**Core question:** Does the *profile* of failure types (TLE, IF, IA, completed_failure, error) shift across quantization levels?

---

### Test 6 · Chi-Square Test of Homogeneity

| Aspect | Detail |
|---|---|
| **Design** | Tests whether the **distribution** of failure types is the same across quantization levels |
| **Data** | Construct a contingency table: rows = quant levels (bf16, q8_0, q4_k_m), columns = failure categories (tle, if, ia, completed_failure, error). Cells contain absolute counts aggregated across runs. |
| **Why** | Directly tests the RQ — whether the *composition* of failures changes with quantization |
| **Post-hoc** | Standardized residuals to identify which failure types drive significant differences |
| **Effect size** | Cramér's V |

> [!WARNING]
> Check expected cell frequencies. If any expected frequency < 5, use **Fisher's exact test** (computationally expensive for larger tables) or collapse sparse categories.

---

### Test 7 · Multinomial Logistic Regression

| Aspect | Detail |
|---|---|
| **Model** | `failure_type ~ quant + size + domain` |
| **Why** | Models the **probability of each failure type** as a function of quantization, while controlling for confounders (model size, domain). This is more powerful than Chi-Square because it handles multiple predictors simultaneously. |
| **DV** | Categorical: {TLE, IF, IA, completed_failure, error} |
| **Implementation** | R: `nnet::multinom()`, Python: `sklearn.linear_model.LogisticRegression(multi_class='multinomial')` |
| **Inference** | Likelihood ratio tests for each predictor; odds ratios with 95% CIs for interpretation |

---

### Test 8 · Friedman Test (Non-parametric repeated-measures alternative)

| Aspect | Detail |
|---|---|
| **Design** | Treats each model-size-domain combination as a "subject" and compares failure-type proportions across quantization levels |
| **Why** | The same model configuration is tested at all 3 quant levels — this is a **within-subject** (matched/paired) design. Friedman accounts for this pairing. |
| **Usage** | Run separately for each failure metric: `tle_rate`, `if_rate`, `ia_rate`, `completed_failure_rate`, `error_rate` |
| **Post-hoc** | Nemenyi test or Wilcoxon signed-rank with Bonferroni correction |

---

### Test 9 · Compositional Data Analysis (CoDA) via Log-Ratio Transformation

| Aspect | Detail |
|---|---|
| **Design** | Treats the failure profile as **compositional data** (parts of a whole that sum to ~1) |
| **Why** | Standard tests on proportions violate independence because if one failure type increases, others must decrease. CoDA handles this constraint properly — this is methodologically rigorous and reviewers will appreciate it. |
| **Method** | Apply **Isometric Log-Ratio (ILR)** or **Centered Log-Ratio (CLR)** transformation to the failure proportions, then run MANOVA or pairwise Hotelling's T² on the transformed coordinates |
| **Implementation** | R: `compositions` package; Python: `scikit-bio` or manual transformation |
| **Interpretation** | Biplot of ILR coordinates to visualize how failure composition shifts across quant levels |

> [!TIP]
> CoDA is increasingly expected in serious research involving proportional/compositional outcomes. Including it demonstrates methodological sophistication and addresses reviewer concerns about spurious correlations in proportional data.

---

### Test 10 · Stacked Proportion Comparison with Bootstrap CIs

| Aspect | Detail |
|---|---|
| **Design** | For each quantization level, compute the proportion of total failures attributed to each type. Compare using bootstrap confidence intervals. |
| **Why** | Provides visual and inferential evidence of shifts in failure composition. Non-parametric, no distributional assumptions. |
| **Method** | 10,000 bootstrap resamples → 95% BCa confidence intervals for each failure-type proportion at each quant level |
| **Visualization** | Stacked bar charts with error bars, or alluvial/Sankey diagrams showing how failure mass flows between categories across quant levels |

---

## RQ3: What is the empirical trade-off between agentic task effectiveness and computational efficiency across varying quantization levels?

**Core question:** How does `success_rate` trade off against `energy_per_task` as quantization changes?

---

### Test 11 · Pareto Efficiency Analysis

| Aspect | Detail |
|---|---|
| **Design** | Identify Pareto-optimal configurations on the `success_rate` vs. `energy_per_task` frontier |
| **Why** | Directly characterizes the trade-off — which configurations are never dominated on both dimensions? This is the most natural framing for a multi-objective optimization perspective. |
| **Method** | For each configuration, check if any other configuration has both higher success and lower energy. Plot the Pareto frontier. |
| **Statistical enrichment** | Bootstrap the Pareto frontier to get confidence bands; compute the **hypervolume indicator** as a scalar measure of the trade-off surface |

---

### Test 12 · Two-Way MANOVA (Quantization × Size on {success_rate, energy_per_task})

| Aspect | Detail |
|---|---|
| **Design** | Multivariate test with two DVs: `success_rate` and `energy_per_task` |
| **Why** | Tests whether quantization level affects the **joint distribution** of effectiveness and efficiency simultaneously, rather than testing each in isolation |
| **Assumptions** | Multivariate normality (Mardia's test), homogeneity of covariance matrices (Box's M) |
| **Post-hoc** | Univariate ANOVAs on each DV if MANOVA is significant, followed by discriminant analysis |
| **Effect size** | Pillai's trace (most robust to assumption violations) |

---

### Test 13 · Spearman Correlation + Regression of Energy–Performance Trade-off

| Aspect | Detail |
|---|---|
| **Design** | Correlate `success_rate` with `energy_per_task`; fit a regression model |
| **Model** | `success_rate ~ energy_per_task * quant + (1 | model/size) + (1 | domain)` |
| **Why** | Tests whether the **slope** of the performance–energy relationship differs by quantization level (i.e., does q4_k_m give you less "bang for your joule"?) |
| **Interpretation** | Interaction term (`energy_per_task × quant`) tells you if efficiency-effectiveness curves have different slopes per quant level |

---

### Test 14 · Efficiency Ratio Analysis with Non-parametric Comparison

| Aspect | Detail |
|---|---|
| **Derived metric** | `efficiency_ratio = success_rate / energy_per_task` (or `success_rate / log(energy_per_task)`) |
| **Test** | Kruskal-Wallis on efficiency ratio across quant levels, with Dunn's post-hoc |
| **Why** | Creates a single scalar "value for money" metric and tests whether quantization levels differ on it |
| **Robustness** | Paired Wilcoxon signed-rank test comparing bf16 vs. q8_0 and bf16 vs. q4_k_m (matched by model-size-domain) |

---

### Test 15 · Relative Change Analysis (Degradation Metrics)

| Aspect | Detail |
|---|---|
| **Derived metrics** | `Δ_success = (success_quant - success_bf16) / success_bf16` and `Δ_energy = (energy_bf16 - energy_quant) / energy_bf16` |
| **Design** | Paired analysis: each model-size-domain combo at bf16 is paired with its q8_0 and q4_k_m counterpart |
| **Test** | One-sample Wilcoxon signed-rank test — is the median relative degradation significantly different from zero? |
| **Why** | Directly quantifies: "How much performance do you lose for how much energy you save?" — the core trade-off question |
| **Visualization** | Scatter plot of Δ_success vs. Δ_energy with quadrant analysis (win-win, lose-win, etc.) |

---

## Summary Table

| RQ | Test | Type | Primary Purpose |
|---|---|---|---|
| **RQ1** | Two-Way ANOVA | Parametric | Main effects + interaction of size × quant |
| | Kruskal-Wallis | Non-parametric | Robustness check for RQ1 |
| | **Linear Mixed-Effects Model** | Parametric (mixed) | **Primary analysis** — handles nested/crossed design |
| | Spearman Correlation | Non-parametric | Monotonic scaling trend |
| | Cochran-Mantel-Haenszel | Stratified categorical | Association test on raw counts |
| **RQ2** | Chi-Square Homogeneity | Categorical | Failure distribution equality |
| | Multinomial Logistic Regression | Parametric | Failure type prediction with controls |
| | Friedman Test | Non-parametric | Matched comparison across quant levels |
| | **Compositional Data Analysis** | Specialized | **Methodologically rigorous** for proportional data |
| | Bootstrap CIs | Non-parametric | Assumption-free inference on composition |
| **RQ3** | Pareto Efficiency | Descriptive/Computational | Trade-off frontier characterization |
| | MANOVA | Parametric multivariate | Joint test on success + energy |
| | Regression with Interaction | Mixed-effects | Slope comparison across quant levels |
| | Kruskal-Wallis on Efficiency Ratio | Non-parametric | Scalar trade-off comparison |
| | **Relative Change Analysis** | Paired non-parametric | **Core trade-off quantification** |

---

## Multiple Comparisons Correction

> [!CAUTION]
> With many tests, apply family-wise error rate corrections:
> - **Within each RQ family**: Bonferroni or Holm-Bonferroni correction
> - **For post-hoc pairwise tests**: Tukey HSD (for ANOVA), Dunn's with Bonferroni (for Kruskal-Wallis), Nemenyi (for Friedman)
> - **Report both uncorrected and corrected p-values** for transparency

---

## Recommended Software

| Tool | Packages |
|---|---|
| **R** | `lme4`, `lmerTest`, `emmeans`, `car`, `compositions`, `nnet`, `PMCMRplus`, `boot` |
| **Python** | `statsmodels`, `scipy.stats`, `pingouin`, `scikit-learn`, `scikit-bio` |

---

## Open Questions

1. **Should domain be treated as a fixed or random effect?** If you want to generalize to *new* unseen domains, treat as random (as in the LMM above). If you only care about these 4 specific domains, treat as fixed and include it as a factor in ANOVA.

2. **How to handle the nesting of sizes within model families?** The 1.5B and 7B sizes only exist for deepseek-r1-qwen, 3B and 8B for ministral3, and 4B and 8B for qwen3. A crossed ANOVA treating size as a single factor conflates model architecture effects with scale effects. The LMM approach with `(1 | model)` handles this, but it's worth discussing explicitly.

3. **Should we include `energy_per_success` or `energy_per_action` in RQ3?** These may be more meaningful efficiency metrics than `energy_per_task`, but they have `Infinity` values when success_rate = 0 (e.g., deepseek 1.5B on alfworld). These would need to be handled (excluded or imputed).
