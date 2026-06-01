# Statistical Tests — Results Glossary & Methodology Guide

This document explains every key in the JSON result files, how each test works, and how to interpret the output.

---

## Common Keys (appear across many tests)

| Key | Meaning |
|---|---|
| `p` | **p-value** — probability of observing results this extreme if the null hypothesis (no effect) were true. If p < 0.05, we reject the null hypothesis and conclude a statistically significant effect exists. |
| `n` | **Sample size** — number of observations used in the test. |
| `size` | **Model parameter scale** — e.g., "1.5B", "4B", "8B". The number of parameters in billions. |
| `quant` | **Quantization level** — `bf16` (full precision baseline), `q8_0` (8-bit), or `q4_k_m` (4-bit). |
| `domain` | **Benchmark task domain** — `alfworld`, `dbbench`, `os`, or `webshop`. |

---

## RQ1 Results (`rq1_results.json`)

---

### Test 1: `test1_twoway_anova` — Two-Way Factorial ANOVA

**What it does:** Tests whether two categorical factors (model size and quantization level) have a statistically significant effect on a continuous outcome (success_rate), and whether they interact.

**How it works:**
1. Groups all observations by size and quant level.
2. Decomposes total variance in success_rate into: variance due to size, variance due to quant, variance due to their interaction, and unexplained residual variance.
3. Computes an F-statistic for each factor = (variance explained by factor) / (residual variance). Large F → the factor explains more variance than expected by chance.

**What we expect:** If quantization harms performance, C(quant) should be significant (p < 0.05). If larger models perform better, C(size) should be significant.

#### `shapiro_wilk` — Assumption check: is the data normally distributed?

```json
{ "size": "8B", "quant": "q4_k_m", "W": 0.771, "p": 0.0001, "n": 24 }
```

| Key | Meaning |
|---|---|
| `W` | **Shapiro-Wilk test statistic** — ranges from 0 to 1. Values close to 1 indicate normal distribution. Values well below 1 indicate non-normality. |
| `p` | If p < 0.05, the data **deviates significantly from normal**. Here p = 0.0001 means the 8B-q4_k_m success rates are NOT normally distributed. |
| `n` | Number of data points in this group (24 = 2 models × 4 domains × 3 runs). |

**Interpretation of our result:** Most groups have p < 0.05, meaning normality is violated. This is why we also run the non-parametric Kruskal-Wallis test (Test 2) as a robustness check.

#### `levene` — Assumption check: do all groups have equal variance?

```json
{ "F": 3.376, "p": 6.65e-05 }
```

| Key | Meaning |
|---|---|
| `F` | **Levene's F-statistic** — compares variance across groups. Large F → unequal variances. |
| `p` | If p < 0.05, variances are **significantly unequal**. Here p ≈ 0.00007 → variances are unequal (assumption violated). |

#### `anova_raw` / `anova_transformed` — The actual ANOVA results

```json
{
  "sum_sq": { "C(size)": 4.489, "C(quant)": 0.0009, "C(size):C(quant)": 0.022, "Residual": 8.026 },
  "df": { "C(size)": 4.0, "C(quant)": 2.0, "C(size):C(quant)": 8.0, "Residual": 201.0 },
  "F": { "C(size)": 28.11, "C(quant)": 0.012, "C(size):C(quant)": 0.068 },
  "PR(>F)": { "C(size)": 1.50e-18, "C(quant)": 0.988, "C(size):C(quant)": 0.9998 }
}
```

| Key | Meaning |
|---|---|
| `sum_sq` | **Sum of Squares** — total variance explained by each factor. C(size) = 4.49 means model size explains 4.49 units of variance. C(quant) = 0.0009 means quantization explains almost nothing. |
| `df` | **Degrees of freedom** — number of independent comparisons. Size has 5 levels → df = 4. Quant has 3 levels → df = 2. |
| `F` | **F-statistic** — ratio of explained variance to unexplained variance. Larger F = stronger effect. C(size) F = 28.11 is very large; C(quant) F = 0.012 is essentially zero. |
| `PR(>F)` | **p-value for each factor**. C(size) p = 1.50 × 10⁻¹⁸ (extremely significant). C(quant) p = 0.988 (completely non-significant). |
| `C(size):C(quant)` | **Interaction effect** — does the effect of quantization depend on model size? p = 0.9998 → no interaction at all. |

**How to compute partial η²:** η² = SS_factor / (SS_factor + SS_residual). For size: 4.489 / (4.489 + 8.026) = 0.359.

---

### Test 2: `test2_kruskal_wallis` — Kruskal-Wallis H Test

**What it does:** Non-parametric alternative to ANOVA. Ranks all observations and tests if rank distributions differ across groups. Does NOT require normality.

**How it works:**
1. Rank all success_rate values from lowest to highest.
2. Compare mean ranks across groups. If one group consistently has higher ranks, H will be large.

**What we expect:** Same conclusions as ANOVA but robust to non-normality.

```json
{
  "quant": { "H": 0.063, "p": 0.969, "epsilon_sq": -0.009 },
  "size": { "H": 124.72, "p": 5.25e-26, "epsilon_sq": 0.572 }
}
```

| Key | Meaning |
|---|---|
| `H` | **Kruskal-Wallis H statistic** — analogous to F in ANOVA but for ranks. Larger H → bigger differences between groups. H = 0.063 for quant means no difference at all. H = 124.72 for size means massive differences. |
| `epsilon_sq` (ε²) | **Effect size** — proportion of variance in ranks explained by the grouping. Ranges 0 to 1. ε² = 0.572 for size is a large effect. ε² ≈ −0.009 for quant is essentially zero (negative due to sampling). |

#### `size_dunn` — Post-hoc pairwise comparisons (Dunn's test)

```json
{ "4B": { "1.5B": 2.14e-20, "7B": 9.20e-13, "8B": 0.046 } }
```

Each value is a **Bonferroni-corrected p-value** for the pairwise comparison. If p < 0.05, those two groups are significantly different. Example: 4B vs. 1.5B has p = 2.14 × 10⁻²⁰ → 4B models vastly outperform 1.5B models.

---

### Test 3: `test3_lmm` — Linear Mixed-Effects Model

**What it does:** The gold-standard analysis. Models success_rate as a function of quantization, model size, and domain (fixed effects) while accounting for the fact that observations within the same model family are correlated (random effect).

**How it works:**
1. Fixed effects estimate the average effect of each predictor.
2. Random intercept for "model" allows each model family (deepseek, ministral, qwen) to have its own baseline performance.
3. Uses REML (Restricted Maximum Likelihood) estimation.

```json
{
  "converged": true,
  "fixed_effects": {
    "Intercept": { "coef": 0.037, "p": 0.715 },
    "C(quant, Treatment('bf16'))[T.q8_0]": { "coef": 0.008, "p": 0.892 },
    "C(quant, Treatment('bf16'))[T.q4_k_m]": { "coef": 0.031, "p": 0.611 },
    "C(domain)[T.webshop]": { "coef": 0.359, "p": 3.58e-31 },
    "size_num": { "coef": -0.0007, "p": 0.925 },
    "C(quant, Treatment('bf16'))[T.q8_0]:size_num": { "coef": -0.001, "p": 0.925 }
  },
  "random_effects": { "model_var": 0.0246, "resid_var": 0.0258, "icc": 0.488 },
  "lr_test_interaction": { "chi2": 0.260, "df": 2, "p": 0.878 }
}
```

| Key | Meaning |
|---|---|
| `converged` | Whether the model optimization converged. Must be `true` for results to be valid. |
| `coef` | **Coefficient (β)** — the estimated change in success_rate for that predictor vs. the reference level. Example: q8_0 coef = 0.008 means q8_0 success rate is 0.008 higher than bf16, on average. |
| `p` | p-value for that coefficient. If p < 0.05, the predictor has a significant effect. q8_0 p = 0.892 → not significant. |
| `model_var` | **Random intercept variance** — how much model families differ from each other. |
| `resid_var` | **Residual variance** — unexplained variance. |
| `icc` | **Intraclass Correlation Coefficient** — proportion of total variance due to model family. ICC = 0.488 means 48.8% of variance is between model families. |
| `chi2` (lr_test) | **Likelihood Ratio chi-squared** — compares the full model (with interaction) to the reduced model (without). |
| `df` (lr_test) | Degrees of freedom for the likelihood ratio test. |

---

### Test 4: `test4_spearman` — Spearman Rank Correlation

**What it does:** Tests whether there is a monotonic (consistently increasing or decreasing) relationship between model size and success rate. Unlike Pearson, it does NOT assume linearity or normality.

**How it works:**
1. Rank both variables (size and success_rate).
2. Compute the correlation between ranks.

```json
{ "overall": { "rho": 0.406, "p": 5.61e-10, "n": 216 } }
```

| Key | Meaning |
|---|---|
| `rho` (ρ) | **Spearman correlation coefficient** — ranges from −1 to +1. +1 = perfect positive monotonic relationship. 0 = no relationship. ρ = 0.406 indicates a moderate positive relationship (bigger models tend to have higher success). |
| `p` | If p < 0.05, the correlation is statistically significant. p = 5.61 × 10⁻¹⁰ → highly significant. |

---

### Test 5: `test5_cmh` — Cochran-Mantel-Haenszel Test

**What it does:** Tests association between quantization and success/failure outcome using raw counts, while controlling for domain as a stratification variable.

> [!NOTE]
> The CMH test encountered a compatibility error in our run (`'numpy.float64' object is not callable`). Results are unavailable but the other 4 tests provide comprehensive coverage of RQ1.

---

## RQ2 Results (`rq2_results.json`)

---

### Test 6: `test6_chi_square` — Chi-Square Test of Homogeneity

**What it does:** Tests whether the distribution of failure types (TLE, IF, IA, CF, TE, SysErr) is the same across all three quantization levels.

**How it works:**
1. Builds a contingency table: rows = quant levels, columns = failure types, cells = raw failure counts.
2. Computes expected frequencies under the null hypothesis (all quant levels have the same failure distribution).
3. χ² measures how much the observed counts deviate from expected.

```json
{ "overall": { "chi2": 94.55, "df": 10, "p": 6.68e-16, "cramers_v": 0.039 } }
```

| Key | Meaning |
|---|---|
| `chi2` (χ²) | **Chi-square statistic** — sum of (observed − expected)² / expected across all cells. Larger = bigger deviation from the null. |
| `df` | Degrees of freedom = (rows − 1) × (cols − 1). |
| `cramers_v` | **Cramér's V** — effect size for chi-square, ranges 0 to 1. V = 0.039 is negligible. The test is significant only because the sample size (tens of thousands of failures) is very large. |

#### `standardized_residuals`

```json
{ "TLE": { "bf16": -3.97, "q8_0": -0.88, "q4_k_m": 4.86 } }
```

| Key | Meaning |
|---|---|
| Each cell value | **Standardized residual (z-score)** — how many standard deviations the observed count is from the expected. \|z\| > 2 indicates a cell that significantly deviates. bf16 TLE = −3.97 means bf16 has significantly fewer TLE failures than expected. q4_k_m TLE = +4.86 means q4_k_m has significantly more. |

#### `pairwise`

```json
{ "bf16_vs_q4_k_m": { "chi2": 80.03, "p": 8.25e-16, "p_adjusted": 2.48e-15 } }
```

| Key | Meaning |
|---|---|
| `p_adjusted` | **Bonferroni-corrected p-value** — original p multiplied by the number of comparisons (3). Prevents false positives from multiple testing. |

---

### Test 7: `test7_multinomial_logit` — Multinomial Logistic Regression

**What it does:** Predicts the probability of each failure type as a function of quantization, model size, and domain simultaneously.

**How it works:**
1. Expands aggregated failure counts into individual observations (one row per failure).
2. Fits a multinomial logistic model where the DV is the failure category.
3. Estimates coefficients for each predictor for each failure category.

```json
{
  "llf": -33648.17,
  "aic": 67366.35,
  "bic": 67659.30,
  "pseudo_r2": 0.334,
  "lr_test": { "chi2": 33747.91, "df": 30, "p": 0.0 }
}
```

| Key | Meaning |
|---|---|
| `llf` | **Log-likelihood** — measure of model fit. Higher (less negative) = better fit. |
| `aic` | **Akaike Information Criterion** — penalizes model complexity. Lower = better. Used to compare models. |
| `bic` | **Bayesian Information Criterion** — like AIC but with stronger penalty for complexity. |
| `pseudo_r2` | **McFadden's pseudo-R²** — proportion of improvement over the null (intercept-only) model. 0.334 means the predictors explain 33.4% of the deviance. Domain and size drive most of this, not quantization. |
| `lr_test` | **Likelihood Ratio Test** — compares the full model to an intercept-only model. p = 0 means the model is significantly better than random guessing. |

---

### Test 8: `test8_friedman` — Friedman Test

**What it does:** Non-parametric matched test. Treats each (model, size, domain) combination as a "subject" measured at all 3 quantization levels. Tests if failure rates differ across quant levels.

**How it works:**
1. For each subject, rank the 3 quant values (1st, 2nd, 3rd).
2. If quant has no effect, mean ranks should be equal across levels.
3. χ²_F tests for systematic rank differences.

```json
{
  "TLE": { "chi2_F": 2.51, "p": 0.285, "kendalls_w": 0.052, "n": 24 },
  "SysErr": { "chi2_F": 7.27, "p": 0.026, "kendalls_w": 0.151, "n": 24 }
}
```

| Key | Meaning |
|---|---|
| `chi2_F` | **Friedman chi-squared** — test statistic. Larger = more systematic rank differences across quant levels. |
| `kendalls_w` | **Kendall's W** — effect size for Friedman test, ranges 0 to 1. W = 0.052 (TLE) is tiny; W = 0.151 (SysErr) is small-to-moderate. |
| `n` | Number of matched "subjects" (model-size-domain combinations). |

#### Post-hoc Wilcoxon signed-rank (when Friedman is significant)

```json
{ "SysErr_bf16_vs_q8_0": { "W": 17.5, "p": 0.016, "p_adjusted": 0.047 } }
```

| Key | Meaning |
|---|---|
| `W` | **Wilcoxon signed-rank statistic** — based on the ranks of the differences between paired values. Smaller W (relative to n) indicates one group is consistently higher. |
| `p_adjusted` | Bonferroni-corrected p-value. p_adj = 0.047 < 0.05 → bf16 has significantly more system errors than q8_0. |

---

### Test 9: `test9_coda` — Compositional Data Analysis (CLR + MANOVA)

**What it does:** Applies the methodologically correct approach for analyzing proportional data that sums to a whole. Transforms failure proportions using the Centered Log-Ratio (CLR) and then runs MANOVA.

**How it works:**
1. CLR transform: for each observation, take log of each proportion and subtract the mean of the logs. This maps compositional data into unconstrained Euclidean space.
2. MANOVA tests whether the mean CLR vectors differ across quantization levels.

```json
{
  "pillai": { "value": 0.021, "p": 0.999 },
  "wilks_lambda": { "value": 0.979, "p": 0.999 }
}
```

| Key | Meaning |
|---|---|
| `pillai` (value) | **Pillai's trace** — multivariate test statistic, ranges 0 to 1. Closer to 0 = groups are similar. 0.021 = essentially identical failure compositions. |
| `wilks_lambda` (value) | **Wilks' Lambda (Λ)** — ranges 0 to 1. Closer to 1 = groups are similar. Λ = 0.979 = nearly identical. |
| `p` | p = 0.999 → completely non-significant. Failure composition does NOT differ by quantization. |

---

### Test 10: `test10_bootstrap_ci` — Bootstrap Confidence Intervals

**What it does:** Estimates the uncertainty around each failure-type proportion using resampling (no distributional assumptions).

**How it works:**
1. For each quant level, resample the runs 10,000 times (with replacement).
2. Compute the failure-type proportion for each resample.
3. Take the 2.5th and 97.5th percentiles as the 95% confidence interval.

```json
{ "bf16": { "TLE": { "proportion": 0.215, "ci_low": 0.154, "ci_high": 0.284, "se": 0.033 } } }
```

| Key | Meaning |
|---|---|
| `proportion` | Observed proportion of total failures that are this type. 0.215 = 21.5% of bf16 failures are TLE. |
| `ci_low` / `ci_high` | **95% confidence interval bounds**. We are 95% confident the true proportion lies in [0.154, 0.284]. |
| `se` | **Bootstrap standard error** — standard deviation of the bootstrap distribution. Smaller SE = more precise estimate. |

**Interpretation:** If CIs for two quant levels overlap, the proportions are not significantly different.

---

## RQ3 Results (`rq3_results.json`)

---

### Test 11: `test11_pareto` — Pareto Efficiency Analysis

**What it does:** Identifies configurations that are "non-dominated" — no other configuration achieves both higher success AND lower energy.

**How it works:**
1. For each configuration, check if any other has ≥ success AND ≤ energy (with at least one strict inequality).
2. If no such configuration exists, it is Pareto-optimal.

```json
{ "overall_pareto": [
    { "config": "qwen3-4B-q8_0", "success_rate": 0.435, "energy_per_task": 21538 },
    { "config": "ministral3-3B-q4_k_m", "success_rate": 0.216, "energy_per_task": 2611 }
]}
```

| Key | Meaning |
|---|---|
| `config` | Model-size-quantization identifier. |
| `success_rate` | Mean task success rate for this configuration. |
| `energy_per_task` | Mean energy consumption in Joules per task. |
| `hypervolume_normalized` | Scalar measure of the area dominated by the Pareto frontier. Higher = better overall trade-off surface. |

---

### Test 12: `test12_manova` — Two-Way MANOVA

**What it does:** Tests whether quantization affects success_rate AND energy_per_task **jointly** (as a multivariate outcome).

```json
{
  "Pillais_trace": { "value": 0.045, "F": 2.44, "p": 0.046 },
  "Wilks_lambda": { "value": 0.955, "F": 2.46, "p": 0.045 }
}
```

| Key | Meaning |
|---|---|
| `value` | The multivariate test statistic value. |
| `F` | Approximate F-statistic derived from the multivariate statistic. |
| `p` | p = 0.046 → significant at α = 0.05. Quantization affects the joint {success, energy} distribution. Since RQ1 showed success is unaffected, this significance is driven by the energy dimension. |

---

### Test 13: `test13_regression` — Energy-Performance Correlation & Regression

**What it does:** Tests (a) if success rate correlates with energy, and (b) if the energy-success relationship slope differs by quantization level.

```json
{
  "spearman_overall": { "rho": 0.045, "p": 0.512 },
  "spearman_bf16": { "rho": 0.183, "p": 0.124, "n": 72 },
  "lmm_converged": true
}
```

| Key | Meaning |
|---|---|
| `rho` | Spearman correlation between energy_per_task and success_rate. ρ = 0.045 → no relationship. More energy does NOT mean more success. |
| `lmm_converged` | Whether the mixed-effects regression converged. |

---

### Test 14: `test14_efficiency_ratio` — Efficiency Ratio Analysis

**What it does:** Computes a single "value for money" metric: efficiency = success_rate / log(energy_per_task), then compares across quant levels.

```json
{
  "kruskal_wallis": { "H": 0.069, "p": 0.966, "epsilon_sq": -0.009 },
  "wilcoxon_bf16_vs_q8_0": { "W": 35.0, "p": 0.016, "median_diff": -0.00015, "r": 0.493, "n": 24 }
}
```

| Key | Meaning |
|---|---|
| `H` / `p` (kruskal_wallis) | Kruskal-Wallis test. p = 0.966 → efficiency ratios are statistically identical across quant levels when compared as independent groups. |
| `W` (wilcoxon) | Paired Wilcoxon statistic. Compares matched configurations (same model-size-domain at different quant levels). |
| `median_diff` | Median of the paired differences. ≈ 0 means efficiency is nearly identical. |
| `r` | **Effect size (r = Z / √n)** — ranges 0 to 1. r = 0.493 is a medium-to-large effect. Interpretation: q8_0 is slightly more efficient per configuration, though the absolute magnitude is tiny. |

---

### Test 15: `test15_relative_change` — Relative Change Analysis (Degradation Metrics)

**What it does:** Pairs each bf16 configuration with its quantized counterpart and measures: (a) how much success changed, (b) how much energy was saved.

```json
{
  "bf16_to_q8_0": {
    "n": 24,
    "success_delta_mean": 0.003,
    "success_delta_median": 0.0,
    "energy_savings_mean": 2951.07,
    "energy_savings_median": 1391.01,
    "relative_energy_median_pct": 27.41,
    "wilcoxon_success": { "W": 44.5, "p": 0.615, "r": 0.103 },
    "wilcoxon_energy": { "W": 18.0, "p": 3.02e-05, "r": 0.852 },
    "quadrant": { "win_win": 16, "trade_off": 7, "inverse": 1, "lose_lose": 0 }
  }
}
```

| Key | Meaning |
|---|---|
| `success_delta_mean` / `median` | Mean/median change in success rate (quant − bf16). Positive = quantized is better. Median = 0 → no change. |
| `energy_savings_mean` / `median` | Mean/median energy saved (bf16 − quant) in Joules per task. Positive = quantized uses less energy. |
| `relative_energy_median_pct` | Median percentage of energy saved. 27.41% means q8_0 uses ~27% less energy. |
| `wilcoxon_success` | Paired Wilcoxon testing if success change ≠ 0. p = 0.615 → no significant change. |
| `wilcoxon_energy` | Paired Wilcoxon testing if energy change ≠ 0. p = 3.02 × 10⁻⁵ → highly significant savings. |
| `r` (effect size) | r = 0.852 for energy is a **large effect** (Cohen's thresholds: 0.1 small, 0.3 medium, 0.5 large). |
| `quadrant` | Counts of configurations in each trade-off quadrant: |
| — `win_win` | ↑ performance AND ↓ energy (best case). 16/24 = 67%. |
| — `trade_off` | ↓ performance but ↓ energy (acceptable). 7/24 = 29%. |
| — `inverse` | ↑ performance but ↑ energy (unusual). 1/24 = 4%. |
| — `lose_lose` | ↓ performance AND ↑ energy (worst case). **0/24 = 0%**. |

---

## Quick Decision Guide

| If you see... | It means... |
|---|---|
| p < 0.001 | Very strong evidence against the null hypothesis |
| p < 0.05 | Statistically significant at conventional threshold |
| p > 0.05 | No significant effect detected |
| Cramér's V < 0.1 | Negligible practical effect (even if p is small) |
| ε² or η² > 0.14 | Large effect size |
| Spearman ρ > 0.4 | Moderate-to-strong monotonic relationship |
| ICC > 0.4 | Substantial clustering by group (model family) |
| Wilcoxon r > 0.5 | Large paired effect |
| Kendall's W < 0.1 | Negligible agreement/effect |
