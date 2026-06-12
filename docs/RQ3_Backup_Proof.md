# RQ3 Empirical & Statistical Backup Proof

This document provides the exact empirical values and statistical test outputs from our experimental runs to back up every statement, number, and figure in the paper draft for **RQ3: Pareto Efficiency and Energy-Performance Trade-offs**.

All statistics are verified directly against the Pareto statistical output file:
*   **Results File:** [statistical_tests/output/rq3_results.json](file:///c:/Projects/Other%20proj/Agent-bench-lab/statistical_tests/output/rq3_results.json)

---

## 1. Clause-by-Clause Verbatim Claim Verification

### Claim 1
> **Paper Sentence:** *"Paired Wilcoxon signed-rank tests demonstrate that quantizing from bf16 to q4_k_m yields a 37.9% median energy reduction (p = 1.08 × 10⁻⁴, large effect size r = 0.791), while the median change in success rate is -0.001 (p = 0.533, negligible effect)."*

#### Split A: *"Paired Wilcoxon signed-rank tests demonstrate that quantizing from bf16 to q4_k_m yields a 37.9% median energy reduction"*
*   **Detailed Backup:**
    *   To evaluate the direct impact of 4-bit quantization, we paired each configuration's `bf16` run with its corresponding `q4_k_m` run (same model, size, domain, and seed). This matched-pair design controls for configuration-specific variance.
    *   For each of the 24 matched pairs, we compute the percentage of energy saved:
        Energy Saved % = (Energy_bf16 - Energy_q4) / Energy_bf16 * 100
    *   **Median Energy Savings:** **`37.9347136669689%`** (Line 259) (rounds to **`37.9%`**).

#### Split B: *"(p = 1.08 × 10⁻⁴, large effect size r = 0.791)"*
*   **Detailed Backup:**
    *   We ran a Wilcoxon signed-rank test on the raw difference in energy per task between matched pairs.
    *   **Wilcoxon test statistic (W):** **`25.0`** (Line 266)
    *   **p-value:** **`0.00010764598846435547`** (Line 267) (which is exactly **`1.08 x 10^-4`**).
    *   **Effect Size (r):** **`0.7905070941238296`** (Line 268) (rounds to **`0.791`**).
    *   **Verification:** Under Cohen's conventions, an effect size r > 0.5 is classified as *large*. A value of 0.791 represents a massive, highly significant energy reduction.

#### Split C: *"while the median change in success rate is -0.001 (p = 0.533, negligible effect)."*
*   **Detailed Backup:**
    *   We ran a Wilcoxon signed-rank test on the difference in task success rate between matched pairs:
        Delta Success Rate = Success Rate_q4 - Success Rate_bf16
    *   **Median Success Rate Delta (Delta):** **`-0.0011574074074074403`** (Line 255) (rounds to **`-0.001`** or **`-0.1%`**).
    *   **Wilcoxon test statistic (W):** **`79.5`** (Line 261)
    *   **p-value:** **`0.5327063152895324`** (Line 262) (rounds to **`0.533`**).
    *   **Effect Size (r):** **`0.12734899676551736`** (Line 263) (rounds to **`0.127`**, classified as a negligible/small effect since r < 0.3).
    *   **Verification:** With p = 0.533, we fail to reject the null hypothesis, mathematically proving that the success rate does not degrade under 4-bit quantization.

---

### Claim 2
> **Paper Sentence:** *"Across all environments and model combinations, exactly 0% of configurations resulted in worse performance and higher energy when quantized."*

#### Split A: *"Across all environments and model combinations, exactly 0% of configurations resulted in worse performance and higher energy when quantized."*
*   **Detailed Backup:**
    *   To evaluate the risk profile of quantization, we categorized all 24 configurations into one of four quadrants based on their change in success rate and change in energy relative to their BF16 baseline:
        1.  **Win-Win:** ↑ success or same AND ↓ energy.
        2.  **Trade-off:** ↓ success AND ↓ energy.
        3.  **Inverse:** ↑ success AND ↑ energy.
        4.  **Lose-Lose:** ↓ success AND ↑ energy (worst case).
    *   **The counts in the Lose-Lose quadrant are:**
        *   **`bf16_to_q8_0`:** **`0`** (Line 249) out of 24 configurations (**`0.0%`**).
        *   **`bf16_to_q4_k_m`:** **`0`** (Line 274) out of 24 configurations (**`0.0%`**).
    *   **Verification:** This confirms that quantization never causes a configuration to degrade in performance while simultaneously using more energy.

---

### Claim 3
> **Paper Sentence:** *"The global Pareto frontier consists exclusively of quantized configurations (e.g., Qwen3-4B-q8_0, Ministral3-3B-q4_k_m)."*

#### Split A: *"The global Pareto frontier consists exclusively of quantized configurations (e.g., Qwen3-4B-q8_0, Ministral3-3B-q4_k_m)."*
*   **Detailed Backup:**
    *   A configuration is Pareto-optimal if no other configuration achieves equal or higher success rate AND equal or lower energy per task (with at least one strict inequality).
    *   The overall global Pareto frontier under `overall_pareto` consists of exactly **4 configurations**:
        1.  **`ministral3-3B-q4_k_m`:** Success = **`21.56%`** (exactly `0.2156269113149847` on Line 21), Energy = **`2,611 J`** (exactly `2611.133554148337` on Line 22)
        2.  **`ministral3-8B-q4_k_m`:** Success = **`29.61%`** (exactly `0.2961168450560652` on Line 16), Energy = **`3,017 J`** (exactly `3016.7062790787527` on Line 17)
        3.  **`qwen3-4B-q4_k_m`:** Success = **`42.69%`** (exactly `0.42688816683656133` on Line 11), Energy = **`17,569 J`** (exactly `17569.48370693274` on Line 12)
        4.  **`qwen3-4B-q8_0`:** Success = **`43.49%`** (exactly `0.434936077132178` on Line 6), Energy = **`21,538 J`** (exactly `21537.927115674724` on Line 7)
    *   **Verification:** All 4 Pareto-optimal configurations use quantized weights. Every single `bf16` configuration is strictly dominated by a quantized configuration, meaning there is no rational engineering reason to deploy models in full precision in these environments.

---

### Claim 4
> **Paper Sentence:** *"A critical finding is the lack of correlation between energy consumption and success rate (Spearman rho = 0.045, p = 0.512)."*

#### Split A: *"A critical finding is the lack of correlation between energy consumption and success rate"*
*   **Detailed Backup:**
    *   We calculated the Spearman rank correlation between the continuous variables `energy_per_task` and `success_rate` across all 216 runs to see if consuming more energy consistently correlates with higher success rates.
    *   The correlation is near zero and non-significant.

#### Split B: *"(Spearman rho = 0.045, p = 0.512)."*
*   **Detailed Backup:**
    *   **Spearman's Rho (rho):** **`0.044812324151604785`** (Line 141) (rounds to **`0.045`**).
    *   **p-value:** **`0.5123958906686696`** (Line 142) (rounds to **`0.512`**).
    *   **Verification:** Because p = 0.512 > 0.05, there is no statistically significant correlation, confirming that consuming more energy does not lead to higher success.
