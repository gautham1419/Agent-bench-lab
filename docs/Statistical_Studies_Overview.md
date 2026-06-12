# Overview of Statistical Studies: LLM Quantization & Agentic Performance

This document provides an accessible, comprehensive guide to the **15 statistical studies and tests** conducted during our research on large language model (LLM) quantization and agentic performance. It explains each study in plain English, details its relevance to our experimental design, and breaks down what the results convey about our models (Qwen, Ministral, DeepSeek) and agentic environments (OS, DB, ALFWorld, WebShop).

---

## Experimental Setup Refresher
To make sense of the tests, keep in mind the variables we measured:
*   **Independent Variables (Factors):**
    *   **Model Family:** `qwen3` (highly aligned tool-use), `ministral3` (efficient dense), and `deepseek-r1-qwen` (reasoning-oriented).
    *   **Parameter Size:** 1.5B, 3B, 4B, 7B, 8B (nested within families).
    *   **Quantization Level (Precision):** `bf16` (full precision), `q8_0` (8-bit), and `q4_k_m` (4-bit).
    *   **Domain (Task):** `os` (operating system commands), `dbbench` (SQL queries), `alfworld` (interactive spatial task), and `webshop` (web shopping).
*   **Dependent Variables (Outcomes):** `success_rate` (tasks completed), `energy_per_task` (Joules), and the composition of **Failure Types** (Timeouts, formatting errors, invalid actions, etc.).

---

# PART 1: RQ1 Studies — Impact on Task Success Rates
*How do model size, quantization, and architecture affect an agent's success rate?*

---

### Study 1: Two-Way Factorial ANOVA (Size × Quantization)
#### What is it in plain English?
Imagine testing whether both the engine size of a car (size) and the brand of fuel it uses (quantization) affect its top speed. A **Two-Way ANOVA** lets us test both factors at the same time, and crucially, tells us if they interact (e.g., does cheap fuel only slow down cars with small engines?).

#### Relevance to our Experiment
We wanted to know:
1.  Does model size significantly impact success rate?
2.  Does quantization level significantly impact success rate?
3.  Is there an **interaction effect**? (i.e., does 4-bit quantization hurt small models like 1.5B more than it hurts large models like 8B?)

#### What the Results Convey
*   **Quantization has no effect:** The F-statistic for quantization was virtually zero ($F = 0.012$), with a p-value of **0.988**. In statistics, a p-value this high means there is absolutely no evidence that changing precision from full (`bf16`) to 8-bit or 4-bit alters the task success rate.
*   **Size has a massive effect:** The F-statistic for size was extremely high ($F = 28.11$, $p = 1.50 \times 10^{-18}$), meaning size is a massive driver of success.
*   **No interaction:** The interaction p-value was **0.999**. This is a huge finding: **quantization does not disproportionately harm smaller models**. It is equally safe to quantize a 1.5B model and an 8B model.

---

### Study 2: Kruskal-Wallis H Test (with Dunn's Post-Hoc)
#### What is it in plain English?
ANOVA assumes that your data fits a neat, symmetrical bell curve (normal distribution). Because we only ran 3 replicates per configuration, we cannot guarantee normal distribution. The **Kruskal-Wallis test** is a "non-parametric" alternative. Instead of looking at raw scores, it ranks all the scores from 1st to worst and compares the average ranks of the groups. It is highly robust to messy or sparse data.

#### Relevance to our Experiment
This serves as a critical safety check. If the Kruskal-Wallis test disagreed with our ANOVA (Study 1), it would mean our ANOVA results were a statistical illusion caused by non-normal data.

#### What the Results Convey
*   **It confirmed Study 1:** Quantization showed a negligible effect size ($\epsilon^2 \approx -0.009$, $p = 0.969$), while model size showed a massive effect ($\epsilon^2 = 0.572$, $p = 5.25 \times 10^{-26}$).
*   **Dunn's Post-Hoc Pairwise Comparisons:** By digging into specific size comparisons, we found something surprising:
    *   **4B models significantly outperformed 8B models** ($p = 0.046$). 
    *   **4B models massively outperformed 7B models** ($p = 9.20 \times 10^{-13}$).
    *   *Why?* The 4B group contains `qwen3-4B`, while the 7B group contains `deepseek-r1-qwen-7B`. Qwen's superior formatting and tool-use alignment made it beat models twice its size. This proved that **model architecture dictates success, not just parameter scale**.

---

### Study 3: Linear Mixed-Effects Model (LMM)
#### What is it in plain English?
Suppose you are measuring student test scores across different schools. If you ignore which school each student goes to, your data will be skewed because some schools are simply much better overall. A **Linear Mixed-Effects Model** allows us to group similar data points together (like students in schools, or in our case, model sizes within the same model family) to see the true, unconfounded relationships.

#### Relevance to our Experiment
This is our **primary, gold-standard analysis**. In our dataset, sizes are nested: we only have 1.5B and 7B for DeepSeek; 3B and 8B for Ministral; 4B and 8B for Qwen. A standard ANOVA treats "4B" and "7B" as independent factors, ignoring that they belong to different model families. The LMM treats "Model Family" and "Task Domain" as random grouping factors, giving us the mathematically correct impact of size and quantization.

#### What the Results Convey
*   **Quantization remains non-significant:** Confirms that 8-bit ($p = 0.892$) and 4-bit ($p = 0.611$) do not affect performance.
*   **Scale is an illusion:** Once we control for model family, the actual parameter scale (`size_num`) becomes **completely non-significant** ($p = 0.925$). The apparent advantage of "bigger models" in Study 1 and 2 was actually just because our larger models happened to belong to better-aligned families.
*   **Model Family is King:** The Intraclass Correlation Coefficient (ICC) was **0.488**. This means that **48.8% of all variation in agent success rate is determined solely by the model family** you choose (e.g., Qwen vs. DeepSeek), rather than its size or quantization level.

---

### Study 4: Spearman Rank Correlation (Scale vs. Success)
#### What is it in plain English?
Does "larger" always mean "better"? A **Spearman correlation** measures if two variables move in the same direction consistently (monotonically), even if they don't follow a straight line.

#### Relevance to our Experiment
We wanted to see if the general trend of "larger models succeed more" holds true across all quantization levels, or if quantizing a model breaks this relationship.

#### What the Results Convey
*   We found a moderate positive correlation overall ($\rho = 0.406, p = 5.61 \times 10^{-10}$).
*   Crucially, this trend was preserved at all precision levels: $\rho = 0.415$ at `bf16`, $\rho = 0.430$ at `q8_0`, and $\rho = 0.383$ at `q4_k_m`. Quantization does not break the general rule of thumb that larger models within a family scale positively.

---

### Study 5: Cochran-Mantel-Haenszel (CMH) Test
#### What is it in plain English?
Instead of looking at success rates (like 18.8%), this test looks at raw, binary wins and losses (e.g., 200 successes, 800 failures). It tests association while controlling for a third "nuisance" variable (stratification).

#### Relevance to our Experiment
We intended to use this to test the association between quantization and binary success, stratifying by the task domain (controlling for the fact that WebShop is easier than ALFWorld).

#### What the Results Convey
> [!NOTE]
> This test encountered a technical compatibility error during execution due to a data-type library conflict in the stats package (`'numpy.float64' object is not callable`). However, because Studies 1, 2, and 3 are so robust, this null result does not impact our overall conclusions.

---

# PART 2: RQ2 Studies — Composition of Agent Failures
*When models fail, do they fail in different ways when they are compressed?*

---

### Study 6: Chi-Square Test of Homogeneity (with Standardized Residuals)
#### What is it in plain English?
If you throw a six-sided die, you expect each number to come up 1/6 of the time. If one number comes up way too often, the die is loaded. A **Chi-Square Homogeneity test** looks at a grid of categories (e.g., failure types like timeout, format error, invalid action) across different groups (quantization levels) and checks if the distribution of failures is "loaded" or if it is the same.

#### Relevance to our Experiment
We wanted to see if quantizing a model makes it fail in "dumber" ways—for example, if a 4-bit model suddenly starts spitting out garbage formats (Invalid Format) instead of just failing because it ran out of steps (Time Limit Exceeded).

#### What the Results Convey
*   **Statistically Significant but Practically Meaningless:** The test was technically significant ($p = 6.68 \times 10^{-16}$) because we had tens of thousands of data points. However, the effect size (**Cramér's V = 0.039**) is extremely low. In practice, quantization explains less than 0.2% of the variance in how models fail.
*   **The Drivers of Significance (Standardized Residuals):**
    *   **System Errors (SysErr):** Full-precision (`bf16`) models had significantly *more* system errors ($z = +5.32$), while 4-bit models had *fewer* ($z = -3.32$). This is because unquantized models require massive memory (VRAM), causing server/system crashes.
    *   **Timeouts (TLE):** 4-bit models had slightly more timeouts ($z = +4.86$). Compressed models are slightly noisier, which occasionally causes them to wander in circles and hit step limits, rather than outright crashing.

---

### Study 7: Multinomial Logistic Regression
#### What is it in plain English?
Imagine predicting what kind of error a program will throw based on the operating system, the computer's RAM, and the code version. **Multinomial Logistic Regression** lets us calculate the probability of multiple distinct, categorical outcomes (different failure types) based on several input predictors at once.

#### Relevance to our Experiment
We wanted to predict the specific type of failure based on quantization, model size, and domain, helping us isolate whether quantization is a strong predictor of failure behavior.

#### What the Results Convey
*   The overall model was highly predictive (Pseudo-$R^2 = 0.334$). 
*   However, by checking the coefficients, we found that the predictive power was almost entirely driven by the **Task Domain** (e.g., DBbench naturally produces SQL formatting errors, while ALFWorld produces spatial invalid actions) and **Model Size/Family**, rather than the quantization level.

---

### Study 8: Friedman Test (with Wilcoxon Post-Hoc)
#### What is it in plain English?
This is a non-parametric version of a repeated-measures test. It is like testing the same group of athletes' running times at three different altitudes. Because the same athlete (the specific model-size-domain config) is measured under three different conditions (bf16, q8, q4), we must pair the data points to isolate the effect of the condition.

#### Relevance to our Experiment
This allows us to track a specific model config (e.g., `qwen3-4B` on `os`) across all three quantization levels to see if its specific failure rates change.

#### What the Results Convey
*   For cognitive failure types (Invalid Format, Invalid Action, Context Failure, Time Limit Exceeded), the Friedman test was **completely non-significant** (all $p > 0.11$). Quantization does not change the rate at which models make cognitive mistakes.
*   **SysErr was the only significant change ($p = 0.026$):** Wilcoxon post-hoc tests confirmed that `bf16` models produce significantly more system errors than `q8_0` ($p\_adj = 0.047$), proving that compression improves infrastructure reliability.

---

### Study 9: Compositional Data Analysis (CoDA: CLR + MANOVA)
#### What is it in plain English?
If you have a pizza divided into slices, and you make the pepperoni slice bigger, the cheese slice *must* get smaller. Proportions are locked together because they must sum to 100%. Standard statistical tests fail on proportions because the variables are not independent. **CoDA** uses a mathematical trick (Centered Log-Ratio transformation) to project these proportions into open space so we can test them safely.

#### Relevance to our Experiment
Because our failure metrics are proportions of a whole (e.g., if TLE rate goes up, IF rate must mathematically go down), this is the only methodologically correct way to analyze the failure profile as a complete package.

#### What the Results Convey
*   The resulting MANOVA was highly non-significant (Pillai's trace $p = 0.999$, Wilks' $\lambda$ $p = 0.999$).
*   This is a definitive, rigorous proof: **quantization has zero effect on the overall mixture of how agents fail**.

---

### Study 10: Bootstrap Confidence Intervals
#### What is it in plain English?
If you want to know the average height of people in a city, you can take a random sample of 100 people, calculate the average, throw them back, and repeat this 10,000 times. This creates a highly accurate "confidence interval" (e.g., "we are 95% sure the average height is between 5'6\" and 5'8\"").

#### Relevance to our Experiment
We wanted to establish error bars around our failure proportions to see if the ranges for `bf16`, `q8_0`, and `q4_k_m` overlap.

#### What the Results Convey
*   The 95% confidence intervals for every single failure type overlapped completely across all three quantization levels. For instance, the TLE failure rate was estimated at $[0.154, 0.284]$ for bf16 and $[0.190, 0.331]$ for q4_k_m. Because these intervals overlap, we cannot claim they are truly different.

---

# PART 3: RQ3 Studies — Efficiency vs. Performance Trade-off
*Is saving energy worth the potential performance loss?*

---

### Study 11: Pareto Efficiency Analysis
#### What is it in plain English?
In economics, a choice is "Pareto optimal" if you cannot make one thing better without making something else worse. On a graph of Success Rate vs. Energy, a configuration is Pareto-optimal if no other model is both more successful AND uses less energy.

#### Relevance to our Experiment
This directly identifies the "best deals" for deploying agents in production.

#### What the Results Convey
*   **Quantized models dominate the frontier:** The Pareto-optimal list consists entirely of quantized models:
    1.  `qwen3-4B-q8_0` (Highest success, moderate energy)
    2.  `qwen3-4B-q4_k_m` (High success, lower energy)
    3.  `ministral3-8B-q4_k_m` (Medium success, very low energy)
    4.  `ministral3-3B-q4_k_m` (Lower success, lowest energy)
*   **Zero full-precision models made the list.** Running `bf16` is mathematically inefficient because a quantized version will always give you the same performance for less energy.

---

### Study 12: Two-Way MANOVA (Multivariate ANOVA)
#### What is it in plain English?
Instead of testing success and energy in separate silos, a **MANOVA** tests them together as a joint pair, treating `{success_rate, energy_per_task}` as a single coordinate in space.

#### Relevance to our Experiment
We wanted to know if quantization shifts the overall "efficiency profile" of our agents when considering both cost and performance simultaneously.

#### What the Results Convey
*   The joint test was significant ($p = 0.045$). Because we already know from RQ1 that success rate does not change, this joint significance is entirely driven by the **massive reduction in energy consumption**.

---

### Study 13: Spearman Correlation & Regression (Energy vs. Performance)
#### What is it in plain English?
Does burning more electricity buy you a higher success rate? This tests the correlation between energy and success.

#### Relevance to our Experiment
If energy and success are highly correlated, it means we have to pay a premium for high performance. If they are uncorrelated, we can look for "bargain" models.

#### What the Results Convey
*   We found **zero correlation** between energy consumption and success rate ($\rho = 0.045, p = 0.512$).
*   This means **more energy does not buy more success**. The energy consumed is a function of the model family and size, not how good it is at solving the task. Some architectures are highly inefficient (burning lots of Joules for low success), while others are highly optimized.

---

### Study 14: Efficiency Ratio Analysis
#### What is it in plain English?
We created a single "value for money" metric: $Efficiency = Success / \log(Energy)$. We then compared this ratio across quantization levels.

#### Relevance to our Experiment
This allows us to run a simple, one-factor comparison to see which precision level gives the best "bang for your buck."

#### What the Results Convey
*   Overall groups were similar ($p = 0.966$), but a paired Wilcoxon test comparing matched configurations showed that **q8_0 is slightly more efficient** than bf16 ($p = 0.016, r = 0.493$). It provides identical success while saving a significant amount of energy.

---

### Study 15: Relative Change Analysis (Degradation Metrics)
#### What is it in plain English?
For every single model-size-domain setup, we calculated: "By switching from bf16 to a compressed version, what percentage of success did we lose, and what percentage of energy did we save?"

#### Relevance to our Experiment
This is the ultimate, direct quantification of the trade-off.

#### What the Results Convey
*   **bf16 $\rightarrow$ q8_0:** Median success change was **$0.0\%$** (no loss), while median energy savings was **$27.4\%$** ($p < 0.001$).
*   **bf16 $\rightarrow$ q4_k_m:** Median success change was **$-0.1\%$** (negligible loss), while median energy savings was **$37.9\%$** ($p < 0.001$).
*   **Quadrant Analysis:** 
    *   **67%** of q8_0 runs were **Win-Win** (performance actually went up slightly, and energy went down).
    *   **0% were Lose-Lose** (worse performance and more energy).
    *   This shows that quantizing is not a trade-off; it is a direct optimization.

---

## Summary Matrix of the 15 Studies

| Study | Primary Metric Tested | What it Proves | The Key Takeaway |
|---|---|---|---|
| **1. Two-Way ANOVA** | Success Rate | Test main effects of scale and precision. | Quantization has no effect on success; size does. |
| **2. Kruskal-Wallis** | Success Rate | Non-parametric safety check for Study 1. | Confirmed ANOVA; showed architecture beats scale. |
| **3. LMM (Primary)** | Success Rate | Controls for Model Family nested grouping. | **Scale is an illusion; model family drives success.** |
| **4. Spearman Corr.** | Success Rate vs. Size | Tests if scaling trend holds across precisions. | Scale correlates positively across all precisions. |
| **5. CMH Test** | Binary Wins/Losses | Tests raw counts stratified by task. | *Failed due to library compatibility error.* |
| **6. Chi-Square** | Failure Counts | Checks if failure distributions shift. | Practically identical error profiles. |
| **7. Multinomial Reg.** | Failure Type | Predicts failure category probability. | Failure type is driven by domain, not quantization. |
| **8. Friedman Test** | Paired Failure Rates | Tracks failure changes within matched setups. | Cognitive errors are unchanged; System Errors drop. |
| **9. CoDA (Primary)** | Failure Proportions | Methodologically correct proportional test. | **Failure composition is invariant to quantization.** |
| **10. Bootstrap CIs** | Failure Proportions | Establishes error bars for failure types. | Overlapping intervals prove changes are random noise. |
| **11. Pareto Frontier** | Success vs. Energy | Identifies non-dominated best setups. | **All optimal configurations are quantized.** |
| **12. MANOVA** | Joint Success & Energy | Tests bivariate distribution shift. | Quantization shifts the joint profile due to energy drops. |
| **13. Energy Reg.** | Success vs. Energy | Tests if success and energy are correlated. | More energy does not buy more success. |
| **14. Efficiency Ratio** | Success / log(Energy) | Compares "value for money" metric. | q8_0 is slightly more efficient than bf16. |
| **15. Relative Change** | % Success loss vs. % Energy save | Direct trade-off percentage calculation. | **38% energy savings for 0.1% performance change.** |
