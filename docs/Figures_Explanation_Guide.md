# Explanatory Guide to the Paper Figures

This guide explains what each figure in the paper represents, what it is trying to communicate, and why it matters. It is written for readers with **zero background in statistics or machine learning**.

---

## Part 1: How Compressing and Scaling Models Affects Success (RQ1)

### Figure 1(a): Success by Quantization
* **What is "Quantization"?** 
  Think of quantization like compressing a large video file so it takes up less space on your phone. In AI, it means reducing the precision of the numbers that make up the AI's brain (from the original `F16/BF16` down to `8-bit` or `4-bit`). 
* **What the graph shows:** 
  It shows the average percentage of tasks the models completed successfully at each compression level. 
  * Uncompressed (`F16/BF16`): **18.8%** success
  * Lightly compressed (`8-bit` / `Q8_0`): **19.1%** success
  * Heavily compressed (`4-bit` / `Q4_K_M`): **19.3%** success
* **The Takeaway:** 
  The three bars are almost identical (differing by less than 1 percentage point). This means **compressing the models does not hurt their ability to solve tasks.** You can run a smaller, compressed model and get the same success rate as the massive, uncompressed version.

---

### Figure 1(b): Success by Family & Size
* **Why did we group it this way?** 
  We compared three different AI "families" (DeepSeek, Ministral, and Qwen). Each family has a "Smaller" and a "Larger" size.
* **What the graph shows:**
  * **DeepSeek (1.5B vs. 7B):** Success is extremely low for both (0.3% vs. 3.3%). Making the model bigger did not help because the DeepSeek model family struggles with the basic formatting requirements of these tests.
  * **Ministral3 (3B vs. 8B):** Success goes up from **18.3%** to **29.0%**. Here, making the model larger worked as expected: bigger size equals better performance.
  * **Qwen3 (4B vs. 8B):** Success actually **dropped** from **43.2%** (Qwen3-4B) to **20.6%** (Qwen3-8B). This is an anomaly! It happens because the 4B version is specialized for tool-use, while the 8B version is a general chat model.
* **The Takeaway:** 
  **Raw size does not guarantee success.** An AI's specific family and how it was trained (its architecture) is far more important than how many parameters (size) it has.

---

### Figure 1(c): Domain Effect
* **What is a "Domain" and "LMM $\hat{\beta}$"?** 
  A domain is a specific testing environment (like managing an operating system, querying a database, or shopping online). The statistic $\hat{\beta}$ (beta) is a way of measuring how much "easier" a task is compared to a starting baseline.
* **What the graph shows:**
  We set `ALFWorld` (a text-based simulator game) as our starting baseline difficulty (valued at `0.00`). The other bars show how much higher the success rate gets in other environments:
  * Operating System (`OS`): adds **+9.4%** success compared to ALFWorld.
  * Database (`DB`): adds **+16.6%** success.
  * WebShop (Online Shopping): adds **+35.9%** success.
* **The Takeaway:** 
  WebShop is the easiest environment for these AIs, while ALFWorld is by far the hardest. The difficulty of the test environment itself plays a massive role in whether the AI succeeds.

---

## Part 2: How and Why Models Fail (RQ2)

### Figure 2(a): Failure Type Proportions
* **What are the failure types?**
  * **TLE (Time Limit Exceeded):** The AI took too many steps.
  * **IF (Interaction Failure):** The AI could not communicate with the environment.
  * **IA (Invalid Action):** The AI tried to perform an impossible action.
  * **CF (Crash Failure):** The AI program crashed.
  * **TE (Token Exceeded):** The AI generated too much text and ran out of memory.
* **What the graph shows:**
  It shows the proportion of each failure type across the three compression levels.
* **The Takeaway:**
  The heights of the bars for each failure category remain almost identical across all compression levels. Compressing an AI **does not change why or how it fails.** The failure patterns are baked into the model family.

---

### Figure 2(b): System Errors with 95% Confidence Intervals
* **What is a "System Error" and a "Confidence Interval"?**
  System errors are infrastructure crashes (like out-of-memory errors on the server running the AI). The black vertical lines are "error bars" (95% Confidence Intervals) showing the range of uncertainty. If two error bars do not overlap much, the difference between them is statistically real, not a random fluke.
* **What the graph shows:**
  * Uncompressed (`F16/BF16`) has **5.03%** system errors (green bar).
  * Compressed variants (`Q8_0` and `Q4_K_M`) have lower rates (**3.61%** and **3.35%**, green bars).
* **The Takeaway:**
  This is counter-intuitive! The uncompressed model crashed *more* often. Why? Because uncompressed models are massive and put severe memory pressure on the servers, leading to system crashes. **Quantization actually makes systems more stable** by reducing memory demand.

---

## Part 3: Balancing Success and Energy Cost (RQ3)

### Figure 3(a): Quadrant Classification
* **What are the quadrants?**
  When we compress a model from `F16/BF16` to a quantized level, we compare the compressed version to its original self. We classify the transition into one of four categories:
  * **Win-win (Green):** The compressed model used less energy AND had a higher success rate.
  * **Trade-off (Yellow):** The compressed model saved energy, but success rate dropped.
  * **Inverse (Grey):** The compressed model used more energy, but got a higher success rate.
  * **Lose-lose (Red):** The compressed model used more energy AND performed worse.
* **What the graph shows:**
  * When compressing to **8-bit** (`F16/BF16 -> Q8_0`), **66.7%** of the transitions are **Win-win** (green).
  * When compressing to **4-bit** (`F16/BF16 -> Q4_K_M`), **50.0%** are **Trade-offs** (yellow) and **41.7%** are **Win-wins** (green).
* **The Takeaway:**
  Moving to 8-bit is almost always a "free lunch" (Win-win). Moving to 4-bit is a classic engineering trade-off: you save massive amounts of energy but might lose a bit of task success.

---

### Figure 3(b): Pareto Frontier
* **What is a "Pareto Frontier"?**
  Imagine shopping for a car. You want high speed (Success Rate) and low cost (Energy per Task). The "Pareto Frontier" (the grey dashed line) connects the optimal choices. These are models where **you cannot improve success rate without spending more energy**, or **you cannot reduce energy without hurting success rate**. 
  * Points below the dashed line are "dominated"—meaning there is another option that is either more successful, cheaper, or both.
* **What the graph shows:**
  * **Quantization Levels** are represented by both colors and shapes:
    * **F16/BF16**: Blue Circles (`o`)
    * **Q8_0**: Green Squares (`s`)
    * **Q4_K_M**: Orange Triangles (`^`)
  * **Pareto-optimal configurations** are highlighted with a **thick black outline** and sit directly on the dashed line:
    1. **Min3-3B-Q4_K_M** (Orange Triangle): Easiest on energy (2.61 kJ/task) but lower success (21.6%).
    2. **Min3-8B-Q4_K_M** (Orange Triangle): High success (29.6%) at very low energy (3.02 kJ/task).
    3. **Qwen-4B-Q4_K_M** (Orange Triangle): Very high success (42.7%) but uses more energy (17.57 kJ/task).
    4. **Qwen-4B-Q8** (Green Square): Highest success (43.5%) but uses the most energy (20.05 kJ/task).
  * Muted, borderless shapes represent sub-optimal (dominated) configurations.
* **The Takeaway:**
  This chart serves as a lookup guide. If you have an extremely tight energy budget, you should pick `Min3-8B-Q4` (highest success for under 5 kJ). If you only care about maximum success and don't care about energy bills, pick `Qwen-4B-Q8`. All other configurations are inefficient.

