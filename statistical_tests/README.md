# Statistical Tests

All statistical analyses for the paper. Each script corresponds to one research question or analysis type.

## Scripts

| Script | Analysis | Paper Section |
|---|---|---|
| `rq1_tests.py` | RQ1: effect of scale & quantization on success rate | §V-A |
| `rq2_tests.py` | RQ2: failure composition under quantization | §V-B |
| `rq3_tests.py` | RQ3: effectiveness-efficiency trade-off | §V-C |
| `rq3_4bit_vs_8bit.py` | RQ3 supplement: direct 4-bit vs 8-bit paired comparison (energy, VRAM, latency, throughput, success) | §V-C |
| `tost_equivalence.py` | Task-level paired TOST equivalence analysis | §V-A (TOST subsection) |
| `deployment_metrics.py` | Operational latency, throughput, GPU memory | §V-C (deployment table) |
| `behavioral_consistency.py` | Outcome flips (McNemar) + interaction turns | §V-A (discordance), §V-B (turns) |
| `run_all_tests.py` | Master runner — executes all analyses | — |
| `data_loader.py` | Shared data loading utilities | — |

## Usage

```bash
cd statistical_tests

# Run everything
python run_all_tests.py

# Individual analyses
python run_all_tests.py rq1
python run_all_tests.py rq2
python run_all_tests.py rq3
python run_all_tests.py rq3_48    # direct 4-bit vs 8-bit comparison
python run_all_tests.py tost
python run_all_tests.py deployment
python run_all_tests.py behavior
```

All scripts read from `results/all_runs_master.json`.
The three new scripts (`tost_equivalence.py`, `deployment_metrics.py`, `behavioral_consistency.py`) additionally read the raw logs in `new_outputs/`.

## Output Files

```
statistical_tests/output/
├── rq1_results.json             # Tests 1–5
├── rq2_results.json             # Tests 6–10
├── rq3_results.json             # Tests 11–15
├── rq3_4bit_vs_8bit_results.json  # Direct 4-bit vs 8-bit paired comparison
├── all_results_combined.json    # All tests merged
├── tost_results.json            # Task-level TOST (all cells)
├── tost_summary_table.csv       # Per-cell equivalence summary
├── fig_tost_equivalence.pdf     # Forest-plot figure
├── behavioral_consistency.json  # Pooled flip/McNemar + turns statistics
├── flip_rates_table.csv         # Per-cell flip decomposition
└── fig_behavioral_consistency.pdf  # Flip-rate panels

results/
├── deployment_metrics.csv       # Per-run latency/memory
├── deployment_metrics_agg.csv   # Aggregated by configuration
├── deployment_table.csv         # Publication-ready compact table
└── fig_deployment_metrics.pdf   # 3-panel bar chart
```

## Tests by RQ

### RQ1 — Scale & Quantization on Task Success

| # | Test | Purpose |
|---|---|---|
| 1 | Two-Way Factorial ANOVA | Main effects and interaction of size × quantization |
| 2 | Kruskal-Wallis | Non-parametric robustness check |
| 3 | Linear Mixed-Effects Model | Primary analysis with domain random effects |
| 4 | Spearman Rank Correlation | Scaling trend with parameter count |
| 5 | Cochran-Mantel-Haenszel | Stratified association across domains |

### RQ2 — Failure Composition

| # | Test | Purpose |
|---|---|---|
| 6 | Chi-Square Homogeneity | Are failure distributions the same across quantizations? |
| 7 | Multinomial Logistic Regression | Model failure type from quantization level |
| 8 | Friedman Test | Matched comparison across quant levels |
| 9 | CoDA via CLR + MANOVA | Compositional analysis of failure mix |
| 10 | Bootstrap CIs | Confidence intervals on failure proportions |

### RQ3 — Effectiveness-Efficiency Trade-off

| # | Test | Purpose |
|---|---|---|
| 11 | Pareto Efficiency Analysis | Identify dominated configurations |
| 12 | Two-Way MANOVA | Joint effect on success + energy |
| 13 | Spearman + Mixed Regression | Energy-success correlation |
| 14 | Efficiency Ratio Analysis | Composite metric comparison |
| 15 | Relative Change Analysis | Degradation metrics BF16→Q4 |

### Additional Analyses (new)

| Analysis | Method | Key result |
|---|---|---|
| TOST Equivalence | Two one-sided Wilcoxon signed-rank tests on 39,386 paired task outcomes | Q4_K_M equivalent to BF16 in 23/24 cells (±5 pp margin) |
| Deployment Metrics | Timestamps from `runs.jsonl` + telemetry from `resource_metrics.json` | Q4_K_M: −32% latency, +29% throughput, −41% peak GPU memory |
| Outcome Discordance | Exact McNemar test on 12,738 paired task outcomes | 9.55% of pairs flip under Q4_K_M, directionally balanced (597 regressions vs 619 improvements, p=0.547) |
| Interaction Turns | Assistant-turn counts from transcripts, matched-pair Wilcoxon over 24 cells | Turns invariant to precision (3.75/3.82/3.78 mean; p ≥ 0.06) |

## Interpreting Results

See `results_glossary.md` for a full explanation of every key in the JSON output files, including how each test statistic is computed and what values indicate significance.

## Data Loading

`data_loader.py` provides:

- `load_run_data()` → flat DataFrame with one row per (model, size, quant, domain, run)
- `get_failure_counts()` / `get_failure_rates()` → failure taxonomy DataFrames
- `get_matched_data(df, col)` → pivoted DataFrame for paired tests
- `save_results(dict, filename)` → JSON serialisation with numpy type handling
