# Processed Results

Pre-computed metrics and figures. Every file here can be regenerated from `new_outputs/` by running the evaluation pipeline — see the top-level README.

## Files

### Primary Dataset

| File | Format | Description |
|---|---|---|
| `all_runs_master.json` | JSON array | One entry per run (216 total). Primary input to all statistical tests. |
| `master_results.csv` | CSV | Same data in tabular form. |

### Deployment Metrics (new analyses)

| File | Description |
|---|---|
| `deployment_metrics.csv` | Per-run wall-clock time, throughput, GPU memory |
| `deployment_metrics_agg.csv` | Aggregated (mean across replicates) per configuration |
| `deployment_table.csv` | Publication-ready compact table (paper Table I) |
| `fig_deployment_metrics.pdf` | 3-panel bar chart (wall-clock / throughput / GPU memory) |

### Aggregated Comparison Tables (`aggregated/`)

| File | Groups by |
|---|---|
| `quantization_comparison.csv/json` | Quantization level |
| `model_comparison.csv/json` | Model family |
| `size_comparison.csv/json` | Parameter count |
| `domain_comparison.csv/json` | Task domain |

Each aggregated file reports `mean ± std` for all metrics across the grouped dimension.

### Per-Run JSON Files (`runs/`)

Organised as `runs/{model}/{size}/{quant}/{domain}/{runN}.json`.
Each file mirrors a single entry in `all_runs_master.json`.

### RQ Figures (`rq_plots/`)

| Path | Figure |
|---|---|
| `rq_plots/rq1/quant_vs_reward.png` | Success rate by quantization level |
| `rq_plots/rq1/success_decay.png` | Success rate decay across precision levels |
| `rq_plots/rq2/` | Failure composition plots |
| `rq_plots/rq3/` | Energy-success trade-off plots |
| `rq_plots/tables/rq1_quantization.csv` | Numeric summaries for RQ1 |
| `rq_plots/tables/rq2_failures.csv` | Numeric summaries for RQ2 |
| `rq_plots/tables/rq3_efficiency.csv` | Numeric summaries for RQ3 |

### Per-Domain Figures (`domain_plots/`)

Breakdown of all metrics for each of the four domains: `os/`, `dbbench/`, `webshop/`, `alfworld/`, and `overall/`.

## Key Numbers (cross-reference with paper)

All values below are taken directly from the pre-computed files and match the paper exactly.

| Metric | BF16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| Mean success rate | 18.8% | 19.1% | 19.3% |
| Median energy saving vs BF16 | — | −27.4% | −37.9% |
| Mean wall-clock / task | 55.4 s | 194.0 s* | 37.8 s |
| Effective throughput | 1.96 tasks/min | 2.27 | 2.52 |
| Peak GPU memory | 15,672 MiB | 11,214 MiB | 9,210 MiB |
| TOST equivalent cells (vs BF16) | — | 24/24 | 23/24 |

\* Q8_0 mean elevated by one anomalous Ministral3-8B run (909 s); median is consistent with the BF16→Q4 trend.
