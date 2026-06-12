# Evaluation Pipeline

Converts raw experiment logs in `new_outputs/` into structured metrics and figures.
Run from this directory or via the top-level replication instructions.

## Scripts

| Script | Role |
|---|---|
| `run_pipeline.py` | **Entry point.** Runs all steps in order. |
| `compute_metric.py` | Parses each `runs.jsonl` + `resource_metrics.json` → per-run metrics JSON |
| `aggregate_result.py` | Averages metrics across replicates → comparison tables + `all_runs_master.json` |
| `rq_plots.py` | Generates RQ summary figures (saved to `results/rq_plots/`) |
| `domain_plots.py` | Generates per-domain breakdown figures (saved to `results/domain_plots/`) |

## Usage

```bash
cd evaluation
python run_pipeline.py
```

Configuration is read from `configs/eval_config.yaml`:

```yaml
outputs_path: new_outputs     # where raw logs live
results_path: results         # where processed outputs are written
plots_path:   results/rq_plots
domain_plots_path: results/domain_plots
experiment:
  runs_to_average: 3          # replicates per configuration
```

## Output Files

After running, `results/` will contain:

```
results/
├── all_runs_master.json        # Primary dataset: one entry per run (216 total)
├── master_results.csv          # Same, CSV format
├── runs/                       # Per-run JSON files, organised by model/size/quant/domain
├── aggregated/                 # Mean ± SD tables by quantization, model, size, domain
├── rq_plots/                   # RQ figures and numeric summary tables
└── domain_plots/               # Per-domain breakdown figures
```

## Metric Reference

Each entry in `all_runs_master.json` has the following structure:

```json
{
  "model":  "qwen3",
  "size":   "4B",
  "quant":  "q4_k_m",
  "domain": "os",
  "run":    "run1",
  "metrics": {
    "total_tasks":      144,
    "successes":        54,
    "success_rate":     0.375,
    "failure_rate":     0.611,
    "completion_rate":  0.965,
    "error_rate":       0.035,
    "tle_rate":         0.125,
    "if_rate":          0.319,
    "ia_rate":          0.000,
    "task_error_rate":  0.007,
    "gpu_mem_peak":     8480.4,
    "gpu_power_avg":    250.4,
    "total_energy_joules": 2979558.1,
    "energy_per_task":  20691.4,
    "energy_per_success": 55176.0
  }
}
```

See `statistical_tests/results_glossary.md` for full field descriptions.
