# Replication Package

This repository is the complete replication package for a research paper.
It contains all raw experiment logs, telemetry data, and analysis scripts.
Every number reported in the paper can be reproduced from the artefacts here without rerunning any model.

---

## Quick Navigation

| What you want | Where to go |
|---|---|
| Raw experiment logs | [`new_outputs/`](new_outputs/) |
| Processed results & figures | [`results/`](results/) |
| Statistical test scripts | [`statistical_tests/`](statistical_tests/) |
| Evaluation pipeline | [`evaluation/`](evaluation/) |
| Reproduce all analyses | [Reproduce Results](#reproduce-results) |
| Reproduce from scratch | [Re-run Experiments](#re-run-experiments) |

---

## Repository Structure

```
Agent-bench-lab/
│
├── new_outputs/               # Raw experiment logs (216 runs)
│   └── TIMESTAMP_model-run/   # One folder per run
│       ├── config.yaml            # Exact agent/task configuration used
│       ├── resource_metrics.json  # GPU/CPU energy & memory telemetry
│       └── agent/domain-std/
│           ├── runs.jsonl         # Per-task outcomes + timestamps
│           └── error.jsonl        # System-level errors
│
├── results/                   # Processed datasets and figures
│   ├── all_runs_master.json       # Run-level metrics (primary input to stats)
│   ├── master_results.csv         # Same, CSV format
│   ├── deployment_metrics.csv     # Per-run latency/memory metrics (new)
│   ├── deployment_metrics_agg.csv # Aggregated deployment metrics
│   ├── deployment_table.csv       # Publication-ready deployment table
│   ├── fig_deployment_metrics.pdf # Deployment metrics figure (new)
│   ├── runs/                      # Per-run JSON files by model/size/quant/domain
│   ├── aggregated/                # Comparison tables by dimension
│   ├── rq_plots/                  # RQ figures and summary tables
│   └── domain_plots/              # Per-domain breakdown figures
│
├── statistical_tests/         # All statistical analyses
│   ├── data_loader.py             # Shared data loading utilities
│   ├── rq1_tests.py               # RQ1: scale & quantization on success
│   ├── rq2_tests.py               # RQ2: failure composition
│   ├── rq3_tests.py               # RQ3: effectiveness-efficiency trade-off
│   ├── tost_equivalence.py        # TOST task-level equivalence analysis (new)
│   ├── deployment_metrics.py      # Operational deployment metrics (new)
│   ├── run_all_tests.py           # Master runner for all analyses
│   ├── results_glossary.md        # Guide to interpreting all result keys
│   └── output/                    # Statistical test outputs (JSON + figures)
│
├── evaluation/                # Metric computation and aggregation pipeline
│   ├── run_pipeline.py            # One-command pipeline runner
│   ├── compute_metric.py          # Parses raw logs → per-run metrics
│   ├── aggregate_result.py        # Aggregates runs → comparison tables
│   ├── rq_plots.py                # Generates RQ figures
│   └── domain_plots.py            # Generates per-domain figures
│
├── src/                       # AgentBench agent & task client code
├── configs/                   # Experiment configuration files
│   ├── eval_config.yaml           # Paths and settings for evaluation pipeline
│   ├── agents/                    # Agent (LLM) definitions
│   ├── tasks/                     # Task environment definitions
│   └── assignments/               # Per-run assignment configs (one per run)
├── data/                      # Task data for each AgentBench domain
├── extra/                     # Docker Compose and container entrypoints
└── requirements.txt           # Python dependencies
```

---

## Experimental Design

| Factor | Levels |
|---|---|
| **Model families** | Qwen3 · Ministral3 · DeepSeek-R1-Distill-Qwen |
| **Parameter scales** | 1.5 B · 3 B · 4 B · 7 B · 8 B |
| **Quantization levels** | BF16/F16 (baseline) · Q8_0 · Q4_K_M |
| **Domains** | OS Interaction · Database (DB) · WebShop · ALFWorld |
| **Replicates** | 3 independent runs per configuration |
| **Total runs** | 216 |
| **Individual task outcomes** | 39,386 |

All models were served locally via **Ollama** using **GGUF** weights on a single **NVIDIA RTX 4090** with no concurrent workloads.

---

## Reproduce Results

All analyses run entirely from the pre-collected logs in `new_outputs/` — no model re-execution required.

### Prerequisites

```bash
# Python 3.10+ recommended
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install pandas matplotlib statsmodels pingouin
```

### Step 1 — Recompute metrics from raw logs

```bash
cd evaluation
python run_pipeline.py
```

This reads every `runs.jsonl` and `resource_metrics.json` in `new_outputs/`, recomputes all metrics, and writes:
- `results/all_runs_master.json` — primary dataset
- `results/runs/` — per-run JSON files
- `results/aggregated/` — dimension comparison tables
- `results/rq_plots/` — RQ figures
- `results/domain_plots/` — per-domain figures

> The pre-computed outputs already exist in `results/` and match the paper exactly.
> Re-running `run_pipeline.py` will overwrite them with identical values.

### Step 2 — Run statistical tests

```bash
cd statistical_tests

# Run all tests (RQ1 + RQ2 + RQ3 + TOST + deployment metrics)
python run_all_tests.py

# Or run individual analyses:
python run_all_tests.py rq1         # RQ1: scale & quantization on success
python run_all_tests.py rq2         # RQ2: failure composition
python run_all_tests.py rq3         # RQ3: effectiveness-efficiency trade-off
python run_all_tests.py tost        # TOST task-level equivalence analysis
python run_all_tests.py deployment  # Operational deployment metrics
```

Outputs go to `statistical_tests/output/` and `results/`.

### Step 3 — Check key outputs

| File | Contents |
|---|---|
| `results/all_runs_master.json` | Run-level metrics for all 216 runs |
| `statistical_tests/output/rq1_results.json` | Tests 1–5 (ANOVA, KW, LMM, Spearman, CMH) |
| `statistical_tests/output/rq2_results.json` | Tests 6–10 (Chi-square, MLR, Friedman, CoDA, Bootstrap) |
| `statistical_tests/output/rq3_results.json` | Tests 11–15 (Pareto, MANOVA, Spearman, Efficiency ratio, Δ) |
| `statistical_tests/output/tost_results.json` | Task-level TOST equivalence results |
| `statistical_tests/output/tost_summary_table.csv` | Per-cell TOST summary (paper Table) |
| `statistical_tests/output/fig_tost_equivalence.pdf` | Forest-plot figure |
| `results/deployment_metrics_agg.csv` | Aggregated latency / throughput / memory |
| `results/deployment_table.csv` | Publication-ready deployment table |
| `results/fig_deployment_metrics.pdf` | Deployment metrics bar chart |

---

## Re-run Experiments

> **Warning:** Full re-execution requires an NVIDIA GPU with ≥24 GB VRAM, Docker, and Ollama installed. Total compute time was approximately **3 weeks** on an RTX 4090.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pull models via Ollama

```bash
# Example: pull Qwen3-4B at all three quantization levels
ollama pull hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M
ollama pull hf.co/unsloth/Qwen3-4B-GGUF:Q8_0
ollama pull hf.co/unsloth/Qwen3-4B-GGUF:F16

# See configs/agents/ for the exact model tags used for each configuration
```

### 3. Start task environments (Docker)

Run in a dedicated terminal and keep it running throughout the experiment:

```bash
# OS Interaction
docker compose -f extra/docker-compose.yml up redis controller os_interaction-std

# Database
docker compose -f extra/docker-compose.yml up redis controller dbbench-std

# ALFWorld
docker compose -f extra/docker-compose.yml up redis controller alfworld-std

# WebShop
docker compose -f extra/docker-compose.yml up redis controller webshop-std
```

### 4. Run an agent

```bash
# Edit configs/assignments/os_only.yaml to uncomment the desired model
python -m src.assigner --config configs/assignments/os_only.yaml
```

Output is written to `new_outputs/{TIMESTAMP}_{agent}-run{N}/`.

**Resource monitoring** must be started separately before each run:

```bash
python -m src.utils.monitor --output new_outputs/{TIMESTAMP}_{agent}-run{N}/resource_metrics.json
```

### 5. Configuration files

Each run in `new_outputs/` contains a `config.yaml` that records the exact agent and task configuration. Use these to reproduce any individual run precisely.

---

## Raw Log Format

### `runs.jsonl` — Per-task outcomes

Each line is a JSON object for one completed task:

```json
{
  "index":  42,
  "error":  null,
  "output": {
    "index":  42,
    "status": "completed",
    "result": {
      "reward":   1,
      "status":   "completed",
      "messages": [ ... ],
      "metrics":  { "score": 1 }
    }
  },
  "time": { "timestamp": 1773121938273, "str": "2026-03-10 11:22:18" }
}
```

| Field | Meaning |
|---|---|
| `index` | Integer task ID (0-based, fixed across all runs of the same domain) |
| `output.result.reward` | Task score (1 = success, 0 = failure) |
| `output.status` | Terminal status: `completed` · `task limit reached` · `task error` · `agent invalid action` |
| `time.timestamp` | Unix millisecond timestamp of task completion |

### `error.jsonl` — System-level errors

Tasks that could not be started or interacted with (e.g., Docker container crash). These count toward the denominator of success rate but not toward `runs.jsonl`.

### `resource_metrics.json` — Hardware telemetry

```json
{
  "gpu_mem_peak":      8480.4,
  "gpu_power_avg":     250.4,
  "gpu_energy_joules": 2556350.98,
  "cpu_energy_joules": 423207.14,
  "total_energy_joules": 2979558.13
}
```

Sampled at 1-second intervals using `pynvml` (GPU) and `pyRAPL` (CPU) during each run.

---

## What the Analyses Measure

### RQ1 — Effect of scale and quantization on task success
Tests: Two-Way ANOVA · Kruskal-Wallis · Linear Mixed-Effects Model · Spearman correlation · Cochran-Mantel-Haenszel

**Key result:** No significant effect of quantization on success rate (KW *H* = 0.063, *p* = 0.969). Task-level TOST confirms positive equivalence within ±5 pp in 23/24 model-domain cells.

### RQ2 — Failure composition under quantization
Tests: Chi-square homogeneity · Multinomial logistic regression · Friedman · CoDA/CLR-MANOVA · Bootstrap CIs

**Key result:** Cognitive failure types are invariant to quantization (Friedman *p* > 0.10 for all five types). System errors *decrease* at lower precision.

### RQ3 — Effectiveness-efficiency trade-off
Tests: Pareto efficiency · MANOVA · Spearman correlation · Efficiency ratio · Relative change analysis

**Key result:** Q4_K_M saves 37.9% median energy (*r* = 0.791) with no measurable loss in success. All globally Pareto-optimal configurations are quantized.

### TOST — Task-level paired equivalence
Method: Two one-sided Wilcoxon signed-rank tests on 39,386 paired binary task outcomes. Equivalence margin ±5 pp.

**Key result:** Q4_K_M is statistically equivalent to BF16 in 23/24 (95.8%) model-domain cells.

### Deployment metrics — Latency, throughput, GPU memory
Derived from `runs.jsonl` timestamps and `resource_metrics.json` without rerunning experiments.

**Key result:** Q4_K_M reduces mean task wall-clock time by 32%, improves throughput by 29%, and cuts peak GPU memory by 41% vs BF16.

---

## License

[MIT](LICENSE)



 













































































































































































