"""
deployment_metrics.py
=====================
Operational deployment metrics extracted from existing telemetry.

What is available
-----------------
Each run folder contains:
  - resource_metrics.json   → gpu_mem_peak, gpu_mem_avg, gpu_power_avg,
                               total_energy_joules, cpu_peak …
  - runs.jsonl (per task)   → time.timestamp  (Unix ms, task completion time)
  - config.yaml             → folder name encodes experiment start time
                               (format: YYYY-MM-DD-HH-MM-SS)

What can be derived
-------------------
1. Run wall-clock time (seconds):
       folder_start_time (from folder name) → last task completion timestamp
   This is correct because tasks run in parallel (concurrency=32) and the
   run ends when the last task completes.

2. Effective throughput (tasks / minute):
       total_tasks / run_wall_clock_minutes
   "Effective" because concurrency=32 is constant across all runs, so this
   is a fair apples-to-apples comparison.

3. Peak GPU memory (MiB):
       gpu_mem_peak  from resource_metrics.json

4. Mean task wall-clock (seconds / task):
       run_wall_clock_s / total_tasks_completed
   This is the throughput-adjusted per-task time; not per-individual-task
   because concurrency is parallel.

What is NOT available
---------------------
- Token counts per task: the Ollama API response is captured via a
  `return_format` template that extracts only the text content, not usage
  metadata.  The raw `message.content` fields do not carry `prompt_tokens` /
  `completion_tokens`.  Tokens/s is therefore not computable.

Outputs
-------
- results/deployment_metrics.csv          (per-run metrics)
- results/deployment_metrics_agg.csv      (aggregated per configuration)
- results/deployment_table.csv            (publication-ready compact table)
- results/fig_deployment_metrics.pdf      (multi-panel bar chart)
"""

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "new_outputs"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QUANT_ORDER = ["bf16", "q8_0", "q4_k_m"]
QUANT_LABELS = {"bf16": "BF16/F16", "q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}
COLORS = {"bf16": "#4c72b0", "q8_0": "#dd8452", "q4_k_m": "#55a868"}


# ── metadata parsing ───────────────────────────────────────────────────────


def parse_folder_start(folder_name: str) -> datetime | None:
    """Extract datetime from folder name prefix YYYY-MM-DD-HH-MM-SS."""
    m = re.match(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", folder_name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        return None


def parse_run_metadata(folder_name: str) -> dict | None:
    """Parse model, size, quant, run from folder name."""
    parts = folder_name.split("_", 1)
    if len(parts) < 2:
        return None
    slug = parts[1].lstrip("-").strip()

    m = re.search(r"-(run\d+)$", slug)
    if not m:
        return None
    run_id = m.group(1)
    slug_no_run = slug[: m.start()]

    quant = None
    for q in ["q4_k_ms", "q4_k_m", "q8_0", "bf16", "f16"]:
        if slug_no_run.endswith(q):
            quant = q
            slug_no_run = slug_no_run[: -(len(q) + 1)]
            break
    if quant is None:
        return None
    if quant in ("f16", "bf16"):
        quant = "bf16"
    if quant == "q4_k_ms":
        quant = "q4_k_m"

    toks = slug_no_run.split("-")
    size = None
    for i in reversed(range(len(toks))):
        if re.match(r"^\d+(\.\d+)?[bB]$", toks[i], re.IGNORECASE):
            size = toks[i].upper()
            model_toks = toks[:i]
            break
    if size is None:
        return None

    model_str = "-".join(model_toks).lstrip("-").lower()
    if model_str.startswith("ollama-"):
        model_str = model_str[len("ollama-") :]
    if model_str.startswith("deepseek-r1-qwen"):
        model = "deepseek-r1-qwen"
    elif "ministral3" in model_str:
        model = "ministral3"
    elif "qwen3" in model_str:
        model = "qwen3"
    else:
        model = model_str

    return {"model": model, "size": size, "quant": quant, "run": run_id}


def get_domain(run_folder: Path) -> str | None:
    agent_dirs = [p for p in run_folder.iterdir() if p.is_dir()]
    if not agent_dirs:
        return None
    domain_dirs = [p for p in agent_dirs[0].iterdir() if p.is_dir()]
    if not domain_dirs:
        return None
    return domain_dirs[0].name.replace("-std", "")


# ── per-run metric extraction ──────────────────────────────────────────────


def extract_run_metrics(folder: Path) -> dict | None:
    """Return deployment metrics for a single run folder."""
    meta = parse_run_metadata(folder.name)
    if meta is None:
        return None

    folder_start_dt = parse_folder_start(folder.name)
    if folder_start_dt is None:
        return None

    domain = get_domain(folder)
    if domain is None:
        return None

    # resource_metrics.json
    rm_file = folder / "resource_metrics.json"
    if not rm_file.exists():
        return None
    with open(rm_file) as f:
        rm = json.load(f)

    # runs.jsonl – collect task completion timestamps
    agent_dirs = [p for p in folder.iterdir() if p.is_dir()]
    if not agent_dirs:
        return None
    domain_dir = agent_dirs[0] / f"{domain}-std"
    if not domain_dir.exists():
        return None
    runs_file = domain_dir / "runs.jsonl"
    if not runs_file.exists():
        return None

    timestamps_ms = []
    total_tasks = 0
    with open(runs_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_tasks += 1
            d = json.loads(line)
            t = d.get("time")
            if t and "timestamp" in t:
                timestamps_ms.append(t["timestamp"])

    if not timestamps_ms:
        return None

    # wall-clock: folder start → last task completion
    folder_start_ms = folder_start_dt.timestamp() * 1000
    last_ts_ms = max(timestamps_ms)
    wall_clock_s = (last_ts_ms - folder_start_ms) / 1000.0
    if wall_clock_s <= 0:
        wall_clock_s = (max(timestamps_ms) - min(timestamps_ms)) / 1000.0

    tasks_completed = len(timestamps_ms)
    throughput_tpm = (
        tasks_completed / (wall_clock_s / 60.0) if wall_clock_s > 0 else np.nan
    )
    mean_task_s = wall_clock_s / tasks_completed if tasks_completed > 0 else np.nan

    return {
        "model": meta["model"],
        "size": meta["size"],
        "quant": meta["quant"],
        "domain": domain,
        "run": meta["run"],
        "wall_clock_s": round(wall_clock_s, 1),
        "tasks_completed": tasks_completed,
        "throughput_tpm": round(throughput_tpm, 3),
        "mean_task_s": round(mean_task_s, 2),
        "gpu_mem_peak_mib": round(rm.get("gpu_mem_peak", np.nan), 1),
        "gpu_mem_avg_mib": round(rm.get("gpu_mem_avg", np.nan), 1),
        "gpu_power_avg_w": round(rm.get("gpu_power_avg", np.nan), 1),
        "total_energy_j": round(rm.get("total_energy_joules", np.nan), 1),
    }


# ── build dataset ──────────────────────────────────────────────────────────


def build_deployment_dataset() -> pd.DataFrame:
    rows = []
    for folder in OUTPUTS_DIR.iterdir():
        if not folder.is_dir():
            continue
        rec = extract_run_metrics(folder)
        if rec is not None:
            rows.append(rec)
    df = pd.DataFrame(rows)
    return df


# ── aggregate ─────────────────────────────────────────────────────────────


def aggregate_deployment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run metrics to (model, size, quant, domain) level."""
    agg = (
        df.groupby(["model", "size", "quant", "domain"])
        .agg(
            n_runs=("run", "count"),
            wall_clock_s=("wall_clock_s", "mean"),
            throughput_tpm=("throughput_tpm", "mean"),
            mean_task_s=("mean_task_s", "mean"),
            gpu_mem_peak_mib=("gpu_mem_peak_mib", "mean"),
            gpu_mem_avg_mib=("gpu_mem_avg_mib", "mean"),
            gpu_power_avg_w=("gpu_power_avg_w", "mean"),
            total_energy_j=("total_energy_j", "mean"),
        )
        .reset_index()
    )
    return agg


# ── publication table ─────────────────────────────────────────────────────


def build_deployment_table(agg: pd.DataFrame, master_json: Path) -> pd.DataFrame:
    """
    Merge deployment metrics with success_rate from all_runs_master.json
    to produce the compact publication-ready table.
    """
    # load success rates
    with open(master_json) as f:
        raw = json.load(f)
    sr_rows = []
    for e in raw:
        sr_rows.append(
            {
                "model": e["model"],
                "size": e["size"],
                "quant": e["quant"],
                "domain": e["domain"],
                "run": e["run"],
                "success_rate": e["metrics"]["success_rate"],
                "energy_per_task_j": e["metrics"].get("energy_per_task", np.nan),
            }
        )
    sr_df = pd.DataFrame(sr_rows)
    sr_agg = (
        sr_df.groupby(["model", "size", "quant", "domain"])
        .agg(
            success_rate=("success_rate", "mean"),
            energy_per_task=("energy_per_task_j", "mean"),
        )
        .reset_index()
    )

    merged = pd.merge(agg, sr_agg, on=["model", "size", "quant", "domain"], how="inner")

    # compute per-domain-averaged row per (model, size, quant)
    overall = (
        merged.groupby(["model", "size", "quant"])
        .agg(
            success_rate=("success_rate", "mean"),
            energy_per_task=("energy_per_task", "mean"),
            mean_task_s=("mean_task_s", "mean"),
            throughput_tpm=("throughput_tpm", "mean"),
            gpu_mem_peak_mib=("gpu_mem_peak_mib", "mean"),
        )
        .reset_index()
    )
    overall["domain"] = "ALL"

    table = pd.concat(
        [
            merged[
                [
                    "model",
                    "size",
                    "quant",
                    "domain",
                    "success_rate",
                    "energy_per_task",
                    "mean_task_s",
                    "throughput_tpm",
                    "gpu_mem_peak_mib",
                ]
            ],
            overall[
                [
                    "model",
                    "size",
                    "quant",
                    "domain",
                    "success_rate",
                    "energy_per_task",
                    "mean_task_s",
                    "throughput_tpm",
                    "gpu_mem_peak_mib",
                ]
            ],
        ],
        ignore_index=True,
    )

    # Prettify
    table["Configuration"] = (
        table["model"] + "-" + table["size"] + "-" + table["quant"].str.upper()
    )
    table["Domain"] = table["domain"].str.upper()
    table["Success (%)"] = (table["success_rate"] * 100).round(1)
    table["Energy/Task (kJ)"] = (table["energy_per_task"] / 1000).round(2)
    table["Wall-Clock/Task (s)"] = table["mean_task_s"].round(1)
    table["Throughput (tasks/min)"] = table["throughput_tpm"].round(2)
    table["Peak GPU Mem (MiB)"] = table["gpu_mem_peak_mib"].round(0).astype("Int64")

    out_cols = [
        "Configuration",
        "Domain",
        "Success (%)",
        "Energy/Task (kJ)",
        "Wall-Clock/Task (s)",
        "Throughput (tasks/min)",
        "Peak GPU Mem (MiB)",
    ]
    return table[out_cols].sort_values(["Configuration", "Domain"])


# ── figure ─────────────────────────────────────────────────────────────────


def plot_deployment_metrics(agg: pd.DataFrame):
    """
    3-panel bar chart: mean_task_s, throughput_tpm, gpu_mem_peak_mib
    grouped by (model, size) with bars for each quantization level.
    Averaged across domains.
    """
    overall = (
        agg.groupby(["model", "size", "quant"])
        .agg(
            mean_task_s=("mean_task_s", "mean"),
            throughput_tpm=("throughput_tpm", "mean"),
            gpu_mem_peak_mib=("gpu_mem_peak_mib", "mean"),
        )
        .reset_index()
    )
    overall["model_size"] = overall["model"] + "-" + overall["size"]

    configs = sorted(overall["model_size"].unique())
    quants = [q for q in QUANT_ORDER if q in overall["quant"].values]

    x = np.arange(len(configs))
    width = 0.25
    offsets = np.linspace(
        -(len(quants) - 1) * width / 2, (len(quants) - 1) * width / 2, len(quants)
    )

    metrics = [
        ("mean_task_s", "Mean Wall-Clock / Task (s)", "seconds"),
        ("throughput_tpm", "Effective Throughput (tasks/min)", "tasks/min"),
        ("gpu_mem_peak_mib", "Peak GPU Memory (MiB)", "MiB"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    for ax, (col, ylabel, unit) in zip(axes, metrics):
        for q, off in zip(quants, offsets):
            sub = overall[overall["quant"] == q].set_index("model_size")
            vals = [sub.loc[c, col] if c in sub.index else np.nan for c in configs]
            bars = ax.bar(
                x + off,
                vals,
                width=width,
                label=QUANT_LABELS[q],
                color=COLORS[q],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            # value labels on bar tops
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height()
                        + 0.01 * max(v for v in vals if not np.isnan(v)),
                        f"{v:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Operational Deployment Metrics by Configuration\n"
        "(averaged across domains and replicates)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()

    out = RESULTS_DIR / "fig_deployment_metrics.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Figure saved: {out}")
    plt.close(fig)


# ── entry point ────────────────────────────────────────────────────────────


def run_deployment_metrics():
    print("=" * 72)
    print("  Operational Deployment Metrics")
    print("=" * 72)

    print("\n  Extracting per-run metrics from raw logs …")
    df = build_deployment_dataset()
    print(f"  Runs extracted: {len(df)}")

    # save per-run
    per_run_path = RESULTS_DIR / "deployment_metrics.csv"
    df.to_csv(per_run_path, index=False)
    print(f"  Per-run metrics: {per_run_path}")

    print("\n  Aggregating …")
    agg = aggregate_deployment(df)
    agg_path = RESULTS_DIR / "deployment_metrics_agg.csv"
    agg.to_csv(agg_path, index=False)
    print(f"  Aggregated metrics: {agg_path}")

    print("\n  Building publication table …")
    master_json = ROOT / "results" / "all_runs_master.json"
    table = build_deployment_table(agg, master_json)
    table_path = RESULTS_DIR / "deployment_table.csv"
    table.to_csv(table_path, index=False)
    print(f"  Publication table: {table_path}")

    # Print overall summary
    overall = table[table["Domain"] == "ALL"].copy()
    print("\n" + "─" * 72)
    print("  OVERALL DEPLOYMENT SUMMARY (averaged across all domains)")
    print("─" * 72)
    print(
        overall[
            [
                "Configuration",
                "Success (%)",
                "Energy/Task (kJ)",
                "Wall-Clock/Task (s)",
                "Throughput (tasks/min)",
                "Peak GPU Mem (MiB)",
            ]
        ]
        .sort_values("Configuration")
        .to_string(index=False)
    )

    # Compute quantization changes for paper text
    print("\n" + "─" * 72)
    print("  LATENCY & MEMORY: BF16 vs Q4_K_M (across all model-domains)")
    print("─" * 72)
    agg_overall = (
        agg.groupby(["model", "size", "quant"])
        .agg(
            mean_task_s=("mean_task_s", "mean"),
            throughput_tpm=("throughput_tpm", "mean"),
            gpu_mem_peak_mib=("gpu_mem_peak_mib", "mean"),
        )
        .reset_index()
    )

    for q in ["bf16", "q8_0", "q4_k_m"]:
        sub = agg_overall[agg_overall["quant"] == q]
        print(
            f"  {q:8s}: "
            f"mean_task_s={sub['mean_task_s'].mean():.1f}s  "
            f"throughput={sub['throughput_tpm'].mean():.2f} tasks/min  "
            f"gpu_mem_peak={sub['gpu_mem_peak_mib'].mean():.0f} MiB"
        )

    print("\n  Generating figure …")
    plot_deployment_metrics(agg)

    return df, agg, table


if __name__ == "__main__":
    run_deployment_metrics()
