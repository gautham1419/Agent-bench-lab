import json
import csv
import math
from pathlib import Path

from metrics.tradeoffs import compute_tradeoffs


# -------------------------------
# COUNT METRICS (not averaged)
# -------------------------------
COUNT_METRICS = [
    "total_tasks",
    "successes",
    "failures",
    "errors",

    # optional reliability counts
    "interaction_failures",
    "timeout_failures",
    "tool_format_failures"
]


# -------------------------------
# TARGET METRICS (averaged)
# -------------------------------
TARGET_METRICS = [
    # --- performance ---
    "success_rate",
    "mean_reward",
    "overall_failure_rate",
    "agent_failure_rate",
    "avg_tool_calls",
    "avg_turns",

    # --- reliability ---
    "interaction_failure_rate",
    "timeout_rate",
    "tool_format_rate",

    # --- system ---
    "crash_rate",

    # --- hardware ---
    "energy",
    "cpu",
    "ram",
    "gpu_util",
    "gpu_mem",
    "gpu_power"
]


# -------------------------------
# METRIC ALIASES (for compatibility)
# -------------------------------
METRIC_ALIASES = {
    # hardware
    "energy": "total_energy_joules",
    "cpu": "cpu_avg",
    "ram": "ram_avg",
    "gpu_util": "gpu_util_avg",
    "gpu_mem": "gpu_mem_avg",
    "gpu_power": "gpu_power_avg",

    # old naming compatibility
    "tool_format_rate": "tool_format_violation_rate"
}


# -------------------------------
# NORMALIZE METRICS
# -------------------------------
def normalize_metrics(metrics):

    normalized = {}

    for m in TARGET_METRICS:
        source = METRIC_ALIASES.get(m, m)

        if source in metrics:
            normalized[m] = metrics[source]

    return normalized


# -------------------------------
# MEAN + STD
# -------------------------------
def compute_mean_std(values):

    if not values:
        return 0, 0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)

    return mean, std


# -------------------------------
# AVERAGE RUN FILES
# -------------------------------
def average_metrics(files):

    metric_values = {m: [] for m in TARGET_METRICS}
    counts = {}

    for file in files:

        with open(file) as f:
            data = json.load(f)

        raw_metrics = data["metrics"]
        metrics = normalize_metrics(raw_metrics)

        # ---- averaged metrics ----
        for m in TARGET_METRICS:
            if m in metrics:
                metric_values[m].append(metrics[m])

        # ---- counts (take first run only) ----
        for c in COUNT_METRICS:
            if c in raw_metrics and c not in counts:
                counts[c] = raw_metrics[c]

    results = {}

    for m, values in metric_values.items():
        mean, std = compute_mean_std(values)
        results[f"{m}_mean"] = mean
        results[f"{m}_std"] = std

    # attach counts (no averaging)
    results.update(counts)

    return results, len(files)


# -------------------------------
# COLLECT RUN AVERAGES
# -------------------------------
def collect_run_averages(results_dir, runs_to_average):

    runs_dir = results_dir / "runs"
    master_records = []

    for domain_folder in runs_dir.rglob("*"):

        if not domain_folder.is_dir():
            continue

        run_files = sorted(
            f for f in domain_folder.glob("*.json")
            if f.name.startswith("run")
        )[:runs_to_average]

        if not run_files:
            continue

        print(f"Using {len(run_files)} runs: {[f.name for f in run_files]}")

        avg_metrics, num_runs = average_metrics(run_files)

        parts = domain_folder.relative_to(runs_dir).parts

        if len(parts) < 4:
            continue

        model, size, quant, domain = parts[:4]

        avg_data = {
            "metadata": {
                "model": model,
                "size": size,
                "quant": quant,
                "domain": domain,
                "num_runs": num_runs,
                "runs_used": [f.name for f in run_files]
            },
            "metrics": avg_metrics
        }

        avg_file = domain_folder / "summary.json"

        with open(avg_file, "w") as f:
            json.dump(avg_data, f, indent=2)

        print("Saved:", avg_file)

        record = {
            "model": model,
            "size": size,
            "quant": quant,
            "domain": domain
        }

        record.update(avg_metrics)
        master_records.append(record)

    return master_records


# -------------------------------
# AGGREGATE BY DIMENSION
# -------------------------------
def aggregate_dimension(master_records, key):

    results = {}

    for r in master_records:

        group = r[key]

        if group not in results:
            results[group] = {}

        for k, v in r.items():

            if k in ["model", "size", "quant", "domain"]:
                continue

            if not k.endswith("_mean"):
                continue

            results[group].setdefault(k, []).append(v)

    final = {}

    for group, metrics in results.items():

        final[group] = {}

        for m, values in metrics.items():

            mean, std = compute_mean_std(values)
            base = m.replace("_mean", "")

            final[group][f"{base}_mean"] = mean
            final[group][f"{base}_std"] = std

    return final


# -------------------------------
# SAVE HELPERS
# -------------------------------
def save_json(data, path):

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print("Saved:", path)


def save_csv(records, path):

    if not records:
        return

    keys = records[0].keys()

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

    print("Saved:", path)


# -------------------------------
# MAIN ENTRY
# -------------------------------
def run(results_dir, runs_to_average):

    agg_dir = results_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    master_records = collect_run_averages(results_dir, runs_to_average)

    if not master_records:
        print("No runs found.")
        return

    model_comp = aggregate_dimension(master_records, "model")
    size_comp = aggregate_dimension(master_records, "size")
    quant_comp = aggregate_dimension(master_records, "quant")
    domain_comp = aggregate_dimension(master_records, "domain")

    save_json(model_comp, agg_dir / "model_comparison.json")
    save_json(size_comp, agg_dir / "size_comparison.json")
    save_json(quant_comp, agg_dir / "quantization_comparison.json")
    save_json(domain_comp, agg_dir / "domain_comparison.json")

    save_json(master_records, results_dir / "master_results.json")
    save_csv(master_records, results_dir / "master_results.csv")

    # ---- tradeoffs ----
    tradeoffs = compute_tradeoffs(master_records)
    tradeoff_file = agg_dir / "tradeoff_metrics.json"

    with open(tradeoff_file, "w") as f:
        json.dump(tradeoffs, f, indent=2)

    print("Saved:", tradeoff_file)