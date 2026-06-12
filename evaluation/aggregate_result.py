import json
import csv
import math
from pathlib import Path

# Take count as such for this metric
COUNT_METRICS = [
    "total_tasks"
]

# Take Mean and Std for these metrics
MEAN_STD_METRICS = [
    # RQ 1
    "success_rate",
    "failure_rate",
    "completion_rate",
    "error_rate",
    "mean_reward",
    "avg_tool_calls",
    "avg_turns",

    # RQ 2
    "tle_rate",
    "if_rate",
    "ia_rate",
    "task_error_rate",
    "completed_failure_rate",

    # RQ 3
    "energy",
    "cpu",
    "ram",
    "gpu_util",
    "gpu_mem",
    "gpu_power",
    "gpu_energy",
    "cpu_energy",
    "energy_per_task",
    "energy_per_success",
    "energy_per_action",
]

# Take mean only for these
MEAN_ONLY_METRICS = [
    "runs_completed",
    "successes",
    "failures",
    "errors",
    "tle_failures",
    "if_failures",
    "ia_failures",
    "task_errors",
    "completed_failures",
    "agent_failed",
    "interact_failed",
    "start_failed",
]

# Metrics shown in dimension comparison files
COMPARISON_METRICS = [
    "success_rate",
    "failure_rate",
    "error_rate",
    "energy_per_task",
]

# METRIC ALIASES (for improved naming)
METRIC_ALIASES = {
    "energy": "total_energy_joules",
    "cpu": "cpu_avg",
    "ram": "ram_avg",
    "gpu_util": "gpu_util_avg",
    "gpu_mem": "gpu_mem_avg",
    "gpu_power": "gpu_power_avg",
    "gpu_energy": "gpu_energy_joules",
    "cpu_energy": "cpu_energy_joules"
}

# Normalize naming
def normalize_metrics(raw_metrics):

    normalized = {}
    all_metrics = MEAN_STD_METRICS + MEAN_ONLY_METRICS

    for m in all_metrics:
        source = METRIC_ALIASES.get(m, m)

        if source in raw_metrics:
            normalized[m] = raw_metrics[source]

    return normalized

# Compute Mean and Std
def compute_mean_std(values):

    valid_values = [x for x in values if isinstance(x, (int, float)) and not math.isnan(x)]

    if not valid_values:
        return 0.0, 0.0

    mean = sum(valid_values) / len(valid_values)
    variance = sum((x - mean) ** 2 for x in valid_values) / len(valid_values)
    std = math.sqrt(variance)

    return round(mean, 6), round(std, 6)

# Average metrics of the run files  
def average_metrics(files):

    mean_std_values = {m: [] for m in MEAN_STD_METRICS}
    mean_only_values = {m: [] for m in MEAN_ONLY_METRICS}
    counts = {}

    for file in files:

        with open(file) as f:
            data = json.load(f)

        raw_metrics = data["metrics"]
        metrics = normalize_metrics(raw_metrics)

        for m in MEAN_STD_METRICS:
            if m in metrics:
                mean_std_values[m].append(metrics[m])

        for m in MEAN_ONLY_METRICS:
            if m in metrics:
                mean_only_values[m].append(metrics[m])

        # take total task for a certain domain
        for c in COUNT_METRICS:
            if c in raw_metrics and c not in counts:
                counts[c] = raw_metrics[c]

    results = {}

    # mean + std
    for m, values in mean_std_values.items():
        mean, std = compute_mean_std(values)
        results[f"{m}_mean"] = mean
        results[f"{m}_std"] = std

    # mean only
    for m, values in mean_only_values.items():
        mean, _ = compute_mean_std(values)
        results[f"{m}_mean"] = mean

    # attach counts (no averaging)
    results.update(counts)

    return results, len(files)

# COLLECT RUN AVERAGES
def collect_run_averages(results_dir, runs_to_average):

    runs_dir = results_dir / "runs"
    master_records = []

    # Target only domain subdirectories containing run result JSON files efficiently
    domain_folders = sorted(list(set(f.parent for f in runs_dir.rglob("run*.json"))))

    import re
    def run_num_key(f):
        match = re.search(r"run(\d+)", f.name)
        return int(match.group(1)) if match else 0

    for domain_folder in domain_folders:

        run_files = sorted(
            domain_folder.glob("run*.json"),
            key=run_num_key
        )[:runs_to_average]

        if not run_files:
            continue

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
        domain = r["domain"]

        if group not in results:
            results[group] = {}

        if domain not in results[group]:
            results[group][domain] = {m: [] for m in COMPARISON_METRICS}

        for m in COMPARISON_METRICS:
            src = f"{m}_mean"
            if src in r:
                results[group][domain][m].append(r[src])

    final = {}

    for group, domains in results.items():

        final[group] = {}
        domain_means = {m: [] for m in COMPARISON_METRICS}

        for domain, metric_lists in domains.items():

            domain_entry = {}

            for m, values in metric_lists.items():
                if values:
                    mean = round(sum(values) / len(values), 6)
                    domain_entry[m] = mean
                    domain_means[m].append(mean)

            final[group][domain] = domain_entry

        # overall_mean is the macro-average of the per-domain means
        overall = {}
        for m, vals in domain_means.items():
            if vals:
                overall[m] = round(sum(vals) / len(vals), 6)

        final[group]["overall_mean"] = overall

    return final


# Domain-level aggregation structured as domain -> metric -> full model means and overall_mean
def aggregate_by_domain(master_records):

    result = {}

    for r in master_records:

        domain = r["domain"]
        model = r["model"]
        size = r["size"]
        quant = r["quant"]
        full_model = f"{model}-{size}-{quant}"

        if domain not in result:
            result[domain] = {m: {} for m in COMPARISON_METRICS}

        for m in COMPARISON_METRICS:
            src = f"{m}_mean"
            if src in r and r[src] is not None:
                result[domain][m][full_model] = r[src]

    # Compute the overall_mean across all models for each domain and metric
    for domain, metrics in result.items():
        for m, model_values in metrics.items():
            if model_values:
                vals = list(model_values.values())
                overall_mean = round(sum(vals) / len(vals), 6)
                model_values["overall_mean"] = overall_mean

    return result


def save_json(data, path):

    path.parent.mkdir(parents=True, exist_ok=True)

    def sanitize(obj):
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(sanitize(data), f, indent=2)

    print("Saved:", path)


def save_csv(records, path):

    if not records:
        return

    # Find the union of all keys across all records to prevent ValueErrors
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    # Sort keys to maintain a consistent column order, putting metadata fields first
    metadata_keys = ["model", "size", "quant", "domain"]
    other_keys = sorted(list(all_keys - set(metadata_keys)))
    keys = [k for k in metadata_keys if k in all_keys] + other_keys

    def sanitize(v):
        # Replace inf/nan with empty string so CSV cells are properly blank
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            return ""
        return v

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: sanitize(r.get(k, "")) for k in keys})

    print("Saved:", path)


def save_comparison_csv(data, key_name, path):

    records = []
    for group, domains in data.items():
        for domain, metrics in domains.items():
            row = {
                key_name: group,
                "domain": domain
            }
            row.update(metrics)
            records.append(row)

    if not records:
        return

    keys = [key_name, "domain"] + COMPARISON_METRICS

    def sanitize(v):
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            return ""
        return v

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: sanitize(r.get(k, "")) for k in keys})

    print("Saved:", path)


def save_domain_comparison_csv(data, path):

    # Gather all unique full model columns
    all_models = set()
    for domain, metrics in data.items():
        for metric, model_values in metrics.items():
            for model in model_values.keys():
                if model != "overall_mean":
                    all_models.add(model)
    sorted_models = sorted(list(all_models))

    records = []
    for domain, metrics in data.items():
        for metric, model_values in metrics.items():
            row = {
                "domain": domain,
                "metric": metric
            }
            for model in sorted_models:
                row[model] = model_values.get(model, "")
            row["overall_mean"] = model_values.get("overall_mean", "")
            records.append(row)

    if not records:
        return

    keys = ["domain", "metric"] + sorted_models + ["overall_mean"]

    def sanitize(v):
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            return ""
        return v

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: sanitize(r.get(k, "")) for k in keys})

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
    domain_comp = aggregate_by_domain(master_records)

    save_json(model_comp, agg_dir / "model_comparison.json")
    save_comparison_csv(model_comp, "model", agg_dir / "model_comparison.csv")

    save_json(size_comp, agg_dir / "size_comparison.json")
    save_comparison_csv(size_comp, "size", agg_dir / "size_comparison.csv")

    save_json(quant_comp, agg_dir / "quantization_comparison.json")
    save_comparison_csv(quant_comp, "quant", agg_dir / "quantization_comparison.csv")

    save_json(domain_comp, agg_dir / "domain_comparison.json")
    save_domain_comparison_csv(domain_comp, agg_dir / "domain_comparison.csv")

    save_json(master_records, results_dir / "master_results.json")
    save_csv(master_records, results_dir / "master_results.csv")