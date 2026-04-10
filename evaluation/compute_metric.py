import json
import yaml
from pathlib import Path

# import metric functions
from metrics.task_performance import compute_performance
from metrics.reliability import compute_reliability
from metrics.efficiency import compute_efficiency


def parse_metadata(config_file):

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    agent = cfg["assignments"][0]["agent"]
    task = cfg["assignments"][0]["task"]

    domain = task.split("-")[0]
    parts = agent.split("-")

    model = parts[1]
    size = parts[2].upper()
    quant = parts[-1]
    if quant == "f16":
        quant = "bf16"

    return model, size, quant, domain


def compute_metrics(runs_file, error_file, resource_file):

    metrics = {}

    # handle missing error file safely
    error_file_safe = error_file if error_file.exists() else None

    if error_file_safe is None:
        print(f"No error.jsonl found for {runs_file}, assuming 0 errors")

    # -------------------------------
    # TASK PERFORMANCE
    # -------------------------------
    if runs_file.exists():
        metrics.update(compute_performance(runs_file, error_file_safe))

    # -------------------------------
    # RELIABILITY
    # -------------------------------
    if runs_file.exists() and error_file.exists():
        metrics.update(compute_reliability(error_file, runs_file))
    else:
        # default values when no error file
        metrics.update({
            "error_count": 0,
            "failure_rate": 0.0
        })

    # -------------------------------
    # EFFICIENCY
    # -------------------------------
    if resource_file.exists():
        metrics.update(compute_efficiency(resource_file))
    else:
        print(f"No resource_metrics.json found for {runs_file}")

    return metrics


def run(outputs_dir, results_dir):

    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for run_folder in outputs_dir.iterdir():

        if not run_folder.is_dir():
            continue

        config_file = run_folder / "config.yaml"
        if not config_file.exists():
            continue

        model, size, quant, domain = parse_metadata(config_file)

        run_id = run_folder.name.split("-")[-1]

        # find agent folder safely
        agent_folders = [p for p in run_folder.iterdir() if p.is_dir()]
        if not agent_folders:
            print(f"No agent folder found in {run_folder}")
            continue

        agent_folder = agent_folders[0]

        domain_folder = agent_folder / f"{domain}-std"

        runs_file = domain_folder / "runs.jsonl"
        error_file = domain_folder / "error.jsonl"
        resource_file = run_folder / "resource_metrics.json"

        if not runs_file.exists():
            print(f"Skipping run (missing runs.jsonl): {run_folder}")
            continue

        metrics = compute_metrics(runs_file, error_file, resource_file)

        output_dir = runs_dir / model / size / quant / domain
        output_dir.mkdir(parents=True, exist_ok=True)

        result_file = output_dir / f"{run_id}.json"

        result = {
            "model": model,
            "size": size,
            "quant": quant,
            "domain": domain,
            "run": run_id,
            "metrics": metrics
        }

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        print("Saved:", result_file)