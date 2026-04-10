from pathlib import Path
import yaml

import compute_metric
import aggregate_result
import generate_plots
import enhanced_plots

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config():

    config_file = PROJECT_ROOT / "configs" / "eval_config.yaml"

    with open(config_file) as f:
        return yaml.safe_load(f)


def run_pipeline():

    config = load_config()

    outputs_path = PROJECT_ROOT / config["outputs_path"]
    results_path = PROJECT_ROOT / config["results_path"]
    plots_path = PROJECT_ROOT / config["plots_path"]
    enhanced_plots_path = PROJECT_ROOT / config["enhanced_plots_path"]

    runs_to_average = config["experiment"]["runs_to_average"]

    print("Project root:", PROJECT_ROOT)

    if not outputs_path.exists():
        print(f"ERROR: outputs path not found: {outputs_path}")
        return

    results_path.mkdir(parents=True, exist_ok=True)

    print("\n1. Computing Metrics")
    compute_metric.run(outputs_path, results_path)

    print("\n2. Aggregating Runs")
    aggregate_result.run(results_path, runs_to_average)

    print("\n3. Generating Plots")
    generate_plots.run(results_path, plots_path)
    #enhanced_plots.run(results_path, enhanced_plots_path)
    print("\nPipeline completed.\n")

if __name__ == "__main__":
    run_pipeline()