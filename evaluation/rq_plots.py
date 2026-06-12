import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------
# Global Style
# ------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.2
)

FIGSIZE = (8, 5)


# ------------------------------------------------
# Label Standardization
# ------------------------------------------------

QUANT_LABELS = {
    "bf16": "16-bit",
    "q8_0": "8-bit",
    "q4_k_m": "4-bit",
}

QUANT_ORDER = ["16-bit", "8-bit", "4-bit"]

MODEL_LABELS = {
    "ministral3": "Ministral3",
    "qwen3": "Qwen3",
    "deepseek-r1-qwen": "DeepSeek R1 Qwen",
}

MODEL_COLORS = {
    "Ministral3 3B": "#9575CD",
    "Ministral3 8B": "#512DA8",
    "Qwen3 4B": "#FFB74D",
    "Qwen3 8B": "#E65100",
    "DeepSeek R1 Qwen 1.5B": "#4DB6AC",
    "DeepSeek R1 Qwen 7B": "#004D40",
}

PALETTE = sns.color_palette("colorblind")


def standardize_labels(df):

    df = df.copy()

    if "quant" in df.columns:
        df["quant"] = df["quant"].replace(QUANT_LABELS)
        df["quant"] = pd.Categorical(df["quant"], categories=QUANT_ORDER, ordered=True)

    if "model" in df.columns:
        df["model_base"] = df["model"].replace(MODEL_LABELS)
        df["model_label"] = df["model_base"] + " " + df["size"]

    return df


# ------------------------------------------------
# Utilities
# ------------------------------------------------

def save_plot(fig, path):

    path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()

    fig.savefig(path.with_suffix(".png"), dpi=300)

    plt.close(fig)


def place_legend(ax, title):

    ax.legend(
        title=title,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0
    )


# ------------------------------------------------
# RQ1 – Quantization vs Performance
# ------------------------------------------------

def plot_rq1_success_decay(df, plots_path):
    """Line plot: success rate decay across quantization levels per model."""

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.lineplot(
        data=df,
        x="quant",
        y="success_rate_mean",
        hue="model_label",
        marker="o",
        linewidth=2,
        palette=MODEL_COLORS,
        ax=ax
    )

    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Success Rate")
    ax.set_title("RQ1: Task Success Decay vs Quantization")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    place_legend(ax, "Model")

    save_plot(fig, plots_path / "rq1/success_decay")


def plot_rq1_wandering(df, plots_path):
    """Bar plot: average turns to success across quantization levels."""

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(
        data=df,
        x="quant",
        y="avg_turns_mean",
        hue="model_label",
        palette=MODEL_COLORS,
        ax=ax
    )

    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Average Conversational Turns")
    ax.set_title("RQ1: Wandering (Avg Turns Before Success)")

    place_legend(ax, "Model")

    save_plot(fig, plots_path / "rq1/wandering_turns")


def plot_quant_vs_reward(df, plots_path):
    """Line plot: reward across quantization levels per model."""

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.lineplot(
        data=df,
        x="quant",
        y="mean_reward_mean",
        hue="model_label",
        marker="o",
        linewidth=2,
        palette=MODEL_COLORS,
        ax=ax
    )

    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Quantization vs Reward")

    place_legend(ax, "Model")

    save_plot(fig, plots_path / "rq1/quant_vs_reward")


def plot_tool_calls(df, plots_path):
    """Bar plot: tool calls across quantization levels."""

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(
        data=df,
        x="quant",
        y="avg_tool_calls_mean",
        hue="model_label",
        palette=MODEL_COLORS,
        errorbar=None,
        ax=ax
    )

    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Avg Tool Calls")
    ax.set_title("Quantization vs Tool Usage")

    place_legend(ax, "Model")

    save_plot(fig, plots_path / "rq1/tool_calls")


# ------------------------------------------------
# RQ2 – Failure Analysis
# ------------------------------------------------

def plot_rq2_failure_anatomy(df, plots_path):
    """Stacked bar chart: failure type composition per model-quantization config."""

    # Use the failure taxonomy columns from compute_metric.py
    failure_cols = {
        "tle_rate_mean": "TLE (Task Limit Exceeded)",
        "ia_rate_mean": "IA (Invalid Action)",
        "if_rate_mean": "IF (Invalid Format)",
        "task_error_rate_mean": "TE (Task Error)",
        "completed_failure_rate_mean": "CF (Completed Failure)",
    }

    available_cols = [c for c in failure_cols.keys() if c in df.columns]

    if not available_cols:
        print("  [SKIP] No failure rate columns found for RQ2 failure anatomy plot.")
        return

    failure_df = df[["model_label", "quant"] + available_cols].copy()

    failure_df = failure_df.melt(
        id_vars=["model_label", "quant"],
        var_name="Failure Type",
        value_name="Rate"
    )

    failure_df["Failure Type"] = failure_df["Failure Type"].replace(failure_cols)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(
        data=failure_df,
        x="quant",
        y="Rate",
        hue="Failure Type",
        palette=PALETTE,
        errorbar=None,
        ax=ax
    )

    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Rate")
    ax.set_title("RQ2: Failure Breakdown Across Quantization")

    place_legend(ax, "Failure Type")

    save_plot(fig, plots_path / "rq2/failure_rates")


def plot_failure_heatmap(df, plots_path):
    """Heatmap: failure rates and success rate by model and quantization."""

    heatmap_cols = []
    for col in ["tle_rate_mean", "if_rate_mean", "ia_rate_mean", "success_rate_mean"]:
        if col in df.columns:
            heatmap_cols.append(col)

    if not heatmap_cols:
        print("  [SKIP] No columns available for failure heatmap.")
        return

    heatmap_data = df.pivot_table(
        values=heatmap_cols,
        index="model_label",
        columns="quant"
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="viridis",
        fmt=".3f",
        linewidths=0.5,
        ax=ax
    )

    ax.set_xlabel("Quantization")
    ax.set_ylabel("Model")
    ax.set_title("Failure Pattern Heatmap")

    save_plot(fig, plots_path / "rq2/failure_heatmap")


# ------------------------------------------------
# RQ3 – Performance vs Efficiency
# ------------------------------------------------

def plot_rq3_pareto(df, plots_path):
    """Scatter plot: energy per task vs success rate (Pareto frontier)."""

    if "energy_per_task_mean" not in df.columns:
        print("  [SKIP] energy_per_task_mean not found for RQ3 Pareto plot.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.scatterplot(
        data=df,
        x="energy_per_task_mean",
        y="success_rate_mean",
        hue="model_label",
        style="quant",
        palette=MODEL_COLORS,
        s=150,
        alpha=0.8,
        ax=ax
    )

    ax.set_xlabel("Energy per Task (J)")
    ax.set_ylabel("Success Rate")
    ax.set_title("RQ3: Performance vs Energy (Pareto Frontier)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    place_legend(ax, "Model / Quantization")

    save_plot(fig, plots_path / "rq3/pareto_frontier")


def plot_rq3_true_cost(df, plots_path):
    """Bar plot: energy per success across quantization levels (log scale)."""

    if "energy_per_success_mean" not in df.columns:
        print("  [SKIP] energy_per_success_mean not found for RQ3 true cost plot.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(
        data=df,
        x="quant",
        y="energy_per_success_mean",
        hue="model_label",
        palette=MODEL_COLORS,
        ax=ax
    )

    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Energy (Joules) per Success")
    ax.set_title("RQ3: True Cost of a Successful Task")
    ax.set_yscale("log")

    place_legend(ax, "Model")

    save_plot(fig, plots_path / "rq3/true_energy_cost")


def plot_rq3_hardware_ceiling(df, plots_path):
    """Bar plot: peak GPU memory across quantization levels."""

    if "gpu_mem_mean" not in df.columns:
        print("  [SKIP] gpu_mem_mean not found for RQ3 hardware ceiling plot.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(
        data=df,
        x="quant",
        y="gpu_mem_mean",
        hue="model_label",
        palette=MODEL_COLORS,
        ax=ax
    )

    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("GPU Memory Usage (MB)")
    ax.set_title("RQ3: Hardware Barrier to Entry (GPU VRAM)")

    place_legend(ax, "Model")

    save_plot(fig, plots_path / "rq3/hardware_vram")


def plot_rq3_energy_composition(df, plots_path):
    """Stacked bar: GPU vs CPU energy per task per configuration."""

    if "gpu_energy_mean" not in df.columns or "cpu_energy_mean" not in df.columns:
        print("  [SKIP] GPU/CPU energy columns not found for energy composition plot.")
        return

    # Compute per-task energy breakdown
    composition = df[["model_label", "quant", "gpu_energy_mean", "cpu_energy_mean", "total_tasks"]].copy()
    composition["gpu_energy_per_task"] = composition["gpu_energy_mean"] / composition["total_tasks"]
    composition["cpu_energy_per_task"] = composition["cpu_energy_mean"] / composition["total_tasks"]

    composition["Configuration"] = composition["model_label"] + " " + composition["quant"].astype(str)

    plot_df = composition.set_index("Configuration")[["gpu_energy_per_task", "cpu_energy_per_task"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(kind="bar", stacked=True, color=["#4CAF50", "#2196F3"], ax=ax)

    ax.set_title("RQ3: Energy Draw Composition (GPU vs CPU per Task)")
    ax.set_ylabel("Average Energy Consumed (Joules)")
    ax.legend(["Dedicated GPU Energy", "System CPU Energy"], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')

    save_plot(fig, plots_path / "rq3/energy_composition")


def plot_resource_usage(df, plots_path):
    """Bar plot: CPU, RAM, and GPU power usage by model."""

    resource_cols = []
    resource_labels = {}
    for col, label in [("cpu_mean", "CPU (%)"), ("ram_mean", "RAM (MB)"), ("gpu_power_mean", "GPU Power (W)")]:
        if col in df.columns:
            resource_cols.append(col)
            resource_labels[col] = label

    if not resource_cols:
        print("  [SKIP] No resource columns found for resource usage plot.")
        return

    resource_df = df[["model_label"] + resource_cols].copy()

    resource_df = resource_df.melt(
        id_vars="model_label",
        var_name="Resource",
        value_name="Usage"
    )

    resource_df["Resource"] = resource_df["Resource"].replace(resource_labels)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(
        data=resource_df,
        x="model_label",
        y="Usage",
        hue="Resource",
        palette=PALETTE,
        errorbar=None,
        ax=ax
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Usage")
    ax.set_title("Resource Usage by Model")

    place_legend(ax, "Resource")

    save_plot(fig, plots_path / "rq3/resource_usage")


# ------------------------------------------------
# Tables
# ------------------------------------------------

def save_tables(df, plots_path):

    tables_path = plots_path / "tables"
    tables_path.mkdir(parents=True, exist_ok=True)

    # RQ1: Quantization performance table
    rq1_cols = ["model", "size", "quant", "domain",
                "success_rate_mean", "success_rate_std",
                "mean_reward_mean", "mean_reward_std",
                "avg_tool_calls_mean", "avg_tool_calls_std",
                "avg_turns_mean", "avg_turns_std"]
    rq1_available = [c for c in rq1_cols if c in df.columns]
    df[rq1_available].to_csv(tables_path / "rq1_quantization.csv", index=False)

    # RQ2: Failure rates table
    rq2_cols = ["model", "size", "quant", "domain",
                "tle_rate_mean", "tle_rate_std",
                "if_rate_mean", "if_rate_std",
                "ia_rate_mean", "ia_rate_std",
                "task_error_rate_mean", "task_error_rate_std",
                "completed_failure_rate_mean", "completed_failure_rate_std",
                "error_rate_mean", "error_rate_std"]
    rq2_available = [c for c in rq2_cols if c in df.columns]
    df[rq2_available].to_csv(tables_path / "rq2_failures.csv", index=False)

    # RQ3: Efficiency table
    rq3_cols = ["model", "size", "quant", "domain",
                "energy_per_task_mean", "energy_per_task_std",
                "energy_per_success_mean", "energy_per_success_std",
                "energy_per_action_mean", "energy_per_action_std",
                "gpu_mem_mean", "gpu_mem_std",
                "gpu_energy_mean", "gpu_energy_std",
                "cpu_energy_mean", "cpu_energy_std",
                "energy_mean", "energy_std"]
    rq3_available = [c for c in rq3_cols if c in df.columns]
    df[rq3_available].to_csv(tables_path / "rq3_efficiency.csv", index=False)

    print(f"  Tables saved to: {tables_path}")


# ------------------------------------------------
# Pipeline Entry
# ------------------------------------------------

def run(results_path, plots_path):

    master_csv = results_path / "master_results.csv"

    if not master_csv.exists():
        raise FileNotFoundError(f"Master results CSV not found: {master_csv}")

    df = pd.read_csv(master_csv)
    df = standardize_labels(df)

    # Replace inf/nan for safe plotting
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # RQ1 plots
    plot_rq1_success_decay(df, plots_path)
    plot_rq1_wandering(df, plots_path)
    plot_quant_vs_reward(df, plots_path)
    plot_tool_calls(df, plots_path)

    # RQ2 plots
    plot_rq2_failure_anatomy(df, plots_path)
    plot_failure_heatmap(df, plots_path)

    # RQ3 plots
    plot_rq3_pareto(df, plots_path)
    plot_rq3_true_cost(df, plots_path)
    plot_rq3_hardware_ceiling(df, plots_path)
    plot_rq3_energy_composition(df, plots_path)
    plot_resource_usage(df, plots_path)

    # Tables
    save_tables(df, plots_path)

    print(f"\nPlots saved to: {plots_path}")


if __name__ == "__main__":
    from pathlib import Path

    results_path = Path(__file__).resolve().parents[1] / "results"
    plots_path = Path(__file__).resolve().parents[1] / "plots"

    run(results_path, plots_path)