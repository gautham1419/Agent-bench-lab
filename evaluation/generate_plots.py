import json
from pathlib import Path

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

PALETTE = sns.color_palette("colorblind")
FIGSIZE = (6, 4)


# ------------------------------------------------
# Label Standardization
# ------------------------------------------------

QUANT_LABELS = {
    "fp16": "FP16",
    "q8_0": "8-bit",
    "q4_k_m": "4-bit",
}

MODEL_LABELS = {
    "ministral3": "Ministral3",
    "qwen3": "Qwen3",
}


def standardize_labels(df):

    df = df.copy()

    if "quant" in df.columns:
        df["quant"] = df["quant"].replace(QUANT_LABELS)

    if "model" in df.columns:
        df["model"] = df["model"].replace(MODEL_LABELS)

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
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )


def load_tradeoff_metrics(results_path):

    trade_file = results_path / "aggregated" / "tradeoff_metrics.json"

    with open(trade_file) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    return standardize_labels(df)


# ------------------------------------------------
# RQ1 – Quantization vs Performance
# ------------------------------------------------

def plot_quant_vs_success(df, plots_path):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(
        data=df,
        x="quant",
        y="success_rate_mean",
        hue="size",
        palette=PALETTE,
        errorbar=None,
        ax=ax
    )

    ax.set_xlabel("Quantization")
    ax.set_ylabel("Success Rate")
    ax.set_title("Impact of Quantization on Task Success")

    place_legend(ax, "Model Size")

    save_plot(fig, plots_path / "rq1/quant_vs_success")


def plot_quant_vs_reward(df, plots_path):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.lineplot(
        data=df,
        x="quant",
        y="mean_reward_mean",
        hue="size",
        marker="o",
        palette=PALETTE,
        ax=ax
    )

    ax.set_xlabel("Quantization")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Quantization vs Reward")

    place_legend(ax, "Model Size")

    save_plot(fig, plots_path / "rq1/quant_vs_reward")


def plot_tool_calls(df, plots_path):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(
        data=df,
        x="quant",
        y="avg_tool_calls_mean",
        hue="size",
        palette=PALETTE,
        errorbar=None,
        ax=ax
    )

    ax.set_xlabel("Quantization")
    ax.set_ylabel("Tool Calls")
    ax.set_title("Quantization vs Tool Usage")

    place_legend(ax, "Model Size")

    save_plot(fig, plots_path / "rq1/tool_calls")


# ------------------------------------------------
# RQ2 – Failure Analysis (UPDATED)
# ------------------------------------------------

def plot_failure_rates(df, plots_path):

    failure_df = df[
        [
            "quant",
            "agent_failure_rate_mean",
            "interaction_failure_rate_mean",
            "timeout_rate_mean",
            "tool_format_rate_mean",
            "crash_rate_mean",
        ]
    ]

    failure_df = failure_df.melt(
        id_vars="quant",
        var_name="Failure Type",
        value_name="Rate"
    )

    # Clean labels
    failure_df["Failure Type"] = failure_df["Failure Type"].replace({
        "agent_failure_rate_mean": "Agent Failure",
        "interaction_failure_rate_mean": "Interaction Failure",
        "timeout_rate_mean": "Timeout",
        "tool_format_rate_mean": "Tool Format",
        "crash_rate_mean": "Crash"
    })

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

    ax.set_xlabel("Quantization")
    ax.set_ylabel("Rate")
    ax.set_title("Failure Breakdown Across Quantization")

    place_legend(ax, "Failure Type")

    save_plot(fig, plots_path / "rq2/failure_rates")


def plot_failure_heatmap(df, plots_path):

    heatmap_data = df.pivot_table(
        values=[
            "agent_failure_rate_mean",
            "interaction_failure_rate_mean",
            "timeout_rate_mean",
            "success_rate_mean"
        ],
        index="model",
        columns="quant"
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="viridis",
        fmt=".2f",
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

def plot_success_vs_energy(trade_df, plots_path):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.scatterplot(
        data=trade_df,
        x="energy_per_task",
        y="success_rate_mean",
        hue="model",
        style="quant",
        palette=PALETTE,
        s=100,
        ax=ax
    )

    ax.set_xlabel("Energy per Task (J)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Performance vs Energy")

    place_legend(ax, "Model / Quantization")

    save_plot(fig, plots_path / "rq3/success_vs_energy")


def plot_success_vs_memory(df, plots_path):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.scatterplot(
        data=df,
        x="ram_mean",
        y="success_rate_mean",
        hue="model",
        style="quant",
        palette=PALETTE,
        s=100,
        ax=ax
    )

    ax.set_xlabel("Memory Usage (MB)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Performance vs Memory")

    place_legend(ax, "Model / Quantization")

    save_plot(fig, plots_path / "rq3/success_vs_memory")


def plot_energy_efficiency(trade_df, plots_path):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(
        data=trade_df,
        x="model",
        y="success_per_energy",
        hue="quant",
        palette=PALETTE,
        errorbar=None,
        ax=ax
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Success per Joule")
    ax.set_title("Energy Efficiency")

    place_legend(ax, "Quantization")

    save_plot(fig, plots_path / "rq3/energy_efficiency")


def plot_tool_efficiency(trade_df, plots_path):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.scatterplot(
        data=trade_df,
        x="tool_calls_per_success",
        y="success_rate_mean",
        hue="model",
        style="quant",
        palette=PALETTE,
        s=100,
        ax=ax
    )

    ax.set_xlabel("Tool Calls per Success")
    ax.set_ylabel("Success Rate")
    ax.set_title("Tool Efficiency")

    place_legend(ax, "Model / Quantization")

    save_plot(fig, plots_path / "rq3/tool_efficiency")


def plot_resource_usage(df, plots_path):

    resource_df = df[
        [
            "model",
            "cpu_mean",
            "ram_mean",
            "gpu_power_mean"
        ]
    ]

    resource_df = resource_df.melt(
        id_vars="model",
        var_name="Resource",
        value_name="Usage"
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)

    sns.barplot(
        data=resource_df,
        x="model",
        y="Usage",
        hue="Resource",
        palette=PALETTE,
        errorbar=None,
        ax=ax
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Usage")
    ax.set_title("Resource Usage")

    place_legend(ax, "Resource")

    save_plot(fig, plots_path / "rq3/resource_usage")


# ------------------------------------------------
# Tables
# ------------------------------------------------

def save_tables(df, trade_df, plots_path):

    tables_path = plots_path / "tables"
    tables_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_path / "rq1_quantization.csv", index=False)
    df.to_csv(tables_path / "rq2_failures.csv", index=False)
    trade_df.to_csv(tables_path / "rq3_efficiency.csv", index=False)


# ------------------------------------------------
# Pipeline Entry
# ------------------------------------------------

def run(results_path, plots_path):

    master_csv = results_path / "master_results.csv"

    if not master_csv.exists():
        raise FileNotFoundError(master_csv)

    df = pd.read_csv(master_csv)
    df = standardize_labels(df)

    trade_df = load_tradeoff_metrics(results_path)

    plot_quant_vs_success(df, plots_path)
    plot_quant_vs_reward(df, plots_path)
    plot_tool_calls(df, plots_path)

    plot_failure_rates(df, plots_path)
    plot_failure_heatmap(df, plots_path)

    plot_success_vs_energy(trade_df, plots_path)
    plot_success_vs_memory(df, plots_path)
    plot_energy_efficiency(trade_df, plots_path)
    plot_tool_efficiency(trade_df, plots_path)
    plot_resource_usage(df, plots_path)

    save_tables(df, trade_df, plots_path)

    print("Plots saved to:", plots_path)