"""
visualize_results.py
====================
Generates publication-quality Pareto frontier visualizations for RQ3.
It creates four required plots:
1. Overall Pareto Frontier (pareto_overall.png)
2. Domain-wise Pareto Frontiers (pareto_domains.png)
3. Quantization-Focused Plot (pareto_quantization.png)
4. Model-Family Comparison Plot (pareto_by_model.png)

Uses data_loader.py to load and process the database results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# Import shared data loader
from data_loader import load_run_data

# Set publication-quality style globally
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    # Spines
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})

# Paul Tol's colorblind-safe palette for quantization
QUANT_COLOR = {
    "bf16": "#4477AA",    # Blue
    "q8_0": "#228833",    # Green
    "q4_k_m": "#AA3377",  # Purple
}

QUANT_LABEL = {
    "bf16": "bf16 (Full precision)",
    "q8_0": "q8_0 (8-bit quantization)",
    "q4_k_m": "q4_k_m (4-bit quantization)",
}

# Consistent marker shape by model family
MODEL_MARKERS = {
    "ministral3": "o",       # Circle
    "deepseek-r1-qwen": "s",  # Square
    "qwen3": "^",            # Triangle Up
}

MODEL_LABELS = {
    "ministral3": "Ministral 3",
    "deepseek-r1-qwen": "DeepSeek-R1-Qwen",
    "qwen3": "Qwen 3",
}

def find_pareto(data):
    """
    Find Pareto-optimal points (max success_rate, min energy_per_task).
    A point is dominated if another point has:
      success_rate >= current success_rate
      AND energy_per_task <= current energy_per_task
      with at least one strict inequality.
    """
    is_pareto = np.ones(len(data), dtype=bool)
    sr = data["success_rate"].values
    ep = data["energy_per_task"].values
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                # j dominates i if:
                if sr[j] >= sr[i] and ep[j] <= ep[i] and (sr[j] > sr[i] or ep[j] < ep[i]):
                    is_pareto[i] = False
                    break
    return is_pareto

def generate_overall_pareto(df_overall):
    """Plot 1: Overall Pareto Frontier."""
    print("Generating Plot 1: Overall Pareto Frontier...")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    
    # Identify Pareto optimal points
    is_p = find_pareto(df_overall)
    df_overall["is_pareto"] = is_p
    pareto_pts = df_overall[df_overall["is_pareto"]].sort_values("energy_per_task")
    
    # Plot dominated points
    for idx, row in df_overall[~df_overall["is_pareto"]].iterrows():
        color = QUANT_COLOR[row["quant"]]
        marker = MODEL_MARKERS[row["model"]]
        ax.scatter(
            row["energy_per_task"], row["success_rate"],
            color=color, marker=marker, s=65,
            edgecolors="gray", linewidths=0.5, alpha=0.55, zorder=3
        )
        
    # Plot Pareto-optimal points (larger, black outlines)
    for idx, row in df_overall[df_overall["is_pareto"]].iterrows():
        color = QUANT_COLOR[row["quant"]]
        marker = MODEL_MARKERS[row["model"]]
        ax.scatter(
            row["energy_per_task"], row["success_rate"],
            color=color, marker=marker, s=140,
            edgecolors="black", linewidths=1.5, zorder=5
        )
        
        # Annotate overall frontier configurations
        config_name = f"{row['model']}-{row['size']}-{row['quant']}"
        # Shorten representation for clean plot space
        short_name = config_name.replace("deepseek-r1-qwen", "DS-R1").replace("ministral3", "M3").replace("qwen3", "Q3")
        
        # Smart positioning offset
        offset_x, offset_y = 7, 3
        if "DS-R1-1.5B-q4_k_m" in short_name:
            offset_x, offset_y = 7, -11
        elif "Q3-4B-q4_k_m" in short_name:
            offset_x, offset_y = -75, -5
            
        ax.annotate(
            short_name,
            xy=(row["energy_per_task"], row["success_rate"]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8,
            weight="semibold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCCCCC", alpha=0.85, lw=0.5),
            zorder=6
        )

    # Draw frontier line
    ax.plot(
        pareto_pts["energy_per_task"], pareto_pts["success_rate"],
        color="black", linestyle="--", linewidth=1.5, zorder=4, label="Pareto Frontier"
    )
    
    ax.set_xscale("log")
    ax.set_xlabel("Energy per Task (J, Log Scale)", labelpad=6)
    ax.set_ylabel("Task Success Rate", labelpad=6)
    ax.set_title("Overall Pareto Frontier: Success Rate vs. Energy per Task", pad=12)
    
    # Formatting
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    
    # Adjust axes limits for better visual spacing
    ax.set_xlim(df_overall["energy_per_task"].min() * 0.8, df_overall["energy_per_task"].max() * 1.25)
    ax.set_ylim(-0.02, 1.05)
    
    # Legend construction
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=QUANT_LABEL['bf16'], markerfacecolor=QUANT_COLOR['bf16'], markersize=9),
        Line2D([0], [0], marker='o', color='w', label=QUANT_LABEL['q8_0'], markerfacecolor=QUANT_COLOR['q8_0'], markersize=9),
        Line2D([0], [0], marker='o', color='w', label=QUANT_LABEL['q4_k_m'], markerfacecolor=QUANT_COLOR['q4_k_m'], markersize=9),
        Line2D([0], [0], marker='None', color='w', label=''),  # Spacer
        Line2D([0], [0], marker=MODEL_MARKERS['ministral3'], color='w', label=MODEL_LABELS['ministral3'], markerfacecolor='gray', markersize=9),
        Line2D([0], [0], marker=MODEL_MARKERS['deepseek-r1-qwen'], color='w', label=MODEL_LABELS['deepseek-r1-qwen'], markerfacecolor='gray', markersize=9),
        Line2D([0], [0], marker=MODEL_MARKERS['qwen3'], color='w', label=MODEL_LABELS['qwen3'], markerfacecolor='gray', markersize=9),
        Line2D([0], [0], marker='None', color='w', label=''),  # Spacer
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Pareto Frontier Line'),
        Line2D([0], [0], marker='o', color='w', label='Pareto-Optimal Configuration', markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.5, markersize=10)
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, edgecolor="#DDDDDD")
    
    plt.tight_layout()
    fig.savefig("pareto_overall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pareto_overall.png")

def generate_domain_pareto(df_domain):
    """Plot 2: Domain-wise Pareto Frontiers (2x2 grid)."""
    print("Generating Plot 2: Domain-wise Pareto Frontiers...")
    domains = ["dbbench", "alfworld", "webshop", "os"]
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Global limits for strict scale sharing
    x_min = df_domain["energy_per_task"].min() * 0.8
    x_max = df_domain["energy_per_task"].max() * 1.2
    y_min = -0.05
    y_max = 1.05
    
    for i, domain in enumerate(domains):
        ax = axes[i]
        dom_df = df_domain[df_domain["domain"] == domain].copy()
        
        # Calculate domain-specific Pareto
        is_p = find_pareto(dom_df)
        dom_df["is_pareto"] = is_p
        pareto_pts = dom_df[dom_df["is_pareto"]].sort_values("energy_per_task")
        
        # Dominated points
        for idx, row in dom_df[~dom_df["is_pareto"]].iterrows():
            color = QUANT_COLOR[row["quant"]]
            marker = MODEL_MARKERS[row["model"]]
            ax.scatter(
                row["energy_per_task"], row["success_rate"],
                color=color, marker=marker, s=55,
                edgecolors="gray", linewidths=0.5, alpha=0.5, zorder=3
            )
            
        # Pareto-optimal points
        for idx, row in dom_df[dom_df["is_pareto"]].iterrows():
            color = QUANT_COLOR[row["quant"]]
            marker = MODEL_MARKERS[row["model"]]
            ax.scatter(
                row["energy_per_task"], row["success_rate"],
                color=color, marker=marker, s=110,
                edgecolors="black", linewidths=1.5, zorder=5
            )
            
        # Frontier line
        ax.plot(
            pareto_pts["energy_per_task"], pareto_pts["success_rate"],
            color="black", linestyle="--", linewidth=1.5, zorder=4
        )
        
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="both", linestyle=":", alpha=0.25)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        
        # Nicer titles
        domain_title = domain.upper() if domain in ["os"] else domain.capitalize()
        ax.set_title(f"Domain: {domain_title}", fontsize=11, fontweight="bold")
        
    # Global labels
    fig.text(0.5, 0.07, "Energy per Task (J, Log Scale)", ha="center", va="center", fontsize=11, fontweight="bold")
    fig.text(0.04, 0.53, "Success Rate", ha="center", va="center", rotation="vertical", fontsize=11, fontweight="bold")
    fig.suptitle("Domain-wise Pareto Frontiers Comparison", fontsize=13, fontweight="bold", y=0.97)
    
    # Legend below the subplots
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='bf16', markerfacecolor=QUANT_COLOR['bf16'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='q8_0', markerfacecolor=QUANT_COLOR['q8_0'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='q4_k_m', markerfacecolor=QUANT_COLOR['q4_k_m'], markersize=8),
        Line2D([0], [0], marker=MODEL_MARKERS['ministral3'], color='w', label=MODEL_LABELS['ministral3'], markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker=MODEL_MARKERS['deepseek-r1-qwen'], color='w', label=MODEL_LABELS['deepseek-r1-qwen'], markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker=MODEL_MARKERS['qwen3'], color='w', label=MODEL_LABELS['qwen3'], markerfacecolor='gray', markersize=8),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Pareto Frontier')
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.01), frameon=True, edgecolor="#DDDDDD")
    
    plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.94])
    fig.savefig("pareto_domains.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pareto_domains.png")

def generate_quantization_pareto(df_overall):
    """Plot 3: Quantization-Focused Plot."""
    print("Generating Plot 3: Quantization-Focused Plot...")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    
    # Identify Pareto optimal points
    is_p = find_pareto(df_overall)
    df_overall["is_pareto"] = is_p
    pareto_pts = df_overall[df_overall["is_pareto"]].sort_values("energy_per_task")
    
    # Count and percentages of frontier points per quantization level
    pareto_df = df_overall[df_overall["is_pareto"]]
    total_frontier = len(pareto_df)
    stats_lines = []
    for q in ["bf16", "q8_0", "q4_k_m"]:
        count = len(pareto_df[pareto_df["quant"] == q])
        pct = (count / total_frontier * 100) if total_frontier > 0 else 0.0
        stats_lines.append(f" {q:<6} : {count} ({pct:.1f}%)")
    text_content = "Overall Frontier Composition:\n" + "\n".join(stats_lines)

    # Plot all configurations (circle markers only, colored by quantization)
    for idx, row in df_overall[~df_overall["is_pareto"]].iterrows():
        color = QUANT_COLOR[row["quant"]]
        ax.scatter(
            row["energy_per_task"], row["success_rate"],
            color=color, marker="o", s=65,
            edgecolors="gray", linewidths=0.5, alpha=0.55, zorder=3
        )
        
    for idx, row in df_overall[df_overall["is_pareto"]].iterrows():
        color = QUANT_COLOR[row["quant"]]
        ax.scatter(
            row["energy_per_task"], row["success_rate"],
            color=color, marker="o", s=140,
            edgecolors="black", linewidths=1.5, zorder=5
        )
        
    # Draw overall Pareto frontier line
    ax.plot(
        pareto_pts["energy_per_task"], pareto_pts["success_rate"],
        color="black", linestyle="-", linewidth=1.5, zorder=4, label="Pareto Frontier"
    )
    
    # Add Text Box with stats
    ax.text(
        0.05, 0.95, text_content,
        transform=ax.transAxes,
        fontsize=9.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.9, lw=0.75)
    )
    
    ax.set_xscale("log")
    ax.set_xlabel("Energy per Task (J, Log Scale)", labelpad=6)
    ax.set_ylabel("Task Success Rate", labelpad=6)
    ax.set_title("Quantization-Focused Pareto Frontier", pad=12)
    
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    
    ax.set_xlim(df_overall["energy_per_task"].min() * 0.8, df_overall["energy_per_task"].max() * 1.2)
    ax.set_ylim(-0.02, 1.05)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=QUANT_LABEL['bf16'], markerfacecolor=QUANT_COLOR['bf16'], markersize=9),
        Line2D([0], [0], marker='o', color='w', label=QUANT_LABEL['q8_0'], markerfacecolor=QUANT_COLOR['q8_0'], markersize=9),
        Line2D([0], [0], marker='o', color='w', label=QUANT_LABEL['q4_k_m'], markerfacecolor=QUANT_COLOR['q4_k_m'], markersize=9),
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='Pareto Frontier Line'),
        Line2D([0], [0], marker='o', color='w', label='Frontier Configuration', markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.5, markersize=10)
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, edgecolor="#DDDDDD")
    
    plt.tight_layout()
    fig.savefig("pareto_quantization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pareto_quantization.png")

def generate_by_model_pareto(df_overall):
    """Plot 4: Model-Family Comparison Plot (1x3 grid)."""
    print("Generating Plot 4: Model-Family Comparison Plot...")
    model_families = ["ministral3", "deepseek-r1-qwen", "qwen3"]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    
    x_min = df_overall["energy_per_task"].min() * 0.8
    x_max = df_overall["energy_per_task"].max() * 1.2
    
    for i, model in enumerate(model_families):
        ax = axes[i]
        model_df = df_overall[df_overall["model"] == model].copy()
        
        # Calculate within-family Pareto frontier
        is_p = find_pareto(model_df)
        model_df["is_pareto_fam"] = is_p
        pareto_pts = model_df[model_df["is_pareto_fam"]].sort_values("energy_per_task")
        
        # Plot dominated family configurations
        for idx, row in model_df[~model_df["is_pareto_fam"]].iterrows():
            color = QUANT_COLOR[row["quant"]]
            marker = MODEL_MARKERS[row["model"]]
            ax.scatter(
                row["energy_per_task"], row["success_rate"],
                color=color, marker=marker, s=65,
                edgecolors="gray", linewidths=0.5, alpha=0.55, zorder=3
            )
            
        # Plot Pareto-optimal family configurations
        for idx, row in model_df[model_df["is_pareto_fam"]].iterrows():
            color = QUANT_COLOR[row["quant"]]
            marker = MODEL_MARKERS[row["model"]]
            ax.scatter(
                row["energy_per_task"], row["success_rate"],
                color=color, marker=marker, s=130,
                edgecolors="black", linewidths=1.5, zorder=5
            )
            
        # Draw family frontier line
        ax.plot(
            pareto_pts["energy_per_task"], pareto_pts["success_rate"],
            color="black", linestyle="--", linewidth=1.5, zorder=4
        )
        
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which="both", linestyle=":", alpha=0.25)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")
        ax.set_xlabel("Energy per Task (J, Log Scale)", labelpad=6)
        
    axes[0].set_ylabel("Task Success Rate", fontsize=11, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    
    fig.suptitle("Model-Family Comparison: Within-Family Pareto Frontiers", fontsize=13, fontweight="bold", y=0.98)
    
    # Legend below the subplots
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='bf16', markerfacecolor=QUANT_COLOR['bf16'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='q8_0', markerfacecolor=QUANT_COLOR['q8_0'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='q4_k_m', markerfacecolor=QUANT_COLOR['q4_k_m'], markersize=8),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Family Pareto Frontier'),
        Line2D([0], [0], marker='o', color='w', label='Non-Dominated (Optimal)', markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.5, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Dominated Configuration', markerfacecolor='none', markeredgecolor='gray', markeredgewidth=0.5, markersize=8)
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05), frameon=True, edgecolor="#DDDDDD")
    
    plt.tight_layout()
    fig.savefig("pareto_by_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pareto_by_model.png")

def main():
    print("Loading data...")
    df = load_run_data()
    
    # Drop rows with missing metrics needed for Pareto frontier
    df = df.dropna(subset=["success_rate", "energy_per_task"])
    
    # Aggregate data at the configuration-domain level (for domain-specific plots)
    df_domain = df.groupby(["model", "size", "quant", "domain"]).agg(
        success_rate=("success_rate", "mean"),
        energy_per_task=("energy_per_task", "mean")
    ).reset_index()
    df_domain["config"] = df_domain["model"] + "-" + df_domain["size"] + "-" + df_domain["quant"]
    
    # Aggregate data at the overall configuration level (for overall/model plots)
    df_overall = df.groupby(["model", "size", "quant"]).agg(
        success_rate=("success_rate", "mean"),
        energy_per_task=("energy_per_task", "mean")
    ).reset_index()
    df_overall["config"] = df_overall["model"] + "-" + df_overall["size"] + "-" + df_overall["quant"]
    
    # Check output directory
    print(f"Data observations count: {len(df)}")
    print(f"Unique configurations count: {len(df_overall)}")
    
    # Generate the 4 plots
    generate_overall_pareto(df_overall)
    generate_domain_pareto(df_domain)
    generate_quantization_pareto(df_overall)
    generate_by_model_pareto(df_overall)
    
    print("\nAll publication-quality plots successfully generated!")

if __name__ == "__main__":
    main()
