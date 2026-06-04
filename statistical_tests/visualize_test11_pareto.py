"""
visualize_test11_pareto.py
==========================
Generates 5 publication-quality graphs for Test 11: Pareto Efficiency Analysis.

Graphs produced:
  1. Overall Pareto Frontier with dominance arrows
  2. Domain-wise Pareto Frontiers (2x2 grid)
  3. Quantization Frontier Composition (bar + scatter combo)
  4. Dominance Heatmap (who dominates whom)
  5. Energy Savings vs Success Cost scatter (trade-off summary)

All data is loaded from the same data_loader used by the statistical tests.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

from data_loader import load_run_data

# ── Style ──────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linestyle": ":", "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})

QUANT_COLOR = {"bf16": "#4477AA", "q8_0": "#228833", "q4_k_m": "#AA3377"}
QUANT_LABEL = {"bf16": "bf16 (Full)", "q8_0": "q8_0 (8-bit)", "q4_k_m": "q4_k_m (4-bit)"}
MODEL_MARKERS = {"ministral3": "o", "deepseek-r1-qwen": "s", "qwen3": "^"}
MODEL_LABELS = {"ministral3": "Ministral 3", "deepseek-r1-qwen": "DeepSeek-R1", "qwen3": "Qwen 3"}

OUT_DIR = "output"


def find_pareto(data):
    """Return boolean mask of Pareto-optimal rows (max success, min energy)."""
    is_pareto = np.ones(len(data), dtype=bool)
    sr = data["success_rate"].values
    ep = data["energy_per_task"].values
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j and sr[j] >= sr[i] and ep[j] <= ep[i] and (sr[j] > sr[i] or ep[j] < ep[i]):
                is_pareto[i] = False
                break
    return is_pareto


def short_name(config):
    return (config.replace("deepseek-r1-qwen", "DS-R1")
                  .replace("ministral3", "M3")
                  .replace("qwen3", "Q3"))


def load_and_prepare():
    df = load_run_data().dropna(subset=["success_rate", "energy_per_task"])
    overall = df.groupby(["model", "size", "quant"]).agg(
        success_rate=("success_rate", "mean"),
        energy_per_task=("energy_per_task", "mean"),
    ).reset_index()
    overall["config"] = overall["model"] + "-" + overall["size"] + "-" + overall["quant"]
    overall["is_pareto"] = find_pareto(overall)

    domain_agg = df.groupby(["model", "size", "quant", "domain"]).agg(
        success_rate=("success_rate", "mean"),
        energy_per_task=("energy_per_task", "mean"),
    ).reset_index()
    domain_agg["config"] = domain_agg["model"] + "-" + domain_agg["size"] + "-" + domain_agg["quant"]
    return df, overall, domain_agg


# ── Plot 1: Overall Pareto Frontier ───────────────────────────────────
def plot_overall_frontier(overall):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pareto_pts = overall[overall["is_pareto"]].sort_values("energy_per_task")

    # Dominated points (faded)
    for _, row in overall[~overall["is_pareto"]].iterrows():
        ax.scatter(row["energy_per_task"], row["success_rate"],
                   color=QUANT_COLOR[row["quant"]], marker=MODEL_MARKERS[row["model"]],
                   s=60, edgecolors="gray", linewidths=0.5, alpha=0.45, zorder=3)

    # Pareto-optimal points (bold)
    offsets = {"M3-3B-q4_k_m": (8, -14), "M3-8B-q4_k_m": (8, 8),
               "Q3-4B-q4_k_m": (-90, -10), "Q3-4B-q8_0": (8, 8)}
    for _, row in pareto_pts.iterrows():
        ax.scatter(row["energy_per_task"], row["success_rate"],
                   color=QUANT_COLOR[row["quant"]], marker=MODEL_MARKERS[row["model"]],
                   s=150, edgecolors="black", linewidths=1.5, zorder=5)
        sn = short_name(row["config"])
        ox, oy = offsets.get(sn, (8, 6))
        ax.annotate(sn, (row["energy_per_task"], row["success_rate"]),
                    xytext=(ox, oy), textcoords="offset points", fontsize=7.5,
                    weight="semibold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCC", alpha=0.9, lw=0.5),
                    zorder=6)

    # Frontier line
    ax.plot(pareto_pts["energy_per_task"], pareto_pts["success_rate"],
            color="black", ls="--", lw=1.5, zorder=4)

    # Shade dominated region
    p_sr = pareto_pts["success_rate"].values
    p_ep = pareto_pts["energy_per_task"].values
    ax.fill_between(p_ep, p_sr, 0, alpha=0.06, color="red", label="Dominated region")

    # Annotations: show bf16 baseline being dominated
    bf16_best = overall[(overall["quant"] == "bf16")].sort_values("success_rate", ascending=False).iloc[0]
    ax.annotate(f"Best bf16:\n{short_name(bf16_best['config'])}\n(dominated)",
                (bf16_best["energy_per_task"], bf16_best["success_rate"]),
                xytext=(-20, 25), textcoords="offset points", fontsize=7,
                color="#4477AA", fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color="#4477AA", lw=0.8), zorder=7)

    ax.set_xscale("log")
    ax.set_xlabel("Energy per Task (J) — lower is better →", labelpad=6)
    ax.set_ylabel("Task Success Rate — higher is better →", labelpad=6)
    ax.set_title("Test 11: Overall Pareto Frontier\n(Success Rate vs Energy Cost)", pad=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(-0.02, 0.55)

    # Legend
    handles = [
        Line2D([0],[0], marker='o', color='w', label=QUANT_LABEL[q], markerfacecolor=QUANT_COLOR[q], markersize=8)
        for q in ["bf16","q8_0","q4_k_m"]
    ] + [Line2D([0],[0], marker='None', color='w', label='')] + [
        Line2D([0],[0], marker=MODEL_MARKERS[m], color='w', label=MODEL_LABELS[m], markerfacecolor='gray', markersize=8)
        for m in MODEL_MARKERS
    ] + [
        Line2D([0],[0], color='black', ls='--', lw=1.5, label='Pareto frontier'),
        Line2D([0],[0], marker='o', color='w', label='★ Pareto-optimal', markerfacecolor='none',
               markeredgecolor='black', markeredgewidth=1.5, markersize=10),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, edgecolor="#DDD")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/test11_pareto_overall.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: test11_pareto_overall.png")


# ── Plot 2: Domain-wise Pareto Frontiers ──────────────────────────────
def plot_domain_frontiers(domain_agg):
    domains = ["dbbench", "alfworld", "webshop", "os"]
    domain_titles = {"dbbench": "DB-Bench", "alfworld": "ALFWorld", "webshop": "WebShop", "os": "OS"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for i, domain in enumerate(domains):
        ax = axes[i // 2][i % 2]
        dom = domain_agg[domain_agg["domain"] == domain].copy()
        dom["is_pareto"] = find_pareto(dom)
        pareto_pts = dom[dom["is_pareto"]].sort_values("energy_per_task")

        for _, row in dom[~dom["is_pareto"]].iterrows():
            ax.scatter(row["energy_per_task"], row["success_rate"],
                       color=QUANT_COLOR[row["quant"]], marker=MODEL_MARKERS[row["model"]],
                       s=50, edgecolors="gray", linewidths=0.4, alpha=0.45, zorder=3)
        for _, row in pareto_pts.iterrows():
            ax.scatter(row["energy_per_task"], row["success_rate"],
                       color=QUANT_COLOR[row["quant"]], marker=MODEL_MARKERS[row["model"]],
                       s=110, edgecolors="black", linewidths=1.3, zorder=5)
        if len(pareto_pts) > 1:
            ax.plot(pareto_pts["energy_per_task"], pareto_pts["success_rate"],
                    color="black", ls="--", lw=1.3, zorder=4)

        # Count frontier composition for this domain
        frontier_quants = pareto_pts["quant"].value_counts()
        comp_text = "  ".join(f"{q}:{frontier_quants.get(q,0)}" for q in ["bf16","q8_0","q4_k_m"])
        ax.text(0.03, 0.97, f"Frontier: {comp_text}", transform=ax.transAxes,
                fontsize=7, va="top", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9FA", ec="#CCC", alpha=0.9, lw=0.5))

        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(domain_titles[domain], fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)

    fig.text(0.5, 0.02, "Energy per Task (J, log scale)", ha="center", fontsize=11, fontweight="bold")
    fig.text(0.02, 0.5, "Success Rate", ha="center", va="center", rotation=90, fontsize=11, fontweight="bold")
    fig.suptitle("Test 11: Domain-wise Pareto Frontiers", fontsize=13, fontweight="bold", y=0.98)

    handles = [
        Line2D([0],[0], marker='o', color='w', label=q, markerfacecolor=QUANT_COLOR[q], markersize=7)
        for q in QUANT_COLOR
    ] + [
        Line2D([0],[0], marker=MODEL_MARKERS[m], color='w', label=MODEL_LABELS[m], markerfacecolor='gray', markersize=7)
        for m in MODEL_MARKERS
    ] + [Line2D([0],[0], color='black', ls='--', lw=1.3, label='Pareto frontier')]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor="#DDD")

    plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
    fig.savefig(f"{OUT_DIR}/test11_pareto_domains.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: test11_pareto_domains.png")


# ── Plot 3: Frontier Composition Bar Chart ────────────────────────────
def plot_frontier_composition(overall, domain_agg):
    """Bar chart: how many configs from each quant level are Pareto-optimal, overall + per domain."""
    domains = ["Overall", "dbbench", "alfworld", "webshop", "os"]
    quants = ["bf16", "q8_0", "q4_k_m"]

    counts = {q: [] for q in quants}
    totals = {q: [] for q in quants}

    # Overall
    for q in quants:
        n_pareto = overall[(overall["is_pareto"]) & (overall["quant"] == q)].shape[0]
        n_total = overall[overall["quant"] == q].shape[0]
        counts[q].append(n_pareto)
        totals[q].append(n_total)

    # Per domain
    for domain in domains[1:]:
        dom = domain_agg[domain_agg["domain"] == domain].copy()
        dom["is_pareto"] = find_pareto(dom)
        for q in quants:
            n_pareto = dom[(dom["is_pareto"]) & (dom["quant"] == q)].shape[0]
            n_total = dom[dom["quant"] == q].shape[0]
            counts[q].append(n_pareto)
            totals[q].append(n_total)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(domains))
    width = 0.25

    for i, q in enumerate(quants):
        bars = ax.bar(x + (i - 1) * width, counts[q], width, label=QUANT_LABEL[q],
                      color=QUANT_COLOR[q], edgecolor="white", linewidth=0.5, alpha=0.85)
        # Add fraction labels on bars
        for j, (c, t) in enumerate(zip(counts[q], totals[q])):
            if c > 0:
                ax.text(x[j] + (i - 1) * width, c + 0.08, f"{c}/{t}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Overall", "DB-Bench", "ALFWorld", "WebShop", "OS"])
    ax.set_ylabel("Number of Pareto-Optimal Configurations")
    ax.set_title("Test 11: Pareto Frontier Composition by Quantization Level\n"
                 "(How many configs from each quant level are non-dominated?)", pad=10)
    ax.legend(frameon=True, edgecolor="#DDD")
    ax.set_ylim(0, max(max(v) for v in counts.values()) + 1.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Key takeaway annotation
    ax.text(0.98, 0.95,
            "Key finding: q4_k_m dominates the\n"
            "overall frontier (3/4 optimal configs).\n"
            "No bf16 config is Pareto-optimal overall.",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9E6", ec="#E6C300", alpha=0.9, lw=0.8))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/test11_frontier_composition.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: test11_frontier_composition.png")


# ── Plot 4: Dominance Heatmap ─────────────────────────────────────────
def plot_dominance_heatmap(overall):
    """Heatmap showing how many configs each configuration dominates."""
    configs = overall.sort_values("success_rate", ascending=False)
    names = [short_name(c) for c in configs["config"]]
    sr = configs["success_rate"].values
    ep = configs["energy_per_task"].values
    n = len(configs)

    # Build dominance matrix: dom[i,j] = 1 means config i dominates config j
    dom = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and sr[i] >= sr[j] and ep[i] <= ep[j] and (sr[i] > sr[j] or ep[i] < ep[j]):
                dom[i, j] = 1

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = matplotlib.colors.ListedColormap(["#F5F5F5", "#66BB6A"])
    ax.imshow(dom, cmap=cmap, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7.5)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("← is dominated by (column dominates row)")
    ax.set_ylabel("Row config →")
    ax.set_title("Test 11: Dominance Matrix\n(Green = row is dominated by column)", pad=12)

    # Add count annotations in margins
    dominated_by_count = dom.sum(axis=0)  # how many others each config dominates
    is_dominated_count = dom.sum(axis=1)  # how many dominate this config

    # Right margin: how many dominate this config
    for i in range(n):
        ax.text(n + 0.3, i, f"dominated by {is_dominated_count[i]}", fontsize=7, va="center", color="#D32F2F")
    # Bottom margin: how many this config dominates
    for j in range(n):
        ax.text(j, n + 0.3, f"{dominated_by_count[j]}", fontsize=7, ha="center", va="top", color="#2E7D32",
                fontweight="bold")

    # Highlight Pareto configs (row not dominated by anyone)
    for i in range(n):
        if is_dominated_count[i] == 0:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), n, 1, fill=False,
                                        edgecolor="gold", linewidth=2.5, zorder=10, clip_on=False))
            ax.text(-1.5, i, "★", fontsize=12, va="center", ha="center", color="gold", fontweight="bold")

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/test11_dominance_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: test11_dominance_heatmap.png")


# ── Plot 5: bf16 Baseline Comparison ─────────────────────────────────
def plot_bf16_comparison(overall):
    """For each model-size, show how quantized variants compare to bf16 baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Group by model-size
    overall["model_size"] = overall["model"] + "-" + overall["size"]
    model_sizes = sorted(overall["model_size"].unique())

    # Left panel: Energy savings from quantization (vs bf16)
    ax = axes[0]
    x_positions = np.arange(len(model_sizes))
    width = 0.35

    for idx, q in enumerate(["q8_0", "q4_k_m"]):
        savings = []
        for ms in model_sizes:
            bf16_row = overall[(overall["model_size"] == ms) & (overall["quant"] == "bf16")]
            q_row = overall[(overall["model_size"] == ms) & (overall["quant"] == q)]
            if len(bf16_row) > 0 and len(q_row) > 0:
                pct = (bf16_row["energy_per_task"].values[0] - q_row["energy_per_task"].values[0]) / bf16_row["energy_per_task"].values[0] * 100
                savings.append(pct)
            else:
                savings.append(0)
        bars = ax.bar(x_positions + idx * width, savings, width, label=QUANT_LABEL[q],
                      color=QUANT_COLOR[q], alpha=0.85, edgecolor="white")
        for j, s in enumerate(savings):
            ax.text(x_positions[j] + idx * width, s + 0.5, f"{s:.0f}%", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x_positions + width / 2)
    short_ms = [ms.replace("deepseek-r1-qwen", "DS-R1").replace("ministral3", "M3").replace("qwen3", "Q3") for ms in model_sizes]
    ax.set_xticklabels(short_ms, rotation=30, ha="right")
    ax.set_ylabel("Energy Savings vs bf16 (%)")
    ax.set_title("Energy Savings from Quantization", fontweight="bold")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.5)

    # Right panel: Success rate change from quantization (vs bf16)
    ax = axes[1]
    for idx, q in enumerate(["q8_0", "q4_k_m"]):
        deltas = []
        for ms in model_sizes:
            bf16_row = overall[(overall["model_size"] == ms) & (overall["quant"] == "bf16")]
            q_row = overall[(overall["model_size"] == ms) & (overall["quant"] == q)]
            if len(bf16_row) > 0 and len(q_row) > 0:
                delta = (q_row["success_rate"].values[0] - bf16_row["success_rate"].values[0]) * 100
                deltas.append(delta)
            else:
                deltas.append(0)
        bars = ax.bar(x_positions + idx * width, deltas, width, label=QUANT_LABEL[q],
                      color=QUANT_COLOR[q], alpha=0.85, edgecolor="white")
        for j, d in enumerate(deltas):
            offset = 0.3 if d >= 0 else -0.8
            ax.text(x_positions[j] + idx * width, d + offset, f"{d:+.1f}pp", ha="center", fontsize=6.5)

    ax.set_xticks(x_positions + width / 2)
    ax.set_xticklabels(short_ms, rotation=30, ha="right")
    ax.set_ylabel("Success Rate Change vs bf16 (pp)")
    ax.set_title("Accuracy Impact of Quantization", fontweight="bold")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.8, ls="-")

    fig.suptitle("Test 11: Why Quantized Models Dominate the Pareto Frontier",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/test11_bf16_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: test11_bf16_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    df, overall, domain_agg = load_and_prepare()
    print(f"  {len(overall)} unique configurations, {len(df)} run-level observations")
    print(f"  Pareto-optimal configs: {overall[overall['is_pareto']]['config'].tolist()}")

    plot_overall_frontier(overall)
    plot_domain_frontiers(domain_agg)
    plot_frontier_composition(overall, domain_agg)
    plot_dominance_heatmap(overall)
    plot_bf16_comparison(overall)

    print("\nAll 5 Test 11 Pareto visualizations generated successfully!")


if __name__ == "__main__":
    main()
