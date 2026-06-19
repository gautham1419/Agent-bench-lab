import os

import matplotlib.pyplot as plt
import numpy as np

# Set global styles for premium academic look
plt.rcParams.update(
    {
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 9.0,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.7,
        "grid.color": "#e5e5e5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    }
)

# Colors
C_BF16 = "#7cbf7c"  # Light Green   (16-bit)  — matches Pareto fig_rq3b
C_Q8 = "#bda4d4"  # Purple/Mauve  (8-bit)   — matches Pareto fig_rq3b
C_Q4 = "#9fc0e8"  # Light Blue    (4-bit)   — matches Pareto fig_rq3b
C_RED = "#ef4444"  # Bright Coral Red
C_GREY = "#64748b"  # Cool Slate Grey

OUT_DIR = "docs"
os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(fig, name):
    # Save PDF (vector) and PNG (raster)
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{name}.png")

    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name} (PDF, PNG)")


# =========================================================================
# fig_rq1a: Success by quantization
# =========================================================================
def make_rq1a():
    fig, ax = plt.subplots(figsize=(2.5, 2.3))
    x = ["16-bit", "8-bit", "4-bit"]
    y = [18.8, 19.1, 19.3]
    colors = [C_BF16, C_Q8, C_Q4]

    bars = ax.bar(x, y, color=colors, width=0.5, edgecolor="none")

    # Annotate bar heights
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    ax.set_ylabel("Success (%)")
    ax.set_title("(a) Success by quantization")
    ax.set_ylim(0, 28)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig(fig, "fig_rq1a")


# =========================================================================
# fig_rq1b: Success by family & size (grouped)
# =========================================================================
def make_rq1b():
    fig, ax = plt.subplots(figsize=(3.6, 2.3))

    families = ["DeepSeek", "Ministral3", "Qwen3"]
    smaller_scale = [0.3, 18.3, 43.2]  # 1.5B, 3B, 4B
    larger_scale = [3.3, 29.0, 20.6]  # 7B, 8B, 8B

    x = np.arange(len(families))
    width = 0.3

    rects1 = ax.bar(
        x - width / 2,
        smaller_scale,
        width,
        label="Smaller Scale",
        color=C_Q4,
        edgecolor="none",
    )
    rects2 = ax.bar(
        x + width / 2,
        larger_scale,
        width,
        label="Larger Scale",
        color=C_Q8,
        edgecolor="none",
    )

    # Annotate heights
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    ax.set_ylabel("Success (%)")
    ax.set_title("(b) Success by family & size")
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylim(0, 52)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left",
        frameon=True,
        edgecolor="none",
        facecolor="#ffffff",
        framealpha=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig(fig, "fig_rq1b")


# =========================================================================
# fig_rq1c: Raw success rate by domain
# =========================================================================
def make_rq1c():
    fig, ax = plt.subplots(figsize=(2.5, 2.3))

    # Raw (natural) mean success rate per domain, computed across all runs
    # from results/all_runs_master.json (n=54 runs per domain).
    domains = ["ALFWorld", "OS", "DB", "WebShop"]
    success = [3.64, 13.03, 20.22, 39.54]

    bars = ax.barh(domains, success, color=C_BF16, height=0.5, edgecolor="none")

    # Annotate values
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.1f}%",
            xy=(w + 0.8, bar.get_y() + bar.get_height() / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.0,
        )

    ax.set_xlabel("Success (%)")
    ax.set_title("(c) Success by domain")
    ax.set_xlim(0, 45)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig(fig, "fig_rq1c")


# =========================================================================
# fig_rq2a: Failure type proportions (grouped by type)
# =========================================================================
def make_rq2a():
    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    # Failure types
    types = ["TLE", "IF", "IA", "CF", "TE"]

    # Proportions for BF16, Q8_0, Q4_K_M
    bf16_prop = [21.5, 26.3, 9.2, 31.0, 6.9]
    q8_prop = [23.0, 25.4, 9.2, 31.8, 7.1]
    q4_prop = [25.6, 25.2, 9.3, 29.8, 6.7]

    x = np.arange(len(types))
    width = 0.25

    # Use proper labels without backslashes!
    ax.bar(x - width, bf16_prop, width, label="16-bit", color=C_BF16, edgecolor="none")
    ax.bar(x, q8_prop, width, label="8-bit", color=C_Q8, edgecolor="none")
    ax.bar(x + width, q4_prop, width, label="4-bit", color=C_Q4, edgecolor="none")

    ax.set_ylabel("Proportion (%)")
    ax.set_title("(a) Failure type proportions")
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylim(0, 42)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", frameon=True, edgecolor="none")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig(fig, "fig_rq2a")


# =========================================================================
# fig_rq2b: System errors with 95% CI
# =========================================================================
def make_rq2b():
    fig, ax = plt.subplots(figsize=(2.5, 2.67))

    x = ["16-bit", "8-bit", "4-bit"]
    y = [5.03, 3.61, 3.35]

    # Error bar calculations (lower error, upper error)
    # BF16: 5.03% (CI [3.37, 6.92]) -> lower=1.66, upper=1.89
    # Q8_0: 3.61% (CI [2.34, 5.11]) -> lower=1.27, upper=1.50
    # Q4: 3.35% (CI [2.53, 4.22]) -> lower=0.82, upper=0.87
    yerr = [
        [1.66, 1.27, 0.82],  # lower limits
        [1.89, 1.50, 0.87],  # upper limits
    ]

    colors = [C_Q8, C_Q8, C_Q8]  # Every color is green now

    bars = ax.bar(x, y, color=colors, width=0.5, edgecolor="none")

    # Add error bars manually
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4, elinewidth=1.2)

    # Annotate heights above the upper limits of error bars to avoid overlap
    for i, bar in enumerate(bars):
        h = bar.get_height()
        upper_val = h + yerr[1][i]
        ax.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, upper_val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    ax.set_ylabel("SysErr (%)")
    ax.set_title("(b) System errors with 95% CI")
    ax.set_ylim(0, 9)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig(fig, "fig_rq2b")


# =========================================================================
# fig_rq3b: Pareto frontier
# =========================================================================
def make_rq3b():
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter

    # Muted academic palette, matching Fig. 2 / Fig. 3 of the paper.
    P_GREEN = "#7cbf7c"  # F16/BF16
    P_MAUVE = "#bda4d4"  # Q8_0
    P_BLUE = "#9fc0e8"  # Q4_K_M
    P_EDGE = "#2b2b2b"  # Pareto outline / frontier line
    P_NEUT = "#9a9a9a"  # neutral grey for legend keys

    # Visual encoding (so no point needs a hand-placed text label):
    #   colour  -> precision      shape -> model family   size -> model scale
    QUANT_COLOR = {"F16": P_GREEN, "Q8": P_MAUVE, "Q4": P_BLUE}
    FAMILY_MARK = {"DeepSeek": "o", "Ministral": "s", "Qwen": "^"}
    SCALE_SIZE = {"small": 22, "large": 100}

    # --- Data: (energy kJ/task, success %, family, precision, scale, pareto) -
    points = [
        # F16/BF16
        (8.85, 0.17, "DeepSeek", "F16", "small", False),
        (10.52, 3.63, "DeepSeek", "F16", "large", False),
        (4.93, 15.86, "Ministral", "F16", "small", False),
        (5.13, 28.07, "Ministral", "F16", "large", False),
        (29.75, 43.34, "Qwen", "F16", "small", False),
        (4.15, 21.91, "Qwen", "F16", "large", False),
        # Q8_0
        (7.75, 0.17, "DeepSeek", "Q8", "small", False),
        (8.11, 3.49, "DeepSeek", "Q8", "large", False),
        (3.23, 17.56, "Ministral", "Q8", "small", False),
        (3.73, 29.18, "Ministral", "Q8", "large", False),
        (3.19, 20.95, "Qwen", "Q8", "large", False),
        # Q4_K_M
        (5.97, 0.58, "DeepSeek", "Q4", "small", False),
        (6.81, 2.78, "DeepSeek", "Q4", "large", False),
        (2.77, 18.82, "Qwen", "Q4", "large", False),
        # Pareto-optimal (the frontier)
        (2.61, 21.56, "Ministral", "Q4", "small", True),
        (3.02, 29.61, "Ministral", "Q4", "large", True),
        (17.57, 42.69, "Qwen", "Q4", "small", True),
        (20.05, 43.49, "Qwen", "Q8", "small", True),
    ]

    fig, ax = plt.subplots(figsize=(5.0, 3.1))

    # --- Axes setup ---------------------------------------------------------
    ax.set_xscale("log")
    ax.set_xlim(2.2, 36.0)
    ax.set_ylim(0, 50)
    ax.set_xticks([2.5, 4, 7, 10, 20, 30])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Energy/task (kJ, log scale)")
    ax.set_ylabel("Success Rate (%)")
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Pareto frontier line -----------------------------------------------
    frontier = sorted([p for p in points if p[5]], key=lambda p: p[0])
    ax.plot(
        [p[0] for p in frontier],
        [p[1] for p in frontier],
        color=P_EDGE,
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )

    # --- All configurations (Pareto = dark outline, dominated = none) -------
    for x, y, family, quant, scale, is_pareto in points:
        ax.scatter(
            [x],
            [y],
            marker=FAMILY_MARK[family],
            c=QUANT_COLOR[quant],
            s=SCALE_SIZE[scale],
            edgecolor=(P_EDGE if is_pareto else "none"),
            linewidth=(1.2 if is_pareto else 0),
            alpha=(1.0 if is_pareto else 0.8),
            zorder=(5 if is_pareto else 3),
        )

    # --- Single boxed legend (right): family=shape, precision=colour, --------
    #     size=scale, dark outline=Pareto. Bold rows act as group headers.
    def hdr(text):
        return Line2D([], [], marker="None", linestyle="None", label=text)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor="w",
            markeredgecolor=P_EDGE,
            markeredgewidth=1.2,
            markersize=7,
            linestyle="None",
            label="Pareto-optimal",
        ),
        hdr("Family"),
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=P_NEUT,
            markersize=6,
            linestyle="None",
            label="DeepSeek-R1",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color="w",
            markerfacecolor=P_NEUT,
            markersize=6,
            linestyle="None",
            label="Ministral-3",
        ),
        Line2D(
            [],
            [],
            marker="^",
            color="w",
            markerfacecolor=P_NEUT,
            markersize=6,
            linestyle="None",
            label="Qwen3",
        ),
        hdr("Precision"),
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=QUANT_COLOR["F16"],
            markersize=6,
            linestyle="None",
            label="16-bit",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=QUANT_COLOR["Q8"],
            markersize=6,
            linestyle="None",
            label="8-bit",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=QUANT_COLOR["Q4"],
            markersize=6,
            linestyle="None",
            label="4-bit",
        ),
        hdr("Scale"),
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=P_NEUT,
            markersize=3.5,
            linestyle="None",
            label="Smaller model",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=P_NEUT,
            markersize=10,
            linestyle="None",
            label="Larger model",
        ),
    ]

    leg = ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=7.0,
        handletextpad=0.4,
        labelspacing=0.45,
        borderpad=0.8,
    )
    leg.get_frame().set_edgecolor("#bbbbbb")
    leg.get_frame().set_linewidth(0.7)
    leg.get_frame().set_facecolor("white")
    # Bold the group-header rows.
    texts = leg.get_texts()
    for i in (1, 5, 9):
        texts[i].set_fontweight("bold")

    fig.subplots_adjust(right=0.72)
    save_fig(fig, "fig_rq3b")


def main():
    make_rq1a()
    make_rq1b()
    make_rq1c()
    make_rq2a()
    make_rq2b()
    make_rq3b()


if __name__ == "__main__":
    main()
