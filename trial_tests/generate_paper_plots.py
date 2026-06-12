import matplotlib.pyplot as plt
import numpy as np
import os

# Set global styles for premium academic look
plt.rcParams.update({
    'font.size': 8.0,
    'axes.labelsize': 8.0,
    'axes.titlesize': 9.0,
    'xtick.labelsize': 8.0,
    'ytick.labelsize': 8.0,
    'legend.fontsize': 8.0,
    'figure.titlesize': 9.0,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.7,
    'grid.color': '#e5e5e5',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

# Colors
C_BF16 = '#2563eb'  # Vivid Royal Blue
C_Q8 = '#10b981'    # Vibrant Emerald Green
C_Q4 = '#f97316'    # Punchy Tangerine Orange
C_RED = '#ef4444'   # Bright Coral Red
C_GREY = '#64748b'  # Cool Slate Grey

OUT_DIR = "docs"
os.makedirs(OUT_DIR, exist_ok=True)

def save_fig(fig, name):
    # Save PDF (vector) and PNG (raster)
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    png_double_path = os.path.join(OUT_DIR, f"{name}.png.png")
    
    fig.savefig(pdf_path, bbox_inches='tight', transparent=True)
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_double_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {name} (PDF, PNG)")

# =========================================================================
# fig_rq1a: Success by quantization
# =========================================================================
def make_rq1a():
    fig, ax = plt.subplots(figsize=(2.5, 2.3))
    x = ['F16/BF16', r'Q8$_0$', 'Q4_K_M']
    y = [18.8, 19.1, 19.3]
    colors = [C_BF16, C_Q8, C_Q4]
    
    bars = ax.bar(x, y, color=colors, width=0.5, edgecolor='none')
    
    # Annotate bar heights
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.0)
                    
    ax.set_ylabel('Success (%)')
    ax.set_title('(a) Success by quantization')
    ax.set_ylim(0, 28)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_fig(fig, "fig_rq1a")

# =========================================================================
# fig_rq1b: Success by family & size (grouped)
# =========================================================================
def make_rq1b():
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    
    families = ['DeepSeek', 'Ministral3', 'Qwen3']
    smaller_scale = [0.3, 18.3, 43.2] # 1.5B, 3B, 4B
    larger_scale = [3.3, 29.0, 20.6]  # 7B, 8B, 8B
    
    x = np.arange(len(families))
    width = 0.3
    
    rects1 = ax.bar(x - width/2, smaller_scale, width, label='Smaller Scale', color=C_Q4, edgecolor='none')
    rects2 = ax.bar(x + width/2, larger_scale, width, label='Larger Scale', color=C_Q8, edgecolor='none')
    
    # Annotate heights
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.0)
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.0)
    
    ax.set_ylabel('Success (%)')
    ax.set_title('(b) Success by family & size')
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylim(0, 52)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper left', frameon=True, edgecolor='none', facecolor='#ffffff', framealpha=0.8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_fig(fig, "fig_rq1b")

# =========================================================================
# fig_rq1c: Domain effect (LMM beta relative to ALFWorld)
# =========================================================================
def make_rq1c():
    fig, ax = plt.subplots(figsize=(2.5, 2.3))
    
    domains = ['ALFWorld', 'OS', 'DB', 'WebShop']
    betas = [0.0, 0.094, 0.166, 0.359]
    
    bars = ax.barh(domains, betas, color=C_BF16, height=0.5, edgecolor='none')
    
    # Annotate values
    for bar in bars:
        w = bar.get_width()
        # For ALFWorld (0.0), shift slightly right so text is visible
        x_pos = w + 0.01 if w > 0 else 0.01
        ax.annotate(f'+{w:.3f}' if w > 0 else 'Ref (0.00)', xy=(x_pos, bar.get_y() + bar.get_height()/2),
                    xytext=(0, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=8.0)
                    
    ax.set_xlabel(r'LMM $\hat{\beta}$')
    ax.set_title(r'(c) Domain effect ($\hat{\beta}$)')
    ax.set_xlim(0, 0.45)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_fig(fig, "fig_rq1c")

# =========================================================================
# fig_rq2a: Failure type proportions (grouped by type)
# =========================================================================
def make_rq2a():
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    
    # Failure types
    types = ['TLE', 'IF', 'IA', 'CF', 'TE']
    
    # Proportions for BF16, Q8_0, Q4_K_M
    bf16_prop = [21.5, 26.3, 9.2, 31.0, 6.9]
    q8_prop = [23.0, 25.4, 9.2, 31.8, 7.1]
    q4_prop = [25.6, 25.2, 9.3, 29.8, 6.7]
    
    x = np.arange(len(types))
    width = 0.25
    
    # Use proper labels without backslashes!
    ax.bar(x - width, bf16_prop, width, label='F16/BF16', color=C_BF16, edgecolor='none')
    ax.bar(x, q8_prop, width, label=r'Q8$_0$', color=C_Q8, edgecolor='none')
    ax.bar(x + width, q4_prop, width, label='Q4_K_M', color=C_Q4, edgecolor='none')
    
    ax.set_ylabel('Proportion (%)')
    ax.set_title('(a) Failure type proportions')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylim(0, 42)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper right', frameon=True, edgecolor='none')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_fig(fig, "fig_rq2a")

# =========================================================================
# fig_rq2b: System errors with 95% CI
# =========================================================================
def make_rq2b():
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    
    x = ['F16/BF16', r'Q8$_0$', 'Q4_K_M']
    y = [5.03, 3.61, 3.35]
    
    # Error bar calculations (lower error, upper error)
    # BF16: 5.03% (CI [3.37, 6.92]) -> lower=1.66, upper=1.89
    # Q8_0: 3.61% (CI [2.34, 5.11]) -> lower=1.27, upper=1.50
    # Q4: 3.35% (CI [2.53, 4.22]) -> lower=0.82, upper=0.87
    yerr = [
        [1.66, 1.27, 0.82], # lower limits
        [1.89, 1.50, 0.87]  # upper limits
    ]
    
    colors = [C_Q8, C_Q8, C_Q8]  # Every color is green now
    
    bars = ax.bar(x, y, color=colors, width=0.5, edgecolor='none')
    
    # Add error bars manually
    ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='black', capsize=4, elinewidth=1.2)
    
    # Annotate heights above the upper limits of error bars to avoid overlap
    for i, bar in enumerate(bars):
        h = bar.get_height()
        upper_val = h + yerr[1][i]
        ax.annotate(f'{h:.2f}%', xy=(bar.get_x() + bar.get_width()/2, upper_val),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.0)
                    
    ax.set_ylabel('SysErr (%)')
    ax.set_title('(b) System errors with 95% CI')
    ax.set_ylim(0, 9)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_fig(fig, "fig_rq2b")

# =========================================================================
# fig_rq3a: Stacked horizontal quadrant analysis (fixed legend overlap!)
# =========================================================================
def make_rq3a():
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    
    y = [r'F16/BF16$\rightarrow$Q4_K_M', r'F16/BF16$\rightarrow$Q8$_0$']
    
    # Proportions: Win-win, Trade-off, Inverse, Lose-lose
    win_win = [41.7, 66.7]
    trade_off = [50.0, 29.2]
    inverse = [8.3, 4.2]
    lose_lose = [0.0, 0.0]
    
    # Accumulate bar positions for stacking
    lefts = np.zeros(2)
    
    # Plot segments
    ax.barh(y, win_win, label='Win-win', color=C_Q8, height=0.4, edgecolor='none')
    lefts += win_win
    
    ax.barh(y, trade_off, left=lefts, label='Trade-off', color=C_Q4, height=0.4, edgecolor='none')
    lefts += trade_off
    
    ax.barh(y, inverse, left=lefts, label='Inverse', color=C_GREY, height=0.4, edgecolor='none')
    lefts += inverse
    
    ax.barh(y, lose_lose, left=lefts, label='Lose-lose', color=C_RED, height=0.4, edgecolor='none')
    
    ax.set_xlabel('% of n=24 pairs')
    ax.set_title('(a) Quadrant classification')
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    
    # Place legend BELOW the plot (or above in white space) to completely avoid overlap
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.38), ncol=4, frameon=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(FalseSpine := True) # matplotlib typo fix
    ax.spines['right'].set_visible(False)
    
    save_fig(fig, "fig_rq3a")

# =========================================================================
# fig_rq3b: Pareto frontier
# =========================================================================
def make_rq3b():
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    
    # Dominated BF16 points (Blue circles)
    bf16_points = [
        (8.85, 0.17),   # DS 1.5B BF16
        (10.52, 3.63),  # DS 7B BF16
        (4.93, 15.86),  # Ministral 3B BF16
        (5.13, 28.07),  # Ministral 8B BF16
        (29.75, 43.34), # Qwen 4B BF16
        (4.15, 21.91)   # Qwen 8B BF16
    ]
    
    # Dominated Q8_0 points (Green squares)
    q8_dominated_points = [
        (7.75, 0.17),   # DS 1.5B Q8
        (8.11, 3.49),   # DS 7B Q8
        (3.23, 17.56),  # Ministral 3B Q8
        (3.73, 29.18),  # Ministral 8B Q8
        (3.19, 20.95)   # Qwen 8B Q8
    ]
    
    # Dominated Q4_K_M points (Orange triangles)
    q4_dominated_points = [
        (5.97, 0.58),   # DS 1.5B Q4
        (6.81, 2.78),   # DS 7B Q4
        (2.77, 18.82)   # Qwen 8B Q4
    ]
    
    # Pareto-optimal points: Q4_K_M (Orange triangles, bold outline)
    pareto_q4_points = [
        (2.61, 21.56),  # Ministral 3B Q4
        (3.02, 29.61),  # Ministral 8B Q4
        (17.57, 42.69)  # Qwen 4B Q4
    ]
    
    # Pareto-optimal points: Q8_0 (Green squares, bold outline)
    pareto_q8_points = [
        (20.05, 43.49)  # Qwen 4B Q8
    ]
    
    # Separate x and y for plotting
    bf16_x, bf16_y = zip(*bf16_points)
    q8_dom_x, q8_dom_y = zip(*q8_dominated_points)
    q4_dom_x, q4_dom_y = zip(*q4_dominated_points)
    
    pareto_q4_x, pareto_q4_y = zip(*pareto_q4_points)
    pareto_q8_x, pareto_q8_y = zip(*pareto_q8_points)
    
    # Log scale on x-axis
    ax.set_xscale('log')
    ax.set_xticks([2.5, 3, 4, 5, 7, 10, 15, 20, 30])
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    
    # Plot dominated points
    ax.scatter(bf16_x, bf16_y, color=C_BF16, marker='o', s=45, alpha=0.6, label='F16/BF16 (dominated)', zorder=2, edgecolor='none')
    ax.scatter(q8_dom_x, q8_dom_y, color=C_Q8, marker='s', s=45, alpha=0.6, label='Q8$_0$ (dominated)', zorder=2, edgecolor='none')
    ax.scatter(q4_dom_x, q4_dom_y, color=C_Q4, marker='^', s=45, alpha=0.6, label='Q4_K_M (dominated)', zorder=2, edgecolor='none')
    
    # Plot Pareto-optimal points
    ax.scatter(pareto_q4_x, pareto_q4_y, color=C_Q4, marker='^', s=65, label='Pareto-optimal (Q4_K_M)', zorder=3, edgecolor='black', linewidth=1.5)
    ax.scatter(pareto_q8_x, pareto_q8_y, color=C_Q8, marker='s', s=65, label='Pareto-optimal (Q8$_0$)', zorder=3, edgecolor='black', linewidth=1.5)
    
    # Connect all Pareto frontier points with a dashed line
    all_pareto_points = pareto_q4_points + pareto_q8_points
    sorted_pareto = sorted(all_pareto_points, key=lambda pt: pt[0])
    px, py = zip(*sorted_pareto)
    ax.plot(px, py, color='#444444', linestyle='--', linewidth=1.2, zorder=1)
    
    # Label Pareto optimal configurations
    labels = {
        (2.61, 21.56): ('Min3-3B-Q4_K_M', -10, -12, 'right', 'top'),
        (3.02, 29.61): ('Min3-8B-Q4_K_M', -10, 12, 'right', 'bottom'),
        (17.57, 42.69): ('Qwen-4B-Q4_K_M', -10, 8, 'right', 'bottom'),
        (20.05, 43.49): ('Qwen-4B-Q8', 8, 4, 'left', 'bottom')
    }
    for pt, (label, x_off, y_off, ha, va) in labels.items():
        x_coord, y_coord = pt
        ax.annotate(label, xy=(x_coord, y_coord), xytext=(x_off, y_off),
                    textcoords='offset points', fontsize=8.0, fontweight='bold',
                    color='#333333', ha=ha, va=va)
    
    ax.set_xlabel('Energy/task (kJ)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('(b) Pareto frontier')
    
    # Format Y axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0f}%'))
    
    ax.set_xlim(0.8, 35.0)
    ax.set_ylim(0, 50)
    ax.grid(True, which='both')
    ax.set_axisbelow(True)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_BF16, markersize=7, alpha=0.6, label='F16/BF16 (dominated)', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_Q8, markersize=7, alpha=0.6, label='Q8$_0$ (dominated)', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=C_Q4, markersize=7, alpha=0.6, label='Q4_K_M (dominated)', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_Q8, markeredgecolor='black', markeredgewidth=1.2, markersize=8, label='Pareto (Q8$_0$)', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=C_Q4, markeredgecolor='black', markeredgewidth=1.2, markersize=8, label='Pareto (Q4_K_M)', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_fig(fig, "fig_rq3b")

def main():
    make_rq1a()
    make_rq1b()
    make_rq1c()
    make_rq2a()
    make_rq2b()
    make_rq3a()
    make_rq3b()

if __name__ == "__main__":
    main()
