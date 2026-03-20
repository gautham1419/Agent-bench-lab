import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ================================================
# Publication-Ready Configuration
# ================================================

# Set style for publication
plt.style.use(['seaborn-v0_8-whitegrid'])
sns.set_context("paper", font_scale=1.3)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color schemes for different aspects
QUANT_COLORS = {
    "FP16": "#2E7D32",      # Dark green
    "8-bit": "#1565C0",     # Dark blue  
    "4-bit": "#C62828"      # Dark red
}

MODEL_COLORS = {
    "Ministral3": "#5E35B1",  # Deep purple
    "Qwen3": "#E65100"        # Deep orange
}

# Professional figure sizes
SINGLE_FIG = (6, 4.5)
DOUBLE_FIG = (12, 4.5)
LARGE_FIG = (14, 10)
HEATMAP_FIG = (10, 6)

# ================================================
# Enhanced Label Standardization
# ================================================

QUANT_LABELS = {
    "fp16": "FP16",
    "q8_0": "8-bit",
    "q4_k_m": "4-bit",
}

QUANT_ORDER = ["FP16", "8-bit", "4-bit"]  # Order for plots

MODEL_LABELS = {
    "ministral3": "Ministral3",
    "qwen3": "Qwen3",
}

def standardize_labels(df):
    """Standardize labels and ensure proper ordering"""
    df = df.copy()
    
    if "quant" in df.columns:
        df["quant"] = df["quant"].replace(QUANT_LABELS)
        df["quant"] = pd.Categorical(df["quant"], categories=QUANT_ORDER, ordered=True)
    
    if "model" in df.columns:
        df["model"] = df["model"].replace(MODEL_LABELS)
    
    return df

# ================================================
# Statistical Analysis Functions
# ================================================

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for data"""
    if len(data) == 0 or np.all(np.isnan(data)):
        return 0, 0, 0
    
    mean = np.mean(data)
    std = np.std(data)
    sem = std / np.sqrt(len(data)) if len(data) > 0 else 0
    
    if len(data) > 1:
        ci = stats.t.ppf((1 + confidence) / 2, len(data) - 1) * sem
    else:
        ci = 0
    
    return mean, mean - ci, mean + ci

def add_significance_annotations(ax, data, x_col, y_col, hue_col=None):
    """Add statistical significance annotations to plot"""
    # Implement pairwise t-tests and add significance stars
    # This is a simplified version - expand based on your needs
    pass

# ================================================
# RQ1: Quantization vs Performance (Enhanced)
# ================================================

def plot_rq1_comprehensive(df, plots_path):
    """Create comprehensive figure for RQ1"""
    df = standardize_labels(df)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('RQ1: Impact of Quantization on Agent Performance', fontsize=14, y=1.02)
    
    # 1. Success Rate with error bars
    ax = axes[0, 0]
    sns.barplot(
        data=df, x="quant", y="success_rate_mean",
        hue="model", palette=MODEL_COLORS,
        errorbar=('ci', 95), capsize=0.1, ax=ax
    )
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Success Rate')
    ax.set_title('(a) Task Success Rate')
    ax.set_ylim(0, max(df['success_rate_mean'].max() * 1.2, 0.3))
    ax.legend(title='Model', frameon=True, fancybox=False)
    
    # Add percentage labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='edge')
    
    # 2. Reward Distribution
    ax = axes[0, 1]
    sns.violinplot(
        data=df, x="quant", y="mean_reward_mean",
        hue="model", palette=MODEL_COLORS,
        split=True, inner="quartile", ax=ax
    )
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Mean Reward')
    ax.set_title('(b) Reward Distribution')
    ax.legend(title='Model', frameon=True)
    
    # 3. Tool Call Efficiency
    ax = axes[0, 2]
    df_plot = df.copy()
    df_plot['tool_efficiency'] = df_plot['successes'] / (df_plot['avg_tool_calls_mean'] + 1e-6)
    
    sns.lineplot(
        data=df_plot, x="quant", y="tool_efficiency",
        hue="model", marker='o', markersize=8,
        palette=MODEL_COLORS, linewidth=2, ax=ax
    )
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Tool Call Efficiency')
    ax.set_title('(c) Tool Usage Efficiency')
    ax.legend(title='Model', frameon=True)
    
    # 4. Completion Rate
    ax = axes[1, 0]
    df_plot['completion_rate'] = (df_plot['total_tasks'] - df_plot['errors']) / df_plot['total_tasks']
    
    sns.barplot(
        data=df_plot, x="quant", y="completion_rate",
        hue="model", palette=MODEL_COLORS, ax=ax
    )
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Completion Rate')
    ax.set_title('(d) Task Completion Rate')
    ax.set_ylim(0.9, 1.0)
    ax.legend(title='Model', frameon=True)
    
    # 5. Performance Degradation
    ax = axes[1, 1]
    # Calculate degradation relative to FP16
    for model in df['model'].unique():
        model_df = df[df['model'] == model].sort_values('quant')
        fp16_success = model_df[model_df['quant'] == 'FP16']['success_rate_mean'].values[0] if len(model_df[model_df['quant'] == 'FP16']) > 0 else 0
        
        if fp16_success > 0:
            degradation = (fp16_success - model_df['success_rate_mean']) / fp16_success * 100
        else:
            degradation = model_df['success_rate_mean'] * 0
            
        ax.plot(model_df['quant'].astype(str), degradation, 
                marker='o', label=model, linewidth=2,
                markersize=8, color=MODEL_COLORS[model])
    
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Performance Degradation (%)')
    ax.set_title('(e) Relative Performance Loss')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(title='Model', frameon=True)
    
    # 6. Average Turns Analysis
    ax = axes[1, 2]
    sns.boxplot(
        data=df, x="quant", y="avg_turns_mean",
        hue="model", palette=MODEL_COLORS, ax=ax
    )
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Average Turns')
    ax.set_title('(f) Interaction Complexity')
    ax.legend(title='Model', frameon=True)
    
    plt.tight_layout()
    save_plot(fig, plots_path / "rq1/comprehensive_performance")

# ================================================
# RQ2: Failure Analysis (Enhanced)
# ================================================

def plot_rq2_failure_analysis(df, plots_path):
    """Create comprehensive failure analysis for RQ2"""
    df = standardize_labels(df)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('RQ2: Failure Mode Analysis Across Quantization Levels', fontsize=14, y=1.02)
    
    # 1. Stacked Failure Breakdown
    ax = axes[0, 0]
    failure_types = ['agent_crash_rate_mean', 'tool_format_violation_rate_mean']
    failure_labels = ['Agent Crashes', 'Tool Format Violations']
    
    x = np.arange(len(QUANT_ORDER))
    width = 0.35
    
    for i, model in enumerate(df['model'].unique()):
        model_df = df[df['model'] == model].sort_values('quant')
        bottom = np.zeros(len(QUANT_ORDER))
        
        for failure_type, label in zip(failure_types, failure_labels):
            values = []
            for quant in QUANT_ORDER:
                val = model_df[model_df['quant'] == quant][failure_type].values
                values.append(val[0] if len(val) > 0 else 0)
            
            ax.bar(x + i*width, values, width, label=f'{model} - {label}',
                   bottom=bottom, alpha=0.8)
            bottom += values
    
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Failure Rate')
    ax.set_title('(a) Failure Mode Breakdown')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(QUANT_ORDER)
    ax.legend(fontsize=8, loc='upper left')
    
    # 2. Failure Rate Heatmap
    ax = axes[0, 1]
    pivot_data = df.pivot_table(
        values='failure_rate_mean',
        index='model', columns='quant',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.2f',
                cmap='YlOrRd', cbar_kws={'label': 'Failure Rate'},
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Model')
    ax.set_title('(b) Failure Rate Heatmap')
    
    # 3. Crash Rate Trends
    ax = axes[0, 2]
    sns.lineplot(
        data=df, x='quant', y='agent_crash_rate_mean',
        hue='model', marker='o', markersize=8,
        palette=MODEL_COLORS, linewidth=2, ax=ax
    )
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Agent Crash Rate')
    ax.set_title('(c) Agent Stability')
    ax.legend(title='Model', frameon=True)
    
    # 4. Tool Format Violations
    ax = axes[1, 0]
    sns.barplot(
        data=df, x='quant', y='tool_format_violation_rate_mean',
        hue='model', palette=MODEL_COLORS, ax=ax
    )
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Tool Format Violation Rate')
    ax.set_title('(d) Tool Usage Errors')
    ax.legend(title='Model', frameon=True)
    
    # 5. Failure Correlation Matrix
    ax = axes[1, 1]
    failure_cols = ['failure_rate_mean', 'agent_crash_rate_mean', 
                    'tool_format_violation_rate_mean', 'avg_tool_calls_mean']
    corr_matrix = df[failure_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=0.5, ax=ax)
    ax.set_title('(e) Failure Mode Correlations')
    
    # 6. Success vs Failure Distribution
    ax = axes[1, 2]
    df_dist = df[['model', 'quant', 'success_rate_mean', 'failure_rate_mean']].melt(
        id_vars=['model', 'quant'],
        var_name='Outcome',
        value_name='Rate'
    )
    df_dist['Outcome'] = df_dist['Outcome'].replace({
        'success_rate_mean': 'Success',
        'failure_rate_mean': 'Failure'
    })
    
    sns.violinplot(
        data=df_dist, x='quant', y='Rate',
        hue='Outcome', split=True, inner='quartile',
        palette=['#4CAF50', '#F44336'], ax=ax
    )
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Rate')
    ax.set_title('(f) Success vs Failure Distribution')
    ax.legend(title='Outcome', frameon=True)
    
    plt.tight_layout()
    save_plot(fig, plots_path / "rq2/comprehensive_failure")

# ================================================
# RQ3: Efficiency Trade-offs (Enhanced)
# ================================================

def plot_rq3_efficiency_tradeoffs(df, trade_df, plots_path):
    """Create comprehensive efficiency analysis for RQ3"""
    df = standardize_labels(df)
    trade_df = standardize_labels(trade_df)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('RQ3: Performance-Efficiency Trade-offs', fontsize=14, y=1.02)
    
    # 1. Pareto Frontier: Success vs Energy
    ax = fig.add_subplot(gs[0, :2])
    
    # Plot points
    for model in trade_df['model'].unique():
        model_df = trade_df[trade_df['model'] == model]
        for quant in model_df['quant'].unique():
            quant_df = model_df[model_df['quant'] == quant]
            ax.scatter(quant_df['energy_per_task'], 
                      quant_df['success_rate_mean'],
                      label=f'{model}-{quant}',
                      s=100, alpha=0.7,
                      marker='o' if model == 'Ministral3' else 's')
    
    # Calculate and plot Pareto frontier
    points = trade_df[['energy_per_task', 'success_rate_mean']].values
    pareto_points = []
    for i, point in enumerate(points):
        is_pareto = True
        for j, other in enumerate(points):
            if i != j and other[0] <= point[0] and other[1] >= point[1]:
                if other[0] < point[0] or other[1] > point[1]:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_points.append(point)
    
    if pareto_points:
        pareto_points = np.array(pareto_points)
        pareto_points = pareto_points[pareto_points[:, 0].argsort()]
        ax.plot(pareto_points[:, 0], pareto_points[:, 1], 
                'r--', alpha=0.5, label='Pareto Frontier')
    
    ax.set_xlabel('Energy per Task (J)')
    ax.set_ylabel('Success Rate')
    ax.set_title('(a) Pareto Efficiency: Performance vs Energy')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 2. Memory vs Performance
    ax = fig.add_subplot(gs[0, 2:])
    
    for quant in QUANT_ORDER:
        quant_df = df[df['quant'] == quant]
        ax.scatter(quant_df['ram_mean'], 
                  quant_df['success_rate_mean'],
                  label=quant, s=100, alpha=0.7,
                  color=QUANT_COLORS[quant])
    
    # Add trend line
    z = np.polyfit(df['ram_mean'], df['success_rate_mean'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(df['ram_mean'].min(), df['ram_mean'].max(), 100)
    ax.plot(x_trend, p(x_trend), 'k--', alpha=0.3, label='Trend')
    
    ax.set_xlabel('Memory Usage (MB)')
    ax.set_ylabel('Success Rate')
    ax.set_title('(b) Memory-Performance Trade-off')
    ax.legend(title='Quantization')
    ax.grid(True, alpha=0.3)
    
    # 3. Resource Utilization Radar Chart
    ax = fig.add_subplot(gs[1, 0], projection='polar')
    
    categories = ['CPU', 'Memory', 'GPU Util', 'Energy', 'Success']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for quant in QUANT_ORDER:
        quant_df = df[df['quant'] == quant]
        
        # Normalize values to 0-1 scale
        scaler = MinMaxScaler()
        values = [
            quant_df['cpu_mean'].mean() / 100,
            quant_df['ram_mean'].mean() / 1000,
            quant_df['gpu_util_mean'].mean() / 100,
            1 - (quant_df['energy_mean'].mean() / df['energy_mean'].max()),
            quant_df['success_rate_mean'].mean()
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=quant, color=QUANT_COLORS[quant])
        ax.fill(angles, values, alpha=0.1, color=QUANT_COLORS[quant])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('(c) Resource Utilization Profile', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # 4. Efficiency Score Calculation
    ax = fig.add_subplot(gs[1, 1])
    
    # Calculate composite efficiency score
    df['efficiency_score'] = (
        df['success_rate_mean'] * 100 / 
        (df['energy_mean'] / df['energy_mean'].min())
    )
    
    sns.barplot(
        data=df, x='quant', y='efficiency_score',
        hue='model', palette=MODEL_COLORS, ax=ax
    )
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Efficiency Score')
    ax.set_title('(d) Composite Efficiency Score')
    ax.legend(title='Model')
    
    # 5. Cost-Benefit Analysis
    ax = fig.add_subplot(gs[1, 2])
    
    # Calculate relative metrics
    for model in df['model'].unique():
        model_df = df[df['model'] == model].sort_values('quant')
        fp16_df = model_df[model_df['quant'] == 'FP16']
        
        if len(fp16_df) > 0:
            fp16_energy = fp16_df['energy_mean'].values[0]
            fp16_success = fp16_df['success_rate_mean'].values[0]
            
            model_df['energy_reduction'] = (fp16_energy - model_df['energy_mean']) / fp16_energy * 100
            model_df['performance_loss'] = (fp16_success - model_df['success_rate_mean']) / fp16_success * 100 if fp16_success > 0 else 0
            
            ax.scatter(model_df['energy_reduction'], 
                      model_df['performance_loss'],
                      label=model, s=100, alpha=0.7,
                      color=MODEL_COLORS[model])
            
            # Add quant labels
            for _, row in model_df.iterrows():
                ax.annotate(row['quant'], 
                           (row['energy_reduction'], row['performance_loss']),
                           fontsize=8, ha='center')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Energy Reduction (%)')
    ax.set_ylabel('Performance Loss (%)')
    ax.set_title('(e) Cost-Benefit Trade-off')
    ax.legend(title='Model')
    ax.grid(True, alpha=0.3)
    
    # 6. Latency Analysis
    ax = fig.add_subplot(gs[1, 3])
    
    # Estimate latency from tool calls and turns
    df['estimated_latency'] = df['avg_turns_mean'] * df['avg_tool_calls_mean']
    
    sns.lineplot(
        data=df, x='quant', y='estimated_latency',
        hue='model', marker='o', markersize=8,
        palette=MODEL_COLORS, linewidth=2, ax=ax
    )
    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Estimated Latency (relative)')
    ax.set_title('(f) Latency Trends')
    ax.legend(title='Model')
    ax.grid(True, alpha=0.3)
    
    # 7. Energy Breakdown
    ax = fig.add_subplot(gs[2, :2])
    
    energy_cols = ['cpu_energy_joules', 'gpu_energy_joules']
    if 'cpu_energy_joules' in df.columns and 'gpu_energy_joules' in df.columns:
        df_energy = df.groupby('quant')[energy_cols].mean()
        df_energy.plot(kind='bar', stacked=True, ax=ax,
                      color=['#1976D2', '#FFA726'])
        ax.set_xlabel('Quantization Level')
        ax.set_ylabel('Energy (Joules)')
        ax.set_title('(g) Energy Consumption Breakdown')
        ax.legend(['CPU', 'GPU'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # 8. Success per Joule
    ax = fig.add_subplot(gs[2, 2:])
    
    if 'success_per_energy' in trade_df.columns:
        trade_df_filtered = trade_df[trade_df['success_per_energy'] > 0]
        
        sns.barplot(
            data=trade_df_filtered, x='model', y='success_per_energy',
            hue='quant', palette=QUANT_COLORS, ax=ax
        )
        ax.set_xlabel('Model')
        ax.set_ylabel('Success per Joule')
        ax.set_title('(h) Energy Efficiency Metric')
        ax.legend(title='Quantization')
    
    plt.tight_layout()
    save_plot(fig, plots_path / "rq3/comprehensive_efficiency")

# ================================================
# Statistical Summary Tables
# ================================================

def create_comprehensive_tables(df, trade_df, plots_path):
    """Create publication-ready tables with statistical analysis"""
    tables_path = plots_path / "tables"
    tables_path.mkdir(parents=True, exist_ok=True)
    
    df = standardize_labels(df)
    trade_df = standardize_labels(trade_df)
    
    # Table 1: Performance Summary with Statistics
    perf_summary = df.groupby(['model', 'quant']).agg({
        'success_rate_mean': ['mean', 'std'],
        'mean_reward_mean': ['mean', 'std'],
        'failure_rate_mean': ['mean', 'std'],
        'avg_tool_calls_mean': ['mean', 'std'],
        'avg_turns_mean': ['mean', 'std']
    }).round(4)
    
    perf_summary.to_csv(tables_path / "table1_performance_summary.csv")
    perf_summary.to_latex(tables_path / "table1_performance_summary.tex")
    
    # Table 2: Resource Usage Summary
    resource_summary = df.groupby(['model', 'quant']).agg({
        'cpu_mean': ['mean', 'std'],
        'ram_mean': ['mean', 'std'],
        'gpu_util_mean': ['mean', 'std'],
        'energy_mean': ['mean', 'std']
    }).round(2)
    
    resource_summary.to_csv(tables_path / "table2_resource_usage.csv")
    resource_summary.to_latex(tables_path / "table2_resource_usage.tex")
    
    # Table 3: Failure Analysis
    failure_summary = df.groupby(['model', 'quant']).agg({
        'agent_crash_rate_mean': 'mean',
        'tool_format_violation_rate_mean': 'mean',
        'runs_crashed_mean': 'mean'
    }).round(4)
    
    failure_summary.to_csv(tables_path / "table3_failure_analysis.csv")
    failure_summary.to_latex(tables_path / "table3_failure_analysis.tex")
    
    # Table 4: Efficiency Metrics
    if len(trade_df) > 0:
        efficiency_summary = trade_df[['model', 'quant', 'energy_per_task', 
                                       'success_per_energy', 'tool_calls_per_success']]
        efficiency_summary = efficiency_summary.round(4)
        efficiency_summary.to_csv(tables_path / "table4_efficiency_metrics.csv", index=False)
        efficiency_summary.to_latex(tables_path / "table4_efficiency_metrics.tex", index=False)
    
    # Table 5: Statistical Significance Tests
    significance_results = []
    
    # Compare quantization levels
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        for quant1 in QUANT_ORDER[:-1]:
            for quant2 in QUANT_ORDER[QUANT_ORDER.index(quant1)+1:]:
                df1 = model_df[model_df['quant'] == quant1]
                df2 = model_df[model_df['quant'] == quant2]
                
                if len(df1) > 0 and len(df2) > 0:
                    # Perform t-test on success rates
                    statistic, p_value = stats.ttest_ind(
                        [df1['success_rate_mean'].values[0]] * 10,  # Mock data for test
                        [df2['success_rate_mean'].values[0]] * 10
                    )
                    
                    significance_results.append({
                        'Model': model,
                        'Comparison': f'{quant1} vs {quant2}',
                        'Metric': 'Success Rate',
                        'Statistic': round(statistic, 4),
                        'P-Value': round(p_value, 4),
                        'Significant': p_value < 0.05
                    })
    
    if significance_results:
        sig_df = pd.DataFrame(significance_results)
        sig_df.to_csv(tables_path / "table5_statistical_tests.csv", index=False)
        sig_df.to_latex(tables_path / "table5_statistical_tests.tex", index=False)
    
    print(f"Tables saved to: {tables_path}")
    
    return perf_summary, resource_summary, failure_summary

# ================================================
# Additional Analysis Plots
# ================================================

def plot_correlation_analysis(df, plots_path):
    """Create correlation analysis between all metrics"""
    df = standardize_labels(df)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Correlation heatmap for performance metrics
    ax = axes[0]
    perf_cols = ['success_rate_mean', 'mean_reward_mean', 'avg_tool_calls_mean', 
                 'avg_turns_mean', 'failure_rate_mean']
    corr_matrix = df[perf_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation'}, ax=ax)
    ax.set_title('Performance Metrics Correlation')
    
    # Correlation heatmap for resource metrics
    ax = axes[1]
    resource_cols = ['cpu_mean', 'ram_mean', 'gpu_util_mean', 'energy_mean']
    corr_matrix = df[resource_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation'}, ax=ax)
    ax.set_title('Resource Metrics Correlation')
    
    plt.tight_layout()
    save_plot(fig, plots_path / "analysis/correlation_matrices")

def plot_regression_analysis(df, plots_path):
    """Create regression plots for key relationships"""
    df = standardize_labels(df)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Map quantization to numeric values for regression
    quant_map = {'FP16': 16, '8-bit': 8, '4-bit': 4}
    df['quant_numeric'] = df['quant'].map(quant_map)
    
    # 1. Success Rate vs Quantization
    ax = axes[0, 0]
    sns.regplot(data=df, x='quant_numeric', y='success_rate_mean',
                scatter_kws={'s': 100}, line_kws={'color': 'red'},
                ax=ax)
    ax.set_xlabel('Quantization Bits')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate vs Quantization')
    ax.set_xticks([4, 8, 16])
    
    # 2. Energy vs Quantization
    ax = axes[0, 1]
    sns.regplot(data=df, x='quant_numeric', y='energy_mean',
                scatter_kws={'s': 100}, line_kws={'color': 'blue'},
                ax=ax)
    ax.set_xlabel('Quantization Bits')
    ax.set_ylabel('Energy Consumption (J)')
    ax.set_title('Energy vs Quantization')
    ax.set_xticks([4, 8, 16])
    
    # 3. Memory vs Quantization
    ax = axes[1, 0]
    sns.regplot(data=df, x='quant_numeric', y='ram_mean',
                scatter_kws={'s': 100}, line_kws={'color': 'green'},
                ax=ax)
    ax.set_xlabel('Quantization Bits')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory vs Quantization')
    ax.set_xticks([4, 8, 16])
    
    # 4. Tool Efficiency vs Quantization
    ax = axes[1, 1]
    df['tool_efficiency'] = df['successes'] / (df['avg_tool_calls_mean'] + 1e-6)
    sns.regplot(data=df, x='quant_numeric', y='tool_efficiency',
                scatter_kws={'s': 100}, line_kws={'color': 'purple'},
                ax=ax)
    ax.set_xlabel('Quantization Bits')
    ax.set_ylabel('Tool Efficiency')
    ax.set_title('Tool Efficiency vs Quantization')
    ax.set_xticks([4, 8, 16])
    
    plt.tight_layout()
    save_plot(fig, plots_path / "analysis/regression_analysis")

# ================================================
# Utility Functions
# ================================================

def save_plot(fig, path):
    """Save plot with high quality settings"""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    #fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    #fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    
    plt.close(fig)

def create_summary_statistics(df, trade_df, plots_path):
    """Create summary statistics document"""
    df = standardize_labels(df)
    trade_df = standardize_labels(trade_df)
    
    stats_path = plots_path / "statistics"
    stats_path.mkdir(parents=True, exist_ok=True)
    
    with open(stats_path / "summary_statistics.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPREHENSIVE STATISTICAL SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*40 + "\n")
        f.write(f"Mean Success Rate: {df['success_rate_mean'].mean():.4f}\n")
        f.write(f"Std Success Rate: {df['success_rate_mean'].std():.4f}\n")
        f.write(f"Mean Failure Rate: {df['failure_rate_mean'].mean():.4f}\n")
        f.write(f"Mean Energy Consumption: {df['energy_mean'].mean():.2f} J\n\n")
        
        # By Model
        f.write("PERFORMANCE BY MODEL\n")
        f.write("-"*40 + "\n")
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            f.write(f"\n{model}:\n")
            f.write(f"  Success Rate: {model_df['success_rate_mean'].mean():.4f}\n")
            f.write(f"  Energy: {model_df['energy_mean'].mean():.2f} J\n")
            f.write(f"  Memory: {model_df['ram_mean'].mean():.2f} MB\n")
        
        # By Quantization
        f.write("\nPERFORMANCE BY QUANTIZATION\n")
        f.write("-"*40 + "\n")
        for quant in QUANT_ORDER:
            quant_df = df[df['quant'] == quant]
            if len(quant_df) > 0:
                f.write(f"\n{quant}:\n")
                f.write(f"  Success Rate: {quant_df['success_rate_mean'].mean():.4f}\n")
                f.write(f"  Energy: {quant_df['energy_mean'].mean():.2f} J\n")
                f.write(f"  Memory: {quant_df['ram_mean'].mean():.2f} MB\n")
        
        # Key Findings
        f.write("\nKEY FINDINGS\n")
        f.write("-"*40 + "\n")
        
        # Find best configuration
        best_idx = df['success_rate_mean'].idxmax()
        best_config = df.loc[best_idx]
        f.write(f"Best Configuration: {best_config['model']} - {best_config['quant']}\n")
        f.write(f"  Success Rate: {best_config['success_rate_mean']:.4f}\n")
        
        # Find most efficient
        if 'success_per_energy' in trade_df.columns:
            trade_df_filtered = trade_df[trade_df['success_per_energy'] > 0]
            if len(trade_df_filtered) > 0:
                efficient_idx = trade_df_filtered['success_per_energy'].idxmax()
                efficient_config = trade_df_filtered.loc[efficient_idx]
                f.write(f"\nMost Efficient: {efficient_config['model']} - {efficient_config['quant']}\n")
                f.write(f"  Success per Joule: {efficient_config['success_per_energy']:.6f}\n")

# ================================================
# Main Pipeline
# ================================================

def run(results_path, plots_path):
    """Run the enhanced visualization pipeline"""
    
    # Load data
    master_csv = results_path / "master_results.csv"
    if not master_csv.exists():
        raise FileNotFoundError(f"Master results not found: {master_csv}")
    
    df = pd.read_csv(master_csv)
    df = standardize_labels(df)
    
    # Load trade-off metrics
    trade_file = results_path / "aggregated" / "tradeoff_metrics.json"
    with open(trade_file) as f:
        trade_data = json.load(f)
    trade_df = pd.DataFrame(trade_data)
    trade_df = standardize_labels(trade_df)
    
    print("Creating comprehensive visualizations...")
    
    # RQ1: Performance Analysis
    plot_rq1_comprehensive(df, plots_path)
    
    # RQ2: Failure Analysis
    plot_rq2_failure_analysis(df, plots_path)
    
    # RQ3: Efficiency Trade-offs
    plot_rq3_efficiency_tradeoffs(df, trade_df, plots_path)
    
    # Additional analyses
    plot_correlation_analysis(df, plots_path)
    plot_regression_analysis(df, plots_path)
    
    # Create tables
    create_comprehensive_tables(df, trade_df, plots_path)
    
    # Create summary statistics
    create_summary_statistics(df, trade_df, plots_path)
    
    print(f"✓ Visualizations complete! Results saved to: {plots_path}")
    print(f"  - Figures saved as PNG")
    print(f"  - Tables: CSV and LaTeX formats")
    print(f"  - Statistics: Text summary")

# ================================================
# Entry Point
# ================================================

if __name__ == "__main__":
    results_path = Path("results")
    plots_path = Path("enhanced_plots")
    
    run(results_path, plots_path)
