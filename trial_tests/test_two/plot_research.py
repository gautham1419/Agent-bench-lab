import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
FIGSIZE = (8, 5)

QUANT_LABELS = {
    "fp16": "16-bit", "f16": "16-bit", "bf16": "16-bit",
    "q8_0": "8-bit", "q4_k_m": "4-bit"
}
QUANT_ORDER = ["16-bit", "8-bit", "4-bit"]

MODEL_COLORS = {
    "Ministral3 3B": "#9575CD",  
    "Ministral3 8B": "#512DA8",  
    "Qwen3 4B": "#FFB74D",       
    "Qwen3 8B": "#E65100",
    "DeepSeek R1 Qwen 1.5B": "#4DB6AC",
    "DeepSeek R1 Qwen 7B": "#004D40"
}

MODEL_MAP = {
    "ministral3": "Ministral3",
    "qwen3": "Qwen3",
    "deepseek-r1-qwen": "DeepSeek R1 Qwen"
}

def save_plot(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    records = []
    for d in data:
        row = {
            "model_raw": d["model"],
            "size": d["size"],
            "quant_raw": d["quant"],
            "domain": d["domain"],
            "agent_name": d["agent_name"]
        }
        row.update(d["metrics"])
        records.append(row)
        
    df = pd.DataFrame(records)
    df["quant"] = df["quant_raw"].replace(QUANT_LABELS)
    df["quant"] = pd.Categorical(df["quant"], categories=QUANT_ORDER, ordered=True)
    df["model_base"] = df["model_raw"].replace(MODEL_MAP)
    df["model"] = df["model_base"] + " " + df["size"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["gpu_energy_per_task"] = df["gpu_energy_joules"] / df["total_tasks"]
    df["cpu_energy_per_task"] = df["cpu_energy_joules"] / df["total_tasks"]
    return df

def plot_rq1_success_decay(df, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.lineplot(data=df, x="quant", y="success_rate", hue="model", marker="o", linewidth=2, palette=MODEL_COLORS, ax=ax)
    ax.set_title("RQ1: Task Success Decay vs Quantization")
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Success Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(fig, out_path / "rq1_success_decay")

def plot_rq1_wandering(df, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=df, x="quant", y="avg_turns_to_success", hue="model", palette=MODEL_COLORS, ax=ax)
    ax.set_title("RQ1: Wandering (Avg Turns Before Success)")
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Average Conversational Turns")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(fig, out_path / "rq1_wandering_turns")

def plot_rq2_failure_anatomy(df, out_path):
    failures = df[["model", "quant"]].copy()
    
    # Strictly map the Explicit 5 framework states isolated by the user
    failures["TLE (Task Limit Exceeded)"] = df["task_limit_reached"]
    failures["IA (Invalid Action)"] = df["agent_invalid_action"]
    failures["IF (Invalid Format)"] = df["invalid_format"]
    
    # Task Errors + raw JSON log errors all crash the specific instance
    failures["CLE (Framework Crash)"] = df["task_error"] + df["errors"]
    
    # Generic completions that were factually incorrect
    failures["Interaction Failure (False Answer)"] = df["completed_failure"]
    
    # We calculate the percentage composition of total errors
    failures["total_errors"] = (failures["TLE (Task Limit Exceeded)"] + 
                                failures["IA (Invalid Action)"] + 
                                failures["IF (Invalid Format)"] + 
                                failures["CLE (Framework Crash)"] + 
                                failures["Interaction Failure (False Answer)"])
                                
    failures = failures[failures["total_errors"] > 0].copy()
    
    for col in ["TLE (Task Limit Exceeded)", "IA (Invalid Action)", "IF (Invalid Format)", "CLE (Framework Crash)", "Interaction Failure (False Answer)"]:
        failures[f"{col} %"] = (failures[col] / failures["total_errors"]) * 100
        
    failures["Configuration"] = failures["model"] + " " + failures["quant"].astype(str)
    
    plot_df = failures.set_index("Configuration")[
        ["TLE (Task Limit Exceeded) %", "IA (Invalid Action) %", "IF (Invalid Format) %", "CLE (Framework Crash) %", "Interaction Failure (False Answer) %"]
    ]

    plot_df.columns  = ["TLE (Task Limit Exceeded) %", "IA (Invalid Action) %", "IF (Invalid Format) %", "CLE (Context Limit Exceeded) %", "Interaction Failure (False Answer) %"]
    
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    color_map = ["#FFB300", "#1E88E5", "#D81B60", "#455A64", "#00897B"]
    plot_df.plot(kind="bar", stacked=True, color=color_map, ax=ax)
    
    ax.set_title("RQ2: Explicit AgentBench Failure Anatomy by Model")
    ax.set_ylabel("Share of Total Failures (%)")
    ax.legend(title="Failure Taxonomy", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    
    save_plot(fig, out_path / "rq2_failure_anatomy")

def plot_rq3_pareto(df, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.scatterplot(data=df, x="energy_per_task", y="success_rate", hue="model", style="quant", s=150, palette=MODEL_COLORS, alpha=0.8, ax=ax)
    ax.set_title("RQ3: Scatter Plot (Energy vs Success)")
    ax.set_xlabel("Energy per Task (Joules)")
    ax.set_ylabel("Success Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(fig, out_path / "rq3_pareto_frontier")
    
def plot_rq3_true_cost(df, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=df, x="quant", y="energy_per_success", hue="model", palette=MODEL_COLORS, ax=ax)
    ax.set_title("RQ3: True Cost of a Successful Task")
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Energy (Joules) per Success")
    ax.set_yscale("log") 
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(fig, out_path / "rq3_true_energy_cost")

def plot_rq3_hardware_ceiling(df, out_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=df, x="quant", y="gpu_mem_peak", hue="model", palette=MODEL_COLORS, ax=ax)
    ax.set_title("RQ3: Hardware Barrier to Entry (Peak VRAM)")
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel("Peak VRAM Allocated (MB)")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(fig, out_path / "rq3_hardware_vram_peak")

def plot_rq3_energy_composition(df, out_path):
    composition = df[["model", "quant", "gpu_energy_per_task", "cpu_energy_per_task"]].copy()
    composition["Configuration"] = composition["model"] + " " + composition["quant"].astype(str)
    
    plot_df = composition.set_index("Configuration")[["gpu_energy_per_task", "cpu_energy_per_task"]]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(kind="bar", stacked=True, color=["#4CAF50", "#2196F3"], ax=ax)
    
    ax.set_title("RQ3: Energy Draw Composition (GPU vs CPU per Task)")
    ax.set_ylabel("Average Energy Consumed (Joules)")
    ax.legend(["Dedicated GPU Energy", "System CPU Energy"], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    
    save_plot(fig, out_path / "rq3_energy_composition_split")

def main():
    json_path = Path("computed_results/mean/master_mean.json")
    if not json_path.exists():
        print(f"Error: {json_path} not found! Please run calculate_metrics.py first.")
        return
        
    df = load_data(json_path)
    base_out = Path("research_plots_v2")
    domains = [None] + list(df["domain"].dropna().unique())
    
    print(f"Mapping the strictly defined native AgentBench definitions for {len(df)} environment runs...")
    
    for domain in domains:
        if domain is None:
            slice_df = df
            out_path = base_out / "overall"
        else:
            slice_df = df[df["domain"] == domain].copy()
            out_path = base_out / domain
            
        if slice_df.empty:
            continue
            
        plot_rq1_success_decay(slice_df, out_path)
        plot_rq1_wandering(slice_df, out_path)
        
        if slice_df["failures"].sum() + slice_df["errors"].sum() > 0:
            plot_rq2_failure_anatomy(slice_df, out_path)
            
        plot_rq3_pareto(slice_df, out_path)
        plot_rq3_true_cost(slice_df, out_path)
        plot_rq3_hardware_ceiling(slice_df, out_path)
        plot_rq3_energy_composition(slice_df, out_path)
        
    print(f"\nSuccessfully populated exact {base_out}/ graphs utilizing explicit AgentBench nomenclature!")

if __name__ == "__main__":
    main()
