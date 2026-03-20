import os
import json
import matplotlib.pyplot as plt
import numpy as np
from test_os import summarize



# ---- paste your load_jsonl(), summarize() here ----

def savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print("saved:", path)

base = "/home/labadmin/aura_r8b/AgentBench/outputs/2026-01-29-23-18-22_os_qwen_ollama"
models = ["ollama-qwen-4b", "ollama-qwen-8b"]

summaries = {}
for model in models:
    run_path = f"{base}/{model}/os-std/runs.jsonl"
    err_path = f"{base}/{model}/os-std/error.jsonl"
    summaries[model] = summarize(run_path, err_path)

outdir = f"{base}/figures"
os.makedirs(outdir, exist_ok=True)

# ------------------------
# Figure 1: end-to-end rates
# ------------------------
metrics = ["task_success_rate", "completed_failure_rate", "agent_crash_rate"]
labels = ["Success", "Completed-Fail", "Agent-Crash"]

x = np.arange(len(models))
w = 0.25

plt.figure(figsize=(7,4))
for i, (mkey, lab) in enumerate(zip(metrics, labels)):
    vals = [summaries[model][mkey] for model in models]
    plt.bar(x + (i-1)*w, vals, width=w, label=lab)

plt.xticks(x, models, rotation=15)
plt.ylim(0, 1.0)
plt.ylabel("Rate")
plt.title("OS-STD End-to-End Outcome Rates")
plt.legend()
savefig(f"{outdir}/rates.pdf")
plt.close()

# ------------------------
# Figure 2: stacked status breakdown (including crashes)
# ------------------------
status_keys = ["completed", "task limit reached", "task error"]
status_labels = ["Completed", "Task Limit", "Task Error"]

plt.figure(figsize=(7,4))
bottom = np.zeros(len(models))
for k, lab in zip(status_keys, status_labels):
    vals = []
    for model in models:
        bd = summaries[model]["status_breakdown"]
        vals.append((bd.get(k, 0) / summaries[model]["total_tasks"]))
    vals = np.array(vals)
    plt.bar(x, vals, bottom=bottom, label=lab)
    bottom += vals

# add crashes
crash_vals = np.array([summaries[m]["runs_crashed"] / summaries[m]["total_tasks"] for m in models])
plt.bar(x, crash_vals, bottom=bottom, label="Agent Crash")

plt.xticks(x, models, rotation=15)
plt.ylim(0, 1.0)
plt.ylabel("Fraction of tasks")
plt.title("OS-STD Status Breakdown (Normalized by Total Tasks)")
plt.legend()
savefig(f"{outdir}/status_breakdown.pdf")
plt.close()

# ------------------------
# Optional: latency bars
# ------------------------
plt.figure(figsize=(7,4))
avg_lat = [summaries[m]["avg_latency_per_completed_task_sec"] or 0 for m in models]
avg_succ_lat = [summaries[m]["avg_latency_per_successful_task_sec"] or 0 for m in models]

plt.bar(x - 0.15, avg_lat, width=0.3, label="Avg latency (completed)")
plt.bar(x + 0.15, avg_succ_lat, width=0.3, label="Avg latency (successful)")

plt.xticks(x, models, rotation=15)
plt.ylabel("Seconds")
plt.title("OS-STD Latency")
plt.legend()
savefig(f"{outdir}/latency.pdf")
plt.close()

print("done")