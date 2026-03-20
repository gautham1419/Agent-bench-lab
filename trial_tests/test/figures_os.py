import json
import os
import matplotlib.pyplot as plt

# -------------------------------------------------
# Resolve paths RELATIVE TO THIS FILE
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # configs/test
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
PLOTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "plots"))

os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------------------------------
# Load results
# -------------------------------------------------

def load_results():
    results = {}
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".json"):
            model = fname.replace(".json", "")
            with open(os.path.join(RESULTS_DIR, fname), "r") as f:
                results[model] = json.load(f)
    return results


results = load_results()
models = sorted(results.keys())

# -------------------------------------------------
# Safe metric getter
# -------------------------------------------------

def safe_value(val):
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

# -------------------------------------------------
# Generic bar plot (crash-proof)
# -------------------------------------------------

def plot_bar(metric, title, ylabel, filename):
    values = [safe_value(results[m].get(metric)) for m in models]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, values)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)

    # label even tiny values
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved → {path}")

# -------------------------------------------------
# MAIN PAPER FIGURES
# -------------------------------------------------

plot_bar(
    "task_success_rate",
    "Task Success Rate (All Models)",
    "Success Rate",
    "task_success_rate.png"
)

plot_bar(
    "avg_latency_sec_per_completed_task",
    "Average Latency per Task",
    "Seconds",
    "latency.png"
)

plot_bar(
    "avg_total_tokens_per_completed_task",
    "Average Tokens per Task",
    "Tokens",
    "tokens.png"
)

plot_bar(
    "command_validity_rate",
    "Command Validity Rate",
    "Valid / Total Commands",
    "command_validity.png"
)

plot_bar(
    "recovery_rate",
    "Recovery Rate After Errors",
    "Recovered / Error Tasks",
    "recovery_rate.png"
)

# -------------------------------------------------
# Efficiency–Performance Tradeoff
# -------------------------------------------------

plt.figure(figsize=(6, 4))

for m in models:
    x = safe_value(results[m].get("avg_total_tokens_per_completed_task"))
    y = safe_value(results[m].get("task_success_rate"))

    plt.scatter(x, y, s=80)
    plt.text(x, y, m, fontsize=8, ha="center", va="bottom")

plt.xlabel("Average Tokens per Task")
plt.ylabel("Task Success Rate")
plt.title("Efficiency–Performance Tradeoff")
plt.tight_layout()

path = os.path.join(PLOTS_DIR, "success_vs_tokens.png")
plt.savefig(path, dpi=300)
plt.close()
print(f"Saved → {path}")

print("\nAll figures generated successfully.")
