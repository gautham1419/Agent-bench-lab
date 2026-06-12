import json
import os
import matplotlib.pyplot as plt

# -------------------------------------------------
# Resolve paths RELATIVE TO THIS FILE
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results", "results_dbbench"))
PLOTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "plots", "dbbench"))

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
# Generic bar plot
# -------------------------------------------------

def plot_bar(metric, title, ylabel, filename):
    values = [safe_value(results[m].get(metric)) for m in models]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, values)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}",
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
# MAIN DBBENCH FIGURES (PAPER-READY)
# -------------------------------------------------

# 1. Primary metric
plot_bar(
    "task_success_rate",
    "Task Success Rate on DBBench",
    "Success Rate",
    "task_success_rate.png"
)

# 2. Secondary: SQL execution validity
plot_bar(
    "sql_execution_validity_rate",
    "SQL Execution Validity Rate",
    "Valid / Total SQL Calls",
    "sql_validity_rate.png"
)

# 3. Secondary: recovery after SQL errors
plot_bar(
    "recovery_rate",
    "Recovery Rate After SQL Errors",
    "Recovered / Error Tasks",
    "recovery_rate.png"
)

# -------------------------------------------------
# Reliability–Performance Tradeoff
# -------------------------------------------------

plt.figure(figsize=(6, 4))

for m in models:
    x = safe_value(results[m].get("sql_execution_validity_rate"))
    y = safe_value(results[m].get("task_success_rate"))

    plt.scatter(x, y, s=80)
    plt.text(x, y, m, fontsize=8, ha="center", va="bottom")

plt.xlabel("SQL Execution Validity Rate")
plt.ylabel("Task Success Rate")
plt.title("Reliability–Performance Tradeoff (DBBench)")
plt.tight_layout()

path = os.path.join(PLOTS_DIR, "success_vs_sql_validity.png")
plt.savefig(path, dpi=300)
plt.close()
print(f"Saved → {path}")

print("\nAll DBBench figures generated successfully.")
