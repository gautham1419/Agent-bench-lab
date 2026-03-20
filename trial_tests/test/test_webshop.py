import json
import os
from collections import Counter

TOTAL_TASKS = 500  # WebShop task count (change if needed)

# --------- paths ---------

BASE_OUTPUT = "/home/labadmin/aura_r8b/AgentBench/outputs/2026-02-05-13-18-40_webshop_ollama"
RESULTS_DIR = os.path.abspath("../results/results_webshop")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------- helpers ---------

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def mean(xs):
    return (sum(xs) / len(xs)) if xs else None

def safe_int(x, default=0):
    try:
        return default if x is None else int(x)
    except Exception:
        return default

def extract_score(out):
    res = (out or {}).get("result") or {}
    metrics = res.get("metrics") or {}
    if metrics.get("score") is not None:
        return metrics["score"]
    return res.get("reward")

# --------- main summary ---------

def summarize(run_path, err_path):
    runs = load_jsonl(run_path)
    errs = load_jsonl(err_path)

    status_ctr = Counter()
    successes = 0
    scored_tasks = 0

    # --- WebShop metrics ---
    steps_per_task = []
    total_tokens_per_task = []

    for r in runs:
        out = r.get("output") or {}
        status = out.get("status") or "unknown"
        status_ctr[status] += 1

        score = extract_score(out)
        if score is not None:
            scored_tasks += 1
            if score == 1:
                successes += 1

        res = out.get("result") or {}
        msgs = res.get("messages") or []

        steps = 0
        total_tokens = 0

        for m in msgs:
            if m.get("role") == "assistant":
                tcalls = m.get("tool_calls") or []
                steps += len(tcalls)

                usage = m.get("usage")
                if usage:
                    total_tokens += safe_int(usage.get("total_tokens"))

        steps_per_task.append(steps)
        total_tokens_per_task.append(total_tokens)

    return {
        "total_tasks": TOTAL_TASKS,
        "runs_completed": len(runs),
        "runs_crashed": len(errs),

        "successes": successes,
        "task_success_rate": successes / TOTAL_TASKS,
        "agent_crash_rate": len(errs) / TOTAL_TASKS,

        # ---- Secondary metrics ----
        "avg_steps_per_task": mean(steps_per_task),
        "avg_total_tokens_per_task": mean(total_tokens_per_task),

        "status_breakdown": dict(status_ctr),
    }

# --------- auto-discover models + save ---------

for model in sorted(os.listdir(BASE_OUTPUT)):
    run_path = os.path.join(BASE_OUTPUT, model, "webshop-std", "runs.jsonl")
    err_path = os.path.join(BASE_OUTPUT, model, "webshop-std", "error.jsonl")

    if not os.path.exists(run_path):
        continue

    metrics = summarize(run_path, err_path)

    print(f"\n=== {model} ===")
    print(json.dumps(metrics, indent=2))

    out_file = os.path.join(RESULTS_DIR, f"{model}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
