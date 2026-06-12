import json
import os
from collections import Counter

TOTAL_TASKS = 300  # set DBBench task count explicitly

# --------- paths ---------

BASE_OUTPUT = "/home/labadmin/aura_r8b/AgentBench/outputs/2026-02-05-01-13-31"
RESULTS_DIR = os.path.abspath("../results/results_dbbench")
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

def extract_score(out):
    res = (out or {}).get("result") or {}
    metrics = res.get("metrics") or {}
    if metrics.get("score") is not None:
        return metrics["score"]
    return res.get("reward")

# --------- SQL error detection ---------

SQL_ERROR_PATTERNS = [
    "you have an error in your sql syntax",
    "unknown column",
    "unknown table",
    "doesn't exist",
    "cannot be null",
    "duplicate entry",
    "foreign key constraint fails",
]

def is_sql_error(output_text):
    if not output_text:
        return False
    lower = output_text.lower()
    return any(pat in lower for pat in SQL_ERROR_PATTERNS)

# --------- main summary ---------

def summarize(run_path, err_path):
    runs = load_jsonl(run_path)
    errs = load_jsonl(err_path)

    status_ctr = Counter()
    err_ctr = Counter((e.get("error") for e in errs if e))

    successes = 0
    scored_tasks = 0

    total_sql_calls = 0
    valid_sql_calls = 0

    tasks_with_sql_error = 0
    tasks_recovered = 0

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

        task_had_sql_error = False
        last_tool_name = None

        for m in msgs:
            if m.get("role") == "assistant":
                for tc in (m.get("tool_calls") or []):
                    fn = ((tc.get("function") or {}).get("name")) or ""
                    last_tool_name = fn
                    if fn == "execute_sql":
                        total_sql_calls += 1

            if m.get("role") == "tool" and last_tool_name == "execute_sql":
                content = m.get("content") or ""
                if is_sql_error(content):
                    task_had_sql_error = True
                else:
                    valid_sql_calls += 1

        if task_had_sql_error:
            tasks_with_sql_error += 1
            if score == 1:
                tasks_recovered += 1

    return {
        "total_tasks": TOTAL_TASKS,
        "runs_completed": len(runs),
        "runs_crashed": len(errs),

        # ---- primary (AgentBench-native) ----
        "successes": successes,
        "task_success_rate": successes / TOTAL_TASKS,
        "completed_failure_rate": (scored_tasks - successes) / TOTAL_TASKS,
        "agent_crash_rate": len(errs) / TOTAL_TASKS,

        "status_breakdown": dict(status_ctr),
        "error_breakdown": dict(err_ctr),

        # ---- secondary (direct only) ----
        "sql_execution_validity_rate": (
            valid_sql_calls / total_sql_calls
            if total_sql_calls > 0 else None
        ),
        "tasks_with_sql_errors": tasks_with_sql_error,
        "recovery_rate": (
            tasks_recovered / tasks_with_sql_error
            if tasks_with_sql_error > 0 else None
        ),
    }

# --------- auto-discover models + save ---------

for model in sorted(os.listdir(BASE_OUTPUT)):
    run_path = os.path.join(BASE_OUTPUT, model, "dbbench-std", "runs.jsonl")
    err_path = os.path.join(BASE_OUTPUT, model, "dbbench-std", "error.jsonl")

    if not os.path.exists(run_path):
        continue

    metrics = summarize(run_path, err_path)

    print(f"\n=== {model} ===")
    print(json.dumps(metrics, indent=2))

    out_file = os.path.join(RESULTS_DIR, f"{model}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
