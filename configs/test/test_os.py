import json, os
from collections import Counter

TOTAL_TASKS = 144  # AgentBench OS fixed task count

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def summarize(run_path, err_path):
    runs = load_jsonl(run_path)
    errs = load_jsonl(err_path)

    status_ctr = Counter()
    err_ctr = Counter((e.get("error") for e in errs))

    scores = []
    successes = 0
    completed_failures = 0

    turns = []
    bash_calls = []
    answer_calls = []
    finish_calls = []

    # -------- latency tracking --------
    latencies = []
    success_latencies = []

    # sort runs by completion time
    runs_sorted = sorted(
        runs,
        key=lambda r: (r.get("time") or {}).get("timestamp", 0)
    )

    prev_ts = None

    for r in runs_sorted:
        out = r.get("output") or {}
        status = out.get("status")
        status_ctr[status] += 1

        # ----- latency -----
        ts = (r.get("time") or {}).get("timestamp")
        if ts is not None and prev_ts is not None:
            latency_sec = (ts - prev_ts) / 1000.0
            latencies.append(latency_sec)
        prev_ts = ts

        res = out.get("result") or {}
        metrics = res.get("metrics") or {}
        sc = metrics.get("score")

        if sc is not None:
            scores.append(sc)
            if sc == 1:
                successes += 1
                if latencies:
                    success_latencies.append(latencies[-1])
            else:
                completed_failures += 1

        msgs = res.get("messages") or []

        t = 0
        bc = ac = fc = 0
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            tcalls = m.get("tool_calls") or []
            if tcalls:
                t += 1
            for tc in tcalls:
                fn = (((tc.get("function") or {}).get("name")) or "")
                if fn == "bash_action":
                    bc += 1
                elif fn == "answer_action":
                    ac += 1
                elif fn == "finish_action":
                    fc += 1

        turns.append(t)
        bash_calls.append(bc)
        answer_calls.append(ac)
        finish_calls.append(fc)

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    summary = {
        "total_tasks": TOTAL_TASKS,
        "runs_completed": len(runs),
        "runs_crashed": len(errs),

        "task_success_rate": successes / TOTAL_TASKS,
        "completed_failure_rate": completed_failures / TOTAL_TASKS,
        "agent_crash_rate": len(errs) / TOTAL_TASKS,

        "mean_score_over_completed_runs": mean(scores),

        # -------- latency metrics --------
        "avg_latency_per_completed_task_sec": mean(latencies),
        "avg_latency_per_successful_task_sec": mean(success_latencies),

        "status_breakdown": dict(status_ctr),
        "error_breakdown": dict(err_ctr),

        "avg_turns_with_toolcalls": mean(turns),
        "avg_bash_calls": mean(bash_calls),
        "avg_answer_calls": mean(answer_calls),
        "avg_finish_calls": mean(finish_calls),
    }

    return summary

# Example usage
base = "/home/labadmin/aura_r8b/AgentBench/outputs/2026-01-29-23-18-22_os_qwen_ollama"

for model in ["ollama-qwen-4b", "ollama-qwen-8b"]:
    run_path = f"{base}/{model}/os-std/runs.jsonl"
    err_path = f"{base}/{model}/os-std/error.jsonl"
    print("\n===", model, "===")
    print(json.dumps(summarize(run_path, err_path), indent=2))
