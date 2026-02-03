import json, os
from collections import Counter

TOTAL_TASKS = 144  # AgentBench OS fixed task count

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
        if x is None:
            return default
        return int(x)
    except Exception:
        return default

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def extract_score(out):
    """
    Prefer metrics.score (what your logs show), fallback to reward.
    Return None if truly missing.
    """
    res = (out or {}).get("result") or {}
    metrics = res.get("metrics") or {}
    sc = metrics.get("score")
    if sc is not None:
        return sc
    rw = res.get("reward")
    return rw

def summarize(run_path, err_path):
    runs = load_jsonl(run_path)
    errs = load_jsonl(err_path)

    status_ctr = Counter()
    err_ctr = Counter((e.get("error") for e in errs if e))

    # ----- accuracy -----
    successes = 0
    scored_tasks = 0  # how many runs had score/reward available

    # ----- tool calling -----
    turns_with_toolcalls = []
    bash_calls = []
    answer_calls = []
    finish_calls = []
    total_tool_calls = []
    tool_format_violations = 0  # "No executable tool calls found..."

    # ----- tokens -----
    prompt_tokens_per_task = []
    completion_tokens_per_task = []
    total_tokens_per_task = []
    tokens_seen_tasks = 0  # tasks where we saw any usage

    # optional: latency from ollama durations (more meaningful than timestamp diff)
    # ollama durations look like nanoseconds in your log; we convert to seconds
    latency_sec_per_task = []

    for r in runs:
        out = r.get("output") or {}
        status = out.get("status") or "unknown"
        status_ctr[status] += 1

        sc = extract_score(out)
        if sc is not None:
            scored_tasks += 1
            if sc == 1:
                successes += 1

        res = out.get("result") or {}
        msgs = res.get("messages") or []

        # tool-format violation detection (this is useful to report)
        violated = False
        for m in msgs:
            if m.get("role") == "user":
                c = m.get("content") or ""
                if "No executable tool calls found" in c:
                    violated = True
                    break
        if violated:
            tool_format_violations += 1

        # per-task counters
        t_turns = 0
        bc = ac = fc = 0
        tc_total = 0

        pt = 0
        ct = 0
        tt = 0
        saw_usage = False

        task_latency_sec = 0.0
        saw_duration = False

        for m in msgs:
            if m.get("role") != "assistant":
                continue

            # tool calls
            tcalls = m.get("tool_calls") or []
            if tcalls:
                t_turns += 1
            for tc in tcalls:
                tc_total += 1
                fn = (((tc.get("function") or {}).get("name")) or "")
                if fn == "bash_action":
                    bc += 1
                elif fn == "answer_action":
                    ac += 1
                elif fn == "finish_action":
                    fc += 1

            # token usage (new log field)
            usage = m.get("usage") or None
            if usage:
                # These are per-LLM-call counters from Ollama
                p = usage.get("prompt_tokens")
                c = usage.get("completion_tokens")
                t = usage.get("total_tokens")

                # Only add if present; if missing, treat as 0 for that call
                pt += safe_int(p, 0)
                ct += safe_int(c, 0)
                tt += safe_int(t, 0)

                # optional: latency from ollama total_duration (nanoseconds)
                td = usage.get("total_duration")
                if td is not None:
                    # convert ns -> seconds
                    task_latency_sec += safe_float(td, 0.0) / 1e9
                    saw_duration = True

                saw_usage = True

        turns_with_toolcalls.append(t_turns)
        bash_calls.append(bc)
        answer_calls.append(ac)
        finish_calls.append(fc)
        total_tool_calls.append(tc_total)

        if saw_usage:
            tokens_seen_tasks += 1
        prompt_tokens_per_task.append(pt)
        completion_tokens_per_task.append(ct)
        total_tokens_per_task.append(tt)

        if saw_duration:
            latency_sec_per_task.append(task_latency_sec)

    # Definitions:
    # - success rate: successes / TOTAL_TASKS
    # - crash rate: len(errs) / TOTAL_TASKS
    # - completed failure rate: (non-crash completed runs - successes) / TOTAL_TASKS
    #   Here we treat "runs" as non-crash recorded runs. (crashes are in error.jsonl)
    completed_failure_rate = (len(runs) - successes) / TOTAL_TASKS

    summary = {
        "total_tasks": TOTAL_TASKS,

        # basic accounting
        "runs_logged_in_runs_jsonl": len(runs),
        "runs_crashed_in_error_jsonl": len(errs),

        # accuracy
        "successes": successes,
        "task_success_rate": successes / TOTAL_TASKS,
        "completed_failure_rate": completed_failure_rate,
        "agent_crash_rate": len(errs) / TOTAL_TASKS,
        "scored_tasks_in_runs_jsonl": scored_tasks,  # sanity check

        # status / error breakdown
        "status_breakdown": dict(status_ctr),
        "error_breakdown": dict(err_ctr),

        # tool calling metrics
        "tool_format_violation_rate": tool_format_violations / TOTAL_TASKS,
        "avg_turns_with_toolcalls": mean(turns_with_toolcalls),
        "avg_total_tool_calls": mean(total_tool_calls),
        "avg_bash_calls": mean(bash_calls),
        "avg_answer_calls": mean(answer_calls),
        "avg_finish_calls": mean(finish_calls),

        # token metrics
        "tasks_with_any_token_usage_logged": tokens_seen_tasks,
        "avg_prompt_tokens_per_task": mean(prompt_tokens_per_task),
        "avg_completion_tokens_per_task": mean(completion_tokens_per_task),
        "avg_total_tokens_per_task": mean(total_tokens_per_task),

        # optional (only if usage.total_duration was present)
        "avg_latency_sec_per_task_from_ollama_durations": mean(latency_sec_per_task),
    }

    return summary

# Example usage (your base path)
base = "/home/labadmin/aura_r8b/AgentBench/outputs/2026-01-31-17-57-40_os_qwen_ollama"

for model in ["ollama-qwen-4b", "ollama-qwen-8b"]:
    run_path = f"{base}/{model}/os-std/runs.jsonl"
    err_path = f"{base}/{model}/os-std/error.jsonl"
    print("\n===", model, "===")
    print(json.dumps(summarize(run_path, err_path), indent=2))
    