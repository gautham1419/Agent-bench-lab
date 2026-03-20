import json
import os
from collections import Counter

TOTAL_TASKS = 144  # AgentBench OS fixed task count

# --------- paths ---------

BASE_OUTPUT = "/home/labadmin/aura_r8b/AgentBench/outputs/2026-02-05-07-48-57_os_ollama"
RESULTS_DIR = os.path.abspath("../results")   # sibling to tests / plots
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

def safe_float(x, default=0.0):
    try:
        return default if x is None else float(x)
    except Exception:
        return default

def extract_score(out):
    res = (out or {}).get("result") or {}
    metrics = res.get("metrics") or {}
    if metrics.get("score") is not None:
        return metrics["score"]
    return res.get("reward")

# --------- OS error detection ---------

SHELL_ERROR_PATTERNS = [
    "command not found",
    "no such file",
    "permission denied",
    "syntax error",
    "cannot access",
    "not recognized",
    "missing operand",
]

def is_command_error(output_text):
    if not output_text:
        return False
    lower = output_text.lower()
    return any(pat in lower for pat in SHELL_ERROR_PATTERNS)

# --------- main summary ---------

def summarize(run_path, err_path):
    runs = load_jsonl(run_path)
    errs = load_jsonl(err_path)

    status_ctr = Counter()
    err_ctr = Counter((e.get("error") for e in errs if e))

    successes = 0
    scored_tasks = 0

    turns_with_toolcalls = []
    total_tool_calls = []
    bash_calls = []
    answer_calls = []
    finish_calls = []
    tool_format_violations = 0

    prompt_tokens_per_task = []
    completion_tokens_per_task = []
    total_tokens_per_task = []
    latency_sec_per_task = []

    total_bash_commands = 0
    valid_bash_commands = 0
    tasks_with_command_error = 0
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

        for m in msgs:
            if m.get("role") == "user":
                if "No executable tool calls found" in (m.get("content") or ""):
                    tool_format_violations += 1
                    break

        t_turns = 0
        tc_total = 0
        bc = ac = fc = 0
        pt = ct = tt = 0
        task_latency = 0.0
        saw_latency = False

        task_had_command_error = False
        last_tool_name = None

        for m in msgs:
            role = m.get("role")

            if role == "assistant":
                tcalls = m.get("tool_calls") or []
                if tcalls:
                    t_turns += 1

                for tc in tcalls:
                    tc_total += 1
                    fn = ((tc.get("function") or {}).get("name")) or ""
                    last_tool_name = fn

                    if fn == "bash_action":
                        bc += 1
                        total_bash_commands += 1
                    elif fn == "answer_action":
                        ac += 1
                    elif fn == "finish_action":
                        fc += 1

                usage = m.get("usage")
                if usage:
                    pt += safe_int(usage.get("prompt_tokens"))
                    ct += safe_int(usage.get("completion_tokens"))
                    tt += safe_int(usage.get("total_tokens"))

                    td = usage.get("total_duration")
                    if td is not None:
                        task_latency += safe_float(td) / 1e9
                        saw_latency = True

            if role == "tool" and last_tool_name == "bash_action":
                content = m.get("content") or ""
                if is_command_error(content):
                    task_had_command_error = True
                else:
                    valid_bash_commands += 1

        if task_had_command_error:
            tasks_with_command_error += 1
            if score == 1:
                tasks_recovered += 1

        turns_with_toolcalls.append(t_turns)
        total_tool_calls.append(tc_total)
        bash_calls.append(bc)
        answer_calls.append(ac)
        finish_calls.append(fc)

        prompt_tokens_per_task.append(pt)
        completion_tokens_per_task.append(ct)
        total_tokens_per_task.append(tt)

        if saw_latency:
            latency_sec_per_task.append(task_latency)

    return {
        "total_tasks": TOTAL_TASKS,
        "runs_completed": len(runs),
        "runs_crashed": len(errs),

        "successes": successes,
        "task_success_rate": successes / TOTAL_TASKS,
        "completed_failure_rate": (scored_tasks - successes) / TOTAL_TASKS,
        "agent_crash_rate": len(errs) / TOTAL_TASKS,

        "status_breakdown": dict(status_ctr),
        "error_breakdown": dict(err_ctr),

        "tool_format_violation_rate": tool_format_violations / TOTAL_TASKS,
        "avg_turns_with_toolcalls_per_completed_task": mean(turns_with_toolcalls),
        "avg_total_tool_calls_per_completed_task": mean(total_tool_calls),
        "avg_bash_calls_per_completed_task": mean(bash_calls),
        "avg_answer_calls_per_completed_task": mean(answer_calls),
        "avg_finish_calls_per_completed_task": mean(finish_calls),

        "command_validity_rate": (
            valid_bash_commands / total_bash_commands
            if total_bash_commands > 0 else None
        ),
        "tasks_with_command_errors": tasks_with_command_error,
        "recovery_rate": (
            tasks_recovered / tasks_with_command_error
            if tasks_with_command_error > 0 else None
        ),

        "avg_prompt_tokens_per_completed_task": mean(prompt_tokens_per_task),
        "avg_completion_tokens_per_completed_task": mean(completion_tokens_per_task),
        "avg_total_tokens_per_completed_task": mean(total_tokens_per_task),
        "avg_latency_sec_per_completed_task": mean(latency_sec_per_task),
    }

# --------- auto-discover models + save ---------

for model in sorted(os.listdir(BASE_OUTPUT)):
    run_path = os.path.join(BASE_OUTPUT, model, "os-std", "runs.jsonl")
    err_path = os.path.join(BASE_OUTPUT, model, "os-std", "error.jsonl")

    if not os.path.exists(run_path):
        continue

    metrics = summarize(run_path, err_path)

    print(f"\n=== {model} ===")
    print(json.dumps(metrics, indent=2))

    out_file = os.path.join(RESULTS_DIR, f"{model}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
