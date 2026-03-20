import json, os
from collections import Counter, defaultdict

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

    attempts = len(runs) + len(errs)

    status_ctr = Counter()
    scores = []
    turns = []
    bash_calls = []
    answer_calls = []
    finish_calls = []

    for r in runs:
        out = (r.get("output") or {})
        status = out.get("status")
        status_ctr[status] += 1

        res = (out.get("result") or {})
        sc = ((res.get("metrics") or {}).get("score"))
        if sc is not None:
            scores.append(sc)

        msgs = res.get("messages") or []
        # Count assistant tool calls
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

    err_ctr = Counter((e.get("error") for e in errs))

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    summary = {
        "attempts_total": attempts,
        "runs": len(runs),
        "errors": len(errs),
        "success_rate(score_mean_over_runs_with_score)": mean(scores),
        "status_breakdown": dict(status_ctr),
        "error_breakdown": dict(err_ctr),
        "avg_turns_with_toolcalls": mean(turns),
        "avg_bash_calls": mean(bash_calls),
        "avg_answer_calls": mean(answer_calls),
        "avg_finish_calls": mean(finish_calls),
        "agent_fail_rate": (len(errs) / attempts) if attempts else None,
    }
    return summary

# Example usage:
base = "/home/labadmin/aura_r8b/AgentBench/outputs/2026-01-29-23-18-22_os_qwen_ollama"
for model in ["ollama-qwen-4b", "ollama-qwen-8b"]:
    run_path = f"{base}/{model}/os-std/runs.jsonl"
    err_path = f"{base}/{model}/os-std/error.jsonl"
    print("\n===", model, "===")
    print(json.dumps(summarize(run_path, err_path), indent=2))