import json


def extract_score(out):
    res = (out or {}).get("result") or {}
    metrics = res.get("metrics") or {}

    if metrics.get("score") is not None:
        return metrics["score"]

    return res.get("reward")


def count_turns(messages):
    return sum(1 for m in messages if m.get("role") == "assistant")


def count_tool_calls(messages):
    total = 0
    for m in messages:
        if "tool_calls" in m:
            total += len(m["tool_calls"])
    return total


def count_lines(file_path):
    if file_path is None or not file_path.exists():
        return 0
    with open(file_path) as f:
        return sum(1 for _ in f)


def compute_performance(runs_file, error_file=None):

    runs_completed = 0
    errors = count_lines(error_file)

    successes = 0
    failures = 0
    rewards = []

    tool_calls_total = 0
    turns_total = 0

    with open(runs_file) as f:
        for line in f:
            r = json.loads(line)

            runs_completed += 1

            out = r.get("output") or {}
            res = out.get("result") or {}
            messages = res.get("messages") or []

            score = extract_score(out)

            if score is not None:
                rewards.append(score)

                if score > 0:
                    successes += 1
                else:
                    failures += 1

            tool_calls_total += count_tool_calls(messages)
            turns_total += count_turns(messages)

    total_tasks = runs_completed + errors

    # ---- rates ----

    if total_tasks > 0:
        success_rate = successes / total_tasks
        failure_rate = failures / total_tasks
        completion_rate = runs_completed / total_tasks
        crash_rate = errors / total_tasks
    else:
        success_rate = failure_rate = completion_rate = crash_rate = 0

    mean_reward = sum(rewards) / len(rewards) if rewards else 0

    avg_tool_calls = tool_calls_total / runs_completed if runs_completed else 0
    avg_turns = turns_total / runs_completed if runs_completed else 0

    return {
        "total_tasks": total_tasks,
        "runs_completed": runs_completed,
        "errors": errors,

        "reward": sum(rewards),
        "score": mean_reward,

        "tool_calls_total": tool_calls_total,

        "successes": successes,
        "failures": failures,

        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "completion_rate": completion_rate,
        "crash_rate": crash_rate,

        "mean_reward": mean_reward,
        "avg_tool_calls": avg_tool_calls,
        "avg_turns": avg_turns
    }