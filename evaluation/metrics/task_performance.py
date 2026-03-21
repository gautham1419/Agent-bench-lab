import json


def extract_score(out):
    res = (out or {}).get("result") or {}
    metrics = res.get("metrics") or {}

    if metrics.get("score") is not None:
        return metrics["score"]

    return res.get("reward")


def count_lines(file_path):
    if file_path is None or not file_path.exists():
        return 0
    with open(file_path) as f:
        return sum(1 for _ in f)


def compute_performance(runs_file, error_file=None):

    runs_completed = 0

    successes = 0
    failures = 0
    timeouts = 0
    rewards = []

    # -------------------------------
    # PARSE runs.jsonl
    # -------------------------------
    with open(runs_file) as f:
        for line in f:
            r = json.loads(line)

            runs_completed += 1

            out = r.get("output") or {}
            res = out.get("result") or {}

            status = (res.get("status") or "").lower()
            score = extract_score(out)

            # ---- success ----
            if score is not None and score > 0:
                successes += 1
                rewards.append(score)

            else:
                failures += 1

            # ---- timeout ----
            if "limit" in status:
                timeouts += 1

    # -------------------------------
    # PARSE error.jsonl
    # -------------------------------
    crashes = 0
    extra_failures = 0

    if error_file is not None and error_file.exists():
        with open(error_file) as f:
            for line in f:
                r = json.loads(line)
                err = (r.get("error") or "").lower()

                if "agent_failed" in err or "interact_failed" in err:
                    extra_failures += 1
                else:
                    crashes += 1

    # -------------------------------
    # FINAL COUNTS
    # -------------------------------
    failures += extra_failures

    total_tasks = runs_completed + extra_failures + crashes

    # -------------------------------
    # RATES
    # -------------------------------
    if total_tasks > 0:
        success_rate = successes / total_tasks
        failure_rate = failures / total_tasks
        completion_rate = runs_completed / total_tasks
        crash_rate = crashes / total_tasks
    else:
        success_rate = failure_rate = completion_rate = crash_rate = 0

    agent_failure_rate = failures / runs_completed if runs_completed else 0

    mean_reward = sum(rewards) / len(rewards) if rewards else 0

    return {
        "total_tasks": total_tasks,
        "runs_completed": runs_completed,
        "crashes": crashes,

        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,

        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "agent_failure_rate": agent_failure_rate,
        "completion_rate": completion_rate,
        "crash_rate": crash_rate,

        "mean_reward": mean_reward,
        "reward": sum(rewards)
    }