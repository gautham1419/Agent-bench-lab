import json


def has_tool_format_error(messages):
    for m in messages:
        if "tool_calls" in m:
            for call in m["tool_calls"]:
                args = call.get("function", {}).get("arguments")
                if args == "{}" or args == "" or args is None:
                    return True
    return False


def count_lines(file_path):
    if file_path is None or not file_path.exists():
        return 0
    with open(file_path) as f:
        return sum(1 for _ in f)


def compute_reliability(error_file, runs_file):

    runs_total = 0
    agent_crash_failures = 0
    tool_format_failures = 0

    # -------------------------------
    # PARSE RUNS
    # -------------------------------
    with open(runs_file) as f:
        for line in f:
            runs_total += 1

            r = json.loads(line)

            # skip infra-level errors already logged separately
            if r.get("error") is not None:
                continue

            out = r.get("output") or {}
            res = out.get("result") or {}
            messages = res.get("messages") or []

            status = res.get("status")

            if status == "task error":
                agent_crash_failures += 1

            if has_tool_format_error(messages):
                tool_format_failures += 1

    # -------------------------------
    # INFRASTRUCTURE CRASHES
    # -------------------------------
    runs_crashed = count_lines(error_file)

    # -------------------------------
    # RATES
    # -------------------------------
    if runs_total > 0:
        agent_crash_rate = agent_crash_failures / runs_total
        tool_format_violation_rate = tool_format_failures / runs_total
    else:
        agent_crash_rate = 0
        tool_format_violation_rate = 0

    return {
        "runs_crashed": runs_crashed,
        "agent_crash_failures": agent_crash_failures,
        "tool_format_failures": tool_format_failures,

        "agent_crash_rate": agent_crash_rate,
        "tool_format_violation_rate": tool_format_violation_rate
    }