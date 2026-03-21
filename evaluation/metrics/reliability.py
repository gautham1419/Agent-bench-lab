import json


def has_tool_format_error(messages):
    for m in messages:
        if "tool_calls" in m:
            for call in m["tool_calls"]:
                args = call.get("function", {}).get("arguments")

                if args in [None, "", "{}"]:
                    return True

                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                    if not isinstance(parsed, dict) or len(parsed) == 0:
                        return True
                except:
                    return True
    return False


def compute_reliability(runs_file, error_file=None):

    runs_total = 0

    agent_failures = 0
    interaction_failures = 0
    timeout_failures = 0
    tool_format_failures = 0

    error_failures = 0
    crashes = 0

    # -------------------------------
    # PARSE runs.jsonl
    # -------------------------------
    with open(runs_file) as f:
        for line in f:
            runs_total += 1

            r = json.loads(line)

            out = r.get("output") or {}
            res = out.get("result") or {}
            messages = res.get("messages") or []
            status = (res.get("status") or "").lower()

            # ---- agent failure ----
            if "task error" in status:
                agent_failures += 1

            # ---- interaction failure ----
            if "interact_failed" in status:
                interaction_failures += 1

            # ---- timeout ----
            if "limit" in status or "timeout" in status:
                timeout_failures += 1

            # ---- tool format ----
            if has_tool_format_error(messages):
                tool_format_failures += 1

    # -------------------------------
    # PARSE error.jsonl
    # -------------------------------
    if error_file is not None and error_file.exists():
        with open(error_file) as f:
            for line in f:
                r = json.loads(line)
                err = (r.get("error") or "").lower()

                if "agent_failed" in err:
                    agent_failures += 1
                    error_failures += 1

                elif "interact_failed" in err:
                    interaction_failures += 1
                    error_failures += 1

                elif "timeout" in err:
                    timeout_failures += 1
                    error_failures += 1

                else:
                    crashes += 1  # true infra crash

    # -------------------------------
    # FINAL DENOMINATOR
    # -------------------------------
    total_attempts = runs_total + error_failures + crashes
    denom = total_attempts if total_attempts > 0 else 1

    return {
        "total_attempts": total_attempts,

        "agent_failures": agent_failures,
        "interaction_failures": interaction_failures,
        "timeout_failures": timeout_failures,
        "tool_format_failures": tool_format_failures,

        # ---- rates (aligned with performance) ----
        "agent_failure_rate": agent_failures / denom,
        "interaction_failure_rate": interaction_failures / denom,
        "timeout_rate": timeout_failures / denom,
        "tool_format_rate": tool_format_failures / denom,
    }