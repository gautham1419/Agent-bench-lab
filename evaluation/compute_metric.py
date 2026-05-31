import json
import yaml
from pathlib import Path

# extract score & reward 
def extract_score(out):
    res = (out or {}).get("result") or {}
    metrics = res.get("metrics") or {}

    if metrics.get("score") is not None:
        return float(metrics["score"])

    reward = res.get("reward")
    if reward is not None:
        return float(reward)

    return None

# calculate invalid format error
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

# parse model & domain metadata from yaml file
def parse_metadata(config_file):

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    agent = cfg["assignments"][0]["agent"]
    task = cfg["assignments"][0]["task"]

    domain = task.split("-")[0]
    parts = agent.split("-")

    if len(parts) >= 5 and parts[0] == "deepseek" and parts[1] == "r1" and parts[2] == "qwen":
        model = "deepseek-r1-qwen"
        size = parts[3].upper()
        quant = parts[-1]
    else:
        model = parts[1]
        size = parts[2].upper()
        quant = parts[-1]

    if quant == "f16":
        quant = "bf16"

    return model, size, quant, domain


def compute_metrics(runs_file, error_file, resource_file):

    metrics = {}

    runs_completed = 0
    successes = 0
    failures = 0
    rewards = []

    # Failure taxonomy counters
    tle_failures = 0       # Task Limit Exceeded
    if_failures = 0        # Invalid Format
    ia_failures = 0        # Invalid Action
    task_errors = 0        # Runtime / Task Error
    completed_failures = 0 # Implicit Failures (score <= 0, status completed)

    # total counts from runs
    total_tool_calls = 0
    turns_to_success_total = 0
    turns_to_failure_total = 0

    # Parse runs.jsonl 
    if runs_file.exists():
        with open(runs_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                runs_completed += 1

                data = json.loads(line)
                out = data.get("output", {})
                res = out.get("result") or {}

                score = extract_score(out)
                messages = res.get("messages") or []
                status = res.get("status", "")

                # tool calls 
                run_tool_calls = sum(len(m.get("tool_calls", [])) for m in messages if "tool_calls" in m)
                total_tool_calls += run_tool_calls

                # conversational turns
                conversational_turns = len(messages)

                is_success = (score is not None and score > 0)

                # successes vs failures
                if is_success:
                    successes += 1
                    rewards.append(score)
                    turns_to_success_total += conversational_turns
                else:
                    failures += 1
                    turns_to_failure_total += conversational_turns

                    # parse failure taxonomy 
                    if has_tool_format_error(messages):
                        if_failures += 1
                    elif status == "task limit reached":
                        tle_failures += 1
                    elif status == "agent invalid action":
                        ia_failures += 1
                    elif status == "task error":
                        task_errors += 1
                    elif status == "completed":
                        completed_failures += 1

    # error.jsonl counters
    agent_failed = 0 #AGENT_FAILED
    interact_failed = 0 #INTERACT_FAILED
    start_failed = 0 #START_FAILED
    errors = 0 

    # Parse error.jsonl 
    if error_file.exists():
        with open(error_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                err = (r.get("error") or "").upper()
                errors += 1

                if "AGENT_FAILED" in err:
                    agent_failed += 1

                elif "INTERACT_FAILED" in err:
                    interact_failed += 1

                elif "START_FAILED" in err:
                    start_failed += 1

    # Total Count & Denominator
    total_tasks = runs_completed + errors
    denom = total_tasks if total_tasks > 0 else 1

    # Rates
    success_rate = successes / denom
    failure_rate = failures / denom
    completion_rate = runs_completed / denom
    error_rate = errors / denom
    mean_reward = sum(rewards) / len(rewards) if rewards else 0

    # Averages
    avg_tool_calls = total_tool_calls / runs_completed if runs_completed else 0
    avg_turns_to_success = turns_to_success_total / successes if successes else 0
    avg_turns_to_failure = turns_to_failure_total / failures if failures else 0
    avg_turns = (turns_to_success_total + turns_to_failure_total) / runs_completed if runs_completed else 0

    # Populate final consolidated metrics dictionary
    metrics.update({
        # TASK PERFORMANCE(RQ1)
        "total_tasks": total_tasks,
        "runs_completed": runs_completed,
        "successes": successes,
        "failures": failures,
        "errors": errors,

        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "completion_rate": completion_rate,
        "error_rate": error_rate,

        "mean_reward": mean_reward,

        "total_tool_calls": total_tool_calls,
        "avg_tool_calls": avg_tool_calls,
        "avg_turns_to_success": avg_turns_to_success,
        "avg_turns_to_failure": avg_turns_to_failure,
        "avg_turns": avg_turns,

        # RELIABILITY (RQ2)
        "tle_failures": tle_failures,
        "if_failures": if_failures,
        "ia_failures": ia_failures,
        "task_errors": task_errors,
        "completed_failures": completed_failures,

        "tle_rate": tle_failures / denom,
        "if_rate": if_failures / denom,
        "ia_rate": ia_failures / denom,
        "task_error_rate": task_errors / denom,
        "completed_failure_rate": completed_failures / denom,

        "agent_failed": agent_failed,
        "interact_failed": interact_failed,
        "start_failed": start_failed,

        "agent_failed_error_rate": agent_failed / denom,
        "interact_failed_error_rate": interact_failed / denom,
        "start_failed_error_rate": start_failed / denom,
        "error_rate": errors / denom,

    })

    # EFFICIENCY
    if resource_file.exists():
        with open(resource_file) as f:
            resource_metrics = json.load(f)
        metrics.update(resource_metrics)
        
        eb_total = resource_metrics.get("total_energy_joules", 0)
        metrics["energy_per_task"] = eb_total / total_tasks if total_tasks else 0
        metrics["energy_per_success"] = eb_total / successes if successes else float("inf")
        metrics["energy_per_action"] = eb_total / total_tool_calls if total_tool_calls else 0
    else:
        print(f"No resource_metrics.json found for {runs_file}")

    return metrics


def run(outputs_dir, results_dir):

    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []

    for run_folder in outputs_dir.iterdir():

        if not run_folder.is_dir():
            continue

        config_file = run_folder / "config.yaml"
        if not config_file.exists():
            continue

        model, size, quant, domain = parse_metadata(config_file)

        run_id = run_folder.name.split("-")[-1]

        # find agent folder safely
        agent_folders = [p for p in run_folder.iterdir() if p.is_dir()]
        if not agent_folders:
            print(f"No agent folder found in {run_folder}")
            continue

        agent_folder = agent_folders[0]

        domain_folder = agent_folder / f"{domain}-std"

        runs_file = domain_folder / "runs.jsonl"
        error_file = domain_folder / "error.jsonl"
        resource_file = run_folder / "resource_metrics.json"

        if not runs_file.exists():
            print(f"Skipping run (missing runs.jsonl): {run_folder}")
            continue

        metrics = compute_metrics(runs_file, error_file, resource_file)

        output_dir = runs_dir / model / size / quant / domain
        output_dir.mkdir(parents=True, exist_ok=True)

        result_file = output_dir / f"{run_id}.json"

        result = {
            "model": model,
            "size": size,
            "quant": quant,
            "domain": domain,
            "run": run_id,
            "metrics": metrics
        }

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        all_runs.append(result)

    # Save master file with all runs
    master_file = results_dir / "all_runs_master.json"
    with open(master_file, "w") as f:
        json.dump(all_runs, f, indent=2)
    # print("Saved all runs to master file:", master_file)
