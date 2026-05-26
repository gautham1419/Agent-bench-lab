import os
import json
import yaml
from pathlib import Path

def parse_agent_string(agent_str):
    parts = agent_str.split("-")
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
        
    return model, size, quant

def extract_score(out):
    res = (out or {}).get("result") or {}
    metrics = res.get("metrics") or {}
    
    if metrics.get("score") is not None:
        return float(metrics["score"])
        
    reward = res.get("reward")
    if reward is not None:
        return float(reward)
        
    return None

def has_tool_format_error(messages):
    for m in messages:
        if "tool_calls" in m:
            for call in m["tool_calls"]:
                args = call.get("function", {}).get("arguments")
                if args == "{}" or args == "" or args is None:
                    return True
    return False

def calculate_metrics_for_run(run_folder):
    config_file = run_folder / "config.yaml"
    if not config_file.exists():
        return None
        
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    assignment = config.get("assignments", [{}])[0]
    
    agent_val = assignment.get("agent", "")
    agent_str = agent_val[0] if isinstance(agent_val, list) else agent_val
        
    task_val = assignment.get("task", "")
    task_str = task_val[0] if isinstance(task_val, list) else task_val
        
    if not agent_str or not task_str:
        return None
        
    model, size, quant = parse_agent_string(agent_str)
    
    domain_full = task_str 
    domain = domain_full.split("-")[0] 
    if domain == "dbbench": domain = "db"
    if domain == "webshop": domain = "ws"
    if domain == "alfworld": domain = "alf"
    
    agent_folders = [p for p in run_folder.iterdir() if p.is_dir() and (p.name == agent_str or "ollama" in p.name or "deepseek" in p.name)]
    if agent_folders:
        log_dir = agent_folders[0] / domain_full
    else:
        log_dir = run_folder / domain_full
        
    runs_file = log_dir / "runs.jsonl"
    error_file = log_dir / "error.jsonl"
    resource_file = run_folder / "resource_metrics.json"
    
    if not runs_file.exists():
        return None
        
    # Standard Counters
    successes = 0
    failures = 0
    runs_completed = 0
    total_rewards = 0
    total_tool_calls = 0
    turns_to_success_total = 0
    turns_to_failure_total = 0
    
    # Native AgentBench Explicit Taxonomy
    task_limit_reached = 0      # TLE
    agent_invalid_action = 0     # IA
    invalid_format = 0           # IF
    task_error = 0               # CLE / Runtime
    completed_failure = 0        # Generic

    with open(runs_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            runs_completed += 1
            data = json.loads(line)
            out = data.get("output", {})
            res = out.get("result", {})
            
            score = extract_score(out)
            messages = res.get("messages", [])
            status = res.get("status", "")
            
            run_tool_calls = sum(len(m.get("tool_calls", [])) for m in messages if "tool_calls" in m)
            total_tool_calls += run_tool_calls
            
            conversational_turns = len(messages)
            
            if score is not None:
                total_rewards += score
                if score > 0:
                    successes += 1
                    turns_to_success_total += conversational_turns
                else:
                    failures += 1
                    turns_to_failure_total += conversational_turns
                    
                    # Natively parse failure taxonomy string priorities
                    if has_tool_format_error(messages):
                        # Formatting failure guarantees the LLM strictly lost syntax capability
                        invalid_format += 1
                    elif status == "task limit reached":
                        task_limit_reached += 1
                    elif status == "agent invalid action":
                        agent_invalid_action += 1
                    elif status == "task error":
                        task_error += 1
                    elif status == "completed":
                        completed_failure += 1
            
    # Parse Master Hardware Crashes
    errors = 0
    if error_file.exists():
        with open(error_file, 'r') as f:
            errors = sum(1 for line in f if line.strip())
            
    total_tasks = runs_completed + errors
    success_rate = successes / total_tasks if total_tasks else 0
    
    metrics = {
        "total_tasks": total_tasks,
        "runs_completed": runs_completed,
        "successes": successes,
        "failures": failures,
        "errors": errors, 
        
        # New Strict Taxonomy (RQ2)
        "task_limit_reached": task_limit_reached,
        "agent_invalid_action": agent_invalid_action,
        "invalid_format": invalid_format,
        "task_error": task_error,
        "completed_failure": completed_failure,
        
        "success_rate": success_rate,
        "crash_rate": (errors + task_error) / total_tasks if total_tasks else 0,
        "mean_reward": total_rewards / runs_completed if runs_completed else 0,
        "avg_tool_calls": total_tool_calls / runs_completed if runs_completed else 0,
        "avg_turns_to_success": turns_to_success_total / successes if successes else 0,
        "avg_turns_to_failure": turns_to_failure_total / failures if failures else 0,
    }
    
    # Parse Hardware Resources (RQ3)
    if resource_file.exists():
        with open(resource_file, 'r') as f:
            res_data = json.load(f)
            eb_total = res_data.get("total_energy_joules", 0)
            metrics["total_energy_joules"] = eb_total
            metrics["gpu_energy_joules"] = res_data.get("gpu_energy_joules", 0)
            metrics["cpu_energy_joules"] = res_data.get("cpu_energy_joules", 0)
            metrics["gpu_mem_peak"] = res_data.get("gpu_mem_peak", 0)
            metrics["ram_peak"] = res_data.get("ram_peak", 0)
            metrics["cpu_avg"] = res_data.get("cpu_avg", 0)
            metrics["gpu_util_avg"] = res_data.get("gpu_util_avg", 0)
            metrics["duration"] = res_data.get("duration", 0)
            
            metrics["energy_per_task"] = eb_total / total_tasks if total_tasks else 0
            metrics["energy_per_success"] = eb_total / successes if successes else float("inf")
            metrics["energy_per_action"] = eb_total / total_tool_calls if total_tool_calls else 0
            metrics["latency_per_turn"] = metrics["duration"] / (turns_to_success_total + turns_to_failure_total) if (turns_to_success_total + turns_to_failure_total) else 0

    return {
        "model": model,
        "size": size,
        "quant": quant,
        "domain": domain,
        "agent_name": agent_str,
        "metrics": metrics
    }

def main():
    outputs_dir = Path("new_outputs")
    results_dir = Path("computed_results")
    results_dir.mkdir(exist_ok=True)
    
    if not outputs_dir.exists():
        print("Error: new_outputs/ folder not found!")
        return
        
    all_results = []
    
    print("Executing pure mathematical AgentBench error extraction...")
    for run_folder in outputs_dir.iterdir():
        if not run_folder.is_dir():
            continue
            
        folder_name = run_folder.name.lower()
            
        res = calculate_metrics_for_run(run_folder)
        if res:
            env_dir = results_dir / res['domain']
            env_dir.mkdir(exist_ok=True)
            
            # Try parsing run1, run2, or run3 naming
            run_id = "1"
            if "run2" in folder_name:
                run_id = "2"
            elif "run3" in folder_name:
                run_id = "3"
                
            res['run_id'] = run_id
            
            model_variant = f"{res['model']}_{res['size']}_{res['quant']}"
            out_file = env_dir / f"{model_variant}_run{run_id}.json"
            
            with open(out_file, 'w') as f:
                json.dump(res, f, indent=4)
                
            all_results.append(res)
            
    # Save flat list of all parsed runs
    with open(results_dir / "master_computed_all_runs.json", 'w') as f:
        json.dump(all_results, f, indent=4)
        
    mean_dir = results_dir / "mean"
    mean_dir.mkdir(exist_ok=True)
    
    aggregated_results = []
    from collections import defaultdict
    import math
    
    grouped = defaultdict(list)
    for r in all_results:
        key = (r["model"], r["size"], r["quant"], r["domain"], r["agent_name"])
        grouped[key].append(r)
        
    for key, runs in grouped.items():
        if not runs: continue
        model, size, quant, domain, agent_name = key
        
        avg_metrics = {}
        metric_keys = runs[0]["metrics"].keys()
        
        for mk in metric_keys:
            valid_vals = [r["metrics"][mk] for r in runs if isinstance(r["metrics"][mk], (int, float))]
            if not valid_vals:
                avg_metrics[mk] = 0
                continue
                
            # Treat infinity gracefully across averages
            non_inf = [v for v in valid_vals if not math.isinf(v)]
            if not non_inf:
                avg_metrics[mk] = float("inf")
            else:
                avg_metrics[mk] = sum(non_inf) / len(non_inf)
                
        mean_data = {
            "model": model,
            "size": size,
            "quant": quant,
            "domain": domain,
            "agent_name": agent_name,
            "metrics": avg_metrics
        }
        aggregated_results.append(mean_data)
        
        # Save individual mean files for transparency
        model_variant = f"{model}_{size}_{quant}"
        env_mean_dir = mean_dir / domain
        env_mean_dir.mkdir(exist_ok=True)
        out_file = env_mean_dir / f"{model_variant}_mean.json"
        with open(out_file, 'w') as f:
            json.dump(mean_data, f, indent=4)
            
    with open(mean_dir / "master_mean.json", 'w') as f:
        json.dump(aggregated_results, f, indent=4)
        
    print(f"Successfully extracted {len(all_results)} runs and aggregated into {len(aggregated_results)} mean configurations inside computed_results/mean/!")

if __name__ == "__main__":
    main()
