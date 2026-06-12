import os
import yaml
from pathlib import Path

def list_all_7b_f16():
    outputs_dir = Path("new_outputs")
    found = []
    for run_folder in sorted(outputs_dir.iterdir()):
        if not run_folder.is_dir():
            continue
        if "deepseek-r1-qwen-7b-f16" in run_folder.name:
            config_file = run_folder / "config.yaml"
            task = "N/A"
            agent = "N/A"
            if config_file.exists():
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                assignment = config.get("assignments", [{}])[0]
                task = assignment.get("task", "N/A")
                agent = assignment.get("agent", "N/A")
            
            subdirs = [p for p in run_folder.iterdir() if p.is_dir()]
            has_runs = False
            for sd in subdirs:
                for subsd in sd.iterdir():
                    if subsd.is_dir():
                        runs_file = subsd / "runs.jsonl"
                        if runs_file.exists() and os.path.getsize(runs_file) > 0:
                            has_runs = True
            
            found.append({
                "folder": run_folder.name,
                "agent": agent,
                "task": task,
                "has_runs": has_runs
            })
            
    print("All deepseek-r1-qwen-7b-f16 runs:")
    for f in found:
        print(f"Folder: {f['folder']}")
        print(f"  Agent: {f['agent']}")
        print(f"  Task:  {f['task']}")
        print(f"  Has runs: {f['has_runs']}")

if __name__ == "__main__":
    list_all_7b_f16()
