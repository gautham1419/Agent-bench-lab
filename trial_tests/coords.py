import json
import pandas as pd

def main():
    with open("results/all_runs_master.json") as f:
        data = json.load(f)
    
    rows = []
    for entry in data:
        row = {
            "model": entry["model"],
            "size": entry["size"],
            "quant": entry["quant"],
            "domain": entry["domain"],
            "success_rate": entry["metrics"]["success_rate"],
            "total_energy": entry["metrics"]["total_energy_joules"],
            "total_tasks": entry["metrics"]["total_tasks"]
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    # Aggregate by configuration: (model, size, quant)
    agg = df.groupby(["model", "size", "quant"]).agg(
        success_rate=("success_rate", "mean"),
        total_energy=("total_energy", "sum"),
        total_tasks=("total_tasks", "sum")
    ).reset_index()
    
    agg["energy_per_task_kJ"] = (agg["total_energy"] / agg["total_tasks"]) / 1000.0
    
    print("=== Configuration Coordinates ===")
    print(agg[["model", "size", "quant", "success_rate", "energy_per_task_kJ"]].to_string(index=False))

if __name__ == "__main__":
    main()
