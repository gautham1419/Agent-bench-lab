import json
from pathlib import Path
import pandas as pd

def check_means():
    with open("results/all_runs_master.json") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df["success_rate"] = df["metrics"].apply(lambda m: m["success_rate"])
    
    print("=== Success Rate by Model and Size ===")
    print(df.groupby(["model", "size"])["success_rate"].mean())
    print()
    
    print("=== Success Rate by Size ===")
    print(df.groupby("size")["success_rate"].mean())

if __name__ == "__main__":
    check_means()
