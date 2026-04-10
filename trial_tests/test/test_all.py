import json
from pathlib import Path
from collections import Counter


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


RUNS = {
    "os": Path(r"F:\workstation_agentbench\Agent-bench-lab\outputs\2026-02-05-07-48-57_os_ollama"),
    "dbbench": Path(r"F:\workstation_agentbench\Agent-bench-lab\outputs\2026-02-05-01-13-31"),
    "webshop": Path(r"F:\workstation_agentbench\Agent-bench-lab\outputs\2026-02-05-13-18-40_webshop_ollama"),
}


MODEL_SIZE_B = {
    "ollama-gemma2-2b": 2,
    "ollama-qwen-4b": 4,
    "ollama-qwen-8b": 8,
    "ollama-gemma2-9b": 9,
}


INVALID_TOOLCALL_MARKERS = [
    "No valid tool call found from agent",
    "No executable tool calls found",
    "You should call a tool instead",
    "Please call a tool instead",
]


def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_time_ts_ms(rec: dict):
    t = (rec or {}).get("time") or {}
    ts = t.get("timestamp")
    if ts is None:
        return np.nan
    try:
        return float(ts)
    except Exception:
        return np.nan


def count_tool_calls(messages):
    if not isinstance(messages, list):
        return 0
    c = 0
    for m in messages:
        tc = (m or {}).get("tool_calls", None)
        if tc:
            c += len(tc)
    return c


def count_invalid_toolcall_markers(messages):
    if not isinstance(messages, list):
        return 0
    hits = 0
    for m in messages:
        content = (m or {}).get("content") or ""
        if not isinstance(content, str):
            continue
        for marker in INVALID_TOOLCALL_MARKERS:
            if marker in content:
                hits += 1
                break
    return hits


def iter_task_dirs(run_dir: Path):
    for agent_dir in run_dir.iterdir():
        if not agent_dir.is_dir():
            continue
        agent = agent_dir.name
        for task_dir in agent_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            yield agent, task, task_dir


def load_runs(run_name: str, run_dir: Path) -> pd.DataFrame:
    rows = []
    for agent, task, task_dir in iter_task_dirs(run_dir):
        runs_path = task_dir / "runs.jsonl"
        if not runs_path.exists():
            continue
        for rec in iter_jsonl(runs_path):
            out = (rec or {}).get("output") or {}
            result = out.get("result") or {}
            messages = result.get("messages") or []
            reward = result.get("reward")
            rows.append({
                "run": run_name,
                "agent": agent,
                "model_size_b": MODEL_SIZE_B.get(agent),
                "task": task,
                "index": rec.get("index"),
                "status": out.get("status"),
                "reward": reward,
                "time_ts_ms": extract_time_ts_ms(rec),
                "turns": len(messages) if isinstance(messages, list) else np.nan,
                "tool_calls": count_tool_calls(messages),
                "invalid_toolcall_events": count_invalid_toolcall_markers(messages),
            })
    return pd.DataFrame(rows)


def load_errors(run_name: str, run_dir: Path) -> pd.DataFrame:
    rows = []
    for agent, task, task_dir in iter_task_dirs(run_dir):
        err_path = task_dir / "error.jsonl"
        if not err_path.exists():
            continue
        for rec in iter_jsonl(err_path):
            rows.append({
                "run": run_name,
                "agent": agent,
                "model_size_b": MODEL_SIZE_B.get(agent),
                "task": task,
                "index": (rec or {}).get("index"),
                "error": (rec or {}).get("error"),
                "info": (rec or {}).get("info"),
                "time_ts_ms": extract_time_ts_ms(rec),
            })
    return pd.DataFrame(rows)


def summarize_basic(df_runs: pd.DataFrame) -> pd.DataFrame:
    grp = df_runs.groupby(["run", "task", "agent"], dropna=False)
    summary = grp.agg(
        n=("index", "count"),
        mean_reward=("reward", "mean"),
        completion_rate=("status", lambda s: (s == "completed").mean()),
    ).reset_index()
    summary["model_size_b"] = summary["agent"].map(MODEL_SIZE_B)
    return summary


def approx_seconds_per_episode_from_timestamps(ts_ms: pd.Series):
    ts = ts_ms.dropna().astype(float).values
    if ts.size < 2:
        return np.nan
    ts = np.sort(ts)
    dt = np.diff(ts) / 1000.0
    dt = dt[(dt > 0) & (dt < 600)]
    if dt.size == 0:
        return np.nan
    return float(np.mean(dt))


def summarize_extended(df_runs: pd.DataFrame, df_err: pd.DataFrame) -> pd.DataFrame:
    if df_err is None or df_err.empty:
        df_err = pd.DataFrame(columns=["run", "task", "agent", "index", "error", "time_ts_ms"])


    run_grp = df_runs.groupby(["run", "task", "agent"], dropna=False)
    err_grp = df_err.groupby(["run", "task", "agent"], dropna=False)


    rows = []
    keys = set(run_grp.groups.keys()) | set(err_grp.groups.keys())
    for key in sorted(keys):
        run, task, agent = key
        sub_runs = run_grp.get_group(key) if key in run_grp.groups else df_runs.iloc[0:0]
        sub_err = err_grp.get_group(key) if key in err_grp.groups else df_err.iloc[0:0]


        n_runs = int(len(sub_runs))
        n_err = int(len(sub_err))
        n_total = n_runs + n_err if (n_runs + n_err) > 0 else 1


        rewards = pd.to_numeric(sub_runs.get("reward"), errors="coerce")
        mean_reward = float(rewards.mean()) if n_runs > 0 else np.nan


        completion_rate = float((sub_runs.get("status") == "completed").mean()) if n_runs > 0 else np.nan


        success_rate_strict = float((rewards == 1).mean()) if n_runs > 0 else np.nan
        success_rate_gt0 = float((rewards > 0).mean()) if n_runs > 0 else np.nan


        crash_rate = float(n_err / n_total)
        agent_failed_rate = float((sub_err.get("error") == "AGENT_FAILED").mean()) if n_err > 0 else 0.0


        invalid_events = pd.to_numeric(sub_runs.get("invalid_toolcall_events"), errors="coerce").fillna(0)
        invalid_episode_rate = float((invalid_events > 0).mean()) if n_runs > 0 else np.nan
        toolcall_valid_episode_rate = float(1.0 - invalid_episode_rate) if pd.notna(invalid_episode_rate) else np.nan
        avg_invalid_events = float(invalid_events.mean()) if n_runs > 0 else np.nan


        turns = pd.to_numeric(sub_runs.get("turns"), errors="coerce")
        tool_calls = pd.to_numeric(sub_runs.get("tool_calls"), errors="coerce")
        avg_turns = float(turns.mean()) if n_runs > 0 else np.nan
        avg_tool_calls = float(tool_calls.mean()) if n_runs > 0 else np.nan
        tool_calls_per_turn = float((tool_calls / turns).replace([np.inf, -np.inf], np.nan).mean()) if n_runs > 0 else np.nan


        approx_sec_per_ep = approx_seconds_per_episode_from_timestamps(sub_runs.get("time_ts_ms", pd.Series(dtype=float)))


        rows.append({
            "run": run,
            "task": task,
            "agent": agent,
            "model_size_b": MODEL_SIZE_B.get(agent),


            "n_runs": n_runs,
            "n_errors": n_err,
            "n_total": n_runs + n_err,


            "mean_reward": mean_reward,
            "completion_rate": completion_rate,


            "success_rate_strict_reward_eq_1": success_rate_strict,
            "success_rate_reward_gt_0": success_rate_gt0,


            "crash_rate": crash_rate,
            "agent_failed_rate": agent_failed_rate,


            "invalid_toolcall_episode_rate": invalid_episode_rate,
            "toolcall_valid_episode_rate": toolcall_valid_episode_rate,
            "avg_invalid_toolcall_events": avg_invalid_events,


            "avg_turns": avg_turns,
            "avg_tool_calls": avg_tool_calls,
            "avg_tool_calls_per_turn": tool_calls_per_turn,


            "approx_seconds_per_episode": approx_sec_per_ep,
        })


    return pd.DataFrame(rows)


def status_breakdown(df_runs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (run, task, agent), sub in df_runs.groupby(["run", "task", "agent"], dropna=False):
        counts = Counter(sub["status"].fillna("null").tolist())
        total = sum(counts.values()) or 1
        for status, c in counts.items():
            out.append({
                "run": run,
                "task": task,
                "agent": agent,
                "status": status,
                "fraction": c / total,
                "model_size_b": MODEL_SIZE_B.get(agent),
            })
    return pd.DataFrame(out)


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    sns.set_theme(style="whitegrid", font_scale=1.1)


    df_runs = pd.concat([load_runs(k, v) for k, v in RUNS.items()], ignore_index=True)
    df_err = pd.concat([load_errors(k, v) for k, v in RUNS.items()], ignore_index=True)


    out_dir = Path(r"F:\workstation_agentbench\Agent-bench-lab\paper_figures")
    out_dir.mkdir(parents=True, exist_ok=True)


    df_runs.to_csv(out_dir / "all_samples.csv", index=False)
    df_err.to_csv(out_dir / "all_errors.csv", index=False)


    summ = summarize_basic(df_runs)
    summ.to_csv(out_dir / "summary.csv", index=False)


    summ_ex = summarize_extended(df_runs, df_err)
    summ_ex.to_csv(out_dir / "summary_extended.csv", index=False)


    for task in sorted(summ["task"].unique()):
        sub = summ[summ["task"] == task].sort_values(["model_size_b", "agent"])
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(data=sub, x="agent", y="mean_reward", hue="run")
        ax.set_title(f"Mean reward — {task}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Mean reward")
        plt.xticks(rotation=30, ha="right")
        save_fig(out_dir / f"fig_mean_reward_{task}.png")


    for task in sorted(summ["task"].unique()):
        sub = summ[summ["task"] == task].sort_values(["model_size_b", "agent"])
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(data=sub, x="agent", y="completion_rate", hue="run")
        ax.set_title(f"Completion rate — {task}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Completion rate")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=30, ha="right")
        save_fig(out_dir / f"fig_completion_rate_{task}.png")


    sb = status_breakdown(df_runs)
    for task in sorted(sb["task"].unique()):
        sub = sb[sb["task"] == task].copy()
        sub = sub.sort_values(["model_size_b", "agent", "status"])
        pivot = sub.pivot_table(index="agent", columns="status", values="fraction", aggfunc="sum", fill_value=0)
        pivot = pivot.loc[sorted(pivot.index, key=lambda a: (MODEL_SIZE_B.get(a, 999), a))]


        plt.figure(figsize=(10, 4))
        bottom = None
        for col in pivot.columns:
            vals = pivot[col].values
            if bottom is None:
                plt.bar(pivot.index, vals, label=col)
                bottom = vals
            else:
                plt.bar(pivot.index, vals, bottom=bottom, label=col)
                bottom = bottom + vals
        plt.title(f"Outcome breakdown — {task}")
        plt.xlabel("Model")
        plt.ylabel("Fraction")
        plt.xticks(rotation=30, ha="right")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        save_fig(out_dir / f"fig_status_breakdown_{task}.png")


    for task in sorted(summ_ex["task"].unique()):
        sub = summ_ex[summ_ex["task"] == task].sort_values(["model_size_b", "agent"])
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(data=sub, x="agent", y="success_rate_reward_gt_0", hue="run")
        ax.set_title(f"Task success rate (reward > 0) — {task}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Success rate")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=30, ha="right")
        save_fig(out_dir / f"fig_success_rate_gt0_{task}.png")


    for task in sorted(summ_ex["task"].unique()):
        sub = summ_ex[summ_ex["task"] == task].sort_values(["model_size_b", "agent"])
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(data=sub, x="agent", y="crash_rate", hue="run")
        ax.set_title(f"Crash rate (error.jsonl / total) — {task}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Crash rate")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=30, ha="right")
        save_fig(out_dir / f"fig_crash_rate_{task}.png")


    plt.figure(figsize=(7, 4))
    sub = summ_ex.dropna(subset=["avg_tool_calls", "mean_reward"]).copy()
    ax = sns.scatterplot(data=sub, x="avg_tool_calls", y="mean_reward", hue="task", style="agent", s=120)
    ax.set_title("Efficiency–performance (tool calls vs reward)")
    ax.set_xlabel("Avg tool calls / episode")
    ax.set_ylabel("Mean reward")
    save_fig(out_dir / "fig_efficiency_toolcalls_vs_reward.png")


    sub_lat = summ_ex.dropna(subset=["approx_seconds_per_episode", "mean_reward"]).copy()
    if not sub_lat.empty:
        plt.figure(figsize=(7, 4))
        ax = sns.scatterplot(data=sub_lat, x="approx_seconds_per_episode", y="mean_reward", hue="task", style="agent", s=120)
        ax.set_title("Latency proxy–performance (sec/episode vs reward)")
        ax.set_xlabel("Approx seconds per episode (timestamp diff, filtered)")
        ax.set_ylabel("Mean reward")
        save_fig(out_dir / "fig_latency_proxy_vs_reward.png")


    print(f"Wrote figures + CSVs to: {out_dir}")


if __name__ == "__main__":
    main()


