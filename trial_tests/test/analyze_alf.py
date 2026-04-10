import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


WS_TOKEN_RE = re.compile(r"\S+")


def est_tokens(text: str) -> int:
    if not text:
        return 0
    return len(WS_TOKEN_RE.findall(text))


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


@dataclass
class RunMetrics:
    model: str
    index: int
    status: str
    reward: float

    est_total_tokens: int
    est_assistant_tokens: int

    tool_calls_total: int
    action_calls: int
    answer_calls: int
    finish_calls: int
    other_tool_calls: int

    action_args_parse_ok: int
    action_args_nonempty: int


def extract_tool_calls_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        tcs = m.get("tool_calls", []) or []
        if isinstance(tcs, list):
            tool_calls.extend([tc for tc in tcs if isinstance(tc, dict)])
    return tool_calls


def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def parse_run_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None


def collect_metrics_for_runs_jsonl(model_name: str, runs_path: Path) -> List[RunMetrics]:
    out: List[RunMetrics] = []
    with runs_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            rec = parse_run_line(raw_line)
            if not rec:
                continue

            idx = rec.get("index")
            tco = rec.get("output") or {}
            status = str(tco.get("status") or "unknown")

            result = (tco.get("result") or {}) if isinstance(tco, dict) else {}
            reward = result.get("reward", 0)
            try:
                reward_f = float(reward)
            except Exception:
                reward_f = 0.0

            messages = result.get("messages", [])
            if not isinstance(messages, list):
                messages = []

            # Token estimate: count whitespace-separated tokens across all message contents
            total_tok = 0
            assistant_tok = 0
            for m in messages:
                if not isinstance(m, dict):
                    continue
                content = m.get("content")
                if isinstance(content, str):
                    t = est_tokens(content)
                    total_tok += t
                    if m.get("role") == "assistant":
                        assistant_tok += t

                # Also count tool arguments as tokens (important for action arguments)
                tcs = m.get("tool_calls", []) or []
                if isinstance(tcs, list):
                    for tc in tcs:
                        if not isinstance(tc, dict):
                            continue
                        fn = (tc.get("function") or {})
                        args = fn.get("arguments")
                        if isinstance(args, str):
                            total_tok += est_tokens(args)
                            if m.get("role") == "assistant":
                                assistant_tok += est_tokens(args)

            tool_calls = extract_tool_calls_from_messages(messages)

            action_calls = 0
            answer_calls = 0
            finish_calls = 0
            other_calls = 0

            action_args_parse_ok = 0
            action_args_nonempty = 0

            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                if name == "alfworld_action":
                    action_calls += 1
                    args = fn.get("arguments")
                    args_obj = safe_json_loads(args) if isinstance(args, str) else None
                    if isinstance(args_obj, dict):
                        action_args_parse_ok += 1
                        action_text = args_obj.get("action")
                        if isinstance(action_text, str) and action_text.strip():
                            action_args_nonempty += 1
                elif name == "answer_action":
                    answer_calls += 1
                elif name == "finish_action":
                    finish_calls += 1
                else:
                    other_calls += 1

            if not isinstance(idx, int):
                # some logs might have non-int indices; skip those
                continue

            out.append(
                RunMetrics(
                    model=model_name,
                    index=idx,
                    status=status,
                    reward=reward_f,
                    est_total_tokens=total_tok,
                    est_assistant_tokens=assistant_tok,
                    tool_calls_total=len(tool_calls),
                    action_calls=action_calls,
                    answer_calls=answer_calls,
                    finish_calls=finish_calls,
                    other_tool_calls=other_calls,
                    action_args_parse_ok=action_args_parse_ok,
                    action_args_nonempty=action_args_nonempty,
                )
            )

    return out


def find_model_runs(root: Path) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        runs = model_dir / "alfworld-std" / "runs.jsonl"
        if runs.exists():
            pairs.append((model_dir.name, runs))
    return sorted(pairs, key=lambda x: x[0])


def plot_bar_with_ci(df_model: pd.DataFrame, metric_col: str, ci_lo: str, ci_hi: str, title: str, ylabel: str, outpath: Path):
    order = df_model.sort_values(metric_col, ascending=False)["model"].tolist()
    plt.figure(figsize=(10, 4.8))
    ax = sns.barplot(data=df_model, x="model", y=metric_col, order=order)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)

    # error bars
    for i, m in enumerate(order):
        row = df_model[df_model["model"] == m].iloc[0]
        y = float(row[metric_col])
        lo = float(row[ci_lo])
        hi = float(row[ci_hi])
        ax.errorbar(i, y, yerr=[[y - lo], [hi - y]], fmt="none", c="black", capsize=4, linewidth=1)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to the timestamped output dir (contains model subdirs)")
    ap.add_argument("--outdir", required=True, help="Where to write CSV + figures")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_runs = find_model_runs(root)
    if not model_runs:
        raise SystemExit(f"No runs.jsonl found under: {root}")

    all_runs: List[RunMetrics] = []
    for model, runs_path in model_runs:
        all_runs.extend(collect_metrics_for_runs_jsonl(model, runs_path))

    df = pd.DataFrame([r.__dict__ for r in all_runs])

    # Per-run derived fields
    df["is_completed"] = df["status"].str.lower().eq("completed")
    df["is_success"] = df["reward"].astype(float).ge(1.0)
    df["is_task_limit"] = df["status"].str.lower().eq("task limit reached")
    df["is_agent_invalid_action"] = df["status"].str.lower().eq("agent invalid action")
    df["is_task_error"] = df["status"].str.lower().eq("task error")

    # Validity definition (for paper): alfworld_action calls where arguments parse + non-empty action
    df["action_valid_calls"] = df["action_args_nonempty"]
    df["action_valid_rate"] = df.apply(lambda r: (r["action_valid_calls"] / r["action_calls"]) if r["action_calls"] > 0 else 1.0, axis=1)

    # Aggregate per model
    rows = []
    for model, g in df.groupby("model"):
        n = len(g)
        succ = int(g["is_success"].sum())
        comp = int(g["is_completed"].sum())
        task_limit = int(g["is_task_limit"].sum())
        agent_invalid = int(g["is_agent_invalid_action"].sum())
        task_err = int(g["is_task_error"].sum())

        succ_rate = succ / n if n else 0.0
        comp_rate = comp / n if n else 0.0

        succ_ci = wilson_ci(succ, n)
        comp_ci = wilson_ci(comp, n)

        # avg token estimate for completed tasks only (to avoid mixing unfinished)
        completed = g[g["is_completed"]]
        avg_tok_completed = float(completed["est_total_tokens"].mean()) if len(completed) else 0.0
        avg_asst_tok_completed = float(completed["est_assistant_tokens"].mean()) if len(completed) else 0.0

        # action validity across all action calls
        action_calls_total = int(g["action_calls"].sum())
        action_valid_total = int(g["action_args_nonempty"].sum())
        action_valid_rate = (action_valid_total / action_calls_total) if action_calls_total else 1.0
        action_valid_ci = wilson_ci(action_valid_total, action_calls_total) if action_calls_total else (1.0, 1.0)

        rows.append(
            dict(
                model=model,
                n_tasks=n,
                success_rate=succ_rate,
                success_ci_lo=succ_ci[0],
                success_ci_hi=succ_ci[1],
                completion_rate=comp_rate,
                completion_ci_lo=comp_ci[0],
                completion_ci_hi=comp_ci[1],
                task_limit_rate=(task_limit / n) if n else 0.0,
                agent_invalid_action_rate=(agent_invalid / n) if n else 0.0,
                task_error_rate=(task_err / n) if n else 0.0,
                avg_tool_calls=float(g["tool_calls_total"].mean()) if n else 0.0,
                avg_action_calls=float(g["action_calls"].mean()) if n else 0.0,
                action_validity_rate=action_valid_rate,
                action_validity_ci_lo=action_valid_ci[0],
                action_validity_ci_hi=action_valid_ci[1],
                avg_est_tokens_completed=avg_tok_completed,
                avg_est_assistant_tokens_completed=avg_asst_tok_completed,
            )
        )

    df_model = pd.DataFrame(rows).sort_values("model")
    df_model.to_csv(outdir / "alfworld_model_metrics.csv", index=False)
    df.to_csv(outdir / "alfworld_per_run_metrics.csv", index=False)

    sns.set_theme(style="whitegrid")

    # 1) Success rate with 95% CI
    plot_bar_with_ci(
        df_model,
        metric_col="success_rate",
        ci_lo="success_ci_lo",
        ci_hi="success_ci_hi",
        title="ALFWorld Success Rate (reward==1) with 95% Wilson CI",
        ylabel="Success rate",
        outpath=outdir / "success_rate.png",
    )

    # 2) Completion rate with 95% CI
    plot_bar_with_ci(
        df_model,
        metric_col="completion_rate",
        ci_lo="completion_ci_lo",
        ci_hi="completion_ci_hi",
        title="ALFWorld Completion Rate (status==completed) with 95% Wilson CI",
        ylabel="Completion rate",
        outpath=outdir / "completion_rate.png",
    )

    # 3) Action validity rate (alfworld_action args parse + non-empty action) with 95% CI on calls
    plot_bar_with_ci(
        df_model.assign(
            action_ci_lo=df_model["action_validity_ci_lo"],
            action_ci_hi=df_model["action_validity_ci_hi"],
        ),
        metric_col="action_validity_rate",
        ci_lo="action_ci_lo",
        ci_hi="action_ci_hi",
        title="Action Validity Rate (alfworld_action has parseable args + non-empty action)",
        ylabel="Validity rate",
        outpath=outdir / "action_validity_rate.png",
    )

    # 4) Efficiency vs performance tradeoff (token estimate vs success)
    plt.figure(figsize=(7.6, 5.2))
    ax = sns.scatterplot(
        data=df_model,
        x="avg_est_tokens_completed",
        y="success_rate",
        hue="model",
        s=120,
    )
    ax.set_title("Efficiency–Performance Tradeoff\n(x = avg estimated tokens per completed task, y = success rate)")
    ax.set_xlabel("Avg estimated tokens per completed task")
    ax.set_ylabel("Success rate (reward==1)")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(outdir / "tradeoff_success_vs_tokens.png", dpi=220)
    plt.close()

    # 5) Extra plot: avg action calls vs success
    plt.figure(figsize=(7.6, 5.2))
    ax = sns.scatterplot(
        data=df_model,
        x="avg_action_calls",
        y="success_rate",
        hue="model",
        s=120,
    )
    ax.set_title("Action-use vs Performance\n(x = avg alfworld_action calls per task, y = success rate)")
    ax.set_xlabel("Avg alfworld_action calls per task")
    ax.set_ylabel("Success rate (reward==1)")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(outdir / "success_vs_action_calls.png", dpi=220)
    plt.close()

    print(f"Wrote:\n- {outdir / 'alfworld_model_metrics.csv'}\n- {outdir / 'alfworld_per_run_metrics.csv'}\n- {outdir}/*.png")


if __name__ == "__main__":
    main()