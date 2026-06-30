"""
behavioral_consistency.py
=========================
Task-level behavioral consistency analysis under quantization.

Motivation
----------
TOST (tost_equivalence.py) establishes that quantized and full-precision
agents are equivalent *in aggregate*: the median paired success difference
lies within +/-5 pp.  Aggregate equivalence, however, can hide per-task
churn: a quantized model may fail tasks the full-precision model solved and
solve tasks it failed, in equal numbers.  For an engineer this distinction
matters - it decides whether a quantized agent can be validated by per-task
regression testing against the full-precision baseline (expecting identical
outcomes) or must be accepted with statistical criteria (expecting equal
rates but different individual outcomes).

Two analyses
------------
1. Outcome discordance (McNemar):
   For every paired task (same model, size, domain, replicate, task index),
   classify the (reference, test) outcome pair into concordant
   (both succeed / both fail) or discordant (flip).  Report the flip rate
   and test directional symmetry with an exact McNemar (binomial) test.
   A symmetric flip pattern with a non-trivial flip rate means quantization
   *re-randomises* borderline tasks rather than degrading specific ones.

2. Interaction verbosity (turns per task):
   Number of assistant turns in each task's message transcript.  Compared
   across precision levels with matched-pair Wilcoxon tests at the
   model x domain cell level.  This tests whether lower precision changes
   the *length* of the agent's decision process even when the outcome is
   unchanged (relevant to the elevated time-limit-exceeded residual for
   Q4_K_M observed in RQ2).

Outputs
-------
- statistical_tests/output/behavioral_consistency.json
- statistical_tests/output/flip_rates_table.csv
- statistical_tests/output/fig_behavioral_consistency.pdf
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tost_equivalence import get_domain, parse_run_metadata

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "new_outputs"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
COMPARISONS = [("q4_k_m", "bf16"), ("q8_0", "bf16")]
QUANT_LABELS = {"bf16": "BF16/F16", "q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}


# ── task-level extraction (success + turns) ────────────────────────────────


def load_task_records(run_folder: Path) -> list[dict] | None:
    """
    Parse runs.jsonl for one run folder.
    Returns [{task_index, success, n_turns}] or None.
    n_turns = number of assistant messages in the final transcript.
    """
    agent_dirs = [p for p in run_folder.iterdir() if p.is_dir()]
    if not agent_dirs:
        return None
    domain_dirs = [p for p in agent_dirs[0].iterdir() if p.is_dir()]
    if not domain_dirs:
        return None
    runs_file = domain_dirs[0] / "runs.jsonl"
    if not runs_file.exists():
        return None

    records = []
    with open(runs_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            output = d.get("output")
            if output is None:
                continue
            idx = output.get("index")
            if idx is None:
                continue
            res = output.get("result") or {}
            score = res.get("reward") or res.get("metrics", {}).get("score") or 0
            msgs = res.get("messages") or output.get("history") or []
            n_turns = sum(1 for m in msgs if m.get("role") == "assistant")
            records.append(
                {
                    "task_index": idx,
                    "success": 1 if (score and score > 0) else 0,
                    "n_turns": n_turns,
                }
            )
    return records or None


def build_dataset() -> pd.DataFrame:
    rows = []
    for folder in OUTPUTS_DIR.iterdir():
        if not folder.is_dir():
            continue
        meta = parse_run_metadata(folder.name)
        if meta is None:
            continue
        domain = get_domain(folder)
        if domain is None:
            continue
        recs = load_task_records(folder)
        if not recs:
            continue
        for r in recs:
            rows.append({**meta, "domain": domain, **r})
    df = pd.DataFrame(rows)
    df["quant"] = df["quant"].str.lower().replace({"f16": "bf16"})
    df = df.drop_duplicates(
        subset=["model", "size", "quant", "domain", "run", "task_index"]
    )
    return df


# ── analysis 1: outcome discordance (McNemar) ──────────────────────────────


def mcnemar_cell(merged: pd.DataFrame) -> dict:
    """Discordance statistics for a set of paired outcomes."""
    n = len(merged)
    n11 = int(((merged["s_ref"] == 1) & (merged["s_test"] == 1)).sum())
    n00 = int(((merged["s_ref"] == 0) & (merged["s_test"] == 0)).sum())
    n01 = int(((merged["s_ref"] == 1) & (merged["s_test"] == 0)).sum())  # regression
    n10 = int(((merged["s_ref"] == 0) & (merged["s_test"] == 1)).sum())  # improvement
    n_disc = n01 + n10
    flip_rate = n_disc / n if n else np.nan
    # exact McNemar: binomial test of n01 against Binomial(n_disc, 0.5)
    p_mcnemar = float(stats.binomtest(n01, n_disc, 0.5).pvalue) if n_disc > 0 else 1.0
    return {
        "n_pairs": n,
        "n_both_success": n11,
        "n_both_fail": n00,
        "n_regression": n01,
        "n_improvement": n10,
        "flip_rate": round(flip_rate, 4),
        "p_mcnemar": round(p_mcnemar, 6),
    }


def run_discordance(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cell_rows = []
    pooled = {}
    for test_q, ref_q in COMPARISONS:
        comp_key = f"{test_q} vs {ref_q}"
        merged_all = []
        for (model, size, domain), cell in df.groupby(["model", "size", "domain"]):
            test = cell[cell["quant"] == test_q]
            ref = cell[cell["quant"] == ref_q]
            if test.empty or ref.empty:
                continue
            merged = pd.merge(
                test[["run", "task_index", "success"]].rename(
                    columns={"success": "s_test"}
                ),
                ref[["run", "task_index", "success"]].rename(
                    columns={"success": "s_ref"}
                ),
                on=["run", "task_index"],
                how="inner",
            )
            if merged.empty:
                continue
            merged_all.append(merged)
            cell_rows.append(
                {
                    "comparison": comp_key,
                    "model": model,
                    "size": size,
                    "domain": domain,
                    **mcnemar_cell(merged),
                }
            )
        if merged_all:
            pooled[comp_key] = mcnemar_cell(pd.concat(merged_all, ignore_index=True))
    return pd.DataFrame(cell_rows), pooled


# ── analysis 2: interaction turns ──────────────────────────────────────────


def run_turns_analysis(df: pd.DataFrame) -> dict:
    """
    Mean turns per task by quant level (pooled and per cell), plus
    matched-pair Wilcoxon across the model x domain cells.
    """
    out = {"pooled_mean_turns": {}, "cell_tests": {}}
    for q in ["bf16", "q8_0", "q4_k_m"]:
        sub = df[df["quant"] == q]
        out["pooled_mean_turns"][q] = {
            "mean": round(float(sub["n_turns"].mean()), 3),
            "median": float(sub["n_turns"].median()),
            "n_tasks": int(len(sub)),
        }

    # cell-level mean turns
    cell = (
        df.groupby(["model", "size", "domain", "quant"])["n_turns"].mean().reset_index()
    )
    wide = cell.pivot_table(
        index=["model", "size", "domain"], columns="quant", values="n_turns"
    ).dropna()

    for test_q, ref_q in COMPARISONS:
        diff = (wide[test_q] - wide[ref_q]).values
        w, p = stats.wilcoxon(diff, alternative="two-sided")
        r = abs(stats.norm.ppf(p / 2)) / np.sqrt(len(diff)) if p > 0 else np.nan
        out["cell_tests"][f"{test_q} vs {ref_q}"] = {
            "n_cells": int(len(diff)),
            "median_diff_turns": round(float(np.median(diff)), 3),
            "mean_diff_turns": round(float(np.mean(diff)), 3),
            "wilcoxon_W": round(float(w), 1),
            "p_value": round(float(p), 4),
            "effect_size_r": round(float(r), 3) if not np.isnan(r) else None,
        }

    # turns conditioned on outcome (success vs failure), pooled
    out["turns_by_outcome"] = {}
    for q in ["bf16", "q8_0", "q4_k_m"]:
        sub = df[df["quant"] == q]
        out["turns_by_outcome"][q] = {
            "success_mean_turns": round(
                float(sub[sub.success == 1]["n_turns"].mean()), 3
            ),
            "failure_mean_turns": round(
                float(sub[sub.success == 0]["n_turns"].mean()), 3
            ),
        }
    return out, wide


# ── figure ─────────────────────────────────────────────────────────────────


def plot_results(cell_df: pd.DataFrame, pooled: dict, wide: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # panel A: flip decomposition per comparison (pooled)
    ax = axes[0]
    comps = list(pooled.keys())
    x = np.arange(len(comps))
    reg = [pooled[c]["n_regression"] / pooled[c]["n_pairs"] * 100 for c in comps]
    imp = [pooled[c]["n_improvement"] / pooled[c]["n_pairs"] * 100 for c in comps]
    ax.bar(
        x - 0.18, reg, width=0.36, label="Regressions (ref ✓ → test ✗)", color="#c44e52"
    )
    ax.bar(
        x + 0.18,
        imp,
        width=0.36,
        label="Improvements (ref ✗ → test ✓)",
        color="#55a868",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in comps], fontsize=9)
    ax.set_ylabel("% of paired tasks", fontsize=9)
    ax.set_title("Discordant task outcomes (pooled)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # panel B: cell-level flip rates by domain
    ax = axes[1]
    sub = cell_df[cell_df["comparison"] == "q4_k_m vs bf16"]
    domains = sorted(sub["domain"].unique())
    data = [sub[sub["domain"] == d]["flip_rate"] * 100 for d in domains]
    ax.boxplot(data, tick_labels=[d.upper() for d in domains])
    ax.set_ylabel("Flip rate (% of paired tasks)", fontsize=9)
    ax.set_title(
        "Q4_K_M vs BF16 flip rate by domain\n(one point per model)",
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig_behavioral_consistency.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Figure saved: {out}")
    plt.close(fig)


# ── entry point ────────────────────────────────────────────────────────────


def run_behavioral_consistency():
    print("=" * 72)
    print("  Behavioral Consistency Analysis (flips + interaction turns)")
    print("=" * 72)

    print("\n  Loading task-level records (success + turns) …")
    df = build_dataset()
    print(f"  Task-level rows: {len(df):,}")

    print("\n  [1] Outcome discordance (McNemar)")
    cell_df, pooled = run_discordance(df)
    for comp, stats_ in pooled.items():
        print(
            f"    {comp}: n={stats_['n_pairs']:,}  "
            f"flip_rate={stats_['flip_rate'] * 100:.2f}%  "
            f"regressions={stats_['n_regression']}  "
            f"improvements={stats_['n_improvement']}  "
            f"p_McNemar={stats_['p_mcnemar']:.4f}"
        )
    print("\n    Per-cell flip rates (q4_k_m vs bf16):")
    sub = cell_df[cell_df["comparison"] == "q4_k_m vs bf16"]
    print(
        f"      mean={sub['flip_rate'].mean() * 100:.2f}%  "
        f"min={sub['flip_rate'].min() * 100:.2f}%  "
        f"max={sub['flip_rate'].max() * 100:.2f}%"
    )

    print("\n  [2] Interaction turns")
    turns, wide = run_turns_analysis(df)
    for q, v in turns["pooled_mean_turns"].items():
        print(
            f"    {q:8s}: mean={v['mean']:.2f}  median={v['median']:.0f}  n={v['n_tasks']:,}"
        )
    for comp, v in turns["cell_tests"].items():
        print(
            f"    {comp}: median diff={v['median_diff_turns']:+.2f} turns/cell  "
            f"W={v['wilcoxon_W']}  p={v['p_value']}  r={v['effect_size_r']}"
        )
    print("    Turns by outcome:")
    for q, v in turns["turns_by_outcome"].items():
        print(
            f"      {q:8s}: success={v['success_mean_turns']:.2f}  failure={v['failure_mean_turns']:.2f}"
        )

    # save
    cell_df.to_csv(OUTPUT_DIR / "flip_rates_table.csv", index=False)
    with open(OUTPUT_DIR / "behavioral_consistency.json", "w") as f:
        json.dump(
            {
                "pooled_discordance": pooled,
                "turns": {k: v for k, v in turns.items()},
            },
            f,
            indent=2,
        )
    print(f"\n  Saved: {OUTPUT_DIR / 'flip_rates_table.csv'}")
    print(f"  Saved: {OUTPUT_DIR / 'behavioral_consistency.json'}")

    plot_results(cell_df, pooled, wide)
    return df, cell_df, pooled, turns


if __name__ == "__main__":
    run_behavioral_consistency()
