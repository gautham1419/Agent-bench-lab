"""
tost_equivalence.py
===================
Task-level paired equivalence analysis (TOST) for RQ1.

Motivation
----------
The paper currently reports "no significant difference" in success rate across
quantization levels based on run-level aggregates (n ≈ 72 runs).  This is
underpowered for making a positive equivalence claim.  Each run, however,
contains hundreds of individual task outcomes.  Pairing identical tasks (matched
by task index) across precision levels within the same model × size × domain ×
replicate cell gives thousands of matched binary observations that support a
much stronger statement: Q4_K_M is statistically *equivalent* to BF16, not
merely "not significantly different".

Pairing strategy
----------------
Within every (model, size, domain, replicate) group, three precision variants
(BF16/F16, Q8_0, Q4_K_M) are available.  Tasks within each variant are
identified by the `index` field in runs.jsonl (0-indexed integer assigned by
AgentBench; identical across runs because the task set is fixed and tasks are
drawn from the same data files in the same order).  A pair is formed only when
the same index appears in both precision variants being compared; unmatched
indices (due to errors/timeouts) are excluded.

TOST procedure
--------------
For each model × domain cell we aggregate across replicates: for every task
index that appears in all three precision variants we compute three binary
success vectors (BF16, Q8_0, Q4_K_M).  We then perform TOST on the paired
*differences* in binary outcomes (0/1).  The equivalence margin Δ = 0.05
(5 percentage points) is used, matching the engineering threshold assumed in
the paper (a configuration is acceptable if its success rate is within ±5 pp
of the full-precision baseline).

Because the outcomes are binary, the paired differences are bounded {-1, 0, +1};
we use a sign test / Wilcoxon signed-rank test as the within-pair statistic
rather than a t-test, and we implement TOST as two one-sided Wilcoxon tests:
    H₁_lower:  median(diff) > -Δ
    H₁_upper:  median(diff) < +Δ
Equivalence is concluded when both one-sided p-values are below α = 0.05.

Outputs
-------
- statistical_tests/output/tost_results.json      (full numerical results)
- statistical_tests/output/tost_summary_table.csv (publication table)
- statistical_tests/output/fig_tost_equivalence.pdf
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "new_outputs"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EQUIV_MARGIN = 0.05  # ±5 pp
ALPHA = 0.05

# ── helpers ────────────────────────────────────────────────────────────────


def load_task_outcomes(run_folder: Path) -> dict[int, int] | None:
    """
    Parse runs.jsonl for a single run folder.
    Returns {task_index: 1/0} or None if the file is missing.
    """
    # find the agent sub-dir
    agent_dirs = [p for p in run_folder.iterdir() if p.is_dir()]
    if not agent_dirs:
        return None
    agent_dir = agent_dirs[0]

    domain_dirs = [p for p in agent_dir.iterdir() if p.is_dir()]
    if not domain_dirs:
        return None
    domain_dir = domain_dirs[0]

    runs_file = domain_dir / "runs.jsonl"
    if not runs_file.exists():
        return None

    outcomes: dict[int, int] = {}
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
            outcomes[idx] = 1 if (score and score > 0) else 0
    return outcomes


def parse_run_metadata(folder_name: str) -> dict | None:
    """
    Extract model, size, quant, run from the run folder name.
    Format: YYYY-MM-DD-HH-MM-SS_[prefix-]model-size-quant-runN
    """
    # strip date prefix (first two dash-separated tokens are date+time)
    parts = folder_name.split("_", 1)
    if len(parts) < 2:
        return None
    slug = parts[1].lstrip("-").strip()

    # detect run suffix
    import re

    m = re.search(r"-(run\d+)$", slug)
    if not m:
        return None
    run_id = m.group(1)
    slug_no_run = slug[: m.start()]

    # determine quant
    quant = None
    for q in ["q4_k_ms", "q4_k_m", "q8_0", "bf16", "f16"]:
        if slug_no_run.endswith(q):
            quant = q
            slug_no_run = slug_no_run[: -(len(q) + 1)]
            break
    if quant is None:
        return None
    if quant == "f16":
        quant = "bf16"
    if quant == "q4_k_ms":
        quant = "q4_k_m"

    # slug_no_run is now: [prefix-]model-size
    # parse model & size
    toks = slug_no_run.split("-")
    # size is last token matching \d+[bB]
    size = None
    for i in reversed(range(len(toks))):
        if re.match(r"^\d+(\.\d+)?[bB]$", toks[i], re.IGNORECASE):
            size = toks[i].upper()
            model_toks = toks[:i]
            break
    if size is None:
        return None

    # skip leading "ollama" or "deepseek-r1-qwen"
    model_str = "-".join(model_toks).lstrip("-").lower()
    if model_str.startswith("ollama-"):
        model_str = model_str[len("ollama-") :]
    # normalise known model names
    if model_str.startswith("deepseek-r1-qwen"):
        model = "deepseek-r1-qwen"
    elif "ministral3" in model_str:
        model = "ministral3"
    elif "qwen3" in model_str:
        model = "qwen3"
    else:
        model = model_str

    return {"model": model, "size": size, "quant": quant, "run": run_id}


def get_domain(run_folder: Path) -> str | None:
    agent_dirs = [p for p in run_folder.iterdir() if p.is_dir()]
    if not agent_dirs:
        return None
    domain_dirs = [p for p in agent_dirs[0].iterdir() if p.is_dir()]
    if not domain_dirs:
        return None
    return domain_dirs[0].name.replace("-std", "")


# ── build task-level dataset ───────────────────────────────────────────────


def build_task_level_dataset() -> pd.DataFrame:
    """
    Iterate all run folders; return a DataFrame with columns:
    model, size, quant, domain, run, task_index, success
    """
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
        outcomes = load_task_outcomes(folder)
        if not outcomes:
            continue
        for idx, success in outcomes.items():
            rows.append(
                {
                    "model": meta["model"],
                    "size": meta["size"],
                    "quant": meta["quant"],
                    "domain": domain,
                    "run": meta["run"],
                    "task_index": idx,
                    "success": success,
                }
            )
    df = pd.DataFrame(rows)
    # deduplicate: if multiple runs with same label, keep last
    df = df.drop_duplicates(
        subset=["model", "size", "quant", "domain", "run", "task_index"]
    )
    return df


# ── TOST via two one-sided Wilcoxon signed-rank tests ─────────────────────


def tost_wilcoxon(diff: np.ndarray, margin: float = EQUIV_MARGIN) -> dict:
    """
    TOST using two one-sided Wilcoxon signed-rank tests.
    diff = paired_differences = x_test - x_reference   (e.g. Q4 - BF16)

    H1_lower: median(diff) > -margin   (lower one-sided)
    H1_upper: median(diff) <  margin   (upper one-sided)

    Equivalence is concluded when both p-values < alpha.

    Returns a dict with effect estimate, CI, and TOST p-values.
    """
    n = len(diff)
    if n < 10:
        return {"n": n, "skipped": True, "reason": "too few pairs"}

    # Hodges-Lehmann estimator (robust location estimate of the median difference)
    hl_estimate = (
        np.median([(diff[i] + diff[j]) / 2 for i in range(n) for j in range(i, n)])
        if n <= 200
        else np.median(diff)
    )

    # 90% CI via Wilcoxon (corresponds to two α=0.05 one-sided tests)
    try:
        ci_res = stats.wilcoxon(diff, alternative="two-sided")
        # scipy does not expose CI directly; use bootstrap
        rng = np.random.default_rng(42)
        boot_medians = [
            np.median(rng.choice(diff, size=n, replace=True)) for _ in range(5000)
        ]
        ci_lo, ci_hi = np.percentile(boot_medians, [5, 95])
    except Exception:
        ci_lo = ci_hi = np.nan

    # Lower TOST: test  diff - (-margin) > 0  i.e.  (diff + margin) > 0
    shifted_lower = diff + margin
    try:
        _, p_lower = stats.wilcoxon(shifted_lower, alternative="greater")
    except Exception:
        p_lower = 1.0

    # Upper TOST: test  margin - diff > 0  i.e.  (margin - diff) > 0
    shifted_upper = margin - diff
    try:
        _, p_upper = stats.wilcoxon(shifted_upper, alternative="greater")
    except Exception:
        p_upper = 1.0

    p_tost = max(p_lower, p_upper)
    equivalent = bool(p_tost < ALPHA)
    mean_diff = float(np.mean(diff))

    return {
        "n": n,
        "mean_diff": mean_diff,
        "hl_estimate": float(hl_estimate),
        "ci_90_lo": float(ci_lo),
        "ci_90_hi": float(ci_hi),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_tost": float(p_tost),
        "margin": margin,
        "equivalent": equivalent,
        "skipped": False,
    }


# ── main analysis ──────────────────────────────────────────────────────────


def run_tost_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, size, domain) cell, pool all replicates and compute
    TOST for Q4_K_M vs BF16 and Q8_0 vs BF16.
    Returns a summary DataFrame.
    """
    # Normalise quant labels (f16 → bf16; guard against bf16→bbf16)
    df = df.copy()
    df["quant"] = df["quant"].str.lower().replace({"f16": "bf16"})

    comparisons = [("q4_k_m", "bf16"), ("q8_0", "bf16")]
    records = []

    for (model, size, domain), cell in df.groupby(["model", "size", "domain"]):
        bf16_tasks = cell[cell["quant"] == "bf16"]
        q8_tasks = cell[cell["quant"] == "q8_0"]
        q4_tasks = cell[cell["quant"] == "q4_k_m"]

        for test_label, ref_label in comparisons:
            test_tasks = cell[cell["quant"] == test_label]
            ref_tasks = cell[cell["quant"] == ref_label]

            if test_tasks.empty or ref_tasks.empty:
                continue

            # merge on task_index (pooling all replicates)
            merged = pd.merge(
                test_tasks[["run", "task_index", "success"]].rename(
                    columns={"success": "s_test"}
                ),
                ref_tasks[["run", "task_index", "success"]].rename(
                    columns={"success": "s_ref"}
                ),
                on=["run", "task_index"],
                how="inner",
            )
            if merged.empty:
                continue

            diff = (merged["s_test"] - merged["s_ref"]).values.astype(float)
            result = tost_wilcoxon(diff)

            # observed success rates
            sr_test = test_tasks["success"].mean()
            sr_ref = ref_tasks["success"].mean()

            records.append(
                {
                    "model": model,
                    "size": size,
                    "domain": domain,
                    "comparison": f"{test_label} vs {ref_label}",
                    "test_quant": test_label,
                    "ref_quant": ref_label,
                    "sr_test": round(sr_test, 4),
                    "sr_ref": round(sr_ref, 4),
                    "sr_diff_pp": round((sr_test - sr_ref) * 100, 2),
                    **result,
                }
            )

    return pd.DataFrame(records)


# ── publication figure ─────────────────────────────────────────────────────


def plot_tost_results(summary: pd.DataFrame):
    """
    Forest-plot-style figure: one panel per comparison (Q4 vs BF16, Q8 vs BF16).
    Each row is a model×domain cell.  Equivalence margin shown as grey band.
    """
    comps = [("q4_k_m", "bf16"), ("q8_0", "bf16")]
    comp_labels = {"q4_k_m vs bf16": "Q4_K_M vs BF16", "q8_0 vs bf16": "Q8_0 vs BF16"}
    colors = {"q4_k_m vs bf16": "#e07b54", "q8_0 vs bf16": "#5b8db8"}

    fig, axes = plt.subplots(
        1, 2, figsize=(12, max(5, len(summary) // 2 * 0.55 + 2)), sharey=False
    )

    for ax, (tq, rq) in zip(axes, comps):
        comp_key = f"{tq} vs {rq}"
        sub = summary[summary["comparison"] == comp_key].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        sub = sub[~sub.get("skipped", pd.Series(False, index=sub.index))].copy()
        sub["label"] = (
            sub["model"] + "-" + sub["size"] + "\n" + sub["domain"].str.upper()
        )
        sub = sub.sort_values(["model", "size", "domain"])
        sub = sub.reset_index(drop=True)

        y = np.arange(len(sub))
        color = colors[comp_key]

        # grey equivalence band
        ax.axvspan(
            -EQUIV_MARGIN * 100,
            EQUIV_MARGIN * 100,
            color="#e8e8e8",
            zorder=0,
            label=f"±{int(EQUIV_MARGIN * 100)} pp margin",
        )
        ax.axvline(0, color="#999999", lw=0.8, zorder=1)

        for i, row in sub.iterrows():
            y_pos = sub.index.get_loc(i)
            lo = (
                row["ci_90_lo"] * 100
                if not pd.isna(row.get("ci_90_lo", np.nan))
                else row["mean_diff"] * 100
            )
            hi = (
                row["ci_90_hi"] * 100
                if not pd.isna(row.get("ci_90_hi", np.nan))
                else row["mean_diff"] * 100
            )
            est = (
                row["hl_estimate"] * 100
                if not pd.isna(row.get("hl_estimate", np.nan))
                else row["mean_diff"] * 100
            )

            eq = row.get("equivalent", False)
            marker = "D" if eq else "o"
            mcolor = color if eq else "#cccccc"

            ax.plot([lo, hi], [y_pos, y_pos], color=color, lw=1.5, zorder=2, alpha=0.7)
            ax.plot(
                est,
                y_pos,
                marker=marker,
                color=mcolor,
                markersize=7,
                markeredgecolor=color,
                zorder=3,
            )

        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["label"].tolist(), fontsize=7)
        ax.set_xlabel("Δ Success Rate (pp)", fontsize=9)
        ax.set_title(comp_labels[comp_key], fontsize=10, fontweight="bold")
        ax.set_xlim(-15, 15)
        ax.invert_yaxis()

        # legend
        eq_patch = mpatches.Patch(color=color, label="Equivalent (TOST p<0.05)")
        neq_patch = mpatches.Patch(color="#cccccc", label="Not equivalent")
        band_patch = mpatches.Patch(color="#e8e8e8", label="±5 pp margin")
        ax.legend(
            handles=[eq_patch, neq_patch, band_patch], fontsize=7, loc="lower right"
        )

    fig.suptitle(
        "Task-Level TOST Equivalence Analysis\n"
        "90% CI on Δ Success Rate (paired by task index)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    out = OUTPUT_DIR / "fig_tost_equivalence.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Figure saved: {out}")
    plt.close(fig)


# ── entry point ────────────────────────────────────────────────────────────


def run_tost():
    print("=" * 72)
    print("  TOST Equivalence Analysis  (task-level paired, Wilcoxon TOST)")
    print("=" * 72)

    print("\n  Loading per-task outcomes from raw logs …")
    df = build_task_level_dataset()
    print(f"  Task-level rows loaded: {len(df):,}")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Quants: {sorted(df['quant'].unique())}")
    print(f"  Domains: {sorted(df['domain'].unique())}")

    print("\n  Running TOST …")
    summary = run_tost_analysis(df)

    # ── print results ──────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print(f"  Equivalence margin: ±{EQUIV_MARGIN * 100:.0f} pp,  α = {ALPHA}")
    print("─" * 72)

    for comp, grp in summary.groupby("comparison"):
        print(f"\n  [{comp.upper()}]")
        for _, row in grp.iterrows():
            if row.get("skipped"):
                print(
                    f"    {row['model']}-{row['size']} / {row['domain']:8s}  SKIPPED ({row.get('reason', '')})"
                )
                continue
            eq_str = "EQUIVALENT ✓" if row["equivalent"] else "not equivalent"
            print(
                f"    {row['model']}-{row['size']} / {row['domain']:8s}"
                f"  n={row['n']:5d}"
                f"  SR_test={row['sr_test']:.3f}  SR_ref={row['sr_ref']:.3f}"
                f"  Δ={row['sr_diff_pp']:+5.2f} pp"
                f"  90%CI=[{row['ci_90_lo'] * 100:+5.2f},{row['ci_90_hi'] * 100:+5.2f}]"
                f"  p_TOST={row['p_tost']:.4f}  →  {eq_str}"
            )

    # ── aggregate summary ──────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  Aggregate (across all model-domain cells):")
    for comp, grp in summary.groupby("comparison"):
        valid = grp[~grp.get("skipped", pd.Series(False, index=grp.index))]
        n_eq = valid["equivalent"].sum()
        n_tot = len(valid)
        pool_diff = valid["sr_diff_pp"].mean()
        print(
            f"    {comp}: {n_eq}/{n_tot} cells equivalent  |  mean Δ = {pool_diff:+.2f} pp"
        )

    # ── overall pooled TOST ────────────────────────────────────────────────
    print("\n  [Pooled across all cells]")
    df_norm = df.copy()
    df_norm["quant"] = df_norm["quant"].str.lower().str.replace("f16", "bf16")
    for tq, rq in [("q4_k_m", "bf16"), ("q8_0", "bf16")]:
        test_all = df_norm[df_norm["quant"] == tq]
        ref_all = df_norm[df_norm["quant"] == rq]
        merged = pd.merge(
            test_all[
                ["model", "size", "domain", "run", "task_index", "success"]
            ].rename(columns={"success": "s_test"}),
            ref_all[["model", "size", "domain", "run", "task_index", "success"]].rename(
                columns={"success": "s_ref"}
            ),
            on=["model", "size", "domain", "run", "task_index"],
            how="inner",
        )
        if merged.empty:
            continue
        diff = (merged["s_test"] - merged["s_ref"]).values.astype(float)
        # use sign test for large n (faster, still valid)
        pos = np.sum(diff > 0)
        neg = np.sum(diff < 0)
        ties = np.sum(diff == 0)
        non_zero = diff[diff != 0]
        if len(non_zero) > 0:
            res = tost_wilcoxon(non_zero)
        else:
            res = {
                "n": len(diff),
                "mean_diff": 0.0,
                "p_tost": 0.0,
                "equivalent": True,
                "hl_estimate": 0.0,
                "ci_90_lo": 0.0,
                "ci_90_hi": 0.0,
            }
        eq_str = "EQUIVALENT ✓" if res.get("equivalent") else "not equivalent"
        print(
            f"    {tq} vs {rq}:  n={len(diff):,}  pos={pos}  neg={neg}  ties={ties}"
            f"  p_TOST={res.get('p_tost', 'n/a'):.4f}  →  {eq_str}"
        )

    # ── save outputs ───────────────────────────────────────────────────────
    summary.to_csv(OUTPUT_DIR / "tost_summary_table.csv", index=False)
    print(f"\n  Table saved: {OUTPUT_DIR / 'tost_summary_table.csv'}")

    # JSON – convert numpy types
    def _safe(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v) if not np.isnan(v) else None
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    records = [
        {k: _safe(v) for k, v in row.items()} for row in summary.to_dict("records")
    ]
    with open(OUTPUT_DIR / "tost_results.json", "w") as f:
        json.dump(
            {"margin": EQUIV_MARGIN, "alpha": ALPHA, "results": records}, f, indent=2
        )
    print(f"  JSON saved:  {OUTPUT_DIR / 'tost_results.json'}")

    print("\n  Generating figure …")
    plot_tost_results(summary)

    return summary


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run_tost()
