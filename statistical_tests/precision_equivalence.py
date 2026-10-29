"""
precision_equivalence.py
========================
Task-clustered equivalence, outcome-flip, and run-level analyses used in the
revised paper (supersedes the Wilcoxon-based TOST in tost_equivalence.py).

Why a new analysis
------------------
tost_equivalence.py shifts binary paired differences by the margin and runs
one-sided Wilcoxon signed-rank tests. With differences in {-1, 0, +1} this
tests the *median* difference (almost always 0), not the success-rate
difference, so pairs such as Qwen3-8B/WebShop (-10 pp) were declared
equivalent. It also treats the three replicates of a task as independent.

Procedure
---------
* Per model x size x domain and comparison (Q4/Q8 vs 16-bit), average each
  task's binary outcome over the replicates, take the per-task difference, and
  run TOST with paired t-tests on the task-level differences (margin +-5 pp,
  alpha 0.05) == 90% CI inside (-5, +5). Non-inferiority: lower bound > -5.
  A task-clustered bootstrap CI is reported as a sensitivity check.
* Floor pairs: success <= 1.2% at both precisions.
* Pooled: task-clustered mean difference and a logistic GEE (exchangeable,
  clustered by task) with precision and model-domain pair as predictors.
* Outcome flips between run-matched tasks, compared with the flip rate between
  replicates of the same 16-bit configuration; exact McNemar test.
* Run-level metrics: replicates averaged per pair, Friedman over 24 pairs,
  Wilcoxon signed-rank with Holm correction.

Outputs: statistical_tests/output/precision_equivalence.csv and a printed summary.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tost_equivalence import build_task_level_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MARGIN = 5.0  # pp
FLOOR = 1.25  # pp; both precisions at or below this -> floor pair
TESTS = ["q4_k_m", "q8_0"]
REF = "bf16"


def load_tasks() -> pd.DataFrame:
    df = build_task_level_dataset()
    df["quant"] = df["quant"].replace({"f16": "bf16"})
    return df


def decision(lo, hi, sr_ref, sr_test):
    if sr_ref < FLOOR and sr_test < FLOOR:
        return "floor"
    if lo > -MARGIN and hi < MARGIN:
        return "equivalent"
    if lo > MARGIN:
        return "superior"
    if lo > -MARGIN:
        return "non-inferior"
    if hi < 0:
        return "regression"
    return "inconclusive"


def per_pair_equivalence(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    task_means = df.groupby(
        ["model", "size", "domain", "quant", "task_index"]
    )["success"].mean().unstack("quant")
    records = []
    for (model, size, domain), cell in task_means.groupby(level=[0, 1, 2]):
        for test in TESTS:
            x = cell[[test, REF]].dropna() * 100
            d = (x[test] - x[REF]).values
            n, mean = len(d), d.mean()
            se = d.std(ddof=1) / np.sqrt(n)
            if se > 0:
                tc = stats.t.ppf(0.95, n - 1)
                lo, hi = mean - tc * se, mean + tc * se
                p_lower = 1 - stats.t.cdf((mean + MARGIN) / se, n - 1)
                p_upper = stats.t.cdf((mean - MARGIN) / se, n - 1)
            else:
                lo = hi = mean
                p_lower = p_upper = 0.0 if abs(mean) < MARGIN else 1.0
            boot = [rng.choice(d, n).mean() for _ in range(4000)]
            b_lo, b_hi = np.percentile(boot, [5, 95])
            sr_ref, sr_test = x[REF].mean(), x[test].mean()
            records.append({
                "model": model, "size": size, "domain": domain,
                "comparison": f"{test} vs {REF}", "n_tasks": n,
                "sr_ref": sr_ref, "sr_test": sr_test, "diff_pp": mean,
                "ci90_lo": lo, "ci90_hi": hi, "boot90_lo": b_lo, "boot90_hi": b_hi,
                "p_tost": max(p_lower, p_upper),
                "equivalent": max(p_lower, p_upper) < 0.05,
                "non_inferior": p_lower < 0.05,
                "decision": decision(lo, hi, sr_ref, sr_test),
            })
    return pd.DataFrame(records)


def pooled(df: pd.DataFrame):
    df = df.copy()
    df["cluster"] = (df["model"] + df["size"] + df["domain"]
                     + df["task_index"].astype(str))
    df["pair"] = df["model"] + df["size"] + df["domain"]
    tm = df.groupby(["cluster", "quant"])["success"].mean().unstack()
    for test in TESTS:
        d = (tm[test] - tm[REF]).dropna() * 100
        se = d.std(ddof=1) / np.sqrt(len(d))
        tc = stats.t.ppf(0.95, len(d) - 1)
        print(f"  pooled {test}: {d.mean():+.2f} pp, 90% CI "
              f"[{d.mean() - tc * se:+.2f}, {d.mean() + tc * se:+.2f}]")
    gee = smf.gee(
        f"success ~ C(quant, Treatment('{REF}')) + C(pair)", "cluster", df,
        family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    ci = gee.conf_int()
    for k in gee.params.index:
        if "quant" in k:
            print(f"  GEE {k}: OR={np.exp(gee.params[k]):.3f} "
                  f"[{np.exp(ci.loc[k, 0]):.3f}, {np.exp(ci.loc[k, 1]):.3f}]")


def flips(df: pd.DataFrame):
    w = df.pivot_table(index=["model", "size", "domain", "run", "task_index"],
                       columns="quant", values="success").dropna()
    for test in TESTS:
        reg = int(((w[REF] == 1) & (w[test] == 0)).sum())
        imp = int(((w[REF] == 0) & (w[test] == 1)).sum())
        print(f"  {test}: flips={(reg + imp) / len(w) * 100:.1f}%  "
              f"regressed/solved={reg / (w[REF] == 1).sum() * 100:.1f}%  "
              f"McNemar p={stats.binomtest(reg, reg + imp).pvalue:.3f}")
    r = df[df.quant == REF].pivot_table(
        index=["model", "size", "domain", "task_index"], columns="run",
        values="success")
    reg = solved = disc = n = 0
    for a, b in itertools.permutations(r.columns, 2):
        x = r[[a, b]].dropna()
        reg += ((x[a] == 1) & (x[b] == 0)).sum()
        solved += (x[a] == 1).sum()
        disc += (x[a] != x[b]).sum()
        n += len(x)
    print(f"  16-bit replicate baseline: flips={disc / n * 100:.1f}%  "
          f"regressed/solved={reg / solved * 100:.1f}%")


def run_level():
    with open(ROOT / "results" / "all_runs_master.json", encoding="utf-8") as f:
        raw = json.load(f)
    runs = pd.DataFrame([{**{k: e[k] for k in ["model", "size", "quant", "domain", "run"]},
                          **e["metrics"]} for e in raw])
    runs["quant"] = runs["quant"].replace({"f16": "bf16"})
    for metric in ["success_rate", "mean_reward", "completion_rate", "tle_rate",
                   "error_rate", "energy_per_task"]:
        p = runs.groupby(["model", "size", "domain", "quant"])[metric].mean().unstack()
        fr = stats.friedmanchisquare(p["bf16"], p["q8_0"], p["q4_k_m"])
        contrasts = [("q8_0", "bf16"), ("q4_k_m", "bf16"), ("q4_k_m", "q8_0")]
        raw_p = [stats.wilcoxon(p[a], p[b]).pvalue if (p[a] != p[b]).any() else 1.0
                 for a, b in contrasts]
        adj = multipletests(raw_p, method="holm")[1]
        pw = ", ".join(
            f"{a}-{b}: med {((p[a] / p[b] - 1).replace([np.inf, -np.inf], np.nan).median() * 100):+.1f}% "
            f"p_holm={q:.4f}" for (a, b), q in zip(contrasts, adj))
        print(f"  {metric}: Friedman p={fr.pvalue:.4f} W={fr.statistic / (len(p) * 2):.3f} | {pw}")


def main():
    df = load_tasks()
    print(f"Task-level rows: {len(df):,}")
    summary = per_pair_equivalence(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "precision_equivalence.csv", index=False)
    for comp, g in summary.groupby("comparison"):
        print(f"\n[{comp}] equivalent {g.equivalent.sum()}/{len(g)}, "
              f"non-inferior {g.non_inferior.sum()}/{len(g)}")
        print("  decisions:", g.decision.value_counts().to_dict())
        strong = g[g.sr_ref >= 10]
        print(f"  16-bit success >= 10%: n={len(strong)}, "
              f"equivalent {strong.equivalent.sum()}, non-inferior {strong.non_inferior.sum()}")
    print("\n[Pooled]")
    pooled(df)
    print("\n[Outcome flips]")
    flips(df)
    print("\n[Run-level, replicates averaged, 24 pairs]")
    run_level()


if __name__ == "__main__":
    main()
