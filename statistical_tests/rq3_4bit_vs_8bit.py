"""
rq3_4bit_vs_8bit.py
===================
Direct 4-bit versus 8-bit efficiency comparison for RQ3.

All other RQ3 analyses use 16-bit as the shared reference baseline.
This script closes the remaining step by directly comparing the two
quantized levels, answering the practitioner question:
  "Why 4-bit rather than 8-bit, given that both achieve capability
   equivalence with 16-bit?"

Metrics compared (paired across 24 model-domain combinations):
  - Energy per task (kJ)          [lower is better]
  - Peak GPU memory (MiB)         [lower is better]
  - Mean task wall-clock (s)      [lower is better]
  - Effective throughput (t/min)  [higher is better]
  - Task success rate (%)         [should be unchanged]

Outlier note
------------
The Ministral3-8B / OS / 8-bit configuration has an anomalous mean
task latency (3,584.6 s) caused by a Docker-container hang in replicate
run 2 (wall-clock 1,444,141 s vs 2,482 s and 5,314 s for runs 1 and 3).
Energy, VRAM, and success-rate figures for that cell are unaffected and
are included in those paired comparisons.  Latency and throughput pairs
for Ministral3-8B / OS are excluded from the latency and throughput
Wilcoxon tests (n = 23 instead of 24); this is noted in the output.
See also Table I footnote (†) in the paper.

Outputs
-------
  statistical_tests/output/rq3_4bit_vs_8bit_results.json

Usage
-----
  cd statistical_tests
  python rq3_4bit_vs_8bit.py

  # or via the master runner
  python run_all_tests.py rq3_48
"""

import numpy as np
import pandas as pd
from data_loader import DATA_DIR, ensure_output_dir, save_results
from scipy import stats

SEPARATOR = "=" * 72

# ── Outlier configuration excluded from latency / throughput comparisons ──
LATENCY_OUTLIER = ("ministral3-8B", "OS")  # model_key, Domain


# ── helpers ───────────────────────────────────────────────────────────────


def _load_deployment_table() -> pd.DataFrame:
    """Load deployment_table.csv and return per-domain rows only."""
    path = DATA_DIR / "deployment_table.csv"
    df = pd.read_csv(path)
    domains = {"ALFWORLD", "DBBENCH", "OS", "WEBSHOP"}
    df = df[df["Domain"].isin(domains)].copy()

    def _quant(cfg: str) -> str | None:
        if "Q4_K_M" in cfg:
            return "4bit"
        if "Q8_0" in cfg:
            return "8bit"
        if "BF16" in cfg or "F16" in cfg:
            return "16bit"
        return None

    def _model_key(cfg: str) -> str:
        for suf in ("-Q4_K_M", "-Q8_0", "-BF16", "-F16"):
            if cfg.endswith(suf):
                return cfg[: -len(suf)]
        return cfg

    df["quant"] = df["Configuration"].apply(_quant)
    df["model_key"] = df["Configuration"].apply(_model_key)
    return df


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per (model_key, Domain) with 4bit/8bit/16bit columns."""
    piv = df.pivot_table(
        index=["model_key", "Domain"],
        columns="quant",
        values=[
            "Energy/Task (kJ)",
            "Wall-Clock/Task (s)",
            "Throughput (tasks/min)",
            "Peak GPU Mem (MiB)",
            "Success (%)",
        ],
        aggfunc="first",
    ).reset_index()
    piv.columns = ["_".join(c).strip("_") for c in piv.columns]
    return piv.sort_values(["model_key", "Domain"]).reset_index(drop=True)


def _wilcoxon_lower_better(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    """
    Paired Wilcoxon signed-rank test: is a (4-bit) < b (8-bit)?
    Returns a results dict and prints a formatted summary.
    """
    valid = [(ai, bi) for ai, bi in zip(a, b) if not np.isnan(ai) and not np.isnan(bi)]
    arr_a = np.array([x[0] for x in valid])
    arr_b = np.array([x[1] for x in valid])
    n = len(arr_a)

    pct_savings = (arr_b - arr_a) / arr_b * 100
    stat, p = stats.wilcoxon(arr_a, arr_b, alternative="less")
    # Rank-biserial correlation: r = 1 - 2W / (n(n+1)/2)
    r_rb = 1 - (2 * stat) / (n * (n + 1) / 2)

    print(f"\n  {label}  (n={n})")
    print(f"    Median savings (8-bit → 4-bit): {np.median(pct_savings):.1f}%")
    print(f"    Wilcoxon W={stat:.0f},  p={p:.4f},  r_rb={r_rb:.3f}")
    print(f"    Pairs where 4-bit is cheaper:   {(pct_savings > 0).sum()}/{n}")

    return {
        "n": n,
        "median_pct_savings": float(np.median(pct_savings)),
        "wilcoxon_W": float(stat),
        "p_value": float(p),
        "rank_biserial_r": float(r_rb),
        "n_favoring_4bit": int((pct_savings > 0).sum()),
        "all_pct_savings": [round(float(x), 2) for x in sorted(pct_savings)],
    }


def _wilcoxon_higher_better(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    """
    Paired Wilcoxon: is a (4-bit) > b (8-bit)?
    """
    valid = [(ai, bi) for ai, bi in zip(a, b) if not np.isnan(ai) and not np.isnan(bi)]
    arr_a = np.array([x[0] for x in valid])
    arr_b = np.array([x[1] for x in valid])
    n = len(arr_a)

    pct_gain = (arr_a - arr_b) / arr_b * 100
    stat, p = stats.wilcoxon(arr_a, arr_b, alternative="greater")
    # For 'greater': large W supports H1 (a > b), so r_rb = 2W/max - 1
    r_rb = (2 * stat) / (n * (n + 1) / 2) - 1

    print(f"\n  {label}  (n={n})")
    print(f"    Median improvement (8-bit → 4-bit): {np.median(pct_gain):.1f}%")
    print(f"    Wilcoxon W={stat:.0f},  p={p:.4f},  r_rb={r_rb:.3f}")
    print(f"    Pairs where 4-bit is faster/higher: {(pct_gain > 0).sum()}/{n}")

    return {
        "n": n,
        "median_pct_gain": float(np.median(pct_gain)),
        "wilcoxon_W": float(stat),
        "p_value": float(p),
        "rank_biserial_r": float(r_rb),
        "n_favoring_4bit": int((pct_gain > 0).sum()),
        "all_pct_gains": [round(float(x), 2) for x in sorted(pct_gain)],
    }


# ── main analysis ─────────────────────────────────────────────────────────


def run_rq3_4bit_vs_8bit() -> dict:
    print(f"\n{SEPARATOR}")
    print("  RQ3 SUPPLEMENT: Direct 4-bit versus 8-bit Efficiency Comparison")
    print(SEPARATOR)
    print(
        "\n  24 model-domain pairs (6 models × 4 domains).\n"
        f"  Latency / throughput: n=23 ('{LATENCY_OUTLIER[0]} / {LATENCY_OUTLIER[1]}'"
        " 8-bit excluded — Docker-container hang; see deployment_table.csv footnote)."
    )

    df = _load_deployment_table()
    piv = _pivot(df)

    results = {}

    # ── Energy ──────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Energy per task (kJ)  [lower is better]")
    print(SEPARATOR)
    results["energy"] = _wilcoxon_lower_better(
        piv["Energy/Task (kJ)_4bit"].values,
        piv["Energy/Task (kJ)_8bit"].values,
        "Energy/task (kJ)",
    )

    # ── VRAM ────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Peak GPU memory (MiB)  [lower is better]")
    print(SEPARATOR)
    results["vram"] = _wilcoxon_lower_better(
        piv["Peak GPU Mem (MiB)_4bit"].values,
        piv["Peak GPU Mem (MiB)_8bit"].values,
        "Peak GPU memory (MiB)",
    )

    # ── Latency (outlier excluded) ───────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Latency (s)  [lower is better; n=23, outlier excluded]")
    print(SEPARATOR)
    mask_lat = ~(
        (piv["model_key"] == LATENCY_OUTLIER[0]) & (piv["Domain"] == LATENCY_OUTLIER[1])
    )
    results["latency"] = _wilcoxon_lower_better(
        piv.loc[mask_lat, "Wall-Clock/Task (s)_4bit"].values,
        piv.loc[mask_lat, "Wall-Clock/Task (s)_8bit"].values,
        "Mean task wall-clock (s)",
    )
    results["latency"]["outlier_excluded"] = (
        f"{LATENCY_OUTLIER[0]} / {LATENCY_OUTLIER[1]} (Docker-container hang)"
    )

    # ── Throughput (outlier excluded) ────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Throughput (t/min)  [higher is better; n=23, outlier excluded]")
    print(SEPARATOR)
    results["throughput"] = _wilcoxon_higher_better(
        piv.loc[mask_lat, "Throughput (tasks/min)_4bit"].values,
        piv.loc[mask_lat, "Throughput (tasks/min)_8bit"].values,
        "Effective throughput (t/min)",
    )
    results["throughput"]["outlier_excluded"] = results["latency"]["outlier_excluded"]

    # ── Success rate (two-sided) ─────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Task success rate (%)  [should be unchanged]")
    print(SEPARATOR)
    s4 = piv["Success (%)_4bit"].values
    s8 = piv["Success (%)_8bit"].values
    valid_s = [(a, b) for a, b in zip(s4, s8) if not np.isnan(a) and not np.isnan(b)]
    arr_s4 = np.array([x[0] for x in valid_s])
    arr_s8 = np.array([x[1] for x in valid_s])
    diffs_s = arr_s4 - arr_s8
    stat_s, p_s = stats.wilcoxon(arr_s4, arr_s8)
    n_s = len(arr_s4)
    print(f"\n  Success rate  (n={n_s})")
    print(f"    Median difference (4-bit − 8-bit): {np.median(diffs_s):.2f} pp")
    print(f"    Wilcoxon (two-sided) W={stat_s:.0f},  p={p_s:.4f}")
    results["success_rate"] = {
        "n": n_s,
        "median_diff_pp": float(np.median(diffs_s)),
        "wilcoxon_W": float(stat_s),
        "p_value": float(p_s),
    }

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  SUMMARY")
    print(SEPARATOR)
    print(
        f"  Energy:     {results['energy']['median_pct_savings']:.1f}% median saving,"
        f"  p={results['energy']['p_value']:.4f},  r={results['energy']['rank_biserial_r']:.3f},"
        f"  {results['energy']['n_favoring_4bit']}/24 pairs favour 4-bit"
    )
    print(
        f"  VRAM:       {results['vram']['median_pct_savings']:.1f}% median saving,"
        f"  p={results['vram']['p_value']:.4f},  r={results['vram']['rank_biserial_r']:.3f},"
        f"  {results['vram']['n_favoring_4bit']}/24 pairs favour 4-bit"
    )
    print(
        f"  Latency:    {results['latency']['median_pct_savings']:.1f}% median saving,"
        f"  p={results['latency']['p_value']:.4f},  r={results['latency']['rank_biserial_r']:.3f},"
        f"  {results['latency']['n_favoring_4bit']}/23 pairs favour 4-bit"
    )
    print(
        f"  Throughput: {results['throughput']['median_pct_gain']:.1f}% median gain,"
        f"  p={results['throughput']['p_value']:.4f},"
        f"  {results['throughput']['n_favoring_4bit']}/23 pairs favour 4-bit"
    )
    print(
        f"  Success:    {results['success_rate']['median_diff_pp']:.2f} pp median diff,"
        f"  p={results['success_rate']['p_value']:.4f}  (no significant change)"
    )

    save_results({"rq3_4bit_vs_8bit": results}, "rq3_4bit_vs_8bit_results.json")
    print("\n  Results saved to output/rq3_4bit_vs_8bit_results.json")

    return results


if __name__ == "__main__":
    ensure_output_dir()
    run_rq3_4bit_vs_8bit()
