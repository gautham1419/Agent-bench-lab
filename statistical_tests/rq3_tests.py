"""
rq3_tests.py
============
RQ3: What is the empirical trade-off between agentic task effectiveness
     and computational efficiency across varying quantization levels?

Tests implemented:
  11. Pareto Efficiency Analysis
  12. Two-Way MANOVA (Quant × Size on {success_rate, energy_per_task})
  13. Spearman Correlation + Mixed-Effects Regression
  14. Efficiency Ratio Analysis with Non-parametric Comparison
  15. Relative Change Analysis (Degradation Metrics)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations

try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

from data_loader import load_run_data, get_matched_data, save_results

SEPARATOR = "=" * 72


# ─────────────────────────────────────────────────────────────────────────
# Test 11: Pareto Efficiency Analysis
# ─────────────────────────────────────────────────────────────────────────
def test_pareto(df):
    """
    Identify Pareto-optimal configurations on the
    success_rate (maximize) vs energy_per_task (minimize) frontier.
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 11: Pareto Efficiency Analysis")
    print(SEPARATOR)

    results = {}

    # Average across runs for each configuration
    agg = df.groupby(["model", "size", "quant", "domain"]).agg(
        success_rate=("success_rate", "mean"),
        energy_per_task=("energy_per_task", "mean"),
    ).reset_index()
    agg["config"] = agg["model"] + "-" + agg["size"] + "-" + agg["quant"]

    # Also compute overall (across domains)
    overall = df.groupby(["model", "size", "quant"]).agg(
        success_rate=("success_rate", "mean"),
        energy_per_task=("energy_per_task", "mean"),
    ).reset_index()
    overall["config"] = overall["model"] + "-" + overall["size"] + "-" + overall["quant"]

    def find_pareto(data):
        """Find Pareto-optimal points (max success, min energy)."""
        is_pareto = np.ones(len(data), dtype=bool)
        sr = data["success_rate"].values
        ep = data["energy_per_task"].values
        for i in range(len(data)):
            for j in range(len(data)):
                if i != j:
                    # j dominates i if j has >= success AND <= energy (with at least one strict)
                    if sr[j] >= sr[i] and ep[j] <= ep[i] and (sr[j] > sr[i] or ep[j] < ep[i]):
                        is_pareto[i] = False
                        break
        return is_pareto

    # Overall Pareto frontier
    print("\n  --- Overall Pareto Frontier (across all domains) ---")
    overall_pareto = find_pareto(overall)
    pareto_configs = overall[overall_pareto].sort_values("success_rate", ascending=False)
    print(f"  {pareto_configs[['config', 'success_rate', 'energy_per_task']].to_string(index=False)}")
    results["overall_pareto"] = pareto_configs[["config", "success_rate", "energy_per_task"]].to_dict("records")

    # Pareto by domain
    print("\n  --- Pareto Frontier by Domain ---")
    domain_pareto = {}
    for domain in sorted(agg["domain"].unique()):
        dom_data = agg[agg["domain"] == domain].copy()
        is_pareto = find_pareto(dom_data)
        pareto_dom = dom_data[is_pareto].sort_values("success_rate", ascending=False)
        print(f"\n  {domain}:")
        print(f"  {pareto_dom[['config', 'success_rate', 'energy_per_task']].to_string(index=False)}")
        domain_pareto[domain] = pareto_dom[["config", "success_rate", "energy_per_task"]].to_dict("records")
    results["domain_pareto"] = domain_pareto

    # Hypervolume indicator (reference point: 0 success, max energy)
    ref_success = 0.0
    ref_energy = overall["energy_per_task"].max() * 1.1
    pareto_points = overall[overall_pareto].sort_values("energy_per_task")
    hv = 0.0
    prev_energy = ref_energy
    for _, row in pareto_points.iterrows():
        hv += (row["success_rate"] - ref_success) * (prev_energy - row["energy_per_task"])
        prev_energy = row["energy_per_task"]
    # Normalize by reference area
    max_hv = (1.0 - ref_success) * ref_energy
    hv_normalized = hv / max_hv if max_hv > 0 else 0
    print(f"\n  Hypervolume indicator (normalized): {hv_normalized:.4f}")
    results["hypervolume_normalized"] = hv_normalized

    # Count how many Pareto-optimal configs per quant level
    print("\n  Pareto-optimal configurations per quantization level:")
    for q in ["bf16", "q8_0", "q4_k_m"]:
        n_pareto = len(pareto_configs[pareto_configs["config"].str.contains(q)])
        n_total = len(overall[overall["config"].str.contains(q)])
        print(f"    {q:<8}: {n_pareto}/{n_total} configs are Pareto-optimal")

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 12: Two-Way MANOVA
# ─────────────────────────────────────────────────────────────────────────
def test_manova(df):
    """
    MANOVA: joint test on {success_rate, energy_per_task} ~ quant × size.
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 12: Two-Way MANOVA (Quant × Size on Success + Energy)")
    print(SEPARATOR)

    results = {}

    # Prepare data
    df_manova = df.dropna(subset=["success_rate", "energy_per_task"]).copy()

    # Log-transform energy for better distributional properties
    df_manova["log_energy"] = np.log1p(df_manova["energy_per_task"])

    # Assumption check: Box's M test for homogeneity of covariance matrices
    # (simplified check using Levene's on each DV)
    print("\n  [Assumption Check] Levene's test per DV:")
    for dv in ["success_rate", "log_energy"]:
        groups = [grp[dv].dropna().values for _, grp in df_manova.groupby("quant") if len(grp) >= 2]
        lev_stat, lev_p = stats.levene(*groups)
        print(f"    {dv}: F={lev_stat:.4f}, p={lev_p:.4f}")

    # MANOVA
    formula = "success_rate + log_energy ~ C(quant)"
    print(f"\n  Formula: {formula}")

    try:
        from statsmodels.multivariate.manova import MANOVA
        manova = MANOVA.from_formula(formula, data=df_manova)
        mv_result = manova.mv_test()
        print(f"\n  MANOVA Results (effect of Quantization):")
        print(mv_result.summary())

        # Extract test statistics
        quant_results = mv_result.results.get("C(quant)", None)
        if quant_results:
            stat_table = quant_results["stat"]
            for test_name in ["Pillai's trace", "Wilks' lambda", "Hotelling-Lawley trace", "Roy's greatest root"]:
                if test_name in stat_table.index:
                    val = stat_table.loc[test_name, "Value"]
                    f_val = stat_table.loc[test_name, "F Value"]
                    p_val = stat_table.loc[test_name, "Pr > F"]
                    print(f"    {test_name}: value={val:.4f}, F={f_val:.4f}, p={p_val:.6f}")
                    results[test_name.replace("'", "").replace(" ", "_")] = {
                        "value": val, "F": f_val, "p": p_val
                    }

    except Exception as e:
        print(f"  [ERROR] MANOVA failed: {e}")
        results["error"] = str(e)

    # Also test with size as a factor
    formula2 = "success_rate + log_energy ~ C(quant) + C(size)"
    print(f"\n  Extended formula: {formula2}")
    try:
        manova2 = MANOVA.from_formula(formula2, data=df_manova)
        mv_result2 = manova2.mv_test()
        print(mv_result2.summary())
    except Exception as e:
        print(f"  [ERROR] Extended MANOVA failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 13: Spearman Correlation + Regression
# ─────────────────────────────────────────────────────────────────────────
def test_energy_performance_regression(df):
    """
    Spearman correlation between success_rate and energy_per_task,
    plus mixed-effects regression with quant interaction.
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 13: Energy–Performance Correlation & Regression")
    print(SEPARATOR)

    results = {}
    df_reg = df.dropna(subset=["success_rate", "energy_per_task"]).copy()
    df_reg["log_energy"] = np.log1p(df_reg["energy_per_task"])

    # Overall Spearman
    rho, p = stats.spearmanr(df_reg["energy_per_task"], df_reg["success_rate"])
    print(f"\n  Overall Spearman: ρ = {rho:.4f}, p = {p:.6f}")
    results["spearman_overall"] = {"rho": rho, "p": p}

    # Stratified by quant
    print("\n  Stratified by Quantization:")
    for quant in ["bf16", "q8_0", "q4_k_m"]:
        sub = df_reg[df_reg["quant"] == quant]
        rho, p = stats.spearmanr(sub["energy_per_task"], sub["success_rate"])
        print(f"    {quant:<8}: ρ = {rho:.4f}, p = {p:.6f}, n = {len(sub)}")
        results[f"spearman_{quant}"] = {"rho": rho, "p": p, "n": len(sub)}

    # Mixed-effects regression: success_rate ~ log_energy * quant + (1|model)
    print(f"\n  --- Mixed-Effects Regression ---")
    df_reg["quant"] = pd.Categorical(df_reg["quant"], categories=["bf16", "q8_0", "q4_k_m"])
    formula = "success_rate ~ log_energy * C(quant, Treatment('bf16')) + C(domain)"

    try:
        lmm = smf.mixedlm(formula, data=df_reg, groups=df_reg["model"])
        fit = lmm.fit(reml=True)
        print(fit.summary())

        # Key: interaction terms tell us if the energy→success slope differs by quant
        print("\n  Interaction term interpretation:")
        print("  (If significant, the energy-performance trade-off differs by quant level)")
        for param in fit.fe_params.index:
            if ":" in param:
                coef = fit.fe_params[param]
                pval = fit.pvalues[param]
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                print(f"    {param}: coef={coef:.6f}, p={pval:.6f} {sig}")

        results["lmm_aic"] = fit.aic
        results["lmm_bic"] = fit.bic
        results["lmm_converged"] = fit.converged

    except Exception as e:
        print(f"  [ERROR] Regression failed: {e}")
        results["regression_error"] = str(e)

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 14: Efficiency Ratio Analysis
# ─────────────────────────────────────────────────────────────────────────
def test_efficiency_ratio(df):
    """
    Compute efficiency_ratio = success_rate / log(energy_per_task),
    then compare across quantization levels.
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 14: Efficiency Ratio Analysis")
    print(SEPARATOR)

    results = {}
    df_eff = df.dropna(subset=["success_rate", "energy_per_task"]).copy()
    df_eff = df_eff[df_eff["energy_per_task"] > 0].copy()

    # Compute efficiency ratio
    df_eff["log_energy"] = np.log(df_eff["energy_per_task"])
    df_eff["efficiency_ratio"] = df_eff["success_rate"] / df_eff["log_energy"]
    # Handle cases where success_rate is 0
    df_eff.loc[df_eff["success_rate"] == 0, "efficiency_ratio"] = 0

    # Descriptive stats per quant
    print("\n  Efficiency Ratio (success_rate / log(energy_per_task)):")
    print("\n  Descriptive statistics per quantization level:")
    desc = df_eff.groupby("quant")["efficiency_ratio"].describe()
    print(desc.round(6).to_string())
    results["descriptives"] = desc.to_dict()

    # Kruskal-Wallis
    quant_groups = [grp["efficiency_ratio"].dropna().values
                    for _, grp in df_eff.groupby("quant")]
    kw_stat, kw_p = stats.kruskal(*quant_groups)
    n_total = sum(len(g) for g in quant_groups)
    k = len(quant_groups)
    epsilon_sq = (kw_stat - k + 1) / (n_total - k)

    print(f"\n  Kruskal-Wallis H test:")
    print(f"    H = {kw_stat:.4f}, p = {kw_p:.6f}, ε² = {epsilon_sq:.4f}")
    results["kruskal_wallis"] = {"H": kw_stat, "p": kw_p, "epsilon_sq": epsilon_sq}

    if HAS_POSTHOCS and kw_p < 0.05:
        dunn = sp.posthoc_dunn(df_eff, val_col="efficiency_ratio",
                               group_col="quant", p_adjust="bonferroni")
        print(f"    Dunn's post-hoc (Bonferroni):")
        print(f"    {dunn.to_string()}")
        results["dunn"] = dunn.to_dict()

    # Paired Wilcoxon: bf16 vs each quantized variant (matched by model-size-domain)
    print(f"\n  Paired Wilcoxon Signed-Rank (matched by model-size-domain):")
    pivoted = df_eff.groupby(["model", "size", "domain", "quant"])["efficiency_ratio"].mean()
    pivoted = pivoted.reset_index().pivot_table(
        index=["model", "size", "domain"],
        columns="quant",
        values="efficiency_ratio"
    ).dropna()

    for q_compare in ["q8_0", "q4_k_m"]:
        if "bf16" in pivoted.columns and q_compare in pivoted.columns:
            v_bf16 = pivoted["bf16"].values
            v_q = pivoted[q_compare].values
            diff = v_bf16 - v_q

            if not np.all(diff == 0):
                try:
                    w_stat, w_p = stats.wilcoxon(v_bf16, v_q, alternative="two-sided")
                    median_diff = np.median(diff)
                    # Effect size: r = Z / sqrt(N)
                    n = len(diff)
                    z = stats.norm.ppf(1 - w_p / 2)
                    r = z / np.sqrt(n)
                    print(f"    bf16 vs {q_compare}: W={w_stat:.1f}, p={w_p:.6f}, "
                          f"median_diff={median_diff:.6f}, r={r:.4f}")
                    results[f"wilcoxon_bf16_vs_{q_compare}"] = {
                        "W": w_stat, "p": w_p, "median_diff": median_diff, "r": r, "n": n
                    }
                except Exception as e:
                    print(f"    bf16 vs {q_compare}: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 15: Relative Change Analysis (Degradation Metrics)
# ─────────────────────────────────────────────────────────────────────────
def test_relative_change(df):
    """
    Paired analysis of performance degradation and energy savings
    from bf16 → q8_0 and bf16 → q4_k_m.
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 15: Relative Change Analysis (Degradation Metrics)")
    print(SEPARATOR)

    results = {}

    # Average across runs for each (model, size, quant, domain)
    agg = df.groupby(["model", "size", "quant", "domain"]).agg(
        success_rate=("success_rate", "mean"),
        energy_per_task=("energy_per_task", "mean"),
    ).reset_index()

    # Pivot to get bf16 as baseline
    sr_pivot = agg.pivot_table(index=["model", "size", "domain"],
                                columns="quant", values="success_rate")
    en_pivot = agg.pivot_table(index=["model", "size", "domain"],
                                columns="quant", values="energy_per_task")

    for q_compare in ["q8_0", "q4_k_m"]:
        print(f"\n  --- bf16 → {q_compare} ---")

        # Filter to configs that have both bf16 and the comparison quant
        valid_idx = sr_pivot[["bf16", q_compare]].dropna().index
        sr_base = sr_pivot.loc[valid_idx, "bf16"]
        sr_quant = sr_pivot.loc[valid_idx, q_compare]
        en_base = en_pivot.loc[valid_idx, "bf16"]
        en_quant = en_pivot.loc[valid_idx, q_compare]

        n = len(valid_idx)
        print(f"  N matched configurations: {n}")

        # Absolute differences
        delta_sr = sr_quant - sr_base  # negative = performance loss
        delta_en = en_base - en_quant  # positive = energy saving

        # Relative changes (avoid division by zero)
        rel_sr = np.where(sr_base > 0.001,
                          (sr_quant - sr_base) / sr_base,
                          np.where(sr_quant > sr_base, 1.0, 0.0))
        rel_en = np.where(en_base > 0.001,
                          (en_base - en_quant) / en_base,
                          0.0)

        print(f"\n  Success Rate Change (absolute):")
        print(f"    Mean Δ = {delta_sr.mean():.6f}")
        print(f"    Median Δ = {delta_sr.median():.6f}")
        print(f"    Std = {delta_sr.std():.6f}")
        print(f"    Range = [{delta_sr.min():.6f}, {delta_sr.max():.6f}]")

        print(f"\n  Energy Savings (absolute, Joules/task):")
        print(f"    Mean savings = {delta_en.mean():.2f}")
        print(f"    Median savings = {delta_en.median():.2f}")

        print(f"\n  Relative Changes:")
        print(f"    Success rate change: median = {np.median(rel_sr)*100:.2f}%")
        print(f"    Energy savings: median = {np.median(rel_en)*100:.2f}%")

        # Wilcoxon signed-rank: is the success rate change significantly != 0?
        print(f"\n  Wilcoxon Signed-Rank Tests:")
        sr_result = {}
        en_result = {}

        try:
            if not np.all(delta_sr.values == 0):
                w_sr, p_sr = stats.wilcoxon(delta_sr.values, alternative="two-sided")
                z_sr = stats.norm.ppf(1 - p_sr / 2)
                r_sr = z_sr / np.sqrt(n)
                print(f"    Success rate Δ ≠ 0:  W={w_sr:.1f}, p={p_sr:.6f}, r={r_sr:.4f}")
                sr_result = {"W": w_sr, "p": p_sr, "r": r_sr}
            else:
                print(f"    Success rate: all differences are zero")
                sr_result = {"note": "all zero"}
        except Exception as e:
            print(f"    Success rate: {e}")

        try:
            if not np.all(delta_en.values == 0):
                w_en, p_en = stats.wilcoxon(delta_en.values, alternative="two-sided")
                z_en = stats.norm.ppf(1 - p_en / 2)
                r_en = z_en / np.sqrt(n)
                print(f"    Energy savings Δ ≠ 0: W={w_en:.1f}, p={p_en:.6f}, r={r_en:.4f}")
                en_result = {"W": w_en, "p": p_en, "r": r_en}
            else:
                print(f"    Energy: all differences are zero")
                en_result = {"note": "all zero"}
        except Exception as e:
            print(f"    Energy: {e}")

        # Quadrant analysis
        print(f"\n  Quadrant Analysis (Performance × Energy):")
        n_win_win = np.sum((delta_sr >= 0) & (delta_en >= 0))  # better perf + less energy
        n_lose_win = np.sum((delta_sr < 0) & (delta_en >= 0))  # worse perf + less energy
        n_win_lose = np.sum((delta_sr >= 0) & (delta_en < 0))  # better perf + more energy
        n_lose_lose = np.sum((delta_sr < 0) & (delta_en < 0))  # worse perf + more energy
        print(f"    Win-Win  (↑perf, ↓energy): {n_win_win:>3} ({n_win_win/n*100:.1f}%)")
        print(f"    Trade-off(↓perf, ↓energy): {n_lose_win:>3} ({n_lose_win/n*100:.1f}%)")
        print(f"    Inverse  (↑perf, ↑energy): {n_win_lose:>3} ({n_win_lose/n*100:.1f}%)")
        print(f"    Lose-Lose(↓perf, ↑energy): {n_lose_lose:>3} ({n_lose_lose/n*100:.1f}%)")

        # Per-configuration breakdown
        print(f"\n  Per-configuration details:")
        configs = valid_idx.to_frame(index=False)
        configs["delta_success"] = delta_sr.values
        configs["delta_energy"] = delta_en.values
        configs["rel_success_%"] = rel_sr * 100
        configs["rel_energy_%"] = rel_en * 100
        configs = configs.sort_values("delta_success")
        print(configs.to_string(index=False, float_format="{:.4f}".format))

        results[f"bf16_to_{q_compare}"] = {
            "n": n,
            "success_delta_mean": float(delta_sr.mean()),
            "success_delta_median": float(delta_sr.median()),
            "energy_savings_mean": float(delta_en.mean()),
            "energy_savings_median": float(delta_en.median()),
            "relative_success_median_pct": float(np.median(rel_sr) * 100),
            "relative_energy_median_pct": float(np.median(rel_en) * 100),
            "wilcoxon_success": sr_result,
            "wilcoxon_energy": en_result,
            "quadrant": {
                "win_win": int(n_win_win),
                "trade_off": int(n_lose_win),
                "inverse": int(n_win_lose),
                "lose_lose": int(n_lose_lose),
            }
        }

    return results


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def run_all_rq3():
    """Run all RQ3 tests and save results."""
    print("\n" + "=" * 72)
    print("  RQ3: Task Effectiveness vs. Computational Efficiency Trade-off")
    print("=" * 72)

    df = load_run_data()
    all_results = {}

    all_results["test11_pareto"] = test_pareto(df)
    all_results["test12_manova"] = test_manova(df)
    all_results["test13_regression"] = test_energy_performance_regression(df)
    all_results["test14_efficiency_ratio"] = test_efficiency_ratio(df)
    all_results["test15_relative_change"] = test_relative_change(df)

    save_results(all_results, "rq3_results.json")
    return all_results


if __name__ == "__main__":
    run_all_rq3()
