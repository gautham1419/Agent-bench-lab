"""
rq1_tests.py
============
RQ1: How do model parameter scale and precision quantization influence
     task success rates in interactive, goal-oriented agent environments?

Tests implemented:
  1. Two-Way ANOVA  (Size × Quantization)
  2. Kruskal-Wallis H Test  (non-parametric robustness check)
  3. Linear Mixed-Effects Model  (primary analysis)
  4. Spearman Rank Correlation  (scaling trend)
  5. Cochran-Mantel-Haenszel Test  (stratified association)
"""

import warnings, textwrap
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

from data_loader import load_run_data, get_matched_data, save_results

SEPARATOR = "=" * 72


# ─────────────────────────────────────────────────────────────────────────
# Test 1: Two-Way Factorial ANOVA
# ─────────────────────────────────────────────────────────────────────────
def test_twoway_anova(df):
    """Two-Way ANOVA: Size x Quantization on success_rate."""
    print(f"\n{SEPARATOR}")
    print("  TEST 1: Two-Way Factorial ANOVA (Size x Quantization)")
    print(SEPARATOR)

    results = {}

    # Assumption checks
    # 1. Normality per group (Shapiro-Wilk)
    print("\n  [Assumption Check] Shapiro-Wilk normality test per cell:")
    normality_results = []
    for (sz, q), grp in df.groupby(["size", "quant"]):
        vals = grp["success_rate"].dropna()
        if len(vals) >= 3:
            stat, p = stats.shapiro(vals)
            normality_results.append({"size": sz, "quant": q, "W": stat, "p": p, "n": len(vals)})
            flag = " FAIL" if p < 0.05 else " OK"
            print(f"    {sz:>5} x {q:<8}  W={stat:.4f}  p={p:.4f}  n={len(vals)}{flag}")
    results["shapiro_wilk"] = normality_results

    # 2. Levene's test for homogeneity of variance
    groups = [grp["success_rate"].dropna().values for _, grp in df.groupby(["size", "quant"]) if len(grp) >= 2]
    if len(groups) >= 2:
        lev_stat, lev_p = stats.levene(*groups)
        print(f"\n  [Assumption Check] Levene's test: F={lev_stat:.4f}, p={lev_p:.4f}")
        results["levene"] = {"F": lev_stat, "p": lev_p}

    # ANOVA via OLS
    # Arcsine-sqrt transform for proportional data
    df_anova = df.copy()
    df_anova["success_asin"] = np.arcsin(np.sqrt(df_anova["success_rate"].clip(0, 1)))

    print("\n  --- ANOVA on raw success_rate ---")
    model_raw = smf.ols("success_rate ~ C(size) * C(quant)", data=df_anova).fit()
    anova_raw = anova_lm(model_raw, typ=2)
    print(anova_raw.to_string())
    results["anova_raw"] = anova_raw.to_dict()

    # Partial eta-squared
    ss_residual = anova_raw.loc["Residual", "sum_sq"]
    for factor in anova_raw.index:
        if factor != "Residual":
            ss_factor = anova_raw.loc[factor, "sum_sq"]
            eta_sq = ss_factor / (ss_factor + ss_residual)
            print(f"    Partial eta^2 for {factor}: {eta_sq:.4f}")

    print("\n  --- ANOVA on arcsine-sqrt transformed success_rate ---")
    model_asin = smf.ols("success_asin ~ C(size) * C(quant)", data=df_anova).fit()
    anova_asin = anova_lm(model_asin, typ=2)
    print(anova_asin.to_string())
    results["anova_transformed"] = anova_asin.to_dict()

    # Post-hoc: Tukey HSD for size
    print("\n  --- Post-hoc: Tukey HSD for Size ---")
    tukey_size = pairwise_tukeyhsd(df["success_rate"].values, df["size"].values, alpha=0.05)
    print(tukey_size.summary())

    # Post-hoc: Tukey HSD for quant
    print("\n  --- Post-hoc: Tukey HSD for Quantization ---")
    tukey_quant = pairwise_tukeyhsd(df["success_rate"].values, df["quant"].values, alpha=0.05)
    print(tukey_quant.summary())

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 2: Kruskal-Wallis H Test
# ─────────────────────────────────────────────────────────────────────────
def test_kruskal_wallis(df):
    """Kruskal-Wallis H test — non-parametric alternative for each factor."""
    print(f"\n{SEPARATOR}")
    print("  TEST 2: Kruskal-Wallis H Test (Non-Parametric)")
    print(SEPARATOR)

    results = {}

    # By quantization level
    quant_groups = [grp["success_rate"].dropna().values for _, grp in df.groupby("quant")]
    kw_stat, kw_p = stats.kruskal(*quant_groups)
    n_total = sum(len(g) for g in quant_groups)
    epsilon_sq = kw_stat / (n_total - 1)  # effect size
    print(f"\n  Grouping by Quantization:")
    print(f"    H = {kw_stat:.4f},  p = {kw_p:.6f},  e^2 = {epsilon_sq:.4f}")
    results["quant"] = {"H": kw_stat, "p": kw_p, "epsilon_sq": epsilon_sq}

    if HAS_POSTHOCS and kw_p < 0.05:
        print("    Post-hoc (Dunn's test with Bonferroni):")
        dunn = sp.posthoc_dunn(df, val_col="success_rate", group_col="quant", p_adjust="bonferroni")
        print(dunn.to_string())
        results["quant_dunn"] = dunn.to_dict()

    # By model size
    size_groups = [grp["success_rate"].dropna().values for _, grp in df.groupby("size")]
    kw_stat, kw_p = stats.kruskal(*size_groups)
    n_total = sum(len(g) for g in size_groups)
    epsilon_sq = kw_stat / (n_total - 1)
    print(f"\n  Grouping by Size:")
    print(f"    H = {kw_stat:.4f},  p = {kw_p:.6f},  e^2 = {epsilon_sq:.4f}")
    results["size"] = {"H": kw_stat, "p": kw_p, "epsilon_sq": epsilon_sq}

    if HAS_POSTHOCS and kw_p < 0.05:
        print("    Post-hoc (Dunn's test with Bonferroni):")
        dunn = sp.posthoc_dunn(df, val_col="success_rate", group_col="size", p_adjust="bonferroni")
        print(dunn.to_string())
        results["size_dunn"] = dunn.to_dict()

    # By model family
    model_groups = [grp["success_rate"].dropna().values for _, grp in df.groupby("model")]
    kw_stat, kw_p = stats.kruskal(*model_groups)
    n_total = sum(len(g) for g in model_groups)
    epsilon_sq = kw_stat / (n_total - 1)
    print(f"\n  Grouping by Model Family:")
    print(f"    H = {kw_stat:.4f},  p = {kw_p:.6f},  e^2 = {epsilon_sq:.4f}")
    results["model"] = {"H": kw_stat, "p": kw_p, "epsilon_sq": epsilon_sq}

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 3: Linear Mixed-Effects Model
# ─────────────────────────────────────────────────────────────────────────
def test_lmm(df):
    """
    Linear Mixed-Effects Model.
    Fixed: C(size_num) * C(quant, Treatment('bf16')), domain
    Random: (1 | model)
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 3: Linear Mixed-Effects Model (Primary Analysis)")
    print(SEPARATOR)

    results = {}

    df_lmm = df.dropna(subset=["success_rate"]).copy()
    df_lmm["quant"] = pd.Categorical(df_lmm["quant"], categories=["bf16", "q8_0", "q4_k_m"])

    # Model with quant as categorical, size as numeric, domain as fixed, model as random
    formula = "success_rate ~ C(size_num) * C(quant, Treatment('bf16')) + C(domain)"

    print(f"\n  Formula: {formula}")
    print(f"  Random effect: (1 | model)")
    print(f"  N = {len(df_lmm)}")

    try:
        lmm = smf.mixedlm(formula, data=df_lmm, groups=df_lmm["model"])
        lmm_fit = lmm.fit(reml=True)
        print(lmm_fit.summary())

        # Extract key results
        results["converged"] = lmm_fit.converged
        results["log_likelihood"] = lmm_fit.llf

        # AIC/BIC are not defined for REML fits; store only if valid
        import math
        if not math.isnan(lmm_fit.aic):
            results["aic"] = lmm_fit.aic
            results["bic"] = lmm_fit.bic

        # Fixed effects
        fe = lmm_fit.fe_params
        pvals = lmm_fit.pvalues
        print("\n  Fixed Effects Summary:")
        print(f"    {'Parameter':<55} {'Coef':>8} {'p-value':>10}")
        print(f"    {'-'*75}")
        for param in fe.index:
            print(f"    {param:<55} {fe[param]:>8.4f} {pvals[param]:>10.6f}")

        results["fixed_effects"] = {k: {"coef": fe[k], "p": pvals[k]} for k in fe.index}

        # Random effects variance
        re_var = lmm_fit.cov_re.iloc[0, 0]
        resid_var = lmm_fit.scale
        icc = re_var / (re_var + resid_var)
        print(f"\n  Random Effects:")
        print(f"    Model intercept variance: {re_var:.6f}")
        print(f"    Residual variance: {resid_var:.6f}")
        print(f"    ICC (model): {icc:.4f}")
        results["random_effects"] = {"model_var": re_var, "resid_var": resid_var, "icc": icc}

    except Exception as e:
        print(f"  [ERROR] LMM failed: {e}")
        results["error"] = str(e)

    # Simpler model without interaction for comparison
    print(f"\n  --- Reduced Model (no interaction) ---")
    formula2 = "success_rate ~ C(size_num) + C(quant, Treatment('bf16')) + C(domain)"
    try:
        lmm2 = smf.mixedlm(formula2, data=df_lmm, groups=df_lmm["model"])
        lmm2_fit = lmm2.fit(reml=False)  # ML for model comparison
        lmm_full = smf.mixedlm(formula, data=df_lmm, groups=df_lmm["model"]).fit(reml=False)

        lr_stat = -2 * (lmm2_fit.llf - lmm_full.llf)
        df_diff = lmm_full.df_modelwc - lmm2_fit.df_modelwc
        lr_p = 1 - stats.chi2.cdf(lr_stat, max(df_diff, 1))
        print(f"  Likelihood Ratio Test (interaction term):")
        print(f"    chi2 = {lr_stat:.4f}, df = {df_diff}, p = {lr_p:.6f}")
        results["lr_test_interaction"] = {"chi2": lr_stat, "df": int(df_diff), "p": lr_p}
    except Exception as e:
        print(f"  [NOTE] Model comparison failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 4: Spearman Rank Correlation
# ─────────────────────────────────────────────────────────────────────────
def test_spearman(df):
    """Spearman correlation between parameter count and success_rate."""
    print(f"\n{SEPARATOR}")
    print("  TEST 4: Spearman Rank Correlation (Size -> Success Rate)")
    print(SEPARATOR)

    results = {}

    # Overall
    rho, p = stats.spearmanr(df["size_num"], df["success_rate"])
    print(f"\n  Overall:  rho = {rho:.4f},  p = {p:.6f},  n = {len(df)}")
    results["overall"] = {"rho": rho, "p": p, "n": len(df)}

    # Stratified by quantization level
    print("\n  Stratified by Quantization:")
    for quant_level in ["bf16", "q8_0", "q4_k_m"]:
        sub = df[df["quant"] == quant_level]
        rho, p = stats.spearmanr(sub["size_num"], sub["success_rate"])
        print(f"    {quant_level:<8}:  rho = {rho:.4f},  p = {p:.6f},  n = {len(sub)}")
        results[quant_level] = {"rho": rho, "p": p, "n": len(sub)}

    # Stratified by domain
    print("\n  Stratified by Domain:")
    for domain in df["domain"].unique():
        sub = df[df["domain"] == domain]
        rho, p = stats.spearmanr(sub["size_num"], sub["success_rate"])
        print(f"    {domain:<10}:  rho = {rho:.4f},  p = {p:.6f},  n = {len(sub)}")
        results[f"domain_{domain}"] = {"rho": rho, "p": p, "n": len(sub)}

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 5: Cochran-Mantel-Haenszel Test
# ─────────────────────────────────────────────────────────────────────────
def test_cmh(df):
    """
    Cochran-Mantel-Haenszel test: association between quantization and
    success/failure outcome, stratified by domain.
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 5: Cochran-Mantel-Haenszel Test (Stratified by Domain)")
    print(SEPARATOR)

    results = {}

    # Build stratified 2×3 tables (outcome: success/failure × quant: bf16/q8_0/q4_k_m)
    # Aggregate counts across model-size-run within each domain×quant
    tables = []
    quant_order = ["bf16", "q8_0", "q4_k_m"]

    print("\n  Stratified contingency tables (Success vs Failure × Quant):")
    for domain in sorted(df["domain"].unique()):
        print(f"\n    Domain: {domain}")
        sub = df[df["domain"] == domain]
        table_data = []
        for q in quant_order:
            q_data = sub[sub["quant"] == q]
            total_successes = q_data["successes"].sum()
            total_failures = q_data["failures"].sum() + q_data["errors"].sum()
            table_data.append([total_successes, total_failures])
            print(f"      {q:<8}: successes={total_successes:>5}, failures={total_failures:>5}")
        table_array = np.array(table_data).T  # 2 rows (success/fail) × 3 cols (quant)
        tables.append(table_array)

    # CMH test using statsmodels StratifiedTable
    try:
        from statsmodels.stats.contingency_tables import StratifiedTable
        st = StratifiedTable(tables)
        cmh_result = st.test_null_odds()
        print(f"\n  CMH Test Results:")
        print(f"    Statistic = {cmh_result.statistic:.4f}")
        print(f"    p-value   = {cmh_result.pvalue:.6f}")
        results["cmh_statistic"] = cmh_result.statistic
        results["cmh_pvalue"] = cmh_result.pvalue

        # Common odds ratio
        print(f"\n  Common Odds Ratio (Mantel-Haenszel):")
        summary = st.summary()
        print(f"    {summary}")
        results["summary"] = str(summary)
    except Exception as e:
        print(f"\n  [NOTE] StratifiedTable requires 2x2 tables per stratum.")
        print(f"         Running pairwise CMH tests instead (bf16 vs each quant).")

        # Pairwise: bf16 vs q4_k_m, bf16 vs q8_0
        for q_compare in ["q8_0", "q4_k_m"]:
            print(f"\n    --- bf16 vs {q_compare} ---")
            tables_2x2 = []
            for domain in sorted(df["domain"].unique()):
                sub = df[df["domain"] == domain]
                bf16_data = sub[sub["quant"] == "bf16"]
                q_data = sub[sub["quant"] == q_compare]
                table = np.array([
                    [bf16_data["successes"].sum(), bf16_data["failures"].sum() + bf16_data["errors"].sum()],
                    [q_data["successes"].sum(), q_data["failures"].sum() + q_data["errors"].sum()]
                ])
                tables_2x2.append(table)

            try:
                st2 = StratifiedTable(tables_2x2)
                cmh2 = st2.test_null_odds()
                print(f"    Statistic = {cmh2.statistic:.4f}")
                print(f"    p-value   = {cmh2.pvalue:.6f}")
                oddsratio = st2.oddsratio_pooled
                print(f"    Pooled OR = {oddsratio:.4f}")
                results[f"cmh_bf16_vs_{q_compare}"] = {
                    "statistic": cmh2.statistic,
                    "p": cmh2.pvalue,
                    "pooled_or": oddsratio,
                }
            except Exception as e2:
                print(f"    [ERROR] {e2}")
                results[f"cmh_bf16_vs_{q_compare}"] = {"error": str(e2)}

    return results


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def run_all_rq1():
    """Run all RQ1 tests and save results."""
    print("\n" + "=" * 72)
    print("  RQ1: Model Scale & Quantization -> Task Success Rates")
    print("=" * 72)

    df = load_run_data()
    all_results = {}

    all_results["test1_twoway_anova"] = test_twoway_anova(df)
    all_results["test2_kruskal_wallis"] = test_kruskal_wallis(df)
    all_results["test3_lmm"] = test_lmm(df)
    all_results["test4_spearman"] = test_spearman(df)
    all_results["test5_cmh"] = test_cmh(df)

    save_results(all_results, "rq1_results.json")
    return all_results


if __name__ == "__main__":
    run_all_rq1()
