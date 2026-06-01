"""
rq2_tests.py
============
RQ2: How does the structural composition of agent execution failures
     change as language models are quantized?

Tests implemented:
  6.  Chi-Square Test of Homogeneity
  7.  Multinomial Logistic Regression  (on aggregated failure counts)
  8.  Friedman Test  (matched comparison across quant levels)
  9.  Compositional Data Analysis (CoDA) via CLR + MANOVA
  10. Bootstrap Confidence Intervals on failure proportions
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from itertools import combinations

from data_loader import (
    load_run_data, get_failure_counts, get_failure_rates,
    get_matched_data, save_results,
)

SEPARATOR = "=" * 72
FAILURE_TYPES = ["TLE", "IF", "IA", "CF", "TE", "SysErr"]


# ─────────────────────────────────────────────────────────────────────────
# Test 6: Chi-Square Test of Homogeneity
# ─────────────────────────────────────────────────────────────────────────
def test_chi_square(df):
    """
    Chi-Square test of homogeneity:
    Are failure type distributions the same across quantization levels?
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 6: Chi-Square Test of Homogeneity")
    print(SEPARATOR)

    results = {}
    fc = get_failure_counts(df)

    # Aggregate failure counts by quantization level
    agg = fc.groupby("quant")[FAILURE_TYPES].sum()
    agg = agg.loc[["bf16", "q8_0", "q4_k_m"]]  # ensure order

    print("\n  Contingency Table (rows=quant, cols=failure type):")
    print(agg.to_string())

    # Remove columns with all zeros (can cause chi2 issues)
    non_zero_cols = agg.columns[agg.sum() > 0]
    agg_clean = agg[non_zero_cols]

    chi2, p, dof, expected = stats.chi2_contingency(agg_clean.values)
    n = agg_clean.values.sum()
    k = min(agg_clean.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0

    print(f"\n  χ² = {chi2:.4f}")
    print(f"  df = {dof}")
    print(f"  p  = {p:.6f}")
    print(f"  Cramér's V = {cramers_v:.4f}")

    results["overall"] = {"chi2": chi2, "df": dof, "p": p, "cramers_v": cramers_v}

    # Expected frequencies check
    print(f"\n  Expected frequencies (check all ≥ 5):")
    exp_df = pd.DataFrame(expected, index=agg_clean.index, columns=agg_clean.columns)
    print(exp_df.round(1).to_string())
    min_expected = expected.min()
    print(f"  Min expected frequency: {min_expected:.1f}", end="")
    print("  ✓" if min_expected >= 5 else "  ✗ (consider Fisher's exact or collapse categories)")
    results["min_expected_freq"] = min_expected

    # Standardized residuals to identify which cells drive significance
    print(f"\n  Standardized Residuals (|z| > 2 indicates significant deviation):")
    std_resid = (agg_clean.values - expected) / np.sqrt(expected)
    resid_df = pd.DataFrame(std_resid, index=agg_clean.index, columns=agg_clean.columns)
    print(resid_df.round(3).to_string())
    results["standardized_residuals"] = resid_df.to_dict()

    # Pairwise Chi-Square tests between quant levels
    print(f"\n  Pairwise Chi-Square tests (Bonferroni-corrected):")
    quants = ["bf16", "q8_0", "q4_k_m"]
    n_comparisons = 3
    pairwise_results = {}
    for q1, q2 in combinations(quants, 2):
        pair_table = agg_clean.loc[[q1, q2]].values
        # Only keep non-zero columns for this pair
        col_mask = pair_table.sum(axis=0) > 0
        pair_table = pair_table[:, col_mask]
        if pair_table.shape[1] >= 2:
            c2, pv, d, _ = stats.chi2_contingency(pair_table)
            pv_adj = min(pv * n_comparisons, 1.0)
            print(f"    {q1} vs {q2}:  χ²={c2:.4f}, p={pv:.6f}, p_adj={pv_adj:.6f}")
            pairwise_results[f"{q1}_vs_{q2}"] = {"chi2": c2, "p": pv, "p_adjusted": pv_adj}
    results["pairwise"] = pairwise_results

    # Stratified by domain
    print(f"\n  Stratified Chi-Square by Domain:")
    for domain in sorted(df["domain"].unique()):
        fc_dom = fc[fc["domain"] == domain]
        agg_dom = fc_dom.groupby("quant")[FAILURE_TYPES].sum()
        agg_dom = agg_dom.loc[agg_dom.index.isin(quants)]
        non_zero = agg_dom.columns[agg_dom.sum() > 0]
        agg_dom = agg_dom[non_zero]
        if agg_dom.shape[0] >= 2 and agg_dom.shape[1] >= 2:
            c2, pv, d, exp = stats.chi2_contingency(agg_dom.values)
            n_dom = agg_dom.values.sum()
            k_dom = min(agg_dom.shape) - 1
            cv = np.sqrt(c2 / (n_dom * k_dom)) if n_dom * k_dom > 0 else 0
            print(f"    {domain:<10}: χ²={c2:>8.4f}, p={pv:.6f}, V={cv:.4f}")
            results[f"domain_{domain}"] = {"chi2": c2, "p": pv, "cramers_v": cv}

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 7: Multinomial Logistic Regression
# ─────────────────────────────────────────────────────────────────────────
def test_multinomial_logit(df):
    """
    Multinomial logistic regression on failure-type proportions.
    Since we have aggregated data, we expand failure counts into
    individual observations for proper multinomial modeling.
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 7: Multinomial Logistic Regression")
    print(SEPARATOR)

    results = {}
    fc = get_failure_counts(df)

    # Create individual-level data by repeating rows based on counts
    print("\n  Expanding aggregated counts to individual-level observations...")
    rows = []
    for _, row in fc.iterrows():
        meta = {
            "model": row["model"], "size": row["size"], "quant": row["quant"],
            "domain": row["domain"], "size_num": row["size_num"],
        }
        for ftype in FAILURE_TYPES:
            count = int(row[ftype])
            for _ in range(count):
                r = meta.copy()
                r["failure_type"] = ftype
                rows.append(r)

    indiv_df = pd.DataFrame(rows)
    print(f"  Created {len(indiv_df)} individual failure observations")
    print(f"  Distribution:\n{indiv_df['failure_type'].value_counts().to_string()}")

    if len(indiv_df) < 10:
        print("  [SKIP] Too few observations for multinomial logistic regression")
        return {"error": "insufficient data"}

    # Encode variables
    indiv_df["quant_code"] = indiv_df["quant"].map({"bf16": 0, "q8_0": 1, "q4_k_m": 2})

    # Fit multinomial logit
    # DV: failure_type, IVs: quant + size_num + domain
    y = pd.Categorical(indiv_df["failure_type"]).codes
    categories = pd.Categorical(indiv_df["failure_type"]).categories.tolist()

    X = pd.get_dummies(indiv_df[["quant", "domain"]], drop_first=True, dtype=float)
    X["size_num"] = indiv_df["size_num"].values
    X = sm.add_constant(X)

    try:
        model = sm.MNLogit(y, X)
        fit = model.fit(disp=0, maxiter=500)

        print(f"\n  Model Summary:")
        print(f"    Log-Likelihood: {fit.llf:.2f}")
        print(f"    AIC: {fit.aic:.2f}")
        print(f"    BIC: {fit.bic:.2f}")
        print(f"    Pseudo R²: {fit.prsquared:.4f}")

        results["llf"] = fit.llf
        results["aic"] = fit.aic
        results["bic"] = fit.bic
        results["pseudo_r2"] = fit.prsquared
        results["categories"] = categories

        # Likelihood ratio test (full vs intercept-only)
        model_null = sm.MNLogit(y, sm.add_constant(np.ones(len(y))))
        fit_null = model_null.fit(disp=0)
        lr_stat = -2 * (fit_null.llf - fit.llf)
        lr_df = fit.df_model
        lr_p = 1 - stats.chi2.cdf(lr_stat, lr_df)
        print(f"\n  Likelihood Ratio Test (vs null):")
        print(f"    χ² = {lr_stat:.4f}, df = {lr_df}, p = {lr_p:.6f}")
        results["lr_test"] = {"chi2": lr_stat, "df": int(lr_df), "p": lr_p}

        # Print coefficients for quant variables
        print(f"\n  Quantization coefficients (reference: bf16):")
        param_names = X.columns.tolist()
        quant_params = [p for p in param_names if "quant" in p.lower()]
        for qp in quant_params:
            idx = param_names.index(qp)
            print(f"    {qp}:")
            for j, cat in enumerate(categories[1:], 0):  # skip reference category
                coef = fit.params.iloc[j, idx] if j < fit.params.shape[0] else np.nan
                pval = fit.pvalues.iloc[j, idx] if j < fit.pvalues.shape[0] else np.nan
                print(f"      → {cat}: coef={coef:.4f}, p={pval:.6f}")

    except Exception as e:
        print(f"  [ERROR] Multinomial logit failed: {e}")
        results["error"] = str(e)

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 8: Friedman Test
# ─────────────────────────────────────────────────────────────────────────
def test_friedman(df):
    """
    Friedman test: matched comparison of failure rates across quant levels.
    Each (model, size, domain) combo is a 'subject' measured at 3 quant levels.
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 8: Friedman Test (Matched Across Quantization)")
    print(SEPARATOR)

    results = {}
    fr = get_failure_rates(df)

    failure_metrics = FAILURE_TYPES
    quant_order = ["bf16", "q8_0", "q4_k_m"]

    for metric in failure_metrics:
        # Pivot: rows = (model, size, domain), aggregated across runs
        agg = fr.groupby(["model", "size", "domain", "quant"])[metric].mean().reset_index()
        pivoted = agg.pivot_table(index=["model", "size", "domain"],
                                  columns="quant", values=metric).dropna()

        if pivoted.shape[0] < 3 or not all(q in pivoted.columns for q in quant_order):
            print(f"\n  {metric}: Insufficient matched data (n={pivoted.shape[0]})")
            continue

        pivoted = pivoted[quant_order]
        n_subjects = pivoted.shape[0]

        try:
            stat, p = stats.friedmanchisquare(
                pivoted["bf16"].values,
                pivoted["q8_0"].values,
                pivoted["q4_k_m"].values
            )
            # Kendall's W as effect size
            k = len(quant_order)
            w = stat / (n_subjects * (k - 1))

            print(f"\n  {metric}:")
            print(f"    χ²_F = {stat:.4f},  p = {p:.6f},  Kendall's W = {w:.4f},  n = {n_subjects}")
            results[metric] = {"chi2_F": stat, "p": p, "kendalls_w": w, "n": n_subjects}

            # Post-hoc: pairwise Wilcoxon signed-rank tests with Bonferroni
            if p < 0.05:
                print(f"    Post-hoc Wilcoxon signed-rank (Bonferroni-corrected):")
                n_pairs = 3
                for q1, q2 in combinations(quant_order, 2):
                    v1 = pivoted[q1].values
                    v2 = pivoted[q2].values
                    diff = v1 - v2
                    if np.all(diff == 0):
                        print(f"      {q1} vs {q2}: identical values, skipping")
                        continue
                    try:
                        w_stat, w_p = stats.wilcoxon(v1, v2, alternative="two-sided")
                        w_p_adj = min(w_p * n_pairs, 1.0)
                        print(f"      {q1} vs {q2}: W={w_stat:.1f}, p={w_p:.6f}, p_adj={w_p_adj:.6f}")
                        results[f"{metric}_{q1}_vs_{q2}"] = {
                            "W": w_stat, "p": w_p, "p_adjusted": w_p_adj
                        }
                    except Exception as e:
                        print(f"      {q1} vs {q2}: {e}")

        except Exception as e:
            print(f"\n  {metric}: [ERROR] {e}")
            results[metric] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 9: Compositional Data Analysis (CoDA) via CLR + MANOVA
# ─────────────────────────────────────────────────────────────────────────
def test_coda(df):
    """
    Compositional Data Analysis using Centered Log-Ratio (CLR) transformation
    followed by MANOVA to test whether failure composition differs by quant.
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 9: Compositional Data Analysis (CLR + MANOVA)")
    print(SEPARATOR)

    results = {}
    fr = get_failure_rates(df)

    # Average failure rates per (model, size, quant, domain)
    agg = fr.groupby(["model", "size", "quant", "domain"])[FAILURE_TYPES].mean().reset_index()

    # Only keep failure types with non-trivial variation
    comp_cols = [c for c in FAILURE_TYPES if agg[c].sum() > 0]
    comp_data = agg[comp_cols].copy()

    # Replace zeros with small value (requirement for log-ratio transforms)
    epsilon = 1e-6
    comp_data = comp_data + epsilon

    # Close the composition (normalize rows to sum to 1)
    comp_data = comp_data.div(comp_data.sum(axis=1), axis=0)

    # CLR transformation
    log_comp = np.log(comp_data)
    geometric_mean = log_comp.mean(axis=1)
    clr = log_comp.sub(geometric_mean, axis=0)

    print(f"\n  Composition columns used: {comp_cols}")
    print(f"  N observations: {len(clr)}")
    print(f"\n  CLR-transformed data summary:")
    print(clr.describe().round(4).to_string())

    # Add metadata back
    clr_df = pd.concat([agg[["model", "size", "quant", "domain"]].reset_index(drop=True),
                         clr.reset_index(drop=True)], axis=1)

    # MANOVA on CLR-transformed failure composition ~ quant
    dv_formula = " + ".join(comp_cols)
    formula = f"{dv_formula} ~ C(quant)"
    print(f"\n  MANOVA formula: {formula}")

    try:
        from statsmodels.multivariate.manova import MANOVA
        manova = MANOVA.from_formula(formula, data=clr_df)
        mv_result = manova.mv_test()
        print(f"\n  MANOVA Results:")
        print(mv_result.summary())

        # Extract Pillai's trace
        intercept_results = mv_result.results.get("C(quant)", None)
        if intercept_results is not None:
            stat_table = intercept_results["stat"]
            pillai = stat_table.loc["Pillai's trace", "Value"] if "Pillai's trace" in stat_table.index else None
            wilks = stat_table.loc["Wilks' lambda", "Value"] if "Wilks' lambda" in stat_table.index else None
            print(f"\n  Key Statistics:")
            if pillai is not None:
                p_pillai = stat_table.loc["Pillai's trace", "Pr > F"]
                print(f"    Pillai's trace = {pillai:.4f}, p = {p_pillai:.6f}")
                results["pillai"] = {"value": pillai, "p": p_pillai}
            if wilks is not None:
                p_wilks = stat_table.loc["Wilks' lambda", "Pr > F"]
                print(f"    Wilks' λ = {wilks:.4f}, p = {p_wilks:.6f}")
                results["wilks_lambda"] = {"value": wilks, "p": p_wilks}

    except Exception as e:
        print(f"  [ERROR] MANOVA failed: {e}")
        results["error"] = str(e)

        # Fallback: univariate Kruskal-Wallis on each CLR component
        print("\n  Fallback: Kruskal-Wallis on each CLR component")
        for col in comp_cols:
            groups = [grp[col].dropna().values for _, grp in clr_df.groupby("quant")]
            if all(len(g) >= 2 for g in groups):
                h, p = stats.kruskal(*groups)
                print(f"    {col}: H={h:.4f}, p={p:.6f}")
                results[f"kw_{col}"] = {"H": h, "p": p}

    return results


# ─────────────────────────────────────────────────────────────────────────
# Test 10: Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────
def test_bootstrap_ci(df):
    """
    Bootstrap confidence intervals on failure-type proportions
    within each quantization level.
    """
    print(f"\n{SEPARATOR}")
    print("  TEST 10: Bootstrap Confidence Intervals on Failure Composition")
    print(SEPARATOR)

    results = {}
    fc = get_failure_counts(df)
    quant_order = ["bf16", "q8_0", "q4_k_m"]
    n_bootstrap = 10000
    alpha = 0.05

    rng = np.random.default_rng(42)

    for quant in quant_order:
        print(f"\n  Quantization: {quant}")
        sub = fc[fc["quant"] == quant]
        results[quant] = {}

        # Total failures in each category
        totals = sub[FAILURE_TYPES].sum()
        grand_total = totals.sum()

        if grand_total == 0:
            print("    No failures observed")
            continue

        observed_props = totals / grand_total
        print(f"    Observed proportions (n_failures={int(grand_total)}):")

        # Bootstrap: resample runs (cluster bootstrap)
        run_indices = sub.index.values
        n_runs = len(run_indices)

        boot_props = np.zeros((n_bootstrap, len(FAILURE_TYPES)))
        for b in range(n_bootstrap):
            boot_idx = rng.choice(run_indices, size=n_runs, replace=True)
            boot_sample = fc.loc[boot_idx, FAILURE_TYPES].sum()
            boot_total = boot_sample.sum()
            if boot_total > 0:
                boot_props[b] = boot_sample.values / boot_total
            else:
                boot_props[b] = 0

        # BCa confidence intervals (simplified: percentile method)
        for i, ftype in enumerate(FAILURE_TYPES):
            obs = observed_props[ftype]
            ci_low = np.percentile(boot_props[:, i], 100 * alpha / 2)
            ci_high = np.percentile(boot_props[:, i], 100 * (1 - alpha / 2))
            se = np.std(boot_props[:, i])
            print(f"      {ftype:>6}: {obs:.4f}  [{ci_low:.4f}, {ci_high:.4f}]  SE={se:.4f}")
            results[quant][ftype] = {
                "proportion": obs, "ci_low": ci_low, "ci_high": ci_high, "se": se
            }

    # Pairwise comparison: do CIs overlap?
    print(f"\n  Non-overlapping CIs (indicating significant differences):")
    significant_diffs = []
    for ftype in FAILURE_TYPES:
        for q1, q2 in combinations(quant_order, 2):
            if q1 in results and q2 in results:
                r1 = results[q1].get(ftype, {})
                r2 = results[q2].get(ftype, {})
                if r1 and r2:
                    # Check if CIs don't overlap
                    if r1["ci_high"] < r2["ci_low"] or r2["ci_high"] < r1["ci_low"]:
                        diff = r1["proportion"] - r2["proportion"]
                        significant_diffs.append(f"    {ftype}: {q1} vs {q2} (Δ={diff:+.4f})")
                        print(f"    {ftype}: {q1} ({r1['proportion']:.4f}) vs "
                              f"{q2} ({r2['proportion']:.4f})  Δ={diff:+.4f}")

    if not significant_diffs:
        print("    None detected (all CIs overlap)")

    return results


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def run_all_rq2():
    """Run all RQ2 tests and save results."""
    print("\n" + "=" * 72)
    print("  RQ2: Structural Composition of Execution Failures")
    print("=" * 72)

    df = load_run_data()
    all_results = {}

    all_results["test6_chi_square"] = test_chi_square(df)
    all_results["test7_multinomial_logit"] = test_multinomial_logit(df)
    all_results["test8_friedman"] = test_friedman(df)
    all_results["test9_coda"] = test_coda(df)
    all_results["test10_bootstrap_ci"] = test_bootstrap_ci(df)

    save_results(all_results, "rq2_results.json")
    return all_results


if __name__ == "__main__":
    run_all_rq2()
