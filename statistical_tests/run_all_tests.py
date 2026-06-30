"""
run_all_tests.py
================
Master runner: executes all statistical tests for RQ1, RQ2, and RQ3.
Saves individual results per RQ and a combined summary.

Usage:
    python run_all_tests.py          # Run all
    python run_all_tests.py rq1      # Run only RQ1
    python run_all_tests.py rq2      # Run only RQ2
    python run_all_tests.py rq3      # Run only RQ3
    python run_all_tests.py rq3_48   # Run 4-bit vs 8-bit efficiency comparison (RQ3 supplement)
    python run_all_tests.py tost       # Run TOST equivalence analysis
    python run_all_tests.py deployment # Run operational deployment metrics
    python run_all_tests.py behavior   # Run behavioral consistency analysis
"""

import sys
import time

from data_loader import ensure_output_dir, load_run_data, save_results


def main():
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    print("+" + "=" * 70 + "+")
    print("|  Agent-Bench Quantization Study -- Statistical Tests Runner       |")
    print("+" + "=" * 70 + "+")

    # Verify data loads
    df = load_run_data()
    print(f"\n  Data loaded: {len(df)} run-level observations")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(
        f"  Sizes:  {sorted(df['size'].unique(), key=lambda x: float(x.replace('B', '')))}"
    )
    print(f"  Quants: {sorted(df['quant'].unique())}")
    print(f"  Domains: {sorted(df['domain'].unique())}")
    ensure_output_dir()

    all_results = {}
    start = time.time()

    # RQ1
    if target in ("all", "rq1"):
        from rq1_tests import run_all_rq1

        t0 = time.time()
        all_results["rq1"] = run_all_rq1()
        print(f"\n  [RQ1 completed in {time.time() - t0:.1f}s]")

    # RQ2
    if target in ("all", "rq2"):
        from rq2_tests import run_all_rq2

        t0 = time.time()
        all_results["rq2"] = run_all_rq2()
        print(f"\n  [RQ2 completed in {time.time() - t0:.1f}s]")

    # RQ3
    if target in ("all", "rq3"):
        from rq3_tests import run_all_rq3

        t0 = time.time()
        all_results["rq3"] = run_all_rq3()
        print(f"\n  [RQ3 completed in {time.time() - t0:.1f}s]")

    # TOST equivalence analysis (task-level paired)
    if target in ("all", "tost"):
        from tost_equivalence import run_tost

        t0 = time.time()
        run_tost()
        print(f"\n  [TOST completed in {time.time() - t0:.1f}s]")

    # Operational deployment metrics
    if target in ("all", "deployment"):
        from deployment_metrics import run_deployment_metrics

        t0 = time.time()
        run_deployment_metrics()
        print(f"\n  [Deployment metrics completed in {time.time() - t0:.1f}s]")

    # Direct 4-bit vs 8-bit efficiency comparison (RQ3 supplement)
    if target in ("all", "rq3_48"):
        from rq3_4bit_vs_8bit import run_rq3_4bit_vs_8bit

        t0 = time.time()
        all_results["rq3_4bit_vs_8bit"] = run_rq3_4bit_vs_8bit()
        print(f"\n  [4-bit vs 8-bit comparison completed in {time.time() - t0:.1f}s]")

    # Behavioral consistency (outcome flips + interaction turns)
    if target in ("all", "behavior"):
        from behavioral_consistency import run_behavioral_consistency

        t0 = time.time()
        run_behavioral_consistency()
        print(f"\n  [Behavioral consistency completed in {time.time() - t0:.1f}s]")

    elapsed = time.time() - start

    print("\n" + "+" + "=" * 70 + "+")
    print(f"|  All tests completed in {elapsed:.1f}s".ljust(71) + "|")
    print("+" + "=" * 70 + "+")

    # Save combined results
    if len(all_results) > 1:
        save_results(all_results, "all_results_combined.json")


if __name__ == "__main__":
    main()
