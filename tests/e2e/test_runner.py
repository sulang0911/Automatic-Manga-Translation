import sys
import os
import time
import pytest

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["QT_QPA_PLATFORM"] = "offscreen"

SUITES = [
    ("Tier 1: GUI Shell & Controls (F-GUI-01 ~ F-GUI-08)", "tests/e2e/test_tier1_gui.py"),
    ("Tier 1: Local OCR Engine (F-OCR-01 ~ F-OCR-05)", "tests/e2e/test_tier1_ocr.py"),
    ("Tier 1: Background Inpainting (F-INP-01 ~ F-INP-05)", "tests/e2e/test_tier1_inpaint.py"),
    ("Tier 1: Multi-Provider LLM (F-TRN-01 ~ F-TRN-06)", "tests/e2e/test_tier1_translation.py"),
    ("Tier 1: Smart Typography (F-TYP-01 ~ F-TYP-06)", "tests/e2e/test_tier1_typography.py"),
    ("Tier 1: Interactive Bubble Editor (F-EDT-01 ~ F-EDT-06)", "tests/e2e/test_tier1_editing.py"),
    ("Tier 1: Batch & Chapter Export (F-EXP-01 ~ F-EXP-05)", "tests/e2e/test_tier1_export.py"),
    ("Tier 1: Async Concurrency (F-ASY-01 ~ F-ASY-04)", "tests/e2e/test_tier1_async.py"),
    ("Tier 1: Error & Resilience (F-ERR-01 ~ F-ERR-06)", "tests/e2e/test_tier1_error.py"),
    ("Tier 2: Boundary & Corner Cases", "tests/e2e/test_tier2_boundary.py"),
    ("Tier 3: Pairwise Cross-Feature Interactions", "tests/e2e/test_tier3_combinations.py"),
    ("Tier 4: Real-World Scenarios", "tests/e2e/test_tier4_scenarios.py"),
]

def main():
    print("=" * 80)
    print("  AETHERLENS MANGA TRANSLATION SYSTEM — 4-TIER E2E TEST RUNNER")
    print("=" * 80)
    print(f"Project Directory : {BASE_DIR}")
    print(f"Timestamp         : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    total_passed = 0
    total_failed = 0
    results = []

    start_time = time.time()

    for title, test_file in SUITES:
        print(f"\n[RUNNING] {title} -> {test_file}")
        t0 = time.time()
        # Run pytest programmatically
        exit_code = pytest.main(["-q", test_file])
        duration = time.time() - t0
        passed = (exit_code == 0)
        results.append((title, test_file, passed, duration))
        if passed:
            print(f"[PASS] {title} ({duration:.2f}s)")
        else:
            print(f"[FAIL] {title} ({duration:.2f}s)")
            total_failed += 1

    total_duration = time.time() - start_time

    # Summary Table
    print("\n" + "=" * 80)
    print("  E2E TEST EXECUTION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Tier / Category':<56} | {'Status':<8} | {'Time':<7}")
    print("-" * 80)
    for title, _, passed, duration in results:
        status_str = "PASSED" if passed else "FAILED"
        print(f"{title:<56} | {status_str:<8} | {duration:>5.2f}s")
    print("=" * 80)
    
    if total_failed == 0:
        print(f"\n>>> ALL SUITES PASSED CLEANLY! Total time: {total_duration:.2f}s <<<\n")
        sys.exit(0)
    else:
        print(f"\n>>> {total_failed} SUITE(S) FAILED. Total time: {total_duration:.2f}s <<<\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
