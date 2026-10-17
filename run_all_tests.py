"""
NeuroSeg - Master System Verification & Benchmark Suite
Runs comprehensive end-to-end testing across all 5 project modules:
- Module 1: Preprocessing & Multimodal Co-Registration
- Module 2: 3D Deep Learning Backbones & Loss Functions
- Module 3: Sliding-Window Inference & Validation Metrics (Dice & HD95)
- Module 4: Morphological Filtering, Volumetrics & RANO Engine
- Module 5: 3D Surface Reconstruction & Clinical PDF Generation
"""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from test_module1 import test_module1
from test_module2 import test_module2
from test_module3 import test_module3
from test_module4 import test_module4
from test_module5 import test_module5


def run_all():
    print("=" * 72)
    print("  NEUROSEG: MASTER SYSTEM VALIDATION & BENCHMARK AUDIT")
    print("=" * 72)

    start_time = time.time()
    results = {}

    tests = [
        ("Module 1: Preprocessing & Alignment", test_module1),
        ("Module 2: 3D Architectures & Composite Loss", test_module2),
        ("Module 3: Sliding-Window & Spatial Metrics", test_module3),
        ("Module 4: Morphological Post-Processing & RANO", test_module4),
        ("Module 5: 3D Mesh & Clinical PDF Reporting", test_module5),
    ]

    for name, test_fn in tests:
        t0 = time.time()
        try:
            test_fn()
            elapsed = time.time() - t0
            results[name] = ("PASSED", elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = (f"FAILED: {e}", elapsed)
            print(f"[FAIL] Error in {name}: {e}")

    total_elapsed = time.time() - start_time
    print("=" * 72)
    print("  EXECUTIVE VERIFICATION SCORECARD")
    print("=" * 72)
    all_passed = True
    for name, (status, elap) in results.items():
        icon = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"  {icon} {name:<46} | {status:<8} ({elap:.2f}s)")
        if status != "PASSED":
            all_passed = False

    print("-" * 72)
    print(f"  Total Test Execution Time: {total_elapsed:.2f} seconds")
    if all_passed:
        print("  ALL 5 MODULES VERIFIED SUCCESSFULLY -- PROJECT READY FOR DEFENSE!")
    else:
        print("  SOME MODULES ENCOUNTERED ISSUES -- REVIEW LOGS ABOVE.")
    print("=" * 72)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
