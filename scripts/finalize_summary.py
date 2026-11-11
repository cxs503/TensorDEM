"""Add the C7 (pytest case count) criterion to a freshly written summary.json.

``run_verification`` writes ``summary.json`` with the physics criteria C1-C6; the
C7 criterion is the *number of pytest cases* of the verification surface, which
is a property of ``tests/test_breakage_pbm.py`` and is therefore counted here,
programmatically, from the file (zero hand transcription).  No numeric result of
the run is touched: only the C7 entry is added and ``passed_all`` is recomputed.

Usage::

    python scripts/finalize_summary.py artifacts/breakage_pbm/summary.json
"""
from __future__ import annotations

import json
import os
import sys

THRESHOLD = 15


def count_tests(path: str) -> int:
    with open(path) as fh:
        return sum(1 for line in fh if line.lstrip().startswith("def test_"))


def main() -> int:
    summary_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/breakage_pbm/summary.json"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "tests/test_breakage_pbm.py"
    if not os.path.exists(test_path):
        print(f"test file not found: {test_path}")
        return 2
    with open(summary_path) as fh:
        s = json.load(fh)
    n = count_tests(test_path)
    s["criteria"]["C7_pytest_cases"] = {
        "value": int(n), "threshold": THRESHOLD,
        "passed": bool(n >= THRESHOLD), "file": test_path,
    }
    s["n_tests"] = int(n)
    s["passed_all"] = bool(all(c["passed"] for c in s["criteria"].values()))
    with open(summary_path, "w") as fh:
        json.dump(s, fh, indent=2)
    print(f"{test_path}: {n} test cases -> C7 {'PASS' if n >= THRESHOLD else 'FAIL'}; "
          f"passed_all = {s['passed_all']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())