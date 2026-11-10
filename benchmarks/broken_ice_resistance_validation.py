#!/usr/bin/env python
"""Run the broken-ice direct-navigation validation campaign.

    python benchmarks/broken_ice_resistance_validation.py --jobs 24

Writes ``artifacts/broken_ice_resistance_validation/`` (cases, criteria, anchors,
figures) and ``docs/BROKEN_ICE_RESISTANCE_VALIDATION.md``.  Every number in the
report is injected from the campaign objects; nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.broken_ice_resistance.ledger import CASE_SPECS  # noqa: E402
from tensordem.broken_ice_resistance.report import (build_markdown,  # noqa: E402
                                                    make_figures, write_artifacts)
from tensordem.broken_ice_resistance.validation import (count_tests,  # noqa: E402
                                                        evaluate_criteria,
                                                        run_campaign)

ARTIFACT_DIR = ROOT / "artifacts" / "broken_ice_resistance_validation"
DOC_PATH = ROOT / "docs" / "BROKEN_ICE_RESISTANCE_VALIDATION.md"
TEST_PATH = ROOT / "tests" / "test_broken_ice_resistance.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="*", default=None,
                        help="subset of cases (default: every case in the ledger)")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--reuse", type=Path, default=None,
                        help="reuse the per-case results of an earlier run "
                             "(post-processing only, no simulation)")
    args = parser.parse_args(argv)

    if args.reuse is not None:
        payload = json.loads(Path(args.reuse).read_text())
        from tensordem.broken_ice_resistance.simulate import CaseResult
        results = {name: CaseResult(**body) for name, body in payload["cases"].items()}
        if args.names:
            results = {k: v for k, v in results.items() if k in set(args.names)}
        print(f"reused {len(results)} case results from {args.reuse}", flush=True)
    else:
        names = args.names or list(CASE_SPECS)
        print(f"running {len(names)} cases on {args.jobs} workers", flush=True)
        results = run_campaign(names, jobs=args.jobs,
                               progress=lambda message: print("  " + message, flush=True))

    pytest_summary = ""
    tests_passed = True
    if not args.skip_pytest:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q",
                               "-c", str(ROOT / "pyproject.toml"), str(TEST_PATH)],
                              cwd=ROOT, capture_output=True, text=True)
        pytest_summary = proc.stdout.strip().splitlines()[-1] if proc.stdout else ""
        tests_passed = proc.returncode == 0
        print("pytest:", pytest_summary, "rc", proc.returncode, flush=True)

    criteria = evaluate_criteria(results, test_count=count_tests(TEST_PATH),
                                 tests_passed=tests_passed,
                                 test_path=str(TEST_PATH.relative_to(ROOT)))
    paths = write_artifacts(results, criteria, args.out, include_series=True)
    figures = {}
    if not args.no_figures:
        try:
            figures = make_figures(results, criteria, args.out)
        except Exception as exc:  # pragma: no cover - plotting is optional
            print(f"figure generation failed: {exc}", flush=True)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(build_markdown(results, criteria, figures,
                                       pytest_summary=pytest_summary))

    summary = criteria["summary"]
    print(json.dumps({"criteria": {k: v["verdict"] for k, v in
                                   criteria["criteria"].items()},
                      "summary": summary,
                      "standard_kN": round(results["standard"].mean_kN, 2),
                      "lindqvist_kN": round(criteria["criteria"]["C1"]["lindqvist_kN"], 2),
                      "artifacts": str(args.out),
                      "doc": str(args.doc)}, indent=1))
    return 0 if summary["fail"] == 0 and summary["withheld"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())