"""Execute named, reproducible DEM3D benchmark cases and summarize acceptance checks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensordem.dem3d import DEM3DConfig
from scripts.benchmark_dem3d import run_benchmark


def load_case_suite(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark suite not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse benchmark suite {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("benchmark suite must be an object with a 'cases' array")
    if not payload["cases"]:
        raise ValueError("benchmark suite must define at least one case")
    seen: set[str] = set()
    for index, case in enumerate(payload["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"cases[{index}] must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", case_id):
            raise ValueError(f"cases[{index}].id must be a lowercase slug")
        if case_id in seen:
            raise ValueError(f"duplicate benchmark case id: {case_id}")
        seen.add(case_id)
        if not isinstance(case.get("config"), dict):
            raise ValueError(f"{case_id}.config must be an object")
        for key in ("steps", "sample_every"):
            value = case.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{case_id}.{key} must be a positive integer")
        minimum = case.get("minimum_broken_bonds", 0)
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            raise ValueError(f"{case_id}.minimum_broken_bonds must be a nonnegative integer")
        if "verify_repeat" in case and not isinstance(case["verify_repeat"], bool):
            raise ValueError(f"{case_id}.verify_repeat must be boolean")
    return payload


def _check(case: dict[str, Any], output: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any, detail: str) -> None:
        checks.append({
            "case_id": case["id"], "check": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed, "expected": expected, "detail": detail,
        })

    numeric_fields = (
        "dt_s", "final_time_s", "peak_abs_reaction_z_N",
        "final_mechanical_energy_J", "energy_balance_residual_J",
        "energy_balance_residual_relative",
    )
    finite = True
    nonfinite_fields = []
    for key in numeric_fields:
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            finite = False
            nonfinite_fields.append(key)
    add("finite_summary", finite, nonfinite_fields, [],
        "All reported time, force, energy, and residual values must be finite.")

    broken = summary.get("broken_bonds")
    min_broken = case.get("minimum_broken_bonds", 0)
    add("minimum_damage", isinstance(broken, int) and broken >= min_broken,
        broken, f">={min_broken}",
        "Damage threshold is a case-specific regression expectation, not a material law.")

    signature = summary.get("signature_sha256")
    add("signature_present", isinstance(signature, str) and re.fullmatch(r"[0-9a-f]{64}", signature) is not None,
        signature, "64 lowercase hexadecimal characters", "SHA-256 signature of sampled diagnostics and final state.")

    events_path = output / "fracture_events_3d.csv"
    event_count = -1
    if events_path.is_file():
        with events_path.open(newline="", encoding="utf-8") as stream:
            event_count = sum(1 for _ in csv.DictReader(stream))
    add("fracture_event_count_matches", event_count == broken,
        event_count, broken, "Each broken bond must have exactly one exported fracture-event row.")

    if summary.get("repeat_verified"):
        repeat_sig = summary.get("repeat_signature_sha256")
        add("repeat_signature_matches", repeat_sig == signature,
            repeat_sig, signature, "Repeated CPU execution must produce an identical state signature.")
    return checks


def run_benchmark_suite(
    suite_path: Path,
    output: Path,
    selected_cases: Sequence[str] | None = None,
    *,
    verify_repeat: bool = False,
) -> dict[str, Any]:
    suite = load_case_suite(suite_path)
    definitions = suite["cases"]
    requested = set(selected_cases or [])
    known = {case["id"] for case in definitions}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown benchmark case(s): {', '.join(sorted(unknown))}")
    cases = [case for case in definitions if not requested or case["id"] in requested]
    if not cases:
        raise ValueError("no benchmark cases selected")

    output.mkdir(parents=True, exist_ok=True)
    all_checks: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    for case in cases:
        case_output = output / case["id"]
        try:
            config = DEM3DConfig(**case["config"])
            summary = run_benchmark(
                case_output, config, steps=case["steps"],
                sample_every=case["sample_every"],
                verify_repeat=verify_repeat or case.get("verify_repeat", False),
            )
            checks = _check(case, case_output, summary)
            all_checks.extend(checks)
            case_results.append({
                "case_id": case["id"], "title": case.get("title", case["id"]),
                "purpose": case.get("purpose", ""),
                "status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
                "output_dir": str(case_output),
                "steps": case["steps"], "sample_every": case["sample_every"],
                "particle_count": summary["particle_count"],
                "broken_bonds": summary["broken_bonds"],
                "final_time_s": summary["final_time_s"],
                "peak_abs_reaction_z_N": summary["peak_abs_reaction_z_N"],
                "energy_balance_residual_J": summary["energy_balance_residual_J"],
                "signature_sha256": summary["signature_sha256"],
                "checks": checks,
            })
        except (ValueError, RuntimeError, AssertionError, OSError) as exc:
            failure = {
                "case_id": case["id"], "check": "execution",
                "status": "FAIL", "observed": str(exc), "expected": "successful benchmark run",
                "detail": "See the exception for the execution or numerical assertion that failed.",
            }
            all_checks.append(failure)
            case_results.append({
                "case_id": case["id"], "title": case.get("title", case["id"]),
                "purpose": case.get("purpose", ""), "status": "FAIL",
                "output_dir": str(case_output), "error": str(exc), "checks": [failure],
            })

    failed = sum(row["status"] == "FAIL" for row in all_checks)
    report = {
        "protocol": suite.get("protocol", "tensordem-dem3d-benchmark-suite-v1"),
        "suite_file": str(suite_path),
        "selected_cases": [case["id"] for case in cases],
        "verdict": "FAIL" if failed else "PASS",
        "case_count": len(case_results),
        "pass_case_count": sum(row["status"] == "PASS" for row in case_results),
        "fail_case_count": sum(row["status"] == "FAIL" for row in case_results),
        "check_count": len(all_checks),
        "failed_check_count": failed,
        "cases": case_results,
        "checks": all_checks,
        "interpretation": (
            "PASS means only that these configured regression invariants passed. "
            "It is not experimental validation, material calibration, proof of convergence, "
            "or certification for full-scale icebreaking."
        ),
    }
    (output / "benchmark_suite_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "benchmark_suite_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = ["case_id", "check", "status", "observed", "expected", "detail"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_checks)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run named DEM3D regression benchmark cases")
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmarks/dem3d/cases.json")
    parser.add_argument("--case", action="append", dest="cases",
                        help="case id to run; repeat option for multiple cases; default runs all")
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-benchmark-suite"))
    parser.add_argument("--verify-repeat", action="store_true",
                        help="repeat every selected case and require deterministic signatures")
    args = parser.parse_args(argv)
    try:
        report = run_benchmark_suite(
            args.suite, args.output, args.cases, verify_repeat=args.verify_repeat
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "protocol": report["protocol"], "verdict": report["verdict"],
        "case_count": report["case_count"], "pass_case_count": report["pass_case_count"],
        "fail_case_count": report["fail_case_count"], "check_count": report["check_count"],
        "failed_check_count": report["failed_check_count"],
        "summary": str(args.output / "benchmark_suite_summary.json"),
    }, indent=2, sort_keys=True))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
