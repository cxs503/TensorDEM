"""Negative-control harness for the restart-boundary matrix.

Deliberately injects three single-point defects into
``IceDEM.from_snapshot`` (one restored coordinate, one ``last_step`` scalar and
the whole fracture ledger), reruns ``tests/test_coupling_restart.py`` and
records that the suite turns red each time.  The source file is restored in a
``finally`` block and its hash is re-checked, so the checkout is never left
modified.

Usage::

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/verify_restart_boundary_negative_controls.py

The JSON report is written to ``artifacts/coupling_restart_negative_controls.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "tensordem" / "dem.py"
TARGET = "tests/test_coupling_restart.py"
REPORT = ROOT / "artifacts" / "coupling_restart_negative_controls.json"
ANCHOR = '        result.time, result.step_count = state["time_s"], count\n'

INJECTIONS = {
    "ulp_position": (
        "        result.positions = result.positions.clone()\n"
        "        result.positions[0, 0] = math.nextafter(result.positions[0, 0], math.inf)\n"
    ),
    "last_step_work_ulp": (
        '        result.last_step = copy.deepcopy(state["last_step"])\n'
        '        if "external_work_J" in result.last_step:\n'
        '            result.last_step["external_work_J"] = math.nextafter(\n'
        '                result.last_step["external_work_J"], math.inf)\n'
    ),
    "drop_failure_ledger": "        result.failure_mode = torch.zeros_like(result.failure_mode)\n",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_target() -> dict:
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", TARGET, "-q", "--tb=line"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    failures = [line for line in lines if line.startswith("FAILED")]
    assertions = [line for line in lines if line.startswith("E ") or line.startswith("AssertionError")]
    return {
        "returncode": proc.returncode,
        "summary": next((line for line in reversed(lines) if "passed" in line or "failed" in line), ""),
        "failed_tests": failures,
        "first_assertion": assertions[0] if assertions else "",
    }


def main() -> int:
    original = SOURCE.read_text()
    expected = digest(SOURCE)
    assert ANCHOR in original, "restart anchor moved; update this harness"
    report = {"target": TARGET, "injections": {}}
    try:
        report["clean"] = run_target()
        for name, snippet in INJECTIONS.items():
            SOURCE.write_text(original.replace(ANCHOR, ANCHOR + snippet, 1))
            report["injections"][name] = run_target()
    finally:
        SOURCE.write_text(original)
    assert digest(SOURCE) == expected, "src/tensordem/dem.py was not restored"
    report["source_restored_sha256"] = digest(SOURCE)
    ok = (report["clean"]["returncode"] == 0
          and all(entry["returncode"] != 0 for entry in report["injections"].values()))
    report["negative_controls_red"] = ok
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("clean", "negative_controls_red")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())