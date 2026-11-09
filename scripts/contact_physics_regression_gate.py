"""Run the upstream node test suites as the integration regression gate (d).

Acceptance criterion (d) of ``P121W5-N11``: the pytest suites of the matured
nodes must still pass after the integration layer lands.

* ``P121W5-N02`` temperature-dependent bond -> ``test_ice_bond_*``
* ``P121W5-N05`` ridge morphology statistics -> ``test_ridge_dem_core_physics``,
  ``test_ice_rubble_microstructure``
* ``P121W5-N08`` lubrication-contact coupling -> ``test_interface_thermal_*``

The gate is executed out-of-process (each suite in its own pytest run, so an
import-time failure of one suite cannot mask another) and the result is written
to ``artifacts/contact_physics/regression_gate.json``; the integration test
asserts that artifact is green and newer than the module sources.

Usage
-----
``python3 scripts/contact_physics_regression_gate.py``
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "contact_physics" / "regression_gate.json"

NODE_TESTS = {
    "P121W5-N02": (
        "tests/test_ice_bond_viscoelastic.py",
        "tests/test_ice_bond_engine.py",
        "tests/test_ice_bond_single_bond_engine.py",
    ),
    "P121W5-N05": (
        "tests/test_ridge_dem_core_physics.py",
        "tests/test_ice_rubble_microstructure.py",
    ),
    "P121W5-N08": (
        "tests/test_interface_thermal_conduction.py",
        "tests/test_interface_thermal_coupling.py",
        "tests/test_interface_thermal_friction.py",
        "tests/test_interface_thermal_stefan.py",
    ),
}

_SUMMARY = re.compile(
    r"(?P<passed>\d+) passed|(?P<failed>\d+) failed|(?P<errors>\d+) error")


def _parse(output: str) -> dict[str, int]:
    tail = output.strip().splitlines()[-1] if output.strip() else ""
    counts = {"passed": 0, "failed": 0, "errors": 0}
    for part in tail.split(","):
        match = re.search(r"(\d+) (passed|failed|error|errors|skipped)", part)
        if not match:
            continue
        value, kind = int(match.group(1)), match.group(2)
        if kind == "passed":
            counts["passed"] = value
        elif kind == "failed":
            counts["failed"] = value
        elif kind in ("error", "errors"):
            counts["errors"] = value
    return counts


def main() -> int:
    results = []
    total = {"passed": 0, "failed": 0, "errors": 0}
    returncode = 0
    started = time.time()
    for node, files in NODE_TESTS.items():
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   *files]
        completed = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                                   text=True)
        counts = _parse(completed.stdout)
        returncode = returncode or completed.returncode
        total["passed"] += counts["passed"]
        total["failed"] += counts["failed"]
        total["errors"] += counts["errors"]
        results.append({
            "node": node,
            "files": list(files),
            "returncode": completed.returncode,
            **counts,
            "tail": completed.stdout.strip().splitlines()[-1:] or [""],
        })
        print(f"{node}: rc={completed.returncode} {counts} {len(files)} files")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_unix": time.time(),
        "wall_s": time.time() - started,
        "node_tests": sorted(path for files in NODE_TESTS.values() for path in files),
        "returncode": returncode,
        **total,
        "per_node": results,
        "command": "python3 -m pytest -q -p no:cacheprovider <files>",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: rc={returncode} {total}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())