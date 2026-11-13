"""Generate ``MATURITY_EVIDENCE_M17651.md`` (L2/L3/L4 evidence bundle).

Everything is computed, not typed: file hashes, the stored artefact contents,
the integration-caller replay and the pytest summary are executed/read here and
embedded verbatim.

``PYTHONPATH=src python benchmarks/biaxial_shear_band/make_maturity_evidence.py``
``--no-pytest`` reuses the last recorded test summary instead of running pytest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

DOC = "MATURITY_EVIDENCE_M17651.md"
CASE_DIR = "artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE"

FILES = [
    ("implementation (case)", "src/tensordem/biaxial_anchor_case.py"),
    ("implementation (DEM cell + measurement chain)", "src/tensordem/biaxial_shear_band.py"),
    ("entry point", "benchmarks/biaxial_shear_band/run_anchor_case.py"),
    ("benchmark (4 configurations)", "benchmarks/biaxial_shear_band/run_benchmark.py"),
    ("benchmark artefact (4 configurations)", "artifacts/biaxial_shear_band/results.json"),
    ("run artefact", f"{CASE_DIR}/run.json"),
    ("figure", f"{CASE_DIR}/shear_band_case.png"),
    ("doc generator", "benchmarks/biaxial_shear_band/make_anchor_doc.py"),
    ("generated report", "docs/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE.md"),
    ("integration caller / maturity gate", "src/tensordem/integration_callers.py"),
    ("tests", "tests/test_biaxial_anchor_case.py"),
]


def _sha256(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _pytest_summary(run: bool) -> str:
    if not run:
        return "(not run in this invocation)"
    env = dict(os.environ, PYTHONPATH="src")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_biaxial_anchor_case.py",
         "tests/test_biaxial_shear_band.py", "tests/test_biaxial_band_diagnostics.py",
         "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env, timeout=1800)
    tail = [line for line in proc.stdout.strip().splitlines() if line.strip()][-1:]
    return f"`{' '.join(tail)}` (exit {proc.returncode})"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pytest", dest="pytest", action="store_false")
    args = parser.parse_args(argv)

    from tensordem.integration_callers import (
        check_biaxial_maturity_gate,
        load_biaxial_case_result,
    )

    run_json = os.path.join(CASE_DIR, "run.json")
    with open(run_json) as handle:
        payload = json.load(handle)
    case = load_biaxial_case_result(CASE_DIR)
    gate = check_biaxial_maturity_gate(case, required_level="L4")
    verdict = case["verdict"]
    known = case["known_angle"]

    lines: list[str] = []
    add = lines.append
    add("# MATURITY EVIDENCE m17651 - CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE")
    add("")
    add(f"Case: `{payload['case']}` (TensorDEM). Anchor: shear-band inclination of a "
        f"plane-strain biaxial cell, 24 x 48 spacings, `sigma_3` = "
        f"{payload['config']['confining_stress'] / 1e3:.0f} kPa, wall friction "
        f"{payload['config']['wall_friction']}, "
        f"{payload['config']['consolidation_steps']} consolidation steps; reference "
        f"{payload['anchor']['ref_deg']:.0f} deg (Desrues & Viggiani 2004, 55-60 deg), "
        f"tolerance {100 * payload['anchor']['tolerance']:.0f} %.")
    add("")
    add(f"**Outcome: {'PASS' if payload['summary']['all_passed'] else 'honest FAIL'}** - "
        f"`theta` = {verdict['got_deg']:.2f} deg vs {verdict['ref_deg']:.0f} deg = "
        f"{100 * verdict['relative_error']:.2f} % error, localisation z = "
        f"{verdict['z_score']:.1f} (needs > 3), known-angle operator error max "
        f"{known['max_abs_error_deg']:.2f} deg. Maturity gate replay: "
        f"**{gate['level_achieved']}** (requested {gate['requested_level']}, "
        f"passed={gate['passed']}, gaps={gate['gaps']}).")
    add("")
    add("## Evidence files (hashes are of the delivered tree)")
    add("")
    add("| role | path | bytes | sha256[:16] |")
    add("|---|---|---|---|")
    for role, path in FILES:
        size = os.path.getsize(path) if os.path.exists(path) else 0
        add(f"| {role} | `{path}` | {size} | `{_sha256(path)}` |")
    add("")
    add("## L2 - implementation + entry + run artefacts")
    add("")
    add("* Implementation: `src/tensordem/biaxial_anchor_case.py` "
        "(known-angle benchmark, anchor run, pre-declared acceptance arithmetic, "
        "figure) on top of the production DEM cell "
        "`src/tensordem/biaxial_shear_band.py`; no placeholder code.")
    add("* Entry: `benchmarks/biaxial_shear_band/run_anchor_case.py` "
        "(`--quick` smoke, `--no-dem` known-angle only, exit code = verdict).")
    add(f"* Run artefacts: `{run_json}` (full band history, config echo, environment "
        f"echo, repro command) and `{CASE_DIR}/shear_band_case.png`.")
    add("")
    add("## L3 - integration caller (platform side)")
    add("")
    add("`tensordem.integration_callers.run_biaxial_shear_band_case(output_dir=..., "
        "quick=False)` returns the machine-checkable verdict dict; "
        "`load_biaxial_case_result()` replays a stored artefact **without** re-running "
        "the DEM; `check_biaxial_maturity_gate(result, 'L4')` re-derives the ladder from "
        "files on disk. CLI: `python -m tensordem.integration_callers --biaxial-case`.")
    add("")
    add("Replay of the delivered artefact:")
    add("")
    add("```json")
    add(json.dumps({"overall": case["overall"], "verdict": verdict, "gate": gate},
                   indent=1, default=float))
    add("```")
    add("")
    add("## L4 - benchmark (reference vs got) + repro command")
    add("")
    add("| quantity | reference | got | error |")
    add("|---|---|---|---|")
    add(f"| band inclination (deg) | {verdict['ref_deg']:.1f} (literature) | "
        f"{verdict['got_deg']:.2f} | {100 * verdict['relative_error']:.2f} % |")
    for row in known["rows"]:
        add(f"| known-angle band, injected {row['injected_deg']:.0f} deg (deg) | "
            f"{row['injected_deg']:.1f} (analytic) | {row['recovered_deg']:.1f} | "
            f"{row['abs_error_deg']:.2f} deg |")
    add("")
    add("```")
    add(payload["repro_cmd"])
    add("```")
    add("")
    add("## Test evidence")
    add("")
    add("Commands: `PYTHONPATH=src python -m pytest tests/test_biaxial_anchor_case.py "
        "tests/test_biaxial_shear_band.py tests/test_biaxial_band_diagnostics.py -q`")
    add("")
    add(f"Result: {_pytest_summary(args.pytest)}")
    add("")
    add("Documented pitfalls / limits: global `psi` of the servo cell is unphysical "
        "(`psi > phi`), so Roscoe/Arthur rows are not evidence; the band-local dilation "
        "estimator stays WITHHELD; 16 x 32 specimens do not resolve a band and are "
        "reported as `theta = NaN`, never as a pass.")
    add("")

    with open(DOC, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"[written] {DOC}")
    print(json.dumps({"case": case["overall"], "gate": gate}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())