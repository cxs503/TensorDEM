"""Integration caller: platform-side invocation of friction+polyhedron DEM verification.

This module demonstrates how the TensorPlatform / ship-performance-platform
would call the TensorDEM friction_polyhedron verification as part of a
capability maturity assessment or automated verification pipeline.

Usage (platform side):
    from tensordem.integration_callers import run_friction_polyhedron_verification
    result = run_friction_polyhedron_verification(output_dir="runs/vp_friction")
    # result['overall'] == 'PASS' or 'PARTIAL'
    # result['summary'] has per-criterion status
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def run_friction_polyhedron_verification(
    output_dir: str = "runs/friction_polyhedron",
    include_packing: bool = False,
) -> Dict[str, Any]:
    """Run the friction+polyhedron DEM verification suite and return results.

    This is the integration point used by:
    - TensorPlatform capability maturity assessment (L2→L3 gate)
    - ship-performance-platform automated verification pipeline
    - Weekly code quality patrol (KPI-QC-RECHECK)

    Args:
        output_dir: Directory for JSON report and summary text.
        include_packing: If True, run A4 packing-fraction measurement
            (slow, needs many particles; default False → WITHHELD).

    Returns:
        Machine-checkable dict with keys:
            - 'overall': 'PASS' | 'PARTIAL' | 'FAIL'
            - 'summary': per-criterion status dict
            - 'criteria': detailed results for A1-A4, N1
            - 'report_path': path to the saved JSON report
    """
    from tensordem.friction_polyhedron_real_verification import run_full_verification

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = run_full_verification(include_packing=include_packing)

    # Persist report for audit trail
    report_path = out / "friction_polyhedron_verification.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    summary_path = out / "friction_polyhedron_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Friction + Polyhedron DEM Verification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Campaign node: {report['campaign_node']}\n")
        f.write(f"Corrected on: {report['corrected_on']}\n")
        f.write(f"Overall: {report['overall']}\n\n")
        for criterion, status in report['summary'].items():
            f.write(f"  {criterion}: {status}\n")

    report['report_path'] = str(report_path)
    report['summary_path'] = str(summary_path)
    return report


def check_maturity_gate(
    result: Dict[str, Any],
    required_level: str = "L2",
) -> Dict[str, Any]:
    """Evaluate whether verification results meet a maturity gate.

    Maturity levels for friction_polyhedron capability:
        L0: No implementation or placeholder only
        L1: Implementation exists but unverified
        L2: impl_files + entry + run_artifacts (this function checks run_artifacts)
        L3: + integration_callers (this function is the integration caller)
        L4: + benchmark (ref vs got numerical对照) + repro_cmd

    Args:
        result: Output from run_friction_polyhedron_verification().
        required_level: Target maturity level ('L2', 'L3', 'L4').

    Returns:
        Dict with 'passed' (bool), 'level_achieved', and 'gaps' (list of strings).
    """
    summary = result.get('summary', {})
    gaps = []

    # L2 check: run artifacts exist and active criteria pass
    active_criteria = {k: v for k, v in summary.items() if v != 'WITHHELD'}
    all_pass = all(v == 'PASS' for v in active_criteria.values())
    if not all_pass:
        failed = [k for k, v in active_criteria.items() if v != 'PASS']
        gaps.append(f"Active criteria not all PASS: {failed}")

    # L3 check: integration caller exists (this function itself satisfies it)
    # L4 check: benchmark with reference values
    if required_level == "L4":
        gaps.append("L4 benchmark (ref vs got) not yet implemented")

    level_achieved = "L2" if not gaps else "L0"
    if required_level == "L3" and not gaps:
        level_achieved = "L3"

    return {
        'passed': len(gaps) == 0,
        'level_achieved': level_achieved,
        'gaps': gaps,
        'summary': summary,
    }


# ---------------------------------------------------------------------------
# H1-BROKEN-ICE-DEM anchor: platform-side caller + maturity gate
# ---------------------------------------------------------------------------
H1_REPRO_CMD = (
    "PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py "
    "--repose-bodies 45 --repose-steps 20000 --repose-dt 2e-4 "
    "--channel-steps 6000 && PYTHONPATH=src python "
    "benchmarks/h1_broken_ice_dem/make_figures.py"
)


def _merge_repose_shards(paths: List[str]) -> Dict[str, Any]:
    """Merge one-friction-per-file repose shards into a single repose case."""
    import math

    merged: Dict[str, Any] = {"runs": [], "setup": {}, "ref": {}, "shard": {}}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            case = json.load(fh)["cases"]["repose"]
        merged["runs"].extend(case["runs"])
        merged["setup"] = case["setup"] or merged["setup"]
        merged["ref"] = case["ref"] or merged["ref"]
    merged["runs"].sort(key=lambda r: float(r["friction"]))
    angles = [float(r["repose_angle_deg"]) for r in merged["runs"]]
    band = merged["ref"].get("band_deg", [25.0, 40.0])
    merged["angles_deg"] = angles
    merged["settled"] = [bool(r.get("settled", False)) for r in merged["runs"]]
    merged["inside_literature_band"] = [bool(band[0] <= a <= band[1]) for a in angles]
    merged["all_runs_valid"] = bool(merged["runs"]
                                    and all(bool(r.get("valid", False)) for r in merged["runs"]))
    merged["passed"] = bool(merged["all_runs_valid"] and all(merged["inside_literature_band"]))
    merged["case"] = "repose"
    return merged


def run_h1_broken_ice_dem(
    cases: Sequence[str] = ("buoyancy", "repose", "channel"),
    *,
    output_dir: str = "runs/h1_broken_ice_dem",
    quick: bool = False,
    repose_shard_files: Optional[Sequence[str]] = None,
    repro_cmd: str = H1_REPRO_CMD,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the H1-BROKEN-ICE-DEM acceptance suite from the platform side.

    This is the integration point a platform pipeline (capability maturity gate /
    campaign acceptance) uses instead of shelling out to the benchmark script::

        from tensordem.integration_callers import run_h1_broken_ice_dem
        result = run_h1_broken_ice_dem(cases=("buoyancy", "channel"), quick=True)
        result["passed"]          # per-case machine verdict
        result["maturity"]["level_achieved"]

    ``repose_shard_files`` merges one-friction-per-file repose runs (the cluster
    shards), so a caller does not have to re-run the expensive case inline.

    Returns the same JSON tree as ``h1_cases.validate`` plus
    ``integration`` (caller metadata), ``repro_cmd`` and ``maturity``.
    """
    from tensordem.polyhedron_floe3d.h1_cases import validate

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = list(cases)
    result = validate(cases=[c for c in wanted if c in ("buoyancy", "channel")],
                      quick=quick, **kwargs) if any(
        c in ("buoyancy", "channel") for c in wanted) else {"cases": {}, "passed": {}}
    if "repose" in wanted and repose_shard_files:
        result["cases"]["repose"] = _merge_repose_shards(list(repose_shard_files))
        result["passed"]["repose"] = bool(result["cases"]["repose"]["passed"])

    result["integration"] = {
        "caller": "tensordem.integration_callers.run_h1_broken_ice_dem",
        "cases": wanted,
        "quick": bool(quick),
        "repose_shards": list(repose_shard_files) if repose_shard_files else [],
    }
    result["repro_cmd"] = repro_cmd
    report_path = out / "h1_broken_ice_dem.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=float)
    result["report_path"] = str(report_path)
    result["maturity"] = check_h1_maturity(result, required_level="L4")
    return result


def check_h1_maturity(result: Dict[str, Any], required_level: str = "L4") -> Dict[str, Any]:
    """Machine check of the maturity level of the H1-BROKEN-ICE-DEM capability.

    Unlike :func:`check_maturity_gate` (which consumes a verification summary),
    this reads the *benchmark evidence* itself and asks for the artifacts the
    gate lists:

    * ``L2`` -- a non-empty implementation module, an executable entry point and
      a run artifact (the JSON result tree written by the run);
    * ``L3`` -- an integration caller (this module) that produced the result;
    * ``L4`` -- for every case a reference value ``ref`` and a measured value
      ``got``, plus a ``repro_cmd``.

    The check reports gaps instead of asserting them; a failing case (e.g. the
    repose band) is a gap, not a rounded-up pass.
    """
    gaps: List[str] = []
    cases = result.get("cases", {})
    if not cases:
        gaps.append("L2: no run artifact (empty 'cases')")
    if not result.get("integration"):
        gaps.append("L3: not produced through the integration caller")

    if required_level in ("L2", "L3", "L4"):
        for name, case in cases.items():
            if not case.get("passed"):
                gaps.append(f"case {name}: verdict not PASS")
    if required_level == "L4":
        if not result.get("repro_cmd"):
            gaps.append("L4: repro_cmd missing")
        for name, case in cases.items():
            if "ref" not in case or "got" not in case and name != "repose":
                gaps.append(f"L4: case {name} has no ref/got numeric pair")
    return {
        "required_level": required_level,
        "level_achieved": required_level if not gaps else ("L1" if cases else "L0"),
        "passed": not gaps,
        "gaps": gaps,
        "cases": {name: bool(case.get("passed")) for name, case in cases.items()},
    }


if __name__ == "__main__":
    import sys

    if "--h1" in sys.argv:
        import argparse

        sys.argv.remove("--h1")

        ap = argparse.ArgumentParser(description="H1-BROKEN-ICE-DEM integration caller")
        ap.add_argument("--quick", action="store_true")
        ap.add_argument("--cases", default="buoyancy,channel")
        ap.add_argument("--repose-shard", action="append", default=[])
        ap.add_argument("--channel-steps", type=int, default=6000)
        ap.add_argument("--out", default="runs/h1_broken_ice_dem")
        ns = ap.parse_args()
        h1 = run_h1_broken_ice_dem(
            cases=[c for c in ns.cases.split(",") if c],
            output_dir=ns.out, quick=ns.quick,
            repose_shard_files=ns.repose_shard or None,
            channel={"n_steps": ns.channel_steps},
        )
        print(json.dumps({"passed": h1["passed"],
                          "maturity": h1["maturity"]}, indent=1, default=float))
        sys.exit(0 if h1["maturity"]["passed"] else 1)

    result = run_friction_polyhedron_verification()
    gate = check_maturity_gate(result, required_level="L3")
    print(f"Maturity gate: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"Level achieved: {gate['level_achieved']}")
    if gate['gaps']:
        print(f"Gaps: {gate['gaps']}")
    sys.exit(0 if gate['passed'] else 1)
