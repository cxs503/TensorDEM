"""Tests for CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE (anchor case plumbing).

The case is called *known-angle* on purpose: the measurement operator is
validated against an analytically known band inclination before it is applied to
the DEM result.  These tests pin that validation, the pre-declared acceptance
arithmetic, the artefact-writing entry, and the platform-side integration
caller / maturity gate.  They do not re-assert the DEM headline number (that is
the benchmark's job, ``benchmarks/biaxial_shear_band/run_anchor_case.py``).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from tensordem.biaxial_anchor_case import (
    ANCHOR_CONFIG,
    CASE_KEY,
    THETA_REF_DEG,
    TOLERANCE,
    case_lattice,
    judge_anchor,
    measurement_window,
    run_known_angle_benchmark,
)

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_anchor_configuration_matches_the_task_board():
    """The pre-declared anchor configuration is not silently mutated."""
    assert CASE_KEY == "CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE"
    assert ANCHOR_CONFIG["columns"] == 24
    assert ANCHOR_CONFIG["rows"] == 48
    assert ANCHOR_CONFIG["confining_stress"] == pytest.approx(100.0e3)
    assert ANCHOR_CONFIG["wall_friction"] == pytest.approx(0.55)
    assert ANCHOR_CONFIG["consolidation_steps"] == 4000
    assert THETA_REF_DEG == pytest.approx(60.0)
    assert TOLERANCE == pytest.approx(0.15)


def test_case_lattice_and_window_are_consistent():
    pos = case_lattice(24, 48)
    # triangular lattice, one or two particles per row: 24 or 23 per row
    assert pos.shape[0] in (24 * 48, 23 * 24 + 24 * 24)
    assert np.isfinite(pos).all()
    xmin, xmax, ymin, ymax = measurement_window(pos)
    assert xmax > xmin and ymax > ymin
    window_aspect = (xmax - xmin) / (ymax - ymin)
    assert window_aspect < 0.75          # elongated specimen: the biased case


def test_known_angle_recovery_is_unbiased_on_the_case_lattice():
    """Injected 30/45/60/70 deg bands must come back within 2 deg (<=3 %)."""
    benchmark = run_known_angle_benchmark((30.0, 45.0, 60.0, 70.0))
    wrong = [(row["injected_deg"], row["recovered_deg"])
             for row in benchmark["rows"] if row["abs_error_deg"] > 2.0]
    assert not wrong, f"known-angle recovery off by >2 deg: {wrong}"
    assert benchmark["passed"] is True
    assert benchmark["max_abs_error_deg"] <= 2.0
    # every band must actually be detected as a band, not guessed
    assert all(row["z_score"] > 3.0 for row in benchmark["rows"])


def test_known_angle_benchmark_keeps_the_biased_baseline_visible():
    """Regression guard: the pre-R2 estimator must stay reproducibly biased."""
    benchmark = run_known_angle_benchmark((45.0, 60.0))
    legacy = {row["injected_deg"]: row["legacy_recovered_deg"]
              for row in benchmark["rows"]}
    # documented R2 diagnosis: 60 deg -> 43 deg, 45 deg -> 28 deg
    assert abs(legacy[60.0] - 43.0) <= 3.0
    assert abs(legacy[45.0] - 28.0) <= 3.0
    assert benchmark["legacy_max_abs_error_deg"] > 10.0


def _fake_dem(theta=55.5, z=15.0, eta2=0.5, resolved=True, std=1.0):
    return {
        "band": {"angle_deg": theta, "resolved": resolved,
                 "angle_std_deg": std, "late_snapshots": 4,
                 "representative": {"cumulative": {"z_score": z, "eta2": eta2}}},
    }


def test_judge_anchor_accepts_a_resolved_band_within_tolerance():
    verdict = judge_anchor(_fake_dem(55.5))
    assert verdict["within_tolerance"] is True       # 7.5 % < 15 %
    assert verdict["localised"] is True
    assert verdict["passed"] is True


@pytest.mark.parametrize("theta,z,resolved", [
    (44.0, 15.0, True),      # 26.7 % error: the pre-R2 failure mode
    (55.5, 1.8, True),       # no localisation (z below the null level)
    (float("nan"), 15.0, False),   # unresolved band must not be scored as pass
])
def test_judge_anchor_rejects(theta, z, resolved):
    assert judge_anchor(_fake_dem(theta, z=z, resolved=resolved))["passed"] is False


def test_entry_script_writes_machine_checkable_artifacts(tmp_path):
    """The L2 entry: a CLI run must leave a JSON artefact with ref-vs-got."""
    import subprocess
    import sys

    out_dir = tmp_path / "case"
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "benchmarks", "biaxial_shear_band",
                                      "run_anchor_case.py"),
         "--no-dem", "--no-figures", "--out-dir", str(out_dir)],
        capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = json.load(open(out_dir / "run.json"))
    assert data["case"] == CASE_KEY
    assert data["anchor"]["ref_deg"] == pytest.approx(60.0)
    assert data["known_angle"]["rows"]
    assert "run_anchor_case.py" in data["repro_cmd"]
    assert data["summary"]["known_angle_passed"] is True


def test_integration_caller_and_maturity_gate(tmp_path):
    """L3/L4: the platform-side caller returns a verdict and the gate agrees."""
    from tensordem.integration_callers import (
        check_biaxial_maturity_gate,
        run_biaxial_shear_band_case,
    )

    case = run_biaxial_shear_band_case(str(tmp_path / "caller"), quick=True,
                                       run_dem=True, figures=False)
    assert case["overall"] in ("PASS", "FAIL", "PARTIAL")
    assert case["known_angle"]["passed"] is True
    assert case["verdict"] is not None
    assert os.path.exists(case["artifacts"]["run_json"])
    gate = check_biaxial_maturity_gate(case, required_level="L4")
    assert gate["passed"] is True, gate["gaps"]
    assert gate["level_achieved"] == "L4"

def test_delivered_run_artifact_replays_to_a_pass():
    """L2 guard: the committed artefact must exist and re-derive a PASS verdict.

    If the run artefacts are ever dropped from the tree (the defect this
    rework round fixes), this test fails instead of the delivery silently
    degrading to L1.
    """
    from tensordem.integration_callers import (
        check_biaxial_maturity_gate,
        load_biaxial_case_result,
    )

    case_dir = os.path.join("artifacts", "CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE")
    assert os.path.exists(os.path.join(case_dir, "run.json"))
    assert os.path.exists(os.path.join(case_dir, "shear_band_case.png"))
    case = load_biaxial_case_result(case_dir)
    assert case["overall"] == "PASS", case["verdict"]
    assert case["verdict"]["got_deg"] == pytest.approx(55.5, abs=2.0)
    assert case["known_angle"]["max_abs_error_deg"] <= 2.0
    gate = check_biaxial_maturity_gate(case, required_level="L4")
    assert gate["passed"] is True, gate["gaps"]
