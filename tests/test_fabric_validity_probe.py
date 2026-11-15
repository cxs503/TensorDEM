"""Tests for the m22376 micro-mechanical validity probe (CASE-DEM-BIAXIAL-FRICTION R10).

Three layers, matching the three things the mission claims:

1. **Geometry/diagnostics arithmetic** -- `pairwise_census` and
   `tensor_anisotropy` are checked against hand-computable configurations, so the
   reported overlap ratio, coordination number and anisotropy magnitudes cannot
   silently drift.
2. **Judgement logic** -- the V1/V2/V3 verdicts are pure functions of the arm
   table, so they are checked against synthetic tables: a frictionless control
   that reports a finite angle must be FALSIFIED, a mu-insensitive ladder must be
   FALSIFIED, a missing arm must be WITHHELD (not passed).
3. **End-to-end pipeline** -- one tiny specimen (6 x 8, log strain 0.05) is run
   through the probe to prove the record is JSON-serialisable and carries the
   fields the analysis consumes.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load("scripts/m22376_fabric_validity_probe.py", "m22376_probe")
analysis = _load("benchmarks/case_dem_biaxial_friction/analyse_fabric_validity.py",
                 "m22376_analysis")

sys.path.insert(0, os.path.join(REPO, "src"))


# --------------------------------------------------------------------------
# 1. geometry / diagnostics arithmetic
# --------------------------------------------------------------------------
def test_pairwise_census_counts_overlap_and_gap():
    r = 1.0
    pos = np.array([[0.0, 0.0], [1.5, 0.0], [10.0, 0.0]])  # 0-1 overlap 0.5
    census = probe.pairwise_census(pos, r)
    assert census["n_particles"] == 3
    assert census["n_contacts"] == 1
    assert census["overlap"][0] == pytest.approx(0.5)
    assert census["dist"][0] == pytest.approx(1.5)
    np.testing.assert_allclose(census["normal"][0], [1.0, 0.0])


def test_tensor_anisotropy_extremes():
    # all normals along x -> fabric tensor diag(1, 0), a = 1, major axis 0 deg
    normals = np.tile(np.array([1.0, 0.0]), (50, 1))
    a, theta, f = probe.tensor_anisotropy(normals)
    assert a == pytest.approx(1.0, abs=1e-12)
    assert theta == pytest.approx(0.0, abs=1e-9)
    assert f[0, 0] == pytest.approx(1.0)

    # all normals along y -> a = 1, major axis 90 deg
    a, theta, _ = probe.tensor_anisotropy(np.tile(np.array([0.0, 1.0]), (20, 1)))
    assert a == pytest.approx(1.0, abs=1e-12)
    assert abs(theta) == pytest.approx(90.0, abs=1e-9)

    # uniform orientation -> isotropic, a = 0
    ang = np.linspace(0.0, math.pi, 4096, endpoint=False)
    normals = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    a, _, _ = probe.tensor_anisotropy(normals)
    assert a == pytest.approx(0.0, abs=1e-3)

    # weights = 0 -> undefined, not a spurious number
    a, _, _ = probe.tensor_anisotropy(normals, np.zeros(len(normals)))
    assert math.isnan(a)


def test_micro_diagnostics_reports_expected_keys():
    r = 1.0
    pos = np.array([[0.0, 0.0], [1.8, 0.0], [0.9, 1.6], [0.9, -1.6]])
    snap = {"strain": 0.1, "positions": pos}
    out = probe.micro_diagnostics(snap, r, kn=1.0e7, sigma_v=200.0e3,
                                  sigma_h=100.0e3)
    for key in ("coordination_number", "overlap_ratio_mean", "overlap_ratio_max",
                "fabric_anisotropy_a_c", "normal_force_anisotropy_a_f",
                "dimensionless_stiffness_kn_over_pd",
                "phi_stress_force_fabric_deg", "phi_from_measured_stress_deg"):
        assert key in out, key
    # kn / (p d) with d = 2 r = 2 m, p = 100 kPa -> 1e7 / (1e5 * 2) = 50
    assert out["dimensionless_stiffness_kn_over_pd"] == pytest.approx(50.0)
    # coordination: of the 6 pairs only 2-3 (dist 3.2 > 2 r) is not a contact,
    # so Z = 2 * 5 / 4 = 2.5
    assert out["coordination_number"] == pytest.approx(2.5)


# --------------------------------------------------------------------------
# 2. judgement logic (pure functions of the arm table)
# --------------------------------------------------------------------------
def _row(tag, kn=1e7, mu=0.55, rate=0.05, phi=5.8, vreg=0.01, conv=False):
    return {"tag": tag, "kn_N_per_m": kn, "mu": mu, "rate_1_per_s": rate,
            "vreg": vreg, "phi_cv_deg": phi, "peak_phi_deg": phi + 1.0,
            "q_cv_Pa": 1.0, "sigma_h_cv_Pa": 1.0, "q_drift": 0.3,
            "converged": conv, "flags": {}, "coordination_number": 3.9,
            "overlap_ratio_mean": 0.03, "overlap_ratio_max": 0.1,
            "dimensionless_stiffness": kn / (1e5 * 1.5),
            "fabric_anisotropy_a_c": 0.05,
            "normal_force_anisotropy_a_f": 0.10,
            "phi_stress_force_fabric_deg": 4.0,
            "phi_from_measured_stress_deg": phi,
            "mobilisation": {}, "elapsed_s": 1.0, "source": "synthetic"}


def test_judge_falsifies_dirty_control():
    """A frictionless packing reporting a finite angle must be FALSIFIED."""
    rows = [_row("mu000", mu=0.0, phi=4.0),
            _row("kn1e7", phi=6.0),
            _row("mu150", mu=1.5, phi=20.0)]
    j = analysis.judge(rows)
    assert j["V2_friction"]["mu0_ok"] is False
    assert "FALSIFIED" in j["V2_friction"]["verdict"]


def test_judge_falsifies_mu_insensitivity():
    rows = [_row("mu000", mu=0.0, phi=0.1),
            _row("kn1e7", phi=6.0),
            _row("mu150", mu=1.5, phi=7.0)]
    j = analysis.judge(rows)
    assert j["V2_friction"]["mu0_ok"] is True
    assert j["V2_friction"]["gap_ok"] is False
    assert "mu insensitive" in j["V2_friction"]["verdict"]


def test_judge_withholds_missing_arms():
    rows = [_row("kn1e7", phi=6.0)]
    j = analysis.judge(rows)
    assert j["V1_stiffness"]["verdict"].startswith("WITHHELD")
    assert j["V2_friction"]["verdict"].startswith("WITHHELD")
    assert j["V3_rate"]["verdict"].startswith("WITHHELD")


def test_judge_v1_needs_a_rise_and_the_8pct_gate():
    """Stiffness that recovers the angle supports the soft-particle reading."""
    rows = [_row("kn1e7", kn=1e7, phi=6.0),
            _row("kn1e8", kn=1e8, phi=20.0),
            _row("kn1e9", kn=1e9, phi=28.0),
            _row("mu000", mu=0.0, phi=0.1)]
    j = analysis.judge(rows)
    assert j["V1_stiffness"]["rise_ok"] is True
    assert j["V1_stiffness"]["top_arm_within_8pct"] is True
    assert "SUPPORTED" in j["V1_stiffness"]["verdict"]

    # same rise, but the top arm is still far off -> FALSIFIED reading
    rows[2] = _row("kn1e9", kn=1e9, phi=12.0)
    j = analysis.judge(rows)
    assert j["V1_stiffness"]["top_arm_within_8pct"] is False
    assert "FALSIFIED" in j["V1_stiffness"]["verdict"]


def test_judge_v3_detects_rate_dependence():
    rows = [_row("kn1e7", phi=6.0),
            _row("rate_half", rate=0.025, phi=12.0)]
    j = analysis.judge(rows)
    assert j["V3_rate"]["rate_independent"] is False
    assert "viscous" in j["V3_rate"]["verdict"]


def test_anchor_diagnostic_is_not_a_pass_at_6_deg():
    rows = [_row("kn1e7", phi=5.77)]
    j = analysis.judge(rows)
    anc = j["anchor_diagnostic"]
    assert anc["pass"] is False
    assert anc["relative_error"] == pytest.approx(
        abs(5.77 - math.degrees(math.atan(0.55))) / math.degrees(math.atan(0.55)),
        rel=1e-6)


# --------------------------------------------------------------------------
# 3. end-to-end pipeline on a tiny specimen
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_probe_end_to_end_tiny(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"),
               TENSORDEM_THREADS="2")
    cmd = [sys.executable, os.path.join(REPO, "scripts",
                                        "m22376_fabric_validity_probe.py"),
           "--tag", "tiny", "--columns", "6", "--rows", "8",
           "--rate", "0.05", "--strain-target", "0.05",
           "--consolidation-steps", "150", "--snapshot-strain", "0.02",
           "--min-strain", "0.02",
           "--out-dir", str(tmp_path)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rec = json.load(open(tmp_path / "arm_tiny.json"))
    for key in ("config", "critical_state", "micro", "phi_cv_deg",
                "rowe_reference_deg", "rowe_relative_error", "loading"):
        assert key in rec, key
    assert rec["config"]["kn"] == pytest.approx(1.0e7)
    assert len(rec["micro"]) >= 1
    assert math.isfinite(rec["micro"][-1]["coordination_number"])
    assert rec["micro"][-1]["n_particles"] >= 40