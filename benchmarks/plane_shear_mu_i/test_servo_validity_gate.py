"""CASE-DEM-PLANE-SHEAR-MUIDI round 5 tests: constant-stress validity gate.

The round-3 mu(I)/phi(I) sweep was measured with a servo whose platen does not
hold the applied normal stress (up to +57.8 % / -16.5 % at I >= 0.1), so the
stored high-I points are *not* constant-stress measurements.  These tests lock
in (a) the reproduction of the stored artifacts by the gate-off protocol,
(b) the validity verdict of the stored points, (c) the gate's effect, and
(d) the analytic invariants of the device.  Artifact-based, so the suite stays
cheap; the raw numbers come from ``servo_validity_fix.py``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("_psmi_run_sweep", HERE / "run_sweep.py")
rs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rs)   # loaded by path: avoids sys.modules['run_sweep']
                               # collisions when the whole benchmarks/ tree runs
from tensordem.mu_i_rheology import ServoShearCase  # noqa: E402

RUNS = HERE / "runs"
STORED = {"baseline": "sweep_baseline_round2.json",
          "stiff": "sweep_stiff.json",
          "thick": "sweep_thick.json"}
GATE_RAW = RUNS / "servo_validity_gate.json"
TOL = 0.05


def _stored_rows() -> dict:
    out = {}
    for arm, fname in STORED.items():
        path = RUNS / fname
        if not path.exists():
            continue
        for row in json.loads(path.read_text())["rows"]:
            out[(arm, round(float(row["I_target"]), 6))] = row
    return out


def _gate_rows() -> list:
    if not GATE_RAW.exists():
        return []
    return json.loads(GATE_RAW.read_text()).get("rows", [])


# ---------------------------------------------------------------- device spec
def test_default_wall_is_rough_enough_not_to_cap_mu():
    """The platen must be a rough wall: a platen Coulomb cap below the
    reference mu_2 (0.84) would make high-I mu unreachable by construction."""
    case = ServoShearCase()
    assert case.wall_mu > 0.84, (
        f"platen wall_mu={case.wall_mu} < reference mu_2=0.84: high-I mu "
        "would be capped by the wall, not by the material")
    assert case.pressure_control_tol == pytest.approx(TOL)
    # gate off must be reachable without changing the contact law
    assert case.max_recover_steps == 20000


@pytest.mark.parametrize("arm", sorted(STORED))
def test_measured_mu_bounded_by_wall_friction(arm):
    """Sanity invariant: tau_wall <= mu_w * p_wall, so the reported mu can never
    exceed the platen friction coefficient (catches gross measurement errors)."""
    wall_mu = ServoShearCase().wall_mu
    for (a, I), row in _stored_rows().items():
        if a != arm:
            continue
        assert 0.0 < row["mu_measured"] <= wall_mu + 1e-9, (a, I, row["mu_measured"])


# ------------------------------------------------- stored points are invalid
def test_stored_high_i_points_violate_constant_stress():
    """Round-3 documents itself as a constant-stress sweep but the platen stress
    drifts |err| > 5 % for every stored point with I_target >= 0.1."""
    rows = _stored_rows()
    checked = 0
    for (arm, I), row in rows.items():
        err = row["normal_stress_pa"] / rs.P - 1.0
        if I >= 0.1:
            checked += 1
            assert abs(err) > TOL, (
                f"{arm} I={I}: pressure error {err:+.3f} inside tolerance -- "
                "the round-3 point would be valid, re-audit this finding")
    assert checked >= 4, f"only {checked} high-I stored points found"
    # the two extremes of the sweep, both at I = 0.3
    assert rows[("baseline", 0.3)]["normal_stress_pa"] / rs.P - 1.0 == pytest.approx(-0.165, abs=5e-3)
    assert rows[("thick", 0.3)]["normal_stress_pa"] / rs.P - 1.0 == pytest.approx(+0.578, abs=5e-3)


# ------------------------------------------------------- gate reproduces/fixes
def test_gate_off_reproduces_round3_artifacts_bit_for_bit():
    """Gate off == rounds 1-3 protocol, so the 'before' numbers are live, not
    stale artifacts (a stale-artifact explanation is thereby excluded)."""
    rows = [r for r in _gate_rows() if not r["gate"]]
    if not rows:
        pytest.skip("no gate-off reproduction run recorded")
    stored = _stored_rows()
    for r in rows:
        ref = stored[(r["arm"], round(r["I_target"], 6))]
        assert r["mu"] == pytest.approx(ref["mu_measured"], rel=1e-9)
        assert r["phi"] == pytest.approx(ref["phi_bed_steady"], rel=1e-9)
        assert r["recover_steps"] == 0


def test_gate_on_holds_pressure_and_flags_validity():
    rows = [r for r in _gate_rows() if r["gate"]]
    if not rows:
        pytest.skip("no gate-on run recorded")
    for r in rows:
        # the verdict must be consistent with the final-window error
        assert r["servo_valid"] == (abs(r["pressure_control_error_final"]) <= TOL), r
        if r["servo_valid"]:
            assert abs(r["pressure_control_error_final"]) <= TOL
    valid = [r for r in rows if r["servo_valid"]]
    assert valid, "the gate never achieved constant stress -- report honestly"


def test_gate_is_noop_when_the_servo_already_holds():
    """Low I is already valid: the gate must not invent recovery steps there."""
    rows = [r for r in _gate_rows() if r["gate"] and r["arm"] == "baseline"
            and abs(r["I_target"] - 0.01) < 1e-9]
    if not rows:
        pytest.skip("no low-I gate run recorded")
    r = rows[0]
    assert r["servo_valid"] and abs(r["pressure_control_error_final"]) <= TOL
    assert r["strain_achieved"] < 0.5, "gate wandered far past the strain budget"


def test_gate_changes_mu_by_more_than_the_model_error_it_was_judging():
    """The control violation is not cosmetic: enforcing constant stress at
    (baseline, I=0.3) moves mu by far more than the 10 % acceptance band."""
    off = [r for r in _gate_rows() if not r["gate"] and r["arm"] == "baseline"
           and abs(r["I_target"] - 0.3) < 1e-9]
    on = [r for r in _gate_rows() if r["gate"] and r["arm"] == "baseline"
          and abs(r["I_target"] - 0.3) < 1e-9]
    if not (off and on):
        pytest.skip("no (baseline, I=0.3) gate pair recorded")
    shift = abs(on[0]["mu_final"] - off[0]["mu_final"]) / off[0]["mu_final"]
    assert shift > 0.10, f"mu shift only {shift:.1%} -- finding overstated"


# ------------------------------------------------------- reference provenance
def test_phi_reference_slope_contradicts_dilatancy():
    """The in-repo phi(I) reference *increases* with I (faster shear -> denser
    bed), while the device dilates in every round-3 arm: phi_bed_steady
    decreases with I.  This documents that the case's phi criterion is
    mis-specified / provenance-unverified, so agreement with it is not
    evidence of correctness."""
    assert rs.ref_phi(0.3) > rs.ref_phi(1e-3)      # reference: densifies
    rows = _stored_rows()
    for arm in ("baseline", "thick"):
        pts = sorted([(I, r["phi_bed_steady"]) for (a, I), r in rows.items() if a == arm])
        assert len(pts) >= 3, arm
        assert pts[-1][1] < pts[0][1], (
            f"{arm}: phi_bed_steady does not decrease with I ({pts}) -- the "
            "dilatancy contradiction claim would be wrong")


def test_gate_on_bed_is_not_stationary_at_high_i():
    """Honest limitation, pinned so it cannot be forgotten: with the pressure
    enforced (gate on) at baseline I=0.3 the bed keeps compacting over the
    20000-step recovery, so phi_final ends up *above* the low-I value: the
    measured phi is not a steady-state value at high I even though the stress
    is valid.  Any phi(I) claim in that range stays WITHHELD."""
    rows = [r for r in _gate_rows() if r["gate"] and r["arm"] == "baseline"]
    lo = [r for r in rows if abs(r["I_target"] - 0.01) < 1e-9]
    hi = [r for r in rows if abs(r["I_target"] - 0.3) < 1e-9]
    if not (lo and hi):
        pytest.skip("no baseline low/high-I gate pair recorded")
    assert hi[0]["recover_steps"] == lo[0]["recover_steps"] * 0 + 20000 or \
        hi[0]["recover_steps"] > lo[0]["recover_steps"]
    assert hi[0]["phi_final"] > lo[0]["phi_final"], (
        "bed no longer compacts under the enforced stress -- re-check the "
        "stationarity limitation")
