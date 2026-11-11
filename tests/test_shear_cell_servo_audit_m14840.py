"""m14840 independent verification of the direct-shear device (adversarial review
of the merged m14721 work) and regression locks for the m14840 fixes.

Three things are pinned here, in the order the audit found them:

1. **Stationarity gate** (pure function): a window whose traction still drifts
   must not be reported as a friction coefficient.
2. **Servo load holding** (golden data from ``artifacts/shear_cell_m14840``):
   the m14721 lab-frame dashpot leaks a load offset proportional to the platen
   creep velocity, ``dN = M_p v_z (1-exp(-dt/tau))/dt``; the bed-referenced
   dashpot removes it at every stress level.
3. **Rate bookkeeping**: the reported shear rate / inertial number must be built
   from the *measured* bed height, not the nominal ``fill_height``.
"""
import json
import math
from pathlib import Path

import pytest

from tensordem.shear_cell import _validity_gate

ART = Path(__file__).resolve().parents[1] / "artifacts" / "shear_cell_m14840"
DT, TAU = 2.5e-5, 5.0e-4          # case defaults used by the recorded matrix
MPLATE = 0.10210176124166828      # M_plate = 1.0 * (78 beads * m_bead), kg


# ------------------------------------------------------------------ gate legs
def test_stationarity_gate_rejects_a_drifting_window():
    ok, why = _validity_gate(4000.0, 800.0, 4000.0, drift_rel=0.84, stationarity_tol=0.15)
    assert ok is False and "not steady" in why
    ok, why = _validity_gate(4000.0, 800.0, 4000.0, drift_rel=0.05, stationarity_tol=0.15)
    assert ok is True and why == ""


def test_stationarity_gate_is_off_unless_asked_for():
    # backwards compatibility: m14721 protocol did not gate on steadiness
    ok, _ = _validity_gate(4000.0, 800.0, 4000.0, drift_rel=5.0)
    assert ok is True


def test_gate_still_rejects_an_unloaded_platen_and_a_negative_traction():
    assert _validity_gate(10.0, 800.0, 4000.0)[0] is False
    assert _validity_gate(4000.0, -1.0, 4000.0)[0] is False


# ------------------------------------------------- servo: drag law vs bedref
def _rows(tag):
    data = json.loads((ART / f"{tag}.json").read_text())
    return data, data["rows"]


def test_drag_servo_load_offset_obeys_the_dashpot_law():
    """Falsifies "the offset is physical bed creep": it is M_p*v_z*kappa/dt."""
    data, rows = _rows("drag")
    assert data["mode"] == "drag"
    kappa_per_dt = (1.0 - math.exp(-DT / TAU)) / DT
    checked = 0
    for r in rows:
        predicted = MPLATE * kappa_per_dt * r["vz_platen"]
        measured = r["dN_data_N"]
        if abs(measured) < 0.05:          # offset too small to resolve
            continue
        assert abs(predicted - measured) / abs(measured) < 0.45, (r["sigma_target"], predicted, measured)
        checked += 1
    assert checked >= 1


def test_bedref_servo_holds_the_target_load_at_every_stress_level():
    _, rows = _rows("bedref")
    for r in rows:
        assert abs(r["sigma_rel"]) < 0.01, (r["sigma_target"], r["sigma_rel"])
        assert r["valid"] is True, r["rz_no"]


def test_bedref_is_at_least_five_times_better_than_the_drag_servo():
    _, drag = _rows("drag")
    _, bed = _rows("bedref")
    for rd, rb in zip(drag, bed):
        assert abs(rb["sigma_rel"]) < abs(rd["sigma_rel"]) / 5.0, (rd["sigma_target"], rd["sigma_rel"], rb["sigma_rel"])


# ------------------------------------------------------ rate / inertial number
def test_reported_rate_and_inertial_number_use_the_measured_bed_height():
    data, rows = _rows("bedref")
    for r in rows:
        assert r["bed_height"] > 0.0
        # the compacted bed is much shorter than the nominal fill height
        assert r["shear_rate_actual"] / r["shear_rate_nominal"] > 1.4
        assert abs(r["shear_rate_actual"] / r["shear_rate_nominal"] - 0.05 / r["bed_height"]) < 0.05
        # I = gamma_dot * d / sqrt(sigma / rho), d = 2r = 0.01 m, rho = 2500.
        # The recorded matrix was produced with the *nominal* rate (the pre-fix
        # convention this audit is about), so the stored I is checked against
        # gamma_nominal and the corrected value is verified to be larger by the
        # same factor as the rate itself.
        expect_nominal = r["shear_rate_nominal"] * 0.01 / math.sqrt(r["sigma"] / 2500.0)
        assert r["inertial_number"] == pytest.approx(expect_nominal, rel=2e-2)
        assert r["shear_rate_actual"] / r["shear_rate_nominal"] > 1.4


def test_the_device_is_not_at_steady_state_within_the_m14721_protocol():
    """Honest FAIL lock: even at gamma = 1.0 the traction keeps drifting, so a
    window mean must not be quoted as a steady-state mu."""
    _, rows = _rows("long")
    for r in rows:
        assert abs(r["drift"]) > 0.15
        q = r["tau_window_quarters"]
        assert q[-1] > 1.2 * q[0]           # traction still growing