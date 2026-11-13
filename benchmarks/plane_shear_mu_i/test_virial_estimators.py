"""m17126 / round 8 verification: virial estimators against analytic anchors.

Three independent checks, none of which needs a literature constant:

1. ``bulk_virial_stress(rule="both")`` drops a contact whose endpoints straddle
   the slab, ``rule="midpoint"`` counts it exactly once, and both agree on the
   contact that lies wholly inside -- checked against the hand-computed
   ``sum f (x) r / V`` on a synthetic pair list.
2. ``full_cell_virial_stress`` reproduces the same sum over *all* pairs on a
   synthetic list (no selection, centre-accessible volume).
3. On a real (small, fast) servo-shear run the virial identity
   ``sigma_zz = -P_applied`` holds to a few percent: in static equilibrium
   ``sum_pairs f (x) (x_j - x_i) = -sum_i F_i^ext (x) x_i`` and the external
   loads act at ``z ~ r`` and ``z ~ h - r``, so the domain average normalised by
   the centre-accessible volume ``A (h - 2r)`` must reproduce the applied stress.
   This is the calibration the thin-slab estimator never had (rounds 6-7).

Usage: ``pytest benchmarks/plane_shear_mu_i/test_virial_estimators.py -q``
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensordem.mu_i_rheology import (  # noqa: E402
    ServoShearCase, bulk_virial_stress, full_cell_virial_stress,
    simulate_servo_shear)

W = 0.1
Z_LO, Z_HI = 0.004, 0.010
Y_LO, Y_HI = 0.02, 0.08
SLAB_VOL = W * (Y_HI - Y_LO) * (Z_HI - Z_LO)


def _synthetic_state():
    """p0-p1 straddles the slab (one endpoint in, one out); p2-p3 sits inside."""
    pos = np.array([[0.050, 0.050, 0.002],
                    [0.050, 0.050, 0.009],
                    [0.020, 0.050, 0.005],
                    [0.080, 0.050, 0.008]])
    vel = np.zeros((4, 3))
    pairs = {
        "iu": np.array([0, 2]),
        "ju": np.array([1, 3]),
        "branch": np.array([[0.0, 0.0, 0.007],
                            [0.0, 0.0, 0.003]]),
        "force_on_i": np.array([[0.3, 0.0, -2.0],
                                [0.1, 0.0, -1.0]]),
    }
    return pos, vel, pairs


def _manual(slices):
    pos, vel, pairs = _synthetic_state()
    acc = np.zeros((3, 3))
    for k in slices:
        acc += np.outer(pairs["force_on_i"][k], pairs["branch"][k])
    return acc / SLAB_VOL


def test_both_endpoints_rule_drops_the_straddling_contact():
    pos, vel, pairs = _synthetic_state()
    out = bulk_virial_stress(pos, vel, pairs, z_lo=Z_LO, z_hi=Z_HI,
                             y_lo=Y_LO, y_hi=Y_HI, box_width=W, n=4,
                             mass=1.0, rule="both")
    assert out["n_contacts"] == 1                      # only the p2-p3 contact
    assert out["n_contacts_straddle"] == 1             # p0-p1 crosses the slab
    assert out["straddle_fraction"] == pytest.approx(0.5)
    assert np.allclose(out["sigma"], _manual([1]), rtol=0.0, atol=1e-12)


def test_midpoint_rule_counts_the_straddling_contact_once():
    pos, vel, pairs = _synthetic_state()
    out = bulk_virial_stress(pos, vel, pairs, z_lo=Z_LO, z_hi=Z_HI,
                             y_lo=Y_LO, y_hi=Y_HI, box_width=W, n=4,
                             mass=1.0, rule="midpoint")
    assert out["n_contacts"] == 2
    # both contacts land in the sum exactly once -> no double counting, no loss
    assert np.allclose(out["sigma"], _manual([0, 1]), rtol=0.0, atol=1e-12)


def test_unknown_rule_is_rejected():
    pos, vel, pairs = _synthetic_state()
    with pytest.raises(ValueError):
        bulk_virial_stress(pos, vel, pairs, z_lo=Z_LO, z_hi=Z_HI,
                           y_lo=Y_LO, y_hi=Y_HI, box_width=W, n=4,
                           mass=1.0, rule="bogus")


def test_full_cell_estimator_sums_every_pair():
    pos, vel, pairs = _synthetic_state()
    height = Z_HI + 0.001
    radius = 0.001
    out = full_cell_virial_stress(pos, vel, pairs, box_width=W, box_depth=0.1,
                                  height=height, radius=radius, mass=1.0)
    vol = W * 0.1 * (height - 2.0 * radius)
    expected = np.zeros((3, 3))
    for k in range(2):
        expected += np.outer(pairs["force_on_i"][k], pairs["branch"][k])
    assert out["n_contacts"] == 2
    assert np.allclose(out["sigma"], expected / vol, rtol=0.0, atol=1e-12)
    assert out["kinetic_pressure"] == pytest.approx(0.0, abs=1e-12)


def test_full_cell_momentum_balance_on_a_real_cell():
    """sigma_zz (domain average) must equal -P_applied: the analytic anchor."""
    case = ServoShearCase(inertial_number=0.05, kn=6.25e4,
                          normal_stress=6.25e4, box_width=0.06,
                          box_depth=0.03, fill_height=0.035, radius=0.004,
                          settle_steps=600, min_shear_steps=300,
                          shear_strain=0.04, sample_stride=20, seed=5279,
                          bulk_stress=True)
    res = simulate_servo_shear(case)
    assert res["full_momentum_balance_final"] is not None
    balance = res["full_momentum_balance_final"]      # sigma_zz / (-P_applied)
    # exact value +1; the dynamic correction at I=0.05 is a few percent, while
    # the truncated slab estimator of rounds 6-7 sits at 0.5 ... 0.8.
    assert abs(1.0 - balance) <= 0.10, balance
    # and it is materially better calibrated than the both-endpoints slab value
    slab = res["bulk_pressure_pa"] / case.normal_stress - 1.0
    assert abs(1.0 - balance) <= abs(1.0 - (1.0 + slab)), (balance, slab)
    assert math.isfinite(res["full_mu_final"])
    assert res["full_kinetic_pressure_pa"] < 0.05 * case.normal_stress


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))