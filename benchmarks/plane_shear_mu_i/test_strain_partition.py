"""m21528: the whole-gap strain partition must tell a flat interior apart
from a static bed.

These tests are synthetic (analytic velocity fields, no simulation), so they
pin the *estimator* semantics that round 21 of CASE-DEM-PLANE-SHEAR-MUIDI rests
on.  A regression here silently moves the mu(I) abscissa, so each estimator is
checked against a field whose true shear rate is known in closed form.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensordem.mu_i_rheology import (  # noqa: E402
    ServoShearCase, shear_rate_profile, simulate_servo_shear, strain_partition,
)

RNG_SEED = 20261115
N = 6000
GAP = 0.09
Y_LO, Y_HI = 0.2 * 0.08, 0.8 * 0.08
GAMMA = 1.5          # 1/s, the I = 3e-3 imposed rate of the round-20 device


def _slab(rng, n=N):
    """A uniform random slab filling the gap (kinematics only, no contacts)."""
    pos = np.zeros((n, 3))
    pos[:, 0] = rng.uniform(0.0, 0.16, n)
    pos[:, 1] = rng.uniform(Y_LO, Y_HI, n)
    pos[:, 2] = rng.uniform(0.0, GAP, n)
    return pos


def _linear(pos):
    """Couette flow, no-slip at both walls: v_x = gamma z."""
    vel = np.zeros_like(pos)
    vel[:, 0] = GAMMA * pos[:, 2]
    return vel


def _localised(pos):
    """Deformation confined to the outer quarter of the gap at each wall.

    ``phi = v_x / v_platen`` rises linearly from 0 at the floor to 0.5 at
    ``z = 0.25 gap``, is a *plug* (phi = 0.5) across the whole interior window
    ``0.25 - 0.75 gap``, and rises to 1 at the platen.  The central 50 % window
    that ``shear_rate_measured`` fits is therefore exactly flat, which is the
    state round 20 reported as "the bed is not shearing" - while the bed is in
    fact shearing at the imposed rate, only inside the wall layers.
    """
    z = pos[:, 2] / GAP
    phi = np.where(z <= 0.25, 0.5 * z / 0.25, 0.5)
    phi = np.where(z > 0.75, 0.5 + 0.5 * (z - 0.75) / 0.25, phi)
    vel = np.zeros_like(pos)
    vel[:, 0] = GAMMA * GAP * phi
    return vel


def _static(pos):
    return np.zeros_like(pos)


# --------------------------------------------------------------------------
# 1. Couette reference: every estimator must recover the imposed rate
# --------------------------------------------------------------------------
def test_linear_profile_all_estimators_recover_the_rate():
    rng = np.random.default_rng(RNG_SEED)
    pos = _slab(rng)
    vel = _linear(pos)
    part = strain_partition(pos, vel, plate_vx=GAMMA * GAP, gap=GAP,
                            y_lo=Y_LO, y_hi=Y_HI, n_bins=9)
    assert part["n_bins"] >= 8
    for key in ("shear_rate_full", "shear_rate_adjacent", "shear_rate_interior"):
        assert abs(part[key] / GAMMA - 1.0) < 0.03, (key, part[key])
    # a linear profile has the same gradient inside and outside the window
    assert abs(part["interior_strain_fraction"] - 1.0) < 0.05
    assert abs(part["wall_strain_fraction"]) < 0.05
    # no-slip at both walls: the top bin is at ~gap - half a bin, not at gap
    assert part["v_bot_over_platen"] < 0.1
    assert 0.85 < part["v_top_over_platen"] < 1.0
    assert abs(part["q_adjacent"] - 1.0) < 0.2
    assert part["wall_strain_fraction"] == pytest.approx(
        1.0 - part["interior_strain_fraction"])


# --------------------------------------------------------------------------
# 2. Localised shear: the legacy interior estimator is blind, the whole-gap
#    ones are not.  This is the discriminator round 20 lacked.
# --------------------------------------------------------------------------
def test_localised_shear_is_invisible_to_the_interior_estimator():
    rng = np.random.default_rng(RNG_SEED)
    pos = _slab(rng)
    vel = _localised(pos)
    part = strain_partition(pos, vel, plate_vx=GAMMA * GAP, gap=GAP,
                            y_lo=Y_LO, y_hi=Y_HI, n_bins=9)
    legacy = shear_rate_profile(pos, vel, z_lo=0.25 * GAP, z_hi=0.75 * GAP,
                                y_lo=Y_LO, y_hi=Y_HI, n_bins=6)
    # What separates "localised" from "not shearing" is that the bed is moving
    # at the walls: the extreme-bin velocity ratio is of order 1 ...
    assert part["v_top_over_platen"] > 0.8
    # ... while the interior fit sees a flat plug and reports ~zero rate ...
    assert abs(legacy["shear_rate"]) < 0.15 * GAMMA, legacy["shear_rate"]
    assert abs(part["q_interior"]) < 0.2 * GAMMA
    # ... and the wall-to-wall difference is recovered by the difference
    # estimator, so the point is quotable on the whole-gap abscissa.
    assert part["q_adjacent"] > 0.6
    # The index is a coarse discriminator, not an exact gradient: on a 9-bin
    # grid with the kink exactly on the window edge it reads ~0.2 for a plug
    # interior instead of 0.  It still separates the two worlds by a factor 4.
    assert part["interior_strain_fraction"] < 0.35
    assert part["wall_strain_fraction"] > 0.65
    # the whole-gap LSQ carries more strain than the interior fit
    assert abs(part["shear_rate_full"]) > abs(legacy["shear_rate"])


def test_localised_shear_is_flagged_not_quotable():
    """The round-21 decision rule must call this LOCALISED, not 'not shearing'."""
    sys.path.insert(0, str(REPO / "benchmarks" / "plane_shear_mu_i"))
    from low_i_profile_audit import classify
    row = {"q_interior": 0.02, "q_adjacent": 0.98, "q_full": 0.55,
           "v_top_over_platen": 0.96, "interior_strain_fraction": 0.05,
           "mu_wall": 0.23}
    verdict = classify(row)
    assert verdict["verdict"] == "LOCALISED", verdict
    by_name = {c["name"]: c["pass"] for c in verdict["checks"]}
    assert by_name["localised_shear"] is True
    assert by_name["wall_slip"] is False


def test_round20_signature_is_reproduced_by_the_rule():
    """The literal round-20 numbers: q~0 with the platen sticking (mu_wall<mu_c).

    Whatever the underlying state, the rule must not certify this as a quotable
    steady-state point; the only admissible readings are LOCALISED or
    UNDETERMINED, never QUOTABLE.
    """
    sys.path.insert(0, str(REPO / "benchmarks" / "plane_shear_mu_i"))
    from low_i_profile_audit import classify
    verdict = classify({"q_interior": -0.010, "q_adjacent": None, "q_full": None,
                        "v_top_over_platen": None,
                        "interior_strain_fraction": None, "mu_wall": 0.2299})
    assert verdict["verdict"] != "QUOTABLE", verdict


# --------------------------------------------------------------------------
# 3. Static bed with a moving platen = wall slip; must be distinguishable
# --------------------------------------------------------------------------
def test_static_bed_driven_wall_is_slip_not_localisation():
    rng = np.random.default_rng(RNG_SEED)
    pos = _slab(rng)
    vel = _static(pos)
    part = strain_partition(pos, vel, plate_vx=GAMMA * GAP, gap=GAP,
                            y_lo=Y_LO, y_hi=Y_HI, n_bins=9)
    assert abs(part["shear_rate_full"]) < 1e-9
    assert abs(part["v_top_over_platen"]) < 1e-9
    assert abs(part["q_full"]) < 1e-9
    # degenerate profile: no velocity difference to partition
    assert math.isnan(part["interior_strain_fraction"])
    sys.path.insert(0, str(REPO / "benchmarks" / "plane_shear_mu_i"))
    from low_i_profile_audit import classify
    assert classify({"q_full": 0.0, "v_top_over_platen": 0.0,
                     "q_interior": 0.0, "q_adjacent": 0.0,
                     "interior_strain_fraction": float("nan"),
                     "mu_wall": 0.5})["verdict"] == "SLIP"


def test_static_bed_with_transmitted_wall_velocity_is_inconsistent():
    """Flat profile + platen velocity transmitted would violate no-slip."""
    sys.path.insert(0, str(REPO / "benchmarks" / "plane_shear_mu_i"))
    from low_i_profile_audit import classify
    verdict = classify({"q_full": 0.0, "v_top_over_platen": 0.95,
                        "q_interior": 0.0, "q_adjacent": 0.0,
                        "interior_strain_fraction": 0.02, "mu_wall": 0.23})
    assert verdict["verdict"] in ("INCONSISTENT", "LOCALISED"), verdict


# --------------------------------------------------------------------------
# 4. The dead ``slip_top`` key is now defined, and only when it can be
# --------------------------------------------------------------------------
def test_slip_top_is_defined_when_the_platen_kinematics_are_given():
    rng = np.random.default_rng(RNG_SEED)
    pos = _slab(rng)
    vel = _linear(pos)
    with_pv = shear_rate_profile(pos, vel, z_lo=0.25 * GAP, z_hi=0.75 * GAP,
                                 y_lo=Y_LO, y_hi=Y_HI, n_bins=6,
                                 plate_vx=GAMMA * GAP, gap=GAP)
    without = shear_rate_profile(pos, vel, z_lo=0.25 * GAP, z_hi=0.75 * GAP,
                                 y_lo=Y_LO, y_hi=Y_HI, n_bins=6)
    assert math.isfinite(with_pv["slip_top"])
    # linear no-slip profile -> the top bin sits on the linear expectation
    assert abs(with_pv["slip_top"]) < 0.05
    assert math.isnan(without["slip_top"])


# --------------------------------------------------------------------------
# 5. The diagnostic must not disturb the physics
# --------------------------------------------------------------------------
def test_profile_detail_flag_does_not_change_the_trajectory():
    """Diagnostic-only extension: same seed, same protocol, same stress history.

    If this ever fails, an archived filename can no longer be reproduced and
    every round-1..20 number is unreproducible.
    """
    base = dict(inertial_number=None, shear_rate=20.0, mu=0.5, mu_rolling=0.05,
                restitution=0.3, kn=6.25e4, normal_stress=6.25e4,
                shear_strain=0.02, settle_steps=200, min_shear_steps=1,
                sample_stride=5, packing_gap=0.02, jitter=0.01, seed=11,
                radius=0.005, density=2500.0, box_width=0.10, box_depth=0.05,
                fill_height=0.04, bulk_stress=False)
    off = simulate_servo_shear(ServoShearCase(profile_detail=False, **base))
    on = simulate_servo_shear(ServoShearCase(profile_detail=True, **base))
    for key in ("normal_stress_measured_pa", "shear_stress_pa",
                "mu_effective", "shear_strain_achieved", "n_samples",
                "plate_height", "shear_rate_measured"):
        a, b = off[key], on[key]
        if a is None or b is None or (
                isinstance(a, float) and isinstance(b, float)
                and math.isnan(a) and math.isnan(b)):
            continue                      # tiny cell: the window can be empty
        assert a == pytest.approx(b, rel=0, abs=0), key
    assert "gap_kinematics" not in off
    assert on["gap_kinematics"]["gap_q_adjacent"] is not None
    assert on["gap_profile_final"]["profile"], "per-bin profile not archived"