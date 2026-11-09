"""Tests for :mod:`tensordem.mts_contact` -- the multi-rate contact contract.

The suite pins the pieces a reviewer can recompute without trusting the solver:

* the closed-form contact time and the gate verdicts (pure arithmetic),
* the Hunt-Crossley restitution *formula* against the ledger implementation,
  and the *measured* restitution against the independent 1-DOF velocity-Verlet
  reference (the damping level itself is calibrated on that reference, never on
  the integrator under test),
* the sign of the dissipative term (a wrong sign adds energy: the regression is
  the ``e > 1`` blow-up found when this module was written),
* the shift-sign consistency of the periodic pair list (a wrong sign turns a
  neighbour image into a spurious far force),
* the structural conservation statements (momentum, shard partition) and the
  short-horizon energy drift,
* the jamming protocol against its own legacy arm on a small system.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem import mts_contact as mc
from tensordem.contact_timestep_ledger import (
    hunt_crossley_semi_analytic_restitution,
    hertz_contact_time_weber_stillinger,
)
from tensordem.granular2d.stress import stress_invariants, virial_tensor

RADIUS = 0.025
DENSITY = 917.0
YOUNG = 1.0e9
POISSON = 0.3
V_IMPACT = 0.05


def _impact_config(**kwargs):
    m = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
    k_h = mc.hertz_stiffness_plane(YOUNG, POISSON, RADIUS)
    cfg = dict(kn=k_h, contact_model="hertz", damping="none", mu=0.0,
               density=DENSITY,
               thickness=m / (DENSITY * math.pi * RADIUS ** 2),
               skin=0.5 * RADIUS)
    cfg.update(kwargs)
    return cfg, m, k_h


def _single_impact(cfg, v_impact=V_IMPACT, wall=True):
    wall_arr = np.array([[[-0.2, 0.0], [0.2, 0.0]]]) if wall else None
    system = mc.MTSSystem(np.array([[0.0, RADIUS]]), np.array([RADIUS]),
                          mc.MTSConfig(**cfg), walls=wall_arr)
    system.vel[0, 1] = -v_impact
    return system


# --------------------------------------------------------------------- closed forms
def test_hertz_prefactor_is_the_textbook_294():
    # I = (2/5) B(2/5, 1/2)  =>  C = 2 I = 2.943275...
    # (the value "2.9427" quoted in ``hertz_impact_time`` is 0.02 % low: the
    # quadrature and the Beta function agree on 2.943275 to 1e-13)
    from scipy.special import beta as _beta
    assert mc.HERTZ_TIME_PREFACTOR == pytest.approx(2.0 * 0.4 * float(_beta(0.4, 0.5)),
                                                    rel=1e-12)
    assert mc.HERTZ_TIME_PREFACTOR == pytest.approx(2.9433, abs=5e-4)


def test_contact_time_matches_the_ledger_formula():
    m = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
    k_h = mc.hertz_stiffness_plane(YOUNG, POISSON, RADIUS)
    mine = mc.hertz_contact_time(m, k_h, V_IMPACT)
    theirs = hertz_contact_time_weber_stillinger(V_IMPACT, m, k_h)
    assert mine == pytest.approx(theirs, rel=1e-3)


def test_hertz_plane_stiffness_is_the_rigid_platen_convention():
    # rigid platen: 1/E* = (1-nu^2)/E and R* = R, so the two-body prefactor is
    # smaller by 2 (modulus) * sqrt(2) (radius) = 2^{3/2}
    k_plane = mc.hertz_stiffness_plane(YOUNG, POISSON, RADIUS)
    k_pair = mc.hertz_stiffness_pair(YOUNG, POISSON, RADIUS, RADIUS)
    assert k_pair == pytest.approx(k_plane * 2.0 ** -1.5, rel=1e-12)
    # ... and the contact time, t_c ~ (m/k)^{2/5}, by 2^{3/5}
    m = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
    assert (mc.hertz_contact_time(m, k_pair, V_IMPACT)
            / mc.hertz_contact_time(m, k_plane, V_IMPACT)
            == pytest.approx(2.0 ** 0.6, rel=1e-12))


def test_linear_contact_time_is_half_a_period():
    assert mc.linear_contact_time(0.5, 200.0) == pytest.approx(
        math.pi * math.sqrt(0.5 / 200.0), rel=1e-15)
    with pytest.raises(ValueError):
        mc.linear_contact_time(0.0, 1.0)


@pytest.mark.parametrize("ratio,status", [(0.02, "ok"), (0.2, "ok"),
                                          (0.21, "tight"), (0.5, "tight"),
                                          (0.51, "violated"), (2.0, "violated")])
def test_gate_verdicts(ratio, status):
    gate = mc.stability_gate(ratio, 1.0)
    assert gate["status"] == status
    assert gate["ratio"] == pytest.approx(ratio)
    assert gate["n_sub_required"] == max(1, math.ceil(ratio / mc.GATE_MAX_RATIO - 1e-12))


def test_gate_rejects_nonpositive_inputs():
    with pytest.raises(ValueError):
        mc.stability_gate(0.0, 1.0)
    with pytest.raises(ValueError):
        mc.stability_gate(1.0, -1.0)


# ------------------------------------------------------------------- restitution
def test_hc_formula_matches_the_ledger_implementation():
    m = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
    k_h = mc.hertz_stiffness_plane(YOUNG, POISSON, RADIUS)
    alpha = _hc_alpha()
    mine = mc.hunt_crossley_restitution(V_IMPACT, m, k_h, alpha)
    theirs = hunt_crossley_semi_analytic_restitution(V_IMPACT, m, k_h, alpha)
    assert mine == pytest.approx(theirs, rel=5e-3)


def _hc_alpha():
    m = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
    k_h = mc.hertz_stiffness_plane(YOUNG, POISSON, RADIUS)
    return mc.alpha_for_restitution(0.9, V_IMPACT, m, k_h)


def test_alpha_calibration_reaches_the_target_on_the_reference():
    m = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
    k_h = mc.hertz_stiffness_plane(YOUNG, POISSON, RADIUS)
    for target in (0.5, 0.9):
        alpha = mc.calibrate_alpha(target, V_IMPACT, m, k_h)
        measured = mc.reference_restitution(V_IMPACT, m, k_h, alpha)
        assert measured == pytest.approx(target, abs=2e-3)


def test_damping_sign_is_dissipative():
    """The dashpot must remove energy: a flipped sign gives ``e > 1``."""
    m = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
    k_h = mc.hertz_stiffness_plane(YOUNG, POISSON, RADIUS)
    alpha = mc.calibrate_alpha(0.9, V_IMPACT, m, k_h)
    cfg, _m, _k = _impact_config(damping="hunt_crossley", alpha=alpha,
                                 dt=0.02 * mc.hertz_contact_time(m, k_h, V_IMPACT),
                                 n_sub=1, scheme="mts", enforce_gate=False)
    system = _single_impact(cfg)
    e = _vs_wall(system, V_IMPACT)
    assert e < 1.0
    assert e > 0.5


def _vs_wall(system, v_impact):
    """Return ``|v_out|/v_in`` of the first rebound against the platen."""
    for _ in range(4000):
        system.step(1)
        if system.vel[0, 1] > 0.0 and system.pos[0, 1] > RADIUS:
            return float(system.vel[0, 1]) / v_impact
    raise AssertionError("no rebound")


def test_restitution_ladder_is_within_two_percent():
    ladder = mc.restitution_ladder(e_targets=(0.5, 0.9), ratios=(0.02, 0.2))
    assert ladder["within_2pct"]
    assert ladder["worst_abs_restitution_error"] <= 0.02
    for row in ladder["rows"]:
        assert row["gate"]["status"] == "ok"
        assert row["h_over_tc"] <= mc.GATE_MAX_RATIO + 1e-12


def test_negative_control_without_subcycling_fails():
    """``dt/t_c = 0.5`` with one sub-step must be detectably wrong (gate tightness)."""
    ref = mc.measure_restitution(ratio=0.002, e_target=0.9, n_sub=1)
    bad = mc.measure_restitution(ratio=0.5, e_target=0.9, n_sub=1,
                                 scheme="mts_overstep", allow_tight=True)
    error = abs(bad["restitution_measured"] / ref["restitution_measured"] - 1.0)
    assert error > 0.10
    with pytest.raises(ValueError):
        mc.measure_restitution(ratio=0.5, e_target=0.9, n_sub=1,
                               scheme="mts_overstep")


# ------------------------------------------------------- periodic pair convention
def test_periodic_image_shift_sign_is_consistent():
    """A pair straddling the periodic border must be repelled, not attracted."""
    radius = 0.01
    box = 0.10
    n = 4
    density = 1000.0
    mass = density * (4.0 / 3.0) * math.pi * radius ** 3
    cfg = mc.MTSConfig(kn=1.0e4, contact_model="linear", damping="none", mu=0.0,
                       density=density,
                       thickness=mass / (density * math.pi * radius ** 2),
                       dt=1.0e-5, n_sub=1, scheme="mts", skin=0.5 * radius,
                       enforce_gate=False)
    # two disks, one near x=0 and one near x=Lx, i.e. touching across the border
    pos = np.array([[0.001, 0.05], [0.099, 0.05], [0.05, 0.001], [0.05, 0.099]])
    system = mc.MTSSystem(pos, np.full(n, radius), cfg, periodic=(box, box))
    system._slow_pass()
    i_pairs, _j, shift, _key = system._pairs_disk(system.pos)
    # the border pair must be registered with the *image offset* -Lx (a body near
    # the right edge seen from one near the left edge sits one period to the left)
    assert np.any((i_pairs == 0) & (np.isclose(shift[:, 0], -box)))
    force, _torque = system._forces(0.0, passes=1)
    # periodicity: disk 0 (x=0.001) sits *right* of the image of disk 1
    # (x=-0.001), so it is pushed to +x while disk 1 is pushed to -x.  A flipped
    # shift sign would swap these (and would also put the pair 0.098 apart, i.e.
    # make the contact disappear).
    assert force[0, 0] > 0.0
    assert force[1, 0] == pytest.approx(-force[0, 0], rel=1e-12)
    assert np.max(np.abs(force)) < 1.0e3  # no spurious far force


def test_contacts_dict_feeds_the_shipped_virial():
    """``MTSSystem.contacts`` is layout-compatible with the shipped stress code."""
    system, _box = mc._gas_system(0.8, 32, 0.2, n_sub=2, scheme="mts", seed=1,
                                  shear_rate=1.0, mu=0.5, kn=1.0e5)
    system._slow_pass()
    system._forces(system.dt / 2, passes=1)
    area = system.domain_area()
    shipped = stress_invariants(virial_tensor(system, area=area))["pressure"]
    mine = mc.pressure_and_shear(system, area=area)["pressure"]
    assert shipped == pytest.approx(mine, rel=1e-12)


# ------------------------------------------------------------------- conservation
def test_momentum_is_conserved_structurally():
    ledger = mc.gas_ledger(n=16, phi=0.6, ratio=0.2, n_sub=2, steps=300,
                           confirm_every=100)
    assert ledger["momentum_drift_rel"] <= 1e-12
    assert ledger["momentum_within_1e12"]


def test_energy_drift_over_a_short_horizon():
    ledger = mc.gas_ledger(n=16, phi=0.6, ratio=0.2, n_sub=4, steps=1500,
                           confirm_every=100)
    assert ledger["energy_max_abs_drift_rel"] <= 0.01, ledger


def test_overstep_negative_control_blows_up():
    ledger = mc.gas_ledger(n=16, phi=0.6, ratio=0.5, n_sub=1, steps=400,
                           confirm_every=50, scheme="mts_overstep",
                           enforce_gate=False)
    assert ledger["energy_max_abs_drift_rel"] > 0.10


# ---------------------------------------------------------------------- sharding
def test_shard_partitions_only_perturb_roundoff():
    check = mc.shard_force_consistency(n=32, shard_counts=(1, 2, 4))
    assert check["worst_rel_force_diff"] <= 1e-12
    a = mc.run_shards(shards=1, n=32, steps=20)
    b = mc.run_shards(shards=4, n=32, steps=20)
    dv = float(np.max(np.linalg.norm(b["vel"] - a["vel"], axis=1)))
    scale = max(float(np.max(np.linalg.norm(a["vel"], axis=1))), 1e-300)
    assert dv / scale <= 1e-8


# ------------------------------------------------------------------ shear + jamming
def test_shear_cell_stress_agrees_between_the_friction_arms():
    """The two friction arms must produce the same steady cell stress.

    A 32-disk cell is a small statistics problem: the pressure (a sum of ~200
    contact forces) agrees to well under 2 %, while the *deviatoric* stress is a
    near-cancelling difference and is only pinned to 25 % here -- the production
    comparison (64 disks, 250 outer steps) is in the acceptance run, where it
    agrees to 1.6 %.
    """
    kw = dict(phi=0.86, mu=0.5, n=32, ratio=0.2, steps=200, relax_steps=1200)
    mts = mc.shear_cell_run(scheme="mts", **kw)
    legacy = mc.shear_cell_run(scheme="legacy3x", **kw)
    assert mts["pressure"] > 0.0 and legacy["pressure"] > 0.0
    assert abs(mts["pressure"] / legacy["pressure"] - 1.0) <= 0.02
    assert abs(mts["shear"] / legacy["shear"] - 1.0) <= 0.25
    assert abs(mts["tau_over_p"]) <= 0.5  # cannot exceed the Coulomb coefficient


def test_jamming_protocol_arms_agree_on_a_small_system():
    kw = dict(n=64, mean_radius=0.008, compress_time=0.01, level_time=0.02)
    mts = mc.jamming_protocol(scheme="mts", **kw)
    legacy = mc.jamming_protocol(scheme="legacy3x", **kw)
    assert 0.5 < mts["phi_j"] < 1.0
    # These stages (0.01 / 0.02 s) are deliberately far too short to converge the
    # pressure servo, so only a loose band is asserted here; the production run
    # (``benchmarks/mts_contact_gate.py --only jamming``, 0.06 s stages) is where
    # the 0.5 % agreement is required and reported.
    assert abs(mts["phi_j"] / legacy["phi_j"] - 1.0) <= 0.15
    assert mts["gate"]["status"] == "ok"
    assert legacy["gate"]["status"] == "ok"
    # cost accounting: the legacy arm evaluates the whole contact kernel three
    # times per step at the fine step, so on the same physical protocol it pays
    # several times the MTS arm's kernel count (measured ratio ~2.7x here)
    assert legacy["return_mapping_passes"] == legacy["kinematics_calls"]
    assert legacy["kinematics_calls"] >= 2.5 * mts["kinematics_calls"]


def test_phi_j_extrapolates_a_known_linear_series():
    # exact linear EOS: phi = 0.786 + 1e-4 P  =>  the P -> 0 intercept is exact
    series = [[30.0, 0.789, 30.0], [10.0, 0.787, 10.0],
              [3.0, 0.7863, 3.0], [1.0, 0.7861, 1.0]]
    result = mc.phi_j_from_levels(series)
    assert result["phi_j"] == pytest.approx(0.786, abs=1e-9)
    assert result["dphi_dp"] == pytest.approx(1e-4, rel=1e-9)
    with pytest.raises(ValueError):
        mc.phi_j_from_levels([[1.0, 0.5]])
