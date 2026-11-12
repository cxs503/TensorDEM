"""Analytic anchors and invariants for the gravity-deposition packing kernel.

The crate's merged CASE-DEM-RCP-PACKING-FRACTION measurement is a tautology
(``make_random_close_packing`` returns ``phi = 1 - target_porosity``), so the
tests here are split the same way as the m16308 audit: a negative control that
pins the tautology down, and analytic anchors for every piece of the new
deposition machinery (wall law, friction/rolling transition, ballistic phase,
profile estimator, periodic neighbour search, energy sign).
"""

import math

import numpy as np
import pytest

from tensordem.bedcoupling.packing import make_random_close_packing
from tensordem.deposition_packing import (
    DepositionSpec,
    initial_configuration,
    measure_packing,
    packing_profile,
    run_deposition,
)

R = 0.5
D = 1.0
MASS = 4.0 / 3.0 * math.pi * R ** 3   # density 1


# ---------------------------------------------------------------------------
# negative control: the merged gate cannot fail
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [0.30, 0.36, 0.45])
def test_legacy_solid_fraction_is_the_input(eps):
    bed = make_random_close_packing(200, 1e-3, eps, seed=42,
                                    final_relax_iterations=20)
    assert abs(bed.solid_fraction - (1.0 - eps)) < 1e-12


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------
def test_static_rest_on_floor_is_the_analytic_deflection():
    """A sphere at rest on the floor sits at z = R - m g / k_n exactly."""
    kn = 4000.0
    spec = DepositionSpec(n_particles=1, kn=kn, settle_tol=0.0, t_max=3.0,
                          initial_positions=np.array([[3.0, 3.0, 0.5]]))
    res = run_deposition(spec)
    z_analytic = R - MASS * spec.gravity / kn
    assert abs(res.positions[0, 2] - z_analytic) < 1e-5
    assert abs(res.velocities[0, 2]) < 1e-3


def test_ballistic_phase_follows_half_g_t_squared():
    """Free fall (no contacts) reproduces z0 - g t^2 / 2 to integrator order."""
    spec = DepositionSpec(n_particles=1, settle_tol=0.0, t_max=1.0,
                          initial_positions=np.array([[3.0, 3.0, 40.0]]),
                          initial_velocity=(0.0, 0.0, 0.0))
    res = run_deposition(spec)
    z = 40.0 - 0.5 * spec.gravity * res.final_time ** 2
    assert abs(res.positions[0, 2] - z) / 40.0 < 1e-6


@pytest.mark.parametrize("mu", [0.0, 0.1, 0.3, 0.5])
def test_sliding_to_rolling_transition(mu):
    """Rigid sphere on a frictional floor: rolling sets in at v = (5/7) v0.

    Sliding: ``a = -mu g`` and ``alpha = mu m g R / I`` with ``I = 0.4 m R^2``;
    rolling starts when ``v = omega R``, which happens at
    ``t = v0 / (3.5 mu g)`` with ``v = (5/7) v0`` (classic result, independent
    of the mass and of ``R``).
    """
    v0 = 1.0
    t_stop = v0 / (3.5 * mu * 1.0) if mu > 0 else 4.0
    spec = DepositionSpec(n_particles=1, mu=0.0, mu_wall=mu, settle_tol=0.0,
                          t_max=4.0 * t_stop, sample_every=t_stop,
                          initial_positions=np.array([[3.0, 3.0, 0.5]]),
                          initial_velocity=(v0, 0.0, 0.0))
    res = run_deposition(spec)
    if mu == 0.0:
        # frictionless floor: no tangential force at all, so no spin either
        assert abs(res.velocities[0, 0] - v0) < 1e-9
        assert abs(res.omegas[0, 1]) < 1e-12
        return
    v_roll = res.velocities[0, 0]
    assert abs(v_roll - 5.0 / 7.0 * v0) < 0.03 * v0
    # rolling condition omega_y * R = v_x (rotation about -y for +x motion)
    assert abs(res.omegas[0, 1] * R - v_roll) < 0.05 * v0


def test_drop_dissipates_and_never_gains_energy():
    """A sphere dropped on the floor must lose mechanical energy at the impact.

    (The exact restitution is a *tuning* question for the dashpot; what the
    deposition measurement relies on is the sign of the dissipation, which is
    what this asserts -- and it is also the audit that caught the anti-damped
    tangential branch.)
    """
    for e in (0.3, 0.7):
        spec = DepositionSpec(n_particles=1, restitution=e, settle_tol=0.0,
                              t_max=6.0, sample_every=0.05,
                              initial_positions=np.array([[3.0, 3.0, 8.0]]))
        res = run_deposition(spec)
        tot = np.array(res.history["total_norm"])
        assert np.all(np.diff(tot) < 1e-6), "mechanical energy increased"
        assert tot[-1] < tot[0] - 1e-3, "no dissipation at the impact"


def test_periodic_neighbour_search_across_the_seam():
    """Two spheres straddling the periodic x seam are 0.4 d apart, not 0.6 L."""
    width = 10.0
    spec = DepositionSpec(n_particles=2, mu=0.0, settle_tol=0.0, t_max=0.2,
                          initial_positions=np.array([[0.1, 5.0, 6.0],
                                                      [width - 0.1, 5.0, 6.0]]))
    res = run_deposition(spec)
    assert res.metrics["n_contacts_total"] >= 2   # one pair, counted twice


def test_initial_placement_has_no_overlap():
    pos, info = initial_configuration(400, 7.0, 18.0, R, seed=5)
    from scipy.spatial import cKDTree
    tree = cKDTree(pos, boxsize=(7.0, 7.0, 1e6))
    pairs = tree.query_pairs(D, output_type="ndarray")
    assert len(pairs) == 0
    assert info["min_separation"] >= D


# ---------------------------------------------------------------------------
# profile estimator
# ---------------------------------------------------------------------------
def test_profile_of_an_fcc_lattice_is_the_packing_fraction():
    """Profile estimator on FCC: phi = pi / (3 sqrt 2) = 0.7405 in the bulk."""
    a = 2.0 ** 0.5            # nearest-neighbour distance in units of d = 1
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    n_cell = 5
    pos = []
    for i in range(n_cell):
        for j in range(n_cell):
            for k in range(20):
                pos.extend(basis + np.array([i, j, k]) * a)
    pos = np.array(pos)
    area = (n_cell * a) ** 2
    vol = np.full(len(pos), 4.0 / 3.0 * math.pi * R ** 3)
    prof = packing_profile(pos, vol, area, 0.5 * D, float(pos[:, 2].max() + R))
    band = (prof["z"] >= 2.5 * D) & (prof["z"] <= pos[:, 2].max() - 2.5 * D)
    phi = float(np.mean(prof["phi"][band]))
    assert abs(phi - math.pi / (3.0 * 2.0 ** 0.5)) < 0.02


def test_profile_matches_volume_balance():
    rng = np.random.default_rng(3)
    pos = rng.random((500, 3)) * np.array([6.0, 6.0, 10.0])
    vol = np.full(500, 4.0 / 3.0 * math.pi * R ** 3)
    prof = packing_profile(pos, vol, 36.0, 0.5, 10.0)
    total = float(np.sum(prof["phi"]) * 0.5 * 36.0)
    assert abs(total - 500 * 4.0 / 3.0 * math.pi * R ** 3) / (500 * 4.0 / 3.0 * math.pi * R ** 3) < 1e-10


# ---------------------------------------------------------------------------
# invariants of a small deposition
# ---------------------------------------------------------------------------
def test_small_deposition_invariants():
    spec = DepositionSpec(n_particles=400, mu=0.1, seed=11, t_max=70.0,
                          sample_every=2.0)
    res = run_deposition(spec)
    m = res.metrics
    # a deposited monodisperse bed must land in a physically plausible band
    assert 0.55 < m["phi_bulk"] < 0.70
    assert m["max_pair_overlap_over_d"] < 0.02
    # coordination of a dense 3-D packing: 5..8 (isostatic frictionless = 6)
    assert 4.5 < m["coordination_bulk"] < 8.0
    # the overlap correction is a documented ~1 % systematic, never the claim
    assert abs(m["phi_bulk"] - m["phi_bulk_overlap_corrected"]) < 0.02
    # dissipation: the total mechanical energy never increases by more than
    # the first-order tangential update can explain
    tot = np.array(res.history["total_norm"])
    assert np.all(np.diff(tot) < 5e-3)
    # the bed did compact (phi at the end above the placement value)
    assert res.history["phi_bulk_instant"][-1] > 3.0 * res.placement["placement_phi"]


def test_deposition_is_deterministic():
    def phi(seed):
        spec = DepositionSpec(n_particles=300, mu=0.0, seed=seed, t_max=40.0,
                              sample_every=5.0)
        return run_deposition(spec).metrics["phi_bulk"]
    assert abs(phi(3) - phi(3)) < 1e-12
    assert abs(phi(3) - phi(4)) < 0.03      # seed-to-seed scatter is small


def test_friction_lowers_the_packing_fraction():
    """Documented physics: inter-particle friction makes the packing looser."""
    def phi(mu):
        spec = DepositionSpec(n_particles=400, mu=mu, seed=6, t_max=70.0,
                              sample_every=5.0)
        return run_deposition(spec).metrics["phi_bulk"]
    assert phi(0.3) <= phi(0.0) + 0.005


# ---------------------------------------------------------------------------
# reading the soft-sphere number in the hard-sphere limit
# ---------------------------------------------------------------------------
from tensordem.deposition_packing import stiffness_limit  # noqa: E402


def test_stiffness_limit_recovers_an_exact_power_law():
    """phi = 0.634 + 3/k_n^2 over three levels -> the limit is exact."""
    limit_true, amp, power = 0.634, 3.0, 2.0
    pts = [(kn, limit_true + amp / kn ** power) for kn in (4000.0, 16000.0, 64000.0)]
    out = stiffness_limit(pts)
    assert out["order"] is not None
    assert abs(out["order"] - power) < 1e-6  # float cancellation in the deltas
    assert abs(out["limit"] - limit_true) < 1e-9
    # the residual of the stiffest measurement is what the extrapolation removes
    assert out["delta_from_last"] == pytest.approx(amp / 64000.0 ** power, rel=1e-6)


def test_stiffness_limit_two_levels_uses_the_default_order():
    out = stiffness_limit([(4000.0, 0.638), (16000.0, 0.635)])
    assert out["order"] is None and out["order_used"] == 1.0
    # value_inf = v + Delta/(r^p - 1) with Delta = -0.003 and r = 4
    assert out["limit"] == pytest.approx(0.635 + (-0.003) / 3.0)


def test_stiffness_limit_non_monotone_deltas_are_flagged_not_hidden():
    """Deltas that change sign have no order; the flag must say so."""
    out = stiffness_limit([(4000.0, 0.640), (16000.0, 0.634), (64000.0, 0.641)])
    assert out["order"] is None
    assert out["order_used"] == 1.0


@pytest.mark.parametrize("pts", [[(4000.0, 0.64)],
                                 [(4000.0, 0.64), (4000.0, 0.63)],
                                 [(0.0, 0.64), (4000.0, 0.63)]])
def test_stiffness_limit_rejects_degenerate_input(pts):
    with pytest.raises(ValueError):
        stiffness_limit(pts)


def test_deposition_plan_is_consistent_with_the_shard_runner():
    """The cluster shard indexes the plan file; a stale bound would drop cases."""
    import json
    import os
    plan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "benchmarks", "case_rcp_deposition", "plan.json")
    cases = json.load(open(plan_path))["cases"]
    assert len(cases) >= 12
    for c in cases:
        assert {"n", "mu", "seed", "kn", "role"} <= set(c)
    # production tier must carry the largest N and both friction tiers
    n_max = max(c["n"] for c in cases)
    tiers = {c["mu"] for c in cases if c["n"] == n_max and c["kn"] == 4000 and "dt_frac" not in c}
    assert {0.0, 0.1} <= tiers
    stiff_kn = {c["kn"] for c in cases if c["kn"] > 4000}
    assert len(stiff_kn) >= 2
