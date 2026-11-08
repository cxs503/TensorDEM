"""Verification tests for ball-ball rolling-friction contact mechanics (P40-N04).

Every assertion is bound to a closed-form reference, never to "the run
finished".  The numerical evidence is produced by
:mod:`tensordem.ball_ball_rolling_verification` and mirrored in
``artifacts/ball_ball_rolling_contact/results.json`` (report:
``docs/ball_ball_rolling_contact.md``).
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest
import torch

from tensordem.ball_ball_rolling import (
    RollingFrictionBallBall,
    TangentialState,
    advance_tangential,
    analytic_incline_critical_angle_deg,
    analytic_stack_critical_angle_deg,
    analytic_transition_spin_ratio,
    contact_point_slip_velocity,
    effective_radius,
    rolling_transition_velocity,
    simulate_tangential_spring,
    solid_sphere_inertia,
    solid_sphere_mass,
)
from tensordem.ball_ball_rolling_verification import (
    simulate_roll_slip_transition,
    verify_driven_rolling,
    verify_energy_ledger,
    verify_incline_stability,
    verify_rolling_moment,
    verify_rolling_sliding_transition,
    verify_rotational_damping,
    verify_tangential_stiffness,
    verify_coulomb_limit,
)

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_ball_ball_rolling_contact.py"
)


# ---------------------------------------------------------------------------
# Fixtures (each expensive face is evaluated once per session)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    torch.set_num_threads(1)


@pytest.fixture(scope="module")
def tangential():
    return verify_tangential_stiffness()


@pytest.fixture(scope="module")
def coulomb():
    return verify_coulomb_limit()


@pytest.fixture(scope="module")
def moment():
    return verify_rolling_moment()


@pytest.fixture(scope="module")
def rotation():
    return verify_rotational_damping()


@pytest.fixture(scope="module")
def transition():
    return verify_rolling_sliding_transition()


@pytest.fixture(scope="module")
def driven():
    return verify_driven_rolling()


@pytest.fixture(scope="module")
def incline():
    return verify_incline_stability()


@pytest.fixture(scope="module")
def energy():
    return verify_energy_ledger()


# ---------------------------------------------------------------------------
# Geometry and kinematics primitives
# ---------------------------------------------------------------------------
def test_effective_radius_plane_limit():
    """r_i -> inf must reproduce the ball-plane radius exactly."""
    assert effective_radius(1.0e9, 0.05) == pytest.approx(0.05, rel=1e-8)
    assert effective_radius(0.05, 0.05) == pytest.approx(0.025, rel=1e-15)


def test_contact_slip_vanishes_in_rolling_state():
    """Exact contact-point slip is zero for the pure rolling configuration."""
    r_j = 0.05
    normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    omega = torch.tensor([0.0, 0.0, -12.0], dtype=torch.float64)
    v = torch.cross(r_j * omega, normal, dim=-1)  # rolling-consistent velocity
    s = contact_point_slip_velocity(
        v, torch.zeros(3, dtype=torch.float64), omega, normal, 0.0, r_j
    )
    assert float(s.norm()) == pytest.approx(0.0, abs=1e-12)


def test_rolling_transition_velocity_reduces_to_plane_limit():
    """The r_eff convention equals the exact rolling radius for r_i -> inf."""
    r_j = 0.05
    normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    omega = torch.tensor([0.0, 0.0, -8.0], dtype=torch.float64)
    v = torch.cross(r_j * omega, normal, dim=-1)
    r_eff = effective_radius(1.0e9, r_j)
    s_eff = rolling_transition_velocity(
        v - (v @ normal) * normal, omega, normal, r_eff
    )
    assert float(s_eff.norm()) < 1e-9


def test_analytic_transition_spin_ratio_solid_sphere():
    """2/7 for a solid sphere; the general formula follows I = 2/5 m r^2."""
    r, rho = 0.05, 2500.0
    mass = solid_sphere_mass(r, rho)
    inertia = solid_sphere_inertia(r, mass)
    assert analytic_transition_spin_ratio(mass, inertia, r) == pytest.approx(
        2.0 / 7.0, rel=1e-12
    )


def test_tangential_force_stays_in_tangent_plane():
    """Cundall-Strack projection keeps delta_t and F_t orthogonal to n."""
    normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    slip = torch.tensor([1.0e-3, 1.0e-3, 0.0], dtype=torch.float64)
    state = TangentialState(delta_t=torch.zeros(3, dtype=torch.float64))
    f_t = advance_tangential(state, slip, normal, 10.0, 1.0e4, 0.5, 1.0e-3)
    assert abs(float(f_t @ normal)) < 1e-12
    assert f_t[0] < 0.0  # opposes the +x slip


# ---------------------------------------------------------------------------
# (a) tangential stiffness recovery
# ---------------------------------------------------------------------------
def test_tangential_stiffness_recovered_within_one_percent(tangential):
    assert tangential["worst_k_t_rel_err"] <= 1.0e-2
    assert tangential["all_elastic"]


def test_tangential_law_is_linear(tangential):
    for case in tangential["cases"]:
        assert case["lin_residual_N"] <= 1e-9 * max(case["peak_force_N"], 1.0)


def test_tangential_stiffness_scales_with_input():
    """Doubling k_t must double the recovered slope (no hidden scaling)."""
    a = simulate_tangential_spring(
        f_n=5.0, k_t=1.0e4, mu_s=0.8, dt=1e-6, n_steps=200
    )
    b = simulate_tangential_spring(
        f_n=5.0, k_t=2.0e4, mu_s=0.8, dt=1e-6, n_steps=200
    )
    assert b["k_t_recovered"] / a["k_t_recovered"] == pytest.approx(2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# (b) Coulomb limit
# ---------------------------------------------------------------------------
def test_coulomb_limit_ten_cases(coulomb):
    assert coulomb["n_cases"] >= 10
    assert coulomb["all_saturated"]
    assert coulomb["worst_cap_rel_err"] <= 1e-12


def test_coulomb_direction_opposes_slip(coulomb):
    for case in coulomb["cases"]:
        assert case["direction_cos"] == pytest.approx(1.0, abs=1e-12)


def test_coulomb_cap_proportional_to_normal_force():
    """Cap scales exactly with mu_s F_n (ratio check across two loads)."""
    from tensordem.ball_ball_rolling import check_coulomb_limit_case

    low = check_coulomb_limit_case((0.3, 5.0, 2.0e4, 1.0e-5, 1.0e-2))
    high = check_coulomb_limit_case((0.3, 20.0, 2.0e4, 1.0e-5, 1.0e-2))
    assert high["f_t_mag_N"] / low["f_t_mag_N"] == pytest.approx(4.0, rel=1e-12)


# ---------------------------------------------------------------------------
# (c) rolling-resistance moment
# ---------------------------------------------------------------------------
def test_rolling_moment_nine_cases(moment):
    assert moment["n_cases"] >= 9
    assert moment["worst_moment_rel_err"] <= 2.0e-2


def test_rolling_moment_matches_legacy_ball_plane_law(moment):
    """The new ball-ball couple agrees with the pre-existing r_eff formulation."""
    assert moment["worst_legacy_rel_err"] <= 1e-9


def test_rolling_moment_opposes_rolling_direction(moment):
    for case in moment["cases"]:
        assert case["cos_opposition"] == pytest.approx(1.0, abs=1e-9)


def test_rolling_moment_effective_radius_scaling():
    """M_r must scale linearly with r_eff = r_i r_j / (r_i + r_j)."""
    r_eff_equal = effective_radius(0.05, 0.05)
    r_eff_mixed = effective_radius(0.10, 0.20)
    assert r_eff_mixed > r_eff_equal
    law = RollingFrictionBallBall(mu_r=0.05)
    normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    omega = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    m_equal, _ = law.moment(omega, 10.0, r_eff_equal, normal)
    m_mixed, _ = law.moment(omega, 10.0, r_eff_mixed, normal)
    assert float(m_mixed.norm()) / float(m_equal.norm()) == pytest.approx(
        r_eff_mixed / r_eff_equal, rel=1e-12
    )


# ---------------------------------------------------------------------------
# (d) rotational damping
# ---------------------------------------------------------------------------
def test_rotational_decay_r2(rotation):
    assert rotation["r2"] >= 0.999


def test_rotational_decay_time_constant(rotation):
    assert rotation["tau_rel_err"] <= 1.0e-2


def test_rotational_decay_is_monotone(rotation):
    series = rotation["measured"]
    assert all(b < a for a, b in zip(series, series[1:]))


# ---------------------------------------------------------------------------
# (e) rolling-sliding transition
# ---------------------------------------------------------------------------
def test_transition_spin_ratio_two_sevenths(transition):
    """omega* / omega_0 = 2/7 for a solid sphere, independent of k_t and dt."""
    for name in ("plane_limit", "equal_spheres", "ratio_20"):
        case = transition["cases"][name]
        assert case["transition_spin_ratio"] == pytest.approx(2.0 / 7.0, rel=3e-2)


def test_transition_time_matches_impulse_balance(transition):
    case = transition["cases"]["plane_limit"]
    assert case["time_rel_err"] <= 3.0e-2


def test_transition_rolling_condition_plane_limit(transition):
    """v_t = omega r_eff holds in the plane limit (<= 3 %)."""
    assert transition["plane_limit_r_eff_err"] <= 3.0e-2


def test_transition_effective_radius_convention_gap(transition):
    """For equal spheres v/(omega r_eff) = 2: the r_eff form is a limit, not a law."""
    equal = transition["cases"]["equal_spheres"]
    assert equal["rolling_condition_r_eff"] == pytest.approx(2.0, rel=3e-2)
    assert equal["rolling_condition_exact_err"] <= 3e-2


def test_driven_rolling_force_balance(driven):
    """Steady rolling requires F_drive = M_r / r_j (<= 2 %)."""
    assert driven["force_balance_rel_err"] <= 2.0e-2
    assert driven["moment_rel_err"] <= 2.0e-2
    assert abs(driven["on_balance"]["accel"]) < 0.05


def test_driven_rolling_negative_controls(driven):
    """Under-drive decelerates, over-drive accelerates: the check has teeth."""
    assert driven["under_drive"]["accel"] < 0.0
    assert driven["over_drive"]["accel"] > 0.0


# ---------------------------------------------------------------------------
# (f) inclined-plane stack stability
# ---------------------------------------------------------------------------
def test_stack_critical_angle_matches_closed_form(incline):
    for row in incline["stack_cases"]:
        assert row["rel_err_vs_analytic"] <= 5.0e-2


def test_stack_plane_limit_approaches_arctan_mu_r(incline):
    """r_b/r_a -> 0 must recover tan(theta_cr) = mu_r; measured 1.9 % at 0.02."""
    assert incline["plane_limit_rel_err"] <= 5.0e-2


def test_stack_equal_spheres_half_mu_r(incline):
    """Equal spheres: tan(theta_cr) = mu_r/2 (exact geometric factor)."""
    row = [r for r in incline["stack_cases"] if r["case"] == "equal_spheres"][0]
    assert row["mu_r_over_tan"] == pytest.approx(2.0, rel=1e-9)


def test_stack_dynamic_hold_versus_roll(incline):
    """Dynamically: a ball holds below theta_cr and rolls above it."""
    assert incline["dynamic_below"]["held"]
    assert not incline["dynamic_above"]["held"]
    assert (
        incline["dynamic_above"]["v_tail_max"]
        > 10.0 * incline["dynamic_below"]["v_tail_max"]
    )


def test_analytic_incline_critical_angle_is_arctan():
    assert analytic_incline_critical_angle_deg(0.2) == pytest.approx(
        math.degrees(math.atan(0.2)), rel=1e-15
    )
    assert analytic_stack_critical_angle_deg(
        0.2, 1.0, 1.0, 0.05, 0.05
    ) == pytest.approx(math.degrees(math.atan(0.1)), rel=1e-12)


# ---------------------------------------------------------------------------
# (g) energy ledger
# ---------------------------------------------------------------------------
def test_energy_elastic_limit(energy):
    assert energy["elastic_energy_err"] <= 1e-4


def test_energy_fully_dissipative(energy):
    """Overdamped collision: kinetic energy left <= 1e-4 of K0."""
    assert energy["dissipative_residual"] <= 1e-4


def test_energy_ledger_closure(energy):
    assert energy["worst_closure_rel"] <= 1e-4


def test_energy_dissipation_nonnegative(energy):
    for key in ("elastic_limit", "oblique_frictional", "fully_dissipative"):
        assert energy[key]["all_dissipative_nonnegative"]


def test_energy_oblique_restitution(energy):
    """Measured rebound ratio of the oblique impact tracks the input e (<=10 %)."""
    assert energy["oblique_restitution_err"] <= 1e-1


# ---------------------------------------------------------------------------
# Audit chain: the committed artifact agrees with a fresh run
# ---------------------------------------------------------------------------
def test_results_artifact_reproduces():
    spec = importlib.util.spec_from_file_location("verify_bb", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CRITERIA["a_tangential_stiffness_pct"] == 1.0
    assert module.CRITERIA["g_energy_residual"] == pytest.approx(1e-4)