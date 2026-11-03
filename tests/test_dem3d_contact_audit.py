"""Adjudication tests for the 3-D contact model of :mod:`tensordem.dem3d`.

The external audit (dev_report_tensordem-3d-contact-audit.md) required each
suspicion about ``src/tensordem/dem3d.py`` + ``src/tensordem/contact_tangential.py``
to be settled with a small, analytically solvable case and an explicit
pass/fail verdict:

* ``(a)`` four contact states - frictionless / pure sliding / pure rolling /
  resting;
* ``(b)`` action-reaction pairing of every contact term;
* ``(c)`` the friction-dissipation ledger;
* ``(d)`` how bond separation interacts with the tangential contact state.

The numerical probes live in :mod:`scripts.audit_dem3d_contact_3d` (the same
module that writes ``artifacts/dem3d_contact_audit/``), so the numbers asserted
here are byte-identical to the ones reported in the campaign artifacts.

Honest failures are marked ``xfail(strict=True)`` with the reason; they are real
capability/accuracy gaps, not tolerances that were relaxed to obtain green.
"""
from __future__ import annotations

import math

import pytest
import torch

torch.set_num_threads(1)  # tiny state vectors: extra threads only add
                          # contention when the machine is busy

from scripts.audit_dem3d_contact_3d import (
    TOLERANCES,
    bond_separation_history,
    build_pair,
    dissipation_ledger,
    legacy_rolling_force,
    legacy_sliding_state_defect,
    normal_rotation_projection,
    pairing_antisymmetry,
    pairing_torque_budget,
    pair_forces,
    recontact_reset,
    restitution_engine,
    rolling_dissipation,
    state_frictionless,
    state_pure_rolling,
    state_pure_sliding,
    state_resting,
)
from tensordem.contact_tangential import (
    advance_tangential_contact,
    compute_tangential_force,
    saturation_displacement,
)


# ---------------------------------------------------------------------------
# Session-scoped probes: each one is a full (cheap) analytic campaign
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def frictionless():
    return state_frictionless()


@pytest.fixture(scope="session")
def pure_sliding():
    return state_pure_sliding()


@pytest.fixture(scope="session")
def pure_rolling():
    return state_pure_rolling()


@pytest.fixture(scope="session")
def resting():
    return state_resting()


@pytest.fixture(scope="session")
def antisymmetry():
    return pairing_antisymmetry()


@pytest.fixture(scope="session")
def couple_budget():
    return pairing_torque_budget()


@pytest.fixture(scope="session")
def ledger():
    return dissipation_ledger()


@pytest.fixture(scope="session")
def legacy_defect():
    return legacy_sliding_state_defect()


@pytest.fixture(scope="session")
def bond_separation():
    return bond_separation_history()


@pytest.fixture(scope="session")
def recontact():
    return recontact_reset()


@pytest.fixture(scope="session")
def normal_rotation():
    return normal_rotation_projection()


@pytest.fixture(scope="session")
def rolling_terms():
    return rolling_dissipation()


# ---------------------------------------------------------------------------
# (a) Four contact states
# ---------------------------------------------------------------------------
class TestContactStates:
    """``a) 无摩擦 / 纯滑动 / 纯滚动 / 静止接触`` - one analytic case each."""

    def test_frictionless_contact_has_no_tangential_force(self, frictionless):
        """mu = mu_r = 0: only the normal spring acts, |F_n| = k_n * overlap."""
        assert frictionless["verdict"] == "PASS", frictionless["checks"]
        checks = frictionless["checks"]
        assert checks["tangential_force"]["measured"] == 0.0
        assert checks["normal_force"]["rel_error"] <= TOLERANCES["identity"]
        assert checks["net_pair_force"]["measured"] == 0.0

    def test_pure_sliding_holds_the_coulomb_cone(self, pure_sliding):
        """Saturated sliding: |F_t| = mu*F_n exactly, direction opposed to v_t."""
        assert pure_sliding["verdict"] == "PASS", pure_sliding["checks"]
        limit = pure_sliding["info"]["mu"] * pure_sliding["info"]["normal_force_N"]
        assert limit == pytest.approx(4.0)
        driver = pure_sliding["checks"]["tangential_on_driver"]
        assert driver["measured"] == pytest.approx(-limit, rel=1e-10)
        support = pure_sliding["checks"]["tangential_on_support"]
        assert support["measured"] == pytest.approx(limit, rel=1e-10)
        # the stored state is the clamped displacement delta_s = mu*F_n/k_t
        assert pure_sliding["stored_disp_m"] == pytest.approx(
            pure_sliding["cone_radius_m"], rel=1e-12)

    def test_resting_contact_does_not_creep(self, resting):
        """v = 0: no drift, no tangential force, and a sub-limit slip is held."""
        assert resting["verdict"] == "PASS", resting["checks"]
        checks = resting["checks"]
        assert checks["position_drift"]["measured"] <= 1e-12
        assert checks["velocity"]["measured"] <= 1e-12
        assert checks["tangential_force_free"]["measured"] == 0.0
        # static friction carries 0.5*delta_s without relaxing
        assert checks["held_slip"]["rel_error"] <= TOLERANCES["ledger"]
        assert checks["below_cone"]["verdict"] == "PASS"

    @pytest.mark.xfail(
        strict=True,
        reason="no rotational DOF in IceDEM3D: a pure-rolling contact cannot be "
               "represented and mu_r produces no resistance at zero slip",
    )
    def test_pure_rolling_is_resisted(self, pure_rolling):
        """A rolling contact must resist rolling: |M_r| = mu_r*F_n*R."""
        assert pure_rolling["rotational_dof_present"] is True
        assert pure_rolling["equivalent_moment_at_zero_slip_Nm"] == pytest.approx(
            pure_rolling["expected_rolling_moment_Nm"], rel=1e-6)

    def test_pure_rolling_capability_is_documented_as_absent(self, pure_rolling):
        """The limitation itself is pinned down so a future fix must update it."""
        assert pure_rolling["verdict"] == "NOT_SUPPORTED"
        assert pure_rolling["rotational_dof_present"] is False
        assert pure_rolling["resistance_at_zero_slip_N"] == 0.0
        assert pure_rolling["expected_rolling_moment_Nm"] > 0.0


# ---------------------------------------------------------------------------
# (b) Action / reaction pairing
# ---------------------------------------------------------------------------
class TestPairing:
    """``b) 作用反作用成对性`` - every contact term must be antisymmetric."""

    def test_all_contact_terms_are_antisymmetric(self, antisymmetry):
        assert antisymmetry["verdict"] == "PASS", antisymmetry["cases"]
        for name, case in antisymmetry["cases"].items():
            assert case["measured"] <= 1e-10, name
            assert case["verdict"] == "PASS", name

    def test_pair_couple_is_documented_as_structural_limitation(self, couple_budget):
        """A sliding pair applies a net couple 2*R*mu*F_n; without a spin DOF the
        reaction has no sink, so angular momentum is not conserved by design."""
        assert couple_budget["verdict"] == "STRUCTURAL_LIMITATION"
        assert couple_budget["net_force_N"] == pytest.approx(0.0, abs=1e-12)
        assert couple_budget["couple_Nm"] == pytest.approx(
            couple_budget["couple_analytic_Nm"], rel=1e-6)
        assert couple_budget["couple_analytic_Nm"] > 0.0


# ---------------------------------------------------------------------------
# (c) Friction dissipation ledger
# ---------------------------------------------------------------------------
class TestDissipationLedger:
    """``c) 摩擦功耗散账目`` - Coulomb work must equal the kinetic-energy drop."""

    def test_sliding_dissipation_matches_coulomb_work(self, ledger):
        assert ledger["verdict"] == "PASS", ledger
        assert ledger["saturated"] is True
        assert ledger["closure_rel_error"] <= TOLERANCES["ledger"]
        # dissipated = mu*F_n*s_slip inside the sliding window
        assert ledger["predicted_dissipation_J"] == pytest.approx(
            ledger["cone_limit_N"] * ledger["slip_in_sliding_window_m"], rel=1e-12)
        assert ledger["sliding_ke_loss_J"] == pytest.approx(
            ledger["predicted_dissipation_J"], rel=TOLERANCES["ledger"])

    def test_force_never_exceeds_the_cone(self, ledger):
        assert ledger["force_ceiling_respected"] is True
        assert ledger["peak_tangential_force_N"] <= ledger["cone_limit_N"] * (1 + 1e-9)

    def test_spring_energy_matches_closed_form(self, ledger):
        assert ledger["spring_energy_J"] == pytest.approx(
            ledger["analytic_spring_energy_J"], rel=1e-6)

    def test_unclamped_state_defect_is_quantified(self, legacy_defect):
        """The pre-fix raw-state stepper needs the whole accumulated slip to
        reverse the friction force instead of the elastic length delta_s."""
        assert legacy_defect["legacy_rel_error"] > 10.0 * TOLERANCES["ledger"]
        assert legacy_defect["projected_rel_error"] <= TOLERANCES["ledger"]
        assert legacy_defect["legacy_to_projected_ratio"] > 5.0
        assert (legacy_defect["stored_state_legacy_before_reversal_m"]
                > 5.0 * legacy_defect["cone_radius_m"])
        assert legacy_defect["stored_state_projected_before_reversal_m"] == pytest.approx(
            legacy_defect["cone_radius_m"], rel=1e-12)


# ---------------------------------------------------------------------------
# (d) Bond separation interaction
# ---------------------------------------------------------------------------
class TestBondSeparation:
    """``d) 断键后接触`` - a bonded pair must not cache contact history."""

    def test_bonded_pair_caches_no_tangential_state(self, bond_separation):
        assert bond_separation["verdict"] == "PASS", bond_separation["checks"]
        assert bond_separation["stored_disp_while_bonded_m"] == 0.0
        assert bond_separation["tangential_pair_force_while_bonded_N"] == 0.0

    def test_first_force_after_break_is_one_increment(self, bond_separation):
        """After the bond breaks, the contact starts from delta_t = 0 (one step of
        slip), not from a saturated force built up while the bond was alive."""
        checks = bond_separation["checks"]
        assert checks["first_step_after_break"]["verdict"] == "PASS"
        one_increment = bond_separation["analytic_one_increment_N"]
        assert bond_separation["force_first_step_after_break_N"] == pytest.approx(
            one_increment, rel=TOLERANCES["ledger"])
        assert one_increment < 0.01 * bond_separation["cone_limit_N"]

    def test_contact_loss_resets_the_history(self, recontact):
        assert recontact["verdict"] == "PASS", recontact["checks"]
        assert recontact["state_pruned_on_contact_loss"] is True
        assert recontact["tangential_force_after_recontact_N"] <= 1e-12
        # what the stale state would have produced
        assert recontact["stale_force_if_not_reset_N"] > 0.5 * recontact["cone_limit_N"]

    def test_rotating_normal_is_projected_out(self, normal_rotation):
        assert normal_rotation["verdict"] == "PASS", normal_rotation["checks"]
        assert normal_rotation["checks"]["spurious_normal_history"]["measured"] <= 1e-12
        assert normal_rotation["tangential_force_N"] <= 1e-12
        assert normal_rotation["stale_force_if_not_projected_N"] > 0.0


# ---------------------------------------------------------------------------
# Q1 / Q3: rolling resistance and restitution verified independently
# ---------------------------------------------------------------------------
class TestRollingResistance:
    """The effective rolling resistance must be dissipative and on the right axis."""

    def test_rolling_term_is_dissipative(self, rolling_terms):
        assert rolling_terms["verdict"] == "PASS", rolling_terms
        assert rolling_terms["dissipative"] is True
        assert rolling_terms["contact_work_J"] < 0.0
        assert rolling_terms["kinetic_energy_monotone"] is True

    def test_rolling_magnitude_and_axis(self, rolling_terms):
        assert rolling_terms["resistance_magnitude_N"] == pytest.approx(
            rolling_terms["analytic_resistance_N"], rel=1e-10)
        assert rolling_terms["moment_on_rolling_axis"] is True
        # documented residual: the couple of the point-mass model is 2*R*mu_r*F_n
        assert rolling_terms["moment_factor"] == pytest.approx(2.0, rel=0.05)

    def test_rolling_work_matches_kinetic_energy_drop(self, rolling_terms):
        assert rolling_terms["closure_rel_error"] <= TOLERANCES["ledger"]
        assert rolling_terms["resistance_at_rest_N"] == 0.0

    def test_legacy_transverse_force_dissipated_nothing(self):
        """The pre-fix term was orthogonal to the sliding velocity: power = 0."""
        legacy = legacy_rolling_force()
        assert legacy["verdict"] == "FAIL_LEGACY"
        assert legacy["power_legacy_W"] == pytest.approx(0.0, abs=1e-15)
        assert legacy["power_current_W"] == pytest.approx(
            legacy["analytic_dissipation_W"], rel=1e-12)
        assert legacy["magnitude_current_N"] == pytest.approx(
            legacy["magnitude_legacy_N"], rel=1e-12)


class TestRestitution:
    """``restitution_coefficient`` must be reproduced by the engine, not only by
    the analytic algebra."""

    def test_weak_and_moderate_damping_reproduce_e(self):
        for target in (0.9, 0.8):
            result = restitution_engine(e_target=target, dt=4.0e-5)
            assert result["rel_error"] <= TOLERANCES["restitution"], result

    def test_perfectly_elastic_impact_is_exact(self):
        result = restitution_engine(e_target=1.0, dt=4.0e-5)
        assert result["rel_error"] <= 1e-4, result

    @pytest.mark.xfail(
        strict=True,
        reason="linear spring-dashpot with a unilateral (F_n >= 0) clamp does not "
               "reproduce e for strong damping: measured e=0.55 for target 0.5; "
               "use restitution.damping_for_clamped_restitution to calibrate",
    )
    def test_strong_damping_reproduces_e(self):
        result = restitution_engine(e_target=0.5, dt=1.0e-5)
        assert result["rel_error"] <= TOLERANCES["restitution"], result

    def test_deviation_is_a_model_effect_not_discretisation(self):
        coarse = restitution_engine(e_target=0.5, dt=4.0e-5)
        fine = restitution_engine(e_target=0.5, dt=1.0e-5)
        shift = abs(fine["e_measured"] - coarse["e_measured"])
        assert shift < 0.01              # dt refinement changes almost nothing
        assert coarse["rel_error"] > TOLERANCES["restitution"]
        assert fine["rel_error"] > TOLERANCES["restitution"]


# ---------------------------------------------------------------------------
# Projected state machine (unit level, independent of the engine)
# ---------------------------------------------------------------------------
class TestProjectedStateMachine:
    """Unit-level anchors of the kernel the engine now uses."""

    def test_cone_radius(self):
        value = saturation_displacement(
            torch.tensor([100.0], dtype=torch.float64), k_t=1.0e4, mu=0.3)
        assert float(value[0]) == pytest.approx(3.0e-3, rel=1e-12)

    def test_lost_contact_resets_the_state(self):
        force, disp, sliding = advance_tangential_contact(
            torch.tensor([[0.0, 2.0e-3]], dtype=torch.float64),
            torch.zeros((1, 2), dtype=torch.float64),
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            k_t=1.0e4, mu=0.3, dt=1.0e-4)
        assert float(torch.linalg.vector_norm(force[0])) == 0.0
        assert float(torch.linalg.vector_norm(disp[0])) == 0.0
        assert bool(sliding[0]) is False

    def test_stored_state_never_exceeds_the_cone(self):
        disp = torch.zeros((1, 3), dtype=torch.float64)
        velocity = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64)
        normal = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        force_mag = torch.tensor([10.0], dtype=torch.float64)
        limit = 0.4 * 10.0
        delta_s = limit / 5_000.0
        for index in range(50):
            force, disp, sliding = advance_tangential_contact(
                disp, velocity, normal, force_mag, 5_000.0, 0.4, 1.0e-4)
            state = float(torch.linalg.vector_norm(disp[0]))
            assert state <= delta_s * (1.0 + 1e-12)
            assert float(torch.linalg.vector_norm(force[0])) <= limit * (1.0 + 1e-12)
            # first step still sticks (5e-4 < delta_s = 8e-4), then it slides
            assert bool(sliding[0]) is (index >= 1)

    def test_normal_component_is_projected_out(self):
        """A history vector along the normal must not produce a force."""
        normal = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        force, disp, _sliding = advance_tangential_contact(
            torch.tensor([[3.0e-3, 0.0, 0.0]], dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.float64),
            normal, torch.tensor([10.0], dtype=torch.float64),
            5_000.0, 0.4, 1.0e-4)
        assert float(torch.linalg.vector_norm(disp[0])) == 0.0
        assert float(torch.linalg.vector_norm(force[0])) == 0.0

    def test_stateless_kernel_returns_the_clamped_state(self):
        force, state = compute_tangential_force(
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.5]], dtype=torch.float64),
            torch.tensor([100.0], dtype=torch.float64), 1_000.0, 0.3)
        assert float(force[0, 1]) == pytest.approx(30.0, rel=1e-12)
        assert float(state[0, 1]) == pytest.approx(0.03, rel=1e-12)
        torch.testing.assert_close(force, 1_000.0 * state)


# ---------------------------------------------------------------------------
# Diagnostics purity regression (found during the audit)
# ---------------------------------------------------------------------------
def test_diagnostics_does_not_advance_the_tangential_history():
    """``diagnostics()`` evaluates forces read-only; it must not age the contact
    state (a double increment per step would otherwise inflate the friction)."""
    sim, info = build_pair(mu=0.4, overlap=2.0e-3)
    sim.velocities[1] = torch.tensor([0.0, 0.5, 0.0], dtype=sim.dtype)
    key = sim._pair_key(0, 1)
    for _ in range(5):
        sim.forces()
    reference = sim._tangential_state[key][0].clone()
    for _ in range(3):
        sim.diagnostics()
    assert torch.equal(sim._tangential_state[key][0], reference)

    # and a committed evaluation does advance it by exactly one increment
    sim.forces()
    increment = float(sim.velocities[1, 1] - sim.velocities[0, 1]) * sim.dt
    delta = float(torch.linalg.vector_norm(sim._tangential_state[key][0] - reference))
    assert delta == pytest.approx(increment, rel=1e-12)
    assert math.isfinite(delta)


def test_state_dict_and_friction_energy_are_consistent():
    """The tangential spring energy must appear in the energy ledger."""
    sim, info = build_pair(mu=0.4, overlap=2.0e-3)
    sim.velocities[1] = torch.tensor([0.0, 0.5, 0.0], dtype=sim.dtype)
    for _ in range(200):
        sim.forces()
    row = sim.energy()
    limit = 0.4 * info["normal_force_N"]
    expected = 0.5 * limit ** 2 / info["k_t_N_per_m"]
    assert row["particle_tangential_energy_J"] == pytest.approx(expected, rel=1e-6)
    assert row["mechanical_energy_J"] >= row["particle_tangential_energy_J"]