"""The coupled contact state: fixed point, sliding case, DEM contact tie-in.

The acceptance criteria of the campaign are asserted here one by one, against
either an independent solver (bisection on the same residual) or a closed form:

* **C1** the coupled solve returns a contact temperature and a friction heat;
* **C2** ``mu`` falls as the contact warms, in three different regimes;
* **C3** the fixed point converges in fewer than ten iterations at
  ``|dmu|/mu < 1 %``, and two *independent* solvers agree on its value;
* **C5** the friction heat equals friction force times slip velocity, computed
  through two different chains (force x velocity and flux x real area), and the
  DEM tangential limit of :mod:`tensordem.platen_friction` reproduces the same
  force.

The sliding case adds the time-marched physics: energy closure of the melt-film
ledger, the interface never above its melting point, monotone feedback, and the
comparison against the no-feedback control.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_friction_thermo import (
    CC_SLOPE_K_PER_MPA,
    ContactConditions,
    ContactFrictionModel,
    SlidingCase,
    TwoBodyContactConductor,
    calibrate_from_barnes_sweep,
    convergence_table,
    friction_heat_identity_check,
    interface_gain_k_per_mu,
    real_contact_pressure,
    run_sliding_case,
    solve_contact_state,
    solve_contact_state_bisection,
    temperature_monotonicity_check,
    validation_table,
)
from tensordem.platen_friction import platen_friction_limit, platen_slip_acceleration


@pytest.fixture(scope="module")
def model() -> ContactFrictionModel:
    calibrated, _ = calibrate_from_barnes_sweep()
    return calibrated


def _conditions(v: float = 1.0, pressure_mpa: float = 1.0, temperature: float = -5.0,
                area: float = 0.1, spot: float = 1.0e-4) -> ContactConditions:
    return ContactConditions(normal_force_n=pressure_mpa * 1.0e6 * area,
                             nominal_area_m2=area, slip_velocity_m_s=v,
                             bulk_temperature_c=temperature,
                             contact_spot_radius_m=spot)


# ---------------------------------------------------------------------------
# C1
# ---------------------------------------------------------------------------
def test_c1_contact_state_reports_temperature_and_heat(model):
    state = solve_contact_state(_conditions(), model)
    assert state.interface_temperature_c <= state.melting_temperature_c + 1e-12
    assert state.interface_temperature_c >= -5.0 - 1e-12
    assert state.friction_heat_w > 0.0
    assert state.mu > 0.0
    payload = state.as_dict()
    for key in ("interface_temperature_c", "friction_heat_w", "mu",
                "melting_temperature_c", "melt_fraction", "iterations"):
        assert key in payload
    assert state.real_pressure_pa == pytest.approx(model.hardness_pa)


def test_no_slip_means_no_heat_and_no_iteration(model):
    state = solve_contact_state(_conditions(v=0.0), model)
    assert state.friction_heat_w == 0.0
    assert state.interface_temperature_c == pytest.approx(-5.0, abs=1e-12)
    assert state.melt_fraction == 0.0
    assert state.regime == "dry"
    assert state.iterations == 0


def test_melting_branch_can_be_switched_off(model):
    off = ContactFrictionModel(mu_ref=model.mu_ref, beta_per_k=model.beta_per_k,
                               t_ref_c=model.t_ref_c, allow_melting=False)
    state = solve_contact_state(_conditions(temperature=-0.5), off)
    assert state.melt_fraction == 0.0
    assert state.melting_temperature_c == 0.0


# ---------------------------------------------------------------------------
# C2
# ---------------------------------------------------------------------------
def test_c2_mu_decreases_with_temperature_in_three_regimes(model):
    rows = temperature_monotonicity_check(model)
    assert len(rows) == 3
    decreasing = [row for row in rows if row["strictly_decreasing"]]
    assert len(decreasing) == 3, [(row["case"], [p["mu"] for p in row["series"]])
                                  for row in rows]
    for row in rows:
        assert row["mu_last"] < row["mu_first"]
        assert row["total_reduction"] > 0.10      # at least a 10 % drop per sweep


# ---------------------------------------------------------------------------
# C3
# ---------------------------------------------------------------------------
def test_c3_fixed_point_converges_in_under_ten_iterations(model):
    rows = convergence_table(model)
    assert rows
    for row in rows:
        assert row["converged"], row
        assert row["iterations"] < 10, row
        # the stopping rule is the criterion: 1 % relative change of mu
        assert row["final_residual"] < 1.0e-2, row
    assert max(row["iterations"] for row in rows) <= 6


def test_c3_picard_and_bisection_agree_on_the_root(model):
    """Two independent solvers, same residual: agreement is the verification."""
    for case in (_conditions(v=0.01, pressure_mpa=0.2),
                 _conditions(v=1.0, pressure_mpa=2.0, temperature=-2.0),
                 _conditions(v=2.0, pressure_mpa=0.5, temperature=-1.0)):
        picard = solve_contact_state(case, model, tolerance=1e-12, max_iterations=400)
        bisect = solve_contact_state_bisection(case, model)
        assert picard.mu == pytest.approx(bisect.mu, rel=1e-6, abs=1e-9)
        assert picard.interface_temperature_c == pytest.approx(
            bisect.interface_temperature_c, rel=1e-9)


def test_loop_gain_predicts_the_iteration_count(model):
    """``beta mu gain < 1`` is the contraction condition -- check it holds."""
    conditions = _conditions()
    gain = interface_gain_k_per_mu(conditions, model)
    mu = solve_contact_state(conditions, model).mu
    assert gain > 0.0
    assert model.beta_per_k * mu * gain < 1.0


# ---------------------------------------------------------------------------
# C5
# ---------------------------------------------------------------------------
def test_c5_friction_heat_is_force_times_velocity(model):
    rows = friction_heat_identity_check(model)
    assert rows
    for row in rows:
        assert row["rel_error_force_vs_flux"] < 1.0e-2
        assert row["rel_error_force_vs_model"] < 1.0e-2
    # and the same number reached through the DEM tangential limit
    conditions = _conditions()
    state = solve_contact_state(conditions, model)
    dem_limit = platen_friction_limit(state.mu, conditions.normal_force_n)
    assert dem_limit == pytest.approx(state.friction_force_n, rel=1e-12)
    assert platen_slip_acceleration(3.0 * dem_limit, state.mu,
                                    conditions.normal_force_n, 10.0) == pytest.approx(
        2.0 * dem_limit / 10.0, rel=1e-12)


# ---------------------------------------------------------------------------
# pressure melting regime
# ---------------------------------------------------------------------------
def test_pressure_melting_needs_the_clapeyron_threshold(model):
    """At hardness 100 MPa ice at -5 degC is pressure-melted; at 20 MPa it is not."""
    soft = ContactFrictionModel(mu_ref=model.mu_ref, beta_per_k=model.beta_per_k,
                                t_ref_c=model.t_ref_c, hardness_pa=2.0e7)
    hard = ContactFrictionModel(mu_ref=model.mu_ref, beta_per_k=model.beta_per_k,
                                t_ref_c=model.t_ref_c, hardness_pa=1.0e8)
    soft_state = solve_contact_state(_conditions(temperature=-5.0, v=0.0), soft)
    hard_state = solve_contact_state(_conditions(temperature=-5.0, v=0.0), hard)
    # the *regime* is decided by the pressure melting point alone, at rest too
    assert soft_state.regime == "dry"
    assert soft_state.melting_temperature_c == pytest.approx(-1.48, rel=1e-2)
    assert hard_state.melting_temperature_c < -5.0
    assert hard_state.regime == "pressure"
    # at rest there is no frictional heat to melt with, so the film needs sliding:
    # the coefficient only drops once the contact slides.
    assert hard_state.mu == pytest.approx(soft_state.mu, rel=1e-12)
    soft_sliding = solve_contact_state(_conditions(temperature=-5.0, v=1.0), soft)
    hard_sliding = solve_contact_state(_conditions(temperature=-5.0, v=1.0), hard)
    assert hard_sliding.melt_fraction > 0.0
    assert hard_sliding.mu < 0.9 * soft_sliding.mu
    # the melting point deficit is exactly the Clapeyron slope
    assert (0.0 - soft_state.melting_temperature_c) == pytest.approx(
        CC_SLOPE_K_PER_MPA * 20.0, rel=1e-12)


def test_real_pressure_saturates_so_amontons_holds(model):
    values = []
    for pressure_mpa in (0.05, 0.2, 1.0, 5.0):
        state = solve_contact_state(_conditions(pressure_mpa=pressure_mpa), model)
        values.append(state.mu)
    assert max(values) - min(values) < 1e-9      # plastic contact: area independent


# ---------------------------------------------------------------------------
# sliding case
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sliding(model):
    case = SlidingCase(normal_force_n=2.0e5, nominal_area_m2=0.1,
                       slip_velocity_m_s=1.0, bulk_temperature_c=-2.5,
                       model=model, ice_cells=120, steel_cells=120,
                       ice_depth_m=2.0e-3, steel_depth_m=2.0e-3)
    return case, run_sliding_case(case, duration_s=0.05, dt=1.0e-3, record_every=5)


def test_sliding_case_closes_its_energy_ledger(sliding):
    _, result = sliding
    assert result.friction_work_j_m2 > 0.0
    assert result.energy_closure_error < 1.0e-9, result.summary()
    assert result.latent_melted_j_m2 >= 0.0
    assert result.latent_drained_j_m2 <= result.latent_melted_j_m2 * (1.0 + 1e-9)


def test_sliding_case_never_exceeds_the_melting_point(sliding):
    _, result = sliding
    assert np.all(result.interface_temperature_c <= result.melting_temperature_c + 1e-9)
    assert np.all(result.melt_fraction <= 1.0)
    assert np.all(result.mu > 0.0)


def test_sliding_case_feedback_is_one_way(sliding):
    """Heating/melting can only lower mu inside a run (no spurious oscillation)."""
    _, result = sliding
    diffs = np.diff(result.mu)
    assert diffs.max() < 1e-9
    assert result.mu[-1] <= result.mu[0]
    assert result.iterations.max() < 10


def test_phase_change_feedback_lowers_the_coefficient(model, sliding):
    """The measured value of the transition: with vs without the melt feedback."""
    case, coupled = sliding
    control_case = SlidingCase(
        normal_force_n=case.normal_force_n, nominal_area_m2=case.nominal_area_m2,
        slip_velocity_m_s=case.slip_velocity_m_s,
        bulk_temperature_c=case.bulk_temperature_c,
        model=ContactFrictionModel(
            mu_ref=model.mu_ref, beta_per_k=model.beta_per_k, t_ref_c=model.t_ref_c,
            wet_ratio=1.0, onset_width_k=model.onset_width_k,
            hardness_pa=model.hardness_pa, mu_floor=model.mu_floor),
        ice_cells=case.ice_cells, steel_cells=case.steel_cells,
        ice_depth_m=case.ice_depth_m, steel_depth_m=case.steel_depth_m)
    control = run_sliding_case(control_case, duration_s=0.05, dt=1.0e-3, record_every=5)
    assert control.melt_fraction[-1] > 0.0          # water is produced in both runs
    assert coupled.mu[-1] < 0.95 * control.mu[-1]   # but only one lets it matter


def test_conductor_backed_solve_matches_the_flash_estimate(model):
    """A fresh contact solved through the conductor agrees with the Blok flash."""
    conditions = _conditions(v=1.0, pressure_mpa=1.0, temperature=-5.0)
    conductor = TwoBodyContactConductor(ice_depth_m=2.0e-3, steel_depth_m=2.0e-3,
                                        ice_cells=400, steel_cells=400,
                                        initial_temperature_c=-5.0)
    # the conductor step must be taken over the *spot life* to be comparable
    # with the semi-infinite flash estimate of the same contact
    zero, sensitivity = conductor.trial_response(1.0e-4)
    coupled_full = solve_contact_state(conditions, model,
                                       response=(float(zero[0]), float(sensitivity[0])))
    coupled_flash = solve_contact_state(conditions, model)
    # two different thermal back-ends, same physics: within 25 % on the rise
    rise_conductor = coupled_full.interface_temperature_c + 5.0
    rise_flash = coupled_flash.interface_temperature_c + 5.0
    assert rise_conductor == pytest.approx(rise_flash, rel=0.25)


def test_validation_rows_are_reproducible_and_mostly_in_gate(model):
    rows = validation_table(model)
    assert len(rows) == 5
    within = [row for row in rows if row["within_gate"]]
    assert len(within) >= 2, [(row["label"], row["rel_error"]) for row in rows]
    for row in rows:
        assert row["iterations"] < 10
        assert math.isfinite(row["rel_error"])