"""Verification of the melt-lubrication law and of the two-way coupled contact.

Two acceptance criteria are measured here directly:

* (d) the DEM <-> FVM iteration converges within 5 steps at a residual below 1 %;
* (e) the friction coefficient decreases monotonically with temperature and
  tends to zero at the melting point.

Two negative controls guard the coupling: no slip means no heating at all (the
temperature field must stay at its initial value), and a contact that never
reaches the lubrication band must reproduce plain Coulomb mechanics exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.interface_thermal.coupling import (
    CoupledInterface,
    IceSteelContact,
    hertz_contact_radius,
)
from tensordem.interface_thermal.lubrication import (
    MeltLubricationModel,
    film_normal_stiffness,
    load_bearing_area_fraction,
    series_normal_stiffness,
    stefan_melt_rate,
)

CASE = dict(normal_force_n=500.0, mu_cold=0.08, nominal_contact_area_m2=1.0e-4,
            initial_temperature_c=-10.0, cells=161, grading=1.05)


def _case(**overrides):
    settings = dict(CASE)
    settings.update(overrides)
    return IceSteelContact(**settings)


# ---------------------------------------------------------------------------
# lubrication law
# ---------------------------------------------------------------------------
def test_mu_decreases_monotonically_and_vanishes_at_the_melting_point():
    """Criterion (e): mu(T) monotone down, mu(Tmelt) = 0 exactly."""
    model = MeltLubricationModel(mu_base=0.10, t_melt=0.0, t_onset=-3.0, exponent=2.0)
    temperatures = np.linspace(-8.0, 0.0, 401)
    values = np.array([model.mu(t) for t in temperatures])
    assert np.all(np.diff(values) <= 1e-15), "mu must not increase with T"
    assert model.mu(0.0) == 0.0
    assert model.mu(-10.0) == pytest.approx(0.10, rel=1e-15)
    assert model.mu(-3.0) == pytest.approx(0.10, rel=1e-15)
    assert model.mu(-1.5) == pytest.approx(0.10 * 0.25, rel=1e-12)


def test_mu_derivative_is_negative_inside_the_lubrication_band_only():
    model = MeltLubricationModel(mu_base=0.10, t_melt=0.0, t_onset=-3.0, exponent=2.0)
    assert model.dmu_d_temperature(-10.0) == 0.0
    assert model.dmu_d_temperature(0.5) == 0.0
    assert model.dmu_d_temperature(-1.5) < 0.0
    # finite difference agrees with the analytic derivative
    h = 1e-6
    numeric = (model.mu(-1.5 + h) - model.mu(-1.5 - h)) / (2 * h)
    assert numeric == pytest.approx(model.dmu_d_temperature(-1.5), rel=1e-5)


def test_mu_with_wet_residual_stays_monotone_below_the_dry_value():
    """A boundary-lubrication floor keeps mu > 0 but must not break monotonicity."""
    model = MeltLubricationModel(mu_base=0.10, t_melt=0.0, t_onset=-3.0,
                                 exponent=2.0, mu_wet=0.03)
    temperatures = np.linspace(-8.0, 0.0, 401)
    values = np.array([model.mu(t) for t in temperatures])
    assert np.all(np.diff(values) <= 1e-15)
    assert model.mu(0.0) == pytest.approx(0.03, rel=1e-15)


def test_mu_wet_above_dry_friction_is_rejected():
    model = MeltLubricationModel(mu_base=0.05, mu_wet=0.06)
    with pytest.raises(ValueError):
        model.mu(-1.0)


def test_lubrication_model_rejects_inverted_band():
    with pytest.raises(ValueError):
        MeltLubricationModel(t_melt=0.0, t_onset=1.0)


def test_film_stiffness_limits():
    """Film stiffness: series combination reduces to the contact in both limits."""
    k_contact = 2.0e7
    area = 1.0e-4
    model = MeltLubricationModel()
    assert model.effective_normal_stiffness(k_contact, area, 0.0) == pytest.approx(k_contact)
    thin = model.effective_normal_stiffness(k_contact, area, 1e-12)
    assert thin == pytest.approx(k_contact, rel=1e-6)
    # h -> infinity is the mathematical limit of the series spring; a 1 m probe
    # is used only to approach it (a physical melt film is 1e-6 .. 1e-4 m)
    thick = model.effective_normal_stiffness(k_contact, area, 1.0)
    assert thick < 0.05 * k_contact
    assert model.effective_normal_stiffness(k_contact, area, 1e-2) < k_contact
    assert film_normal_stiffness(area, 1e-6) > film_normal_stiffness(area, 1e-3)
    assert series_normal_stiffness(k_contact, 1e30) == pytest.approx(k_contact, rel=1e-12)


def test_stefan_melt_rate_scales_with_flux_and_latent_heat():
    rate = stefan_melt_rate(1.0e4, 917.0, 3.34e5)
    assert rate == pytest.approx(1.0e4 / (917.0 * 3.34e5), rel=1e-12)
    assert stefan_melt_rate(0.0, 917.0, 3.34e5) == 0.0
    assert stefan_melt_rate(-1.0e4, 917.0, 3.34e5) < 0.0     # refreezing allowed


def test_load_bearing_fraction_is_complementary_and_guarded():
    assert load_bearing_area_fraction(0.0) == 1.0
    assert load_bearing_area_fraction(1.0) == 0.0
    with pytest.raises(ValueError):
        load_bearing_area_fraction(1.5)


# ---------------------------------------------------------------------------
# contact mechanics
# ---------------------------------------------------------------------------
def test_hertz_contact_radius_grows_as_force_to_one_third():
    a1 = hertz_contact_radius(1000.0, 0.05, 9e9, 0.3, 210e9, 0.3)
    a8 = hertz_contact_radius(8000.0, 0.05, 9e9, 0.3, 210e9, 0.3)
    assert a8 / a1 == pytest.approx(8.0 ** (1.0 / 3.0), rel=1e-12)


def test_blok_partition_is_reported_by_the_case():
    case = _case()
    partition = case.partition()
    assert partition["partition_1"] + partition["partition_2"] == pytest.approx(1.0, abs=1e-15)
    assert 0.0 < partition["partition_1"] < 1.0


# ---------------------------------------------------------------------------
# coupled solver: criterion (d), energy balance, negative controls
# ---------------------------------------------------------------------------
def test_coupled_iteration_converges_within_five_steps():
    """Criterion (d): <= 5 iterations, residual < 1 %."""
    interface = CoupledInterface(_case(slip_velocity_m_s=0.5))
    result = interface.run(2.0, 0.005)
    assert result["all_converged"]
    assert result["max_iterations_used"] <= 5
    assert result["max_residual"] < 1e-2


def test_coupled_iteration_converges_in_the_melting_regime():
    case = _case(slip_velocity_m_s=1.0,
                 lubrication=MeltLubricationModel(mu_base=0.08, mu_wet=0.03))
    interface = CoupledInterface(case)
    result = interface.run(2.0, 0.005)
    assert result["all_converged"]
    assert result["max_iterations_used"] <= 5
    assert result["max_residual"] < 1e-2
    assert result["film_thickness_m"][-1] > 0.0
    assert "melting" in set(result["mode"])


def test_coupled_energy_ledger_closes():
    for velocity, key in ((0.5, {}), (1.0, {"lubrication": MeltLubricationModel(
            mu_base=0.08, mu_wet=0.03)})):
        interface = CoupledInterface(_case(slip_velocity_m_s=velocity, **key))
        result = interface.run(2.0, 0.005)
        closure = result["energy_ledger"]["closure_error"]
        assert abs(closure) < 1e-4, f"v={velocity}: closure {closure:.3e}"


def test_coupled_zero_slip_produces_no_heating():
    """Negative control: no slip -> no friction heat -> isothermal contact."""
    interface = CoupledInterface(_case(slip_velocity_m_s=0.0))
    result = interface.run(1.0, 0.005)
    assert np.allclose(result["interface_temperature_c"], -10.0, atol=1e-9)
    assert np.allclose(result["heat_flux_w_m2"], 0.0, atol=1e-12)
    assert np.allclose(result["mu"], 0.08, atol=1e-12)
    assert result["film_thickness_m"][-1] == 0.0
    assert abs(result["energy_ledger"]["ice_energy_in_j_m2"]) < 1e-9


def test_coupled_without_phase_change_reduces_to_coulomb_mechanics():
    """Negative control: below the lubrication band the law is pure Coulomb."""
    interface = CoupledInterface(_case(slip_velocity_m_s=0.5))
    result = interface.run(0.02, 0.005)     # too short to warm the interface
    assert np.all(result["interface_temperature_c"] < -3.0)
    assert np.allclose(result["mu"], 0.08, atol=1e-12)
    assert np.allclose(result["friction_force_n"], 0.08 * 500.0, rtol=1e-12)
    assert result["film_thickness_m"][-1] == 0.0


def test_coupled_friction_force_decreases_as_the_interface_warms():
    interface = CoupledInterface(_case(slip_velocity_m_s=0.5))
    result = interface.run(3.0, 0.005)
    force = result["friction_force_n"]
    assert force[0] == pytest.approx(0.08 * 500.0, rel=1e-12)
    assert force[-1] < 0.5 * force[0]
    assert np.all(np.diff(result["interface_temperature_c"]) > -1e-9)   # heating
    assert np.all(np.diff(result["mu"]) <= 1e-12)


def test_normal_force_drops_with_the_melt_film_at_fixed_approach():
    case = _case(slip_velocity_m_s=1.0,
                 lubrication=MeltLubricationModel(mu_base=0.08, mu_wet=0.03))
    interface = CoupledInterface(case)
    force_before = interface.normal_force_at_approach(1e-4)
    interface.run(2.0, 0.005)
    force_after = interface.normal_force_at_approach(1e-4)
    assert interface.film_thickness_m > 0.0
    assert force_after < force_before
    assert force_after == pytest.approx(
        case.lubrication.effective_normal_stiffness(
            case.contact_stiffness_n_m, case.contact_area_m2,
            interface.film_thickness_m) * 1e-4, rel=1e-12)


def test_coupled_solver_rejects_bad_tolerance():
    with pytest.raises(ValueError):
        CoupledInterface(_case(), tolerance=0.0)
    with pytest.raises(ValueError):
        CoupledInterface(_case(), max_iterations=1)