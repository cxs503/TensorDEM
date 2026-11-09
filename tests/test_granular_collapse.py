"""Verification of the granular column-collapse bench (2D DEM + mu(I) rheology).

Covers the geometry/mass bookkeeping of the released column, the mu(I) closure
limits, the power-law fitter used for the runout scaling, the contact-level
energy ledger (closure < 2 %), the monotone response of the runout to the aspect
ratio, determinism, and the API contract.

The cheap configurations here (a few hundred grains, a few thousand steps) are
*contract* tests: the quantitative runout scaling law is produced by
``scripts/run_dem_thermal_granular.py`` and reported in
``docs/DEM_THERMAL_GRANULAR_VERIFICATION.md``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tensordem.granular_collapse import (
    AXISYMMETRIC_EXPONENT,
    JOP_I_0,
    JOP_MU_2,
    JOP_MU_S,
    LAJEUNESSE_RECTANGULAR_EXPONENT_LOW,
    CollapseCase,
    CollapseError,
    build_column,
    column_collapse,
    fit_power_law,
    granular_collapse_calculation,
    jop_mu_I,
    local_inertial_number,
    potential_energy,
    runout_scan,
    scaling_comparison,
)

SMALL = dict(width=5, radius=0.02, ke_tol=1e-2, ke_hold=100)


def test_column_geometry_mass_and_potential_energy_are_consistent():
    case = CollapseCase(aspect=2.0, width=6, radius=0.02)
    system = build_column(case)
    rows = int(round(case.height / case.diameter))
    assert system.n == 6 * rows
    assert case.length == pytest.approx(6 * case.diameter)
    assert case.height == pytest.approx(2.0 * case.length)
    # lattice packed on the floor: no initial overlap, all above the floor
    assert np.all(system.radii > 0.0)
    assert np.min(system.pos[:, 1]) >= case.radius * 0.9
    # potential energy is the sum of m g y about the floor
    pe = potential_energy(system, case.gravity)
    assert pe == pytest.approx(float(np.sum(system.mass * case.gravity
                                           * system.pos[:, 1])), rel=1e-12)
    assert pe == pytest.approx(system.mass.sum() * case.gravity
                               * system.pos[:, 1].mean(), rel=1e-9)
    # the step is at (not beyond) the stability limit
    assert system.dt == pytest.approx(case.dt_safety * system.recommended_dt, rel=1e-12)


def test_mu_I_law_limits_and_monotonicity():
    assert float(jop_mu_I(0.0)) == pytest.approx(JOP_MU_S, rel=1e-6)
    assert float(jop_mu_I(1e9)) == pytest.approx(JOP_MU_2, rel=1e-6)
    values = jop_mu_I(np.logspace(-4, 2, 25))
    assert np.all(np.diff(values) > 0.0)
    assert np.all((values >= JOP_MU_S) & (values <= JOP_MU_2))
    # I == I_0 sits exactly halfway between the two asymptotes
    assert float(jop_mu_I(JOP_I_0)) == pytest.approx(0.5 * (JOP_MU_S + JOP_MU_2),
                                                    rel=1e-12)


def test_inertial_number_is_measured_from_the_state_not_imposed():
    case = CollapseCase(aspect=1.5, **SMALL)
    system = build_column(case)
    quiescent = local_inertial_number(system, case)
    assert quiescent == pytest.approx(0.0, abs=1e-12)
    # a shear profile must produce a positive inertial number
    system.vel[:, 0] = np.linspace(0.0, 1.0, system.n)
    system.contact_forces()
    sheared = local_inertial_number(system, case)
    assert sheared > 0.0
    # and it must scale with the shear rate
    system.vel[:, 0] *= 2.0
    doubled = local_inertial_number(system, case)
    assert doubled == pytest.approx(2.0 * sheared, rel=0.2)


def test_power_law_fit_recovers_a_synthetic_exponent():
    a = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    y = 0.8 * a**0.667
    fit = fit_power_law(a, y)
    assert fit["exponent"] == pytest.approx(0.667, rel=1e-9)
    assert fit["prefactor"] == pytest.approx(0.8, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(CollapseError):
        fit_power_law([1.0], [1.0])


def test_scaling_comparison_grades_against_the_literature_exponents():
    perfect = scaling_comparison(0.5, reference=AXISYMMETRIC_EXPONENT)
    assert perfect["passes"] and perfect["relative_error"] == pytest.approx(0.0)
    off = scaling_comparison(0.57, reference=LAJEUNESSE_RECTANGULAR_EXPONENT_LOW)
    assert off["relative_error"] == pytest.approx(0.43, rel=1e-9)
    assert not off["passes"]
    inside = scaling_comparison(0.545, reference=AXISYMMETRIC_EXPONENT)
    assert inside["passes"] and not perfect["passes"] is False


def test_energy_ledger_closes_below_two_percent():
    """Campaign gate (c): potential-energy / dissipation ledger closure < 2 %."""
    case = CollapseCase(aspect=1.5, max_steps=2500, **SMALL)
    result = column_collapse(case, with_ledger=True, record_every=500)
    ledger = result.ledger
    assert ledger["measured"].startswith("contact-level")
    assert ledger["dissipated_j"] > 0.0
    assert ledger["closure_error"] < 0.02, ledger
    # the dissipative channels are physical: damping and friction both positive
    assert ledger["dissipated_damping_j"] > 0.0
    assert ledger["dissipated_friction_j"] > 0.0
    # mechanical + dissipated reconstructs the released potential energy
    total = (ledger["potential_j"] + ledger["kinetic_j"] + ledger["elastic_normal_j"]
             + ledger["elastic_tangential_j"] + ledger["dissipated_j"])
    assert total == pytest.approx(ledger["potential_initial_j"], rel=0.02)


def test_runout_grows_with_the_aspect_ratio():
    low = column_collapse(CollapseCase(aspect=1.0, max_steps=2000, **SMALL),
                          record_every=500)
    high = column_collapse(CollapseCase(aspect=3.0, max_steps=2000, **SMALL),
                           record_every=500)
    assert high.runout > low.runout
    assert high.relative_runout > low.relative_runout
    assert 0.0 < high.inertial_number_mean  # I is measured, not imposed
    assert 0.0 < high.max_overlap_ratio < 0.05  # soft-contact validity
    assert len(low.front_history) == 5


def test_collapse_is_deterministic_for_a_fixed_seed():
    case = CollapseCase(aspect=1.0, max_steps=300, **SMALL)
    first = column_collapse(case, record_every=100)
    second = column_collapse(case, record_every=100)
    assert first.digest == second.digest
    assert first.runout == second.runout
    other = column_collapse(replace(case, seed=7), record_every=100)
    assert other.digest != first.digest


def test_runout_scan_returns_fits_for_both_regimes():
    case = CollapseCase(aspect=1.0, max_steps=800, **SMALL)
    scan = runout_scan([1.0, 2.0, 3.0, 4.0], case=case, record_every=400)
    assert len(scan["cases"]) == 4
    assert set(scan["fit_all"]) == {"prefactor", "exponent", "r2", "n_points"}
    assert scan["fit_low_aspect"] is not None
    assert scan["fit_high_aspect"] is not None


@pytest.mark.parametrize("kwargs", [
    {"aspect": 0.0},
    {"aspect": -1.0},
    {"width": 2},
    {"radius": 0.0},
    {"restitution": 1.0},
    {"mu_s": 0.9, "mu_2": 0.5},
    {"mu_rolling": -0.1},
])
def test_collapse_case_rejects_unphysical_inputs(kwargs):
    with pytest.raises(CollapseError):
        CollapseCase(**kwargs)


def test_api_contract_single_and_scan_modes():
    single = granular_collapse_calculation({"aspect": 1.0, "width": 5, "radius": 0.02,
                                            "max_steps": 200})
    assert single["mode"] == "single"
    assert single["endpoint"] == "/api/v1/calculations/dem/granular_collapse"
    assert single["capability"] == "REAL"
    assert "relative_runout" in single["result"]
    scan = granular_collapse_calculation({"aspects": [1.0, 2.0], "width": 5,
                                          "radius": 0.02, "max_steps": 200})
    assert scan["mode"] == "scan"
    assert scan["verdict"] is not None
    with pytest.raises(CollapseError):
        granular_collapse_calculation("not a mapping")