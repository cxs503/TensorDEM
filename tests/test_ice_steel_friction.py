"""Tests for the calibrated ice-steel friction law ``mu(v, T, p)``.

The suite covers four independent questions:

1. **Identifiability.** Synthetic anchors generated from a known law must be
   recovered exactly by the closed-form fit (a least-squares implementation that
   is off by any amount fails here, not in a downstream comparison).
2. **Trends.** Speed weakening, colder-is-frictionier and pressure weakening are
   physical statements, checked as monotonicity of the fitted law over decades.
3. **Validation protocol.** The independent holdout and the k-fold protocol must
   pass the 10 % gate for the recommended form, and leave-one-source-out must
   honestly report its own degeneracy (each source isolates one variable).
4. **Coupling.** The calibrated coefficient must be what the DEM contact actually
   realises (slip-branch probe), and the ice-channel adapter must set exactly the
   model coefficient without touching the rest of the contact parameters.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.ice_channel import ContactParameters
from tensordem.ice_steel_friction import (
    CALIBRATION_ANCHORS,
    EXTRAPOLATION_PROBES,
    MU_CONSTANT_BASELINE,
    MU_FLOOR,
    VALIDATION_ANCHORS,
    VALIDATION_GATE,
    FrictionAnchor,
    IceResistanceCase,
    IceSteelFrictionModel,
    attach_friction_field,
    calibrated_envelope,
    dem_friction_probe,
    fit_ice_steel_friction,
    holdout_error_table,
    ice_resistance,
    kfold_validation,
    leave_one_source_out,
    negative_control_table,
    operating_envelope,
    sensitivity_table,
)


@pytest.fixture(scope="module")
def thermal() -> IceSteelFrictionModel:
    return fit_ice_steel_friction(form="loglinear+thermal")


@pytest.fixture(scope="module")
def loglinear() -> IceSteelFrictionModel:
    return fit_ice_steel_friction(form="loglinear")


# ---------------------------------------------------------------------------
# 1. Identifiability of the closed-form fit
# ---------------------------------------------------------------------------
def _synthetic_anchors(model: IceSteelFrictionModel, jitter: float = 0.0):
    rows = []
    for index, row in enumerate(CALIBRATION_ANCHORS):
        mu = model.mu(row.v_m_per_s, row.temperature_c, row.pressure_mpa)
        rows.append(FrictionAnchor(row.v_m_per_s, row.temperature_c, row.pressure_mpa,
                                   mu * (1.0 + jitter * ((index % 3) - 1)),
                                   row.sigma, "synthetic"))
    return tuple(rows)


def test_fit_recovers_known_loglinear_coefficients():
    target = IceSteelFrictionModel(form="loglinear", mu_ref=0.104, a_log_v=-0.031,
                                   b_temperature=-0.0031, c_temperature2=0.0,
                                   d_log_p=-0.018)
    recovered = fit_ice_steel_friction(_synthetic_anchors(target), form="loglinear")
    assert recovered.mu_ref == pytest.approx(target.mu_ref, abs=1e-12)
    assert recovered.a_log_v == pytest.approx(target.a_log_v, abs=1e-12)
    assert recovered.b_temperature == pytest.approx(target.b_temperature, abs=1e-12)
    assert recovered.d_log_p == pytest.approx(target.d_log_p, abs=1e-12)
    assert recovered.fit_max_rel < 1e-10


def test_fit_recovers_known_thermal_curvature():
    target = IceSteelFrictionModel(form="loglinear+thermal", mu_ref=0.101,
                                   a_log_v=-0.027, b_temperature=-0.0035,
                                   c_temperature2=-4.0e-5, d_log_p=-0.02)
    recovered = fit_ice_steel_friction(_synthetic_anchors(target),
                                       form="loglinear+thermal")
    assert recovered.c_temperature2 == pytest.approx(target.c_temperature2, abs=1e-12)
    assert recovered.fit_max_rel < 1e-10


def test_reference_point_equals_fitted_intercept(thermal):
    assert thermal.mu(thermal.v_ref, thermal.t_ref_c, thermal.p_ref_mpa) == pytest.approx(
        thermal.mu_ref, abs=1e-12)


def test_thermal_form_fits_the_compiled_set_better_than_the_campaign_form(thermal, loglinear):
    assert thermal.fit_rms_rel < loglinear.fit_rms_rel
    assert thermal.fit_max_rel < loglinear.fit_max_rel


# ---------------------------------------------------------------------------
# 2. Physical trends
# ---------------------------------------------------------------------------
def test_speed_weakening_is_monotone(thermal):
    speeds = np.logspace(-4, 0, 17)
    values = [thermal.mu(float(v), -10.0, 0.2) for v in speeds]
    assert all(b < a for a, b in zip(values[:-1], values[1:]))


def test_colder_ice_is_frictionier(thermal):
    temperatures = [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, -2.0]
    values = [thermal.mu(0.01, t, 0.2) for t in temperatures]
    assert all(b < a for a, b in zip(values[:-1], values[1:]))


def test_pressure_weakening_is_monotone(thermal):
    pressures = np.logspace(-1.5, 0.5, 13)
    values = [thermal.mu(0.01, -10.0, float(p)) for p in pressures]
    assert all(b < a for a, b in zip(values[:-1], values[1:]))


def test_coefficient_is_positive_everywhere_in_the_envelope(thermal):
    for v, temperature, pressure in operating_envelope():
        assert thermal.mu(v, temperature, pressure) >= MU_FLOOR


# ---------------------------------------------------------------------------
# 3. Validation protocol
# ---------------------------------------------------------------------------
def test_independent_holdout_passes_the_gate(thermal):
    table = holdout_error_table(thermal)
    assert table["n"] == len(VALIDATION_ANCHORS) == 5
    assert table["max_rel_error"] <= VALIDATION_GATE
    assert table["passes_gate"]


def test_campaign_four_parameter_form_is_flagged_on_the_near_melting_row(loglinear):
    """Honest record of why the campaign form was extended.

    The four-parameter law exceeds the 10 % gate on the near-melting validation
    row; that failure is the documented reason for the thermal curvature term.
    """
    table = holdout_error_table(loglinear)
    assert table["passes_gate"] is False
    assert table["max_rel_error"] > VALIDATION_GATE
    worst = max(table["rows"], key=lambda row: row["rel_error"])
    assert worst["anchor"]["part"] == "validation"


def test_kfold_validation_passes_for_both_forms(thermal, loglinear):
    for model in (thermal, loglinear):
        fold = kfold_validation(form=model.form, k=5)
        assert fold["passes_gate"]
        assert fold["pooled_max_rel_error"] <= VALIDATION_GATE


def test_leave_one_source_out_reports_its_own_degeneracy(thermal):
    """Each source isolates one variable: removing it makes a coefficient blind.

    The protocol must say so (``identifiable=False``) instead of quoting a huge
    error as if it were a model failure.
    """
    loso = leave_one_source_out(CALIBRATION_ANCHORS, form=thermal.form)
    assert len(loso["folds"]) == 3
    assert all(fold["identifiable"] is False for fold in loso["folds"])
    assert loso["identifiable_folds"] == []
    assert math.isnan(loso["pooled_max_rel_error_identifiable_folds"])


def test_extrapolation_probes_are_outside_the_power_density_envelope(thermal):
    coverage = thermal.coverage(0.2, -25.0, 2.0)
    assert coverage["inside"]["v"] and coverage["inside"]["pressure"]
    assert coverage["inside"]["power_density"] is False
    assert coverage["fully_inside"] is False
    table = holdout_error_table(thermal, EXTRAPOLATION_PROBES)
    assert table["n"] == 2
    assert table["max_rel_error"] > VALIDATION_GATE   # documented limitation, not hidden
    envelope = calibrated_envelope()
    assert envelope["power_density_w_per_m2"][1] == pytest.approx(2.0e5)


# ---------------------------------------------------------------------------
# 4. Domain guards, tensor path, coupling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    dict(v=0.0, temperature_c=-10.0, pressure_mpa=0.2),
    dict(v=-1e-3, temperature_c=-10.0, pressure_mpa=0.2),
    dict(v=1e9, temperature_c=-10.0, pressure_mpa=0.2),
    dict(v=0.01, temperature_c=0.0, pressure_mpa=0.2),
    dict(v=0.01, temperature_c=5.0, pressure_mpa=0.2),
    dict(v=0.01, temperature_c=-10.0, pressure_mpa=0.0),
    dict(v=0.01, temperature_c=-10.0, pressure_mpa=float("nan")),
])
def test_domain_guards_reject_unphysical_arguments(thermal, kwargs):
    with pytest.raises(ValueError):
        thermal.mu(**kwargs)


def test_floor_binds_only_outside_the_validated_envelope(thermal):
    diagnostic = thermal.mu_diagnostics(2.0, -2.0, 2.0)
    assert diagnostic["raw_min"] < MU_FLOOR
    assert diagnostic["floor_binds"] == 1
    assert float(thermal.mu(2.0, -2.0, 2.0)) == pytest.approx(MU_FLOOR)
    inside = thermal.mu_diagnostics([row.v_m_per_s for row in CALIBRATION_ANCHORS],
                                    [row.temperature_c for row in CALIBRATION_ANCHORS],
                                    [row.pressure_mpa for row in CALIBRATION_ANCHORS])
    assert inside["floor_binds"] == 0


def test_tensor_and_scalar_paths_agree(thermal):
    speeds = torch.tensor([1e-3, 1e-2, 0.1, 1.0], dtype=torch.float64)
    temperatures = torch.tensor([-20.0, -10.0, -5.0, -2.0], dtype=torch.float64)
    pressures = torch.full((4,), 0.5, dtype=torch.float64)
    tensor_values = thermal.mu_tensor(speeds, temperatures, pressures)
    scalar_values = [thermal.mu(float(v), float(t), float(p))
                     for v, t, p in zip(speeds, temperatures, pressures)]
    assert tensor_values.dtype == torch.float64
    assert np.allclose(tensor_values.numpy(), scalar_values, rtol=1e-12)


def test_resistance_decomposition_is_consistent(thermal):
    case = IceResistanceCase()
    row = ice_resistance(0.1, -10.0, 0.5, thermal, case=case)
    assert row["total_resistance_n"] == pytest.approx(
        row["crushing_resistance_n"] + row["friction_resistance_n"], rel=1e-12)
    assert row["friction_resistance_n"] == pytest.approx(
        row["mu"] * row["normal_load_n"], rel=1e-12)
    assert 0.0 < row["friction_share"] < 1.0


def test_sensitivity_is_monotone_and_bounded(thermal):
    table = sensitivity_table(thermal)
    assert table["all_monotone_increase"]
    assert table["all_monotone_increase_inside_envelope"]
    for row in table["rows"]:
        totals = [row["totals_by_scale"][key] for key in sorted(row["totals_by_scale"])]
        assert all(later > earlier for earlier, later in zip(totals[:-1], totals[1:]))
        assert abs(row["relative_resistance_change"]) <= 0.2 + 1e-12


def test_negative_control_quantifies_the_constant_mu_error(thermal):
    control = negative_control_table(thermal)
    assert control["constant_mu"] == pytest.approx(MU_CONSTANT_BASELINE)
    assert control["in_envelope"]["n"] >= 20
    assert control["in_envelope"]["max_abs_mu_rel_error"] > 0.5
    assert control["in_envelope"]["max_abs_resistance_rel_error"] > 0.1
    assert control["over_predicts_count"] > 0
    assert control["n_floor_bound"] >= 1


def test_dem_realises_the_calibrated_coefficient(thermal):
    # default resolution (1200 slip steps after 4000 settling steps): a coarser
    # run leaves a tangential transient that is 4 % of the slope, which is the
    # measurement error of the probe, not of the contact law.
    probe = dem_friction_probe(thermal, v=0.1, temperature_c=-10.0, pressure_mpa=0.5)
    assert probe["passes_dem_gate"]
    assert probe["slope_rel_error"] <= probe["gate"]
    assert probe["mu_rel_error"] <= probe["gate"]


def test_attach_friction_field_preserves_contact_and_sets_the_law(thermal):
    base = ContactParameters()
    updated = attach_friction_field(base, thermal, v=0.1, temperature_c=-10.0,
                                    pressure_mpa=0.5)
    assert updated.friction == pytest.approx(thermal.mu(0.1, -10.0, 0.5), rel=1e-15)
    assert updated.crushing_strength == base.crushing_strength
    assert updated.reference_penetration_fraction == base.reference_penetration_fraction
    assert updated.normal_damping_ratio == base.normal_damping_ratio
    assert updated.tangential_damping_ratio == base.tangential_damping_ratio
    with pytest.raises(TypeError):
        attach_friction_field(object(), thermal, v=0.1, temperature_c=-10.0,
                             pressure_mpa=0.5)