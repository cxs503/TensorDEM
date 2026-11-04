"""Engine-level verification of the explicit time-step stability threshold.

The claims under test are the ones ``docs/dem3d_timestep_stability.md`` listed as
``【未验证】``: that the solver itself, not only the closed-form amplification
factor, crosses from bounded to growing at ``dt_crit = 2 / omega_max``.
"""
import math

import pytest
import torch

from tensordem import stability
from tensordem import stability_verification as sv
from tensordem.dem3d import DEM3DConfig, IceDEM3D


# ---------------------------------------------------------------------------
# The guard rail is relaxed only on request
# ---------------------------------------------------------------------------
def test_default_config_keeps_the_timestep_guard():
    config = DEM3DConfig()
    assert config.allow_dt_above_recommended is False
    with pytest.raises(ValueError, match="dt exceeds conservative limit"):
        IceDEM3D(DEM3DConfig(dt=10.0 * DEM3DConfig().recommended_dt))
    # A normal run never reports a bypass.
    assert IceDEM3D(DEM3DConfig()).dt_guard_bypassed is False


def test_escape_hatch_is_explicit_and_recorded():
    config = DEM3DConfig(dt=1.5e-3, allow_dt_above_recommended=True)
    assert config.dt > config.recommended_dt
    model = IceDEM3D(config)
    assert model.dt_guard_bypassed is True
    # Below the guard the flag is inert.
    assert IceDEM3D(DEM3DConfig(dt=0.5 * config.recommended_dt,
                                allow_dt_above_recommended=True)).dt_guard_bypassed is False
    with pytest.raises(ValueError, match="must be a boolean"):
        DEM3DConfig(allow_dt_above_recommended="yes")


# ---------------------------------------------------------------------------
# omega_max: solver-free assembly, engine force routine, Gershgorin bound
# ---------------------------------------------------------------------------
def test_engine_force_law_matches_independent_stiffness_assembly():
    model = sv.reference_model()
    analytic = sv.top_mode_analytic(model)
    engine = sv.engine_rayleigh_omega(model, analytic.eigenvector)
    assert engine["residual"] < 1.0e-8, "returned field is not an exact mode"
    assert abs(engine["omega_rad_s"] - analytic.omega_rad_s) / analytic.omega_rad_s < 1.0e-8
    assert engine["omega_rad_s"] > 0.0


def test_stiffness_matrix_is_symmetric_and_positive_semidefinite():
    model = sv.reference_model()
    k_matrix = sv.assemble_stiffness(model)
    assert torch.allclose(k_matrix, k_matrix.T, atol=1.0e-9)
    lowest = float(torch.linalg.eigvalsh(k_matrix).min())
    assert lowest > -1.0e-6 * float(torch.linalg.eigvalsh(k_matrix).max())
    # Free-free lattice: three rigid translations, so the smallest eigenvalue is 0.
    assert abs(lowest) < 1.0e-6 * float(torch.linalg.eigvalsh(k_matrix).max())


def test_gershgorin_bound_brackets_the_top_mode():
    model = sv.reference_model()
    analytic = sv.top_mode_analytic(model)
    bound = stability.stability_report(model)["omega_upper_rad_s"]
    assert analytic.omega_rad_s <= bound * (1.0 + 1.0e-12)


# ---------------------------------------------------------------------------
# Measured growth across the threshold
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sweep():
    model = sv.reference_model()
    analytic = sv.top_mode_analytic(model)
    # short fraction list keeps the suite fast; the full audit sweeps further.
    return sv.threshold_sweep(model, analytic=analytic, fractions=(0.5, 0.99, 1.01, 1.2))


def test_measured_growth_matches_the_closed_form(sweep):
    for row in sweep["rows"]:
        assert row["relative_error"] < 1.0e-6, row
        assert row["timestep_guard_would_refuse"] is True, (
            "the audited steps must be ones the production guard refuses")


def test_bounded_below_and_growing_above_the_threshold(sweep):
    for row in sweep["rows"]:
        expected_growth = row["dt_over_critical"] > 1.0
        assert row["grows"] is expected_growth, row


def test_crossing_found_from_engine_measurements_alone(sweep):
    crossing = sweep["crossing"]
    assert crossing["relative_error"] < 1.0e-6, crossing
    assert crossing["bracket_width_s"] > 0.0
    assert crossing["iterations"] < 60


# ---------------------------------------------------------------------------
# Long runs: marginal below, exponential above
# ---------------------------------------------------------------------------
def test_long_run_above_threshold_follows_rho_power_n():
    model = sv.reference_model()
    analytic = sv.top_mode_analytic(model)
    unit_mode = analytic.eigenvector / torch.linalg.vector_norm(analytic.eigenvector)
    critical = stability.elastic_timestep_limit(analytic.omega_rad_s)
    hot = sv.long_run_growth(1.05 * critical, unit_mode, analytic.omega_rad_s,
                             steps=60, amplitude_limit=10.0)
    assert hot["steps"] >= 8
    assert hot["growth_per_step_relative_error"] < 1.0e-6, hot
    assert abs(hot["measured_amplitude_final"] - hot["predicted_amplitude_after_steps"]) / \
        hot["predicted_amplitude_after_steps"] < 1.0e-6


def test_long_run_below_threshold_preserves_the_quadratic_invariant():
    model = sv.reference_model()
    analytic = sv.top_mode_analytic(model)
    unit_mode = analytic.eigenvector / torch.linalg.vector_norm(analytic.eigenvector)
    critical = stability.elastic_timestep_limit(analytic.omega_rad_s)
    calm = sv.long_run_growth(0.95 * critical, unit_mode, analytic.omega_rad_s, steps=600)
    assert calm["steps"] == 600
    assert calm["invariant_relative_drift"] < 1.0e-5, calm
    # The plain energy is not conserved by the scheme; the invariant is.
    assert calm["rho_closed_form"] == pytest.approx(1.0, abs=1.0e-12)


# ---------------------------------------------------------------------------
# Closed-form helper algebra
# ---------------------------------------------------------------------------
def test_step_matrix_reproduces_the_published_growth_factor():
    omega = 600.0
    for fraction in (0.25, 0.9, 1.0, 1.1, 1.4):
        dt = fraction * 2.0 / omega
        matrix = sv.closed_form_step_matrix(dt, omega)
        trace = float(torch.trace(matrix))
        det = float(torch.linalg.det(matrix))
        discriminant = trace * trace - 4.0 * det
        rho = (max(abs(0.5 * (trace + math.sqrt(discriminant))),
                   abs(0.5 * (trace - math.sqrt(discriminant))))
               if discriminant >= 0.0 else math.sqrt(abs(det)))
        assert rho == pytest.approx(sv.closed_form_rho(dt, omega), rel=1.0e-12)


def test_invariant_form_is_exact_and_requires_a_stable_map():
    omega = 500.0
    dt = 0.8 * 2.0 / omega
    form = sv.invariant_form(dt, omega)
    matrix = sv.closed_form_step_matrix(dt, omega)
    defect = float((matrix.T @ form @ matrix - form).abs().max())
    assert defect < 1.0e-8
    assert float(torch.linalg.eigvalsh(form).min()) > 0.0
    with pytest.raises(ValueError, match="no positive definite invariant"):
        sv.invariant_form(1.1 * 2.0 / omega, omega)