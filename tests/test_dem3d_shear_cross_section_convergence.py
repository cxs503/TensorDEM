"""Tests for the DEM3D direct-shear cross-section convergence study.

Combines the shear_convergence module tests (main) with the interior_traction
and window-based measurement tests (agent/m2772).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.shear_specimen import lattice_spacing, shear_modulus_closed_form
from tensordem.shear_static import (
    cross_section_series,
    fit_boundary_bias,
    fit_boundary_layer,
    interior_traction,
    static_direct_shear,
)
from tensordem.shear_convergence import (
    ACCURACY_GATE,
    boundary_layer_fit,
    cross_section_series as convergence_cross_section_series,
    surface_lag_profile,
)

# ---------------------------------------------------------------------------
# Tests from main (shear_convergence module)
# ---------------------------------------------------------------------------

SIZES = [5, 7, 9, 11, 13]


def test_series_is_monotone_and_conserves_the_affine_census():
    series = convergence_cross_section_series(SIZES)
    rows = series["rows"]
    deficits = [row["deficit"] for row in rows]
    assert all(d > 0.0 for d in deficits)
    assert all(b < a for a, b in zip(deficits, deficits[1:]))          # deficit shrinks
    areas = [row["area_ratio"] for row in rows]
    assert all(b > a for a, b in zip(areas, areas[1:]))                # census -> nominal
    assert all(a < 1.0 for a in areas)
    # the static solve reproduces the solver's own reaction at every size
    assert max(row["reaction_relative_error"] for row in rows) < 1.0e-5
    assert max(row["free_residual_over_bond_force"] for row in rows) < 5.0e-2


def test_boundary_layer_fit_is_first_order_with_no_asymptotic_bias():
    series = convergence_cross_section_series(SIZES)
    fit = series["fit"]
    model_b = fit["models"]["B"]
    # a pure 1/n boundary layer already explains the series to well below 1 %
    assert model_b["rms_residual"] < 2.0e-3
    assert model_b["max_abs_residual"] < 5.0e-3
    # observed order must be close to one (first-order surface deficit)
    assert 0.6 < fit["observed_order"] < 2.5
    # the falsification model (constant + 1/n + 1/n^2) must not find a large bias
    assert abs(fit["asymptotic_deficit_model_C"]) < 5.0e-2
    assert abs(fit["extrapolated_G_over_C44_model_C"] - 1.0) < 5.0e-2


def test_gate_prediction_sits_above_the_fitted_range():
    fit = convergence_cross_section_series(SIZES)["fit"]
    assert np.isfinite(fit["predicted_n_for_gate"])
    assert fit["predicted_n_for_gate"] > max(SIZES)
    assert fit["predicted_n_for_gate"] < 200.0


def test_wider_cross_section_reduces_the_deficit_by_the_predicted_ratio():
    series = convergence_cross_section_series([5, 9, 13, 21])
    rows = {row["n"]: row for row in series["rows"]}
    predicted = rows[5]["deficit"] * (5.0 / 21.0)
    assert 0.5 * predicted < rows[21]["deficit"] < 1.5 * predicted


def test_surface_lag_decays_inside_a_few_lattice_spacings():
    profile = surface_lag_profile(n=17)
    bins = {b["distance_over_spacing"]: b["mean_lag"] for b in profile["bins"]}
    assert abs(profile["lag_at_center"]) < 1.0e-3          # far field is affine
    assert bins[1.0] < -0.03                                # first interior row lags
    assert bins[1.0] < bins[2.0] < bins[3.0] < 0.0          # monotone recovery
    assert 0.5 < profile["decay_length_over_spacing"] < 6.0


def test_displacement_field_is_optional_and_consistent():
    cfg = replace(DEM3DConfig(), nx=5, ny=5, nz=3, drag=0.0, fix_x_edges=False)
    plain = static_direct_shear(cfg)
    field = static_direct_shear(cfg, return_displacement=True)
    assert "displacement_m" not in plain
    u = np.asarray(field["displacement_m"])
    positions = np.asarray(field["initial_positions_m"])
    assert u.shape == positions.shape == (75, 3)
    # top layer is prescribed, bottom layer is fixed
    top = np.isclose(positions[:, 2], positions[:, 2].max())
    bottom = np.isclose(positions[:, 2], positions[:, 2].min())
    h = positions[:, 2].max() - positions[:, 2].min()
    assert np.allclose(u[top, 0], 1.0e-3 * h, rtol=1e-9, atol=0)
    assert np.allclose(u[bottom], 0.0, atol=1e-12)
    assert field["G_effective_over_C44"] == pytest.approx(plain["G_effective_over_C44"])


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        convergence_cross_section_series([5, 7])
    with pytest.raises(ValueError):
        convergence_cross_section_series([9, 7, 5, 11])
    with pytest.raises(ValueError):
        boundary_layer_fit([5, 7, 9, 11], [0.3, 0.2, 0.25, 0.05])  # deficit must decrease
    with pytest.raises(ValueError):
        surface_lag_profile(n=5, config=replace(DEM3DConfig(), nz=4))  # even nz: no mid layer
    assert ACCURACY_GATE == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Tests from agent/m2772 (interior_traction, window-based measurements)
# ---------------------------------------------------------------------------

GAMMA = 1.0e-6


def _cfg(nx, ny, nz):
    return replace(DEM3DConfig(), nx=nx, ny=ny, nz=nz)


def test_interior_window_is_exactly_c44_under_affine_shear():
    """The windowed bond census/force bookkeeping is exact, not calibrated."""
    for nx, ny in ((9, 9), (17, 13)):
        cfg = _cfg(nx, ny, 3)
        sim = IceDEM3D(cfg)
        pos = sim.initial_positions.cpu().numpy()
        u = np.zeros_like(pos)
        u[:, 0] = GAMMA * pos[:, 2]
        res = interior_traction(sim, u, gamma=GAMMA, margin=2)
        assert res["G_window_over_C44"] == pytest.approx(1.0, rel=1.0e-5)
        assert res["bond_count"] > 0.0


def test_whole_layer_modulus_increases_with_cross_section():
    """Lateral free surfaces soften the specimen: wider is stiffer."""
    rows = cross_section_series([(5, 5, 3), (9, 9, 3), (13, 13, 3)], gamma=GAMMA,
                                margin=2)
    values = [r["G_effective_over_C44"] for r in rows]
    assert values == sorted(values)
    assert values[0] < 0.85 and values[-1] > 0.85
    for r in rows:
        assert r["G_nominal_over_C44"] < r["G_effective_over_C44"]   # area bias
        assert r["reaction_relative_error"] < 1.0e-6                 # equilibrium


def test_quadratic_extrapolation_recovers_the_analytic_c44():
    """1/n boundary-layer extrapolation of the square family reaches 7k/(3a)."""
    sizes = [(9, 9, 3), (13, 13, 3), (17, 17, 3), (21, 21, 3), (25, 25, 3)]
    rows = cross_section_series(sizes, gamma=GAMMA, margin=2)
    fit = fit_boundary_bias(rows, degree=2)
    assert fit["r2"] > 0.999
    assert fit["limit_over_C44"] == pytest.approx(1.0, abs=3.0e-2)
    linear = fit_boundary_bias(rows, degree=1)
    assert linear["limit_over_C44"] == pytest.approx(1.0, abs=3.0e-2)


def test_thin_specimen_interior_window_recovers_c44_within_one_percent():
    cfg = _cfg(21, 21, 3)
    payload, sim, u = static_direct_shear(cfg, gamma=GAMMA, include_state=True)
    window = interior_traction(sim, u, gamma=GAMMA, margin=3)
    assert abs(window["G_window_over_C44"] - 1.0) < 0.01
    # the whole-layer value is the edge-contaminated one, markedly lower
    assert payload["G_effective_over_C44"] < window["G_window_over_C44"] - 0.02
    assert window["margin_sites"] == 3.0


def test_window_measure_is_linear_in_gamma():
    cfg = _cfg(17, 17, 3)
    values = []
    for gamma in (1.0e-7, 1.0e-6):
        _, sim, u = static_direct_shear(cfg, gamma=gamma, include_state=True)
        values.append(interior_traction(sim, u, gamma=gamma, margin=3)["G_window_over_C44"])
    assert values[0] == pytest.approx(values[1], rel=1.0e-8)


def test_thick_specimen_modulus_is_a_specimen_property_not_a_material_constant():
    """h>=3a: shear lag fills the cross-section, so no interior window helps."""
    tall = cross_section_series([(21, 21, 3), (21, 21, 7)], gamma=GAMMA, margin=2)
    short, high = tall
    assert high["G_effective_over_C44"] < short["G_effective_over_C44"] - 0.05
    cfg = _cfg(21, 21, 7)
    _, sim, u = static_direct_shear(cfg, gamma=GAMMA, include_state=True)
    mid = interior_traction(sim, u, gamma=GAMMA, margin=3, layer=3)
    assert mid["G_window_over_C44"] < 0.99        # still no C44 in the middle
    assert mid["layer"] == 3.0


def test_layer_argument_rejects_out_of_range_indices():
    cfg = _cfg(9, 9, 3)
    _, sim, u = static_direct_shear(cfg, gamma=GAMMA, include_state=True)
    with pytest.raises(ValueError):
        interior_traction(sim, u, gamma=GAMMA, margin=1, layer=5)


def test_fit_boundary_layer_recovers_a_synthetic_decay_length():
    margins = [1, 2, 3, 4, 5]
    lam, amp = 1.7, 0.2

    def value(m):
        return 1.0 - amp * np.exp(-m / lam)

    fit = fit_boundary_layer(margins, [value(m) for m in margins])
    assert fit["decay_length_sites"] == pytest.approx(lam, rel=1.0e-6)
    assert fit["r2"] > 0.999


def test_margin_is_clamped_on_specimens_too_small_for_the_window():
    (row,) = cross_section_series([(5, 5, 3)], gamma=GAMMA, margin=3)
    assert row["window_margin_used"] == 1
    assert np.isfinite(row["G_window_over_C44"])
    assert row["window_fraction_of_area"] > 0.0


def test_lattice_spacing_and_c44_reference_are_the_documented_closed_forms():
    cfg = DEM3DConfig()
    assert lattice_spacing(cfg) == pytest.approx(2.0 * cfg.radius)
    assert shear_modulus_closed_form(cfg) == pytest.approx(
        7.0 * cfg.bond_stiffness / (3.0 * lattice_spacing(cfg)))
