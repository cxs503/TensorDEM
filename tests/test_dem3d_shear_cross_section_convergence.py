"""Tests for the DEM3D direct-shear cross-section convergence study."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tensordem.dem3d import DEM3DConfig
from tensordem.shear_convergence import (
    ACCURACY_GATE,
    boundary_layer_fit,
    cross_section_series,
    surface_lag_profile,
)
from tensordem.shear_static import static_direct_shear

SIZES = [5, 7, 9, 11, 13]


def test_series_is_monotone_and_conserves_the_affine_census():
    series = cross_section_series(SIZES)
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
    series = cross_section_series(SIZES)
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
    fit = cross_section_series(SIZES)["fit"]
    assert np.isfinite(fit["predicted_n_for_gate"])
    assert fit["predicted_n_for_gate"] > max(SIZES)
    assert fit["predicted_n_for_gate"] < 200.0


def test_wider_cross_section_reduces_the_deficit_by_the_predicted_ratio():
    series = cross_section_series([5, 9, 13, 21])
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
        cross_section_series([5, 7])
    with pytest.raises(ValueError):
        cross_section_series([9, 7, 5, 11])
    with pytest.raises(ValueError):
        boundary_layer_fit([5, 7, 9, 11], [0.3, 0.2, 0.25, 0.05])  # deficit must decrease
    with pytest.raises(ValueError):
        surface_lag_profile(n=5, config=replace(DEM3DConfig(), nz=4))  # even nz: no mid layer
    assert ACCURACY_GATE == pytest.approx(0.03)