"""Tests for the direct-shear cross-section convergence study (mite closure of
the "material shear modulus" WITHHELD item in
``docs/dem3d_shear_static_reference.md``)."""
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