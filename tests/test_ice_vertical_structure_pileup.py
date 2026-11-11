"""Tests for the 2-D ice pile-up / ISO 19906 module (campaign p227w2-N04).

Run with::

    python -m pytest tests/test_ice_vertical_structure_pileup.py -q

The DEM smoke tests use a deliberately short run (0.6 s) so the whole file stays
well under a minute; the full campaign lives in
``benchmarks/ice_pileup_iso19906/run_benchmark.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_vertical_structure_pileup import (
    PileupConfig,
    fit_porosity_scaling,
    iso19906_load,
    make_floe_field,
    measure_pile_porosity,
    normal_force,
    run_pileup_case,
)


# ---------------------------------------------------------------------------
# ISO 19906 reference load
# ---------------------------------------------------------------------------
def test_iso19906_pressure_factors_and_load():
    """p_G follows C_R (h/h_ref)^n (w/h)^m and F = p_G w h."""
    res = iso19906_load(sigma_c=3.0e6, h=1.0, D=5.0, shape="cylinder")
    n_exp = max(-0.5 + 1.0 / 10.0, -0.25)
    p_expected = 2.8e6 * (1.0) ** n_exp * (5.0 / 1.0) ** (-0.16)
    assert res.p_G == pytest.approx(p_expected, rel=1e-12)
    assert res.F_ISO == pytest.approx(res.p_G * 5.0 * 1.0, rel=1e-12)
    # campaign form F = C sigma_c h^2 must reproduce the same force
    assert res.C_task * 3.0e6 * 1.0 ** 2 == pytest.approx(res.F_ISO, rel=1e-12)
    assert not res.pressure_capped


def test_iso19906_pressure_is_capped_by_ice_strength():
    weak = iso19906_load(sigma_c=0.5e6, h=1.0, D=5.0)
    assert weak.pressure_capped
    assert weak.p_G == pytest.approx(0.5e6)
    assert weak.F_ISO == pytest.approx(0.5e6 * 5.0 * 1.0)


def test_iso19906_load_grows_with_thickness():
    base = iso19906_load(3.0e6, 1.0, 5.0)
    thick = iso19906_load(3.0e6, 2.0, 5.0)
    # exponents n, m keep the load increasing with thickness
    assert thick.F_ISO > base.F_ISO


def test_iso19906_pressure_is_capped_by_strength_only():
    """ISO 19906 pressure comes from C_R; sigma_c only caps it.

    Strong ice (sigma_c > p_G): the load does not depend on sigma_c.
    Weak ice (sigma_c < p_G): the load is proportional to sigma_c.
    """
    strong_a = iso19906_load(3.0e6, 1.0, 5.0)
    strong_b = iso19906_load(6.0e6, 1.0, 5.0)
    assert strong_a.F_ISO == pytest.approx(strong_b.F_ISO, rel=1e-12)
    weak_a = iso19906_load(1.0e6, 1.0, 5.0)
    weak_b = iso19906_load(2.0e6, 1.0, 5.0)
    assert weak_b.F_ISO == pytest.approx(2.0 * weak_a.F_ISO, rel=1e-12)


def test_iso19906_rejects_bad_input():
    with pytest.raises(ValueError):
        iso19906_load(3.0e6, 0.0, 5.0)
    with pytest.raises(ValueError):
        iso19906_load(3.0e6, 1.0, 5.0, shape="ellipse")


# ---------------------------------------------------------------------------
# crushing-limited contact law
# ---------------------------------------------------------------------------
def test_normal_force_is_crushing_capped():
    sigma_c, h, k_n = 3.0e6, 1.0, 9.0e7
    r_eff = np.array([0.4])
    deltas = np.array([1e-4, 1e-3, 5e-3])
    fn = normal_force(sigma_c, h, r_eff, deltas, k_n)
    cap = sigma_c * h * 2.0 * r_eff[0]  # <= full-diameter contact chord
    assert np.all(fn > 0)
    assert np.all(fn <= cap + 1e-6)
    # linear spring branch dominates at tiny overlap
    assert fn[0] == pytest.approx(k_n * deltas[0], rel=1e-9)
    assert np.all(np.diff(fn) > 0)  # monotone in penetration


def test_normal_force_hard_core_prevents_tunnelling():
    sigma_c, h, k_n, r_eff = 3.0e6, 1.0, 9.0e7, 0.4
    f_cap = normal_force(sigma_c, h, np.array([r_eff]),
                         np.array([0.20 * r_eff]), k_n)[0]
    f_deep = normal_force(sigma_c, h, np.array([r_eff]),
                          np.array([0.60 * r_eff]), k_n)[0]
    assert f_deep > f_cap * 4.0


# ---------------------------------------------------------------------------
# floe field generation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["cylinder", "square"])
def test_floe_field_is_clear_of_the_structure(shape):
    cfg = PileupConfig(shape=shape, ice_concentration=0.8)
    pos, rad, phi = make_floe_field(cfg, np.random.default_rng(3))
    assert rad.size > 20
    assert 0.0 < phi <= 1.0
    a = 0.5 * cfg.D
    if shape == "cylinder":
        assert np.all(np.hypot(pos[:, 0], pos[:, 1]) - rad >= -1e-9)
    else:
        dx = np.abs(pos[:, 0]) - a
        dy = np.abs(pos[:, 1]) - a
        outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
        assert np.all(outside + np.minimum(np.maximum(dx, dy), 0.0) - rad >= -1e-9)


def test_floe_field_concentration_is_controllable():
    dense_pos, dense_rad, dense_phi = make_floe_field(
        PileupConfig(ice_concentration=0.85), np.random.default_rng(5))
    loose_pos, loose_rad, loose_phi = make_floe_field(
        PileupConfig(ice_concentration=0.45), np.random.default_rng(5))
    assert dense_phi > loose_phi
    assert dense_rad.size > loose_rad.size


# ---------------------------------------------------------------------------
# porosity-resistance fit
# ---------------------------------------------------------------------------
def test_fit_recovers_synthetic_exponent():
    n = np.linspace(0.15, 0.45, 8)
    F = 4.0e6 * (1.0 - n) ** 1.6
    fit = fit_porosity_scaling(n, F, n_boot=200, seed=1)
    assert fit["alpha"] == pytest.approx(1.6, rel=1e-6)
    assert fit["amplitude"] == pytest.approx(4.0e6, rel=1e-6)
    assert fit["R2"] == pytest.approx(1.0, abs=1e-9)
    lo, hi = fit["alpha_ci95"]
    assert lo <= fit["alpha"] <= hi


def test_fit_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_porosity_scaling([0.3, 0.4], [1.0, 2.0], n_boot=10)


# ---------------------------------------------------------------------------
# DEM smoke tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["cylinder", "square"])
def test_short_pileup_run_is_stable_and_bounded(shape):
    """A short run must stay finite, keep contacts non-negative and build a pile."""
    cfg = PileupConfig(duration=0.6, seed=4, shape=shape, ice_concentration=0.8)
    res = run_pileup_case(cfg)
    assert np.all(np.isfinite(res.load_x))
    assert res.F_max >= 0.0
    assert res.ratio_max < 10.0  # within one order of magnitude of ISO 19906
    assert 0.0 < res.porosity < 1.0
    assert res.n_floes_initial > 20
    assert res.platen_travel > 0.0


def test_pileup_run_is_deterministic_for_a_seed():
    cfg = PileupConfig(duration=0.4, seed=9)
    a = run_pileup_case(cfg)
    b = run_pileup_case(cfg)
    assert a.F_max == b.F_max
    assert a.porosity == b.porosity
    assert np.allclose(a.load_x, b.load_x)


def test_measure_pile_porosity_matches_independent_grid():
    """The rasterised window porosity must match a fine independent grid."""
    cfg = PileupConfig(D=5.0, grid_cell=0.02)
    R = 0.5 * cfg.D
    rng = np.random.default_rng(0)
    n = 12
    xy = np.column_stack([rng.uniform(-R - cfg.D + 0.5, -R - 0.3, n),
                          rng.uniform(-3.0, 3.0, n)])
    rr = np.full(n, 0.35)
    out = measure_pile_porosity(xy, rr, cfg)

    # independent fine-grid estimate of the coverage of the same window
    s = 0.005
    gx = np.arange(-R - cfg.D, -R, s)
    gy = np.arange(-cfg.lateral_factor * cfg.D, cfg.lateral_factor * cfg.D, s)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    solid = np.zeros_like(X, dtype=bool)
    for (px, py), r0 in zip(xy, rr):
        solid |= (X - px) ** 2 + (Y - py) ** 2 <= r0 * r0
    ref = 1.0 - solid.mean()
    assert out["porosity_window"] == pytest.approx(ref, abs=0.01)
    assert 0.0 <= out["porosity"] <= 1.0
    assert out["solids_fraction"] == pytest.approx(1.0 - out["porosity"], rel=1e-12)
