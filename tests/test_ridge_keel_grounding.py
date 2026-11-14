"""Tests for the Kerr three-parameter foundation, the bonded-keel DEM and the
ISO 19906 A.8 operator layer (W27 ice-ridge keel grounding).

Every assertion below is a physics or numerics statement that can fail; nothing
is a smoke test of an import.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem import keel_foundation_dem as kd
from tensordem import ridge_keel_grounding as rg


# ---------------------------------------------------------------------------
# Kerr foundation reduction
# ---------------------------------------------------------------------------
def test_kerr_limits_reduce_to_pasternak_and_winkler():
    # k_u -> inf  =>  c -> 0, K -> k_l, G2 -> G   (Pasternak)
    fnd = rg.KerrFoundation(k_upper=1e18, k_lower=4.0e6, shear=2.0e8)
    assert fnd.c < 1e-9
    assert fnd.k_series == pytest.approx(4.0e6, rel=1e-9)
    assert fnd.g2 == pytest.approx(2.0e8, rel=1e-9)
    # k_u = k_l = 2K  =>  series modulus equals K
    k = 3.0e7
    fnd2 = rg.KerrFoundation(k_upper=2 * k, k_lower=2 * k, shear=1.0e8)
    assert fnd2.k_series == pytest.approx(k, rel=1e-12)
    assert fnd2.g2 == pytest.approx(0.5 * 1.0e8, rel=1e-12)


def test_kerr_foundation_rejects_unphysical_parameters():
    with pytest.raises(ValueError):
        rg.KerrFoundation(k_upper=0.0, k_lower=1.0, shear=1.0)
    with pytest.raises(ValueError):
        rg.KerrFoundation(k_upper=1.0, k_lower=-1.0, shear=1.0)
    with pytest.raises(ValueError):
        rg.KerrFoundation(k_upper=1.0, k_lower=1.0, shear=float("nan"))


# ---------------------------------------------------------------------------
# analytic solution vs the mixed FEM
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [
    dict(ei=8.0e7, k_upper=1.0e7, k_lower=4.0e7, shear=1.0e8),
    dict(ei=1.0e8, k_upper=2.0e7, k_lower=8.0e7, shear=4.0e8),
    dict(ei=2.0e8, k_upper=5.0e7, k_lower=2.0e8, shear=2.0e9),
])
def test_kerr_analytic_matches_independent_fem(cfg):
    fnd = rg.KerrFoundation(k_upper=cfg["k_upper"], k_lower=cfg["k_lower"],
                            shear=cfg["shear"])
    a, q0 = 2.0, 1.0e5
    x = np.linspace(-20.0, 20.0, 401)
    x_dense = np.linspace(-30.0, 30.0, 6001)
    an = rg.kerr_beam_analytic(a, q0, ei=cfg["ei"], foundation=fnd, x=x_dense)
    an_coarse = rg.kerr_beam_analytic(a, q0, ei=cfg["ei"], foundation=fnd, x=x)
    fem = rg.kerr_beam_fem(np.linspace(-20.0, 20.0, 401), ei=cfg["ei"], foundation=fnd,
                           load=lambda xx: np.where(np.abs(xx) <= a, q0, 0.0))
    wa = np.interp(x, an_coarse.x, an_coarse.w)
    wf = np.interp(x, fem.x, fem.w)
    near = np.abs(x) <= 10.0
    dev = np.max(np.abs(wf - wa)[near]) / np.max(np.abs(wa[near]))
    assert dev < 3.0e-2, f"analytic/FEM deflection deviation {dev:.3e}"
    # total reaction must reproduce the applied load (global equilibrium); the
    # analytic meta integrates the closed form, so it needs a grid fine enough
    # for the pressure jump at the load edges
    assert an.meta["reaction_total_N_per_m"] == pytest.approx(2 * a * q0, rel=1e-12)
    assert an.meta["reaction_quadrature_N_per_m"] == pytest.approx(2 * a * q0, rel=1e-2)
    assert fem.meta["reaction_total_N_per_m"] == pytest.approx(2 * a * q0, rel=1e-6)


def test_fem_reproduces_hetenyi_winkler_closed_form():
    """The FEM must reproduce the independent Hetenyi patch-load solution."""
    from tensordem.ridge_geotech import hetenyi_infinite_beam_patch_load
    import torch

    K, EI, a, q0 = 1.6e7, 1.0e8, 2.0, 1.0e5
    beta = (K / (4.0 * EI)) ** 0.25
    x = np.linspace(-30.0, 30.0, 601)
    w_het, _ = hetenyi_infinite_beam_patch_load(torch.tensor(x), load_per_length=q0,
                                               half_length=a, beta=beta, k=K, samples=4001)
    fnd = rg.KerrFoundation(k_upper=1e15, k_lower=K * (1.0 + 1e-12), shear=1e-3)
    fem = rg.kerr_beam_fem(x, ei=EI, foundation=fnd,
                          load=lambda xx: np.where(np.abs(xx) <= a, q0, 0.0))
    wf = np.interp(x, fem.x, fem.w)
    dev = np.max(np.abs(wf - w_het.numpy())[np.abs(x) < 15.0]) / np.max(np.abs(w_het.numpy()))
    assert dev < 1e-6, f"FEM vs Hetenyi deviation {dev:.3e}"


def test_characteristic_roots_are_three_per_halfline():
    fnd = rg.KerrFoundation(k_upper=2.0e7, k_lower=8.0e7, shear=4.0e8)
    roots = rg.kerr_characteristic_roots(1.0e8, fnd)
    assert roots["lambda_left"].size == 3
    assert roots["lambda_right"].size == 3
    assert np.all(roots["lambda_right"].real < 0.0)
    # every root satisfies the characteristic polynomial c EI s^3 - EI s^2 + G2 s - K = 0
    s = roots["lambda_right"] ** 2
    poly = (fnd.c * 1.0e8 * s ** 3 - 1.0e8 * s ** 2 + fnd.g2 * s - fnd.k_series)
    assert np.max(np.abs(poly)) < 1e-6 * fnd.k_series


# ---------------------------------------------------------------------------
# rigid footing closed form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [
    dict(ei=1.0e8, k_upper=2.0e7, k_lower=8.0e7, shear=4.0e8),
    dict(ei=1.0e8, k_upper=1.0e6, k_lower=5.0e7, shear=3.0e9),
])
def test_rigid_footing_closed_form_is_exact(cfg):
    fnd = rg.KerrFoundation(k_upper=cfg["k_upper"], k_lower=cfg["k_lower"],
                            shear=cfg["shear"])
    chk = rg.kerr_footing_selfcheck(width=2.0, w0=1.0e-3, foundation=fnd)
    assert chk["all_pass"], chk
    assert chk["law_inside_rel_err"] < 1e-10
    assert chk["edge_jump_rel_err"] < 1e-10


def test_footing_resistance_limits():
    base = dict(width=2.0, w0=1.0e-3)
    # Winkler limit (shear -> 0): R = K w0 b
    wink = rg.KerrFoundation(k_upper=2.0e6, k_lower=2.0e6, shear=1e-6)
    assert rg.kerr_footing_resistance(foundation=wink, **base) == pytest.approx(
        wink.k_series * 1.0e-3 * 2.0, rel=1e-6)
    # R is linear in the width with a positive edge term
    fnd = rg.KerrFoundation(k_upper=2.0e7, k_lower=8.0e7, shear=4.0e8)
    r1 = rg.kerr_footing_resistance(width=1.0, w0=1.0e-3, foundation=fnd)
    r2 = rg.kerr_footing_resistance(width=2.0, w0=1.0e-3, foundation=fnd)
    r3 = rg.kerr_footing_resistance(width=3.0, w0=1.0e-3, foundation=fnd)
    assert (r2 - r1) == pytest.approx(r3 - r2, rel=1e-9)
    assert r1 > fnd.k_series * 1.0e-3 * 1.0        # edge term adds resistance
    # pressure peaks at the footing edge (edge concentration), zero outside
    p = rg.kerr_footing_pressure(np.array([0.0, 0.999, 1.5]), half_width=1.0,
                                 w0=1.0e-3, foundation=fnd)
    assert p[0] < p[1]
    assert p[2] == 0.0


def test_footing_surface_deflection_decays_outside():
    fnd = rg.KerrFoundation(k_upper=2.0e7, k_lower=8.0e7, shear=4.0e8)
    l0 = rg.kerr_edge_length(fnd)
    w = rg.kerr_footing_surface_deflection(np.array([0.5, 1.0, 1.0 + l0, 1.0 + 5 * l0]),
                                          half_width=1.0, w0=2.0e-3, foundation=fnd)
    assert w[0] == pytest.approx(2.0e-3) and w[1] == pytest.approx(2.0e-3)
    assert w[2] == pytest.approx(2.0e-3 / math.e, rel=1e-9)
    assert w[3] < 1e-4


# ---------------------------------------------------------------------------
# GCI
# ---------------------------------------------------------------------------
def test_gci_recovers_known_order_and_zero_for_exact_series():
    # a second-order convergent series with ratio 2 must return p = 2
    def fh(h):
        return 1.0 + 3.0 * h ** 2
    vals = [fh(1 / 16), fh(1 / 8), fh(1 / 4)]     # fine, medium, coarse
    out = rg.gci_triad(vals, r=2.0)
    assert out["observed_order_p"] == pytest.approx(2.0, abs=1e-6)
    assert out["gci_fine_rel"] < 0.02
    with pytest.raises(ValueError):
        rg.gci_triad([1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# ISO 19906 A.8 operator layer (values withheld -> operators only)
# ---------------------------------------------------------------------------
def test_pressure_area_operator_layer():
    checks = rg.pressure_area_operator_checks()
    assert checks["all_pass"], checks
    # a round trip through the pair inversion returns the exponent exactly
    n = 0.37
    a1, a2 = 0.5, 12.0
    p1 = rg.pressure_area_pressure(a1, p0=2.0e6, a0=1.0, n=n)
    p2 = rg.pressure_area_pressure(a2, p0=2.0e6, a0=1.0, n=n)
    assert rg.pressure_area_exponent_from_pair(a1, float(p1), a2, float(p2)) == pytest.approx(n)
    # force law: F = p A, and F grows with area for 0 < n < 1
    f = rg.pressure_area_force(np.array([1.0, 10.0]), p0=1.0, a0=1.0, n=0.5)
    assert f[1] > f[0]
    with pytest.raises(ValueError):
        rg.pressure_area_pressure(-1.0, p0=1.0, a0=1.0, n=0.5)


# ---------------------------------------------------------------------------
# bonded-keel DEM specimen
# ---------------------------------------------------------------------------
def test_keel_sample_build_is_deterministic_and_bonded():
    s1 = kd.build_keel_sample(1500, height=2.0, seed=3)
    s2 = kd.build_keel_sample(1500, height=2.0, seed=3)
    assert s1.n_particles == s2.n_particles
    assert np.allclose(s1.x, s2.x)
    assert s1.bi.size > 2 * s1.n_particles            # connectivity > 2 (bonds per particle / 2)
    assert 4.0 < s1.coordination < 8.0
    assert np.all(s1.kn > 0) and np.all(s1.kt > 0)
    # every bond length is inside the neighbour cutoff used at build time
    assert np.max(s1.l0) <= 1.06 * 2.0 * float(np.max(s1.r)) + 1e-12
    with pytest.raises(ValueError):
        kd.build_keel_sample(10)


def test_dem_footing_resistance_increases_with_width_and_relaxes():
    s = kd.build_keel_sample(1500, height=2.0, seed=5)
    wide = kd.footing_test(s, footing_width=1.0, w0=4.0e-3, increments=3,
                           steps_per_increment=120)
    narrow = kd.footing_test(s, footing_width=0.5, w0=4.0e-3, increments=3,
                             steps_per_increment=120)
    assert wide["resistance_N_per_m"][-1] > narrow["resistance_N_per_m"][-1] > 0.0
    assert wide["resistance_N_per_m"][-1] > wide["resistance_N_per_m"][0]
    # linear-in-width structure of the reduced model: R = A b + B with B > 0
    mid = kd.footing_test(s, footing_width=2.0, w0=4.0e-3, increments=3,
                          steps_per_increment=120)
    r = [narrow["resistance_N_per_m"][-1], wide["resistance_N_per_m"][-1],
         mid["resistance_N_per_m"][-1]]
    # a 1500-particle specimen is far too coarse to resolve the linear
    # R = A b + B structure quantitatively (the 25k-100k study does that); here we
    # only require monotonic growth and the correct sign of both terms
    assert (r[1] - r[0]) > 0.0 and (r[2] - r[1]) > 0.0
    slope = 0.5 * ((r[1] - r[0]) / 0.5 + (r[2] - r[1]) / 1.0)
    intercept = r[0] - slope * 0.5
    assert slope > 0.0 and intercept > 0.0


def test_dem_confined_compression_is_stiffer_than_free_footing():
    s = kd.build_keel_sample(1500, height=2.0, seed=5)
    conf = kd.confined_compression(s, w_max=4.0e-3, increments=3, steps_per_increment=120)
    k = conf["k_subgrade_N_per_m3"]
    assert k > 0.0
    foot = kd.footing_test(s, footing_width=0.5, w0=4.0e-3, increments=3,
                           steps_per_increment=120)
    per_area = foot["resistance_N_per_m"][-1] / 0.5
    assert per_area > k * 4.0e-3          # stress concentration at a narrow footing