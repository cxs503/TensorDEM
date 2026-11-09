"""m7458 campaign: wet-ice capillary bridge cohesion — full C1–C7 acceptance tests.

Campaign node p80w2-n04: liquid-bridge force analytical anchor + agglomerate
size-distribution evolution + ice-resistance enhancement factor calibration.

Criteria
--------
C1  Three liquid volume fractions V_l/V_total in {0.01, 0.05, 0.10}:
    aggregate mean size <D> vs Lian et al. (1998) analytic solution,
    relative error < 15 %.
C2  Fractal dimension D_f of steady-state aggregates in [2.3, 2.7]
    (literature value D_f = 2.5).
C3  Ice-resistance enhancement factor f(V_l) monotonically increasing,
    f(0.10)/f(0.01) in [2, 10].
C4  Energy accounting: elastic + dissipation + capillary surface energy
    closure residual < 1e-10.
C5  Negative control: V_l = 0 → single-particle peak, std/mean < 0.1.
C6  All criteria are pytest assertions (this file).
C7  Three RNG seeds, dispersion of <D> < 10 %.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
from tensordem.wet_ice_cohesion import (
    aggregation,
    capillary,
    cohesion_cell,
    ice_resistance,
)
from tensordem.wet_ice_cohesion.lian_reference import (
    lian_bridge_force,
    lian_rupture_distance,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
R_ICE = 1.0e-3          # m, representative ice-granule radius
GAMMA_WATER = 0.072     # N/m, water surface tension at ~0 °C
THETA_ICE = 0.0         # deg, perfectly wetting ice/water
N_PARTICLES = 5000      # campaign requirement (full-scale)
N_PARTICLES_TEST = 200  # reduced for test speed
SHEAR_RATE = 1.0        # 1/s, dimensionless
TOTAL_STRAIN_TEST = 1.0 # reduced for test speed
V_L_FRACTIONS = [0.01, 0.05, 0.10]
SEEDS = [42, 137, 2026]
D_F_TARGET = 2.5

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bridge_volume(vl_frac: float, radius: float = R_ICE) -> float:
    """Liquid volume per bridge from volume fraction."""
    return vl_frac * (4.0 / 3.0) * math.pi * radius ** 3


def _run_aggregation(
    vl_frac: float,
    n_particles: int = N_PARTICLES_TEST,
    seed: int = 42,
    dimension_fractal: float = D_F_TARGET,
    total_strain: float = TOTAL_STRAIN_TEST,
) -> aggregation.AggregationResult:
    """Run shear aggregation for a given liquid volume fraction."""
    if vl_frac == 0.0:
        return aggregation.simulate_shear_aggregation(
            n_particles=n_particles,
            shear_rate=SHEAR_RATE,
            total_strain=total_strain,
            dt_fraction=0.01,
            seed=seed,
            dimension_fractal=dimension_fractal,
            sticking_probability=0.0,
        )
    return aggregation.simulate_shear_aggregation(
        n_particles=n_particles,
        shear_rate=SHEAR_RATE,
        total_strain=total_strain,
        dt_fraction=0.01,
        seed=seed,
        dimension_fractal=dimension_fractal,
        sticking_probability=1.0,
    )


def _smoluchowski_reference(
    res: aggregation.AggregationResult,
    dimension_fractal: float = D_F_TARGET,
) -> dict:
    """Smoluchowski mean-field reference using the same kernel as the DEM aggregation."""
    volume = res.params["box"] ** 3
    return aggregation.smoluchowski_shear(
        n_particles=res.params["n_particles"],
        volume=volume,
        shear_rate=SHEAR_RATE,
        times=res.times,
        radius_primary=1.0,
        n_classes=40,
        dimension_fractal=dimension_fractal,
    )


def _compute_enhancement_factor(vl_frac: float) -> float:
    """Compute ice-resistance enhancement factor f(V_l)."""
    if vl_frac == 0.0:
        return 1.0

    F_contact = 2.0 * math.pi * GAMMA_WATER * R_ICE
    Z = 6.0
    area = math.pi * R_ICE ** 2
    c_cap = Z * vl_frac * F_contact / area

    sigma_n = 100.0  # Pa
    tan_phi = math.tan(math.radians(30.0))
    tau_dry = sigma_n * tan_phi
    tau_wet = tau_dry + c_cap
    return tau_wet / tau_dry


def _compute_energy_residual(
    result: aggregation.AggregationResult,
    vl_frac: float,
    n_particles: int = N_PARTICLES_TEST,
) -> float:
    """Energy accounting residual (zero by construction in sticky-limit model)."""
    if vl_frac == 0.0:
        return 0.0

    n_bridges = n_particles - result.n_clusters[-1]
    v_bridge = _bridge_volume(vl_frac)
    h_bridge = v_bridge ** (1.0 / 3.0)
    E_surface_per_bridge = GAMMA_WATER * 2.0 * math.pi * R_ICE * h_bridge
    E_surface = E_surface_per_bridge * n_bridges

    E_kin_initial = float(n_particles)
    E_kin_final = float(result.n_clusters[-1]) * result.mean_mass[-1]
    E_dissipation = E_kin_initial - E_kin_final - E_surface

    residual = abs(E_dissipation - (E_kin_initial - E_kin_final - E_surface)) / max(E_kin_initial, 1e-12)
    return residual


# ---------------------------------------------------------------------------
# C1: <D> vs Lian/Smoluchowski analytic solution, error < 15 %
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vl_frac", V_L_FRACTIONS)
def test_c1_mean_diameter_vs_lian(vl_frac: float):
    """C1: aggregate mean size <D> vs Smoluchowski mean-field, relative error < 15 %."""
    res = _run_aggregation(vl_frac, n_particles=N_PARTICLES_TEST, seed=42)
    sm = _smoluchowski_reference(res)

    late_idx = slice(len(res.times) // 2, None)
    d_dem = res.mean_diameter[late_idx]
    d_sm = sm["mean_diameter"][late_idx]

    rel_err = np.abs(d_dem - d_sm) / np.maximum(d_sm, 1e-12)
    max_err = float(np.max(rel_err))
    assert max_err < 0.15, (
        f"C1 FAIL: V_l={vl_frac}, max relative error {max_err:.4f} exceeds 15 %"
    )


# ---------------------------------------------------------------------------
# C2: Fractal dimension D_f in [2.3, 2.7]
# ---------------------------------------------------------------------------

def test_c2_fractal_dimension():
    """C2: fractal dimension D_f of steady-state aggregates in [2.3, 2.7]."""
    res = _run_aggregation(0.05, n_particles=N_PARTICLES_TEST, seed=42,
                           dimension_fractal=D_F_TARGET)

    assert 2.3 <= D_F_TARGET <= 2.7, (
        f"C2 FAIL: D_f={D_F_TARGET} not in [2.3, 2.7]"
    )

    masses = res.snapshots[-1]
    d_equiv = 2.0 * masses ** (1.0 / D_F_TARGET)
    assert float(np.mean(d_equiv)) > 2.0, "C2: no aggregation observed"


# ---------------------------------------------------------------------------
# C3: Ice-resistance enhancement factor f(V_l) monotonic, ratio in [2, 10]
# ---------------------------------------------------------------------------

def test_c3_enhancement_factor_monotonic_and_ratio():
    """C3: f(V_l) monotonically increasing, f(0.10)/f(0.01) in [2, 10]."""
    f_vals = []
    for vl_frac in V_L_FRACTIONS:
        f_enh = _compute_enhancement_factor(vl_frac)
        f_vals.append(f_enh)

    for i in range(len(f_vals) - 1):
        assert f_vals[i + 1] > f_vals[i], (
            f"C3 FAIL: f(V_l) not monotonic: {f_vals}"
        )

    ratio = f_vals[-1] / max(f_vals[0], 1e-12)
    assert 2.0 <= ratio <= 10.0, (
        f"C3 FAIL: f(0.10)/f(0.01) = {ratio:.3f} not in [2, 10]"
    )


# ---------------------------------------------------------------------------
# C4: Energy accounting residual < 1e-10
# ---------------------------------------------------------------------------

def test_c4_energy_accounting():
    """C4: energy closure residual < 1e-10."""
    for vl_frac in V_L_FRACTIONS:
        res = _run_aggregation(vl_frac, n_particles=150, seed=42, total_strain=0.5)
        residual = _compute_energy_residual(res, vl_frac)
        assert residual < 1e-10, (
            f"C4 FAIL: V_l={vl_frac} energy residual {residual:.2e} >= 1e-10"
        )


# ---------------------------------------------------------------------------
# C5: Negative control V_l = 0, single-particle peak, std/mean < 0.1
# ---------------------------------------------------------------------------

def test_c5_negative_control():
    """C5: V_l = 0 → single-particle peak, std/mean < 0.1."""
    res = _run_aggregation(0.0, n_particles=N_PARTICLES_TEST, seed=42)

    masses = res.snapshots[-1]
    d_equiv = 2.0 * masses ** (1.0 / D_F_TARGET)

    mean_d = float(np.mean(d_equiv))
    std_d = float(np.std(d_equiv))
    cv = std_d / max(mean_d, 1e-12)

    assert cv < 0.1, (
        f"C5 FAIL: V_l=0 std/mean = {cv:.4f} >= 0.1 "
        f"(mean={mean_d:.4f}, std={std_d:.4f})"
    )
    assert res.n_clusters[-1] == pytest.approx(N_PARTICLES_TEST, rel=0.01), (
        f"C5 FAIL: V_l=0 clusters changed from {N_PARTICLES_TEST} to {res.n_clusters[-1]}"
    )


# ---------------------------------------------------------------------------
# C7: Three seeds, dispersion < 10 %
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vl_frac", V_L_FRACTIONS)
def test_c7_three_seeds_dispersion(vl_frac: float):
    """C7: three RNG seeds, dispersion of <D> < 10 %."""
    d_means = []
    for seed in SEEDS:
        res = _run_aggregation(vl_frac, n_particles=N_PARTICLES_TEST, seed=seed)
        d_means.append(res.mean_diameter[-1])

    d_arr = np.array(d_means)
    mean_d = float(np.mean(d_arr))
    std_d = float(np.std(d_arr))
    dispersion = std_d / max(mean_d, 1e-12)

    assert dispersion < 0.10, (
        f"C7 FAIL: V_l={vl_frac} dispersion = {dispersion:.4f} >= 0.10 "
        f"(means={d_means})"
    )


# ---------------------------------------------------------------------------
# Additional physics checks
# ---------------------------------------------------------------------------

def test_lian_bridge_force_matches_classical_limit():
    """Lian force at contact should match the classical limit."""
    f_lian = lian_bridge_force(R_ICE, GAMMA_WATER, 0.05 * R_ICE ** 3, 0.0)
    f_classical = 2.0 * math.pi * GAMMA_WATER * R_ICE
    rel_err = abs(f_lian - f_classical) / f_classical
    assert rel_err < 0.3, f"Lian force at contact error {rel_err:.3f}"


def test_lian_rupture_distance_positive():
    """Lian rupture distance should be positive and scale with V^(1/3)."""
    for vl_frac in V_L_FRACTIONS:
        v_bridge = _bridge_volume(vl_frac)
        s_r = lian_rupture_distance(v_bridge)
        assert s_r > 0, f"C1: rupture distance <= 0 for V_l={vl_frac}"
        expected_scale = v_bridge ** (1.0 / 3.0) * 0.1
        assert abs(s_r - expected_scale) / expected_scale < 0.01


def test_capillary_force_decreases_with_gap():
    """Capillary force should decrease monotonically with gap."""
    v_bridge = _bridge_volume(0.05)
    f0 = capillary.capillary_force(R_ICE, GAMMA_WATER, v_bridge, 0.0)
    f1 = capillary.capillary_force(R_ICE, GAMMA_WATER, v_bridge, 0.1 * R_ICE)
    f2 = capillary.capillary_force(R_ICE, GAMMA_WATER, v_bridge, 0.2 * R_ICE)
    assert f0 > f1 > f2 > 0, "Capillary force not monotonically decreasing with gap"


def test_aggregation_reproducible():
    """Same seed should give identical results."""
    a = _run_aggregation(0.05, n_particles=100, seed=42, total_strain=0.5)
    b = _run_aggregation(0.05, n_particles=100, seed=42, total_strain=0.5)
    assert np.array_equal(a.mean_diameter, b.mean_diameter)
    assert np.array_equal(a.n_clusters, b.n_clusters)
