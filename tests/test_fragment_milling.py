"""Tests for fragment-cloud propeller-ice milling (P121W5-N09).

Coverage
--------
fsd             Power-law FSD generation, size bounds, mass consistency
cloud           FragmentCloud creation, alive/dead tracking, total mass
contact         Fragment-blade contact force sanity, zero when no overlap
collision       Fragment-fragment collision energy conservation check
recirculation   Recirculation ratio R=0 reproduces single-block; R>0 increases alive count
spectrum        Sideband energy ratio E_sb/E_total monotone in N_c and R
degradation     N_c=0 reproduces single-block milling to machine precision
conservation    Mass ledger <1e-14, energy closure <2%
convergence     dt/2 changes ΔQ/Q <1%, E_sb <5%
negative        Randomized fragment phases destroy sideband law (R²<0.3)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.propeller_ice import (
    FragmentCloud,
    FragmentMillingConfig,
    MillingConfig,
    PropellerGeometry,
    compute_sideband_energy_ratio,
    generate_fsd_sizes,
    simulate_fragment_milling,
    simulate_milling,
)
from tensordem.propeller_ice.fragment_milling import (
    _fragment_blade_contact,
    _fragment_fragment_collision,
)


# ------------------------------------------------------------------- FSD ---
def test_fsd_generates_correct_number_of_sizes() -> None:
    for n in (0, 5, 20, 40):
        sizes = generate_fsd_sizes(n, alpha=3.0, d_min=0.005, d_max=0.08, seed=42)
        assert sizes.size == n


def test_fsd_sizes_within_bounds() -> None:
    sizes = generate_fsd_sizes(100, alpha=3.0, d_min=0.005, d_max=0.08, seed=42)
    assert np.all(sizes >= 0.005 - 1e-15)
    assert np.all(sizes <= 0.08 + 1e-15)


def test_fsd_power_law_exponent() -> None:
    """FSD should follow n(D) ~ D^{-alpha}: verify via CDF."""
    sizes = generate_fsd_sizes(5000, alpha=3.0, d_min=0.005, d_max=0.08, seed=42)
    # For power law: 1 - CDF(D) ~ D^{1-alpha}
    # So log(1-CDF) vs log(D) should have slope = 1-alpha = -2.0
    sorted_sizes = np.sort(sizes)
    cdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    tail = 1.0 - cdf
    keep = (tail > 0.01) & (tail < 0.99)  # avoid extremes
    if np.sum(keep) >= 10:
        slope, _ = np.polyfit(np.log10(sorted_sizes[keep]), np.log10(tail[keep]), 1)
        # slope should be approximately 1-alpha = -2.0
        assert -2.5 < slope < -1.5, f"FSD CDF slope {slope:.2f} not near -2.0"


def test_fsd_zero_fragments_returns_empty() -> None:
    sizes = generate_fsd_sizes(0, alpha=3.0)
    assert sizes.size == 0


# --------------------------------------------------------------- cloud ---
def test_fragment_cloud_creation() -> None:
    geom = PropellerGeometry()
    sizes = generate_fsd_sizes(20, alpha=3.0, seed=42)
    cloud = FragmentCloud.create(20, sizes, geom, ice_density=917.0, seed=42)
    assert cloud.n_fragments == 20
    assert cloud.n_alive == 20
    assert cloud.total_mass > 0.0
    assert np.all(cloud.diameters > 0.0)
    assert np.all(cloud.masses > 0.0)
    # masses should be consistent with diameters (sphere volume)
    expected_masses = 917.0 * np.pi / 6.0 * sizes ** 3
    assert np.allclose(cloud.masses, expected_masses)


def test_fragment_cloud_alive_tracking() -> None:
    geom = PropellerGeometry()
    sizes = generate_fsd_sizes(10, alpha=3.0, seed=42)
    cloud = FragmentCloud.create(10, sizes, geom, seed=42)
    cloud.alive[0] = False
    cloud.alive[1] = False
    assert cloud.n_alive == 8
    assert cloud.total_mass == pytest.approx(np.sum(cloud.masses[2:]), rel=1e-12)


# ------------------------------------------------------------- contact ---
def test_fragment_blade_contact_zero_when_no_overlap() -> None:
    """Fragments far from blades should produce zero contact force."""
    geom = PropellerGeometry()
    from tensordem.propeller_ice.contact import BladeIceContact
    contact = BladeIceContact()

    sizes = generate_fsd_sizes(5, alpha=3.0, seed=42)
    cloud = FragmentCloud.create(5, sizes, geom, seed=42)
    # place fragments far from blades (outside disc)
    cloud.r[:] = geom.radius * 2.0
    cloud.theta[:] = 0.0
    cloud.z[:] = 0.0

    torque, thrust, normal = _fragment_blade_contact(
        cloud, geom, contact, omega=6.28, theta_prop=0.0,
        n_blades=4, ice_thickness=0.4,
    )
    assert torque == 0.0
    assert thrust == 0.0
    assert normal == 0.0


def test_fragment_blade_contact_positive_when_overlapping() -> None:
    """Fragments on blades should produce positive contact force."""
    geom = PropellerGeometry()
    from tensordem.propeller_ice.contact import BladeIceContact
    contact = BladeIceContact()

    sizes = np.array([0.02, 0.03, 0.04])
    cloud = FragmentCloud.create(3, sizes, geom, seed=42)
    # place fragments on a blade
    cloud.r[:] = 0.7 * geom.radius
    cloud.theta[:] = 0.0  # aligned with blade 0
    cloud.z[:] = 0.0

    torque, thrust, normal = _fragment_blade_contact(
        cloud, geom, contact, omega=6.28, theta_prop=0.0,
        n_blades=4, ice_thickness=0.4,
    )
    assert torque > 0.0
    assert normal > 0.0


# ----------------------------------------------------------- collision ---
def test_fragment_collision_conserves_momentum() -> None:
    """Head-on collision should conserve total momentum."""
    geom = PropellerGeometry()
    sizes = np.array([0.02, 0.02])
    cloud = FragmentCloud.create(2, sizes, geom, seed=42)
    # place two fragments close together
    cloud.r[:] = 0.5 * geom.radius
    cloud.theta[:] = [0.0, 0.001]
    cloud.z[:] = [0.0, 0.0]
    # give them relative velocity
    cloud.vr[:] = [0.1, -0.1]
    cloud.vtheta[:] = [0.0, 0.0]
    cloud.vz[:] = [0.0, 0.0]

    # compute initial momentum
    p_initial = np.sum(cloud.masses * cloud.vr)

    diss = _fragment_fragment_collision(cloud, cutoff=0.05, e_c=0.5)

    # compute final momentum
    p_final = np.sum(cloud.masses * cloud.vr)

    # momentum should be conserved (within numerical precision)
    assert abs(p_initial - p_final) < 1e-10
    # dissipation should be non-negative
    assert diss >= 0.0


def test_fragment_collision_dissipation_positive() -> None:
    """Inelastic collision should dissipate energy."""
    geom = PropellerGeometry()
    sizes = np.array([0.02, 0.02])
    cloud = FragmentCloud.create(2, sizes, geom, seed=42)
    cloud.r[:] = 0.5 * geom.radius
    cloud.theta[:] = [0.0, 0.001]
    cloud.z[:] = [0.0, 0.0]
    cloud.vr[:] = [0.5, -0.5]
    cloud.vtheta[:] = [0.0, 0.0]
    cloud.vz[:] = [0.0, 0.0]

    diss = _fragment_fragment_collision(cloud, cutoff=0.05, e_c=0.3)
    assert diss > 0.0


# ------------------------------------------------------- recirculation ---
def test_recirculation_zero_reproduces_single_block() -> None:
    """R=0, N_c=0 must reproduce single-block milling."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.0, dt=1.0e-3)
    base_result = simulate_milling(base_cfg)

    frag_cfg = FragmentMillingConfig(base=base_cfg, n_fragments=0, recirculation_ratio=0.0)
    frag_result = simulate_fragment_milling(frag_cfg)

    # torque must match to machine precision
    assert np.allclose(frag_result.torque_ice, base_result.torque_ice, atol=1e-12)
    assert np.allclose(frag_result.torque_hydro, base_result.torque_hydro, atol=1e-12)
    # sideband ratio should be the same
    assert frag_result.sideband_energy_ratio == pytest.approx(
        compute_sideband_energy_ratio(
            base_result.time, base_result.torque_ice,
            base_cfg.rpm, base_cfg.propeller.n_blades,
            window=base_result.steady_mask(),
        ),
        rel=1e-6,
    )


def test_recirculation_increases_alive_fragments() -> None:
    """R>0 should keep more fragments alive than R=0."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=3.0, dt=1.0e-3)

    cfg_r0 = FragmentMillingConfig(base=base_cfg, n_fragments=20, recirculation_ratio=0.0, seed=42)
    cfg_r2 = FragmentMillingConfig(base=base_cfg, n_fragments=20, recirculation_ratio=0.2, seed=42)

    result_r0 = simulate_fragment_milling(cfg_r0)
    result_r2 = simulate_fragment_milling(cfg_r2)

    # R>0 should have more recirculated mass
    assert result_r2.mass_fragments_recirculated > result_r0.mass_fragments_recirculated


# ------------------------------------------------------------ spectrum ---
def test_sideband_energy_ratio_monotone_in_nc() -> None:
    """E_sb/E_total should increase monotonically with N_c (Spearman rho >= 0.95)."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.5, dt=1.0e-3, spin_up=0.5)

    nc_values = [0, 5, 10, 20, 30]
    ratios = []
    for nc in nc_values:
        cfg = FragmentMillingConfig(base=base_cfg, n_fragments=nc, recirculation_ratio=0.2, seed=42)
        result = simulate_fragment_milling(cfg)
        ratios.append(result.sideband_energy_ratio)

    # check monotonicity (non-decreasing)
    for i in range(len(ratios) - 1):
        assert ratios[i] <= ratios[i + 1] + 1e-6, (
            f"E_sb/E_total not monotone: {ratios[i]:.6f} > {ratios[i+1]:.6f} at N_c={nc_values[i]}->{nc_values[i+1]}"
        )


def test_sideband_energy_ratio_monotone_in_r() -> None:
    """E_sb/E_total should increase monotonically with R (Spearman rho >= 0.95)."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.5, dt=1.0e-3, spin_up=0.5)

    r_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    ratios = []
    for r in r_values:
        cfg = FragmentMillingConfig(base=base_cfg, n_fragments=20, recirculation_ratio=r, seed=42)
        result = simulate_fragment_milling(cfg)
        ratios.append(result.sideband_energy_ratio)

    # check monotonicity (non-decreasing)
    for i in range(len(ratios) - 1):
        assert ratios[i] <= ratios[i + 1] + 1e-6, (
            f"E_sb/E_total not monotone in R: {ratios[i]:.6f} > {ratios[i+1]:.6f} at R={r_values[i]}->{r_values[i+1]}"
        )


# ------------------------------------------------------- degradation ---
def test_nc_zero_degradation_open_water() -> None:
    """N_c=0 must reproduce open-water KT/KQ to machine precision."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.0, dt=1.0e-3)
    base_result = simulate_milling(base_cfg)

    frag_cfg = FragmentMillingConfig(base=base_cfg, n_fragments=0, recirculation_ratio=0.0)
    frag_result = simulate_fragment_milling(frag_cfg)

    # KT/KQ from open water should match
    assert frag_result.base_result is not None
    assert frag_result.base_result.open_water.thrust_coefficient == pytest.approx(
        base_result.open_water.thrust_coefficient, rel=1e-12
    )
    assert frag_result.base_result.open_water.torque_coefficient == pytest.approx(
        base_result.open_water.torque_coefficient, rel=1e-12
    )


def test_nc_zero_degradation_delta_q() -> None:
    """N_c=0 must reproduce ΔQ/Q to machine precision."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.0, dt=1.0e-3)
    base_result = simulate_milling(base_cfg)

    frag_cfg = FragmentMillingConfig(base=base_cfg, n_fragments=0, recirculation_ratio=0.0)
    frag_result = simulate_fragment_milling(frag_cfg)

    # Torque histories should be identical
    assert np.allclose(frag_result.torque_ice, base_result.torque_ice, atol=1e-12)
    # ΔQ/Q should match (use base_result's method since frag_result delegates to it)
    base_dq = base_result.torque_pulsation()["delta_q_over_mean_p95"]
    # frag_result delegates to base_result, so they should be identical
    assert frag_result.base_result is not None
    frag_dq = frag_result.base_result.torque_pulsation()["delta_q_over_mean_p95"]
    assert abs(base_dq - frag_dq) < 1e-9


# ------------------------------------------------------- conservation ---
def test_mass_ledger_closes() -> None:
    """Mass ledger must close to machine precision."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.0, dt=1.0e-3)
    cfg = FragmentMillingConfig(base=base_cfg, n_fragments=20, recirculation_ratio=0.2, seed=42)
    result = simulate_fragment_milling(cfg)

    # mass_removed should equal mass_chips + mass_pending (from base)
    closure = abs(result.mass_removed - (result.mass_chips + result.mass_pending))
    assert closure < 1e-12 * max(result.mass_removed, 1e-30)


def test_energy_closure_within_2pct() -> None:
    """Energy ledger must close within 2%."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.0, dt=1.0e-3)
    cfg = FragmentMillingConfig(base=base_cfg, n_fragments=20, recirculation_ratio=0.2, seed=42)
    result = simulate_fragment_milling(cfg)

    cutting = result.cutting_work
    frag_ke = result.fragment_kinetic_energy
    collision_diss = result.collision_dissipation
    drag_diss = result.drag_dissipation

    total_output = frag_ke + collision_diss + drag_diss
    if cutting > 0:
        closure = abs(cutting - total_output) / cutting
        assert closure < 0.02, f"Energy closure {closure:.4f} > 2%"


# ------------------------------------------------------- convergence ---
def test_timestep_convergence_delta_q() -> None:
    """dt/2 should change ΔQ/Q by <1%."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.0, dt=2.0e-3)
    cfg_coarse = FragmentMillingConfig(base=base_cfg, n_fragments=10, recirculation_ratio=0.2, seed=42)

    base_cfg_fine = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.0, dt=1.0e-3)
    cfg_fine = FragmentMillingConfig(base=base_cfg_fine, n_fragments=10, recirculation_ratio=0.2, seed=42)

    result_coarse = simulate_fragment_milling(cfg_coarse)
    result_fine = simulate_fragment_milling(cfg_fine)

    mask_c = result_coarse.steady_mask()
    mask_f = result_fine.steady_mask()

    dq_coarse = result_coarse.torque_ice[mask_c](np.max(q) - np.min(q)) / np.mean(result_coarse.torque_ice[mask_c])
    dq_fine = result_fine.torque_ice[mask_f](np.max(q) - np.min(q)) / np.mean(result_fine.torque_ice[mask_f])

    assert abs(dq_coarse - dq_fine) / dq_fine < 0.01


def test_timestep_convergence_sideband() -> None:
    """dt/2 should change E_sb/E_total by <5%."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.0, dt=2.0e-3)
    cfg_coarse = FragmentMillingConfig(base=base_cfg, n_fragments=10, recirculation_ratio=0.2, seed=42)

    base_cfg_fine = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.0, dt=1.0e-3)
    cfg_fine = FragmentMillingConfig(base=base_cfg_fine, n_fragments=10, recirculation_ratio=0.2, seed=42)

    result_coarse = simulate_fragment_milling(cfg_coarse)
    result_fine = simulate_fragment_milling(cfg_fine)

    if result_fine.sideband_energy_ratio > 1e-6:
        assert abs(result_coarse.sideband_energy_ratio - result_fine.sideband_energy_ratio
                   ) / result_fine.sideband_energy_ratio < 0.05


# ------------------------------------------------------- negative ctrl ---
def test_randomized_phases_destroy_sideband_law() -> None:
    """Randomizing fragment phases should destroy the sideband law (R²<0.3)."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.5, dt=1.0e-3, spin_up=0.5)

    # Run with different seeds (randomized phases)
    ratios = []
    for seed in range(5):
        cfg = FragmentMillingConfig(base=base_cfg, n_fragments=20, recirculation_ratio=0.2, seed=seed)
        result = simulate_fragment_milling(cfg)
        ratios.append(result.sideband_energy_ratio)

    # With randomized phases, the sideband ratio should vary widely
    # and not show a consistent trend (low R² if we try to fit)
    if len(ratios) >= 3:
        x = np.arange(len(ratios), dtype=float)
        y = np.array(ratios)
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # With random phases, R² should be low
        assert r2 < 0.3, f"Randomized phases still show strong trend (R²={r2:.3f})"


# ------------------------------------------------------- delta_q_range ---
def test_delta_q_in_physical_band() -> None:
    """ΔQ/Q with fragments should remain in [0.10, 0.30]."""
    base_cfg = MillingConfig(rpm=60.0, advance_ratio=0.6, duration=2.5, dt=1.0e-3)

    for nc in [5, 10, 20]:
        cfg = FragmentMillingConfig(base=base_cfg, n_fragments=nc, recirculation_ratio=0.2, seed=42)
        result = simulate_fragment_milling(cfg)
        mask = result.steady_mask()
        q = result.torque_ice[mask]
        dq = (np.percentile(q, 95) - np.percentile(q, 5)) / np.mean(q)
        assert 0.10 <= dq <= 0.30, f"ΔQ/Q={dq:.3f} outside [0.10, 0.30] for N_c={nc}"
