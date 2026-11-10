"""Acceptance tests for seepage–piping closed-loop verification (p299w4-n02).

Tests cover all four verification links plus mass conservation:

C1  Effective stress / pore pressure vs analytic (error < 10 %)
C2  Critical hydraulic gradient vs Terzaghi (error < 15 %)
C3  Permeability vs Kozeny–Carman (error < 20 %)
C4  Suffusion trend vs Skempton–Brogan (trend consistent)
C5  Mass conservation (error < 2 %)
C6  pytest >= 12 all green (this file provides 16 tests)
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.seepage import (  # noqa: E402
    SeepagePrototype,
    SkemptonBroganEnvelope,
    SuffusionAnalysis,
    buoyant_unit_weight,
    effective_stress_profile,
    fine_fraction_loss,
    hydrostatic_pore_pressure,
    kozeny_carman_conductivity,
    kozeny_carman_intrinsic,
    kozeny_carman_porosity_sweep,
    mass_conservation_audit,
    piping_factor_of_safety,
    run_seepage_chain,
    seepage_force_per_volume,
    skempton_brogan_critical_gradient,
    specific_surface_sphere,
    suffusion_erosion_rate,
    terzaghi_critical_gradient,
    terzaghi_effective_stress,
)
from tensordem.seepage.critical_gradient import PipingAnalysis  # noqa: E402
from tensordem.seepage.effective_stress import SaturatedColumn  # noqa: E402
from tensordem.seepage.kozeny_carman import KozenyCarmanModel  # noqa: E402


# ===================================================================
# C1: Effective stress / pore pressure
# ===================================================================

def test_hydrostatic_pore_pressure_linear():
    """Hydrostatic pore pressure must be linear with depth."""
    z = np.linspace(0.0, 1.0, 101)
    u = hydrostatic_pore_pressure(z, gamma_w=9810.0)
    # u = gamma_w * z
    expected = 9810.0 * z
    assert np.allclose(u, expected, rtol=1e-12)


def test_hydrostatic_zero_above_water_table():
    """Pore pressure must be zero above the water table."""
    z = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    u = hydrostatic_pore_pressure(z, water_table_depth=1.0)
    assert u[0] == pytest.approx(0.0)
    assert u[1] == pytest.approx(0.0)
    assert u[2] == pytest.approx(0.0)
    assert u[3] > 0.0
    assert u[4] > u[3]


def test_effective_stress_equals_buoyant_weight():
    """sigma'(z) = gamma' * z for a saturated column with top water table."""
    col = SaturatedColumn(height=1.0, porosity=0.40, rho_s=2650.0)
    z = np.linspace(0.0, col.height, 101)
    sigma_prime = col.effective_stress(z)
    gamma_prime = col.gamma_prime
    expected = gamma_prime * z
    assert np.allclose(sigma_prime, expected, rtol=1e-12)


def test_pore_pressure_profile_c1():
    """C1: DEM-measured pore pressure vs analytic hydrostatic, error < 10 %.

    We simulate a 'DEM measurement' by adding small noise to the analytic
    solution and verify the chain correctly judges it.
    """
    col = SaturatedColumn(height=0.10, porosity=0.40, rho_s=2650.0)
    z = np.linspace(0.0, col.height, 21)
    u_analytic = col.pore_pressure(z)
    # Simulate DEM measurement with 5 % noise
    rng = np.random.default_rng(42)
    u_measured = u_analytic * (1.0 + 0.05 * rng.standard_normal(len(z)))
    u_measured[0] = 0.0  # drained boundary

    result = run_seepage_chain(
        prototype=SeepagePrototype(height=0.10, porosity=0.40),
        measured_pore_pressures={"z": z.tolist(), "u": u_measured.tolist()},
    )
    assert result["C1_pore_pressure"]["pass"], (
        f"C1 failed: max_dev = {result['C1_pore_pressure']['max_deviation_pct']:.2f} %"
    )
    assert result["C1_pore_pressure"]["max_deviation_pct"] < 10.0


def test_pore_pressure_profile_fails_with_large_error():
    """C1 negative control: 30 % error must fail the 10 % gate."""
    col = SaturatedColumn(height=0.10, porosity=0.40, rho_s=2650.0)
    z = np.linspace(0.0, col.height, 21)
    u_analytic = col.pore_pressure(z)
    u_bad = u_analytic * 1.30  # 30 % systematic error
    u_bad[0] = 0.0

    result = run_seepage_chain(
        prototype=SeepagePrototype(height=0.10, porosity=0.40),
        measured_pore_pressures={"z": z.tolist(), "u": u_bad.tolist()},
    )
    assert not result["C1_pore_pressure"]["pass"]
    assert result["C1_pore_pressure"]["max_deviation_pct"] > 10.0


# ===================================================================
# C2: Critical hydraulic gradient
# ===================================================================

def test_terzaghi_critical_gradient_formula():
    """i_c = (G_s - 1)/(1 + e) must match gamma'/gamma_w."""
    rho_s, n = 2650.0, 0.40
    i_c = terzaghi_critical_gradient(rho_s, n)
    G_s = rho_s / 1000.0
    e = n / (1.0 - n)
    i_c_alt = (G_s - 1.0) / (1.0 + e)
    assert abs(i_c - i_c_alt) / i_c_alt < 1e-10


def test_critical_gradient_typical_sand():
    """For quartz sand (G_s=2.65, e~0.65), i_c should be ~1.0."""
    i_c = terzaghi_critical_gradient(2650.0, 0.40)
    assert 0.9 < i_c < 1.1, f"i_c = {i_c} for typical sand"


def test_buoyant_unit_weight_positive():
    """gamma' must be positive for any realistic soil."""
    for n in [0.2, 0.3, 0.4, 0.5, 0.6]:
        gp = buoyant_unit_weight(2650.0, n)
        assert gp > 0.0, f"gamma' <= 0 at n={n}"


def test_seepage_force_at_critical():
    """At i = i_c, seepage force must equal buoyant unit weight."""
    rho_s, n = 2650.0, 0.40
    i_c = terzaghi_critical_gradient(rho_s, n)
    gamma_w = 9810.0
    j = seepage_force_per_volume(i_c, gamma_w)
    gp = buoyant_unit_weight(rho_s, n)
    assert abs(j - gp) / gp < 1e-10


def test_critical_gradient_c2():
    """C2: DEM-measured i_c vs Terzaghi, error < 15 %.

    Simulate a DEM measurement with 10 % error (within tolerance).
    """
    proto = SeepagePrototype(rho_s=2650.0, porosity=0.40)
    i_c_analytic = proto.terzaghi_i_c
    i_c_measured = i_c_analytic * 1.10  # 10 % high

    result = run_seepage_chain(
        prototype=proto,
        measured_critical_gradient=i_c_measured,
    )
    assert result["C2_critical_gradient"]["pass"], (
        f"C2 failed: dev = {result['C2_critical_gradient']['deviation_pct']:.2f} %"
    )
    assert result["C2_critical_gradient"]["deviation_pct"] < 15.0


def test_critical_gradient_fails_with_large_error():
    """C2 negative control: 25 % error must fail the 15 % gate."""
    proto = SeepagePrototype(rho_s=2650.0, porosity=0.40)
    i_c_analytic = proto.terzaghi_i_c
    i_c_measured = i_c_analytic * 1.25  # 25 % high

    result = run_seepage_chain(
        prototype=proto,
        measured_critical_gradient=i_c_measured,
    )
    assert not result["C2_critical_gradient"]["pass"]


def test_piping_factor_of_safety():
    """FS = i_c / i: FS > 1 stable, FS < 1 piping."""
    i_c = terzaghi_critical_gradient(2650.0, 0.40)
    fs_safe = piping_factor_of_safety(0.5 * i_c, 2650.0, 0.40)
    fs_critical = piping_factor_of_safety(i_c, 2650.0, 0.40)
    fs_unsafe = piping_factor_of_safety(1.5 * i_c, 2650.0, 0.40)
    assert fs_safe > 1.0
    assert abs(fs_critical - 1.0) < 1e-10
    assert fs_unsafe < 1.0


# ===================================================================
# C3: Kozeny–Carman permeability
# ===================================================================

def test_specific_surface_sphere():
    """S = 6/d for spheres."""
    d = 2.0e-4
    assert specific_surface_sphere(d) == pytest.approx(6.0 / d, rel=1e-12)


def test_kozeny_carman_porosity_dependence():
    """k must increase strongly with porosity."""
    d = 2.0e-4
    k_lo = kozeny_carman_intrinsic(0.30, d)
    k_hi = kozeny_carman_intrinsic(0.50, d)
    assert k_hi > k_lo
    # The phi^3/(1-phi)^2 dependence means k(0.5)/k(0.3) should be large
    ratio = k_hi / k_lo
    assert ratio > 5.0, f"k(0.5)/k(0.3) = {ratio}, expected > 5"


def test_kozeny_carman_conductivity_units():
    """K = k_int * gamma_w / mu_w must have units of m/s."""
    k_int = kozeny_carman_intrinsic(0.40, 2.0e-4)
    K = kozeny_carman_conductivity(0.40, 2.0e-4)
    expected = k_int * 9810.0 / 1.0e-3
    assert K == pytest.approx(expected, rel=1e-12)
    # Typical sand: K ~ 1e-4 to 1e-3 m/s
    assert 1e-6 < K < 1e-1


def test_permeability_c3():
    """C3: DEM-measured k vs K-C formula, error < 20 %.

    Simulate DEM measurements with 15 % error (within tolerance).
    """
    proto = SeepagePrototype(diameter=2.0e-4)
    porosities = [0.35, 0.40, 0.45, 0.50]
    kc_model = KozenyCarmanModel(diameter=proto.diameter)
    k_analytic = [kc_model.conductivity(phi) for phi in porosities]
    # Simulate DEM with 15 % error
    k_measured = [k * 1.15 for k in k_analytic]

    result = run_seepage_chain(
        prototype=proto,
        measured_permeabilities={
            "porosities": porosities,
            "k_measured": k_measured,
        },
    )
    assert result["C3_permeability"]["pass"], (
        f"C3 failed: max_dev = {result['C3_permeability']['max_deviation_pct']:.2f} %"
    )
    assert result["C3_permeability"]["max_deviation_pct"] < 20.0


def test_permeability_fails_with_large_error():
    """C3 negative control: 30 % error must fail the 20 % gate."""
    proto = SeepagePrototype(diameter=2.0e-4)
    porosities = [0.35, 0.40, 0.45]
    kc_model = KozenyCarmanModel(diameter=proto.diameter)
    k_analytic = [kc_model.conductivity(phi) for phi in porosities]
    k_measured = [k * 1.30 for k in k_analytic]  # 30 % error

    result = run_seepage_chain(
        prototype=proto,
        measured_permeabilities={
            "porosities": porosities,
            "k_measured": k_measured,
        },
    )
    assert not result["C3_permeability"]["pass"]


# ===================================================================
# C4: Suffusion trend
# ===================================================================

def test_skempton_brogan_alpha_range():
    """alpha must be in [0.3, 1.0] for F in [0, 0.5]."""
    env = SkemptonBroganEnvelope()
    for F in np.linspace(0.0, 0.5, 20):
        a = env.alpha_for(F)
        assert 0.3 <= a <= 1.0, f"alpha={a} at F={F}"


def test_suffusion_zero_below_threshold():
    """Erosion rate must be zero when i <= i_c,suff."""
    rate = suffusion_erosion_rate(0.5, 1.0, erosion_coefficient=1e-4, beta=1.5)
    assert rate == 0.0


def test_suffusion_positive_above_threshold():
    """Erosion rate must be positive when i > i_c,suff."""
    rate = suffusion_erosion_rate(1.5, 1.0, erosion_coefficient=1e-4, beta=1.5)
    assert rate > 0.0


def test_suffusion_trend_c4():
    """C4: fine-particle loss trend must be consistent with Skempton–Brogan.

    Simulate measurements showing zero erosion below threshold and
    increasing erosion above.
    """
    proto = SeepagePrototype(fine_fraction=0.20)
    env = SkemptonBroganEnvelope()
    i_c_suff = env.critical_gradient(proto.terzaghi_i_c, proto.fine_fraction)

    # Sorted gradients: clearly below and above threshold
    gradients = [0.1, 0.2, 0.3, 0.4, i_c_suff * 0.95,
                 i_c_suff * 1.05, i_c_suff * 1.2, i_c_suff * 1.5]
    rates = []
    for g in gradients:
        if g <= i_c_suff:
            rates.append(0.0)
        else:
            rates.append(1e-4 * (g - i_c_suff) ** 1.5)

    result = run_seepage_chain(
        prototype=proto,
        measured_erosion_rates={
            "gradients": gradients,
            "rates": rates,
        },
    )
    assert result["C4_suffusion_trend"]["pass"], (
        f"C4 failed: trend_consistent = {result['C4_suffusion_trend']['trend_consistent']}"
    )
    assert result["C4_suffusion_trend"]["trend_consistent"]


def test_suffusion_trend_fails_with_wrong_pattern():
    """C4 negative control: erosion below threshold must fail."""
    proto = SeepagePrototype(fine_fraction=0.20)
    env = SkemptonBroganEnvelope()
    i_c_suff = env.critical_gradient(proto.terzaghi_i_c, proto.fine_fraction)

    # Wrong: erosion below threshold
    gradients = [0.3, 0.5, i_c_suff * 0.5, i_c_suff * 1.5]
    rates = [0.0, 1e-5, 1e-4, 1e-3]  # non-zero at g=0.5 < i_c_suff

    result = run_seepage_chain(
        prototype=proto,
        measured_erosion_rates={
            "gradients": gradients,
            "rates": rates,
        },
    )
    assert not result["C4_suffusion_trend"]["pass"]


# ===================================================================
# C5: Mass conservation
# ===================================================================

def test_mass_conservation_exact():
    """C5: exact mass conservation must pass."""
    audit = mass_conservation_audit(
        initial_solid_mass=10.0,
        initial_fluid_mass=5.0,
        final_solid_mass=10.0,
        final_fluid_mass=5.0,
    )
    assert audit["pass"]
    assert audit["mass_error_pct"] == pytest.approx(0.0)


def test_mass_conservation_with_fine_loss():
    """C5: mass conservation accounting for fine particle loss."""
    audit = mass_conservation_audit(
        initial_solid_mass=10.0,
        initial_fluid_mass=5.0,
        final_solid_mass=9.9,       # 0.1 kg fine lost
        final_fluid_mass=5.0,
        fine_mass_lost=0.1,
    )
    assert audit["pass"]
    assert audit["mass_error_pct"] == pytest.approx(0.0)


def test_mass_conservation_fails_with_large_error():
    """C5 negative control: 5 % mass error must fail the 2 % gate."""
    audit = mass_conservation_audit(
        initial_solid_mass=10.0,
        initial_fluid_mass=5.0,
        final_solid_mass=9.5,       # 0.5 kg missing
        final_fluid_mass=5.0,
        fine_mass_lost=0.0,
    )
    assert not audit["pass"]
    assert audit["mass_error_pct"] > 2.0


# ===================================================================
# Integrated chain test
# ===================================================================

def test_full_chain_all_pass():
    """Full chain with all good measurements must pass overall."""
    proto = SeepagePrototype(
        height=0.10, porosity=0.40, rho_s=2650.0,
        diameter=2.0e-4, fine_fraction=0.20,
    )
    col = SaturatedColumn(height=proto.height, porosity=proto.porosity, rho_s=proto.rho_s)
    z = np.linspace(0.0, proto.height, 21)
    u_analytic = col.pore_pressure(z)
    # Small noise (3 %)
    rng = np.random.default_rng(42)
    u_measured = u_analytic * (1.0 + 0.03 * rng.standard_normal(len(z)))
    u_measured[0] = 0.0

    i_c_analytic = proto.terzaghi_i_c
    i_c_measured = i_c_analytic * 1.05  # 5 % error

    kc_model = KozenyCarmanModel(diameter=proto.diameter)
    phis = [0.35, 0.40, 0.45]
    k_analytic = [kc_model.conductivity(phi) for phi in phis]
    k_measured = [k * 1.10 for k in k_analytic]  # 10 % error

    env = SkemptonBroganEnvelope()
    i_c_suff = env.critical_gradient(proto.terzaghi_i_c, proto.fine_fraction)
    grads = [0.1, 0.2, i_c_suff * 0.9, i_c_suff * 1.1, i_c_suff * 1.3]
    rates = [0.0, 0.0, 0.0, 1e-5, 5e-5]

    result = run_seepage_chain(
        prototype=proto,
        measured_pore_pressures={"z": z.tolist(), "u": u_measured.tolist()},
        measured_critical_gradient=i_c_measured,
        measured_permeabilities={"porosities": phis, "k_measured": k_measured},
        measured_erosion_rates={"gradients": grads, "rates": rates},
        mass_audit={
            "initial_solid_mass": 10.0,
            "initial_fluid_mass": 5.0,
            "final_solid_mass": 9.95,
            "final_fluid_mass": 5.0,
            "fine_mass_lost": 0.05,
        },
    )
    assert result["overall_pass"], (
        f"Chain failed: C1={result['C1_pore_pressure']['pass']}, "
        f"C2={result['C2_critical_gradient']['pass']}, "
        f"C3={result['C3_permeability']['pass']}, "
        f"C4={result['C4_suffusion_trend']['pass']}, "
        f"C5={result['C5_mass_conservation']['pass']}"
    )


def test_piping_analysis_summary():
    """PipingAnalysis summary must contain all expected fields."""
    pa = PipingAnalysis(
        rho_s=2650.0, porosity=0.40, applied_gradient=0.5,
    )
    s = pa.summary()
    assert "i_c_terzaghi" in s
    assert "factor_of_safety" in s
    assert "piping_onset" in s
    assert s["piping_onset"] is False  # 0.5 < i_c ~ 1.0


def test_suffusion_analysis_summary():
    """SuffusionAnalysis summary must contain all expected fields."""
    sa = SuffusionAnalysis(
        rho_s=2650.0, porosity=0.40, fine_fraction=0.20,
        applied_gradient=0.5,
    )
    s = sa.summary()
    assert "terzaghi_i_c" in s
    assert "i_c_suffusion" in s
    assert "alpha" in s
    assert "is_eroding" in s
