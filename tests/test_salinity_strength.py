"""Acceptance tests for P54-W3-N01: MYI/FYI salinity-strength differentiation.

Each test is tagged with the campaign criterion it discharges:

* (a) brine volume fraction at T=-10 C, S=5 ppt within 2% of the literature value
* (b) FYI compressive strength at T=-10 C, S=5 ppt within 10% of the literature value
* (c) MYI/FYI ordering -- recorded as an internal *conflict* of the campaign
      metadata, see ``test_criterion_c_conflict_memorandum``
* (d) sigma_c monotone decreasing in T over 17 levels (|Spearman rho| = 1)
* (e) sigma_c monotone decreasing in S over 16 levels (|Spearman rho| = 1)
* (f) negative controls: S=0 -> nu_b=0 and sigma_c=sigma_0 ; T=0 -> sigma_c=0
* (g) provenance: every model constant has a documented source
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.salinity_strength import (
    DEFAULT_PARAMETERS,
    ICECLASS_DEFAULT_SALINITY,
    LIQUIDUS_A1,
    LIQUIDUS_A2,
    PARAMETER_PROVENANCE,
    RHO_ICE_G_CM3,
    StrengthParameters,
    brine_density,
    brine_salinity,
    brine_volume_fraction,
    compressive_strength,
    monte_carlo_strength,
    porosity_strength_factor,
    salinity_scan,
    spearman_rho,
    strength_surface,
    temperature_scan,
)

# --- literature anchors (digitised, see docs REPORT.md "reference values") ---
NU_B_REF_T10_S5 = 0.0270      # Timco & O'Brien (1994) brine-volume nomograph
SIGMA_REF_T10_S5_PA = 4.4e6   # Timco & Frederking (1984), 1e-3 1/s, columnar FY


# --------------------------------------------------------------------- liquidus
def test_brine_salinity_zero_at_freezing_point():
    """No brine above 0 C: S_b(0) = 0 and S_b(+1) = 0."""
    assert brine_salinity(0.0) == 0.0
    assert brine_salinity(1.0) == 0.0


def test_liquidus_round_trip_is_exact():
    """Inverting T_f(S) must reproduce the temperature it started from."""
    for t in (-0.5, -2.0, -5.0, -10.0, -20.0, -30.0):
        sb = brine_salinity(t)
        assert -((LIQUIDUS_A1 * sb) + LIQUIDUS_A2 * sb * sb) == pytest.approx(t, abs=1e-9)


def test_brine_salinity_monotone_in_temperature():
    """Colder ice holds more saline brine (strictly monotone)."""
    temps = np.linspace(-0.1, -40.0, 40)
    sb = [brine_salinity(t) for t in temps]
    assert all(b > a for a, b in zip(sb, sb[1:]))


def test_brine_density_increases_with_salinity():
    assert brine_density(0.0) == pytest.approx(1.0)
    assert brine_density(200.0) > brine_density(100.0) > brine_density(50.0)


# ---------------------------------------------------------- criterion (a): nu_b
def test_criterion_a_brine_volume_fraction_vs_literature():
    """Cox & Weeks mass balance at T=-10 C, S=5 ppt within 2% of nomograph."""
    nu = brine_volume_fraction(-10.0, 5.0)
    rel = abs(nu - NU_B_REF_T10_S5) / NU_B_REF_T10_S5
    assert rel < 0.02, f"nu_b={nu:.5f} vs ref {NU_B_REF_T10_S5}: rel={rel:.2%}"


def test_criterion_a_brine_volume_linear_in_salinity():
    """Mass balance is linear in bulk salinity with a temperature-only slope."""
    nu5 = brine_volume_fraction(-15.0, 5.0)
    nu10 = brine_volume_fraction(-15.0, 10.0)
    assert nu10 == pytest.approx(2.0 * nu5, rel=1e-12)


# ------------------------------------------------- criterion (b): FYI strength
def test_criterion_b_fyi_strength_vs_literature():
    """Timco & Frederking porosity law at T=-10 C, S=5 ppt within 10%."""
    sig = compressive_strength(-10.0, 5.0, ice_class="FY")
    rel = abs(sig - SIGMA_REF_T10_S5_PA) / SIGMA_REF_T10_S5_PA
    assert rel < 0.10, f"sigma_c={sig:.4e} Pa vs ref {SIGMA_REF_T10_S5_PA:.2e}: rel={rel:.2%}"


def test_strength_is_positive_and_plausible_for_fy_ice():
    """FY strength at field conditions lies in the documented 1-6 MPa band."""
    for t, s in ((-2.0, 5.0), (-5.0, 8.0), (-10.0, 5.0), (-20.0, 4.0)):
        sig = compressive_strength(t, s, ice_class="FY")
        assert 1.0e6 < sig < 6.0e6, (t, s, sig)


# ------------------------------------------------- criterion (c): MYI vs FYI
def test_criterion_c_conflict_memorandum():
    """The campaign metadata for (c) contradicts its own path (3) and physics.

    Path (3) requires ``sigma_c_MY > sigma_c_FY`` at equal (T, S); criterion (c)
    requires ``sigma_c_MY < sigma_c_FY``.  Both cannot hold.  The model follows
    path (3) + literature (Timco & Weeks 2010: MY ice is stronger), and this
    test *pins the conflict* so a future reader cannot silently "fix" it.
    """
    my = compressive_strength(-10.0, 5.0, ice_class="MY")
    fy = compressive_strength(-10.0, 5.0, ice_class="FY")
    assert my > fy  # resolves in favour of path (3)
    assert my / fy == pytest.approx(DEFAULT_PARAMETERS.myi_structural_factor, rel=1e-12)
    # the literal criterion (c) reading -- recorded, and it FAILS
    literal_c_met = my < fy
    assert literal_c_met is False


def test_myi_typical_salinity_beats_fyi_typical_salinity():
    """With *typical* salinities (MY 1 ppt, FY 5 ppt) MY ice is ~1.3-2x stronger."""
    my = compressive_strength(-10.0, ICECLASS_DEFAULT_SALINITY["MY"], ice_class="MY")
    fy = compressive_strength(-10.0, ICECLASS_DEFAULT_SALINITY["FY"], ice_class="FY")
    assert 1.3 <= my / fy <= 2.0, my / fy


# ------------------------------------------------ criteria (d)/(e): monotonicity
def test_criterion_d_temperature_trend_spearman():
    """T from -40 C to 0 C: sigma_c strictly decreases over 17 levels."""
    rows = temperature_scan(5.0, t_min=-40.0, t_max=0.0, n=17, ice_class="FY")
    assert len(rows) == 17
    t = [r["temperature_c"] for r in rows]
    sig = [r["sigma_c_pa"] for r in rows]
    rho = spearman_rho(t, sig)
    assert rho == pytest.approx(-1.0)
    assert all(b < a for a, b in zip(sig, sig[1:])), "sigma_c must strictly decrease with T"


def test_criterion_e_salinity_trend_spearman():
    """S from 0 to 15 ppt: sigma_c strictly decreases over 16 levels."""
    rows = salinity_scan(-10.0, s_min=0.0, s_max=15.0, n=16, ice_class="FY")
    assert len(rows) == 16
    s = [r["salinity_ppt"] for r in rows]
    sig = [r["sigma_c_pa"] for r in rows]
    rho = spearman_rho(s, sig)
    assert rho == pytest.approx(-1.0)
    assert all(b < a for a, b in zip(sig, sig[1:])), "sigma_c must strictly decrease with S"


# ---------------------------------------------------- criterion (f): controls
def test_criterion_f_zero_salinity_gives_zero_brine_and_sigma0():
    assert brine_volume_fraction(-10.0, 0.0) == 0.0
    sig = compressive_strength(-10.0, 0.0, ice_class="FY")
    assert sig == pytest.approx(DEFAULT_PARAMETERS.sigma0_pa, rel=1e-12)


def test_criterion_f_freezing_point_gives_zero_strength():
    """T = 0 C: fully liquid, no load-bearing matrix, sigma_c = 0."""
    assert brine_volume_fraction(0.0, 5.0) == 1.0
    assert compressive_strength(0.0, 5.0, ice_class="FY") == 0.0
    assert compressive_strength(0.0, 5.0, ice_class="MY") == 0.0


def test_strength_vanishes_beyond_porosity_limit():
    """nu_b >= nu_p -> zero strength (clipped, never negative)."""
    assert porosity_strength_factor(0.27) == 0.0
    assert porosity_strength_factor(0.40) == 0.0
    # warm, very saline ice is beyond the limit
    assert compressive_strength(-0.5, 15.0) == 0.0


# ------------------------------------------------------------ uncertainty (4)
def test_monte_carlo_band_ordering_and_containment():
    mc = monte_carlo_strength(-10.0, 5.0, ice_class="FY", n_samples=4000)
    assert mc["p05_pa"] < mc["p50_pa"] < mc["p95_pa"]
    assert mc["p05_pa"] <= mc["deterministic_sigma_c_pa"] <= mc["p95_pa"]


def test_monte_carlo_is_bitwise_reproducible():
    a = monte_carlo_strength(-10.0, 5.0, ice_class="MY", n_samples=2000, seed=7)
    b = monte_carlo_strength(-10.0, 5.0, ice_class="MY", n_samples=2000, seed=7)
    assert a == b


def test_monte_carlo_band_widens_towards_the_strength_cliff():
    """Relative spread grows as porosity approaches the limit (nonlinearity)."""
    cold = monte_carlo_strength(-30.0, 5.0, n_samples=4000)
    warm = monte_carlo_strength(-5.0, 5.0, n_samples=4000)
    rel_cold = (cold["p95_pa"] - cold["p05_pa"]) / cold["p50_pa"]
    rel_warm = (warm["p95_pa"] - warm["p05_pa"]) / warm["p50_pa"]
    assert rel_warm > rel_cold


# ------------------------------------------------------------------- utilities
def test_spearman_rho_matches_scipy_with_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    x = rng.normal(size=25)
    y = np.round(rng.normal(size=25), 1)  # force ties
    assert spearman_rho(x, y) == pytest.approx(scipy_stats.spearmanr(x, y).statistic, rel=1e-12)


def test_strength_surface_shape_and_consistency():
    temps = [-30.0, -20.0, -10.0, -5.0]
    sals = [0.5, 2.0, 5.0, 10.0, 14.0]
    surf = strength_surface(temps, sals, ice_class="MY")
    assert surf.shape == (len(temps), len(sals))
    for i, t in enumerate(temps):
        for j, s in enumerate(sals):
            assert surf[i, j] == pytest.approx(
                compressive_strength(t, s, ice_class="MY"), rel=1e-12
            )


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        brine_volume_fraction(-10.0, -1.0)
    with pytest.raises(ValueError):
        compressive_strength(-10.0, 5.0, ice_class="THIRD_YEAR")
    with pytest.raises(ValueError):
        brine_salinity(float("nan"))
    with pytest.raises(ValueError):
        spearman_rho([1.0], [1.0])


# ------------------------------------------------------ criterion (g): provenance
def test_criterion_g_every_constant_has_a_source():
    required = {
        "liquidus_a1",
        "liquidus_a2",
        "brine_density_slope",
        "rho_ice",
        "sigma0_pa",
        "porosity_limit",
        "porosity_exponent",
        "myi_structural_factor",
        "myi_salinity_ppt",
        "fyi_salinity_ppt",
    }
    assert required <= set(PARAMETER_PROVENANCE)
    for name, entry in PARAMETER_PROVENANCE.items():
        assert isinstance(entry.get("source"), str) and len(entry["source"]) > 20, name
        assert "value" in entry, name


def test_parameters_are_overridable_and_change_the_answer():
    weak = StrengthParameters(sigma0_pa=3.0e6)
    strong = StrengthParameters(sigma0_pa=6.0e6)
    a = compressive_strength(-10.0, 5.0, params=weak)
    b = compressive_strength(-10.0, 5.0, params=strong)
    assert b > a
    assert RHO_ICE_G_CM3 == pytest.approx(0.917)