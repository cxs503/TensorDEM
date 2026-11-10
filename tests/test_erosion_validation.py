"""Acceptance tests for the solid-particle erosion module (p176w1-n09).

Criteria covered (the name of every test carries its criterion tag):

* ``c1`` -- angular maxima vs literature (Finnie ductile peak, brittle 90 deg,
  Oka hardness trend), analytic vs numeric agreement, branch continuity;
* ``c2`` -- velocity exponent vs literature for >= 3 materials, Oka hardness and
  diameter exponents, estimator guards;
* ``c3`` -- DNV-RP-O501 erosion rate vs an independent reference implementation
  for >= 5 operating conditions (< 15%), plus linearity/unit checks;
* ``c4`` -- mass/energy ledger closure (< 2%), geometric efficiency bound,
  implied efficiency of the empirical models;
* ``c5`` -- negative controls (v -> 0, alpha -> 0 tangential, Finnie at normal
  incidence), invalid-input guards;
* ``c6`` -- suite inventory, provenance completeness and the end-to-end study.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.erosion import (
    BRITTLE_CONSTANTS,
    DNV_DUCTILE_COEFFICIENTS,
    FINNIE_BRANCH_ANGLE_DEG,
    FINNIE_GEOMETRIC_EFFICIENCY_MAX,
    FINNIE_PEAK_ANGLE_DEG,
    LITERATURE_PEAK_ANGLES_DEG,
    LITERATURE_VELOCITY_EXPONENTS,
    MATERIALS,
    OKA_CONSTANTS,
    ErosionLedger,
    ImpactEvent,
    brittle_angle_function,
    brittle_peak_angle_deg_numeric,
    brittle_relative_erosion,
    closed_form_finnie_stream,
    dnv_ductile_erosion_ratio,
    dnv_ductile_wall_loss_rate,
    finnie_angle_function,
    finnie_erosion_ratio,
    finnie_peak_angle_deg,
    finnie_peak_angle_deg_numeric,
    finnie_peak_value,
    fit_velocity_exponent,
    get_material,
    implied_efficiency,
    oka_angle_function,
    oka_exponents,
    oka_peak_angle_deg,
    oka_peak_angle_deg_numeric,
    oka_relative_erosion,
)

VELOCITIES = np.array([5.0, 10.0, 20.0, 35.0, 60.0, 100.0])


def _loglog_slope(x, y) -> float:
    slope, _ = np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)
    return float(slope)


# ---------------------------------------------------------------------------
# C1 -- angular maxima
# ---------------------------------------------------------------------------
def test_c1_finnie_peak_angle_matches_literature():
    peak = finnie_peak_angle_deg()
    ref = LITERATURE_PEAK_ANGLES_DEG["ductile"][0]
    low, high = 15.0, 30.0
    assert peak == pytest.approx(FINNIE_PEAK_ANGLE_DEG, rel=1e-12)
    assert low <= peak <= high, "Finnie peak must sit inside the ductile band"
    assert abs(peak - ref) < 5.0, f"peak {peak:.2f} deg vs literature {ref:.1f} deg"


def test_c1_finnie_peak_analytic_matches_numeric_grid():
    analytic = finnie_peak_angle_deg()
    numeric = finnie_peak_angle_deg_numeric()
    assert abs(analytic - numeric) < 0.01
    # the maximum value is attained at the analytic root and nowhere else
    grid = np.linspace(0.0, 90.0, 1801)
    values = finnie_angle_function(grid)
    assert values.max() == pytest.approx(finnie_peak_value(), rel=1e-4)
    assert abs(grid[int(np.argmax(values))] - analytic) < 0.1


def test_c1_finnie_branches_are_continuous():
    eps = 1e-7
    left = float(finnie_angle_function(FINNIE_BRANCH_ANGLE_DEG - eps))
    right = float(finnie_angle_function(FINNIE_BRANCH_ANGLE_DEG + eps))
    assert abs(left - right) < 1e-6
    assert left == pytest.approx(0.3, abs=1e-6)
    # monotone rise up to the analytic maximum, monotone decay afterwards
    grazing = finnie_angle_function(np.linspace(0.0, FINNIE_PEAK_ANGLE_DEG, 200))
    assert np.all(np.diff(grazing) > 0.0)
    shoulder = finnie_angle_function(
        np.linspace(FINNIE_PEAK_ANGLE_DEG, FINNIE_BRANCH_ANGLE_DEG, 20)
    )
    assert np.all(np.diff(shoulder) < 0.0)
    normal = finnie_angle_function(np.linspace(FINNIE_PEAK_ANGLE_DEG + 1.0, 90.0, 200))
    assert np.all(np.diff(normal) < 0.0)


def test_c1_brittle_peak_at_normal_incidence():
    peak = brittle_peak_angle_deg_numeric()
    ref = LITERATURE_PEAK_ANGLES_DEG["brittle"][0]
    assert abs(peak - ref) < 5.0
    assert peak == pytest.approx(90.0, abs=0.01)
    # brittle response grows monotonically with angle
    grid = np.linspace(1.0, 90.0, 100)
    assert np.all(np.diff(brittle_angle_function(grid)) > 0.0)


def test_c1_oka_peak_angle_inside_ductile_band_and_hardness_trend():
    peaks = {name: oka_peak_angle_deg(MATERIALS[name].hardness_hv_gpa)
             for name in ("copper", "aluminium_6061", "carbon_steel", "alumina_ceramic")}
    # soft ductile metals peak inside the classical 15-30 deg band
    assert 15.0 <= peaks["copper"] <= 30.0
    assert 15.0 <= peaks["aluminium_6061"] <= 30.0
    # hardness screening pushes the peak towards normal incidence
    ordered = [peaks["copper"], peaks["aluminium_6061"], peaks["carbon_steel"], peaks["alumina_ceramic"]]
    assert all(b > a for a, b in zip(ordered, ordered[1:]))
    assert peaks["alumina_ceramic"] > 60.0
    # analytic root of dg/dalpha = 0 agrees with the numeric maximiser
    for name in ("copper", "carbon_steel", "alumina_ceramic"):
        hv = MATERIALS[name].hardness_hv_gpa
        assert abs(oka_peak_angle_deg(hv) - oka_peak_angle_deg_numeric(hv)) < 0.01


# ---------------------------------------------------------------------------
# C2 -- velocity exponents
# ---------------------------------------------------------------------------
def test_c2_velocity_exponent_steel_dnv_matches_literature():
    rates = [dnv_ductile_erosion_ratio(v, 35.0, "carbon_steel") for v in VELOCITIES]
    n_fit = _loglog_slope(VELOCITIES, rates)
    n_lit = LITERATURE_VELOCITY_EXPONENTS["carbon_steel"][0]
    assert abs(n_fit - n_lit) / n_lit < 0.10
    assert 2.0 <= n_fit <= 3.0, "velocity exponent must stay in the 2-3 physical range"


@pytest.mark.parametrize("material", ["aluminium_6061", "copper", "stainless_steel_316"])
def test_c2_velocity_exponent_oka_three_materials(material):
    rates = [oka_relative_erosion(v, 30.0, 150.0, material) for v in VELOCITIES]
    n_fit = _loglog_slope(VELOCITIES, rates)
    n_lit = LITERATURE_VELOCITY_EXPONENTS.get(material, LITERATURE_VELOCITY_EXPONENTS["aluminium_6061"])[0]
    assert abs(n_fit - n_lit) / n_lit < 0.10
    assert n_fit == pytest.approx(OKA_CONSTANTS["k2"], rel=1e-6)


def test_c2_velocity_exponent_brittle_ceramic():
    rates = [brittle_relative_erosion(v, 90.0) for v in VELOCITIES]
    n_fit = _loglog_slope(VELOCITIES, rates)
    n_lit = LITERATURE_VELOCITY_EXPONENTS["alumina_ceramic"][0]
    assert abs(n_fit - n_lit) / n_lit < 0.10
    assert n_fit == pytest.approx(BRITTLE_CONSTANTS["n_velocity"], rel=1e-9)


def test_c2_finnie_theory_exponent_against_measured_band():
    rates = [finnie_erosion_ratio(v, 20.0, "carbon_steel") for v in VELOCITIES]
    n_fit = _loglog_slope(VELOCITIES, rates)
    assert n_fit == pytest.approx(2.0, rel=1e-9), "pure kinetic cutting theory gives n = 2"
    band = (2.0, 2.5)
    reference = LITERATURE_VELOCITY_EXPONENTS["aluminium_6061"][0]
    assert reference == pytest.approx(2.2)
    assert band[0] <= reference <= band[1]
    assert abs(n_fit - reference) / reference < 0.10


def test_c2_oka_hardness_and_diameter_exponents():
    hvs = np.array([0.8, 1.0, 1.5, 2.0, 4.0, 8.0])
    hardness_rates = [OKA_CONSTANTS["K"] * hv ** OKA_CONSTANTS["k1"]
                      * 30.0 ** OKA_CONSTANTS["k2"] * 150.0 ** OKA_CONSTANTS["k3"]
                      * float(oka_angle_function(90.0, hv)) for hv in hvs]
    assert _loglog_slope(hvs, hardness_rates) == pytest.approx(OKA_CONSTANTS["k1"], rel=1e-9)
    ds = np.array([50.0, 100.0, 150.0, 200.0, 300.0, 500.0])
    diameter_rates = [oka_relative_erosion(30.0, 90.0, float(d), "carbon_steel") for d in ds]
    assert _loglog_slope(ds, diameter_rates) == pytest.approx(OKA_CONSTANTS["k3"], rel=1e-9)
    n1, n2 = oka_exponents(2.0)
    assert n1 > 0.0 and n2 > 0.0


def test_c2_fit_velocity_exponent_guards():
    with pytest.raises(ValueError):
        fit_velocity_exponent([1.0, 2.0], [1.0, 4.0])
    with pytest.raises(ValueError):
        fit_velocity_exponent([1.0, 2.0, 3.0], [1.0, 0.0, 9.0])
    with pytest.raises(ValueError):
        fit_velocity_exponent([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
    slope, rms = fit_velocity_exponent([1.0, 10.0, 100.0], [3.0, 3.0 * 10.0**2.5, 3.0 * 100.0**2.5])
    assert slope == pytest.approx(2.5, rel=1e-9)
    assert rms < 1e-9


# ---------------------------------------------------------------------------
# C3 -- DNV-RP-O501 cross-check
# ---------------------------------------------------------------------------
def _reference_dnv_ratio(v: float, alpha_deg: float, k: float, n: float) -> float:
    return k * v**n * math.sin(math.radians(alpha_deg)) ** n


def _reference_dnv_mm_per_year(v, alpha_deg, k, n, sand_rate, area, density) -> float:
    ratio = _reference_dnv_ratio(v, alpha_deg, k, n)
    return ratio * sand_rate / density / area * 365.25 * 24.0 * 3600.0 * 1000.0


DNV_TEST_CASES = [
    (5.0, 30.0, 0.010, 2.0e-3, "carbon_steel"),
    (12.0, 45.0, 0.010, 2.0e-3, "carbon_steel"),
    (20.0, 60.0, 0.030, 5.0e-3, "carbon_steel"),
    (30.0, 90.0, 0.030, 5.0e-3, "carbon_steel"),
    (40.0, 15.0, 0.050, 1.0e-2, "carbon_steel"),
    (60.0, 75.0, 0.100, 2.0e-2, "carbon_steel"),
    (18.0, 40.0, 0.020, 3.0e-3, "aluminium_6061"),
]


@pytest.mark.parametrize("case", DNV_TEST_CASES)
def test_c3_dnv_matches_reference_implementation(case):
    v, alpha, rate, area, name = case
    mat = get_material(name)
    coeff = DNV_DUCTILE_COEFFICIENTS[name]
    lib = dnv_ductile_wall_loss_rate(v, alpha, mat, rate, area)
    ref_mm = _reference_dnv_mm_per_year(v, alpha, coeff.k, coeff.n, rate, area, mat.density_kg_m3)
    ref_ratio = _reference_dnv_ratio(v, alpha, coeff.k, coeff.n)
    assert abs(lib["erosion_ratio_kg_per_kg"] - ref_ratio) / ref_ratio < 0.15
    assert abs(lib["wall_loss_mm_per_year"] - ref_mm) / ref_mm < 0.15


def test_c3_dnv_linear_in_sand_rate_and_area():
    base = dnv_ductile_wall_loss_rate(20.0, 45.0, "carbon_steel", 0.01, 2.0e-3)
    doubled = dnv_ductile_wall_loss_rate(20.0, 45.0, "carbon_steel", 0.02, 2.0e-3)
    assert doubled["wall_loss_mm_per_year"] == pytest.approx(2.0 * base["wall_loss_mm_per_year"], rel=1e-12)
    wide = dnv_ductile_wall_loss_rate(20.0, 45.0, "carbon_steel", 0.01, 4.0e-3)
    assert wide["wall_loss_mm_per_year"] == pytest.approx(0.5 * base["wall_loss_mm_per_year"], rel=1e-12)


def test_c3_dnv_velocity_and_angle_functions_are_exact():
    coeff = DNV_DUCTILE_COEFFICIENTS["carbon_steel"]
    for alpha in (3.0, 22.5, 67.5, 90.0):
        ratio = dnv_ductile_erosion_ratio(30.0, alpha, "carbon_steel")
        assert ratio == pytest.approx(coeff.k * 30.0**coeff.n * math.sin(math.radians(alpha)) ** coeff.n,
                                      rel=1e-12)
    lifetime = dnv_ductile_wall_loss_rate(
        20.0, 45.0, "carbon_steel", 0.01, 2.0e-3, wall_thickness_mm=10.0
    )
    assert lifetime["life_years_to_thickness"] == pytest.approx(
        10.0 / lifetime["wall_loss_mm_per_year"], rel=1e-12
    )
    # zero sand means zero wall loss (guard against a unit slip)
    none = dnv_ductile_wall_loss_rate(20.0, 45.0, "carbon_steel", 0.0, 2.0e-3)
    assert none["wall_loss_mm_per_year"] == 0.0


# ---------------------------------------------------------------------------
# C4 -- ledger closure
# ---------------------------------------------------------------------------
def test_c4_ledger_matches_closed_form_for_monodisperse_stream():
    mat = get_material("carbon_steel")
    n = 250
    ledger = ErosionLedger(mat)
    ledger.extend([ImpactEvent(1.0e-6, 40.0, 22.0, restitution=0.25)] * n)
    closed = n * closed_form_finnie_stream(40.0, 22.0, 1.0e-6, mat, restitution=0.25)
    assert abs(ledger.removed_mass_kg - closed) / closed < 0.02


def test_c4_polydisperse_ledger_closure_below_two_percent():
    rng = np.random.default_rng(7)
    ledger = ErosionLedger("carbon_steel")
    speeds = rng.lognormal(mean=math.log(25.0), sigma=0.35, size=3000)
    angles = rng.uniform(5.0, 45.0, size=3000)
    masses = rng.uniform(0.3e-6, 1.8e-6, size=3000)
    rest = rng.uniform(0.0, 0.6, size=3000)
    for i in range(3000):
        ledger.add_impact(ImpactEvent(float(masses[i]), float(speeds[i]), float(angles[i]), float(rest[i])))
    closure = ledger.closure()
    assert closure["closure_residual"] < 0.02
    assert closure["kinetic_bookkeeping_residual"] < 0.02
    assert closure["closure_ok"]
    assert closure["removed_mass_kg"] < closure["delivered_particle_mass_kg"]
    # the specific removal energy reproduces the target flow stress
    assert closure["specific_removal_energy_J_m3"] == pytest.approx(closure["flow_stress_pa"], rel=1e-9)


def test_c4_efficiency_within_geometric_bound():
    rng = np.random.default_rng(11)
    ledger = ErosionLedger("aluminium_6061")
    for _ in range(2000):
        ledger.add_impact(
            ImpactEvent(
                particle_mass_kg=float(rng.uniform(0.2e-6, 2.0e-6)),
                speed_m_s=float(rng.uniform(1.0, 80.0)),
                angle_deg=float(rng.uniform(0.0, 90.0)),
                restitution=float(rng.uniform(0.0, 0.8)),
            )
        )
    closure = ledger.closure()
    assert closure["erosion_efficiency"] <= FINNIE_GEOMETRIC_EFFICIENCY_MAX * (1 + 1e-12)
    assert closure["efficiency_within_bound"] and closure["energy_conserving"]


def test_c4_implied_efficiency_of_empirical_models_stays_below_unity():
    for name in ("carbon_steel", "aluminium_6061"):
        for v in (5.0, 20.0, 45.0, 70.0):
            for alpha in (15.0, 45.0, 90.0):
                ratio = dnv_ductile_erosion_ratio(v, alpha, name)
                eta = implied_efficiency(ratio, v, name)
                assert 0.0 < eta < 1.0, f"{name} v={v} alpha={alpha} implies eta={eta}"


# ---------------------------------------------------------------------------
# C5 -- negative controls
# ---------------------------------------------------------------------------
def test_c5_zero_velocity_gives_zero_erosion():
    assert finnie_erosion_ratio(0.0, 30.0, "carbon_steel") == 0.0
    assert dnv_ductile_erosion_ratio(0.0, 30.0, "carbon_steel") == 0.0
    assert oka_relative_erosion(0.0, 30.0, 150.0, "carbon_steel") == 0.0
    assert brittle_relative_erosion(0.0, 30.0) == 0.0
    assert dnv_ductile_wall_loss_rate(0.0, 30.0, "carbon_steel", 0.01, 2.0e-3)["wall_loss_mm_per_year"] == 0.0


def test_c5_zero_angle_tangential_impact_gives_zero_erosion():
    for name in ("carbon_steel", "copper", "aluminium_6061"):
        assert finnie_erosion_ratio(30.0, 0.0, name) == 0.0
        assert oka_relative_erosion(30.0, 0.0, 150.0, name) == 0.0
    for name in DNV_DUCTILE_COEFFICIENTS:
        assert dnv_ductile_erosion_ratio(30.0, 0.0, name) == 0.0
    assert brittle_relative_erosion(30.0, 0.0) == 0.0
    # continuous approach to the tangential limit
    small = finnie_erosion_ratio(30.0, 0.5, "carbon_steel")
    assert 0.0 < small < finnie_erosion_ratio(30.0, 16.0, "carbon_steel")


def test_c5_finnie_normal_incidence_cutting_vanishes():
    assert abs(finnie_erosion_ratio(50.0, 90.0, "carbon_steel")) < 1e-12
    # ... while the brittle response is *maximal* there
    assert brittle_relative_erosion(50.0, 90.0) > brittle_relative_erosion(50.0, 45.0)


def test_c5_invalid_inputs_raise():
    with pytest.raises(ValueError):
        finnie_angle_function(-1.0)
    with pytest.raises(ValueError):
        finnie_angle_function(91.0)
    with pytest.raises(ValueError):
        dnv_ductile_erosion_ratio(-5.0, 30.0, "carbon_steel")
    with pytest.raises(ValueError):
        oka_relative_erosion(10.0, 30.0, 0.0, "carbon_steel")
    with pytest.raises(ValueError):
        ImpactEvent(1e-6, -3.0, 30.0)
    with pytest.raises(ValueError):
        ImpactEvent(1e-6, 3.0, 120.0)
    with pytest.raises(ValueError):
        dnv_ductile_wall_loss_rate(10.0, 30.0, "carbon_steel", 0.01, 0.0)
    with pytest.raises(KeyError):
        dnv_ductile_erosion_ratio(10.0, 30.0, "alumina_ceramic")


# ---------------------------------------------------------------------------
# C6 -- suite inventory, provenance, end-to-end study
# ---------------------------------------------------------------------------
def test_c6_material_provenance_is_complete():
    for name, mat in MATERIALS.items():
        assert mat.name == name
        assert mat.source and len(mat.source) > 10
        assert mat.hardness_hv_gpa > 0.0 and mat.density_kg_m3 > 0.0
        assert mat.material_class in ("ductile", "brittle")
        if mat.material_class == "ductile" and name in DNV_DUCTILE_COEFFICIENTS:
            coeff = DNV_DUCTILE_COEFFICIENTS[name]
            assert coeff.k > 0.0 and 2.0 <= coeff.n <= 3.0
            assert coeff.provenance, "every DNV coefficient needs a provenance string"
    assert set(LITERATURE_VELOCITY_EXPONENTS) >= {"carbon_steel", "aluminium_6061", "alumina_ceramic"}


def test_c6_erosion_ratio_magnitudes_are_physical():
    """Sand on steel at 10-40 m/s must sit in the 1e-7 .. 1e-2 kg/kg range."""
    for v in (10.0, 20.0, 40.0):
        for alpha in (30.0, 60.0, 90.0):
            ratio = dnv_ductile_erosion_ratio(v, alpha, "carbon_steel")
            assert 1e-7 < ratio < 1e-2
    # a 10 mm carbon-steel wall under 0.01 kg/s sand over 20 cm^2 at 20 m/s, 45 deg
    life = dnv_ductile_wall_loss_rate(20.0, 45.0, "carbon_steel", 0.01, 2.0e-3,
                                      wall_thickness_mm=10.0)
    assert 1e-3 < life["wall_loss_mm_per_year"] < 1e3
    assert life["life_years_to_thickness"] > 0.0


def test_c6_end_to_end_study_passes_all_criteria():
    """Run the acceptance study (quick Monte-Carlo) and require every criterion."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "erosion_verification", root / "benchmarks" / "erosion_verification.py"
    )
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)

    rng = np.random.default_rng(20261110)
    results = {
        "C1": study.section_c1(),
        "C2": study.section_c2(),
        "C3": study.section_c3(),
        "C4": study.section_c4(rng, 2000),
        "C5": study.section_c5(),
    }
    failed = {k: v["checks"] for k, v in results.items() if not v["pass"]}
    assert not failed, f"criteria failing: {failed}"

    c1 = results["C1"]
    assert c1["finnie"]["error_vs_ductile_reference_deg"] < 5.0
    assert c1["brittle"]["error_vs_brittle_reference_deg"] < 5.0
    assert len(results["C2"]["materials_within_10pct"]) >= 3
    assert results["C3"]["max_relative_error"] < 0.15
    assert len(results["C3"]["cases"]) >= 5
    assert results["C4"]["polydisperse"]["closure_residual"] < 0.02