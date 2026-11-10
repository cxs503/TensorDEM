"""Brash-ice channel resistance from granular mu(I) rheology.

Campaign node ``p307w1-n03-brash-ice-rheology`` acceptance surface.  The tests
are grouped exactly like the node's criteria:

(a) mu(I) constants: law limits, inertial-number definition, fitter recovery
    (the DEM measurement itself is exercised by ``test_dem_*`` which is opt-in);
(b) total resistance vs the in-repository brash anchor at B = 15/25/35 m;
(c) component split bands;  (d) speed exponent;  (e) porosity sensitivity;
(f) negative controls (v -> 0, phi -> 1) and monotonicity;
plus structural tests: component identity, beam/speed/porosity scaling laws,
input validation, provenance hashing, determinism, calibration reproducibility.
"""
from __future__ import annotations

import math
import os

import pytest

from tensordem.brash_ice_rheology.anchors import (anchor_components,
                                                  reference_anchor,
                                                  scaling_diagnostic)
from tensordem.brash_ice_rheology.calibration import (calibrate_confinement,
                                                      calibrate_family)
from tensordem.brash_ice_rheology.coefficients import (COEFFICIENT_LEDGER,
                                                       check_coefficient,
                                                       ledger_sha256)
from tensordem.brash_ice_rheology.material import STANDARD_BRASH, BrashIceState
from tensordem.brash_ice_rheology.model import (BrashHull, Closure,
                                                STANDARD_BEAMS_M,
                                                brash_resistance,
                                                buoyancy_component,
                                                keel_drag_component,
                                                porosity_sweep,
                                                power_law_exponent,
                                                shear_component, speed_sweep,
                                                standard_hull)
from tensordem.brash_ice_rheology.mu_i import (JopParameters, LITERATURE_JOP,
                                               SWEEP_TARGETS, inertial_number,
                                               mu_of_I)
from tensordem.brash_ice_rheology.validation import (B_TOLERANCE,
                                                     POROSITY_BAND,
                                                     SHARE_BANDS,
                                                     SPEED_EXPONENT_BAND,
                                                     criterion_b, criterion_c,
                                                     criterion_d, criterion_e,
                                                     criterion_f)

CLOSURE = Closure.from_ledger()


# ---------------------------------------------------------------------------
# (a) the mu(I) closure
# ---------------------------------------------------------------------------

def test_mu_i_low_inertial_limit_is_mu_s():
    params = JopParameters()
    assert params.mu(0.0) == pytest.approx(params.mu_s)
    assert params.mu(1e-9) == pytest.approx(params.mu_s, rel=1e-6)


def test_mu_i_high_inertial_limit_is_mu_2():
    params = JopParameters()
    assert params.mu(1e9) == pytest.approx(params.mu_2, rel=1e-6)


def test_mu_i_monotone_in_inertial_number():
    params = JopParameters()
    values = [params.mu(i) for i in (1e-3, 1e-2, 0.1, 0.3, 1.0, 10.0)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_inertial_number_matches_definition():
    # I = gamma_dot * d / sqrt(P/rho) = 2.0 * 0.5 / sqrt(2500/1000) = 0.632
    assert inertial_number(2.0, 0.5, 2500.0, 1000.0) == pytest.approx(
        2.0 * 0.5 / math.sqrt(2500.0 / 1000.0))
    assert mu_of_I(0.3) == pytest.approx(JopParameters().mu(0.3))


def test_mu_i_parameter_validation():
    with pytest.raises(ValueError):
        JopParameters(mu_s=0.9, mu_2=0.5)
    with pytest.raises(ValueError):
        JopParameters(i_0=0.0)
    with pytest.raises(ValueError):
        inertial_number(1.0, 0.5, 0.0, 1000.0)


def test_fit_recovers_jop_constants_from_law_generated_points():
    """The fitter (shared with the DEM calibration) recovers the law it fits."""
    truth = JopParameters(0.36, 0.84, 0.30)
    records = [{"inertial_number_measured": i, "mu_measured": truth.mu(i)}
               for i in SWEEP_TARGETS]
    from tensordem.brash_ice_rheology.mu_i import fit_sweep
    fit = fit_sweep(records)
    for name, value in truth.as_dict().items():
        assert fit["parameters"][name] == pytest.approx(value, rel=0.25), name
    assert fit["rms_relative"] < 1e-6


def test_literature_constants_are_the_pinned_jop_values():
    assert LITERATURE_JOP == (check_coefficient("mu_s")[0],
                              check_coefficient("mu_2")[0],
                              check_coefficient("i_0")[0])


# ---------------------------------------------------------------------------
# material state
# ---------------------------------------------------------------------------

def test_bulk_density_porosity_limits():
    solid = BrashIceState(porosity=0.0)
    water = BrashIceState(porosity=0.999999)
    assert solid.bulk_density_kg_m3 == pytest.approx(solid.solid_density_kg_m3)
    assert water.bulk_density_kg_m3 == pytest.approx(water.water_density_kg_m3, rel=1e-5)


def test_layer_floats_with_submerged_thickness_below_total():
    state = STANDARD_BRASH
    assert 0.0 < state.submerged_thickness_m < state.layer_thickness_m
    assert state.density_deficit_kg_m3 > 0.0
    # ~92 % submerged for phi = 0.3
    assert state.submerged_thickness_m / state.layer_thickness_m == pytest.approx(
        0.926, abs=0.01)


def test_static_confinement_is_a_few_hundred_pascal():
    """Buoyant confinement alone is O(100 Pa): the dynamic term must dominate."""
    p_static = STANDARD_BRASH.static_confinement_pa
    assert 100.0 < p_static < 1000.0
    res = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
    assert res.dynamic_confinement_pa > 3.0 * p_static


def test_state_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        BrashIceState(layer_thickness_m=0.0)
    with pytest.raises(ValueError):
        BrashIceState(porosity=1.0)
    with pytest.raises(ValueError):
        BrashIceState(solid_density_kg_m3=1100.0)  # denser than water -> sinks


# ---------------------------------------------------------------------------
# model structure
# ---------------------------------------------------------------------------

def test_total_is_the_sum_of_the_three_components():
    res = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
    assert res.total_N == pytest.approx(res.F_shear_N + res.F_buoy_N + res.F_drag_N)


def test_shares_sum_to_one_and_are_positive():
    res = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
    assert sum(res.shares.values()) == pytest.approx(1.0)
    assert all(s > 0.0 for s in res.shares.values())


def test_buoyancy_is_linear_in_beam():
    state = STANDARD_BRASH
    f15 = buoyancy_component(state, standard_hull(15.0), 3.0, CLOSURE)["F_buoy_N"]
    f35 = buoyancy_component(state, standard_hull(35.0), 3.0, CLOSURE)["F_buoy_N"]
    assert f35 / f15 == pytest.approx(35.0 / 15.0, rel=1e-12)


def test_keel_drag_is_quadratic_in_speed():
    state, hull = STANDARD_BRASH, standard_hull(25.0)
    d1 = keel_drag_component(state, hull, 1.0, CLOSURE)["F_drag_N"]
    d3 = keel_drag_component(state, hull, 3.0, CLOSURE)["F_drag_N"]
    # processing closure f(1)=1/1.05, f(3)=3/3.05 -> ratio contains 9 * f3/f1
    f = lambda v: v / (v + CLOSURE.mobilization_speed_m_s)
    assert d3 / d1 == pytest.approx(9.0 * f(3.0) / f(1.0))


def test_shear_grows_with_speed_and_with_confinement():
    state, hull = STANDARD_BRASH, standard_hull(25.0)
    s1 = shear_component(state, hull, 1.0, CLOSURE)["F_shear_N"]
    s3 = shear_component(state, hull, 3.0, CLOSURE)["F_shear_N"]
    assert s3 > s1
    weak = shear_component(state, hull, 3.0, CLOSURE.with_confinement(0.1))["F_shear_N"]
    strong = shear_component(state, hull, 3.0, CLOSURE.with_confinement(1.0))["F_shear_N"]
    assert strong > weak


def test_shear_stress_equals_mu_times_pressure():
    res = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
    assert res.shear_stress_pa == pytest.approx(
        res.mu_effective * res.confinement_pa * STANDARD_BRASH.solid_fraction)
    assert res.mu_effective == pytest.approx(CLOSURE.jop.mu(res.inertial_number))


def test_buoyancy_and_drag_fall_with_porosity():
    state, hull = STANDARD_BRASH, standard_hull(25.0)
    low = brash_resistance(state.with_porosity(0.2), hull, 3.0)
    high = brash_resistance(state.with_porosity(0.4), hull, 3.0)
    assert high.F_buoy_N < low.F_buoy_N
    assert high.F_drag_N < low.F_drag_N
    assert high.F_shear_N < low.F_shear_N


def test_input_validation_rejects_bad_cases():
    with pytest.raises(ValueError):
        brash_resistance(STANDARD_BRASH, standard_hull(25.0), -1.0)
    with pytest.raises(ValueError):
        BrashHull(beam_m=0.0)
    with pytest.raises(ValueError):
        Closure(mu_s=0.9, mu_2=0.5)
    with pytest.raises(ValueError):
        Closure(keel_drag_coefficient=-1.0)


def test_determinism_of_the_model():
    a = brash_resistance(STANDARD_BRASH, standard_hull(35.0), 2.5).as_dict()
    b = brash_resistance(STANDARD_BRASH, standard_hull(35.0), 2.5).as_dict()
    assert a == b


# ---------------------------------------------------------------------------
# (b) reference-anchor comparison
# ---------------------------------------------------------------------------

def test_reference_anchor_composition_and_source():
    payload = reference_anchor(STANDARD_BRASH, standard_hull(25.0), 3.0)
    assert "broken_ice_anchor" in payload["source"]
    total = payload["total_N"]
    comp = anchor_components(STANDARD_BRASH, standard_hull(25.0), 3.0)
    assert total == pytest.approx(comp["crushing_N"] + comp["bending_N"]
                                  + comp["submersion_N"] + comp["momentum_N"])
    assert comp["bending_N"] == 0.0  # floe 0.5 m << bending length


def test_criterion_b_within_tolerance_at_three_beams():
    record = criterion_b()
    assert record["verdict"] == "PASS", record
    assert record["max_abs_relative_error"] < B_TOLERANCE
    assert [row["beam_m"] for row in record["rows"]] == list(STANDARD_BEAMS_M)
    for row in record["rows"]:
        assert abs(row["relative_error"]) < B_TOLERANCE


def test_scaling_diagnostic_quantifies_the_beam_slope_mismatch():
    diag = scaling_diagnostic()
    # model terms all scale with the beam; the anchor's crushing term does not
    assert diag["model_beam_exponent"] > diag["anchor_beam_exponent"]
    assert diag["model_beam_exponent"] == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# (c) component split
# ---------------------------------------------------------------------------

def test_criterion_c_component_share_bands():
    record = criterion_c()
    assert record["verdict"] == "PASS", record
    for row in record["rows"]:
        for name, (lo, hi) in SHARE_BANDS.items():
            assert lo <= row["shares"][name] <= hi, (row["beam_m"], name)


# ---------------------------------------------------------------------------
# (d) speed dependence
# ---------------------------------------------------------------------------

def test_criterion_d_speed_exponent_in_band():
    record = criterion_d()
    lo, hi = SPEED_EXPONENT_BAND
    assert record["verdict"] == "PASS", record
    assert lo <= record["exponent"] <= hi


def test_power_law_exponent_recovers_a_known_exponent():
    speeds = [1.0, 2.0, 4.0, 8.0]
    totals = [10.0 * v ** 1.15 for v in speeds]
    assert power_law_exponent(speeds, totals) == pytest.approx(1.15, rel=1e-9)


def test_speed_sweep_is_monotone():
    records = speed_sweep(STANDARD_BRASH, standard_hull(25.0),
                          (0.5, 1.0, 2.0, 3.0, 5.0), CLOSURE)
    totals = [r["total_N"] for r in records]
    assert all(b > a for a, b in zip(totals, totals[1:]))


# ---------------------------------------------------------------------------
# (e) porosity sensitivity
# ---------------------------------------------------------------------------

def test_criterion_e_porosity_drop_in_band():
    record = criterion_e()
    lo, hi = POROSITY_BAND
    assert record["verdict"] == "PASS", record
    assert lo <= record["relative_change"] <= hi


def test_porosity_sweep_is_monotone_decreasing():
    records = porosity_sweep(STANDARD_BRASH, standard_hull(25.0), 3.0,
                             (0.2, 0.3, 0.4), CLOSURE)
    totals = [r["total_N"] for r in records]
    assert all(b < a for a, b in zip(totals, totals[1:]))


# ---------------------------------------------------------------------------
# (f) negative controls
# ---------------------------------------------------------------------------

def test_zero_speed_gives_exactly_zero_resistance():
    res = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 0.0)
    assert res.total_N == 0.0
    assert res.F_shear_N == 0.0 and res.F_buoy_N == 0.0 and res.F_drag_N == 0.0


def test_pure_water_limit_vanishes():
    res = brash_resistance(STANDARD_BRASH.with_porosity(0.999), standard_hull(25.0), 3.0)
    ref = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
    assert res.total_N / ref.total_N < 1.0e-2


def test_no_solid_ice_no_resistance():
    """phi -> 1 removes contacts (1-phi), buoyancy deficit and keel area alike."""
    water = STANDARD_BRASH.with_porosity(0.999999)
    for name, value in brash_resistance(water, standard_hull(25.0), 3.0).shares.items():
        assert value == pytest.approx(value)  # finite, not NaN
    assert brash_resistance(water, standard_hull(25.0), 3.0).total_N / \
        brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0).total_N < 1e-4


def test_criterion_f_negative_controls_pass():
    record = criterion_f()
    assert record["verdict"] == "PASS", record
    assert record["total_at_zero_speed_N"] == 0.0
    assert record["creep_speed_ratio"] < 1.0e-2
    assert record["pure_water_ratio"] < 1.0e-2
    assert record["monotone_in_speed"] is True


def test_zero_layer_thickness_vanishes():
    thin = BrashIceState(layer_thickness_m=1.0e-6)
    ref = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
    res = brash_resistance(thin, standard_hull(25.0), 3.0)
    assert res.total_N / ref.total_N < 1.0e-5


# ---------------------------------------------------------------------------
# calibration and provenance
# ---------------------------------------------------------------------------

def test_family_calibration_reproduces_the_ledger_constant():
    record = calibrate_family()
    assert record["matches_ledger"], record
    assert record["confinement_coefficient"] == pytest.approx(
        check_coefficient("confinement_coefficient")[0], abs=1e-6)
    assert record["max_abs_relative_error"] < B_TOLERANCE


def test_single_point_calibration_hits_its_target():
    from tensordem.brash_ice_rheology.anchors import reference_anchor
    hull = standard_hull(25.0)
    target = float(reference_anchor(STANDARD_BRASH, hull, 3.0)["total_N"])
    record = calibrate_confinement(STANDARD_BRASH, hull, 3.0, target)
    assert abs(record["relative_error"]) < 1e-6


def test_ledger_is_graded_and_hashed():
    grades = {grade for _v, grade, _n in COEFFICIENT_LEDGER.values()}
    assert {"LITERATURE", "DERIVED", "CONVENTION", "CALIBRATED"} <= grades
    assert any(grade == "WITHHELD" for _v, grade, _n in COEFFICIENT_LEDGER.values())
    digest = ledger_sha256()
    assert len(digest) == 64 and digest == ledger_sha256()
    # the digest must move when a coefficient moves
    from tensordem.brash_ice_rheology import coefficients as coeff
    original = coeff.COEFFICIENT_LEDGER["keel_drag_coefficient"]
    try:
        coeff.COEFFICIENT_LEDGER["keel_drag_coefficient"] = (
            original[0] + 0.1, original[1], original[2])
        assert ledger_sha256() != digest
    finally:
        coeff.COEFFICIENT_LEDGER["keel_drag_coefficient"] = original
    assert ledger_sha256() == digest


def test_results_carry_the_ledger_digest():
    res = brash_resistance(STANDARD_BRASH, standard_hull(25.0), 3.0)
    assert res.ledger_sha256 == ledger_sha256()


# ---------------------------------------------------------------------------
# opt-in DEM measurement (the calibration device itself, ~25 s per point)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("BRASH_DEM") != "1",
                    reason="set BRASH_DEM=1 to run the DEM servo-shear point")
def test_dem_servo_shear_returns_a_physical_mu_point():
    from tensordem.brash_ice_rheology.mu_i import run_shear_sweep
    records = run_shear_sweep((0.05,), contact_friction=0.36)
    point = records[0]
    assert 0.0 < point["inertial_number_measured"] < 5.0
    assert 0.0 < point["mu_measured"] < 2.0
    assert point["normal_stress_pa"] > 0.0
