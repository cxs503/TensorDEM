"""Reference verification tests for the subgrade capacity model.

These tests compare ``tensordem.ridge_geotech`` against closed-form and published
values (Prandtl 1920, Skempton 1951, Vesic 1973, Meyerhof/Hansen) rather than
re-deriving the implementation, so they can actually fail when the model is
wrong.  They also lock in the corrected undrained branch: the Skempton embedment
factor must be applied exactly once and the cap must hold.
"""
from __future__ import annotations

import math

import pytest

from tensordem.ridge_geotech import (MohrCoulombSoil, SofteningVerticalSpring,
                                     bearing_capacity_factors, skempton_nc)
from tensordem.ridge_geotech_verification import (
    PRANDTL_NC, SKEMPTON_SQUARE_TABLE, SKEMPTON_STRIP_TABLE, VESIC_FACTOR_TABLE,
    undrained_capacity_impact, verify_subgrade_capacity_model)


def _undrained(cohesion=30.0e3, **kwargs):
    defaults = dict(friction_angle_deg=0.0, cohesion=cohesion,
                    saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                    water_unit_weight=10.06e3, undrained=True)
    defaults.update(kwargs)
    return MohrCoulombSoil(**defaults)


# ----------------------------------------------------------------- Skempton N_c
def test_skempton_strip_table_against_published_values():
    for z_over_b, reference in SKEMPTON_STRIP_TABLE:
        assert skempton_nc(z_over_b) == pytest.approx(reference, rel=5.0e-3), z_over_b


def test_skempton_square_table_against_published_values():
    for z_over_b, reference in SKEMPTON_SQUARE_TABLE:
        assert skempton_nc(z_over_b, footing="square") == pytest.approx(
            reference, rel=5.0e-3), z_over_b


def test_skempton_reduces_to_prandtl_solution_at_the_surface():
    # phi = 0 surface strip capacity is the slip-line result (2 + pi) s_u.
    assert skempton_nc(0.0) == pytest.approx(PRANDTL_NC, rel=5.0e-3)
    assert PRANDTL_NC == pytest.approx(5.1416, abs=1.0e-4)


def test_skempton_caps_and_rejects_bad_input():
    assert skempton_nc(10.0) == 7.5
    assert skempton_nc(2.5) == pytest.approx(7.5)
    assert skempton_nc(10.0, footing="circular") == 9.0
    with pytest.raises(ValueError):
        skempton_nc(-0.1)
    with pytest.raises(ValueError):
        skempton_nc(1.0, footing="annular")


# ------------------------------------------------------- undrained embedment bug
def test_undrained_embedment_factor_is_applied_once():
    """Regression guard: the Skempton factor used to be applied twice."""
    soil = _undrained()
    diameter = 0.914
    for z_over_b in (0.0, 0.5, 1.0, 2.0, 3.0, 6.0):
        pressure = soil.vertical_pressure_capacity(diameter, z_over_b * diameter)
        assert pressure == pytest.approx(30.0e3 * skempton_nc(z_over_b), rel=1e-12)


def test_undrained_capacity_never_exceeds_the_skempton_cap():
    soil = _undrained()
    diameter = 0.914
    for embedment in (0.0, 0.5, 2.0, 10.0, 40.0):
        capacity = soil.vertical_capacity_per_length(diameter, embedment)
        assert capacity <= 7.5 * 30.0e3 * diameter * (1.0 + 1e-12)


def test_undrained_capacity_is_linear_in_shear_strength():
    diameter, embedment = 0.914, 0.5
    values = [_undrained(s).vertical_capacity_per_length(diameter, embedment) / s
              for s in (10.0e3, 30.0e3, 60.0e3)]
    assert max(values) - min(values) < 1e-9


# --------------------------------------------------------------- Vesic factors
def test_vesic_factors_match_published_table():
    for phi_deg, (nc_ref, nq_ref, ng_ref) in VESIC_FACTOR_TABLE.items():
        nc, nq, ng = bearing_capacity_factors(math.radians(phi_deg))
        assert nq == pytest.approx(nq_ref, rel=5.0e-3), phi_deg
        assert nc == pytest.approx(nc_ref, rel=5.0e-3), phi_deg
        assert ng == pytest.approx(ng_ref, rel=5.0e-3), phi_deg


# ------------------------------------------------------- spring / impact wiring
def test_softening_spring_brackets_the_verified_capacity():
    soil = _undrained()
    spring = SofteningVerticalSpring(soil, 0.914, 0.5)
    assert spring.peak == pytest.approx(0.9 * spring.ultimate)
    assert spring.residual == pytest.approx(0.8 * spring.peak)


def test_impact_reports_the_legacy_over_prediction():
    impact = undrained_capacity_impact(diameter=0.914, embedment=0.5, cohesion=30.0e3)
    # legacy double-counted factor at z/B = 1.547 is 1.2 -> exactly -1/6 change
    assert impact["relative_change"] == pytest.approx(-1.0 / 6.0, rel=1e-9)
    assert impact["corrected_capacity_per_length"] == pytest.approx(184.5e3, rel=1e-3)


# ----------------------------------------------------------- whole-model ledger
def test_verification_ledger_passes_every_reference_check():
    result = verify_subgrade_capacity_model()
    failures = [row["id"] for row in result["checks"] if not row["passed"]]
    assert not failures, failures
    summary = result["summary"]
    assert summary["all_passed"] is True
    assert summary["n_checks"] == summary["n_passed"]
    assert summary["max_relative_error"] < 0.01
    assert summary["all_within_3_percent"] is True
    # the lateral-limit-pressure reference text is unavailable -> declared WITHHELD
    assert summary["evidence_roll_up"]["WITHHELD"] == 1


def test_defect_quantification_matches_the_old_implementation():
    defect = verify_subgrade_capacity_model()["defect"]
    rows = {row["z_over_b"]: row for row in defect["rows"]}
    assert rows[1.0]["over_prediction"] == pytest.approx(0.2, rel=1e-9)
    assert rows[6.0]["legacy_nc"] == pytest.approx(10.8, rel=1e-9)
    assert defect["max_over_prediction"] > 0.4