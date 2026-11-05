"""Reference verification of the ridge-grounding bearing-capacity chain.

Reference values are Prandtl (1920) slip-line theory for ``phi = 0`` and the
Vesic (1973) / Skempton (1951) tables as printed in standard geotechnical
references (values quoted to two decimals below).  These tests verify the *code*
against those references rather than re-deriving the implementation, so they can
actually fail when the model is wrong; it is not a physical validation.

They also lock in the corrected undrained branch: the Skempton embedment factor
must be applied exactly once and the cap must hold.  The two verification
campaigns that grew in parallel (the Skempton/depth-factor audit and the
reference-ledger campaign) are both kept here; the colliding helper names are
disambiguated instead of shadowed.
"""
from __future__ import annotations

import math

import pytest

from tensordem.ridge_geotech import (
    SKEMPTON_NC_STRIP_MAX,
    MohrCoulombSoil,
    SofteningVerticalSpring,
    bearing_capacity_factors,
    skempton_nc,
    skempton_undrained_nc,
)
from tensordem.ridge_geotech_verification import (
    PRANDTL_NC,
    SKEMPTON_SQUARE_TABLE,
    SKEMPTON_STRIP_TABLE,
    VESIC_FACTOR_TABLE,
    undrained_capacity_impact,
    verify_subgrade_capacity_model,
)

# Vesic (1973) bearing-capacity factors, as tabulated (phi in degrees).
VESIC_TABLE = {
    0.0: (5.14, 1.00, 0.00),
    10.0: (8.35, 2.47, 1.22),
    20.0: (14.83, 6.40, 5.39),
    25.0: (20.72, 10.66, 10.88),
    30.0: (30.14, 18.40, 22.40),
    35.0: (46.12, 33.30, 48.03),
    40.0: (75.31, 64.20, 109.41),
}

# Skempton (1951) undrained strip-footing embedment factor, quoted to two
# decimals (the module-level SKEMPTON_STRIP_TABLE from
# ``ridge_geotech_verification`` carries the same points as ``(z/B, N_c)`` rows).
SKEMPTON_STRIP_TABLE_2DP = {
    0.0: 5.14,
    0.5: 5.65,
    1.0: 6.17,
    2.0: 7.20,
    2.5: 7.50,
    4.0: 7.50,
}


def _undrained(cohesion: float = 30.0e3, **kwargs) -> MohrCoulombSoil:
    """phi = 0 total-stress soil; ``cohesion`` is the undrained shear strength."""
    defaults = dict(friction_angle_deg=0.0, cohesion=cohesion,
                    saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                    water_unit_weight=10.06e3, undrained=True)
    defaults.update(kwargs)
    return MohrCoulombSoil(**defaults)


# --------------------------------------------------------------- Vesic factors
@pytest.mark.parametrize("phi_deg", sorted(VESIC_TABLE))
def test_vesic_factors_reproduce_published_table(phi_deg: float) -> None:
    n_c, n_q, n_gamma = bearing_capacity_factors(math.radians(phi_deg))
    ref_c, ref_q, ref_gamma = VESIC_TABLE[phi_deg]
    # table entries are rounded to two decimals; compare on that scale
    assert n_c == pytest.approx(ref_c, abs=0.01 + 1e-3 * ref_c)
    assert n_q == pytest.approx(ref_q, abs=0.01 + 1e-3 * ref_q)
    assert n_gamma == pytest.approx(ref_gamma, abs=0.01 + 1e-3 * ref_gamma)


def test_vesic_factors_match_published_table():
    for phi_deg, (nc_ref, nq_ref, ng_ref) in VESIC_FACTOR_TABLE.items():
        nc, nq, ng = bearing_capacity_factors(math.radians(phi_deg))
        assert nq == pytest.approx(nq_ref, rel=5.0e-3), phi_deg
        assert nc == pytest.approx(nc_ref, rel=5.0e-3), phi_deg
        assert ng == pytest.approx(ng_ref, rel=5.0e-3), phi_deg


# ----------------------------------------------------------------- Skempton N_c
def test_undrained_surface_factor_matches_prandtl_slip_line() -> None:
    """``N_c = 2 + pi`` is the exact Prandtl solution for a weightless strip."""
    prandtl = 2.0 + math.pi
    n_c = skempton_undrained_nc(0.0, 0.914)
    assert n_c == pytest.approx(5.14, rel=0.0, abs=0.0)
    assert abs(n_c - prandtl) / prandtl < 5.0e-4  # 3-s.f. truncation, 0.03 %


@pytest.mark.parametrize("ratio", sorted(SKEMPTON_STRIP_TABLE_2DP))
def test_skempton_strip_embedment_matches_table(ratio: float) -> None:
    width = 0.914
    n_c = skempton_undrained_nc(ratio * width, width)
    assert n_c == pytest.approx(SKEMPTON_STRIP_TABLE_2DP[ratio], abs=0.01)


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


def test_skempton_cap_is_strip_limit_and_scales_with_shape() -> None:
    width = 0.914
    assert SKEMPTON_NC_STRIP_MAX == pytest.approx(7.5)
    assert skempton_undrained_nc(100.0 * width, width) == pytest.approx(7.5)
    # square/circular footing: (1 + 0.2 B/L) = 1.2 on value and on the cap
    assert skempton_undrained_nc(0.0, width, length_ratio=1.0) == pytest.approx(5.14 * 1.2)
    assert skempton_undrained_nc(100.0 * width, width, length_ratio=1.0) \
        == pytest.approx(7.5 * 1.2)


# ------------------------------------------------------- undrained embedment bug
def test_embedment_enters_exactly_once() -> None:
    """The former code applied Skempton's 0.2 z/B *and* a Meyerhof depth factor."""
    soil = _undrained(30.0e3)
    width = 0.914
    for ratio in (0.25, 1.0, 1.5, 2.0):
        pressure = soil.vertical_pressure_capacity(width, ratio * width)
        skempton = 30.0e3 * 5.14 * (1.0 + 0.2 * ratio)
        assert pressure == pytest.approx(skempton, rel=1e-12)
        double_counted = skempton * (1.0 + 0.2 * min(ratio, 1.0))
        assert pressure < double_counted * (1.0 - 1e-6)


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


def test_undrained_pressure_capacity_is_linear_in_shear_strength() -> None:
    width = 0.914
    a = _undrained(10.0e3).vertical_pressure_capacity(width, 1.5 * width)
    b = _undrained(30.0e3).vertical_pressure_capacity(width, 1.5 * width)
    assert b / a == pytest.approx(3.0, rel=1e-12)


def test_undrained_pressure_slope_matches_skempton_gradient() -> None:
    """Below the cap, ``dq/dz = 0.2 s_u 5.14 / B`` exactly."""
    soil = _undrained(20.0e3)
    width = 1.0
    z0, h = 0.3, 1.0e-6
    slope = (soil.vertical_pressure_capacity(width, z0 + h)
             - soil.vertical_pressure_capacity(width, z0 - h)) / (2.0 * h)
    assert slope == pytest.approx(0.2 * 20.0e3 * 5.14 / width, rel=1e-7)


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


# ------------------------------------------------- drained branch / unverified
def test_drained_branch_unchanged_by_undrained_fix() -> None:
    """Hand computation of the Vesic/Meyerhof sum for the default soil."""
    soil = MohrCoulombSoil(friction_angle_deg=30.0, cohesion=5.0e3,
                           saturated_unit_weight=18.0e3, water_unit_weight=10.06e3)
    width, z = 0.914, 1.5
    n_c, n_q, n_gamma = bearing_capacity_factors(math.radians(30.0))
    k = min(z / width, 1.0)
    phi = math.radians(30.0)
    d_c = 1.0 + 0.4 * k
    d_q = 1.0 + 2.0 * math.tan(phi) * (1.0 - math.sin(phi)) ** 2 * k
    sigma_v = soil.effective_vertical_stress(z)
    gamma_prime = soil.effective_unit_weight
    expected = (5.0e3 * n_c * d_c + sigma_v * n_q * d_q
                + 0.5 * gamma_prime * width * n_gamma)
    assert soil.vertical_pressure_capacity(width, z) == pytest.approx(expected, rel=1e-12)


def test_lateral_limit_pressure_has_no_independent_anchor() -> None:
    """Documented WITHHELD: ``N_p = 2 + 1.5 z/D`` is pipe-soil practice only.

    The check only pins the implemented formula and its cap, so a future change
    cannot silently alter it; it is *not* a literature verification.
    """
    soil = _undrained(20.0e3)
    depth, diameter = 3.0, 0.914
    z = depth + 0.5 * diameter
    expected = min(2.0 + 1.5 * z / diameter, 9.0) * 20.0e3 * diameter
    assert soil.lateral_limit_pressure(depth, diameter) == pytest.approx(expected, rel=1e-12)