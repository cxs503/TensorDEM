"""Verification of the ridge-grounding bearing-capacity chain against closed forms.

Reference values are Prandtl (1920) slip-line theory for ``phi = 0`` and the
Vesic (1973) / Skempton (1951) tables as printed in standard geotechnical
references (values quoted to two decimals below).  This module verifies the
*code* against those references; it is not a physical validation.
"""
from __future__ import annotations

import math

import pytest

from tensordem.ridge_geotech import (
    SKEMPTON_NC_STRIP_MAX,
    MohrCoulombSoil,
    bearing_capacity_factors,
    skempton_undrained_nc,
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

# Skempton (1951) undrained strip-footing embedment factor.
SKEMPTON_STRIP_TABLE = {
    0.0: 5.14,
    0.5: 5.65,
    1.0: 6.17,
    2.0: 7.20,
    2.5: 7.50,
    4.0: 7.50,
}


def _undrained(su: float = 30.0e3) -> MohrCoulombSoil:
    return MohrCoulombSoil(friction_angle_deg=0.0, cohesion=su,
                           water_unit_weight=10.06e3, undrained=True)


@pytest.mark.parametrize("phi_deg", sorted(VESIC_TABLE))
def test_vesic_factors_reproduce_published_table(phi_deg: float) -> None:
    n_c, n_q, n_gamma = bearing_capacity_factors(math.radians(phi_deg))
    ref_c, ref_q, ref_gamma = VESIC_TABLE[phi_deg]
    # table entries are rounded to two decimals; compare on that scale
    assert n_c == pytest.approx(ref_c, abs=0.01 + 1e-3 * ref_c)
    assert n_q == pytest.approx(ref_q, abs=0.01 + 1e-3 * ref_q)
    assert n_gamma == pytest.approx(ref_gamma, abs=0.01 + 1e-3 * ref_gamma)


def test_undrained_surface_factor_matches_prandtl_slip_line() -> None:
    """``N_c = 2 + pi`` is the exact Prandtl solution for a weightless strip."""
    prandtl = 2.0 + math.pi
    n_c = skempton_undrained_nc(0.0, 0.914)
    assert n_c == pytest.approx(5.14, rel=0.0, abs=0.0)
    assert abs(n_c - prandtl) / prandtl < 5.0e-4  # 3-s.f. truncation, 0.03 %


@pytest.mark.parametrize("ratio", sorted(SKEMPTON_STRIP_TABLE))
def test_skempton_strip_embedment_matches_table(ratio: float) -> None:
    width = 0.914
    n_c = skempton_undrained_nc(ratio * width, width)
    assert n_c == pytest.approx(SKEMPTON_STRIP_TABLE[ratio], abs=0.01)


def test_skempton_cap_is_strip_limit_and_scales_with_shape() -> None:
    width = 0.914
    assert SKEMPTON_NC_STRIP_MAX == pytest.approx(7.5)
    assert skempton_undrained_nc(100.0 * width, width) == pytest.approx(7.5)
    # square/circular footing: (1 + 0.2 B/L) = 1.2 on value and on the cap
    assert skempton_undrained_nc(0.0, width, length_ratio=1.0) == pytest.approx(5.14 * 1.2)
    assert skempton_undrained_nc(100.0 * width, width, length_ratio=1.0) \
        == pytest.approx(7.5 * 1.2)


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


def test_undrained_pressure_slope_matches_skempton_gradient() -> None:
    """Below the cap, ``dq/dz = 0.2 s_u 5.14 / B`` exactly."""
    soil = _undrained(20.0e3)
    width = 1.0
    z0, h = 0.3, 1.0e-6
    slope = (soil.vertical_pressure_capacity(width, z0 + h)
             - soil.vertical_pressure_capacity(width, z0 - h)) / (2.0 * h)
    assert slope == pytest.approx(0.2 * 20.0e3 * 5.14 / width, rel=1e-7)


def test_undrained_capacity_is_linear_in_shear_strength() -> None:
    width = 0.914
    a = _undrained(10.0e3).vertical_pressure_capacity(width, 1.5 * width)
    b = _undrained(30.0e3).vertical_pressure_capacity(width, 1.5 * width)
    assert b / a == pytest.approx(3.0, rel=1e-12)


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