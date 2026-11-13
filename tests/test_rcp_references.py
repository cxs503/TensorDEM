"""Tests for friction-dependent RCP reference values.

Validates that the reference table is self-consistent, that interpolation
behaves correctly, and that the friction-dependent references produce
physically sensible validation outcomes for the existing deposition results.
"""

import math

import numpy as np
import pytest

from tensordem.rcp_references import (
    RCPReference,
    all_friction_values,
    frictionless_reference,
    get_reference,
    reference_table,
)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_reference_table_is_nonempty():
    tbl = reference_table()
    assert len(tbl) >= 5, "Should cover frictionless through high-friction"


def test_all_friction_values_are_sorted():
    vals = all_friction_values()
    assert vals == sorted(vals)
    assert 0.0 in vals
    assert 1.0 in vals


def test_frictionless_reference_is_canonical():
    ref = frictionless_reference()
    assert ref.mu == 0.0
    assert abs(ref.phi_rcp - 0.6366) < 1e-4
    assert "Scott" in ref.source or "Bernal" in ref.source


# ---------------------------------------------------------------------------
# RCPReference properties
# ---------------------------------------------------------------------------

def test_band_is_symmetric():
    ref = RCPReference(mu=0.0, phi_rcp=0.64, phi_uncertainty=0.005, source="test")
    lo, hi = ref.band
    assert abs((lo + hi) / 2 - 0.64) < 1e-12
    assert abs(hi - lo - 0.02) < 1e-12


def test_in_band():
    ref = RCPReference(mu=0.0, phi_rcp=0.64, phi_uncertainty=0.005, source="test")
    assert ref.in_band(0.64)
    assert ref.in_band(0.635)
    assert ref.in_band(0.645)
    assert not ref.in_band(0.62)
    assert not ref.in_band(0.66)


def test_relative_error():
    ref = RCPReference(mu=0.0, phi_rcp=0.64, phi_uncertainty=0.005, source="test")
    assert abs(ref.relative_error(0.64)) < 1e-12
    assert abs(ref.relative_error(0.6464) - 0.01) < 1e-6
    assert abs(ref.relative_error(0.6336) + 0.01) < 1e-6


# ---------------------------------------------------------------------------
# get_reference: nearest
# ---------------------------------------------------------------------------

def test_nearest_exact_match():
    ref = get_reference(0.0, method="nearest")
    assert ref.mu == 0.0


def test_nearest_closest():
    ref = get_reference(0.05, method="nearest")
    assert ref.mu == 0.0  # 0.0 is closer than 0.1


def test_nearest_midpoint_goes_lower():
    ref = get_reference(0.05, method="nearest")
    # 0.05 is equidistant from 0.0 and 0.1; min() picks the first
    assert ref.mu == 0.0


def test_nearest_high_friction():
    ref = get_reference(0.4, method="nearest")
    assert ref.mu == 0.5  # 0.5 is slightly closer due to float: |0.5-0.4| < |0.3-0.4|


def test_nearest_very_high():
    ref = get_reference(0.8, method="nearest")
    assert ref.mu == 1.0  # 1.0 is closer than 0.5


# ---------------------------------------------------------------------------
# get_reference: interpolate
# ---------------------------------------------------------------------------

def test_interpolate_at_table_point():
    ref = get_reference(0.0, method="interpolate")
    assert ref.mu == 0.0
    assert abs(ref.phi_rcp - 0.6366) < 1e-4


def test_interpolate_midpoint():
    ref = get_reference(0.05, method="interpolate")
    # Between mu=0.0 (phi=0.6366) and mu=0.1 (phi=0.6250)
    expected_phi = 0.6366 + 0.5 * (0.6250 - 0.6366)
    assert abs(ref.phi_rcp - expected_phi) < 1e-6
    assert ref.mu == 0.05


def test_interpolate_monotone_decrease():
    """RCP should decrease monotonically with friction."""
    mus = np.linspace(0.0, 1.0, 21)
    phis = [get_reference(mu, method="interpolate").phi_rcp for mu in mus]
    for i in range(1, len(phis)):
        assert phis[i] <= phis[i - 1] + 1e-10, \
            f"phi increased at mu={mus[i]}: {phis[i-1]} -> {phis[i]}"


def test_interpolate_bounds():
    """Below/above table range should return the boundary reference."""
    ref_lo = get_reference(-0.1, method="interpolate")
    assert ref_lo.mu == 0.0
    ref_hi = get_reference(2.0, method="interpolate")
    assert ref_hi.mu == 1.0


def test_interpolate_uncertainty_increases_with_mu():
    """Uncertainty should generally increase with friction (fewer data points)."""
    ref_0 = get_reference(0.0, method="interpolate")
    ref_1 = get_reference(1.0, method="interpolate")
    assert ref_1.phi_uncertainty >= ref_0.phi_uncertainty


# ---------------------------------------------------------------------------
# Physical sanity: friction-dependent validation
# ---------------------------------------------------------------------------

def test_frictionless_result_passes_frictionless_ref():
    """The existing mu=0 result (phi=0.64237) should pass the frictionless ref."""
    ref = get_reference(0.0, method="nearest")
    phi_measured = 0.64237
    assert ref.in_band(phi_measured), \
        f"mu=0 result {phi_measured} should be in frictionless band {ref.band}"


def test_low_friction_result_passes_friction_dependent_ref():
    """The existing mu=0.1 result (phi=0.62628) should pass the mu=0.1 ref."""
    ref = get_reference(0.1, method="nearest")
    phi_measured = 0.62628
    assert ref.in_band(phi_measured), \
        f"mu=0.1 result {phi_measured} should be in mu=0.1 band {ref.band}"


def test_low_friction_result_fails_frictionless_ref():
    """The mu=0.1 result should be *outside* the frictionless 2σ band.

    This is the key physical insight: using a single frictionless reference
    for all friction tiers is incorrect.  The mu=0.1 result (0.62628) is
    physically correct for its friction level but appears to 'fail' when
    compared against the frictionless reference.
    """
    ref = get_reference(0.0, method="nearest")
    phi_measured = 0.62628
    # The mu=0.1 result is below the frictionless 2σ band
    assert not ref.in_band(phi_measured), \
        f"mu=0.1 result {phi_measured} should NOT be in frictionless band {ref.band}"


def test_hard_sphere_limit_mu0_passes():
    """The mu=0 hard-sphere limit (0.63644) should pass the frictionless ref."""
    ref = get_reference(0.0, method="nearest")
    phi_inf = 0.63644
    assert ref.in_band(phi_inf)
    assert abs(ref.relative_error(phi_inf)) < 0.03


def test_hard_sphere_limit_mu01_passes_friction_dependent_ref():
    """The mu=0.1 hard-sphere limit (0.61705) should pass the mu=0.1 ref.

    This is the critical fix: the original G13 gate failed because it compared
    the mu=0.1 hard-sphere limit against the frictionless reference.  With the
    friction-dependent reference, this should pass.
    """
    ref = get_reference(0.1, method="nearest")
    phi_inf = 0.61705
    # The mu=0.1 ref has phi_rcp=0.6250, uncertainty=0.008
    # 2σ band: [0.609, 0.641]
    assert ref.in_band(phi_inf), \
        f"mu=0.1 hard-sphere limit {phi_inf} should be in mu=0.1 band {ref.band}"


def test_moderate_friction_trend():
    """Results at mu=0.3 and mu=0.5 should be consistent with their refs."""
    # From the existing runs: mu=0.3 -> phi=0.60454, mu=0.5 -> phi=0.59310
    for mu, phi in [(0.3, 0.60454), (0.5, 0.59310)]:
        ref = get_reference(mu, method="nearest")
        assert ref.in_band(phi), \
            f"mu={mu} result {phi} should be in band {ref.band}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        get_reference(0.0, method="bogus")


def test_reference_table_has_required_keys():
    tbl = reference_table()
    required = {"mu", "phi_rcp", "phi_uncertainty", "band_2sigma", "source"}
    for row in tbl:
        assert required <= set(row.keys())
