"""Verification of friction heat, Blok partition and the Jaeger moving source.

The moving band source is computed twice by independent means -- by quadrature
of the Bessel-function Green's function and by a finite-volume solve of the same
boundary value problem -- and twice more against analytic limits (the
transit-time asymptote at high Peclet, the logarithmic divergence at low
Peclet).  The friction flux carries the mandatory negative control: no slip, no
heat.
"""
from __future__ import annotations

import math

import pytest

from tensordem.interface_thermal.conduction import thermal_diffusivity
from tensordem.interface_thermal.friction_heat import (
    blok_partition_coefficient,
    friction_heat_flux,
    friction_power,
    jaeger_band_max_temperature,
    jaeger_band_surface_temperature,
    jaeger_high_speed_limit,
    jaeger_low_speed_limit,
    moving_band_fvm_solution,
)

ALPHA = thermal_diffusivity(2.2, 917.0, 2100.0)
HALF_WIDTH = 5.0e-3
FLUX = 1.0e6


def test_friction_heat_flux_zero_without_slip():
    """Negative control: a contact that does not slide generates no heat."""
    assert friction_heat_flux(0.1, 5.0e6, 0.0) == 0.0
    assert friction_heat_flux(0.0, 5.0e6, 2.0) == 0.0
    assert friction_power(0.1, 500.0, 0.0) == 0.0


def test_friction_heat_flux_is_mu_p_v():
    flux = friction_heat_flux(0.08, 5.0e6, 0.5)
    assert flux == pytest.approx(0.08 * 5.0e6 * 0.5, rel=1e-12)
    assert friction_power(0.08, 500.0, 0.5) == pytest.approx(flux * 1.0e-4, rel=1e-9)


@pytest.mark.parametrize("bad", [(0.1, 5.0e6, -1.0), (-0.1, 5.0e6, 1.0)])
def test_friction_heat_flux_rejects_negative_inputs(bad):
    with pytest.raises(ValueError):
        friction_heat_flux(*bad)


def test_blok_partition_sums_to_one_and_is_symmetric():
    ice_steel = blok_partition_coefficient(2.2, 917.0, 2100.0, 45.0, 7850.0, 470.0)
    assert ice_steel["partition_1"] + ice_steel["partition_2"] == pytest.approx(1.0, abs=1e-15)
    assert ice_steel["partition_2"] > ice_steel["partition_1"]   # steel is stiffer thermally
    same = blok_partition_coefficient(2.2, 917.0, 2100.0, 2.2, 917.0, 2100.0)
    assert same["partition_1"] == pytest.approx(0.5, abs=1e-15)


def test_jaeger_quadrature_matches_finite_volume_solution():
    """Two independent discretisations of Rosenthal's equation (<2 %)."""
    for velocity in (0.005, 0.01, 0.02):
        quadrature = jaeger_band_max_temperature(HALF_WIDTH, velocity, ALPHA, 2.2, FLUX,
                                                 samples=81)["max_temperature_k"]
        fvm = moving_band_fvm_solution(HALF_WIDTH, velocity, ALPHA, 2.2, FLUX,
                                       cells_x=1001, cells_z=100, grading_z=1.05)
        error = abs(fvm["max_temperature_k"] - quadrature) / quadrature
        assert error < 0.02, f"v={velocity}: {error:.3%}"


def test_jaeger_quadrature_matches_high_speed_transit_asymptote():
    """``T_max -> (2q/k) sqrt(alpha (2a/v) / pi)`` for V >> 1 (<2 %)."""
    for velocity in (0.02, 0.05):
        quadrature = jaeger_band_max_temperature(HALF_WIDTH, velocity, ALPHA, 2.2, FLUX,
                                                 samples=81)["max_temperature_k"]
        limit = jaeger_high_speed_limit(HALF_WIDTH, velocity, ALPHA, 2.2, FLUX)
        assert abs(quadrature - limit) / limit < 0.02


def test_jaeger_quadrature_matches_low_speed_log_divergence():
    """``T(0) -> (qa/pi k)[-2 ln(V/2) - 2 gamma + 2]`` for V << 1 (<5 %)."""
    for velocity in (2e-4, 1e-4):
        centre = float(jaeger_band_surface_temperature(
            0.0, HALF_WIDTH, velocity, ALPHA, 2.2, FLUX)[0])
        limit = jaeger_low_speed_limit(HALF_WIDTH, ALPHA, 2.2, FLUX, velocity)
        assert abs(centre - limit) / limit < 0.05


def test_jaeger_peak_sits_at_the_trailing_edge():
    """Peak at the trailing edge, monotone cooling towards the leading edge."""
    peak = jaeger_band_max_temperature(HALF_WIDTH, 0.02, ALPHA, 2.2, FLUX, samples=81)
    profile = peak["profile_t"]
    index = int(profile.argmax())
    # the corner itself is marginally cooler than the point just inside it, so
    # the peak sits within the trailing quarter of the band
    assert peak["position_m"] < -0.5 * HALF_WIDTH
    assert index <= len(profile) // 4
    assert all(profile[i] >= profile[i + 1] - 1e-9
               for i in range(index, len(profile) - 1))
    assert profile[0] > profile[-1]        # hot wake behind, cold upstream


def test_jaeger_temperature_falls_with_sliding_speed():
    """Faster sliding spreads the same flux over a longer track -> cooler."""
    values = [jaeger_band_max_temperature(HALF_WIDTH, v, ALPHA, 2.2, FLUX,
                                          samples=41)["max_temperature_k"]
              for v in (0.005, 0.01, 0.02, 0.04)]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))


def test_jaeger_heat_is_linear_in_flux():
    single = jaeger_band_max_temperature(HALF_WIDTH, 0.01, ALPHA, 2.2, FLUX,
                                         samples=41)["max_temperature_k"]
    double = jaeger_band_max_temperature(HALF_WIDTH, 0.01, ALPHA, 2.2, 2 * FLUX,
                                         samples=41)["max_temperature_k"]
    assert double == pytest.approx(2 * single, rel=1e-12)


@pytest.mark.parametrize("kwargs", [
    dict(half_width_m=0.0), dict(velocity_m_s=-1.0), dict(alpha_m2_s=0.0),
])
def test_jaeger_rejects_unphysical_inputs(kwargs):
    settings = dict(half_width_m=HALF_WIDTH, velocity_m_s=0.01, alpha_m2_s=ALPHA,
                    k_w_m_k=2.2, flux_w_m2=FLUX)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        jaeger_band_surface_temperature(0.0, **settings)