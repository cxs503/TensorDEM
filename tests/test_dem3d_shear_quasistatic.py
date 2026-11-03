"""Tests for the quasi-static convergence audit of the direct-shear specimen.

Two groups:

* closed-form / fit algebra (fast, exact): the wave transit, the viscous
  diffusion time, the quasi-static margin arithmetic, and the two linear
  extrapolators recovered on synthetic data with a known intercept;
* a small end-to-end pair of ramps (fast + slow) on the reference specimen that
  shows the apparent modulus moving *towards* the slow limit and the fit-window
  spread collapsing, i.e. the observable the benchmark relies on actually does
  the job it is claimed to do.
"""
from __future__ import annotations

import math

import pytest
import torch

torch.set_num_threads(1)

from tensordem.dem3d import DEM3DConfig
from tensordem.shear_quasistatic import (
    apparent_modulus_at_rate,
    fit_infinite_width,
    fit_zero_rate_limit,
    lattice_wave_speed,
    quasi_static_config,
    quasi_static_margins,
    shear_diffusion_time,
    shear_modulus_closed_form,
    shear_wave_transit,
)
from tensordem.shear_specimen import direct_shear_specimen_height, lattice_spacing


def test_wave_speed_and_transit_closed_form():
    config = quasi_static_config()
    a = lattice_spacing(config)
    expected_c = a * math.sqrt(config.bond_stiffness / config.mass)
    assert lattice_wave_speed(config) == pytest.approx(expected_c, rel=1e-14)
    # documented value for the shipped parameters
    assert expected_c == pytest.approx(9.1274, rel=1e-4)
    height = direct_shear_specimen_height(config)
    assert shear_wave_transit(config) == pytest.approx(height / expected_c, rel=1e-14)
    assert shear_wave_transit(config) == pytest.approx(0.0219121, rel=1e-5)


def test_diffusion_time_closed_form_and_drag_scaling():
    config = quasi_static_config()
    a = lattice_spacing(config)
    height = direct_shear_specimen_height(config)
    expected = config.drag * height ** 2 / (config.bond_stiffness * a ** 2)
    assert shear_diffusion_time(config) == pytest.approx(expected, rel=1e-14)
    # the previously documented 0.08 s estimate
    assert shear_diffusion_time(config) == pytest.approx(0.08, rel=1e-6)
    # linear in drag, quadratic in height
    assert shear_diffusion_time(config, drag=2.0 * config.drag) == pytest.approx(
        2.0 * expected, rel=1e-14)
    assert shear_diffusion_time(quasi_static_config(nz=9)) == pytest.approx(
        4.0 * expected, rel=1e-14)


def test_quasi_static_margin_criterion():
    config = quasi_static_config()
    slow = quasi_static_margins(config, velocity=0.01, gamma_max=0.025)
    fast = quasi_static_margins(config, velocity=0.4, gamma_max=0.025)
    # ramp time, transits and diffusion ratios shrink exactly with speed
    assert slow["ramp_time_s"] == pytest.approx(0.5, rel=1e-12)
    assert slow["transits"] == pytest.approx(
        fast["transits"] * 0.4 / 0.0125, rel=1e-12)
    assert slow["diffusion_times"] == pytest.approx(
        fast["diffusion_times"] * 0.4 / 0.0125, rel=1e-12)
    assert fast["transits"] < 20.0 and not fast["quasi_static"]
    assert slow["transits"] > 20.0 and slow["diffusion_times"] >= 5.0
    assert slow["quasi_static"]
    with pytest.raises(ValueError):
        quasi_static_margins(config, velocity=0.0, gamma_max=0.025)


def test_zero_rate_extrapolation_recovers_synthetic_intercept():
    # synthetic G(v) = 62000 + 435 v, sampled at the campaign's slowest rates
    def row(v):
        return {"velocity_m_per_s": v,
                "apparent_shear_modulus_Pa": 62000.0 - 435.0 * v,
                "C44_Pa": 93333.333}

    rows = [row(0.4), row(0.2), row(0.1), row(0.05), row(0.025), row(0.0125)]
    fit = fit_zero_rate_limit(rows, points=3)
    assert fit["G0_Pa"] == pytest.approx(62000.0, rel=1e-12)
    assert fit["slope_Pa_per_m_per_s"] == pytest.approx(-435.0, rel=1e-9)
    assert fit["max_abs_residual_Pa"] == pytest.approx(0.0, abs=1e-9)
    assert fit["G0_over_C44"] == pytest.approx(62000.0 / 93333.333, rel=1e-9)
    assert fit["stability_relative_change"] < 1e-12
    with pytest.raises(ValueError):
        fit_zero_rate_limit(rows[:1], points=2)


def test_infinite_width_extrapolation_recovers_synthetic_intercept():
    c44 = 93333.333

    def row(nx, ratio):
        return {"nx": nx,
                "inverse_width_a_over_Lx": 1.0 / (nx * 0.05),
                "apparent_shear_modulus_Pa": ratio * c44,
                "C44_Pa": c44}

    # linear in 1/L_x with the affine constant at 1/L_x = 0
    def synthetic(nx):
        ratio = 1.0 - 0.09 / (nx * 0.05)
        return {"nx": nx, "inverse_width_a_over_Lx": 1.0 / (nx * 0.05),
                "apparent_shear_modulus_Pa": ratio * c44, "C44_Pa": c44,
                "specimen_height_m": 0.2, "lattice_spacing_m": 0.05}

    rows = [synthetic(5), synthetic(9), synthetic(13), synthetic(17)]
    fit = fit_infinite_width(rows)
    assert fit["G_infinite_width_over_C44"] == pytest.approx(1.0, abs=1e-9)
    assert fit["relative_error_vs_C44"] < 1e-9
    assert fit["max_abs_residual_Pa"] < 1e-6
    with pytest.raises(ValueError):
        fit_infinite_width(rows[:1])


def test_reference_protocol_keeps_the_specimen_intact():
    config = quasi_static_config()
    # the axial threshold is parked so the elastic ramp cannot trip it
    assert config.breaking_strain > 1.0e2
    assert config.drag == pytest.approx(10.0)
    # driving the block with the *shipped* axial threshold breaks bonds early
    shipped = quasi_static_config(breaking_strain=DEM3DConfig().breaking_strain)
    # the shipped axial threshold breaks bonds inside the fit window, so the
    # modulus measurement is refused rather than silently fitted through damage
    with pytest.raises(ValueError):
        apparent_modulus_at_rate(shipped, velocity=0.2, gamma_max=0.025)
    intact = apparent_modulus_at_rate(velocity=0.2, gamma_max=0.025)
    assert intact["broken_bonds"] == 0


def test_rate_convergence_direction_and_window_spread():
    """A slow ramp is stiffer than a fast one and has a cleaner fit window."""
    fast = apparent_modulus_at_rate(velocity=0.2, gamma_max=0.025)
    slow = apparent_modulus_at_rate(velocity=0.1, gamma_max=0.025)
    assert fast["broken_bonds"] == 0 and slow["broken_bonds"] == 0
    assert slow["margins"]["transits"] > fast["margins"]["transits"]
    assert slow["apparent_shear_modulus_Pa"] > fast["apparent_shear_modulus_Pa"]
    assert slow["window_spread_relative"] < fast["window_spread_relative"]
    # the slow ramp is still not quasi-static: that is why the benchmark
    # extrapolates instead of quoting a single slow run
    assert not slow["margins"]["quasi_static"]
    assert slow["ratio_to_C44"] < 1.0
    assert shear_modulus_closed_form(quasi_static_config()) == pytest.approx(
        7.0 * 2000.0 / (3.0 * 0.05), rel=1e-12)