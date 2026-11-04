"""Verification of the Hertz contact stability bound.

Tests that the recommended_dt for Hertz contact provides adequate stability margin
across a range of overlaps, and identifies the instability threshold.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem import stability


@pytest.fixture(autouse=True)
def _single_thread():
    torch.set_num_threads(1)


def _hertz_secant_stiffness(overlap, young_modulus=1e9, poisson_ratio=0.3, radius=0.025):
    """Secant stiffness of Hertz contact at given overlap."""
    e_star = young_modulus / (2.0 * (1.0 - poisson_ratio ** 2))
    r_star = radius / 2.0
    force = (4.0 / 3.0) * e_star * math.sqrt(r_star) * overlap ** 1.5
    return force / overlap if overlap > 0 else 0.0


def _particle_mass(radius=0.025, density=917.0):
    return density * (4.0 / 3.0) * math.pi * radius ** 3


class TestHertzStabilityMargin:
    """Stability margin of the Hertz contact at the recommended time step."""

    @pytest.mark.parametrize("overlap_frac,expected_rho_max", [
        (0.01, 0.32),
        (0.05, 0.45),
        (0.10, 0.58),
        (0.20, 0.82),
    ])
    def test_rho_below_one_at_design_overlaps(self, overlap_frac, expected_rho_max):
        """Amplification factor rho must be < 1 at the recommended step for overlaps <= 0.2 R."""
        radius = 0.025
        mass = _particle_mass(radius)
        overlap = overlap_frac * radius
        k_secant = _hertz_secant_stiffness(overlap)
        omega = math.sqrt(k_secant / mass)

        cfg = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=radius, density=917.0,
            contact_model="hertz", young_modulus=1e9, poisson_ratio=0.3,
            bond_stiffness=0.0, contact_stiffness=0.0, drag=0.0,
            fix_x_edges=False, tool_speed=1e-3, tool_gap=1.0,
        )
        dt_rec = cfg.recommended_dt
        rho = stability.symplectic_euler_growth_factor(dt_rec, omega, 0.0)
        assert rho < 1.0, f"rho={rho} at overlap_frac={overlap_frac}"
        assert rho == pytest.approx(expected_rho_max, abs=0.05)

    def test_instability_threshold_near_03r(self):
        """The instability threshold (rho > 1) should be near 0.3 R."""
        radius = 0.025
        mass = _particle_mass(radius)

        cfg = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=radius, density=917.0,
            contact_model="hertz", young_modulus=1e9, poisson_ratio=0.3,
            bond_stiffness=0.0, contact_stiffness=0.0, drag=0.0,
            fix_x_edges=False, tool_speed=1e-3, tool_gap=1.0,
        )
        dt_rec = cfg.recommended_dt

        # At 0.3 R, rho should be close to 1.0
        overlap = 0.30 * radius
        k_secant = _hertz_secant_stiffness(overlap)
        omega = math.sqrt(k_secant / mass)
        rho = stability.symplectic_euler_growth_factor(dt_rec, omega, 0.0)
        assert rho == pytest.approx(1.0, abs=0.05), f"rho={rho} at 0.3R"

        # At 0.4 R, rho should exceed 1.0
        overlap = 0.40 * radius
        k_secant = _hertz_secant_stiffness(overlap)
        omega = math.sqrt(k_secant / mass)
        rho = stability.symplectic_euler_growth_factor(dt_rec, omega, 0.0)
        assert rho > 1.0, f"rho={rho} at 0.4R should be unstable"

    def test_design_point_margin(self):
        """At the design point (0.1 R), the safety margin should be > 50%."""
        radius = 0.025
        mass = _particle_mass(radius)
        overlap = 0.10 * radius
        k_secant = _hertz_secant_stiffness(overlap)
        omega = math.sqrt(k_secant / mass)

        cfg = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=radius, density=917.0,
            contact_model="hertz", young_modulus=1e9, poisson_ratio=0.3,
            bond_stiffness=0.0, contact_stiffness=0.0, drag=0.0,
            fix_x_edges=False, tool_speed=1e-3, tool_gap=1.0,
        )
        dt_rec = cfg.recommended_dt
        rho = stability.symplectic_euler_growth_factor(dt_rec, omega, 0.0)

        # Safety margin = 1/rho - 1
        margin = 1.0 / rho - 1.0
        assert margin > 0.5, f"margin={margin} at design point, expected > 0.5"
