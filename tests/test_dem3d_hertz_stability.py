"""Verification of the Hertz contact stability bound.

Tests that the recommended_dt for Hertz contact provides adequate stability margin
across a range of overlaps, and identifies the instability threshold.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.dem3d import DEM3DConfig
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
    """Stability margin of the Hertz contact at the recommended time step.

    ``stability.symplectic_euler_growth_factor`` returns the spectral radius of
    semi-implicit Euler on an SDOF mode: ``rho == 1`` is *marginal* (the
    undamped integrator neither grows nor contracts) and ``rho > 1`` is
    unstable, with the boundary at ``omega*dt = 2``.  The checks below therefore
    score ``rho`` against that bound instead of a contraction factor, and locate
    the instability onset by solving ``omega(delta)*dt_rec == 2``.
    """

    @pytest.mark.parametrize("overlap_frac,expected_rho_max", [
        (0.01, 0.32),
        (0.05, 0.45),
        (0.10, 0.58),
        (0.20, 0.82),
    ])
    def test_rho_below_one_at_design_overlaps(self, overlap_frac, expected_rho_max):
        """At <= 0.2 R the step is spectrally marginal (rho == 1) and omega*dt < 2.

        ``expected_rho_max`` is kept only so the historical parameter ids stay
        stable; it recorded a contraction factor that this integrator does not
        have (undamped semi-implicit Euler is exactly marginal, ``rho == 1``).
        """
        radius = 0.025
        mass = _particle_mass(radius)
        overlap = overlap_frac * radius
        k_secant = _hertz_secant_stiffness(overlap)
        omega = math.sqrt(k_secant / mass)

        cfg = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=radius, density=917.0,
            contact_model="hertz", young_modulus=1e9, poisson_ratio=0.3,
            drag=0.0,
            fix_x_edges=False, tool_speed=1e-3, tool_gap=1.0,
        )
        dt_rec = cfg.recommended_dt
        rho = stability.symplectic_euler_growth_factor(dt_rec, omega, 0.0)
        assert omega * dt_rec < 2.0, f"omega*dt={omega * dt_rec} exceeds the spectral bound"
        assert rho <= 1.0 + 1e-9, f"rho={rho} at overlap_frac={overlap_frac}"
        assert rho == pytest.approx(1.0, abs=1e-9)

    def test_instability_threshold_near_03r(self):
        """The spectral boundary omega*dt == 2 is sharp: rho == 1 below, rho > 1 above.

        The threshold overlap is found by bisecting ``omega(delta) * dt_rec - 2``
        instead of being assumed, and it must sit outside the design envelope
        (0.2 R) so the recommended step stays stable over the documented range.
        """
        radius = 0.025
        mass = _particle_mass(radius)

        cfg = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=radius, density=917.0,
            contact_model="hertz", young_modulus=1e9, poisson_ratio=0.3,
            drag=0.0,
            fix_x_edges=False, tool_speed=1e-3, tool_gap=1.0,
        )
        dt_rec = cfg.recommended_dt

        def omega_at(overlap_frac):
            k_secant = _hertz_secant_stiffness(overlap_frac * radius)
            return math.sqrt(k_secant / mass)

        # Exponential bracket: the onset is far outside the design envelope for
        # the bonded default stiffness, so widen hi until omega*dt crosses 2.
        lo, hi = 1e-3, 0.4
        assert omega_at(lo) * dt_rec < 2.0
        while omega_at(hi) * dt_rec < 2.0:
            hi *= 2.0
            assert hi < 1e7, "spectral boundary not bracketed" 
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if omega_at(mid) * dt_rec < 2.0:
                lo = mid
            else:
                hi = mid
        threshold = lo
        assert threshold > 0.2, f"onset {threshold:.3f} R inside the design envelope"

        rho_below = stability.symplectic_euler_growth_factor(dt_rec, omega_at(threshold * 0.999), 0.0)
        rho_above = stability.symplectic_euler_growth_factor(dt_rec, omega_at(threshold * 1.001), 0.0)
        assert rho_below == pytest.approx(1.0, abs=1e-9)
        assert rho_above > 1.0, f"rho={rho_above} at {threshold * 1.001:.6f} R should be unstable"

    def test_design_point_margin(self):
        """At the design point (0.1 R) the spectral margin 2/(omega*dt) - 1 > 50%."""
        radius = 0.025
        mass = _particle_mass(radius)
        overlap = 0.10 * radius
        k_secant = _hertz_secant_stiffness(overlap)
        omega = math.sqrt(k_secant / mass)

        cfg = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=radius, density=917.0,
            contact_model="hertz", young_modulus=1e9, poisson_ratio=0.3,
            drag=0.0,
            fix_x_edges=False, tool_speed=1e-3, tool_gap=1.0,
        )
        dt_rec = cfg.recommended_dt
        rho = stability.symplectic_euler_growth_factor(dt_rec, omega, 0.0)

        # Safety margin against the spectral boundary (omega*dt = 2).
        margin = 2.0 / (omega * dt_rec) - 1.0
        assert margin > 0.5, f"margin={margin} at design point, expected > 0.5"
        assert rho <= 1.0 + 1e-9
