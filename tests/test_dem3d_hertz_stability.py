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
    """Margin of ``recommended_dt`` against the marginal step of the Hertz mode.

    ``stability.symplectic_euler_growth_factor`` is the exact spectral radius of
    semi-implicit Euler on one SDOF mode: it is marginal (``rho == 1``) for any
    undamped step with ``omega * dt < 2`` and only exceeds 1 beyond that
    threshold.  So the question a stability bound can answer is *how far* the
    recommended step sits below ``omega * dt = 2``, not "how much below 1" --
    the latter is unanswerable for a symplectic integrator.
    """

    def _config(self, radius=0.025):
        return DEM3DConfig(
            nx=2, ny=2, nz=2, radius=radius, density=917.0,
            contact_model="hertz", young_modulus=1e9, poisson_ratio=0.3,
            drag=0.0,
            fix_x_edges=False, tool_speed=1e-3, tool_gap=1.0,
        )

    def _omega(self, overlap, radius=0.025):
        k_secant = _hertz_secant_stiffness(overlap, radius=radius)
        return math.sqrt(k_secant / _particle_mass(radius))

    @pytest.mark.parametrize("overlap_frac", [0.01, 0.05, 0.10, 0.15, 0.20])
    def test_rho_is_marginal_below_the_threshold(self, overlap_frac):
        """At every design overlap the recommended step stays in the marginal band."""
        radius = 0.025
        dt_rec = self._config(radius).recommended_dt
        omega = self._omega(overlap_frac * radius, radius)
        rho = stability.symplectic_euler_growth_factor(dt_rec, omega, 0.0)
        assert omega * dt_rec < 2.0, f"omega*dt={omega * dt_rec} at overlap_frac={overlap_frac}"
        assert rho == pytest.approx(1.0, abs=1e-12), f"rho={rho} at overlap_frac={overlap_frac}"

    def test_design_point_margin(self):
        """At the design overlap (0.1 R) the step is an order of magnitude below the limit."""
        radius = 0.025
        dt_rec = self._config(radius).recommended_dt
        omega = self._omega(0.10 * radius, radius)
        ratio = omega * dt_rec
        # At least a factor of ~7 of head room below the marginal omega*dt = 2.
        assert 0.0 < ratio < 0.3, f"design-point margin omega*dt={ratio}"

    def test_instability_is_detected_beyond_the_threshold(self):
        """The growth factor does flag the marginal threshold, so it is not vacuous.

        The bound only sees the secant stiffness at the design overlap, so the
        threshold is reached at an overlap of ``(2 / 0.15) ** 4`` design fractions:
        far outside any physically admissible contact, which is exactly why the
        recommended step is safe for Hertz contacts.
        """
        radius = 0.025
        dt_rec = self._config(radius).recommended_dt
        ratio = self._omega(0.10 * radius, radius) * dt_rec
        # omega ~ k_secant^(1/2) ~ overlap^(1/4), so the threshold sits at
        # overlap/radius = 0.10 * (2 / ratio) ** 4 design fractions.
        crossover = 0.10 * (2.0 / ratio) ** 4
        omega_in = self._omega(1.05 * crossover * radius, radius)
        omega_out = self._omega(0.95 * crossover * radius, radius)
        assert stability.symplectic_euler_growth_factor(dt_rec, omega_out, 0.0) == pytest.approx(1.0)
        assert stability.symplectic_euler_growth_factor(dt_rec, omega_in, 0.0) > 1.0


