"""Analytical verification of the turbulent fragmentation physics core.

Verification strategy
---------------------
Every assertion is graded against a reference that does NOT reuse the code
path under test:

1. **Bending stress formula derivation**: the ``bending_stress_pa`` formula is
   verified against the Euler-Bernoulli beam theory for a free-free beam under
   uniform distributed load.  The peak bending moment occurs at the centre
   (where the beam is supported by buoyancy) and equals M_max = q L² / 8.
   The section modulus is S = h² / 6 (unit width), so σ_max = M_max / S =
   3 q L² / (4 h²).

2. **Critical floe length is the exact inverse of bending stress**: setting
   ``bending_stress_pa(L_crit, ...) == flexural_strength`` and solving for
   ``L_crit`` must reproduce ``critical_floe_length(...)`` to machine precision.

3. **Mass conservation in the fragmentation cascade**: the total mass before
   and after fragmentation must be identical (to floating-point round-off)
   for every combination of input sizes, turbulence intensity, and density.

4. **Weibull fit recovers known parameters**: samples drawn from a Weibull
   distribution with known (shape, scale) must be recovered by ``fit_weibull``
   to within 2 % for shape and 1 % for scale at n = 50 000.

5. **KS statistic correctness**: ``kolmogorov_smirnov`` is checked against an
   independent two-sided computation.

6. **Dimensional consistency**: every output has the correct SI dimensions,
   checked by varying each input by a known factor and verifying the output
   scales correctly.

Scope caveat (must be quoted with any number taken from here):
The beam model treats the floe as a *free-free* beam under uniform distributed
load, supported at the centre by buoyancy.  Real floes in a propeller race
experience non-uniform pressure, wave-induced dynamic loading, and ice-ice
contact.  The verification therefore confirms the *model's own* internal
consistency, not the physical accuracy of the breakup criterion for real ice.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from tensordem.threefield.fragmentation import (
    BreakupProperties,
    bending_stress_pa,
    cdf_agreement_r2,
    critical_floe_length,
    fit_weibull,
    fit_weibull_cdf,
    floe_volume_m3,
    fragment,
    kolmogorov_smirnov,
    weibull_cdf,
)


def _rel(a: float, b: float) -> float:
    """Relative error, safe for nonzero b."""
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------------------
# 1. Bending stress formula derivation (Euler-Bernoulli beam theory)
# ---------------------------------------------------------------------------

class BendingStressBeamTheoryTests(unittest.TestCase):
    """``bending_stress_pa`` vs Euler-Bernoulli beam theory."""

    def test_analytic_formula_directly(self) -> None:
        """sigma = 3 q L^2 / (4 h^2) with q = 0.5 rho C_d u^2 h.
        
        Derivation: free-free beam of length L, thickness h, unit width,
        under uniform distributed load q (from turbulent drag).  The beam
        is supported at the centre by buoyancy.  Maximum bending moment
        occurs at the centre: M_max = q L^2 / 8.  Section modulus for
        rectangular cross-section (unit width): S = h^2 / 6.  Peak bending
        stress: sigma = M_max / S = (q L^2 / 8) / (h^2 / 6) = 3 q L^2 / (4 h^2).
        """
        props = BreakupProperties(
            flexural_strength_pa=0.7e6,
            water_density_kg_m3=1025.0,
            drag_coefficient=1.2,
            thickness_m=0.2,
        )
        L, u = 3.0, 2.0
        q = 0.5 * props.water_density_kg_m3 * props.drag_coefficient * u ** 2 * props.thickness_m
        expected = 3.0 * q * L ** 2 / (4.0 * props.thickness_m ** 2)
        self.assertAlmostEqual(bending_stress_pa(L, u, props), expected, places=10)

    def test_bending_stress_at_various_parameters(self) -> None:
        """Spot-check at multiple parameter combinations."""
        test_cases = [
            (1.0, 0.5, 1.0, 0.1),   # L, u, sigma_f, h
            (2.0, 1.0, 0.5e6, 0.15),
            (5.0, 2.0, 0.7e6, 0.2),
            (0.5, 0.1, 0.3e6, 0.05),
        ]
        for L, u, sigma_f, h in test_cases:
            props = BreakupProperties(
                flexural_strength_pa=sigma_f,
                thickness_m=h,
            )
            q = 0.5 * props.water_density_kg_m3 * props.drag_coefficient * u ** 2 * h
            expected = 3.0 * q * L ** 2 / (4.0 * h ** 2)
            self.assertAlmostEqual(
                bending_stress_pa(L, u, props), expected, places=10,
                msg=f"L={L}, u={u}, h={h}")


# ---------------------------------------------------------------------------
# 2. Critical floe length is the exact inverse of bending stress
# ---------------------------------------------------------------------------

class CriticalLengthInverseTests(unittest.TestCase):
    """``critical_floe_length`` must be the exact inverse of ``bending_stress_pa``."""

    def test_round_trip_at_various_parameters(self) -> None:
        for sigma_f in (0.3e6, 0.5e6, 0.7e6, 1.0e6):
            for h in (0.05, 0.1, 0.2, 0.5):
                for u in (0.1, 0.5, 1.0, 2.0, 5.0):
                    for C_d in (0.8, 1.0, 1.2, 2.0):
                        props = BreakupProperties(
                            flexural_strength_pa=sigma_f,
                            thickness_m=h,
                            drag_coefficient=C_d,
                        )
                        L_crit = critical_floe_length(u, props)
                        # At L_crit the bending stress must equal sigma_f
                        stress_at_Lcrit = bending_stress_pa(L_crit, u, props)
                        self.assertAlmostEqual(
                            stress_at_Lcrit / sigma_f, 1.0, places=12,
                            msg=f"sigma_f={sigma_f}, h={h}, u={u}, C_d={C_d}")

    def test_zero_turbulence_gives_infinity(self) -> None:
        props = BreakupProperties()
        self.assertEqual(critical_floe_length(0.0, props), math.inf)

    def test_scaling_with_turbulence(self) -> None:
        """L_crit ~ 1/u' (from the formula)."""
        props = BreakupProperties()
        L1 = critical_floe_length(1.0, props)
        L2 = critical_floe_length(2.0, props)
        self.assertAlmostEqual(L1 / L2, 2.0, places=12)

    def test_scaling_with_strength(self) -> None:
        """L_crit ~ sqrt(sigma_f)."""
        props1 = BreakupProperties(flexural_strength_pa=0.5e6)
        props2 = BreakupProperties(flexural_strength_pa=2.0e6)
        L1 = critical_floe_length(1.0, props1)
        L2 = critical_floe_length(1.0, props2)
        self.assertAlmostEqual(L2 / L1, 2.0, places=12)


# ---------------------------------------------------------------------------
# 3. Mass conservation in the fragmentation cascade
# ---------------------------------------------------------------------------

class MassConservationTests(unittest.TestCase):
    """Total mass must be conserved by ``fragment()``."""

    def test_mass_conserved_for_various_densities(self) -> None:
        for density in (800.0, 900.0, 917.0, 950.0, 1000.0):
            props = BreakupProperties(ice_density_kg_m3=density, thickness_m=0.15)
            sizes = [1.0, 2.0, 3.5, 5.0, 8.0]
            initial_mass = sum(density * floe_volume_m3(s, props) for s in sizes)
            for u in (0.5, 1.0, 2.0, 5.0):
                result = fragment(sizes, u, props)
                self.assertAlmostEqual(
                    result["final_mass_kg"] / initial_mass, 1.0, places=12,
                    msg=f"density={density}, u={u}")

    def test_mass_conserved_when_no_breakup(self) -> None:
        """When all floes are below L_crit, nothing breaks up."""
        props = BreakupProperties(thickness_m=0.1)
        # Very low turbulence -> very large L_crit -> no breakup
        sizes = [0.5, 1.0, 2.0]
        result = fragment(sizes, 0.001, props)
        initial_mass = sum(props.ice_density_kg_m3 * floe_volume_m3(s, props) for s in sizes)
        self.assertAlmostEqual(result["final_mass_kg"] / initial_mass, 1.0, places=13)

    def test_mass_conserved_under_extreme_breakup(self) -> None:
        """High turbulence breaks everything down to min_fragment."""
        props = BreakupProperties(
            flexural_strength_pa=0.3e6,
            thickness_m=0.05,
            min_fragment_m=0.05,
        )
        sizes = [5.0, 10.0]
        result = fragment(sizes, 10.0, props)
        initial_mass = sum(props.ice_density_kg_m3 * floe_volume_m3(s, props) for s in sizes)
        self.assertAlmostEqual(result["final_mass_kg"] / initial_mass, 1.0, places=12)


# ---------------------------------------------------------------------------
# 4. Weibull fit recovers known parameters
# ---------------------------------------------------------------------------

class WeibullFitRecoveryTests(unittest.TestCase):
    """``fit_weibull`` must recover known (shape, scale) from samples."""

    def test_recovers_known_parameters_large_sample(self) -> None:
        rng = np.random.default_rng(42)
        true_shape, true_scale = 2.5, 1.5
        n = 50_000
        samples = (true_scale * np.power(-np.log1p(-rng.random(n)), 1.0 / true_shape)).tolist()
        result = fit_weibull(samples)
        self.assertLess(_rel(result["shape"], true_shape), 0.02,
                        f"shape: {result['shape']:.4f} vs {true_shape}")
        self.assertLess(_rel(result["scale"], true_scale), 0.01,
                        f"scale: {result['scale']:.4f} vs {true_scale}")

    def test_recovers_exponential_distribution(self) -> None:
        """Weibull with shape=1 is exponential."""
        rng = np.random.default_rng(123)
        true_scale = 2.0
        n = 50_000
        samples = (-true_scale * np.log1p(-rng.random(n))).tolist()
        result = fit_weibull(samples)
        self.assertLess(_rel(result["shape"], 1.0), 0.03)
        self.assertLess(_rel(result["scale"], true_scale), 0.02)

    def test_cdf_agreement_r2_high_for_weibull_data(self) -> None:
        rng = np.random.default_rng(99)
        samples = (1.5 * np.power(-np.log1p(-rng.random(10_000)), 1.0 / 2.0)).tolist()
        result = fit_weibull_cdf(samples)
        self.assertGreater(result["r_squared"], 0.99)


# ---------------------------------------------------------------------------
# 5. KS statistic correctness
# ---------------------------------------------------------------------------

class KSStatisticTests(unittest.TestCase):
    """``kolmogorov_smirnov`` vs independent two-sided computation."""

    def test_matches_independent_computation(self) -> None:
        rng = np.random.default_rng(7)
        samples = (1.0 * np.power(-np.log1p(-rng.random(200)), 1.0 / 1.5)).tolist()
        shape, scale = 1.5, 1.0
        code_ks = kolmogorov_smirnov(samples, shape, scale)

        # Independent computation
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        max_dev = 0.0
        for i, x in enumerate(sorted_samples):
            cdf_val = weibull_cdf(x, shape, scale)
            max_dev = max(max_dev, abs((i + 1) / n - cdf_val), abs(i / n - cdf_val))
        self.assertAlmostEqual(code_ks, max_dev, places=12)

    def test_ks_small_for_exact_match(self) -> None:
        """If samples are exactly the quantiles of the distribution, KS is small."""
        shape, scale = 2.0, 1.0
        n = 100
        # Use midpoints: x_i = F^{-1}((i + 0.5) / n)
        samples = []
        for i in range(n):
            p = (i + 0.5) / n
            x = scale * (-math.log1p(-p)) ** (1.0 / shape)
            samples.append(x)
        ks = kolmogorov_smirnov(samples, shape, scale)
        # The KS should be small (not exactly zero because of plotting positions)
        self.assertLess(ks, 0.05)


# ---------------------------------------------------------------------------
# 6. Dimensional consistency
# ---------------------------------------------------------------------------

class DimensionalConsistencyTests(unittest.TestCase):
    """Verify that outputs scale correctly with inputs."""

    def test_bending_stress_scales_with_L_squared(self) -> None:
        props = BreakupProperties()
        s1 = bending_stress_pa(1.0, 1.0, props)
        s2 = bending_stress_pa(2.0, 1.0, props)
        self.assertAlmostEqual(s2 / s1, 4.0, places=12)

    def test_bending_stress_scales_with_u_squared(self) -> None:
        props = BreakupProperties()
        s1 = bending_stress_pa(1.0, 1.0, props)
        s2 = bending_stress_pa(1.0, 2.0, props)
        self.assertAlmostEqual(s2 / s1, 4.0, places=12)

    def test_bending_stress_scales_inversely_with_h(self) -> None:
        """sigma ~ q/h^2 * L^2, q ~ h => sigma ~ L^2 / h."""
        props = BreakupProperties()
        s1 = bending_stress_pa(1.0, 1.0, props)
        props2 = BreakupProperties(thickness_m=2.0 * props.thickness_m)
        s2 = bending_stress_pa(1.0, 1.0, props2)
        self.assertAlmostEqual(s1 / s2, 2.0, places=12)

    def test_floe_volume_scales_with_L_squared(self) -> None:
        """V = L^2 * h (square floe, unit width) => V ~ L^2."""
        props = BreakupProperties(thickness_m=0.1)
        v1 = floe_volume_m3(1.0, props)
        v2 = floe_volume_m3(2.0, props)
        self.assertAlmostEqual(v2 / v1, 4.0, places=12)


if __name__ == "__main__":
    unittest.main()
