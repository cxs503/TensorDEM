"""Regression guards for the fragmentation statistics layer.

Two defects were found and fixed in ``tensordem.threefield.fragmentation``:

1. ``fragment()`` rebuilt every fragment mass with a hard-coded
   ``rho_i = 917.0`` instead of ``props.ice_density_kg_m3``.  With the default
   density nothing showed up; at 900 kg/m^3 the coupled mass ledger drifted by
   1.9 % of the whole inventory, and at 800 kg/m^3 by 14.6 %.
2. ``kolmogorov_smirnov()`` returned the Weibull plotting positions
   ``(i+1)/(n+1)`` evaluated on one side of the empirical CDF, which is
   systematically below the true statistic (23 % low at n = 10, 4.5 % at
   n = 200), so any threshold gate built on it was under-powered.

These tests are deliberately cheap (no Monte-Carlo sweeps); the full evidence
lives in ``benchmarks/fragmentation_statistics.py`` and
``docs/fragmentation_statistics_verification.md``.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from tensordem.threefield.coupling import fragmentation_stage
from tensordem.threefield.fragmentation import (
    BreakupProperties,
    bending_stress_pa,
    cdf_agreement_r2,
    critical_floe_length,
    fit_weibull,
    floe_volume_m3,
    fragment,
    kolmogorov_smirnov,
    weibull_cdf,
)

from fragmentation_statistics import (
    _beam_lumped_load_moment,
    _beam_moment_fd,
    coupling_ledger_consistency,
    mass_ledger,
    support_condition_sensitivity,
)


def _exact_ks(samples: list[float], shape: float, scale: float) -> float:
    values = sorted(samples)
    n = len(values)
    deviation = 0.0
    for rank, value in enumerate(values):
        reference = weibull_cdf(value, shape, scale)
        deviation = max(deviation, (rank + 1) / n - reference, reference - rank / n)
    return deviation


def _weibull_sample(rng: np.random.Generator, n: int, shape: float, scale: float) -> list[float]:
    return (scale * np.power(-np.log1p(-rng.random(n)), 1.0 / shape)).tolist()


class MassLedgerDensityTests(unittest.TestCase):
    def test_fragment_uses_property_density(self) -> None:
        thickness = 0.2
        sizes = [3.0, 5.0]
        for density in (800.0, 900.0, 950.0, 917.0):
            props = BreakupProperties(thickness_m=thickness, ice_density_kg_m3=density)
            result = fragment(sizes, 5.0, props)
            expected = sum(density * floe_volume_m3(size, props) for size in sizes)
            self.assertAlmostEqual(result["initial_mass_kg"] / expected, 1.0, places=13)
            self.assertLess(abs(result["final_mass_kg"] - expected) / expected, 1e-13)
            rebuilt = sum(density * floe_volume_m3(length, props) for length in result["lengths_m"])
            self.assertLess(abs(rebuilt - result["final_mass_kg"]) / expected, 1e-13)

    def test_legacy_hardcode_would_have_drifted(self) -> None:
        # the size of the defect the fix removes, at a realistic cold-ice density
        props = BreakupProperties(thickness_m=0.2, ice_density_kg_m3=900.0)
        result = fragment([3.0, 5.0], 5.0, props)
        legacy = 917.0 * sum(floe_volume_m3(size, props) for size in (3.0, 5.0))
        drift = abs(legacy - result["initial_mass_kg"]) / result["initial_mass_kg"]
        self.assertAlmostEqual(drift, 917.0 / 900.0 - 1.0, places=12)
        self.assertGreater(drift, 0.018)

    def test_coupling_stage_ledger_closes_for_non_default_density(self) -> None:
        report = coupling_ledger_consistency()
        self.assertTrue(report["verdict_pass"], report["rows"])
        self.assertTrue(all(row["split_events"] > 0 for row in report["rows"]))

    def test_benchmark_mass_ledger_table(self) -> None:
        report = mass_ledger()
        self.assertTrue(report["verdict_pass"], report["rows"])
        self.assertLess(report["worst_initial_mass_rel_error"], 1e-13)


class KolmogorovSmirnovTests(unittest.TestCase):
    def test_matches_exact_two_sided_statistic(self) -> None:
        rng = np.random.default_rng(5)
        for n in (8, 25, 120):
            samples = _weibull_sample(rng, n, 1.8, 1.0)
            self.assertAlmostEqual(
                kolmogorov_smirnov(samples, 1.8, 1.0), _exact_ks(samples, 1.8, 1.0), places=12
            )

    def test_matches_scipy_statistic(self) -> None:
        from scipy import stats

        rng = np.random.default_rng(6)
        samples = _weibull_sample(rng, 60, 1.8, 1.0)
        reference = stats.kstest(samples, lambda v: 1.0 - np.exp(-np.power(v / 1.0, 1.8))).statistic
        self.assertAlmostEqual(kolmogorov_smirnov(samples, 1.8, 1.0), float(reference), places=12)

    def test_never_below_plotting_position_proxy(self) -> None:
        rng = np.random.default_rng(7)
        for n in (12, 40, 200):
            samples = _weibull_sample(rng, n, 1.8, 1.0)
            values = sorted(samples)
            ranks = np.arange(1, n + 1)
            reference = 1.0 - np.exp(-np.power(np.asarray(values) / 1.0, 1.8))
            proxy = float(np.abs(ranks / (n + 1.0) - reference).max())
            self.assertGreaterEqual(kolmogorov_smirnov(samples, 1.8, 1.0), proxy - 1e-12)

    def test_rejects_shifted_law_at_five_percent(self) -> None:
        rng = np.random.default_rng(8)
        samples = _weibull_sample(rng, 400, 2.4, 1.0)
        self.assertGreater(kolmogorov_smirnov(samples, 1.8, 1.0), 1.358 / math.sqrt(400))

    def test_empty_sample_rejected(self) -> None:
        with self.assertRaises(ValueError):
            kolmogorov_smirnov([0.0, -1.0], 1.8, 1.0)


class BeamCoefficientTests(unittest.TestCase):
    def test_finite_difference_recovers_simply_supported_moment(self) -> None:
        for cells in (10, 40, 160):
            self.assertAlmostEqual(_beam_moment_fd(cells), 0.125, places=8)

    def test_sampled_statics_is_exact_for_symmetric_meshes(self) -> None:
        for loads in (2, 8, 32):
            self.assertAlmostEqual(_beam_lumped_load_moment(loads), 0.125, places=14)

    def test_module_stress_coefficient_is_three_quarters(self) -> None:
        props = BreakupProperties(thickness_m=0.2, flexural_strength_pa=5.0e5)
        q = 0.5 * props.water_density_kg_m3 * props.drag_coefficient * 0.75**2 * props.thickness_m
        coefficient = bending_stress_pa(4.0, 0.75, props) * props.thickness_m**2 / (q * 16.0)
        self.assertAlmostEqual(coefficient, 0.75, places=14)

    def test_support_condition_sensitivity(self) -> None:
        report = support_condition_sensitivity()
        self.assertTrue(report["module_matches_simply_supported"])
        self.assertAlmostEqual(report["spread_factor"], math.sqrt(6.0), places=4)


class CriticalLengthTests(unittest.TestCase):
    def test_bisection_and_exponents(self) -> None:
        props = BreakupProperties(thickness_m=0.1, flexural_strength_pa=0.5e6)
        length = critical_floe_length(0.75, props)
        low, high = 1.0, 200.0
        for _ in range(200):
            mid = 0.5 * (low + high)
            if bending_stress_pa(mid, 0.75, props) < props.flexural_strength_pa:
                low = mid
            else:
                high = mid
        self.assertLess(abs(0.5 * (low + high) - length) / length, 1e-10)
        self.assertAlmostEqual(critical_floe_length(1.5, props) / length, 0.5, places=12)
        self.assertEqual(critical_floe_length(0.0, props), math.inf)

    def test_cascade_postconditions(self) -> None:
        rng = np.random.default_rng(11)
        population = list(np.exp(rng.uniform(math.log(0.5), math.log(40.0), 80)))
        props = BreakupProperties(thickness_m=0.1, ice_density_kg_m3=900.0)
        result = fragment(population, 0.75, props)
        repeat = fragment(population, 0.75, props)
        self.assertGreater(result["split_events"], 0)
        self.assertLess(result["mass_error_kg"] if "mass_error_kg" in result else 0.0, 1e-9)
        self.assertLess(
            abs(result["final_mass_kg"] - result["initial_mass_kg"]) / result["initial_mass_kg"], 1e-13
        )
        self.assertLessEqual(
            max(result["lengths_m"]), max(result["critical_length_m"], 2.0 * props.min_fragment_m) + 1e-12
        )
        self.assertEqual(result["lengths_m"], repeat["lengths_m"])

    def test_reference_law_is_not_commensurate_with_field_census(self) -> None:
        # the coupled gate is quoted with a 0.12 m scale; a metre-scale census
        # cannot agree with it, which is why the gate is reported, not gated on
        sizes = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
        self.assertGreater(cdf_agreement_r2(sizes, 1.8, 0.12)["r_squared"], 0.5)
        field = [1.0, 2.0, 4.0, 7.0, 11.0, 18.0]
        self.assertLess(cdf_agreement_r2(field, 1.8, 0.12)["r_squared"], 0.0)
        self.assertGreater(fit_weibull(field)["sample_count"], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()