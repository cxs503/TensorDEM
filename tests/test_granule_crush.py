"""Verification tests for the single-granule crushing surface (P97-N02).

Covers: Weibull strength model (closed forms, simulation vs analytic
median, size-law fit), Gaudin-Schuhmann spectrum (CDF/PDF consistency,
sampling, alpha recovery, mass-weighted exponent), sub-granule
replacement conservation (mass/momentum, 1000-event ledger), negative
controls (zero load -> zero crushing), and the full acceptance runner.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensordem.granule_crush import (  # noqa: E402
    CrushLedger,
    FragmentSet,
    GaudinSpectrum,
    GranuleCrushConfig,
    WeibullFitResult,
    WeibullStrength,
    crush_single_granule,
    fit_weibull_size_law,
    replace_granule,
    run_acceptance_suite,
    run_conservation_ledger,
    run_gaudin_spectrum_benchmark,
    run_negative_control,
    run_weibull_size_study,
    sphere_volume,
)

QUARTZ = {
    "density": 2650.0,
    "youngs_modulus": 37.0e9,
    "poisson_ratio": 0.17,
    "dt": 1.0e-6,
    "t_max": 200.0e-3,
}


def _quartz_strength() -> WeibullStrength:
    return WeibullStrength(sigma_0=100.0e6, m=7.0, V_0=sphere_volume(1.0e-3))


def _quartz_cfg(radius: float = 1.0e-3) -> GranuleCrushConfig:
    return GranuleCrushConfig(
        radius=radius,
        density=QUARTZ["density"],
        youngs_modulus=QUARTZ["youngs_modulus"],
        poisson_ratio=QUARTZ["poisson_ratio"],
        platen_velocity=0.01,
        dt=QUARTZ["dt"],
        t_max=QUARTZ["t_max"],
    )


# ---------------------------------------------------------------------------
# Weibull strength model
# ---------------------------------------------------------------------------
class TestWeibullStrength:
    def test_failure_probability_bounds_and_monotonicity(self):
        s = _quartz_strength()
        V = sphere_volume(2.0e-3)
        assert s.failure_probability(0.0, V) == 0.0
        p1 = s.failure_probability(50.0e6, V)
        p2 = s.failure_probability(100.0e6, V)
        assert 0.0 < p1 < p2 < 1.0
        # at (sigma_0, V_0) the exponent is exactly 1
        assert s.failure_probability(s.sigma_0, s.V_0) == pytest.approx(1.0 - math.exp(-1.0))
        # larger volume -> larger probability at the same stress
        assert s.failure_probability(50.0e6, sphere_volume(4.0e-3)) > p1

    def test_size_effect_closed_form(self):
        s = _quartz_strength()
        for r in (0.5e-3, 1.0e-3, 2.0e-3, 8.0e-3):
            V = sphere_volume(r)
            expected = s.sigma_0 * (V / s.V_0) ** (-1.0 / s.m)
            assert s.failure_stress(V) == pytest.approx(expected, rel=1e-12)
        # doubling the radius lowers the strength by 2^(-3/m)
        ratio = s.failure_stress(sphere_volume(2.0e-3)) / s.failure_stress(sphere_volume(1.0e-3))
        assert ratio == pytest.approx(2.0 ** (-3.0 / s.m), rel=1e-12)

    def test_median_and_mean_closed_forms(self):
        s = _quartz_strength()
        V = sphere_volume(2.0e-3)
        med = s.sigma_0 * (s.V_0 / V) ** (1.0 / s.m) * math.log(2.0) ** (1.0 / s.m)
        assert s.median_failure_stress(V) == pytest.approx(med, rel=1e-12)
        from scipy.special import gamma

        mean = s.sigma_0 * (s.V_0 / V) ** (1.0 / s.m) * float(gamma(1.0 + 1.0 / s.m))
        assert s.mean_failure_stress(V) == pytest.approx(mean, rel=1e-12)
        # for m = 7 the Weibull shape puts the mean just below the median
        # (Gamma(1+1/7) = 0.9388 < (ln 2)^(1/7) = 0.9490)
        assert s.mean_failure_stress(V) < s.median_failure_stress(V)

    def test_invalid_parameters_rejected(self):
        with pytest.raises(ValueError):
            WeibullStrength(sigma_0=-1.0, m=7.0, V_0=1.0e-9)
        with pytest.raises(ValueError):
            WeibullStrength(sigma_0=100.0e6, m=0.0, V_0=1.0e-9)
        with pytest.raises(ValueError):
            WeibullStrength(sigma_0=100.0e6, m=7.0, V_0=0.0)
        s = _quartz_strength()
        with pytest.raises(ValueError):
            s.failure_probability(-1.0, 1.0e-9)
        with pytest.raises(ValueError):
            s.failure_stress(0.0)


# ---------------------------------------------------------------------------
# Crushing simulation
# ---------------------------------------------------------------------------
class TestCrushSimulation:
    def test_simulation_median_matches_analytic(self):
        s = _quartz_strength()
        cfg = _quartz_cfg()
        rng = np.random.default_rng(123)
        res = crush_single_granule(cfg, s, rng, n_trials=2000)
        assert res.n_failed == 2000
        analytic = s.median_failure_stress(cfg.volume)
        assert res.median_failure_stress == pytest.approx(analytic, rel=0.02)

    def test_failure_stress_monotone_in_size(self):
        s = _quartz_strength()
        rng = np.random.default_rng(7)
        medians = []
        for d in (0.5e-3, 2.0e-3, 8.0e-3):
            res = crush_single_granule(_quartz_cfg(0.5 * d), s, rng, n_trials=400)
            medians.append(res.median_failure_stress)
        assert medians[0] > medians[1] > medians[2]

    def test_weibull_size_study_recovers_m(self):
        s = _quartz_strength()
        out = run_weibull_size_study(
            s, QUARTZ, [0.5e-3, 1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3], n_trials=300, seed=99
        )
        assert out["verdict"] == "PASS"
        assert out["r2"] >= 0.95
        assert out["m_rel_dev"] <= 0.20
        assert out["m_fit"] == pytest.approx(s.m, rel=0.20)

    def test_fit_rejects_bad_input(self):
        with pytest.raises(ValueError):
            fit_weibull_size_law([1.0, 2.0], [100.0, 50.0])
        with pytest.raises(ValueError):
            fit_weibull_size_law([1.0, 2.0, 4.0], [100.0, 0.0, 50.0])
        with pytest.raises(ValueError):
            fit_weibull_size_law([1.0, 2.0, 4.0], [100.0, 50.0])


# ---------------------------------------------------------------------------
# Gaudin-Schuhmann spectrum
# ---------------------------------------------------------------------------
class TestGaudinSpectrum:
    def test_cdf_pdf_consistency(self):
        spec = GaudinSpectrum(alpha=1.5, d_max=10.0e-3)
        assert spec.cdf(0.0) == 0.0
        assert spec.cdf(spec.d_max) == 1.0
        d = np.linspace(1e-6, spec.d_max, 20001)
        integral = np.trapezoid(spec.pdf(d), d)
        assert integral == pytest.approx(1.0, rel=1e-4)
        assert np.all(spec.pdf(d) >= 0.0)
        with pytest.raises(ValueError):
            GaudinSpectrum(alpha=0.0, d_max=1.0e-3)
        with pytest.raises(ValueError):
            GaudinSpectrum(alpha=1.5, d_max=-1.0e-3)

    def test_sampling_matches_cdf(self):
        spec = GaudinSpectrum(alpha=1.5, d_max=10.0e-3)
        rng = np.random.default_rng(5)
        d = spec.sample(50000, rng)
        assert d.min() > 0.0 and d.max() <= spec.d_max
        grid = np.linspace(0.02, 1.0, 50)
        F_emp = np.array([(d <= x * spec.d_max).mean() for x in grid])
        assert np.abs(F_emp - grid ** 1.5).max() < 0.02

    def test_fit_recovers_alpha(self):
        spec = GaudinSpectrum(alpha=1.5, d_max=10.0e-3)
        rng = np.random.default_rng(6)
        d = spec.sample(20000, rng)
        for alpha_hat in (spec.fit_alpha_mle(d), spec.fit_alpha_cdf(d)):
            assert abs(alpha_hat - 1.5) / 1.5 <= 0.15

    def test_mass_weighted_exponent_is_alpha_plus_3(self):
        spec = GaudinSpectrum(alpha=1.5, d_max=10.0e-3)
        rng = np.random.default_rng(8)
        d = spec.sample(200000, rng)
        mass = d ** 3
        order = np.argsort(d)
        d_s, m_s = d[order], mass[order]
        F_m = np.cumsum(m_s) / m_s.sum()
        # log-log slope of the mass CDF must be alpha + 3
        mask = F_m > 1e-3
        slope = np.polyfit(np.log(d_s[mask]), np.log(F_m[mask]), 1)[0]
        assert slope == pytest.approx(1.5 + 3.0, abs=0.15)

    def test_gaudin_benchmark_passes(self):
        out = run_gaudin_spectrum_benchmark(
            alpha_true=1.5, d_max=10.0e-3, n_samples=20000, n_events=500, seed=42
        )
        assert out["verdict"] == "PASS"
        for v in out["rel_dev"].values():
            assert v <= 0.15


# ---------------------------------------------------------------------------
# Sub-granule replacement + conservation ledger
# ---------------------------------------------------------------------------
class TestConservation:
    def test_single_event_mass_and_momentum(self):
        spec = GaudinSpectrum(alpha=1.5, d_max=8.0e-3)
        rng = np.random.default_rng(11)
        r_p, m_p = 3.0e-3, 2650.0 * sphere_volume(3.0e-3)
        v_p = np.array([0.1, -0.05, 0.02])
        fr = replace_granule(r_p, m_p, v_p, spec, 9, rng)
        assert fr.masses.sum() == pytest.approx(m_p, rel=1e-12)
        assert np.allclose(fr.masses @ fr.velocities, m_p * v_p, rtol=1e-12, atol=1e-18)

    def test_1000_event_ledger_mass(self):
        out = run_conservation_ledger(
            n_events=1000,
            spectrum=GaudinSpectrum(alpha=1.5, d_max=8.0e-3),
            seed=20261108,
        )
        assert out["verdict"] == "PASS"
        assert out["n_events"] == 1000
        assert out["max_mass_residual_rel"] <= 1.0e-3

    def test_1000_event_ledger_momentum(self):
        out = run_conservation_ledger(
            n_events=1000,
            spectrum=GaudinSpectrum(alpha=1.5, d_max=8.0e-3),
            seed=20261108,
        )
        assert out["max_momentum_residual_rel"] <= 1.0e-3

    def test_replace_granule_rejects_bad_input(self):
        spec = GaudinSpectrum(alpha=1.5, d_max=8.0e-3)
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError):
            replace_granule(-1.0e-3, 1.0, [0.0, 0.0, 0.0], spec, 4, rng)
        with pytest.raises(ValueError):
            replace_granule(1.0e-3, 1.0, [0.0, 0.0, 0.0], spec, 1, rng)

    def test_ledger_serialization_roundtrip(self, tmp_path):
        spec = GaudinSpectrum(alpha=1.5, d_max=8.0e-3)
        rng = np.random.default_rng(2)
        ledger = CrushLedger()
        for i in range(5):
            r_p = 2.0e-3
            m_p = 2650.0 * sphere_volume(r_p)
            v_p = rng.normal(0.0, 0.1, size=3)
            fr = replace_granule(r_p, m_p, v_p, spec, 6, rng)
            ledger.record(i, m_p, v_p, fr)
        ledger.to_json(tmp_path / "ledger.json")
        ledger.to_csv(tmp_path / "ledger.csv")
        saved = json.loads((tmp_path / "ledger.json").read_text())
        assert len(saved) == 5
        assert saved[0]["event_id"] == 0
        assert saved[0]["mass_residual_rel"] <= 1.0e-3
        csv_lines = (tmp_path / "ledger.csv").read_text().strip().splitlines()
        assert len(csv_lines) == 6
        assert csv_lines[0].startswith("event_id,")


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------
class TestNegativeControls:
    def test_zero_load_zero_crushing(self):
        out = run_negative_control(
            _quartz_strength(), QUARTZ, radius=1.0e-3, t_run=50.0e-3, n_trials=50, seed=3
        )
        assert out["verdict"] == "PASS"
        assert out["n_crushed"] == 0

    def test_infinite_strength_zero_crushing(self):
        s = WeibullStrength(sigma_0=1.0e12, m=7.0, V_0=sphere_volume(1.0e-3))
        cfg = _quartz_cfg()
        rng = np.random.default_rng(4)
        res = crush_single_granule(cfg, s, rng, n_trials=100)
        assert res.n_failed == 0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
class TestAcceptance:
    def test_full_acceptance_suite_passes(self, tmp_path):
        rep = run_acceptance_suite(tmp_path / "out")
        assert rep["verdict"] == "PASS"
        for key in ("C1", "C2C3", "C4", "C4_rate", "C5"):
            assert rep["criteria"][key]["verdict"] == "PASS", key
        assert (tmp_path / "out" / "acceptance.json").is_file()
        assert (tmp_path / "out" / "conservation_ledger.csv").is_file()
        assert (tmp_path / "out" / "weibull_size_law.csv").is_file()
        assert (tmp_path / "out" / "gaudin_samples.npz").is_file()
