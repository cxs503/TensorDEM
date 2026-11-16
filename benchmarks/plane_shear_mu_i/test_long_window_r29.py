#!/usr/bin/env python3
"""Tests for the round-29 long-window stationarity analysis (m22733).

The analysis is what decides whether CASE-DEM-PLANE-SHEAR-MUIDI's judged ``mu``
is a steady state, so its statistics are pinned on analytic values rather than
on "it runs":

* ``integrated_autocorr_time``  - white noise (``tau = 1``) and AR(1)
  (``tau = (1+phi)/(1-phi)``);
* ``block_means``               - exact block assignment and counts;
* ``weighted_trend``            - recovers a known slope, does not fire on a
  constant series;
* ``saturation_fit``            - recovers ``mu_inf`` of a synthetic exponential
  relaxation;
* ``verdict_point``             - the 10 % acceptance band is applied in the
  right direction;
* ``case_for``                  - the campaign reproduces the archived round-26
  case bit for bit except for ``strain_measure``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analyse_long_window_r29 as an  # noqa: E402
import plane_shear_2d as ps  # noqa: E402
import run_long_window_r29 as camp  # noqa: E402


def test_autocorr_white_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    st = an.integrated_autocorr_time(x)
    assert 0.9 <= st["tau"] <= 1.6
    assert abs(st["n_eff"] - st["n"] / st["tau"]) < 1e-9
    # sem must be sd/sqrt(n_eff): for white noise that is ~sd/sqrt(n)
    assert 0.8 <= st["sem"] / (x.std(ddof=0) / math.sqrt(x.size)) <= 1.3


def test_autocorr_ar1():
    rng = np.random.default_rng(1)
    phi = 0.8
    n = 200000
    noise = rng.normal(size=n)
    x = np.empty(n)
    x[0] = noise[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    st = an.integrated_autocorr_time(x)
    exact = (1.0 + phi) / (1.0 - phi)          # = 9
    assert 0.6 * exact <= st["tau"] <= 1.6 * exact
    assert st["sem"] > x.std(ddof=0) / math.sqrt(n)   # correlated -> larger error


def test_autocorr_degenerate_inputs():
    assert math.isnan(an.integrated_autocorr_time(np.zeros(3))["tau"])
    st = an.integrated_autocorr_time(np.ones(100))
    assert st["tau"] == 1.0 and st["sem"] == 0.0


def test_block_means_assignment():
    rows = [{"strain": 0.2 + 0.05 * i, "mu_wall": float(i)} for i in range(40)]
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    blocks = an.block_means(rows, "mu_wall", edges)
    assert [b["n"] for b in blocks] == [16, 20, 4]
    # first block holds strains 0.2 .. 0.95 -> values 0..15
    assert blocks[0]["mean"] == pytest.approx(np.mean(np.arange(0, 16)))
    assert blocks[2]["mean"] == pytest.approx(np.mean(np.arange(36, 40)))


def test_weighted_trend_recovers_slope():
    blocks = [{"lo": i, "hi": i + 1, "n": 100, "mean": 0.10 + 0.01 * i, "sem": 1e-4}
              for i in range(8)]
    tr = an.weighted_trend(blocks)
    assert tr["slope"] == pytest.approx(0.01, rel=1e-6)
    assert abs(tr["z"]) > 10.0
    flat = [dict(b, mean=0.2) for b in blocks]
    assert abs(an.weighted_trend(flat)["z"]) < 1e-6


def test_saturation_fit_recovers_asymptote():
    strain = np.linspace(0.01, 8.0, 400)
    mu_inf, a, gamma0 = 0.21, -0.06, 1.5
    mu = mu_inf + a * np.exp(-strain / gamma0)
    fit = an.saturation_fit(strain, mu, np.full(strain.shape, 1e-4))
    assert fit["mu_inf"] == pytest.approx(mu_inf, abs=2e-3)
    assert fit["gamma0"] == pytest.approx(gamma0, rel=0.35)


def test_saturation_fit_too_few_points():
    fit = an.saturation_fit(np.arange(3.0), np.arange(3.0), np.ones(3))
    assert math.isnan(fit["mu_inf"])


def test_verdict_point_band_and_direction():
    I = 0.05
    ref = float(an.rl.ref_mu(I))
    ok = an.verdict_point(ref * 1.05, I)
    assert ok["within_tol"] and abs(ok["err_pct"] - 5.0) < 1e-9
    bad = an.verdict_point(ref * 0.80, I)
    assert not bad["within_tol"] and bad["err_pct"] < 0.0
    # a *single* point 20 % below the law still sits inside the wide provenance
    # envelope (mu_s can be as low as 0.34), so the oracle must not fire on it -
    # the envelope question is a curve-level question, see the next test.
    assert bad["envelope_best_case_err_pct"] < 10.0
    # ... and the oracle still refuses to call that a rescue: the best fit sits
    # on the edge of the search box (mu_s = 0.34), i.e. it is not identified.
    assert bad["envelope_can_rescue"] is False


def test_verdict_curve_envelope_cannot_rescue_a_factor_three_deficit():
    """Round 25's archived 2-D probe is a factor ~3 low: no admissible law fits."""
    I = [0.0261, 0.0631, 0.1393, 0.2464]
    mu = [0.0963, 0.1003, 0.1238, 0.1294]
    calls = [an.verdict_point(m, i) for i, m in zip(I, mu)]
    assert all(not c["within_tol"] for c in calls)
    curve = an.curve_envelope(I, mu)
    assert curve["min_max_abs_rel_err"] > 1.0        # > 100 % worst point
    assert curve["provenance_can_rescue"] is False


def test_seed_ensemble_aggregates_and_flags_stationarity():
    def run(mu, phi, stationary, seed):
        return {"file": f"f{seed}", "w": 0.5, "I_target": 0.05, "seed": seed,
                "mu_wall_last_half": mu, "phi_gap_last_half": phi,
                "I_bulk_last_half": 0.05, "stationary": stationary}
    ens = an.seed_ensemble([run(0.20, 0.81, True, 0), run(0.22, 0.79, False, 1)])
    assert len(ens) == 1
    row = ens[0]
    assert row["n_seeds"] == 2
    assert row["mu"] == pytest.approx(0.21)
    assert row["mu_seed_sem"] == pytest.approx(0.01)
    assert row["all_stationary"] is False and row["any_stationary"] is True


def test_campaign_matches_archived_round26_case():
    """Only ``strain_measure`` may differ from the archived round-26 config."""
    c = camp.case_for(0.5, 0.05, 0)
    assert c.strain_measure == camp.STRAIN_MEASURE == 8.0
    assert (c.n_particles, c.width, c.gap) == (458, 0.30, 0.15)
    assert (c.settle_time_units, c.shear_settle_strain) == (40.0, 0.5)
    assert (c.init_mode, c.polydispersity, c.mu_p, c.normal_stress) == \
        ("random", 0.5, 0.4, 1.0e4)
    ref = ps.PlaneShear2DCase(I_target=0.05, n_particles=458, width=0.30, gap=0.15,
                              settle_time_units=40.0, shear_settle_strain=0.5,
                              strain_measure=1.0, init_mode="random", seed=0,
                              polydispersity=0.5)
    for field in ("I_target", "radius", "density", "kn", "kt_ratio", "zeta",
                  "mu_p", "mu_rolling", "seed", "servo_gain", "phi_init",
                  "sample_every", "init_mode", "polydispersity"):
        assert getattr(c, field) == getattr(ref, field), field


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))