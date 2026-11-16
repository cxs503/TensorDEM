"""Round 27 (m22662): invariants of the protocol/stationarity audit.

The audit in ``benchmarks/plane_shear_mu_i/audit_r27_protocol.py`` produces the
numbers that round 27's verdict rests on, so its statistics are pinned here on
synthetic series with known answers - not merely executed on the artifacts.

Three groups:

* **statistics** - the integrated autocorrelation time, its effect on the error
  bar of a mean, the OLS slope, and the saturation fit are checked against their
  analytic values on AR(1) / straight-line / exponential data;
* **artifact binding** - the audited ``mu`` is the *same number* stored in the run
  artifact (bit-level, not re-transcribed), and the ordinate ambiguity really is
  as large as the report claims;
* **honesty of the verdict wiring** - a measured series three times below the
  whole reference family must not be rescuable by any admissible parameter set.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "plane_shear_mu_i"
sys.path.insert(0, str(BENCH))

import audit_r27_protocol as au  # noqa: E402
import reference_provenance as rp  # noqa: E402

ARTIFACT = BENCH / "runs" / "plane_shear_2d" / "probe_I0.05.json"


def ar1(n: int, phi: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# --------------------------------------------------------------------------- #
# 1. statistics
# --------------------------------------------------------------------------- #
def test_tau_int_of_white_noise_is_one():
    rng = np.random.default_rng(0)
    tau = au.integrated_autocorr(rng.normal(size=20000))
    assert abs(tau - 1.0) < 0.2, tau


@pytest.mark.parametrize("phi", [0.5, 0.8, 0.9])
def test_tau_int_of_ar1_matches_analytic(phi):
    """For AR(1), ``tau = 1 + 2*sum_k phi^k = (1+phi)/(1-phi)``."""
    tau = au.integrated_autocorr(ar1(40000, phi))
    theory = (1.0 + phi) / (1.0 - phi)
    assert abs(tau / theory - 1.0) < 0.35, (tau, theory)


def test_naive_error_bar_is_optimistic_for_correlated_samples():
    white = au.mean_se(np.random.default_rng(1).normal(size=5000))
    red = au.mean_se(ar1(5000, 0.9))
    assert abs(white["inflation"] - 1.0) < 0.3
    assert red["inflation"] > 3.0, red["inflation"]
    assert red["n_eff"] < red["n"] / 5.0


def test_ols_slope_is_exact_on_a_line():
    x = np.linspace(0.0, 1.0, 500)
    out = au.ols_slope(x, 0.10 - 0.07 * x)
    assert abs(out["slope"] + 0.07) < 1e-9


def test_half_split_finds_a_real_drift():
    g = np.linspace(0.5, 1.5, 2000)
    rng = np.random.default_rng(11)
    out = au.half_split(g, 0.20 - 0.05 * (g - 0.5) + rng.normal(scale=1e-4, size=g.size))
    assert out["status"] == "measured"
    assert out["z"] < -3.0


def test_saturation_fit_recovers_known_asymptote_when_window_is_long():
    g = np.linspace(1.0, 1.0 + 5 * 0.5, 3000)
    mu_inf, amp, lam = 0.35, 0.20, 0.5
    out = au.saturation_fit(g, mu_inf + amp * np.exp(-(g - g[0]) / lam))
    assert out["status"] == "measured", out
    assert abs(out["mu_inf"] - mu_inf) < 0.02 * mu_inf
    assert abs(out["decay_length_strain"] - lam) < 0.1 * lam


def test_saturation_fit_refuses_to_claim_an_asymptote_from_a_short_window():
    """One strain of data, relaxation time 20 strains: 'saturated' is unknowable."""
    g = np.linspace(0.5, 1.5, 800)
    out = au.saturation_fit(g, 0.35 + 0.2 * np.exp(-(g - 0.5) / 20.0))
    assert out["status"] != "measured", out
    if out["status"] == "unidentifiable":
        assert out["at_bound"]


def test_quartile_means_are_monotone_for_a_monotone_series():
    g = np.linspace(0.5, 1.5, 400)
    q = au.quartile_means(g, 0.20 - 0.05 * (g - 0.5))
    assert len(q) == 4
    means = [r["mu"] for r in q]
    assert all(b < a for a, b in zip(means, means[1:]))


def test_random_walk_is_not_reported_as_a_drift():
    """The null case for stationarity: a pure random walk is not a drift.

    An OLS t-statistic on this series is *large* (spurious regression), which is
    exactly why the audit does not use it; the variance-based half-split test
    must stay quiet on the same data.
    """
    rng = np.random.default_rng(7)
    g = np.linspace(0, 1, 600)
    walk = np.cumsum(rng.normal(size=600))
    out = au.half_split(g, walk)
    assert abs(out["z"]) < 4.0, out["z"]


def test_a_flat_series_has_zero_half_split_delta():
    g = np.linspace(0.5, 1.5, 500)
    out = au.half_split(g, np.full_like(g, 0.1234))
    assert abs(out["delta"]) < 1e-12


# --------------------------------------------------------------------------- #
# 2. artifact binding and ordinate ambiguity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not ARTIFACT.exists(), reason="archived run not in this checkout")
def test_audited_mu_equals_the_stored_mu():
    row = au.audit_run("probe_I0.05", ARTIFACT)
    stored = json.loads(ARTIFACT.read_text())["mu_wall"]
    assert row["mu_wall"]["mean"] == pytest.approx(stored, rel=1e-12)
    assert row["status"] == "measured"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="archived run not in this checkout")
def test_pressure_convention_ambiguity_exceeds_the_ledger_tolerance():
    """The judged ordinate swings by far more than 10 % with the choice of P."""
    row = au.audit_run("probe_I0.05", ARTIFACT)
    assert row["mu_variant_spread_rel"] > 0.40, row["mu_variant_spread_rel"]
    assert 0.5 < row["p_bulk_over_p_wall"]["mean"] < 1.0
    # the wall normal stress is the one that reproduces the imposed pressure
    assert row["sigma_yy_over_P"]["mean"] == pytest.approx(1.0, abs=0.2)


# --------------------------------------------------------------------------- #
# 3. honesty of the verdict wiring
# --------------------------------------------------------------------------- #
def test_a_threefold_low_mu_cannot_be_rescued_by_any_admissible_triple():
    I = [0.026, 0.063, 0.139]
    mu = [0.096, 0.100, 0.124]
    out = rp.best_case_over_envelope(I, mu)
    assert out["min_max_abs_rel_err"] > 0.10
    assert out["provenance_can_rescue"] is False


def test_verdict_table_flags_a_pass_only_when_every_point_passes():
    I = 0.05
    good = [rp.PRIMARY_REPO.mu(I)]
    rows = rp.verdict_table([I], good)
    assert any(r["passes_all"] for r in rows)
    bad = [0.5 * rp.PRIMARY_REPO.mu(I)]
    rows = rp.verdict_table([I], bad)
    assert not any(r["passes_all"] for r in rows)


def test_audit_main_writes_all_three_products(tmp_path, monkeypatch):
    monkeypatch.setattr(au, "OUT", tmp_path / "out")
    monkeypatch.setattr(au, "FIG", tmp_path / "fig")
    au.OUT.mkdir(parents=True, exist_ok=True)
    assert au.main() == 0
    assert (tmp_path / "out" / "protocol_audit.json").exists()
    assert (tmp_path / "out" / "protocol_audit.md").exists()
    assert (tmp_path / "fig" / "r27_protocol.png").exists()