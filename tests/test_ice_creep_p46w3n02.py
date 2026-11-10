"""Tests for mission P46W3-N02: Andrade primary creep, fractional Maxwell
relaxation, the four mission acceptance gates and the literature anchors.

The gates themselves are exercised too, so a loosened threshold or a broken gate
fails the suite rather than silently producing a green report.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfc, gamma as _gamma

from tensordem.ice_creep_andrade import (
    ALPHA_ANDRADE,
    AndradeCreepLaw,
    FractionalDemChain,
    andrade_strain,
    exponential_fit_quality,
    fit_andrade,
    fractional_relaxation_reference,
    l1_fractional_integral,
    l1_weights,
    mittag_leffler_e,
    mittag_leffler_e_asymptotic,
    mittag_leffler_e_series,
    relaxation_time_constant,
    tau_scale_correlation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"
LIT_DIR = BENCH_DIR / "creep" / "data"

import sys
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from creep.gates import (  # noqa: E402
    THRESHOLDS,
    andrade_law_for_retardation,
    gate_andrade_curve,
    gate_glen_exponent,
    gate_negative_control,
    norton_bond_relaxation,
)
from creep.literature import (  # noqa: E402
    DIGITIZED_SCHEMA,
    load_digitized_curves,
    load_literature,
    literature_checks,
)

YOUNG = 9.0e9


# --------------------------------------------------------------------------- #
# fractional-calculus kernel
# --------------------------------------------------------------------------- #
def test_l1_weights_are_positive_and_normalised():
    b = l1_weights(50, ALPHA_ANDRADE)
    assert b[0] == pytest.approx(1.0, rel=1e-15)
    assert np.all(b > 0.0)
    assert np.all(np.diff(b) < 0.0)              # monotone decreasing
    # alpha = 1 collapses to the backward-difference weights
    b1 = l1_weights(4, 1.0)
    assert b1[0] == pytest.approx(1.0) and np.allclose(b1[1:], 0.0, atol=1e-15)
    with pytest.raises(ValueError):
        l1_weights(4, 0.0)


def test_l1_fractional_integral_reproduces_power_law():
    """``D^alpha y = const`` -> ``y = g t^alpha / Gamma(1+alpha)``."""
    alpha = ALPHA_ANDRADE
    g = 3.0e-6
    errors = []
    for n_steps in (200, 400, 800):
        dt = 100.0 / n_steps
        values = l1_fractional_integral(np.full(n_steps + 1, g), dt, alpha)
        t = np.arange(n_steps + 1) * dt
        exact = g * t ** alpha / float(_gamma(1.0 + alpha))
        errors.append(abs(values[-1] / exact[-1] - 1.0))
    assert errors[-1] < 5.0e-3
    assert errors[-1] < errors[0]                 # converges with dt
    assert errors[-1] < errors[1]


# --------------------------------------------------------------------------- #
# Mittag-Leffler
# --------------------------------------------------------------------------- #
def test_mittag_leffler_alpha_one_is_exponential():
    xs = np.array([0.0, 0.1, 0.5, 1.0, 2.5, 3.5, 20.0])
    values = np.array([mittag_leffler_e(float(x), 1.0) for x in xs])
    assert np.allclose(values, np.exp(-xs), rtol=1e-12, atol=1e-300)


def test_mittag_leffler_half_order_matches_closed_form():
    """Exact identity ``E_{1/2}(-x) = exp(x^2) erfc(x)``."""
    for x in (0.05, 0.3, 1.0, 2.0, 3.0, 6.0, 12.0):
        exact = math.exp(x * x) * erfc(x)
        assert mittag_leffler_e(x, 0.5) == pytest.approx(exact, rel=2e-3)


def test_mittag_leffler_branches_join_continuously():
    left = mittag_leffler_e_series(2.999, ALPHA_ANDRADE)
    right = mittag_leffler_e_asymptotic(3.001, ALPHA_ANDRADE)
    assert abs(left / right - 1.0) < 2e-2


def test_mittag_leffler_against_high_precision_inversion():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    alpha = mp.mpf(1) / 3

    def reference(x: float) -> float:
        t = mp.mpf(x) ** (1 / alpha)
        return float(mp.invertlaplace(lambda s: s ** (alpha - 1) / (s ** alpha + 1),
                                      t, method="talbot", degree=40))

    for x in (0.3, 1.0, 3.0, 10.0):
        assert mittag_leffler_e(x, float(alpha)) == pytest.approx(
            reference(x), rel=5e-3)


def test_fractional_relaxation_reference_limits():
    t = np.array([0.0, 10.0, 100.0, 1000.0])
    tau = 100.0
    assert np.allclose(fractional_relaxation_reference(t, tau, 1.0),
                       np.exp(-t / tau), rtol=1e-13)
    x = np.sqrt(t / tau)
    assert np.allclose(fractional_relaxation_reference(t, tau, 0.5),
                       np.exp(x * x) * erfc(x), rtol=1e-12)


# --------------------------------------------------------------------------- #
# criterion (b): Andrade creep curve
# --------------------------------------------------------------------------- #
def test_dem_chain_creep_reproduces_andrade_curve():
    """Criterion (b): the strain-time curve matches ``eps_0 + eps_1 t^(1/3)``."""
    chain = FractionalDemChain(n_bonds=8, law=andrade_law_for_retardation())
    out = chain.creep(1.0e6, 1000.0, 2000)
    rel = np.abs(out["strain"] - out["analytic_strain"]) / np.abs(out["analytic_strain"])
    assert float(rel[1:].max()) <= 0.05
    fit = fit_andrade(out["t"][10:], out["strain"][10:])
    assert fit["alpha"] == pytest.approx(ALPHA_ANDRADE, rel=0.02)
    assert fit["eps_0"] == pytest.approx(out["eps_0_analytic"], rel=0.01)
    assert fit["eps_1"] == pytest.approx(out["eps_1_analytic"], rel=0.02)


def test_dem_chain_creep_converges_with_time_step():
    chain = FractionalDemChain(n_bonds=8, law=andrade_law_for_retardation())
    errors = []
    for n_steps in (250, 500, 1000):
        out = chain.creep(1.0e6, 1000.0, n_steps)
        rel = np.abs(out["strain"] - out["analytic_strain"]) / np.abs(out["analytic_strain"])
        errors.append(float(rel[1:].max()))
    assert errors[-1] < errors[0] and errors[-1] < errors[1]
    assert errors[-1] < 0.01


def test_elastic_offset_and_top_displacement_are_dem_outputs():
    """The strain really is read from the nodal displacements of the chain."""
    chain = FractionalDemChain(n_bonds=6, law=AndradeCreepLaw(eta_alpha=1.0e14))
    out = chain.creep(2.0e6, 100.0, 200)
    eps_el = 2.0e6 / YOUNG
    assert out["strain"][0] == pytest.approx(eps_el, rel=1e-12)
    # no creep yet at t = 0: the top displacement is the elastic elongation only
    assert out["top_displacement"][0] == pytest.approx(eps_el * chain.length, rel=1e-12)
    # and the creep term stays negligible for a viscous springpot
    assert out["creep_strain"][-1] / eps_el < 1e-2


# --------------------------------------------------------------------------- #
# criterion (d): negative controls
# --------------------------------------------------------------------------- #
def test_no_creep_parameter_means_no_creep():
    law = AndradeCreepLaw(alpha=ALPHA_ANDRADE, eta_alpha=1.0e18)
    chain = FractionalDemChain(n_bonds=8, law=law)
    out = chain.creep(1.0e6, 1000.0, 200)
    assert out["creep_strain"][-1] / out["elastic_strain"][-1] < 1e-2
    # and the collapse is linear in 1/eta
    stiff = FractionalDemChain(n_bonds=8, law=AndradeCreepLaw(
        alpha=ALPHA_ANDRADE, eta_alpha=law.eta_alpha * 100.0))
    assert stiff.creep(1.0e6, 1000.0, 200)["creep_strain"][-1] == pytest.approx(
        out["creep_strain"][-1] / 100.0, rel=1e-9)


def test_gate_negative_control_passes():
    gate = gate_negative_control()
    assert gate["pass"] is True
    assert gate["norton"]["rel_change"] <= THRESHOLDS["d1_norton_no_creep_rel"]
    assert gate["fractional"]["creep_over_elastic"] <= THRESHOLDS["d2_fractional_no_creep_rel"]


# --------------------------------------------------------------------------- #
# criterion (c): relaxation time versus creep parameter
# --------------------------------------------------------------------------- #
def test_norton_bond_relaxation_matches_closed_form_crossing():
    hist = norton_bond_relaxation(fluidity=1.0e-23, n=3.0, t_end=2.0, n_steps=4000)
    tau = relaxation_time_constant(hist["t"], hist["force_normalised"])
    beta = (2.0 * hist["stiffness"] * hist["fluidity"] * hist["rest"]
            * hist["area"] ** (-3.0) * hist["f0"] ** 2.0)
    assert tau == pytest.approx((math.e ** 2 - 1.0) / beta, rel=0.02)


def test_norton_tau_inversely_proportional_to_fluidity():
    fluidities = (1.0e-23, 1.0e-22, 1.0e-21)
    taus = []
    for fluidity in fluidities:
        hist = norton_bond_relaxation(fluidity=fluidity, n=3.0, t_end=0.5, n_steps=2000)
        taus.append(relaxation_time_constant(hist["t"], hist["force_normalised"]))
    corr = tau_scale_correlation(fluidities, taus)
    assert corr["abs_r"] >= 0.9
    assert abs(corr["exponent"] + 1.0) <= 0.05


def test_fractional_tau_scales_as_eta_to_the_one_over_alpha():
    alpha = ALPHA_ANDRADE
    base = andrade_law_for_retardation(alpha)
    etas, taus = [], []
    for mult in (0.25, 1.0, 4.0):
        chain = FractionalDemChain(n_bonds=8, law=AndradeCreepLaw(
            alpha=alpha, eta_alpha=base.eta_alpha * mult))
        out = chain.relax(1.0e-3, 6.0 * chain.retardation_time, 800)
        etas.append(chain.law.eta_alpha)
        taus.append(relaxation_time_constant(out["t"], out["force_normalised"]))
    corr = tau_scale_correlation(etas, taus)
    assert corr["abs_r"] >= 0.9
    assert abs(corr["exponent"] - 1.0 / alpha) / (1.0 / alpha) <= 0.05


def test_single_exponential_is_exact_only_for_alpha_one():
    chain = FractionalDemChain(n_bonds=8, law=andrade_law_for_retardation(1.0))
    out = chain.relax(1.0e-3, 500.0, 1000)
    assert np.max(np.abs(out["force_normalised"]
                         - np.exp(-out["t"] / out["tau"]))) <= 0.01
    quality = exponential_fit_quality(out["t"], out["force_normalised"])
    assert quality["log_linear_r2"] > 0.999

    third = FractionalDemChain(n_bonds=8, law=andrade_law_for_retardation(ALPHA_ANDRADE))
    run = third.relax(1.0e-3, 2000.0, 1000)
    bad = exponential_fit_quality(run["t"], run["force_normalised"])
    # the Andrade element has a heavy tail: a single exponential cannot fit it
    assert bad["tail_ratio"] > 100.0


# --------------------------------------------------------------------------- #
# gate-level regression tests (they must not silently turn green)
# --------------------------------------------------------------------------- #
def test_predeclared_thresholds_are_not_loosened():
    assert THRESHOLDS["a1_glen_rel_error"] == 0.10
    assert THRESHOLDS["b1_curve_max_rel_error"] == 0.05
    assert THRESHOLDS["b2_alpha_rel_error"] == 0.10
    assert THRESHOLDS["c1_abs_r_norton"] == 0.90
    assert THRESHOLDS["c3_abs_r_fractional"] == 0.90
    assert THRESHOLDS["d1_norton_no_creep_rel"] == 0.01
    assert THRESHOLDS["d2_fractional_no_creep_rel"] == 0.01


def test_gate_b_and_curve_gate_passes():
    gate = gate_andrade_curve()
    assert gate["pass"] is True
    assert gate["max_rel_error_curve"] <= THRESHOLDS["b1_curve_max_rel_error"]


def test_gate_a_mission_criterion_and_honest_statistical_fail():
    """The mission 10 % criterion passes; the statistical CI item does not.

    The CI covers random noise only, so it cannot contain 3.0 while the lattice
    bias is 3 %; that failure is recorded, not hidden.  This test pins the
    interpretation down so a future change cannot quietly flip it.
    """
    gate = gate_glen_exponent()
    assert gate["mission_criterion_pass"] is True
    assert gate["rel_error_vs_literature"] <= THRESHOLDS["a1_glen_rel_error"]
    assert gate["statistical_ci_contains_three"] is False
    assert gate["pass"] is False
    assert gate["error_budget_posthoc"]["total_within_mission_10pct"] is True


# --------------------------------------------------------------------------- #
# literature anchors
# --------------------------------------------------------------------------- #
def test_literature_file_declares_provenance_for_every_entry():
    lit = load_literature()
    for key, entry in lit["quantities"].items():
        assert entry["provenance"], key
        if entry["kind"] == "band":
            assert "value" not in entry, key
    assert any(item["status"] == "WITHHELD" for item in lit["withheld_digitizations"])


def test_literature_checks_pass_for_calibrated_fluidity_and_flag_the_default():
    n_fit = 2.9134
    alpha_fit = 0.3326
    q = 61.5
    for a_ref, expect_all in ((3.5e-25, False), (3.5e-24, False)):
        rate = a_ref * (1.0e6) ** 3.0
        out = literature_checks({
            "n_fit": n_fit, "a_minus10": a_ref, "q_kj_per_mol": q,
            "eta_eff_1mpa": 1.0e6 / rate, "min_rate_1mpa_minus10": rate,
            "alpha_fit": alpha_fit,
        })
        assert out["glen_exponent"]["pass"] is True
        assert out["andrade_exponent"]["pass"] is True
        assert out["activation_energy"]["warm_pass"] is True
        assert out["curve_overlay"]["status"] == "WITHHELD"
        assert out["curve_overlay"]["pass"] is None
        if a_ref == 3.5e-25:
            assert out["fluidity_minus10C"]["inside_band"] is True
            assert out["eta_eff_1MPa"]["inside_band"] is True
    # the module default sits one decade above the anchor band: it must be
    # reported as a failure of that item, never re-banded to pass
    rate = 3.5e-24 * 1.0e18
    default = literature_checks({
        "n_fit": n_fit, "a_minus10": 3.5e-24, "q_kj_per_mol": q,
        "eta_eff_1mpa": 1.0e6 / rate, "min_rate_1mpa_minus10": rate,
        "alpha_fit": alpha_fit,
    })
    assert default["fluidity_minus10C"]["pass"] is False
    assert "fluidity_minus10C" in default["summary"]["failed"]


def test_digitized_curve_loader_is_empty_and_rejects_incomplete_files(tmp_path):
    assert load_digitized_curves() == {}
    (tmp_path / "fake.csv").write_text("time_s,strain\n0,0\n1,1e-5\n")
    with pytest.raises(ValueError):
        load_digitized_curves(tmp_path)          # no sidecar json -> rejected
    assert DIGITIZED_SCHEMA["columns"] == ["time_s", "strain"]
    (tmp_path / "fake.json").write_text(json.dumps({"source": "?"}))
    with pytest.raises(ValueError):
        load_digitized_curves(tmp_path)          # sidecar lacks temperature/stress


def test_andrade_strain_is_the_declared_criterion_shape():
    t = np.array([1.0, 8.0, 27.0])
    values = andrade_strain(t, 1.0e-4, 2.0e-5, ALPHA_ANDRADE)
    assert np.allclose(values, 1.0e-4 + 2.0e-5 * t ** (1.0 / 3.0), rtol=1e-14)


def test_fit_andrade_rejects_short_records():
    with pytest.raises(ValueError):
        fit_andrade([1.0, 2.0], [1e-4, 2e-4])