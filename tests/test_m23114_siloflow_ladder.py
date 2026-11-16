"""Pin the m23114 numerical-ladder estimator on synthetic ground truth.

Every assertion here is about an *invariant* of the estimator (order recovery,
exact Richardson recovery, verdict classification, sign conventions), not about a
stored number from a DEM run, so the suite cannot be green because a file was
copied.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "m23114_siloflow_ladder"))

import analyse as A  # noqa: E402


# --------------------------------------------------------------------- N1 maths
def test_observed_order_recovers_a_known_power_law():
    w_inf, a, p = 1.5, 0.4, 2.0
    w = [w_inf + a * (0.25 ** p), w_inf + a * (0.125 ** p), w_inf + a * (0.0625 ** p)]
    got = A.observed_order(*w)
    assert abs(got - p) < 1e-9


def test_richardson_recovers_the_exact_limit():
    w_inf, a, p = 1.5, 0.4, 1.0
    w = [w_inf + a * (0.25 ** p), w_inf + a * (0.125 ** p), w_inf + a * (0.0625 ** p)]
    order = A.observed_order(*w)
    assert abs(A.richardson(*w, order) - w_inf) < 1e-12


def test_richardson_handles_first_order_classic_form():
    # classic Aitken/Richardson: W(n) = L + a/n  =>  L = 2*W(2n) - W(n)
    L, a = 2.0, 0.06
    w1, w2, w3 = L + a / 20, L + a / 40, L + a / 80
    assert abs(A.richardson(w1, w2, w3, 1.0) - L) < 1e-12


def test_observed_order_is_nan_for_non_monotone_sequences():
    assert math.isnan(A.observed_order(1.0, 1.1, 1.0))       # ratio < 0
    assert math.isnan(A.observed_order(1.0, 1.0, 1.0))       # division by zero


def test_richardson_is_nan_without_a_positive_order():
    assert math.isnan(A.richardson(1.0, 1.1, 1.2, float("nan")))
    assert math.isnan(A.richardson(1.0, 1.1, 1.2, 0.0))


# --------------------------------------------------------------------- verdicts
@pytest.mark.parametrize("e_dt,e_phys,expected", [
    (0.01, 0.25, "NUMERICS-NEGLIGIBLE"),     # share 0.04
    (0.07, 0.25, "NUMERICS-MIXED"),          # share 0.28
    (0.20, 0.25, "NUMERICS-DOMINATED"),      # share 0.80
    (-0.01, -0.25, "NUMERICS-NEGLIGIBLE"),   # sign-blind magnitudes
    (0.0, 0.25, "NUMERICS-NEGLIGIBLE"),
])
def test_classify_numerics_branches(e_dt, e_phys, expected):
    assert A.classify_numerics(e_dt, e_phys) == expected


def test_classify_numerics_is_not_decidable_when_missing():
    assert A.classify_numerics(None, 0.2) == "NOT_DECIDABLE"
    assert A.classify_numerics(0.1, None) == "NOT_DECIDABLE"
    assert A.classify_numerics(float("nan"), 0.2) == "NOT_DECIDABLE"


# ------------------------------------------------------- overlap mechanism (§5)
def test_overlap_prediction_scales_as_inverse_stiffness():
    kw = dict(radius=0.0015, orifice=0.015, fill_height=0.036)
    a = A.overlap_predicted_bias(5.0e3, **kw)
    b = A.overlap_predicted_bias(1.0e4, **kw)
    assert a["delta_static_m"] == pytest.approx(2.0 * b["delta_static_m"], rel=1e-12)
    # dynamic overlap scales as 1/sqrt(kn)
    assert a["delta_dynamic_m"] == pytest.approx(math.sqrt(2.0) * b["delta_dynamic_m"],
                                                rel=1e-12)
    assert a["predicted_bias_dynamic"] == pytest.approx(
        2.5 * A.K_LIT * 2.0 * a["delta_dynamic_m"] / (0.015 - A.K_LIT * 0.003), rel=1e-12)
    # static overlap for the campaign's own numbers is a few 1e-8 m — negligible
    assert a["delta_static_m"] < 1e-6


def test_overlap_prediction_matches_analytic_single_particle_weight():
    kn = 5.0e3
    got = A.overlap_predicted_bias(kn, radius=0.0015, orifice=0.015, fill_height=0.036)
    m = 2500.0 * 4.0 / 3.0 * math.pi * 0.0015 ** 3
    assert got["delta_static_m"] == pytest.approx(m * 9.81 / kn, rel=1e-12)
    assert got["orifice_velocity_m_s"] == pytest.approx(math.sqrt(2 * 9.81 * 0.036), rel=1e-12)


def test_kn_story_agreement_needs_sign_and_magnitude():
    assert A.kn_story_agrees(0.02, 0.015)            # within a factor 3, same sign
    assert not A.kn_story_agrees(0.02, -0.015)       # wrong sign
    assert not A.kn_story_agrees(0.30, 0.01)         # 30x too large
    assert not A.kn_story_agrees(0.0, 0.01)


# ---------------------------------------------------------------- end to end
def _record(level, seed, w, *, dt, kn=5.0e3, rho_meas=1500.0, rho_nom=1560.0):
    return {
        "case": {
            "orifice_diameter": 0.015, "particle_diameter": 0.003,
            "orifice_over_particle": 5.0, "silo_radius": 0.024,
            "fill_height": 0.036, "wall_mu": 0.0, "seed": seed,
            "kn": kn, "mu": 0.5, "mu_rolling": 0.05,
        },
        "dt": dt,
        "n_bed": 766,
        "rho_bulk": rho_nom,
        "rho_bulk_measured": rho_meas,
        "flow_rate_kg_s": w,
        "flow_rate_windows_kg_s": [w] * 8,
        "window_particles": [40] * 8,
        "steps": 100,
        "wall_clock_s": 1.0,
        "m23114": {"level": level, "seed": seed, "axis": "dt"},
    }


def _ladder(w_inf_scale, numeric_share, *, seed=1, rho_meas=1500.0):
    """Synthetic ladder whose limit sits ``w_inf_scale`` × the literature value."""
    w_ref = A.beverloo_ref(0.015, 0.003, rho_meas)
    w_inf = w_inf_scale * w_ref
    e_dt = numeric_share * (w_inf_scale - 1.0)       # (W∞ − W_nom)/W_nom
    w20 = w_inf / (1.0 + e_dt)
    a = 20.0 * (w20 - w_inf)                          # W(n) = W∞ + a/n  ⇒ p = 1
    recs = {}
    for level, n, kn in (("nom20", 20, 5.0e3), ("nom40", 40, 5.0e3), ("nom80", 80, 5.0e3)):
        recs[(level, seed)] = _record(level, seed, w_inf + a / n, dt=1.0 / n,
                                      kn=kn, rho_meas=rho_meas)
    return recs


def test_analyse_end_to_end_flags_negligible_numerics():
    runs = _ladder(1.25, 0.04)
    s = A.analyse(runs)
    sm = s["N1_dt_convergence"]["seed_mean"]
    assert sm["verdict"] == "NUMERICS-NEGLIGIBLE"
    # nominal rate is W∞/(1+e_dt), so the physics error is 1.25/1.01 − 1
    assert sm["E_phys"] == pytest.approx(1.25 / 1.01 - 1.0, abs=1e-9)
    assert sm["order_obs_mean"] == pytest.approx(1.0, abs=1e-9)
    for v in s["N1_dt_convergence"]["per_seed"].values():
        assert v["W_inf"] == pytest.approx(A.beverloo_ref(0.015, 0.003, 1500.0) * 1.25,
                                          rel=1e-9)


def test_analyse_end_to_end_flags_dominated_numerics():
    runs = _ladder(1.25, 0.80)
    s = A.analyse(runs)
    assert s["N1_dt_convergence"]["seed_mean"]["verdict"] == "NUMERICS-DOMINATED"


def test_analyse_is_not_decidable_when_seed_scatter_swamps_the_bias():
    runs = _ladder(1.25, 0.80, seed=1)
    runs.update(_ladder(1.25, 0.80, seed=2))
    # push one seed far away: the two seeds then disagree by much more than E_dt
    for level in ("nom20", "nom40", "nom80"):
        runs[(level, 2)]["flow_rate_kg_s"] *= 1.30
    s = A.analyse(runs)
    assert s["N1_dt_convergence"]["seed_mean"]["verdict"] == "NOT_DECIDABLE"


def test_analyse_reports_ledger_acceptance_on_the_limit():
    ok = A.analyse(_ladder(1.05, 0.04))
    assert ok["N3_ledger"]["pass_ledger_10pct"] is True
    bad = A.analyse(_ladder(1.25, 0.04))
    assert bad["N3_ledger"]["pass_ledger_10pct"] is False


def test_analyse_stiffness_axis_uses_nom20_as_the_base():
    runs = _ladder(1.25, 0.04)
    base = runs[("nom20", 1)]["flow_rate_kg_s"]
    pred = A.overlap_predicted_bias(8.0e4, radius=0.0015, orifice=0.015, fill_height=0.036)
    w8 = base * (1.0 + 2.0 * pred["predicted_bias_dynamic"])
    runs[("kn8e4", 1)] = _record("kn8e4", 1, w8, dt=1.0 / 20, kn=8.0e4)
    s = A.analyse(runs)
    entry = s["N2_stiffness"]["per_seed"][1]
    assert entry["W_nom"] == pytest.approx(base, rel=1e-12)
    assert entry["E_kn8e4"] == pytest.approx(2.0 * pred["predicted_bias_dynamic"], rel=1e-12)
    assert s["N2_stiffness"]["soft_contact_claim"] == "SUBSTANTIATED"

    # a 20x-too-large stiffness effect must not be accepted as the same story
    runs[("kn8e4", 1)] = _record("kn8e4", 1, base * 1.05, dt=1.0 / 20, kn=8.0e4)
    s2 = A.analyse(runs)
    assert s2["N2_stiffness"]["soft_contact_claim"] == "NOT_SUBSTANTIATED"


def test_load_runs_ignores_records_without_the_ladder_tag(tmp_path, monkeypatch):
    (tmp_path / "a.json").write_text(json.dumps(_record("nom20", 1, 1.0, dt=1e-5)))
    bogus = _record("nom20", 2, 1.0, dt=1e-5)
    del bogus["m23114"]
    (tmp_path / "b.json").write_text(json.dumps(bogus))
    monkeypatch.setattr(A, "RUNS", tmp_path)
    runs = A.load_runs()
    assert list(runs) == [("nom20", 1)]


def test_bulk_density_conventions_are_distinct():
    rec = _record("nom20", 1, 1.0, dt=1e-5, rho_meas=1500.0, rho_nom=1560.0)
    assert A.bulk_density(rec, "measured") == 1500.0
    assert A.bulk_density(rec, "nominal") == 1560.0
    with pytest.raises(ValueError):
        A.bulk_density(rec, "whatever")


def test_rel_error_rejects_a_nonpositive_reference():
    with pytest.raises(ValueError):
        A.rel_error(1.0, 0.0)
    with pytest.raises(ValueError):
        A.rel_error(float("nan"), 1.0)