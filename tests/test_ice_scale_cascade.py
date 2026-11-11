"""Tests for :mod:`tensordem.ice_scale_cascade` and its benchmark driver.

Two families:

* analytic tests -- the scaling rules, the dimensionless algebra and the
  regression/criteria machinery, all cheap and exact;
* three small DEM runs (18 floes, short transit) that check the parts only a
  simulation can check: determinism, the geometry the case builder produces,
  the contacts-per-floe scale invariance and a miniature Cauchy collapse.

The production cascade (152 floes, 5 scale ratios, 27 cases) lives in
``benchmarks/ice_scale_cascade.py`` and is run on the cluster.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from tensordem.ice_scale_cascade import (
    PROTOCOL_CAUCHY,
    PROTOCOL_FROUDE,
    PROTOCOL_MISMATCH,
    PROTOTYPE,
    REFERENCE_FRACTION,
    REFERENCE_MODULUS_RATIO,
    ScaleCase,
    build_case,
    build_simulation,
    collapse_deviation,
    collapse_statistics,
    contact_sensitivity_pairs,
    convergence_check,
    evaluate_criteria,
    exact_covariance_pairs,
    fit_deviation_law,
    group_table,
    linear_fit,
    loglog_fit,
    modulus_fraction,
    nondimensional_groups,
    simulate_scale_case,
)

REPO = Path(__file__).resolve().parents[1]


def _load_benchmark():
    path = REPO / "benchmarks" / "ice_scale_cascade.py"
    spec = importlib.util.spec_from_file_location("ice_scale_cascade_bench", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# fixtures: two small DEM members of a cascade
# ---------------------------------------------------------------------------

TINY = dict(dt_scale=0.5, domain_length_ratio=4.0, domain_width_ratio=2.5,
            contact_sample_every=5)


@pytest.fixture(scope="module")
def tiny_pair():
    """(lambda=10, lambda=20) Cauchy members, 18 floes, ~800 steps each."""
    out = {}
    for lam in (10.0, 20.0):
        case = build_case(PROTOCOL_CAUCHY, lam, label=f"tiny_lam{lam:g}", **TINY)
        out[lam] = {"case": case, "record": simulate_scale_case(case)}
    return out


# ---------------------------------------------------------------------------
# 1. modulus / penalty mapping
# ---------------------------------------------------------------------------

def test_modulus_fraction_inverse_square():
    a = modulus_fraction(REFERENCE_MODULUS_RATIO)
    b = modulus_fraction(2.0 * REFERENCE_MODULUS_RATIO)
    assert a == pytest.approx(REFERENCE_FRACTION)
    assert b / a == pytest.approx(0.25)          # frac ~ Mod^-2
    c = modulus_fraction(0.5 * REFERENCE_MODULUS_RATIO)
    assert c / a == pytest.approx(4.0)


# 100 is a real "too soft" modulus ratio: frac would exceed 0.5
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), 100.0])
def test_modulus_fraction_rejects_invalid(bad):
    with pytest.raises(ValueError):
        modulus_fraction(bad)


# ---------------------------------------------------------------------------
# 2. protocol algebra
# ---------------------------------------------------------------------------

def test_cauchy_case_matches_every_group():
    proto = nondimensional_groups(
        speed_m_s=PROTOTYPE["speed_m_s"], hull_length_m=4.0 * PROTOTYPE["beam_m"],
        strength_pa=PROTOTYPE["strength_pa"], thickness_m=PROTOTYPE["thickness_m"],
        modulus_pa=PROTOTYPE["modulus_pa"], concentration=PROTOTYPE["concentration"],
        friction=PROTOTYPE["friction"], beam_m=PROTOTYPE["beam_m"])
    for lam in (5.0, 10.0, 20.0, 40.0):
        g = build_case(PROTOCOL_CAUCHY, lam).groups()
        for key in ("Fn", "Sigma", "Mod", "t_over_L", "phi", "mu",
                    "B_over_t", "L_over_t"):
            assert g[key] == pytest.approx(proto[key], rel=1e-12), (lam, key)


def test_cauchy_length_speed_strength_scaling():
    case = build_case(PROTOCOL_CAUCHY, 20.0)
    assert case.beam_m == pytest.approx(PROTOTYPE["beam_m"] / 20.0)
    assert case.thickness_m == pytest.approx(PROTOTYPE["thickness_m"] / 20.0)
    assert case.speed_m_s == pytest.approx(PROTOTYPE["speed_m_s"] / math.sqrt(20.0))
    assert case.strength_pa == pytest.approx(PROTOTYPE["strength_pa"] / 20.0)
    assert case.modulus_pa == pytest.approx(PROTOTYPE["modulus_pa"] / 20.0)
    assert case.modulus_factor == pytest.approx(1.0)


@pytest.mark.parametrize("p_e", [0.0, 0.5, 1.0])
def test_froude_protocol_modulus_mismatch_exponent(p_e):
    for lam in (5.0, 10.0, 20.0):
        case = build_case(PROTOCOL_FROUDE, lam, modulus_exponent=p_e)
        assert case.modulus_factor == pytest.approx(lam ** (1.0 - p_e))
        # Froude and strength groups still matched:
        g = case.groups()
        assert g["Fn"] == pytest.approx(2.0 / math.sqrt(9.80665 * 80.0))
        assert g["Sigma"] == pytest.approx(
            PROTOTYPE["strength_pa"] / (917.0 * 9.80665 * PROTOTYPE["thickness_m"]))


@pytest.mark.parametrize("dmod", [-0.3, -0.1, 0.1, 0.3])
def test_mismatch_protocol_applies_delta(dmod):
    case = build_case(PROTOCOL_MISMATCH, 10.0, modulus_factor=1.0 + dmod)
    assert case.modulus_factor == pytest.approx(1.0 + dmod)
    # Cauchy modulus times q:
    cauchy = build_case(PROTOCOL_CAUCHY, 10.0)
    assert case.modulus_pa == pytest.approx(cauchy.modulus_pa * (1.0 + dmod))


def test_time_step_follows_froude_time_scale():
    """dt ~ lam^-1/2, i.e. the step itself is Cauchy covariant."""
    a = build_case(PROTOCOL_CAUCHY, 10.0)
    b = build_case(PROTOCOL_CAUCHY, 40.0)
    assert b.dt / a.dt == pytest.approx(0.5, rel=1e-9)   # sqrt(10/40)


def test_contact_stiffness_scales_as_length_squared():
    a = build_case(PROTOCOL_CAUCHY, 10.0)
    b = build_case(PROTOCOL_CAUCHY, 20.0)
    # k_n = 2 sigma t / sqrt(frac): sigma and t both halve -> k_n / 4
    assert b.contact_stiffness_n_m / a.contact_stiffness_n_m == pytest.approx(0.25)


def test_geometry_scales_geometrically():
    a = build_case(PROTOCOL_CAUCHY, 10.0)
    b = build_case(PROTOCOL_CAUCHY, 20.0)
    assert b.floe_radius_m / a.floe_radius_m == pytest.approx(0.5)
    assert b.domain_m[0] / a.domain_m[0] == pytest.approx(0.5)
    assert b.domain_m[1] / a.domain_m[1] == pytest.approx(0.5)


def test_transit_time_and_window():
    a = build_case(PROTOCOL_CAUCHY, 10.0)
    b = build_case(PROTOCOL_CAUCHY, 40.0)
    assert b.transit_time_s / a.transit_time_s == pytest.approx(0.5, rel=1e-9)
    t0, t1 = a.steady_window_s
    assert 0.0 < t0 < t1 < a.transit_time_s


@pytest.mark.parametrize("kwargs", [
    dict(lam=0.5), dict(protocol="P-nonsense"), dict(thickness_m=-1.0),
    dict(contact_sample_every=0), dict(dt_scale=0.0),
])
def test_scale_case_validation(kwargs):
    base = dict(lam=10.0, protocol=PROTOCOL_CAUCHY)
    base.update(kwargs)
    with pytest.raises(ValueError):
        ScaleCase(**base)


# ---------------------------------------------------------------------------
# 3. field construction and DEM behaviour
# ---------------------------------------------------------------------------

def test_build_simulation_field_is_geometrically_similar():
    small = build_case(PROTOCOL_CAUCHY, 20.0, **TINY)
    large = build_case(PROTOCOL_CAUCHY, 10.0, **TINY)
    sim_s, _ = build_simulation(small)
    sim_l, _ = build_simulation(large)
    assert sim_s.n == sim_l.n                      # same field, same seed
    assert float(sim_s.radius.mean() / sim_l.radius.mean()) == pytest.approx(0.5,
                                                                            rel=1e-9)
    stats = sim_s.field_statistics()
    assert stats["realised_concentration"] == pytest.approx(0.7, rel=0.02)
    assert abs(float(sim_s.initial_overlap)) < 1e-3 * small.floe_radius_m


def test_tiny_pair_is_deterministic(tiny_pair):
    case = tiny_pair[10.0]["case"]
    again = simulate_scale_case(case)
    assert again["R_star"] == pytest.approx(tiny_pair[10.0]["record"]["R_star"],
                                            rel=1e-12)
    assert again["state_digest"] == tiny_pair[10.0]["record"]["state_digest"]


def test_tiny_cascade_collapses(tiny_pair):
    """Miniature of criterion (a): |delta| < 15 % across a 2:1 scale step."""
    ref = tiny_pair[10.0]["record"]["R_star"]
    d = collapse_deviation(tiny_pair[20.0]["record"]["R_star"], ref)
    assert abs(d["delta_pct"]) < 15.0, d
    assert tiny_pair[10.0]["record"]["floe_count"] == \
        tiny_pair[20.0]["record"]["floe_count"]


def test_tiny_contacts_per_floe_is_scale_invariant(tiny_pair):
    a = tiny_pair[10.0]["record"]["contacts_per_floe"]
    b = tiny_pair[20.0]["record"]["contacts_per_floe"]
    assert a > 0.0
    assert abs(b - a) / a < 0.10, (a, b)


def test_record_carries_dimensionless_resistance(tiny_pair):
    rec = tiny_pair[10.0]["record"]
    case = tiny_pair[10.0]["case"]
    assert rec["R_star"] == pytest.approx(
        rec["R_mean_N"] / (case.strength_pa * case.thickness_m ** 2))
    assert rec["samples"] > 10
    assert math.isfinite(rec["R_rel_sem"])


# ---------------------------------------------------------------------------
# 4. regression / criteria machinery
# ---------------------------------------------------------------------------

def test_loglog_fit_recovers_power_law():
    xs = [1.0, 2.0, 4.0, 8.0, 16.0]
    ys = [3.0 * x ** 1.75 for x in xs]
    fit = loglog_fit(xs, ys)
    assert fit["exponent"] == pytest.approx(1.75, rel=1e-9)
    assert fit["prefactor"] == pytest.approx(3.0, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


def test_loglog_fit_rejects_bad_input():
    with pytest.raises(ValueError):
        loglog_fit([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        loglog_fit([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        loglog_fit([1.0, 2.0, 3.0], [1.0, 2.0])


def test_linear_fit():
    fit = linear_fit([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(1.0)
    assert fit["r2"] == pytest.approx(1.0)


def test_fit_deviation_law_selects_quadratic_form():
    pts = [{"dMod": dm, "delta_pct": 10.0 * dm + 40.0 * dm ** 2}
           for dm in (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)]
    law = fit_deviation_law(pts)
    assert law["form"] == "quadratic"
    assert law["fits"]["quadratic"]["a"] == pytest.approx(10.0, rel=1e-6)
    assert law["fits"]["quadratic"]["b"] == pytest.approx(40.0, rel=1e-6)
    assert law["fits"]["quadratic"]["r2"] > 0.99


def test_fit_deviation_law_needs_points():
    with pytest.raises(ValueError):
        fit_deviation_law([{"dMod": 0.1, "delta_pct": 1.0}])


def test_convergence_check_uses_finest_pair():
    records = [
        {"case": {"dt_scale": 0.5}, "dt_s": 2.0, "steps": 10, "R_star": 1.0,
         "wall_time_s": 1.0},
        {"case": {"dt_scale": 0.25}, "dt_s": 1.0, "steps": 20, "R_star": 1.1,
         "wall_time_s": 1.0},
        {"case": {"dt_scale": 0.125}, "dt_s": 0.5, "steps": 40, "R_star": 1.105,
         "wall_time_s": 1.0},
    ]
    conv = convergence_check(records)
    # levels are sorted ascending in dt_scale -> rel[0] is the finest pair,
    # rel[-1] the coarsest; each is normalised by the finer R*
    assert conv["rel_change_finest"] == pytest.approx(0.005 / 1.1, rel=1e-9)
    assert conv["rel_change_coarsest"] == pytest.approx(0.1 / 1.0, rel=1e-9)
    # finest step first (levels ascend in dt_scale)
    assert [lv["dt_scale"] for lv in conv["levels"]] == [0.125, 0.25, 0.5]


def test_convergence_check_needs_three_levels():
    with pytest.raises(ValueError):
        convergence_check([{"case": {"dt_scale": 0.5}, "dt_s": 1.0, "steps": 1,
                            "R_star": 1.0, "wall_time_s": 1.0}] * 2)


def test_contact_sensitivity_pairs_math():
    base = {"id": "base", "contacts_per_floe": 4.0, "delta_pct": 2.0}
    variants = [{"id": "v1", "contacts_per_floe": 4.4, "delta_pct": 2.5,
                 "knob": "seed"}]
    out = contact_sensitivity_pairs(base, variants)
    assert out["pairs"][0]["contact_change_pct"] == pytest.approx(10.0)
    assert out["pairs"][0]["delta_change_pct"] == pytest.approx(0.5)


def test_exact_covariance_pairs_selects_power_of_four():
    def rec(cid, lam, r_star, seed=20261110, **over):
        case = build_case(PROTOCOL_CAUCHY, lam, seed=seed, **over)
        return {"id": cid, "R_star": r_star, "case": case.as_dict()}

    records = [rec("a", 10.0, 0.10), rec("b", 40.0, 0.10), rec("c", 20.0, 0.11),
               rec("d", 40.0, 0.10, seed=7)]
    pairs = exact_covariance_pairs(records)
    # only the ratio-4 pair is included: a/b (the ratio-2 pair a/c is excluded
    # because the Froude velocity is not exact under a factor-2 scaling)
    names = {(p["a"], p["b"]) for p in pairs}
    assert ("a", "b") in names
    assert not any("c" in pair for pair in names)
    assert all(p["lam_ratio"] == pytest.approx(4.0) for p in pairs)
    exact = [p for p in pairs if p["exact"]]
    assert exact and exact[0]["rel_diff"] == pytest.approx(0.0)


def test_collapse_statistics_and_deviation():
    stats = collapse_statistics([1.0, 1.02, 0.98])
    assert stats["n"] == 3
    assert stats["max_rel_departure"] == pytest.approx(0.04 / 1.0, rel=1e-6)
    d = collapse_deviation(1.05, 1.0)
    assert d["delta_pct"] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        collapse_deviation(1.0, 0.0)


def _synthetic_criteria_input(*, cauchy_delta=2.0, froude_exponent=1.0,
                              contact_change=0.5, conv=0.002):
    cauchy = [{"id": f"c{i}", "lam": lam, "delta_pct": cauchy_delta}
              for i, lam in enumerate((5.0, 10.0, 20.0, 40.0))]
    froude = [{"id": f"f{i}", "lam": lam, "delta_pct": froude_exponent * lam}
              for i, lam in enumerate((5.0, 10.0, 20.0, 40.0))]
    mismatch = [{"dMod": dm, "delta_pct": 12.0 * dm} for dm in (-0.2, -0.1, 0.1, 0.2)]
    convergence = {"rel_change_finest": conv, "levels": []}
    pairs = [{"knob": "seed", "contacts_per_floe": 4.0,
              "contact_change_pct": 6.0, "delta_pct": 1.0 + contact_change,
              "delta_change_pct": contact_change}]
    return dict(cauchy=cauchy, froude=froude, mismatch=mismatch,
                convergence=convergence, contact_sensitivity={"pairs": pairs})


def test_evaluate_criteria_all_pass():
    crit = evaluate_criteria(**_synthetic_criteria_input())
    assert crit["a"]["pass"] and crit["b"]["pass"]
    assert crit["c"]["pass"] and crit["d"]["pass"]
    assert crit["all_pass"]
    assert crit["deviation_law"]["form"] in ("linear", "quadratic", "power")


def test_evaluate_criteria_detects_failures():
    crit = evaluate_criteria(**_synthetic_criteria_input(cauchy_delta=20.0,
                                                         contact_change=9.0,
                                                         conv=0.05))
    assert not crit["a"]["pass"]
    assert not crit["c"]["pass"]
    assert not crit["d"]["pass"]
    assert not crit["all_pass"]


def test_group_table_shape():
    rec = {"id": "x", "lam": 10.0, "protocol": PROTOCOL_CAUCHY, "modulus_factor": 1.0,
           "groups": {"Fn": 0.07, "Sigma": 0.11, "Mod": 5000.0, "t_over_L": 0.0125,
                      "phi": 0.7, "mu": 0.3, "B_over_t": 20.0, "L_over_t": 80.0},
           "R_star": 0.01, "contacts_per_floe": 4.0}
    table = group_table([rec])
    assert table[0]["Fn"] == pytest.approx(0.07)
    assert table[0]["Mod"] == pytest.approx(5000.0)
    assert set(("id", "lam", "protocol", "R_star")) <= set(table[0])


# ---------------------------------------------------------------------------
# 5. benchmark driver (design + analysis plumbing, no DEM)
# ---------------------------------------------------------------------------

def test_design_tasks_is_complete_and_valid():
    bench = _load_benchmark()
    tasks = bench.design_tasks()
    kinds = {t["kind"] for t in tasks}
    assert kinds == {"cauchy", "froude", "mismatch", "resolution", "contacts",
                     "ensemble"}
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    assert len(tasks) >= 25
    for t in tasks:
        case = t["case"]
        assert isinstance(case, ScaleCase)
        assert 0.0 < case.reference_penetration_fraction < 0.5
    # the Cauchy ladder spans the task book's two model scales
    cauchy = sorted(t["case"].lam for t in tasks if t["kind"] == "cauchy")
    assert {10.0, 20.0} <= set(cauchy)
    # the mismatch family brackets +-30 %
    dm = sorted(t["case"].modulus_factor - 1.0 for t in tasks if t["kind"] == "mismatch")
    assert dm[0] == pytest.approx(-0.30) and dm[-1] == pytest.approx(0.30)


def test_benchmark_ingest_log_roundtrip(tmp_path, monkeypatch):
    bench = _load_benchmark()
    monkeypatch.setattr(bench, "CASE_DIR", tmp_path)
    record = {"id": "caselog", "protocol": PROTOCOL_CAUCHY, "lam": 10.0,
              "R_star": 0.0123, "kind": "cauchy"}
    log = tmp_path / "sweep.log"
    log.write_text("noise\nRECORD " + json.dumps(record) + "\nDONE {}\n")
    n = bench.ingest_log(log)
    assert n == 1
    written = json.loads((tmp_path / "caselog.json").read_text())
    assert written["R_star"] == pytest.approx(0.0123)


def _synthetic_records():
    """A small, fully synthetic case set that exercises ``analyse`` + report."""
    bench = _load_benchmark()
    records = []
    ref_groups = build_case(PROTOCOL_CAUCHY, 1.0).groups()
    for lam in (1.0, 5.0, 10.0, 20.0, 40.0):
        case = build_case(PROTOCOL_CAUCHY, lam)
        records.append({"id": case.id(), "kind": "cauchy", "knob": "",
                        "protocol": PROTOCOL_CAUCHY, "lam": lam,
                        "modulus_factor": 1.0, "groups": case.groups(),
                        "R_star": 0.0100 * (1.0 + 0.01 * math.log(lam)),
                        "contacts_per_floe": 4.0 + 0.05 * math.log(lam),
                        "floe_count": 152, "case": case.as_dict(),
                        "delta_pct": float("nan")})
    for lam in (5.0, 10.0, 20.0, 40.0):
        case = build_case(PROTOCOL_FROUDE, lam)
        records.append({"id": case.id(), "kind": "froude", "knob": "",
                        "protocol": PROTOCOL_FROUDE, "lam": lam,
                        "modulus_factor": case.modulus_factor,
                        "groups": case.groups(),
                        "R_star": 0.0100 * (1.0 + 0.004 * lam ** 0.9),
                        "contacts_per_floe": 4.0, "floe_count": 152,
                        "case": case.as_dict(), "delta_pct": float("nan")})
    for dm in (-0.3, -0.1, 0.1, 0.3):
        case = build_case(PROTOCOL_MISMATCH, 10.0, modulus_factor=1.0 + dm)
        records.append({"id": case.id(), "kind": "mismatch", "knob": "",
                        "protocol": PROTOCOL_MISMATCH, "lam": 10.0,
                        "modulus_factor": case.modulus_factor,
                        "groups": case.groups(),
                        "R_star": 0.0100 * (1.0 + 0.05 * dm),
                        "contacts_per_floe": 4.0, "floe_count": 152,
                        "case": case.as_dict(), "delta_pct": float("nan")})
    for dt_scale, r_star in ((0.5, 0.0098), (0.25, 0.0100), (0.125, 0.01001)):
        case = build_case(PROTOCOL_CAUCHY, 10.0, dt_scale=dt_scale)
        records.append({"id": case.id(), "kind": "resolution", "knob": "",
                        "protocol": PROTOCOL_CAUCHY, "lam": 10.0,
                        "modulus_factor": 1.0, "groups": case.groups(),
                        "R_star": r_star, "contacts_per_floe": 4.0,
                        "dt_s": case.dt, "steps": 10, "wall_time_s": 1.0,
                        "floe_count": 152, "case": case.as_dict(),
                        "delta_pct": float("nan")})
    base = build_case(PROTOCOL_MISMATCH, 10.0, modulus_factor=1.3)
    records.append({"id": base.id(), "kind": "contacts", "knob": "",
                    "protocol": PROTOCOL_MISMATCH, "lam": 10.0,
                    "modulus_factor": 1.3, "groups": base.groups(),
                    "R_star": 0.0105, "contacts_per_floe": 4.0,
                    "floe_count": 152, "case": base.as_dict(),
                    "delta_pct": float("nan")})
    for knob, cpf in (("seed_a", 4.2), ("field_jitter_fine", 3.9)):
        records.append({"id": f"contacts_{knob}", "kind": "contacts", "knob": knob,
                        "protocol": PROTOCOL_MISMATCH, "lam": 10.0,
                        "modulus_factor": 1.3, "groups": base.groups(),
                        "R_star": 0.0106, "contacts_per_floe": cpf,
                        "floe_count": 152, "case": base.as_dict(),
                        "delta_pct": float("nan")})
    return bench, records


def test_analyse_and_report_end_to_end():
    bench, records = _synthetic_records()
    summary = bench.analyse(records)
    assert summary["reference"]["lam"] == 1.0
    crit = summary["criteria"]
    assert crit["a"]["pass"] is True           # synthetic Cauchy is flat
    assert crit["d"]["pass"] is True
    assert "deviation_law" in summary
    text = bench.render_report(summary)
    assert "criterion" in text and "Cauchy collapse" in text
    assert "PASS" in text or "FAIL" in text