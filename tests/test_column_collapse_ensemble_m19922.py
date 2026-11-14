"""Gate for the CASE-COLUMN-COLLAPSE ensemble wave (m19922).

Two independent classes of checks:

1. **Machinery** — the reducer, the ensemble statistics, the archive
   reproduction check and the five pre-registered hypothesis adjudications are
   exercised on synthetic input, including controls that *must* fail (an
   out-of-band member, an over-wide ensemble, a shallow-column time check that
   is inside the band, an archive that does not reproduce).  A silent change of
   the decision logic therefore cannot pass unnoticed.
2. **Data regression** — the committed bundle
   (``docs/artifacts/column_collapse_ensemble_m19922/``) is re-adjudicated
   here.  Assertions are *relations* recomputed from the case rows (reference
   values from the Lube formulas, verdicts from the pre-registered thresholds),
   not typed numbers, so report and data cannot drift apart.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "analyze_column_collapse_ensemble_m19922.py")
BENCH = os.path.join(ROOT, "benchmarks",
                     "case_column_collapse_ensemble_m19922.py")
ART = os.path.join(ROOT, "docs", "artifacts", "column_collapse_ensemble_m19922")
BUNDLE = os.path.join(ART, "summary.json")
CASES_CSV = os.path.join(ART, "cases_ensemble.csv")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load(SCRIPT, "m19922_ensemble_reducer")


def _raw(a, jitter, seed, ratio, t99=None, member=None, r_i=None):
    """Synthetic per-case JSON in the shape ``simulate_axisymmetric_column`` returns."""
    r_i = 8.0 * 2.0 * 0.03 if r_i is None else r_i  # = AxisymCase.column_radius
    g = 9.81
    t_ref = M.lube_2004_emplacement_time(a, r_i, g)
    t99 = (1.2 * t_ref) if t99 is None else t99
    label = f"ens_a{a:g}_m{seed}"
    return {
        "case": {"label": label, "aspect_ratio": float(a), "grains_per_radius": 8.0,
                 "radius": 0.03, "gravity": g, "jitter": float(jitter),
                 "seed": int(seed)},
        "ensemble": {"member": member or f"m{seed}", "jitter": float(jitter),
                     "seed": int(seed)},
        "n_particles": 981, "dt": 4.8e-4, "n_steps": 2156, "stop_reason": "at_rest",
        "aspect_measured": float(a) * 1.1,
        "runout_ratio_p995": float(ratio),
        "runout_ratio_max": float(ratio) * 1.05,
        "t_reach_by_fraction": {"0.9": 0.95 * t99, "0.95": 0.97 * t99, "0.99": t99},
        "energy": {"residual_fraction_of_potential": 0.01},
        "peak_front_speed": 0.8,
        "wall_clock_s": 20.0,
    }


# ------------------------------- machinery --------------------------------- #
def test_reference_values_are_evaluated_from_the_lube_laws():
    law = M.lube_2004_runout_ratio
    assert law(0.5) == pytest.approx(1.62, abs=1e-12)
    assert law(2.0) == pytest.approx(1.0 + 1.6 * math.sqrt(2.0), abs=1e-12)
    row = M.reduce_case(_raw(2.0, 0.05, 101, law(2.0) * 0.98))
    assert row["ref_runout_ratio"] == pytest.approx(law(2.0), abs=1e-12)
    assert row["err_runout"] == pytest.approx(-0.02, abs=2e-3)
    # the measured-a convention is a different, strictly larger reference at a>1.7
    assert row["ref_runout_ratio_measured_a"] != row["ref_runout_ratio"]


def test_reduce_case_tolerates_missing_ensemble_block():
    raw = _raw(1.0, 0.0, 12345, 2.1)
    raw.pop("ensemble")
    row = M.reduce_case(raw)
    assert row["jitter"] == 0.0 and row["seed"] == 12345
    assert math.isfinite(row["err_t95"])


def test_summary_statistics_and_band_counting():
    law = M.lube_2004_runout_ratio
    ref = law(1.0)
    rows = [M.reduce_case(_raw(1.0, 0.0, 1, ref * 0.97)),
            M.reduce_case(_raw(1.0, 0.05, 2, ref * 1.00)),
            M.reduce_case(_raw(1.0, 0.10, 3, ref * 1.03))]
    s = M.summarize(rows)[1.0]
    jit = [ref * 1.00, ref * 1.03]
    mean = sum(jit) / 2
    sigma = math.sqrt(sum((v - mean) ** 2 for v in jit) / (len(jit) - 1))
    assert s["runout_mean"] == pytest.approx(mean, rel=1e-12)
    assert s["runout_sigma"] == pytest.approx(sigma, rel=1e-12)
    assert s["n_in_band"] == 3 and s["all_in_band"] is True
    assert s["jitter_shift"] == pytest.approx(abs(ref * 1.03 / (ref * 0.97) - 1.0),
                                              rel=1e-9)
    assert s["spread_ok"] is True


def test_summary_flags_an_out_of_band_member():
    ref = M.lube_2004_runout_ratio(0.5)
    rows = [M.reduce_case(_raw(0.5, 0.0, 1, ref)),
            M.reduce_case(_raw(0.5, 0.05, 2, ref * 1.25))]
    s = M.summarize(rows)[0.5]
    assert s["all_in_band"] is False and s["n_in_band"] == 1
    v = M.hypothesis_verdicts({0.5: s}, {"all_match": True})
    assert v["H_BAND"]["pass"] is False


def test_hypothesis_verdicts_pass_on_a_clean_ensemble():
    law = M.lube_2004_runout_ratio
    rows = []
    for a in (0.5, 1.0, 2.0):
        ref = law(a)
        t_ref = M.lube_2004_emplacement_time(a, 8.0 * 2.0 * 0.03, 9.81)
        t99 = t_ref * (1.45 if a == 0.5 else 1.05)   # shallow column stays late
        rows.append(M.reduce_case(_raw(a, 0.0, 1, ref * 0.99, t99=t99)))
        rows.append(M.reduce_case(_raw(a, 0.05, 2, ref * 1.00, t99=t99)))
        rows.append(M.reduce_case(_raw(a, 0.10, 3, ref * 1.01, t99=t99)))
    summary = M.summarize(rows)
    v = M.hypothesis_verdicts(summary, {"all_match": True})
    assert v["H_SPREAD"]["pass"] is True
    assert v["H_BAND"]["pass"] is True
    assert v["H_JITTER"]["pass"] is True
    assert v["H_SIGN"]["pass"] is True


def test_h_sign_fails_when_the_shallow_time_overshoot_is_gone():
    law = M.lube_2004_runout_ratio
    a = 0.5
    ref = law(a)
    t_ref = M.lube_2004_emplacement_time(a, 8.0 * 2.0 * 0.03, 9.81)
    rows = [M.reduce_case(_raw(a, 0.0, 1, ref, t99=t_ref * 1.30)),
            M.reduce_case(_raw(a, 0.10, 2, ref, t99=t_ref * 1.10))]  # inside band
    v = M.hypothesis_verdicts(M.summarize(rows), {"all_match": True})
    assert v["H_SIGN"]["pass"] is False


def test_h_spread_fails_on_a_wide_ensemble():
    law = M.lube_2004_runout_ratio
    ref = law(2.0)
    rows = [M.reduce_case(_raw(2.0, 0.05, 1, ref * 0.94)),
            M.reduce_case(_raw(2.0, 0.05, 2, ref * 1.06))]
    v = M.hypothesis_verdicts(M.summarize(rows), {"all_match": True})
    assert v["H_SPREAD"]["pass"] is False


def test_repro_check_matches_and_mismatches(tmp_path):
    law = M.lube_2004_runout_ratio
    archive = tmp_path / "archive.csv"
    ref = law(1.0)
    with open(archive, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "sha256", "a_nominal",
                                           "grains_per_radius", "dt_scale",
                                           "mu_wall", "mu", "mu_rolling",
                                           "stored_runout_ratio"])
        w.writeheader()
        w.writerow({"case": "w5b_confirm_wall0.8_a1", "sha256": "deadbeef",
                    "a_nominal": 1.0, "grains_per_radius": 8.0, "dt_scale": 1.0,
                    "mu_wall": 0.8, "mu": 0.5, "mu_rolling": 0.5,
                    "stored_runout_ratio": ref})
    good = M.repro_check([M.reduce_case(_raw(1.0, 0.0, 1, ref))], str(archive))
    assert good["all_match"] is True
    assert good["used"][0]["archived_case"] == "w5b_confirm_wall0.8_a1"
    assert good["archive_sha256"]
    bad = M.repro_check([M.reduce_case(_raw(1.0, 0.0, 1, ref * 1.02))], str(archive))
    assert bad["all_match"] is False
    missing = M.repro_check([M.reduce_case(_raw(1.0, 0.0, 1, ref))],
                            str(tmp_path / "nope.csv"))
    assert missing["available"] is False and missing["all_match"] is None


def test_benchmark_declares_fifteen_unique_members_at_the_closing_arm():
    B = _load(BENCH, "m19922_ensemble_bench")
    cases = B.ensemble_cases()
    assert len(cases) == 15
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == 15
    assert {c["aspect"] for c in cases} == {0.5, 1.0, 2.0}
    arm = B.CLOSING_ARM
    assert (arm["mu"], arm["mu_rolling"], arm["mu_wall"]) == (0.5, 0.5, 0.8)
    for i, c in enumerate(cases):
        assert c["params"]["aspect_ratio"] == c["aspect"]
        assert c["params"]["jitter"] == c["jitter"]
        assert c["params"]["seed"] == c["seed"]
        assert i % 5 < 5
    # one deterministic lattice control per aspect, plus four disordered members
    for a in (0.5, 1.0, 2.0):
        grp = [c for c in cases if c["aspect"] == a]
        assert sum(1 for c in grp if c["jitter"] == 0.0) == 1
        assert sum(1 for c in grp if c["jitter"] > 0.0) == 4
        assert max(c["jitter"] for c in grp) == pytest.approx(0.10)


def test_sharding_partitions_every_case_exactly_once():
    B = _load(BENCH, "m19922_ensemble_bench")
    cases = B.ensemble_cases()
    for nshards in (1, 3, 5, 15):
        seen = []
        for k in range(nshards):
            seen += [c["id"] for i, c in enumerate(cases) if i % nshards == k]
        assert sorted(seen) == sorted(c["id"] for c in cases)


# ----------------------------- data regression ------------------------------ #
pytestmark_bundle = pytest.mark.skipif(
    not os.path.exists(CASES_CSV),
    reason="committed ensemble bundle not built yet")


@pytestmark_bundle
def test_committed_bundle_is_complete_and_in_band():
    with open(CASES_CSV) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 15
    assert {round(float(r["aspect_nominal"]), 6) for r in rows} == {0.5, 1.0, 2.0}
    for a in ("0.5", "1.0", "2.0"):
        grp = [r for r in rows if round(float(r["aspect_nominal"]), 6) == float(a)]
        assert len(grp) == 5
        assert sum(1 for r in grp if float(r["jitter"]) == 0.0) == 1
    assert all(r["stop_reason"] == "at_rest" for r in rows)
    # every reference value is the Lube law at the nominal aspect ratio
    for r in rows:
        assert float(r["ref_runout_ratio"]) == pytest.approx(
            M.lube_2004_runout_ratio(float(r["aspect_nominal"])), abs=1e-12)
        assert float(r["err_runout"]) == pytest.approx(
            float(r["runout_ratio_p995"]) / float(r["ref_runout_ratio"]) - 1.0,
            abs=1e-12)


@pytestmark_bundle
def test_committed_summary_reproduces_its_own_verdicts():
    with open(BUNDLE) as fh:
        bundle = json.load(fh)
    assert set(bundle["verdicts"]) == {"H_REPRO", "H_SPREAD", "H_BAND",
                                      "H_SIGN", "H_JITTER"}
    assert bundle["n_cases"] == 15
    summary = {float(k): v for k, v in bundle["summary"].items()}
    recomputed = M.hypothesis_verdicts(summary, bundle["repro"])
    for h, v in recomputed.items():
        assert v["pass"] == bundle["verdicts"][h]["pass"], h
    # manifest hashes the csv, the summary and every figure it lists
    with open(os.path.join(ART, "manifest_sha256.json")) as fh:
        manifest = json.load(fh)
    for rel in ["docs/artifacts/column_collapse_ensemble_m19922/cases_ensemble.csv",
                "docs/artifacts/column_collapse_ensemble_m19922/summary.json"]:
        assert rel in manifest