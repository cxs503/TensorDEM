"""Machine check of the committed round-3 repose-metrology evidence (H1 C2).

The round-3 document (``docs/H1_BROKEN_ICE_DEM_ROUND3.md`` §1) rests on a table
of *recovered* cone angles for 13 known slopes.  The generator that produced it
(``benchmarks/h1_broken_ice_dem/make_figures_round3.py``) writes into
``artifacts/``, which is not part of the tree, so from a checkout the table was
unverifiable and its numbers could drift from the code without anything going
red.  The evidence is now committed under
``runs/h1_broken_ice_dem/round3/round3_figures.json`` (regenerate with
``PYTHONPATH=src python scripts/render_h1_round3_evidence.py``) and this module
holds it to three claims:

* the *pre-fix* BIC break search really was 4-5 deg wrong on clean cones
  (baseline, so the fix below is falsifiable rather than decorative);
* the admissibility rule ("an apron cannot be steeper than the flank") brings
  that worst-case error under 1.5 deg **without touching** the apron columns
  that round 3 was built to measure (bit-identical there);
* the committed numbers are still what the current estimator produces
  (recomputed here on the same fixed-seed profiles), so stale evidence fails.
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EVIDENCE = os.path.join(ROOT, "runs", "h1_broken_ice_dem", "round3", "round3_figures.json")
BENCH = os.path.join(ROOT, "benchmarks", "h1_broken_ice_dem", "make_figures_round3.py")

TRUTH = [15.0 + 2.5 * i for i in range(13)]  # 15 .. 45 deg, the document's sweep


def _evidence():
    if not os.path.exists(EVIDENCE):
        pytest.fail("round-3 evidence missing: %s (run scripts/render_h1_round3_evidence.py)"
                    % os.path.relpath(EVIDENCE, ROOT))
    with open(EVIDENCE, encoding="utf-8") as fh:
        return json.load(fh)


def _bench():
    spec = importlib.util.spec_from_file_location("h1_round3_bench_test", BENCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _worst(values, truth=TRUTH):
    err = np.abs(np.asarray(values, dtype=np.float64) - np.asarray(truth))
    return float(np.nanmax(err)) if np.any(np.isfinite(err)) else float("nan")


def test_evidence_covers_the_documented_sweep():
    data = _evidence()
    fig = data["fig1"]
    assert [float(v) for v in fig["truth_deg"]] == pytest.approx(TRUTH)
    for key in ("naive_clean", "naive_apron", "flank_clean", "flank_apron"):
        assert len(fig[key]) == len(TRUTH), key
    assert "provenance" in data and data["provenance"]["generator"].endswith("make_figures_round3.py")


def test_pre_fix_split_was_multi_degree_wrong_on_clean_cones():
    """Baseline: the unconstrained BIC break loses > 4 deg on a *clean* cone.

    This is the number the fix has to move; without it the improvement claim
    would have no contrast.
    """
    abl = _evidence()["fig1_pre_fix_ablation"]
    worst = _worst(abl["flank_clean"])
    assert worst >= 4.0, "pre-fix baseline weakened: worst = %.2f deg" % worst
    # and it refused 9 of the 13 clean cones outright (relief gate on the short
    # apex segment it carved out) -- a clean cone is never "no pile"
    nan_cells = int(np.sum(~np.isfinite(np.asarray(abl["flank_clean"], dtype=float))))
    assert nan_cells >= 6
    # a clean 40 deg cone came back as 35.10 deg: the split put the *steeper*
    # segment outboard and reported the shallow inner slope
    assert abl["flank_clean"][TRUTH.index(40.0)] == pytest.approx(35.097, abs=0.05)


def test_admissibility_rule_fixes_clean_cones():
    """Post-fix: every clean cone is recovered to better than 1.5 deg."""
    fig = _evidence()["fig1"]
    worst = _worst(fig["flank_clean"])
    assert worst <= 1.5, "clean-cone worst error = %.2f deg" % worst
    # no refusal on a resolved cone: the pre-fix split left a short apex segment
    # whose relief was under one particle, so 9 of these 13 cells returned nan
    assert all(np.isfinite(fig["flank_clean"])), fig["flank_clean"]
    for angle in (15.0, 30.0, 40.0, 45.0):
        assert fig["flank_clean"][TRUTH.index(angle)] == pytest.approx(angle, abs=1.5)


def test_apron_columns_are_untouched_by_the_fix():
    """The rule must not buy clean-cone accuracy by moving the apron cases.

    Round 3 exists to measure piles *with* a foot; if the admissibility rule
    changed those numbers it would be re-tuning, not fixing.
    """
    data = _evidence()
    for key in ("flank_apron", "flank_clean"):
        if key == "flank_clean":
            # the clean column is exactly what the rule is meant to change
            continue
        a = np.asarray(data["fig1"][key], dtype=np.float64)
        b = np.asarray(data["fig1_pre_fix_ablation"][key], dtype=np.float64)
        assert np.allclose(a, b, rtol=0.0, atol=1e-12), key
    assert _worst(data["fig1"]["flank_apron"]) <= 3.0


def test_document_table_is_backed_by_the_evidence():
    """docs/H1_BROKEN_ICE_DEM_ROUND3.md §1 rows, checked against the JSON."""
    fig = _evidence()["fig1"]
    rows = {  # truth: (naive_apron, flank_apron) as quoted in the document
        30.0: (24.09, 28.05),
        40.0: (33.02, 37.75),
        45.0: (37.76, 42.70),
    }
    for angle, (naive_apron, flank_apron) in rows.items():
        i = TRUTH.index(angle)
        assert fig["naive_apron"][i] == pytest.approx(naive_apron, abs=0.01)
        assert fig["flank_apron"][i] == pytest.approx(flank_apron, abs=0.01)


def test_committed_numbers_match_a_live_recomputation():
    """Evidence freshness: the JSON must be what today's estimator produces."""
    from tensordem.polyhedron_floe3d.pile_surface import fit_flank

    bench = _bench()
    fig = _evidence()["fig1"]
    for angle in (20.0, 40.0):
        i = TRUTH.index(angle)
        for share, key in ((0.0, "flank_clean"), (0.4, "flank_apron")):
            prof = bench.synthetic(angle, 400, share)
            live = fit_flank(prof["radius"], prof["height"], prof["count"],
                             min_drop=0.05)["angle_deg"]
            committed = fig[key][i]
            if number_is_nan(committed):
                assert number_is_nan(live), (key, angle, live)
            else:
                assert live == pytest.approx(committed, abs=1e-6), (key, angle)


def number_is_nan(value) -> bool:
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False