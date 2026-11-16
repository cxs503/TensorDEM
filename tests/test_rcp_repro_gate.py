"""Invariants of the RCP audit reproducibility gate (``repro_gate.py``).

Anchor: ``CASE-DEM-RCP-PACKING-FRACTION``.

The gate is the tool that decides whether a committed audit artefact can be
regenerated from committed code + data.  Its own failure modes are therefore
worth testing: a differ that misses a change certifies a stale artefact, and a
differ that over-reports turns every numeric tweak into a false alarm.  These
tests pin the discriminating behaviour on synthetic payloads (no DEM runs, no
subprocesses), plus the two defects this mission fixed:

* ``phi_definition_audit.parse_tag`` used to abort on any new run-tag token
  (``mur0``), which made the whole measurement-operator audit unrunnable;
* ``stiffness_seed_audit`` used to store absolute figure paths, which baked a
  foreign checkout's location into a committed artefact.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "benchmarks", "case_rcp_deposition")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(BENCH, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rg = _load("repro_gate")
pda = _load("phi_definition_audit")


# ---------------------------------------------------------------------------
# leaf differ
# ---------------------------------------------------------------------------
def test_identical_payloads_have_no_records():
    a = {"phi": 0.6366, "band": [0.62, 0.66], "nested": {"x": 1}}
    d = rg.compare_leaves(rg.flatten(a), rg.flatten(a))
    assert d["records"] == []
    assert d["counts"]["n_drift"] == 0
    assert d["counts"]["n_leaves"] == d["counts"]["n_same"]


def test_tolerance_is_relative_and_tight():
    ref = rg.flatten({"phi": 0.6366})
    cur = rg.flatten({"phi": 0.6366 * (1 + 1e-12)})
    d = rg.compare_leaves(ref, cur)
    assert d["counts"]["n_drift"] == 0
    assert all(r["status"] == "WITHIN_TOL" for r in d["records"])
    cur = rg.flatten({"phi": 0.6366 * (1 + 1e-6)})
    rec = rg.compare_leaves(ref, cur)["records"]
    assert len(rec) == 1 and rec[0]["status"] == "DRIFT"


def test_absolute_tolerance_dominates_near_zero():
    # A relative-only tolerance would flag any change from exactly zero; the
    # gate must not, because zero-initialised leaves are common in these files.
    ref = rg.flatten({"x": 0.0})
    cur = rg.flatten({"x": 5e-13})
    assert rg.compare_leaves(ref, cur)["counts"]["n_drift"] == 0
    cur = rg.flatten({"x": 1e-6})
    assert rg.compare_leaves(ref, cur)["records"][0]["status"] == "DRIFT"


def test_nan_equals_nan():
    # ``nan != nan`` would otherwise report a permanent phantom drift.
    ref = rg.flatten({"margin": float("nan")})
    cur = rg.flatten({"margin": float("nan")})
    assert rg.compare_leaves(ref, cur)["records"] == []
    d = rg.compare_leaves(ref, rg.flatten({"margin": 1.0}))
    assert d["records"][0]["status"] == "DRIFT"


def test_missing_and_type_change_are_separate_classes():
    ref = rg.flatten({"a": 1.0, "b": 2.0})
    cur = rg.flatten({"a": "1.0"})
    st = {r["path"]: r["status"] for r in rg.compare_leaves(ref, cur)["records"]}
    assert st["/a"] == "TYPE_CHANGE"
    assert st["/b"] == "MISSING_CUR"


# ---------------------------------------------------------------------------
# identity-aware alignment
# ---------------------------------------------------------------------------
def test_gates_are_keyed_by_id_not_by_index():
    ref = {"gates": [{"id": "G1", "pass": True}, {"id": "G2", "pass": True}]}
    cur = {"gates": [{"id": "G0", "pass": True}, {"id": "G1", "pass": True},
                     {"id": "G2", "pass": True}]}
    d = rg.compare_leaves(rg.flatten(rg.canonicalise(ref)),
                          rg.flatten(rg.canonicalise(cur)))
    # inserting a gate must not report every later gate as changed
    assert d["counts"]["n_drift"] == 0
    assert d["counts"]["n_value_change"] == 0
    assert d["counts"]["n_missing"] == 2        # the new gate's two leaves


def test_duplicate_identities_get_occurrence_keys():
    rec = rg.canonicalise([{"name": "x", "v": 1}, {"name": "x", "v": 2}])
    assert set(rec) == {"name=x#0", "name=x#1"}


def test_composite_identity_for_parameter_rows():
    rows = [{"n": 10000, "mu": 0.0, "kn": 4000.0, "phi": 0.64},
            {"n": 10000, "mu": 0.1, "kn": 4000.0, "phi": 0.62}]
    keyed = rg.canonicalise(rows)
    assert len(keyed) == 2 and all(k.startswith("n=") for k in keyed)


# ---------------------------------------------------------------------------
# categorical verdicts
# ---------------------------------------------------------------------------
def test_verdict_flip_is_detected():
    ref = rg.flatten(rg.canonicalise({"gates": [{"id": "G13", "pass": False}]}))
    cur = rg.flatten(rg.canonicalise({"gates": [{"id": "G13", "pass": True}]}))
    v = rg.compare_verdicts(ref, cur)
    assert v["preserved"] is False and v["changed"][0]["path"].endswith("/pass")


def test_numeric_centre_move_alone_does_not_change_verdicts():
    ref = rg.flatten(rg.canonicalise({"gates": [{"id": "G13", "pass": True,
                                                 "got": 0.001}]}))
    cur = rg.flatten(rg.canonicalise({"gates": [{"id": "G13", "pass": True,
                                                 "got": 0.002}]}))
    assert rg.compare_verdicts(ref, cur)["preserved"] is True


# ---------------------------------------------------------------------------
# foreign-path detection
# ---------------------------------------------------------------------------
def test_foreign_absolute_path_is_flagged_and_relative_is_not():
    flat = rg.flatten({"figures": [
        "benchmarks/case_rcp_deposition/figures/a.png",
        "/root/tp-agent/work/m22796/benchmarks/case_rcp_deposition/figures/b.png",
        os.path.join(rg.ROOT, "x.png"),
    ]})
    hits = rg.find_foreign_paths(flat)
    assert len(hits) == 1 and "m22796" in hits[0]["value"]


def test_markdown_path_scan():
    txt = ("see `benchmarks/fig.png` and /root/tp-agent/work/m22796/fig.png\n"
           f"{rg.ROOT}/local.png\n")
    hits = rg.scan_text_for_paths(txt)
    assert len(hits) == 1 and hits[0]["line"] == 1


# ---------------------------------------------------------------------------
# restore guard
# ---------------------------------------------------------------------------
def test_restore_never_leaves_root(tmp_path):
    res = rg.restore_paths(["../outside.txt", "/etc/passwd"], protect=())
    assert res["restored"] == []
    assert len(res["refused"]) == 2


# ---------------------------------------------------------------------------
# registry sanity
# ---------------------------------------------------------------------------
def test_registry_scripts_and_artefacts_exist():
    for e in rg.REGISTRY:
        assert os.path.exists(os.path.join(BENCH, e["script"])), e["script"]
        assert os.path.exists(os.path.join(BENCH, e["json"])), e["json"]
        for d in e["doc"]:
            assert os.path.exists(os.path.join(ROOT, d)), d


def test_summary_flags_blocking_statuses():
    recs = [dict(verdict="REPRODUCED"), dict(verdict="STALE")]
    s = rg.summarise(recs)
    assert s["overall"] == "REVIEW" and s["blocking"] == 1
    assert rg.summarise([dict(verdict="REPRODUCED")])["overall"] == "PASS"


# ---------------------------------------------------------------------------
# regressions fixed by this mission
# ---------------------------------------------------------------------------
def test_parse_tag_survives_rolling_friction_token():
    p = pda.parse_tag("deposition_n2000_mu0.1_mur0_kn4000_e0.1_seed1")
    assert p["mu"] == pytest.approx(0.1)
    assert p["mu_roll"] == pytest.approx(0.0)     # not "mu" = "r0"
    assert p["kn"] == pytest.approx(4000.0)
    assert p["n"] == 2000 and p["seed"] == 1


def test_parse_tag_production_and_dtf():
    p = pda.parse_tag("deposition_n10000_mu0_kn64000_e0.1_seed2_dtf0.025")
    assert p["dt_frac"] == pytest.approx(0.025)
    assert p["mu"] == 0.0
    assert pda.parse_tag("deposition_n10000_mu0_kn4000_e0.1_seed1")["dt_frac"] \
        == pytest.approx(1.0 / 20.0)


def test_parse_tag_ignores_unknown_tokens():
    p = pda.parse_tag("weird_x1")
    assert p == {"dt_frac": pytest.approx(1.0 / 20.0)}


def test_stiffness_audit_figure_paths_are_relative():
    """The committed artefact must not encode a checkout location."""
    import json
    path = os.path.join(BENCH, "runs", "stiffness_seed_audit.json")
    payload = json.load(open(path))
    for f in payload.get("figures", []):
        assert not os.path.isabs(f), f