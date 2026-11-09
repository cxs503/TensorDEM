"""Tests for the ice-polygon-floe-field external-anchor artifact (W18 paper-exp #1).

Two things are tested, and they are different:

* the **operator layer** — dimension algebra, the average↔local pressure
  identity, exponent recovery, monotonicity, aggregation routes and the
  anchor-free ratios: these must pass *without* the standard's text;
* the **discipline layer** — that a value can only enter through a VERIFIED
  registry slot, that no PASS criterion may rest on an unverified anchor, and
  that the honest-FAIL state of the shipped artifact is the state it should be.

The "flip" test uses a *labelled synthetic fixture*: it proves the harness can
turn C1/C2 into PASS once real transcriptions are injected.  The fixture is not
ISO text and never reaches the artifact.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANCHORS = os.path.join(REPO, "benchmarks", "anchors")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, ANCHORS)

import polygon_floe_anchor as op  # noqa: E402

ARTIFACT = os.path.join(ANCHORS, "ice-polygon-floe-field.json")
N = 0.4284


@pytest.fixture(scope="module")
def doc():
    with open(ARTIFACT, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# operator layer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0.2, N, 0.7])
def test_field_load_is_dimensionally_homogeneous(n):
    assert op.dim_of_field_load(n) == op.DIM_FORCE
    assert op.dim_of_pressure(n) == op.DIM_PRESSURE


def test_kernel_carries_a_unit_so_absolute_newtons_need_a_ref():
    k = op.dim_of_field_kernel(N)
    assert k != op.DIMLESS, "K(n) is not dimensionless for n != 1"
    trap = op.dimensional_checks(N, a_ref_exponent=0.0)
    row = [r for r in trap["rows"] if r["check"].startswith("A_ref 归一化")][0]
    assert row["resulting_dims"] != {kk: str(v) for kk, v in op.DIM_FORCE.items()}


def test_dimensional_checks_all_pass():
    res = op.dimensional_checks(N)
    assert res["pass"], [r for r in res["rows"] if not r["pass"]]
    assert len(res["rows"]) >= 6


@pytest.mark.parametrize("area", [0.05, 0.5, 1.0, 5.0, 88.8])
def test_average_pressure_identity(area):
    r = op.integral_identity_residual(area, p0=3.1, a_ref=2.7, n=N)
    assert r < 1e-10


def test_local_pressure_rejects_admissible_branch_violation():
    with pytest.raises(ValueError):
        op.contact_pressure(np.array([1.0]), p0=1.0, a_ref=1.0, n=1.5)


def test_log_log_slope_recovers_the_exponent():
    a = np.logspace(-2, 2, 41)
    p = op.average_pressure(a, p0=2.0, a_ref=0.7, n=N)
    slope, resid = op.log_log_slope(a, p)
    assert abs(slope + N) < 1e-10 and resid < 1e-12


def test_scale_invariance():
    got, exp = op.scale_invariance_ratio(p0=1.0, a_ref=1.0, n=N, area=3.0, lam=10.0)
    assert abs(got - exp) <= 1e-12 * abs(exp)


def test_consistency_passes_and_negative_control_fails(monkeypatch):
    areas = [1.0, 2.0, 5.0, 10.0]
    ok = op.consistency_checks(n=N, p0=1.0, a_ref=1.0, areas=areas)
    assert ok["pass"]

    # negative control 1: an inadmissible exponent (0 < n < 1 is required by the
    # average<->local pressure identity) must be refused, not silently accepted
    for bad_n in (-0.3, 0.0, 1.2):
        with pytest.raises(ValueError):
            op.consistency_checks(n=bad_n, p0=1.0, a_ref=1.0, areas=areas)

    # negative control 2: swap the law for a constant pressure — the monotonicity
    # and exponent-recovery rows must go red, i.e. the checker really checks
    monkeypatch.setattr(op, "average_pressure",
                        lambda a, *, p0, a_ref, n: np.full_like(np.asarray(a, dtype=float), p0))
    bad = op.consistency_checks(n=N, p0=1.0, a_ref=1.0, areas=areas)
    assert not bad["pass"]
    assert not all(r["pass"] for r in bad["rows"])


def test_aggregate_two_routes_agree():
    areas = [0.8, 1.5, 3.0, 4.2]
    direct = float(np.sum(op.floe_force(np.array(areas), p0=1.0, a_ref=1.0, n=N)))
    via = 1.0 * 1.0 ** N * op.field_kernel(areas, N)
    assert abs(direct - via) <= 1e-9 * abs(direct)


def test_kernel_rejects_nonpositive_area():
    with pytest.raises(ValueError):
        op.field_kernel([1.0, 0.0], N)


def test_patch_area_caliber_uses_equivalent_radius():
    a = op.floe_patch_areas([5.0], 0.5, kappa=0.5)
    assert a[0] == pytest.approx(2 * 0.5 * 5.0 * 0.5)


# --------------------------------------------------------------------------- #
# artifact content, recomputed from code (no hand-typed numbers)
# --------------------------------------------------------------------------- #
def test_artifact_declares_the_four_criteria_with_gaps(doc):
    assert [c["id"] for c in doc["criteria"]] == ["C1", "C2", "C3", "C4"]
    for c in doc["criteria"]:
        assert c["criterion_pre_registered"] is True
        assert "status" in c and "measured" in c and "value_source_id" in c
        if str(c["status"]).split()[0] != "PASS":
            assert c["gap"], f"{c['id']} 未达标却无 gap 说明"


def test_verdict_follows_c1(doc):
    c1 = str(doc["criteria"][0]["status"]).split()[0]
    assert doc["verdict"] == ("FAIL" if c1 != "PASS" else doc["verdict"])
    assert doc["verdict"] in {"PASS", "FAIL", "PARTIAL"}
    assert doc["honesty"]["no_compliance_claim"] is True
    assert doc["honesty"]["memory_or_secondhand_values_used_as_anchor"] is False


def test_anchor_slots_are_withheld_with_reason(doc):
    assert len(doc["sources"]) == 2
    for s in doc["sources"]:
        assert s["transcription_status"] == "WITHHELD"
        assert s["quote_verbatim"] is None and s["table_or_page"] is None
        assert s["value"] is None and s["withheld_reason"]


def test_no_absolute_number_is_claimed(doc):
    conv = doc["conversion_chain"]
    assert conv["absolute_layer_status"] == "WITHHELD"
    assert conv["normalization"]["status"] == "unit_normalization_only"
    assert doc["criteria"][1]["measured"]["absolute_layer"] == "WITHHELD"


def test_stored_samples_recompute(doc):
    conv = doc["conversion_chain"]
    a = np.asarray(conv["samples"]["A_m2"], dtype=float)
    n = float(conv["exponent_used"])
    p = op.average_pressure(a, p0=1.0, a_ref=1.0, n=n)
    assert np.max(np.abs(p - np.asarray(conv["samples"]["p_norm_Pa"]))) <= 1e-12
    assert doc["recompute"]["residual_max"] < 1e-9
    assert abs(doc["recompute"]["n_recovered_from_stored_samples"] - n) <= 1e-9 * n


def test_kernels_and_ratios_recompute(doc):
    conv = doc["conversion_chain"]
    kernels = conv["calibers"]["floe_patch"]["kernel_over_fields"]
    fields = doc["case_caliber"]["fields"]
    for key, rec in kernels.items():
        areas = fields[key]["patch_area_mean_m2"]      # mean only: ratio route below
        assert op.field_kernel([areas], N) == pytest.approx(areas ** (1 - N), rel=1e-12)
        assert rec["kernel_K_n_m2_1mn"] > 0.0
    base = "PowerLawFSD(2.5)|phi=0.85"
    for label, rec in conv["anchor_free_ratios"].items():
        other = label.split(" / ")[0]
        expect = (kernels[other]["kernel_K_n_m2_1mn"] / kernels[base]["kernel_K_n_m2_1mn"])
        assert rec["ratio"] == pytest.approx(expect, rel=1e-12)
        assert len(rec["ratio_over_n_grid"]) >= 5


def test_caliber_degeneracy_is_recorded(doc):
    rows = doc["checks"]["consistency"]["rows"]
    row = [r for r in rows if r["check"].startswith("口径退化机检")][0]
    assert row["pass"] and row["residual_max"] < 1e-12
    assert any("口径退化" in g for g in doc["gaps"])
    for fam in ("PowerLawFSD(2.5)", "BimodalFSD()"):
        vals = [doc["case_caliber"]["fields"][k]["patch_area_mean_m2"]
                for k in doc["case_caliber"]["fields"] if k.startswith(fam + "|")]
        assert len(vals) == 3
        assert max(abs(v - vals[0]) / vals[0] for v in vals) < 1e-12, (
            f"{fam}: concentration must not change the floe patches")


def test_candidate_values_are_comparison_only(doc):
    cands = doc["candidates_unverified"]
    assert cands and all(c["verified_against_primary"] is False for c in cands)
    assert all(c["usage"] == "comparison_only" for c in cands)
    assert doc["criteria"][0]["value_source_id"] is None


def test_source_access_probe_is_recorded(doc):
    probe = doc["source_access_probe"]
    methods = {r["method"] for r in probe}
    assert {"tcp_connect", "http_get", "filesystem_inventory"} <= methods
    inv = [r for r in probe if r["method"] == "filesystem_inventory"][0]
    assert inv["primary_text_hits"] == 0
    assert all(h["kind"] != "possible_copy" for h in inv["hits"])


def test_registry_discipline_catches_an_unverified_pass():
    registry = op.load_registry(None)
    good = op.registry_discipline(registry, [
        {"id": "C3", "status": "PASS", "requires_anchor_value": False, "measured": {}}])
    assert good["pass"]
    bad = op.registry_discipline(registry, [
        {"id": "C1", "status": "PASS", "requires_anchor_value": True,
         "value_source_id": "iso19906-2019-AnnexA-pA", "measured": {}}])
    assert not bad["pass"] and bad["violations"]


# --------------------------------------------------------------------------- #
# the injection path (labelled synthetic fixture — proves C1/C2 can flip)
# --------------------------------------------------------------------------- #
def _fixture(tmp_path, *, complete=True):
    text = ("TEST FIXTURE — synthetic string used only to exercise the registry "
            "validator; this is not an ISO quotation and carries no value.")
    entry = {"quote_verbatim": text, "table_or_page": "synthetic-table-A1",
             "value": {"n": 0.5, "p0": 1.0, "A_ref": 1.0},
             "transcription_status": "VERIFIED"}
    if not complete:
        entry.pop("table_or_page")
    payload = {"sources": [dict(entry, id="iso19906-2019-AnnexA-pA"),
                           dict(entry, id="iso19906-2019-Clause14-15-global")]}
    path = os.path.join(tmp_path, "fixture.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def test_partial_transcription_cannot_be_promoted(tmp_path):
    with pytest.raises(ValueError):
        op.load_registry(_fixture(tmp_path, complete=False))


def test_verified_injection_flips_c1_and_c2(tmp_path):
    registry = op.load_registry(_fixture(tmp_path))
    checks = {"dimensional": op.dimensional_checks(N),
              "consistency": op.consistency_checks(n=N, p0=1.0, a_ref=1.0,
                                                   areas=[1.0, 2.0, 3.0])}
    checks["registry_discipline"] = {"pass": True, "violations": [],
                                     "verified_slots": [s["id"] for s in registry]}
    crit = op.evaluate_criteria(registry, recompute_residual=1e-15,
                                absolute_loads={"structure_face_N": 1.0}, checks=checks)
    assert str(crit[0]["status"]).split()[0] == "PASS"
    assert str(crit[1]["status"]).split()[0] == "PASS"
    checks["registry_discipline"] = op.registry_discipline(registry, crit)
    assert checks["registry_discipline"]["pass"]
    assert op.overall_verdict(crit)[0] == "PASS"


def test_shipped_artifact_is_the_honest_fail_state(doc):
    """The shipped artifact must be the WITHHELD state — no silent promotion."""
    assert doc["verdict"] == "FAIL"
    assert doc["criteria"][0]["status"] == "FAIL"
    assert doc["honesty"]["absolute_layer_declared"] is False
    assert doc["honesty"]["hand_typed_numbers"] is False
    disc = doc["checks"]["registry_discipline"]
    assert disc["pass"] and disc["verified_slots"] == []