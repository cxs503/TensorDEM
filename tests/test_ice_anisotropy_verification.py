"""Acceptance-criteria tests for the columnar anisotropy study (node p474w1-n08).

These tests pin the *numbers* the node is judged on:

(a) column/transverse strength ratio inside the literature band 3-5
(b) 45 deg strength error against the slip-system reference <= 10 %
(c) uniform-ODF degeneration: strength ratio -> 1 (<= 5 %), no directional trend
(d) fabric ledger record/query/round-trip
(f) feature-ledger linkage record present and pointing at the right feature
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensordem.ice_anisotropy import FabricODF, run_verification
from tensordem.ice_anisotropy.verification import default_angles
from tensordem.ice_anisotropy.yield_surface import COLUMNAR_RATIO_RANGE

REPO_ROOT = Path(__file__).resolve().parents[1]
SIGMA_PERP = 3.0e6


@pytest.fixture(scope="module")
def study() -> dict:
    return run_verification(sigma_perp=SIGMA_PERP, ratio_columnar=3.5)


def _criterion(study: dict, cid: str) -> dict:
    for c in study["criteria"]:
        if c["id"] == cid:
            return c
    raise KeyError(cid)


def test_all_acceptance_criteria_pass(study):
    missing = {"a", "b", "c", "d"} - {c["id"] for c in study["criteria"]}
    assert not missing
    for c in study["criteria"]:
        assert c["pass"], f"criterion {c['id']} failed: {c}"
        assert c["statement"] and "threshold" in c


def test_criterion_a_ratio_inside_literature_band(study):
    c = _criterion(study, "a")
    lo, hi = COLUMNAR_RATIO_RANGE
    assert lo <= c["value"] <= hi
    assert lo <= c["value_slip_reference"] <= hi
    assert c["value"] == pytest.approx(3.5, rel=1e-8)
    assert c["value_slip_reference"] == pytest.approx(3.5, rel=1e-8)


def test_criterion_b_45deg_error_within_ten_percent(study):
    c = _criterion(study, "b")
    assert abs(c["relative_error"]) <= 0.10
    assert c["value_over_perp_continuum"] == pytest.approx(0.894427, rel=1e-6)
    assert c["value_over_perp_reference"] == pytest.approx(0.871176, rel=1e-5)
    # the Sachs bound brackets the quadratic prediction from above
    assert c["value_over_perp_continuum"] < c["reference_bound_sachs"]


def test_criterion_c_uniform_fabric_degenerates(study):
    c = _criterion(study, "c")
    iso = study["isotropic_control"]
    assert c["pass"]
    assert iso["strength_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert iso["ratio_error"] <= 0.05
    assert iso["max_directional_strength_variation_slip_dense"] <= 0.05
    # and the mean-field slip reference lands near the von Mises shear value
    assert iso["tau_ratio_error"] <= 0.05


def test_criterion_d_ledger_roundtrip(study):
    c = _criterion(study, "d")
    assert c["pass"]
    assert c["roundtrip_anisotropy_index_error"] == pytest.approx(0.0, abs=1e-12)
    assert {"odf-columnar-S2", "odf-uniform"} <= set(c["sample_ids"])


def test_strength_curve_is_bracketed_by_the_reference_bounds(study):
    """Continuum vs slip reference over the whole sweep, not just 45 deg."""
    rows = study["comparison"]
    assert [r["theta_deg"] for r in rows] == [float(a) for a in default_angles()]
    errs = {r["theta_deg"]: r["relative_error"] for r in rows}
    assert abs(errs[45.0]) <= 0.10
    # the largest deviation sits in the steep 10-20 deg shoulder; it is reported
    # rather than hidden (see the report's error table)
    assert max(abs(v) for v in errs.values()) < 0.25
    # the quadratic surface under-predicts on the steep shoulder and slightly
    # over-predicts around the 45-60 deg dip; the endpoints are matched exactly
    assert errs[10.0] < 0.0
    assert errs[45.0] > 0.0
    assert abs(errs[0.0]) < 1e-9 and abs(errs[90.0]) < 1e-9


def test_report_declares_its_limitations(study):
    text = " ".join(study["limitations"]).lower()
    for keyword in ("slip-system reference", "mean-field", "pressure-insensitive"):
        assert keyword in text


def test_feature_ledger_link_record():
    """Criterion (f): the capability is linked to the H1-BROKEN-ICE-DEM feature."""
    path = REPO_ROOT / "docs" / "ice_anisotropy" / "feature_ledger_link.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["feature_id"] == "H1-BROKEN-ICE-DEM"
    assert payload["node"] == "p474w1-n08-dem-anisotropic"
    assert payload["linkage"]["feature"] == "H1-BROKEN-ICE-DEM"
    for key in ("package", "study_script", "report_json", "fabric_ledger", "documentation"):
        assert payload["artifacts"][key]
    # every referenced in-repo artifact exists
    for key in ("package", "study_script", "documentation"):
        assert (REPO_ROOT / payload["artifacts"][key]).exists(), key
    for test_file in payload["artifacts"]["tests"]:
        assert (REPO_ROOT / test_file).exists(), test_file


def test_uniform_fabric_in_the_ledger_has_zero_fabric_index(study):
    assert study["fabric"]["uniform"]["anisotropy_index"] == pytest.approx(0.0, abs=1e-12)
    assert study["fabric"]["columnar"]["anisotropy_index"] == pytest.approx(1.0, abs=1e-12)
    assert FabricODF.uniform().anisotropy_index() == pytest.approx(0.0, abs=1e-12)