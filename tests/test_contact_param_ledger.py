"""Dimension ledger and similarity-gate tests (P482-N06, M1/M4).

The tests cover three things:

1. the dimension algebra (an exponent must follow from the SI vector, not from
   a hand-written table),
2. the fail-closed behaviour of the fleet scan (a new contact parameter without
   a dimension label is an error; an out-of-range restitution is an error; a
   silently clamped restitution is a warning),
3. the similarity-manifest audit used as the CI gate (green on the correct
   transform, red - i.e. ``CAUGHT`` - on each pre-registered mistake).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensordem.contact_param_ledger import (
    DIMENSIONLESS,
    FROUDE,
    PARAMETERS,
    SimilarityMode,
    audit_manifest,
    audit_variant,
    exponent_table,
    label_parameter,
    scale_case_json,
    scale_value,
    scan_fleet,
    write_ledger,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fleet_scan():
    """One fleet scan shared by the scan tests (parsing the fleet is costly)."""
    return scan_fleet(REPO)


# ---------------------------------------------------------------------------
# 1. dimension algebra
# ---------------------------------------------------------------------------


def test_froude_base_scales():
    mode = FROUDE
    assert mode.length == 1.0
    assert mode.mass == 3.0          # fixed density
    assert mode.time == 0.5          # fixed gravity, g = L/T^2


@pytest.mark.parametrize("name,exponent", [
    ("radius", 1.0),
    ("mu", 0.0),
    ("restitution", 0.0),
    ("kt_ratio", 0.0),
    ("kn", 2.0),                     # linear spring  F = kn delta
    ("kt", 2.0),
    ("kr", 4.0),                     # kt * r^2
    ("rolling_lever", 1.0),          # mu_r carries length
    ("mu_rolling", 0.0),             # dimensionless ratio convention
    ("gamma", 2.5),                  # N s/m
    ("dt", 0.5),
    ("normal_stress", 1.0),          # rho g H
    ("shear_velocity", 0.5),
    ("density", 0.0),
    ("gravity", 0.0),
])
def test_required_exponents(name, exponent):
    assert FROUDE.exponent_of(name) == pytest.approx(exponent)


def test_hertz_parameterisation_is_not_the_sqrt_rule():
    """The Hertz stiffness [N/m^1.5] must scale as lam^1.5, not lam^0.5.

    kn = (4/3) E* sqrt(R) scales as lam^0.5 *only* at fixed E*, which breaks
    gravity similarity; similarity requires E* ~ lam and hence kn ~ lam^1.5.
    """
    hertz = PARAMETERS["kn"].alternative
    assert hertz is not None
    assert FROUDE.exponent(hertz) == pytest.approx(1.5)
    modulus = PARAMETERS["youngs_modulus"].dimension
    assert FROUDE.exponent(modulus) == pytest.approx(1.0)


def test_kr_is_consistent_with_kt_and_radius():
    kr = FROUDE.exponent(PARAMETERS["kr"].dimension)
    kt = FROUDE.exponent(PARAMETERS["kt"].dimension)
    r = FROUDE.exponent(PARAMETERS["radius"].dimension)
    assert kr == pytest.approx(kt + 2.0 * r)


def test_density_free_mode_changes_the_answer():
    """A different similarity family must change the exponents."""
    heavy = SimilarityMode(name="fixed_mass", density=0.0, length=1.0,
                           gravity=0.0)
    fixed_rho = SimilarityMode(name="fixed_mass_density", length=1.0)
    # both keep rho fixed, so mass still scales as L^3
    assert heavy.exponent(PARAMETERS["kn"].dimension) == pytest.approx(
        fixed_rho.exponent(PARAMETERS["kn"].dimension))


def test_exponent_table_is_serialisable():
    table = exponent_table(FROUDE)
    assert table["kn"]["exponent"] == pytest.approx(2.0)
    assert table["mu_rolling"]["alternative_exponent"] == pytest.approx(1.0)
    json.dumps(table)


# ---------------------------------------------------------------------------
# 2. transform and variant audit
# ---------------------------------------------------------------------------


def test_scale_value_and_missing_parameter():
    assert scale_value("kn", 600.0, 2.0) == pytest.approx(2400.0)
    assert scale_value("rolling_lever", 1.5e-3, 2.0) == pytest.approx(3.0e-3)
    assert scale_value("mu", 0.55, 2.0) == pytest.approx(0.55)
    assert scale_value("restitution", None, 2.0) is None
    with pytest.raises(KeyError):
        scale_value("not_a_contact_parameter", 1.0, 2.0)


def test_scale_case_json_is_fail_closed():
    base = {"kn": 600.0, "radius": 0.005, "mu": 0.5}
    scaled = scale_case_json(base, 2.0)
    assert scaled["kn"] == pytest.approx(2400.0)
    assert scaled["radius"] == pytest.approx(0.01)
    with pytest.raises(KeyError):
        scale_case_json({"mystery_stiffness": 1.0}, 2.0)


def _manifest(control_params: dict) -> dict:
    base = {"radius": 0.005, "kn": 600.0, "rolling_lever": 1.5e-3,
            "mu": 0.55, "restitution": 0.4, "box_width": 0.07}
    correct = scale_case_json(base, 2.0)
    return {"cases": [{
        "name": "probe", "lam": 2.0, "base": base,
        "variants": {
            "similar": {"expected": "OK", "params": correct},
            "kn_constant": {"expected": "violation",
                            "params": {**correct, "kn": base["kn"]}},
            "lever_constant": {"expected": "violation",
                               "params": {**correct,
                                          "rolling_lever": base["rolling_lever"]}},
            "restitution_boost": {"expected": "violation",
                                  "params": {**correct, "restitution": 0.5}},
            **control_params,
        }}]}


def test_audit_variant_flags_each_mistake():
    base = {"kn": 600.0, "rolling_lever": 1.5e-3, "restitution": 0.4}
    correct = scale_case_json(base, 2.0)
    assert all(r["status"] == "OK" for r in
               audit_variant(base, correct, 2.0))
    rows = {r["parameter"]: r for r in
            audit_variant(base, {**correct, "kn": 600.0}, 2.0)}
    assert rows["kn"]["status"] == "VIOLATION"
    assert rows["kn"]["observed_exponent"] == pytest.approx(0.0)
    rows = {r["parameter"]: r for r in
            audit_variant(base, {**correct, "rolling_lever": 1.5e-3}, 2.0)}
    assert rows["rolling_lever"]["status"] == "VIOLATION"
    rows = {r["parameter"]: r for r in
            audit_variant(base, {**correct, "restitution": 0.5}, 2.0)}
    assert rows["restitution"]["status"] == "VIOLATION"


def test_gate_is_green_on_correct_transform():
    report = audit_manifest(_manifest({}))
    assert report["status"] == "OK"
    variants = report["cases"][0]["variants"]
    assert variants[0]["status"] == "OK"
    for entry in variants[1:]:
        assert entry["status"] == "CAUGHT"


def test_gate_reports_a_missed_control():
    """If a declared control is in fact similarity preserving, the gate must
    say MISSED (it cannot be satisfied by mislabelling a variant)."""
    base = {"radius": 0.005, "kn": 600.0, "rolling_lever": 1.5e-3}
    correct = scale_case_json(base, 2.0)
    manifest = {"cases": [{
        "name": "x", "lam": 2.0, "base": base,
        "variants": {"fake_control": {"expected": "violation",
                                      "params": correct}},
    }]}
    report = audit_manifest(manifest)
    assert report["status"] == "FAIL"
    assert report["cases"][0]["variants"][0]["status"] == "MISSED"


# ---------------------------------------------------------------------------
# 3. fleet scan (fail-closed)
# ---------------------------------------------------------------------------


def test_fleet_scan_is_green_and_covers_the_cases(fleet_scan):
    scan = fleet_scan
    summary = scan.summary()
    assert summary["modules_in_scope"] > 50
    assert summary["fields_registered"] > 500
    errors = [v for v in scan.violations if v.severity == "error"]
    assert errors == [], [e.as_dict() for e in errors[:5]]
    modules = set(scan.in_scope)
    assert "src/tensordem/shear_cell.py" in modules
    assert "src/tensordem/granular3d_core.py" in modules


def test_fleet_scan_labels_the_rolling_conventions():
    """Both rolling conventions must be labelled, with different dimensions."""
    ratio = label_parameter("mu_rolling")
    lever = label_parameter("rolling_lever")
    assert ratio is not None and lever is not None
    assert ratio.dimension == DIMENSIONLESS
    assert lever.dimension.l == 1.0
    assert FROUDE.exponent(ratio.dimension) == pytest.approx(0.0)
    assert FROUDE.exponent(lever.dimension) == pytest.approx(1.0)


def _write_module(tmp_path: Path, body: str) -> Path:
    pkg = tmp_path / "src" / "tensordem"
    pkg.mkdir(parents=True)
    (pkg / "case_mod.py").write_text(body, encoding="utf-8")
    return tmp_path


def test_unregistered_contact_parameter_is_an_error(tmp_path):
    root = _write_module(tmp_path, """
from dataclasses import dataclass

@dataclass(frozen=True)
class NewCase:
    radius: float = 0.005
    kn: float = 1000.0
    hertz_contact_law_parameter: float = 1.0
""")
    scan = scan_fleet(root)
    kinds = {v.kind for v in scan.violations if v.severity == "error"}
    assert "UNREGISTERED_CONTACT_PARAMETER" in kinds


def test_out_of_range_restitution_is_an_error(tmp_path):
    root = _write_module(tmp_path, """
from dataclasses import dataclass

@dataclass(frozen=True)
class BadCase:
    radius: float = 0.005
    restitution: float = 1.05
""")
    scan = scan_fleet(root)
    errors = [v for v in scan.violations if v.severity == "error"]
    assert any("outside valid range" in v.message for v in errors)


def test_silent_clamp_is_reported(tmp_path):
    root = _write_module(tmp_path, """
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    radius: float = 0.005
    restitution: float = 0.4


def damping_ratio(restitution: float) -> float:
    e = min(max(restitution, 1e-6), 1.0 - 1e-12)
    return abs(math.log(e)) / math.pi
""")
    scan = scan_fleet(root)
    assert any(v.kind == "SILENT_CLAMP" for v in scan.violations)


def test_ledger_json_roundtrip(tmp_path, fleet_scan):
    out = tmp_path / "ledger.json"
    payload = fleet_scan.as_json()
    payload["exponent_table"] = exponent_table(FROUDE)
    out.write_text(json.dumps(payload), encoding="utf-8")
    data = json.loads(out.read_text())
    assert payload["summary"]["errors_total"] >= 0
    assert data["schema"] == "tensordem.contact_param_ledger/1"
    assert payload["summary"]["errors_total"] == 0
    assert "kn" in data["parameters"]
    assert data["exponent_table"]["rolling_lever"]["exponent"] == pytest.approx(1.0)