"""M4 CI gate: the shipped similarity manifest must stay green, and it must
turn red when a parameter is deliberately mis-scaled.

This is the red/green contrast required by the node: the *shipped* artifact
`artifacts/contact_scale_ledger/gate_manifest.json` is audited against the
dimension ledger (green), and a corrupted copy - one where a parameter is kept
constant instead of being scaled - must be reported as ``VIOLATION``/``MISSED``.

If someone later edits the manifest to make a mistake look legitimate, this test
fails; if someone edits the ledger exponents, `test_required_exponents` in
``test_contact_param_ledger.py`` fails first.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tensordem.contact_param_ledger import audit_manifest

MANIFEST = (Path(__file__).resolve().parents[1] / "artifacts"
            / "contact_scale_ledger" / "gate_manifest.json")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_shipped_manifest_is_green(manifest):
    report = audit_manifest(manifest)
    assert report["status"] == "OK", json.dumps(report)[:2000]
    variants = {v["variant"]: v for v in report["cases"][0]["variants"]}
    assert variants["lamN_similar"]["status"] == "OK"
    for name in ("ctl_kn_constant", "ctl_lever_constant",
                 "ctl_restitution_boost", "ctl_restitution_out_of_range"):
        assert variants[name]["status"] == "CAUGHT", name


def test_gate_turns_red_when_kh_is_not_scaled(manifest):
    broken = copy.deepcopy(manifest)
    case = broken["cases"][0]
    good = case["variants"]["lamN_similar"]["params"]
    case["variants"]["lamN_similar"]["params"] = {**good, "kn": good["kn"] / 4.0}
    report = audit_manifest(broken)
    assert report["status"] == "FAIL"
    statuses = {v["variant"]: v for v in report["cases"][0]["variants"]}
    assert statuses["lamN_similar"]["status"] == "VIOLATION"
    assert "kn" in statuses["lamN_similar"]["deviating_parameters"]


def test_gate_turns_red_when_the_rolling_lever_is_not_scaled(manifest):
    broken = copy.deepcopy(manifest)
    case = broken["cases"][0]
    good = case["variants"]["lamN_similar"]["params"]
    case["variants"]["lamN_similar"]["params"] = {
        **good, "rolling_lever": good["rolling_lever"] / 2.0}
    report = audit_manifest(broken)
    assert report["status"] == "FAIL"
    first = report["cases"][0]["variants"][0]
    assert first["status"] == "VIOLATION"
    assert first["deviating_parameters"] == ["rolling_lever"]


def test_gate_turns_red_on_unregistered_parameter(manifest):
    broken = copy.deepcopy(manifest)
    case = broken["cases"][0]
    good = case["variants"]["lamN_similar"]["params"]
    case["variants"]["lamN_similar"]["params"] = {
        **good, "mystery_hertz_stiffness": 1.0}
    report = audit_manifest(broken)
    first = report["cases"][0]["variants"][0]
    assert "mystery_hertz_stiffness" in first["unregistered_parameters"]
    assert report["status"] == "FAIL"