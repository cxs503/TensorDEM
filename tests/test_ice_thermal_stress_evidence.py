"""The acceptance ladder itself, and its tamper-evident digest.

The benchmark script ``benchmarks/ice_thermal_stress/run_validation.py`` writes
the same payload to ``artifacts/ice_thermal_stress/validation.json``.  These
tests make sure the ladder still passes *and* that a hand-edited artifact is
detectable (the digest covers every number).
"""
from __future__ import annotations

import json

import shutil
from pathlib import Path

import pytest

from tensordem.ice_thermal_stress import validation as V


@pytest.fixture(scope="module")
def report():
    return V.acceptance_report(quick=True)


def _writable_dir() -> Path:
    """A directory that actually accepts writes on this (often full) filesystem.

    ``tmp_path`` is deliberately not used: when the box is out of space pytest's
    own temporary-directory fixture fails during setup, before any test code can
    react.  The candidates are repo-local, so git keeps them out of the tree.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here / ".pytest_evidence", here.parent / ".pytest_tmp"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".probe"
            probe.write_text("x")
            probe.unlink()
            return candidate
        except OSError:
            continue
    pytest.skip("no writable directory available (filesystem full)")


def test_every_measurable_criterion_passes(report):
    verdict = report["verdict"]
    assert verdict["all_pass"], {
        k: v for k, v in report["criteria"].items() if v.get("pass") is False}
    assert verdict["n_pass"] == verdict["n_criteria"]
    # the two criteria that cannot be closed offline are declared, not hidden
    assert set(verdict["withheld"]) == {"C11_experimental_arm", "C12_layer_depth_band"}


def test_the_impact_case_is_the_documented_one(report):
    impact = report["impact"]
    assert impact["case"]["impact_speed_m_s"] == 1.0
    assert impact["case"]["bulk_temperature_c"] == -10.0
    assert impact["peak_delta_t_k"] > 1.0                     # the patch does heat up
    assert 0.0 < impact["thermal_fraction_of_strength"] < 10.0
    assert impact["sigma_f_bulk_pa"] == pytest.approx(
        float(__import__("tensordem.ice_material", fromlist=["pure_ice_strength"])
              .pure_ice_strength(-10.0)) * 1.0e6)


def test_digest_detects_a_tampered_artifact(report):
    work = _writable_dir()
    payload = dict(report)
    payload["digest_sha256"] = V.canonical_digest(payload)
    path = work / "validation.json"
    path.write_text(json.dumps(payload, sort_keys=True, default=V._json_default))
    assert V.verify_evidence(path)["match"] is True

    data = json.loads(path.read_text())
    data["impact"]["peak_delta_t_k"] = 999.0        # forged headline number
    path.write_text(json.dumps(data, sort_keys=True, default=V._json_default))
    verdict = V.verify_evidence(path)
    assert verdict["match"] is False                  # digest catches the edit


def test_write_and_verify_roundtrip():
    work = _writable_dir()
    path = work / "evidence.json"
    try:
        V.write_evidence(path, quick=True)
    except OSError:                        # pragma: no cover - full filesystem
        pytest.skip("could not write the evidence file (filesystem full)")
    assert V.verify_evidence(path)["match"] is True
    shutil.rmtree(work, ignore_errors=True)