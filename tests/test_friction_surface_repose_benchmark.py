"""Tests for the pre-registered repose benchmark (node P152W1-N04, anchors A/B).

The benchmark is the machine-adjudicable face of the friction-contact surface:
it compares DEM repose angles with the literature bands stored in
``prereg/P152W1_N04_repose_protocol_v2_prereg.json`` and writes a JSON artifact.
These tests pin the *contract* (thresholds come from the pre-registration, the
payload is strict JSON, a run that yields no angle is WITHHELD, the CLI entry
points produce the artifact, and runs are reproducible), not the anchor numbers
themselves -- the numbers live in ``artifacts/P152W1-N04/repose_benchmark.json``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensordem.friction_surface import benchmark
from tensordem.friction_surface.benchmark import (
    BenchmarkPlan, load_prereg, run_benchmark, run_case, summarise,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def _plan(**kw) -> BenchmarkPlan:
    base = dict(particles=18, bisection_steps=1, seeds=(15204,), surface_mus=(),
                jobs=2, max_time=0.20, budget="test")
    base.update(kw)
    return BenchmarkPlan(**base)


@pytest.fixture(scope="module")
def payload() -> dict:
    """One cheap benchmark run, shared by the contract tests."""
    prereg = load_prereg()
    return benchmark._jsonable(run_benchmark(_plan(), prereg))


def _strict_loads(text: str):
    """json.loads that refuses NaN/Infinity -- the artifact must stay portable."""
    def _bad(token):                       # pragma: no cover - failure path
        raise ValueError(f"non-portable JSON constant: {token}")
    return json.loads(text, parse_constant=_bad)


def test_pre_registration_is_the_single_threshold_source():
    prereg = load_prereg()
    sha = benchmark.hashlib.sha256(open(prereg["_path"], "rb").read()).hexdigest()
    assert prereg["_sha256"] == sha
    by_id = {c["id"]: c for c in prereg["criteria_inherited"]}
    mats = by_id["B-repose-experiment"]["materials"]
    assert mats["glass_beads"]["theta_exp_deg"] == 23.0
    assert mats["quartz_sand"]["theta_exp_deg"] == 32.0
    # the benchmark must not carry a second copy of the bars
    assert benchmark.MU_ROLLING_BRACKET == (0.02, 0.40)


def test_payload_references_come_from_the_prereg(payload):
    prereg = load_prereg()
    mats = {c["id"]: c for c in prereg["criteria_inherited"]}["B-repose-experiment"]["materials"]
    assert payload["prereg"]["sha256"] == prereg["_sha256"]
    for name, mat in mats.items():
        row = payload["materials"][name]
        assert row["reference_deg"] == mat["theta_exp_deg"]
        assert row["reference_band_deg"] == [float(v) for v in mat["band"]]
        assert row["mu"] == mat["mu"]
    assert payload["negative_control"]["criterion_theta_r_deg_max"] == 5.0


def test_payload_schema_and_portability(payload):
    assert payload["schema"] == benchmark.SCHEMA
    for key in ("plan", "provenance", "calibration", "materials", "verdicts",
                "runs", "figure_positions", "honest_note"):
        assert key in payload
    text = json.dumps(payload)
    reloaded = _strict_loads(text)            # raises on NaN / Infinity
    assert reloaded["schema"] == payload["schema"]
    # verdict status vocabulary is closed
    assert {v["status"] for v in payload["verdicts"]} <= {"PASS", "FAIL", "WITHHELD"}
    assert payload["overall"] in {"PASS", "FAIL", "PARTIAL"}


def test_no_angle_means_withheld_not_a_pass(payload):
    """A starved run (0.2 s window) cannot reach rest -> WITHHELD, never PASS."""
    for name, row in payload["materials"].items():
        if row["got_deg"] is None:
            assert row["status"] == "WITHHELD"
            assert row["relative_error"] is None
    control = payload["negative_control"]
    if control["theta_r_deg"] is None:
        assert payload["verdicts"][0]["status"] == "WITHHELD"


def test_overall_status_is_derived_from_the_verdicts(payload):
    statuses = [v["status"] for v in payload["verdicts"]]
    expected = ("FAIL" if "FAIL" in statuses
                else ("PASS" if all(s == "PASS" for s in statuses) else "PARTIAL"))
    assert payload["overall"] == expected
    assert payload["overall"] in summarise(payload)


def test_worker_is_reproducible_and_keeps_the_nozzle_fixed():
    a = run_case(0.5, 0.15, 15204, n_particles=16, max_time=0.12)
    b = run_case(0.5, 0.15, 15204, n_particles=16, max_time=0.12)
    assert a["n_particles"] == b["n_particles"]
    assert a["t_end"] == b["t_end"]
    assert (a["theta_r_deg"] != a["theta_r_deg"]) == (b["theta_r_deg"] != b["theta_r_deg"])
    if a["theta_r_deg"] == a["theta_r_deg"]:
        assert a["theta_r_deg"] == pytest.approx(b["theta_r_deg"], abs=1e-12)
    # v2 protocol invariants survive the harness wrapper
    assert a["nozzle_z_max"] == pytest.approx(0.020, abs=1e-12)
    assert a["n_particles"] <= a["n_particles_requested"]


def test_module_entry_point_writes_the_artifact(tmp_path):
    out = tmp_path / "repose_benchmark.json"
    env = dict(os.environ, PYTHONPATH=str(SRC))
    proc = subprocess.run(
        [sys.executable, "-m", "tensordem.friction_surface",
         "--particles", "16", "--bisection-steps", "1", "--seeds", "15204",
         "--surface-mus", "", "--jobs", "2", "--max-time", "0.18",
         "--out", str(out)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert out.exists()
    data = _strict_loads(out.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["plan"]["particles"] == 16
    assert data["plan"]["budget"] == "reduced"
    assert "artifact:" in proc.stdout


def test_study_stage_benchmark_is_an_integration_caller(tmp_path):
    """The study CLI (production report driver) must run the benchmark stage."""
    env = dict(os.environ, PYTHONPATH=str(SRC))
    proc = subprocess.run(
        [sys.executable, "-m", "tensordem.friction_surface.study",
         "--stage", "benchmark", "--repose-particles", "16", "--jobs", "2",
         "--bench-max-time", "0.18", "--out", str(tmp_path)],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    artifact = tmp_path / "repose_benchmark.json"
    assert artifact.exists()
    data = _strict_loads(artifact.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["provenance"]["command"].startswith(sys.executable)