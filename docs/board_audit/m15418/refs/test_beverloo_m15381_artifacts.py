"""m15381 — Beverloo benchmark entry-point and artifact verification.

Ensures the benchmark script produces valid JSON artifacts with the expected
structure and that the measured flow rate is within the 15% tolerance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "beverloo_m15381"


@pytest.fixture(scope="module")
def medium_record():
    """Load the medium benchmark run record."""
    path = RUN_DIR / "beverloo_medium_benchmark_seed42.json"
    assert path.exists(), f"Run artifact missing: {path}"
    with open(path) as f:
        return json.load(f)


def test_artifact_exists():
    """The run artifact must exist after benchmark execution."""
    path = RUN_DIR / "beverloo_medium_benchmark_seed42.json"
    assert path.exists(), f"Run artifact missing: {path}"
    assert path.stat().st_size > 100, "Artifact is suspiciously small"


def test_artifact_structure(medium_record):
    """The record must have the expected top-level keys."""
    r = medium_record
    assert "mission_id" in r
    assert r["mission_id"] == 15381
    assert "benchmark" in r
    assert "reference" in r
    assert "case" in r
    assert "results" in r
    assert "comparison" in r


def test_reference_fields(medium_record):
    """Reference section must contain Beverloo constants."""
    ref = medium_record["reference"]
    assert ref["C"] == 0.58
    assert ref["k"] == 1.4
    assert "doi" in ref
    assert "10.1016/0009-2509(61)80059-4" in ref["doi"]


def test_case_geometry(medium_record):
    """Case geometry must be in the Beverloo-valid regime."""
    case = medium_record["case"]
    assert case["D_over_d"] >= 4.0, "D/d must be >= 4 for Beverloo validity"
    assert case["wall_friction"] == 0.0, "Smooth wall required for Beverloo comparison"
    assert case["particle_friction"] == 0.15


def test_results_consistency(medium_record):
    """Results must be internally consistent."""
    r = medium_record["results"]
    assert r["n_discharged"] > 0, "No particles discharged"
    assert r["n_discharged"] <= r["n_initial"]
    assert r["flow_rate_kg_s"] > 0
    assert r["bulk_density_kg_m3"] > 0
    assert abs(r["mass_balance_residual_kg"]) < 1e-12
    assert 0 <= r["steadiness_r2"] <= 1.0


def test_comparison_within_tolerance(medium_record):
    """The measured flow rate must be within 15% of the Beverloo prediction."""
    comp = medium_record["comparison"]
    assert comp["threshold_pct"] == 15.0
    assert comp["passed"] is True, (
        f"Benchmark failed: rel_err={comp['relative_error_pct']:+.2f}%"
    )
    assert abs(comp["relative_error"]) < 0.15


def test_recovered_C_reasonable(medium_record):
    """Recovered C should be in a plausible range (not wildly off)."""
    comp = medium_record["comparison"]
    # C can deviate from 0.58 due to finite-size effects, but should be < 2.0
    assert 0.1 < comp["recovered_C"] < 2.0, (
        f"Recovered C={comp['recovered_C']:.4f} is unreasonable"
    )


def test_report_artifact_exists():
    """The report JSON must exist."""
    path = RUN_DIR / "report_m15381.json"
    assert path.exists(), f"Report artifact missing: {path}"
    with open(path) as f:
        report = json.load(f)
    assert report["mission_id"] == 15381
    assert "figures" in report
    assert len(report["figures"]) >= 2


def test_figure_files_exist():
    """Generated figure files must exist."""
    fig_dir = ROOT / "docs" / "figures"
    assert (fig_dir / "m15381_beverloo_comparison.png").exists()
    assert (fig_dir / "m15381_window_series.png").exists()
