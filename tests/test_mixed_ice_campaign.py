"""Campaign integration: artifact, verdicts, JSON sanitising, figures, report.

The DEM factorial is skipped here (it is the expensive part and is exercised by
``scripts/mixed_ice_campaign.py``); everything else -- the sweeps, the
sensitivity, the joint envelope, the estimator check -- runs for real.
"""
from __future__ import annotations

import json
import re

import pytest

from tensordem.mixed_ice import campaign, figures, report


@pytest.fixture(scope="module")
def artifact():
    return campaign.run_campaign(with_dem=False)


def test_artifact_contains_every_table(artifact):
    for key in ("ageing_table", "my_fraction_sweep", "class_ratio_decomposition",
                "sensitivity", "joint_envelope", "estimator_check",
                "field_summary_p04", "field_histogram_p04", "provenance", "verdicts"):
        assert key in artifact, key
    assert artifact["my_fraction_sweep"]["primary_endpoints"]["resistance_end_N"] > 0.0


def test_anchor_only_verdicts_all_pass(artifact):
    verdicts = artifact["verdicts"]
    assert artifact["n_fail"] == 0, {k: v for k, v in verdicts.items() if v["verdict"] == "FAIL"}
    for key in ("a_my_fraction_sweep_monotone", "b_class_weight_identity",
                "c_estimator_agreement", "g_anchor_cross_model",
                "h_fsd_sensitivity_reported", "i_ageing_law_limits",
                "j_provenance_hashed"):
        assert verdicts[key]["verdict"] == "PASS", key
    # the DEM criteria are absent, not silently passing
    assert "d_dem_sigma_c_exponent" not in verdicts


def test_primary_endpoint_uses_the_unit_strength_level(artifact):
    rows = [r for r in artifact["my_fraction_sweep"]["curves"]["lindqvist_v3"]["rows"]
            if r["my_strength_multiplier"] == 1.0 and r["my_fraction"] == 0.8]
    endpoint = artifact["my_fraction_sweep"]["primary_endpoints"]
    assert endpoint["resistance_end_N"] == pytest.approx(rows[0]["resistance_mean_N"], rel=1e-12)


def test_artifact_json_is_standards_compliant(tmp_path, artifact):
    path = campaign.save_artifact(artifact, tmp_path / "artifact.json")
    payload = json.loads(path.read_text())  # fails on NaN / Infinity
    assert payload["verdicts"]["a_my_fraction_sweep_monotone"]["verdict"] == "PASS"
    assert payload["my_fraction_sweep"]["curves"]["lindqvist_v3"]["rows"][0]["class_ratio"] is None


def test_figures_render(tmp_path, artifact):
    made = figures.make_all(artifact, tmp_path / "figs")
    assert len(made) == 3  # no DEM figure without the DEM
    for path in made:
        assert path.exists() and path.stat().st_size > 10_000, path


def test_report_renders_without_placeholders(artifact, tmp_path):
    figures.make_all(artifact, tmp_path / "figs")
    text = report.render_report(artifact, figures=["fig1_my_fraction_curve.png"])
    # no non-finite number may leak into the prose (word-bounded check: "dominant"
    # and the sha-256 hex strings must not trip it)
    assert re.search(r"\b(nan|inf|infinity)\b", text, flags=re.IGNORECASE) is None
    for section in ("## 1.", "## 2.", "## 3.", "## 5.", "## 6.", "## 7.", "## 8."):
        assert section in text
    # every verdict key appears in the verdict table
    for key in artifact["verdicts"]:
        assert key in text
    results = report.render_results_markdown(artifact)
    assert "R(0)" in results and "Verdict" not in results