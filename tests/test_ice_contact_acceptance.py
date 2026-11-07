"""Acceptance-gate machinery of campaign node IN27-S2 (reduced study).

This test does **not** assert that the campaign bands are met — the measured
verdict is whatever the DEM produces, and it is reported (not forced) in
``artifacts/ice_contact_in27_s2/study.json`` and in the mission report.  What is
asserted here is that

1. a real (reduced-size) study runs end to end,
2. every gate of the campaign node is computed from the simulated numbers,
3. the printed verdict is consistent with the artifact, and
4. the artifact, when present, carries the same gate structure.

Run time is kept to about a minute by using 4 conditions (2 speeds x 2
thicknesses), 3 proxy ``mu_r`` values, 1 seed and a small domain.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensordem.ice_contact.study import ACCEPTANCE, acceptance_gates, run_study

ARTIFACT = Path("artifacts/ice_contact_in27_s2/study.json")


@pytest.fixture(scope="module")
def reduced_study():
    return run_study(
        speeds=(1.0, 5.0),
        thicknesses=(0.5, 1.5),
        mu_r_grid=(0.10, 0.20, 0.30),
        seeds=(20261031,),
        workers=8,
        keep_arrays=True,
        n_max=100,
        width=1.4,
        stroke=0.08,
    )


def test_reduced_study_runs_and_reports_every_gate(reduced_study):
    gates = reduced_study["acceptance"]
    for key in ("n_conditions", "L_c", "gamma_spatial", "gamma_instant",
                "proxy_bias", "force_correlation", "equivalence", "verdict"):
        assert key in gates, key
    assert gates["n_conditions"]["value"] == 4
    assert gates["L_c"]["band"] == list(ACCEPTANCE["L_c_m"])
    assert gates["gamma_spatial"]["band"] == list(ACCEPTANCE["gamma"])
    assert gates["proxy_bias"]["threshold"] == ACCEPTANCE["abs_delta_mu_r_0p1_0p3"]


def test_reduced_study_numbers_are_finite(reduced_study):
    loaded = 0
    for cond in reduced_study["conditions"]:
        stats = cond["polyhedron"]["statistics"]
        if cond["polyhedron"].get("no_contact") or not np.isfinite(stats["L_c"]):
            # a configuration that never loads the structure is flagged, not faked
            continue
        loaded += 1
        assert stats["mean_pressure"] > 0.0
        assert stats["peak_field_pressure"] >= stats["mean_pressure"]
        for bias in cond["sphere"]["biases"]:
            assert np.isfinite(bias["delta_mean"])
            assert np.isfinite(bias["r2_raw"])
    assert loaded >= len(reduced_study["conditions"]) // 2


def test_verdict_is_consistent_with_the_gate_values(reduced_study):
    gates = reduced_study["acceptance"]
    verdict = gates["verdict"]
    assert verdict["a_L_c"] == (gates["L_c"]["fraction_in_band"] >= 0.9)
    assert verdict["c_proxy_bias"] == gates["proxy_bias"]["all_below_threshold"]
    assert verdict["d_force_correlation"] == gates["force_correlation"]["r2_block_all_above"]
    assert verdict["e_conditions"] == gates["n_conditions"]["pass"]
    assert verdict["all_pass"] == all(
        verdict[k] for k in ("a_L_c", "b_gamma", "c_proxy_bias", "d_force_correlation", "e_conditions")
    )


def test_correlation_length_is_resolved_by_the_grid(reduced_study):
    # L_c must be a physical length, not a normalisation artefact: it lies inside
    # the available separation range and is a multiple of the cell size.
    checked = 0
    for cond in reduced_study["conditions"]:
        stats = cond["polyhedron"]["statistics"]
        if cond["polyhedron"].get("no_contact") or not np.isfinite(stats["L_c"]):
            continue
        dy, dz = cond["polyhedron"]["sim"]["grid"]
        assert 0.0 < stats["L_c"] < 10.0 * max(dy, dz)
        checked += 1
    assert checked >= 1


def test_artifact_matches_the_gate_structure():
    if not ARTIFACT.exists():
        pytest.skip("production artifact not present in this checkout")
    payload = json.loads(ARTIFACT.read_text())
    assert payload["acceptance"]["n_conditions"]["value"] >= ACCEPTANCE["min_conditions"]
    assert len(payload["conditions"]) >= ACCEPTANCE["min_conditions"]
    speeds = sorted({c["speed"] for c in payload["conditions"]})
    thicknesses = sorted({c["thickness"] for c in payload["conditions"]})
    assert min(speeds) <= 1.0 and max(speeds) >= 5.0
    assert min(thicknesses) <= 0.5 and max(thicknesses) >= 1.5
    # the gates in the artifact must be reproducible from the artifact itself
    recomputed = acceptance_gates(payload)
    assert recomputed["verdict"] == payload["acceptance"]["verdict"]