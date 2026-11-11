"""Tests for the repose-model qualification gates (round-2 gate 1).

The gates decide whether a DEM repose response may be used as a calibration
target, so the logic itself has to be verified independently of any DEM run:
synthetic cell records with a known answer (signal present / absent / noise
dominated) plus an internal-consistency check on the stored artefact.
"""
from __future__ import annotations

import json
import os

import pytest

from tensordem.dem_calibration import repose_qualification as rq

ARTIFACT = os.path.join("artifacts", "gap_dem_autocalib_r2", "qualification.json")


def _cell(n, seed, mu, irf, theta, phi=0.55, r2=0.99, block="DESIGN"):
    return {"block": block, "n_grains": n, "seed": seed, "mu": mu,
            "inject_radius_factor": irf, "theta_r_deg": theta,
            "phi_solid": phi, "r2": r2, "ke_ratio": 0.01}


def _good_matrix():
    """A population that satisfies every gate: real heap, converged, material-led."""
    cells = []
    for n, theta in ((240, 25.0), (480, 28.0), (960, 30.0), (1920, 30.5)):
        for seed in (0, 1):
            cells.append(_cell(n, seed, 0.52, 7.0, theta + 0.2 * seed, block="SIZE"))
    # material knob at fixed geometry: 10 deg of signal
    for mu, theta in ((0.30, 22.0), (0.70, 32.0)):
        for seed in (0, 1):
            cells.append(_cell(960, seed, mu, 3.5, theta + 0.2 * seed))
    # geometry knob at fixed material: 0.5 deg of artefact
    for seed in (0, 1):
        cells.append(_cell(960, seed, 0.30, 14.0, 22.5 + 0.2 * seed))
    return cells


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def test_spearman_monotone_and_reversed():
    x = [0.2, 0.35, 0.52, 0.7, 0.85]
    assert rq.spearman(x, [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert rq.spearman(x, [5, 4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_ties_and_degenerate():
    # all-equal y has no rank variance -> defined as 0
    assert rq.spearman([1, 2, 3, 4], [2, 2, 2, 2]) == 0.0
    with pytest.raises(ValueError):
        rq.spearman([1.0], [1.0])


def test_seed_sigma_pools_replicas_only():
    cells = [_cell(960, 0, 0.52, 7.0, 10.0), _cell(960, 1, 0.52, 7.0, 12.0),
             _cell(480, 0, 0.52, 7.0, 40.0)]     # singleton: no spread to pool
    # one replicated group with dof 1: sample sd of (10, 12) is sqrt(2)
    assert rq.seed_sigma(cells) == pytest.approx(2 ** 0.5)


def test_seed_sigma_nan_without_replicas():
    import math
    cells = [_cell(240, 0, 0.52, 7.0, 10.0), _cell(480, 0, 0.52, 7.0, 20.0)]
    assert math.isnan(rq.seed_sigma(cells))


def test_dominance_pass_and_fail():
    ok = rq.dominance(signal_delta=10.0, geometry_delta=0.5, sigma=1.0)
    assert ok["passed"] and ok["ratio"] == pytest.approx(20.0)
    bad = rq.dominance(signal_delta=0.5, geometry_delta=10.0, sigma=1.0)
    assert not bad["passed"]
    # resolvable but not dominant
    weak = rq.dominance(signal_delta=4.0, geometry_delta=2.0, sigma=1.0)
    assert weak["snr_signal"] >= rq.SIGNAL_TO_NOISE_MIN and not weak["passed"]
    with pytest.raises(ValueError):
        rq.dominance(1.0, 1.0, 0.0)


def test_size_plateau_absolute_and_relative():
    slope, tol, ok = rq.size_plateau({960: 30.0, 1920: 31.0})
    assert ok and slope == pytest.approx(1.0) and tol == pytest.approx(2.0)
    slope, tol, ok = rq.size_plateau({960: 20.0, 1920: 40.0})
    assert not ok and slope == pytest.approx(20.0)
    with pytest.raises(ValueError):
        rq.size_plateau({960: 30.0})


# --------------------------------------------------------------------------
# whole-population verdicts
# --------------------------------------------------------------------------
def test_evaluate_qualified_population():
    rep = rq.evaluate(_good_matrix(), size_match={"mu": 0.52,
                                                  "inject_radius_factor": 7.0})
    assert rep["verdict"] == "qualified", rep["failed_gates"]
    assert rep["failed_gates"] == []
    assert rep["size_sweep"][1920] == pytest.approx(30.6)
    assert rep["dominance"]["ratio"] > 3.0


def test_evaluate_sheet_deposit_fails_solid_fraction():
    cells = _good_matrix()
    for c in cells:
        c["phi_solid"] = 0.01
    rep = rq.evaluate(cells, size_match={"mu": 0.52, "inject_radius_factor": 7.0})
    assert rep["verdict"] == "not_qualified"
    assert "G2_solid_fraction" in rep["failed_gates"]
    assert "G1_flank_quality" not in rep["failed_gates"]


def test_evaluate_geometry_dominated_fails_dominance():
    cells = _good_matrix()
    for c in cells:                       # wipe the material signal
        if c["n_grains"] == 960 and c["inject_radius_factor"] == 3.5:
            c["theta_r_deg"] = 22.0
    for c in cells:                       # make the protocol knob dominant
        if c["inject_radius_factor"] == 14.0:
            c["theta_r_deg"] = 45.0
    rep = rq.evaluate(cells, size_match={"mu": 0.52, "inject_radius_factor": 7.0})
    assert "G4b_material_dominance" in rep["failed_gates"]
    assert rep["verdict"] == "not_qualified"


def test_evaluate_unconverged_size_fails_plateau():
    cells = _good_matrix()
    for c in cells:
        if c["n_grains"] == 1920:
            c["theta_r_deg"] += 15.0
    rep = rq.evaluate(cells, size_match={"mu": 0.52, "inject_radius_factor": 7.0})
    assert "G3_size_convergence" in rep["failed_gates"]


# --------------------------------------------------------------------------
# stored artefact: re-evaluation must reproduce the stored gates exactly
# --------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(ARTIFACT),
                    reason="round-2 artefact not generated in this checkout")
def test_artefact_reevaluates_consistently():
    with open(ARTIFACT) as fh:
        stored = json.load(fh)
    rep = rq.evaluate(stored["cells"],
                      size_match={"mu": 0.52, "inject_radius_factor": 7.0})
    assert rep["verdict"] == stored["verdict"]
    assert rep["failed_gates"] == stored["failed_gates"]
    for name, gate in rep["gates"].items():
        assert gate["value"] == pytest.approx(stored["gates"][name]["value"], rel=1e-9)
    # verdict bookkeeping: qualified iff no gate failed
    assert (stored["verdict"] == "qualified") == (stored["failed_gates"] == [])
    if stored.get("determinism"):
        assert stored["determinism"]["bitwise_identical"] is True