"""Integrity gate for the committed C2 (rubble repose) evidence of ``H1-BROKEN-ICE-DEM``.

The C2 shards of :mod:`scripts.run_h1_repose` are compute bound (30 000 steps per
friction value), so their numbers are produced once and committed under
``runs/h1_repose/``.  That makes the evidence itself the artefact that has to be
guarded: a committed JSON must not be able to claim a repose angle it did not
measure, a settled heap it did not reach, or a literature-band hit that the
reported angle does not have.

These tests assert *invariants of the ledger*, not success: an honest
``settled = False`` run (which is what a failed shard must be) passes here, a
fabricated angle does not.  The last test keeps the markdown report equal to the
JSON (zero hand copying).

Missing shard files are skipped individually so that a partial campaign does not
turn CI red for the wrong reason -- the report renders missing shards as
``NOT_RUN`` instead.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS_DIR = os.path.join(ROOT, "runs", "h1_repose")
DOC = os.path.join(ROOT, "docs", "H1_BROKEN_ICE_DEM_ROUND5_C2.md")

FRICTION_GRID = (0.2, 0.4, 0.6)
ENERGY_GATE = 0.05  # study.case3_repose: `valid` requires energy_residual < 0.05


def _shard_files():
    return sorted(glob.glob(os.path.join(RUNS_DIR, "repose_mu*.json")))


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _nan(x):
    try:
        return float(x) != float(x)
    except (TypeError, ValueError):
        return True


@pytest.mark.parametrize("path", _shard_files() or [None])
def test_shard_file_is_self_consistent(path):
    if path is None:
        pytest.skip("no C2 shard evidence committed under runs/h1_repose/")
    data = _load(path)
    run = data["runs"][0]
    setup = data["setup"]
    band = data["ref"]["band_deg"]

    # --- identity: file name, shard index and the friction grid agree ---------
    mu_name = float(re.search(r"repose_mu([0-9.]+)\.json$", path).group(1))
    assert float(run["friction"]) == pytest.approx(mu_name, abs=1e-12)
    assert data["shard"]["of"] == list(FRICTION_GRID)
    assert FRICTION_GRID[data["shard"]["index"] - 1] == pytest.approx(mu_name)
    assert setup["friction_values"] == [pytest.approx(mu_name)]

    # --- the run is what the setup block says it is --------------------------
    assert 1 <= run["steps"] <= setup["n_steps"]
    assert run["sim_time_s"] == pytest.approx(run["steps"] * setup["dt"], rel=1e-9)
    assert run["mass_residual"] < 1e-12, "polyhedron mass may not drift"

    # --- no fabricated measurement -------------------------------------------
    if not run["settled"]:
        assert _nan(run["repose_angle_deg"]), "an unsettled heap must not report an angle"
        assert run["flank_reason"], "an unsettled heap must say why there is no flank"
        assert run["valid"] is False
    else:
        assert math.isfinite(float(run["repose_angle_deg"]))

    # --- `valid` means exactly what the case doc says it means ---------------
    expected_valid = bool(
        run["settled"]
        and run["energy_residual"] < ENERGY_GATE
        and run["n_contacts"] > 0
        and math.isfinite(float(run["repose_angle_deg"]))
        if not _nan(run["repose_angle_deg"])
        else False
    )
    assert bool(run["valid"]) is expected_valid

    # --- the band flag is a function of the angle, not a hope ---------------
    if _nan(run["repose_angle_deg"]):
        assert data["inside_literature_band"][0] is False
    else:
        a = float(run["repose_angle_deg"])
        assert data["inside_literature_band"][0] is (band[0] <= a <= band[1])

    # --- campaign verdict is the AND of its parts ---------------------------
    assert data["passed"] is bool(
        data["all_runs_valid"] and all(data["inside_literature_band"]))


def test_shards_cover_disjoint_friction_values():
    files = _shard_files()
    if not files:
        pytest.skip("no C2 shard evidence committed under runs/h1_repose/")
    mus = [float(_load(f)["runs"][0]["friction"]) for f in files]
    assert len(mus) == len(set(mus)), "two files claim the same friction value"
    assert set(mus) <= set(FRICTION_GRID)


def test_documented_angle_of_repose_matches_the_measurement():
    """Zero hand copy: what the report prints is what the JSONs measured."""
    files = _shard_files()
    if not files or not os.path.exists(DOC):
        pytest.skip("need both the shard evidence and the rendered report")
    doc = open(DOC).read()
    for f in files:
        data = _load(f)
        run = data["runs"][0]
        mu = float(run["friction"])
        row = [ln for ln in doc.splitlines()
               if ln.startswith("|") and re.search(r"\|\s*%g\s*\|" % mu, ln)]
        assert row, "report has no C2 row for mu=%g" % mu
        cells = [c.strip() for c in row[0].strip().strip("|").split("|")]
        angle_cell = cells[6]
        if _nan(run["repose_angle_deg"]):
            assert angle_cell in ("nan", "—", "NOT_RUN"), angle_cell
        else:
            assert float(angle_cell) == pytest.approx(
                float(run["repose_angle_deg"]), abs=5e-3), (
                    "report angle %s != measured %s" % (angle_cell, run["repose_angle_deg"]))


def test_report_verdict_matches_the_measurements():
    files = _shard_files()
    if not files or not os.path.exists(DOC):
        pytest.skip("need both the shard evidence and the rendered report")
    doc = open(DOC).read()
    runs = {float(_load(f)["runs"][0]["friction"]): _load(f) for f in files}
    all_pass = (len(runs) == len(FRICTION_GRID)
                and all(d["passed"] for d in runs.values()))
    verdict_line = [ln for ln in doc.splitlines() if ln.startswith("**C2 判定")]
    assert verdict_line, "report has no machine-readable C2 verdict line"
    stated = verdict_line[0]
    if all_pass:
        assert "PASS" in stated and "FAIL" not in stated
    else:
        assert "PASS" not in stated.replace("部分/honest FAIL", "") or "FAIL" in stated