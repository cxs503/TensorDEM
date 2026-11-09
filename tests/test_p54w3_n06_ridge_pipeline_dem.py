"""Node p54w3-n06: ridge-keel grounding on a buried pipeline (DEM vs theory).

Three groups of tests:

* the **adjudication algebra** - the energy partition, the closure scan, the
  acceptance evaluator - checked on synthetic records with a known answer;
* a **short DEM smoke run** that exercises the full case -> run -> payload
  chain (the heavy campaign itself lives in
  ``scripts/p54w3_n06_campaign.py``);
* an **artifact regression** on ``artifacts/p54w3_n06/campaign.json``: the
  numbers the report quotes must still satisfy the thresholds the campaign node
  defines.
"""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from tensordem.ridge_pipeline_dem import (ACCEPTANCE_THRESHOLDS, PipelineCase,
                                          case_scenario, closure_from_track,
                                          closure_scan, dt_refinement_cases,
                                          evaluate_acceptance, run_case)
from tensordem.ridge_pipeline_dem.energy import dt_refinement_table

ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "artifacts", "p54w3_n06", "campaign.json")


# --------------------------------------------------------------- synthetic tracks

def _sample(**kwargs) -> dict:
    base = {"time": 0.0, "platen_work": 0.0, "body_work": 0.0,
            "exchanged_pipe": 0.0, "exchanged_soil": 0.0, "exchanged_plane": 0.0,
            "dissipated": 0.0, "bond_fracture": 0.0, "kinetic": 0.0,
            "elastic_bond": 0.0, "elastic_contact": 0.0, "elastic_soil": 0.0}
    base.update(kwargs)
    return base


def _closed_track(*, dissipated: float = 4.0, soil_absorbed: float = 3.0,
                  soil_storage: float = 1.0, pipe_absorbed: float = 2.0) -> list[dict]:
    """Track whose corrected partition closes exactly and whose legacy one does not.

    Sources 10 J = dissipation 4 + A_soil 3 + A_pipe 2 + dK 1.  Of the 3 J the
    subgrade absorbed, 1 J is reversible (``elastic_soil``) and therefore
    already inside ``A_soil``: the corrected closure must not add it again.
    """
    head = _sample()
    tail = _sample(time=1.0, platen_work=8.0, body_work=2.0, dissipated=dissipated,
                   exchanged_soil=-soil_absorbed, exchanged_pipe=-pipe_absorbed,
                   kinetic=1.0, elastic_soil=soil_storage)
    return [head, tail]


def test_corrected_partition_closes_exactly():
    track = _closed_track()
    corrected = closure_from_track(track)
    assert corrected["closure_error"] == pytest.approx(0.0, abs=1e-12)
    assert corrected["sinks"]["soil_storage_double_count"] == 0.0
    assert corrected["closed"] is True


def test_legacy_partition_double_counts_the_subgrade_storage():
    track = _closed_track()
    legacy = closure_from_track(track, partition="legacy")
    corrected = closure_from_track(track)
    sinks = legacy["sinks"]
    assert sinks["soil_storage_double_count"] == pytest.approx(1.0)
    assert legacy["closure_error"] > corrected["closure_error"]
    scale = max(abs(sinks["total"]), 1.0)
    assert legacy["closure_error"] == pytest.approx(1.0 / scale, rel=1e-12)


def test_fracture_energy_is_a_sink():
    head = _sample()
    tail = _sample(time=1.0, platen_work=6.0, dissipated=4.0, bond_fracture=2.0)
    assert closure_from_track([head, tail])["closure_error"] == pytest.approx(0.0)
    without = closure_from_track([head, tail], include_fracture=False)
    assert without["closure_error"] == pytest.approx(2.0 / 6.0)


def test_partition_and_window_validation():
    track = _closed_track()
    with pytest.raises(ValueError):
        closure_from_track(track, partition="creativity")
    with pytest.raises(ValueError):
        closure_from_track(track, start=1, end=1)
    with pytest.raises(ValueError):
        closure_from_track([_sample()])
    scan = closure_scan(track * 4, fractions=(0.0, 0.5))
    assert [row["fraction"] for row in scan] == [0.0, 0.5]
    assert all(row["closed"] for row in scan)


def test_dt_refinement_table_reports_the_refinement():
    # coarse: 1 J of the 10 J input is unaccounted; fine: the budget closes.
    coarse = [_sample(), _sample(time=1.0, platen_work=10.0, dissipated=4.0,
                                 exchanged_soil=-3.0, exchanged_pipe=-2.0,
                                 kinetic=1.0, elastic_soil=1.0)]
    coarse[1]["dissipated"] = 3.0
    fine = [_sample(), _sample(time=1.0, platen_work=10.0, dissipated=4.0,
                               exchanged_soil=-3.0, exchanged_pipe=-2.0,
                               kinetic=1.0, elastic_soil=1.0)]
    table = dt_refinement_table([{"dt": 4e-4, "track": coarse},
                                 {"dt": 1e-4, "track": fine}])
    assert table["rows"][0]["dt"] == 4e-4
    assert table["finest_whole_record"] < table["rows"][0]["whole_record"]["closure_error"]
    assert table["satisfied_loading_phase"] is True


# ------------------------------------------------------------------- case plumbing

def test_case_validation_and_scenario():
    case = PipelineCase("ok", cover_depth=0.5)
    scenario = case_scenario(case)
    assert scenario.cover_depth == 0.5
    assert scenario.track_energy is True
    assert scenario.stroke == pytest.approx(case.stroke_used)
    with pytest.raises(ValueError):
        PipelineCase("bad", cover_depth=-0.1)
    with pytest.raises(ValueError):
        PipelineCase("bad", loading="torsion")
    with pytest.raises(ValueError):
        PipelineCase("bad", pipe_response="elastic")
    with pytest.raises(ValueError):
        PipelineCase("bad", su=0.0)


def test_dt_ladder_is_a_fraction_of_the_recommended_step():
    ladder = dt_refinement_cases(PipelineCase("base", cover_depth=0.5),
                                 fractions=(1.0, 0.5))
    assert ladder[0].dt > ladder[1].dt
    assert ladder[0].dt == pytest.approx(2.0 * ladder[1].dt)


def test_short_dem_run_payload_structure():
    case = PipelineCase("smoke", keel_depth=3.0, keel_width=6.0,
                        steps_factor=0.05, max_steps=400)
    payload = run_case(case)
    for key in ("case", "dem", "iso", "comparison", "energy", "history"):
        assert key in payload
    assert payload["case"]["steps"] > 1
    assert payload["iso"]["vertical"]["n_c"] > 0.0
    assert math.isfinite(payload["comparison"]["vertical"]["deviation"])
    assert payload["energy"]["whole_record"]["sources"]["total"] is not None
    assert len(payload["history"]["time"]) == len(payload["history"]["pipe_load"])
    assert payload["dem"]["metrics_checks"]["finite_and_nonnegative"] is True


# ---------------------------------------------------------------- acceptance algebra

def _study(deviation: float, threshold: float) -> dict:
    return {"rows": [{"case": "c", "deviation": deviation}],
            "cases": 1, "max_abs_deviation": abs(deviation),
            "threshold": threshold, "satisfied": abs(deviation) < threshold,
            "worst_case": "c"}


def test_acceptance_requires_every_criterion():
    vertical = _study(0.05, 0.20)
    lateral = _study(0.10, 0.25)
    cover = {"satisfied": True, "consistent_predictions": 4, "cases": 4,
             "max_abs_capacity_deviation": 0.04}
    energy = {"satisfied": True, "finest_loading_phase": 0.02,
              "finest_whole_record": 0.30, "threshold": 0.05, "finest_dt": 1e-4}
    grid = {"satisfied": True, "max_abs_change": 0.03, "threshold": 0.05,
            "particle_ratio": 2.5}
    verdict = evaluate_acceptance(vertical=vertical, lateral=lateral, cover=cover,
                                  energy=energy, grid=grid)
    assert verdict["passed"] is True and verdict["failed"] == []

    grid_bad = dict(grid, satisfied=False)
    verdict = evaluate_acceptance(vertical=vertical, lateral=lateral, cover=cover,
                                  energy=energy, grid=grid_bad)
    assert verdict["passed"] is False
    assert verdict["failed"] == ["e_grid_independence"]
    assert verdict["thresholds"]["grid_change"] == ACCEPTANCE_THRESHOLDS["grid_change"]


# ----------------------------------------------------------------- artifact regression

@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="campaign artifact not built")
def test_campaign_artifact_thresholds():
    with open(ARTIFACT) as handle:
        campaign = json.load(handle)
    acceptance = campaign["acceptance"]
    criteria = acceptance["criteria"]
    thresholds = campaign["thresholds"]
    assert thresholds["vertical_deviation"] == 0.20
    assert thresholds["lateral_deviation"] == 0.25
    assert thresholds["energy_closure"] == 0.05
    assert thresholds["grid_change"] == 0.05
    # every reported number must be recomputable from the criterion block
    for name, key in (("a_vertical_vs_iso", "max_abs_deviation"),
                      ("b_lateral_vs_nc", "max_abs_deviation"),
                      ("e_grid_independence", "max_abs_change")):
        value = criteria[name][key]
        assert value is not None
        assert criteria[name]["satisfied"] == (value < criteria[name]["threshold"])
    assert acceptance["passed"] == all(value["satisfied"] for value in criteria.values())
    assert criteria["d_energy_closure"]["finest_closure_loading_phase"] is not None