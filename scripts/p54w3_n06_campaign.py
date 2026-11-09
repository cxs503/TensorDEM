"""Run the p54w3-n06 campaign and write its artifacts.

Usage (worktree root)::

    python scripts/p54w3_n06_campaign.py --out artifacts/p54w3_n06 --figs

The campaign is heavy (each DEM case is O(10^4) explicit steps of a bonded
lattice), so it is meant to be run once on a compute node; the produced
``campaign.json`` is the provenance of every number in
``docs/p54w3_n06_ridge_pipeline_dem.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from tensordem.ridge_pipeline_dem import (  # noqa: E402
    ACCEPTANCE_THRESHOLDS, PIPELINE_CASES, PipelineCase, case_scenario,
    cover_study, dt_refinement_cases, energy_study, evaluate_acceptance,
    grid_study, lateral_scenario_gap, lateral_study, run_case, vertical_study)


def _strip(payload: dict) -> dict:
    """Drop the bulky trajectory arrays that are only needed for the figures."""
    small = {key: value for key, value in payload.items()
             if key not in ("energy_track", "history")}
    if payload.get("snapshot"):
        snapshot = payload["snapshot"]
        small["snapshot"] = {"time": snapshot["time"],
                             "positions": snapshot["positions"],
                             "force_magnitude": snapshot["force_magnitude"],
                             "contacts": snapshot["contacts"],
                             "pipe_contacts": snapshot["pipe_contacts"],
                             "broken_bonds": snapshot["broken_bonds"]}
    history = payload.get("history", {})
    stride = max(1, len(history.get("time", [])) // 400)
    small["curves"] = {key: (value[::stride] if isinstance(value, list) else value)
                       for key, value in history.items()}
    return small


def _summary_block(title: str, rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    return [{key: row.get(key) for key in keys} for row in rows]


STUDY_KEYS = ("vertical", "lateral", "cover", "grid", "rigid_reference", "energy")
EXTRA_KEYS = ("lateral_default_mode",)


def run(out_dir: str, *, make_figures: bool, quick: bool = False,
        only: tuple[str, ...] | None = None) -> dict:
    """Run the campaign; ``only`` recomputes a subset and merges into the existing file.

    The DEM cases are expensive, so the driver is restartable: with
    ``only=("lateral",)`` it recomputes just that study and merges the result
    into ``campaign.json`` (the other studies keep their audited numbers).
    """
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "campaign.json")
    campaign = {}
    if only and os.path.exists(path):
        with open(path) as handle:
            campaign = json.load(handle)
    cases = PIPELINE_CASES()
    if "lateral_default_mode" in cases:
        pass
    clock = {}
    payload_by_case = {}

    def timed(fn, label, *args, **kwargs):
        start = time.time()
        value = fn(*args, **kwargs)
        clock[label] = time.time() - start
        print(f"[campaign] {label}: {clock[label]:.1f}s", flush=True)
        for payload in value.get("payloads", []):
            payload_by_case[payload["case"]["name"]] = payload
        return value

    studies = {}
    wanted = set(only) if only else set(STUDY_KEYS)
    if "vertical" in wanted:
        studies["vertical"] = timed(vertical_study, "vertical", cases["vertical"])
    if "lateral" in wanted:
        studies["lateral"] = timed(lateral_study, "lateral", cases["lateral"])
        gap_case = cases["lateral_default_mode"][0]
        gap = lateral_scenario_gap(gap_case)
        payload_by_case[gap_case.name] = gap.pop("payload")
        studies["lateral"]["scenario_gap"] = gap
    if "cover" in wanted:
        studies["cover"] = timed(cover_study, "cover", cases["cover"])
    if "grid" in wanted:
        studies["grid"] = timed(grid_study, "grid",
                                (case.particle_radius for case in cases["grid"]),
                                base=cases["grid"][0])
    if "rigid_reference" in wanted:
        studies["rigid_reference"] = timed(vertical_study, "rigid_reference",
                                           cases["rigid_reference"])
    if "energy" in wanted:
        energy_cases = dt_refinement_cases(
            cases["vertical"][1],
            fractions=(1.0, 0.5, 0.25) if not quick else (1.0, 0.5))
        energy_entries = []
        start = time.time()
        for case in energy_cases:
            payload = run_case(case)
            payload_by_case[case.name] = payload
            energy_entries.append({"name": case.name, "dt": payload["case"]["dt"],
                                   "steps": payload["case"]["steps"],
                                   "track": payload["energy_track"]})
        clock["energy"] = time.time() - start
        print(f"[campaign] energy: {clock['energy']:.1f}s", flush=True)
        studies["energy"] = energy_study(energy_entries)

    results = campaign.get("results", {})
    for name, study in studies.items():
        results[name] = _strip_block(study)
    payloads = campaign.get("case_payloads", {})
    payloads.update({name: _strip(payload) for name, payload in payload_by_case.items()})
    study_for = {name: results[name] for name in STUDY_KEYS if name in results}
    acceptance = evaluate_acceptance(vertical=study_for["vertical"],
                                     lateral=study_for["lateral"],
                                     cover=study_for["cover"],
                                     energy=study_for["energy"],
                                     grid=study_for["grid"])

    runtime = dict(campaign.get("runtime_seconds", {}))
    runtime.update(clock)
    campaign |= {
        "node": "p54w3-n06-ridge-pipeline-dem",
        "repository": "TensorDEM",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": runtime,
        "thresholds": dict(ACCEPTANCE_THRESHOLDS),
        "cases": {key: [asdict(case) for case in value]
                  for key, value in cases.items()},
        "results": results,
        "acceptance": acceptance,
        "case_payloads": payloads,
    }
    with open(os.path.join(out_dir, "campaign.json"), "w") as handle:
        json.dump(campaign, handle, indent=1, sort_keys=False)
    print(f"[campaign] wrote {out_dir}/campaign.json", flush=True)

    if make_figures:
        from tensordem.ridge_pipeline_dem import figures  # pragma: no cover
        figures.make_all(campaign, os.path.join(out_dir, "figs"))
    return campaign


def _strip_block(study: dict) -> dict:
    return {key: value for key, value in study.items()
            if key not in ("payloads", "case_payloads")}


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/p54w3_n06")
    parser.add_argument("--figs", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="two-step energy ladder instead of three")
    parser.add_argument("--only", default="",
                        help="comma separated studies to recompute and merge, e.g. lateral")
    args = parser.parse_args()
    only = tuple(item.strip() for item in args.only.split(",") if item.strip()) or None
    result = run(args.out, make_figures=args.figs, quick=args.quick, only=only)
    print(json.dumps(result["acceptance"]["satisfied"], indent=1))