"""Acceptance criteria and the validation campaign of the broken-ice surface.

The campaign predeclares seven machine-checkable criteria (C1-C7) and reports
each one with the number behind it.  Verdicts are one of ``PASS``, ``FAIL`` (the
criterion was evaluated and the number misses) or ``WITHHELD`` (the criterion
cannot be evaluated with the data in the repository, with the data gap named).
Nothing is silently dropped: every criterion appears in every artifact.

Criteria
--------
C1  standard case (V = 3 m/s, h = 1.0 m, C_ice = 0.6): |R_DEM - R_Lindqvist| /
    R_Lindqvist <= 15 %.
C2  speed sweep V in [2, 5] m/s: R_DEM monotone increasing, Pearson r > 0.95.
C3  thickness sweep h in [0.5, 1.5] m: R_DEM monotone increasing, Pearson
    r > 0.95.
C4  MT Uikku broken-ice comparison: |R_DEM - R_Uikku| / R_Uikku <= 20 %.
C5  negative control C_ice -> 0: R_DEM -> 0 (field-free case exactly zero, and
    the sparse-field cases far below the standard case).
C6  end-to-end case: standard ship, V = 3 m/s, h = 1.0 m, C_ice = 0.6, with
    R_ice and both comparisons written to the artifact.
C7  pytest suite with >= 12 cases covering the standard case, the sweeps, the
    negative control and the MT Uikku anchor.

C1 is deliberately *not* softened.  Lindqvist is an intact-sheet formula and a
60 %-concentration broken field is physically expected to sit below it, so C1 is
the strict version of the cross-check; the like-for-like analytic reference
(the repository's broken-ice closed form) is reported next to it as diagnostic
D1 and the DEM/level-ice ratio is reported as D2.
"""
from __future__ import annotations

import ast
import math
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .anchors import (broken_ice_closed_form, lindqvist_anchor_scores,
                      lindqvist_reference, mt_uikku_level_ice_cases)
from .ledger import CASE_SPECS, STANDARD_CASE, CaseSpec, ledger_sha256
from .simulate import CaseResult, run_case

#: Criteria thresholds, predeclared *before* the campaign is run.
THRESHOLDS = {
    "C1_rel_error_vs_lindqvist": 0.15,
    "C2_pearson_r_speed": 0.95,
    "C3_pearson_r_thickness": 0.95,
    "C4_rel_error_vs_mt_uikku": 0.20,
    "C5_null_field_resistance_kN": 1.0e-9,
    "C5_sparse_ratio": 0.25,
    "C7_min_tests": 12,
}


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2 or n != len(y):
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    denominator = math.sqrt(sxx * syy)
    return sxy / denominator if denominator > 0 else float("nan")


def _monotone_increasing(values: Sequence[float]) -> bool:
    return all(b > a for a, b in zip(values, values[1:]))


def count_tests(path: str | Path) -> int:
    """Number of ``def test_*`` functions in a pytest module (AST, no import)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def evaluate_criteria(results: Mapping[str, CaseResult],
                      *, test_count: int,
                      tests_passed: bool = True,
                      test_path: str | None = None,
                      mt_uikku_rows: Iterable[dict[str, Any]] | None = None
                      ) -> dict[str, Any]:
    """Evaluate C1-C7 on the campaign results (pure function, no I/O)."""
    standard = results["standard"]
    lindqvist = lindqvist_reference(CASE_SPECS["standard"])
    closed_form = broken_ice_closed_form(CASE_SPECS["standard"])
    r_lind = lindqvist["total_kN"]
    c1_error = abs(standard.mean_kN - r_lind) / r_lind

    speed_keys = ["speed_2", "speed_2.5", "standard", "speed_3.5", "speed_4",
                  "speed_4.5", "speed_5"]
    thickness_keys = ["thick_0.5", "thick_0.75", "standard", "thick_1.25",
                      "thick_1.5"]
    speeds = [CASE_SPECS[k].speed_m_s for k in speed_keys]
    r_speed = [results[k].mean_kN for k in speed_keys]
    thickness = [CASE_SPECS[k].thickness_m for k in thickness_keys]
    r_thick = [results[k].mean_kN for k in thickness_keys]
    r_speed_pearson = _pearson(speeds, r_speed)
    r_thick_pearson = _pearson(thickness, r_thick)

    null_field = results.get("no_ice")
    sparse = results.get("sparse_ice")
    null_ok = (null_field is not None
               and abs(null_field.mean_kN) <= THRESHOLDS["C5_null_field_resistance_kN"])
    sparse_ok = (sparse is not None and standard.mean_kN > 0
                 and sparse.mean_kN / standard.mean_kN <= THRESHOLDS["C5_sparse_ratio"])

    # C4: the repository ships the ten MT Uikku *level-ice* cases only; the
    # paper's broken-ice channel cases (304-306, 404-406) are not transcribed.
    rows = list(mt_uikku_rows) if mt_uikku_rows is not None else mt_uikku_level_ice_cases()
    broken_rows = [r for r in rows if r.get("ice_type") == "broken"]
    c4 = {
        "verdict": "WITHHELD",
        "threshold": THRESHOLDS["C4_rel_error_vs_mt_uikku"],
        "value": None,
        "reason": ("no broken-ice experimental data in the repository: the shipped "
                   "MT Uikku table contains the ten level-ice cases only, the "
                   "paper's broken-ice channel cases (304-306, 404-406) are not "
                   "transcribed. A 60 %-concentration DEM field cannot be scored "
                   "against a level-ice measurement without confounding regime "
                   "and speed."),
        "available_level_ice_cases": len(rows),
        "available_broken_ice_cases": len(broken_rows),
        "data_gap": "benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv"
                    " -> add tests 304-306 and 404-406 (broken-ice channel)",
    }

    scores = lindqvist_anchor_scores(rows)
    verdicts = {
        "C1": {"value": c1_error, "threshold": THRESHOLDS["C1_rel_error_vs_lindqvist"],
               "dem_kN": standard.mean_kN, "lindqvist_kN": r_lind,
               "verdict": _verdict(c1_error <= THRESHOLDS["C1_rel_error_vs_lindqvist"]),
               "note": "level-ice formula used as the strict analytic cross-check"},
        "C2": {"value": r_speed_pearson,
               "threshold": THRESHOLDS["C2_pearson_r_speed"],
               "speeds_m_s": speeds, "resistance_kN": r_speed,
               "monotone": _monotone_increasing(r_speed),
               "verdict": _verdict(r_speed_pearson > THRESHOLDS["C2_pearson_r_speed"]
                                   and _monotone_increasing(r_speed)),
               "note": "Pearson r of R_ice against V over 2-5 m/s"},
        "C3": {"value": r_thick_pearson,
               "threshold": THRESHOLDS["C3_pearson_r_thickness"],
               "thickness_m": thickness, "resistance_kN": r_thick,
               "monotone": _monotone_increasing(r_thick),
               "verdict": _verdict(r_thick_pearson > THRESHOLDS["C3_pearson_r_thickness"]
                                   and _monotone_increasing(r_thick)),
               "note": "Pearson r of R_ice against h over 0.5-1.5 m"},
        "C4": c4,
        "C5": {"value": None if null_field is None else null_field.mean_kN,
               "threshold": THRESHOLDS["C5_null_field_resistance_kN"],
               "null_field_kN": None if null_field is None else null_field.mean_kN,
               "sparse_kN": None if sparse is None else sparse.mean_kN,
               "sparse_ratio": (None if sparse is None or standard.mean_kN == 0
                                else sparse.mean_kN / standard.mean_kN),
               "sparse_ratio_threshold": THRESHOLDS["C5_sparse_ratio"],
               "verdict": _verdict(null_ok and sparse_ok),
               "note": "field-free case must give exactly zero; a 10 % field must "
                       "stay below a quarter of the standard resistance"},
        "C6": {"verdict": _verdict(standard.mean_kN > 0.0),
               "value": standard.mean_kN,
               "comparisons": {"lindqvist_kN": r_lind,
                               "broken_ice_closed_form_kN": closed_form["total_kN"]},
               "note": "end-to-end standard case exported with both comparisons"},
        "C7": {"value": test_count, "threshold": THRESHOLDS["C7_min_tests"],
               "tests_passed": bool(tests_passed),
               "test_path": test_path,
               "verdict": _verdict(test_count >= THRESHOLDS["C7_min_tests"]
                                   and bool(tests_passed)),
               "note": "pytest suite of the validation surface"},
    }

    diagnostics = {
        "D1_broken_ice_closed_form": {
            "dem_kN": standard.mean_kN,
            "closed_form_kN": closed_form["total_kN"],
            "rel_error": (standard.mean_kN - closed_form["total_kN"]) / closed_form["total_kN"],
            "components_kN": {k: closed_form[k] / 1e3 for k in
                              ("crushing_N", "bending_N", "submersion_N", "momentum_N")
                              if k in closed_form},
            "note": ("like-for-like broken-ice reference (bending suppressed below the "
                     "bending length, momentum flux included); diagnostic, not a gate"),
        },
        "D2_dem_over_lindqvist": standard.mean_kN / r_lind,
        "D3_lindqvist_implementation_vs_published": scores,
        "D4_friction_power_share": standard.friction_power_share,
        "D5_stationarity_drift": standard.stationarity_drift,
        "D6_penalty_vs_crushing": None,
        "D7_block_vs_ship_hull": None,
        "D8_floe_size_sensitivity": None,
        "D9_dt_convergence": None,
        "D10_momentum_flux_scale": None,
        "D11_hydrostatic_submersion": None,
        "D12_seed_ensemble": None,
    }
    seeds = [results[k].mean_kN for k in
             ("standard", "standard_seed_11", "standard_seed_23") if k in results]
    if len(seeds) > 1:
        mean = sum(seeds) / len(seeds)
        spread = max(seeds) - min(seeds)
        diagnostics["D12_seed_ensemble"] = {
            "per_seed_kN": seeds,
            "ensemble_mean_kN": mean,
            "spread_kN": spread,
            "relative_spread": spread / mean if mean else None,
            "note": ("field realisations differ only by the placement seed; the "
                     "spread is the sampling error of an impact-dominated resistance"),
        }
    # analytic scales that bracket what a plan-view DEM can and cannot capture
    from .ledger import MATERIAL as _MATERIAL
    momentum_scale = (_MATERIAL["ice_density_kg_m3"] * STANDARD_CASE.concentration
                      * STANDARD_CASE.thickness_m * 26.0
                      * STANDARD_CASE.speed_m_s ** 2 / 1e3)
    diagnostics["D10_momentum_flux_scale"] = {
        "scale_kN": momentum_scale,
        "dem_kN": standard.mean_kN,
        "dem_over_scale": standard.mean_kN / momentum_scale,
        "definition": "rho_i C h B V^2: the momentum flux of the ice the hull "
                      "must deflect sideways (full-deflection upper scale)",
    }
    submersion = lindqvist["submersion_N"] / 1e3
    diagnostics["D11_hydrostatic_submersion"] = {
        "lindqvist_submersion_kN": submersion,
        "dem_kN": standard.mean_kN,
        "dem_over_submersion": standard.mean_kN / submersion,
        "note": "the intact-sheet hydrostatic submersion term has no counterpart "
                "in a plan-view DEM (floes never change their submerged volume); "
                "it is the largest single missing mechanism",
    }
    for key, name in (("D6_penalty_vs_crushing", "penalty_baseline"),
                      ("D7_block_vs_ship_hull", "block_hull_baseline")):
        baseline = results.get(name)
        if baseline is not None and standard.mean_kN:
            diagnostics[key] = {
                "baseline_kN": baseline.mean_kN,
                "primary_kN": standard.mean_kN,
                "ratio": standard.mean_kN / baseline.mean_kN,
                "baseline_case": name,
            }
    small = results.get("floe_small")
    large = results.get("floe_large")
    if small is not None and large is not None:
        diagnostics["D8_floe_size_sensitivity"] = {
            "small_floe_kN": small.mean_kN,
            "standard_kN": standard.mean_kN,
            "large_floe_kN": large.mean_kN,
            "small_over_large": (small.mean_kN / large.mean_kN) if large.mean_kN else None,
        }
    fine = results.get("dt_fine")
    if fine is not None and standard.mean_kN:
        diagnostics["D9_dt_convergence"] = {
            "dt_s": CASE_SPECS["dt_fine"].time_step_s,
            "fine_kN": fine.mean_kN,
            "standard_kN": standard.mean_kN,
            "rel_change": (fine.mean_kN - standard.mean_kN) / standard.mean_kN,
        }

    passed = sum(1 for v in verdicts.values() if v["verdict"] == "PASS")
    failed = sum(1 for v in verdicts.values() if v["verdict"] == "FAIL")
    withheld = sum(1 for v in verdicts.values() if v["verdict"] == "WITHHELD")
    return {
        "criteria": verdicts,
        "diagnostics": diagnostics,
        "summary": {"pass": passed, "fail": failed, "withheld": withheld,
                    "total": len(verdicts)},
        "ledger_sha256": ledger_sha256(),
        "thresholds": dict(THRESHOLDS),
    }


# ---------------------------------------------------------------------------
# campaign driver
# ---------------------------------------------------------------------------
def _worker(item: tuple[str, CaseSpec]) -> tuple[str, CaseResult]:
    name, spec = item
    return name, run_case(spec)


def run_campaign(names: Sequence[str] | None = None, *, jobs: int = 16,
                 progress: Callable[[str], None] | None = None
                 ) -> dict[str, CaseResult]:
    """Run the named cases in parallel (one process per case, one torch thread)."""
    selected = [(name, CASE_SPECS[name]) for name in (names or CASE_SPECS)]
    results: dict[str, CaseResult] = {}
    if jobs <= 1:
        for item in selected:
            name, result = _worker(item)
            results[name] = result
            if progress:
                progress(f"{name}: {result.mean_kN:.1f} kN")
        return results
    context = mp.get_context("fork")
    with context.Pool(processes=min(jobs, len(selected))) as pool:
        for name, result in pool.imap_unordered(_worker, selected):
            results[name] = result
            if progress:
                progress(f"{name}: {result.mean_kN:.1f} kN "
                         f"({result.wall_time_s:.0f} s)")
    return {name: results[name] for name, _ in selected if name in results}