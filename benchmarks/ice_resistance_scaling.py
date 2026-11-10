"""Broken-ice resistance scale law: unit-width DEM -> full ship (P121W5-N04).

Two stages, both re-runnable:

``sweep``    108-case full factorial (B x h x v x c) plus the convergence and
             negative-control runs; one JSON record per case under
             ``runs/ice_resistance_scaling/cases/``.
``analysis`` reads those records (never re-runs the DEM) and writes the fit,
             the ship-scale comparison, the FSD transfer, the criteria verdicts
             and the figures into ``artifacts/ice_resistance_scaling/``.

Usage
-----
``python benchmarks/ice_resistance_scaling.py --stage sweep --jobs 96``
``python benchmarks/ice_resistance_scaling.py --stage analysis``
``python benchmarks/ice_resistance_scaling.py --stage all --jobs 96``
``python benchmarks/ice_resistance_scaling.py --stage all --jobs 8 --fast``

``--fast`` runs a 3 x 2 x 2 x 2 = 24 case subset at a coarser time step; it is
only for CI smoke, and the artifact records that it was a fast run.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

RUN_DIR = _REPO / "runs" / "ice_resistance_scaling"
CASE_DIR = RUN_DIR / "cases"
OUT_DIR = _REPO / "artifacts" / "ice_resistance_scaling"
FIG_DIR = OUT_DIR

BASE_CASE = dict(beam_m=0.4, thickness_m=0.5, speed_m_s=1.0, concentration=0.7)
#: convergence / control runs use the fastest dynamically-legal setting (v = 2)
#: so that the extra runs do not dominate the wall clock; the choice is recorded
CTRL_SPEED = 2.0
CTRL_DOMAIN_LENGTH_RATIO = 6.0


# ---------------------------------------------------------------------------
# task design
# ---------------------------------------------------------------------------

def build_tasks(fast: bool) -> list[dict]:
    from tensordem.ice_resistance_scaling import (
        BEAM_LEVELS_M, CONCENTRATION_LEVELS, SPEED_LEVELS_M_S, THICKNESS_LEVELS_M)

    beams = (0.2, 0.4, 1.6) if fast else BEAM_LEVELS_M
    thicknesses = (0.3, 0.8) if fast else THICKNESS_LEVELS_M
    speeds = (0.5, 2.0) if fast else SPEED_LEVELS_M_S
    concentrations = (0.6, 0.8) if fast else CONCENTRATION_LEVELS
    dt_scale = 0.5 if fast else 0.25

    tasks: list[dict] = []
    for beam in beams:
        for thickness in thicknesses:
            for speed in speeds:
                for concentration in concentrations:
                    kwargs = dict(beam_m=beam, thickness_m=thickness,
                                  speed_m_s=speed,
                                  concentration=concentration,
                                  dt_scale=dt_scale)
                    label = f"case_B{beam:g}_h{thickness:g}_v{speed:g}_c{concentration:g}"
                    tasks.append({"kind": "case", "id": label, "kwargs": kwargs})
    # dt refinement (3 levels) at the fast control setting
    for scale, name in ((0.5, "dt1"), (0.25, "dt2"), (0.125, "dt4")):
        tasks.append({"kind": "case", "id": f"conv_dt_{name}",
                      "kwargs": dict(BASE_CASE, speed_m_s=CTRL_SPEED,
                                     dt_scale=scale)})
    # floe-mesh refinement (3 levels) at the fast control setting
    for mesh, name in ((0.5, "coarse"), (1.0, "base"), (1.5, "fine")):
        tasks.append({"kind": "case", "id": f"conv_mesh_{name}",
                      "kwargs": dict(BASE_CASE, speed_m_s=CTRL_SPEED,
                                     mesh_refinement=mesh, dt_scale=dt_scale)})
    # sigma_c sweep (crushing-strength negative control: saturation)
    for strength in (1.5e6, 1.5e7, 1.5e8):
        tasks.append({"kind": "case", "id": f"ctrl_strength_{strength:.0e}",
                      "kwargs": dict(BASE_CASE, speed_m_s=CTRL_SPEED,
                                     strength_pa=strength, dt_scale=dt_scale,
                                     domain_length_ratio=CTRL_DOMAIN_LENGTH_RATIO)})
    # concentration -> 0 (resistance must vanish with the ice)
    for concentration in (0.02, 0.1, 0.3):
        tasks.append({"kind": "case", "id": f"ctrl_concentration_{concentration:g}",
                      "kwargs": dict(BASE_CASE, speed_m_s=CTRL_SPEED,
                                     concentration=concentration, dt_scale=dt_scale,
                                     domain_length_ratio=CTRL_DOMAIN_LENGTH_RATIO)})
    # zero-ice structural control
    tasks.append({"kind": "zero_ice", "id": "ctrl_zero_ice",
                  "kwargs": dict(BASE_CASE, speed_m_s=CTRL_SPEED, dt_scale=dt_scale,
                                 domain_length_ratio=CTRL_DOMAIN_LENGTH_RATIO)})
    return tasks


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _decimate(series: list[float], target: int = 400) -> list[float]:
    if len(series) <= target:
        return [float(v) for v in series]
    stride = int(math.ceil(len(series) / target))
    return [float(v) for v in series[::stride]]


def run_task(task: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from tensordem.ice_resistance_scaling import UnitCase, simulate_unit_case, zero_ice_control

    case = UnitCase(**task["kwargs"])
    tic = time.perf_counter()
    if task["kind"] == "zero_ice":
        record = zero_ice_control(case)
    else:
        record = simulate_unit_case(case)
        record["R_series"] = _decimate(record["R_series"])
        record["time_series"] = _decimate(record["time_series"])
        record["diameters_after_m"] = _decimate(record["diameters_after_m"])
    record["task_id"] = task["id"]
    record["kind"] = task["kind"]
    record["wall_time_s"] = time.perf_counter() - tic
    path = CASE_DIR / f"{task['id']}.json"
    path.write_text(json.dumps(record, indent=1))
    return {"id": task["id"], "wall": record["wall_time_s"],
            "R": record.get("R_mean_N")}


def run_sweep(tasks: list[dict], jobs: int) -> None:
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    pending = [t for t in tasks if not (CASE_DIR / f"{t['id']}.json").exists()]
    print(f"  {len(tasks) - len(pending)} cases already on disk, "
          f"{len(pending)} to run", flush=True)
    tasks = pending
    if not tasks:
        return
    tic = time.perf_counter()
    done = 0
    if jobs <= 1:
        for task in tasks:
            run_task(task)
            done += 1
    else:
        with mp.get_context("spawn").Pool(processes=jobs) as pool:
            for _result in pool.imap_unordered(run_task, tasks):
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} cases "
                          f"({time.perf_counter() - tic:.0f} s)", flush=True)
    print(f"sweep done: {len(tasks)} tasks in {time.perf_counter() - tic:.0f} s")


def load_records() -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(CASE_DIR.glob("*.json"))}


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _convergence(records: dict[str, dict]) -> dict:
    def r(name: str) -> dict:
        return records[f"conv_dt_{name}"]

    dt_levels = [r("dt1")["case"]["dt_s"], r("dt2")["case"]["dt_s"],
                 r("dt4")["case"]["dt_s"]]
    dt_values = [r("dt1")["R_mean_N"], r("dt2")["R_mean_N"], r("dt4")["R_mean_N"]]
    mesh_levels = [records[f"conv_mesh_{m}"]["case"]["floe_radius_m"] * 2.0
                   for m in ("coarse", "base", "fine")]
    mesh_values = [records[f"conv_mesh_{m}"]["R_mean_N"]
                   for m in ("coarse", "base", "fine")]
    # dimensionless coefficient invariance across the four beam levels
    coefficients = []
    for label, rec in records.items():
        if "case" not in rec:
            continue
        case = rec["case"]
        if (rec["kind"] == "case"
                and abs(case["thickness_m"] - 0.5) < 1e-9
                and abs(case["speed_m_s"] - 1.0) < 1e-9
                and abs(case["concentration"] - 0.7) < 1e-9):
            coefficients.append(rec["dimensionless_C_strength"])
    return {
        "dt": {
            "production_scale": 0.25,
            "levels_s": dt_levels,
            "R_values_N": dt_values,
            "relative_change": abs(dt_values[0] - dt_values[2]) / dt_values[2],
            "definition": "|R(2 dt_prod) - R(dt_prod/2)| / R(dt_prod/2), production "
                          "dt_prod sits in the middle of the three levels",
        },
        "mesh": {
            "levels_m": mesh_levels,
            "R_values_N": mesh_values,
            "relative_change": abs(mesh_values[0] - mesh_values[2]) / mesh_values[2],
            "definition": "|R(coarse floe field) - R(fine floe field)| / R(fine)",
        },
        "beam_invariance": {
            "coefficients": coefficients,
            "relative_spread": ((max(coefficients) - min(coefficients))
                                / (sum(coefficients) / len(coefficients))),
            "definition": "R_unit / (sigma_c h B) over the four beam levels, "
                          "(max-min)/mean",
        },
    }


def _controls(records: dict[str, dict]) -> dict:
    strengths = sorted((float(rec["case"]["strength_pa"]), rec["R_mean_N"])
                       for name, rec in records.items() if name.startswith("ctrl_strength_"))
    if len(strengths) < 3:
        raise RuntimeError("strength sweep incomplete")
    (s2, r2), (s3, r3) = strengths[-2], strengths[-1]
    exponent = math.log(r3 / r2) / math.log(s3 / s2)
    concentrations = sorted((float(rec["case"]["concentration"]), rec["R_mean_N"])
                            for name, rec in records.items()
                            if name.startswith("ctrl_concentration_"))
    return {
        "zero_ice": records["ctrl_zero_ice"],
        "strength_sweep": {
            "strength_pa": [s for s, _ in strengths],
            "R_mean_N": [v for _, v in strengths],
            "saturation_exponent": exponent,
            "definition": "local d ln R / d ln sigma_c between the two stiffest "
                          "levels; -> 0 means the resistance saturates "
                          "(breaks the 'resistance is just stiffness' reading)",
            "strongest_over_weakest": strengths[-1][1] / strengths[0][1],
        },
        "concentration_sweep": {
            "concentration": [c for c, _ in concentrations],
            "R_mean_N": [v for _, v in concentrations],
            "definition": "resistance at vanishing ice concentration; the fit's "
                          "intercept forces R -> 0 as c -> 0 by construction, these "
                          "runs check that the DEM itself does so",
        },
    }


def _figures(result: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures: list[dict] = []
    rows = result["sweep"]

    def select(records, **criteria):
        out = []
        for rec in records:
            case = rec.get("case")
            if case is None:
                continue
            if all(abs(case[k] - v) < 1e-9 for k, v in criteria.items()):
                out.append(rec)
        return sorted(out, key=lambda r: (r["case"]["speed_m_s"], r["case"]["beam_m"]))

    # Fig 1: R_unit vs the four factors
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    ax = axes[0, 0]
    for speed in sorted({r["case"]["speed_m_s"] for r in rows}):
        subset = select(rows, thickness_m=0.5, concentration=0.7, speed_m_s=speed)
        ax.plot([r["case"]["beam_m"] for r in subset],
                [r["R_mean_N"] for r in subset], "o-", label=f"v={speed:g} m/s")
    ax.set_xlabel("model beam B [m]")
    ax.set_ylabel("$R_{unit}$ [N]")
    ax.set_title("(a) beam, h=0.5 m, c=0.7")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    for beam in sorted({r["case"]["beam_m"] for r in rows}):
        subset = select(rows, thickness_m=0.5, concentration=0.7, beam_m=beam)
        ax.plot([r["case"]["speed_m_s"] for r in subset],
                [r["R_mean_N"] for r in subset], "o-", label=f"B={beam:g} m")
    ax.set_xlabel("model speed v [m/s]")
    ax.set_ylabel("$R_{unit}$ [N]")
    ax.set_title("(b) speed, h=0.5 m, c=0.7")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    for thickness in sorted({r["case"]["thickness_m"] for r in rows}):
        subset = select(rows, beam_m=0.4, concentration=0.7, thickness_m=thickness)
        ax.plot([r["case"]["speed_m_s"] for r in subset],
                [r["R_mean_N"] for r in subset], "o-", label=f"h={thickness:g} m")
    ax.set_xlabel("model speed v [m/s]")
    ax.set_ylabel("$R_{unit}$ [N]")
    ax.set_title("(c) thickness, B=0.4 m, c=0.7")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    for concentration in sorted({r["case"]["concentration"] for r in rows}):
        subset = select(rows, beam_m=0.4, thickness_m=0.5, concentration=concentration)
        ax.plot([r["case"]["speed_m_s"] for r in subset],
                [r["R_mean_N"] for r in subset], "o-", label=f"c={concentration:g}")
    ax.set_xlabel("model speed v [m/s]")
    ax.set_ylabel("$R_{unit}$ [N]")
    ax.set_title("(d) concentration, B=0.4 m, h=0.5 m")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("Unit-width DEM resistance vs the four design factors (108 cases)")
    fig.tight_layout()
    path = FIG_DIR / "fig_parameters.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "R_unit vs beam, speed, thickness and concentration "
                               "of the unit-width DEM (log-log)."})

    # Fig 2: fit quality
    fit = result["fit"]
    predicted = [math.exp(v) for v in
                 (np.array(fit["observed_log_R"]) - np.array(fit["residuals"]))]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].scatter(fit["observed_log_R"], np.log(predicted), s=14, alpha=0.8)
    lo, hi = min(fit["observed_log_R"]), max(fit["observed_log_R"])
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1")
    axes[0].set_xlabel("ln R_unit observed")
    axes[0].set_ylabel("ln R_unit fitted")
    axes[0].set_title(f"(a) power-law fit, $R^2$ = {fit['r2_log']:.4f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].hist(fit["residuals"], bins=20, color="steelblue", edgecolor="k")
    axes[1].set_xlabel("log residual")
    axes[1].set_ylabel("count")
    axes[1].set_title("(b) residuals, RMS = "
                      f"{fit['rms_log_residual']:.3f}")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "fig_fit_residuals.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Scale-law fit: observed vs fitted resistance and "
                               "log-residual distribution (108 cases)."})

    # Fig 3: ship-scale comparison
    anchors = result["anchors"]
    labels = ["DEM scale law", "broken-ice anchor", "level-ice anchor (Lindqvist)"]
    values = [anchors["R_scale_N"] / 1e3, anchors["R_empirical_N"] / 1e3,
              anchors["R_level_ice_N"] / 1e3]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars = ax.bar(labels, values, color=["crimson", "steelblue", "grey"], width=0.6)
    ax.set_ylabel("resistance [kN]")
    ax.set_title("Full ship (L=150 m, B=22 m, v=5 m/s, h=0.5 m): "
                 f"$R_{{scale}}/R_{{emp}}$ = {anchors['R_scale_over_empirical']:.2f}")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    path = FIG_DIR / "fig_ship_comparison.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Scale-law prediction against the broken-ice and "
                               "level-ice engineering anchors for the 150 m ship."})

    # Fig 4: FSD transfer
    fsd = result["fsd"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    model = np.array(fsd["model_diameters_m"])
    ship = np.array(fsd["ship_reference_diameters_m"])
    for _sc, _col, _nm, _pv in [
            (fsd["hypotheses"]["geometric"]["scale_factor"], "tab:green",
             "geometric", fsd["hypotheses"]["geometric"]["p_value"]),
            (fsd["hypotheses"]["sqrt_length"]["scale_factor"], "tab:red",
             "$\\sqrt{L}$", fsd["hypotheses"]["sqrt_length"]["p_value"])]:
        factor, colour, label = _sc, _col, f"{_nm} x{_sc:.1f} (p={_pv:.3f})"
        axes[0].hist(model * factor, bins=18, histtype="step", lw=1.6, color=colour,
                     density=True, label=label)
    axes[0].hist(ship, bins=18, histtype="step", lw=1.6, color="k", density=True,
                 label="full-ship reference FSD")
    axes[0].set_xlabel("floe diameter [m]")
    axes[0].set_ylabel("normalised n(D)")
    axes[0].set_title("(a) scaled model FSD vs ship FSD")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    for factor, colour in ((fsd["hypotheses"]["geometric"]["scale_factor"], "tab:green"),
                           (fsd["hypotheses"]["sqrt_length"]["scale_factor"], "tab:red")):
        data = np.sort(model * factor)
        axes[1].plot(data, np.arange(1, data.size + 1) / data.size, color=colour, lw=1.6,
                     label=f"x{factor:.1f}")
    ship_sorted = np.sort(ship)
    axes[1].plot(ship_sorted, np.arange(1, ship_sorted.size + 1) / ship_sorted.size,
                 "k--", lw=1.4, label="ship reference")
    axes[1].set_xlabel("floe diameter [m]")
    axes[1].set_ylabel("empirical CDF")
    axes[1].set_title("(b) empirical CDFs")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "fig_fsd.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "FSD transfer: unit-width floe size distribution "
                               "scaled by the geometric rule and by sqrt(length) "
                               "against the full-ship reference distribution."})

    # Fig 5: energy budget (kept as a paper figure; criteria use the JSON)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    energy = result["energy"]
    terms = ["E_push", "E_break", "E_friction", "E_kinetic"]
    shares = [energy["median_shares"][t] for t in terms]
    ax.bar(terms, shares, color=["steelblue", "crimson", "darkorange", "grey"])
    ax.set_ylabel("share of W_ext (median over cases)")
    ax.set_title(f"Energy ledger closure: max residual "
                 f"{energy['closure_ratio_max'] * 100:.2f} % of $W_{{ext}}$")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = FIG_DIR / "fig_energy.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Median energy split of the 108 unit cases and the "
                               "worst-case closure residual."})
    return figures


def analyse(records: dict[str, dict], *, fast: bool, jobs: int) -> dict:
    from tensordem.ice_resistance_scaling import (
        ShipCase, broken_ice_anchor, evaluate_criteria, fsd_transfer,
        level_ice_anchor, power_law_fit, scale_to_ship)

    sweep = [rec for rec in records.values() if rec["kind"] == "case"
             and not rec["task_id"].startswith(("conv_", "ctrl_"))]
    sweep.sort(key=lambda r: r["task_id"])
    fit = power_law_fit(sweep)
    ship = ShipCase()
    reference = next(r for r in sweep
                     if abs(r["case"]["beam_m"] - 0.4) < 1e-9
                     and abs(r["case"]["thickness_m"] - 0.5) < 1e-9
                     and abs(r["case"]["speed_m_s"] - 1.0) < 1e-9
                     and abs(r["case"]["concentration"] - 0.7) < 1e-9)
    scale = scale_to_ship(fit=fit, unit_record=reference, ship=ship)
    empirical = broken_ice_anchor(ship)
    level = level_ice_anchor(ship)
    concentration_correction = ship.concentration ** fit["exponents"]["concentration"]["value"]
    r_concentration_corrected = empirical["total_N"] * concentration_correction
    anchors = {
        "ship_case": ship.as_dict(),
        "R_scale_N": scale["R_scale_N"],
        "R_empirical_N": empirical["total_N"],
        "R_empirical_concentration_corrected_N": r_concentration_corrected,
        "R_level_ice_N": level["total_N"],
        "R_scale_over_empirical": scale["R_scale_N"] / empirical["total_N"],
        "R_scale_over_empirical_corrected": scale["R_scale_N"] / r_concentration_corrected,
        "R_scale_over_level_ice": scale["R_scale_N"] / level["total_N"],
        "scale": scale,
        "broken_ice_anchor": empirical,
        "level_ice_anchor": level,
    }
    length_scale = ship.length_m / reference["case"]["hull_length_m"]
    fsd = fsd_transfer(model_diameters=reference["diameters_after_m"], ship=ship,
                       length_scale=length_scale, exponent=2.0)
    fsd["model_diameters_m"] = reference["diameters_after_m"]
    fsd["ship_reference_diameters_m"] = [float(v) for v in
                                         _ship_reference_fsd(ship, len(reference["diameters_after_m"]))]
    energy_records = [r for r in records.values() if "energy" in r]
    energy = {
        "cases": len(energy_records),
        "closure_ratio_max": max(r["energy"]["closure_ratio"] for r in energy_records),
        "closure_ratio_median": sorted(r["energy"]["closure_ratio"]
                                       for r in energy_records)[len(energy_records) // 2],
        "three_term_ratio_max": max(r["energy"]["three_term_ratio"]
                                    for r in energy_records),
        "median_shares": {
            term: _median([_share(r["energy"], term) for r in energy_records])
            for term in ("E_push", "E_break", "E_friction", "E_kinetic")
        },
    }
    convergence = _convergence(records)
    controls = _controls(records)
    criteria = evaluate_criteria(fit=fit, scale=scale, anchors=anchors, fsd=fsd,
                                 energy_records=energy_records,
                                 convergence=convergence, controls=controls)
    result = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fast_run": bool(fast),
        "host": {"platform": platform.platform(),
                 "processor": platform.processor(),
                 "python": sys.version.split()[0],
                 "cpu_count": os.cpu_count(), "jobs": jobs},
        "design": {
            "levels": {"beam_m": sorted({r["case"]["beam_m"] for r in sweep}),
                       "thickness_m": sorted({r["case"]["thickness_m"] for r in sweep}),
                       "speed_m_s": sorted({r["case"]["speed_m_s"] for r in sweep}),
                       "concentration": sorted({r["case"]["concentration"] for r in sweep})},
            "n_cases": len(sweep),
            "geometry": {"hull_length_over_beam": sweep[0]["case"]["hull_length_ratio"],
                         "floe_radius_over_beam": sweep[0]["case"]["floe_radius_ratio"],
                         "domain_over_beam": sweep[0]["case"]["domain_m"][0]
                         / sweep[0]["case"]["beam_m"]},
            "time_step": "dt = dt_scale * 0.5 sqrt(m_floe/k_n) (contact CFL), "
                         "dt_scale recorded per case",
        },
        "sweep": [{"case": r["case"], "R_mean_N": r["R_mean_N"],
                   "R_sem_N": r["R_sem_N"], "R_p95_N": r["R_p95_N"],
                   "contacts_mean": r["contacts_mean"],
                   "floe_count": r["floe_count"],
                   "realised_concentration": r["realised_concentration"],
                   "dimensionless_C_strength": r["dimensionless_C_strength"],
                   "cleared_width_m": r["cleared_width_m"],
                   "energy": r.get("energy", {})} for r in sweep],
        "fit": fit,
        "anchors": anchors,
        "fsd": fsd,
        "energy": energy,
        "convergence": convergence,
        "controls": controls,
        "criteria": criteria,
    }
    return result


def _ship_reference_fsd(ship, n: int) -> list[float]:
    from tensordem.ice_resistance_scaling import power_law_fsd
    return power_law_fsd(d_min_m=0.5 * ship.floe_diameter_m,
                         d_max_m=6.0 * ship.floe_diameter_m,
                         exponent=2.0, n=n, seed=7).tolist()


def _share(energy: dict, term: str) -> float:
    if term == "E_kinetic":
        numerator = energy.get("E_kinetic_final_J", 0.0)
    else:
        numerator = energy.get(f"{term}_J", 0.0)
    denom = max(abs(energy.get("W_ext_J", 0.0)), 1e-30)
    return numerator / denom


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sweep", "analysis", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=min(96, os.cpu_count() or 1))
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--shard", default="0/1",
                        help="i/n: run only tasks with index %% n == i (cluster split)")
    parser.add_argument("--force", action="store_true",
                        help="recompute cases that already have a record")
    args = parser.parse_args()

    if args.stage in ("sweep", "all"):
        tasks = build_tasks(args.fast)
        shard_index, shard_total = (int(v) for v in args.shard.split("/"))
        if shard_total > 1:
            tasks = [t for i, t in enumerate(tasks) if i % shard_total == shard_index]
        if args.force:
            for task in tasks:
                (CASE_DIR / f"{task['id']}.json").unlink(missing_ok=True)
        print(f"sweep: {len(tasks)} tasks (shard {args.shard}), {args.jobs} workers")
        run_sweep(tasks, args.jobs)

    if args.stage in ("analysis", "all"):
        records = load_records()
        if not records:
            raise SystemExit("no case records found: run --stage sweep first")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        result = analyse(records, fast=args.fast, jobs=args.jobs)
        figures = _figures(result)
        result["figures"] = figures
        (OUT_DIR / "ice_resistance_scaling.json").write_text(json.dumps(result, indent=1))
        (OUT_DIR / "fit.json").write_text(json.dumps(result["fit"], indent=1))
        (OUT_DIR / "criteria.json").write_text(json.dumps(result["criteria"], indent=1))
        print(json.dumps(result["criteria"], indent=1))
        print(f"artifacts -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())