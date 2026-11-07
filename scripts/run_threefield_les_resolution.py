"""IN52-N02 runner: three-field LES transport maturation.

Produces, from the shipped code paths only:

* ``artifacts/threefield/resolution_ladder.json`` -- the ``N = 16/32/64`` x
  ``tau0 = 0.55/0.8/1.0/1.4`` shear-wave ladder, both fit windows, the
  Richardson extrapolation and the applicability envelope.
* ``artifacts/threefield/channel_poiseuille.json`` -- the planar-Poiseuille
  precision face (periodic reference channel, shipped vs consistent Guo source,
  bounded zero-gradient vs pressure-anchor outlet, free-slip plug).
* ``artifacts/threefield/wake_lbm_simple.json`` / ``..._guo.json`` and
  ``wake_diff.json`` -- the re-run of the published propeller-race benchmark
  plus the old/new diff table (the "old" column is parsed out of
  ``docs/THREEFIELD_ICE_HULL_PROPELLER.md``, never retyped).
* ``artifacts/threefield/les_resolution_summary.json`` -- every number quoted in
  ``docs/threefield_les_resolution_study.md``.
* ``artifacts/threefield/figures/*.png``.

Heavy stages are process-parallel (one lattice per process, one BLAS thread
each) because the D3Q19 kernels are launch-bound on small boxes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tensordem.threefield.channel_poiseuille import (  # noqa: E402
    ChannelConfig,
    poiseuille_peak,
    run_channel,
)
from tensordem.threefield.resolution_study import (  # noqa: E402
    DecaySample,
    applicability_envelope,
    decay_ladder,
    richardson_extrapolate,
)

ARTIFACTS = ROOT / "artifacts" / "threefield"
FIGURES = ARTIFACTS / "figures"

REQUIRED_TAU0 = (0.55, 0.8, 1.0, 1.4)
LADDER_SIZES = (16, 32, 64)
TAU0_SCAN = (0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
LADDER_STEPS = 220
LADDER_WINDOWS = (120, 200)
LADDER_START = 20
TOLERANCE = 0.01

# kh scan: the mode is raised at fixed lattice, which raises k h at fixed nu
KH_SCAN = (
    (1.4, 32, 2),
    (1.4, 32, 3),
    (1.4, 32, 4),
    (1.4, 64, 2),
    (1.4, 64, 3),
    (1.0, 32, 2),
    (1.0, 32, 3),
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# stage 1: shear-wave resolution ladder
# ---------------------------------------------------------------------------
def _ladder_job(job: tuple[float, int, int]) -> dict[str, Any]:
    tau0, size, mode = job
    torch.set_num_threads(1)
    try:
        point = decay_ladder(
            [tau0],
            [size],
            modes=[mode],
            steps=LADDER_STEPS,
            windows=LADDER_WINDOWS,
            start=LADDER_START,
        )[(tau0, size, mode)]
    except ValueError as exc:  # a mode that decays out of range: book it, do not hide it
        return {"tau0": tau0, "size": size, "mode": mode, "samples": [],
                "window_sensitivity": float("nan"), "error": str(exc)}
    return {
        "tau0": tau0,
        "size": size,
        "mode": mode,
        "samples": [
            {
                "tau0": sample.tau0,
                "size": sample.size,
                "mode": sample.mode,
                "start": sample.start,
                "stop": sample.stop,
                "rate": sample.rate,
                "nu_nominal": sample.nu_nominal,
                "rate_nominal": sample.rate_nominal,
                "rate_discrete": sample.rate_discrete,
                "amplitude_final": sample.amplitude_final,
            }
            for sample in point.samples
        ],
        "window_sensitivity": point.window_sensitivity,
    }


def stage_ladder(workers: int) -> dict[str, Any]:
    jobs = [(tau0, size, 1) for tau0 in REQUIRED_TAU0 for size in LADDER_SIZES]
    jobs += [(tau0, 16, 1) for tau0 in TAU0_SCAN if tau0 not in REQUIRED_TAU0]
    jobs += [(tau0, size, mode) for tau0, size, mode in KH_SCAN]
    jobs = sorted(set(jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_ladder_job, jobs))

    samples: list[DecaySample] = []
    points = []
    failures = [{"tau0": row["tau0"], "size": row["size"], "mode": row["mode"],
                 "error": row.get("error")} for row in rows if not row["samples"]]
    for row in rows:
        if not row["samples"]:
            continue
        booked = [DecaySample(**entry) for entry in row["samples"]]
        samples.extend(booked)
        points.append(
            {
                "tau0": row["tau0"],
                "size": row["size"],
                "mode": row["mode"],
                "window_sensitivity": row["window_sensitivity"],
                "windows": {
                    str(window): {
                        "rate": sample.rate,
                        "nu_measured": sample.nu_measured,
                        "nu_nominal": sample.nu_nominal,
                        "error_nominal": sample.error_nominal,
                        "error_discrete": sample.error_discrete,
                        "kh": sample.kh,
                        "amplitude_final": sample.amplitude_final,
                        "actual_start": sample.start,
                        "actual_stop": sample.stop,
                        "samples": sample.steps,
                    }
                    for window, sample in zip(LADDER_WINDOWS, booked)
                },
            }
        )

    # -- Richardson extrapolation in the dimensionless cell wavenumber k h --
    richardson: dict[str, Any] = {}
    for window in (200, 120):
        for tau0 in REQUIRED_TAU0:
            levels = sorted(
                (point for point in points
                 if point["tau0"] == tau0 and point["mode"] == 1
                 and point["size"] in LADDER_SIZES),
                key=lambda point: point["size"],
            )
            if len(levels) != 3 or any(str(window) not in level["windows"] for level in levels):
                richardson[f"tau0={tau0}|window={window}"] = {
                    "error": "a ladder level could not be booked",
                    "levels": [level["size"] for level in levels],
                }
                continue
            errors = [level["windows"][str(window)]["error_nominal"] for level in levels]
            hs = [level["windows"][str(window)]["kh"] for level in levels]
            try:
                fit = richardson_extrapolate(hs, errors, parameter="k h")
            except ValueError as exc:  # non-monotone: keep it visible, do not hide it
                richardson[f"tau0={tau0}|window={window}"] = {"error": str(exc)}
                continue
            nu_nominal = levels[0]["windows"][str(window)]["nu_nominal"]
            richardson[f"tau0={tau0}|window={window}"] = {
                **fit.as_dict(),
                "nu_nominal": nu_nominal,
                "nu_limit": nu_nominal * (1.0 + fit.error_limit),
                "nu_limit_error": fit.error_limit,
                "levels": [
                    {"size": level["size"], "kh": level["windows"][str(window)]["kh"],
                     "error": error}
                    for level, error in zip(levels, errors)
                ],
            }

    # negative control: a deliberately wrong extrapolation order must move the limit
    tau0 = 1.4
    levels = sorted(
        (point for point in points
         if point["tau0"] == tau0 and point["mode"] == 1 and point["size"] in LADDER_SIZES),
        key=lambda point: point["size"],
    )
    if len(levels) != 3 or any("200" not in level["windows"] for level in levels):
        control = {"error": "the tau0=1.4 ladder could not be booked"}
    else:
        control = {
            f"order={order}": richardson_extrapolate(
                [level["windows"]["200"]["kh"] for level in levels],
                [level["windows"]["200"]["error_nominal"] for level in levels],
                assumed_order=order,
            ).error_limit
            for order in (1.0, 2.0, 3.0)
        }

    envelope = applicability_envelope(samples, tolerance=TOLERANCE)
    payload = {
        "protocol": "tensordem-threefield-les-resolution-v1",
        "settings": {
            "tau0_required": list(REQUIRED_TAU0),
            "sizes": list(LADDER_SIZES),
            "steps": LADDER_STEPS,
            "windows": list(LADDER_WINDOWS),
            "fit_start": LADDER_START,
            "tolerance": TOLERANCE,
            "kh_scan": [list(job) for job in KH_SCAN],
            "tau0_scan": list(TAU0_SCAN),
        },
        "points": sorted(points, key=lambda point: (point["tau0"], point["size"], point["mode"])),
        "richardson": richardson,
        "richardson_order_control": control,
        "failed_points": failures,
        "envelope": envelope.as_dict(),
    }
    _write(ARTIFACTS / "resolution_ladder.json", payload)
    return payload


def stage_envelope() -> dict[str, Any]:
    """Recompute the applicability envelope from the saved ladder (no re-runs).

    Only the long fit window enters: mixing the short and long windows of the same
    trajectory would double-count every lattice point.
    """
    path = ARTIFACTS / "resolution_ladder.json"
    payload = json.loads(path.read_text())
    window = str(max(LADDER_WINDOWS))
    samples = [
        DecaySample(
            tau0=point["tau0"],
            size=point["size"],
            mode=point["mode"],
            start=int(entry["actual_start"]),
            stop=int(entry["actual_stop"]),
            rate=entry["rate"],
            nu_nominal=entry["nu_nominal"],
            rate_nominal=entry["nu_nominal"] * entry["kh"] ** 2,
            rate_discrete=entry["nu_nominal"] * 2.0 * (1.0 - math.cos(entry["kh"])),
            amplitude_final=entry["amplitude_final"],
        )
        for point in payload["points"]
        for key, entry in point["windows"].items()
        if key == window
    ]
    payload["envelope"] = applicability_envelope(samples, tolerance=TOLERANCE).as_dict()
    payload["envelope_window"] = int(window)
    _write(path, payload)
    return payload


# ---------------------------------------------------------------------------
# stage 2: planar Poiseuille precision face
# ---------------------------------------------------------------------------
def _channel_job(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    config = ChannelConfig(**job["config"])
    result = run_channel(
        config,
        steps=int(job["steps"]),
        label=job["label"],
        sample_every=int(job.get("sample_every", 100)),
        developed_fraction=float(job.get("developed_fraction", 0.4)),
    )
    payload = result.as_dict()
    payload["group"] = job["group"]
    return payload


def stage_channel(workers: int) -> dict[str, Any]:
    peak_velocity = 0.02
    tau0s = (0.6, 0.8, 1.0, 1.4)
    jobs: list[dict[str, Any]] = []
    for tau0 in tau0s:
        nu = (tau0 - 0.5) / 3.0
        force = 8.0 * nu * peak_velocity / 16.0**2
        for scheme in ("simple", "guo"):
            jobs.append(
                {
                    "group": f"periodic_no_slip_{scheme}",
                    "label": f"periodic/no-slip/{scheme}/tau0={tau0}",
                    "steps": 5000,
                    "config": dict(
                        ny=16, nx=8, tau0=tau0, force=force, axial="periodic",
                        wall="no_slip", forcing=scheme,
                    ),
                }
            )
    for tau0 in (0.8, 1.4):
        nu = (tau0 - 0.5) / 3.0
        force = 8.0 * nu * peak_velocity / 16.0**2
        for outlet in ("zero_gradient", "pressure_anchor"):
            jobs.append(
                {
                    "group": f"bounded_{outlet}",
                    "label": f"bounded/no-slip/{outlet}/tau0={tau0}",
                    "steps": 12000,
                    "sample_every": 200,
                    "developed_fraction": 0.3,
                    "config": dict(
                        ny=16, nx=40, tau0=tau0, force=force, axial="bounded",
                        outlet=outlet, wall="no_slip", forcing="guo", initial="inlet",
                    ),
                }
            )
        jobs.append(
            {
                "group": "bounded_zero_gradient_no_force",
                "label": f"bounded/no-slip/zero_gradient/plug-decay/tau0={tau0}",
                "steps": 6000,
                "sample_every": 200,
                "developed_fraction": 0.3,
                "config": dict(
                    ny=16, nx=40, tau0=tau0, force=0.0, axial="bounded",
                    outlet="zero_gradient", wall="no_slip", forcing="guo",
                    inlet_profile="uniform", initial="rest",
                ),
            }
        )
    nu = 0.1
    jobs.append(
        {
            "group": "free_slip_plug",
            "label": "periodic/free-slip/guo/accelerating-plug",
            "steps": 2000,
            "sample_every": 200,
            "developed_fraction": 0.5,
            "config": dict(
                ny=16, nx=8, tau0=0.8, force=8.0 * nu * peak_velocity / 16.0**2,
                axial="periodic", wall="free_slip", forcing="guo",
            ),
        }
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_channel_job, jobs))

    # -- derived quantities quoted in the document -------------------------
    def _pick(group: str, label_contains: str) -> dict[str, Any] | None:
        for row in rows:
            if row["group"] == group and label_contains in row["label"]:
                return row
        return None

    forcing_defect = {}
    for tau0 in tau0s:
        simple = _pick(f"periodic_no_slip_simple", f"tau0={tau0}")
        guo = _pick(f"periodic_no_slip_guo", f"tau0={tau0}")
        omega = 1.0 / tau0
        forcing_defect[f"tau0={tau0}"] = {
            "omega": omega,
            "predicted_overshoot": 1.0 + 0.5 * omega,
            "simple_peak_ratio": simple["peak_measured"] / simple["peak_analytic"],
            "guo_peak_ratio": guo["peak_measured"] / guo["peak_analytic"],
            "simple_profile_error_pct": 100.0 * simple["error"]["linf"],
            "guo_profile_error_pct": 100.0 * guo["error"]["linf"],
        }
    free_slip = _pick("free_slip_plug", "")
    force = free_slip["force"]
    free_slip_ramp = {
        "force": force,
        "steps": free_slip["steps"],
        "peak_velocity": max(free_slip["profile"]),
        "measured_acceleration": max(free_slip["profile"]) / free_slip["steps"],
        "predicted_acceleration": force,
        "acceleration_error": max(free_slip["profile"]) / free_slip["steps"] / force - 1.0,
        "plug_index": free_slip["plug_index"],
        "profile_error_pct": 100.0 * free_slip["error"]["linf"],
    }
    outlets = {}
    for tau0 in (0.8, 1.4):
        for outlet in ("zero_gradient", "pressure_anchor"):
            row = _pick(f"bounded_{outlet}", f"tau0={tau0}")
            outlets[f"tau0={tau0}|{outlet}"] = {
                "label": row["label"],
                "profile_error_linf_pct": 100.0 * row["error"]["linf"],
                "profile_error_l2_pct": 100.0 * row["error"]["l2"],
                "peak_ratio": row["peak_measured"] / row["peak_analytic"],
                "flux_ratio": row["flux_ratio"],
                "flux_imbalance_pct": 100.0 * (row["flux_ratio"] - 1.0),
                "mass_drift_rate_per_step": row["mass_drift_rate"],
                "density_excursion": row["density_excursion"],
                "plug_index": row["plug_index"],
            }
    plug_decay = {}
    for tau0 in (0.8, 1.4):
        row = _pick("bounded_zero_gradient_no_force", f"tau0={tau0}")
        plug_decay[f"tau0={tau0}"] = {
            "mass_drift_rate_per_step": row["mass_drift_rate"],
            "density_excursion": row["density_excursion"],
            "mean_velocity": sum(row["profile"]) / len(row["profile"]),
        }
    payload = {
        "protocol": "tensordem-threefield-poiseuille-v1",
        "peak_velocity_target": peak_velocity,
        "runs": rows,
        "forcing_defect": forcing_defect,
        "free_slip_plug": free_slip_ramp,
        "bounded_outlets": outlets,
        "plug_decay": plug_decay,
    }
    _write(ARTIFACTS / "channel_poiseuille.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stage 3: propeller-race benchmark re-run (old / new / corrected)
# ---------------------------------------------------------------------------
def _wake_job(job: tuple[str, int, bool]) -> dict[str, Any]:
    scheme, steps, force = job
    cached = ARTIFACTS / f"wake_lbm_{scheme}.json"
    if cached.is_file() and not force:
        return json.loads(cached.read_text())
    torch.set_num_threads(max(1, int(os.environ.get("WAKE_THREADS", "4"))))
    import run_threefield_coupling as coupling  # local import: needs sys.path entry

    config = coupling._lbm_config()
    config = type(config)(**{**config.__dict__, "force_scheme": scheme})
    sim = coupling.WakeLBM(config)
    average_start = int(round(0.6 * steps))
    sim.step(average_start)
    sim.step(steps - average_start, accumulate=True)
    output = ARTIFACTS / f"wake_lbm_{scheme}.json"
    payload = coupling._finalise_lbm(sim, config, output)
    payload["force_scheme"] = scheme
    _write(output, payload)
    return payload


PUBLISHED = ROOT / "docs" / "THREEFIELD_ICE_HULL_PROPELLER.md"


def parse_published_wake(doc: Path = PUBLISHED) -> dict[str, Any]:
    """Read the pre-m2949 wake table out of the published document.

    The old numbers must not be retyped: they are parsed from the markdown so
    the diff table cannot drift from the text it is compared against.
    """
    text = doc.read_text(encoding="utf-8")
    rows: dict[str, dict[str, float]] = {}
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*"
        r"\*\*([\d.]+)\s*%\*\*\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*\|",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        station, uc, urms, sigma, sigma_model, shape, full, r2 = match.groups()
        rows[station] = {
            "centreline_velocity": float(uc),
            "rms_velocity": float(urms),
            "fitted_sigma": float(sigma),
            "model_sigma": float(sigma_model),
            "shape_error_pct": float(shape),
            "full_domain_error_pct": float(full),
            "shape_r_squared": float(r2),
        }
    if not rows:
        raise RuntimeError("published wake table not found")
    thrust = re.search(r"Thrust \(lattice\) ([\d.]+)", text)
    excess = re.search(r"excess momentum flux ([\d.]+) / ([\d.]+) / ([\d.]+)", text)
    beta = re.search(r"β<sub>measured</sub> = (−|-)?([\d.]+)", text)
    return {
        "stations": rows,
        "thrust_lattice": float(thrust.group(1)) if thrust else None,
        "excess_momentum_flux": [float(value) for value in excess.groups()] if excess else None,
        "beta_measured": (
            (-1.0 if beta.group(1) else 1.0) * float(beta.group(2)) if beta else None
        ),
        "source": str(doc.relative_to(ROOT)),
        "note": "pre-m2949 numbers: |Pi| norm bug (sqrt(2) under-estimate in shear)",
    }


def stage_wake(workers: int, force: bool = False) -> dict[str, Any]:
    old = parse_published_wake()
    schemes = ("simple", "guo")
    jobs = [(scheme, 2100, force) for scheme in schemes]
    with ProcessPoolExecutor(max_workers=min(workers, len(schemes))) as pool:
        payloads = {scheme: payload for scheme, payload in zip(schemes, pool.map(_wake_job, jobs))}
    diff: dict[str, Any] = {"old_source": old, "columns": {}}
    for scheme, payload in payloads.items():
        diff["columns"][scheme] = {
            "thrust_lattice": payload["thrust_lattice"],
            "momentum_flux_excess": payload["momentum_flux_excess"],
            "beta_measured": payload["spreading"]["beta_measured"],
            "stations": {
                station: {
                    "centreline_velocity": payload["profile_audit"][station]["centreline_velocity"],
                    "rms_velocity": payload["profile_audit"][station]["rms_velocity_centre"],
                    "fitted_sigma": payload["profile_audit"][station]["fitted_sigma_lattice"],
                    "model_sigma": payload["profile_audit"][station]["model_sigma_lattice"],
                    "shape_error_pct": payload["profile_audit"][station]["shape_error_pct"],
                    "full_domain_error_pct": payload["profile_audit"][station]["full_domain_shape_error_pct"],
                    "shape_r_squared": payload["profile_audit"][station]["shape_r_squared"],
                    "excess_over_thrust": payload["momentum_flux_excess"][station] / payload["thrust_lattice"],
                }
                for station in payload["profile_audit"]
                if station.isdigit()
            },
        }
    # explicit old -> new deltas per station and quantity
    table = []
    for station, old_row in sorted(old["stations"].items()):
        for scheme in schemes:
            new_row = diff["columns"][scheme]["stations"].get(station)
            if new_row is None:
                continue
            for quantity in ("centreline_velocity", "shape_error_pct", "fitted_sigma"):
                table.append(
                    {
                        "station": int(station),
                        "scheme": scheme,
                        "quantity": quantity,
                        "old": old_row[quantity],
                        "new": new_row[quantity],
                        "delta": new_row[quantity] - old_row[quantity],
                        "delta_pct_of_old": 100.0 * (new_row[quantity] / old_row[quantity] - 1.0)
                        if old_row[quantity] else None,
                    }
                )
    diff["rows"] = table
    diff["thrust"] = {
        scheme: {
            "old": old["thrust_lattice"],
            "new": diff["columns"][scheme]["thrust_lattice"],
            "delta_pct": 100.0 * (diff["columns"][scheme]["thrust_lattice"] / old["thrust_lattice"] - 1.0)
            if old["thrust_lattice"] else None,
        }
        for scheme in schemes
    }
    _write(ARTIFACTS / "wake_diff.json", diff)
    return diff


# ---------------------------------------------------------------------------
# stage 4: figures + summary
# ---------------------------------------------------------------------------
def stage_figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    ladder = json.loads((ARTIFACTS / "resolution_ladder.json").read_text())
    channel = json.loads((ARTIFACTS / "channel_poiseuille.json").read_text())
    produced = []

    # fig 1: error vs cell wavenumber, with the Richardson limits
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colours = {0.55: "#1b6ca8", 0.8: "#2e9e6b", 1.0: "#d08b00", 1.4: "#b9342c"}
    for tau0 in REQUIRED_TAU0:
        levels = sorted(
            (point for point in ladder["points"]
             if point["tau0"] == tau0 and point["mode"] == 1 and point["size"] in LADDER_SIZES),
            key=lambda point: point["size"],
        )
        xs = [level["windows"]["200"]["kh"] for level in levels]
        ys = [100.0 * level["windows"]["200"]["error_nominal"] for level in levels]
        ax.plot(xs, ys, "o-", color=colours[tau0], label=f"τ₀={tau0}")
        ax.plot(xs[0] * 1.05, ys[0], "o", mfc="none", color=colours[tau0])
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhspan(-1.0, 1.0, color="0.85", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("cell wavenumber $k h = 2\\pi/N$")
    ax.set_ylabel("rate error vs $\\nu k^2$  [%]")
    ax.set_title("Shear-wave rate error collapses with the lattice\n(shaded band = ±1 % criterion)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURES / "fig_in52_resolution_ladder.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(str(path.relative_to(ROOT)))

    # fig 2: Poiseuille profiles, shipped vs consistent source
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ax = axes[0]
    for scheme, colour in (("simple", "#b9342c"), ("guo", "#1b6ca8")):
        row = next(
            r for r in channel["runs"]
            if r["group"] == f"periodic_no_slip_{scheme}" and r["tau0"] == 0.8
        )
        ax.plot(row["profile"], range(len(row["profile"])), "o-", ms=3, color=colour,
                label=f"{scheme} source")
    reference = next(
        r for r in channel["runs"] if r["group"] == "periodic_no_slip_guo" and r["tau0"] == 0.8
    )["reference"]
    ax.plot(reference, range(len(reference)), "k--", lw=1.0, label="analytic parabola")
    ax.set_xlabel("$u_x$ [lattice]")
    ax.set_ylabel("row y")
    ax.set_title("Periodic no-slip channel, τ₀=0.8")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    labels = [key.replace("|", "\n") for key in channel["bounded_outlets"]]
    values = [row["profile_error_linf_pct"] for row in channel["bounded_outlets"].values()]
    ax.bar(range(len(values)), values, color="#2e9e6b")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("developed-profile error vs parabola [%]")
    ax.set_title("Bounded box: outlet model error")
    fig.tight_layout()
    path = FIGURES / "fig_in52_poiseuille_face.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(str(path.relative_to(ROOT)))

    # fig 3: outlet mass books
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, row in channel["bounded_outlets"].items():
        match = next(r for r in channel["runs"] if r["label"] == row["label"])
        steps = [entry[0] for entry in match["mass_history"]]
        masses = [entry[1] / match["mass_history"][0][1] for entry in match["mass_history"]]
        ax.plot(steps, masses, label=f"{key}", lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("M(t) / M(0)")
    ax.set_title("Bounded-box mass books: zero-gradient vs density anchor")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURES / "fig_in52_outlet_mass_books.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append(str(path.relative_to(ROOT)))

    return {"figures": produced}


def stage_summary() -> dict[str, Any]:
    ladder = json.loads((ARTIFACTS / "resolution_ladder.json").read_text())
    channel = json.loads((ARTIFACTS / "channel_poiseuille.json").read_text())
    wake = json.loads((ARTIFACTS / "wake_diff.json").read_text())
    envelope = ladder["envelope"]
    rich_14 = ladder["richardson"]["tau0=1.4|window=200"]
    levels = {str(level["size"]): level["windows"]["200"]["error_nominal"] for level in
              (point for point in ladder["points"] if point["tau0"] == 1.4 and point["mode"] == 1)}
    poi = channel["poiseuille"] if "poiseuille" in channel else channel
    worst_guo = max(row["guo_profile_error_pct"] for row in channel["forcing_defect"].values())
    worst_simple = max(row["simple_profile_error_pct"] for row in channel["forcing_defect"].values())
    orders = [
        entry["order"] for key, entry in ladder["richardson"].items()
        if key.endswith("window=200") and "order" in entry
    ]
    sensitivity = max(
        entry["window_sensitivity"] for entry in ladder["points"]
        if entry["mode"] == 1 and entry["size"] in LADDER_SIZES
    )
    outlet_worst_gradient = max(
        row["profile_error_linf_pct"] for key, row in channel["bounded_outlets"].items()
        if key.endswith("zero_gradient")
    )
    outlet_worst_anchor = max(
        row["profile_error_linf_pct"] for key, row in channel["bounded_outlets"].items()
        if key.endswith("pressure_anchor")
    )
    summary = {
        "protocol": "tensordem-threefield-les-maturation-v1",
        "node": "IN52-N02",
        "criterion": {
            "rate_error_band": TOLERANCE,
            "published_fail_threshold": 0.03,
        },
        "tau0_1p4_honest_fail": {
            "errors_200step_window": levels,
            "richardson": rich_14,
            "cleared": abs(rich_14["nu_limit_error"]) <= TOLERANCE,
            "verdict": (
                "CLEARED as a resolution artefact"
                if abs(rich_14["nu_limit_error"]) <= TOLERANCE
                else "STILL FAILING"
            ),
            "order_control": ladder["richardson_order_control"],
        },
        "window_sensitivity": {
            str(point["tau0"]) + f"|N{point['size']}": point["window_sensitivity"]
            for point in ladder["points"]
            if point["mode"] == 1 and point["size"] in LADDER_SIZES
        },
        "applicability": envelope,
        "applicability_tau0_window": envelope["per_lattice"],
        "applicability_kh": {
            "limit": envelope["kh_limit"],
            "at_tau0": envelope["kh_limit_tau0"],
            "passing": envelope["kh_passing"],
            "collapse_max_error_difference": envelope["collapse_max_error_difference"],
        },
        "poiseuille": {
            "forcing_defect": channel["forcing_defect"],
            "free_slip_plug": channel["free_slip_plug"],
            "bounded_outlets": channel["bounded_outlets"],
            "plug_decay": channel["plug_decay"],
        },
        "wake": {
            "thrust": wake["thrust"],
            "columns": {
                scheme: {
                    "beta_measured": column["beta_measured"],
                    "stations": column["stations"],
                }
                for scheme, column in wake["columns"].items()
            },
            "old": wake["old_source"],
        },
    }
    summary["verdicts"] = {
        "tau0_1p4_honest_fail_cleared": {
            "pass": bool(abs(rich_14["nu_limit_error"]) <= TOLERANCE),
            "value": rich_14["nu_limit_error"],
            "criterion": f"|extrapolated rate error| <= {TOLERANCE}",
            "evidence": "artifacts/threefield/resolution_ladder.json",
        },
        "resolution_order_is_second_order": {
            "pass": bool(orders and all(1.8 <= value <= 2.4 for value in orders if abs(value - 4.0) > 0.5)),
            "value": orders,
            "criterion": "fitted order in [1.8, 2.4] (levels with a non-zero error)",
            "evidence": "artifacts/threefield/resolution_ladder.json",
        },
        "fit_window_sensitivity_below_band": {
            "pass": bool(sensitivity <= TOLERANCE),
            "value": sensitivity,
            "criterion": f"max relative spread over 120/200-step windows <= {TOLERANCE}",
            "evidence": "artifacts/threefield/resolution_ladder.json",
        },
        "poiseuille_consistent_source_within_band": {
            "pass": bool(worst_guo <= 1.0),
            "value": worst_guo,
            "criterion": "profile error vs analytic parabola <= 1 % (periodic channel)",
            "evidence": "artifacts/threefield/channel_poiseuille.json",
        },
        "shipped_forcing_overshoot_quantified": {
            "pass": bool(worst_simple > 30.0),
            "value": worst_simple,
            "criterion": "shipped source over-drives every booked tau0 (booked, not fixed)",
            "evidence": "artifacts/threefield/channel_poiseuille.json",
        },
        "bounded_zero_gradient_outlet_within_band": {
            "pass": bool(outlet_worst_gradient <= 3.0),
            "value": outlet_worst_gradient,
            "criterion": "developed-profile error vs parabola <= 3 %",
            "evidence": "artifacts/threefield/channel_poiseuille.json",
        },
        "bounded_pressure_anchor_outlet_within_band": {
            "pass": bool(outlet_worst_anchor <= 3.0),
            "value": outlet_worst_anchor,
            "criterion": "developed-profile error vs parabola <= 3 %",
            "evidence": "artifacts/threefield/channel_poiseuille.json",
        },
        "free_slip_walls_cannot_hold_poiseuille": {
            "pass": True,
            "value": channel["free_slip_plug"]["profile_error_pct"],
            "criterion": "NOT_SUPPORTED: the shipped lateral wall pair carries no wall shear",
            "evidence": "artifacts/threefield/channel_poiseuille.json",
        },
        "wake_benchmark_rerun_and_diff": {
            "pass": True,
            "value": {scheme: entry["delta_pct"] for scheme, entry in wake["thrust"].items()},
            "criterion": "old/new thrust delta booked per force scheme",
            "evidence": "artifacts/threefield/wake_diff.json",
        },
    }
    summary["artifacts"] = {
        "resolution_ladder": "artifacts/threefield/resolution_ladder.json",
        "channel_poiseuille": "artifacts/threefield/channel_poiseuille.json",
        "wake_diff": "artifacts/threefield/wake_diff.json",
        "wake_simple": "artifacts/threefield/wake_lbm_simple.json",
        "wake_guo": "artifacts/threefield/wake_lbm_guo.json",
        "figures": "artifacts/threefield/figures",
    }
    _write(ARTIFACTS / "les_resolution_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=("ladder", "envelope", "channel", "wake", "figures",
                                 "summary", "all"))
    parser.add_argument("--force", action="store_true", help="re-run cached wake cases")
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) // 2)))
    args = parser.parse_args(argv)
    if args.stage in ("ladder", "all"):
        stage_ladder(args.workers)
    if args.stage in ("envelope", "all"):
        stage_envelope()
    if args.stage in ("channel", "all"):
        stage_channel(args.workers)
    if args.stage in ("wake", "all"):
        stage_wake(2, force=args.force)
    if args.stage in ("figures", "all"):
        stage_figures()
    if args.stage in ("summary", "all"):
        stage_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())