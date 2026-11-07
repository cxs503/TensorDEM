"""Formation ice-channel CLOUD benchmark (topic ``formation-ice-channel``).

Purpose
-------
Supply the missing CLOUD evidence (spatial / angular field distributions) for the
``formation-ice-channel`` topic, on top of the verified single-hull channel path
(:mod:`tensordem.ice_channel`, ``benchmarks/ice_channel_dem.py``,
``docs/ice_channel_dem.md``).

Physics
-------
Three prescribed rectangular hulls in line ahead (lead + in-track follower +
laterally offset follower) advance through a floating flat-lying disk-floe field
at 8/10 concentration.  For every hull the instantaneous ice load is the reaction
of the penalty contact forces (``force_on_hull_N`` from
:meth:`IceChannelSimulation.hull_contact`); its streamwise component is the ice
resistance.  Each hull is averaged over its *own* traversal window (shifted by
``separation / speed``).

Fields (5 m x 5 m cells over a 120 m x 60 m domain)
--------------------------------------------------
``local_concentration``        final floe area fraction per cell            [-]
``mean_speed``                 time-mean floe speed per cell                [m/s]
``kinetic_energy_density``     time-mean floe kinetic energy per cell area  [J/m^2]
``mean_ice_load``              time-mean hull contact load per cell area    [Pa]
``mean_ice_load_lead``         same, lead hull only                         [Pa]
``mean_ice_load_follower``     same, both followers                         [Pa]
``lateral_displacement``       final-minus-initial transverse displacement  [m]
``floe_samples``               sampling denominator (floe-snapshots/cell)   [-]

Angular distributions (12 bins x 30 deg, atan2 of the force / velocity vector)
-----------------------------------------------------------------------------
``velocity_direction_in_track``   heading of floes initially inside the beam band
``velocity_direction_off_track``  heading of floes initially outside it
``hull_load_direction_lead``      direction of the lead load on floes (force weighted)
``hull_load_direction_follower``  same for the in-track follower

Longitudinal profile
--------------------
``cleared_half_width_m`` per 5 m station: ``min_i(|y_i - y_track| - R_i)`` clipped at
zero [m]; the exact geometric reference is the hull half beam ``B/2`` (the band the
hull itself sweeps).  The excess over ``B/2`` is physical over-clearance, not error.

Stages
------
``production`` (dt = 8 ms, seed 20261107) | ``dt_halved`` (4 ms) | ``dt_doubled``
(16 ms) | ``seed_alt`` (seed + 1).  ``assemble`` merges the parts into the cloud
JSON, the table CSV and the figures.

Usage
-----
``python benchmarks/formation_ice_channel_cloud.py --stage production``
``python benchmarks/formation_ice_channel_cloud.py --stage all``
``python benchmarks/formation_ice_channel_cloud.py --fast --stage all``
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tensordem.ice_channel import (  # noqa: E402
    HullGeometry,
    IceChannelSimulation,
    WaterEnvironment,
)

OUT_DIR = _REPO_ROOT / "artifacts" / "formation_ice_channel_cloud"
PART_DIR = OUT_DIR / "parts"
CLOUD_JSON = OUT_DIR / "field_cloud.json"
TABLE_CSV = OUT_DIR / "field_table.csv"

#: Pre-registered tolerance for the timestep sensitivity of the time-mean cell
#: fields (relative L1 norm).  A contact-driven DEM is event-chaotic, so the cell
#: fields only have to reproduce their *structure*; the strict criteria sit on the
#: aggregate quantities (mean load, cleared width, concentration).
FIELD_L1_TOL = 0.30
#: floes slower than this are excluded from the heading histogram (the heading of
#: a resting floe is sampling noise) [m/s]
VELOCITY_CUT = 0.05
N_ANGLE_BINS = 12
CELL_M = 5.0
RECORD_DT = 0.32


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def base_config(*, fast: bool) -> dict:
    if fast:
        domain = (48.0, 24.0)
        length, beam, speed = 32.0, 10.0, 2.0
        separation, offset, y_track = 24.0, 6.0, 12.0
        duration, cell = 30.0, 4.0
    else:
        domain = (120.0, 60.0)
        length, beam, speed = 80.0, 20.0, 1.6
        separation, offset, y_track = 60.0, 12.0, 30.0
        duration, cell = 165.0, CELL_M
    lead_x_start = -0.5 * length
    follower_x_start = lead_x_start - separation
    return {
        "domain_m": list(domain),
        "concentration": 0.8,
        "seed": 20261107,
        "cell_m": cell,
        "duration_s": duration,
        "cell_sampling_dt_s": RECORD_DT,
        "hulls": [
            {"name": "lead", "role": "航道领航 (lead)", "length": length, "beam": beam,
             "speed": speed, "y_position": y_track, "x_start": lead_x_start},
            {"name": "follower_in_track", "role": "编队跟随-航道内 (in-track follower)",
             "length": length, "beam": beam, "speed": speed, "y_position": y_track,
             "x_start": follower_x_start},
            {"name": "follower_offset", "role": "编队跟随-横向偏移 (offset follower)",
             "length": length, "beam": beam, "speed": speed,
             "y_position": y_track + offset, "x_start": follower_x_start},
        ],
    }


def stage_config(stage: str, *, fast: bool) -> dict:
    cfg = base_config(fast=fast)
    if stage == "production":
        cfg.update(dt=8.0e-3, note="reported fields")
    elif stage == "dt_halved":
        cfg.update(dt=4.0e-3, note="timestep refinement (dt/2)")
    elif stage == "dt_doubled":
        cfg.update(dt=16.0e-3, note="timestep coarsening (2 dt)")
    elif stage == "seed_alt":
        cfg.update(dt=8.0e-3, seed=cfg["seed"] + 1, note="field-realisation uncertainty")
    else:
        raise ValueError(stage)
    return cfg


def geometry_check(cfg: dict) -> dict:
    """Geometric verification of the prescribed hulls, before any physics.

    The hull is a rectangular prism, hence ``Cb = 1`` by definition; ``L``, ``B``
    and ``L/B`` are reported and ``half_length*2 == length`` / ``half_beam*2 ==
    beam`` are asserted to machine precision.
    """
    rows = []
    for spec in cfg["hulls"]:
        hull = HullGeometry(length=spec["length"], beam=spec["beam"], speed=spec["speed"],
                            y_position=spec["y_position"], x_start=spec["x_start"])
        assert math.isclose(2.0 * hull.half_length, spec["length"], rel_tol=0, abs_tol=1e-12)
        assert math.isclose(2.0 * hull.half_beam, spec["beam"], rel_tol=0, abs_tol=1e-12)
        rows.append({
            "name": spec["name"],
            "length_m": spec["length"], "beam_m": spec["beam"],
            "L_over_B": spec["length"] / spec["beam"],
            "block_coefficient_Cb": 1.0,
            "length_check_abs_err_m": abs(2.0 * hull.half_length - spec["length"]),
            "beam_check_abs_err_m": abs(2.0 * hull.half_beam - spec["beam"]),
            "track_y_m": spec["y_position"],
        })
    return {"hulls": rows,
            "passed": all(r["length_check_abs_err_m"] < 1e-12
                          and r["beam_check_abs_err_m"] < 1e-12 for r in rows)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _grid(domain, cell):
    nx = int(round(domain[0] / cell))
    ny = int(round(domain[1] / cell))
    xe = [i * domain[0] / nx for i in range(nx + 1)]
    ye = [j * domain[1] / ny for j in range(ny + 1)]
    return nx, ny, xe, ye


def _angle_hist(angles_deg, weights=None):
    w = torch.ones_like(angles_deg) if weights is None else weights.to(torch.float64)
    idx = torch.clamp((torch.remainder(angles_deg, 360.0)
                       / (360.0 / N_ANGLE_BINS)).long(), 0, N_ANGLE_BINS - 1)
    out = torch.zeros(N_ANGLE_BINS, dtype=torch.float64)
    out.index_add_(0, idx, w)
    return out


def _normalise(counts) -> list:
    total = float(sum(counts))
    return [float(c) / total for c in counts] if total > 0 else [0.0] * len(counts)


def equilibrium_drift_speed(water: WaterEnvironment, radius: float, thickness: float) -> float:
    """Steady drift speed of an isolated flat-lying floe in the wind field [m/s].

    Balances the quadratic wind drag on the above-water side wall against the
    quadratic water drag on the submerged side wall -- the same two laws that
    :mod:`tensordem.ice_channel` integrates -- solved by bisection.
    """
    ratio = water.ice_density / water.water_density
    draught = ratio * thickness
    freeboard = max(thickness - draught, 0.0)
    a_sail = 2.0 * radius * freeboard * water.sail_area_fraction
    a_wet = 2.0 * radius * draught
    wind = float(water.wind_velocity[0])

    def residual(u: float) -> float:
        wind_force = (0.5 * water.air_density * water.wind_drag_coefficient * a_sail
                      * abs(wind - u) * (wind - u))
        water_force = (0.5 * water.water_density * water.water_drag_coefficient * a_wet
                       * u * abs(u))
        return wind_force - water_force

    lo, hi = 0.0, wind
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run_stage(name: str, cfg: dict) -> dict:
    domain = cfg["domain_m"]
    nx, ny, xe, ye = _grid(domain, cfg["cell_m"])
    dx, dy = domain[0] / nx, domain[1] / ny
    cell_area = dx * dy

    hulls = [HullGeometry(length=s["length"], beam=s["beam"], speed=s["speed"],
                          y_position=s["y_position"], x_start=s["x_start"])
             for s in cfg["hulls"]]
    dt = float(cfg["dt"])
    sim = IceChannelSimulation.from_field(
        cfg["concentration"], domain_length=domain[0], domain_width=domain[1],
        seed=int(cfg["seed"]), dt=dt, hull=hulls[0], extra_hulls=tuple(hulls[1:]))

    x0 = sim.pos[:, 0].clone()
    y0 = sim.pos[:, 1].clone()
    area0 = math.pi * float((sim.radius ** 2).sum())

    n_steps = int(round(float(cfg["duration_s"]) / dt))
    record_every = max(1, int(round(float(cfg["cell_sampling_dt_s"]) / dt)))

    shape = (ny, nx)
    load_all = torch.zeros(shape, dtype=torch.float64)
    load_lead = torch.zeros(shape, dtype=torch.float64)
    load_follower = torch.zeros(shape, dtype=torch.float64)
    speed_sum = torch.zeros(shape, dtype=torch.float64)
    energy_sum = torch.zeros(shape, dtype=torch.float64)
    samples = torch.zeros(shape, dtype=torch.float64)
    n_snapshots = 0

    ang_vel_in = torch.zeros(N_ANGLE_BINS, dtype=torch.float64)
    ang_vel_off = torch.zeros(N_ANGLE_BINS, dtype=torch.float64)
    ang_load_lead = torch.zeros(N_ANGLE_BINS, dtype=torch.float64)
    ang_load_follower = torch.zeros(N_ANGLE_BINS, dtype=torch.float64)

    y_track = float(cfg["hulls"][0]["y_position"])
    in_track0 = (y0 - y_track).abs() <= 0.5 * float(cfg["hulls"][0]["beam"])

    windows = []
    for hull in hulls:
        windows.append(((-hull.length - hull.x_start) / hull.speed,
                        (domain[0] + hull.length - hull.x_start) / hull.speed))
    force_x = [[] for _ in hulls]
    force_mag = [[] for _ in hulls]
    contact_flag = [[] for _ in hulls]

    tic = time.perf_counter()
    for step in range(n_steps):
        sim.step()
        if (step + 1) % record_every:
            continue
        n_snapshots += 1
        pos, vel = sim.pos, sim.vel
        ix = torch.clamp((pos[:, 0] / dx).long(), 0, nx - 1)
        iy = torch.clamp((pos[:, 1] / dy).long(), 0, ny - 1)
        flat = iy * nx + ix
        speed = torch.linalg.norm(vel, dim=1)
        energy = 0.5 * sim.mass * speed ** 2

        samples += torch.bincount(flat, minlength=ny * nx).reshape(shape).to(torch.float64)
        speed_sum += torch.bincount(flat, weights=speed, minlength=ny * nx).reshape(
            shape).to(torch.float64)
        energy_sum += torch.bincount(flat, weights=energy, minlength=ny * nx).reshape(
            shape).to(torch.float64)

        ledger = sim.hull_contact(sim.time)
        magnitudes = [entry["per_floe_force_magnitude_N"] for entry in ledger]
        total = magnitudes[0].clone()
        for mag in magnitudes[1:]:
            total = total + mag
        load_all += torch.bincount(flat, weights=total, minlength=ny * nx).reshape(
            shape).to(torch.float64)
        load_lead += torch.bincount(flat, weights=magnitudes[0], minlength=ny * nx).reshape(
            shape).to(torch.float64)
        followers = total - magnitudes[0]
        load_follower += torch.bincount(flat, weights=followers, minlength=ny * nx).reshape(
            shape).to(torch.float64)

        for index, entry in enumerate(ledger):
            mag = magnitudes[index]
            active = mag > 0.0
            if bool(active.any()):
                vec = entry["force_on_floes_N"]
                ang = torch.rad2deg(torch.atan2(vec[active, 1], vec[active, 0]))
                if index == 0:
                    ang_load_lead += _angle_hist(ang, mag[active])
                elif index == 1:
                    ang_load_follower += _angle_hist(ang, mag[active])
            t0, t1 = windows[index]
            if t0 <= sim.time <= t1:
                f_hull = entry["force_on_hull_N"]
                force_x[index].append(float(-f_hull[0]))
                force_mag[index].append(float(torch.linalg.norm(f_hull)))
                contact_flag[index].append(1.0 if entry["contact_count"] > 0 else 0.0)

        moving = speed >= VELOCITY_CUT
        if bool(moving.any()):
            ang = torch.rad2deg(torch.atan2(vel[moving, 1], vel[moving, 0]))
            idx = torch.clamp((torch.remainder(ang, 360.0) / (360.0 / N_ANGLE_BINS)).long(),
                              0, N_ANGLE_BINS - 1)
            in_track = in_track0[moving]
            for k in range(N_ANGLE_BINS):
                sel = idx == k
                if bool(sel.any()):
                    ang_vel_in[k] += float(sel[in_track].sum())
                    ang_vel_off[k] += float(sel[~in_track].sum())

    wall = time.perf_counter() - tic
    n_snap = max(n_snapshots, 1)

    pos_final = sim.pos
    ix = torch.clamp((pos_final[:, 0] / dx).long(), 0, nx - 1)
    iy = torch.clamp((pos_final[:, 1] / dy).long(), 0, ny - 1)
    flat = iy * nx + ix
    area_cell = torch.bincount(flat, weights=math.pi * sim.radius ** 2,
                               minlength=ny * nx).reshape(shape).to(torch.float64)
    count_cell = torch.bincount(flat, minlength=ny * nx).reshape(shape).to(torch.float64)
    dy_floe = pos_final[:, 1] - y0
    dy_sum = torch.bincount(flat, weights=dy_floe, minlength=ny * nx).reshape(
        shape).to(torch.float64)

    fields = {
        "local_concentration": (area_cell / cell_area).tolist(),
        "mean_speed": (speed_sum / samples.clamp(min=1e-30)).tolist(),
        "kinetic_energy_density": (energy_sum / cell_area / n_snap).tolist(),
        "mean_ice_load": (load_all / cell_area / n_snap).tolist(),
        "mean_ice_load_lead": (load_lead / cell_area / n_snap).tolist(),
        "mean_ice_load_follower": (load_follower / cell_area / n_snap).tolist(),
        "lateral_displacement": (dy_sum / count_cell.clamp(min=1.0)).tolist(),
        "floe_samples": (samples / n_snap).tolist(),
    }

    half, min_clear = [None] * nx, [None] * nx
    for i in range(nx):
        sel = ix == i
        if bool(sel.any()):
            clearance = (pos_final[sel, 1] - y_track).abs() - sim.radius[sel]
            min_clear[i] = float(clearance.min())
            half[i] = float(clearance.min().clamp(min=0.0))
    profile = {
        "x_centres_m": [0.5 * (xe[i] + xe[i + 1]) for i in range(nx)],
        "cleared_half_width_m": half,
        "min_signed_clearance_m": min_clear,
        "reference_half_beam_m": 0.5 * float(cfg["hulls"][0]["beam"]),
    }

    hull_rows = []
    for index, spec in enumerate(cfg["hulls"]):
        fx, fm = force_x[index], force_mag[index]
        hull_rows.append({
            "name": spec["name"], "role": spec["role"],
            "y_position_m": spec["y_position"],
            "window_s": list(windows[index]),
            "samples": len(fx),
            "mean_streamwise_ice_force_N": (sum(fx) / len(fx)) if fx else None,
            "mean_abs_ice_force_N": (sum(fm) / len(fm)) if fm else None,
            "max_abs_ice_force_N": max(fm) if fm else None,
            "contact_time_fraction": (sum(contact_flag[index]) / len(contact_flag[index]))
            if contact_flag[index] else None,
            "cleared_width_final_m": sim.cleared_width(hull=hulls[index]),
        })

    area_final = math.pi * float((sim.radius ** 2).sum())
    water = sim.water
    return {
        "stage": name,
        "note": cfg.get("note", ""),
        "config": cfg,
        "meta": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "module": "tensordem.ice_channel",
            "benchmark": "benchmarks/formation_ice_channel_cloud.py",
        },
        "geometry_check": geometry_check(cfg),
        "field_statistics": sim.field_statistics(),
        "velocity_statistics": sim.velocity_statistics(),
        "reference_equilibrium_drift_mps": equilibrium_drift_speed(
            water, float(sim.radius.mean()), float(sim.thickness.mean())),
        "grid": {"x_edges_m": xe, "y_edges_m": ye, "nx": nx, "ny": ny,
                 "cell_area_m2": cell_area},
        "fields": fields,
        "angular": {
            "bin_edges_deg": [k * 360.0 / N_ANGLE_BINS for k in range(N_ANGLE_BINS + 1)],
            "velocity_direction_in_track": _normalise(ang_vel_in),
            "velocity_direction_off_track": _normalise(ang_vel_off),
            "hull_load_direction_lead": _normalise(ang_load_lead),
            "hull_load_direction_follower": _normalise(ang_load_follower),
            "velocity_direction_in_track_counts": ang_vel_in.tolist(),
            "velocity_direction_off_track_counts": ang_vel_off.tolist(),
            "velocity_cut_mps": VELOCITY_CUT,
            "isotropic_reference_frequency": 1.0 / N_ANGLE_BINS,
        },
        "profile": profile,
        "hulls": hull_rows,
        "conservation": {
            "floe_area_initial_m2": area0,
            "floe_area_final_m2": area_final,
            "floe_area_rel_drift": abs(area_final - area0) / area0,
            "max_abs_heave_m": float(sim.z.abs().max()),
            "max_abs_tilt_rad": float(sim.tilt.abs().max()),
        },
        "run": {"dt": dt, "steps": n_steps, "sim_time_s": sim.time,
                "record_every": record_every, "n_snapshots": n_snapshots,
                "wall_time_s": wall, "state_digest": sim.state_digest()},
    }# ---------------------------------------------------------------------------
# assemble: convergence, uncertainty, table, figures
# ---------------------------------------------------------------------------

def _rel_l1(a, b) -> float:
    num = den = 0.0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            num += abs(va - vb)
            den += abs(vb)
    return num / den if den else float("nan")


def _field_mean(field) -> float:
    vals = [v for row in field for v in row]
    return sum(vals) / len(vals) if vals else float("nan")


def _observed_order(coarse, medium, fine) -> dict:
    """Roache three-level observed order and GCI, refinement ratio r = 2."""
    out = {"coarse": coarse, "medium": medium, "fine": fine}
    if coarse is None or medium is None or fine is None:
        out.update({"observed_order": None, "gci_fine_pct": None,
                    "fallback_reason": "level missing"})
        return out
    e1, e2 = abs(medium - coarse), abs(fine - medium)
    out.update({"e_coarse_medium": e1, "e_medium_fine": e2, "refinement_ratio": 2.0})
    if e1 == 0.0 or e2 == 0.0:
        out.update({"observed_order": None, "gci_fine_pct": None,
                    "fallback_reason": "zero change between levels: convergence not "
                                       "resolvable (round-off / identical events)"})
        return out
    p = math.log(e1 / e2) / math.log(2.0)
    out["observed_order"] = p
    if not 0.0 < p < 4.0:
        out.update({"gci_fine_pct": None,
                    "fallback_reason": f"observed order p={p:.3f} outside (0,4): "
                                       "non-asymptotic / non-monotone convergence "
                                       "(contact-event chaos), GCI not applicable"})
        return out
    out["gci_fine_pct"] = 100.0 * 1.25 * e2 / (2.0 ** p - 1.0)
    return out


def _cell_u95(prod, parts, field, j, i):
    value = prod["fields"][field][j][i]
    comps = []
    if "dt_halved" in parts:
        comps.append(abs(value - parts["dt_halved"]["fields"][field][j][i]))
    if "seed_alt" in parts:
        comps.append(abs(value - parts["seed_alt"]["fields"][field][j][i]))
    if not comps:
        return None
    return 2.0 * math.sqrt(sum(c * c for c in comps))


def assemble(*, figures: bool = True) -> dict:
    parts = {}
    for path in sorted(PART_DIR.glob("part_*.json")):
        payload = json.loads(path.read_text())
        parts[payload["stage"]] = payload
    if "production" not in parts:
        raise SystemExit("production part missing: run --stage production first")

    prod = parts["production"]
    cfg = prod["config"]
    nx, ny = prod["grid"]["nx"], prod["grid"]["ny"]
    reference_half_beam = prod["profile"]["reference_half_beam_m"]
    lead_force = {row["name"]: row["mean_abs_ice_force_N"] for row in prod["hulls"]}["lead"]

    def stage_agg(part):
        hulls = {row["name"]: row for row in part["hulls"]}
        lead, follow = hulls["lead"], hulls["follower_in_track"]
        offset = hulls["follower_offset"]
        reduction = None
        if (lead["mean_abs_ice_force_N"] is not None
                and follow["mean_abs_ice_force_N"] is not None
                and lead["mean_abs_ice_force_N"] > 0.0):
            reduction = 100.0 * (lead["mean_abs_ice_force_N"]
                                 - follow["mean_abs_ice_force_N"]) / lead["mean_abs_ice_force_N"]
        profile = [h for h in part["profile"]["cleared_half_width_m"] if h is not None]
        return {
            "mean_abs_ice_force_lead_N": lead["mean_abs_ice_force_N"],
            "mean_abs_ice_force_follower_in_track_N": follow["mean_abs_ice_force_N"],
            "mean_abs_ice_force_follower_offset_N": offset["mean_abs_ice_force_N"],
            "formation_ice_force_reduction_pct": reduction,
            "mean_cleared_half_width_m": sum(profile) / len(profile) if profile else None,
            "mean_speed_field_mps": _field_mean(part["fields"]["mean_speed"]),
            "mean_ice_load_field_Pa": _field_mean(part["fields"]["mean_ice_load"]),
        }

    aggregates = {name: stage_agg(part) for name, part in parts.items()}

    convergence = {}
    if all(k in parts for k in ("dt_doubled", "production", "dt_halved")):
        for key in ("mean_abs_ice_force_lead_N", "formation_ice_force_reduction_pct",
                    "mean_cleared_half_width_m", "mean_ice_load_field_Pa"):
            convergence[key] = _observed_order(
                aggregates["dt_doubled"][key], aggregates["production"][key],
                aggregates["dt_halved"][key])

    field_convergence = {}
    if "dt_halved" in parts:
        for field_name in ("mean_ice_load", "mean_speed", "lateral_displacement"):
            field_convergence[field_name] = {
                "rel_l1_dt8_vs_dt4": _rel_l1(prod["fields"][field_name],
                                             parts["dt_halved"]["fields"][field_name]),
                "tolerance": FIELD_L1_TOL,
            }
    seed_field = {}
    if "seed_alt" in parts:
        for field_name in ("mean_ice_load", "mean_speed"):
            a, b = prod["fields"][field_name], parts["seed_alt"]["fields"][field_name]
            diffs = [abs(va - vb) for ra, rb in zip(a, b) for va, vb in zip(ra, rb)]
            seed_field[field_name] = {
                "rel_l1_seed_vs_seed": _rel_l1(a, b),
                "rms_abs_diff": math.sqrt(sum(d * d for d in diffs) / len(diffs)),
            }

    # ---- table -----------------------------------------------------------
    rows = []
    unit_of = {"mean_ice_load": "Pa", "local_concentration": "-",
               "lateral_displacement": "m"}
    for name in ("mean_ice_load", "local_concentration", "lateral_displacement"):
        field = prod["fields"][name]
        for jj in range(ny):
            y_centre = 0.5 * (prod["grid"]["y_edges_m"][jj] + prod["grid"]["y_edges_m"][jj + 1])
            for ii in range(nx):
                x_centre = 0.5 * (prod["grid"]["x_edges_m"][ii]
                                  + prod["grid"]["x_edges_m"][ii + 1])
                value = field[jj][ii]
                if name == "local_concentration":
                    reference = None
                    error = None
                    source = ("NO-DATA-TBD: cell footprint ratio (floe area by centre / "
                              "cell area) is not a concentration at 5 m cells (mean floe "
                              "radius 4.9 m > cell); the domain-mean value is the "
                              "meaningful concentration")
                else:
                    reference, source, error = None, "NO-DATA-TBD", None
                rows.append({
                    "class": "grid_cell", "quantity": name, "unit": unit_of[name],
                    "x_centre_m": f"{x_centre:.2f}", "y_centre_m": f"{y_centre:.2f}",
                    "value": value, "reference": reference,
                    "reference_source": source, "error": error,
                    "u95": _cell_u95(prod, parts, name, jj, ii),
                })

    ang = prod["angular"]
    for key in ("velocity_direction_in_track", "velocity_direction_off_track",
                "hull_load_direction_lead", "hull_load_direction_follower"):
        freqs = ang[key]
        for k, freq in enumerate(freqs):
            lo, hi = ang["bin_edges_deg"][k], ang["bin_edges_deg"][k + 1]
            u95 = None
            if "seed_alt" in parts and key in parts["seed_alt"].get("angular", {}):
                u95 = 2.0 * abs(freq - parts["seed_alt"]["angular"][key][k])
            rows.append({
                "class": "angle_bin", "quantity": key, "unit": "-",
                "bin_deg": f"{lo:.0f}-{hi:.0f}", "value": freq,
                "reference": ang["isotropic_reference_frequency"],
                "reference_source": "isotropic (uniform 1/12) reference",
                "error": freq - ang["isotropic_reference_frequency"],
                "u95": u95,
            })

    domain_half_width = 0.5 * (prod["grid"]["y_edges_m"][-1] - prod["grid"]["y_edges_m"][0])
    raw_half = prod["profile"]["cleared_half_width_m"]
    half_clipped = [domain_half_width if h is None else min(h, domain_half_width)
                    for h in raw_half]
    for i, x_centre in enumerate(prod["profile"]["x_centres_m"]):
        half = half_clipped[i]
        u95 = None
        if "dt_halved" in parts:
            other = parts["dt_halved"]["profile"]["cleared_half_width_m"][i]
            if half is not None and other is not None:
                u95 = 2.0 * abs(half - other)
        rows.append({
            "class": "profile_station", "quantity": "cleared_half_width_m", "unit": "m",
            "x_centre_m": f"{x_centre:.2f}",
            "value": half, "reference": reference_half_beam,
            "reference_source": "hull half beam B/2 (exact swept band, lower bound); "
                                "excess is physical over-clearance, not error; value "
                                f"clipped at the domain half width "
                                f"({domain_half_width:.1f} m)",
            "error": (None if half is None else half - reference_half_beam),
            "u95": u95,
        })

    for row in prod["hulls"]:
        is_lead = row["name"] == "lead"
        rows.append({
            "class": "hull", "quantity": f"mean_abs_ice_force_{row['name']}", "unit": "N",
            "value": row["mean_abs_ice_force_N"],
            "reference": lead_force,
            "reference_source": "self" if is_lead else "lead hull of the same formation run",
            "error": None if is_lead or row["mean_abs_ice_force_N"] is None
            else row["mean_abs_ice_force_N"] - lead_force,
            "u95": None,
        })

    for stage, agg in aggregates.items():
        for key, value in agg.items():
            rows.append({"class": "parameter_level", "quantity": key, "unit": "-",
                         "level": stage, "value": value, "reference": None,
                         "reference_source": "NO-DATA-TBD", "error": None, "u95": None})

    realised = prod["field_statistics"]["realised_concentration"]
    target = prod["field_statistics"]["target_concentration"]
    clearance_vals = [v for v in prod["profile"]["min_signed_clearance_m"] if v is not None]
    global_rows = {
        "concentration_target": target,
        "concentration_realised": realised,
        "concentration_rel_error_pct": 100.0 * abs(realised - target) / target,
        "floe_area_rel_drift": prod["conservation"]["floe_area_rel_drift"],
        "min_signed_clearance_m": min(clearance_vals) if clearance_vals else None,
        "geometry_check_passed": prod["geometry_check"]["passed"],
        "max_abs_heave_m": prod["conservation"]["max_abs_heave_m"],
        "max_abs_tilt_rad": prod["conservation"]["max_abs_tilt_rad"],
        "reference_equilibrium_drift_mps": prod["reference_equilibrium_drift_mps"],
        "production_state_digest": prod["run"]["state_digest"],
    }
    for key, value in global_rows.items():
        rows.append({"class": "global", "quantity": key, "unit": "-", "value": value,
                     "reference": None, "reference_source": "NO-DATA-TBD",
                     "error": None, "u95": None})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["class", "quantity", "unit", "x_centre_m", "y_centre_m", "bin_deg",
                  "level", "value", "reference", "reference_source", "error", "u95"]
    with open(TABLE_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "topic": "formation-ice-channel",
        "scenario_key": "paper-mat-formation-ice-channel-cloud",
        "deliverable": "CLOUD: spatial/angular field distributions of a formation "
                       "ice-channel transit",
        "meta": prod["meta"],
        "config": cfg,
        "geometry_check": prod["geometry_check"],
        "stages_present": sorted(parts),
        "grid_definition": {
            "cell_m": cfg["cell_m"], "nx": nx, "ny": ny,
            "x_extent_m": [prod["grid"]["x_edges_m"][0], prod["grid"]["x_edges_m"][-1]],
            "y_extent_m": [prod["grid"]["y_edges_m"][0], prod["grid"]["y_edges_m"][-1]],
            "field_convention": "cell value at (x_centre, y_centre) = edge index [j][i], "
                                "i along x, j along y",
        },
        "field_notes": {
            "local_concentration": "area of the floes whose final centre falls in the cell "
                                  "divided by the cell area; with mean floe radius 4.9 m "
                                  "and 5 m cells this is a footprint/coverage ratio that "
                                  "exceeds 1 where floes pile up, NOT a concentration. "
                                  "The concentration statement is the domain mean "
                                  "(field_statistics.realised_concentration).",
            "mean_ice_load": "time-mean of the summed per-floe hull contact force "
                            "magnitude divided by the cell area; a contact-load density "
                            "[Pa], not a contact pressure (no contact area is resolved).",
        },
        "field_units": {
            "local_concentration": "-", "mean_speed": "m/s",
            "kinetic_energy_density": "J/m^2", "mean_ice_load": "Pa",
            "mean_ice_load_lead": "Pa", "mean_ice_load_follower": "Pa",
            "lateral_displacement": "m", "floe_samples": "-",
        },
        "fields": prod["fields"],
        "angular": prod["angular"],
        "angular_units": "bin frequency [-]; bin edges in deg (0-360, atan2 of the "
                         "force or velocity vector)",
        "profile": dict(prod["profile"],
                        cleared_half_width_clipped_m=half_clipped,
                        raw_half_width_m=raw_half,
                        domain_half_width_m=domain_half_width,
                        fully_cleared_stations=sum(1 for h in raw_half if h is None),
                        stations=len(raw_half)),
        "profile_units": "x [m]; half width [m] (clipped at the domain half width); "
                         "signed clearance [m]; a station with no floe left is fully "
                         "cleared and reported at the domain half width",
        "hulls": prod["hulls"],
        "field_statistics": prod["field_statistics"],
        "velocity_statistics": prod["velocity_statistics"],
        "conservation": prod["conservation"],
        "global_rows": global_rows,
        "aggregates": aggregates,
        "convergence": {
            "three_level_gci": convergence,
            "field_rel_l1": field_convergence,
            "seed_field_variability": seed_field,
            "field_l1_tolerance": FIELD_L1_TOL,
        },
        "table_rows": len(rows),
        "table_csv": str(TABLE_CSV.relative_to(_REPO_ROOT)),
    }

    figures_made = []
    if figures:
        figures_made = make_figures(payload)
        payload["figures"] = figures_made

    CLOUD_JSON.write_text(json.dumps(payload, indent=1))
    print(f"[assemble] {CLOUD_JSON} ({CLOUD_JSON.stat().st_size/1024:.1f} KB), "
          f"{len(rows)} table rows, {len(figures_made)} figures")
    return payload


def make_figures(payload: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cfg = payload["config"]
    grid = payload["grid_definition"]
    nx, ny = grid["nx"], grid["ny"]
    x_edges = np.linspace(grid["x_extent_m"][0], grid["x_extent_m"][1], nx + 1)
    y_edges = np.linspace(grid["y_extent_m"][0], grid["y_extent_m"][1], ny + 1)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    y_track = cfg["hulls"][0]["y_position"]
    half_beam = 0.5 * cfg["hulls"][0]["beam"]
    figures = []

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4))
    panels = [
        ("mean_ice_load", "time-mean hull ice load [Pa] (3 hulls)", "viridis"),
        ("mean_speed", "time-mean floe speed [m/s]", "magma"),
        ("lateral_displacement", r"transverse displacement $\Delta y$ [m]", "coolwarm"),
        ("local_concentration", "final local concentration [-]", "cividis"),
    ]
    for ax, (key, label, cmap) in zip(axes.ravel(), panels):
        data = np.array(payload["fields"][key], dtype=float)
        mesh = ax.pcolormesh(x_edges, y_edges, data, cmap=cmap, shading="flat")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.axhline(y_track, color="w", lw=0.9, ls="-", alpha=0.9)
        ax.axhline(y_track + half_beam, color="w", lw=0.9, ls="--", alpha=0.8)
        ax.axhline(y_track - half_beam, color="w", lw=0.9, ls="--", alpha=0.8)
        for spec in cfg["hulls"][1:]:
            ax.axhline(spec["y_position"], color="cyan", lw=0.7, ls=":", alpha=0.9)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        if np.isfinite(data).any():
            ax.contour(xc, yc, data, levels=5, colors="w", linewidths=0.4, alpha=0.6)
        else:
            ax.text(0.5, 0.5, "NO-DATA-TBD", transform=ax.transAxes, ha="center")
    fig.suptitle("Formation ice-channel transit (8/10, 3 hulls in line ahead): "
                 "5 m x 5 m cell field cloud", fontsize=10)
    fig.tight_layout()
    path = OUT_DIR / "fig_field_cloud.png"
    fig.savefig(path, dpi=105)
    plt.close(fig)
    figures.append({
        "name": path.name,
        "caption": "Spatial field cloud, 120 m x 60 m domain at 5 m resolution: "
                   "(a) time-mean hull ice load [Pa]; (b) time-mean floe speed [m/s]; "
                   "(c) transverse displacement [m]; (d) final local concentration [-]. "
                   "White solid: track y=30 m; white dashed: beam band |y-30|<=10 m; "
                   "cyan dotted: in-track (y=30) and offset (y=42) follower lanes.",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    speed = np.array(payload["fields"]["mean_speed"], dtype=float)
    load = np.array(payload["fields"]["mean_ice_load"], dtype=float)
    profile_speed = np.nanmean(np.where(np.isfinite(speed), speed, np.nan), axis=1)
    profile_load = np.nanmean(np.where(np.isfinite(load), load, np.nan), axis=1)
    ref = payload["global_rows"]["reference_equilibrium_drift_mps"]
    ax1.plot(profile_speed, yc, "o-", color="tab:blue", ms=4,
             label=r"measured $\langle|v|\rangle(y)$")
    ax1.axvline(ref, color="tab:red", ls="--",
                label=f"wind-drift equilibrium {ref:.4f} m/s")
    ax1.axvspan(y_track - half_beam, y_track + half_beam, color="grey", alpha=0.15,
                label="beam band")
    ax1.set_xlabel("time-mean floe speed [m/s]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("(a) transverse speed profile vs drift equilibrium", fontsize=9)
    ax1.legend(fontsize=6.5, loc="lower right")
    ax1.grid(alpha=0.3)
    twin = ax1.twiny()
    twin.plot(profile_load, yc, "s-", color="tab:orange", ms=3.5, alpha=0.8,
              label="ice load [Pa]")
    twin.set_xlabel("time-mean ice load [Pa]", fontsize=8)
    twin.tick_params(labelsize=7)

    half = payload["profile"]["cleared_half_width_clipped_m"]
    ax2.plot(payload["profile"]["x_centres_m"],
             [np.nan if h is None else h for h in half], "o-", color="tab:green", ms=4,
             label="cleared half width from floe positions")
    ax2.axhline(payload["profile"]["reference_half_beam_m"], color="k", ls="--",
                label="reference B/2 = %.1f m (exact swept band)"
                      % payload["profile"]["reference_half_beam_m"])
    ax2.set_xlabel("x station [m]")
    ax2.set_ylabel("cleared half width [m]")
    ax2.set_title("(b) longitudinal channel profile vs geometric reference", fontsize=9)
    ax2.legend(fontsize=6.5)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=7)
    fig.tight_layout()
    path = OUT_DIR / "fig_profile_reference.png"
    fig.savefig(path, dpi=105)
    plt.close(fig)
    figures.append({
        "name": path.name,
        "caption": "(a) Transverse profiles of the time-mean floe speed and hull ice load "
                   "against the analytic wind-drift equilibrium speed of an isolated floe; "
                   "(b) longitudinal cleared half width per 5 m station against the exact "
                   "geometric reference B/2 = 10 m.",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2),
                                   subplot_kw={"projection": "polar"})
    ang = payload["angular"]
    edges = np.deg2rad(np.array(ang["bin_edges_deg"]))
    width = np.diff(edges)[0]
    ax1.bar(edges[:-1] + 0.5 * width, ang["velocity_direction_in_track"], width=width,
            color="tab:blue", alpha=0.7, label="floes initially in beam band")
    ax1.bar(edges[:-1] + 0.5 * width, ang["velocity_direction_off_track"], width=width,
            color="tab:green", alpha=0.5, label="floes initially outside")
    ax1.plot(edges, [ang["isotropic_reference_frequency"]] * len(edges), "k--", lw=0.9,
             label="isotropic 1/12")
    ax1.set_title("(a) floe velocity heading", fontsize=9)
    ax1.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.28, 1.1))
    follower_freqs = ang["hull_load_direction_follower"]
    ax2.bar(edges[:-1] + 0.5 * width, follower_freqs, width=width,
            color="tab:red", alpha=0.7, label="in-track follower")
    ax2.bar(edges[:-1] + 0.5 * width, ang["hull_load_direction_lead"], width=width,
            color="tab:orange", alpha=0.45, label="lead")
    ax2.plot(edges, [ang["isotropic_reference_frequency"]] * len(edges), "k--", lw=0.9)
    if sum(follower_freqs) <= 0.0:
        ax2.text(0.5, -0.16, "NO-DATA-TBD: in-track follower had zero contact events",
                 transform=ax2.transAxes, ha="center", fontsize=6.5)
    ax2.set_title("(b) hull load direction on floes", fontsize=9)
    ax2.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.28, 1.1))
    fig.suptitle("Angular field distributions (12 bins x 30 deg, 0 deg = +x)", fontsize=10)
    fig.tight_layout()
    path = OUT_DIR / "fig_angular_cloud.png"
    fig.savefig(path, dpi=105)
    plt.close(fig)
    figures.append({
        "name": path.name,
        "caption": "Angular field distributions over 12 bins of 30 deg (0 deg = +x): "
                   "(a) floe velocity heading for floes initially inside / outside the "
                   "beam band, against the isotropic 1/12 reference; (b) direction of the "
                   "hull load exerted on the floes, in-track follower vs lead (force "
                   "weighted, normalised per hull).",
    })
    return figures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

STAGES = ("production", "dt_halved", "dt_doubled", "seed_alt")


def main(argv=None) -> int:
    global OUT_DIR, PART_DIR, CLOUD_JSON, TABLE_CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="production",
                        choices=STAGES + ("all", "assemble"))
    parser.add_argument("--fast", action="store_true",
                        help="reduced domain/timestep for unit tests")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--out-dir", default=None,
                        help="override the artifact root (used by the unit tests)")
    args = parser.parse_args(argv)

    fast = bool(args.fast)
    root = Path(args.out_dir) if args.out_dir else Path(OUT_DIR)
    out_dir = root if not fast else root / "fast"
    part_dir = PART_DIR if not fast else out_dir / "parts"
    OUT_DIR, PART_DIR = out_dir, part_dir
    CLOUD_JSON = out_dir / "field_cloud.json"
    TABLE_CSV = out_dir / "field_table.csv"
    PART_DIR.mkdir(parents=True, exist_ok=True)

    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "assemble":
            continue
        cfg = stage_config(stage, fast=fast)
        if args.seed is not None:
            cfg["seed"] = int(args.seed)
        tic = time.perf_counter()
        part = run_stage(stage, cfg)
        path = PART_DIR / f"part_{stage}.json"
        path.write_text(json.dumps(part, indent=1))
        print(f"[{stage}] dt={cfg['dt']:.4f} seed={cfg['seed']} steps={part['run']['steps']} "
              f"wall={part['run']['wall_time_s']:.1f}s digest={part['run']['state_digest'][:12]} "
              f"-> {path.name}", flush=True)

    if args.stage in ("assemble", "all"):
        payload = assemble(figures=not args.no_figures)
        print(f"[done] stages={payload['stages_present']} "
              f"reduction={payload['aggregates']['production']['formation_ice_force_reduction_pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())