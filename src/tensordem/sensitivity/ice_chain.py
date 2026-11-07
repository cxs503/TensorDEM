"""The DEM ice-field -> macro ice-resistance evaluation chain.

Two DEM experiments are driven by the UQ study:

``wedge``
    A 3-D bonded-sphere ice sheet indented by a moving 45 deg wedge
    (:class:`tensordem.dem3d_wedge.MovingWedgeBowIceDEM3D`), the repository's
    bow-ice-resistance prototype.  From one run we read

    * ``R``    -- mean horizontal resistance over the second half of the
      indentation window (crushing phase),
    * ``R_ch`` -- mean resistance after the first bond break (the broken /
      channelled regime, i.e. the rubble-channel resistance),
    * ``trace`` -- the resistance history resampled to ``TRACE_POINTS`` equally
      spaced samples.  This trace *is* the one-dimensional calibration curve
      used by the identifiability study.

``repose``
    A column-collapse heap (:mod:`tensordem.repose_calibration`) giving the
    angle of repose ``theta_r``.  A dry heap has no bonds and no buoyancy, so
    only ``mu``, ``e``, ``k_t/k_n``, the shape proxy ``mu_r`` and (weakly)
    ``k_n`` enter; the other three parameters are held at their nominal values
    and their zero influence on ``theta_r`` is verified as a negative control
    in the test suite.

Every case is deterministic (fixed seed, single-threaded reduction order) and is
cached as one JSON file, so a cluster shard can be re-run and resumed without
recomputing finished cases.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .design import Parameter, ParameterSpace

__all__ = [
    "PARAMETER_NAMES",
    "ICE_SPACE",
    "REPOSE_ACTIVE",
    "ICE_THICKNESS_M",
    "wedge_case",
    "repose_case",
    "evaluate_spec",
    "run_batch",
]

# --- chain constants ---------------------------------------------------------
ICE_THICKNESS_M = 0.16          # ice sheet thickness (fixed; psi carries h/D)
BOW_SPEED_M_S = 0.2             # wedge advance speed
WEDGE_ANGLE_DEG = 45.0
BOW_HALF_WIDTH_M = 0.03
WINDOW_S = 0.04                 # simulated indentation window
TRACE_POINTS = 12               # samples of the resistance trace
BOND_STIFFNESS = 1200.0         # N/m, fixed bond stiffness of the lattice

ICE_SPACE = ParameterSpace([
    Parameter("mu", 0.10, 0.80, "-",
              "particle-particle Coulomb friction coefficient"),
    Parameter("e", 0.10, 0.90, "-", "contact restitution coefficient"),
    Parameter("k_n", 2000.0, 8000.0, "N/m", "normal contact stiffness"),
    Parameter("k_t_ratio", 0.30, 1.00, "-", "tangential/normal stiffness ratio k_t/k_n"),
    Parameter("eps_b", 0.004, 0.012, "-",
              "bond tensile failure strain (bond strength sigma_b = k_b eps_b 2R/(pi R^2))"),
    Parameter("mu_r", 0.05, 0.50, "-",
              "rolling-resistance coefficient, the sphere-DEM proxy for particle angularity"),
    Parameter("psi", 4.0, 6.0, "-", "ice thickness / particle diameter ratio h/(2R)"),
    Parameter("c_wet", 0.0, 30.0, "N s/m",
              "buoyancy/added-mass wet drag coefficient on the submerged fraction"),
])
PARAMETER_NAMES = ICE_SPACE.names
REPOSE_ACTIVE = ("mu", "e", "k_n", "k_t_ratio", "mu_r")

# Repose hull-SPEC: ~92 particles, 0.185 s simulated, ~15 s wall on one CPU.
REPOSE_RADIUS_M = 0.008
REPOSE_CYLINDER_M = 0.06
REPOSE_FILL_M = 0.07


def _lattice(psi: float) -> tuple[int, float]:
    """Map the h/(2R) ratio onto an integer layer count and a particle radius."""
    if not math.isfinite(psi) or psi < 3.0 or psi > 10.0:
        raise ValueError("psi must be finite and in [3, 10]")
    nz = int(round(psi))
    radius = ICE_THICKNESS_M / (2.0 * nz)
    return nz, radius


def wedge_case(u_unit, *, dt_factor: float = 1.0, trace_points: int = TRACE_POINTS) -> dict:
    """Run one wedge-indentation case and return its resistance diagnostics."""
    import torch

    from ..dem3d import DEM3DConfig
    from ..dem3d_wedge import MovingWedgeBowIceDEM3D

    torch.set_num_threads(1)
    u = np.asarray(u_unit, dtype=float).ravel()
    if u.size != len(ICE_SPACE):
        raise ValueError(f"expected {len(ICE_SPACE)} unit-cube parameters")
    physical = dict(zip(PARAMETER_NAMES, ICE_SPACE.to_physical(u)[0]))
    nz, radius = _lattice(physical["psi"])
    if not (0.0 < dt_factor <= 1.0):
        raise ValueError("dt_factor must be in (0, 1]")
    config = DEM3DConfig(
        nx=7, ny=3, nz=nz, radius=radius, density=917.0,
        bond_stiffness=BOND_STIFFNESS,
        contact_stiffness=physical["k_n"],
        tangential_stiffness=physical["k_t_ratio"] * physical["k_n"],
        breaking_strain=physical["eps_b"],
        shear_breaking_strain=2.0 * physical["eps_b"],
        contact_damping=0.5, drag=0.1, tool_speed=BOW_SPEED_M_S,
        friction_coefficient=physical["mu"],
        restitution_coefficient=physical["e"],
        rolling_resistance_coefficient=physical["mu_r"],
        gravity=9.81, water_level=ICE_THICKNESS_M / 2.0, water_density=1025.0,
        wet_drag_coefficient=physical["c_wet"],
        fix_x_edges=False, fix_bottom=False,
    )
    config = replace(config, dt=config.recommended_dt * dt_factor)
    sim = MovingWedgeBowIceDEM3D(
        config, bow_speed=BOW_SPEED_M_S, wedge_angle_deg=WEDGE_ANGLE_DEG,
        initial_gap=0.0, tip_height_fraction=0.5, bow_half_width_m=BOW_HALF_WIDTH_M,
    )
    steps = max(2, int(math.ceil(WINDOW_S / sim.dt)))
    start = time.time()
    resistance: list[float] = []
    broken: list[int] = []
    for _ in range(steps + 1):
        if resistance:
            sim.step()
        diag = sim.diagnostics()
        resistance.append(max(0.0, -float(diag["bow_reaction_x_N"])))
        broken.append(int(diag["broken_bonds"]))
    wall = time.time() - start
    values = np.asarray(resistance, dtype=float)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite resistance history")
    if values.size < 4:
        raise RuntimeError("resistance history too short to summarise")

    half = values[len(values) // 2:]
    first_break = next((i for i, b in enumerate(broken) if b > 0), None)
    channel = values[int(round(0.8 * (values.size - 1))):]
    indices = np.linspace(0, values.size - 1, trace_points).round().astype(int)
    trace = values[indices]
    if np.any(trace <= 0.0):
        # A calibration trace of exactly zero carries no information; use a
        # small positive floor so the log-scale diagnostics stay finite.  This
        # is recorded, never silently hidden.
        trace = np.maximum(trace, 1e-6)
    return {
        "R_N": float(half.mean()),
        "R_channel_N": float(channel.mean()),
        "R_peak_N": float(values.max()),
        "R_window_mean_N": float(values.mean()),
        "trace_N": trace.tolist(),
        "first_break_step": None if first_break is None else int(first_break),
        "first_break_time_s": None if first_break is None else float(first_break * sim.dt),
        "channel_established": bool(first_break is not None
                                    and first_break <= 0.6 * values.size),
        "broken_bonds": int(broken[-1]),
        "bonds_total": int(np.count_nonzero(np.asarray(sim.bonded.cpu()))),
        "steps": int(steps),
        "dt_s": float(sim.dt),
        "particles": int(len(sim.positions)),
        "layers": int(nz),
        "radius_m": float(radius),
        "travel_m": float(BOW_SPEED_M_S * steps * sim.dt),
        "wall_s": float(wall),
    }


def repose_case(u_unit) -> dict:
    """Run one heap case and return the angle of repose."""
    import torch

    from .. import repose_calibration as repose

    torch.set_num_threads(1)
    u = np.asarray(u_unit, dtype=float).ravel()
    physical = dict(zip(PARAMETER_NAMES, ICE_SPACE.to_physical(u)[0]))
    case = repose.ReposeCase(
        mu=physical["mu"],
        mu_rolling=physical["mu_r"],
        radius=REPOSE_RADIUS_M,
        kn=physical["k_n"] * 20.0,
        kt_ratio=physical["k_t_ratio"],
        restitution=physical["e"],
        cylinder_radius=REPOSE_CYLINDER_M,
        fill_height=REPOSE_FILL_M,
        hold_time=0.02,
        lift_time=0.06,
        settle_time=0.15,
        seed=4970,
    )
    start = time.time()
    out = repose.simulate(case)
    wall = time.time() - start
    theta = float(out["theta_r_deg"])
    if not math.isfinite(theta):
        raise FloatingPointError("non-finite repose angle")
    return {
        "theta_r_deg": theta,
        "theta_r_r2": float(out.get("r2", float("nan"))),
        "repose_particles": int(out.get("n_particles", -1)),
        "repose_r2": float(out.get("r2", float("nan"))),
        "repose_steps": int(out.get("steps", -1)),
        "wall_s": float(wall),
    }


def evaluate_spec(spec: dict) -> dict:
    """Evaluate one case specification (used by the process pool)."""
    kind = spec["kind"]
    u = np.asarray(spec["u"], dtype=float)
    if kind == "wedge":
        metrics = wedge_case(u, dt_factor=float(spec.get("dt_factor", 1.0)))
    elif kind == "repose":
        metrics = repose_case(u)
    else:
        raise ValueError(f"unknown case kind {kind!r}")
    return {"case_id": spec["case_id"], "kind": kind,
            "u_unit": u.tolist(), "dt_factor": float(spec.get("dt_factor", 1.0)),
            "tag": spec.get("tag", ""), "parent": spec.get("parent", ""),
            "metrics": metrics}


def _evaluate_cached(spec: dict, cache_dir: str) -> dict:
    path = Path(cache_dir) / f"{spec['case_id']}.json"
    if path.exists():
        with path.open() as handle:
            return json.load(handle)
    record = evaluate_spec(spec)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return record


def run_batch(specs, cache_dir, *, workers: int = 8) -> dict:
    """Evaluate ``specs`` with a process pool, caching one JSON file per case."""
    if workers < 1:
        raise ValueError("workers must be positive")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    specs = list(specs)
    pending = [s for s in specs if not (cache / f"{s['case_id']}.json").exists()]
    done, failures = 0, []
    if workers == 1:
        for spec in pending:
            try:
                _evaluate_cached(spec, str(cache))
                done += 1
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                failures.append({"case_id": spec["case_id"], "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_evaluate_cached, spec, str(cache)): spec for spec in pending}
            for future in futures:
                spec = futures[future]
                try:
                    future.result()
                    done += 1
                except Exception as exc:  # noqa: BLE001
                    failures.append({"case_id": spec["case_id"], "error": repr(exc)})
    return {
        "requested": len(specs),
        "cached": len(specs) - len(pending),
        "computed": done,
        "failures": failures,
        "cache_dir": str(cache),
    }


def load_cases(cache_dir, case_ids=None) -> list[dict]:
    """Load cached case records, skipping ids that were never computed."""
    cache = Path(cache_dir)
    records = []
    for path in sorted(cache.glob("*.json")):
        if case_ids is not None and path.stem not in case_ids:
            continue
        with path.open() as handle:
            records.append(json.load(handle))
    return records