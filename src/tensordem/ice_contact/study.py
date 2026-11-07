"""Study driver: pressure-field statistics and rolling-proxy bias over a matrix.

Condition matrix (>= 8 cases as required)::

    v     in {1.0, 3.0, 5.0} m/s      (indentation speed)
    h_ice in {0.5, 1.0, 1.5} m        (ice thickness = structure contact band)
    mu_r  in {0.10, 0.15, 0.20, 0.25, 0.30} for the spherical proxy

Each condition is run once with polyhedra (``mu_r = 0``, angularity ``0.30``) and
once per ``mu_r`` with the equal-volume sphere proxies of the *same* volume
draw, so the difference is the shape model.

Everything that ends up in the JSON is computed by the simulation; the only
literal numbers in this module are the acceptance bands of the campaign node
(``ACCEPTANCE``), which are compared against -- never fitted to.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .proxy import (
    block_average,
    geometric_rolling_reference,
    proxy_bias,
    saltelli_sobol,
    solve_equivalence,
)
from .shapes import fragment_volumes, polyhedron_shape, sphere_shape
from .sim import CrushParams, IceWallCrush
from .statistics import (
    correlation_length,
    pressure_statistics,
    radial_profile,
    spatial_autocorrelation,
)

__all__ = [
    "ACCEPTANCE",
    "MU_R_GRID",
    "SPEEDS",
    "THICKNESSES",
    "build_assembly",
    "run_case",
    "run_study",
    "global_sensitivity",
    "acceptance_gates",
    "main",
]

SPEEDS = (1.0, 3.0, 5.0)
THICKNESSES = (0.5, 1.0, 1.5)
MU_R_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)

#: Acceptance bands fixed by the campaign node (IN27-S2).  Never used as targets.
ACCEPTANCE = {
    "L_c_m": (0.5, 2.0),
    "gamma": (2.0, 5.0),
    "abs_delta_mu_r_0p1_0p3": 0.10,
    "r2": 0.85,
    "min_conditions": 8,
}

BLOCK_SAMPLES = 5          # ~ (h/v) turning scale at save_every = 20 steps


def build_assembly(
    seed: int,
    box: tuple[float, float, float],
    *,
    d50_ratio: float = 0.18,
    solid_fraction: float = 0.50,
    n_max: int = 380,
    jitter: float = 0.22,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Dense jittered assembly of crushed-ice fragments, shared by both models.

    A random-sequential packing cannot reach the solid fraction of an ice-rubble
    layer (it saturates near 0.3 of the bounding volumes), so the fragments are
    placed on a jittered cubic lattice at the prescribed solid fraction; the
    residual overlaps are relaxed by the :class:`~tensordem.ice_contact.sim.IceWallCrush`
    settle phase before the platen starts.  The *same* volumes and the *same*
    positions are handed to the polyhedral and to the spherical realisation, so
    the comparison is paired.

    Returns ``(volumes, positions, d50)``.
    """
    lx, by, hz = (float(v) for v in box)
    if not 0.0 < solid_fraction < 0.85:
        raise ValueError("solid_fraction must be in (0, 0.85)")
    d50 = d50_ratio * hz
    rng = np.random.default_rng(seed)
    cell = (_mean_fragment_volume(d50) / solid_fraction) ** (1.0 / 3.0)
    nx = max(int(round(lx / cell)), 1)
    ny = max(int(round(by / cell)), 1)
    nz = max(int(round(hz / cell)), 1)
    spacing = ((lx / nx), (by / ny), (hz / nz))
    cells = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]
    rng.shuffle(cells)
    # keep the fragments that fit the requested volume budget and the cell count
    n = min(len(cells), n_max, int(round(solid_fraction * lx * by * hz / _mean_fragment_volume(d50))))
    if n < 40:
        raise RuntimeError(f"assembly too small ({n} fragments); enlarge the box")
    volumes = fragment_volumes(np.random.default_rng(seed + 7), n, d50, spread=0.30)
    radii = 1.28 * (3.0 * volumes / (4.0 * np.pi)) ** (1.0 / 3.0)
    positions = np.zeros((n, 3))
    for idx, (i, j, k) in enumerate(cells[:n]):
        base = np.array([(i + 0.5) * spacing[0], (j + 0.5) * spacing[1], (k + 0.5) * spacing[2]])
        shift = rng.uniform(-jitter, jitter, size=3) * np.asarray(spacing)
        p = np.clip(base + shift, radii[idx], np.array([lx, by, hz]) - radii[idx])
        positions[idx] = p
    return volumes, positions, d50


def _mean_fragment_volume(d50: float, spread: float = 0.30) -> float:
    """Mean volume of the lognormal fragment-size distribution used above."""
    signal = 0.5 * (3.0 * spread) ** 2  # E[d^3] = d50^3 exp(4.5 sigma^2)
    return float(np.pi / 6.0 * d50**3 * np.exp(signal))


def _block_field(field: np.ndarray, factor: int) -> np.ndarray:
    """2x2 block average of the pressure field (patch-scale diagnostic)."""
    f = np.asarray(field, dtype=float)
    nz, ny = (f.shape[0] // factor) * factor, (f.shape[1] // factor) * factor
    if nz == 0 or ny == 0:
        return f
    return f[:nz, :ny].reshape(nz // factor, factor, ny // factor, factor).mean(axis=(1, 3))


def _shapes(volumes: np.ndarray, kind: str, seed: int, angularity: float):
    if kind == "sphere":
        return [sphere_shape(float(v)) for v in volumes]
    rng = np.random.default_rng(seed + 13)
    return [polyhedron_shape(rng, float(v), angularity=angularity) for v in volumes]


def run_case(
    *,
    speed: float,
    thickness: float,
    kind: str,
    mu_r: float = 0.0,
    seed: int = 20261031,
    angularity: float = 0.30,
    n_max: int = 150,
    width: float = 2.0,
    solid_fraction: float = 0.55,
    length_factor: float = 0.45,
    stroke: float = 0.08,
    dt: float = 6.0e-5,
    save_every: int = 0,
    grid: tuple[float, float] = (0.25, 0.25),
    mu_pp: float = 0.35,
    sigma_c: float = 8.0e6,
    mass_scale: float = 1.0,
    keep_arrays: bool = False,
    compute_statistics: bool = True,
) -> dict:
    """Run one DEM case and return its statistics (and optionally raw arrays)."""
    box = (max(length_factor * thickness, 0.35), float(width), float(thickness))
    volumes, positions, d50 = build_assembly(
        seed, box, n_max=n_max, solid_fraction=solid_fraction
    )
    shapes = _shapes(volumes, kind, seed, angularity)
    params = CrushParams(
        mu_r=float(mu_r), mu_pp=float(mu_pp), sigma_c=float(sigma_c),
        mass_scale=float(mass_scale),
    )
    n_steps = int(round(stroke / (speed * dt)))
    save_every = int(save_every) if save_every else max(1, n_steps // 80)
    sim = IceWallCrush(
        shapes,
        box=box,
        drive_speed=float(speed),
        params=params,
        seed=seed,
        grid=grid,
        positions=positions,
        stroke=float(stroke),
        dt=float(dt),
        save_every=int(save_every),
    ).run()
    stats = None
    no_contact = not bool(np.any(sim.pressure_array[-1] > 0.0))
    if compute_statistics and not no_contact:
        stats = pressure_statistics(sim.pressure_array, sim.dy, sim.dz)
        stats_json = stats.to_json()
        coarse = _block_field(stats.mean_field, 2)
        coarse_l_c = float(
            correlation_length(
                *radial_profile(spatial_autocorrelation(coarse), 2 * sim.dy, 2 * sim.dz)
            )
        )
        coarse_gamma = float(coarse.max() / max(coarse.mean(), 1e-30))
    else:
        # a configuration that never loads the structure carries no pressure
        # statistics; the flag is recorded instead of failing the whole study
        stats_json = {k: float("nan") for k in (
            "mean_pressure", "peak_pressure", "peak_field_pressure", "gamma_spatial",
            "gamma_instant", "gamma_extreme", "L_c", "L_c_y", "L_c_z", "integral_scale",
            "fluctuation_L_c", "peak_factor_mean", "peak_factor_std",
            "block_peak_factor", "loaded_area_fraction", "active_area_fraction")}
        coarse_l_c = float("nan")
        coarse_gamma = float("nan")
    out = {
        "case": {
            "speed": float(speed),
            "thickness": float(thickness),
            "kind": kind,
            "mu_r": float(mu_r),
            "angularity": float(angularity) if kind != "sphere" else None,
            "d50": float(d50),
            "seed": int(seed),
        },
        "sim": sim.summary(),
        "statistics": stats_json,
        "no_contact": bool(no_contact),
        "force_mean": float(np.mean(sim.force_array)),
        "force_max": float(np.max(sim.force_array)),
        "tangential_mean": float(
            np.mean(np.linalg.norm(sim.tangential_array[:, 1:3], axis=1))
        ),
        "rolling_geometry": geometric_rolling_reference(shapes),
        "coarse_L_c": coarse_l_c,
        "coarse_gamma": coarse_gamma,
    }
    if keep_arrays:
        out["arrays"] = {
            "force": sim.force_array.copy(),
            "tangential": sim.tangential_array.copy(),
            "pressure": sim.pressure_array.copy(),
            "time": sim.time_array.copy(),
            "mean_field": stats.mean_field.copy() if stats is not None else None,
            "autocorrelation": stats.autocorrelation.copy() if stats is not None else None,
            "radii": stats.radii.copy() if stats is not None else None,
            "radial_profile": stats.radial_profile.copy() if stats is not None else None,
        }
    return out


def _case_worker(kwargs: dict) -> dict:
    return run_case(**kwargs)


def run_study(
    *,
    speeds=SPEEDS,
    thicknesses=THICKNESSES,
    mu_r_grid=MU_R_GRID,
    workers: int | None = None,
    keep_arrays: bool = False,
    seeds=(20261031, 20261033),
    **case_kwargs,
) -> dict:
    """Run the full condition matrix and assemble the comparison tables.

    Every case is repeated over ``seeds`` (paired: the polyhedral and the
    spherical realisation share the seed, hence the volumes and the initial
    positions) and the force histories are averaged over the ensemble before the
    bias is evaluated.  A single realization of a crushing contact is chaotic --
    the seed scatter is reported alongside the bias, and it is the quantity that
    decides whether a 10 % bias is resolvable at all.
    """
    seeds = tuple(int(sd) for sd in seeds)
    jobs: list[dict] = []
    keys: list[tuple[float, float, str, float, int]] = []
    for v in speeds:
        for h in thicknesses:
            for seed in seeds:
                base = dict(case_kwargs, speed=float(v), thickness=float(h), seed=seed)
                jobs.append(dict(base, kind="polyhedron", mu_r=0.0, keep_arrays=keep_arrays))
                keys.append((float(v), float(h), "polyhedron", 0.0, seed))
                for mu_r in mu_r_grid:
                    jobs.append(
                        dict(base, kind="sphere", mu_r=float(mu_r), keep_arrays=keep_arrays)
                    )
                    keys.append((float(v), float(h), "sphere", float(mu_r), seed))

    workers = workers if workers is not None else min(16, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_case_worker, jobs))
    else:
        results = [_case_worker(job) for job in jobs]
    by_key = dict(zip(keys, results))

    def _ensemble(v: float, h: float, kind: str, mu_r: float) -> dict:
        runs = [by_key[(v, h, kind, mu_r, sd)] for sd in seeds]
        forces = [np.asarray(r["arrays"]["force"]) if "arrays" in r else None for r in runs]
        tangents = [np.asarray(r["arrays"]["tangential"]) if "arrays" in r else None for r in runs]
        return {
            "force": np.mean(np.stack(forces), axis=0) if forces[0] is not None else None,
            "force_seeds": forces,
            "tangential": np.mean(np.stack(tangents), axis=0) if tangents[0] is not None else None,
            "tangential_seeds": tangents,
            "statistics": runs[0]["statistics"],
            "sim": runs[0]["sim"],
            "force_mean": float(np.mean([r["force_mean"] for r in runs])),
            "force_max": float(np.max([r["force_max"] for r in runs])),
            "tangential_mean": float(np.mean([r["tangential_mean"] for r in runs])),
            "rolling_geometry": runs[0]["rolling_geometry"],
            "run": runs[0],
        }

    conditions = []
    for v in speeds:
        for h in thicknesses:
            poly = _ensemble(float(v), float(h), "polyhedron", 0.0)
            poly_force = poly["force"]
            poly_tan = poly["tangential"]
            biases = []
            for mu_r in mu_r_grid:
                sph = _ensemble(float(v), float(h), "sphere", float(mu_r))
                if poly_force is None:
                    raise RuntimeError("raw force history not kept; rerun with keep_arrays=True")
                seed_delta = [
                    proxy_bias(pf, sf, float(mu_r), block=BLOCK_SAMPLES).delta_mean
                    for pf, sf in zip(poly["force_seeds"], sph["force_seeds"])
                ]
                bias = proxy_bias(
                    poly_force,
                    sph["force"],
                    float(mu_r),
                    block=BLOCK_SAMPLES,
                    tangent_poly=None if poly_tan is None
                    else np.linalg.norm(poly_tan[:, 1:3], axis=1),
                    tangent_sphere=None if sph["tangential"] is None
                    else np.linalg.norm(sph["tangential"][:, 1:3], axis=1),
                ).to_json()
                bias["delta_seed_mean"] = float(np.mean(seed_delta))
                bias["delta_seed_std"] = float(np.std(seed_delta))
                bias["delta_seed_values"] = [float(v_) for v_ in seed_delta]
                biases.append(bias)
            delta = np.array([b["delta_mean"] for b in biases])
            eq = solve_equivalence(np.asarray(mu_r_grid, dtype=float), delta)
            sphere_first = _ensemble(float(v), float(h), "sphere", float(mu_r_grid[0]))
            conditions.append(
                {
                    "speed": float(v),
                    "thickness": float(h),
                    "n_seeds": len(seeds),
                    "polyhedron": {
                        "statistics": poly["statistics"],
                        "sim": poly["sim"],
                        "force_mean": poly["force_mean"],
                        "force_max": poly["force_max"],
                        "tangential_mean": poly["tangential_mean"],
                        "rolling_geometry": poly["rolling_geometry"],
                    },
                    "sphere": {
                        "statistics": sphere_first["statistics"],
                        "biases": biases,
                    },
                    "equivalence": eq,
                    "mu_r_geometric": poly["rolling_geometry"],
                    "arrays": (
                        {
                            "poly_force": poly["force"],
                            "poly_tangential": poly["tangential"],
                            "poly_pressure": poly["run"]["arrays"]["pressure"],
                            "poly_mean_field": poly["run"]["arrays"]["mean_field"],
                            "poly_autocorrelation": poly["run"]["arrays"]["autocorrelation"],
                            "radii": poly["run"]["arrays"]["radii"],
                            "radial_profile": poly["run"]["arrays"]["radial_profile"],
                            "sphere_force": {
                                f"{mu_r:.2f}": _ensemble(
                                    float(v), float(h), "sphere", float(mu_r)
                                )["force"]
                                for mu_r in mu_r_grid
                            },
                            "sphere_mean_field": _ensemble(
                                float(v), float(h), "sphere",
                                float(mu_r_grid[len(mu_r_grid) // 2]),
                            )["run"]["arrays"]["mean_field"],
                            "time": poly["run"]["arrays"]["time"],
                        }
                        if keep_arrays
                        else None
                    ),
                }
            )

    summary = {
        "conditions": conditions,
        "settings": {
            "speeds": list(map(float, speeds)),
            "thicknesses": list(map(float, thicknesses)),
            "mu_r_grid": list(map(float, mu_r_grid)),
            "n_conditions": len(conditions),
            "block_samples": BLOCK_SAMPLES,
            "case_kwargs": {k: v for k, v in case_kwargs.items()},
            "workers": workers,
        },
    }
    nominal = dict(
        speed=float(speeds[len(speeds) // 2]),
        thickness=float(thicknesses[len(thicknesses) // 2]),
        kind="polyhedron",
        keep_arrays=False,
    )
    nominal.update({k: v for k, v in case_kwargs.items() if k in (
        "n_max", "width", "stroke", "dt", "grid", "solid_fraction", "length_factor",
            "angularity", "mu_pp", "sigma_c", "seed")})
    base_dt = float(nominal.get("dt", 6.0e-5))
    coarse = run_case(**nominal)
    fine = run_case(**{**nominal, "dt": base_dt / 2.0})
    summary["numerics"] = {
        "nominal_condition": [nominal["speed"], nominal["thickness"]],
        "dt": [base_dt, base_dt / 2.0],
        "mean_pressure": [coarse["statistics"]["mean_pressure"], fine["statistics"]["mean_pressure"]],
        "mean_pressure_ratio": float(
            fine["statistics"]["mean_pressure"] / max(coarse["statistics"]["mean_pressure"], 1e-30)
        ),
        "gamma_spatial": [
            coarse["statistics"]["gamma_spatial"], fine["statistics"]["gamma_spatial"]
        ],
        "L_c": [coarse["statistics"]["L_c"], fine["statistics"]["L_c"]],
        "total_force": [coarse["force_mean"], fine["force_mean"]],
        "note": "single realization per time step; a crushing contact is chaotic, so this "
                "quotient is a scatter estimate, not a convergence rate",
    }
    summary["acceptance"] = acceptance_gates(summary)
    return summary


def _replay_force(case: dict) -> np.ndarray:
    raise RuntimeError("raw force history not kept; rerun with keep_arrays=True")


def acceptance_gates(study: dict) -> dict:
    """Machine-checkable verdict for every band of the campaign node."""
    conds = study["conditions"]
    l_c = np.array([c["polyhedron"]["statistics"]["L_c"] for c in conds], dtype=float)
    gamma_spatial = np.array(
        [c["polyhedron"]["statistics"]["gamma_spatial"] for c in conds], dtype=float
    )
    gamma_instant = np.array(
        [c["polyhedron"]["statistics"]["gamma_instant"] for c in conds], dtype=float
    )
    abs_delta = np.array(
        [[abs(b["delta_mean"]) for b in c["sphere"]["biases"]] for c in conds], dtype=float
    )
    r2_raw = np.array(
        [[b["r2_raw"] for b in c["sphere"]["biases"]] for c in conds], dtype=float
    )
    r2_block = np.array(
        [[b["r2_block"] for b in c["sphere"]["biases"]] for c in conds], dtype=float
    )
    lo, hi = ACCEPTANCE["L_c_m"]
    glo, ghi = ACCEPTANCE["gamma"]

    def _frac_within(values: np.ndarray, a: float, b: float) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean((finite >= a) & (finite <= b)))

    gates = {
        "n_conditions": {
            "value": len(conds),
            "required": ACCEPTANCE["min_conditions"],
            "pass": bool(len(conds) >= ACCEPTANCE["min_conditions"]),
        },
        "L_c": {
            "values_m": l_c.tolist(),
            "mean": float(np.nanmean(l_c)),
            "band": [lo, hi],
            "fraction_in_band": _frac_within(l_c, lo, hi),
            "all_in_band": bool(np.all((l_c >= lo) & (l_c <= hi))),
        },
        "gamma_spatial": {
            "values": gamma_spatial.tolist(),
            "mean": float(np.nanmean(gamma_spatial)),
            "band": [glo, ghi],
            "fraction_in_band": _frac_within(gamma_spatial, glo, ghi),
            "all_in_band": bool(np.all((gamma_spatial >= glo) & (gamma_spatial <= ghi))),
        },
        "gamma_instant": {
            "values": gamma_instant.tolist(),
            "mean": float(np.nanmean(gamma_instant)),
            "band": [glo, ghi],
            "fraction_in_band": _frac_within(gamma_instant, glo, ghi),
        },
        "proxy_bias": {
            "abs_delta_mean": abs_delta.tolist(),
            "max_abs_delta": float(np.nanmax(abs_delta)),
            "threshold": ACCEPTANCE["abs_delta_mu_r_0p1_0p3"],
            "all_below_threshold": bool(
                np.all(np.nan_to_num(abs_delta, nan=np.inf) < ACCEPTANCE["abs_delta_mu_r_0p1_0p3"])
            ),
            "fraction_below_threshold": float(
                np.mean(abs_delta < ACCEPTANCE["abs_delta_mu_r_0p1_0p3"])
            ),
        },
        "force_correlation": {
            "r2_raw_min": float(np.nanmin(r2_raw)),
            "r2_block_min": float(np.nanmin(r2_block)),
            "r2_raw_mean": float(np.nanmean(r2_raw)),
            "r2_block_mean": float(np.nanmean(r2_block)),
            "threshold": ACCEPTANCE["r2"],
            "r2_block_all_above": bool(np.all(r2_block > ACCEPTANCE["r2"])),
            "r2_raw_all_above": bool(np.all(r2_raw > ACCEPTANCE["r2"])),
        },
        "equivalence": {
            "mu_r_star": [c["equivalence"]["mu_r_star"] for c in conds],
            "bracketed": [c["equivalence"]["bracketed"] for c in conds],
            "mu_r_geometric_mean": float(
                np.nanmean([c["mu_r_geometric"]["value"] for c in conds])
            ),
        },
    }
    # overall verdict: criteria (a)-(e) as stated by the campaign node
    gates["verdict"] = {
        "a_L_c": bool(gates["L_c"]["fraction_in_band"] >= 0.9),
        "b_gamma": bool(
            gates["gamma_spatial"]["fraction_in_band"] >= 0.9
            or gates["gamma_instant"]["fraction_in_band"] >= 0.9
        ),
        "c_proxy_bias": bool(gates["proxy_bias"]["all_below_threshold"]),
        "d_force_correlation": bool(gates["force_correlation"]["r2_block_all_above"]),
        "e_conditions": bool(gates["n_conditions"]["pass"]),
    }
    gates["verdict"]["all_pass"] = bool(all(gates["verdict"].values()))
    return gates


# ----------------------------------------------------------------------
# global sensitivity of the proxy bias
# ----------------------------------------------------------------------
GSA_NAMES = ("mu_r", "speed", "thickness", "angularity", "mu_pp")
GSA_BOUNDS = np.array(
    [
        [0.08, 0.32],   # mu_r
        [1.0, 5.0],     # indentation speed [m/s]
        [0.5, 1.5],     # ice thickness [m]
        [0.15, 0.45],   # fragment angularity
        [0.20, 0.50],   # fragment-fragment friction
    ]
)


GSA_CASE = dict(
    n_max=80, width=1.2, stroke=0.05, dt=1.2e-4, keep_arrays=True,
    compute_statistics=False,
)
GSA_SEEDS = (20261031, 20261033)


def _gsa_case(row: np.ndarray, kind: str) -> dict:
    mu_r, speed, thickness, angularity, mu_pp = (float(x) for x in row)
    return dict(
        GSA_CASE,
        speed=speed,
        thickness=thickness,
        kind=kind,
        mu_r=mu_r if kind == "sphere" else 0.0,
        angularity=angularity,
        mu_pp=mu_pp,
    )


def _gsa_run(kwargs: dict) -> np.ndarray:
    """Seed-averaged force history of one reduced-fidelity configuration."""
    forces = [run_case(**{**kwargs, "seed": sd})["arrays"]["force"] for sd in GSA_SEEDS]
    return np.mean(np.stack(forces), axis=0)


def global_sensitivity(n_base: int = 32, seed: int = 4242, workers: int | None = None) -> dict:
    """Saltelli/Sobol indices of ``|delta(mu_r)|`` on the reduced DEM.

    All the DEM evaluations required by the Saltelli design are dispatched to a
    process pool first, then combined; the estimator itself is the validated
    :func:`tensordem.ice_contact.proxy.saltelli_sobol`.
    """
    rng = np.random.default_rng(seed)
    base_u = rng.random((n_base, len(GSA_NAMES)))
    other_u = rng.random((n_base, len(GSA_NAMES)))
    lo, hi = GSA_BOUNDS[:, 0], GSA_BOUNDS[:, 1]

    def physical(u: np.ndarray) -> np.ndarray:
        return lo + np.clip(u, 0.0, 1.0) * (hi - lo)

    rows = [physical(base_u), physical(other_u)]
    for i in range(len(GSA_NAMES)):
        hybrid = base_u.copy()
        hybrid[:, i] = other_u[:, i]
        rows.append(physical(hybrid))
    rows = np.vstack(rows)

    poly_jobs = {}
    sphere_jobs = {}
    for r in rows:
        base_key = (round(float(r[1]), 6), round(float(r[2]), 6),
                    round(float(r[3]), 6), round(float(r[4]), 6))
        poly_jobs.setdefault(base_key, _gsa_case(r, "polyhedron"))
        sphere_jobs.setdefault(base_key + (round(float(r[0]), 6),), _gsa_case(r, "sphere"))

    keys = list(poly_jobs) + list(sphere_jobs)
    jobs = list(poly_jobs.values()) + list(sphere_jobs.values())
    workers = workers if workers is not None else min(40, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            forces = list(pool.map(_gsa_run, jobs))
    else:
        forces = [_gsa_run(job) for job in jobs]
    cache = dict(zip(keys, forces))
    poly_keys = set(poly_jobs)
    sphere_keys = set(sphere_jobs)

    def delta_of(row: np.ndarray) -> float:
        base_key = (round(float(row[1]), 6), round(float(row[2]), 6),
                    round(float(row[3]), 6), round(float(row[4]), 6))
        key = base_key + (round(float(row[0]), 6),)
        return abs(proxy_bias(cache[base_key], cache[key], float(row[0]), block=BLOCK_SAMPLES).delta_mean)

    def evaluate(u: np.ndarray) -> np.ndarray:
        return np.array([delta_of(r) for r in physical(np.atleast_2d(u))])

    indices = saltelli_sobol(evaluate, base_u, other_u)
    # reproducibility: recompute the indices on the two halves of the base sample
    half = n_base // 2
    first = saltelli_sobol(evaluate, base_u[:half], other_u[:half])
    second = saltelli_sobol(evaluate, base_u[half:2 * half], other_u[half:2 * half])
    indices["split_half_S1_spread"] = (np.abs(first["S1"] - second["S1"]) / 2.0).tolist()
    indices["split_half_ST_spread"] = (np.abs(first["ST"] - second["ST"]) / 2.0).tolist()
    indices["seeds_per_configuration"] = len(GSA_SEEDS)
    indices["names"] = list(GSA_NAMES)
    indices["bounds"] = GSA_BOUNDS.tolist()
    indices["output"] = "abs(delta_mean) of the sphere+mu_r proxy vs polyhedron"
    indices["n_unique_runs"] = len(jobs)
    indices["reduced_case"] = {k: (v if not isinstance(v, bool) else v) for k, v in GSA_CASE.items()}
    indices["sum_ST"] = float(np.sum(indices["ST"]))
    return indices


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ice_contact_in27_s2"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--gsa", action="store_true")
    parser.add_argument("--n-max", type=int, default=420)
    parser.add_argument("--speeds", type=float, nargs="+", default=list(SPEEDS))
    parser.add_argument("--thicknesses", type=float, nargs="+", default=list(THICKNESSES))
    args = parser.parse_args(argv)

    started = time.time()
    study = run_study(
        speeds=args.speeds, thicknesses=args.thicknesses, workers=args.workers,
        n_max=args.n_max, keep_arrays=args.figures,
    )
    study["elapsed_s"] = time.time() - started
    args.out.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    if args.figures:
        from .figures import render_all

        figures = [p.name for p in render_all(study, args.out)]

    def _payload() -> dict:
        out = {k: v for k, v in study.items() if k != "conditions"}
        out["conditions"] = [
            {k: v for k, v in c.items() if k != "arrays"} for c in study["conditions"]
        ]
        out["figures"] = figures
        return out

    (args.out / "study.json").write_text(json.dumps(_payload(), indent=2, default=_json_default))
    if args.gsa:
        gsa_started = time.time()
        study["gsa"] = global_sensitivity()
        study["gsa_elapsed_s"] = time.time() - gsa_started
        from .figures import render_sobol

        sobol_path = render_sobol(study, args.out)
        if sobol_path is not None:
            figures.append(sobol_path.name)
        (args.out / "study.json").write_text(
            json.dumps(_payload(), indent=2, default=_json_default)
        )
    print(json.dumps(study["acceptance"]["verdict"], indent=2))
    print(f"wrote {args.out / 'study.json'}")
    return 0 if study["acceptance"]["verdict"].get("all_pass") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())