"""Weakest-link (size-effect) verification of the heterogeneous 3-D DEM bond.

Why this campaign exists
------------------------
The gap ledger ``docs/dem_calibration_gap_ledger.md`` closes the bond's
constitutive rate dependence at the single-bond level and leaves the *specimen*
level open, with a measured negative result: the 4x4x3 opposed-platen specimen
is inertia dominated, so its emergent strength exponent could not be compared to
``1/(m+1)``.

One of the two routes named there is a *single-chain (weakest-link) driver*.
That is what this script builds, and it is deliberately cheaper and cleaner than
the platen specimen:

* strength **heterogeneity** is now a first-class, opt-in engine feature
  (``DEM3DConfig.bond_strength_weibull_modulus``): every bond draws its tensile
  breaking strain from a Weibull law whose *median* is ``breaking_strain``, so
  the material keeps its nominal strength and only the scatter is added;
* a chain of ``N`` bonds is loaded along its axis, so the assembly can only fail
  when its **weakest** bond fails.  The strength of a chain of ``N`` iid
  Weibull bonds is therefore a classical closed-form reference::

      P(chain survives sigma)  = exp(-N (sigma/sigma_0)**m)
      median(N)                = sigma_0 ln(2)**(1/m) N**(-1/m)
      median(N)/median(1)      = N**(-1/m)                      (size effect)

  with ``sigma_0`` fixed so that the single-bond median equals
  ``eps_b * k * L0``.  Nothing in the engine knows about ``N``: the size effect
  has to emerge from the per-bond draws plus the series load path.

What is measured
----------------
``arm="engine-uniform"``
    Exact, inertia-free arm.  All chain particles are placed so that every bond
    carries the *same* strain, then the engine's own ``forces()`` ->
    ``_commit_fracture`` path decides which bond dies.  Verifies (a) the
    per-bond threshold really is what the fracture criterion reads, and (b) that
    a series chain fails at its weakest bond.

``arm="ecdf"``
    High-statistics statistics arm: ``R`` realisations of the min of ``N`` iid
    draws (the same draw helper the engine uses) are compared to the analytic
    chain CDF above with a Kolmogorov-Smirnov statistic against the
    ``1.36/sqrt(R)`` 5 % band, for ``N = 1 ... 64``.

``arm="rate"``
    Quasi-static convergence of the *dynamic* engine: at fixed seeds and
    ``N = 4`` the ramp duration ``T`` is doubled; the measured load at first
    failure, the bond-force spread (the quasi-static diagnostic) and the
    identity of the bond that failed are reported for each ``T``.  This is how
    the validity window of the dynamic arm is established instead of assumed.

``arm="dynamic"``
    The engine is integrated (explicit dynamics, material drag only), a
    displacement ramp is applied to the top particle and the load at first
    failure is recorded per realisation.  Compared against the analytic
    weakest-link value of *that same* realisation (paired, so the multiplicative
    dynamic bias is visible instead of hidden).

Honest limits (recorded in the artefact, not just here)
------------------------------------------------------
* The chain is 1-D and its bonds are collinear, so no contact network, no
  bending and no shear coupling enter the strength; this verifies the *failure
  selection* and its statistics, not a bulk specimen.
* The dynamic arm carries a small rate-dependent bias, measured in the
  ``rate`` arm (bond-force spread ~ 2-5 % at the chosen ramp).
* Only the tensile threshold is heterogeneous; the shear criterion stays a
  scalar material constant.

Usage
-----
    PYTHONPATH=src python scripts/verify_ice_bond_weakest_link.py
    PYTHONPATH=src python scripts/verify_ice_bond_weakest_link.py --quick
    PYTHONPATH=src python scripts/verify_ice_bond_weakest_link.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.ice_bond_viscoelastic import weibull_bond_strengths

# ---------------------------------------------------------------------------
# Single source of truth for the campaign parameters
# ---------------------------------------------------------------------------
RADIUS = 0.025
L0 = 2.0 * RADIUS                 # chain bond rest length (m)
BOND_STIFFNESS = 2_000.0          # N/m
BREAKING_STRAIN = 0.015           # median tensile breaking strain
DENSITY = 917.0                   # kg/m^3
MODULUS = 2.0                     # Weibull modulus of the strength scatter
MODULUS_ALT = 4.0                 # second modulus, proves the slope is -1/m
DRAG = 2.0                        # N s/m material drag used by the dynamic arm
RAMP_DURATION_S = 2.5             # T of the dynamic arm (chosen by the rate arm)
OVERSHOOT_TARGET_STRAIN = 3.5     # ramp reaches 3.5 eps_b per bond: no censoring
ECDF_SAMPLES = 20_000
ECDF_SIZES = (1, 2, 4, 8, 16, 32, 64)
DYNAMIC_SIZES = (1, 2, 4, 8)
# Per-size realisation counts: the cheap short chains buy the statistics that
# make the log-log slope tight, the expensive long chains only have to show the
# trend.  Cost scales roughly linearly with N (the ramp covers N * 1.4 eps_b).
DYNAMIC_REALISATIONS = {1: 32, 2: 20, 4: 12, 8: 6}
DYNAMIC_REALISATIONS_QUICK = {1: 8, 2: 6, 4: 4, 8: 2}
RATE_STUDY_DURATIONS = (1.0, 2.5, 5.0)
RATE_STUDY_SEED = 3
ALT_MODULUS_REALISATIONS = 8

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ice_bond_weakest_link"


# ---------------------------------------------------------------------------
# Analytic references (computed from the same constants, nothing transcribed)
# ---------------------------------------------------------------------------
def weibull_scale(modulus: float = MODULUS) -> float:
    """Weibull scale of the *bond strength force* with median ``k eps_b L0``."""
    median_force = BOND_STIFFNESS * BREAKING_STRAIN * L0
    return median_force / math.log(2.0) ** (1.0 / modulus)


def chain_median_force(n_bonds: int, modulus: float = MODULUS) -> float:
    """Median strength of a chain of ``n_bonds`` iid bonds: ``F_0 (ln2/N)^(1/m)``."""
    scale = weibull_scale(modulus)
    return scale * (math.log(2.0) / n_bonds) ** (1.0 / modulus)


def chain_quantile_force(n_bonds: int, q: float, modulus: float = MODULUS) -> float:
    """``q``-quantile of the chain strength: ``F_0 (-ln(1-q)/N)^(1/m)``."""
    scale = weibull_scale(modulus)
    return scale * (-math.log1p(-q) / n_bonds) ** (1.0 / modulus)


def size_effect_exponent(modulus: float = MODULUS) -> float:
    """``median(N) ~ N**(-1/m)``."""
    return -1.0 / modulus


# ---------------------------------------------------------------------------
# Chain assembly on the real engine
# ---------------------------------------------------------------------------
def build_chain(n_bonds: int, seed: int, *, modulus: float = MODULUS,
                drag: float = DRAG) -> tuple[IceDEM3D, list[int]]:
    """Build a collinear ``n_bonds``-bond chain inside ``IceDEM3D``.

    The engine lattice is ``2 x 2 x (n_bonds+1)``; the ``x = y = 0`` column is
    moved onto the z axis and kept, every other particle is parked far away and
    clamped, and the bond list is reduced to the chain bonds.  The per-bond
    tensile thresholds are the ones the engine itself drew from
    ``bond_strength_weibull_modulus`` / ``bond_strength_seed`` (looked up by bond
    key), so the heterogeneity path under test is the production one.
    """
    config = DEM3DConfig(
        nx=2, ny=2, nz=n_bonds + 1, radius=RADIUS, density=DENSITY,
        bond_stiffness=BOND_STIFFNESS, breaking_strain=BREAKING_STRAIN,
        contact_damping=0.0, fix_x_edges=False, drag=drag,
        tool_gap=1.0, tool_speed=1e-6,          # park the cutting tool far away
        bond_strength_weibull_modulus=modulus, bond_strength_seed=seed,
    )
    sim = IceDEM3D(config)
    n_total = len(sim.positions)
    column = [4 * z for z in range(n_bonds + 1)]   # (z, y=0, x=0) -> z*ny*nx
    others = [i for i in range(n_total) if i not in column]

    positions = sim.positions.clone()
    positions[others] = torch.tensor(
        [[50.0 + 3.0 * i, 50.0, 50.0] for i in range(len(others))],
        dtype=sim.dtype, device=sim.device)
    z = torch.arange(n_bonds + 1, dtype=sim.dtype, device=sim.device) * L0
    positions[torch.tensor(column, device=sim.device)] = torch.stack(
        (torch.zeros_like(z), torch.zeros_like(z), z), dim=1)
    sim.positions = positions
    sim.initial_positions = positions.clone()
    sim.velocities.zero_()

    # Look the chain thresholds up in the engine's own draw (bond keys are
    # i * n_total + j on the original lattice).
    drawn = {}
    if sim.bond_breaking_strain is not None:
        for key, value in zip(sim._bond_keys.tolist(), sim.bond_breaking_strain.tolist()):
            drawn[int(key)] = float(value)

    pairs = torch.tensor([[column[k], column[k + 1]] for k in range(n_bonds)],
                         dtype=torch.long, device=sim.device)
    sim.pairs = pairs
    i, j = pairs.T
    reference = sim.positions[j] - sim.positions[i]
    sim.rest_lengths = torch.linalg.vector_norm(reference, dim=1)
    sim.reference_bond_vectors = reference.clone()
    sim.initial_normals = reference / sim.rest_lengths.clamp_min(1e-30)[:, None]
    sim.bonded = torch.ones(n_bonds, dtype=torch.bool, device=sim.device)
    sim.alive = sim.bonded.clone()
    sim.bond_indices = pairs
    sim.failure_mode = torch.zeros(n_bonds, dtype=torch.uint8, device=sim.device)
    sim.failure_time_s = torch.full((n_bonds,), -1.0, dtype=sim.dtype, device=sim.device)
    sim.failure_extension_m = torch.zeros(n_bonds, dtype=sim.dtype, device=sim.device)
    sim.failure_energy_J = torch.zeros(n_bonds, dtype=sim.dtype, device=sim.device)
    sim.bond_damage = torch.zeros(n_bonds, dtype=sim.dtype, device=sim.device)
    sim.branch_offsets = torch.zeros((n_bonds, 0), dtype=sim.dtype, device=sim.device)
    sim._bond_elongation_prev = torch.zeros(n_bonds, dtype=sim.dtype, device=sim.device)
    sim._bond_keys = pairs[:, 0] * n_total + pairs[:, 1]
    if drawn:
        thresholds = [drawn[int(key)] for key in sim._bond_keys.tolist()]
        sim.bond_breaking_strain = torch.tensor(thresholds, dtype=sim.dtype,
                                                device=sim.device)
        sim.bond_strength_factors = sim.bond_breaking_strain / BREAKING_STRAIN

    fixed = torch.zeros(n_total, dtype=torch.bool, device=sim.device)
    fixed[others] = True
    fixed[column[0]] = True          # support
    fixed[column[-1]] = True         # displacement-controlled top
    sim.fixed = fixed
    return sim, column


def bond_forces(sim: IceDEM3D) -> torch.Tensor:
    """Axial bond forces (N) without touching the history or the fracture state."""
    i, j = sim.pairs.T
    extension = (torch.linalg.vector_norm(sim.positions[j] - sim.positions[i], dim=1)
                 - sim.rest_lengths)
    return sim._bond_force(extension, update_history=False)


def _threshold_forces(sim: IceDEM3D) -> torch.Tensor:
    return BOND_STIFFNESS * sim.bond_breaking_strain * sim.rest_lengths


# ---------------------------------------------------------------------------
# Arm 1: exact weakest-link selection under a uniform strain field
# ---------------------------------------------------------------------------
def _uniform_run(n_bonds: int, seed: int, strain: float, modulus: float) -> dict:
    """Place every bond at the same strain and run the engine's fracture path."""
    sim, column = build_chain(n_bonds, seed, modulus=modulus)
    z = torch.arange(n_bonds + 1, dtype=sim.dtype, device=sim.device) * L0 * (1.0 + strain)
    positions = sim.positions.clone()
    positions[torch.tensor(column, device=sim.device)] = torch.stack(
        (torch.zeros_like(z), torch.zeros_like(z), z), dim=1)
    sim.positions = positions
    sim.initial_positions = positions.clone()
    sim.forces()                      # engine path: bond force -> _commit_fracture
    thresholds = _threshold_forces(sim)
    failed = torch.nonzero(~sim.alive).flatten().tolist()
    return {
        "n_bonds": n_bonds, "seed": seed, "imposed_strain": strain,
        "failed_bonds": [int(b) for b in failed],
        "n_failed": len(failed),
        "weakest_bond": int(torch.argmin(thresholds)),
        "threshold_forces_N": [float(v) for v in thresholds],
        "failure_modes": [int(sim.failure_mode[b]) for b in failed],
    }


def arm_engine_uniform(n_bonds: int, seed: int, modulus: float = MODULUS) -> dict:
    """Exact weakest-link selection: bracket the weakest threshold with two strains.

    ``strain_below`` sits just under the weakest bond's threshold (nothing may
    fail) and ``strain_above`` just over it but under the second weakest (exactly
    one bond -- the weakest -- may fail).  Both are read off the engine's own
    per-bond thresholds, so no assumption about the material is baked in.
    """
    probe, _ = build_chain(n_bonds, seed, modulus=modulus)
    thresholds, _ = torch.sort(_threshold_forces(probe))
    reference = BOND_STIFFNESS * L0
    lower_strain = 1.0e-3 * float(thresholds[0]) / reference
    upper_force = (0.5 * float(thresholds[0] + thresholds[1])
                   if len(thresholds) > 1 else 1.01 * float(thresholds[0]))
    upper_strain = upper_force / reference
    below = _uniform_run(n_bonds, seed, lower_strain, modulus)
    above = _uniform_run(n_bonds, seed, upper_strain, modulus)
    passed = (below["n_failed"] == 0
              and above["n_failed"] == 1
              and above["failed_bonds"] == [above["weakest_bond"]])
    return {"arm": "engine-uniform", "n_bonds": n_bonds, "seed": seed,
            "below": below, "above": above, "pass": passed}


# ---------------------------------------------------------------------------
# Arm 2: high-statistics ECDF of the chain strength vs the analytic Weibull CDF
# ---------------------------------------------------------------------------
def arm_ecdf(samples: int = ECDF_SAMPLES, modulus: float = MODULUS) -> dict:
    """Min of ``N`` iid draws vs ``exp(-N (s/sigma_0)^m)``, KS-banded."""
    generator = torch.Generator().manual_seed(20260101)
    rows = []
    for n_bonds in ECDF_SIZES:
        draws = weibull_bond_strengths(samples * n_bonds, 1.0, modulus,
                                       generator=generator)
        strength = draws.reshape(samples, n_bonds).min(dim=1).values * (
            BOND_STIFFNESS * BREAKING_STRAIN * L0)
        ordered, _ = torch.sort(strength)
        empirical = (torch.arange(1, samples + 1, dtype=torch.float64) / samples)
        analytic = 1.0 - torch.exp(
            -n_bonds * (ordered / weibull_scale(modulus)).pow(modulus))
        ks = float((empirical - analytic).abs().max())
        # Weibull-plot slope of the same sample: should be the *input* modulus.
        y = torch.log(-torch.log1p(-torch.clamp(empirical, max=1 - 1e-12)))
        x = torch.log(ordered)
        xm, ym = x.mean(), y.mean()
        slope = float(((x - xm) * (y - ym)).sum() / ((x - xm).square()).sum())
        rows.append({
            "n_bonds": n_bonds,
            "median_measured_N": float(strength.median()),
            "median_analytic_N": chain_median_force(n_bonds, modulus),
            "median_rel_error": float(strength.median()) / chain_median_force(n_bonds, modulus) - 1.0,
            "mean_measured_N": float(strength.mean()),
            "ks_statistic": ks,
            "ks_5pct_band": 1.36 / math.sqrt(samples),
            "ks_pass": ks < 1.36 / math.sqrt(samples),
            "fitted_modulus": slope,
        })
    return {"arm": "ecdf", "samples": samples, "modulus": modulus, "rows": rows}


def size_effect_fit_all_points(rows: list[dict]) -> dict:
    """Least squares on every realisation, ``log(load) ~ log(N)``, with bootstrap CI."""
    def fit(sample: list[dict]) -> float:
        xs = [math.log(r["n_bonds"]) for r in sample]
        ys = [math.log(r["load_N"]) for r in sample]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        return num / den if den else float("nan")

    slope = fit(rows)
    generator = __import__("random").Random(20260101)
    groups: dict[int, list[float]] = {}
    for row in rows:
        groups.setdefault(row["n_bonds"], []).append(row["load_N"])
    bootstrap = []
    for _ in range(400):
        sample = []
        for n_bonds, loads in groups.items():
            draw = generator.choices(loads, k=len(loads))
            sample.extend({"n_bonds": n_bonds, "load_N": value} for value in draw)
        bootstrap.append(fit(sample))
    bootstrap.sort()
    return {
        "fitted_slope": slope,
        "analytic_slope": size_effect_exponent(),
        "slope_rel_error": slope / size_effect_exponent() - 1.0,
        "bootstrap_ci95": [bootstrap[10], bootstrap[-11]],
        "realisations": len(rows),
        "sizes": sorted(groups),
    }


def size_effect_fit(sizes: list[int], medians: list[float]) -> dict:
    """Least-squares slope of ``log(median)`` against ``log(N)``."""
    xs = [math.log(float(n)) for n in sizes]
    ys = [math.log(float(m)) for m in medians]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    residual = max(abs(y - (my + slope * (x - mx))) for x, y in zip(xs, ys))
    return {"fitted_slope": slope, "analytic_slope": size_effect_exponent(),
            "slope_rel_error": slope / size_effect_exponent() - 1.0,
            "max_log_residual": residual, "n_points": n}


# ---------------------------------------------------------------------------
# Arms 3/4: the dynamic engine
# ---------------------------------------------------------------------------
def arm_dynamic(n_bonds: int, seed: int, ramp_duration: float = RAMP_DURATION_S,
                modulus: float = MODULUS, drag: float = DRAG) -> dict:
    """Ramp the top particle and record the load at the first bond failure."""
    sim, column = build_chain(n_bonds, seed, modulus=modulus, drag=drag)
    top = column[-1]
    base = L0 * n_bonds
    steps = int(round(ramp_duration / sim.dt))
    rate = OVERSHOOT_TARGET_STRAIN * BREAKING_STRAIN * L0 * n_bonds / ramp_duration
    thresholds = _threshold_forces(sim)
    weakest = int(torch.argmin(thresholds))
    analytic = float(thresholds.min())

    previous = None
    for step in range(1, steps + 1):
        sim.initial_positions[top, 2] = base + rate * step * sim.dt
        sim.step()
        if not bool(sim.alive.all()):
            alive = previous["alive"]
            load = float(previous["forces"][alive].mean())
            spread = float((previous["forces"][alive].max()
                            - previous["forces"][alive].min()) / abs(load))
            broken = int(torch.nonzero(~sim.alive)[0])
            return {
                "arm": "dynamic", "n_bonds": n_bonds, "seed": seed,
                "modulus": modulus, "ramp_duration_s": ramp_duration,
                "steps": step, "time_s": step * sim.dt,
                "dt_s": sim.dt, "strain_rate_s-1": rate / (L0 * n_bonds),
                "load_N": load, "analytic_weakest_link_N": analytic,
                "rel_error": load / analytic - 1.0,
                "force_spread": spread,
                "first_failed_bond": broken, "weakest_bond": weakest,
                "weakest_link_selected": broken == weakest,
                "failure_mode": int(sim.failure_mode[broken]),
                "n_failed": int((~sim.alive).sum()),
            }
        previous = {"forces": bond_forces(sim), "alive": sim.alive.clone()}
    return {"arm": "dynamic", "n_bonds": n_bonds, "seed": seed,
            "modulus": modulus, "ramp_duration_s": ramp_duration,
            "failed": False, "steps": steps, "strains_per_bond": OVERSHOOT_TARGET_STRAIN * 1.0}


def _dynamic_task(task: tuple[int, int, float, float]) -> dict:
    n_bonds, seed, duration, modulus = task
    return arm_dynamic(n_bonds, seed, duration, modulus)


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------
def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def run_campaign(workers: int, quick: bool) -> dict:
    sizes = (1, 2) if quick else DYNAMIC_SIZES
    realisations = DYNAMIC_REALISATIONS_QUICK if quick else DYNAMIC_REALISATIONS
    ecdf_samples = 2_000 if quick else ECDF_SAMPLES
    durations = (1.0,) if quick else RATE_STUDY_DURATIONS
    alt_realisations = 2 if quick else ALT_MODULUS_REALISATIONS

    # --- arm 1: exact selection under uniform strain -----------------------
    uniform_rows = [arm_engine_uniform(n_bonds, seed)
                    for n_bonds in sizes for seed in range(4)]
    uniform_pass = all(r["pass"] for r in uniform_rows)

    # --- arm 2: statistics ------------------------------------------------
    ecdf = arm_ecdf(ecdf_samples)
    ecdf_fit = size_effect_fit([r["n_bonds"] for r in ecdf["rows"]],
                               [r["median_measured_N"] for r in ecdf["rows"]])

    # --- arm 3: quasi-static convergence of the dynamic arm ---------------
    # --- arms 3 and 4 are both dynamic: dispatch them in one pool --------
    tasks = [(n_bonds, 1000 + seed, RAMP_DURATION_S, MODULUS)
             for n_bonds in sizes for seed in range(realisations[n_bonds])]
    tasks += [(4, 5000 + seed, RAMP_DURATION_S, MODULUS_ALT)
              for seed in range(alt_realisations)]
    rate_tasks = [(4, RATE_STUDY_SEED, duration, MODULUS) for duration in durations]
    all_tasks = tasks + rate_tasks
    if workers > 1 and len(all_tasks) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(
                workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
            results = list(pool.imap(_dynamic_task, all_tasks))
    else:
        results = [_dynamic_task(task) for task in all_tasks]
    dynamic_rows = results[:len(tasks)]
    rate_rows = results[len(tasks):]

    main_rows = [r for r in dynamic_rows
                 if r["modulus"] == MODULUS and r.get("failed", True)]
    alt_rows = [r for r in dynamic_rows
                if r["modulus"] == MODULUS_ALT and r.get("failed", True)]
    no_failure = [r for r in dynamic_rows if not r.get("failed", True)]

    per_size = {}
    for n_bonds in sizes:
        rows = [r for r in main_rows if r["n_bonds"] == n_bonds]
        loads = [r["load_N"] for r in rows]
        per_size[n_bonds] = {
            "realisations": len(rows),
            "median_load_N": _median(loads),
            "analytic_median_N": chain_median_force(n_bonds),
            "median_rel_error": _median(loads) / chain_median_force(n_bonds) - 1.0,
            "max_abs_rel_error": max(abs(r["rel_error"]) for r in rows),
            "median_force_spread": _median([r["force_spread"] for r in rows]),
            "weakest_link_rate": sum(r["weakest_link_selected"] for r in rows) / len(rows),
        }
    dynamic_fit = size_effect_fit(list(per_size), [per_size[n]["median_load_N"]
                                                   for n in per_size])
    dynamic_fit_all = size_effect_fit_all_points(main_rows)
    for n_bonds in sizes:
        rows = [r for r in main_rows if r["n_bonds"] == n_bonds]
        per_size[n_bonds]["median_paired_rel_error"] = _median(
            [r["rel_error"] for r in rows])
    alt_fit = ({}
               if not alt_rows else
               {"modulus": MODULUS_ALT,
                "analytical_slope": size_effect_exponent(MODULUS_ALT),
                "n_rows": len(alt_rows),
                "median_rel_error": _median([r["rel_error"] for r in alt_rows]),
                "weakest_link_rate": sum(r["weakest_link_selected"]
                                         for r in alt_rows) / len(alt_rows)})

    payload = {
        "campaign": "ice_bond_weakest_link",
        "material": {
            "radius_m": RADIUS, "rest_length_m": L0,
            "bond_stiffness_N_m": BOND_STIFFNESS,
            "median_breaking_strain": BREAKING_STRAIN,
            "density_kg_m3": DENSITY,
            "weibull_modulus": MODULUS,
            "weibull_modulus_alt": MODULUS_ALT,
            "drag_N_s_m": DRAG,
            "ramp_duration_s": RAMP_DURATION_S,
        },
        "analytic_references": {
            "bond_strength_scale_N": weibull_scale(),
            "single_bond_median_force_N": chain_median_force(1),
            "chain_median_force_N": {str(n): chain_median_force(n)
                                     for n in (1, 2, 4, 8, 16, 32, 64)},
            "size_effect_exponent": size_effect_exponent(),
            "size_effect_exponent_alt_modulus": size_effect_exponent(MODULUS_ALT),
        },
        "arm_engine_uniform": {"rows": uniform_rows, "pass": uniform_pass},
        "arm_ecdf": {**ecdf, "size_effect_fit": ecdf_fit},
        "arm_rate": {"rows": rate_rows},
        "arm_dynamic": {
            "rows": dynamic_rows,
            "rows_without_failure": len(no_failure),
            "per_size": {str(k): v for k, v in per_size.items()},
            "size_effect_fit": dynamic_fit,
            "size_effect_fit_all_points": dynamic_fit_all,
            "alt_modulus": alt_fit,
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="smoke run for CI (few realisations, short ramp)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)

    payload = run_campaign(max(1, args.workers), args.quick)
    OUT.mkdir(parents=True, exist_ok=True)
    path = args.out or OUT / ("quick.json" if args.quick else "verification.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("arm 1 - exact weakest-link selection (uniform strain field)")
    print(f"  rows={len(payload['arm_engine_uniform']['rows'])}  "
          f"pass={payload['arm_engine_uniform']['pass']}")
    print("\narm 2 - chain strength ECDF vs analytic Weibull")
    print("  N   median(N)   analytic    rel.err    KS      band     m_fit")
    for row in payload["arm_ecdf"]["rows"]:
        print(f"  {row['n_bonds']:3d} {row['median_measured_N']:.6f} "
              f"{row['median_analytic_N']:.6f} {row['median_rel_error']:+.2e} "
              f"{row['ks_statistic']:.4f} {row['ks_5pct_band']:.4f}  "
              f"{row['fitted_modulus']:.3f}")
    print(f"  size-effect slope {ecdf_fit_slope(payload):+.4f} "
          f"(analytic {size_effect_exponent():+.4f})")
    print("\narm 3 - quasi-static convergence of the dynamic engine (N=4)")
    print("  T [s]   steps  load [N]  rel.err   spread   weakest-link")
    for row in payload["arm_rate"]["rows"]:
        print(f"  {row['ramp_duration_s']:5.2f} {row['steps']:6d} "
              f"{row['load_N']:.4f}  {row['rel_error']:+.3%}  {row['force_spread']:5.2%}   "
              f"{row['weakest_link_selected']}")
    print("\narm 4 - dynamic ensemble")
    for key, row in payload["arm_dynamic"]["per_size"].items():
        print(f"  N={key:>2}  R={row['realisations']:3d}  median load {row['median_load_N']:.4f} N "
              f"(analytic {row['analytic_median_N']:.4f})  rel.err "
              f"{row['median_rel_error']:+.3%}  weakest-link {row['weakest_link_rate']:.0%}  "
              f"spread {row['median_force_spread']:.2%}")
    dynamic_fit = payload["arm_dynamic"]["size_effect_fit"]
    print(f"  dynamic size-effect slope {dynamic_fit['fitted_slope']:+.4f} "
          f"(analytic {dynamic_fit['analytic_slope']:+.4f}, rel.err "
          f"{dynamic_fit['slope_rel_error']:+.2%}, max log residual "
          f"{dynamic_fit['max_log_residual']:.3f})")
    print(f"\nwrote {path}")
    return 0


def ecdf_fit_slope(payload: dict) -> float:
    return payload["arm_ecdf"]["size_effect_fit"]["fitted_slope"]


if __name__ == "__main__":
    raise SystemExit(main())