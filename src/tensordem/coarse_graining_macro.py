"""Coarse-graining macro-invariance ledger (IN102-N02).

The CG question this module answers is falsifiable and quantitative:

    Under contact-time-invariant coarse graining (``k_n -> k^3 k_n``, geometry
    ``-> k L``, ``mu``/``mu_r``/``e``/``rho``/gravity/target stress fixed), does
    the *macroscopic* response of a granular assembly stay the same when the
    inertial number ``I = gamma_dot d / sqrt(P/rho)`` is matched?

    If yes, the coarse model is a valid surrogate above the contact level and
    the largest k for which every anchor stays inside tolerance is a *safe*
    coarse-graining factor.

Three macro anchors are read off one device (``simulate_servo_shear``):

    mu(I)          effective friction (rheology)
    Z              coordination number (contact network / jamming state)
    phi            solid fraction of the bulk slab (porosity)

plus the repose angle of a gravity-driven column collapse
(``simulate_repose_collapse``), which probes the same quasi-static limit from a
free-surface configuration.

``k``-matching recipe (all lengths scale, stress is invariant):

    radius      R_k    = k R_1
    box         L_k    = k L_1              (box / particle ratio fixed)
    stiffness   kn_k   = k^3 kn_1           (contact time invariant)
    shear rate  gd_k   = gd_1 / k           (I matched at fixed P)
    collapse    t_k    = sqrt(k) t_1        (Froude similarity, g fixed)

Acceptance thresholds (predeclared, from the campaign DAG):

    rheology   |mu_k(I) - mu_1(I)| / mu_1(I)   <= 5 %
    micro      |Z_k - Z_1| / Z_1, |phi_k - phi_1| / phi_1 <= 3 %
    repose     |theta_k - theta_1| / theta_1   <= 3 %

No threshold in this module is tuned to the observed data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

#: Inertial-number nodes of the rheology sweep (campaign DAG).
INERTIAL_TARGETS: tuple[float, ...] = (1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1)

#: Base (k = 1) material and device settings, SI.
BASE_MATERIAL: dict = {
    "radius": 0.005,
    "density": 2500.0,
    "mu": 0.5,
    "mu_rolling": 0.05,
    "restitution": 0.3,
    "kn": 6.25e4,
    "kt_ratio": 0.5,
}
BASE_GEOMETRY: dict = {
    "box_width": 0.12,
    "box_depth": 0.05,
    "fill_height": 0.06,
}
BASE_STRESS: float = 6.25e4  # Pa, servo-held normal stress (invariant in k)
BASE_SHEAR_STRAIN: float = 0.2
BASE_REPOSE: dict = {
    "cylinder_radius": 0.05,
    "fill_height": 0.10,
    "floor_radius": 0.18,
    "collapse_time": 0.40,
    "settle_time": 0.05,
    "hold_time": 0.05,
}

#: Predeclared tolerances (percent).
TOL_RHEOLOGY_PCT: float = 5.0
TOL_MICRO_PCT: float = 3.0
TOL_REPOSE_PCT: float = 3.0


#: Scaling recipes compared by the campaign.
#:   "A" contact-time invariant  k_n ~ k^3, t_c and restitution invariant,
#:       relative overlap u = P d / k_n ~ 1/k   (softness drifts with k)
#:   "B" sound-speed invariant   k_n ~ k,     u invariant, t_c ~ k
RECIPES = ("A", "B")


def build_plan(k: float, recipe: str = "A") -> dict:
    """Return the k-scaled device plan (case factories + the scaling ledger)."""
    if k < 1:
        raise ValueError("coarse-graining factor k must be >= 1")
    if recipe not in RECIPES:
        raise ValueError(f"recipe must be one of {RECIPES}")
    kf = float(k)
    mat = dict(BASE_MATERIAL)
    mat["radius"] = BASE_MATERIAL["radius"] * kf
    mat["kn"] = BASE_MATERIAL["kn"] * (kf ** 3 if recipe == "A" else kf)
    geom = {key: value * kf for key, value in BASE_GEOMETRY.items()}

    def case_factory(target: float) -> dict:
        return {
            **mat,
            **geom,
            "gravity": 0.0,
            "normal_stress": BASE_STRESS,
            "inertial_number": float(target),
            "shear_strain": BASE_SHEAR_STRAIN,
            "settle_steps": 2500,
            "min_shear_steps": 400,
            "seed": 5279,
        }

    repose = {
        "radius": mat["radius"],
        "density": mat["density"],
        "mu": mat["mu"],
        "mu_rolling": mat["mu_rolling"],
        "restitution": mat["restitution"],
        "kn": mat["kn"],
        "kt_ratio": mat["kt_ratio"],
        "cylinder_radius": BASE_REPOSE["cylinder_radius"] * kf,
        "fill_height": BASE_REPOSE["fill_height"] * kf,
        "floor_radius": BASE_REPOSE["floor_radius"] * kf,
        "collapse_time": BASE_REPOSE["collapse_time"] * math.sqrt(kf),
        "settle_time": BASE_REPOSE["settle_time"] * math.sqrt(kf),
        "hold_time": BASE_REPOSE["hold_time"] * math.sqrt(kf),
        "seed": 4970,
    }

    return {
        "k": kf,
        "material": mat,
        "geometry": geom,
        "normal_stress": BASE_STRESS,
        "case": case_factory,
        "repose": repose,
        "recipe": recipe,
        "scaling": {
            "recipe": ("A: contact-time invariant (kn ~ k^3)"
                       if recipe == "A" else
                       "B: sound-speed invariant (kn ~ k), relative overlap invariant"),
            "radius": "k R",
            "box_width/depth/fill_height": "k L",
            "kn": ("k^3 kn      (contact time invariant, u = P d/kn ~ 1/k)"
                   if recipe == "A" else
                   "k kn        (sound-speed invariant, u = P d/kn invariant)"),
            "kt": ("k^3 kt" if recipe == "A" else "k kt"),
            "density": "invariant",
            "mu / mu_rolling / restitution": "invariant",
            "normal_stress": "invariant (servo target)",
            "shear_rate": "gamma_dot / k   (I matched at fixed P)",
            "gravity": "invariant (0 in the shear box)",
            "collapse_time": "sqrt(k) t   (Froude similarity)",
        },
    }


def _interp_mu_at_targets(runs: list[dict]) -> tuple[list[float], list[float]]:
    """Interpolate mu onto the inertial-number nodes (log-I linear)."""
    i_meas = np.array([run["inertial_number"] for run in runs], dtype=float)
    mu_meas = np.array([run["mu_effective"] for run in runs], dtype=float)
    order = np.argsort(i_meas)
    i_sorted, mu_sorted = i_meas[order], mu_meas[order]
    mu_nodes = []
    inside = []
    for target in INERTIAL_TARGETS:
        # 5 % band: a run that lands on its I target within the control error
        # counts as covering that node (the residual is reported separately as
        # inertial_number_dev_pct_max)
        if i_sorted[0] * 0.95 <= target <= i_sorted[-1] * 1.05:
            inside.append(True)
            mu_nodes.append(float(np.interp(math.log10(target),
                                            np.log10(i_sorted), mu_sorted)))
        else:
            inside.append(False)
            mu_nodes.append(float(mu_sorted[0] if target < i_sorted[0]
                                  else mu_sorted[-1]))
    return mu_nodes, inside


def pareto_front(rows: list[dict]) -> list[dict]:
    """Non-dominated (speedup max, deviation min) subset, ordered by speedup."""
    front = []
    for row in rows:
        dominated = any(
            (other["speedup"] >= row["speedup"]
             and other["max_deviation_pct"] <= row["max_deviation_pct"])
            and (other["speedup"] > row["speedup"]
                 or other["max_deviation_pct"] < row["max_deviation_pct"])
            for other in rows
        )
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda r: r["speedup"])


def _bound(rows, key, tol):
    ok = [row["k"] for row in rows
          if row.get(key) is not None and row[key] <= tol]
    return float(max(ok)) if ok else 1.0


def safe_k_upper_bound(rows: list[dict],
                       keys=("rheology_dev_pct", "micro_dev_pct", "repose_dev_pct")) -> float:
    """Largest k whose every requested anchor stays inside its tolerance."""
    tol = {"rheology_dev_pct": TOL_RHEOLOGY_PCT, "micro_dev_pct": TOL_MICRO_PCT,
           "repose_dev_pct": TOL_REPOSE_PCT}
    ok = [row["k"] for row in rows
          if all(row.get(key) is not None and row[key] <= tol[key] for key in keys)]
    return float(max(ok)) if ok else 1.0


def shard_path(out_dir: Path, k: int, recipe: str = "A") -> Path:
    return (out_dir / f"k{k}" / "result.json" if recipe == "A"
            else out_dir / f"{recipe}_k{k}" / "result.json")


def aggregate(out_dir: str | Path, k_values=(1, 2, 4, 8), recipe: str = "A") -> dict:
    """Join the per-k shard results into the invariance / Pareto / k-bound ledger."""
    out_dir = Path(out_dir)
    shards = {}
    for k in k_values:
        path = shard_path(out_dir, k, recipe)
        if path.exists():
            shards[int(k)] = json.loads(path.read_text())
    if not shards:
        raise FileNotFoundError(f"no shard results found under {out_dir}")
    if 1 not in shards:
        raise FileNotFoundError("the k=1 baseline shard is required")

    base = shards[1]
    base_mu, _ = _interp_mu_at_targets(base["runs"])
    base_z = float(np.mean([r["coordination_number"] for r in base["runs"]]))
    base_phi = float(np.mean([r["solid_fraction"] for r in base["runs"]]))
    base_theta = base.get("repose", {}).get("repose_angle_deg")
    base_vol_rate = base.get("box_volume", 0.0) / max(base["wall_clock_s_total"], 1e-9)
    if base_vol_rate <= 0:
        box = BASE_GEOMETRY
        base_vol_rate = (box["box_width"] * box["box_depth"] * box["fill_height"]
                         / max(base["wall_clock_s_total"], 1e-9))

    rows = []
    for k, shard in sorted(shards.items()):
        mu_nodes, inside = _interp_mu_at_targets(shard["runs"])
        rheo_dev = [abs(m - b) / b * 100.0 if b else float("nan")
                    for m, b in zip(mu_nodes, base_mu)]
        z = float(np.mean([r["coordination_number"] for r in shard["runs"]]))
        phi = float(np.mean([r["solid_fraction"] for r in shard["runs"]]))
        z_dev = abs(z - base_z) / base_z * 100.0
        phi_dev = abs(phi - base_phi) / base_phi * 100.0
        theta = shard.get("repose", {}).get("repose_angle_deg")
        if theta is not None and isinstance(theta, float) and math.isnan(theta):
            theta = None
        repose_dev = (abs(theta - base_theta) / base_theta * 100.0
                      if theta and base_theta else None)
        box = BASE_GEOMETRY
        volume = (box["box_width"] * box["box_depth"] * box["fill_height"] * k ** 3)
        vol_rate = shard.get("box_volume", volume) * 0.0 + volume
        speedup = (vol_rate / max(shard["wall_clock_s_total"], 1e-9)) / base_vol_rate
        i_dev = [abs(r["inertial_number"] - r["inertial_number_target"])
                 / r["inertial_number_target"] * 100.0 for r in shard["runs"]]
        micro_dev = max(z_dev, phi_dev)
        anchor_devs = [d for d in (max(rheo_dev), micro_dev, repose_dev) if d is not None]
        rows.append({
            "k": k,
            "mu_at_targets": mu_nodes,
            "targets_inside_measured_range": inside,
            "rheology_dev_pct": max(rheo_dev),
            "rheology_dev_by_i_pct": rheo_dev,
            "coordination_number": z,
            "solid_fraction": phi,
            "coordination_dev_pct": z_dev,
            "solid_fraction_dev_pct": phi_dev,
            "micro_dev_pct": micro_dev,
            "repose_angle_deg": (None if theta is None
                                 or (isinstance(theta, float) and math.isnan(theta))
                                 else float(theta)),
            "repose_dev_pct": repose_dev,
            "max_deviation_pct": max(anchor_devs),
            "inertial_number_dev_pct_max": max(i_dev),
            "speedup": speedup,
            "wall_clock_s_total": shard["wall_clock_s_total"],
            "n_particles": shard["runs"][0]["n_particles"],
        })

    summary = {
        "campaign": "IN102-N02",
        "recipe": recipe,
        "title": "DEM coarse-graining macro anchors: mu(I), coordination, porosity, repose",
        "device": "tensordem.mu_i_rheology.simulate_servo_shear",
        "inertial_targets": list(INERTIAL_TARGETS),
        "normal_stress_pa": BASE_STRESS,
        "base_material": BASE_MATERIAL,
        "base_geometry": BASE_GEOMETRY,
        "scaling_rule": build_plan(1, recipe)["scaling"],
        "tolerances_pct": {
            "rheology": TOL_RHEOLOGY_PCT,
            "micro": TOL_MICRO_PCT,
            "repose": TOL_REPOSE_PCT,
        },
        "rows": rows,
        "max_rheology_dev_pct": max(row["rheology_dev_pct"] for row in rows),
        "max_micro_dev_pct": max(row["micro_dev_pct"] for row in rows),
        "max_repose_dev_pct": max((row["repose_dev_pct"] for row in rows
                                   if row["repose_dev_pct"] is not None), default=None),
        "safe_k_upper_bound": safe_k_upper_bound(rows),
        "safe_k_by_anchor": {
            "rheology": _bound(rows, "rheology_dev_pct", TOL_RHEOLOGY_PCT),
            "microstructure": _bound(rows, "micro_dev_pct", TOL_MICRO_PCT),
            "repose": _bound(rows, "repose_dev_pct", TOL_REPOSE_PCT),
            "rheology_and_microstructure": safe_k_upper_bound(
                rows, ("rheology_dev_pct", "micro_dev_pct")),
        },
        "pareto_front": pareto_front(rows),
        "k_available": sorted(shards),
    }
    return summary


def legacy_results(summary: dict) -> dict:
    """Map the macro ledger onto the legacy ``runs/cg_scaling/results.json`` keys.

    The legacy campaign file (contact scaling + per-friction-angle/repose blocks)
    is regenerated from *measured* N02 data so the older pytest items execute
    against real numbers instead of skipping.
    """
    from .coarse_graining import verify_contact_scaling

    contact = verify_contact_scaling(k_values=[1.0, 2.0, 4.0, 8.0],
                                     gravity=9.81, **BASE_MATERIAL)
    shear = {}
    repose = {}
    for row in summary["rows"]:
        key = f"{float(row['k']):.1f}"
        shear[key] = {
            "friction_angle_deg": math.degrees(math.atan(row["mu_at_targets"][0])),
            "mu_quasi_static": row["mu_at_targets"][0],
            "inertial_number": INERTIAL_TARGETS[0],
            "micro_dev_pct": row["micro_dev_pct"],
        }
        if row["repose_angle_deg"] is not None:
            repose[key] = {"theta_r_deg": row["repose_angle_deg"],
                           "dev_pct": row["repose_dev_pct"]}
    return {
        "campaign": "IN102-N02-legacy-bridge",
        "note": ("regenerated from measured servo-shear-cell data; the UCS / E_eff "
                 "block of the m7821 campaign is out of scope for N02 and stays absent"),
        "contact_scaling": contact,
        "shear": shear,
        "repose": repose,
        "macro_invariance": {
            "max_rheology_dev_pct": summary["max_rheology_dev_pct"],
            "max_micro_dev_pct": summary["max_micro_dev_pct"],
            "safe_k_upper_bound": summary["safe_k_upper_bound"],
        },
    }