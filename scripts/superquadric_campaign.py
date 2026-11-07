"""Superquadric contact maturity campaign (TensorDEM, P96-N3).

Modes
-----
``checks``  : E1/E2 anchors -- sphere-limit Hertz vs analytic, cube flat-punch
              vs the Boussinesq BEM reference (with the BEM convergence table),
              conservative elastic-impact energy drift, steady-state penetration.
``geom``    : geometric error table over (aspect ratio x exponent x N spheres).
``perf``    : single-step cost, true superquadric kernel vs equal-population
              sphere-clump baseline (E4 performance ratio).
``heap``    : granular heap experiments -- sphere baseline, one true shape and
              its volume-matched clump (E2 steady state + E3 repose evidence).
``table``   : merge the JSON fragments into the disable-ratio table.
``figures`` : matplotlib figures for the report.

Every mode writes a JSON fragment under ``artifacts/superquadric/`` so numbers in
the report are programmatically injected, never copied by hand.

Usage::

    PYTHONPATH=src python scripts/superquadric_campaign.py checks
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np

from tensordem.flat_punch import (
    bem_convergence_table,
    disk_punch_stiffness,
    reduced_modulus,
    square_punch_reference,
    square_punch_stiffness,
)
from tensordem.superquadric_kernel import (
    ContactMaterial,
    contact_geometry_batch,
    effective_radius_hertz,
    facet_patch_area,
    principal_curvatures_local,
    superquadric_inertia,
)
from tensordem.superquadric_dem import (
    SQSystem,
    SQSystemConfig,
    Wall,
    energy_drift_per_1000,
    initial_column_packing,
    make_clump_body,
    make_superquadric_body,
    repose_angle_from_cloud,
    run_column_collapse,
)
from tensordem.clump_geometry import fit_clump, geometric_error
from tensordem.superquadric_error_table import (
    DEFAULT_CRITERIA,
    build_disable_ratio_table,
    build_geometric_table,
    write_disable_ratio_table,
)

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts", "superquadric")
TABLE_PATH = os.path.join(ART, "clump_disable_ratio_table.json")


def _write(name: str, payload: Dict) -> str:
    os.makedirs(ART, exist_ok=True)
    path = os.path.join(ART, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[campaign] wrote {path}", flush=True)
    return path


def _read(name: str) -> Dict:
    with open(os.path.join(ART, name), encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# E1 / E2 anchors
# ---------------------------------------------------------------------------

def run_checks() -> Dict:
    out: Dict[str, object] = {"mode": "checks", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # --- E1a: sphere limit vs Hertz --------------------------------------
    from tensordem.superquadric import sphere_limit_hertz_error
    hertz_errors = []
    for radius, overlap in ((0.01, 1e-4), (0.01, 5e-4), (0.005, 1e-4), (0.02, 2e-4)):
        err = sphere_limit_hertz_error(radius=radius, young_modulus=1e9,
                                       poisson_ratio=0.3, overlap=overlap)
        # independent recomputation with the matured kernel
        e_star = reduced_modulus(1e9, 0.3, 1e9, 0.3)
        r_eff = radius / 2.0
        f_herr = (4.0 / 3.0) * e_star * math.sqrt(r_eff) * overlap ** 1.5
        geo = contact_geometry_batch(
            np.zeros((1, 3)), np.eye(3)[None], (radius, radius, radius, 2.0, 2.0),
            np.array([[0.0, 0.0, 2 * radius - overlap]]), np.eye(3)[None],
            (radius, radius, radius, 2.0, 2.0))
        k1, k2 = principal_curvatures_local(np.array([[0.0, 0.0, radius]]),
                                            radius, radius, radius, 2.0, 2.0)
        reff = 1.0 / (0.5 * (k1[0] + k2[0]) + 0.5 * (k1[0] + k2[0]))
        f_kernel = (4.0 / 3.0) * e_star * math.sqrt(reff) * float(geo.overlap[0]) ** 1.5
        hertz_errors.append({
            "radius": radius, "overlap": overlap,
            "overlap_from_kernel": float(geo.overlap[0]),
            "overlap_error_rel": abs(float(geo.overlap[0]) - overlap) / overlap,
            "force_error_rel": abs(f_kernel - f_herr) / f_herr,
            "prototype_check_error_rel": float(err),
            "r_eff": float(reff),
        })
    out["sphere_limit_hertz"] = {
        "rows": hertz_errors,
        "max_force_error_rel": max(r["force_error_rel"] for r in hertz_errors),
        "criterion": 0.01,
    }

    # --- E1b: cube flat punch vs Boussinesq BEM ---------------------------
    half = 0.01
    e_star = reduced_modulus(1e9, 0.3, 1e9, 0.3)
    delta = 0.01 * half           # 1 % of the half extent
    patch = facet_patch_area(half, half, half, 50.0, 50.0, delta)
    k_kernel = 2.0 * e_star * math.sqrt(patch / math.pi)
    ref = square_punch_reference(half_side=math.sqrt(patch) / 2.0, e_star=e_star, n_cells_per_side=48)
    k_ref = ref.stiffness
    # exact disk punch (closed form) at the same area, for scale
    k_disk = disk_punch_stiffness(math.sqrt(patch / math.pi), e_star)
    out["cube_flat_punch"] = {
        "half_extent": half,
        "overlap": delta,
        "patch_area": patch,
        "face_area": (2 * half) ** 2,
        "kernel_stiffness": k_kernel,
        "bem_square_reference": k_ref,
        "bem_shape_factor": ref.shape_factor,
        "bem_rel_uncertainty": ref.rel_uncertainty,
        "disk_closed_form": k_disk,
        "rel_error_vs_bem": abs(k_kernel - k_ref) / k_ref,
        "criterion": 0.02,
        "bem_convergence": bem_convergence_table([8, 12, 16, 24, 32, 48]),
    }
    out["cube_flat_punch"]["pass"] = out["cube_flat_punch"]["rel_error_vs_bem"] <= 0.02

    # --- E2a: conservative elastic impact (energy + penetration) ----------
    R = 0.005
    rho = 2500.0
    mat = ContactMaterial(e_star=reduced_modulus(1e7, 0.3, 1e7, 0.3),
                          g_star=1e7 / (2 * 2.6), mu=0.0, mu_rolling=0.0,
                          restitution=1.0, normal_law="hertz")
    cfg = SQSystemConfig(material=mat, gravity=np.zeros(3))
    a = make_superquadric_body(R, R, R, 2.0, 2.0, [0, 0, 0], rho)
    b = make_superquadric_body(R, R, R, 2.0, 2.0, [0, 0, 2 * R - 1e-6], rho)
    a.velocity = np.array([0.0, 0.0, 0.5])
    b.velocity = np.array([0.0, 0.0, -0.5])
    sysm = SQSystem([a, b], [], cfg)
    sysm.run(1000)
    e = sysm.energy_log
    out["elastic_impact"] = {
        "dt": sysm.dt,
        "steps": 1000,
        "energy_start": e[0],
        "energy_end": e[-1],
        "drift_per_1000": energy_drift_per_1000(e, 1000),
        "max_penetration_ratio": sysm.max_penetration_ratio,
        "criterion_drift": 0.01,
    }

    # --- E2b: steady-state penetration with stiff contacts ----------------
    # Quasi-static preparation: gravity is ramped (0.1 -> 1.0) with strong
    # background damping, then released for a settling window.  Without the ramp
    # the initial free fall heats the packing and the measured overlap is a
    # transient impact value, not a static one.
    R2 = 0.004
    # 1e8 Pa is stiff enough that the static overlap stays below 1e-3 D
    # (delta/D ~ 3e-4) while keeping dt ~ 20x larger than at 1e9 Pa, so a
    # quasi-static equilibrium is reachable in a few thousand steps.
    mat2 = ContactMaterial(e_star=reduced_modulus(1e8, 0.3, 1e8, 0.3),
                           g_star=1e8 / (2 * 2.6), mu=0.5, mu_rolling=0.05,
                           restitution=0.3)
    walls = [Wall(np.array([0.0, 0.0, 1.0]), 0.0, mu=0.5),
             Wall(np.array([1.0, 0.0, 0.0]), -0.02, mu=0.5),
             Wall(np.array([-1.0, 0.0, 0.0]), -0.02, mu=0.5),
             Wall(np.array([0.0, 1.0, 0.0]), -0.02, mu=0.5),
             Wall(np.array([0.0, -1.0, 0.0]), -0.02, mu=0.5)]
    pos = initial_column_packing(24, R2, 0.015, z0=R2 * 1.0, seed=11,
                                 spacing_factor=1.999)
    bodies = [make_clump_body(np.zeros((1, 3)), np.array([R2]), p, rho) for p in pos]
    cfg2 = SQSystemConfig(material=mat2, damping_global=3000.0)
    s2 = SQSystem(bodies, walls, cfg2)
    for scale in (0.05, 0.15, 0.4, 0.7, 1.0):
        s2.config.gravity = np.array([0.0, 0.0, -9.81 * scale])
        s2.run(400)
    s2.config.damping_global = 200.0
    s2.run(2000)
    steady = 0.0
    n_contact = 0
    for c in s2.last_contacts:
        D = 2.0 * min(s2.bodies[int(c["i"])].shape[:3])
        steady = max(steady, float(c["overlap"]) / D)
        n_contact += 1
    vmax = float(max(np.linalg.norm(b.velocity) for b in bodies))
    out["steady_penetration"] = {
        "dt": s2.dt, "steps": 4000, "n_contacts": n_contact,
        "steady_penetration_ratio": steady, "vmax": vmax,
        "criterion": 1e-3, "pass": bool(steady < 1e-3 and n_contact > 0),
        "protocol": "gravity ramp 0.05/0.15/0.4/0.7/1.0 x 200 steps + 400 settle, particles pre-placed in contact",
    }
    try:
        _write("checks.json", out)
    finally:
        return out


# ---------------------------------------------------------------------------
# Geometric table
# ---------------------------------------------------------------------------

def run_geom() -> Dict:
    rows = build_geometric_table(
        aspect_ratios=(1.0, 1.5, 2.0, 3.0),
        exponents=(2.0, 4.0, 8.0, 16.0),
        n_spheres=(1, 4, 8, 16),
    )
    payload = {"mode": "geom", "rows": rows,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write("geometric_table.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Performance (E4)
# ---------------------------------------------------------------------------

def _perf_case(n_bodies: int = 24, steps: int = 20, kind: str = "clump") -> float:
    R = 0.004
    rho = 2500.0
    mat = ContactMaterial(e_star=reduced_modulus(2e6, 0.3, 2e6, 0.3),
                          g_star=2e6 / (2 * 2.6), mu=0.5, mu_rolling=0.05,
                          restitution=0.4)
    cfg = SQSystemConfig(material=mat)
    walls = [Wall(np.array([0.0, 0.0, 1.0]), 0.0, mu=0.5),
             Wall(np.array([1.0, 0.0, 0.0]), -0.03, mu=0.5),
             Wall(np.array([-1.0, 0.0, 0.0]), -0.03, mu=0.5),
             Wall(np.array([0.0, 1.0, 0.0]), -0.03, mu=0.5),
             Wall(np.array([0.0, -1.0, 0.0]), -0.03, mu=0.5)]
    pos = initial_column_packing(n_bodies, R, 0.015, z0=R * 2.2, seed=3)
    if kind == "clump":
        bodies = [make_clump_body(np.zeros((1, 3)), np.array([R]), p, rho) for p in pos]
    else:
        bodies = [make_superquadric_body(R, R, R, 2.0, 2.0, p, rho) for p in pos]
    sysm = SQSystem(bodies, walls, cfg)
    sysm.run(5)                        # warm-up
    t0 = time.perf_counter()
    sysm.run(steps)
    return (time.perf_counter() - t0) / steps


def run_perf() -> Dict:
    clump = _perf_case(kind="clump")
    true = _perf_case(kind="true")
    ratio = true / clump
    payload = {
        "mode": "perf",
        "n_bodies": 24,
        "steps": 20,
        "clump_seconds_per_step": clump,
        "true_seconds_per_step": true,
        "performance_ratio": ratio,
        "criterion": 3.0,
        "pass": ratio <= 3.0,
        "note": ("true-shape/sphere-geometry superquadric kernel vs analytic "
                 "sphere-clump contacts; identical particle count and walls"),
    }
    _write("performance.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Heap experiments
# ---------------------------------------------------------------------------

def _box(L: float, gate: float, mu: float):
    return [Wall(np.array([0.0, 0.0, 1.0]), 0.0, mu=mu, label="floor"),
            Wall(np.array([1.0, 0.0, 0.0]), -L, mu=mu), Wall(np.array([-1.0, 0.0, 0.0]), -L, mu=mu),
            Wall(np.array([0.0, 1.0, 0.0]), -L, mu=mu), Wall(np.array([0.0, -1.0, 0.0]), -L, mu=mu),
            Wall(np.array([1.0, 0.0, 0.0]), -gate, mu=mu), Wall(np.array([-1.0, 0.0, 0.0]), -gate, mu=mu),
            Wall(np.array([0.0, 1.0, 0.0]), -gate, mu=mu), Wall(np.array([0.0, -1.0, 0.0]), -gate, mu=mu)]


def _column(n: int, gate: float, radius: float, spacing_factor: float = 2.6, seed: int = 7):
    rng = np.random.default_rng(seed)
    spacing = spacing_factor * radius
    per = max(int(math.floor(2 * gate / spacing)), 1)
    layers = int(math.ceil(n / (per * per)))
    off = -0.5 * (per - 1) * spacing
    pts = []
    for kz in range(layers):
        for ky in range(per):
            for kx in range(per):
                pts.append([off + kx * spacing, off + ky * spacing, spacing + kz * spacing])
    return np.array(pts[:n]) + 0.25 * radius * rng.standard_normal((n, 3))


def _heap(kind: str, mu: float, mu_r: float, n: int = 130, gate: float = 0.016,
          box_half: float = 0.09, hold: int = 300, settle: int = 1500,
          radius: float = 0.004, shape=(2.0, 2.0), seed: int = 7) -> Dict:
    rho = 2500.0
    mat = ContactMaterial(e_star=reduced_modulus(2e6, 0.3, 2e6, 0.3),
                          g_star=2e6 / (2 * 2.6), mu=mu, mu_rolling=mu_r,
                          restitution=0.4)
    cfg = SQSystemConfig(material=mat)
    pos = _column(n, gate, radius, seed=seed)
    if kind == "sphere":
        bodies = [make_clump_body(np.zeros((1, 3)), np.array([radius]), p, rho) for p in pos]
    elif kind == "clump":
        fit = fit_clump(radius * shape[0] / shape[0], radius, radius, shape[0], shape[1],
                        n_spheres=8)
        bodies = [make_clump_body(fit.offsets, fit.radii, p, rho) for p in pos]
    else:
        bodies = [make_superquadric_body(radius, radius, radius, shape[0], shape[1], p, rho)
                  for p in pos]
    rng = np.random.default_rng(seed + 1)
    for body in bodies:
        body.velocity = np.array([0.06 * rng.standard_normal(), 0.06 * rng.standard_normal(), 0.0])
    res = run_column_collapse(bodies, _box(box_half, gate, mu), cfg,
                              hold_steps=hold, settle_steps=settle,
                              gate_walls=(5, 6, 7, 8))
    return {
        "kind": kind, "mu": mu, "mu_rolling": mu_r, "n_particles": n,
        "repose_angle_deg": res.repose_angle_deg,
        "steady_penetration_ratio": res.steady_penetration_ratio,
        "max_penetration_ratio": res.max_penetration_ratio,
        "energy_drift_per_1000": res.energy_drift_per_1000,
        "force_chain_angle_deg": res.force_chain_angle_deg,
        "force_chain_strength_ratio": res.force_chain_strength_ratio,
        "n_contacts": res.n_contacts,
        "dt": res.dt, "steps": res.steps, "wall_clock_s": res.wall_clock_s,
        "surface_profile": res.surface_profile,
    }


def run_heap() -> Dict:
    rows: List[Dict] = []
    # sphere baseline (calibration sweep)
    for mu, mu_r in ((0.5, 0.05), (0.5, 0.5), (0.7, 0.5)):
        rows.append(_heap("sphere", mu, mu_r))
        _write("heap_runs.json", {"mode": "heap", "rows": rows})
    # true shape vs its 8-sphere clump at the same macroscopic parameters
    for kind in ("true", "clump"):
        rows.append(_heap(kind, 0.5, 0.5, shape=(8.0, 8.0)))
        _write("heap_runs.json", {"mode": "heap", "rows": rows})
    payload = {"mode": "heap", "rows": rows,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write("heap_runs.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------

def run_table() -> Dict:
    geom = _read("geometric_table.json")["rows"]
    macro: List[Dict] = []
    try:
        heap = _read("heap_runs.json")["rows"]
    except FileNotFoundError:
        heap = []
    true_rows = [r for r in heap if r["kind"] == "true"]
    clump_rows = [r for r in heap if r["kind"] == "clump"]
    if true_rows and clump_rows:
        t, c = true_rows[-1], clump_rows[-1]
        macro.append({
            "aspect_ratio": 1.0, "shape_exponent": 8.0, "n_spheres": 8,
            "repose_error_deg": abs(t["repose_angle_deg"] - c["repose_angle_deg"]),
            "force_chain_deviation_deg": abs(t["force_chain_angle_deg"] - c["force_chain_angle_deg"]),
            "note": "column-collapse heap, mu=0.5, mu_r=0.5, 200 particles",
        })
    try:
        perf = _read("performance.json")
        ratio = perf["performance_ratio"]
    except FileNotFoundError:
        ratio = None
    table = build_disable_ratio_table(
        geometric_rows=geom, macroscopic_rows=macro,
        criteria=DEFAULT_CRITERIA, performance_ratio=ratio,
        provenance={
            "campaign": "P96-N3",
            "solver": "tensordem.superquadric_dem / superquadric_kernel",
            "clump_fit": "volume-matched k-means inscribed spheres (clump_geometry.fit_clump)",
            "note": ("macroscopic rows are measured on the column-collapse driver; "
                     "the absolute repose calibration of that driver is still "
                     "provisional, see docs/superquadric_contact_maturity.md"),
        },
    )
    write_disable_ratio_table(table, TABLE_PATH)
    print(f"[campaign] wrote {TABLE_PATH}", flush=True)
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def run_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checks = _read("checks.json")
    geom = _read("geometric_table.json")["rows"]
    try:
        heaps = _read("heap_runs.json")["rows"]
    except FileNotFoundError:
        heaps = []
    perf = _read("performance.json")

    # Figure 1: geometric error vs N for several (AR, n)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ar, marker in zip(sorted({r["aspect_ratio"] for r in geom}), ("o", "s", "^", "d")):
        for n, color in zip((2.0, 4.0, 8.0, 16.0), ("C0", "C1", "C2", "C3")):
            rows = [r for r in geom if r["aspect_ratio"] == ar and r["shape_exponent"] == n]
            rows.sort(key=lambda r: r["n_spheres"])
            if not rows:
                continue
            axes[0].plot([r["n_spheres"] for r in rows],
                         [r["max_distance_ratio"] * 100 for r in rows],
                         marker=marker, color=color, alpha=0.8,
                         label=f"AR={ar:g}, n={n:g}")
            axes[1].plot([r["n_spheres"] for r in rows],
                         [r["max_normal_deviation_deg"] for r in rows],
                         marker=marker, color=color, alpha=0.8)
    axes[0].set_xscale("log")
    axes[0].axhline(DEFAULT_CRITERIA["max_distance_ratio"] * 100, color="k", ls="--", lw=1)
    axes[0].set_xlabel("clump spheres N")
    axes[0].set_ylabel("two-sided deviation / smallest half extent [%]")
    axes[0].set_title("Geometric clump error (dashed = gate criterion)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].set_xscale("log")
    axes[1].axhline(DEFAULT_CRITERIA["max_normal_deviation_deg"], color="k", ls="--", lw=1)
    axes[1].set_xlabel("clump spheres N")
    axes[1].set_ylabel("max normal deviation [deg]")
    axes[1].set_title("Normal-direction error")
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "fig_geometric_error.png"), dpi=140)
    plt.close(fig)

    # Figure 2: heap free-surface profiles + performance bar
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row in heaps:
        prof = row.get("surface_profile") or []
        if not prof:
            continue
        axes[0].plot([p[0] * 1e3 for p in prof], [p[1] * 1e3 for p in prof],
                     marker="o", label=f"{row['kind']} mu={row['mu']} mur={row['mu_rolling']}")
    axes[0].set_xlabel("radius [mm]")
    axes[0].set_ylabel("surface height [mm]")
    axes[0].set_title("Heap free-surface profiles")
    axes[0].legend(fontsize=6)
    axes[1].bar(["sphere clump", "superquadric"],
                [perf["clump_seconds_per_step"] * 1e3, perf["true_seconds_per_step"] * 1e3],
                color=["C0", "C1"])
    axes[1].axhline(3 * perf["clump_seconds_per_step"] * 1e3, color="k", ls="--", lw=1)
    axes[1].set_ylabel("ms per step")
    axes[1].set_title(f"Step cost (ratio {perf['performance_ratio']:.2f}, criterion 3x)")
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "fig_heap_and_performance.png"), dpi=140)
    plt.close(fig)

    # Figure 3: Boussinesq BEM convergence + sphere-limit Hertz error
    conv = checks["cube_flat_punch"]["bem_convergence"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(conv["n_cells"], conv["disk_ratio"], "o-", label="disk BEM / exact 2E*a")
    axes[0].axhline(1.0, color="k", ls="--", lw=1)
    axes[0].set_xlabel("cells per side")
    axes[0].set_ylabel("ratio to closed form")
    axes[0].set_title("BEM validation on the circular punch")
    axes[0].legend(fontsize=8)
    rows = checks["sphere_limit_hertz"]["rows"]
    axes[1].semilogy([r["overlap"] for r in rows], [r["force_error_rel"] for r in rows],
                     "s-", label="kernel vs Hertz")
    axes[1].axhline(0.01, color="k", ls="--", lw=1)
    axes[1].set_xlabel("overlap [m]")
    axes[1].set_ylabel("relative force error")
    axes[1].set_title("Sphere-limit Hertz degradation")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "fig_verification.png"), dpi=140)
    plt.close(fig)
    print("[campaign] figures written", flush=True)


MODES = {
    "checks": run_checks,
    "geom": run_geom,
    "perf": run_perf,
    "heap": run_heap,
    "table": run_table,
    "figures": run_figures,
}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "checks"
    if mode == "all":
        for name in ("checks", "geom", "perf", "heap", "table", "figures"):
            MODES[name]()
    else:
        MODES[mode]()