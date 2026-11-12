#!/usr/bin/env python3
"""Real ship-hull-form contact verification of ``tensordem.hull_mesh_contact``.

Pays the two open parts of the gap-ledger row **GAP-DEM-G3**
(``docs/dem_calibration_gap_ledger.md``):

* "curved-hull validation on a real hull not re-audited" -- the earlier audit
  (``docs/dem3d_hull_mesh_contact_curved_audit.md``) closed the *cylinder*; a
  ship-hull form (doubly curved waterline + section, closing keel line, stem)
  was still open.  Here the mesh kernel is scored against the **exact Wigley
  hull distance** (``tensordem.hull_form_mesh``) -- a reference that does not
  use any kernel code path.
* "penalty stiffness/damping calibration ... not benchmarked" -- the linear
  spring-dashpot restitution ``e = exp(-beta*pi/sqrt(1-beta^2))``,
  ``beta = c/(2 sqrt(k m))``, is a closed-form property of the *implemented*
  force law, so it is measurable and falsifiable with a 1-DOF impact.

Everything printed here is emitted by this script; the report's tables are read
back from ``metrics.json`` (nothing transcribed by hand).

Usage
-----
    PYTHONPATH=src python scripts/m16794_wigley_hull_contact_audit.py
    PYTHONPATH=src python -m pytest tests/test_hull_form_mesh_contact.py -q
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tensordem.hull_form_mesh import (
    WigleyHull,
    wigley_directional_curvature,
    wigley_contact_reference,
    wigley_curvature,
    wigley_distance,
    wigley_inside,
    wigley_mesh,
)
from tensordem.hull_mesh_contact import _SpatialHash, mesh_contact_forces

HULL = WigleyHull()          # L = 4 m, B = 0.4 m, T = 0.25 m (L/B=10, B/T=1.6)
R_P = 0.05                   # ice-floe / particle radius [m]
K = 1.0e6                    # penalty stiffness [N/m]

# (name, u, v) -- u along the hull (aft -> fore), v keel -> waterline
ZONES = (
    ("midship_side", 0.50, 0.70),
    ("fore_shoulder", 0.85, 0.50),
    ("aft_shoulder", 0.18, 0.50),
    ("bilge", 0.70, 0.25),
    ("keel", 0.50, 0.12),
    ("stem", 0.97, 0.60),
)
PENS = (0.004, 0.008, 0.012, 0.020, 0.030, 0.040)
RESOLUTIONS = ((12, 5), (24, 10), (48, 20), (96, 40), (160, 64))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def zone_frame(u: float, v: float, hull: WigleyHull = HULL, h: float = 1e-6):
    """Surface point and outward unit normal at parametric ``(u, v)``."""
    foot = hull.surface(u, v)
    Xu = (hull.surface(u + h, v) - hull.surface(u - h, v)) / (2 * h)
    Xv = (hull.surface(u, v + h) - hull.surface(u, v - h)) / (2 * h)
    n = np.cross(Xu, Xv)
    n = n / np.linalg.norm(n)
    if n[1] < 0:                     # outward = away from the centreplane
        n = -n
    return foot, n


def sphere_at(u: float, v: float, penetration: float, radius: float = R_P,
              side: float = 1.0, hull: WigleyHull = HULL):
    """Place a sphere of ``radius`` so that it penetrates the hull by ``penetration``."""
    foot, n = zone_frame(u, v, hull)
    foot = foot * np.array([1.0, side, 1.0])
    n = n * np.array([1.0, side, 1.0])
    centre = foot + (radius - penetration) * n
    return centre, foot, n


def kernel_force(centres, verts, faces, radius=R_P, stiffness=K, damping=0.0,
                 velocities=None):
    """``mesh_contact_forces`` wrapper returning numpy forces + reaction."""
    c = torch.as_tensor(np.atleast_2d(centres), dtype=torch.float64)
    v = torch.zeros_like(c) if velocities is None else torch.as_tensor(
        np.atleast_2d(velocities), dtype=torch.float64)
    f, r = mesh_contact_forces(
        c, v, torch.as_tensor(verts, dtype=torch.float64),
        torch.as_tensor(faces, dtype=torch.int64),
        torch.zeros(3, dtype=torch.float64),
        particle_radius=(torch.as_tensor(np.asarray(radius), dtype=torch.float64)
                         if np.ndim(radius) else radius),
        stiffness=stiffness, damping=damping)
    return f.numpy(), r.numpy()


def ang_deg(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-300 or nb < 1e-300:
        return 0.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(c))


# ---------------------------------------------------------------------------
# 1. the reference itself
# ---------------------------------------------------------------------------

def reference_self_validation(n_points: int = 40, seed: int = 7) -> dict:
    """Check ``wigley_distance`` against two things that do not use it.

    1. A brute-force global grid gives an *upper bound* on the true minimum, so
       the solver must never return a larger distance than the grid minimum
       (``n_solver_worse_than_grid`` must be 0).
    2. A 401x401 ultra-fine grid rebuilt around the solver's own foot point is
       an independent local search: it must not find a smaller distance
       (``max_local_gain_m`` ~ 0), which is what rules out a failed polish /
       wrong basin.

    The residual gap to the *coarse* global grid is grid resolution, not solver
    error -- reported as ``max_grid_overshoot_m`` so the reader can see it.
    """
    rng = np.random.default_rng(seed)
    hull = HULL
    ug = np.linspace(0.0, 1.0, 1601)
    vg = np.linspace(0.0, 1.0, 801)
    U, V = np.meshgrid(ug, vg, indexing="ij")
    S = hull.surface(U, V)

    worse = 0
    max_overshoot = 0.0
    max_local_gain = 0.0
    min_dist = float("inf")
    for _ in range(n_points):
        p = np.array([rng.uniform(-0.52, 0.52) * hull.length,
                      rng.uniform(-0.35, 0.35),
                      rng.uniform(-0.30, 0.05)])
        d, _foot, (u0, v0) = wigley_distance(p, hull)
        min_dist = min(min_dist, d)
        ps = p.copy()
        ps[1] = abs(ps[1])
        grid = float(np.sqrt(((S - ps) ** 2).sum(-1).min()))
        if d > grid + 1e-12:
            worse += 1
            max_overshoot = max(max_overshoot, d - grid)
        uu = np.linspace(max(u0 - 0.01, 0.0), min(u0 + 0.01, 1.0), 401)
        vv = np.linspace(max(v0 - 0.01, 0.0), min(v0 + 0.01, 1.0), 401)
        U2, V2 = np.meshgrid(uu, vv, indexing="ij")
        S2 = hull.surface(U2, V2)
        local = float(np.sqrt(((S2 - ps) ** 2).sum(-1).min()))
        max_local_gain = max(max_local_gain, d - local)
    return {
        "n_points": n_points,
        "coarse_grid": [len(ug), len(vg)],
        "local_grid": [401, 401],
        "n_solver_worse_than_grid": worse,
        "max_overshoot_m": max_overshoot,
        "max_local_gain_m": max_local_gain,
        "min_distance_sampled_m": min_dist,
        "criterion": "solver <= global grid minimum (upper bound) and a 401x401 "
                     "local grid around the solver foot finds no smaller distance",
    }


# ---------------------------------------------------------------------------
# 2. force error vs penetration per hull zone (fine mesh)
# ---------------------------------------------------------------------------

def fine_mesh_zone_study(n_u: int = 160, n_v: int = 64) -> dict:
    verts, faces = wigley_mesh(n_u, n_v, HULL)
    rows = []
    worst = {"rel_err": 0.0}
    for name, u, v in ZONES:
        for side in (1.0, -1.0):
            for pen in PENS:
                centre, foot, n_true = sphere_at(u, v, pen, side=side)
                f_mesh, _ = kernel_force(centre, verts, faces)
                f_ref, d_ref = wigley_contact_reference(centre, R_P, stiffness=K, hull=HULL)
                fr = f_ref[0]
                rel = abs(np.linalg.norm(f_mesh[0]) - np.linalg.norm(fr)) / max(
                    np.linalg.norm(fr), 1e-30)
                ang = ang_deg(f_mesh[0], fr)
                rows.append({
                    "zone": name, "side": side, "penetration_m": pen,
                    "d_true_m": float(d_ref[0]),
                    "f_mesh_n": float(np.linalg.norm(f_mesh[0])),
                    "f_ref_n": float(np.linalg.norm(fr)),
                    "rel_err": rel, "dir_err_deg": ang,
                })
                if rel > worst["rel_err"]:
                    worst = rows[-1]
    return {
        "mesh": {"n_u": n_u, "n_v": n_v, "n_triangles": int(faces.shape[0])},
        "n_points": len(rows),
        "max_rel_err": worst["rel_err"],
        "max_rel_err_row": worst,
        "max_dir_err_deg": max(r["dir_err_deg"] for r in rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 3. discretisation convergence vs the local curvature rule
# ---------------------------------------------------------------------------

def facet_rule_study() -> dict:
    """Error vs mesh size, compared with the prediction sagitta/(R - pen)."""
    out = []
    for n_u, n_v in RESOLUTIONS:
        verts, faces = wigley_mesh(n_u, n_v, HULL)
        rec = {"n_u": n_u, "n_v": n_v, "n_triangles": int(faces.shape[0]),
               "max_rel_err": 0.0, "max_pred_err": 0.0, "worst": None}
        for name, u, v in ZONES:
            ku, kv = wigley_directional_curvature(u, v, HULL)
            du = 1.0 / n_u
            dv = 1.0 / n_v
            p0 = HULL.surface(u, v)
            cu = float(np.linalg.norm(HULL.surface(min(u + du, 1.0), v) - p0))
            cv = float(np.linalg.norm(HULL.surface(u, min(v + dv, 1.0)) - p0))
            chord = max(cu, cv)
            sagitta = (ku * cu * cu + kv * cv * cv) / 8.0
            pen = 0.010
            pred = sagitta / pen
            for side in (1.0, -1.0):
                centre, _foot, _n = sphere_at(u, v, pen, side=side)
                f_mesh, _ = kernel_force(centre, verts, faces)
                f_ref, _d = wigley_contact_reference(centre, R_P, stiffness=K, hull=HULL)
                fr = f_ref[0]
                rel = abs(np.linalg.norm(f_mesh[0]) - np.linalg.norm(fr)) / max(
                    np.linalg.norm(fr), 1e-30)
                if rel > rec["max_rel_err"]:
                    rec["max_rel_err"] = rel
                    rec["worst"] = {"zone": name, "side": side, "chord_m": chord,
                                    "kappa_u": ku, "kappa_v": kv,
                                    "chord_u_m": cu, "chord_v_m": cv,
                                    "sagitta_m": sagitta, "pred_rel_err": pred,
                                    "ratio_measured_over_pred": rel / pred if pred > 0 else None}
                rec["max_pred_err"] = max(rec["max_pred_err"], pred)
        out.append(rec)
    return {"penetration_m": 0.010, "resolutions": out}


def overlap_sensitivity(n_u: int = 48, n_v: int = 20) -> dict:
    """Error vs penetration at a *fixed* mesh: the predictor is sagitta/overlap.

    Overlap is ``R - pen``; as the penetration grows towards the particle radius
    the analytic force shrinks while the facet offset stays fixed, so the
    relative error must grow even though the mesh did not change.
    """
    verts, faces = wigley_mesh(n_u, n_v, HULL)
    rows = []
    for pen in (0.005, 0.010, 0.020, 0.030, 0.038, 0.040):
        worst = 0.0
        pred = 0.0
        worst_zone = None
        for name, u, v in ZONES:
            ku, kv = wigley_directional_curvature(u, v, HULL)
            cu = float(np.linalg.norm(HULL.surface(min(u + 1.0 / n_u, 1.0), v)
                                      - HULL.surface(u, v)))
            cv = float(np.linalg.norm(HULL.surface(u, min(v + 1.0 / n_v, 1.0))
                                      - HULL.surface(u, v)))
            sagitta = (ku * cu * cu + kv * cv * cv) / 8.0
            p_est = sagitta / pen
            for side in (1.0, -1.0):
                centre, _f, _n = sphere_at(u, v, pen, side=side)
                f_mesh, _ = kernel_force(centre, verts, faces)
                f_ref, _d = wigley_contact_reference(centre, R_P, stiffness=K, hull=HULL)
                fr = f_ref[0]
                rel = abs(np.linalg.norm(f_mesh[0]) - np.linalg.norm(fr)) / max(
                    np.linalg.norm(fr), 1e-30)
                if rel > worst:
                    worst, worst_zone = rel, name
            pred = max(pred, p_est)
        rows.append({"penetration_m": pen, "stand_off_m": R_P - pen,
                     "max_rel_err": worst, "pred_rel_err": pred,
                     "worst_zone": worst_zone})
    return {"mesh": {"n_u": n_u, "n_v": n_v}, "rows": rows}


# ---------------------------------------------------------------------------
# 4. global load from an ice-floe-like cloud
# ---------------------------------------------------------------------------

def global_load_study(n_spheres: int = 240, seed: int = 11,
                      n_u: int = 160, n_v: int = 64) -> dict:
    rng = np.random.default_rng(seed)
    verts, faces = wigley_mesh(n_u, n_v, HULL)
    centres, radii = [], []
    for _ in range(n_spheres):
        u = float(rng.uniform(0.02, 0.98))
        v = float(rng.uniform(0.05, 0.98))
        pen = float(rng.uniform(0.002, 0.020))
        r = float(rng.uniform(0.02, 0.06))
        side = 1.0 if rng.random() < 0.5 else -1.0
        c, _f, _n = sphere_at(u, v, pen, radius=r, side=side)
        centres.append(c)
        radii.append(r)
    centres = np.asarray(centres)
    f_mesh, reaction = kernel_force(centres, verts, faces, radius=np.asarray(radii))
    f_ana = np.zeros_like(centres)
    n_contact_ana = 0
    for i, c in enumerate(centres):
        fr, d = wigley_contact_reference(c, radii[i], stiffness=K, hull=HULL)
        f_ana[i] = fr[0] if fr.ndim == 2 else fr
        if d[0] < radii[i]:
            n_contact_ana += 1
    rel_mag = abs(np.linalg.norm(f_mesh) - np.linalg.norm(f_ana)) / np.linalg.norm(f_ana)
    comp = np.abs(f_mesh.sum(0) - f_ana.sum(0)) / np.maximum(np.abs(f_ana.sum(0)), 1e-30)
    per = []
    for i in range(len(centres)):
        if np.linalg.norm(f_ana[i]) < 1e-12:
            continue
        per.append(abs(np.linalg.norm(f_mesh[i]) - np.linalg.norm(f_ana[i]))
                   / np.linalg.norm(f_ana[i]))
    return {
        "n_spheres": n_spheres, "mesh": {"n_u": n_u, "n_v": n_v},
        "n_contacts_kernel": int((np.linalg.norm(f_mesh, axis=1) > 0).sum()),
        "n_contacts_analytic": n_contact_ana,
        "global_force_mesh_n": float(np.linalg.norm(f_mesh.sum(0))),
        "global_force_analytic_n": float(np.linalg.norm(f_ana.sum(0))),
        "global_rel_err": float(rel_mag),
        "component_rel_err": comp.tolist(),
        "per_sphere_max_rel_err": float(max(per)) if per else 0.0,
        "per_sphere_median_rel_err": float(np.median(per)) if per else 0.0,
        "reaction_residual_n": float(np.linalg.norm(reaction + f_mesh.sum(0))),
        "centres": centres.tolist(), "radii": np.asarray(radii).tolist(),
        "f_mesh": f_mesh.tolist(), "f_analytic": f_ana.tolist(),
    }


# ---------------------------------------------------------------------------
# 5. broad phase on the hull mesh
# ---------------------------------------------------------------------------

def broad_phase_completeness(n_queries: int = 300, seed: int = 23,
                             n_u: int = 96, n_v: int = 40) -> dict:
    verts, faces = wigley_mesh(n_u, n_v, HULL)
    rng = np.random.default_rng(seed)
    centres = np.stack([
        rng.uniform(-0.55, 0.55, n_queries) * HULL.length,
        rng.uniform(-0.30, 0.30, n_queries),
        rng.uniform(-0.30, 0.06, n_queries)], axis=1)
    Rq = 0.05
    sh = _SpatialHash(torch.as_tensor(verts, dtype=torch.float64),
                      torch.as_tensor(faces, dtype=torch.int64), max(4.0 * Rq, 1e-6))
    cands = sh.query(torch.as_tensor(centres, dtype=torch.float64), 2.0 * Rq)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    tri_min = np.minimum(np.minimum(v0, v1), v2)
    tri_max = np.maximum(np.maximum(v0, v1), v2)
    missing = 0
    for i in range(n_queries):
        lo = centres[i] - 2.0 * Rq
        hi = centres[i] + 2.0 * Rq
        need = np.nonzero(((tri_max >= lo) & (tri_min <= hi)).all(axis=1))[0]
        have = set(cands[i])
        if any(int(fi) not in have for fi in need):
            missing += 1
    return {"n_queries": n_queries, "n_missing_superset_violations": missing,
            "n_faces": int(faces.shape[0])}


def action_reaction_check(n_u: int = 160, n_v: int = 64) -> dict:
    verts, faces = wigley_mesh(n_u, n_v, HULL)
    c, _f, _n = sphere_at(0.5, 0.7, 0.010)
    f, r = kernel_force(c, verts, faces)
    return {"residual_n": float(np.linalg.norm(r + f.sum(0))),
            "force_n": float(np.linalg.norm(f[0]))}


# ---------------------------------------------------------------------------
# 6. restitution / damping calibration (the linear spring-dashpot law)
# ---------------------------------------------------------------------------

def restitution_study(mass: float = 1.0, v0: float = 0.02,
                      betas=(0.0, 0.1, 0.2, 0.3), steps_per_tc=(200, 400, 800),
                      n_u: int = 160, n_v: int = 64) -> dict:
    """1-DOF impact at midship vs ``e = exp(-beta*pi/sqrt(1-beta^2))``.

    The sphere starts clear of the surface and moves along the local outward
    normal; the kernel's own ``k`` and ``c`` are the only physics.  ``c`` is
    set from the requested ``beta`` so the reference value does not depend on
    the integrator.
    """
    verts, faces = wigley_mesh(n_u, n_v, HULL)
    centre0, foot, n = sphere_at(0.5, 0.7, 0.0)         # touching point
    n = n / np.linalg.norm(n)
    rows = []
    for beta in betas:
        c_damp = 2.0 * beta * math.sqrt(K * mass)
        e_ref = math.exp(-beta * math.pi / math.sqrt(max(1.0 - beta * beta, 1e-12))) \
            if beta < 1.0 else 0.0
        tc = math.pi * math.sqrt(mass / K)
        for spd in steps_per_tc:
            dt = tc / spd
            pos = centre0 + (R_P + 0.02) * n       # 2 cm stand-off
            vel = -v0 * n
            v_in = None
            force_peak = 0.0
            for _ in range(int(6 * tc / dt)):
                f, _r = kernel_force(pos[None, :], verts, faces, damping=c_damp,
                                     velocities=vel[None, :])
                if v_in is None and np.linalg.norm(f[0]) > 0:
                    v_in = float(np.dot(vel, n))
                force_peak = max(force_peak, float(np.linalg.norm(f[0])))
                acc = f[0] / mass
                vel = vel + acc * dt                  # semi-implicit Euler
                pos = pos + vel * dt
                if v_in is not None and np.linalg.norm(f[0]) == 0.0 and np.dot(vel, n) > 0:
                    break
            v_out = float(np.dot(vel, n))
            e_meas = abs(v_out) / abs(v_in) if v_in else float("nan")
            rows.append({"beta": beta, "c_damping": c_damp, "dt": dt,
                         "steps_per_tc": spd, "e_reference": e_ref,
                         "e_measured": e_meas,
                         "rel_err": abs(e_meas - e_ref) / e_ref if e_ref > 0 else abs(e_meas),
                         "force_peak_n": force_peak})
    return {"mass_kg": mass, "v0_m_s": v0, "stiffness": K,
            "t_contact_analytic_s": math.pi * math.sqrt(mass / K), "rows": rows}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/m16794_wigley_hull_audit/metrics.json")
    ap.add_argument("--figures", default="artifacts/m16794_wigley_hull_audit")
    ap.add_argument("--skip-slow", action="store_true")
    args = ap.parse_args()

    metrics = {
        "hull": {"length_m": HULL.length, "beam_m": HULL.beam, "draft_m": HULL.draft,
                 "form": "Wigley (parabolic waterlines + sections)",
                 "L_over_B": HULL.length / HULL.beam, "B_over_T": HULL.beam / HULL.draft},
        "particle_radius_m": R_P, "stiffness_n_m": K,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        out_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print("[1/6] reference self-validation ...", flush=True)
    metrics["reference_self_validation"] = reference_self_validation()
    rsv = metrics["reference_self_validation"]
    print(f"      solver worse than grid: {rsv['n_solver_worse_than_grid']}, "
          f"local-grid gain {rsv['max_local_gain_m']:.2e} m")

    dump(); print("[2/6] fine-mesh zone study ...")
    metrics["fine_mesh_zone_study"] = fine_mesh_zone_study()
    print("      max rel err =", metrics["fine_mesh_zone_study"]["max_rel_err"],
          " max dir err =", metrics["fine_mesh_zone_study"]["max_dir_err_deg"])

    dump(); print("[3/6] facet-rule convergence ...")
    metrics["facet_rule_study"] = facet_rule_study()
    for r in metrics["facet_rule_study"]["resolutions"]:
        print(f"      {r['n_u']:4d}x{r['n_v']:<3d} max_rel_err={r['max_rel_err']:.4f} "
              f"pred={r['max_pred_err']:.4f}")

    dump(); print("[3b/6] overlap sensitivity ...")
    metrics["overlap_sensitivity"] = overlap_sensitivity()
    for r in metrics["overlap_sensitivity"]["rows"]:
        print(f"      pen={r['penetration_m']*1e3:5.1f} mm stand_off={r['stand_off_m']*1e3:5.1f} mm "
              f"max_rel_err={r['max_rel_err']:.4f} pred={r['pred_rel_err']:.4f}")

    dump(); print("[4/6] global load (ice-floe cloud) ...")
    metrics["global_load_study"] = global_load_study()
    g = metrics["global_load_study"]
    print("      global rel err =", g["global_rel_err"],
          " contacts", g["n_contacts_kernel"], "/", g["n_contacts_analytic"])

    dump(); print("[5/6] broad phase + action/reaction ...")
    metrics["broad_phase"] = broad_phase_completeness()
    metrics["action_reaction"] = action_reaction_check()
    print("      missing =", metrics["broad_phase"]["n_missing_superset_violations"],
          " reaction residual =", metrics["action_reaction"]["residual_n"])

    dump(); print("[6/6] restitution / damping calibration ...")
    metrics["restitution_study"] = restitution_study()
    for r in metrics["restitution_study"]["rows"]:
        print(f"      beta={r['beta']:.2f} steps/tc={r['steps_per_tc']:4d} "
              f"e_meas={r['e_measured']:.4f} e_ref={r['e_reference']:.4f} "
              f"rel={r['rel_err']*100:.2f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"metrics -> {out}")

    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = make_figures(metrics, fig_dir)
    print(f"figures -> {', '.join(figs)}")


def make_figures(metrics: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    # (a) force vs penetration per zone, mesh vs analytic
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rows = metrics["fine_mesh_zone_study"]["rows"]
    for name, _u, _v in ZONES:
        zr = [r for r in rows if r["zone"] == name and r["side"] == 1.0]
        pens = np.array([r["penetration_m"] for r in zr]) * 1e3
        ax.plot(pens, [r["f_ref_n"] for r in zr], "-", lw=1.4, label=f"{name} analytic")
        ax.plot(pens, [r["f_mesh_n"] for r in zr], "o", ms=3.2, mfc="none")
    ax.set_xlabel("penetration [mm]")
    ax.set_ylabel("contact force [N]")
    ax.set_title(f"Wigley hull, mesh {metrics['fine_mesh_zone_study']['mesh']['n_u']}x"
                 f"{metrics['fine_mesh_zone_study']['mesh']['n_v']} "
                 f"(open circles = mesh kernel, lines = analytic penalty)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    p = out_dir / "fig_force_vs_penetration.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    # (b) convergence vs facet chord (log-log) with the sagitta/(R-pen) prediction
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    res = metrics["facet_rule_study"]["resolutions"]
    chords = [r["worst"]["chord_m"] for r in res if r["worst"]]
    meas = [r["max_rel_err"] for r in res if r["worst"]]
    pred = [r["worst"]["pred_rel_err"] for r in res if r["worst"]]
    ax.loglog(np.array(chords) * 1e3, meas, "o-", label="measured max rel. error")
    ax.loglog(np.array(chords) * 1e3, pred, "s--", label=r"predicted $\kappa_{max}h^2/8\,/\,(R-p)$")
    ax.axhline(0.03, color="k", ls=":", lw=1, label="3 % acceptance")
    ax.set_xlabel("local facet chord [mm]")
    ax.set_ylabel("relative force error [-]")
    ax.set_title("Digitisation convergence on the ship-hull form (penetration 10 mm)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    p = out_dir / "fig_resolution_convergence.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    # (c) parity plot: per-sphere force, mesh vs analytic
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    g = metrics["global_load_study"]
    fm = np.linalg.norm(np.asarray(g["f_mesh"]), axis=1)
    fa = np.linalg.norm(np.asarray(g["f_analytic"]), axis=1)
    ax.loglog(fa, fm, "o", ms=3, mfc="none")
    lim = [max(fa.min() * 0.8, 1e-3), fa.max() * 1.2]
    ax.plot(lim, lim, "k-", lw=1, label="exact")
    ax.set_xlabel("analytic penalty force [N]")
    ax.set_ylabel("mesh kernel force [N]")
    ax.set_title(f"Ice-floe cloud on the hull ({g['n_spheres']} spheres)\n"
                 f"global force error {g['global_rel_err']*100:.2f} %")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    p = out_dir / "fig_global_parity.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    # (d) restitution calibration
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    rr = metrics["restitution_study"]["rows"]
    betas = sorted({r["beta"] for r in rr})
    for spd in sorted({r["steps_per_tc"] for r in rr}):
        sel = [r for r in rr if r["steps_per_tc"] == spd]
        ax.plot([r["beta"] for r in sel], [r["e_measured"] for r in sel], "o-",
                ms=4, label=f"measured, dt = $t_c$/{spd}")
    ax.plot(betas, [math.exp(-b * math.pi / math.sqrt(1 - b * b)) for b in betas],
            "k--", label=r"$\exp(-\beta\pi/\sqrt{1-\beta^2})$")
    ax.set_xlabel(r"damping ratio $\beta = c/(2\sqrt{km})$")
    ax.set_ylabel("restitution coefficient e")
    ax.set_title("Penalty spring-dashpot restitution on the hull surface")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    p = out_dir / "fig_restitution.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)
    return made


if __name__ == "__main__":
    main()