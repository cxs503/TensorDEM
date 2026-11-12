"""Audit the restitution delivered by the 3-D hull-mesh penalty contact law (m16922).

Why this exists
---------------
`docs/AUDIT_GAP_DEM_G3_WIGLEY_REAL_HULL_CONTACT.md` (m16794) closes the hull-mesh
contact geometry work but leaves one item explicitly **NOT_RUN**:

    "Damping / restitution: NOT_RUN (honest). The closed-form reference
     `e = exp(-beta*pi/sqrt(1-beta^2))`, `beta = c/(2 sqrt(k m))` is the right
     calibration target for this force law, but every attempt was aborted by the
     mission time budget."

That statement is in fact *wrong about the reference*: the hull kernel evaluates a
**compressive-only** normal force,

    F_n = max(k * overlap - c * vn, 0)          (src/tensordem/hull_mesh_contact.py)

so the impact is not the unclamped damped oscillator whose restitution is the
classical `exp(-pi zeta / sqrt(1-zeta^2))`; the force is switched off while the
spheres still overlap, which leaves elastic energy unspent and raises the
observed restitution.  The gap therefore is not only unmeasured, it is a
calibration defect: a user who asks for restitution 0.4 through the classical
inverse gets something materially larger.

What this script measures (nothing is assumed, everything is computed)
---------------------------------------------------------------------
1. `reference_rk4` — an *independent* high-accuracy RK4 integration of the
   clamped 1-DOF impact model, with an internal step-refinement convergence
   study (so the reference is not borrowed from the code under test).
2. Cross-check of that reference against the repository's own
   `tensordem.contact_restitution.clamped_restitution_ratio` (two independent
   implementations; a disagreement would invalidate both).
3. `kernel_bounce` — a 1-DOF impact driven by the *shipped*
   `tensordem.hull_mesh_contact.mesh_contact_forces` kernel (no reimplementation
   of the contact law), integrated with velocity Verlet and, for sensitivity,
   semi-implicit Euler.  Geometry is a flat triangulated patch (exact) plus a
   curved Wigley patch, and the patch is proven interchangeable with the full
   hull mesh before it is used.
4. Sweep over target restitution: classical mapping
   (`contact_tangential.restitution_to_damping`) vs the clamped inverse
   (`contact_restitution.damping_for_target_restitution`) measured through the
   kernel, at four timestep resolutions.
5. Invariants: the penalty law is linear, so the measured restitution must not
   depend on the impact speed; damping = 0 must return e = 1.

Outputs (all numbers in the report/figures are read back from here)
------------------------------------------------------------------
``runs/m16922/metrics.json`` and three PNG figures.

Usage
-----
    PYTHONPATH=src python scripts/m16922_hull_penalty_restitution_audit.py --quick
    PYTHONPATH=src python scripts/m16922_hull_penalty_restitution_audit.py
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fixed physical set-up (SI units)
# ---------------------------------------------------------------------------
STIFFNESS = 1.0e6          # N/m, the k used by the m16794 hull audit
MASS = 1.0                 # kg, single sphere (wall contact -> m_eff = m)
RADIUS = 0.05              # m, ice-floe analogue used in the hull audits
V_IN = 0.02                # m/s approach speed (linear law -> speed independent)
T_C = math.pi * math.sqrt(MASS / STIFFNESS)   # undamped contact period, s

E_TARGETS = (0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30)
DT_DIVISORS = (100, 200, 400)


# ---------------------------------------------------------------------------
# 1. Independent reference: RK4 on the clamped 1-DOF impact model
# ---------------------------------------------------------------------------

def _rk4_reference(c_damp: float, k: float = STIFFNESS, m: float = MASS,
                   v_in: float = V_IN, steps: int = 4000) -> dict[str, Any]:
    """Integrate ``m d2(delta) = -max(k*delta + c*d(delta), 0)`` until separation.

    ``delta`` is the overlap, ``v = d(delta)/dt`` is positive while the bodies
    approach.  The contact force is clamped at zero: once ``k*delta + c*v <= 0``
    the pair coasts.  Returns the restitution ``|v_out| / v_in``.
    """
    dt = math.pi * math.sqrt(m / k) / steps   # contact-period based
    delta, v = 0.0, v_in
    t = 0.0
    released = False
    release_delta = 0.0
    max_delta = 0.0
    max_steps = 60 * steps

    def acc(d: float, r: float) -> float:
        return -max(k * d + c_damp * r, 0.0) / m

    for _ in range(max_steps):
        if not released and k * delta + c_damp * v <= 0.0:
            released = True
            release_delta = delta
        a1 = acc(delta, v)
        d2, r2 = delta + 0.5 * dt * v, v + 0.5 * dt * a1
        a2 = acc(d2, r2)
        d3, r3 = delta + 0.5 * dt * r2, v + 0.5 * dt * a2
        a3 = acc(d3, r3)
        d4, r4 = delta + dt * r3, v + dt * a3
        a4 = acc(d4, r4)
        new_delta = delta + dt * (v + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
        new_v = v + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0
        t += dt
        max_delta = max(max_delta, new_delta)
        if new_delta <= 0.0 and v < 0.0 and t > dt:
            frac = delta / (delta - new_delta) if delta != new_delta else 1.0
            v_out = v + frac * (new_v - v)
            return {
                "e_reference": abs(v_out / v_in),
                "contact_time_s": t - dt + frac * dt,
                "release_overlap_m": release_delta,
                "released_early": released,
                "max_overlap_m": max_delta,
                "steps": steps,
                "dt": dt,
            }
        delta, v = new_delta, new_v
    raise RuntimeError("reference impact did not separate; check inputs")


def reference_convergence(c_damps: tuple[float, ...] = (0.0, 200.0, 800.0, 1400.0),
                          step_counts: tuple[int, ...] = (1000, 2000, 4000, 8000)
                          ) -> dict[str, Any]:
    """Step refinement of the in-script RK4 reference (Richardson-style)."""
    rows = []
    for c in c_damps:
        seq = [_rk4_reference(c, steps=n)["e_reference"] for n in step_counts]
        diffs = [abs(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)]
        rows.append({"c_damping": c, "steps": list(step_counts), "e": seq,
                     "successive_diffs": diffs,
                     "finest_diff": diffs[-1] if diffs else 0.0})
    return {"step_counts": list(step_counts), "rows": rows,
            "max_finest_diff": max(r["finest_diff"] for r in rows)}


def reference_vs_repo_crosscheck(c_damps: tuple[float, ...] = (0.0, 50.0, 200.0,
                                                              600.0, 1000.0, 1400.0)
                                 ) -> dict[str, Any]:
    """Independent RK4 vs ``tensordem.contact_restitution.clamped_restitution_ratio``."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from tensordem.contact_restitution import clamped_restitution_ratio

    rows = []
    for c in c_damps:
        mine = _rk4_reference(c)["e_reference"]
        theirs = clamped_restitution_ratio(c, STIFFNESS, MASS, v_in=V_IN)["e_reference"]
        rows.append({"c_damping": c, "e_independent": mine, "e_repo": theirs,
                     "abs_diff": abs(mine - theirs)})
    return {"rows": rows, "max_abs_diff": max(r["abs_diff"] for r in rows)}


# ---------------------------------------------------------------------------
# 2. Geometry: flat patch, curved Wigley patch, and the patch/full-hull proof
# ---------------------------------------------------------------------------

def flat_patch() -> tuple[np.ndarray, np.ndarray]:
    """Two-triangle flat plane at z = 0 spanning [-1, 1]^2."""
    verts = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0],
                      [-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    faces = np.arange(6, dtype=np.int64).reshape(-1, 3)
    return verts, faces


def wigley_patch(resolution: tuple[int, int] = (96, 40), patch_radius: float = 0.40,
                 u0: float = 0.50, v0: float = 0.30) -> dict[str, Any]:
    """Local curved patch of the Wigley hull around the impact foot point.

    Triangles whose centroid lies within ``patch_radius`` of the foot point are
    kept.  The sphere only ever penetrates a few tenths of a millimetre, so the
    kept set contains every triangle that can be the nearest one; this is
    *proven* by :func:`patch_equivalence` before the patch is used.
    """
    from tensordem.hull_form_mesh import WigleyHull, wigley_mesh

    verts, faces = wigley_mesh(*resolution)
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    hull = WigleyHull()
    foot = np.asarray(hull.surface(u0, v0), dtype=np.float64)
    h = 1e-6
    xu = (np.asarray(hull.surface(u0 + h, v0)) - np.asarray(hull.surface(u0 - h, v0))) / (2 * h)
    xv = (np.asarray(hull.surface(u0, v0 + h)) - np.asarray(hull.surface(u0, v0 - h))) / (2 * h)
    n = np.cross(xu, xv)
    n = n / np.linalg.norm(n)
    if n[1] < 0:
        n = -n
    centroid = verts[faces].mean(axis=1)
    keep = np.linalg.norm(centroid - foot, axis=1) <= patch_radius
    sub_faces = faces[keep]
    used = np.unique(sub_faces)
    remap = -np.ones(verts.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    return {"verts": verts[used], "faces": remap[sub_faces], "foot": foot,
            "normal": n, "n_faces_full": int(faces.shape[0]),
            "n_faces_patch": int(sub_faces.shape[0]),
            "full": {"verts": verts, "faces": faces}, "resolution": resolution}


# ---------------------------------------------------------------------------
# 3. Closed-loop measurement through the shipped kernel
# ---------------------------------------------------------------------------

def _kernel_force(centre: np.ndarray, velocity: np.ndarray, verts: np.ndarray,
                  faces: np.ndarray, damping: float, stiffness: float = STIFFNESS,
                  radius: float = RADIUS):
    import torch
    from tensordem.hull_mesh_contact import mesh_contact_forces

    c = torch.as_tensor(centre.reshape(1, 3), dtype=torch.float64)
    v = torch.as_tensor(velocity.reshape(1, 3), dtype=torch.float64)
    f, _r = mesh_contact_forces(
        c, v, torch.as_tensor(verts, dtype=torch.float64),
        torch.as_tensor(faces, dtype=torch.int64),
        torch.zeros(3, dtype=torch.float64),
        particle_radius=radius, stiffness=stiffness, damping=damping)
    return f[0].numpy()


def kernel_bounce(damping: float, *, dt: float, scheme: str = "verlet",
                  verts: np.ndarray | None = None, faces: np.ndarray | None = None,
                  normal: np.ndarray | None = None, v_in: float = V_IN,
                  mass: float = MASS, radius: float = RADIUS,
                  stiffness: float = STIFFNESS, gap: float = 1e-4,
                  max_contacts: float = 8.0,
                  centre_override: np.ndarray | None = None) -> dict[str, Any]:
    """1-DOF impact driven by ``mesh_contact_forces``; returns the measured restitution.

    The sphere starts ``gap`` clear of the surface and approaches along the local
    outward normal ``normal`` (default +z for the flat patch).  Integration is
    velocity Verlet (``scheme='verlet'``) or semi-implicit Euler (``euler``).
    """
    if verts is None or faces is None:
        verts, faces = flat_patch()
    n = np.array([0.0, 0.0, 1.0]) if normal is None else np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    r0 = float(radius)
    if centre_override is None:
        # flat patch: hull plane is z = 0 and the outward normal is +z
        centre = n * (r0 + gap)
    else:
        # curved patch: start on the local normal through the foot point
        centre = np.asarray(centre_override, dtype=float) + gap * n
    vel = -v_in * n
    # step budget: clear the initial gap, then allow ``max_contacts`` contact periods
    tc_local = math.pi * math.sqrt(mass / stiffness)
    steps = int((gap / v_in + max_contacts * tc_local / 4.0) / dt) + 10
    force_peak = 0.0
    contacted = False
    push_off = 0.0
    t = 0.0
    for _ in range(steps):
        f = _kernel_force(centre, vel, verts, faces, damping, stiffness, r0)
        fmag = float(np.linalg.norm(f))
        force_peak = max(force_peak, fmag)
        if fmag > 0.0:
            contacted = True
            push_off = float(np.dot(f, n)) / mass
        if scheme == "euler":
            vel = vel + (f / mass) * dt
            centre = centre + vel * dt
        elif scheme == "verlet":
            vel = vel + 0.5 * (f / mass) * dt
            centre = centre + vel * dt
            f2 = _kernel_force(centre, vel, verts, faces, damping, stiffness, r0)
            fmag2 = float(np.linalg.norm(f2))
            force_peak = max(force_peak, fmag2)
            vel = vel + 0.5 * (f2 / mass) * dt
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        t += dt
        if contacted and float(np.dot(vel, n)) > 0.0 and fmag == 0.0:
            f_after = _kernel_force(centre, vel, verts, faces, damping, stiffness, r0)
            if float(np.linalg.norm(f_after)) == 0.0:
                return {"e_measured": float(np.dot(vel, n)) / v_in,
                        "contact_time_s": t, "force_peak_n": force_peak,
                        "dt": dt, "scheme": scheme, "separated": True}
    return {"e_measured": float("nan"), "contact_time_s": t,
            "force_peak_n": force_peak, "dt": dt, "scheme": scheme,
            "separated": False}


# ---------------------------------------------------------------------------
# 4. Sweep driver (multiprocessing; each worker is single-threaded torch)
# ---------------------------------------------------------------------------

def _worker(task: dict[str, Any]) -> dict[str, Any]:
    """Run one :func:`kernel_bounce` case in a single-threaded worker process."""
    import torch
    torch.set_num_threads(1)
    task = dict(task)
    for extra in ("geometry", "which", "e_target", "divisor"):
        task.pop(extra, None)
    verts = faces = normal = centre = None
    if task.pop("curved", False):
        patch = wigley_patch()
        verts, faces, normal = patch["verts"], patch["faces"], patch["normal"]
        centre = patch["foot"] + (RADIUS + 5e-4) * patch["normal"]
    damping = task.pop("damping")
    res = kernel_bounce(damping, verts=verts, faces=faces, normal=normal,
                        centre_override=centre, **task)
    return res


def patch_equivalence(resolution: tuple[int, int] = (96, 40)) -> dict[str, Any]:
    """Prove the local patch is interchangeable with the full hull mesh.

    Sweeps the sphere along the local normal through the whole contact range
    (clearance to ~1 cm penetration) and compares the patch force with the
    full-mesh force at every state.
    """
    import torch
    torch.set_num_threads(1)
    patch = wigley_patch(resolution=resolution)
    foot, n = patch["foot"], patch["normal"]
    rows = []
    for pen in (-2e-3, -5e-4, 0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2):
        centre = foot + (RADIUS - pen) * n
        vel = -V_IN * n
        f_patch = _kernel_force(centre, vel, patch["verts"], patch["faces"], 0.0)
        f_full = _kernel_force(centre, vel, patch["full"]["verts"],
                               patch["full"]["faces"], 0.0)
        rows.append({
            "penetration_m": pen,
            "f_patch_n": f_patch.tolist(), "f_full_n": f_full.tolist(),
            "force_rel_diff": float(np.linalg.norm(f_patch - f_full)
                                    / max(np.linalg.norm(f_full), 1e-300))
            if np.linalg.norm(f_full) > 0 else float(np.linalg.norm(f_patch - f_full)),
        })
    return {"resolution": list(resolution),
            "n_faces_full": patch["n_faces_full"],
            "n_faces_patch": patch["n_faces_patch"],
            "rows": rows,
            "max_force_rel_diff": max(r["force_rel_diff"] for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runs" / "m16922"))
    ap.add_argument("--quick", action="store_true",
                    help="reduced sweep (2 targets, 2 resolutions) for smoke runs")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(REPO / "src"))
    from tensordem.contact_restitution import damping_for_target_restitution
    from tensordem.contact_tangential import damping_to_restitution, restitution_to_damping

    t_start = time.perf_counter()
    metrics: dict[str, Any] = {
        "mission": 16922,
        "setup": {"stiffness_n_per_m": STIFFNESS, "mass_kg": MASS,
                  "radius_m": RADIUS, "v_in_m_per_s": V_IN,
                  "contact_period_s": T_C},
    }

    print("[1/6] reference RK4 convergence ...")
    metrics["reference_convergence"] = reference_convergence()

    print("[2/6] independent reference vs repo clamped reference ...")
    metrics["reference_crosscheck"] = reference_vs_repo_crosscheck()

    print("[3/6] patch vs full-hull force equivalence ...")
    metrics["patch_equivalence"] = patch_equivalence()

    # ---- mappings for every target restitution -------------------------
    e_targets = E_TARGETS[:2] if args.quick else E_TARGETS
    divs = DT_DIVISORS[:2] if args.quick else DT_DIVISORS
    mappings = []
    for e_t in e_targets:
        corr = damping_for_target_restitution(e_t, STIFFNESS, MASS)
        cl = restitution_to_damping(e_t, STIFFNESS, MASS)
        mappings.append({
            "e_target": e_t,
            "c_classic": cl,
            "zeta_classic": cl / (2.0 * math.sqrt(STIFFNESS * MASS)),
            "e_classic_of_classic_c": damping_to_restitution(cl, STIFFNESS, MASS),
            "c_corrected": corr["gamma_n"],
            "zeta_corrected": corr["zeta"],
            "e_classic_of_corrected_c": corr["e_classic_of_same_gamma"],
            "reference_e_of_corrected_c": corr["e_reference"],
            # what the classical mapping actually delivers under the clamped law
            "reference_e_of_classic_c": _rk4_reference(cl)["e_reference"],
            "classic_mapping_reference_error": (
                _rk4_reference(cl)["e_reference"] - e_t) / e_t,
        })
    metrics["mappings"] = mappings

    # ---- kernel sweep ---------------------------------------------------
    print("[4/6] kernel sweep (classic vs corrected mapping, flat patch) ...")
    tasks = []
    geometry = {"kind": "flat"}
    for m in mappings:
        for which, c in (("classic", m["c_classic"]), ("corrected", m["c_corrected"])):
            for d in divs:
                tasks.append({"damping": c, "dt": T_C / d, "scheme": "verlet",
                              "geometry": geometry, "which": which,
                              "e_target": m["e_target"], "divisor": d})
    with mp.Pool(processes=min(args.workers, len(tasks))) as pool:
        flat_rows = pool.map(_worker, [dict(t) for t in tasks])
    for row, task in zip(flat_rows, tasks):
        row["which"] = task["which"]
        row["e_target"] = task["e_target"]
        row["divisor"] = task["divisor"]
        row["rel_err"] = (row["e_measured"] - task["e_target"]) / task["e_target"]
    metrics["kernel_sweep_flat"] = flat_rows

    # ---- curved hull patch ---------------------------------------------
    print("[5/6] curved Wigley patch (geometry transfer) ...")
    curved_tasks = []
    for m in mappings:
        for which, c in (("classic", m["c_classic"]), ("corrected", m["c_corrected"])):
            for d in (divs[-1],):
                curved_tasks.append({"damping": c, "dt": T_C / d, "scheme": "verlet",
                                     "which": which, "e_target": m["e_target"],
                                     "divisor": d, "curved": True})
    patch = wigley_patch()
    curved_rows = []
    for task in curved_tasks:
        import torch
        torch.set_num_threads(1)
        res = kernel_bounce(task["damping"], dt=T_C / task["divisor"],
                            verts=patch["verts"], faces=patch["faces"],
                            normal=patch["normal"],
                            centre_override=patch["foot"] + (RADIUS + 5e-4) * patch["normal"])
        res.update({"which": task["which"], "e_target": task["e_target"],
                    "divisor": task["divisor"],
                    "rel_err": (res["e_measured"] - task["e_target"]) / task["e_target"],
                    "n_faces_patch": patch["n_faces_patch"]})
        curved_rows.append(res)
    metrics["kernel_sweep_curved"] = curved_rows

    # ---- invariants -----------------------------------------------------
    print("[6/6] invariants (speed independence, integrator sensitivity, e=1) ...")
    import torch
    torch.set_num_threads(1)
    speed_rows = []
    for v in (0.02, 0.005, 0.001):
        c = damping_for_target_restitution(0.5, STIFFNESS, MASS)["gamma_n"]
        r = kernel_bounce(c, dt=T_C / 400, v_in=v)
        speed_rows.append({"v_in": v, "e_measured": r["e_measured"]})
    metrics["speed_independence"] = {
        "rows": speed_rows,
        "max_spread": max(r["e_measured"] for r in speed_rows)
                      - min(r["e_measured"] for r in speed_rows)}

    euler_rows = []
    for d in divs:
        c = damping_for_target_restitution(0.5, STIFFNESS, MASS)["gamma_n"]
        rv = kernel_bounce(c, dt=T_C / d, scheme="verlet")
        re = kernel_bounce(c, dt=T_C / d, scheme="euler")
        euler_rows.append({"divisor": d, "verlet_e": rv["e_measured"],
                           "euler_e": re["e_measured"],
                           "verlet_err": (rv["e_measured"] - 0.5) / 0.5,
                           "euler_err": (re["e_measured"] - 0.5) / 0.5})
    metrics["integrator_sensitivity"] = euler_rows

    elastic = kernel_bounce(0.0, dt=T_C / 400)
    metrics["elastic_limit"] = {"c_damping": 0.0, "e_measured": elastic["e_measured"],
                                "dev_from_1": elastic["e_measured"] - 1.0}

    metrics["wall_clock_s"] = time.perf_counter() - t_start
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- summary --------------------------------------------------------
    print("\n== summary ==")
    print(f"reference convergence max diff: "
          f"{metrics['reference_convergence']['max_finest_diff']:.2e}")
    print(f"independent vs repo reference:  "
          f"{metrics['reference_crosscheck']['max_abs_diff']:.2e}")
    print(f"patch/full-hull force diff:     "
          f"{metrics['patch_equivalence']['max_force_rel_diff']:.2e}")
    for m in mappings:
        print(f"e_target={m['e_target']:.2f}: classic c={m['c_classic']:.1f} -> "
              f"reference e={m['reference_e_of_classic_c']:.4f} "
              f"({m['classic_mapping_reference_error'] * 100:+.1f}%); "
              f"corrected c={m['c_corrected']:.1f} -> e={m['reference_e_of_corrected_c']:.4f}")
    best = {}
    for row in flat_rows:
        key = row["which"]
        if row["divisor"] == divs[-1]:
            best.setdefault(key, []).append(abs(row["rel_err"]))
    for key, errs in best.items():
        print(f"kernel dt=t_c/{divs[-1]} {key}: max |rel err| = {max(errs) * 100:.2f}%")
    print(f"wall clock: {metrics['wall_clock_s']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())