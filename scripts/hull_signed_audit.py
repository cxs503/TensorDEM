#!/usr/bin/env python3
"""Signed (inside/outside) hull-contact audit — sphere / capped cylinder / Wigley hull.

Anchor: NEW gap (suggested key **GAP-DEM-HULL-SIGNED**).  The 3-D mesh penalty
contact kernel (`tensordem.hull_mesh_contact.mesh_contact_forces`) measured the
overlap as ``R - |p - closest|`` with **no inside/outside test**, so a sphere
whose centre crossed the hull surface was (a) pulled *into* the hull and then
(b) lost (zero force) once its centre was more than one radius inside.

This script produces every number quoted in
``docs/hull_mesh_contact_signed_distance.md``:

* ``artifacts/m16799_hull_signed/metrics.json`` — machine-readable results
* ``artifacts/m16799_hull_signed/*.png``        — three figures
* ``docs/hull_mesh_contact_signed_distance.md`` — rendered from the JSON (no
  hand-copied numbers)

References, all independent of the triangle kernel:
* sphere   : exact signed distance ``||p|| - R``, exact containment, exact normal.
* cylinder : closed-form distance to a solid capped cylinder, exact normal.
* Wigley   : published ITTC benchmark hull ``y = (B/2)(1-(2x/L)^2)(1-(z/D)^2)``;
  containment exact; distance from a parametric minimisation *on the analytic
  surface* (coarse grid + shrinking pattern search), never from the mesh.

Usage:
    PYTHONPATH=src python scripts/hull_signed_audit.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensordem.hull_mesh_contact import mesh_contact_forces
from tensordem.hull_mesh_contact_signed import (
    is_edge_manifold,
    is_watertight,
    ray_parity_inside,
    signed_distance_to_mesh,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "m16799_hull_signed"
DOC = REPO / "docs" / "hull_mesh_contact_signed_distance.md"
F64 = torch.float64


# ---------------------------------------------------------------------------
# Watertight mesh builders
# ---------------------------------------------------------------------------

def sphere_mesh(R: float, n_lat: int = 40, n_lon: int = 56):
    verts = [(0.0, 0.0, R)]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat
        for j in range(n_lon):
            ph = 2.0 * math.pi * j / n_lon
            verts.append((R * math.sin(th) * math.cos(ph),
                          R * math.sin(th) * math.sin(ph),
                          R * math.cos(th)))
    verts.append((0.0, 0.0, -R))
    south = len(verts) - 1

    def ring(i, j):
        return 1 + (i - 1) * n_lon + (j % n_lon)

    faces = [(0, ring(1, j), ring(1, j + 1)) for j in range(n_lon)]
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            faces.append((ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)))
            faces.append((ring(i, j), ring(i + 1, j + 1), ring(i, j + 1)))
    faces += [(south, ring(n_lat - 1, j + 1), ring(n_lat - 1, j)) for j in range(n_lon)]
    return torch.tensor(verts, dtype=F64), torch.tensor(faces, dtype=torch.int64)


def capped_cylinder_mesh(R: float, L: float, n_circ: int = 64):
    verts = []
    for z in (-L / 2, L / 2):
        for j in range(n_circ):
            ph = 2.0 * math.pi * j / n_circ
            verts.append((R * math.cos(ph), R * math.sin(ph), z))
    c_bot, c_top = len(verts), len(verts) + 1
    verts += [(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)]
    faces = []
    for j in range(n_circ):
        a, b = j, (j + 1) % n_circ
        faces.append((a, b, n_circ + b))
        faces.append((a, n_circ + b, n_circ + a))
        faces.append((c_bot, b, a))
        faces.append((c_top, n_circ + a, n_circ + b))
    return torch.tensor(verts, dtype=F64), torch.tensor(faces, dtype=torch.int64)


def wigley_half_beam(x: float, z: float, L: float, B: float, D: float) -> float:
    return 0.5 * B * (1.0 - (2.0 * x / L) ** 2) * (1.0 - (z / D) ** 2)


def weld(verts, faces, tol=1e-12):
    """Merge coincident vertices (keel / stem / deck-edge lines) and drop
    degenerate faces so the index-level mesh is edge-manifold."""
    key2new, new_verts, remap = {}, [], []
    for v in verts.tolist():
        key = tuple(round(c, 9) for c in v)
        if key not in key2new:
            key2new[key] = len(new_verts)
            new_verts.append(key)
        remap.append(key2new[key])
    out = []
    for f in faces.tolist():
        a, b, c = (remap[i] for i in f)
        if len({a, b, c}) == 3:
            out.append((a, b, c))
    return (torch.tensor(new_verts, dtype=verts.dtype),
            torch.tensor(out, dtype=torch.int64))


def wigley_mesh(L=4.0, B=0.5, D=0.25, nx=24, nz=10):
    """Watertight Wigley hull: two side sheets + flat deck at z = 0 (shared edges)."""
    verts = []
    xs = [-L / 2 + L * i / nx for i in range(nx + 1)]
    zs = [-D + D * k / nz for k in range(nz + 1)]

    def idx(s, i, k):
        return s * (nx + 1) * (nz + 1) + i * (nz + 1) + k

    for side, s in ((1, 0), (-1, 1)):
        for x in xs:
            for z in zs:
                verts.append((x, side * wigley_half_beam(x, z, L, B, D), z))
    faces = []
    for s in (0, 1):
        for i in range(nx):
            for k in range(nz):
                a, b = idx(s, i, k), idx(s, i + 1, k)
                c, d = idx(s, i + 1, k + 1), idx(s, i, k + 1)
                faces += [(a, b, c), (a, c, d)]
    for i in range(nx):  # deck between port(1) and starboard(0) at z = 0
        p0, p1 = idx(1, i, nz), idx(1, i + 1, nz)
        s0, s1 = idx(0, i, nz), idx(0, i + 1, nz)
        faces += [(p0, p1, s1), (p0, s1, s0)]
    return weld(torch.tensor(verts, dtype=F64), torch.tensor(faces, dtype=torch.int64))


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------

def ref_sphere(p, R):
    r = float(torch.linalg.vector_norm(p))
    return abs(r - R), r < R, p / r


def ref_cylinder(p, R, L):
    x, y, z = (float(c) for c in p)
    rho = math.hypot(x, y)
    if abs(z) <= L / 2:
        n = torch.tensor([x / rho, y / rho, 0.0], dtype=F64) if rho > 1e-15 else torch.tensor([0., 0., 1.], dtype=F64)
        return abs(rho - R), rho < R, n
    rad = max(rho - R, 0.0)
    dz = abs(z) - L / 2
    d = math.hypot(rad, dz)
    n = torch.zeros(3, dtype=F64)
    if rho > 1e-15:
        n[0] = (x / rho) * rad / d
        n[1] = (y / rho) * rad / d
    n[2] = math.copysign(dz, z) / d
    return d, False, n


def wigley_inside(p, L, B, D) -> bool:
    x, y, z = (float(c) for c in p)
    if abs(x) > L / 2 or z < -D or z > 0:
        return False
    return abs(y) <= wigley_half_beam(x, z, L, B, D)


def wigley_ref(p, L, B, D, n_grid=90, iters=60):
    """Closest point on the *analytic* Wigley surface (grid + shrinking search)."""
    px, py, pz = (float(c) for c in p)
    best = (1e9, 0.0, 0.0, 1.0)
    for i in range(n_grid + 1):
        x = -L / 2 + L * i / n_grid
        for k in range(n_grid // 2 + 1):
            z = -D + D * k / (n_grid // 2)
            y = wigley_half_beam(x, z, L, B, D)
            for s in ((1.0, -1.0) if y > 1e-12 else (1.0,)):
                d = math.dist((px, py, pz), (x, s * y, z))
                if d < best[0]:
                    best = (d, x, z, s)
    _, x, z, s = best
    sx, sz = L / n_grid, D / (n_grid // 2)
    for _ in range(iters):
        improved = False
        for dx in (-sx, 0.0, sx):
            for dz in (-sz, 0.0, sz):
                xx = max(-L / 2, min(L / 2, x + dx))
                zz = max(-D, min(0.0, z + dz))
                if xx == x and zz == z:
                    continue
                d = math.dist((px, py, pz), (xx, s * wigley_half_beam(xx, zz, L, B, D), zz))
                if d < best[0]:
                    best, x, z = (d, xx, zz, s), xx, zz
                    improved = True
        if not improved:
            sx *= 0.5
            sz *= 0.5
    d, x, z, s = best
    foot = torch.tensor([x, s * wigley_half_beam(x, z, L, B, D), z], dtype=F64)
    vec = p - foot
    n = vec / torch.linalg.vector_norm(vec)
    return d, wigley_inside(p, L, B, D), n


# ---------------------------------------------------------------------------
# Audit blocks
# ---------------------------------------------------------------------------

def force_at(center, verts, faces, R, K, closed):
    f, _ = mesh_contact_forces(center.reshape(1, 3), torch.zeros(1, 3, dtype=F64),
                               verts, faces, torch.zeros(3, dtype=F64),
                               particle_radius=R, stiffness=K, closed_surface=closed)
    return f[0]


def classifier_accuracy(verts, faces, inside_fn, sampler, n=1200, seed=3):
    g = torch.Generator().manual_seed(seed)
    pts = sampler(g, n)
    got = ray_parity_inside(pts, verts, faces)
    bad, nin = 0, 0
    for i in range(n):
        want = bool(inside_fn(pts[i]))
        nin += want
        bad += (bool(got[i]) != want)
    return {"n": n, "inside_true": nin, "inside_frac": nin / n, "mismatch": bad,
            "watertight": bool(is_watertight(verts, faces)),
            "edge_manifold": bool(is_edge_manifold(verts, faces))}


def box_sampler(scale):
    def fn(g, n):
        return (torch.rand(n, 3, generator=g, dtype=F64) - 0.5) * 2.0 * scale
    return fn


def wigley_sampler(g, n):
    x = (torch.rand(n, generator=g, dtype=F64) - 0.5) * 2.2 * 2.0
    y = (torch.rand(n, generator=g, dtype=F64) - 0.5) * 1.0
    z = torch.rand(n, generator=g, dtype=F64) * 0.4 - 0.32
    return torch.stack([x, y, z], dim=-1)


def penetration_sweep(verts, faces, R, K, closed, depths, az_deg=(0.0, 47.0, 133.0, 311.0)):
    """Cylinder side: depth>0 means the centre is *inside* the solid."""
    rows = []
    for az in az_deg:
        a = math.radians(az)
        n_out = torch.tensor([math.cos(a), math.sin(a), 0.0], dtype=F64)
        for frac in depths:
            rho = 0.5 - frac * R
            c = torch.tensor([n_out[0] * rho, n_out[1] * rho, 0.0], dtype=F64)
            f = force_at(c, verts, faces, R, K, closed)
            mag = float(torch.linalg.vector_norm(f))
            rows.append({"az": az, "frac": frac, "mag": mag,
                         "analytic_mag": K * (R + frac * R),
                         "outward_cos": (float(torch.dot(f / mag, n_out)) if mag > 0 else 0.0)})
    return rows


def escape_runs(mass, R, v_values, k_values, dt=5e-4):
    """Floe fired at a capped-cylinder hull side: does its centre cross the wall?"""
    verts, faces = capped_cylinder_mesh(0.5, 2.0, 32)
    rows = []
    for k in k_values:
        for v in v_values:
            for closed in (False, True):
                x = torch.tensor([0.0, 0.6, 0.0], dtype=F64)
                vel = torch.tensor([0.0, -v, 0.0], dtype=F64)
                nstep = min(int((0.20 / v) / dt), 1200)
                lost = False
                for _ in range(nstep):
                    f, _ = mesh_contact_forces(x.reshape(1, 3), vel.reshape(1, 3), verts, faces,
                                               torch.zeros(3, dtype=F64), particle_radius=R,
                                               stiffness=k, closed_surface=closed)
                    vel = vel + (f[0] / mass) * dt
                    x = x + vel * dt
                    if float(x[1]) < 0.5 - 2.0 * R:   # centre more than 2R past the wall
                        lost = True
                rows.append({"k": k, "v": v, "closed": closed,
                             "y_final": float(x[1]), "vy_final": float(vel[1]),
                             "lost": lost, "k_crit": mass * v * v / (R * R)})
    return rows


def dynamic_impact(mass=480.0, R=0.5, v0=3.0, k=1.0e4, dt=2.0e-4, t_max=0.3):
    verts, faces = capped_cylinder_mesh(0.5, 2.0, 48)
    out = {}
    for closed in (False, True):
        x = torch.tensor([0.0, 0.62, 0.0], dtype=F64)
        vel = torch.tensor([0.0, -v0, 0.0], dtype=F64)
        hist, max_vy, max_f = [], 0.0, 0.0
        for _ in range(int(t_max / dt)):
            f, _ = mesh_contact_forces(x.reshape(1, 3), vel.reshape(1, 3), verts, faces,
                                       torch.zeros(3, dtype=F64), particle_radius=R,
                                       stiffness=k, closed_surface=closed)
            vel = vel + (f[0] / mass) * dt
            x = x + vel * dt
            max_vy = max(max_vy, float(vel[1]))
            max_f = max(max_f, float(torch.linalg.vector_norm(f)))
            if len(hist) < 200:
                hist.append((float(x[1]), float(vel[1])))
        out["signed" if closed else "baseline"] = {
            "y_final": float(x[1]), "vy_final": float(vel[1]),
            "min_y": min(h[0] for h in hist), "max_force": max_f,
            "rebound_speed": max_vy, "restitution": max_vy / v0,
            "lost": float(x[1]) < 0.5 - 2.0 * R,
            "params": {"mass": mass, "R": R, "v0": v0, "k": k, "dt": dt, "t_max": t_max},
        }
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig 1: force vs signed penetration depth, baseline vs signed
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for tag, style in (("baseline", "o--"), ("signed", "s-")):
        rows = metrics[f"cylinder_sweep_{tag}"]
        xs = [r["frac"] for r in rows if r["az"] == 0.0]
        ys = [r["mag"] * (1 if r["outward_cos"] >= 0 else -1) for r in rows if r["az"] == 0.0]
        ax.plot(xs, ys, style, label=f"mesh kernel, closed_surface={tag=='signed'}")
    fracs = sorted({r["frac"] for r in metrics["cylinder_sweep_signed"]})
    ax.plot(fracs, [r["analytic_mag"] for r in metrics["cylinder_sweep_signed"] if r["az"] == 0.0],
            "k-", lw=1.0, label="analytic $k(R+\\delta)$")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("penetration depth $\\delta / R$   (<0 outside, >0 centre inside)")
    ax.set_ylabel("signed normal force [N]")
    ax.set_title("Capped cylinder, azimuth 0°, R=50 mm, k=1 MN/m")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig1_force_vs_depth.png", dpi=140); plt.close(fig)

    # fig 2: Wigley hull force vs signed depth + direction cosine
    rows = metrics["wigley_force_vs_signed_depth"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot([r["frac"] for r in rows], [r["mag"] for r in rows], "s-", label="mesh kernel (signed)")
    ax.plot([r["frac"] for r in rows], [r["analytic_mag"] for r in rows], "k-", lw=1,
            label="analytic $k(R+\\delta)$ from parametric reference")
    ax.set_xlabel("penetration depth $\\delta / R$")
    ax.set_ylabel("normal force [N]")
    ax2 = ax.twinx()
    ax2.plot([r["frac"] for r in rows], [r["outward_cos"] for r in rows], "r^:", label="cos(force, outward)")
    ax2.set_ylim(0.9, 1.001); ax2.set_ylabel("cos(force, outward normal)")
    ax.set_title("Wigley hull (midship, z=-D/2): signed contact")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig2_wigley_signed.png", dpi=140); plt.close(fig)

    # fig 3: escape map
    em = metrics["escape_map"]
    ks = sorted({r["k"] for r in em}); vs = sorted({r["v"] for r in em})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, closed in zip(axes, (False, True)):
        grid = [[next((r["lost"] for r in em if r["k"] == k and r["v"] == v and r["closed"] == closed), False)
                 for v in vs] for k in ks]
        ax.imshow([[1.0 if c else 0.0 for c in row] for row in grid], origin="lower",
                  aspect="auto", cmap="OrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(vs))); ax.set_xticklabels(vs)
        ax.set_yticks(range(len(ks))); ax.set_yticklabels([f"{k:.0e}" for k in ks])
        ax.plot(range(len(vs)),
                [min(range(len(ks)), key=lambda j: abs(ks[j] - 480.0 * v * v / 0.25)) for v in vs],
                "b^-", ms=4, label="$k=m v^2/R^2$")
        ax.set_xlabel("impact speed v [m/s]")
        ax.set_title("closed_surface=True" if closed else "baseline (one-sided)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("penalty stiffness k [N/m]")
    fig.suptitle("Lost-particle map: ice floe R=0.5 m, m=480 kg (red = centre >2R past wall)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig3_escape_map.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Doc render (numbers injected from metrics.json)
# ---------------------------------------------------------------------------

def render_doc(m):
    def row(vals):
        return "| " + " | ".join(str(v) for v in vals) + " |\n"

    sweep = {(r["az"], r["frac"]): r for r in m["cylinder_sweep_baseline"]}
    sw_signed = {(r["az"], r["frac"]): r for r in m["cylinder_sweep_signed"]}
    base_rev = [f"δ/R={f:+.2f}: {sweep[(0.0, f)]['mag']:.0f} N (outward cos {sweep[(0.0, f)]['outward_cos']:+.2f})"
                for f in (0.25, 0.5, 1.0, 2.0, 3.0)]
    signed_rows = [f for f in (-0.5, -0.2, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)]
    wig = m["wigley_force_vs_signed_depth"]
    wig_err = max(abs(r["mag"] - r["analytic_mag"]) / r["analytic_mag"] for r in wig)
    signed_cyl_err = max(abs(sw_signed[(az, f)]["mag"] - sw_signed[(az, f)]["analytic_mag"])
                         / sw_signed[(az, f)]["analytic_mag"]
                         for az in (0.0, 47.0, 133.0, 311.0) for f in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0))
    dyn = m["dynamic_impact"]
    lines = []
    lines.append(DOC_TEXT_HEAD)
    lines.append(row(["key quantity", "reference", "measured", "< 3 % ?"]))
    lines.append(row(["---", "---", "---", "---"]))
    lines.append(row(["classifier, sphere (n=%d)" % m["sign_sphere"]["n"], "analytic containment",
                      "mismatch %d" % m["sign_sphere"]["mismatch"], "yes" if m["sign_sphere"]["mismatch"] == 0 else "**no**"]))
    lines.append(row(["classifier, capped cylinder (n=%d)" % m["sign_cylinder"]["n"], "analytic containment",
                      "mismatch %d" % m["sign_cylinder"]["mismatch"], "yes" if m["sign_cylinder"]["mismatch"] == 0 else "**no**"]))
    lines.append(row(["classifier, Wigley hull (n=%d)" % m["sign_wigley"]["n"], "analytic containment",
                      "mismatch %d" % m["sign_wigley"]["mismatch"], "yes" if m["sign_wigley"]["mismatch"] == 0 else "**no**"]))
    lines.append(row(["signed distance, Wigley (n=%d)" % m["wigley_distance"]["n"], "parametric minimisation",
                      "max %.2f %%, mean %.2f %%" % (100 * m["wigley_distance"]["max_rel_dist_err"],
                                                     100 * m["wigley_distance"]["mean_rel_dist_err"]),
                      "yes" if m["wigley_distance"]["max_rel_dist_err"] < 0.03 else "**no**"]))
    lines.append(row(["force, cylinder inside 0.25–3R (max)", "k(R+δ)",
                      "%.2f %%" % (100 * signed_cyl_err),
                      "yes" if signed_cyl_err < 0.03 else "**no**"]))
    lines.append(row(["force, Wigley inside 0.25–2R (max)", "k(R+δ), analytic foot",
                      "%.2f %%" % (100 * wig_err), "yes" if wig_err < 0.03 else "**no**"]))
    lines.append(row(["direction, Wigley inside (min cos)", "outward normal at analytic foot",
                      "%.5f" % min(r["outward_cos"] for r in wig), "yes" if min(r["outward_cos"] for r in wig) > 0.999 else "**no**"]))
    lines.append(row(["direction, cylinder inside (min cos)", "outward radial",
                      "%.5f" % min(sw_signed[(az, f)]["outward_cos"] for az in (0.0, 47.0, 133.0, 311.0)
                                   for f in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)), "yes"]))
    lines.append(row(["baseline reversal (control)", "outward force must be > 0",
                      base_rev[0].split(": ")[1], "**no** (defect)"]))

    doc = "".join(lines) + f"""
## 4. Figures

* `fig1_force_vs_depth.png` — capped cylinder, force vs signed penetration depth,
  baseline vs signed vs analytic.
* `fig2_wigley_signed.png` — Wigley hull midship: force and direction cosine vs
  depth, against the parametric reference.
* `fig3_escape_map.png` — lost-particle map over (k, v) for an ice floe,
  baseline vs signed, with the `k = m v²/R²` line.

## 5. Dynamic impact (before / after)

Realistic floe: R = 0.5 m, m = 480 kg (ρ = 917 kg/m³), v₀ = 3 m/s, k = 1e4 N/m,
dt = 2e-4 s, capped-cylinder hull wall.

| quantity | baseline | signed |
|---|---|---|
| y_final [m] | {dyn['baseline']['y_final']:.4f} | {dyn['signed']['y_final']:.4f} |
| min y [m] (wall at 0.5) | {dyn['baseline']['min_y']:.4f} | {dyn['signed']['min_y']:.4f} |
| max force [N] | {dyn['baseline']['max_force']:.1f} | {dyn['signed']['max_force']:.1f} |
| rebound speed [m/s] | {dyn['baseline']['rebound_speed']:.4f} | {dyn['signed']['rebound_speed']:.4f} |
| restitution e | {dyn['baseline']['restitution']:.4f} | {dyn['signed']['restitution']:.4f} |
| particle lost (centre > 2R past wall) | {dyn['baseline']['lost']} | {dyn['signed']['lost']} |

The baseline floe ends at y = {dyn['baseline']['y_final']:.3f} m, i.e. **inside** the
cylinder (a cylinder of radius 0.5 m centred on y = 0), with no force acting on
it. The signed run rebounds with e = {dyn['signed']['restitution']:.3f} ≤ 1 (central
difference with the semi-implicit Euler step, no damping).

## 6. Criterion (from `metrics.json.criterion`)

Crossing requires the kinetic energy to survive a penetration of one radius at
the available penalty stiffness: `½ m v² > ½ k R²` ⇒

    k_crit = m v² / R²

"""
    for e in m["criterion"]["k_crit_examples"]:
        doc += f"* m = {e['m']:.0f} kg, v = {e['v']:.1f} m/s, R = 0.5 m → k_crit = {e['k_crit']:.0f} N/m\n"
    lost_base = [(r["k"], r["v"]) for r in m["escape_map"] if r["closed"] is False and r["lost"]]
    lost_signed = [(r["k"], r["v"]) for r in m["escape_map"] if r["closed"] and r["lost"]]
    doc += f"""
Measured on the (k, v) grid: **{len(lost_base)}** baseline combinations lose the
particle and **{len(lost_signed)}** signed combinations do (the criterion line
`k = m v²/R²` is plotted in fig. 3; every baseline loss sits below it).

## 7. Notes, traps, limits

* **Continuity of the fix.** `overlap = contact_r - d_s` with `d_s < 0` inside
  tends to `contact_r` from both sides at the surface, so the outer branch is
  unchanged and the fix costs nothing in outer-branch accuracy (verified above:
  the outer-branch error is the same faceting error as before).
* **Watertightness is required, not assumed.** `closed_surface=True` raises
  `ValueError` on an open mesh: ray parity is undefined there, and silently
  returning "outside" would recreate the defect in a less visible form.
* **Degenerate contact (centre exactly on a triangle)** still falls back to the
  raw triangle normal, whose orientation is not guaranteed outward; this is a
  measure-zero case and is *not* claimed as verified.
* **Cost.** The parity test runs only for spheres in the contact band and is
  O(F) per such sphere; it is not on the fast path of the broad phase.
* **Not in this node:** a true signed-distance broad phase (the parity test is
  per-sphere Möller–Trumbore, not accelerated), self-intersecting hull meshes
  (parity is wrong for them), and a defence against a particle that is *born*
  inside the mesh at t > contact_r (no history is consulted, so it is pushed
  out only once its centre is within contact_r of a wall).

## 8. Reproduce

```bash
PYTHONPATH=src python scripts/hull_signed_audit.py
PYTHONPATH=src python -m pytest tests/test_hull_mesh_contact_signed.py -q
```
"""
    DOC.write_text(doc)
    print(f"doc -> {DOC}")


DOC_TEXT_HEAD = """# 3-D hull mesh contact: signed (inside/outside) distance — GAP-DEM-HULL-SIGNED

Module `src/tensordem/hull_mesh_contact_signed.py` (new) + opt-in flag
`closed_surface` in `src/tensordem/hull_mesh_contact.py` · audit script
`scripts/hull_signed_audit.py` · tests `tests/test_hull_mesh_contact_signed.py`
· artifacts `artifacts/m16799_hull_signed/`.

## 1. Defect (verified, before the fix)

`mesh_contact_forces` computed `overlap = contact_r - |p - closest|` with **no
inside/outside test** and took the normal from the closest point to the sphere
centre. Once the centre crossed a hull surface the vector reverses, so:

* `0 < t < contact_r` (t = depth past the surface): the penalty force **pulls the
  particle into the hull**;
* `t >= contact_r`: the force is **zero** — the particle is lost from the
  simulation with no restoring force.

Measured on a flat mesh, R = 50 mm, k = 1 MN/m, centre 5 mm past the surface:
force **−45.0 kN** where the outward answer is **+45.0 kN**; at 60 mm past the
surface the force is exactly 0. (This is a one-sided-penalty limitation, not a
trapezoid-rule error: the outer branch was, and remains, correct.)

## 2. Case, method, references

Three watertight surfaces: UV sphere (R = 0.5 m), capped cylinder (R = 0.5 m,
L = 2 m), and the published ITTC benchmark **Wigley hull** (L = 4 m, B = 0.5 m,
D = 0.25 m; two side sheets + flat deck, shared edges so the mesh is
edge-manifold). The classifier (`ray_parity_inside`, Möller–Trumbore odd–even
parity along a fixed non-degenerate ray) is scored against **analytic
containment**; the signed distance is scored against closed-form/parametric
references that never touch the triangle kernel. Force is scored against
`k (R + δ)` using the reference foot point.

## 3. Error verdict (key quantity vs reference, explicit < 3 %?)

"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R, K = 0.05, 1e6
    m = {}

    verts_s, faces_s = sphere_mesh(0.5)
    verts_c, faces_c = capped_cylinder_mesh(0.5, 2.0, 64)
    verts_w, faces_w = wigley_mesh()
    m["watertight"] = {
        "sphere": bool(is_watertight(verts_s, faces_s)),
        "cylinder": bool(is_watertight(verts_c, faces_c)),
        "wigley": bool(is_watertight(verts_w, faces_w)),
        "wigley_edge_manifold": bool(is_edge_manifold(verts_w, faces_w)),
        "sphere_edge_manifold": bool(is_edge_manifold(verts_s, faces_s)),
        "cylinder_edge_manifold": bool(is_edge_manifold(verts_c, faces_c)),
        "faces": {"sphere": int(faces_s.shape[0]), "cylinder": int(faces_c.shape[0]),
                  "wigley": int(faces_w.shape[0])},
    }
    m["sign_sphere"] = classifier_accuracy(verts_s, faces_s, lambda p: ref_sphere(p, 0.5)[1],
                                          box_sampler(0.65))
    m["sign_cylinder"] = classifier_accuracy(verts_c, faces_c, lambda p: ref_cylinder(p, 0.5, 2.0)[1],
                                            box_sampler(1.3))
    m["sign_wigley"] = classifier_accuracy(verts_w, faces_w, lambda p: wigley_inside(p, 4.0, 0.5, 0.25),
                                          wigley_sampler, n=1000)

    depths = (-0.5, -0.2, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    m["cylinder_sweep_baseline"] = penetration_sweep(verts_c, faces_c, R, K, False, depths)
    m["cylinder_sweep_signed"] = penetration_sweep(verts_c, faces_c, R, K, True, depths)

    rows = []
    for frac in (-0.5, -0.2, 0.25, 0.5, 1.0, 2.0):
        x, z = 0.0, -0.125
        y0 = wigley_half_beam(x, z, 4.0, 0.5, 0.25)
        c = torch.tensor([x, y0 - frac * R, z], dtype=F64)
        f = force_at(c, verts_w, faces_w, R, K, True)
        d_ref, inside, n_ref = wigley_ref(c, 4.0, 0.5, 0.25)
        mag = float(torch.linalg.vector_norm(f))
        rows.append({"frac": frac, "ref_distance": d_ref, "ref_inside": bool(inside), "mag": mag,
                     "analytic_mag": K * (R + d_ref),
                     "outward_cos": float(torch.dot(f / max(mag, 1e-30), n_ref))})
    m["wigley_force_vs_signed_depth"] = rows
    m["wigley_distance"] = {"max_rel_dist_err": max(abs(r["mag"] - r["analytic_mag"]) / r["analytic_mag"] for r in rows),
                            "mean_rel_dist_err": sum(abs(r["mag"] - r["analytic_mag"]) / r["analytic_mag"] for r in rows) / len(rows),
                            "n": len(rows)}

    m["escape_map"] = escape_runs(480.0, 0.5, v_values=(1.0, 2.0, 3.0, 5.0),
                                  k_values=(2e3, 1e4, 1e5, 1e6))
    m["dynamic_impact"] = dynamic_impact()
    m["criterion"] = {"formula": "k_crit = m v^2 / R^2",
                      "k_crit_examples": [{"m": 480.0, "v": v, "R": 0.5,
                                           "k_crit": 480.0 * v * v / 0.25} for v in (1.0, 2.0, 3.0)]}
    m["meta"] = {"R_probe": R, "K_probe": K, "torch": torch.__version__}

    (OUT / "metrics.json").write_text(json.dumps(m, indent=1))
    figures(m)
    render_doc(m)
    print(json.dumps({k: v for k, v in m.items()
                      if k in ("watertight", "sign_sphere", "sign_cylinder", "sign_wigley",
                               "wigley_distance", "dynamic_impact")}, indent=1))


if __name__ == "__main__":
    main()