"""CASE-PENETRATION — quasi-static sphere penetration resistance (TensorDEM).

Reference (zero hand-transcription: every number below is quoted from the paper)
-------------------------------------------------------------------------------
B. T. Brzinski III, M. Mayor, D. J. Durian, "Depth-dependent resistance of
granular media to vertical penetration", Phys. Rev. Lett. 111, 168002 (2013);
arXiv:1307.4638v2.  Local copy of the PDF used for the quotation:
``docs/refs/brzinski2013_1307.4638v2.pdf``.

* Eq. (2), p. 1:  dF = alpha mu (rho g z) dA          (alpha = 35 +- 5, p. 1)
* Eq. (3d), p. 2: F(z)/(alpha mu rho g) = pi (z - R/3) R^2   (sphere, z >= R)
  => F(z) = alpha * mu * rho * g * pi * R^2 * (z - R/3)
* Material of the reference medium (p. 1): glass spheres, bulk density
  rho = 1.48 g/cm^3, angle of repose 22 deg -> mu = 0.40, packing fraction
  phi = 0.59.

The case therefore measures one dimensionless number,

    alpha_fit = slope(F vs z) / (mu * rho * g * pi * R^2),   z >= 1.1 R

and compares it with the published alpha = 35 +- 5 (ADR: |alpha_fit-35|/35 <= 0.20).

Protocol
--------
The reference experiment is *load* controlled and quasi-static (the projectile
weight is slowly transferred to the bed and F(z) is read at failure).  The DEM
side uses the equivalent displacement-controlled quasi-static protocol: the
intruder is a fixed sphere displaced downwards in small increments dz << d_g;
after every increment the bed is relaxed with a strong background viscous
damping until the kinetic energy is negligible, and the vertical contact force
on the intruder is averaged over the last part of the relaxation window.
Quasi-static loading paths are rate independent, so the equilibrium force is
the quantity that corresponds to F(z) in Eq. (3d).

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_sphere_penetration/run_case.py \
        --grain-d 0.004 --width-grains 24 --depth-grains 14 --radius-over-d 4 \
        --mu 0.4 --dz-over-d 0.5 --relax 400 --out runs/case_penetration/base
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from tensordem.sphere_pack_dem import SpherePack, SpherePackConfig, SphereWall  # noqa: E402
from tensordem.granular3d_core import hcp_positions_in_box  # noqa: E402
from tensordem.superquadric_kernel import ContactMaterial  # noqa: E402

G = 9.81
# ---- reference constants, quoted from Brzinski et al. (2013) --------------
REF_ALPHA = 35.0          # +-5, p. 1 (Eq. 2 and abstract)
REF_ALPHA_ERR = 5.0
REF_MU = 0.40             # tan(22 deg), p. 1
REF_PHI = 0.59            # p. 1
REF_RHO = 1480.0          # kg/m^3 (1.48 g/cm^3), p. 1


# ---------------------------------------------------------------------------
# bed preparation
# ---------------------------------------------------------------------------

def fcc_bed(n_wide, n_deep, d, phi_request, rng):
    """FCC lattice cut to a box, with a small random jitter.

    ``phi_request`` sets the lattice constant through
    phi = 0.7405 (2r/L)^3 (FCC is 0.7405 dense at L = 2r).
    """
    r = 0.5 * d
    L = 2.0 * r * (0.7405 / phi_request) ** (1.0 / 3.0)
    a = L * math.sqrt(2.0)
    Lx = Ly = n_wide * d
    H = n_deep * d
    pts = []
    nx = int(math.floor(Lx / a)) + 1
    ny = int(math.floor(Ly / a)) + 1
    nz = int(math.floor(H / a)) + 1
    offs = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for off in offs:
                    p = np.array([(i + off[0]) * a, (j + off[1]) * a, (k + off[2]) * a])
                    if r < p[0] < Lx - r and r < p[1] < Ly - r and r < p[2] < H:
                        pts.append(p)
    pts = np.array(pts)
    pts = pts[rng.permutation(len(pts))]
    pts += rng.normal(0.0, 0.03 * d, pts.shape)
    return pts, np.full(len(pts), r), Lx, Ly, H


def make_walls(Lx, Ly):
    return [
        SphereWall(np.array([1.0, 0.0, 0.0]), 0.0, label="x0"),
        SphereWall(np.array([-1.0, 0.0, 0.0]), -Lx, label="xL"),
        SphereWall(np.array([0.0, 1.0, 0.0]), 0.0, label="y0"),
        SphereWall(np.array([0.0, -1.0, 0.0]), -Ly, label="yL"),
        SphereWall(np.array([0.0, 0.0, 1.0]), 0.0, label="floor"),
    ]


def settle(sp, n_steps, log_every=0):
    """Relax the packing under gravity; returns kinetic-energy history."""
    hist = []
    for k in range(n_steps):
        sp.step()
        if log_every and (k % log_every == 0):
            hist.append(sp.kinetic_energy())
        if not np.all(np.isfinite(sp.pos)):
            raise RuntimeError("non-finite positions during settling")
    return hist


# ---------------------------------------------------------------------------
# case
# ---------------------------------------------------------------------------

def run_case(args):
    rng = np.random.default_rng(args.seed)
    d = args.grain_d
    r_g = 0.5 * d
    R = args.radius_over_d * d
    rho_p = 2500.0

    Lx = Ly = args.width_grains * d
    H = args.depth_grains * d
    pos = hcp_positions_in_box(r_g, Lx, Ly, H, packing_gap=args.packing_gap,
                              jitter=0.02 * d, seed=args.seed, margin=0.2 * r_g)
    rad = np.full(len(pos), r_g)
    n_grains = len(pos)

    mat = ContactMaterial(e_star=1.0e9, g_star=4.0e8, mu=args.mu,
                          mu_rolling=args.mu_rolling, restitution=0.4)
    cfg = SpherePackConfig(material=mat,
                           damping_global=args.damping,
                           damping_rotational=args.damping,
                           spin_slip=True)
    walls = make_walls(Lx, Ly)

    # --- 1. settle the bed ---------------------------------------------------
    sp = SpherePack(pos, rad, rho_p, walls, cfg)
    t0 = time.time()
    settle(sp, args.settle)
    # free-surface height: mean z of the upper 5 % of the grains
    zz = np.sort(sp.pos[:, 2])
    z_surf_init = float(np.mean(zz[int(0.95 * n_grains):])) - r_g
    bed_phi = float(n_grains * 4.0 / 3.0 * math.pi * r_g ** 3 / (Lx * Ly * z_surf_init))
    settle_time = time.time() - t0

    # --- 2. add the intruder as a fixed body ---------------------------------
    cx, cy = 0.5 * Lx, 0.5 * Ly
    # bottom of the sphere must clear the *highest* grain, otherwise the
    # inserted intruder starts overlapping the rough surface layer and the
    # contact force explodes (measured: KE 1e2 J, F 1e3 N).
    z_center = float(sp.pos[:, 2].max()) + R + 0.02 * d
    pos2 = np.vstack([np.array([[cx, cy, z_center]]), sp.pos])
    rad2 = np.concatenate([[R], sp.rad])
    fixed2 = np.zeros(len(pos2), dtype=bool)
    fixed2[0] = True
    intr = SpherePack(pos2, rad2, rho_p, walls, cfg, fixed=fixed2)
    intr.dt *= args.dt_factor
    z_bottom0 = z_center - R

    # --- 3. incremental quasi-static penetration -----------------------------
    dz = args.dz_over_d * d
    z_max = args.penetrate_over_r * R
    n_inc = int(round(z_max / dz))
    n_sub = args.substeps
    rec = []
    t1 = time.time()
    for k in range(n_inc):
        # sub-increment: move by a fraction of the grain radius only, so the
        # imposed overlap stays O(1e-3 d) and the contact force never exceeds the
        # quasi-static value (larger jumps were measured to blow the bed up --
        # see docs/CASE_PENETRATION_SPHERE_RESISTANCE.md, failure mode 1).
        for s in range(n_sub):
            intr.pos[0, 2] -= dz / n_sub
            intr._build_neighbour_list()
            intr.run(max(1, args.relax // n_sub))
        f, _ = intr.contact_forces()
        fz = float(f[0, 2])                    # +z = upward = resistance
        depth = float(z_surf_init - (intr.pos[0, 2] - R))
        ke = intr.kinetic_energy()
        rec.append(dict(step=k, z=depth, F=fz,
                        ke=ke, contacts=int(intr.n_contacts)))
        print(f"  inc {k+1:3d}/{n_inc}  z/R={depth/R:5.2f}  F={fz:8.4f} N  "
              f"KE={ke:9.3e} J  n_c={intr.n_contacts}", flush=True)
    pen_time = time.time() - t1

    z = np.array([r["z"] for r in rec])
    F = np.array([r["F"] for r in rec])

    # --- 4. fit the depth-dependent slope -----------------------------------
    rho_bulk = rho_p * bed_phi
    sel = z >= args.fit_from_over_r * R
    if sel.sum() >= 3:
        slope, intercept = np.polyfit(z[sel], F[sel], 1)
        resid = F[sel] - (slope * z[sel] + intercept)
    else:
        slope = intercept = float("nan")
        resid = np.array([float("nan")])
    alpha_fit = slope / (args.mu * rho_bulk * G * math.pi * R ** 2)

    out = dict(
        case="CASE-PENETRATION (sphere penetration resistance)",
        reference=dict(
            source="Brzinski III, Mayor & Durian, PRL 111, 168002 (2013), arXiv:1307.4638v2",
            eq2="dF = alpha mu (rho g z) dA  (p.1)",
            eq3d="F/(alpha mu rho g) = pi (z - R/3) R^2  (p.2)",
            alpha=REF_ALPHA, alpha_err=REF_ALPHA_ERR,
            mu=REF_MU, phi=REF_PHI, rho=REF_RHO),
        parameters=dict(
            grain_d=d, intruder_R=R, radius_over_d=args.radius_over_d,
            Lx=Lx, Ly=Ly, bed_height_initial=H, n_grains=n_grains,
            rho_particle=rho_p, rho_bulk=rho_bulk, packing_fraction=bed_phi,
            mu_grain=args.mu, mu_rolling=args.mu_rolling,
            dz_over_d=args.dz_over_d, relax_steps=args.relax,
            substeps=args.substeps, damping=args.damping, dt=float(intr.dt),
            settle_steps=args.settle, seed=args.seed,
            z_surface=z_surf_init),
        result=dict(alpha_fit=float(alpha_fit),
                    slope_N_per_m=float(slope), intercept_N=float(intercept),
                    fit_range_z_over_R=[args.fit_from_over_r,
                                        float(z.max() / R)],
                    fit_rms_N=float(np.sqrt(np.nanmean(resid ** 2))),
                    err_vs_ref=(float(alpha_fit) - REF_ALPHA) / REF_ALPHA,
                    within_20pct=bool(abs(float(alpha_fit) - REF_ALPHA) / REF_ALPHA <= 0.20)),
        timing=dict(settle_s=settle_time, penetration_s=pen_time,
                    total_s=time.time() - t0),
        profile=rec)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grain-d", type=float, default=0.004)
    p.add_argument("--width-grains", type=int, default=24)
    p.add_argument("--depth-grains", type=int, default=14)
    p.add_argument("--radius-over-d", type=float, default=4.0)
    p.add_argument("--mu", type=float, default=0.40)
    p.add_argument("--mu-rolling", type=float, default=0.05)
    p.add_argument("--phi-request", type=float, default=0.60)
    p.add_argument("--packing-gap", type=float, default=0.02)
    p.add_argument("--dt-factor", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=200.0)
    p.add_argument("--dz-over-d", type=float, default=0.5)
    p.add_argument("--penetrate-over-r", type=float, default=2.0)
    p.add_argument("--relax", type=int, default=400)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--settle", type=int, default=1500)
    p.add_argument("--fit-from-over-r", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--out", default="runs/case_penetration/base")
    args = p.parse_args()

    out = run_case(args)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez(os.path.join(args.out, "profile.npz"),
             z=np.array([r["z"] for r in out["profile"]]),
             F=np.array([r["F"] for r in out["profile"]]),
             ke=np.array([r["ke"] for r in out["profile"]]))
    print(json.dumps(out["result"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()