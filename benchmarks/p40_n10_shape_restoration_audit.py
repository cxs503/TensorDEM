#!/usr/bin/env python3
"""P40-N10 shape-representation restoration audit (machine-readable evidence sheet).

Re-measures, from the restored public API only, every number quoted in
``docs/SHAPE_REPRESENTATION_MODULE_RESTORATION.md``:

* stage A -- elliptical Hertz: closed form vs. the independent Boussinesq-quadrature
  reference, and the isotropic equivalent-radius error (the node's finding);
* stage B -- inscribed clump convergence in N (two-sided surface error, fill);
* stage C -- 2-D pile: energy stability (peak KE / released PE), static
  pseudo-dissipation and the repose estimator against a synthetic 32 deg pile;
* stage D -- the import-contract scan summary (verification-debt account).

Nothing is hand-copied: both JSON and the figure are produced by this script.

Run:  PYTHONPATH=src python3 benchmarks/p40_n10_shape_restoration_audit.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from tensordem.shape_representation import (
    PileConfig2D,
    Superquadric,
    _Pile,
    clump_surface_deviation,
    complete_elliptic_KE,
    elliptic_hertz,
    elliptic_hertz_quadrature,
    equivalent_radius_hertz,
    fit_inscribed_clump,
    legendre_relation,
    repose_angle_from_pile,
    run_pile,
)

OUT = Path("artifacts/p40_n10_restoration")
E_STAR = 1.0e7
FORCE = 1.0
A_FIX = 1.0 / 0.05


def stage_hertz() -> dict:
    rows = []
    for ratio in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0):
        a = A_FIX
        b = A_FIX / ratio
        cf = elliptic_hertz(a, b, E_STAR, FORCE)
        q = elliptic_hertz_quadrature(a, b, E_STAR, FORCE)
        eq = equivalent_radius_hertz(a, b, E_STAR, FORCE)
        rows.append({
            "curvature_ratio": ratio,
            "delta_closed": cf.delta, "delta_quad": q.delta,
            "stiffness_closed": cf.stiffness, "stiffness_quad": q.stiffness,
            "stiffness_err_pct": 100.0 * (cf.stiffness / q.stiffness - 1.0),
            "delta_err_pct": 100.0 * (cf.delta / q.delta - 1.0),
            "a_tight_err_pct": 100.0 * (cf.a_tight / q.a_tight - 1.0),
            "equiv_radius_err_pct": 100.0 * (eq.stiffness / q.stiffness - 1.0),
            "a_flat_over_a_tight": cf.a_flat / cf.a_tight,
        })
    r = 0.05
    delta_ana = (9.0 * FORCE ** 2 / (16.0 * E_STAR ** 2 * r)) ** (1.0 / 3.0)
    stiff_ana = 2.0 * E_STAR * math.sqrt(r * delta_ana)
    circ = elliptic_hertz(A_FIX, A_FIX, E_STAR, FORCE)
    circ_q = elliptic_hertz_quadrature(A_FIX, A_FIX, E_STAR, FORCE)
    k_ana = math.pi / 2
    return {
        "stage": "hertz",
        "E_star_pa": E_STAR, "force_n": FORCE, "A_fix_inv_m": A_FIX,
        "rows": rows,
        "circular_limit": {
            "delta_analytic": delta_ana,
            "delta_err_closed_pct": 100.0 * (circ.delta / delta_ana - 1.0),
            "delta_err_quadrature_pct": 100.0 * (circ_q.delta / delta_ana - 1.0),
            "stiffness_err_closed_pct": 100.0 * (circ.stiffness / stiff_ana - 1.0),
            "stiffness_err_quadrature_pct": 100.0 * (circ_q.stiffness / stiff_ana - 1.0),
        },
        "elliptic_integrals": {
            "K_0.9": complete_elliptic_KE(0.9)[0],
            "E_0.9": complete_elliptic_KE(0.9)[1],
            "E_0.9_reference_AS_17.1": 1.1716970527816143,
            "legendre_residual_max": max(abs(legendre_relation(k) - math.pi / 2)
                                         for k in (0.3, 0.7, 0.9, 0.99)),
        },
        "max_abs_stiffness_err_pct": max(abs(x["stiffness_err_pct"]) for x in rows),
        "max_abs_equiv_err_pct": max(abs(x["equiv_radius_err_pct"]) for x in rows),
    }


def stage_clump() -> dict:
    shapes = {
        "spheroid_1.3": Superquadric(a=0.0194, b=0.0194, c=0.0252),
        "superquadric_n3": Superquadric(a=0.022, b=0.018, c=0.016, n1=3.0, n2=3.0),
    }
    rows = []
    for name, sq in shapes.items():
        for n in (8, 32, 128):
            centres, radii = fit_inscribed_clump(sq, n)
            dev = clump_surface_deviation(sq, centres, radii, n_samples=2000)
            rows.append({"shape": name, "n_spheres": n, **dev.as_dict(),
                         "radius_mean_m": float(np.mean(radii))})
    return {"stage": "clump", "rows": rows}


def stage_pile() -> dict:
    cfg = PileConfig2D(channel="sphere", n_particles=60, radius=0.02, dt=1.2e-4,
                       n_steps=1200, n_settle=900, mu_slide=0.55, mu_roll=0.06)
    pile = _Pile(cfg)
    ke0, pe0 = pile.energy()
    released = ke0 + pe0
    peak = 0.0
    for _ in range(cfg.n_steps):
        pile.step()
        peak = max(peak, pile.energy()[0])
    ke, pe = pile.energy()

    cfg_static = PileConfig2D(channel="sphere", n_particles=60, radius=0.02, dt=1.2e-4,
                              n_steps=1600, n_settle=1200, viscous_damp=0.97)
    static = _Pile(cfg_static)
    for _ in range(1200):
        static.step()
    prev = sum(static.energy())
    ratios = []
    for _ in range(400):
        static.step()
        e = sum(static.energy())
        ratios.append(abs(e - prev) / max(abs(e), 1e-30))
        prev = e

    angle = math.radians(32.0)
    rng = np.random.default_rng(5)
    pts = []
    for _ in range(400):
        x = rng.uniform(-0.15, 0.15)
        h = 0.18 - abs(x) * math.tan(angle)
        pts.append((x, rng.uniform(0.0, max(h, 0.01))))
    phi, spread = repose_angle_from_pile(np.array(pts), 0.02)

    study = run_pile(PileConfig2D(channel="sphere", n_particles=120, radius=0.02,
                                  dt=2.0e-4, n_steps=40000, n_settle=30000,
                                  gate_half=0.25, box_half=0.50,
                                  viscous_damp=0.99, record_every=10000))
    return {
        "stage": "pile",
        "energy_stability": {
            "released_energy_j": released,
            "peak_kinetic_over_released": peak / released,
            "final_kinetic_over_released": ke / released,
            "final_kinetic_j": ke, "final_potential_j": pe,
            "min_height_m": float(pile.pos[:, 1].min()),
        },
        "static_pseudo_dissipation_per_step": float(np.mean(ratios)),
        "repose_validation": {"synthetic_deg": 32.0, "recovered_deg": phi,
                              "error_deg": phi - 32.0, "spread_deg": spread},
        "study_run": {
            "n_particles": study.n_particles, "repose_deg": study.repose_deg,
            "repose_deg_std": study.repose_deg_std,
            "pseudo_dissipation": study.pseudo_dissipation,
            "contacts_final": study.contacts_final,
            "mass_kg": study.mass_kg,
        },
    }


def stage_imports() -> dict:
    sys.path.insert(0, "tools")
    from import_contract_scan import scan  # noqa: PLC0415

    broken = scan(Path("."))
    return {"stage": "import_contract",
            "n_broken_symbols": len(broken),
            "affected_test_modules": sorted({r["test"] for r in broken}),
            "packages": sorted({r["module"] for r in broken})}


def figure(hertz: dict, clump: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    rows = hertz["rows"]
    ratios = [r["curvature_ratio"] for r in rows]
    ax[0].plot(ratios, [abs(r["stiffness_err_pct"]) + 1e-14 for r in rows], "o-",
               label="closed form vs quadrature")
    ax[0].plot(ratios, [abs(r["equiv_radius_err_pct"]) for r in rows], "s--",
               label="equivalent-radius shortcut")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].axhline(2.0, color="r", lw=0.8, ls=":", label="acceptance 2 %")
    ax[0].set_xlabel("curvature ratio A/B"); ax[0].set_ylabel("stiffness error [%]")
    ax[0].legend(fontsize=8); ax[0].set_title("(a) elliptic Hertz anchors")

    for shape, marker in (("spheroid_1.3", "o-"), ("superquadric_n3", "s--")):
        sub = [r for r in clump["rows"] if r["shape"] == shape]
        ax[1].plot([r["n_spheres"] for r in sub], [r["rel_max"] for r in sub], marker,
                   label=shape)
    ax[1].set_xscale("log"); ax[1].set_xlabel("N spheres")
    ax[1].set_ylabel("two-sided rel. error")
    ax[1].legend(fontsize=8); ax[1].set_title("(b) inscribed clump convergence")

    sub = [r for r in clump["rows"] if r["shape"] == "spheroid_1.3"]
    ax[2].plot([r["n_spheres"] for r in sub], [r["volume_ratio"] for r in sub], "o-")
    ax[2].axhline(1.0, color="r", lw=0.8, ls=":")
    ax[2].set_xscale("log"); ax[2].set_xlabel("N spheres")
    ax[2].set_ylabel("sum of ball volumes / body volume")
    ax[2].set_title("(c) fill (overlap-inflated upper bound)")
    fig.tight_layout()
    png = OUT / "p40_n10_restoration.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return str(png)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hertz = stage_hertz()
    clump = stage_clump()
    pile = stage_pile()
    imports = stage_imports()
    png = figure(hertz, clump)
    audit = {"hertz": hertz, "clump": clump, "pile": pile, "import_contract": imports,
             "figure": png}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps({k: audit[k] for k in ("hertz", "pile", "import_contract")},
                     indent=1)[:3000])
    print("figure:", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())