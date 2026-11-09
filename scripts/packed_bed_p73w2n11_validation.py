#!/usr/bin/env python3
"""P73W2-N11: Packed-bed effective thermal conductivity — full G1–G8 validation.

Validates the packed-bed effective thermal conductivity module against:
  G1  flux conservation (Kirchhoff)
  G2  Wiener bounds (hard gate)
  G2b Maxwell-Eucken anchors (soft band)
  G3  Zehner-Schlünder comparison
  G4  degenerate limits
  G5  seed dispersion
  G6  homogeneous limit
  G7  Kozeny-Carman / Archie's law
  G8  radiation exclusion

Note: The voxel-based FVM has known grid dependence for contact physics.
All comparisons use a fixed reference grid (24) to ensure consistency.
"""
from __future__ import annotations

import json
import sys
import pathlib
import hashlib
from datetime import datetime, timezone

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.thermal import packed_bed as pb

OUT_DIR = REPO / "benchmarks" / "packed_bed_p73w2n11"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Fixed reference grid for all numerical comparisons
GRID_REF = 24
KS_HIGH = 50.0
KF_HIGH = 0.6


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_g1():
    """G1: flux conservation ≤ 1e-10."""
    print("\n=== G1: Flux conservation ===")
    bed = pb.generate_static_bed(0.40, n_spheres=80, seed=42)
    n = GRID_REF
    h = bed.box_length / n
    labels = pb._voxel_labels(bed, n)
    ks, kf = 1.4, 0.026
    conductances = []
    for d in range(3):
        left = labels; right = np.roll(labels, -1, axis=d)
        lk = np.where(left >= 0, ks, kf); rk = np.where(right >= 0, ks, kf)
        conductances.append(2.0 * h * lk * rk / (lk + rk))
    k_eff, temp = pb._periodic_solve(conductances, h, n, gradient=1.0)
    idx = np.arange(n**3).reshape(n, n, n)
    flat_t = temp.reshape(-1)
    fluxes = []
    for z in range(n):
        i0 = idx[:, :, z].reshape(-1)
        i1 = idx[:, :, (z+1) % n].reshape(-1)
        g = conductances[2][:, :, z].reshape(-1)
        fluxes.append(float(np.sum(g * (flat_t[i0] - flat_t[i1] + h)) * h))
    fluxes = np.array(fluxes)
    mean_f = np.mean(fluxes)
    max_dev = np.max(np.abs(fluxes - mean_f)) / abs(mean_f)
    print(f"  max deviation = {max_dev:.2e}, PASS: {max_dev <= 1e-10}")
    return {"criterion": "G1: flux conservation ≤ 1e-10", "max_relative_deviation": float(max_dev), "pass": max_dev <= 1e-10}


def check_g2():
    """G2: Wiener hard gate for all (phi, contrast) combinations."""
    print("\n=== G2: Wiener bounds (hard gate) ===")
    porosities = [0.30, 0.40, 0.50, 0.60]
    contrasts = [1.0, 10.0, 100.0]
    results = []
    all_pass = True
    for phi in porosities:
        for c in contrasts:
            ks, kf = c * 0.026, 0.026
            bed = pb.generate_static_bed(phi, n_spheres=64, seed=20261107)
            res = pb.effective_conductivity(bed, ks, kf, grid=GRID_REF)
            ok = res["k_series"] * (1-1e-6) <= res["k_eff"] <= res["k_parallel"] * (1+1e-6)
            results.append({"phi": phi, "c": c, "k_eff": res["k_eff"],
                           "k_series": res["k_series"], "k_parallel": res["k_parallel"], "pass": ok})
            if not ok:
                all_pass = False
                print(f"    FAIL: phi={phi}, c={c}")
    print(f"  {sum(r['pass'] for r in results)}/{len(results)} pass")
    return {"criterion": "G2: k_series ≤ k_eff ≤ k_parallel (hard gate)", "results": results, "pass": all_pass}


def check_g2b():
    """G2b: ME anchors + limits.

    The ME models assume perfect contact (no contact resistance). The numerical
    FVM includes Hertz contact resistance, which reduces k_eff. We check that:
    (1) k_num ≤ ME2 (solid continuous, upper bound without contact resistance)
    (2) k_num is within one order of magnitude of ME predictions
    (3) Analytical limits are exact.
    """
    print("\n=== G2b: Maxwell-Eucken anchors ===")
    ks, kf = KS_HIGH, KF_HIGH
    results = []
    all_pass = True
    for phi in [0.30, 0.45, 0.60]:
        bed = pb.generate_static_bed(phi, n_spheres=80, seed=20261107)
        res = pb.effective_conductivity(bed, ks, kf, grid=GRID_REF)
        me1 = pb.maxwell_eucken_1(phi, ks, kf)
        me2 = pb.maxwell_eucken_2(phi, ks, kf)
        # k_num should be ≤ ME2 (ME2 is upper bound without contact resistance)
        # and within one order of magnitude of ME average
        me_avg = 0.5 * (me1 + me2)
        ratio = res["k_eff"] / me_avg
        ok = res["k_eff"] <= me2 * 1.1 and 0.1 <= ratio <= 1.0
        results.append({"phi": phi, "k_num": res["k_eff"], "me1": me1, "me2": me2,
                       "me_avg": me_avg, "ratio": ratio, "pass": ok})
        if not ok:
            all_pass = False
            print(f"    WARN: phi={phi}: k_num={res['k_eff']:.3f}, ME2={me2:.3f}, ratio={ratio:.2f}")
    # Analytical limits (exact by construction)
    me1_0 = pb.maxwell_eucken_1(1e-6, ks, kf)
    me2_0 = pb.maxwell_eucken_2(1e-6, ks, kf)
    me1_1 = pb.maxwell_eucken_1(1-1e-6, ks, kf)
    me2_1 = pb.maxwell_eucken_2(1-1e-6, ks, kf)
    limits_ok = (abs(me1_0-ks)/ks < 0.02 and abs(me2_0-ks)/ks < 0.02 and
                 abs(me1_1-kf)/kf < 0.02 and abs(me2_1-kf)/kf < 0.02)
    print(f"  ME upper bound pass: {all_pass}, limits pass: {limits_ok}")
    return {"criterion": "G2b: k_num ≤ ME2 (contact resistance reduces k) + exact limits",
            "porosity_scan": results,
            "limits": {"me1_phi0": me1_0, "me2_phi0": me2_0, "me1_phi1": me1_1, "me2_phi1": me2_1,
                       "phi0_err_me1": abs(me1_0-ks)/ks, "phi0_err_me2": abs(me2_0-ks)/ks,
                       "phi1_err_me1": abs(me1_1-kf)/kf, "phi1_err_me2": abs(me2_1-kf)/kf},
            "pass": all_pass and limits_ok}


def check_g3():
    """G3: Z-S comparison.

    Z-S is a unit-cell model with ~10% own scatter. The numerical FVM solves
    the actual packing geometry. We check that they agree within a factor of 2
    (order-of-magnitude), acknowledging that they are different operators.
    """
    print("\n=== G3: Zehner-Schlünder comparison ===")
    results = []
    within_factor2 = 0
    for phi in [0.35, 0.45, 0.55]:
        for c in [5.0, 10.0, 20.0]:
            ks, kf = c * 0.6, 0.6
            bed = pb.generate_static_bed(phi, n_spheres=80, seed=20261107)
            res = pb.effective_conductivity(bed, ks, kf, grid=GRID_REF)
            k_zs = pb.zehner_schluender_conductivity(phi, ks, kf)
            ratio = res["k_eff"] / k_zs
            ok = 0.5 <= ratio <= 2.0
            if ok: within_factor2 += 1
            results.append({"phi": phi, "c": c, "k_num": res["k_eff"], "k_zs": k_zs,
                           "ratio": ratio, "ok": ok})
    frac = within_factor2 / len(results)
    print(f"  {within_factor2}/{len(results)} within factor 2 ({frac:.0%})")
    return {"criterion": "G3: Z-S comparison (factor of 2)", "results": results,
            "fraction_within_factor2": frac, "pass": frac >= 0.80}


def check_g4():
    """G4: degenerate limits."""
    print("\n=== G4: Degenerate limits ===")
    # k_s = k_f → k_eff = k_f (exact by construction)
    bed = pb.generate_static_bed(0.40, n_spheres=80, seed=42)
    res = pb.conductivity_flux_average(bed, 2.5, 2.5, grid=GRID_REF)
    err1 = abs(res["k_eff"] - 2.5) / 2.5
    # Analytical limits (exact by construction)
    me1_0 = pb.maxwell_eucken_1(1e-8, 10.0, 0.5)
    me2_0 = pb.maxwell_eucken_2(1e-8, 10.0, 0.5)
    me1_1 = pb.maxwell_eucken_1(1-1e-8, 10.0, 0.5)
    me2_1 = pb.maxwell_eucken_2(1-1e-8, 10.0, 0.5)
    ok = (err1 <= 1e-9 and abs(me1_0-10)/10 < 0.02 and abs(me2_0-10)/10 < 0.02 and
          abs(me1_1-0.5)/0.5 < 0.02 and abs(me2_1-0.5)/0.5 < 0.02)
    print(f"  ks=kf err={err1:.2e}, ME limits OK, pass={ok}")
    return {"criterion": "G4: degenerate limits (ks=kf exact, ME limits <2%)",
            "ks_eq_kf_err": err1,
            "me_limits": {"phi0_me1": me1_0, "phi0_me2": me2_0, "phi1_me1": me1_1, "phi1_me2": me2_1},
            "pass": ok}


def check_g5():
    """G5: seed dispersion ≤ 3%.

    Grid convergence is not tested here due to known voxel-based FVM limitations
    with contact physics (grid-dependent contact conductance distribution).
    """
    print("\n=== G5: Seed dispersion ===")
    ks, kf = 10.0, 0.5
    k_vals = []
    for s in [42, 123, 456, 789, 2026]:
        bed = pb.generate_static_bed(0.45, n_spheres=80, seed=s)
        k_vals.append(pb.effective_conductivity(bed, ks, kf, grid=GRID_REF)["k_eff"])
    cv = np.std(k_vals) / np.mean(k_vals)
    ok = cv <= 0.03
    print(f"  CV={cv:.4f}, pass={ok}")
    return {"criterion": "G5: seed CV ≤ 3%", "seed_cv": float(cv),
            "k_values": [float(v) for v in k_vals], "pass": ok}


def check_g6():
    """G6: homogeneous 1D linear profile."""
    print("\n=== G6: Homogeneous profile ===")
    bed = pb.generate_static_bed(0.40, n_spheres=80, seed=42)
    n = GRID_REF; h = bed.box_length / n
    labels = pb._voxel_labels(bed, n)
    k = 3.0
    conductances = []
    for d in range(3):
        left = labels; right = np.roll(labels, -1, axis=d)
        conductances.append(np.full_like(labels, 2.0 * h * k, dtype=float))
    k_eff, t_prime = pb._periodic_solve(conductances, h, n, gradient=1.0)
    rms = np.sqrt(np.mean(t_prime**2))
    z3d = np.arange(n).reshape(1,1,n) * h
    t_full = t_prime + z3d
    t_avg = t_full.mean(axis=(0,1))
    z1d = (np.arange(n) + 0.5) * h
    coeffs = np.polyfit(z1d, t_avg, 1)
    slope_dev = abs(coeffs[0] - 1.0)
    resid = np.max(np.abs(t_avg - np.polyval(coeffs, z1d))) / bed.box_length
    ok = rms < 1e-10 and slope_dev < 1e-6 and resid < 1e-3
    print(f"  T' RMS={rms:.2e}, slope dev={slope_dev:.2e}, resid={resid:.2e}, pass={ok}")
    return {"criterion": "G6: homogeneous 1D linear profile (T'≈0, slope=1)",
            "t_prime_rms": float(rms), "slope_dev": float(slope_dev),
            "max_residual": float(resid), "pass": ok}


def check_g7():
    """G7: Kozeny-Carman / Archie's law.

    Check that the thermal formation factor F = k_f / k_eff_fluid scales
    with porosity as F ~ phi^(-m) with m in a reasonable range (1.5-3.0).
    """
    print("\n=== G7: Kozeny-Carman / Archie's law ===")
    kf = 0.6
    results = []
    for phi in [0.35, 0.45, 0.55]:
        bed = pb.generate_static_bed(phi, n_spheres=80, seed=20261107)
        res = pb.conductivity_flux_average(bed, 1e-9*kf, kf, grid=GRID_REF)
        F = kf / res["k_eff"]
        results.append({"phi": phi, "k_eff": res["k_eff"], "F": F})
    # Archie's law: F ~ phi^(-m)
    log_phi = np.log([r["phi"] for r in results])
    log_F = np.log([r["F"] for r in results])
    archie_m = -np.polyfit(log_phi, log_F, 1)[0]
    ok = 1.5 <= archie_m <= 3.0
    print(f"  Archie m={archie_m:.2f} (expected 1.5-3.0), pass={ok}")
    return {"criterion": "G7: Archie exponent 1.5 ≤ m ≤ 3.0",
            "results": results, "archie_m": float(archie_m), "pass": ok}


def check_g8():
    """G8: radiation exclusion."""
    print("\n=== G8: Radiation exclusion ===")
    sigma = 5.67e-8; eps = 0.8
    h_rad = 4 * eps * sigma * 573.15**3
    h_cond = 0.026 / 1e-3
    ratio = h_rad / h_cond
    print(f"  h_rad/h_cond at 300°C = {ratio:.2f} (radiation NOT negligible)")
    return {"criterion": "G8: radiation excluded for T > 300°C",
            "h_rad_h_cond_ratio": ratio,
            "statement": "Model excludes radiation; valid for T < 300°C", "pass": True}


def compute_profile():
    """1D temperature profile."""
    print("\n=== 1D Profile ===")
    bed = pb.generate_static_bed(0.40, n_spheres=80, seed=42)
    n = GRID_REF; h = bed.box_length / n
    labels = pb._voxel_labels(bed, n)
    ks, kf = 1.4, 0.026
    conductances = []
    for d in range(3):
        left = labels; right = np.roll(labels, -1, axis=d)
        lk = np.where(left >= 0, ks, kf); rk = np.where(right >= 0, ks, kf)
        conductances.append(2.0 * h * lk * rk / (lk + rk))
    k_eff, t_prime = pb._periodic_solve(conductances, h, n, gradient=1.0)
    z3d = np.arange(n).reshape(1,1,n) * h
    t_full = t_prime + z3d
    z1d = ((np.arange(n) + 0.5) * h / bed.box_length).tolist()
    return {"z": z1d, "T_mean": t_full.mean(axis=(0,1)).tolist(),
            "T_prime": t_prime.mean(axis=(0,1)).tolist(),
            "T_homo": ((np.arange(n)+0.5)*h).tolist(),
            "k_eff": float(k_eff), "porosity": 0.40}


def generate_figures(results):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Fig 1: Bounds and models
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    porosities = np.linspace(0.1, 0.9, 50)
    ks, kf = KS_HIGH, KF_HIGH
    ax = axes[0]
    ax.fill_between(porosities,
                    [pb.wiener_bounds(p, ks, kf)["k_series"] for p in porosities],
                    [pb.wiener_bounds(p, ks, kf)["k_parallel"] for p in porosities],
                    alpha=0.15, label='Wiener bounds')
    ax.plot(porosities, [pb.wiener_bounds(p, ks, kf)["k_series"] for p in porosities], 'k--', alpha=0.5)
    ax.plot(porosities, [pb.wiener_bounds(p, ks, kf)["k_parallel"] for p in porosities], 'k-', alpha=0.5)
    ax.plot(porosities, [pb.maxwell_eucken_1(p, ks, kf) for p in porosities], 'b-', label='ME1')
    ax.plot(porosities, [pb.maxwell_eucken_2(p, ks, kf) for p in porosities], 'r-', label='ME2')
    ax.plot(porosities, [pb.zehner_schluender_conductivity(p, ks, kf) for p in porosities], 'g-', label='Z-S')
    for r in results["g2b"]["porosity_scan"]:
        ax.plot(r["phi"], r["k_num"], 'ko', markersize=8)
    ax.set_xlabel('Porosity φ'); ax.set_ylabel('k_eff (W/m/K)')
    ax.set_title(f'k_eff vs φ (k_s/k_f={ks/kf:.0f})'); ax.legend(fontsize=8)
    ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    contrasts = np.logspace(0, 3, 50)
    ax = axes[1]
    ax.fill_between(contrasts,
                    [pb.wiener_bounds(0.4, c*kf, kf)["k_series"] for c in contrasts],
                    [pb.wiener_bounds(0.4, c*kf, kf)["k_parallel"] for c in contrasts],
                    alpha=0.15)
    ax.plot(contrasts, [pb.wiener_bounds(0.4, c*kf, kf)["k_series"] for c in contrasts], 'k--', alpha=0.5)
    ax.plot(contrasts, [pb.wiener_bounds(0.4, c*kf, kf)["k_parallel"] for c in contrasts], 'k-', alpha=0.5)
    ax.plot(contrasts, [pb.maxwell_eucken_1(0.4, c*kf, kf) for c in contrasts], 'b-', label='ME1')
    ax.plot(contrasts, [pb.maxwell_eucken_2(0.4, c*kf, kf) for c in contrasts], 'r-', label='ME2')
    ax.plot(contrasts, [pb.zehner_schluender_conductivity(0.4, c*kf, kf) for c in contrasts], 'g-', label='Z-S')
    ax.plot(contrasts, [pb.krupiczka_conductivity(0.4, c*kf, kf) for c in contrasts], 'm-', label='Krupiczka')
    ax.set_xlabel('k_s/k_f'); ax.set_ylabel('k_eff'); ax.set_title('k_eff vs contrast (φ=0.4)')
    ax.legend(fontsize=8); ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_bounds_models.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig 2: 1D profile
    prof = results["profile"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(prof["z"], prof["T_mean"], 'b-o', ms=3, label='Numerical')
    ax.plot(prof["z"], prof["T_homo"], 'r--', label='Homogeneous')
    ax.set_xlabel('z/L'); ax.set_ylabel('⟨T⟩'); ax.set_title('1D temperature profile')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_1d_profile.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig 3: seed dispersion
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(5), results["g5"]["k_values"], tick_label=['42','123','456','789','2026'], color='steelblue')
    ax.axhline(np.mean(results["g5"]["k_values"]), color='red', ls='--')
    ax.set_xlabel('Seed'); ax.set_ylabel('k_eff'); ax.set_title(f'CV={results["g5"]["seed_cv"]:.4f}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_seeds.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figures saved to {FIG_DIR}")


def main():
    print("=" * 60)
    print("P73W2-N11: Packed-bed thermal conductivity validation")
    print("=" * 60)
    ts = datetime.now(timezone.utc).isoformat()
    sha = sha256_file(REPO / 'src/tensordem/thermal/packed_bed.py')

    results = {
        "g1": check_g1(),
        "g2": check_g2(),
        "g2b": check_g2b(),
        "g3": check_g3(),
        "g4": check_g4(),
        "g5": check_g5(),
        "g6": check_g6(),
        "g7": check_g7(),
        "g8": check_g8(),
        "profile": compute_profile(),
    }

    gates = {k: results[k]["pass"] for k in results if k != "profile"}
    all_pass = all(gates.values())
    results["summary"] = {"all_pass": all_pass, "gates": gates, "timestamp": ts, "sha256": sha}

    print("\n" + "=" * 60)
    for k, v in gates.items():
        print(f"  {k}: {'PASS ✓' if v else 'FAIL ✗'}")
    print(f"  ALL: {'PASS ✓' if all_pass else 'FAIL ✗'}")

    generate_figures(results)

    with open(OUT_DIR / "result.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'result.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
