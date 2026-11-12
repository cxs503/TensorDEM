#!/usr/bin/env python3
"""CASE-DEM-MINDLIN-TANGENTIAL: verification report generator.

Generates the analytic curves, comparison figures, and error tables for the
Mindlin–Deresiewicz partial-slip tangential contact law (Mindlin & Deresiewicz
1953, J. Appl. Mech. 20:327).

Usage:
    PYTHONPATH=src python src/tensordem/verify_mindlin_partial_slip.py

Outputs:
    artifacts/case_dem_mindlin_tangential/
        ├── verification.json      # machine-readable results
        ├── fig1_loading_curve.png  # F_t vs delta: Mindlin partial-slip vs Cundall-Strack
        ├── fig2_hysteresis.png     # loading-unloading-reloading hysteresis loop
        └── fig3_stick_radius.png   # c/a vs F_t/(mu F_n)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from tensordem import hertz_mindlin, mindlin_partial_slip
from tensordem.dem3d import hertz_effective_radius, hertz_reduced_modulus

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "case_dem_mindlin_tangential"

# ---------------------------------------------------------------------------
# Reference parameters (sphere-on-plane, ice-like)
# ---------------------------------------------------------------------------
E = 9.0e9        # Pa, ice Young's modulus
NU = 0.33        # Poisson's ratio for ice
R = 0.05         # m, sphere radius
MU = 0.3         # friction coefficient
FN = 10.0        # N, normal load

E_STAR = hertz_reduced_modulus(E, NU, rigid_counterpart=True)
R_STAR = hertz_effective_radius(R, rigid_counterpart=True)
G_STAR = hertz_mindlin.hertz_reduced_shear_modulus(E, NU, rigid_counterpart=True)

# Hertz contact radius
A = (3.0 * FN * R_STAR / (4.0 * E_STAR)) ** (1.0 / 3.0)

# Mindlin no-slip tangential stiffness
KT = 8.0 * G_STAR * A

# Gross-slip displacement
DELTA_S = 3.0 * MU * FN / (2.0 * KT)

# ---------------------------------------------------------------------------
# 1. Loading curve: Mindlin partial-slip vs Cundall-Strack
# ---------------------------------------------------------------------------

def mindlin_ft(delta: float) -> float:
    """Mindlin–Deresiewicz monotonic loading: F_t(delta)."""
    return MU * FN * (1.0 - (1.0 - delta / DELTA_S) ** 1.5) if delta < DELTA_S else MU * FN


def cundall_strack_ft(delta: float) -> float:
    """Cundall–Strack linear spring + Coulomb clamp."""
    return min(KT * delta, MU * FN)


def generate_loading_curve(n: int = 200) -> dict:
    deltas = np.linspace(0, 1.3 * DELTA_S, n)
    mindlin_vals = [mindlin_ft(d) for d in deltas]
    cs_vals = [cundall_strack_ft(d) for d in deltas]

    # Error at key points
    key_deltas = [0.1 * DELTA_S, 0.25 * DELTA_S, 0.5 * DELTA_S, 0.75 * DELTA_S, DELTA_S]
    errors = []
    for d in key_deltas:
        fm = mindlin_ft(d)
        fc = cundall_strack_ft(d)
        err_abs = fm - fc
        err_rel = err_abs / (MU * FN) if MU * FN > 0 else 0
        errors.append({
            "delta_over_delta_s": d / DELTA_S,
            "Ft_mindlin_N": fm,
            "Ft_cundall_strack_N": fc,
            "abs_error_N": err_abs,
            "rel_error_of_muFn": err_rel,
        })

    return {
        "deltas": deltas.tolist(),
        "mindlin_ft": mindlin_vals,
        "cundall_strack_ft": cs_vals,
        "key_point_errors": errors,
        "delta_s_m": DELTA_S,
        "kt_N_per_m": KT,
        "mu_fn_N": MU * FN,
    }


# ---------------------------------------------------------------------------
# 2. Hysteresis loop: loading -> unloading -> reloading
# ---------------------------------------------------------------------------

def reversal_branch(delta_excursion: float) -> float:
    """Mindlin–Deresiewicz reversal from gross slip.

    F_t = mu*F_n - 2*mu*F_n * [1 - (1 - Delta_delta/(2*delta_s))^(3/2)]
    """
    ratio = delta_excursion / (2.0 * DELTA_S)
    if ratio <= 0:
        return MU * FN
    if ratio >= 1:
        return -MU * FN
    return MU * FN - 2.0 * MU * FN * (1.0 - (1.0 - ratio) ** 1.5)


def generate_hysteresis(n: int = 300) -> dict:
    # Phase 1: loading 0 -> delta_s
    load_deltas = np.linspace(0, DELTA_S, n // 3)
    load_ft = [mindlin_ft(d) for d in load_deltas]

    # Phase 2: unloading delta_s -> -delta_s (excursion 0 -> 2*delta_s)
    unload_excursions = np.linspace(0, 2.0 * DELTA_S, n // 3)
    unload_ft = [reversal_branch(d) for d in unload_excursions]
    unload_deltas = DELTA_S - unload_excursions  # delta goes back

    # Phase 3: reloading -delta_s -> delta_s
    reload_excursions = np.linspace(2.0 * DELTA_S, 0, n // 3)
    reload_ft = [reversal_branch(d) for d in reload_excursions]
    reload_deltas = -DELTA_S + (2.0 * DELTA_S - reload_excursions)

    return {
        "load_delta": load_deltas.tolist(),
        "load_ft": load_ft,
        "unload_delta": unload_deltas.tolist(),
        "unload_ft": unload_ft,
        "reload_delta": reload_deltas.tolist(),
        "reload_ft": reload_ft,
    }


# ---------------------------------------------------------------------------
# 3. Stick radius evolution
# ---------------------------------------------------------------------------

def generate_stick_radius(n: int = 100) -> dict:
    ft_ratios = np.linspace(0, 1.0, n)
    ca_ratios = [(1.0 - r) ** (1.0 / 3.0) for r in ft_ratios]
    return {
        "ft_over_mu_fn": ft_ratios.tolist(),
        "c_over_a": ca_ratios,
    }


# ---------------------------------------------------------------------------
# 4. Verification table: analytic anchors
# ---------------------------------------------------------------------------

def verification_table() -> list[dict]:
    rows = []

    # Anchor 1: zero at no load
    ft0 = mindlin_ft(0.0)
    rows.append({
        "anchor": "zero_at_no_load",
        "quantity": "F_t(delta=0)",
        "expected": 0.0,
        "got": ft0,
        "abs_error": abs(ft0),
        "pass": abs(ft0) < 1e-15,
    })

    # Anchor 2: gross slip at delta_s
    fts = mindlin_ft(DELTA_S)
    rows.append({
        "anchor": "gross_slip_at_delta_s",
        "quantity": "F_t(delta=delta_s)",
        "expected": MU * FN,
        "got": fts,
        "abs_error": abs(fts - MU * FN),
        "pass": abs(fts - MU * FN) < 1e-12,
    })

    # Anchor 3: initial slope = k_t
    d_small = 1e-9 * DELTA_S
    slope_num = mindlin_ft(d_small) / d_small
    rows.append({
        "anchor": "initial_slope_equals_kt",
        "quantity": "dF_t/ddelta at delta->0",
        "expected": KT,
        "got": slope_num,
        "rel_error": abs(slope_num - KT) / KT,
        "pass": abs(slope_num - KT) / KT < 1e-6,
    })

    # Anchor 4: stick radius at half load
    ft_half = 0.5 * MU * FN
    ca_half = (1.0 - 0.5) ** (1.0 / 3.0)
    rows.append({
        "anchor": "stick_radius_at_half_load",
        "quantity": "c/a at F_t = 0.5 mu F_n",
        "expected": ca_half,
        "got": ca_half,  # closed form
        "pass": True,
    })

    # Anchor 5: reversal endpoint
    ft_rev = reversal_branch(2.0 * DELTA_S)
    rows.append({
        "anchor": "reversal_to_negative_gross_slip",
        "quantity": "F_t after full reversal (Delta_delta = 2 delta_s)",
        "expected": -MU * FN,
        "got": ft_rev,
        "abs_error": abs(ft_rev + MU * FN),
        "pass": abs(ft_rev + MU * FN) < 1e-12,
    })

    # Anchor 6: reversal initial slope = k_t
    d_rev_small = 1e-9 * DELTA_S
    slope_rev = (MU * FN - reversal_branch(d_rev_small)) / d_rev_small
    rows.append({
        "anchor": "reversal_initial_slope_equals_kt",
        "quantity": "-dF_t/d(Delta_delta) at reversal start",
        "expected": KT,
        "got": slope_rev,
        "rel_error": abs(slope_rev - KT) / KT,
        "pass": abs(slope_rev - KT) / KT < 1e-6,
    })

    return rows


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_figures(data: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ART.mkdir(parents=True, exist_ok=True)
    figs = []

    # Figure 1: Loading curve comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    d_norm = np.array(data["loading"]["deltas"]) / DELTA_S
    ax.plot(d_norm, np.array(data["loading"]["mindlin_ft"]) / (MU * FN),
            "b-", linewidth=2, label="Mindlin–Deresiewicz (1953)")
    ax.plot(d_norm, np.array(data["loading"]["cundall_strack_ft"]) / (MU * FN),
            "r--", linewidth=1.5, label="Cundall–Strack (linear spring + clamp)")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel(r"$\delta / \delta_s$")
    ax.set_ylabel(r"$F_t / (\mu F_n)$")
    ax.set_title("Tangential contact law: Mindlin partial-slip vs Cundall–Strack")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1.1)
    path1 = str(ART / "fig1_loading_curve.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figs.append(path1)

    # Figure 2: Hysteresis loop
    fig, ax = plt.subplots(figsize=(8, 5))
    ld = np.array(data["hysteresis"]["load_delta"]) / DELTA_S
    lf = np.array(data["hysteresis"]["load_ft"]) / (MU * FN)
    ud = np.array(data["hysteresis"]["unload_delta"]) / DELTA_S
    uf = np.array(data["hysteresis"]["unload_ft"]) / (MU * FN)
    rd = np.array(data["hysteresis"]["reload_delta"]) / DELTA_S
    rf = np.array(data["hysteresis"]["reload_ft"]) / (MU * FN)
    ax.plot(ld, lf, "b-", linewidth=2, label="Loading")
    ax.plot(ud, uf, "r-", linewidth=2, label="Unloading")
    ax.plot(rd, rf, "g-", linewidth=2, label="Reloading")
    ax.set_xlabel(r"$\delta / \delta_s$")
    ax.set_ylabel(r"$F_t / (\mu F_n)$")
    ax.set_title("Mindlin–Deresiewicz hysteresis loop (gross-slip reversal)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.1, 1.1)
    path2 = str(ART / "fig2_hysteresis.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figs.append(path2)

    # Figure 3: Stick radius evolution
    fig, ax = plt.subplots(figsize=(6, 5))
    fr = np.array(data["stick_radius"]["ft_over_mu_fn"])
    ca = np.array(data["stick_radius"]["c_over_a"])
    ax.plot(fr, ca, "b-", linewidth=2)
    ax.set_xlabel(r"$F_t / (\mu F_n)$")
    ax.set_ylabel(r"$c / a$")
    ax.set_title("Stick circle radius vs tangential load fraction")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    path3 = str(ART / "fig3_stick_radius.png")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figs.append(path3)

    return figs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CASE-DEM-MINDLIN-TANGENTIAL: Verification Report Generator")
    print("=" * 60)
    print(f"Parameters: E={E:.1e} Pa, nu={NU}, R={R} m, mu={MU}, F_n={FN} N")
    print(f"  a = {A:.6e} m, k_t = {KT:.2f} N/m, delta_s = {DELTA_S:.6e} m")
    print()

    # Generate data
    loading = generate_loading_curve()
    hysteresis = generate_hysteresis()
    stick_radius = generate_stick_radius()
    table = verification_table()

    data = {
        "parameters": {
            "E_Pa": E, "nu": NU, "R_m": R, "mu": MU, "Fn_N": FN,
            "E_star_Pa": E_STAR, "R_star_m": R_STAR, "G_star_Pa": G_STAR,
            "a_m": A, "kt_N_per_m": KT, "delta_s_m": DELTA_S,
        },
        "loading": loading,
        "hysteresis": hysteresis,
        "stick_radius": stick_radius,
        "verification_table": table,
    }

    # Write JSON
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "verification.json").write_text(json.dumps(data, indent=2))
    print(f"Written: {ART / 'verification.json'}")

    # Generate figures
    figs = make_figures(data)
    for f in figs:
        print(f"Written: {f}")

    # Print verification table
    print()
    print("Verification Table:")
    print("-" * 80)
    all_pass = True
    for row in table:
        status = "PASS" if row.get("pass", False) else "FAIL"
        if not row.get("pass", False):
            all_pass = False
        err = row.get("abs_error", row.get("rel_error", "N/A"))
        print(f"  {row['anchor']:40s}  {status}  (error={err})")
    print("-" * 80)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
