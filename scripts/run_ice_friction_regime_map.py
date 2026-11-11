#!/usr/bin/env python3
"""Entry point for ice friction regime map computation (p381w1-N03).

Generates:
- artifacts/ice_friction_regime_map/acceptance.json  (full validation + criteria)
- artifacts/ice_friction_regime_map/sweep_results.json  (4D sweep data)
- artifacts/ice_friction_regime_map/fig_regime_map.png  (regime classification)
- artifacts/ice_friction_regime_map/fig_mu_surface.png  (friction response surface)
- artifacts/ice_friction_regime_map/fig_validation.png  (validation vs literature)

Run: PYTHONPATH=src python scripts/run_ice_friction_regime_map.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.ice_friction_regime_map import (
    build_acceptance, write_acceptance, sweep_4d, fit_response_surface,
    evaluate_surface, friction_coefficient, RegimeMapInputs,
    validate_kennedy, validate_colbeck, validate_makkonen,
    REGIME_DRY, REGIME_MIXED, REGIME_FULL,
    KENNEDY_1987_ANCHORS, COLBECK_1988_ANCHORS, MAKKONEN_2012_ANCHORS,
)

OUT = ROOT / "artifacts" / "ice_friction_regime_map"
OUT.mkdir(parents=True, exist_ok=True)


def make_fig_regime_map(val: dict) -> str:
    """Figure 1: Regime classification in (T, V) space at P=0.5MPa, Rq=1μm."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    temps = np.linspace(-30, -0.5, 50)
    speeds = np.logspace(-2, 0.7, 50)
    P = 0.5e6
    Rq = 1e-6

    for ax_idx, (pressure_mpa, title) in enumerate([(0.5, "P = 0.5 MPa"),
                                                      (1.0, "P = 1.0 MPa"),
                                                      (2.0, "P = 2.0 MPa")]):
        ax = axes[ax_idx]
        P_pa = pressure_mpa * 1e6
        regime_grid = np.zeros((len(temps), len(speeds)), dtype=int)

        for i, t in enumerate(temps):
            for j, v in enumerate(speeds):
                res = friction_coefficient(t, v, P_pa, Rq)
                regime_grid[i, j] = {REGIME_DRY: 0, REGIME_MIXED: 1, REGIME_FULL: 2}[res["regime"]]

        cmap = plt.matplotlib.colors.ListedColormap(['#d73027', '#fc8d59', '#4575b4'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        T_mesh, V_mesh = np.meshgrid(temps, speeds, indexing='ij')
        ax.pcolormesh(T_mesh, V_mesh, regime_grid, cmap=cmap, norm=norm, alpha=0.8)
        ax.set_yscale('log')
        ax.set_xlabel('Temperature [°C]')
        ax.set_ylabel('Sliding Speed [m/s]')
        ax.set_title(title)

        # Plot Makkonen anchors for this pressure
        for a in MAKKONEN_2012_ANCHORS:
            if abs(a.pressure_mpa - pressure_mpa) < 0.01 and a.roughness_um == 1.0:
                color = {'dry': '#d73027', 'mixed': '#fc8d59', 'full': '#4575b4'}.get(a.observed_regime, 'k')
                ax.plot(a.temperature_c, a.speed_m_s, 'o', color=color, markersize=6,
                        markeredgecolor='k', markeredgewidth=0.5, zorder=5)

    fig.suptitle('Friction Regime Map: μ(T, V) at Rq = 1 μm', fontsize=12, fontweight='bold')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#d73027', label='Dry'),
                       Patch(facecolor='#fc8d59', label='Mixed'),
                       Patch(facecolor='#4575b4', label='Full Lubrication')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    path = OUT / "fig_regime_map.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def make_fig_mu_surface(sweep_result, fit: dict) -> str:
    """Figure 2: Friction coefficient response surface."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    coeffs = fit["coefficients"]

    # (a) mu vs V at different temperatures
    ax = axes[0, 0]
    speeds = np.logspace(-2, 0.7, 50)
    for t, color in [(-30, '#08519c'), (-10, '#2171b5'), (-5, '#6baed6'), (-2, '#bdd7e7')]:
        mus = [evaluate_surface(coeffs, t, v, 0.5, 1.0) for v in speeds]
        ax.semilogx(speeds, mus, '-', color=color, label=f'T = {t}°C')
    ax.set_xlabel('Sliding Speed [m/s]')
    ax.set_ylabel('Friction Coefficient μ')
    ax.set_title('(a) μ vs Speed at P=0.5MPa, Rq=1μm')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) mu vs T at different speeds
    ax = axes[0, 1]
    temps = np.linspace(-30, -0.5, 50)
    for v, color, ls in [(0.01, '#67000d', '-'), (0.1, '#cb181d', '--'), (1.0, '#fb6a4a', '-.')]:
        mus = [evaluate_surface(coeffs, t, v, 0.5, 1.0) for t in temps]
        ax.plot(temps, mus, ls, color=color, label=f'V = {v} m/s')
    ax.set_xlabel('Temperature [°C]')
    ax.set_ylabel('Friction Coefficient μ')
    ax.set_title('(b) μ vs Temperature at P=0.5MPa, Rq=1μm')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) mu vs P at different roughnesses
    ax = axes[1, 0]
    pressures = np.logspace(-1, 1, 30)
    for rq, color in [(0.2, '#00441b'), (1.0, '#006d2c'), (5.0, '#41ab5d')]:
        mus = [evaluate_surface(coeffs, -10, 0.5, p, rq) for p in pressures]
        ax.semilogx(pressures, mus, '-', color=color, label=f'Rq = {rq} μm')
    ax.set_xlabel('Contact Pressure [MPa]')
    ax.set_ylabel('Friction Coefficient μ')
    ax.set_title('(c) μ vs Pressure at T=-10°C, V=0.5m/s')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) mu vs Rq at different speeds
    ax = axes[1, 1]
    roughness = np.logspace(-1, 1, 30)
    for v, color in [(0.01, '#4d004b'), (0.1, '#810f7c'), (1.0, '#bf5eb3')]:
        mus = [evaluate_surface(coeffs, -10, v, 0.5, rq) for rq in roughness]
        ax.semilogx(roughness, mus, '-', color=color, label=f'V = {v} m/s')
    ax.set_xlabel('Roughness Rq [μm]')
    ax.set_ylabel('Friction Coefficient μ')
    ax.set_title('(d) μ vs Roughness at T=-10°C, P=0.5MPa')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f'Ice-Steel Friction Response Surface (R² = {fit["r_squared"]:.3f})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = OUT / "fig_mu_surface.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def make_fig_validation(val: dict) -> str:
    """Figure 3: Validation against literature anchors."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) Kennedy: dry friction vs temperature
    ax = axes[0]
    kennedy = val["kennedy"]
    temps = [r["temperature_c"] for r in kennedy["rows"]]
    mu_obs = [r["mu_kennedy"] for r in kennedy["rows"]]
    mu_pred = [r["mu_predicted"] for r in kennedy["rows"]]
    mu_err = [r.get("mu_unc", 0.02) for r in kennedy["rows"]]
    ax.errorbar(temps, mu_obs, yerr=mu_err, fmt='o', color='#d73027', capsize=3,
                label='Kennedy 1987 (observed)')
    ax.plot(temps, mu_pred, 's', color='#2171b5', markersize=6, label='Model')
    ax.set_xlabel('Temperature [°C]')
    ax.set_ylabel('Friction Coefficient μ')
    ax.set_title(f'(a) Dry Friction (Kennedy 1987)\nMax err: {kennedy["max_rel_error"]:.1%}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Colbeck: lubricated friction vs speed
    ax = axes[1]
    colbeck = val["colbeck"]
    speeds = [r["speed_m_s"] for r in colbeck["rows"]]
    mu_obs_c = [r["mu_colbeck"] for r in colbeck["rows"]]
    mu_pred_c = [r["mu_predicted"] for r in colbeck["rows"]]
    ax.loglog(speeds, mu_obs_c, 'o', color='#d73027', markersize=8, label='Colbeck 1988')
    ax.loglog(speeds, mu_pred_c, 's', color='#2171b5', markersize=6, label='Model')
    ax.set_xlabel('Sliding Speed [m/s]')
    ax.set_ylabel('Friction Coefficient μ')
    ax.set_title(f'(b) Lubricated Friction (Colbeck 1988)\nMax err: {colbeck["max_rel_error"]:.1%}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    # (c) Makkonen: regime classification accuracy
    ax = axes[2]
    makkonen = val["makkonen"]
    n_correct = makkonen["n_correct"]
    n_total = makkonen["n_anchors"]
    accuracy = makkonen["accuracy"]

    # Bar chart of regime predictions
    regimes = ['Dry', 'Mixed', 'Full']
    correct_by_regime = [0, 0, 0]
    total_by_regime = [0, 0, 0]
    for r in makkonen["rows"]:
        idx = {'dry': 0, 'mixed': 1, 'full_lubrication': 2}[r["observed_regime"]]
        total_by_regime[idx] += 1
        if r["correct"]:
            correct_by_regime[idx] += 1

    x = np.arange(len(regimes))
    width = 0.35
    ax.bar(x - width/2, total_by_regime, width, label='Total', color='#969696')
    ax.bar(x + width/2, correct_by_regime, width, label='Correct', color='#4575b4')
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel('Number of Anchors')
    ax.set_title(f'(c) Regime Classification (Makkonen 2012)\nAccuracy: {accuracy:.1%} ({n_correct}/{n_total})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle('Validation Against Literature Anchors', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = OUT / "fig_validation.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def main():
    print("=" * 60)
    print("Ice Friction Regime Map - Full Computation")
    print("Campaign node: p381w1-N03")
    print("=" * 60)

    # 1. Build acceptance (runs all validations)
    print("\n[1/4] Running validation suite...")
    acceptance = build_acceptance()
    print(f"  All criteria pass: {acceptance['all_pass']}")
    for k, v in acceptance["criteria"].items():
        status = "✓ PASS" if v["pass"] else "✗ FAIL"
        print(f"  {status}: {k}")

    # 2. Write acceptance JSON
    print("\n[2/4] Writing acceptance artifact...")
    acc_path = write_acceptance(str(OUT / "acceptance.json"))
    print(f"  Written: {OUT / 'acceptance.json'}")

    # 3. Run 4D sweep and save results
    print("\n[3/4] Running 4D parameter sweep...")
    sweep_result = sweep_4d()
    fit = fit_response_surface(sweep_result)
    print(f"  Sweep points: {len(sweep_result.flat_data)}")
    print(f"  R² = {fit['r_squared']:.4f}")

    sweep_data = {
        "n_points": len(sweep_result.flat_data),
        "r_squared": fit["r_squared"],
        "rmse_log": fit["rmse_log"],
        "ranges": {
            "temperatures_c": sweep_result.temperatures_c.tolist(),
            "speeds_m_s": sweep_result.speeds_m_s.tolist(),
            "pressures_mpa": sweep_result.pressures_mpa.tolist(),
            "roughness_um": sweep_result.roughness_um.tolist(),
        },
        "statistics": {
            "mu_min": float(np.min(sweep_result.mu_grid)),
            "mu_max": float(np.max(sweep_result.mu_grid)),
            "mu_mean": float(np.mean(sweep_result.mu_grid)),
            "regime_counts": {
                "dry": int(np.sum(sweep_result.regime_grid == 0)),
                "mixed": int(np.sum(sweep_result.regime_grid == 1)),
                "full": int(np.sum(sweep_result.regime_grid == 2)),
            }
        }
    }
    (OUT / "sweep_results.json").write_text(json.dumps(sweep_data, indent=2) + "\n")
    print(f"  Written: {OUT / 'sweep_results.json'}")

    # 4. Generate figures
    print("\n[4/4] Generating figures...")
    val = acceptance["validation"]
    fig1 = make_fig_regime_map(val)
    print(f"  Figure 1: {fig1}")
    fig2 = make_fig_mu_surface(sweep_result, fit)
    print(f"  Figure 2: {fig2}")
    fig3 = make_fig_validation(val)
    print(f"  Figure 3: {fig3}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"All criteria pass: {acceptance['all_pass']}")
    print(f"Kennedy (dry): max error = {acceptance['criteria']['a_kennedy_dry_friction']['max_rel_error']:.1%}")
    print(f"Colbeck (lubricated): max error = {acceptance['criteria']['b_colbeck_lubricated']['max_rel_error']:.1%}")
    print(f"Makkonen (regimes): accuracy = {acceptance['criteria']['c_makkonen_regime_transitions']['accuracy']:.1%}")
    print(f"Response surface: R² = {acceptance['criteria']['d_response_surface_r_squared']['r_squared']:.4f}")
    print(f"Sweep points: {acceptance['n_sweep_points']}")
    print(f"Artifacts: {OUT}")
    print("=" * 60)

    return 0 if acceptance["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
