#!/usr/bin/env python3
"""Round 28: Comprehensive validation of CASE-DEM-PLANE-SHEAR-MUIDI.

This script consolidates all archived data from rounds 1-27 and produces:
1. A comprehensive mu(I) and phi(I) comparison with the reference law
2. Publication-quality figures showing all estimators and their spread
3. An honest assessment of the current maturity level
4. Identification of what's needed to close the case

The key finding from rounds 1-27:
- 3D servo shear: mu_wall is 18-41% below reference (da Cruz 2005 / GDR MiDi 2004)
- 2D plane shear: mu is 60-72% below the 3D reference (expected for dimensionality)
- Full-cell virial estimator captures 93% of applied stress (momentum balance)
- Rolling friction increases macroscopic mu but still doesn't reach reference
- Stationarity issues at some I values (mu still drifting after 1 strain)

This round does NOT run new simulations (too slow for single session).
Instead, it processes all archived data and produces the final validation report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks" / "plane_shear_mu_i"
sys.path.insert(0, str(BENCH))

import reference_law as rl

OUT = REPO / "outputs" / "r28"
OUT.mkdir(parents=True, exist_ok=True)
FIG = REPO / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def load_bulk_rheometry():
    """Load round 6 bulk rheometry data (3D servo shear)."""
    path = BENCH / "runs" / "bulk_rheometry" / "result.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_analysis():
    """Load round 1-3 analysis data."""
    path = BENCH / "runs" / "analysis.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_2d_probe():
    """Load round 24-25 2D probe data."""
    results = []
    d = BENCH / "runs" / "plane_shear_2d"
    for p in sorted(d.glob("probe_I*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results

def fig_mu_i_comparison(data_3d, data_2d):
    """Figure 1: mu(I) comparison - all estimators vs reference."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Reference law
    I_ref = np.logspace(-3, -0.5, 50)
    mu_ref = [rl.ref_mu(I) for I in I_ref]
    
    # Left panel: 3D servo shear
    ax = axes[0]
    ax.plot(I_ref, mu_ref, 'k-', lw=2, label='Reference (da Cruz 2005)')
    
    if data_3d:
        I_wall = [r["I_measured"] for r in data_3d["rows"]]
        mu_wall = [r["mu_wall"] for r in data_3d["rows"]]
        mu_bulk = [r["mu_bulk"] for r in data_3d["rows"]]
        
        ax.plot(I_wall, mu_wall, 'ro-', ms=8, lw=2, label='mu_wall (wall traction)')
        ax.plot(I_wall, mu_bulk, 'bs-', ms=8, lw=2, label='mu_bulk (slab virial)')
        
        # Add error bars
        for i, (I, mu_w, mu_b) in enumerate(zip(I_wall, mu_wall, mu_bulk)):
            mu_ref_i = rl.ref_mu(I)
            ax.plot([I, I], [mu_ref_i, mu_w], 'r-', alpha=0.3, lw=1)
            ax.plot([I, I], [mu_ref_i, mu_b], 'b-', alpha=0.3, lw=1)
    
    ax.set_xscale('log')
    ax.set_xlabel('Inertial number I')
    ax.set_ylabel('Effective friction coefficient mu(I)')
    ax.set_title('3D Servo Shear Cell (Round 6)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-3, 0.5)
    ax.set_ylim(0.2, 0.7)
    
    # Add text box with key stats
    if data_3d:
        errors = [abs(r["mu_wall_err_pct"]) for r in data_3d["rows"]]
        text = f'Max |error|: {max(errors):.1f}%\nMean |error|: {np.mean(errors):.1f}%\nTolerance: 10%'
        ax.text(0.02, 0.98, text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right panel: 2D plane shear
    ax = axes[1]
    ax.plot(I_ref, mu_ref, 'k-', lw=2, label='Reference (3D, da Cruz 2005)')
    
    if data_2d:
        I_2d = [r["case"]["I_target"] for r in data_2d]
        mu_2d = [r["mu_wall"] for r in data_2d]
        ax.plot(I_2d, mu_2d, 'g^-', ms=8, lw=2, label='mu_wall (2D disks)')
        
        # 2D-specific reference (from round 24 doc)
        I_2d_ref = np.linspace(0.02, 0.25, 20)
        mu_2d_ref = 0.22 + 1.00 * I_2d_ref
        ax.plot(I_2d_ref, mu_2d_ref, 'g--', lw=1.5, alpha=0.5, label='2D reference (linear fit)')
    
    ax.set_xscale('log')
    ax.set_xlabel('Inertial number I')
    ax.set_ylabel('Effective friction coefficient mu(I)')
    ax.set_title('2D Plane Shear (Rounds 24-25)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-2, 0.5)
    ax.set_ylim(0.0, 0.5)
    
    plt.tight_layout()
    return fig

def fig_phi_i_comparison(data_3d):
    """Figure 2: phi(I) comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Reference law
    I_ref = np.logspace(-3, -0.5, 50)
    phi_ref = [rl.ref_phi(I) for I in I_ref]
    
    ax.plot(I_ref, phi_ref, 'k-', lw=2, label='Reference (da Cruz 2005)')
    
    if data_3d:
        I_meas = [r["I_measured"] for r in data_3d["rows"]]
        phi_bed = [r["phi_bed_final"] for r in data_3d["rows"]]
        phi_slab = [r.get("phi_slab", float('nan')) for r in data_3d["rows"]]
        
        ax.plot(I_meas, phi_bed, 'ro-', ms=8, lw=2, label='phi_bed (bed-averaged)')
        ax.plot(I_meas, phi_slab, 'bs-', ms=8, lw=2, label='phi_slab (slab)')
        
        # Add error bars
        for I, phi in zip(I_meas, phi_bed):
            phi_ref_i = rl.ref_phi(I)
            ax.plot([I, I], [phi_ref_i, phi], 'r-', alpha=0.3, lw=1)
    
    ax.set_xscale('log')
    ax.set_xlabel('Inertial number I')
    ax.set_ylabel('Solid fraction phi(I)')
    ax.set_title('Volume Fraction vs Inertial Number')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-3, 0.5)
    ax.set_ylim(0.45, 0.70)
    
    # Add text box
    if data_3d:
        errors = [abs(r["phi_err_pct"]) for r in data_3d["rows"]]
        text = f'Max |error|: {max(errors):.1f}%\nMean |error|: {np.mean(errors):.1f}%\nTolerance: 10%\n\nNote: phi_ref provenance\nis UNVERIFIED (see docs)'
        ax.text(0.02, 0.98, text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    return fig

def fig_estimator_spread(data_3d):
    """Figure 3: Spread of mu over different estimators."""
    if not data_3d:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    I_targets = [r["I_target"] for r in data_3d["rows"]]
    mu_wall = [r["mu_wall"] for r in data_3d["rows"]]
    mu_bulk = [r["mu_bulk"] for r in data_3d["rows"]]
    mu_ref = [rl.ref_mu(r["I_measured"]) for r in data_3d["rows"]]
    
    x = np.arange(len(I_targets))
    width = 0.25
    
    ax.bar(x - width, mu_wall, width, label='mu_wall', color='red', alpha=0.7)
    ax.bar(x, mu_bulk, width, label='mu_bulk', color='blue', alpha=0.7)
    ax.bar(x + width, mu_ref, width, label='mu_ref', color='black', alpha=0.7)
    
    ax.set_xlabel('Target inertial number I')
    ax.set_ylabel('Effective friction coefficient mu')
    ax.set_title('Estimator Spread: Wall Traction vs Bulk Virial vs Reference')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{I:.3f}' for I in I_targets])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add pressure ratio annotation
    p_ratios = [r["p_bulk"] / r["p_wall"] for r in data_3d["rows"]]
    text = 'p_bulk/p_wall ratios:\n' + '\n'.join([f'I={I:.3f}: {r:.3f}' for I, r in zip(I_targets, p_ratios)])
    ax.text(0.98, 0.98, text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig

def fig_momentum_balance(data_3d):
    """Figure 4: Momentum balance check (full-cell virial)."""
    if not data_3d:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # This data is not in the archived result.json (it was added later)
    # So we'll create a schematic showing what the check should be
    I_targets = [r["I_target"] for r in data_3d["rows"]]
    p_bulk = [r["p_bulk"] for r in data_3d["rows"]]
    p_wall = [r["p_wall"] for r in data_3d["rows"]]
    p_applied = 62500.0  # from the case setup
    
    ax.plot(I_targets, [p/p_applied for p in p_wall], 'ro-', ms=8, lw=2, 
            label='p_wall/P_applied (should be ~1.0)')
    ax.plot(I_targets, [p/p_applied for p in p_bulk], 'bs-', ms=8, lw=2,
            label='p_bulk/P_applied (slab virial)')
    ax.axhline(1.0, color='k', ls='--', lw=1, alpha=0.5, label='Ideal')
    
    ax.set_xlabel('Inertial number I')
    ax.set_ylabel('Pressure ratio')
    ax.set_title('Pressure Estimator Validation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text
    text = 'Round 27 finding:\nfull_cell virial captures\n93% of applied stress\n(sigma_zz/P = -0.93)\n\nSlab virial captures\nonly 72-82% (boundary\nlayer effect)'
    ax.text(0.02, 0.98, text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    return fig

def generate_tables(data_3d, data_2d):
    """Generate markdown tables."""
    lines = []
    lines.append("# CASE-DEM-PLANE-SHEAR-MUIDI Round 28: Comprehensive Validation\n")
    lines.append("## Table 1: 3D Servo Shear Cell Results (Round 6)\n")
    lines.append("| I_target | I_measured | mu_wall | mu_bulk | mu_ref | mu_wall_err% | mu_bulk_err% | phi_bed | phi_ref | phi_err% | p_bulk/p_wall |")
    lines.append("|----------|------------|---------|---------|--------|--------------|--------------|---------|---------|----------|---------------|")
    
    if data_3d:
        for r in data_3d["rows"]:
            lines.append(f"| {r['I_target']:.3f} | {r['I_measured']:.4f} | {r['mu_wall']:.4f} | {r['mu_bulk']:.4f} | {rl.ref_mu(r['I_measured']):.4f} | {r['mu_wall_err_pct']:.1f} | {r['mu_bulk_err_pct']:.1f} | {r['phi_bed_final']:.4f} | {rl.ref_phi(r['I_measured']):.4f} | {r['phi_err_pct']:.1f} | {r['p_bulk']/r['p_wall']:.3f} |")
    
    lines.append("\n## Table 2: 2D Plane Shear Results (Rounds 24-25)\n")
    lines.append("| I_target | I_measured | mu_wall | mu_ref(3D) | mu_err% | phi_gap | phi_ref(2D) | phi_err% |")
    lines.append("|----------|------------|---------|------------|---------|---------|-------------|----------|")
    
    if data_2d:
        for r in data_2d:
            I = r["case"]["I_target"]
            mu = r["mu_wall"]
            # 2D reference from round 24: mu = 0.22 + 1.00*I, phi = 0.81 - 0.30*I
            mu_ref_2d = 0.22 + 1.00 * r.get("inertial_number", I)
            phi_ref_2d = 0.81 - 0.30 * r.get("inertial_number", I)
            phi_gap = r.get("phi_gap", float('nan'))
            mu_err = 100 * (mu - mu_ref_2d) / mu_ref_2d if mu_ref_2d > 0 else float('nan')
            phi_err = 100 * (phi_gap - phi_ref_2d) / phi_ref_2d if phi_ref_2d > 0 else float('nan')
            lines.append(f"| {I:.3f} | {r.get('inertial_number', float('nan')):.4f} | {mu:.4f} | {rl.ref_mu(I):.4f} | {mu_err:.1f} | {phi_gap:.4f} | {phi_ref_2d:.4f} | {phi_err:.1f} |")
    
    return '\n'.join(lines)

def main():
    print("Loading archived data...")
    data_3d = load_bulk_rheometry()
    data_1 = load_analysis()
    data_2d = load_2d_probe()
    
    print("Generating figures...")
    fig1 = fig_mu_i_comparison(data_3d, data_2d)
    fig1.savefig(FIG / "r28_mu_i_comparison.png", dpi=150, bbox_inches='tight')
    print(f"  Saved {FIG / 'r28_mu_i_comparison.png'}")
    
    fig2 = fig_phi_i_comparison(data_3d)
    fig2.savefig(FIG / "r28_phi_i_comparison.png", dpi=150, bbox_inches='tight')
    print(f"  Saved {FIG / 'r28_phi_i_comparison.png'}")
    
    fig3 = fig_estimator_spread(data_3d)
    if fig3:
        fig3.savefig(FIG / "r28_estimator_spread.png", dpi=150, bbox_inches='tight')
        print(f"  Saved {FIG / 'r28_estimator_spread.png'}")
    
    fig4 = fig_momentum_balance(data_3d)
    if fig4:
        fig4.savefig(FIG / "r28_momentum_balance.png", dpi=150, bbox_inches='tight')
        print(f"  Saved {FIG / 'r28_momentum_balance.png'}")
    
    print("Generating tables...")
    tables = generate_tables(data_3d, data_2d)
    with open(OUT / "r28_validation_tables.md", 'w') as f:
        f.write(tables)
    print(f"  Saved {OUT / 'r28_validation_tables.md'}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    if data_3d:
        mu_wall_errors = [abs(r["mu_wall_err_pct"]) for r in data_3d["rows"]]
        mu_bulk_errors = [abs(r["mu_bulk_err_pct"]) for r in data_3d["rows"]]
        phi_errors = [abs(r["phi_err_pct"]) for r in data_3d["rows"]]
        
        print("\n3D Servo Shear (Round 6):")
        print(f"  mu_wall: max |error| = {max(mu_wall_errors):.1f}%, mean = {np.mean(mu_wall_errors):.1f}%")
        print(f"  mu_bulk: max |error| = {max(mu_bulk_errors):.1f}%, mean = {np.mean(mu_bulk_errors):.1f}%")
        print(f"  phi_bed: max |error| = {max(phi_errors):.1f}%, mean = {np.mean(phi_errors):.1f}%")
        print(f"  Tolerance: 10%")
        print(f"  Verdict: FAIL (mu_wall), FAIL (mu_bulk), {'PASS' if max(phi_errors) < 10 else 'FAIL'} (phi)")
    
    if data_2d:
        print("\n2D Plane Shear (Rounds 24-25):")
        print("  mu values are 60-72% below 3D reference (expected for dimensionality)")
        print("  2D-specific reference law needed for proper validation")
    
    print("\n" + "="*70)
    print("MATURITY ASSESSMENT")
    print("="*70)
    print("""
Current status: WITHHELD (not PASS, not FAIL - evidence insufficient)

Key findings:
1. The 3D device gives mu systematically 18-41% below the da Cruz/GDR MiDi reference
2. The full-cell virial estimator captures 93% of applied stress (good momentum balance)
3. The slab virial estimator captures only 72-82% (boundary layer effect)
4. Rolling friction increases macroscopic mu but doesn't reach reference
5. Stationarity issues at some I values (mu still drifting after 1 strain)
6. The reference law parameters (mu_s=0.36, mu_2=0.84, I_0=0.30) are QUOTED but not
   archived from the original papers - provenance is UNVERIFIED

What's needed to close the case:
1. Longer shear strains (5-10 strain instead of 0.2) to ensure steady state
2. Larger system size (1000+ particles instead of 294) to reduce finite-size effects
3. Parameter calibration sweep (mu_contact, mu_rolling) to match reference
4. Archive the actual tables from da Cruz 2005 / GDR MiDi 2004 to verify reference
5. 2D-specific reference law validation (separate case)

Honest assessment:
The device is physically reasonable (momentum balance, dilatancy, coordination number
all make sense), but it does not reproduce the reference mu(I) curve within the 10%
tolerance. This could be due to:
- The reference law being for a different system (glass beads vs this contact model)
- Insufficient strain to reach true steady state
- Finite-size effects
- The contact model parameters not being appropriate

The case CANNOT be marked as PASS with the current evidence. It is WITHHELD pending
further investigation.
""")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
