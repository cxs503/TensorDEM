"""Comprehensive RCP validation: geometric compression vs physical DEM settling.

This test validates that both the geometric compression and physical DEM settling
methods produce RCP volume fractions consistent with literature references.

References:
- Scott, G. D. (1960). "Packing of Spheres". Nature 188, 908.
- Bernal, J. D. & Mason, J. (1960). "Packing of Spheres". Nature 188, 910.
- Torquato, S. et al. (2000). "Random close packing of hard spheres". Phys. Rev. E 62, 3404.
- Donev, A. et al. (2004). "Improving the density of jammed disordered packings". Science 303, 990.
- Silbert, L. E. et al. (2002). "Geometry of frictionless and frictional sphere packings". Phys. Rev. E 65, 031304.
"""
import json
import math
import os
import glob

import numpy as np
import pytest

from tensordem.bedcoupling.packing import make_random_close_packing, local_porosity, porosity_statistics
from tensordem.rcp_references import get_reference, reference_table

# Reference values
RCP_REF_PHI = 0.6366  # Bernal & Mason 1960
RCP_TOLERANCE_ABS = 0.02  # task tolerance: 0.62-0.66


# ---------------------------------------------------------------------------
# Geometric compression tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_particles,seed", [
    (1000, 42),
    (3000, 42),
    (5000, 42),
])
def test_geometric_rcp_in_tolerance(n_particles, seed):
    """Geometric compression should produce φ in [0.62, 0.66]."""
    bed = make_random_close_packing(
        n_particles, 0.001, 0.36,
        seed=seed,
        final_relax_iterations=6000,
        overlap_tolerance=1e-3,
    )
    phi = bed.solid_fraction
    assert 0.62 <= phi <= 0.66, f"φ={phi} outside tolerance [0.62, 0.66]"
    assert abs(phi - RCP_REF_PHI) < RCP_TOLERANCE_ABS


def test_geometric_rcp_error_less_than_3pct():
    """Geometric compression error vs reference should be < 3%."""
    bed = make_random_close_packing(
        3000, 0.001, 0.36,
        seed=42,
        final_relax_iterations=6000,
        overlap_tolerance=1e-3,
    )
    phi = bed.solid_fraction
    err_pct = abs(phi - RCP_REF_PHI) / RCP_REF_PHI * 100
    assert err_pct < 3.0, f"Error {err_pct:.2f}% >= 3%"


def test_geometric_overlap_acceptable():
    """Geometric compression overlap/d should be < 0.05."""
    bed = make_random_close_packing(
        3000, 0.001, 0.36,
        seed=42,
        final_relax_iterations=6000,
        overlap_tolerance=1e-3,
    )
    overlap_d = bed.max_overlap / bed.diameter
    assert overlap_d < 0.05, f"overlap/d={overlap_d} >= 0.05"


def test_geometric_local_porosity_uniform():
    """Local porosity std should be < 0.02."""
    bed = make_random_close_packing(
        3000, 0.001, 0.36,
        seed=42,
        final_relax_iterations=6000,
        overlap_tolerance=1e-3,
    )
    probe_radius = 2.5 * 0.001
    eps_local = local_porosity(bed.positions, 0.001, bed.box_length, probe_radius)
    eps_stats = porosity_statistics(eps_local, bed.porosity)
    assert eps_stats["std"] < 0.02, f"Local porosity std={eps_stats['std']} >= 0.02"


# ---------------------------------------------------------------------------
# Physical DEM settling tests (using existing results)
# ---------------------------------------------------------------------------

def _load_deposition_results():
    """Load deposition results from the runs directory."""
    runs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmarks", "case_rcp_deposition", "runs"
    )
    results = []
    pattern = os.path.join(runs_dir, "deposition_n10000_mu*_kn4000_e0.1_seed*.json")
    for f in glob.glob(pattern):
        if "_dtf" in f:
            continue
        with open(f) as fh:
            d = json.load(fh)
        m = d.get("metrics", {})
        spec = d.get("spec", {})
        results.append({
            "mu": spec.get("mu", 0),
            "seed": spec.get("seed", 0),
            "phi_bulk": m.get("phi_bulk", 0),
            "phi_bulk_corr": m.get("phi_bulk_overlap_corrected", 0),
            "coordination": m.get("coordination_bulk", 0),
            "settled": d.get("settled", False),
        })
    return results


@pytest.fixture(scope="module")
def deposition_results():
    return _load_deposition_results()


def test_dem_mu0_phi_in_tolerance(deposition_results):
    """DEM settling with μ=0 should produce φ in [0.62, 0.66]."""
    mu0_results = [r for r in deposition_results if r["mu"] == 0.0]
    assert len(mu0_results) >= 3, "Need at least 3 seeds for μ=0"

    phis = [r["phi_bulk_corr"] for r in mu0_results]
    phi_mean = np.mean(phis)
    assert 0.62 <= phi_mean <= 0.66, f"μ=0 φ_mean={phi_mean} outside tolerance"


def test_dem_mu0_error_less_than_3pct(deposition_results):
    """DEM settling with μ=0 error vs reference should be < 3%."""
    mu0_results = [r for r in deposition_results if r["mu"] == 0.0]
    phis = [r["phi_bulk_corr"] for r in mu0_results]
    phi_mean = np.mean(phis)

    err_pct = abs(phi_mean - RCP_REF_PHI) / RCP_REF_PHI * 100
    assert err_pct < 3.0, f"μ=0 error {err_pct:.2f}% >= 3%"


def test_dem_mu01_in_friction_band(deposition_results):
    """DEM settling with μ=0.1 should be in the friction-dependent reference band."""
    mu01_results = [r for r in deposition_results if r["mu"] == 0.1]
    assert len(mu01_results) >= 3, "Need at least 3 seeds for μ=0.1"

    phis = [r["phi_bulk_corr"] for r in mu01_results]
    phi_mean = np.mean(phis)

    ref = get_reference(0.1, method="nearest")
    assert ref.in_band(phi_mean), f"μ=0.1 φ={phi_mean} not in band {ref.band}"


def test_dem_friction_lowers_packing(deposition_results):
    """Physical DEM: friction should lower the packing fraction."""
    mu0_results = [r for r in deposition_results if r["mu"] == 0.0]
    mu01_results = [r for r in deposition_results if r["mu"] == 0.1]

    phi_mu0 = np.mean([r["phi_bulk_corr"] for r in mu0_results])
    phi_mu01 = np.mean([r["phi_bulk_corr"] for r in mu01_results])

    assert phi_mu01 < phi_mu0, f"μ=0.1 φ={phi_mu01} >= μ=0 φ={phi_mu0}"


def test_dem_coordination_numbers_physical(deposition_results):
    """Coordination numbers should be in physically plausible ranges."""
    mu0_results = [r for r in deposition_results if r["mu"] == 0.0]
    mu01_results = [r for r in deposition_results if r["mu"] == 0.1]

    Z_mu0 = np.mean([r["coordination"] for r in mu0_results])
    Z_mu01 = np.mean([r["coordination"] for r in mu01_results])

    # Frictionless: near isostatic Z=6 (slightly hyperstatic due to soft contacts)
    assert 5.5 < Z_mu0 < 7.0, f"μ=0 Z={Z_mu0} outside plausible range"
    # Frictional: between isostatic Z=4 and Z=6
    assert 4.5 < Z_mu01 < 6.5, f"μ=0.1 Z={Z_mu01} outside plausible range"
    # Friction should lower coordination
    assert Z_mu01 < Z_mu0, f"μ=0.1 Z={Z_mu01} >= μ=0 Z={Z_mu0}"


def test_dem_all_settled(deposition_results):
    """All deposition runs should have settled."""
    for r in deposition_results:
        assert r["settled"], f"Run mu={r['mu']} seed={r['seed']} did not settle"


# ---------------------------------------------------------------------------
# Comprehensive comparison test
# ---------------------------------------------------------------------------

def test_comprehensive_comparison_report_exists():
    """The comprehensive comparison report should exist and be valid."""
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "case_rcp_comprehensive", "comprehensive_comparison.json"
    )
    assert os.path.exists(report_path), f"Report not found: {report_path}"

    with open(report_path) as f:
        report = json.load(f)

    # Check required keys
    assert "title" in report
    assert "summary" in report
    assert "methods" in report
    assert "error_analysis" in report
    assert "gates" in report
    assert "figures" in report
    assert "references" in report

    # Check all gates pass
    assert report["gates"]["all_gates_pass"], "Not all gates pass"


def test_geometric_vs_dem_consistency(deposition_results):
    """Geometric compression and DEM μ=0 should give consistent results."""
    # Geometric compression
    bed = make_random_close_packing(
        3000, 0.001, 0.36,
        seed=42,
        final_relax_iterations=6000,
        overlap_tolerance=1e-3,
    )
    phi_geo = bed.solid_fraction

    # DEM μ=0
    mu0_results = [r for r in deposition_results if r["mu"] == 0.0]
    phi_dem = np.mean([r["phi_bulk_corr"] for r in mu0_results])

    # Should be within 1% of each other
    assert abs(phi_geo - phi_dem) < 0.01, \
        f"Geometric φ={phi_geo} vs DEM μ=0 φ={phi_dem} differ by {abs(phi_geo - phi_dem):.4f}"
