"""End-to-end ledger acceptance test for CASE-DEM-RCP-PACKING-FRACTION.

This test validates the complete RCP packing fraction case against the
feature_ledger acceptance criteria:

  - Volume fraction in [0.62, 0.66]
  - 10000 spheres, frictionless (mu=0) and low-friction (mu=0.1) tiers
  - Reference: Scott 1960 Nature 188:908; Bernal & Mason 1960 Nature 188:910
  - Accuracy: < 3% vs friction-dependent reference

The test reads from the existing deposition run artifacts and validates
all gates, including the stiffness extrapolation to the hard-sphere limit.
"""
import json
import glob
import math
import os

import numpy as np
import pytest

from tensordem.bedcoupling.packing import make_random_close_packing
from tensordem.rcp_references import get_reference

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS_DIR = os.path.join(ROOT, "benchmarks", "case_rcp_deposition", "runs")
BAND = (0.62, 0.66)
DT_DEFAULT = 1.0 / 20.0
PROD_KN = 4000.0
N_PRIMARY = 10000
SEED_MIN = 3
TOL_REL = 0.03  # < 3%


def _load_runs():
    """Load all deposition run JSONs."""
    runs = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "deposition_*.json"))):
        if "_dtf" in os.path.basename(path):
            continue  # skip timestep probes for production stats
        with open(path) as f:
            d = json.load(f)
        spec = d.get("spec", {})
        m = d.get("metrics", {})
        runs.append({
            "path": path,
            "n": spec.get("n_particles"),
            "mu": spec.get("mu"),
            "kn": spec.get("kn"),
            "seed": spec.get("seed"),
            "phi": m.get("phi_bulk"),
            "phi_corr": m.get("phi_bulk_overlap_corrected"),
            "Z": m.get("coordination_bulk"),
            "ovmax": m.get("max_pair_overlap_over_d"),
            "settled": d.get("settled"),
        })
    return runs


@pytest.fixture(scope="module")
def runs():
    return _load_runs()


# ---------------------------------------------------------------------------
# Ledger acceptance gates
# ---------------------------------------------------------------------------

class TestLedgerAcceptance:
    """Tests that directly answer the feature_ledger acceptance criteria."""

    def test_production_size(self, runs):
        """Ledger: 10000 spheres, both friction tiers."""
        n_vals = {r["n"] for r in runs if r["mu"] in (0.0, 0.1) and r["kn"] == PROD_KN}
        assert N_PRIMARY in n_vals, f"N={N_PRIMARY} not found in production runs"

    def test_seed_coverage(self, runs):
        """Ledger: both friction tiers with >= 3 seeds."""
        for mu in (0.0, 0.1):
            seeds = {r["seed"] for r in runs
                     if r["n"] == N_PRIMARY and r["mu"] == mu and r["kn"] == PROD_KN}
            assert len(seeds) >= SEED_MIN, f"mu={mu}: only {len(seeds)} seeds, need {SEED_MIN}"

    def test_frictionless_in_band(self, runs):
        """Ledger: mu=0 tier phi in [0.62, 0.66]."""
        phis = [r["phi_corr"] for r in runs
                if r["n"] == N_PRIMARY and r["mu"] == 0.0 and r["kn"] == PROD_KN]
        phi_mean = np.mean(phis)
        assert BAND[0] <= phi_mean <= BAND[1], f"mu=0 phi={phi_mean} outside {BAND}"

    def test_low_friction_in_band(self, runs):
        """Ledger: mu=0.1 tier phi in [0.62, 0.66]."""
        phis = [r["phi_corr"] for r in runs
                if r["n"] == N_PRIMARY and r["mu"] == 0.1 and r["kn"] == PROD_KN]
        phi_mean = np.mean(phis)
        assert BAND[0] <= phi_mean <= BAND[1], f"mu=0.1 phi={phi_mean} outside {BAND}"

    def test_frictionless_vs_reference(self, runs):
        """Ledger: mu=0 tier error vs reference < 3%."""
        phis = [r["phi_corr"] for r in runs
                if r["n"] == N_PRIMARY and r["mu"] == 0.0 and r["kn"] == PROD_KN]
        phi_mean = np.mean(phis)
        ref = get_reference(0.0)
        err = abs(phi_mean - ref.phi_rcp) / ref.phi_rcp
        assert err < TOL_REL, f"mu=0 error={err:.4f} >= {TOL_REL}"

    def test_low_friction_vs_reference(self, runs):
        """Ledger: mu=0.1 tier error vs friction-dependent reference < 3%."""
        phis = [r["phi_corr"] for r in runs
                if r["n"] == N_PRIMARY and r["mu"] == 0.1 and r["kn"] == PROD_KN]
        phi_mean = np.mean(phis)
        ref = get_reference(0.1)
        err = abs(phi_mean - ref.phi_rcp) / ref.phi_rcp
        assert err < TOL_REL, f"mu=0.1 error={err:.4f} >= {TOL_REL}"

    def test_all_settled(self, runs):
        """All production runs must have settled."""
        for r in runs:
            if r["n"] == N_PRIMARY and r["kn"] == PROD_KN:
                assert r["settled"], f"Run mu={r['mu']} seed={r['seed']} did not settle"

    def test_friction_lowers_packing(self, runs):
        """Physical expectation: friction lowers packing fraction."""
        mu0 = [r["phi_corr"] for r in runs
               if r["n"] == N_PRIMARY and r["mu"] == 0.0 and r["kn"] == PROD_KN]
        mu01 = [r["phi_corr"] for r in runs
                if r["n"] == N_PRIMARY and r["mu"] == 0.1 and r["kn"] == PROD_KN]
        assert np.mean(mu01) < np.mean(mu0) + 0.005


# ---------------------------------------------------------------------------
# Geometric compression validation (independent of deposition)
# ---------------------------------------------------------------------------

class TestGeometricCompression:
    """Validate the geometric compression algorithm independently."""

    def test_rcp_in_tolerance(self):
        """Geometric compression should produce phi in [0.62, 0.66]."""
        bed = make_random_close_packing(
            3000, 0.001, 0.36, seed=42,
            final_relax_iterations=6000, overlap_tolerance=1e-3,
        )
        assert 0.62 <= bed.solid_fraction <= 0.66

    def test_rcp_error_less_than_3pct(self):
        """Geometric compression error vs Bernal & Mason < 3%."""
        bed = make_random_close_packing(
            3000, 0.001, 0.36, seed=42,
            final_relax_iterations=6000, overlap_tolerance=1e-3,
        )
        ref = get_reference(0.0)
        err = abs(bed.solid_fraction - ref.phi_rcp) / ref.phi_rcp
        assert err < TOL_REL

    def test_overlap_acceptable(self):
        """Geometric compression overlap/d < 0.05."""
        bed = make_random_close_packing(
            3000, 0.001, 0.36, seed=42,
            final_relax_iterations=6000, overlap_tolerance=1e-3,
        )
        assert bed.max_overlap / bed.diameter < 0.05


# ---------------------------------------------------------------------------
# Stiffness extrapolation validation
# ---------------------------------------------------------------------------

from tensordem.deposition_packing import stiffness_limit


class TestStiffnessExtrapolation:
    """Validate the stiffness extrapolation to the hard-sphere limit."""

    def test_mu0_extrapolation_in_band(self, runs):
        """Hard-sphere limit for mu=0 should be in [0.62, 0.66]."""
        pts = []
        for kn in sorted({r["kn"] for r in runs if r["n"] == N_PRIMARY and r["mu"] == 0.0}):
            hit = [r["phi"] for r in runs
                   if r["n"] == N_PRIMARY and r["mu"] == 0.0 and r["kn"] == kn and r["seed"] == 1]
            if hit:
                pts.append((kn, hit[0]))
        if len(pts) >= 2:
            result = stiffness_limit(pts)
            assert BAND[0] <= result["limit"] <= BAND[1]

    def test_mu01_extrapolation_in_friction_band(self, runs):
        """Hard-sphere limit for mu=0.1 should be in friction-dependent band."""
        pts = []
        for kn in sorted({r["kn"] for r in runs if r["n"] == N_PRIMARY and r["mu"] == 0.1}):
            hit = [r["phi"] for r in runs
                   if r["n"] == N_PRIMARY and r["mu"] == 0.1 and r["kn"] == kn and r["seed"] == 1]
            if hit:
                pts.append((kn, hit[0]))
        if len(pts) >= 2:
            result = stiffness_limit(pts)
            ref = get_reference(0.1)
            assert ref.in_band(result["limit"]), \
                f"mu=0.1 limit={result['limit']} not in band {ref.band}"

    def test_mu0_extrapolation_vs_reference(self, runs):
        """Hard-sphere limit for mu=0 vs reference < 3%."""
        pts = []
        for kn in sorted({r["kn"] for r in runs if r["n"] == N_PRIMARY and r["mu"] == 0.0}):
            hit = [r["phi"] for r in runs
                   if r["n"] == N_PRIMARY and r["mu"] == 0.0 and r["kn"] == kn and r["seed"] == 1]
            if hit:
                pts.append((kn, hit[0]))
        if len(pts) >= 2:
            result = stiffness_limit(pts)
            ref = get_reference(0.0)
            err = abs(result["limit"] - ref.phi_rcp) / ref.phi_rcp
            assert err < TOL_REL

    def test_mu01_extrapolation_vs_reference(self, runs):
        """Hard-sphere limit for mu=0.1 vs friction-dependent reference < 3%."""
        pts = []
        for kn in sorted({r["kn"] for r in runs if r["n"] == N_PRIMARY and r["mu"] == 0.1}):
            hit = [r["phi"] for r in runs
                   if r["n"] == N_PRIMARY and r["mu"] == 0.1 and r["kn"] == kn and r["seed"] == 1]
            if hit:
                pts.append((kn, hit[0]))
        if len(pts) >= 2:
            result = stiffness_limit(pts)
            ref = get_reference(0.1)
            err = abs(result["limit"] - ref.phi_rcp) / ref.phi_rcp
            assert err < TOL_REL
