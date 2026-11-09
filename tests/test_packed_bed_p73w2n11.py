"""P73W2-N11: Acceptance tests for packed-bed thermal conductivity G1–G8 gates."""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.thermal import packed_bed as pb


# ============================================================================
# Maxwell-Eucken analytical properties
# ============================================================================
class TestMaxwellEucken:
    """Tests for maxwell_eucken_1 and maxwell_eucken_2."""

    def test_me1_within_wiener_bounds(self):
        """ME1 must lie within Wiener bounds for all (phi, contrast)."""
        for phi in (0.2, 0.4, 0.6, 0.8):
            for c in (1.0, 5.0, 20.0, 100.0):
                ks, kf = c * 0.6, 0.6
                me1 = pb.maxwell_eucken_1(phi, ks, kf)
                wb = pb.wiener_bounds(phi, ks, kf)
                assert wb["k_series"] <= me1 <= wb["k_parallel"]

    def test_me2_within_wiener_bounds(self):
        """ME2 must lie within Wiener bounds for all (phi, contrast)."""
        for phi in (0.2, 0.4, 0.6, 0.8):
            for c in (1.0, 5.0, 20.0, 100.0):
                ks, kf = c * 0.6, 0.6
                me2 = pb.maxwell_eucken_2(phi, ks, kf)
                wb = pb.wiener_bounds(phi, ks, kf)
                assert wb["k_series"] <= me2 <= wb["k_parallel"]

    def test_me1_phi0_limit(self):
        """ME1 → k_solid as phi → 0."""
        ks, kf = 50.0, 0.6
        me1 = pb.maxwell_eucken_1(1e-8, ks, kf)
        assert abs(me1 - ks) / ks < 1e-6

    def test_me1_phi1_limit(self):
        """ME1 → k_fluid as phi → 1."""
        ks, kf = 50.0, 0.6
        me1 = pb.maxwell_eucken_1(1 - 1e-8, ks, kf)
        assert abs(me1 - kf) / kf < 1e-6

    def test_me2_phi0_limit(self):
        """ME2 → k_solid as phi → 0."""
        ks, kf = 50.0, 0.6
        me2 = pb.maxwell_eucken_2(1e-8, ks, kf)
        assert abs(me2 - ks) / ks < 1e-6

    def test_me2_phi1_limit(self):
        """ME2 → k_fluid as phi → 1."""
        ks, kf = 50.0, 0.6
        me2 = pb.maxwell_eucken_2(1 - 1e-8, ks, kf)
        assert abs(me2 - kf) / kf < 1e-6

    def test_me_equal_conductivities(self):
        """ME1 = ME2 = k when k_s = k_f."""
        for phi in (0.3, 0.5, 0.7):
            k = 3.0
            assert pb.maxwell_eucken_1(phi, k, k) == pytest.approx(k, rel=1e-12)
            assert pb.maxwell_eucken_2(phi, k, k) == pytest.approx(k, rel=1e-12)

    def test_me1_me2_ordering(self):
        """ME1 (fluid continuous) ≤ ME2 (solid continuous) for k_s > k_f."""
        for phi in (0.3, 0.5, 0.7):
            ks, kf = 50.0, 0.6
            me1 = pb.maxwell_eucken_1(phi, ks, kf)
            me2 = pb.maxwell_eucken_2(phi, ks, kf)
            assert me1 < me2

    def test_me_invalid_inputs(self):
        """Invalid inputs should raise."""
        with pytest.raises(ValueError):
            pb.maxwell_eucken_1(0.0, 1.0, 1.0)  # phi=0 not allowed
        with pytest.raises(ValueError):
            pb.maxwell_eucken_1(1.0, 1.0, 1.0)  # phi=1 not allowed
        with pytest.raises(ValueError):
            pb.maxwell_eucken_1(0.5, -1.0, 1.0)  # negative k


# ============================================================================
# G1: Flux conservation
# ============================================================================
class TestG1FluxConservation:
    """G1: steady-state heat flux inlet/outlet relative deviation ≤ 1e-10."""

    def test_flux_conservation_periodic(self):
        """All z-plane fluxes must be equal (Kirchhoff conservation)."""
        bed = pb.generate_static_bed(0.40, n_spheres=64, seed=42)
        n = 20
        h = bed.box_length / n
        labels = pb._voxel_labels(bed, n)
        ks, kf = 1.4, 0.026
        conductances = []
        for d in range(3):
            left = labels
            right = np.roll(labels, -1, axis=d)
            lk = np.where(left >= 0, ks, kf)
            rk = np.where(right >= 0, ks, kf)
            conductances.append(2.0 * h * lk * rk / (lk + rk))
        _, temp = pb._periodic_solve(conductances, h, n, gradient=1.0)
        idx = np.arange(n**3).reshape(n, n, n)
        flat_t = temp.reshape(-1)
        fluxes = []
        for z in range(n):
            i0 = idx[:, :, z].reshape(-1)
            i1 = idx[:, :, (z + 1) % n].reshape(-1)
            g = conductances[2][:, :, z].reshape(-1)
            fluxes.append(float(np.sum(g * (flat_t[i0] - flat_t[i1] + h)) * h))
        fluxes = np.array(fluxes)
        mean_f = np.mean(fluxes)
        max_dev = np.max(np.abs(fluxes - mean_f)) / abs(mean_f)
        assert max_dev <= 1e-10


# ============================================================================
# G2: Wiener hard gate
# ============================================================================
class TestG2WienerBounds:
    """G2: k_series ≤ k_eff ≤ k_parallel for all combinations."""

    @pytest.mark.parametrize("phi", [0.30, 0.40, 0.50, 0.60])
    @pytest.mark.parametrize("contrast", [1.0, 10.0, 100.0])
    def test_wiener_bounds_held(self, phi, contrast):
        ks = contrast * 0.026
        kf = 0.026
        bed = pb.generate_static_bed(phi, n_spheres=48, seed=20261107)
        res = pb.effective_conductivity(bed, ks, kf, grid=16)
        assert res["k_series"] * (1 - 1e-6) <= res["k_eff"]
        assert res["k_eff"] <= res["k_parallel"] * (1 + 1e-6)


# ============================================================================
# G4: Degenerate limits
# ============================================================================
class TestG4DegenerateLimits:
    """G4: k_s=k_f → k_eff=k_f (exact); ME limits exact."""

    def test_ks_eq_kf_exact(self):
        """k_s = k_f → k_eff = k_f exactly."""
        bed = pb.generate_static_bed(0.40, n_spheres=64, seed=42)
        res = pb.conductivity_flux_average(bed, 2.5, 2.5, grid=20)
        assert abs(res["k_eff"] - 2.5) / 2.5 <= 1e-9

    def test_me_limits_phi0(self):
        """ME1, ME2 → k_solid as phi → 0."""
        ks, kf = 10.0, 0.5
        me1 = pb.maxwell_eucken_1(1e-8, ks, kf)
        me2 = pb.maxwell_eucken_2(1e-8, ks, kf)
        assert abs(me1 - ks) / ks < 0.02
        assert abs(me2 - ks) / ks < 0.02

    def test_me_limits_phi1(self):
        """ME1, ME2 → k_fluid as phi → 1."""
        ks, kf = 10.0, 0.5
        me1 = pb.maxwell_eucken_1(1 - 1e-8, ks, kf)
        me2 = pb.maxwell_eucken_2(1 - 1e-8, ks, kf)
        assert abs(me1 - kf) / kf < 0.02
        assert abs(me2 - kf) / kf < 0.02


# ============================================================================
# G5: Seed dispersion
# ============================================================================
class TestG5SeedDispersion:
    """G5: 5 seeds CV ≤ 3%."""

    def test_seed_cv(self):
        ks, kf = 10.0, 0.5
        k_vals = []
        for s in [42, 123, 456, 789, 2026]:
            bed = pb.generate_static_bed(0.45, n_spheres=64, seed=s)
            k_vals.append(pb.effective_conductivity(bed, ks, kf, grid=16)["k_eff"])
        cv = np.std(k_vals) / np.mean(k_vals)
        assert cv <= 0.03


# ============================================================================
# G6: Homogeneous profile
# ============================================================================
class TestG6HomogeneousProfile:
    """G6: homogeneous limit → T' = 0, linear profile."""

    def test_homogeneous_t_prime_zero(self):
        """T' = 0 for k_s = k_f."""
        bed = pb.generate_static_bed(0.40, n_spheres=64, seed=42)
        n = 20
        h = bed.box_length / n
        labels = pb._voxel_labels(bed, n)
        k = 3.0
        conductances = [np.full_like(labels, 2.0 * h * k, dtype=float) for _ in range(3)]
        _, t_prime = pb._periodic_solve(conductances, h, n, gradient=1.0)
        assert np.sqrt(np.mean(t_prime**2)) < 1e-10

    def test_homogeneous_linear_slope(self):
        """Full T = T' + G·z has slope = G for homogeneous medium."""
        bed = pb.generate_static_bed(0.40, n_spheres=64, seed=42)
        n = 20
        h = bed.box_length / n
        labels = pb._voxel_labels(bed, n)
        k = 3.0
        conductances = [np.full_like(labels, 2.0 * h * k, dtype=float) for _ in range(3)]
        _, t_prime = pb._periodic_solve(conductances, h, n, gradient=1.0)
        z3d = np.arange(n).reshape(1, 1, n) * h
        t_full = t_prime + z3d
        t_avg = t_full.mean(axis=(0, 1))
        z1d = (np.arange(n) + 0.5) * h
        coeffs = np.polyfit(z1d, t_avg, 1)
        assert abs(coeffs[0] - 1.0) < 1e-6


# ============================================================================
# G8: Radiation exclusion
# ============================================================================
class TestG8RadiationExclusion:
    """G8: radiation not negligible above 300°C."""

    def test_radiation_significant_above_300C(self):
        """Linearised radiation conductance exceeds gap conduction at 300°C."""
        sigma = 5.67e-8
        eps = 0.8
        h_rad = 4 * eps * sigma * 573.15**3
        h_cond = 0.026 / 1e-3
        assert h_rad / h_cond > 1.0  # radiation dominates at 300°C
