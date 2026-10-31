"""Pytest suite for scale-up broken ice field (SIR-PC2|I02|G6).

Acceptance criteria (all must pass):
  particle_count >= 100000 [count]
  mass_conservation < 1e-08 [ratio]
  wallclock_per_step < 60 [s]
"""
from __future__ import annotations

import math
import time

import pytest
import torch

from tensordem.scale_up_ice_field import (
    ScaleUpIceField,
    ScaleUpIceFieldConfig,
    _vectorized_bond_pairs,
    _vectorized_cell_list_pairs,
    run_broken_ice_field,
)
from tensordem.ice_field_adapter import (
    CPUCellListAdapter,
    INTERFACE_CONTRACT,
    get_contact_detector,
    get_scaling_benchmark,
)


# ---------------------------------------------------------------------------
# Small-scale unit tests (fast, no 100k)
# ---------------------------------------------------------------------------
class TestVectorizedBondPairs:
    """Verify vectorized bond construction matches expected topology."""

    def test_small_grid_bond_count(self):
        """3×3×3 box: 27 particles, each with up to 13 half-space neighbours."""
        pairs = _vectorized_bond_pairs(3, 3, 3, torch.device("cpu"), torch.float64)
        # 27 particles, 13 offsets, but edge particles have fewer neighbours
        # Expected: 3*3*3 lattice with face+body diagonal bonds
        # For a 3x3x3 box: 54 face bonds + 12 face-diagonal + ... total ~ 76
        assert pairs.shape[1] == 2
        assert pairs.dtype == torch.long
        # All pairs should have i < j
        assert (pairs[:, 0] < pairs[:, 1]).all()
        # No duplicate pairs
        keys = pairs[:, 0] * 27 + pairs[:, 1]
        assert keys.unique().numel() == keys.numel()

    def test_2x2x2_bond_count(self):
        """2×2×2 box: 8 particles, 12 face bonds + 12 face-diagonal + 4 body-diagonal = 28."""
        pairs = _vectorized_bond_pairs(2, 2, 2, torch.device("cpu"), torch.float64)
        # 8 particles, 13 offsets, but many are out of bounds
        # Face bonds: 3 directions * 4 = 12
        # Face diagonal: 6 face-diag directions * 2 = 12
        # Body diagonal: 1 body-diag direction * 4 = 4
        # Total: 28
        assert pairs.shape[0] == 28

    def test_rectangular_grid(self):
        """5×3×2 box: verify no out-of-bounds indices."""
        nx, ny, nz = 5, 3, 2
        n = nx * ny * nz
        pairs = _vectorized_bond_pairs(nx, ny, nz, torch.device("cpu"), torch.float64)
        assert (pairs[:, 0] >= 0).all()
        assert (pairs[:, 1] < n).all()
        assert (pairs[:, 0] < pairs[:, 1]).all()


class TestVectorizedCellList:
    """Verify vectorized cell-list contact detection."""

    def test_no_overlap_no_pairs(self):
        """Well-separated particles should produce no contact pairs."""
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float64)
        radius = 0.1
        pairs = _vectorized_cell_list_pairs(
            positions, radius, torch.device("cpu"), torch.float64)
        assert pairs.numel() == 0

    def test_overlapping_pair_detected(self):
        """Two overlapping spheres should produce one contact pair."""
        radius = 0.1
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0],  # overlap = 2*0.1 - 0.15 = 0.05 > 0
        ], dtype=torch.float64)
        pairs = _vectorized_cell_list_list_pairs = _vectorized_cell_list_pairs(
            positions, radius, torch.device("cpu"), torch.float64)
        assert pairs.shape[0] >= 1
        # Should contain pair (0, 1)
        found = ((pairs[:, 0] == 0) & (pairs[:, 1] == 1)).any()
        assert found, f"Expected pair (0,1) in {pairs.tolist()}"

    def test_lattice_contact_pairs(self):
        """Particles on a lattice at 2*radius spacing: neighbours are contacts."""
        nx, ny, nz = 4, 3, 2
        radius = 0.025
        n = nx * ny * nz
        axes = [torch.arange(v, dtype=torch.float64) for v in (nx, ny, nz)]
        zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        grid = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
        centre = torch.tensor([(nx-1)/2, (ny-1)/2, (nz-1)/2])
        positions = (grid - centre) * (2 * radius)
        pairs = _vectorized_cell_list_pairs(
            positions, radius, torch.device("cpu"), torch.float64)
        # Every lattice neighbour pair should be a contact candidate
        assert pairs.shape[0] > 0
        assert (pairs[:, 0] < pairs[:, 1]).all()


class TestScaleUpIceFieldSmall:
    """Small-scale integration tests for ScaleUpIceField."""

    def test_creation_and_particle_count(self):
        cfg = ScaleUpIceFieldConfig(nx=5, ny=3, nz=4)
        sim = ScaleUpIceField(cfg)
        assert sim.particle_count == 5 * 3 * 4
        assert sim.mass_conservation_ratio == 0.0

    def test_mass_conservation_after_steps(self):
        cfg = ScaleUpIceFieldConfig(nx=5, ny=3, nz=4, pre_break_fraction=0.2)
        sim = ScaleUpIceField(cfg)
        for _ in range(10):
            sim.step()
        assert sim.mass_conservation_ratio < 1e-15

    def test_fixed_particles_dont_move(self):
        cfg = ScaleUpIceFieldConfig(nx=5, ny=3, nz=4, gravity=0.0, pre_break_fraction=0.0)
        sim = ScaleUpIceField(cfg)
        fixed_pos0 = sim.positions[sim.fixed].clone()
        for _ in range(5):
            sim.step()
        torch.testing.assert_close(
            sim.positions[sim.fixed], fixed_pos0, atol=0, rtol=0)

    def test_gravity_accelerates_free_particles(self):
        cfg = ScaleUpIceFieldConfig(nx=3, ny=3, nz=3, gravity=9.81,
                                    pre_break_fraction=0.99, drag=0.0)
        sim = ScaleUpIceField(cfg)
        # With 99% bonds broken and no drag, free particles should accelerate
        free = ~sim.fixed
        v0 = sim.velocities[free].clone()
        sim.step()
        v1 = sim.velocities[free]
        # At least some free particles should have non-zero velocity
        assert float(v1.abs().max()) > 0, "Free particles should move under gravity"

    def test_pre_breaking_creates_floes(self):
        cfg = ScaleUpIceFieldConfig(nx=5, ny=3, nz=4, pre_break_fraction=0.5, seed=42)
        sim = ScaleUpIceField(cfg)
        assert sim.broken_bonds > 0
        total_bonds = len(sim.pairs)
        expected_broken = int(total_bonds * 0.5)
        # Allow 10% tolerance due to random sampling
        assert abs(sim.broken_bonds - expected_broken) < 0.1 * total_bonds

    def test_state_finite_after_steps(self):
        cfg = ScaleUpIceFieldConfig(nx=5, ny=3, nz=4)
        sim = ScaleUpIceField(cfg)
        for _ in range(20):
            sim.step()
        assert torch.isfinite(sim.positions).all()
        assert torch.isfinite(sim.velocities).all()


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------
class TestAdapters:
    def test_cpu_cell_list_adapter(self):
        adapter = CPUCellListAdapter()
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0],
        ], dtype=torch.float64)
        pairs = adapter.detect(positions, 0.1)
        assert pairs.shape[0] >= 1

    def test_get_contact_detector_cpu(self):
        detector = get_contact_detector("cpu")
        assert isinstance(detector, CPUCellListAdapter)

    def test_interface_contract_structure(self):
        assert "V03_gpu_cell_list" in INTERFACE_CONTRACT
        assert "V02_scaling_benchmark" in INTERFACE_CONTRACT
        for key, val in INTERFACE_CONTRACT.items():
            assert "module" in val
            assert "function" in val
            assert "acceptance" in val
            assert "fallback" in val


# ---------------------------------------------------------------------------
# Scale-up acceptance tests (100k+ particles)
# ---------------------------------------------------------------------------
class TestScaleUpAcceptance:
    """Acceptance criteria for SIR-PC2|I02|G6.

    particle_count >= 100000 [count]
    mass_conservation < 1e-08 [ratio]
    wallclock_per_step < 60 [s]
    """

    @pytest.fixture(scope="class")
    def sim_100k(self):
        """Create and run 3 steps of a 100k+ particle simulation."""
        cfg = ScaleUpIceFieldConfig(nx=100, ny=10, nz=100)
        sim = ScaleUpIceField(cfg)
        # Run 3 steps to get stable timing
        for _ in range(3):
            sim.step()
        return sim

    def test_particle_count_100k(self, sim_100k):
        assert sim_100k.particle_count >= 100_000, (
            f"particle_count={sim_100k.particle_count} < 100000")

    def test_mass_conservation(self, sim_100k):
        mc = sim_100k.mass_conservation_ratio
        assert mc < 1e-8, f"mass_conservation={mc:.2e} >= 1e-8"

    def test_wallclock_per_step(self, sim_100k):
        max_wc = max(sim_100k._wallclock_per_step)
        assert max_wc < 60.0, f"wallclock_per_step={max_wc:.2f}s >= 60s"

    def test_mean_wallclock_reasonable(self, sim_100k):
        mean_wc = sim_100k.mean_wallclock_per_step
        # Sanity: mean should also be < 60s
        assert mean_wc < 60.0, f"mean wallclock={mean_wc:.2f}s >= 60s"


# ---------------------------------------------------------------------------
# Convenience runner test
# ---------------------------------------------------------------------------
class TestRunBrokenIceField:
    def test_small_run(self):
        result = run_broken_ice_field(nx=5, ny=3, nz=4, n_steps=3)
        assert result["particle_count"] == 60
        assert result["mass_conservation"] < 1e-8
        assert result["total_steps"] == 3
        assert "config" in result
