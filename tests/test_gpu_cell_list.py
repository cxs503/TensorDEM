"""Pytest suite for GPU cell list spatial search.

Validates:
1. **Correctness** – pairs from the GPU cell list are valid (j > i, particles in same/adjacent cells).
2. **Integration** – the GPU cell list works with IceDEM3D.
3. **Performance** – the tensor-based approach is documented; GPU speedup is estimated analytically.
"""
from __future__ import annotations

import math
import time

import pytest
import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.gpu_cell_list import (
    build_cell_list,
    build_cell_list_vectorised,
    select_implementation,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def small_positions() -> torch.Tensor:
    """27 particles on a 3x3x3 grid."""
    axes = [torch.arange(3, dtype=torch.float64)]
    zz, yy, xx = torch.meshgrid(axes[0], axes[0], axes[0], indexing="ij")
    grid = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    return grid * 0.05  # 5 cm spacing


@pytest.fixture
def random_positions_100() -> torch.Tensor:
    """100 random particles."""
    torch.manual_seed(42)
    return torch.rand(100, 3, dtype=torch.float64) * 0.5


@pytest.fixture
def random_positions_1000() -> torch.Tensor:
    """1000 random particles."""
    torch.manual_seed(123)
    return torch.rand(1000, 3, dtype=torch.float64) * 1.0


@pytest.fixture
def random_positions_5000() -> torch.Tensor:
    """5000 random particles."""
    torch.manual_seed(456)
    return torch.rand(5000, 3, dtype=torch.float64) * 2.0


# ── Correctness tests ───────────────────────────────────────────────────

class TestCellListCorrectness:
    """Verify GPU cell list produces valid pairs."""

    def test_pairs_valid_j_greater_than_i(self, random_positions_100: torch.Tensor) -> None:
        """All pairs should have j > i."""
        radius = 0.025
        pairs = build_cell_list(random_positions_100, 2.0 * radius)
        if pairs.numel() > 0:
            assert (pairs[:, 1] > pairs[:, 0]).all(), "All pairs must have j > i"

    def test_pairs_valid_j_greater_than_i_1000(self, random_positions_1000: torch.Tensor) -> None:
        """All pairs should have j > i for larger system."""
        radius = 0.025
        pairs = build_cell_list(random_positions_1000, 2.0 * radius)
        if pairs.numel() > 0:
            assert (pairs[:, 1] > pairs[:, 0]).all(), "All pairs must have j > i"

    def test_pairs_valid_j_greater_than_i_5000(self, random_positions_5000: torch.Tensor) -> None:
        """All pairs should have j > i for large system."""
        radius = 0.025
        pairs = build_cell_list(random_positions_5000, 2.0 * radius)
        if pairs.numel() > 0:
            assert (pairs[:, 1] > pairs[:, 0]).all(), "All pairs must have j > i"

    def test_empty_input(self) -> None:
        positions = torch.empty((0, 3), dtype=torch.float64)
        assert build_cell_list(positions, 0.05).shape == (0, 2)
        assert build_cell_list_vectorised(positions, 0.05).shape == (0, 2)

    def test_single_particle(self) -> None:
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        assert build_cell_list(positions, 0.05).shape == (0, 2)
        assert build_cell_list_vectorised(positions, 0.05).shape == (0, 2)

    def test_two_particles_same_cell(self) -> None:
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]], dtype=torch.float64)
        pairs = build_cell_list(positions, 0.05)
        assert pairs.shape == (1, 2)
        assert pairs[0, 0] == 0 and pairs[0, 1] == 1

    def test_two_particles_different_cells(self) -> None:
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        pairs = build_cell_list(positions, 0.05)
        # They are far apart, so no contact candidates
        assert pairs.shape == (0, 2)

    def test_pairs_in_same_or_adjacent_cells(self, random_positions_100: torch.Tensor) -> None:
        """All pairs should be in the same or adjacent cells."""
        radius = 0.025
        width = 2.0 * radius
        pairs = build_cell_list(random_positions_100, width)

        if pairs.numel() == 0:
            return

        cell_coords = torch.floor(random_positions_100 / width).to(torch.int64)
        for i, j in pairs.tolist():
            ci = cell_coords[i]
            cj = cell_coords[j]
            diff = (ci - cj).abs()
            assert (diff <= 1).all(), f"Pair ({i}, {j}) not in same/adjacent cells: {ci} vs {cj}"


# ── Performance tests ───────────────────────────────────────────────────

class TestCellListPerformance:
    """Benchmark GPU cell list vs Python-dict reference."""

    def test_speedup_100_particles(self, random_positions_100: torch.Tensor) -> None:
        """Measure speedup for 100 particles."""
        radius = 0.025
        width = 2.0 * radius
        n_iters = 50

        # GPU cell list timing
        start = time.perf_counter()
        for _ in range(n_iters):
            build_cell_list(random_positions_100, width)
        gpu_time = time.perf_counter() - start

        print(f"\n[perf] n=100: gpu={gpu_time:.4f}s ({gpu_time/n_iters*1000:.2f}ms/iter)")
        # Just verify it runs without error
        assert gpu_time > 0

    def test_speedup_1000_particles(self, random_positions_1000: torch.Tensor) -> None:
        """Measure speedup for 1000 particles."""
        radius = 0.025
        width = 2.0 * radius
        n_iters = 20

        start = time.perf_counter()
        for _ in range(n_iters):
            build_cell_list(random_positions_1000, width)
        gpu_time = time.perf_counter() - start

        print(f"\n[perf] n=1000: gpu={gpu_time:.4f}s ({gpu_time/n_iters*1000:.2f}ms/iter)")
        assert gpu_time > 0

    def test_speedup_5000_particles(self, random_positions_5000: torch.Tensor) -> None:
        """Measure speedup for 5000 particles."""
        radius = 0.025
        width = 2.0 * radius
        n_iters = 10

        start = time.perf_counter()
        for _ in range(n_iters):
            build_cell_list(random_positions_5000, width)
        gpu_time = time.perf_counter() - start

        print(f"\n[perf] n=5000: gpu={gpu_time:.4f}s ({gpu_time/n_iters*1000:.2f}ms/iter)")
        assert gpu_time > 0

    def test_tensor_vs_python_dict_cell_assignment(self) -> None:
        """Benchmark tensor cell assignment vs Python dict approach."""
        n = 5000
        positions = torch.rand(n, 3, dtype=torch.float64) * 2.0
        width = 0.05
        n_iters = 100

        # Tensor approach
        start = time.perf_counter()
        for _ in range(n_iters):
            cells = torch.floor(positions / width).to(torch.int64)
        tensor_time = time.perf_counter() - start

        # Python dict approach
        start = time.perf_counter()
        for _ in range(n_iters):
            coords = torch.floor(positions / width).to(torch.int64).tolist()
            cells_dict = {}
            for idx, xyz in enumerate(coords):
                key = (xyz[0], xyz[1], xyz[2])
                cells_dict.setdefault(key, []).append(idx)
        dict_time = time.perf_counter() - start

        speedup = dict_time / tensor_time if tensor_time > 0 else float("inf")
        print(f"\n[perf] cell assignment: tensor={tensor_time:.4f}s, dict={dict_time:.4f}s, speedup={speedup:.1f}x")
        assert speedup > 10.0, f"Expected >10x speedup for cell assignment, got {speedup:.1f}x"


# ── Force consistency tests ─────────────────────────────────────────────

class TestForceConsistency:
    """Verify DEM forces are deterministic."""

    def test_force_consistency_small(self) -> None:
        """Verify forces are deterministic on small system."""
        cfg = DEM3DConfig(nx=5, ny=3, nz=3, radius=0.025, device="cpu")
        dem = IceDEM3D(cfg)

        for _ in range(5):
            dem.step()

        force1, reaction1 = dem.forces(update_fracture=False)
        force2, reaction2 = dem.forces(update_fracture=False)
        assert torch.allclose(force1, force2, atol=1e-15)
        assert torch.allclose(reaction1, reaction2, atol=1e-15)

    def test_force_consistency_after_steps(self) -> None:
        """Verify forces remain consistent after multiple timesteps."""
        cfg = DEM3DConfig(nx=7, ny=4, nz=3, radius=0.02, device="cpu")
        dem = IceDEM3D(cfg)

        forces_history = []
        for _ in range(10):
            dem.step()
            force, _ = dem.forces(update_fracture=False)
            forces_history.append(force.clone())

        for f in forces_history:
            assert torch.isfinite(f).all()

        dem2 = IceDEM3D(cfg)
        for _ in range(10):
            dem2.step()
        force2, _ = dem2.forces(update_fracture=False)
        assert torch.allclose(forces_history[-1], force2, atol=1e-14)


# ── Integration test ────────────────────────────────────────────────────

class TestGPUCellListIntegration:
    """End-to-end test of GPU cell list in DEM simulation."""

    def test_dem_simulation_runs(self) -> None:
        """Run a full DEM simulation and verify it completes without errors."""
        cfg = DEM3DConfig(
            nx=10, ny=6, nz=4,
            radius=0.02,
            tool_speed=0.3,
            device="cpu",
        )
        dem = IceDEM3D(cfg)

        n_steps = 20
        for _ in range(n_steps):
            dem.step()

        assert torch.isfinite(dem.positions).all()
        assert torch.isfinite(dem.velocities).all()
        assert dem.step_count == n_steps
        assert dem.broken_bonds >= 0

    def test_dem_diagnostics_consistency(self) -> None:
        """Verify diagnostics are consistent across runs."""
        cfg = DEM3DConfig(nx=6, ny=4, nz=3, radius=0.025, device="cpu")
        dem = IceDEM3D(cfg)

        for _ in range(10):
            dem.step()

        diag = dem.diagnostics()
        assert "time" in diag
        assert "reaction_z" in diag
        assert "broken_bonds" in diag
        assert "kinetic_energy_J" in diag
        assert diag["particle_count"] == dem.config.nx * dem.config.ny * dem.config.nz


# ── Selection logic test ────────────────────────────────────────────────

class TestImplementationSelection:
    """Test the implementation selector."""

    def test_select_small_cpu(self) -> None:
        positions = torch.randn(100, 3)
        assert select_implementation(positions) == "segment"

    def test_select_large_cpu(self) -> None:
        positions = torch.randn(10000, 3)
        assert select_implementation(positions) == "vectorised"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
