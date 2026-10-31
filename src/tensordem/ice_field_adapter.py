"""Adapter layer for upstream dependencies (V02, V03).

Upstream modules are in-flight (contract-parallel mode).  This adapter
abstracts their interfaces so that when they land, only this file needs
to change if the API drifts.

V02 (m705, TensorDEM): DEM solver scaling benchmarks
    - Expected interface: ScalingBenchmark.run(n_particles) -> dict
    - Fallback: local timing via ScaleUpIceField

V03 (m710, TensorDEM): GPU cell-list contact detection
    - Expected interface: GPUCellList.detect(positions, radius) -> pairs
    - Fallback: _vectorized_cell_list_pairs (CPU torch)
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# V03 adapter: GPU cell-list contact detection
# ---------------------------------------------------------------------------
@runtime_checkable
class ContactDetectionProtocol(Protocol):
    """Contract for contact-detection backends."""

    def detect(
        self,
        positions: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        """Return (M, 2) long tensor of candidate contact pairs (i < j)."""
        ...


class CPUCellListAdapter:
    """Fallback: use the vectorized CPU cell list from scale_up_ice_field."""

    def detect(self, positions: torch.Tensor, radius: float) -> torch.Tensor:
        from tensordem.scale_up_ice_field import _vectorized_cell_list_pairs
        return _vectorized_cell_list_pairs(
            positions, radius, positions.device, positions.dtype)


class GPUCellListAdapter:
    """Placeholder for V03 GPU cell list.

    When V03 lands, replace the body with:
        from tensordem.gpu_cell_list import GPUCellList
        return GPUCellList.detect(positions, radius)

    Contract (V03 acceptance criteria):
        gpu_speedup_vs_cpu >= 10.0 [ratio]
        force_consistency_gpu_vs_cpu < 1e-10 [N]
        pytest_test_gpu_cell_list == PASS [bool]
    """

    def detect(self, positions: torch.Tensor, radius: float) -> torch.Tensor:
        raise NotImplementedError(
            "GPUCellList not yet available (V03/m710 in-flight). "
            "Use CPUCellListAdapter as fallback.")


def get_contact_detector(device: str = "cpu") -> ContactDetectionProtocol:
    """Return the best available contact-detection backend."""
    if device == "cuda" and torch.cuda.is_available():
        try:
            return GPUCellListAdapter()
        except NotImplementedError:
            pass
    return CPUCellListAdapter()


# ---------------------------------------------------------------------------
# V02 adapter: DEM solver scaling benchmarks
# ---------------------------------------------------------------------------
@runtime_checkable
class ScalingBenchmarkProtocol(Protocol):
    """Contract for scaling-benchmark backends."""

    def run(self, n_particles: int, n_steps: int = 5) -> dict[str, Any]:
        """Run benchmark and return timing + accuracy metrics."""
        ...


class LocalScalingBenchmarkAdapter:
    """Fallback: use ScaleUpIceField for local scaling benchmarks."""

    def run(self, n_particles: int, n_steps: int = 5) -> dict[str, Any]:
        from tensordem.scale_up_ice_field import run_broken_ice_field
        # Approximate grid dimensions for target particle count
        nz = max(2, int(round(n_particles ** (1 / 3))))
        ny = max(2, int(round(n_particles ** (1 / 3))))
        nx = max(2, n_particles // (ny * nz))
        # Adjust to hit target more closely
        while nx * ny * nz < n_particles:
            nx += 1
        return run_broken_ice_field(nx=nx, ny=ny, nz=nz, n_steps=n_steps)


class UpstreamScalingBenchmarkAdapter:
    """Placeholder for V02 scaling benchmarks.

    When V02 lands, replace the body with:
        from tensordem.scaling_benchmark import ScalingBenchmark
        return ScalingBenchmark.run(n_particles, n_steps)

    Contract (V02 acceptance criteria):
        wallclock_100k_ratio_vs_ideal < 2.0 [ratio]
        energy_drift_per_step < 1e-06 [ratio]
        reproducible_hash == True [bool]
    """

    def run(self, n_particles: int, n_steps: int = 5) -> dict[str, Any]:
        raise NotImplementedError(
            "Upstream ScalingBenchmark not yet available (V02/m705 in-flight). "
            "Use LocalScalingBenchmarkAdapter as fallback.")


def get_scaling_benchmark() -> ScalingBenchmarkProtocol:
    """Return the best available scaling-benchmark backend."""
    try:
        return UpstreamScalingBenchmarkAdapter()
    except NotImplementedError:
        pass
    return LocalScalingBenchmarkAdapter()


# ---------------------------------------------------------------------------
# Interface contract documentation (for report JSON)
# ---------------------------------------------------------------------------
INTERFACE_CONTRACT = {
    "V03_gpu_cell_list": {
        "module": "tensordem.gpu_cell_list (in-flight m710)",
        "function": "GPUCellList.detect(positions: Tensor[N,3], radius: float) -> Tensor[M,2]",
        "units": "positions [m], radius [m], pairs are particle indices",
        "acceptance": {
            "gpu_speedup_vs_cpu": ">= 10.0 [ratio]",
            "force_consistency_gpu_vs_cpu": "< 1e-10 [N]",
            "pytest_test_gpu_cell_list": "== PASS [bool]",
        },
        "fallback": "tensordem.scale_up_ice_field._vectorized_cell_list_pairs",
    },
    "V02_scaling_benchmark": {
        "module": "tensordem.scaling_benchmark (in-flight m705)",
        "function": "ScalingBenchmark.run(n_particles: int, n_steps: int=5) -> dict",
        "output_keys": [
            "wallclock_per_step_mean [s]",
            "wallclock_100k_ratio_vs_ideal [ratio]",
            "energy_drift_per_step [ratio]",
            "reproducible_hash [bool]",
        ],
        "acceptance": {
            "wallclock_100k_ratio_vs_ideal": "< 2.0 [ratio]",
            "energy_drift_per_step": "< 1e-06 [ratio]",
            "reproducible_hash": "== True [bool]",
        },
        "fallback": "tensordem.scale_up_ice_field.run_broken_ice_field",
    },
}
