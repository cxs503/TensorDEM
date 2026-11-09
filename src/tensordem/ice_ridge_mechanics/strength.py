"""Confined-compression measurement of the rubble strength ``sigma_f``.

The analytic anchor of the campaign node is ``F_max = sigma_f A_keel``.  The
strength is *measured* on the same disk assembly (same particle size, friction
and porosity) instead of being assumed: a blocky sample is confined laterally by
a pressure ``p_c`` (applied as a body force on the boundary disks) and loaded
vertically by a platen.  The peak vertical stress is the major principal stress
at failure, which is the quantity that appears in the anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from scipy.spatial import cKDTree

from .engine2d import Assembly, IceRubbleMaterial, WallSegment

__all__ = ["RubbleStrengthConfig", "measure_rubble_strength"]


@dataclass(frozen=True)
class RubbleStrengthConfig:
    sample_width: float = 3.0
    sample_height: float = 2.0
    particle_radius: float = 0.12
    block_particles: int = 3
    confining_pressure: float = 4.0e4
    platen_velocity: float = 0.08
    strain: float = 0.25
    reference_strain: float = 0.10
    dt: float | None = None
    waterline: float = 0.0
    material: IceRubbleMaterial | None = None

    def __post_init__(self) -> None:
        for name in ("sample_width", "sample_height", "particle_radius",
                     "confining_pressure", "platen_velocity", "strain"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def material_or_default(self) -> IceRubbleMaterial:
        return self.material if self.material is not None else IceRubbleMaterial(
            particle_radius=self.particle_radius)

    @property
    def time_step(self) -> float:
        return float(self.dt) if self.dt is not None else \
            self.material_or_default.recommended_dt()


def _build_sample(config: RubbleStrengthConfig) -> dict:
    spacing = 2.0 * config.particle_radius * 1.02
    nb = config.block_particles
    nx = max(1, int(round(config.sample_width / (nb * spacing * 1.06))))
    ny = max(1, int(round(config.sample_height / (nb * spacing * 1.06))))
    positions: list[tuple[float, float]] = []
    blocks: list[int] = []
    bonds: list[tuple[int, int]] = []
    block = 0
    width = nx * nb * spacing * 1.06
    for bx in range(nx):
        for by in range(ny):
            first = len(positions)
            for r in range(nb):
                for c in range(nb):
                    positions.append((-0.5 * width + bx * nb * spacing * 1.06 + c * spacing,
                                      by * nb * spacing * 1.06 + r * spacing))
                    blocks.append(block)
            local = torch.tensor(positions[first:], dtype=torch.float64)
            tree = cKDTree(local.numpy())
            for i, j in tree.query_pairs(spacing * 1.45, output_type="ndarray"):
                bonds.append((first + int(i), first + int(j)))
            block += 1
    pos = torch.tensor(positions, dtype=torch.float64)
    radii = torch.full((pos.shape[0],), config.particle_radius, dtype=torch.float64)
    top = float(pos[:, 1].max() + config.particle_radius)
    return {"positions": pos, "radii": radii, "bonds": bonds, "top": top,
            "width": float(width), "blocks": block}


def measure_rubble_strength(config: RubbleStrengthConfig = RubbleStrengthConfig()
                            ) -> dict:
    """Peak vertical stress of a laterally confined rubble sample.

    Returns ``sigma_f`` (Pa), the confining pressure, the stress history and the
    sample geometry.  The mean contact pressure is also reported, because the
    choice of ``p_c`` is documented as the mean pressure measured in the ridge at
    the start of the penetration run (self-consistent anchor).
    """
    material = config.material_or_default
    sample = _build_sample(config)
    assembly = Assembly(sample["positions"], sample["radii"], material,
                        waterline=config.waterline)
    if sample["bonds"]:
        assembly.add_bonds(sample["bonds"])
    width = sample["width"]
    top = sample["top"]
    bottom = float(sample["positions"][:, 1].min() - config.particle_radius)
    low_wall = WallSegment(torch.tensor([-0.5 * width - 0.5, bottom]),
                           torch.tensor([-0.5 * width - 0.5, top + 1.0]),
                           torch.zeros(2, dtype=torch.float64), friction=0.0,
                           label="left")
    right_wall = WallSegment(torch.tensor([0.5 * width + 0.5, bottom]),
                             torch.tensor([0.5 * width + 0.5, top + 1.0]),
                             torch.zeros(2, dtype=torch.float64), friction=0.0,
                             label="right")
    floor = WallSegment(torch.tensor([-0.6 * width, bottom]),
                        torch.tensor([0.6 * width, bottom]),
                        torch.zeros(2, dtype=torch.float64), friction=0.3,
                        label="floor")
    platen = WallSegment(torch.tensor([-0.6 * width, top]),
                         torch.tensor([0.6 * width, top]),
                         torch.tensor([0.0, -config.platen_velocity],
                                      dtype=torch.float64), friction=0.3,
                         label="platen")
    for wall in (low_wall, right_wall, floor):
        assembly.add_wall(wall)
    assembly.add_wall(platen)
    dt = config.time_step
    steps = max(int(round(config.strain * sample_height_actual(sample, config)
                          / config.platen_velocity / dt)), 1)
    stress_history: list[dict] = []
    peak = 0.0
    peak_strain = 0.0
    reference_stress = 0.0
    reference_error = float("inf")
    initial_top = top
    report_every = max(steps // 40, 1)
    pressure_force = config.confining_pressure * 2.0 * config.particle_radius
    for step in range(steps):
        # lateral confining pressure, applied as an external body force on the
        # disks closest to each lateral boundary (a membrane boundary)
        x_min = float(assembly.pos[:, 0].min())
        x_max = float(assembly.pos[:, 0].max())
        left = assembly.pos[:, 0] < x_min + 1.2 * config.particle_radius
        right = assembly.pos[:, 0] > x_max - 1.2 * config.particle_radius
        assembly.external_force = torch.zeros_like(assembly.external_force)
        assembly.external_force[left, 0] += pressure_force
        assembly.external_force[right, 0] -= pressure_force
        assembly.step(dt)
        if (step + 1) % report_every == 0 or step == steps - 1:
            reaction = assembly.wall_reaction
            vertical = float(reaction[-1, 1]) if reaction.numel() else 0.0
            stress = abs(vertical) / width
            strain = max(initial_top - float(platen.p0[1]), 0.0) / sample["top"]
            stress_history.append({"time": assembly.time, "strain": strain,
                                   "stress": stress,
                                   "platen_force": vertical})
            if stress > peak:
                peak = stress
                peak_strain = strain
            if abs(strain - config.reference_strain) < reference_error:
                reference_error = abs(strain - config.reference_strain)
                reference_stress = stress
    mean_pressure = float(assembly.forces(update_memory=False)["normal_force"].mean()) \
        if assembly._pairs.numel() else 0.0
    # a frictional rubble hardens under confinement, so there is no peak: the
    # strength is quoted at the reference axial strain (10 % by default)
    sigma_f = reference_stress if reference_stress > 0.0 else peak
    return {
        "sigma_f": sigma_f,
        "stress_peak": peak,
        "reference_strain": config.reference_strain,
        "peak_strain": peak_strain,
        "confining_pressure": config.confining_pressure,
        "mean_contact_pressure": mean_pressure,
        "sample_width": width,
        "sample_particles": assembly.count,
        "blocks": sample["blocks"],
        "stress_history": stress_history,
        "steps": steps,
        "dt": dt,
    }


def sample_height_actual(sample: dict, config: RubbleStrengthConfig) -> float:
    """Initial sample height (m), used to convert strain into platen travel."""
    return float(sample["top"] - sample["positions"][:, 1].min())
