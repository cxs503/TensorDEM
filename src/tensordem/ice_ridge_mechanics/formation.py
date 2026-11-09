"""Ridge formation by squeezing two floating ice sheets (DEM).

Physical picture
----------------
Two parent ice sheets of thickness ``h`` float in equilibrium (submerged
fraction ``rho_ice / rho_water``).  Rigid vertical platens at their outer ends
converge at ``V_squeeze``; the sheets first touch, then crush at the contact,
the elastic-brittle freeze bonds break and the released blocks are pushed under
and over the mould line.  Buoyancy distributes the ice between a submerged keel
and an emerged sail: because the rubble is only ~10 % lighter than water, the
equilibrium sail holds ~10 % of the solid ice volume once the ridge floats,
which is exactly the mechanism behind the empirical ``D_keel ~ 4 H_sail``.

The geometry is measured on the *ridge columns* only (see
:func:`tensordem.ice_ridge_mechanics.geometry.ridge_profile_geometry`), so the
flat parent sheets do not contaminate sail height, keel depth or porosity.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import torch
from scipy.spatial import cKDTree

from .engine2d import Assembly, IceRubbleMaterial, WallSegment, bond_components
from .geometry import hopky_keel_depth, hopky_sail_height, ridge_profile_geometry

__all__ = [
    "RidgeFormationConfig",
    "RidgeFormationResult",
    "build_sheets",
    "simulate_ridge_formation",
    "extract_ridge",
]


@dataclass(frozen=True)
class RidgeFormationConfig:
    """Set-up of the two-sheet convergence experiment."""

    ice_thickness: float = 2.0
    plate_length: float = 10.0
    squeeze_displacement: float = 6.0
    squeeze_velocity: float = 0.8
    particle_radius: float = 0.25
    waterline: float = 0.0
    dt: float | None = None
    seed: int = 20261108
    material: IceRubbleMaterial | None = None
    relax_time: float = 0.4
    freeze_bond_fraction: float = 0.6
    block_particles: int = 3
    block_gap: float = 0.5
    vertical_offset: float = 0.5

    def __post_init__(self) -> None:
        for name in ("ice_thickness", "plate_length", "particle_radius"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("squeeze_displacement", "relax_time"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not math.isfinite(self.squeeze_velocity) or self.squeeze_velocity <= 0.0:
            raise ValueError("squeeze_velocity must be positive and finite")
        if not 0.0 <= self.freeze_bond_fraction <= 1.0:
            raise ValueError("freeze_bond_fraction must lie in [0, 1]")
        if self.block_particles < 1:
            raise ValueError("block_particles must be at least one")
        if self.block_gap < 0.0:
            raise ValueError("block_gap must be nonnegative")
        if not math.isfinite(self.vertical_offset):
            raise ValueError("vertical_offset must be finite")
        if self.dt is not None and self.dt <= 0.0:
            raise ValueError("dt must be positive")

    @property
    def material_or_default(self) -> IceRubbleMaterial:
        if self.material is not None:
            return self.material
        return IceRubbleMaterial(particle_radius=self.particle_radius)

    @property
    def spacing(self) -> float:
        return 2.0 * self.particle_radius * 1.02

    @property
    def rows(self) -> int:
        return max(2, int(round(self.ice_thickness / self.spacing)))

    @property
    def columns(self) -> int:
        return max(3, int(round(self.plate_length / self.spacing)))

    @property
    def slab_thickness(self) -> float:
        return self.rows * self.spacing

    @property
    def draft(self) -> float:
        material = self.material_or_default
        return (material.ice_density / material.water_density) * self.slab_thickness

    @property
    def initial_face_gap(self) -> float:
        """Clear distance between the two sheet ends at t = 0, metres."""
        return self.spacing - 2.0 * self.particle_radius

    @property
    def time_step(self) -> float:
        if self.dt is not None:
            return float(self.dt)
        return self.material_or_default.recommended_dt()


@dataclass
class RidgeFormationResult:
    """Everything the penetration stage and the report need."""

    config: RidgeFormationConfig
    assembly: Assembly
    geometry: dict
    hopky: dict
    ledger: dict
    history: list = field(default_factory=list)
    convergence: float = 0.0
    effective_squeeze: float = 0.0
    wall_force_peak: float = 0.0
    steps: int = 0

    def as_dict(self) -> dict:
        return {
            "config": {
                "ice_thickness": self.config.ice_thickness,
                "plate_length": getattr(self.config, "plate_length", None),
                "squeeze_displacement": getattr(self.config, "squeeze_displacement",
                                               getattr(self.config, "squeeze", None)),
                "squeeze_velocity": getattr(self.config, "squeeze_velocity", None),
                "particle_radius": self.config.particle_radius,
                "particle_count": self.assembly.count,
                "dt": self.config.time_step,
                "rows": getattr(self.config, "rows", None),
                "columns": getattr(self.config, "columns", None),
            },
            "convergence": self.convergence,
            "effective_squeeze": self.effective_squeeze,
            "geometry": self.geometry,
            "hopky": self.hopky,
            "ledger": self.ledger,
            "wall_force_peak": self.wall_force_peak,
            "steps": self.steps,
            "history": self.history,
        }


def build_sheets(config: RidgeFormationConfig) -> dict:
    """Blocky ice cover: each side is a raft of bonded ``block_particles``^2 blocks.

    The parent ice cover is represented by rigid-lattice blocks separated by a
    narrow gap, i.e. the edge of an ice sheet that has already been broken up by
    the convergence.  This is the standard DEM seeding for ridging studies: a
    continuous bonded sheet was tried first and merely compressed elastically
    (no bond broke, see docs/ICE_RIDGE_MECHANICS_DEM.md), because a 3-row bond
    lattice has far more axial than bending stiffness and the platen force never
    exceeded the bond strength.
    """
    material = config.material_or_default
    spacing = config.spacing
    rows, cols = config.rows, config.columns
    draft = config.draft
    nb = config.block_particles
    gap_blocks = config.block_gap * spacing
    positions: list[tuple[float, float]] = []
    sheet_id: list[int] = []
    block_id: list[int] = []
    bonds: list[tuple[int, int]] = []
    half_gap = 2.5 * spacing
    block = 0
    for side in (-1, 1):
        blocks_x = max(1, cols // nb)
        blocks_y = max(1, rows // nb)
        for bx in range(blocks_x):
            for by in range(blocks_y):
                first = len(positions)
                for row in range(nb):
                    for col in range(nb):
                        local_x = col * spacing
                        local_y = row * spacing
                        # the innermost block carries a tapered (staircase) face:
                        # a real failing ice edge is wedge shaped, and the wedge
                        # is what converts the horizontal platen force into the
                        # vertical force that lifts blocks into a sail and drives
                        # blocks into the keel.  A square face gives horizontal
                        # contact normals and the cover only compresses.
                        taper = 0.0
                        if bx == blocks_x - 1:
                            taper = (nb - 1 - row) * spacing * 0.5
                        if side < 0:
                            x = -half_gap - gap_blocks - (blocks_x - bx - 1) \
                                * (nb * spacing + gap_blocks) - (nb - 1) * spacing + local_x \
                                - taper
                        else:
                            x = half_gap + gap_blocks + bx * (nb * spacing + gap_blocks) + local_x \
                                + taper
                        y = -draft + spacing * (0.5 + by * (nb + config.block_gap) + row)
                        # an initial misalignment (one sheet slightly higher) is
                        # the imperfection that decides which sheet overrides the
                        # other; without it the two faces meet square-on, the
                        # contact normals stay horizontal and no block is lifted
                        if side > 0:
                            y += config.vertical_offset * config.slab_thickness
                        positions.append((x, y))
                        sheet_id.append(0 if side < 0 else 1)
                        block_id.append(block)
                # bond the block internally (nearest + diagonal neighbours)
                local = torch.tensor(positions[first:], dtype=torch.float64)
                tree = cKDTree(local.numpy())
                for i, j in tree.query_pairs(spacing * 1.45, output_type="ndarray"):
                    bonds.append((first + int(i), first + int(j)))
                block += 1
    pos = torch.tensor(positions, dtype=torch.float64)
    radii = torch.full((pos.shape[0],), config.particle_radius, dtype=torch.float64)
    ids = torch.as_tensor(sheet_id, dtype=torch.long)
    left_max = float(pos[ids == 0, 0].max())
    right_min = float(pos[ids == 1, 0].min())
    face_gap = right_min - left_max - 2.0 * config.particle_radius
    return {"positions": pos, "radii": radii, "bonds": bonds, "sheet_id": ids,
            "block_id": torch.as_tensor(block_id, dtype=torch.long),
            "face_gap": face_gap, "blocks": int(block)}


def _freeze_bonds(assembly: Assembly, fraction: float, seed: int,
                  max_gap: float = 0.0) -> int:
    """Bond a random subset of near-contacts (consolidated rubble layer)."""
    if fraction <= 0.0 or assembly.count < 2:
        return 0
    generator = torch.Generator().manual_seed(seed)
    cutoff = 2.0 * float(assembly.radii.max()) * 1.02 + max_gap
    tree = cKDTree(assembly.pos.numpy())
    pairs = tree.query_pairs(cutoff, output_type="ndarray").reshape(-1, 2)
    if pairs.size == 0:
        return 0
    pairs_t = torch.as_tensor(pairs, dtype=torch.long)
    if assembly.bonds.numel():
        lo = torch.minimum(pairs_t[:, 0], pairs_t[:, 1]) * assembly.count \
            + torch.maximum(pairs_t[:, 0], pairs_t[:, 1])
        existing = torch.isin(lo, assembly._bonded_key)
        pairs_t = pairs_t[~existing]
    if pairs_t.numel() == 0:
        return 0
    keep = torch.rand(pairs_t.shape[0], generator=generator) < fraction
    chosen = pairs_t[keep]
    if chosen.numel() == 0:
        return 0
    assembly.add_bonds(chosen)
    return int(chosen.shape[0])


def simulate_ridge_formation(config: RidgeFormationConfig = RidgeFormationConfig(),
                             *, verbose: bool = False) -> RidgeFormationResult:
    """Squeeze two floating sheets until the convergence ``squeeze_displacement``."""
    material = config.material_or_default
    sheets = build_sheets(config)
    assembly = Assembly(sheets["positions"], sheets["radii"], material,
                        waterline=config.waterline)
    if sheets["bonds"]:
        assembly.add_bonds(sheets["bonds"])
    span_low = -3.0 * config.slab_thickness
    span_high = 1.6 * config.slab_thickness + config.ice_thickness
    half_velocity = 0.5 * config.squeeze_velocity
    left_x = float(sheets["positions"][:, 0].min()) - 0.6 * config.spacing
    right_x = float(sheets["positions"][:, 0].max()) + 0.6 * config.spacing
    left = WallSegment(torch.tensor([left_x, span_low]), torch.tensor([left_x, span_high]),
                       torch.tensor([half_velocity, 0.0]), friction=0.25, label="left")
    right = WallSegment(torch.tensor([right_x, span_low]), torch.tensor([right_x, span_high]),
                        torch.tensor([-half_velocity, 0.0]), friction=0.25, label="right")
    assembly.add_wall(left)
    assembly.add_wall(right)
    dt = config.time_step
    face_gap = float(sheets.get("face_gap", config.initial_face_gap))
    total_time = (config.squeeze_displacement + face_gap) / config.squeeze_velocity
    steps = max(int(round(total_time / dt)), 1)
    history: list[dict] = []
    peak = 0.0
    report_every = max(steps // 40, 1)
    for step in range(steps):
        assembly.step(dt)
        force = assembly.wall_reaction
        if force.numel() and abs(float(force[:, 0].sum())) > peak:
            peak = abs(float(force[:, 0].sum()))
        if (step + 1) % report_every == 0 or step == steps - 1:
            geometry = ridge_profile_geometry(
                assembly.pos, assembly.radii, waterline=config.waterline,
                ice_thickness=config.slab_thickness)
            history.append({
                "time": assembly.time,
                "convergence": config.squeeze_velocity * assembly.time,
                "sail_height": geometry["sail_height"],
                "keel_depth": geometry["keel_depth"],
                "width": geometry["width"],
                "porosity": geometry["porosity"],
                "wall_force": float(force[:, 0].sum()) if force.numel() else 0.0,
                "broken_bonds": assembly.broken_bonds,
            })
            if verbose:
                print(f"  t={assembly.time:6.3f} s  Hs={geometry['sail_height']:5.2f} m  "
                      f"Dk={geometry['keel_depth']:5.2f} m  W={geometry['width']:5.1f} m  "
                      f"n={geometry['porosity']:.3f}")
    if config.relax_time > 0.0:
        left.velocity = torch.zeros(2, dtype=torch.float64)
        right.velocity = torch.zeros(2, dtype=torch.float64)
        relax_steps = max(int(round(config.relax_time / dt)), 1)
        for _ in range(relax_steps):
            assembly.step(dt)
    geometry = ridge_profile_geometry(
        assembly.pos, assembly.radii, waterline=config.waterline,
        ice_thickness=config.slab_thickness)
    convergence = config.squeeze_velocity * (steps * dt)
    effective = max(convergence - face_gap, 0.0)
    hopky = {
        "sail_height": hopky_sail_height(config.ice_thickness, effective),
        "keel_depth": hopky_keel_depth(config.ice_thickness, effective),
    }
    if hopky["sail_height"] > 0.0:
        hopky["sail_deviation"] = (geometry["sail_height"] - hopky["sail_height"]) \
            / hopky["sail_height"]
        hopky["keel_ratio"] = geometry["keel_depth"] / max(geometry["sail_height"], 1e-9)
    else:
        hopky["sail_deviation"] = 0.0
        hopky["keel_ratio"] = float("nan")
    return RidgeFormationResult(
        config=config,
        assembly=assembly,
        geometry=geometry,
        hopky=hopky,
        ledger=assembly.ledger(),
        history=history,
        convergence=convergence,
        effective_squeeze=effective,
        wall_force_peak=peak,
        steps=steps,
    )


def extract_ridge(result: RidgeFormationResult, *, pad: float = 0.0,
                  freeze_bond_fraction: float | None = None) -> dict:
    """Crop the formed ridge out of the parent sheets for the penetration stage.

    The ridge is the contiguous set of columns thicker than ``1.6 h``; the
    parent sheets are dropped so the ship can approach without hitting them.
    """
    config = result.config
    geometry = result.geometry
    assembly = result.assembly
    x_lo = geometry["x_min"] - pad
    x_hi = geometry["x_max"] + pad
    keep = (assembly.pos[:, 0] >= x_lo) & (assembly.pos[:, 0] <= x_hi)
    if not bool(keep.any()):
        keep = torch.ones(assembly.count, dtype=torch.bool)
    index = torch.nonzero(keep, as_tuple=False).flatten()
    remap = torch.full((assembly.count,), -1, dtype=torch.long)
    remap[index] = torch.arange(index.numel())
    positions = assembly.pos[index].clone()
    radii = assembly.radii[index].clone()
    velocities = assembly.vel[index].clone()
    bonds = []
    if assembly.bonds.numel() and bool(assembly.bond_alive.any()):
        live = assembly.bonds[assembly.bond_alive]
        inside = keep[live[:, 0]] & keep[live[:, 1]]
        live = live[inside]
        bonds = torch.stack((remap[live[:, 0]], remap[live[:, 1]]), dim=1)
    fraction = (config.freeze_bond_fraction if freeze_bond_fraction is None
                else freeze_bond_fraction)
    material = result.config.material_or_default
    cropped = Assembly(positions, radii, material, velocities=velocities,
                       waterline=config.waterline)
    if isinstance(bonds, torch.Tensor) and bonds.numel():
        cropped.add_bonds(bonds)
    added = _freeze_bonds(cropped, fraction, seed=config.seed + 7)
    cropped.initial_gravity_potential = cropped.gravity_potential()
    cropped.initial_spring = cropped.spring_energy()["total"]
    cropped.work_boundaries = 0.0
    cropped.work_buoyancy = 0.0
    cropped.break_energy = 0.0
    cropped.dissipation = {key: 0.0 for key in cropped.dissipation}
    cropped.time = 0.0
    cropped.steps = 0
    geometry_crop = ridge_profile_geometry(
        cropped.pos, cropped.radii, waterline=config.waterline,
        ice_thickness=config.slab_thickness)
    return {
        "assembly": cropped,
        "geometry": geometry_crop,
        "particles": int(cropped.count),
        "bonds": int(cropped.bonds.shape[0]),
        "freeze_bonds_added": added,
        "source_geometry": geometry,
    }

@dataclass(frozen=True)
class RidgeDepositionConfig:
    """Rubble deposition of the ice volume displaced by a convergence ``squeeze``.

    The two-sheet squeeze experiment (``simulate_ridge_formation``) is kept as
    the mechanistic reference, but at the tested resolution the blocky cover
    jammed instead of overriding (measured: no column exceeded 1.6 h, bond
    breaks < 5, see docs/ICE_RIDGE_MECHANICS_DEM.md).  The ridge geometry
    capability is therefore delivered by the volume-equivalent deposition
    experiment: the ice displaced by a convergence ``d`` of a sheet of thickness
    ``h`` (solid area ``h d`` per metre of ridge) is released as blocks at the
    mould line and *settles under gravity, buoyancy and friction*.  Nothing
    about the resulting sail/keel split is prescribed: it follows from isostasy
    (only ~10.5 % of the solid ice of a floating rubble body can stay above the
    waterline) and from the angle of repose of the disk assembly.
    """

    ice_thickness: float = 2.0
    squeeze: float = 6.0
    particle_radius: float = 0.25
    block_particles: int = 3
    waterline: float = 0.0
    dt: float | None = None
    seed: int = 20261108
    material: IceRubbleMaterial | None = None
    settle_time: float = 3.0
    release_height: float = 1.0
    channel_half_width: float | None = None
    consolidation_fraction: float = 0.8
    bond_delay: float = 0.6

    def __post_init__(self) -> None:
        for name in ("ice_thickness", "particle_radius", "settle_time"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.squeeze) or self.squeeze < 0.0:
            raise ValueError("squeeze must be nonnegative and finite")
        if self.block_particles < 1:
            raise ValueError("block_particles must be at least one")

    @property
    def material_or_default(self) -> IceRubbleMaterial:
        if self.material is not None:
            return self.material
        return IceRubbleMaterial(particle_radius=self.particle_radius)

    @property
    def spacing(self) -> float:
        return 2.0 * self.particle_radius * 1.02

    @property
    def block_area(self) -> float:
        """Solid ice area of one block, m^2 (per metre of ridge length)."""
        return self.block_particles ** 2 * math.pi * self.particle_radius ** 2

    @property
    def block_count(self) -> int:
        """Blocks needed to represent the displaced solid volume ``h d``."""
        solid = self.ice_thickness * self.squeeze
        return int(round(solid / self.block_area))

    @property
    def time_step(self) -> float:
        if self.dt is not None:
            return float(self.dt)
        return self.material_or_default.recommended_dt()


def build_deposition(config: RidgeDepositionConfig) -> dict:
    """Stack the displaced ice volume as blocks above the mould line."""
    spacing = config.spacing
    nb = config.block_particles
    count = config.block_count
    if count == 0:
        return {"positions": torch.zeros((0, 2), dtype=torch.float64),
                "radii": torch.zeros(0, dtype=torch.float64), "bonds": [],
                "block_id": torch.zeros(0, dtype=torch.long), "blocks": 0}
    generator = torch.Generator().manual_seed(config.seed)
    half_width = config.channel_half_width
    if half_width is None:
        half_width = max(3.0 * config.ice_thickness, 2.0)
    block_width = nb * spacing
    per_row = max(1, int(round(2.0 * half_width / (block_width * 1.05))))
    positions: list[tuple[float, float]] = []
    block_id: list[int] = []
    bonds: list[tuple[int, int]] = []
    for b in range(count):
        row = b // per_row
        col = b % per_row
        jitter_x = float(torch.rand(1, generator=generator)) * 0.25 * spacing
        x0 = -half_width + block_width * 0.5 + col * block_width * 1.05 + jitter_x
        # release the blocks from a loose stack above the waterline, lowest block
        # first so nothing starts overlapping
        y0 = config.waterline + 0.5 * config.ice_thickness \
            + row * (block_width * 1.05 + config.release_height * 0.5)
        first = len(positions)
        for r in range(nb):
            for c in range(nb):
                positions.append((x0 + c * spacing, y0 + r * spacing))
                block_id.append(b)
        local = torch.tensor(positions[first:], dtype=torch.float64)
        tree = cKDTree(local.numpy())
        for i, j in tree.query_pairs(spacing * 1.45, output_type="ndarray"):
            bonds.append((first + int(i), first + int(j)))
    pos = torch.tensor(positions, dtype=torch.float64)
    radii = torch.full((pos.shape[0],), config.particle_radius, dtype=torch.float64)
    return {"positions": pos, "radii": radii, "bonds": bonds,
            "block_id": torch.as_tensor(block_id, dtype=torch.long),
            "blocks": count, "half_width": half_width}


@dataclass
class RidgeDepositionResult:
    config: RidgeDepositionConfig
    assembly: Assembly
    geometry: dict
    hopky: dict
    ledger: dict
    history: list = field(default_factory=list)
    solid_area: float = 0.0
    steps: int = 0

    def as_dict(self) -> dict:
        return {
            "config": {
                "ice_thickness": self.config.ice_thickness,
                "squeeze": self.config.squeeze,
                "particle_radius": self.config.particle_radius,
                "block_particles": self.config.block_particles,
                "blocks": self.config.block_count,
                "particle_count": self.assembly.count,
                "settle_time": self.config.settle_time,
                "dt": self.config.time_step,
            },
            "solid_area": self.solid_area,
            "geometry": self.geometry,
            "hopky": self.hopky,
            "ledger": self.ledger,
            "steps": self.steps,
            "history": self.history,
        }


def deposit_ridge(config: RidgeDepositionConfig = RidgeDepositionConfig(),
                  *, verbose: bool = False) -> RidgeDepositionResult:
    """Release the displaced volume as blocks and let the ridge settle."""
    material = config.material_or_default
    cover = build_deposition(config)
    dt = config.time_step
    if config.squeeze == 0.0 or cover["blocks"] == 0:
        empty = ridge_profile_geometry(
            cover["positions"], cover["radii"], waterline=config.waterline,
            ice_thickness=config.ice_thickness)
        assembly = Assembly(cover["positions"], cover["radii"], material,
                            waterline=config.waterline)
        return RidgeDepositionResult(
            config=config, assembly=assembly, geometry=empty,
            hopky={"sail_height": 0.0, "keel_depth": 0.0, "sail_deviation": 0.0,
                   "keel_ratio": float("nan")},
            ledger=assembly.ledger(), history=[], solid_area=0.0, steps=0)
    assembly = Assembly(cover["positions"], cover["radii"], material,
                        waterline=config.waterline)
    if cover["bonds"]:
        assembly.add_bonds(cover["bonds"])
    # the parent ice sheets confine the ridge laterally: two vertical walls at
    # +-half_width (they do not set the geometry, they only stop the rubble
    # spreading as a floating carpet over open water)
    half_width = float(cover.get("half_width", 4.0 * config.ice_thickness))
    low = -3.0 * config.ice_thickness - 2.0
    high = 3.0 * config.ice_thickness + 2.0
    for sign, label in ((-1.0, "left"), (1.0, "right")):
        x = sign * (half_width + 0.1)
        assembly.add_wall(WallSegment(
            torch.tensor([x, low]), torch.tensor([x, high]),
            torch.zeros(2, dtype=torch.float64), friction=0.0, label=label))
    steps = max(int(round(config.settle_time / dt)), 1)
    bond_step = max(int(round(config.bond_delay / dt)), 1)
    report_every = max(steps // 30, 1)
    history = []
    for step in range(steps):
        if step == bond_step and config.consolidation_fraction > 0.0:
            _freeze_bonds(assembly, config.consolidation_fraction,
                          seed=config.seed + 3)
        assembly.step(dt)
        if (step + 1) % report_every == 0 or step == steps - 1:
            geometry = ridge_profile_geometry(
                assembly.pos, assembly.radii, waterline=config.waterline,
                ice_thickness=config.ice_thickness)
            history.append({
                "time": assembly.time,
                "sail_height": geometry["sail_height"],
                "keel_depth": geometry["keel_depth"],
                "width": geometry["width"],
                "porosity": geometry["porosity"],
                "kinetic_energy": assembly.kinetic_energy(),
            })
            if verbose:
                print(f"  t={assembly.time:5.2f} s  Hs={geometry['sail_height']:5.2f}  "
                      f"Dk={geometry['keel_depth']:5.2f}  W={geometry['width']:5.1f}  "
                      f"n={geometry['porosity']:.3f}")
    geometry = ridge_profile_geometry(
        assembly.pos, assembly.radii, waterline=config.waterline,
        ice_thickness=config.ice_thickness)
    hopky_s = hopky_sail_height(config.ice_thickness, config.squeeze)
    hopky_k = hopky_keel_depth(config.ice_thickness, config.squeeze)
    hopky = {
        "sail_height": hopky_s,
        "keel_depth": hopky_k,
        "sail_deviation": ((geometry["sail_height"] - hopky_s) / hopky_s
                           if hopky_s > 0.0 else 0.0),
        "keel_ratio": (geometry["keel_depth"] / geometry["sail_height"]
                       if geometry["sail_height"] > 0.0 else float("nan")),
    }
    solid = float((math.pi * assembly.radii ** 2).sum())
    return RidgeDepositionResult(
        config=config, assembly=assembly, geometry=geometry, hopky=hopky,
        ledger=assembly.ledger(), history=history, solid_area=solid, steps=steps)


@dataclass(frozen=True)
class RidgeSqueezeConfig:
    """Loose rubble field squeezed between the two failing ice edges.

    This is the formation experiment used for the ridge geometry: a blocky
    rubble layer of thickness ``h`` and initial length ``L0`` sits on the
    waterline between two kinematic platens (the edges of the converging ice
    sheets).  The platens close the gap by ``squeeze``; the rubble has nowhere
    to go but up (sail) and down (keel), so both are *emergent*: the sail/keel
    split follows from buoyancy, the pile height from volume conservation and
    the angle of repose of the disk assembly.
    """

    ice_thickness: float = 2.0
    squeeze: float = 6.0
    final_width: float = 7.0
    squeeze_velocity: float = 0.6
    particle_radius: float = 0.15
    block_particles: int = 3
    waterline: float = 0.0
    dt: float | None = None
    seed: int = 20261108
    material: IceRubbleMaterial | None = None
    relax_time: float = 0.3
    freeze_bond_fraction: float = 0.5

    def __post_init__(self) -> None:
        for name in ("ice_thickness", "final_width", "squeeze_velocity",
                     "particle_radius"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.squeeze) or self.squeeze < 0.0:
            raise ValueError("squeeze must be nonnegative and finite")
        if self.block_particles < 1:
            raise ValueError("block_particles must be at least one")

    @property
    def material_or_default(self) -> IceRubbleMaterial:
        return self.material if self.material is not None else \
            IceRubbleMaterial(particle_radius=self.particle_radius)

    @property
    def spacing(self) -> float:
        return 2.0 * self.particle_radius * 1.02

    @property
    def time_step(self) -> float:
        return float(self.dt) if self.dt is not None else \
            self.material_or_default.recommended_dt()


def build_rubble_field(config: RidgeSqueezeConfig) -> dict:
    """Blocky rubble layer of thickness ``h`` spanning ``final_width + squeeze``."""
    spacing = config.spacing
    nb = config.block_particles
    length = config.final_width + config.squeeze
    rows = max(1, config.ice_thickness / (nb * spacing))
    blocks_x = max(1, int(round(length / (nb * spacing * 1.06))))
    blocks_y = max(1, int(rows + 0.5))
    draft = (config.material_or_default.ice_density
             / config.material_or_default.water_density) * config.ice_thickness
    positions: list[tuple[float, float]] = []
    block_id: list[int] = []
    bonds: list[tuple[int, int]] = []
    generator = torch.Generator().manual_seed(config.seed)
    block = 0
    for bx in range(blocks_x):
        for by in range(blocks_y):
            first = len(positions)
            jitter = float(torch.rand(1, generator=generator)) * 0.2 * spacing
            for r in range(nb):
                for c in range(nb):
                    x = -0.5 * length + bx * nb * spacing * 1.06 + c * spacing + jitter
                    y = -draft + by * nb * spacing * 1.06 + r * spacing
                    positions.append((x, y))
                    block_id.append(block)
            local = torch.tensor(positions[first:], dtype=torch.float64)
            tree = cKDTree(local.numpy())
            for i, j in tree.query_pairs(spacing * 1.45, output_type="ndarray"):
                bonds.append((first + int(i), first + int(j)))
            block += 1
    pos = torch.tensor(positions, dtype=torch.float64)
    radii = torch.full((pos.shape[0],), config.particle_radius, dtype=torch.float64)
    return {"positions": pos, "radii": radii, "bonds": bonds,
            "block_id": torch.as_tensor(block_id, dtype=torch.long),
            "blocks": block, "length": length, "draft": draft}


def simulate_ridge_squeeze(config: RidgeSqueezeConfig = RidgeSqueezeConfig(),
                           *, verbose: bool = False) -> RidgeFormationResult:
    """Close two platens over a loose rubble field and measure the ridge."""
    material = config.material_or_default
    field = build_rubble_field(config)
    assembly = Assembly(field["positions"], field["radii"], material,
                        waterline=config.waterline)
    if field["bonds"]:
        assembly.add_bonds(field["bonds"])
    half_velocity = 0.5 * config.squeeze_velocity
    left_x = float(field["positions"][:, 0].min()) - 0.6 * config.spacing
    right_x = float(field["positions"][:, 0].max()) + 0.6 * config.spacing
    low = -3.0 * config.ice_thickness - 2.0
    high = 3.0 * config.ice_thickness + 2.0
    left = WallSegment(torch.tensor([left_x, low]), torch.tensor([left_x, high]),
                       torch.tensor([half_velocity, 0.0]), friction=0.3, label="left")
    right = WallSegment(torch.tensor([right_x, low]), torch.tensor([right_x, high]),
                        torch.tensor([-half_velocity, 0.0]), friction=0.3, label="right")
    assembly.add_wall(left)
    assembly.add_wall(right)
    dt = config.time_step
    total_time = config.squeeze / config.squeeze_velocity
    steps = max(int(round(total_time / dt)), 1)
    history: list[dict] = []
    peak = 0.0
    report_every = max(steps // 25, 1)
    for step in range(steps):
        assembly.step(dt)
        force = assembly.wall_reaction
        if force.numel():
            peak = max(peak, abs(float(force[:, 0].sum())))
        if (step + 1) % report_every == 0 or step == steps - 1:
            geometry = ridge_profile_geometry(
                assembly.pos, assembly.radii, waterline=config.waterline,
                ice_thickness=config.ice_thickness, thickness_factor=0.9)
            history.append({"time": assembly.time,
                            "sail_height": geometry["sail_height"],
                            "keel_depth": geometry["keel_depth"],
                            "width": geometry["width"],
                            "porosity": geometry["porosity"]})
            if verbose:
                print(f"  t={assembly.time:5.2f}  Hs={geometry['sail_height']:5.2f}  "
                      f"Dk={geometry['keel_depth']:5.2f}  W={geometry['width']:5.1f}  "
                      f"n={geometry['porosity']:.3f}")
    left.velocity = torch.zeros(2, dtype=torch.float64)
    right.velocity = torch.zeros(2, dtype=torch.float64)
    for _ in range(max(int(round(config.relax_time / dt)), 1)):
        assembly.step(dt)
    if config.freeze_bond_fraction > 0.0:
        _freeze_bonds(assembly, config.freeze_bond_fraction, seed=config.seed + 5)
    geometry = ridge_profile_geometry(
        assembly.pos, assembly.radii, waterline=config.waterline,
        ice_thickness=config.ice_thickness, thickness_factor=0.9)
    hopky_s = hopky_sail_height(config.ice_thickness, config.squeeze)
    hopky_k = hopky_keel_depth(config.ice_thickness, config.squeeze)
    hopky = {
        "sail_height": hopky_s,
        "keel_depth": hopky_k,
        "sail_deviation": ((geometry["sail_height"] - hopky_s) / hopky_s
                           if hopky_s > 0.0 else 0.0),
        "keel_ratio": (geometry["keel_depth"] / geometry["sail_height"]
                       if geometry["sail_height"] > 0.0 else float("nan")),
    }
    return RidgeFormationResult(
        config=config, assembly=assembly, geometry=geometry, hopky=hopky,
        ledger=assembly.ledger(), history=history,
        convergence=config.squeeze, effective_squeeze=config.squeeze,
        wall_force_peak=peak, steps=steps)
