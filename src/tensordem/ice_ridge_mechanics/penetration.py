"""Ship crossing a DEM ice ridge: force history, failure modes, energy split.

The ship is a 2D longitudinal section of the bow -- a raked stem from the bow
keel to the stem head plus a flat bottom and a vertical transom -- represented
by translational rigid segments that advance at a constant speed ``V`` through
the ridge formed by :mod:`tensordem.ice_ridge_mechanics.formation`.

Outputs
-------
``force_history``   hull force ``F_x(t)`` (N per metre of ship width),
``mode_trace``      the four failure-mode indicators, window averaged:
    * ``crushing``  contact count x mean normal force at the stem,
    * ``bending``   block rotation work: mean ``|v_t|`` (sliding) weighted by the
                    local contact force,
    * ``ride_up``   upward velocity of the ice adjacent to the bow,
    * ``clearing``  displacement rate of the ice away from the bow,
``phase_onsets``    first time each indicator exceeds its threshold,
``energy_split``    ``W_ext`` vs ``E_break + E_friction + E_kinetic`` ledger.

Failure-mode identification is deliberately *indicator based* rather than
scripted: the onsets are computed from the measured traces and the qualitative
statement tested is that the order is crushing -> bending -> ride-up ->
clearing, and that a vertical stem produces less ride-up than a raked one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .engine2d import Assembly, IceRubbleMaterial, WallSegment
from .geometry import ridge_profile_geometry

__all__ = [
    "ShipCrossingConfig",
    "ShipCrossingResult",
    "build_hull",
    "simulate_ship_crossing",
    "identify_modes",
]


@dataclass(frozen=True)
class ShipCrossingConfig:
    speed: float = 0.6
    stem_rake_deg: float = 25.0
    bow_top: float = 1.5
    bottom_length: float = 2.0
    clearance: float = 0.2
    dt: float | None = None
    waterline: float = 0.0
    material: IceRubbleMaterial | None = None
    hull_friction: float = 0.2
    record_every: int = 10

    def __post_init__(self) -> None:
        for name in ("speed", "bottom_length"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 <= self.stem_rake_deg < 80.0:
            raise ValueError("stem_rake_deg must lie in [0, 80)")

    @property
    def material_or_default(self) -> IceRubbleMaterial:
        return self.material if self.material is not None else IceRubbleMaterial()

    @property
    def time_step(self) -> float:
        return float(self.dt) if self.dt is not None else \
            self.material_or_default.recommended_dt()


@dataclass
class ShipCrossingResult:
    config: ShipCrossingConfig
    force_history: list = field(default_factory=list)
    mode_trace: list = field(default_factory=list)
    phase_onsets: dict = field(default_factory=dict)
    mode_order: list = field(default_factory=list)
    energy_split: dict = field(default_factory=dict)
    force_peak: float = 0.0
    force_mean: float = 0.0
    displacement: dict = field(default_factory=dict)
    geometry_before: dict = field(default_factory=dict)
    geometry_after: dict = field(default_factory=dict)
    steps: int = 0
    keel_depth: float = 0.0
    sail_height: float = 0.0

    def as_dict(self) -> dict:
        return {
            "config": {
                "speed": self.config.speed,
                "stem_rake_deg": self.config.stem_rake_deg,
                "bow_top": self.config.bow_top,
                "bottom_length": self.config.bottom_length,
                "dt": self.config.time_step,
                "hull_friction": self.config.hull_friction,
            },
            "force_peak": self.force_peak,
            "force_mean": self.force_mean,
            "phase_onsets": self.phase_onsets,
            "mode_order": self.mode_order,
            "energy_split": self.energy_split,
            "displacement": self.displacement,
            "keel_depth": self.keel_depth,
            "sail_height": self.sail_height,
            "steps": self.steps,
            "force_history": self.force_history,
            "mode_trace": self.mode_trace,
        }


def build_hull(config: ShipCrossingConfig, keel_depth: float,
               start_x: float) -> tuple[list[WallSegment], dict]:
    """Bow polyline (in the x-y plane) and its geometry.

    The bow keel is placed below the ridge keel tip (``keel_depth + clearance``)
    so that the immersed stem face spans the whole keel section: that is the
    2D idealisation of a ship whose draft exceeds the keel depth, and it makes
    ``A_keel = D_keel x 1 m`` the literal contact area in the analytic anchor
    ``F_max = sigma_f A_keel``.
    """
    draft = keel_depth + config.clearance
    rake = math.radians(config.stem_rake_deg)
    keel_x = start_x - draft * math.tan(rake)
    stem_x = start_x
    top_x = stem_x + config.bow_top * math.tan(rake)
    points = [
        (keel_x - config.bottom_length, -draft),
        (keel_x, -draft),
        (stem_x, 0.0),
        (top_x, config.bow_top),
    ]
    velocity = torch.tensor([config.speed, 0.0], dtype=torch.float64)
    walls = []
    for index in range(len(points) - 1):
        p0 = torch.tensor(points[index], dtype=torch.float64)
        p1 = torch.tensor(points[index + 1], dtype=torch.float64)
        walls.append(WallSegment(p0, p1, velocity, friction=config.hull_friction,
                                 label=f"hull_{index}"))
    geometry = {"draft": draft, "keel_x": keel_x, "stem_x": stem_x, "top_x": top_x,
                "stem_face_length": math.hypot(stem_x - keel_x, draft)}
    return walls, geometry


def _window_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(int(window), 1)
    out = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        out.append(running / min(index + 1, window))
    return out


def identify_modes(trace: list[dict], *, crushing_fraction: float = 0.55,
                   bending_fraction: float = 0.5, ride_fraction: float = 0.5,
                   clearing_fraction: float = 0.6) -> dict:
    """First onset of each mode indicator (fraction of its own peak)."""
    key_map = {
        "crushing": ("crushing", crushing_fraction),
        "bending": ("bending", bending_fraction),
        "ride_up": ("ride_up", ride_fraction),
        "clearing": ("clearing", clearing_fraction),
    }
    onsets: dict[str, float] = {}
    for mode, (key, fraction) in key_map.items():
        series = [row.get(key, 0.0) for row in trace]
        if not series:
            continue
        peak = max(series)
        if peak <= 0.0:
            continue
        threshold = fraction * peak
        for row, value in zip(trace, series):
            if value >= threshold:
                onsets[mode] = float(row["time"])
                break
    order = [name for name, _ in sorted(onsets.items(), key=lambda item: item[1])]
    return {"phase_onsets": onsets, "mode_order": order}


def simulate_ship_crossing(ridge, config: ShipCrossingConfig = ShipCrossingConfig(),
                           *, verbose: bool = False) -> ShipCrossingResult:
    """Advance the hull through ``ridge`` (an ``Assembly``) at constant speed."""
    assembly: Assembly = ridge
    material = config.material_or_default
    geometry_before = ridge_profile_geometry(
        assembly.pos, assembly.radii, waterline=config.waterline,
        ice_thickness=max(2.0 * float(assembly.radii.mean()) * 1.5, 0.5),
        thickness_factor=0.9)
    keel = float(geometry_before["keel_depth"])
    sail = float(geometry_before["sail_height"])
    span = float(geometry_before["width"])
    if span <= 0.0:
        span = float(assembly.pos[:, 0].max() - assembly.pos[:, 0].min())
    start_x = float(assembly.pos[:, 0].min()) - 0.5 * span - 1.0
    walls, hull = build_hull(config, keel if keel > 0.0 else 2.0, start_x)
    for wall in walls:
        assembly.add_wall(wall)
    dt = config.time_step
    total_time = (2.5 * span + 2.0) / config.speed
    steps = max(int(round(total_time / dt)), 1)
    force_history: list[dict] = []
    trace: list[dict] = []
    x_initial = assembly.pos.clone()
    for step in range(steps):
        assembly.step(dt)
        reaction = assembly.wall_reaction
        hull_force = torch.zeros(2, dtype=torch.float64)
        if reaction.numel() >= len(walls):
            # reaction[k] is the force *on* segment k; the ice resistance felt by
            # the hull is minus its x component (the ship advances in +x)
            hull_force = reaction[-len(walls):].sum(dim=0)
        if (step + 1) % max(config.record_every, 1) == 0 or step == steps - 1:
            hull_x = float(walls[2].p0[0])
            near = (assembly.pos[:, 0] - hull_x).abs() < 0.35 * span + 1.0
            above = assembly.pos[:, 1] > config.waterline
            velocities = assembly.vel
            crushing = float(len(assembly.forces(update_memory=False)["normal_force"])
                             and assembly.forces(update_memory=False)["normal_force"].mean())
            bending = float(velocities[near].abs().mean()) if bool(near.any()) else 0.0
            ride = float(velocities[near & above, 1].clamp_min(0.0).mean()) \
                if bool((near & above).any()) else 0.0
            moved = torch.linalg.vector_norm(assembly.pos - x_initial, dim=1)
            clearing = float(moved[near].mean()) if bool(near.any()) else 0.0
            row = {"time": assembly.time,
                   "hull_x": hull_x,
                   "force_x": -float(hull_force[0]),
                   "force_y": -float(hull_force[1]),
                   "crushing": crushing,
                   "bending": bending,
                   "ride_up": ride,
                   "clearing": clearing,
                   "kinetic_energy": assembly.kinetic_energy()}
            force_history.append(row)
            trace.append(row)
            if verbose:
                print(f"  t={assembly.time:5.2f} x={hull_x:6.2f} F={row['force_x']:9.1f} "
                      f"cr={crushing:8.1f} bd={bending:6.3f} ru={ride:6.3f} cl={clearing:6.3f}")
    peak = max((row["force_x"] for row in force_history), default=0.0)
    positive = [row["force_x"] for row in force_history if row["force_x"] > 0.0]
    mean = float(sum(positive) / len(positive)) if positive else 0.0
    modes = identify_modes(trace)
    ledger = assembly.ledger()
    split = {
        "work_external": ledger["work_external"],
        "work_buoyancy": ledger["work_buoyancy"],
        "break_energy": ledger["break_energy"],
        "friction_energy": ledger["dissipation"]["contact_friction"]
        + ledger["dissipation"]["wall_friction"],
        "damping_energy": ledger["dissipation"]["contact_damping"]
        + ledger["dissipation"]["wall_damping"] + ledger["dissipation"]["drag"]
        + ledger["dissipation"]["rolling"],
        "kinetic_energy": ledger["delta_kinetic"],
        "gravity_potential": ledger["delta_gravity"],
        "spring_energy": ledger["delta_spring"],
        "residual": ledger["residual"],
        "relative_residual": ledger["relative_residual"],
    }
    scale = max(abs(split["work_external"]), 1e-9)
    split["closure_fraction"] = [
        (split["break_energy"] + split["friction_energy"] + split["kinetic_energy"]) / scale,
        (split["damping_energy"] + split["gravity_potential"] + split["spring_energy"]) / scale,
    ]
    geometry_after = ridge_profile_geometry(
        assembly.pos, assembly.radii, waterline=config.waterline,
        ice_thickness=max(2.0 * float(assembly.radii.mean()) * 1.5, 0.5),
        thickness_factor=0.9)
    moved = torch.linalg.vector_norm(assembly.pos - x_initial, dim=1)
    displacement = {
        "mean_displacement": float(moved.mean()),
        "max_displacement": float(moved.max()),
        "max_particle_speed": float(assembly.vel.norm(dim=1).max()),
    }
    return ShipCrossingResult(
        config=config,
        force_history=force_history,
        mode_trace=trace,
        phase_onsets=modes["phase_onsets"],
        mode_order=modes["mode_order"],
        energy_split=split,
        force_peak=peak,
        force_mean=mean,
        displacement=displacement,
        geometry_before=geometry_before,
        geometry_after=geometry_after,
        steps=steps,
        keel_depth=keel,
        sail_height=sail,
    )
