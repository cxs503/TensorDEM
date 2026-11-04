"""Multi-event platen bounce chain: repeated-impact closure of the contact law.

Why this module exists
----------------------
Every restitution verification in this repository (``scripts/verify_dem3d_
restitution.py``, ``scripts/calibrate_dem3d_restitution.py``,
``tensordem.restitution.platen_bounce_experiment``) is a **single** impact: the
engine is asked for one contact episode and the measured ``e`` is compared with
the reference solution of one episode.  A single impact cannot see a defect
class that only appears when the *same* contact law is engaged repeatedly:

* contact state that is not fully released (residual overlap, stale normal /
  tangential accumulator), so the next episode starts from a non-zero history;
* a dashpot that keeps acting outside the contact interval, which is
  invisible in one impact but compounds over a chain;
* mismatch between the clipped law's speed dependence and the engine's, which
  propagates multiplicatively: a 1 % error per bounce becomes 6 % over six
  bounces.

The bounce chain makes the engine hit an infinite-mass plane ``N`` times in a
row.  With no gravity and no drag the free flight between episodes is exactly
force-free, so the chain is a *pure* test of the contact branch: the only
physics in the problem is the clipped law ``F = max(0, k*delta - c*dt)``, whose
per-episode reference solution is available in closed numerical form
(:func:`tensordem.restitution.unilateral_contact_detail`).

Model and analytic chain
------------------------
For a rigid plane the effective mass is the particle mass ``m`` (the partner
does not accelerate), so the reference restitution ``e(c, k, m, v_in)`` is the
:func:`~tensordem.restitution.unilateral_restitution` value.  Because the
clipped law is only weakly velocity dependent over the tested range, the chain
follows approximately

    v_out,n = e * v_in,n,     v_in,n = v_out,n-1
    =>  v_n = e**n * v_0,      T_n = 2 * L / v_n

where ``L`` is the free travel distance between the two planes (contact is
instantaneous on the ballistic scale).  The module reports both the
*per-episode* closure error (engine vs reference at the measured impact speed)
and the *chain* error (engine vs ``e**n v_0``), because they fail for different
reasons: the first is a law-fidelity error, the second is an accumulation /
state-release error.

Geometry convention (inherited from :class:`tensordem.dem3d.IceDEM3D`)
-----------------------------------------------------------------------
The planes are horizontal: the bottom plane sits at ``z_b`` and contacts the
particle when ``z - r < z_b``; the top plane sits at ``z_t`` and contacts when
``z + r > z_t``.  Both planes apply ``F = max(0, k*delta - c*v_n)`` where ``v_n``
is the approach rate, i.e. the same clipped unilateral law the pair branch uses.

This module does not modify the solver: it only drives ``IceDEM3D`` and reads
back its state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import torch

from .dem3d import DEM3DConfig, IceDEM3D
from .restitution import classical_restitution, unilateral_contact_detail

__all__ = [
    "BounceChainConfig",
    "BounceEvent",
    "BounceChainResult",
    "run_bounce_chain",
    "chain_prediction",
]


@dataclass
class BounceChainConfig:
    """Inputs of the repeated-impact experiment (SI units)."""

    radius: float = 0.025
    density: float = 917.0
    stiffness: float = 5_000.0
    damping: float = 5.0
    travel: float = 0.05          # free distance between the two planes (m)
    impact_speed: float = 1.0     # initial speed towards the bottom plane (m/s)
    bounces: int = 6
    dt: float | None = None       # None -> IceDEM3D recommended bound
    dt_fraction: float = 0.5      # dt = dt_fraction * recommended_dt when dt is None
    max_steps: int = 5_000_000
    device: str = "cpu"

    def validate(self) -> None:
        for name in ("radius", "density", "stiffness", "travel", "impact_speed"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if not math.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError(f"damping must be finite and non-negative, got {self.damping}")
        if int(self.bounces) < 1:
            raise ValueError("bounces must be >= 1")
        if not 0.0 < float(self.dt_fraction) <= 1.0:
            raise ValueError("dt_fraction must lie in (0, 1]")


@dataclass
class BounceEvent:
    """One contact episode of the chain."""

    index: int
    plane: str                 # "bottom" | "top"
    t_start: float
    t_end: float
    impact_speed: float        # |approach speed| at first overlapped step
    rebound_speed: float       # separation speed at the first free step
    restitution: float | None  # rebound_speed / impact_speed (None if degenerate)
    reference: dict[str, Any] = field(default_factory=dict)
    max_overlap: float = 0.0
    steps: int = 0
    # Force window: the interval during which the plane actually exerts a
    # reaction.  It ends *before* the geometric overlap when the clipped
    # dashpot detaches, and it is the quantity the reference solution reports
    # as ``duration``.  Comparing geometric overlap time with it is wrong.
    force_duration: float = 0.0
    force_end_overlap: float = 0.0
    free_flight_to_next: float | None = None


@dataclass
class BounceChainResult:
    config: dict[str, Any]
    mass: float
    dt: float
    recommended_dt: float
    events: list[BounceEvent]
    checks: dict[str, Any]
    free_flight_velocity_leak: float   # max |dv_z| over force-free steps (m/s)
    total_time: float
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "mass": self.mass,
            "dt": self.dt,
            "recommended_dt": self.recommended_dt,
            "events": [asdict(e) for e in self.events],
            "checks": self.checks,
            "free_flight_velocity_leak": self.free_flight_velocity_leak,
            "total_time": self.total_time,
            "steps": self.steps,
        }


def chain_prediction(impact_speed: float, restitution: float, bounces: int) -> list[float]:
    """Ideal chain speeds ``e**n * v_0`` (n = 1 .. bounces)."""
    return [impact_speed * restitution ** (n + 1) for n in range(bounces)]


def _build_engine(cfg: BounceChainConfig) -> tuple[IceDEM3D, float, float]:
    """Engine with one free particle between two rigid planes and no bonds/gravity."""
    kwargs = dict(
        nx=2, ny=2, nz=2, radius=cfg.radius, density=cfg.density,
        contact_stiffness=cfg.stiffness, contact_damping=cfg.damping,
        platen_stiffness=cfg.stiffness, platen_damping=cfg.damping,
        drag=0.0, fix_x_edges=False, fix_bottom=False,
        bottom_platen_enabled=True, top_platen_enabled=True,
        tool_gap=100.0, tool_speed=1e-9,
        device=cfg.device,
        # The platen branch always uses platen_damping as an absolute dashpot;
        # "absolute" makes the pair branch use the same convention so the whole
        # engine has one documented contact law.
        contact_damping_model="absolute",
    )
    base = DEM3DConfig(**kwargs)
    kwargs["dt"] = cfg.dt if cfg.dt is not None else cfg.dt_fraction * base.recommended_dt
    sim = IceDEM3D(DEM3DConfig(**kwargs))
    sim.alive[:] = False              # contact-only: no bond can interfere
    sim.failure_energy_J[:] = 0.0

    r = float(cfg.radius)
    # Park every idle particle on the mid-plane, laterally far outside the
    # 2r contact range of the runner, so their only role is to keep the cell
    # list bounding box compact.
    z_mid = r
    sites = ((-0.20, -0.20), (0.20, -0.20), (-0.20, 0.20),
             (0.20, 0.20), (0.0, 0.20), (0.0, -0.20), (-0.20, 0.0))
    with torch.no_grad():
        for idx in range(1, sim.positions.shape[0]):
            dx, dy = sites[(idx - 1) % len(sites)]
            sim.positions[idx] = torch.tensor((dx, dy, z_mid), dtype=sim.dtype)
            sim.velocities[idx] = torch.zeros(3, dtype=sim.dtype)

    # Fix the plane positions explicitly: they must span exactly ``travel`` of
    # free flight, independent of where the idle particles were parked.
    bottom_z = 0.0
    top_z = bottom_z + 2.0 * r + float(cfg.travel)
    sim.bottom_platen_start_z = bottom_z
    sim.top_platen_start_z = top_z
    sim.positions[0] = torch.tensor((0.0, 0.0, bottom_z + r + 0.5 * float(cfg.travel)),
                                    dtype=sim.dtype)
    sim.velocities[0] = torch.tensor((0.0, 0.0, -float(cfg.impact_speed)), dtype=sim.dtype)
    return sim, bottom_z, top_z


def run_bounce_chain(cfg: BounceChainConfig | None = None) -> BounceChainResult:
    """Integrate the repeated-impact chain and reduce it to closure metrics."""
    cfg = cfg if cfg is not None else BounceChainConfig()
    cfg.validate()

    sim, bottom_z, top_z = _build_engine(cfg)
    r = float(cfg.radius)
    dt = float(sim.dt)
    mass = float(sim.config.mass)

    events: list[BounceEvent] = []
    steps = 0
    leak = 0.0                 # max |dv_z| over force-free steps
    force_leak_steps = 0       # steps with a reaction while outside the plane
    force_leak_worst = 0.0
    prev_vz = float(sim.velocities[0, 2])

    in_episode = False
    plane: str | None = None
    t_start = t_force_start = t_force_end = 0.0
    v_in = 0.0
    max_overlap = force_end_overlap = 0.0
    contact_steps = 0

    while steps < cfg.max_steps and len(events) < cfg.bounces:
        sim.step()
        steps += 1
        z = float(sim.positions[0, 2])
        vz = float(sim.velocities[0, 2])
        ov_bottom = max(0.0, bottom_z + r - z)
        ov_top = max(0.0, z + r - top_z)
        in_contact = ov_bottom > 0.0 or ov_top > 0.0
        reaction = max(abs(float(sim.last_bottom_platen_reaction_z)),
                       abs(float(sim.last_top_platen_reaction_z)))
        # Invariant: a plane may only act on a particle that intersects it.
        if reaction > 0.0 and not in_contact:
            force_leak_steps += 1
            force_leak_worst = max(force_leak_worst, reaction)

        if not in_episode:
            if in_contact:
                in_episode = True
                plane = "bottom" if ov_bottom >= ov_top else "top"
                t_start = sim.time - dt
                v_in = abs(prev_vz)
                max_overlap = max(ov_bottom, ov_top)
                contact_steps = 1
                t_force_start = sim.time - dt if reaction > 0.0 else 0.0
                t_force_end = sim.time
                force_end_overlap = max(ov_bottom, ov_top)
            else:
                # Force-free flight: semi-implicit Euler must not change v at all.
                leak = max(leak, abs(vz - prev_vz))
        else:
            max_overlap = max(max_overlap, ov_bottom, ov_top)
            contact_steps += 1
            if reaction > 0.0:
                if t_force_start == 0.0:
                    t_force_start = sim.time - dt
                t_force_end = sim.time
                force_end_overlap = max(ov_bottom, ov_top)
            if not in_contact:
                v_out = abs(vz)
                rest = v_out / v_in if v_in > 0.0 else None
                ref = (asdict(unilateral_contact_detail(cfg.damping, cfg.stiffness,
                                                        mass, v_in))
                       if rest is not None and v_in > 0.0 else {})
                event = BounceEvent(
                    index=len(events), plane=str(plane),
                    t_start=t_start, t_end=sim.time,
                    impact_speed=v_in, rebound_speed=v_out, restitution=rest,
                    reference=ref, max_overlap=max_overlap, steps=contact_steps,
                    force_duration=(max(0.0, t_force_end - t_force_start)
                                    if t_force_start > 0.0 else 0.0),
                    force_end_overlap=force_end_overlap,
                )
                if events:
                    events[-1].free_flight_to_next = t_start - events[-1].t_end
                events.append(event)
                in_episode = False
                t_force_start = t_force_end = 0.0
                force_end_overlap = 0.0
        prev_vz = vz

    checks = _reduce(cfg, mass, events)
    checks["separated_force_leak_steps"] = force_leak_steps
    checks["separated_force_leak_worst_N"] = force_leak_worst
    return BounceChainResult(
        config=asdict(cfg),
        mass=mass,
        dt=dt,
        recommended_dt=config_recommended_dt(cfg),
        events=events,
        checks=checks,
        free_flight_velocity_leak=leak,
        total_time=float(sim.time),
        steps=steps,
    )


def config_recommended_dt(cfg: BounceChainConfig) -> float:
    """Conservative step bound of the same engine configuration (s)."""
    return float(DEM3DConfig(
        nx=2, ny=2, nz=2, radius=cfg.radius, density=cfg.density,
        contact_stiffness=cfg.stiffness, contact_damping=cfg.damping,
        platen_stiffness=cfg.stiffness, platen_damping=cfg.damping,
        bottom_platen_enabled=True, top_platen_enabled=True,
    ).recommended_dt)


def _overlap_spread(closed: list[BounceEvent]) -> float | None:
    """Relative spread of the geometric overlap time (state-release check)."""
    vals = [e.t_end - e.t_start for e in closed]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    return (max(vals) - min(vals)) / mean if mean else None


def _reduce(cfg: BounceChainConfig, mass: float, events: list[BounceEvent]) -> dict[str, Any]:
    """Turn the event list into the closure metrics quoted in the docs."""
    closed = [e for e in events if e.restitution is not None and e.reference]
    e_open = closed[0].restitution if closed else None

    per_event_err = [
        abs(e.restitution - e.reference["restitution"]) / e.reference["restitution"]
        for e in closed
    ]
    duration_err = [
        abs(e.force_duration - e.reference["duration"]) / e.reference["duration"]
        for e in closed
    ]
    energy_err = []
    for e in closed:
        loss = 0.5 * mass * (e.impact_speed ** 2 - e.rebound_speed ** 2)
        predicted = 0.5 * mass * e.impact_speed ** 2 * (1.0 - e.reference["restitution"] ** 2)
        energy_err.append(abs(loss - predicted) / predicted if predicted > 0 else 0.0)

    chain_err = []
    if e_open is not None:
        predicted = chain_prediction(cfg.impact_speed, e_open, len(closed))
        chain_err = [
            abs(e.rebound_speed - v) / v for e, v in zip(closed, predicted)
        ]

    durations = [e.force_duration for e in closed]
    mean_dur = sum(durations) / len(durations) if durations else 0.0
    spread = (max(durations) - min(durations)) / mean_dur if mean_dur else 0.0

    def worst(values: list[float]) -> float | None:
        return max(values) if values else None

    return {
        "n_events": len(events),
        "n_closed_events": len(closed),
        "restitution_worst_rel_error": worst(per_event_err),
        "restitution_mean_rel_error": (sum(per_event_err) / len(per_event_err)
                                       if per_event_err else None),
        "chain_worst_rel_error": worst(chain_err),
        "force_duration_worst_rel_error": worst(duration_err),
        "force_duration_relative_spread": spread,
        "geometric_overlap_duration_spread": _overlap_spread(closed),
        "energy_loss_worst_rel_error": worst(energy_err),
        "restitution_first_event": e_open,
        "restitution_mean": (sum(e.restitution for e in closed) / len(closed)
                             if closed else None),
        "detached_all_events": all(e.reference.get("detached", False) for e in closed),
        # Negative control: the bilateral (textbook) damping-to-restitution map
        # is *not* the engine's clipped law.  Reporting its error alongside the
        # clipped-law error shows that the 3 % agreement above discriminates
        # between "the right law" and "any plausible law".
        "bilateral_restitution": classical_restitution(cfg.damping, cfg.stiffness, mass),
        "bilateral_worst_rel_error": (
            worst([abs(e.restitution - classical_restitution(cfg.damping, cfg.stiffness, mass))
                   / classical_restitution(cfg.damping, cfg.stiffness, mass)
                   for e in closed]) if closed else None),
    }