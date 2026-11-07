"""DEM ice-hull contact-force record used as the whipping excitation.

The generator is the repository's own 2-D bonded-lattice DEM
(:class:`tensordem.dem.IceDEM`) with a prescribed bow profile advancing into a
finite ice sheet.  Every lattice bond that fails produces a load drop, so the
hull reaction is a train of pulses -- exactly the *dynamic* ice load that a
pseudo-static design load cannot represent.

The lattice is a reduced prototype (tens of bonded disks, model scale), so the
absolute force level is not a full-scale prediction.  The record is therefore
used as the pulse *shape generator*: the pipeline rescales it linearly to a
documented full-scale bow load (:class:`~tensordem.whipping.pipeline.ShipCase`),
which leaves every ratio-based result (whipping share, damage share, pulse
statistics) independent of the scale factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DEMLoadConfig", "dem_contact_force"]


@dataclass(frozen=True)
class DEMLoadConfig:
    """Reduced indentation case (model scale, SI units)."""

    nx: int = 10
    ny: int = 4
    radius: float = 0.15
    thickness: float = 0.4
    density: float = 917.0
    bond_stiffness: float = 2.0e4
    contact_stiffness: float = 2.0e4
    contact_damping: float = 20.0
    breaking_strain: float = 0.012
    shear_breaking_strain: float = 0.02
    tool_radius: float = 0.03
    hull_speed: float = 1.5
    duration: float = 2.4
    start_x: float = -1.8
    sampling_rate_hz: float = 1000.0
    profile: tuple = ((-0.8, 1.0), (0.0, 0.45), (-0.8, -0.1))


def dem_contact_force(config: DEMLoadConfig = DEMLoadConfig(), *, seed: int = 0):
    """Integrate the DEM indentation case and return a 1 kHz ice-load record.

    Returns ``(time, force, info)`` where ``force`` is the magnitude of the
    hull reaction (N) on a uniform ``config.sampling_rate_hz`` grid and
    ``info`` carries the DEM diagnostics needed for the audit trail.
    """
    import torch

    from ..dem import DEMConfig, IceDEM

    cfg = DEMConfig(nx=config.nx, ny=config.ny, radius=config.radius,
                    thickness=config.thickness, density=config.density,
                    bond_stiffness=config.bond_stiffness,
                    contact_stiffness=config.contact_stiffness,
                    contact_damping=config.contact_damping,
                    drag=2.0, tool_radius=config.tool_radius,
                    tool_speed=config.hull_speed,
                    breaking_strain=config.breaking_strain,
                    shear_breaking_strain=config.shear_breaking_strain,
                    device="cpu")
    profile = torch.tensor(config.profile, dtype=torch.float64)
    sim = IceDEM(cfg, hull_profile=profile,
                 hull_start=(config.start_x, 0.0),
                 hull_velocity=(config.hull_speed, 0.0))
    n_steps = int(round(config.duration / sim.dt))
    raw = np.empty(n_steps)
    for k in range(n_steps):
        _, reaction = sim.forces()
        raw[k] = float(torch.linalg.vector_norm(reaction))
        sim.step()
    t_raw = np.arange(n_steps) * sim.dt
    fs = 1.0 / sim.dt
    dt = 1.0 / config.sampling_rate_hz
    grid = t_raw[0] + dt * np.arange(int(np.floor((t_raw[-1] - t_raw[0]) / dt)) + 1)
    force = np.interp(grid, t_raw, raw)
    info = {
        "dem_dt_s": sim.dt,
        "dem_native_sampling_hz": fs,
        "dem_steps": n_steps,
        "sampling_rate_hz": config.sampling_rate_hz,
        "duration_s": float(grid[-1]),
        "particle_count": int(len(sim.positions)),
        "broken_bonds": int(sim.broken_bonds),
        "peak_force_N": float(force.max()),
        "mean_force_N": float(force.mean()),
        "hull_speed_m_s": config.hull_speed,
        "profile": [list(p) for p in config.profile],
        "seed": seed,
    }
    return grid, force, info