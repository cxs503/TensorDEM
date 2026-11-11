"""Discrete (DEM) directional uniaxial test on an anisotropic bonded specimen.

Why this module exists
----------------------
The continuum surface of :mod:`tensordem.ice_anisotropy.yield_surface` needs a
*discrete* counterpart for two jobs:

1. **Negative control on the engine, not on the algebra.**  With an isotropic
   bond-strength law the bonded specimen must give the same peak stress whether
   the column axis is parallel (0 deg) or perpendicular (90 deg) to the load.
   That is measured by actually stepping the DEM, so it tests the whole call
   path (per-bond threshold -> fracture commit -> platen reaction), not a
   formula.
2. **A measured statement about what this lattice can and cannot resolve.**
   The cubic bonded lattice has three orthogonal bond families, and the
   specimen is redundancy-dominated: weakening one family leaves the other two
   carrying the load.  Sweeping the micro-strength contrast therefore produces a
   *saturating* macro anisotropy, which is why the 45 deg criterion of this node
   is scored against the slip-system reference and not against this lattice.
   The measured saturation curve is reported as evidence for that choice.

Micro-strength law
------------------
Per-bond tensile breaking strain, orientation dependent about the column axis
``a``:

    eps_b(n) = eps_b0 * (1 + beta * (n . a)^2)

with ``n`` the bond normal (centre-to-centre direction, taken from the engine's
``initial_normals``).  ``beta = 0`` reproduces the homogeneous legacy specimen
bit-for-bit; ``beta < 0`` makes bonds parallel to the column axis weaker;
``beta > 0`` makes them stronger.  The law is written into the engine's
``bond_breaking_strain`` array, i.e. no engine file is modified and every
pre-existing trajectory is untouched.

The direction is rotated by rotating the *column axis* ``a`` relative to a fixed
loading axis (+z), so all three runs share one lattice, one seed and one
protocol -- only the fabric orientation changes.

Protocol
--------
``dem3d_triaxial.TriaxialRunner`` with ``protocol="ucs_opposed"`` (both platens
advance, no confinement), peak axial stress of the platen reaction reported as
the uniaxial strength.  The energy-ledger closure of the triaxial record is
carried through so a non-closed run cannot be quoted as a strength.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from ..dem3d_triaxial import TriaxialConfig, TriaxialRunner

__all__ = ["DirectionalTestConfig", "run_directional_test", "micro_contrast_sweep"]


@dataclass
class DirectionalTestConfig:
    """Specimen + protocol parameters for one directional uniaxial DEM test."""

    nx: int = 4
    ny: int = 4
    nz: int = 3
    radius: float = 0.025
    loading_speed: float = 0.05
    max_axial_steps: int = 1200
    dt_factor: float = 1.0
    seed: int = 0
    #: micro-strength contrast ``beta`` of ``eps_b(n) = eps_b0 (1 + beta (n.a)^2)``
    micro_contrast: float = 0.0
    #: load angle from the column axis, degrees
    theta_deg: float = 0.0
    #: angle of the column axis out of the x-z plane, degrees (fabric azimuth)
    azimuth_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.micro_contrast <= -1.0:
            raise ValueError("micro_contrast must be > -1 (strengths stay positive)")
        if not (0.0 <= self.theta_deg <= 180.0):
            raise ValueError("theta_deg must lie in [0, 180]")

    def column_axis(self) -> torch.Tensor:
        """Column axis for this run (unit, in the specimen frame)."""
        th = math.radians(self.theta_deg)
        az = math.radians(self.azimuth_deg)
        return torch.tensor([math.sin(th) * math.cos(az),
                             math.sin(th) * math.sin(az),
                             math.cos(th)], dtype=torch.float64)


def _triaxial_config(cfg: DirectionalTestConfig) -> TriaxialConfig:
    return TriaxialConfig(
        nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, radius=cfg.radius,
        protocol="ucs_opposed",
        loading_speed=cfg.loading_speed,
        max_axial_steps=cfg.max_axial_steps,
        dt_factor=cfg.dt_factor,
        seed=cfg.seed,
    )


def run_directional_test(cfg: DirectionalTestConfig | None = None, **overrides: Any) -> dict[str, Any]:
    """Run one directional uniaxial DEM test and return its record.

    Returns
    -------
    dict
        ``sigma_peak_Pa`` (peak axial stress), ``strain_at_peak``,
        ``broken_bonds``, ``n_steps``, the energy-ledger gate, the config and
        the micro-strength statistics actually applied to the bonds.
    """
    cfg = cfg or DirectionalTestConfig(**overrides)
    runner = TriaxialRunner(_triaxial_config(cfg))
    sim = runner.sim
    axis = cfg.column_axis().to(sim.initial_normals.dtype)
    cos_a = sim.initial_normals @ axis
    factors = 1.0 + cfg.micro_contrast * cos_a ** 2
    base = float(sim.config.breaking_strain)
    sim.bond_breaking_strain = (base * factors).to(sim.initial_normals.dtype)
    record = runner.run()
    eps = sim.bond_breaking_strain
    return {
        "theta_deg": cfg.theta_deg,
        "azimuth_deg": cfg.azimuth_deg,
        "micro_contrast": cfg.micro_contrast,
        "sigma_peak_Pa": float(record["sigma1_Pa"]),
        "strain_at_peak": float(record["strain_at_peak"]),
        "broken_bonds": int(record["final_broken_bonds"]),
        "n_steps": int(record["n_steps"]),
        "energy_ledger_pass": bool(record["energy_ledger"]["pass"]),
        "energy_ledger_residual": float(record["energy_ledger"]["closure_residual_relative"]),
        "bond_breaking_strain": {
            "median": float(eps.median()),
            "min": float(eps.min()),
            "max": float(eps.max()),
        },
        "config": {
            "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz, "radius": cfg.radius,
            "loading_speed": cfg.loading_speed, "max_axial_steps": cfg.max_axial_steps,
            "seed": cfg.seed,
        },
    }


def micro_contrast_sweep(contrasts: Sequence[float] = (-0.7, 0.0, 3.0), *,
                         angles: Sequence[float] = (0.0, 90.0),
                         base: DirectionalTestConfig | None = None) -> dict[str, Any]:
    """Peak strength of the 0 deg and 90 deg specimens for each micro contrast.

    Returns the per-run records plus the macro anisotropy
    ``sigma(0)/sigma(90)`` for every contrast -- the curve that documents how
    much macro anisotropy this redundancy-dominated lattice can produce.
    """
    base = base or DirectionalTestConfig()
    runs: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    for beta in contrasts:
        by_angle: dict[float, float] = {}
        for th in angles:
            rec = run_directional_test(DirectionalTestConfig(
                nx=base.nx, ny=base.ny, nz=base.nz, radius=base.radius,
                loading_speed=base.loading_speed, max_axial_steps=base.max_axial_steps,
                dt_factor=base.dt_factor, seed=base.seed,
                micro_contrast=float(beta), theta_deg=float(th)))
            runs.append(rec)
            by_angle[float(th)] = rec["sigma_peak_Pa"]
        if 0.0 in by_angle and 90.0 in by_angle:
            ratio = by_angle[0.0] / max(by_angle[90.0], 1e-30)
        else:
            ratio = float("nan")
        table.append({
            "micro_contrast": float(beta),
            "sigma_0deg_Pa": by_angle.get(0.0),
            "sigma_90deg_Pa": by_angle.get(90.0),
            "macro_anisotropy": float(ratio),
            "all_ledgers_pass": all(r["energy_ledger_pass"] for r in runs),
        })
    return {"runs": runs, "table": table}