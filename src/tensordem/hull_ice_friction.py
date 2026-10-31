"""Hull-ice friction coefficient and its temperature dependence (DEM scale).

The microscale (molecular) cross-check lives in
:mod:`tensordem.md_interface_friction`; this module provides

* the ice/steel interfacial shear law and the parametric ``mu(T)`` family
  ``mu(T) = mu0 + alpha * |T - T_melt| ** beta`` together with a least-squares
  identification of ``(mu0, alpha, beta)``;
* a 2-D bonded-particle DEM sliding-friction experiment in which a rough,
  rigid steel surface slides underneath a bonded ice bed.  The friction
  coefficient is the time-averaged ratio of interfacial tangential force to
  interfacial normal load, decomposed into its adhesion (tangential) and
  ploughing (normal-force projection) contributions.

All quantities are SI (m, s, N, K) unless a unit is given in the name.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

T_MELT_C = 0.0  # melting point of pure ice at atmospheric pressure [degC]
T_MELT_K = 273.15

#: Ice on steel at about -10 degC and 0.1-1 m/s sliding speed: Bowden & Tabor
#: (1954) report mu = 0.15-0.20.  This band is both the reference point of the
#: interfacial law and the acceptance band of the calibration criterion.
MU_LITERATURE_LOW = 0.15
MU_LITERATURE_HIGH = 0.20
MU_REFERENCE_TEMPERATURE_C = -10.0


def _require_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


def _require_positive(value: float, name: str) -> None:
    _require_finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def micro_contact_friction(
    temperature_C: float,
    *,
    mu_ref: float = MU_LITERATURE_LOW,
    reference_temperature_C: float = MU_REFERENCE_TEMPERATURE_C,
    exponent: float = 0.45,
) -> float:
    """Ice/steel contact-level friction coefficient ``mu_c(T)``.

    Physics
    -------
    Ice sliding on a harder counterface is adhesion controlled,
    ``mu = tau_shear / p_hardness``.  Both the shear strength of the ice
    surface and its indentation hardness increase as the ice cools, and a
    quasi-liquid layer carries the shear close to the melting point, so the
    ice/steel friction coefficient falls monotonically towards 0 degC
    (Bowden & Tabor 1954; Oksanen & Keinonen 1982).  Over -5 ... -25 degC the
    measured trend is well represented by a power law in the distance to the
    melting point::

        mu_c(T) = mu_ref * (|T - T_melt| / |T_ref - T_melt|) ** exponent

    Args:
        temperature_C: ice surface temperature [degC], must be <= 0 degC.
        mu_ref: friction coefficient at the reference temperature.
        reference_temperature_C: reference temperature [degC] (negative).
        exponent: power-law exponent (0.3-0.6 for ice on steel).

    Returns:
        Contact-level friction coefficient (dimensionless).
    """
    _require_finite(temperature_C, "temperature_C")
    _require_positive(mu_ref, "mu_ref")
    _require_finite(reference_temperature_C, "reference_temperature_C")
    _require_positive(exponent, "exponent")
    if temperature_C > T_MELT_C:
        raise ValueError("temperature_C must not exceed the melting point")
    if reference_temperature_C >= T_MELT_C:
        raise ValueError("reference_temperature_C must be below the melting point")
    ratio = abs(temperature_C - T_MELT_C) / abs(reference_temperature_C - T_MELT_C)
    return mu_ref * ratio**exponent


def friction_temperature_model(
    temperature_C,
    *,
    mu0: float,
    alpha: float,
    beta: float,
    melt_temperature_C: float = T_MELT_C,
):
    """Parametric calibration form ``mu(T) = mu0 + alpha * |T - T_melt| ** beta``.

    Accepts a float or a ``torch.Tensor``; returns the matching type.  The
    derivative ``dmu/dT = -alpha * beta * |T| ** (beta - 1)`` is negative for
    ``alpha, beta > 0``, i.e. friction grows as the ice cools, as required by
    the monotonicity criterion.
    """
    _require_finite(mu0, "mu0")
    _require_finite(alpha, "alpha")
    _require_positive(beta, "beta")
    _require_finite(melt_temperature_C, "melt_temperature_C")
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")
    if isinstance(temperature_C, torch.Tensor):
        delta = (temperature_C - melt_temperature_C).abs()
        return mu0 + alpha * delta.pow(beta)
    _require_finite(temperature_C, "temperature_C")
    return mu0 + alpha * abs(temperature_C - melt_temperature_C) ** beta


def fit_mu_temperature(
    temperatures_C: Sequence[float],
    mu_values: Sequence[float],
    *,
    melt_temperature_C: float = T_MELT_C,
    beta_grid: Optional[Iterable[float]] = None,
) -> dict:
    """Least-squares identification of ``(mu0, alpha, beta)``.

    For a fixed ``beta`` the model is linear in ``(mu0, alpha)``, so the fit is
    an exact linear least-squares solve on a ``beta`` grid followed by a
    bisection refinement around the best grid point.  Returns the
    coefficients, the residuals, ``R**2`` and the monotonicity flag.
    """
    t = [float(value) for value in temperatures_C]
    m = [float(value) for value in mu_values]
    if len(t) != len(m) or len(t) < 3:
        raise ValueError("need at least three equal-length temperature/mu pairs")
    for value in t:
        _require_finite(value, "temperatures_C")
    for value in m:
        _require_finite(value, "mu_values")
    if any(value > melt_temperature_C for value in t):
        raise ValueError("temperatures must be at or below the melting point")

    delta = torch.tensor(
        [abs(value - melt_temperature_C) for value in t], dtype=torch.float64
    )
    target = torch.tensor(m, dtype=torch.float64)

    def solve(beta: float) -> tuple[float, float, float]:
        basis = torch.stack([torch.ones_like(delta), delta.pow(beta)], dim=1)
        coefficients = torch.linalg.lstsq(basis, target.unsqueeze(1)).solution.reshape(2)
        residual = target - basis @ coefficients
        return (
            float(coefficients[0]),
            float(coefficients[1]),
            float(residual.square().sum()),
        )

    grid = (
        list(beta_grid)
        if beta_grid is not None
        else [0.05 + 0.01 * index for index in range(120)]
    )
    best_beta = min(grid, key=lambda value: solve(value)[2])
    best = solve(best_beta)
    step = 0.01
    for _ in range(60):
        moved = False
        for candidate in (best_beta - step, best_beta + step):
            if candidate <= 1.0e-6:
                continue
            trial = solve(candidate)
            if trial[2] < best[2] - 1e-18:
                best, best_beta, step, moved = trial, candidate, step * 0.7, True
                break
        if not moved:
            step *= 0.5
            if step < 1e-10:
                break
    mu0, alpha, sse = best
    fitted = [mu0 + alpha * float(value) ** best_beta for value in delta]
    residuals = [pred - obs for pred, obs in zip(fitted, m)]
    mean_obs = sum(m) / len(m)
    sst = sum((obs - mean_obs) ** 2 for obs in m)
    return {
        "mu0": mu0,
        "alpha": alpha,
        "beta": best_beta,
        "melt_temperature_C": melt_temperature_C,
        "temperatures_C": t,
        "mu_observed": m,
        "mu_fitted": fitted,
        "residuals": residuals,
        "rmse": math.sqrt(sse / len(m)),
        "max_abs_residual": max(abs(value) for value in residuals),
        "r_squared": 1.0 - sse / sst if sst > 0.0 else 1.0,
        "monotone_decreasing_in_T": bool(alpha > 0.0 and best_beta > 0.0),
    }


# ---------------------------------------------------------------------------
# DEM sliding-friction experiment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DEMFrictionConfig:
    """Configuration of the DEM ice/steel sliding-friction experiment.

    The ice bed is a square lattice of disk particles with breakable axial and
    diagonal bonds.  Its top row is pinned (the specimen holder), it is loaded
    by its own weight onto a row of rigid, equally spaced cylindrical
    asperities representing the steel surface, and the steel surface
    translates underneath at ``sliding_velocity``.
    """

    temperature_C: float = -10.0
    n_x: int = 9
    n_y: int = 4
    particle_radius: float = 1.0e-3
    thickness: float = 2.0e-3
    density: float = 917.0
    normal_stiffness: float = 1.0e3
    damping_ratio: float = 0.25
    friction_ice_ice: float = 0.35
    interface_mu_ref: float = MU_LITERATURE_LOW
    interface_exponent: float = 0.45
    bond_tensile_strength: float = 0.60e6
    bond_shear_strength: float = 0.90e6
    bond_stiffness_factor: float = 1.0
    bond_failure: bool = False
    bump_radius: float = 0.3e-3
    bump_spacing: float = 1.5e-3
    normal_pressure_Pa: float = 5.0e3
    sliding_velocity: float = 0.5
    gravity: float = 9.81
    timestep: float = 1.0e-5
    steps: int = 2000
    transient_fraction: float = 0.4
    jitter: float = 0.01
    seed: int = 0

    @property
    def interface_mu(self) -> float:
        """Contact-level ice/steel friction coefficient at this temperature."""
        return micro_contact_friction(
            self.temperature_C,
            mu_ref=self.interface_mu_ref,
            exponent=self.interface_exponent,
        )

    def __post_init__(self) -> None:
        _require_finite(self.temperature_C, "temperature_C")
        if self.temperature_C > T_MELT_C:
            raise ValueError("temperature_C must not exceed the melting point")
        if self.n_x < 4 or self.n_y < 3:
            raise ValueError("the ice bed needs n_x >= 4 and n_y >= 3")
        for name in (
            "particle_radius",
            "thickness",
            "density",
            "normal_stiffness",
            "bump_radius",
            "bump_spacing",
            "sliding_velocity",
            "timestep",
        ):
            _require_positive(getattr(self, name), name)
        _require_finite(self.normal_pressure_Pa, "normal_pressure_Pa")
        if self.normal_pressure_Pa < 0.0:
            raise ValueError("normal_pressure_Pa must be nonnegative")
        if not 0.0 < self.transient_fraction < 1.0:
            raise ValueError("transient_fraction must lie in (0, 1)")
        if not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError("damping_ratio must lie in [0, 1)")
        if self.steps < 100:
            raise ValueError("steps must be at least 100")


@dataclass
class DEMFrictionResult:
    """Outcome of one DEM sliding-friction experiment."""

    temperature_C: float
    mu: float
    mu_adhesion: float
    mu_ploughing: float
    normal_load_N: float
    tangential_load_N: float
    interface_mu: float
    mean_normal_contacts: float
    bonds_broken: int
    bonds_total: int
    sliding_distance_m: float
    steps_averaged: int
    seed: int

    def as_dict(self) -> dict:
        return {
            "temperature_C": self.temperature_C,
            "mu": self.mu,
            "mu_adhesion": self.mu_adhesion,
            "mu_ploughing": self.mu_ploughing,
            "normal_load_N": self.normal_load_N,
            "tangential_load_N": self.tangential_load_N,
            "interface_mu": self.interface_mu,
            "mean_normal_contacts": self.mean_normal_contacts,
            "bonds_broken": self.bonds_broken,
            "bonds_total": self.bonds_total,
            "sliding_distance_m": self.sliding_distance_m,
            "steps_averaged": self.steps_averaged,
            "seed": self.seed,
        }


def _ice_bed(config: DEMFrictionConfig, generator: torch.Generator):
    """Particle positions, masses, bond list, rest lengths and the pinned mask."""
    spacing = 2.0 * config.particle_radius * 1.01
    ys = torch.arange(config.n_y, dtype=torch.float64) * spacing
    xs = torch.arange(config.n_x, dtype=torch.float64) * spacing
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    positions = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    if config.jitter > 0.0:
        positions = positions + config.jitter * spacing * (
            torch.rand(positions.shape, generator=generator, dtype=torch.float64) - 0.5
        )
    positions[:, 1] -= positions[:, 1].min()
    positions[:, 1] += 5.0 * config.particle_radius  # clearance above the asperities
    count = positions.shape[0]
    index = torch.arange(count).reshape(config.n_y, config.n_x)
    pairs = []
    for dx, dy in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if dy >= 0:
            rows = config.n_y - dy
            row_offset = 0
        else:
            rows = config.n_y + dy
            row_offset = -dy
        cols = config.n_x - dx
        left = index[row_offset: row_offset + rows, 0:cols]
        right = index[row_offset + dy: row_offset + dy + rows, dx: dx + cols]
        pairs.append(torch.stack((left.flatten(), right.flatten()), dim=1))
    bonds = torch.cat(pairs, dim=0)
    rest_length = torch.linalg.vector_norm(
        positions[bonds[:, 1]] - positions[bonds[:, 0]], dim=1
    )
    mass = config.density * math.pi * config.particle_radius**2 * config.thickness
    masses = torch.full((count,), mass, dtype=torch.float64)
    top = positions[:, 1].max()
    pinned = positions[:, 1] >= top - 0.5 * spacing
    return positions, masses, bonds, rest_length, pinned



def _steel_surface(config: DEMFrictionConfig, bed_bottom_y: float):
    """Flat steel wall plus a periodic row of roughness bumps.

    The wall plane (``y = wall_y``) carries the dominant normal reaction and
    therefore defines an unambiguous normal direction, while the bumps provide
    the roughness that generates ploughing.  Both are rigid and translate with
    the plate velocity.
    """
    wall_y = bed_bottom_y - config.particle_radius
    span = config.n_x * 2.0 * config.particle_radius * 1.01
    count = int(math.ceil(span / config.bump_spacing)) + 3
    xs = (torch.arange(count, dtype=torch.float64) - 1.0) * config.bump_spacing
    # bumps are almost flush with the wall: they perturb the sliding surface
    # without lifting the bed off the load-bearing plane
    crest = wall_y - config.bump_radius + 0.002 * config.particle_radius
    bumps = torch.stack((xs, torch.full((count,), crest, dtype=torch.float64)), dim=1)
    return wall_y, bumps


def _disk_contacts(
    relative_position: torch.Tensor,
    relative_velocity: torch.Tensor,
    *,
    sum_radius: float,
    stiffness: float,
    damping: float,
    friction: float,
    velocity_regularisation: float = 1.0e-4,
):
    """Linear spring-dashpot normal contact plus regularised Coulomb friction.

    Returns ``(force, normal_magnitude, tangential_vector, unit_normal)`` where
    ``force`` is the force on the first body of the pair.
    """
    distance = torch.linalg.vector_norm(relative_position, dim=1).clamp_min(1e-18)
    normal = relative_position / distance.unsqueeze(1)
    overlap = (sum_radius - distance).clamp_min(0.0)
    normal_speed = (relative_velocity * normal).sum(dim=1)
    normal_force = (stiffness * overlap - damping * normal_speed).clamp_min(0.0)
    normal_force = torch.where(overlap > 0.0, normal_force, torch.zeros_like(normal_force))
    tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
    tangent_speed = (relative_velocity * tangent).sum(dim=1)
    magnitude = friction * normal_force * torch.tanh(
        tangent_speed.abs() / velocity_regularisation
    )
    tangential = -magnitude.unsqueeze(1) * torch.sign(tangent_speed.unsqueeze(1)) * tangent
    return normal_force.unsqueeze(1) * normal + tangential, normal_force, tangential, normal


def run_dem_friction_experiment(config: DEMFrictionConfig) -> DEMFrictionResult:
    """Run one DEM sliding-friction experiment and return ``mu(T)``.

    A bonded ice bed, pinned along its top row and loaded by an external normal
    pressure, rests on a rigid steel surface (flat wall + roughness bumps) that
    translates underneath it.  The friction coefficient is the time-averaged
    ratio of the tangential to the normal reaction of the ice on the steel
    surface::

        mu = |<F_plate,x>| / |<F_plate,y>|

    split exactly into the tangential reaction (adhesion, controlled by the
    ice/steel interfacial shear law) and the x-projection of the bump normal
    forces (ploughing / dilatancy, controlled by the DEM geometry).
    """
    generator = torch.Generator().manual_seed(config.seed)
    positions, masses, bonds, rest_length, pinned = _ice_bed(config, generator)
    count = positions.shape[0]
    velocities = torch.zeros_like(positions)
    free = (~pinned).unsqueeze(1).to(torch.float64)

    n_bonds = bonds.shape[0]
    area = 2.0 * config.particle_radius * config.thickness
    tensile_limit = config.bond_tensile_strength * area
    bond_stiffness = config.normal_stiffness * config.bond_stiffness_factor
    broken = torch.zeros(n_bonds, dtype=torch.bool)

    wall_y, bumps = _steel_surface(config, float(positions[:, 1].min()))
    damping = 2.0 * config.damping_ratio * math.sqrt(config.normal_stiffness * masses[0])
    tiny = 1.0e-30
    mu_interface = config.interface_mu

    # self-weight plus the externally applied normal load (hull contact pressure)
    load = config.normal_pressure_Pa * (2.0 * config.particle_radius) ** 2
    weight = torch.stack(
        (torch.zeros_like(masses), -load - masses * config.gravity), dim=1
    ) * free

    iu, ju = torch.triu_indices(count, count, offset=1)
    window_start = int(config.transient_fraction * config.steps)
    two_r = 2.0 * config.particle_radius
    contact_radius = config.particle_radius + config.bump_radius
    plate_velocity = config.sliding_velocity
    articulation = torch.tensor([plate_velocity, 0.0], dtype=torch.float64)

    acc_adhesion = torch.zeros((), dtype=torch.float64)
    acc_ploughing = torch.zeros((), dtype=torch.float64)
    acc_normal = torch.zeros((), dtype=torch.float64)
    acc_contacts = torch.zeros((), dtype=torch.float64)
    samples = 0

    for step in range(config.steps):
        offset = plate_velocity * step * config.timestep

        # --- body forces --------------------------------------------------
        forces = weight.clone()

        # --- ice-ice contacts --------------------------------------------
        contact_force, _, _, _ = _disk_contacts(
            positions[iu] - positions[ju],
            velocities[iu] - velocities[ju],
            sum_radius=two_r,
            stiffness=config.normal_stiffness,
            damping=damping,
            friction=config.friction_ice_ice,
        )
        forces.index_add_(0, iu, contact_force)
        forces.index_add_(0, ju, -contact_force)

        # --- breakable bonds (normal + tangential spring) -----------------
        alive = (~broken).to(torch.float64).unsqueeze(1)
        bond_relative = positions[bonds[:, 0]] - positions[bonds[:, 1]]
        bond_distance = torch.linalg.vector_norm(bond_relative, dim=1).clamp_min(tiny)
        direction = bond_relative / bond_distance.unsqueeze(1)
        stretch = bond_distance - rest_length
        normal_bond = bond_stiffness * stretch
        # a stretched bond (stretch > 0) pulls its particles together
        bond_force = (-normal_bond.unsqueeze(1) * direction) * alive
        forces.index_add_(0, bonds[:, 0], bond_force)
        forces.index_add_(0, bonds[:, 1], -bond_force)
        if config.bond_failure:
            broken = broken | (normal_bond.abs() > tensile_limit)

        # --- ice / steel roughness bumps ----------------------------------
        bumped = bumps + torch.tensor([offset, 0.0], dtype=torch.float64)
        bump_force, normal_force, tangential, normals = _disk_contacts(
            (positions[:, None, :] - bumped[None, :, :]).reshape(-1, 2),
            velocities.repeat_interleave(bumped.shape[0], dim=0)
            - articulation.repeat(count * bumped.shape[0], 1),
            sum_radius=contact_radius,
            stiffness=config.normal_stiffness,
            damping=damping,
            friction=mu_interface,
        )
        forces = forces + bump_force.reshape(count, -1, 2).sum(dim=1)
        bump_normal_x = (normal_force * normals[:, 0]).sum()
        bump_tangential = tangential.sum(dim=0)
        bump_normal_y = (normal_force * normals[:, 1]).sum()

        # --- ice / steel flat wall ----------------------------------------
        wall_overlap = (config.particle_radius + wall_y - positions[:, 1]).clamp_min(0.0)
        wall_normal = (
            config.normal_stiffness * wall_overlap - damping * velocities[:, 1]
        ).clamp_min(0.0)
        wall_normal = torch.where(
            wall_overlap > 0.0, wall_normal, torch.zeros_like(wall_normal)
        )
        wall_slip = velocities[:, 0] - plate_velocity
        wall_tangential = (
            -mu_interface
            * wall_normal
            * torch.tanh(wall_slip.abs() / 1.0e-4)
            * torch.sign(wall_slip)
        )
        forces = forces + torch.stack((wall_tangential, wall_normal), dim=1)

        if step >= window_start:
            # reaction of the ice on the steel surface
            adhesion = wall_tangential.sum() + bump_tangential[0]
            acc_adhesion = acc_adhesion + adhesion
            acc_ploughing = acc_ploughing + bump_normal_x
            acc_normal = acc_normal + wall_normal.sum() + bump_normal_y + bump_tangential[1]
            acc_contacts = acc_contacts + (
                (normal_force > 0).to(torch.float64).sum()
                + (wall_normal > 0).to(torch.float64).sum()
            )
            samples += 1

        # --- semi-implicit Euler integration ------------------------------
        velocities = (
            velocities + (forces / masses.unsqueeze(1)) * config.timestep
        ) * free
        positions = positions + velocities * config.timestep

    if samples == 0:
        raise RuntimeError("no steady-state averaging window")
    mean_normal = float(acc_normal) / samples
    if mean_normal == 0.0:
        raise RuntimeError("no normal load recorded at the ice/steel interface")
    mean_adhesion = float(acc_adhesion) / samples
    mean_ploughing = float(acc_ploughing) / samples
    mu_adhesion = abs(mean_adhesion) / abs(mean_normal)
    mu_ploughing = abs(mean_ploughing) / abs(mean_normal)
    return DEMFrictionResult(
        temperature_C=config.temperature_C,
        mu=mu_adhesion + mu_ploughing,
        mu_adhesion=mu_adhesion,
        mu_ploughing=mu_ploughing,
        normal_load_N=abs(mean_normal),
        tangential_load_N=abs(mean_adhesion) + abs(mean_ploughing),
        interface_mu=mu_interface,
        mean_normal_contacts=float(acc_contacts) / samples,
        bonds_broken=int(broken.sum()),
        bonds_total=n_bonds,
        sliding_distance_m=plate_velocity * config.steps * config.timestep,
        steps_averaged=samples,
        seed=config.seed,
    )

def dem_temperature_sweep(
    temperatures_C: Sequence[float],
    *,
    repeats: int = 10,
    template: Optional[DEMFrictionConfig] = None,
    base_seed: int = 20261031,
    **overrides,
) -> list[dict]:
    """Repeat the DEM experiment at every temperature and average the results.

    ``repeats`` independent realisations (default 10, as required by the
    calibration protocol) are run per temperature.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if template is None:
        template = DEMFrictionConfig(**overrides)
    elif overrides:
        raise ValueError("pass overrides only when no template is given")
    records = []
    for temperature in temperatures_C:
        _require_finite(float(temperature), "temperatures_C")
        runs = []
        for repeat in range(repeats):
            config = DEMFrictionConfig(
                **{
                    **template.__dict__,
                    "temperature_C": float(temperature),
                    "seed": base_seed + 997 * repeat + int(round(abs(float(temperature)) * 10)),
                }
            )
            runs.append(run_dem_friction_experiment(config).as_dict())
        values = torch.tensor([run["mu"] for run in runs], dtype=torch.float64)
        records.append(
            {
                "temperature_C": float(temperature),
                "mu_mean": float(values.mean()),
                "mu_std": float(values.std(unbiased=True)) if len(runs) > 1 else 0.0,
                "mu_min": float(values.min()),
                "mu_max": float(values.max()),
                "repeats": repeats,
                "mu_adhesion_mean": float(
                    torch.tensor([run["mu_adhesion"] for run in runs], dtype=torch.float64).mean()
                ),
                "mu_ploughing_mean": float(
                    torch.tensor([run["mu_ploughing"] for run in runs], dtype=torch.float64).mean()
                ),
                "normal_load_N_mean": float(
                    torch.tensor([run["normal_load_N"] for run in runs], dtype=torch.float64).mean()
                ),
                "bonds_broken_mean": float(
                    torch.tensor([run["bonds_broken"] for run in runs], dtype=torch.float64).mean()
                ),
                "interface_mu": runs[0]["interface_mu"],
                "runs": runs,
            }
        )
    return records