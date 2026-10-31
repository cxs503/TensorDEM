"""Reduced molecular-dynamics (MD) ice/iron interface shear experiment.

Purpose: an independent, molecular-scale estimate of the hull-ice friction
coefficient used to cross-check the DEM calibration in
:mod:`tensordem.hull_ice_friction`.

Model
-----
* **Substrate:** three frozen layers of an fcc(111)-like triangular iron
  lattice.  Iron atoms are uncharged in this reduced model, so the adhesion is
  dispersive (Lennard-Jones); the substrate atoms are held fixed and their
  forces are recorded, which makes the traction at the interface directly
  measurable.
* **Ice film:** two layers of a honeycomb (bipartite) lattice of single-site
  water pseudo-molecules carrying alternating partial charges.  The Coulomb
  term stands in for the directional hydrogen-bond network of the reduced
  model; the charge magnitude is scaled so that the surrogate melting point
  falls close to that of ice (see ``melting_scan``).
* **Drive:** the top film layer is pulled along x by a harmonic spring whose
  anchor moves at constant velocity; a Langevin thermostat holds the film
  temperature.

The friction coefficient is the time-averaged ratio of the tangential to the
normal force transmitted to the frozen substrate (``mu_MD``), i.e. the same
definition as in the DEM experiment but resolved at the molecular scale.

All internal computations are in SI (m, s, N, K); the configuration interface
uses Angstrom and kJ/mol for readability where stated in the field names.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .hull_ice_friction import T_MELT_C, T_MELT_K, _require_finite, _require_positive

BOLTZMANN = 1.380649e-23  # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C
COULOMB_CONSTANT = 8.9875517923e9  # N m^2 / C^2
AMU = 1.66053906660e-27  # kg
AVOGADRO = 6.02214076e23
KJ_PER_MOL_IN_J = 1000.0 / AVOGADRO


@dataclass(frozen=True)
class MDFrictionConfig:
    """Configuration of the reduced MD ice/iron shear experiment."""

    temperature_C: float = -10.0
    width: int = 5
    height: int = 4
    ice_layers: int = 2
    iron_layers: int = 3
    # Angstrom / kJ per mol descriptions of the potentials
    ice_bond_A: float = 3.50
    ice_layer_spacing_A: float = 2.05
    film_offset_A: float = 3.30
    lj_sigma_water_A: float = 3.00
    lj_epsilon_water_kJ: float = 0.30
    lj_sigma_iron_A: float = 2.60
    lj_epsilon_iron_water_kJ: float = 0.20
    iron_lattice_A: float = 2.60
    iron_layer_spacing_A: float = 2.05
    partial_charge_e: float = 0.09
    relative_permittivity: float = 3.4
    cutoff_A: float = 12.0
    drive_velocity_m_s: float = 0.5
    drive_stiffness_N_m: float = 5.0
    normal_load_nN: float = 60.0
    thermostat_time_ps: float = 0.3
    timestep_fs: float = 1.0
    steps: int = 2000
    transient_fraction: float = 0.5
    seed: int = 0
    #: Melting temperature [K] of the *reduced* water model.  When positive, the
    #: thermostat temperature is obtained by corresponding states,
    #: ``T_sim = T_melt_surrogate * T_physical / T_melt_ice``, so that the
    #: physical reduced temperature ``T_phys/T_melt`` is reproduced even though
    #: the surrogate potential has its own melting point.
    surrogate_melting_K: float = 0.0

    def __post_init__(self) -> None:
        _require_finite(self.temperature_C, "temperature_C")
        if self.temperature_C > T_MELT_C:
            raise ValueError("temperature_C must not exceed the melting point")
        for name in (
            "ice_bond_A",
            "ice_layer_spacing_A",
            "lj_sigma_water_A",
            "lj_epsilon_water_kJ",
            "lj_sigma_iron_A",
            "lj_epsilon_iron_water_kJ",
            "iron_lattice_A",
            "iron_layer_spacing_A",
            "relative_permittivity",
            "cutoff_A",
            "drive_stiffness_N_m",
            "normal_load_nN",
            "thermostat_time_ps",
            "timestep_fs",
        ):
            _require_positive(getattr(self, name), name)
        if self.width < 2 or self.height < 2:
            raise ValueError("the interface patch must be at least 2x2 unit cells")
        if self.steps < 500:
            raise ValueError("steps must be at least 500")
        if not 0.0 < self.transient_fraction < 1.0:
            raise ValueError("transient_fraction must lie in (0, 1)")

    @property
    def temperature_K(self) -> float:
        return T_MELT_K + self.temperature_C

    @property
    def thermostat_temperature_K(self) -> float:
        """Temperature imposed on the thermostat (corresponding states)."""
        if self.surrogate_melting_K > 0.0:
            return self.surrogate_melting_K * self.temperature_K / T_MELT_K
        return self.temperature_K


@dataclass
class MDFrictionResult:
    temperature_C: float
    temperature_K: float
    mu: float
    normal_force_nN: float
    friction_force_nN: float
    kinetic_temperature_K: float
    molecules: int
    substrate_atoms: int
    steps_averaged: int
    seed: int

    def as_dict(self) -> dict:
        return {
            "temperature_C": self.temperature_C,
            "temperature_K": self.temperature_K,
            "mu": self.mu,
            "normal_force_nN": self.normal_force_nN,
            "friction_force_nN": self.friction_force_nN,
            "kinetic_temperature_K": self.kinetic_temperature_K,
            "molecules": self.molecules,
            "substrate_atoms": self.substrate_atoms,
            "steps_averaged": self.steps_averaged,
            "seed": self.seed,
        }


def _honeycomb_film(config: MDFrictionConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-sublattice honeycomb water film, returned as (positions, charges)."""
    bond = config.ice_bond_A * 1.0e-10
    a1 = torch.tensor([1.5 * bond, 0.5 * math.sqrt(3.0) * bond], dtype=torch.float64)
    a2 = torch.tensor([1.5 * bond, -0.5 * math.sqrt(3.0) * bond], dtype=torch.float64)
    basis = torch.tensor([[0.0, 0.0], [bond, 0.0]], dtype=torch.float64)
    points = []
    for layer in range(config.ice_layers):
        z = (config.film_offset_A + config.ice_layer_spacing_A * layer) * 1.0e-10
        shift = (0.75 * bond, 0.433 * bond) if layer % 2 else (0.0, 0.0)
        for i in range(config.width):
            for j in range(config.height):
                origin = i * a1 + j * a2
                for site in range(2):
                    xy = origin + basis[site] + torch.tensor(shift, dtype=torch.float64)
                    points.append([float(xy[0]), float(xy[1]), z])
    positions = torch.tensor(points, dtype=torch.float64)
    # centre the patch in x/y so that the slab and film overlap
    positions[:, 0] -= positions[:, 0].mean()
    positions[:, 1] -= positions[:, 1].mean()
    n_cells = config.width * config.height * config.ice_layers
    charge = torch.empty(2 * n_cells, dtype=torch.float64)
    charge[0::2] = config.partial_charge_e * ELEMENTARY_CHARGE
    charge[1::2] = -config.partial_charge_e * ELEMENTARY_CHARGE
    return positions, charge


def _iron_slab(config: MDFrictionConfig) -> torch.Tensor:
    """Frozen fcc(111)-like triangular iron slab, z <= 0."""
    spacing = config.iron_lattice_A * 1.0e-10
    dy = 0.5 * math.sqrt(3.0) * spacing
    points = []
    for layer in range(config.iron_layers):
        z = -config.iron_layer_spacing_A * 1.0e-10 * layer
        shift = 0.5 * spacing if layer % 2 else 0.0
        for i in range(config.width + 1):
            for j in range(config.height + 1):
                points.append([i * spacing + shift, j * dy, z])
    positions = torch.tensor(points, dtype=torch.float64)
    positions[:, 0] -= positions[:, 0].mean()
    positions[:, 1] -= positions[:, 1].mean()
    return positions


def _pairs(n: int, m: int, same_group: bool):
    """Index pairs (i, j) for a group of n and m particles."""
    if same_group:
        iu, ju = torch.triu_indices(n, n, offset=1)
        return iu, ju
    iu = torch.arange(n).repeat_interleave(m)
    ju = torch.arange(m).repeat(n)
    return iu, ju


def run_md_interface_shear(config: MDFrictionConfig) -> MDFrictionResult:
    """Shear the reduced ice film over the frozen iron slab at fixed T.

    Spring-driven shear cell: the top film layer is pulled along x by a
    harmonic spring whose anchor advances at ``drive_velocity_m_s`` and is held
    in z by a second spring; a Langevin thermostat keeps the film at the target
    temperature.  The friction coefficient is the time-averaged ratio of the
    tangential to the normal force transmitted to the frozen substrate,

        ``mu_MD = |<F_x>| / |<F_z>|``,

    i.e. the same operational definition as the DEM experiment.
    """
    generator = torch.Generator().manual_seed(config.seed)
    ice, charge = _honeycomb_film(config)
    iron = _iron_slab(config)
    n_ice = ice.shape[0]
    n_iron = iron.shape[0]
    molecules = torch.full((n_ice,), 18.015 * AMU, dtype=torch.float64)
    kt = BOLTZMANN * config.thermostat_temperature_K

    sigma_ww = config.lj_sigma_water_A * 1.0e-10
    epsilon_ww = config.lj_epsilon_water_kJ * KJ_PER_MOL_IN_J
    sigma_iw = config.lj_sigma_iron_A * 1.0e-10
    epsilon_iw = config.lj_epsilon_iron_water_kJ * KJ_PER_MOL_IN_J
    cutoff = config.cutoff_A * 1.0e-10
    k_coulomb = COULOMB_CONSTANT / config.relative_permittivity
    dt = config.timestep_fs * 1.0e-15
    gamma = molecules / (config.thermostat_time_ps * 1.0e-12)

    positions = ice.clone()
    velocities = torch.sqrt(kt / molecules).unsqueeze(1) * torch.randn(
        positions.shape, generator=generator, dtype=torch.float64
    )
    ii, jj = _pairs(n_ice, n_ice, same_group=True)
    qq = charge[ii] * charge[jj]

    top = positions[:, 2] > positions[:, 2].max() - 0.5 * config.ice_layer_spacing_A * 1e-10
    anchor_x = positions[top, 0].clone()
    anchor_z = positions[top, 2].clone()

    window_start = int(config.transient_fraction * config.steps)
    sum_fx = torch.zeros((), dtype=torch.float64)
    sum_fz = torch.zeros((), dtype=torch.float64)
    sum_kinetic = torch.zeros((), dtype=torch.float64)
    samples = 0

    for step in range(config.steps):
        # --- ice-ice: Lennard-Jones + Coulomb -----------------------------
        r_vec = positions[ii] - positions[jj]
        distance = torch.linalg.vector_norm(r_vec, dim=1).clamp_min(0.6 * sigma_ww)
        inv6 = (sigma_ww / distance).pow(6)
        magnitude = 24.0 * epsilon_ww * (2.0 * inv6.square() - inv6) / distance
        pair_force = (magnitude - k_coulomb * qq / distance.pow(2)).unsqueeze(1) * (
            r_vec / distance.unsqueeze(1)
        )
        pair_force = pair_force * (distance < cutoff).unsqueeze(1).to(torch.float64)
        forces = torch.zeros_like(positions)
        forces.index_add_(0, ii, pair_force)
        forces.index_add_(0, jj, -pair_force)

        # --- ice-iron: dispersive adhesion only ---------------------------
        r_vec = positions[:, None, :] - iron[None, :, :]
        distance = torch.linalg.vector_norm(r_vec, dim=2).clamp_min(0.6 * sigma_iw)
        inv6 = (sigma_iw / distance).pow(6)
        magnitude = 24.0 * epsilon_iw * (2.0 * inv6.square() - inv6) / distance
        magnitude = magnitude * (distance < cutoff).to(torch.float64)
        ice_iron = (magnitude.unsqueeze(2) * (r_vec / distance.unsqueeze(2))).sum(dim=1)
        forces = forces + ice_iron
        substrate_force = -ice_iron.sum(dim=0)

        # --- spring drive on the top layer --------------------------------
        target_x = anchor_x + config.drive_velocity_m_s * step * dt
        forces[top, 0] += -config.drive_stiffness_N_m * (positions[top, 0] - target_x)
        forces[top, 2] += -config.drive_stiffness_N_m * (positions[top, 2] - anchor_z)

        # --- Langevin thermostat (Ornstein-Uhlenbeck) ---------------------
        c1 = torch.exp(-gamma * dt / molecules)
        c2 = torch.sqrt(kt * (1.0 - c1 * c1)) / molecules.sqrt()
        noise = torch.randn(positions.shape, generator=generator, dtype=torch.float64)
        velocities = c1.unsqueeze(1) * velocities + c2.unsqueeze(1) * noise

        velocities = velocities + (forces / molecules.unsqueeze(1)) * dt
        positions = positions + velocities * dt

        if step >= window_start:
            sum_fx = sum_fx + substrate_force[0]
            sum_fz = sum_fz + substrate_force[2]
            kinetic = 0.5 * (molecules.unsqueeze(1) * velocities.square()).sum()
            sum_kinetic = sum_kinetic + kinetic / (1.5 * n_ice * BOLTZMANN)
            samples += 1

    if samples == 0:
        raise RuntimeError("no averaging window")
    mean_fx = float(sum_fx) / samples
    mean_fz = float(sum_fz) / samples
    if abs(mean_fz) < 1.0e-14:
        raise RuntimeError("no normal force recorded on the substrate")
    return MDFrictionResult(
        temperature_C=config.temperature_C,
        temperature_K=config.temperature_K,
        mu=abs(mean_fx) / abs(mean_fz),
        normal_force_nN=mean_fz * 1.0e9,
        friction_force_nN=mean_fx * 1.0e9,
        kinetic_temperature_K=float(sum_kinetic) / samples,
        molecules=n_ice,
        substrate_atoms=n_iron,
        steps_averaged=samples,
        seed=config.seed,
    )


def md_temperature_sweep(
    temperatures_C: Sequence[float],
    *,
    repeats: int = 2,
    template: Optional[MDFrictionConfig] = None,
    base_seed: int = 7717,
    **overrides,
) -> list[dict]:
    """Run the MD shear experiment at every temperature and average."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if template is None:
        template = MDFrictionConfig(**overrides)
    elif overrides:
        raise ValueError("pass overrides only when no template is given")
    records = []
    for temperature in temperatures_C:
        _require_finite(float(temperature), "temperatures_C")
        runs = [
            run_md_interface_shear(
                MDFrictionConfig(
                    **{
                        **template.__dict__,
                        "temperature_C": float(temperature),
                        "seed": base_seed + 431 * repeat,
                    }
                )
            ).as_dict()
            for repeat in range(repeats)
        ]
        values = torch.tensor([run["mu"] for run in runs], dtype=torch.float64)
        records.append(
            {
                "temperature_C": float(temperature),
                "mu_mean": float(values.mean()),
                "mu_std": float(values.std(unbiased=True)) if len(runs) > 1 else 0.0,
                "repeats": repeats,
                "normal_force_nN_mean": sum(run["normal_force_nN"] for run in runs) / len(runs),
                "kinetic_temperature_K_mean": sum(
                    run["kinetic_temperature_K"] for run in runs
                )
                / len(runs),
                "runs": runs,
            }
        )
    return records


def calibrate_epsilon_iron_water(
    target_mu: float,
    temperature_C: float,
    *,
    grid: Sequence[float] = (0.03, 0.08, 0.13, 0.18, 0.23, 0.29, 0.35, 0.45, 0.55),
    template: Optional[MDFrictionConfig] = None,
    repeats: int = 1,
    **overrides,
) -> dict:
    """Identify the single free MD parameter (Fe/water well depth) by grid search.

    The Fe/water dispersive well depth is the only parameter of the reduced
    model that is not taken directly from the literature, so it is identified
    at the reference temperature only and then frozen for the temperature
    sweep; the other temperatures are predictions.  A direct grid search is
    used instead of bisection because the response is non-monotone: at
    commensurate adhesion strengths the film locks into the substrate lattice
    (structural lubricity), which produces local minima of the friction
    coefficient.
    """
    _require_positive(target_mu, "target_mu")
    _require_finite(temperature_C, "temperature_C")
    base = template or MDFrictionConfig(**overrides)
    history = []
    for epsilon in grid:
        config = MDFrictionConfig(
            **{
                **base.__dict__,
                "temperature_C": temperature_C,
                "lj_epsilon_iron_water_kJ": float(epsilon),
            }
        )
        # average over the same number of realisations as the sweep, so that the
        # identified value and the validation values share one estimator
        if repeats > 1:
            record = md_temperature_sweep(
                [temperature_C], repeats=repeats, template=config
            )[0]
            mu = record["mu_mean"]
            normal_force = record["normal_force_nN_mean"]
        else:
            result = run_md_interface_shear(config)
            mu = result.mu
            normal_force = result.normal_force_nN
        history.append(
            {
                "lj_epsilon_iron_water_kJ": float(epsilon),
                "mu": mu,
                "normal_force_nN": normal_force,
            }
        )
    best = min(history, key=lambda record: abs(record["mu"] - target_mu))
    return {
        "lj_epsilon_iron_water_kJ": best["lj_epsilon_iron_water_kJ"],
        "mu": best["mu"],
        "target_mu": target_mu,
        "relative_error": abs(best["mu"] - target_mu) / target_mu,
        "temperature_C": temperature_C,
        "history": history,
        "converged": abs(best["mu"] - target_mu) / target_mu < 0.30,
        "protocol": "grid search at the reference temperature; value frozen for the sweep",
    }


def film_mean_square_displacement(
    thermostat_temperature_K: float,
    *,
    steps: int = 1500,
    seed: int = 11,
    config: Optional[MDFrictionConfig] = None,
) -> float:
    """Centre-of-mass corrected mean-square displacement of a free film [A^2].

    Used as the order parameter of the reduced water model: a jump with
    temperature marks its melting transition.
    """
    base = config or MDFrictionConfig()
    generator = torch.Generator().manual_seed(seed)
    ice, charge = _honeycomb_film(base)
    n = ice.shape[0]
    molecules = torch.full((n,), 18.015 * AMU, dtype=torch.float64)
    kt = BOLTZMANN * thermostat_temperature_K
    sigma = base.lj_sigma_water_A * 1.0e-10
    epsilon = base.lj_epsilon_water_kJ * KJ_PER_MOL_IN_J
    cutoff = base.cutoff_A * 1.0e-10
    k_coulomb = COULOMB_CONSTANT / base.relative_permittivity
    gamma = molecules / (base.thermostat_time_ps * 1.0e-12)
    dt = base.timestep_fs * 1.0e-15
    ii, jj = _pairs(n, n, same_group=True)
    qq = charge[ii] * charge[jj]
    positions = ice.clone()
    velocities = torch.sqrt(kt / molecules).unsqueeze(1) * torch.randn(
        positions.shape, generator=generator, dtype=torch.float64
    )
    reference = positions.clone()
    for _ in range(steps):
        r_vec = positions[ii] - positions[jj]
        distance = torch.linalg.vector_norm(r_vec, dim=1).clamp_min(0.6 * sigma)
        inv6 = (sigma / distance).pow(6)
        magnitude = 24.0 * epsilon * (2.0 * inv6.square() - inv6) / distance
        pair_force = (magnitude - k_coulomb * qq / distance.pow(2)).unsqueeze(1) * (
            r_vec / distance.unsqueeze(1)
        )
        pair_force = pair_force * (distance < cutoff).unsqueeze(1).to(torch.float64)
        forces = torch.zeros_like(positions)
        forces.index_add_(0, ii, pair_force)
        forces.index_add_(0, jj, -pair_force)
        c1 = torch.exp(-gamma * dt / molecules)
        c2 = torch.sqrt(kt * (1.0 - c1 * c1)) / molecules.sqrt()
        noise = torch.randn(positions.shape, generator=generator, dtype=torch.float64)
        velocities = c1.unsqueeze(1) * velocities + c2.unsqueeze(1) * noise
        velocities = velocities + (forces / molecules.unsqueeze(1)) * dt
        positions = positions + velocities * dt
    delta = (positions - reference) - (positions - reference).mean(dim=0)
    return float(delta.square().sum(dim=1).mean()) * 1.0e20


def melting_scan(
    temperatures_K: Sequence[float],
    *,
    steps: int = 1500,
    seed: int = 11,
    config: Optional[MDFrictionConfig] = None,
) -> dict:
    """Locate the melting point of the reduced water model by diffusion onset.

    Returns the measured mean-square displacement per temperature and the
    estimated melting temperature (the midpoint of the largest jump).
    """
    records = [
        {
            "thermostat_temperature_K": float(value),
            "msd_A2": film_mean_square_displacement(float(value), steps=steps, seed=seed, config=config),
        }
        for value in temperatures_K
    ]
    ordered = sorted(records, key=lambda record: record["thermostat_temperature_K"])
    best_jump = 0.0
    melting = float("nan")
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        jump = upper["msd_A2"] - lower["msd_A2"]
        if jump > best_jump:
            best_jump = jump
            melting = 0.5 * (
                lower["thermostat_temperature_K"] + upper["thermostat_temperature_K"]
            )
    return {
        "scan_points": records,
        "largest_msd_jump_A2": best_jump,
        "surrogate_melting_K": melting,
        "protocol": "diffusion onset of the free reduced water film (centre-of-mass corrected MSD)",
    }
