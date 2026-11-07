"""Three-dimensional bonded-sphere DEM engine for ice fracture.

All quantities use SI units. This first 3-D solver is deliberately independent of
the established 2-D implementation so existing cases and output formats remain
unchanged. The lattice is a simple-cubic network with face/body diagonals inside
the initial cutoff; calibrate its directional stiffness before engineering use.

Contact constitutive model (L1_foundation — SIR-PC2|F05):
- Normal: linear spring-dashpot with restitution-coefficient parameterization.
  ``DEM3DConfig(contact_model="hertz", young_modulus=..., poisson_ratio=...)``
  opts into the classical Hertz law ``F = (4/3) E* sqrt(R*) delta^(3/2)``, with
  the same absolute damping convention. The default stays ``"linear"`` so every
  existing case is bit-for-bit unchanged.
- Tangential: Cundall-Strack incremental Coulomb-friction spring (|F_t| <= mu * |F_n|).
- Rolling resistance: torque T_r = mu_r * R_eff * |F_n| opposing relative rotation.

Hertz helpers (public, used by calibration and verification code):
:func:`hertz_reduced_modulus`, :func:`hertz_effective_radius`,
:func:`hertz_normal_force`, :func:`hertz_contact_potential`,
:func:`hertz_secant_stiffness`, :func:`hertz_equivalent_linear_stiffness`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch

from .buoyancy_wet3d import submerged_fraction_sphere_tensor
from . import hertz_mindlin


# --------------------------------------------------------------------------
# Hertz contact law primitives
# --------------------------------------------------------------------------
# Classical two-body Hertz relations, coded here so that calibration and
# verification code share one source of truth:
#
#   1/E* = sum_i (1 - nu_i^2)/E_i,   1/R* = sum_i 1/R_i,
#   F    = (4/3) E* sqrt(R*) delta^(3/2),
#   U    = (8/15) E* sqrt(R*) delta^(5/2)   (= integral of F).
#
# For two equal spheres of the same material E* = E/(2(1-nu^2)) and R* = R/2;
# against a rigid counter-surface (indenter, platen) E* = E/(1-nu^2), R* = R.

def _hertz_argument(value: float | torch.Tensor) -> torch.Tensor:
    """Promote a scalar overlap to a float64 tensor without changing tensor input."""
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise TypeError("overlap must be a floating-point tensor")
        return value.to(torch.float64) if value.dtype != torch.float64 else value
    return torch.tensor(float(value), dtype=torch.float64)


def hertz_reduced_modulus(young_modulus: float, poisson_ratio: float, *,
                          rigid_counterpart: bool = False) -> float:
    """Effective modulus ``E*`` for a sphere-sphere or sphere-rigid contact."""
    if not math.isfinite(young_modulus) or young_modulus <= 0.0:
        raise ValueError("young_modulus must be finite and positive")
    if not (math.isfinite(poisson_ratio) and 0.0 <= poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be in [0, 0.5)")
    compliance = 1.0 - poisson_ratio ** 2
    return float(young_modulus / (compliance if rigid_counterpart else 2.0 * compliance))


def hertz_effective_radius(radius: float, *, rigid_counterpart: bool = False) -> float:
    """Effective radius ``R*``: ``R/2`` for equal spheres, ``R`` against a plane."""
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    return float(radius) if rigid_counterpart else 0.5 * float(radius)


def hertz_normal_force(overlap: float | torch.Tensor, e_star: float,
                       r_eff: float) -> torch.Tensor:
    """Hertz normal force ``F = (4/3) E* sqrt(R*) delta^(3/2)``, zero for ``delta <= 0``."""
    delta = _hertz_argument(overlap).clamp_min(0.0)
    return (4.0 / 3.0) * float(e_star) * math.sqrt(float(r_eff)) * delta.pow(1.5)


def hertz_contact_potential(overlap: float | torch.Tensor, e_star: float,
                            r_eff: float) -> torch.Tensor:
    """Recoverable contact energy ``U = (8/15) E* sqrt(R*) delta^(5/2)``."""
    delta = _hertz_argument(overlap).clamp_min(0.0)
    return (8.0 / 15.0) * float(e_star) * math.sqrt(float(r_eff)) * delta.pow(2.5)


def hertz_secant_stiffness(overlap: float | torch.Tensor, e_star: float,
                           r_eff: float) -> torch.Tensor:
    """Secant stiffness ``k_s = F/delta = (4/3) E* sqrt(R*) sqrt(delta)``."""
    delta = _hertz_argument(overlap).clamp_min(0.0)
    return (4.0 / 3.0) * float(e_star) * math.sqrt(float(r_eff)) * delta.sqrt()


def hertz_equivalent_linear_stiffness(young_modulus: float, poisson_ratio: float,
                                      radius: float, overlap: float, *,
                                      rigid_counterpart: bool = False) -> float:
    """Secant stiffness at ``overlap``: seeds a linear-penalty run from Hertz data."""
    e_star = hertz_reduced_modulus(young_modulus, poisson_ratio,
                                   rigid_counterpart=rigid_counterpart)
    r_eff = hertz_effective_radius(radius, rigid_counterpart=rigid_counterpart)
    return float(hertz_secant_stiffness(overlap, e_star, r_eff))


@dataclass(frozen=True)
class DEM3DConfig:
    """Numerical and physical parameters for the 3-D bonded-sphere model."""
    nx: int = 9
    ny: int = 5
    nz: int = 3
    radius: float = 0.025
    density: float = 917.0
    bond_stiffness: float = 2_000.0
    contact_stiffness: float = 5_000.0
    breaking_strain: float = 0.015
    shear_breaking_strain: float = 0.03
    contact_damping: float = 5.0
    drag: float = 2.0
    tool_radius: float = 0.1
    tool_speed: float = 0.5
    tool_gap: float = 0.01
    dt: float | None = None
    device: str = "cpu"
    geometry: str = "box"
    fix_x_edges: bool = True
    fix_bottom: bool = False
    top_platen_enabled: bool = False
    bottom_platen_enabled: bool = False
    platen_stiffness: float = 5_000.0
    platen_damping: float = 0.0
    # --- Platen (plane boundary) Coulomb friction ---
    # Default 0.0 keeps both platens frictionless (legacy behaviour, bit-for-bit).
    # Positive: Cundall-Strack tangential spring at each plane contact, saturated
    # at mu_p * N, so a specimen pushed sideways on a loaded platen sticks until
    # the tangential load reaches mu_p * N and then slides at (F - mu_p N)/m.
    platen_friction_coefficient: float = 0.0   # mu_p, shared by both platens
    platen_friction_stiffness: float | None = None  # k_t; None -> platen_stiffness
    top_platen_gap: float = 0.0
    bottom_platen_gap: float = 0.0
    top_platen_velocity: float = 0.0
    bottom_platen_velocity: float = 0.0
    # --- SIR-PC2|F05: Coulomb friction + rolling resistance ---
    friction_coefficient: float = 0.0        # mu: Coulomb friction coefficient
    rolling_resistance_coefficient: float = 0.0  # mu_r: rolling resistance coefficient
    restitution_coefficient: float = 1.0     # e: coefficient of restitution (0,1]
    # --- Contact law selection: legacy linear penalty (default) or Hertz ---
    contact_model: str = "linear"            # "linear" | "hertz"
    young_modulus: float = 1.0e9             # Pa, used only by contact_model="hertz"
    poisson_ratio: float = 0.3               # used only by contact_model="hertz"
    hertz_stability_overlap_fraction: float = 0.1  # delta_ref / radius for the dt bound
    # --- Normal dashpot convention for particle-particle contacts ---
    # "restitution" (default, bit-for-bit legacy): the pair dashpot is derived
    #   from restitution_coefficient and contact_damping does not enter the pair
    #   law (it still sets the tool branch and the recommended_dt bound).
    # "absolute": gamma_n = contact_damping, the same convention the tool and
    #   platen branches already use, so the pair law is the clipped
    #   F = max(k*delta - c*delta_dot, 0) whose reference solution is
    #   tensordem.restitution.unilateral_restitution(c, k, m_eff).
    contact_damping_model: str = "restitution"  # "restitution" | "absolute"
    # --- Opt-in body forces: gravity + Archimedes buoyancy ---
    # Both are OFF by default (gravity = 0, water_level = None), so every
    # pre-existing DEM3D trajectory is bit-for-bit unchanged.  Vertical axis is
    # +z; see docs/dem3d_buoyancy_wiring.md for the analytic anchors they are
    # scored against.
    gravity: float = 0.0                 # m/s^2, body force along -z
    water_level: float | None = None      # free-surface elevation z (m); None = dry
    water_density: float = 1025.0         # kg/m^3
    wet_drag_coefficient: float = 0.0     # N s/m, applied to the submerged fraction only

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        for name in ("radius", "density", "bond_stiffness", "contact_stiffness",
                     "breaking_strain", "shear_breaking_strain", "tool_radius", "tool_speed"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("contact_damping", "drag", "tool_gap", "platen_damping",
                     "top_platen_gap", "bottom_platen_gap"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.platen_stiffness) or self.platen_stiffness <= 0:
            raise ValueError("platen_stiffness must be finite and positive")
        if (not math.isfinite(self.platen_friction_coefficient)
                or self.platen_friction_coefficient < 0):
            raise ValueError("platen_friction_coefficient must be finite and nonnegative")
        if self.platen_friction_stiffness is not None and (
                not math.isfinite(self.platen_friction_stiffness)
                or self.platen_friction_stiffness <= 0):
            raise ValueError("platen_friction_stiffness must be finite and positive when given")
        for name in ("top_platen_velocity", "bottom_platen_velocity"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("top_platen_enabled", "bottom_platen_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0):
            raise ValueError("dt must be finite and positive")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.geometry not in ("box", "cylinder_xz"):
            raise ValueError("geometry must be 'box' or 'cylinder_xz'")
        if self.geometry == "cylinder_xz" and (self.nx < 3 or self.nz < 3):
            raise ValueError("cylinder_xz geometry requires nx and nz >= 3")
        # Validate friction / rolling / restitution parameters
        if not math.isfinite(self.friction_coefficient) or self.friction_coefficient < 0:
            raise ValueError("friction_coefficient must be finite and nonnegative")
        if not math.isfinite(self.rolling_resistance_coefficient) or self.rolling_resistance_coefficient < 0:
            raise ValueError("rolling_resistance_coefficient must be finite and nonnegative")
        if not (0.0 < self.restitution_coefficient <= 1.0):
            raise ValueError("restitution_coefficient must be in (0, 1]")
        if self.contact_damping_model not in ("restitution", "absolute"):
            raise ValueError(
                "contact_damping_model must be 'restitution' or 'absolute'")
        if (self.contact_damping_model == "absolute"
                and self.restitution_coefficient < 1.0):
            raise ValueError(
                "contact_damping_model='absolute' and restitution_coefficient < 1 "
                "specify the same dashpot twice; pick one convention")
        if self.contact_model not in ("linear", "hertz"):
            raise ValueError("contact_model must be 'linear' or 'hertz'")
        if not math.isfinite(self.young_modulus) or self.young_modulus <= 0.0:
            raise ValueError("young_modulus must be finite and positive")
        if not (math.isfinite(self.poisson_ratio) and 0.0 <= self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must be in [0, 0.5)")
        if not (math.isfinite(self.hertz_stability_overlap_fraction)
                and 0.0 < self.hertz_stability_overlap_fraction < 1.0):
            raise ValueError("hertz_stability_overlap_fraction must be in (0, 1)")
        # Opt-in body forces
        if not math.isfinite(self.gravity) or self.gravity < 0.0:
            raise ValueError("gravity must be finite and nonnegative")
        if self.water_level is not None and not math.isfinite(self.water_level):
            raise ValueError("water_level must be finite or None")
        if not math.isfinite(self.water_density) or self.water_density <= 0.0:
            raise ValueError("water_density must be finite and positive")
        if (not math.isfinite(self.wet_drag_coefficient)
                or self.wet_drag_coefficient < 0.0):
            raise ValueError("wet_drag_coefficient must be finite and nonnegative")

    @property
    def mass(self) -> float:
        return self.density * (4.0 / 3.0) * math.pi * self.radius**3

    def _bound_stiffness(self, linear_stiffness: float, *,
                         rigid_counterpart: bool) -> float:
        """Stiffness used by the explicit stability bound for one contact branch.

        The linear law returns ``linear_stiffness`` unchanged (bit-for-bit legacy
        behaviour). The Hertz law is non-linear, so the bound uses the secant
        stiffness at the reference overlap
        ``delta_ref = hertz_stability_overlap_fraction * radius``; overlaps larger
        than ``delta_ref`` need a correspondingly smaller step, which the
        ``dt <= recommended_dt`` guard cannot see in advance.
        """
        if self.contact_model == "linear":
            return float(linear_stiffness)
        delta_ref = self.hertz_stability_overlap_fraction * self.radius
        return float(hertz_secant_stiffness(
            delta_ref,
            hertz_reduced_modulus(self.young_modulus, self.poisson_ratio,
                                  rigid_counterpart=rigid_counterpart),
            hertz_effective_radius(self.radius, rigid_counterpart=rigid_counterpart)))

    @property
    def recommended_dt(self) -> float:
        # Conservative explicit stability/travel bound; validate by convergence.
        n = self.nx * self.ny * self.nz
        degree_bound = min(n - 1, 26)
        # The tangential platen spring is part of the stiff set even when the
        # friction is switched off, so a later mu_p switch cannot outrun the bound.
        platen_k = (self.platen_stiffness if self.platen_friction_stiffness is None
                    else max(self.platen_stiffness, self.platen_friction_stiffness))
        contact_stiffness = max(
            self.bond_stiffness,
            self._bound_stiffness(self.contact_stiffness, rigid_counterpart=False),
            self._bound_stiffness(platen_k, rigid_counterpart=True)
            if (self.top_platen_enabled or self.bottom_platen_enabled) else 0.0,
        )
        k = (2 * degree_bound + 1) * contact_stiffness
        c = ((2 * degree_bound + 1) * self.contact_damping + self.drag
             + (2 if (self.top_platen_enabled or self.bottom_platen_enabled) else 0)
             * self.platen_damping)
        elastic = 0.15 * math.sqrt(self.mass / k)
        damping = 0.15 * self.mass / c if c > 0 else math.inf
        travel = 0.1 * self.radius / self.tool_speed
        return min(elastic, damping, travel)


class IceDEM3D:
    """3-D bonded-sphere DEM with spherical prescribed indenter.

    State arrays have shape (N, 3); external forces are instantaneous nodal
    loads in newtons. The indenter translates downward at constant speed.
    Bond failure is irreversible. Permanent local bonds are stored in O(N);
    contacts use a spatial cell list rebuilt each force evaluation. The current
    CPU-side cell hashing is a prototype and can synchronize CUDA execution.

    Contact model (SIR-PC2|F05):
    - Normal: linear spring-dashpot, damping derived from restitution_coefficient.
    - Tangential: Cundall-Strack incremental Coulomb-friction spring.
    - Rolling: resistance torque T_r = mu_r * R_eff * |F_n|.
    """

    def __init__(self, config: DEM3DConfig = DEM3DConfig()) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.dtype = torch.float64
        self.dt = config.dt if config.dt is not None else config.recommended_dt
        if self.dt > config.recommended_dt:
            raise ValueError(f"dt exceeds conservative limit {config.recommended_dt:.6g} s")

        axes = [torch.arange(v, dtype=self.dtype, device=self.device) for v in
                (config.nx, config.ny, config.nz)]
        zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        full_grid = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
        # Geometry is defined on the original structured lattice, then compacted
        # so every state array and every bond uses contiguous particle indices.
        if config.geometry == "cylinder_xz":
            xc, zc = (config.nx - 1) / 2.0, (config.nz - 1) / 2.0
            disk_radius = min(config.nx - 1, config.nz - 1) / 2.0
            keep = ((xx - xc).square() + (zz - zc).square()) <= disk_radius**2 + 1e-12
        else:
            keep = torch.ones_like(xx, dtype=torch.bool)
        compact_to_full = torch.nonzero(keep.reshape(-1), as_tuple=False).flatten()
        full_to_compact = torch.full((len(full_grid),), -1, dtype=torch.long, device=self.device)
        full_to_compact[compact_to_full] = torch.arange(len(compact_to_full), device=self.device)
        grid = full_grid[compact_to_full]
        z_origin = (config.nz - 1) / 2.0 if config.geometry == "cylinder_xz" else 0.0
        self.positions = (grid - torch.tensor(
            [(config.nx - 1) / 2, (config.ny - 1) / 2, z_origin],
            dtype=self.dtype, device=self.device
        )) * (2 * config.radius)
        self.initial_positions = self.positions.clone()
        self.velocities = torch.zeros_like(self.positions)
        self.fixed = torch.zeros(len(grid), dtype=torch.bool, device=self.device)
        gx, gy, gz = grid.unbind(dim=1)
        if config.fix_x_edges and config.geometry == "box":
            self.fixed |= (gx == 0) | (gx == config.nx - 1)
        if config.fix_bottom:
            self.fixed |= gz == 0

        # Build only valid neighbour bonds and remap full-grid IDs to compact IDs.
        n = len(grid)
        bond_pairs: list[tuple[int, int]] = []
        for dz in range(0, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == dy == dz == 0:
                        continue
                    # Keep one half-space of offsets to avoid duplicate bonds.
                    if not (dz > 0 or (dz == 0 and dy > 0) or
                            (dz == 0 and dy == 0 and dx > 0)):
                        continue
                    for z in range(config.nz):
                        zz2 = z + dz
                        if not 0 <= zz2 < config.nz:
                            continue
                        for y in range(config.ny):
                            yy2 = y + dy
                            if not 0 <= yy2 < config.ny:
                                continue
                            for x in range(config.nx):
                                xx2 = x + dx
                                if not 0 <= xx2 < config.nx:
                                    continue
                                full_a = (z * config.ny + y) * config.nx + x
                                full_b = (zz2 * config.ny + yy2) * config.nx + xx2
                                a = int(full_to_compact[full_a].item())
                                b = int(full_to_compact[full_b].item())
                                if a >= 0 and b >= 0:
                                    bond_pairs.append((min(a, b), max(a, b)))
        self.pairs = torch.tensor(bond_pairs, dtype=torch.long, device=self.device)
        order = torch.argsort(self.pairs[:, 0] * n + self.pairs[:, 1])
        self.pairs = self.pairs[order]
        i, j = self.pairs.T
        ref = self.initial_positions[j] - self.initial_positions[i]
        self.rest_lengths = torch.linalg.vector_norm(ref, dim=1)
        self.bonded = torch.ones(len(self.pairs), dtype=torch.bool, device=self.device)
        self.alive = self.bonded.clone()
        self.bond_indices = self.pairs
        self.reference_bond_vectors = ref.clone()
        self.initial_normals = ref / self.rest_lengths.clamp_min(torch.finfo(self.dtype).eps)[:, None]
        self.failure_mode = torch.zeros(len(self.pairs), dtype=torch.uint8, device=self.device)
        self.failure_time_s = torch.full((len(self.pairs),), -1.0, dtype=self.dtype, device=self.device)
        self.failure_extension_m = torch.zeros(len(self.pairs), dtype=self.dtype, device=self.device)
        self.failure_energy_J = torch.zeros(len(self.pairs), dtype=self.dtype, device=self.device)
        self._bond_keys = self.pairs[:, 0] * len(grid) + self.pairs[:, 1]

        self.time = 0.0
        self.step_count = 0
        self.tool_start = torch.tensor(
            [0.0, 0.0, (config.nz - 1) * 2 * config.radius + config.radius
             + config.tool_radius + config.tool_gap],
            dtype=self.dtype, device=self.device)
        self.tool_velocity = torch.tensor([0.0, 0.0, -config.tool_speed],
                                          dtype=self.dtype, device=self.device)
        self.last_reaction = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.top_platen_start_z = (
            float(self.initial_positions[:, 2].max()) + config.radius + config.top_platen_gap
        )
        self.bottom_platen_start_z = (
            float(self.initial_positions[:, 2].min()) - config.radius - config.bottom_platen_gap
        )
        self.last_top_platen_reaction_z = torch.zeros((), dtype=self.dtype, device=self.device)
        self.last_bottom_platen_reaction_z = torch.zeros((), dtype=self.dtype, device=self.device)

        # --- SIR-PC2|F05: tangential friction & rolling resistance state ---
        # Tangential displacement accumulator per contact pair (keyed by pair hash).
        # Stored as dict: pair_key -> (tangential_disp_3d, last_normal_3d)
        self._tangential_state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Rolling resistance torque accumulator per particle (N, 3).
        self._rolling_torque = torch.zeros((n, 3), dtype=self.dtype, device=self.device)
        # Power delivered to the particles by the tangential friction force and
        # by the rolling-resistance couple, published by the step-owner force call
        # (update_history=True) for the energy-audit ledger.  Probes leave them
        # untouched so monitoring never clobbers a running accumulation.
        self._last_friction_power = 0.0
        self._last_rolling_power = 0.0
        # Flag indicating rolling resistance torque is physically modelled.
        self.rolling_resistance_torque_physical = config.rolling_resistance_coefficient > 0.0
        # --- Platen Coulomb friction: one tangential spring per particle and plane ---
        # A plane contact is unique per particle, so the history is a plain (n, 3)
        # tensor instead of the pair-keyed dict used for particle contacts.
        self._top_platen_tang_disp = torch.zeros((n, 3), dtype=self.dtype, device=self.device)
        self._bottom_platen_tang_disp = torch.zeros((n, 3), dtype=self.dtype, device=self.device)
        # Last evaluated tangential force (N) at each plane, for diagnostics only.
        self.last_top_platen_friction = torch.zeros((n, 3), dtype=self.dtype, device=self.device)
        self.last_bottom_platen_friction = torch.zeros((n, 3), dtype=self.dtype, device=self.device)

    @property
    def platen_friction_k(self) -> float:
        """Tangential stiffness used by the platen friction spring (N/m).

        Falls back to ``platen_stiffness``, the same ``k_t = k_n`` convention the
        particle-particle Cundall-Strack branch uses, and it is also the value
        the ``recommended_dt`` bound sees even when friction is switched off.
        """
        k_t = self.config.platen_friction_stiffness
        return float(self.config.platen_stiffness if k_t is None else k_t)

    @property
    def _hertz_pair_params(self) -> tuple[float, float]:
        """``(E*, R*)`` for particle-particle contact (unused by the linear model)."""
        return (hertz_reduced_modulus(self.config.young_modulus, self.config.poisson_ratio),
                hertz_effective_radius(self.config.radius))

    @property
    def _hertz_rigid_params(self) -> tuple[float, float]:
        """``(E*, R*)`` for contact against a rigid surface (indenter, platens)."""
        return (hertz_reduced_modulus(self.config.young_modulus, self.config.poisson_ratio,
                                      rigid_counterpart=True),
                hertz_effective_radius(self.config.radius, rigid_counterpart=True))

    @property
    def _hertz_g_star(self) -> float:
        """Reduced shear modulus G* for Hertz--Mindlin tangential stiffness."""
        return hertz_mindlin.hertz_reduced_shear_modulus(
            self.config.young_modulus, self.config.poisson_ratio)

    @property
    def tool_position(self) -> torch.Tensor:
        return self.tool_start + self.time * self.tool_velocity

    @property
    def broken_bonds(self) -> int:
        return int((self.bonded & ~self.alive).sum().item())

    @torch.no_grad()
    def _contact_candidate_pairs(self) -> torch.Tensor:
        """Build current overlap candidates with a uniform spatial cell list.

        Cell width is the contact diameter, so overlapping spheres can only be
        in the same or one of the 26 adjacent cells. Bond topology is stored
        separately and is never discarded when particles move or bonds break.
        """
        n = len(self.positions)
        if n < 2:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        width = 2.0 * self.config.radius
        coords = torch.floor(self.positions.detach().cpu() / width).to(torch.int64).tolist()
        cells: dict[tuple[int, int, int], list[int]] = {}
        for idx, xyz in enumerate(coords):
            key = (xyz[0], xyz[1], xyz[2])
            cells.setdefault(key, []).append(idx)
        pairs: list[tuple[int, int]] = []
        for key, members in cells.items():
            x, y, z = key
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        other = cells.get((x + dx, y + dy, z + dz))
                        if other is None:
                            continue
                        for i in members:
                            for j in other:
                                if j > i:
                                    pairs.append((i, j))
        if not pairs:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
        return torch.tensor(pairs, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _alive_bond_for_pairs(self, pairs: torch.Tensor) -> torch.Tensor:
        """Return an alive-bond mask aligned with arbitrary sorted node pairs."""
        if pairs.numel() == 0 or len(self._bond_keys) == 0:
            # No bond at all (a purely frictional / unbonded assembly): every
            # pair is a contact, so nothing can be masked out.  Without this
            # guard the searchsorted/clamp below indexes an empty key array.
            return torch.zeros((len(pairs),), dtype=torch.bool, device=self.device)
        keys = pairs[:, 0] * len(self.positions) + pairs[:, 1]
        where = torch.searchsorted(self._bond_keys, keys)
        safe = where.clamp(max=max(len(self._bond_keys) - 1, 0))
        matched = (where < len(self._bond_keys)) & (self._bond_keys[safe] == keys)
        result = torch.zeros_like(matched)
        result[matched] = self.alive[safe[matched]]
        return result

    @torch.no_grad()
    def _objective_shear_strain(self) -> torch.Tensor:
        """Least-squares local F and Green-Lagrange strain invariant in 3-D."""
        n = len(self.positions)
        i, j = self.bond_indices.T
        r = self.reference_bond_vectors
        q = self.positions[j] - self.positions[i]
        rr = r[:, :, None] * r[:, None, :]
        qr = q[:, :, None] * r[:, None, :]
        cov = torch.zeros((n, 3, 3), dtype=self.dtype, device=self.device)
        cross = torch.zeros_like(cov)
        cov.index_add_(0, i, rr); cov.index_add_(0, j, rr)
        cross.index_add_(0, i, qr); cross.index_add_(0, j, qr)
        # Rank-deficient local reference (fewer than 3 non-coplanar bonds) makes
        # the least-squares deformation gradient ill-posed: the pseudo-inverse
        # degenerates to a projection and reports a spurious 1/3 shear invariant
        # even for a rigid, undeformed bond. Skip the shear test on such
        # degenerate neighbourhoods (report 0) so thin/linear specimens and
        # single-bond unit setups do not spuriously fracture. Full-rank lattices
        # (every particle with >= 3 non-coplanar reference bonds) are unchanged.
        rank = torch.linalg.matrix_rank(cov)
        f = cross @ torch.linalg.pinv(cov)
        eye = torch.eye(3, dtype=self.dtype, device=self.device)
        e = 0.5 * (f.transpose(1, 2) @ f - eye)
        dev = e - torch.diag_embed(e.diagonal(dim1=1, dim2=2).mean(dim=1).unsqueeze(1).expand(-1, 3))
        # sqrt(2/3 devE:devE), invariant under rigid rotation.
        shear = torch.sqrt((2.0 / 3.0) * dev.square().sum(dim=(1, 2)).clamp_min(0.0))
        return torch.where(rank < 3, torch.zeros_like(shear), shear)

    def _loads(self, external_forces: torch.Tensor | None) -> torch.Tensor | None:
        if external_forces is None:
            return None
        if not isinstance(external_forces, torch.Tensor) or tuple(external_forces.shape) != tuple(self.positions.shape):
            raise ValueError(f"external_forces must be a tensor of shape {tuple(self.positions.shape)}")
        loads = external_forces.to(device=self.device, dtype=self.dtype)
        if not bool(torch.isfinite(loads).all()):
            raise ValueError("external_forces must contain only finite values")
        return loads

    # ------------------------------------------------------------------
    # Opt-in body forces: gravity + Archimedes buoyancy
    # ------------------------------------------------------------------
    @property
    def buoyancy_active(self) -> bool:
        """True when the Archimedes term is part of the force balance."""
        return self.config.water_level is not None

    @torch.no_grad()
    def submerged_fraction(self) -> torch.Tensor:
        """(N,) submerged volume fraction of every sphere, in [0, 1].

        Uses the verified spherical-cap kernel of :mod:`tensordem.buoyancy_wet3d`
        against the free surface ``z = water_level`` (the solver's vertical axis is
        ``+z``; that module names its vertical axis ``y``).  Returns exactly zero
        for a dry configuration, so the term is provably inert when disabled.
        """
        cfg = self.config
        if cfg.water_level is None:
            return torch.zeros(len(self.positions), dtype=self.dtype, device=self.device)
        return submerged_fraction_sphere_tensor(self.positions[:, 2], cfg.radius,
                                                cfg.water_level)

    @torch.no_grad()
    def body_forces(self) -> torch.Tensor:
        """(N, 3) gravity + buoyancy + submerged wet drag, zero when disabled.

        * gravity  ``-m g e_z`` per particle,
        * buoyancy ``+rho_w g V f_sub e_z`` with ``V = 4/3 pi R^3`` and ``f_sub``
          the spherical-cap submerged fraction (Archimedes: the force equals the
          weight of the displaced water, independent of how deep the cap sits),
        * wet drag ``-C_wet f_sub v`` (linear placeholder, damped only while
          submerged), only when ``wet_drag_coefficient > 0``.

        No history or fracture state is touched, so probing it is side-effect free.
        """
        cfg = self.config
        body = torch.zeros_like(self.positions)
        if cfg.water_level is None and cfg.gravity == 0.0:
            return body
        if cfg.gravity > 0.0:
            body[:, 2] -= cfg.mass * cfg.gravity
        if cfg.water_level is None:
            return body
        f_sub = self.submerged_fraction()
        volume = (4.0 / 3.0) * math.pi * cfg.radius ** 3
        body[:, 2] += cfg.water_density * cfg.gravity * volume * f_sub
        if cfg.wet_drag_coefficient > 0.0:
            body -= cfg.wet_drag_coefficient * f_sub[:, None] * self.velocities
        return body

    @torch.no_grad()
    def body_force_ledger(self) -> dict[str, float]:
        """Vertically summed gravity / buoyancy / wet-drag terms (N) for auditing."""
        cfg = self.config
        n = len(self.positions)
        f_sub = self.submerged_fraction()
        volume = (4.0 / 3.0) * math.pi * cfg.radius ** 3
        gravity = -cfg.mass * cfg.gravity * n
        buoyancy = float((cfg.water_density * cfg.gravity * volume * f_sub).sum())
        wet_drag = float(-cfg.wet_drag_coefficient
                         * (f_sub[:, None] * self.velocities)[:, 2].sum())
        return {
            "gravity_force_z_N": float(gravity),
            "buoyancy_force_z_N": float(buoyancy),
            "wet_drag_force_z_N": float(wet_drag),
            "net_body_force_z_N": float(gravity + buoyancy + wet_drag),
            "submerged_volume_m3": float((volume * f_sub).sum()),
        }

    @torch.no_grad()
    def _commit_fracture(self, extension: torch.Tensor) -> None:
        """Commit irreversible bond events only during an explicit force update."""
        cfg = self.config
        strain = extension / self.rest_lengths
        bi, bj = self.bond_indices.T
        shear = self._objective_shear_strain()
        shear_bond = 0.5 * (shear[bi] + shear[bj])
        tensile_fail = strain > cfg.breaking_strain
        shear_fail = shear_bond > cfg.shear_breaking_strain
        old_alive = self.alive.clone()
        self.alive &= ~(tensile_fail | shear_fail)
        newly = old_alive & ~self.alive
        self.failure_mode[newly] = tensile_fail[newly].to(torch.uint8) + 2 * shear_fail[newly].to(torch.uint8)
        self.failure_time_s[newly] = self.time
        self.failure_extension_m[newly] = extension[newly]
        self.failure_energy_J[newly] = 0.5 * cfg.bond_stiffness * extension[newly].square()

    # ------------------------------------------------------------------
    # SIR-PC2|F05: Coulomb friction + rolling resistance helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pair_key(i: int, j: int) -> int:
        """Deterministic key for a sorted particle pair."""
        return min(i, j) * 1_000_003 + max(i, j)

    def _get_tangential_state(self, ci: torch.Tensor, cj: torch.Tensor,
                              cnormal: torch.Tensor) -> torch.Tensor:
        """Retrieve or initialise tangential displacement for each contact pair.

        Returns (N, 3) tangential displacement vectors.
        """
        n_contacts = ci.numel()
        tang_disp = torch.zeros((n_contacts, 3), dtype=self.dtype, device=self.device)
        for idx in range(n_contacts):
            i = int(ci[idx].item())
            j = int(cj[idx].item())
            key = self._pair_key(i, j)
            if key in self._tangential_state:
                stored_disp, stored_normal = self._tangential_state[key]
                # Rotate stored tangential displacement to current normal frame
                # (simple approach: keep as-is in global frame; projection handles it)
                tang_disp[idx] = stored_disp
            else:
                tang_disp[idx] = torch.zeros(3, dtype=self.dtype, device=self.device)
        return tang_disp

    def _update_tangential_state(self, ci: torch.Tensor, cj: torch.Tensor,
                                 cnormal: torch.Tensor,
                                 tang_disp: torch.Tensor) -> None:
        """Persist tangential displacement for the next step."""
        n_contacts = ci.numel()
        for idx in range(n_contacts):
            i = int(ci[idx].item())
            j = int(cj[idx].item())
            key = self._pair_key(i, j)
            self._tangential_state[key] = (
                tang_disp[idx].clone(),
                cnormal[idx].clone(),
            )

    def _plane_friction_force(
        self, mask: torch.Tensor, normal_mag: torch.Tensor,
        tang_disp: torch.Tensor, *, update_history: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cundall-Strack tangential force at a horizontal plane boundary.

        The planes translate only along ``z`` (normal), so the tangential relative
        velocity of a particle is just its horizontal velocity.  The tangential
        displacement accumulates ``v_t dt`` and the spring force ``k_t * delta_t``
        is saturated at ``mu_p * N`` (Coulomb).  A particle that is not touching
        the plane loses its history, so the next touchdown starts unloaded.

        ``update_history=False`` (probe calls from diagnostics/energy audits)
        returns the force implied by the stored state without advancing it.
        """
        cfg = self.config
        eps = torch.finfo(self.dtype).eps
        mu = cfg.platen_friction_coefficient
        v_tangent = torch.where(mask[:, None], self.velocities, torch.zeros_like(self.velocities))
        v_tangent = torch.stack(
            (v_tangent[:, 0], v_tangent[:, 1], torch.zeros_like(v_tangent[:, 2])), dim=1)
        if update_history:
            tang_disp = tang_disp + v_tangent * self.dt
            tang_disp = torch.where(mask[:, None], tang_disp, torch.zeros_like(tang_disp))
        # The offset is the particle's own tangential travel relative to the plane
        # (the plane does not translate tangentially), so the spring pulls the
        # particle back towards its anchor: F = -k_t * delta, saturated at mu*N.
        spring = -self.platen_friction_k * tang_disp
        limit = mu * normal_mag
        magnitude = torch.linalg.vector_norm(spring, dim=1)
        scale = torch.where(magnitude > limit, limit / magnitude.clamp_min(eps),
                            torch.ones_like(magnitude))
        return spring * scale[:, None], tang_disp

    def _rolling_resistance_couple(
        self, ci: torch.Tensor, cj: torch.Tensor,
        cnormal: torch.Tensor, contact_mag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rolling resistance as a momentum-conserving couple (rotationless DEM).

        Declared law (see ``forces``): ``T_r = mu_r * R * |F_n|``. The 3-D
        bonded-sphere model carries no rotational degree of freedom, so the
        torque cannot act on a particle orientation; it is realised as an
        equal-and-opposite couple of tangential contact forces

            ``f_j = -mu_r |F_n| v_hat_rel``, ``f_i = -f_j``,

        where ``v_hat_rel`` is the unit relative-slip direction
        ``(v_j - v_i)_t / |(v_j - v_i)_t|`` in the contact tangent plane. Consequences, both verified analytically in
        ``docs/dem3d_rolling_resistance_verification.md``:

        * ``f_i + f_j = 0`` -- the couple conserves linear momentum;
        * ``f_j . v_rel <= 0`` -- the couple is strictly dissipative (it can
          never inject energy), and it vanishes as ``|v_t| -> 0``.

        The returned torques are diagnostic only: they are the moment of the
        couple force about each particle centre with lever arm ``R``, so
        ``|T| = mu_r R |F_n|`` reproduces the declared law. The equivalent
        rolling coefficient in the equal-sphere convention of Ai et al. (2011),
        ``M_lim = mu_r R* |F_n|`` with ``R* = R/2``, is therefore
        ``mu_r_Ai = 2 * mu_r_impl``.

        Returns ``(f_i, f_j, torque_i, torque_j)``, each ``(N, 3)``.
        """
        n_contacts = ci.numel()
        mu_r = self.config.rolling_resistance_coefficient
        R = self.config.radius

        relative_v = self.velocities[cj] - self.velocities[ci]
        vn = (relative_v * cnormal).sum(dim=1, keepdim=True)
        v_tangent = relative_v - vn * cnormal
        v_t_mag = torch.linalg.vector_norm(v_tangent, dim=1, keepdim=True)

        # Kinetic resistance opposes the *relative* slip velocity
        # v_rel = v_j - v_i: the force on j is -|f| * v_rel_hat and the force on
        # i is its reaction.  No force when there is no slip (v_t == 0).
        slip_hat = torch.where(
            v_t_mag > 1e-15,
            v_tangent / v_t_mag.clamp_min(torch.finfo(self.dtype).eps),
            torch.zeros_like(v_tangent),
        )
        f_mag = (mu_r * contact_mag)[:, None]
        force_j = -f_mag * slip_hat
        force_i = -force_j

        torque_i = R * torch.linalg.cross(cnormal, force_i)
        torque_j = -torque_i
        return force_i, force_j, torque_i, torque_j

    # ------------------------------------------------------------------
    # Main force evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forces(self, external_forces: torch.Tensor | None = None, *,
               update_fracture: bool = True,
               update_history: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate forces; default commits fracture for API compatibility.

        ``update_fracture=False`` evaluates the currently stored bond topology
        without changing fracture history or the stored last reaction.

        ``update_history=False`` evaluates the constitutive forces *without*
        advancing the Cundall-Strack tangential displacement or the diagnostic
        rolling-torque accumulator, so the call is side-effect free with respect
        to the friction history.  Probing routines (``diagnostics``, energy
        audits, invariance checks) must use it: with the default the call adds
        one ``v_tangent * dt`` slip increment per invocation, so merely
        monitoring a friction simulation changes its trajectory.

        Contact model (SIR-PC2|F05):
        - Normal: linear spring-dashpot with restitution-based damping.
        - Tangential: Cundall-Strack Coulomb-friction spring.
        - Rolling: rotationless rolling-resistance surrogate, |f| = mu_r * |F_n|
          applied as an equal-and-opposite tangential couple along the slip
          direction (equivalent torque T_r = mu_r * R * |F_n|).
        """
        cfg = self.config
        loads = self._loads(external_forces)
        eps = torch.finfo(self.dtype).eps
        # Permanent bond interactions retain compact topology and failure history.
        hertz = cfg.contact_model == "hertz"
        e_star_pairs, r_eff_pairs = self._hertz_pair_params
        e_star_rigid, r_eff_rigid = self._hertz_rigid_params
        i, j = self.pairs.T
        delta = self.positions[j] - self.positions[i]
        distance = torch.linalg.vector_norm(delta, dim=1)
        normal = delta / distance.clamp_min(eps)[:, None]
        normal = torch.where((distance > eps)[:, None], normal, self.initial_normals)
        extension = distance - self.rest_lengths
        if update_fracture:
            self._commit_fracture(extension)

        bond_mag = cfg.bond_stiffness * extension
        bond_force = bond_mag[:, None] * normal
        force = -cfg.drag * self.velocities
        if cfg.gravity > 0.0 or cfg.water_level is not None:
            # Opt-in body forces (gravity / Archimedes buoyancy / wet drag).
            # Disabled by default, so legacy trajectories stay bit-for-bit.
            force = force + self.body_forces()
        if loads is not None:
            # External nodal loads enter the force balance here. Dropping this
            # term silently removes every externally applied load from the
            # simulation (no error, zero response), so it must stay explicit.
            force = force + loads
        active_bond_force = torch.where(self.alive[:, None], bond_force, torch.zeros_like(bond_force))
        force.index_add_(0, i, active_bond_force)
        force.index_add_(0, j, -active_bond_force)

        contact_pairs = self._contact_candidate_pairs()
        friction_power = torch.zeros((), dtype=self.dtype, device=self.device)
        rolling_power = torch.zeros((), dtype=self.dtype, device=self.device)
        if contact_pairs.numel():
            ci, cj = contact_pairs.T
            cdelta = self.positions[cj] - self.positions[ci]
            cdistance = torch.linalg.vector_norm(cdelta, dim=1)
            overlap = (2 * cfg.radius - cdistance).clamp_min(0)
            near = overlap > 0
            ci, cj = ci[near], cj[near]
            cdelta, cdistance, overlap = cdelta[near], cdistance[near], overlap[near]
            if ci.numel():
                cnormal = cdelta / cdistance.clamp_min(eps)[:, None]
                bond_alive = self._alive_bond_for_pairs(torch.stack((ci, cj), dim=1))
                relative_v = self.velocities[cj] - self.velocities[ci]
                vn = (relative_v * cnormal).sum(dim=1)

                # --- Normal contact force with restitution-based damping ---
                # Effective mass of the relative coordinate.  Two *mobile*
                # spheres share the relative velocity, so m_eff = m/2.  A pair
                # with exactly one pinned particle (hard support: its velocity
                # is zeroed and its position reset every step) has an infinitely
                # massive partner, so only the free particle accelerates and
                # the correct value is m -- the same convention as
                # tensordem.restitution.effective_mass(m, None) and the
                # rigid-counterpart tool/platen branches.  Using m/2 there
                # under-damps by sqrt(2) and the realised restitution
                # overshoots e_target (docs/dem3d_fixed_support_reduced_mass.md).
                # The all-mobile case keeps the exact scalar expression, so
                # legacy trajectories stay bit-for-bit.
                m_eff: float | torch.Tensor = cfg.mass / 2.0  # equal-mass mobile spheres
                if bool(self.fixed.any()):
                    pinned = self.fixed[ci] ^ self.fixed[cj]
                    if bool(pinned.any()):
                        m_eff = torch.where(
                            pinned,
                            torch.as_tensor(cfg.mass, dtype=self.dtype, device=self.device),
                            torch.as_tensor(cfg.mass / 2.0, dtype=self.dtype,
                                            device=self.device),
                        )
                # Hertz: the elastic stiffness is the per-contact secant
                # stiffness k_s(delta), so both the penalty term and the
                # restitution-derived damper follow the local overlap.
                k_local = (hertz_secant_stiffness(overlap, e_star_pairs, r_eff_pairs)
                           if hertz else cfg.contact_stiffness)
                # Map restitution coefficient to damping
                if cfg.contact_damping_model == "absolute":
                    # Absolute dashpot convention shared with the tool/platen
                    # branches: gamma_n = c_n [N s/m], no mass/stiffness mapping.
                    # This is the branch that makes contact_damping observable in
                    # the pair law; the reference solution is the clipped-law
                    # RK4 integration in tensordem.restitution.
                    gamma_n = cfg.contact_damping
                elif cfg.restitution_coefficient < 1.0:
                    ln_e = math.log(cfg.restitution_coefficient)
                    abs_ln_e = -ln_e
                    zeta = abs_ln_e / math.sqrt(math.pi ** 2 + abs_ln_e ** 2)
                    # The pinned-pair branch makes m_eff a per-pair tensor;
                    # the all-mobile scalar path keeps math.sqrt bit-for-bit.
                    gamma_n = (2.0 * zeta * torch.sqrt(k_local * m_eff) if hertz
                               else 2.0 * zeta * (
                                   torch.sqrt(cfg.contact_stiffness * m_eff)
                                   if torch.is_tensor(m_eff)
                                   else math.sqrt(cfg.contact_stiffness * m_eff)))
                else:
                    gamma_n = 0.0

                contact_mag = (k_local * overlap - gamma_n * vn).clamp_min(0)
                contact_mag = torch.where(~bond_alive, contact_mag, torch.zeros_like(contact_mag))
                contact_force = -contact_mag[:, None] * cnormal

                # --- SIR-PC2|F05: Tangential Coulomb friction ---
                if cfg.friction_coefficient > 0.0:
                    # Retrieve tangential displacement state
                    tang_disp = self._get_tangential_state(ci, cj, cnormal)
                    # Integrate tangential displacement.  Only a force
                    # evaluation that owns the step may advance the history;
                    # a probe must see the stored state unchanged.
                    v_tangent = relative_v - vn[:, None] * cnormal
                    if update_history:
                        tang_disp = tang_disp + v_tangent * self.dt

                    # --- Hertz-Mindlin tangential stiffness ---
                    # For Hertz contacts, use Mindlin no-slip tangential stiffness:
                    #   k_t = 8 G* sqrt(R* delta)
                    # For linear contacts, use the constant contact_stiffness.
                    # The ratio k_t/k_n for equal spheres is 3/(1-nu) ~ 4.3 for nu=0.3.
                    if hertz:
                        k_t = hertz_mindlin.hertz_mindlin_tangential_stiffness(
                            overlap, e_star_pairs, self._hertz_g_star, r_eff_pairs)
                    else:
                        k_t = torch.full_like(overlap, cfg.contact_stiffness)

                    F_t_raw = k_t[:, None] * tang_disp
                    F_t_mag = torch.linalg.vector_norm(F_t_raw, dim=1)
                    F_limit = cfg.friction_coefficient * contact_mag

                    # Coulomb saturation
                    scale = torch.where(
                        F_t_mag > F_limit,
                        F_limit / F_t_mag.clamp_min(1e-15),
                        torch.ones_like(F_t_mag),
                    )
                    F_t = F_t_raw * scale[:, None]

                    # Power delivered to the particles by the friction force
                    # (force on i is F_t, on j is -F_t): P = F_t . (v_i - v_j).
                    # Negative during stick (energy stored in the tangential
                    # spring) and during slip (Coulomb dissipation).
                    friction_power = friction_power + (
                        F_t * (self.velocities[ci] - self.velocities[cj])
                    ).sum()

                    # Apply tangential force
                    force.index_add_(0, ci, F_t)
                    force.index_add_(0, cj, -F_t)

                    # Update tangential state (step owner only).
                    # Project onto friction cone: delta_t <- F_t / k_t
                    if update_history:
                        tang_disp_next = F_t / k_t[:, None]
                        self._update_tangential_state(ci, cj, cnormal, tang_disp_next)

                # --- SIR-PC2|F05: Rolling resistance torque ---
                if cfg.rolling_resistance_coefficient > 0.0:
                    rr_i, rr_j, torque_i, torque_j = self._rolling_resistance_couple(
                        ci, cj, cnormal, contact_mag,
                    )
                    # The couple is strictly dissipative: P = rr_i . (v_i - v_j)
                    # = -mu_r |F_n| |v_t| <= 0.
                    rolling_power = rolling_power + (
                        rr_i * (self.velocities[ci] - self.velocities[cj])
                    ).sum()
                    # Accumulate rolling torque (diagnostic only).  Guarded
                    # like the friction history so that probing does not inflate
                    # a running sum that no step ever resets.
                    if update_history:
                        self._rolling_torque.index_add_(0, ci, torque_i)
                        self._rolling_torque.index_add_(0, cj, torque_j)
                    # Apply the equal-and-opposite couple along the slip direction
                    force.index_add_(0, ci, rr_i)
                    force.index_add_(0, cj, rr_j)

                force.index_add_(0, ci, contact_force)
                force.index_add_(0, cj, -contact_force)

        if update_history:
            self._last_friction_power = float(friction_power)
            self._last_rolling_power = float(rolling_power)

        tool_delta = self.positions - self.tool_position
        tool_dist = torch.linalg.vector_norm(tool_delta, dim=1)
        tool_normal = tool_delta / tool_dist.clamp_min(eps)[:, None]
        fallback = torch.zeros_like(tool_normal); fallback[:, 2] = -1
        tool_normal = torch.where((tool_dist > eps)[:, None], tool_normal, fallback)
        tool_overlap = (cfg.radius + cfg.tool_radius - tool_dist).clamp_min(0)
        tool_vn = ((self.velocities - self.tool_velocity) * tool_normal).sum(dim=1)
        tool_elastic = (hertz_normal_force(tool_overlap, e_star_rigid, r_eff_rigid) if hertz
                        else cfg.contact_stiffness * tool_overlap)
        tool_mag = (tool_elastic - cfg.contact_damping * tool_vn).clamp_min(0)
        tool_mag = torch.where(tool_overlap > 0, tool_mag, torch.zeros_like(tool_mag))
        tool_force = tool_mag[:, None] * tool_normal
        reaction = -tool_force.sum(dim=0)
        force += tool_force

        # Optional infinite horizontal platens. Positive overlap means a particle
        # intersects the plane; damping opposes the relative normal approach.
        top_reaction_z = torch.zeros((), dtype=self.dtype, device=self.device)
        bottom_reaction_z = torch.zeros((), dtype=self.dtype, device=self.device)
        if cfg.top_platen_enabled:
            top_z = self.top_platen_start_z + self.time * cfg.top_platen_velocity
            top_overlap = (self.positions[:, 2] + cfg.radius - top_z).clamp_min(0.0)
            top_relative_v = self.velocities[:, 2] - cfg.top_platen_velocity
            top_elastic = (hertz_normal_force(top_overlap, e_star_rigid, r_eff_rigid) if hertz
                           else cfg.platen_stiffness * top_overlap)
            top_mag = (top_elastic
                       + cfg.platen_damping * top_relative_v).clamp_min(0.0)
            # A plane boundary acts only on particles that actually intersect it.
            # Without this mask the damping term also acts on particles far from
            # the plane (same convention as the spherical tool contact above).
            top_mag = torch.where(top_overlap > 0, top_mag, torch.zeros_like(top_mag))
            top_force_z = -top_mag
            force[:, 2] += top_force_z
            top_reaction_z = -top_force_z.sum()
            if cfg.platen_friction_coefficient > 0.0:
                top_friction, self._top_platen_tang_disp = self._plane_friction_force(
                    top_overlap > 0, top_mag, self._top_platen_tang_disp,
                    update_history=update_history)
                force += top_friction
                self.last_top_platen_friction.copy_(top_friction)
        if cfg.bottom_platen_enabled:
            bottom_z = self.bottom_platen_start_z + self.time * cfg.bottom_platen_velocity
            bottom_overlap = (bottom_z + cfg.radius - self.positions[:, 2]).clamp_min(0.0)
            bottom_relative_v = self.velocities[:, 2] - cfg.bottom_platen_velocity
            # Damper opposes relative normal approach: with the bottom platen
            # normal +z the damping term is -c*(v_z - v_platen). Using +c turns
            # the damper into a negative-damping (energy-injecting) term.
            bottom_elastic = (hertz_normal_force(bottom_overlap, e_star_rigid, r_eff_rigid)
                              if hertz else cfg.platen_stiffness * bottom_overlap)
            bottom_mag = (bottom_elastic
                          - cfg.platen_damping * bottom_relative_v).clamp_min(0.0)
            # Contact-masked like the top platen: no force on particles outside.
            bottom_mag = torch.where(bottom_overlap > 0, bottom_mag, torch.zeros_like(bottom_mag))
            bottom_force_z = bottom_mag
            force[:, 2] += bottom_force_z
            bottom_reaction_z = -bottom_force_z.sum()
            if cfg.platen_friction_coefficient > 0.0:
                bottom_friction, self._bottom_platen_tang_disp = self._plane_friction_force(
                    bottom_overlap > 0, bottom_mag, self._bottom_platen_tang_disp,
                    update_history=update_history)
                force += bottom_friction
                self.last_bottom_platen_friction.copy_(bottom_friction)

        if update_fracture:
            self.last_reaction.copy_(reaction)
        self.last_top_platen_reaction_z.copy_(top_reaction_z)
        self.last_bottom_platen_reaction_z.copy_(bottom_reaction_z)
        return force, reaction

    @torch.no_grad()
    def step(self, external_forces: torch.Tensor | None = None) -> None:
        """Single explicit Euler step."""
        force, _ = self.forces(external_forces)
        acc = force / self.config.mass
        # Fixed particles are hard supports: they neither accelerate nor move
        # away from their initial position.
        acc[self.fixed] = 0.0
        self.velocities += acc * self.dt
        self.velocities[self.fixed] = 0.0
        self.positions += self.velocities * self.dt
        self.positions[self.fixed] = self.initial_positions[self.fixed]
        self.time += self.dt
        self.step_count += 1
        if not bool(torch.isfinite(self.positions).all() and torch.isfinite(self.velocities).all()):
            raise FloatingPointError("non-finite DEM3D state; reduce dt or contact stiffness")

    @torch.no_grad()
    def diagnostics(self, external_forces: torch.Tensor | None = None) -> dict[str, float | int]:
        """Return a full diagnostic snapshot including forces, reactions, and energy."""
        force, reaction = self.forces(external_forces, update_fracture=False,
                                      update_history=False)
        support = -force[self.fixed].sum(dim=0)
        kinetic = 0.5 * self.config.mass * self.velocities.square().sum()
        # Recoverable elastic energy in surviving bonds only. Broken bonds are
        # represented separately by their recorded release-energy estimate.
        i, j = self.pairs.T
        distance = torch.linalg.vector_norm(self.positions[j] - self.positions[i], dim=1)
        extension = distance - self.rest_lengths
        bond_energy = (0.5 * self.config.bond_stiffness * extension.square() * self.alive).sum()
        contact_pairs = self._contact_candidate_pairs()
        contact_energy = torch.zeros((), dtype=self.dtype, device=self.device)
        if contact_pairs.numel():
            ci, cj = contact_pairs.T
            cdistance = torch.linalg.vector_norm(self.positions[cj] - self.positions[ci], dim=1)
            overlap = (2 * self.config.radius - cdistance).clamp_min(0)
            broken_or_unbonded = ~self._alive_bond_for_pairs(contact_pairs)
            if self.config.contact_model == "hertz":
                contact_energy = hertz_contact_potential(
                    overlap, *self._hertz_pair_params)[broken_or_unbonded].sum()
            else:
                contact_energy = (0.5 * self.config.contact_stiffness
                                  * overlap.square() * broken_or_unbonded).sum()
        tool_overlap = (self.config.radius + self.config.tool_radius
                        - torch.linalg.vector_norm(self.positions - self.tool_position, dim=1)).clamp_min(0)
        if self.config.contact_model == "hertz":
            tool_energy = hertz_contact_potential(tool_overlap, *self._hertz_rigid_params).sum()
        else:
            tool_energy = (0.5 * self.config.contact_stiffness * tool_overlap.square()).sum()
        top_platen_energy = torch.zeros((), dtype=self.dtype, device=self.device)
        if self.config.top_platen_enabled:
            top_z = self.top_platen_start_z + self.time * self.config.top_platen_velocity
            top_overlap = (self.positions[:, 2] + self.config.radius - top_z).clamp_min(0.0)
            top_platen_energy = (
                hertz_contact_potential(top_overlap, *self._hertz_rigid_params).sum()
                if self.config.contact_model == "hertz"
                else (0.5 * self.config.platen_stiffness * top_overlap.square()).sum())
        bottom_platen_energy = torch.zeros((), dtype=self.dtype, device=self.device)
        if self.config.bottom_platen_enabled:
            bottom_z = self.bottom_platen_start_z + self.time * self.config.bottom_platen_velocity
            bottom_overlap = (bottom_z + self.config.radius - self.positions[:, 2]).clamp_min(0.0)
            bottom_platen_energy = (
                hertz_contact_potential(bottom_overlap, *self._hertz_rigid_params).sum()
                if self.config.contact_model == "hertz"
                else (0.5 * self.config.platen_stiffness * bottom_overlap.square()).sum())
        platen_energy = top_platen_energy + bottom_platen_energy
        released = self.failure_energy_J.sum()
        return {
            "time": self.time,
            "tool_x": float(self.tool_position[0]), "tool_y": float(self.tool_position[1]),
            "tool_z": float(self.tool_position[2]),
            "reaction_x": float(reaction[0]), "reaction_y": float(reaction[1]),
            "reaction_z": float(reaction[2]),
            "top_platen_reaction_z_N": float(self.last_top_platen_reaction_z),
            "bottom_platen_reaction_z_N": float(self.last_bottom_platen_reaction_z),
            "boundary_reaction_x": float(support[0]), "boundary_reaction_y": float(support[1]),
            "boundary_reaction_z": float(support[2]),
            "broken_bonds": self.broken_bonds,
            "particle_count": len(self.positions),
            "kinetic_energy_J": float(kinetic),
            "bond_elastic_energy_J": float(bond_energy),
            "particle_contact_energy_J": float(contact_energy),
            "tool_contact_energy_J": float(tool_energy),
            "platen_contact_energy_J": float(platen_energy),
            "released_bond_energy_J": float(released),
            "mechanical_energy_J": float(kinetic + bond_energy + contact_energy + tool_energy + platen_energy),
            **self.body_force_ledger(),
        }


    def energy(self) -> dict[str, float]:
        """Return a snapshot of mechanical energy components (J)."""
        cfg = self.config
        kinetic = 0.5 * cfg.mass * (self.velocities ** 2).sum().item()
        i, j = self.pairs.T
        delta = self.positions[j] - self.positions[i]
        distance = torch.linalg.vector_norm(delta, dim=1)
        extension = distance - self.rest_lengths
        bond_energy = 0.5 * cfg.bond_stiffness * (extension[self.alive] ** 2).sum().item()
        contact_energy = 0.0
        contact_pairs = self._contact_candidate_pairs()
        if contact_pairs.numel():
            ci, cj = contact_pairs.T
            cdelta = self.positions[cj] - self.positions[ci]
            cdistance = torch.linalg.vector_norm(cdelta, dim=1)
            overlap = (2 * cfg.radius - cdistance).clamp_min(0)
            near = overlap > 0
            ci, cj = ci[near], cj[near]
            overlap = overlap[near]
            if ci.numel():
                bond_alive = self._alive_bond_for_pairs(torch.stack((ci, cj), dim=1))
                active = ~bond_alive & (overlap > 0)
                if cfg.contact_model == "hertz":
                    contact_energy = float(hertz_contact_potential(
                        overlap[active], *self._hertz_pair_params).sum().item())
                else:
                    contact_energy = 0.5 * cfg.contact_stiffness * (overlap[active] ** 2).sum().item()
        tool_delta = self.positions - self.tool_position
        tool_dist = torch.linalg.vector_norm(tool_delta, dim=1)
        tool_overlap = (cfg.radius + cfg.tool_radius - tool_dist).clamp_min(0)
        if cfg.contact_model == "hertz":
            tool_energy = float(hertz_contact_potential(
                tool_overlap, *self._hertz_rigid_params).sum().item())
        else:
            tool_energy = 0.5 * cfg.contact_stiffness * (tool_overlap ** 2).sum().item()
        platen_energy = 0.0
        if cfg.top_platen_enabled:
            top_z = self.top_platen_start_z + self.time * cfg.top_platen_velocity
            top_overlap = (self.positions[:, 2] + cfg.radius - top_z).clamp_min(0.0)
            platen_energy += (float(hertz_contact_potential(
                top_overlap, *self._hertz_rigid_params).sum().item())
                if cfg.contact_model == "hertz"
                else 0.5 * cfg.platen_stiffness * (top_overlap ** 2).sum().item())
        if cfg.bottom_platen_enabled:
            bottom_z = self.bottom_platen_start_z + self.time * cfg.bottom_platen_velocity
            bottom_overlap = (bottom_z + cfg.radius - self.positions[:, 2]).clamp_min(0.0)
            platen_energy += (float(hertz_contact_potential(
                bottom_overlap, *self._hertz_rigid_params).sum().item())
                if cfg.contact_model == "hertz"
                else 0.5 * cfg.platen_stiffness * (bottom_overlap ** 2).sum().item())
        released = 0.5 * cfg.bond_stiffness * (extension[~self.alive] ** 2).sum().item()
        return {
            "time_s": self.time,
            "step": self.step_count,
            "broken_bonds": self.broken_bonds,
            "particle_count": len(self.positions),
            "kinetic_energy_J": float(kinetic),
            "bond_elastic_energy_J": float(bond_energy),
            "particle_contact_energy_J": float(contact_energy),
            "tool_contact_energy_J": float(tool_energy),
            "platen_contact_energy_J": float(platen_energy),
            "released_bond_energy_J": float(released),
            "mechanical_energy_J": float(kinetic + bond_energy + contact_energy + tool_energy + platen_energy),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return restart-friendly state (tensors remain on the active device)."""
        return {
            "dimension": 3, "config": asdict(self.config), "dt": self.dt,
            "time": self.time, "step_count": self.step_count,
            "positions": self.positions.clone(), "velocities": self.velocities.clone(),
            "initial_positions": self.initial_positions.clone(), "fixed": self.fixed.clone(),
            "pairs": self.pairs.clone(), "bonded": self.bonded.clone(), "alive": self.alive.clone(),
            "failure_mode": self.failure_mode.clone(), "failure_time_s": self.failure_time_s.clone(),
            "failure_extension_m": self.failure_extension_m.clone(), "failure_energy_J": self.failure_energy_J.clone(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot created by :meth:`state_dict` after strict validation."""
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary returned by state_dict()")
        required = {
            "dimension", "config", "dt", "time", "step_count", "positions",
            "velocities", "initial_positions", "fixed", "pairs", "bonded",
            "alive", "failure_mode", "failure_time_s", "failure_extension_m",
            "failure_energy_J",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
        if state["dimension"] != 3:
            raise ValueError("checkpoint dimension must be 3")
        saved_config = state["config"]
        if not isinstance(saved_config, dict):
            raise ValueError("checkpoint config must be a dictionary")
        # Backward compatibility: configs saved before plane-platen support omit
        # new optional keys, whose defaults preserve the previous solver behavior.
        normalized_config = {**asdict(DEM3DConfig()), **saved_config}
        if normalized_config != asdict(self.config):
            raise ValueError("checkpoint config does not match this solver configuration")
        if not math.isfinite(float(state["dt"])) or float(state["dt"]) != self.dt:
            raise ValueError("checkpoint dt does not match this solver")
        time_value = float(state["time"])
        step_value = state["step_count"]
        if not math.isfinite(time_value) or time_value < 0:
            raise ValueError("checkpoint time must be finite and nonnegative")
        if isinstance(step_value, bool) or not isinstance(step_value, int) or step_value < 0:
            raise ValueError("checkpoint step_count must be a nonnegative integer")

        tensor_shapes = {
            "positions": tuple(self.positions.shape),
            "velocities": tuple(self.velocities.shape),
            "initial_positions": tuple(self.initial_positions.shape),
            "fixed": tuple(self.fixed.shape),
            "pairs": tuple(self.pairs.shape),
            "bonded": tuple(self.bonded.shape),
            "alive": tuple(self.alive.shape),
            "failure_mode": tuple(self.failure_mode.shape),
            "failure_time_s": tuple(self.failure_time_s.shape),
            "failure_extension_m": tuple(self.failure_extension_m.shape),
            "failure_energy_J": tuple(self.failure_energy_J.shape),
        }
        staged: dict[str, torch.Tensor] = {}
        for name, expected_shape in tensor_shapes.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                raise ValueError(f"checkpoint {name} must be a tensor with shape {expected_shape}")
            target = getattr(self, name)
            # Validate storage type before conversion: e.g. float NaN->bool
            # silently becomes True, and fractional pair indices truncate.
            if value.dtype != target.dtype:
                raise ValueError(f"checkpoint {name} dtype must be {target.dtype}, got {value.dtype}")
            value = value.to(device=self.device)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"checkpoint {name} contains non-finite values")
            staged[name] = value
        if not torch.equal(staged["initial_positions"], self.initial_positions):
            raise ValueError("checkpoint reference geometry does not match this solver")
        if not torch.equal(staged["pairs"], self.pairs):
            raise ValueError("checkpoint pair topology does not match this solver")
        if not torch.equal(staged["bonded"], self.bonded):
            raise ValueError("checkpoint bond topology does not match this solver")
        if bool((staged["alive"] & ~staged["bonded"]).any()):
            raise ValueError("checkpoint marks a non-bond pair as alive")
        if not torch.equal(staged["fixed"], self.fixed):
            raise ValueError("checkpoint fixed-boundary mask does not match this solver")

        for name in ("positions", "velocities", "initial_positions", "fixed", "pairs",
                     "bonded", "alive", "failure_mode", "failure_time_s",
                     "failure_extension_m", "failure_energy_J"):
            getattr(self, name).copy_(staged[name])
        self.time = time_value
        self.step_count = step_value
        self.last_reaction.zero_()
