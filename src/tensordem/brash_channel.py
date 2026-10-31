"""Brash-ice channel resistance DEM engine (TensorDEM).

Physics
-------
The model represents a brash-ice channel as a polydisperse cloud of floating
spherical ice pieces confined laterally by channel walls.  A rigid hull (bow
wedge + parallel sides, extruded over the brash layer) translates along the
channel at a prescribed speed; the ice reaction on the hull is the channel
resistance.

Force model (translational degrees of freedom only):

* gravity                     ``-m g z``
* buoyancy                    Archimedes force from the submerged sphere-cap
                              fraction ``f(u)``, ``u = -z/r`` (waterline ``z=0``),
                              with the closed-form potential ``U_b`` used below
* particle-particle contact   linear spring + dashpot, viscous-Coulomb friction
* hull contact                same law, plan-view rounded-wedge signed distance
* channel walls               spring-dashpot with friction (sides and floor)
* hydrodynamic drag           linear ``-c_d (v - v_water)``, ``v_water = 0``

Time integration is symplectic (semi-implicit) Euler.  Since
``v_{n+1} = v_n + dt F_n/m`` and ``x_{n+1} = x_n + dt v_{n+1}``, the kinetic
energy increment equals ``sum_i F_i . v_mid dt`` *exactly* for
``v_mid = (v_n + v_{n+1})/2``.  Each force category is integrated with the same
trapezoidal velocity, so the work/energy ledger closes to round-off and the
residual is a genuine consistency check of the contact laws, the hull reaction
pair and the integrator (rather than an artefact of a small timestep).

Brash ice is already fragmented, so no inter-particle bonds are used; channel
"consolidation" is represented in the engineering layer
(:mod:`tensordem.channel_resistance`) as a channel-age effect.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

GRAVITY = 9.81

__all__ = [
    "BrashChannelConfig",
    "HullSpec",
    "BrashChannelDEM",
    "generate_field",
    "submerged_fraction",
    "buoyancy_force_and_potential",
]


def submerged_fraction(u: torch.Tensor) -> torch.Tensor:
    """Submerged sphere-cap volume fraction for centre depth ``u = d/r``."""
    uc = u.clamp(-1.0, 1.0)
    return 0.5 + 0.75 * uc - 0.25 * uc**3


def buoyancy_force_and_potential(
    z: torch.Tensor, radius: torch.Tensor, rho_water: float, gravity: float = GRAVITY
) -> tuple[torch.Tensor, torch.Tensor]:
    """Archimedes force (N, +z up) and buoyancy potential energy (J).

    ``z`` is the centre elevation relative to the waterline (``z = 0``).
    ``U_b = int F_z dz`` of the cap model, zero for a fully emerged piece and
    linear in ``z`` for a fully submerged one.
    """
    volume = (4.0 / 3.0) * math.pi * radius**3
    u = (-z) / radius
    f_cap = submerged_fraction(u)
    f = torch.where(u > 1.0, torch.ones_like(u), f_cap)
    f = torch.where(u < -1.0, torch.zeros_like(u), f)
    force = rho_water * volume * gravity * f
    uc = u.clamp(-1.0, 1.0)
    g_cap = 0.5 * uc + 0.375 * uc**2 - 0.0625 * uc**4 + 0.1875
    g_val = torch.where(u > 1.0, 1.0 + (u - 1.0), g_cap)
    g_val = torch.where(u < -1.0, torch.zeros_like(u), g_val)
    potential = rho_water * volume * gravity * radius * g_val
    return force, potential


@dataclass(frozen=True)
class HullSpec:
    """Rigid plan-view hull boundary (rounded wedge bow + parallel sides)."""

    beam_m: float = 4.0
    bow_half_angle_deg: float = 28.0
    bow_length_m: float = 1.5
    body_length_m: float = 1.5
    tip_radius_m: float = 0.15
    y_center_m: float = 2.5
    x_apex_m: float = 0.0
    speed_ms: float = 0.5
    name: str = "hull"

    def __post_init__(self) -> None:
        for name in ("beam_m", "bow_length_m", "body_length_m", "tip_radius_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 5.0 <= self.bow_half_angle_deg <= 80.0:
            raise ValueError("bow_half_angle_deg must be in [5, 80]")
        if not math.isfinite(self.speed_ms) or self.speed_ms <= 0:
            raise ValueError("speed_ms must be positive")

    @property
    def tan_half_angle(self) -> float:
        return math.tan(math.radians(self.bow_half_angle_deg))

    @property
    def length_overall_m(self) -> float:
        return self.bow_length_m + self.body_length_m


@dataclass(frozen=True)
class BrashChannelConfig:
    """Geometry, ice state and contact parameters of the brash channel."""

    length_m: float = 6.0
    width_m: float = 5.0
    concentration: float = 0.6
    thickness_m: float = 1.0
    mean_radius_m: float = 0.10
    radius_spread: float = 0.20
    ice_density: float = 917.0
    water_density: float = 1025.0
    normal_stiffness: float = 1.0e6
    damping_ratio: float = 0.6
    friction_mu: float = 0.30
    hull_friction_mu: float = 0.15
    wall_friction_mu: float = 0.30
    drag_coefficient: float = 40.0
    floor_depth_m: float = 3.0
    stiffness_reference_radius_m: float = 0.13
    mass_scale: float = 1.0
    dt: float | None = None
    seed: int = 20261031
    deoverlap_iterations: int = 12
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("length_m", "width_m", "thickness_m", "mean_radius_m",
                     "ice_density", "water_density", "normal_stiffness",
                     "floor_depth_m", "mass_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.concentration <= 0.65:
            raise ValueError("concentration must be in (0, 0.65]")
        if not 0.0 <= self.radius_spread < 0.5:
            raise ValueError("radius_spread must be in [0, 0.5)")
        if not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError("damping_ratio must be in [0, 1)")
        for name in ("friction_mu", "hull_friction_mu", "wall_friction_mu", "drag_coefficient"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0):
            raise ValueError("dt must be finite and positive")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")

    @property
    def contact_stiffness(self) -> float:
        """Contact stiffness scaled with the particle radius (Hertz-like k ~ E* r).

        Keeping ``k/r`` constant makes the *relative* penetration at a given
        load independent of the particle resolution, which is what a resolution
        study needs; at the calibration radius the reference stiffness is used
        unchanged.
        """
        return self.normal_stiffness * (self.mean_radius_m / self.stiffness_reference_radius_m)

    @property
    def mean_particle_volume(self) -> float:
        return (4.0 / 3.0) * math.pi * self.mean_radius_m**3

    @property
    def target_ice_volume(self) -> float:
        return self.concentration * self.length_m * self.width_m * self.thickness_m

    @property
    def particle_count(self) -> int:
        return max(1, int(round(self.target_ice_volume / self.mean_particle_volume)))

    def recommended_dt(self, min_mass: float | None = None) -> float:
        """Stability-bound timestep for the worst-case 27-neighbour stencil."""
        r = self.mean_radius_m * (1.0 - self.radius_spread)
        mass = self.ice_density * self.mass_scale * (4.0 / 3.0) * math.pi * r**3
        if min_mass is not None:
            mass = min(mass, min_mass)
        k_eff = 2.0 * 27.0 * self.contact_stiffness
        return 0.2 / math.sqrt(k_eff / mass)


def generate_field(config: BrashChannelConfig) -> dict[str, torch.Tensor]:
    """Polydisperse brash field on a jittered lattice matching the concentration."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    count = config.particle_count
    radii = config.mean_radius_m * (
        1.0 + config.radius_spread * (2.0 * torch.rand(count, generator=generator, dtype=torch.float64) - 1.0)
    )
    volumes = (4.0 / 3.0) * math.pi * radii**3
    scale = (config.target_ice_volume / volumes.sum()).pow(1.0 / 3.0)
    radii = radii * scale
    volumes = (4.0 / 3.0) * math.pi * radii**3

    # One particle per lattice cell; the radii already sum to the target ice
    # volume, so the layer thickness follows from the cell aspect ratio.
    cell = (config.length_m * config.width_m * config.thickness_m / count) ** (1.0 / 3.0)
    nx = max(1, int(round(config.length_m / cell)))
    ny = max(1, int(round(config.width_m / cell)))
    nz = max(1, int(round(config.thickness_m / cell)))
    while nx * ny * nz < count:
        nz += 1
        if nx * ny * nz < count:
            ny += 1
        if nx * ny * nz < count:
            nx += 1
    total_cells = nx * ny * nz
    sites = torch.randperm(total_cells, generator=generator)[:count]
    iz = sites % nz
    iy = (sites // nz) % ny
    ix = sites // (nz * ny)
    dx = config.length_m / nx
    dy = config.width_m / ny
    dz = config.thickness_m / nz
    jitter = 0.2
    positions = torch.stack(
        [
            (ix.to(torch.float64) + 0.5 + jitter * (torch.rand(count, generator=generator, dtype=torch.float64) - 0.5)) * dx,
            (iy.to(torch.float64) + 0.5 + jitter * (torch.rand(count, generator=generator, dtype=torch.float64) - 0.5)) * dy,
            (iz.to(torch.float64) + 0.5) * dz,
        ],
        dim=1,
    )
    positions[:, 0] = positions[:, 0].clamp(radii, config.length_m - radii)
    positions[:, 1] = positions[:, 1].clamp(radii, config.width_m - radii)
    positions[:, 2] = positions[:, 2].clamp(min=-(config.floor_depth_m - radii))
    # Place the pile bottom so that the assembly floats in Archimedes
    # equilibrium: rho_ice/rho_water = 89.5 % of the ice volume is submerged.
    # The current z is 0.5*dz above the pile bottom.
    submergence = config.ice_density / config.water_density
    positions[:, 2] += -submergence * config.thickness_m
    return {"positions": positions, "radii": radii, "volumes": volumes}


class BrashChannelDEM:
    """Translating-hull brash-ice channel DEM with a closed energy ledger."""

    def __init__(self, config: BrashChannelConfig, hulls: Sequence[HullSpec]) -> None:
        if not hulls:
            raise ValueError("at least one hull is required")
        self.config = config
        self.hulls = list(hulls)
        self.device = torch.device(config.device)
        self.dtype = torch.float64
        state = generate_field(config)
        self.positions = state["positions"].to(self.device)
        self.radii = state["radii"].to(self.device)
        self.volumes = state["volumes"].to(self.device)
        # Physical mass drives every force and potential; the *inertial* mass may
        # be scaled (DEM density scaling) so that the explicit timestep grows while
        # the quasi-static force balance -- and therefore the resistance -- stays
        # physical.
        self.mass = config.ice_density * self.volumes
        self.inertia_mass = self.mass * config.mass_scale
        self.velocities = torch.zeros_like(self.positions)
        self.time = 0.0
        self.dt = (
            config.dt
            if config.dt is not None
            else config.recommended_dt(float(self.inertia_mass.min()))
        )
        self.hull_x = [float(h.x_apex_m) for h in self.hulls]
        self.step_index = 0
        self.max_overlap_ratio = 0.0
        self.clamped_contacts = 0
        self.ledger = {
            "bow_work_j": 0.0,
            "contact_dissipation_j": 0.0,
            "hull_slide_dissipation_j": 0.0,
            "wall_dissipation_j": 0.0,
            "drag_dissipation_j": 0.0,
        }
        self._pairs: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pairs_step = -10**9
        self._rebuild_interval = 40
        self._deoverlap(iterations=int(config.deoverlap_iterations))
        self.initial_energy = self.energy_state()

    # ------------------------------------------------------------------ geometry
    @property
    def lower_corner(self) -> torch.Tensor:
        c = self.config
        return torch.tensor(
            [-2.0, -0.5, -(c.floor_depth_m + 2.0 * c.mean_radius_m)],
            dtype=self.dtype, device=self.device,
        )

    @property
    def upper_corner(self) -> torch.Tensor:
        c = self.config
        return torch.tensor(
            [c.length_m + 2.0, c.width_m + 0.5, c.thickness_m], dtype=self.dtype, device=self.device
        )

    # --------------------------------------------------------------- neighbour list
    def _build_pairs(self) -> None:
        pos = self.positions
        lo, hi = self.lower_corner, self.upper_corner
        r_max = float(self.radii.max())
        cutoff = 2.0 * r_max
        skin = 0.8 * float(self.radii.mean())
        cell = cutoff + skin
        ncell = ((hi - lo) / cell).floor().clamp(min=1).long()
        cs = (hi - lo) / ncell
        zero = torch.zeros(3, dtype=torch.long, device=self.device)
        ridx = ((pos - lo) / cs).floor().long().clamp(min=zero, max=ncell - 1)
        lin = (ridx[:, 0] * ncell[1] + ridx[:, 1]) * ncell[2] + ridx[:, 2]
        counts = torch.bincount(lin, minlength=int(ncell.prod()))
        starts = torch.cumsum(counts, 0) - counts
        order = torch.argsort(lin)
        index = torch.arange(len(pos), dtype=torch.long, device=self.device)
        idx_i: list[torch.Tensor] = []
        idx_j: list[torch.Tensor] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cand = ridx + torch.tensor([dx, dy, dz], dtype=torch.long, device=self.device)
                    valid = ((cand >= 0) & (cand < ncell)).all(dim=1)
                    cc = cand.clamp(min=zero, max=ncell - 1)
                    cl = (cc[:, 0] * ncell[1] + cc[:, 1]) * ncell[2] + cc[:, 2]
                    cnt = counts[cl] * valid.to(counts.dtype)
                    total = int(cnt.sum())
                    if total == 0:
                        continue
                    owner = torch.repeat_interleave(index, cnt)
                    base = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
                    block_start = torch.repeat_interleave(starts[cl], cnt)
                    within = torch.arange(total, dtype=torch.long, device=self.device) - base
                    partner = order[block_start + within]
                    if dx == 0 and dy == 0 and dz == 0:
                        keep = owner < partner
                        owner, partner = owner[keep], partner[keep]
                    idx_i.append(owner)
                    idx_j.append(partner)
        i = torch.cat(idx_i)
        j = torch.cat(idx_j)
        delta = pos[j] - pos[i]
        dist = torch.linalg.norm(delta, dim=1)
        keep = dist <= (cutoff + 2.0 * skin)
        self._pairs = (i[keep], j[keep])
        self._pairs_step = self.step_index

    def _deoverlap(self, iterations: int = 40) -> int:
        """Projection-based de-overlapping of the initial lattice packing."""
        moved = 0
        for _ in range(iterations):
            self._build_pairs()
            i, j = self._pairs  # type: ignore[misc]
            delta = self.positions[j] - self.positions[i]
            dist = torch.linalg.norm(delta, dim=1).clamp(min=1e-12)
            overlap = (self.radii[i] + self.radii[j]) - dist
            active = overlap > 1e-9
            if not bool(active.any()):
                break
            i, j = i[active], j[active]
            overlap = overlap[active]
            dist = dist[active]
            normal = delta[active] / dist.unsqueeze(1)
            correction = 0.5 * overlap.unsqueeze(1) * normal
            self.positions.index_add_(0, j, correction)
            self.positions.index_add_(0, i, -correction)
            moved += int(active.sum())
        self.positions[:, 0] = self.positions[:, 0].clamp(
            self.radii, self.config.length_m - self.radii
        )
        self.positions[:, 1] = self.positions[:, 1].clamp(
            self.radii, self.config.width_m - self.radii
        )
        self._pairs = None
        return moved

    # -------------------------------------------------------------------- energies
    def energy_state(self) -> dict[str, float]:
        _, u_b = buoyancy_force_and_potential(
            self.positions[:, 2], self.radii, self.config.water_density
        )
        kinetic = float((0.5 * self.inertia_mass * (self.velocities**2).sum(dim=1)).sum())
        gravity = float((self.mass * GRAVITY * (self.positions[:, 2] + self.config.floor_depth_m)).sum())
        return {
            "kinetic_j": kinetic,
            "potential_gravity_j": gravity,
            "potential_buoyancy_j": float(u_b.sum()),
        }

    # ------------------------------------------------------------------- forces
    def _particle_contacts(self) -> tuple[torch.Tensor, float, float, float]:
        """Particle-particle contact forces and mechanical diagnostics."""
        n = len(self.positions)
        force = torch.zeros_like(self.positions)
        if self._pairs is None or (self.step_index - self._pairs_step) >= self._rebuild_interval:
            self._build_pairs()
        i, j = self._pairs  # type: ignore[misc]
        if i.numel() == 0:
            return force, 0.0, 0.0, 0.0
        pos, vel, rad = self.positions, self.velocities, self.radii
        delta = pos[j] - pos[i]
        dist = torch.linalg.norm(delta, dim=1)
        gap = rad[i] + rad[j] - dist
        active = gap > 0.0
        overlap_ratio = 0.0
        if not bool(active.any()):
            return force, 0.0, 0.0, 0.0
        i_a, j_a = i[active], j[active]
        gap = gap[active]
        dist = dist[active]
        overlap_ratio = float((gap / (rad[i_a] + rad[j_a])).max())
        normal = delta[active] / dist.unsqueeze(1)
        mi = self.inertia_mass[i_a]
        mj = self.inertia_mass[j_a]
        m_eff = (mi * mj) / (mi + mj)
        c_n = 2.0 * self.config.damping_ratio * torch.sqrt(self.config.contact_stiffness * m_eff)
        v_rel = vel[j_a] - vel[i_a]
        v_n = (v_rel * normal).sum(dim=1, keepdim=True)
        f_n = self.config.contact_stiffness * gap.unsqueeze(1) - c_n.unsqueeze(1) * v_n
        f_n = f_n.clamp(min=0.0)
        v_t = v_rel - v_n * normal
        v_t_mag = torch.linalg.norm(v_t, dim=1, keepdim=True)
        c_t = 0.5 * c_n
        f_t_mag = torch.minimum(
            self.config.friction_mu * f_n, c_t.unsqueeze(1) * v_t_mag
        )
        f_t = -f_t_mag * v_t / v_t_mag.clamp(min=1e-12)
        contact = f_n * normal + f_t
        force.index_add_(0, j_a, contact)
        force.index_add_(0, i_a, -contact)
        # Dissipation is evaluated by the caller through the applied forces.
        return force, overlap_ratio, float(f_t.abs().sum()), 0.0

    def _hull_contact(self) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
        """Hull contact forces on particles plus per-hull reaction diagnostics."""
        n = len(self.positions)
        force = torch.zeros_like(self.positions)
        reaction = torch.zeros(len(self.hulls), dtype=self.dtype, device=self.device)
        stats: list[dict[str, float]] = []
        pos, vel, rad = self.positions, self.velocities, self.radii
        for hidx, hull in enumerate(self.hulls):
            x_apex = self.hull_x[hidx]
            y_c = hull.y_center_m
            half_beam = 0.5 * hull.beam_m
            tan_b = hull.tan_half_angle
            sin_b = math.sin(math.radians(hull.bow_half_angle_deg))
            cos_b = math.cos(math.radians(hull.bow_half_angle_deg))
            ax = pos[:, 0] - x_apex
            ay = pos[:, 1] - y_c
            abs_ay = ay.abs()
            # Primitive 1/2: starboard / port bow face (valid inside bow length).
            s_face = abs_ay / tan_b
            # Bow faces exist aft of the apex, laterally out to the shoulder.
            face_valid = (
                (ax <= 0.0) & (ax >= -hull.bow_length_m)
                & (abs_ay <= half_beam + rad)
            )
            d_star = sin_b * ax + cos_b * ay
            d_port = sin_b * ax - cos_b * ay
            npc = len(pos)
            n_star = torch.tensor([sin_b, cos_b, 0.0], dtype=self.dtype, device=self.device).expand(npc, 3)
            n_port = torch.tensor([sin_b, -cos_b, 0.0], dtype=self.dtype, device=self.device).expand(npc, 3)
            # Primitive 3: rounded tip.
            d_tip = torch.linalg.norm(torch.stack([ax, ay], dim=1), dim=1) - hull.tip_radius_m
            n_tip = torch.stack([ax, ay, torch.zeros_like(ax)], dim=1) / torch.linalg.norm(
                torch.stack([ax, ay], dim=1), dim=1
            ).clamp(min=1e-12).unsqueeze(1)
            # Primitive 4/5: starboard / port parallel side.
            d_side = abs_ay - half_beam
            n_side = torch.stack(
                [torch.zeros_like(ay), torch.sign(ay).clamp(min=-1.0, max=1.0), torch.zeros_like(ay)], dim=1
            )
            side_valid = (ax <= -hull.bow_length_m) & (ax > -hull.bow_length_m - hull.body_length_m)
            big = torch.full_like(ax, -1e9)
            pen = torch.stack(
                [
                    torch.where(face_valid, rad - d_star, big),
                    torch.where(face_valid, rad - d_port, big),
                    rad - d_tip,
                    torch.where(side_valid, rad - d_side, big),
                    torch.where(side_valid, rad - d_side, big),
                ],
                dim=1,
            )
            normals = torch.stack([n_star, n_port, n_tip, n_side, -n_side], dim=1)
            best_pen, best_idx = pen.max(dim=1)
            touching = best_pen > 0.0
            count = int(touching.sum())
            if count == 0:
                stats.append({"hull": hull.name, "contacts": 0.0, "mean_penetration_m": 0.0})
                continue
            sel = torch.nonzero(touching, as_tuple=False).squeeze(1)
            pen_v = best_pen[sel].clamp(max=0.5 * rad[sel])
            clamped = int((best_pen[sel] > 0.5 * rad[sel]).sum())
            self.clamped_contacts += clamped
            normal = normals[sel, best_idx[sel], :]
            normal = normal / torch.linalg.norm(normal, dim=1, keepdim=True).clamp(min=1e-12)
            m_v = self.inertia_mass[sel]
            c_n = 2.0 * self.config.damping_ratio * torch.sqrt(
                self.config.contact_stiffness * m_v
            )
            v_hull = torch.zeros_like(normal)
            v_hull[:, 0] = hull.speed_ms
            v_rel = vel[sel] - v_hull
            v_n = (v_rel * normal).sum(dim=1, keepdim=True)
            f_n = (self.config.contact_stiffness * pen_v.unsqueeze(1) - c_n.unsqueeze(1) * v_n).clamp(min=0.0)
            v_t = v_rel - v_n * normal
            v_t_mag = torch.linalg.norm(v_t, dim=1, keepdim=True)
            f_t_mag = torch.minimum(
                self.config.hull_friction_mu * f_n,
                0.5 * c_n.unsqueeze(1) * v_t_mag,
            )
            f_t = -f_t_mag * v_t / v_t_mag.clamp(min=1e-12)
            contact = f_n * normal + f_t
            force.index_add_(0, sel, contact)
            reaction[hidx] = contact[:, 0].sum()
            stats.append(
                {
                    "hull": hull.name,
                    "contacts": float(count),
                    "mean_penetration_m": float(pen_v.mean()),
                }
            )
        return force, reaction, stats

    def _wall_contact(self) -> torch.Tensor:
        pos, vel, rad = self.positions, self.velocities, self.radii
        cfg = self.config
        force = torch.zeros_like(pos)
        for axis, limit, sign in ((1, 0.0, 1.0), (1, cfg.width_m, -1.0), (2, -cfg.floor_depth_m, 1.0)):
            if axis == 1:
                gap = (pos[:, axis] - limit) * sign
            else:
                gap = pos[:, axis] - limit
            pen = rad - gap
            active = pen > 0.0
            if not bool(active.any()):
                continue
            sel = torch.nonzero(active, as_tuple=False).squeeze(1)
            pen_v = pen[sel].clamp(max=0.5 * rad[sel])
            normal = torch.zeros(len(sel), 3, dtype=self.dtype, device=self.device)
            normal[:, axis] = sign if axis == 1 else 1.0
            m_v = self.inertia_mass[sel]
            c_n = 2.0 * cfg.damping_ratio * torch.sqrt(cfg.contact_stiffness * m_v)
            v_rel = vel[sel]
            v_n = (v_rel * normal).sum(dim=1, keepdim=True)
            f_n = (cfg.contact_stiffness * pen_v.unsqueeze(1) - c_n.unsqueeze(1) * v_n).clamp(min=0.0)
            v_t = v_rel - v_n * normal
            v_t_mag = torch.linalg.norm(v_t, dim=1, keepdim=True)
            f_t_mag = torch.minimum(cfg.wall_friction_mu * f_n, 0.5 * c_n.unsqueeze(1) * v_t_mag)
            contact = f_n * normal - f_t_mag * v_t / v_t_mag.clamp(min=1e-12)
            force.index_add_(0, sel, contact)
        return force

    # ------------------------------------------------------------------- stepping
    def step(self, record: bool = True) -> dict[str, float]:
        cfg = self.config
        pos, vel, mass = self.positions, self.velocities, self.mass
        gravity_force = torch.zeros_like(pos)
        gravity_force[:, 2] = -mass * GRAVITY
        buoy_force, _ = buoyancy_force_and_potential(pos[:, 2], self.radii, cfg.water_density)
        gravity_force[:, 2] += buoy_force
        f_pp, overlap_ratio, _, _ = self._particle_contacts()
        f_hull, hull_reaction, hull_stats = self._hull_contact()
        f_wall = self._wall_contact()
        f_drag = -cfg.drag_coefficient * (self.radii / cfg.mean_radius_m).unsqueeze(1) * vel
        total = gravity_force + f_pp + f_hull + f_wall + f_drag
        accel = total / self.inertia_mass.unsqueeze(1)
        v_mid = vel + 0.5 * self.dt * accel
        if record:
            bow_power = sum(
                float(hull_reaction[k]) * self.hulls[k].speed_ms for k in range(len(self.hulls))
            )
            hull_power_on_ice = float((f_hull * v_mid).sum())
            self.ledger["bow_work_j"] += bow_power * self.dt
            self.ledger["contact_dissipation_j"] += float(-(f_pp * v_mid).sum()) * self.dt
            self.ledger["wall_dissipation_j"] += float(-(f_wall * v_mid).sum()) * self.dt
            self.ledger["drag_dissipation_j"] += float(-(f_drag * v_mid).sum()) * self.dt
            self.ledger["hull_slide_dissipation_j"] += (bow_power - hull_power_on_ice) * self.dt
        new_vel = vel + self.dt * accel
        self.positions = pos + self.dt * new_vel
        self.velocities = new_vel
        for hidx, hull in enumerate(self.hulls):
            self.hull_x[hidx] += hull.speed_ms * self.dt
        self.time += self.dt
        self.step_index += 1
        self.max_overlap_ratio = max(self.max_overlap_ratio, overlap_ratio)
        self.last_hull_stats = hull_stats
        self.last_hull_reaction = [float(x) for x in hull_reaction]
        return {
            "time_s": self.time,
            "resistance_n": [float(x) for x in hull_reaction],
            "overlap_ratio": overlap_ratio,
        }

    # ---------------------------------------------------------------------- runs
    def settle(self, steps: int) -> dict[str, float]:
        for _ in range(steps):
            self.step(record=False)
        self.velocities.zero_()
        return self.energy_state()

    def run(
        self,
        *,
        settle_steps: int = 400,
        lead_in_m: float = 0.0,
        spinup_travel_m: float = 0.3,
        measure_travel_m: float = 0.5,
        sample_interval: int = 10,
        sample_hull: int = 0,
    ) -> dict[str, Any]:
        """Settle, spin up and measure the time-averaged hull resistance.

        ``lead_in_m`` advances the convoy before the transient/spin-up phase, so
        that a follower can enter the channel that the leader has already swept.
        Resistance samples are only taken while the monitored hull (index
        ``sample_hull``) is inside the ice field; the sample count is reported so
        a partially-sampled window is visible rather than hidden.
        """
        cfg = self.config
        settle_energy = self.settle(settle_steps)
        self.ledger = {key: 0.0 for key in self.ledger}
        speed = self.hulls[sample_hull].speed_ms
        lead_steps = max(0, int(round(lead_in_m / (speed * self.dt))))
        measure_steps = max(1, int(round(measure_travel_m / (speed * self.dt))))
        sample_interval = max(1, int(sample_interval))
        for _ in range(lead_steps):
            self.step(record=False)
        self.ledger = {key: 0.0 for key in self.ledger}
        self.max_overlap_ratio = 0.0
        e_measure_start = self.energy_state()
        samples: list[list[float]] = []
        ice_speed_sum = 0.0
        ice_speed_max = 0.0
        for step in range(measure_steps):
            out = self.step(record=True)
            apex = self.hull_x[sample_hull]
            in_window = 0.05 <= apex <= cfg.length_m - 0.05
            if step % sample_interval == 0 and in_window:
                samples.append(out["resistance_n"])
                speeds = torch.linalg.norm(self.velocities, dim=1)
                ice_speed_sum += float(speeds.mean())
                ice_speed_max = max(ice_speed_max, float(speeds.max()))
        if not samples:
            samples = [[0.0] * len(self.hulls)]
        e_end = self.energy_state()
        n_samples = len(samples)
        mean_resistance = [
            sum(s[k] for s in samples) / n_samples for k in range(len(self.hulls))
        ]
        std_resistance = []
        for k in range(len(self.hulls)):
            mean = mean_resistance[k]
            std_resistance.append(
                math.sqrt(sum((s[k] - mean) ** 2 for s in samples) / max(1, n_samples - 1))
            )
        delta_energy = sum(e_end[key] - e_measure_start[key] for key in e_end)
        dissipations = {
            "contact": self.ledger["contact_dissipation_j"],
            "hull_slide": self.ledger["hull_slide_dissipation_j"],
            "wall": self.ledger["wall_dissipation_j"],
            "drag": self.ledger["drag_dissipation_j"],
        }
        bow_work = self.ledger["bow_work_j"]
        total_diss = sum(dissipations.values())
        residual = bow_work - (delta_energy + total_diss)
        scale = max(abs(bow_work), 1e-30)
        ledger = {
            "bow_work_j": bow_work,
            "delta_mechanical_energy_j": delta_energy,
            "delta_kinetic_j": e_end["kinetic_j"] - e_measure_start["kinetic_j"],
            "delta_potential_j": (e_end["potential_gravity_j"] + e_end["potential_buoyancy_j"])
            - (e_measure_start["potential_gravity_j"] + e_measure_start["potential_buoyancy_j"]),
            **{f"{k}_dissipation_j": v for k, v in dissipations.items()},
            "total_dissipation_j": total_diss,
            "residual_j": residual,
            "relative_residual": abs(residual) / scale,
        }
        return {
            "concentration": cfg.concentration,
            "thickness_m": cfg.thickness_m,
            "particle_count": len(self.positions),
            "mean_radius_m": float(self.radii.mean()),
            "dt_s": self.dt,
            "steps_measured": min(len(samples) * sample_interval, measure_steps),
            "steps_window": measure_steps,
            "samples": n_samples,
            "travel_measured_m": measure_travel_m,
            "mean_resistance_n": mean_resistance[0],
            "std_resistance_n": std_resistance[0],
            "hull_resistances_n": mean_resistance,
            "hull_resistance_std_n": std_resistance,
            "mean_resistance_per_hull_n": mean_resistance,
            "mean_ice_speed_m_s": ice_speed_sum / max(1, n_samples),
            "max_ice_speed_m_s": ice_speed_max,
            "max_overlap_ratio": self.max_overlap_ratio,
            "clamped_contacts": self.clamped_contacts,
            "energy_ledger": ledger,
            "settle": settle_energy,
            "hull_stats": getattr(self, "last_hull_stats", []),
        }