"""Hertz-Mindlin frictional sphere packing with servo-controlled confining pressure.

This module is the *micromechanical device* behind the granular wave-speed
verification: it builds a three-dimensional packing of elastic frictional
spheres, confines it to a prescribed isotropic pressure with the same
velocity-proportional servo law that drives the true-triaxial protocol
(:func:`tensordem.dem3d_triaxial.servo_velocity`), and exposes the diagnostics
that connect contact parameters to macroscopic response:

* contact law    Hertz normal ``F_n = (4/3) E* sqrt(R*) delta^(3/2)`` plus Mindlin
  tangential stiffness ``k_t = 8 G* sqrt(R* delta)`` with a Coulomb cap
  ``|F_t| <= mu F_n`` (Cundall-Strack slip projection, rotations included);
* pressure       measured from the six platen reactions (what the platens feel),
  plus a per-axis isotropy check;
* coordination number ``Z`` of the load-bearing contact network;
* porosity       ``phi`` from the current box volume;
* static shear   small-strain shear modulus from a tangential platen
  displacement, i.e. independent of any wave source, which is what separates
  "medium rigidity" from "source coupling" in the frictionless control.

Preparation protocol
--------------------
The packing starts on a lattice whose spacing exceeds ``2 R_max`` (no initial
overlap, hence no startup force spike) and is then densified *by the servo*
while a velocity quench drains the kinetic energy released as contacts form.
That is a standard DEM isotropic compaction: the final porosity is an output of
the target pressure, not an input.

Units are SI throughout; no domain constants are baked in.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from ..dem3d_triaxial import servo_velocity

# Wall ordering 0=-x 1=+x 2=-y 3=+y 4=-z 5=+z
_WALL_SIGN = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
_WALL_AXIS = (0, 0, 1, 1, 2, 2)


@dataclass
class GranularWaveConfig:
    """Parameters of a confined frictional-sphere packing (SI units)."""

    # --- packing geometry ---
    nx: int = 3
    ny: int = 3
    nz: int = 64
    radius: float = 5.0e-3           # mean radius, m
    polydispersity: float = 0.12     # radii uniform in R*(1 +- p)
    jitter: float = 0.08             # random centre displacement, in units of R
    initial_spacing: float = 1.001   # lattice spacing in units of 2*R_max
    # --- grain material ---
    density: float = 2500.0
    young_modulus: float = 5.0e9
    poisson_ratio: float = 0.30
    # --- contact rheology ---
    friction: float = 0.30           # interparticle Coulomb coefficient mu
    wall_friction: float = 0.30      # platen friction (shear-wave source coupling)
    normal_damping: float = 0.10     # dashpot ratio of critical damping
    tangential_damping: float = 0.10
    # --- numerics ---
    dt_factor: float = 0.04
    neighbor_skin: float = 0.12
    seed: int = 0
    # --- servo (true-triaxial protocol law) ---
    servo_gain: float = 3.0e-3       # (m/s) per N of force error
    servo_v_max: float = 0.05        # m/s clamp (<< sound speed: quasi-static)
    servo_tol: float = 0.02          # |p - p_target| / p_target settle tolerance
    # --- environment for the current run ---
    pressure: float = 1.0e5          # Pa, isotropic confining pressure target

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            if int(getattr(self, name)) < 3:
                raise ValueError(f"{name} must be >= 3")
        for name in ("radius", "density", "young_modulus"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in [0, 0.5)")
        if not 0.0 <= self.polydispersity < 0.5:
            raise ValueError("polydispersity must be in [0, 0.5)")
        if self.friction < 0 or self.wall_friction < 0:
            raise ValueError("friction coefficients must be nonnegative")
        if self.pressure <= 0:
            raise ValueError("pressure must be positive")
        if not 1.0 <= self.initial_spacing <= 1.3:
            raise ValueError("initial_spacing must be in [1.0, 1.3] (no initial overlap)")

    # -- derived elastic constants ----------------------------------------
    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def e_star(self) -> float:
        """Reduced modulus of two equal spheres, ``E* = E / (2 (1 - nu^2))``."""
        nu = self.poisson_ratio
        return self.young_modulus / (2.0 * (1.0 - nu * nu))

    @property
    def e_star_wall(self) -> float:
        """Reduced modulus against a rigid platen, ``E* = E / (1 - nu^2)``."""
        nu = self.poisson_ratio
        return self.young_modulus / (1.0 - nu * nu)

    @property
    def g_star(self) -> float:
        """Reduced shear modulus of two equal spheres, ``G* = G/(2(2-nu))``."""
        return self.shear_modulus / (2.0 * (2.0 - self.poisson_ratio))

    @property
    def g_star_wall(self) -> float:
        return self.shear_modulus / (2.0 - self.poisson_ratio)

    @property
    def radius_min(self) -> float:
        return self.radius * (1.0 - self.polydispersity)

    @property
    def radius_max(self) -> float:
        return self.radius * (1.0 + self.polydispersity)

    @property
    def recommended_dt(self) -> float:
        """Conservative step from the stiffest Hertz/Mindlin contact.

        The reference overlap is 5% of the mean radius, comfortably above the
        ~1-2% overlaps reached at the highest pressure of the verification scan;
        :attr:`GranularPacking.max_overlap_seen` records the realised maximum so a
        run can be rejected a posteriori if the servo ever overran.
        """
        r_star = 0.5 * self.radius_max
        delta_ref = 0.05 * self.radius
        sq = math.sqrt(r_star * delta_ref)
        k_n = 2.0 * self.e_star * sq
        k_t = 8.0 * self.g_star * sq
        k_max = max(k_n, k_t)
        m_min = self.density * (4.0 / 3.0) * math.pi * self.radius_min**3
        return self.dt_factor * 2.0 * math.pi * math.sqrt(m_min / k_max)


def build_packing(cfg: GranularWaveConfig) -> dict[str, Any]:
    """Initial *non-overlapping* lattice packing (compaction is the servo's job)."""
    gen = torch.Generator().manual_seed(int(cfg.seed))
    axes = [torch.arange(v, dtype=torch.float64) for v in (cfg.nx, cfg.ny, cfg.nz)]
    zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    grid = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    spacing = cfg.initial_spacing * 2.0 * cfg.radius_max
    n = len(grid)
    u = torch.rand(n, generator=gen, dtype=torch.float64)
    radius = cfg.radius * (1.0 + cfg.polydispersity * (2.0 * u - 1.0))
    pos = grid * spacing + (torch.rand(n, 3, generator=gen, dtype=torch.float64) - 0.5) \
        * (2.0 * cfg.jitter * cfg.radius)
    pos = pos - 0.5 * (pos.max(dim=0).values + pos.min(dim=0).values)
    extent = pos.max(dim=0).values - pos.min(dim=0).values
    # platens sit one *largest* radius beyond the outermost centres: with the
    # mean radius the biggest grains would already be 0.15 R inside the platen,
    # i.e. ~10 kN of spurious initial force per grain
    m = cfg.radius_max
    wall_pos = torch.tensor(
        [-extent[0] / 2 - m, extent[0] / 2 + m,
         -extent[1] / 2 - m, extent[1] / 2 + m,
         -extent[2] / 2 - m, extent[2] / 2 + m],
        dtype=torch.float64)
    mass = cfg.density * (4.0 / 3.0) * math.pi * radius.pow(3)
    inertia = 0.4 * mass * radius.pow(2)
    return {
        "positions": pos,
        "radii": radius,
        "mass": mass,
        "inertia": inertia,
        "wall_pos": wall_pos,
    }


class GranularPacking:
    """Frictional Hertz-Mindlin sphere packing confined by six servo platens."""

    def __init__(self, cfg: GranularWaveConfig | None = None, *,
                 init: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or GranularWaveConfig()
        self.dtype = torch.float64
        init = init or build_packing(self.cfg)
        self.pos = init["positions"].clone()
        self.radius = init["radii"].clone()
        self.mass = init["mass"].clone()
        self.inertia = init["inertia"].clone()
        self.wall_pos = init["wall_pos"].clone()
        self.n = len(self.pos)
        self.dt = self.cfg.recommended_dt
        self.vel = torch.zeros_like(self.pos)
        self.omega = torch.zeros_like(self.pos)
        self.wall_vel = torch.zeros((6, 3), dtype=self.dtype)
        self.wall_vnormal = torch.zeros(6, dtype=self.dtype)
        self.wall_normal_force = torch.zeros(6, dtype=self.dtype)
        self.wall_tangent_force = torch.zeros((6, 3), dtype=self.dtype)
        self.wall_tan = torch.zeros((self.n, 6, 3), dtype=self.dtype)
        self.wall_tangential_drive = torch.zeros((6, 3), dtype=self.dtype)
        self._axis_idx = torch.tensor(_WALL_AXIS, dtype=torch.long)
        self._signs = torch.tensor(_WALL_SIGN, dtype=self.dtype)
        self._inward = -self._signs            # +1 for the -side platen, -1 for the +side
        self._normals = self._wall_normals()
        self.time = 0.0
        self.step_count = 0
        self.pairs = torch.zeros((0, 2), dtype=torch.long)
        self.pair_tan = torch.zeros((0, 3), dtype=self.dtype)
        self.pair_force_normal = torch.zeros(0, dtype=self.dtype)
        self.sponge: dict[str, float] | None = None
        self.quench = 1.0
        self.soft_scale = 1.0          # multiplier on contact stiffness (soft start)
        self.servo_vmax = None         # optional override of cfg.servo_v_max
        self.damping_scale = 1.0       # multiplier on the dashpot ratios
        self.max_overlap_seen = 0.0
        self.rebuild_pairs()

    # -- geometry helpers --------------------------------------------------
    def _wall_normals(self) -> torch.Tensor:
        """Inward unit normals of the six platens, shape (6, 3)."""
        nrm = torch.zeros((6, 3), dtype=self.dtype)
        for k in range(6):
            nrm[k, _WALL_AXIS[k]] = -_WALL_SIGN[k]
        return nrm

    def box_lengths(self) -> torch.Tensor:
        w = self.wall_pos
        return torch.stack((w[1] - w[0], w[3] - w[2], w[5] - w[4]))

    def free_lengths(self) -> torch.Tensor:
        """Platen gap available to particle centres (used for the face areas)."""
        return (self.box_lengths() - 2.0 * self.radius.mean()).clamp_min(1e-9)

    def face_areas(self) -> torch.Tensor:
        L = self.free_lengths()
        return torch.stack((L[1] * L[2], L[0] * L[2], L[0] * L[1]))

    def volume(self) -> float:
        """Volume enclosed by the six platens (the physical specimen volume)."""
        return float(self.box_lengths().prod())

    def porosity(self) -> float:
        solid = float((self.mass / self.cfg.density).sum())
        return 1.0 - solid / self.volume()

    # -- neighbour bookkeeping --------------------------------------------
    def rebuild_pairs(self) -> None:
        x = self.pos
        d = torch.cdist(x, x)
        rs = self.radius[:, None] + self.radius[None, :]
        keep = d <= rs * (1.0 + self.cfg.neighbor_skin)
        keep.fill_diagonal_(False)
        keep = torch.triu(keep, diagonal=1)
        self.pairs = keep.nonzero()
        self.pair_tan = torch.zeros((len(self.pairs), 3), dtype=self.dtype)
        self.pair_force_normal = torch.zeros(len(self.pairs), dtype=self.dtype)

    # -- forces ------------------------------------------------------------
    def _pair_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        force = torch.zeros_like(self.pos)
        torque = torch.zeros_like(self.pos)
        self.pair_force_normal.zero_()
        if len(self.pairs) == 0:
            return force, torque
        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        d_vec = self.pos[j] - self.pos[i]
        dist = torch.linalg.vector_norm(d_vec, dim=1)
        rs = self.radius[i] + self.radius[j]
        delta = rs - dist
        active = delta > 0.0
        if not bool(active.any()):
            return force, torque
        idx = active.nonzero().flatten()
        ii, jj = i[idx], j[idx]
        n_ij = d_vec[idx] / dist[idx].clamp_min(1e-30).unsqueeze(1)
        delta_c = delta[idx].clamp_min(0.0)
        self.max_overlap_seen = max(self.max_overlap_seen,
                                    float(delta_c.max()) / cfg.radius)

        r_star = (self.radius[ii] * self.radius[jj]) / (self.radius[ii] + self.radius[jj])
        sq = torch.sqrt(r_star * delta_c)
        e_star = cfg.e_star * self.soft_scale
        g_star = cfg.g_star * self.soft_scale
        f_n = (4.0 / 3.0) * e_star * sq * delta_c
        k_n = 2.0 * e_star * sq
        k_t = 8.0 * g_star * sq

        r_i = self.radius[ii].unsqueeze(1)
        r_j = self.radius[jj].unsqueeze(1)
        vc_i = self.vel[ii] + torch.cross(self.omega[ii], r_i * n_ij, dim=1)
        vc_j = self.vel[jj] - torch.cross(self.omega[jj], r_j * n_ij, dim=1)
        v_rel = vc_j - vc_i
        v_n = (v_rel * n_ij).sum(dim=1)
        v_t = v_rel - v_n.unsqueeze(1) * n_ij

        m_eff = (self.mass[ii] * self.mass[jj]) / (self.mass[ii] + self.mass[jj])
        c_n = 2.0 * cfg.normal_damping * self.damping_scale * torch.sqrt(k_n * m_eff)
        c_t = 2.0 * cfg.tangential_damping * self.damping_scale * torch.sqrt(k_t * m_eff)
        fn_total = (f_n - c_n * v_n).clamp_min(0.0)

        tan = self.pair_tan[idx] + v_t * self.dt
        f_t = -k_t.unsqueeze(1) * tan - c_t.unsqueeze(1) * v_t
        ft_mag = torch.linalg.vector_norm(f_t, dim=1)
        cap = cfg.friction * fn_total
        slipping = ft_mag > cap
        scale = torch.where(slipping, cap / ft_mag.clamp_min(1e-30),
                            torch.ones_like(ft_mag))
        f_t = f_t * scale.unsqueeze(1)
        tan = torch.where(slipping.unsqueeze(1),
                          -f_t / k_t.unsqueeze(1).clamp_min(1e-30), tan)
        # only overlapping pairs own a tangential history: a separated pair must
        # not accumulate relative displacement that would explode on re-contact
        self.pair_tan = torch.zeros_like(self.pair_tan)
        self.pair_tan[idx] = tan
        self.pair_force_normal[idx] = fn_total

        f_on_j = fn_total.unsqueeze(1) * n_ij + f_t
        force.index_add_(0, jj, f_on_j)
        force.index_add_(0, ii, -f_on_j)
        torque.index_add_(0, jj, torch.cross(-r_j * n_ij, f_t, dim=1))
        torque.index_add_(0, ii, torch.cross(r_i * n_ij, f_t, dim=1))
        return force, torque

    def _wall_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched contact forces for the six platens (vectorised over N x 6)."""
        cfg = self.cfg
        r = self.radius.unsqueeze(1)                               # (N,1)
        coord = self.pos[:, self._axis_idx]                        # (N,6)
        overlap = r + self._signs.unsqueeze(0) * (coord - self.wall_pos.unsqueeze(0))
        hit = overlap > 0.0
        delta = overlap.clamp_min(0.0)
        self.max_overlap_seen = max(self.max_overlap_seen,
                                    float(delta.max()) / cfg.radius)
        sq = torch.sqrt(r * delta)
        e_star_w = cfg.e_star_wall * self.soft_scale
        g_star_w = cfg.g_star_wall * self.soft_scale
        f_n = (4.0 / 3.0) * e_star_w * sq * delta
        k_n = 2.0 * e_star_w * sq
        k_t = 8.0 * g_star_w * sq
        normals = self._normals.unsqueeze(0)                       # (1,6,3)
        r_vec = -r.unsqueeze(2) * normals                          # (N,6,3)
        v_c = self.vel.unsqueeze(1) + torch.cross(
            self.omega.unsqueeze(1).expand(-1, 6, -1), r_vec, dim=2)
        v_slip = v_c - self.wall_vel.unsqueeze(0)
        v_n = (v_slip * normals).sum(dim=2)                        # >0 into the wall
        v_t = v_slip - v_n.unsqueeze(2) * normals
        m_eff = self.mass.unsqueeze(1)
        c_n = 2.0 * cfg.normal_damping * self.damping_scale * torch.sqrt(k_n * m_eff)
        c_t = 2.0 * cfg.tangential_damping * self.damping_scale * torch.sqrt(k_t * m_eff)
        fn_total = (f_n - c_n * v_n).clamp_min(0.0) * hit.to(self.dtype)
        tan = self.wall_tan + v_t * self.dt
        f_t = -k_t.unsqueeze(2) * tan - c_t.unsqueeze(2) * v_t
        ft_mag = torch.linalg.vector_norm(f_t, dim=2)
        cap = cfg.wall_friction * fn_total
        slipping = ft_mag > cap
        scale = torch.where(slipping, cap / ft_mag.clamp_min(1e-30),
                            torch.ones_like(ft_mag))
        f_t = f_t * scale.unsqueeze(2)
        tan_keep = torch.where(slipping.unsqueeze(2),
                               -f_t / k_t.unsqueeze(2).clamp_min(1e-30), tan)
        # a platen that is not touching a grain owns no tangential history
        self.wall_tan = torch.where(hit.unsqueeze(2), tan_keep,
                                    torch.zeros_like(tan_keep))
        f_on_p = fn_total.unsqueeze(2) * normals + f_t
        force = f_on_p.sum(dim=1)
        torque = torch.cross(r_vec, f_t, dim=2).sum(dim=1)
        self.wall_normal_force = fn_total.sum(dim=0)
        self.wall_tangent_force = f_t.sum(dim=0)
        return force, torque

    def forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        f_p, t_p = self._pair_forces()
        f_w, t_w = self._wall_forces()
        return f_p + f_w, t_p + t_w

    # -- integration -------------------------------------------------------
    def step(self) -> None:
        force, torque = self.forces()
        self.vel = self.vel + (force / self.mass.unsqueeze(1)) * self.dt
        self.omega = self.omega + (torque / self.inertia.unsqueeze(1)) * self.dt
        if self.quench < 1.0:
            self.vel = self.vel * self.quench
            self.omega = self.omega * self.quench
        if self.sponge is not None:
            self._apply_sponge()
        self.pos = self.pos + self.vel * self.dt
        # ``wall_vnormal`` is inward-positive, so the scalar coordinate moves by
        # ``-sign * v`` (the -x platen advances towards +x, the +x platen towards -x)
        self.wall_pos = self.wall_pos + (self._inward * self.wall_vnormal) * self.dt
        self.wall_vel = (self._normals * self.wall_vnormal.unsqueeze(1)
                         + self.wall_tangential_drive)
        self.time += self.dt
        self.step_count += 1
        if not bool(torch.isfinite(self.vel).all() and torch.isfinite(self.pos).all()):
            raise RuntimeError("granular packing diverged (non-finite state)")

    def _apply_sponge(self) -> None:
        assert self.sponge is not None
        z = self.pos[:, 2]
        z0 = self.sponge["z0"]
        z_top = self.sponge["z_top"]
        ramp = ((z - z0) / max(z_top - z0, 1e-12)).clamp(0.0, 1.0) ** 2
        decay = torch.exp(-self.sponge["gamma"] * ramp * self.dt).unsqueeze(1)
        self.vel = self.vel * decay
        self.omega = self.omega * decay

    # -- servo confining pressure (true-triaxial protocol law) -------------
    def pressures(self) -> torch.Tensor:
        """Confining pressure per axis from the platen reactions, Pa."""
        area = self.face_areas()
        f = self.wall_normal_force
        return torch.stack(((f[0] + f[1]) / (2.0 * area[0]),
                            (f[2] + f[3]) / (2.0 * area[1]),
                            (f[4] + f[5]) / (2.0 * area[2])))

    def servo(self, target_pressure: float | None = None) -> torch.Tensor:
        """Set platen normal velocities from the force error (inward positive)."""
        cfg = self.cfg
        target = cfg.pressure if target_pressure is None else float(target_pressure)
        area = self.face_areas()
        p = self.pressures()
        gains = cfg.servo_gain * area
        for a in range(3):
            v = servo_velocity(float(target - p[a]), float(gains[a]),
                               cfg.servo_v_max if self.servo_vmax is None else self.servo_vmax)
            self.wall_vnormal[2 * a] = v
            self.wall_vnormal[2 * a + 1] = v
        return p

    def compact(self, target_pressure: float | None = None, *,
                strain_increment: float = 2.5e-3, relax_steps: int = 80,
                max_increments: int = 400, quench: float = 0.97) -> dict[str, Any]:
        """Strain-controlled isotropic compaction until the target pressure.

        Positions and platens are scaled affinely by ``1 - strain_increment`` per
        increment and the packing is relaxed in between.  Because every step is
        small the overlaps grow smoothly, so no startup impact spike can occur
        (an uncontrolled servo has no force to react to while the grains are
        still apart and would simply run into the packing at its speed limit).
        """
        target = self.cfg.pressure if target_pressure is None else float(target_pressure)
        self.quench = quench
        increments = 0
        strain = 0.0
        for k in range(max_increments):
            p = float(self.pressures().mean())
            if p >= target:
                break
            self.pos = self.pos * (1.0 - strain_increment)
            self.wall_pos = self.wall_pos * (1.0 - strain_increment)
            strain += strain_increment
            increments = k + 1
            self.rebuild_pairs()
            for n in range(relax_steps):
                self.step()
                if (n + 1) % 20 == 0:
                    self.rebuild_pairs()
        self.quench = 1.0
        self.vel.zero_()
        self.omega.zero_()
        self.rebuild_pairs()
        self.max_overlap_seen = 0.0
        return {"increments": increments, "applied_strain": strain,
                "pressure_Pa": float(self.pressures().mean()),
                "target_pressure_Pa": target,
                "overlap_fraction": self.max_overlap_seen}

    def set_pressure(self, target: float, *, fraction: float = 0.9,
                     strain_increment: float = 2.5e-3, relax_steps: int = 80,
                     max_increments: int = 400, tol: float | None = None,
                     settle_steps: int = 60) -> dict[str, Any]:
        """Bring the packing to ``target`` pressure (ascending or descending).

        Loading is deliberately split: a strain-controlled compaction up to
        ``fraction * target`` (monotone, impact-free) and then the servo for the
        last few percent.  Approaching the target from *below* matters: a packing
        driven past its target and then unloaded can lose its load-bearing
        network entirely and fall back to a much looser, unjammed state.
        """
        target = float(target)
        p_now = float(self.pressures().mean())
        comp: dict[str, Any] | None = None
        if p_now < fraction * target:
            comp = self.compact(fraction * target, strain_increment=strain_increment,
                                relax_steps=relax_steps,
                                max_increments=max_increments)
        # unload quickly (nothing can be damaged), load gently
        self.servo_vmax = 0.05 if p_now > target else 0.01
        try:
            eq = self.equilibrate(target, max_steps=6000, tol=tol,
                                  settle_steps=settle_steps)
        finally:
            self.servo_vmax = None
        eq["compaction"] = comp
        eq["pressure_start_Pa"] = p_now
        return eq

    def mean_contact_overlap(self) -> float:
        """Mean overlap of the load-bearing contacts, in units of the mean radius."""
        if len(self.pairs) == 0:
            return 0.0
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d = torch.linalg.vector_norm(self.pos[j] - self.pos[i], dim=1)
        delta = (self.radius[i] + self.radius[j]) - d
        act = delta > 0.0
        if not bool(act.any()):
            return 0.0
        return float(delta[act].mean()) / float(self.radius.mean())

    def solid_fraction(self) -> float:
        return 1.0 - self.porosity()

    def compact_to_solid_fraction(self, target: float = 0.60, *,
                                  strain_increment: float = 1.0e-2,
                                  relax_steps: int = 200, quench: float = 0.97,
                                  max_increments: int = 60) -> dict[str, Any]:
        """Affine compaction to a prescribed solid fraction (density control).

        On a near-lattice packing the coordination number jumps from 0 to ~6
        within a very narrow density window, so ``Z`` is a poor stopping rule;
        the solid fraction is monotone and unambiguous.  Compacting a couple of
        percent *past* the expected jamming density and then letting the servo
        relieve the excess is what makes the prepared state independent of how
        the search started.
        """
        self.quench = quench
        increments = 0
        strain = 0.0
        sf = self.solid_fraction()
        while increments < max_increments and sf < target:
            self.pos = self.pos * (1.0 - strain_increment)
            self.wall_pos = self.wall_pos * (1.0 - strain_increment)
            strain += strain_increment
            increments += 1
            self.rebuild_pairs()
            for n in range(relax_steps):
                self.step()
                if (n + 1) % 20 == 0:
                    self.rebuild_pairs()
            sf = self.solid_fraction()
        self.quench = 1.0
        self.vel.zero_()
        self.omega.zero_()
        self.rebuild_pairs()
        self.max_overlap_seen = 0.0
        return {"increments": increments, "applied_strain": strain,
                "solid_fraction": sf, "target_solid_fraction": target,
                "mean_overlap": self.mean_contact_overlap(),
                "pressure_Pa": float(self.pressures().mean()),
                "porosity": self.porosity()}

    def compact_to_coordination(self, z_target: float = 6.0, *,
                                strain_increment: float = 1.0e-3,
                                relax_steps: int = 100, max_increments: int = 400,
                                quench: float = 0.97) -> dict[str, Any]:
        """Affine compaction until the load-bearing network reaches ``z_target``.

        The coordination number, not the pressure, is the right stopping rule for
        the *soft* phase: the pressure can be carried by a handful of contacts
        while most grains are still apart, whereas ``Z`` is zero before jamming
        and grows once a load-bearing network exists.
        """
        self.quench = quench
        increments = 0
        strain = 0.0
        z = self.contact_network()["coordination_number"]
        while increments < max_increments and z < z_target:
            self.pos = self.pos * (1.0 - strain_increment)
            self.wall_pos = self.wall_pos * (1.0 - strain_increment)
            strain += strain_increment
            increments += 1
            self.rebuild_pairs()
            for n in range(relax_steps):
                self.step()
                if (n + 1) % 20 == 0:
                    self.rebuild_pairs()
            self.rebuild_pairs()
            z = self.contact_network()["coordination_number"]
        self.quench = 1.0
        self.vel.zero_()
        self.omega.zero_()
        self.max_overlap_seen = 0.0
        return {"increments": increments, "applied_strain": strain,
                "coordination_number": z, "z_target": z_target,
                "mean_overlap": self.mean_contact_overlap(),
                "pressure_Pa": float(self.pressures().mean()),
                "porosity": self.porosity()}

    def compact_to_overlap(self, target_overlap: float, *, coarse: float = 1.0e-3,
                           fine: float = 3.0e-4, relax_steps: int = 100,
                           max_increments: int = 800, quench: float = 0.97) -> dict[str, Any]:
        """Affine strain-controlled compaction until the mean overlap is reached.

        Stopping on the *overlap* rather than on the pressure is what makes the
        preparation uniform: with soft contacts the pressure at a given overlap is
        small, so the compaction cannot stop early on a light force chain, and the
        target overlap is exactly the one that the real stiffness needs to produce
        the target pressure (Hertz: ``p ~ E (delta/R)^{3/2}``).
        """
        self.quench = quench
        increments = 0
        strain = 0.0
        while increments < max_increments:
            ov = self.mean_contact_overlap()
            if ov >= target_overlap:
                break
            inc = coarse if ov < 0.5 * target_overlap else fine
            self.pos = self.pos * (1.0 - inc)
            self.wall_pos = self.wall_pos * (1.0 - inc)
            strain += inc
            increments += 1
            self.rebuild_pairs()
            for n in range(relax_steps):
                self.step()
                if (n + 1) % 20 == 0:
                    self.rebuild_pairs()
            self.rebuild_pairs()
        self.quench = 1.0
        self.vel.zero_()
        self.omega.zero_()
        self.max_overlap_seen = 0.0
        return {"increments": increments, "applied_strain": strain,
                "mean_overlap": self.mean_contact_overlap(),
                "target_overlap": target_overlap,
                "pressure_Pa": float(self.pressures().mean()),
                "porosity": self.porosity(),
                "coordination_number": self.contact_network()["coordination_number"]}

    def _probe_scale(self, s_rel: float, *, relax_steps: int, tail: int = 60,
                     quench: float = 0.98) -> float:
        """Apply a relative affine scale, relax, and return the tail-mean pressure."""
        self.pos = self.pos * s_rel
        self.wall_pos = self.wall_pos * s_rel
        self.vel.zero_()
        self.omega.zero_()
        self.rebuild_pairs()
        self.quench = quench
        samples: list[float] = []
        for step in range(relax_steps):
            self.step()
            if (step + 1) % 20 == 0:
                self.rebuild_pairs()
            if step >= relax_steps - tail:
                samples.append(float(self.pressures().mean()))
        self.quench = 1.0
        self.rebuild_pairs()
        return sum(samples) / max(len(samples), 1)

    def balance_pressure(self, target_pressure: float, *, relax_steps: int = 250,
                         max_probes: int = 14, tol: float = 0.02,
                         step0: float = 0.01) -> dict[str, Any]:
        """Set the confining pressure by *density* control (bracketed bisection).

        The six-platen servo is the physical actuator, but a proportional servo on
        a near-marginal packing overshoots: unloading by a few tenths of a percent
        of strain can drop the pressure by a decade and unjam the network.  The
        affine bisection below is monotone in the density, so it cannot overshoot;
        each probe relaxes the packing before the pressure is read (tail mean).
        """
        target = float(target_pressure)
        p_cur = self._probe_scale(1.0, relax_steps=relax_steps)
        history = [{"scale": 1.0, "cumulative_scale": 1.0, "pressure_Pa": p_cur}]
        cum = 1.0
        # bracket
        if p_cur < target:
            s_lo, s_hi = 1.0 - step0, 1.0            # smaller scale -> higher pressure
            p_lo = self._probe_scale(s_lo, relax_steps=relax_steps)
            cum *= s_lo
            history.append({"scale": s_lo, "cumulative_scale": cum, "pressure_Pa": p_lo})
            widen = step0
            while p_lo < target and len(history) < max_probes:
                widen *= 2.0
                s_lo, s_hi = 1.0 - widen, s_lo
                p_hi, p_lo = p_lo, self._probe_scale(1.0 - widen, relax_steps=relax_steps)
                cum *= (1.0 - widen)
                history.append({"scale": 1.0 - widen, "cumulative_scale": cum,
                                "pressure_Pa": p_lo})
            if p_lo < target:      # could not reach: accept the densest state probed
                return {"converged": False, "target_pressure_Pa": target,
                        "pressure_Pa": p_lo, "history": history,
                        "relative_error": (p_lo - target) / target}
        else:
            s_hi, s_lo = 1.0 + step0, 1.0            # larger scale -> lower pressure
            p_hi = self._probe_scale(s_hi, relax_steps=relax_steps)
            cum *= s_hi
            history.append({"scale": s_hi, "cumulative_scale": cum, "pressure_Pa": p_hi})
            widen = step0
            while p_hi > target and len(history) < max_probes:
                widen *= 2.0
                s_lo, s_hi = s_hi, 1.0 + widen
                p_lo, p_hi = p_hi, self._probe_scale(1.0 + widen, relax_steps=relax_steps)
                cum *= (1.0 + widen)
                history.append({"scale": 1.0 + widen, "cumulative_scale": cum,
                                "pressure_Pa": p_hi})
            if p_hi > target:
                return {"converged": False, "target_pressure_Pa": target,
                        "pressure_Pa": p_hi, "history": history,
                        "relative_error": (p_hi - target) / target}
            p_lo, p_hi = p_hi, p_lo
            s_lo, s_hi = s_hi, s_lo
        # bisection on the relative scale
        for _ in range(max_probes):
            if abs(p_cur - target) <= tol * target:
                break
            s_mid = 0.5 * (s_lo + s_hi)
            p_mid = self._probe_scale(s_mid, relax_steps=relax_steps)
            cum *= s_mid
            p_cur = p_mid
            history.append({"scale": s_mid, "cumulative_scale": cum,
                            "pressure_Pa": p_mid})
            if p_mid < target:
                s_hi = s_mid
                p_lo = p_mid
            else:
                s_lo = s_mid
                p_hi = p_mid
        err = abs(p_cur - target) / target
        return {"converged": bool(err <= max(tol * 3.0, 0.05)),
                "target_pressure_Pa": target, "pressure_Pa": p_cur,
                "relative_error": err, "probes": len(history),
                "cumulative_scale": cum, "history": history[-6:]}

    def set_solid_fraction(self, target: float, *, relax_steps: int = 1200,
                           max_substep_strain: float = 2.0e-3,
                           quench: float = 0.98,
                           damping_scale: float = 2.0) -> dict[str, Any]:
        """Constant-volume step to a prescribed solid fraction, then relax.

        This is the oedometric alternative to the servo: a *fixed* box cannot
        overshoot, so a near-marginal packing stays where it is put, and the
        confining pressure becomes a measured output (``p`` and the overlap are
        both functions of the solid fraction, which is all the ``v ~ p^(1/6)``
        test needs).  The affine move is split into sub-steps of at most
        ``max_substep_strain`` so that the contact forces never spike.
        """
        sf0 = self.solid_fraction()
        lam_total = (sf0 / target) ** (1.0 / 3.0)
        n_sub = max(1, int(math.ceil(abs(1.0 - lam_total) / max_substep_strain)))
        self.quench = quench
        self.damping_scale = damping_scale
        for k in range(n_sub):
            lam = (lam_total - 1.0) / n_sub
            self.pos = self.pos * (1.0 + lam)
            self.wall_pos = self.wall_pos * (1.0 + lam)
            self.rebuild_pairs()
            for n in range(max(1, relax_steps // n_sub)):
                self.step()
                if (n + 1) % 20 == 0:
                    self.rebuild_pairs()
        self.quench = 1.0
        self.damping_scale = 1.0
        self.vel.zero_()
        self.omega.zero_()
        self.rebuild_pairs()
        self.max_overlap_seen = 0.0
        return {"solid_fraction": self.solid_fraction(),
                "target_solid_fraction": target,
                "substeps": n_sub,
                "pressure_Pa": float(self.pressures().mean()),
                "porosity": self.porosity(),
                "coordination_number": self.contact_network()["coordination_number"],
                "mean_contact_overlap": self.mean_contact_overlap()}

    def prepare(self, *, target_pressure: float | None = None,
                solid_fraction: float = 0.525, soft_ratio: float = 50.0,
                strain_increment: float = 1.0e-2, relax_steps: int = 150,
                stiffen_steps: int = 800, final_steps: int = 500) -> dict[str, Any]:
        """Jam a loose lattice with soft contacts, then stiffen to the real modulus.

        A stiff Hertz contact on a loose lattice is a bad starting point: the first
        contacts to appear carry the whole load, so the network is a light force
        chain rather than a jammed packing.  The contacts are therefore jammed with
        stiffness ``E/soft_ratio`` (and a step ``sqrt(soft_ratio)/1.75`` times
        larger, still stable), which lets the network rearrange while it forms, and
        the stiffness is then ramped back to the real value at fixed geometry.

        ``target_pressure`` is accepted for interface compatibility with the servo
        protocol but is *not* used to drive the preparation: the pressure of this
        engine is an output (constant-volume states), see
        :meth:`set_solid_fraction`.
        """
        dt0 = self.dt
        s_soft = 1.0 / float(soft_ratio)
        self.dt = dt0 * math.sqrt(soft_ratio) / 1.75
        self.soft_scale = s_soft
        self.damping_scale = 4.0
        self.servo_vmax = 0.0
        self.wall_vnormal.zero_()
        comp = self.compact_to_solid_fraction(solid_fraction,
                                              strain_increment=strain_increment,
                                              relax_steps=relax_steps)
        self.dt = dt0
        self.damping_scale = 1.5
        self.quench = 0.99
        for step in range(stiffen_steps):
            frac = (step + 1) / max(stiffen_steps, 1)
            self.soft_scale = s_soft + (1.0 - s_soft) * frac
            self.step()
            if (step + 1) % 25 == 0:
                self.rebuild_pairs()
        self.soft_scale = 1.0
        self.damping_scale = 1.0
        self.quench = 0.995
        for step in range(final_steps):
            self.step()
            if (step + 1) % 25 == 0:
                self.rebuild_pairs()
        self.quench = 1.0
        self.vel.zero_()
        self.omega.zero_()
        self.rebuild_pairs()
        self.max_overlap_seen = 0.0
        eq = self.equilibrate(max_steps=1)   # populate the pressure/force diagnostics
        eq["compaction"] = comp
        return eq

    def equilibrate(self, target_pressure: float | None = None, *,
                    max_steps: int = 2500, tol: float | None = None,
                    settle_steps: int = 40, rebuild_every: int = 25) -> dict[str, Any]:
        """Servo the six platens until the pressure settles within ``tol``."""
        target = self.cfg.pressure if target_pressure is None else float(target_pressure)
        tol = self.cfg.servo_tol if tol is None else tol
        settled = 0
        history: list[dict[str, float]] = []
        for step in range(max_steps):
            self.servo(target)
            self.step()
            if (step + 1) % rebuild_every == 0:
                self.rebuild_pairs()
            p = self.pressures()
            rel = float((p - target).abs().max() / target)
            history.append({"step": self.step_count, "pressure": float(p.mean()),
                            "rel_error": rel})
            settled = settled + 1 if rel <= tol else 0
            if settled >= settle_steps:
                break
        p = self.pressures().clone()
        self.wall_vnormal.zero_()
        self.wall_tangential_drive.zero_()
        return {
            "target_pressure_Pa": target,
            "measured_pressure_Pa": [float(v) for v in p],
            "mean_pressure_Pa": float(p.mean()),
            "relative_error": float((p - target).abs().max() / target),
            "converged": bool((p - target).abs().max() / target <= tol),
            "steps_used": int(history[-1]["step"] if history else self.step_count),
            "tolerance": tol,
            "history": history[-8:],
        }

    # -- diagnostics -------------------------------------------------------
    def contact_network(self, *, threshold_rel: float = 1e-4) -> dict[str, Any]:
        """Coordination number of the contacting network (geometry based).

        Computed from the *current gaps*, not from the last force evaluation, so
        it is valid immediately after a neighbour-list rebuild.  A pair counts as
        a contact when it overlaps by more than ``threshold_rel`` radii.
        """
        if len(self.pairs) == 0:
            return {"coordination_number": 0.0, "n_contacts": 0,
                    "mean_contact_overlap": 0.0, "mean_normal_force_N": 0.0,
                    "coordination_number_force": 0.0}
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d = torch.linalg.vector_norm(self.pos[j] - self.pos[i], dim=1)
        delta = (self.radius[i] + self.radius[j]) - d
        r_mean = float(self.radius.mean())
        keep = delta > threshold_rel * r_mean
        n_c = int(keep.sum())
        f = self.pair_force_normal
        z_force = 0.0
        f_mean = 0.0
        if len(f) > 0 and float(f.max()) > 0.0:
            f_mean = float(f[f > 0].mean())
            z_force = 2.0 * int((f > 1e-2 * f_mean).sum()) / self.n
        return {
            "coordination_number": float(2.0 * n_c / self.n),
            "n_contacts": n_c,
            "mean_contact_overlap": float(delta[keep].mean()) / r_mean if n_c else 0.0,
            "mean_normal_force_N": f_mean,
            "coordination_number_force": float(z_force),
        }

    def kinetic_energy(self) -> float:
        return float(0.5 * (self.mass.unsqueeze(1) * self.vel.square()).sum()
                     + 0.5 * (self.inertia.unsqueeze(1) * self.omega.square()).sum())

    def bulk_density(self) -> float:
        return float(self.mass.sum()) / self.volume()

    def state(self) -> dict[str, Any]:
        p = self.pressures()
        net = self.contact_network()
        return {
            "n_particles": self.n,
            "radius_mean_m": float(self.radius.mean()),
            "porosity": self.porosity(),
            "bulk_density_kg_m3": self.bulk_density(),
            "pressure_per_axis_Pa": [float(v) for v in p],
            "pressure_mean_Pa": float(p.mean()),
            "pressure_anisotropy": float((p.max() - p.min()) / p.mean()),
            "coordination_number": net["coordination_number"],
            "n_load_contacts": net["n_contacts"],
            "mean_normal_force_N": net["mean_normal_force_N"],
            "box_lengths_m": [float(v) for v in self.box_lengths()],
            "kinetic_energy_J": self.kinetic_energy(),
            "time_s": self.time,
            "steps": self.step_count,
            "dt_s": self.dt,
            "max_overlap_fraction": self.max_overlap_seen,
        }

    # -- static small-strain shear modulus ---------------------------------
    def static_shear_modulus(self, *, strain: float = 2.0e-5, relax_steps: int = 400,
                             rebuild_every: int = 25) -> dict[str, Any]:
        """Small-strain shear modulus from a tangential platen displacement.

        The top and bottom platens are displaced by ``+-gamma L_z / 2`` in x, the
        packing relaxes at fixed platen positions, and ``G = tau_xy / gamma`` is
        taken from the tangential platen reaction.  Independent of any wave
        source, so it separates medium rigidity from source coupling.
        """
        self.vel.zero_()
        self.omega.zero_()
        self.wall_vnormal.zero_()
        top, bot = 5, 4
        z0_top = float(self.wall_pos[top])
        z0_bot = float(self.wall_pos[bot])
        disp = 0.5 * strain * (z0_top - z0_bot)
        self.wall_pos[top] = z0_top + disp
        self.wall_pos[bot] = z0_bot - disp
        self.rebuild_pairs()
        for step in range(relax_steps):
            self.step()
            if (step + 1) % rebuild_every == 0:
                self.rebuild_pairs()
        fx_top = float(self.wall_tangent_force[top, 0])
        fx_bot = float(self.wall_tangent_force[bot, 0])
        f_shear = 0.5 * (abs(fx_top) + abs(fx_bot))
        area = float(self.face_areas()[2])
        tau = f_shear / area
        g_static = tau / strain
        self.wall_pos[top] = z0_top
        self.wall_pos[bot] = z0_bot
        self.vel.zero_()
        self.omega.zero_()
        self.rebuild_pairs()
        return {
            "strain": strain,
            "shear_force_N": f_shear,
            "shear_stress_Pa": tau,
            "shear_modulus_Pa": g_static,
            "relax_steps": relax_steps,
            "kinetic_energy_J": self.kinetic_energy(),
            "wall_friction": self.cfg.wall_friction,
        }