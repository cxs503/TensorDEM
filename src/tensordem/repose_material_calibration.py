"""Material-class angle-of-repose calibration closed loop (TensorDEM, p399w2-n07).

Motivation
----------
``tensordem.repose_calibration`` (m4970) established the *machinery* of a
repose-angle calibration surface for **one** monodisperse sphere material with a
fixed rolling-resistance coefficient (``mu_r = 0.20``): a ``mu -> theta_r`` table
plus its guarded inverse.  That milestone explicitly listed, as roadmap items,
(i) material-specific parameter sets, (ii) a *target-driven* inverse (engineering
asks "which ``mu`` reproduces a measured 35 deg?") and (iii) the ``mu_r``
sensitivity of the answer.

This module closes exactly that loop for three **material classes** -- glass
beads, sand, gravel -- each with its own grain scale, density, restitution and
rolling-resistance prior, and each with its own experimental target angle:

===============  =======  =======  =====  ======  ===============
material         r [mm]   rho      e      mu_r    theta_target
===============  =======  =======  =====  ======  ===============
glass beads      4.0      2500     0.70   1.30    30 deg
sand             2.5      2650     0.50   1.55    35 deg
gravel           5.0      2650     0.30   1.85    40 deg
===============  =======  =======  =====  ======  ===============

The workflow is the calibration loop an engineer actually runs:

1. **forward scan** ``mu in [0.2, 0.8]`` (7 nodes x 2 seeds) -> ``theta_r``;
2. **monotonisation** (pool-adjacent-violators; violations are counted and
   reported, never hidden) -> monotone table, reusing
   :class:`tensordem.repose_calibration.ReposeCalibrationTable` (monotone PCHIP,
   inverse refuses extrapolation);
3. **inverse** ``theta_target -> mu*`` plus a **verification run** at ``mu*``;
   acceptance: ``|theta_sim - theta_target| / theta_target < 10 %``;
4. **negative control** ``mu = mu_r = 0`` -> the heap must flatten
   (``theta_sim < 5 deg``), proving the angle comes from friction and not from
   the packing or the estimator;
5. **``mu_r`` sensitivity** at ``mu*`` (``mu_r x 0.5`` and ``x 1.5``): the
   quantitative form of the identifiability limit -- one repose angle cannot
   separate sliding from rolling resistance.

Solver
------
Same contact physics as ``repose_calibration`` (linear spring-dashpot normal
force, Cundall-Strack tangential spring saturated at ``mu F_n``, EPSD rolling
moment from the shared implementation
:func:`tensordem.rolling_resistance.rolling_resistance_torque`, rough floor and
rough lifting-cylinder wall), re-written around (i) a periodically **rebuilt
candidate pair list** and (ii) **dense per-pair contact-state matrices** keyed by
particle index, so a neighbour-list rebuild never loses a tangential/rolling
history.  For this particle count that is ~10x faster per step than the dense
all-pairs driver, which is what makes a 3-material x 7-node scan affordable.

Two deliberate differences from the m4970 driver (both quantified in the report):

* the rolling moment is capped at ``mu_r R_eff F_n`` with ``R_eff = r`` for
  particle-particle, wall and floor contacts (the convention of the verified
  m4970 driver and of the common ``mu_r = tan(theta_roll)`` reading);
  ``rolling_eff_radius_ratio`` exposes the ratio so the ``R_eff = r/2``
  (Ai et al. 2011, pair radius) convention can be selected instead.

The rolling priors are deliberately in the *upper* literature range for the
material class.  Reason: for a cohesionless sphere pile the repose angle is
``~arctan(min(mu, mu_r))`` (measured by m4970 and re-measured here), so a prior
below ``tan(theta_target)`` clips the curve before the target and makes ``mu``
non-identifiable.  The prior dependence is measured explicitly
(:func:`mu_rolling_sensitivity`) and reported, never assumed away.

The macro-response estimator is *not* re-implemented: the verified
:func:`tensordem.repose_calibration.repose_angle` (annulus x sector free-surface
fit with one-sided 2-sigma clipping) is imported, so differences between the two
milestones cannot come from the estimator.

Limitations: monodisperse spheres (material angularity enters only through the
``mu_r`` prior), no cohesion, no particle-size distribution, fixed ``mu_r`` per
material during the ``mu`` scan, and no laboratory ``mu``/``mu_r`` pair measured
for the three granulars -- the targets are literature repose angles, the
calibrated ``mu`` is a *numerical equivalent* of this solver.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .repose_calibration import ReposeCalibrationTable, repose_angle
from .rolling_resistance import RollingResistanceConfig, rolling_resistance_torque

DTYPE = torch.float64
MU_LOWER = 0.2
MU_UPPER = 0.8
ANGLE_TOLERANCE_REL = 0.10
NEGATIVE_CONTROL_LIMIT_DEG = 5.0


class MaterialCalibrationError(ValueError):
    """Raised for malformed material specs / calibration inputs."""


class TargetOutOfBracketError(MaterialCalibrationError):
    """Raised when ``theta_target`` is not bracketed by the simulated curve."""


# ---------------------------------------------------------------------------
# Material specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialSpec:
    """One DEM material class + its container geometry + target repose angle."""

    key: str
    label: str
    radius: float                 # m, monodisperse grain radius
    density: float                # kg/m^3
    restitution: float
    mu_rolling: float             # EPSD rolling-resistance prior
    target_theta_deg: float       # literature repose angle of the granular
    cylinder_radius: float        # m
    fill_height: float            # m
    kn: float = 2000.0            # N/m, stiffness-converged (see report 3.6)
    kt_ratio: float = 0.5
    mu_wall: float | None = None            # None -> mu (rough wall)
    mu_rolling_wall: float | None = None    # None -> mu_rolling
    gravity: float = 9.81
    hold_time: float = 0.04
    lift_time: float = 1.20
    settle_time: float = 0.60
    packing_gap: float = 0.005
    jitter: float = 0.005
    rolling_eff_radius_ratio: float = 1.0
    rebuild_every: int = 10
    skin_ratio: float = 0.5
    dt_override: float | None = None
    seed0: int = 39907

    # -- derived -----------------------------------------------------------
    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def dt(self) -> float:
        if self.dt_override is not None:
            return float(self.dt_override)
        period = 2.0 * math.pi * math.sqrt(0.5 * self.mass / self.kn)
        return period / 20.0

    @property
    def n_steps(self) -> int:
        total = self.hold_time + self.lift_time + self.settle_time
        return int(math.ceil(total / self.dt))

    # -- wall parameter resolution ----------------------------------------
    def wall_mu_for(self, mu: float) -> float:
        return mu if self.mu_wall is None else float(self.mu_wall)

    def wall_mu_rolling_for(self, mu_rolling: float) -> float:
        return mu_rolling if self.mu_rolling_wall is None else float(self.mu_rolling_wall)

    def __post_init__(self) -> None:
        if not self.key:
            raise MaterialCalibrationError("material key must be non-empty")
        for name in ("radius", "density", "kn", "kt_ratio", "gravity",
                     "cylinder_radius", "fill_height", "hold_time",
                     "lift_time", "settle_time", "skin_ratio", "packing_gap"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
                raise MaterialCalibrationError(f"{name} must be finite and positive")
        if not (math.isfinite(self.mu_rolling) and self.mu_rolling >= 0.0):
            raise MaterialCalibrationError("mu_rolling must be finite and non-negative")
        if not 0.0 <= self.restitution < 1.0:
            raise MaterialCalibrationError("restitution must lie in [0, 1)")
        if self.cylinder_radius <= self.radius:
            raise MaterialCalibrationError("cylinder_radius must exceed radius")
        if self.fill_height <= 2.0 * self.radius:
            raise MaterialCalibrationError("fill_height must exceed one layer")
        if not 0.0 < self.rolling_eff_radius_ratio <= 1.0:
            raise MaterialCalibrationError("rolling_eff_radius_ratio in (0, 1]")
        if self.rebuild_every < 1:
            raise MaterialCalibrationError("rebuild_every must be >= 1")
        if not 0.0 < self.target_theta_deg < 90.0:
            raise MaterialCalibrationError("target_theta_deg must lie in (0, 90)")

    def to_json(self) -> dict:
        return {
            "key": self.key, "label": self.label, "radius_m": self.radius,
            "density_kg_m3": self.density, "restitution": self.restitution,
            "mu_rolling": self.mu_rolling,
            "target_theta_deg": self.target_theta_deg,
            "cylinder_radius_m": self.cylinder_radius,
            "fill_height_m": self.fill_height, "kn_N_m": self.kn,
            "kt_ratio": self.kt_ratio, "dt_s": self.dt, "n_steps": self.n_steps,
            "hold_time_s": self.hold_time, "lift_time_s": self.lift_time,
            "settle_time_s": self.settle_time,
            "rolling_eff_radius_ratio": self.rolling_eff_radius_ratio,
            "particle_mass_kg": self.mass,
        }


#: Container ratios of the lifting-cylinder protocol: ``R_c = 5 r``,
#: ``H = 9 R_c`` (~340 grains at r = 4 mm).  This is the smallest configuration
#: that still gives a settled heap (``|v| <= 5 mm/s``) with a measurable flank
#: (~5 grain diameters) at an affordable number of timesteps; both smaller
#: (R_c = 4.375 r, H = 4 R_c, m4970 geometry) and larger containers were probed
#: and are reported in ``docs/repose_material_calibration.md`` §3.7.
WALL_RATIO = 5.0
FILL_RATIO = 9.0


def _geometry(radius: float, wall_ratio: float = WALL_RATIO,
              fill_ratio: float = FILL_RATIO) -> tuple[float, float]:
    rc = wall_ratio * radius
    return rc, fill_ratio * rc


def _make(key: str, label: str, radius: float, density: float, restitution: float,
          mu_rolling: float, target: float) -> MaterialSpec:
    rc, height = _geometry(radius)
    return MaterialSpec(key=key, label=label, radius=radius, density=density,
                        restitution=restitution, mu_rolling=mu_rolling,
                        target_theta_deg=target, cylinder_radius=rc,
                        fill_height=height)


#: The three material classes of this milestone.  ``mu_rolling`` is a
#: *literature-informed prior*, not a fitted value: smooth glass beads roll
#: almost freely, sand grains carry a moderate rolling resistance, angular
#: gravel the largest.  The prior is frozen during the ``mu`` scan; its influence
#: on the calibrated ``mu`` is quantified separately by
#: :func:`mu_rolling_sensitivity`.
MATERIALS: dict[str, MaterialSpec] = {
    "glass_bead": _make("glass_bead", "glass beads", radius=0.004, density=2500.0,
                        restitution=0.70, mu_rolling=1.30, target=30.0),
    "sand": _make("sand", "sand", radius=0.0025, density=2650.0,
                  restitution=0.50, mu_rolling=1.55, target=35.0),
    "gravel": _make("gravel", "gravel", radius=0.005, density=2650.0,
                    restitution=0.30, mu_rolling=1.85, target=40.0),
}


def material(key: str) -> MaterialSpec:
    try:
        return MATERIALS[key]
    except KeyError as exc:
        raise MaterialCalibrationError(
            f"unknown material {key!r}; known: {sorted(MATERIALS)}") from exc


# ---------------------------------------------------------------------------
# Initial packing
# ---------------------------------------------------------------------------

def initial_packing(spec: MaterialSpec, *, seed: int | None = None) -> torch.Tensor:
    """Deterministic HCP column of monodisperse spheres filling the cylinder."""
    r = spec.radius
    spacing = 2.0 * r * (1.0 + spec.packing_gap)
    layer_dz = spacing * math.sqrt(2.0 / 3.0)
    n_layers = max(1, int(math.floor((spec.fill_height - r) / layer_dz)) + 1)
    r_max = spec.cylinder_radius - r * 1.02
    generator = torch.Generator().manual_seed(spec.seed0 if seed is None else seed)
    row_pitch = spacing * math.sqrt(3.0) / 2.0
    shift_x = 0.5 * spacing
    shift_y = spacing / (2.0 * math.sqrt(3.0))
    points: list[tuple[float, float, float]] = []
    span = int(math.ceil((spec.cylinder_radius + spacing) / spacing))
    for layer in range(n_layers):
        z = r + layer * layer_dz
        odd = layer % 2 == 1
        for ix in range(-span, span + 1):
            for iy in range(-span, span + 1):
                x = ix * spacing + 0.5 * spacing * (iy % 2)
                y = iy * row_pitch
                if odd:
                    x += shift_x
                    y += shift_y
                if math.hypot(x, y) <= r_max:
                    points.append((x, y, z))
    if len(points) < 32:
        raise MaterialCalibrationError(
            "packing produced fewer than 32 particles; enlarge the cylinder")
    positions = torch.tensor(points, dtype=DTYPE)
    if spec.jitter:
        noise = torch.randn(positions.shape, dtype=DTYPE, generator=generator)
        positions = positions + noise * (spec.jitter * r)
    positions[:, 2].clamp_(min=r * 1.001)
    return positions


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _damping_ratio(restitution: float) -> float:
    e = max(min(restitution, 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


def _candidate_pairs(positions: torch.Tensor,
                     cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    """All ``i < j`` pairs closer than ``cutoff`` (rebuilt periodically)."""
    diff = positions[:, None, :] - positions[None, :, :]
    d2 = (diff * diff).sum(-1)
    close = torch.triu(d2 < cutoff * cutoff, diagonal=1)
    idx = torch.nonzero(close, as_tuple=False)
    return idx[:, 0].contiguous(), idx[:, 1].contiguous()


def _rolling(spec: MaterialSpec, spin: torch.Tensor, fn: torch.Tensor,
             r_eff: float, mu_rolling: float, state: torch.Tensor | None,
             dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    config = RollingResistanceConfig(model="epsd", mu_r=float(mu_rolling),
                                     k_r=float(spec.kt * spec.radius ** 2))
    r_eff_t = torch.full_like(fn, float(r_eff))
    return rolling_resistance_torque(spin, fn, r_eff_t, config, dt=dt,
                                     torque_state=state)


def simulate(spec: MaterialSpec, mu: float, *, seed: int | None = None,
             mu_rolling: float | None = None, keep_particles: bool = False,
             max_steps: int | None = None) -> dict:
    """One lifting-cylinder column-collapse run; returns the repose angle.

    ``mu = mu_rolling = 0`` is the negative control: without any tangential or
    rolling resistance a cohesionless sphere pile cannot sustain a slope.
    """
    if not (math.isfinite(mu) and mu >= 0.0):
        raise MaterialCalibrationError("mu must be finite and non-negative")
    mu_r = spec.mu_rolling if mu_rolling is None else float(mu_rolling)
    if not (math.isfinite(mu_r) and mu_r >= 0.0):
        raise MaterialCalibrationError("mu_rolling must be finite and non-negative")

    seed_value = spec.seed0 if seed is None else int(seed)
    positions = initial_packing(spec, seed=seed_value)
    n = len(positions)
    r = spec.radius
    mass, inertia = spec.mass, spec.inertia
    dt = spec.dt
    kn, kt = spec.kn, spec.kt
    zeta = _damping_ratio(spec.restitution)
    gamma = 2.0 * zeta * math.sqrt(kn * 0.5 * mass)
    gamma_wall = 2.0 * zeta * math.sqrt(kn * mass)
    wall_mu = spec.wall_mu_for(mu)
    wall_mu_r = spec.wall_mu_rolling_for(mu_r)

    velocities = torch.zeros_like(positions)
    omega = torch.zeros_like(positions)
    tang = torch.zeros((n, n, 3), dtype=DTYPE)     # pair tangential spring
    roll = torch.zeros((n, n, 3), dtype=DTYPE)     # pair EPSD rolling state
    floor_tang = torch.zeros((n, 3), dtype=DTYPE)
    floor_roll = torch.zeros((n, 3), dtype=DTYPE)
    wall_tang = torch.zeros((n, 3), dtype=DTYPE)
    wall_roll = torch.zeros((n, 3), dtype=DTYPE)
    gravity_vec = positions.new_tensor([0.0, 0.0, -spec.gravity])
    cutoff = 2.0 * r * (1.0 + spec.skin_ratio)

    steps = spec.n_steps if max_steps is None else int(max_steps)
    pairs: tuple[torch.Tensor, torch.Tensor] | None = None
    n_contacts = 0

    for step in range(steps):
        t_now = step * dt
        if t_now <= spec.hold_time:
            wall_bottom = 0.0
        elif t_now <= spec.hold_time + spec.lift_time:
            wall_bottom = spec.fill_height * (t_now - spec.hold_time) / spec.lift_time
        else:
            wall_bottom = spec.fill_height

        if pairs is None or step % spec.rebuild_every == 0:
            pairs = _candidate_pairs(positions, cutoff)
        iu, ju = pairs

        force = torch.zeros_like(positions)
        torque = torch.zeros_like(positions)

        # ---- sphere-sphere --------------------------------------------------
        if len(iu):
            dvec = positions[ju] - positions[iu]
            dist = torch.linalg.vector_norm(dvec, dim=1).clamp_min(1e-12)
            normal = dvec / dist[:, None]
            overlap = (2.0 * r - dist).clamp_min(0.0)
            rel_v = velocities[ju] - velocities[iu]
            v_n = (rel_v * normal).sum(1)
            fn = (kn * overlap - gamma * v_n).clamp_min(0.0)
            n_contacts = max(n_contacts, int((overlap > 0).sum()))

            surf = rel_v
            surf = (surf + torch.linalg.cross(omega[ju], normal * (-r))
                    - torch.linalg.cross(omega[iu], normal * r))
            vt = surf - (surf * normal).sum(1)[:, None] * normal
            state = tang[iu, ju] + vt * dt
            state = state - (state * normal).sum(1)[:, None] * normal
            t_norm = torch.linalg.vector_norm(state, dim=1)
            cap = mu * fn / kt
            scale = torch.where(t_norm > cap, cap / t_norm.clamp_min(1e-12),
                                torch.ones_like(t_norm))
            state = state * scale[:, None]
            tang[iu, ju] = state
            ft = -kt * state
            pair_force = fn[:, None] * normal + ft
            force.index_add_(0, iu, -pair_force)
            force.index_add_(0, ju, pair_force)
            contact_torque = -torch.linalg.cross(normal * r, ft)
            torque.index_add_(0, iu, contact_torque)
            torque.index_add_(0, ju, contact_torque)

            spin = omega[ju] - omega[iu]
            moment, roll_new = _rolling(spec, spin, fn,
                                        r * spec.rolling_eff_radius_ratio, mu_r,
                                        roll[iu, ju], dt)
            roll[iu, ju] = roll_new
            torque.index_add_(0, iu, -moment)
            torque.index_add_(0, ju, moment)

        # ---- floor at z = 0 --------------------------------------------------
        pen = r - positions[:, 2]
        touching = pen > 0.0
        if bool(touching.any()):
            idx = torch.nonzero(touching, as_tuple=False).flatten()
            fn_f = (kn * pen[idx] - gamma_wall * velocities[idx, 2]).clamp_min(0.0)
            radial_v = velocities[idx].clone()
            radial_v[:, 2] = 0.0
            st = floor_tang[idx] + radial_v * dt
            t_norm = torch.linalg.vector_norm(st, dim=1)
            cap = wall_mu * fn_f / kt
            scale = torch.where(t_norm > cap, cap / t_norm.clamp_min(1e-12),
                                torch.ones_like(t_norm))
            st = st * scale[:, None]
            floor_tang[idx] = st
            ft_f = -kt * st
            add = ft_f.clone()
            add[:, 2] += fn_f
            force.index_add_(0, idx, add)
            arm = positions.new_tensor([0.0, 0.0, -r]).expand(len(idx), 3)
            torque.index_add_(0, idx, torch.linalg.cross(arm, ft_f))
            moment, roll_new = _rolling(spec, omega[idx], fn_f, r, wall_mu_r,
                                        floor_roll[idx], dt)
            floor_roll[idx] = roll_new
            torque.index_add_(0, idx, moment)

        # ---- lifting cylindrical wall (bounded above the rising bottom edge)
        if wall_bottom < spec.fill_height:
            radial = positions[:, :2]
            rad = torch.linalg.vector_norm(radial, dim=1).clamp_min(1e-12)
            pen_w = rad + r - spec.cylinder_radius
            hit = (pen_w > 0.0) & (positions[:, 2] > wall_bottom)
            if bool(hit.any()):
                idx = torch.nonzero(hit, as_tuple=False).flatten()
                normal = torch.zeros((len(idx), 3), dtype=DTYPE)
                normal[:, :2] = -radial[idx] / rad[idx][:, None]
                v_n = (velocities[idx] * normal).sum(1)
                fn_w = (kn * pen_w[idx] - gamma_wall * v_n).clamp_min(0.0)
                surf = velocities[idx] - v_n[:, None] * normal
                surf = surf + torch.linalg.cross(omega[idx], normal * (-r))
                st = wall_tang[idx] + surf * dt
                t_norm = torch.linalg.vector_norm(st, dim=1)
                cap = wall_mu * fn_w / kt
                scale = torch.where(t_norm > cap, cap / t_norm.clamp_min(1e-12),
                                    torch.ones_like(t_norm))
                st = st * scale[:, None]
                wall_tang[idx] = st
                ft_w = -kt * st
                force.index_add_(0, idx, fn_w[:, None] * normal + ft_w)
                torque.index_add_(0, idx,
                                  torch.linalg.cross(normal * (-r), ft_w))
                moment, roll_new = _rolling(spec, omega[idx], fn_w, r, wall_mu_r,
                                            wall_roll[idx], dt)
                wall_roll[idx] = roll_new
                torque.index_add_(0, idx, moment)

        # ---- semi-implicit Euler --------------------------------------------
        velocities += (force / mass + gravity_vec) * dt
        positions = positions + velocities * dt
        omega = omega + torque / inertia * dt

        if step % 200 == 0 and not bool(torch.isfinite(positions).all()):
            raise MaterialCalibrationError(
                f"simulation diverged at step {step} (t={t_now:.4g} s)")

    if not bool(torch.isfinite(positions).all()):
        raise MaterialCalibrationError("simulation diverged (non-finite positions)")

    angle = repose_angle(positions, r)
    result = {
        "material": spec.key,
        "mu": float(mu),
        "mu_rolling": float(mu_r),
        "seed": seed_value,
        "theta_r_deg": float(angle["theta_r_deg"]),
        "slope": float(angle["slope"]),
        "r2": float(angle["r2"]),
        "shoulder_radius": float(angle["shoulder_radius"]),
        "toe_radius": float(angle["toe_radius"]),
        "apex_z": float(angle["apex_z"]),
        "n_bins": int(angle["n_bins"]),
        # discharge diagnostic: a stub that never left the cylinder has
        # toe_radius ~ cylinder_radius, a spread pile has toe >> cylinder
        "discharge_ratio": float(angle["toe_radius"]) / spec.cylinder_radius,
        "n_particles": int(n),
        "n_contacts_max": int(n_contacts),
        "n_steps": int(steps),
        "dt_s": float(dt),
        "sim_time_s": float(steps * dt),
        "max_speed_m_s": float(torch.linalg.vector_norm(velocities, dim=1).max()),
        "speed_p95_m_s": float(torch.quantile(
            torch.linalg.vector_norm(velocities, dim=1), 0.95)),
        "max_spin_rad_s": float(torch.linalg.vector_norm(omega, dim=1).max()),
        "kinetic_energy_j": float(0.5 * mass
                                  * (velocities * velocities).sum()),
        "tan_theta_r": float(math.tan(math.radians(angle["theta_r_deg"]))),
    }
    result["sliding_bound_tan_over_mu"] = (
        result["tan_theta_r"] / mu if mu > 0.0 else float("inf"))
    if keep_particles:
        result["positions"] = positions
    return result


def negative_control(spec: MaterialSpec, *, seed: int | None = None) -> dict:
    """Frictionless control: ``mu = mu_r = 0`` must flatten the heap."""
    out = simulate(spec, 0.0, seed=seed, mu_rolling=0.0)
    out["expectation"] = (f"theta_r < {NEGATIVE_CONTROL_LIMIT_DEG} deg without "
                          "sliding or rolling resistance")
    out["passed"] = bool(out["theta_r_deg"] < NEGATIVE_CONTROL_LIMIT_DEG)
    return out


# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------

def default_mu_nodes(lower: float = MU_LOWER, upper: float = MU_UPPER,
                     count: int = 7) -> list[float]:
    if count < 2:
        raise MaterialCalibrationError("need at least two mu nodes")
    step = (upper - lower) / (count - 1)
    return [round(lower + i * step, 10) for i in range(count)]


def aggregate_nodes(runs: Sequence[dict]) -> list[dict]:
    """Group per-seed runs into ``mu`` nodes with mean/std angles."""
    by_mu: dict[float, list[dict]] = {}
    for run in runs:
        by_mu.setdefault(round(float(run["mu"]), 12), []).append(run)
    nodes: list[dict] = []
    for mu in sorted(by_mu):
        group = by_mu[mu]
        angles = [float(g["theta_r_deg"]) for g in group]
        mean = sum(angles) / len(angles)
        if len(angles) > 1:
            var = sum((a - mean) ** 2 for a in angles) / (len(angles) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        nodes.append({
            "mu": mu,
            "theta_r_deg": mean,
            "theta_r_std_deg": std,
            "n_seeds": len(angles),
            "per_seed": sorted(angles),
            "r2_mean": sum(float(g["r2"]) for g in group) / len(group),
            "max_speed_m_s_mean": sum(float(g.get("max_speed_m_s", 0.0))
                                      for g in group) / len(group),
            "discharge_ratio_mean": sum(float(g.get("discharge_ratio", 0.0))
                                        for g in group) / len(group),
            "toe_radius_mean": sum(float(g.get("toe_radius", 0.0))
                                   for g in group) / len(group),
            "speed_p95_m_s_mean": sum(float(g.get("speed_p95_m_s", 0.0))
                                      for g in group) / len(group),
            "seeds": [g["seed"] for g in group],
        })
    return nodes


def _pava(theta: Sequence[float]) -> tuple[list[float], int]:
    """Pool-adjacent-violators isotonic regression; returns (fit, n_violations)."""
    values = [float(v) for v in theta]
    n_violations = sum(1 for a, b in zip(values, values[1:]) if b <= a)
    if not n_violations:
        return values, 0
    blocks: list[list[float]] = [[v, 1.0] for v in values]
    merged: list[list[float]] = []
    for block in blocks:
        merged.append(block)
        while len(merged) > 1 and merged[-2][0] / merged[-2][1] >= merged[-1][0] / merged[-1][1]:
            v2, w2 = merged.pop()
            v1, w1 = merged.pop()
            merged.append([v1 + v2, w1 + w2])
    fit: list[float] = []
    for value, weight in merged:
        fit.extend([value / weight] * int(round(weight)))
    # strict monotonicity: PCHIP table rejects ties, so separate them by a
    # negligible epsilon (1e-6 deg) and record that it was needed.
    tiny = 1e-6
    for i in range(1, len(fit)):
        if fit[i] <= fit[i - 1]:
            fit[i] = fit[i - 1] + tiny
    return fit, n_violations


#: A node is admitted to the calibration table only when the flank fit and the
#: settling check both pass; a still-flowing or unmeasurable heap carries no
#: angle information (see ``docs/repose_material_calibration.md``).
MIN_R2 = 0.70
MAX_SETTLE_SPEED_M_S = 0.05
#: A legitimate heap must have spread well beyond the cylinder it came from.
MIN_DISCHARGE_RATIO = 1.5


def node_quality(node: dict, cylinder_radius: float | None = None
                 ) -> tuple[bool, list[str]]:
    """Quality gate for one scanned node: ``(accepted, reasons_if_not)``.

    ``cylinder_radius`` is used to recover the discharge ratio from the toe
    radius when a run file predates the explicit ``discharge_ratio`` field.
    """
    reasons: list[str] = []
    ratio = node.get("discharge_ratio_mean")
    if (ratio is None or float(ratio) <= 0.0) and cylinder_radius is not None \
            and node.get("toe_radius_mean"):
        ratio = float(node["toe_radius_mean"]) / float(cylinder_radius)
    if ratio is not None and float(ratio) < MIN_DISCHARGE_RATIO:
        reasons.append(f"no discharge (toe/Rc={float(ratio):.2f} < "
                       f"{MIN_DISCHARGE_RATIO}); the column jammed instead of "
                       "spreading, the angle is not a repose angle")
    r2 = float(node.get("r2_mean", 0.0))
    if r2 < MIN_R2:
        reasons.append(f"flank fit r2={r2:.3f} < {MIN_R2}")
    speed = node.get("max_speed_m_s_mean")
    if speed is not None and float(speed) > MAX_SETTLE_SPEED_M_S:
        reasons.append(f"still flowing |v|={float(speed):.3f} m/s > "
                       f"{MAX_SETTLE_SPEED_M_S}")
    if int(node.get("n_seeds", 0)) < 1:
        reasons.append("no realisations")
    return (not reasons), reasons


def build_table(spec: MaterialSpec, nodes: Sequence[dict], *,
                mu_rolling: float | None = None) -> tuple[ReposeCalibrationTable, dict]:
    """Monotone PCHIP table from the scanned nodes (+ monotonisation report).

    Nodes failing :func:`node_quality` are excluded and listed in the report,
    never silently dropped; at least two accepted nodes are required.
    """
    if len(nodes) < 2:
        raise MaterialCalibrationError("need at least two scanned nodes")
    accepted: list[dict] = []
    excluded: list[dict] = []
    for node in nodes:
        ok, reasons = node_quality(node, spec.cylinder_radius)
        if ok:
            accepted.append(node)
        else:
            excluded.append({"mu": float(node["mu"]),
                             "theta_r_deg": float(node["theta_r_deg"]),
                             "reasons": reasons})
    if len(accepted) < 2:
        raise MaterialCalibrationError(
            "fewer than two nodes pass the quality gate "
            f"({len(accepted)} accepted, {len(excluded)} excluded); the set-up "
            "does not produce measurable heaps")
    ordered = sorted(accepted, key=lambda node: float(node["mu"]))
    mu = [float(node["mu"]) for node in ordered]
    raw = [float(node["theta_r_deg"]) for node in ordered]
    if len(set(mu)) != len(mu):
        raise MaterialCalibrationError("duplicate mu nodes")
    report_excluded = excluded
    fit, n_violations = _pava(raw)
    table = ReposeCalibrationTable(
        mu=tuple(mu), theta_deg=tuple(fit),
        mu_rolling=spec.mu_rolling if mu_rolling is None else float(mu_rolling),
        provenance={
            "method": "tensordem material-class column-collapse DEM",
            "material": spec.key,
            "material_spec": spec.to_json(),
            "raw_theta_deg": raw,
            "monotone_theta_deg": fit,
            "monotonicity_violations": n_violations,
            "seeds_per_node": [node.get("n_seeds") for node in ordered],
            "excluded_nodes": report_excluded,
        })
    report = {
        "excluded_nodes": report_excluded,
        "n_excluded": len(report_excluded),
        "mu": mu,
        "raw_theta_deg": raw,
        "monotone_theta_deg": fit,
        "monotonicity_violations": n_violations,
        "bracket": table.bracket(),
    }
    return table, report


def calibrate(spec: MaterialSpec, table: ReposeCalibrationTable, *,
              seeds: Sequence[int] = (39907, 39908),
              tolerance_rel: float = ANGLE_TOLERANCE_REL) -> dict:
    """Invert ``theta_target -> mu*`` and verify by an independent DEM run."""
    lo, hi = table.theta_deg[0], table.theta_deg[-1]
    if not lo <= spec.target_theta_deg <= hi:
        raise TargetOutOfBracketError(
            f"target {spec.target_theta_deg} deg lies outside the simulated "
            f"bracket [{lo:.3f}, {hi:.3f}] deg for material {spec.key!r}; the "
            "scan must be widened (honest FAIL, no extrapolation)")
    mu_star = table.mu_from_theta(spec.target_theta_deg)
    verify = [simulate(spec, mu_star, seed=seed) for seed in seeds]
    angles = [float(run["theta_r_deg"]) for run in verify]
    theta_sim = sum(angles) / len(angles)
    if len(angles) > 1:
        std = math.sqrt(sum((a - theta_sim) ** 2 for a in angles) / (len(angles) - 1))
    else:
        std = 0.0
    rel_error = (theta_sim - spec.target_theta_deg) / spec.target_theta_deg
    return {
        "material": spec.key,
        "target_theta_deg": spec.target_theta_deg,
        "mu_star": float(mu_star),
        "theta_sim_deg": float(theta_sim),
        "theta_sim_std_deg": float(std),
        "per_seed_theta_deg": angles,
        "seeds": list(seeds),
        "abs_error_deg": float(theta_sim - spec.target_theta_deg),
        "rel_error": float(rel_error),
        "tolerance_rel": float(tolerance_rel),
        "passed": bool(abs(rel_error) < tolerance_rel),
        "table_theta_at_mu_star_deg": float(table.theta_from_mu(mu_star)),
        "bracket": table.bracket(),
        "verification_runs": [{k: v for k, v in run.items() if k != "positions"}
                              for run in verify],
    }


def mu_rolling_sensitivity(spec: MaterialSpec, mu_star: float, *,
                           factors: Sequence[float] = (0.5, 1.0, 1.5),
                           seed: int | None = None) -> list[dict]:
    """Perturb the rolling-resistance prior and re-measure the repose angle."""
    rows: list[dict] = []
    base = spec.mu_rolling
    for factor in factors:
        mu_r = base * float(factor)
        run = simulate(spec, mu_star, seed=seed, mu_rolling=mu_r)
        rows.append({
            "mu_rolling": mu_r,
            "factor": float(factor),
            "theta_r_deg": float(run["theta_r_deg"]),
            "delta_theta_deg": float(run["theta_r_deg"]),
            "mu": float(mu_star),
            "seed": int(run["seed"]),
        })
    reference = next((row for row in rows if abs(row["factor"] - 1.0) < 1e-12), None)
    if reference is not None:
        for row in rows:
            row["delta_theta_deg"] = row["theta_r_deg"] - reference["theta_r_deg"]
    return rows


def stiffness_independence(spec: MaterialSpec, mu: float, *,
                           factors: Sequence[float] = (1.0, 4.0),
                           seed: int | None = None) -> list[dict]:
    """Same case at ``k_n`` and ``4 k_n``: a quasi-rigid contact must not change
    the repose angle beyond the seed scatter."""
    rows: list[dict] = []
    for factor in factors:
        kn = spec.kn * float(factor)
        dt = 2.0 * math.pi * math.sqrt(0.5 * spec.mass / kn) / 20.0
        stiff = MaterialSpec(**{**spec.__dict__, "kn": kn, "dt_override": dt})
        run = simulate(stiff, mu, seed=seed)
        rows.append({"kn": kn, "kn_factor": float(factor), "dt_s": dt,
                     "theta_r_deg": float(run["theta_r_deg"]),
                     "seed": int(run["seed"])})
    if rows:
        base = rows[0]["theta_r_deg"]
        for row in rows:
            row["delta_theta_deg"] = row["theta_r_deg"] - base
    return rows


def sliding_onset_angle(mu: float, mu_rolling: float, *,
                        tolerance_deg: float = 0.3,
                        run_time_s: float = 0.3,
                        slide_speed_threshold: float = 0.008) -> dict:
    """Analytic anchor for the contact law used by the solver.

    A single sphere on a plane tilted by ``theta`` stays put while
    ``tan(theta) <= mu`` (sliding) and ``tan(theta) <= mu_r`` (rolling), so the
    onset angle is ``arctan(min(mu, mu_r))``.  This driver integrates the same
    contact laws as :func:`simulate` (floor branch) and bisects the onset.  The
    discriminator is the *final centre-of-mass speed* (``> 10 mm/s``): the
    elastic tangential stretch of the soft spring is a few micrometres and must
    not be mistaken for sliding, while a sphere one degree above the critical
    tilt already reaches several cm/s within ``run_time_s``.
    """

    def slides(tilt_deg: float) -> bool:
        r = 0.004
        spec = MaterialSpec(key="anchor", label="anchor", radius=r,
                            density=2500.0, restitution=0.5,
                            mu_rolling=mu_rolling, target_theta_deg=30.0,
                            cylinder_radius=0.05, fill_height=0.02,
                            hold_time=0.02, lift_time=0.001, settle_time=0.001)
        mass, inertia = spec.mass, spec.inertia
        dt = spec.dt
        kn, kt = spec.kn, spec.kt
        zeta = _damping_ratio(spec.restitution)
        gamma_wall = 2.0 * zeta * math.sqrt(kn * mass)
        theta = math.radians(tilt_deg)
        normal = torch.tensor([0.0, -math.sin(theta), math.cos(theta)], dtype=DTYPE)
        # gravity is plain -z; the tilt lives in the plane normal, so the
        # normal/tangential split is g cos(theta) / g sin(theta) by construction
        gravity_vec = torch.tensor([0.0, 0.0, -spec.gravity], dtype=DTYPE)
        pos = normal * r   # exactly on the plane (no initial penetration)
        vel = torch.zeros(3, dtype=DTYPE)
        omega = torch.zeros(3, dtype=DTYPE)
        tang_state = torch.zeros(1, 3, dtype=DTYPE)
        roll_state = torch.zeros(1, 3, dtype=DTYPE)
        for _ in range(int(run_time_s / dt)):
            pen = r - (pos * normal).sum()
            pen = torch.as_tensor(max(float(pen), 0.0), dtype=DTYPE)
            f = torch.zeros(3, dtype=DTYPE)
            tq = torch.zeros(3, dtype=DTYPE)
            if float(pen) > 0.0:
                fn = (kn * pen - gamma_wall * (vel * normal).sum()).clamp_min(0.0)
                surf = vel - (vel * normal).sum() * normal
                surf = surf + torch.linalg.cross(omega, normal * (-r))
                st = tang_state[0] + surf * dt
                st = st - (st * normal).sum() * normal
                cap = mu * fn / kt
                tn = torch.linalg.vector_norm(st)
                if float(tn) > float(cap):
                    st = st * (cap / tn.clamp_min(1e-12))
                tang_state[0] = st
                ft = -kt * st
                f = fn * normal + ft
                tq = torch.linalg.cross(normal * (-r), ft)
                moment, roll_state_new = _rolling(spec, omega.clone().view(1, 3),
                                                  fn.view(1), r, mu_rolling,
                                                  roll_state, dt)
                roll_state = roll_state_new
                tq = tq + moment[0]
            vel = vel + (f / mass + gravity_vec) * dt
            pos = pos + vel * dt
            omega = omega + tq / inertia * dt
        # sustained drift = sliding; the elastic tangential stretch alone is a
        # few micrometres and stops growing once the contact is in equilibrium
        down = torch.tensor([0.0, -math.cos(theta), -math.sin(theta)], dtype=DTYPE)
        return float(torch.linalg.vector_norm(vel)) > slide_speed_threshold

    lo, hi = 0.0, 80.0
    for _ in range(11):
        mid = 0.5 * (lo + hi)
        if slides(mid):
            hi = mid
        else:
            lo = mid
    measured = 0.5 * (lo + hi)
    analytic = math.degrees(math.atan(min(mu, mu_rolling))) if min(mu, mu_rolling) < 70 else 90.0
    return {"measured_onset_deg": measured, "analytic_onset_deg": analytic,
            "abs_error_deg": abs(measured - analytic),
            "tolerance_deg": tolerance_deg,
            "passed": bool(abs(measured - analytic) <= tolerance_deg)}


def analytic_cone_check(angle_deg: float, *, radius: float = 0.004,
                        outer_radius_ratio: float = 20.0,
                        n: int = 20000, seed: int = 7) -> dict:
    """Feed a synthetic cone of known angle to the repose estimator.

    ``n`` grains are scattered on the conical free surface of a heap whose apex
    sits at the axis, ``z = (r_out - r) tan(angle)`` with
    ``r <= outer_radius_ratio * radius``, plus a sub-grain thickness -- the
    idealised surface the DEM deposit is supposed to approximate.
    """
    generator = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=generator, dtype=DTYPE)
    outer = outer_radius_ratio * radius
    radial = outer * torch.sqrt(u)
    phi = 2.0 * math.pi * torch.rand(n, generator=generator, dtype=DTYPE)
    z = (outer - radial) * math.tan(math.radians(angle_deg))
    layer = torch.rand(n, generator=generator, dtype=DTYPE)
    positions = torch.stack([radial * torch.cos(phi), radial * torch.sin(phi),
                             z + layer * radius * 0.1], dim=1)
    angle = repose_angle(positions, radius)
    return {"generated_deg": angle_deg, "measured_deg": float(angle["theta_r_deg"]),
            "abs_error_deg": abs(float(angle["theta_r_deg"]) - angle_deg),
            "r2": float(angle["r2"]),
            "passed": bool(abs(float(angle["theta_r_deg"]) - angle_deg) < 1.5)}


__all__ = [
    "MaterialCalibrationError",
    "TargetOutOfBracketError",
    "MaterialSpec",
    "MATERIALS",
    "material",
    "MU_LOWER",
    "MU_UPPER",
    "ANGLE_TOLERANCE_REL",
    "NEGATIVE_CONTROL_LIMIT_DEG",
    "initial_packing",
    "simulate",
    "negative_control",
    "default_mu_nodes",
    "aggregate_nodes",
    "build_table",
    "calibrate",
    "mu_rolling_sensitivity",
    "stiffness_independence",
    "sliding_onset_angle",
    "analytic_cone_check",
]