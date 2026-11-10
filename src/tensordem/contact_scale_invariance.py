"""Geometric-scale invariance of DEM contact parameter sets (M2/M3).

The ledger in :mod:`tensordem.contact_param_ledger` answers *what* the required
scale exponent of every contact parameter is.  This module answers the
physical question behind it: **does a macroscopically identical granular case
still give the same answer after the particles (and the geometry) are scaled by
``lambda``?**

Three macroscopic probes are used, all driven by the repository's own
``granular3d_core`` contact laws (linear spring + dashpot, Cundall-Strack
tangential spring, isotropic rolling spring):

``repose``
    Column collapse on a frictional floor with removable side walls; the probe
    is the free-surface slope of the deposited heap, ``theta_r`` [deg].
``shear``
    Direct shear of a confined specimen under a constant normal stress with an
    imposed platen velocity; the probe is the mobilised peak ratio
    ``tau_peak / sigma`` [-].
``drag``
    A vertical blade pushed horizontally through a settled bed; the probe is
    the dimensionless resistance ``F / (rho g R^3)`` [-].

Similarity family
-----------------
``L -> lam L`` with the grain density and gravity held fixed (Froude/gravity
similarity), which is the assumption behind DEM coarse-graining.  Then

* all lengths (``R``, box, blade, fill height) scale as ``lam``,
* time, velocity and the time step scale as ``lam^(1/2)``,
* ``kn``, ``kt``, ``kr`` (linear springs, ``F = k`` ``delta``) scale as
  ``lam^2``,
* the *rolling lever arm* ``mu_r [m]`` scales as ``lam`` (equivalently: the
  dimensionless ratio ``M_r = mu_r R_eff |F_n|`` is invariant),
* ``mu``, ``e``, ``kt_ratio`` and lattice ratios are invariant,
* stresses scale as ``lam`` and forces as ``lam^3``.

The transform itself is not hand-coded here: it is applied by
:func:`tensordem.contact_param_ledger.scale_dataclass`, so the exponents used
by the experiment are exactly the ones the ledger publishes.

Negative controls (``M3``)
--------------------------
Each control is applied *on top of* the correctly scaled case, so any
deviation is attributable to the single mistake:

``kn_constant``      ``kn`` left at its base value (stiffness not scaled)
``kn_sqrt``          ``kn ~ lam^(1/2)`` - the "Hertz sqrt(R)" recipe
``kn_hertz``         ``kn ~ lam^(3/2)`` - Hertz parameterisation, gravity
                     similarity violated for a linear-spring kernel
``lever_constant``   rolling lever arm held constant, i.e. ``mu_r [m]``
                     treated as a dimensionless number
``restitution_boost`` ``e + 0.1`` (out of the admissible range when e = 0.95)

The first, fourth and fifth are the controls required by the node contract;
the two extra stiffness laws are reported as a diagnostic because the
literature quote for the stiffness exponent is ambiguous.

Limitations
-----------
Monodisperse spheres, no cohesion, smooth (frictionless) walls except the
floor, rolling resistance only between particles, and a modest particle count
(see :data:`DEFAULT_RESOLUTION`).  The probes are *numerical* invariance
probes of this solver, not laboratory validations.

All quantities are SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np

from .contact_param_ledger import scale_dataclass
from .granular3d_core import (
    ContactState,
    Material3D,
    build_pairs,
    contact_forces,
    damping_ratio,
    hcp_positions_in_box,
)


# ---------------------------------------------------------------------------
# Case / wall description
# ---------------------------------------------------------------------------

from .contact_param_ledger import PARAMETERS as _PARAMS

_PARAM_NAMES = frozenset(_PARAMS)


@dataclass(frozen=True)
class GranularCase:
    """SI parameters of one probe case.

    Field names are the ledger's canonical names, so
    :func:`tensordem.contact_param_ledger.scale_dataclass` scales the case with
    the published exponents and raises on anything unlabelled.

    ``rolling_lever`` is the physical lever arm ``mu_r [m]`` of
    ``M_r = mu_r |F_n|``; the kernel consumes the dimensionless ratio
    ``rolling_lever / radius``.
    """

    # material
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.55
    rolling_lever: float = 1.5e-3
    restitution: float = 0.40
    kn: float = 6.0e2
    kt_ratio: float = 0.7
    gravity: float = 9.81
    # geometry (all lengths; scaled by lambda)
    box_width: float = 0.070
    box_depth: float = 0.060
    fill_height: float = 0.120
    plate_width: float = 0.060
    # protocol
    packing_gap: float = 0.005
    jitter: float = 0.0
    hold_time: float = 0.12
    normal_stress: float = 4000.0
    shear_velocity: float = 0.05
    drag_velocity: float = 0.06
    volume_fraction_target: float = 0.6
    seed: int = 20260810
    dt: float | None = None

    # -- derived ---------------------------------------------------------
    @property
    def mu_rolling(self) -> float:
        """Dimensionless rolling ratio consumed by ``granular3d_core``."""
        return self.rolling_lever / self.radius

    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    def material(self) -> Material3D:
        return Material3D(radius=self.radius, density=self.density, mu=self.mu,
                          mu_rolling=self.mu_rolling,
                          restitution=self.restitution, kn=self.kn,
                          kt_ratio=self.kt_ratio, gravity=self.gravity)

    def timestep(self, steps_per_period: float = 25.0) -> float:
        """Contact-resolved step; ``dt`` overrides it when set."""
        if self.dt is not None:
            return float(self.dt)
        period = 2.0 * math.pi * math.sqrt(0.5 * self.mass / self.kn)
        return period / steps_per_period

    def gravity_time(self) -> float:
        return math.sqrt(self.fill_height / self.gravity)

    def as_parameters(self) -> dict[str, float]:
        """Registered parameters only (the ledger's vocabulary, no derivations)."""
        import dataclasses as _dc

        from .contact_param_ledger import normalise

        out: dict[str, float] = {}
        for f in _dc.fields(self):
            value = getattr(self, f.name)
            if value is None or isinstance(value, bool):
                continue
            if normalise(f.name) in _PARAM_NAMES:
                out[f.name] = float(value)
        return out

    def as_dict(self) -> dict[str, float]:
        return {
            "radius": self.radius, "density": self.density, "mu": self.mu,
            "rolling_lever": self.rolling_lever,
            "mu_rolling_dimensionless": self.mu_rolling,
            "restitution": self.restitution, "kn": self.kn,
            "kt_ratio": self.kt_ratio, "gravity": self.gravity,
            "box_width": self.box_width, "box_depth": self.box_depth,
            "fill_height": self.fill_height, "plate_width": self.plate_width,
            "packing_gap": self.packing_gap, "jitter": self.jitter,
            "hold_time": self.hold_time, "normal_stress": self.normal_stress,
            "shear_velocity": self.shear_velocity,
            "drag_velocity": self.drag_velocity, "seed": float(self.seed),
            "timestep": self.timestep(),
            "dt_override": self.dt,
            "overlap_ratio_at_own_weight": (
                self.mass * self.gravity / (self.kn * self.radius)),
        }


@dataclass
class Plane:
    """A bounded or unbounded friction plane (wall / floor / platen / blade).

    ``normal`` points into the domain; ``point`` is on the plane.  A particle
    penetrates when ``(pos - point) . normal < radius``.  ``z_top``/``z_bottom``
    bound the plane vertically (used for the blade); ``t_on``/``t_off`` switch
    it on/off in time (removable confinement).
    """

    label: str
    normal: np.ndarray
    point: np.ndarray
    mu: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    z_top: float | None = None
    z_bottom: float | None = None
    t_on: float = 0.0
    t_off: float | None = None
    state: dict[int, np.ndarray] = field(default_factory=dict)
    roll_state: dict[int, np.ndarray] = field(default_factory=dict)
    last_force: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=np.float64)
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.point = np.asarray(self.point, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)

    def active(self, t: float) -> bool:
        if t < self.t_on:
            return False
        if self.t_off is not None and t >= self.t_off:
            return False
        return True


# ---------------------------------------------------------------------------
# Engine (single set of contact laws, shared by all probes)
# ---------------------------------------------------------------------------


class _Engine:
    """Minimal monodisperse DEM loop on top of ``granular3d_core``."""

    def __init__(self, case: GranularCase, pos: np.ndarray) -> None:
        self.case = case
        self.mat = case.material()
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.vel = np.zeros_like(self.pos)
        self.omega = np.zeros_like(self.pos)
        self.state = ContactState()
        self.mass = np.full(len(self.pos), case.mass)
        self.inertia = case.inertia
        self.dt = case.timestep()
        self.t = 0.0

    # -- forces ---------------------------------------------------------
    def _wall_forces(self, planes: Sequence[Plane]
                     ) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.pos)
        force = np.zeros((n, 3))
        torque = np.zeros((n, 3))
        r = self.case.radius
        kn, kt = self.mat.kn, self.mat.kt
        gamma_w = 2.0 * damping_ratio(self.case.restitution) * math.sqrt(
            kn * self.case.mass)
        for plane in planes:
            if not plane.active(self.t):
                plane.last_force = np.zeros(3)
                continue
            d = (self.pos - plane.point) @ plane.normal
            mask = d < r
            if plane.z_top is not None:
                mask &= self.pos[:, 2] <= plane.z_top + r
            if plane.z_bottom is not None:
                mask &= self.pos[:, 2] >= plane.z_bottom - r
            if not mask.any():
                plane.last_force = np.zeros(3)
                continue
            idx = np.flatnonzero(mask)
            overlap = r - d[idx]
            nvec = np.broadcast_to(plane.normal, (idx.size, 3))
            v_rel = self.vel[idx] - plane.velocity
            v_n = (v_rel * nvec).sum(axis=1)
            f_n = np.maximum(kn * overlap - gamma_w * v_n, 0.0)
            f = f_n[:, None] * nvec
            if plane.mu > 0.0:
                # contact-point surface velocity of the particle
                surf = np.cross(self.omega[idx], -nvec * r)
                v_t = (v_rel + surf) - ((v_rel + surf) * nvec).sum(
                    axis=1)[:, None] * nvec
                tang = np.array([plane.state.get(int(i), np.zeros(3))
                                 for i in idx])
                tang = tang + v_t * self.dt
                tn = np.linalg.norm(tang, axis=1)
                cap = plane.mu * f_n / kt
                scale = np.where(tn > cap, cap / np.maximum(tn, 1e-30), 1.0)
                tang = tang * scale[:, None]
                for k, i in enumerate(idx):
                    plane.state[int(i)] = tang[k]
                f_t = -kt * tang
                f = f + f_t
                torque[idx] += np.cross(-nvec * r, f_t)
            # rolling resistance at the wall: without it a sphere rolling on
            # the floor coasts forever (no rolling torque is transmitted), and
            # the heap never comes to rest.
            mu_r = self.mat.mu_rolling
            if mu_r > 0.0:
                kr = self.mat.kr
                omega_t = self.omega[idx] - (self.omega[idx]
                                             * nvec).sum(axis=1)[:, None] * nvec
                roll = np.array([plane.roll_state.get(int(i), np.zeros(3))
                                 for i in idx])
                roll = roll + omega_t * self.dt
                rn = np.linalg.norm(roll, axis=1)
                cap_r = (mu_r * r * f_n) / kr
                scale_r = np.where(rn > cap_r, cap_r / np.maximum(rn, 1e-30), 1.0)
                roll = roll * scale_r[:, None]
                for k, i in enumerate(idx):
                    plane.roll_state[int(i)] = roll[k]
                torque[idx] += -kr * roll
            force[idx] += f
            plane.last_force = -f.sum(axis=0)
        return force, torque

    def step(self, planes: Sequence[Plane],
             external: Callable[[], np.ndarray] | None = None) -> None:
        iu, ju = build_pairs(self.pos, self.case.radius)
        force, torque = contact_forces(self.pos, self.vel, self.omega, iu, ju,
                                       self.state, self.mat, self.dt)
        f_wall, t_wall = self._wall_forces(planes)
        force += f_wall
        torque += t_wall
        force[:, 2] -= self.case.mass * self.case.gravity
        if external is not None:
            force += external()
        acc = force / self.mass[:, None]
        ang_acc = torque / self.inertia
        self.pos += self.vel * self.dt + 0.5 * acc * self.dt ** 2
        self.vel += acc * self.dt
        self.omega += ang_acc * self.dt
        self.t += self.dt

    def kinetic_energy(self) -> float:
        return float(0.5 * (self.case.mass * (self.vel ** 2).sum()
                            + self.inertia * (self.omega ** 2).sum()))


def _box_planes(case: GranularCase, *, floor_mu: float,
                wall_mu: float = 0.0, t_off: float | None = None
                ) -> list[Plane]:
    W, D = case.box_width, case.box_depth
    zero = np.zeros(3)
    return [
        Plane("floor", np.array([0.0, 0.0, 1.0]), zero, mu=floor_mu),
        Plane("wall_x0", np.array([1.0, 0.0, 0.0]), zero, mu=wall_mu,
              t_off=t_off),
        Plane("wall_x1", np.array([-1.0, 0.0, 0.0]), np.array([W, 0.0, 0.0]),
              mu=wall_mu, t_off=t_off),
        Plane("wall_y0", np.array([0.0, 1.0, 0.0]), zero, mu=wall_mu,
              t_off=t_off),
        Plane("wall_y1", np.array([0.0, -1.0, 0.0]), np.array([0.0, D, 0.0]),
              mu=wall_mu, t_off=t_off),
    ]


def _pack(case: GranularCase) -> np.ndarray:
    return hcp_positions_in_box(case.radius, case.box_width, case.box_depth,
                                case.fill_height,
                                packing_gap=case.packing_gap,
                                jitter=case.jitter, seed=case.seed,
                                margin=0.2 * case.radius)


# ---------------------------------------------------------------------------
# Probe A: angle of repose from a column collapse
# ---------------------------------------------------------------------------


def _heap_angle(pos: np.ndarray, case: GranularCase, *, n_bins: int = 16
                ) -> tuple[float, dict[str, Any]]:
    """Deposit flank slope (repose angle) from the settled heap.

    Protocol: the upper envelope (97th percentile) of the particle centres in
    x-bins, taken over the *whole* deposited footprint (0.5-99.5 percentile of
    x so that a stray particle cannot stretch the window).  A straight line is
    fitted over both flanks (every bin between the crest and the toe), and the
    reported angle is the mean of the two flank angles.

    This is the dam-break deposit slope, which for a cohesionless packing
    reproduces the material's repose angle only in the limit of a
    friction-dominated (low aspect ratio) collapse; the probe therefore also
    reports the crest height and the run-out so that the aspect ratio can be
    checked, and the *value* of the angle is used as an invariance probe, not as
    a calibrated repose angle of a real material.
    """
    x = pos[:, 0]
    lo_x, hi_x = np.percentile(x, [0.5, 99.5])
    if hi_x - lo_x <= 4.0 * case.radius:
        return 0.0, {"note": "no run-out"}
    edges = np.linspace(lo_x, hi_x, n_bins + 1)
    centres: list[float] = []
    heights: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() < 3:
            continue
        centres.append(0.5 * (lo + hi))
        heights.append(float(np.percentile(pos[sel, 2], 97.0)))
    cx = np.asarray(centres)
    cz = np.asarray(heights)
    if cx.size < 6:
        return 0.0, {"note": "too few bins"}
    crest = int(np.argmax(cz))
    flanks: list[float] = []
    meta: dict[str, Any] = {"crest_height": float(cz[crest]),
                            "runout": float(hi_x - lo_x),
                            "aspect_ratio": float(cz[crest] / (hi_x - lo_x)),
                            "n_bins_used": int(cx.size)}
    for side in ("left", "right"):
        sl = slice(0, crest + 1) if side == "left" else slice(crest, cx.size)
        bx, bz = cx[sl], cz[sl]
        if bx.size < 3:
            continue
        slope = float(np.polyfit(bx, bz, 1)[0])
        flanks.append(abs(slope))
        meta[f"slope_{side}"] = slope
    if not flanks:
        return 0.0, {**meta, "note": "no fittable flank"}
    meta["flank_angles_deg"] = [math.degrees(math.atan(f)) for f in flanks]
    return math.degrees(math.atan(float(np.mean(flanks)))), meta


def repose_probe(case: GranularCase, *, relax_time_factor: float = 6.0,
                 shots: Sequence[float] = ()) -> dict[str, Any]:
    """Column collapse -> repose angle (``theta_r`` in degrees)."""
    pos = _pack(case)
    engine = _Engine(case, pos)
    planes = _box_planes(case, floor_mu=case.mu, t_off=case.hold_time)
    total = case.hold_time + relax_time_factor * case.gravity_time()
    steps = int(math.ceil(total / engine.dt))
    ke_hist: list[float] = []
    for _ in range(steps):
        engine.step(planes)
        ke_hist.append(engine.kinetic_energy())
    angle, meta = _heap_angle(engine.pos, case)
    g_time = case.gravity_time()
    return {
        "case": "repose",
        "theta_r_deg": float(angle),
        "n_particles": int(len(engine.pos)),
        "steps": steps,
        "timestep": engine.dt,
        "ke_final_over_mg_h": float(
            ke_hist[-1] / (len(engine.pos) * case.mass * case.gravity
                           * case.fill_height)),
        "shear_rate_at_end": float(
            np.abs(engine.vel).mean() * g_time / case.fill_height),
        **meta,
    }


# ---------------------------------------------------------------------------
# Probe B: direct-shear yield ratio
# ---------------------------------------------------------------------------


def shear_probe(case: GranularCase, *, shear_strain_target: float = 0.25
                ) -> dict[str, Any]:
    """Constant-stress direct shear -> mobilised peak ``tau / sigma``."""
    pos = _pack(case)
    engine = _Engine(case, pos)
    planes = _box_planes(case, floor_mu=case.mu)
    settle_steps = int(math.ceil(case.hold_time / engine.dt))
    for _ in range(settle_steps):
        engine.step(planes)
    z_max = engine.pos[:, 2].max()
    top = np.flatnonzero(engine.pos[:, 2] > z_max - 3.0 * case.radius)
    area = case.box_width * case.box_depth
    force_per_top = case.normal_stress * area / max(top.size, 1)
    shear_time = shear_strain_target * case.fill_height / case.shear_velocity
    n_steps = int(math.ceil(shear_time / engine.dt))
    tau: list[float] = []
    strain: list[float] = []
    displacement = 0.0
    floor = planes[0]

    def external() -> np.ndarray:
        f = np.zeros((len(engine.pos), 3))
        f[top, 2] -= force_per_top
        return f

    for _ in range(n_steps):
        displacement += case.shear_velocity * engine.dt
        engine.step(planes, external)
        engine.vel[top, 0] = case.shear_velocity
        engine.vel[top, 2] = np.minimum(engine.vel[top, 2], 0.0)
        # the shear transmitted through the specimen is the horizontal
        # reaction mobilised on the floor (force balance of the specimen)
        tau.append(abs(float(floor.last_force[0])) / area)
        strain.append(displacement / case.fill_height)
    tau_a = np.asarray(tau)
    strain_a = np.asarray(strain)
    steady = strain_a >= 0.5 * strain_a[-1] if strain_a.size else None
    peak = float(tau_a.max()) if tau_a.size else 0.0
    tail = (float(tau_a[steady].mean())
            if steady is not None and steady.any() else peak)
    return {
        "case": "shear",
        "tau_over_sigma_peak": float(peak / case.normal_stress),
        "tau_over_sigma_steady": float(tail / case.normal_stress),
        "n_particles": int(len(engine.pos)),
        "n_top": int(top.size),
        "steps": settle_steps + n_steps,
        "timestep": engine.dt,
        "shear_strain": float(strain_a[-1]) if strain_a.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Probe C: blade resistance in a settled bed
# ---------------------------------------------------------------------------


def drag_probe(case: GranularCase, *, blade_height_factor: float = 3.0,
               travel_factor: float = 2.0) -> dict[str, Any]:
    """Blade pushed through a settled bed -> ``F / (rho g R^3)``."""
    pos = _pack(case)
    engine = _Engine(case, pos)
    planes = _box_planes(case, floor_mu=case.mu)
    settle_steps = int(math.ceil(case.hold_time / engine.dt))
    for _ in range(settle_steps):
        engine.step(planes)

    x_start = -0.5 * case.radius
    blade_height = blade_height_factor * case.radius
    blade = Plane("blade", np.array([1.0, 0.0, 0.0]),
                  np.array([x_start, 0.0, 0.0]), mu=0.0,
                  velocity=np.array([case.drag_velocity, 0.0, 0.0]),
                  z_top=blade_height)
    planes = planes + [blade]
    travel = travel_factor * case.radius
    n_steps = int(math.ceil(travel / (case.drag_velocity * engine.dt)))
    forces: list[float] = []
    for k in range(n_steps):
        blade.point = np.array(
            [x_start + case.drag_velocity * k * engine.dt, 0.0, 0.0])
        engine.step(planes)
        forces.append(float(-blade.last_force[0]))
    f_a = np.asarray(forces)
    # steady window: the second half of the travel
    half = f_a[f_a.size // 2:]
    scale = case.density * case.gravity * case.radius ** 3
    return {
        "case": "drag",
        "resistance_dimensionless": float(half.mean() / scale),
        "resistance_dimensionless_peak": float(f_a.max() / scale),
        "force_N": float(half.mean()),
        "n_particles": int(len(engine.pos)),
        "steps": settle_steps + n_steps,
        "timestep": engine.dt,
        "blade_height": blade_height,
        "travel": travel,
    }


# ---------------------------------------------------------------------------
# Probe D: restitution anchor (single-sphere rebound)
# ---------------------------------------------------------------------------


def rebound_probe(case: GranularCase, *, drop_heights: float = 20.0,
                  steps_per_period: float = 100.0) -> dict[str, Any]:
    """Single sphere dropped on the floor -> rebound height ratio (``e^2``).

    This is both a verification anchor for the dashpot law (the linear
    spring-dashpot reproduces ``e`` through
    ``zeta = |ln e| / sqrt(pi^2 + ln^2 e)``) and the observable that actually
    responds to a restitution mistake, which the quasi-static probes do not.
    """
    r = case.radius
    z0 = r + drop_heights * r
    dt = 2.0 * math.pi * math.sqrt(0.5 * case.mass / case.kn) / steps_per_period
    engine = _Engine(replace(case, dt=dt),
                     np.array([[0.5 * case.box_width,
                                0.5 * case.box_depth, z0]]))
    floor = Plane("floor", np.array([0.0, 0.0, 1.0]), np.zeros(3), mu=case.mu)
    t_fall = math.sqrt(2.0 * max(z0 - r, 1e-12) / case.gravity)
    n_steps = int(math.ceil(2.5 * t_fall / dt))
    bounced = False
    apex = r
    first_contact_t = None
    for _ in range(n_steps):
        engine.step([floor])
        z = float(engine.pos[0, 2])
        vz = float(engine.vel[0, 2])
        if not bounced and z <= 1.02 * r and vz <= 0.0:
            bounced = True
            first_contact_t = engine.t
        if bounced:
            apex = max(apex, z)
    ratio = (apex - r) / (z0 - r)
    return {
        "case": "rebound",
        "rebound_height_ratio": float(ratio),
        "restitution_measured": float(math.sqrt(max(ratio, 0.0))),
        "restitution_input": case.restitution,
        "drop_height": float(z0 - r),
        "n_particles": 1,
        "steps": n_steps,
        "timestep": dt,
        "first_contact_time": float(first_contact_t or 0.0),
    }


PROBES: dict[str, Callable[..., dict[str, Any]]] = {
    "repose": repose_probe,
    "shear": shear_probe,
    "drag": drag_probe,
    "rebound": rebound_probe,
}

#: The probe's decision metric (the number compared across scale variants).
METRIC_KEYS: dict[str, str] = {
    "repose": "theta_r_deg",
    "shear": "tau_over_sigma_peak",
    "drag": "resistance_dimensionless",
    "rebound": "rebound_height_ratio",
}


# ---------------------------------------------------------------------------
# Scale variants: correct transform + negative controls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    name: str
    lam: float
    case: GranularCase
    kind: str           # "reference" | "similar" | "control" | "diagnostic"
    expected: str       # "invariant" | "violation"
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "lam": self.lam, "kind": self.kind,
                "expected": self.expected, "note": self.note,
                "case": self.case.as_dict()}


def scaled_variants(base: GranularCase, lam: float = 2.0,
                    *, controls: bool = True) -> list[Variant]:
    """Reference, correctly scaled case and the negative controls."""
    scaled = scale_dataclass(base, lam)
    assert isinstance(scaled, GranularCase)
    variants = [
        Variant("lam1_reference", 1.0, base, "reference", "invariant",
                "unscaled reference"),
        Variant("lamN_similar", lam, scaled, "similar", "invariant",
                "all lengths x lam, kn x lam^2, mu_r[m] x lam, dt x lam^0.5"),
    ]
    if not controls:
        return variants
    dt_ref = scaled.timestep()
    # (a) stiffness not scaled at all
    variants.append(Variant("ctl_kn_constant", lam, replace(scaled, kn=base.kn, dt=dt_ref),
                            "control", "violation",
                            "kn held at the base value (wrong scaling)"))
    # (b) rolling lever arm held constant = mu_r treated as dimensionless
    variants.append(Variant("ctl_lever_constant", lam,
                            replace(scaled, rolling_lever=base.rolling_lever, dt=dt_ref),
                            "control", "violation",
                            "mu_r [m] not scaled with R"))
    # (c) restitution pushed out of the admissible range
    variants.append(Variant("ctl_restitution_boost", lam,
                            replace(scaled,
                                    restitution=base.restitution + 0.1,
                                    dt=dt_ref),
                            "control", "violation",
                            "e + 0.1 (in range: sensitivity to restitution)"))
    variants.append(Variant("ctl_restitution_out_of_range", lam,
                            replace(scaled, restitution=1.05, dt=dt_ref),
                            "control", "violation",
                            "e = 1.05, outside the admissible [0, 1] band"))
    # diagnostics: the two literature stiffness exponents
    variants.append(Variant("diag_kn_sqrt", lam,
                            replace(scaled, kn=base.kn * lam ** 0.5,
                                           dt=dt_ref),
                            "diagnostic", "unknown",
                            "kn ~ lam^0.5 (fixed-modulus Hertz sqrt(R) rule)"))
    variants.append(Variant("diag_kn_hertz", lam,
                            replace(scaled, kn=base.kn * lam ** 1.5,
                                           dt=dt_ref),
                            "diagnostic", "unknown",
                            "kn ~ lam^1.5 (Hertz stiffness, E* fixed)"))
    return variants


def run_matrix(base: GranularCase, probes: Sequence[str] = ("repose", "shear",
                                                           "drag"),
               lam: float = 2.0, controls: bool = True
               ) -> list[dict[str, Any]]:
    """Run every probe on every scale variant and return raw records."""
    variants = scaled_variants(base, lam, controls=controls)
    records: list[dict[str, Any]] = []
    for probe_name in probes:
        probe = PROBES[probe_name]
        metric = METRIC_KEYS[probe_name]
        for variant in variants:
            result = probe(variant.case)
            records.append({
                "probe": probe_name,
                "metric": metric,
                "variant": variant.name,
                "lam": variant.lam,
                "kind": variant.kind,
                "expected": variant.expected,
                "note": variant.note,
                "value": result[metric],
                "raw": {k: v for k, v in result.items()
                        if isinstance(v, (int, float, str))},
                "case": variant.case.as_dict(),
            })
    return records


def analyse(records: Sequence[dict[str, Any]],
            tolerance: float = 0.02) -> dict[str, Any]:
    """Compare every variant against the correctly scaled variant.

    ``deviation`` is ``|x - x_similar| / |x_similar|`` on the probe's decision
    metric.  The predeclared acceptance thresholds are: correct transform
    ``<= tolerance`` (2 %), negative controls ``> 10 %``.
    """
    table: dict[str, Any] = {"tolerance": tolerance,
                             "control_threshold": 0.10, "probes": {}}
    by_probe: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in records:
        by_probe.setdefault(rec["probe"], {})[rec["variant"]] = rec
    for probe_name, variants in by_probe.items():
        ref = variants.get("lamN_similar")
        entry: dict[str, Any] = {"metric": METRIC_KEYS[probe_name],
                                 "variants": {}}
        if ref is None:
            table["probes"][probe_name] = entry
            continue
        ref_value = ref["value"]
        for name, rec in variants.items():
            value = rec["value"]
            dev = abs(value - ref_value) / abs(ref_value) if ref_value else math.inf
            rec_entry = {
                "value": value,
                "deviation": dev,
                "expected": rec["expected"],
                "kind": rec["kind"],
            }
            if rec["kind"] == "reference":
                rec_entry["verdict"] = "REFERENCE"
            elif rec["expected"] == "invariant":
                rec_entry["verdict"] = "PASS" if dev <= tolerance else "FAIL"
            elif rec["expected"] == "violation":
                rec_entry["verdict"] = ("CAUGHT" if dev > 0.10 else "MISSED")
            else:
                rec_entry["verdict"] = "DIAGNOSTIC"
            entry["variants"][name] = rec_entry
        table["probes"][probe_name] = entry
    table["gate"] = _gate(table)
    return table


def _gate(table: dict[str, Any]) -> dict[str, Any]:
    """Machine verdict: invariance met and all negative controls caught."""
    invariance = {}
    controls = {}
    for probe_name, entry in table["probes"].items():
        for name, rec in entry["variants"].items():
            if rec["verdict"] in ("PASS", "FAIL"):
                invariance[f"{probe_name}:{name}"] = rec["verdict"]
            elif rec["verdict"] in ("CAUGHT", "MISSED"):
                controls[f"{probe_name}:{name}"] = rec["verdict"]
    n_inv_fail = sum(1 for v in invariance.values() if v == "FAIL")
    n_ctl_miss = sum(1 for v in controls.values() if v == "MISSED")
    return {
        "invariance_checks": invariance,
        "negative_controls": controls,
        "n_invariance_fail": n_inv_fail,
        "n_controls_missed": n_ctl_miss,
        "pass": n_inv_fail == 0 and n_ctl_miss == 0,
    }


__all__ = [
    "GranularCase", "Plane", "Variant", "scaled_variants", "run_matrix",
    "analyse", "repose_probe", "shear_probe", "drag_probe", "PROBES",
    "METRIC_KEYS",
]