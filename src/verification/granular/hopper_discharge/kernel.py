"""Vectorised 3-D DEM kernel for a conical-hopper granular system (TensorDEM m7751).

Why a new kernel
----------------
``tensordem.granular3d_core`` is the house kernel, but its per-pair tangential
state lives in Python ``dict`` objects rebuilt with Python-level loops every
step.  Measured on this bench (see ``docs/hopper_discharge_verification.md``
section 2) that costs O(10^-1) s per step at the 3e4-particle inventory this
verification surface needs, i.e. hours per run.  This module keeps the *exact*
contact laws of ``granular3d_core`` (linear spring-dashpot normal force with the
same damping-ratio/restitution relation, Cundall-Strack incremental tangential
spring saturated at ``mu F_n``, optional rolling spring saturated at
``mu_r r F_n``, same torque convention) but stores the incremental state in
sorted NumPy arrays keyed by ``i*n0 + j`` and scatters forces with
``np.bincount``, so a step is O(N log N) with no Python-level inner loop.

Geometry
--------
Axisymmetric hopper, ``alpha`` = hopper half-angle measured from the vertical:

* barrel: cylinder of radius ``r_bin`` above the shoulder ``z_shoulder``;
* cone:   wall radius ``R(z) = R0 + z tan(alpha)`` between the orifice plate
          (``z = 0``) and the shoulder;
* plate:  annular flat bottom at ``z = 0`` outside the orifice radius ``R0``;
* heap mode: the barrel wall is lifted at ``t = wall_hold_time`` (linear ramp)
          so a free column collapses onto a frictional floor.

Conventions
-----------
SI everywhere.  Contact normal ``n`` points from body ``i`` to body ``j`` (pair
contact) or from the wall into the material (wall contact); the force on ``i``
is ``-f_total`` and on ``j`` ``+f_total``, matching ``granular3d_core``.

The integration scheme is velocity Verlet (kick-drift-kick) with the damping
term evaluated at the half-step velocity; one force evaluation per step.

Energy bookkeeping
------------------
Per step the kernel accumulates the mechanical energy
``E = KE_trans + KE_rot + m g z + U_normal + U_tangential + U_rolling`` and the
dissipated work ``W_diss = sum over contacts of (damping + sliding + rolling
power) * dt``.  ``W_diss`` is built from the *forces actually applied*, so the
ledger residual ``(E + W_diss - E_0) / E_scale`` measures the integrator's
trapezoidal error, never a fudge factor; it is reported as-is.

All quantities are SI (m, kg, s, N, Pa, rad).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


class HopperError(ValueError):
    """Raised when a hopper case violates the documented invariants."""


# ---------------------------------------------------------------------------
# Material / contact parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GranularParams:
    """Monodisperse sphere + linear spring-dashpot contact parameters (SI).

    The formulas are identical to ``tensordem.granular3d_core.Material3D`` so
    the two kernels describe the same material.
    """

    radius: float = 0.025          # 50 mm particles (campaign specification)
    density: float = 2500.0        # kg/m^3
    kn: float = 1.0e6              # N/m normal stiffness
    kt_ratio: float = 0.5          # kt / kn
    restitution: float = 0.5       # e
    mu: float = 0.4                # particle-particle Coulomb friction
    mu_rolling: float = 0.0        # rolling-resistance coefficient
    gravity: float = 9.81

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def kr(self) -> float:
        return self.kt * self.radius ** 2

    @property
    def zeta(self) -> float:
        """Damping ratio from the coefficient of restitution."""
        e = min(max(self.restitution, 1.0e-6), 1.0 - 1.0e-12)
        ln_e = math.log(e)
        return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)

    @property
    def gamma_pair(self) -> float:
        return 2.0 * self.zeta * math.sqrt(self.kn * 0.5 * self.mass)

    @property
    def gamma_wall(self) -> float:
        return 2.0 * self.zeta * math.sqrt(self.kn * self.mass)

    @property
    def timestep(self) -> float:
        """20 steps per linear-spring period (same rule as granular3d_core)."""
        return 2.0 * math.pi * math.sqrt(0.5 * self.mass / self.kn) / 20.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HopperGeometry:
    """Axisymmetric hopper / cylinder geometry (SI).

    ``kind='hopper'``  -> conical hopper with a circular orifice of radius
    ``orifice_radius`` (discharge face).
    ``kind='cylinder'`` -> straight cylinder with a closed flat bottom
    (Janssen face, and the initial container of the repose face).
    """

    r_bin: float = 0.8
    half_angle_deg: float = 30.0
    orifice_radius: float = 0.5
    fill_height: float = 1.2       # barrel height above the shoulder
    kind: str = "hopper"

    def __post_init__(self) -> None:
        if self.kind not in ("hopper", "cylinder"):
            raise HopperError("kind must be 'hopper' or 'cylinder'")
        if not 5.0 <= self.half_angle_deg < 90.0:
            raise HopperError("half angle must lie in [5, 90) degrees")
        for name in ("r_bin", "fill_height"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0.0):
                raise HopperError(f"{name} must be finite and positive")
        if self.kind == "hopper":
            if not (0.0 < self.orifice_radius < self.r_bin):
                raise HopperError("orifice radius must lie in (0, r_bin)")
        else:
            # a straight cylinder has a closed bottom: no aperture at all
            object.__setattr__(self, "orifice_radius", 0.0)

    @property
    def alpha(self) -> float:
        return math.radians(self.half_angle_deg)

    @property
    def tan_alpha(self) -> float:
        return math.tan(self.alpha)

    @property
    def cos_alpha(self) -> float:
        return math.cos(self.alpha)

    @property
    def sin_alpha(self) -> float:
        return math.sin(self.alpha)

    @property
    def cone_height(self) -> float:
        if self.kind == "cylinder":
            return 0.0
        return (self.r_bin - self.orifice_radius) / self.tan_alpha

    @property
    def z_shoulder(self) -> float:
        """Height of the cone/barrel junction."""
        return self.cone_height

    @property
    def z_top(self) -> float:
        """Free-surface height of the initial packing."""
        return self.cone_height + self.fill_height

    @property
    def orifice_diameter(self) -> float:
        return 2.0 * self.orifice_radius

    def radius_at(self, z: float) -> float:
        """Wall radius at height ``z`` (0 <= z <= z_top)."""
        if self.kind == "hopper" and z <= self.z_shoulder:
            return self.orifice_radius + max(z, 0.0) * self.tan_alpha
        return self.r_bin

    @property
    def volume(self) -> float:
        barrel = math.pi * self.r_bin ** 2 * self.fill_height
        if self.kind == "cylinder":
            return barrel
        r0 = self.orifice_radius
        cone = math.pi * self.cone_height / 3.0 * (r0 * r0 + r0 * self.r_bin
                                                  + self.r_bin ** 2)
        return barrel + cone

    def expected_count(self, radius: float, packing_fraction: float = 0.60) -> float:
        return self.volume * packing_fraction / (4.0 / 3.0 * math.pi * radius ** 3)

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "r_bin_m": self.r_bin,
            "half_angle_deg": self.half_angle_deg,
            "orifice_diameter_m": self.orifice_diameter,
            "orifice_radius_m": self.orifice_radius,
            "cone_height_m": self.cone_height,
            "barrel_fill_height_m": self.fill_height,
            "free_surface_height_m": self.z_top,
            "volume_m3": self.volume,
            "hydraulic_radius_m": 0.5 * self.r_bin,
        }


def lattice_packing(geom: HopperGeometry, radius: float, *, gap: float = 0.06,
                    jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """Deterministic triangular-lattice filling of the hopper interior.

    ``gap`` is the relative separation between neighbours (2r(1+gap)), so a
    modest gap guarantees the initial state has **no** interpenetration (a
    deeply overlapped birth state is what made earlier DEM silo cases diverge,
    see ``docs/beverloo_discharge_measurement_audit.md`` section 6).
    """
    if radius <= 0.0:
        raise HopperError("radius must be positive")
    if gap < 0.0:
        raise HopperError("gap must be nonnegative")
    step = 2.0 * radius * (1.0 + gap)
    row = step * math.sqrt(3.0) / 2.0
    # HCP stacking: the next layer sits above the triangle centroids, so the
    # interlayer distance that keeps the nearest-neighbour distance at ``step``
    # is step*sqrt(2/3).  (Omitting the centroid offset -- i.e. stacking layers
    # directly on top of each other -- gives 13% initial interpenetration at
    # these parameters and blows the run up within ~100 steps.)
    dz = step * math.sqrt(2.0 / 3.0)
    layers = []
    z = radius * (1.0 + gap)
    k = 0
    rng = np.random.default_rng(seed)
    while z <= geom.z_top - radius * (1.0 + gap) + 1e-12:
        r_max = geom.radius_at(z) - radius * (1.0 + 0.5 * gap)
        if r_max > 0.0:
            m = int(r_max / step) + 3
            x_off = 0.5 * step if k % 2 else 0.0
            y_off = row / 3.0 if k % 2 else 0.0
            xs_base = np.arange(-m, m + 1) * step + x_off
            ys = np.arange(-m, m + 1) * row + y_off
            pts = []
            for j, y in enumerate(ys):
                xs = xs_base + (step * 0.5 if j % 2 else 0.0)
                keep = np.hypot(xs, y) <= r_max
                if keep.any():
                    pts.append(np.stack([xs[keep], np.full(int(keep.sum()), y),
                                         np.full(int(keep.sum()), z)], axis=1))
            if pts:
                layers.append(np.vstack(pts))
        z += dz
        k += 1
    if not layers:
        raise HopperError("packing produced no particles; check geometry/radius")
    pos = np.vstack(layers)
    if jitter > 0.0:
        pos = pos + rng.standard_normal(pos.shape) * (jitter * radius)
    return pos


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HopperCase:
    """A complete DEM case: geometry + material + wall friction + protocol."""

    geometry: HopperGeometry = field(default_factory=HopperGeometry)
    params: GranularParams = field(default_factory=GranularParams)
    mu_wall: float = 0.4
    mu_rolling_wall: float = 0.0
    seed: int = 0
    packing_gap: float = 0.06
    jitter: float = 0.0
    dt: Optional[float] = None
    # protocol
    mode: str = "discharge"        # discharge | static | heap | pour
    floor_z: Optional[float] = None   # catch floor for the poured-heap face
    settle_time: float = 0.15
    window: float = 0.15
    n_windows: int = 6
    sample_time: float = 0.15      # static (Janssen) sampling window
    wall_hold_time: float = 0.12   # heap: wall in place
    wall_lift_time: float = 0.10   # heap: wall removal ramp
    free_settle_time: float = 0.70 # heap: collapse + repose
    slab_height: float = 0.10
    label: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("discharge", "static", "heap", "pour"):
            raise HopperError("mode must be discharge|static|heap|pour")
        if self.mode == "pour" and self.floor_z is None:
            raise HopperError("pour mode needs a catch floor (floor_z)")
        for name in ("mu_wall", "settle_time", "window", "sample_time",
                     "packing_gap", "jitter", "wall_hold_time", "wall_lift_time",
                     "free_settle_time"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0.0):
                raise HopperError(f"{name} must be finite and nonnegative")
        if self.n_windows < 5 and self.mode == "discharge":
            raise HopperError("discharge needs at least 5 flow windows")
        if self.geometry.kind == "hopper" and self.mode not in ("discharge", "pour"):
            raise HopperError("hopper geometry is only used in discharge/pour mode")
        if self.mode in ("discharge", "pour"):
            ratio = self.geometry.orifice_diameter / self.params.diameter
            if ratio < 4.0:
                raise HopperError("orifice must be at least 4 particle diameters")
        if self.dt is not None and not (0.0 < self.dt <= self.params.timestep):
            raise HopperError("dt must be positive and <= the stability timestep")

    @property
    def timestep(self) -> float:
        return self.params.timestep if self.dt is None else self.dt

    @property
    def initial_packing(self) -> np.ndarray:
        return lattice_packing(self.geometry, self.params.radius,
                               gap=self.packing_gap, jitter=self.jitter,
                               seed=self.seed)

    def describe(self) -> dict:
        p = self.params
        return {
            "label": self.label,
            "mode": self.mode,
            "geometry": self.geometry.describe(),
            "material": {
                "radius_m": p.radius, "diameter_m": p.diameter,
                "density_kg_m3": p.density, "kn_N_m": p.kn,
                "kt_ratio": p.kt_ratio, "restitution": p.restitution,
                "mu": p.mu, "mu_rolling": p.mu_rolling, "gravity_m_s2": p.gravity,
            },
            "wall": {"mu_wall": self.mu_wall,
                     "mu_rolling_wall": self.mu_rolling_wall},
            "floor_z_m": self.floor_z,
            "protocol": {
                "dt_s": self.timestep, "seed": self.seed,
                "packing_gap": self.packing_gap, "jitter": self.jitter,
                "settle_time_s": self.settle_time, "window_s": self.window,
                "n_windows": self.n_windows, "sample_time_s": self.sample_time,
                "wall_hold_time_s": self.wall_hold_time,
                "wall_lift_time_s": self.wall_lift_time,
                "free_settle_time_s": self.free_settle_time,
                "slab_height_m": self.slab_height,
            },
            "D_over_d": (self.geometry.orifice_diameter / p.diameter
                         if self.geometry.kind == "hopper" else None),
        }


# ---------------------------------------------------------------------------
# Incremental contact state (array based)
# ---------------------------------------------------------------------------

class _ContactBook:
    """Sorted-key incremental state for pair contacts (tangential + rolling)."""

    __slots__ = ("keys", "tang", "roll")

    def __init__(self) -> None:
        self.keys = np.zeros(0, dtype=np.int64)
        self.tang = np.zeros((0, 3), dtype=np.float64)
        self.roll = np.zeros((0, 3), dtype=np.float64)

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.keys)
        if n == 0 or len(keys) == 0:
            return np.zeros((len(keys), 3)), np.zeros((len(keys), 3))
        idx = np.searchsorted(self.keys, keys)
        idxc = np.clip(idx, 0, n - 1)
        ok = (idx < n) & (self.keys[idxc] == keys)
        tang = np.zeros((len(keys), 3))
        roll = np.zeros((len(keys), 3))
        tang[ok] = self.tang[idxc[ok]]
        roll[ok] = self.roll[idxc[ok]]
        return tang, roll

    def store(self, keys: np.ndarray, tang: np.ndarray,
              roll: np.ndarray) -> None:
        self.keys = keys
        self.tang = tang
        self.roll = roll


@dataclass
class _Ledger:
    """Mechanical-energy and dissipation bookkeeping (SI).

    ``dissipated`` accumulates two physically distinct losses:

    * the normal dashpot work, integrated with the trapezoidal rule over the
      step (``p_damping`` is the instantaneous power of the damping forces
      actually applied, so this is a measurement, not a model);
    * the *projection* loss of the incremental tangential/rolling springs,
      ``0.5 k (|s_pre|^2 - |s_post|^2)``, i.e. exactly the elastic energy that
      the Coulomb cap removes from the contact store in this step.  These
      losses are energies (J), not powers, and are not multiplied by ``dt``.

    Everything else in the system is conservative at the discrete level
    (linear spring potential, gravity, velocity-Verlet kinetic energy), so the
    residual ``E_mech + dissipated - E_0`` measures the trapezoidal
    integration floor only.
    """

    pe_normal: float = 0.0
    pe_tangential: float = 0.0
    pe_rolling: float = 0.0
    p_damping: float = 0.0
    e_clamp: float = 0.0
    dissipated: float = 0.0
    p_prev: float = 0.0
    n_samples: int = 0

    def begin_step(self) -> None:
        self.pe_normal = 0.0
        self.pe_tangential = 0.0
        self.pe_rolling = 0.0
        self.p_damping = 0.0
        self.e_clamp = 0.0

    def end_step(self, dt: float, *, first: bool = False) -> None:
        if first:
            self.p_prev = self.p_damping
            self.dissipated += self.e_clamp
        else:
            self.dissipated += 0.5 * (self.p_prev + self.p_damping) * dt
            self.dissipated += self.e_clamp
            self.p_prev = self.p_damping
        self.n_samples += 1


# ---------------------------------------------------------------------------
# Force evaluation
# ---------------------------------------------------------------------------

def _pair_forces(pos: np.ndarray, vel: np.ndarray, omg: np.ndarray,
                 gid: np.ndarray, n0: int, book: _ContactBook,
                 mat: GranularParams, dt: float, ledger: _Ledger,
                 pairs: Optional[np.ndarray] = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Particle-particle contacts: force, torque, energy/dissipation update."""
    n = len(pos)
    force = np.zeros((n, 3))
    torque = np.zeros((n, 3))
    if n < 2:
        book.store(np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3)))
        return force, torque

    if pairs is None:
        tree = cKDTree(pos, balanced_tree=False, compact_nodes=False)
        pairs = tree.query_pairs(2.0 * mat.radius, output_type="ndarray")
    if len(pairs) == 0:
        book.store(np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3)))
        return force, torque

    iu = pairs[:, 0].astype(np.int64)
    ju = pairs[:, 1].astype(np.int64)
    keys = gid[iu] * n0 + gid[ju]
    order = np.argsort(keys)
    iu, ju, keys = iu[order], ju[order], keys[order]

    r = mat.radius
    dvec = pos[ju] - pos[iu]
    dist = np.sqrt((dvec * dvec).sum(axis=1))
    overlap = 2.0 * r - dist
    active = overlap > 0.0
    iu, ju, keys = iu[active], ju[active], keys[active]
    if len(book.keys):
        keep = np.isin(book.keys, keys, assume_unique=True)
        if not keep.all():
            ledger.e_clamp += float(
                0.5 * mat.kt * (book.tang[~keep] ** 2).sum()
                + 0.5 * mat.kr * (book.roll[~keep] ** 2).sum())
    if len(keys) == 0:
        book.store(np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3)))
        return force, torque
    dvec = dvec[active]
    overlap = overlap[active]
    nvec = dvec / np.maximum(dist[active], 1e-30)[:, None]

    v_rel = vel[ju] - vel[iu]
    v_n = (v_rel * nvec).sum(axis=1)
    f_n = np.maximum(mat.kn * overlap - mat.gamma_pair * v_n, 0.0)

    tang, roll = book.lookup(keys)
    kt, kr = mat.kt, mat.kr
    if mat.mu > 0.0:
        surf_i = np.cross(omg[iu], nvec * r)
        surf_j = np.cross(omg[ju], -nvec * r)
        v_t = (v_rel - v_n[:, None] * nvec) + (surf_j - surf_i)
        tang_old = tang
        tang_pre = tang_old + v_t * dt
        tnorm = np.sqrt((tang_pre * tang_pre).sum(axis=1))
        cap = mat.mu * f_n / kt
        scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
        tang = tang_pre * scale[:, None]
        f_t = -kt * tang
        # dissipation := (work the friction force really does) - (change of the
        # contact store).  This identity is exact for the discrete scheme, so
        # the ledger cannot hide a bookkeeping error.
        # midpoint of the contact force over the step (velocity-Verlet
        # identity: dKE = -mean(f)·dx), so a sticking contact closes exactly
        ledger.e_clamp += float(
            (kt * (0.5 * (tang_old + tang) * v_t).sum(axis=1) * dt
             - 0.5 * kt * ((tang * tang).sum(axis=1)
                           - (tang_old * tang_old).sum(axis=1))).sum())
    else:
        tang = np.zeros_like(tang)
        f_t = np.zeros_like(v_rel)

    f_total = f_n[:, None] * nvec + f_t
    np.add.at(force, iu, -f_total)
    np.add.at(force, ju, f_total)
    t_i = np.cross(-nvec * r, f_t)
    np.add.at(torque, iu, t_i)
    np.add.at(torque, ju, t_i)

    if mat.mu_rolling > 0.0:
        spin = omg[ju] - omg[iu]
        roll_old = roll
        roll_pre = roll_old + spin * dt
        rnorm = np.sqrt((roll_pre * roll_pre).sum(axis=1))
        cap_r = mat.mu_rolling * r * f_n / kr
        scale_r = np.where(rnorm > cap_r, cap_r / np.maximum(rnorm, 1e-30), 1.0)
        roll = roll_pre * scale_r[:, None]
        torque_r = -kr * roll
        np.add.at(torque, iu, -torque_r)
        np.add.at(torque, ju, torque_r)
        ledger.e_clamp += float(
            (kr * (0.5 * (roll_old + roll) * spin).sum(axis=1) * dt
             - 0.5 * kr * ((roll * roll).sum(axis=1)
                           - (roll_old * roll_old).sum(axis=1))).sum())
    else:
        roll = np.zeros_like(roll)

    book.store(keys, tang, roll)
    ledger.p_damping += float((mat.gamma_pair * v_n * v_n * (f_n > 0.0)).sum())
    ledger.pe_normal += float(0.5 * mat.kn * (overlap * overlap).sum())
    ledger.pe_tangential += float(0.5 * kt * (tang * tang).sum())
    ledger.pe_rolling += float(0.5 * kr * (roll * roll).sum())
    return force, torque


def _wall_contacts(pos: np.ndarray, geom: HopperGeometry, radius: float,
                   *, wall_top: float = math.inf,
                   floor_z: Optional[float] = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Deepest wall contact per particle: (penetration, inward unit normal).

    Positive penetration means overlap with the wall.  Normals point **into the
    material** (the wall pushes the particle along +n).  For the conical part
    the normal is ``(-cos(alpha) e_r + sin(alpha) e_z)``, for the barrel
    ``-e_r`` and for the flat plate ``+e_z``.
    """
    n = len(pos)
    pen = np.zeros(n)
    nrm = np.zeros((n, 3))
    if n == 0:
        return pen, nrm
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    rho = np.hypot(x, y)
    inv = 1.0 / np.maximum(rho, 1e-30)
    er = np.stack([x * inv, y * inv, np.zeros(n)], axis=1)

    # --- lateral wall (barrel + cone) -----------------------------------
    if geom.kind == "hopper":
        in_cone = z <= geom.z_shoulder
        width = np.where(
            in_cone,
            (geom.orifice_radius + np.maximum(z, 0.0) * geom.tan_alpha - rho)
            * geom.cos_alpha,
            geom.r_bin - rho,
        )
        normal_lat = -geom.cos_alpha * er
        normal_lat[:, 2] = geom.sin_alpha
        normal_lat[~in_cone] = -er[~in_cone]
    else:
        width = geom.r_bin - rho
        normal_lat = -er
    pen_lat = radius - width
    pen_lat = np.where(z <= wall_top, pen_lat, 0.0)

    # --- bottom plate (annulus outside the orifice) ----------------------
    # ``>=`` so that the axis (rho = 0) is still supported when the
    # geometry has no orifice (closed cylinder); with an orifice the
    # axis is inside the aperture anyway.
    on_plate = rho >= geom.orifice_radius - 1e-12
    pen_plate = np.where(on_plate & (z >= 0.0), radius - z, 0.0)
    pen_plate = np.where(z < 0.0, 0.0, pen_plate)

    use_plate = pen_plate > pen_lat
    pen = np.where(use_plate, pen_plate, pen_lat)
    nrm[~use_plate] = normal_lat[~use_plate]
    nrm[use_plate, 2] = 1.0
    if floor_z is not None:
        # frictional catch floor: the poured-heap face piles the material on it
        pen_floor = np.where(z >= floor_z, radius - (z - floor_z), 0.0)
        use_floor = pen_floor > pen
        pen = np.where(use_floor, pen_floor, pen)
        nrm[use_floor] = np.array([0.0, 0.0, 1.0])
    pen = np.maximum(pen, 0.0)
    return pen, nrm


def _wall_forces(pos: np.ndarray, vel: np.ndarray, omg: np.ndarray,
                 gid: np.ndarray, wall_tang: np.ndarray,
                 geom: HopperGeometry, mat: GranularParams, mu_wall: float,
                 mu_rolling_wall: float, dt: float, ledger: _Ledger,
                 *, wall_top: float = math.inf,
                 floor_z: Optional[float] = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wall contacts.  Returns (force, torque, normal force per particle, mask).

    ``wall_tang`` is the per-call tangential displacement state (n x 3) which
    is updated in place; it is indexed like ``pos``.
    """
    n = len(pos)
    pen, nrm = _wall_contacts(pos, geom, mat.radius, wall_top=wall_top,
                              floor_z=floor_z)
    mask = pen > 0.0
    force = np.zeros((n, 3))
    torque = np.zeros((n, 3))
    # A particle that leaves the wall carries away the tangential spring
    # energy it had stored; that energy must be booked as dissipated,
    # otherwise the ledger silently loses it (and the state would be reused
    # stale if the particle touched the wall again later).
    if n:
        # ``wall_tang`` is indexed by *global* particle id (``gid``), so the
        # state survives when discharged particles are dropped from the arrays.
        tg = wall_tang[gid]
        dropped = ~mask & (np.abs(tg).sum(axis=1) > 0.0)
        if dropped.any():
            ledger.e_clamp += float(0.5 * mat.kt * (tg[dropped] ** 2).sum())
            wall_tang[gid[dropped]] = 0.0
    if not mask.any():
        return force, torque, pen, mask, nrm

    idx = np.nonzero(mask)[0]
    nn = nrm[idx]
    v = vel[idx]
    v_n = (v * nn).sum(axis=1)
    f_n = np.maximum(mat.kn * pen[idx] - mat.gamma_wall * v_n, 0.0)

    if mu_wall > 0.0:
        surf = np.cross(omg[idx], -nn * mat.radius)
        v_t = (v - v_n[:, None] * nn) + surf
        t_old = wall_tang[gid[idx]]
        t_pre = t_old + v_t * dt
        tnorm = np.sqrt((t_pre * t_pre).sum(axis=1))
        cap = mu_wall * f_n / mat.kt
        scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
        t = t_pre * scale[:, None]
        wall_tang[gid[idx]] = t
        f_t = -mat.kt * t
        ledger.e_clamp += float(
            (mat.kt * (0.5 * (t_old + t) * v_t).sum(axis=1) * dt
             - 0.5 * mat.kt * ((t * t).sum(axis=1)
                               - (t_old * t_old).sum(axis=1))).sum())
    else:
        wall_tang[gid[idx]] = 0.0
        f_t = np.zeros((len(idx), 3))

    f_wall = f_n[:, None] * nn + f_t
    force[idx] = f_wall
    torque[idx] = np.cross(-nn * mat.radius, f_t)

    if mu_rolling_wall > 0.0:
        spin = omg[idx]
        rn = np.sqrt((spin * spin).sum(axis=1))
        cap_r = mu_rolling_wall * mat.radius * f_n / mat.kr
        torch = -mat.kr * np.where(rn > cap_r, cap_r / np.maximum(rn, 1e-30), 1.0)[:, None] * spin
        torque[idx] += torch

    ledger.p_damping += float((mat.gamma_wall * v_n * v_n * (f_n > 0.0)).sum())
    ledger.pe_normal += float(0.5 * mat.kn * (pen[idx] ** 2).sum())
    ledger.pe_tangential += float(0.5 * mat.kt * (wall_tang[gid[idx]] ** 2).sum())
    return force, torque, f_n, mask, nrm


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def simulate(case: HopperCase, *, return_positions: bool = False) -> dict:
    """Integrate ``case`` and return the measured observables.

    Modes
    -----
    ``discharge`` : conical hopper drains; cumulative discharged mass is
        sampled at equally spaced window edges; per-window flow rates and the
        steady mean +/- 1 sigma are returned.
    ``static``    : closed straight cylinder; after settling the wall-normal
        force is accumulated per horizontal slab and returned as a stress
        profile (Pa) together with the bottom-plate pressure.
    ``heap``      : cylinder wall lifted, column collapses on a frictional
        floor; final positions are returned for the repose-angle measurement.

    Returns
    -------
    dict with the measured quantities, the mass/energy ledger and metadata.
    """
    t_start = time.time()
    geom = case.geometry
    mat = case.params
    r = mat.radius
    mass = mat.mass
    dt = case.timestep

    pos = case.initial_packing
    n0 = len(pos)
    vel = np.zeros((n0, 3))
    omg = np.zeros((n0, 3))
    alive = np.ones(n0, dtype=bool)
    wall_tang = np.zeros((n0, 3))
    book = _ContactBook()
    ledger = _Ledger()
    gid = np.arange(n0, dtype=np.int64)

    exit_z = -2.0 * r

    # --- protocol timing -------------------------------------------------
    if case.mode == "discharge":
        t_settle = case.settle_time
        t_total = t_settle + case.n_windows * case.window
        win_edges = t_settle + case.window * np.arange(case.n_windows + 1)
        win_steps = [int(round(e / dt)) for e in win_edges]
        t_sample_from, t_sample_to = 0.0, 0.0
    elif case.mode == "static":
        t_settle = case.settle_time
        t_total = t_settle + case.sample_time
        win_steps = []
        t_sample_from, t_sample_to = t_settle, t_total
    elif case.mode == "heap":
        t_total = (case.wall_hold_time + case.wall_lift_time
                   + case.free_settle_time)
        win_steps = []
        t_sample_from, t_sample_to = 0.0, 0.0
    else:  # pour: drain the hopper onto the catch floor, then relax
        t_total = case.settle_time + case.free_settle_time
        win_steps = []
        t_sample_from, t_sample_to = 0.0, 0.0
    n_steps = int(math.ceil(t_total / dt))

    # slab bookkeeping for the wall-stress profile
    n_slabs = max(int(math.ceil(geom.z_top / case.slab_height)), 1)
    wall_force_sum = np.zeros(n_slabs)
    wall_area_sum = np.zeros(n_slabs)
    wall_sample_count = 0
    n_wall_samples = 0
    plate_pressure_sum = 0.0
    n_plate_samples = 0

    win_mass = [0.0]
    win_ptr = 0
    discharged_mass = 0.0
    discharge_times: list[float] = []
    discharge_masses: list[float] = []

    # force buffers (velocity Verlet keeps the previous force)
    idx_a = np.nonzero(alive)[0]
    pa, va, oa = pos[idx_a], vel[idx_a], omg[idx_a]
    fa, ta, _, _, _ = _wall_forces(pa, va, oa, gid[idx_a], wall_tang,
                                geom, mat, case.mu_wall, case.mu_rolling_wall,
                                dt, ledger, wall_top=math.inf,
                                floor_z=case.floor_z)
    fb, tb = _pair_forces(pa, va, oa, gid[idx_a], n0, book, mat, dt, ledger)
    f_init = fa + fb
    f_init[:, 2] -= mass * mat.gravity
    t_init = ta + tb
    # full-length force buffers so that the velocity-Verlet kick survives
    # particles leaving the domain (the active set shrinks during discharge)
    f_prev_full = np.zeros((n0, 3))
    t_prev_full = np.zeros((n0, 3))
    f_prev_full[idx_a] = f_init
    t_prev_full[idx_a] = t_init

    # --- measured bulk density from the barrel slice of the initial packing ---
    z_lo = geom.z_shoulder + 0.02
    z_hi = geom.z_top - r
    if z_hi - z_lo > 2.0 * r:
        in_slice = (pos[:, 2] >= z_lo) & (pos[:, 2] <= z_hi)
        n_slice = int(in_slice.sum())
        slice_volume = math.pi * geom.r_bin ** 2 * (z_hi - z_lo)
        packing_fraction = n_slice * (4.0 / 3.0 * math.pi * r ** 3) / slice_volume
        bulk_density = packing_fraction * mat.density
    else:  # pragma: no cover - very shallow inventories
        packing_fraction = mass * n0 / geom.volume / mat.density
        bulk_density = mass * n0 / geom.volume

    # initial mechanical energy of the whole inventory (reference for the ledger)
    e_scale = abs(mass * mat.gravity * float(pos[:, 2].sum())) + 1e-30
    e_removed = 0.0

    for step in range(n_steps):
        t_now = step * dt

        # wall position: the heap face lifts its container wall
        if case.mode == "heap":
            if t_now < case.wall_hold_time:
                wall_top = math.inf
            elif t_now < case.wall_hold_time + case.wall_lift_time:
                frac = (t_now - case.wall_hold_time) / case.wall_lift_time
                wall_top = geom.z_top * (1.0 - frac)
            else:
                wall_top = -1.0
        else:
            wall_top = math.inf

        # kick (half) + drift
        idx_a = np.nonzero(alive)[0]
        pa, va, oa = pos[idx_a], vel[idx_a], omg[idx_a]
        gid_a = gid[idx_a]
        va = va + f_prev_full[idx_a] * (0.5 * dt / mass)
        oa = oa + t_prev_full[idx_a] * (0.5 * dt / mat.inertia)
        pa = pa + va * dt

        # energy terms are rebuilt from scratch at the new configuration
        ledger.begin_step()
        fw, tw, f_n_wall, mask_wall, n_wall = _wall_forces(
            pa, va, oa, gid_a, wall_tang, geom, mat, case.mu_wall,
            case.mu_rolling_wall, dt, ledger, wall_top=wall_top,
            floor_z=case.floor_z)
        fp, tp = _pair_forces(pa, va, oa, gid_a, n0, book, mat, dt, ledger)
        f = fw + fp
        f[:, 2] -= mass * mat.gravity
        tq = tw + tp
        ledger.end_step(dt, first=(step == 0))

        # wall-stress sampling (static mode)
        if t_sample_from <= t_now < t_sample_to and mask_wall.any():
            # ``f_n_wall`` and ``n_wall[mask_wall]`` are per-contact; the
            # lateral area of a barrel slab is 2 pi R dz per particle-step,
            # so summing over particles accumulates exactly the sampled wall
            # area (area and force are both contact sums).
            sub_z = pa[mask_wall, 2]
            n_sub = n_wall[mask_wall]
            # only *lateral* wall contacts carry the Janssen wall stress; the
            # flat bottom plate supports the column weight and is booked
            # separately as ``bottom_pressure_pa`` (mixing it in would put the
            # whole column weight into the lowest slab).
            lateral = np.abs(n_sub[:, 2]) < 0.9
            slab = np.clip((sub_z[lateral] / case.slab_height).astype(np.int64),
                           0, n_slabs - 1)
            # time-averaged wall pressure per slab: sum the contact normal
            # forces of the slab at this instant and divide by the *geometric*
            # wall area 2 pi R dz of that slab (slab counter below).
            wall_force_sum += np.bincount(slab, weights=f_n_wall[lateral],
                                          minlength=n_slabs)
            wall_sample_count += 1
            # bottom-plate pressure: normal force from the plate region
            plate = n_sub[:, 2] > 0.999
            if plate.any():
                plate_pressure_sum += float(f_n_wall[plate].sum())
                n_plate_samples += 1

        # second half kick
        va = va + f * (0.5 * dt / mass)
        oa = oa + tq * (0.5 * dt / mat.inertia)

        pos[idx_a] = pa
        vel[idx_a] = va
        omg[idx_a] = oa
        f_prev_full[idx_a] = f
        t_prev_full[idx_a] = tq

        # energy ledger (mechanical energy of the alive set)
        e_mech = (0.5 * mass * (va * va).sum()
                  + 0.5 * mat.inertia * (oa * oa).sum()
                  + mass * mat.gravity * pa[:, 2].sum()
                  + ledger.pe_normal + ledger.pe_tangential
                  + ledger.pe_rolling)
        # discharge bookkeeping
        if case.mode == "discharge" and case.floor_z is None:
            below = alive & (pos[:, 2] < exit_z)
            if below.any():
                idx_b = np.nonzero(below)[0]
                cnt = len(idx_b)
                discharged_mass += mass * cnt
                # energy leaving with the discharged particles (kept in the
                # ledger so that the residual stays meaningful)
                vb, ob, pb = vel[idx_b], omg[idx_b], pos[idx_b]
                e_removed += float(
                    (0.5 * mass * (vb * vb).sum(axis=1)
                     + 0.5 * mat.inertia * (ob * ob).sum(axis=1)
                     + mass * mat.gravity * pb[:, 2]).sum())
                alive[below] = False
                discharge_times.append(t_now)
                discharge_masses.append(discharged_mass)
            if win_ptr < len(win_steps) and step >= win_steps[win_ptr]:
                win_mass.append(discharged_mass)
                win_ptr += 1

    # ---- post-processing -------------------------------------------------
    idx_a = np.nonzero(alive)[0]
    pa, va, oa = pos[idx_a], vel[idx_a], omg[idx_a]
    e_mech = (0.5 * mass * (va * va).sum()
              + 0.5 * mat.inertia * (oa * oa).sum()
              + mass * mat.gravity * pa[:, 2].sum()
              + ledger.pe_normal + ledger.pe_tangential + ledger.pe_rolling)
    mass_domain = mass * float(alive.sum())
    mass_residual = abs(mass_domain + discharged_mass - mass * n0) / (mass * n0)
    energy_residual = abs(e_mech + ledger.dissipated + e_removed - e_scale) / e_scale

    out: dict = {
        "label": case.label,
        "mode": case.mode,
        "n_initial": int(n0),
        "n_alive": int(alive.sum()),
        "n_discharged": int(n0 - alive.sum()),
        "particle_mass_kg": mass,
        "initial_mass_kg": mass * n0,
        "initial_bulk_density_kg_m3": bulk_density,
        "packing_fraction": packing_fraction,
        "geometric_bulk_density_kg_m3": mass * n0 / geom.volume,
        "energy_removed_j": e_removed,
        "dt_s": dt,
        "steps": n_steps,
        "simulated_time_s": n_steps * dt,
        "wall_clock_s": time.time() - t_start,
        "mass_ledger_residual": mass_residual,
        "energy_ledger_residual": energy_residual,
        "energy_initial_j": e_scale,
        "energy_mechanical_final_j": e_mech,
        "energy_dissipated_j": ledger.dissipated,
        "case": case.describe(),
    }

    if case.mode == "discharge":
        rates = []
        for i in range(case.n_windows):
            dm = win_mass[i + 1] - win_mass[i]
            rates.append(dm / case.window)
        rates = np.asarray(rates)
        out.update({
            "window_mass_kg": win_mass[1:],
            "flow_rate_windows_kg_s": [float(x) for x in rates],
            "flow_rate_kg_s": float(rates.mean()),
            "flow_rate_std_kg_s": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
            "discharge_time_s": discharge_times,
            "discharge_mass_kg": discharge_masses,
            "bronze_window_start_s": float(win_edges[0]),
            "window_s": case.window,
        })
    elif case.mode == "static":
        z_center = (np.arange(n_slabs) + 0.5) * case.slab_height
        slab_area = 2.0 * math.pi * geom.r_bin * case.slab_height
        n_samp = max(int(wall_sample_count), 1)
        sigma = wall_force_sum / (n_samp * slab_area)
        wall_area_sum[:] = n_samp * slab_area
        base_area = math.pi * geom.r_bin ** 2
        out.update({
            "slab_height_m": case.slab_height,
            "slab_center_z_m": [float(z) for z in z_center],
            "wall_stress_pa": [float(s) for s in sigma],
            "wall_force_sum_n": [float(s) for s in wall_force_sum],
            "wall_area_sum_m2": [float(s) for s in wall_area_sum],
            "bottom_pressure_pa": (plate_pressure_sum / (n_plate_samples * base_area)
                                   if n_plate_samples else float("nan")),
            "n_plate_samples": n_plate_samples,
            "free_surface_height_m": float(np.percentile(pa[:, 2], 99.5)) if len(pa) else geom.z_top,
            "initial_free_surface_height_m": geom.z_top,
        })
    if return_positions or case.mode == "pour":
        out["positions"] = pos
        out["alive"] = alive
    return out