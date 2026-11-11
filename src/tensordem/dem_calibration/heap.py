"""Forward DEM model of the angle-of-repose benchmark (lifting cylinder).

Physics
-------
A hexagonal-close-packed column of monodisperse spheres is prepared inside a
frictionless-to-rough cylindrical container that sits on a rough floor.  The
container is then raised at constant speed (the ``lift_time`` protocol of
:mod:`tensordem.repose_calibration`), the column collapses and the grains come
to rest in a heap.  The *free-surface* slope of that heap, fitted over the
outer flank, is the macroscopic response ``theta_r`` that the inverse problem
of :mod:`tensordem.dem_calibration.pipeline` calibrates against.  The bulk
solid fraction of the deposit is returned as a second, weaker response
(``phi_solid``).

Contact laws are exactly the verified ones of
:mod:`tensordem.granular_column.kernel` (linear spring-dashpot normal force
with restitution-derived dashpot, Cundall-Strack tangential spring saturated at
``mu F_n``, isotropic rolling spring saturated at ``mu_r r F_n``).  The only
addition here is the *cylindrical* side wall, which :func:`wall_contact_forces`
does not provide (it handles the horizontal floor plane); it is written with
the same law and the same per-particle spring bookkeeping.

Why not reuse :func:`tensordem.repose_calibration.simulate`?
------------------------------------------------------------
That implementation is dense-pair torch (``O(N^2)`` pairs, one Python step per
contact inside some paths) and costs ~60 s per 157-grain case on the reference
host, i.e. it cannot carry a Bayesian-optimisation loop (hundreds of forward
calls).  This module is the same physics on the sparse fast kernel (sorted-key
spring bookkeeping, ``cKDTree`` neighbour search) and is ~20x cheaper per
forward evaluation, which is what makes automatic calibration affordable.

Determinism
-----------
Everything is float64 NumPy with a fixed neighbour ordering (``cKDTree`` on a
fixed point array), explicit ``np.add.at`` accumulation and a seeded
jitter, so a given :class:`HeapCase` produces a bit-identical trajectory and a
bit-identical response.  :func:`run_heap` returns a ``digest`` (SHA-256 of the
final positions) as the reproducibility witness.

All quantities are SI (m, kg, s, N, rad); angles are returned in degrees.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from ..granular3d_core import Material3D, build_pairs
from ..granular_column.kernel import (
    PairSpringState,
    WallSpringState,
    pair_contact_forces,
    wall_contact_forces,
)

__all__ = [
    "HeapCase",
    "HEAP_KEYS",
    "build_pile",
    "build_cone",
    "pluviation_layer_capacity",
    "repose_angle",
    "run_heap",
    "run_pluviation",
    "hash_positions",
]


#: The three calibrated contact parameters of the inverse problem.
HEAP_KEYS = ("mu", "restitution", "stiffness_ratio")


class HeapError(ValueError):
    """Raised when a heap case is not physically or numerically well posed."""


@dataclass(frozen=True)
class HeapCase:
    """Geometry, material and driving protocol of one repose test (SI).

    The calibrated parameters are ``mu`` (grain-grain sliding friction),
    ``restitution`` (coefficient of restitution ``e``) and
    ``stiffness_ratio`` (``k_n / k_s``); every other field is part of the
    *experiment definition* and is held fixed by the material class.
    """

    # --- calibrated contact parameters -----------------------------------
    mu: float = 0.40
    restitution: float = 0.30
    stiffness_ratio: float = 2.0          # k_n / k_s  in [1, 3]

    # --- material (fixed by the grain class) -----------------------------
    radius: float = 0.006
    density: float = 2500.0
    mu_rolling: float = 0.10
    kn: float = 800.0                     # N/m (soft contact, dt affordable)
    gravity: float = 9.81

    # --- container + protocol (fixed by the experiment) ------------------
    cylinder_radius: float | None = None  # None -> 7 r
    fill_height: float | None = None      # None -> 15 r
    packing_gap: float = 0.03             # relative lattice gap (dense)
    jitter: float = 0.02                  # seeded jitter, x radius
    mu_wall: float | None = None          # None -> mu (rough wall)
    mu_rolling_wall: float | None = None  # None -> mu_rolling
    hold_time: float = 0.03               # container down, pre-settle
    lift_time: float = 0.35               # container raised at constant speed
    settle_time: float = 0.25             # free relaxation to repose
    dt_scale: float = 1.0                 # dt = dt_scale * T_contact / 20
    #: pluviation parameters (``shape == "pluviation"``)
    n_grains: int = 400
    layer_size: int = 20
    settle_per_layer: int = 120
    inject_radius_factor: float = 2.5
    drop_height_factor: float = 2.0
    #: initial condition: ``"cone"`` (over-steep axisymmetric heap that relaxes
    #: to the repose slope) or ``"column"`` (confined column released by the
    #: retreating wall of the lifting-cylinder protocol)
    shape: str = "cone"
    #: apex slope of the initial cone [deg] (must exceed the expected repose angle)
    initial_angle_deg: float = 55.0
    #: normal-force mode of the shared kernel ("one_sided" closes the energy
    #: ledger; "reference" is the tension-cutoff variant of granular3d_core)
    contact_mode: str = "one_sided"
    seed: int = 20261110

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        for name in HEAP_KEYS:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise HeapError(f"{name} must be finite")
        if not 0.0 < self.mu:
            raise HeapError("mu must be strictly positive")
        if not 0.0 <= self.restitution < 1.0:
            raise HeapError("restitution must lie in [0, 1)")
        if not 1.0 <= self.stiffness_ratio <= 3.0:
            raise HeapError("stiffness_ratio (k_n/k_s) must lie in [1, 3]")
        for name in ("radius", "density", "kn", "gravity",
                     "lift_time", "settle_time", "dt_scale"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise HeapError(f"{name} must be finite and positive")
        if not (math.isfinite(self.hold_time) and self.hold_time >= 0.0):
            raise HeapError("hold_time must be finite and nonnegative")
        if not 0.0 <= self.jitter < 0.5:
            raise HeapError("jitter must lie in [0, 0.5)")
        if not 0.0 < self.packing_gap < 1.0:
            raise HeapError("packing_gap must lie in (0, 1)")
        if self.shape not in ("cone", "column", "pluviation"):
            raise HeapError("shape must be 'cone' or 'column'")
        if not 20.0 < self.initial_angle_deg < 89.0:
            raise HeapError("initial_angle_deg must lie in (20, 89)")
        if self.contact_mode not in ("one_sided", "reference"):
            raise HeapError("contact_mode must be 'one_sided' or 'reference'")

    # ------------------------------------------------------------------
    @property
    def wall_radius(self) -> float:
        return 7.0 * self.radius if self.cylinder_radius is None else float(self.cylinder_radius)

    @property
    def column_height(self) -> float:
        return 15.0 * self.radius if self.fill_height is None else float(self.fill_height)

    @property
    def kt_ratio(self) -> float:
        """``k_t / k_n`` from the calibrated ``k_n / k_s`` ratio."""
        return 1.0 / float(self.stiffness_ratio)

    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def material(self) -> Material3D:
        return Material3D(radius=self.radius, density=self.density, mu=float(self.mu),
                          mu_rolling=float(self.mu_rolling),
                          restitution=float(self.restitution), kn=float(self.kn),
                          kt_ratio=self.kt_ratio, gravity=self.gravity)

    @property
    def wall_mu(self) -> float:
        return float(self.mu) if self.mu_wall is None else float(self.mu_wall)

    @property
    def wall_mu_rolling(self) -> float:
        return (float(self.mu_rolling) if self.mu_rolling_wall is None
                else float(self.mu_rolling_wall))

    @property
    def timestep(self) -> float:
        period = 2.0 * math.pi * math.sqrt(0.5 * self.mass / float(self.kn))
        return self.dt_scale * period / 20.0

    def with_parameters(self, mu: float, restitution: float,
                        stiffness_ratio: float) -> "HeapCase":
        """Copy of the case with the three calibrated parameters replaced."""
        return HeapCase(
            mu=float(mu), restitution=float(restitution),
            stiffness_ratio=float(stiffness_ratio),
            radius=self.radius, density=self.density, mu_rolling=self.mu_rolling,
            kn=self.kn, gravity=self.gravity,
            cylinder_radius=self.cylinder_radius, fill_height=self.fill_height,
            packing_gap=self.packing_gap, jitter=self.jitter,
            mu_wall=self.mu_wall, mu_rolling_wall=self.mu_rolling_wall,
            hold_time=self.hold_time, lift_time=self.lift_time,
            settle_time=self.settle_time, dt_scale=self.dt_scale,
            contact_mode=self.contact_mode, shape=self.shape,
            initial_angle_deg=self.initial_angle_deg,
            n_grains=self.n_grains, layer_size=self.layer_size,
            settle_per_layer=self.settle_per_layer,
            inject_radius_factor=self.inject_radius_factor,
            drop_height_factor=self.drop_height_factor, seed=self.seed,
        )

    def signature(self) -> dict:
        """Plain-dict provenance of the experiment definition."""
        return {
            "mu": float(self.mu), "restitution": float(self.restitution),
            "stiffness_ratio": float(self.stiffness_ratio),
            "radius": self.radius, "density": self.density,
            "mu_rolling": self.mu_rolling, "kn": self.kn,
            "gravity": self.gravity,
            "wall_radius": self.wall_radius,
            "column_height": self.column_height,
            "packing_gap": self.packing_gap, "jitter": self.jitter,
            "mu_wall": self.wall_mu,
            "mu_rolling_wall": self.wall_mu_rolling,
            "hold_time": self.hold_time, "lift_time": self.lift_time,
            "settle_time": self.settle_time, "dt": self.timestep,
            "dt_scale": self.dt_scale, "contact_mode": self.contact_mode,
            "shape": self.shape, "initial_angle_deg": self.initial_angle_deg,
            "n_grains": self.n_grains, "layer_size": self.layer_size,
            "settle_per_layer": self.settle_per_layer,
            "inject_radius_factor": self.inject_radius_factor,
            "drop_height_factor": self.drop_height_factor, "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Deterministic hcp column inside the container
# ---------------------------------------------------------------------------

def build_pile(case: HeapCase) -> np.ndarray:
    """HCP column of spheres inside ``r <= wall_radius - r``, ``z in [r, H]``."""
    r = case.radius
    a = 2.0 * r * (1.0 + case.packing_gap)          # nearest-neighbour distance
    dz = a * math.sqrt(2.0 / 3.0)                   # hcp interlayer spacing
    row = a * math.sqrt(3.0) / 2.0                  # in-plane row pitch
    shift_xy = np.array([0.5 * a, a / (2.0 * math.sqrt(3.0))])
    r_fill = case.wall_radius - r * 1.02
    n_layers = max(1, int(math.floor((case.column_height - r) / dz)) + 1)
    span = int(math.ceil(r_fill / a)) + 2
    pts: list[tuple[float, float, float]] = []
    for k in range(n_layers):
        z = r * 1.001 + k * dz
        for j in range(-span, span + 1):
            y = j * row
            for i in range(-span, span + 1):
                x = i * a + (0.5 * a if (j % 2) else 0.0)
                if k % 2:
                    x += shift_xy[0]
                    y_here = y + shift_xy[1]
                else:
                    y_here = y
                if x * x + y_here * y_here <= r_fill * r_fill:
                    pts.append((x, y_here, z))
    if not pts:
        raise HeapError("packing produced no particles")
    pos = np.asarray(pts, dtype=np.float64)
    if case.jitter:
        rng = np.random.default_rng(case.seed)
        pos = pos + rng.standard_normal(pos.shape) * (case.jitter * r)
    pos[:, 2] = np.maximum(pos[:, 2], r * 1.001)
    return pos


def build_cone(case: HeapCase) -> np.ndarray:
    """Deterministic hcp filling of an over-steep axisymmetric cone on the floor.

    The lattice is filled layer by layer inside
    ``z <= r + tan(initial_angle) * (R_toe - rad)``; when the cone relaxes under
    gravity the free surface drops to the material's repose slope, which is the
    quantity the inverse problem calibrates.
    """
    r = case.radius
    a = 2.0 * r * (1.0 + case.packing_gap)
    dz = a * math.sqrt(2.0 / 3.0)
    row = a * math.sqrt(3.0) / 2.0
    slope = math.tan(math.radians(case.initial_angle_deg))
    r_toe = case.wall_radius - r * 1.02
    height = r + slope * (r_toe - 0.0)
    n_layers = max(1, int(math.floor((height - r) / dz)) + 1)
    span = int(math.ceil(r_toe / a)) + 2
    pts: list[tuple[float, float, float]] = []
    for k in range(n_layers):
        z = r * 1.001 + k * dz
        for j in range(-span, span + 1):
            y = j * row
            for i in range(-span, span + 1):
                x = i * a + (0.5 * a if (j % 2) else 0.0)
                yy = y + (a / (2.0 * math.sqrt(3.0)) if k % 2 else 0.0)
                xx = x + (0.5 * a if k % 2 else 0.0)
                rad = math.hypot(xx, yy)
                if rad > r_toe or z > r + slope * (r_toe - rad):
                    continue
                pts.append((xx, yy, z))
    if not pts:
        raise HeapError("cone packing produced no particles")
    pos = np.asarray(pts, dtype=np.float64)
    if case.jitter:
        rng = np.random.default_rng(case.seed)
        pos = pos + rng.standard_normal(pos.shape) * (case.jitter * r)
    pos[:, 2] = np.maximum(pos[:, 2], r * 1.001)
    return pos


# ---------------------------------------------------------------------------
# Cylindrical side wall (the one contact the shared kernel does not provide)
# ---------------------------------------------------------------------------

def _cylinder_wall_forces(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
                          mat: Material3D, dt: float, *, radius_wall: float,
                          bottom: float, mu_wall: float, mu_rolling_wall: float,
                          tang: np.ndarray, roll: np.ndarray,
                          update_state: bool = True,
                          contact_mode: str = "one_sided") -> tuple[np.ndarray,
                                                                    np.ndarray, int]:
    """Inward radial traction from a rigid cylinder for ``z < bottom`` only.

    ``bottom`` is the height of the raised container bottom: particles above it
    are free.  The law is the same linear spring-dashpot + Cundall-Strack +
    rolling spring as :func:`tensordem.granular_column.kernel.wall_contact_forces`,
    with the plane normal replaced by the inward radial direction.  ``tang`` and
    ``roll`` are updated in place (per particle).
    """
    n = pos.shape[0]
    force = np.zeros((n, 3), dtype=np.float64)
    torque = np.zeros((n, 3), dtype=np.float64)
    if bottom <= 0.0:
        tang[:] = 0.0
        roll[:] = 0.0
        return force, torque, 0

    r = mat.radius
    rad = np.hypot(pos[:, 0], pos[:, 1])
    overlap = rad - (radius_wall - r)
    # The wall acts from the inside only: a grain whose centre is outside the
    # cylinder (rad > radius_wall) has spilled over the rim and must be free.
    # Without this test the radial traction "sucks" spilt grains back into the
    # container and pumps energy into the heap (observed as ke_released > 1e2).
    act = (overlap > 0.0) & (pos[:, 2] < bottom) & (rad < radius_wall)
    if not act.any():
        tang[:] = 0.0
        roll[:] = 0.0
        return force, torque, 0

    idx = np.nonzero(act)[0]
    nrm = -pos[idx, :2] / np.maximum(rad[idx], 1e-30)[:, None]   # inward 2-D
    n3 = np.concatenate([nrm, np.zeros((idx.size, 1))], axis=1)

    v_rel = vel[idx]
    v_n = np.einsum("ij,ij->i", v_rel, n3)
    kn, gamma = float(mat.kn), float(mat.gamma_wall)
    if contact_mode == "reference":
        f_n = np.maximum(kn * overlap[idx] - gamma * v_n, 0.0)
    elif contact_mode == "one_sided":
        f_n = kn * overlap[idx] + gamma * np.maximum(-v_n, 0.0)
    else:
        raise HeapError(f"unknown contact_mode {contact_mode!r}")

    # contact-point surface velocity (spin about the contact at -r n)
    spin = np.cross(omega[idx], -n3 * r)
    u_t = v_rel + spin
    u_t = u_t - np.einsum("ij,ij->i", u_t, n3)[:, None] * n3

    trial = tang[idx] + u_t * dt
    t_norm = np.linalg.norm(trial, axis=1)
    cap = mu_wall * f_n / float(mat.kt)
    scale = np.where(t_norm > cap, cap / np.maximum(t_norm, 1e-30), 1.0)
    trial = trial * scale[:, None]
    f_t = -float(mat.kt) * trial
    if update_state:
        tang[:] = 0.0
        tang[idx] = trial

    kr = float(mat.kr)
    if mu_rolling_wall > 0.0:
        trial_r = roll[idx] + omega[idx] * dt
        r_norm = np.linalg.norm(trial_r, axis=1)
        cap_r = mu_rolling_wall * r * f_n / kr
        scale_r = np.where(r_norm > cap_r, cap_r / np.maximum(r_norm, 1e-30), 1.0)
        trial_r = trial_r * scale_r[:, None]
        torque_r = -kr * trial_r
        if update_state:
            roll[:] = 0.0
            roll[idx] = trial_r
    else:
        torque_r = np.zeros((idx.size, 3), dtype=np.float64)

    force[idx] += f_n[:, None] * n3 + f_t
    torque[idx] += np.cross(-n3 * r, f_t) + torque_r
    return force, torque, int(idx.size)


# ---------------------------------------------------------------------------
# Repose-angle estimator (numpy port of the verified annulus/sector fit)
# ---------------------------------------------------------------------------

def repose_angle(pos: np.ndarray, radius: float, *, sectors: int = 8,
                 min_count: int = 2, mass_fraction: float = 0.95) -> dict:
    """Free-surface slope of an axisymmetric heap, in degrees.

    Same estimator as the verified :func:`tensordem.repose_calibration.repose_angle`
    (annulus of width ``radius``, sector-wise maximum elevation, least-squares
    flank fit between 15 % and 90 % of the shoulder-to-toe span with one-sided
    2-sigma clipping), re-implemented in NumPy so that the calibration loop does
    not depend on torch.  ``tests/test_dem_calibration.py`` checks it against a
    synthetic cone of known angle.

    One addition is required by this benchmark: the toe is placed at the
    ``mass_fraction`` quantile of the *radial mass distribution* instead of the
    outermost grain.  A few grains that escaped over the container rim sit at
    2-3x the pile radius and would otherwise set the flank span; the fitted
    "slope" then describes the scattered tail (measured: 0.9 deg instead of
    39 deg on the same deposit).  For a well-formed cone the radial mass grows
    like ``r^2``, so the 95 % quantile sits at 0.98 of the toe radius and this
    guard leaves the cone test unchanged.
    """
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise HeapError("positions must be an (n, 3) array")
    if len(pos) < 4:
        raise HeapError("at least 4 particles are needed")

    centre = pos[:, :2].mean(axis=0)
    rel = pos[:, :2] - centre
    rad = np.linalg.norm(rel, axis=1)
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    z = pos[:, 2]

    width = float(radius)
    r_toe = float(np.quantile(rad, mass_fraction))
    n_bins = int(math.floor(r_toe / width)) + 1
    if n_bins < 2:
        raise HeapError("heap has no measurable radial extent")
    sec = np.clip(((ang + math.pi) / (2.0 * math.pi) * sectors).astype(np.int64),
                  0, sectors - 1)
    bini = np.clip((rad / width).astype(np.int64), 0, n_bins - 1)
    bini = np.where(rad <= r_toe, bini, -1)      # drop the scattered spilt tail

    heights: list[float] = []
    centres: list[float] = []
    for b in range(n_bins):
        in_bin = bini == b
        if int(in_bin.sum()) < min_count:
            continue
        sec_heights = [float(z[in_bin & (sec == s)].max())
                       for s in range(sectors) if bool((in_bin & (sec == s)).any())]
        heights.append(float(np.mean(sec_heights)))
        centres.append((b + 0.5) * width)
    if len(heights) < 4:
        raise HeapError("not enough radial bins to fit a flank")

    h = np.asarray(heights)
    c = np.asarray(centres)
    apex = int(np.argmax(h))
    shoulder = float(c[apex])
    toe = float(r_toe)
    span = toe - shoulder
    if span <= 3.0 * width:
        raise HeapError("heap has no measurable flank")
    window = (c >= shoulder + 0.15 * span) & (c <= shoulder + 0.90 * span)
    x = c[window]
    y = h[window]
    if x.size < 3:
        raise HeapError("flank window has fewer than 3 bins")

    index = np.arange(x.size)
    slope = intercept = 0.0
    r2 = 0.0
    for _ in range(3):
        if index.size < 3:
            break
        xk, yk = x[index], y[index]
        xm, ym = xk.mean(), yk.mean()
        sxx = float(((xk - xm) ** 2).sum())
        if sxx <= 0.0:
            raise HeapError("degenerate flank window")
        slope = float(((xk - xm) * (yk - ym)).sum() / sxx)
        intercept = float(ym - slope * xm)
        resid = yk - (slope * xk + intercept)
        std = float(resid.std()) if xk.size > 2 else 0.0
        ss_tot = float(((yk - ym) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0.0 else 0.0
        keep = resid < 2.0 * std
        if int(keep.sum()) < 3 or bool(keep.all()):
            break
        index = index[keep]
    return {
        "theta_r_deg": math.degrees(math.atan(-slope)),
        "slope": slope,
        "r2": r2,
        "shoulder_radius": shoulder,
        "toe_radius": toe,
        "apex_z": float(h.max()),
        "n_bins": len(heights),
        "n_fit_bins": int(index.size),
    }


def hash_positions(pos: np.ndarray) -> str:
    """SHA-256 of the final float64 positions (bit-identical = same digest)."""
    return hashlib.sha256(np.ascontiguousarray(pos, dtype=np.float64).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# The forward run
# ---------------------------------------------------------------------------

def run_heap(case: HeapCase, *, max_steps: int | None = None,
             ke_stop: float | None = None, settle_check_every: int = 25,
             return_positions: bool = False) -> dict:
    """Run one lifting-cylinder repose test and measure the macro responses.

    Returns a dict with

    ``theta_r_deg``      free-surface slope of the final heap [deg]
    ``phi_solid``        bulk solid fraction of the deposit [-]
    ``r2``               quality of the flank fit [-]
    ``ke_ratio``         residual kinetic energy / released potential energy
    ``digest``           SHA-256 of the final positions (reproducibility witness)

    Adaptive termination
    --------------------
    The prescribed ``hold + lift + settle`` window of :class:`HeapCase` is a
    *time budget*, not a proof that the heap came to rest.  Passing
    ``ke_stop`` (e.g. ``0.02``) stops the run as soon as the residual kinetic
    energy first falls below ``ke_stop`` times the potential energy released so
    far, i.e. it terminates on the *filled* measurement criterion "heap at
    rest" instead of on an arbitrary clock.  The default ``None`` reproduces
    the historical fixed-window behaviour bit for bit.
    """
    mat = case.material
    pos = build_cone(case) if case.shape == "cone" else build_pile(case)
    n = int(pos.shape[0])
    vel = np.zeros((n, 3), dtype=np.float64)
    omega = np.zeros((n, 3), dtype=np.float64)
    mass, inertia = mat.mass, mat.inertia
    gravity_vec = np.array([0.0, 0.0, -float(case.gravity)])
    dt = case.timestep

    pair_state = PairSpringState()
    floor_state = WallSpringState(n)
    wall_tang = np.zeros((n, 3), dtype=np.float64)
    wall_roll = np.zeros((n, 3), dtype=np.float64)

    cone_mode = case.shape == "cone"
    pe0 = float(mass * case.gravity * np.sum(pos[:, 2]))
    z_start = float(pos[:, 2].max() + mat.radius)
    z_end = float(case.column_height)

    total_time = (case.hold_time + case.settle_time) if cone_mode else \
        (case.hold_time + case.lift_time + case.settle_time)
    n_steps = int(math.ceil(total_time / dt))
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))

    speed_max = 0.0
    wall_contacts = 0
    mode = case.contact_mode
    # The rest criterion may only be applied once the prescribed driving phase
    # is over: a cone has no lift (only the pre-settle hold), a column has the
    # lift, and pluviation is grown elsewhere (run_pluviation).
    settle_start = (case.hold_time if cone_mode
                    else case.hold_time + case.lift_time)
    settled = False
    steps_used = int(n_steps)

    def _bottom_height(t: float) -> float:
        if cone_mode:
            return 0.0                            # no container at all
        if t <= case.hold_time:
            return z_end + mat.radius                 # container at full height
        if t <= case.hold_time + case.lift_time:
            frac = (t - case.hold_time) / case.lift_time
            return z_end + mat.radius - frac * z_end
        return 0.0

    # Velocity-Verlet with the contact springs advanced exactly once per step
    # (predictor: forces at the old springs; corrector: forces at the new
    # positions with the springs advanced).  This is the integrator of
    # tensordem.granular_column.column3d, whose energy ledger closes to ~1e-4 of
    # the released potential energy; a plain symplectic-Euler kick was found to
    # inject energy for soft contacts under a tall column.
    for step in range(n_steps):
        bottom = _bottom_height(step * dt)
        iu, ju = build_pairs(pos, mat.radius)
        pred = pair_contact_forces(pos, vel, omega, iu, ju, pair_state, mat, dt,
                                   update_state=False, contact_mode=mode)
        pred_floor = wall_contact_forces(pos, vel, omega, mat, dt, floor_z=0.0,
                                         mu_wall=case.wall_mu,
                                         mu_rolling_wall=case.wall_mu_rolling,
                                         state=floor_state, update_state=False,
                                         contact_mode=mode)
        if cone_mode:
            pred_wall = np.zeros((n, 3))
            pred_wall_t = np.zeros((n, 3))
        else:
            pred_wall, pred_wall_t, nwc = _cylinder_wall_forces(
                pos, vel, omega, mat, dt, radius_wall=case.wall_radius, bottom=bottom,
                mu_wall=case.wall_mu, mu_rolling_wall=case.wall_mu_rolling,
                tang=wall_tang, roll=wall_roll, update_state=False, contact_mode=mode)
            wall_contacts = nwc

        acc = (pred["force"] + pred_floor["force"] + pred_wall) / mass + gravity_vec
        alpha = (pred["torque"] + pred_floor["torque"] + pred_wall_t) / inertia
        v_half = vel + 0.5 * dt * acc
        w_half = omega + 0.5 * dt * alpha
        pos = pos + v_half * dt

        cur = pair_contact_forces(pos, v_half, w_half, iu, ju, pair_state, mat, dt,
                                  update_state=True, contact_mode=mode)
        cur_floor = wall_contact_forces(pos, v_half, w_half, mat, dt, floor_z=0.0,
                                        mu_wall=case.wall_mu,
                                        mu_rolling_wall=case.wall_mu_rolling,
                                        state=floor_state, update_state=True,
                                        contact_mode=mode)
        if cone_mode:
            cur_wall = np.zeros((n, 3))
            cur_wall_t = np.zeros((n, 3))
        else:
            cur_wall, cur_wall_t, _ = _cylinder_wall_forces(
                pos, v_half, w_half, mat, dt, radius_wall=case.wall_radius, bottom=bottom,
                mu_wall=case.wall_mu, mu_rolling_wall=case.wall_mu_rolling,
                tang=wall_tang, roll=wall_roll, update_state=True, contact_mode=mode)

        acc = (cur["force"] + cur_floor["force"] + cur_wall) / mass + gravity_vec
        alpha = (cur["torque"] + cur_floor["torque"] + cur_wall_t) / inertia
        vel = v_half + 0.5 * dt * acc
        omega = w_half + 0.5 * dt * alpha

        if n:
            speed_max = max(speed_max, float(np.sqrt((vel * vel).sum(axis=1)).max()))

        if ke_stop is not None and (step + 1) % settle_check_every == 0:
            t_now = (step + 1) * dt
            if t_now >= settle_start:
                ke_now = float(0.5 * mass * np.sum(vel * vel)
                               + 0.5 * inertia * np.sum(omega * omega))
                pe_now = float(mass * case.gravity * np.sum(pos[:, 2]))
                if ke_now <= float(ke_stop) * max(pe0 - pe_now, 1e-30):
                    settled = True
                    steps_used = step + 1
                    break

    ke = float(0.5 * mass * np.sum(vel * vel) + 0.5 * inertia * np.sum(omega * omega))
    pe = float(mass * case.gravity * np.sum(pos[:, 2]))
    released = max(pe0 - pe, 1e-30)

    # bulk solid fraction: deposit volume from a cylindrical envelope fit
    rad = np.hypot(pos[:, 0], pos[:, 1])
    r_toe = float(rad.max() + mat.radius)
    height = float(pos[:, 2].max() + mat.radius)
    envelope = math.pi * r_toe ** 2 * height
    grain_volume = n * 4.0 / 3.0 * math.pi * mat.radius ** 3
    phi_solid = grain_volume / envelope if envelope > 0.0 else float("nan")

    estimate = repose_angle(pos, mat.radius)
    result = {
        "case": case.signature(),
        "n_particles": n,
        "dt": dt,
        "n_steps": int(n_steps),
        "steps_used": int(steps_used),
        "time_end": steps_used * dt,
        "settled": bool(settled),
        "ke_stop": None if ke_stop is None else float(ke_stop),
        "theta_r_deg": estimate["theta_r_deg"],
        "phi_solid": phi_solid,
        "r2": estimate["r2"],
        "n_fit_bins": estimate["n_fit_bins"],
        "shoulder_radius": estimate["shoulder_radius"],
        "toe_radius": estimate["toe_radius"],
        "apex_z": estimate["apex_z"],
        "ke_final_J": ke,
        "ke_ratio": ke / released,
        "pe_released_J": released,
        "max_speed_m_s": speed_max,
        "n_wall_contacts_final": wall_contacts,
        "digest": hash_positions(pos),
    }
    if return_positions:
        result["positions"] = pos.tolist()
    return result

# ---------------------------------------------------------------------------
# Pluviation heap: layer-by-layer deposition (the test used for real sands)
# ---------------------------------------------------------------------------

def _injection_positions(case: HeapCase, apex: tuple[float, float], z_layer: float,
                         n_particles: int, *, rng: np.random.Generator,
                         occupied: np.ndarray) -> np.ndarray:
    """Lattice positions of one rain layer inside ``inject_radius`` of the apex.

    In-plane spacing is ``2 r (1 + gap)``, so the grains of one layer do not
    overlap each other; ``jitter`` adds the seeded scatter that keeps the layer
    from being a perfect crystal.
    """
    r = case.radius
    a = 2.0 * r * (1.0 + case.packing_gap)
    radius = case.inject_radius_factor * r
    points: list[tuple[float, float, float]] = []
    span = int(math.ceil(radius / a)) + 1
    for j in range(-span, span + 1):
        for i in range(-span, span + 1):
            x = i * a + (0.5 * a if (j % 2) else 0.0)
            y = j * a * math.sqrt(3.0) / 2.0
            if x * x + y * y > radius * radius:
                continue
            points.append((apex[0] + x, apex[1] + y, z_layer))
    if not points:
        points = [(apex[0], apex[1], z_layer)]
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) > n_particles:
        order = rng.permutation(len(pts))[:n_particles]
        pts = pts[np.sort(order)]
    if case.jitter:
        pts[:, :2] += rng.standard_normal((len(pts), 2)) * (case.jitter * r)
    return pts


def pluviation_layer_capacity(case: HeapCase) -> int:
    """Number of non-overlapping lattice sites of one rain layer.

    This is the *exact* capacity of the lattice generated by
    :func:`_injection_positions` (a circular mask over a staggered square
    lattice of spacing ``2 r (1 + gap)``), i.e. the largest ``layer_size`` that
    :func:`run_pluviation` can pour in one layer at the case's injection radius.
    The hexagon-area estimate used for the historical input check over-predicts
    it for small injection radii (e.g. ``inject_radius_factor = 3.5``), which is
    how a geometry sweep used to end in a bare numpy broadcast error.
    """
    r = case.radius
    a = 2.0 * r * (1.0 + case.packing_gap)
    radius = case.inject_radius_factor * r
    span = int(math.ceil(radius / a)) + 1
    count = 0
    for j in range(-span, span + 1):
        for i in range(-span, span + 1):
            x = i * a + (0.5 * a if (j % 2) else 0.0)
            y = j * a * math.sqrt(3.0) / 2.0
            if x * x + y * y <= radius * radius:
                count += 1
    return max(1, count)


def run_pluviation(case: HeapCase, *, return_positions: bool = False) -> dict:
    """Build a heap by rain deposition and measure its repose slope.

    The heap is grown in ``n_layers`` of ``layer_size`` grains.  Each layer is
    placed on a lattice ``drop_height_factor * r`` above the current apex with
    zero velocity and then given ``settle_steps`` steps to roll down the flank
    and come to rest, which is the classical pluviation ("rain") method used to
    prepare sand specimens at a controlled density in the laboratory and the
    standard way to grow a DEM heap.

    ``n_grains`` particles are created up front and parked far from the pile
    (no contacts, no neighbour search overlap); a parked grain is *teleported*
    into its injection position when its layer is poured.  Keeping the particle
    count constant keeps the pair-spring keys of the shared kernel stable, so no
    contact history is lost at an injection.

    Returns the same response record as :func:`run_heap`.
    """
    mat = case.material
    n_total = int(case.n_grains)
    layer_size = int(case.layer_size)
    if layer_size < 1 or n_total < layer_size:
        raise HeapError("n_grains must be >= layer_size >= 1")

    r = mat.radius
    area = math.pi * (case.inject_radius_factor * r) ** 2
    spacing = 2.0 * r * (1.0 + case.packing_gap)
    per_layer_lattice = max(1, int(area / (spacing * spacing * math.sqrt(3.0) / 2.0)))
    if layer_size > per_layer_lattice:
        raise HeapError(
            f"layer_size {layer_size} exceeds the {per_layer_lattice} non-overlapping "
            "positions of one injection layer")

    # parked positions: far away on the floor, mutually isolated
    pos = np.zeros((n_total, 3), dtype=np.float64)
    park = 200.0 * r
    side = int(math.ceil(math.sqrt(n_total))) + 1
    for k in range(n_total):
        pos[k] = (park + (k % side) * 3.0 * r, park + (k // side) * 3.0 * r, r * 1.001)
    vel = np.zeros((n_total, 3), dtype=np.float64)
    omega = np.zeros((n_total, 3), dtype=np.float64)
    active = np.zeros(n_total, dtype=bool)

    mass, inertia = mat.mass, mat.inertia
    gravity_vec = np.array([0.0, 0.0, -float(case.gravity)])
    dt = case.timestep
    pair_state = PairSpringState()
    floor_state = WallSpringState(n_total)
    mode = case.contact_mode

    rng = np.random.default_rng(case.seed)
    injected = 0
    n_layers = int(math.ceil(n_total / layer_size))
    settle_steps = int(case.settle_per_layer)
    speed_max = 0.0

    def run_steps(count: int) -> None:
        nonlocal speed_max, pos, vel, omega
        for _ in range(count):
            iu, ju = build_pairs(pos, r)
            pred = pair_contact_forces(pos, vel, omega, iu, ju, pair_state, mat, dt,
                                       update_state=False, contact_mode=mode)
            pred_floor = wall_contact_forces(pos, vel, omega, mat, dt, floor_z=0.0,
                                             mu_wall=case.wall_mu,
                                             mu_rolling_wall=case.wall_mu_rolling,
                                             state=floor_state, update_state=False,
                                             contact_mode=mode)
            acc = (pred["force"] + pred_floor["force"]) / mass + gravity_vec
            alpha = (pred["torque"] + pred_floor["torque"]) / inertia
            v_half = vel + 0.5 * dt * acc
            w_half = omega + 0.5 * dt * alpha
            pos += v_half * dt
            cur = pair_contact_forces(pos, v_half, w_half, iu, ju, pair_state, mat, dt,
                                      update_state=True, contact_mode=mode)
            cur_floor = wall_contact_forces(pos, v_half, w_half, mat, dt, floor_z=0.0,
                                            mu_wall=case.wall_mu,
                                            mu_rolling_wall=case.wall_mu_rolling,
                                            state=floor_state, update_state=True,
                                            contact_mode=mode)
            acc = (cur["force"] + cur_floor["force"]) / mass + gravity_vec
            alpha = (cur["torque"] + cur_floor["torque"]) / inertia
            vel = v_half + 0.5 * dt * acc
            omega = w_half + 0.5 * dt * alpha
            speed_max = max(speed_max, float(np.sqrt((vel * vel).sum(axis=1)).max()))

    for _layer in range(n_layers):
        take = min(layer_size, n_total - injected)
        if take <= 0:
            break
        if injected == 0:
            apex = (0.0, 0.0)
            z_layer = r * (1.0 + float(case.drop_height_factor))
        else:
            z = pos[active, 2]
            top = z.max()
            near = active.copy()
            near[active] = z >= top - 2.0 * r
            apex = (float(pos[near, 0].mean()), float(pos[near, 1].mean()))
            z_layer = top + float(case.drop_height_factor) * r
        idx = np.nonzero(~active)[0]
        if idx.size < take:
            raise HeapError("not enough parked particles left for this layer")
        layer_idx = idx[:take]
        pts = _injection_positions(case, apex, z_layer, take, rng=rng,
                                   occupied=pos[active])
        # A grain injected inside an existing grain would give an O(r) overlap
        # and inject elastic energy (observed: ke_released ~ 1e28).  Lift the
        # layer until every injected grain is at least one diameter away from
        # the pile (at most ten pushes of r).
        existing = pos[active]
        if existing.shape[0]:
            from scipy.spatial import cKDTree
            tree = cKDTree(existing)
            for _ in range(20):
                dist, _ = tree.query(pts)
                if float(dist.min()) >= 2.0 * r * 1.001:
                    break
                # lift the *whole* layer: lifting single grains would create
                # new overlaps inside the layer (same failure mode)
                pts[:, 2] += 0.5 * r
        if pts.shape[0] < take:
            # A short layer used to surface as "shape mismatch: value array of
            # shape (7,3) could not be broadcast to indexing result of shape
            # (10,3)" from the assignment below; report the domain fact instead.
            raise HeapError(
                f"injection layer holds {pts.shape[0]} grains but layer_size is "
                f"{take}: reduce layer_size to <= "
                f"{pluviation_layer_capacity(case)} for inject_radius_factor = "
                f"{case.inject_radius_factor:g}")
        pos[layer_idx] = pts[:take]
        vel[layer_idx] = 0.0
        omega[layer_idx] = 0.0
        active[layer_idx] = True
        injected += take
        run_steps(settle_steps if injected > layer_size else settle_steps)

    # final relaxation
    run_steps(settle_steps)

    live = pos[active]
    ke = float(0.5 * mass * np.sum(vel[active] ** 2)
               + 0.5 * inertia * np.sum(omega[active] ** 2))
    # The parked-grain baseline makes "released potential energy" meaningless for
    # a growth method (the heap *gains* potential energy), so the residual
    # kinetic energy is normalised by the pile's own potential energy scale.
    scale = float(live.shape[0] * mass * case.gravity
                  * max(float(live[:, 2].max() + r), r))
    released = max(scale, 1e-30)
    rad = np.hypot(live[:, 0], live[:, 1])
    r_toe = float(rad.max() + r)
    height = float(live[:, 2].max() + r)
    envelope = math.pi * r_toe ** 2 * height
    grain_volume = live.shape[0] * 4.0 / 3.0 * math.pi * r ** 3
    phi_solid = grain_volume / envelope if envelope > 0.0 else float("nan")
    estimate = repose_angle(live, r)
    result = {
        "case": case.signature(),
        "n_particles": int(live.shape[0]),
        "n_layers": int(n_layers),
        "dt": dt,
        "n_steps": int(n_layers * settle_steps + settle_steps),
        "time_end": (n_layers * settle_steps + settle_steps) * dt,
        "theta_r_deg": estimate["theta_r_deg"],
        "phi_solid": phi_solid,
        "r2": estimate["r2"],
        "n_fit_bins": estimate["n_fit_bins"],
        "shoulder_radius": estimate["shoulder_radius"],
        "toe_radius": estimate["toe_radius"],
        "apex_z": estimate["apex_z"],
        "ke_final_J": ke,
        "ke_ratio": ke / released,
        "pe_released_J": released,
        "max_speed_m_s": speed_max,
        "n_wall_contacts_final": 0,
        "digest": hash_positions(live),
    }
    if return_positions:
        result["positions"] = live.tolist()
        result["speeds"] = np.sqrt((vel * vel).sum(axis=1)).tolist()
    return result
