"""Shared DEM engine for the three-benchmark joint calibration (P284W3-N08).

The engine is deliberately small and dependency-light: velocity-Verlet in
float64 NumPy, particle-particle contacts delegated to the verified vectorised
kernel :func:`tensordem.granular3d_core.contact_forces`, and one plane-wall
contact law written here because the wall law of
:mod:`tensordem.shear_cell` is bound to its frozen ``ShearCellCase`` dataclass
and cannot carry a *calibrated* tangential stiffness.

Everything the calibration node needs to share between the three benchmarks
lives here: the parameter vector, the grain material, the wall law, the elastic
brittle bonds used by the uniaxial test, and the integrator.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from ..granular3d_core import ContactState, Material3D, build_pairs, contact_forces

__all__ = [
    "PARAMETER_NAMES",
    "PARAMETER_BOUNDS",
    "ContactParams",
    "GrainMaterial",
    "WallState",
    "BondSet",
    "make_bonds",
    "plane_forces",
    "integrate",
    "hash_positions",
    "moving_average",
    "bond_statistics",
]

PARAMETER_NAMES = ("kn", "kt", "mu", "restitution", "sigma_bond")

#: Calibration box (SI).  ``kn`` is a soft-sphere stiffness: the overlap at the
#: base of a 20-layer pile stays below ~2 % of the grain radius, which keeps the
#: explicit integrator at 20 steps per contact period in all three benchmarks.
PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "kn": (400.0, 2400.0),
    "kt": (150.0, 1600.0),
    "mu": (0.15, 0.75),
    "restitution": (0.10, 0.80),
    "sigma_bond": (1.0e3, 2.0e4),
}


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContactParams:
    """The five calibrated contact parameters (SI)."""

    kn: float = 900.0            # normal stiffness [N/m]
    kt: float = 450.0            # tangential stiffness [N/m]
    mu: float = 0.45             # grain-grain sliding friction [-]
    restitution: float = 0.30    # coefficient of restitution [-]
    sigma_bond: float = 6.0e3    # tensile bond strength [Pa]

    def __post_init__(self) -> None:
        if self.kn <= 0.0 or self.kt <= 0.0:
            raise ValueError("kn and kt must be positive")
        if not 0.0 <= self.mu <= 1.5:
            raise ValueError("mu must lie in [0, 1.5]")
        if not 0.0 < self.restitution < 1.0:
            raise ValueError("restitution must lie in (0, 1)")
        if self.sigma_bond < 0.0:
            raise ValueError("sigma_bond must be non-negative")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in PARAMETER_NAMES}

    @classmethod
    def from_mapping(cls, values) -> "ContactParams":
        missing = [n for n in PARAMETER_NAMES if n not in values]
        if missing:
            raise KeyError(f"missing parameters: {missing}")
        return cls(**{n: float(values[n]) for n in PARAMETER_NAMES})

    @property
    def kt_ratio(self) -> float:
        return self.kt / self.kn


@dataclass(frozen=True)
class GrainMaterial:
    """Grain-scale constants that are *not* calibrated (fixed by the class)."""

    radius: float = 0.004        # m
    density: float = 2500.0      # kg/m3
    mu_rolling: float = 0.05     # rolling-resistance coefficient
    gravity: float = 9.81        # m/s2

    @property
    def mass(self) -> float:
        return self.density * 4.0 / 3.0 * math.pi * self.radius ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    def core(self, params: ContactParams) -> Material3D:
        """Project onto the verified ``granular3d_core`` material."""
        return Material3D(
            radius=self.radius,
            density=self.density,
            mu=params.mu,
            mu_rolling=self.mu_rolling,
            restitution=params.restitution,
            kn=params.kn,
            kt_ratio=params.kt_ratio,
            gravity=self.gravity,
        )


def hash_positions(pos: np.ndarray) -> str:
    """SHA-256 of the final positions (bit-level reproducibility witness)."""
    return hashlib.sha256(np.ascontiguousarray(pos, dtype=np.float64).tobytes()).hexdigest()


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge normalisation (same length as ``x``).

    Edge samples are averaged over the samples that exist, so a constant signal
    stays exactly constant and the curve endpoints are not biased towards the
    first/last value (a plain zero-padded convolution would be).
    """
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or x.size < window:
        return x.copy()
    kernel = np.ones(window)
    num = np.convolve(x, kernel, mode="same")
    counts = np.convolve(np.ones_like(x), kernel, mode="same")
    return num / counts


# ---------------------------------------------------------------------------
# plane wall contact
# ---------------------------------------------------------------------------
class WallState:
    """Per-particle tangential/rolling spring state for one wall."""

    __slots__ = ("tang", "roll")

    def __init__(self) -> None:
        self.tang: dict[int, np.ndarray] = {}
        self.roll: dict[int, np.ndarray] = {}


def plane_forces(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
                 state: WallState, *, normal: np.ndarray, point: np.ndarray,
                 surf_vel, radius: float, kn: float, kt: float, kr: float,
                 mu: float, mu_roll: float, gamma_wall: float,
                 dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contact force/torque of one infinite plane on every particle.

    ``normal`` points into the particle domain, so a particle touches the plane
    when ``(pos - point) @ normal < radius``.  Returns ``(force, torque, f_n)``
    with ``f_n >= 0`` so callers can integrate the wall load.
    """
    n_part = len(pos)
    force = np.zeros((n_part, 3), dtype=np.float64)
    torque = np.zeros((n_part, 3), dtype=np.float64)
    f_norm = np.zeros(n_part, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    d = (pos - np.asarray(point, dtype=np.float64)) @ normal
    active = d < radius
    if not active.any():
        state.tang.clear()
        state.roll.clear()
        return force, torque, f_norm
    idx = np.nonzero(active)[0]
    pen = radius - d[idx]
    if np.ndim(surf_vel) == 2:
        sv = np.asarray(surf_vel, dtype=np.float64)[idx]
    else:
        sv = np.broadcast_to(np.asarray(surf_vel, dtype=np.float64), (n_part, 3))[idx]
    v_rel = vel[idx] - sv
    v_n = v_rel @ normal
    f_n = np.maximum(kn * pen - gamma_wall * v_n, 0.0)
    f_norm[idx] = f_n
    f = f_n[:, None] * normal[None, :]
    if mu > 0.0:
        v_t = v_rel - v_n[:, None] * normal[None, :]
        tang = np.array([state.tang.get(int(i), np.zeros(3)) for i in idx], dtype=np.float64)
        tang = tang + v_t * dt
        tnorm = np.linalg.norm(tang, axis=1)
        cap = mu * f_n / kt
        scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
        tang = tang * scale[:, None]
        for k, i in enumerate(idx):
            state.tang[int(i)] = tang[k]
        f_t = -kt * tang
        f = f + f_t
        torque[idx] += np.cross(np.broadcast_to(-normal * radius, f_t.shape), f_t)
    else:
        state.tang.clear()
    if mu_roll > 0.0:
        roll = np.array([state.roll.get(int(i), np.zeros(3)) for i in idx], dtype=np.float64)
        roll = roll + omega[idx] * dt
        rnorm = np.linalg.norm(roll, axis=1)
        cap_r = mu_roll * radius * f_n / kr
        scale_r = np.where(rnorm > cap_r, cap_r / np.maximum(rnorm, 1e-30), 1.0)
        roll = roll * scale_r[:, None]
        for k, i in enumerate(idx):
            state.roll[int(i)] = roll[k]
        torque[idx] += -kr * roll
    else:
        state.roll.clear()
    force[idx] = f
    return force, torque, f_norm


# ---------------------------------------------------------------------------
# elastic-brittle bonds (uniaxial benchmark only)
# ---------------------------------------------------------------------------
@dataclass
class BondSet:
    """Bonds between initially touching grain pairs, elastic-brittle."""

    i: np.ndarray
    j: np.ndarray
    tang: np.ndarray
    intact: np.ndarray
    tensile_strength: np.ndarray   # N per bond (sigma_bond * pi r^2, scattered)
    shear_strength: np.ndarray     # N per bond

    def __len__(self) -> int:
        return int(self.i.size)

    @property
    def n_intact(self) -> int:
        return int(self.intact.sum())


def make_bonds(pos: np.ndarray, radius: float, params: ContactParams,
               shear_ratio: float = 0.5, tol: float = 1.05,
               scatter: float = 0.35, seed: int = 0) -> BondSet:
    """Bond every pair closer than ``tol * 2r`` in the initial packing.

    Bond strengths are scattered log-normally around ``sigma_bond`` (geometric
    standard deviation ``scatter``, clipped to [0.25, 2.5]) because a perfectly
    uniform bond lattice under uniaxial compression stays homogeneous and never
    nucleates the tensile/shear micro-cracks that give a real UCS peak -- a DEM
    specimen needs that structural disorder to fail.
    """
    iu, ju = build_pairs(pos, radius * tol)
    area = math.pi * radius ** 2
    base = params.sigma_bond * area
    rng = np.random.default_rng(seed)
    if len(iu) and scatter > 0.0:
        factor = np.clip(rng.lognormal(mean=0.0, sigma=scatter, size=len(iu)), 0.25, 2.5)
    else:
        factor = np.ones(len(iu))
    return BondSet(i=iu.copy(), j=ju.copy(), tang=np.zeros((len(iu), 3)),
                   intact=np.ones(len(iu), dtype=bool),
                   tensile_strength=base * factor,
                   shear_strength=shear_ratio * base * factor)


def _bond_keys(bonds: BondSet, active: np.ndarray, n: int) -> np.ndarray:
    gap = int(n)
    return np.minimum(bonds.i[active], bonds.j[active]) * gap + \
        np.maximum(bonds.i[active], bonds.j[active])


def bond_forces(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
                bonds: BondSet, params: ContactParams, radius: float,
                dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Elastic normal + tangential + rolling bond forces with brittle failure.

    A bond carries tension (unlike the granular kernel, which clamps
    ``F_n >= 0``).  It breaks when the normal force reaches the tensile strength
    (``sigma_bond * pi r^2``) or the tangential force reaches the shear
    strength; afterwards the pair is handled by the frictional granular law.
    """
    n = len(pos)
    force = np.zeros((n, 3), dtype=np.float64)
    torque = np.zeros((n, 3), dtype=np.float64)
    active = bonds.intact
    if not active.any():
        return force, torque
    bi, bj = bonds.i[active], bonds.j[active]
    dvec = pos[bj] - pos[bi]
    dist = np.maximum(np.linalg.norm(dvec, axis=1), 1e-12)
    nvec = dvec / dist[:, None]
    delta = 2.0 * radius - dist
    f_n = params.kn * delta                      # negative = tension
    v_rel = vel[bj] - vel[bi]
    v_n = (v_rel * nvec).sum(axis=1)
    v_t = v_rel - v_n[:, None] * nvec
    tang = bonds.tang[active] + v_t * dt
    f_t = -params.kt * tang
    # An elastic rolling spring on a bonded pair has an unbounded torque and a
    # very stiff rotational mode (k_t r^2 / I), which destabilises the explicit
    # integrator; the bond therefore restrains relative *translation* only and
    # rolling resistance appears after breakage through the granular kernel.
    torque_r = np.zeros_like(f_t)

    broken = (f_n < -bonds.tensile_strength[active]) | \
             (np.linalg.norm(f_t, axis=1) > bonds.shear_strength[active])
    f_total = f_n[:, None] * nvec + f_t
    np.add.at(force, bi, -f_total)
    np.add.at(force, bj, f_total)
    t_i = np.cross(-nvec * radius, f_t)
    np.add.at(torque, bi, t_i + torque_r)
    np.add.at(torque, bj, t_i - torque_r)
    bonds.tang[active] = tang
    if broken.any():
        idx = np.nonzero(active)[0][broken]
        bonds.intact[idx] = False
        bonds.tang[idx] = 0.0
    return force, torque


def bond_statistics(bonds: BondSet) -> dict:
    return {
        "n_bonds": int(len(bonds)),
        "n_intact": int(bonds.n_intact),
        "intact_fraction": float(bonds.n_intact / max(len(bonds), 1)),
        "tensile_strength_N": float(np.mean(bonds.tensile_strength)),
        "shear_strength_N": float(np.mean(bonds.shear_strength)),
    }


# ---------------------------------------------------------------------------
# integrator
# ---------------------------------------------------------------------------
def integrate(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
              state: ContactState, floor_state: WallState, core: Material3D,
              mat: GrainMaterial, params: ContactParams, dt: float, n_steps: int,
              floor_mu: float, floor_point: np.ndarray, normal: np.ndarray,
              surf_vel, side_walls: list | None = None,
              platen: dict | None = None, bonds: BondSet | None = None,
              on_step=None, walls: dict | None = None) -> int:
    """Velocity-Verlet with gravity, particle contacts and optional walls.

    ``side_walls`` is a list of ``(normal, point, mu)`` planes; ``platen`` is a
    dict with ``normal``, ``point``, ``velocity`` (platen translation), ``mu``,
    optional free vertical DOF (``free_vertical``, ``vz``, ``mass``, ``load``,
    ``damp``) and the outputs ``force_on_platen`` / ``f_n_total``.  ``bonds``
    switches the bonded law on; ``on_step(pos, vel, omega, platen, bonds)`` is
    called after every step.
    """
    r = mat.radius
    m = mat.mass
    inertia = mat.inertia
    body_force = np.array([0.0, 0.0, -mat.gravity])
    side_walls = side_walls or []
    # The tangential/rolling spring state of every wall MUST persist across
    # calls: a fresh ``WallState`` per one-step call resets the friction spring
    # and silently turns every wall frictionless.  Callers that drive the
    # integrator one step at a time therefore pass a persistent ``walls`` dict.
    if walls is None:
        walls = {"floor": WallState(),
                 "sides": [WallState() for _ in side_walls],
                 "platen": WallState()}
    floor_state = walls["floor"]
    side_state = walls["sides"]
    platen_state = walls["platen"]
    kr = params.kt * r * r
    for _ in range(int(n_steps)):
        acc = np.tile(body_force, (len(pos), 1))
        ang_acc = np.zeros((len(pos), 3))
        iu, ju = build_pairs(pos, r)
        if bonds is not None and bonds.intact.any():
            f_bond, t_bond = bond_forces(pos, vel, omega, bonds, params, r, dt)
            acc += f_bond / m
            ang_acc += t_bond / inertia
            if len(iu) and bonds.intact.any():
                gap = len(pos)
                keys = _bond_keys(bonds, bonds.intact, gap)
                query = np.minimum(iu, ju) * gap + np.maximum(iu, ju)
                keep = ~np.isin(query, keys)
                if keep.any():
                    fp, tp = contact_forces(pos, vel, omega, iu[keep], ju[keep],
                                            state, core, dt)
                    acc += fp / m
                    ang_acc += tp / inertia
        elif len(iu):
            fp, tp = contact_forces(pos, vel, omega, iu, ju, state, core, dt)
            acc += fp / m
            ang_acc += tp / inertia
        ff, tf, _ = plane_forces(
            pos, vel, omega, floor_state, normal=normal, point=floor_point,
            surf_vel=surf_vel, radius=r, kn=params.kn, kt=params.kt, kr=kr,
            mu=floor_mu, mu_roll=mat.mu_rolling, gamma_wall=core.gamma_wall, dt=dt)
        acc += ff / m
        ang_acc += tf / inertia
        for (wn, wp, wmu), wstate in zip(side_walls, side_state):
            fw, tw, _ = plane_forces(
                pos, vel, omega, wstate, normal=wn, point=wp,
                surf_vel=np.zeros(3), radius=r, kn=params.kn, kt=params.kt, kr=kr,
                mu=wmu, mu_roll=0.0, gamma_wall=core.gamma_wall, dt=dt)
            acc += fw / m
            ang_acc += tw / inertia
        if platen is not None:
            fp2, tp2, f_n2 = plane_forces(
                pos, vel, omega, platen_state, normal=platen["normal"],
                point=platen["point"], surf_vel=platen["velocity"], radius=r,
                kn=params.kn, kt=params.kt, kr=kr, mu=platen["mu"],
                mu_roll=mat.mu_rolling, gamma_wall=core.gamma_wall, dt=dt)
            acc += fp2 / m
            ang_acc += tp2 / inertia
            platen["force_on_platen"] = -fp2.sum(axis=0)
            platen["f_n_total"] = float(f_n2.sum())
            platen["n_contacts"] = int(np.count_nonzero(f_n2 > 0.0))
            if platen.get("free_vertical"):
                mp = platen["mass"]
                a = (platen["force_on_platen"][2] - platen["load"]
                     - mp * mat.gravity - platen["damp"] * platen["vz"]) / mp
                platen["vz"] += a * dt
                platen["point"] = platen["point"] + np.array([0.0, 0.0, platen["vz"] * dt])
                platen["velocity"] = np.array([platen["velocity"][0],
                                               platen["velocity"][1], platen["vz"]])
        vel += 0.5 * acc * dt
        pos += vel * dt
        omega += 0.5 * ang_acc * dt
        if on_step is not None:
            on_step(pos, vel, omega, platen, bonds)
    return int(n_steps)