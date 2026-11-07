"""3-D granular DEM core: spatial-hash neighbour search + contact laws.

Shared numerical kernel for the m5481 granular-flow benchmarks (Beverloo silo
discharge, direct-shear mu(I) rheology).  Deliberately a standalone,
dependency-light kernel (numpy + scipy cKDTree) so the heavy runs can be
sharded across cores without a torch dependency.

Contact laws (identical convention to ``tensordem.repose_calibration``):

* normal: linear spring-dashpot, ``F_n = max(k_n delta - gamma v_n, 0)`` with
  the dashpot ``gamma`` derived from the coefficient of restitution;
* tangential: Cundall-Strack incremental spring, saturated at ``mu |F_n|``;
* rolling: isotropic rolling spring saturated at ``mu_r R_eff |F_n|``.

All quantities are SI (m, kg, s, N, rad).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def damping_ratio(restitution: float) -> float:
    """zeta = |ln e| / sqrt(pi^2 + ln^2 e) (dashpot from restitution)."""
    e = max(min(restitution, 1.0 - 1e-12), 1e-6)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


@dataclass(frozen=True)
class Material3D:
    """Monodisperse-sphere material + contact parameters (SI)."""
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.5
    mu_rolling: float = 0.05
    restitution: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5
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
        return damping_ratio(self.restitution)

    @property
    def gamma_pair(self) -> float:
        """Pair dashpot coefficient (reduced mass m/2)."""
        return 2.0 * self.zeta * math.sqrt(self.kn * 0.5 * self.mass)

    @property
    def gamma_wall(self) -> float:
        """Rigid-counterpart dashpot (full mass)."""
        return 2.0 * self.zeta * math.sqrt(self.kn * self.mass)

    @property
    def timestep(self) -> float:
        """Resolve the stiffest linear spring by 20 steps per period."""
        period = 2.0 * math.pi * math.sqrt((0.5 * self.mass) / self.kn)
        return period / 20.0


def hcp_positions_in_cylinder(radius: float, cylinder_radius: float,
                             fill_height: float, *, packing_gap: float = 0.02,
                             jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """Deterministic HCP column inside a cylinder (same lattice as the
    torch reference solver, so the two are directly comparable)."""
    r = radius
    spacing = 2.0 * r * (1.0 + packing_gap)
    layer_dz = spacing * math.sqrt(2.0 / 3.0)
    n_layers = max(1, int(math.floor((fill_height - r) / layer_dz)) + 1)
    r_max = cylinder_radius - r * 1.02
    row_pitch = spacing * math.sqrt(3.0) / 2.0
    shift_x = 0.5 * spacing
    shift_y = spacing / (2.0 * math.sqrt(3.0))
    pts: list[tuple[float, float, float]] = []
    for layer in range(n_layers):
        z = r + layer * layer_dz
        odd = layer % 2 == 1
        span = int(math.ceil((cylinder_radius + spacing) / spacing))
        for ix in range(-span, span + 1):
            for iy in range(-span, span + 1):
                x = ix * spacing + 0.5 * spacing * (iy % 2)
                y = iy * row_pitch
                if odd:
                    x += shift_x
                    y += shift_y
                if math.hypot(x, y) <= r_max:
                    pts.append((x, y, z))
    pos = np.asarray(pts, dtype=np.float64)
    if pos.size == 0:
        raise ValueError("packing produced no particles")
    if jitter:
        rng = np.random.default_rng(seed)
        pos = pos + rng.standard_normal(pos.shape) * (jitter * r)
    pos[:, 2] = np.maximum(pos[:, 2], r * 1.001)
    return pos


def hcp_positions_in_box(radius: float, box_width: float, box_depth: float,
                         fill_height: float, *, packing_gap: float = 0.02,
                         jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """Deterministic HCP packing inside a rectangular box (shear cell)."""
    r = radius
    spacing = 2.0 * r * (1.0 + packing_gap)
    layer_dz = spacing * math.sqrt(2.0 / 3.0)
    n_layers = max(1, int(math.floor((fill_height - r) / layer_dz)) + 1)
    row_pitch = spacing * math.sqrt(3.0) / 2.0
    shift_x = 0.5 * spacing
    shift_y = spacing / (2.0 * math.sqrt(3.0))
    x_min, x_max = r + 1e-3, box_width - r - 1e-3
    y_min, y_max = r + 1e-3, box_depth - r - 1e-3
    pts: list[tuple[float, float, float]] = []
    for layer in range(n_layers):
        z = r + layer * layer_dz
        odd = layer % 2 == 1
        span_x = int(math.ceil((box_width + spacing) / spacing))
        span_y = int(math.ceil((box_depth + spacing) / spacing))
        for ix in range(-span_x, span_x + 1):
            for iy in range(-span_y, span_y + 1):
                    x = ix * spacing + 0.5 * spacing * (iy % 2)
                    y = iy * row_pitch
                    if odd:
                        x += shift_x
                        y += shift_y
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        pts.append((x, y, z))
    pos = np.asarray(pts, dtype=np.float64)
    if pos.size == 0:
        raise ValueError("packing produced no particles")
    if jitter:
        rng = np.random.default_rng(seed)
        pos = pos + rng.standard_normal(pos.shape) * (jitter * r)
    pos[:, 2] = np.maximum(pos[:, 2], r * 1.001)
    return pos


class ContactState:
    """Per-pair tangential/rolling state, keyed by (i, j) with i < j.

    The pair list changes every step as particles move; the state is carried
    across steps by pair key so that a contact that persists keeps its
    incremental displacement, and a broken contact is dropped.
    """
    __slots__ = ("tang", "roll")

    def __init__(self) -> None:
        self.tang: dict[tuple[int, int], np.ndarray] = {}
        self.roll: dict[tuple[int, int], np.ndarray] = {}


def build_pairs(pos: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """All pairs within 2r (candidate contacts), via a cKDTree.

    Returns (iu, ju) integer arrays with iu < ju.
    """
    n = len(pos)
    if n < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    tree = cKDTree(pos)
    pairs = tree.query_pairs(2.0 * radius, output_type="ndarray")
    iu = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    ju = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    return iu, ju


def contact_forces(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
                   iu: np.ndarray, ju: np.ndarray, state: ContactState,
                   mat: Material3D, dt: float,
                   state_keys: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised particle-particle contact forces and torques.

    ``iu``/``ju`` index into ``pos``/``vel``/``omega``.  ``state_keys`` is an
    integer array (one per pair) used as the stable key for the per-pair
    tangential/rolling state, so the state survives when the active particle
    set changes (e.g. discharged particles dropped from the silo).  Default:
    ``iu * len(pos) + ju``.

    Updates ``state.tang`` / ``state.roll`` in place.  Returns
    ``(force (n,3), torque (n,3))``.
    """
    n = len(pos)
    if state_keys is None:
        state_keys = iu * n + ju
    force = np.zeros((n, 3), dtype=np.float64)
    torque = np.zeros((n, 3), dtype=np.float64)
    m_pairs = len(iu)
    if m_pairs == 0:
        return force, torque

    r = mat.radius
    kn, kt, kr = mat.kn, mat.kt, mat.kr
    mu, mu_r = mat.mu, mat.mu_rolling
    gamma = mat.gamma_pair

    dvec = pos[ju] - pos[iu]
    dist = np.linalg.norm(dvec, axis=1)
    overlap = 2.0 * r - dist
    active = overlap > 0.0
    if not active.any():
        return force, torque

    iu_a, ju_a = iu[active], ju[active]
    dvec_a, overlap_a = dvec[active], overlap[active]
    nvec = dvec_a / np.maximum(np.linalg.norm(dvec_a, axis=1), 1e-30)[:, None]

    v_rel = vel[ju_a] - vel[iu_a]
    v_n = (v_rel * nvec).sum(axis=1)
    f_n = np.maximum(kn * overlap_a - gamma * v_n, 0.0)

    # tangential relative velocity at the contact point (includes rotation)
    v_t = v_rel - v_n[:, None] * nvec
    if mat.mu > 0.0:
        # surface velocity of each body at the contact point
        surf_i = np.cross(omega[iu_a], -nvec * r)
        surf_j = np.cross(omega[ju_a], nvec * r)
        v_t_surf = (v_rel - v_n[:, None] * nvec) + (surf_j - surf_i)
        keys = state_keys[active]
        tang = np.array([state.tang.get(k, np.zeros(3)) for k in keys], dtype=np.float64)
        tang = tang + v_t_surf * dt
        tnorm = np.linalg.norm(tang, axis=1)
        cap = mu * f_n / kt
        scale = np.where(tnorm > cap, cap / np.maximum(tnorm, 1e-30), 1.0)
        tang = tang * scale[:, None]
        f_t = -kt * tang
        for k, key in enumerate(keys):
            state.tang[key] = tang[k]
    else:
        f_t = np.zeros_like(v_t)

    f_total = f_n[:, None] * nvec + f_t
    # n points i -> j; the repulsive force pushes i away from j (-n) and j
    # away from i (+n).  (Same convention as repose_calibration.)
    np.add.at(force, iu_a, -f_total)
    np.add.at(force, ju_a, f_total)

    if mat.mu > 0.0:
        # torque on i = r_i x f_t ; on j = r_j x (-f_t); both equal r_i x f_t
        t_i = np.cross(-nvec * r, f_t)
        np.add.at(torque, iu_a, t_i)
        np.add.at(torque, ju_a, t_i)

    if mat.mu_rolling > 0.0:
        spin = omega[ju_a] - omega[iu_a]
        keys = state_keys[active]
        roll = np.array([state.roll.get(k, np.zeros(3)) for k in keys], dtype=np.float64)
        roll = roll + spin * dt
        rnorm = np.linalg.norm(roll, axis=1)
        cap_r = (mu_r * r * f_n) / kr
        scale_r = np.where(rnorm > cap_r, cap_r / np.maximum(rnorm, 1e-30), 1.0)
        roll = roll * scale_r[:, None]
        for k, key in enumerate(keys):
            state.roll[key] = roll[k]
        torque_r = -kr * roll
        np.add.at(torque, iu_a, -torque_r)
        np.add.at(torque, ju_a, torque_r)

    # drop state for pairs that are no longer in contact
    active_keys = set(state_keys[active].tolist())
    for key in list(state.tang.keys()):
        if key not in active_keys:
            del state.tang[key]
    for key in list(state.roll.keys()):
        if key not in active_keys:
            del state.roll[key]

    return force, torque
