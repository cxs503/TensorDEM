"""Vectorised 3-D contact kernel with channel-resolved dissipation accounting.

The contact laws are exactly those of :mod:`tensordem.granular3d_core`:

* normal: linear spring-dashpot, ``F_n = max(k_n * delta - gamma * v_n, 0)``
  with the dashpot coefficient derived from the coefficient of restitution
  (``zeta = |ln e| / sqrt(pi^2 + ln^2 e)``);
* tangential: Cundall-Strack incremental spring saturated at ``mu |F_n|``;
* rolling: isotropic rolling spring saturated at ``mu_r R_eff |F_n|``.

Two things differ, both required by the free-surface-flow benchmark that uses
this kernel (:mod:`tensordem.granular_column.column3d`):

1. **Bookkeeping.**  The per-pair tangential/rolling springs live in dense
   arrays ordered by a sorted integer pair key and are remapped with
   ``np.searchsorted`` instead of a Python ``dict`` loop.  For a 2k-particle
   collapse this is the difference between a few tens of milliseconds and a
   few milliseconds per step, which is what makes an aspect-ratio scan up to
   ``a = 10`` affordable.  :func:`pair_contact_forces` is checked against
   :func:`tensordem.granular3d_core.contact_forces` to machine precision in
   ``tests/test_granular_column3d.py``.

2. **Per-channel dissipated work.**  Every call returns the normal-dashpot
   ("collision") dissipation plus the tangential and rolling ("friction")
   dissipation, computed *within* the step from the slip residual
   ``s = u_rel * dt - d(tang)``: a sticking contact dissipates nothing, a
   sliding contact dissipates ``|f_t| * |s|``.  This is what closes the energy
   ledger of a free-surface collapse to ~1e-4 of the released potential energy
   (criterion (3) of the campaign node) instead of the O(10 %) closure obtained
   when only the dashpot work is booked.

Wall contacts (rigid plane) are provided by :func:`wall_contact_forces` with
the same law and the same channel split; the shared reference kernel has no
walls.

All quantities are SI (m, kg, s, N, rad).
"""
from __future__ import annotations

import numpy as np

from ..granular3d_core import Material3D, build_pairs
from .inertial_friction import InertialFriction

__all__ = [
    "Material3D",
    "PairSpringState",
    "WallSpringState",
    "build_pairs",
    "pair_contact_forces",
    "wall_contact_forces",
]


class PairSpringState:
    """Per-pair tangential/rolling springs in sorted-key (vectorised) form."""

    __slots__ = ("keys", "tang", "roll", "live")

    def __init__(self) -> None:
        self.keys = np.zeros(0, dtype=np.int64)
        self.tang = np.zeros((0, 3), dtype=np.float64)
        self.roll = np.zeros((0, 3), dtype=np.float64)
        self.live = np.zeros(0, dtype=np.float64)

    def take(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gather the stored springs of ``keys`` (missing keys -> zeros)."""
        m = keys.size
        tang = np.zeros((m, 3), dtype=np.float64)
        roll = np.zeros((m, 3), dtype=np.float64)
        if self.keys.size == 0 or m == 0:
            return tang, roll
        idx = np.searchsorted(self.keys, keys)
        np.clip(idx, 0, self.keys.size - 1, out=idx)
        hit = self.keys[idx] == keys
        if hit.any():
            tang[hit] = self.tang[idx[hit]]
            roll[hit] = self.roll[idx[hit]]
        return tang, roll

    def take_live(self, keys: np.ndarray) -> tuple[np.ndarray, float]:
        """Stored live elastic energy per key plus the stored total."""
        live = np.zeros(keys.size, dtype=np.float64)
        if self.keys.size == 0 or keys.size == 0:
            return live, float(self.live.sum())
        idx = np.searchsorted(self.keys, keys)
        np.clip(idx, 0, self.keys.size - 1, out=idx)
        hit = self.keys[idx] == keys
        live[hit] = self.live[idx[hit]]
        return live, float(self.live.sum())

    def store(self, keys: np.ndarray, tang: np.ndarray, roll: np.ndarray,
              live: np.ndarray | None = None) -> None:
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.tang = tang[order]
        self.roll = roll[order]
        self.live = (np.zeros(keys.size) if live is None else live)[order]

    def clear(self) -> None:
        self.keys = np.zeros(0, dtype=np.int64)
        self.tang = np.zeros((0, 3), dtype=np.float64)
        self.roll = np.zeros((0, 3), dtype=np.float64)
        self.live = np.zeros(0, dtype=np.float64)


class WallSpringState:
    """Per-particle tangential/rolling spring against a rigid plane."""

    __slots__ = ("tang", "roll", "active", "live")

    def __init__(self, n: int) -> None:
        self.tang = np.zeros((n, 3), dtype=np.float64)
        self.roll = np.zeros((n, 3), dtype=np.float64)
        self.active = np.zeros(n, dtype=bool)
        self.live = np.zeros(n, dtype=np.float64)


def _empty_pair_result(n: int) -> dict:
    return {
        "force": np.zeros((n, 3), dtype=np.float64),
        "torque": np.zeros((n, 3), dtype=np.float64),
        "n_contacts": 0,
        "normal_force_sum": 0.0,
        "dissip_collision": 0.0,
        "dissip_friction": 0.0,
        "dissip_rolling": 0.0,
        "energy_normal": 0.0,
        "energy_tangential": 0.0,
        "energy_rolling": 0.0,
        "energy_dead": 0.0,
        "release_loss": 0.0,
        "ia": np.zeros(0, dtype=np.int64),
        "ja": np.zeros(0, dtype=np.int64),
        "v_n": np.zeros(0),
        "u_t": np.zeros((0, 3)),
        "f_n": np.zeros(0),
        "f_t": np.zeros((0, 3)),
        "slip_tangential": np.zeros((0, 3)),
        "slip_rolling": np.zeros((0, 3)),
        "mu_eff_mean": float("nan"),
        "mu_eff_max": float("nan"),
        "mu_eff_min": float("nan"),
        "inertial_number_mean": float("nan"),
        "inertial_number_max": float("nan"),
    }


def _mean_or(values) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _max_or(values) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.max()) if arr.size else float("nan")


def _min_or(values) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.min()) if arr.size else float("nan")


def _contact_friction(mu: float, mu_i: InertialFriction | None,
                      slip_velocity: np.ndarray, f_n: np.ndarray,
                      diameter: float, density: float):
    """Friction coefficient per contact plus its inertial number.

    Returns ``(None, None)`` when friction is switched off, a constant array
    when ``mu_i is None`` (bit-identical to the historical ``mu*|F_n|`` cap),
    and the ``mu(I_c)`` array otherwise.
    """
    if mu_i is None:
        if mu <= 0.0:
            return None, None
        return np.full(f_n.shape, float(mu)), None
    if mu_i.mu_2 <= 0.0:
        return None, None
    speed = np.sqrt(np.einsum("ij,ij->i", slip_velocity, slip_velocity))
    i_c = mu_i.inertial_number(speed, f_n, diameter, density)
    return mu_i.effective_mu(speed, f_n, diameter, density), i_c


def pair_contact_forces(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
                        iu: np.ndarray, ju: np.ndarray,
                        state: PairSpringState, mat: Material3D, dt: float,
                        *, update_state: bool = True,
                        contact_mode: str = "one_sided",
                        box: tuple[float, float] | None = None,
                        mu_i: InertialFriction | None = None) -> dict:
    """Particle-particle contact forces, torques and dissipation channels.

    ``iu < ju`` index the candidate pairs (``|x_j - x_i| < 2 r``).  With
    ``update_state=False`` the tangential/rolling springs are neither advanced
    nor stored -- used for the predictor half-kick of the velocity-Verlet
    integrator.  ``box=(Lx, Ly)`` applies the minimum-image convention in the
    two periodic directions (chute flow); default ``None`` is the unwrapped
    (open) geometry and reproduces the previous results bit for bit.

    ``mu_i`` replaces the constant friction coefficient by the rate-dependent
    :class:`~tensordem.granular_column.inertial_friction.InertialFriction`
    closure ``mu(I_c)``, evaluated **per contact** from the same contact-point
    slip velocity the Cundall-Strack spring uses.  ``mu_i=None`` (default) keeps
    the constant-``mu`` law, so every existing caller is unchanged.
    """
    n = pos.shape[0]
    res = _empty_pair_result(n)
    if iu.size == 0:
        if update_state:
            state.clear()
        return res

    r = mat.radius
    kn, kt, kr = mat.kn, mat.kt, mat.kr
    mu, mu_r, gamma = mat.mu, mat.mu_rolling, mat.gamma_pair

    dvec = pos[ju] - pos[iu]
    if box is not None:
        # minimum image in the two periodic directions
        dvec[:, 0] -= box[0] * np.round(dvec[:, 0] / box[0])
        dvec[:, 1] -= box[1] * np.round(dvec[:, 1] / box[1])
    dist = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
    overlap = 2.0 * r - dist
    act = overlap > 0.0
    if not act.any():
        if update_state:
            state.clear()
        return res

    ia, ja = iu[act], ju[act]
    dvec_a, dist_a, overlap_a = dvec[act], dist[act], overlap[act]
    nvec = dvec_a / dist_a[:, None]

    v_rel = vel[ja] - vel[ia]
    v_n = np.einsum("ij,ij->i", v_rel, nvec)
    if contact_mode == "reference":
        # exact convention of tensordem.granular3d_core: linear spring-dashpot
        # clipped at zero (tension cutoff)
        f_n = np.maximum(kn * overlap_a - gamma * v_n, 0.0)
    elif contact_mode == "one_sided":
        # compression-only dashpot: continuous in the overlap (no stored elastic
        # energy is discarded at pull-off), damping only while approaching.
        f_n = kn * overlap_a + gamma * np.maximum(-v_n, 0.0)
    else:
        raise ValueError(f"unknown contact_mode {contact_mode!r}")

    # relative velocity of the two contact points (translation + spin)
    surf_i = np.cross(omega[ia], nvec * r)
    surf_j = np.cross(omega[ja], -nvec * r)
    u_t = (v_rel - v_n[:, None] * nvec) + (surf_j - surf_i)

    keys = ia.astype(np.int64) * n + ja
    tang_old, roll_old = state.take(keys)

    mu_eff, i_c = _contact_friction(mu, mu_i, u_t, f_n, 2.0 * r, mat.density)
    if mu_eff is not None:
        tang = tang_old + u_t * dt
        t_norm = np.sqrt(np.einsum("ij,ij->i", tang, tang))
        cap = mu_eff * f_n / kt
        scale = np.where(t_norm > cap, cap / np.maximum(t_norm, 1e-30), 1.0)
        tang = tang * scale[:, None]
        f_t = -kt * tang
        slip_t = u_t * dt - (tang - tang_old)
    else:
        tang = np.zeros_like(tang_old)
        f_t = np.zeros_like(u_t)
        slip_t = np.zeros_like(u_t)

    if mu_r > 0.0:
        spin = omega[ja] - omega[ia]
        roll = roll_old + spin * dt
        r_norm = np.sqrt(np.einsum("ij,ij->i", roll, roll))
        cap_r = (mu_r * r * f_n) / kr
        scale_r = np.where(r_norm > cap_r, cap_r / np.maximum(r_norm, 1e-30), 1.0)
        roll = roll * scale_r[:, None]
        torque_r = -kr * roll
        slip_r = spin * dt - (roll - roll_old)
    else:
        roll = np.zeros_like(roll_old)
        torque_r = np.zeros_like(roll_old)
        slip_r = np.zeros_like(roll_old)

    force = res["force"]
    torque = res["torque"]
    f_total = f_n[:, None] * nvec + f_t
    np.add.at(force, ia, -f_total)
    np.add.at(force, ja, f_total)
    t_i = np.cross(-nvec * r, f_t)
    np.add.at(torque, ia, t_i)
    np.add.at(torque, ja, t_i)
    if mu_r > 0.0:
        np.add.at(torque, ia, -torque_r)
        np.add.at(torque, ja, torque_r)

    live = np.where(f_n > 0.0,
                    0.5 * kn * overlap_a * overlap_a
                    + 0.5 * kt * np.einsum("ij,ij->i", tang, tang), 0.0)
    release_loss = 0.0
    if update_state:
        prev_live, prev_total = state.take_live(keys)
        release_loss = float(np.sum(prev_live[live <= 0.0])
                             + max(prev_total - float(prev_live.sum()), 0.0))
        state.store(keys, tang, roll, live)

    res.update({
        "n_contacts": int(np.count_nonzero(f_n > 0.0)),
        "normal_force_sum": float(f_n.sum()),
        "dissip_collision": float(np.sum(np.where(f_n > 0.0, gamma * v_n * v_n, 0.0)) * dt),
        "dissip_friction": float(-np.einsum("ij,ij->i", f_t, slip_t).sum()),
        "dissip_rolling": float(-np.einsum("ij,ij->i", torque_r, slip_r).sum()),
        "energy_normal": float(0.5 * kn * np.sum(overlap_a * overlap_a)),
        "energy_tangential": float(0.5 * kt * np.sum(tang * tang)),
        "energy_rolling": float(0.5 * kr * np.sum(roll * roll)),
        # stored elastic energy in contacts whose normal force was truncated to
        # zero by the tension cutoff (f_n clipped); the model cannot do work
        # with it, so it is a separate (irreversible-in-practice) channel.
        "release_loss": release_loss,
        "energy_dead": float(np.sum(np.where(f_n <= 0.0,
                                             0.5 * kn * overlap_a * overlap_a
                                             + 0.5 * kt * np.einsum("ij,ij->i", tang, tang),
                                             0.0))),
        "ia": ia,
        "ja": ja,
        "v_n": v_n,
        "u_t": u_t,
        "f_n": f_n,
        "f_t": f_t,
        "slip_tangential": slip_t,
        "slip_rolling": slip_r,
        "mu_eff_mean": _mean_or(mu_eff),
        "mu_eff_max": _max_or(mu_eff),
        "mu_eff_min": _min_or(mu_eff),
        "inertial_number_mean": _mean_or(i_c),
        "inertial_number_max": _max_or(i_c),
    })
    return res


def wall_contact_forces(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray,
                        mat: Material3D, dt: float, *, floor_z: float = 0.0,
                        mu_wall: float | None = None,
                        mu_rolling_wall: float = 0.0,
                        state: WallSpringState | None = None,
                        update_state: bool = True,
                        contact_mode: str = "one_sided",
                        mu_i: InertialFriction | None = None) -> dict:
    """Contact forces against the rigid horizontal plane ``z = floor_z``.

    The traction state is per particle (the wall is rigid, so every particle
    owns its contact spring).  ``mu_wall`` defaults to the grain-grain
    friction, i.e. a fully rough floor.
    """
    n = pos.shape[0]
    out = {
        "force": np.zeros((n, 3), dtype=np.float64),
        "torque": np.zeros((n, 3), dtype=np.float64),
        "n_contacts": 0,
        "normal_force_sum": 0.0,
        "dissip_collision": 0.0,
        "dissip_friction": 0.0,
        "dissip_rolling": 0.0,
        "energy_normal": 0.0,
        "energy_tangential": 0.0,
        "energy_rolling": 0.0,
        "energy_dead": 0.0,
        "release_loss": 0.0,
        "mu_eff_mean": 0.0,
        "mu_eff_max": 0.0,
        "mu_eff_min": 0.0,
        "inertial_number_mean": 0.0,
        "inertial_number_max": 0.0,
    }
    mu_w = mat.mu if mu_wall is None else float(mu_wall)
    r = mat.radius
    kn, kt, kr = mat.kn, mat.kt, mat.kr
    gamma = mat.gamma_wall

    z = pos[:, 2] - floor_z
    overlap = r - z
    act = overlap > 0.0
    if not act.any():
        if state is not None and update_state:
            state.tang[:] = 0.0
            state.roll[:] = 0.0
            state.active[:] = False
        return out

    idx = np.nonzero(act)[0]
    ov = overlap[idx]
    vz = vel[idx, 2]
    if contact_mode == "reference":
        f_n = np.maximum(kn * ov - gamma * vz, 0.0)
    elif contact_mode == "one_sided":
        f_n = kn * ov + gamma * np.maximum(-vz, 0.0)
    else:
        raise ValueError(f"unknown contact_mode {contact_mode!r}")

    # contact-point velocity: translation + spin about the contact at -r z
    omega_c = omega[idx]
    spin_part = np.stack([
        -r * omega_c[:, 1],
        r * omega_c[:, 0],
        np.zeros(idx.size),
    ], axis=1)
    u_surf = vel[idx] + spin_part
    u_t = u_surf.copy()
    u_t[:, 2] = 0.0

    force = out["force"]
    torque = out["torque"]
    force[idx, 2] += f_n

    if state is None:
        tang_old = np.zeros((idx.size, 3))
        roll_old = np.zeros((idx.size, 3))
    else:
        tang_old = state.tang[idx].copy()
        roll_old = state.roll[idx].copy()

    mu_eff_w, i_c_w = _contact_friction(mu_w, mu_i, u_t, f_n, 2.0 * r,
                                       mat.density)
    if mu_eff_w is not None:
        tang = tang_old + u_t * dt
        t_norm = np.sqrt(np.einsum("ij,ij->i", tang, tang))
        cap = mu_eff_w * f_n / kt
        scale = np.where(t_norm > cap, cap / np.maximum(t_norm, 1e-30), 1.0)
        tang = tang * scale[:, None]
        f_t = -kt * tang
        slip_t = u_t * dt - (tang - tang_old)
    else:
        tang = np.zeros_like(tang_old)
        f_t = np.zeros_like(u_t)
        slip_t = np.zeros_like(u_t)

    if mu_rolling_wall > 0.0:
        roll = roll_old + omega_c * dt
        r_norm = np.sqrt(np.einsum("ij,ij->i", roll, roll))
        cap_r = (mu_rolling_wall * r * f_n) / kr
        scale_r = np.where(r_norm > cap_r, cap_r / np.maximum(r_norm, 1e-30), 1.0)
        roll = roll * scale_r[:, None]
        torque_r = -kr * roll
        slip_r = omega_c * dt - (roll - roll_old)
    else:
        roll = np.zeros_like(roll_old)
        torque_r = np.zeros_like(roll_old)
        slip_r = np.zeros_like(roll_old)

    force[idx, :2] += f_t[:, :2]
    # torque of the tangential traction applied at the contact point (-r z)
    torque[idx] += np.cross(np.broadcast_to(np.array([0.0, 0.0, -r]), (idx.size, 3)),
                            f_t)
    if mu_rolling_wall > 0.0:
        torque[idx] += torque_r

    live = np.where(f_n > 0.0,
                    0.5 * kn * ov * ov + 0.5 * kt * np.einsum("ij,ij->i", tang, tang),
                    0.0)
    release_loss = 0.0
    if state is not None and update_state:
        prev_live = state.live[idx]
        release_loss = float(np.sum(prev_live[live <= 0.0])
                             + max(float(state.live.sum() - prev_live.sum()), 0.0))
        state.tang[:] = 0.0
        state.roll[:] = 0.0
        state.active[:] = False
        state.live[:] = 0.0
        state.tang[idx] = tang
        state.roll[idx] = roll
        state.active[idx] = True
        state.live[idx] = live

    out.update({
        "n_contacts": int(idx.size),
        "normal_force_sum": float(f_n.sum()),
        "dissip_collision": float(np.sum(gamma * vz * vz) * dt),
        "dissip_friction": float(-np.einsum("ij,ij->i", f_t, slip_t).sum()),
        "dissip_rolling": float(-np.einsum("ij,ij->i", torque_r, slip_r).sum()),
        "energy_normal": float(0.5 * kn * np.sum(ov * ov)),
        "energy_tangential": float(0.5 * kt * np.sum(tang * tang)),
        "energy_rolling": float(0.5 * kr * np.sum(roll * roll)),
        "release_loss": release_loss,
        "release_loss": release_loss,
        "energy_dead": float(np.sum(np.where(f_n <= 0.0,
                                             0.5 * kn * ov * ov
                                             + 0.5 * kt * np.einsum("ij,ij->i", tang, tang),
                                             0.0))),
        "mu_eff_mean": _mean_or(mu_eff_w),
        "mu_eff_max": _max_or(mu_eff_w),
        "mu_eff_min": _min_or(mu_eff_w),
        "inertial_number_mean": _mean_or(i_c_w),
        "inertial_number_max": _max_or(i_c_w),
    })
    return out
