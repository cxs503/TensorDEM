"""Rotating-drum granular-flow kernel (2-D, monodisperse disks, SI units).

Why this module exists
----------------------
``tensordem.granular2d`` provides rigid frictional disks with *static* polyline
boundaries (channel banks, hull).  A rotary drum needs the boundary to be a
**moving** circle whose material points carry a prescribed tangential velocity,
plus a mechanical power ledger (torque x angular velocity in, contact
dissipation out).  Neither exists elsewhere in the tree, so this module adds
them as a self-contained, dependency-light kernel (numpy + scipy cKDTree,
float64, deterministic).

Model
-----
* particles: monodisperse disks, ``m = rho*pi*r^2*t``, ``I = 0.5*m*r^2``;
* normal contact: ``F_n = max(k_n*delta - gamma*v_n, 0)`` with
  ``gamma = 2*zeta*sqrt(k_n*m_eff)`` and ``zeta`` from the restitution;
* tangential contact: Cundall-Strack incremental spring saturated at
  ``mu*F_n`` (history kept per Verlet pair, so static friction exists);
* rolling resistance: viscous-regularised Coulomb law ``|M_r| <= mu_r*r*F_n``;
* drum wall: circle of radius ``R`` at the origin rotating at ``omega``
  (CCW positive); wall material velocity ``v_wall = omega * z_hat x x_contact``.

Power ledger (book-keeping identity)
------------------------------------
Splitting wall contacts out of the kinetic-energy balance gives

    d/dt (K + U_g + U_el) = P_in - Phi_diss ,
    P_in     = sum_wall F_on_particle . v_wall      (= -tau_drum * omega)
    Phi_diss = Phi_n + Phi_t + Phi_roll >= 0 .

``P_in`` (global wall torque) and ``Phi_diss`` (local contact dissipation) are
accumulated by independent code paths, so comparing them is a genuine closure
check.  In a steady state the left-hand side vanishes over one revolution and
``P_in = Phi_diss``.

All quantities SI (m, kg, s, N, rad).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "DrumConfig",
    "DrumState",
    "cross_z",
    "damping_ratio",
    "make_drum_state",
    "run_drum",
]


def cross_z(a, b):
    """z-component of the 2-D cross product (works on (..., 2) arrays)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def damping_ratio(restitution: float) -> float:
    """zeta = |ln e| / sqrt(pi^2 + ln^2 e)."""
    e = min(max(float(restitution), 1e-6), 1.0 - 1e-12)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


@dataclass(frozen=True)
class DrumConfig:
    """Drum geometry, material and contact parameters (SI)."""

    radius: float = 0.15
    particle_radius: float = 0.006
    density: float = 2500.0
    thickness: float = 0.012
    kn: float = 2.0e4
    kt_ratio: float = 0.3
    restitution: float = 0.3
    mu: float = 0.5
    mu_wall: float = 0.5
    mu_rolling: float = 0.15
    c_rolling: float = 0.02
    gravity: float = 9.81
    dt: float = 5.0e-5
    skin: float = 0.0015
    rebuild_every: int = 10

    def __post_init__(self) -> None:
        for name in ("radius", "particle_radius", "density", "thickness", "kn", "gravity", "dt"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if 2.0 * self.particle_radius >= self.radius:
            raise ValueError("particle radius must be smaller than the drum radius")
        for name in ("mu", "mu_wall", "mu_rolling", "c_rolling", "skin", "kt_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 < float(self.restitution) <= 1.0:
            raise ValueError("restitution must be in (0, 1]")

    @property
    def mass(self) -> float:
        return self.density * math.pi * self.particle_radius ** 2 * self.thickness

    @property
    def inertia(self) -> float:
        return 0.5 * self.mass * self.particle_radius ** 2

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def zeta(self) -> float:
        return damping_ratio(self.restitution)

    def gamma(self, m_eff: float | None = None) -> float:
        if m_eff is None:
            m_eff = 0.5 * self.mass
        return 2.0 * self.zeta * math.sqrt(self.kn * m_eff)

    @property
    def contact_time(self) -> float:
        """Contact duration estimate ``pi*sqrt(m_eff/k_n)`` (s)."""
        return math.pi * math.sqrt(0.5 * self.mass / self.kn)


@dataclass
class DrumState:
    pos: np.ndarray          # (N, 2)
    vel: np.ndarray          # (N, 2)
    spin: np.ndarray         # (N,)  angular velocity about z
    omega: float             # drum angular velocity (rad/s)
    config: DrumConfig


def _hex_bed_positions(cfg: DrumConfig, fill: float, seed: int) -> np.ndarray:
    """Deterministic triangular-lattice bed filling the bottom of the drum."""
    r = cfg.particle_radius
    target_area = fill * math.pi * cfg.radius ** 2
    spacing = 2.0 * r * 0.995   # dense lattice: every particle starts in contact
    n_wanted = max(8, int(round(target_area / (0.5 * math.sqrt(3.0) * spacing ** 2))))
    pts = []
    y = -cfg.radius + 1.2 * r
    row = 0
    dy = spacing * math.sqrt(3.0) / 2.0
    while len(pts) < n_wanted and y < cfg.radius:
        xoff = (row % 2) * 0.5 * spacing
        x = -cfg.radius + 1.2 * r + xoff
        while x < cfg.radius - 1.2 * r and len(pts) < n_wanted:
            if math.hypot(x, y) <= cfg.radius - 1.05 * r:
                pts.append((x, y))
            x += spacing
        y += dy
        row += 1
    arr = np.asarray(pts, dtype=float)
    rng = np.random.default_rng(seed)
    return arr + rng.normal(scale=0.02 * r, size=arr.shape)


def make_drum_state(cfg: DrumConfig, fill: float, omega: float, seed: int = 0) -> DrumState:
    """Initialise a lattice bed; ``fill`` is the 2-D area fraction of solids."""
    if not 0.0 < fill < 0.9:
        raise ValueError("fill must be in (0, 0.9)")
    pos = _hex_bed_positions(cfg, fill, seed)
    n = pos.shape[0]
    return DrumState(
        pos=pos,
        vel=np.zeros((n, 2), dtype=float),
        spin=np.zeros(n, dtype=float),
        omega=float(omega),
        config=cfg,
    )


def _elastic_energy(cfg: DrumConfig, pos, pair_i, pair_j, xi_pp, xi_w, r) -> float:
    """Stored normal/tangential spring energy of all current contacts (J)."""
    u = 0.0
    if pair_i.size:
        d = pos[pair_i] - pos[pair_j]
        dist = np.maximum(np.sqrt(np.einsum("ij,ij->i", d, d)), 1e-12)
        delta = 2.0 * r - dist
        act = delta > 0.0
        if act.any():
            u += 0.5 * cfg.kn * float(np.sum(delta[act] ** 2))
            u += 0.5 * cfg.kt * float(np.sum(np.einsum("ij,ij->i", xi_pp[act], xi_pp[act])))
    dist_c = np.sqrt(np.einsum("ij,ij->i", pos, pos))
    dw = r + dist_c - cfg.radius
    act = dw > 0.0
    if act.any():
        u += 0.5 * cfg.kn * float(np.sum(dw[act] ** 2))
        u += 0.5 * cfg.kt * float(np.sum(np.einsum("ij,ij->i", xi_w[act], xi_w[act])))
    return u


def _mechanical_energy(cfg: DrumConfig, pos, vel, spin) -> float:
    """Kinetic + gravitational energy of the particles (J)."""
    ke = 0.5 * cfg.mass * float(np.sum(vel * vel)) + 0.5 * cfg.inertia * float(np.sum(spin * spin))
    pg = cfg.mass * cfg.gravity * float(np.sum(pos[:, 1]))
    return ke + pg


def run_drum(
    state: DrumState,
    revolutions: float = 3.0,
    measure_last: float = 1.0,
    sample_every: int = 50,
    settle_time: float = 0.0,
    keep_snapshot: bool = True,
) -> dict:
    """Integrate the rotating drum and return observables plus a power ledger.

    Parameters
    ----------
    revolutions : revolutions integrated *after* the settle phase.
    measure_last : of those, the last this-many revolutions are the measurement
        window (time averages and the power ledger are evaluated there only).
    sample_every : sampling stride in steps.
    settle_time : gravity settling time with a static wall before rotation.
    """
    cfg = state.config
    pos, vel, spin = state.pos.copy(), state.vel.copy(), state.spin.copy()
    n = pos.shape[0]
    r = cfg.particle_radius
    m, inertia = cfg.mass, cfg.inertia
    dt = cfg.dt
    omega_abs = abs(state.omega)
    if omega_abs < 1e-12:
        rev_time = None
    else:
        rev_time = 2.0 * math.pi / omega_abs
    gamma = cfg.gamma()

    settle_steps = int(round(settle_time / dt))
    spin_steps = 0 if rev_time is None else int(round(revolutions * rev_time / dt))
    total_steps = settle_steps + spin_steps
    if measure_last <= 0.0:
        measure_from = 0
    elif rev_time is None:
        measure_from = total_steps // 2
    else:
        measure_from = max(0, total_steps - int(round(measure_last * rev_time / dt)))

    pair_i = np.empty(0, dtype=np.int64)
    pair_j = np.empty(0, dtype=np.int64)
    xi_pp = np.zeros((0, 2))
    keys_pp = np.empty(0, dtype=np.int64)
    xi_w = np.zeros((n, 2))
    nn_contacts = np.zeros(n, dtype=float)

    work_in = 0.0
    phi_n = phi_t = phi_r = 0.0
    win = {"work_in": 0.0, "phi_n": 0.0, "phi_t": 0.0, "phi_r": 0.0,
           "e_start": None, "e_end": None, "t_win": 0.0, "tau_int": 0.0}
    samples: list[dict] = []
    snapshot = None
    energy = _mechanical_energy(cfg, pos, vel, spin)

    step = 0
    while step < total_steps:
        omega_now = state.omega if step >= settle_steps else 0.0

        # ---------------- neighbour list --------------------------------
        if step % cfg.rebuild_every == 0:
            tree = cKDTree(pos)
            pairs = np.asarray(tree.query_pairs(2.0 * r + cfg.skin, output_type="ndarray"),
                               dtype=np.int64).reshape(-1, 2)
            order = np.lexsort((pairs[:, 1], pairs[:, 0])) if pairs.size else None
            if pairs.size:
                pairs = pairs[order]
            keys_new = pair_i_new = pairs[:, 0] * n + pairs[:, 1]
            # carry the tangential spring history across neighbour-list rebuilds
            # (pair identity is the sorted integer key i*n+j)
            pos_idx = np.searchsorted(keys_pp, keys_new)
            pos_idx = np.clip(pos_idx, 0, max(keys_pp.size - 1, 0))
            if keys_pp.size:
                hit = keys_pp[pos_idx] == keys_new
            else:
                hit = np.zeros(keys_new.size, dtype=bool)
            xi_new = np.zeros((keys_new.size, 2))
            if hit.any():
                xi_new[hit] = xi_pp[pos_idx[hit]]
            xi_pp, keys_pp = xi_new, keys_new
            pair_i, pair_j = pairs[:, 0].copy(), pairs[:, 1].copy()

        force = np.zeros((n, 2))
        torque = np.zeros(n)
        phi_n_step = phi_t_step = phi_r_step = 0.0
        dbg = {}
        p_in_step = 0.0
        nn_contacts[:] = 0.0

        # ---------------- particle-particle ----------------------------
        if pair_i.size:
            d = pos[pair_i] - pos[pair_j]
            dist = np.maximum(np.sqrt(np.einsum("ij,ij->i", d, d)), 1e-12)
            nrm = d / dist[:, None]
            delta = 2.0 * r - dist
            active = delta > 0.0
            if (~active).any():
                phi_t_step += 0.5 * cfg.kt * float(np.sum(np.einsum("ij,ij->i", xi_pp[~active], xi_pp[~active]))) / dt
                xi_pp[~active] = 0.0
            if active.any():
                ia, ja = pair_i[active], pair_j[active]
                na, delta_a = nrm[active], delta[active]
                nn_contacts += np.bincount(ia, minlength=n)
                nn_contacts += np.bincount(ja, minlength=n)
                vrel = vel[ia] - vel[ja]
                vn = np.einsum("ij,ij->i", vrel, na)
                fn = np.maximum(cfg.kn * delta_a - gamma * vn, 0.0)
                zx = np.stack([-na[:, 1], na[:, 0]], axis=1)      # z_hat x n
                vrel_pt = vrel - r * (spin[ia] + spin[ja])[:, None] * zx
                vt = vrel_pt - np.einsum("ij,ij->i", vrel_pt, na)[:, None] * na
                vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt))
                idx = np.flatnonzero(active)
                xi_pp[idx] += vt * dt
                xi_pp[idx] -= np.einsum("ij,ij->i", xi_pp[idx], na)[:, None] * na
                ft_vec = -cfg.kt * xi_pp[idx]
                ft_mag = np.sqrt(np.einsum("ij,ij->i", ft_vec, ft_vec))
                ft_cap = cfg.mu * fn
                over = ft_mag > ft_cap
                if over.any():
                    ratio = np.where(over, ft_cap / np.maximum(ft_mag, 1e-300), 1.0)
                    ft_vec = ft_vec * ratio[:, None]
                    xi_pp[idx] = np.where(over[:, None], -ft_vec / max(cfg.kt, 1e-300),
                                          xi_pp[idx])
                fvec = fn[:, None] * na + ft_vec
                for k in range(2):
                    force[:, k] += np.bincount(ia, weights=fvec[:, k], minlength=n)
                    force[:, k] -= np.bincount(ja, weights=fvec[:, k], minlength=n)
                # contact point on i is x_i - r*n, on j is x_j + r*n, so the
                # friction *couple* gives tau_i = tau_j (equal signs); applying
                # opposite signs here would pump angular momentum (bug fixed).
                tq = cross_z(-r * na, fvec)
                torque += np.bincount(ia, weights=tq, minlength=n)
                torque += np.bincount(ja, weights=tq, minlength=n)
                dw = spin[ia] - spin[ja]
                mr_cap = cfg.mu_rolling * r * fn
                mr = np.clip(-0.5 * cfg.c_rolling * r * fn * dw, -mr_cap, mr_cap)
                torque += np.bincount(ia, weights=mr, minlength=n)
                torque -= np.bincount(ja, weights=mr, minlength=n)
                dbg.update(n_pair=int(active.sum()), max_delta=float(delta_a.max()),
                           max_fn=float(fn.max()), max_vt=float(vt_mag.max()),
                           max_ft=float(ft_mag.max()))
                phi_n_step += float(np.sum(gamma * vn * vn * (fn > 0.0)))
                if over.any():
                    phi_t_step += float(np.sum(ft_cap[over] * vt_mag[over]))
                phi_r_step += float(np.sum(np.abs(mr * dw)))

        # ---------------- drum wall -------------------------------------
        dist_c = np.maximum(np.sqrt(np.einsum("ij,ij->i", pos, pos)), 1e-12)
        n_out = pos / dist_c[:, None]                 # wall -> particle (outward)
        delta_w = r + dist_c - cfg.radius             # penetration (>0: touching)
        wact = np.flatnonzero(delta_w > 0.0)
        left = delta_w <= 0.0
        if left.any():
            phi_t_step += 0.5 * cfg.kt * float(np.sum(np.einsum("ij,ij->i", xi_w[left], xi_w[left]))) / dt
            xi_w[left] = 0.0
        if wact.size:
            n_o = n_out[wact]
            n_in = -n_o                               # inward normal = force direction
            nn_contacts[wact] += 1.0
            zx = np.stack([-n_o[:, 1], n_o[:, 0]], axis=1)     # z_hat x n_out
            v_wall = omega_now * cfg.radius * zx
            # material point of the particle at the contact: x_i + r*n_out
            vrel = vel[wact] + r * spin[wact][:, None] * zx - v_wall
            vn = np.einsum("ij,ij->i", vrel, n_in)
            fn = np.maximum(cfg.kn * delta_w[wact] - gamma * vn, 0.0)
            vt = vrel - np.einsum("ij,ij->i", vrel, n_in)[:, None] * n_in
            vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt))
            xi_w[wact] += vt * dt
            xi_w[wact] -= np.einsum("ij,ij->i", xi_w[wact], n_in)[:, None] * n_in
            ft_vec = -cfg.kt * xi_w[wact]
            ft_mag = np.sqrt(np.einsum("ij,ij->i", ft_vec, ft_vec))
            ft_cap = cfg.mu_wall * fn
            over = ft_mag > ft_cap
            if over.any():
                ratio = np.where(over, ft_cap / np.maximum(ft_mag, 1e-300), 1.0)
                ft_vec = ft_vec * ratio[:, None]
                xi_w[wact] = np.where(over[:, None], -ft_vec / max(cfg.kt, 1e-300), xi_w[wact])
            fvec = fn[:, None] * n_in + ft_vec
            force[wact] += fvec
            tq = cross_z(-r * n_in, fvec)             # contact offset = +r*n_out
            mr_cap = cfg.mu_rolling * r * fn
            dw_w = spin[wact] - omega_now
            mr = np.clip(-0.5 * cfg.c_rolling * r * fn * dw_w, -mr_cap, mr_cap)
            torque[wact] += tq + mr
            p_in_step += float(np.sum(np.einsum("ij,ij->i", fvec, v_wall)))
            phi_n_step += float(np.sum(gamma * vn * vn * (fn > 0.0)))
            if over.any():
                phi_t_step += float(np.sum(ft_cap[over] * vt_mag[over]))
            phi_r_step += float(np.sum(np.abs(mr * dw_w)))

        # ---------------- integrate (symplectic Euler) -------------------
        force[:, 1] -= m * cfg.gravity
        vel += force / m * dt
        pos += vel * dt
        spin += torque / inertia * dt

        work_in += p_in_step * dt
        phi_n += phi_n_step * dt
        phi_t += phi_t_step * dt
        phi_r += phi_r_step * dt
        energy = _mechanical_energy(cfg, pos, vel, spin)

        if step == measure_from:
            win["e_start"] = energy + _elastic_energy(cfg, pos, pair_i, pair_j, xi_pp, xi_w, r)
            win["work_in"] = 0.0
            win["phi_n"] = win["phi_t"] = win["phi_r"] = 0.0
            win["t_win"] = 0.0
        if step >= measure_from:
            win["work_in"] += p_in_step * dt
            win["phi_n"] += phi_n_step * dt
            win["phi_t"] += phi_t_step * dt
            win["phi_r"] += phi_r_step * dt
            win["t_win"] += dt
            win["tau_int"] += p_in_step * dt
            win["e_end"] = energy + _elastic_energy(cfg, pos, pair_i, pair_j, xi_pp, xi_w, r)

            if step % sample_every == 0:
                r_norm = np.linalg.norm(pos, axis=1) / cfg.radius
                ang = cross_z(pos, vel) / np.maximum(np.sum(pos * pos, axis=1), 1e-12)
                interior = (r_norm < 0.8) & (nn_contacts > 0.0)
                w_bed = float(np.median(ang[interior])) if interior.any() else 0.0
                # bulk rotation from the angular momentum of the loaded particles
                loaded = nn_contacts > 0.0
                rad2 = np.maximum(np.sum(pos * pos, axis=1), 1e-12)
                w_bulk = (float(np.sum(cross_z(pos, vel)[loaded]))
                          / float(np.sum(rad2[loaded]))) if loaded.any() else 0.0
                # tangential (co-rotating) velocity of the wall-adjacent layer
                e_t = np.stack([-pos[:, 1], pos[:, 0]], axis=1) / np.sqrt(rad2)[:, None]
                vt_all = np.einsum("ij,ij->i", vel, e_t)
                wall_layer = (r_norm > 0.85) & loaded
                ratio_wall = (float(np.mean(vt_all[wall_layer])) / (omega_abs * cfg.radius)
                              if (wall_layer.any() and omega_abs > 0) else 0.0)
                surf = (r_norm > 0.5) & loaded & (pos[:, 1] > 0.0)
                airborne = float(np.mean(nn_contacts == 0.0))
                wall_layer = (r_norm > 0.9) & (nn_contacts > 0.0)
                w_wall_layer = float(np.median(ang[wall_layer])) if wall_layer.any() else 0.0
                samples.append({
                    "t": (step - settle_steps) * dt,
                    "omega_bed": w_bed,
                    "omega_wall_layer": w_wall_layer,
                    "airborne_fraction": airborne,
                    "r_p05_over_R": float(np.percentile(r_norm, 5)),
                    "ke": 0.5 * m * float(np.sum(vel * vel)),
                    "omega_bulk": w_bulk,
                    "wall_velocity_ratio": ratio_wall,
                    "surface_speed_ratio": (float(np.mean(vt_all[surf])) / (omega_abs * cfg.radius)
                                            if (surf.any() and omega_abs > 0) else 0.0),
                    "height_com": float(np.mean(pos[:, 1])) / cfg.radius,
                    "flight_upper_fraction": float(np.mean((nn_contacts == 0.0)
                                                            & (pos[:, 1] > 0.25 * cfg.radius))),
                    "max_v": float(np.max(np.linalg.norm(vel, axis=1))),
                    "max_spin": float(np.max(np.abs(spin))),
                    "n_pair": dbg.get("n_pair", 0),
                    "max_delta": dbg.get("max_delta", 0.0),
                    "max_fn": dbg.get("max_fn", 0.0),
                    "max_ft": dbg.get("max_ft", 0.0),
                    "max_vt": dbg.get("max_vt", 0.0),
                })

        if step == total_steps - 1 and keep_snapshot:
            snapshot = {
                "pos": pos.copy(),
                "vel": vel.copy(),
                "spin": spin.copy(),
                "nn_contacts": nn_contacts.copy(),
                "r_norm": np.linalg.norm(pos, axis=1) / cfg.radius,
            }
        step += 1

    solid_area = n * math.pi * r ** 2
    fill_realised = solid_area / (math.pi * cfg.radius ** 2)
    t_win = max(win["t_win"], dt)
    p_in = win["work_in"] / t_win
    phi_n = win["phi_n"] / t_win
    phi_t = win["phi_t"] / t_win
    phi_r = win["phi_r"] / t_win
    phi_diss = phi_n + phi_t + phi_r
    de = (win["e_end"] - win["e_start"]) if win["e_start"] is not None else float("nan")
    ledger = {
        "p_in_w": p_in,
        "tau_drum_nm": (-p_in / state.omega) if state.omega != 0.0 else float("nan"),
        "phi_n_w": phi_n,
        "phi_t_w": phi_t,
        "phi_roll_w": phi_r,
        "phi_diss_w": phi_diss,
        "closure_rel": abs(p_in - phi_diss) / p_in if p_in > 0 else float("nan"),
        "dE_w": de / t_win,
        "energy_balance_rel": (abs(de / t_win - (p_in - phi_diss)) / p_in) if p_in > 0 else float("nan"),
        "window_s": t_win,
        "steps": total_steps,
    }
    mean = {}
    if samples:
        for key in samples[0]:
            vals = np.array([s[key] for s in samples], dtype=float)
            mean[key] = float(np.mean(vals))
        ang = np.array([s["omega_bed"] for s in samples])
        mean["omega_bed_std"] = float(np.std(ang))
    return {
        "fill_realised": fill_realised,
        "n_particles": n,
        "omega": state.omega,
        "steps": total_steps,
        "samples": samples,
        "mean": mean,
        "ledger": ledger,
        "snapshot": snapshot,
    }