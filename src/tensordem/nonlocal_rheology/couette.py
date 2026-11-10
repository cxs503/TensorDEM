"""2D annular (Couette) shear cell used as the nonlocal-rheology test bed.

Why this geometry
-----------------
In an annulus the shear stress decays as ``tau(r) ~ 1/r^2`` while the pressure
is much flatter, so the stress ratio ``mu(r) = tau/P`` sweeps a *range* within a
single run and crosses the yield value ``mu_s`` somewhere inside the cell.  The
local ``mu(I)`` law therefore predicts a shear band of a few tenths of a
particle diameter and a perfectly rigid plug beyond it, whereas DEM and
experiments show a band several diameters wide that keeps flowing with an
exponentially decaying rate — the classical nonlocal signature.  Changing the
annulus gap changes the *local* band width, which is what the two-scale
(wide band / thin band) calibration exploits.

Implementation
--------------
A standalone 2D soft-disk kernel (numpy + ``scipy.spatial.cKDTree``):

* linear spring-dashpot normal contact, damping from the restitution,
* Coulomb (velocity-based) friction at contacts and walls,
* rough inner cylinder rotating at ``omega`` (a torque meter),
* rough outer wall whose radius follows a proportional servo so that the mean
  radial pressure tracks ``case.pressure`` (a stress-controlled cell).

Fields are extracted by coarse-graining (virial stress + solid fraction per
radial bin) so that every material point yields one ``(I, mu, P)`` sample.  The
dimensionless groups follow the local rheology convention
``I = gammadot d50 / sqrt(P/rho)``, ``mu = tau/P``.

All quantities SI (m, kg, s, Pa); the cell is unit thickness so areas are in m^2.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


class CouetteError(ValueError):
    """Base class for annular-cell errors."""


@dataclass(frozen=True)
class CouetteCase:
    """Annular Couette cell (2D, unit thickness)."""

    radius: float = 1.0e-3            # mean particle radius (m)
    polydispersity: float = 0.12
    density: float = 2500.0
    mu: float = 0.40                  # particle-particle friction
    mu_wall: float = 3.0          # rough wall: avoids gross slip at the drive
    restitution: float = 0.10
    kn: float = 3.0e5                 # normal stiffness (N/m)
    kt_ratio: float = 0.5
    r_inner: float = 0.010            # inner cylinder radius (m)
    r_outer: float = 0.045            # outer wall radius (m) (gap = 17.5 d50)
    fill_fraction: float = 0.79       # initial solid fraction
    pressure: float = 300.0           # servo target radial pressure (Pa)
    omega: float = 20.0               # inner-cylinder angular velocity (rad/s)
    wall_servo: bool = True
    servo_relax: float = 0.5
    servo_span: float = 0.005         # max wall-radius change per step (in d)
    dt: Optional[float] = None
    settle_strain: float = 6.0        # settling (no shear) before shearing
    shear_strain: float = 30.0        # shear strain to reach steady state
    average_strain: float = 15.0      # strain window averaged over
    n_bins: int = 12
    seed: int = 20261110

    def __post_init__(self) -> None:
        for name in ("radius", "density", "kn", "kt_ratio", "r_inner",
                     "r_outer", "pressure", "shear_strain", "average_strain"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
                raise CouetteError(f"{name} must be finite and positive")
        if not 0.0 <= self.restitution < 1.0:
            raise CouetteError("restitution must lie in [0, 1)")
        if not 0.0 < self.fill_fraction < 0.95:
            raise CouetteError("fill_fraction must lie in (0, 0.95)")
        if not 0.0 <= self.polydispersity < 0.5:
            raise CouetteError("polydispersity must lie in [0, 0.5)")
        if not self.r_outer > self.r_inner + 4.0 * self.radius:
            raise CouetteError("annulus gap must hold at least ~4 particles")
        if self.n_bins < 4:
            raise CouetteError("need at least 4 radial bins")
        if self.omega < 0.0:
            raise CouetteError("omega must be nonnegative")

    @property
    def d50(self) -> float:
        return 2.0 * self.radius

    @property
    def particle_mass(self) -> float:
        return self.density * math.pi * self.radius ** 2

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return float(self.dt)
        m_eff = 0.5 * self.particle_mass
        period = 2.0 * math.pi * math.sqrt(m_eff / self.kn)
        return period / 30.0

    @property
    def gap(self) -> float:
        return self.r_outer - self.r_inner

    def wall_strain_rate(self) -> float:
        """Nominal shear rate ``omega r_inner / gap`` (1/s)."""
        return self.omega * self.r_inner / self.gap


# ---------------------------------------------------------------------------
# packing preparation
# ---------------------------------------------------------------------------
def _make_packing(case: CouetteCase) -> tuple[np.ndarray, np.ndarray]:
    """Dense, overlap-free packing: lattice at the target solid fraction.

    The lattice pitch is set by the target fill fraction (a triangular lattice
    has ``phi = pi/(2 sqrt(3)) (2r/pitch)^2``), a small jitter removes the
    crystalline order, and a positional relaxation removes the residual overlaps
    introduced by the polydispersity.  The result is a dense (``phi ~ fill``),
    nearly stress-free start so that the outer-wall servo only has to trim the
    pressure — a loose start would need a large radial compression which the
    servo cannot supply in a 6-diameter gap.
    """
    rng = np.random.default_rng(case.seed)
    r_max = case.radius * (1.0 + case.polydispersity)
    phi_lattice = case.fill_fraction
    pitch = 2.0 * case.radius * math.sqrt(math.pi / (2.0 * math.sqrt(3.0) * phi_lattice))
    pts: list[list[float]] = []
    radii: list[float] = []
    row = 0
    reach = case.r_outer + r_max
    y = -reach
    while y < reach:
        x = -reach + (0.5 * pitch if row % 2 else 0.0)
        while x < reach:
            r = case.radius * (1.0 + case.polydispersity * rng.uniform(-1.0, 1.0))
            px = x + rng.uniform(-0.04, 0.04) * pitch
            py = y + rng.uniform(-0.04, 0.04) * pitch
            rr = math.hypot(px, py)
            if case.r_inner + 1.01 * r < rr < case.r_outer - 1.01 * r:
                pts.append([px, py])
                radii.append(r)
            x += pitch
        y += pitch * math.sqrt(3.0) / 2.0
        row += 1
    if len(pts) < 8:
        raise CouetteError("lattice packing failed to place particles")
    pos = np.asarray(pts, dtype=float)
    rad = np.asarray(radii, dtype=float)
    # positional relaxation: remove residual overlaps without changing phi much
    for _ in range(40):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(2.0 * float(rad.max()), output_type="ndarray")
        if pairs.size == 0:
            break
        d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
        dist = np.linalg.norm(d, axis=1)
        good = dist > 1e-15
        pairs, d, dist = pairs[good], d[good], dist[good]
        overlap = rad[pairs[:, 0]] + rad[pairs[:, 1]] - dist
        hit = overlap > 1e-12
        if not hit.any():
            break
        nvec = d[hit] / dist[hit][:, None]
        corr = overlap[hit][:, None] * nvec
        np.add.at(pos, pairs[hit, 0], -0.55 * corr)
        np.add.at(pos, pairs[hit, 1], 0.55 * corr)
        # keep the material inside the annulus
        rr = np.hypot(pos[:, 0], pos[:, 1])
        inner = rr < (case.r_inner + rad)
        if inner.any():
            scale = (case.r_inner + rad[inner]) / np.maximum(rr[inner], 1e-30)
            pos[inner] *= scale[:, None]
        outer = rr > (case.r_outer - rad)
        if outer.any():
            scale = (case.r_outer - rad[outer]) / np.maximum(rr[outer], 1e-30)
            pos[outer] *= scale[:, None]
    return pos, rad


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------
def _damping(restitution: float) -> float:
    """Dashpot coefficient in units of ``sqrt(k m_eff)`` (log-decrement fit)."""
    if restitution <= 0.0:
        return 1.0
    e = min(restitution, 0.999)
    return -math.log(e) / math.pi if e < 1.0 else 0.0


def simulate_couette(case: CouetteCase, *, history: bool = False) -> dict:
    """Run the annular cell and return the coarse-grained radial profiles."""
    rng = np.random.default_rng(case.seed + 17)
    pos, rad = _make_packing(case)
    n = pos.shape[0]
    mass = case.density * np.pi * rad ** 2
    vel = np.zeros((n, 2))
    dt = case.timestep
    kn, kt = case.kn, case.kt_ratio * case.kn
    gamma_n = 2.0 * _damping(case.restitution) * np.sqrt(kn * 0.5 * mass.mean())
    r_wall = case.r_outer
    bin_edges = np.linspace(case.r_inner, r_wall, case.n_bins + 1)

    cache: dict = {"pairs": None, "anchor": None}

    def contact_pairs(p, r):
        """Verlet-style cached neighbour list (rebuilt on large displacement)."""
        rmax = float(r.max())
        need = (cache["pairs"] is None or cache["anchor"] is None
                or float(np.max(np.linalg.norm(p - cache["anchor"], axis=1))) > 0.15 * rmax)
        if need:
            tree = cKDTree(p)
            pairs = tree.query_pairs(2.0 * rmax * 1.25 + 1e-12, output_type="ndarray")
            cache["pairs"] = pairs
            cache["anchor"] = p.copy()
        pairs = cache["pairs"]
        if pairs.size == 0:
            return (pairs, np.zeros((0, 2)), np.zeros(0), np.zeros(0))
        d = p[pairs[:, 1]] - p[pairs[:, 0]]
        dist = np.linalg.norm(d, axis=1)
        good = dist > 1e-15
        pairs, d, dist = pairs[good], d[good], dist[good]
        nvec = d / dist[:, None]
        overlap = r[pairs[:, 0]] + r[pairs[:, 1]] - dist
        hit = overlap > -0.02 * rmax
        return pairs[hit], nvec[hit], overlap[hit], d[hit]

    def accumulate_scatter(index, values, n):
        """Fast scatter-add (bincount based, much faster than np.add.at)."""
        out = np.empty((n,) + values.shape[1:])
        for k in range(values.shape[1]):
            out[:, k] = np.bincount(index, weights=values[:, k], minlength=n)
        return out

    def wall_forces(p, v, r):
        """Contact forces with the two cylinders; returns (force, torque, p_wall)."""
        force = np.zeros_like(p)
        rr = np.hypot(p[:, 0], p[:, 1])
        radial = p / np.maximum(rr, 1e-30)[:, None]
        torque = 0.0
        pen_in = (case.r_inner + r) - rr
        inner = pen_in > 0.0
        if inner.any():
            idx = np.nonzero(inner)[0]
            nv = radial[idx]                         # wall pushes outward
            v_wall = case.omega * np.stack([-p[idx, 1], p[idx, 0]], axis=1)
            fn = np.maximum(kn * pen_in[idx], 0.0)
            f = fn[:, None] * nv
            vt = v[idx] - v_wall
            vt_rad = np.einsum("ij,ij->i", vt, nv)
            vt_t = vt - vt_rad[:, None] * nv
            sp = np.linalg.norm(vt_t, axis=1)
            ft = np.where(sp > 1e-14, case.mu_wall * fn, 0.0)
            dir_t = -vt_t / np.maximum(sp, 1e-30)[:, None]
            f = f + ft[:, None] * dir_t
            force[idx] += f
            torque += float(np.sum(p[idx, 0] * f[:, 1] - p[idx, 1] * f[:, 0]))
        pen_out = rr - (r_wall - r)
        outer = pen_out > 0.0
        p_wall = 0.0
        if outer.any():
            idx = np.nonzero(outer)[0]
            nv = -radial[idx]                        # wall pushes inward
            fn = np.maximum(kn * pen_out[idx], 0.0)
            force[idx] += fn[:, None] * nv
            vt_rad = np.einsum("ij,ij->i", v[idx], nv)
            vt_t = v[idx] - vt_rad[:, None] * nv
            sp = np.linalg.norm(vt_t, axis=1)
            ft = np.where(sp > 1e-14, case.mu_wall * fn, 0.0)
            dir_t = -vt_t / np.maximum(sp, 1e-30)[:, None]
            force[idx] += ft[:, None] * dir_t
            p_wall = float(np.sum(fn))
        return force, torque, p_wall

    samples: list[dict] = []
    traces = {"torque": [], "pressure": [], "r_wall": [], "time": []}
    strain = 0.0
    strain_limit = case.settle_strain + case.shear_strain
    step = 0
    sample_from = case.settle_strain + case.shear_strain - case.average_strain
    gamma_n = 2.0 * _damping(case.restitution) * np.sqrt(kn * 0.5 * float(mass.mean()))
    while strain < strain_limit:
        pair_idx, nvec, overlap, _ = contact_pairs(pos, rad)
        force = np.zeros_like(pos)
        if pair_idx.size:
            vrel = vel[pair_idx[:, 1]] - vel[pair_idx[:, 0]]
            vn = np.einsum("ij,ij->i", vrel, nvec)
            fn = np.maximum(kn * overlap - gamma_n * vn, 0.0)
            vt = vrel - vn[:, None] * nvec
            sp = np.linalg.norm(vt, axis=1)
            ft = np.where(sp > 1e-14, case.mu * fn, 0.0)
            fvec = fn[:, None] * nvec + ft[:, None] * (-vt / np.maximum(sp, 1e-30)[:, None])
            force += (accumulate_scatter(pair_idx[:, 1], fvec, n)
                      - accumulate_scatter(pair_idx[:, 0], fvec, n))
        wf, torque, p_wall = wall_forces(pos, vel, rad)
        force += wf
        vel += force / mass[:, None] * dt
        pos += vel * dt
        step += 1
        strain += case.wall_strain_rate() * dt
        if case.wall_servo:
            area_wall = 2.0 * math.pi * r_wall
            p_meas = p_wall / area_wall
            err = (p_meas - case.pressure) / max(case.pressure, 1.0)
            delta = float(np.clip(case.servo_relax * err, -1.0, 1.0))
            r_wall = float(np.clip(r_wall + delta * case.servo_span * case.d50,
                                   case.r_inner + 2.5 * case.d50, case.r_outer * 1.4))
            bin_edges = np.linspace(case.r_inner, r_wall, case.n_bins + 1)
        if history and step % 50 == 0:
            traces["torque"].append(torque)
            traces["pressure"].append(p_wall / (2.0 * math.pi * r_wall))
            traces["r_wall"].append(r_wall)
            traces["time"].append(step * dt)
        if strain >= sample_from and step % 20 == 0:
            samples.append(_coarse_grain(case, pos, vel, rad, mass, torque,
                                         p_wall, r_wall, bin_edges))
    profile = _merge_samples(samples)
    profile.update({
        "case": case,
        "dt": dt, "steps": step, "n_particles": n,
        "r_wall": r_wall, "omega": case.omega,
        "wall_strain_rate": case.wall_strain_rate(),
        "n_samples": len(samples),
        "d50": case.d50,
        "torque_mean": float(np.mean([s["torque"] for s in samples])) if samples else 0.0,
        "pressure_mean": float(np.mean([s["pressure"] for s in samples])) if samples else 0.0,
    })
    if history:
        profile["traces"] = traces
    return profile


def _clipped_disk_area(a, b, rho, r):
    """Area of a disk (centre radius ``rho``, radius ``r``) inside [a, b] (radii)."""
    def antiderivative(u):
        return (u * np.sqrt(np.maximum(r * r - u * u, 0.0))
                + r * r * np.arcsin(np.clip(u / r, -1.0, 1.0)))
    lo = np.clip(a - rho, -r, r)
    hi = np.clip(b - rho, -r, r)
    return antiderivative(hi) - antiderivative(lo)


def _coarse_grain(case, pos, vel, rad, mass, torque, p_wall, r_wall,
                  bin_edges) -> dict:
    """Coarse-grain one snapshot into radial bins.

    Stress is the Love-Weber virial sum *projected on each contact's own polar
    frame* before binning (projecting a Cartesian bin average onto one mean
    angle would cancel the shear component identically in an annulus), plus the
    kinetic (Reynolds) contribution evaluated in each particle's own frame.
    The solid fraction uses the exact clipped disk area per annulus bin, which
    is unbiased for bin widths below one diameter.
    """
    nb = case.n_bins
    rr = np.hypot(pos[:, 0], pos[:, 1])
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    area = math.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # ---- solid fraction (exact clipped disk areas) ----
    lo = bin_edges[:-1][None, :]
    hi = bin_edges[1:][None, :]
    rho = rr[:, None]
    rcol = rad[:, None]
    solid = np.sum(_clipped_disk_area(lo, hi, rho, rcol), axis=0)
    phi = solid / area

    # ---- virial stress, per-contact polar frame ----
    v_rr = np.zeros(3 * nb)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(2.0 * float(rad.max()) + 1e-12, output_type="ndarray")
    if pairs.size:
        d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
        dist = np.linalg.norm(d, axis=1)
        good = dist > 1e-15
        pairs, d, dist = pairs[good], d[good], dist[good]
        overlap = rad[pairs[:, 0]] + rad[pairs[:, 1]] - dist
        hit = overlap > 0.0
        pairs, d, dist, overlap = pairs[hit], d[hit], dist[hit], overlap[hit]
        nvec = d / dist[:, None]
        vrel = vel[pairs[:, 1]] - vel[pairs[:, 0]]
        vn = np.einsum("ij,ij->i", vrel, nvec)
        gamma_n = 2.0 * _damping(case.restitution) * np.sqrt(case.kn * 0.5 * mass.mean())
        fn = np.maximum(case.kn * overlap - gamma_n * vn, 0.0)
        vt = vrel - vn[:, None] * nvec
        sp = np.linalg.norm(vt, axis=1)
        ft = np.where(sp > 1e-14, case.mu * fn, 0.0)
        f_on_j = fn[:, None] * nvec + ft[:, None] * (-vt / np.maximum(sp, 1e-30)[:, None])
        f_on_i = -f_on_j                       # Love-Weber uses the force on i
        mid = 0.5 * (pos[pairs[:, 0]] + pos[pairs[:, 1]])
        ang = np.arctan2(mid[:, 1], mid[:, 0])
        er = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        et = np.stack([-np.sin(ang), np.cos(ang)], axis=1)
        lr = np.einsum("ij,ij->i", d, er)
        lt = np.einsum("ij,ij->i", d, et)
        fr = np.einsum("ij,ij->i", f_on_i, er)
        fth = np.einsum("ij,ij->i", f_on_i, et)
        b = np.clip(np.digitize(np.hypot(mid[:, 0], mid[:, 1]), bin_edges) - 1, 0, nb - 1)
        virial = np.stack([lr * fr, lt * fth, lr * fth])
        for k in range(3):
            v_rr[k * nb:(k + 1) * nb] += np.bincount(b, weights=virial[k],
                                                     minlength=nb) / np.maximum(area, 1e-30)
    s_rr, s_tt, s_rt = v_rr[0:nb], v_rr[nb:2 * nb], v_rr[2 * nb:3 * nb]

    # ---- kinetic (Reynolds) part in each particle's own frame ----
    which = np.clip(np.digitize(rr, bin_edges) - 1, 0, nb - 1)
    er_p = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    et_p = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    v_r = np.einsum("ij,ij->i", vel, er_p)
    v_t = np.einsum("ij,ij->i", vel, et_p)
    mean_r = np.zeros(nb)
    mean_t = np.zeros(nb)
    cnt = np.bincount(which, minlength=nb).astype(float)
    np.add.at(mean_r, which, v_r)
    np.add.at(mean_t, which, v_t)
    good = cnt > 0
    mean_r[good] /= cnt[good]
    mean_t[good] /= cnt[good]
    k_rr = np.bincount(which, weights=mass * (v_r - mean_r[which]) ** 2,
                       minlength=nb) / np.maximum(area, 1e-30)
    k_tt = np.bincount(which, weights=mass * (v_t - mean_t[which]) ** 2,
                       minlength=nb) / np.maximum(area, 1e-30)
    k_rt = np.bincount(which, weights=mass * (v_r - mean_r[which]) * (v_t - mean_t[which]),
                       minlength=nb) / np.maximum(area, 1e-30)

    sigma_rr = -s_rr + k_rr
    sigma_tt = -s_tt + k_tt
    sigma_rt = -s_rt + k_rt
    pressure = 0.5 * (sigma_rr + sigma_tt)
    tau = np.abs(sigma_rt)
    mu = np.divide(tau, np.maximum(pressure, 1e-300))

    vtheta_mean = mean_t.copy()
    vtheta_se = np.zeros(nb)
    for k in range(nb):
        sel = which == k
        if sel.sum() > 1:
            vtheta_se[k] = v_t[sel].std(ddof=1) / math.sqrt(sel.sum())
    gamma_dot = np.zeros(nb)
    for k in range(nb):
        k0, k1 = max(k - 1, 0), min(k + 1, nb - 1)
        if k1 > k0:
            dvdr = (vtheta_mean[k1] - vtheta_mean[k0]) / (centers[k1] - centers[k0])
            gamma_dot[k] = abs(dvdr - vtheta_mean[k] / centers[k])
        else:
            gamma_dot[k] = abs(-vtheta_mean[k] / centers[k])
    return {"r": centers, "phi": phi, "sigma_rr": sigma_rr, "sigma_tt": sigma_tt,
            "sigma_rt": sigma_rt, "pressure": pressure, "tau": tau, "mu": mu,
            "tau_torque": np.abs(torque) / (2.0 * math.pi * centers ** 2),
            "gamma_dot": gamma_dot, "v_theta": vtheta_mean,
            "v_theta_se": vtheta_se, "torque": torque,
            "pressure_wall": float(p_wall) / (2.0 * math.pi * r_wall),
            "r_wall": r_wall}


def _merge_samples(samples: list[dict]) -> dict:
    if not samples:
        raise CouetteError("no coarse-grained samples were collected")
    keys = [k for k, v in samples[0].items() if isinstance(v, np.ndarray)]
    out = {}
    for k in keys:
        stack = np.stack([s[k] for s in samples], axis=0)
        out[k] = stack.mean(axis=0)
        out[k + "_se"] = stack.std(axis=0, ddof=1) / math.sqrt(stack.shape[0]) \
            if stack.shape[0] > 1 else np.zeros_like(out[k])
    out["n_snapshots"] = len(samples)
    return out


__all__ = ["CouetteError", "CouetteCase", "simulate_couette"]