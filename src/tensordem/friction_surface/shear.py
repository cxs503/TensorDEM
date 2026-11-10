"""Periodic planar shear cell: direct-shear strength and Mohr-Coulomb envelope.

Why this cell and not a rigid split box
--------------------------------------
The first implementation in this node was a Casagrande-style split box
(:class:`tensordem.friction_surface.kernel.SplitBox`) driven at constant volume.
That setup never reached a steady state: the dense specimen jammed against the
rigid side walls, the shear traction grew monotonically (measured 18.6 kPa at
``sigma = 5 kPa``, i.e. an apparent friction angle of 75 deg), and the number of
contacts kept climbing.  The run is recorded as a falsified protocol in
``docs/P152W1-N04_friction_contact_surface.md``; no result of it is used.

The cell shipped here is the standard *periodic planar shear cell*
(da Cruz et al. 2005): periodic in ``x`` and ``y``, two rough platens (bottom
fixed, top translating at ``v_shear`` and force-controlled to the target normal
stress ``sigma``).  Periodicity removes the side-wall jam, the specimen is free
to dilate (the top platen follows), and the stress is homogenised over the cell
with the Love-Weber contact-stress definition:

    sigma_ij = (1/V) sum_c f_i^c l_j^c,   f^c = force on particle i from j,
                                          l^c = x_j - x_i

reported compression-positive (``sigma_zz > 0`` in compression, ``tau = sigma_xz``).
The specimen-level friction angle is ``phi = arctan(tau / sigma_zz)`` — the
"macroscopic friction" of the packing, a quantity that is *not* fixed by the
interparticle coefficient (that is exactly why it has to be measured).

Peak / residual and the Mohr-Coulomb criterion
----------------------------------------------
``tau(gamma)`` is binned in engineering shear strain ``gamma = u/H``; the peak is
the maximum of the binned history and the residual is the mean of the last 20 %
of the bins.  Strengths at several ``sigma`` are least-squares fitted with
``tau = c + sigma tan(phi)`` and the worst per-point relative deviation from the
fitted envelope is the gated number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .kernel import ContactModel, PairState, PlaneWall, _scatter_add, \
    apply_walls, integrate, pair_forces

__all__ = [
    "PeriodicShearCase",
    "build_specimen",
    "stress_tensor",
    "bin_history",
    "simulate_shear",
    "mohr_coulomb_fit",
]

_FAR = 1.0e9


@dataclass(frozen=True)
class PeriodicShearCase:
    """Periodic planar shear cell (SI)."""

    sigma: float = 5000.0
    mu: float = 0.5
    mu_rolling: float = 0.1
    mu_wall: float | None = None
    mu_rolling_wall: float | None = None
    radius: float = 1.5e-3
    density: float = 2500.0
    kn: float = 3.0e3
    kt_ratio: float = 0.5
    kr_ratio: float = 0.1
    restitution: float = 0.3
    ct_ratio: float = 0.0
    cap_total: bool = True
    gravity: float = 9.81
    nx: int = 6                       # hex lattice cells in x
    ny: int = 6                       # hex lattice rows in y (even AND a multiple of 3)
    nz: int = 4                       # A-B layer pairs in z
    lattice_gap: float = 0.05         # in-plane spacing = 2r (1 + gap) -> phi_HCP = 0.64
    jitter: float = 0.02
    settle_time: float = 0.45
    shear_velocity: float = 4.0e-2
    shear_strain_target: float = 0.50
    sample_every: int = 100
    bin_width: float = 0.02
    servo_gain: float = 0.5
    impose_platen_velocity: bool = True
    platen_max_speed: float = 0.05
    dt: float | None = None
    seed: int = 152041

    @property
    def model(self) -> ContactModel:
        return ContactModel(
            radius=self.radius, density=self.density, kn=self.kn,
            kt_ratio=self.kt_ratio, kr_ratio=self.kr_ratio, mu=self.mu,
            mu_rolling=self.mu_rolling, restitution=self.restitution,
            ct_ratio=self.ct_ratio, cap_total=self.cap_total,
            gravity=self.gravity,
        )

    @property
    def spacing(self) -> float:
        return 2.0 * self.radius * (1.0 + self.lattice_gap)

    @property
    def width(self) -> float:
        return self.nx * self.spacing

    @property
    def depth(self) -> float:
        return self.ny * self.spacing * math.sqrt(3.0) / 2.0

    @property
    def layer_pitch(self) -> float:
        return self.spacing * math.sqrt(2.0 / 3.0)

    @property
    def height(self) -> float:
        return 2 * self.nz * self.layer_pitch

    @property
    def area(self) -> float:
        return self.width * self.depth

    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height


def build_specimen(case: PeriodicShearCase) -> np.ndarray:
    """HCP (A-B stacked) specimen in a *commensurate* periodic cell.

    ``nx x ny x 2 nz`` particles on a hexagonal lattice with in-plane pitch
    ``a = 2r(1 + gap)``, A/B layer offsets of ``(a/2, dy/3)`` (hollow centroid)
    and layer
    pitch ``a sqrt(2/3)``; the cell is periodic in all three directions by
    construction (``ny`` even, offsets are half-lattice), so no pair is closer
    than ``a`` and no wrap-around overlap is possible.  ``gap = 0.05`` gives
    ``phi = 0.7405/(1.05)^3 = 0.64``, an RCP-density dense specimen.
    """
    a = case.spacing
    dy = a * math.sqrt(3.0) / 2.0
    c = case.layer_pitch
    rng = np.random.default_rng(case.seed)
    pts = []
    for k in range(2 * case.nz):
        z = (k + 0.5) * c
        parity = k % 2
        for iy in range(case.ny):
            # B layers sit over the *hollow centroid* of the A triangle, i.e. a
            # third of a row pitch, not half (half leaves the pair 0.89 a apart
            # and the specimen starts interpenetrating).
            y = (iy + parity / 3.0) * dy
            for ix in range(case.nx):
                x = (ix + 0.5 * (iy % 2) + 0.5 * parity) * a
                pts.append((x % case.width, y % case.depth, z))
    pos = np.asarray(pts, dtype=float)
    if case.jitter:
        pos = pos + rng.standard_normal(pos.shape) * case.jitter * case.radius
    pos[:, 0] %= case.width
    pos[:, 1] %= case.depth
    return pos


def solid_fraction(case: PeriodicShearCase, pos: np.ndarray) -> float:
    return len(pos) * 4.0 / 3.0 * math.pi * case.radius ** 3 / case.volume


def _min_image_xy(dvec: np.ndarray, width: float, depth: float) -> np.ndarray:
    out = dvec.copy()
    out[:, 0] -= width * np.round(out[:, 0] / width)
    out[:, 1] -= depth * np.round(out[:, 1] / depth)
    return out


def build_pairs_periodic(pos, radius, width, depth):
    n = len(pos)
    tree = cKDTree(pos, boxsize=(width, depth, _FAR))
    pairs = tree.query_pairs(2.0 * radius, output_type="ndarray")
    if pairs.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    iu = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    ju = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    keys = iu * n + ju
    order = np.argsort(keys)
    return iu[order], ju[order], keys[order]


def pair_forces_periodic(pos, vel, omega, iu, ju, ps, model, dt, width, depth,
                        contacts=None):
    """Same laws as the open kernel, with minimum-image pair separations."""
    n = len(pos)
    force = np.zeros((n, 3))
    torque = np.zeros((n, 3))
    if contacts is not None:
        contacts.clear()
    if iu.size == 0:
        return force, torque
    r = model.radius
    dvec = _min_image_xy(pos[ju] - pos[iu], width, depth)
    dist = np.linalg.norm(dvec, axis=1)
    overlap = 2.0 * r - dist
    active = overlap > 0.0
    if not active.any():
        ps.carry(np.zeros(0, dtype=np.int64))
        return force, torque
    iu_a, ju_a = iu[active], ju[active]
    dvec_a, dist_a, overlap_a = dvec[active], dist[active], overlap[active]
    nvec = dvec_a / np.maximum(dist_a, 1e-300)[:, None]
    v_rel = vel[ju_a] - vel[iu_a]
    v_n = (v_rel * nvec).sum(axis=1)
    from .laws import normal_force, rolling_torque, tangential_force
    f_n = normal_force(overlap_a, v_n, model.kn, model.cn)
    surf_i = np.cross(omega[iu_a], nvec * r)
    surf_j = np.cross(omega[ju_a], -nvec * r)
    v_t = (v_rel - v_n[:, None] * nvec) + (surf_j - surf_i)
    ps.carry(iu_a * n + ju_a)
    ps.tang = ps.tang + v_t * dt
    f_t = tangential_force(ps.tang, v_t, f_n, kt=model.kt, mu=model.mu,
                           ct=model.ct, cap_total=model.cap_total)
    f_total = f_n[:, None] * nvec + f_t
    if contacts is not None:
        contacts["i"] = iu_a
        contacts["j"] = ju_a
        contacts["normal"] = nvec
        contacts["f_n"] = f_n
        contacts["f_total"] = f_total
        contacts["branch"] = dvec_a
    force = _scatter_add(iu_a, -f_total, n) + _scatter_add(ju_a, f_total, n)
    t_i = np.cross(-nvec * r, f_t)
    torque = _scatter_add(iu_a, t_i, n) + _scatter_add(ju_a, t_i, n)
    if model.mu_rolling > 0.0:
        spin = omega[ju_a] - omega[iu_a]
        ps.roll = ps.roll + spin * dt
        m_r = rolling_torque(ps.roll, spin, f_n, kr=model.kr,
                             mu_r=model.mu_rolling, r_eff=r, cr=model.cr,
                             cap_total=model.cap_total)
        torque += _scatter_add(iu_a, -m_r, n) + _scatter_add(ju_a, m_r, n)
    return force, torque


def stress_tensor(contacts: dict, volume: float) -> np.ndarray:
    """Love-Weber contact stress (compression positive, Pa)."""
    if not contacts or "f_total" not in contacts:
        return np.zeros((3, 3))
    f = contacts["f_total"]          # force on j from i
    l = contacts["branch"]           # x_j - x_i (minimum image)
    # stress from the (i,j) pair: symmetric contribution f (x) l
    s = (f[:, :, None] * l[:, None, :]).sum(axis=0)
    s = 0.5 * (s + s.T)
    # f is the force on j from i and l = x_j - x_i, so a compressive pair
    # (f along +l) gives a positive diagonal: compression positive, no sign flip.
    return s / volume


def bin_history(strains, taus, sigmas, *, width: float = 0.02) -> dict:
    g = np.asarray(strains, dtype=float)
    t = np.asarray(taus, dtype=float)
    sg = np.asarray(sigmas, dtype=float)
    if g.size == 0:
        return {"strain": np.zeros(0), "tau": np.zeros(0), "sigma": np.zeros(0)}
    idx = np.floor(g / width).astype(int)
    gs, ts, ss = [], [], []
    for k in np.unique(idx):
        m = idx == k
        if m.sum() < 2:
            continue
        gs.append(float(g[m].mean()))
        ts.append(float(t[m].mean()))
        ss.append(float(sg[m].mean()))
    return {"strain": np.asarray(gs), "tau": np.asarray(ts),
            "sigma": np.asarray(ss), "bin_width": width}


def simulate_shear(case: PeriodicShearCase) -> dict:
    """Run one periodic shear cell at the case's normal stress."""
    model = case.model
    dt = case.dt or model.timestep
    pos = build_specimen(case)
    n = len(pos)
    phi_initial = solid_fraction(case, pos)
    vel = np.zeros_like(pos)
    omega = np.zeros_like(pos)
    ps = PairState()
    u_max = case.shear_strain_target * case.height

    def displacement(t):
        return case.shear_velocity * max(0.0, t - case.settle_time)

    wall_mu = case.mu_wall if case.mu_wall is not None else case.mu
    wall_mu_r = (case.mu_rolling_wall if case.mu_rolling_wall is not None
                 else case.mu_rolling)
    floor = PlaneWall((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), mu=wall_mu,
                      mu_rolling=wall_mu_r)
    z_top = float(pos[:, 2].max() + model.radius)
    z_top_init = z_top
    lid = PlaneWall((0.0, 0.0, z_top), (0.0, 0.0, -1.0), mu=wall_mu,
                    mu_rolling=wall_mu_r)
    target_force = case.sigma * case.area
    contacts: dict = {}
    n_steps = int((case.settle_time + u_max / case.shear_velocity) / dt)
    samples = []
    for step in range(n_steps):
        t = step * dt
        shearing = t > case.settle_time
        lid.velocity = np.array([case.shear_velocity if shearing else 0.0, 0.0, 0.0])
        iu, ju, keys = build_pairs_periodic(pos, model.radius, case.width, case.depth)
        f, tq = pair_forces_periodic(pos, vel, omega, iu, ju, ps, model, dt,
                                     case.width, case.depth, contacts=contacts)
        fw, tw = apply_walls([floor, lid], pos, vel, omega, model, dt, t)
        f += fw
        f[:, 2] -= model.mass * model.gravity
        pos, vel, omega = integrate(pos, vel, omega, f, tq + tw, model, dt)
        pos[:, 0] %= case.width
        pos[:, 1] %= case.depth
        if case.impose_platen_velocity:
            # Kinematic platen: the particles the lid touches are carried at the
            # lid velocity.  Without this the lid drives the specimen only
            # through the *elastic* tangential spring of the rough wall, whose
            # force is tiny compared with the bulk strength: the cell measured
            # tau/sigma = 0.04 (phi ~ 3 deg) instead of the material value.
            mask = lid.contact_geometry(pos, model.radius, t)[0]
            if shearing and mask.any():
                vel[mask, 0] = case.shear_velocity
                vel[mask, 2] = np.minimum(vel[mask, 2], 0.0)
        # platen servo: hold sigma*A. z_top is the true lid height.
        lid_normal = -lid.traction[2]
        dz = -case.servo_gain * (target_force - lid_normal) / (model.kn * 8.0)
        dz = float(np.clip(dz, -case.platen_max_speed * dt, case.platen_max_speed * dt))
        z_top = float(np.clip(z_top + dz, 0.5 * case.height, 2.0 * case.height))
        lid.point[2] = z_top
        if step % case.sample_every == 0:
            st = stress_tensor(contacts, case.volume)
            samples.append({
                "t": t,
                "strain": displacement(t) / case.height,
                "tau": float(st[0, 2]),
                "sigma": float(st[2, 2]),
                "sigma_xy": float(st[0, 1]),
                "lid_height": z_top,
                "lid_normal_force": lid_normal,
                "n_contacts": int(len(contacts.get("i", []))),
                "traction_wall": float(lid.traction[0]) / case.area,
            })
    hist = [s for s in samples if s["strain"] > 1e-9]
    if not hist:
        raise RuntimeError("no shearing samples recorded")
    strains = np.array([s["strain"] for s in hist])
    taus = np.abs(np.array([s["tau"] for s in hist]))
    sigmas = np.array([s["sigma"] for s in hist])
    binned = bin_history(strains, taus, sigmas, width=case.bin_width)
    if binned["tau"].size < 4:
        raise RuntimeError("shear history too short to bin")
    peak = float(np.nanmax(binned["tau"]))
    n_res = max(2, int(0.2 * binned["tau"].size))
    resid = float(np.nanmean(binned["tau"][-n_res:]))
    sig_res = float(np.nanmean(binned["sigma"][-n_res:]))
    lid_heights = np.array([s["lid_height"] for s in hist if s["strain"] > 0.05])
    return {
        "sigma_applied": case.sigma,
        "sigma_measured": sig_res,
        "sigma_measured_mean": float(np.nanmean(sigmas)),
        "tau_peak": peak,
        "tau_residual": resid,
        "phi_peak_deg_from_applied": math.degrees(math.atan(peak / case.sigma)),
        "phi_residual_deg_from_applied": math.degrees(math.atan(resid / case.sigma)),
        "phi_peak_deg_measured": math.degrees(math.atan(peak / max(sig_res, 1e-12))),
        "phi_residual_deg_measured": math.degrees(math.atan(resid / max(sig_res, 1e-12))),
        "strains": strains.tolist(),
        "taus": taus.tolist(),
        "sigmas": sigmas.tolist(),
        "binned_strain": binned["strain"].tolist(),
        "binned_tau": binned["tau"].tolist(),
        "binned_sigma": binned["sigma"].tolist(),
        "dilatancy_m": float(lid_heights.max() - lid_heights.min()) if lid_heights.size else 0.0,
        "lid_height_start": z_top_init,
        "lid_height_end": float(z_top),
        "n_particles": n,
        "solid_fraction_initial": phi_initial,
        "solid_fraction_final": solid_fraction(case, pos),
        "dt": dt,
        "steps": n_steps,
        "shear_velocity": case.shear_velocity,
        "mu": case.mu,
        "mu_rolling": case.mu_rolling,
    }


def mohr_coulomb_fit(sigmas, strengths) -> dict:
    """Least-squares Mohr-Coulomb fit ``tau = c + sigma tan(phi)``."""
    s = np.asarray(sigmas, dtype=float)
    tau = np.asarray(strengths, dtype=float)
    A = np.vstack([s, np.ones_like(s)]).T
    (tan_phi, c), *_ = np.linalg.lstsq(A, tau, rcond=None)
    pred = A @ np.array([tan_phi, c])
    rel = np.abs(pred - tau) / np.maximum(np.abs(tau), 1e-30)
    ss_res = float(np.sum((tau - pred) ** 2))
    ss_tot = float(np.sum((tau - tau.mean()) ** 2)) if tau.size > 1 else 0.0
    return {
        "tan_phi": float(tan_phi),
        "phi_deg": math.degrees(math.atan(float(tan_phi))),
        "cohesion_pa": float(c),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "max_relative_deviation": float(np.nanmax(rel)),
        "per_point": [
            {"sigma": float(si), "tau_measured": float(ti),
             "tau_mohr_coulomb": float(pi), "relative_deviation": float(ri)}
            for si, ti, pi, ri in zip(s, tau, pred, rel)
        ],
        "n_points": int(s.size),
    }