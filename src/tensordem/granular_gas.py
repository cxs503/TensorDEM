"""Homogeneous cooling of a granular gas -- Haff's law verification kernel.

Physics background
------------------
A *granular gas* is a collection of particles at number density ``n`` and
negligible volume fraction that interact only through instantaneous inelastic
contacts.  Because every collision removes ``(1 - e**2)`` of the kinetic energy
carried by the normal component of the relative velocity, an isolated gas cools
down without any boundary or gravity: this is the cleanest *multi-body*
dissipation benchmark in granular mechanics.

Haff's law (Haff, J. Fluid Mech. 132 (1983) 401; Jenkins & Savage, J. Fluid
Mech. 130 (1983) 187) states that the granular temperature

    theta = <v**2> / d          (d = 2 or 3 spatial dimensions)

obeys, in the dilute (Boltzmann, pair correlation chi = 1) limit,

    d(theta)/dt = -K * theta**(3/2)
    =>   theta(t)   = theta0 / (1 + t/tau)**2 ,      tau = 2 / (K sqrt(theta0))
    =>   theta**(-1/2)  is *linear* in t with slope K/2 .

The coefficient ``K`` follows from the collision frequency and the mean energy
loss per collision.  Both are evaluated below in closed form for a Maxwellian
velocity distribution (see :func:`theory_cooling_coefficient`); the derivation
is reproduced in that function's docstring because the whole point of this
module is that the constant is *predicted*, not fitted.

What this module is for
-----------------------
The repository already verifies the *single-impact* restitution mapping
(:mod:`tensordem.contact_restitution`).  Nothing however verified what happens
when thousands of such contacts act together: whether collision statistics and
energy loss are coupled correctly, i.e. whether the code reproduces
``theta**(3/2)`` cooling, its ``-2`` decay exponent, its ``n`` / ``(1-e**2)``
scaling and the absolute Boltzmann prefactor.  That is the gap this kernel
closes.  It is deliberately a *self-contained* soft-sphere integrator so the
benchmark does not depend on gravity, walls or a container geometry.

Model (all quantities non-dimensional: sigma = 1, m = 1, theta0 = 1)
--------------------------------------------------------------------
* monodisperse frictionless disks / spheres, diameter ``sigma = 1``, mass 1;
* normal contact ``F_n = max(k_n * delta - gamma_n * v_n, 0)`` (identical force
  law and clamping convention to :mod:`tensordem.drum_segregation.kernel2d`
  and :mod:`tensordem.dem3d`);
* ``gamma_n`` from :func:`tensordem.contact_tangential.restitution_to_damping`
  with ``m_eff = 1/2`` (the shipped mapping -- not re-derived here);
* periodic cubic/quadratic box, velocity-Verlet integration.

Because the contact law is the shipped one and the anchors are external
(Haff 1983 / Boltzmann kinetic theory), a pass certifies the *algorithmic*
coupling rather than a self-consistent re-implementation.

Known modelling caveats (reported, not hidden)
----------------------------------------------
* frictionless: tangential friction/rotation would add a second dissipation
  channel that the Boltzmann prediction above does not contain;
* soft contacts: the finite contact duration ``t_c`` makes the collision
  effectively "smeared"; runs therefore keep ``t_c / t_collision << 1`` and a
  stiffness refinement is part of the acceptance protocol;
* finite density: ``chi > 1`` increases the collision rate; the benchmark
  extrapolates ``K/n`` to ``phi -> 0`` instead of ignoring it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from .contact_tangential import restitution_to_damping

__all__ = [
    "GasConfig",
    "CoolingRun",
    "make_gas",
    "simulate_cooling",
    "theory_cooling_coefficient",
    "theory_collision_frequency",
    "fit_haff_law",
    "local_cooling_coefficient",
    "maxwell_kurtosis",
    "two_body_restitution",
]


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GasConfig:
    """Parameter record for one homogeneous-cooling run (sigma = m = 1 units)."""

    dim: int = 2                  # 2 or 3
    n_particles: int = 2500
    packing: float = 0.02         # area/volume fraction phi
    restitution: float = 0.90     # target coefficient of restitution
    kn: float = 200.0             # normal stiffness [N/m] (non-dimensional)
    dt_per_contact: float = 20.0  # dt = t_contact / dt_per_contact
    theta0: float = 1.0           # initial granular temperature
    steps: int = 400_000          # hard cap; the run also stops on cooling
    sample_every: int = 200       # sampling interval (steps)
    stop_ratio: float = 16.0      # stop when theta0/theta reaches this
    skin: float = 0.25            # neighbour skin (fraction of sigma)
    rebuild: int = 10             # neighbour-list rebuild interval
    seed: int = 0
    record_velocity: bool = True  # keep a velocity sample for Maxwell checks

    # --- derived -------------------------------------------------------
    @property
    def sigma(self) -> float:
        return 1.0

    @property
    def mass(self) -> float:
        return 1.0

    @property
    def m_eff(self) -> float:
        return 0.5 * self.mass

    @property
    def zeta(self) -> float:
        """Damping ratio implied by the shipped restitution mapping."""
        if self.restitution >= 1.0:
            return 0.0
        gamma = restitution_to_damping(self.restitution, self.kn, self.m_eff)
        return gamma / (2.0 * math.sqrt(self.kn * self.m_eff))

    @property
    def gamma_n(self) -> float:
        return restitution_to_damping(self.restitution, self.kn, self.m_eff)

    @property
    def contact_time(self) -> float:
        """Duration of one damped contact, ``t_c = pi / omega_d``."""
        omega_d = math.sqrt(self.kn / self.m_eff) * math.sqrt(max(1.0 - self.zeta ** 2, 0.0))
        return math.pi / omega_d

    @property
    def dt(self) -> float:
        return self.contact_time / self.dt_per_contact

    @property
    def volume(self) -> float:
        if self.dim == 2:
            return self.n_particles * math.pi * (0.5 * self.sigma) ** 2 / self.packing
        return self.n_particles * math.pi / 6.0 * self.sigma ** 3 / self.packing

    @property
    def box(self) -> float:
        return self.volume ** (1.0 / self.dim)

    @property
    def number_density(self) -> float:
        return self.n_particles / self.volume

    @property
    def cross_section(self) -> float:
        """Hard-core cross-section measure: 2*sigma in 2-D, pi*sigma**2 in 3-D."""
        if self.dim == 2:
            return 2.0 * self.sigma
        return math.pi * self.sigma ** 2

    @property
    def mean_relative_speed_factor(self) -> float:
        """``E|g| / sqrt(theta)`` for a Maxwellian (see theory docstring)."""
        return 2.0 * math.gamma(0.5 * (self.dim + 1.0)) / math.gamma(0.5 * self.dim)

    @property
    def collision_time(self) -> float:
        """Mean time between collisions of one particle at ``theta = 1``.

        ``omega_c = n * S * E|g|`` (Enskog collision frequency at chi = 1), so
        ``tau_coll = 1 / (n * S * factor * sqrt(theta0))``.
        """
        omega_c = self.number_density * self.cross_section * self.mean_relative_speed_factor
        return 1.0 / (omega_c * math.sqrt(self.theta0))

    @property
    def knudsen(self) -> float:
        """Contact duration over mean collision time (small = hard-sphere limit)."""
        return self.contact_time / self.collision_time

    def with_(self, **kw) -> "GasConfig":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# theory: Boltzmann cooling coefficient (no fitted constants)
# ---------------------------------------------------------------------------
def theory_cooling_coefficient(dim: int, number_density: float, restitution: float,
                               sigma: float = 1.0) -> float:
    """Dilute-gas cooling coefficient ``K`` in ``dtheta/dt = -K theta**(3/2)``.

    Derivation (all steps independent of the code under test)
    --------------------------------------------------------
    * granular temperature ``theta = <v**2>/dim``; a Maxwellian therefore has
      per-component variance ``theta``, so the relative velocity has component
      variance ``2 theta`` and, with ``|g|`` the magnitude of ``v_i - v_j``,

          ``E[|g|**k] = (2 theta)**(k/2) * 2**(k/2) * Gamma((d+k)/2) / Gamma(d/2)``

      (check k=1: d=2 -> sqrt(pi*theta), d=3 -> 4 sqrt(theta/pi), the textbook
      Maxwellian mean relative speeds).
    * hard-core collision rate per unit volume (chi = 1, monodisperse):

          ``R = (1/2) n**2 * S * E|g|``,  S = 2 sigma (d=2), pi sigma**2 (d=3),
          i.e. per-particle Enskog frequency ``omega_c = n S E|g|``, which
          reduces to the textbook ``24 phi g0 sqrt(theta/pi)/sigma`` in 3-D.
    * energy loss per collision, equal masses ``m``: ``(m/4)(1-e**2) g_n**2``
      with ``g_n`` the normal component along the line of centres.
    * two averages must be taken over *collisions*, not over the gas:
      (i) the rate is weighted by the relative speed, so ``E[|g| g_n**2]``
      enters rather than ``E|g| E[g_n**2]``; (ii) the impact geometry is
      weighted by the cross-section element, which for hard particles in d
      dimensions gives ``<cos**2 phi>_coll = 2/(d+1)`` rather than the naive
      isotropic ``1/d``.  For a fixed ``g`` the angular weight is
      ``cos(phi) sin**(d-2)(phi) dphi`` (d=2: ``cos**3/cos`` = 2/3,
      d=3: ``cos**3 sin/cos sin`` = 1/2, d=4: 2/5).
    * hence the dissipation per unit volume is
      ``(1/2) n**2 S (m/4)(1-e**2) E[|g|**3] * 2/(d+1)`` with
      ``E[|g|**3] = 8 theta**(3/2) Gamma((d+3)/2)/Gamma(d/2)``, and dividing by
      the energy density ``n (d/2) m theta`` gives

          ``dtheta/dt = -K theta**(3/2)``,
          ``K = (4 / (d (d+1))) n S (1-e**2) Gamma((d+3)/2)/Gamma(d/2)``

      i.e. ``K(2-D) = sqrt(pi) n sigma (1-e**2) = 1.7725 n sigma (1-e**2)`` and
      ``K(3-D) = 2.3633 n sigma**2 (1-e**2)``.
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    if not (0.0 < restitution <= 1.0):
        raise ValueError("restitution must be in (0, 1]")
    if number_density <= 0 or sigma <= 0:
        raise ValueError("number_density and sigma must be positive")
    s = 2.0 * sigma if dim == 2 else math.pi * sigma ** 2
    f = math.gamma(0.5 * (dim + 3.0)) / math.gamma(0.5 * dim)
    return (4.0 / (dim * (dim + 1.0))) * number_density * s * (1.0 - restitution ** 2) * f


def theory_collision_frequency(dim: int, number_density: float, theta: float,
                               sigma: float = 1.0) -> float:
    """Per-particle Enskog collision frequency ``omega_c = n S E|g|``."""
    s = 2.0 * sigma if dim == 2 else math.pi * sigma ** 2
    f = 2.0 * math.gamma(0.5 * (dim + 1.0)) / math.gamma(0.5 * dim)
    return number_density * s * f * math.sqrt(theta)


def two_body_restitution(config: GasConfig, speed: float = 1.0) -> float:
    """Re-bound coefficient measured with the *same* force law and integrator.

    Two particles approach head-on along x inside the periodic box; the ratio
    ``|v_rel|_after / |v_rel|_before`` is the coefficient of restitution that
    the shipped clamped spring-dashpot actually delivers at this ``k_n``/``dt``
    (the unclamped analytic mapping is only exact in the continuum limit).
    """
    cfg = config.with_(n_particles=2, packing=1.0e-6, steps=20_000,
                       sample_every=1, stop_ratio=1.0e9, record_velocity=False)
    box = cfg.box
    if box < 8.0 * cfg.sigma:          # keep the pair far from its images
        box = 8.0 * cfg.sigma
    pos = np.full((2, cfg.dim), 0.5 * box)     # same transverse coordinate
    pos[0, 0] = 0.5 * box - 0.6 * cfg.sigma    # head-on along x
    pos[1, 0] = 0.5 * box + 0.6 * cfg.sigma
    vel = np.zeros((2, cfg.dim))
    vel[0, 0] = 0.5 * speed
    vel[1, 0] = -0.5 * speed

    rel_before = speed
    rel_after = None
    pair_list = np.array([[0, 1]], dtype=int)   # the two particles are the only pair
    in_contact = False
    for _ in range(cfg.steps):
        pos, vel = _verlet_step(pos, vel, cfg, box, pair_list)
        _, n_f = _pair_forces(pos, vel, cfg, box, pair_list)
        if n_f > 0:
            in_contact = True
        elif in_contact:                        # contact opened: impact finished
            rel_after = abs(float(vel[0, 0] - vel[1, 0]))
            break
    if rel_after is None:
        raise RuntimeError("two-body impact did not rebound; check dt/stiffness")
    return rel_after / rel_before


def maxwell_kurtosis(vel: np.ndarray) -> float:
    """Excess kurtosis of the per-component velocity distribution.

    Zero for a Maxwellian; the estimator bias for ``n`` samples is
    ``-6/(n+1)`` (subtracted below) so the returned value is unbiased in the
    Gaussian limit.
    """
    v = np.asarray(vel, dtype=float).ravel()
    n = v.size
    if n < 20:
        raise ValueError("need at least 20 velocity samples")
    mean = v.mean()
    var = v.var()
    if var <= 0.0:
        return 0.0
    m4 = np.mean((v - mean) ** 4)
    return float(m4 / var ** 2 - 3.0 + 6.0 / (n + 1.0))


# ---------------------------------------------------------------------------
# gas construction and integration
# ---------------------------------------------------------------------------
def make_gas(config: GasConfig):
    """Random non-overlapping configuration + Maxwellian velocities at theta0."""
    rng = np.random.default_rng(config.seed)
    box = config.box
    n, dim = config.n_particles, config.dim
    pos = np.empty((n, dim))
    # RSA with a dense-target guard: at phi <= 0.1 rejection is cheap.
    tree = None
    placed = 0
    attempts = 0
    while placed < n:
        attempts += 1
        if attempts > 200_000:
            raise RuntimeError("RSA insertion failed; lower the packing fraction")
        cand = rng.random(dim) * box
        if placed == 0:
            pos[0] = cand
            placed = 1
            continue
        have = pos[:placed]
        d = have - cand
        d -= box * np.round(d / box)
        if np.min(np.einsum("ij,ij->i", d, d)) < (0.999 * config.sigma) ** 2:
            continue
        pos[placed] = cand
        placed += 1
    tree = cKDTree(pos, boxsize=box)      # noqa: F841  (kept for debugging)
    vel = rng.normal(0.0, math.sqrt(config.theta0), size=(n, dim))
    vel -= vel.mean(axis=0)                # no net momentum
    return pos, vel, box


def _minimum_image(disp: np.ndarray, box: float) -> np.ndarray:
    return disp - box * np.round(disp / box)


def _pair_forces(pos: np.ndarray, vel: np.ndarray, config: GasConfig, box: float,
                 pairs: np.ndarray):
    """Normal contact forces; returns (acc, n_contacts)."""
    if pairs.size == 0:
        return np.zeros_like(pos), 0
    i, j = pairs[:, 0], pairs[:, 1]
    disp = _minimum_image(pos[i] - pos[j], box)
    r2 = np.einsum("ij,ij->i", disp, disp)
    active = r2 < (config.sigma + config.skin) ** 2
    if not np.any(active):
        return np.zeros_like(pos), 0
    i, j, disp, r2 = i[active], j[active], disp[active], r2[active]
    r = np.sqrt(r2)
    overlap = config.sigma - r
    contact = overlap > 0.0
    if not np.any(contact):
        return np.zeros_like(pos), 0
    i, j, disp, r, overlap = i[contact], j[contact], disp[contact], r[contact], overlap[contact]
    n_hat = disp / r[:, None]
    v_rel = vel[i] - vel[j]
    v_n = np.einsum("ij,ij->i", v_rel, n_hat)
    f_mag = config.kn * overlap - config.gamma_n * v_n
    f_mag = np.maximum(f_mag, 0.0)          # compression-only (shipped clamp)
    f = f_mag[:, None] * n_hat
    acc = np.zeros_like(pos)
    np.add.at(acc, i, f / config.mass)
    np.add.at(acc, j, -f / config.mass)
    return acc, int(np.count_nonzero(f_mag))


def _contact_mask(pos: np.ndarray, pairs: np.ndarray, box: float,
                  config: GasConfig) -> np.ndarray:
    """Boolean contact state per pair (used to count *new* impacts only)."""
    if pairs.size == 0:
        return np.zeros(0, dtype=bool)
    disp = _minimum_image(pos[pairs[:, 0]] - pos[pairs[:, 1]], box)
    r = np.sqrt(np.einsum("ij,ij->i", disp, disp))
    return (config.sigma - r) > 0.0


def _potential_energy(pos: np.ndarray, pairs: np.ndarray, box: float,
                      config: GasConfig) -> float:
    """Elastic potential energy stored in the (compression-only) contacts."""
    if pairs.size == 0:
        return 0.0
    disp = _minimum_image(pos[pairs[:, 0]] - pos[pairs[:, 1]], box)
    r = np.sqrt(np.einsum("ij,ij->i", disp, disp))
    overlap = np.maximum(config.sigma - r, 0.0)
    return 0.5 * config.kn * float(np.sum(overlap ** 2))


def _neighbour_pairs(pos: np.ndarray, box: float, r_cut: float) -> np.ndarray:
    tree = cKDTree(pos, boxsize=box)
    pairs = tree.query_pairs(r_cut, output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=int)
    return pairs.astype(int)


def _verlet_step(pos: np.ndarray, vel: np.ndarray, config: GasConfig, box: float,
                 pairs: np.ndarray):
    """One velocity-Verlet step with the pair list held fixed for this step."""
    acc, _ = _pair_forces(pos, vel, config, box, pairs)
    vel = vel + 0.5 * config.dt * acc
    pos = pos + config.dt * vel
    pos = np.mod(pos, box)
    acc_new, _ = _pair_forces(pos, vel, config, box, pairs)
    vel = vel + 0.5 * config.dt * acc_new
    return pos, vel


@dataclass
class CoolingRun:
    """Result record of one homogeneous-cooling run."""

    config: GasConfig
    times: np.ndarray
    theta: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    n_contacts: np.ndarray
    impacts: np.ndarray
    theta0_measured: float
    max_overlap: float
    velocity_sample: np.ndarray | None = None
    stopped_early: bool = False

    @property
    def fit(self) -> dict:
        return fit_haff_law(self.times, self.theta)

    @property
    def kurtosis(self) -> float:
        if self.velocity_sample is None:
            raise ValueError("run was made with record_velocity=False")
        return maxwell_kurtosis(self.velocity_sample)


def simulate_cooling(config: GasConfig) -> CoolingRun:
    """Integrate the cooling gas and return ``theta(t)`` plus diagnostics."""
    pos, vel, box = make_gas(config)
    r_cut = config.sigma + config.skin
    if 2.0 * r_cut > box:
        raise ValueError("box too small for the neighbour cutoff; lower the packing")
    theta0 = float(np.mean(np.einsum("ij,ij->i", vel, vel)) / config.dim)

    pairs = _neighbour_pairs(pos, box, r_cut)
    prev_contact = _contact_mask(pos, pairs, box, config)

    times, thetas, kinetics, potentials, contacts, impacts = [], [], [], [], [], []
    n_impacts_total = 0
    max_overlap = 0.0
    velocity_sample = None
    stopped_early = True

    for step in range(config.steps):
        if step % config.rebuild == 0 and step > 0:
            pairs = _neighbour_pairs(pos, box, r_cut)
            # carry the contact state over instead of zeroing it: zeroing would
            # re-count every ongoing contact as a fresh impact once per rebuild
            prev_contact = _contact_mask(pos, pairs, box, config)
        # overlap bookkeeping for reporting
        if len(pairs):
            disp = _minimum_image(pos[pairs[:, 0]] - pos[pairs[:, 1]], box)
            r2 = np.einsum("ij,ij->i", disp, disp)
            overlap = config.sigma - np.sqrt(r2)
            max_overlap = max(max_overlap, float(overlap.max()) if overlap.size else 0.0)
            contact_now = overlap > 0.0
            n_impacts_total += int(np.count_nonzero(contact_now & ~prev_contact))
            prev_contact = contact_now
            n_contacts = int(np.count_nonzero(contact_now))
        else:
            n_contacts = 0
        pos, vel = _verlet_step(pos, vel, config, box, pairs)

        if step % config.sample_every == 0:
            v2 = np.einsum("ij,ij->i", vel, vel)
            theta = float(np.mean(v2) / config.dim)
            times.append(step * config.dt)
            thetas.append(theta)
            kinetics.append(0.5 * config.mass * float(np.sum(v2)))
            potentials.append(_potential_energy(pos, pairs, box, config))
            contacts.append(n_contacts)
            impacts.append(n_impacts_total)
            if theta <= theta0 / config.stop_ratio:
                stopped_early = False
                break
    else:
        stopped_early = True

    if config.record_velocity:
        # final-state velocities are used for the distribution (quasi-static
        # cooling keeps the shape Maxwellian at all times)
        velocity_sample = vel.copy()

    return CoolingRun(
        config=config,
        times=np.asarray(times),
        theta=np.asarray(thetas),
        kinetic=np.asarray(kinetics),
        potential=np.asarray(potentials),
        n_contacts=np.asarray(contacts),
        impacts=np.asarray(impacts),
        theta0_measured=theta0,
        max_overlap=max_overlap if isinstance(max_overlap, float) else 0.0,
        velocity_sample=velocity_sample,
        stopped_early=stopped_early,
    )


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def fit_haff_law(times: np.ndarray, theta: np.ndarray, start_frac: float = 0.10,
                 end_frac: float = 0.95) -> dict:
    """Fit ``theta**(-1/2) = a + b t`` over ``theta0/theta in [1, stop]``.

    Haff's law predicts ``b = K/2`` and ``theta ~ t**(-2)``; the equivalent
    log-log exponent is reported as ``exponent`` (should be -2).
    """
    t = np.asarray(times, dtype=float)
    th = np.asarray(theta, dtype=float)
    if t.size < 10:
        raise ValueError("too few samples to fit")
    inv = th ** -0.5
    lo = int(start_frac * t.size)
    hi = max(lo + 5, int(end_frac * t.size))
    tt, yy = t[lo:hi], inv[lo:hi]
    slope, intercept = np.polyfit(tt, yy, 1)
    pred = slope * tt + intercept
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    theta_fit0 = 1.0 / intercept ** 2 if intercept > 0 else float("nan")

    # Apparent log-log exponent of theta itself (approaches -2 only once
    # t >> tau = 2/(K sqrt(theta0)); reported for transparency, the criterion
    # used for the pass/fail decision is the linearity of theta**(-1/2)).
    late = max(5, int(0.5 * tt.size))
    e_slope = float(np.polyfit(np.log(tt[-late:]), np.log(yy[-late:]), 1)[0])
    return {
        "K_meas": float(2.0 * slope),
        "slope": float(slope),
        "intercept": float(intercept),
        "theta_fit0": float(theta_fit0),
        "r2": float(r2),
        "exponent": -2.0 * e_slope,          # theta^{-1/2} slope -> theta exponent
        "n_points": int(tt.size),
        "window": [float(tt[0]), float(tt[-1])],
    }


def local_cooling_coefficient(times: np.ndarray, theta: np.ndarray) -> dict:
    """Instantaneous ``K(t) = -(dtheta/dt)/theta**(3/2)`` from a smooth fit.

    A drifting ``K`` means the run is outside Haff's scaling regime; reporting
    it is required so the fit above cannot hide a transient.
    """
    t = np.asarray(times, dtype=float)
    th = np.asarray(theta, dtype=float)
    if t.size < 20:
        raise ValueError("too few samples")
    inv = th ** -0.5
    slope = np.polyfit(t, inv, 1)[0]        # global linear reference
    # piecewise slopes on four equal windows
    edges = np.linspace(0, t.size, 5).astype(int)
    vals = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 5:
            vals.append(float("nan"))
            continue
        vals.append(2.0 * float(np.polyfit(t[a:b], inv[a:b], 1)[0]))
    return {"K_global": 2.0 * float(slope), "K_windows": vals}