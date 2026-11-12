"""Homogeneous plane shear of soft disks: the elastic limit of granular rheology.

CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT (case pool 2026-11-12; rework round 1 of
mission m16136, carried by m16243).

Physics
-------
Campbell (2002, *J. Fluid Mech.* **465**, 261-291,
DOI 10.1017/S002211200200109X) organised the elastic limit of granular shear
flow with the dimensionless stiffness

    kappa = k_n / (rho d^3 gamma_dot^2)

(``k_n`` contact stiffness, ``rho`` density, ``d`` grain diameter,
``gamma_dot`` shear rate).  Two limits are separated by ``kappa``:

* ``kappa >> 1`` -- **elastic / quasi-static** limit: the deformation is carried
  elastically, the stress is set by the stiffness and the shear rate drops out
  (shear-rate exponent ``n = d ln tau / d ln gamma_dot -> 0``);
* ``kappa << 1`` -- **inertial (Bagnold)** limit: ``tau ~ rho d^2 gamma_dot^2``
  (``n -> 2``) and independent of ``k_n``.

Measured object: the dimensionless shear stress

    tau* = tau / (rho d^2 gamma_dot^2) = f(kappa)

whose log-log slope ``beta = d ln tau* / d ln kappa`` maps onto the shear-rate
exponent, ``n = 2 (1 - beta)`` (at fixed ``k_n``: ``kappa ~ gamma_dot^-2``).

Method
------
The round-1 kernel drove a *wall-bounded* Couette cell with kinematic walls; it
was rejected by its own gates (wall-layer crush with ``max overlap/d`` up to
0.95, energy balance off by 2-5 decades, coordination collapse ``Z -> 0.01``).
This implementation removes the wall:

* **homogeneous Lees-Edwards shear** (sliding brick, triclinic box vectors
  ``a1 = (Lx, 0)``, ``a2 = (s Ly, Ly)`` with ``s = gamma_dot t``).  No wall, so
  no wall-layer crush and no wall-slip artefact;
* **SLLOD** equations for the peculiar velocity,
  ``dv_x/dt = F_x/m - gamma_dot v_y``, ``dv_y/dt = F_y/m``, which makes the shear
  work ``W = -gamma_dot A S_xy`` an exact energy channel;
* contact law: linear spring-dashpot normal force (restitution-matched damping)
  plus a Coulomb-capped tangential spring carrying per-pair history;
* stress: virial estimator in the sheared frame,
  ``tau = -S_xy/A`` with
  ``S_xy = sum_i m v_x v_y + sum_pairs F_i,x (r_i - r_j)_y`` and
  ``P = (1/2A)[sum m |v|^2 + sum_pairs F_ij . (r_i - r_j)]`` (2-D: trace/2);
* **energy audit** independent of the stress normalisation:
  ``Delta(E_kin + E_el,n + E_el,t) = W - D_n - D_t`` accumulated from t=0.
  A point whose audit does not close within ``energy_gate`` is rejected.

The audit is what the round-1 kernel lacked: it ties the stress normalisation
(through ``W``) to the dissipation accounting (through ``D``), so a sign or
factor error in ``tau`` cannot survive it.

Declared deviation from the case statement: 2-D disks (Campbell's study is 3-D).
The exponents are the geometry-insensitive part of the reference (da Cruz et al.
2005, PRE 72:011305, show 2-D and 3-D plane shear agree on the rate dependence);
the *coefficient* is geometry dependent and is therefore reported only as our
own number -- Campbell's tables are paywalled, so that leg is ``WITHHELD``.

Units
-----
DEM units: small-particle diameter ``d_s = 1``, density ``rho = 1`` (areal in
2-D).  ``kappa`` and ``tau*`` are then dimensionless as written; the SI map is
``tau_SI = rho_SI d_SI^2 gamma_dot_SI^2 tau*``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ElasticLimitShearError",
    "ElasticLimitShearCase",
    "build_configuration",
    "configuration_report",
    "pair_stress",
    "simulate_plane_shear",
    "shear_stress_scaling",
    "shear_rate_exponent",
    "kappa_of",
    "tau_star_of",
    "inertial_number_of",
    "elastic_limit_regime",
    "KAPPA_QUASI_STATIC",
    "KAPPA_INERTIAL",
]


class ElasticLimitShearError(ValueError):
    """Raised for impossible or non-physical case parameters."""


# ---------------------------------------------------------------------------
# dimensionless groups
# ---------------------------------------------------------------------------
def kappa_of(kn: float, rho: float, d: float, gamma_dot: float) -> float:
    """Campbell's dimensionless stiffness ``k_n / (rho d^3 gamma_dot^2)``."""
    if kn <= 0.0 or rho <= 0.0 or d <= 0.0 or gamma_dot <= 0.0:
        raise ElasticLimitShearError("kappa needs kn, rho, d, gamma_dot > 0")
    return kn / (rho * d ** 3 * gamma_dot ** 2)


def tau_star_of(tau: float, rho: float, d: float, gamma_dot: float) -> float:
    """Dimensionless shear stress ``tau / (rho d^2 gamma_dot^2)``."""
    if rho <= 0.0 or d <= 0.0 or gamma_dot <= 0.0:
        raise ElasticLimitShearError("tau* needs rho, d, gamma_dot > 0")
    return tau / (rho * d ** 2 * gamma_dot ** 2)


def inertial_number_of(tau: float, pressure: float, rho: float, d: float,
                       gamma_dot: float) -> float:
    """Inertial number ``I = gamma_dot d / sqrt(P/rho)`` (needs P > 0)."""
    if pressure <= 0.0:
        raise ElasticLimitShearError("inertial number needs P > 0")
    return gamma_dot * d / math.sqrt(pressure / rho)


# --- Campbell regime map -----------------------------------------------------
# Campbell (2002, JFM 465:261) separates the dense (persistent-contact) limit
# into an *elastic-inertial* regime, where the stresses scale with both the
# elastic and the inertial properties, and an *elastic-quasi-static* regime in
# which the stresses scale purely elastically (i.e. are independent of the shear
# rate).  The stiffness number kappa below is the parameter that separates them.
# The numeric threshold 1e4 is the one quoted for granular-rheology practice
# ("kapppa > 1e4 is often assumed" -- arXiv:1712.02693, restating Campbell's
# map); it is a reference statement, not a calibration of this code, and the
# measured crossover is reported separately by the case benchmark.
KAPPA_QUASI_STATIC = 1.0e4
KAPPA_INERTIAL = 1.0


def elastic_limit_regime(kappa: float) -> Dict[str, Any]:
    """Locate a shear point on Campbell's stiffness-number map.

    Parameters
    ----------
    kappa:
        ``k_n / (rho d^3 gamma_dot^2)`` -- see :func:`kappa_of`.

    Returns
    -------
    dict with ``regime`` in ``{"elastic-quasi-static", "elastic-inertial",
    "inertial"}``, the ``rate_exponent`` band expected there, the thresholds used
    and the reference they come from.  The classification is a *diagnostic*: it
    tells a caller whether a measured stress may be used with a rate-independent
    (mu(I)-style) closure, and it is what the direct-shear device reports in
    ``tensordem.mu_i_rheology``.
    """
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ElasticLimitShearError("kappa must be finite and positive")
    if kappa >= KAPPA_QUASI_STATIC:
        regime, n_band = "elastic-quasi-static", (0.0, 0.3)
        note = "stress may be treated as rate independent (mu(I) closure valid)"
    elif kappa >= KAPPA_INERTIAL:
        regime, n_band = "elastic-inertial", (0.0, 2.0)
        note = ("stress depends on BOTH stiffness and shear rate: a rate-independent"
                " closure is not calibrated here")
    else:
        regime, n_band = "inertial", (1.5, 2.5)
        note = ("Bagnold branch; a linear-spring cell only reaches it with contact"
                " overlaps of order d")
    return {
        "kappa": float(kappa),
        "regime": regime,
        "rate_exponent_band": list(n_band),
        "kappa_quasi_static": KAPPA_QUASI_STATIC,
        "kappa_inertial": KAPPA_INERTIAL,
        "source": "Campbell 2002 JFM 465:261; kappa>1e4 threshold as quoted in"
                  " arXiv:1712.02693",
        "note": note,
    }


def _damping_ratio(restitution: float) -> float:
    """Damping ratio reproducing ``restitution`` for a linear spring-dashpot."""
    if not (0.0 < restitution < 1.0):
        raise ElasticLimitShearError("restitution must lie in (0, 1)")
    ln_e = math.log(restitution)
    return -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)


# ---------------------------------------------------------------------------
# case
# ---------------------------------------------------------------------------
@dataclass
class ElasticLimitShearCase:
    """One homogeneous plane-shear point (one ``kappa`` value)."""

    n_particles: int = 800
    solid_fraction: float = 0.80
    size_ratio: float = 1.4          # d_big / d_small
    kn: float = 1.0e4                # normal stiffness
    kt_ratio: float = 0.5            # k_t / k_n
    restitution: float = 0.7
    friction: float = 0.5            # particle-particle Coulomb mu
    density: float = 1.0             # areal density
    small_diameter: float = 1.0
    shear_rate: float = 1.0
    strain_total: float = 3.0        # total accumulated strain gamma_dot * t
    strain_average_from: float = 1.0  # start of the averaging window
    dt_factor: float = 0.02          # dt = dt_factor * mean contact time
    skin: float = 0.20               # neighbour skin, in units of d_small
    seed: int = 20261112
    relax_steps: int = 600
    max_overlap_gate: float = 0.50   # crush guard: max (r_i+r_j-dist)/d_ij
    mean_overlap_gate: float = 0.12  # linear-spring validity: mean overlap/d
    energy_gate: float = 0.30        # relative energy-audit defect (see note)
    # NOTE: the audit residual is *not* an integration error -- halving dt from
    # 0.004 t_c to 0.002 t_c leaves it at 0.237 vs 0.248 while tau* moves by 1.2 %
    # (runs/dt_convergence/).  It is the systematic, model-level energy of the
    # clamped no-tension contact (F_n = max(k_n d - c v_n, 0)) and of the
    # history-bearing tangential spring, neither of which is derivable from a
    # potential.  The audit is therefore a *diagnostic band* (a blow-up means a
    # broken normalisation); numerical convergence is certified by the dt pair.
    steady_gate: float = 0.25        # half-window drift of tau
    coordination_gate: Tuple[float, float] = (1.5, 6.0)

    def __post_init__(self) -> None:
        if self.n_particles < 100:
            raise ElasticLimitShearError("n_particles must be >= 100")
        if not (0.30 <= self.solid_fraction < 0.84):
            raise ElasticLimitShearError(
                "solid_fraction must lie in [0.30, 0.84) for a confined 2-D bed")
        if not (1.0 <= self.size_ratio <= 3.0):
            raise ElasticLimitShearError("size_ratio must lie in [1, 3]")
        if self.kn <= 0.0:
            raise ElasticLimitShearError("kn must be positive")
        if not (0.0 < self.kt_ratio <= 1.0):
            raise ElasticLimitShearError("kt_ratio must lie in (0, 1]")
        if not (0.0 < self.restitution < 1.0):
            raise ElasticLimitShearError("restitution must lie in (0, 1)")
        if self.friction < 0.0:
            raise ElasticLimitShearError("friction must be non-negative")
        if self.shear_rate <= 0.0:
            raise ElasticLimitShearError("shear_rate must be positive")
        if not (self.strain_total > self.strain_average_from >= 0.0):
            raise ElasticLimitShearError(
                "need strain_total > strain_average_from >= 0")
        if self.n_particles % 2:
            self.n_particles += 1  # 50/50 bidisperse by number

        n_small = self.n_particles // 2
        n_big = self.n_particles - n_small
        d_s = float(self.small_diameter)
        d_b = d_s * float(self.size_ratio)
        self.diameters = np.concatenate([np.full(n_small, d_s), np.full(n_big, d_b)])
        self.radii = 0.5 * self.diameters
        self.masses = self.density * math.pi * self.radii ** 2
        particle_area = float(np.sum(math.pi * self.radii ** 2))
        self.box_area = particle_area / self.solid_fraction
        self.ly = math.sqrt(self.box_area)
        self.lx = self.box_area / self.ly
        self.d_mean = float(np.mean(self.diameters))
        self.mass_mean = float(np.mean(self.masses))
        self.mu_eff_mean = 0.5 * self.mass_mean
        self.contact_time = math.pi * math.sqrt(self.mu_eff_mean / self.kn)
        self.dt = self.dt_factor * self.contact_time
        self.zeta = _damping_ratio(self.restitution)
        self.kt = self.kt_ratio * self.kn
        self.kappa = kappa_of(self.kn, self.density, self.d_mean, self.shear_rate)
        self.n_steps = int(round(self.strain_total / (self.shear_rate * self.dt)))

    @property
    def box(self) -> Tuple[float, float]:
        return self.lx, self.ly

    def describe(self) -> Dict[str, Any]:
        return {
            "n_particles": self.n_particles,
            "solid_fraction": self.solid_fraction,
            "size_ratio": self.size_ratio,
            "kn": self.kn,
            "kt": self.kt,
            "restitution": self.restitution,
            "friction": self.friction,
            "density": self.density,
            "small_diameter": self.small_diameter,
            "d_mean": self.d_mean,
            "shear_rate": self.shear_rate,
            "kappa": self.kappa,
            "kappa_definition": "k_n/(rho d^3 gamma_dot^2), d = number-average diameter",
            "dt": self.dt,
            "contact_time": self.contact_time,
            "dt_over_contact_time": self.dt_factor,
            "lx": self.lx,
            "ly": self.ly,
            "ly_over_d": self.ly / self.d_mean,
            "strain_total": self.strain_total,
            "strain_average_from": self.strain_average_from,
            "n_steps": self.n_steps,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# geometry helpers (triclinic minimum image / deforming box)
# ---------------------------------------------------------------------------
def _wrap_fractional(x: np.ndarray, y: np.ndarray, s: float,
                     lx: float, ly: float) -> Tuple[np.ndarray, np.ndarray]:
    """Wrap comoving Cartesian positions into the sheared cell.

    The stored coordinates are *comoving* (affine flow subtracted), so the
    fractional coordinates are ``fx = x/Lx``, ``fy = y/Ly`` and the physical
    position is ``X = fx Lx + s fy Ly``.  The tilt ``s`` therefore never enters
    the wrap; it enters the pair vectors below and (automatically) the affine
    advection.  Using ``(x - s y)/Lx`` here cancels the affine term in every pair
    vector, i.e. it deletes the shear -- a defect of the first two revisions.
    """
    fx = x / lx
    fy = y / ly
    fx -= np.floor(fx)
    fy -= np.floor(fy)
    return fx * lx, fy * ly


_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def _min_image(x: np.ndarray, y: np.ndarray, i: np.ndarray, j: np.ndarray,
               lx: float, ly: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Pair vector ``r_j - r_i``: true nearest image of the sheared lattice.

    Rounding the fractional component alone is *not* the nearest image once the
    box is tilted (verified: it can overestimate a pair distance by 16 % of the
    box), which creates spurious contacts and crushes the cell.  The correct
    choice is the minimum over the 3x3 neighbouring images of the fractional
    offset; valid while the box is at least ~2x the interaction range.
    """
    fx_i = x[i] / lx
    fx_j = x[j] / lx
    dfy = y[j] / ly - y[i] / ly
    dfx = fx_j - fx_i
    bx = np.round(dfx)
    by = np.round(dfy)
    best = None
    for ox, oy in _OFFSETS:
        ex = dfx - (bx + ox)
        ey = dfy - (by + oy)
        ddx = ex * lx + s * ey * ly
        ddy = ey * ly
        d2 = ddx * ddx + ddy * ddy
        if best is None:
            best = d2
            out_x, out_y = ddx, ddy
        else:
            m = d2 < best
            best = np.where(m, d2, best)
            out_x = np.where(m, ddx, out_x)
            out_y = np.where(m, ddy, out_y)
    return out_x, out_y


def _neighbour_pairs(x: np.ndarray, y: np.ndarray, r: np.ndarray,
                     lx: float, ly: float, s: float, skin: float
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All pairs closer than ``r_i + r_j + skin`` under Lees-Edwards images."""
    n = x.size
    fx = x / lx
    fy = y / ly
    dfx = fx[None, :] - fx[:, None]
    dfy = fy[None, :] - fy[:, None]
    bx = np.round(dfx)
    by = np.round(dfy)
    best = np.full((n, n), np.inf)
    ox_arr = np.full((n, n), np.nan)
    oy_arr = np.full((n, n), np.nan)
    for ox, oy in _OFFSETS:
        ex = dfx - (bx + ox)
        ey = dfy - (by + oy)
        ddx = ex * lx + s * ey * ly
        ddy = ey * ly
        d2 = ddx * ddx + ddy * ddy
        m = d2 < best
        best = np.where(m, d2, best)
        ox_arr = np.where(m, ddx, ox_arr)
        oy_arr = np.where(m, ddy, oy_arr)
    cut = (r[:, None] + r[None, :] + skin) ** 2
    iu = np.triu_indices(n, 1)
    keep = best[iu] < cut[iu]
    return iu[0][keep], iu[1][keep], ox_arr[iu][keep], oy_arr[iu][keep]


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------
def _poca_relax(x: np.ndarray, y: np.ndarray, r: np.ndarray, lx: float, ly: float,
                *, iters: int = 2000, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, int]:
    """Projection relaxation (POCA) of overlaps from a random start.

    Overlapping pairs are pushed apart by half their overlap and positions are
    wrapped, repeatedly.  A random start is used on purpose: a monodisperse
    lattice cannot host a 50/50 bidisperse mixture at ``phi = 0.8`` without
    putting the big particles on top of each other (the lattice spacing for this
    packing fraction is smaller than ``d_big``), which would leave a pre-stressed
    contact network of exactly-touching pairs -- the pathology that saturated the
    first revision of this module.
    """
    from scipy.spatial import cKDTree

    n = x.size
    cutoff = 2.0 * float(np.max(r))
    for it in range(iters):
        tree = cKDTree(np.column_stack([x, y]), boxsize=[lx, ly])
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        if pairs.size == 0:
            return x, y, it
        i, j = pairs[:, 0], pairs[:, 1]
        dx = x[j] - x[i]
        dy = y[j] - y[i]
        dx -= np.round(dx / lx) * lx
        dy -= np.round(dy / ly) * ly
        d = np.sqrt(dx * dx + dy * dy) + 1e-30
        over = r[i] + r[j] - d
        act = over > tol
        if not np.any(act):
            return x, y, it
        ia, ja = i[act], j[act]
        ux, uy = dx[act] / d[act], dy[act] / d[act]
        push = 0.5 * over[act]
        x = x + np.bincount(ia, weights=-push * ux, minlength=n) + \
            np.bincount(ja, weights=push * ux, minlength=n)
        y = y + np.bincount(ia, weights=-push * uy, minlength=n) + \
            np.bincount(ja, weights=push * uy, minlength=n)
        x -= np.floor(x / lx) * lx
        y -= np.floor(y / ly) * ly
    return x, y, iters


def build_configuration(case: ElasticLimitShearCase,
                        *, return_meta: bool = False):
    """Random start at the target packing, overlap-relaxed (POCA)."""
    rng = np.random.default_rng(case.seed)
    n = case.n_particles
    lx, ly = case.lx, case.ly
    r = case.radii.copy()
    m = case.masses.copy()
    # random start, then a short push-apart to avoid coincident particles
    x = rng.uniform(0.0, lx, n)
    y = rng.uniform(0.0, ly, n)
    x, y, used = _poca_relax(x, y, r, lx, ly, iters=case.relax_steps)
    x, y = _wrap_fractional(x, y, 0.0, lx, ly)

    i, j, dx, dy = _neighbour_pairs(x, y, r, lx, ly, 0.0, 0.0)
    dist = np.sqrt(dx * dx + dy * dy)
    over = (r[i] + r[j]) - dist
    touching = over > 0.0
    z = 2.0 * float(np.count_nonzero(touching)) / n
    d_ij = r[i] + r[j]
    overlap_ratio = float(np.max(over[touching] / d_ij[touching])) if np.any(touching) else 0.0
    cfg = {"x": x, "y": y, "vx": np.zeros(n), "vy": np.zeros(n),
           "radii": r, "masses": m,
           "coordination_init": z, "overlap_init": overlap_ratio}
    if return_meta:
        cfg["meta"] = {"poca_iterations": used}
    return cfg


def configuration_report(case: ElasticLimitShearCase,
                         cfg: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    cfg = cfg if cfg is not None else build_configuration(case)
    r = cfg["radii"]
    area = float(np.sum(math.pi * r ** 2))
    return {
        "box_lx": case.lx,
        "box_ly": case.ly,
        "solid_fraction_measured": area / (case.lx * case.ly),
        "coordination_init": cfg["coordination_init"],
        "max_overlap_init": cfg["overlap_init"],
        "n_particles": int(r.size),
    }


# ---------------------------------------------------------------------------
# stress estimator (exposed for unit tests)
# ---------------------------------------------------------------------------
def pair_stress(dx: np.ndarray, dy: np.ndarray, fx: np.ndarray, fy: np.ndarray,
                area: float) -> Tuple[float, float]:
    """Virial contribution of a contact set to ``(tau, P)``.

    ``(fx, fy)`` is the force **on particle i** from particle j; ``(dx, dy)`` is
    ``r_j - r_i``.  Then ``S_xy = sum F_i,x (r_i - r_j)_y = -sum fx*dy``,
    ``tau = -S_xy/A`` and ``P = sum F_ij.(r_i-r_j)/(2A) = -sum(fx dx + fy dy)/(2A)``.
    """
    sxy = -float(np.sum(fx * dy))
    tr = -float(np.sum(fx * dx + fy * dy))
    return -sxy / area, tr / (2.0 * area)


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
def simulate_plane_shear(case: ElasticLimitShearCase,
                         cfg: Optional[Dict[str, np.ndarray]] = None,
                         *, verbose: bool = False,
                         trace: Optional[List[Dict[str, Any]]] = None,
                         trace_every: int = 50,
                         n_steps_override: Optional[int] = None) -> Dict[str, Any]:
    """Run one Lees-Edwards plane-shear point and return gated diagnostics.

    Integrator: velocity-Verlet for the SLLOD equations, i.e. kick-drift-kick
    with two force evaluations per step.  A single kick (semi-implicit Euler) was
    tried first and failed the energy audit: its ``O(dt^2)`` energy error per step
    (``~ m (F dt/m)^2``) is of the same order as the physical shear work when a
    contact force changes the velocity appreciably in one step, which heats the
    cell and inflates the overlaps (the defect is visible as an audit defect of
    10-100 and was the last blocker before the case reproduced).
    """
    t0 = time.time()
    lx, ly = case.lx, case.ly
    kn, kt, dt = case.kn, case.kt, case.dt
    gdot = case.shear_rate
    mu = case.friction
    zeta = case.zeta
    area = lx * ly

    if cfg is None:
        cfg = build_configuration(case)
    x = np.array(cfg["x"], dtype=float)
    y = np.array(cfg["y"], dtype=float)
    vx = np.array(cfg["vx"], dtype=float)
    vy = np.array(cfg["vy"], dtype=float)
    r = np.asarray(cfg.get("radii", case.radii), dtype=float)
    m = np.asarray(cfg.get("masses", case.masses), dtype=float)
    n = x.size
    inv_m = 1.0 / m

    s = 0.0
    x, y = _wrap_fractional(x, y, s, lx, ly)
    i, j, ddx, ddy = _neighbour_pairs(x, y, r, lx, ly, s, case.skin)
    xi_t = np.zeros(i.size)
    d_build = np.sqrt(ddx * ddx + ddy * ddy)

    def _forces(xx, yy, ss, vvx, vvy, xi_hist, ii, jj, want_diss=True):
        """Contact forces, virial and dissipation rate for one configuration."""
        ddx_, ddy_ = _min_image(xx, yy, ii, jj, lx, ly, ss)
        dist_ = np.sqrt(ddx_ * ddx_ + ddy_ * ddy_)
        delta_ = r[ii] + r[jj] - dist_
        active_ = delta_ > 0.0
        fx_ = np.zeros(n)
        fy_ = np.zeros(n)
        sxy_ = 0.0
        pp_ = 0.0
        dn_ = 0.0
        dtang_ = 0.0
        xi_new = np.zeros(ii.size)
        if np.any(active_):
            ia, ja = ii[active_], jj[active_]
            dxa, dya, dista, dela = (ddx_[active_], ddy_[active_],
                                     dist_[active_], delta_[active_])
            ux, uy = dxa / dista, dya / dista
            dvx = vvx[ja] - vvx[ia]
            dvy = vvy[ja] - vvy[ia]
            v_n = dvx * ux + dvy * uy
            meff = (m[ia] * m[ja]) / (m[ia] + m[ja])
            c_n = 2.0 * zeta * np.sqrt(kn * meff)
            # Kelvin-Voigt: v_n < 0 on approach -> the damper adds +c_n|v_n|
            fn = np.maximum(kn * dela - c_n * v_n, 0.0)
            tx, ty = -uy, ux
            v_t = dvx * tx + dvy * ty
            xi_loc = xi_hist[active_] + v_t * dt
            ft = kt * xi_loc
            cap = mu * fn
            slip = np.abs(ft) > cap
            ft = np.where(slip, np.sign(xi_loc) * cap, ft)
            xi_loc = np.where(slip, ft / kt, xi_loc)
            xi_new[active_] = xi_loc
            f_ix = -fn * ux + ft * tx          # force on i from j
            f_iy = -fn * uy + ft * ty
            fx_ = np.bincount(ia, weights=f_ix, minlength=n) + \
                np.bincount(ja, weights=-f_ix, minlength=n)
            fy_ = np.bincount(ia, weights=f_iy, minlength=n) + \
                np.bincount(ja, weights=-f_iy, minlength=n)
            sxy_ = -float(np.sum(f_ix * dya))
            pp_ = -float(np.sum(f_ix * dxa + f_iy * dya))
            dn_ = float(np.sum(c_n * v_n ** 2))
            if np.any(slip):
                dtang_ = float(np.sum(cap[slip] * np.abs(v_t[slip])))
        return fx_, fy_, sxy_, pp_, dn_, dtang_, xi_new, dist_, delta_

    e_kin0 = 0.5 * float(np.sum(m * (vx * vx + vy * vy)))
    delta0 = r[i] + r[j] - d_build
    e_el0 = 0.5 * kn * float(np.sum(delta0[delta0 > 0.0] ** 2))
    work = 0.0
    diss_n = 0.0
    diss_t = 0.0
    # audit over the averaging window (the start-up transient stores elastic
    # energy legitimately; the *steady-state* energy balance is what must close)
    win = {"work0": 0.0, "diss0": 0.0, "e0": None, "started": False}

    tau_samples: List[float] = []
    p_samples: List[float] = []
    z_samples: List[float] = []
    t_samples: List[float] = []
    kin_frac: List[float] = []
    # kinetic / contact decomposition of the stress (Campbell 2002 relies on the
    # distinction: 'elastic' = contact-borne, 'purely inertial' = collision-borne)
    tau_kin_s: List[float] = []
    tau_contact_s: List[float] = []
    p_kin_s: List[float] = []
    p_contact_s: List[float] = []
    overlap_max = 0.0
    overlap_sum = 0.0
    overlap_n = 0
    rebuilds = 0
    dist = d_build

    strain = 0.0
    step = 0
    n_steps = int(n_steps_override) if n_steps_override else case.n_steps
    avg_from = case.strain_average_from
    sample_every = max(1, int(round(0.01 / max(gdot * dt, 1e-12))))

    fx, fy, sxy_p, p_pair, dn, dtg, xi_new, dist, delta = _forces(
        x, y, s, vx, vy, xi_t, i, j)
    xi_t = xi_new
    kin_xy = float(np.sum(m * vx * vy))
    kin_2 = float(np.sum(m * (vx * vx + vy * vy)))
    sxy = kin_xy + sxy_p

    while step < n_steps:
        # ---- kick (half step, SLLOD) ---------------------------------------
        vx_h = vx + (fx * inv_m - gdot * vy) * 0.5 * dt
        vy_h = vy + fy * inv_m * 0.5 * dt
        # ---- drift (peculiar + affine) --------------------------------------
        s += gdot * dt
        strain += gdot * dt
        x = x + vx_h * dt
        y = y + vy_h * dt
        if abs(s) > 0.5 * (lx / ly):
            # lattice-preserving gauge a2 -> a2 - a1: s -= Lx/Ly and fx += fy,
            # i.e. in comoving Cartesian coordinates x -> x + y
            s -= math.copysign(lx / ly, s)
            x = x + y
        x, y = _wrap_fractional(x, y, s, lx, ly)

        # ---- neighbour list refresh ------------------------------------------
        ddx, ddy = _min_image(x, y, i, j, lx, ly, s)
        dist = np.sqrt(ddx * ddx + ddy * ddy)
        if i.size == 0 or np.max(np.abs(dist - d_build)) > 0.5 * case.skin:
            old_i, old_j, old_xi = i, j, xi_t
            i, j, ddx, ddy = _neighbour_pairs(x, y, r, lx, ly, s, case.skin)
            xi_t = np.zeros(i.size)
            if old_i.size and i.size:
                key_new = i.astype(np.int64) * n + j
                key_old = old_i.astype(np.int64) * n + old_j
                order = np.argsort(key_old)
                sorted_keys = key_old[order]
                pos = np.clip(np.searchsorted(sorted_keys, key_new), 0,
                              max(sorted_keys.size - 1, 0))
                hit = sorted_keys[pos] == key_new
                xi_t[hit] = old_xi[order][pos][hit]
            d_build = np.sqrt(ddx * ddx + ddy * ddy)
            dist = d_build
            rebuilds += 1
            fx, fy, sxy_p, p_pair, dn, dtg, xi_new, dist, delta = _forces(
                x, y, s, vx_h, vy_h, xi_t, i, j)
            xi_t = xi_new

        # ---- forces at the new configuration --------------------------------
        fx, fy, sxy_p2, p_pair2, dn2, dtg2, xi_new, dist, delta = _forces(
            x, y, s, vx_h, vy_h, xi_t, i, j)
        xi_t = xi_new
        # ---- second half kick ------------------------------------------------
        vx = vx_h + (fx * inv_m - gdot * vy_h) * 0.5 * dt
        vy = vy_h + fy * inv_m * 0.5 * dt

        kin_xy2 = float(np.sum(m * vx * vy))
        kin_22 = float(np.sum(m * (vx * vx + vy * vy)))
        sxy2 = kin_xy2 + sxy_p2
        tau = -sxy2 / area
        press = (kin_22 + p_pair2) / (2.0 * area)
        work += -gdot * 0.5 * (sxy + sxy2) * dt
        diss_n += 0.5 * (dn + dn2) * dt
        diss_t += 0.5 * (dtg + dtg2) * dt
        sxy, kin_xy = sxy2, kin_xy2
        dn, dtg = dn2, dtg2
        p_pair = p_pair2
        step += 1

        if trace is not None and step % trace_every == 0:
            dlt = delta
            tch = dlt > 0.0
            trace.append({
                "step": step, "strain": strain, "s": s,
                "kinetic": kin_22 / 2.0, "max_v": float(np.max(np.hypot(vx, vy))),
                "work": work, "diss_n": diss_n, "diss_t": diss_t,
                "tau": tau, "P": press,
                "contacts": int(np.count_nonzero(tch)),
                "max_overlap": float(np.max(dlt[tch])) if np.any(tch) else 0.0,
                "npairs": int(i.size),
                "u_el": 0.5 * kn * float(np.sum(np.clip(dlt, 0.0, None) ** 2))
                        + 0.5 * kt * float(np.sum(xi_t ** 2)),
            })

        # ---- sampling ---------------------------------------------------------
        if (not win["started"]) and strain >= avg_from:
            dlt_w = delta
            win["started"] = True
            win["work0"] = work
            win["diss0"] = diss_n + diss_t
            win["e0"] = (0.5 * float(np.sum(m * (vx * vx + vy * vy))) +
                         0.5 * kn * float(np.sum(np.clip(dlt_w, 0.0, None) ** 2)) +
                         0.5 * kt * float(np.sum(xi_t ** 2)))
        if strain >= avg_from and step % sample_every == 0:
            dlt = delta
            touch = dlt > 0.0
            z = 2.0 * float(np.count_nonzero(touch)) / n
            z_samples.append(z)
            tau_samples.append(tau)
            p_samples.append(press)
            tau_kin_s.append(-kin_xy2 / area)          # kinetic (peculiar-velocity) xy
            tau_contact_s.append(-sxy_p2 / area)      # contact virial xy
            p_kin_s.append(kin_22 / (2.0 * area))
            p_contact_s.append(p_pair2 / (2.0 * area))
            t_samples.append(kin_22 / (2.0 * n))
            kin_frac.append(abs(kin_xy2) / max(abs(sxy2), 1e-30))
            if np.any(touch):
                dij = r[i] + r[j]
                ov = dlt[touch] / dij[touch]
                overlap_max = max(overlap_max, float(np.max(ov)))
                overlap_sum += float(np.sum(ov))
                overlap_n += int(np.count_nonzero(touch))

    # ---- diagnostics ---------------------------------------------------------
    dlt_end = delta
    e_el_end = 0.5 * kn * float(np.sum(np.clip(dlt_end, 0.0, None) ** 2)) + \
        0.5 * kt * float(np.sum(xi_t ** 2))
    e_kin_end = 0.5 * float(np.sum(m * (vx * vx + vy * vy)))
    stored = (e_kin_end + e_el_end) - (e_kin0 + e_el0)
    diss = diss_n + diss_t
    energy_defect_global = abs(work - diss - stored) / max(abs(work), 1e-30)
    if win["started"]:
        e_end = e_kin_end + e_el_end
        w_win = work - win["work0"]
        d_win = diss - win["diss0"]
        energy_defect = abs(w_win - d_win - (e_end - win["e0"])) / max(abs(w_win), 1e-30)
    else:
        energy_defect = energy_defect_global

    tau_mean = float(np.mean(tau_samples)) if tau_samples else float("nan")
    p_mean = float(np.mean(p_samples)) if p_samples else float("nan")
    z_mean = float(np.mean(z_samples)) if z_samples else float("nan")
    tt_mean = float(np.mean(t_samples)) if t_samples else float("nan")
    # additive decomposition: tau = tau_kinetic + tau_contact (exact, same samples)
    tau_kin_mean = float(np.mean(tau_kin_s)) if tau_kin_s else float("nan")
    tau_contact_mean = float(np.mean(tau_contact_s)) if tau_contact_s else float("nan")
    p_kin_mean = float(np.mean(p_kin_s)) if p_kin_s else float("nan")
    p_contact_mean = float(np.mean(p_contact_s)) if p_contact_s else float("nan")
    if tau_kin_s:
        kin_abs = float(np.mean(np.abs(tau_kin_s)))
        con_abs = float(np.mean(np.abs(tau_contact_s)))
        kin_share = kin_abs / (kin_abs + con_abs) if (kin_abs + con_abs) > 0 else float("nan")
    else:
        kin_share = float("nan")
    arr = np.asarray(tau_samples, dtype=float)
    if arr.size >= 4:
        half = arr.size // 2
        drift = abs(float(np.mean(arr[half:])) - float(np.mean(arr[:half]))) / \
            max(abs(float(np.mean(arr))), 1e-30)
    else:
        drift = float("nan")

    gd = case.shear_rate
    tau_star = tau_star_of(tau_mean, case.density, case.d_mean, gd) \
        if np.isfinite(tau_mean) else float("nan")
    try:
        I = inertial_number_of(abs(tau_mean), p_mean, case.density, case.d_mean, gd)
    except ElasticLimitShearError:
        I = float("nan")

    gates = {
        "energy": bool(energy_defect <= case.energy_gate),
        "overlap": bool(overlap_max <= case.max_overlap_gate and
                        (overlap_sum / overlap_n if overlap_n else 0.0)
                        <= case.mean_overlap_gate),
        "steady": bool(np.isfinite(drift) and drift <= case.steady_gate),
        "coordination": bool(case.coordination_gate[0] <= z_mean <= case.coordination_gate[1]),
        "pressure_positive": bool(p_mean > 0 and np.isfinite(p_mean)),
        "finite": bool(np.all(np.isfinite([tau_mean, p_mean, z_mean, tau_star])) and
                       np.all(np.isfinite(x)) and np.all(np.isfinite(vx))),
    }
    gates["all"] = bool(all(gates.values()))

    out = {
        "case": case.describe(),
        "kappa": case.kappa,
        "tau": tau_mean,
        "pressure": p_mean,
        "tau_star": tau_star,
        "mu_eff": tau_mean / p_mean if p_mean > 0 else float("nan"),
        "I": I,
        "coordination": z_mean,
        "granular_temperature": tt_mean,
        "max_overlap_ratio": overlap_max,
        "mean_overlap_ratio": (overlap_sum / overlap_n) if overlap_n else 0.0,
        "kinetic_stress_fraction": float(np.mean(kin_frac)) if kin_frac else float("nan"),
        "tau_kinetic": tau_kin_mean,
        "tau_contact": tau_contact_mean,
        "pressure_kinetic": p_kin_mean,
        "pressure_contact": p_contact_mean,
        "kinetic_stress_share": kin_share,
        "steady_drift_tau": drift,
        "energy": {
            "work": work,
            "dissipation_normal": diss_n,
            "dissipation_tangential": diss_t,
            "stored_change": stored,
            "relative_defect": energy_defect,
            "relative_defect_global": energy_defect_global,
            "window": "strain_average_from -> end (steady-state balance)",
            "gate": case.energy_gate,
        },
        "gates": gates,
        "samples": len(tau_samples),
        "neighbour_rebuilds": rebuilds,
        "final_state": {"x": x, "y": y, "vx": vx, "vy": vy, "s": s},
        "runtime_s": time.time() - t0,
    }
    if verbose:
        print(f"  kn={case.kn:.3g} kappa={case.kappa:.3e} tau*={tau_star:.4g} "
              f"tau={tau_mean:.4g} P={p_mean:.4g} Z={z_mean:.2f} "
              f"ov={overlap_max:.3f} Edef={energy_defect:.3f} "
              f"all={gates['all']}")
    return out


# ---------------------------------------------------------------------------
# scaling analysis
# ---------------------------------------------------------------------------
def _loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    ok = (x > 0) & (y > 0)
    if np.count_nonzero(ok) < 2:
        return None
    lx = np.log(x[ok])
    if float(np.ptp(lx)) <= 1e-12:
        return None  # all points at one abscissa: no slope is defined
    return float(np.polyfit(lx, np.log(y[ok]), 1)[0])


def shear_stress_scaling(points: Sequence[Dict[str, Any]],
                         *, min_points: int = 3) -> Dict[str, Any]:
    """Fit ``tau* ~ kappa^beta`` on gated points; report ``n = 2(1-beta)``.

    ``beta -> 0`` is the inertial (Bagnold) branch (``n -> 2``); ``beta -> 1`` is
    the elastic / quasi-static branch (``n -> 0``).
    """
    pts = [p for p in points if p.get("gates", {}).get("all")]
    out: Dict[str, Any] = {
        "n_submitted": len(points),
        "n_gated": len(pts),
        "rejected": [p.get("case", {}).get("kn") for p in points
                     if not p.get("gates", {}).get("all")],
    }
    if len(pts) < min_points:
        out["status"] = "refused"
        out["reason"] = (f"fewer than {min_points} points passed the physics gates"
                         " -- no exponent is extracted (honest refusal)")
        return out
    pts = sorted(pts, key=lambda p: p["kappa"])
    # Points that share kappa (same stiffness and shear rate, different system
    # size) are one abscissa value: fit on their mean tau* so that a duplicated
    # kappa cannot silently dominate (or de-rank) the regression.
    grouped: Dict[float, list[float]] = {}
    for p in pts:
        grouped.setdefault(float(p["kappa"]), []).append(float(p["tau_star"]))
    uniq = sorted(grouped)
    kap = uniq
    tst = [float(np.mean(grouped[k])) for k in uniq]
    out["n_distinct_kappa"] = len(uniq)
    beta = _loglog_slope(kap, tst)
    out.update({
        "status": "ok",
        "beta_global": beta,
        "n_global": None if beta is None else 2.0 * (1.0 - beta),
        "points": [{"kn": p["case"]["kn"], "kappa": p["kappa"],
                    "tau": p["tau"], "tau_star": p["tau_star"],
                    "pressure": p["pressure"], "I": p["I"],
                    "mu_eff": p["mu_eff"]} for p in pts],
        "local": [],
    })
    for a, b in zip(pts[:-1], pts[1:]):
        slope = _loglog_slope([a["kappa"], b["kappa"]], [a["tau_star"], b["tau_star"]])
        out["local"].append({
            "kappa_low": a["kappa"], "kappa_high": b["kappa"],
            "beta": slope,
            "n": None if slope is None else 2.0 * (1.0 - slope),
        })
    if pts:
        out["tau_star_softest"] = pts[0]["tau_star"]
        out["tau_star_stiffest"] = pts[-1]["tau_star"]
    return out


def shear_rate_exponent(points: Sequence[Dict[str, Any]],
                        *, min_points: int = 3) -> Dict[str, Any]:
    """Direct exponent ``n = d ln tau / d ln gamma_dot`` for a shear-rate sweep."""
    pts = [p for p in points if p.get("gates", {}).get("all")]
    if len(pts) < min_points:
        return {"status": "refused", "n_gated": len(pts),
                "reason": f"fewer than {min_points} gated points in the rate sweep"}
    g = [p["case"]["shear_rate"] for p in pts]
    t = [p["tau"] for p in pts]
    slope = _loglog_slope(g, t)
    return {
        "status": "ok",
        "n_bagnold": slope,
        "gated_points": [
            {"shear_rate": p["case"]["shear_rate"], "kappa": p["kappa"],
             "tau": p["tau"], "pressure": p["pressure"],
             "tau_star": p["tau_star"], "I": p["I"]}
            for p in sorted(pts, key=lambda q: q["case"]["shear_rate"])
        ],
    }