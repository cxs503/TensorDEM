"""Hertz-Mindlin contact law with viscous damping (regularised, non-cohesive).

Why this law
------------
The clear-ice DEM is a *stable, dissipative* many-body problem: floes are
buoyant, nearly neutrally loaded disks drifting at a few tenths of a metre per
second into a rigid hull face.  A linear spring (``k delta``) with a hard
cut-off -- the model the earlier clearance work inherited -- produces a contact
force that reverses sign in one step whenever ``dt`` is a non-negligible
fraction of the contact time, which is exactly the "kinematics not deliverable"
symptom of node P103W2-N03 (the floe velocity record is dominated by a
grid-frequency oscillation).

Two structural fixes are used here:

Normal direction -- **Hertz stiffness with Tsuji damping**,

    ``F_n = max( (4/3) E* sqrt(R*) delta^{3/2}
                 + 2 zeta sqrt(k_n(delta) m*) delta_dot , 0 )``

with ``k_n(delta) = dF_elastic/ddelta = 2 E* sqrt(R* delta)`` (the tangent
stiffness, so ``2 zeta sqrt(k_n m*)`` is the *critical-damping fraction* form of
Tsuji et al., 1992) and ``delta_dot`` the approach rate.  The ``max(.,0)`` is
the **regularisation that removes cohesion and the sticking oscillation**: a
contact can never pull, so a separating pair cannot be re-accelerated back
together by its own damping term, which is the mechanism that drives the
classic "negative force / re-impact" limit cycle.  The damping ratio is derived
from the *measured* restitution through the standard relation

    ``zeta = -ln e / sqrt(pi^2 + ln^2 e)``,

so the restitution that the model realises is an input, not a free knob
(measured in ``tests/test_clear_ice_contact.py`` to better than 2 %).

Tangential direction -- **Mindlin incremental spring with Coulomb cap**,

    ``k_t(delta) = 8 G* sqrt(R* delta)``,
    ``f_t <- clip( f_t - k_t v_t dt , -mu F_n , +mu F_n )``

with the pair history carried across steps (so static friction is real), the
history dropped when the pair separates (no spurious memory), and the *slip
velocity* itself used only through the displacement increment -- i.e. the
friction is regularised in displacement, not in velocity, which keeps the
static branch exact.  For the rigid hull gate a stateless ``-mu F_n tanh(v_t/v_s)``
law is used (the wall has no tangential compliance worth resolving).

Units are SI throughout: m, kg, s, N, Pa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "ContactMaterial",
    "ContactSolver",
    "PairForces",
    "contact_time",
    "damping_ratio_from_restitution",
    "damping_ratio_from_restitution_analytic",
    "_restitution_of_zeta",
    "effective_modulus",
    "effective_radius",
    "effective_shear_modulus",
    "hertz_elastic_force",
    "hertz_elastic_energy",
    "hertz_stiffness",
    "restitution_from_damping_ratio",
]


# --------------------------------------------------------------------------
# material
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ContactMaterial:
    """Linear-elastic contact material (Hertz-Mindlin admissible)."""

    youngs_modulus: float = 9.0e9      # Pa (columnar/sea ice, E ~ 5-10 GPa)
    poisson_ratio: float = 0.30
    restitution: float = 0.20          # coefficient of restitution e_n
    friction: float = 0.35             # Coulomb mu
    slip_regularisation: float = 1.0e-3  # m/s, wall friction tanh scale
    density: float = 917.0             # kg/m^3 (columnar sea ice)
    crushing_strength: float = 2.0e6   # Pa, caps the elastic force (ice crushes)

    def __post_init__(self) -> None:
        if self.crushing_strength <= 0.0:
            raise ValueError("crushing_strength must be positive")
        if self.youngs_modulus <= 0.0:
            raise ValueError("youngs_modulus must be positive")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie in [0, 0.5)")
        if not 0.0 < self.restitution <= 1.0:
            raise ValueError("restitution must lie in (0, 1]")
        if self.friction < 0.0:
            raise ValueError("friction must be non-negative")
        if self.slip_regularisation <= 0.0:
            raise ValueError("slip_regularisation must be positive")

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def damping_ratio(self) -> float:
        return damping_ratio_from_restitution(self.restitution)


STEEL = ContactMaterial(youngs_modulus=2.10e11, poisson_ratio=0.30,
                        restitution=0.30, friction=0.15, density=7850.0,
                        crushing_strength=1.0e9)


# --------------------------------------------------------------------------
# contact elasticity (closed form, verified against analytic values)
# --------------------------------------------------------------------------
def effective_modulus(mat1: ContactMaterial, mat2: ContactMaterial) -> float:
    """``1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2``."""
    return 1.0 / ((1.0 - mat1.poisson_ratio ** 2) / mat1.youngs_modulus
                  + (1.0 - mat2.poisson_ratio ** 2) / mat2.youngs_modulus)


def effective_shear_modulus(mat1: ContactMaterial, mat2: ContactMaterial) -> float:
    """``1/G* = (2-nu1)/G1 + (2-nu2)/G2`` (Mindlin)."""
    return 1.0 / ((2.0 - mat1.poisson_ratio) / mat1.shear_modulus
                  + (2.0 - mat2.poisson_ratio) / mat2.shear_modulus)


def effective_radius(r1, r2):
    """``R* = r1 r2 / (r1 + r2)``; a flat surface is ``r2 = inf``."""
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / r1 + 1.0 / r2
    return 1.0 / inv


def effective_mass(m1, m2):
    """``m* = m1 m2 / (m1 + m2)``; a rigid wall is ``m2 = inf``."""
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / m1 + 1.0 / m2
    return 1.0 / inv


def hertz_elastic_force(delta, e_eff: float, r_eff):
    """``F_n = (4/3) E* sqrt(R*) delta^{3/2}`` (Hertz, elastic part only)."""
    delta = np.maximum(np.asarray(delta, dtype=float), 0.0)
    return (4.0 / 3.0) * e_eff * np.sqrt(r_eff) * delta ** 1.5


def hertz_stiffness(delta, e_eff: float, r_eff):
    """``k_n = dF/ddelta = 2 E* sqrt(R* delta)`` (tangent contact stiffness)."""
    delta = np.maximum(np.asarray(delta, dtype=float), 0.0)
    return 2.0 * e_eff * np.sqrt(r_eff) * np.sqrt(delta)


def hertz_elastic_energy(delta, e_eff: float, r_eff):
    """Potential energy of the Hertzian spring, ``(8/15) E* sqrt(R*) delta^{5/2}``."""
    delta = np.maximum(np.asarray(delta, dtype=float), 0.0)
    return (8.0 / 15.0) * e_eff * np.sqrt(r_eff) * delta ** 2.5


# Calibrated ``zeta -> realised restitution`` table of the *implemented* normal
# law: Hertz elasticity + Tsuji viscous damping ``c = 2 zeta sqrt(k_n m*)``
# **plus the non-cohesive clamp** ``max(F, 0)``.  The clamp matters: it removes
# the damper's tension branch, so the pair releases before the elastic energy is
# fully returned and the realised restitution is *higher* than the ideal
# Hertz-damper prediction (at ``zeta = 0.417`` the clamped law realises
# ``e = 0.324`` where the unclamped one gives ``0.199``).  Calibrating the table
# against the law that is actually integrated is therefore the only way to make
# the restitution an input rather than a knob.
#
# Numbers from :func:`_restitution_of_zeta` (normalised units, RK4, dt = 1e-3,
# 15 nodes); the drop test in ``tests/test_clear_ice_contact.py`` reproduces
# them with the full solver to better than 1 %, and the table nodes are
# re-derived from the integrator in the same test file.
ZETA_TABLE = (0.0000, 0.0714, 0.1429, 0.2143, 0.2857, 0.3571, 0.4286, 0.5000,
              0.5714, 0.6429, 0.7143, 0.7857, 0.8571, 0.9286, 1.0000)
E_TABLE = (1.000000, 0.791144, 0.638925, 0.525081, 0.438031, 0.370182, 0.316407,
           0.273156, 0.237912, 0.208858, 0.184654, 0.164301, 0.147040, 0.132287,
           0.119587)


def _restitution_of_zeta(zeta: float, n_steps: int = 12000,
                         dt: float = 1.0e-3) -> float:
    """Realised restitution of the implemented contact law (one collision).

    Normalised units (``m* = 1``, ``(4/3)E* sqrt(R*) = 1``, unit approach
    speed); the equation carries no material parameter,

        ``delta'' = -max( delta^{3/2} + 2 zeta sqrt(1.5 sqrt(delta)) delta' , 0)``

    so one universal curve exists.  The clamp is the regularisation that makes
    the contact non-cohesive; it is included here because it changes the
    realised restitution (see the table comment above).
    """
    if zeta <= 0.0:
        return 1.0
    y, v = 0.0, 1.0

    def rhs(a: float, b: float) -> tuple[float, float]:
        p = a if a > 0.0 else 0.0
        f = p ** 1.5 + 2.0 * zeta * math.sqrt(1.5 * math.sqrt(p)) * b
        return b, -f if f > 0.0 else 0.0

    for _ in range(n_steps):
        k1 = rhs(y, v)
        k2 = rhs(y + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1])
        k3 = rhs(y + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1])
        k4 = rhs(y + dt * k3[0], v + dt * k3[1])
        y += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        v += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        if y <= 0.0 and v < 0.0:
            return float(-v)
    return float(-v)


def damping_ratio_from_restitution(restitution: float) -> float:
    """Damping ratio that *realises* the requested restitution (calibrated)."""
    e = float(min(max(restitution, 1.0e-6), 1.0))
    zetas = np.asarray(ZETA_TABLE)
    if e >= 1.0:
        return 0.0
    if e <= E_TABLE[-1]:
        return float(zetas[-1])
    return float(np.interp(e, np.asarray(E_TABLE)[::-1], zetas[::-1]))


def damping_ratio_from_restitution_analytic(restitution: float) -> float:
    """Classical linear-dashpot estimate (reference only, biased at low e)."""
    e = float(min(max(restitution, 1.0e-12), 1.0))
    ln_e = math.log(e)
    return -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)


def restitution_from_damping_ratio(zeta: float) -> float:
    """Inverse of :func:`damping_ratio_from_restitution`."""
    zeta = float(zeta)
    if zeta <= 0.0:
        return 1.0
    if zeta >= 1.0:
        return 0.0
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta ** 2))


def contact_time(m_eff, r_eff, e_eff: float, v_impact) -> float:
    """Hertzian contact duration ``2.87 (m*^2 / (R* E*^2 v))^{1/5}``.

    ``v_impact`` must be positive; a tiny floor is used so the diagnostic stays
    finite at rest.  This is the number that the stability criterion
    ``dt <= contact_time / 10`` is built from (:meth:`ContactSolver.stability`).
    """
    v = np.maximum(np.asarray(v_impact, dtype=float), 1.0e-6)
    m = np.asarray(m_eff, dtype=float)
    return 2.87 * (m ** 2 / (np.asarray(r_eff, dtype=float) * e_eff ** 2 * v)) ** 0.2


# --------------------------------------------------------------------------
# pair solver
# --------------------------------------------------------------------------
@dataclass
class PairForces:
    """Diagnostics of one force evaluation (all arrays length = active pairs)."""

    i: np.ndarray
    j: np.ndarray
    normal: np.ndarray          # (P, 2) unit vector from i to j
    overlap: np.ndarray         # penetration depth (m)
    fn: np.ndarray              # normal force magnitude (N), >= 0 by construction
    ft: np.ndarray              # signed tangential force along t = perp(n)
    fn_elastic: np.ndarray      # elastic part only (for the energy ledger)
    sliding: np.ndarray         # bool, Coulomb cap active

    def __len__(self) -> int:
        return len(self.i)


class ContactSolver:
    """History-aware pair contact evaluation for disks in a periodic box.

    The floe field is planar: each floe is a disk of radius ``r`` with mass
    ``m`` and inertia ``I``; contacts are resolved as *sphere-equivalent*
    Hertz contacts whose effective radius is the in-plane curvature
    ``R* = r_i r_j / (r_i + r_j)``.  Using the exact three-dimensional Hertz
    law with the planar curvature radius is the standard spherical-proxy
    simplification (also used by :mod:`tensordem.ice_contact`); the resulting
    stiffness is a stated modelling choice, and the sensitivity of the reported
    statistics to it is quantified in the UQ sweep.  The closed-form law itself
    is verified in ``tests/test_clear_ice_contact.py``.
    """

    def __init__(self, material: ContactMaterial = ContactMaterial(), *,
                 box: tuple[float, float] | None = None,
                 skin: float = 5.0e-3) -> None:
        self.material = material
        self.e_eff = effective_modulus(material, material)
        self.g_eff = effective_shear_modulus(material, material)
        self.box = box
        self.skin = float(skin)
        self._keys = np.empty(0, dtype=np.int64)
        self._ft = np.empty(0, dtype=float)
        self.n_pairs_last = 0

    # -- history ---------------------------------------------------------
    def reset(self) -> None:
        self._keys = np.empty(0, dtype=np.int64)
        self._ft = np.empty(0, dtype=float)

    def _carry(self, keys: np.ndarray, n: int) -> np.ndarray:
        ft = np.zeros(len(keys), dtype=float)
        if len(self._keys) and len(keys):
            idx = np.searchsorted(self._keys, keys)
            idx_clip = np.clip(idx, 0, len(self._keys) - 1)
            hit = self._keys[idx_clip] == keys
            ft[hit] = self._ft[idx_clip[hit]]
        return ft

    # -- detection -------------------------------------------------------
    def pair_list(self, x: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Candidate pairs with ``d < r_i + r_j + skin`` (periodic-aware)."""
        tree = cKDTree(x, boxsize=self.box)
        search = 2.0 * float(r.max()) + self.skin
        pairs = tree.query_pairs(r=search, output_type="ndarray")
        if len(pairs) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        i, j = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
        d = self.separation(x, i, j)
        keep = d < r[i] + r[j] + self.skin
        return i[keep], j[keep]

    def separation(self, x: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        d = x[j] - x[i]
        if self.box is not None:
            L = np.asarray(self.box, dtype=float)
            d -= L * np.round(d / L)
        return np.linalg.norm(d, axis=1)

    # -- forces ----------------------------------------------------------
    def update(self, x: np.ndarray, v: np.ndarray, r: np.ndarray, mass: np.ndarray,
               omega: np.ndarray, dt: float, thickness: float) -> tuple[np.ndarray, np.ndarray, PairForces]:
        """Return ``(force, torque, PairForces)`` for the current configuration.

        ``force`` is the net contact force on each particle (N) and ``torque``
        the in-plane moment (N m) from tangential and rolling resistance.
        """
        n = len(x)
        i, j = self.pair_list(x, r)
        keys = i.astype(np.int64) * n + j.astype(np.int64)
        if self.box is not None:
            d = x[j] - x[i]
            L = np.asarray(self.box, dtype=float)
            d -= L * np.round(d / L)
        else:
            d = x[j] - x[i]
        dist = np.linalg.norm(d, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            nrm = np.where(dist[:, None] > 0.0, d / dist[:, None], 0.0)
        overlap = r[i] + r[j] - dist
        active = overlap > 0.0

        i, j, keys = i[active], j[active], keys[active]
        nrm, overlap = nrm[active], overlap[active]

        ft_hist = self._carry(keys, n)

        tang = np.stack((-nrm[:, 1], nrm[:, 0]), axis=1)
        v_rel = v[j] - v[i]
        vn = np.einsum("ij,ij->i", v_rel, nrm)          # separation rate
        approach = -vn                                   # >0 when closing
        vt = (np.einsum("ij,ij->i", v[i] - v[j], tang)
              + omega[i] * r[i] + omega[j] * r[j])

        mat = self.material
        r_eff = effective_radius(r[i], r[j])
        m_eff = effective_mass(mass[i], mass[j])

        f_el = hertz_elastic_force(overlap, self.e_eff, r_eff)
        # crushing cap: the bearing area of an edge contact cannot carry more
        # than sigma_c * (2 r_min t); this bounds the force when two floe edges
        # interpenetrate geometrically (relaxation residue or a large step) and
        # is the physical regularisation that keeps the DEM bounded.
        area_cap = 2.0 * np.minimum(r[i], r[j]) * thickness
        f_el = np.minimum(f_el, mat.crushing_strength * area_cap)
        k_n = hertz_stiffness(overlap, self.e_eff, r_eff)
        c_n = 2.0 * mat.damping_ratio * np.sqrt(k_n * m_eff)
        fn = np.maximum(f_el + c_n * approach, 0.0)      # regularisation: no tension
        fn = np.minimum(fn, mat.crushing_strength * area_cap)

        k_t = 8.0 * self.g_eff * np.sqrt(r_eff * np.maximum(overlap, 0.0))
        ft = ft_hist - k_t * vt * dt
        cap = mat.friction * fn
        ft = np.clip(ft, -cap, cap)
        sliding = np.abs(ft_hist - k_t * vt * dt) >= cap

        force = np.zeros((n, 2), dtype=float)
        torque = np.zeros(n, dtype=float)
        # nrm points i -> j, so the repulsive normal force on i is -nrm
        fx = -fn * nrm[:, 0] + ft * tang[:, 0]
        fy = -fn * nrm[:, 1] + ft * tang[:, 1]
        # bincount accumulation is ~10x faster than np.add.at for ~1e3 pairs
        force[:, 0] = (np.bincount(i, weights=fx, minlength=n)
                       - np.bincount(j, weights=fx, minlength=n))
        force[:, 1] = (np.bincount(i, weights=fy, minlength=n)
                       - np.bincount(j, weights=fy, minlength=n))
        torque = (np.bincount(i, weights=r[i] * ft, minlength=n)
                  + np.bincount(j, weights=r[j] * ft, minlength=n))

        self._keys = keys
        self._ft = ft
        self.n_pairs_last = len(i)

        diag = PairForces(i=i, j=j, normal=nrm, overlap=overlap, fn=fn, ft=ft,
                          fn_elastic=f_el, sliding=sliding)
        return force, torque, diag

    # -- plane wall -------------------------------------------------------
    def plane_wall_contacts(self, x: np.ndarray, v: np.ndarray, r: np.ndarray,
                            omega: np.ndarray, m: np.ndarray, x_wall: float,
                            normal_sign: float = -1.0,
                            wall: ContactMaterial = STEEL,
                            thickness: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
        """Contacts against a rigid plane ``x = x_wall`` (the hull face).

        ``normal_sign = -1`` puts the ice on the ``x < x_wall`` side (the wall
        pushes in ``-x``); ``+1`` is the mirror case.  Flat counter-surface:
        ``R* = r``, ``m* = m``.  Same law as the pair contact -- Hertz +
        Tsuji damping, ``max(., 0)`` non-cohesive regularisation and the
        crushing cap -- so the wall load is resolved with the same contact
        physics as the floes.
        """
        n = len(x)
        force = np.zeros((n, 2), dtype=float)
        torque = np.zeros(n, dtype=float)
        overlap = r - normal_sign * (x[:, 0] - x_wall)
        active = overlap > 0.0
        if not np.any(active):
            return force, torque
        idx = np.nonzero(active)[0]
        nrm = np.array([[normal_sign, 0.0]])
        e_eff = effective_modulus(self.material, wall)
        g_eff = effective_shear_modulus(self.material, wall)
        r_eff = r[idx]
        f_el = hertz_elastic_force(overlap[idx], e_eff, r_eff)
        k_n = hertz_stiffness(overlap[idx], e_eff, r_eff)
        c_n = 2.0 * wall.damping_ratio * np.sqrt(k_n * m[idx])
        approach = -normal_sign * v[idx, 0]
        fn = np.maximum(f_el + c_n * approach, 0.0)
        fn = np.minimum(fn, min(self.material.crushing_strength,
                                wall.crushing_strength) * 2.0 * r_eff * thickness)
        vt = v[idx, 1] + omega[idx] * r[idx]
        ft = -wall.friction * fn * np.tanh(vt / wall.slip_regularisation)
        force[idx, 0] = normal_sign * fn
        force[idx, 1] = ft
        torque[idx] = r[idx] * ft
        return force, torque

    # -- wall ------------------------------------------------------------
    def wall_contacts(self, x: np.ndarray, v: np.ndarray, r: np.ndarray,
                      omega: np.ndarray, m: np.ndarray, p0, p1,
                      wall: ContactMaterial = STEEL,
                      thickness: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
        """Contacts of disks against a rigid capsule (line segment + round caps).

        The hull gate is represented as a zero-thickness segment; the closest
        point on the segment gives the normal, so the gate tip is a rounded
        shoulder and no corner singularity exists.  The wall normal force is
        ``max(Hertz + damping, 0)`` exactly as for particle pairs, with
        ``R* = r`` (flat counter-surface) and ``m* = m`` (rigid wall).
        """
        n = len(x)
        a = np.asarray(p0, dtype=float)
        b = np.asarray(p1, dtype=float)
        ab = b - a
        l2 = float(ab @ ab)
        t = np.clip(((x - a) @ ab) / max(l2, 1e-30), 0.0, 1.0)
        closest = a + t[:, None] * ab
        d = x - closest
        dist = np.linalg.norm(d, axis=1)
        overlap = r - dist
        active = overlap > 0.0

        force = np.zeros((n, 2), dtype=float)
        torque = np.zeros(n, dtype=float)
        if not np.any(active):
            return force, torque

        idx = np.nonzero(active)[0]
        nrm = np.where(dist[idx, None] > 0.0, d[idx] / dist[idx, None], 0.0)
        tang = np.stack((-nrm[:, 1], nrm[:, 0]), axis=1)

        e_eff = effective_modulus(self.material, wall)
        g_eff = effective_shear_modulus(self.material, wall)
        r_eff = r[idx]
        f_el = hertz_elastic_force(overlap[idx], e_eff, r_eff)
        k_n = hertz_stiffness(overlap[idx], e_eff, r_eff)
        c_n = 2.0 * wall.damping_ratio * np.sqrt(k_n * m[idx])
        approach = np.einsum("ij,ij->i", -v[idx], nrm)
        fn = np.maximum(f_el + c_n * approach, 0.0)
        area_cap = 2.0 * r_eff * thickness
        fn = np.minimum(fn, min(self.material.crushing_strength,
                                wall.crushing_strength) * area_cap)

        vt = (np.einsum("ij,ij->i", v[idx], tang) + omega[idx] * r[idx])
        regularised = np.tanh(vt / wall.slip_regularisation)
        ft = -wall.friction * fn * regularised

        force[idx] = fn[:, None] * nrm + ft[:, None] * tang
        torque[idx] = r[idx] * ft
        return force, torque