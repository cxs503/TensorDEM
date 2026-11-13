"""Multi-rate (multiple-time-step) contact integration with a machinable
time-step / sub-cycle contract.

Why this module exists
----------------------
``tensordem.hertz_impact_time`` pins the Hertzian contact duration against a
rigid platen and ``tensordem.contact_timestep_ledger`` pins the integrator's own
verification surface (Hookean contact time, Cundall-Strack critical step,
Hunt-Crossley restitution, energy ledger).  What neither provides is the *cost*
statement every frictional DEM production run needs: with Coulomb friction the
per-step work is dominated by the **return mapping** of the tangential spring,
and the classical implementation re-evaluates the contact set three times per
step (trial -> project -> re-evaluate) to keep the tangential history
consistent.  That is the "3x single-step cost" of the friction arm: it buys
stability at the *outer* step but pays for it in every step, in every contact.

This module replaces that trade with a multi-rate scheme and, more importantly,
makes the trade **machine-decidable**:

``hertz_contact_time`` / ``linear_contact_time``
    The contact duration ``t_c`` of the contact law, i.e. the physical scale a
    step must be compared against (no magic constants: the Hertz prefactor is
    re-derived from ``I = (2/5) B(2/5, 1/2)``).

``stability_gate``
    The contract.  Given an outer step ``dt`` and the contact time ``t_c`` it
    returns ``ratio = dt / t_c`` and a verdict:

    ===========  ==========================================================
    ``ok``       ``ratio <= 0.2``: the MTS sub-cycle resolves the contact
                 (``h <= h_target * t_c``) and the slow pass amortises.
    ``tight``    ``0.2 < ratio <= 0.5``: usable only with a sub-cycle count
                 that grows faster than the slow-pass saving; the caller must
                 ask for it explicitly.
    ``violated`` ``ratio > 0.5``: refuse.  A step this long cannot resolve a
                 contact whatever the sub-cycling, so the gate is closed.
    ===========  ==========================================================

``MTSSystem``
    A 2D frictional disk integrator with three interchangeable arms:

    ``mts``          slow pass (pair list / rolling bookkeeping) once per outer
                     step ``H``, contact sub-cycle of ``n_sub`` velocity-Verlet
                     substeps on the cached pair list, **one** contact-kernel
                     evaluation (and hence one return-mapping pass) per substep.
    ``legacy3x``     the replaced friction arm: monorate semi-implicit Euler at
                     the fine step with **three** full contact-kernel evaluations
                     per step (trial, project, re-evaluate), each restarting from
                     the same trial tangential history so the converged physics is
                     identical and only the cost differs.
    ``mts_overstep`` the same MTS machinery driven past the gate
                     (``ratio = 0.5``, no sub-cycle).  It exists only as the
                     negative control that proves the gate is *tight*: if a step
                     of half a contact time still integrated the impact
                     correctly, the gate would be decorative.

    Both friction arms carry the same physics (linear or Hertz normal law,
    Hunt-Crossley or linear dashpot, tangential spring with Coulomb clipping,
    rolling resistance), so the comparison isolates the integration scheme.

What is verified (see ``benchmarks/mts_contact_gate.py``)
---------------------------------------------------------
1. **restitution** of a single Hertzian impact with Hunt-Crossley damping,
   ``e = 0.5`` and ``e = 0.9``, at ``dt/t_c in {0.02, 0.05, 0.1, 0.2}``: the
   error against the fine-step reference must stay below 2 %;
2. **energy / momentum ledger** of an undamped doubly periodic gas over
   ``1e5`` outer steps: mechanical energy drift < 1 %, linear momentum
   conserved to ``1e-12`` relative (pair forces are accumulated as ``+f`` /
   ``-f``, so the conservation is structural);
3. **frictional jamming** ``mu = 0.5``, ``N = 256``: the MTS arm reproduces the
   legacy arm's ``phi_J`` within 0.5 % while running at least 2x faster in wall
   clock on the same physical protocol;
4. **negative control**: ``dt/t_c = 0.5`` without sub-cycling must fail
   detectably (restitution error > 10 % or energy growth > 10 %);
5. **shard reproducibility**: the contact force is accumulated over 1 / 2 / 4
   contiguous shards of the cached pair list; the shard count must not change
   the trajectory beyond a stated floating-point bound.

Conventions (stated once, used everywhere)
------------------------------------------
Contact of body ``i`` with body ``j`` (or with a wall, ``j = -1``): ``n`` is the
unit vector from ``i`` to the contact, ``d`` the centre distance, ``delta =
r_i + r_j - d >= 0`` the overlap and ``v_n = (v_i - v_j) . n`` the closing
speed.  Then ::

    linear : F_n = k_n delta        - c_n v_n ,   c_n = 2 zeta sqrt(k_n m_eff)
    hertz  : F_n = k_h delta^{3/2}  - (3/2) alpha k_h delta^{1/2} v_n  (Hunt-Crossley)
    F_n    = max(0, F_n)
    f_i    = -F_n n + f_t ,      f_t = -k_t xi ,   |f_t| <= mu F_n
    xi    <- proj_perp(n)(xi + v_t h) ,  rescaled when the Coulomb cone clips
    M_r    = -k_r theta_rel ,  k_r = k_t r_eff^2 / 4 ,  |M_r| <= mu_r r_eff F_n

with ``m_eff`` the harmonic pair mass (``m_i m_j / (m_i + m_j)``, equal to ``m``
against a static wall) and ``r_eff = r_i r_j / (r_i + r_j)``.

All quantities are SI.  Reduced units are used by the jamming protocol only,
where the reported ``phi_J`` is dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "HERTZ_TIME_PREFACTOR",
    "hertz_stiffness_plane",
    "hertz_stiffness_pair",
    "hertz_contact_time",
    "hertz_peak_overlap",
    "linear_contact_time",
    "hunt_crossley_restitution",
    "alpha_for_restitution",
    "calibrate_alpha",
    "reference_restitution",
    "stability_gate",
    "GATE_MAX_RATIO",
    "GATE_TIGHT_RATIO",
    "MTSConfig",
    "MTSSystem",
    "measure_restitution",
    "gas_ledger",
    "shear_cell_run",
    "pressure_and_shear",
    "jamming_protocol",
    "phi_j_from_levels",
]

#: outer step at or below which the sub-cycled MTS contract is closed
GATE_MAX_RATIO = 0.2
#: above this the gate is violated whatever the sub-cycle count
GATE_TIGHT_RATIO = 0.5


# ---------------------------------------------------------------------------
# Closed forms: contact time and the stability gate
# ---------------------------------------------------------------------------
def _hertz_time_prefactor() -> float:
    """``C = 2 I`` with ``I = int_0^1 du / sqrt(1 - u^{5/2}) = (2/5) B(2/5, 1/2)``.

    Derived from the Beta function so the number is not copied from a table
    (``C = 2.9427...``), with a quadrature fallback of the same value.
    """
    try:
        from scipy.special import beta as _beta

        integral = 0.4 * float(_beta(0.4, 0.5))
    except Exception:  # pragma: no cover - only if scipy.special is unavailable
        u = np.linspace(0.0, 1.0, 200001)[:-1]
        integral = float(np.trapezoid(1.0 / np.sqrt(1.0 - u ** 2.5), u))
    return 2.0 * integral


HERTZ_TIME_PREFACTOR = _hertz_time_prefactor()


def hertz_stiffness_plane(young_modulus: float, poisson_ratio: float, radius: float) -> float:
    """``k_h = (4/3) E* sqrt(R)`` for a *rigid platen* (only the sphere deforms).

    ``1/E* = (1 - nu^2)/E`` -- the two-body factor ``2(1 - nu^2)/E`` shifts the
    contact time by ``2^{-2/5} = 0.758`` (see :mod:`tensordem.hertz_impact_time`).
    """
    for name, value in (("young_modulus", young_modulus), ("radius", radius)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    return (4.0 / 3.0) * young_modulus / (1.0 - poisson_ratio ** 2) * math.sqrt(radius)


def hertz_stiffness_pair(young_modulus: float, poisson_ratio: float,
                         radius_i: float, radius_j: float) -> float:
    """``k_h = (4/3) E* sqrt(R*)`` for two spheres (both deform)."""
    if radius_i <= 0.0 or radius_j <= 0.0:
        raise ValueError("radii must be positive")
    reduced = radius_i * radius_j / (radius_i + radius_j)
    return (4.0 / 3.0) * young_modulus / (2.0 * (1.0 - poisson_ratio ** 2)) * math.sqrt(reduced)


def hertz_contact_time(m_eff: float, stiffness: float, v_impact: float) -> float:
    """Full Hertzian contact duration ``C (5 m_eff / (4 k_h))^{2/5} v^{-1/5}``."""
    for name, value in (("m_eff", m_eff), ("stiffness", stiffness),
                        ("v_impact", v_impact)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    return (HERTZ_TIME_PREFACTOR * (5.0 * m_eff / (4.0 * stiffness)) ** 0.4
            * v_impact ** -0.2)


def hertz_peak_overlap(m_eff: float, stiffness: float, v_impact: float) -> float:
    """Peak overlap ``delta_max = (5 m_eff v^2 / (4 k_h))^{2/5}``."""
    return (5.0 * m_eff * v_impact ** 2 / (4.0 * stiffness)) ** 0.4


def linear_contact_time(m_eff: float, stiffness: float) -> float:
    """Hookean contact duration ``pi sqrt(m_eff / k)`` (half period)."""
    if m_eff <= 0.0 or stiffness <= 0.0:
        raise ValueError("m_eff and stiffness must be positive")
    return math.pi * math.sqrt(m_eff / stiffness)


def _hc_integral() -> float:
    """``J = int_0^1 sqrt(u (1 - u^{5/2})) du`` -- the Hunt-Crossley *work* integral.

    Derivation (the shape integral ``int sqrt(1 - u^{5/2}) du`` is the wrong one
    and gives 0.8176 instead of 0.5045 -- see the trap note in the report)::

        W = 2 int_0^{dm} (3/2) alpha k sqrt(d) |d'| dd ,
        |d'| = sqrt(4k/(5m) (dm^{5/2} - d^{5/2})) ,
        u = d/dm  =>  W = 3 alpha k dm^{3/2} sqrt(2E0/m) J .
    """
    u = np.linspace(0.0, 1.0, 800001)
    return float(np.trapezoid(np.sqrt(u * (1.0 - u ** 2.5)), u))


_HC_INTEGRAL = _hc_integral()


def hunt_crossley_restitution(v_impact: float, m_eff: float, stiffness: float,
                              alpha: float) -> float:
    """Semi-analytic restitution of a Hunt-Crossley Hertzian contact.

    First-order dissipation balance (Hunt & Crossley 1975)::

        e = sqrt(1 - W/E0),
        W/E0 = 3 alpha k delta_max^{3/2} v J,   J = int_0^1 sqrt(u(1-u^{5/2})) du

    identical to
    ``tensordem.contact_timestep_ledger.hunt_crossley_semi_analytic_restitution``
    (pinned against it by ``test_hc_matches_ledger``).  The ledger module is not
    imported here so this module keeps a minimal dependency set.
    """
    if alpha <= 0.0:
        return 1.0
    d_max = hertz_peak_overlap(m_eff, stiffness, v_impact)
    energy = 0.5 * m_eff * v_impact ** 2
    work = (3.0 * alpha * stiffness * d_max ** 1.5
            * math.sqrt(2.0 * energy / m_eff) * _HC_INTEGRAL)
    return math.sqrt(max(0.0, 1.0 - work / energy))


def alpha_for_restitution(target_e: float, v_impact: float, m_eff: float,
                          stiffness: float, *, tol: float = 1e-12,
                          max_iter: int = 200) -> float:
    """Invert :func:`hunt_crossley_restitution` for ``alpha`` (bisection).

    ``alpha`` is set from the *semi-analytic* law; the integrator's measured
    restitution is then compared against that law, so the target is never
    fitted to the solver under test.
    """
    if not (0.0 < target_e < 1.0):
        raise ValueError("target_e must lie in (0, 1)")
    lo, hi = 0.0, 1.0
    while hunt_crossley_restitution(v_impact, m_eff, stiffness, hi) > target_e:
        hi *= 2.0
        if hi > 1e3:
            raise ValueError("target restitution not reachable with Hunt-Crossley damping")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = hunt_crossley_restitution(v_impact, m_eff, stiffness, mid)
        if abs(value - target_e) < tol:
            return mid
        if value > target_e:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_ALPHA_CACHE: dict[tuple, float] = {}


def reference_restitution(v_impact: float, m_eff: float, stiffness: float,
                          alpha: float, *, samples_per_contact: int = 2000) -> float:
    """Restitution of the *1-DOF* normal impact, velocity-Verlet reference.

    The single-sphere/platen problem is exactly 1-DOF, so this is an independent
    (higher-order) reference for the 2D integrator under test.  It uses the
    velocity-Verlet Hunt-Crossley integrator of
    :class:`tensordem.contact_timestep_ledger.WallImpact`, i.e. a different
    code path, a different integrator and a different time step.
    """
    from .contact_timestep_ledger import WallImpact

    t_c = hertz_contact_time(m_eff, stiffness, v_impact)
    impact = WallImpact(mass=m_eff, stiffness=stiffness, law="hertz", damping=alpha)
    impact.rate = float(v_impact)
    dt = t_c / samples_per_contact
    for _ in range(int(2.0e6)):
        old = impact.overlap
        impact.step(dt)
        if impact.overlap <= 0.0 < old and old > 0.0:
            return abs(impact.rate) / float(v_impact)
    raise RuntimeError("reference impact did not rebound")  # pragma: no cover


def calibrate_alpha(target_e: float, v_impact: float, m_eff: float,
                    stiffness: float, *, samples_per_contact: int = 2000,
                    tol: float = 1e-4, max_iter: int = 60) -> float:
    """``alpha`` whose *measured* (reference) restitution equals ``target_e``.

    The first-order Hunt-Crossley formula is used only as a starting bracket: at
    ``e = 0.5`` it is 9 % low (documented in
    :mod:`tensordem.contact_timestep_ledger`), so the damping level that defines
    the acceptance case ``e = 0.5`` is calibrated on the fine-step *reference*
    integrator, never on the MTS integrator under test.
    """
    key = (round(target_e, 9), round(v_impact, 12), round(m_eff, 15),
           round(stiffness, 6), samples_per_contact)
    if key in _ALPHA_CACHE:
        return _ALPHA_CACHE[key]
    if not (0.0 < target_e < 1.0):
        raise ValueError("target_e must lie in (0, 1)")
    lo, hi = 0.0, float(alpha_for_restitution(target_e, v_impact, m_eff, stiffness))
    while reference_restitution(v_impact, m_eff, stiffness, hi,
                                samples_per_contact=samples_per_contact) > target_e:
        hi *= 2.0
        if hi > 1e3:
            raise ValueError("target restitution not reachable")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        e = reference_restitution(v_impact, m_eff, stiffness, mid,
                                  samples_per_contact=samples_per_contact)
        if abs(e - target_e) < tol:
            _ALPHA_CACHE[key] = mid
            return mid
        if e > target_e:
            lo = mid
        else:
            hi = mid
    _ALPHA_CACHE[key] = 0.5 * (lo + hi)
    return _ALPHA_CACHE[key]


def stability_gate(dt: float, contact_time: float, *,
                   max_ratio: float = GATE_MAX_RATIO,
                   tight_ratio: float = GATE_TIGHT_RATIO) -> dict:
    """The time-step / sub-cycle contract as a machine-readable verdict.

    Returns ``ratio = dt / t_c``, the verdict ``status`` in
    ``{"ok", "tight", "violated"}``, the sub-step resolution implied by the
    contract (``h / t_c``) and the number of sub-steps it demands so that
    ``h <= max_ratio * t_c``.
    """
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(contact_time) or contact_time <= 0.0:
        raise ValueError("contact_time must be finite and positive")
    ratio = dt / contact_time
    if ratio <= max_ratio * (1.0 + 1e-12):
        status = "ok"
    elif ratio <= tight_ratio:
        status = "tight"
    else:
        status = "violated"
    n_sub_required = max(1, int(math.ceil(ratio / max_ratio - 1e-12)))
    return {
        "dt": float(dt),
        "contact_time": float(contact_time),
        "ratio": float(ratio),
        "status": status,
        "max_ratio": float(max_ratio),
        "tight_ratio": float(tight_ratio),
        "h_over_tc": float(ratio / n_sub_required),
        "n_sub_required": int(n_sub_required),
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MTSConfig:
    """Contact law + integration parameters (SI).

    ``contact_model``: ``"linear"`` (``F = k_n delta``) or ``"hertz"``
    (``F = k_h delta^{3/2}``); in the Hertz branch ``kn`` is the stiffness
    prefactor ``k_h`` [N m^{-3/2}].

    ``damping``: ``"hunt_crossley"`` (uses ``alpha``), ``"dashpot"`` (uses
    ``zeta``, ``c_n = 2 zeta sqrt(k_n m_eff)``) or ``"none"``.

    ``scheme``: ``"mts"``, ``"legacy3x"`` or ``"mts_overstep"`` (negative
    control only).
    """

    kn: float = 3.0e4
    kt_ratio: float = 0.5
    mu: float = 0.5
    mu_rolling: float = 0.0
    contact_model: str = "linear"
    damping: str = "dashpot"
    alpha: float = 0.0
    zeta: float = 0.0
    density: float = 917.0
    thickness: float = 0.03
    damping_time: float = 0.0
    gravity: tuple[float, float] = (0.0, 0.0)
    dt: float = 1.0e-5
    n_sub: int = 1
    h_target_tc: float = 0.02
    scheme: str = "mts"
    shards: int = 1
    skin: float = 1.0e-3
    enforce_gate: bool = True
    allow_tight: bool = False

    def __post_init__(self) -> None:
        for name in ("kn", "kt_ratio", "density", "thickness", "dt", "skin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("mu", "mu_rolling", "alpha", "zeta", "damping_time",
                     "h_target_tc"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.contact_model not in ("linear", "hertz"):
            raise ValueError("contact_model must be 'linear' or 'hertz'")
        if self.damping not in ("none", "dashpot", "hunt_crossley"):
            raise ValueError("damping must be 'none', 'dashpot' or 'hunt_crossley'")
        if self.scheme not in ("mts", "legacy3x", "mts_overstep"):
            raise ValueError("scheme must be 'mts', 'legacy3x' or 'mts_overstep'")
        if self.n_sub < 1:
            raise ValueError("n_sub must be >= 1")
        if self.shards < 1:
            raise ValueError("shards must be >= 1")
        if len(tuple(self.gravity)) != 2:
            raise ValueError("gravity must have two components")


# ---------------------------------------------------------------------------
# The integrator
# ---------------------------------------------------------------------------
class MTSSystem:
    """2D frictional disk system with a multi-rate contact integrator.

    Parameters
    ----------
    positions, radii
        ``(N, 2)`` centres (inside the cell) and ``(N,)`` radii.
    config
        :class:`MTSConfig`.
    walls
        optional ``(S, 2, 2)`` segment array (a static, infinitely massive body).
    periodic
        optional ``(Lx, Ly)``; ``None`` on an axis leaves it open.
    shear_rate
        optional homogeneous shear rate ``d v_x / d y`` (periodic shear cell);
        it enters both the affine advection of the positions and the relative
        velocity at a contact (SLLOD without the Lees-Edwards tilt -- see the
        limitation note in the driven report).

    Attributes
    ----------
    ``contacts``
        per-contact records in the layout of
        :mod:`tensordem.granular2d.stress` (``i``, ``j``, ``kind``, ``n``,
        ``lever``, ``f``, ``f_on_i``, ``fn``, ``ft``, ``overlap``, ``point``),
        so the *shipped* Love-Weber / pressure code can be reused verbatim.
    """

    def __init__(self, positions, radii, config: MTSConfig | None = None, *,
                 walls: np.ndarray | None = None,
                 periodic: tuple[float | None, float | None] | None = None,
                 shear_rate: float = 0.0, lees_edwards: bool = False):
        pos = np.asarray(positions, dtype=np.float64)
        rad = np.asarray(radii, dtype=np.float64).reshape(-1)
        if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] != rad.size:
            raise ValueError("positions must be (N, 2) and radii (N,)")
        if not np.isfinite(pos).all() or not np.isfinite(rad).all():
            raise ValueError("positions and radii must be finite")
        if np.any(rad <= 0.0):
            raise ValueError("radii must be positive")
        self.pos = pos.copy()
        self.radii = rad.copy()
        self.config = config or MTSConfig()
        self.vel = np.zeros((rad.size, 2))
        self.omega = np.zeros(rad.size)
        self.mass = (self.config.density * self.config.thickness
                     * math.pi * self.radii ** 2)
        self.inertia = 0.5 * self.mass * self.radii ** 2
        walls_arr = np.asarray(walls, dtype=np.float64) if walls is not None else None
        self.walls = (walls_arr.reshape(-1, 2, 2) if walls_arr is not None
                      and walls_arr.size else np.zeros((0, 2, 2)))
        if periodic is None:
            self.periodic_x = None
            self.periodic_y = None
        else:
            self.periodic_x = None if periodic[0] is None else float(periodic[0])
            self.periodic_y = None if periodic[1] is None else float(periodic[1])
        self.shear_rate = float(shear_rate)
        # ``lees_edwards`` makes the *periodic lattice itself* shear with the
        # flow (tilt ``strain = shear_rate * t``), i.e. the cell is triclinic.
        # The legacy arm keeps an orthogonal lattice, so a contact across the
        # x boundary is evaluated at a displacement that misses the accumulated
        # tilt and its affine velocity (see docs/XVAL_PERIODIC_SLLOD_*).
        self.lees_edwards = bool(lees_edwards)
        self.strain = 0.0
        self.time = 0.0
        self.steps = 0
        self.dt = float(self.config.dt)
        self.contacts: dict[str, np.ndarray] = _empty_contacts()
        self.wall_force = np.zeros(2)
        self._xi = np.zeros((0, 2))
        self._roll = np.zeros(0)
        self._keys = np.zeros(0, dtype=np.int64)
        self._slots: dict[str, np.ndarray] | None = None
        self._last: dict[str, np.ndarray] = {}
        self._p0: np.ndarray | None = None
        self.momentum_drift = 0.0
        self.kinematics_calls = 0
        self.return_mapping_passes = 0
        self._verlet_force: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ setup
    @property
    def n(self) -> int:
        return int(self.radii.size)

    @property
    def mean_radius(self) -> float:
        return float(self.radii.mean())

    def domain_area(self) -> float | None:
        if self.periodic_x is not None and self.periodic_y is not None:
            return float(self.periodic_x * self.periodic_y)
        if len(self.walls):
            lo = self.walls.reshape(-1, 2).min(axis=0)
            hi = self.walls.reshape(-1, 2).max(axis=0)
            return float(np.prod(hi - lo))
        return None

    def area_fraction(self, area: float | None = None) -> float:
        if area is None:
            area = self.domain_area()
        if area is None or area <= 0.0:
            raise ValueError("an explicit positive area is required")
        return float(np.sum(math.pi * self.radii ** 2) / area)

    def volume_fraction(self) -> float:
        return self.area_fraction()

    # --------------------------------------------------- contact-time helpers
    def contact_time(self, v_impact: float | None = None, *,
                     radius: float | None = None) -> float:
        """``t_c`` of the stiffest contact present, from the closed forms.

        Linear law: ``pi sqrt(m_eff / k_n)`` with the *smallest* mass (the
        Cundall-Strack worst case).  Hertz law: the impact duration at
        ``v_impact`` (default: the velocity scale set by the smallest mass and
        ``k_h delta^{3/2}`` at ``delta = 0.01 r``), which is the physically
        correct scale because ``t_c`` depends on the impact speed.
        """
        m_min = float(self.mass.min())
        if self.config.contact_model == "linear":
            return linear_contact_time(m_min, self.config.kn)
        if v_impact is None:
            v_impact = (float(np.max(np.linalg.norm(self.vel, axis=1)))
                        if self.vel.size else 0.0)
            # a resting contact has no impact scale: use a grazing floor so the
            # gate stays open (a static problem is not a time-step problem)
            v_impact = max(v_impact, 1.0e-3)
        return hertz_contact_time(m_min, self.config.kn, float(v_impact))

    def stability_gate(self, *, v_impact: float | None = None) -> dict:
        return stability_gate(self.dt, self.contact_time(v_impact=v_impact))

    # ------------------------------------------------------- neighbour search
    def _pairs_disk(self, pos: np.ndarray):
        """Candidate disk-disk pairs ``(i, j, shift, key)`` with the minimum image.

        ``shift`` is the image offset added to ``x_j - x_i`` (a cell vector, or
        zero); ``key`` encodes ``(i, j, image)`` so the tangential history
        follows a pair across pair-list rebuilds.
        """
        reach = 2.0 * float(self.radii.max()) + self.config.skin
        lx, ly = self.periodic_x, self.periodic_y
        # In a sheared cell the *search* is done in the co-moving (untilted)
        # frame s = (x - strain*y, y), where the lattice is orthogonal again --
        # so the four mirror combinations stay complete and no pair can be found
        # twice.  The shift vectors handed to ``_forces`` are then the images of
        # those combinations in the true (tilted) frame.
        lee = bool(self.lees_edwards and lx is not None and ly is not None
                   and self.strain != 0.0)
        pos_search = pos
        if lee:
            pos_search = pos.copy()
            pos_search[:, 0] = np.mod(pos[:, 0] - self.strain * pos[:, 1], lx)
        tree = cKDTree(pos_search)
        pairs = tree.query_pairs(reach, output_type="ndarray").reshape(-1, 2)
        near = {}
        if lx is not None:
            near["x0"] = pos_search[:, 0] < reach
            near["x1"] = pos_search[:, 0] > lx - reach
        if ly is not None:
            near["y0"] = pos_search[:, 1] < reach
            near["y1"] = pos_search[:, 1] > ly - reach
        # ``shift`` is the offset *added to x_j* to reach the image in contact:
        # for a body of the "x1" set (right edge) seen from a body of the "x0"
        # set (left edge) that image lies one period to the *left*, so
        # shift = -Lx.  ``_forces`` uses the same convention; getting the sign
        # wrong silently turns a border contact into a far-field ghost force
        # (regression: test_periodic_image_shift_sign_is_consistent).
        combos = []
        if lx is not None:
            combos.append(((-lx, 0.0), ("x0",), ("x1",)))
        if ly is not None:
            combos.append(((0.0, -ly), ("y0",), ("y1",)))
        if lx is not None and ly is not None:
            combos.append(((-lx, -ly), ("x0", "y0"), ("x1", "y1")))
            combos.append(((-lx, ly), ("x0", "y1"), ("x1", "y0")))
        rows = []
        if len(pairs):
            rows.append(np.column_stack([pairs, np.zeros(len(pairs), dtype=np.int64)]))
        for k, (shift, a_keys, b_keys) in enumerate(combos, start=1):
            a_mask = np.ones(len(pos), dtype=bool)
            b_mask = np.ones(len(pos), dtype=bool)
            for key in a_keys:
                a_mask &= near[key]
            for key in b_keys:
                b_mask &= near[key]
            a = np.flatnonzero(a_mask)
            b = np.flatnonzero(b_mask)
            if not len(a) or not len(b):
                continue
            # consistent with ``_forces``: l = x_j - x_i + shift
            d = pos_search[b][None, :, :] - pos_search[a][:, None, :] + np.asarray(shift)
            if lee:  # orthogonal minimum image in the co-moving frame
                d = d.copy()
                d[..., 0] -= lx * np.round(d[..., 0] / lx)
                d[..., 1] -= ly * np.round(d[..., 1] / ly)
            else:
                d = self._minimum_image(d)
            ii, jj = np.nonzero(np.linalg.norm(d, axis=2) < reach)
            if len(ii):
                rows.append(np.column_stack([a[ii], b[jj],
                                             np.full(len(ii), k, dtype=np.int64)]))
        if not rows:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                    np.zeros((0, 2)), np.zeros(0, dtype=np.int64))
        table = np.vstack(rows).astype(np.int64)
        i, j, s_idx = table[:, 0], table[:, 1], table[:, 2]
        shift = np.zeros((len(table), 2))
        for k, (vec, _a, _b) in enumerate(combos, start=1):
            sel = s_idx == k
            if sel.any():
                vec = np.asarray(vec, dtype=float)
                if lee:   # co-moving -> true lattice vector: (sx, sy) -> (sx + strain*sy, sy)
                    vec = np.array([vec[0] + self.strain * vec[1], vec[1]])
                shift[sel] = vec
        key = (i * (self.n + 1) + j) * 8 + s_idx
        # a physical interaction is unique: the same body pair *and* image slot
        # must not be counted twice (the direct KD-tree list and the mirror combos
        # can otherwise both report it)
        uniq, first = np.unique(key, return_index=True)
        order = np.sort(first)
        return i[order], j[order], shift[order], key[order]

    def _minimum_image(self, d: np.ndarray, *, keep_shift: bool = False):
        out = d
        px, py = self.periodic_x, self.periodic_y
        if self.lees_edwards and px is not None and py is not None                 and self.strain != 0.0:
            # Triclinic (sheared) lattice: the periodic images are generated by
            # a1 = (Lx, 0) and a2 = (strain*Ly, Ly).  ``u = dx - strain*dy`` is
            # invariant under a1/a2, so the minimum image is found by scanning
            # the few y-images that can host it and wrapping x analytically.
            dd = np.asarray(d, dtype=np.float64)
            u = dd[..., 0] - self.strain * dd[..., 1]
            v0 = dd[..., 1]
            best = None
            for k in (-2, -1, 0, 1, 2):
                # each k is a *distinct* lattice point (a2 image): wrapping v back
                # into the box would collapse all five candidates into one
                v = v0 + k * py
                x = u + self.strain * v
                x = x - px * np.round(x / px)
                cand = np.stack([x, v], axis=-1)
                norm2 = np.sum(cand * cand, axis=-1)
                if best is None:
                    best, norm2_best = cand, norm2
                else:
                    take = norm2 < norm2_best
                    best = np.where(take[..., None], cand, best)
                    norm2_best = np.where(take, norm2, norm2_best)
            return (best, None) if keep_shift else best
        if px is not None:
            out = out.copy()
            out[..., 0] -= px * np.round(out[..., 0] / px)
        if py is not None:
            if out is d:
                out = out.copy()
            out[..., 1] -= py * np.round(out[..., 1] / py)
        return (out, None) if keep_shift else out

    def _slow_pass(self) -> None:
        """Refresh the cached contact set (pair list + tangential history)."""
        i, j, shift, key = self._pairs_disk(self.pos)
        n_wall = 0
        if len(self.walls):
            wi = np.repeat(np.arange(self.n, dtype=np.int64), len(self.walls))
            ws = np.tile(np.arange(len(self.walls), dtype=np.int64), self.n)
            n_wall = wi.size
        else:
            wi = np.zeros(0, dtype=np.int64)
            ws = np.zeros(0, dtype=np.int64)
        m_all = len(i) + n_wall
        seg = np.zeros((m_all, 2, 2))
        if n_wall:
            seg[len(i):] = self.walls[ws]
        slots = {
            "i": np.concatenate([i, wi]).astype(np.int64),
            "j": np.concatenate([j, -np.ones(n_wall, dtype=np.int64)]),
            "shift": np.vstack([shift, np.zeros((n_wall, 2))]) if m_all
            else np.zeros((0, 2)),
            "seg": seg,
            "key": np.concatenate([key, -1 - ws]).astype(np.int64),
            "wall": np.concatenate([np.zeros(len(i), dtype=bool),
                                    np.ones(n_wall, dtype=bool)]),
        }
        slots["rj"] = np.where(slots["wall"], 0.0, self.radii[slots["j"]])
        self._slots = slots
        xi = np.zeros((slots["i"].size, 2))
        roll = np.zeros(slots["i"].size)
        if self._keys.size and slots["key"].size:
            pos = np.searchsorted(self._keys, slots["key"])
            pos = np.clip(pos, 0, self._keys.size - 1)
            hit = self._keys[pos] == slots["key"]
            xi[hit] = self._xi[pos[hit]]
            roll[hit] = self._roll[pos[hit]]
        self._keys = slots["key"]
        self._xi = xi
        self._roll = roll
        self._verlet_force = None  # the cached kick is stale after a rebuild

    # ----------------------------------------------------------------- forces
    def _forces(self, h: float, *, passes: int,
                record: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Contact forces at the current state; ``h`` is the substep length.

        ``passes`` repeats the tangential return mapping (the legacy arm pays
        three kernel evaluations per step; the physics of the converged state is
        identical because each pass starts from the same trial stretch).
        """
        cfg = self.config
        s = self._slots
        assert s is not None, "call _slow_pass() before _forces()"
        force = np.zeros((self.n, 2))
        torque = np.zeros(self.n)
        m = s["i"].size
        if m == 0:
            self.contacts = _empty_contacts()
            self._last = {}
            self.wall_force = np.zeros(2)
            return force, torque
        pos = self.pos
        vel = self.vel + self._affine_velocity(pos) if self.shear_rate else self.vel
        ndof_omega = self.omega
        ri = self.radii[s["i"]]
        i = s["i"]
        j = s["j"]
        wall = s["wall"]
        jc = np.where(wall, 0, j)
        pi_ = pos[i]
        l = pos[jc] - pi_ + s["shift"]
        if wall.any():
            a = s["seg"][:, 0]
            ab = s["seg"][:, 1] - a
            denom = np.maximum(np.sum(ab * ab, axis=1), 1e-300)
            t = np.clip(np.sum((pi_ - a) * ab, axis=1) / denom, 0.0, 1.0)
            cp = a + t[:, None] * ab
            l = np.where(wall[:, None], cp - pi_, l)
        dist = np.linalg.norm(l, axis=1)
        good = dist > 1e-12
        n = np.zeros_like(l)
        n[good] = l[good] / dist[good, None]
        overlap = ri + s["rj"] - dist
        active = good & (overlap > 0.0)

        # contact point offsets (centre -> contact) and relative velocity
        r_i = -n * (ri - 0.5 * np.maximum(overlap, 0.0))[:, None]
        r_j = n * (s["rj"] - 0.5 * np.maximum(overlap, 0.0))[:, None]
        v_i = vel[i] + ndof_omega[i][:, None] * _perp(r_i)
        v_j = np.where(wall[:, None], 0.0,
                       vel[jc] + ndof_omega[jc][:, None] * _perp(r_j))
        v_rel = v_i - v_j
        if self.shear_rate and self.lees_edwards:
            # The SLLOD field u = shear_rate * y is not periodic: the image of
            # body ``j`` that owns this contact sits at ``y_j + shift_y``, so its
            # affine velocity is shear_rate*(y_j + shift_y).  ``vel`` above was
            # built from the wrapped positions and therefore misses
            # ``shear_rate*shift_y``.  Without this term the shear rate seen by
            # every contact that straddles the y boundary is wrong by
            # ``shear_rate*Ly`` (regression: test_lees_edwards_*).
            v_rel = v_rel - self.shear_rate * s["shift"][:, 1][:, None] * np.array(
                [1.0, 0.0])[None, :]
        v_n = np.sum(v_rel * n, axis=1)
        v_t = v_rel - v_n[:, None] * n

        # normal elastic + dissipative part; ``v_n > 0`` is the *closing* speed
        # (n points from the body to the contact), so the dashpot force *adds*
        # to the elastic force while the contact compresses and subtracts while
        # it restores -- the sign that makes the Hunt-Crossley/dashpot work
        # negative definite (a minus sign here injects energy: see the module
        # test ``test_damping_sign_is_dissipative``).
        ov_pos = np.maximum(overlap, 0.0)
        if cfg.contact_model == "hertz":
            f_el = cfg.kn * ov_pos ** 1.5
        else:
            f_el = cfg.kn * ov_pos
        if cfg.damping == "dashpot":
            m_eff = np.where(wall, self.mass[i],
                             self.mass[i] * self.mass[jc]
                             / np.maximum(self.mass[i] + self.mass[jc], 1e-300))
            f_d = 2.0 * cfg.zeta * np.sqrt(cfg.kn * m_eff) * v_n
        elif cfg.damping == "hunt_crossley":
            if cfg.contact_model != "hertz":
                raise ValueError("Hunt-Crossley damping requires contact_model='hertz'")
            f_d = 1.5 * cfg.alpha * cfg.kn * np.sqrt(ov_pos) * v_n
        else:
            f_d = np.zeros(m)
        f_n = np.where(active, np.maximum(0.0, f_el + f_d), 0.0)

        # tangential return mapping (repeated ``passes`` times by the legacy arm)
        kt = np.full(m, cfg.kt_ratio * cfg.kn)
        if cfg.contact_model == "hertz":
            kt = cfg.kt_ratio * cfg.kn * np.sqrt(np.maximum(overlap, 0.0))
        xi = self._xi.copy()
        r_eff = np.where(wall, ri, ri * s["rj"] / np.maximum(ri + s["rj"], 1e-300))
        roll = self._roll.copy()
        f_t = np.zeros((m, 2))
        f_n_final = f_n
        for _ in range(max(1, int(passes))):
            trial = xi + v_t * h
            trial = trial - np.sum(trial * n, axis=1)[:, None] * n
            mag = np.linalg.norm(trial, axis=1)
            cap = cfg.mu * f_n_final
            scale = np.where(mag > 1e-300,
                             np.minimum(1.0, cap / np.maximum(mag, 1e-300)), 1.0)
            xi = trial * scale[:, None]
            f_t = -kt[:, None] * xi
            self.return_mapping_passes += 1
        f_on_i = -f_n_final[:, None] * n + f_t

        # rolling resistance (slow variable, once per outer step in the MTS arm)
        if cfg.mu_rolling > 0.0:
            rel_omega = ndof_omega[i] - np.where(wall, 0.0, ndof_omega[jc])
            roll = roll + rel_omega * h
            k_r = 0.25 * kt * r_eff ** 2
            m_r = k_r * roll
            cap_r = cfg.mu_rolling * r_eff * f_n_final
            scale_r = np.where(np.abs(m_r) > 1e-300,
                               np.minimum(1.0, cap_r / np.maximum(np.abs(m_r), 1e-300)),
                               1.0)
            m_r = m_r * scale_r
            roll = roll * scale_r
        else:
            m_r = np.zeros(m)

        # external body forces
        g = np.asarray(cfg.gravity, dtype=np.float64)
        if np.any(g != 0.0):
            force += self.mass[:, None] * g[None, :]

        # accumulation, optionally over contiguous shards of the slot list
        shards = max(1, int(cfg.shards))
        if shards == 1:
            self._accumulate(force, torque, f_on_i, m_r, i, j, wall, r_i, r_j)
        else:
            bounds = np.linspace(0, m, shards + 1).astype(int)
            partial_f = np.zeros((shards, self.n, 2))
            partial_t = np.zeros((shards, self.n))
            for k in range(shards):
                sl = slice(bounds[k], bounds[k + 1])
                fk, tk = np.zeros((self.n, 2)), np.zeros(self.n)
                self._accumulate(fk, tk, f_on_i[sl], m_r[sl], i[sl], j[sl],
                                 wall[sl], r_i[sl], r_j[sl])
                partial_f[k] = fk
                partial_t[k] = tk
            for k in range(shards):
                force += partial_f[k]
                torque += partial_t[k]
            self.shard_partials = partial_f

        # bookkeeping (once per outer step in the MTS arm; the sub-cycle skips
        # the record construction because it is pure output)
        self._xi = xi
        self._roll = roll
        self._last = {"active": active, "overlap": overlap, "fn": f_n_final,
                      "ft": np.linalg.norm(f_t, axis=1), "kt": kt, "xi": xi,
                      "n": n}
        self.kinematics_calls += 1
        if not record:
            return force, torque
        point = np.where(wall[:, None], pos[i] + (ri - 0.5 * np.maximum(overlap, 0.0))[:, None] * n,
                         pos[i] - r_i)
        self.contacts = {
            "i": i,
            "j": j,
            "kind": np.where(wall, 2, 0).astype(np.int8),
            "n": n,
            "lever": -r_i,
            # true centre-to-centre branch vector (image offset included): the
            # virial of a contact must use the same displacement as its force
            "branch": l,
            "f": f_on_i,
            "f_on_i": f_on_i,
            "fn": f_n_final,
            "ft": np.linalg.norm(f_t, axis=1),
            "overlap": overlap,
            "point": point,
        }
        if wall.any():
            self.wall_force = -np.sum(f_on_i[wall], axis=0)
        return force, torque

    @staticmethod
    def _accumulate(force, torque, f_on_i, m_r, i, j, wall, r_i, r_j) -> None:
        np.add.at(force, i, f_on_i)
        np.add.at(torque, i, _cross(r_i, f_on_i) + m_r)
        nd = ~wall
        if nd.any():
            np.add.at(force, j[nd], -f_on_i[nd])
            np.add.at(torque, j[nd], _cross(r_j[nd], -f_on_i[nd]) - m_r[nd])

    def _affine_velocity(self, pos: np.ndarray) -> np.ndarray:
        if not self.shear_rate:
            return np.zeros_like(pos)
        out = np.zeros_like(pos)
        out[:, 0] = self.shear_rate * pos[:, 1]
        return out

    # -------------------------------------------------------------- integration
    def step(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError("n must be nonnegative")
        if self._p0 is None:
            self._p0 = self.momentum().copy()
        for _ in range(n):
            self._outer_step()

    def _outer_step(self) -> None:
        cfg = self.config
        H = self.dt
        if cfg.enforce_gate:
            gate = stability_gate(H, self.contact_time())
            if gate["status"] == "violated" or (gate["status"] == "tight"
                                                and not cfg.allow_tight):
                raise ValueError(
                    f"dt/t_c = {gate['ratio']:.3g} ('{gate['status']}') violates "
                    f"the MTS contract (closed up to {gate['max_ratio']}); pass "
                    "allow_tight=True or enforce_gate=False -- the latter only for "
                    "the negative control")
        self._slow_pass()
        if cfg.scheme == "legacy3x":
            self._substep_legacy(H)
        elif cfg.scheme == "mts_overstep":
            self._substep(H, passes=1)
        else:
            n_sub = max(1, int(cfg.n_sub))
            h = H / n_sub
            for k in range(n_sub):
                self._substep(h, passes=1, record=(k == n_sub - 1))
        self.steps += 1
        self.time += H

    def _substep_legacy(self, h: float) -> None:
        """The replaced friction arm: **three** full contact-kernel evaluations.

        A classical iterative return mapping evaluates the contact set three times
        per step (trial state, projected state, re-evaluation to restore force and
        torque consistency with the clipped tangential spring).  Each iteration
        starts from the *same* trial tangential history, so the converged state --
        and therefore the physics -- is identical to a single converged pass; only
        the cost differs.  That is exactly the 3x single-step cost the MTS
        sub-cycle removes.
        """
        xi0, roll0 = self._xi.copy(), self._roll.copy()
        force = torque = None
        for iteration in range(3):
            self._xi, self._roll = xi0.copy(), roll0.copy()
            # the last iteration also refreshes the per-contact records, so the
            # pressure/stress read-out of this arm is never stale
            force, torque = self._forces(h, passes=1, record=(iteration == 2))
        # the last iteration already left the converged history in place
        self._integrate(h, force, torque)
        if self._p0 is not None:
            self.momentum_drift = float(np.linalg.norm(self.momentum() - self._p0))

    def _integrate(self, h: float, force: np.ndarray, torque: np.ndarray) -> None:
        """Semi-implicit (symplectic) Euler half of the legacy arm."""
        cfg = self.config
        self.vel += h * (force / self.mass[:, None])
        self.omega += h * (torque / self.inertia)
        if cfg.damping_time > 0.0:
            decay = math.exp(-h / cfg.damping_time)
            self.vel *= decay
            self.omega *= decay
        self.pos += h * (self.vel + (self._affine_velocity(self.pos)
                                     if self.shear_rate else 0.0))
        if self.lees_edwards:
            self.strain += h * self.shear_rate
        self._wrap()

    def _substep(self, h: float, *, passes: int, record: bool = True) -> None:
        """One velocity-Verlet step (kick-drift-kick) with a cached force.

        The contact force at the end of a substep is reused as the first half-kick
        of the next one, so a substep still costs exactly **one** contact-kernel
        evaluation -- the same as a plain Euler substep -- while the energy
        behaviour is second order (the Euler arm heats an undamped gas by 12 % in
        1e5 steps; this one does not).  The tangential return mapping is advanced
        exactly once per substep, inside that single evaluation.
        """
        cfg = self.config
        if getattr(self, "_verlet_force", None) is None:
            self._verlet_force = self._forces(h, passes=passes, record=False)
        force, torque = self._verlet_force
        inv_m = 1.0 / self.mass[:, None]
        inv_i = 1.0 / self.inertia
        v_half = self.vel + 0.5 * h * (force * inv_m)
        w_half = self.omega + 0.5 * h * (torque * inv_i)
        self.pos += h * (v_half + (self._affine_velocity(self.pos)
                                  if self.shear_rate else 0.0))
        if self.lees_edwards:
            self.strain += h * self.shear_rate
        self._wrap()
        # evaluate at the drifted position with the half-step velocity (standard
        # velocity-Verlet for velocity-dependent forces)
        self.vel, self.omega = v_half, w_half
        force2, torque2 = self._forces(h, passes=passes, record=record)
        self.vel = v_half + 0.5 * h * (force2 * inv_m)
        self.omega = w_half + 0.5 * h * (torque2 * inv_i)
        if cfg.damping_time > 0.0:
            decay = math.exp(-h / cfg.damping_time)
            self.vel *= decay
            self.omega *= decay
        self._verlet_force = (force2, torque2)
        if self._p0 is not None:  # momentum diagnostics (cheap)
            self.momentum_drift = float(np.linalg.norm(self.momentum() - self._p0))

    def _wrap(self) -> None:
        if self.periodic_x is not None:
            self.pos[:, 0] = np.mod(self.pos[:, 0], self.periodic_x)
        if self.periodic_y is not None:
            self.pos[:, 1] = np.mod(self.pos[:, 1], self.periodic_y)

    # --------------------------------------------------------------- diagnostic
    def momentum(self) -> np.ndarray:
        return np.sum(self.mass[:, None] * self.vel, axis=0)

    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.mass[:, None] * self.vel ** 2)
                     + 0.5 * np.sum(self.inertia * self.omega ** 2))

    def potential_energy(self) -> float:
        """Elastic + tangential energy of the contact set of the last evaluation."""
        last = self._last
        if not last:
            return 0.0
        active = last["active"]
        if not active.any():
            return 0.0
        overlap = last["overlap"][active]
        if self.config.contact_model == "hertz":
            elastic = (2.0 / 5.0) * self.config.kn * overlap ** 2.5
        else:
            elastic = 0.5 * self.config.kn * overlap ** 2
        tangential = 0.5 * last["kt"][active] * np.sum(last["xi"][active] ** 2, axis=1)
        return float(np.sum(elastic) + np.sum(tangential))

    def energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()

    def mean_speed(self) -> float:
        return float(np.mean(np.linalg.norm(self.vel, axis=1)))

    def copy(self) -> "MTSSystem":
        out = MTSSystem(self.pos, self.radii, self.config, walls=self.walls,
                        periodic=(self.periodic_x, self.periodic_y),
                        shear_rate=self.shear_rate)
        out.vel = self.vel.copy()
        out.omega = self.omega.copy()
        return out


# --------------------------------------------------------------------------- helpers
def _empty_contacts() -> dict[str, np.ndarray]:
    return {
        "i": np.zeros(0, dtype=np.int64),
        "j": np.zeros(0, dtype=np.int64),
        "kind": np.zeros(0, dtype=np.int8),
        "n": np.zeros((0, 2)),
        "lever": np.zeros((0, 2)),
        "f": np.zeros((0, 2)),
        "f_on_i": np.zeros((0, 2)),
        "fn": np.zeros(0),
        "ft": np.zeros(0),
        "overlap": np.zeros(0),
        "point": np.zeros((0, 2)),
    }


def _perp(vec: np.ndarray) -> np.ndarray:
    out = np.empty_like(vec)
    out[..., 0] = -vec[..., 1]
    out[..., 1] = vec[..., 0]
    return out


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# ---------------------------------------------------------------------------
# Scenario 1: single impact -- restitution vs the dt/t_c ladder
# ---------------------------------------------------------------------------
def measure_restitution(*, v_impact: float = 0.05, e_target: float = 0.5,
                        ratio: float = 0.02, n_sub: int | None = None,
                        radius: float = 0.025, density: float = 917.0,
                        young_modulus: float = 1.0e9, poisson_ratio: float = 0.3,
                        h_target_tc: float = 0.02, scheme: str = "mts",
                        enforce_gate: bool = True, allow_tight: bool = False,
                        max_contacts: int = 4) -> dict:
    """Drop one Hertzian disk on a rigid platen and measure the restitution.

    ``dt`` is set from the contract (``dt = ratio * t_c``) and the sub-cycle
    count from ``h <= h_target_tc * t_c`` unless ``n_sub`` is given explicitly
    (the negative control sets ``n_sub = 1`` on purpose).  ``e_reference`` is the
    fine-step value of the *same* integrator (``ratio = 0.002``, ``n_sub = 1``),
    so the error is an integration-error statement, not a model statement; the
    semi-analytic Hunt-Crossley value is reported next to it as the external
    anchor.
    """
    m = density * (4.0 / 3.0) * math.pi * radius ** 3
    k_h = hertz_stiffness_plane(young_modulus, poisson_ratio, radius)
    t_c = hertz_contact_time(m, k_h, v_impact)
    alpha = calibrate_alpha(e_target, v_impact, m, k_h)
    alpha_first_order = alpha_for_restitution(e_target, v_impact, m, k_h)
    dt = ratio * t_c
    if n_sub is None:
        n_sub = max(1, int(math.ceil(ratio / h_target_tc - 1e-12)))
    # the mass is rho * t * pi r^2, so pick t to reproduce the sphere mass
    cfg = MTSConfig(kn=k_h, contact_model="hertz", damping="hunt_crossley",
                    alpha=alpha, mu=0.0, kt_ratio=0.5, density=density,
                    thickness=m / (density * math.pi * radius ** 2),
                    dt=dt, n_sub=n_sub, scheme=scheme, skin=0.5 * radius,
                    enforce_gate=enforce_gate, allow_tight=allow_tight)
    wall = np.array([[[-0.2, 0.0], [0.2, 0.0]]])
    system = MTSSystem(np.array([[0.0, radius]]), np.array([radius]), cfg, walls=wall)
    system.vel[0, 1] = -v_impact
    contact_steps = 0
    contacts = 0
    in_contact = False
    e_measured = None
    max_steps = int(8.0 * t_c / dt) + 100
    for _ in range(max_steps):
        system.step(1)
        overlap = radius - float(system.pos[0, 1])
        if overlap > 0.0:
            contact_steps += 1
            if not in_contact:
                in_contact = True
                contacts += 1
        elif in_contact:
            in_contact = False
            e_measured = abs(float(system.vel[0, 1])) / v_impact
            if contacts >= 1:
                break
    if e_measured is None:
        raise RuntimeError("impact did not separate inside the step budget")
    e_semi = hunt_crossley_restitution(v_impact, m, k_h, alpha)
    gate = stability_gate(dt, t_c)
    return {
        "v_impact": float(v_impact),
        "e_target": float(e_target),
        "alpha": float(alpha),
        "alpha_first_order": float(alpha_first_order),
        "dt": float(dt),
        "dt_over_tc": float(ratio),
        "n_sub": int(n_sub),
        "h_over_tc": float(dt / n_sub / t_c),
        "contact_time": float(t_c),
        "contact_steps": int(contact_steps),
        "restitution_measured": float(e_measured),
        "restitution_semi_analytic": float(e_semi),
        "gate": gate,
        "scheme": scheme,
        "mass": float(m),
        "hertz_stiffness": float(k_h),
    }


def restitution_ladder(*, e_targets: Sequence[float] = (0.5, 0.9),
                       ratios: Sequence[float] = (0.02, 0.05, 0.1, 0.2),
                       v_impact: float = 0.05, h_target_tc: float = 0.02,
                       **kwargs) -> dict:
    """Restitution over the ``dt/t_c`` ladder, scored against the fine reference.

    Returns ``rows`` (one per ``(e, ratio)``), the reference per ``e``, and the
    worst errors that the acceptance criterion reads.
    """
    references = {}
    for e_target in e_targets:
        ref = measure_restitution(v_impact=v_impact, e_target=e_target,
                                  ratio=0.002, n_sub=1, scheme="mts",
                                  h_target_tc=h_target_tc, **kwargs)
        references[float(e_target)] = ref
    rows = []
    for e_target in e_targets:
        ref = references[float(e_target)]
        for ratio in ratios:
            row = measure_restitution(v_impact=v_impact, e_target=e_target,
                                      ratio=ratio, h_target_tc=h_target_tc, **kwargs)
            row["restitution_reference"] = ref["restitution_measured"]
            row["restitution_error"] = (row["restitution_measured"]
                                        / ref["restitution_measured"] - 1.0)
            semi = row["restitution_semi_analytic"]
            # the first-order Hunt-Crossley formula is only meaningful while
            # W/E0 << 1; at e = 0.5 it collapses to zero (documented in the
            # report), so the deviation is reported as None there instead of
            # dividing by zero
            row["restitution_error_vs_semi"] = (
                row["restitution_measured"] / semi - 1.0 if semi > 1e-9 else None)
            row["alpha_first_order_over_calibrated"] = (
                row["alpha_first_order"] / row["alpha"] if row["alpha"] else float("nan"))
            rows.append(row)
    worst = max(abs(r["restitution_error"]) for r in rows)
    return {
        "rows": rows,
        "references": {str(k): v for k, v in references.items()},
        "worst_abs_restitution_error": float(worst),
        "within_2pct": bool(worst <= 0.02),
    }


# ---------------------------------------------------------------------------
# Scenario 2: undamped doubly periodic gas -- energy / momentum ledger
# ---------------------------------------------------------------------------
def _participating_radius(phi: float, n: int, radius: float) -> float:
    box = math.sqrt(n * math.pi * radius ** 2 / phi)
    spacing = box / math.sqrt(n)
    if 2.0 * radius > spacing:
        radius = 0.45 * spacing
    return radius


def _gas_system(phi: float, n: int, ratio: float, *, n_sub: int | None,
                scheme: str, radius: float = 5.0e-3, density: float = 2000.0,
                young_modulus: float = 1.0e9, poisson_ratio: float = 0.25,
                seed: int = 0, h_target_tc: float = 0.02,
                enforce_gate: bool = True, shear_rate: float = 0.0,
                mu: float = 0.0, contact_model: str = "linear",
                kn: float = 1.0e5, zeta: float = 0.4,
                damping_time: float = 0.0, bidisperse: bool = False,
                random_positions: bool = False,
                lees_edwards: bool = False) -> tuple[MTSSystem, float]:
    """Doubly periodic packing of ``n`` disks at volume fraction ``phi``.

    ``bidisperse``/``random_positions`` make the preparation amorphous: a grid
    (even jittered) relaxes into a crystal, and a 2D crystal carries no shear
    stress and mobilises no friction by symmetry, so it cannot probe the friction
    arm at all.
    """
    rng = np.random.default_rng(seed)
    radius = _participating_radius(phi, n, radius)
    if bidisperse:
        from .granular2d.jamming import bidisperse_radii

        radii_all = bidisperse_radii(n, radius, seed=seed)
        radius = float(np.mean(radii_all))
    else:
        radii_all = None
    box = math.sqrt(float(np.sum(math.pi * (radii_all ** 2 if radii_all is not None
                                            else np.full(n, radius) ** 2))) / phi)
    side = box / math.sqrt(n)
    grid = int(math.ceil(math.sqrt(n)))
    xs = (np.arange(grid) + 0.5) * (box / grid)
    xx, yy = np.meshgrid(xs, xs)
    if random_positions:
        rmax_local = float(np.max(radii_all if radii_all is not None else radius))
        pos = rmax_local + rng.random((n, 2)) * (box - 2.0 * rmax_local)
    else:
        pos = np.column_stack([xx.ravel(), yy.ravel()])[:n]
        pos = pos + (rng.random(pos.shape) - 0.5) * 0.1 * side
        pos = np.mod(pos, box)
    stiffness = (kn if contact_model == "linear"
                 else hertz_stiffness_pair(young_modulus, poisson_ratio, radius, radius))
    m_part = density * (4.0 / 3.0) * math.pi * radius ** 3
    thickness = m_part / (density * math.pi * radius ** 2)
    delta_ref = 0.01 * radius
    if shear_rate:
        v_scale = shear_rate * box * 0.1
    else:
        # thermal kick at a reference overlap: v = omega * delta_ref, i.e. the
        # velocity a contact carries when it is compressed by delta_ref
        v_scale = math.sqrt(stiffness / m_part) * delta_ref
    # a *linear* spring has a speed independent contact time, which is what a
    # many-body stability contract should key on: t_c = pi sqrt(m_min / k_n)
    t_c = linear_contact_time(m_part, stiffness)
    dt = ratio * t_c
    if n_sub is None:
        n_sub = max(1, int(math.ceil(ratio / h_target_tc - 1e-12)))
    cfg = MTSConfig(kn=stiffness, contact_model=contact_model,
                    damping="none" if mu == 0.0 else "dashpot", zeta=zeta,
                    mu=mu, kt_ratio=0.5, density=density, thickness=thickness,
                    damping_time=damping_time,
                    dt=dt, n_sub=n_sub, scheme=scheme, skin=0.5 * radius,
                    enforce_gate=enforce_gate)
    system = MTSSystem(pos, (radii_all if radii_all is not None
                             else np.full(n, radius)), cfg, periodic=(box, box),
                       shear_rate=shear_rate, lees_edwards=lees_edwards)
    system.vel = (rng.random((n, 2)) - 0.5) * v_scale
    return system, box


def gas_ledger(*, n: int = 48, phi: float = 0.60, ratio: float = 0.2,
               n_sub: int | None = None, steps: int = 100_000,
               scheme: str = "mts", seed: int = 0, h_target_tc: float = 0.02,
               confirm_every: int = 500, kn: float = 1.0e5,
               enforce_gate: bool = True) -> dict:
    """Undamped Hertzian gas: mechanical-energy drift and momentum conservation.

    The ledger is *structural*: the contact force of a pair is accumulated as
    ``+f`` on ``i`` and ``-f`` on ``j``, so the total momentum is conserved to
    the round-off of the accumulation order.  The energy drift measures the
    integrator.
    """
    system, box = _gas_system(phi, n, ratio, n_sub=n_sub, scheme=scheme,
                              seed=seed, h_target_tc=h_target_tc, kn=kn,
                              mu=0.0, enforce_gate=enforce_gate)
    system._slow_pass()
    system._substep(0.0, passes=1)
    e0 = system.energy()
    if e0 <= 0.0:
        raise ValueError("the gas has no initial mechanical energy: pick a denser "
                         "packing or a non-zero velocity scale")
    p0 = system.momentum().copy()
    vscale = float(np.sum(system.mass[:, None] * np.linalg.norm(system.vel, axis=1)))
    system._p0 = p0.copy()
    trace = []
    max_rel_energy = 0.0
    for k in range(steps):
        system.step(1)
        if (k + 1) % confirm_every == 0 or k == 0:
            energy = system.energy()
            rel = energy / e0 - 1.0
            max_rel_energy = max(max_rel_energy, abs(rel))
            trace.append((float(system.time), float(rel),
                          float(np.linalg.norm(system.momentum() - p0))))
    system._slow_pass()
    system._substep(0.0, passes=1)
    energy_end = system.energy()
    p_end = system.momentum()
    dp = float(np.linalg.norm(p_end - p0))
    return {
        "n": int(n),
        "phi": float(phi),
        "box": float(box),
        "ratio": float(ratio),
        "n_sub": int(system.config.n_sub),
        "steps": int(steps),
        "scheme": scheme,
        "energy_initial": float(e0),
        "energy_end": float(energy_end),
        "energy_drift_rel": float(energy_end / e0 - 1.0),
        "energy_max_abs_drift_rel": float(max_rel_energy),
        "momentum_drift_abs": dp,
        "momentum_drift_rel": float(dp / vscale) if vscale > 0.0 else 0.0,
        "trace": trace,
        # the *secular* drift is the end-to-end change; the sampled maximum is
        # reported next to it because a symplectic integrator's energy error
        # oscillates around the true value instead of growing
        "within_1pct": bool(abs(energy_end / e0 - 1.0) <= 0.01),
        "oscillation_amplitude_rel": float(max_rel_energy),
        "momentum_within_1e12": bool(dp / vscale <= 1e-12 if vscale > 0.0 else True),
    }


# ---------------------------------------------------------------------------
# Scenario 3: doubly periodic shear cell (friction arm, physics unchanged)
# ---------------------------------------------------------------------------
def pressure_and_shear(system: MTSSystem, area: float | None = None) -> dict:
    """Love-Weber pressure and shear stress of the last contact evaluation.

    Uses the same definition as :func:`tensordem.granular2d.stress.virial_tensor`
    (``sigma = sum branch (x) f_on_i / A``, ``P = -tr(sigma)/2``), so the number
    is comparable with the shipped channel/jamming code.
    """
    c = system.contacts
    if area is None:
        area = system.domain_area()
    if area is None or area <= 0.0:
        raise ValueError("area must be positive")
    n_pair = int(np.sum(c["kind"] == 0)) if len(c["i"]) else 0
    if not n_pair:
        return {"pressure": 0.0, "shear": 0.0, "coord": 0.0}
    i = c["i"][:n_pair]
    j = c["j"][:n_pair]
    if "branch" in c and len(c["branch"]) == len(c["i"]):
        # the branch vector the force was evaluated on (image offset and, for a
        # sheared lattice, the tilt included)
        branch = c["branch"][:n_pair]
    else:
        branch = system.pos[j] - system.pos[i]
        branch = system._minimum_image(branch)
    f = c["f_on_i"][:n_pair]
    w = np.einsum("ma,mb->ab", branch, f) / area
    sigma = 0.5 * (w + w.T)
    return {
        "pressure": float(-0.5 * np.trace(sigma)),
        "shear": float(sigma[0, 1]),
        "coord": float(2.0 * n_pair / system.n),
    }


def shear_cell_run(*, phi: float = 0.80, mu: float = 0.5, n: int = 64,
                   gamma_dot: float = 20.0, ratio: float = 0.2,
                   n_sub: int | None = None, scheme: str = "mts",
                   relax_steps: int = 4000, steps: int = 2000,
                   sample_every: int = 20, seed: int = 0,
                   h_target_tc: float = 0.02, contact_model: str = "hertz",
                   young_modulus: float = 1.0e9, poisson_ratio: float = 0.25,
                   density: float = 2000.0, radius: float = 5.0e-3,
                   zeta: float = 0.4, kn: float = 1.0e5,
                   thermostat_tc: float = 40.0, bidisperse: bool = True,
                   lees_edwards: bool = False) -> dict:
    """Doubly periodic shear cell: macroscopic friction and the dissipation ledger.

    Preparation: a jittered periodic packing is relaxed under a dashpot
    (overdamped, no shear) to a force-balanced state, then sheared at
    ``gamma_dot`` for ``steps`` outer steps with the velocity gradient entering
    the contact relative velocity (SLLOD without the Lees-Edwards tilt).

    Reported: steady-state ``tau``, ``P``, ``tau/P``, the fraction of contacts at
    the Coulomb cone (sliding contacts), the friction-dissipation ledger
    ``sum f_t . du_t + dE_t``, and the wall clock.
    """
    system, box = _gas_system(phi, n, ratio, n_sub=n_sub, scheme=scheme,
                              seed=seed, h_target_tc=h_target_tc, radius=radius,
                              density=density, young_modulus=young_modulus,
                              poisson_ratio=poisson_ratio, shear_rate=gamma_dot,
                              lees_edwards=lees_edwards,
                              mu=mu, contact_model=contact_model, kn=kn,
                              zeta=zeta, bidisperse=bidisperse,
                              random_positions=True,
                              damping_time=(thermostat_tc * linear_contact_time(
                                  float(np.mean(density * (4.0 / 3.0) * math.pi
                                                * radius ** 3)), kn)
                                  if thermostat_tc > 0 else 0.0))
    cfg = system.config
    # ``ratio * t_c`` is the MTS outer step, i.e. an arm independent physical time
    # unit: both friction arms must cover the *same physical duration* per stage,
    # otherwise they are compared on different protocols (that mistake made the
    # first version of this driver compare 10x different relaxations).
    step_ref = linear_contact_time(float(system.mass.min()), cfg.kn) * ratio
    # relax (overdamped, no shear) at the same substep resolution
    rel_cfg = replace(cfg, damping="dashpot", zeta=zeta, mu=0.0,
                      mu_rolling=0.0, scheme="mts",
                      n_sub=max(1, int(round(cfg.n_sub))))
    relax = MTSSystem(system.pos.copy(), system.radii.copy(), rel_cfg,
                      periodic=(box, box), shear_rate=0.0,
                      lees_edwards=lees_edwards)
    n_relax = max(1, int(round(relax_steps * step_ref / relax.dt)))
    t0 = time.perf_counter()
    relax.step(n_relax)
    relax_time = time.perf_counter() - t0
    system.pos = relax.pos.copy()
    system.vel = np.zeros_like(system.pos)
    system.omega = np.zeros(system.n)
    system._xi = np.zeros((0, 2))
    system._keys = np.zeros(0, dtype=np.int64)
    # sheared run
    dissip = 0.0
    samples = []
    contact_stats = {"mean_fn": [], "mean_ft": [], "mobilization": [],
                     "fraction_at_cone": []}
    # equal physical duration: the legacy arm takes ``ratio / h_target`` times
    # more steps, which is exactly the per-step cost the MTS amortises
    steps = max(1, int(round(steps * step_ref / system.dt)))
    t0 = time.perf_counter()
    system.step(1)
    for k in range(steps):
        # ``step`` already ends with a fresh contact evaluation, so no extra
        # neighbour rebuild is needed here (an earlier version did one per step and
        # inflated the legacy arm's cost)
        dissip += _friction_dissipation(system)
        system.step(1)
        if (k + 1) % sample_every == 0:
            st = pressure_and_shear(system)
            samples.append((float(system.time), st["pressure"], st["shear"],
                            st["coord"]))
            cs = system.contacts
            act = cs["overlap"] > 0.0
            if act.any():
                fn = cs["fn"][act]
                ft = cs["ft"][act]
                loaded = fn > 0.0
                contact_stats["mean_fn"].append(float(np.mean(fn)))
                contact_stats["mean_ft"].append(float(np.mean(ft)))
                mob = ft[loaded] / (mu * fn[loaded]) if (mu > 0 and loaded.any()) \
                    else np.zeros(1)
                contact_stats["mobilization"].append(float(np.mean(mob)))
                contact_stats["fraction_at_cone"].append(float(np.mean(mob >= 0.999)))
    shear_time = time.perf_counter() - t0
    tail = samples[max(0, len(samples) // 2):]
    tau = float(np.mean([s[2] for s in tail])) if tail else 0.0
    p = float(np.mean([s[1] for s in tail])) if tail else 0.0
    z = float(np.mean([s[3] for s in tail])) if tail else 0.0
    # sampling error of the two stress averages: the deviatoric stress of a
    # 64-disk cell is a near-cancelling difference, so its *uncertainty* is what
    # a cross-arm comparison has to be judged against
    tau_se = float(np.std([s[2] for s in tail]) / math.sqrt(len(tail))) if len(tail) > 1 else float("nan")
    p_se = float(np.std([s[1] for s in tail]) / math.sqrt(len(tail))) if len(tail) > 1 else float("nan")
    tau_std = float(np.std([s[2] for s in tail])) if len(tail) > 1 else float("nan")
    st = pressure_and_shear(system)
    ft = system.contacts["ft"]
    fn = system.contacts["fn"]
    active = system.contacts["overlap"] > 0.0
    sliding = (fn > 0) & active & (np.abs(ft - mu * fn) <= 1e-9 * np.maximum(fn, 1e-30))
    frac_sliding = float(np.mean(sliding[active])) if active.any() else 0.0
    stats_mean = {k: (float(np.mean(v)) if len(v) else float("nan"))
                  for k, v in contact_stats.items()}
    return {
        "phi": float(phi),
        "mu": float(mu),
        "contact_stats": stats_mean,
        "n": int(n),
        "gamma_dot": float(gamma_dot),
        "ratio": float(ratio),
        "n_sub": int(cfg.n_sub),
        "scheme": scheme,
        "pressure": p,
        "shear": tau,
        "tau_over_p": float(tau / p) if p > 0 else float("nan"),
        "shear_stderr": tau_se,
        "pressure_stderr": p_se,
        "shear_std": tau_std,
        "samples_averaged": int(len(tail)),
        "coord": z,
        "fraction_sliding": frac_sliding,
        "friction_dissipation": float(dissip),
        "relax_wall_clock": float(relax_time),
        "relax_steps": int(n_relax),
        "shear_wall_clock": float(shear_time),
        "samples": samples,
        "final_pressure": st["pressure"],
        "final_shear": st["shear"],
        # periodic-lattice information (m17503): a cross-arm comparison has to
        # know whether the cell sheared with the flow or not
        "lees_edwards": bool(lees_edwards),
        "strain_end": float(system.strain),
    }


def _friction_dissipation(system: MTSSystem) -> float:
    """``sum |f_t| * |du_t|`` of the last evaluation (sliding work, positive)."""
    last = system._last
    if not last:
        return 0.0
    active = last["active"]
    if not active.any():
        return 0.0
    xi = last["xi"][active]
    ft = last["kt"][active] * np.linalg.norm(xi, axis=1)
    return float(np.sum(ft))


# ---------------------------------------------------------------------------
# Scenario 4: frictional jamming with a pressure servo (phi_J)
# ---------------------------------------------------------------------------
def _scale_box(system: MTSSystem, scale: float) -> None:
    system.pos *= scale
    if len(system.walls):
        system.walls = system.walls * scale


def _servo_state(system: MTSSystem) -> dict:
    # fresh contact records, whatever the arm wrote last (the legacy arm's first
    # two iterations deliberately skip the record construction)
    system._forces(0.0, passes=1, record=True)
    area = system.domain_area()
    st = pressure_and_shear(system, area=area)
    return {"phi": float(system.area_fraction(area)), "pressure": st["pressure"],
            "side": float(np.sqrt(area)) if area else float("nan")}


def jamming_protocol(*, scheme: str = "mts", mu: float = 0.5, mu_rolling: float = 0.05,
                     n: int = 256, mean_radius: float = 0.008, seed: int = 0,
                     ratio: float = 0.2, h_target_tc: float = 0.02,
                     p_compress: float = 3.0e3,
                     compress_time: float = 0.020,
                     levels: Sequence[float] = (300.0, 100.0, 30.0, 10.0, 3.0, 1.0),
                     level_time: float = 0.040, eta: float = 150.0,
                     dt_ref: float = 1.53e-5, max_step: float = 3.0e-4,
                     kn: float = 3.0e4, kt_ratio: float = 0.5,
                     density: float = 917.0, thickness: float = 0.03,
                     zeta: float = 0.4, damping_time: float = 1.0e-3,
                     every: int = 20, extrapolate_below: float = 30.0) -> dict:
    """Dense-branch isotropic decompression protocol on the MTS integrator.

    Physics identical to :func:`tensorderm.granular2d.jamming.jamming_density_isotropic`
    (linear normal law ``k_n``, dashpot ``zeta``, tangential spring with Coulomb
    clipping, rolling resistance, servo ``d ln L / dt = (P - P_target)/eta``
    clipped at ``max_step/dt_ref``), but the integrator is selectable:

    ``legacy3x``  monorate at ``dt = h_target_tc * t_c`` with three return-mapping
                  passes per step -- the arm whose per-step cost is 3x.
    ``mts``       outer step ``ratio * t_c`` with ``n_sub = ratio / h_target_tc``
                  cached-geometry substeps, one pass each.

    Both arms integrate the *same physical duration* per stage, so the wall
    clock difference is the integration scheme itself.
    """
    from .granular2d.jamming import bidisperse_radii

    radii = bidisperse_radii(n, mean_radius, seed=seed)
    side = math.sqrt(float(np.sum(math.pi * radii ** 2)) / 0.45)
    rng = np.random.default_rng(seed + 11)
    rmax = float(radii.max())
    pos = rmax + rng.random((n, 2)) * (side - 2.0 * rmax)
    cfg = MTSConfig(kn=kn, kt_ratio=kt_ratio, mu=mu, mu_rolling=mu_rolling,
                    contact_model="linear", damping="dashpot", zeta=zeta,
                    density=density, thickness=thickness,
                    damping_time=damping_time, dt=1.0, n_sub=1, scheme="mts",
                    skin=0.003)
    m_min = float((density * thickness * math.pi * radii ** 2).min())
    t_c = linear_contact_time(m_min, kn)
    h_star = h_target_tc * t_c
    if scheme == "legacy3x":
        dt = h_star
        n_sub = 1
    else:
        dt = ratio * t_c
        n_sub = max(1, int(math.ceil(ratio / h_target_tc - 1e-12)))
    cfg = replace(cfg, dt=dt, n_sub=n_sub, scheme=scheme)
    walls = np.array([[[0.0, 0.0], [side, 0.0]], [[side, 0.0], [side, side]],
                      [[side, side], [0.0, side]], [[0.0, side], [0.0, 0.0]]])
    system = MTSSystem(pos, radii, cfg, walls=walls)
    rate_cap = max_step / dt_ref * dt  # strain increment cap per outer step
    wall_clock = 0.0

    def run_stage(target: float, duration: float) -> dict:
        nonlocal wall_clock
        n_steps = max(1, int(round(duration / dt)))
        t0 = time.perf_counter()
        for k in range(n_steps):
            if k % every == 0:
                system._slow_pass()
                st = _servo_state(system)
                pressure = st["pressure"]
            delta = dt * (pressure - target) / eta
            delta = float(min(max(delta, -rate_cap), rate_cap))
            _scale_box(system, 1.0 + delta)
            system.step(1)
        system._slow_pass()
        st = _servo_state(system)
        wall_clock += time.perf_counter() - t0
        st["steps"] = n_steps
        return st

    compress = run_stage(p_compress, compress_time)
    series = []
    for level in levels:
        st = run_stage(float(level), level_time)
        series.append([float(level), float(st["phi"]), float(st["pressure"])])
    result = phi_j_from_levels(series, extrapolate_below=extrapolate_below)
    result.update({
        "scheme": scheme,
        "mu": float(mu),
        "mu_rolling": float(mu_rolling),
        "n": int(n),
        "mean_radius": float(mean_radius),
        "seed": int(seed),
        "ratio": float(ratio if scheme != "legacy3x" else h_target_tc),
        "n_sub": int(n_sub),
        "dt": float(dt),
        "contact_time": float(t_c),
        "gate": stability_gate(dt, t_c),
        "compress": compress,
        "series": series,
        "wall_clock": float(wall_clock),
        "kinematics_calls": int(system.kinematics_calls),
        "return_mapping_passes": int(system.return_mapping_passes),
    })
    return result


def phi_j_from_levels(series, *, extrapolate_below: float = 30.0) -> dict:
    """``phi_J`` from the ``P -> 0`` extrapolation of the dense branch."""
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3) or len(arr) < 2:
        raise ValueError("series must be a sequence of (P, phi[, P_measured]) rows")
    p = arr[:, 0]
    phi = arr[:, 1]
    sel = p <= extrapolate_below
    if int(np.sum(sel)) < 2:
        sel = np.ones_like(p, dtype=bool)
    slope, intercept = np.polyfit(p[sel], phi[sel], 1)
    return {
        "phi_j": float(intercept),
        "dphi_dp": float(slope),
        "fit_points": int(np.sum(sel)),
        "phi_at_lowest_pressure": float(phi[-1]),
    }


def shard_force_consistency(*, phi: float = 0.80, mu: float = 0.5, n: int = 64,
                            ratio: float = 0.2, n_sub: int | None = None,
                            seed: int = 0, shear_rate: float = 1.0,
                            kn: float = 1.0e5, shard_counts=(1, 2, 4)) -> dict:
    """Force of the *same* state accumulated over different shard partitions.

    The tangential history is saved and restored around every evaluation, so the
    three force values are functions of one single state: any difference is the
    floating-point difference of the accumulation order alone.
    """
    system, box = _gas_system(phi, n, ratio, n_sub=n_sub, scheme="mts",
                              seed=seed, shear_rate=shear_rate, mu=mu, kn=kn)
    system._slow_pass()
    reference = None
    out = {}
    for shards in shard_counts:
        system.config = replace(system.config, shards=int(shards))
        xi, roll, keys = system._xi.copy(), system._roll.copy(), system._keys.copy()
        force, torque = system._forces(system.dt / max(1, system.config.n_sub),
                                      passes=1, record=False)
        system._xi, system._roll, system._keys = xi, roll, keys
        if reference is None:
            reference = force.copy()
            out[str(shards)] = {"max_abs_force_diff": 0.0, "max_rel_force_diff": 0.0}
        else:
            scale = max(float(np.max(np.abs(reference))), 1e-300)
            out[str(shards)] = {
                "max_abs_force_diff": float(np.max(np.abs(force - reference))),
                "max_rel_force_diff": float(np.max(np.abs(force - reference)) / scale),
            }
    worst = max(v["max_rel_force_diff"] for v in out.values())
    return {"per_shard": out, "worst_rel_force_diff": float(worst),
            "bitwise_identical": bool(worst == 0.0)}


def run_shards(*, shards: int, phi: float = 0.80, mu: float = 0.5, n: int = 64,
               ratio: float = 0.2, n_sub: int | None = None, steps: int = 200,
               seed: int = 0, kn: float = 1.0e5, **kwargs) -> dict:
    """Shear-cell trajectory with the contact force accumulated over ``shards``."""
    system, box = _gas_system(phi, n, ratio, n_sub=n_sub, scheme="mts",
                              seed=seed, shear_rate=1.0, mu=mu, kn=kn, **kwargs)
    system.config = replace(system.config, shards=int(shards))
    system.step(steps)
    st = pressure_and_shear(system)
    return {
        "shards": int(shards),
        "pressure": st["pressure"],
        "shear": st["shear"],
        "vel": system.vel.copy(),
        "pos": system.pos.copy(),
    }
