"""Blade-element / momentum open-water model (supporting model for milling).

The milling simulator needs two things from hydrodynamics:

1. the hydrodynamic torque ``Q_hydro(t)`` that the drive has to overcome beside
   the ice load, and
2. the axial induced velocity in the propeller race, which advects the ice chips.

Both come from a blade-element / momentum (BEM) solve of the same
:class:`~tensordem.propeller_ice.geometry.PropellerGeometry`:

* section loads from a linear lift slope ``2 pi alpha`` up to the stall angle and
  the flat-plate separated laws ``C_L = 2 sin a cos a``, ``C_D = 2 sin^2 a``
  beyond it (standard post-stall continuation, no tabulated section data);
* axial/angular induction ``a``, ``a'`` from the momentum identities
  ``dT = 4 pi r rho V^2 (1+a) a F dr`` and
  ``dQ = 4 pi r^3 rho V omega (1+a) a' F dr`` with the Prandtl tip-loss factor
  ``F``.  The classical relation ``CT = 4 F a (1+a)`` is inverted in closed form
  and is *continued* above its empirical validity boundary ``CT_s = 0.96 F``
  (i.e. ``a_s = 0.2``) by the quadratic high-loading closure of
  :func:`empirical_ct`, built from the same three conditions Glauert's empirical
  actuator-disk parabola uses (value, slope and ``CT(1) = 2``); the published
  turbine-convention instance of that construction is :func:`_buhl`;
* two *independent* residuals are returned: ``residual`` (max ``|delta a|`` of
  the fixed point iteration) and ``momentum_residual`` (blade-element versus
  annulus-momentum thrust at the converged state).  The second one is the
  physical one: it is exactly the quantity that a chord/solidity combination too
  heavily loaded for the annular streamtube model makes non-zero, and
  ``model_valid`` / ``validity_flag`` report that condition instead of letting
  the induction cap absorb it.

The result is a standard open-water diagram ``K_T(J)``, ``K_Q(J)``, ``eta(J)``
plus the radial induction profile.  This module is *supporting*: it is verified
against its own momentum identities and strip-count convergence, not against a
proprietary open-water chart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import PropellerGeometry

__all__ = ["OpenWaterState", "OpenWaterConfig", "solve_open_water", "section_coefficients"]

_STALL = math.radians(12.0)


def section_coefficients(alpha: np.ndarray | float, cd0: float = 0.012,
                         cl_slope: float = 2.0 * math.pi) -> tuple[Any, Any]:
    """2-D section ``C_L, C_D`` with separated post-stall continuation.

    ``|alpha| <= 12 deg``: attached flow, ``C_L = cl_slope * alpha``,
    ``C_D = cd0 + 0.010 C_L^2``.  Beyond stall: ``C_L = 2 sin a cos a``,
    ``C_D = cd0 + 2 sin^2 a`` (flat-plate separated flow).
    """
    a = np.asarray(alpha, dtype=float)
    attached = np.abs(a) <= _STALL
    cl_att = cl_slope * a
    cl_sep = 2.0 * np.sin(a) * np.cos(a)
    cl = np.where(attached, cl_att, cl_sep)
    cd_att = cd0 + 0.010 * cl_att ** 2
    cd_sep = cd0 + 2.0 * np.sin(a) ** 2
    cd = np.where(attached, cd_att, cd_sep)
    return cl, cd


@dataclass(frozen=True)
class OpenWaterConfig:
    """Open-water solve controls."""

    n_strips: int = 40
    rho: float = 1025.0
    kinematic_viscosity: float = 1.8e-6
    cd0: float = 0.012
    iterations: int = 1500
    relaxation: float = 0.10
    convergence_tol: float = 1.0e-8
    a_cap: float = 0.85
    # "classical"     a from the closed-form classical annulus-momentum
    #                 inversion, with the empirical high-loading closure used to
    #                 *report* whether the loading is still inside the model's
    #                 validity bound CT_s = ct_limit * F (repaired default: well
    #                 conditioned, and it can no longer claim convergence while
    #                 violating its own momentum identity).
    # "high_loading"  iterate on the empirical closure itself, so the state is
    #                 the closure's root where one exists; where none exists the
    #                 strip is pinned to the branch end and flagged.
    # "clipped"       the pre-repair behaviour (legacy fixed point clipped at
    #                 a_cap); kept only so the defect stays reproducible in the
    #                 verification campaign.  It reports the same validity
    #                 diagnostics, which is what makes the defect visible.
    induction_model: str = "classical"
    # Empirical actuator-disk saturation thrust coefficient that defines the
    # validity boundary of the annular momentum relation (Glauert / Buhl).
    ct_limit: float = 0.96

    def __post_init__(self) -> None:
        if self.n_strips < 4:
            raise ValueError("n_strips must be >= 4")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive")
        if self.induction_model not in ("classical", "high_loading", "clipped"):
            raise ValueError(
                "induction_model must be 'classical', 'high_loading' or 'clipped'")
        if not 0.0 < self.ct_limit < 1.0:
            raise ValueError("ct_limit must lie in (0, 1)")
        if not 0.5 < self.a_cap < 1.0:
            raise ValueError("a_cap must lie in (0.5, 1)")


@dataclass(frozen=True)
class OpenWaterState:
    """Result of one open-water (BEM) solve."""

    advance_ratio: float
    thrust_coefficient: float
    torque_coefficient: float
    efficiency: float
    thrust: float
    torque: float
    induced_mean: float
    residual: float
    iterations: int
    r: np.ndarray
    a: np.ndarray
    a_prime: np.ndarray
    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    # --- independent (physical) convergence / validity diagnostics ---------
    thrust_momentum: float = float("nan")
    momentum_residual: float = float("nan")
    model_valid: bool = False
    validity_flag: str = "not_evaluated"
    n_strips_above_ct_limit: int = 0
    ct_max_loading: float = float("nan")
    ct_local: np.ndarray | None = None
    induction_model: str = "classical"
    converged: bool = False

    # -- derived transport quantities --------------------------------------
    def race_axial_velocity(self, advance_velocity: float, factor: float = 1.5) -> float:
        """Axial velocity used to advect debris: ``V_a + factor * u_induced``."""
        return float(advance_velocity + factor * self.induced_mean)

    def as_dict(self) -> dict[str, Any]:
        return {
            "J": self.advance_ratio,
            "KT": self.thrust_coefficient,
            "KQ": self.torque_coefficient,
            "eta": self.efficiency,
            "thrust_N": self.thrust,
            "torque_Nm": self.torque,
            "induced_mean_m_s": self.induced_mean,
            "bem_residual": self.residual,
            "bem_iterations": self.iterations,
            "thrust_momentum_N": self.thrust_momentum,
            "momentum_residual": self.momentum_residual,
            "model_valid": self.model_valid,
            "validity_flag": self.validity_flag,
            "n_strips_above_ct_limit": self.n_strips_above_ct_limit,
            "ct_max_loading": self.ct_max_loading,
            "induction_model": self.induction_model,
            "converged": self.converged,
        }


def momentum_breakdown_induction(ct_limit: float = 0.96) -> float:
    """Induction where the *classical propeller* relation reaches ``ct_limit * F``.

    ``4 a (1 + a) = ct_limit`` (``F`` cancels), so
    ``a_s = (-1 + sqrt(1 + ct_limit)) / 2``.  For the Glauert empirical
    actuator-disk saturation value ``ct_limit = 0.96`` this is exactly ``0.2``.
    Above ``a_s`` the annular streamtube assumption is no longer supported by
    the classical momentum relation and a closure is required.
    """
    if not 0.0 < ct_limit < 1.0:
        raise ValueError("ct_limit must lie in (0, 1)")
    return 0.5 * (-1.0 + math.sqrt(1.0 + ct_limit))


def _turbine_breakdown_induction(ct_limit: float = 0.96) -> float:
    """Same boundary for the wind-turbine convention ``CT = 4 F a (1 - a)``."""
    if not 0.0 < ct_limit <= 1.0:
        raise ValueError("ct_limit must lie in (0, 1]")
    return 0.5 * (1.0 - math.sqrt(1.0 - ct_limit))


def classical_induction(ct: np.ndarray, F: np.ndarray) -> np.ndarray:
    """``a`` from the classical propeller annulus-momentum relation.

    Solves ``CT = 4 F a (1 + a)`` exactly: ``a = (-1 + sqrt(1 + 4 CT / (4F))) / 2``
    (the ``4`` cancels once, hence ``CT / F`` under the root after the ``4a(1+a)``
    normalisation).  A negative radicand (``CT < -F``, extreme windmill) is
    clamped to ``a = -1`` and left to the caller's induction clip.
    """
    ct = np.asarray(ct, dtype=float)
    F = np.asarray(F, dtype=float)
    xi = ct / np.maximum(4.0 * F, 1.0e-12)
    return 0.5 * (-1.0 + np.sqrt(np.maximum(1.0 + 4.0 * xi, 0.0)))


def empirical_ct_coefficients(F: float, ct_limit: float = 0.96,
                              base: str = "propeller") -> tuple[float, float, float]:
    """``(c0, c1, c2)`` of ``CT(a) = c0 + c1 a + c2 a**2`` on the high-loading branch.

    The parabola is fixed by the three conditions Glauert's empirical
    actuator-disk relation uses: value and slope continuity with the classical
    momentum relation *in the requested convention* at the breakdown induction,
    and the measured high-loading anchor ``CT(a = 1) = 2`` (``F``-independent).

    ``base="turbine"`` reproduces Buhl's published coefficients exactly
    ``(8/9, 4F - 40/9, 50/9 - 4F)``; that identity is a test gate, not a claim.
    ``base="propeller"`` is the same construction for this module's own
    convention ``CT = 4 F a (1 + a)``.
    """
    F = float(F)
    if base == "propeller":
        a_s = momentum_breakdown_induction(ct_limit)
        rhs = (ct_limit * F, 4.0 * F * (1.0 + 2.0 * a_s), 2.0)
    elif base == "turbine":
        a_s = _turbine_breakdown_induction(ct_limit)
        rhs = (ct_limit * F, 4.0 * F * (1.0 - 2.0 * a_s), 2.0)
    else:
        raise ValueError("base must be 'propeller' or 'turbine'")
    design = np.array([[1.0, a_s, a_s * a_s], [0.0, 1.0, 2.0 * a_s], [1.0, 1.0, 1.0]])
    c0, c1, c2 = np.linalg.solve(design, np.array(rhs, dtype=float))
    return float(c0), float(c1), float(c2)


def _propeller_coefficient_arrays(F: Any, ct_limit: float = 0.96
                                  ) -> tuple[Any, Any, Any]:
    """``(c0, c1, c2)(F)`` broadcast over ``F`` (the coefficients are affine in F)."""
    c_lo = empirical_ct_coefficients(0.0, ct_limit, base="propeller")
    c_hi = empirical_ct_coefficients(1.0, ct_limit, base="propeller")
    return tuple(np.asarray(c_lo[i], dtype=float) + F * (c_hi[i] - c_lo[i])
                 for i in range(3))


def empirical_ct(a: Any, F: Any, ct_limit: float = 0.96) -> Any:
    """High-loading thrust coefficient ``CT(a) = c0 + c1 a + c2 a**2``."""
    a, F = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(F, dtype=float))
    c0, c1, c2 = _propeller_coefficient_arrays(F, ct_limit)
    out = c0 + c1 * a + c2 * a * a
    return out if out.ndim else float(out)


def high_loading_branch_end(F: Any, ct_limit: float = 0.96,
                            a_cap: float = 0.85) -> Any:
    """Upper end of the *monotone* part of the high-loading branch.

    The parabola is concave for ``F > 0.3676`` (``c2 < 0``), so its inverse is
    single-valued only up to the vertex; beyond it the closure is reported as
    exceeded instead of being followed onto the falling side.
    """
    a_s = momentum_breakdown_induction(ct_limit)
    F = np.asarray(F, dtype=float)
    c0, c1, c2 = _propeller_coefficient_arrays(F, ct_limit)
    del c0
    with np.errstate(divide="ignore", invalid="ignore"):
        a_vertex = np.where(c2 < 0.0, -c1 / (2.0 * c2), float(a_cap))
    a_hi = np.where(c2 < 0.0, np.minimum(float(a_cap), a_vertex), float(a_cap))
    a_hi = np.maximum(a_hi, a_s)
    return a_hi if a_hi.ndim else float(a_hi)


def empirical_induction(ct: Any, F: Any, ct_limit: float = 0.96,
                        a_cap: float = 0.85) -> tuple[Any, Any]:
    """Invert the piecewise closure ``CT(a)``; returns ``(a, exceeded_mask)``.

    Below ``CT_s = ct_limit * F`` the classical relation ``CT = 4 F a (1 + a)``
    is inverted in closed form; above it the empirical parabola is inverted in
    closed form on its monotone segment ``[a_s, a_hi]``.  Where the requested
    ``CT`` exceeds ``CT(a_hi)`` the strip is pinned to ``a_hi`` and flagged, so
    the caller reports an unsupported loading instead of a converged one.
    """
    ct, F = np.broadcast_arrays(np.asarray(ct, dtype=float), np.asarray(F, dtype=float))
    a_s = momentum_breakdown_induction(ct_limit)
    c0, c1, c2 = _propeller_coefficient_arrays(F, ct_limit)
    a = classical_induction(ct, F)
    exceeded = np.zeros(ct.shape, dtype=bool)
    high = ct > ct_limit * F
    if np.any(high):
        a_hi = np.broadcast_to(high_loading_branch_end(F, ct_limit, a_cap), ct.shape)
        ct_max = c0 + c1 * a_hi + c2 * a_hi * a_hi
        target = ct[high]
        a_hi_h = np.asarray(a_hi, dtype=float)[high]
        over = target >= np.asarray(ct_max, dtype=float)[high]
        c0h, c1h, c2h = (np.asarray(c, dtype=float)[high] for c in (c0, c1, c2))
        disc = np.maximum(c1h * c1h - 4.0 * c0h * c2h + 4.0 * c2h * target, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            quad = (-c1h + np.sqrt(disc)) / (2.0 * c2h)
            linear = (target - c0h) / c1h
        root = np.where(np.abs(c2h) > 1.0e-9, quad, linear)
        root = np.where(over, a_hi_h, root)
        root = np.clip(root, a_s, a_hi_h)
        a[high] = root
        exceeded[high] = over
    return a, exceeded


def empirical_ct_max(F: Any, ct_limit: float = 0.96, a_cap: float = 0.85) -> Any:
    """Largest ``CT`` the closure can represent at the given loss factor ``F``."""
    a_hi = np.asarray(high_loading_branch_end(F, ct_limit, a_cap), dtype=float)
    out = np.asarray(empirical_ct(a_hi, F, ct_limit), dtype=float)
    return out if out.ndim else float(out)


def _buhl(a: np.ndarray, F: np.ndarray, cap: float) -> np.ndarray:
    """Published Buhl empirical high-induction correction (wind-turbine convention).

    **Not applied to this module's ``a``.**  ``_buhl`` solves the turbine form
    ``CT = 4 F a (1 - a)`` whose breakdown induction is ``a_t = 0.4``, while the
    propeller form used here, ``CT = 4 F a (1 + a)``, breaks down at
    ``a_p = 0.2`` and is not the same variable under any sign change
    (``a_t -> -a_p`` flips the sign of ``CT``).  The function is kept as the
    published reference instance of :func:`empirical_ct_coefficients`
    (``base="turbine"``) reproduces it bit-for-bit; the propeller branch is
    built by the same construction in :func:`empirical_ct`.
    """
    a_c = np.where(a > 0.4, 0.0, a)
    numerator = 18.0 * F - 20.0 - 3.0 * np.sqrt(np.maximum(F * (50.0 - 36.0 * F), 0.0))
    denominator = 36.0 * F - 50.0
    corrected = np.where(a > 0.4, numerator / denominator, a_c)
    return np.clip(corrected, -0.5, cap)


def solve_open_water(geometry: PropellerGeometry, revolutions_per_second: float,
                     advance_velocity: float,
                     config: OpenWaterConfig | None = None) -> OpenWaterState:
    """Solve the BEM system for one operating point ``(n, V_a)``.

    ``revolutions_per_second`` is ``n`` [1/s]; ``advance_velocity`` is the axial
    inflow ``V_a`` [m/s]; ``J = V_a / (n D)``.
    """
    cfg = config or OpenWaterConfig()
    n = float(revolutions_per_second)
    if n <= 0.0:
        raise ValueError("revolutions_per_second must be positive")
    va = float(advance_velocity)
    if va < 0.0:
        raise ValueError("advance_velocity must be non-negative")
    omega = 2.0 * math.pi * n
    strips = geometry.radial_strips(cfg.n_strips)
    r = strips["r"]
    dr = strips["dr"]
    chord = strips["chord"]
    phi_geom = strips["pitch_angle"]
    sigma = geometry.n_blades * chord / (2.0 * math.pi * r)
    tip = np.maximum(strips["tip_loss"], 1.0e-3)
    hub_span = geometry.hub_radius
    hub = (2.0 / math.pi) * np.arccos(np.clip(
        np.exp(-0.5 * geometry.n_blades * (r - hub_span) /
               np.maximum(r * np.sin(np.maximum(phi_geom, 1.0e-6)), 1.0e-9)), 0.0, 1.0))
    loss = np.clip(tip * np.clip(hub, 1.0e-3, 1.0), 1.0e-3, 1.0)

    a = np.zeros_like(r)
    a_prime = np.zeros_like(r)
    residual = math.inf
    iterations = 0
    bollard = va <= 1.0e-9
    closure_mode = cfg.induction_model == "high_loading"
    for iterations in range(1, cfg.iterations + 1):
        u_axial = va * (1.0 + a)
        u_tang = omega * r * (1.0 - a_prime)
        phi_in = np.arctan2(u_axial, np.maximum(u_tang, 1.0e-9))
        alpha = phi_geom - phi_in
        cl, cd = section_coefficients(alpha, cd0=cfg.cd0)
        w2 = u_axial ** 2 + u_tang ** 2
        cn = cl * np.cos(phi_in) - cd * np.sin(phi_in)
        ct = cl * np.sin(phi_in) + cd * np.cos(phi_in)

        # blade-element increment per unit radius (annulus, Z blades)
        d_load_n = 0.5 * cfg.rho * w2 * geometry.n_blades * chord * cn * dr
        d_load_t = 0.5 * cfg.rho * w2 * geometry.n_blades * chord * ct * r * dr

        # local thrust coefficient of the annulus (0.5 rho V_a^2 2 pi r dr basis)
        d_ref = 0.5 * cfg.rho * max(va, 0.0) ** 2 * 2.0 * math.pi * r * dr
        ct_local = np.where(d_ref > 0.0, d_load_n / np.maximum(d_ref, 1.0e-30), np.inf)

        if bollard:
            # V_a = 0 makes the CT convention degenerate (d_ref -> 0); the
            # annular momentum model is not solved here and the state is
            # reported with validity_flag="bollard_unsupported".
            a_new = np.full_like(a, cfg.a_cap)
        elif closure_mode:
            a_new, _ = empirical_induction(ct_local, loss, cfg.ct_limit, cfg.a_cap)
        elif cfg.induction_model == "classical":
            a_new = classical_induction(ct_local, loss)
        else:
            a_new = d_load_n / np.maximum(
                4.0 * math.pi * r * cfg.rho * va ** 2 * loss * dr, 1.0e-12)
            a_new = a_new / np.maximum(1.0 + a, 1.0e-9)
        a_prime_new = d_load_t / np.maximum(
            4.0 * math.pi * r ** 3 * cfg.rho * va * omega * loss * dr, 1.0e-12)
        a_prime_new = np.where(va > 1.0e-9, a_prime_new / np.maximum(1.0 + a, 1.0e-9), 0.0)
        a_new = np.clip(a_new, -0.3, cfg.a_cap)
        a_prime_new = np.clip(a_prime_new, -0.2, 0.5)
        if not np.all(np.isfinite(a_new)):
            a_new = np.nan_to_num(a_new, nan=0.0, posinf=cfg.a_cap)

        relax = cfg.relaxation
        a_old = a.copy()
        a = (1.0 - relax) * a + relax * a_new
        a_prime = (1.0 - relax) * a_prime + relax * a_prime_new
        residual = float(np.max(np.abs(a - a_old)))
        if residual < cfg.convergence_tol:
            break

    # final loads from the converged induction
    u_axial = va * (1.0 + a)
    u_tang = omega * r * (1.0 - a_prime)
    phi_in = np.arctan2(u_axial, np.maximum(u_tang, 1.0e-9))
    alpha = phi_geom - phi_in
    cl, cd = section_coefficients(alpha, cd0=cfg.cd0)
    w2 = u_axial ** 2 + u_tang ** 2
    cn = cl * np.cos(phi_in) - cd * np.sin(phi_in)
    ct = cl * np.sin(phi_in) + cd * np.cos(phi_in)
    d_thrust = 0.5 * cfg.rho * w2 * geometry.n_blades * chord * cn * dr
    thrust = float(np.sum(d_thrust))
    torque = float(np.sum(0.5 * cfg.rho * w2 * geometry.n_blades * chord * ct * r * dr))

    # --- independent physical residual: annulus momentum vs blade element ---
    # The momentum side is evaluated through the same piecewise closure that
    # defines the model: CT = 4 F a (1 + a) below the breakdown induction and the
    # empirical parabola above it.  The residual is therefore exactly zero for a
    # state that satisfies the model and non-zero for a cap-limited one -- which
    # is the quantity the iteration residual cannot see.
    a_s = momentum_breakdown_induction(cfg.ct_limit)
    d_ref = 0.5 * cfg.rho * max(va, 0.0) ** 2 * 2.0 * math.pi * r * dr
    ct_local = np.where(d_ref > 0.0, d_thrust / np.maximum(d_ref, 1.0e-30),
                        np.inf if bollard else 0.0)
    classical_ct = 4.0 * loss * a * (1.0 + a)
    ct_model = np.where(a <= a_s, classical_ct,
                        empirical_ct(np.minimum(a, cfg.a_cap), loss, cfg.ct_limit))
    d_momentum = ct_model * d_ref
    thrust_momentum = float(np.sum(d_momentum))
    scale = max(float(np.sum(np.abs(d_thrust))), 1.0e-30)
    momentum_residual = float(np.max(np.abs(d_thrust - d_momentum)) / scale)

    # --- validity of the annular streamtube model -------------------------
    ct_max = np.asarray(empirical_ct_max(loss, cfg.ct_limit, cfg.a_cap), dtype=float)
    ratio_strip = ct_local / np.maximum(ct_max, 1.0e-30)
    converged = bool(residual < cfg.convergence_tol)
    if bollard:
        n_above = int(r.size)
        validity_flag = "bollard_unsupported"
        ratio = float("inf")
    else:
        n_above = int(np.sum(ratio_strip > 1.0))
        ratio = float(np.max(ratio_strip))
        validity_flag = "ok" if n_above == 0 else "thrust_closure_exceeded"
    model_valid = bool(not bollard and n_above == 0 and converged)

    d = geometry.diameter
    kt = thrust / (cfg.rho * n ** 2 * d ** 4)
    kq = torque / (cfg.rho * n ** 2 * d ** 5)
    jv = va / (n * d)
    eta = (jv * kt / (2.0 * math.pi * kq)) if (kq > 0.0 and kt > 0.0) else 0.0
    induced = float(np.sum(a * va * chord * dr) / max(np.sum(chord * dr), 1e-12))
    return OpenWaterState(
        advance_ratio=jv,
        thrust_coefficient=kt,
        torque_coefficient=kq,
        efficiency=eta,
        thrust=thrust,
        torque=torque,
        induced_mean=induced,
        residual=residual,
        iterations=iterations,
        r=r,
        a=a,
        a_prime=a_prime,
        alpha_deg=np.degrees(alpha),
        cl=cl,
        cd=cd,
        thrust_momentum=thrust_momentum,
        momentum_residual=momentum_residual,
        model_valid=model_valid,
        validity_flag=validity_flag,
        n_strips_above_ct_limit=n_above,
        ct_max_loading=ratio,
        ct_local=ct_local,
        induction_model=cfg.induction_model,
        converged=converged,
    )
