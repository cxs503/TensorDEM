"""Mindlin--Deresiewicz partial-slip tangential contact law (1953).

Physical model
--------------
Two elastic bodies pressed together with a normal force ``F_n`` and loaded
tangentially.  Mindlin (1949) showed that the no-slip tangential response is a
spring of stiffness ``k_t = 8 G* a`` (``a`` = Hertz contact radius, ``G*`` the
reduced shear modulus).  Mindlin & Deresiewicz (1953) extended this to the
*frictional* case: because the tangential traction cannot exceed ``mu p``
anywhere on the contact patch, a **slip annulus** forms at the contact rim from
the very first tangential increment, while an inner **stick circle** of radius
``c < a`` carries the load.  The resulting force--displacement relation is

.. math::

    F_t = \\mu F_n \\left[ 1 - \\left(1 - \\frac{\\delta}{\\delta_s}\\right)^{3/2} \\right],
    \\qquad \\delta_s = \\frac{3 \\mu F_n}{2 k_t}, \\qquad
    \\frac{c}{a} = \\left(1 - \\frac{F_t}{\\mu F_n}\\right)^{1/3}

This is the **micro-slip -> gross-slip transition**:

* ``delta << delta_s``: ``F_t -> k_t delta`` (the no-slip Mindlin spring),
  i.e. the curve starts on the linear slope 8 G* a.
* ``delta -> delta_s``: ``F_t -> mu F_n`` (gross slip, friction saturated).

A linear Cundall--Strack spring with a Coulomb clamp uses a *single* stiffness
up to the clamp and therefore misses the whole micro-slip curvature: at
``delta = 0.5 delta_s`` the two laws differ by 18.5 % of ``mu F_n``
(``1 - 2^{-1.5} = 0.6464`` vs ``0.5``).  That gap is what the validation case
``CASE-DEM-MINDLIN-TANGENTIAL`` measures.

Unloading / reloading from gross slip (Mindlin & Deresiewicz 1953, the
"steady-state" hysteresis cycle) follows a shifted branch

.. math::

    F_t = F^{*} - 2\\mu F_n \\left[1 - \\left(1 - \\frac{\\Delta\\delta}{2\\delta_s}\\right)^{3/2}\\right]

with ``Delta delta`` the excursion measured *back* from the reversal point.
This branch has the same initial slope ``k_t`` (``dF/dDelta = 3 mu F_n / 2
delta_s = k_t``) and lands on ``-mu F_n`` after a full reversal
``Delta delta = 2 delta_s``.

Geometry conventions
--------------------
*  ``e_star``, ``g_star``, ``r_eff`` follow
   :mod:`tensordem.hertz_mindlin` (``1/E* = sum (1-nu_i^2)/E_i``,
   ``1/G* = sum (2-nu_i)/G_i``, ``1/R* = sum 1/R_i``), so the same numbers can
   be used for sphere--sphere and (with ``rigid_counterpart=True``)
   sphere--plane contacts.
*  ``k_t`` in this module is always the **no-slip tangential stiffness**
   ``8 G* a``, evaluated at the *current* normal force
   (``a = (3 F_n R* / 4 E*)^{1/3}``), so the law is a function of the normal
   load alone and does not need the normal overlap.
*  All functions are pure (no hidden state) and accept scalars or tensors of
  any broadcastable shape.

Scope and stated limits
-----------------------
*  Monotonic loading from a virgin contact and reversals *from gross slip* are
   closed-form here.  A general multi-reversal path (arbitrary partial-slip
   turn-around inside the envelope) needs a state machine with a moving
   kinematic anchor; the opt-in DEM branch in :mod:`tensordem.dem3d` therefore
   documents ``tangential_model="mindlin_partial_slip"`` as monotonic-backbone
   accurate, with the reversal handling left to
   :func:`tensordem.contact_tangential.mindlin_tangential_step`.
*  The law is quasi-static and rate independent; no tangential dashpot is
   included here (the solver adds its own damping terms).

References
----------
Mindlin, R. D. & Deresiewicz, H. (1953). "Elastic Spheres in Contact Under
Varying Oblique Forces." *Journal of Applied Mechanics* 20, 327-344.
DOI: 10.1115/1.4010702.

Johnson, K. L. (1985). *Contact Mechanics*, Cambridge University Press,
Sec. 4.3-4.4 (secondary source used for the closed forms; the 1953 original is
paywalled and was not transcribed -- no table values are copied here).

Mindlin, R. D. (1949). "Compliance of Elastic Bodies in Contact." *Journal of
Applied Mechanics* 16, 259-268.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "hertz_contact_radius",
    "mindlin_no_slip_stiffness",
    "gross_slip_displacement",
    "partial_slip_magnitude",
    "partial_slip_force",
    "stick_radius_ratio",
    "inverse_displacement",
    "reversal_branch",
    "reversal_loop_area",
    "steady_state_cycle_area",
]


def _t(value) -> torch.Tensor:
    """Promote a scalar or tensor to a float64 tensor (dtype preserved for tensors)."""
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise TypeError("arguments must be floating point")
        return value
    return torch.tensor(float(value), dtype=torch.float64)


def _check_mu(mu) -> torch.Tensor:
    mu_t = _t(mu)
    if bool(torch.any(mu_t < 0.0)) or not bool(torch.all(torch.isfinite(mu_t))):
        raise ValueError("friction coefficient mu must be finite and nonnegative")
    return mu_t


def hertz_contact_radius(normal_force, e_star, r_eff) -> torch.Tensor:
    """Hertz contact radius ``a = (3 F_n R* / (4 E*))^{1/3}``.

    Args:
        normal_force: Contact normal force ``F_n`` [N] (>= 0).
        e_star: Reduced Young's modulus ``E*`` [Pa].
        r_eff: Effective radius ``R*`` [m].

    Returns:
        Contact radius ``a`` [m], same shape as the broadcast inputs.
    """
    f_n = _t(normal_force).clamp_min(0.0)
    e = float(e_star)
    r = float(r_eff)
    if not (math.isfinite(e) and e > 0.0):
        raise ValueError("e_star must be finite and positive")
    if not (math.isfinite(r) and r > 0.0):
        raise ValueError("r_eff must be finite and positive")
    return (3.0 * f_n * r / (4.0 * e)).clamp_min(0.0).pow(1.0 / 3.0)


def mindlin_no_slip_stiffness(normal_force, e_star, g_star, r_eff) -> torch.Tensor:
    """No-slip tangential stiffness ``k_t = 8 G* a`` at the given normal force.

    This is the *initial slope* of the partial-slip curve; it is identical to
    :func:`tensordem.hertz_mindlin.hertz_mindlin_tangential_stiffness` evaluated
    through the overlap (both reduce to ``8 G* sqrt(R* delta)``).
    """
    a = hertz_contact_radius(normal_force, e_star, r_eff)
    g = float(g_star)
    if not (math.isfinite(g) and g > 0.0):
        raise ValueError("g_star must be finite and positive")
    return 8.0 * g * a


def gross_slip_displacement(normal_force, mu, k_t) -> torch.Tensor:
    """Tangential displacement at which gross slip starts, ``delta_s``.

    ``delta_s = 3 mu F_n / (2 k_t)``.  Equivalently, with the Hertz/Mindlin
    constants substituted, ``delta_s = 3 mu F_n (1-nu_1)/(... )`` in the
    classical form ``3 mu F_n (2 - nu) / (8 G a)`` for two equal spheres; the
    ``3 mu F_n / (2 k_t)`` form is used here because it holds for every
    combination that defines ``k_t = 8 G* a``.
    """
    f_n = _t(normal_force).clamp_min(0.0)
    mu_t = _check_mu(mu)
    k = _t(k_t)
    if bool(torch.any(k < 0.0)):
        raise ValueError("k_t must be nonnegative")
    tiny = torch.finfo(k.dtype if k.is_floating_point() else torch.float64).tiny
    # k_t = 8 G* a vanishes only on an unloaded contact (a = 0): there is no
    # slip displacement to define either, and the force is zero as well.
    return 3.0 * mu_t * f_n / (2.0 * k.clamp_min(tiny))


def partial_slip_magnitude(delta_mag, normal_force, mu, k_t) -> torch.Tensor:
    """Magnitude of the Mindlin--Deresiewicz tangential force.

    ``F_t = mu F_n [1 - (1 - delta/delta_s)^{3/2}]`` for ``delta <= delta_s``,
    and ``mu F_n`` beyond (gross slip).

    Args:
        delta_mag: Magnitude of the tangential contact displacement [m].
        normal_force: Contact normal force ``F_n`` [N].
        mu: Coulomb friction coefficient.
        k_t: No-slip tangential stiffness ``8 G* a`` [N/m].

    Returns:
        ``|F_t|`` [N].
    """
    delta = _t(delta_mag).clamp_min(0.0)
    f_n = _t(normal_force).clamp_min(0.0)
    mu_t = _check_mu(mu)
    d_s = gross_slip_displacement(f_n, mu_t, k_t)
    cap = mu_t * f_n
    # Virgin contact has no adhesion: no force without normal load.
    ratio = torch.where(d_s > 0.0, delta / d_s.clamp_min(torch.finfo(delta.dtype).tiny),
                        torch.ones_like(delta))
    ratio = ratio.clamp(0.0, 1.0)
    return cap * (1.0 - (1.0 - ratio).clamp_min(0.0).pow(1.5))


def partial_slip_force(tang_disp, normal_force, mu, k_t):
    """Vector Mindlin--Deresiewicz force, aligned with the displacement.

    The force is returned with the sign of the (vector) tangential
    displacement, ``F_t = |F_t| * delta_hat``, which is the convention used by
    the particle--particle branch of :class:`tensordem.dem3d.IceDEM3D`
    (``F_t = k_t delta``); plane/platen branches apply the opposite sign
    themselves.

    Args:
        tang_disp: (..., 3) accumulated tangential contact displacement [m].
        normal_force: Contact normal force ``F_n`` [N] (broadcastable).
        mu: Coulomb friction coefficient.
        k_t: No-slip tangential stiffness ``8 G* a`` [N/m] (broadcastable).

    Returns:
        ``(force, slip_ratio)`` with ``force`` shaped like ``tang_disp`` and
        ``slip_ratio = |delta| / delta_s`` (>= 1 flags gross slip).
    """
    disp = _t(tang_disp)
    if disp.shape[-1] != 3:
        raise ValueError("tang_disp must have a trailing dimension of 3")
    delta_mag = torch.linalg.vector_norm(disp, dim=-1)
    f_mag = partial_slip_magnitude(delta_mag, normal_force, mu, k_t)
    d_s = gross_slip_displacement(normal_force, mu, k_t)
    tiny = torch.finfo(disp.dtype).tiny
    unit = disp / delta_mag.clamp_min(tiny).unsqueeze(-1)
    # delta = 0 -> no direction; force is exactly zero there, so any unit is fine.
    unit = torch.where((delta_mag > 0.0).unsqueeze(-1), unit, torch.zeros_like(disp))
    slip_ratio = torch.where(d_s > 0.0, delta_mag / d_s.clamp_min(tiny),
                             torch.zeros_like(delta_mag))
    return f_mag.unsqueeze(-1) * unit, slip_ratio


def stick_radius_ratio(f_t_mag, normal_force, mu) -> torch.Tensor:
    """Radius of the stick circle relative to the contact patch, ``c/a``.

    ``c/a = (1 - F_t/(mu F_n))^{1/3}``: 1 on a virgin contact (no slip),
    0 at gross slip.  This is the quantity a photoelastic or ultrasonic
    experiment would measure inside the contact patch.
    """
    f = _t(f_t_mag).clamp_min(0.0)
    f_n = _t(normal_force).clamp_min(0.0)
    mu_t = _check_mu(mu)
    cap = mu_t * f_n
    tiny = torch.finfo(f.dtype).tiny
    inside = torch.where(cap > 0.0, 1.0 - f / cap.clamp_min(tiny), torch.zeros_like(f))
    return inside.clamp(0.0, 1.0).pow(1.0 / 3.0)


def inverse_displacement(f_t_mag, normal_force, mu, k_t) -> torch.Tensor:
    """Inverse of :func:`partial_slip_magnitude`: ``delta(F_t)``.

    ``delta = delta_s [1 - (1 - F_t/(mu F_n))^{2/3}]``, clamped at ``delta_s``.
    """
    c_over_a = stick_radius_ratio(f_t_mag, normal_force, mu)
    d_s = gross_slip_displacement(normal_force, mu, k_t)
    return d_s * (1.0 - c_over_a.pow(2.0))


def reversal_branch(delta_excursion, peak_force_mag, normal_force, mu, k_t) -> torch.Tensor:
    """Unloading/reloading branch after a load reversal from ``peak_force_mag``.

    ``F_t = F* - 2 mu F_n [1 - (1 - Delta/(2 delta_s))^{3/2}]`` with ``Delta``
    the excursion measured back from the reversal point (>= 0).  This is the
    Mindlin--Deresiewicz steady-state cycle branch: it starts with the same
    slope ``k_t`` and reaches ``F* - 2 mu F_n`` at ``Delta = 2 delta_s``.

    Args:
        delta_excursion: ``Delta`` [m], measured from the reversal point.
        peak_force_mag: ``F*`` [N], force magnitude at the reversal point.
        normal_force: Contact normal force ``F_n`` [N].
        mu: Coulomb friction coefficient.
        k_t: No-slip tangential stiffness [N/m].

    Returns:
        Force magnitude [N] (signed along the *reverse* direction, i.e. it can
        go negative once the reversal is complete).
    """
    d = _t(delta_excursion).clamp_min(0.0)
    peak = _t(peak_force_mag)
    f_n = _t(normal_force).clamp_min(0.0)
    mu_t = _check_mu(mu)
    d_s = gross_slip_displacement(f_n, mu_t, k_t)
    cap = mu_t * f_n
    tiny = torch.finfo(d.dtype).tiny
    x = torch.where(d_s > 0.0, d / (2.0 * d_s).clamp_min(tiny), torch.zeros_like(d))
    x = x.clamp(0.0, 1.0)
    return peak - 2.0 * cap * (1.0 - (1.0 - x).pow(1.5))


def steady_state_cycle_area(normal_force, mu, k_t, n_quad: int = 4001) -> torch.Tensor:
    """Work absorbed per *closed* steady-state stick-slip cycle [J].

    The steady-state Mindlin--Deresiewicz cycle has both turning points on the
    Coulomb cap: the force runs ``+mu F_n -> -mu F_n`` along the reversal
    branch while the contact-point displacement runs back by ``2 delta_s``, and
    then returns.  The enclosed area is therefore the integral, over the shared
    displacement span ``Delta in [0, 2 delta_s]``, of the *forward* branch minus
    the *reverse* branch::

        W = int_0^{2 delta_s} [ F_fwd(Delta) - F_rev(Delta) ] d Delta
          = 0.8 mu F_n delta_s = 1.2 (mu F_n)^2 / k_t

    (the dimensionless coefficient follows from
    ``int_0^1 [2 g(u) + 2 g(1-u) - 2] du`` with ``g(u) = 1-(1-u)^{3/2}``:
    ``2*(0.6) + 2*(0.6) - 2 = 0.4``, times ``2 mu F_n`` times ``2 delta_s``
    ... see the quadrature below, which is the authoritative evaluation).

    The quadrature is a plain trapezoid rule on a fixed grid, so the value is
    deterministic and independent of any DEM time stepping.

    Note
    ----
    This is **not** what :func:`reversal_loop_area` used to return: that
    function integrated the *virgin loading backbone* against the reversal
    branch over ``Delta in [0, 2 delta_s]`` and doubled the result.  Those two
    curves do not form a closed loop (they are separated by ``2 mu F_n`` at
    ``Delta = 2 delta_s``), so the old value, ``6 (mu F_n)^2 / k_t``, is
    ``5x`` the dissipation of the steady-state cycle.  See
    ``docs/AUDIT_MINDLIN_ENGINE_VS_MD_m16810.md``.
    """
    f_n = _t(normal_force).clamp_min(0.0)
    mu_t = _check_mu(mu)
    d_s = gross_slip_displacement(f_n, mu_t, k_t)
    cap = mu_t * f_n
    if float(d_s) == 0.0:
        return torch.zeros_like(cap)
    span = 2.0 * float(d_s)
    grid = torch.linspace(0.0, span, n_quad, dtype=f_n.dtype)
    f_rev = reversal_branch(grid, cap, f_n, mu_t, k_t)      # +cap -> -cap
    # Forward branch over the same displacement span: the mirror image, i.e.
    # the same branch traversed with Delta measured back from the far end.
    f_fwd = -reversal_branch((span - grid).clamp_min(0.0), cap, f_n, mu_t, k_t)
    trap = getattr(torch, "trapezoid", None) or getattr(torch, "trapz")
    return trap(f_fwd - f_rev, grid)


def reversal_loop_area(normal_force, mu, k_t, n_quad: int = 4001) -> torch.Tensor:
    """Hysteretic energy dissipated per steady-state cycle [J].

    .. deprecated:: m16810
       Until m16810 this function returned ``6 (mu F_n)^2 / k_t``, obtained by
       integrating the *virgin loading backbone* against the reversal branch
       over ``Delta in [0, 2 delta_s]`` and doubling.  Those two curves do not
       close: at ``Delta = 2 delta_s`` the backbone sits at ``+mu F_n`` while
       the reversal branch sits at ``-mu F_n``, so the integral is an *open*
       two-branch quantity, not a cycle.  It overestimated the closed
       steady-state loop by a factor of five (400 %) and the merge-time
       "absolute closed form" check only re-derived that same definition, so
       it could not see the error.

    The function now delegates to :func:`steady_state_cycle_area`; the returned
    number changed from ``6 (mu F_n)^2 / k_t`` to ``1.2 (mu F_n)^2 / k_t``.
    """
    return steady_state_cycle_area(normal_force, mu, k_t, n_quad=n_quad)