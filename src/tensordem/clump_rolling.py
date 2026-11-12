r"""Rolling / drilling resistance for multi-sphere (clump) contacts, plus the
tilt-table calibration of the rolling-friction coefficient ``mu_r``.

Why this module exists
----------------------
:mod:`tensordem.rolling_resistance` gives a sphere-on-plane rolling law and
:mod:`tensordem.clump_dynamics` gives member-summed contacts for rigid
multi-sphere ("clump") particles, but **no rolling or drilling resistance was
available on the clump contact path** (``docs/clump_multisphere_particles.md``
listed both as ``NOT_SUPPORTED``).  A clump that cannot resist a rolling moment
rolls far too freely, which biases exactly the macroscopic quantities an ice
rubble / broken-channel calculation reports (angle of repose of a rubble field,
transport resistance of broken ice).

This module supplies

* the per-member resistance law :func:`resistance_moment` (Cundall-type ``cdt``,
  viscous, and elastic-plastic ``epsd`` branches; model families follow Ai et
  al. 2011, *Powder Technology* 206:269-282, the elastic-plastic branch after
  Iwashita & Oda 1998 / Jiang et al. 2005),
* the closed-form tilt thresholds used as verification references,
  :func:`critical_tilt_angle_deg` and :func:`calibration_regime`,
* the inverse (calibration) relation :func:`invert_rolling_friction`,
* the tilt-table experiment harness :func:`tilt_test` /
  :func:`bracket_critical_tilt`.

Statics used as the reference
-----------------------------
Take a clump of equal member radius :math:`r` resting flat on a plane tilted by
:math:`\theta`: every load-bearing member has its centre at height :math:`r`
above the plane, so the centre of mass sits at height :math:`h` (``h = r`` for a
flat pose).  Quasi-static force balance fixes the totals
:math:`\sum_i N_i = m g \cos\theta` and
:math:`\sum_i f_i = m g \sin\theta`.  Gravity exerts no moment about the centre
of mass -- the contact resultants do.  With :math:`x_i` the in-plane (downslope)
coordinate of member :math:`i` relative to the centre of mass,

.. math::

   \sum_i (h f_i - x_i N_i) + \sum_i M_i = 0
   \quad\Longrightarrow\quad
   h\,m g\sin\theta = \sum_i x_i N_i + \sum_i M_i .

Two independent mechanisms can supply the right-hand side: a shift of the normal
resultant (centre of pressure) inside the contact support, bounded by
:math:`|\sum_i x_i N_i| \le a\, m g \cos\theta` with :math:`a` the distance from
the centre of mass to the **downhill** support edge, and the rolling resistance
moments, bounded by :math:`|\sum_i M_i| \le \mu_r r\, m g \cos\theta`.  Static
rest also requires the Coulomb cone :math:`\tan\theta \le \mu_s`, hence

.. math::

   \tan\theta_c = \min\left(\mu_s,\; \frac{a + \mu_r r}{h}\right).

Two limits deserve emphasis.  With a single-point support (``a = 0``: a sphere,
or a clump resting on one member) this is *exactly* the verified single-sphere
law ``tan(theta_c) = mu_r``, independent of mass and inertia.  For an elongated
clump ``a/h`` is of order one and dominates, so the tilt test then measures the
support geometry (or sliding), **not** the rolling coefficient: ``mu_r`` is
identifiable from a tilt test only while

.. math::

   \mu_r < \frac{\mu_s h - a}{r},

i.e. while rolling resistance fails before the friction cone does
(:func:`rolling_friction_identifiable`).  Where that fails, only a dynamic test
(repose angle, rotating drum) can size ``mu_r`` -- quantified in
``docs/clump_rolling_resistance_calibration.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

__all__ = [
    "TiltTestResult",
    "bracket_critical_tilt",
    "calibration_regime",
    "clump_tilt_reference",
    "com_height",
    "couple_threshold_slope",
    "critical_tilt_angle_deg",
    "drilling_resistance_moment",
    "inertial_moment_about",
    "invert_rolling_friction",
    "resistance_moment",
    "rolling_component",
    "rolling_friction_identifiable",
    "rolling_threshold_slope",
    "support_offset",
    "tilt_test",
    "tipping_threshold_slope",
    "viscous_damping_gain",
    "viscous_stability_dt",
]

_ROLLING_MODELS = ("cdt", "viscous", "epsd")


def _unit(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError("cannot normalise a zero vector")
    return array / norm


def rolling_component(omega_rel: Sequence[float],
                      normal: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Split a relative angular velocity into rolling and drilling parts.

    ``omega_roll`` is the part perpendicular to the contact normal (true
    rolling), ``omega_spin`` the part along it (drilling / torsion).
    """
    omega = np.asarray(omega_rel, dtype=np.float64)
    n = _unit(normal)
    spin = float(omega @ n) * n
    return omega - spin, spin


# --------------------------------------------------------------------------
# Stability of the viscous (rate-proportional) branch
# --------------------------------------------------------------------------
def inertial_moment_about(inverse_inertia_world: Sequence[Sequence[float]],
                          axis: Sequence[float]) -> float:
    """Moment of inertia [kg m^2] of a rigid body about ``axis`` (body frame free).

    ``inverse_inertia_world`` is the world-frame inverse inertia tensor, so
    ``I_axis = 1 / (a^T I_world^-1 a)`` for a unit ``a``.  Needed because the
    viscous rolling law is a *rate* law: its stability limit is set by the
    inertia about the rolling axis, not by the mass.
    """
    a = _unit(axis)
    inverse = np.asarray(inverse_inertia_world, dtype=np.float64)
    if inverse.shape != (3, 3):
        raise ValueError("inverse_inertia_world must be 3x3")
    projected = float(a @ inverse @ a)
    if not math.isfinite(projected) or projected <= 0.0:
        raise ValueError("inverse inertia projected on the axis must be positive")
    return 1.0 / projected


def viscous_damping_gain(gamma: float, dt: float, inertia: float | None = None) -> float:
    r"""Effective damping gain [N m s] to apply over one step of length ``dt``.

    The viscous branch is the rate law :math:`M = -\gamma\,\omega_{roll}` with
    :math:`\gamma = \mu_r |F_n| r / \omega_{ref}`, i.e. the rolling rate obeys
    :math:`\dot\omega = -\lambda\,\omega` with :math:`\lambda = \gamma / I`.
    Applied *explicitly* it requires :math:`\lambda \Delta t \ll 1`; at the
    default :math:`\omega_{ref} = 10^{-3}` rad/s and :math:`\Delta t = 2\cdot10^{-4}` s
    the ratio is ~90 and the configuration diverges (honest FAIL recorded in
    ``docs/clump_rolling_resistance_calibration.md`` §3.4).

    With ``inertia`` given this returns the implicit-damping gain of the
    velocity-Verlet half-kick splitting used by
    :meth:`tensordem.clump_dynamics.ClumpBody.step` (two half kicks of
    :math:`h = \Delta t/2`, each damping the rate it starts from):

    .. math::

       g = \frac{\gamma}{1 + \gamma h / I},\qquad
       \frac{\omega_{half}}{\omega} = 1 - \frac{g h}{I} = \frac{1}{1 + \lambda h}.

    Three properties justify the choice, and they are what
    ``tests/test_clump_viscous_rolling_stability.py`` checks:

    * **Unconditional stability / no energy creation.** The one-half-step factor
      lies in :math:`(0, 1]` for any :math:`\Delta t`, so the branch is monotone
      dissipative and can never reverse the spin; :math:`g \le \gamma` for all
      steps.
    * **Exact steady state (DC gain).** Backward Euler preserves the fixed point
      of a constant forcing: with an external torque :math:`T`, the two half
      kicks settle at :math:`\omega^{*} = T/\gamma` *to round-off, for any*
      :math:`\Delta t`.  (A pure exponential-integrator gain
      :math:`\gamma(1-e^{-y})/y` reproduces the free decay better but shifts
      that fixed point by :math:`\lambda\Delta t/(1-e^{-\lambda\Delta t})`,
      +7.4 % at :math:`\lambda\Delta t = 0.29` -- rejected for that reason.)
    * **Consistency.** :math:`g \to \gamma` as :math:`\Delta t \to 0`, i.e. the
      declared law is recovered in the resolved limit; the transient error is
      first order, :math:`(1+\lambda h)^{-2n}` vs :math:`e^{-\lambda t}`
      (0.25 % per unit :math:`\lambda\Delta t`, quantified in
      ``docs/clump_rolling_viscous_stability.md``).

    Without ``inertia`` (legacy call signature) the declared gain is returned
    unchanged, so the explicit branch stays bit-identical.
    """
    gamma = float(gamma)
    if gamma <= 0.0:
        return 0.0
    if inertia is None:
        return gamma
    inertia = float(inertia)
    dt = float(dt)
    if not (math.isfinite(inertia) and inertia > 0.0):
        raise ValueError("inertia must be finite and positive")
    if not (math.isfinite(dt) and dt >= 0.0):
        raise ValueError("dt must be finite and non-negative")
    if dt == 0.0:
        return gamma
    return gamma / (1.0 + 0.5 * gamma * dt / inertia)


def viscous_stability_dt(inertia: float, normal_force: float, radius: float,
                         mu_r: float, omega_ref: float, *,
                         safety: float = 0.25) -> dict[str, float]:
    """Time-step limits of the **explicit** viscous rolling branch.

    The branch is integrated by the velocity-Verlet half-kick splitting of
    :meth:`tensordem.clump_dynamics.ClumpBody.step`, so each half step of length
    ``h = dt/2`` multiplies the rolling rate by ``1 - lambda*h`` and one full
    step by ``(1 - lambda*dt/2)**2``.  Hence:

    * ``dt_monotone = 2/lambda`` -- beyond this the one-step factor turns
      negative: the rate reverses sign every half step (overshoot), although the
      amplitude still decays towards ``-1``.
    * ``dt_divergence = 4/lambda`` -- beyond this ``|1 - lambda*dt/2| > 1`` and
      the run grows without bound; this is the blow-up reported as the honest
      FAIL in ``docs/clump_rolling_resistance_calibration.md`` §3.4.
    * ``dt_safe = safety * dt_monotone`` -- conservative usable step.

    All values in seconds; ``inf`` when ``mu_r == 0`` (no rate law).
    """
    for name, value in (("inertia", inertia), ("radius", radius),
                        ("omega_ref", omega_ref), ("safety", safety)):
        if not (math.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if mu_r < 0.0 or not math.isfinite(mu_r):
        raise ValueError("mu_r must be finite and non-negative")
    force = max(float(normal_force), 0.0)
    limit = mu_r * force * float(radius)
    if limit <= 0.0:
        return {"dt_monotone": math.inf, "dt_divergence": math.inf,
                "dt_safe": math.inf, "lambda": 0.0}
    lam = limit / (float(omega_ref) * float(inertia))
    return {"dt_monotone": 2.0 / lam, "dt_divergence": 4.0 / lam,
            "dt_safe": float(safety) * 2.0 / lam, "lambda": lam}


def resistance_moment(
    normal: Sequence[float],
    omega_rel: Sequence[float],
    normal_force: float,
    radius: float,
    *,
    mu_r: float = 0.0,
    mu_t: float = 0.0,
    model: str = "epsd",
    moment_state: Sequence[float] | None = None,
    dt: float = 0.0,
    k_r: float | None = None,
    normal_stiffness: float | None = None,
    stiffness_factor: float = 2.25,
    omega_ref: float = 1.0e-3,
    inertia: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Contact moment resisting rolling (``mu_r``) and drilling (``mu_t``).

    Parameters
    ----------
    normal, omega_rel
        Contact normal (direction from the plane/obstacle towards the clump
        member) and the *relative* angular velocity
        ``omega_member - omega_obstacle`` [rad/s].
    normal_force, radius
        Normal contact force [N] and the member radius [m] used as the moment
        arm (multi-sphere convention ``r_eff = r_member``).
    mu_r, mu_t
        Rolling and drilling friction coefficients [-].
    model
        ``"cdt"`` memoryless saturated moment ``mu_r |F_n| r`` (Cundall-type),
        ``"viscous"`` linear in ``omega_roll`` normalised by ``omega_ref``, or
        ``"epsd"`` elastic-plastic spring that can hold a static moment (the
        only branch with a genuine static tilt threshold).
    moment_state
        Previous rolling moment (world frame) for the ``epsd`` branch.
    dt
        Time step [s]; the ``epsd`` branch integrates the rolling spring over it.
    k_r
        Rolling stiffness [N m/rad]; defaults to
        ``stiffness_factor * mu_r * normal_stiffness * radius**2``.
    inertia
        Moment of inertia [kg m^2] of the body about the rolling axis.  When
        supplied, the ``"viscous"`` gain is ``dt``-limited through
        :func:`viscous_damping_gain`, which reproduces the exact exponential
        decay of the rate law for *any* step and therefore removes the
        conditional-stability failure at the default ``omega_ref``.  ``None``
        keeps the declared (explicit) gain, i.e. the legacy behaviour.

    Returns
    -------
    (moment, new_state)
        Total resistance moment about the member centre [N m] and the updated
        rolling-spring state to feed back on the next call.

    Notes
    -----
    ``cdt`` and ``viscous`` are pointwise dissipative, ``moment . omega_rel <= 0``.
    ``epsd`` is *elastic* up to the cap ``mu_r |F_n| r`` and dissipative only
    while the cap is active, so it can return stored energy when the rotation
    reverses -- the price of having a static branch at all.  Its stored energy is
    bounded by ``M_lim**2 / (2 k_r)`` per member, which is what
    ``tests/test_clump_rolling_resistance.py`` checks instead of pointwise
    dissipation.
    """
    if model not in _ROLLING_MODELS:
        raise ValueError(f"unknown rolling model {model!r}; expected one of {_ROLLING_MODELS}")
    if mu_r < 0.0 or mu_t < 0.0:
        raise ValueError("mu_r and mu_t must be non-negative")
    n = _unit(normal)
    omega = np.asarray(omega_rel, dtype=np.float64)
    force = max(float(normal_force), 0.0)
    arm = float(radius)
    moment = np.zeros(3)
    state = np.zeros(3) if moment_state is None else np.array(moment_state, dtype=np.float64)
    if force <= 0.0 or arm <= 0.0:
        return moment, np.zeros(3)

    omega_roll, omega_spin = rolling_component(omega, n)
    limit = mu_r * force * arm
    if mu_r > 0.0 and limit > 0.0:
        if model == "cdt":
            magnitude = float(np.linalg.norm(omega_roll))
            if magnitude > 0.0:
                moment -= limit * omega_roll / magnitude
            state = np.zeros(3)
        elif model == "viscous":
            gain = viscous_damping_gain(limit / max(float(omega_ref), 1.0e-12),
                                        dt, inertia)
            moment -= gain * omega_roll
            state = moment.copy()
        else:  # epsd
            stiffness = k_r
            if stiffness is None:
                if normal_stiffness is None:
                    raise ValueError("epsd model needs k_r or normal_stiffness")
                stiffness = stiffness_factor * mu_r * float(normal_stiffness) * arm ** 2
            trial = state - stiffness * omega_roll * dt
            magnitude = float(np.linalg.norm(trial))
            if magnitude > limit:                        # plasticity: clamp at the cap
                trial = trial * (limit / magnitude)
            state = trial
            moment += trial
    else:
        state = np.zeros(3)

    if mu_t > 0.0:
        spin_limit = mu_t * force * arm
        magnitude = float(np.linalg.norm(omega_spin))
        if magnitude > 0.0:
            moment -= spin_limit * omega_spin / magnitude

    return moment, state


def drilling_resistance_moment(normal_force: float, radius: float, mu_t: float) -> float:
    """Saturated drilling (torsion about the normal) moment ``mu_t |F_n| r``."""
    return max(float(normal_force), 0.0) * float(radius) * max(float(mu_t), 0.0)


# --------------------------------------------------------------------------
# Closed-form tilt thresholds
# --------------------------------------------------------------------------
def rolling_threshold_slope(mu_r: float, member_radius: float, com_height: float) -> float:
    """``tan(theta_c)`` of the rolling-only branch: ``mu_r r / h``."""
    if mu_r < 0.0:
        raise ValueError("mu_r must be non-negative")
    if member_radius <= 0.0 or com_height <= 0.0:
        raise ValueError("member_radius and com_height must be positive")
    return mu_r * float(member_radius) / float(com_height)


def tipping_threshold_slope(offset: float, com_height: float) -> float:
    """``tan(theta_c)`` of the contact-couple capacity: ``a / h``."""
    if offset < 0.0:
        raise ValueError("offset must be non-negative")
    if com_height <= 0.0:
        raise ValueError("com_height must be positive")
    return float(offset) / float(com_height)


def couple_threshold_slope(offset: float, com_height: float) -> float:
    """Alias of :func:`tipping_threshold_slope` (normal-resultant shift)."""
    return tipping_threshold_slope(offset, com_height)


def critical_tilt_angle_deg(
    mu_r: float,
    member_radius: float,
    com_height: float,
    *,
    mu_s: float = math.inf,
    downhill_offset: float = 0.0,
) -> float:
    """First-motion tilt angle [deg] of a clump resting flat on a plane.

    ``theta_c = arctan(min(mu_s, (a + mu_r r) / h))``: sliding gives way either to
    a combination of contact-couple (``a``) and rolling resistance
    (``mu_r r``).  ``downhill_offset=0`` is the single-point support (sphere or
    clump on one member), for which this returns the verified ``arctan(mu_r)``.
    """
    slope = (tipping_threshold_slope(downhill_offset, com_height)
             + rolling_threshold_slope(mu_r, member_radius, com_height))
    if math.isfinite(mu_s):
        if mu_s < 0.0:
            raise ValueError("mu_s must be non-negative")
        slope = min(slope, float(mu_s))
    return math.degrees(math.atan(slope))


def rolling_friction_identifiable(
    member_radius: float,
    com_height: float,
    *,
    mu_s: float,
    downhill_offset: float = 0.0,
) -> bool:
    """Whether a tilt test can resolve ``mu_r`` at all for this shape.

    Rolling resistance must be the mechanism that fails *before* Coulomb
    sliding, i.e. the sliding branch must sit above the rolling+couple branch:
    ``mu_s h > a + (unknown) mu_r r``.  This function tests the weaker condition
    ``mu_s h > a`` (some positive ``mu_r`` is still resolvable); when it is
    false the measured angle is set by sliding alone and carries no information
    on ``mu_r``.
    """
    if member_radius <= 0.0 or com_height <= 0.0:
        raise ValueError("member_radius and com_height must be positive")
    if not math.isfinite(mu_s) or mu_s < 0.0:
        raise ValueError("mu_s must be finite and non-negative")
    return float(mu_s) * float(com_height) - float(downhill_offset) > 0.0


def calibration_regime(
    angle_deg: float,
    mu_r: float,
    member_radius: float,
    com_height: float,
    *,
    mu_s: float = math.inf,
    downhill_offset: float = 0.0,
) -> str:
    """Which mechanism sets the measured tilt angle.

    ``"sliding"`` when the Coulomb cone binds first, otherwise
    ``"rolling+couple"`` (support geometry and rolling resistance together);
    only the latter carries information on ``mu_r``.
    """
    slope = (tipping_threshold_slope(downhill_offset, com_height)
             + rolling_threshold_slope(mu_r, member_radius, com_height))
    if math.isfinite(mu_s) and float(mu_s) <= slope:
        return "sliding"
    return "rolling+couple"


def invert_rolling_friction(angle_deg: float, member_radius: float,
                            com_height: float, *, downhill_offset: float = 0.0) -> float:
    """Calibrate ``mu_r`` from a measured tilt angle.

    Inverts the rolling+couple branch: ``mu_r = (h tan(theta) - a) / r``.  Clamped
    at zero -- a negative value means the measured angle is explained by the
    support geometry alone, so the experiment only yields an upper bound (never
    feed a negative coefficient to the solver).
    """
    if member_radius <= 0.0 or com_height <= 0.0:
        raise ValueError("member_radius and com_height must be positive")
    if downhill_offset < 0.0:
        raise ValueError("downhill_offset must be non-negative")
    value = (float(com_height) * math.tan(math.radians(angle_deg))
             - float(downhill_offset)) / float(member_radius)
    return max(value, 0.0)


# --------------------------------------------------------------------------
# Clump geometry references
# --------------------------------------------------------------------------
def com_height(clump) -> float:
    """Height of a clump's centre of mass above the plane in its flat pose [m].

    With the clump oriented so its member plane is parallel to the supporting
    plane, the distance from the centre of mass down to the lowest member
    surface is ``max_i (r_i - z_i)`` in the body frame (``z`` normal to the
    member plane).
    """
    local = np.asarray(clump.local_centers, dtype=np.float64)
    radii = np.asarray(clump.radii, dtype=np.float64)
    return float(np.max(radii - local[:, 2]))


def support_offset(clump) -> float:
    """In-plane distance from the centre of mass to the downhill support edge [m].

    Downhill is body ``+x`` in the pose used by :func:`tilt_test`; the clump can
    shift its normal resultant at most to the downhill-most member, at
    ``max_i x_i``.
    """
    local = np.asarray(clump.local_centers, dtype=np.float64)
    return float(np.max(local[:, 0]))


def clump_tilt_reference(clump, mu_r: float, *, mu_s: float = math.inf) -> dict[str, object]:
    """All analytic branches plus the resulting first-motion angle."""
    h = com_height(clump)
    r_eff = float(np.max(clump.radii))
    offset = support_offset(clump)
    slope_roll = rolling_threshold_slope(mu_r, r_eff, h)
    slope_couple = couple_threshold_slope(offset, h)
    reference: dict[str, object] = {
        "members": int(len(clump.radii)),
        "mass_kg": float(clump.mass),
        "com_height_m": h,
        "member_radius_m": r_eff,
        "downhill_offset_m": offset,
        "slope_rolling": slope_roll,
        "slope_couple": slope_couple,
        "slope_rolling_plus_couple": slope_roll + slope_couple,
        "theta_rolling_only_deg": math.degrees(math.atan(slope_roll)),
        "theta_c_deg": critical_tilt_angle_deg(
            mu_r, r_eff, h, mu_s=mu_s, downhill_offset=offset),
        "rolling_identifiable": rolling_friction_identifiable(
            r_eff, h, mu_s=mu_s, downhill_offset=offset) if math.isfinite(mu_s) else True,
    }
    if math.isfinite(mu_s):
        reference["slope_sliding"] = float(mu_s)
        reference["theta_sliding_deg"] = math.degrees(math.atan(float(mu_s)))
    reference["active_branch"] = calibration_regime(
        float(reference["theta_c_deg"]), mu_r, r_eff, h, mu_s=mu_s, downhill_offset=offset)
    reference["mu_r_from_theta_c"] = invert_rolling_friction(
        float(reference["theta_c_deg"]), r_eff, h, downhill_offset=offset)
    return reference


# --------------------------------------------------------------------------
# Tilt-table experiment harness
# --------------------------------------------------------------------------
@dataclass
class TiltTestResult:
    """Outcome of one tilt-table run (SI units)."""

    angle_deg: float
    model: str
    downhill_displacement: float
    late_drift: float
    final_speed: float
    rotation_deg: float
    dissipated_moment_work: float
    driving_moment_work: float
    moment_power_max: float
    speed_tolerance: float
    drift_tolerance: float

    @property
    def moved(self) -> bool:
        """Rest criterion: no net drift in the settled part of the run.

        ``late_drift`` is used on its own: the tangential/rolling springs are
        elastic, so a resting clump still rings with a bounded, zero-mean
        residual speed (quantified in the calibration report) while net drift
        separates resting from moving by 2-3 orders of magnitude.
        """
        return bool(self.late_drift > self.drift_tolerance)

    def as_dict(self) -> dict[str, object]:
        return {
            "angle_deg": self.angle_deg,
            "model": self.model,
            "downhill_displacement_m": self.downhill_displacement,
            "late_drift_m": self.late_drift,
            "final_speed_m_s": self.final_speed,
            "rotation_deg": self.rotation_deg,
            "dissipated_moment_work_J": self.dissipated_moment_work,
            "driving_moment_work_J": self.driving_moment_work,
            "moment_power_max_W": self.moment_power_max,
            "speed_tolerance_m_s": self.speed_tolerance,
            "drift_tolerance_m": self.drift_tolerance,
            "moved": self.moved,
        }


def _rotation_matrix_from_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    from .clump_dynamics import quaternion_to_matrix
    return quaternion_to_matrix(quaternion)


def tilt_test(
    clump,
    angle_deg: float,
    *,
    mu_r: float,
    mu_s: float = 0.5,
    mu_t: float = 0.0,
    model: str = "epsd",
    normal_stiffness: float = 1.0e5,
    restitution: float = 0.2,
    omega_ref: float = 1.0e-3,
    gravity: float = 9.81,
    duration: float = 0.30,
    dt: float = 1.0e-4,
    settle_fraction: float = 0.6,
    speed_tolerance: float = 2.0e-4,
    drift_tolerance: float | None = None,
) -> TiltTestResult:
    """Rest a clump on a tilted plane and measure whether it stays put.

    The plane is ``(x - point) . n >= 0`` with ``n = (sin t, 0, cos t)``; the clump
    is oriented so that its member plane is parallel to the plane (all
    load-bearing members touch simultaneously) and placed at centre-of-mass
    height ``h`` above it.  Coulomb friction ``mu_s`` acts on the tangential
    spring, the rolling/drilling resistance of :func:`resistance_moment` acts
    per member.

    ``late_drift`` is the downhill centre-of-mass displacement between
    ``settle_fraction`` of the run and its end; together with the residual speed
    it forms the rest criterion (an elastic-plastic rolling spring settles to a
    static pose, so drift -> 0, while a sliding/rolling clump keeps moving).
    """
    # local import keeps the dependency one-way: clump_dynamics imports this module
    from .clump_dynamics import (ClumpBody, ContactConfig, PlaneContact,
                                 quaternion_from_axis_angle)

    theta = math.radians(angle_deg)
    normal = np.array([math.sin(theta), 0.0, math.cos(theta)])
    downslope = np.array([math.cos(theta), 0.0, -math.sin(theta)])
    config = ContactConfig(normal_stiffness=normal_stiffness, restitution=restitution,
                           friction=mu_s, rolling_friction=mu_r,
                           drilling_friction=mu_t, rolling_model=model,
                           rolling_omega_ref=omega_ref)
    plane = PlaneContact(normal, point=(0.0, 0.0, 0.0), config=config)

    h = com_height(clump)
    radii = np.asarray(clump.radii, dtype=np.float64)
    local = np.asarray(clump.local_centers, dtype=np.float64)
    load_bearing = int(np.count_nonzero(np.isclose(radii - local[:, 2], h, atol=1e-12)))
    # start in static contact instead of dropping the clump onto the plane
    penetration = clump.mass * gravity * math.cos(theta) / (
        max(load_bearing, 1) * normal_stiffness)
    position = normal * (h - penetration)
    orientation = quaternion_from_axis_angle((0, 1, 0), theta)
    body = ClumpBody(clump, position=position, orientation=orientation,
                     gravity=(0.0, 0.0, -gravity))

    steps = max(int(round(duration / dt)), 2)
    settle_step = int(round(settle_fraction * steps))
    record_every = max(1, steps // 400)
    history = body.run(steps, dt, (plane,), record_every=record_every)
    positions = np.asarray(history["position"], dtype=np.float64)
    recorded_steps = np.rint(np.asarray(history["time"], dtype=np.float64) / dt).astype(int)
    downhill = (positions - positions[0]) @ downslope
    late_mask = recorded_steps >= settle_step
    if np.count_nonzero(late_mask) >= 2:
        late = downhill[late_mask]
        late_drift = float(abs(late[-1] - late[0]))
    else:                                    # very short runs: last two samples
        late_drift = float(abs(downhill[-1] - downhill[-2]))

    rotation = body.rotation_matrix()
    # rotation angle of the body relative to its initial (tilted) pose
    delta = rotation @ _rotation_matrix_from_quaternion(orientation).T
    trace = float(np.trace(delta))
    rotation_deg = math.degrees(math.acos(max(min(0.5 * (trace - 1.0), 1.0), -1.0)))

    drift_tol = 0.02 * float(np.max(radii)) if drift_tolerance is None else drift_tolerance
    return TiltTestResult(
        angle_deg=float(angle_deg),
        model=model,
        downhill_displacement=float(downhill[-1]),
        late_drift=late_drift,
        final_speed=float(np.linalg.norm(body.velocity)),
        rotation_deg=rotation_deg,
        dissipated_moment_work=float(plane.moment_work()),
        driving_moment_work=float(clump.mass * gravity * math.sin(theta) * downhill[-1]),
        moment_power_max=float(plane.moment_power_max()),
        speed_tolerance=speed_tolerance,
        drift_tolerance=drift_tol,
    )


def bracket_critical_tilt(
    clump,
    *,
    mu_r: float,
    mu_s: float = 0.5,
    mu_t: float = 0.0,
    model: str = "epsd",
    low: float = 0.0,
    high: float = 60.0,
    iterations: int = 12,
    tolerance_deg: float = 0.02,
    **kwargs,
) -> dict[str, object]:
    """Bisect the first-motion angle of a clump on a tilted plane.

    ``tilt_test`` is monotone in the tilt angle (the clump either stays or it
    does not), so a bisection on ``moved`` converges to the measured threshold.
    Returns the bracket, the midpoint estimate, the run count and every sample
    run (for the report tables).
    """
    runs = 0
    samples: list[dict[str, object]] = []

    def moved(angle: float) -> bool:
        nonlocal runs
        result = tilt_test(clump, angle, mu_r=mu_r, mu_s=mu_s, mu_t=mu_t, model=model,
                           **kwargs)
        runs += 1
        samples.append(result.as_dict())
        return result.moved

    while (high - low) > tolerance_deg and iterations > 0:
        mid = 0.5 * (low + high)
        if moved(mid):
            high = mid
        else:
            low = mid
        iterations -= 1
    return {
        "model": model,
        "mu_r": mu_r,
        "mu_s": mu_s,
        "theta_c_deg": 0.5 * (low + high),
        "bracket_deg": [low, high],
        "bracket_width_deg": high - low,
        "runs": runs,
        "samples": samples,
    }