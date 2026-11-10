"""Contact friction laws of the P152W1-N04 friction-contact verification surface.

Two laws are implemented *literally as declared* and then adjudicated against
closed-form anchors, before anything macroscopic is simulated:

tangential (Coulomb + viscous branch)
-------------------------------------
    F_t = min(mu * F_n, k_t * |delta_t|) * (-delta_hat_t)  -  c_t * v_t

The declared two-term law is ``min(mu F_n, k_t delta_t) + c_t v_t`` with both
terms resisting the relative motion; written vectorially that is
``-min(mu F_n, k_t |delta_t|) delta_hat_t - c_t v_t`` (the dashpot must enter
with a minus sign, otherwise it would *feed* the sliding motion instead of
damping it - verified by the unit test on the dissipated power).

``delta_t`` is the accumulated tangential (surface) displacement, ``v_t`` the
current tangential relative velocity at the contact point.  The spring branch is
the Cundall-Strack incremental law; the ``c_t * v_t`` term is a tangential
dashpot, i.e. the microscopic source of the *dynamic* friction of a rate
dependent friction model.

Literal (uncapped) addition of a dashpot to a Coulomb slider is not the same as
Coulomb friction: above onset the dashpot balances gravity at a *terminal
velocity*, so the macroscopic friction becomes rate dependent.  The total force
can also leave the friction cone at high sliding speed.
The kernel therefore offers ``cap_total`` (default ``True``), which re-projects
  F_t  ->  F_t * min(1, mu*F_n / |F_t|)
so that the *dynamic* friction stays at ``mu*F_n`` (textbook Coulomb), while the
dashpot only regularises micro-slip.  Both variants are kept, and
``onset_angle_shift_deg`` quantifies the difference, because the declared
formula (uncapped) measurably over-stabilises a free surface.

rolling resistance
------------------
    M_r = min(mu_r * R_eff * F_n, k_r * |theta_r|) * (-theta_hat_r) + c_r*(omega_rel)
which caps the rolling couple the way the tangential law caps the tangential
force.  The classical single-sphere anchor is that a sphere on an incline starts
to roll down when ``tan(theta) = mu_r``.

normal
------
    F_n = max(k_n * overlap - c_n * v_n, 0),
    c_n = 2 zeta sqrt(k_n m_eff),  zeta = |ln e| / sqrt(pi^2 + ln^2 e)

All quantities SI (m, kg, s, N, rad).  Angles are returned in degrees by the
public ``*_deg`` helpers.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "damping_ratio",
    "normal_force",
    "tangential_force",
    "rolling_torque",
    "sliding_onset_angle_deg",
    "rolling_onset_angle_deg",
    "simulate_slider_onset_deg",
    "simulate_rolling_onset_deg",
    "slider_acceleration",
    "tangential_damping_effect",
]


def damping_ratio(restitution: float) -> float:
    """Viscous damping ratio from the coefficient of restitution (linear dashpot)."""
    e = min(max(restitution, 1e-6), 1.0 - 1e-12)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2)


def normal_force(overlap: np.ndarray, v_n: np.ndarray, kn: float, cn: float) -> np.ndarray:
    """``F_n = max(k_n*overlap - c_n*v_n, 0)``; tension is not transmitted."""
    return np.maximum(kn * np.asarray(overlap, dtype=float)
                      - cn * np.asarray(v_n, dtype=float), 0.0)


def tangential_force(delta_t: np.ndarray, v_t: np.ndarray, f_n: np.ndarray,
                     *, kt: float, mu: float, ct: float = 0.0,
                     cap_total: bool = True) -> np.ndarray:
    """Coulomb tangential force ``min(mu F_n, k_t delta_t) + c_t v_t``.

    Parameters
    ----------
    delta_t : (M, 3) accumulated tangential displacement per contact.
    v_t : (M, 3) tangential relative velocity at the contact point.
    f_n : (M,) normal force magnitude (>= 0).
    kt, mu : tangential stiffness, Coulomb coefficient.
    ct : tangential dashpot coefficient [N s/m].  ``0`` recovers pure Coulomb.
    cap_total : re-project the *total* force onto the cone ``|F_t| <= mu F_n``.

    Returns
    -------
    (M, 3) tangential force.
    """
    delta_t = np.atleast_2d(np.asarray(delta_t, dtype=float))
    v_t = np.atleast_2d(np.asarray(v_t, dtype=float))
    f_n = np.atleast_1d(np.asarray(f_n, dtype=float))
    spring = -kt * delta_t
    mag = np.linalg.norm(spring, axis=1)
    cap = mu * f_n
    scale = np.where(mag > cap, cap / np.maximum(mag, 1e-300), 1.0)
    spring = spring * scale[:, None]
    # both terms resist: the spring opposes delta_t, the dashpot opposes v_t
    total = spring - ct * v_t
    if cap_total:
        tmag = np.linalg.norm(total, axis=1)
        scale_t = np.where(tmag > cap, cap / np.maximum(tmag, 1e-300), 1.0)
        total = total * scale_t[:, None]
    return total


def rolling_torque(theta_r: np.ndarray, omega_rel: np.ndarray, f_n: np.ndarray,
                   *, kr: float, mu_r: float, r_eff: float,
                   cr: float = 0.0, cap_total: bool = True) -> np.ndarray:
    """Rolling resistance couple ``min(mu_r R F_n, k_r theta_r)`` + optional damping."""
    theta_r = np.atleast_2d(np.asarray(theta_r, dtype=float))
    omega_rel = np.atleast_2d(np.asarray(omega_rel, dtype=float))
    f_n = np.atleast_1d(np.asarray(f_n, dtype=float))
    spring = -kr * theta_r
    mag = np.linalg.norm(spring, axis=1)
    cap = mu_r * r_eff * f_n
    scale = np.where(mag > cap, cap / np.maximum(mag, 1e-300), 1.0)
    spring = spring * scale[:, None]
    total = spring - cr * omega_rel
    if cap_total:
        tmag = np.linalg.norm(total, axis=1)
        scale_t = np.where(tmag > cap, cap / np.maximum(tmag, 1e-300), 1.0)
        total = total * scale_t[:, None]
    return total


def sliding_onset_angle_deg(mu: float) -> float:
    """Closed-form sliding-onset angle of an irrotational slider: ``arctan(mu)``."""
    return math.degrees(math.atan(mu))


def rolling_onset_angle_deg(mu_r: float) -> float:
    """Closed-form rolling-onset angle of a sphere on an incline: ``arctan(mu_r)``."""
    return math.degrees(math.atan(mu_r))



# ---------------------------------------------------------------------------
# Time-stepping anchors: the same law, integrated by itself
# ---------------------------------------------------------------------------

def slider_acceleration(mu: float, tilt_deg: float, *, mass: float = 1.0,
                        kn: float = 1.0e5, kt_ratio: float = 0.5,
                        ct: float = 0.0, cap_total: bool = True,
                        gravity: float = 9.81, t_end: float = 0.40,
                        t_transient: float = 0.10,
                        dt: float | None = None) -> float:
    """Steady down-slope acceleration of an irrotational slider (m/s^2).

    The particle translates but does not rotate, so the only resistance is the
    tangential law.  The acceleration is the curvature ``2 c2`` of a quadratic
    least-squares fit of the displacement history over
    ``t_transient < t < t_end``.  A displacement fit is used instead of a
    velocity slope because below the Coulomb threshold the tangential spring
    oscillates at ``sqrt(k_t/m)``: that oscillation is bounded in displacement
    (amplitude ``mg sin(theta)/k_t``) but appears as a large spurious velocity
    slope.  The fit window spans many oscillation periods.

    Closed form for ``ct = 0``: ``a = g (sin(theta) - mu cos(theta))`` above the
    Coulomb threshold and ``a = 0`` below it.
    """
    theta = math.radians(tilt_deg)
    kt = kt_ratio * kn
    if dt is None:
        dt = 2.0 * math.pi * math.sqrt(mass / kn) / 80.0
    steps = int(math.ceil(t_end / dt))
    i0 = int(math.ceil(t_transient / dt))
    f_n = max(mass * gravity * math.cos(theta), 1e-12)
    a_drive = gravity * math.sin(theta)
    # Start at the static equilibrium of the tangential spring instead of at
    # delta_t = 0.  Starting from zero makes the slider overshoot to 2*delta_eq,
    # which crosses the Coulomb cap and ratchets ("creep") even *below* the
    # threshold, contaminating the onset measurement.
    delta_t = np.array([[0.0, 0.0, mass * a_drive / kt]])
    v = 0.0
    s_disp = 0.0
    ts, ds = [], []
    for k in range(steps):
        v_t = np.array([[0.0, 0.0, v]])
        f_t = tangential_force(delta_t, v_t, np.array([f_n]), kt=kt, mu=mu,
                               ct=ct, cap_total=cap_total)[0]
        a = a_drive + f_t[2] / mass
        v += a * dt
        s_disp += v * dt
        delta_t = delta_t + np.array([[0.0, 0.0, v * dt]])
        if k >= i0:
            ts.append(k * dt)
            ds.append(s_disp)
    return _quadratic_curvature(np.asarray(ts), np.asarray(ds))


def _quadratic_curvature(ts: np.ndarray, ds: np.ndarray) -> float:
    """Return ``2 c2`` of the least-squares quadratic fit ``c0 + c1 t + c2 t^2``."""
    if ts.size < 5:
        return float("nan")
    t = ts - ts.mean()
    A = np.vstack([np.ones_like(t), t, t * t]).T
    c = np.linalg.lstsq(A, ds, rcond=None)[0]
    return float(2.0 * c[2])


def simulate_slider_onset_deg(mu: float, *, tolerance_deg: float = 0.02,
                              acceleration_tolerance: float = 1.0e-2,
                              **kwargs) -> float:
    """Bisect the tilt angle where the measured steady acceleration changes sign.

    Below ``arctan(mu)`` the tangential spring holds the slider (zero steady
    acceleration); above it the slider runs away.  Bisecting the *sign* of the
    measured acceleration (not a displacement threshold) removes the detection
    bias of a "did it move 1 mm" test.
    """
    def slides(tilt: float) -> bool:
        return slider_acceleration(mu, tilt, **kwargs) > acceleration_tolerance

    lo, hi = 0.0, 80.0
    if not slides(hi):
        return hi
    if slides(lo):
        return lo
    while hi - lo > tolerance_deg:
        mid = 0.5 * (lo + hi)
        if slides(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def tangential_damping_effect(mu: float, ct: float, *, tilt_offset_deg: float = 5.0,
                              **kwargs) -> dict:
    """Quantify the ``c_t * v_t`` branch: declared form vs cone-projected form.

    Three integrations of the *same* slider at a tilt 5 deg above
    ``arctan(mu)``: (i) pure Coulomb (``ct = 0``), (ii) the declared formula
    ``F_t = min(mu F_n, k_t delta_t) + c_t v_t`` without a cone projection,
    (iii) the same but with the total force projected back onto
    ``|F_t| <= mu F_n`` (kernel default).  Reporting the measured accelerations
    answers "what does the extra dashpot term cost in macroscopic friction?",
    which is the reason the kernel caps it.
    """
    tilt = sliding_onset_angle_deg(mu) + tilt_offset_deg
    analytic = 9.81 * (math.sin(math.radians(tilt)) - mu * math.cos(math.radians(tilt)))
    pure = slider_acceleration(mu, tilt, ct=0.0, **kwargs)
    literal = slider_acceleration(mu, tilt, ct=ct, cap_total=False, **kwargs)
    capped = slider_acceleration(mu, tilt, ct=ct, cap_total=True, **kwargs)
    return {
        "mu": float(mu),
        "ct": float(ct),
        "tilt_deg": tilt,
        "analytic_acceleration_m_s2": analytic,
        "pure_coulomb_acceleration_m_s2": pure,
        "literal_form_acceleration_m_s2": literal,
        "cone_projected_acceleration_m_s2": capped,
        "literal_relative_error": (literal - analytic) / analytic if analytic else None,
        "cone_projected_relative_error": (capped - analytic) / analytic if analytic else None,
    }


def simulate_rolling_onset_deg(mu_r: float, *, tolerance_deg: float = 0.02,
                               acceleration_tolerance: float = 1.0e-2,
                               radius: float = 5e-3, gravity: float = 9.81,
                               kn_ratio: float = 1.0e5, kr_ratio: float = 0.1,
                               t_end: float = 0.40, t_transient: float = 0.10,
                               dt: float | None = None) -> float:
    """Bisect the incline angle at which a pure-rolling sphere runs away.

    Single sphere rolling without slip (the tangential constraint is imposed
    exactly), so only the rolling-resistance couple resists:
    ``m a (1 + I/(mR^2)) = m g sin(theta) - M_r/R`` with
    ``M_r <= mu_r R F_n``, hence equilibrium is lost at ``tan(theta) = mu_r``.
    """
    inertia_ratio = 0.4                      # I / (m R^2), solid sphere
    kr = kr_ratio * kn_ratio * radius ** 2
    if dt is None:
        dt = 2.0 * math.pi * math.sqrt(inertia_ratio * radius ** 2 / kr) / 80.0
    steps = int(math.ceil(t_end / dt))
    i0 = int(math.ceil(t_transient / dt))

    def acceleration(tilt: float) -> float:
        """Curvature of the displacement fit; signed resisting couple included."""
        theta = math.radians(tilt)
        omega = 0.0
        v = 0.0
        s_disp = 0.0
        theta_r = np.zeros((1, 3))
        f_n = gravity * math.cos(theta)       # per unit mass
        ts, ds = [], []
        for k in range(steps):
            m_r = rolling_torque(theta_r, np.array([[0.0, omega, 0.0]]),
                                 np.array([f_n]), kr=kr, mu_r=mu_r,
                                 r_eff=radius)[0]
            # rotational/translation/rolling-constraint elimination:
            #   a (1 + I/(mR^2)) = g sin(theta) + M_r/(m R)
            a = (gravity * math.sin(theta) + float(m_r[1]) / radius) / (1.0 + inertia_ratio)
            v += a * dt
            s_disp += v * dt
            omega = v / radius
            theta_r = theta_r + np.array([[0.0, omega * dt, 0.0]])
            if k >= i0:
                ts.append(k * dt)
                ds.append(s_disp)
        return _quadratic_curvature(np.asarray(ts), np.asarray(ds))

    def rolls(tilt: float) -> bool:
        return acceleration(tilt) > acceleration_tolerance

    lo, hi = 0.0, 60.0
    if not rolls(hi):
        return hi
    if rolls(lo):
        return lo
    while hi - lo > tolerance_deg:
        mid = 0.5 * (lo + hi)
        if rolls(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
