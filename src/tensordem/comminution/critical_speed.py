"""Critical speed of a tumbling mill.

Theory
------
A ball riding on the smooth inner shell of a mill of inner diameter ``D`` loses
contact with the shell at the *top* of the path when the centrifugal
acceleration can no longer supply the centripetal requirement of gravity:

.. math:: m \\omega_c^2 r = m g ,\\qquad r = D/2 - d/2 ,

hence

.. math:: N_c = \\frac{60}{2\\pi}\\sqrt{\\frac{2g}{D}}\\ \\text{[rpm]}
    = \\frac{42.3}{\\sqrt{D}}\\ \\text{[rpm]} \\quad (D\\ \\text{in m}).

The constant ``42.3`` is *not* a free parameter: with ``g = 9.80665`` m/s^2,
``60/(2*pi)*sqrt(2g) = 42.297``, i.e. the textbook 42.3 is the rounded value of
``60/(2*pi)*sqrt(2g)``.  :func:`critical_speed_constant_from_first_principles`
recomputes it (a test asserts the two agree to <0.1%), so the implementation is
anchored on ``g`` and ``pi`` rather than on a memorised digit.

Two definitions are exposed because the difference matters at the 2% level of
the acceptance criterion:

``critical_speed_rpm(D)``
    the textbook value using the shell radius (``r = D/2``);
``critical_speed_rpm_charge_centre(D, ball_diameter)``
    the *exact* single-ball value using the ball-centre radius
    ``r = (D - d)/2``.  A discrete element simulation necessarily detects the
    latter (contact is defined centre-to-shell), so a DEM-vs-formula comparison
    must say which one it is comparing against.

Everything is SI except the returned speed, which is rpm because every mill
specification in the industry is in rpm.
"""
from __future__ import annotations

import math

__all__ = [
    "CRITICAL_SPEED_CONSTANT",
    "GRAVITY",
    "critical_speed_constant_from_first_principles",
    "critical_speed_rad_s",
    "critical_speed_rpm",
    "critical_speed_rpm_charge_centre",
    "fraction_of_critical",
    "rpm_for_fraction_of_critical",
    "shell_detachment_angle_deg",
    "froude_from_speed",
]

#: Textbook constant of ``N_c = 42.3 / sqrt(D)`` (rpm, D in m).
CRITICAL_SPEED_CONSTANT = 42.3

#: Standard gravity used by the platform (ISO 80000-3).
GRAVITY = 9.80665


def _positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return v


def critical_speed_constant_from_first_principles(gravity: float = GRAVITY) -> float:
    """``60/(2*pi)*sqrt(2g)`` -- the theoretical value of the 42.3 constant.

    With ``g = 9.80665`` this is ``42.2974...``, i.e. the textbook 42.3 to
    0.006%.
    """
    g = _positive(gravity, "gravity")
    return 60.0 / (2.0 * math.pi) * math.sqrt(2.0 * g)


def critical_speed_rpm(diameter: float) -> float:
    """Textbook critical speed ``N_c = 42.3/sqrt(D)`` [rpm], ``D`` in metres."""
    d = _positive(diameter, "diameter")
    return CRITICAL_SPEED_CONSTANT / math.sqrt(d)


def critical_speed_rpm_charge_centre(diameter: float, ball_diameter: float,
                                     gravity: float = GRAVITY) -> float:
    """Single-ball critical speed using the ball-centre radius ``(D-d)/2``.

    This is what a discrete element simulation resolves (the contact is
    detected centre-to-shell), so it is the correct DEM-side reference; it is
    always *larger* than :func:`critical_speed_rpm` by
    ``sqrt(D/(D-d))``.
    """
    d = _positive(diameter, "diameter")
    b = _positive(ball_diameter, "ball_diameter")
    g = _positive(gravity, "gravity")
    if b >= d:
        raise ValueError("ball_diameter must be smaller than the mill diameter")
    return 60.0 / (2.0 * math.pi) * math.sqrt(2.0 * g / (d - b))


def critical_speed_rad_s(diameter: float) -> float:
    """Textbook critical angular velocity ``omega_c = sqrt(2g/D)`` [rad/s]."""
    d = _positive(diameter, "diameter")
    return math.sqrt(2.0 * GRAVITY / d)


def fraction_of_critical(rpm: float, diameter: float) -> float:
    """``phi_c = N / N_c`` (mill speed as a fraction of critical)."""
    n = float(rpm)
    if not math.isfinite(n) or n < 0.0:
        raise ValueError("rpm must be finite and non-negative")
    return n / critical_speed_rpm(diameter)


def rpm_for_fraction_of_critical(fraction: float, diameter: float) -> float:
    """Inverse of :func:`fraction_of_critical`."""
    f = float(fraction)
    if not math.isfinite(f) or f <= 0.0:
        raise ValueError("fraction must be finite and positive")
    return f * critical_speed_rpm(diameter)


def froude_from_speed(rpm: float, diameter: float, gravity: float = GRAVITY) -> float:
    """Froude number ``Fr = omega^2 (D/2) / g``; ``Fr = 1`` is critical."""
    n = float(rpm)
    d = _positive(diameter, "diameter")
    g = _positive(gravity, "gravity")
    omega = 2.0 * math.pi * n / 60.0
    return omega * omega * (0.5 * d) / g


def shell_detachment_angle_deg(froude: float) -> float:
    """Release angle of a single ball measured from the 6 o'clock position.

    Balancing gravity against the centrifugal field in the rotating frame gives
    ``cos(theta_d) = -Fr`` (``theta_d`` measured from the bottom, rotating in
    the direction of the shell), so ``theta_d -> 180 deg`` exactly at
    ``Fr = 1``: this is the geometrical statement of the critical speed and is
    what the DEM detection in
    :func:`tensordem.comminution.mill_dem.measure_centrifuging_onset_rpm`
    measures.
    """
    fr = float(froude)
    if not math.isfinite(fr) or fr < 0.0:
        raise ValueError("froude must be finite and non-negative")
    if fr >= 1.0:
        return 180.0
    return math.degrees(math.acos(-fr))