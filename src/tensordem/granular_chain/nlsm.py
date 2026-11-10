"""Strongly nonlinear solitary waves (Nesterenko) in uncompressed Hertzian chains.

Physics
-------
A chain of elastic spheres with centres on a line and **no precompression**
(``sonic vacuum``: the linear sound speed is zero) supports a compact,
shape-preserving solitary wave because the only restoring mechanism is the
Hertzian contact law

    F(delta) = k delta^{3/2},   k = (4/3) E* sqrt(R*),   a = R_i + R_j

with ``a`` the equilibrium centre spacing.  Balance of the leading nonlinearity
against the leading (discrete) dispersion gives a one-parameter family of
travelling waves whose *shape* is amplitude independent and whose *speed* grows
as the 1/6 power of the peak force -- the hallmark Nesterenko scaling.

Derivation implemented here (no fitted constants)
-------------------------------------------------
Discrete chain, ``u_n`` bead displacement, strain ``s = (u_n - u_{n+1})/a``
(dimensionless, compression positive), contact deformation ``delta = a s``::

    m u_tt = k a^{3/2} [ s(x - a/2)^{3/2} - s(x + a/2)^{3/2} ]

Expanding around ``x`` with ``f = s^{3/2}`` the even terms cancel, giving the
second-order-accurate (modified-equation) continuum limit::

    m s_tt = A (s^{3/2})_{xx} + C (s^{3/2})_{xxxx},
    A = k a^{5/2},   C = k a^{7/2} / 12

The 4th-order coefficient is *fixed by consistency with the exact discrete
dispersion relation of the linearised chain* ``omega^2 = 4 k_t/m sin^2(q a/2)``:
a truncated continuum operator must reproduce the ``q^4`` term
``k_t a^2 q^2 (1 - q^2 a^2/12)``, i.e. ``C = A a^2/12``.  (A naive half-spaced
central-difference expansion of ``f`` alone yields the factor-2 smaller
``a^2/24`` and a pulse 1.4x too narrow -- the difference is that the deformation
itself is a finite difference, so the a^2 correction enters *inside* the 3/2
power.  The measured pulse width confirms ``C = k a^{7/2}/12``.)

Travelling wave ``s(x - v t)``, integrated twice with ``s -> 0`` at infinity::

    m v^2 s = A s^{3/2} + C (s^{3/2})''

which is solved *exactly* by::

    s(x,t) = s_m cos^4( (x - x_0 - v t) / L ),   |x - x_0 - v t| <= pi L / 2
    L  = a sqrt(3)                        (width: amplitude independent)
    v^2 = (5/6) (A/m) sqrt(s_m)           (speed: amplitude dependent)

Converting to the peak contact force ``F_m = k a^{3/2} s_m^{3/2}``::

    v_s = sqrt(5/6) sqrt(k a^{5/2} / m) (F_m / (k a^{3/2}))^{1/6}   => v_s ~ F^{1/6}

and ``v_s ~ V_0^{1/5}`` for an impactor of velocity ``V_0`` (the well known
sonic-vacuum result), because ``F_m ~ V_0^{6/5}``.

The width is *amplitude independent* in this limit: the pulse **duration**
``tau = 2L/v_s`` therefore scales as ``F^{-1/6}``.  The  ``width ~ F^{1/3}``
statement sometimes attached to the sonic vacuum is checked explicitly in
:func:`tensordem.granular_chain.study.SolitaryWaveStudy` and reported as a
falsified sub-hypothesis when the measurement disagrees.

Precompressed (linear) limit
----------------------------
With a static contact force ``F_0`` (overlap ``delta_0 = (F_0/k)^{2/3}``) the
linearised tangent stiffness is ``k_t = (3/2) k sqrt(delta_0)`` and the long
wavelength sound speed is ``c_0 = a sqrt(k_t/m)``; for ``F_0 >> F_m`` the same
chain must return to ``v_s = c_0`` with no amplitude dependence (negative
control 3 of the campaign).

References (standard results reproduced, not copied constants)
--------------------------------------------------------------
* Nesterenko V.F., *Dynamics of Heterogeneous Materials*, Springer 2001, Ch.1.
* Coste C., Falcon E., Fauve S., PRB 56 (1997) 6104 (sonic vacuum pulse).
* Job S. et al., PRL 94 (2005) 178002 (amplitude-independent width).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# exact constants of the cos^4 solitary wave (see module docstring)
WIDTH_PREFACTOR = math.sqrt(3.0)          # L = a * sqrt(3)
SPEED_PREFACTOR = math.sqrt(5.0 / 6.0)    # v^2 = (5/6) A/m sqrt(s_m)


def hertz_stiffness(radius_i: float, radius_j: float, e_star: float,
                    multiplier: float = 1.0) -> float:
    """Hertz coefficient ``k`` of ``F = k delta^{3/2}`` for two spheres."""
    r_star = radius_i * radius_j / (radius_i + radius_j)
    return multiplier * (4.0 / 3.0) * e_star * math.sqrt(r_star)


def nesterenko_width(a: float) -> float:
    """cos^4 half-width parameter ``L`` (support is ``|x - x_c| <= pi L / 2``)."""
    return WIDTH_PREFACTOR * a


def nesterenko_speed_from_strain(strain_max: float, k: float, a: float, m: float) -> float:
    """``v_s = sqrt((5/6)(k a^{5/2}/m) sqrt(s_m))``."""
    return math.sqrt((5.0 / 6.0) * (k * a ** 2.5 / m) * math.sqrt(strain_max))


def nesterenko_speed(force_peak: float, k: float, a: float, m: float) -> float:
    """Solitary wave speed for a peak contact force ``F_m`` (``~ F_m^{1/6}``)."""
    strain = (force_peak / (k * a ** 1.5)) ** (2.0 / 3.0)
    return nesterenko_speed_from_strain(strain, k, a, m)


def nesterenko_peak_force_from_speed(speed: float, k: float, a: float, m: float) -> float:
    """Inverse of :func:`nesterenko_speed` (used to seed impact amplitudes)."""
    strain = ((speed ** 2) / ((5.0 / 6.0) * (k * a ** 2.5 / m))) ** 2
    return k * a ** 1.5 * strain ** 1.5


def nesterenko_pulse_duration(force_peak: float, k: float, a: float, m: float,
                              half_width_factor: float = math.pi / 2.0) -> float:
    """Time the pulse spends above its support: ``2 (pi/2) L / v_s``."""
    speed = nesterenko_speed(force_peak, k, a, m)
    return 2.0 * half_width_factor * nesterenko_width(a) / speed


def impact_peak_force(velocity: float, k: float, a: float, m: float) -> float:
    """Peak contact force implied by an impactor of mass ``m`` at ``velocity``.

    Energy balance on the first contact, ``(2/5) k delta^{5/2} = m V^2 / 2``.
    Used only to size the run length (``t_end``), never in a scored quantity.
    """
    if velocity <= 0.0:
        return 0.0
    delta = (1.25 * m * velocity ** 2 / k) ** 0.4
    return k * delta ** 1.5


def estimate_run_time(n_beads: int, k: float, a: float, m: float,
                      velocity: float, margin: float = 2.0) -> float:
    """Run length that lets the pulse cross the chain with a safety margin."""
    f_est = impact_peak_force(velocity, k, a, m)
    v_est = max(nesterenko_speed(f_est, k, a, m), 1.0) if f_est > 0 else 1.0
    return margin * n_beads * a / v_est


def linear_sound_speed(static_force: float, k: float, a: float, m: float) -> float:
    """Long-wavelength speed of a precompressed chain, ``c_0 = a sqrt(k_t/m)``."""
    if static_force <= 0.0:
        return 0.0
    delta0 = (static_force / k) ** (2.0 / 3.0)
    k_t = 1.5 * k * math.sqrt(delta0)
    return a * math.sqrt(k_t / m)


def nesterenko_profile_strain(x: Sequence[float], strain_max: float, x_centre: float,
                              a: float, width: float | None = None) -> np.ndarray:
    """``s(x) = s_m cos^4((x - x_c)/L)`` with compact support (Nesterenko approx.)."""
    xs = np.asarray(x, dtype=float)
    L = nesterenko_width(a) if width is None else float(width)
    u = (xs - x_centre) / L
    inside = np.abs(u) <= 0.5 * math.pi
    out = np.zeros_like(xs)
    out[inside] = strain_max * np.cos(u[inside]) ** 4
    return out


def nesterenko_profile_force(x: Sequence[float], force_peak: float, x_centre: float,
                             a: float, width: float | None = None) -> np.ndarray:
    """Contact force profile ``F = k (a s)^{3/2}`` of the Nesterenko solution."""
    xs = np.asarray(x, dtype=float)
    L = nesterenko_width(a) if width is None else float(width)
    u = (xs - x_centre) / L
    inside = np.abs(u) <= 0.5 * math.pi
    out = np.zeros_like(xs)
    out[inside] = force_peak * np.cos(u[inside]) ** 6
    return out


# --------------------------------------------------------------------------
# fits
# --------------------------------------------------------------------------
def loglog_slope(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Slope + intercept + R^2 of ``ln y = q ln x + b``."""
    xx = np.log(np.asarray(x, dtype=float))
    yy = np.log(np.asarray(y, dtype=float))
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[ok], yy[ok]
    if xx.size < 2:
        raise ValueError("need at least two finite points")
    q, b = np.polyfit(xx, yy, 1)
    pred = q * xx + b
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    return {
        "exponent": float(q),
        "prefactor": float(np.exp(b)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_points": int(xx.size),
    }


def exponential_decay_length(values: Sequence[float], spacing: float,
                              index0: int, index1: int) -> dict[str, Any]:
    """Fit ``ln F_n = ln F_0 - (n - n0) a / xi`` over a contact window.

    ``xi`` is returned in metres and normalised by the bead *diameter* so that it
    can be compared with the literature convention ``xi = (x_1/2 - x_0)/...``
    used for disordered chains (attenuation per bead diameter).
    """
    v = np.asarray(values, dtype=float)
    idx = np.arange(v.size, dtype=float)
    lo, hi = int(index0), int(index1)
    window = slice(lo, hi)
    sel = v[window] > 0.0
    x = idx[window][sel]
    y = np.log(v[window][sel])
    if x.size < 3:
        return {"xi_m": float("nan"), "slope_per_bead": float("nan"),
                "r2": float("nan"), "x0": float("nan"), "n_points": int(x.size)}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    xi = -spacing / slope if slope < 0 else float("inf")
    return {
        "xi_m": float(xi),
        "xi_in_diameters": float(xi / (spacing)),
        "slope_per_bead": float(slope),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "x0": float(np.exp(intercept)),
        "n_points": int(x.size),
    }


@dataclass
class NesterenkoWave:
    """Container for the analytic wave used as the comparison reference."""

    strain_max: float
    force_peak: float
    speed: float
    width: float
    spacing: float

    @classmethod
    def from_force(cls, force_peak: float, k: float, a: float, m: float) -> "NesterenkoWave":
        strain = (force_peak / (k * a ** 1.5)) ** (2.0 / 3.0)
        return cls(strain_max=strain, force_peak=force_peak,
                   speed=nesterenko_speed(force_peak, k, a, m),
                   width=nesterenko_width(a), spacing=a)

    @property
    def support(self) -> float:
        """Full spatial support of the pulse (``pi L``)."""
        return math.pi * self.width

    def strain(self, x: Sequence[float], x_centre: float) -> np.ndarray:
        return nesterenko_profile_strain(x, self.strain_max, x_centre, self.spacing,
                                         width=self.width)

    def force(self, x: Sequence[float], x_centre: float) -> np.ndarray:
        return nesterenko_profile_force(x, self.force_peak, x_centre, self.spacing,
                                        width=self.width)

    def as_dict(self) -> dict[str, float]:
        return {
            "strain_max": self.strain_max,
            "force_peak_N": self.force_peak,
            "speed_m_s": self.speed,
            "width_m": self.width,
            "support_m": self.support,
            "spacing_m": self.spacing,
        }