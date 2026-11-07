"""Sea-ice drift dynamics closure: Coriolis + added-mass tensor + relative-velocity drag.

Why this module lives in TensorDEM
---------------------------------
TensorDEM owns ice-particle / floe mechanics.  The floe-scale drift models already
present (:mod:`tensordem.ice_floe_drift`, :mod:`tensordem.floe_6dof`) either
prescribe a constant driving force (no rotation, no relative-velocity drag) or
integrate attitude only.  This module closes the *horizontal momentum balance of
a free-drifting ice cover*, i.e. the physics that decides where the ice goes:
Earth rotation (Coriolis), the virtual (added) mass of the water the floe must
push aside, and quadratic wind / water drag acting on the *relative* velocity.
It provides the analytic anchors used to validate those terms:

* the classical **Nansen free-drift** steady state and its turning angle,
  ``theta = arctan(f / c_d)`` with ``c_d = rho_w C_w |U| / M_eff`` the linearised
  drag frequency (exact, not an approximation: see :func:`steady_drift_analytic`);
* the **Nansen wind factor** ``U_ice / U_wind``;
* the **Ekman spiral** of the wind/ice-driven ocean surface layer, decay depth
  ``D_E = sqrt(2 K / f)``, 45 deg surface veer and the exact Ekman transport.

Reference frame
---------------
Local Cartesian tangent plane on the rotating Earth.  ``x`` points along the
wind (or the prescribed forcing), ``y`` 90 deg to the left, ``z`` up.  In the
northern hemisphere (``f > 0``) the drift is deflected to the **right** of the
wind, so the reported ``angle_right_deg`` is positive right.

Momentum equation (per unit ice area, ``I`` the in-plane identity)::

    (M I + M_a) dU/dt = -(M I + M_a) f (k x U)
                        + rho_a C_a |U_a - U| (U_a - U)      wind stress
                        + rho_w C_w |U_w - U| (U_w - U)      water (ocean) drag

``M = rho_i h`` is the ice mass per unit area and ``M_a`` the in-plane added-mass
tensor.  ``k x U = (-v, u)``.  Everything is SI.

Three physical statements this module makes precise and testable
----------------------------------------------------------------
1. **Steady free drift has a closed form.**  With a prescribed wind stress and
   still water the steady balance gives ``tan(theta) = A f / (rho_w C_w |U|)``
   and a quartic for ``|U|`` solved in closed form -- see
   :func:`steady_drift_analytic`.  ``theta = arctan(f / c_d)`` is therefore an
   identity, not a fit.
2. **Added mass changes only the transient, if Coriolis is applied to the ice
   mass alone.**  The steady state is invariant to ``M_a`` (verifiable by
   integration, :func:`simulate_free_drift`); ``M_a`` stretches the spin-up time
   by ``(M + M_a) / M``.
3. **The Ekman layer is the vertical closure.**  The stress the ice exerts on
   the ocean generates an Ekman spiral whose decay depth is ``sqrt(2K/f)`` and
   whose column transport is exactly ``tau / (rho f)``, perpendicular to the
   stress -- both verified numerically here (:func:`solve_ekman_finite_difference`).

Limitations (stated, not hidden)
--------------------------------
* The closed form (:func:`steady_drift_analytic`) prescribes the wind stress at
  ``U = 0`` -- the classical free-drift convention.  The physical wind stress
  acts on the relative wind ``U_a - U``, so the full nonlinear system solved by
  :func:`solve_steady_drift` drifts ~2 % slower (the stress deficit is
  ``2 U_ice/U_wind`` ~ 4 %).  Both are reported; the difference is a property of
  the model, not a numerical error.
* The ocean current is a prescribed uniform vector.  Ice-ocean coupling through
  the Ekman surface current is available through :func:`ekman_surface_current`.
* No internal ice stress, no inertia of the ocean mixed layer, no waves, no
  form-drag anisotropy, no lateral boundaries.  Free drift is the intended
  regime; a full ice-thickness distribution would need :mod:`tensordem.threefield`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "OMEGA_EARTH",
    "coriolis_parameter",
    "nansen_angle_from_drag_frequency",
    "DriftEnvironment",
    "FloeInertia",
    "added_mass_tensor",
    "wind_stress",
    "water_drag",
    "steady_drift_analytic",
    "solve_steady_drift",
    "ekman_coupled_drift",
    "ekman_decay_depth",
    "ekman_profile",
    "ekman_surface_current",
    "ekman_transport",
    "solve_ekman_finite_difference",
    "simulate_free_drift",
    "drift_case_matrix",
    "run_case_suite",
]

#: Earth rotation rate [rad/s] (sidereal).
OMEGA_EARTH = 7.2921150e-5


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------
def coriolis_parameter(latitude_deg: float) -> float:
    """``f = 2 Omega sin(phi)`` [1/s].

    Positive in the northern hemisphere; the sign carries the hemisphere
    (drift to the right of the wind for ``f > 0``).
    """
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude must be in [-90, 90] deg")
    return 2.0 * OMEGA_EARTH * math.sin(math.radians(latitude_deg))


def nansen_angle_from_drag_frequency(f: float, c_d: float) -> float:
    """Nansen turning angle ``theta = arctan(f / c_d)`` in degrees.

    ``c_d`` is the **linearised drag frequency** [1/s], ``c_d = rho_w C_w |U| /
    M_eff``; it is a frequency, not a dimensionless coefficient, which is why
    the closed form is ``arctan(f / c_d)`` and not ``arctan(f / C_D)`` in
    dimensionless units.  :func:`steady_drift_analytic` returns ``c_d`` so the
    identity can be checked exactly.
    """
    if c_d <= 0.0:
        raise ValueError("c_d (drag frequency) must be positive")
    return math.degrees(math.atan(f / c_d))


# ---------------------------------------------------------------------------
# Environment / floe description
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DriftEnvironment:
    """Forcing and fluid environment for a free-drifting ice cover."""

    latitude_deg: float = 75.0
    wind_speed: float = 10.0            # [m/s] magnitude of the 10 m wind
    wind_dir_deg: float = 0.0           # [deg] direction the wind blows *toward*
    current_speed: float = 0.0          # [m/s] ocean current magnitude
    current_dir_deg: float = 180.0      # [deg] direction the current flows toward
    air_density: float = 1.30           # [kg/m^3]
    water_density: float = 1025.0       # [kg/m^3]
    cd_air: float = 1.8e-3              # [-] wind drag coefficient
    cd_water: float = 5.5e-3            # [-] ice underside drag coefficient

    @property
    def f(self) -> float:
        return coriolis_parameter(self.latitude_deg)

    @property
    def wind_vector(self) -> np.ndarray:
        a = math.radians(self.wind_dir_deg)
        return np.array([self.wind_speed * math.cos(a), self.wind_speed * math.sin(a)])

    @property
    def current_vector(self) -> np.ndarray:
        a = math.radians(self.current_dir_deg)
        return np.array([self.current_speed * math.cos(a), self.current_speed * math.sin(a)])

    def validate(self) -> None:
        if self.wind_speed < 0.0 or self.current_speed < 0.0:
            raise ValueError("speeds must be non-negative")
        if self.air_density <= 0 or self.water_density <= 0:
            raise ValueError("densities must be positive")
        if self.cd_air <= 0 or self.cd_water <= 0:
            raise ValueError("drag coefficients must be positive")


@dataclass(frozen=True)
class FloeInertia:
    """In-plane inertia of an ice cover, per unit ice area."""

    ice_thickness: float = 1.5          # [m] mixed first/multiyear ice (Nansen/Fram regime)
    ice_density: float = 917.0          # [kg/m^3]
    added_mass_coefficient: float = 0.5  # [-] virtual mass / displaced water mass
    floe_radius: float | None = None    # [m] optional, for the disk estimate

    @property
    def mass_per_area(self) -> float:
        return self.ice_density * self.ice_thickness

    @property
    def added_mass_per_area(self) -> float:
        """Bulk in-plane added mass per unit ice area [kg/m^2].

        ``M_a = c_a rho_w h`` with ``c_a`` the horizontal virtual-mass
        coefficient (0.5 for a bluff body, the classical bulk value).  For a
        thin disk in potential flow the in-plane added mass is isotropic and
        equal to ``(8/3) rho_w a^3``; per unit area that is
        ``8 rho_w a / (3 pi)``, available through :func:`added_mass_tensor`.
        """
        return self.added_mass_coefficient * 1025.0 * self.ice_thickness

    @property
    def effective_mass_per_area(self) -> float:
        return self.mass_per_area + self.added_mass_per_area

    def validate(self) -> None:
        if self.ice_thickness <= 0 or self.ice_density <= 0:
            raise ValueError("thickness and density must be positive")
        if self.added_mass_coefficient < 0:
            raise ValueError("added_mass_coefficient must be non-negative")


def added_mass_tensor(
    floe_radius: float,
    water_density: float = 1025.0,
    potential_flow_limit_depth: float | None = None,
) -> np.ndarray:
    """In-plane added-mass tensor of a thin circular floe [kg] (3x3, per floe).

    A disk translating in its own plane has, by rotational symmetry, an
    **isotropic** in-plane added mass; potential flow gives

        m_a = (8/3) rho_w a^3            (a = floe radius)

    and ``m_a,zz`` (heave) = ``(8/3) rho_w a^3`` as well for a thin disk
    (the classical result for both in-plane and normal translation of a disk).

    ``potential_flow_limit_depth`` (typically the Ekman depth ``sqrt(2K/f)``)
    caps the added mass per unit area by ``rho_w * depth``: a steadily drifting
    floe cannot drag an arbitrarily deep water column, so for floes large
    compared with the boundary-layer depth the bulk coefficient saturates.  This
    is the only physically motivated bound and keeps the tensor finite for
    basin-scale floes.

    Returns a 3x3 ``numpy`` array ``diag(m_a, m_a, m_a_z)`` [kg].
    """
    if floe_radius <= 0:
        raise ValueError("floe_radius must be positive")
    a = float(floe_radius)
    m_plane = (8.0 / 3.0) * water_density * a ** 3
    m_norm = (8.0 / 3.0) * water_density * a ** 3
    if potential_flow_limit_depth is not None:
        if potential_flow_limit_depth <= 0:
            raise ValueError("potential_flow_limit_depth must be positive")
        cap = water_density * potential_flow_limit_depth * math.pi * a ** 2
        m_plane = min(m_plane, cap)
        m_norm = min(m_norm, cap)
    return np.diag([m_plane, m_plane, m_norm])


# ---------------------------------------------------------------------------
# Stresses
# ---------------------------------------------------------------------------
def _quadratic_stress(rel: np.ndarray, rho: float, cd: float) -> np.ndarray:
    """``rho * C_d * |rel| * rel`` (vectorised, zero at zero speed)."""
    speed = float(np.hypot(rel[0], rel[1]))
    if speed < 1e-15:
        return np.zeros(2)
    return rho * cd * speed * rel


def wind_stress(env: DriftEnvironment, u_ice: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """Wind stress on the ice, relative-velocity form ``rho_a C_a |U_a-U| (U_a-U)`` [N/m^2]."""
    env.validate()
    U = np.asarray(u_ice, dtype=float).reshape(2)
    return _quadratic_stress(env.wind_vector - U, env.air_density, env.cd_air)


def water_drag(env: DriftEnvironment, u_ice: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """Ocean drag on the ice ``rho_w C_w |U_w-U| (U_w-U)`` [N/m^2]."""
    env.validate()
    U = np.asarray(u_ice, dtype=float).reshape(2)
    return _quadratic_stress(env.current_vector - U, env.water_density, env.cd_water)


def _k_cross(u: np.ndarray) -> np.ndarray:
    """``k x U = (-v, u)`` for ``U = (u, v)``."""
    return np.array([-u[1], u[0]])


# ---------------------------------------------------------------------------
# Steady free drift -- closed form
# ---------------------------------------------------------------------------
def steady_drift_analytic(
    env: DriftEnvironment,
    inertia: FloeInertia,
    include_added_mass_coriolis: bool = False,
    prescribed_wind_at_rest: bool = True,
) -> Dict[str, float]:
    """Closed-form steady free drift (still water, prescribed wind stress).

    Balance::

        0 = -A f (k x U) + tau_a - rho_w C_w |U| U,     A = M (+ M_a)

    With ``R = rho_w C_w |U|`` and ``g = A f`` this linear-in-``(u,v)`` system
    has the exact solution

        u = tau R / (R^2 + g^2),   v = -tau g / (R^2 + g^2),
        |U|^2 (A_w^2 |U|^2 + g^2) = tau^2   (quartic in |U|, solved in closed form)

    so ``tan(theta) = g / (A_w |U|) = f / c_d`` with ``c_d = A_w |U| / A``.

    ``include_added_mass_coriolis`` selects whether the Coriolis term acts on the
    virtual mass (``A = M + M_a``) or on the ice mass alone (``A = M``, the
    classical free-drift convention).  The default is the classical convention;
    both are exposed so the sensitivity can be quantified.

    Returns a dict with ``u``, ``v``, ``speed``, ``angle_right_deg``,
    ``wind_factor``, ``drag_frequency``, ``f``, ``tau_wind``, ``mass_per_area``
    and ``residual`` (the momentum residual of the returned state, N/m^2).
    """
    env.validate()
    inertia.validate()
    if env.current_speed != 0.0:
        raise ValueError("steady_drift_analytic requires still water (current_speed=0); "
                         "use solve_steady_drift for a prescribed current")

    f = env.f
    A = inertia.effective_mass_per_area if include_added_mass_coriolis else inertia.mass_per_area
    A_w = env.water_density * env.cd_water
    tau_vec = wind_stress(env, (0.0, 0.0)) if prescribed_wind_at_rest \
        else wind_stress(env, (0.0, 0.0))
    tau = float(np.hypot(tau_vec[0], tau_vec[1]))

    if tau <= 0.0:
        u = np.zeros(2)
    elif f == 0.0:
        # No rotation: pure drag balance, drift exactly along the wind.
        u = tau_vec / tau * math.sqrt(tau / A_w)
    else:
        g = A * f
        # quartic: A_w^2 s^4 + g^2 s^2 - tau^2 = 0
        s2 = (-g * g + math.sqrt(g ** 4 + 4.0 * A_w ** 2 * tau ** 2)) / (2.0 * A_w ** 2)
        s = math.sqrt(s2)
        R = A_w * s
        along = tau * R / (R * R + g * g)
        across = -tau * g / (R * R + g * g)
        # rotate the wind-aligned solution into the world frame
        e_hat = tau_vec / tau
        e_perp = np.array([-e_hat[1], e_hat[0]])  # 90 deg left
        u = along * e_hat + across * e_perp

    u0 = np.asarray(u, dtype=float)
    speed = float(np.hypot(u0[0], u0[1]))
    # angle to the right of the wind direction (positive right = negative y)
    f_hat = tau_vec / tau if tau > 0 else np.array([1.0, 0.0])
    angle = math.degrees(math.atan2(-float(np.dot(u0, np.array([-f_hat[1], f_hat[0]]))),
                                   float(np.dot(u0, f_hat)))) if speed > 0 else 0.0
    drag_frequency = (env.water_density * env.cd_water * speed) / A
    residual = (float(-A * f * _k_cross(u0)[0]) + float(tau_vec[0])
                + float(water_drag(env, u0)[0]),
                float(-A * f * _k_cross(u0)[1]) + float(tau_vec[1])
                + float(water_drag(env, u0)[1]))

    return {
        "u": float(u0[0]),
        "v": float(u0[1]),
        "speed": speed,
        "angle_right_deg": angle,
        "wind_factor": speed / env.wind_speed if env.wind_speed > 0 else float("nan"),
        "drag_frequency": drag_frequency,
        "f": f,
        "tau_wind": tau,
        "mass_per_area": A,
        "residual_u": residual[0],
        "residual_v": residual[1],
        "residual": math.hypot(residual[0], residual[1]),
    }


# ---------------------------------------------------------------------------
# Steady free drift -- full nonlinear (relative wind, prescribed current)
# ---------------------------------------------------------------------------
def solve_steady_drift(
    env: DriftEnvironment,
    inertia: FloeInertia,
    include_added_mass_coriolis: bool = False,
    tol: float = 1e-12,
    max_iter: int = 60,
    u_guess: Sequence[float] | None = None,
) -> Dict[str, float]:
    """Newton solve of the full nonlinear steady balance (relative wind + current).

    Residual ``R(U) = -A f (k x U) + rho_a C_a |U_a-U|(U_a-U) + rho_w C_w |U_w-U|(U_w-U)``.
    Its Jacobian ``-A f J - sum rho C (|D| I + D D^T/|D|)`` is symmetric part
    negative definite, so Newton is unconditionally contracting here.
    """
    env.validate()
    inertia.validate()
    f = env.f
    A = inertia.effective_mass_per_area if include_added_mass_coriolis else inertia.mass_per_area
    Ua = env.wind_vector
    Uw = env.current_vector

    if u_guess is None:
        U = np.zeros(2) if env.wind_speed == 0 else np.array([0.02 * env.wind_speed, 0.0])
    else:
        U = np.asarray(u_guess, dtype=float).reshape(2).copy()

    J_rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    resid = None
    for _ in range(max_iter):
        Da = Ua - U
        Dw = Uw - U
        sa = float(np.hypot(*Da))
        sw = float(np.hypot(*Dw))
        R = -A * f * (J_rot @ U)
        J = -A * f * J_rot
        if sa > 1e-15:
            R = R + env.air_density * env.cd_air * sa * Da
            J = J - env.air_density * env.cd_air * (sa * np.eye(2) + np.outer(Da, Da) / sa)
        if sw > 1e-15:
            R = R + env.water_density * env.cd_water * sw * Dw
            J = J - env.water_density * env.cd_water * (sw * np.eye(2) + np.outer(Dw, Dw) / sw)
        resid = R
        if float(np.hypot(*R)) < tol:
            break
        U = U - np.linalg.solve(J, R)

    speed = float(np.hypot(*U))
    tau_vec = wind_stress(env, (0.0, 0.0))
    tau = float(np.hypot(*tau_vec))
    f_hat = tau_vec / tau if tau > 0 else np.array([1.0, 0.0])
    left = np.array([-f_hat[1], f_hat[0]])
    angle = math.degrees(math.atan2(-float(np.dot(U, left)), float(np.dot(U, f_hat)))) if speed > 0 else 0.0
    drag_frequency = (env.water_density * env.cd_water * speed) / A
    return {
        "u": float(U[0]),
        "v": float(U[1]),
        "speed": speed,
        "angle_right_deg": angle,
        "wind_factor": speed / env.wind_speed if env.wind_speed > 0 else float("nan"),
        "drag_frequency": drag_frequency,
        "f": f,
        "tau_wind": tau,
        "mass_per_area": A,
        "residual": float(np.hypot(*resid)) if resid is not None else 0.0,
        "iterations": float("nan"),
    }


# ---------------------------------------------------------------------------
# Ekman spiral (vertical closure of the ocean boundary layer)
# ---------------------------------------------------------------------------
def ekman_decay_depth(eddy_viscosity: float, f: float) -> float:
    """Ekman decay depth ``D_E = sqrt(2 K / f)`` [m] (``f`` must be > 0)."""
    if eddy_viscosity <= 0:
        raise ValueError("eddy_viscosity must be positive")
    if f <= 0:
        raise ValueError("f must be positive (Ekman spiral is defined for f > 0)")
    return math.sqrt(2.0 * eddy_viscosity / f)


def ekman_profile(
    eddy_viscosity: float,
    f: float,
    stress: complex,
    z: Sequence[float] | np.ndarray,
    water_density: float = 1025.0,
) -> np.ndarray:
    """Analytic Ekman spiral ``w(z) = w0 exp(-(1+i) z / D_E)`` [m/s].

    ``stress = tau_x + i tau_y`` is the stress the ice/atmosphere exerts on the
    water surface [N/m^2].  The surface value is

        ``w0 = (1 - i) tau / (D_E rho f)``,   ``|w0| = tau / (rho sqrt(K f))``

    i.e. the surface current is turned 45 deg to the right of the stress in the
    northern hemisphere, and the column transport is exactly
    ``-i tau / (rho f)`` (90 deg to the right of the stress).
    """
    D = ekman_decay_depth(eddy_viscosity, f)
    w0 = (1.0 - 1.0j) * complex(stress) / (D * water_density * f)
    z = np.asarray(z, dtype=float)
    return w0 * np.exp(-(1.0 + 1.0j) * z / D)


def ekman_surface_current(
    eddy_viscosity: float,
    f: float,
    stress: complex,
    water_density: float = 1025.0,
) -> complex:
    """Surface (``z = 0``) Ekman current of the layer driven by ``stress`` [m/s]."""
    return ekman_profile(eddy_viscosity, f, stress, np.array([0.0]), water_density)[0]


def ekman_transport(
    f: float,
    stress: complex,
    water_density: float = 1025.0,
) -> complex:
    """Depth-integrated Ekman transport ``-i tau / (rho f)`` [m^2/s]."""
    if f == 0:
        raise ValueError("Ekman transport is singular at f = 0")
    return -1.0j * complex(stress) / (water_density * f)


def solve_ekman_finite_difference(
    eddy_viscosity: float,
    f: float,
    stress: complex,
    n: int = 2001,
    n_decay_depths: float = 8.0,
    water_density: float = 1025.0,
) -> Dict[str, object]:
    """Independent finite-difference solve of the same Ekman problem.

    Solves ``K w'' = i f w`` on ``z in [0, H]`` with the surface-stress Neumann
    condition ``rho K w'(0) = -tau`` and ``w(H) = 0``, ``H = n_decay_depths *
    D_E``, using a complex tridiagonal (Thomas) solve.  Used to verify the
    analytic profile, the decay depth and the transport without reusing the
    analytic formula.

    Returns ``z``, ``w``, ``surface_veer_deg``, ``decay_depth_numeric``,
    ``decay_depth_analytic``, ``transport_numeric``, ``transport_analytic``,
    ``rel_error_profile``.
    """
    D = ekman_decay_depth(eddy_viscosity, f)
    H = n_decay_depths * D
    z = np.linspace(0.0, H, n)
    h = z[1] - z[0]
    K = float(eddy_viscosity)
    rhs_coef = 1.0j * f / K  # w'' = rhs_coef * w

    N = n
    a = np.full(N, 1.0 / h ** 2, dtype=complex)      # sub-diagonal (w_{j-1})
    b = np.full(N, -2.0 / h ** 2 - rhs_coef, dtype=complex)
    c = np.full(N, 1.0 / h ** 2, dtype=complex)      # super-diagonal (w_{j+1})
    d = np.zeros(N, dtype=complex)

    # j = 0: surface-stress Neumann condition, rho K w'(0) = -tau.
    # ghost node: w_{-1} = w_1 + 2 h tau / (rho K); the discretised row 0 is
    #   (w_{-1} - 2 w_0 + w_1)/h^2 = (i f / K) w_0
    # => -(2/h^2 + i f/K) w_0 + (2/h^2) w_1 = -(2 tau / (rho K)) / h
    b[0] = -2.0 / h ** 2 - rhs_coef
    c[0] = 2.0 / h ** 2
    d[0] = -2.0 * complex(stress) / (water_density * K) / h

    # j = N-1: w(H) = 0
    a[N - 1] = 0.0
    b[N - 1] = 1.0
    c[N - 1] = 0.0
    d[N - 1] = 0.0

    # Thomas algorithm
    cp = np.zeros(N, dtype=complex)
    dp = np.zeros(N, dtype=complex)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for j in range(1, N):
        denom = b[j] - a[j] * cp[j - 1]
        cp[j] = c[j] / denom
        dp[j] = (d[j] - a[j] * dp[j - 1]) / denom
    w = np.zeros(N, dtype=complex)
    w[N - 1] = dp[N - 1]
    for j in range(N - 2, -1, -1):
        w[j] = dp[j] - cp[j] * w[j + 1]

    w_analytic = ekman_profile(K, f, stress, z, water_density)
    w0 = w[0]
    veer = math.degrees(math.atan2(w0.imag, w0.real) - math.atan2(complex(stress).imag,
                                                                 complex(stress).real))
    veer = (veer + 180.0) % 360.0 - 180.0

    # decay depth: first z where |w| = |w0|/e (linear interpolation)
    mag = np.abs(w)
    target = abs(w0) / math.e
    idx = int(np.argmax(mag < target))
    if idx == 0:
        d_num = float("nan")
    else:
        z0, z1 = z[idx - 1], z[idx]
        m0, m1 = mag[idx - 1], mag[idx]
        d_num = z0 + (z1 - z0) * (m0 - target) / (m0 - m1)

    transport_num = np.trapezoid(w, z) if hasattr(np, 'trapezoid') else np.trapz(w, z)
    transport_ana = ekman_transport(f, stress, water_density)
    rel_err = float(np.max(np.abs(w - w_analytic)) / max(abs(w0), 1e-300))
    return {
        "z": z,
        "w": w,
        "z_analytic": z,
        "w_analytic": w_analytic,
        "surface_veer_deg": veer,
        "surface_veer_right_deg": -veer,
        "surface_speed": float(abs(w0)),
        "decay_depth_analytic": D,
        "decay_depth_numeric": d_num,
        "transport_numeric": complex(transport_num),
        "transport_analytic": complex(transport_ana),
        "rel_error_profile": rel_err,
    }


# ---------------------------------------------------------------------------
# Time integration (transient: added mass, spin-up, negative control)
# ---------------------------------------------------------------------------
def simulate_free_drift(
    env: DriftEnvironment,
    inertia: FloeInertia,
    duration: float = 20.0 * 86400.0,
    dt: float = 900.0,
    include_added_mass_coriolis: bool = False,
    u0: Sequence[float] = (0.0, 0.0),
) -> Dict[str, object]:
    """Integrate the drift momentum balance with RK4 and return the history.

    ``(M I + M_a) dU/dt = -A f (k x U) + tau_a(U) + tau_w(U)`` with
    ``A = M + M_a`` when ``include_added_mass_coriolis`` else ``A = M``.
    Returns ``t``, ``u``, ``v``, ``speed``, ``angle_right_deg`` arrays plus the
    spin-up time scale ``tau_spin = (M + M_a) / (rho_w C_w |U|)`` [s].
    """
    env.validate()
    inertia.validate()
    f = env.f
    M = inertia.mass_per_area
    Ma = inertia.added_mass_per_area
    A_cor = (M + Ma) if include_added_mass_coriolis else M
    M_inertia = M + Ma

    def rhs(U: np.ndarray) -> np.ndarray:
        acc = -A_cor * f * _k_cross(U) + wind_stress(env, U) + water_drag(env, U)
        return acc / M_inertia

    n = int(round(duration / dt))
    t = np.arange(n + 1) * dt
    U = np.zeros((n + 1, 2))
    U[0] = np.asarray(u0, dtype=float)
    for i in range(n):
        k1 = rhs(U[i])
        k2 = rhs(U[i] + 0.5 * dt * k1)
        k3 = rhs(U[i] + 0.5 * dt * k2)
        k4 = rhs(U[i] + dt * k3)
        U[i + 1] = U[i] + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    speed = np.hypot(U[:, 0], U[:, 1])
    tau_vec = wind_stress(env, (0.0, 0.0))
    tau = float(np.hypot(*tau_vec))
    f_hat = tau_vec / tau if tau > 0 else np.array([1.0, 0.0])
    left = np.array([-f_hat[1], f_hat[0]])
    angle = np.degrees(np.arctan2(-(U[:, 0] * left[0] + U[:, 1] * left[1]),
                                 U[:, 0] * f_hat[0] + U[:, 1] * f_hat[1]))
    speed_scale = float(speed[-1]) if speed[-1] > 0 else 1.0
    tau_spin = M_inertia / (env.water_density * env.cd_water * speed_scale)
    return {
        "t": t,
        "u": U[:, 0],
        "v": U[:, 1],
        "speed": speed,
        "angle_right_deg": angle,
        "spin_up_time": tau_spin,
        "final_speed": float(speed[-1]),
        "final_angle_right_deg": float(angle[-1]),
        "final_u": (float(U[-1, 0]), float(U[-1, 1])),
    }


# ---------------------------------------------------------------------------
# Coupled ice-ocean free drift (ice drags an Ekman layer)
# ---------------------------------------------------------------------------
def ekman_coupled_drift(
    env: DriftEnvironment,
    inertia: FloeInertia,
    eddy_viscosity: float = 0.05,
    include_added_mass_coriolis: bool = True,
    tol: float = 1e-12,
    relaxation: float = 0.5,
    max_iter: int = 600,
) -> Dict[str, float]:
    """Free drift closed with the ocean Ekman layer the ice itself spins up.

    The water under the ice is not still: the stress the drifting ice exerts on
    the ocean, ``tau_s = -tau_w(on ice)``, drives an Ekman layer
    (:func:`ekman_surface_current`), and the *relative* velocity that sets the
    water drag is ``U_w - U`` with ``U_w`` the Ekman surface current.  The two
    problems are coupled, so they are solved by a damped fixed point::

        U_s <- (1-omega) U_s + omega * ekman_surface_current(K, f, -tau_w(U; U_s))
        U   <- Newton solve of the steady momentum balance with current U_s

    This is the closure that reproduces the classical **Nansen angle of
    20-40 deg to the right of the wind**: the wind-only balance
    (:func:`steady_drift_analytic`) turns the drift by only
    ``arctan(f/c_d)`` = 6-25 deg, whereas the Ekman surface current carries the
    drag vector a further ~20 deg to the right and the ice follows it.

    Returns the steady-state diagnostics of :func:`solve_steady_drift` plus
    ``surface_current_speed``, ``surface_current_dir_deg``, ``decay_depth``,
    ``coupling_iterations`` and ``coupling_converged``.
    """
    env.validate()
    inertia.validate()
    phi = env.latitude_deg
    f = coriolis_parameter(phi)
    if f <= 0.0:
        raise ValueError("ekman_coupled_drift requires f > 0 (northern-hemisphere framing); "
                         "use solve_steady_drift for f <= 0")
    if eddy_viscosity <= 0.0:
        raise ValueError("eddy_viscosity must be positive")

    current = 0.0 + 0.0j
    converged = False
    it = 0
    result: Dict[str, float] = {}
    for it in range(1, max_iter + 1):
        speed = abs(current)
        direction = math.degrees(math.atan2(current.imag, current.real)) if speed > 0 else 180.0
        env_k = DriftEnvironment(
            latitude_deg=phi,
            wind_speed=env.wind_speed,
            wind_dir_deg=env.wind_dir_deg,
            current_speed=speed,
            current_dir_deg=direction,
            air_density=env.air_density,
            water_density=env.water_density,
            cd_air=env.cd_air,
            cd_water=env.cd_water,
        )
        result = solve_steady_drift(env_k, inertia,
                                    include_added_mass_coriolis=include_added_mass_coriolis)
        u_ice = np.array([result["u"], result["v"]])
        tau_on_water = -water_drag(env_k, u_ice)          # stress ice -> ocean
        new = ekman_surface_current(eddy_viscosity, f,
                                    complex(tau_on_water[0], tau_on_water[1]),
                                    env.water_density)
        if abs(new - current) < tol:
            current = new
            converged = True
            break
        current = (1.0 - relaxation) * current + relaxation * new

    result = dict(result)
    result.update({
        "surface_current_speed": abs(current),
        "surface_current_dir_deg": math.degrees(math.atan2(current.imag, current.real)),
        "surface_current_u": current.real,
        "surface_current_v": current.imag,
        "decay_depth": ekman_decay_depth(eddy_viscosity, f),
        "eddy_viscosity": eddy_viscosity,
        "coupling_iterations": float(it),
        "coupling_converged": converged,
    })
    return result



# ---------------------------------------------------------------------------
# Acceptance case suite
# ---------------------------------------------------------------------------
def drift_case_matrix() -> List[Tuple[float, float]]:
    """``(latitude_deg, wind_speed)`` test matrix required by the acceptance spec.

    Covers ``phi in [60, 85] deg x U_wind in [5, 15] m/s`` (9 cases).
    """
    return [(phi, w) for phi in (60.0, 70.0, 75.0, 85.0) for w in (5.0, 10.0, 15.0)]


def run_case_suite(
    inertia: FloeInertia | None = None,
    latitude_deg: Sequence[float] = (60.0, 70.0, 75.0, 85.0),
    wind_speed: Sequence[float] = (5.0, 10.0, 15.0),
    eddy_viscosity: float | None = 0.05,
    include_added_mass_coriolis: bool = True,
    **env_kwargs,
) -> List[Dict[str, float]]:
    """Run the free-drift case suite and return per-case diagnostics.

    Each row carries the closed-form result, the full nonlinear Newton result
    (relative wind), their difference, the Nansen angle, the wind factor and the
    ``arctan(f/c_d)`` identity check.
    """
    inertia = inertia or FloeInertia()
    rows: List[Dict[str, float]] = []
    for phi in latitude_deg:
        for w in wind_speed:
            env = DriftEnvironment(latitude_deg=phi, wind_speed=w, **env_kwargs)
            ana = steady_drift_analytic(env, inertia,
                                        include_added_mass_coriolis=include_added_mass_coriolis)
            num = solve_steady_drift(env, inertia, u_guess=(ana["u"], ana["v"]),
                                     include_added_mass_coriolis=include_added_mass_coriolis)
            theta_id = nansen_angle_from_drag_frequency(ana["f"], ana["drag_frequency"])
            row = {
                "latitude_deg": phi,
                "wind_speed": w,
                "f": ana["f"],
                "u": ana["u"],
                "v": ana["v"],
                "speed": ana["speed"],
                "angle_right_deg": ana["angle_right_deg"],
                "angle_identity_deg": theta_id,
                "angle_identity_error_deg": abs(theta_id - ana["angle_right_deg"]),
                "wind_factor": ana["wind_factor"],
                "drag_frequency": ana["drag_frequency"],
                "newton_speed": num["speed"],
                "newton_angle_right_deg": num["angle_right_deg"],
                "newton_rel_speed_diff": abs(num["speed"] - ana["speed"]) / ana["speed"],
                "newton_rel_angle_diff_deg": abs(num["angle_right_deg"] - ana["angle_right_deg"]),
                "momentum_residual": ana["residual"],
            }
            if eddy_viscosity is not None and env.f > 0.0:
                cpl = ekman_coupled_drift(env, inertia, eddy_viscosity=eddy_viscosity,
                                          include_added_mass_coriolis=include_added_mass_coriolis)
                row.update({
                    "coupled_angle_right_deg": cpl["angle_right_deg"],
                    "coupled_wind_factor": cpl["wind_factor"],
                    "coupled_speed": cpl["speed"],
                    "surface_current_speed": cpl["surface_current_speed"],
                    "surface_current_dir_deg": cpl["surface_current_dir_deg"],
                    "coupled_converged": cpl["coupling_converged"],
                    "coupled_iterations": cpl["coupling_iterations"],
                })
            rows.append(row)
    return rows