"""Transient (time-domain) anchors for free sea-ice drift: the near-inertial oscillation.

Physics scope
-------------
:mod:`tensordem.sea_ice_drift` closes the *steady* drift state.  This module
closes the *transient*: a slab of sea ice that is released from rest (or from a
misaligned initial velocity) in a rotating frame relaxes towards its steady
drift along a damped near-inertial spiral.

With an **isotropic linear** drag law the momentum balance linearised about the
steady state ``u_ss`` is *exactly* linear and constant-coefficient::

    m dw/dt = -m f (k x w) - m gamma w ,      w = u - u_ss

Writing ``z = w_x + i w_y`` (so that ``k x w <-> i z``)::

    dz/dt = -(gamma + i f) z      =>      z(t) = z(0) exp(-gamma t) exp(-i f t)

hence the exact closed form

    u(t) = u_ss + exp(-gamma t) R(-f t) (u_0 - u_ss)

with ``R(theta)`` the CCW rotation by ``theta``.  Two consequences, both
machine-checkable and used as acceptance anchors:

* the oscillation frequency is **exactly** ``|f|`` (not ``sqrt(f^2+gamma^2)``:
  the drag is a friction, not a restoring force),
* the envelope decays **exactly** as ``exp(-gamma t)`` and the spin-up amplitude
  ratio is ``|osc|/|u_ss| = exp(-gamma t)``.

Time-integration schemes
------------------------
``integrate`` offers four schemes for ``dw/dt = A w``, ``A = -gamma I - f K``:

=====================  ====================================================
``"exponential"``      exact: ``exp(A dt) = exp(-gamma dt) R(-f dt)`` (the
                       "exponential integrator" / semi-implicit limit)
``"crank_nicolson"``   ``(I - A dt/2)^{-1} (I + A dt/2)``, A-stable
``"rk4"``              classical fourth-order Runge-Kutta
``"euler"``            explicit Euler: ``|1 - dt(gamma + i f)| > 1`` as soon as
                       ``f dt`` is O(1) -> the *negative control*
=====================  ====================================================

``amplification`` returns the per-step modulus/phase of each scheme, i.e. the
effective numerical ``gamma``/``f`` a time series would show.  This is the
time-step criterion of the node.  For explicit Euler
``|1 - dt(gamma + i f)|^2 = 1 - 2 gamma dt + (gamma^2 + f^2) dt^2``, so the
scheme is *stable but wrong* for ``dt < dt* = 2 gamma/(gamma^2 + f^2)`` and
divergent above it.  With ``gamma = 8.5e-5 1/s``, ``f = 1.436e-4 1/s`` (80 N):
``dt* = 1.70 h``, and at the acceptance step ``dt = 1 h`` the modulus is
``sqrt((1 - gamma dt)^2 + (f dt)^2) = 0.865`` per step, i.e. the scheme *over-
damps* (62 % amplitude error after one inertial period) instead of growing.
Both regimes are exercised: the acceptance gate at ``dt = 1 h`` and a
divergence demonstration at ``dt = 2 dt*``.

Momentum ledger
---------------
``momentum_ledger`` re-substitutes a state and its analytic time derivative into
the *model's own* force terms (wind, water drag, Coriolis, tilt) and reports the
relative residual; ``discrete_ledger_residual`` does the same for a marched
solution using the exact integral of the linear solution over each step, i.e.
``m (w^{n+1} - w^n) - int_{t_n}^{t_{n+1}} F dt = 0``.  Both include the Coriolis
term, so a sign or rotation-convention error cannot hide.

Estimators
----------
``estimate_inertial_parameters`` recovers ``(gamma, omega, u_ss, A, B)`` from a
time series by two independent routes:

* ``method="recurrence"`` -- linear-prediction (Prony) least squares on the
  second-order recurrence that a two-exponential signal obeys.  Exact to
  machine precision on noise-free data (this is what the closed-form anchor
  check uses).
* ``method="nonlinear"`` -- separable non-linear least squares (grid/Nelder-Mead
  over ``gamma, omega`` with ``A, B`` solved linearly).  Unbiased under additive
  noise, therefore used for the stochastic ensemble.

Agreeing to ``< 1e-6`` on noise-free input is itself a check of the estimator,
i.e. the anchor test does not grade itself with the same formula it validates.

Everything SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .sea_ice_drift import (
    OMEGA_EARTH,
    SeaIceDriftModel,
    coriolis_parameter,
    deflection_angle,
)

__all__ = [
    "K_MATRIX",
    "DampedInertialSolution",
    "amplification",
    "discrete_ledger_residual",
    "estimate_inertial_parameters",
    "integrator_error",
    "integrate",
    "linearized_drag_rate",
    "momentum_ledger",
    "rotation_ccw",
]

#: ``k x w = K_MATRIX @ w`` for ``k = +z`` and ``w = (u, v)``.
K_MATRIX = np.array([[0.0, -1.0], [1.0, 0.0]])


def rotation_ccw(angle_rad: float) -> np.ndarray:
    """2x2 CCW rotation matrix."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------------------
# Drag linearisation
# ---------------------------------------------------------------------------

def linearized_drag_rate(
    model: SeaIceDriftModel,
    u_ice: Sequence[float] = (0.0, 0.0),
) -> float:
    """Isotropic drag rate ``gamma`` [1/s] of the momentum balance at ``u_ice``.

    For a quadratic relative-velocity drag ``tau = C |du| du`` the Jacobian with
    respect to the ice velocity is ``-C (|du| I + du du^T/|du|)``; its isotropic
    (trace) part defines the friction rate that governs the *envelope* of the
    near-inertial oscillation,

        gamma = (C_w (2|du_w|) + C_a (2|du_a|)) / (2 m)
              = (C_w |du_w| + C_a |du_a|) / m ,

    while the traceless part only rotates the perturbation (it is the source of
    the second-order deviation between the quadratic and the linear-drag
    anchors).  ``drag_mode="linear"`` returns the prescribed rate exactly.
    """
    m = model.mass_per_area
    if model.drag_mode == "linear":
        return float(model.linear_drag_rate)
    if model.drag_mode == "ekman":
        # tau_w = -rho_w sqrt(f K) R(+45) u_rel  ->  isotropic rate
        return float(model.ekman_drag_coefficient / m)
    u = np.asarray(u_ice, dtype=float)
    gamma = 0.0
    du_w = np.asarray(model.ocean_current, dtype=float) - u
    nw = float(np.linalg.norm(du_w))
    if nw > 0.0:
        gamma += model.water_drag.coefficient * nw / m
    if model.wind_relative:
        du_a = np.asarray(model.wind, dtype=float) - u
        na = float(np.linalg.norm(du_a))
        if na > 0.0:
            gamma += model.wind_drag.coefficient * na / m
    return float(gamma)


# ---------------------------------------------------------------------------
# Closed form
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DampedInertialSolution:
    """Exact solution of ``m dw/dt = -m f k x w - m gamma w`` about ``u_steady``.

    ``u(t) = u_ss + exp(-gamma t) R(-f t) (u_0 - u_ss)``.
    ``omega`` is the oscillation angular frequency and equals ``|f|`` *exactly*.
    """

    u_steady: np.ndarray
    u_initial: np.ndarray
    gamma: float
    f: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "u_steady", np.asarray(self.u_steady, dtype=float))
        object.__setattr__(self, "u_initial", np.asarray(self.u_initial, dtype=float))
        if self.gamma < 0.0:
            raise ValueError("gamma must be non-negative")

    # -- properties ---------------------------------------------------------
    @property
    def omega(self) -> float:
        """Oscillation angular frequency [rad/s]: exactly ``f``."""
        return self.f

    @property
    def period(self) -> float:
        """Inertial period ``2 pi / |f|`` [s]."""
        return 2.0 * math.pi / abs(self.f)

    @property
    def offset(self) -> np.ndarray:
        """Initial offset ``w_0 = u_0 - u_ss``."""
        return self.u_initial - self.u_steady

    # -- evaluation ---------------------------------------------------------
    def offset_at(self, t: float) -> np.ndarray:
        """``w(t)`` -- the oscillating part."""
        return float(np.exp(-self.gamma * t)) * (
            rotation_ccw(-self.f * t) @ self.offset
        )

    def velocity(self, t: float) -> np.ndarray:
        return self.u_steady + self.offset_at(t)

    def derivative(self, t: float) -> np.ndarray:
        """``du/dt`` from the closed form."""
        w = self.offset_at(t)
        return -self.gamma * w - self.f * (K_MATRIX @ w)

    def series(self, times: Sequence[float]) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        return np.array([self.velocity(float(ti)) for ti in t])

    def derivative_series(self, times: Sequence[float]) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        return np.array([self.derivative(float(ti)) for ti in t])

    def amplitude_ratio(self, t: float) -> float:
        """``|w(t)| / |u_ss|`` -- the spin-up amplitude ratio (analytic)."""
        denom = float(np.linalg.norm(self.u_steady))
        if denom == 0.0:
            raise ValueError("u_ss is zero: amplitude ratio undefined")
        return float(np.linalg.norm(self.offset_at(t))) / denom

    def envelope(self, t: float) -> float:
        """``|w(t)|``."""
        return float(np.linalg.norm(self.offset_at(t)))


def initial_state_closed_form(
    model: SeaIceDriftModel,
    t: float = 0.0,
    u_initial: Sequence[float] = (0.0, 0.0),
    u_steady: Optional[Sequence[float]] = None,
) -> DampedInertialSolution:
    """Build the closed form from a :class:`SeaIceDriftModel` (linear drag only).

    Requires ``drag_mode="linear"`` and ``wind_relative=False``: only then is the
    linearised problem exactly the constant-coefficient system this module
    solves.  Anything else raises instead of silently mislabelling a quadratic
    drag as an exponential decay.
    """
    if model.drag_mode != "linear":
        raise ValueError(
            "the exact transient closed form needs drag_mode='linear'; "
            f"got {model.drag_mode!r} (use linearized_drag_rate for the "
            "quadratic/ekman anchors, or integrate numerically)"
        )
    if model.wind_relative:
        raise ValueError(
            "wind_relative=True makes the wind stress velocity dependent; the "
            "closed form requires a prescribed wind stress (wind_relative=False)"
        )
    if u_steady is None:
        u_steady = model.steady_state()
    return DampedInertialSolution(
        u_steady=np.asarray(u_steady, dtype=float),
        u_initial=np.asarray(u_initial, dtype=float),
        gamma=float(model.linear_drag_rate),
        f=float(model.f),
    )


# ---------------------------------------------------------------------------
# Time integration schemes
# ---------------------------------------------------------------------------

SCHEMES = ("exponential", "crank_nicolson", "rk4", "euler")


def _step_matrix(scheme: str, dt: float, f: float, gamma: float) -> np.ndarray:
    """Per-step propagator for ``dw/dt = (-gamma I - f K) w``."""
    A = -gamma * np.eye(2) - f * K_MATRIX
    if scheme == "exponential":
        return float(np.exp(-gamma * dt)) * rotation_ccw(-f * dt)
    if scheme == "crank_nicolson":
        B = np.eye(2) - 0.5 * dt * A
        return np.linalg.solve(B, np.eye(2) + 0.5 * dt * A)
    if scheme == "rk4":
        P = np.eye(2)
        for coeff in (A, A):
            pass
        # 4th-order Taylor form of the matrix exponential of a 2x2 matrix
        A2 = A @ A
        return np.eye(2) + dt * A + 0.5 * dt**2 * A2 + dt**3 / 6.0 * (A2 @ A) + (
            dt**4 / 24.0 * (A2 @ A2)
        )
    if scheme == "euler":
        return np.eye(2) + dt * A
    raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")


def integrate(
    model_or_solution,
    dt: float,
    n_steps: int,
    scheme: str = "exponential",
    u_initial: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """March the linearised balance in time; returns ``(times, u)`` with ``u`` (n+1, 2).

    The propagator is applied to the *offset* ``w = u - u_ss`` so the scheme is
    exact for ``scheme="exponential"`` regardless of ``u_ss``.
    """
    if isinstance(model_or_solution, DampedInertialSolution):
        sol = model_or_solution
    else:
        sol = initial_state_closed_form(
            model_or_solution, u_initial=u_initial if u_initial is not None else (0.0, 0.0)
        )
    P = _step_matrix(scheme, dt, sol.f, sol.gamma)
    w = sol.offset.copy()
    times = np.zeros(n_steps + 1)
    us = np.zeros((n_steps + 1, 2))
    us[0] = sol.u_steady + w
    for n in range(n_steps):
        w = P @ w
        times[n + 1] = (n + 1) * dt
        us[n + 1] = sol.u_steady + w
    return times, us


def amplification(dt: float, f: float, gamma: float, scheme: str) -> Dict[str, float]:
    """Per-step modulus and phase of a scheme, mapped back to (gamma, f).

    ``modulus`` is the growth factor of the oscillation envelope per step
    (exactly ``exp(-gamma dt)`` for the exponential scheme and ``> 1`` for the
    explicit Euler scheme when ``f dt`` is O(1)); ``gamma_numeric`` and
    ``f_numeric`` are the rates a fit of the marched series would return.
    """
    P = _step_matrix(scheme, dt, f, gamma)
    ev = np.linalg.eigvals(P)
    # both eigenvalues are conjugates with the same modulus for all schemes here
    modulus = float(np.max(np.abs(ev)))
    phase = float(np.max(np.abs(np.angle(ev))))
    return {
        "modulus": modulus,
        "phase": phase,
        "gamma_numeric": -math.log(modulus) / dt,
        "f_numeric": phase / dt,
        "gamma_rel_error": (-math.log(modulus) / dt - gamma) / gamma if gamma else 0.0,
        "f_rel_error": (phase / dt - abs(f)) / abs(f),
        "amplitude_gain_per_day": modulus ** (86400.0 / dt),
    }


def integrator_error(
    sol: DampedInertialSolution,
    dt: float,
    n_steps: int,
    scheme: str,
) -> Dict[str, float]:
    """Max/relative error of a marched solution against the closed form."""
    times, us = integrate(sol, dt, n_steps, scheme=scheme)
    exact = sol.series(times)
    err = us - exact
    scale = max(float(np.max(np.linalg.norm(exact, axis=1))), 1e-300)
    steady_scale = max(float(np.linalg.norm(sol.u_steady)), 1e-300)
    amp_num = float(np.linalg.norm(us[-1] - sol.u_steady))
    amp_exact = float(np.linalg.norm(sol.offset_at(times[-1])))
    return {
        "scheme": scheme,
        "dt": dt,
        "n_steps": n_steps,
        "max_abs_error": float(np.max(np.abs(err))),
        "max_rel_error": float(np.max(np.abs(err)) / scale),
        "final_amplitude_ratio_error": abs(amp_num - amp_exact) / max(amp_exact, 1e-300),
        "final_speed_ratio": float(np.linalg.norm(us[-1])) / steady_scale,
        "diverged": bool(np.max(np.abs(err)) > 0.5 * scale or amp_num > 50.0 * steady_scale),
    }


# ---------------------------------------------------------------------------
# Momentum ledger (Coriolis included)
# ---------------------------------------------------------------------------

def momentum_ledger(
    model: SeaIceDriftModel,
    u_ice: Sequence[float],
    du_dt: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Per-term momentum budget [N/m^2] of ``m du/dt = sum(F)``.

    Returns the terms (wind, water, coriolis, tilt), their sum, the inertia
    ``m du/dt`` and the residual; ``residual_rel`` compares the residual with
    the largest term so it is dimensionless.
    """
    u = np.asarray(u_ice, dtype=float)
    a = np.asarray(du_dt, dtype=float)
    fr = model.forces(u)
    terms = fr["terms"]
    total = fr["net"]
    inertia = model.mass_matrix @ a
    residual = inertia - total
    scale = max(
        float(np.linalg.norm(terms["wind"])),
        float(np.linalg.norm(terms["water"])),
        float(np.linalg.norm(terms["coriolis"])),
        float(np.linalg.norm(terms["tilt"])),
        1e-300,
    )
    return {
        "wind": np.asarray(terms["wind"], dtype=float),
        "water": np.asarray(terms["water"], dtype=float),
        "coriolis": np.asarray(terms["coriolis"], dtype=float),
        "tilt": np.asarray(terms["tilt"], dtype=float),
        "total": np.asarray(total, dtype=float),
        "inertia": inertia,
        "residual": residual,
        "residual_rel": float(np.linalg.norm(residual)) / scale,
    }


def discrete_ledger_residual(
    sol: DampedInertialSolution,
    dt: float,
    n_steps: int,
    scheme: str = "exponential",
) -> float:
    """Discrete momentum-ledger residual of a marched solution.

    For each step the *exact* impulse of the two velocity-dependent forces over
    the step is available in closed form: with ``z(t) = z_n exp(-(gamma+if)(t-t_n))``

        int_0^dt w dt = w_n (1 - e^{-(gamma+if)dt}) / (gamma + i f),

    so an exactly integrated scheme must satisfy
    ``m (w^{n+1} - w^n) = -m gamma int w dt - m f int (k x w) dt`` to round-off.
    The returned value is the maximum over steps of ``|residual| / |m w^n|``.
    """
    times, us = integrate(sol, dt, n_steps, scheme=scheme)
    # exact impulse over a step: int_0^dt exp(A s) ds = A^{-1} (exp(A dt) - I)
    A_mat = -sol.gamma * np.eye(2) - sol.f * K_MATRIX
    E = float(np.exp(-sol.gamma * dt)) * rotation_ccw(-sol.f * dt)
    scale = max(float(np.linalg.norm(sol.offset)), 1e-300)
    if abs(sol.gamma) + abs(sol.f) == 0.0:
        impulse_op = dt * np.eye(2)
    else:
        impulse_op = np.linalg.solve(A_mat, E - np.eye(2))
    worst = 0.0
    for n in range(n_steps):
        w_n = us[n] - sol.u_steady
        w_np1 = us[n + 1] - sol.u_steady
        integral = impulse_op @ w_n          # int_0^dt w dt
        impulse_drag = -sol.gamma * integral
        impulse_coriolis = -sol.f * (K_MATRIX @ integral)
        resid = (w_np1 - w_n) - (impulse_drag + impulse_coriolis)
        worst = max(worst, float(np.linalg.norm(resid)) / scale)
    return worst


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _series_complex(times: np.ndarray, series: np.ndarray) -> np.ndarray:
    return series[:, 0] + 1j * series[:, 1]


def estimate_inertial_parameters(
    times: Sequence[float],
    series: Sequence[Sequence[float]],
    method: str = "recurrence",
    f_hint: Optional[float] = None,
    gamma_hint: Optional[float] = None,
) -> Dict[str, float]:
    """Recover ``(gamma, omega, u_ss)`` from a 2-D velocity time series.

    ``method="recurrence"`` solves the linear-prediction system

        w_x[n] = a1 w_x[n-1] + a2 w_x[n-2]

    whose characteristic roots are ``exp((-gamma -/+ i omega) dt)``; exact for
    noise-free two-exponential signals.  ``method="nonlinear"`` minimises the
    residual of ``w(t) = e^{-gamma t}(A cos omega t + B sin omega t)`` over
    ``(gamma, omega)`` with ``A, B`` solved linearly at each trial point
    (unbiased under additive noise).

    ``u_ss`` is estimated from the mean of the final 20 % of the record, which
    is 3-5 decay times for the cases of this repository.
    """
    t = np.asarray(times, dtype=float)
    u = np.asarray(series, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("series must have shape (n, 2)")
    if t.size != u.shape[0]:
        raise ValueError("times and series length differ")
    if t.size < 8:
        raise ValueError("need at least 8 samples")
    dt = float(t[1] - t[0])
    n_tail = max(4, t.size // 5)
    u_ss = np.mean(u[-n_tail:], axis=0)
    w = u - u_ss

    if method == "recurrence":
        design = np.column_stack([w[1:-1, 0], w[:-2, 0]])
        rhs = w[2:, 0]
        a, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        roots = np.roots([1.0, -a[0], -a[1]])
        # pick the root with the positive imaginary part (oscillation)
        root = roots[np.argmax(np.imag(roots))]
        ls = np.log(root) / dt
        gamma_hat = -float(np.real(ls))
        omega_hat = float(np.imag(ls))
    elif method == "nonlinear":
        gamma_hat, omega_hat = _nonlinear_search(t, w, f_hint, gamma_hint)
    else:
        raise ValueError("method must be 'recurrence' or 'nonlinear'")

    # amplitudes by linear least squares at the recovered rates
    env = np.exp(-gamma_hat * t)
    design = np.column_stack(
        [env * np.cos(omega_hat * t), env * np.sin(omega_hat * t)]
    )
    coef, *_ = np.linalg.lstsq(design, w, rcond=None)
    A = coef[0]
    B = coef[1]
    resid = w - design @ coef
    scale = max(float(np.max(np.linalg.norm(w, axis=1))), 1e-300)
    return {
        "gamma": gamma_hat,
        "omega": abs(omega_hat),
        "f": abs(omega_hat),
        "period": 2.0 * math.pi / abs(omega_hat) if omega_hat else float("inf"),
        "u_steady": u_ss,
        "speed_steady": float(np.linalg.norm(u_ss)),
        "A": A,
        "B": B,
        "amplitude0": float(np.linalg.norm(A)),
        "fit_residual_rel": float(np.max(np.linalg.norm(resid, axis=1))) / scale,
        "method": method,
    }


def _nonlinear_search(
    t: np.ndarray,
    w: np.ndarray,
    f_hint: Optional[float],
    gamma_hint: Optional[float],
    n_grid: int = 41,
) -> Tuple[float, float]:
    """Separable least squares over ``(gamma, omega)`` then Nelder-Mead polish."""
    # crude initial guesses
    env = np.linalg.norm(w, axis=1)
    pos = env > 0.15 * env.max()
    if np.count_nonzero(pos) > 3:
        slope = np.polyfit(t[pos], np.log(env[pos]), 1)[0]
        gamma0 = max(-slope, 1e-12)
    else:
        gamma0 = gamma_hint if gamma_hint else 1e-5
    if f_hint is not None:
        omega0 = abs(f_hint)
    else:
        z = _series_complex(t, w)
        spec = np.abs(np.fft.rfft(z * np.exp(gamma0 * t)))
        freqs = 2.0 * math.pi * np.fft.rfftfreq(t.size, t[1] - t[0])
        omega0 = float(freqs[np.argmax(spec[1:]) + 1])
    if gamma_hint:
        gamma0 = gamma_hint

    def cost(p):
        gam, om = p
        if gam < 0.0 or om <= 0.0:
            return 1e30
        e = np.exp(-gam * t)
        D = np.column_stack([e * np.cos(om * t), e * np.sin(om * t)])
        c, *_ = np.linalg.lstsq(D, w, rcond=None)
        r = w - D @ c
        return float(np.sum(r * r))

    # coarse scan around the initial guess, then local refine
    best = (gamma0, omega0)
    best_cost = cost(best)
    for gam in gamma0 * np.linspace(0.2, 5.0, n_grid):
        for om in omega0 * np.linspace(0.9, 1.1, n_grid):
            c = cost((gam, om))
            if c < best_cost:
                best, best_cost = (gam, om), c
    # Nelder-Mead by hand on a shrinking simplex (no scipy dependency needed)
    simplex = [np.array(best), np.array(best) * np.array([1.05, 1.0]) + 1e-12,
               np.array(best) * np.array([1.0, 1.005]) + 1e-12]
    values = [cost(p) for p in simplex]
    for _ in range(400):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        centroid = 0.5 * (simplex[0] + simplex[1])
        worst = simplex[2]
        refl = centroid + (centroid - worst)
        f_refl = cost(refl)
        if f_refl < values[0]:
            expand = centroid + 2.0 * (centroid - worst)
            f_exp = cost(expand)
            simplex[2], values[2] = (expand, f_exp) if f_exp < f_refl else (refl, f_refl)
        elif f_refl < values[1]:
            simplex[2], values[2] = refl, f_refl
        else:
            contract = centroid + 0.5 * (worst - centroid)
            f_con = cost(contract)
            if f_con < values[2]:
                simplex[2], values[2] = contract, f_con
            else:
                for i in (1, 2):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = cost(simplex[i])
        spread = max(abs(simplex[i][0] - simplex[0][0]) + abs(simplex[i][1] - simplex[0][1])
                     for i in (1, 2))
        if spread < 1e-16 * max(abs(simplex[0][0]), abs(simplex[0][1])):
            break
    best = simplex[int(np.argmin(values))]
    return float(best[0]), float(best[1])


def deflection_of(
    u_ice: Sequence[float], reference: Sequence[float]
) -> float:
    """Deflection of ``u_ice`` from ``reference`` in degrees, positive to the right."""
    return deflection_angle(u_ice, reference)

# ---------------------------------------------------------------------------
# Event-response estimator (piecewise-constant forcing)
# ---------------------------------------------------------------------------

def _event_layout(times: np.ndarray, event_times: Sequence[float]):
    """Segment masks / elapsed-time arrays of a piecewise-constant-forcing record."""
    ev = np.sort(np.asarray(event_times, dtype=float))
    bounds = np.concatenate([[0.0], ev[ev < times[-1]], [times[-1] + 1.0]])
    seg_of = np.searchsorted(bounds, times, side="right") - 1
    seg_of = np.clip(seg_of, 0, bounds.size - 2)
    masks = [np.flatnonzero(seg_of == m) for m in range(bounds.size - 1)]
    taus = [times[mask] - bounds[m] for m, mask in enumerate(masks)]
    durs = np.diff(bounds)
    return bounds, masks, taus, durs


def _event_design(
    layout, gamma: float, omega: float, n_seg: int, include_initial: bool = True
) -> np.ndarray:
    """Design matrix ``(n_t, 2, n_col)`` of the piecewise-constant-target response.

    The target velocity ``p_j`` is piecewise constant on the segments delimited
    by the event times; the exact solution of ``du/dt = A u + (p_j - u)/inf``
    with ``A = -gamma I - omega K`` is ``u = p_j + exp(A tau) (v_j - p_j)`` and
    ``v_{j+1} = p_j + exp(A h_j)(v_j - p_j)``, hence ``u(t)`` is *affine* in
    ``(u_0, p_0, ..., p_{J-1})``.  One column per basis state (2 per segment,
    plus 2 for ``u_0``).
    """
    bounds, masks, taus, durs = layout
    n_t = sum(mask.size for mask in masks)
    n_col = 2 * n_seg + (2 if include_initial else 0)
    design = np.zeros((n_t, 2, n_col))
    v = np.zeros((n_col, 2))
    if include_initial:
        # the last two columns are the response to a unit initial velocity
        v[2 * n_seg, 0] = 1.0
        v[2 * n_seg + 1, 1] = 1.0
    for m, mask in enumerate(masks):
        if mask.size == 0:
            continue
        P = np.zeros((n_col, 2))
        if m < n_seg:
            P[2 * m, 0] = 1.0
            P[2 * m + 1, 1] = 1.0
        w = v - P
        tau = taus[m]
        decay = np.exp(-gamma * tau)
        c = np.cos(omega * tau)[:, None]
        s = np.sin(omega * tau)[:, None]
        ux = decay[:, None] * (c * w[None, :, 0] + s * w[None, :, 1]) + P[None, :, 0]
        uy = decay[:, None] * (-s * w[None, :, 0] + c * w[None, :, 1]) + P[None, :, 1]
        design[mask, 0, :] = ux
        design[mask, 1, :] = uy
        dur = durs[m]
        dec_d = math.exp(-gamma * dur)
        cd, sd = math.cos(omega * dur), math.sin(omega * dur)
        vnew = np.empty_like(v)
        vnew[:, 0] = dec_d * (cd * w[:, 0] + sd * w[:, 1]) + P[:, 0]
        vnew[:, 1] = dec_d * (-sd * w[:, 0] + cd * w[:, 1]) + P[:, 1]
        v = vnew
    return design


def estimate_event_response(
    times: Sequence[float],
    series: Sequence[Sequence[float]],
    event_times: Sequence[float],
    f_hint: Optional[float] = None,
    gamma_hint: Optional[float] = None,
    n_grid: int = 7,
    fixed_omega: Optional[float] = None,
    window_after_event: Optional[float] = None,
) -> Dict[str, float]:
    """Effective ``(gamma, omega)`` of a piecewise-forced linear response.

    Identifies the *effective* constant-coefficient response
    ``du/dt = -gamma u - omega k x u + (target - u)/0`` (i.e. the exact
    piecewise-constant-target solution of :class:`DampedInertialSolution`) by
    separable least squares: the segment target velocities are linear
    parameters, ``(gamma, omega)`` are found by a scan plus a Nelder-Mead polish.

    This is the estimator used for the stochastic ensemble: it does not assume
    the record is a single decaying transient (wind switches re-excite the
    oscillation every ``wind_switch_period_days``) and it is unbiased for a
    linear system with additive noise in the *observations*.
    """
    t = np.asarray(times, dtype=float)
    u = np.asarray(series, dtype=float)
    if u.shape[0] != t.size:
        raise ValueError("times and series length differ")
    ev = np.asarray(event_times, dtype=float)
    omega0 = abs(f_hint) if f_hint is not None else 1.4e-4
    gamma0 = gamma_hint if gamma_hint else 1.0e-4
    if window_after_event is not None:
        # keep only the first ``window_after_event`` seconds of every segment:
        # there the target velocity really is constant, which sharpens the
        # identification (long segments accumulate thickness / concentration
        # evolution that the piecewise-constant-target model cannot represent).
        bounds_all = np.concatenate([[0.0], ev[ev < t[-1]], [t[-1] + 1.0]])
        seg_all = np.clip(np.searchsorted(bounds_all, t, side="right") - 1, 0,
                          bounds_all.size - 2)
        tau_all = t - bounds_all[seg_all]
        keep = tau_all <= window_after_event
        t = t[keep]
        u = u[keep]
    layout = _event_layout(t, ev)
    n_seg = len(layout[1])
    b = u.reshape(t.size * 2)

    if fixed_omega is not None:
        # one-parameter identification: the oscillation frequency is the inertial
        # frequency (known from the physics), only the damping is unknown.  The
        # two-parameter fit is ill conditioned when gamma ~ |f| (the record then
        # contains less than one oscillation), see the report.
        omega_fixed = abs(float(fixed_omega))

        def cost_fixed(p) -> float:
            gam = float(p[0])
            if gam < 0.0:
                return 1e30
            design = _event_design(layout, gam, omega_fixed, n_seg)
            D = design.reshape(t.size * 2, -1)
            coef, *_ = np.linalg.lstsq(D, b, rcond=None)
            r = b - D @ coef
            return float(r @ r)

        grid = gamma0 * np.geomspace(0.05, 40.0, 61)
        vals = np.array([cost_fixed((g,)) for g in grid])
        best = grid[int(np.argmin(vals))]
        # golden-section refinement in log space
        lo, hi = best / 1.5, best * 1.5
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        for _ in range(80):
            c1 = hi - gr * (hi - lo)
            c2 = lo + gr * (hi - lo)
            if cost_fixed((c1,)) < cost_fixed((c2,)):
                hi = c2
            else:
                lo = c1
        gam = 0.5 * (lo + hi)
        design = _event_design(layout, gam, omega_fixed, n_seg)
        D = design.reshape(t.size * 2, -1)
        coef, *_ = np.linalg.lstsq(D, b, rcond=None)
        resid = b - D @ coef
        scale = max(float(np.max(np.linalg.norm(u, axis=1))), 1e-300)
        targets = coef[: 2 * n_seg].reshape(n_seg, 2)
        return {
            "gamma": gam,
            "omega": omega_fixed,
            "f": omega_fixed,
            "period": 2.0 * math.pi / omega_fixed,
            "targets": targets,
            "u_target_final": targets[-1],
            "fit_residual_rel": float(np.max(np.abs(resid))) / scale,
            "n_segments": n_seg,
            "method": "event_response_fixed_omega",
        }


    def cost(p) -> float:
        gam, om = float(p[0]), float(p[1])
        if gam < 0.0 or om <= 0.0:
            return 1e30
        design = _event_design(layout, gam, om, n_seg)
        D = design.reshape(t.size * 2, -1)
        b = u.reshape(t.size * 2)
        coef, *_ = np.linalg.lstsq(D, b, rcond=None)
        r = b - D @ coef
        return float(r @ r)

    best = None
    for gam in gamma0 * np.linspace(0.25, 4.0, n_grid):
        for om in omega0 * np.linspace(0.6, 1.4, n_grid):
            c = cost((gam, om))
            if best is None or c < best[0]:
                best = (c, gam, om)
    _, gam, om = best
    simplex = np.array([[gam, om], [gam * 1.05, om], [gam, om * 1.01]])
    values = np.array([cost(p) for p in simplex])
    for _ in range(300):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        centroid = 0.5 * (simplex[0] + simplex[1])
        refl = centroid + (centroid - simplex[2])
        f_refl = cost(refl)
        if f_refl < values[0]:
            expand = centroid + 2.0 * (centroid - simplex[2])
            f_exp = cost(expand)
            simplex[2], values[2] = (expand, f_exp) if f_exp < f_refl else (refl, f_refl)
        elif f_refl < values[1]:
            simplex[2], values[2] = refl, f_refl
        else:
            contract = centroid + 0.5 * (simplex[2] - centroid)
            f_con = cost(contract)
            if f_con < values[2]:
                simplex[2], values[2] = contract, f_con
            else:
                for i in (1, 2):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = cost(simplex[i])
        spread = float(np.max(np.abs(simplex[1:] - simplex[0])))
        if spread < 1e-15 * max(abs(simplex[0, 0]), abs(simplex[0, 1])):
            break
    ibest = int(np.argmin(values))
    gam, om = float(simplex[ibest, 0]), float(simplex[ibest, 1])
    design = _event_design(layout, gam, om, n_seg)
    D = design.reshape(t.size * 2, -1)
    coef, *_ = np.linalg.lstsq(D, b, rcond=None)
    resid = b - D @ coef
    scale = max(float(np.max(np.linalg.norm(u, axis=1))), 1e-300)
    targets = coef[: 2 * n_seg].reshape(n_seg, 2)
    return {
        "gamma": gam,
        "omega": om,
        "f": om,
        "period": 2.0 * math.pi / om,
        "targets": targets,
        "u_target_final": targets[-1],
        "fit_residual_rel": float(np.max(np.abs(resid))) / scale,
        "n_segments": n_seg,
        "method": "event_response",
    }
