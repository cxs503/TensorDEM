"""Time integrators for the superquadric DEM: symplectic Euler and velocity-Verlet.

Campaign node **P165-W2-N06** (TensorDEM).  The parent maturity node integrated
the DEM with semi-implicit (symplectic) Euler, which is first order: the fitted
damping/decay rate of a damped system carries an ``O(dt)`` error (the documented
~5 % of ``docs/dem_contact_mechanics_verification_report.md``, C3/limitation 4)
and a conservative system shows an ``O(dt)`` energy offset.

This module supplies the second-order scheme requested by the node,
**velocity-Verlet with a contact-force half-step position correction**
(kick-drift-kick)::

    v_half = v0 + dt/2 a(x0, v0)          # kick
    x1     = x0 + dt v_half               # drift (half-step velocity -> position)
    a1     = a(x1, v_pred)                # force at the DRIFTED position
    v1     = v_half + dt/2 a1             # kick

with ``v_pred = v_half`` (bare KDK) or ``v_pred = v_half + dt/2 a0 = v0 + dt a0``
(``velocity_predictor=True``, the default).  The predictor matters only for
velocity-dependent forces (dashpots, viscous drag): KDK freezes the velocity over
the kick and leaves a residual ``O(dt)`` error in the decay *rate*, while the
predicted velocity makes the velocity-dependent term midpoint-accurate, i.e.
``O(dt^2)`` in the rate.  Both variants are implemented so the difference can be
measured instead of asserted (see ``tests/test_dem_superquadric_maturity_p165.py``).

The step functions are pure (no global state) so they can be verified against the
analytic solution of a harmonic/Duffing oscillator independently of the contact
machinery, and so :class:`tensordem.superquadric_dem.SQSystem` can reuse exactly
the code that the convergence study measures.

All quantities are SI.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "symplectic_euler_step",
    "velocity_verlet_step",
    "INTEGRATORS",
    "step",
    "integrate",
    "harmonic_accel",
    "duffing_accel",
    "order_from_errors",
]

Array = np.ndarray
AccelFn = Callable[[Array, Array], Array]


def symplectic_euler_step(
    x: Array, v: Array, dt: float, accel: AccelFn,
) -> Tuple[Array, Array, Array]:
    """Semi-implicit (symplectic) Euler: ``v1 = v0 + dt a(x0,v0)``, ``x1 = x0 + dt v1``.

    First order; the legacy integrator of the parent node.  Returns
    ``(x1, v1, a0)``.
    """
    a0 = accel(x, v)
    v1 = v + dt * a0
    x1 = x + dt * v1
    return x1, v1, a0


def velocity_verlet_step(
    x: Array, v: Array, dt: float, accel: AccelFn, velocity_predictor: bool = True,
) -> Tuple[Array, Array, Array]:
    """Velocity-Verlet (kick-drift-kick) step, second order.

    Returns ``(x1, v1, a1)`` where ``a1`` is the acceleration at the new state
    (reused by the caller to avoid a third force assembly).
    """
    a0 = accel(x, v)
    v_half = v + 0.5 * dt * a0
    x1 = x + dt * v_half
    v_pred = v_half + 0.5 * dt * a0 if velocity_predictor else v_half
    a1 = accel(x1, v_pred)
    v1 = v_half + 0.5 * dt * a1
    return x1, v1, a1


INTEGRATORS = {
    "euler": symplectic_euler_step,
    "symplectic_euler": symplectic_euler_step,
    "velocity_verlet": velocity_verlet_step,
}


def step(name: str, x: Array, v: Array, dt: float, accel: AccelFn,
         **kwargs) -> Tuple[Array, Array, Array]:
    """Dispatch on the integrator name (``euler`` or ``velocity_verlet``)."""
    try:
        fn = INTEGRATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown integrator {name!r}; known: {sorted(INTEGRATORS)}") from None
    return fn(x, v, dt, accel, **kwargs)


def integrate(name: str, accel: AccelFn, x0: Sequence[float], v0: Sequence[float],
              dt: float, n_steps: int, **kwargs) -> Tuple[Array, Array]:
    """Integrate ``x'' = accel(x, v)`` for ``n_steps`` and return the final state."""
    x = np.asarray(x0, dtype=float)
    v = np.asarray(v0, dtype=float)
    for _ in range(n_steps):
        x, v, _ = step(name, x, v, dt, accel, **kwargs)
    return x, v


# ---------------------------------------------------------------------------
# Analytic reference problems used by the order / drift studies
# ---------------------------------------------------------------------------

def harmonic_accel(omega: float = 1.0, gamma: float = 0.0) -> AccelFn:
    """``x'' = -omega^2 x - gamma x'`` (exact frequency ``omega``, decay ``gamma``)."""
    def accel(x: Array, v: Array) -> Array:
        return -omega ** 2 * x - gamma * v
    return accel


def duffing_accel(omega: float = 1.0, beta: float = 2.0) -> AccelFn:
    """``x'' = -omega^2 x - beta x^3`` (nonlinear conservative oscillator)."""
    def accel(x: Array, v: Array) -> Array:
        return -omega ** 2 * x - beta * x ** 3
    return accel


def order_from_errors(dts: Sequence[float], errs: Sequence[float]) -> float:
    """Fitted order ``p`` of ``err ~ dt^p`` (log-log least squares)."""
    d = np.asarray(list(dts), dtype=float)
    e = np.maximum(np.asarray(list(errs), dtype=float), 1e-300)
    keep = e > 1e-14
    if int(np.sum(keep)) < 2:
        return float("nan")
    return float(np.polyfit(np.log(d[keep]), np.log(e[keep]), 1)[0])