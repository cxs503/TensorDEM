"""Minimal single-contact scenes used by the rolling-resistance cross-validation.

Each scene is a *closed* dynamical system whose reference value can be written
down analytically, so a disagreement is either an integrator error or a law
difference — never a case--specific calibration.

Scene 1 ``roll_on_plane`` — one sphere rolling on a rigid plane.  With the
plane at ``z = 0`` and gravity tilted by ``slope`` about ``y``, the along-plane
drive is ``g sin(slope)`` and the normal load is ``N = m g cos(slope)``.  The
contact carries a Cundall-Strack tangential spring (capped at ``mu_s N``) and
the rolling law under test.  Eliminating the tangential force with the rolling
constraint ``v_x = omega_y R``:

.. math:: a = \\frac{5}{7}\\left(\\frac{M_r}{mR} + g \\sin\\theta\\right),

so a saturating rolling couple ``|M_r| = mu_r R N`` decelerates a freely
rolling sphere at ``(5/7) mu_r g cos(theta)``.

Scene 2 ``pair_spin_relax`` — two equal spheres held in contact with a
prescribed normal force ``N`` and opposite spins.  Only the rotational degrees
of freedom evolve, ``I d w_i/dt = M``, and the relative spin decays at
``2 |M| / I``.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["pair_spin_relax", "roll_on_plane", "slice_theory_deceleration"]

_UP = np.array([[0.0, 0.0, 1.0]])


def slice_theory_deceleration(mu_r: float, gravity: float, slope_deg: float) -> float:
    """``a = (5/7) mu_r g cos(theta)`` — magnitude of the rolling deceleration."""
    return (5.0 / 7.0) * float(mu_r) * float(gravity) * math.cos(math.radians(slope_deg))


def roll_on_plane(model, *, radius: float = 0.03, density: float = 2500.0,
                  kn: float = 6.0e4, kt_ratio: float = 0.5, mu_s: float = 0.5,
                  mu_r: float = 0.5, gravity: float = 9.81, dt: float = 2.0e-6,
                  v0: float = 0.30, slope_deg: float = 0.0,
                  t_max: float = 2.0, reff: float | None = None,
                  sample_every: int = 100) -> dict:
    """Integrate one sphere on a rigid plane with the given rolling law.

    ``model`` is any object with ``reset(n)`` / ``step(spin_rel, fn, reff, kt,
    mu_r, dt, normal=None)`` (see :mod:`tensordem.xval.oss_rolling`).
    ``reff`` is the lever arm handed to the law (default: the particle radius,
    which is what the TensorDEM kernel uses).  Returns the history plus the
    stopping time/distance of the along-plane motion.
    """
    m = density * 4.0 / 3.0 * math.pi * radius ** 3
    inertia = 0.4 * m * radius ** 2
    kt = kt_ratio * kn
    reff = float(radius if reff is None else reff)
    theta = math.radians(slope_deg)
    drive = gravity * math.sin(theta)
    n_load = m * gravity * math.cos(theta)

    model.reset(1)
    vx = float(v0)
    wy = float(v0) / radius          # pure rolling start: v = omega R
    x = 0.0
    xi = 0.0                          # tangential spring (x component)
    t = 0.0
    hist = {"t": [], "v": [], "omega": [], "x": [], "dissipation": [],
            "energy": [], "f_t": [], "roll_torque": []}
    t_stop = None
    x_stop = None
    step = 0
    while t < t_max:
        spin = np.array([[0.0, wy, 0.0]])
        out = model.step(spin, np.array([n_load]), reff, kt, mu_r, dt,
                         normal=_UP)
        torque_y = float(out["torque"][0, 1])
        # tangential spring on the contact point (v_x + (omega x r)_x = v_x - wy R)
        slip = vx - wy * radius + drive * 0.0
        xi_new = xi + slip * dt
        f_t = -kt * xi_new
        cap = mu_s * n_load
        if abs(f_t) > cap:                      # Coulomb sliding
            f_t = math.copysign(cap, f_t)
            xi_new = -f_t / kt
        xi = xi_new
        # semi-implicit Euler on (v_x, omega_y)
        vx += dt * (f_t / m + drive)
        wy += dt * (-radius * f_t / inertia + torque_y / inertia)
        x += dt * vx
        t += dt
        step += 1
        if t_stop is None and vx <= 0.0:
            t_stop, x_stop = t, x
        if step % sample_every == 0:
            hist["t"].append(t)
            hist["v"].append(vx)
            hist["omega"].append(wy)
            hist["x"].append(x)
            hist["dissipation"].append(out["dissipation"])
            hist["energy"].append(out["energy"])
            hist["f_t"].append(f_t)
            hist["roll_torque"].append(torque_y)
        if t_stop is not None and slope_deg == 0.0:
            break
    return {"history": {k: np.asarray(v, dtype=float) for k, v in hist.items()},
            "t_stop": t_stop, "x_stop": x_stop,
            "mass": m, "inertia": inertia, "normal_load": n_load,
            "kt": kt, "reff": reff, "model": getattr(model, "name", type(model).__name__)}


def pair_spin_relax(model, *, radius: float = 0.03, density: float = 2500.0,
                    kn: float = 6.0e4, kt_ratio: float = 0.5, mu_r: float = 0.5,
                    w0: float = 20.0, fn: float = 1.0, dt: float = 2.0e-6,
                    t_max: float = 0.5, reff: float | None = None,
                    sample_every: int = 100) -> dict:
    """Two equal spheres in contact, opposite spins: relative-spin relaxation.

    ``w0`` is the *relative* spin (``|w_i - w_j|``); each sphere carries
    ``w0/2``.  The decay rate of a saturating couple is ``2 mu_r reff Fn / I``.
    """
    m = density * 4.0 / 3.0 * math.pi * radius ** 3
    inertia = 0.4 * m * radius ** 2
    kt = kt_ratio * kn
    reff = float(radius if reff is None else reff)
    model.reset(1)
    wi = 0.5 * float(w0)
    wj = -0.5 * float(w0)
    t = 0.0
    hist = {"t": [], "w_rel": [], "wi": [], "torque": [], "dissipation": []}
    step = 0
    t_half = None
    while t < t_max:
        spin_rel = np.array([[0.0, wi - wj, 0.0]])
        out = model.step(spin_rel, np.array([fn]), reff, kt, mu_r, dt,
                         normal=_UP)
        torque_y = float(out["torque"][0, 1])
        wi += dt * torque_y / inertia
        wj -= dt * torque_y / inertia
        t += dt
        step += 1
        w_rel = wi - wj
        if t_half is None and w_rel <= 0.5 * w0:
            t_half = t
        if step % sample_every == 0:
            hist["t"].append(t)
            hist["w_rel"].append(w_rel)
            hist["wi"].append(wi)
            hist["torque"].append(torque_y)
            hist["dissipation"].append(out["dissipation"])
        if abs(w_rel) < 1e-12:
            break
    return {"history": {k: np.asarray(v, dtype=float) for k, v in hist.items()},
            "t_half": t_half, "w0": float(w0), "inertia": inertia,
            "reff": reff, "fn": fn,
            "model": getattr(model, "name", type(model).__name__),
            "theory_rate": 2.0 * mu_r * reff * fn / inertia}