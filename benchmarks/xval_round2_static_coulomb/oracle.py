"""Independent oracle for the round-2 static-Coulomb + COM-lever rolling law.

Reference description (read from the OSS artifact ``TensorDEM/self``, no code
copied into this module):

* ``docs/GAP_DEM_CLUMP_ROUND2.md`` §2.1 — the discrete Coulomb rolling law
  ``|tau| = min(mu_r R |Fn|, I_eff |w_rel| / dt)``, direction ``-w_rel``, with the
  one-step arrest speed ``w_eps = mu_r R |Fn| dt / I_eff``; the shape-aware arm
  ``R = |x_contact - x_com|``.
* ``src/tensordem/contact/rolling_static.py`` module docstring — same two
  statements, plus the note that ``|x - x_com| == r_member`` for a single sphere.

Everything below is re-derived from those *equations* only: this module
deliberately does **not** import ``tensordem`` so that a shared bug cannot hide
the difference (independent-implementation cross-check).
"""

from __future__ import annotations

import numpy as np

TINY = 1e-300


def coulomb_limit(mu_roll, lever_arm, normal_force):
    """``mu_r R |Fn|`` — the plastic (rate independent) rolling torque ceiling."""
    return mu_roll * np.asarray(lever_arm, float) * np.abs(np.asarray(normal_force, float))


def arrest_speed(limit, inertia_eff, dt):
    """``w_eps`` — the relative spin the ceiling can arrest inside one step."""
    return np.asarray(limit, float) * np.asarray(dt, float) / np.asarray(inertia_eff, float)


def torque(w_rel, limit, inertia_eff, dt):
    """Delivered torque on the *i* side of the contact (vector form)."""
    w = np.asarray(w_rel, float).reshape(-1, 3)
    lim = np.asarray(limit, float).reshape(-1)
    inert = np.asarray(inertia_eff, float).reshape(-1)
    w_mag = np.linalg.norm(w, axis=1)
    # below w_eps the law sticks: the ceiling and the arresting impulse coincide
    mag = np.minimum(lim, inert * w_mag / np.asarray(dt, float))
    unit = w / np.maximum(w_mag, TINY)[:, None]
    return -mag[:, None] * unit


def saturation(w_rel, limit, inertia_eff, dt):
    """Fraction of the Coulomb ceiling delivered, ``min(1, |w| / w_eps)``."""
    w_mag = np.linalg.norm(np.asarray(w_rel, float).reshape(-1, 3), axis=1)
    w_eps = arrest_speed(limit, inertia_eff, dt)
    return np.minimum(1.0, w_mag / np.maximum(w_eps, TINY))


def stop_residual(w_rel, limit, inertia_eff, dt):
    """Closed form of the spin left after one step: ``max(0, |w| - w_eps)``."""
    w_mag = np.linalg.norm(np.asarray(w_rel, float).reshape(-1, 3), axis=1)
    return np.maximum(0.0, w_mag - arrest_speed(limit, inertia_eff, dt))


def steps_to_arrest(w0, limit, inertia_eff, dt):
    """Plastic-decay step count: each step removes exactly ``w_eps``.

    ``ceil(w0 / w_eps)`` steps (>= 1): the last one is the arresting step.
    """
    w_eps = arrest_speed(limit, inertia_eff, dt)
    return np.maximum(1.0, np.ceil(np.asarray(w0, float) / np.maximum(w_eps, TINY)))


def lever_arm(contact_point, body_com, member_radius):
    """Shape-aware moment arm ``max(|x - x_com|, r_member)`` (COM lever)."""
    arm = np.linalg.norm(np.asarray(contact_point, float).reshape(-1, 3)
                         - np.asarray(body_com, float).reshape(-1, 3), axis=1)
    return np.maximum(arm, np.asarray(member_radius, float).reshape(-1))


def sphere_eps_omega(mu_roll, g, dt, r_eq, inertia_factor=0.4):
    """Dimensionless-free closed form of ``w_eps`` for a single sphere on a floor.

    ``Fn = m g``, ``I = 0.4 m r^2`` ⇒ the mass cancels:
    ``w_eps = mu_r g dt / (0.4 r)``.
    """
    return mu_roll * g * dt / (inertia_factor * r_eq)