"""Rate-independent (static-capable) and shape-aware rolling resistance.

Motivation (GAP-DEM-CLUMB round-2, see ``docs/GAP_DEM_CLUMP_ROUND2.md``)
------------------------------------------------------------------------
The legacy rolling branch in :mod:`tensordem.particle.clump_dem` is

    ``tau = -mu_roll * R * |Fn| * w_rel / (|w_rel| + eps_w)``      (``eps_w`` = 1 rad/s)

which is **rate proportional** below ``eps_w``: at ``|w| = 0.45 rad/s`` it
delivers only ~31 % of the Coulomb limit and it vanishes as ``|w| -> 0``.  A
rolling friction law built this way cannot *support* a body on a slope, so any
repose-angle trend measured with it cannot be attributed to rolling resistance
as the literature comparison assumes (round-2 diagnosis, ``docs/
GAP_DEM_CLUMP_REPOSE_PROTOCOL_V2.md`` §3).

Two independent defects are fixed here, both without introducing a tuning knob:

1. **Static branch (``model="coulomb"``).**  The discrete form of Coulomb
   rolling friction is the impulse that *stops* the relative rolling within the
   step, clipped by the Coulomb limit:

       ``|tau| = min( mu_roll R |Fn| ,  I_eff |w_rel| / dt )``,  ``tau || -w_rel``

   The clip is not a regularisation parameter: ``I_eff |w| / dt`` is exactly the
   angular impulse that arrests the relative rotation in one step, so for
   ``|w| > w_eps = mu_roll R |Fn| dt / I_eff`` the full Coulomb limit is
   delivered (rate independent, plastic) and below it the contact sticks
   (``w_rel -> 0`` in one step).  Forwards-compatible: with
   ``model="viscous"`` the legacy law is reproduced bit for bit, so round-1
   numbers stay reproducible.

2. **Shape-aware moment arm (``lever="com"``).**  The rolling torque of a clump
   is applied to the *body*, but its magnitude is still built from the *member*
   radius, i.e. a body-level isotropic torque.  That is precisely why round 1
   found the shape effect suppressed: a long clump receives the same specific
   rolling resistance as a sphere of the member radius.  Using the distance from
   the contact point to the body centre of mass

       ``R_roll = | x_contact - x_com |``

   is the geometric moment arm of the normal force about the body centre.  It is
   **exactly** the member radius for a single sphere (so the sphere baseline is
   unchanged, bit for bit), and it makes the resistance anisotropic for a
   clump: tip contacts (|x - x_com| ~ a) resist tumbling end-over-end much more
   than flank contacts (~ b), while rolling about the long axis is still resisted
   only with the ~ b arm.  No free parameter is introduced.

All functions are pure and unit-tested in ``tests/test_rolling_static.py``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

#: Supported rolling-law models.  ``viscous`` = legacy, ``coulomb`` = static-capable.
ROLLING_MODELS = ("viscous", "coulomb")
#: Supported moment arms.  ``member`` = legacy, ``com`` = shape aware.
ROLLING_LEVERS = ("member", "com")

_TINY = 1e-30


def check_model(model: str) -> str:
    if model not in ROLLING_MODELS:
        raise ValueError(f"rolling_model must be one of {ROLLING_MODELS}, got {model!r}")
    return model


def check_lever(lever: str) -> str:
    if lever not in ROLLING_LEVERS:
        raise ValueError(f"rolling_lever must be one of {ROLLING_LEVERS}, got {lever!r}")
    return lever


def rolling_lever_arm(contact_point, body_com, member_radius, lever: str = "member"):
    """Moment arm ``R_roll`` of the rolling resistance, per contact.

    Parameters
    ----------
    contact_point : (n, 3) array
        Contact location in the world frame.
    body_com : (n, 3) array
        Centre of mass of the *body* carrying the contact.
    member_radius : (n,) array
        Radius of the member sphere that owns the contact.
    lever : {"member", "com"}
        ``member`` reproduces the legacy law (arm = member radius);
        ``com`` uses the contact-to-COM distance (shape aware, identical to
        ``member`` for a single sphere).
    """
    check_lever(lever)
    r_mem = np.asarray(member_radius, dtype=float).reshape(-1)
    if lever == "member":
        return r_mem
    arm = np.linalg.norm(np.asarray(contact_point, dtype=float).reshape(-1, 3)
                         - np.asarray(body_com, dtype=float).reshape(-1, 3), axis=1)
    # A contact can only sit on the member surface, so |x - com| >= r_mem for a
    # single-sphere body (equality) and >= r_mem inside a clump; clip for safety
    # against round-off so the sphere case is recovered exactly.
    return np.maximum(arm, r_mem)


def rolling_limit(mu_roll, lever_arm, normal_force):
    """Coulomb rolling-resistance torque limit ``mu_roll * R * |Fn|`` (>= 0)."""
    return np.maximum(
        float(mu_roll) * np.asarray(lever_arm, dtype=float)
        * np.abs(np.asarray(normal_force, dtype=float)),
        0.0,
    )


def rolling_torque(w_rel, limit, inertia_eff, dt, *, model: str = "viscous",
                   eps_w: float = 1.0):
    """Rolling-resistance torque acting on the contact's *i* side.

    Parameters
    ----------
    w_rel : (n, 3) array
        Relative angular velocity (``w_i - w_j``; ``w_i`` for a fixed wall).
    limit : (n,) array
        Coulomb limit from :func:`rolling_limit`.
    inertia_eff : (n,) array
        Effective moment of inertia of the pair about the rolling axis,
        ``1 / (1/I_i + 1/I_j)`` (``I_i`` for a wall).
    dt : float
        Time step [s].
    model : {"viscous", "coulomb"}
        ``viscous`` = legacy law ``-limit * w/(|w| + eps_w)``;
        ``coulomb`` = static-capable law ``-min(limit, I_eff |w|/dt) * w/|w|``.
    eps_w : float
        Legacy regularisation [rad/s], only used by ``viscous``.

    Returns
    -------
    (n, 3) torque array.
    """
    check_model(model)
    w = np.asarray(w_rel, dtype=float).reshape(-1, 3)
    lim = np.asarray(limit, dtype=float).reshape(-1)
    inert = np.asarray(inertia_eff, dtype=float).reshape(-1)
    w_mag = np.linalg.norm(w, axis=1)
    if model == "viscous":
        scale = lim / (w_mag + float(eps_w))
    else:
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        stop = inert * w_mag / dt          # impulse that arrests the spin in one step
        scale = np.minimum(lim, stop) / np.maximum(w_mag, _TINY)
    return -scale[:, None] * w


def rolling_saturation(w_rel, limit, inertia_eff, dt, *, model: str = "viscous",
                       eps_w: float = 1.0):
    """Fraction of the Coulomb limit actually delivered, per contact.

    ``1`` means the contact is at the Coulomb limit (plastic rolling);
    ``< 1`` means it is sticking (``coulomb``) or under-regularised
    (``viscous``).
    """
    check_model(model)
    w = np.asarray(w_rel, dtype=float).reshape(-1, 3)
    lim = np.asarray(limit, dtype=float).reshape(-1)
    inert = np.asarray(inertia_eff, dtype=float).reshape(-1)
    w_mag = np.linalg.norm(w, axis=1)
    if model == "viscous":
        sat = w_mag / (w_mag + float(eps_w))
    else:
        stop = inert * w_mag / float(dt)
        sat = np.where(lim > 0.0,
                       np.minimum(1.0, stop / np.maximum(lim, _TINY)), 0.0)
    return sat


def stopping_speed(limit, inertia_eff, dt):
    """``w_eps``: relative spin arrested by the Coulomb limit within one step."""
    lim = np.asarray(limit, dtype=float)
    inert = np.asarray(inertia_eff, dtype=float)
    return lim * float(dt) / np.maximum(inert, _TINY)