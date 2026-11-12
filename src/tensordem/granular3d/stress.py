"""Three-dimensional Virial / Love-Weber stress: the measurement instrument.

Why this module exists
----------------------
The 2-D engine has a complete stress surface (:mod:`tensordem.granular2d.stress`:
per-contact tensors, plane fluxes, wall stress and the discrete stress-balance
identity).  In 3-D only two things existed, both local to a single consumer:

* ``SpherePacking.evaluate`` (``granular3d.jamming``) computes the **isotropic**
  pressure ``kn * sum(overlap * dist) / (3 V)`` and nothing else;
* ``elastic_limit_shear`` and ``bpm_calibration_feasibility`` each carry their
  own private ``l (x) f`` sum for the one component they need.

Every 3-D case that has recently failed on a *stress* number
(``CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT``: "does the shear stress scale with
the Bagnold number"; ``CASE-DEM-PLANE-SHEAR-MUIDI``: "does mu(I) collapse")
reads its ordinate out of a private sum like that, with no shared, independently
verified instrument behind it.  A sign error, a wrong branch vector or a missing
plane-crossing rule in one of those sums is indistinguishable from a physics
disagreement -- which is the failure mode this module removes.

Definitions
-----------
Everything is float64 NumPy and engine agnostic.  A contact is given by

* ``branch``  ``l = x_j - x_i``  (vector from particle ``i`` to particle ``j``),
* ``force``   the force **acting on particle i** from particle ``j``.

Sign convention: **tension positive, compression negative.**  For a compressed
pair ``force`` is antiparallel to ``branch``, so

.. math:: \\sigma = \\frac{1}{V}\\sum_c l \\otimes f_c

has negative eigenvalues and the scalar pressure is ``p = -tr(sigma)/3 > 0``.

Four reductions of the same per-contact data are provided because they answer
different questions and must agree with each other:

``stress_tensor``
    Volume average ``(1/V) sum l (x) f`` over a periodic or wall-bounded cell.
``plane_traction`` / ``plane_stress`` / ``plane_profile``
    Momentum flux across a plane: only contacts that *cross* the plane
    contribute.  The depth profile ``sigma_zz(z)`` is obtained without binning
    particles, so no particle is double counted and no bin-width bias enters.
``wall_stress``
    Traction transmitted by a flat wall to the particles:
    ``sigma_{ab} = (1/A) sum f_a n_b`` with ``n`` the outward normal of the
    *particle* domain at that wall (for a floor, ``n = -z_hat``).
``moment_residual`` / ``region_balance``
    The discrete counterparts of ``div sigma = 0``:

    .. math:: \\sum_c l\\otimes f_c = -\\sum_p x_p \\otimes F_p^{ext},
              \\qquad
              \\sum_{p\\in R} F_p^{ext} = -\\sum_{c\\,\\mathrm{cut}} f_{p\\in R}

    Both are *rearrangements of the per-particle equilibrium conditions*, i.e.
    exact for a relaxed configuration.  Any non-zero residual is therefore an
    implementation error, not a relaxation error.  (The minus sign in the first
    identity is the price of ``f`` being the force on ``i`` while ``l`` points
    from ``i`` to ``j``; it is what makes a compressed pair give
    ``sigma_zz < 0``, and it is pinned by :func:`pinned_sign_check`.)

Scope and limitations (stated, not implied)
-------------------------------------------
* The contact virial carries **no kinetic contribution**.  A separate
  :func:`kinetic_stress` is provided so that a caller cannot silently mix the
  two: negligible for quasi-static states, dominant for a granular gas.
* Only central contact forces give a symmetric tensor.  Non-central parts
  (rolling / drilling couples, e.g. :mod:`tensordem.clump_rolling`) are not
  representable as a symmetric stress and are outside this contract; the raw
  (unsymmetrised) sum is returned so that ``sigma - sigma.T`` measures them.
* No adapter to a particular solver is provided on purpose: the module works on
  arrays, so a benchmark cannot change the instrument underneath itself.
  :func:`from_pair_list` documents the required layout.
* A plane cut determines only the row ``sigma[axis, :]``; that is a property of
  a plane, not a defect of the implementation.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "contact_outer",
    "from_pair_list",
    "stress_tensor",
    "pressure",
    "deviatoric",
    "kinetic_stress",
    "plane_traction",
    "plane_stress",
    "plane_profile",
    "wall_stress",
    "moment_residual",
    "region_balance",
]

_TOL = 1e-12


# --------------------------------------------------------------------- helpers
def _as_vectors(a, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (m, 3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _check_pair(branch, force) -> tuple[np.ndarray, np.ndarray]:
    l = _as_vectors(branch, "branch")
    f = _as_vectors(force, "force")
    if l.shape != f.shape:
        raise ValueError(f"branch {l.shape} and force {f.shape} must match")
    return l, f


def _positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return v


# ----------------------------------------------------------------- per contact
def contact_outer(branch, force_on_i) -> np.ndarray:
    """Per-contact tensors ``l (x) f`` with shape ``(m, 3, 3)``."""
    l, f = _check_pair(branch, force_on_i)
    return l[:, :, None] * f[:, None, :]


def from_pair_list(positions, i, j, force_on_i, *, box: float | None = None):
    """Build ``(branch, force)`` from a pair list.

    ``positions`` shape ``(N, 3)``; ``i``, ``j`` integer index arrays; ``box`` the
    periodic edge length (``None`` for open boundaries).  The minimum-image
    convention matches ``granular3d.jamming.SpherePacking``.
    """
    pos = _as_vectors(positions, "positions")
    ii = np.asarray(i, dtype=np.int64).ravel()
    jj = np.asarray(j, dtype=np.int64).ravel()
    if ii.shape != jj.shape:
        raise ValueError("i and j must have the same length")
    n = pos.shape[0]
    if ii.size and (ii.min() < 0 or jj.min() < 0 or ii.max() >= n or jj.max() >= n):
        raise IndexError("pair indices out of range")
    branch = pos[jj] - pos[ii]
    if box is not None:
        box = _positive(box, "box")
        branch = branch - box * np.round(branch / box)
    force = _as_vectors(force_on_i, "force_on_i")
    if force.shape[0] != ii.size:
        raise ValueError("force_on_i must have one row per contact")
    return branch, force


# -------------------------------------------------------------- volume average
def stress_tensor(branch, force_on_i, volume: float) -> np.ndarray:
    """Volume-averaged Cauchy stress ``(1/V) sum_c l (x) f``, shape ``(3, 3)``.

    Tension positive, compression negative; unsymmetrised (see module notes).
    """
    volume = _positive(volume, "volume")
    return contact_outer(branch, force_on_i).sum(axis=0) / volume


def pressure(sigma) -> float:
    """Scalar pressure ``p = -tr(sigma)/3`` (positive under compression)."""
    s = np.asarray(sigma, dtype=float)
    if s.shape != (3, 3):
        raise ValueError("sigma must be 3x3")
    return float(-np.trace(s) / 3.0)


def deviatoric(sigma) -> np.ndarray:
    """Deviatoric part ``sigma + p I`` (zero trace)."""
    s = np.asarray(sigma, dtype=float)
    if s.shape != (3, 3):
        raise ValueError("sigma must be 3x3")
    return s - (np.trace(s) / 3.0) * np.eye(3)


def kinetic_stress(mass, velocity, volume: float) -> np.ndarray:
    """Kinetic (Reynolds) stress ``(1/V) sum_p m v (x) v``."""
    volume = _positive(volume, "volume")
    m = np.asarray(mass, dtype=float).ravel()
    v = _as_vectors(velocity, "velocity")
    if m.shape[0] != v.shape[0]:
        raise ValueError("mass and velocity must have one row per particle")
    return (v * m[:, None]).T @ v / volume


def pinned_sign_check(branch, force_on_i, volume: float) -> float:
    """``sigma_zz`` of a single compressed pair -- the sign-convention pin.

    For ``branch = d z_hat`` and ``force_on_i = -F z_hat`` the convention of this
    module requires ``sigma_zz = -F d / V`` (compression negative).  Kept as a
    named function so the pin is callable from any new consumer.
    """
    s = stress_tensor(branch, force_on_i, volume)
    return float(s[2, 2])


# ----------------------------------------------------------------- plane / wall
def plane_traction(branch, force_on_i, center_i, *, axis: int, plane: float):
    """Traction (a force) transmitted across ``x_axis = plane``.

    Only contacts that *cross* the plane contribute.  The returned vector is the
    force that the ``x_axis > plane`` side exerts on the ``x_axis < plane`` side,
    assembled by summing the forces acting on the particles below the plane.
    ``center_i`` is the position of particle ``i`` of each contact (particle
    ``j`` sits at ``center_i + branch``).
    """
    l, f = _check_pair(branch, force_on_i)
    c = _as_vectors(center_i, "center_i")
    if c.shape != l.shape:
        raise ValueError("center_i must have one row per contact")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    below_i = c[:, axis] < plane
    below_j = (c[:, axis] + l[:, axis]) < plane
    force_on_lower = (np.where(below_i[:, None], f, 0.0)
                      + np.where(below_j[:, None], -f, 0.0))
    crossing = (below_i != below_j) & (np.abs(l[:, axis]) > _TOL)
    return force_on_lower[crossing].sum(axis=0), int(np.count_nonzero(crossing))


def plane_stress(branch, force_on_i, center_i, *, axis: int, plane: float,
                 area: float) -> np.ndarray:
    """Stress with the ``axis`` row fixed by a plane cut (see module notes)."""
    area = _positive(area, "area")
    traction, _ = plane_traction(branch, force_on_i, center_i, axis=axis, plane=plane)
    out = np.zeros((3, 3))
    out[int(axis), :] = traction / area
    return out


def plane_profile(branch, force_on_i, center_i, *, axis: int, planes, area: float):
    """Depth profile of the plane flux.

    Returns ``(normal_stress, shear, counts)``: the normal stress
    ``sigma_{axis,axis}``, the two in-plane components of the traction, and the
    number of crossing contacts, per plane.  Exact per plane (no bin width).
    """
    area = _positive(area, "area")
    axes = np.asarray(planes, dtype=float).ravel()
    if not np.isfinite(axes).all():
        raise ValueError("planes must be finite")
    normal = np.zeros(axes.size)
    shear = np.zeros((axes.size, 2))
    counts = np.zeros(axes.size, dtype=np.int64)
    others = [a for a in (0, 1, 2) if a != int(axis)]
    for k, z in enumerate(axes):
        trac, n = plane_traction(branch, force_on_i, center_i,
                                 axis=int(axis), plane=float(z))
        normal[k] = trac[int(axis)] / area
        shear[k] = trac[others] / area
        counts[k] = n
    return normal, shear, counts


def wall_stress(force_on_particles, area: float, *, axis: int,
                normal_sign: int = -1) -> np.ndarray:
    """Stress transmitted by a flat wall: ``sigma_{ab} = (1/A) sum f_a n_b``.

    ``force_on_particles`` holds the wall forces acting on the particles (one
    row per wall contact).  ``normal_sign`` is the sign of the outward normal of
    the *particle* domain: for a floor with the particles above it,
    ``normal_sign = -1``, which is what makes the floor reaction give
    ``sigma_zz < 0``.
    """
    area = _positive(area, "area")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if normal_sign not in (-1, 1):
        raise ValueError("normal_sign must be +1 or -1")
    f = _as_vectors(force_on_particles, "force_on_particles")
    n = np.zeros(3)
    n[axis] = float(normal_sign)
    return (f.sum(axis=0)[:, None] * n[None, :]) / area


# ------------------------------------------------------- discrete balance laws
def moment_residual(positions, branch, force_on_i, external_forces, *,
                    normalise: bool = True) -> float:
    """Residual of ``sum_c l (x) f = sum_p x_p (x) F_p``.

    Exact (to round-off) for any force-balanced configuration: it is the sum of
    the per-particle equilibrium conditions multiplied by ``x_p``.  The
    normalised residual uses Frobenius norms.
    """
    pos = _as_vectors(positions, "positions")
    fext = _as_vectors(external_forces, "external_forces")
    if pos.shape != fext.shape:
        raise ValueError("positions and external_forces must have equal shape")
    left = contact_outer(branch, force_on_i).sum(axis=0)
    right = pos.T @ fext
    num = float(np.linalg.norm(left - right))
    if not normalise:
        return num
    den = float(np.linalg.norm(right))
    return num / den if den > 0.0 else num


def region_balance(positions, i, j, force_on_i, external_forces, mask, *,
                   box: float | None = None):
    """Force balance of a sub-region: ``sum_R F_ext + sum_cut f_on_R = 0``.

    Returns ``(net_external, boundary_force, residual)`` where ``boundary_force``
    contains the contact forces acting on the region across its boundary and
    ``residual`` is the relative imbalance ``|net_external +
    boundary_force| / |net_external|``.  For a relaxed region this is round-off,
    and ``-net_external`` is exactly the traction a plane cut must transmit.
    """
    pos = _as_vectors(positions, "positions")
    fext = _as_vectors(external_forces, "external_forces")
    if pos.shape != fext.shape:
        raise ValueError("positions and external_forces must have equal shape")
    m = np.asarray(mask, dtype=bool).ravel()
    if m.shape[0] != pos.shape[0]:
        raise ValueError("mask must have one entry per particle")
    branch, f = from_pair_list(pos, i, j, force_on_i, box=box)
    ii = np.asarray(i, dtype=np.int64).ravel()
    jj = np.asarray(j, dtype=np.int64).ravel()
    n_contacts = branch.shape[0]
    inside_i = m[ii]
    inside_j = m[jj]
    if n_contacts:
        boundary = (np.where(inside_i[:, None], f, 0.0)
                    + np.where(inside_j[:, None], -f, 0.0))
        boundary = boundary[inside_i != inside_j].sum(axis=0)
    else:
        boundary = np.zeros(3)
    net_ext = fext[m].sum(axis=0)
    num = float(np.linalg.norm(net_ext + boundary))
    den = float(np.linalg.norm(net_ext))
    return net_ext, boundary, (num / den if den > 0.0 else num)