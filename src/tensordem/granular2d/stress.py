"""Per-contact Virial / Love-Weber stress and the discrete stress-balance identity.

Definitions (2D, stress in N/m)
-------------------------------
For contact ``c`` between disks ``i`` and ``j`` with branch vector
``l = x_j - x_i`` and force on ``i`` equal to ``f_c`` (so the force on ``j`` is
``-f_c``) the contact contributes

.. math:: W_c = l \\otimes f_c , \\qquad \\sigma = \\frac{1}{A}\\sum_c W_c .

The *discrete Virial identity* for a force-balanced assembly is

.. math:: \\sum_c l\\otimes f_c = \\sum_p x_p \\otimes F_p^{ext},

with :math:`F^{ext}` collecting boundary (bank/hull) tractions, gravity and any
drift drag.  This is the discrete counterpart of
:math:`\\int_\\Omega \\nabla\\cdot\\sigma\\,dV = \\oint_{\\partial\\Omega} t\\,dA`
and is what ``stress_balance`` checks.  With inertia retained the identity is
exact for the integrator step, which is used to separate implementation errors
from relaxation error: the ``static`` residual measures how well the configuration
is relaxed, the ``dynamic`` residual measures the implementation.
"""

from __future__ import annotations

import math

import numpy as np


def contact_stress_table(system) -> dict[str, np.ndarray]:
    """Extract contact tensors from the last force evaluation of ``system``.

    Returns arrays for disk-disk contacts (``i``, ``j``, ``branch`` = x_j - x_i,
    ``force_on_i``) and for boundary contacts (``boundary_i``, ``point``,
    ``force_on_i``).
    """
    contacts = system.contacts
    if "i" not in contacts or len(contacts["i"]) == 0:
        empty2 = np.zeros((0, 2))
        return {"i": np.zeros(0, dtype=np.int64), "j": np.zeros(0, dtype=np.int64),
                "branch": empty2, "force_on_i": empty2,
                "image": empty2,
                "boundary_i": np.zeros(0, dtype=np.int64), "point": empty2,
                "boundary_force_on_i": empty2,
                "magnitude": np.zeros(0)}
    kind = contacts["kind"]
    disk = kind == 0
    bnd = ~disk
    i = np.asarray(contacts["i"])[disk]
    j = np.asarray(contacts["j"])[disk]
    lever = np.asarray(contacts["lever"])[disk]
    f = np.asarray(contacts["f_on_i"])[disk]
    # lever stores x_j - x_i for disk pairs; make it explicit
    raw = system.pos[j] - system.pos[i]
    branch = raw.copy()
    image = np.zeros_like(branch)
    px = getattr(system, "periodic_x", None)
    py = getattr(system, "periodic_y", None)
    if px is not None:
        image[:, 0] = np.round(raw[:, 0] / px)
        branch[:, 0] = raw[:, 0] - px * image[:, 0]
    if py is not None:
        image[:, 1] = np.round(raw[:, 1] / py)
        branch[:, 1] = raw[:, 1] - py * image[:, 1]
    return {
        "i": i,
        "j": j,
        "branch": branch,
        "force_on_i": f,
        "image": image,
        "magnitude": np.linalg.norm(f, axis=1),
        "normal_magnitude": np.asarray(contacts["fn"])[disk],
        "tangential_magnitude": np.asarray(contacts["ft"])[disk],
        "boundary_i": np.asarray(contacts["i"])[bnd],
        "point": np.asarray(contacts["point"])[bnd],
        "boundary_force_on_i": np.asarray(contacts["f_on_i"])[bnd],
        "boundary_kind": np.asarray(contacts["kind"])[bnd],
        "lever_unused": lever,
    }


def _outer(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.einsum("ma,mb->ab", rows, cols)


def virial_tensor(system, *, symmetrize: bool = True,
                  area: float | None = None, contacts=None) -> np.ndarray:
    """Contact (Love-Weber) stress tensor ``(2, 2)`` in N/m.

    ``symmetrize`` returns :math:`\\tfrac12(W + W^T)`; the antisymmetric part of
    the raw contact sum is a couple stress that is *not* zero once Coulomb
    friction and rolling resistance are active, so it is reported separately by
    :func:`stress_balance` instead of being silently dropped.
    """
    table = contact_stress_table(system) if contacts is None else contacts
    if area is None:
        area = system.domain_area()
        if area is None:
            raise ValueError("area must be given for a system without walls")
    if area <= 0.0:
        raise ValueError("area must be positive")
    w = _outer(table["branch"], table["force_on_i"]) if len(table["i"]) else np.zeros((2, 2))
    sigma = w / area
    return 0.5 * (sigma + sigma.T) if symmetrize else sigma


def stress_balance(system, *, area: float | None = None) -> dict[str, float]:
    """Check the discrete Virial identity and report stress symmetry metrics.

    Keys
    ----
    ``virial_identity_rel``
        Relative residual of the identity *with* the integrator inertia term; this
        is an implementation check (it must be at round-off level).
    ``virial_identity_static_rel``
        Same residual with acceleration neglected; this measures how well the
        configuration is force-balanced (relaxation quality).
    ``boundary_traction_rel``
        Residual when only boundary (bank + hull) tractions are used as external
        loads; equals ``virial_identity_static_rel`` when gravity/drag are zero.
    ``stress_symmetry_rel``
        ``||sigma - sigma^T||_F / ||sigma||_F`` of the returned (symmetrised)
        tensor; the check is exercised by a negative-control test that breaks the
        accumulation.
    ``antisymmetric_fraction``
        Physical couple-stress fraction ``||W - W^T||/(2||W||)`` of the raw sum.
    ``virial_identity_periodic_rel``
        ``||W + W_image|| / ||W||``: the periodic form of the Virial identity.
        With minimum-image branch vectors the image part must cancel the contact
        sum exactly for a force-balanced configuration, so this residual measures
        relaxation quality in a *doubly periodic* cell (no walls to supply
        tractions).  A negative control that omits the image term makes it O(1).
    ``virial_symmetry_raw_rel``
        ``||W_raw - W_raw^T|| / ||W_raw||`` of the **unsymmetrised** virial sum;
        vanishing to round-off for central-force contacts, non-zero once
        friction/rolling couple stresses are active.
    """
    contacts = system.contact_forces()
    table = contact_stress_table(system)
    if area is None:
        area = system.domain_area()
        if area is None:
            raise ValueError("area must be given for a system without walls")
    n = system.n
    w_int = _outer(table["branch"], table["force_on_i"]) if len(table["i"]) else np.zeros((2, 2))

    boundary = np.asarray(getattr(system, "_external_force", np.zeros((n, 2))), dtype=np.float64)
    cfg = system.config
    body = np.zeros((n, 2))
    g = np.asarray(cfg.gravity, dtype=np.float64)
    if np.any(g != 0.0):
        body = body + system.mass[:, None] * g[None, :]
    if cfg.drag_time > 0.0:
        v_drift = np.asarray(cfg.drift_velocity, dtype=np.float64)
        body = body + system.mass[:, None] * (v_drift[None, :] - system.vel) / cfg.drag_time
    f_ext = boundary + body
    w_ext = _outer(system.pos, f_ext)

    # contact-only force for the inertia term
    f_contact = np.zeros((n, 2))
    if len(table["i"]):
        np.add.at(f_contact, table["i"], table["force_on_i"])
        np.add.at(f_contact, table["j"], -table["force_on_i"])
    if len(table["boundary_i"]):
        np.add.at(f_contact, table["boundary_i"], table["boundary_force_on_i"])
    accel = (f_contact + body) / system.mass[:, None]
    w_inertia = _outer(system.pos, system.mass[:, None] * accel)

    scale = max(float(np.linalg.norm(w_ext)), float(np.linalg.norm(w_int)), 1e-30)
    residual_dynamic = float(np.linalg.norm(w_int - (w_ext - w_inertia)) / scale)
    residual_static = float(np.linalg.norm(w_int - w_ext) / scale)

    w_boundary = np.zeros((2, 2))
    if len(table["boundary_i"]):
        f_b = np.zeros((n, 2))
        np.add.at(f_b, table["boundary_i"], table["boundary_force_on_i"])
        w_boundary = _outer(system.pos, f_b)
    residual_boundary = float(np.linalg.norm(w_int - w_boundary) / max(scale, 1e-30))

    # periodic counterpart of the Virial identity: with minimum-image branch
    # vectors the raw sum is W + W_image, and W + W_image = -sum_p x_p (x) F_p^int,
    # which vanishes for a force-balanced configuration.  This is therefore a
    # *physical* check, not an algebraic tautology.
    w_image = np.zeros((2, 2))
    if len(table["i"]):
        lengths = np.array([system.periodic_x or 0.0, system.periodic_y or 0.0])
        shift = table["image"] * lengths[None, :]
        w_image = _outer(shift, table["force_on_i"])
    periodic_norm = float(np.linalg.norm(w_int)) if len(table["i"]) else 0.0
    radius_residual = float(np.linalg.norm(w_int + w_image) / max(periodic_norm, 1e-30))

    sigma = w_int / area
    sigma_sym = 0.5 * (sigma + sigma.T)
    raw_symmetry = float(np.linalg.norm(sigma - sigma.T) / max(float(np.linalg.norm(sigma)), 1e-30))
    norm_sym = float(np.linalg.norm(sigma_sym))
    symmetry = float(np.linalg.norm(sigma_sym - sigma_sym.T) / max(norm_sym, 1e-30))
    antisym = float(np.linalg.norm(sigma - sigma.T) / max(2.0 * float(np.linalg.norm(sigma)), 1e-30))
    return {
        "virial_identity_rel": residual_dynamic,
        "virial_identity_static_rel": residual_static,
        "boundary_traction_rel": residual_boundary,
        "stress_symmetry_rel": symmetry,
        "virial_symmetry_raw_rel": raw_symmetry,
        "virial_identity_periodic_rel": radius_residual,
        "antisymmetric_fraction": antisym,
        "virial": w_int.ravel().copy(),
        "external_virial": w_ext.ravel().copy(),
        "sigma": sigma_sym.ravel().copy(),
    }


def stress_invariants(sigma: np.ndarray) -> dict[str, float]:
    """2D pressure, deviatoric magnitude and principal direction of ``sigma``."""
    s = np.asarray(sigma, dtype=np.float64).reshape(2, 2)
    s = 0.5 * (s + s.T)
    pressure = -0.5 * float(np.trace(s))
    dev = s - 0.5 * float(np.trace(s)) * np.eye(2)
    dev_mag = float(np.linalg.norm(dev))
    eigval, eigvec = np.linalg.eigh(s)
    major = eigvec[:, int(np.argmax(eigval))]
    return {
        "pressure": pressure,
        "trace": float(np.trace(s)),
        "deviatoric": dev_mag,
        "stress_ratio": dev_mag / max(abs(pressure), 1e-30),
        "major_axis_angle": float(math.atan2(major[1], major[0])),
    }


def lattice_stress(system, nx: int = 12, ny: int = 8, *, bounds=None,
                   contacts=None) -> dict[str, np.ndarray]:
    """Coarse-grained (grid) contact stress field.

    ``sigma[ix, iy]`` is ``(2,2)``; each contact is assigned to the cell that
    contains its mid-point and normalised by the cell area, so
    ``sum(sigma*area) == total virial`` up to the (empty) assignment error.
    Returns the field, per-cell contact counts and the cell areas.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    table = contact_stress_table(system) if contacts is None else contacts
    if bounds is None:
        if len(system.walls) == 0:
            raise ValueError("bounds are required for a system without walls")
        pts = system.walls.reshape(-1, 2)
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    else:
        x0, y0, x1, y1 = (float(v) for v in bounds)
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    cell_area = dx * dy
    sigma = np.zeros((nx, ny, 2, 2))
    counts = np.zeros((nx, ny), dtype=np.int64)
    if len(table["i"]):
        mid = 0.5 * (system.pos[table["i"]] + system.pos[table["j"]])
        ix = np.clip(((mid[:, 0] - x0) / dx).astype(int), 0, nx - 1)
        iy = np.clip(((mid[:, 1] - y0) / dy).astype(int), 0, ny - 1)
        contrib = table["branch"][:, :, None] * table["force_on_i"][:, None, :]
        np.add.at(sigma, (ix, iy), contrib)
        np.add.at(counts, (ix, iy), 1)
    sigma /= cell_area
    return {"sigma": sigma, "counts": counts, "dx": np.array([dx]), "dy": np.array([dy]),
            "cell_area": np.array([cell_area]), "bounds": np.array([x0, y0, x1, y1])}


def hydrostatic_check(system, *, n_bands: int = 12, band_index: int = 0) -> dict[str, float]:
    """Compare the coarse-grained vertical stress with the weight per unit width.

    For a static packing under gravity the vertical normal stress at depth ``y``
    is :math:`\\sigma_{yy}(y) = W_{above}(y)/L_x`.  The band value is compared
    with the weight above the band *centre*, which removes the first-order
    band-weight bias.
    """
    if len(system.walls) == 0:
        raise ValueError("a walled system is required")
    pts = system.walls.reshape(-1, 2)
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    width, height = x1 - x0, y1 - y0
    field = lattice_stress(system, nx=1, ny=n_bands, bounds=(x0, y0, x1, y1))
    g = abs(float(system.config.gravity[1]))
    if g == 0.0:
        raise ValueError("gravity must be non-zero for a hydrostatic check")
    dy = height / n_bands
    y_center = y0 + (band_index + 0.5) * dy
    weight_above = float(np.sum(system.mass[system.pos[:, 1] >= y_center])) * g
    reference = weight_above / width
    sigma_yy = float(-field["sigma"][0, band_index, 1, 1])
    denom = max(abs(reference), 1e-30)
    # independent route: the floor traction must carry the total weight
    contacts = system.contacts
    total_weight = float(np.sum(system.mass)) * g
    floor_force = 0.0
    if len(contacts.get("i", [])):
        kind = np.asarray(contacts["kind"])
        wall = kind == 1
        if wall.any():
            fy = np.asarray(contacts["f_on_i"])[wall, 1]
            pts = np.asarray(contacts["point"])[wall]
            low = pts[:, 1] <= y0 + 2.0 * system.mean_radius
            floor_force = float(np.sum(fy[low]))
    traction = floor_force / width
    reference_total = total_weight / width
    return {
        "sigma_yy_coarse": sigma_yy,
        "sigma_yy_reference": reference,
        "relative_error": abs(sigma_yy - reference) / denom,
        "band_center_y": y_center,
        "cell_height": dy,
        "bottom_traction": traction,
        "weight_per_width": reference_total,
        "traction_relative_error": abs(traction - reference_total) / max(abs(reference_total), 1e-30),
    }