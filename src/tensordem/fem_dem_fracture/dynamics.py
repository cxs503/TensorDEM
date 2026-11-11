"""Dynamic FEM -> DEM fracture handoff: velocity inheritance + rigid projection.

Why this module exists
----------------------
The static handoff (:mod:`tensordem.fem_dem_fracture.handoff`) gives the DEM a
census of floes with ``velocity = 0``.  A DEM run started from it moves nothing:
the discrete solver has to invent the motion.  This module supplies the part of
the motion the continuum solve actually knows -- the *deflection rate of the
loading path* -- and converts it into rigid-body initial conditions with an
explicit, checkable conservation ledger.

The inherited velocity field
----------------------------
The Kirchhoff plate problem is linear in the contact-pressure scale ``q``, so
the deflection rate under a monotone pressure ramp ``qdot`` is exactly

    wdot(x, y) = qdot * w_hat(x, y),        K w_hat = b_hat,

with ``w_hat`` the unit-pressure solve [m/Pa].  No time stepping and no
pseudo-dynamics are involved: this is the loading-path rate of the static
solution and is exact for the modelled (monotone, quasi-static) path.  Cell
values are reconstructed at the cell centroid from all 12 ACM dofs
(``w``, ``w,x``, ``w,y`` at each corner) by a first-order Taylor per corner
averaged over the four corners, which is *exact for a linear (rigid) field* --
the property the projection tests below rely on.

Rigid-body projection
---------------------
A DEM body is a rigid floe: it can translate and tilt, it cannot carry the
plate's curvature.  Each floe therefore receives the mass-weighted
least-squares rigid fit of its cell velocities,

    min_{a,b,c} sum_c m_c (u_c - a - b dx_c - c dy_c)^2,   m_c = rho t A_c,

evaluated in the floe's principal axes (``e1``, ``e2`` of the second-moment
tensor ``T = sum m d d^T``), where the tensor is diagonal.  From the normal
equations, with ``M = sum m``, ``I_a = sum m (d.e1)^2``, ``I_b = sum m (d.e2)^2``:

* linear momentum is exact: ``sum_c m_c u_c = M a`` (first moments vanish about
  the mass centre);
* in-plane angular momentum about the floe mass centre is exact:
  ``L = I_a omega_1 e1 + I_b omega_2 e2`` with ``omega_1 = c``, ``omega_2 = -b``;
* kinetic energy splits exactly, ``KE_total = KE_rigid + KE_deform``, because
  every cross term vanishes by the normal equations.  ``KE_deform`` is therefore
  the curvature energy the rigid body cannot represent, and the "rigid fraction"
  ``KE_rigid / KE_total`` is a well-defined number in [0, 1].

Scope and stated limits (nothing below is silently assumed)
-----------------------------------------------------------
* crack-tip stress redistribution during the fracture event is not modelled
  (the crack tracer is a frozen-field pattern generator, see wave 1);
* floe-floe contact during the fracture event and hydrodynamic added mass are
  outside the handoff; the DEM supplies both;
* the handoff is a *quasi-static* initial condition while the loading is slow.
  The module therefore reports ``KE / U_elastic`` (inherited kinetic energy over
  the elastic energy stored in the plate at the fracture load) and the loading
  rate at which that ratio reaches 10 % (``critical_rate_pa_s``).  Above that
  rate the DEM should be started from a genuinely dynamic FEM, not from here;
* in-plane (membrane) motion is not part of a bending solve; the in-plane initial
  velocity is an environmental input (default zero) and is carried in the payload
  with its provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math

import numpy as np

from .crack_network import point_in_polygon
from .handoff import FailureFaces, payload_sha256, read_failure_faces

__all__ = [
    "SCHEMA_DYNAMIC",
    "SCHEMA_DEM_IC",
    "RigidFit",
    "VelocityField",
    "FloeKinematics",
    "MomentumLedger",
    "DynamicFailureFaces",
    "cell_centroids",
    "cell_centroid_unit_rates",
    "elastic_energy",
    "kinetic_energy",
    "velocity_field",
    "quasi_static_ledger",
    "project_point_masses",
    "floe_cell_membership",
    "project_rigid_body_motion",
    "momentum_ledger",
    "build_dynamic_failure_faces",
    "write_dynamic_failure_faces",
    "read_dynamic_failure_faces",
    "dem_initial_conditions_dynamic",
    "linearity_check",
]

SCHEMA_DYNAMIC = "tensordem.failure_faces/2"
SCHEMA_DEM_IC = "tensordem.dem_initial_conditions/2"
_CODE_VERSION = "tensordem.fem_dem_fracture.dynamics 1.0 (campaign p302w2-n03)"
GRAVITY = 9.80665


# --------------------------------------------------------------------------
# inherited velocity field
# --------------------------------------------------------------------------
def cell_centroid_unit_rates(solution) -> np.ndarray:
    """Cell-centroid deflection per unit pressure scale [m/Pa].

    ``solution`` must be the *unit* pressure solve (``load_amplitude = 1``), so
    its nodal ``w`` is already ``d w / d q``.  The centroid value uses the ACM
    corner dofs by first-order Taylor reconstruction at each corner, averaged
    over the four corners: exact for linear fields, ``O(h^2)`` for curved ones.

    Returns an ``(n_cells,)`` array in the flat cell order ``j * nx + i``.
    """
    mesh = solution.mesh
    nx, ny = mesh.nx, mesh.ny
    xn = np.arange(nx + 1) * mesh.hx
    yn = np.arange(ny + 1) * mesh.hy
    xc = (np.arange(nx) + 0.5) * mesh.hx
    yc = (np.arange(ny) + 0.5) * mesh.hy
    XC = np.broadcast_to(xc[None, :], (ny, nx))
    YC = np.broadcast_to(yc[:, None], (ny, nx))
    w = np.asarray(solution.w).reshape(ny + 1, nx + 1)
    tx = np.asarray(solution.theta_x).reshape(ny + 1, nx + 1)
    ty = np.asarray(solution.theta_y).reshape(ny + 1, nx + 1)
    acc = np.zeros((ny, nx), dtype=float)
    for dj in (0, 1):
        for di in (0, 1):
            W = w[dj:dj + ny, di:di + nx]
            TX = tx[dj:dj + ny, di:di + nx]
            TY = ty[dj:dj + ny, di:di + nx]
            dx = XC - xn[di:di + nx][None, :]
            dy = YC - yn[dj:dj + ny][:, None]
            acc += W + TX * dx + TY * dy
    return (acc / 4.0).reshape(-1)


def cell_centroids(mesh) -> np.ndarray:
    """``(n_cells, 2)`` cell centroids in the flat cell order ``j * nx + i``."""
    nx, ny = mesh.nx, mesh.ny
    xc = (np.arange(nx) + 0.5) * mesh.hx
    yc = (np.arange(ny) + 0.5) * mesh.hy
    XX, YY = np.meshgrid(xc, yc, indexing="xy")
    return np.stack([XX.ravel(), YY.ravel()], axis=1)


@dataclass(frozen=True)
class VelocityField:
    """Inherited cell-centroid velocities of the loading path."""

    mesh: object
    centroids: np.ndarray       # (n_cells, 2) [m]
    unit_rates: np.ndarray      # (n_cells,) [m/Pa] = d w / d q
    rates: np.ndarray           # (n_cells,) [m/s] at pressure_rate_pa_s
    pressure_rate_pa_s: float
    fracture_pressure_pa: float

    @property
    def n_cells(self) -> int:
        return int(self.rates.size)

    @property
    def max_abs_rate(self) -> float:
        return float(np.max(np.abs(self.rates)))


def velocity_field(solution, *, pressure_rate_pa_s: float,
                   fracture_pressure_pa: float) -> VelocityField:
    """Build the inherited velocity field from the unit-pressure solution."""
    if not math.isfinite(pressure_rate_pa_s) or pressure_rate_pa_s < 0.0:
        raise ValueError("pressure_rate_pa_s must be finite and non-negative")
    if not math.isfinite(fracture_pressure_pa) or fracture_pressure_pa <= 0.0:
        raise ValueError("fracture_pressure_pa must be finite and positive")
    unit = cell_centroid_unit_rates(solution)
    return VelocityField(
        mesh=solution.mesh,
        centroids=cell_centroids(solution.mesh),
        unit_rates=unit,
        rates=unit * float(pressure_rate_pa_s),
        pressure_rate_pa_s=float(pressure_rate_pa_s),
        fracture_pressure_pa=float(fracture_pressure_pa),
    )


def linearity_check(solution, patches, *, delta: float = 1.0) -> dict:
    """Finite-difference check of ``wdot = qdot * w_hat`` (exact for a linear K).

    Solves the plate at pressure scale ``1 + delta`` and compares
    ``(w(1+delta) - w(1)) / delta`` with the unit solve.  The reference residual
    is zero (the operator is linear), so the measured number is round-off.
    """
    from .plate_fem import solve_plate

    mesh = solution.mesh
    other = solve_plate(mesh, solution.material, patches,
                        foundation_modulus=solution.foundation_modulus,
                        load_amplitude=1.0 + delta,
                        boundary=solution.extra.get("boundary", "free"),
                        constrained_edges=solution.extra.get("constrained_edges",
                                                             ("x0", "x1", "y0", "y1")))
    fd = (np.asarray(other.w) - np.asarray(solution.w)) / delta
    unit = cell_centroid_unit_rates(solution)
    # ``solution`` is the unit-pressure solve, so both fields scale linearly:
    # w(1 + delta) = (1 + delta) w(1) and the quotient is the unit field itself.
    fd_cells = (cell_centroid_unit_rates(other) - unit) / delta
    scale = max(float(np.max(np.abs(fd))), 1e-300)
    scale_cells = max(float(np.max(np.abs(unit))), 1e-300)
    return {
        "delta": delta,
        "nodal_max_abs_fd_m_per_pa": float(np.max(np.abs(fd))),
        "nodal_max_abs_unit_m_per_pa": float(np.max(np.abs(solution.w))),
        "nodal_rel_residual": float(np.max(np.abs(fd - solution.w)) / scale),
        "cell_rel_residual": float(np.max(np.abs(fd_cells - unit)) / scale_cells),
    }


def elastic_energy(solution, patches, *, pressure_scale: float = 1.0) -> float:
    """Elastic energy stored in the plate at the given pressure scale [J].

    For the linear system ``K w = b`` the equilibrium energy is
    ``U = 0.5 w^T b = 0.5 q^2 w_hat^T b_hat``; the consistent nodal load vector
    of the contact patches is the one assembled by the solver, so this is exact.
    """
    from .plate_fem import _pressure_load_vector, acm_inverse

    mesh = solution.mesh
    cinv = acm_inverse(mesh.hx, mesh.hy)
    # The consistent load vector lives on the deflection dofs (3 * node + 0).
    b = _pressure_load_vector(mesh, patches, cinv, 3 * mesh.cell_ids(),
                              3 * mesh.n_nodes)
    dofs = np.empty(3 * mesh.n_nodes, dtype=float)
    dofs[0::3] = np.asarray(solution.w)
    dofs[1::3] = np.asarray(solution.theta_x)
    dofs[2::3] = np.asarray(solution.theta_y)
    return 0.5 * float(pressure_scale) ** 2 * float(dofs @ b)


def kinetic_energy(field: VelocityField, *, density: float, thickness: float) -> float:
    """Translational kinetic energy of the inherited field, ``0.5 rho t int v^2``."""
    area = float(field.mesh.cell_area)
    return 0.5 * density * thickness * area * float(np.sum(field.rates ** 2))


def quasi_static_ledger(solution, patches, field: VelocityField, *,
                        density: float, thickness: float,
                        ratio_target: float = 0.1) -> dict:
    """``KE/U`` at the requested rate plus the rate where it reaches ``ratio_target``.

    ``KE/U = (qdot / qdot_crit)^2`` exactly (both are quadratic in the load
    scale), so ``qdot_crit`` is obtained in closed form rather than by bisection.
    """
    u_el = elastic_energy(solution, patches, pressure_scale=field.fracture_pressure_pa)
    ke = kinetic_energy(field, density=density, thickness=thickness)
    qdot = field.pressure_rate_pa_s
    out = {
        "elastic_energy_j": u_el,
        "kinetic_energy_j": ke,
        "ke_over_elastic": (ke / u_el) if u_el > 0.0 else float("inf"),
        "pressure_rate_pa_s": qdot,
        "ratio_target": ratio_target,
        "quasi_static": bool(u_el > 0.0 and ke / u_el <= ratio_target),
        "ke_scaling_exponent": 2.0,
    }
    if ke > 0.0 and u_el > 0.0:
        out["critical_rate_pa_s"] = qdot * math.sqrt(ratio_target * u_el / ke)
    else:
        out["critical_rate_pa_s"] = float("inf")
    return out


# --------------------------------------------------------------------------
# rigid-body projection
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RigidFit:
    """Mass-weighted rigid fit of a velocity field over a set of point masses."""

    mass: float
    centroid: np.ndarray            # (2,)
    inertia_axis1: float            # I about e1 = sum m (d.e2)^2... see below
    inertia_axis2: float            # I about e2
    principal_angle: float          # atan2(e2_y, e2_x) [rad]
    v_z: float
    omega_1: float                  # about e1
    omega_2: float                  # about e2
    omega_x: float                  # global x component of the rotation vector
    omega_y: float                  # global y component
    ke_rigid: float
    ke_deform: float
    ke_total: float
    momentum_z: float
    angular_momentum: np.ndarray    # (2,) about the centroid, global x/y
    max_abs_deform: float
    n_points: int

    @property
    def rigid_fraction(self) -> float:
        return 0.0 if self.ke_total <= 0.0 else self.ke_rigid / self.ke_total

    def as_dict(self) -> dict:
        return {
            "mass_kg": self.mass,
            "centroid_xy_m": [float(v) for v in self.centroid],
            "inertia_about_e1_kg_m2": self.inertia_axis1,
            "inertia_about_e2_kg_m2": self.inertia_axis2,
            "principal_angle_rad": self.principal_angle,
            "v_z_m_s": self.v_z,
            "omega_1_rad_s": self.omega_1,
            "omega_2_rad_s": self.omega_2,
            "omega_x_rad_s": self.omega_x,
            "omega_y_rad_s": self.omega_y,
            "ke_rigid_j": self.ke_rigid,
            "ke_deform_j": self.ke_deform,
            "ke_total_j": self.ke_total,
            "rigid_fraction": self.rigid_fraction,
            "momentum_z_kg_m_s": self.momentum_z,
            "angular_momentum_xy_kg_m2_s": [float(v) for v in self.angular_momentum],
            "max_abs_deform_m_s": self.max_abs_deform,
            "n_points": int(self.n_points),
        }


def project_point_masses(x, y, u, m) -> RigidFit:
    """Least-squares rigid (translate + tilt) fit of a vertical velocity field.

    ``x``, ``y``, ``u``, ``m`` are flat arrays: point coordinates [m], vertical
    velocity [m/s] and point mass [kg].  The fit is done in the principal axes of
    ``T = sum m d d^T``, where the inertia tensor is diagonal, so momentum and
    angular momentum are reproduced exactly (see the module docstring).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    if not (x.size == y.size == u.size == m.size):
        raise ValueError("x, y, u, m must have the same length")
    if x.size == 0:
        raise ValueError("empty point set")
    if np.any(m <= 0.0) or not np.all(np.isfinite(m)):
        raise ValueError("point masses must be finite and positive")
    total = float(np.sum(m))
    cx = float(np.sum(m * x) / total)
    cy = float(np.sum(m * y) / total)
    dx = x - cx
    dy = y - cy
    sxx = float(np.sum(m * dx * dx))
    syy = float(np.sum(m * dy * dy))
    sxy = float(np.sum(m * dx * dy))
    tensor = np.array([[sxx, sxy], [sxy, syy]], dtype=float)
    _, evecs = np.linalg.eigh(tensor)               # ascending eigenvalues
    e2 = evecs[:, 1]                                # larger second moment axis
    e1 = np.array([e2[1], -e2[0]], dtype=float)     # perpendicular, right handed
    xp = dx * e1[0] + dy * e1[1]
    yp = dx * e2[0] + dy * e2[1]
    ib = float(np.sum(m * xp * xp))                 # inertia about e2
    ia = float(np.sum(m * yp * yp))                 # inertia about e1
    momentum = float(np.sum(m * u))
    a = momentum / total
    b = (float(np.sum(m * xp * u)) / ib) if ib > 0.0 else 0.0
    c = (float(np.sum(m * yp * u)) / ia) if ia > 0.0 else 0.0
    omega_1 = c
    omega_2 = -b
    omega_x = omega_1 * e1[0] + omega_2 * e2[0]
    omega_y = omega_1 * e1[1] + omega_2 * e2[1]
    deform = u - (a + b * xp + c * yp)
    ke_total = 0.5 * float(np.sum(m * u * u))
    ke_rigid = 0.5 * total * a * a + 0.5 * (ia * omega_1 ** 2 + ib * omega_2 ** 2)
    ke_deform = 0.5 * float(np.sum(m * deform * deform))
    return RigidFit(
        mass=total,
        centroid=np.array([cx, cy]),
        inertia_axis1=ia,
        inertia_axis2=ib,
        principal_angle=float(math.atan2(e2[1], e2[0])),
        v_z=a,
        omega_1=omega_1,
        omega_2=omega_2,
        omega_x=float(omega_x),
        omega_y=float(omega_y),
        ke_rigid=float(ke_rigid),
        ke_deform=float(ke_deform),
        ke_total=float(ke_total),
        momentum_z=momentum,
        angular_momentum=np.array([float(np.sum(m * dy * u)),
                                   float(-np.sum(m * dx * u))]),
        max_abs_deform=float(np.max(np.abs(deform))),
        n_points=int(x.size),
    )


def floe_cell_membership(faces: FailureFaces, mesh) -> list[np.ndarray]:
    """``(n_cells,)`` index arrays of the FEM cells belonging to each floe.

    Recovered by point-in-polygon test on the cell centroids, so the mapping is
    rebuilt from the artefact alone (the same discipline as the static reader).
    A cell centroid that belongs to no floe, or a floe that does not recover its
    declared cell count, is an error: it means the polygon tracing and the mesh
    partition disagree.
    """
    centroids = cell_centroids(mesh)
    assignment = np.full(len(centroids), -1, dtype=int)
    for index, floe in enumerate(faces.floes):
        rings = [np.asarray(ring, dtype=float) for ring in floe.polygons]
        inside = np.array([point_in_polygon(c, rings[0]) for c in centroids])
        for hole in rings[1:]:
            inside &= ~np.array([point_in_polygon(c, hole) for c in centroids])
        if inside.any():
            if (assignment[inside] >= 0).any():
                raise ValueError(
                    f"floe {floe.floe_id}: overlapping polygons in the artefact")
            assignment[inside] = index
    missing = np.nonzero(assignment < 0)[0]
    if missing.size:
        raise ValueError(
            f"{missing.size} FEM cells are not covered by any floe polygon "
            f"(first: {missing[:5].tolist()})")
    out = []
    for index, floe in enumerate(faces.floes):
        cells = np.nonzero(assignment == index)[0]
        if cells.size != floe.n_cells:
            raise ValueError(
                f"floe {floe.floe_id}: artefact declares {floe.n_cells} cells but the "
                f"polygon recovers {cells.size}")
        out.append(cells)
    return out


@dataclass(frozen=True)
class FloeKinematics:
    """Rigid-body initial condition of one floe, with its energy split."""

    floe_id: int
    fit: RigidFit
    equivalent_diameter: float
    n_cells: int
    z0: float = 0.0            # heave offset handed to the DEM (mid-plane default)
    v_inplane: tuple = (0.0, 0.0)

    @property
    def mass(self) -> float:
        return self.fit.mass

    @property
    def centroid(self) -> np.ndarray:
        return self.fit.centroid

    @property
    def v_z(self) -> float:
        return self.fit.v_z

    @property
    def omega_x(self) -> float:
        return self.fit.omega_x

    @property
    def omega_y(self) -> float:
        return self.fit.omega_y

    @property
    def ke_total(self) -> float:
        return self.fit.ke_total

    @property
    def rigid_fraction(self) -> float:
        return self.fit.rigid_fraction

    @property
    def inertia_about_e1(self) -> float:
        return self.fit.inertia_axis1

    @property
    def inertia_about_e2(self) -> float:
        return self.fit.inertia_axis2

    def as_dict(self) -> dict:
        payload = self.fit.as_dict()
        payload.update({
            "floe_id": int(self.floe_id),
            "equivalent_diameter_m": float(self.equivalent_diameter),
            "n_cells": int(self.n_cells),
            "z0_m": float(self.z0),
            "v_inplane_m_s": [float(self.v_inplane[0]), float(self.v_inplane[1])],
        })
        return payload


def project_rigid_body_motion(faces: FailureFaces, field: VelocityField) -> list[FloeKinematics]:
    """Per-floe rigid fit of the inherited velocity field."""
    mesh = field.mesh
    if mesh.n_cells != field.n_cells:
        raise ValueError("velocity field does not match the mesh")
    density = float(faces.meta["material"]["density_kg_m3"])
    thickness = float(faces.meta["plate"]["thickness_m"])
    point_mass = density * thickness * float(mesh.cell_area)
    masses = np.full(field.n_cells, point_mass)
    out: list[FloeKinematics] = []
    for floe, cells in zip(faces.floes, floe_cell_membership(faces, mesh)):
        fit = project_point_masses(
            field.centroids[cells, 0], field.centroids[cells, 1],
            field.rates[cells], masses[cells])
        out.append(FloeKinematics(
            floe_id=int(floe.floe_id),
            fit=fit,
            equivalent_diameter=float(floe.equivalent_diameter),
            n_cells=int(cells.size),
        ))
    return out


@dataclass(frozen=True)
class MomentumLedger:
    """Conservation ledger of the FEM -> rigid-DEM velocity transfer."""

    n_bodies: int
    mass_field: float
    mass_bodies: float
    mass_residual: float
    momentum_field: float
    momentum_bodies: float
    momentum_residual: float
    angular_field: np.ndarray
    angular_bodies: np.ndarray
    angular_residual: np.ndarray
    ke_field: float
    ke_rigid: float
    ke_deform: float
    ke_sum_residual: float
    rigid_fraction: float
    max_floe_rigid_fraction: float
    min_floe_rigid_fraction: float
    reference_point: tuple

    def as_dict(self) -> dict:
        return {
            "n_bodies": int(self.n_bodies),
            "mass_field_kg": self.mass_field,
            "mass_bodies_kg": self.mass_bodies,
            "mass_residual_kg": self.mass_residual,
            "momentum_field_kg_m_s": self.momentum_field,
            "momentum_bodies_kg_m_s": self.momentum_bodies,
            "momentum_residual_kg_m_s": self.momentum_residual,
            "angular_field_kg_m2_s": [float(v) for v in self.angular_field],
            "angular_bodies_kg_m2_s": [float(v) for v in self.angular_bodies],
            "angular_residual_kg_m2_s": [float(v) for v in self.angular_residual],
            "ke_field_j": self.ke_field,
            "ke_rigid_j": self.ke_rigid,
            "ke_deform_j": self.ke_deform,
            "ke_sum_residual_j": self.ke_sum_residual,
            "rigid_fraction_plate": self.rigid_fraction,
            "max_floe_rigid_fraction": self.max_floe_rigid_fraction,
            "min_floe_rigid_fraction": self.min_floe_rigid_fraction,
            "reference_point_xy_m": [float(v) for v in self.reference_point],
        }


def momentum_ledger(faces: FailureFaces, field: VelocityField,
                    kinematics: list[FloeKinematics]) -> MomentumLedger:
    """Compare the rigid-body transfer with the continuum field (exact identities)."""
    mesh = field.mesh
    density = float(faces.meta["material"]["density_kg_m3"])
    thickness = float(faces.meta["plate"]["thickness_m"])
    point_mass = density * thickness * float(mesh.cell_area)
    ref = np.array([0.5 * mesh.length_x, 0.5 * mesh.length_y])

    rx = field.centroids[:, 0] - ref[0]
    ry = field.centroids[:, 1] - ref[1]
    u = field.rates
    m = np.full(field.n_cells, point_mass)

    p_field = float(np.sum(m * u))
    l_field = np.array([float(np.sum(m * ry * u)), float(-np.sum(m * rx * u))])
    ke_field = 0.5 * float(np.sum(m * u * u))

    p_bodies = float(sum(k.fit.momentum_z for k in kinematics))
    l_bodies = np.zeros(2)
    for k in kinematics:
        c = k.centroid
        dx = c[0] - ref[0]
        dy = c[1] - ref[1]
        l_bodies += k.fit.angular_momentum + k.mass * np.array(
            [dy * k.fit.v_z, -dx * k.fit.v_z])
    ke_rigid = float(sum(k.fit.ke_rigid for k in kinematics))
    ke_deform = float(sum(k.fit.ke_deform for k in kinematics))
    fractions = [k.rigid_fraction for k in kinematics]
    mass_bodies = float(sum(k.mass for k in kinematics))
    return MomentumLedger(
        n_bodies=len(kinematics),
        mass_field=float(np.sum(m)),
        mass_bodies=mass_bodies,
        mass_residual=mass_bodies - float(np.sum(m)),
        momentum_field=p_field,
        momentum_bodies=p_bodies,
        momentum_residual=p_bodies - p_field,
        angular_field=l_field,
        angular_bodies=l_bodies,
        angular_residual=l_bodies - l_field,
        ke_field=ke_field,
        ke_rigid=ke_rigid,
        ke_deform=ke_deform,
        ke_sum_residual=ke_rigid + ke_deform - ke_field,
        rigid_fraction=0.0 if ke_field <= 0.0 else ke_rigid / ke_field,
        max_floe_rigid_fraction=float(max(fractions)) if fractions else 0.0,
        min_floe_rigid_fraction=float(min(fractions)) if fractions else 0.0,
        reference_point=(float(ref[0]), float(ref[1])),
    )


# --------------------------------------------------------------------------
# artefact: failure_faces.json schema v2
# --------------------------------------------------------------------------
def build_dynamic_failure_faces(faces: FailureFaces, field: VelocityField,
                                kinematics: list[FloeKinematics],
                                ledger: MomentumLedger,
                                *,
                                quasi_static: dict | None = None,
                                extra_meta: dict | None = None) -> dict:
    """Assemble the schema-2 payload (schema-1 body + dynamics block)."""
    payload = faces.to_payload()
    payload["schema"] = SCHEMA_DYNAMIC
    meta = dict(payload["meta"])
    meta["dynamics"] = {
        "model": ("loading-path rate of the static plate solve, qdot * w_hat; "
                  "rigid-body least-squares projection in floe principal axes"),
        "pressure_rate_pa_s": field.pressure_rate_pa_s,
        "fracture_pressure_pa": field.fracture_pressure_pa,
        "max_cell_rate_m_s": field.max_abs_rate,
        "cell_interpolation": ("first-order Taylor from the 12 ACM corner dofs, "
                              "averaged over the four corners (exact for linear fields)"),
        "in_plane_velocity": ("environmental input, zero "
                              "(a bending solve has no membrane kinematics)"),
        **(quasi_static or {}),
    }
    if extra_meta:
        meta.update(extra_meta)
    payload["meta"] = meta

    by_id = {k.floe_id: k for k in kinematics}
    for record in payload["floes"]:
        record["kinematics"] = by_id[record["floe_id"]].as_dict()

    cells_payload = []
    for n in range(field.n_cells):
        i, j = field.mesh.cell_index(n)
        cells_payload.append({
            "cell": [int(i), int(j)],
            "centroid_xy_m": [float(field.centroids[n, 0]), float(field.centroids[n, 1])],
            "unit_rate_m_per_pa": float(field.unit_rates[n]),
            "v_z_m_s": float(field.rates[n]),
        })
    payload["velocity_field"] = {
        "unit": "m/s at meta.dynamics.pressure_rate_pa_s",
        "n_cells": int(field.n_cells),
        "cell_area_m2": float(field.mesh.cell_area),
        "cells": cells_payload,
    }
    payload["momentum_ledger"] = ledger.as_dict()
    payload["provenance"] = {
        "code": _CODE_VERSION,
        "generator": "tensordem.fem_dem_fracture.dynamics",
        "wave": "p302w2-n03 (wave 2: dynamic handoff)",
        "notes": ("momentum / angular momentum / KE sum rule are exact identities of the "
                  "mass-weighted rigid fit and are recomputed by the reader"),
    }
    return payload


def write_dynamic_failure_faces(payload: dict, path: str) -> str:
    """Write the schema-2 artefact with a payload SHA-256 in its provenance."""
    body = {k: v for k, v in payload.items() if k != "provenance"}
    body["provenance"] = {k: v for k, v in payload.get("provenance", {}).items()
                          if k != "payload_sha256"}
    digest = payload_sha256(body)
    out = dict(payload)
    out["provenance"] = {**payload.get("provenance", {}), "payload_sha256": digest}
    text = json.dumps(out, indent=2, sort_keys=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
    return digest


@dataclass
class DynamicFailureFaces:
    """Read-back image of ``failure_faces_dynamic.json`` (schema 2)."""

    faces: FailureFaces
    field: VelocityField
    kinematics: list[FloeKinematics]
    ledger: MomentumLedger
    stored_ledger: dict
    dynamics_meta: dict
    provenance: dict
    stored_kinematics: dict = field(default_factory=dict)

    @property
    def n_bodies(self) -> int:
        return len(self.kinematics)


def read_dynamic_failure_faces(path_or_payload) -> DynamicFailureFaces:
    """Read and *revalidate* a schema-2 handoff artefact.

    The whole dynamics ledger is recomputed from the stored cell velocities and
    the stored floe polygons -- the stored per-floe kinematics are only compared
    against it.  A tampered or stale file therefore cannot silently feed the DEM
    with wrong initial conditions.
    """
    if isinstance(path_or_payload, dict):
        payload = path_or_payload
    else:
        with open(path_or_payload, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if payload.get("schema") != SCHEMA_DYNAMIC:
        raise ValueError(f"unexpected schema {payload.get('schema')!r}, "
                         f"expected {SCHEMA_DYNAMIC!r}")

    if "payload_sha256" in payload.get("provenance", {}):
        body = {k: v for k, v in payload.items() if k != "provenance"}
        body["provenance"] = {k: v for k, v in payload["provenance"].items()
                              if k != "payload_sha256"}
        if payload_sha256(body) != payload["provenance"]["payload_sha256"]:
            raise ValueError("payload SHA-256 mismatch: the artefact was edited after writing")

    faces = read_failure_faces(payload)          # schema-1 validation (area/mass/rim)
    meta = payload["meta"]
    mesh_meta = meta["mesh"]

    from .plate_fem import PlateMesh

    mesh = PlateMesh(nx=int(mesh_meta["nx"]), ny=int(mesh_meta["ny"]),
                     length_x=float(meta["plate"]["length_x_m"]),
                     length_y=float(meta["plate"]["length_y_m"]))
    if mesh.n_cells != int(mesh_meta["n_cells"]):
        raise ValueError("mesh meta does not reproduce the declared cell count")

    cells = payload["velocity_field"]["cells"]
    if len(cells) != mesh.n_cells:
        raise ValueError("velocity field size does not match the mesh")
    centroids = np.array([c["centroid_xy_m"] for c in cells], dtype=float)
    rates = np.array([c["v_z_m_s"] for c in cells], dtype=float)
    unit = np.array([c["unit_rate_m_per_pa"] for c in cells], dtype=float)
    if not np.allclose(centroids, cell_centroids(mesh), rtol=0.0, atol=1e-9):
        raise ValueError("stored cell centroids are not the mesh cell centroids")

    dyn = meta.get("dynamics", {})
    qdot = float(dyn.get("pressure_rate_pa_s", 0.0))
    if qdot > 0.0 and not np.allclose(rates, unit * qdot, rtol=1e-9, atol=1e-15):
        raise ValueError("stored velocities are not qdot * unit_rate")

    field = VelocityField(
        mesh=mesh, centroids=centroids, unit_rates=unit, rates=rates,
        pressure_rate_pa_s=qdot,
        fracture_pressure_pa=float(dyn.get("fracture_pressure_pa", 0.0)),
    )
    recomputed = project_rigid_body_motion(faces, field)
    ledger = momentum_ledger(faces, field, recomputed)
    stored = payload.get("momentum_ledger", {})

    stored_kin = {f["floe_id"]: f.get("kinematics", {}) for f in payload["floes"]}
    for k in recomputed:
        ref = stored_kin.get(k.floe_id)
        if not ref:
            raise ValueError(f"floe {k.floe_id}: kinematics block missing")
        for key, value in (("v_z_m_s", k.fit.v_z),
                           ("omega_x_rad_s", k.fit.omega_x),
                           ("omega_y_rad_s", k.fit.omega_y),
                           ("ke_total_j", k.fit.ke_total),
                           ("rigid_fraction", k.fit.rigid_fraction)):
            if abs(float(ref.get(key, float("nan"))) - value) > 1e-9 * max(1.0, abs(value)):
                raise ValueError(
                    f"floe {k.floe_id}: stored {key} disagrees with the recomputed fit")
    for key, value in (("mass_residual_kg", ledger.mass_residual),
                       ("momentum_residual_kg_m_s", ledger.momentum_residual)):
        if key in stored and abs(float(stored[key]) - value) > 1e-9 * max(1.0, abs(value)):
            raise ValueError(f"stored {key} disagrees with the recomputed ledger")
    return DynamicFailureFaces(
        faces=faces,
        field=field,
        kinematics=recomputed,
        ledger=ledger,
        stored_ledger=stored,
        dynamics_meta=dyn,
        provenance=payload.get("provenance", {}),
        stored_kinematics=stored_kin,
    )


# --------------------------------------------------------------------------
# DEM side
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DemDynamicInitialConditions:
    """Rigid floe bodies (position, velocity, tilt) ready for a DEM solver."""

    positions: np.ndarray       # (N, 3) centroid x, y, z0
    velocities: np.ndarray      # (N, 3) vx, vy, vz [m/s]
    angular_velocities: np.ndarray   # (N, 3) omega_x, omega_y, omega_z [rad/s]
    masses: np.ndarray
    diameters: np.ndarray
    thickness: float
    floe_ids: np.ndarray
    ledger: MomentumLedger
    polygons: list = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.masses.size)

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.masses))

    @property
    def total_kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.masses * np.sum(self.velocities ** 2, axis=1)))

    def as_payload(self) -> dict:
        return {
            "schema": SCHEMA_DEM_IC,
            "n_bodies": len(self),
            "thickness_m": self.thickness,
            "total_mass_kg": self.total_mass,
            "total_kinetic_energy_j": self.total_kinetic_energy,
            "bodies": [
                {
                    "id": int(self.floe_ids[i]),
                    "position_m": [float(v) for v in self.positions[i]],
                    "velocity_m_s": [float(v) for v in self.velocities[i]],
                    "angular_velocity_rad_s": [float(v) for v in self.angular_velocities[i]],
                    "diameter_m": float(self.diameters[i]),
                    "mass_kg": float(self.masses[i]),
                }
                for i in range(len(self))
            ],
            "ledger": self.ledger.as_dict(),
        }


def dem_initial_conditions_dynamic(source) -> DemDynamicInitialConditions:
    """Convert a schema-2 artefact (or a DynamicFailureFaces) into DEM bodies.

    Distinction from the static ``dem_initial_conditions``: the bodies carry the
    inherited vertical velocity and the tilt rates of their floe, at the floe
    mass centroid recovered from the FEM cells (which is what the rigid fit is
    conservative about), not the polygon centroid.
    """
    if isinstance(source, DynamicFailureFaces):
        dynamic = source
    else:
        dynamic = read_dynamic_failure_faces(source)
    n = len(dynamic.kinematics)
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    omega = np.zeros((n, 3))
    masses = np.zeros(n)
    diameters = np.zeros(n)
    ids = np.zeros(n, dtype=int)
    for i, k in enumerate(dynamic.kinematics):
        positions[i] = (k.centroid[0], k.centroid[1], k.z0)
        velocities[i] = (k.v_inplane[0], k.v_inplane[1], k.fit.v_z)
        omega[i] = (k.fit.omega_x, k.fit.omega_y, 0.0)
        masses[i] = k.fit.mass
        diameters[i] = k.equivalent_diameter
        ids[i] = k.floe_id
    return DemDynamicInitialConditions(
        positions=positions,
        velocities=velocities,
        angular_velocities=omega,
        masses=masses,
        diameters=diameters,
        thickness=float(dynamic.faces.meta["plate"]["thickness_m"]),
        floe_ids=ids,
        ledger=dynamic.ledger,
        polygons=[f.polygons for f in dynamic.faces.floes],
    )