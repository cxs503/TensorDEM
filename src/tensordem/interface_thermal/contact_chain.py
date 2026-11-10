"""Contact-conduction and particle-chain verification face for DEM heat transfer.

Physical model
--------------
Two elastic spheres pressed together by a normal force ``F`` touch over a Hertz
patch of radius

.. math:: a = \\left(\\frac{3 F R^*}{4 E^*}\\right)^{1/3}, \\qquad
          \\frac{1}{R^*} = \\frac{1}{R_1} + \\frac{1}{R_2}, \\qquad
          \\frac{1}{E^*} = \\frac{1-\\nu_1^2}{E_1} + \\frac{1-\\nu_2^2}{E_2}

The heat flow through that patch is dominated by the *constriction* resistance
of each sphere.  For an isothermal circular patch on a half space the exact
spreading resistance is ``1/(4 k a)`` (Maxwell), so the pair resistance is

.. math:: R_{th} = \\frac{1}{4 k_1 a} + \\frac{1}{4 k_2 a} = \\frac{1}{2 k a}

for equal conductivities.  This is the analytic anchor of criterion (a); the
verification operator here is an independent axisymmetric finite-volume solution
of ``div(k grad T) = 0`` around the patch (:class:`HalfSpacePatch`), whose only
required input is the patch radius -- no resistance formula is used inside it.

A chain of ``N`` particles carries the series resistance
``N * L_p/(k A) + (N-1) R_c``; the steady temperature field is piecewise linear
with a jump at every contact (analytic in :func:`chain_steady_analytic`).  The
same chain is solved twice: by the lumped DEM network
(:class:`tensordem.interface_thermal.conduction.ParticleThermalNetwork`, one
temperature per particle) and by a fine 1D finite-volume rod with explicit
interface resistances (:func:`rod_temperature_fvm`).  Criterion (b) grades the
lumped field against that independent continuum solution, steady and transient.

Ice bonds add a second, parallel heat path: a bridge of ice of area ``A_b`` and
length ``L_b`` with the temperature dependent conductivity of ice
(:func:`ice_conductivity`, Petrenko & Whitworth 1999 / Slack 1980 fit).  The
thermal-mechanical gate (:func:`ice_bond_gate`) couples the *mechanical* contact
of the pair (which sets the Hertz patch and therefore ``R_th``) to the thermal
path of the ice bridge, and checks that the assembled pair resistance equals the
parallel combination of the two paths.

Units are SI; temperatures are degC and only differences enter the conduction
problems.  Surface heat fluxes are positive into the body.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

__all__ = [
    "hertz_patch_radius",
    "grain_area",
    "pair_contact_resistance",
    "HalfSpacePatch",
    "graded_axis",
    "rod_temperature_fvm",
    "chain_steady_analytic",
    "chain_lumped_steady",
    "chain_transient_lumped",
    "chain_transient_fvm",
    "ice_conductivity",
    "ICE_CONDUCTIVITY_TABLE",
    "ice_bridge_resistance",
    "ice_bond_gate",
]


class ThermalChainError(ValueError):
    """Raised for inconsistent contact-conduction inputs."""


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ThermalChainError(f"{name} must be finite and positive")
    return value


# ---------------------------------------------------------------------------
# Hertz contact and constriction resistance
# ---------------------------------------------------------------------------
def hertz_patch_radius(force_n: float, radius_1_m: float, radius_2_m: float,
                       youngs_1_pa: float, youngs_2_pa: float,
                       poisson_1: float = 0.3, poisson_2: float = 0.3) -> float:
    """Hertz contact-patch radius ``(3 F R* / (4 E*))^(1/3)`` in metres."""
    f = _positive("force_n", force_n)
    r1 = _positive("radius_1_m", radius_1_m)
    r2 = _positive("radius_2_m", radius_2_m)
    e1 = _positive("youngs_1_pa", youngs_1_pa)
    e2 = _positive("youngs_2_pa", youngs_2_pa)
    for name, nu in (("poisson_1", poisson_1), ("poisson_2", poisson_2)):
        if not (-1.0 < float(nu) < 0.5):
            raise ThermalChainError(f"{name} must lie in (-1, 0.5)")
    r_star = 1.0 / (1.0 / r1 + 1.0 / r2)
    e_star = 1.0 / ((1.0 - poisson_1 ** 2) / e1 + (1.0 - poisson_2 ** 2) / e2)
    return (3.0 * f * r_star / (4.0 * e_star)) ** (1.0 / 3.0)


def pair_contact_resistance(k1_w_m_k: float, k2_w_m_k: float,
                            patch_radius_m: float) -> float:
    """Series constriction resistance of a pair, ``1/(4 k1 a) + 1/(4 k2 a)``."""
    k1 = _positive("k1_w_m_k", k1_w_m_k)
    k2 = _positive("k2_w_m_k", k2_w_m_k)
    a = _positive("patch_radius_m", patch_radius_m)
    return 1.0 / (4.0 * k1 * a) + 1.0 / (4.0 * k2 * a)


def graded_axis(near_length: float, near_step: float, far_length: float,
                growth: float = 1.08) -> np.ndarray:
    """Cell faces on ``[0, far_length]``: uniform to ``near_length`` then graded."""
    for name, value in (("near_length", near_length), ("near_step", near_step),
                        ("far_length", far_length), ("growth", growth)):
        _positive(name, value)
    if growth <= 1.0:
        raise ThermalChainError("growth must exceed 1")
    if near_length >= far_length:
        raise ThermalChainError("near_length must be smaller than far_length")
    faces = [0.0]
    x = 0.0
    h = near_step
    while x < far_length * (1.0 - 1e-12):
        h = near_step if x < near_length else min(h * growth, near_step * 40.0)
        x = min(x + h, far_length)
        faces.append(x)
    return np.asarray(faces, dtype=float)


class HalfSpacePatch:
    """Axisymmetric finite-volume solution for an isothermal disk on a half space.

    Solves ``div(k grad T) = 0`` in ``(r, z)`` with ``T = 1`` on the disk
    ``r < a, z = 0``, adiabatic elsewhere on ``z = 0`` and ``T = 0`` on a distant
    outer boundary.  The resistance ``R = 1/Q`` (``Q`` = total heat flux into the
    disk) is compared with Maxwell's ``1/(4 k a)``.
    """

    def __init__(self, conductivity_w_m_k: float = 2.2,
                 patch_radius_m: float = 1.0e-4, *,
                 near_cells: float = 2.0, cells_per_radius: int = 20,
                 outer_radius: float = 60.0, outer_depth: float = 60.0,
                 growth: float = 1.1) -> None:
        self.k = _positive("conductivity_w_m_k", conductivity_w_m_k)
        self.a = _positive("patch_radius_m", patch_radius_m)
        if int(cells_per_radius) < 4:
            raise ThermalChainError("cells_per_radius must be >= 4")
        self.r_faces = graded_axis(near_cells * self.a,
                                   self.a / float(cells_per_radius),
                                   outer_radius * self.a, 1.0 + (growth - 1.0) * 0.5)
        self.z_faces = graded_axis(near_cells * self.a,
                                   self.a / float(cells_per_radius),
                                   outer_depth * self.a, 1.0 + (growth - 1.0) * 0.5)
        self.resistance_k_w = self._solve()

    def _solve(self) -> float:
        rf, zf = self.r_faces, self.z_faces
        nr, nz = len(rf) - 1, len(zf) - 1
        r_c = 0.5 * (rf[:-1] + rf[1:])
        z_c = 0.5 * (zf[:-1] + zf[1:])
        dr = np.diff(rf)
        dz = np.diff(zf)
        index = -np.ones((nr, nz), dtype=int)
        free = 0
        for i in range(nr):
            for j in range(nz):
                index[i, j] = free
                free += 1
        # outer boundary cells are Dirichlet at T = 0
        is_outer = np.zeros((nr, nz), dtype=bool)
        is_outer[-1, :] = True
        is_outer[:, -1] = True
        mat = lil_matrix((free, free))
        rhs = np.zeros(free)
        face_area_radial = 2.0 * math.pi * rf[1:-1][:, None] * dz[None, :]  # (nr-1, nz)
        g_radial = self.k * face_area_radial / np.maximum(dr[:-1, None], 1e-300)
        area_axial = math.pi * (rf[1:] ** 2 - rf[:-1] ** 2)            # (nr,)
        g_axial = self.k * area_axial[:, None] / np.maximum(dz[None, :-1], 1e-300)
        # diagonal + radial links
        for i in range(nr):
            for j in range(nz):
                p = index[i, j]
                if is_outer[i, j]:
                    mat[p, p] = 1.0
                    continue
                diag = 0.0
                if i > 0:
                    g = g_radial[i - 1, j]
                    diag += g
                    if is_outer[i - 1, j]:
                        continue
                    mat[p, index[i - 1, j]] -= g
                if i < nr - 1:
                    g = g_radial[i, j]
                    diag += g
                    if not is_outer[i + 1, j]:
                        mat[p, index[i + 1, j]] -= g
                if j > 0:
                    g = g_axial[i, j - 1]
                    diag += g
                    if not is_outer[i, j - 1]:
                        mat[p, index[i, j - 1]] -= g
                if j < nz - 1:
                    g = g_axial[i, j]
                    diag += g
                    if not is_outer[i, j + 1]:
                        mat[p, index[i, j + 1]] -= g
                mat[p, p] = diag
        # the only non-homogeneous boundary condition: the disk at z = 0
        patch = [i for i in range(nr) if r_c[i] < self.a]
        if not patch:
            raise ThermalChainError("the mesh does not resolve the contact patch")
        g_face = self.k * area_axial / (0.5 * dz[0])
        flux_weights = {}
        for i in patch:
            p = index[i, 0]
            g = g_face[i]
            mat[p, p] += g
            rhs[p] += g * 1.0
            flux_weights[p] = g
        temperature = spsolve(mat.tocsr(), rhs)
        flux = 0.0
        for p, g in flux_weights.items():
            flux += g * (1.0 - temperature[p])
        if flux <= 0.0:
            raise ThermalChainError("non-positive heat flux from the patch")
        return 1.0 / flux

    @property
    def analytic_resistance_k_w(self) -> float:
        """Maxwell spreading resistance ``1/(4 k a)``."""
        return 1.0 / (4.0 * self.k * self.a)

    @property
    def relative_error(self) -> float:
        return self.resistance_k_w / self.analytic_resistance_k_w - 1.0


# ---------------------------------------------------------------------------
# 1D particle chain: continuum FVM reference and lumped DEM network
# ---------------------------------------------------------------------------
def grain_area(radius_m: float) -> float:
    """Cross-section of the sphere-equivalent cylinder, ``2/3 pi R^2``.

    A DEM particle of radius ``R`` carries the heat capacity of a sphere,
    ``rho c_p 4/3 pi R^3``.  The 1D comparison rod therefore uses a cylinder of
    length ``2R`` with the same volume, i.e. ``A = 2/3 pi R^2``.  Using the same
    area for the bulk resistance keeps capacity and series resistance consistent
    between the lumped network and the continuum reference.
    """
    r = _positive("radius_m", radius_m)
    return 2.0 / 3.0 * math.pi * r ** 2


def rod_temperature_fvm(n_particles: int, radius_m: float, conductivity_w_m_k: float,
                        density_kg_m3: float, cp_j_kg_k: float,
                        contact_resistance_k_w: float, *,
                        left_temperature: float = 1.0, right_temperature: float = 0.0,
                        cells_per_particle: int = 24, area_m2: float | None = None,
                        time_steps: int | None = None, total_time: float | None = None,
                        ) -> dict[str, np.ndarray]:
    """Fine 1D finite-volume rod with explicit contact resistances.

    The chain occupies ``[0, N * 2R]``; particle ``p`` covers
    ``[p*2R, (p+1)*2R]`` and the interface between consecutive particles carries
    the lumped resistance ``contact_resistance_k_w``.  Cell faces therefore have
    conductance ``A / (dx/(2k) + dx/(2k) + R_c A)`` at a particle boundary and
    ``k A / dx`` inside a particle.

    Returns the cell-centre coordinates, the initial and final temperatures and
    the time histories of the two end cells (implicit Euler, energy conserving).
    """
    n = int(n_particles)
    if n < 1:
        raise ThermalChainError("a chain needs at least one particle")
    r = _positive("radius_m", radius_m)
    k = _positive("conductivity_w_m_k", conductivity_w_m_k)
    rho = _positive("density_kg_m3", density_kg_m3)
    cp = _positive("cp_j_kg_k", cp_j_kg_k)
    rc = float(contact_resistance_k_w)
    if not math.isfinite(rc) or rc < 0.0:
        raise ThermalChainError("contact_resistance_k_w must be finite and >= 0")
    m = int(cells_per_particle)
    if m < 4:
        raise ThermalChainError("cells_per_particle must be >= 4")
    area = grain_area(r) if area_m2 is None else _positive("area_m2", area_m2)
    length = 2.0 * r * n
    nc = m * n
    dx = length / nc
    x = (np.arange(nc) + 0.5) * dx
    capacity = rho * cp * area * dx
    conductance = np.full(nc - 1, k * area / dx)
    for p in range(1, n):
        conductance[p * m - 1] = area / (dx / k + rc * area)
    if total_time is None:
        alpha = k / (rho * cp)
        total_time = 0.5 * length ** 2 / alpha
    total_time = _positive("total_time", total_time)
    steps = int(time_steps) if time_steps is not None else 400
    if steps < 1:
        raise ThermalChainError("time_steps must be >= 1")
    dt = total_time / steps

    temperature = np.full(nc, right_temperature)
    temperatures = [temperature.copy()]
    left_history = []
    right_history = []
    centre_index = [int(np.argmin(np.abs(x - (p + 0.5) * 2.0 * r))) for p in range(n)]
    centre_history = [temperature[centre_index].copy()]
    for _ in range(steps):
        # Dirichlet ends through half-cell conductances
        g_left = k * area / (0.5 * dx)
        g_right = k * area / (0.5 * dx)
        ab = np.zeros((3, nc))
        ab[1] = capacity / dt
        rhs = capacity / dt * temperature
        ab[1, 0] += g_left
        ab[1, -1] += g_right
        rhs[0] += g_left * left_temperature
        rhs[-1] += g_right * right_temperature
        for f, g in enumerate(conductance):
            ab[1, f] += g
            ab[1, f + 1] += g
            ab[0, f + 1] -= g
            ab[2, f] -= g
        temperature = solve_banded((1, 1), ab, rhs)
        temperatures.append(temperature.copy())
        left_history.append(float(temperature[0]))
        right_history.append(float(temperature[-1]))
        centre_history.append(temperature[centre_index].copy())
    return {
        "x": x,
        "initial": temperatures[0],
        "final": temperature,
        "left_history": np.asarray(left_history),
        "right_history": np.asarray(right_history),
        "time": np.linspace(dt, total_time, steps),
        "dt": np.asarray([dt]),
        "conductance": conductance,
        "centre_history": np.asarray(centre_history),
        "centre_index": np.asarray(centre_index),
    }


def chain_steady_analytic(n_particles: int, radius_m: float,
                          conductivity_w_m_k: float, contact_resistance_k_w: float,
                          *, left_temperature: float = 1.0,
                          right_temperature: float = 0.0,
                          area_m2: float | None = None,
                          samples: int = 401) -> dict[str, np.ndarray]:
    """Exact steady state of the rod: linear in each grain, jump at each contact."""
    n = int(n_particles)
    r = _positive("radius_m", radius_m)
    k = _positive("conductivity_w_m_k", conductivity_w_m_k)
    rc = float(contact_resistance_k_w)
    area = grain_area(r) if area_m2 is None else _positive("area_m2", area_m2)
    grain = 2.0 * r / (k * area)
    total = n * grain + (n - 1) * rc
    q = (left_temperature - right_temperature) / total
    length = 2.0 * r * n
    x = np.linspace(0.0, length, int(samples))
    temperature = np.empty_like(x)
    for idx, xi in enumerate(x):
        p = min(int(xi / (2.0 * r)), n - 1)
        local = xi - p * 2.0 * r
        temperature[idx] = left_temperature - q * (p * (grain + rc) + local / (k * area))
    return {"x": x, "temperature": temperature, "flux": np.asarray([q]),
            "total_resistance": np.asarray([total])}


def chain_lumped_steady(n_particles: int, radius_m: float,
                        conductivity_w_m_k: float, density_kg_m3: float,
                        cp_j_kg_k: float, contact_resistance_k_w: float,
                        *, left_temperature: float = 1.0,
                        right_temperature: float = 0.0,
                        area_m2: float | None = None,
                        steps: int = 6000) -> np.ndarray:
    """Steady node temperatures of the lumped DEM particle thermal network.

    The network link carries the contact resistance *plus* the bulk resistance of
    one grain, because one grain separates two temperature nodes; the two end
    nodes are pinned (Dirichlet) as in the continuum protocol.
    """
    from .conduction import Particle, ParticleThermalNetwork

    n = int(n_particles)
    if n < 2:
        raise ThermalChainError("a chain needs at least two particles")
    area = grain_area(radius_m) if area_m2 is None else _positive("area_m2", area_m2)
    particles = [Particle(radius_m=radius_m, k_w_m_k=conductivity_w_m_k,
                          rho_kg_m3=density_kg_m3, cp_j_kg_k=cp_j_kg_k)
                 for _ in range(n)]
    network = ParticleThermalNetwork(particles, [(i, i + 1) for i in range(n - 1)])
    grain = 2.0 * radius_m / (conductivity_w_m_k * area)
    network.contact_resistance = [float(contact_resistance_k_w) + grain] * (n - 1)
    network.reset([right_temperature] * n)
    capacity = float(np.mean(network.capacity))
    dt = 0.5 * capacity * (float(contact_resistance_k_w) + grain)
    h_end = np.zeros(n)
    ambient = np.zeros(n)
    h_end[0] = h_end[-1] = 1.0 / (0.5 * grain)
    ambient[0] = left_temperature
    ambient[-1] = right_temperature
    for _ in range(int(steps)):
        network.step(dt, ambient=ambient, h_ambient_w_k=h_end)
    return network.temperature.copy()


def chain_transient_lumped(n_particles: int, radius_m: float,
                           conductivity_w_m_k: float, density_kg_m3: float,
                           cp_j_kg_k: float, contact_resistance_k_w: float,
                           *, total_time: float, steps: int = 400,
                           left_temperature: float = 1.0,
                           right_temperature: float = 0.0) -> dict[str, np.ndarray]:
    """Transient lumped network with Dirichlet ends (same protocol as the FVM)."""
    from .conduction import Particle, ParticleThermalNetwork

    n = int(n_particles)
    area = grain_area(radius_m)
    particles = [Particle(radius_m=radius_m, k_w_m_k=conductivity_w_m_k,
                          rho_kg_m3=density_kg_m3, cp_j_kg_k=cp_j_kg_k)
                 for _ in range(n)]
    contacts = [(i, i + 1) for i in range(n - 1)]
    network = ParticleThermalNetwork(particles, contacts)
    grain = 2.0 * radius_m / (conductivity_w_m_k * area)
    network.contact_resistance = [float(contact_resistance_k_w) + grain] * (n - 1)
    network.reset([right_temperature] * n)
    dt = total_time / int(steps)
    h_ambient = np.zeros(n)
    ambient = np.zeros(n)
    h_ambient[0] = h_ambient[-1] = 1.0 / (0.5 * grain)
    ambient[0] = left_temperature
    ambient[-1] = right_temperature
    left, right = [], []
    node_history = [network.temperature.copy()]
    for _ in range(int(steps)):
        network.step(dt, ambient=ambient, h_ambient_w_k=h_ambient)
        left.append(float(network.temperature[0]))
        right.append(float(network.temperature[-1]))
        node_history.append(network.temperature.copy())
    return {"temperature": network.temperature.copy(),
            "left_history": np.asarray(left), "right_history": np.asarray(right),
            "node_history": np.asarray(node_history),
            "time": np.linspace(dt, total_time, int(steps))}


def chain_transient_fvm(n_particles: int, radius_m: float, conductivity_w_m_k: float,
                        density_kg_m3: float, cp_j_kg_k: float,
                        contact_resistance_k_w: float, *, total_time: float,
                        steps: int = 400, cells_per_particle: int = 24,
                        left_temperature: float = 1.0,
                        right_temperature: float = 0.0) -> dict[str, np.ndarray]:
    """Thin wrapper: transient rod FVM with the same sampling as the lumped run."""
    return rod_temperature_fvm(
        n_particles, radius_m, conductivity_w_m_k, density_kg_m3, cp_j_kg_k,
        contact_resistance_k_w, left_temperature=left_temperature,
        right_temperature=right_temperature, cells_per_particle=cells_per_particle,
        time_steps=steps, total_time=total_time)


# ---------------------------------------------------------------------------
# Ice bond: k_ice(T) and the thermal-mechanical gate
# ---------------------------------------------------------------------------
#: Conductivity of Ih ice in W/(m K) from Petrenko & Whitworth (1999), Table 9.1
#: (also Slack 1980): weakly increasing as the ice cools.
ICE_CONDUCTIVITY_TABLE: dict[float, float] = {
    0.0: 2.22,
    -5.0: 2.24,
    -10.0: 2.25,
    -20.0: 2.28,
    -30.0: 2.34,
    -40.0: 2.39,
}


def ice_conductivity(temperature_c: float, *, k0: float = 2.225,
                     slope_per_k: float = 1.3e-3) -> float:
    """Ice conductivity ``k_ice(T) = k0 (1 - slope * T)`` with ``T`` in degC.

    The linear form reproduces the tabulated Ih values in
    :data:`ICE_CONDUCTIVITY_TABLE` to about 2 % between 0 and -40 degC; the
    thermal-mechanical gate uses it so that the interface conductance of an ice
    bond follows the temperature of the ice instead of a frozen constant.
    """
    t = float(temperature_c)
    if not math.isfinite(t):
        raise ThermalChainError("temperature_c must be finite")
    if t > 0.5:
        raise ThermalChainError("the ice phase does not exist above 0 degC")
    k = _positive("k0", k0) * (1.0 - slope_per_k * t)
    if k <= 0.0:
        raise ThermalChainError("ice conductivity became non-positive")
    return k


def ice_bridge_resistance(temperature_c: float, area_m2: float,
                          length_m: float) -> float:
    """Conduction resistance ``L / (k_ice(T) A)`` of an ice bridge (K/W)."""
    a = _positive("area_m2", area_m2)
    l = _positive("length_m", length_m)
    return l / (ice_conductivity(temperature_c) * a)


def ice_bond_gate(*, force_n: float, radius_m: float, youngs_pa: float = 9.0e9,
                  poisson: float = 0.33, k_particle_w_m_k: float = 2.2,
                  temperature_c: float = -10.0, bond_area_fraction: float = 0.25,
                  bond_length_m: float | None = None,
                  mesh_cells_per_radius: int = 14) -> dict[str, float]:
    """Thermal-mechanical gate of an ice-bonded particle pair.

    Three independent routes are checked, all sharing the *mechanical* state
    (normal force -> Hertz patch radius) so that mechanics and heat transfer are
    coupled instead of fitted separately:

    ``dry``    analytic constriction resistance ``1/(2 k a)`` of the pair;
    ``mesh``   the axisymmetric finite-volume half-space solution of
               :class:`HalfSpacePatch` for the same patch radius (criterion a);
    ``ice``    the ice bridge (sintered neck of radius ``bond_area_fraction * R``,
               area ``pi (bond_area_fraction R)^2``) of length ``2 R`` whose
               conductivity follows ``k_ice(T)``; its resistance is
               verified against a one-grain finite-volume rod solution of the
               same slab (``L/(k A)``).

    The pair resistance is the parallel combination of the two paths; the gate
    passes when the mesh matches the analytic constriction to 10 %, the bridge
    matches the slab solution to 10 %, the parallel value lies in
    ``[min(R)/2, min(R))`` as a parallel combination must, and the ice path
    carries more heat than the dry path at the requested temperature
    (bond-dominated contact).
    """
    f = _positive("force_n", force_n)
    r = _positive("radius_m", radius_m)
    e = _positive("youngs_pa", youngs_pa)
    kp = _positive("k_particle_w_m_k", k_particle_w_m_k)
    frac = float(bond_area_fraction)
    if not (0.0 < frac <= 1.0):
        raise ThermalChainError("bond_area_fraction must lie in (0, 1]")
    a = hertz_patch_radius(f, r, r, e, e, poisson, poisson)
    r_dry = pair_contact_resistance(kp, kp, a)
    mesh = HalfSpacePatch(kp, a, cells_per_radius=int(mesh_cells_per_radius),
                          outer_radius=60.0, outer_depth=60.0)
    r_mesh_pair = 2.0 * mesh.resistance_k_w

    k_ice = ice_conductivity(temperature_c)
    # an ice bond is a sintered neck at the contact, not a patch-scale film: its
    # radius scales with the grain (``bond_area_fraction * R``), which is what
    # makes an ice-bonded contact conduction dominated by the ice path.
    bridge_area = math.pi * (frac * r) ** 2
    length = 2.0 * r if bond_length_m is None else _positive("bond_length_m", bond_length_m)
    r_ice = ice_bridge_resistance(temperature_c, bridge_area, length)
    cells = 32
    slab = rod_temperature_fvm(1, 0.5 * length, k_ice, 917.0, 2100.0, 0.0,
                               cells_per_particle=cells, time_steps=200,
                               total_time=1.0e9, area_m2=bridge_area,
                               left_temperature=1.0, right_temperature=0.0)
    dx_slab = length / cells
    g_left = k_ice * bridge_area / (0.5 * dx_slab)
    flux = g_left * (1.0 - float(slab["final"][0]))
    r_ice_fvm = 1.0 / flux
    r_parallel = 1.0 / (1.0 / r_dry + 1.0 / r_ice)
    mesh_error = r_mesh_pair / r_dry - 1.0
    ice_error = r_ice_fvm / r_ice - 1.0
    return {
        "patch_radius_m": a,
        "contact_resistance_k_w": r_dry,
        "mesh_contact_resistance_k_w": r_mesh_pair,
        "mesh_relative_error": mesh_error,
        "ice_conductivity_w_m_k": k_ice,
        "bridge_area_m2": bridge_area,
        "bridge_length_m": length,
        "ice_bridge_resistance_k_w": r_ice,
        "ice_bridge_fvm_resistance_k_w": r_ice_fvm,
        "ice_bridge_relative_error": ice_error,
        "parallel_resistance_k_w": r_parallel,
        "ice_path_fraction": r_parallel / r_ice,
        "dry_path_fraction": r_parallel / r_dry,
        "ice_bond_dominates": bool(r_ice < r_dry),
        # a parallel combination is below the smaller branch and above its half
        "parallel_bounds_ok": bool(0.5 * min(r_dry, r_ice) <= r_parallel
                                   < min(r_dry, r_ice)),
        "gate_pass": bool(abs(mesh_error) < 0.10 and abs(ice_error) < 0.10
                          and 0.5 * min(r_dry, r_ice) <= r_parallel < min(r_dry, r_ice)
                          and r_ice < r_dry),
    }
