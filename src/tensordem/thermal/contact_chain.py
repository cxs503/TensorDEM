"""Particle-chain thermal conduction: Hertz constriction contacts and networks.

Physics
-------
Two elastic spheres pressed together touch over a Hertz patch of radius

    ``a_c = (3 F R_eff / (4 E*))^(1/3)``,   ``R_eff = R/2``,  ``E* = E / (2 (1-nu^2))``

for equal spheres.  Heat crosses that patch through a *constriction*: the
classical Carslaw & Jaeger spot-heating result for a circular isothermal contact
between two identical half-spaces gives the contact resistance

    ``R_c = 1 / (4 k a_c)``        (K W^-1)

while the bulk path between two opposite contacts of a sphere acts as a rod of
length ``2R`` and area ``pi R^2``:

    ``R_s = 2R / (k pi R^2) = 2 / (pi k R)``.

A chain of ``N`` spheres therefore has the exact series resistance

    ``R_chain = (N - 1) R_c + N R_s``           (closed form, :func:`chain_resistance`)

and, with ``q`` the heat flow, the steady temperature along the chain is
piecewise linear with a jump ``q R_c`` at every contact
(:func:`analytic_chain_temperature`).  Because the constriction dominates
(``R_c / R_s = pi R / (8 a_c) ~ 8`` for ``a_c / R = 0.05``) the chain effective
conductivity

    ``k_eff / k = R_s / (R_s + R_c) = 1 / (1 + pi R / (8 a_c))``

is an order of magnitude below the solid conductivity.  That is the mechanism
behind the low effective conductivity of packed ice / granular beds
(cf. ``tensordem.thermal.packed_bed``).

What the solvers in this module are for
---------------------------------------
* :class:`ThermalChain` - node-based 1D network: every sphere is ``cells_per_sphere``
  bulk resistors in series, consecutive spheres are joined by a constriction
  resistor between two coincident interface nodes.  The exact steady profile is
  representable on this mesh, so the sparse solve is graded against the closed
  form without a truncation error to hide behind (the residual that remains is
  solver round-off only).
* :func:`square_lattice_resistance` / :func:`lattice_network_temperatures` -
  the two-dimensional generalisation: an ``nx x ny`` square lattice of equal
  links has the exact two-face resistance ``(nx / ny) R_link`` (``ny`` parallel
  columns of ``nx`` links), which grades the 2D sparse network operator.
* :class:`TransientChain` - implicit-Euler transient of the same network.  Its
  exact discrete reference is the matrix exponential of the same operator
  (isolating the time-discretisation error, whose first-order rate is measured),
  and its slowest decay is compared with the continuum slab value
  ``tau_1 = 4 L^2 / (pi^2 alpha_eff)``.
* :func:`ice_bond_tensile_strength` / :func:`bond_failure_gate` - the
  thermal-mechanical gate: an ice bond whose creep-limited tensile strength
  decays with homologous temperature fails when the applied bond stress exceeds
  ``sigma(T_bond)``.  The bond temperature comes from the network solve, so the
  gate closes the loop "heat -> contact/bond conduction -> strength -> failure"
  and is graded by two independent routes (closed-form bisection on the
  analytic profile vs. the transient march).

Units are SI throughout (m, kg, s, W, K).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

__all__ = [
    "GAS_CONSTANT",
    "ICE_CREEP_ACTIVATION_ENERGY",
    "ICE_CREEP_EXPONENT",
    "ChainCase",
    "BondGateResult",
    "ThermalChain",
    "TransientChain",
    "analytic_chain_temperature",
    "bond_failure_gate",
    "chain_effective_conductivity",
    "chain_resistance",
    "contact_conductance",
    "contact_constriction_resistance",
    "hertz_contact_radius",
    "ice_bond_tensile_strength",
    "lattice_network_temperatures",
    "sphere_bulk_resistance",
    "square_lattice_resistance",
    "thermal_chain_calculation",
]

R_GAS = 8.31446261815324  # J mol^-1 K^-1
GAS_CONSTANT = R_GAS
#: Secondary-creep activation energy of polycrystalline ice (J/mol) in
#: Glen's law ``eps_dot = A sigma^n exp(-Q / R T)``; Q = 60 kJ/mol is the value
#: used for ice indentation/creep over -20..0 degC.
ICE_CREEP_ACTIVATION_ENERGY = 60.0e3
#: Creep stress exponent (Glen's law).
ICE_CREEP_EXPONENT = 3.0


class ChainError(ValueError):
    """Raised for unphysical chain / bond inputs."""


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ChainError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ChainError(f"{name} must be positive")
    return value


# --------------------------------------------------------------------- contact
def hertz_contact_radius(force: float, radius: float, youngs: float,
                         poisson: float = 0.3) -> float:
    """Hertz contact-patch radius of two equal elastic spheres (m).

    ``a_c = (3 F R_eff / (4 E*))^(1/3)`` with ``R_eff = R/2`` and
    ``E* = E / (2 (1 - nu^2))``.
    """
    force = _finite("force", force, positive=True)
    radius = _finite("radius", radius, positive=True)
    youngs = _finite("youngs", youngs, positive=True)
    poisson = _finite("poisson", poisson)
    if not 0.0 <= poisson < 0.5:
        raise ChainError("poisson must lie in [0, 0.5)")
    e_star = youngs / (2.0 * (1.0 - poisson**2))
    r_eff = 0.5 * radius
    return (3.0 * force * r_eff / (4.0 * e_star)) ** (1.0 / 3.0)


def contact_constriction_resistance(contact_radius: float, conductivity: float) -> float:
    """Constriction resistance of a circular contact between two half-spaces.

    Carslaw & Jaeger spot-heating result ``R_c = 1 / (4 k a_c)`` (K/W).
    """
    a = _finite("contact_radius", contact_radius, positive=True)
    k = _finite("conductivity", conductivity, positive=True)
    return 1.0 / (4.0 * k * a)


def contact_conductance(contact_radius: float, conductivity: float) -> float:
    """Inverse of :func:`contact_constriction_resistance` (W/K)."""
    return 1.0 / contact_constriction_resistance(contact_radius, conductivity)


def sphere_bulk_resistance(radius: float, conductivity: float) -> float:
    """Bulk (rod) resistance between two opposite contacts of a sphere (K/W).

    ``R_s = 2R / (k pi R^2)``: length ``2R``, cross-section ``pi R^2``.
    """
    r = _finite("radius", radius, positive=True)
    k = _finite("conductivity", conductivity, positive=True)
    return 2.0 * r / (k * math.pi * r * r)


def chain_resistance(n_spheres: int, contact_radius: float, radius: float,
                     conductivity: float) -> float:
    """Exact series resistance of an ``n_spheres`` chain (K/W)."""
    n = int(n_spheres)
    if n < 2:
        raise ChainError("a chain needs at least two spheres")
    r_c = contact_constriction_resistance(contact_radius, conductivity)
    r_s = sphere_bulk_resistance(radius, conductivity)
    return (n - 1) * r_c + n * r_s


def chain_effective_conductivity(n_spheres: int, contact_radius: float, radius: float,
                                 conductivity: float) -> float:
    """``k_eff`` of the chain treated as a uniform rod of area ``pi R^2`` (W/m/K)."""
    n = int(n_spheres)
    r_total = chain_resistance(n, contact_radius, radius, conductivity)
    length = 2.0 * n * radius
    area = math.pi * radius * radius
    return length / (r_total * area)


def analytic_chain_temperature(n_spheres: int, contact_radius: float, radius: float,
                               conductivity: float, cells_per_sphere: int,
                               t_hot: float, t_cold: float) -> np.ndarray:
    """Closed-form steady node temperatures of the chain network (K).

    Nodes are ordered sphere by sphere: for sphere ``i`` the ``cells_per_sphere + 1``
    nodes ``i*(m+1) .. i*(m+1)+m``; the two interface nodes of a contact are
    adjacent in that ordering and are joined by the constriction resistor.  The
    profile drops ``q R_s / m`` per bulk link and ``q R_c`` per contact, with
    ``q = (t_hot - t_cold) / R_chain``.
    """
    n = int(n_spheres)
    m = int(cells_per_sphere)
    if n < 2:
        raise ChainError("a chain needs at least two spheres")
    if m < 1:
        raise ChainError("cells_per_sphere must be >= 1")
    r_c = contact_constriction_resistance(contact_radius, conductivity)
    r_s = sphere_bulk_resistance(radius, conductivity)
    total = (n - 1) * r_c + n * r_s
    q = (float(t_hot) - float(t_cold)) / total
    out = np.empty(n * (m + 1), dtype=np.float64)
    t = float(t_hot)
    for i in range(n):
        for j in range(m + 1):
            if i > 0 and j == 0:
                t -= q * r_c
            elif j > 0:
                t -= q * r_s / m
            out[i * (m + 1) + j] = t
    return out


def square_lattice_resistance(nx: int, ny: int, r_link: float) -> float:
    """Exact two-face resistance of an ``nx x ny`` node lattice (K/W).

    ``ny`` parallel columns of ``nx - 1`` links each:
    ``R = (nx - 1) / ny * R_link`` - the resistance between the pinned columns
    ``i = 0`` and ``i = nx - 1`` of :func:`lattice_network_temperatures`.
    """
    if int(nx) < 2 or int(ny) < 1:
        raise ChainError("nx must be >= 2 and ny >= 1")
    r = _finite("r_link", r_link, positive=True)
    return (int(nx) - 1) / int(ny) * r


def lattice_network_temperatures(nx: int, ny: int, r_link: float,
                                 t_left: float, t_right: float) -> np.ndarray:
    """Solve the ``nx x ny`` lattice with both x-faces pinned (returns ``(nx, ny)``)."""
    if int(nx) < 2:
        raise ChainError("nx must be >= 2")
    if int(ny) < 1:
        raise ChainError("ny must be >= 1")
    g = 1.0 / _finite("r_link", r_link, positive=True)
    n_cells = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(n_cells)
    for i in range(nx):
        for j in range(ny):
            node = i * ny + j
            if i == 0:
                rows.append(node); cols.append(node); vals.append(1.0)
                rhs[node] = float(t_left)
                continue
            if i == nx - 1:
                rows.append(node); cols.append(node); vals.append(1.0)
                rhs[node] = float(t_right)
                continue
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if not (0 <= ni < nx) or not (0 <= nj < ny):
                    continue
                rows.append(node); cols.append(node); vals.append(g)
                rows.append(node); cols.append(ni * ny + nj); vals.append(-g)
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    t = sp.linalg.spsolve(mat, rhs)
    return t.reshape(nx, ny)


# ------------------------------------------------------------------- chain FVM
@dataclass(frozen=True)
class ChainCase:
    """A 1D chain of equal ice spheres loaded by a constant contact force."""

    n_spheres: int = 8
    radius: float = 0.005
    conductivity: float = 2.2          # ice, W/m/K
    density: float = 917.0             # kg/m^3
    specific_heat: float = 2050.0      # J/kg/K
    youngs: float = 9.0e9              # Pa
    poisson: float = 0.31
    contact_force: float = 20.0        # N per contact
    cells_per_sphere: int = 4
    t_hot: float = 273.15 - 1.0
    t_cold: float = 273.15 - 21.0

    def __post_init__(self) -> None:
        if int(self.n_spheres) < 2:
            raise ChainError("n_spheres must be >= 2")
        if int(self.cells_per_sphere) < 1:
            raise ChainError("cells_per_sphere must be >= 1")
        for name in ("radius", "conductivity", "density", "specific_heat",
                     "youngs", "contact_force"):
            _finite(name, getattr(self, name), positive=True)
        _finite("poisson", self.poisson)
        _finite("t_hot", self.t_hot)
        _finite("t_cold", self.t_cold)
        if self.t_hot <= self.t_cold:
            raise ChainError("t_hot must exceed t_cold")

    # -------------------------------------------------------------- geometry
    @property
    def contact_radius(self) -> float:
        return hertz_contact_radius(self.contact_force, self.radius, self.youngs,
                                    self.poisson)

    @property
    def resistance(self) -> float:
        return chain_resistance(self.n_spheres, self.contact_radius, self.radius,
                                self.conductivity)

    @property
    def heat_flow(self) -> float:
        return (self.t_hot - self.t_cold) / self.resistance

    @property
    def effective_conductivity(self) -> float:
        return chain_effective_conductivity(self.n_spheres, self.contact_radius,
                                            self.radius, self.conductivity)

    @property
    def length(self) -> float:
        return 2.0 * self.n_spheres * self.radius

    @property
    def area(self) -> float:
        return math.pi * self.radius * self.radius

    @property
    def conductivity_ratio(self) -> float:
        """``k_eff / k`` predicted by the constriction-limited series model."""
        r_c = contact_constriction_resistance(self.contact_radius, self.conductivity)
        r_s = sphere_bulk_resistance(self.radius, self.conductivity)
        return r_s / (r_s + r_c)


class ThermalChain:
    """Node-based 1D chain network: bulk links inside spheres, constriction links between."""

    def __init__(self, case: ChainCase | None = None) -> None:
        self.case = case or ChainCase()
        c = self.case
        self.n = c.n_spheres
        self.m = c.cells_per_sphere
        self.n_nodes = self.n * (self.m + 1)
        self.bulk_g = 1.0 / (sphere_bulk_resistance(c.radius, c.conductivity) / self.m)
        self.contact_g = contact_conductance(c.contact_radius, c.conductivity)
        g = np.empty(self.n_nodes - 1, dtype=np.float64)
        for i in range(self.n):
            base = i * (self.m + 1)
            g[base:base + self.m] = self.bulk_g
        for i in range(self.n - 1):
            g[i * (self.m + 1) + self.m] = self.contact_g
        self.link_conductance = g
        #: contact links are the last node of sphere ``i`` -> first of sphere ``i+1``
        self.contact_links = np.asarray(
            [i * (self.m + 1) + self.m for i in range(self.n - 1)], dtype=np.int64)
        sphere_volume = 4.0 / 3.0 * math.pi * c.radius**3
        self.capacity = c.density * c.specific_heat * sphere_volume / (self.m + 1)

    # ------------------------------------------------------------- operators
    def conduction_matrix(self) -> sp.csr_matrix:
        """Sparse Laplacian of the network (both end nodes included)."""
        n_nodes = self.n_nodes
        g = self.link_conductance
        main = np.zeros(n_nodes)
        main[:-1] += g
        main[1:] += g
        off = -g
        return sp.diags([off, main, off], offsets=[-1, 0, 1],
                        format="csr", shape=(n_nodes, n_nodes))

    def _dirichlet_matrix(self) -> sp.csr_matrix:
        mat = self.conduction_matrix().tolil()
        mat[0, :] = 0.0
        mat[-1, :] = 0.0
        mat[0, 0] = 1.0
        mat[-1, -1] = 1.0
        return mat.tocsr()

    def solve_steady(self) -> np.ndarray:
        """Solve the network with the two end nodes pinned to ``t_hot`` / ``t_cold``."""
        c = self.case
        rhs = np.zeros(self.n_nodes)
        rhs[0] = c.t_hot
        rhs[-1] = c.t_cold
        return sp.linalg.spsolve(self._dirichlet_matrix(), rhs)

    # -------------------------------------------------------------- grading
    def analytic_steady(self) -> np.ndarray:
        c = self.case
        return analytic_chain_temperature(c.n_spheres, c.contact_radius, c.radius,
                                          c.conductivity, self.m, c.t_hot, c.t_cold)

    def measured_heat_flow(self, temperature: np.ndarray) -> float:
        """Heat flow taken from the first bulk link of the solved field (W)."""
        t = np.asarray(temperature, dtype=np.float64)
        return float(self.link_conductance[0] * (t[0] - t[1]))

    def measured_effective_conductivity(self, temperature: np.ndarray) -> float:
        t = np.asarray(temperature, dtype=np.float64)
        q = self.measured_heat_flow(t)
        return float(q * self.case.length / (self.case.area * (t[0] - t[-1])))

    def contact_jump(self, temperature: np.ndarray, contact_index: int = 0) -> float:
        """Temperature drop realised across contact ``contact_index`` (K)."""
        t = np.asarray(temperature, dtype=np.float64)
        i = int(self.contact_links[int(contact_index)])
        return float(t[i] - t[i + 1])

    def sphere_temperature(self, temperature: np.ndarray, sphere_index: int) -> float:
        """Mean temperature of one sphere (K)."""
        t = np.asarray(temperature, dtype=np.float64)
        s = int(sphere_index)
        base = s * (self.m + 1)
        return float(t[base:base + self.m + 1].mean())


# ---------------------------------------------------------------- transient
class TransientChain(ThermalChain):
    """Implicit-Euler transient of :class:`ThermalChain`."""

    def _factorisation(self, dt: float):
        """Cached LU of ``C/dt + G`` (the operator does not change with ``t``)."""
        cache = self.__dict__.setdefault("_lu_cache", {})
        key = float(dt)
        if key not in cache:
            g = self.conduction_matrix()
            cap = sp.identity(self.n_nodes, format="csr") * float(self.capacity)
            lhs = (cap / dt + g).tolil()
            lhs[0, :] = 0.0
            lhs[-1, :] = 0.0
            lhs[0, 0] = 1.0
            lhs[-1, -1] = 1.0
            from scipy.sparse.linalg import splu

            cache[key] = (splu(lhs.tocsc()), float(self.capacity) / dt)
        return cache[key]

    def step(self, temperature: np.ndarray, dt: float, *, t_hot: float,
             t_cold: float) -> np.ndarray:
        """One implicit-Euler step with the Dirichlet ends re-imposed."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ChainError("dt must be finite and positive")
        t = np.asarray(temperature, dtype=np.float64)
        lu, coeff = self._factorisation(dt)
        rhs = coeff * t
        rhs[0] = float(t_hot)
        rhs[-1] = float(t_cold)
        return lu.solve(rhs)

    def initial_state(self) -> np.ndarray:
        c = self.case
        t = np.full(self.n_nodes, c.t_cold)
        t[0], t[-1] = c.t_hot, c.t_cold
        return t

    def march(self, dt: float, n_steps: int, *, initial: np.ndarray | None = None,
              record_every: int = 1) -> dict[str, np.ndarray]:
        """March in time; returns ``time``, node history and the hot-end flux."""
        c = self.case
        t = self.initial_state() if initial is None else np.asarray(
            initial, dtype=np.float64).copy()
        times = [0.0]
        fields = [t.copy()]
        flux = [float(self.link_conductance[0] * (t[0] - t[1]))]
        every = max(int(record_every), 1)
        for step in range(int(n_steps)):
            t = self.step(t, dt, t_hot=c.t_hot, t_cold=c.t_cold)
            if (step + 1) % every == 0:
                times.append((step + 1) * dt)
                fields.append(t.copy())
                flux.append(float(self.link_conductance[0] * (t[0] - t[1])))
        return {"time": np.asarray(times), "nodes": np.asarray(fields),
                "hot_flux": np.asarray(flux)}

    def _interior_operator(self) -> np.ndarray:
        """``dT/dt = M T`` restricted to the interior nodes (Dirichlet ends pinned)."""
        g_full = self.conduction_matrix().toarray()
        return -g_full[1:-1, 1:-1] / float(self.capacity)

    def analytic_transient(self, times: Sequence[float]) -> np.ndarray:
        """Exact discrete solution (matrix exponential) of the same RC network.

        The Dirichlet end nodes are eliminated, so the reference is the exact
        solution of the *sampled* operator: comparing it with :meth:`march`
        isolates the implicit-Euler time error from any spatial error.
        """
        c = self.case
        a_int = self._interior_operator()
        steady = self.solve_steady()
        t0 = self.initial_state()
        dev0 = t0[1:-1] - steady[1:-1]
        out = np.empty((len(times), self.n_nodes))
        for idx, tt in enumerate(times):
            out[idx, 0] = c.t_hot
            out[idx, -1] = c.t_cold
            out[idx, 1:-1] = steady[1:-1] + sla.expm(a_int * float(tt)) @ dev0
        return out

    def relaxation_time(self) -> float:
        """Slowest discrete time constant of the RC network (s)."""
        eig = np.linalg.eigvals(self._interior_operator())
        return float(-1.0 / np.max(np.real(eig)))

    def continuum_relaxation_time(self) -> float:
        """Continuum slab relaxation ``tau_1 = L^2 / (pi^2 alpha_eff)`` (s).

        ``alpha_eff = k_eff / (rho c_p)_packed`` with the volumetric heat capacity
        of the *chain* (sphere volume per cell volume), so a sphere chain is
        compared with the equivalent uniform slab.
        """
        c = self.case
        packed_heat_capacity = float(self.capacity) * self.n_nodes / (c.area * c.length)
        return c.length**2 * packed_heat_capacity / (math.pi**2 * c.effective_conductivity)


# ------------------------------------------------------------- ice-bond gate
def ice_bond_tensile_strength(temperature_kelvin: float, *,
                              sigma_ref: float = 1.0e6,
                              t_ref: float = 263.15,
                              activation_energy: float = ICE_CREEP_ACTIVATION_ENERGY,
                              stress_exponent: float = ICE_CREEP_EXPONENT) -> float:
    """Creep-limited tensile strength of an ice bond (Pa).

    Glen's law ``eps_dot = A sigma^n exp(-Q / R T)`` inverted at fixed strain
    rate gives ``sigma(T) = sigma_ref exp[(Q / (n R)) (1/T - 1/T_ref)]``: the
    bond is *weaker* at higher (homologous) temperature - about 30 % lower at
    0 degC than at -10 degC for Q = 60 kJ/mol, n = 3.  ``sigma_ref = 1 MPa`` is
    the adhesive ice-strength scale used for ice-structure bonds
    (0.5-1.5 MPa in the ice-adhesion literature).  The absolute level is a
    CALIBRATED scale; only the temperature *shape* is used by the gate.
    """
    t = _finite("temperature_kelvin", temperature_kelvin, positive=True)
    s = _finite("sigma_ref", sigma_ref, positive=True)
    tr = _finite("t_ref", t_ref, positive=True)
    q = _finite("activation_energy", activation_energy, positive=True)
    n = _finite("stress_exponent", stress_exponent, positive=True)
    return s * math.exp((q / (n * R_GAS)) * (1.0 / t - 1.0 / tr))


def bond_temperature_analytic(case: ChainCase, bond_index: int, t_hot: float) -> float:
    """Closed-form temperature at contact ``bond_index`` for a hot end ``t_hot`` (K)."""
    r_c = contact_constriction_resistance(case.contact_radius, case.conductivity)
    r_s = sphere_bulk_resistance(case.radius, case.conductivity)
    total = (case.n_spheres - 1) * r_c + case.n_spheres * r_s
    up = bond_index * (r_c + r_s) + 0.5 * r_s  # to the bond, at the sphere centre
    return float(t_hot - (t_hot - case.t_cold) * up / total)


@dataclass(frozen=True)
class BondGateResult:
    """Outcome of the ice-bond thermal-mechanical gate."""

    bond_index: int
    bond_area: float
    applied_stress: float
    strength_cold: float
    strength_critical: float
    t_bond_cold: float
    t_bond_critical: float
    t_hot_critical: float
    hot_end_rise: float
    simulated_failure_time: float
    analytic_failure_time: float
    failure_time_error: float
    fails: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "bond_index": self.bond_index,
            "bond_area_m2": self.bond_area,
            "applied_stress_pa": self.applied_stress,
            "strength_cold_pa": self.strength_cold,
            "strength_critical_pa": self.strength_critical,
            "t_bond_cold_k": self.t_bond_cold,
            "t_bond_critical_k": self.t_bond_critical,
            "t_hot_critical_k": self.t_hot_critical,
            "hot_end_rise_k": self.hot_end_rise,
            "simulated_failure_time_s": self.simulated_failure_time,
            "analytic_failure_time_s": self.analytic_failure_time,
            "failure_time_rel_error": self.failure_time_error,
            "fails": self.fails,
        }


def bond_failure_gate(case: ChainCase, *, bond_index: int = 1, bond_radius: float = 1e-4,
                      applied_force: float = 0.02, dt: float | None = None,
                      n_steps: int = 12000) -> BondGateResult:
    """Thermal-mechanical gate: does heating the chain break the ice bond?

    The bond sits at contact ``bond_index``; its tensile stress is
    ``applied_force / (pi r_bond^2)`` and its capacity is
    ``ice_bond_tensile_strength(T_bond)``.  Two independent routes are reported:

    * *analytic* - bisection on the closed-form profile
      :func:`bond_temperature_analytic` for the hot-end temperature at which
      capacity equals load, plus the exact RC transient (matrix exponential) for
      the failure time;
    * *simulated* - implicit-Euler :class:`TransientChain` march with the hot end
      pinned at ``case.t_hot``; failure is the first step at which the capacity
      drops below the applied stress.

    ``failure_time_rel_error`` is the relative difference of the two times.
    """
    bond_index = int(bond_index)
    if not 0 <= bond_index < case.n_spheres - 1:
        raise ChainError("bond_index out of range")
    bond_radius = _finite("bond_radius", bond_radius, positive=True)
    applied_force = _finite("applied_force", applied_force, positive=True)
    area = math.pi * bond_radius * bond_radius
    stress = applied_force / area

    t_bond_cold = bond_temperature_analytic(case, bond_index, case.t_cold)
    strength_cold = ice_bond_tensile_strength(t_bond_cold)
    if stress >= strength_cold:
        t_crit = t_bond_cold
    else:
        lo, hi = t_bond_cold, 273.15
        if ice_bond_tensile_strength(hi) > stress:
            t_crit = hi  # cannot be broken by heating inside the physical range
        else:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if ice_bond_tensile_strength(mid) > stress:
                    lo = mid
                else:
                    hi = mid
            t_crit = 0.5 * (lo + hi)
    strength_crit = ice_bond_tensile_strength(t_crit)
    share = ((bond_temperature_analytic(case, bond_index, case.t_hot) - case.t_cold)
             / (case.t_hot - case.t_cold))
    t_hot_crit = case.t_cold + (t_crit - case.t_cold) / share if share > 0 else math.inf

    chain = TransientChain(case)
    node = bond_index * (chain.m + 1) + chain.m  # interface node upstream of the bond
    analytic_time = math.inf
    if case.t_hot > t_hot_crit:
        tau = chain.relaxation_time()
        grid = np.geomspace(1e-7 * tau, 30.0 * tau, 400)
        hist = chain.analytic_transient(grid)[:, node]
        crossing = np.nonzero(hist >= t_crit)[0]
        if len(crossing):
            analytic_time = float(grid[crossing[0]])

    tau_sim = chain.relaxation_time()
    if dt is None:
        dt = tau_sim / 2000.0
    t = chain.initial_state()
    simulated_time = math.inf
    for step in range(int(n_steps)):
        t = chain.step(t, dt, t_hot=case.t_hot, t_cold=case.t_cold)
        if ice_bond_tensile_strength(t[node]) <= stress:
            simulated_time = (step + 1) * dt
            break

    if math.isfinite(analytic_time) and math.isfinite(simulated_time):
        err = abs(simulated_time - analytic_time) / analytic_time
    else:
        err = 0.0 if analytic_time == simulated_time else math.inf
    return BondGateResult(
        bond_index=bond_index,
        bond_area=area,
        applied_stress=stress,
        strength_cold=strength_cold,
        strength_critical=strength_crit,
        t_bond_cold=t_bond_cold,
        t_bond_critical=t_crit,
        t_hot_critical=t_hot_crit,
        hot_end_rise=t_hot_crit - case.t_cold,
        simulated_failure_time=simulated_time,
        analytic_failure_time=analytic_time,
        failure_time_error=err,
        fails=math.isfinite(simulated_time),
    )

# ---------------------------------------------------------------------- API
def thermal_chain_calculation(request: dict) -> dict:
    """API contract for ``POST /api/v1/calculations/dem/thermal``.

    Request (all keys optional)::

        {"n_spheres": 8, "radius": 0.005, "conductivity": 2.2,
         "contact_force": 20.0, "cells_per_sphere": 4,
         "t_hot": 272.15, "t_cold": 252.15,
         "mode": "steady" | "transient" | "bond",
         "bond_radius": 1e-4, "applied_force": 0.042}

    ``steady``   network solve graded against the closed-form series chain;
    ``transient`` implicit-Euler march graded against the matrix exponential;
    ``bond``     the ice-bond thermal-mechanical gate (two routes).

    Everything returned is computed by this call; the ``capability`` field marks
    the evidence level (REAL: no hard-coded answers, every number solved).
    """
    if not isinstance(request, dict):
        raise ChainError("request must be a mapping")
    mode = str(request.get("mode", "steady"))
    if mode not in ("steady", "transient", "bond"):
        raise ChainError("mode must be 'steady', 'transient' or 'bond'")
    fields = {k: request[k] for k in
              ("n_spheres", "radius", "conductivity", "density", "specific_heat",
               "youngs", "poisson", "contact_force", "cells_per_sphere",
               "t_hot", "t_cold") if k in request}
    case = ChainCase(**fields)
    payload: dict[str, object] = {
        "endpoint": "/api/v1/calculations/dem/thermal",
        "mode": mode,
        "capability": "REAL",
        "case": {k: getattr(case, k) for k in
                 ("n_spheres", "radius", "conductivity", "density",
                  "specific_heat", "youngs", "poisson", "contact_force",
                  "cells_per_sphere", "t_hot", "t_cold")},
        "contact_law": {
            "hertz_contact_radius_m": case.contact_radius,
            "contact_constriction_resistance_k_per_w": contact_constriction_resistance(
                case.contact_radius, case.conductivity),
            "sphere_bulk_resistance_k_per_w": sphere_bulk_resistance(
                case.radius, case.conductivity),
            "chain_resistance_k_per_w": case.resistance,
            "k_eff_w_m_k": case.effective_conductivity,
            "k_eff_over_k": case.conductivity_ratio,
        },
    }
    if mode == "steady":
        chain = ThermalChain(case)
        numeric = chain.solve_steady()
        analytic = chain.analytic_steady()
        span = float(np.ptp(analytic))
        payload["steady"] = {
            "heat_flow_w": case.heat_flow,
            "measured_heat_flow_w": chain.measured_heat_flow(numeric),
            "contact_jump_k": chain.contact_jump(numeric),
            "max_abs_error_k": float(np.max(np.abs(numeric - analytic))),
            "max_rel_error": float(np.max(np.abs(numeric - analytic)) / span),
            "profile_k": numeric.tolist(),
            "analytic_k": analytic.tolist(),
        }
    elif mode == "transient":
        chain = TransientChain(case)
        tau = chain.relaxation_time()
        steps = int(request.get("steps", 3000))
        march = chain.march(dt=tau / max(steps / 3.0, 1.0), n_steps=steps,
                            record_every=max(steps // 10, 1))
        exact = chain.analytic_transient(march["time"])
        payload["transient"] = {
            "relaxation_time_s": tau,
            "continuum_relaxation_time_s": chain.continuum_relaxation_time(),
            "max_rel_error": float(np.max(np.abs(march["nodes"] - exact))
                                   / np.ptp(exact)),
            "time_s": march["time"].tolist(),
            "hot_flux_w": march["hot_flux"].tolist(),
        }
    else:
        gate = bond_failure_gate(
            case,
            bond_index=int(request.get("bond_index", 1)),
            bond_radius=float(request.get("bond_radius", 1e-4)),
            applied_force=float(request.get("applied_force", 0.042)),
            n_steps=int(request.get("n_steps", 12000)))
        payload["bond_gate"] = gate.as_dict()
    return payload
