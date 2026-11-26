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
    "affine_contact_interval",
    "constant_contact_interval",
    "harmonic_contact_interval",
    "harmonic_contact_temperatures",
    "low_frequency_contact_temperatures",
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
    sphere, node = np.divmod(np.arange(n * (m + 1)), m + 1)
    out = float(t_hot) - q * (sphere * (r_c + r_s) + (node / m) * r_s)
    out[[0, -1]] = (t_hot, t_cold)
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


def lattice_network_temperatures(nx: int, ny: int, r_link: float | tuple,
                                 t_left: float | np.ndarray, t_right: float | np.ndarray, *, nodal_power=0.0,
                                 periodic_y: bool = False, lower_exchange=None, upper_exchange=None) -> np.ndarray:
    """Solve a lattice with pinned x faces and optional transverse thermal exchange.

    r_link is a uniform resistance in K/W or a pair (Rx, Ry). Each entry
    is scalar or an array on the edges of its axis: (nx-1, ny) for Rx,
    (nx, ny-1) for Ry. Rx[i,j] connects (i,j) to (i+1,j), and Ry[i,j]
    connects (i,j) to (i,j+1). nodal_power is in W. Each pinned face
    accepts a scalar temperature in K or one temperature per transverse row.
    With periodic_y=True, Ry has shape (nx, ny) and its last column
    connects the last row to the first. Two rows then have parallel links,
    and a single row has no transverse heat transfer. For nonperiodic y,
    lower_exchange and upper_exchange optionally specify (conductance in W/K,
    ambient temperature in K) for the first and last rows. Each value may be
    scalar or a profile of length nx. Conductance includes any resistance
    between the particle center and the ambient. Zero conductance insulates
    the particle. The prescribed x-face temperatures take precedence.
    """
    if int(nx) < 2:
        raise ChainError("nx must be >= 2")
    if int(ny) < 1:
        raise ChainError("ny must be >= 1")
    def conductance(resistance, shape):
        values = np.asarray(resistance, dtype=float)
        if values.ndim and values.shape != shape:
            raise ChainError(f"edge resistance needs shape {shape}, got {values.shape}")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ChainError("edge resistance must be finite and positive")
        return np.broadcast_to(1.0 / values, shape)

    if isinstance(r_link, tuple):
        if len(r_link) != 2:
            raise ChainError("r_link needs the pair (Rx, Ry)")
        rx, ry = r_link
    else:
        rx = ry = r_link
    gx = conductance(rx, (nx - 1, ny))
    gy = conductance(ry, (nx, ny if periodic_y else ny - 1))
    n_cells = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    power = np.broadcast_to(np.asarray(nodal_power, dtype=float), (nx, ny))
    if not np.isfinite(power).all():
        raise ChainError("nodal_power must be finite")
    exchange_links = []
    rhs = power.flatten()
    for side, exchange in ((0, lower_exchange), (-1, upper_exchange)):
        if exchange is None:
            continue
        if periodic_y or not isinstance(exchange, (list, tuple)) or len(exchange) != 2:
            raise ChainError("surface exchange requires a nonperiodic face and a (conductance, ambient) pair")
        profiles = []
        for value in exchange:
            array = np.asarray(value)
            if np.iscomplexobj(array) or array.shape not in ((), (nx,)):
                raise ChainError("exchange inputs must be real scalars or axial profiles of length nx")
            profile = np.broadcast_to(np.asarray(array, dtype=float), (nx,))
            if not np.isfinite(profile).all():
                raise ChainError("exchange inputs must be finite")
            profiles.append(profile)
        g, ambient = profiles
        if np.any(g < 0):
            raise ChainError("exchange conductance must be nonnegative")
        nodes = np.arange(1, nx-1) * ny + (side % ny)
        rows.extend(nodes)
        cols.extend(nodes)
        vals.extend(g[1:-1])
        rhs[nodes] += g[1:-1] * ambient[1:-1]
        exchange_links.append((side, g, ambient))
    for i, values in ((0, t_left), (nx-1, t_right)):
        wall = np.asarray(values)
        if np.iscomplexobj(wall) or wall.shape not in ((), (ny,)):
            raise ChainError("each pinned face needs a real scalar or one temperature per row")
        wall = np.asarray(wall, dtype=float)
        if not np.isfinite(wall).all():
            raise ChainError("pinned face temperatures must be finite")
        rhs.reshape(nx, ny)[i] = wall
    for i in range(nx):
        for j in range(ny):
            node = i * ny + j
            if i in (0, nx-1):
                rows.append(node); cols.append(node); vals.append(1.0)
                continue
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if periodic_y:
                    nj %= ny
                    if ni == i and nj == j:
                        continue
                if not (0 <= ni < nx) or not (0 <= nj < ny):
                    continue
                g = gx[min(i, ni), j] if di else gy[i, j if dj == 1 else nj]
                rows.append(node); cols.append(node); vals.append(g)
                rows.append(node); cols.append(ni * ny + nj); vals.append(-g)
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    factor = sp.linalg.splu(mat.tocsc())
    temperature = factor.solve(rhs).reshape(nx, ny)
    # Refine against the physical link currents. Forming residuals from
    # differences preserves the constant mode even when diagonal sums round.
    residual = power.copy()
    for side, g, ambient in exchange_links:
        residual[:, side] += g * (ambient - temperature[:, side])
    x_heat = gx*np.diff(temperature, axis=0)
    residual[:-1] += x_heat
    residual[1:] -= x_heat
    if periodic_y:
        y_heat = gy*(np.roll(temperature, -1, axis=1)-temperature)
        residual += y_heat-np.roll(y_heat, 1, axis=1)
    else:
        y_heat = gy*np.diff(temperature, axis=1)
        residual[:, :-1] += y_heat
        residual[:, 1:] -= y_heat
    residual[[0, -1]] = rhs.reshape(nx, ny)[[0, -1]]-temperature[[0, -1]]
    return temperature + factor.solve(residual.ravel()).reshape(nx, ny)


def _chain_laplacian(conductance):
    diag = np.zeros(len(conductance) + 1)
    diag[:-1] += conductance
    diag[1:] += conductance
    return sp.diags([-conductance, diag, -conductance], [-1, 0, 1], format="csr")


def cell_centered_slab_network(widths, conductivity, volumetric_heat_capacity, *,
                               area=1., contact_resistance=0., films=(None, None), resolve_contacts=False,
                               interface_heat_capacity=0., left_resistance_fraction=.5, wall_areal_heat_capacity=(0.,0.)):
    """Build links [K/W] and capacities [J/K] for a fixed 1D material stack.

    Widths in m, k in W/(m K), volumetric capacity in J/(m³ K), area in m².
    Area may be scalar or per cell. At an abrupt step, contact resistance
    and sheet capacity use the smaller (overlapping) cross section. The
    model assumes one-dimensional axial conduction and insulated shoulders.
    Each cell stores its full capacity at its centre. Internal contact
    resistances (m² K/W) sit between adjacent half cells. The two end nodes
    are prescribed reservoirs, with zero capacity by default. A positive film coefficient
    (W/(m² K)) adds external resistance; None prescribes the surface temperature.
    With resolve_contacts=True, insert two zero-capacity surface nodes at each
    strictly positive contact resistance. Use allow_massless=True when solving.
    interface_heat_capacity (J/(m² K)) places a lumped sheet at each interface
    with positive areal capacity. left_resistance_fraction specifies the share
    of the contact resistance between the left material and the sheet (scalar
    or one per interface). At fractions 0 or 1, the sheet coincides with the
    corresponding surface node. The default splits the resistance equally.
    wall_areal_heat_capacity (J/(m² K)) places storage at either exterior
    surface. With a film, the surface becomes a separate node between the
    cell and the prescribed ambient reservoir. Without a film, capacity
    resides at the prescribed surface itself.
    """
    widths = np.asarray(widths, dtype=float)
    if widths.ndim != 1 or not widths.size or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ChainError("supply a finite positive width for each cell")
    area = np.broadcast_to(np.asarray(area,dtype=float),widths.shape)
    if not np.isfinite(area).all() or np.any(area<=0):
        raise ChainError("area must be finite and positive for every cell")
    joint_area = np.minimum(area[:-1],area[1:])
    k, cv = (np.broadcast_to(np.asarray(v, dtype=float), widths.shape)
             for v in (conductivity, volumetric_heat_capacity))
    if any(not np.isfinite(v).all() or np.any(v <= 0) for v in (k, cv)):
        raise ChainError("conductivity and volumetric heat capacity must be finite and positive")
    contacts = np.asarray(contact_resistance, dtype=float)
    if not np.isfinite(contacts).all() or np.any(contacts < 0):
        raise ChainError("contact resistance must be finite and nonnegative")
    contacts = np.broadcast_to(contacts, (widths.size - 1,))
    sheet = np.asarray(interface_heat_capacity, dtype=float)
    if not np.isfinite(sheet).all() or np.any(sheet < 0):
        raise ChainError("interface heat capacity must be finite and nonnegative")
    sheet = np.broadcast_to(sheet, contacts.shape)
    fraction = np.asarray(left_resistance_fraction, dtype=float)
    if not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1)):
        raise ChainError("left_resistance_fraction must lie in [0, 1]")
    fraction = np.broadcast_to(fraction, contacts.shape)
    left, right = films
    film_r = [0. if h is None else 1/_finite("film coefficient", h, positive=True) for h in (left, right)]
    wall_capacity = np.broadcast_to(np.asarray(wall_areal_heat_capacity,dtype=float),(2,))
    if np.any(wall_capacity<0) or not np.isfinite(wall_capacity).all():
        raise ChainError("wall areal heat capacity must be finite and nonnegative")
    storing_film = (wall_capacity>0) & (np.asarray(film_r)>0)
    half = .5*widths/(k*area)
    contacts = contacts/joint_area
    sheet = sheet*joint_area
    film_r = np.asarray(film_r)/area[[0,-1]]
    end_links = half[[0,-1]] + np.where(storing_film,0.,film_r)
    links = np.r_[end_links[0],half[:-1]+contacts+half[1:],end_links[1]]
    capacity = np.r_[0., widths*area*cv, 0.]
    if np.any(sheet > 0) or (resolve_contacts and np.any(contacts > 0)):
        resolved_links = [links[0]]
        resolved_capacity = [0., capacity[1]]
        for i, contact in enumerate(contacts):
            left_r, right_r = contact*fraction[i], contact*(1-fraction[i])
            if resolve_contacts and contact > 0:
                resolved_links.append(half[i])
                resolved_capacity.append(0.)
                if sheet[i] > 0:
                    if left_r > 0:
                        resolved_links.append(left_r)
                        resolved_capacity.append(0.)
                    resolved_capacity[-1] = sheet[i]
                    if right_r > 0:
                        resolved_links.append(right_r)
                        resolved_capacity.append(0.)
                else:
                    resolved_links.append(contact)
                    resolved_capacity.append(0.)
                resolved_links.append(half[i+1])
            elif sheet[i] > 0:
                resolved_links.extend((half[i]+left_r, half[i+1]+right_r))
                resolved_capacity.append(sheet[i])
            else:
                resolved_links.append(links[i+1])
            resolved_capacity.append(capacity[i+2])
        resolved_links.append(links[-1])
        resolved_capacity.append(0.)
        links,capacity = np.asarray(resolved_links),np.asarray(resolved_capacity)
    capacity[[0,-1]] = area[[0,-1]]*wall_capacity
    if storing_film[0]:
        links,capacity = np.r_[film_r[0],links],np.r_[0.,capacity]
    if storing_film[1]:
        links,capacity = np.r_[links,film_r[1]],np.r_[capacity,0.]
    return links, capacity


def _link_resistances(link_resistance):
    r = np.asarray(link_resistance, dtype=float)
    if r.ndim != 1 or not r.size or not np.isfinite(r).all() or np.any(r <= 0):
        raise ChainError("supply a positive finite resistance for each link")
    return r


def _massless_nodes(capacity, free, allow_massless):
    massless = free & (capacity == 0)
    if massless.any() and not allow_massless:
        raise ChainError("free thermal nodes require positive capacity unless allow_massless=True")
    if massless.all():
        raise ChainError("a massless chain needs a prescribed temperature or a positive capacity")
    return massless


def harmonic_contact_temperatures(link_resistance, heat_capacity, angular_frequency, *,
                                  t_left=None, t_right=None, nodal_power=0., allow_massless=False,
                                  separate_mean=False):
    """Return complex nodal temperatures for a fixed 1-D thermal network.

    Signals use Re(phasor * exp(i*omega*t)), with omega in rad/s. Resistances
    are in K/W, capacities in J/K, prescribed end amplitudes in K and nodal
    powers in W. None leaves an endpoint free of imposed temperature.
    With allow_massless=True a zero-capacity free node balances its incident
    conductive powers and applied power algebraically. For two insulated ends,
    separate_mean=True returns (capacity-centered temperature phasors in K,
    capacity-weighted mean heating rate phasor in K/s). At zero frequency
    these are the relative steady profile and common drift under constant
    power. With a prescribed end, zero frequency gives the static solution.
    An insulated static solution defaults to zero capacity-weighted mean and
    requires balanced power unless separate_mean=True.
    """
    omega = _finite("angular frequency", angular_frequency)
    if omega < 0:
        raise ChainError("angular frequency must be nonnegative")
    r = _link_resistances(link_resistance)
    n = len(r) + 1
    capacity = np.broadcast_to(np.asarray(heat_capacity, dtype=float), (n,))
    power = np.broadcast_to(np.asarray(nodal_power, dtype=complex), (n,))
    if not np.isfinite(capacity).all() or np.any(capacity < 0) or not np.isfinite(power).all():
        raise ChainError("capacity must be nonnegative and nodal inputs must be finite")
    free = np.ones(n, dtype=bool)
    temperature = np.zeros(n, dtype=complex)
    for index, value in ((0, t_left), (-1, t_right)):
        if value is not None:
            if np.ndim(value) != 0 or not np.isfinite(value):
                raise ChainError("wall temperature must be a finite scalar phasor")
            temperature[index] = value
            free[index] = False
    _massless_nodes(capacity, free, allow_massless)
    if separate_mean and not free.all():
        raise ChainError("separate_mean requires two insulated ends")
    if free.any():
        operator = _chain_laplacian(1/r) + sp.diags(1j*omega*capacity)
        if free.all():
            total = capacity.sum()
            rate = power.sum()/total
            weights = sp.csc_matrix((capacity/total)[:, None])
            constrained = sp.bmat([[operator, weights], [weights.T, None]], format="csc")
            relative = sp.linalg.spsolve(constrained, np.r_[power-capacity*rate, 0.])[:-1]
            if separate_mean:
                return relative, rate
            if omega == 0:
                if abs(power.sum()) > 8*np.finfo(float).eps*np.abs(power).sum():
                    raise ChainError("nonzero net power at zero frequency requires separate_mean=True")
                return relative
            return relative+rate/(1j*omega)
        forcing = power[free] - operator[free][:, ~free] @ temperature[~free]
        temperature[free] = sp.linalg.spsolve(operator[free][:, free].tocsc(), forcing)
    return temperature


def low_frequency_contact_temperatures(link_resistance, heat_capacity, *, t_left=None,
                                       t_right=None, nodal_power=0., allow_massless=False):
    """Return (T0, T1, rate) for frequency-independent thermal loads.

    For small positive omega the nodal phasor is rate/(i*omega) + T0 +
    i*omega*T1 + O(omega²). T0 is in K, T1 in K s and rate in K/s.
    With insulated ends T0 and T1 have zero capacity-weighted mean.
    With a fixed end the rate is zero and the slope vanishes at fixed nodes.
    The rightward power coefficients follow from -diff(T0 or T1)/resistance.
    """
    closed = t_left is None and t_right is None
    options = dict(allow_massless=allow_massless, separate_mean=closed)
    steady = harmonic_contact_temperatures(link_resistance, heat_capacity, 0., t_left=t_left,
                                          t_right=t_right, nodal_power=nodal_power, **options)
    profile, rate = steady if closed else (steady, 0j)
    # Equating coefficients of i*omega gives K*T1 = -C*T0 on free nodes.
    slope = harmonic_contact_temperatures(link_resistance, heat_capacity, 0.,
                        t_left=None if t_left is None else 0., t_right=None if t_right is None else 0.,
                        nodal_power=-np.asarray(heat_capacity, dtype=float)*profile, **options)
    return profile, slope[0] if closed else slope, rate


def harmonic_contact_interval(link_resistance, heat_capacity, angular_frequency, start, duration,
                              initial_temperature, *, t_left, t_right, nodal_power=0.,
                              allow_massless=False, return_interval_temperature=False):
    """Return end temperatures (K) and integrated rightward link heats (J).

    End reservoirs accept prescribed temperature phasors or None for insulation. Free stores
    have positive capacities, or satisfy algebraic power balance when
    allow_massless=True. Initial temperatures are real values at start
    (s); end nodes follow their prescribed phasors. The forcing convention is
    Re(phasor*exp(i*omega*t)). Superpose the harmonic response and the matrix
    exponential of the initial deviation, without time discretization.
    At zero frequency the real parts of the loads are constant in time.
    return_interval_temperature appends time-mean node temperatures (K).
    """
    start = _finite("start", start)
    duration = _finite("duration", duration, positive=True)
    r = _link_resistances(link_resistance)
    closed = t_left is None and t_right is None
    response = harmonic_contact_temperatures(r, heat_capacity, angular_frequency,
                    t_left=t_left, t_right=t_right, nodal_power=nodal_power,
                    allow_massless=allow_massless, separate_mean=closed)
    phasor, mean_rate = response if closed else (response, 0j)
    initial = np.broadcast_to(np.asarray(initial_temperature, dtype=float), phasor.shape)
    if not np.isfinite(initial).all():
        raise ChainError("initial temperature must be finite")
    phase = np.exp(1j*angular_frequency*start)
    angle = 1j*angular_frequency*duration
    final = (phasor*phase*np.exp(angle)).real.copy()
    phi1 = np.expm1(angle)/angle if angle != 0 else 1.
    integral = duration*(phasor*phase*phi1).real
    if allow_massless or return_interval_temperature or t_left is None or t_right is None:
        # Superposition leaves a zero-drive relaxation. Algebraic nodes of
        # this deviation obey the same series resistance interpolation.
        offset,heat,mean = affine_contact_interval(r,heat_capacity,duration,initial-(phase*phasor).real,
                        t_left=None if t_left is None else (0.,0.),
                        t_right=None if t_right is None else (0.,0.),allow_massless=allow_massless,
                        return_interval_temperature=True)
        if closed:
            # Integrate the uniform heating from the initial state without
            # introducing the large harmonic temperature pole into link heats.
            phi2 = (.5+angle/6+angle**2/24 if abs(angle) < 1e-4 else (phi1-1)/angle)
            final += duration*(mean_rate*phase*phi1).real
            mean += duration*(mean_rate*phase*phi2).real
        result = (final+offset,heat-np.diff(integral)/r)
        return (*result,mean+integral/duration) if return_interval_temperature else result
    if len(r) > 1:
        c = np.broadcast_to(np.asarray(heat_capacity, dtype=float), phasor.shape)[1:-1]
        stiffness = _chain_laplacian(1/r)[1:-1, 1:-1].tocsc()
        deviation = initial[1:-1]-(phase*phasor[1:-1]).real
        evolved = sp.linalg.expm_multiply(-duration*(sp.diags(1/c) @ stiffness), deviation)
        final[1:-1] += evolved
        # Integrate C*d(delta)/dt = -K*delta over this interval.
        integral[1:-1] += sp.linalg.spsolve(stiffness, c*(deviation-evolved))
    return final, -np.diff(integral)/r


def constant_contact_interval(link_resistance, heat_capacity, duration, initial_temperature, *,
                              t_left, t_right, nodal_power=0., allow_massless=False,
                              return_interval_temperature=False, return_heat_moments=False):
    """Advance one constant-load interval without temporal discretization.

    Return right-endpoint temperatures (K) and integrated rightward heat on
    every link (J). Set return_interval_temperature=True to append the time-mean
    temperature (K) at every node. None leaves an end insulated. Free nodes may
    receive constant power (W); with allow_massless=True their zero-capacity
    traces follow instantaneous power balance. To switch a load, pass the
    returned temperatures to a new interval with its new boundary data.
    Set return_heat_moments to append two arrays of link moments (J s and
    J s²) weighted by time since the interval start and its square.
    Endpoint powers or storage affect reservoir reactions only; the returned
    heat measures conduction through the links.
    """
    return affine_contact_interval(link_resistance, heat_capacity, duration, initial_temperature,
                    t_left=None if t_left is None else (t_left,t_left),
                    t_right=None if t_right is None else (t_right,t_right), nodal_power_start=nodal_power,
                    allow_massless=allow_massless, return_interval_temperature=return_interval_temperature,
                    return_heat_moments=return_heat_moments)


def affine_contact_interval(link_resistance, heat_capacity, duration, initial_temperature, *,
                            t_left, t_right, nodal_power_start=0., nodal_power_end=None,
                            allow_massless=False, return_interval_temperature=False, return_heat_moments=False):
    """Integrate a fixed chain under linearly varying end temperatures and powers.

    t_left and t_right are (start, end) reservoir temperature pairs in K.
    Supply scalar or per-node powers in W at the start and optionally the end.
    Returns endpoint node temperatures (K), signed rightward link heats (J),
    and optionally the time-mean node temperatures (K). Free massless nodes
    follow instantaneous power balance. None leaves an end insulated; its
    specified nodal power still enters the network energy balance. With
    return_heat_moments, append a (2, links) array integrating t*q and t²*q
    measured from the start of this interval.
    """
    r = _link_resistances(link_resistance)
    dt = _finite("duration",duration,positive=True)
    walls = []
    for wall in (t_left,t_right):
        if wall is not None:
            wall = np.asarray(wall,dtype=float)
            if wall.shape!=(2,) or not np.isfinite(wall).all():
                raise ChainError("supply finite start and end temperatures for each reservoir")
        walls.append(wall)
    size = len(r)+1
    power_end = nodal_power_start if nodal_power_end is None else nodal_power_end
    c,u0,p0,p1 = (np.broadcast_to(np.asarray(v,dtype=float),(size,))
                  for v in (heat_capacity,initial_temperature,nodal_power_start,power_end))
    if np.any(c<0) or not all(np.isfinite(v).all() for v in (c,u0,p0,p1)):
        raise ChainError("capacity must be nonnegative and nodal inputs finite")
    free = np.ones(size,dtype=bool)
    free[[0,-1]] = [w is None for w in walls]
    _massless_nodes(c,free,allow_massless)
    closed = free.all()
    active = np.flatnonzero(free & (c>0))
    anchors = np.flatnonzero(~free | (c>0))
    position = np.r_[0.,np.cumsum(r)]

    def steady(sample,power):
        # Remove uniform heating from the closed network before solving the
        # zero-net-power steady problem. Its constant mode is integrated below.
        uniform_rate = power.sum()/c.sum() if closed else 0.
        power = power-c*uniform_rate
        increase = np.r_[0.,np.cumsum(power[1:-1])]
        if free[0]:
            west = power[0]
        elif free[-1]:
            west = -power[1:].sum()
        else:
            west = (walls[0][sample]-walls[1][sample]-r@increase)/r.sum()
        flux = west+increase
        temperature = np.r_[0.,-np.cumsum(r*flux)]
        if closed:
            temperature -= c@temperature/c.sum()
        elif free[0]:
            temperature += walls[1][sample]-temperature[-1]
        else:
            temperature += walls[0][sample]
        if not free[-1]:
            temperature[-1] = walls[1][sample]
        return temperature,flux,uniform_rate

    start,q0,rate0 = steady(0,p0)
    end,q1,rate1 = steady(1,p1)
    mean = .5*(start+end)
    heat = (.5*dt)*(q0+q1)
    if return_heat_moments:
        moments = np.stack([dt**(j+1)*(q0/(j+1)+(q1-q0)/(j+2)) for j in (1, 2)])
    if active.size:
        # Eliminating massless nodes puts all links between anchors in series.
        # At an insulated massless tail the transient deviation is constant.
        segment_links = r[anchors[0]:anchors[-1]]
        reduced_r = np.add.reduceat(segment_links,anchors[:-1]-anchors[0])
        conductance = 1/reduced_r
        dynamic = free[anchors]
        root = np.sqrt(c[active])
        diagonal = (np.r_[0.,conductance]+np.r_[conductance,0.])[dynamic]/c[active]
        offdiag = -conductance[dynamic[:-1] & dynamic[1:]]/(root[:-1]*root[1:])
        rates,basis = sla.eigh_tridiagonal(diagonal,offdiag)
        if closed:
            rates[0] = 0.
        initial_mean = 0.
        if closed:
            reference = u0[active[0]]
            initial_mean = reference+c[active]@(u0[active]-reference)/c[active].sum()
        modes = basis.T @ (root*((u0[active]-initial_mean)-start[active]))
        ramp = basis.T @ (root*(end[active]-start[active]))
        if closed:
            # The known uniform mode has no temperature gradient or link heat.
            modes[0] = ramp[0] = 0.
        z = dt*rates
        phi1 = np.divide(-np.expm1(-z),z,out=np.ones_like(z),where=z!=0)
        # (1-phi1)/z loses precision for a short interval.
        phi2 = np.empty_like(z)
        small = z<1e-4
        phi2[small] = .5-z[small]/6+z[small]**2/24
        phi2[~small] = (1-phi1[~small])/z[~small]
        evolved = (basis @ (np.exp(-z)*modes-phi1*ramp))/root
        mean_offset = (basis @ (phi1*modes-phi2*ramp))/root
        offset = np.zeros(len(anchors))
        offset[dynamic] = evolved
        end += np.interp(position,position[anchors],offset)
        offset[dynamic] = mean_offset
        heat[anchors[0]:anchors[-1]] -= dt*np.repeat(np.diff(offset)/reduced_r,np.diff(anchors))
        if return_heat_moments:
            from scipy.special import hyp1f1
            for j in (1, 2):
                # Integral of u^j exp(-z*u) on [0,1], and the integrated
                # affine-load response (1-exp(-z*u))/z with the same weight.
                weight = hyp1f1(j+1, j+2, -z)/(j+1)
                ramp_weight = np.empty_like(z)
                ramp_weight[small] = (1/(j+2)-z[small]/(2*(j+3))+z[small]**2/(6*(j+4)))
                ramp_weight[~small] = (1/(j+1)-weight[~small])/z[~small]
                weighted = np.zeros(len(anchors))
                weighted[dynamic] = (basis @ (weight*modes-ramp_weight*ramp))/root
                moments[j-1, anchors[0]:anchors[-1]] -= dt**(j+1)*np.repeat(
                    np.diff(weighted)/reduced_r, np.diff(anchors))
        if return_interval_temperature:
            mean += np.interp(position,position[anchors],offset)
    if closed:
        end += initial_mean+.5*dt*(rate0+rate1)
        mean += initial_mean+dt*(2*rate0+rate1)/6
    result = (end,heat,mean) if return_interval_temperature else (end,heat)
    return (*result, moments) if return_heat_moments else result


def transient_contact_temperatures(link_resistance, heat_capacity, time, initial_temperature,
                                   *, t_left=None, t_right=None, nodal_power=0., theta=.5, allow_massless=False,
                                   return_link_heat=False):
    """Integrate a fixed 1D contact network, returning temperatures [time,node].

    Each link has a resistance in K/W. Each node stores heat with a positive
    capacity in J/K. Supply nodal_power in W as a constant scalar or nodal
    vector, or as a (time, node) history sampled at the requested instants. End temperatures
    default to insulated. A scalar or a vector sampled at the requested times
    pins an end, including at t=0. A pinned reservoir may have zero capacity;
    a film of conductance h*A is a link of resistance 1/(h*A) from that
    reservoir to the wall node, which retains its own heat capacity.
    With allow_massless=True zero-capacity free nodes satisfy instantaneous
    power balance, including at the initial time when their supplied temperatures
    are replaced by consistent values. The theta method spans Crank–Nicolson
    (0.5) to backward Euler (1). With return_link_heat=True also return
    an array (time intervals, links) of theta-integrated heat in J, positive
    from the lower-index node to the higher-index node.
    """
    r = _link_resistances(link_resistance)
    n = r.size + 1
    capacity, initial = (np.broadcast_to(np.asarray(v, dtype=float), (n,))
                         for v in (heat_capacity, initial_temperature))
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ChainError("time must be a strictly increasing finite vector")
    power = np.asarray(nodal_power, dtype=float)
    if power.ndim <= 1:
        power = np.broadcast_to(power, (n,))
    elif power.shape != (len(time), n):
        raise ChainError("power history must have shape (time, node)")
    if not all(np.isfinite(v).all() for v in (capacity, power, initial)) or np.any(capacity < 0):
        raise ChainError("capacity must be nonnegative and nodal inputs must be finite")
    if not .5 <= theta <= 1:
        raise ChainError("theta must lie in [0.5, 1]")

    history = np.broadcast_to(initial, (len(time), n)).copy()
    free = np.ones(n, dtype=bool)
    for i, wall in ((0, t_left), (-1, t_right)):
        if wall is not None:
            trace = np.asarray(wall, dtype=float)
            if trace.shape not in ((), time.shape) or not np.isfinite(trace).all():
                raise ChainError("wall temperature must be finite and scalar or sampled at each time")
            history[:, i] = trace
            free[i] = False
    massless = _massless_nodes(capacity, free, allow_massless)
    def result():
        if not return_link_heat:
            return history
        power = -np.diff(history, axis=1)/r
        heat = np.diff(time)[:, None]*((1-theta)*power[:-1] + theta*power[1:])
        return history, heat

    if not free.any():
        return result()

    laplacian = _chain_laplacian(1 / r)
    if massless.any():
        rows = laplacian[massless]
        initial_power = power if power.ndim == 1 else power[0]
        rhs = initial_power[massless] - rows[:, ~massless] @ history[0, ~massless]
        history[0, massless] = sp.linalg.spsolve(rows[:, massless].tocsc(), rhs)
    boundary = laplacian[free][:, ~free]
    applied_power = power[..., free]
    operator = laplacian[free][:, free].tocsc()
    mass = sp.diags(capacity[free], format="csc")
    previous_dt = None
    for i, dt in enumerate(np.diff(time)):
        if dt != previous_dt:
            lu = sp.linalg.splu(mass + theta * dt * operator)
            previous_dt = dt
        current = history[i, free]
        wall = (1-theta) * history[i, ~free] + theta * history[i+1, ~free]
        step_power = (applied_power if applied_power.ndim == 1 else
                      (1-theta)*applied_power[i] + theta*applied_power[i+1])
        forcing = step_power - boundary @ wall
        rhs = capacity[free] * current + dt * (forcing - (1-theta) * (operator @ current))
        history[i+1, free] = lu.solve(rhs)
    return result()


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
        return _chain_laplacian(self.link_conductance)

    def _dirichlet_matrix(self) -> sp.csr_matrix:
        mat = self.conduction_matrix().tolil()
        mat[0, :] = 0.0
        mat[-1, :] = 0.0
        mat[0, 0] = 1.0
        mat[-1, -1] = 1.0
        return mat.tocsr()

    def solve_steady(self, *, return_link_currents: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return steady temperatures in K and optionally hot-to-cold link currents in W.

        Currents are evaluated from the relative solution before adding the
        cold-end temperature, preserving small differences at large offsets.
        """
        c = self.case
        rhs = np.zeros(self.n_nodes)
        rhs[0] = c.t_hot - c.t_cold
        relative = sp.linalg.spsolve(self._dirichlet_matrix(), rhs)
        temperature = relative + c.t_cold
        temperature[[0, -1]] = (c.t_hot, c.t_cold)
        if return_link_currents:
            return temperature, -self.link_conductance * np.diff(relative)
        return temperature

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
        rhs = coeff * (t - t_cold)
        rhs[0] = float(t_hot) - t_cold
        rhs[-1] = 0.
        result = lu.solve(rhs) + t_cold
        result[[0, -1]] = (t_hot, t_cold)
        return result

    def initial_state(self) -> np.ndarray:
        c = self.case
        t = np.full(self.n_nodes, c.t_cold, dtype=float)
        t[0], t[-1] = c.t_hot, c.t_cold
        return t

    def march(self, dt: float, n_steps: int, *, initial: np.ndarray | None = None,
              record_every: int = 1, relative_nodes: bool = False) -> dict[str, np.ndarray]:
        """Return time, node history and hot-end flux.

        With relative_nodes=True the node history contains T - t_cold in K.
        The optional initial field is always supplied in absolute temperatures.
        """
        c = self.case
        t = self.initial_state() if initial is None else np.asarray(
            initial, dtype=np.float64).copy()
        times = [0.0]
        fields = [t - c.t_cold if relative_nodes else t.copy()]
        # Keep the evolving state relative through all implicit steps. Adding
        # the absolute offset between steps would erase sub-ULP changes.
        t -= c.t_cold
        drop = c.t_hot - c.t_cold
        flux = [float(self.link_conductance[0] * (t[0] - t[1]))]
        every = max(int(record_every), 1)
        for step in range(int(n_steps)):
            t = self.step(t, dt, t_hot=drop, t_cold=0.)
            if (step + 1) % every == 0:
                times.append((step + 1) * dt)
                field = t.copy() if relative_nodes else t + c.t_cold
                if not relative_nodes:
                    field[[0, -1]] = (c.t_hot, c.t_cold)
                fields.append(field)
                flux.append(float(self.link_conductance[0] * (t[0] - t[1])))
        return {"time": np.asarray(times), "nodes": np.asarray(fields),
                "hot_flux": np.asarray(flux)}

    def _interior_operator(self) -> np.ndarray:
        """``dT/dt = M T`` restricted to the interior nodes (Dirichlet ends pinned)."""
        g_full = self.conduction_matrix().toarray()
        return -g_full[1:-1, 1:-1] / float(self.capacity)

    def analytic_transient(self, times: Sequence[float], *, relative_nodes: bool = False) -> np.ndarray:
        """Exact discrete solution (matrix exponential) of the same RC network.

        The Dirichlet end nodes are eliminated, so the reference is the exact
        solution of the *sampled* operator: comparing it with :meth:`march`
        isolates the implicit-Euler time error from any spatial error. With
        relative_nodes=True the exact temperatures are relative to the cold end.
        """
        c = self.case
        a_int = self._interior_operator()
        steady = analytic_chain_temperature(c.n_spheres, c.contact_radius, c.radius,
                                            c.conductivity, self.m, c.t_hot-c.t_cold, 0.)
        dev0 = -steady[1:-1]
        out = np.empty((len(times), self.n_nodes))
        for idx, tt in enumerate(times):
            out[idx, 1:-1] = steady[1:-1] + sla.expm(a_int * float(tt)) @ dev0
        if not relative_nodes:
            out[:, 1:-1] += c.t_cold
        out[:, 0] = c.t_hot-c.t_cold if relative_nodes else c.t_hot
        out[:, -1] = 0. if relative_nodes else c.t_cold
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
        numeric, currents = chain.solve_steady(return_link_currents=True)
        analytic = chain.analytic_steady()
        span = case.t_hot - case.t_cold
        payload["steady"] = {
            "heat_flow_w": case.heat_flow,
            "measured_heat_flow_w": float(currents[0]),
            "measured_k_eff_w_m_k": float(currents[0] * case.length / (case.area * span)),
            "contact_jump_k": float(currents[chain.contact_links[0]] / chain.contact_g),
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
                            record_every=max(steps // 10, 1), relative_nodes=True)
        exact = chain.analytic_transient(march["time"], relative_nodes=True)
        payload["transient"] = {
            "relaxation_time_s": tau,
            "continuum_relaxation_time_s": chain.continuum_relaxation_time(),
            "max_rel_error": float(np.max(np.abs(march["nodes"] - exact))
                                   / (case.t_hot-case.t_cold)),
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
