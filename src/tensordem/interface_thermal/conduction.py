"""Fourier conduction for DEM particles, contact interfaces and structure walls.

This module is the thermal substrate of the ice-structure contact coupling
(``tensordem.interface_thermal``).  It provides three things:

1. **Exact reference solutions** for the semi-infinite body
   (:func:`semi_infinite_temperature_profile`,
   :func:`semi_infinite_surface_temperature`).  These are Carslaw & Jaeger's
   elementary solutions and are used *only* as grading references.
2. **A finite-volume transient conductor** (:class:`Conduction1D`) with a
   theta-scheme time integrator, optional cell grading and Dirichlet / flux /
   adiabatic surface conditions.  This is the "FVM" side of the DEM-FVM
   coupling: it carries the temperature field of the ice body that the DEM
   contact loads feed.
3. **A lumped particle thermal network** (:class:`ParticleThermalNetwork`)
   which adds a temperature degree of freedom to DEM particles, with a
   constriction resistance inside each particle and a contact thermal
   resistance between contacting particles
   (:func:`contact_heat_transfer_coefficient`, Cooper-Mikic-Yovanovich).

Units are SI. Conduction accepts a common temperature datum. Entropy diagnostics require kelvin.
Surface fluxes are positive *into* the body (``q > 0`` heats the body).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erfc
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

__all__ = [
    "semi_infinite_temperature_profile",
    "semi_infinite_surface_temperature",
    "thermal_diffusivity",
    "Conduction1D",
    "contact_heat_transfer_coefficient",
    "maxwell_constriction_resistance",
    "contact_thermal_resistance",
    "Particle",
    "ParticleThermalNetwork",
]


def _finite(name: str, value: float, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def thermal_diffusivity(k_w_m_k: float, rho_kg_m3: float, cp_j_kg_k: float) -> float:
    """``alpha = k / (rho c_p)`` in m^2/s."""
    k = _finite("k_w_m_k", k_w_m_k, positive=True)
    rho = _finite("rho_kg_m3", rho_kg_m3, positive=True)
    cp = _finite("cp_j_kg_k", cp_j_kg_k, positive=True)
    return k / (rho * cp)


# ---------------------------------------------------------------------------
# exact reference solutions
# ---------------------------------------------------------------------------
def semi_infinite_temperature_profile(
    x: Sequence[float], t: float, alpha: float,
    t_initial: float, t_surface: float,
) -> np.ndarray:
    """Step-heated semi-infinite body.

    ``T(x, t) = Ti + (Ts - Ti) erfc(x / (2 sqrt(alpha t)))`` for a body initially
    at ``t_initial`` whose surface is held at ``t_surface`` from ``t = 0`` on
    (``T = Ts`` at ``x = 0``, ``T -> Ti`` as ``x -> inf``).
    """
    alpha = _finite("alpha", alpha, positive=True)
    t = _finite("t", t, nonnegative=True)
    x = np.asarray(x, dtype=float)
    if np.any(x < 0.0):
        raise ValueError("x must be nonnegative")
    if t == 0.0:
        return np.where(x <= 0.0, float(t_surface), float(t_initial)).astype(float)
    return t_initial + (t_surface - t_initial) * erfc(x / (2.0 * math.sqrt(alpha * t)))


def semi_infinite_surface_temperature(
    t: Sequence[float] | float, alpha: float, k_w_m_k: float,
    flux_w_m2: float, t_initial: float,
) -> np.ndarray:
    """Constant-flux surface temperature ``Ts = Ti + 2 q sqrt(alpha t / pi) / k``.

    Carslaw & Jaeger's transient surface temperature of a semi-infinite body
    under a constant heat flux: the "flash temperature" of a frictional contact
    whose contact size is much larger than the thermal penetration depth.
    """
    alpha = _finite("alpha", alpha, positive=True)
    k = _finite("k_w_m_k", k_w_m_k, positive=True)
    flux = _finite("flux_w_m2", flux_w_m2)
    t = np.asarray(t, dtype=float)
    if np.any(t < 0.0):
        raise ValueError("t must be nonnegative")
    return t_initial + 2.0 * flux * np.sqrt(alpha * t / math.pi) / k


# ---------------------------------------------------------------------------
# finite-volume transient conductor
# ---------------------------------------------------------------------------
@dataclass
class Conduction1D:
    """Node-centred finite-volume conductor on ``0 <= x <= length_m``.

    The domain is a slab: node 0 is the *surface* (``x = 0``) and the far end
    (``x = length_m``) is either held at a Dirichlet value or sealed.  Heat
    enters through node 0 as an areal flux (W/m^2), which is how the friction
    heat of a contact is injected.

    ``theta = 1`` (default) is backward Euler, ``theta = 0.5`` Crank-Nicolson;
    the unknown ``T^{n+1}`` is always solved implicitly (tridiagonal), so the
    time step is not limited by the explicit stability bound.  Backward Euler is
    the default because the flux boundary is switched on at ``t = 0``: with a
    discontinuous start the non-L-stable Crank-Nicolson scheme rings with a
    sign-alternating surface cell (measured: surface +5.6 K against a -1.4 K
    neighbour at ``dx = 1.5e-4 m``, i.e. the whole injected energy parks in the
    half cell), while backward Euler reproduces the semi-infinite solution.

    ``surface_capacity_j_m2_k`` adds an areal heat capacity to the surface half
    cell.  The phase-change coupling uses it to hang the latent capacity of the
    melt film on the surface cell, which is the enthalpy method on the finest
    cell of the interface.
    """

    length_m: float = 0.05
    cells: int = 401
    k_w_m_k: float = 2.2
    rho_kg_m3: float = 917.0
    cp_j_kg_k: float = 2100.0
    t_initial: float = -2.0
    grading: float = 1.0
    theta: float = 1.0
    surface_capacity_j_m2_k: float = 0.0
    t: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.length_m = _finite("length_m", self.length_m, positive=True)
        self.k_w_m_k = _finite("k_w_m_k", self.k_w_m_k, positive=True)
        self.rho_kg_m3 = _finite("rho_kg_m3", self.rho_kg_m3, positive=True)
        self.cp_j_kg_k = _finite("cp_j_kg_k", self.cp_j_kg_k, positive=True)
        self.t_initial = _finite("t_initial", self.t_initial)
        self.grading = _finite("grading", self.grading, positive=True)
        self.theta = _finite("theta", self.theta)
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        if isinstance(self.cells, bool) or not isinstance(self.cells, int):
            raise TypeError("cells must be an integer")
        if self.cells < 3:
            raise ValueError("cells must be at least 3")
        self.surface_capacity_j_m2_k = _finite(
            "surface_capacity_j_m2_k", self.surface_capacity_j_m2_k, nonnegative=True)
        self._build_grid()
        self.reset()

    # -- grid / state ---------------------------------------------------
    def _build_grid(self) -> None:
        n = self.cells
        if abs(self.grading - 1.0) < 1e-12:
            self.x = np.linspace(0.0, self.length_m, n)
        else:
            r = self.grading
            dx = self.length_m * (r - 1.0) / (r ** (n - 1) - 1.0)
            widths = dx * r ** np.arange(n - 1, dtype=float)
            self.x = np.concatenate(([0.0], np.cumsum(widths)))
        dx = np.diff(self.x)
        self.dx = dx
        self.cell_width = np.empty(n)
        self.cell_width[0] = 0.5 * dx[0]
        self.cell_width[-1] = 0.5 * dx[-1]
        self.cell_width[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        self.capacity = self.rho_kg_m3 * self.cp_j_kg_k * self.cell_width
        self.capacity[0] += self.surface_capacity_j_m2_k
        # face conductances G_j between node j and j+1, plus the ghost
        # conductance of the Dirichlet far boundary (distance = dx/2).
        self.conductance = self.k_w_m_k / dx
        self.ghost_conductance = 2.0 * self.k_w_m_k / dx[-1]

    def reset(self, t_initial: float | None = None) -> None:
        value = self.t_initial if t_initial is None else _finite("t_initial", t_initial)
        self.t = np.full(self.cells, value, dtype=float)

    @property
    def alpha(self) -> float:
        return self.k_w_m_k / (self.rho_kg_m3 * self.cp_j_kg_k)

    def enthalpy_above_initial(self) -> float:
        """Areal enthalpy stored above the initial state (J/m^2)."""
        return float(np.sum(self.capacity * (self.t - self.t_initial)))

    def surface_temperature(self) -> float:
        return float(self.t[0])

    # -- operators ------------------------------------------------------
    def _apply_operator(self, t: np.ndarray, q_surface: float,
                        t_far: float | None) -> np.ndarray:
        """Finite-volume balance ``sum(face fluxes) + boundary terms`` in W/m^2."""
        flux = self.conductance * (t[1:] - t[:-1])
        out = np.zeros_like(t)
        # net *inflow* convention: op = A_std T + boundary sources, i.e. the
        # term that heats a cell (the implicit matrix is C/dt - theta * op)
        out[1:] -= flux
        out[:-1] += flux
        out[0] += q_surface
        if t_far is not None:
            out[-1] += self.ghost_conductance * (t_far - t[-1])
        return out

    def _solve_tridiagonal(self, dt: float, q_surface: float,
                           t_far: float | None,
                           t_surface: float | None = None) -> None:
        """One implicit theta step; updates ``self.t`` in place."""
        theta = self.theta
        cap_dt = self.capacity / dt
        diag = cap_dt.copy()
        upper = -theta * self.conductance
        lower = -theta * self.conductance
        diag[:-1] += theta * self.conductance
        diag[1:] += theta * self.conductance
        rhs = cap_dt * self.t + (1.0 - theta) * self._apply_operator(
            self.t, q_surface, t_far)
        rhs[0] += theta * q_surface
        if t_far is not None:
            diag[-1] += theta * self.ghost_conductance
            rhs[-1] += theta * self.ghost_conductance * t_far
        n = self.cells
        ab = np.zeros((3, n))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        if t_surface is not None:
            # strong Dirichlet on the surface node (melting interface)
            ab[1, 0] = 1.0
            ab[0, 1] = 0.0
            rhs[0] = float(t_surface)
        self.t = solve_banded((1, 1), ab, rhs)

    def step(self, dt: float, q_surface: float = 0.0,
             t_far: float | None = None,
             t_surface: float | None = None) -> np.ndarray:
        """Advance one time step.

        Either a surface flux (Neumann, default) or a surface temperature
        (Dirichlet, strongly imposed -- the melting interface) may be given.
        """
        dt = _finite("dt", dt, positive=True)
        self._solve_tridiagonal(dt, _finite("q_surface", q_surface), t_far,
                                t_surface)
        return self.t

    def run(
        self,
        t_end: float,
        dt: float,
        *,
        q_surface: float | Callable[[float], float] = 0.0,
        t_far: float | None = None,
        t_surface: float | None = None,
        record_times: Sequence[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """March to ``t_end`` and return the recorded history.

        ``q_surface`` may be a callable of the current time.  ``record_times``
        are absolute times at which the profile is stored (the last step is
        clipped so that every requested time is hit exactly).
        """
        t_end = _finite("t_end", t_end, positive=True)
        dt = _finite("dt", dt, positive=True)
        flux_fn = q_surface if callable(q_surface) else (lambda _t: float(q_surface))
        targets = sorted(float(x) for x in (record_times or ()))
        time = 0.0
        history: dict[str, list] = {"time": [], "profiles": [], "surface": []}
        pending = list(targets)
        while time < t_end - 1e-15:
            step = dt
            if pending and pending[0] - time < step:
                step = max(pending[0] - time, 1e-15)
            self.step(step, flux_fn(time + 0.5 * step), t_far,
                      t_surface=t_surface)
            time += step
            while pending and pending[0] <= time + 1e-12:
                history["time"].append(pending.pop(0))
                history["profiles"].append(self.t.copy())
                history["surface"].append(self.surface_temperature())
        if not targets:
            history["time"].append(time)
            history["profiles"].append(self.t.copy())
            history["surface"].append(self.surface_temperature())
        return {
            "time": np.array(history["time"]),
            "profiles": np.array(history["profiles"]),
            "surface": np.array(history["surface"]),
            "x": self.x.copy(),
        }

    def impedance(self, t_end: float, dt: float, *, t_far: float | None = None,
                  record_times: Sequence[float] | None = None) -> np.ndarray:
        """Surface thermal impedance ``Z(t) = dTs/dq`` (K m^2/W).

        The conductor is linear before phase change, so a unit-flux run returns
        the exact response: ``Ts(t) = t_initial + Z(t) q``.  Used by the
        coupled interface solver to close the DEM -> FVM -> DEM loop with a
        Newton step instead of a slowly converging flux iteration.
        """
        self.reset()
        result = self.run(t_end, dt, q_surface=1.0, t_far=t_far,
                          record_times=record_times)
        return result["surface"] - self.t_initial


# ---------------------------------------------------------------------------
# contact heat transfer
# ---------------------------------------------------------------------------
def contact_heat_transfer_coefficient(
    k1_w_m_k: float, k2_w_m_k: float, rms_roughness_m: float,
    rms_slope: float, contact_pressure_pa: float, hardness_pa: float,
) -> float:
    """Cooper-Mikic-Yovanovich contact conductance ``h_c`` in W/(m^2 K).

    ``h_c = 1.25 (2 k1 k2/(k1+k2)) (m/sigma) (P/H)^0.95`` for plastic
    asperity contact, with ``sigma`` the composite rms roughness, ``m`` the
    composite rms slope, ``P`` the nominal pressure and ``H`` the softer
    hardness.  The two solid conductivities enter as the harmonic mean because
    the constriction is a series path through both bodies.
    """
    k1 = _finite("k1_w_m_k", k1_w_m_k, positive=True)
    k2 = _finite("k2_w_m_k", k2_w_m_k, positive=True)
    sigma = _finite("rms_roughness_m", rms_roughness_m, positive=True)
    slope = _finite("rms_slope", rms_slope, positive=True)
    pressure = _finite("contact_pressure_pa", contact_pressure_pa, nonnegative=True)
    hardness = _finite("hardness_pa", hardness_pa, positive=True)
    if pressure == 0.0:
        return 0.0
    # Scale by the smaller conductivity so no product of materials is needed.
    lower, upper = sorted((k1, k2))
    k_harmonic = lower / (0.5 + 0.5 * (lower / upper))
    ratio = min(pressure / hardness, 1.0)
    return 1.25 * k_harmonic * (slope / sigma) * ratio ** 0.95


def maxwell_constriction_resistance(k_w_m_k: float, patch_radius_m: float) -> float:
    """Constriction resistance ``1 / (4 k a)`` (K/W) of a circular patch.

    Maxwell's spreading resistance for an isothermal circular contact of radius
    ``a`` on a half space of conductivity ``k``.  It is the leading term of the
    constriction resistance between two particles in contact.
    """
    k = _finite("k_w_m_k", k_w_m_k, positive=True)
    a = _finite("patch_radius_m", patch_radius_m, positive=True)
    return 1.0 / (4.0 * k * a)


def contact_thermal_resistance(
    k1_w_m_k: float, k2_w_m_k: float, patch_radius_m: float,
    h_c_w_m2_k: float,
) -> float:
    """Series resistance (K/W), infinite for an insulating asperity film."""
    h_c = _finite("h_c_w_m2_k", h_c_w_m2_k, nonnegative=True)
    area = math.pi * _finite("patch_radius_m", patch_radius_m, positive=True) ** 2
    film_conductance = h_c * area
    return (maxwell_constriction_resistance(k1_w_m_k, patch_radius_m)
            + maxwell_constriction_resistance(k2_w_m_k, patch_radius_m)
            + (math.inf if film_conductance == 0.0 else 1.0 / film_conductance))


# ---------------------------------------------------------------------------
# DEM particle thermal network
# ---------------------------------------------------------------------------
@dataclass
class Particle:
    """Spherical DEM particle carrying a lumped temperature degree of freedom."""

    radius_m: float
    k_w_m_k: float = 2.2
    rho_kg_m3: float = 917.0
    cp_j_kg_k: float = 2100.0

    def __post_init__(self) -> None:
        self.radius_m = _finite("radius_m", self.radius_m, positive=True)
        self.k_w_m_k = _finite("k_w_m_k", self.k_w_m_k, positive=True)
        self.rho_kg_m3 = _finite("rho_kg_m3", self.rho_kg_m3, positive=True)
        self.cp_j_kg_k = _finite("cp_j_kg_k", self.cp_j_kg_k, positive=True)

    @property
    def capacity_j_k(self) -> float:
        """Lumped heat capacity ``rho c_p 4/3 pi R^3`` in J/K."""
        return self.rho_kg_m3 * self.cp_j_kg_k * 4.0 / 3.0 * math.pi * self.radius_m ** 3

    @property
    def maxwell_resistance_k_w(self) -> float:
        """Constriction resistance seen by a contact patch on this particle.

        ``1/(4 k a)`` with ``a = R/2`` -- documented order-of-magnitude choice
        for the intra-particle half of the series path (the patch radius of a
        Hertz contact is much smaller than ``R``, so the spreading resistance is
        the softest thermal link inside a particle).
        """
        return maxwell_constriction_resistance(self.k_w_m_k, 0.5 * self.radius_m)


class ParticleThermalNetwork:
    """Lumped thermal network of DEM particles with contact heat transfer.

    Each particle is one node with capacity ``rho c_p V``; each contact adds a
    resistance ``R_i + R_c + R_j`` (two constrictions in series with the
    asperity contact resistance ``1/(h_c A)``).  The system is integrated with
    backward Euler, which conserves energy exactly (up to the linear solve) and
    is unconditionally stable.
    """

    def __init__(self, particles: Sequence[Particle],
                 contacts: Sequence[tuple[int, int]],
                 *, contact_pressure_pa: float = 2.0e6,
                 rms_roughness_m: float = 5.0e-5, rms_slope: float = 0.2,
                 hardness_pa: float = 1.0e7,
                 patch_radius_fraction: float = 0.25) -> None:
        if not particles:
            raise ValueError("at least one particle is required")
        self.particles = list(particles)
        n = len(self.particles)
        for i, j in contacts:
            if not (0 <= i < n and 0 <= j < n) or i == j:
                raise ValueError("contact indices must be distinct and in range")
        self.contacts = [(int(i), int(j)) for i, j in contacts]
        self.capacity = np.array([p.capacity_j_k for p in self.particles])
        self.contact_resistance: list[float] = []
        for i, j in self.contacts:
            a = patch_radius_fraction * min(self.particles[i].radius_m,
                                           self.particles[j].radius_m)
            h_c = contact_heat_transfer_coefficient(
                self.particles[i].k_w_m_k, self.particles[j].k_w_m_k,
                rms_roughness_m, rms_slope, contact_pressure_pa, hardness_pa)
            self.contact_resistance.append(contact_thermal_resistance(
                self.particles[i].k_w_m_k, self.particles[j].k_w_m_k, a, h_c))
        self.temperature = np.zeros(n, dtype=float)

    def reset(self, temperatures: Sequence[float]) -> None:
        values = np.asarray(temperatures, dtype=float)
        if values.shape != (len(self.particles),):
            raise ValueError("one temperature per particle is required")
        self.temperature = values.copy()

    def conductances(self) -> np.ndarray:
        """Contact conductance vector ``1/R`` in W/K."""
        return 1.0 / np.asarray(self.contact_resistance, dtype=float)

    def energy(self) -> float:
        """Total sensible energy ``sum C_i T_i`` (arbitrary datum)."""
        return float(np.dot(self.capacity, self.temperature))

    def step(self, dt: float, *, ambient: Sequence[float] | None = None,
             h_ambient_w_k: Sequence[float] | None = None,
             particle_power_w: Sequence[float] | None = None,
             return_contact_heat: bool = False,
             return_ambient_heat: bool = False,
             return_contact_entropy: bool = False,
             return_ambient_entropy: bool = False) -> np.ndarray | tuple[np.ndarray, ...]:
        """Advance by backward Euler. Optionally return (temperature, contact heat).

        Heat is in joules over this step, positive from i to j for each
        (i, j) in contacts, in input order. Ambient exchange is external.
        particle_power_w supplies one applied power per particle in watts,
        held constant over the step. Positive power heats the particle.
        return_ambient_heat appends one heat in joules per particle, positive
        into the particle. When both heats are requested, contact heat comes first.
        return_contact_entropy appends conductive entropy production per contact
        (J/K), using backward Euler endpoint quadrature. Temperatures must then
        be absolute (K) and positive. Insulating contacts produce zero entropy.
        return_ambient_entropy appends one entropy production (J/K) per particle
        from its passive ambient link, after all other requested outputs. It
        uses the endpoint particle and reservoir temperatures in kelvin.
        """
        dt = _finite("dt", dt, positive=True)
        n = len(self.particles)
        anchor = self.capacity / dt
        links = np.zeros((n, n))
        conductances = self.conductances()
        for (i, j), g in zip(self.contacts, conductances):
            links[i, j] += g
            links[j, i] += g
        need_entropy = return_contact_entropy or return_ambient_entropy
        need_ambient_heat = return_ambient_heat or return_ambient_entropy
        if need_entropy and not np.all(np.isfinite(self.temperature) & (self.temperature > 0)):
            raise ValueError("entropy requires positive finite temperatures in kelvin")
        report_heat = return_contact_heat or need_ambient_heat or need_entropy
        reference = 0.
        if report_heat:
            # A separate datum per connected component keeps tiny exchanges
            # independent of unrelated particles' absolute temperatures.
            _, labels = connected_components(links != 0., directed=False)
            _, first = np.unique(labels, return_index=True)
            reference = self.temperature[first][labels]
        rhs = anchor * (self.temperature-reference)
        power_w = 0.
        if particle_power_w is not None:
            power_w = np.asarray(particle_power_w, dtype=float)
            if power_w.shape != (n,) or not np.isfinite(power_w).all():
                raise ValueError("particle_power_w requires one finite power per particle")
            rhs += power_w
        if ambient is not None:
            amb = np.asarray(ambient, dtype=float)
            h = np.asarray(h_ambient_w_k if h_ambient_w_k is not None else
                           np.zeros(n), dtype=float)
            if amb.shape != (n,) or h.shape != (n,):
                raise ValueError("ambient and conductance must be per particle")
            if return_ambient_entropy:
                if not np.isfinite(h).all() or (h < 0).any():
                    raise ValueError("ambient entropy requires finite nonnegative conductances")
                if not np.isfinite(amb).all() or (amb <= 0).any():
                    raise ValueError("ambient entropy requires positive finite reservoir temperatures in kelvin")
            anchor += h
            rhs += h * (amb-reference)
        need_contacts = return_contact_heat or return_contact_entropy or (need_ambient_heat and ambient is not None
                                                and np.any(h > self.capacity/dt))

        def heat_result(field, contact_heat=None):
            result = [field + reference]
            if return_contact_heat:
                result.append(contact_heat)
            if need_ambient_heat:
                external = np.zeros(n)
                if ambient is not None:
                    rate = self.capacity / dt
                    strong = h > rate
                    external[~strong] = dt * (h[~strong] * ((amb-reference)-field)[~strong])
                    if strong.any():
                        # Eliminate the final temperature from the Robin law
                        # and the nodal energy balance. A stiff film can erase
                        # its own temperature drop at floating-point precision.
                        outward = np.zeros(n)
                        for (i, j), amount in zip(self.contacts, contact_heat):
                            outward[i] += amount
                            outward[j] -= amount
                        demand = self.capacity*(amb-self.temperature) - dt*power_w + outward
                        external[strong] = demand[strong]/(1 + rate[strong]/h[strong])
                if return_ambient_heat:
                    result.append(external)
            if need_entropy:
                temperature = result[0]
                if not np.all(np.isfinite(temperature) & (temperature > 0)):
                    raise ValueError("entropy requires positive finite temperatures in kelvin")
            if return_contact_entropy:
                entropy = np.zeros(len(self.contacts))
                for e, ((i, j), g) in enumerate(zip(self.contacts, conductances)):
                    if g:
                        # Q*(Ti-Tj)/(Ti*Tj), with the jump recovered from the
                        # contact transfer even when both final temperatures round alike.
                        jump = (contact_heat[e]/dt)/g
                        entropy[e] = (contact_heat[e]/temperature[i])*(jump/temperature[j])
                result.append(entropy)
            if return_ambient_entropy:
                entropy = np.zeros(n)
                if ambient is not None:
                    # The finite heat balance resolves a stiff link's drop even
                    # when its endpoint temperature has rounded to the reservoir.
                    jump = (amb-reference)-field
                    jump[strong] = (external[strong]/dt)/h[strong]
                    entropy = (external/temperature)*(jump/amb)
                result.append(entropy)
            self.temperature = result[0]
            return tuple(result)

        if np.any(anchor <= 0):
            # Signed ambient coefficients can require pivoting.
            field = np.linalg.solve(np.diag(anchor + links.sum(axis=1)) - links, rhs)
            if need_contacts:
                jumps = np.asarray([field[i]-field[j] for i, j in self.contacts])
                return heat_result(field, dt * (conductances * jumps))
            if report_heat:
                return heat_result(field)
            self.temperature = field
            return field

        reductions = []
        if need_contacts:
            original_links = links.copy()
        # Eliminate each particle by transferring its links, storage and
        # ambient coupling separately. This preserves the small diagonal
        # contribution even when adding it to a contact would round it away.
        for i in range(n):
            remaining = i + 1 + np.flatnonzero(links[i, i+1:])
            neighbors = links[i, remaining]
            pivot = anchor[i] + neighbors.sum()
            weights = neighbors / pivot
            anchor[remaining] += weights * anchor[i]
            rhs[remaining] += weights * rhs[i]
            block = np.ix_(remaining, remaining)
            added = np.outer(weights, neighbors)
            if need_contacts:
                # Store how each reduced link divides between its old direct
                # path and the new path through particle i.
                previous = links[block]
                total = previous + added
                total[total == 0.] = 1.
                reductions.append((remaining, weights, anchor[i], rhs[i],
                                   previous/total, added/total))
            links[block] += added
            links[i, remaining] = weights
            rhs[i] /= pivot
        for i in reversed(range(n)):
            rhs[i] += links[i, i+1:] @ rhs[i+1:]
        if need_contacts:
            # Undo the reductions in reverse, dividing the known currents
            # between parallel paths. This retains finite exchanged energy
            # even when two endpoint temperatures round to the same value.
            power = np.zeros((n, n))
            for i in reversed(range(n)):
                nodes, weights, storage, load, direct, indirect = reductions[i]
                block = np.ix_(nodes, nodes)
                reduced = power[block]
                outgoing = weights*(load-storage*rhs[nodes]) - (indirect*reduced).sum(axis=1)
                power[block] = direct*reduced
                power[i, nodes] = outgoing
                power[nodes, i] = -outgoing
            heat = np.zeros(len(self.contacts))
            for e, ((i, j), g) in enumerate(zip(self.contacts, conductances)):
                if g:
                    heat[e] = dt * (power[i, j] * (g/original_links[i, j]))
            return heat_result(rhs, heat)
        if report_heat:
            return heat_result(rhs)
        self.temperature = rhs
        return rhs

    def isolated_equilibration_entropy(self) -> float:
        """Entropy gain (J/K) on relaxing the current network to equilibrium.

        Assume fixed positive heat capacities, no external heating and no
        reservoir exchange. Each component joined by conducting contacts
        conserves its own energy. The current temperatures must be in kelvin.
        """
        t = self.temperature
        if not np.all(np.isfinite(t) & (t > 0)):
            raise ValueError("equilibration entropy requires positive finite temperatures in kelvin")
        edges = np.asarray(self.contacts, dtype=int).reshape(-1, 2)[self.conductances() > 0]
        graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(len(t), len(t)))
        _, labels = connected_components(graph, directed=False)
        gains = []
        for indices in np.split(np.argsort(labels), np.cumsum(np.bincount(labels))[:-1]):
            temp, capacity = t[indices], self.capacity[indices]
            span = temp-temp.min()
            if not span.any():
                continue
            weights = capacity/capacity.max()
            mean_span = np.dot(weights/weights.sum(), span)
            mean = temp.min()+mean_span
            # The weighted sum of u=(T-mean)/mean is zero. Thus the
            # entropy gain is a sum of nonnegative u-log(1+u) terms.
            u = (span-mean_span)/mean
            gap = np.empty_like(u)
            small = abs(u) < .001
            z = u[small]
            gap[small] = z*z*(.5+z*(-1/3+z*(.25+z*(-.2+z/6))))
            cold = u < -.5
            gap[cold] = u[cold] + np.log(mean) - np.log(temp[cold])
            rest = ~(small | cold)
            gap[rest] = u[rest]-np.log1p(u[rest])
            gains.append(float(capacity @ gap))
        return math.fsum(gains)

    def equilibrium_temperature(self) -> float:
        """Energy-weighted equilibrium temperature (``sum C T / sum C``)."""
        return float(np.dot(self.capacity, self.temperature) / np.sum(self.capacity))
