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

Units are SI; temperatures are degC but only differences and derivatives enter.
Surface fluxes are positive *into* the body (``q > 0`` heats the body).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erfc

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
    k_harmonic = 2.0 * k1 * k2 / (k1 + k2)
    ratio = min(max(pressure / hardness, 1e-12), 1.0)
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
    """Series resistance of two constrictions plus the asperity film (K/W)."""
    h_c = _finite("h_c_w_m2_k", h_c_w_m2_k, positive=True)
    area = math.pi * _finite("patch_radius_m", patch_radius_m, positive=True) ** 2
    return (maxwell_constriction_resistance(k1_w_m_k, patch_radius_m)
            + maxwell_constriction_resistance(k2_w_m_k, patch_radius_m)
            + 1.0 / (h_c * area))


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
             h_ambient_w_k: Sequence[float] | None = None) -> np.ndarray:
        """One implicit-Euler step with optional convective links to an ambient."""
        dt = _finite("dt", dt, positive=True)
        n = len(self.particles)
        mat = np.zeros((n, n))
        mat[np.diag_indices(n)] = self.capacity / dt
        rhs = self.capacity / dt * self.temperature
        for (i, j), g in zip(self.contacts, self.conductances()):
            mat[i, i] += g
            mat[j, j] += g
            mat[i, j] -= g
            mat[j, i] -= g
        if ambient is not None:
            amb = np.asarray(ambient, dtype=float)
            h = np.asarray(h_ambient_w_k if h_ambient_w_k is not None else
                           np.zeros(n), dtype=float)
            if amb.shape != (n,) or h.shape != (n,):
                raise ValueError("ambient and conductance must be per particle")
            mat[np.diag_indices(n)] += h
            rhs = rhs + h * amb
        self.temperature = np.linalg.solve(mat, rhs)
        return self.temperature

    def equilibrium_temperature(self) -> float:
        """Energy-weighted equilibrium temperature (``sum C T / sum C``)."""
        return float(np.dot(self.capacity, self.temperature) / np.sum(self.capacity))