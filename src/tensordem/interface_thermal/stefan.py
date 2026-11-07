"""Stefan (phase-change) model of the melt layer at an ice-structure interface.

Two things live here.

**Neumann's similarity solution** -- the exact reference for one-dimensional
melting of a semi-infinite solid initially at its melting temperature ``Tm``
whose surface is suddenly raised to ``Tw > Tm``.  The melt front is
``X(t) = 2 lambda sqrt(alpha t)`` with ``lambda`` from

    lambda exp(lambda^2) erf(lambda) = Ste / sqrt(pi),
    Ste = c_p (Tw - Tm) / L,

and the solid is at ``Tm`` everywhere, so the Stefan condition reduces to
``rho L dX/dt = k dT/dx|_X``.

**An enthalpy solver on a fixed grid** (:class:`EnthalpyStefan1D`) -- a
numerically independent route to the same physics.  The latent heat is spread
over a small one-sided melting band ``[Tm, Tm + 2 dT_melt]``, and the discrete
system

    C (H(T^{n+1}) - H(T^n)) / dt = theta F(T^{n+1}) + (1 - theta) F(T^n)

is solved by **Newton iteration on the enthalpy residual**, with
``dH/dT = c_eff(T)`` as the Jacobian.  That detail matters: an apparent-heat-
capacity *Picard* iteration silently loses the latent heat whenever a cell
crosses the whole melting band in one step (measured on a 0.01 m / 1201 cell
grid at ``dt = 0.01 s``: the Picard version melted 0.21 mm of ice while the
surface had admitted only 16.4 kJ/m^2, i.e. 4x the available energy), because
the ``c_eff`` that would absorb the latent heat is never sampled.  The Newton
form is written in terms of ``H`` itself, so the sum of the residuals is
``sum C (H^{n+1} - H^n) = dt * (boundary flux)`` and the ledger closes to round
off.

The two routes share no code, so comparing them measures the solver rather than
the formula.  ``dT_melt -> 0`` is the classical Stefan problem; the benchmark
halves ``dT_melt`` to show convergence to the Neumann front.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import erf

__all__ = [
    "neumann_lambda",
    "neumann_interface_position",
    "neumann_interface_velocity",
    "neumann_temperature",
    "neumann_stefan_number",
    "EnthalpyStefan1D",
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


def neumann_stefan_number(superheat_k: float, latent_heat_j_kg: float,
                          cp_j_kg_k: float) -> float:
    """``Ste = c_p (Tw - Tm) / L`` (single phase, liquid-side properties)."""
    superheat = _finite("superheat_k", superheat_k, nonnegative=True)
    latent = _finite("latent_heat_j_kg", latent_heat_j_kg, positive=True)
    cp = _finite("cp_j_kg_k", cp_j_kg_k, positive=True)
    return cp * superheat / latent


def neumann_lambda(stefan_number: float) -> float:
    """Solve ``lambda e^{lambda^2} erf(lambda) = Ste / sqrt(pi)`` for lambda.

    The left-hand side ``lambda e^{lambda^2} erf(lambda)`` increases strictly
    from zero, so the root is unique.  For ``Ste << 1``, ``lambda ~ (Ste/2)^{1/3}``.
    """
    ste = _finite("stefan_number", stefan_number, nonnegative=True)
    if ste == 0.0:
        return 0.0
    target = ste / math.sqrt(math.pi)

    def residual(lam: float) -> float:
        return lam * math.exp(lam * lam) * erf(lam) - target

    return brentq(residual, 0.0, 20.0, xtol=1e-15, rtol=1e-15, maxiter=300)


def neumann_interface_position(t: float, lam: float, alpha: float) -> np.ndarray:
    """Melt-front position ``X(t) = 2 lambda sqrt(alpha t)`` in m."""
    lam = _finite("lam", lam, nonnegative=True)
    alpha = _finite("alpha", alpha, positive=True)
    t = np.asarray(t, dtype=float)
    if np.any(t < 0.0):
        raise ValueError("t must be nonnegative")
    return 2.0 * lam * np.sqrt(alpha * np.maximum(t, 0.0))


def neumann_interface_velocity(t: float, lam: float, alpha: float) -> np.ndarray:
    """Melt-front speed ``dX/dt = lambda sqrt(alpha / t)`` in m/s."""
    lam = _finite("lam", lam, nonnegative=True)
    alpha = _finite("alpha", alpha, positive=True)
    t = np.asarray(t, dtype=float)
    if np.any(t <= 0.0):
        raise ValueError("t must be positive for a front speed")
    return lam * np.sqrt(alpha / t)


def neumann_temperature(
    x: Sequence[float], t: float, lam: float, alpha: float,
    t_surface: float, t_melt: float,
) -> np.ndarray:
    """Liquid-side Neumann profile (zero initial superheat).

    ``T = Tm + (Tw - Tm) [1 - erf(x/(2 sqrt(alpha t))) / erf(lambda)]`` for
    ``x <= X(t)``; the solid stays at ``Tm`` ahead of the front.
    """
    lam = _finite("lam", lam)
    alpha = _finite("alpha", alpha, positive=True)
    t = _finite("t", t, nonnegative=True)
    x = np.asarray(x, dtype=float)
    if t == 0.0:
        return np.full_like(x, float(t_melt))
    profile = t_melt + (t_surface - t_melt) * (
        1.0 - erf(x / (2.0 * math.sqrt(alpha * t))) / erf(lam))
    front = float(neumann_interface_position(t, lam, alpha))
    return np.where(x <= front, profile, float(t_melt))


@dataclass
class EnthalpyStefan1D:
    """One-dimensional melting with an enthalpy method on a fixed grid.

    Geometry: a semi-infinite ice body occupying ``x >= 0``, initially at
    ``t_initial <= t_melt``, whose surface ``x = 0`` is driven by a fixed
    temperature ``t_surface`` (Dirichlet) or a fixed flux ``flux_w_m2``.

    The per-unit-mass enthalpy above ``t_initial`` is ``H(T) = int c_eff dT``
    with

    * ``c_eff = cp_solid`` for ``T <= Tm``,
    * ``c_eff = cp_solid + (cp_liquid - cp_solid) f_l + L/(2 dT_melt)`` on the
      one-sided band ``Tm <= T <= Tm + 2 dT_melt``,
    * ``c_eff = cp_liquid`` for ``T >= Tm + 2 dT_melt``,

    where ``f_l = (T - Tm)/(2 dT_melt)`` clipped to [0, 1] is the liquid
    fraction.  The band is placed *above* the melting point so that a solid at
    ``Tm`` has ``f_l = 0`` exactly (a symmetric band would report half of the
    domain as melted before the first step).  The melt thickness is
    ``X = int f_l dx``.
    """

    length_m: float = 0.01
    cells: int = 1201
    k_w_m_k: float = 0.56
    rho_kg_m3: float = 1000.0
    cp_j_kg_k: float = 4200.0
    cp_solid_j_kg_k: float = 2100.0
    latent_heat_j_kg: float = 3.34e5
    t_melt: float = 0.0
    t_initial: float = 0.0
    dT_melt: float = 0.01
    grading: float = 1.0
    theta: float = 1.0
    t: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("length_m", "k_w_m_k", "rho_kg_m3", "cp_j_kg_k",
                     "cp_solid_j_kg_k", "latent_heat_j_kg", "dT_melt"):
            setattr(self, name, _finite(name, getattr(self, name), positive=True))
        self.t_melt = _finite("t_melt", self.t_melt)
        self.t_initial = _finite("t_initial", self.t_initial)
        if self.t_initial > self.t_melt:
            raise ValueError("the ice must start at or below the melting point")
        self.theta = _finite("theta", self.theta)
        if not 0.0 < self.theta <= 1.0:
            raise ValueError("theta must lie in (0, 1]")
        self.grading = _finite("grading", self.grading, positive=True)
        if isinstance(self.cells, bool) or not isinstance(self.cells, int):
            raise TypeError("cells must be an integer")
        if self.cells < 5:
            raise ValueError("cells must be at least 5")
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
            self.x = np.concatenate(([0.0], np.cumsum(dx * r ** np.arange(n - 1))))
        dx = np.diff(self.x)
        self.dx = dx
        self.cell_width = np.empty(n)
        self.cell_width[0] = 0.5 * dx[0]
        self.cell_width[-1] = 0.5 * dx[-1]
        self.cell_width[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        self.conductance = self.k_w_m_k / dx
        self.ghost_conductance = 2.0 * self.k_w_m_k / dx[-1]
        self.capacity_scale = self.rho_kg_m3 * self.cell_width

    def reset(self) -> None:
        self.t = np.full(self.cells, self.t_initial, dtype=float)

    # -- constitutive ---------------------------------------------------
    @property
    def band(self) -> float:
        """Width ``2 dT_melt`` of the regularised melting band."""
        return 2.0 * self.dT_melt

    def liquid_fraction(self, t: np.ndarray) -> np.ndarray:
        """``f_l = clip((T - Tm)/(2 dT_melt), 0, 1)`` -- zero at ``Tm``."""
        return np.clip((np.asarray(t, dtype=float) - self.t_melt) / self.band, 0.0, 1.0)

    def effective_cp(self, t: np.ndarray) -> np.ndarray:
        """``dH/dT`` (J/(kg K)), latent heat included inside the band."""
        t = np.asarray(t, dtype=float)
        liquid = self.liquid_fraction(t)
        sensible = (self.cp_solid_j_kg_k * (1.0 - liquid)
                    + self.cp_j_kg_k * liquid)
        inside = ((t > self.t_melt) & (t < self.t_melt + self.band)).astype(float)
        return sensible + self.latent_heat_j_kg / self.band * inside

    def enthalpy(self, t: np.ndarray) -> np.ndarray:
        """Per-unit-mass enthalpy above ``t_initial`` (J/kg).

        Piecewise integral of :meth:`effective_cp`, continuous and strictly
        increasing in ``T`` -- the property the Newton solve relies on.
        """
        t = np.asarray(t, dtype=float)
        tm, band = self.t_melt, self.band
        cs, cl, lat = (self.cp_solid_j_kg_k, self.cp_j_kg_k,
                       self.latent_heat_j_kg)
        below = cs * (np.minimum(t, tm) - self.t_initial)
        excess = np.clip(t - tm, 0.0, band)
        band_enthalpy = (cs * excess + (cl - cs) * excess ** 2 / (2.0 * band)
                         + lat * excess / band)
        above = cl * np.maximum(t - tm - band, 0.0)
        return below + band_enthalpy + above

    def melt_thickness(self) -> float:
        """Melted layer thickness ``X = int f_l dx`` in m."""
        return float(np.sum(self.liquid_fraction(self.t) * self.cell_width))

    def latent_energy(self) -> float:
        """Latent heat absorbed per unit area (J/m^2)."""
        return self.rho_kg_m3 * self.latent_heat_j_kg * self.melt_thickness()

    def sensible_energy(self) -> float:
        """Sensible heat above the initial state per unit area (J/m^2)."""
        h = self.enthalpy(self.t) - self.latent_heat_j_kg * self.liquid_fraction(self.t)
        return float(np.sum(self.capacity_scale * h))

    def total_energy(self) -> float:
        """Enthalpy stored above the initial state per unit area (J/m^2)."""
        return float(np.sum(self.capacity_scale * self.enthalpy(self.t)))

    # -- solver ---------------------------------------------------------
    def _flux_operator(self, t: np.ndarray, flux_surface: float,
                       t_far: float | None) -> np.ndarray:
        flux = self.conductance * (t[1:] - t[:-1])
        out = np.zeros_like(t)
        # net *inflow* convention: op = A_std T + boundary sources, i.e. the
        # term that heats a cell (the implicit matrix is C/dt - theta * op)
        out[1:] -= flux
        out[:-1] += flux
        out[0] += flux_surface
        if t_far is not None:
            out[-1] += self.ghost_conductance * (t_far - t[-1])
        return out

    def temperature_from_enthalpy(self, h: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`enthalpy` (piecewise, monotone, vectorised)."""
        h = np.asarray(h, dtype=float)
        tm, band = self.t_melt, self.band
        cs, cl, lat = (self.cp_solid_j_kg_k, self.cp_j_kg_k,
                       self.latent_heat_j_kg)
        h1 = cs * (tm - self.t_initial)
        h2 = h1 + cs * band + (cl - cs) * band / 2.0 + lat
        solid = self.t_initial + h / cs
        a = (cl - cs) / (2.0 * band)
        b = cs + lat / band
        u = (-b + np.sqrt(b * b + 4.0 * a * (h - h1))) / (2.0 * a)
        liquid = tm + band + (h - h2) / cl
        return np.where(h <= h1, solid,
                        np.where(h >= h2, liquid, tm + u))

    def step(self, dt: float, *, flux_w_m2: float = 0.0,
             t_surface: float | None = None, t_far: float | None = None,
             max_iterations: int = 60, tolerance: float = 1e-10) -> np.ndarray:
        """Advance one implicit step by Newton iteration **on the enthalpy**.

        Either a surface flux (Neumann) or a surface temperature (Dirichlet,
        strongly imposed) may be prescribed.  The unknown is the cell enthalpy

            ``R_j(H) = C_j (H_j - H_j^n)/dt - theta F_j(T(H)) - (1-theta) F_j(T^n)``

        so the sum of the residuals is the discrete energy balance
        ``sum C (H - H^n) = dt * (boundary flux)`` and the latent heat cannot be
        lost.  Iterating on ``H`` rather than on ``T`` is what makes the band
        tractable: ``dT/dH = 1/c_eff`` is *small* inside the band, while
        ``c_eff = dH/dT`` is enormous there -- a temperature-based Newton
        overshoots across the band in one step (measured: 0.05 K -> -79 K).
        """
        dt = _finite("dt", dt, positive=True)
        flux = _finite("flux_w_m2", flux_w_m2)
        theta = self.theta
        n = self.cells
        old = self.t.copy()
        h_old = self.enthalpy(old)
        base = self._flux_operator(old, flux, t_far)
        enthalpy = h_old.copy()
        if t_surface is not None:
            enthalpy[0] = float(self.enthalpy(np.array([float(t_surface)]))[0])
        t = self.temperature_from_enthalpy(enthalpy)
        g = self.conductance
        for _ in range(max_iterations):
            t = self.temperature_from_enthalpy(enthalpy)
            sigma = 1.0 / self.effective_cp(t)          # dT/dH
            diag = self.capacity_scale / dt
            diag[:-1] += theta * sigma[:-1] * g
            diag[1:] += theta * sigma[1:] * g
            upper = -theta * sigma[1:] * g
            lower = -theta * sigma[:-1] * g
            residual = (self.capacity_scale * (enthalpy - h_old) / dt
                        - theta * self._flux_operator(t, flux, t_far)
                        - (1.0 - theta) * base)
            if t_surface is not None:
                diag[0] = 1.0
                upper[0] = 0.0
                residual[0] = enthalpy[0] - float(
                    self.enthalpy(np.array([float(t_surface)]))[0])
            ab = np.zeros((3, n))
            ab[0, 1:] = upper
            ab[1, :] = diag
            ab[2, :-1] = lower
            delta = solve_banded((1, 1), ab, -residual)
            enthalpy = enthalpy + delta
            if np.max(np.abs(delta)) <= tolerance * max(
                    1.0, np.max(np.abs(enthalpy))):
                break
        self.t = self.temperature_from_enthalpy(enthalpy)
        return self.t

    def run(self, t_end: float, dt: float, *,
            flux_w_m2: float = 0.0, t_surface: float | None = None,
            t_far: float | None = None,
            record_times: Sequence[float] | None = None) -> dict[str, np.ndarray]:
        """March to ``t_end``; record melt thickness, energy and profiles."""
        t_end = _finite("t_end", t_end, positive=True)
        dt = _finite("dt", dt, positive=True)
        targets = sorted(float(x) for x in (record_times or ()))
        pending = list(targets)
        time = 0.0
        hist = {"time": [], "melt_thickness": [], "surface_temperature": [],
                "latent_energy": [], "sensible_energy": [], "profiles": []}
        while time < t_end - 1e-15:
            step = dt if not pending else min(dt, max(pending[0] - time, 1e-15))
            self.step(step, flux_w_m2=flux_w_m2, t_surface=t_surface, t_far=t_far)
            time += step
            while pending and pending[0] <= time + 1e-12:
                pending.pop(0)
                hist["time"].append(time)
                hist["melt_thickness"].append(self.melt_thickness())
                hist["surface_temperature"].append(float(self.t[0]))
                hist["latent_energy"].append(self.latent_energy())
                hist["sensible_energy"].append(self.sensible_energy())
                hist["profiles"].append(self.t.copy())
        if not targets:
            hist["time"].append(time)
            hist["melt_thickness"].append(self.melt_thickness())
            hist["surface_temperature"].append(float(self.t[0]))
            hist["latent_energy"].append(self.latent_energy())
            hist["sensible_energy"].append(self.sensible_energy())
            hist["profiles"].append(self.t.copy())
        return {
            "time": np.array(hist["time"]),
            "melt_thickness": np.array(hist["melt_thickness"]),
            "surface_temperature": np.array(hist["surface_temperature"]),
            "latent_energy": np.array(hist["latent_energy"]),
            "sensible_energy": np.array(hist["sensible_energy"]),
            "profiles": np.array(hist["profiles"]),
            "x": self.x.copy(),
        }

    def interface_temperature(self) -> float:
        """Temperature where the liquid fraction crosses one half."""
        liquid = self.liquid_fraction(self.t)
        if liquid.max() < 0.5:
            return float(self.t[0])
        return float(self.t[int(np.argmax(liquid >= 0.5))])