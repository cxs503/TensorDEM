"""Hibler viscous-plastic sea-ice rheology in a 1-D periodic channel + buoy-scale
inertial-response probes.

Why a reduced 1-D configuration
-------------------------------
The question is: *does the internal ice stress change the observable
near-inertial response of a drifting ice cover, and by how much?*  The reduced
configuration keeps the whole 2-D stress tensor and the full plastic yield
surface but lets the fields vary along one horizontal direction ``x`` only (an
800 km periodic channel with a spatially modulated, time-switching wind).  Then

* strain rates are ``edot_11 = du/dx``, ``edot_22 = 0``, ``edot_12 = (1/2) dv/dx``
  -- the divergent and the shear mode are both present, which is what the
  plastic yield surface sees, and
* the domain-mean momentum balance is **exactly** the free-drift balance, since
  ``<d sigma/dx> = 0`` in a periodic domain (divergence theorem).  That identity
  is used as a zero-control: internal stress must not change the domain-mean
  motion at all; any measured change there would be a discretisation bug.

Rheology (Hibler 1979; CICE documentation eq. 3.33)
---------------------------------------------------
``P = P* h exp(-C (1 - A))``, ``zeta = P / (2 max(Delta, Delta_min))``,
``eta = zeta / e^2`` and (``edot_22 = 0`` in the channel)

``Delta = sqrt((1 + e^-2) edot_11^2 + e^-2 (dv/dx)^2)``,
``sigma_11 = (3 zeta - eta) edot_11 - P/2``,
``sigma_22 = (  zeta - eta) edot_11 - P/2``,
``sigma_12 = eta (dv/dx)``.

``Delta_min`` is a *regularisation*, not a physical constant: it bounds the
viscosity in the small-strain limit and therefore sets the effective viscosity a
low-strain-rate flow actually feels.  It is exposed and swept in the report.

Numerics
--------
Arakawa C-grid staggering (``u, v`` on faces; ``sigma``, ``zeta, eta, P`` on cell
centres), so the discrete stress divergence is compact (tridiagonal) and has no
odd-even null mode.  The momentum balance is advanced **semi-implicitly**:
Coriolis, the Newton-linearised drag Jacobian and the plastic stress divergence
are implicit in a ``2N x 2N`` sparse solve, while the plastic viscosities and the
pressure are Picard-lagged.  For ``P* -> 0`` the same discretisation reduces
exactly to the constant-coefficient linear problem of
:mod:`tensordem.sea_ice_transient`, which is what makes the ``P* -> 0``
degeneration test meaningful.

All quantities SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .sea_ice_drift import coriolis_parameter

__all__ = [
    "ChannelConfig",
    "HiblerRheology",
    "OneDimVPIceChannel",
    "linear_viscous_prediction",
]


# ---------------------------------------------------------------------------
# Rheology
# ---------------------------------------------------------------------------

@dataclass
class HiblerRheology:
    """Hibler (1979) viscous-plastic rheology parameters.

    ``p_star``: ice strength ``P*`` [N/m]; ``P = P* h exp(-C (1 - A))``.
    ``ellipse``: yield-curve aspect ratio ``e`` (``eta = zeta/e^2``; ``e = 2``).
    ``c_concentration``: strength-vs-concentration exponent ``C`` (Hibler: 20).
    ``delta_min``: strain-rate regularisation [1/s]; bounds ``zeta`` by
    ``P/(2 delta_min)``.  Not a physical constant (swept in the report).
    """

    p_star: float = 2.75e4
    ellipse: float = 2.0
    c_concentration: float = 20.0
    delta_min: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.p_star < 0.0:
            raise ValueError("P* must be non-negative")
        if self.ellipse <= 1.0:
            raise ValueError("ellipse ratio e must exceed 1")
        if self.delta_min <= 0.0:
            raise ValueError("delta_min must be positive")

    def pressure(self, thickness: np.ndarray, concentration: np.ndarray) -> np.ndarray:
        """``P = P* h exp(-C(1-A))`` [N/m]; identically zero for ``P* = 0``."""
        h = np.asarray(thickness, dtype=float)
        a = np.clip(np.asarray(concentration, dtype=float), 0.0, 1.0)
        return self.p_star * h * np.exp(-self.c_concentration * (1.0 - a))

    def strain_rate_invariants(
        self, div_u: np.ndarray, shear: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(Delta, Delta_reg)`` with ``edot_22 = 0``: ``Delta^2 = (1+e^-2) d11^2 + e^-2 shear^2``."""
        e = self.ellipse
        d11 = np.asarray(div_u, dtype=float)
        sh = np.asarray(shear, dtype=float)
        delta = np.sqrt((1.0 + e ** -2) * d11**2 + e ** -2 * sh**2)
        return delta, np.maximum(delta, self.delta_min)

    def viscosities(
        self, pressure: np.ndarray, div_u: np.ndarray, shear: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """``(zeta, eta, plastic_fraction)`` for the given strain rates."""
        p = np.asarray(pressure, dtype=float)
        delta, delta_reg = self.strain_rate_invariants(div_u, shear)
        zeta = np.where(delta_reg > 0.0, p / (2.0 * np.where(delta_reg > 0.0, delta_reg, 1.0)), 0.0)
        eta = zeta / self.ellipse**2
        plastic = float(np.mean(delta >= self.delta_min)) if delta.size else 0.0
        return zeta, eta, plastic

    def stress(
        self,
        zeta: np.ndarray,
        eta: np.ndarray,
        pressure: np.ndarray,
        div_u: np.ndarray,
        shear: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        z = np.asarray(zeta, dtype=float)
        n = np.asarray(eta, dtype=float)
        p = np.asarray(pressure, dtype=float)
        d11 = np.asarray(div_u, dtype=float)
        sh = np.asarray(shear, dtype=float)
        return {
            "sigma_11": (3.0 * z - n) * d11 - 0.5 * p,
            "sigma_22": (z - n) * d11 - 0.5 * p,
            "sigma_12": n * sh,
        }


# ---------------------------------------------------------------------------
# Analytic linear-viscous prediction
# ---------------------------------------------------------------------------

def linear_viscous_prediction(
    p_star: float,
    rheology: HiblerRheology,
    mass_per_area: float,
    f: float,
    gamma_drag: float,
    wavelength: float,
    thickness: float = 1.0,
    concentration: float = 1.0,
) -> Dict[str, float]:
    """Frozen-viscosity analytic prediction of the extra damping ``(Delta gamma, Delta theta)``.

    For a plane perturbation ``exp(i k x + s t)`` of the linearised momentum
    balance with constant viscosities, the two velocity components obey

        s u =  f v - gamma_u u ,     s v = -f u - gamma_v v ,
        gamma_u = gamma_drag + 3 zeta k^2 / m ,   gamma_v = gamma_drag + eta k^2 / m

    (``gamma_u`` carries the normal-stress viscosity ``3 zeta`` of
    ``sigma_11 = (3 zeta - eta) edot_11``; ``gamma_v`` carries the shear
    viscosity ``eta`` of ``sigma_12 = eta dv/dx``).  Hence

        gamma_eff = gamma_drag + (3 zeta + eta) k^2 / (2 m)   ->  Delta gamma
        omega_eff = sqrt(f^2 - delta^2),  delta = (3 zeta - 2 eta) k^2 / (2 m)

    and the deflection angle follows from ``theta = arctan(omega_eff/gamma_eff)``.
    ``omega_eff`` is real only while ``delta < f``; beyond that the system is
    overdamped and the oscillation ceases (then ``Delta theta`` is reported as the
    static limit and flagged ``overdamped=True``).
    """
    k = 2.0 * math.pi / wavelength
    p = rheology.p_star * 0.0
    rh = HiblerRheology(
        p_star=p_star,
        ellipse=rheology.ellipse,
        c_concentration=rheology.c_concentration,
        delta_min=rheology.delta_min,
    )
    p = rh.p_star * thickness * math.exp(-rh.c_concentration * (1.0 - concentration))
    zeta = p / (2.0 * rh.delta_min)
    eta = zeta / rh.ellipse**2
    nu_u = 3.0 * zeta / mass_per_area
    nu_v = eta / mass_per_area
    gamma_u = gamma_drag + nu_u * k**2
    gamma_v = gamma_drag + nu_v * k**2
    gamma_eff = 0.5 * (gamma_u + gamma_v)
    delta = 0.5 * (gamma_u - gamma_v)
    overdamped = bool(delta >= abs(f))
    omega_eff = 0.0 if overdamped else math.sqrt(abs(f) ** 2 - delta**2)
    theta0 = math.degrees(math.atan2(abs(f), gamma_drag)) if gamma_drag > 0 else 90.0
    theta_eff = math.degrees(math.atan2(omega_eff, gamma_eff)) if gamma_eff > 0 else 90.0
    return {
        "p_star": p_star,
        "zeta": zeta,
        "eta": eta,
        "k": k,
        "nu_u": nu_u,
        "nu_v": nu_v,
        "gamma_drag": gamma_drag,
        "gamma_eff": gamma_eff,
        "delta_gamma_over_gamma": (gamma_eff - gamma_drag) / gamma_drag if gamma_drag else float("inf"),
        "omega_eff": omega_eff,
        "omega_rel_shift": (omega_eff - abs(f)) / abs(f),
        "overdamped": overdamped,
        "theta0_deg": theta0,
        "theta_eff_deg": theta_eff,
        "delta_theta_deg": theta_eff - theta0,
    }


# ---------------------------------------------------------------------------
# Configuration and forcing
# ---------------------------------------------------------------------------

@dataclass
class ChannelConfig:
    """Geometry / forcing / numerics of the 1-D periodic ice channel."""

    n_cells: int = 96
    length: float = 8.0e5
    latitude_deg: float = 80.0
    ice_thickness: float = 6.0
    concentration: float = 0.95
    ice_density: float = 900.0
    rho_air: float = 1.25
    rho_water: float = 1025.0
    c_d_air: float = 0.0018
    c_d_water: float = 0.0045
    dt: float = 900.0
    duration_days: float = 30.0
    wind_speed: float = 10.0
    wind_angle_deg: float = 25.0
    wind_spatial_amplitude: float = 0.55
    wind_switch_period_days: float = 2.0
    wind_switch_angle_deg: float = 60.0
    wind_switch_speed_ratio: float = 1.0
    wind_variability: float = 2.0
    wind_correlation_time: float = 86400.0
    buoy_window_length: float = 2.0e5
    buoy_x0: float = 4.0e5
    picard_iterations: int = 3
    h_min: float = 0.05

    def __post_init__(self) -> None:
        if self.n_cells < 8:
            raise ValueError("need at least 8 cells")
        if self.length <= 0.0 or self.dt <= 0.0:
            raise ValueError("length and dt must be positive")

    @property
    def dx(self) -> float:
        return self.length / self.n_cells

    @property
    def f(self) -> float:
        return coriolis_parameter(self.latitude_deg)

    @property
    def mass_per_area(self) -> float:
        return self.ice_density * self.ice_thickness

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_days * 86400.0 / self.dt))

    @property
    def buoy_cells(self) -> int:
        return max(1, int(round(self.buoy_window_length / self.dx)))


def _spatial_modulation(x: np.ndarray, length: float, amplitude: float) -> np.ndarray:
    """Deterministic multi-mode spatial structure of the wind."""
    s = np.zeros_like(x)
    for m, phi in zip((1.0, 2.0, 3.0), (0.0, 1.1, 2.4)):
        s += (amplitude / m) * np.cos(2.0 * math.pi * m * x / length + phi)
    return s


def _ou_history(
    n_steps: int, dt: float, sigma: float, tau: float, rng: np.random.Generator
) -> np.ndarray:
    """Exact-discretisation Ornstein-Uhlenbeck process (zero mean, std ``sigma``)."""
    if sigma == 0.0:
        return np.zeros(n_steps + 1)
    phi = math.exp(-dt / tau)
    innov = sigma * math.sqrt(max(0.0, 1.0 - phi * phi))
    out = np.zeros(n_steps + 1)
    noise = rng.standard_normal(n_steps)
    for n in range(n_steps):
        out[n + 1] = phi * out[n] + innov * noise[n]
    return out


class WindDirector:
    """Wind forcing: base wind + spatial structure + abrupt 2-day re-orientations.

    The switch schedule is *deterministic* (identical for every ensemble member),
    so the response is a superposition of responses to known events; only the OU
    variability is seeded.  That is what makes the decay-rate estimator a
    well-posed linear least-squares problem.
    """

    def __init__(self, config: ChannelConfig, rng: np.random.Generator):
        self.config = config
        x = (np.arange(config.n_cells) + 0.5) * config.dx
        self.modulation = 1.0 + _spatial_modulation(
            x, config.length, config.wind_spatial_amplitude
        )
        self.ou = _ou_history(
            config.n_steps, config.dt, config.wind_variability,
            config.wind_correlation_time, rng,
        )
        self.switch_steps = max(
            1, int(round(config.wind_switch_period_days * 86400.0 / config.dt))
        )

    def switch_times(self) -> np.ndarray:
        c = self.config
        n_sw = c.n_steps // self.switch_steps
        return np.arange(1, n_sw + 1) * self.switch_steps * c.dt

    def angle_deg(self, step: int) -> float:
        c = self.config
        k = step // self.switch_steps
        return c.wind_angle_deg + (c.wind_switch_angle_deg if k % 2 else 0.0)

    def speed_factor(self, step: int) -> float:
        """Alternating wind-speed factor (excitation without changing direction)."""
        c = self.config
        k = step // self.switch_steps
        return c.wind_switch_speed_ratio if k % 2 else 1.0

    def wind(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        c = self.config
        angle = math.radians(self.angle_deg(step))
        speed = ((c.wind_speed + self.ou[min(step, self.ou.size - 1)])
                 * self.modulation * self.speed_factor(step))
        speed = np.maximum(speed, 0.0)
        return speed * math.cos(angle), speed * math.sin(angle)


# ---------------------------------------------------------------------------
# The channel model
# ---------------------------------------------------------------------------

class OneDimVPIceChannel:
    """Semi-implicit Hibler VP ice channel with buoy-scale diagnostics.

    Arrays (periodic, length ``n_cells``): ``u, v`` face velocities;
    ``h, A`` cell thickness/concentration; ``zeta, eta, P`` cell rheology.
    """

    def __init__(
        self,
        config: ChannelConfig,
        rheology: Optional[HiblerRheology] = None,
        seed: int = 0,
    ):
        self.config = config
        self.rheology = rheology if rheology is not None else HiblerRheology()
        self.rng = np.random.default_rng(seed)
        self.wind = WindDirector(config, self.rng)
        n = config.n_cells
        self.u = np.zeros(n)
        self.v = np.zeros(n)
        self.h = np.full(n, config.ice_thickness)
        self.A = np.full(n, config.concentration)
        self.P = self.rheology.pressure(self.h, self.A)
        self.zeta = np.zeros(n)
        self.eta = np.zeros(n)
        self._icell = np.arange(n)
        self._im1 = (np.arange(n) - 1) % n
        self._ip1 = (np.arange(n) + 1) % n
        start = int(config.buoy_x0 / config.dx)
        self._buoy = (np.arange(config.buoy_cells) + start) % n
        self._current_step = 0

    # -- diagnostics --------------------------------------------------------
    def divergence(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """``div u`` on cell centres ``= (u_{i+1} - u_i)/dx``."""
        u = self.u if u is None else u
        return (u[self._ip1] - u) / self.config.dx

    def shear_rate(self, v: Optional[np.ndarray] = None) -> np.ndarray:
        """``dv/dx`` on cell centres."""
        v = self.v if v is None else v
        return (v[self._ip1] - v) / self.config.dx

    def buoy_velocity(self) -> np.ndarray:
        """Drift-station mean velocity over the ``buoy_window_length`` footprint."""
        return np.array([np.mean(self.u[self._buoy]), np.mean(self.v[self._buoy])])

    def domain_velocity(self) -> np.ndarray:
        return np.array([np.mean(self.u), np.mean(self.v)])

    def rheology_state(self) -> Dict[str, object]:
        div_u = self.divergence()
        shear = self.shear_rate()
        zeta, eta, plastic = self.rheology.viscosities(self.P, div_u, shear)
        return {
            "div_u": div_u,
            "shear": shear,
            "zeta": zeta,
            "eta": eta,
            "pressure": self.P,
            "plastic_fraction": plastic,
        }

    def stress_divergence(self) -> Dict[str, np.ndarray]:
        """Discrete ``(d sigma_11/dx, d sigma_12/dx)`` at the faces.

        Uses the same stencil as the momentum assembly, so its periodic mean is
        identically zero (``<d sigma/dx> = 0``): the internal stress cannot change
        the domain-mean momentum balance, only redistribute it.
        """
        div_u = self.divergence()
        shear = self.shear_rate()
        sig = self.rheology.stress(self.zeta, self.eta, self.P, div_u, shear)
        dx = self.config.dx
        d11 = (sig["sigma_11"] - sig["sigma_11"][self._im1]) / dx
        d12 = (sig["sigma_12"] - sig["sigma_12"][self._im1]) / dx
        return {"d_sigma_11_dx": d11, "d_sigma_12_dx": d12,
                "sigma_11": sig["sigma_11"], "sigma_12": sig["sigma_12"]}

    def wind_vector_mean(self, step: int) -> np.ndarray:
        wa, wb = self.wind.wind(step)
        return np.array([float(np.mean(wa)), float(np.mean(wb))])

    # -- one step -----------------------------------------------------------
    def _drag(
        self, u_face: np.ndarray, v_face: np.ndarray, rho: float, c_d: float,
        ocean: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Quadratic drag stress and its Jacobian w.r.t. ``(u, v)``."""
        if ocean:
            du_x = -u_face
            du_y = -v_face
        else:
            wa, wb = self.wind.wind(self._current_step)
            du_x = wa - u_face
            du_y = wb - v_face
        mag = np.hypot(du_x, du_y)
        coef = rho * c_d
        tau = np.column_stack([coef * mag * du_x, coef * mag * du_y])
        safe = np.maximum(mag, 1e-9)
        eye = np.eye(2)[None, :, :]
        outer = np.empty((mag.size, 2, 2))
        outer[:, 0, 0] = du_x * du_x / safe
        outer[:, 0, 1] = du_x * du_y / safe
        outer[:, 1, 0] = du_x * du_y / safe
        outer[:, 1, 1] = du_y * du_y / safe
        jac = -coef * (mag[:, None, None] * eye + outer)
        return tau, jac

    def _assemble(
        self, un: np.ndarray, vn: np.ndarray, zeta: np.ndarray, eta: np.ndarray,
        pressure: np.ndarray,
    ) -> Tuple[sp.csr_matrix, np.ndarray]:
        c = self.config
        n = c.n_cells
        m = c.mass_per_area
        dx2 = c.dx * c.dx
        a = 3.0 * zeta - eta            # coefficient of edot_11 in sigma_11
        b = eta                         # coefficient of dv/dx in sigma_12
        a_im1 = a[self._im1]
        b_im1 = b[self._im1]
        diag_u = m / c.dt + (a + a_im1) / dx2
        diag_v = m / c.dt + (b + b_im1) / dx2

        tau_w, jac_w = self._drag(un, vn, c.rho_water, c.c_d_water, ocean=True)
        tau_a, jac_a = self._drag(un, vn, c.rho_air, c.c_d_air, ocean=False)
        jac = jac_w + jac_a
        diag_u = diag_u - jac[:, 0, 0]
        diag_v = diag_v - jac[:, 1, 1]

        rows = np.concatenate([
            self._icell, self._icell, self._icell, self._icell,
            n + self._icell, n + self._icell, n + self._icell, n + self._icell,
        ])
        cols = np.concatenate([
            self._icell, self._ip1, self._im1, n + self._icell,
            n + self._icell, n + self._ip1, n + self._im1, self._icell,
        ])
        vals = np.concatenate([
            diag_u, -a / dx2, -a_im1 / dx2, -m * c.f - jac[:, 0, 1],
            diag_v, -b / dx2, -b_im1 / dx2, +m * c.f - jac[:, 1, 0],
        ])
        A_op = sp.coo_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n)).tocsr()

        src_p = -(pressure - pressure[self._im1]) / (2.0 * c.dx)
        rhs_u = m * un / c.dt + tau_a[:, 0] + tau_w[:, 0] - (
            jac[:, 0, 0] * un + jac[:, 0, 1] * vn
        ) + src_p
        rhs_v = m * vn / c.dt + tau_a[:, 1] + tau_w[:, 1] - (
            jac[:, 1, 0] * un + jac[:, 1, 1] * vn
        )
        return A_op, np.concatenate([rhs_u, rhs_v])

    def _advance_scalars(self) -> None:
        """First-order upwind transport of ``h`` and ``A`` (donor cell)."""
        c = self.config
        h, A, u = self.h, self.A, self.u
        up_h = np.where(u > 0.0, h[self._im1], h)
        up_a = np.where(u > 0.0, A[self._im1], A)
        flux_h = u * up_h
        flux_a = u * up_a
        self.h = np.maximum(h - c.dt / c.dx * (flux_h[self._ip1] - flux_h), c.h_min)
        self.A = np.clip(A - c.dt / c.dx * (flux_a[self._ip1] - flux_a), 0.0, 1.0)

    def step(self, step: int) -> None:
        self._current_step = step
        c = self.config
        n = c.n_cells
        un, vn = self.u.copy(), self.v.copy()
        div_u = self.divergence(un)
        shear = self.shear_rate(vn)
        P = self.rheology.pressure(self.h, self.A)
        zeta, eta, _ = self.rheology.viscosities(P, div_u, shear)
        A_op, rhs = self._assemble(un, vn, zeta, eta, P)
        sol = spla.spsolve(A_op, rhs)
        uk, vk = sol[:n], sol[n:]
        for _ in range(max(1, c.picard_iterations) - 1):
            div_u = (uk[self._ip1] - uk) / c.dx
            shear = (vk[self._ip1] - vk) / c.dx
            zeta, eta, _ = self.rheology.viscosities(P, div_u, shear)
            A_op, rhs = self._assemble(uk, vk, zeta, eta, P)
            sol = spla.spsolve(A_op, rhs)
            uk, vk = sol[:n], sol[n:]
        self.u, self.v = uk, vk
        self.zeta, self.eta = zeta, eta
        self._advance_scalars()
        self.P = self.rheology.pressure(self.h, self.A)

    # -- record -------------------------------------------------------------
    def run(self, record: bool = True) -> Dict[str, object]:
        """March the full record and return the diagnostics used for the fit."""
        c = self.config
        ns = c.n_steps
        times = np.zeros(ns + 1)
        buoy = np.zeros((ns + 1, 2))
        domain = np.zeros((ns + 1, 2))
        div_stats = np.zeros((ns + 1, 3))
        wind = np.zeros((ns + 1, 2))
        plastic = np.zeros(ns + 1)
        max_zeta = np.zeros(ns + 1)
        buoy[:] = self.buoy_velocity()
        domain[:] = self.domain_velocity()
        wind[0] = self.wind_vector_mean(0)
        for n in range(ns):
            self.step(n)
            times[n + 1] = (n + 1) * c.dt
            buoy[n + 1] = self.buoy_velocity()
            domain[n + 1] = self.domain_velocity()
            div = self.divergence()
            div_stats[n + 1] = (
                float(np.mean(np.abs(div))),
                float(np.max(-div)),
                float(np.max(div)),
            )
            wind[n + 1] = self.wind_vector_mean(n + 1)
            plastic[n + 1] = self.rheology_state()["plastic_fraction"]
            max_zeta[n + 1] = float(np.max(self.zeta))
        final = self.rheology_state()
        return {
            "times": times,
            "buoy": buoy,
            "domain": domain,
            "wind": wind,
            "div_stats": div_stats,
            "plastic_fraction": plastic,
            "max_zeta": max_zeta,
            "switch_times": self.wind.switch_times(),
            "final_state": {
                "h": self.h.copy(),
                "A": self.A.copy(),
                "u": self.u.copy(),
                "v": self.v.copy(),
                "div_u": final["div_u"],
                "sigma_11": self.rheology.stress(
                    self.zeta, self.eta, self.P, final["div_u"], final["shear"]
                )["sigma_11"],
                "pressure": self.P.copy(),
            },
            "config": c,
            "rheology": self.rheology,
        }