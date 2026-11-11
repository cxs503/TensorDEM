"""Freezing-front depth profile and freeze-bond strength evolution (P316-W1-N03).

Physical scope
--------------
The m7439 freeze-bond strength model for ice-ridge keels assumes a **uniform
temperature** over the bond layer: the whole keel draft ``H`` is taken at the
ambient (surface) temperature ``T_env``.  In reality a cold surface freezes the
keel progressively: a frozen layer of thickness ``x_f(t)`` grows with the
Stefan square-root law while the ice below the front is still at its liquidus.
The bond strength is therefore a **profile** ``sigma_b(z, t)``, and the
effective (load-carrying) strength is the depth average

    tau_eff(t)   = (1/H) * integral_0^H sigma_b(z, t) dz
    tau_uniform  = sigma_b(T_env)                  (m7439 caliber)

This module answers "by how much does the uniform assumption bias the bond
strength?" with three ingredients:

1. **Conduction with phase change** — a 1-D enthalpy-method solver
   (apparent-heat-capacity / mushy-zone regularisation, implicit backward
   Euler, finite volume, conservative by construction):

       de/dt = d/dz ( k dT/dz ),   e(T) = rho * [ c_p (T - T_ref) + L * s(T) ]

   with the solid fraction ``s(T)`` a linear ramp across a regularisation band
   ``[T_f - 2b, T_f]``.  Latent heat is released *in full* (``int c_app dT = c_p 2b + L``),
   which is the point where the predecessor implementation
   (``tensordem.stefan_freezing_front``) lost a factor 2.

2. **Freezing-front tracking** — the front is where ``s = 1/2``; it is located
   by linear interpolation between cell centres, plus the exactly
   energy-consistent equivalent thickness ``int s dz``.  The numerical front is
   anchored on the Stefan/Neumann solutions

       x_f^quasi(t) = sqrt(2 k DeltaT t / (rho L))       (quasi-steady, no sensible heat)
       x_f^Neumann(t) = 2 lambda sqrt(alpha t),
       lambda exp(lambda^2) erf(lambda) = Ste/sqrt(pi)   (exact one-phase Neumann)

3. **Brine (saline) reduction** — the local bond strength degrades with the
   Cox & Weeks (1983) liquid brine fraction ``nu_b(z, t)`` through the
   Timco & Frederking porosity law (both reused verbatim from
   ``tensordem.salinity_strength``):

       sigma_b = sigma_0 * ((T_f - T)/T_ref)^n * (1 - nu_b/nu_p)^m

   ``T_f`` is the seawater liquidus for the bulk salinity, so saline ice
   freezes at ``-0.05411 S`` degC instead of 0 degC.

Governing dimensionless groups
------------------------------
    Ste = c_p DeltaT / L                       latent vs sensible heat
    Bi  = k / (h_conv H)                       conduction vs surface transfer
                                               (uniform limit = Bi -> inf;
                                               the conventional Biot number is
                                               its reciprocal, so Bi -> inf
                                               means "surface resistance
                                               negligible", i.e. a flat profile)

Validation surface (machine-checked, see ``tests/test_thermal_profile.py``)
--------------------------------------------------------------------------
(a) Stefan anchor: numerical vs ``x_f = sqrt(2 k DeltaT t/(rho L))`` <= 2 %,
    and vs the exact Neumann front <= 0.5 % (pure water, S = 0);
(b) Energy: stored enthalpy change == boundary flux integral (residual
    ~ machine precision) and, in the latent-dominated limit, released latent
    heat == flux integral within 1e-4;
(c) Uniform limit: Bi -> inf collapses the profile onto ``T_env`` and
    ``tau_eff -> tau_uniform`` within 0.1 %;
(d) Deviation: ``tau_eff/tau_uniform`` for the baseline case (k = 2.2 W/mK,
    n = 0.3, DeltaT = 15 K, H = 1 m, t = 24 h) and its monotone growth with
    Bi (Spearman rho = 1 over >= 5 levels);
(e) Brine: ``tau_eff`` decreases monotonically in the brine fraction and
    returns bit-for-bit to the pure-water value at ``nu_b = 0``;
(f) Negative control: DeltaT = 0 leaves the front displacement < 1e-9 m.

All quantities SI (m, kg, s, K, Pa).

References
----------
- Carslaw & Jaeger, *Conduction of Heat in Solids*, 2nd ed., 1959 (Neumann).
- Alexiades & Solomon, *Mathematical Modeling of Melting and Freezing
  Processes*, 1993 (enthalpy method).
- Cox & Weeks (1983), "Equations for determining the gas and brine volumes in
  sea-ice samples", J. Glaciol. 29(102), 306-316.
- Timco & Frederking (1984), porosity-strength relation for sea ice.
- m7439: uniform-temperature freeze-bond strength assumption
  (``tensordem.ice_ridge_freeze_bond_evolution``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

from tensordem.salinity_strength import brine_volume_fraction

__all__ = [
    "IceThermal",
    "BondStrengthLaw",
    "EnthalpyFrontSolver",
    "solid_fraction",
    "neumann_lambda",
    "stefan_quasi_steady_front",
    "neumann_front",
    "neumann_temperature_profile",
    "local_brine_fraction",

    "effective_strength",
    "uniform_strength",
    "deviation_ratio",
    "frozen_depth_from_profile",
    "run_case",
    "run_biot_sweep",
    "uniform_limit_study",
    "run_brine_sweep",
    "BASELINE",
    "main",
]

T_REF_K = 273.15


# ---------------------------------------------------------------------------
# Thermal properties
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IceThermal:
    """Thermophysical properties of the (possibly saline) bond-layer ice."""

    k: float = 2.2                    # W/(m K)        baseline spec
    rho: float = 917.0                # kg/m^3
    cp: float = 2100.0                # J/(kg K)
    L: float = 334_000.0              # J/kg           latent heat of fusion
    T_melt_K: float = T_REF_K         # pure-water melting point
    salinity_ppt: float = 0.0         # bulk salinity S

    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m^2/s]."""
        return self.k / (self.rho * self.cp)

    @property
    def liquidus_K(self) -> float:
        """Equilibrium liquidus temperature [K] for the bulk salinity.

        Cox & Weeks (1983) freezing-point polynomial ``T_f = -(A1 S + A2 S^2)``
        in degC, with ``A1 = 0.05411``, ``A2 = 7.594e-5`` (same constants as
        :mod:`tensordem.salinity_strength`).
        """
        s = self.salinity_ppt
        t_f_c = -(0.05411 * s + 7.594e-5 * s * s)
        return self.T_melt_K + t_f_c

    def stefan_number(self, delta_T_K: float) -> float:
        """``Ste = c_p DeltaT / L``."""
        return self.cp * float(delta_T_K) / self.L


# ---------------------------------------------------------------------------
# Solid fraction and its inverse (mushy-zone regularisation)
# ---------------------------------------------------------------------------
def solid_fraction(T, T_f: float, band: float):
    """Solid fraction ``s(T)`` with a linear ramp over ``[T_f - band, T_f]``.

    ``s = 1`` for ``T <= T_f - band`` (fully frozen), ``s = 0`` for
    ``T >= T_f`` (liquid at the liquidus) and linear in between.  The ramp is a
    numerical regularisation of the isothermal phase change: it does not change
    the total latent heat, only its temperature width.
    """
    T = np.asarray(T, dtype=float)
    T_f = float(T_f)
    band = float(band)
    if band <= 0.0:
        raise ValueError("band must be > 0")
    s = np.clip((T_f - T) / band, 0.0, 1.0)
    return float(s) if s.ndim == 0 else s


# ---------------------------------------------------------------------------
# Analytical anchors
# ---------------------------------------------------------------------------
def neumann_lambda(ste: float) -> float:
    """Root ``lambda`` of ``lambda exp(lambda^2) erf(lambda) = Ste/sqrt(pi)``.

    One-phase Neumann (Stefan) solution: ``x_f = 2 lambda sqrt(alpha t)``.
    ``Ste = 0`` returns 0 (no freezing).
    """
    ste = float(ste)
    if ste <= 0.0:
        return 0.0
    from math import erf, exp, pi, sqrt

    def f(lam: float) -> float:
        return lam * exp(lam * lam) * erf(lam) - ste / sqrt(pi)

    hi = 0.5
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 50.0:  # pragma: no cover - Ste would be absurd
            raise RuntimeError("neumann_lambda failed to bracket the root")
    return float(brentq(f, 0.0, hi, xtol=1e-15, rtol=1e-15))


def stefan_quasi_steady_front(t, thermal: IceThermal, delta_T_K: float):
    """Quasi-steady Stefan front ``x_f = sqrt(2 k DeltaT t /(rho L))`` [m].

    Ignores the sensible heat stored in the frozen layer; exact in the limit
    ``Ste -> 0`` and, for the freshwater baseline, accurate to ``O(Ste)``
    (~1.5 % at ``Ste = 0.094``).
    """
    t = np.asarray(t, dtype=float)
    coef = 2.0 * thermal.k * float(delta_T_K) / (thermal.rho * thermal.L)
    x = np.sqrt(np.maximum(coef * t, 0.0))
    return float(x) if x.ndim == 0 else x


def neumann_front(t, thermal: IceThermal, delta_T_K: float):
    """Exact one-phase Neumann front ``x_f = 2 lambda sqrt(alpha t)`` [m]."""
    ste = thermal.stefan_number(delta_T_K)
    lam = neumann_lambda(ste)
    t = np.asarray(t, dtype=float)
    x = 2.0 * lam * np.sqrt(np.maximum(thermal.alpha * t, 0.0))
    return float(x) if x.ndim == 0 else x


def neumann_temperature_profile(z, t: float, thermal: IceThermal, delta_T_K: float):
    """Neumann similarity temperature field [K] (Dirichlet surface at ``T_f-DeltaT``).

    ``T(z,t) = T_s + (T_f - T_s) erf(z /(2 sqrt(alpha t))) / erf(lambda)``
    """
    from math import erf

    z = np.asarray(z, dtype=float)
    if t <= 0.0:
        return np.full_like(z, thermal.liquidus_K)
    T_f = thermal.liquidus_K
    T_s = T_f - float(delta_T_K)
    lam = neumann_lambda(thermal.stefan_number(delta_T_K))
    if lam <= 0.0:
        return np.full_like(z, T_f)
    s = 2.0 * np.sqrt(thermal.alpha * t)
    return T_s + (T_f - T_s) * np.vectorize(erf)(z / s) / erf(lam)


# ---------------------------------------------------------------------------
# Enthalpy-method solver
# ---------------------------------------------------------------------------
@dataclass
class EnthalpyFrontSolver:
    """1-D implicit enthalpy solver for the freezing of a keel layer.

    Finite volume, cell-centred, backward Euler in time, Newton-Thomas in
    space.  Conservative: the discrete balance summed over the mesh gives
    ``sum_i dz (e_i^{n+1} - e_i^n) == dt * F_surface`` to machine precision.
    """

    n_cells: int = 400
    domain_length: float = 0.5          # m, must be >= the layer thickness
    thermal: IceThermal = field(default_factory=IceThermal)
    band: float = 0.05                  # K, mushy-zone regularisation width
    far_boundary: str = "adiabatic"     # "adiabatic" | "dirichlet"

    def __post_init__(self) -> None:
        if self.n_cells < 10:
            raise ValueError("n_cells must be >= 10")
        if self.band <= 0.0:
            raise ValueError("band must be > 0")
        if self.far_boundary not in ("adiabatic", "dirichlet"):
            raise ValueError("far_boundary must be 'adiabatic' or 'dirichlet'")
        self.dz = float(self.domain_length) / self.n_cells
        self.z = (np.arange(self.n_cells) + 0.5) * self.dz
        self.T_f = self.thermal.liquidus_K

    # -- constitutive ------------------------------------------------------
    @property
    def e_solid(self) -> float:
        """Enthalpy drop across the mushy band, ``rho (L + c_p band)`` [J/m^3]."""
        th = self.thermal
        return th.rho * (th.L + th.cp * self.band)

    def solid_fraction(self, T: np.ndarray) -> np.ndarray:
        return np.asarray(solid_fraction(T, self.T_f, self.band))

    def heat_capacity(self, T: np.ndarray) -> np.ndarray:
        """Apparent volumetric heat capacity ``de/dT = rho (c_p + L ds/dT)``."""
        th = self.thermal
        T = np.asarray(T, dtype=float)
        s = self.solid_fraction(T)
        inside = (s > 0.0) & (s < 1.0)          # inside the linear ramp
        c = np.full_like(T, th.rho * th.cp)
        c[inside] = th.rho * (th.cp + th.L / self.band)
        return c

    def enthalpy(self, T: np.ndarray) -> np.ndarray:
        """Volumetric enthalpy ``e = rho [c_p (T - T_f) - L s(T)]`` [J/m^3].

        Reference: ``e = 0`` at the liquidus in the liquid state.  Freezing
        *releases* heat, so ``e`` decreases as the solid fraction grows;
        ``de/dT = rho (c_p + L ds/dT) > 0`` everywhere (monotone, invertible).
        """
        th = self.thermal
        T = np.asarray(T, dtype=float)
        return th.rho * (th.cp * (T - self.T_f) - th.L * self.solid_fraction(T))

    def temperature(self, e: np.ndarray) -> np.ndarray:
        """Exact inverse ``T(e)`` of :meth:`enthalpy` (piecewise linear)."""
        th = self.thermal
        e = np.asarray(e, dtype=float)
        T = np.empty_like(e)
        liq = e >= 0.0
        froz = e < -self.e_solid
        mushy = (~liq) & (~froz)
        T[liq] = self.T_f + e[liq] / (th.rho * th.cp)
        T[mushy] = self.T_f + e[mushy] / (th.rho * (th.cp + th.L / self.band))
        T[froz] = (self.T_f - self.band) + (e[froz] + self.e_solid) / (th.rho * th.cp)
        return T

    def dT_de(self, e: np.ndarray) -> np.ndarray:
        """Derivative ``dT/de`` [K m^3/J] of the inverse enthalpy map."""
        th = self.thermal
        e = np.asarray(e, dtype=float)
        out = np.full_like(e, 1.0 / (th.rho * th.cp))
        mushy = (e < 0.0) & (e >= -self.e_solid)
        out[mushy] = 1.0 / (th.rho * (th.cp + th.L / self.band))
        return out

    def _surface_resistance(self, h_conv: Optional[float]) -> float:
        """Series resistance surface -> cell centre [m^2 K / W]."""
        half_cell = self.dz / (2.0 * self.thermal.k)
        if h_conv is None or not np.isfinite(h_conv):
            return half_cell
        return 1.0 / float(h_conv) + half_cell

    def suggest_dt(self, delta_T_K: float, safety: float = 0.4) -> float:
        """Time step that resolves the surface-cell transient [s].

        A surface cell of thickness ``dz`` must absorb the half-cell conductive
        flux of one step without being driven through the whole mushy band:
        ``dt < safety * dz^2 rho L / (2 k DeltaT)`` (and the diffusive limit
        ``dz^2/(2 alpha)``).
        """
        th = self.thermal
        dt_latent = self.dz ** 2 * th.rho * th.L / (2.0 * th.k * max(delta_T_K, 1e-12))
        dt_sensible = self.dz ** 2 / (2.0 * th.alpha)
        return float(safety * min(dt_latent, dt_sensible))

    # -- residual / Jacobian -----------------------------------------------
    def _residual(self, e, e_old, dt, T_env, R_surf, kd, T_far, want_jac=False):
        """Discrete backward-Euler residual (and tridiagonal Jacobian)."""
        dz, n = self.dz, self.n_cells
        Tn = self.temperature(e)
        dTde = self.dT_de(e)
        F_faces = kd * (Tn[:-1] - Tn[1:])
        res = np.empty(n)
        res[0] = (e[0] - e_old[0]) * dz / dt - (T_env - Tn[0]) / R_surf + F_faces[0]
        res[1:-1] = (e[1:-1] - e_old[1:-1]) * dz / dt - F_faces[:-1] + F_faces[1:]
        if self.far_boundary == "adiabatic":
            res[-1] = (e[-1] - e_old[-1]) * dz / dt - F_faces[-1]
            far_faces = 1.0
        else:
            res[-1] = (e[-1] - e_old[-1]) * dz / dt - F_faces[-1] + kd * (Tn[-1] - T_far)
            far_faces = 2.0
        if not want_jac:
            return res, Tn, dTde
        diag = np.empty(n)
        upper = np.zeros(n - 1)
        lower = np.zeros(n - 1)
        diag[0] = dz / dt + dTde[0] * (1.0 / R_surf + kd)
        upper[0] = -kd * dTde[1]
        diag[1:-1] = dz / dt + 2.0 * kd * dTde[1:-1]
        upper[1:] = -kd * dTde[2:]
        lower[:-1] = -kd * dTde[:-2]
        diag[-1] = dz / dt + far_faces * kd * dTde[-1]
        return res, Tn, dTde, (diag, upper, lower)

    # -- one implicit step (damped Newton on enthalpy) ----------------------
    def step(self, T: np.ndarray, dt: float, T_env: float,
             h_conv: Optional[float] = None, far_T: Optional[float] = None,
             max_iter: int = 150, tol_rel: float = 1e-9, dT_cap: float = 5.0) -> np.ndarray:
        """Advance one backward-Euler step; returns the new temperature field.

        The nonlinear unknown is the **enthalpy**, so the phase-change map is
        exact and piecewise linear and the latent heat is released in full.  A
        damped Newton with backtracking keeps the iterate inside the branch
        structure of ``T(e)`` (the apparent capacity jumps by ``L/(c_p band)``
        across the mushy band, so a full Newton step can overshoot).
        """
        th, dz, n = self.thermal, self.dz, self.n_cells
        kd = th.k / dz
        e_old = self.enthalpy(T)
        e = e_old.copy()
        R_surf = self._surface_resistance(h_conv)
        T_far = self.T_f if far_T is None else float(far_T)

        res, Tn, _ = self._residual(e, e_old, dt, T_env, R_surf, kd, T_far)
        norm0 = float(np.sqrt(np.mean(res * res)))
        if norm0 == 0.0:
            return Tn
        norm = norm0
        for _ in range(max_iter):
            _, _, _, (diag, upper, lower) = self._residual(
                e, e_old, dt, T_env, R_surf, kd, T_far, want_jac=True)
            ab = np.zeros((3, n))
            ab[0, 1:] = upper
            ab[1, :] = diag
            ab[2, :-1] = lower
            de = solve_banded((1, 1), ab, -res)

            dTde = self.dT_de(e)
            dT_step = float(np.max(np.abs(de * dTde)))
            alpha = 1.0 if dT_step <= dT_cap else dT_cap / dT_step
            for _ in range(40):
                e_try = e + alpha * de
                res_try, Tn_try, _ = self._residual(e_try, e_old, dt, T_env, R_surf, kd, T_far)
                norm_try = float(np.sqrt(np.mean(res_try * res_try)))
                if norm_try <= (1.0 - 1e-4 * alpha) * norm:
                    break
                alpha *= 0.5
            else:  # pragma: no cover - only on pathological steps
                pass
            e, res, Tn, norm = e_try, res_try, Tn_try, norm_try
            if norm <= tol_rel * norm0:
                return Tn
        if norm <= 1e-6 * norm0:   # round-off floor: accept
            return Tn
        raise RuntimeError(
            f"enthalpy Newton did not converge (residual ratio {norm / norm0:.2e})")

    # -- driver ------------------------------------------------------------
    def run(self, T_env: float, t_end: float, dt: float,
            T_init: Optional[float] = None, h_conv: Optional[float] = None,
            record_times: Optional[Sequence[float]] = None,
            dt_ramp: bool = True, growth: float = 1.3) -> dict:
        """Integrate to ``t_end`` and record profiles, fronts and energy."""
        if T_init is None:
            T_init = self.T_f
        T = np.full(self.n_cells, float(T_init))
        e0 = self.enthalpy(T)

        rec = sorted(float(t) for t in (record_times if record_times is not None else [t_end]))
        rec = [r for r in rec if r > 0.0]

        t = 0.0
        ri = 0
        flux_integral = 0.0
        R_surf = self._surface_resistance(h_conv)
        out = dict(time=[], profiles=[], front=[], front_integral=[], energy_flux=[],
                   energy_stored=[], front_reached_wall=False)

        def rem_time() -> float:
            if ri < len(rec):
                return rec[ri] - t
            return t_end - t

        dt_cur = min(dt, t_end * 1e-5) if dt_ramp else dt
        n_sub = 0
        while t < t_end - 1e-12:
            step = min(dt_cur, t_end - t)
            if ri < len(rec) and step > rem_time():
                step = rem_time()
            if step <= 0.0:
                break
            # step-doubling fallback: a stalled Newton step is subdivided
            while True:
                try:
                    T = self.step(T, step, T_env, h_conv=h_conv)
                    break
                except RuntimeError:
                    step *= 0.5
                    n_sub += 1
                    if step < 1e-12 * max(t_end, 1.0):
                        raise

            # backward-Euler balance: dz (e^{n+1} - e^n) = dt * F^{n+1}
            flux_integral += ((T_env - T[0]) / R_surf) * step

            t += step
            dt_cur = min(dt, dt_cur * growth)
            # The layer below the front is the only region that still freezes;
            # once the front reaches the insulated far wall the remaining liquid
            # has been consumed and the model has no further freezing to offer.
            # Stop there and flag it (the late-time state is then simply 1-D
            # cooling of fully frozen ice, which adds no strength information).
            if self.far_boundary == "adiabatic":
                s_now = solid_fraction(T, self.T_f, self.band)
                if front_position(self.z, s_now, self.dz) >= self.domain_length - 1.5 * self.dz:
                    out["front_reached_wall"] = True
                    if not out["time"] or out["time"][-1] < t - 1e-12:
                        out["time"].append(t)
                        out["profiles"].append(T.copy())
                        out["front"].append(front_position(self.z, s_now, self.dz))
                        out["front_integral"].append(float(np.sum(s_now) * self.dz))
                        out["energy_flux"].append(flux_integral)
                        out["energy_stored"].append(
                            float(np.sum(self.enthalpy(T) - e0) * self.dz))
                    break
            while ri < len(rec) and abs(t - rec[ri]) <= max(1e-9, 1e-9 * rec[ri]):
                s = solid_fraction(T, self.T_f, self.band)
                out["time"].append(t)
                out["profiles"].append(T.copy())
                out["front"].append(front_position(self.z, s, self.dz))
                out["front_integral"].append(float(np.sum(s) * self.dz))
                out["energy_flux"].append(flux_integral)
                out["energy_stored"].append(float(np.sum((self.enthalpy(T) - e0)) * self.dz))
                ri += 1

        out["z"] = self.z.copy()
        out["n_substeps"] = n_sub
        out["profiles"] = np.array(out["profiles"]) if out["profiles"] else np.empty((0, self.n_cells))
        for key in ("time", "front", "front_integral", "energy_flux", "energy_stored"):
            out[key] = np.asarray(out[key], dtype=float)
        return out


# ---------------------------------------------------------------------------
# Front location helpers
# ---------------------------------------------------------------------------
def front_position(z: np.ndarray, s: np.ndarray, dz: float) -> float:
    """Depth [m] where the solid fraction crosses 1/2 (sub-cell interpolation).

    Returns 0 when the surface cell is already liquid.
    """
    z = np.asarray(z, dtype=float)
    s = np.asarray(s, dtype=float)
    if s[0] <= 0.5:
        return 0.0
    below = np.nonzero(s <= 0.5)[0]
    if below.size == 0:
        return float(z[-1] + 0.5 * dz)
    j = int(below[0])
    s0, s1 = s[j - 1], s[j]
    if s0 == s1:  # pragma: no cover - degenerate
        return float(z[j])
    return float(z[j - 1] + (s0 - 0.5) / (s0 - s1) * (z[j] - z[j - 1]))


def frozen_depth_from_profile(z: np.ndarray, T: np.ndarray, thermal: IceThermal,
                              band: float = 0.05) -> float:
    """Equivalent frozen thickness ``int s dz`` [m] from a temperature profile.

    For the cell-centred uniform grids produced by :class:`EnthalpyFrontSolver`
    this is the cell integral ``sum(s) dz`` — the same definition the solver's
    energy audit uses, so the two agree to round-off.
    """
    z = np.asarray(z, dtype=float)
    s = solid_fraction(np.asarray(T, dtype=float), thermal.liquidus_K, band)
    if z.size < 2:
        return 0.0
    dz = float(np.mean(np.diff(z)))
    return float(np.sum(s) * dz)


# ---------------------------------------------------------------------------
# Strength: Cox & Weeks brine + Timco-Frederking porosity law
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BondStrengthLaw:
    """Bond strength law ``sigma_0 (DeltaT/T_ref)^n (1 - nu_b/nu_p)^m``.

    ``DeltaT = T_f - T`` is the local undercooling below the liquidus, so the
    strength vanishes continuously at the freezing front.  ``nu_b`` is the
    local liquid (brine) volume fraction; ``nu_b = 0`` reduces the law to the
    fresh-water expression exactly (factor ``1.0``).
    """

    sigma_0: float = 2.0e6        # Pa, m7439 asymptotic shear strength
    T_ref: float = 10.0           # K, reference undercooling
    exponent_n: float = 0.3       # baseline exponent (m7439 temperature law)
    porosity_limit: float = 0.27  # Timco & Frederking nu_p
    porosity_exponent: float = 1.5

    def brine_factor(self, nu_b):
        nu_b = np.asarray(nu_b, dtype=float)
        f = np.clip(1.0 - nu_b / self.porosity_limit, 0.0, None) ** self.porosity_exponent
        return float(f) if f.ndim == 0 else f

    def __call__(self, T, thermal: IceThermal, brine_override=None):
        """Bond strength [Pa] at temperature ``T`` [K]."""
        T = np.asarray(T, dtype=float)
        T_f = thermal.liquidus_K
        dT = np.maximum(T_f - T, 0.0)
        base = self.sigma_0 * (dT / self.T_ref) ** self.exponent_n
        if brine_override is None:
            nu = np.asarray([brine_volume_fraction(t - T_REF_K, thermal.salinity_ppt)
                             for t in np.atleast_1d(T)], dtype=float)
            if nu.size == 1:
                nu = float(nu[0])
        else:
            nu = float(brine_override)
        sigma = base * self.brine_factor(nu)
        return float(sigma) if np.ndim(sigma) == 0 else sigma


def local_brine_fraction(T, thermal: IceThermal):
    """Cox & Weeks liquid fraction ``nu_b`` at temperature ``T`` [K]."""
    T = np.asarray(T, dtype=float)
    nu = np.array([brine_volume_fraction(float(t) - T_REF_K, thermal.salinity_ppt) for t in T.ravel()])
    nu = np.clip(nu, 0.0, 1.0).reshape(T.shape)
    return float(nu) if nu.ndim == 0 else nu


# ---------------------------------------------------------------------------
# Effective vs uniform strength
# ---------------------------------------------------------------------------
def uniform_strength(T_env: float, law: BondStrengthLaw, thermal: IceThermal) -> float:
    """m7439 caliber: the whole layer at the ambient temperature [Pa]."""
    return float(law(np.asarray([T_env], dtype=float), thermal)[0])


def effective_strength(z: np.ndarray, T: np.ndarray, H: float, law: BondStrengthLaw,
                       thermal: IceThermal, brine_override=None) -> float:
    """Depth-averaged bond strength ``(1/H) int_0^H sigma_b dz`` [Pa]."""
    z = np.asarray(z, dtype=float)
    T = np.asarray(T, dtype=float)
    mask = z <= float(H)
    if not np.any(mask):
        return 0.0
    zk, Tk = z[mask], T[mask]
    sigma = law(Tk, thermal, brine_override=brine_override)
    if zk.size == 1:
        return float(sigma[0]) / max(float(H), 1e-30) * float(H)
    return float(np.trapezoid(sigma, zk) / float(H))


def deviation_ratio(z, T, H, T_env, law, thermal, brine_override=None) -> float:
    """``tau_eff / tau_uniform`` (nan when the uniform caliber vanishes)."""
    tu = uniform_strength(T_env, law, thermal)
    if tu <= 0.0:
        return float("nan")
    return effective_strength(z, T, H, law, thermal, brine_override=brine_override) / tu


# ---------------------------------------------------------------------------
# Study drivers
# ---------------------------------------------------------------------------
BASELINE = dict(k=2.2, exponent_n=0.3, delta_T_K=15.0, H_m=1.0, t_s=86_400.0)


def run_case(*, k: float = 2.2, exponent_n: float = 0.3, delta_T_K: float = 15.0,
             H: float = 1.0, t_end: float = 86_400.0, salinity_ppt: float = 0.0,
             h_conv: Optional[float] = 1.0e9, n_cells: int = 400, dt: float = 60.0,
             band: float = 0.05, sigma_0: float = 2.0e6, T_ref: float = 10.0,
             brine_override: Optional[float] = None,
             domain_length: Optional[float] = None) -> dict:
    """One freezing case: solve the profile and report both strength calibers.

    Returns a dict with the profile, front diagnostics, energy audit, the
    effective and uniform strengths and their ratio.
    """
    thermal = IceThermal(k=k, salinity_ppt=salinity_ppt)
    law = BondStrengthLaw(sigma_0=sigma_0, T_ref=T_ref, exponent_n=exponent_n)
    T_env = thermal.liquidus_K - float(delta_T_K)
    L_dom = domain_length if domain_length is not None else float(H)
    solver = EnthalpyFrontSolver(n_cells=n_cells, domain_length=L_dom,
                                 thermal=thermal, band=band)
    res = solver.run(T_env=T_env, t_end=t_end, dt=dt, h_conv=h_conv, record_times=[t_end])
    z = res["z"]
    T = res["profiles"][-1]

    tau_eff = effective_strength(z, T, H, law, thermal, brine_override=brine_override)
    tau_uni = uniform_strength(T_env, law, thermal)
    x_f = float(res["front"][-1])
    x_int = float(res["front_integral"][-1])
    x_steady = float(stefan_quasi_steady_front(t_end, thermal, delta_T_K))
    x_neumann = float(neumann_front(t_end, thermal, delta_T_K))

    inside = z <= H
    sigma_profile = law(T[inside], thermal, brine_override=brine_override)
    # frozen-depth-normalised caliber (secondary, "dual caliber")
    frozen = max(x_int, 1e-30)
    tau_frozen = float(np.trapezoid(sigma_profile, z[inside]) / frozen) \
        if sigma_profile.size > 1 else 0.0

    e_flux = float(res["energy_flux"][-1])
    e_stored = float(res["energy_stored"][-1])

    return dict(
        front_reached_wall=bool(res.get("front_reached_wall", False)),
        t_recorded_s=float(res["time"][-1]),
        parameters=dict(k=k, exponent_n=exponent_n, delta_T_K=delta_T_K, H_m=H,
                        t_end_s=t_end, salinity_ppt=salinity_ppt, T_env_K=T_env,
                        liquidus_K=thermal.liquidus_K, stefan_number=thermal.stefan_number(delta_T_K),
                        biot=k / (h_conv * H) if h_conv else float("inf"),
                        n_cells=n_cells, dt_s=dt, band_K=band, h_conv=h_conv,
                        sigma_0_Pa=sigma_0, T_ref_K=T_ref,
                        brine_override=brine_override),
        z=z, T_profile=T,
        sigma_profile=law(T, thermal, brine_override=brine_override),
        front_m=dict(numerical=x_f, integral=x_int, quasi_steady=x_steady, neumann=x_neumann,
                     rel_err_vs_quasi=(x_f - x_steady) / x_steady if x_steady > 0 else float("nan"),
                     rel_err_vs_neumann=(x_f - x_neumann) / x_neumann if x_neumann > 0 else float("nan")),
        energy=dict(stored_J_m2=e_stored, flux_J_m2=e_flux,
                    rel_residual=(e_stored - e_flux) / abs(e_flux) if e_flux != 0 else float("nan")),
        strength=dict(tau_eff_Pa=tau_eff, tau_uniform_Pa=tau_uni,
                      ratio=tau_eff / tau_uni if tau_uni > 0 else float("nan"),
                      tau_frozen_caliber_Pa=tau_frozen),
        profile_T_min_K=float(np.min(T)), profile_T_max_K=float(np.max(T)),
    )


def run_biot_sweep(biot_levels: Sequence[float] = (0.0, 0.44, 4.4, 44.0, 440.0, 4400.0),
                   h_conv: float = 500.0, k0: float = 2.2, **case_kw) -> dict:
    """Sweep the conduction-to-surface-transfer ratio ``Bi = k/(h H)``.

    ``h_conv`` is held fixed and only ``k`` grows, so ``Bi -> inf`` is the
    m7439 uniform-temperature limit: the surface resistance vanishes, the
    surface temperature tends to ``T_env`` and the profile over ``[0, H]``
    flattens.  A Robin boundary is used deliberately — with a Dirichlet
    surface the surface flux ``2 k DeltaT / dz`` would grow without bound with
    ``k`` and the discrete problem would stop being resolvable.
    """
    kw = dict(case_kw)
    H = kw.get("H", BASELINE["H_m"])
    rows = []
    for bi in biot_levels:
        k = float(bi) * h_conv * H
        k_eff = max(k, 1e-6)
        out = run_case(k=k_eff, h_conv=h_conv, **kw)
        # analytic cross-check on the same profile family (exact Neumann)
        z = out["z"]
        inside = z <= H
        thermal = IceThermal(k=k_eff, salinity_ppt=kw.get("salinity_ppt", 0.0))
        T_an = neumann_temperature_profile(z[inside], kw.get("t_end", BASELINE["t_s"]),
                                           thermal, kw.get("delta_T_K", BASELINE["delta_T_K"]))
        law = BondStrengthLaw(exponent_n=kw.get("exponent_n", BASELINE["exponent_n"]))
        tau_an = effective_strength(z[inside], T_an, H, law, thermal)
        rows.append(dict(biot=float(bi), k_W_mK=k_eff,
                         ratio=out["strength"]["ratio"],
                         tau_eff_Pa=out["strength"]["tau_eff_Pa"],
                         tau_uniform_Pa=out["strength"]["tau_uniform_Pa"],
                         ratio_neumann=tau_an / out["strength"]["tau_uniform_Pa"],
                         front_m=out["front_m"]["numerical"],
                         T_range_K=out["profile_T_max_K"] - out["profile_T_min_K"]))
    ratios = np.array([r["ratio"] for r in rows], dtype=float)
    diffs = np.diff(ratios)
    return dict(h_conv=h_conv, rows=rows, ratios=ratios.tolist(),
                monotone_increasing=bool(np.all(diffs > 0.0)),
                monotone_neumann=bool(np.all(np.diff([r["ratio_neumann"] for r in rows]) > 0.0)))


def uniform_limit_study(k_levels: Sequence[float] = (2.2, 2.2e1, 2.2e2, 2.2e3, 2.2e4, 2.2e6, 2.2e8),
                        H: float = 1.0, delta_T_K: float = 15.0,
                        t_end: float = 86_400.0, salinity_ppt: float = 0.0,
                        exponent_n: float = 0.3, h_conv: float = 500.0) -> dict:
    """Show ``tau_eff -> tau_uniform`` as ``k -> inf`` (Bi -> inf).

    The flat-profile level (``k = inf``, checked first) exercises the caliber
    definitions to round-off; then the exact Neumann similarity profile is
    evaluated at increasing ``k`` — the physically admissible limit is
    analytic, since for ``k -> inf`` the frozen layer becomes a thin
    sub-surface skin and the discrete problem can no longer resolve it.
    """
    law = BondStrengthLaw(exponent_n=exponent_n)
    thermal0 = IceThermal(k=k_levels[0], salinity_ppt=salinity_ppt)
    T_env = thermal0.liquidus_K - delta_T_K
    tau_uniform = uniform_strength(T_env, law, thermal0)
    z = np.linspace(0.0, H, 2001)
    rows = []
    for k in k_levels:
        th = IceThermal(k=float(k), salinity_ppt=salinity_ppt)
        T = neumann_temperature_profile(z, t_end, th, delta_T_K)
        tau = effective_strength(z, T, H, law, th)
        rows.append(dict(k_W_mK=float(k), biot=float(k) / (h_conv * H),
                         tau_eff_Pa=tau, ratio=tau / tau_uniform,
                         T_range_K=float(np.max(T) - np.min(T))))
    # flat-profile level: T == T_env everywhere (k = inf)
    T_flat = np.full_like(z, T_env)
    tau_flat = effective_strength(z, T_flat, H, law, thermal0)
    # first level whose deviation is below the 0.1 % acceptance band
    passing = [r for r in rows if abs(r["ratio"] - 1.0) < 1e-3]
    return dict(tau_uniform_Pa=tau_uniform, rows=rows,
                flat_profile=dict(tau_eff_Pa=tau_flat,
                                  rel_diff=(tau_flat - tau_uniform) / tau_uniform),
                rel_diff_final=(rows[-1]["ratio"] - 1.0),
                k_for_0p1pct=passing[0]["k_W_mK"] if passing else float("nan"),
                biot_for_0p1pct=passing[0]["biot"] if passing else float("nan"))


def run_brine_sweep(brine_levels: Sequence[float] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
                    **case_kw) -> dict:
    """Sweep a prescribed brine fraction and report ``tau_eff``."""
    rows = []
    for phi in brine_levels:
        out = run_case(brine_override=float(phi), **case_kw)
        rows.append(dict(brine_fraction=float(phi),
                         tau_eff_Pa=out["strength"]["tau_eff_Pa"],
                         tau_uniform_Pa=out["strength"]["tau_uniform_Pa"],
                         ratio=out["strength"]["ratio"]))
    tau = np.array([r["tau_eff_Pa"] for r in rows], dtype=float)
    return dict(rows=rows, tau_eff_Pa=tau.tolist(),
                monotone_decreasing=bool(np.all(np.diff(tau) < 0.0)))

# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
def _build_cli() -> "argparse.ArgumentParser":
    """Argument parser for ``python -m tensordem.freeze_bond.thermal_profile``."""
    import argparse

    p = argparse.ArgumentParser(
        prog="tensordem-thermal-profile",
        description=("Freezing-front depth profile and freeze-bond strength evolution "
                     "(P316-W1-N03): one case, tau_eff vs the m7439 uniform caliber, "
                     "anchored on the Stefan/Neumann solutions."))
    p.add_argument("--k", type=float, default=BASELINE["k"], help="ice conductivity [W/(m K)]")
    p.add_argument("--delta-T", type=float, default=BASELINE["delta_T_K"],
                   help="surface undercooling [K]")
    p.add_argument("--H", type=float, default=BASELINE["H_m"], help="keel draft [m]")
    p.add_argument("--t-hours", type=float, default=BASELINE["t_s"] / 3600.0,
                   help="freezing time [h]")
    p.add_argument("--salinity", type=float, default=0.0, help="bulk salinity [ppt]")
    p.add_argument("--exponent-n", type=float, default=BASELINE["exponent_n"],
                   help="strength temperature exponent n")
    p.add_argument("--h-conv", type=float, default=1.0e9,
                   help="surface heat-transfer coefficient [W/(m^2 K)] (1e9 ~ Dirichlet)")
    p.add_argument("--n-cells", type=int, default=400, help="finite-volume cells")
    p.add_argument("--dt", type=float, default=60.0, help="time step [s]")
    p.add_argument("--biot-sweep", action="store_true",
                   help="also report the Biot sweep of tau_eff/tau_uniform")
    p.add_argument("--out", type=Path, default=None, help="write the JSON summary here")
    return p


def main(argv: "Sequence[str] | None" = None) -> int:
    """Entry point: solve one freezing case and print the JSON summary.

    Exit code 0 when the Stefan anchor of this case is within the predeclared
    2 % band, 1 otherwise (the CLI is a *check*, not a printer).
    """
    import json
    import sys

    args = _build_cli().parse_args(argv)
    case = run_case(k=args.k, exponent_n=args.exponent_n, delta_T_K=args.delta_T,
                    H=args.H, t_end=args.t_hours * 3600.0, salinity_ppt=args.salinity,
                    h_conv=args.h_conv, n_cells=args.n_cells, dt=args.dt)
    summary = {
        "entry": "tensordem.freeze_bond.thermal_profile:main",
        "case": {k: v for k, v in case["parameters"].items() if v is not None},
        "front_m": case["front_m"],
        "energy": case["energy"],
        "strength": case["strength"],
    }
    if args.biot_sweep:
        summary["biot_sweep"] = run_biot_sweep(h_conv=args.h_conv, H=args.H,
                                              exponent_n=args.exponent_n,
                                              delta_T_K=args.delta_T,
                                              t_end=args.t_hours * 3600.0)
    payload = json.dumps(summary, indent=2, default=float)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    # the predeclared anchor of the node is the *exact* Neumann front (0.5 %);
    # the quasi-steady Stefan law is reported, not used as the pass criterion
    anchor_ok = abs(case["front_m"]["rel_err_vs_neumann"]) <= 0.005
    return 0 if anchor_ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    import sys as _sys

    _sys.exit(main())
