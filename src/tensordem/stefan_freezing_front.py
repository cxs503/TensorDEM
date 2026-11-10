"""Stefan freezing-front problem with enthalpy method for ice ridge keel freeze-bond.

This module quantifies the bias introduced by the m7439 assumption of uniform
temperature / pure-water-ice in ice ridge keel freeze-bond strength calculations.

Physics
-------
One-dimensional freezing of a semi-infinite domain (the keel) initially at the
melting temperature T_m.  The surface z=0 is suddenly exposed to a cold
environment at T_env < T_m (via a convective boundary condition with heat
transfer coefficient h, or a prescribed surface temperature for the Dirichlet
limit).

The freezing front x_f(t) advances into the domain.  Behind the front the
temperature is non-uniform:

    T(z,t) = T_env + (T_m - T_env) * erf(z / (2 sqrt(alpha t))) / erf(lambda)

where lambda is the Stefan number root.

The effective freeze-bond shear strength is the depth-averaged bond strength
over the frozen layer of thickness H (the keel draft or the frozen depth,
whichever is smaller):

    tau_eff(t) = (1/H) integral_0^H sigma_b(T(z,t)) dz

compared to the m7439 uniform-temperature estimate:

    tau_uniform(t) = sigma_b(T_uniform)

where T_uniform is typically taken as T_env or some average.

Dimensionless groups
--------------------
- Stefan number:  Ste = c_p * (T_m - T_env) / L
- Biot number:    Bi  = h * L_ref / k
- The ratio tau_eff / tau_uniform is a function of (Ste, Bi).

References
----------
- Carslaw & Jaeger, "Conduction of Heat in Solids", 2nd ed., 1959.
- Timco & Frederking, "Ice mechanics", Cold Regions Sci. Tech., 1990.
- m7439: uniform temperature assumption in ice_ridge_freeze_bond_evolution.py.

All quantities SI (m, kg, s, K, N, Pa).

.. warning::
   **Known defect (recorded by P316-W1-N03).**  The apparent heat capacity
   below integrates the latent heat to ``L/2`` only (cosine window amplitude
   ``L/(2b)`` over ``[-b, b]`` gives ``int c_app dT = c_p 2b + L/2``) and the
   freezing front is located on a cell mask without sub-cell interpolation.
   On the anchor case (DeltaT = 15 K, t = 1 h, n = 400) this yields
   ``x_f = 0.0625 m`` against the analytical ``sqrt(2 k DeltaT t/(rho L)) =
   0.02773 m``, i.e. **+125 %**.  The tests in
   ``tests/test_stefan_freezing_front.py`` only assert ``x_f > 0.01 m`` /
   energy positivity, so they do not catch it.

   For production use the verified replacement
   :mod:`tensordem.freeze_bond.thermal_profile` (full latent heat, enthalpy
   Newton, sub-cell front tracking, energy closure ~1e-12, front anchored on
   the Stefan/Neumann solutions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Sequence

import numpy as np
from scipy.special import erf
from scipy.optimize import brentq
from scipy.linalg import solve_banded

__all__ = [
    "IceThermalProperties",
    "StefanFreezingConfig",
    "EnthalpyStefanSolver",
    "FreezeBondStrengthModel",
    "EffectiveStrengthCalculator",
    "solve_stefan_lambda",
    "analytical_freezing_front",
    "analytical_temperature_profile",
    "run_deviation_sweep",
    "run_baseline_study",
]


# ---------------------------------------------------------------------------
# Thermal properties
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IceThermalProperties:
    """Thermal properties of ice (pure or saline)."""
    k_ice: float = 2.18          # W/(m·K)
    cp_ice: float = 2100.0       # J/(kg·K)
    rho_ice: float = 917.0       # kg/m³
    L_fusion: float = 334000.0   # J/kg
    T_melt: float = 273.15       # K
    brine_fraction: float = 0.0

    @property
    def alpha(self) -> float:
        return self.k_ice / (self.rho_ice * self.cp_ice)

    @property
    def effective_latent_heat(self) -> float:
        return self.L_fusion * (1.0 - self.brine_fraction)


@dataclass(frozen=True)
class StefanFreezingConfig:
    """Configuration for the Stefan freezing problem."""
    T_env: float = 253.15        # K (-20°C)
    h_conv: float = 50.0         # W/(m²·K)
    L_ref: float = 2.0           # m
    H_keel: float = 2.0          # m
    t_end: float = 86400.0       # s (1 day)
    props: IceThermalProperties = field(default_factory=IceThermalProperties)

    @property
    def delta_T(self) -> float:
        return self.props.T_melt - self.T_env

    @property
    def stefan_number(self) -> float:
        return self.props.cp_ice * self.delta_T / self.props.effective_latent_heat

    @property
    def biot_number(self) -> float:
        return self.h_conv * self.L_ref / self.props.k_ice


# ---------------------------------------------------------------------------
# Analytical Stefan solution (Neumann problem, one-phase, Dirichlet BC)
# ---------------------------------------------------------------------------

def solve_stefan_lambda(ste: float, tol: float = 1e-12) -> float:
    """Solve λ·exp(λ²)·erf(λ) = Ste/√π for the one-phase Stefan problem."""
    if ste <= 0:
        return 0.0
    target = ste / math.sqrt(math.pi)

    def f(lam):
        return lam * math.exp(lam * lam) * math.erf(lam) - target

    lam_hi = math.sqrt(ste) + 2.0
    while f(lam_hi) < 0:
        lam_hi *= 2.0
    return brentq(f, 1e-14, lam_hi, xtol=tol, maxiter=100)


def analytical_freezing_front(t, ste: float, alpha: float):
    """x_f(t) = 2λ√(αt)."""
    lam = solve_stefan_lambda(ste)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    xf = 2.0 * lam * np.sqrt(alpha * np.maximum(t_arr, 0.0))
    return float(xf[0]) if np.isscalar(t) or getattr(t, 'ndim', 1) == 0 else xf


def analytical_temperature_profile(z, t: float, T_env: float, T_melt: float,
                                   alpha: float, ste: float) -> np.ndarray:
    """T(z,t) in frozen region; T = T_melt beyond front."""
    lam = solve_stefan_lambda(ste)
    z = np.asarray(z, dtype=float)
    if t <= 0 or lam <= 0:
        return np.full_like(z, T_melt)
    xf = 2.0 * lam * math.sqrt(alpha * t)
    eta = z / (2.0 * math.sqrt(alpha * t))
    T_frozen = T_env + (T_melt - T_env) * erf(eta) / erf(lam)
    return np.where(z <= xf, np.clip(T_frozen, T_env, T_melt), T_melt)


# ---------------------------------------------------------------------------
# Enthalpy method Stefan solver (vectorised, fast)
# ---------------------------------------------------------------------------

@dataclass
class EnthalpyStefanSolver:
    """1D enthalpy-method solver for Stefan freezing (apparent heat capacity).

    Implicit Euler with Newton iteration on T.  The mushy-zone regularisation
    uses a cosine bump of half-width ``mushy_half_width`` around T_melt.
    """
    n_cells: int = 200
    domain_length: float = 5.0
    props: IceThermalProperties = field(default_factory=IceThermalProperties)
    mushy_half_width: float = 0.05   # K

    def __post_init__(self):
        self.dz = self.domain_length / self.n_cells
        self.z = np.linspace(0.5 * self.dz, self.domain_length - 0.5 * self.dz, self.n_cells)

    # -- apparent heat capacity (vectorised) --------------------------------
    def effective_cp(self, T: np.ndarray) -> np.ndarray:
        Tm, band = self.props.T_melt, self.mushy_half_width
        L_eff = self.props.effective_latent_heat
        cp_base = self.props.cp_ice
        d = np.abs(T - Tm)
        cp_eff = np.full_like(T, cp_base)
        mask = d < band
        cp_eff[mask] = cp_base + L_eff / (2.0 * band) * 0.5 * (1.0 + np.cos(np.pi * (T[mask] - Tm) / band))
        return cp_eff

    # -- single implicit step (vectorised Thomas algorithm) -----------------
    def step(self, T: np.ndarray, dt: float, T_env: float, h_conv: float) -> np.ndarray:
        rho, k, dz, n = self.props.rho_ice, self.props.k_ice, self.dz, self.n_cells
        T_new = T.copy()
        kd = k / dz                       # conductance per unit area

        for _ in range(25):
            cp_eff = self.effective_cp(T_new)
            C = rho * cp_eff * dz / dt    # diagonal capacity

            # Build diagonals for scipy banded solver (upper, diag, lower)
            diag = np.empty(n)
            upper = np.empty(n - 1)
            lower = np.empty(n - 1)

            # interior
            diag[1:-1] = 2.0 * kd + C[1:-1]
            upper[1:] = -kd
            lower[:-1] = -kd

            # BC: convective at z=0
            diag[0] = kd + h_conv + C[0]
            upper[0] = -kd

            # BC: insulated at z=L
            diag[-1] = kd + C[-1]
            lower[-1] = -kd

            rhs = C * T
            rhs[0] += h_conv * T_env

            # Solve tridiagonal via scipy banded (1,1)
            ab = np.zeros((3, n))
            ab[0, 1:] = upper       # upper diagonal
            ab[1, :] = diag          # main diagonal
            ab[2, :-1] = lower       # lower diagonal
            T_trial = solve_banded((1, 1), ab, rhs)

            if np.max(np.abs(T_trial - T_new)) < 1e-10:
                T_new = T_trial
                break
            T_new = T_trial

        return T_new

    # -- full run -----------------------------------------------------------
    def run(self, T_env: float, h_conv: float, T_init: float,
            t_end: float, dt: float,
            record_times: Optional[Sequence[float]] = None) -> dict:
        T = np.full(self.n_cells, T_init)
        if record_times is None:
            record_times = [t_end]
        record_times = sorted(record_times)

        times_out, profs, fronts, energies = [], [], [], []
        t, ri = 0.0, 0

        while t < t_end - 1e-14:
            step = min(dt, t_end - t)
            if ri < len(record_times) and t + step > record_times[ri] + 1e-14:
                step = record_times[ri] - t
            T = self.step(T, step, T_env, h_conv)
            t += step

            if ri < len(record_times) and abs(t - record_times[ri]) < dt * 0.5:
                Tm = self.props.T_melt
                below = T < Tm - 0.01
                xf = float(self.z[np.max(np.where(below)[0])] + 0.5 * self.dz) if np.any(below) else 0.0
                times_out.append(t)
                profs.append(T.copy())
                fronts.append(xf)
                energies.append(float(np.sum(self.props.rho_ice * self.props.cp_ice * (T_init - T) * self.dz)))
                ri += 1

        return dict(z=self.z.copy(), time=np.array(times_out),
                    profiles=np.array(profs) if profs else np.empty((0, self.n_cells)),
                    freezing_front=np.array(fronts),
                    total_energy=np.array(energies))


# ---------------------------------------------------------------------------
# Freeze-bond strength model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FreezeBondStrengthModel:
    """σ_b(T) = σ₀ · ((T_melt − T) / T_ref)^n  for T < T_melt, else 0."""
    sigma_0: float = 1.5e6
    T_ref_undercooling: float = 10.0
    exponent: float = 0.8
    T_melt: float = 273.15

    def strength(self, T):
        T = np.asarray(T, dtype=float)
        uc = np.maximum(self.T_melt - T, 0.0)
        s = np.where(uc > 0, self.sigma_0 * (uc / self.T_ref_undercooling) ** self.exponent, 0.0)
        return float(s) if s.ndim == 0 else s


# ---------------------------------------------------------------------------
# Effective strength calculator
# ---------------------------------------------------------------------------

@dataclass
class EffectiveStrengthCalculator:
    H_keel: float = 2.0
    strength_model: FreezeBondStrengthModel = field(default_factory=FreezeBondStrengthModel)

    def effective_strength_from_profile(self, z: np.ndarray, T: np.ndarray) -> float:
        z, T = np.asarray(z), np.asarray(T)
        mask = z <= self.H_keel
        if not np.any(mask):
            return 0.0
        z_k, T_k = z[mask], T[mask]
        sigma = self.strength_model.strength(T_k)
        if len(z_k) < 2:
            return float(sigma[0]) if len(sigma) else 0.0
        return float(np.trapezoid(sigma, z_k) / self.H_keel)

    def uniform_strength(self, T_env: float) -> float:
        return float(self.strength_model.strength(T_env))

    def deviation_ratio(self, z, T, T_env: float) -> float:
        tu = self.uniform_strength(T_env)
        if tu <= 0:
            return float('nan')
        return self.effective_strength_from_profile(z, T) / tu


# ---------------------------------------------------------------------------
# High-level study functions
# ---------------------------------------------------------------------------

def run_baseline_study(config: Optional[StefanFreezingConfig] = None,
                       n_cells: int = 200) -> dict:
    if config is None:
        config = StefanFreezingConfig()
    props = config.props
    ste = config.stefan_number
    alpha = props.alpha

    t_an = np.linspace(100, config.t_end, 100)
    xf_an = analytical_freezing_front(t_an, ste, alpha)

    rec = [3600, 7200, 14400, 28800, 57600, 86400]
    rec = [t for t in rec if t <= config.t_end]
    z_fine = np.linspace(0, config.L_ref, 500)
    profs_a = [analytical_temperature_profile(z_fine, t, config.T_env, props.T_melt, alpha, ste) for t in rec]

    solver = EnthalpyStefanSolver(n_cells=n_cells, domain_length=config.L_ref, props=props)
    res_e = solver.run(T_env=config.T_env, h_conv=config.h_conv, T_init=props.T_melt,
                       t_end=config.t_end, dt=60.0, record_times=rec)

    calc = EffectiveStrengthCalculator(H_keel=config.H_keel)
    tau_uni = calc.uniform_strength(config.T_env)
    tau_a = [calc.effective_strength_from_profile(z_fine, p) for p in profs_a]
    tau_e = [calc.effective_strength_from_profile(res_e["z"], res_e["profiles"][i])
             for i in range(len(res_e["time"]))]

    return {
        "config": dict(T_env_K=config.T_env, T_melt_K=props.T_melt, delta_T_K=config.delta_T,
                       stefan_number=ste, biot_number=config.biot_number,
                       h_conv=config.h_conv, H_keel_m=config.H_keel,
                       t_end_s=config.t_end, brine_fraction=props.brine_fraction),
        "analytical": dict(time_s=t_an.tolist(), freezing_front_m=xf_an.tolist(),
                           lam=solve_stefan_lambda(ste)),
        "record_times_s": rec,
        "profiles_analytical": dict(z_m=z_fine.tolist(), profiles_K=[p.tolist() for p in profs_a]),
        "enthalpy": dict(z_m=res_e["z"].tolist(), time_s=res_e["time"].tolist(),
                         freezing_front_m=res_e["freezing_front"].tolist(),
                         profiles_K=[p.tolist() for p in res_e["profiles"]],
                         total_energy_J=res_e["total_energy"].tolist()),
        "strength": dict(tau_uniform_Pa=tau_uni, tau_eff_analytical_Pa=tau_a,
                         tau_eff_enthalpy_Pa=tau_e,
                         ratio_analytical=[t / tau_uni if tau_uni > 0 else float('nan') for t in tau_a],
                         ratio_enthalpy=[t / tau_uni if tau_uni > 0 else float('nan') for t in tau_e]),
    }


def run_deviation_sweep(biot_numbers=None, stefan_numbers=None,
                        H_keel: float = 2.0, t_end: float = 86400.0,
                        n_cells: int = 150) -> dict:
    if biot_numbers is None:
        biot_numbers = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
    if stefan_numbers is None:
        stefan_numbers = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    props = IceThermalProperties()
    calc = EffectiveStrengthCalculator(H_keel=H_keel)
    mat = np.zeros((len(biot_numbers), len(stefan_numbers)))

    for i, Bi in enumerate(biot_numbers):
        h_conv = Bi * props.k_ice / H_keel
        for j, Ste in enumerate(stefan_numbers):
            dT = Ste * props.effective_latent_heat / props.cp_ice
            T_env = props.T_melt - dT
            solver = EnthalpyStefanSolver(n_cells=n_cells, domain_length=H_keel * 1.5, props=props)
            try:
                res = solver.run(T_env=T_env, h_conv=h_conv, T_init=props.T_melt,
                                 t_end=t_end, dt=min(60.0, t_end / 50), record_times=[t_end])
                if len(res["time"]) > 0:
                    te = calc.effective_strength_from_profile(res["z"], res["profiles"][0])
                    tu = calc.uniform_strength(T_env)
                    mat[i, j] = te / tu if tu > 0 else float('nan')
            except Exception:
                mat[i, j] = float('nan')

    return dict(biot_numbers=list(biot_numbers), stefan_numbers=list(stefan_numbers),
                deviation_ratio_matrix=mat.tolist(), H_keel_m=H_keel, t_end_s=t_end)
