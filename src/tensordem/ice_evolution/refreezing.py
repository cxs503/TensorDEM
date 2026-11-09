"""Lumped refreezing of supercooled pore water in broken ice (IN155-N09).

Physical model
--------------
A rubble layer of thickness ``h`` is initially at the ambient temperature
``T_a``.  Its pores are filled with supercooled water (volume fraction
``phi_0``).  The water refreezes with a first-order kinetic law

    dw/dt = -k_f(T) w,      k_f(T) = k_ref exp[-E_a/R (1/T - 1/T_a)]

and the released latent heat is conducted to the ambient.  The layer is thin
and ice conducts well, so the field is well mixed (the 1-D boundary layer is
thin compared with ``h``); the conduction loss is therefore represented by a
volumetric conductance ``h_c = 2 k / h**2`` (two faces, linear profile):

    rho c dT/dt = -h_c (T - T_a) + rho_w L_f (-dw/dt)

The porosity is ``phi(t) = w(t)`` (every pore that loses water becomes solid
ice).

Energy budget (criterion e)
---------------------------
Integrating the energy balance over the run gives, exactly,

    E_latent + E_sinter = E_conduction + E_ambient + dE_sensible

with ``E_ambient = 0`` (a Dirichlet ambient does no work).  The closure is
``|LHS - RHS| / (E_latent + E_sinter)`` and must be < 10 %.  Because the
scheme integrates the same balance it checks, the closure is at the level of
the integrator error.

Units are SI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GAS_CONSTANT = 8.314462618
T_MELT = 273.15                 # K
RHO_ICE = 917.0                 # kg/m^3
RHO_WATER = 1000.0              # kg/m^3
C_ICE = 2100.0                  # J/(kg K)
C_WATER = 4180.0                # J/(kg K)
K_ICE = 2.2                     # W/(m K)
K_WATER = 0.56                  # W/(m K)
L_FUSION = 333.7e3              # J/kg


@dataclass(frozen=True)
class RefreezeConfig:
    """Configuration of the lumped refreezing problem."""
    thickness_m: float = 0.05          # layer thickness
    temperature_k: float = 263.15      # ambient = initial field temperature
    phi_0: float = 0.30                # initial pore-water fraction
    k_ref: float = 1.4e-5             # 1/s, freezing rate at T_a
    ea_freeze: float = 60.0e3          # J/mol, freezing activation energy
    k_cond: float = 1.5               # W/(m K), effective conductivity
    dt_s: float = 60.0

    def __post_init__(self) -> None:
        if self.thickness_m <= 0.0:
            raise ValueError("thickness_m must be positive")
        if not 0.0 < self.phi_0 < 1.0:
            raise ValueError("phi_0 must be in (0, 1)")
        if self.temperature_k >= T_MELT:
            raise ValueError("temperature_k must be below the melting point")
        if self.k_ref <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("k_ref and dt_s must be positive")

    @property
    def h_c(self) -> float:
        """Volumetric conductance to the ambient (W/m^3 K)."""
        return 2.0 * self.k_cond / self.thickness_m**2

    @property
    def rho_c(self) -> float:
        """Volumetric heat capacity (J/m^3 K), ice-dominated."""
        return (1.0 - self.phi_0) * RHO_ICE * C_ICE + self.phi_0 * RHO_WATER * C_WATER


@dataclass
class RefreezeResult:
    t_s: np.ndarray
    porosity: np.ndarray              # w(t)
    temperature: np.ndarray           # T(t)
    e_latent_j_m2: float = 0.0        # per unit area
    e_conduction_j_m2: float = 0.0    # per unit area
    e_sensible_j_m2: float = 0.0      # final sensible energy vs T_a
    e_sinter_j_m3: float = 0.0        # per unit volume, injected by caller
    n_steps: int = 0

    def porosity_series(self) -> tuple[np.ndarray, np.ndarray]:
        return self.t_s, self.porosity


def run_refreezing(cfg: RefreezeConfig, *,
                    duration_s: float,
                    record_every_s: float = 300.0,
                    e_sinter_j_m3: float = 0.0) -> RefreezeResult:
    """Integrate the lumped refreezing problem for ``duration_s`` seconds.

    ``e_sinter_j_m3`` is the sintering surface energy released per unit layer
    volume over the same interval (from :func:`sinter_bond`); it enters the
    energy budget only (the sintering heat is negligible against the latent
    heat and does not feed back into the thermal field).
    """
    t_a = cfg.temperature_k
    w = cfg.phi_0
    t = t_a
    dt = cfg.dt_s
    n_steps = int(math.ceil(duration_s / dt))
    n_rec = int(math.ceil(duration_s / record_every_s)) + 1
    t_out = np.linspace(0.0, duration_s, n_rec)
    phi_out = np.empty(n_rec)
    t_out_arr = np.empty(n_rec)
    phi_out[0] = w
    t_out_arr[0] = 0.0

    e_conduction = 0.0
    rec_idx = 0
    for step in range(n_steps):
        # freezing kinetics
        if t < T_MELT:
            kf = cfg.k_ref * math.exp(-cfg.ea_freeze / GAS_CONSTANT
                                  * (1.0 / max(t, 1.0) - 1.0 / t_a))
        else:
            kf = 0.0
        dw = -kf * w * dt
        w_new = min(max(w + dw, 0.0), cfg.phi_0)
        # energy balance
        latent_rate = RHO_WATER * L_FUSION * (w - w_new) / dt   # W/m^3
        cond_rate = cfg.h_c * (t - t_a)                          # W/m^3
        t_new = t + dt * (-cond_rate + latent_rate) / cfg.rho_c
        t_new = min(max(t_new, t_a - 50.0), T_MELT)
        e_conduction += cond_rate * cfg.thickness_m * dt
        w, t = w_new, t_new
        if rec_idx + 1 < n_rec and (step + 1) * dt >= t_out[rec_idx + 1] - 1e-9:
            rec_idx += 1
            phi_out[rec_idx] = w
            t_out_arr[rec_idx] = (step + 1) * dt
        if rec_idx + 1 >= n_rec:
            break
    phi_out[-1] = w
    t_out_arr[-1] = duration_s
    e_latent = RHO_WATER * L_FUSION * cfg.thickness_m * (cfg.phi_0 - w)
    e_sensible = cfg.rho_c * (t - t_a) * cfg.thickness_m
    return RefreezeResult(
        t_s=t_out_arr,
        porosity=phi_out,
        temperature=np.array([t]),
        e_latent_j_m2=e_latent,
        e_conduction_j_m2=e_conduction,
        e_sensible_j_m2=e_sensible,
        e_sinter_j_m3=e_sinter_j_m3,
        n_steps=step + 1,
    )
