"""Moist-particle bed drying in a rotary drum: transient cell cascade + ledgers.

Model
-----
Co-current rotary drum discretised into ``n_cells`` well-mixed axial cells.

* **Particle phase** (dry-basis moisture X, lumped temperature T_p, dry mass
  m_s per cell).  Particles are advected cell-to-cell at the plug-flow velocity
  v_p = L/tau_res; between cells they exchange heat and water with the gas and
  release latent heat.
* **Gas phase** is quasi-steady (the gas residence time L/u_g is two orders of
  magnitude shorter than the drying time), solved by axial marching at every
  time step with a 3-sweep fixed-point iteration per cell:

      Y_out = Y_in + E/mdot_a
      h_out = h_in - Q/mdot_a + E h_v(T_p)/mdot_a

* **Closures**: Ranz-Marshall Nu and Sh (see :mod:`moist_transfer`);
  evaporation mdot = h_m A [a_w rho_v,sat(T_p) - rho_v,g], sensible exchange
  Q = h A (T_g - T_p).

Conservation
------------
The particle energy equation carries exactly the enthalpy terms that the gas
balance uses and advection is written in flux form, so the water and energy
ledgers close to round-off.  Both residuals are reported, and a negative
control (latent sink switched off) is provided so the ledger test demonstrably
has teeth.

Limitations
-----------
* Particles are lumped capacitance (Bi = h d/(2 k_p) ~ 0.2-0.4 here).
* Fixed particle diameter (no shrinkage / attrition / breakage).
* No mechanical DEM contact network: this is the *thermal-mass* closure of the
  rotary dryer.  The mechanical drum flow lives in :mod:`tensordem.drum_flow`;
  the dry thermo-mechanical coupling of plan 380 n02 is reproduced as the
  w -> 0, Y_g -> 0 degenerate limit (:func:`simulate_dry_heating`).
* Wall contact/radiation heat is available as an optional source term but is
  off (adiabatic wall) at the direct-heat operating point used for validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .kinematics import classify_regime, froude_number
from .moist_transfer import (
    evaporation_rate,
    specific_surface_area,
    transfer_coefficients,
    wet_bulb_from_closures,
)
from .psychrometrics import (
    CP_L,
    P_STD,
    R_DA,
    T_REF,
    moist_air_density,
    moist_enthalpy,
    temperature_from_enthalpy,
    vapour_enthalpy,
    vapour_pressure_from_humidity_ratio,
)

__all__ = [
    "DryerBedConfig",
    "simulate_moist_drying",
    "simulate_dry_heating",
    "solve_steady_profile",
    "bed_ledger",
]


@dataclass
class DryerBedConfig:
    """Operating point of a co-current direct-heat rotary drum dryer."""

    # geometry / solids
    drum_diameter: float = 1.0
    drum_length: float = 5.0
    particle_diameter: float = 0.004
    particle_density: float = 1250.0
    particle_cp: float = 1500.0
    solid_feed_dry: float = 0.12  # kg/s dry solids
    residence_time: float = 600.0  # s
    n_cells: int = 24
    # gas side
    gas_inlet_temperature: float = 400.0
    gas_inlet_humidity: float = 0.010
    gas_inlet_velocity: float = 2.5  # m/s superficial, at drum inlet
    pressure: float = P_STD
    # moisture state
    initial_moisture: float = 0.35  # dry basis
    equilibrium_moisture: float = 0.02
    critical_moisture: float = 0.15
    sorption_kappa: float = 6.0
    particle_initial_temperature: float = 293.15
    # wall (None => adiabatic wall / direct-heat dryer)
    wall_temperature: Optional[float] = None
    wall_emissivity: float = 0.9
    # numerics / physics switches
    dt: float = 0.1
    t_end: Optional[float] = None  # default = residence_time
    driving_force_form: str = "vapour_density"
    include_latent_sink: bool = True
    omega: float = 0.5  # drum rotation, kinematics regime reporting only

    def __post_init__(self) -> None:
        if self.n_cells < 1:
            raise ValueError("n_cells must be >= 1")
        if self.t_end is None:
            self.t_end = self.residence_time
        if not (0.0 <= self.equilibrium_moisture <= self.critical_moisture):
            raise ValueError("need 0 <= X_e <= X_c")
        if self.particle_diameter <= 0 or self.drum_length <= 0:
            raise ValueError("geometry must be positive")

    # derived ---------------------------------------------------------------
    @property
    def cross_section(self) -> float:
        return math.pi * self.drum_diameter ** 2 / 4.0

    @property
    def particle_velocity(self) -> float:
        return self.drum_length / self.residence_time

    @property
    def dry_air_mass_flow(self) -> float:
        p_v = vapour_pressure_from_humidity_ratio(self.gas_inlet_humidity, self.pressure)
        rho_da = (self.pressure - p_v) / (R_DA * self.gas_inlet_temperature)
        return self.gas_inlet_velocity * self.cross_section * rho_da

    @property
    def dry_solid_per_cell(self) -> float:
        return self.solid_feed_dry * self.residence_time / self.n_cells

    @property
    def surface_area_per_cell(self) -> float:
        return self.dry_solid_per_cell * specific_surface_area(
            self.particle_diameter, self.particle_density
        )

    def as_dict(self) -> Dict[str, float]:
        d = dict(self.__dict__)
        d.update(
            {
                "cross_section_m2": self.cross_section,
                "particle_velocity_m_s": self.particle_velocity,
                "dry_air_mass_flow_kg_s": self.dry_air_mass_flow,
                "dry_solid_per_cell_kg": self.dry_solid_per_cell,
                "surface_area_per_cell_m2": self.surface_area_per_cell,
            }
        )
        return d


def _wall_source(cfg: DryerBedConfig, T_p: float) -> float:
    """Radiative wall-to-particle heat for one cell (W); 0 for an adiabatic wall."""
    if cfg.wall_temperature is None:
        return 0.0
    sigma = 5.670374419e-8
    return (
        cfg.wall_emissivity
        * sigma
        * (cfg.wall_temperature ** 4 - T_p ** 4)
        * cfg.surface_area_per_cell
    )


def simulate_moist_drying(cfg: DryerBedConfig) -> Dict[str, object]:
    """Transient co-current cell-cascade simulation of moist-particle drying.

    Returns time histories, final axial profiles, local closure diagnostics,
    ledger terms and the two conservation residuals.
    """
    nc = cfg.n_cells
    dz = cfg.drum_length / nc
    m_s = cfg.dry_solid_per_cell
    area = cfg.surface_area_per_cell
    mdot_a = cfg.dry_air_mass_flow
    A_cs = cfg.cross_section
    v_p = cfg.particle_velocity
    X_e, X_c, kap = cfg.equilibrium_moisture, cfg.critical_moisture, cfg.sorption_kappa
    d_p, rho_p, cp_s = cfg.particle_diameter, cfg.particle_density, cfg.particle_cp
    X_in = cfg.initial_moisture
    T_p_in = cfg.particle_initial_temperature
    p = cfg.pressure
    dt = cfg.dt
    n_steps = int(round(cfg.t_end / dt)) + 1
    t = np.arange(n_steps) * dt
    hist_stride = max(1, n_steps // 400)

    X = np.full(nc, X_in)
    T_p = np.full(nc, T_p_in)
    M_w = m_s * X
    # enthalpy-based energy state: H = (m_s cp_s + M_w cp_l)(T_p - T_REF)
    H_p = (m_s * cp_s + M_w * CP_L) * (T_p - T_REF)

    X_hist: List[np.ndarray] = []
    Tp_hist: List[np.ndarray] = []
    Tg_hist: List[np.ndarray] = []
    Yg_hist: List[np.ndarray] = []

    W_storage_0 = float(np.sum(M_w))
    E_storage_0 = float(np.sum(H_p))
    W_gas_in = W_gas_out = W_solid_in = W_solid_out = 0.0
    E_gas_in_cum = E_gas_out_cum = 0.0
    E_solid_in_cum = E_solid_out_cum = 0.0
    Q_wall_cum = 0.0
    E_evap_cum = 0.0

    h_gas_in = moist_enthalpy(cfg.gas_inlet_temperature, cfg.gas_inlet_humidity)
    T_g_out = cfg.gas_inlet_temperature

    for step in range(n_steps):
        # ---- quasi-steady gas march ---------------------------------------
        T_g_in = cfg.gas_inlet_temperature
        Y_g_in = cfg.gas_inlet_humidity
        h_g_in = h_gas_in
        E_cell = np.empty(nc)
        Q_cell = np.empty(nc)
        Tg_in_cell = np.empty(nc)
        Yg_in_cell = np.empty(nc)
        Tg_out_cell = np.empty(nc)
        Yg_out_cell = np.empty(nc)
        aux: List[Dict[str, float]] = []
        for j in range(nc):
            T_go, Y_go = T_g_in, Y_g_in
            for _ in range(3):
                T_avg = 0.5 * (T_g_in + T_go)
                Y_avg = 0.5 * (Y_g_in + Y_go)
                rho_g = moist_air_density(T_avg, Y_avg, p)
                u_rel = mdot_a / (rho_g * A_cs) - v_p
                mdot, tr = evaporation_rate(
                    T_p[j], X[j], T_avg, Y_avg, u_rel, d_p, area, X_e, X_c, kap,
                    p, None, cfg.driving_force_form,
                )
                m_max = max(0.0, (X[j] - X_e) * m_s / dt)
                mdot = min(mdot, m_max)
                q_conv = tr["h"] * area * (T_avg - T_p[j])
                Y_new = Y_g_in + mdot / mdot_a
                h_out = (h_g_in - q_conv / mdot_a
                         + mdot * vapour_enthalpy(T_p[j]) / mdot_a)
                T_new = temperature_from_enthalpy(h_out, Y_new)
                converged = abs(T_new - T_go) < 1e-10 and abs(Y_new - Y_go) < 1e-15
                T_go, Y_go = T_new, Y_new
                if converged:
                    break
            E_cell[j] = mdot
            Q_cell[j] = q_conv
            Tg_in_cell[j] = T_g_in
            Yg_in_cell[j] = Y_g_in
            Tg_out_cell[j] = T_go
            Yg_out_cell[j] = Y_go
            aux.append(tr)
            T_g_in, Y_g_in, h_g_in = T_go, Y_go, moist_enthalpy(T_go, Y_go)

        T_g_out = float(Tg_out_cell[-1])
        Y_g_out = float(Yg_out_cell[-1])
        h_g_out = moist_enthalpy(T_g_out, Y_g_out)
        mdot_out = mdot_a * Y_g_out

        # ---- particle update (flux-form advection + exchange) --------------
        # outlet state of this step is taken *before* the update so that the
        # boundary flux of the ledger telescopes exactly with the interior
        # advection terms.
        X_out_step = float(X[-1])
        T_p_out_step = float(T_p[-1])
        X_up = np.concatenate(([X_in], X[:-1]))
        T_p_up = np.concatenate(([T_p_in], T_p[:-1]))
        dMdt = cfg.solid_feed_dry * (X_up - X) - E_cell
        dHdt = cfg.solid_feed_dry * (
            (cp_s + X_up * CP_L) * (T_p_up - T_REF)
            - (cp_s + X * CP_L) * (T_p - T_REF)
        )
        q_wall = np.array([_wall_source(cfg, Tpi) for Tpi in T_p])
        h_v = np.array([vapour_enthalpy(Tpi) for Tpi in T_p])
        if not cfg.include_latent_sink:
            h_v = np.zeros_like(h_v)
        dHdt = dHdt + Q_cell + q_wall - E_cell * h_v

        M_w = M_w + dMdt * dt
        X = M_w / m_s
        H_p = H_p + dHdt * dt
        T_p = T_REF + H_p / (m_s * cp_s + M_w * CP_L)

        # ---- ledgers -------------------------------------------------------
        W_gas_in += mdot_a * cfg.gas_inlet_humidity * dt
        W_gas_out += mdot_out * dt
        W_solid_in += cfg.solid_feed_dry * X_in * dt
        W_solid_out += cfg.solid_feed_dry * X_out_step * dt
        E_gas_in_cum += mdot_a * h_gas_in * dt
        E_gas_out_cum += mdot_a * h_g_out * dt
        E_solid_in_cum += cfg.solid_feed_dry * (cp_s + X_in * CP_L) * (T_p_in - T_REF) * dt
        E_solid_out_cum += (
            cfg.solid_feed_dry * (cp_s + X_out_step * CP_L) * (T_p_out_step - T_REF) * dt
        )
        Q_wall_cum += float(np.sum(q_wall)) * dt
        E_evap_cum += float(np.sum(E_cell)) * dt

        if step % hist_stride == 0 or step == n_steps - 1:
            X_hist.append(X.copy())
            Tp_hist.append(T_p.copy())
            Tg_hist.append(0.5 * (Tg_in_cell + Tg_out_cell))
            Yg_hist.append(0.5 * (Yg_in_cell + Yg_out_cell))

    # ---- final axial diagnostics ------------------------------------------
    T_wb_local = np.array(
        [
            wet_bulb_from_closures(
                0.5 * (Tg_in_cell[j] + Tg_out_cell[j]),
                0.5 * (Yg_in_cell[j] + Yg_out_cell[j]),
                max(cfg.gas_inlet_velocity, 1e-6), d_p, p, None, cfg.driving_force_form,
            )
            for j in range(nc)
        ]
    )
    Re = np.array([a["Re"] for a in aux])
    Sh = np.array([a["Sh"] for a in aux])
    Nu = np.array([a["Nu"] for a in aux])
    h_arr = np.array([a["h"] for a in aux])
    hm_arr = np.array([a["h_m"] for a in aux])
    aw = np.array([a["a_w"] for a in aux])
    z = (np.arange(nc) + 0.5) * dz

    W_storage_1 = float(np.sum(M_w))
    W_total_in = W_storage_0 + W_gas_in + W_solid_in
    W_resid_abs = W_total_in - W_gas_out - W_solid_out - W_storage_1
    W_resid_rel = W_resid_abs / W_total_in if W_total_in > 0 else W_resid_abs

    E_storage_1 = float(np.sum(H_p))
    E_total_in = E_storage_0 + E_gas_in_cum + E_solid_in_cum + Q_wall_cum
    E_resid_abs = E_total_in - E_gas_out_cum - E_solid_out_cum - E_storage_1
    E_resid_rel = E_resid_abs / abs(E_total_in) if E_total_in != 0 else E_resid_abs

    Fr = froude_number(cfg.omega, cfg.drum_diameter / 2.0)
    regime = classify_regime(Fr)
    regime = getattr(regime, "value", str(regime))

    return {
        "config": cfg.as_dict(),
        "t": t,
        "hist_stride": hist_stride,
        "X_hist": np.array(X_hist),
        "Tp_hist": np.array(Tp_hist),
        "Tg_hist": np.array(Tg_hist),
        "Yg_hist": np.array(Yg_hist),
        "final": {
            "z": z,
            "X": X.copy(),
            "T_p": T_p.copy(),
            "T_wb_local": T_wb_local,
            "T_g_in_cell": Tg_in_cell,
            "T_g_out_cell": Tg_out_cell,
            "Y_g_in_cell": Yg_in_cell,
            "Y_g_out_cell": Yg_out_cell,
            "mdot_cell": E_cell,
            "Q_cell": Q_cell,
            "Re": Re,
            "Sh": Sh,
            "Nu": Nu,
            "h": h_arr,
            "h_m": hm_arr,
            "a_w": aw,
            "X_out": float(X[-1]),
            "T_p_out": float(T_p[-1]),
            "T_g_out": T_g_out,
            "Y_g_out": Y_g_out,
        },
        "ledger": {
            "water_storage_initial_kg": W_storage_0,
            "water_storage_final_kg": W_storage_1,
            "water_in_gas_kg": W_gas_in,
            "water_out_gas_kg": W_gas_out,
            "water_in_solid_kg": W_solid_in,
            "water_out_solid_kg": W_solid_out,
            "water_evaporated_kg": E_evap_cum,
            "water_residual_abs_kg": W_resid_abs,
            "water_residual_rel": W_resid_rel,
            "energy_storage_initial_J": E_storage_0,
            "energy_storage_final_J": E_storage_1,
            "energy_in_gas_J": E_gas_in_cum,
            "energy_out_gas_J": E_gas_out_cum,
            "energy_in_solid_J": E_solid_in_cum,
            "energy_out_solid_J": E_solid_out_cum,
            "energy_wall_source_J": Q_wall_cum,
            "energy_residual_abs_J": E_resid_abs,
            "energy_residual_rel": E_resid_rel,
        },
        "kinematics": {"froude": Fr, "regime": regime, "omega": cfg.omega},
    }


def simulate_dry_heating(cfg: DryerBedConfig) -> Dict[str, object]:
    """Independent dry-particle (sensible-heat-only) drum heating reference.

    Degenerate negative control: for X0 = 0 and Y_g = 0 the moist model must
    reproduce this run to within 1e-9 (plan-380-n02 dry thermo-mechanical
    limit).  Implemented separately -- no mass transfer terms at all -- so the
    comparison is not a tautology of the moist code path.
    """
    nc = cfg.n_cells
    m_s = cfg.dry_solid_per_cell
    area = cfg.surface_area_per_cell
    mdot_a = cfg.dry_air_mass_flow
    A_cs = cfg.cross_section
    v_p = cfg.particle_velocity
    d_p, cp_s = cfg.particle_diameter, cfg.particle_cp
    p = cfg.pressure
    dt = cfg.dt
    n_steps = int(round(cfg.t_end / dt)) + 1
    T_p = np.full(nc, cfg.particle_initial_temperature)
    T_p_in = cfg.particle_initial_temperature
    h_g_in0 = moist_enthalpy(cfg.gas_inlet_temperature, cfg.gas_inlet_humidity)
    T_g_out = cfg.gas_inlet_temperature

    for _ in range(n_steps):
        T_g_in = cfg.gas_inlet_temperature
        Y_g = cfg.gas_inlet_humidity
        h_g_in = h_g_in0
        Q_cell = np.empty(nc)
        for j in range(nc):
            T_go = T_g_in
            for _ in range(3):
                T_avg = 0.5 * (T_g_in + T_go)
                rho_g = moist_air_density(T_avg, Y_g, p)
                u_rel = mdot_a / (rho_g * A_cs) - v_p
                tr = transfer_coefficients(T_avg, T_p[j], Y_g, u_rel, d_p, p)
                q_conv = tr["h"] * area * (T_avg - T_p[j])
                T_new = temperature_from_enthalpy(h_g_in - q_conv / mdot_a, Y_g)
                converged = abs(T_new - T_go) < 1e-10
                T_go = T_new
                if converged:
                    break
            Q_cell[j] = q_conv
            T_g_in = T_go
            h_g_in = moist_enthalpy(T_go, Y_g)
        T_g_out = float(T_go)
        T_p_up = np.concatenate(([T_p_in], T_p[:-1]))
        dQ = Q_cell + np.array([_wall_source(cfg, Tpi) for Tpi in T_p])
        dH = cfg.solid_feed_dry * cp_s * (T_p_up - T_p) + dQ
        T_p = T_p + dH * dt / (m_s * cp_s)
    return {"T_p": T_p, "T_g_out": T_g_out}


def solve_steady_profile(cfg: DryerBedConfig, nz: int = 400) -> Dict[str, object]:
    """Steady co-current two-phase plug-flow profile by RK4 marching along z.

    Independent of the transient cell cascade in :func:`simulate_moist_drying`
    (different discretisation, no time stepping), so agreement between the two
    at steady state is a genuine consistency check.

    Equations (per unit drum length, area per unit length
    a' = (mdot_s tau_res/L) * 6/(rho_p d), i.e. holdup per length times the
    specific surface area):

        dX/dz    = -E'/mdot_s
        dY_g/dz  =  E'/mdot_a
        dh_g/dz  = (-Q' + E' h_v(T_p))/mdot_a
        dT_p/dz  =  [Q' - E' h_v(T_p)] / [mdot_s (c_ps + X c_pl)] * v_p/v_p

    with E' = a' h_m [a_w rho_v,sat(T_p) - rho_v,g] and Q' = a' h (T_g - T_p).
    """
    dz = cfg.drum_length / nz
    m_s_dot = cfg.solid_feed_dry
    mdot_a = cfg.dry_air_mass_flow
    A_cs = cfg.cross_section
    v_p = cfg.particle_velocity
    d_p, rho_p, cp_s = cfg.particle_diameter, cfg.particle_density, cfg.particle_cp
    X_e, X_c, kap = cfg.equilibrium_moisture, cfg.critical_moisture, cfg.sorption_kappa
    p, cp_l = cfg.pressure, CP_L
    # particle surface area per unit drum length = (holdup per length) * a_s
    # holdup per length = m_s_dot * tau_res / L
    a_per_length = (m_s_dot * cfg.residence_time / cfg.drum_length) * specific_surface_area(
        d_p, rho_p
    )

    def rhs(z, y):
        X, T_p, Y_g, h_g = y
        T_g = temperature_from_enthalpy(h_g, Y_g)
        rho_g = moist_air_density(T_g, Y_g, p)
        u_rel = mdot_a / (rho_g * A_cs) - v_p
        mdot, tr = evaporation_rate(
            T_p, X, T_g, Y_g, u_rel, d_p, 1.0, X_e, X_c, kap, p, None,
            cfg.driving_force_form,
        )
        E_p = mdot * a_per_length           # kg/(s m)
        Q_p = tr["h"] * a_per_length * (T_g - T_p)   # W/m
        hv = vapour_enthalpy(T_p)
        dX = -E_p / m_s_dot
        dTp = (Q_p - E_p * hv) / (m_s_dot * (cp_s + X * cp_l))
        dY = E_p / mdot_a
        dh = (-Q_p + E_p * hv) / mdot_a
        return np.array([dX, dTp, dY, dh])

    z = np.arange(nz + 1) * dz
    y = np.array([cfg.initial_moisture, cfg.particle_initial_temperature,
                  cfg.gas_inlet_humidity,
                  moist_enthalpy(cfg.gas_inlet_temperature, cfg.gas_inlet_humidity)])
    Y = np.empty((nz + 1, 4))
    Y[0] = y
    for i in range(nz):
        k1 = rhs(z[i], y)
        k2 = rhs(z[i] + dz / 2, y + dz * k1 / 2)
        k3 = rhs(z[i] + dz / 2, y + dz * k2 / 2)
        k4 = rhs(z[i] + dz, y + dz * k3)
        y = y + dz * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        y[0] = max(y[0], X_e)
        y[1] = max(y[1], 200.0)
        Y[i + 1] = y
    X, T_p, Y_g, h_g = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]
    T_g = np.array([temperature_from_enthalpy(h, yy) for h, yy in zip(h_g, Y_g)])

    # local diagnostics
    Re = np.empty(nz + 1)
    Sh = np.empty(nz + 1)
    Nu = np.empty(nz + 1)
    h_arr = np.empty(nz + 1)
    hm_arr = np.empty(nz + 1)
    aw = np.empty(nz + 1)
    E_local = np.empty(nz + 1)
    Twb = np.empty(nz + 1)
    for i in range(nz + 1):
        rho_g = moist_air_density(T_g[i], Y_g[i], p)
        u_rel = mdot_a / (rho_g * A_cs) - v_p
        mdot, tr = evaporation_rate(
            T_p[i], X[i], T_g[i], Y_g[i], u_rel, d_p, 1.0, X_e, X_c, kap, p, None,
            cfg.driving_force_form,
        )
        Re[i], Sh[i], Nu[i] = tr["Re"], tr["Sh"], tr["Nu"]
        h_arr[i], hm_arr[i] = tr["h"], tr["h_m"]
        aw[i] = tr["a_w"]
        E_local[i] = mdot * a_per_length
        Twb[i] = wet_bulb_from_closures(
            T_g[i], Y_g[i], max(u_rel, 1e-6), d_p, p, None, cfg.driving_force_form
        )
    return {
        "z": z,
        "X": X,
        "T_p": T_p,
        "T_g": T_g,
        "Y_g": Y_g,
        "T_wb": Twb,
        "Re": Re,
        "Sh": Sh,
        "Nu": Nu,
        "h": h_arr,
        "h_m": hm_arr,
        "a_w": aw,
        "mdot_per_length": E_local,
        "X_out": float(X[-1]),
        "T_g_out": float(T_g[-1]),
        "Y_g_out": float(Y_g[-1]),
        "T_p_out": float(T_p[-1]),
    }


def bed_ledger(result: Dict[str, object]) -> Dict[str, float]:
    """Convenience accessor returning just the conservation residuals."""
    led = result["ledger"]
    return {
        "water_residual_rel": float(led["water_residual_rel"]),
        "energy_residual_rel": float(led["energy_residual_rel"]),
        "water_residual_abs_kg": float(led["water_residual_abs_kg"]),
        "energy_residual_abs_J": float(led["energy_residual_abs_J"]),
    }