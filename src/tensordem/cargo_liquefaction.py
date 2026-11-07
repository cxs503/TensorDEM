"""Bulk cargo liquefaction and cargo-shift stability assessment.

Implements the IN38-N06 campaign: quantifying moisture content *w* and roll
excitation amplitude *a* → excess pore pressure ratio *ru(t)* → liquefaction
threshold *a_thr* → equivalent free surface → GZ correction & stability loss.

Physical model (1-D simplified, no 3-D Biot):
  1. Granular skeleton represented by a DEM-style vertical effective stress
     column; pore pressure evolves via 1-D Terzaghi consolidation
     (σ' = σ − u, ru = u / σ_v0).
  2. Cyclic undrained loading builds excess pore pressure following the
     Seed-et-al. (1975) / Booker-et-al. (1988) parameterisation:
         ru = (2/π) arcsin(N/N_L)^{1/(2θ)}
     where N is cycle count, N_L is the number of cycles to liquefaction at
     the applied CSR, and θ ≈ 0.7 is a material exponent.
  3. Liquefaction criterion: ru ≥ 0.8 → onset of liquefaction (flow state
     begins); ru ≥ 1.0 → full liquefaction (effective stress → 0).
  4. Container acceleration boundary driven by ship roll time history
     φ(t) = φ_0 sin(2π t / T_roll); horizontal acceleration at cargo CoG
     a_h(t) = −g (z_cog / R_roll) φ(t) for small angles, or more generally
     a_h = −ω² x_cog φ(t) where ω = 2π/T_roll.
  5. Equivalent free surface: once ru exceeds threshold, the cargo behaves
     as a fluid. The free surface angle in a rectangular hold of width B
     under roll angle φ is tan(α) = a_h / g_eff, where g_eff = g(1−ru)
     represents the reduced restoring force due to pore pressure. The
     equivalent free-surface moment of inertia is computed using the full
     cargo density (the mass is still present, only the skeleton strength
     is lost).
  6. GZ correction: FSC (free surface correction) = ρ_cargo I_fs / (ρ_sw ∇)
     where ∇ is displacement volume. GZ_corrected = GZ_original − FSC × sin(φ).

References:
  - IMSBC Code (International Maritime Solid Bulk Cargoes), Appendix 2:
    Test procedures for cargoes which may liquefy (TML determination).
  - Seed, H.B., Martin, P.P. & Breen, J. (1975). "Pore pressure buildup
    during cyclic loading." ASCE JGED.
  - Booker, J.R., Rahman, M.S. & Seed, H.B. (1988). "Pore pressure
    generation in cohesive soils."
  - DNV-OS-F101 (2013) for free surface correction methodology.
  - Terzaghi, K. (1943). "Theoretical Soil Mechanics." Wiley.

Scope limits (deliberate):
  - 1-D pore pressure only (no 3-D Biot consolidation).
  - TML test data not available; only threshold trends reported.
  - Rectangular hold geometry assumed for FSC calculation.
  - Small-angle roll assumption for acceleration mapping.
  - No cargo-structure interaction (rigid hold walls).

All quantities SI (m, s, N, Pa, kg).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import torch
import numpy as np

__all__ = [
    "GRAVITY",
    "TerzaghiConsolidation",
    "CyclicPorePressureModel",
    "LiquefactionCriterion",
    "RollExcitation",
    "FreeSurfaceCorrection",
    "CargoLiquefactionConfig",
    "CargoLiquefactionSimulation",
    "run_parametric_sweep",
]

GRAVITY = 9.81
"""Standard gravity, m/s^2."""


# ---------------------------------------------------------------------------
# 1-D Terzaghi consolidation
# ---------------------------------------------------------------------------

@dataclass
class TerzaghiConsolidation:
    """1-D Terzaghi consolidation for vertical effective stress.

    Given an initial total vertical stress σ_v0 and an applied load increment
    Δσ, the excess pore pressure u(t) dissipates according to:
        u(z,t) = Δσ × Σ_{m=0}^{∞} (2/M) sin(M z/H) exp(−M² T_v)
    where M = (2m+1)π/2, T_v = c_v t / H² is the time factor, c_v is the
    coefficient of consolidation, H is the drainage path length, and z is depth.

    For the liquefaction problem we use the *undrained* limit (T_v → 0) where
    all applied cyclic stress goes into pore pressure buildup.
    """
    sigma_v0: float  # initial vertical total stress, Pa
    H: float  # drainage path length (layer thickness for single drainage), m
    c_v: float = 1e-7  # coefficient of consolidation, m²/s (typical fine sand)
    n_terms: int = 50  # number of Fourier terms

    def pore_pressure_at_depth(self, z: float, t: float, delta_sigma: float) -> float:
        """Excess pore pressure u(z, t) under instantaneous load Δσ.

        Parameters
        ----------
        z : depth from drainage surface, m
        t : time, s
        delta_sigma : applied stress increment, Pa

        Returns
        -------
        u : excess pore pressure, Pa

        Notes
        -----
        At t=0, the Fourier series converges to Δσ for z in (0, H) (uniform
        initial excess pore pressure). Convergence is slow near boundaries
        (Gibbs phenomenon), so use n_terms ≥ 50 for accuracy.
        """
        T_v = self.c_v * t / (self.H ** 2)
        u = 0.0
        for m in range(self.n_terms):
            M = (2 * m + 1) * math.pi / 2.0
            term = (2.0 / M) * math.sin(M * z / self.H) * math.exp(-M ** 2 * T_v)
            u += term
        return delta_sigma * u

    def average_degree_of_consolidation(self, t: float) -> float:
        """Average degree of consolidation U_avg(t) = 1 − u_avg / Δσ.

        U_avg = 1 − Σ (2/M²) exp(−M² T_v)
        """
        T_v = self.c_v * t / (self.H ** 2)
        u_ratio = 0.0
        for m in range(self.n_terms):
            M = (2 * m + 1) * math.pi / 2.0
            u_ratio += (2.0 / M ** 2) * math.exp(-M ** 2 * T_v)
        return 1.0 - u_ratio


# ---------------------------------------------------------------------------
# Cyclic pore pressure buildup model
# ---------------------------------------------------------------------------

@dataclass
class CyclicPorePressureModel:
    """Cyclic undrained pore pressure buildup (Seed et al. 1975 parameterisation).

    The pore pressure ratio ru = Δu / σ_v0 evolves with cycle count N as:
        ru = (2/π) arcsin( (N / N_L)^{1/(2θ)} )
    where:
        N_L = number of cycles to liquefaction (ru = 1.0) at the applied CSR
        θ = material exponent (typically 0.5–0.8; 0.7 for medium sand)
        CSR = τ_cyc / σ_v0 = cyclic stress ratio

    N_L is related to CSR via the liquefaction resistance curve:
        CSR = CRR × (N_L / N_ref)^{−k}
    where CRR is the cyclic resistance ratio at N_ref cycles (typically 10 or 15),
    and k ≈ 0.2–0.3 is a slope parameter.

    For the ship-roll problem, each roll cycle applies one stress cycle with
    CSR = a_h / g (horizontal acceleration ratio).
    """
    theta: float = 0.7  # material exponent
    crr_ref: float = 0.15  # cyclic resistance ratio at N_ref
    n_ref: int = 10  # reference number of cycles
    k_slope: float = 0.25  # liquefaction resistance curve slope

    def cycles_to_liquefaction(self, csr: float) -> float:
        """Number of cycles N_L to reach ru=1.0 at given CSR.

        From CSR = CRR × (N_L / N_ref)^{−k}:
            N_L = N_ref × (CRR / CSR)^{1/k}

        Returns float('inf') if CSR ≤ 0 (no liquefaction possible).
        """
        if csr <= 0:
            return float('inf')
        if csr >= self.crr_ref:
            # CSR exceeds CRR at reference cycles → liquefaction in < N_ref cycles
            # Extrapolate: N_L = N_ref × (CRR/CSR)^{1/k}
            return self.n_ref * (self.crr_ref / csr) ** (1.0 / self.k_slope)
        return self.n_ref * (self.crr_ref / csr) ** (1.0 / self.k_slope)

    def pore_pressure_ratio(self, n_cycles: float, csr: float) -> float:
        """Excess pore pressure ratio ru = Δu / σ_v0 after n_cycles.

        ru = (2/π) arcsin( (N / N_L)^{1/(2θ)} )

        Clamped to [0, 1].
        """
        n_l = self.cycles_to_liquefaction(csr)
        if n_l == float('inf') or n_cycles <= 0:
            return 0.0
        ratio = n_cycles / n_l
        if ratio >= 1.0:
            return 1.0
        exponent = 1.0 / (2.0 * self.theta)
        arg = ratio ** exponent
        arg = min(arg, 1.0)  # numerical safety
        ru = (2.0 / math.pi) * math.asin(arg)
        return max(0.0, min(1.0, ru))


# ---------------------------------------------------------------------------
# Liquefaction criterion
# ---------------------------------------------------------------------------

@dataclass
class LiquefactionCriterion:
    """Liquefaction state classification based on pore pressure ratio ru.

    States:
        ru < 0.8       : STABLE (no liquefaction)
        0.8 ≤ ru < 0.95: ONSET (incipient liquefaction, strength loss begins)
        0.95 ≤ ru < 1.0: PARTIAL (significant strength loss, flow state approaching)
        ru ≥ 1.0       : FULL (complete liquefaction, σ' → 0)

    These thresholds follow common geotechnical practice (Ishihara 1993,
    "Liquefaction and Flow Failure During Seismic Events").
    """
    ru_onset: float = 0.80
    ru_partial: float = 0.95
    ru_full: float = 1.00

    def classify(self, ru: float) -> str:
        """Classify liquefaction state."""
        if ru >= self.ru_full:
            return "FULL"
        elif ru >= self.ru_partial:
            return "PARTIAL"
        elif ru >= self.ru_onset:
            return "ONSET"
        else:
            return "STABLE"

    def is_liquefied(self, ru: float) -> bool:
        """True if ru ≥ ru_onset (0.8)."""
        return ru >= self.ru_onset


# ---------------------------------------------------------------------------
# Ship roll excitation
# ---------------------------------------------------------------------------

@dataclass
class RollExcitation:
    """Ship roll time history and resulting horizontal acceleration at cargo CoG.

    Roll motion (small angle, undamped):
        φ(t) = φ_0 sin(ω t)
    where φ_0 is roll amplitude (rad), ω = 2π / T_roll.

    Horizontal acceleration at a point at height z above roll center and
    transverse distance x from centerline:
        a_h(t) = −ω² (x cos(φ) + z sin(φ))  (exact for rigid body)
    For small angles and cargo near centerline (x ≈ 0):
        a_h(t) ≈ −ω² z φ(t) = −ω² z φ_0 sin(ω t)

    The cyclic stress ratio induced in the cargo is:
        CSR = |a_h|_max / g = ω² z φ_0 / g
    """
    phi_0: float  # roll amplitude, rad
    t_roll: float  # roll period, s
    z_cog: float = 2.0  # height of cargo CoG above roll center, m
    x_cog: float = 0.0  # transverse offset of cargo CoG from centerline, m

    @property
    def omega(self) -> float:
        """Angular frequency, rad/s."""
        return 2.0 * math.pi / self.t_roll

    def roll_angle(self, t: float) -> float:
        """Roll angle φ(t), rad."""
        return self.phi_0 * math.sin(self.omega * t)

    def horizontal_acceleration(self, t: float) -> float:
        """Horizontal acceleration a_h(t) at cargo CoG, m/s².

        Exact rigid-body formula for small roll:
            a_h = −ω² (x cos(φ) + z sin(φ))
        """
        phi = self.roll_angle(t)
        return -self.omega ** 2 * (self.x_cog * math.cos(phi) + self.z_cog * math.sin(phi))

    def max_horizontal_acceleration(self) -> float:
        """Maximum |a_h| over one cycle, m/s²."""
        # For small angles: max |a_h| ≈ ω² z φ_0
        # For exact: need to find max of |a_h(t)| numerically
        # Use small-angle approximation for simplicity
        return self.omega ** 2 * self.z_cog * self.phi_0

    def csr_at_cargo(self) -> float:
        """Cyclic stress ratio at cargo location: CSR = max|a_h| / g."""
        return self.max_horizontal_acceleration() / GRAVITY


# ---------------------------------------------------------------------------
# Free surface correction
# ---------------------------------------------------------------------------

@dataclass
class FreeSurfaceCorrection:
    """Equivalent free surface correction for GZ curve.

    When cargo liquefies (ru ≥ threshold), it behaves as a fluid. The free
    surface in a rectangular hold of width B and length L under roll angle φ has:
        tan(α) = a_h / g_eff
    where g_eff = g (1 − ru) is the reduced effective gravity (representing
    the reduced restoring force due to pore pressure).

    The free surface moment of inertia (transverse) for a rectangular tank:
        I_fs = (L B³ / 12) × (ρ_cargo / ρ_sw)

    Note: I_fs uses the full cargo density (not reduced by ru), because the
    cargo mass is still present even when liquefied - only the skeleton
    strength is lost. The ru parameter affects the free surface *angle*
    (via g_eff) but not the inertia.

    The FSC (free surface correction to GZ) is:
        FSC = ρ_cargo × I_fs / (ρ_sw × ∇)
    where ∇ is ship displacement volume.

    The corrected GZ is:
        GZ_corr(φ) = GZ_orig(φ) − FSC × sin(φ)
    """
    hold_length: float  # hold length (fore-aft), m
    hold_width: float  # hold width (transverse), m
    cargo_density: float  # bulk cargo density, kg/m³
    cargo_height: float  # cargo height in hold, m
    rho_sw: float = 1025.0  # seawater density, kg/m³
    displacement: float = 50000.0  # ship displacement, tonnes

    @property
    def displacement_volume(self) -> float:
        """Ship displacement volume ∇, m³."""
        return self.displacement * 1000.0 / self.rho_sw

    def free_surface_inertia(self, ru: float) -> float:
        """Free surface moment of inertia I_fs at given ru, m⁴.

        For a rectangular hold:
            I_fs = (L × B³ / 12) × (ρ_cargo / ρ_sw)

        I_fs is only meaningful when cargo is liquefied (ru ≥ 0.8).
        Uses full cargo density (mass is conserved, only strength is lost).
        """
        if ru < 0.8:
            return 0.0  # solid cargo, no free surface effect
        return (self.hold_length * self.hold_width ** 3 / 12.0) * (self.cargo_density / self.rho_sw)

    def fsc(self, ru: float) -> float:
        """Free surface correction to GZ, m.

        FSC = ρ_cargo × I_fs / (ρ_sw × ∇)
        """
        if ru < 0.8:
            return 0.0
        i_fs = self.free_surface_inertia(ru)
        return self.cargo_density * i_fs / (self.rho_sw * self.displacement_volume)

    def free_surface_angle(self, ru: float, a_h: float) -> float:
        """Free surface angle α in the hold, rad.

        tan(α) = a_h / g_eff
        where g_eff = g × (1 − ru) for liquefied cargo.
        """
        if ru < 0.8:
            return 0.0
        g_eff = GRAVITY * (1.0 - ru)
        if g_eff <= 0:
            return math.pi / 2.0  # fully liquefied, surface is horizontal
        return math.atan(abs(a_h) / g_eff)


# ---------------------------------------------------------------------------
# Top-level simulation
# ---------------------------------------------------------------------------

@dataclass
class CargoLiquefactionConfig:
    """Configuration for cargo liquefaction simulation."""
    # Cargo properties
    cargo_density: float = 1600.0  # bulk density, kg/m³ (iron ore typical)
    moisture_content: float = 0.08  # moisture content w (mass fraction)
    sigma_v0: float = 50000.0  # initial vertical stress at cargo base, Pa
    layer_thickness: float = 3.0  # cargo layer thickness, m
    c_v: float = 1e-7  # coefficient of consolidation, m²/s

    # Liquefaction resistance
    crr_ref: float = 0.15  # cyclic resistance ratio at N_ref
    n_ref: int = 10  # reference cycles
    theta: float = 0.7  # pore pressure model exponent
    k_slope: float = 0.25  # liquefaction resistance curve slope

    # Ship/hold geometry
    roll_amplitude_deg: float = 15.0  # roll amplitude, degrees
    roll_period: float = 12.0  # roll period, s
    z_cog: float = 2.0  # cargo CoG height above roll center, m
    hold_length: float = 20.0  # hold length, m
    hold_width: float = 15.0  # hold width, m
    cargo_height: float = 3.0  # cargo height in hold, m
    displacement: float = 50000.0  # ship displacement, tonnes

    # Simulation
    n_cycles: int = 100  # number of roll cycles to simulate
    dt: float = 0.1  # time step, s

    @property
    def roll_amplitude_rad(self) -> float:
        return math.radians(self.roll_amplitude_deg)


@dataclass
class CargoLiquefactionSimulation:
    """Top-level simulation orchestrating cargo liquefaction under roll excitation.

    Workflow:
        1. Set up roll excitation from ship parameters.
        2. Compute CSR at cargo location.
        3. Evolve pore pressure ratio ru(t) over N roll cycles.
        4. Detect liquefaction threshold crossing (ru ≥ 0.8).
        5. Compute equivalent free surface and GZ correction.
        6. Output time histories and stability metrics.
    """
    config: CargoLiquefactionConfig

    def run(self) -> dict:
        """Run simulation and return results dict.

        Returns
        -------
        results : dict with keys:
            'time': array of time values, s
            'roll_angle': array of roll angles, rad
            'horizontal_accel': array of horizontal accelerations, m/s²
            'ru': array of pore pressure ratios
            'liquefaction_state': array of state strings
            'fsc': array of free surface corrections, m
            'free_surface_angle': array of free surface angles, rad
            't_onset': time of first liquefaction onset (ru≥0.8), s or None
            't_full': time of full liquefaction (ru≥1.0), s or None
            'ru_max': maximum pore pressure ratio
            'n_cycles_to_onset': number of cycles to onset
            'csr': cyclic stress ratio at cargo
            'a_h_max': maximum horizontal acceleration, m/s²
        """
        cfg = self.config

        # Set up components
        roll = RollExcitation(
            phi_0=cfg.roll_amplitude_rad,
            t_roll=cfg.roll_period,
            z_cog=cfg.z_cog,
        )
        pore_model = CyclicPorePressureModel(
            theta=cfg.theta,
            crr_ref=cfg.crr_ref,
            n_ref=cfg.n_ref,
            k_slope=cfg.k_slope,
        )
        criterion = LiquefactionCriterion()
        fsc_calc = FreeSurfaceCorrection(
            hold_length=cfg.hold_length,
            hold_width=cfg.hold_width,
            cargo_density=cfg.cargo_density,
            cargo_height=cfg.cargo_height,
            displacement=cfg.displacement,
        )

        # Compute CSR
        csr = roll.csr_at_cargo()
        a_h_max = roll.max_horizontal_acceleration()

        # Time integration
        total_time = cfg.n_cycles * cfg.roll_period
        n_steps = int(total_time / cfg.dt)
        time_arr = np.linspace(0, total_time, n_steps)
        roll_arr = np.zeros(n_steps)
        accel_arr = np.zeros(n_steps)
        ru_arr = np.zeros(n_steps)
        state_arr = []
        fsc_arr = np.zeros(n_steps)
        fs_angle_arr = np.zeros(n_steps)

        t_onset = None
        t_full = None
        n_cycles_to_onset = None

        for i, t in enumerate(time_arr):
            # Roll kinematics
            phi = roll.roll_angle(t)
            a_h = roll.horizontal_acceleration(t)
            roll_arr[i] = phi
            accel_arr[i] = a_h

            # Cycle count (fractional)
            n_cyc = t / cfg.roll_period

            # Pore pressure buildup
            ru = pore_model.pore_pressure_ratio(n_cyc, csr)
            ru_arr[i] = ru

            # Liquefaction state
            state = criterion.classify(ru)
            state_arr.append(state)

            # Free surface correction
            fsc_val = fsc_calc.fsc(ru)
            fsc_arr[i] = fsc_val

            # Free surface angle
            fs_angle = fsc_calc.free_surface_angle(ru, abs(a_h))
            fs_angle_arr[i] = fs_angle

            # Track threshold crossings
            if t_onset is None and criterion.is_liquefied(ru):
                t_onset = t
                n_cycles_to_onset = n_cyc
            if t_full is None and ru >= 1.0:
                t_full = t

        results = {
            'time': time_arr.tolist(),
            'roll_angle': roll_arr.tolist(),
            'horizontal_accel': accel_arr.tolist(),
            'ru': ru_arr.tolist(),
            'liquefaction_state': state_arr,
            'fsc': fsc_arr.tolist(),
            'free_surface_angle': fs_angle_arr.tolist(),
            't_onset': t_onset,
            't_full': t_full,
            'ru_max': float(ru_arr.max()),
            'n_cycles_to_onset': n_cycles_to_onset,
            'csr': csr,
            'a_h_max': a_h_max,
            'config': {
                'cargo_density': cfg.cargo_density,
                'moisture_content': cfg.moisture_content,
                'roll_amplitude_deg': cfg.roll_amplitude_deg,
                'roll_period': cfg.roll_period,
                'crr_ref': cfg.crr_ref,
                'displacement': cfg.displacement,
            }
        }
        return results


# ---------------------------------------------------------------------------
# Parametric sweep
# ---------------------------------------------------------------------------

def run_parametric_sweep(
    roll_amplitudes_deg: list[float],
    moisture_contents: list[float],
    base_config: Optional[CargoLiquefactionConfig] = None,
) -> list[dict]:
    """Run parametric sweep over roll amplitude and moisture content.

    For each (roll_amplitude, moisture) pair, run simulation and record:
        - ru_max after 100 cycles
        - t_onset (time to liquefaction onset)
        - n_cycles_to_onset
        - fsc_max (maximum free surface correction)

    Note: moisture content affects CRR (higher w → lower CRR). This is
    parameterised as:
        CRR(w) = CRR_ref × (1 − 2.5 × (w − w_opt))
    where w_opt = 0.05 is the optimum moisture content (Proctor). For w > w_opt,
    CRR decreases linearly.

    Parameters
    ----------
    roll_amplitudes_deg : list of roll amplitudes to sweep, degrees
    moisture_contents : list of moisture contents to sweep, mass fraction
    base_config : base configuration (if None, use defaults)

    Returns
    -------
    sweep_results : list of dicts, one per (amplitude, moisture) pair
    """
    if base_config is None:
        base_config = CargoLiquefactionConfig()

    w_opt = 0.05  # optimum moisture content
    results = []

    for phi_deg in roll_amplitudes_deg:
        for w in moisture_contents:
            cfg = CargoLiquefactionConfig(
                cargo_density=base_config.cargo_density,
                moisture_content=w,
                sigma_v0=base_config.sigma_v0,
                layer_thickness=base_config.layer_thickness,
                c_v=base_config.c_v,
                crr_ref=base_config.crr_ref * max(0.1, 1.0 - 2.5 * (w - w_opt)),
                n_ref=base_config.n_ref,
                theta=base_config.theta,
                k_slope=base_config.k_slope,
                roll_amplitude_deg=phi_deg,
                roll_period=base_config.roll_period,
                z_cog=base_config.z_cog,
                hold_length=base_config.hold_length,
                hold_width=base_config.hold_width,
                cargo_height=base_config.cargo_height,
                displacement=base_config.displacement,
                n_cycles=base_config.n_cycles,
                dt=base_config.dt,
            )
            sim = CargoLiquefactionSimulation(config=cfg)
            res = sim.run()

            results.append({
                'roll_amplitude_deg': phi_deg,
                'moisture_content': w,
                'crr_effective': cfg.crr_ref,
                'csr': res['csr'],
                'ru_max': res['ru_max'],
                't_onset': res['t_onset'],
                'n_cycles_to_onset': res['n_cycles_to_onset'],
                'fsc_max': max(res['fsc']),
                'liquefied': res['ru_max'] >= 0.8,
            })

    return results
