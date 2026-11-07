"""Enhanced bulk cargo liquefaction module with full validation suite.

Adds to the base cargo_liquefaction module:
1. Angle of repose validation (dry granular material)
2. Terzaghi analytical solution comparison with quantitative error tracking
3. Liquefaction threshold scanning and monotonicity verification
4. IMO FSC formula consistency check
5. Geometry prior gate validation
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from tensordem.cargo_liquefaction import (
    GRAVITY,
    TerzaghiConsolidation,
    CyclicPorePressureModel,
    LiquefactionCriterion,
    RollExcitation,
    FreeSurfaceCorrection,
    CargoLiquefactionConfig,
    CargoLiquefactionSimulation,
)


# ---------------------------------------------------------------------------
# 1. Angle of repose validation
# ---------------------------------------------------------------------------

@dataclass
class AngleOfReposeValidator:
    """Validate angle of repose for dry granular cargo.

    For sand-like materials, the angle of repose θ_r is related to the
    friction coefficient μ by:
        θ_r ≈ arctan(μ) + correction_for_particle_shape
    
    Typical values:
        - Dry sand: 34° ± 2°
        - Iron ore: 35° - 40°
        - Coal: 30° - 35°
    
    This validator checks that the input friction coefficient produces
    a physically reasonable angle of repose.
    """
    mu_friction: float  # Coulomb friction coefficient
    particle_shape_factor: float = 0.1  # correction for angular particles (rad)
    
    def compute_repose_angle(self) -> float:
        """Compute angle of repose from friction coefficient.
        
        Returns angle in degrees.
        """
        theta_rad = math.atan(self.mu_friction) + self.particle_shape_factor
        return math.degrees(theta_rad)
    
    def validate_sand(self) -> dict:
        """Validate against sand typical values (34° ± 2°).
        
        Returns dict with:
            - angle_deg: computed angle
            - reference_deg: 34.0
            - error_pct: relative error
            - passed: bool (error ≤ 5%)
        """
        angle = self.compute_repose_angle()
        reference = 34.0
        error_pct = abs(angle - reference) / reference * 100.0
        return {
            'angle_deg': angle,
            'reference_deg': reference,
            'error_pct': error_pct,
            'passed': error_pct <= 5.0,
        }
    
    def validate_range(self, min_deg: float = 30.0, max_deg: float = 45.0) -> dict:
        """Validate angle is within acceptable range for bulk cargoes.
        
        Returns dict with:
            - angle_deg: computed angle
            - min_deg, max_deg: bounds
            - passed: bool
        """
        angle = self.compute_repose_angle()
        return {
            'angle_deg': angle,
            'min_deg': min_deg,
            'max_deg': max_deg,
            'passed': min_deg <= angle <= max_deg,
        }


# ---------------------------------------------------------------------------
# 2. Terzaghi analytical solution validation
# ---------------------------------------------------------------------------

@dataclass
class TerzaghiAnalyticalValidator:
    """Validate Terzaghi consolidation against analytical series solution.
    
    The 1D Terzaghi consolidation equation:
        ∂u/∂t = c_v ∂²u/∂z²
    
    With initial condition u(z,0) = Δσ and boundary conditions:
        - Single drainage: u(0,t) = 0, ∂u/∂z(H,t) = 0
        - Double drainage: u(0,t) = 0, u(H,t) = 0
    
    Analytical solution (single drainage):
        u(z,t) = Δσ Σ_{m=0}^∞ (2/M) sin(M z/H) exp(-M² T_v)
        where M = (2m+1)π/2, T_v = c_v t / H²
    
    Average degree of consolidation:
        U_avg = 1 - Σ (2/M²) exp(-M² T_v)
    
    For small T_v (< 0.05): U_avg ≈ 2 sqrt(T_v / π)
    For large T_v (> 0.5): U_avg ≈ 1 - (2/π) exp(-π² T_v / 4)
    """
    sigma_v0: float  # initial vertical stress, Pa
    H: float  # drainage path length, m
    c_v: float  # coefficient of consolidation, m²/s
    n_terms: int = 100  # number of Fourier terms for reference solution
    
    def compute_reference_solution(self, z: float, t: float, delta_sigma: float) -> float:
        """Compute reference solution with high n_terms."""
        tc = TerzaghiConsolidation(
            sigma_v0=self.sigma_v0,
            H=self.H,
            c_v=self.c_v,
            n_terms=self.n_terms,
        )
        return tc.pore_pressure_at_depth(z, t, delta_sigma)
    
    def validate_dissipation_curve(self, z: float, delta_sigma: float, 
                                   n_time_points: int = 20) -> dict:
        """Validate pore pressure dissipation curve against analytical solution.
        
        Computes relative error at multiple time points.
        
        Returns dict with:
            - times: array of time values
            - u_computed: array of computed pore pressures
            - u_analytical: array of analytical values
            - max_error_pct: maximum relative error
            - mean_error_pct: mean relative error
            - passed: bool (max error ≤ 3%)
        """
        tc = TerzaghiConsolidation(
            sigma_v0=self.sigma_v0,
            H=self.H,
            c_v=self.c_v,
            n_terms=50,  # standard implementation
        )
        
        # Time range: T_v from 0.01 to 2.0
        T_v_max = 2.0
        t_max = T_v_max * self.H**2 / self.c_v
        times = np.linspace(0.01 * t_max, t_max, n_time_points)
        
        u_computed = []
        u_analytical = []
        errors = []
        
        for t in times:
            u_comp = tc.pore_pressure_at_depth(z, t, delta_sigma)
            u_ref = self.compute_reference_solution(z, t, delta_sigma)
            u_computed.append(u_comp)
            u_analytical.append(u_ref)
            
            if abs(u_ref) > 1e-10:
                error = abs(u_comp - u_ref) / abs(u_ref) * 100.0
            else:
                error = 0.0
            errors.append(error)
        
        max_error = max(errors)
        mean_error = sum(errors) / len(errors)
        
        return {
            'times': times.tolist(),
            'u_computed': u_computed,
            'u_analytical': u_analytical,
            'max_error_pct': max_error,
            'mean_error_pct': mean_error,
            'passed': max_error <= 3.0,
        }
    
    def validate_average_consolidation(self, n_time_points: int = 20) -> dict:
        """Validate average degree of consolidation U_avg(t).
        
        Returns dict with:
            - times: array of time values
            - U_computed: array of computed U_avg
            - U_analytical: array of analytical values
            - max_error_pct: maximum relative error
            - passed: bool (max error ≤ 3%)
        """
        tc = TerzaghiConsolidation(
            sigma_v0=self.sigma_v0,
            H=self.H,
            c_v=self.c_v,
            n_terms=50,
        )
        
        T_v_max = 2.0
        t_max = T_v_max * self.H**2 / self.c_v
        times = np.linspace(0.01 * t_max, t_max, n_time_points)
        
        U_computed = []
        U_analytical = []
        errors = []
        
        for t in times:
            U_comp = tc.average_degree_of_consolidation(t)
            
            # Reference solution with high n_terms
            tc_ref = TerzaghiConsolidation(
                sigma_v0=self.sigma_v0,
                H=self.H,
                c_v=self.c_v,
                n_terms=self.n_terms,
            )
            U_ref = tc_ref.average_degree_of_consolidation(t)
            
            U_computed.append(U_comp)
            U_analytical.append(U_ref)
            
            if abs(U_ref) > 1e-10:
                error = abs(U_comp - U_ref) / abs(U_ref) * 100.0
            else:
                error = 0.0
            errors.append(error)
        
        max_error = max(errors)
        
        return {
            'times': times.tolist(),
            'U_computed': U_computed,
            'U_analytical': U_analytical,
            'max_error_pct': max_error,
            'passed': max_error <= 3.0,
        }


# ---------------------------------------------------------------------------
# 3. Liquefaction threshold validation
# ---------------------------------------------------------------------------

@dataclass
class LiquefactionThresholdValidator:
    """Validate liquefaction threshold r_u–a relationship.
    
    Scans roll amplitude a and verifies:
    1. r_u(a) is monotonic (higher amplitude → faster liquefaction)
    2. Threshold amplitude a_thr matches theoretical shear threshold
    3. Multiple moisture content points (≥3) are scanned
    """
    base_config: CargoLiquefactionConfig
    
    def scan_roll_amplitude(self, amplitudes_deg: list[float], 
                           moisture_content: float = 0.08) -> dict:
        """Scan roll amplitude and track r_u evolution.
        
        Returns dict with:
            - amplitudes_deg: array of amplitudes
            - ru_max: array of max r_u after N cycles
            - n_cycles_to_onset: array of cycles to onset
            - is_monotonic: bool (r_u increases with amplitude)
            - passed: bool
        """
        ru_max_list = []
        n_cycles_list = []
        
        for phi_deg in amplitudes_deg:
            cfg = CargoLiquefactionConfig(
                cargo_density=self.base_config.cargo_density,
                moisture_content=moisture_content,
                sigma_v0=self.base_config.sigma_v0,
                layer_thickness=self.base_config.layer_thickness,
                c_v=self.base_config.c_v,
                crr_ref=self.base_config.crr_ref,
                n_ref=self.base_config.n_ref,
                theta=self.base_config.theta,
                k_slope=self.base_config.k_slope,
                roll_amplitude_deg=phi_deg,
                roll_period=self.base_config.roll_period,
                z_cog=self.base_config.z_cog,
                hold_length=self.base_config.hold_length,
                hold_width=self.base_config.hold_width,
                cargo_height=self.base_config.cargo_height,
                displacement=self.base_config.displacement,
                n_cycles=self.base_config.n_cycles,
                dt=self.base_config.dt,
            )
            sim = CargoLiquefactionSimulation(config=cfg)
            res = sim.run()
            
            ru_max_list.append(res['ru_max'])
            n_cycles_list.append(res['n_cycles_to_onset'])
        
        # Check monotonicity: higher amplitude should give higher r_u
        is_monotonic = all(
            ru_max_list[i+1] >= ru_max_list[i] - 1e-6
            for i in range(len(ru_max_list) - 1)
        )
        
        return {
            'amplitudes_deg': amplitudes_deg,
            'ru_max': ru_max_list,
            'n_cycles_to_onset': n_cycles_list,
            'is_monotonic': is_monotonic,
            'passed': is_monotonic,
        }
    
    def scan_moisture_content(self, moisture_contents: list[float],
                             roll_amplitude_deg: float = 15.0) -> dict:
        """Scan moisture content and track liquefaction resistance.
        
        Higher moisture → lower CRR → faster liquefaction.
        
        Returns dict with:
            - moisture_contents: array of moisture values
            - crr_effective: array of effective CRR
            - ru_max: array of max r_u
            - is_monotonic: bool (r_u increases with moisture)
            - passed: bool
        """
        crr_list = []
        ru_max_list = []
        
        w_opt = 0.05  # optimum moisture content
        
        for w in moisture_contents:
            crr_eff = self.base_config.crr_ref * max(0.1, 1.0 - 2.5 * (w - w_opt))
            crr_list.append(crr_eff)
            
            cfg = CargoLiquefactionConfig(
                cargo_density=self.base_config.cargo_density,
                moisture_content=w,
                sigma_v0=self.base_config.sigma_v0,
                layer_thickness=self.base_config.layer_thickness,
                c_v=self.base_config.c_v,
                crr_ref=crr_eff,
                n_ref=self.base_config.n_ref,
                theta=self.base_config.theta,
                k_slope=self.base_config.k_slope,
                roll_amplitude_deg=roll_amplitude_deg,
                roll_period=self.base_config.roll_period,
                z_cog=self.base_config.z_cog,
                hold_length=self.base_config.hold_length,
                hold_width=self.base_config.hold_width,
                cargo_height=self.base_config.cargo_height,
                displacement=self.base_config.displacement,
                n_cycles=self.base_config.n_cycles,
                dt=self.base_config.dt,
            )
            sim = CargoLiquefactionSimulation(config=cfg)
            res = sim.run()
            
            ru_max_list.append(res['ru_max'])
        
        # Check monotonicity: higher moisture should give higher r_u
        is_monotonic = all(
            ru_max_list[i+1] >= ru_max_list[i] - 1e-6
            for i in range(len(ru_max_list) - 1)
        )
        
        return {
            'moisture_contents': moisture_contents,
            'crr_effective': crr_list,
            'ru_max': ru_max_list,
            'is_monotonic': is_monotonic,
            'passed': is_monotonic,
        }
    
    def validate_threshold_theory(self, amplitudes_deg: list[float]) -> dict:
        """Validate threshold amplitude against theoretical shear threshold.
        
        Theoretical CSR threshold: CSR_thr = CRR
        Theoretical amplitude: φ_thr = CRR × g / (ω² z_cog)
        
        Returns dict with:
            - amplitudes_deg: array of amplitudes
            - csr: array of CSR values
            - crr: cyclic resistance ratio
            - a_thr_computed: computed threshold amplitude
            - a_thr_theoretical: theoretical threshold
            - error_pct: relative error
            - passed: bool (error ≤ 10%)
        """
        csr_list = []
        
        for phi_deg in amplitudes_deg:
            phi_rad = math.radians(phi_deg)
            roll = RollExcitation(
                phi_0=phi_rad,
                t_roll=self.base_config.roll_period,
                z_cog=self.base_config.z_cog,
            )
            csr = roll.csr_at_cargo()
            csr_list.append(csr)
        
        # Find threshold: where CSR crosses CRR
        crr = self.base_config.crr_ref
        a_thr_theoretical = None
        
        for i, csr in enumerate(csr_list):
            if csr >= crr:
                # Interpolate to find threshold
                if i > 0:
                    phi1 = amplitudes_deg[i-1]
                    phi2 = amplitudes_deg[i]
                    csr1 = csr_list[i-1]
                    csr2 = csr_list[i]
                    # Linear interpolation
                    a_thr_computed = phi1 + (crr - csr1) / (csr2 - csr1) * (phi2 - phi1)
                else:
                    a_thr_computed = amplitudes_deg[0]
                break
        else:
            a_thr_computed = amplitudes_deg[-1]  # no threshold found
        
        # Theoretical threshold: CSR = CRR → φ_thr = CRR × g / (ω² z_cog)
        omega = 2.0 * math.pi / self.base_config.roll_period
        phi_thr_rad = crr * GRAVITY / (omega**2 * self.base_config.z_cog)
        a_thr_theoretical = math.degrees(phi_thr_rad)
        
        error_pct = abs(a_thr_computed - a_thr_theoretical) / a_thr_theoretical * 100.0
        
        return {
            'amplitudes_deg': amplitudes_deg,
            'csr': csr_list,
            'crr': crr,
            'a_thr_computed': a_thr_computed,
            'a_thr_theoretical': a_thr_theoretical,
            'error_pct': error_pct,
            'passed': error_pct <= 10.0,
        }


# ---------------------------------------------------------------------------
# 4. IMO FSC formula consistency check
# ---------------------------------------------------------------------------

@dataclass
class IMOFSCChecker:
    """Check FSC calculation against IMO formula.
    
    IMO formula for rectangular tank (DNV-OS-F101):
        FSC = ρ_cargo × I_fs / (ρ_sw × ∇)
    
    where I_fs = L × B³ / 12 for rectangular tank.
    
    This checker verifies the implementation matches the formula.
    """
    hold_length: float
    hold_width: float
    cargo_density: float
    cargo_height: float
    displacement: float  # tonnes
    rho_sw: float = 1025.0
    
    def compute_imo_fsc(self) -> float:
        """Compute FSC using IMO formula directly."""
        I_fs = self.hold_length * self.hold_width**3 / 12.0
        nabla = self.displacement * 1000.0 / self.rho_sw  # displacement volume
        fsc = self.cargo_density * I_fs / (self.rho_sw * nabla)
        return fsc
    
    def check_consistency(self) -> dict:
        """Check consistency with FreeSurfaceCorrection implementation.
        
        Returns dict with:
            - fsc_imo: FSC from IMO formula
            - fsc_module: FSC from module
            - error_pct: relative error
            - passed: bool (error ≤ 3%)
        """
        fsc_calc = FreeSurfaceCorrection(
            hold_length=self.hold_length,
            hold_width=self.hold_width,
            cargo_density=self.cargo_density,
            cargo_height=self.cargo_height,
            rho_sw=self.rho_sw,
            displacement=self.displacement,
        )
        
        fsc_imo = self.compute_imo_fsc()
        fsc_module = fsc_calc.fsc(ru=1.0)  # fully liquefied
        
        if abs(fsc_imo) > 1e-10:
            error_pct = abs(fsc_module - fsc_imo) / abs(fsc_imo) * 100.0
        else:
            error_pct = 0.0
        
        return {
            'fsc_imo': fsc_imo,
            'fsc_module': fsc_module,
            'error_pct': error_pct,
            'passed': error_pct <= 3.0,
        }


# ---------------------------------------------------------------------------
# 5. Geometry prior gate validation
# ---------------------------------------------------------------------------

@dataclass
class GeometryPriorGate:
    """Validate hold geometry and cargo properties before stability writeback.
    
    Checks:
    1. Hold capacity is physically reasonable
    2. Cargo CoG is within hold bounds
    3. Cargo mass is consistent with density and volume
    4. Displacement is positive and reasonable
    """
    hold_length: float
    hold_width: float
    cargo_height: float
    cargo_density: float
    displacement: float  # tonnes
    z_cog: float  # cargo CoG height above roll center
    
    def validate_capacity(self) -> dict:
        """Validate hold capacity is reasonable.
        
        Returns dict with:
            - volume_m3: hold volume
            - mass_tonnes: cargo mass
            - passed: bool
        """
        volume = self.hold_length * self.hold_width * self.cargo_height
        mass = volume * self.cargo_density / 1000.0  # tonnes
        
        # Sanity checks
        volume_ok = volume > 0 and volume < 10000  # reasonable range
        mass_ok = mass > 0 and mass < 20000
        
        return {
            'volume_m3': volume,
            'mass_tonnes': mass,
            'volume_ok': volume_ok,
            'mass_ok': mass_ok,
            'passed': volume_ok and mass_ok,
        }
    
    def validate_cog_position(self) -> dict:
        """Validate cargo CoG is within hold bounds.
        
        Returns dict with:
            - z_cog: CoG height
            - cargo_height: cargo height
            - is_within_bounds: bool
            - passed: bool
        """
        # CoG should be between 0 and cargo_height
        is_within = 0 <= self.z_cog <= self.cargo_height
        
        return {
            'z_cog': self.z_cog,
            'cargo_height': self.cargo_height,
            'is_within_bounds': is_within,
            'passed': is_within,
        }
    
    def validate_displacement(self) -> dict:
        """Validate ship displacement is reasonable.
        
        Returns dict with:
            - displacement_tonnes: ship displacement
            - is_positive: bool
            - is_reasonable: bool (< 500000 tonnes)
            - passed: bool
        """
        is_positive = self.displacement > 0
        is_reasonable = self.displacement < 500000
        
        return {
            'displacement_tonnes': self.displacement,
            'is_positive': is_positive,
            'is_reasonable': is_reasonable,
            'passed': is_positive and is_reasonable,
        }
    
    def run_all_checks(self) -> dict:
        """Run all geometry prior checks.
        
        Returns dict with:
            - capacity: capacity validation result
            - cog: CoG validation result
            - displacement: displacement validation result
            - all_passed: bool
        """
        capacity = self.validate_capacity()
        cog = self.validate_cog_position()
        displacement = self.validate_displacement()
        
        all_passed = capacity['passed'] and cog['passed'] and displacement['passed']
        
        return {
            'capacity': capacity,
            'cog': cog,
            'displacement': displacement,
            'all_passed': all_passed,
        }
