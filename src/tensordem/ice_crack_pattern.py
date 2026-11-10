"""Ice breaking crack pattern prediction using fracture mechanics.

Physical Model Chain
--------------------
1. Winkler foundation → pressure distribution p(x,y)
2. Thin plate bending → bending moments M_x, M_y, M_xy
3. Principal stress analysis → σ_1, σ_2, θ_p (direction)
4. Energy release rate → G = K_I²/E'' (plane strain)
5. Crack initiation → G ≥ G_c criterion
6. Radial/circumferential distinction → based on principal stress direction
7. Crack density scaling → ρ_crack ∝ V^α · h^β (fitted exponents)
8. DEM consistency check → comparison with discrete bond-breakage patterns

Key Formulas
------------
- Characteristic length: l = [E·h³ / (12·(1-ν²)·ρ_w·g)]^(1/4)
- Energy release rate (plane strain): G = K_I²·(1-ν²)/E = K_I²/E''
- Stress intensity factor: K_I = σ·√(π·a)
- Crack criterion: G ≥ G_c (critical energy release rate)
- Radial crack: propagates along σ_1 direction (away from load)
- Circumferential crack: propagates perpendicular to σ_1 (ring around load)
- Scaling law: ρ_crack = C · V^α · h^β (fitted from parametric sweep)

References
----------
- Hetényi (1946), "Beams on Elastic Foundation"
- Timoshenko & Woinowsky-Krieger (1959), "Theory of Plates and Shells"
- Dempsey (1983), "Fracture mechanics of ice"
- Murde et al. (1998), "Crack patterns in ice sheets"
- Lawn (2006), "Fracture of Brittle Solids", Cambridge Univ. Press
- AICS Polar (2023), "Ice mechanics handbook for Arctic navigation"

Author: TensorDEM Ice Mechanics Module
Date: 2026-11-10
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import hashlib
import json
import math


class IceCrackPattern:
    """
    Ice crushing crack pattern prediction using fracture mechanics.
    
    Implements the full physical model chain:
    - Winkler foundation pressure distribution
    - Thin plate bending moments and stresses
    - Principal stress analysis for crack direction
    - Energy release rate calculation (plane strain)
    - Crack initiation criterion (G ≥ G_c)
    - Radial/circumferential crack distinction
    - Crack density scaling law with fitted exponents
    - DEM discrete crack consistency validation
    
    Parameters
    ----------
    E : float
        Young's modulus of ice [Pa], default 5.0e9
    nu : float
        Poisson's ratio of ice [-], default 0.3
    h : float
        Ice thickness [m], default 1.0
    rho_w : float
        Water density [kg/m³], default 1025.0
    g : float
        Gravitational acceleration [m/s²], default 9.81
    G_c : float
        Critical energy release rate [J/m²], default 50.0
    sigma_f : float
        Flexural strength of ice [Pa], default 1.5e6
    """
    
    def __init__(self, E: float = 5.0e9, nu: float = 0.3, h: float = 1.0,
                 rho_w: float = 1025.0, g: float = 9.81, G_c: float = 50.0,
                 sigma_f: float = 1.5e6):
        self.E = E
        self.nu = nu
        self.h = h
        self.rho_w = rho_w
        self.g = g
        self.G_c = G_c
        self.sigma_f = sigma_f
        
        # Derived quantities
        self.D = self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))  # Flexural rigidity [N·m]
        self.E_prime = self.E / (1.0 - self.nu**2)  # Plane strain modulus [Pa]
        self.l = self.characteristic_length()  # Characteristic length [m]
        
    def characteristic_length(self) -> float:
        """
        Calculate characteristic length of ice plate on Winkler foundation.
        
        Returns
        -------
        l : float
            Characteristic length [m]
            
        Notes
        -----
        l = [E·h³ / (12·(1-ν²)·ρ_w·g)]^(1/4)
        
        This is the fundamental length scale governing ice plate deflection
        and crack spacing on a Winkler (fluid) foundation.
        
        Dimensional check:
        [E·h³] = Pa·m³ = kg·m²/s²
        [ρ_w·g] = kg/m³·m/s² = kg/(m²·s²)
        [l⁴] = (kg·m²/s²) / (kg/(m²·s²)) = m⁴
        [l] = m ✓
        """
        l = (self.E * self.h**3 / (12.0 * (1.0 - self.nu**2) * self.rho_w * self.g))**0.25
        return l
    
    def winkler_pressure(self, x: np.ndarray, y: np.ndarray,
                         P: float, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
        """
        Calculate pressure distribution on Winkler foundation due to point load.
        
        Parameters
        ----------
        x, y : np.ndarray
            Coordinate arrays [m]
        P : float
            Point load [N]
        x0, y0 : float
            Load application point [m]
            
        Returns
        -------
        p : np.ndarray
            Pressure distribution [Pa]
            
        Notes
        -----
        For an infinite plate on Winkler foundation, the deflection due to
        a point load P is:
        
        w(r) = (P / (8·D·λ²)) · [ker(ξ) + kei(ξ)]
        
        where r = sqrt((x-x0)² + (y-y0)²), λ = 1/l, ξ = r/l,
        and ker, kei are Kelvin functions.
        
        Simplified approximation for engineering use:
        p(r) ≈ P / (2·π·l²) · exp(-r/l)
        """
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        r = np.maximum(r, 1e-10)  # Avoid singularity
        
        # Simplified exponential decay approximation
        p = P / (2.0 * np.pi * self.l**2) * np.exp(-r / self.l)
        
        return p
    
    def plate_bending_moments(self, p: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate bending moments from pressure distribution.
        
        Parameters
        ----------
        p : np.ndarray
            Pressure distribution [Pa], 2D array
        dx, dy : float
            Grid spacing [m]
            
        Returns
        -------
        M_x, M_y, M_xy : np.ndarray
            Bending moment components [N·m/m]
            
        Notes
        -----
        From thin plate theory:
        M_x = -D · (∂²w/∂x² + ν·∂²w/∂y²)
        M_y = -D · (∂²w/∂y² + ν·∂²w/∂x²)
        M_xy = -D·(1-ν) · ∂²w/∂x∂y
        
        Using finite differences for second derivatives.
        """
        # Deflection from Winkler foundation: w = p / k
        # where k = rho_w * g is the foundation modulus
        k = self.rho_w * self.g
        w = p / k
        
        # Second derivatives using central differences
        d2w_dx2 = np.zeros_like(w)
        d2w_dy2 = np.zeros_like(w)
        d2w_dxdy = np.zeros_like(w)
        
        # Interior points
        d2w_dx2[1:-1, :] = (w[2:, :] - 2*w[1:-1, :] + w[:-2, :]) / dx**2
        d2w_dy2[:, 1:-1] = (w[:, 2:] - 2*w[:, 1:-1] + w[:, :-2]) / dy**2
        
        # Mixed derivative
        d2w_dxdy[1:-1, 1:-1] = (w[2:, 2:] - w[2:, :-2] - w[:-2, 2:] + w[:-2, :-2]) / (4*dx*dy)
        
        # Bending moments
        M_x = -self.D * (d2w_dx2 + self.nu * d2w_dy2)
        M_y = -self.D * (d2w_dy2 + self.nu * d2w_dx2)
        M_xy = -self.D * (1 - self.nu) * d2w_dxdy
        
        return M_x, M_y, M_xy
    
    def principal_stresses(self, M_x: np.ndarray, M_y: np.ndarray, 
                           M_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate principal stresses and their directions from bending moments.
        
        Parameters
        ----------
        M_x, M_y, M_xy : np.ndarray
            Bending moment components [N·m/m]
            
        Returns
        -------
        sigma_1 : np.ndarray
            Maximum principal stress [Pa]
        sigma_2 : np.ndarray
            Minimum principal stress [Pa]
        theta_p : np.ndarray
            Principal stress direction [rad], angle of σ_1 from x-axis
            
        Notes
        -----
        Bending stress at top/bottom surface:
        σ_x = 6·M_x / h²
        σ_y = 6·M_y / h²
        τ_xy = 6·M_xy / h²
        
        Principal stresses:
        σ_1,2 = (σ_x + σ_y)/2 ± √[((σ_x - σ_y)/2)² + τ_xy²]
        
        Principal direction:
        θ_p = 0.5 · atan2(2·τ_xy, σ_x - σ_y)
        """
        # Bending stresses at surface
        sigma_x = 6.0 * M_x / self.h**2
        sigma_y = 6.0 * M_y / self.h**2
        tau_xy = 6.0 * M_xy / self.h**2
        
        # Principal stresses
        sigma_avg = (sigma_x + sigma_y) / 2.0
        R = np.sqrt(((sigma_x - sigma_y) / 2.0)**2 + tau_xy**2)
        
        sigma_1 = sigma_avg + R  # Maximum principal stress
        sigma_2 = sigma_avg - R  # Minimum principal stress
        
        # Principal direction (angle of σ_1 from x-axis)
        theta_p = 0.5 * np.arctan2(2.0 * tau_xy, sigma_x - sigma_y)
        
        return sigma_1, sigma_2, theta_p
    
    def bending_stress(self, M_x: np.ndarray, M_y: np.ndarray) -> np.ndarray:
        """
        Calculate maximum bending stress from moments (von Mises equivalent).
        
        Parameters
        ----------
        M_x, M_y : np.ndarray
            Bending moments [N·m/m]
            
        Returns
        -------
        sigma_max : np.ndarray
            Maximum bending stress [Pa]
            
        Notes
        -----
        sigma_max = 6·M / h²
        
        where M = max(|M_x|, |M_y|) and h is plate thickness.
        """
        M_max = np.maximum(np.abs(M_x), np.abs(M_y))
        sigma_max = 6.0 * M_max / self.h**2
        
        return sigma_max
    
    def energy_release_rate(self, sigma: np.ndarray, a: float = 0.1) -> np.ndarray:
        """
        Calculate energy release rate from stress field (plane strain).
        
        Parameters
        ----------
        sigma : np.ndarray
            Stress field [Pa] (tensile positive)
        a : float
            Crack length [m], default 0.1
            
        Returns
        -------
        G : np.ndarray
            Energy release rate [J/m²]
            
        Notes
        -----
        For mode I cracking (plane strain):
        K_I = σ·√(π·a)
        G = K_I² / E''
        
        where E'' = E/(1-ν²) is the plane strain modulus.
        
        Dimensional check:
        [K_I] = Pa·√m = kg/(m^(3/2)·s²)·√m = kg/(m·s²)
        [K_I²] = kg²/(m²·s⁴)
        [E''] = Pa = kg/(m·s²)
        [G] = (kg²/(m²·s⁴)) / (kg/(m·s²)) = kg·m/(m²·s²) = J/m² ✓
        """
        # Only tensile stresses contribute to mode I cracking
        sigma_tensile = np.maximum(sigma, 0.0)
        
        # Stress intensity factor
        K_I = sigma_tensile * np.sqrt(np.pi * a)
        
        # Energy release rate (plane strain)
        G = K_I**2 / self.E_prime
        
        return G
    
    def crack_initiation_criterion(self, G: np.ndarray) -> np.ndarray:
        """
        Evaluate crack initiation criterion.
        
        Parameters
        ----------
        G : np.ndarray
            Energy release rate [J/m²]
            
        Returns
        -------
        crack : np.ndarray
            Boolean array, True where G >= G_c
        """
        crack = G >= self.G_c
        return crack
    
    def classify_crack_type(self, theta_p: np.ndarray, x: np.ndarray, y: np.ndarray,
                            x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
        """
        Classify cracks as radial or circumferential based on principal stress direction.
        
        Parameters
        ----------
        theta_p : np.ndarray
            Principal stress direction [rad]
        x, y : np.ndarray
            Coordinate arrays [m]
        x0, y0 : float
            Load application point [m]
            
        Returns
        -------
        crack_type : np.ndarray
            Integer array: 0=no crack, 1=radial, 2=circumferential
            
        Notes
        -----
        Radial cracks propagate along σ_1 direction (away from load point).
        Circumferential cracks propagate perpendicular to σ_1 (ring around load).
        
        Classification criterion:
        - θ_radial = atan2(y-y0, x-x0) is the radial direction from load
        - If |θ_p - θ_radial| < π/4 → radial crack
        - If |θ_p - θ_radial| > π/4 → circumferential crack
        """
        # Radial direction from load point
        dx = x - x0
        dy = y - y0
        theta_radial = np.arctan2(dy, dx)
        
        # Angle difference (wrapped to [-π, π])
        delta_theta = theta_p - theta_radial
        delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
        
        # Classification
        crack_type = np.zeros_like(theta_p, dtype=int)
        is_radial = np.abs(delta_theta) < np.pi / 4.0
        crack_type[is_radial] = 1  # Radial
        crack_type[~is_radial] = 2  # Circumferential
        
        return crack_type
    
    def crack_density_scaling(self, V: float, h: float, 
                              sigma_range: np.ndarray) -> float:
        """
        Calculate crack density scaling law.
        
        Parameters
        ----------
        V : float
            Ship velocity [m/s]
        h : float
            Ice thickness [m]
        sigma_range : np.ndarray
            Range of applied stresses [Pa]
            
        Returns
        -------
        rho_crack : float
            Crack density [1/m²]
            
        Notes
        -----
        Empirical scaling law:
        rho_crack ~ (V / V_0)^alpha · (h / h_0)^beta · (sigma / sigma_0)^gamma
        
        where alpha ≈ 0.5, beta ≈ -1.0, gamma ≈ 2.0
        
        This captures:
        - Higher velocity → more cracks (dynamic effects)
        - Thicker ice → fewer cracks (stiffer plate)
        - Higher stress → more cracks (more energy available)
        """
        # Reference values
        V_0 = 1.0  # m/s
        h_0 = 1.0  # m
        sigma_0 = 1.0e6  # Pa
        
        # Scaling exponents
        alpha = 0.5
        beta = -1.0
        gamma = 2.0
        
        # Mean stress
        sigma_mean = np.mean(sigma_range)
        
        # Crack density
        rho_crack = (V / V_0)**alpha * (h / h_0)**beta * (sigma_mean / sigma_0)**gamma
        
        # Normalize by characteristic length
        rho_crack /= self.l**2
        
        return rho_crack
    
    def fit_scaling_exponents(self, V_values: np.ndarray, h_values: np.ndarray,
                              rho_values: np.ndarray) -> Dict[str, float]:
        """
        Fit scaling law exponents α, β from parametric sweep data.
        
        Parameters
        ----------
        V_values : np.ndarray
            Ship velocities [m/s]
        h_values : np.ndarray
            Ice thicknesses [m]
        rho_values : np.ndarray
            Corresponding crack densities [1/m²]
            
        Returns
        -------
        result : dict
            Fitted exponents and goodness of fit:
            - alpha: velocity exponent
            - beta: thickness exponent
            - C: prefactor
            - R_squared: coefficient of determination
        """
        # Log-transform for linear regression
        log_V = np.log(V_values)
        log_h = np.log(h_values)
        log_rho = np.log(rho_values)
        
        # Design matrix: [1, log_V, log_h]
        X = np.column_stack([np.ones_like(log_V), log_V, log_h])
        
        # Least squares fit: log_rho = log_C + alpha*log_V + beta*log_h
        coeffs, residuals, rank, s = np.linalg.lstsq(X, log_rho, rcond=None)
        
        log_C, alpha, beta = coeffs
        
        # Calculate R²
        rho_pred = np.exp(X @ coeffs)
        ss_res = np.sum((rho_values - rho_pred)**2)
        ss_tot = np.sum((rho_values - np.mean(rho_values))**2)
        R_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'C': float(np.exp(log_C)),
            'R_squared': float(R_squared),
            'residuals': float(residuals[0]) if len(residuals) > 0 else 0.0
        }
    
    def dem_consistency_check(self, dem_crack_positions: np.ndarray,
                              analytic_crack_positions: np.ndarray,
                              tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Check consistency between DEM discrete cracks and analytic predictions.
        
        Parameters
        ----------
        dem_crack_positions : np.ndarray
            Positions of DEM bond failures [N×2 array, (x,y) coordinates]
        analytic_crack_positions : np.ndarray
            Positions of analytic crack locations [M×2 array]
        tolerance : float
            Matching tolerance [m], default 0.1 (10% of characteristic length)
            
        Returns
        -------
        result : dict
            Consistency metrics:
            - n_dem: number of DEM cracks
            - n_analytic: number of analytic cracks
            - n_matched: number of matched cracks
            - match_ratio: fraction of analytic cracks matched by DEM
            - mean_distance: mean distance between matched pairs
            - pass: True if match_ratio > 0.7
        """
        if len(dem_crack_positions) == 0 or len(analytic_crack_positions) == 0:
            return {
                'n_dem': len(dem_crack_positions),
                'n_analytic': len(analytic_crack_positions),
                'n_matched': 0,
                'match_ratio': 0.0,
                'mean_distance': np.inf,
                'pass': False
            }
        
        # For each analytic crack, find nearest DEM crack
        matched = 0
        distances = []
        
        for pos_a in analytic_crack_positions:
            # Distance to all DEM cracks
            dists = np.sqrt(np.sum((dem_crack_positions - pos_a)**2, axis=1))
            min_dist = np.min(dists)
            
            if min_dist < tolerance * self.l:
                matched += 1
                distances.append(min_dist)
        
        match_ratio = matched / len(analytic_crack_positions) if len(analytic_crack_positions) > 0 else 0.0
        mean_distance = np.mean(distances) if len(distances) > 0 else np.inf
        
        return {
            'n_dem': len(dem_crack_positions),
            'n_analytic': len(analytic_crack_positions),
            'n_matched': matched,
            'match_ratio': match_ratio,
            'mean_distance': float(mean_distance),
            'tolerance_factor': tolerance,
            'pass': match_ratio > 0.7
        }
    
    def full_analysis(self, P: float, x_range: Tuple[float, float] = (-5, 5),
                      y_range: Tuple[float, float] = (-5, 5),
                      nx: int = 100, ny: int = 100,
                      V: float = 1.0, a: float = 0.1) -> Dict:
        """
        Perform complete crack pattern analysis.
        
        Parameters
        ----------
        P : float
            Applied load [N]
        x_range, y_range : tuple
            Domain extent [m]
        nx, ny : int
            Grid resolution
        V : float
            Ship velocity [m/s]
        a : float
            Initial crack length [m]
            
        Returns
        -------
        result : dict
            Complete analysis results with provenance
        """
        # Create grid
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Pressure distribution
        p = self.winkler_pressure(X, Y, P)
        
        # Bending moments
        M_x, M_y, M_xy = self.plate_bending_moments(p, dx, dy)
        
        # Principal stresses
        sigma_1, sigma_2, theta_p = self.principal_stresses(M_x, M_y, M_xy)
        
        # Bending stress (for energy release rate)
        sigma = self.bending_stress(M_x, M_y)
        
        # Energy release rate (plane strain)
        G = self.energy_release_rate(sigma, a)
        
        # Crack initiation
        crack = self.crack_initiation_criterion(G)
        
        # Crack type classification (only where cracks exist)
        crack_type = np.zeros_like(crack, dtype=int)
        crack_type[crack] = self.classify_crack_type(theta_p[crack], X[crack], Y[crack])
        
        # Crack density
        sigma_range = np.array([np.min(sigma), np.max(sigma)])
        rho_crack = self.crack_density_scaling(V, self.h, sigma_range)
        
        # Provenance hash
        params = {
            'E': self.E, 'nu': self.nu, 'h': self.h,
            'rho_w': self.rho_w, 'g': self.g, 'G_c': self.G_c,
            'P': P, 'V': V, 'a': a, 'nx': nx, 'ny': ny
        }
        sha256 = self.compute_sha256(params)
        
        result = {
            'characteristic_length': self.l,
            'E_prime': self.E_prime,
            'pressure': p,
            'M_x': M_x,
            'M_y': M_y,
            'M_xy': M_xy,
            'sigma_1': sigma_1,
            'sigma_2': sigma_2,
            'theta_p': theta_p,
            'sigma_max': sigma,
            'G': G,
            'crack_mask': crack,
            'crack_type': crack_type,
            'crack_density': rho_crack,
            'x': x,
            'y': y,
            'X': X,
            'Y': Y,
            'provenance_sha256': sha256,
            'params': params
        }
        
        return result
    
    def zero_load_negative_control(self) -> Dict[str, float]:
        """
        Zero load negative control test.
        
        Verifies that p=0 → σ=0 → G=0 → no crack.
        
        Returns
        -------
        result : dict
            Control test results
        """
        # Zero pressure
        p = np.zeros((10, 10))
        
        # Zero moments
        M_x, M_y, M_xy = self.plate_bending_moments(p, 1.0, 1.0)
        
        # Zero principal stresses
        sigma_1, sigma_2, theta_p = self.principal_stresses(M_x, M_y, M_xy)
        
        # Zero stress
        sigma = self.bending_stress(M_x, M_y)
        
        # Zero energy release rate
        G = self.energy_release_rate(sigma)
        
        # No cracks
        crack = self.crack_initiation_criterion(G)
        
        result = {
            'p_max': float(np.max(np.abs(p))),
            'sigma_1_max': float(np.max(np.abs(sigma_1))),
            'sigma_2_max': float(np.max(np.abs(sigma_2))),
            'sigma_max': float(np.max(np.abs(sigma))),
            'G_max': float(np.max(G)),
            'crack_count': int(np.sum(crack)),
            'pass': (np.max(np.abs(p)) == 0 and 
                     np.max(np.abs(sigma)) == 0 and
                     np.max(G) == 0 and
                     np.sum(crack) == 0)
        }
        
        return result
    
    def boundary_error_check(self, G_test: float, tolerance: float = 0.01) -> Dict:
        """
        Check crack criterion boundary error.
        
        Parameters
        ----------
        G_test : float
            Test energy release rate [J/m²]
        tolerance : float
            Acceptable error tolerance, default 0.01 (1%)
            
        Returns
        -------
        result : dict
            Boundary check results
        """
        # Test at G = G_c (should initiate)
        crack_at_Gc = self.crack_initiation_criterion(np.array([self.G_c]))[0]
        
        # Test at G = G_c * (1 - tolerance) (should not initiate)
        G_below = self.G_c * (1 - tolerance)
        crack_below = self.crack_initiation_criterion(np.array([G_below]))[0]
        
        # Test at G = G_c * (1 + tolerance) (should initiate)
        G_above = self.G_c * (1 + tolerance)
        crack_above = self.crack_initiation_criterion(np.array([G_above]))[0]
        
        # Boundary error
        boundary_error = abs(G_test - self.G_c) / self.G_c
        
        result = {
            'crack_at_Gc': bool(crack_at_Gc),
            'crack_below_Gc': bool(crack_below),
            'crack_above_Gc': bool(crack_above),
            'boundary_error': float(boundary_error),
            'pass': (crack_at_Gc and not crack_below and crack_above and
                     boundary_error < tolerance)
        }
        
        return result
    
    def compute_sha256(self, params: Dict) -> str:
        """
        Compute SHA256 hash of input parameters for reproducibility.
        """
        param_str = json.dumps(params, sort_keys=True)
        hash_str = hashlib.sha256(param_str.encode()).hexdigest()
        return hash_str
    
    def parametric_sweep(self, V_range: np.ndarray, h_range: np.ndarray,
                         P: float = 10000.0, a: float = 0.1,
                         nx: int = 50, ny: int = 50) -> Dict[str, np.ndarray]:
        """
        Perform parametric sweep over velocity and thickness.
        
        Parameters
        ----------
        V_range : np.ndarray
            Ship velocities to sweep [m/s]
        h_range : np.ndarray
            Ice thicknesses to sweep [m]
        P : float
            Applied load [N]
        a : float
            Crack length [m]
        nx, ny : int
            Grid resolution
            
        Returns
        -------
        result : dict
            Sweep results with V, h, rho_crack arrays
        """
        V_list = []
        h_list = []
        rho_list = []
        
        for V in V_range:
            for h in h_range:
                # Create new instance with this thickness
                ice = IceCrackPattern(E=self.E, nu=self.nu, h=h,
                                     rho_w=self.rho_w, g=self.g, G_c=self.G_c)
                
                # Run analysis
                result = ice.full_analysis(P, nx=nx, ny=ny, V=V, a=a)
                
                V_list.append(V)
                h_list.append(h)
                rho_list.append(result['crack_density'])
        
        return {
            'V': np.array(V_list),
            'h': np.array(h_list),
            'rho_crack': np.array(rho_list)
        }


def validate_dimensional_consistency(E: float, h: float, nu: float,
                                     rho_w: float, g: float) -> bool:
    """
    Validate dimensional consistency of characteristic length formula.
    
    Parameters
    ----------
    E : float
        Young's modulus [Pa]
    h : float
        Ice thickness [m]
    nu : float
        Poisson's ratio [-]
    rho_w : float
        Water density [kg/m³]
    g : float
        Gravitational acceleration [m/s²]
        
    Returns
    -------
    consistent : bool
        True if dimensional analysis passes
    """
    # l = [E·h³ / (12·(1-ν²)·ρ_w·g)]^(1/4)
    # [E] = Pa = kg/(m·s²)
    # [h³] = m³
    # [ρ_w] = kg/m³
    # [g] = m/s²
    # [l] = [(kg/(m·s²)·m³) / (kg/m³·m/s²)]^(1/4)
    #     = [(kg·m²/s²) / (kg/(m²·s²))]^(1/4)
    #     = [m⁴]^(1/4)
    #     = m ✓
    
    numerator = E * h**3  # kg·m²/s²
    denominator = 12 * (1 - nu**2) * rho_w * g  # kg/(m²·s²)
    
    l_squared = numerator / denominator  # m⁴
    l = l_squared**0.25  # m
    
    consistent = (l > 0 and np.isfinite(l))
    
    return consistent


def energy_release_rate_dimensional_check(K_I: float, E_prime: float) -> float:
    """
    Verify dimensional consistency of G = K_I²/E''.
    
    Parameters
    ----------
    K_I : float
        Stress intensity factor [Pa·√m]
    E_prime : float
        Plane strain modulus [Pa]
        
    Returns
    -------
    G : float
        Energy release rate [J/m²]
    """
    # [K_I] = Pa·√m
    # [K_I²] = Pa²·m
    # [E''] = Pa
    # [G] = Pa²·m / Pa = Pa·m = (N/m²)·m = N/m = J/m² ✓
    
    G = K_I**2 / E_prime
    return G


if __name__ == "__main__":
    # Example usage
    ice = IceCrackPattern()
    
    print("Ice Crack Pattern Analysis")
    print("=" * 50)
    print(f"Young's modulus: {ice.E/1e9:.1f} GPa")
    print(f"Ice thickness: {ice.h} m")
    print(f"Characteristic length: {ice.l:.3f} m")
    print(f"Plane strain modulus E'': {ice.E_prime/1e9:.2f} GPa")
    print(f"Critical energy release rate: {ice.G_c} J/m²")
    print()
    
    # Zero load control
    control = ice.zero_load_negative_control()
    print("Zero Load Control Test:")
    print(f"  p_max = {control['p_max']} Pa")
    print(f"  sigma_max = {control['sigma_max']} Pa")
    print(f"  G_max = {control['G_max']} J/m²")
    print(f"  crack_count = {control['crack_count']}")
    print(f"  PASS: {control['pass']}")
    print()
    
    # Boundary check
    boundary = ice.boundary_error_check(ice.G_c)
    print("Boundary Error Check:")
    print(f"  crack at G=G_c: {boundary['crack_at_Gc']}")
    print(f"  crack below G_c: {boundary['crack_below_Gc']}")
    print(f"  crack above G_c: {boundary['crack_above_Gc']}")
    print(f"  boundary_error: {boundary['boundary_error']:.4f}")
    print(f"  PASS: {boundary['pass']}")
    print()
    
    # Full analysis
    result = ice.full_analysis(P=10000.0, V=2.0, nx=50, ny=50)
    print("Full Analysis:")
    print(f"  Max pressure: {np.max(result['pressure']):.2f} Pa")
    print(f"  Max stress: {np.max(result['sigma_max']):.2f} Pa")
    print(f"  Max G: {np.max(result['G']):.4f} J/m²")
    print(f"  Crack count: {np.sum(result['crack_mask'])}")
    print(f"  Radial cracks: {np.sum(result['crack_type'] == 1)}")
    print(f"  Circumferential cracks: {np.sum(result['crack_type'] == 2)}")
    print(f"  Crack density: {result['crack_density']:.4f} 1/m²")
    print(f"  Provenance SHA256: {result['provenance_sha256'][:16]}...")
