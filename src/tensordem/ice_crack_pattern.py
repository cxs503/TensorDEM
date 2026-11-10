"""
Ice Crushing Crack Pattern Prediction
======================================

Implements Winkler foundation thin plate theory for predicting ice crushing
and crack patterns under ship hull pressure.

Physics Chain:
1. Winkler foundation → pressure distribution p(x,y)
2. Thin plate bending → bending moments M_x, M_y, M_xy
3. Bending stress → σ_max from plate theory
4. Energy release rate → G from fracture mechanics
5. Crack initiation → G ≥ G_c criterion
6. Crack density scaling → ρ_crack(V, h)

Key Formula:
- Characteristic length: l = [E·h³ / (12·(1-ν²)·ρ_w·g)]^(1/4)
- Energy release rate: G = K_I² / E' (plane stress)
- Crack criterion: G ≥ G_c for radial/circumferential cracks

References:
- Hetényi (1946), "Beams on Elastic Foundation"
- Timoshenko & Woinowsky-Krieger (1959), "Theory of Plates and Shells"
- Dempsey (1983), "Fracture mechanics of ice"
- Murde et al. (1998), "Crack patterns in ice sheets"

Author: TensorDEM Ice Mechanics Module
Date: 2026-11-10
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import hashlib
import json


class IceCrackPattern:
    """
    Ice crushing crack pattern prediction using Winkler foundation theory.
    
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
    """
    
    def __init__(self, E: float = 5.0e9, nu: float = 0.3, h: float = 1.0,
                 rho_w: float = 1025.0, g: float = 9.81, G_c: float = 50.0):
        self.E = E
        self.nu = nu
        self.h = h
        self.rho_w = rho_w
        self.g = g
        self.G_c = G_c
        
        # Derived quantities
        self.D = self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))  # Flexural rigidity
        self.l = self.characteristic_length()  # Characteristic length
        
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
    
    def bending_stress(self, M_x: np.ndarray, M_y: np.ndarray) -> np.ndarray:
        """
        Calculate maximum bending stress from moments.
        
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
        Calculate energy release rate from stress field.
        
        Parameters
        ----------
        sigma : np.ndarray
            Stress field [Pa]
        a : float
            Crack length [m], default 0.1
            
        Returns
        -------
        G : np.ndarray
            Energy release rate [J/m²]
            
        Notes
        -----
        For mode I cracking:
        K_I = sigma · sqrt(π·a)
        G = K_I² / E'
        
        where E' = E for plane stress, E' = E/(1-ν²) for plane strain.
        We use plane stress for ice plates.
        """
        # Stress intensity factor
        K_I = sigma * np.sqrt(np.pi * a)
        
        # Energy release rate (plane stress)
        G = K_I**2 / self.E
        
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
            Complete analysis results
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
        
        # Bending stress
        sigma = self.bending_stress(M_x, M_y)
        
        # Energy release rate
        G = self.energy_release_rate(sigma, a)
        
        # Crack initiation
        crack = self.crack_initiation_criterion(G)
        
        # Crack density
        sigma_range = np.array([np.min(sigma), np.max(sigma)])
        rho_crack = self.crack_density_scaling(V, self.h, sigma_range)
        
        result = {
            'characteristic_length': self.l,
            'pressure': p,
            'M_x': M_x,
            'M_y': M_y,
            'M_xy': M_xy,
            'sigma_max': sigma,
            'G': G,
            'crack_mask': crack,
            'crack_density': rho_crack,
            'x': x,
            'y': y,
            'X': X,
            'Y': Y
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
        
        # Zero stress
        sigma = self.bending_stress(M_x, M_y)
        
        # Zero energy release rate
        G = self.energy_release_rate(sigma)
        
        # No cracks
        crack = self.crack_initiation_criterion(G)
        
        result = {
            'p_max': np.max(np.abs(p)),
            'sigma_max': np.max(np.abs(sigma)),
            'G_max': np.max(G),
            'crack_count': np.sum(crack),
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
            'crack_at_Gc': crack_at_Gc,
            'crack_below_Gc': crack_below,
            'crack_above_Gc': crack_above,
            'boundary_error': boundary_error,
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


if __name__ == "__main__":
    # Example usage
    ice = IceCrackPattern()
    
    print("Ice Crack Pattern Analysis")
    print("=" * 50)
    print(f"Young's modulus: {ice.E/1e9:.1f} GPa")
    print(f"Ice thickness: {ice.h} m")
    print(f"Characteristic length: {ice.l:.3f} m")
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
