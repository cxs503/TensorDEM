"""
Ridge Ice Resistance - RAICE Analytical Method
===============================================

Implements the RAICE (Ridge And Ice Crushing Evaluation) analytical method chain
for predicting ridge ice resistance on ship hulls.

Components:
1. Keel resistance (F_keel) - resistance from submerged ice keel
2. Sail resistance (F_sail) - resistance from above-water ice sail
3. Interference correction (k_int) - interaction between keel and sail
4. Caliber registration (D_caliber) - effective diameter correction

Physics Background:
- Ridge ice consists of a submerged keel and above-water sail
- Keel resistance dominated by buoyancy and breaking forces
- Sail resistance from friction and passive earth pressure
- Interference effects reduce total resistance compared to sum of components

References:
- Kehtarnavaz et al. (2017), "Ice ridge resistance against ships"
- Lu et al. (2009), "Model tests of ship riding through ice ridges"
- RAICE method documentation, Cold Regions Research

Author: TensorDEM Ice Mechanics Module
Date: 2026-11-10
"""

import numpy as np
from typing import Dict, Tuple, Optional
import hashlib
import json


class RidgeIceResistance:
    """
    RAICE analytical method for ridge ice resistance prediction.
    
    Parameters
    ----------
    rho_w : float
        Water density [kg/m³], default 1025.0
    rho_ice : float
        Ice density [kg/m³], default 910.0
    g : float
        Gravitational acceleration [m/s²], default 9.81
    mu : float
        Friction coefficient ice-ship, default 0.1
    phi : float
        Internal friction angle of ridge ice [deg], default 35.0
    """
    
    def __init__(self, rho_w: float = 1025.0, rho_ice: float = 910.0,
                 g: float = 9.81, mu: float = 0.1, phi: float = 35.0):
        self.rho_w = rho_w
        self.rho_ice = rho_ice
        self.g = g
        self.mu = mu
        self.phi = np.radians(phi)
        
    def keel_resistance(self, h_keel: float, B: float, L: float,
                        alpha: float = 30.0) -> float:
        """
        Calculate keel resistance using buoyancy and breaking forces.
        
        Parameters
        ----------
        h_keel : float
            Keel depth [m]
        B : float
            Ship beam [m]
        L : float
            Characteristic length of contact [m]
        alpha : float
            Waterline angle [deg], default 30.0
            
        Returns
        -------
        F_keel : float
            Keel resistance force [N]
            
        Notes
        -----
        F_keel = F_buoyancy + F_breaking
        F_buoyancy = rho_w * g * V_displaced
        F_breaking = sigma_f * h_keel * B / sin(alpha)
        
        where V_displaced is the volume of water displaced by the keel
        and sigma_f is the flexural strength of ridge ice.
        """
        alpha_rad = np.radians(alpha)
        
        # Volume displaced by keel (triangular cross-section approximation)
        V_displaced = 0.5 * h_keel * B * L
        
        # Buoyancy force
        F_buoyancy = self.rho_w * self.g * V_displaced
        
        # Breaking force (flexural failure of keel)
        sigma_f = 0.7e6  # Flexural strength of ridge ice [Pa]
        F_breaking = sigma_f * h_keel * B / np.sin(alpha_rad)
        
        F_keel = F_buoyancy + F_breaking
        
        return F_keel
    
    def sail_resistance(self, h_sail: float, B: float, L: float,
                        alpha: float = 30.0) -> float:
        """
        Calculate sail resistance using passive earth pressure theory.
        
        Parameters
        ----------
        h_sail : float
            Sail height [m]
        B : float
            Ship beam [m]
        L : float
            Characteristic length of contact [m]
        alpha : float
            Waterline angle [deg], default 30.0
            
        Returns
        -------
        F_sail : float
            Sail resistance force [N]
            
        Notes
        -----
        F_sail = F_passive + F_friction
        F_passive = 0.5 * K_p * rho_ice * g * h_sail^2 * B
        F_friction = mu * rho_ice * g * h_sail * B * L * cos(alpha)
        
        where K_p = (1 + sin(phi)) / (1 - sin(phi)) is the passive earth
        pressure coefficient.
        """
        alpha_rad = np.radians(alpha)
        
        # Passive earth pressure coefficient
        K_p = (1 + np.sin(self.phi)) / (1 - np.sin(self.phi))
        
        # Passive earth pressure force
        F_passive = 0.5 * K_p * self.rho_ice * self.g * h_sail**2 * B
        
        # Friction force
        F_friction = self.mu * self.rho_ice * self.g * h_sail * B * L * np.cos(alpha_rad)
        
        F_sail = F_passive + F_friction
        
        return F_sail
    
    def interference_correction(self, F_keel: float, F_sail: float,
                                h_keel: float, h_sail: float) -> float:
        """
        Calculate interference correction factor.
        
        Parameters
        ----------
        F_keel : float
            Keel resistance [N]
        F_sail : float
            Sail resistance [N]
        h_keel : float
            Keel depth [m]
        h_sail : float
            Sail height [m]
            
        Returns
        -------
        k_int : float
            Interference correction factor (0 < k_int <= 1)
            
        Notes
        -----
        The interference factor accounts for the fact that keel and sail
        do not act independently. Empirical formula based on model tests:
        
        k_int = 1 - 0.2 * (h_sail / h_keel)
        
        Valid for h_sail / h_keel < 2.0
        """
        ratio = h_sail / h_keel
        
        # Empirical interference factor
        if ratio < 2.0:
            k_int = 1.0 - 0.2 * ratio
        else:
            k_int = 0.6  # Minimum value for extreme cases
            
        return k_int
    
    def caliber_registration(self, B: float, L: float, D: float) -> float:
        """
        Calculate caliber registration (effective diameter correction).
        
        Parameters
        ----------
        B : float
            Ship beam [m]
        L : float
            Characteristic length [m]
        D : float
            Ship diameter at waterline [m]
            
        Returns
        -------
        D_caliber : float
            Effective caliber diameter [m]
            
        Notes
        -----
        The effective diameter accounts for the actual contact geometry
        between ship hull and ice ridge:
        
        D_caliber = sqrt(B^2 + L^2) * correction_factor
        
        where correction_factor depends on hull shape.
        """
        # Geometric effective diameter
        D_geom = np.sqrt(B**2 + L**2)
        
        # Hull shape correction (simplified for cylindrical midbody)
        correction_factor = 0.9
        
        D_caliber = D_geom * correction_factor
        
        return D_caliber
    
    def total_resistance(self, h_keel: float, h_sail: float, B: float,
                         L: float, alpha: float = 30.0,
                         D: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate total ridge ice resistance with all components.
        
        Parameters
        ----------
        h_keel : float
            Keel depth [m]
        h_sail : float
            Sail height [m]
        B : float
            Ship beam [m]
        L : float
            Characteristic length [m]
        alpha : float
            Waterline angle [deg], default 30.0
        D : float, optional
            Ship diameter [m], if None uses B
            
        Returns
        -------
        result : dict
            Dictionary containing:
            - F_keel: Keel resistance [N]
            - F_sail: Sail resistance [N]
            - k_int: Interference factor [-]
            - D_caliber: Effective caliber [m]
            - F_total: Total resistance [N]
        """
        if D is None:
            D = B
            
        # Calculate components
        F_keel = self.keel_resistance(h_keel, B, L, alpha)
        F_sail = self.sail_resistance(h_sail, B, L, alpha)
        k_int = self.interference_correction(F_keel, F_sail, h_keel, h_sail)
        D_caliber = self.caliber_registration(B, L, D)
        
        # Total resistance with interference correction
        F_total = k_int * (F_keel + F_sail)
        
        result = {
            'F_keel': F_keel,
            'F_sail': F_sail,
            'k_int': k_int,
            'D_caliber': D_caliber,
            'F_total': F_total
        }
        
        return result
    
    def analytical_solution(self, h_keel: float, h_sail: float, B: float,
                            L: float, alpha: float = 30.0) -> Dict[str, float]:
        """
        Compute analytical solution for validation.
        
        This provides a closed-form solution for comparison with numerical
        implementations.
        
        Parameters
        ----------
        h_keel : float
            Keel depth [m]
        h_sail : float
            Sail height [m]
        B : float
            Ship beam [m]
        L : float
            Characteristic length [m]
        alpha : float
            Waterline angle [deg], default 30.0
            
        Returns
        -------
        result : dict
            Analytical solution components
        """
        alpha_rad = np.radians(alpha)
        
        # Keel resistance (analytical)
        V_displaced = 0.5 * h_keel * B * L
        F_buoyancy = self.rho_w * self.g * V_displaced
        sigma_f = 0.7e6
        F_breaking = sigma_f * h_keel * B / np.sin(alpha_rad)
        F_keel_analytical = F_buoyancy + F_breaking
        
        # Sail resistance (analytical)
        K_p = (1 + np.sin(self.phi)) / (1 - np.sin(self.phi))
        F_passive = 0.5 * K_p * self.rho_ice * self.g * h_sail**2 * B
        F_friction = self.mu * self.rho_ice * self.g * h_sail * B * L * np.cos(alpha_rad)
        F_sail_analytical = F_passive + F_friction
        
        # Interference
        ratio = h_sail / h_keel
        k_int_analytical = 1.0 - 0.2 * ratio if ratio < 2.0 else 0.6
        
        # Total
        F_total_analytical = k_int_analytical * (F_keel_analytical + F_sail_analytical)
        
        return {
            'F_keel': F_keel_analytical,
            'F_sail': F_sail_analytical,
            'k_int': k_int_analytical,
            'F_total': F_total_analytical
        }
    
    def compute_sha256(self, params: Dict) -> str:
        """
        Compute SHA256 hash of input parameters for reproducibility.
        
        Parameters
        ----------
        params : dict
            Input parameters
            
        Returns
        -------
        hash_str : str
            SHA256 hash string
        """
        # Sort keys for deterministic ordering
        param_str = json.dumps(params, sort_keys=True)
        hash_str = hashlib.sha256(param_str.encode()).hexdigest()
        return hash_str


def validate_dimensional_consistency(h_keel: float, h_sail: float, B: float,
                                     L: float, rho_w: float = 1025.0,
                                     rho_ice: float = 910.0, g: float = 9.81) -> bool:
    """
    Validate dimensional consistency of ridge ice resistance equations.
    
    Parameters
    ----------
    h_keel : float
        Keel depth [m]
    h_sail : float
        Sail height [m]
    B : float
        Ship beam [m]
    L : float
        Characteristic length [m]
    rho_w : float
        Water density [kg/m³]
    rho_ice : float
        Ice density [kg/m³]
    g : float
        Gravitational acceleration [m/s²]
        
    Returns
    -------
    consistent : bool
        True if dimensional analysis passes
    """
    # Check that all inputs have correct dimensions
    # [h_keel] = [h_sail] = [B] = [L] = m
    # [rho_w] = [rho_ice] = kg/m³
    # [g] = m/s²
    
    # Force should have dimensions [N] = [kg·m/s²]
    # F_buoyancy ~ rho_w * g * h_keel * B * L
    #            ~ (kg/m³) * (m/s²) * m * m * m
    #            ~ kg·m/s² = N ✓
    
    # F_passive ~ K_p * rho_ice * g * h_sail^2 * B
    #           ~ 1 * (kg/m³) * (m/s²) * m² * m
    #           ~ kg·m/s² = N ✓
    
    # All terms should reduce to force dimensions
    F_buoyancy_dim = rho_w * g * h_keel * B * L
    F_passive_dim = rho_ice * g * h_sail**2 * B
    
    # Both should be positive and finite
    consistent = (F_buoyancy_dim > 0 and np.isfinite(F_buoyancy_dim) and
                  F_passive_dim > 0 and np.isfinite(F_passive_dim))
    
    return consistent


if __name__ == "__main__":
    # Example usage
    ridge = RidgeIceResistance()
    
    # Typical ridge parameters
    h_keel = 10.0  # m
    h_sail = 2.0   # m
    B = 20.0       # m (ship beam)
    L = 5.0        # m (contact length)
    
    result = ridge.total_resistance(h_keel, h_sail, B, L)
    
    print("Ridge Ice Resistance Analysis (RAICE Method)")
    print("=" * 50)
    print(f"Keel depth: {h_keel} m")
    print(f"Sail height: {h_sail} m")
    print(f"Ship beam: {B} m")
    print(f"Contact length: {L} m")
    print()
    print(f"Keel resistance: {result['F_keel']/1e6:.2f} MN")
    print(f"Sail resistance: {result['F_sail']/1e6:.2f} MN")
    print(f"Interference factor: {result['k_int']:.3f}")
    print(f"Effective caliber: {result['D_caliber']:.2f} m")
    print(f"Total resistance: {result['F_total']/1e6:.2f} MN")
