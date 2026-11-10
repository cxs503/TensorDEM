"""Bending failure model for ice floes.

When ice floes are compressed, they experience bending moments at contact points.
If the bending stress exceeds the bending strength, the floe fractures.

This module implements a simple bending failure criterion based on the contact
force and floe geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import torch

__all__ = [
    "BendingFailureConfig",
    "BendingFailureModel",
    "check_bending_failure",
]


@dataclass(frozen=True)
class BendingFailureConfig:
    """Configuration for the bending failure model.
    
    Attributes
    ----------
    bending_strength : float
        Bending strength of the ice (Pa).
    ice_thickness : float
        Ice thickness (metres).
    elastic_modulus : float
        Elastic modulus of ice (Pa).
    poisson_ratio : float
        Poisson's ratio of ice.
    fracture_energy : float
        Fracture energy per unit area (J/m^2).
    """
    bending_strength: float = 5.0e5
    ice_thickness: float = 1.0
    elastic_modulus: float = 5.0e9
    poisson_ratio: float = 0.33
    fracture_energy: float = 100.0
    
    def __post_init__(self) -> None:
        if self.bending_strength <= 0:
            raise ValueError("bending_strength must be positive")
        if self.ice_thickness <= 0:
            raise ValueError("ice_thickness must be positive")
        if self.elastic_modulus <= 0:
            raise ValueError("elastic_modulus must be positive")
        if not 0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in (0, 0.5)")
        if self.fracture_energy <= 0:
            raise ValueError("fracture_energy must be positive")


class BendingFailureModel:
    """Model for bending failure of ice floes.
    
    The bending stress in a floe of thickness h subjected to a contact force F
    over a contact length L is approximately:
    
        sigma_b = M * (h/2) / I
    
    where M is the bending moment and I is the second moment of area.
    
    For a rectangular cross-section of width L and thickness h:
        I = L * h^3 / 12
    
    The bending moment is related to the contact force by:
        M ~ F * L / 4  (for a simply supported beam with central load)
    
    Therefore:
        sigma_b = 3 * F / (2 * L * h)
    
    Failure occurs when sigma_b > sigma_f (bending strength).
    """
    
    def __init__(self, config: BendingFailureConfig):
        self.config = config
    
    def compute_bending_stress(
        self,
        contact_force: float,
        contact_length: float,
        ice_thickness: float,
    ) -> float:
        """Compute the bending stress at a contact point.
        
        Parameters
        ----------
        contact_force : float
            Contact force (N).
        contact_length : float
            Contact length (metres).
        ice_thickness : float
            Ice thickness (metres).
        
        Returns
        -------
        float
            Bending stress (Pa).
        """
        if contact_length <= 0 or ice_thickness <= 0:
            return 0.0
        return 3.0 * contact_force / (2.0 * contact_length * ice_thickness)
    
    def check_failure(
        self,
        contact_force: float,
        contact_length: float,
        ice_thickness: float,
        bending_strength: float,
    ) -> bool:
        """Check if bending failure occurs.
        
        Parameters
        ----------
        contact_force : float
            Contact force (N).
        contact_length : float
            Contact length (metres).
        ice_thickness : float
            Ice thickness (metres).
        bending_strength : float
            Bending strength (Pa).
        
        Returns
        -------
        bool
            True if failure occurs, False otherwise.
        """
        stress = self.compute_bending_stress(contact_force, contact_length, ice_thickness)
        return stress > bending_strength
    
    def critical_force(
        self,
        contact_length: float,
        ice_thickness: float,
        bending_strength: float,
    ) -> float:
        """Compute the critical force for bending failure.
        
        Parameters
        ----------
        contact_length : float
            Contact length (metres).
        ice_thickness : float
            Ice thickness (metres).
        bending_strength : float
            Bending strength (Pa).
        
        Returns
        -------
        float
            Critical force (N).
        """
        if contact_length <= 0 or ice_thickness <= 0:
            return float("inf")
        return 2.0 * bending_strength * contact_length * ice_thickness / 3.0


def check_bending_failure(
    contact_force: float,
    contact_length: float,
    ice_thickness: float,
    bending_strength: float,
) -> bool:
    """Convenience function to check bending failure.
    
    Parameters
    ----------
    contact_force : float
        Contact force (N).
    contact_length : float
        Contact length (metres).
    ice_thickness : float
        Ice thickness (metres).
    bending_strength : float
        Bending strength (Pa).
    
    Returns
    -------
    bool
        True if failure occurs, False otherwise.
    """
    config = BendingFailureConfig()
    model = BendingFailureModel(config)
    return model.check_failure(contact_force, contact_length, ice_thickness, bending_strength)
