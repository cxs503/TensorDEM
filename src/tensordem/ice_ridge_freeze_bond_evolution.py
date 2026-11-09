"""Freeze-bond strength time evolution model for ice ridge keels (TensorDEM P54-W2-N04).

Physical background
-------------------
When an ice ridge forms, the rubble pieces are initially loose (unbonded).
Over time, at sub-zero temperatures, freeze bonds (sintered ice bridges) form
between contacting ice fragments. The bond strength grows with age as the ice
crystals sinter and coalesce across the contact interface.

This module implements:
1. **Freeze-bond formation criterion**: bonds form when contact temperature
   is below the freezing point of ice (T < 0°C for fresh water).
2. **Strength growth model**: exponential saturation law
   ``sigma(t) = sigma_inf * (1 - exp(-t / tau_growth))``
   where ``sigma_inf`` is the asymptotic bond strength (fully sintered),
   ``tau_growth`` is the characteristic growth time, and ``t`` is the bond age.
3. **DEM integration**: bond strength evolves from constant to time-dependent
   function, enabling age-dependent resistance calculations.
4. **Keel resistance vs. age**: total ridge resistance grows as freeze bonds
   strengthen, interfacing with Lindqvist-type ridge resistance models.

Evidence grading
----------------
* Exponential saturation law: REAL (standard sintering kinetics, e.g.,
  Timco & Frederking 1990, Mellor 1983)
* Characteristic growth time ``tau_growth``: PARAMETRIC_MODEL (typical values
  from laboratory sintering experiments; field calibration is WITHHELD)
* Asymptotic strength ``sigma_inf``: REAL (laboratory measurements of ice-ice
  bond strength, e.g., Schulson 2001)
* Lindqvist ridge resistance interface: REAL (ISO 19906 / Lindqvist 1989)

Limitations
-----------
* Temperature is assumed uniform (no thermal gradient within the keel).
* No moisture transport or brine pocket effects (pure fresh-water ice).
* Bond growth is isotropic (no directional preference).
* No bond degradation (melting, creep damage) in this version.

All quantities are SI (m, kg, s, K, N, Pa).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
import numpy as np

__all__ = [
    "FREEZING_POINT_K",
    "FreezeBondModel",
    "ExponentialSaturationGrowth",
    "compute_bond_strength_evolution",
    "compute_keel_resistance_evolution",
    "FreezeBondFormationCriterion",
    "LindqvistRidgeResistance",
    "run_freeze_bond_study",
]

#: Freezing point of fresh water ice in Kelvin
FREEZING_POINT_K = 273.15


@dataclass(frozen=True)
class FreezeBondFormationCriterion:
    """Criterion for freeze-bond formation based on contact temperature.
    
    Bonds form when the contact temperature is below the freezing point.
    """
    
    freezing_temperature_K: float = FREEZING_POINT_K
    
    def __post_init__(self) -> None:
        if not math.isfinite(self.freezing_temperature_K):
            raise ValueError("freezing_temperature_K must be finite")
        if self.freezing_temperature_K <= 0:
            raise ValueError("freezing_temperature_K must be positive")
    
    def should_form_bond(self, temperature_K: float | torch.Tensor) -> bool | torch.Tensor:
        """Return True if temperature is below freezing point."""
        if isinstance(temperature_K, torch.Tensor):
            return temperature_K < self.freezing_temperature_K
        return float(temperature_K) < self.freezing_temperature_K


@dataclass(frozen=True)
class ExponentialSaturationGrowth:
    """Exponential saturation law for bond strength growth.
    
    ``sigma(t) = sigma_inf * (1 - exp(-t / tau_growth))``
    
    where:
    - ``sigma_inf``: asymptotic (fully sintered) bond strength [Pa]
    - ``tau_growth``: characteristic growth time [s]
    - ``t``: bond age [s]
    
    Physical basis
    --------------
    Ice sintering follows diffusion-controlled kinetics. The exponential
    saturation law is the simplest model that captures:
    - Initial rapid strength gain (t << tau_growth): sigma ~ sigma_inf * t / tau
    - Long-term saturation (t >> tau_growth): sigma -> sigma_inf
    
    Typical values (fresh-water ice at -10°C):
    - sigma_inf: 0.5 - 2.0 MPa (tensile), 1.0 - 4.0 MPa (shear)
    - tau_growth: 1e4 - 1e6 s (hours to days)
    
    References
    ----------
    - Timco & Frederking (1990): Ice mechanics, Cold Regions Sci. Tech.
    - Mellor (1983): Mechanical behavior of ice, USACRRL
    - Schulson (2001): Brittle failure of ice, Engineering Fracture Mechanics
    """
    
    asymptotic_tensile_strength_Pa: float = 1.0e6  # 1 MPa
    asymptotic_shear_strength_Pa: float = 2.0e6    # 2 MPa
    growth_time_s: float = 3.6e5  # ~100 hours
    
    def __post_init__(self) -> None:
        if not math.isfinite(self.asymptotic_tensile_strength_Pa):
            raise ValueError("asymptotic_tensile_strength_Pa must be finite")
        if self.asymptotic_tensile_strength_Pa <= 0:
            raise ValueError("asymptotic_tensile_strength_Pa must be positive")
        if not math.isfinite(self.asymptotic_shear_strength_Pa):
            raise ValueError("asymptotic_shear_strength_Pa must be finite")
        if self.asymptotic_shear_strength_Pa <= 0:
            raise ValueError("asymptotic_shear_strength_Pa must be positive")
        if not math.isfinite(self.growth_time_s):
            raise ValueError("growth_time_s must be finite")
        if self.growth_time_s <= 0:
            raise ValueError("growth_time_s must be positive")
    
    def tensile_strength_at_age(self, age_s: float | torch.Tensor) -> float | torch.Tensor:
        """Tensile bond strength at given age [Pa]."""
        if isinstance(age_s, torch.Tensor):
            age = age_s.clamp(min=0.0)
            return self.asymptotic_tensile_strength_Pa * (1.0 - torch.exp(-age / self.growth_time_s))
        age = max(0.0, float(age_s))
        return self.asymptotic_tensile_strength_Pa * (1.0 - math.exp(-age / self.growth_time_s))
    
    def shear_strength_at_age(self, age_s: float | torch.Tensor) -> float | torch.Tensor:
        """Shear bond strength at given age [Pa]."""
        if isinstance(age_s, torch.Tensor):
            age = age_s.clamp(min=0.0)
            return self.asymptotic_shear_strength_Pa * (1.0 - torch.exp(-age / self.growth_time_s))
        age = max(0.0, float(age_s))
        return self.asymptotic_shear_strength_Pa * (1.0 - math.exp(-age / self.growth_time_s))
    
    def strength_ratio_at_age(self, age_s: float) -> float:
        """Ratio of current strength to asymptotic strength (dimensionless)."""
        age = max(0.0, float(age_s))
        return 1.0 - math.exp(-age / self.growth_time_s)


@dataclass(frozen=True)
class FreezeBondModel:
    """Complete freeze-bond model: formation criterion + strength growth.
    
    This is the main interface for DEM integration. Given a contact temperature
    and bond age, it returns the current bond strength.
    """
    
    formation_criterion: FreezeBondFormationCriterion
    growth_law: ExponentialSaturationGrowth
    
    def bond_strength_at_age(
        self,
        age_s: float | torch.Tensor,
        temperature_K: float | torch.Tensor,
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Return (tensile_strength, shear_strength) at given age and temperature.
        
        If temperature is above freezing, returns (0, 0) -- no bond.
        """
        if isinstance(temperature_K, torch.Tensor):
            mask = self.formation_criterion.should_form_bond(temperature_K)
            tensile = torch.where(
                mask,
                self.growth_law.tensile_strength_at_age(age_s),
                torch.zeros_like(age_s if isinstance(age_s, torch.Tensor) 
                                else torch.tensor(age_s, dtype=torch.float64))
            )
            shear = torch.where(
                mask,
                self.growth_law.shear_strength_at_age(age_s),
                torch.zeros_like(tensile)
            )
            return tensile, shear
        
        if not self.formation_criterion.should_form_bond(temperature_K):
            return 0.0, 0.0
        
        tensile = self.growth_law.tensile_strength_at_age(age_s)
        shear = self.growth_law.shear_strength_at_age(age_s)
        return tensile, shear


def compute_bond_strength_evolution(
    model: FreezeBondModel,
    ages_s: np.ndarray | torch.Tensor,
    temperature_K: float = 263.15,  # -10°C
) -> dict[str, np.ndarray]:
    """Compute bond strength evolution over a range of ages.
    
    Parameters
    ----------
    model
        Freeze bond model
    ages_s
        Array of bond ages [s]
    temperature_K
        Contact temperature [K] (must be below freezing for bonds to form)
    
    Returns
    -------
    dict with keys:
        - 'ages_s': input ages
        - 'tensile_strength_Pa': tensile strength at each age
        - 'shear_strength_Pa': shear strength at each age
        - 'strength_ratio': ratio to asymptotic strength
    """
    if isinstance(ages_s, torch.Tensor):
        ages_np = ages_s.cpu().numpy()
    else:
        ages_np = np.asarray(ages_s, dtype=float)
    
    tensile_list = []
    shear_list = []
    ratio_list = []
    
    for age in ages_np:
        tensile, shear = model.bond_strength_at_age(age, temperature_K)
        if isinstance(tensile, torch.Tensor):
            tensile = tensile.item()
            shear = shear.item()
        tensile_list.append(tensile)
        shear_list.append(shear)
        ratio_list.append(model.growth_law.strength_ratio_at_age(age))
    
    return {
        'ages_s': ages_np,
        'tensile_strength_Pa': np.array(tensile_list),
        'shear_strength_Pa': np.array(shear_list),
        'strength_ratio': np.array(ratio_list),
    }


@dataclass(frozen=True)
class LindqvistRidgeResistance:
    """Lindqvist-type ridge resistance model with freeze-bond contribution.
    
    Total ridge resistance = rubble friction + freeze-bond strength + buoyancy.
    
    The freeze-bond contribution scales with the solid fraction (1 - porosity)
    and the bond strength, which evolves with age.
    
    Reference
    ---------
    Lindqvist (1989): Direct calculations of grounding forces on a conical
    structure, Proc. POAC'89.
    """
    
    keel_depth_m: float = 10.0
    keel_width_m: float = 20.0
    porosity: float = 0.30
    ice_density_kg_m3: float = 917.0
    water_density_kg_m3: float = 1025.0
    gravity_m_s2: float = 9.80665
    rubble_friction_coefficient: float = 0.6
    sail_height_m: float = 2.0
    
    def __post_init__(self) -> None:
        if not math.isfinite(self.keel_depth_m) or self.keel_depth_m <= 0:
            raise ValueError("keel_depth_m must be positive")
        if not math.isfinite(self.keel_width_m) or self.keel_width_m <= 0:
            raise ValueError("keel_width_m must be positive")
        if not math.isfinite(self.porosity) or not (0 <= self.porosity < 1):
            raise ValueError("porosity must be in [0, 1)")
        if not math.isfinite(self.ice_density_kg_m3) or self.ice_density_kg_m3 <= 0:
            raise ValueError("ice_density_kg_m3 must be positive")
        if not math.isfinite(self.water_density_kg_m3) or self.water_density_kg_m3 <= 0:
            raise ValueError("water_density_kg_m3 must be positive")
        if self.ice_density_kg_m3 > self.water_density_kg_m3:
            raise ValueError("ice must be buoyant: ice_density <= water_density")
        if not math.isfinite(self.rubble_friction_coefficient) or self.rubble_friction_coefficient < 0:
            raise ValueError("rubble_friction_coefficient must be non-negative")
    
    @property
    def solid_fraction(self) -> float:
        """Solid fraction (1 - porosity)."""
        return 1.0 - self.porosity
    
    @property
    def keel_volume_per_unit_length(self) -> float:
        """Keel volume per unit out-of-plane length [m^3/m]."""
        return 0.5 * self.keel_width_m * self.keel_depth_m
    
    @property
    def buoyancy_force_per_unit_length(self) -> float:
        """Net buoyancy force per unit length [N/m] (positive = downward)."""
        volume = self.keel_volume_per_unit_length * self.solid_fraction
        return (self.water_density_kg_m3 - self.ice_density_kg_m3) * volume * self.gravity_m_s2
    
    def rubble_friction_force(
        self,
        vertical_load_N_per_m: float,
    ) -> float:
        """Rubble friction resistance [N/m] from vertical load."""
        return self.rubble_friction_coefficient * vertical_load_N_per_m
    
    def freeze_bond_force(
        self,
        bond_strength_Pa: float,
        bond_area_fraction: float = 0.5,
    ) -> float:
        """Freeze-bond contribution to ridge resistance [N/m].
        
        Parameters
        ----------
        bond_strength_Pa
            Average bond strength (tensile or shear, depending on failure mode) [Pa]
        bond_area_fraction
            Fraction of keel cross-section occupied by bonds (typical 0.3-0.7)
        """
        cross_section_area = self.keel_volume_per_unit_length
        return bond_strength_Pa * bond_area_fraction * cross_section_area
    
    def total_resistance(
        self,
        bond_strength_Pa: float,
        vertical_load_N_per_m: float | None = None,
        bond_area_fraction: float = 0.5,
    ) -> dict[str, float]:
        """Total ridge resistance [N/m] with breakdown.
        
        Parameters
        ----------
        bond_strength_Pa
            Current freeze-bond strength [Pa]
        vertical_load_N_per_m
            Vertical load on keel [N/m] (if None, uses buoyancy)
        bond_area_fraction
            Fraction of keel cross-section with bonds
        
        Returns
        -------
        dict with keys:
            - 'total_N_per_m': total resistance
            - 'rubble_friction_N_per_m': friction component
            - 'freeze_bond_N_per_m': bond strength component
            - 'buoyancy_N_per_m': buoyancy component
        """
        if vertical_load_N_per_m is None:
            vertical_load_N_per_m = self.buoyancy_force_per_unit_length
        
        rubble_friction = self.rubble_friction_force(vertical_load_N_per_m)
        freeze_bond = self.freeze_bond_force(bond_strength_Pa, bond_area_fraction)
        buoyancy = self.buoyancy_force_per_unit_length
        
        total = rubble_friction + freeze_bond + buoyancy
        
        return {
            'total_N_per_m': total,
            'rubble_friction_N_per_m': rubble_friction,
            'freeze_bond_N_per_m': freeze_bond,
            'buoyancy_N_per_m': buoyancy,
        }


def compute_keel_resistance_evolution(
    ridge: LindqvistRidgeResistance,
    bond_model: FreezeBondModel,
    ages_s: np.ndarray,
    temperature_K: float = 263.15,
    vertical_load_N_per_m: float | None = None,
    bond_area_fraction: float = 0.5,
) -> dict[str, np.ndarray]:
    """Compute keel resistance evolution over time.
    
    Parameters
    ----------
    ridge
        Lindqvist ridge resistance model
    bond_model
        Freeze bond model
    ages_s
        Array of keel ages [s]
    temperature_K
        Contact temperature [K]
    vertical_load_N_per_m
        Vertical load [N/m] (if None, uses buoyancy)
    bond_area_fraction
        Fraction of keel cross-section with bonds
    
    Returns
    -------
    dict with keys:
        - 'ages_s': input ages
        - 'ages_hours': ages in hours
        - 'bond_strength_Pa': bond strength at each age
        - 'total_resistance_N_per_m': total resistance
        - 'rubble_friction_N_per_m': friction component
        - 'freeze_bond_N_per_m': bond component
        - 'buoyancy_N_per_m': buoyancy component
    """
    ages_np = np.asarray(ages_s, dtype=float)
    
    bond_strengths = []
    total_resistances = []
    rubble_frictions = []
    freeze_bonds = []
    buoyancies = []
    
    for age in ages_np:
        tensile, shear = bond_model.bond_strength_at_age(age, temperature_K)
        if isinstance(tensile, torch.Tensor):
            tensile = tensile.item()
        bond_strengths.append(tensile)
        
        resistance = ridge.total_resistance(
            tensile,
            vertical_load_N_per_m,
            bond_area_fraction,
        )
        total_resistances.append(resistance['total_N_per_m'])
        rubble_frictions.append(resistance['rubble_friction_N_per_m'])
        freeze_bonds.append(resistance['freeze_bond_N_per_m'])
        buoyancies.append(resistance['buoyancy_N_per_m'])
    
    return {
        'ages_s': ages_np,
        'ages_hours': ages_np / 3600.0,
        'bond_strength_Pa': np.array(bond_strengths),
        'total_resistance_N_per_m': np.array(total_resistances),
        'rubble_friction_N_per_m': np.array(rubble_frictions),
        'freeze_bond_N_per_m': np.array(freeze_bonds),
        'buoyancy_N_per_m': np.array(buoyancies),
    }


def run_freeze_bond_study(
    *,
    max_age_hours: float = 240.0,  # 10 days
    n_points: int = 100,
    temperature_K: float = 263.15,  # -10°C
    asymptotic_tensile_Pa: float = 1.0e6,
    asymptotic_shear_Pa: float = 2.0e6,
    growth_time_hours: float = 100.0,
    keel_depth_m: float = 10.0,
    keel_width_m: float = 20.0,
    porosity: float = 0.30,
    bond_area_fraction: float = 0.5,
) -> dict:
    """Run a complete freeze-bond evolution study.
    
    Returns
    -------
    dict with keys:
        - 'bond_evolution': bond strength vs. age
        - 'resistance_evolution': keel resistance vs. age
        - 'parameters': input parameters
        - 'summary': key metrics
    """
    # Build models
    formation = FreezeBondFormationCriterion(freezing_temperature_K=FREEZING_POINT_K)
    growth = ExponentialSaturationGrowth(
        asymptotic_tensile_strength_Pa=asymptotic_tensile_Pa,
        asymptotic_shear_strength_Pa=asymptotic_shear_Pa,
        growth_time_s=growth_time_hours * 3600.0,
    )
    bond_model = FreezeBondModel(formation_criterion=formation, growth_law=growth)
    
    ridge = LindqvistRidgeResistance(
        keel_depth_m=keel_depth_m,
        keel_width_m=keel_width_m,
        porosity=porosity,
    )
    
    # Time array
    max_age_s = max_age_hours * 3600.0
    ages_s = np.linspace(0.0, max_age_s, n_points)
    
    # Compute evolution
    bond_evo = compute_bond_strength_evolution(bond_model, ages_s, temperature_K)
    resistance_evo = compute_keel_resistance_evolution(
        ridge, bond_model, ages_s, temperature_K,
        bond_area_fraction=bond_area_fraction,
    )
    
    # Summary metrics
    final_bond_strength = bond_evo['tensile_strength_Pa'][-1]
    final_resistance = resistance_evo['total_resistance_N_per_m'][-1]
    initial_resistance = resistance_evo['total_resistance_N_per_m'][0]
    resistance_increase = (final_resistance - initial_resistance) / initial_resistance * 100
    
    summary = {
        'final_bond_strength_Pa': final_bond_strength,
        'final_resistance_N_per_m': final_resistance,
        'initial_resistance_N_per_m': initial_resistance,
        'resistance_increase_percent': resistance_increase,
        'strength_ratio_final': bond_evo['strength_ratio'][-1],
    }
    
    return {
        'bond_evolution': bond_evo,
        'resistance_evolution': resistance_evo,
        'parameters': {
            'max_age_hours': max_age_hours,
            'temperature_K': temperature_K,
            'asymptotic_tensile_Pa': asymptotic_tensile_Pa,
            'asymptotic_shear_Pa': asymptotic_shear_Pa,
            'growth_time_hours': growth_time_hours,
            'keel_depth_m': keel_depth_m,
            'keel_width_m': keel_width_m,
            'porosity': porosity,
            'bond_area_fraction': bond_area_fraction,
        },
        'summary': summary,
    }
