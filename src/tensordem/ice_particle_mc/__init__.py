"""Ice-particle contact mechanics and Mohr-Coulomb bond-rupture verification.

Public surface
--------------
``bond``     Mohr-Coulomb bond criterion, volume-consistent bond area, affine
             lattice modulus predictor (the micro-to-macro map).
``packing``  jittered cylinder specimen (D = 50 mm, H = 100 mm by default) and
             ice-layer packings, with geometry validation.
``engine``   compact 3-D bonded/Hertzian DEM with a closed energy ledger and
             kinematic grips.
``study``    the four verification drivers (uniaxial, indentation/pressure-area,
             ice resistance vs friction, force-chain statistics).
``analysis`` estimators (linear modulus, exponential force tail, power-law fit).

Scope: this is an isothermal, rate-independent bonded-particle model.  Real ice
is rate- and temperature-dependent; that is declared as a limitation, not
modelled here (see docs/ICE_PARTICLE_MC_VERIFICATION.md).
"""
from __future__ import annotations

from .analysis import (exponential_tail_fit, fit_linear_modulus, fit_power_law,
                       pressure_area_exponent, relative_error)
from .bond import (MohrCoulombStrength, affine_stiffness_matrix,
                   affine_uniaxial_modulus, bond_areas, bond_failure_mask,
                   bond_shear_capacity, mohr_coulomb_critical_normal_fraction,
                   mohr_coulomb_ucs, weibull_scale_from_median)
from .engine import AssemblyConfig, IceParticleAssembly, PlaneWall, candidate_pairs
from .packing import (Specimen, bond_pairs, coordination_number, cylinder_specimen,
                      ice_layer, validate_cylinder)
from .study import (IndentationSpec, ResistanceSpec, UniaxialSpec,
                    affine_modulus_prediction, force_chain_stats, run_ice_resistance,
                    run_indentation, run_uniaxial)

__all__ = [
    "MohrCoulombStrength", "AssemblyConfig", "IceParticleAssembly", "PlaneWall",
    "Specimen", "UniaxialSpec", "IndentationSpec", "ResistanceSpec",
    "cylinder_specimen", "ice_layer", "bond_pairs", "coordination_number",
    "validate_cylinder", "candidate_pairs", "affine_stiffness_matrix",
    "affine_uniaxial_modulus", "affine_modulus_prediction", "bond_areas",
    "bond_shear_capacity", "bond_failure_mask", "mohr_coulomb_ucs",
    "mohr_coulomb_critical_normal_fraction", "weibull_scale_from_median",
    "run_uniaxial", "run_indentation", "run_ice_resistance", "force_chain_stats",
    "fit_linear_modulus", "exponential_tail_fit", "fit_power_law",
    "pressure_area_exponent", "relative_error",
]