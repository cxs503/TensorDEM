"""Linear elastic fracture mechanics (LEFM) toolkit for bonded-particle lattices.

Modules
-------
``lattice2d``       2-D bonded-particle spring lattice (isotropic at nu = 1/4)
``jintegral``       contour J-integral, equivalent-domain (EDI) J and the
                    interaction (M) integral for K_I / K_II
``crack_path``      maximum tangential stress criterion, propagation, path metric
``senb``            single-edge-notched three-point-bend (SEN B) driver:
                    critical-load K_IC calibration, negative controls, energy ledger
"""
from .lattice2d import BondedLattice2D, Lattice2DConfig, ISOTROPIC_POISSON
from .jintegral import (contour_j, domain_j, interaction_integral, stress_intensity_from_j,
                        stress_intensity_from_m, AuxiliaryField)
from .crack_path import (mts_angle, max_hoop_stress, trace_crack_path, path_overlap,
                         polyline_distance, propagate, hoop_stress_ring,
                         mts_angle_from_field, bonds_crossing)
from .senb import (SENBGeometry, build_senb, astm_stress_intensity,
                   astm_senb_shape_factor, bond_stress_intensity, tip_bond_intensity,
                   run_senb_elastic, calibrate_kic, uniform_tension_control,
                   critical_break_ledger, energy_release_rate_from_crack_advance)

__all__ = [
    "BondedLattice2D",
    "Lattice2DConfig",
    "ISOTROPIC_POISSON",
    "contour_j",
    "domain_j",
    "interaction_integral",
    "stress_intensity_from_j",
    "mts_angle",
    "trace_crack_path",
    "path_overlap",
    "polyline_distance",
    "max_hoop_stress",
    "bonds_crossing",
    "propagate",
    "hoop_stress_ring",
    "mts_angle_from_field",
    "SENBGeometry",
    "run_senb_elastic",
    "calibrate_kic",
    "uniform_tension_control",
    "build_senb",
    "astm_stress_intensity",
    "astm_senb_shape_factor",
    "bond_stress_intensity",
    "tip_bond_intensity",
    "critical_break_ledger",
    "energy_release_rate_from_crack_advance",
]
