"""Ship-ice collision energy-conservation surface (structural side).

This package provides the *verifiable* structural half of an ice-region
ship-ice collision energy balance:

* :mod:`tensordem.hull_collision.plastic_hinge` - rigid-plastic hinge mechanics
  (``E = M_p*theta``) with an independent numerical anchor.
* :mod:`tensordem.hull_collision.resistance` - stiffened-plating resistance
  (hinge + membrane), serial double-hull energy allocation, penetration solve
  from energy conservation, and the damage geometry handed to SOLAS input.
* :mod:`tensordem.hull_collision.minorsky` - Minorsky (1959) empirical anchor
  and the yield-scaled cross-check.

All quantities are SI (m, N, J, Pa, kg).  Nothing here is calibrated to the
verification cases; the design case data lives in
``scripts/benchmark_hull_collision_energy.py`` and
``prereg/IN63_N01_hull_collision_energy.json``.
"""
from .plastic_hinge import (
    SUPPORTED_MECHANISMS,
    collapse_load,
    hinge_energy,
    hinge_rotations,
    plastic_moment,
    plastic_moment_rect_fiber,
    plastic_section_modulus_rect,
    section_plastic_modulus,
    three_point_bending_anchor,
)
from .resistance import (
    CapacityExhausted,
    DamageGeometry,
    DoubleHullSide,
    PenetrationSolution,
    SideShellLayer,
    damage_geometry,
    solve_penetration,
    verify_layer_decomposition,
)
from .minorsky import (
    MINORSKY_1959_INTERCEPT_MJ,
    MINORSKY_1959_MJ_PER_M3,
    REFERENCE_YIELD_STRESS_PA,
    compare_damage_volume,
    kinetic_energy,
    minorsky_coefficient,
    minorsky_damage_volume,
    minorsky_energy_J,
)

__all__ = [
    "SUPPORTED_MECHANISMS",
    "collapse_load",
    "hinge_energy",
    "hinge_rotations",
    "plastic_moment",
    "plastic_moment_rect_fiber",
    "plastic_section_modulus_rect",
    "section_plastic_modulus",
    "three_point_bending_anchor",
    "CapacityExhausted",
    "DamageGeometry",
    "DoubleHullSide",
    "PenetrationSolution",
    "SideShellLayer",
    "damage_geometry",
    "solve_penetration",
    "verify_layer_decomposition",
    "MINORSKY_1959_INTERCEPT_MJ",
    "MINORSKY_1959_MJ_PER_M3",
    "REFERENCE_YIELD_STRESS_PA",
    "compare_damage_volume",
    "kinetic_energy",
    "minorsky_coefficient",
    "minorsky_damage_volume",
    "minorsky_energy_J",
]


# --- p256w1-N04: strain hardening + ISO 19906 ice pressure indentation --------
from .strain_hardening import (  # noqa: E402
    DEFAULT_ROTATION_CAPACITY,
    ELASTIC_MODULUS_STEEL_PA,
    BilinearHinge,
    HardenedLayer,
    HardenedSide,
    PowerLawHinge,
    bilinear_moment_increment_closed_form,
    build_hardened_side,
    energy_increment_numeric,
    moment_increment_fibre,
    section_moment_per_width,
)
from .indentation import (  # noqa: E402
    ISO_PRESSURE_AREA_EXPONENTS,
    IndentationSolution,
    PressureAreaLaw,
    damage_geometry_pressure_driven,
    force_history,
    iso19906_scaling_exponents,
    solve_indentation,
    structural_coefficients_per_length,
)
from .recalibration import (  # noqa: E402
    DATASET_STATUS_PRESENT,
    DATASET_STATUS_WITHHELD,
    coefficient_uncertainty,
    collision_coefficient_from_share,
    collision_coefficient_from_volume,
    energy_ledger,
    load_collision_dataset,
    recalibrate_coefficient,
)

__all__ += [
    "DEFAULT_ROTATION_CAPACITY",
    "ELASTIC_MODULUS_STEEL_PA",
    "BilinearHinge",
    "HardenedLayer",
    "HardenedSide",
    "PowerLawHinge",
    "bilinear_moment_increment_closed_form",
    "build_hardened_side",
    "energy_increment_numeric",
    "moment_increment_fibre",
    "section_moment_per_width",
    "ISO_PRESSURE_AREA_EXPONENTS",
    "IndentationSolution",
    "PressureAreaLaw",
    "damage_geometry_pressure_driven",
    "force_history",
    "iso19906_scaling_exponents",
    "solve_indentation",
    "structural_coefficients_per_length",
    "DATASET_STATUS_PRESENT",
    "DATASET_STATUS_WITHHELD",
    "coefficient_uncertainty",
    "collision_coefficient_from_share",
    "collision_coefficient_from_volume",
    "energy_ledger",
    "load_collision_dataset",
    "recalibrate_coefficient",
]
