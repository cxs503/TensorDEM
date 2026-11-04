"""PyTorch bonded-particle discrete element ice-breaking simulation."""

from .dem import DEMConfig, IceDEM
from .dem3d import DEM3DConfig, IceDEM3D
from .buoyancy_wet import (
    WetCouplingConfig,
    WaveForceConfig,
    compute_buoyancy_force,
    compute_wet_drag_force,
    compute_wet_forces,
    compute_wave_force,
    compute_all_wet_forces,
    submerged_fraction_tensor,
    submerged_fraction_sphere_tensor,
    wetted_area,
    wetted_area_sphere,
    wetted_perimeter,
    wetted_area_tensor,
    wetted_area_sphere_tensor,
    wave_kinematics,
    wave_elevation,
    solve_floating_equilibrium,
    heave_restoring_stiffness,
    metacentric_height,
    verify_buoyancy_force,
)
from .contact_tangential import (
    TangentialContactConfig,
    advance_tangential_contact,
    compute_tangential_force,
    integrate_tangential_displacement,
    project_tangential_displacement,
    reset_tangential_displacement,
    restitution_to_damping,
    damping_to_restitution,
    saturation_displacement,
    tangential_cycle_dissipated_work,
)
from .contact_impact import (
    ImpactAnalytic,
    PairImpactResult,
    clamped_restitution,
    impact_analytic,
    pair_impact,
    restitution_to_damping_clamped,
    rk4_clamped_restitution,
    unclamped_restitution,
)
from .contact_restitution import (
    clamped_restitution_ratio,
    classic_restitution_ratio,
    damping_for_target_restitution,
    measure_dem3d_impact,
)
from .restitution import (
    RestitutionMeasurement,
    clamped_contact_restitution,
    contact_restitution,
    damping_for_clamped_restitution,
    damping_for_restitution,
    damping_ratio,
    measure_platen_restitution,
    platen_restitution,
    restitution_from_damping,
    restitution_from_damping_clamped,
)
from .dem_export import DemExport, Fragment, save_json, load_json, load_to_tensors

__all__ = [
    "RestitutionMeasurement",
    "damping_ratio", "restitution_from_damping", "restitution_from_damping_clamped",
    "damping_for_restitution", "damping_for_clamped_restitution",
    "contact_restitution", "clamped_contact_restitution", "platen_restitution",
    "measure_platen_restitution",
    "DEMConfig", "IceDEM", "DEM3DConfig", "IceDEM3D",
    "WetCouplingConfig", "WaveForceConfig", "compute_buoyancy_force", "compute_wet_drag_force",
    "compute_wet_forces", "compute_wave_force", "compute_all_wet_forces", "submerged_fraction_tensor",
    "submerged_fraction_sphere_tensor", "wetted_area", "wetted_area_sphere", "wetted_perimeter",
    "wetted_area_tensor", "wetted_area_sphere_tensor", "wave_kinematics", "wave_elevation",
    "solve_floating_equilibrium", "heave_restoring_stiffness", "metacentric_height", "verify_buoyancy_force",
    "TangentialContactConfig", "compute_tangential_force", "integrate_tangential_displacement", "restitution_to_damping",
    "damping_to_restitution", "advance_tangential_contact", "project_tangential_displacement", "reset_tangential_displacement",
    "saturation_displacement", "ImpactAnalytic", "PairImpactResult", "impact_analytic",
    "pair_impact", "unclamped_restitution", "clamped_restitution", "rk4_clamped_restitution",
    "restitution_to_damping_clamped", "DemExport", "Fragment", "save_json",
    "load_json", "load_to_tensors", "tangential_cycle_dissipated_work", "classic_restitution_ratio",
    "clamped_restitution_ratio", "damping_for_target_restitution", "measure_dem3d_impact",
]
