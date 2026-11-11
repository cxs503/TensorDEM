"""Thermal-mechanical coupling of DEM ice-block collisions.

Why this package exists
-----------------------
The repository already owns the *isothermal* ice mechanics chain (contact
laws, crushing strength, breakage statistics) and two *thermal* chains (the
sliding-contact temperature of :mod:`tensordem.ice_friction_thermo` and the
conduction/phase-change substrate of :mod:`tensordem.interface_thermal`).  What
was missing is the middle of the impact problem: a normal collision does work on
the ice, ~90 % of that plastic work becomes heat in a thin patch under the
contact, the patch is restrained by the cold ice around it, and the resulting
thermal stress superposes on the mechanical contact stress against a
temperature dependent strength.  That chain -- collision -> plastic heat ->
conduction -> thermal stress -> additional breakage -- is implemented here, end
to end, with every link gradeable against an independent reference.

Module map
----------
``properties``      temperature-resolved ice properties (engineering fits inside
                    the tabulated bands) and the *delegated* strength surface
                    (``tensordem.ice_material``, Timco & Frederking 1984)
``plastic_heat``    Taylor-Quinney split of the dissipated impact work, Blok
                    effusivity partition, Hertz patch and deposition volume
``conduction``      FV theta-scheme conductor with volumetric source, an
                    independent nodal FE solver, and three closed-form
                    references (Carslaw & Jaeger flux solution, heated-slab
                    sine series, instantaneous plane/point sources)
``thermal_stress``  ``E alpha_T dT/(1-nu)`` in its three classical settings,
                    with the static-admissibility invariant
``fracture``        thermal + mechanical superposition against ``sigma_f(T)``
                    on the compression and tension branches, breakage increment
``collision``       the coupled driver (Hunt-Crossley impact -> heat -> field ->
                    fracture) and the ``(v, T)`` significance sweeps
``validation``      the acceptance ladder C1..C12, with digest

Headline result (see ``docs/ICE_THERMAL_STRESS_COUPLING.md``)
--------------------------------------------------------------
For a 0.6 m ice block (R = 0.3 m) hitting a rigid structure at 1 m/s in
-10 degC ice: patch heating ~13 K, thermal stress ~10 MPa = 42 % of the ice
strength but only ~5 % of the peak Hertz contact pressure, and an *additional*
breakage fraction of ~1.5 % once the contact pressure is capped at the crushing
strength.  Thermal stress therefore matters as a strength-budget term and is
second order as a contact-stress term; above ~1.5-2 m/s the patch would reach
the melting point, where this model stops (no latent heat) -- that boundary is
reported, not hidden.

Typical use
-----------
>>> from tensordem.ice_thermal_stress import BlockImpactCase, simulate_block_impact
>>> res = simulate_block_impact(BlockImpactCase())      # doctest: +SKIP
>>> res.peak_delta_t_k > 0.0                             # doctest: +SKIP
True
"""

from .collision import (
    BlockImpactCase,
    BlockImpactResult,
    layer_depth_sweep,
    significance_sweep,
    simulate_block_impact,
)
from .conduction import (
    SlabHeat1D,
    exact_plane_source,
    exact_point_source_sphere,
    exact_semi_infinite_flux,
    exact_slab_uniform_source,
    observed_order,
    slab_fem_nodes,
)
from .fracture import (
    RATIO_COMPRESSION_TO_TENSION,
    FractureState,
    breakage_increment_curve,
    critical_heating_for_crushing,
    failure_margin,
    superpose,
    tensile_strength,
    thermal_breakage_increment,
    thermal_margin_fraction,
)
from .plastic_heat import (
    TAYLOR_QUINNEY_BETA,
    TAYLOR_QUINNEY_RANGE,
    ImpactHeatBudget,
    adiabatic_temperature_rise,
    deposition_case,
    deposition_volume,
    dissipated_energy,
    hertz_patch_radius,
    layer_depth_sensitivity,
    partition_to_ice,
)
from .properties import (
    ANCHORS,
    IceProperties,
    NU_ICE,
    TEMPERATURE_BAND_C,
    compressive_strength,
    ice_properties,
    thermal_stress_coefficient,
)
from .thermal_stress import (
    free_slab_gradient_peak,
    field_stress_from_temperature,
    restrained_gradient_stress,
    restrained_uniform_stress,
    restrained_vs_free_ratio,
    stress_equilibrium_residual,
    thermal_shock_critical_delta_t,
)

__all__ = [
    # properties
    "IceProperties", "ice_properties", "compressive_strength",
    "thermal_stress_coefficient", "NU_ICE", "TEMPERATURE_BAND_C", "ANCHORS",
    # plastic heat
    "TAYLOR_QUINNEY_BETA", "TAYLOR_QUINNEY_RANGE", "ImpactHeatBudget",
    "dissipated_energy", "hertz_patch_radius", "deposition_volume",
    "partition_to_ice", "adiabatic_temperature_rise", "deposition_case",
    "layer_depth_sensitivity",
    # conduction
    "SlabHeat1D", "slab_fem_nodes", "exact_semi_infinite_flux",
    "exact_slab_uniform_source", "exact_plane_source",
    "exact_point_source_sphere", "observed_order",
    # thermal stress
    "restrained_uniform_stress", "restrained_gradient_stress",
    "free_slab_gradient_peak", "thermal_shock_critical_delta_t",
    "stress_equilibrium_residual", "restrained_vs_free_ratio",
    "field_stress_from_temperature",
    # fracture
    "RATIO_COMPRESSION_TO_TENSION", "FractureState", "superpose",
    "tensile_strength", "failure_margin", "thermal_margin_fraction",
    "thermal_breakage_increment", "breakage_increment_curve",
    "critical_heating_for_crushing",
    # collision
    "BlockImpactCase", "BlockImpactResult", "simulate_block_impact",
    "significance_sweep", "layer_depth_sweep",
]

__version__ = "0.1.0"