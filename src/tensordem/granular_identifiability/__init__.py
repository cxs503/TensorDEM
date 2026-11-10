"""Parameter identifiability, inversion and uncertainty propagation for granular DEM contacts.

The face answers one question with reproducible numbers: *given a set of
measurable granular quantities, can the three contact parameters (friction
coefficient, restitution coefficient, contact stiffness) be recovered?*

Pieces
------
``system``
    ``test_granular_01`` -- the synthetic granular system (spec, packing,
    geometry, and a hard per-case compute budget).
``targets``
    The five forward observables (repose angle, packing fraction, silo
    discharge rate, drop-test restitution, oedometer stiffness) and the
    restitution -> damping-ratio mapping.
``inversion``
    Fisher information / condition number of the parameter set, weighted
    least-squares inversion with a recovery test against synthetic
    observations, and the +-10 % linear-response (uncertainty propagation) test.
``study``
    The acceptance study: forward targets -> discretisation ladder ->
    FIM conditioning -> inversion recovery -> propagation linearity, written
    out as a machine-readable artifact plus figures.

The machinery is domain neutral: no case in this package carries material
semantics beyond the specification above.
"""

from __future__ import annotations

from .system import (  # noqa: F401
    BudgetExceeded,
    CaseBudget,
    GranularSystemSpec,
    TEST_GRANULAR_01,
    build_drop_system,
    build_bed_system,
    bed_walls,
    clone_with_gravity,
    disk_config,
    hex_pack,
    run_for,
)
from .targets import (  # noqa: F401
    TARGET_NAMES,
    TARGET_SPECS,
    bed_packing_fraction,
    drop_observables,
    evaluate_targets,
    impact_stiffness,
    restitution_to_zeta,
    bed_observables,
    grain_observables,
    tilt_onset_deg,
    zeta_to_restitution,
    surface_profile,
    target_sigma,
    target_vector,
)
from .inversion import (  # noqa: F401
    PARAMETER_SPECS,
    ForwardMap,
    central_jacobian,
    conditioning,
    invert_parameters,
    params_from_unit,
    propagation_linearity,
    unit_from_params,
)

__all__ = [
    "BudgetExceeded",
    "CaseBudget",
    "GranularSystemSpec",
    "TEST_GRANULAR_01",
    "TARGET_NAMES",
    "TARGET_SPECS",
    "PARAMETER_SPECS",
    "ForwardMap",
    "build_drop_system",
    "impact_stiffness",
    "zeta_to_restitution",
    "central_jacobian",
    "conditioning",
    "disk_config",
    "evaluate_targets",
    "bed_packing_fraction",
    "bed_observables",
    "grain_observables",
    "tilt_onset_deg",
    "build_bed_system",
    "clone_with_gravity",
    "hex_pack",
    "invert_parameters",
    "params_from_unit",
    "propagation_linearity",
    "restitution_to_zeta",
    "run_for",
    "target_sigma",
    "target_vector",
    "unit_from_params",
]