"""Clear-ice floe-field DEM with a regularised Hertz-Mindlin contact model.

This package is the delivery of campaign node **P103W2-N03** (m6997): it turns
the floe kinematics of the broken/clear-ice DEM that the earlier clearance
operator (P103W2B-N03) left **undeliverable** (numerical oscillation of the
contact force at the stable step size) into a usable output, and it adds the
statistics the campaign asks for (force-chain distribution, coordination
number, porosity) plus convergence / uncertainty evidence and a versioned
interface contract towards the level-ice breakup module.

Modules
-------
``contact``      Hertz-Mindlin pair law with Tsuji viscous damping, restitution
                 inversion, regularised Coulomb friction and pair history.
``field``        the clear-ice floe field: polydisperse disks on a buoyant
                 plane, doubly periodic channel with a rigid hull gate.
``statistics``   force-chain distribution ``P(f)``, coordination number ``Z``,
                 porosity (areal + periodic Voronoi free volume).
``kinematics``   the floe-kinematics deliverable and its quality metrics.
``contract``     ``clear-ice-contract/1.0`` interface to the level-ice breakup
                 module (`tensordem.level_ice`).
``study``        end-to-end study: baseline + convergence + UQ + artifact.

Evidence grade: the contact law is REAL (analytic Hertz verified in
``tests/test_clear_ice_contact.py``); the floe field is REAL DEM (no hard-coded
kinematics); the comparison against the IN12-S4 experiment is **WITHHELD**
because that dataset is not present in this repository -- see
``docs/CLEAR_ICE_DEM_REGULARIZATION.md``.
"""

from .contact import (
    ContactMaterial,
    ContactSolver,
    contact_time,
    damping_ratio_from_restitution,
    effective_modulus,
    effective_radius,
    restitution_from_damping_ratio,
)
from .contract import (
    CONTRACT_VERSION,
    ClearIceRequest,
    ClearIceResult,
    FloePopulation,
    validate_population,
)
from .field import ClearIceField, FieldConfig, RunOutput
from .kinematics import KinematicsSummary, kinematics_summary
from .statistics import (
    force_distribution,
    coordination_number,
    areal_porosity,
    voronoi_free_volume,
)

__all__ = [
    "CONTRACT_VERSION",
    "ClearIceField",
    "ClearIceRequest",
    "ClearIceResult",
    "ContactMaterial",
    "ContactSolver",
    "FieldConfig",
    "FloePopulation",
    "KinematicsSummary",
    "RunOutput",
    "areal_porosity",
    "contact_time",
    "coordination_number",
    "damping_ratio_from_restitution",
    "effective_modulus",
    "effective_radius",
    "force_distribution",
    "kinematics_summary",
    "restitution_from_damping_ratio",
    "validate_population",
    "voronoi_free_volume",
]