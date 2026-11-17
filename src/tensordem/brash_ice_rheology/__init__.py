"""Brash-ice channel rheology and resistance closure (m24013 salvage, m24148).

A brash-ice channel is modelled as a confined granular bed sheared by the
hull: the closure (:mod:`.model`) splits the resistance into a shear
component (Jop-type ``mu(I)`` rheology, :mod:`.mu_i`), a buoyancy component
and a keel-drag component; :mod:`.material` carries the brash state
(porosity, concentration, temperature), :mod:`.coefficients` pins the
coefficient ledger, :mod:`.anchors` and :mod:`.validation` hold the
reference anchors and the acceptance criteria, :mod:`.calibration` fits the
confinement coefficient, :mod:`.fullscale_validation` runs the full-scale
three-way comparison (DEM vs Lindqvist vs anchor), and :mod:`.study` /
:mod:`.report` / :mod:`.figures` drive the sweeps and the report artifacts.

Salvage note (m24148): the ten implementation files were recovered from the
isolated m24013 worktree and are byte-identical to the committed copies;
this ``__init__.py`` is the packaging fix that makes the directory a real
subpackage so the capability registry (``tensordem.registry``) can see it.
"""
from __future__ import annotations

from .anchors import (
    anchor_components,
    level_ice_reference,
    reference_anchor,
    scaling_diagnostic,
)
from .calibration import calibrate_confinement, calibrate_family
from .coefficients import (
    COEFFICIENT_LEDGER,
    check_coefficient,
    coefficient_table,
    ledger_sha256,
)
from .material import STANDARD_BRASH, BrashIceState
from .model import (
    BrashHull,
    Closure,
    ResistanceComponents,
    buoyancy_component,
    keel_drag_component,
    shear_component,
    standard_hull,
)
from .mu_i import (
    JopParameters,
    LITERATURE_JOP,
    fit_sweep,
    inertial_number,
    mu_of_I,
    run_shear_sweep,
)
from .validation import B_TOLERANCE

__all__ = [
    "STANDARD_BRASH",
    "BrashIceState",
    "BrashHull",
    "Closure",
    "ResistanceComponents",
    "standard_hull",
    "shear_component",
    "buoyancy_component",
    "keel_drag_component",
    "JopParameters",
    "LITERATURE_JOP",
    "inertial_number",
    "mu_of_I",
    "run_shear_sweep",
    "fit_sweep",
    "COEFFICIENT_LEDGER",
    "check_coefficient",
    "coefficient_table",
    "ledger_sha256",
    "reference_anchor",
    "anchor_components",
    "level_ice_reference",
    "scaling_diagnostic",
    "calibrate_confinement",
    "calibrate_family",
    "B_TOLERANCE",
]
