"""Ridge-keel grounding on a buried subsea pipeline (campaign node p54w3-n06).

Domain directory of the node: it owns the *adjudication* layer of the
ice-ridge / pipeline / seabed problem - the case definitions of the campaign,
the ISO 19906-form comparison, the cover-instability consistency check, the
grid-refinement study and the energy-ledger closure.  The physics kernels stay
where they are (:mod:`tensordem.ridge_dem`, :mod:`tensordem.ridge_geotech`,
:mod:`tensordem.ridge_grounding`, :mod:`tensordem.ridge_iso19906`).

Public entry points::

    from tensordem.ridge_pipeline_dem import PipelineCase, run_case, grid_study
    from tensordem.ridge_pipeline_dem import closure_from_track, evaluate_acceptance

``scripts/p54w3_n06_campaign.py`` runs the full campaign and writes
``artifacts/p54w3_n06/campaign.json`` plus the figures of the report.
"""
from .campaign import (
    ACCEPTANCE_THRESHOLDS,
    PIPELINE_CASES,
    PIPE_DIAMETER,
    PIPE_MODULUS,
    PIPE_WALL,
    PIPE_YIELD,
    PipelineCase,
    case_scenario,
    cover_study,
    dt_refinement_cases,
    energy_study,
    evaluate_acceptance,
    grid_study,
    lateral_scenario_gap,
    lateral_study,
    pipe_section,
    run_case,
    rubble_material,
    undrained_clay,
    vertical_study,
)
from .energy import PARTITIONS, closure_from_track, closure_scan, dt_refinement_table

__all__ = [
    "ACCEPTANCE_THRESHOLDS",
    "PARTITIONS",
    "PIPELINE_CASES",
    "PIPE_DIAMETER",
    "PIPE_MODULUS",
    "PIPE_WALL",
    "PIPE_YIELD",
    "PipelineCase",
    "case_scenario",
    "closure_from_track",
    "closure_scan",
    "cover_study",
    "dt_refinement_cases",
    "dt_refinement_table",
    "energy_study",
    "evaluate_acceptance",
    "grid_study",
    "lateral_scenario_gap",
    "lateral_study",
    "pipe_section",
    "run_case",
    "rubble_material",
    "undrained_clay",
    "vertical_study",
]