"""Broken-ice field direct-navigation resistance: DEM validation surface.

What this package is
--------------------
A *validation surface* (not a new solver): it drives the repository's plan-view
floe DEM (:class:`tensordem.ice_channel.IceChannelSimulation`) through a
ship-scale transit of a broken-ice field and compares the steady ice resistance
with

* the in-repository Lindqvist three-component anchor
  (:func:`tensordem.level_ice.formula.lindqvist_components`),
* the MT Uikku full-scale model-test table shipped with the repository
  (``benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv``), and
* the in-repository broken-ice closed-form anchor
  (:func:`tensordem.ice_resistance_scaling.broken_ice_anchor`).

Two model extensions are added on top of ``ice_channel`` (both are declared,
hashed into the ledger and A/B-tested against the baseline law):

``hull``     a *ship-formed* convex outline (parallel body + wedge bow of a
             stated entry angle) replaces the rectangular block, so the ice is
             deflected by an inclined bow rather than pushed broadside;
``contact``  a *crushing-limited* normal law ``F_n = sigma_c h w(delta)`` with
             the Hertzian chord width ``w = 2 sqrt(2 R_eff delta)`` replaces the
             linear penalty spring ``F_n = k_n delta``.  The linear spring is
             calibrated at ``delta = 1 % (2R)`` and therefore over-stiffens once
             the penetration grows, which is exactly the 3x ship-scale
             over-prediction reported for the baseline law in
             ``docs/ice_resistance_scaling.md``.

Public surface
--------------
``ledger``      graded parameter ledger + standard case + SHA-256 provenance
``hull``        ship outline, signed-distance contact query
``contact``     :class:`BrokenIceChannel` (crushing law, polygon hull) and
                :class:`PenaltyIceChannel` (baseline: linear spring, block hull)
``simulate``    transit runner, steady-window statistics, dissipation audit
``anchors``     Lindqvist / MT Uikku / broken-ice closed-form references
``validation``  criteria C1-C7, sweeps, verdicts, artifacts, figures
``report``      markdown report builder

All quantities are SI (m, Pa, kg/m^3, m/s, N); reported resistance is in kN.
"""
from .ledger import (CASE_SPECS, COEFFICIENT_LEDGER, MATERIAL, STANDARD_CASE,
                     CaseSpec, MaterialState, ledger_sha256)
from .hull import ShipForm
from .contact import BrokenIceChannel, PenaltyIceChannel
from .simulate import CaseResult, run_case
from .anchors import (broken_ice_closed_form, lindqvist_reference,
                      mt_uikku_level_ice_cases)
from .validation import evaluate_criteria, run_campaign

__all__ = [
    "CASE_SPECS", "COEFFICIENT_LEDGER", "MATERIAL", "STANDARD_CASE",
    "CaseSpec", "MaterialState", "ledger_sha256",
    "ShipForm", "BrokenIceChannel", "PenaltyIceChannel",
    "CaseResult", "run_case",
    "broken_ice_closed_form", "lindqvist_reference", "mt_uikku_level_ice_cases",
    "evaluate_criteria", "run_campaign",
]