"""Acoustic Emission (AE) proxy module for bonded-DEM ice fracture.

This module establishes a falsifiable relationship between DEM bond-break
events and AE observables: event rate λ(t), amplitude A, and b-value
(Gutenberg-Richter / Aki maximum likelihood).

Physical model:
    - Each bond break releases stored elastic energy E_rel
    - AE radiated energy: E_AE = η · E_rel (η = radiation efficiency)
    - Amplitude: A ∝ sqrt(E_rel) / r (P-wave geometric spreading)
    - Statistics: λ(t) = event rate, b-value from Aki (1965) MLE

Submodules:
    events       - Bond break event tracking and energy release
    amplitude    - AE amplitude model with geometric spreading
    statistics   - Event rate, b-value, bootstrap confidence intervals
    indentation  - Indentation test driver for AE calibration
    controls     - Negative control tests
"""
from .events import AEEvent, AEEventCatalog, single_bond_energy_release
from .amplitude import ae_amplitude, geometric_spreading
from .statistics import event_rate, aki_bvalue_mle, bootstrap_bvalue_ci
from .controls import (
    elastic_no_fracture_control,
    single_bond_monotonic_control,
    synchronous_failure_control,
)

__all__ = [
    "AEEvent", "AEEventCatalog", "single_bond_energy_release",
    "ae_amplitude", "geometric_spreading",
    "event_rate", "aki_bvalue_mle", "bootstrap_bvalue_ci",
    "elastic_no_fracture_control",
    "single_bond_monotonic_control",
    "synchronous_failure_control",
]
