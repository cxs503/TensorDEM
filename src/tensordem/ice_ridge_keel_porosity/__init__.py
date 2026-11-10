"""Ice ridge keel porosity evolution under self-weight and wave-induced consolidation.

Physical background
-------------------
When an ice ridge forms from floe-floe compression, broken ice rubble accumulates
both above waterline (sail) and below waterline (keel). The keel is an inverted
conical/pyramidal accumulation of ice blocks. Immediately after formation, the
keel rubble is loosely packed with porosity n_0 ≈ 0.35–0.45 (random loose packing
of angular blocks).

Over time, the keel consolidates under:
1. **Self-weight**: gravitational settling drives particle rearrangement
2. **Wave loading**: cyclic hydrodynamic forces shake the rubble, promoting
   densification (analogous to vibratory compaction of granular media)
3. **Thermal stress**: freeze-thaw cycles cause micro-fracture and re-packing
4. **Freeze-bonding**: eventually, the rubble freezes into a solid mass

The porosity evolution follows an exponential decay:

    n_keel(t) = n_∞ + (n_0 - n_∞) · exp(-t / τ)

where:
- n_0 ≈ 0.4 is the initial (loose) porosity
- n_∞ ≈ 0.2 is the asymptotic (dense) porosity
- τ is the consolidation time scale (typically 10²–10⁴ s for DEM,
  corresponding to hours–months in prototype via scaling)

This module provides:
- :mod:`.geometry` — keel container geometry (inverted truncated pyramid)
- :mod:`.packing` — initial rubble generation (polydisperse spheres)
- :mod:`.consolidation` — DEM settling + wave-shaking simulation
- :mod:`.evolution` — porosity time-series and exponential decay fitting
- :mod:`.sensitivity` — parametric study of τ vs wave amplitude / gradation
- :mod:`.schema` — JSON output schema for downstream consumers

References
----------
- Hopky, V.A. (1983). "Physical characteristics of some pressure ridges"
- Timco, G.W. & Burden, R.P. (1994). "Mechanical properties of sea ice ridges"
- Lu, W., Lubbad, R., Løset, S. (2016). "Out-of-plane failure of an ice ridge"
- Hopkins, M.A. (2001). "The evolution of ice ridges"
"""
from __future__ import annotations

from .geometry import KeelGeometry, keel_volume, keel_containing_box
from .packing import (
    KeelPackingConfig,
    generate_rubble_packing,
    initial_porosity,
    polydisperse_radii,
)
from .consolidation import (
    ConsolidationConfig,
    ConsolidationResult,
    run_consolidation,
    compute_porosity_timeseries,
)
from .evolution import (
    PorosityEvolutionResult,
    exponential_decay,
    fit_exponential_decay,
    run_evolution_analysis,
)
from .sensitivity import (
    SensitivityConfig,
    SensitivityResult,
    run_sensitivity_study,
    tau_parameterization,
)
from .schema import build_output_json, OUTPUT_SCHEMA

__all__ = [
    # Geometry
    "KeelGeometry",
    "keel_volume",
    "keel_containing_box",
    # Packing
    "KeelPackingConfig",
    "generate_rubble_packing",
    "initial_porosity",
    "polydisperse_radii",
    # Consolidation
    "ConsolidationConfig",
    "ConsolidationResult",
    "run_consolidation",
    "compute_porosity_timeseries",
    # Evolution
    "PorosityEvolutionResult",
    "exponential_decay",
    "fit_exponential_decay",
    "run_evolution_analysis",
    # Sensitivity
    "SensitivityConfig",
    "SensitivityResult",
    "run_sensitivity_study",
    "tau_parameterization",
    # Schema
    "build_output_json",
    "OUTPUT_SCHEMA",
]
