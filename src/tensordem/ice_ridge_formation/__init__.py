"""Ice ridge formation dynamics: floe-floe compression to sail/keel emergence.

This module implements the physics of ice ridge formation from the convergence
of polygonal ice floes. The simulation captures:

1. Floe population initialization with power-law size distribution
2. Boundary-driven compression (two walls converging)
3. Floe-floe interaction with bending failure
4. Ice ridge emergence (sail above water, keel below)
5. Volume conservation tracking
6. Porosity scaling law: n = n0 * exp(-epsilon / epsilon_c)

Physical background
-------------------
When an ice field converges (e.g., due to wind or currents), floating ice floes
are compressed against each other. The compressive forces cause:

- **Rafting**: One floe overrides another (low convergence rates)
- **Rubbling**: Floes break and pile up (higher convergence rates)
- **Ridge formation**: Broken ice accumulates into a sail (above water) and
  keel (below water)

The sail/keel geometry follows empirical scaling laws (Hopky 1983):
    H_sail = 0.4 * sqrt(h * d_squeeze)
    D_keel = 4 * H_sail

where h is the parent ice thickness and d_squeeze is the convergence distance.

Volume conservation
-------------------
The total ice volume must be conserved:
    V_initial = V_final = V_sail + V_keel + V_parent_sheets + V_loss

where V_loss < 1% of V_initial.

Porosity scaling
----------------
The porosity n (void fraction) in the ridge decreases with compressive strain:
    n = n0 * exp(-epsilon / epsilon_c)

where n0 is the initial porosity, epsilon is the compressive strain, and
epsilon_c is a characteristic strain (typically 0.1-0.3).

References
----------
- Hopky, V.A. & Lawrence, D.S. (1983). "Physical characteristics of ice ridges"
- Timco, G.W. & Burden, R.P. (1994). "A review of the mechanical properties
  of sea ice ridges"
- Lu, W., Lubbad, R., & Løset, S. (2016). "Out-of-plane failure of an ice
  ridge: Kick-out mechanism"
"""

from .dynamics import (
    IceRidgeFormationConfig,
    IceRidgeFormationResult,
    simulate_ice_ridge_formation,
    compute_volume_conservation,
    fit_porosity_scaling,
    analyze_ridge_geometry,
)
from .floes import (
    PolygonalFloe,
    FloePopulation,
    generate_power_law_floes,
    FloePopulationConfig,
)
from .compression import (
    CompressionDriver,
    CompressionConfig,
    drive_compression,
)
from .failure import (
    BendingFailureModel,
    BendingFailureConfig,
    check_bending_failure,
)

__all__ = [
    # Dynamics
    "IceRidgeFormationConfig",
    "IceRidgeFormationResult",
    "simulate_ice_ridge_formation",
    "compute_volume_conservation",
    "fit_porosity_scaling",
    "analyze_ridge_geometry",
    # Floes
    "PolygonalFloe",
    "FloePopulation",
    "generate_power_law_floes",
    "FloePopulationConfig",
    # Compression
    "CompressionDriver",
    "CompressionConfig",
    "drive_compression",
    # Failure
    "BendingFailureModel",
    "BendingFailureConfig",
    "check_bending_failure",
]
