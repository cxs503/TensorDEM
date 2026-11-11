"""Ice-water slurry rheology: three-layer verification surface.

Physical goal
-------------
Scale bridging from discrete ice blocks to continuous equivalent fluid,
enabling ice-slurry pipeline transport, broken-ice-field equivalent-fluid
models, and ice-water two-phase-flow simplified models.

Three layers
------------
1. **Concentration-viscosity scaling**: Krieger-Dougherty and Eilers formulas
   with mutual-adjudication error bounds.
2. **Shear-rate dependence**: shear-thinning to shear-thickening transition
   detection in DEM Couette shear.
3. **Shape & temperature effects**: angular vs spherical particle viscosity
   ratio; Arrhenius-type temperature dependence of the carrier fluid.

Machine-verifiable criteria (campaign node p387w2-n09)
------------------------------------------------------
(a) Krieger-Dougherty mutual-adjudication error <= 30 %
(b) Eilers formula error <= 25 %
(c) Shear-thinning / shear-thickening transition detected
(d) Shape effect: angular / spherical viscosity ratio >= 1.2
(e) Mass conservation <= 1e-10
(f) Two negative controls (phi=0 -> eta_r=1; phi->phi_m -> divergence)
"""
from __future__ import annotations

from .closures import (
    IceSlurrySpec,
    krieger_dougherty,
    eilers,
    shape_factor_angularity,
    viscosity_temperature,
    cross_adjudicate,
)
from .dem_slurry import (
    IceSlurryCouette,
    run_couette_sweep,
)
from .verification import (
    face_a_kd,
    face_b_eilers,
    face_c_shear_transition,
    face_d_shape_effect,
    face_e_mass_conservation,
    face_f_negative_controls,
    run_all,
)

__all__ = [
    "IceSlurrySpec",
    "krieger_dougherty",
    "eilers",
    "shape_factor_angularity",
    "viscosity_temperature",
    "cross_adjudicate",
    "IceSlurryCouette",
    "run_couette_sweep",
    "face_a_kd",
    "face_b_eilers",
    "face_c_shear_transition",
    "face_d_shape_effect",
    "face_e_mass_conservation",
    "face_f_negative_controls",
    "run_all",
]
