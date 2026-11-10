"""Ice material constitutive models for TensorDEM.

This sub-package provides physics-based parameterizations of sea ice mechanical
properties as functions of temperature, salinity, and ice type (FYI vs MYI).

Modules
-------
brine_volume
    Cox & Weeks (1983) brine volume fraction nu_b = f(T, S).
compressive_strength
    Timco & Frederking (1984) FYI/MYI compressive strength sigma_c = f(T, S, nu_b).
uncertainty
    Monte Carlo uncertainty propagation for strength predictions.
visualization
    3D surface plots, MYI vs FYI comparison, uncertainty bands.

References
----------
- Cox, G. F. N. & Weeks, W. F. (1983). "Equations for determining the gas
  content and density of sea ice." CRREL Report 83-9.
- Timco, G. W. & Frederking, R. M. W. (1984). "The effect of grain size on
  the compressive strength of ice." IAHR Ice Symposium.
- Frankenstein, G. & Garner, R. (1967). "Equations for determining the brine
  volume of sea ice from -0.5 to -22.9°C." J. Glaciol. 7(50), 273-276.
- Timco, G. W. & Weeks, W. F. (2010). "A review of the engineering properties
  of sea ice in the twenty-first century." Cold Regions Science and Technology,
  60(1), 107-129.
"""

from .brine_volume import brine_volume_fraction, cox_weeks_nu_b
from .compressive_strength import (
    compressive_strength_fyi,
    compressive_strength_myi,
    pure_ice_strength,
)
from .uncertainty import monte_carlo_strength, strength_confidence_band

__all__ = [
    "brine_volume_fraction",
    "cox_weeks_nu_b",
    "compressive_strength_fyi",
    "compressive_strength_myi",
    "pure_ice_strength",
    "monte_carlo_strength",
    "strength_confidence_band",
]
