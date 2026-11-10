"""Single-particle crushing verification module for TensorDEM.

This module implements three verification pillars for DEM particle breakage:

1. **Weibull strength–size effect** (``weibull_strength``):
   σ_f = σ_0 (V/V_0)^{-1/m},  P_f = 1 − exp[−(σ/σ_0)^m (V/V_0)]

2. **Fragment replacement with conservation ledger** (``fragment_replacement``):
   Mass: Σ m_i = m_parent
   Momentum: Σ m_i v_i = m_parent v_parent + Δp_fracture

3. **Gaudin–Schuhmann破碎粒度分布** (``gaudin_distribution``):
   P(d) = (d/d_max)^α

References
----------
- Weibull, W. (1951). A statistical distribution function of wide applicability.
- Gaudin, A.M. & Schuhmann, R. (1936). Empirical evaluation of the
  probability-size function in crushing.
- Astrom, J.A. & Herrmann, H.J. (1998). Fragmentation of documents under
  compression. Phys. Rev. E 57, 580.
"""
from __future__ import annotations

from .weibull_strength import (
    WeibullStrengthModel,
    weibull_failure_probability,
    weibull_strength_size,
    fit_weibull_modulus,
    generate_weibull_strengths,
)
from .fragment_replacement import (
    FragmentReplacement,
    ConservationLedger,
    replace_parent_with_fragments,
    generate_fragment_velocities,
)
from .gaudin_distribution import (
    GaudinSchuhmannDistribution,
    gaudin_cdf,
    gaudin_pdf,
    fit_gaudin_exponent,
    generate_gaudin_sizes,
)

__all__ = [
    "WeibullStrengthModel",
    "weibull_failure_probability",
    "weibull_strength_size",
    "fit_weibull_modulus",
    "generate_weibull_strengths",
    "FragmentReplacement",
    "ConservationLedger",
    "replace_parent_with_fragments",
    "generate_fragment_velocities",
    "GaudinSchuhmannDistribution",
    "gaudin_cdf",
    "gaudin_pdf",
    "fit_gaudin_exponent",
    "generate_gaudin_sizes",
]
