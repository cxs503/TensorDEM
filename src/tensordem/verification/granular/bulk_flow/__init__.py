"""P489-N04: Bulk granular flow verification surface.

Three-stage verification chain for TensorDEM in the bulk-solids flow domain:

1. **Beverloo discharge law** -- mass flow rate Q vs orifice diameter D follows
   Q ∝ (D − k·d)^2.5 with R² ≥ 0.95 and exponent 2.5 ± 0.2.
2. **Janssen wall-pressure saturation** -- vertical stress σ_v(z) in a deep silo
   saturates to P_∞ = ρ_b g R / (2 μ_w K); fitted P_∞ within 10 % of theory.
3. **Funnel → mass flow transition** -- critical hopper half-angle θ_c matches
   Jenike chart within 15 % (DEM friction coefficient must be declared with
   calibration uncertainty).

Stage 4 (energy ledger) is audited inline: per-timestep residual ≤ 1 %.

All cases use ≥ 5000 particles (statistical representativity) and report
steady-state quantities from the last 50 % of the measurement window.
"""
from .campaign import (
    CAMPAIGN_PLAN,
    run_beverloo_sweep,
    run_janssen_single,
    run_hopper_sweep,
    run_energy_audit,
    aggregate_results,
    generate_figures,
    write_report,
)

__all__ = [
    "CAMPAIGN_PLAN",
    "run_beverloo_sweep",
    "run_janssen_single",
    "run_hopper_sweep",
    "run_energy_audit",
    "aggregate_results",
    "generate_figures",
    "write_report",
]
