"""Dense two-dimensional granular / broken-ice mechanics on the H1 lattice data model.

Sub-package layout:

``engine``       rigid frictional disks (Hertz-Mindlin + rolling resistance) with
                 rigid wall / hull polyline boundaries; deterministic double precision.
``stress``       per-contact Virial / Love-Weber stress, lattice coarse graining and
                 the discrete stress-balance (Virial) identity.
``force_chain``  strong-force network extraction, chain length statistics, load
                 bearing ratio, percolation (gap spanning) tests, fabric tensor.
``jamming``      quasi-static packing preparation, jamming volume fraction protocol,
                 channel (ship) transit with drift drive, piecewise-linear + BIC
                 breakpoint detection.

Units are SI throughout (m, kg, s, N); 2D stress has units N/m.
"""

from .engine import (
    DiskConfig,
    DiskSystem,
    box_walls,
    polyline_segments,
    ship_segments,
)
from .force_chain import (
    analyze_force_network,
    chain_length_distribution,
    coordination_number,
    fabric_tensor,
    load_bearing_ratio,
    spanning_axis,
    strong_force_mask,
)
from .jamming import (
    isotropic_packing_at_pressure,
    BreakpointFit,
    bidisperse_radii,
    JammingResult,
    grow_to_fraction,
    make_channel_system,
    random_packing_positions,
    jamming_density,
    jamming_density_isotropic,
    piecewise_bic_breakpoint,
    run_channel,
)
from .stress import (
    contact_stress_table,
    hydrostatic_check,
    stress_invariants as stress_invariants,
    lattice_stress,
    stress_balance,
    stress_invariants,
    virial_tensor,
)

__all__ = [
    "DiskConfig",
    "DiskSystem",
    "box_walls",
    "polyline_segments",
    "ship_segments",
    "analyze_force_network",
    "chain_length_distribution",
    "coordination_number",
    "fabric_tensor",
    "load_bearing_ratio",
    "spanning_axis",
    "strong_force_mask",
    "BreakpointFit",
    "bidisperse_radii",
    "JammingResult",
    "isotropic_packing_at_pressure",
    "grow_to_fraction",
    "make_channel_system",
    "random_packing_positions",
    "jamming_density",
    "jamming_density_isotropic",
    "piecewise_bic_breakpoint",
    "run_channel",
    "contact_stress_table",
    "hydrostatic_check",
    "lattice_stress",
    "stress_balance",
    "stress_invariants",
    "virial_tensor",
]