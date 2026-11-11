"""Comminution / tumbling-mill energy surface for TensorDEM.

This package adds the *device-level* comminution face to TensorDEM: a ball mill
is a rotating drum filled with grinding media, and the four anchors that make
that device adjudicable are

1. **critical speed** -- the speed at which the charge centrifuges
   (:mod:`tensordem.comminution.critical_speed`),
2. **mill power** -- wall torque times angular velocity, with a momentum and an
   energy ledger (:mod:`tensordem.comminution.mill_dem`,
   :mod:`tensordem.comminution.mill_power`),
3. **Bond specific energy** -- the third theory of comminution
   (:mod:`tensordem.comminution.bond`),
4. **breakage kinetics** -- selection / breakage functions and the resulting
   size-distribution evolution
   (:mod:`tensordem.comminution.breakage_kinetics`).

The DEM kernel here is a self-contained 2-D soft-sphere mill (polydisperse
disks, rotating circular shell **plus lifter bars**, Cundall-Strack tangential
history, rolling resistance) that reports an explicit power ledger
(``P_in = tau * omega`` vs contact dissipation) and the collision energy
spectrum that the population balance consumes.  It deliberately does not modify
:mod:`tensordem.drum_flow` (smooth-wall drum regime maps), which stays the
smooth-wall reference.
"""
from __future__ import annotations

from .bond import (
    BOND_WORK_INDEX_DB,
    bond_product_size_um,
    bond_specific_energy_kwh_t,
    bond_work_index_from_energy,
    bond_work_index_from_grindability,
    specific_energy_kwh_t,
)
from .breakage_kinetics import (
    AppearanceFunction,
    austin_selection_rates,
    batch_grind,
    cumulative_passing,
    p80_from_psd,
    selection_rates_from_collisions,
    t10_energy_law,
)
from .critical_speed import (
    CRITICAL_SPEED_CONSTANT,
    critical_speed_constant_from_first_principles,
    critical_speed_rpm,
    critical_speed_rpm_charge_centre,
    fraction_of_critical,
    rpm_for_fraction_of_critical,
)
from .mill_dem import (
    MillConfig,
    MillState,
    make_mill_state,
    measure_centrifuging_onset_rpm,
    run_mill,
)
from .mill_power import (
    charge_bulk_density,
    hogg_fuerstenau_power_kw,
    mill_power_density_kw_m3,
    specific_power_kwh_t,
    torque_factor,
)

__all__ = [
    "BOND_WORK_INDEX_DB",
    "CRITICAL_SPEED_CONSTANT",
    "AppearanceFunction",
    "MillConfig",
    "MillState",
    "austin_selection_rates",
    "batch_grind",
    "bond_product_size_um",
    "bond_specific_energy_kwh_t",
    "bond_work_index_from_energy",
    "bond_work_index_from_grindability",
    "charge_bulk_density",
    "critical_speed_constant_from_first_principles",
    "critical_speed_rpm",
    "critical_speed_rpm_charge_centre",
    "cumulative_passing",
    "fraction_of_critical",
    "hogg_fuerstenau_power_kw",
    "make_mill_state",
    "measure_centrifuging_onset_rpm",
    "mill_power_density_kw_m3",
    "p80_from_psd",
    "rpm_for_fraction_of_critical",
    "run_mill",
    "selection_rates_from_collisions",
    "specific_energy_kwh_t",
    "specific_power_kwh_t",
    "t10_energy_law",
    "torque_factor",
]