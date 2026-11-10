"""TensorDEM coupling kernel: numpy-first DEM leg for the TensorPhysics chain.

Public surface
--------------
``DEMKernelConfig`` / ``DEMKernel``
    Contact-resolved soft-sphere DEM (Hertz + dashpot, velocity-Verlet) whose
    array arithmetic runs on a selectable backend: numpy (reference, default)
    or torch (optional accelerator).  Importing this package never imports
    torch.
``candidate_pairs``
    Deterministic numpy cell-linked-list detector (host side, shared by both
    backends).
``EnergyLedger``
    Itemised energy bookkeeping: kinetic, gravity potential, elastic contact
    storage, particle dashpot dissipation, wall dashpot dissipation, drag
    dissipation, plus the closure residual and attachable hooks.
``run_settling`` / ``dilute_single_particle_case`` / ``dense_bed_case``
    Settling scenarios that replace the analytic "reference model" DEM leg of
    the TensorPhysics ``coupling_energy`` scene with the real kernel.

See ``docs/COUPLING_KERNEL_DEM_BACKEND.md`` for the maturity report.
"""

from __future__ import annotations

from .backend import ArrayBackend, NumpyBackend, TorchBackend, available_backends, get_backend
from .celllist import brute_force_pairs, candidate_pairs, cell_geometry
from .kernel import DEMKernel, DEMKernelConfig, damping_from_restitution, hertz_prefactor
from .ledger import LEDGER_FIELDS, EnergyLedger
from .settling import (
    THREE_FIELD_CHANNELS,
    SettlingCase,
    dense_bed_case,
    dilute_single_particle_case,
    drag_relaxation_time,
    momentum_exchange,
    oseen_terminal_velocity,
    relaxation_reference,
    run_settling,
    stokes_terminal_velocity,
    three_field_summary,
)

__all__ = [
    "ArrayBackend",
    "NumpyBackend",
    "TorchBackend",
    "available_backends",
    "get_backend",
    "brute_force_pairs",
    "candidate_pairs",
    "cell_geometry",
    "DEMKernel",
    "DEMKernelConfig",
    "damping_from_restitution",
    "hertz_prefactor",
    "LEDGER_FIELDS",
    "EnergyLedger",
    "THREE_FIELD_CHANNELS",
    "SettlingCase",
    "dense_bed_case",
    "dilute_single_particle_case",
    "drag_relaxation_time",
    "momentum_exchange",
    "oseen_terminal_velocity",
    "relaxation_reference",
    "run_settling",
    "stokes_terminal_velocity",
    "three_field_summary",
]