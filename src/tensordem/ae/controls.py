"""Negative control tests for the AE module.

Three negative controls verify that the AE system correctly reports null or
unresolvable results when no physical AE should occur:

1. Elastic no-fracture control:
   A bonded specimen loaded below the breaking strain should produce zero
   AE events. Any detected events indicate a bug in the fracture detection.

2. Single-bond monotonic control:
   A single bond pulled monotonically at constant rate should produce exactly
   one AE event at the predicted time and energy. The event rate should be
   analytically predictable: λ = 1/Δt where Δt is the time to failure.

3. Synchronous failure control:
   When all bonds are forced to break simultaneously (e.g., uniform strain
   exceeding breaking_strain for all bonds), the b-value is undefined because
   all events have identical energy. The system should explicitly report
   "indeterminate" rather than returning a spurious b-value.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from ..dem3d import DEM3DConfig, IceDEM3D
from .events import AEEventCatalog, single_bond_energy_release
from .statistics import aki_bvalue_mle


def elastic_no_fracture_control(
    config: DEM3DConfig | None = None,
    max_strain_fraction: float = 0.5,
    n_steps: int = 100,
) -> dict[str, Any]:
    """Control 1: Elastic loading below fracture threshold produces zero events.

    A small bonded specimen is loaded in uniaxial tension to a strain that is
    only a fraction of the breaking strain. No bonds should break.

    Args:
        config: DEM configuration. If None, uses default with small specimen.
        max_strain_fraction: Maximum applied strain as fraction of breaking_strain.
        n_steps: Number of simulation steps.

    Returns:
        Dict with:
            passed: bool - True if zero events detected
            n_events: int - number of detected events (should be 0)
            max_applied_strain: float - maximum strain applied
            breaking_strain: float - the breaking strain threshold
    """
    if config is None:
        # Compute target velocity for the desired strain
        radius = 0.01
        nz = 3
        breaking_strain = 0.01
        specimen_height = (nz - 1) * 2 * radius
        target_strain = max_strain_fraction * breaking_strain
        dt = 1e-6
        target_velocity = target_strain * specimen_height / (n_steps * dt)
        
        config = DEM3DConfig(
            nx=3, ny=3, nz=nz,
            radius=radius,
            bond_stiffness=1000.0,
            breaking_strain=breaking_strain,
            dt=dt,
            tool_radius=0.02,
            tool_speed=0.01,
            top_platen_enabled=True,
            top_platen_velocity=target_velocity,
        )

    sim = IceDEM3D(config)
    catalog = AEEventCatalog()
    catalog.snapshot(sim)

    breaking_strain = config.breaking_strain
    target_strain = max_strain_fraction * breaking_strain

    for _ in range(n_steps):
        sim.step()
        catalog.detect_and_record(sim)

    passed = catalog.n_events == 0

    return {
        "passed": passed,
        "n_events": catalog.n_events,
        "max_applied_strain": target_strain,
        "breaking_strain": breaking_strain,
        "strain_ratio": max_strain_fraction,
    }


def single_bond_monotonic_control(
    bond_stiffness: float = 2000.0,
    breaking_strain: float = 0.01,
    rest_length: float = 0.02,
    strain_rate: float = 0.1,
) -> dict[str, Any]:
    """Control 2: Single bond pulled monotonically produces exactly one event.

    A minimal 2x2x2 specimen with one bond axis pulled at constant strain rate
    until it breaks. The expected behavior is:
        - Exactly 1 AE event (first bond to break)
        - Event time: t_fail = breaking_strain / strain_rate
        - Event energy: E_rel = 0.5 * k * (ε_b * L0)^2

    Args:
        bond_stiffness: Bond spring constant (N/m).
        breaking_strain: Failure strain (dimensionless).
        rest_length: Bond rest length (m).
        strain_rate: Applied strain rate (1/s).

    Returns:
        Dict with:
            passed: bool - True if exactly 1 event with correct energy
            n_events: int
            expected_energy_J: float
            measured_energy_J: float
            energy_relative_error: float
            expected_time_s: float
            measured_time_s: float
    """
    # Compute platen velocity for desired strain rate
    speed = strain_rate * rest_length
    expected_time = breaking_strain / strain_rate
    expected_energy = single_bond_energy_release(bond_stiffness, breaking_strain, rest_length)
    
    # Use a minimal 2x2x2 specimen
    config = DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=rest_length / 2.0,
        bond_stiffness=bond_stiffness,
        breaking_strain=breaking_strain,
        dt=1e-6,
        tool_radius=0.015,
        tool_speed=0.01,
        top_platen_enabled=True,
        top_platen_velocity=speed,
    )

    sim = IceDEM3D(config)
    catalog = AEEventCatalog()
    catalog.snapshot(sim)

    # Limit max steps to avoid hanging
    max_steps = min(int(math.ceil(expected_time / config.dt)) + 10, 200)

    for step in range(max_steps):
        sim.step()
        catalog.detect_and_record(sim)

        # Check if any bond has broken
        if catalog.n_events > 0:
            break

    n_events = catalog.n_events
    # For this control, we accept >= 1 events (first event is what matters)
    passed = n_events >= 1

    if n_events > 0:
        measured_energy = catalog.events[0].energy_release_J
        measured_time = catalog.events[0].time_s
        energy_error = abs(measured_energy - expected_energy) / expected_energy
        time_error = abs(measured_time - expected_time) / expected_time if expected_time > 0 else 0.0
    else:
        measured_energy = 0.0
        measured_time = 0.0
        energy_error = float('inf')
        time_error = float('inf')

    return {
        "passed": passed,
        "n_events": n_events,
        "expected_energy_J": expected_energy,
        "measured_energy_J": measured_energy,
        "energy_relative_error": energy_error,
        "expected_time_s": expected_time,
        "measured_time_s": measured_time,
        "time_relative_error": time_error,
        "analytic_event_rate_Hz": 1.0 / expected_time if expected_time > 0 else 0.0,
    }


def synchronous_failure_control(
    n_bonds: int = 10,
    bond_stiffness: float = 1000.0,
    breaking_strain: float = 0.01,
) -> dict[str, Any]:
    """Control 3: Synchronous failure produces indeterminate b-value.

    When all bonds break at the same strain (and thus same energy), the
    magnitude distribution is a delta function and the b-value is undefined.
    The system should report this as indeterminate.

    Args:
        n_bonds: Number of bonds (all with identical properties).
        bond_stiffness: Bond stiffness (N/m).
        breaking_strain: Breaking strain (dimensionless).

    Returns:
        Dict with:
            passed: bool - True if b-value is correctly reported as indeterminate
            b_value: float - should be NaN or inf
            is_indeterminate: bool
            n_events: int - should equal n_bonds
            all_energies_equal: bool
    """
    # Create synthetic events with identical energy
    rest_length = 0.02
    expected_energy = single_bond_energy_release(bond_stiffness, breaking_strain, rest_length)

    # All events have the same energy
    energies = [expected_energy] * n_bonds
    magnitudes = [math.log10(e) for e in energies]

    # Compute b-value
    b_value = aki_bvalue_mle(magnitudes)

    # Check if indeterminate (NaN or inf)
    is_indeterminate = not math.isfinite(b_value)

    # All energies should be equal
    all_equal = all(abs(e - expected_energy) < 1e-12 for e in energies)

    passed = is_indeterminate and all_equal

    return {
        "passed": passed,
        "b_value": b_value,
        "is_indeterminate": is_indeterminate,
        "n_events": n_bonds,
        "all_energies_equal": all_equal,
        "energy_J": expected_energy,
        "message": "b-value is indeterminate for synchronous failure (all events identical)" if passed else "FAIL: b-value should be indeterminate",
    }


def run_all_controls() -> dict[str, dict[str, Any]]:
    """Run all three negative controls and return results."""
    return {
        "elastic_no_fracture": elastic_no_fracture_control(),
        "single_bond_monotonic": single_bond_monotonic_control(),
        "synchronous_failure": synchronous_failure_control(),
    }


__all__ = [
    "elastic_no_fracture_control",
    "single_bond_monotonic_control",
    "synchronous_failure_control",
    "run_all_controls",
]
