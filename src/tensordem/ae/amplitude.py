"""AE amplitude model with P-wave geometric spreading correction.

Physical model:
    When a bond breaks, the released elastic energy E_rel radiates as an
    elastic wave. The peak particle velocity (PPV) or displacement amplitude
    at distance r from the source is:

        A(r) = A₀ · sqrt(E_rel) / r

    where A₀ is a calibration constant that depends on the material properties
    (density, P-wave speed) and the radiation pattern. For a point source in
    an infinite medium:

        A₀ = C / (ρ · c_p)

    where C is a dimensionless radiation coefficient (~1/(4π) for isotropic),
    ρ is density, and c_p is the P-wave speed.

    For the DEM lattice, the P-wave speed is:
        c_p = a · sqrt(k / m)

    where a is the lattice spacing, k is the bond stiffness, and m is the
    particle mass.

Geometric spreading:
    In 3D, amplitude decays as 1/r (spherical spreading).
    In 2D (plane strain), amplitude decays as 1/sqrt(r) (cylindrical spreading).
    This module implements the 3D case by default.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch


def geometric_spreading(distance_m: float | torch.Tensor,
                        dimension: int = 3) -> float | torch.Tensor:
    """Geometric spreading factor for P-wave amplitude decay.

    Args:
        distance_m: Distance from source to receiver (m).
        dimension: Spatial dimension (2 or 3).

    Returns:
        Spreading factor G(r) such that A(r) = A₀ · sqrt(E) · G(r).

    For 3D: G(r) = 1/r
    For 2D: G(r) = 1/sqrt(r)
    """
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")

    if isinstance(distance_m, torch.Tensor):
        r = distance_m.clamp_min(1e-12)
        if dimension == 3:
            return 1.0 / r
        else:
            return 1.0 / r.sqrt()
    else:
        r = max(distance_m, 1e-12)
        if dimension == 3:
            return 1.0 / r
        else:
            return 1.0 / math.sqrt(r)


def p_wave_speed(bond_stiffness: float, particle_mass: float,
                 lattice_spacing: float) -> float:
    """P-wave speed in the bonded DEM lattice (m/s).

    c_p = a · sqrt(k / m)

    This is the long-wavelength limit of the lattice dispersion relation.
    """
    if particle_mass <= 0 or lattice_spacing <= 0 or bond_stiffness <= 0:
        raise ValueError("All inputs must be positive")
    return lattice_spacing * math.sqrt(bond_stiffness / particle_mass)


def radiation_coefficient(dimension: int = 3) -> float:
    """Dimensionless radiation pattern coefficient.

    For an isotropic (explosive) point source:
        3D: C = 1/(4π)
        2D: C = 1/(2π) (per unit thickness)
    """
    if dimension == 3:
        return 1.0 / (4.0 * math.pi)
    elif dimension == 2:
        return 1.0 / (2.0 * math.pi)
    else:
        raise ValueError("dimension must be 2 or 3")


def ae_amplitude(energy_release_J: float,
                 distance_m: float,
                 density: float,
                 p_wave_speed_ms: float,
                 radiation_coeff: float | None = None,
                 dimension: int = 3) -> float:
    """Compute AE peak displacement amplitude at a receiver.

    The amplitude model is:
        A = C · sqrt(E_rel) · G(r) / (ρ · c_p)

    where:
        C = radiation coefficient (default: 1/(4π) for 3D)
        E_rel = released elastic energy (J)
        G(r) = geometric spreading factor
        ρ = density (kg/m³)
        c_p = P-wave speed (m/s)

    Args:
        energy_release_J: Released elastic energy from bond break (J).
        distance_m: Distance from source to receiver (m).
        density: Material density (kg/m³).
        p_wave_speed_ms: P-wave speed in the material (m/s).
        radiation_coeff: Radiation pattern coefficient (default: auto from dimension).
        dimension: Spatial dimension (2 or 3).

    Returns:
        Peak displacement amplitude (m).
    """
    if energy_release_J < 0:
        raise ValueError("energy_release_J must be non-negative")
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    if density <= 0 or p_wave_speed_ms <= 0:
        raise ValueError("density and p_wave_speed_ms must be positive")

    if radiation_coeff is None:
        radiation_coeff = radiation_coefficient(dimension)

    G = geometric_spreading(distance_m, dimension)
    impedance = density * p_wave_speed_ms

    amplitude = radiation_coeff * math.sqrt(energy_release_J) * G / impedance
    return amplitude


def ae_amplitude_array(energies_J: Sequence[float],
                       distances_m: Sequence[float] | float,
                       density: float,
                       p_wave_speed_ms: float,
                       radiation_coeff: float | None = None,
                       dimension: int = 3) -> list[float]:
    """Compute AE amplitudes for multiple events at given distances.

    If distances_m is a scalar, all events are assumed at that distance.
    If it's a sequence, it must have the same length as energies_J.
    """
    if isinstance(distances_m, (int, float)):
        distances_m = [float(distances_m)] * len(energies_J)
    if len(energies_J) != len(distances_m):
        raise ValueError("energies_J and distances_m must have the same length")

    return [
        ae_amplitude(e, r, density, p_wave_speed_ms, radiation_coeff, dimension)
        for e, r in zip(energies_J, distances_m)
    ]


def calibration_factor_from_dem(bond_stiffness: float, particle_mass: float,
                                lattice_spacing: float,
                                density: float,
                                dimension: int = 3) -> float:
    """Compute the amplitude calibration factor A₀ for a DEM configuration.

    A₀ = C / (ρ · c_p)

    This factor converts sqrt(E_rel) · G(r) to displacement amplitude.
    """
    c_p = p_wave_speed(bond_stiffness, particle_mass, lattice_spacing)
    C = radiation_coefficient(dimension)
    return C / (density * c_p)


__all__ = [
    "geometric_spreading", "p_wave_speed", "radiation_coefficient",
    "ae_amplitude", "ae_amplitude_array", "calibration_factor_from_dem",
]
