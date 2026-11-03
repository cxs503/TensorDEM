"""Tests for :mod:`tensordem.lattice_waves` and the phonon-bound correction.

Every analytic reference is cross-checked by an independent route:

* the closed-form [100] branches against the assembled all-pairs dynamical matrix;
* the corrected ``max_phonon_frequency`` against a brute-force 26-neighbour sum and
  against the closed-form zone-edge frequency;
* the spectral response of the discrete bond network against a direct time
  integration of the equations of motion by an independent (pure-numpy style)
  central-difference scheme.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.lattice_elasticity import (
    IDEAL_OFFSETS,
    ideal_box_bonds,
    max_phonon_frequency,
)
from tensordem.lattice_waves import (
    FULL_NEIGHBOURS,
    acoustic_speeds,
    bar_dispersion,
    bulk_dispersion,
    bulk_dynamical_matrix,
    cross_section_bloch,
    dispersion_along_100,
    finite_bar_modes,
    frequency_from_zero_crossings,
    linear_response_trace,
    longitudinal_branch_index,
    mode_experiment,
    one_minus_cos,
    peak_time,
)

BOND_STIFFNESS = 2000.0
RADIUS = 0.025
DENSITY = 917.0
SPACING = 2.0 * RADIUS
MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3


def test_full_neighbour_set_is_the_thirteen_classes_and_their_mirrors():
    assert len(IDEAL_OFFSETS) == 13
    assert len(FULL_NEIGHBOURS) == 26
    assert len(set(FULL_NEIGHBOURS)) == 26
    assert all((-ox, -oy, -oz) in FULL_NEIGHBOURS for ox, oy, oz in IDEAL_OFFSETS)


def test_one_minus_cos_is_stable_at_small_angle():
    # 1 - cos(1e-9) evaluates to exactly 0.0 through the naive expression.
    assert one_minus_cos(1e-9) == pytest.approx(0.5e-18, rel=1e-6)
    assert one_minus_cos(0.0) == 0.0
    for theta in (0.1, 0.7, 1.9, math.pi):
        assert one_minus_cos(theta) == pytest.approx(1.0 - math.cos(theta), rel=1e-12)


@pytest.mark.parametrize("ka", [0.001, 0.05, 0.3, 1.0, math.pi])
def test_closed_form_matches_assembled_branches(ka: float):
    k = ka / SPACING
    closed = dispersion_along_100(k, bond_stiffness=BOND_STIFFNESS, mass=MASS, spacing=SPACING)
    assembled = bulk_dispersion([k, 0.0, 0.0], bond_stiffness=BOND_STIFFNESS, mass=MASS, spacing=SPACING)
    assert assembled["omega_rad_s"][2] == pytest.approx(closed["omega_L_rad_s"], rel=1e-10)
    assert assembled["omega_rad_s"][0] == pytest.approx(closed["omega_T_rad_s"], rel=1e-10)
    assert assembled["omega_rad_s"][1] == pytest.approx(closed["omega_T_rad_s"], rel=1e-10)


def test_long_wave_speeds_reproduce_the_calibrated_elastic_constants():
    speeds = acoustic_speeds(bond_stiffness=BOND_STIFFNESS, mass=MASS, spacing=SPACING)
    limit = dispersion_along_100(1e-6 / SPACING, bond_stiffness=BOND_STIFFNESS,
                                 mass=MASS, spacing=SPACING)
    assert limit["group_velocity_L_m_per_s"] == pytest.approx(
        speeds["c_longitudinal_bulk_m_per_s"], rel=1e-9)
    assert limit["group_velocity_T_m_per_s"] == pytest.approx(
        speeds["c_transverse_m_per_s"], rel=1e-9)
    # The slender-bar speed is below the bulk speed for this anisotropic lattice.
    assert speeds["c_bar_slender_m_per_s"] < speeds["c_longitudinal_bulk_m_per_s"]


def test_dynamical_matrix_is_symmetric_positive_semidefinite():
    dyn = bulk_dynamical_matrix([0.3, -0.2, 0.1], bond_stiffness=BOND_STIFFNESS,
                                mass=MASS, spacing=SPACING)
    assert torch.allclose(dyn, dyn.t(), atol=1e-12)
    assert torch.linalg.eigvalsh(dyn).min() > -1e-12
    zero = bulk_dynamical_matrix([0.0, 0.0, 0.0], bond_stiffness=BOND_STIFFNESS,
                                 mass=MASS, spacing=SPACING)
    assert float(zero.abs().max()) == 0.0


def test_phonon_bound_matches_brute_force_over_all_26_directions():
    """The half-space sum must carry the same weight as the explicit 26 directions."""
    normals, lengths = ideal_box_bonds(SPACING)
    corrected = max_phonon_frequency(normals, lengths, BOND_STIFFNESS, MASS, 15, SPACING)
    omega_edge = math.sqrt(2.0 * (26.0 / 3.0) * BOND_STIFFNESS / MASS)
    assert corrected == pytest.approx(omega_edge, rel=1e-9)
    # Brute force over every sampled q with the explicit 26-neighbour list.
    grid = torch.linspace(-math.pi / SPACING, math.pi / SPACING, 9, dtype=torch.float64)
    worst = 0.0
    for qx in grid.tolist():
        for qy in grid.tolist()[::4]:
            for qz in grid.tolist()[::4]:
                freq = bulk_dispersion([qx, qy, qz], bond_stiffness=BOND_STIFFNESS,
                                       mass=MASS, spacing=SPACING)["omega_rad_s"][-1]
                worst = max(worst, freq)
    assert worst == pytest.approx(corrected, rel=1e-5)


def test_phonon_bound_prior_implementation_was_low_by_root_two():
    normals, lengths = ideal_box_bonds(SPACING)
    corrected = max_phonon_frequency(normals, lengths, BOND_STIFFNESS, MASS, 15, SPACING)
    # Reproduce the pre-fix definition (each half-space class counted once).
    vectors = normals * lengths[:, None]
    outer = normals[:, :, None] * normals[:, None, :] * BOND_STIFFNESS
    grid = torch.linspace(-math.pi / SPACING, math.pi / SPACING, 15, dtype=torch.float64)
    qz, qy, qx = torch.meshgrid(grid, grid, grid, indexing="ij")
    qs = torch.stack((qx, qy, qz), dim=-1).reshape(-1, 3)
    weights = 1.0 - torch.cos(qs @ vectors.T)
    dyn = torch.einsum("qb,bij->qij", weights, outer) / MASS
    prior = float(torch.sqrt(torch.linalg.eigvalsh(dyn).clamp_min(0.0).max()))
    assert corrected / prior == pytest.approx(math.sqrt(2.0), rel=1e-9)


def test_bloch_single_cell_reduces_to_the_chain_dispersion():
    for kx in (0.01, 0.1, 0.4):
        omega = cross_section_bloch(
            kx, bond_stiffness=BOND_STIFFNESS, mass=MASS, spacing=SPACING, ny=1, nz=1
        )["omega_rad_s"]
        expected = math.sqrt(2.0 * BOND_STIFFNESS / MASS * one_minus_cos(kx * SPACING))
        assert omega[-1] == pytest.approx(expected, rel=1e-8)
        assert omega[0] == pytest.approx(0.0, abs=1e-12)


def test_bar_branch_is_acoustic_and_monotone_in_group_velocity():
    branch = longitudinal_branch_index(bond_stiffness=BOND_STIFFNESS, mass=MASS,
                                       spacing=SPACING, ny=2, nz=2)
    speeds = [
        bar_dispersion(kx, bond_stiffness=BOND_STIFFNESS, mass=MASS, spacing=SPACING,
                       ny=2, nz=2, branch=branch)["group_velocity_m_per_s"]
        for kx in (1e-3, 0.05, 0.2, 0.5)
    ]
    assert all(v > 0.0 for v in speeds)
    # Long-wave limit of the branch equals its phase velocity.
    first = bar_dispersion(1e-3, bond_stiffness=BOND_STIFFNESS, mass=MASS,
                           spacing=SPACING, ny=2, nz=2, branch=branch)
    assert first["group_velocity_m_per_s"] == pytest.approx(
        first["phase_velocity_m_per_s"], rel=1e-4)


def test_mode_integration_reproduces_the_exact_eigen_frequency():
    config = DEM3DConfig(nx=11, ny=2, nz=2, radius=RADIUS, density=DENSITY,
                         bond_stiffness=BOND_STIFFNESS, contact_stiffness=100.0,
                         contact_damping=0.0, fix_x_edges=True, drag=0.0,
                         tool_gap=1.0, tool_speed=1e-6)
    dem = IceDEM3D(config)
    modes = finite_bar_modes(dem, count=12)
    axial = modes["mode_vectors"][0::3, :].abs().pow(2).sum(dim=0)
    index = int(torch.argmax(axial[2:]).item()) + 2
    omega_ref = modes["omega_rad_s"][index]
    run = mode_experiment(dem, modes["mode_vectors"][:, index],
                          free_dof=modes["free_dof"], omega_rad_s=omega_ref,
                          amplitude_m=1e-6, periods=2.0)
    measured = frequency_from_zero_crossings(run["times_s"], run["axial_displacement_m"])
    assert measured["omega_rad_s"] == pytest.approx(omega_ref, rel=1e-3)
    assert run["broken_bonds"] == 0


def test_peak_time_interpolates_between_samples():
    times = [0.0, 0.1, 0.2, 0.3]
    signal = [0.2, 0.9, 1.0, 0.4]
    value = peak_time(times, signal)
    assert 0.1 < value < 0.2


def test_pulse_specification_validation():
    from tensordem.lattice_waves import PulseSpec

    with pytest.raises(ValueError):
        PulseSpec(station_cells=(50, 40))
    with pytest.raises(ValueError):
        PulseSpec(nx=20, station_cells=(5, 25))
    with pytest.raises(ValueError):
        PulseSpec(amplitude_m=0.0)
    with pytest.raises(ValueError):
        PulseSpec(packet_cell=30, station_cells=(20, 40))
    PulseSpec(nx=41, packet_cell=4, station_cells=(12, 36))  # valid