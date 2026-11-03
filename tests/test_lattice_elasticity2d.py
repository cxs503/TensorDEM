"""Analytic and solver-level checks for the 2-D lattice elasticity module."""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.calibrated_dem import CalibratedIceDEM
from tensordem.dem import DEMConfig, IceDEM
from tensordem import lattice_elasticity2d as le


K = 2000.0
RADIUS = 0.5e-3
SPACING = 2.0 * RADIUS


def make_config(nx: int = 7, ny: int = 7) -> DEMConfig:
    return DEMConfig(
        nx=nx,
        ny=ny,
        radius=RADIUS,
        thickness=0.2,
        density=917.0,
        bond_stiffness=K,
        contact_stiffness=K,
        fix_edges=False,
    )


def test_closed_form_constants():
    components = le.stiffness_components(K)
    assert components["c11"] == pytest.approx(2.0 * K, rel=1e-15)
    assert components["c12"] == pytest.approx(K, rel=1e-15)
    assert components["c66"] == pytest.approx(K, rel=1e-15)
    assert components["diagonal_stiffness"] == pytest.approx(K, rel=1e-15)

    isotropic = le.stiffness_components(K, le.ISOTROPIC_DIAGONAL_SCALE)
    assert isotropic["c11"] == pytest.approx(1.5 * K, rel=1e-15)
    assert isotropic["c12"] == pytest.approx(0.5 * K, rel=1e-15)
    assert isotropic["c66"] == pytest.approx(0.5 * K, rel=1e-15)


def test_voigt_summation_reproduces_closed_form():
    normals, lengths, stiffness = le.canonical_bonds(SPACING, bond_stiffness=K)
    summed = le.voigt_sum_stiffness(
        normals, lengths, stiffness, SPACING * SPACING, weight=0.5
    )
    closed = le.affine_stiffness_tensor(K, spacing=SPACING)
    assert le.tensor_error(summed, closed) < 1e-12
    assert float(closed[0, 0, 0, 0]) == pytest.approx(2.0 * K, rel=1e-12)
    assert float(closed[0, 0, 1, 1]) == pytest.approx(K, rel=1e-12)
    assert float(closed[0, 1, 0, 1]) == pytest.approx(K, rel=1e-12)


def test_solver_topology_matches_closed_form():
    sim = IceDEM(make_config())
    topology = le.bulk_stiffness_from_sim(sim)
    closed = le.affine_stiffness_tensor(K, spacing=SPACING)
    assert topology["coordination_max"] == 8
    assert topology["interior_particles"] == 25
    assert le.tensor_error(topology["stiffness"], closed) < 1e-12


def test_solver_energy_matches_closed_form():
    sim = IceDEM(make_config())
    closed = le.affine_stiffness_tensor(K, spacing=SPACING)
    measured = le.stiffness_from_strain_energy(sim)
    assert le.tensor_error(measured, closed) < 1e-5


def test_surface_deficit_is_recovered():
    sim = IceDEM(make_config(5, 5))
    normals, lengths, stiffness = le.missing_bond_deficit(sim)
    # 5x5 patch: 16 boundary particles, corner 5 slots, edge 3 slots.
    assert len(normals) == 4 * 5 + 12 * 3 == 56
    assert torch.unique(lengths).numel() == 2
    assert float(stiffness.min()) == pytest.approx(K)


def test_isotropic_calibration_is_exactly_isotropic():
    config = make_config()
    reference = IceDEM(config)
    coefficients = le.isotropic_bond_stiffness(reference)
    assert torch.unique(coefficients).tolist() == [0.5 * K, K]
    assert float(coefficients.max()) <= config.bond_stiffness
    calibrated = CalibratedIceDEM(config, bond_stiffness=coefficients)

    closed = le.affine_stiffness_tensor(
        K, le.ISOTROPIC_DIAGONAL_SCALE, spacing=SPACING
    )
    topology = le.bulk_stiffness_from_sim(calibrated)
    assert le.tensor_error(topology["stiffness"], closed) < 1e-12
    measured = le.stiffness_from_strain_energy(calibrated)
    assert le.tensor_error(measured, closed) < 1e-5

    report = le.anisotropy_report(K, le.ISOTROPIC_DIAGONAL_SCALE, spacing=SPACING)
    assert report["zener_ratio_2d"] == pytest.approx(1.0, abs=1e-12)
    assert report["modulus_ratio_max_min"] == pytest.approx(1.0, abs=1e-9)
    assert report["poisson_ratio_min"] == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert report["poisson_ratio_max"] == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert report["youngs_modulus_0deg_N_per_m"] == pytest.approx(4.0 * K / 3.0, rel=1e-12)
    assert report["coupling_c16_max_abs"] < 1e-9


def test_engine_default_anisotropy_is_quantified():
    report = le.anisotropy_report(K, 1.0, spacing=SPACING)
    assert report["zener_ratio_2d"] == pytest.approx(2.0, rel=1e-12)
    assert report["modulus_ratio_45_over_0"] == pytest.approx(1.6, rel=1e-12)
    assert report["youngs_modulus_0deg_N_per_m"] == pytest.approx(1.5 * K, rel=1e-12)
    assert report["youngs_modulus_45deg_N_per_m"] == pytest.approx(2.4 * K, rel=1e-12)
    assert report["poisson_ratio_min"] == pytest.approx(0.2, abs=1e-9)
    assert report["poisson_ratio_max"] == pytest.approx(0.5, abs=1e-9)
    assert report["theta_at_max_modulus_deg"] == pytest.approx(45.0)


def test_directional_symmetry():
    rows = le.directional_moduli(
        [0.0, 30.0, 60.0, 90.0], K, 1.0, spacing=SPACING
    )
    youngs = {row["theta_deg"]: row["youngs_modulus_N_per_m"] for row in rows}
    assert youngs[0.0] == pytest.approx(youngs[90.0], rel=1e-12)
    assert youngs[30.0] == pytest.approx(youngs[60.0], rel=1e-12)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        le.canonical_bonds(0.0, bond_stiffness=K)
    with pytest.raises(ValueError):
        le.stiffness_components(K, 0.0)
    with pytest.raises(ValueError):
        le.anisotropy_report(K, 1.0, n_theta=3)
    sim = IceDEM(make_config())
    with pytest.raises(ValueError):
        le.stiffness_from_strain_energy(sim, step=1.0)
    with pytest.raises(ValueError):
        le.tensor_error(torch.zeros((2, 2, 2, 2), dtype=torch.float64),
                        torch.zeros((2, 2, 2, 2), dtype=torch.float64))


def test_energy_density_is_quadratic_in_strain():
    """The affine energy of the patch must be quadratic to O(strain**3)."""
    sim = IceDEM(make_config())
    base = le.applied_strain_energy_density(sim, torch.zeros((2, 2), dtype=torch.float64))
    small = 1.0e-6
    strain = torch.tensor([[small, 0.0], [0.0, 0.0]], dtype=torch.float64)
    doubled = torch.tensor([[2.0 * small, 0.0], [0.0, 0.0]], dtype=torch.float64)
    one = le.applied_strain_energy_density(sim, strain) - base
    two = le.applied_strain_energy_density(sim, doubled) - base
    assert two / one == pytest.approx(4.0, rel=1e-6)
    assert math.isfinite(one)