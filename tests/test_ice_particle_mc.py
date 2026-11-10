"""Unit + integration tests for the ice-particle Mohr-Coulomb verification surface.

The tests are deliberately layered:

* closed-form identities (Mohr-Coulomb UCS from the plane-minimisation, the
  volume-consistent bond area, the isotropic-lattice modulus ``E_bond/6``);
* force-law regressions (compression pushes apart, shear *opposes* the relative
  displacement -- the sign error that made an earlier revision diverge);
* integrator energy conservation without dissipation;
* estimator sanity on synthetic data;
* one short end-to-end uniaxial run with the energy ledger closed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.ice_particle_mc import analysis
from tensordem.ice_particle_mc.bond import (MohrCoulombStrength, affine_stiffness_matrix,
                                            affine_uniaxial_modulus, bond_areas,
                                            bond_failure_mask, mohr_coulomb_ucs,
                                            mohr_coulomb_critical_normal_fraction)
from tensordem.ice_particle_mc.engine import AssemblyConfig, IceParticleAssembly
from tensordem.ice_particle_mc.packing import (Specimen, cylinder_specimen,
                                               coordination_number,
                                               orientation_coverage,
                                               validate_cylinder)
from tensordem.ice_particle_mc.study import UniaxialSpec, run_uniaxial

torch.set_num_threads(1)


# ----------------------------------------------------------------------
# 1. closed forms
# ----------------------------------------------------------------------
@pytest.mark.parametrize("phi_deg", [0.5, 15.0, 25.0, 30.0, 35.0, 45.0])
def test_ucs_is_the_plane_minimisation_of_the_mc_criterion(phi_deg):
    """``max_x [sqrt(x(1-x)) - x tan phi] = (1 - sin phi)/(2 cos phi)``.

    That minimum is exactly what makes the bond-level criterion reproduce the
    continuum Mohr-Coulomb uniaxial strength; the identity is checked numerically
    and against the closed form.
    """
    phi = math.radians(phi_deg)
    x = np.linspace(1e-6, 1 - 1e-6, 2_000_001)
    h = np.sqrt(x * (1.0 - x)) - x * math.tan(phi)
    # a plane fails at sigma = c/h(x); the first to fail maximises h
    numeric = float(h.max())
    closed = (1.0 - math.sin(phi)) / (2.0 * math.cos(phi))
    assert numeric == pytest.approx(closed, rel=5e-4)
    cohesion = 1.5e6
    assert cohesion / numeric == pytest.approx(mohr_coulomb_ucs(cohesion, phi_deg), rel=5e-4)


def test_critical_orientation_matches_the_derivation():
    for phi_deg in (10.0, 25.0, 30.0, 40.0):
        x_star = mohr_coulomb_critical_normal_fraction(phi_deg)
        phi = math.radians(phi_deg)
        # dh/dx = (1 - 2x)/(2 sqrt(x(1-x))) - tan(phi) = 0
        derivative = (1.0 - 2.0 * x_star) / (2.0 * math.sqrt(x_star * (1.0 - x_star)))
        assert derivative == pytest.approx(math.tan(phi), rel=1e-12)


def test_bond_area_is_volume_consistent():
    """``sum_b A_b L_b = V``: the stress measure of the lattice is preserved."""
    specimen = cylinder_specimen(0.05, 0.10, 0.006, seed=3)
    area = bond_areas(specimen.volume, specimen.n_bonds, specimen.bond_length)
    represented = float((area * specimen.bond_length).sum())
    assert represented == pytest.approx(specimen.volume, rel=1e-12)


def test_isotropic_lattice_modulus_is_e_bond_over_six():
    """An isotropic central-force bond set has ``E = E_bond / 6``, ``nu = 1/4``.

    This is the micro-macro map that makes the elastic modulus *predictable* from
    the input bond stiffness (and shows ``E_bond != E_macro``).
    """
    gen = torch.Generator().manual_seed(7)
    n = 200_000
    v = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    v = v / v.norm(dim=1, keepdim=True)
    length = torch.full((n,), 0.01, dtype=torch.float64)
    kn = torch.full((n,), 1.0e5, dtype=torch.float64)
    ks = torch.zeros(n, dtype=torch.float64)
    # one bond per unit volume of material, so that C = (1/V) sum_b k L^2 nnnn
    stiffness = affine_stiffness_matrix(v, length, kn, ks, volume=float(n))
    # C1111 = kn L^2 /V * <n^4> = kn L^2/V /5, C1122 = kn L^2/V /15
    unit = float(kn[0] * length[0] ** 2)
    assert float(stiffness[0, 0]) == pytest.approx(unit / 5.0, rel=2e-3)
    assert float(stiffness[0, 1]) == pytest.approx(unit / 15.0, rel=2e-3)
    modulus = affine_uniaxial_modulus(stiffness, axis=2)
    assert modulus == pytest.approx(unit / 6.0, rel=3e-3)


def test_specimen_geometry_is_verified_before_mechanics():
    specimen = cylinder_specimen(0.05, 0.10, 0.006, seed=1)
    assert specimen.positions[:, 2].max() <= 0.10 + 1e-9
    # the digital diameter is *measured* and reported (validate_cylinder) rather
    # than assumed: the relaxation can leave a small outward scatter, which the
    # report has to show because the stress normalisation uses the nominal area
    geometry = validate_cylinder(specimen.positions, specimen.radii, 0.05, specimen.height)
    assert geometry["diameter_error"] < 0.10   # reported, not assumed
    assert geometry["height_error"] < 0.01
    assert abs(specimen.height - 0.10) / 0.10 < 0.02   # reported by validate_cylinder
    # every initially interpenetrating pair must be *bonded*: a bonded pair is
    # governed by the bond law (no contact force), an unbonded overlap would
    # produce a huge Hertz force and an unstable run
    i, j = torch.triu_indices(specimen.n_particles, specimen.n_particles, offset=1)
    overlap = (specimen.radii[i] + specimen.radii[j]) - (
        specimen.positions[j] - specimen.positions[i]).norm(dim=1)
    touching = overlap > 0.0
    keys = set((int(min(a, b)), int(max(a, b)))
               for a, b in zip(specimen.bond_i.tolist(), specimen.bond_j.tolist()))
    for a, b in zip(i[touching].tolist(), j[touching].tolist()):
        assert (min(a, b), max(a, b)) in keys
    assert coordination_number(specimen) > 3.0
    coverage = orientation_coverage(specimen, 30.0)
    assert coverage["fraction_near_critical"] > 0.0


# ----------------------------------------------------------------------
# 2. force-law regressions
# ----------------------------------------------------------------------
def _two_particle(offset: torch.Tensor, config: AssemblyConfig | None = None):
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    positions[1] = offset
    radii = torch.tensor([0.004, 0.004], dtype=torch.float64)
    bond_i = torch.tensor([0], dtype=torch.int64)
    bond_j = torch.tensor([1], dtype=torch.int64)
    specimen = Specimen(positions=positions, radii=radii, bond_i=bond_i, bond_j=bond_j,
                        bond_length=offset.norm().reshape(1).clone(),
                        diameter=0.02, height=0.02, volume=1.0e-5)
    return IceParticleAssembly(specimen, config or AssemblyConfig(cohesion_pa=1e12,
                                                                 tensile_strength_pa=1e12))


def test_compression_pushes_the_particles_apart():
    assembly = _two_particle(torch.tensor([0.0, 0.0, 0.008], dtype=torch.float64))
    assembly.positions[1, 2] -= 1e-7            # compress the bond
    forces = assembly.forces()
    assert float(forces[0, 2]) < 0.0            # particle 0 pushed down (-z)
    assert float(forces[1, 2]) > 0.0            # particle 1 pushed up (+z)
    assert float(forces[0, 2]) == pytest.approx(-float(forces[1, 2]), rel=1e-12)


def test_shear_force_opposes_the_relative_displacement():
    """A ``+k q`` shear force is a negative-stiffness runaway; this pins the sign."""
    assembly = _two_particle(torch.tensor([0.0, 0.0, 0.008], dtype=torch.float64))
    assembly.bond_shear[0, 0] = 1e-7           # j displaced in +x relative to i
    forces = assembly.forces()
    assert float(forces[1, 0]) < 0.0           # restoring: pushed back in -x
    assert float(forces[0, 0]) > 0.0
    assert float(forces[1, 0]) == pytest.approx(-float(assembly.bond_ks[0]) * 1e-7, rel=1e-9)


def test_mohr_coulomb_bond_rupture_surface():
    strength = MohrCoulombStrength(1.5e6, 30.0, 1.0e6)
    area = torch.tensor([3.0e-5], dtype=torch.float64)
    cohesion_force = strength.cohesion_pa * float(area[0])
    # no normal force: rupture exactly at |F_t| = c A
    low = bond_failure_mask(torch.tensor([0.0], dtype=torch.float64),
                            torch.tensor([0.99 * cohesion_force]), area, strength)
    high = bond_failure_mask(torch.tensor([0.0], dtype=torch.float64),
                             torch.tensor([1.01 * cohesion_force]), area, strength)
    assert not bool(low["any"][0])
    assert bool(high["shear"][0])
    # a compressive normal force raises the shear capacity by tan(phi) F_n
    compression = 100.0
    capacity = cohesion_force + compression * strength.tan_phi
    assert not bool(bond_failure_mask(
        torch.tensor([compression]), torch.tensor([0.99 * capacity]), area,
        strength)["any"][0])
    # tension is not allowed to raise the capacity: the tensile cut-off fires first
    tensile = bond_failure_mask(torch.tensor([-1.5e6 * float(area[0])]),
                                torch.tensor([0.0]), area, strength)
    assert bool(tensile["tensile"][0])


def test_free_bonded_system_conserves_energy():
    """No external work, no dissipation: KE + bond energy must be constant."""
    specimen = cylinder_specimen(0.03, 0.04, 0.004, seed=0)
    assembly = IceParticleAssembly(specimen, AssemblyConfig(
        cohesion_pa=1.0e12, tensile_strength_pa=1.0e12))
    assembly.velocities = torch.randn(assembly.n, 3, dtype=torch.float64) * 1e-3
    energy0 = assembly.contact_kinetic() + assembly.bond_elastic
    for _ in range(400):
        assembly.step()
    energy1 = assembly.contact_kinetic() + assembly.bond_elastic
    assert abs(energy1 - energy0) / energy0 < 0.02


# ----------------------------------------------------------------------
# 3. estimators
# ----------------------------------------------------------------------
def test_exponential_tail_fit_recovers_a_known_exponential():
    gen = np.random.default_rng(0)
    # P(f) ~ exp(-f/<f>): exponential sample, shifted so the tail starts at <f>
    sample = gen.exponential(scale=1.0, size=200_000) + 0.0
    stats = analysis.exponential_tail_fit(sample, n_bins=25, min_count=20)
    assert stats["r2"] > 0.98
    assert stats["decay_length"] == pytest.approx(1.0 / stats["mean_force_N"], rel=0.5)


def test_pressure_area_exponent_recovers_minus_half():
    areas = np.array([0.001, 0.002, 0.004, 0.008, 0.016])
    pressures = 3.0e6 * (areas / 0.001) ** -0.5
    fit = analysis.pressure_area_exponent(areas, pressures)
    assert fit["exponent"] == pytest.approx(-0.5, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit["exponent_error"] < 1e-6


# ----------------------------------------------------------------------
# 4. end-to-end smoke: uniaxial run with a closed ledger
# ----------------------------------------------------------------------
def test_uniaxial_run_reaches_mohr_coulomb_and_closes_the_energy_ledger():
    result = run_uniaxial(UniaxialSpec(cohesion_pa=1.5e6, friction_angle_deg=30.0,
                                       particle_radius=0.006, strain_target=8.0e-4,
                                       step_displacement=1.0e-7))
    assert result["n_particles"] > 40
    assert result["E_measured_Pa"] > 0.0
    assert result["E_r2"] > 0.95
    assert abs(result["energy_residual_relative"]) < 0.05
    # the analytic fabric prediction is well defined and finite
    assert math.isfinite(result["fabric_mc"]["sigma_ucs_Pa"])
    assert result["fabric_mc"]["critical_bond_n_z_squared"] == pytest.approx(0.25, abs=0.2)