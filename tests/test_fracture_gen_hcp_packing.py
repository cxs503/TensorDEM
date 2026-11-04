"""HCP (hexagonal close packed) fracture-field packing: analytic verification.

Verification debt being cleared
-------------------------------
``docs/icebreaking_development_roadmap.md`` (Increment 2.5, "Known limits")
declared that the FEM->DEM fracture-field generator supports *only* a simple
cubic (SC) lattice.  The SC lattice of touching spheres has packing fraction
``pi/6 = 0.5236``, i.e. 29 % below the densest possible arrangement of equal
spheres, so the ice field it builds is artificially porous: 47.6 % of every
cubic cell is void.  The HCP branch added in this increment keeps the *contact*
radius (so every contact law, bond cutoff and timestep criterion of
``IceDEM3D`` is unchanged) but fills the domain with the ABAB close packing,
whose coordination number is 12 instead of 6 and whose cell volume per particle
drops from ``8 r^3`` to ``4 sqrt(2) r^3``.

Every reference in this file is a hand-derived closed form; nothing is compared
against the implementation's own formulas or against another run of the solver.
Numerical references used below (``r`` = contact radius, ``h = 2 r``):

* nearest-neighbour distance of a touching-sphere HCP: ``2 r`` (in-plane and
  interlayer are equal by construction);
* bulk coordination number: ``Z = 12`` (SC: 6);
* per-particle directional fingerprint ``S4 = sum_{j~i} n_{j,x}^4``: ``2.5`` for
  HCP, ``2.0`` for SC (see ``test_directional_fingerprint_*``);
* affine uniaxial modulus of a central-force lattice,
  ``E = (k / (2 V)) sum_b l_b^2 n_{b,x}^4`` per bond counted once, hence
  ``E_SC = k / h`` (classical cubic-lattice result) and
  ``E_HCP = 5 k / (2 sqrt 2 h)``; ratio ``5 / (2 sqrt 2) = 1.767767``;
* packing fraction ``pi / (3 sqrt 2) = 0.7404804896930611``;
* cell volume ``4 sqrt 2 r^3`` and cell mass ``4 sqrt 2 rho r^3``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem import fracture_gen as fg
from tensordem import fracture_mass_mapping as fmm
from tensordem.lattice_packing_audit import (
    affine_axial_modulus_over_bond_stiffness,
    coordination_histogram,
    coordination_numbers,
    interior_mask,
    min_pair_distance,
    structure_report,
)

R = 0.05
H = 2.0 * R


def _box(length: float) -> tuple[float, float, float, float, float, float]:
    half = 0.5 * length
    return (-half, half, -half, half, -half, half)


def _lattice(packing: str, length: float = 0.8):
    domain = _box(length)
    positions = fg.generate_particle_lattice(domain, R, packing=packing)
    bonds, rest = fg.build_bond_topology(positions, R)
    return positions, bonds, rest


# ---------------------------------------------------------------- structure ---


def test_sc_branch_is_bit_for_bit_unchanged():
    """The default path must reproduce the historical SC numbers exactly."""
    positions = fg.generate_particle_lattice(_box(1.2), 0.1)
    bonds, rest = fg.build_bond_topology(positions, 0.1)
    assert positions.shape == (216, 3)
    assert len(bonds) == 540
    assert torch.allclose(rest, torch.full_like(rest, 0.2), rtol=0, atol=1e-15)
    # first centre convention: xmin + r, upper bound excluded
    assert positions[0].tolist() == [-0.5, -0.5, -0.5]


@pytest.mark.parametrize("packing", ["sc", "hcp"])
def test_every_bond_is_exactly_one_diameter(packing):
    """Touching-sphere packing: the bond length is 2r with no overlap at all."""
    positions, bonds, rest = _lattice(packing)
    assert min_pair_distance(positions) == pytest.approx(H, rel=0, abs=1e-12)
    unique = torch.unique(rest.round(decimals=12))
    assert unique.numel() == 1
    assert float(unique[0]) == pytest.approx(H, rel=0, abs=1e-12)


def test_hcp_bulk_coordination_is_twelve():
    """Coordination number 12 (HCP) versus 6 (SC) -- exact integers."""
    for packing, bulk in (("sc", 6), ("hcp", 12)):
        positions, bonds, _ = _lattice(packing)
        counts = coordination_numbers(len(positions), bonds)
        assert int(counts.max()) == bulk
        histogram = coordination_histogram(len(positions), bonds)
        assert histogram[bulk] > 0, histogram
        assert all(value <= bulk for value in histogram)


def test_directional_fingerprint_matches_hand_derived_sums():
    """``sum_{j~i} n_x^4`` per interior particle: 2.5 (HCP) and 2.0 (SC).

    Hand derivation.  SC: the two x-bonds give ``n_x^4 = 1`` each, the four
    y/z bonds give 0 -> 2.0.  HCP: two in-row bonds along x (``1`` each), four
    in-plane bonds with ``n_x = 1/2`` (``4 * 1/16``), four interlayer bonds with
    ``n_x = 1/2`` (``4 * 1/16``) and two interlayer bonds with ``n_x = 0`` ->
    ``2 + 0.25 + 0.25 = 2.5``.
    """
    expected = {"sc": 2.0, "hcp": 2.5}
    for packing, target in expected.items():
        positions, bonds, rest = _lattice(packing)
        mask = interior_mask(len(positions), bonds, packing)
        delta = positions[bonds[:, 1]] - positions[bonds[:, 0]]
        n_x = delta[:, 0] / rest
        per_particle = torch.zeros(len(positions), dtype=positions.dtype)
        for sign in (1.0, -1.0):  # each bond contributes to both endpoints
            per_particle.index_add_(0, bonds[:, 0] if sign > 0 else bonds[:, 1], n_x ** 4)
        values = per_particle[mask]
        assert values.numel() > 0
        assert torch.allclose(values, torch.full_like(values, target), rtol=0, atol=1e-12)


def test_affine_axial_modulus_matches_closed_form():
    """``E_SC = k/h`` and ``E_HCP = 5 k / (2 sqrt 2 h)`` (ratio 1.7678), exact."""
    e_sc_expected = 1.0 / H
    e_hcp_expected = 5.0 / (2.0 * math.sqrt(2.0) * H)
    measured = {}
    for packing, target in (("sc", e_sc_expected), ("hcp", e_hcp_expected)):
        positions, bonds, rest = _lattice(packing)
        value = affine_axial_modulus_over_bond_stiffness(
            positions, bonds, rest, R, packing
        )
        measured[packing] = value
        assert value == pytest.approx(target, rel=1e-12)
    assert measured["hcp"] / measured["sc"] == pytest.approx(5.0 / (2.0 * math.sqrt(2.0)), rel=1e-12)
    assert 5.0 / (2.0 * math.sqrt(2.0)) == pytest.approx(1.7677669529663689, rel=1e-15)


def test_hcp_number_density_converges_to_close_packed_value():
    """The realised number density approaches ``1 / (4 sqrt 2 r^3)`` with size.

    The trimming convention keeps every sphere inside the domain, so a finite
    lattice loses whole rows/layers on the boundary; the *measured* number of
    spheres per spanned volume therefore approaches the exact HCP value from
    below.  Monotone approach over a 4x volume increase is asserted, together
    with a bound on the largest lattice (documented value: 0.93 of the exact
    density at ``L = 2.0 m = 20`` diameters).
    """
    exact_cell = 4.0 * math.sqrt(2.0) * R ** 3
    fractions = []
    for length in (0.8, 1.2, 2.0):
        positions, bonds, rest = _lattice("hcp", length)
        report = structure_report(positions, bonds, rest, R, _box(length), "hcp")
        fractions.append(report["num_particles"] * exact_cell / report["spanned_volume_m3"])
    assert fractions[0] < fractions[1] < fractions[2] < 1.0
    assert fractions[2] > 0.90
    # SC fills its own box exactly, so this is the reference the HCP trend
    # approaches from below: 0.74048 / 0.52360 = sqrt(2).
    assert fmm.packing_fraction("hcp") / fmm.packing_fraction("sc") == pytest.approx(
        math.sqrt(2.0), rel=1e-12
    )


# ------------------------------------------------------------------- masses ---


def test_cell_volume_and_mass_follow_the_close_packing():
    cell = fmm.particle_cell_volume(R, "hcp")
    assert cell == pytest.approx(4.0 * math.sqrt(2.0) * R ** 3, rel=1e-15)
    assert cell == pytest.approx(fmm.particle_sphere_volume(R) / (math.pi / (3 * math.sqrt(2))), rel=1e-14)
    assert fmm.particle_mass(917.0, R, "cell", "hcp") == pytest.approx(
        917.0 * cell, rel=1e-15
    )
    # The legacy sphere mapping is packing independent by definition.
    assert fmm.particle_mass(917.0, R, "sphere", "hcp") == pytest.approx(
        fmm.particle_mass(917.0, R, "sphere", "sc"), rel=0
    )
    # Volume-equivalent mass radius of the HCP cell: (3 sqrt 2 / pi)^(1/3) r
    assert fmm.mass_radius_factor("hcp") == pytest.approx(
        (3.0 * math.sqrt(2.0) / math.pi) ** (1.0 / 3.0), rel=1e-14
    )
    # SC numbers are untouched (audited regression).
    assert fmm.particle_cell_volume(R, "sc") == pytest.approx(8.0 * R ** 3, rel=1e-15)
    assert fmm.mass_radius_factor("sc") == pytest.approx(fg.MASS_RADIUS_FACTOR, rel=1e-15)


def test_packing_aliases_and_rejection():
    for spelling in ("hcp", "HCP", "hexagonal", "hexagonal_close_packed", "abab"):
        assert fmm.normalise_packing(spelling) == "hcp"
    for spelling in ("sc", "simple-cubic", "cubic"):
        assert fmm.normalise_packing(spelling) == "sc"
    with pytest.raises(ValueError):
        fmm.normalise_packing("bcc")
    with pytest.raises(ValueError):
        fg.FractureGenConfig(packing="bcc")
    with pytest.raises(ValueError):
        fg.generate_particle_lattice(_box(0.4), R, packing="fcc")


# ------------------------------------------------------- fracture field glue ---


def test_fracture_field_carries_the_packing():
    faces = [
        fg.FailureFace(
            id=0,
            area=0.01,
            normal=torch.tensor([1.0, 0.0, 0.0]),
            centroid=torch.tensor([0.0, 0.0, 0.0]),
            G=100.0,
            released_energy=1.0,
        )
    ]
    for packing in ("sc", "hcp"):
        config = fg.FractureGenConfig(
            density=917.0, particle_radius=R, bond_stiffness=2000.0,
            domain_bounds=_box(0.6), packing=packing,
        )
        field = fg.generate_fracture_field(faces, config)
        assert field.packing == packing
        assert field.particle_cell_volume == pytest.approx(
            fmm.particle_cell_volume(R, packing), rel=1e-15
        )
        assert field.num_broken_bonds >= 1  # the face severs the mid plane
        assert field.total_mass == pytest.approx(
            field.num_particles * 917.0 * fmm.particle_cell_volume(R, packing), rel=1e-12
        )
        state = field.to_icedem3d_initial_state()
        assert state["positions"].shape == (field.num_particles, 3)


def test_hcp_field_is_a_legal_icedem3d_initial_state():
    """Shapes/dtypes/geometry cross-checks against the IceDEM3D contract.

    The engine's own cubic grid is rebuilt from ``DEM3DConfig`` in ``__init__``,
    so a foreign bond network cannot be swapped in without also rebuilding its
    derived per-pair arrays; that swap is out of scope here.  What is verified
    is that the HCP field satisfies every invariant the engine relies on: the
    bond list is one half-space of unique pairs, ``rest_lengths`` matches the
    geometry, there is no overlap, and every particle carries at most 12 bonds.
    """
    positions, bonds, rest = _lattice("hcp", 0.4)
    assert (bonds[:, 0] < bonds[:, 1]).all()
    assert len(torch.unique(bonds[:, 0] * len(positions) + bonds[:, 1])) == len(bonds)
    geometric = torch.linalg.vector_norm(
        positions[bonds[:, 1]] - positions[bonds[:, 0]], dim=1
    )
    assert torch.allclose(geometric, rest, rtol=0, atol=1e-15)
    assert float(rest.max()) <= 2.02 * R
    assert int(coordination_numbers(len(positions), bonds).max()) <= 12
    assert positions.shape[1] == 3 and positions.dtype == torch.float64
