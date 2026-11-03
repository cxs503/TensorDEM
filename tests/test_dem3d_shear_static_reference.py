"""Tests for the rate-free static reference of the DEM3D direct-shear specimen."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.shear_specimen import (
    direct_shear_specimen_area,
    lattice_spacing,
    shear_modulus_closed_form,
)
from tensordem.shear_static import (
    affine_effective_area,
    bond_census,
    shear_stiffness_matrix,
    static_direct_shear,
)


def _specimen(nx=9, ny=5, nz=3, **overrides):
    values = dict(nx=nx, ny=ny, nz=nz, fix_bottom=True, fix_x_edges=False, drag=0.0)
    values.update(overrides)
    return DEM3DConfig(**values)


def test_affine_reaction_reproduced_by_bond_census_area():
    """Under affine shear ``R / A_eff`` equals ``C44 * gamma`` exactly."""
    cfg = _specimen()
    sim = IceDEM3D(cfg)
    gamma = 1.0e-6
    positions = sim.initial_positions.clone()
    positions[:, 0] += gamma * positions[:, 2]
    sim.positions = positions
    forces, _ = sim.forces(update_fracture=False)
    reaction = abs(float(forces[sim.fixed].sum(dim=0)[0]))
    a = lattice_spacing(cfg)
    c44 = shear_modulus_closed_form(cfg)
    tau = reaction / affine_effective_area(cfg)
    # exact in the linear limit; the residual is the axial spring's geometric
    # second-order term, which scales with gamma
    assert tau == pytest.approx(c44 * gamma, rel=1.0e-5)


def test_nominal_area_underestimates_affine_traction():
    """The legacy nominal area misses the bonds of the lateral boundary nodes."""
    cfg = _specimen()
    census = bond_census(cfg)
    ratio = census["weight"] / (7.0 / 3.0) / direct_shear_specimen_area(cfg)
    assert 0.7 < ratio < 0.85          # measured 0.7873 for 9x5
    assert ratio < 0.90                # small specimen: O(1/nx) deficit
    wide = replace(cfg, nx=21, ny=21)
    ratio_wide = bond_census(wide)["weight"] / (7.0 / 3.0) / direct_shear_specimen_area(wide)
    assert ratio_wide > ratio          # deficit shrinks with the cross-section
    assert ratio_wide < 1.0


def test_stiffness_matrix_matches_finite_difference_jacobian():
    """The assembled K must equal d(force)/du of the solver's own force routine."""
    cfg = _specimen(nx=4, ny=3, nz=3)
    sim = IceDEM3D(cfg)
    K = shear_stiffness_matrix(sim).toarray()
    eps = 1.0e-7
    n = 3 * len(sim.positions)
    base = sim.initial_positions.clone()
    columns = np.zeros((n, n))
    for dof in range(n):
        perturb = torch.zeros_like(base)
        perturb.view(-1)[dof] = eps
        sim.positions = base + perturb
        plus, _ = sim.forces(update_fracture=False)
        sim.positions = base - perturb
        minus, _ = sim.forces(update_fracture=False)
        columns[:, dof] = ((plus - minus).reshape(-1).numpy() / (2.0 * eps))
    sim.positions = base
    scale = np.abs(K).max()
    # dF/du = -K: the bond force on a node decreases as its neighbour moves away
    assert np.abs(columns + K).max() / scale < 1.0e-6


def test_static_equilibrium_is_reached_and_matches_solver_reaction():
    result = static_direct_shear(_specimen(), gamma=1.0e-3)
    assert result["reaction_relative_error"] < 1.0e-5
    assert result["free_residual_over_bond_force"] < 5.0e-2
    assert result["C44_Pa"] == pytest.approx(shear_modulus_closed_form(_specimen()))
    # effective-area normalisation always exceeds the nominal one
    assert result["G_effective_over_C44"] > result["G_nominal_over_C44"]
    assert result["G_effective_over_C44"] < 1.0     # free-surface relaxation remains


def test_static_response_is_linear_in_gamma():
    small = static_direct_shear(_specimen(), gamma=1.0e-5)
    large = static_direct_shear(_specimen(), gamma=1.0e-3)
    assert large["G_effective_over_C44"] == pytest.approx(
        small["G_effective_over_C44"], rel=1.0e-6)


def test_static_modulus_depends_on_specimen_shape_not_on_stiffness():
    """Scaling k scales G linearly; the normalised ratio is unchanged."""
    base = static_direct_shear(_specimen(), gamma=1.0e-3)
    stiffer = static_direct_shear(_specimen(bond_stiffness=10_000.0), gamma=1.0e-3)
    assert stiffer["G_effective_over_C44"] == pytest.approx(
        base["G_effective_over_C44"], rel=1.0e-9)
    assert stiffer["tau_effective_Pa"] == pytest.approx(
        5.0 * base["tau_effective_Pa"], rel=1.0e-9)
    taller = static_direct_shear(_specimen(nz=9), gamma=1.0e-3)
    assert taller["G_effective_over_C44"] < base["G_effective_over_C44"]


def test_layer_profile_is_monotone_and_pinned():
    result = static_direct_shear(_specimen(nz=9), gamma=1.0e-3)
    means = [row["mean_ux_over_gamma_h"] for row in result["profile"]]
    assert means[0] == pytest.approx(0.0, abs=1.0e-12)   # fixed bottom layer
    assert means[-1] == pytest.approx(1.0, rel=1.0e-9)   # prescribed top layer
    assert all(b >= a - 1.0e-12 for a, b in zip(means, means[1:]))
    # free lateral surfaces allow a non-affine spread inside each layer
    assert max(row["spread_ux_over_gamma_h"] for row in result["profile"]) > 0.01


def test_invalid_gamma_rejected():
    with pytest.raises(ValueError):
        static_direct_shear(_specimen(), gamma=0.0)
    with pytest.raises(ValueError):
        static_direct_shear(_specimen(), gamma=float("nan"))