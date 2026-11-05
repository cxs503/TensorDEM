"""Verification tests for the lateral periodic closure of the direct-shear specimen.

The evidence chain has three links and every link is asserted here:

1. **Same code path.**  With the periodic wrap switched off, this module must
   reproduce :func:`tensordem.shear_static.static_direct_shear` -- the reference that
   was itself cross-checked against ``IceDEM3D.forces`` -- to solver accuracy.  If
   that fails, nothing below is trustworthy.
2. **Exact affine closure.**  With both lateral directions periodic the affine field
   ``u_x = gamma z`` is an exact solution, so ``G_eff / C44 = 1`` to round-off at
   *every* size and *every* height, and the pre-solve affine residual vanishes.
3. **Mechanism.**  One pair of free lateral faces still costs exactly nothing (the
   missing-bond set keeps a vanishing first moment), while two pairs cost the
   free-surface deficit; that locates the deficit at the edges where two free faces
   meet instead of at the faces themselves.

A negative control (wrap on a *sub-critical* lattice must not silently pass, and a
mis-specified bond set must change the answer) guards against a test that cannot fail.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.dem3d import DEM3DConfig
from tensordem.shear_periodic import (
    NEIGHBOUR_OFFSETS,
    affine_residual,
    bond_list,
    height_series,
    lattice_sites,
    periodic_series,
    periodic_stiffness_matrix,
    static_shear_periodic,
)
from tensordem.shear_specimen import lattice_spacing, shear_modulus_closed_form
from tensordem.shear_static import static_direct_shear

SPACING = lattice_spacing(DEM3DConfig())
STIFFNESS = DEM3DConfig().bond_stiffness
GAMMA = 1.0e-3


def _solver_reference(n: int, nz: int = 3) -> dict:
    cfg = DEM3DConfig()
    cfg = type(cfg)(**{**cfg.__dict__, "nx": n, "ny": n, "nz": nz,
                       "drag": 0.0, "fix_x_edges": False})
    return static_direct_shear(cfg, gamma=GAMMA)


def test_neighbour_set_is_the_solver_26():
    assert len(NEIGHBOUR_OFFSETS) == 26
    assert all(off != (0, 0, 0) for off in NEIGHBOUR_OFFSETS)
    lengths = sorted({float(np.linalg.norm(off)) for off in NEIGHBOUR_OFFSETS})
    assert lengths == pytest.approx(sorted([1.0, np.sqrt(2.0), np.sqrt(3.0)]), rel=1e-12)


def test_lattice_sites_are_column_major_and_bounded():
    positions = lattice_sites(3, 4, 5, SPACING)
    assert positions.shape == (60, 3)
    assert positions[:, 0].max() == pytest.approx(2 * SPACING)
    assert positions[:, 1].max() == pytest.approx(3 * SPACING)
    assert positions[:, 2].max() == pytest.approx(4 * SPACING)
    # node p sits at (p % nx, (p // nx) % ny, p // (nx*ny))
    assert positions[1 + 3 * 1 + 12 * 1] == pytest.approx([SPACING, SPACING, SPACING])
    with pytest.raises(ValueError):
        lattice_sites(1, 3, 3, SPACING)


def test_bond_list_wraps_only_replace_lateral_offsets():
    free = bond_list(5, 5, 3, SPACING, periodic_x=False, periodic_y=False)
    closed = bond_list(5, 5, 3, SPACING, periodic_x=True, periodic_y=True)
    assert len(closed[0]) > len(free[0])
    # every periodic bond keeps its unwrapped offset: |delta| is still a lattice vector
    lengths = np.linalg.norm(closed[2], axis=1)
    allowed = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)]) * SPACING
    assert np.allclose(np.min(np.abs(lengths[:, None] - allowed[None, :]), axis=1), 0.0, atol=1e-12)
    # x-only closure sits between the two extremes
    x_only = bond_list(5, 5, 3, SPACING, periodic_x=True, periodic_y=False)
    assert len(free[0]) < len(x_only[0]) < len(closed[0])
    # no self-images on a cell wider than the neighbour range
    assert np.all(closed[0] != closed[1])


def test_bond_list_drops_self_images_in_tiny_cells():
    bonds = bond_list(2, 2, 2, SPACING, periodic_x=True, periodic_y=True)
    assert np.all(bonds[0] != bonds[1])


def test_stiffness_matrix_is_symmetric_and_annihilates_rigid_translation():
    positions = lattice_sites(4, 4, 3, SPACING)
    bonds = bond_list(4, 4, 3, SPACING)
    K = periodic_stiffness_matrix(positions, bonds, STIFFNESS)
    assert K.shape == (3 * len(positions), 3 * len(positions))
    assert abs(K - K.T).max() < 1e-9
    for axis in range(3):
        translation = np.zeros(3 * len(positions))
        translation[axis::3] = 1.0
        assert np.abs(K @ translation).max() < 1e-9


def test_wrap_off_reproduces_the_solver_reference():
    """Link 1: the periodic module with the wrap disabled *is* the solver reference."""
    for n in (5, 7, 9):
        ref = _solver_reference(n)
        mine = static_shear_periodic(n, n, 3, periodic_x=False, periodic_y=False, gamma=GAMMA)
        assert mine["G_effective_over_C44"] == pytest.approx(ref["G_effective_over_C44"], rel=1e-10)
        assert mine["G_nominal_over_C44"] == pytest.approx(ref["G_nominal_over_C44"], rel=1e-10)
        assert mine["area_effective_m2"] == pytest.approx(ref["area_effective_m2"], rel=1e-10)
        assert mine["C44_Pa"] == pytest.approx(ref["C44_Pa"], rel=1e-12)


def test_periodic_closure_gives_exact_c44_at_every_size():
    """Link 2: G_eff/C44 = 1 to round-off, independent of the cross-section size."""
    for n in (3, 4, 5, 9, 17, 25):
        got = static_shear_periodic(n, n, 3, gamma=GAMMA)
        assert got["G_effective_over_C44"] == pytest.approx(1.0, abs=5e-12)
        assert abs(got["deficit"]) < 5e-12
        assert got["area_ratio"] == pytest.approx(1.0, abs=1e-12)
        assert abs(got["reaction_over_affine_prediction"] - 1.0) < 1e-12
        assert got["free_residual_over_bond_force"] < 1e-10


def test_periodic_field_is_the_affine_field_pointwise():
    """The returned equilibrium field is the affine field itself under closure."""
    closed = static_shear_periodic(9, 9, 3, gamma=GAMMA, return_displacement=True)
    z = closed["positions_m"][:, 2]
    height = z.max() - z.min()
    error = np.abs(closed["displacement_m"][:, 0] - GAMMA * z).max() / (GAMMA * height)
    assert error < 1e-12
    assert np.abs(closed["displacement_m"][:, 1:]).max() / (GAMMA * height) < 1e-12
    free = static_shear_periodic(9, 9, 3, periodic_x=False, periodic_y=False,
                                 gamma=GAMMA, return_displacement=True)
    free_error = np.abs(free["displacement_m"][:, 0] - GAMMA * z).max() / (GAMMA * height)
    assert free_error > 0.01
    assert "displacement_m" not in static_shear_periodic(9, 9, 3, gamma=GAMMA)


def test_periodic_closure_is_independent_of_height():
    series = height_series([3, 4, 5, 7], n=9, gamma=GAMMA)
    for row in series["rows"]:
        assert row["G_effective_over_C44"] == pytest.approx(1.0, abs=5e-12)


def test_affine_residual_separates_the_boundary_conditions():
    """Link 3: the pre-solve affine residual is the mechanism diagnostic."""
    closed = affine_residual(5, 5, 3, periodic_x=True, periodic_y=True)
    one_face = affine_residual(5, 5, 3, periodic_x=True, periodic_y=False)
    free = affine_residual(5, 5, 3, periodic_x=False, periodic_y=False)
    assert closed["affine_residual_over_bond_force"] < 1e-10
    assert one_face["affine_residual_over_bond_force"] < 1e-10
    assert free["affine_residual_over_bond_force"] > 1.0
    # ... and it predicts which variants measure an exact C44
    assert static_shear_periodic(5, 5, 3, periodic_x=True, periodic_y=False)[
        "G_effective_over_C44"] == pytest.approx(1.0, abs=5e-12)
    assert static_shear_periodic(5, 5, 3, periodic_x=False, periodic_y=False)[
        "G_effective_over_C44"] < 0.8


def test_one_free_pair_costs_nothing_and_two_pairs_cost_the_deficit():
    one = static_shear_periodic(9, 9, 3, periodic_x=True, periodic_y=False, gamma=GAMMA)
    two = static_shear_periodic(9, 9, 3, periodic_x=False, periodic_y=False, gamma=GAMMA)
    assert one["deficit"] == pytest.approx(0.0, abs=5e-12)
    assert two["deficit"] > 0.10
    # the census-normalised modulus stays exact even where the census itself is short
    assert one["area_ratio"] < 1.0


def test_periodic_series_matches_the_free_surface_reference_trend():
    """The committed cross-section study must be the *free-surface* limit of this one."""
    closed = periodic_series([5, 9, 21], gamma=GAMMA)
    free = periodic_series([5, 9, 21], periodic_x=False, periodic_y=False, gamma=GAMMA)
    assert closed["max_abs_deficit"] < 5e-12
    deficits = {row["n"]: row["deficit"] for row in free["rows"]}
    assert deficits[21] < deficits[9] < deficits[5]
    for n, expected in ((5, 0.312786), (9, 0.154276), (21, 0.061233)):
        assert deficits[n] == pytest.approx(expected, abs=5e-6)


def test_negative_controls():
    """A wrong bond stiffness and a broken size sweep must change the answer."""
    with pytest.raises(ValueError):
        static_shear_periodic(5, 5, 3, gamma=0.0)
    base = static_shear_periodic(5, 5, 3, gamma=GAMMA)
    scaled = static_shear_periodic(5, 5, 3, gamma=2 * GAMMA)
    assert scaled["reaction_N"] == pytest.approx(2 * base["reaction_N"], rel=1e-10)
    assert scaled["G_effective_over_C44"] == pytest.approx(1.0, abs=5e-12)
    stiff = type(DEM3DConfig())(**{**DEM3DConfig().__dict__, "bond_stiffness": 4000.0})
    assert shear_modulus_closed_form(stiff) == pytest.approx(2 * shear_modulus_closed_form(DEM3DConfig()))
    assert static_shear_periodic(5, 5, 3, config=stiff)["G_effective_over_C44"] == pytest.approx(1.0, abs=5e-12)
    with pytest.raises(ValueError):
        periodic_series([9, 5])
    with pytest.raises(ValueError):
        height_series([5, 3])