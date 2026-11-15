"""Round 26 (m22550): polydispersity as the physical symmetry break of the 2-D
plane-shear device, and the periodic-image defect fix in the seed relaxation.

Round 25 established that a monodisperse disk bed re-crystallises during the
constant-P compaction whatever the seed (psi6 -> 0.91-0.96, mu on the
frictionless branch).  These tests pin the *device*, not the physics verdict:
the size spread must leave the ordinates' definitions intact (mean radius
invariant, bounded draw), the seeded state must be admissible *including
periodic images in x*, and the polydisperse seed must actually be disordered.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "plane_shear_mu_i"))

import plane_shear_2d as ps  # noqa: E402


def _case(**kw) -> ps.PlaneShear2DCase:
    base = dict(I_target=0.05, n_particles=458, width=0.30, gap=0.15,
                settle_time_units=40.0, shear_settle_strain=0.5,
                strain_measure=1.0, init_mode="random", seed=0)
    base.update(kw)
    return ps.PlaneShear2DCase(**base)


# ---------------------------------------------------------------- the draw ----
@pytest.mark.parametrize("w", [0.0, 0.1, 0.25, 0.5])
def test_radius_draw_preserves_mean_radius_and_is_bounded(w):
    """The ordinates use ``case.radius``: the draw must not move the mean."""
    case = _case(polydispersity=w)
    r = ps._draw_radii(case, 458)
    assert r.shape == (458,)
    assert np.all(r > 0.0)
    assert r.mean() == pytest.approx(case.radius, rel=1e-12)
    if w == 0.0:
        assert np.all(r == case.radius)
    else:
        # bounded uniform draw: exactly inside radius*(1 -/+ w), and the size
        # contrast that the literature bound refers to is (1+w)/(1-w)
        assert r.min() >= case.radius * (1.0 - w) - 1e-12
        assert r.max() <= case.radius * (1.0 + w) + 1e-12
        assert r.max() / r.min() <= (1.0 + w) / (1.0 - w) + 1e-9


def test_draw_is_reproducible_and_streams_are_decorrelated():
    case = _case(polydispersity=0.5)
    a = ps._draw_radii(case, 200, stream=1)
    b = ps._draw_radii(case, 200, stream=1)
    c = ps._draw_radii(case, 200, stream=2)
    assert np.array_equal(a, b), "same seed/stream must reproduce the radii"
    assert not np.array_equal(a, c), "positional and radius streams must differ"


# ------------------------------------------------------------ the contract ----
@pytest.mark.parametrize("w", [-0.01, 0.51, 1.0])
def test_out_of_range_polydispersity_is_rejected(w):
    with pytest.raises(ValueError, match="polydispersity"):
        _case(polydispersity=w).validate()


def test_literature_boundary_w_half_is_accepted():
    _case(polydispersity=0.5).validate()


# ------------------------------------------------ the periodic-image defect ----
def test_relax_removes_overlaps_across_the_periodic_boundary():
    """A pair straddling x = 0 (and x = W) is a contact, not a coincidence.

    The pre-round-26 acceptance test built a non-periodic k-d tree, so such a
    pair was invisible and an overlapping seed could be certified clean.
    """
    width, radius = 0.10, 0.005
    # two disks 0.8 d apart *through* the periodic boundary (x = 0.002, 0.098)
    pos = np.array([[0.002, 0.05], [0.098, 0.05], [0.050, 0.05]])
    rad = np.full(3, radius)
    out = ps._relax_overlaps(pos, rad, width, 0.0, 0.10)
    dx = out[:, 0][:, None] - out[:, 0][None, :]
    dx -= width * np.round(dx / width)
    dy = out[:, 1][:, None] - out[:, 1][None, :]
    dist = np.hypot(dx, dy)
    iu = np.triu_indices(3, 1)
    assert np.all(dist[iu] >= 2.0 * radius * (1.0 - 1e-3)), \
        "periodic images must be resolved by the projection"


# ------------------------------------------------- the seed is admissible ----
@pytest.mark.parametrize("w", [0.0, 0.25, 0.5])
def test_seeded_cell_is_overlap_free_at_production_size(w):
    cell = ps.RoughWallShearCell(_case(polydispersity=w))
    pos, width = cell.system.pos, cell.case.width
    rad = cell.system.radii
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dx -= width * np.round(dx / width)
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    dij = rad[:, None] + rad[None, :]
    dist = np.hypot(dx, dy)
    iu = np.triu_indices(len(pos), 1)
    bad = int(np.sum(dist[iu] < dij[iu] * (1.0 - 1e-3)))
    assert bad == 0, f"{bad} overlapping pairs in the seeded {w=} cell"
    # the mean radius (hence I, gap_eff, phi definitions) is untouched
    assert cell.system.radii[cell.free_idx].mean() == pytest.approx(
        cell.case.radius, rel=1e-12)


# ------------------------------------------------------- structure contrast ----
def test_polydisperse_seed_is_less_ordered_than_the_monodisperse_one():
    mono = ps.RoughWallShearCell(_case(polydispersity=0.0))
    poly = ps.RoughWallShearCell(_case(polydispersity=0.5))
    o_mono = ps.hexatic_order(mono.system.pos[mono.free_idx],
                              mono.system.radii[mono.free_idx], mono.case.width)
    o_poly = ps.hexatic_order(poly.system.pos[poly.free_idx],
                              poly.system.radii[poly.free_idx], poly.case.width)
    assert o_poly["psi6_mean"] < o_mono["psi6_mean"]
    assert o_poly["psi6_hexatic_fraction"] < 0.5


def test_hexatic_order_scalar_and_array_radius_agree():
    """Backward compatibility: rounds 24/25 passed a scalar radius."""
    cell = ps.RoughWallShearCell(_case(polydispersity=0.0))
    p = cell.system.pos[cell.free_idx]
    a = ps.hexatic_order(p, cell.case.radius, cell.case.width)
    b = ps.hexatic_order(p, np.full(len(p), cell.case.radius), cell.case.width)
    assert a == b
