"""Packing geometry and kernel-based local voidage.

The exact sphere-sphere intersection volume is the load-bearing piece of the
local-voidage estimator, so it is checked against closed forms (equal spheres,
tangency, containment) and against a brute-force Monte-Carlo integration.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.bedcoupling.closures import sphere_volume
from tensordem.bedcoupling.packing import (
    PackedBed,
    local_porosity,
    make_jittered_cubic_array,
    make_random_close_packing,
    porosity_statistics,
    sphere_intersection_volume,
)

D = 1.0e-3


def test_intersection_volume_closed_forms() -> None:
    r = 0.5 * D
    full = sphere_volume(D)
    assert float(sphere_intersection_volume(r, r, 0.0)) == pytest.approx(full)
    assert float(sphere_intersection_volume(r, r, D)) == pytest.approx(0.0, abs=1e-30)
    # equal spheres at half overlap: lens volume 5/16 pi r^3... use the integral
    s = D
    lens = math.pi * (2 * r - s) ** 2 * (s**2 + 4 * s * r) / (12 * s)
    assert float(sphere_intersection_volume(r, r, s)) == pytest.approx(lens)
    # sphere of radius 0.2 d fully inside the probe: the intersection is that sphere
    inside = 4.0 / 3.0 * math.pi * (0.2 * D) ** 3
    assert float(sphere_intersection_volume(D, 0.2 * D, 0.1 * D)) == pytest.approx(inside)


def test_intersection_volume_matches_monte_carlo() -> None:
    rng = np.random.default_rng(5)
    ra, rb, s = 1.0, 0.6, 1.2
    n = 400_000
    pts = rng.uniform(-ra, ra, size=(n, 3))
    inside_a = (pts**2).sum(axis=1) <= ra**2
    inside_b = ((pts - np.array([s, 0.0, 0.0])) ** 2).sum(axis=1) <= rb**2
    mc = inside_a.sum() / n * (2 * ra) ** 3
    overlap = inside_a & inside_b
    vol_mc = overlap.sum() / n * (2 * ra) ** 3
    # Monte-Carlo error is ~2e-3 in absolute volume at n = 4e5
    assert float(sphere_intersection_volume(ra, rb, s)) == pytest.approx(vol_mc, abs=4e-3)
    assert mc == pytest.approx(4 / 3 * math.pi * ra**3, rel=5e-3)


def test_local_porosity_of_a_single_particle_is_analytic() -> None:
    box = 20 * D
    pos = np.array([[box / 2, box / 2, box / 2]])
    probe = 3 * D
    eps = local_porosity(pos, D, box, probe)
    expected = 1.0 - sphere_volume(D) / (4.0 / 3.0 * math.pi * probe**3)
    assert float(eps[0]) == pytest.approx(expected, rel=1e-14)


def test_local_porosity_rejects_probe_too_large_for_minimum_image() -> None:
    box = 6 * D
    pos = np.array([[D, D, D]])
    with pytest.raises(ValueError):
        local_porosity(pos, D, box, 3 * D)


def test_jittered_array_has_prescribed_porosity_and_no_contacts() -> None:
    bed = make_jittered_cubic_array(5, D, porosity=0.8)
    assert bed.porosity == pytest.approx(0.8)
    assert bed.max_overlap == 0.0
    assert bed.minimum_separation() >= D * (1 - 1e-12)


def test_random_packing_geometry_and_porosity_window() -> None:
    bed = make_random_close_packing(800, D, 0.38, seed=12345, final_relax_iterations=6000)
    # porosity is exact by volume balance
    assert bed.porosity == pytest.approx(0.38, abs=1e-12)
    assert bed.n_particles == 800
    # admissible geometry: residual overlap below 1e-2 of a diameter
    assert bed.max_overlap <= 1e-2 * D
    # local voidage spreads by well under the 0.02 acceptance bound
    eps = local_porosity(bed.positions, D, bed.box_length, 2.5 * D)
    stats = porosity_statistics(eps, bed.porosity)
    assert 0.36 <= stats["mean"] <= 0.40
    assert stats["std"] <= 0.02
    assert stats["n_samples"] == 800


def test_packing_is_deterministic_for_a_fixed_seed() -> None:
    a = make_random_close_packing(300, D, 0.40, seed=9, final_relax_iterations=2000)
    b = make_random_close_packing(300, D, 0.40, seed=9, final_relax_iterations=2000)
    assert np.array_equal(a.positions, b.positions)
    c = make_random_close_packing(300, D, 0.40, seed=10, final_relax_iterations=2000)
    assert not np.allclose(a.positions, c.positions)


def test_packed_bed_summary_fields() -> None:
    bed = PackedBed(
        diameter=D, box_length=(8 * sphere_volume(D) / 0.4) ** (1 / 3), positions=np.zeros((8, 3))
    )
    assert bed.solid_fraction == pytest.approx(0.4)
    assert bed.porosity == pytest.approx(0.6)
    assert bed.volumes.shape == (8,)
