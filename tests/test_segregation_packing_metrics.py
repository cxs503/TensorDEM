"""Specimen generation, segregation indices and the mass/moment audit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.segregation.metrics import (
    BedStatistics,
    mass_moment_audit,
    particle_measure,
    rise_velocity,
    segregation_indices,
    zonal_masses,
)
from tensordem.segregation.packing import (
    bidisperse_radii,
    large_volume_fraction,
    mixture_area,
    random_sequential_pack,
)


def test_bidisperse_radii_realises_the_requested_volume_fraction():
    for ratio in (1.5, 2.0, 3.0, 4.0):
        for frac in (0.2, 0.4):
            radii, is_large = bidisperse_radii(20000, ratio, frac, small_radius=1e-3, seed=1)
            assert abs(large_volume_fraction(radii, is_large) - frac) < 5e-3
            assert radii[is_large].min() == pytest.approx(ratio * 1e-3)
            assert radii[~is_large].max() == pytest.approx(1e-3)


def test_ratio_one_is_geometrically_monodisperse():
    radii, is_large = bidisperse_radii(1000, 1.0, 0.3, seed=4)
    assert np.allclose(radii, 1e-3)
    assert 0.25 < is_large.mean() < 0.35  # random colouring only


def test_random_sequential_pack_is_overlap_free_and_deterministic():
    radii, _ = bidisperse_radii(400, 3.0, 0.3, seed=2)
    width, height = 0.12, 0.10
    pos = random_sequential_pack(radii, width, height, seed=7, target_fraction=0.35)
    assert np.isfinite(pos).all()
    assert (pos[:, 0] >= radii - 1e-12).all() and (pos[:, 0] <= width - radii + 1e-12).all()
    assert (pos[:, 1] >= radii - 1e-12).all() and (pos[:, 1] <= height - radii + 1e-12).all()
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert float(np.min(d - (radii[:, None] + radii[None, :]))) > -1e-12
    pos2 = random_sequential_pack(radii, width, height, seed=7, target_fraction=0.35)
    assert np.array_equal(pos, pos2)
    pos3 = random_sequential_pack(radii, width, height, seed=8, target_fraction=0.35)
    assert not np.array_equal(pos, pos3)


def test_random_sequential_pack_refuses_an_overfilled_box():
    radii, _ = bidisperse_radii(400, 2.0, 0.3, seed=2)
    with pytest.raises(ValueError):
        random_sequential_pack(radii, 0.02, 0.02, seed=1, target_fraction=0.6)


def _synthetic_stats(n=40000, segregation=0.0, seed=0, h_bed=0.11):
    edges = np.linspace(0.0, 0.16, 161)
    stats = BedStatistics(z_edges=edges, x_edges=np.linspace(0.0, 0.4, 9))
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0005, h_bed, n)
    x = rng.uniform(0.0, 0.4, n)
    measure = np.full(n, 1.0 / n)
    # linear depth-dependent enrichment of the large species
    p_large = 0.3 + segregation * (z / h_bed - 0.5)
    is_large = rng.random(n) < np.clip(p_large, 0.0, 1.0)
    stats.add(np.stack([x, z], axis=1), measure, is_large)
    return stats


def test_segregation_index_is_zero_for_a_random_mixture():
    stats = _synthetic_stats(segregation=0.0)
    out = segregation_indices(stats, h_ref_m=0.11)
    assert abs(out["S_fine"]) < 0.02
    assert abs(out["S_bn"]) < 0.08


def test_segregation_index_grows_monotonically_with_enrichment():
    vals = [segregation_indices(_synthetic_stats(segregation=s), h_ref_m=0.11)["S_bn"]
            for s in (0.0, 0.2, 0.4, 0.8)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert vals[-1] > 0.2


def test_top_layer_definition_matches_the_geometry():
    stats = _synthetic_stats(segregation=0.0)
    out = segregation_indices(stats, top_fraction=0.25, h_ref_m=0.11)
    assert out["z_cut_m"] == pytest.approx(0.75 * 0.11, rel=1e-9)
    assert out["n_top_bins"] > 5
    assert out["h_profile_m"] > 0.0


def test_mass_and_moment_audit_is_exact():
    stats = _synthetic_stats(segregation=0.0)
    out = mass_moment_audit(stats, density=1000.0, thickness=1.0,
                            reference_measure=stats.total_measure)
    assert out["mass_conservation_rel"] < 1e-12
    assert out["mass_conservation_zone_rel"] < 1e-12
    assert out["mass_conservation_vs_reference_rel"] < 1e-12
    assert out["z_com_consistency_rel"] < 1e-12
    assert out["z_com_zone_rel"] < 1e-12
    # the centre of mass of a uniform slab is half way up
    assert out["z_com_m"] == pytest.approx(0.5 * 0.11, rel=0.02)


def test_zonal_masses_reconstruct_the_total():
    stats = _synthetic_stats(segregation=0.0)
    zones = zonal_masses(stats, density=1000.0, thickness=1.0)
    audit = mass_moment_audit(stats, density=1000.0, thickness=1.0)
    assert float(zones["total_mass_kg"]) == pytest.approx(audit["mass_kg"], rel=1e-12)
    assert zones["mass_kg"].shape == (8,)
    assert (zones["mass_kg"] > 0.0).all()


def test_rise_velocity_recovers_a_known_slope():
    t = np.linspace(0.0, 1.0, 101)
    z = 0.02 + 3.0e-3 * t
    out = rise_velocity(t, z, t_start=0.2)
    assert out["v_up_m_s"] == pytest.approx(3.0e-3, rel=1e-9)
    assert out["r2"] == pytest.approx(1.0, abs=1e-12)
    assert out["n_samples"] == 81


def test_particle_measure_is_the_disk_area():
    radii = np.array([1e-3, 2e-3])
    assert np.allclose(particle_measure(radii, 2), math.pi * radii**2)
    assert np.allclose(particle_measure(radii, 3), 4.0 / 3.0 * math.pi * radii**3)
    with pytest.raises(ValueError):
        particle_measure(radii, 4)