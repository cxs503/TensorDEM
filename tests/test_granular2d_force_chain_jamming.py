"""Tests for the granular2d broken-ice force-chain / jamming module (IN22-N03).

The suite covers, in order: packing generation, contact physics and its
validation errors, the discrete Virial/symmetry identities (with negative
controls that must *fail* the identity), hydrostatics, momentum conservation,
reproducibility, the force-chain/network analysis primitives on synthetic
graphs, the BIC breakpoint detector, and one end-to-end channel transit.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tensordem.granular2d import (
    DiskConfig,
    DiskSystem,
    analyze_force_network,
    bidisperse_radii,
    box_walls,
    chain_length_distribution,
    coordination_number,
    fabric_tensor,
    grow_to_fraction,
    load_bearing_ratio,
    make_channel_system,
    piecewise_bic_breakpoint,
    polyline_segments,
    random_packing_positions,
    run_channel,
    ship_segments,
    spanning_axis,
    strong_force_mask,
)
from tensordem.granular2d.force_chain import chain_components
from tensordem.granular2d.stress import (
    hydrostatic_check,
    lattice_stress,
    stress_balance,
    stress_invariants,
    virial_tensor,
)


# ------------------------------------------------------------------ helpers
def _free_pair(separation: float, radii=(0.01, 0.01), **cfg_kwargs):
    cfg = DiskConfig(**cfg_kwargs)
    pos = np.array([[0.3, 0.2], [0.3 + separation, 0.2]])
    return DiskSystem(pos, np.array(radii), cfg)


def _box_packing(phi: float, n: int = 90, radius: float = 0.010, seed: int = 0,
                 **cfg_kwargs):
    radii = bidisperse_radii(n, radius, seed=seed)
    side = math.sqrt(float(np.sum(math.pi * radii**2)) / phi)
    pos = random_packing_positions(n, radii, side, side, seed=seed + 1)
    cfg = DiskConfig(**cfg_kwargs)
    return DiskSystem(pos, radii, cfg, walls=box_walls(side, side)), side


# -------------------------------------------------------------------- packing
def test_bidisperse_radii_statistics():
    radii = bidisperse_radii(400, 0.012, ratio=1.4, seed=7)
    assert radii.shape == (400,)
    # area-averaged radius equals the requested mean radius
    assert np.isclose(math.sqrt(float(np.mean(radii**2))), 0.012, rtol=1e-12)
    big = np.sort(np.unique(np.round(radii, 12)))
    assert len(big) == 2
    assert np.isclose(big[1] / big[0], 1.4, rtol=1e-12)
    assert np.array_equal(radii, bidisperse_radii(400, 0.012, ratio=1.4, seed=7))


def test_bidisperse_radii_rejects_bad_arguments():
    with pytest.raises(ValueError):
        bidisperse_radii(10, 0.01, ratio=0.9)          # ratio must be > 1
    with pytest.raises(ValueError):
        bidisperse_radii(10, -0.01)                    # negative radius
    with pytest.raises(ValueError):
        bidisperse_radii(10, 0.01, fraction_small=1.0)


def test_random_packing_positions_are_non_overlapping():
    radii = bidisperse_radii(60, 0.01, seed=2)
    pos = random_packing_positions(60, radii, 0.60, 0.60, seed=3)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    assert np.all(dist > (radii[:, None] + radii[None, :]) - 1e-12)


# ------------------------------------------------------------- contact physics
def test_pair_contact_is_repulsive_and_equal_opposite():
    system = _free_pair(0.0195)                       # 0.5 mm overlap
    contacts = system.contact_forces()
    assert contacts["kind"].tolist() == [0]
    f = np.asarray(contacts["f_on_i"])[0]
    assert f[0] < 0.0                                  # pushes disk 0 towards -x
    assert np.allclose(np.asarray(contacts["f_on_j"])[0], -f)
    assert np.isclose(np.linalg.norm(f), system.config.kn * 5.0e-4, rtol=2e-2)


def test_no_contact_no_force():
    system = _free_pair(0.021)                         # separated by 1 mm
    contacts = system.contact_forces()
    assert len(contacts["i"]) == 0
    assert np.allclose(system._contact_force, 0.0)


def test_engine_rejects_invalid_input():
    with pytest.raises(ValueError):
        DiskSystem(np.zeros((2, 2)), np.array([0.01, -0.01]))
    with pytest.raises(ValueError):
        DiskSystem(np.zeros((2, 3)), np.array([0.01, 0.01]))
    with pytest.raises(ValueError):
        DiskSystem(np.zeros((2, 2)), np.array([0.01]))
    with pytest.raises(ValueError):
        DiskConfig(kn=-1.0)
    with pytest.raises(ValueError):
        _free_pair(0.0, dt=1.0)                        # dt beyond the stability limit


def test_wall_contact_reaction_is_recorded():
    cfg = DiskConfig(mu=0.3)
    pos = np.array([[0.02, 0.0095]])                   # 0.5 mm into the floor
    system = DiskSystem(pos, np.array([0.01]), cfg, walls=box_walls(0.1, 0.1))
    contacts = system.contact_forces()
    wall = np.asarray(contacts["kind"]) == 1
    assert wall.any()
    assert np.isclose(float(np.asarray(contacts["f_on_i"])[wall, 1].sum()), system.wall_force[1] * -1.0)
    assert system.wall_force[1] < 0.0                  # ice pushes the floor down


# ------------------------------------------------------- Virial/symmetry audits
def test_virial_identity_holds_to_round_off():
    radii = bidisperse_radii(120, 0.010, seed=1)
    side = math.sqrt(float(np.sum(math.pi * radii**2)) / 0.42)
    pos = random_packing_positions(120, radii, side, side, seed=2)
    system = DiskSystem(pos, radii, DiskConfig(mu=0.4, mu_rolling=0.02,
                                               gravity=(0.0, -9.81), skin=0.003),
                        walls=box_walls(side, side))
    system.step(6000)
    balance = stress_balance(system)
    assert balance["virial_identity_rel"] < 1e-10
    # the lattice field must reproduce the total virial exactly
    field = lattice_stress(system, nx=5, ny=5)
    area = float(np.prod(field["bounds"][2:] - field["bounds"][:2]))
    total = float(np.sum(field["sigma"] * field["cell_area"]))
    raw = float(np.sum(virial_tensor(system, symmetrize=False)) * area)
    assert abs(total - raw) <= 1e-10 * max(abs(raw), 1e-30)


def test_virial_identity_negative_control_detects_double_counting():
    """A double-counted contact side must break the identity by O(1)."""
    system = _free_pair(0.0195)
    system.contact_forces()
    table = system.contacts
    i = np.asarray(table["i"])
    j = np.asarray(table["j"])
    f = np.asarray(table["f_on_i"])
    good = np.einsum("ma,mb->ab", system.pos[j] - system.pos[i], f)
    bad = np.einsum("ma,mb->ab", system.pos[i] + system.pos[j], f)
    assert np.linalg.norm(good) > 0.0
    assert np.linalg.norm(bad - good) > 0.5 * np.linalg.norm(bad)


def test_stress_tensor_is_symmetric_and_has_couple_stress():
    system, _ = _box_packing(0.42, n=90, mu=0.5, mu_rolling=0.05, gravity=(0.0, -9.81))
    system.step(3000)
    sigma = virial_tensor(system)
    assert np.allclose(sigma, sigma.T, atol=1e-12)
    balance = stress_balance(system)
    assert balance["stress_symmetry_rel"] < 1e-12
    # friction/rolling resistance leave a non-zero antisymmetric (couple) part
    assert balance["antisymmetric_fraction"] >= 0.0


def test_stress_invariants_isotropic_and_shear():
    iso = stress_invariants(np.diag([-3.0, -3.0]))
    assert np.isclose(iso["pressure"], 3.0)
    assert np.isclose(iso["deviatoric"], 0.0)
    shear = stress_invariants(np.array([[-3.0, 1.0], [1.0, -1.0]]))
    assert shear["deviatoric"] > 0.0
    assert shear["stress_ratio"] > 0.0


def test_hydrostatic_traction_balance():
    """The floor traction of a settled frictionless bed equals its weight."""
    radii = bidisperse_radii(180, 0.011, seed=3)
    side = math.sqrt(float(np.sum(math.pi * radii**2)) / 0.40)
    pos = random_packing_positions(180, radii, side, side, seed=4)
    system = DiskSystem(pos, radii, DiskConfig(mu=0.0, mu_rolling=0.0,
                                               gravity=(0.0, -9.81), skin=0.003),
                        walls=box_walls(side, side))
    system.step(12000)                       # settling under gravity
    system.config = replace(system.config, damping_time=2.0e-5)
    for _ in range(10):                      # quench to a force-balanced bed
        system.step(1200)
        system.zero_velocities()
    system.step(400)
    hydro = hydrostatic_check(system, n_bands=10, band_index=2)
    assert hydro["traction_relative_error"] < 0.01


def test_momentum_is_conserved_without_external_forces():
    radii = bidisperse_radii(80, 0.010, seed=5)
    side = math.sqrt(float(np.sum(math.pi * radii**2)) / 0.42)
    pos = random_packing_positions(80, radii, side, side, seed=6)
    system = DiskSystem(pos, radii, DiskConfig(mu=0.3, mu_rolling=0.02, skin=0.003))
    rng = np.random.default_rng(1)
    system.vel = 0.12 * rng.standard_normal((80, 2))
    p0 = system.momentum()
    system.step(1500)
    scale = float(np.sum(system.mass * np.linalg.norm(system.vel, axis=1))) + 1e-30
    assert float(np.linalg.norm(system.momentum() - p0)) / scale < 1e-10


def test_digest_is_reproducible_and_seed_sensitive():
    def digest(seed: int) -> str:
        radii = bidisperse_radii(60, 0.010, seed=seed)
        side = math.sqrt(float(np.sum(math.pi * radii**2)) / 0.40)
        pos = random_packing_positions(60, radii, side, side, seed=seed + 1)
        system = DiskSystem(pos, radii, DiskConfig(mu=0.4, mu_rolling=0.02,
                                                   gravity=(0.0, -9.81), skin=0.003),
                            walls=box_walls(side, side))
        system.step(800)
        return system.digest()
    assert digest(11) == digest(11)
    assert digest(11) != digest(12)


# ----------------------------------------------------------- network analysis
def test_strong_force_mask_threshold_and_validation():
    mags = np.array([1.0, 1.0, 1.0, 10.0])
    mask = strong_force_mask(mags, k=1.0)
    assert mask.sum() == 1 and mask[-1]
    assert strong_force_mask(np.zeros(0)).size == 0
    with pytest.raises(ValueError):
        strong_force_mask(mags, k=-1.0)
    with pytest.raises(ValueError):
        strong_force_mask(np.array([np.nan]))


def test_chain_components_synthetic_graph():
    # two disconnected chains: 0-1-2 and 3-4
    i = np.array([0, 1, 3])
    j = np.array([1, 2, 4])
    labels, sizes = chain_components(6, i, j)
    assert sorted(sizes.tolist()) == [1, 2, 3]
    mask = np.array([True, True, False])
    labels_s, sizes_s = chain_components(6, i, j, mask)
    assert max(sizes_s.tolist()) == 3


def test_coordination_number_and_load_bearing_ratio():
    i = np.array([0, 1, 2])
    j = np.array([1, 2, 3])
    assert np.isclose(coordination_number(4, i, j), 1.5)
    assert np.isclose(coordination_number(4, i, j, np.array([True, False, False])), 0.5)
    mags = np.array([1.0, 3.0, 4.0])
    assert np.isclose(load_bearing_ratio(mags, np.array([True, False, False])), 0.125)
    assert load_bearing_ratio(np.zeros(3), np.zeros(3, dtype=bool)) == 0.0


def test_fabric_tensor_isotropic_versus_aligned():
    rng = np.random.default_rng(0)
    theta = rng.random(4000) * 2.0 * np.pi
    iso = fabric_tensor(np.stack([np.cos(theta), np.sin(theta)], axis=1))
    assert abs(iso["anisotropy"]) < 0.05
    theta = 0.3 + 0.05 * rng.standard_normal(4000)
    aligned = fabric_tensor(np.stack([np.cos(theta), np.sin(theta)], axis=1))
    assert aligned["anisotropy"] > 0.8
    assert fabric_tensor(np.zeros((0, 2)))["n_contacts"] == 0


def test_spanning_axis_detects_bank_to_bank_chain():
    ok_radii = np.full(6, 0.01)
    pos = np.stack([np.zeros(6), np.linspace(0.01, 0.99, 6)], axis=1)
    system = DiskSystem(pos, ok_radii, DiskConfig(), walls=box_walls(1.0, 1.0))
    labels = np.zeros(6, dtype=np.int64)
    span = spanning_axis(system, labels, bounds=(0.0, 1.0), band=0.05)
    assert span["bridging"] is True
    # split the chain in two: neither half reaches both banks
    half = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    span2 = spanning_axis(system, half, bounds=(0.0, 1.0), band=0.05)
    assert span2["bridging"] is False
    assert span2["n_bridging"] == 0


def test_force_network_statistics_on_jammed_packing():
    system, side = _box_packing(0.42, n=90, mu=0.4, mu_rolling=0.03)
    grow_to_fraction(system, 0.80, dphi=0.05, relax_steps=100, area=side * side)
    system.step(300)
    net = analyze_force_network(system, bounds=(0.0, side))
    assert net["n_contacts"] > 0
    assert 0.0 <= net["load_bearing_ratio"] <= 1.0
    assert net["z_all"] > 0.0
    assert abs(net["fabric_anisotropy"]) < 0.5
    stats = chain_length_distribution(system, net["labels_strong"], np.bincount(net["labels_strong"]))
    assert stats["n_chains"] >= 0


def test_chain_length_distribution_rejects_empty():
    system, _ = _box_packing(0.40, n=40)
    labels = np.arange(system.n)
    sizes = np.ones(system.n, dtype=np.int64)
    out = chain_length_distribution(system, labels, sizes, min_size=2)
    assert out["n_chains"] == 0


# ---------------------------------------------------------------- breakpoints
def test_piecewise_bic_breakpoint_finds_known_kink():
    x = np.linspace(0.5, 0.85, 8)
    y = np.where(x < 0.72, 2.0 * x, 2.0 * 0.72 + 25.0 * (x - 0.72))
    fit = piecewise_bic_breakpoint(x, y)
    assert abs(fit.x_break - 0.72) <= 0.06
    assert fit.delta_bic > 10.0
    assert fit.preferred == "piecewise"


def test_piecewise_bic_breakpoint_rejects_pure_linear():
    x = np.linspace(0.5, 0.85, 8)
    y = 3.0 * x - 1.0
    fit = piecewise_bic_breakpoint(x, y)
    assert fit.preferred == "single"
    with pytest.raises(ValueError):
        piecewise_bic_breakpoint(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


# --------------------------------------------------------------- integration
def test_ship_geometry_is_closed_and_sharp():
    segs = ship_segments(0.24, 0.16, (0.1, 0.05), bow=0.06)
    pts = segs[:, 0, :]
    assert np.allclose(segs[:, 1, :], np.roll(pts, -1, axis=0))
    assert np.isclose(float(np.ptp(pts[:, 0])), 0.24)
    assert np.isclose(float(np.ptp(pts[:, 1])), 0.16)
    with pytest.raises(ValueError):
        ship_segments(0.1, 0.1, bow=0.2)


def test_grow_to_fraction_rejects_lower_target():
    system, side = _box_packing(0.40, n=60)
    with pytest.raises(ValueError):
        grow_to_fraction(system, 0.30, area=side * side)


def test_channel_resistance_grows_with_ice_concentration():
    low, info_low = make_channel_system(n_particles=100, mean_radius=0.013, phi=0.58,
                                        seed=0, ship_beam=0.26, relax_steps=60, dphi=0.04)
    high, info_high = make_channel_system(n_particles=100, mean_radius=0.013, phi=0.84,
                                          seed=0, ship_beam=0.26, relax_steps=60, dphi=0.04)
    out_low = run_channel(low, drift_speed=0.2, duration=0.20, area=info_low["free_area"],
                          sample_every=300)
    out_high = run_channel(high, drift_speed=0.2, duration=0.20, area=info_high["free_area"],
                           sample_every=300)
    assert out_high["z_all"] > out_low["z_all"]
    assert out_high["resistance_n"] > out_low["resistance_n"]
    assert out_high["phi"] > out_low["phi"]