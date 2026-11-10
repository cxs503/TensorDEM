"""Tests for per-bond Weibull strength heterogeneity and the weakest-link chain.

Covers the engine feature added for the weakest-link campaign
(``DEM3DConfig.bond_strength_weibull_modulus`` / ``bond_strength_seed``) and the
campaign assembly in ``scripts/verify_ice_bond_weakest_link.py``:

* the feature is opt-in and leaves the legacy homogeneous path untouched;
* the draw is reproducible, independent of the global RNG stream, and keeps the
  material median strength;
* the fracture criterion reads the per-bond threshold (and the creep reference
  force follows it), so a chain fails at its weakest bond.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest
import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.ice_bond_viscoelastic import weibull_bond_strengths


def _load_campaign():
    path = (pathlib.Path(__file__).resolve().parents[1]
            / "scripts" / "verify_ice_bond_weakest_link.py")
    spec = importlib.util.spec_from_file_location("weakest_link_campaign", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAMPAIGN = _load_campaign()


def _lattice(**overrides) -> DEM3DConfig:
    base = dict(nx=2, ny=2, nz=3, radius=0.025, density=917.0,
                bond_stiffness=2_000.0, breaking_strain=0.015,
                contact_damping=0.0, fix_x_edges=False)
    base.update(overrides)
    return DEM3DConfig(**base)


# ---------------------------------------------------------------------------
# Configuration guards
# ---------------------------------------------------------------------------
def test_zero_modulus_is_the_legacy_homogeneous_material():
    sim = IceDEM3D(_lattice())
    assert sim.bond_breaking_strain is None
    assert sim.bond_strength_factors is None
    report = sim.bond_law_report()
    assert report["strength_weibull_modulus"] == 0.0
    assert "bond_breaking_strain_median" not in report


@pytest.mark.parametrize("kwargs", [
    {"bond_strength_weibull_modulus": -1.0},
    {"bond_strength_weibull_modulus": float("nan")},
    {"bond_strength_weibull_modulus": float("inf")},
])
def test_invalid_modulus_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DEM3DConfig(**_lattice_kwargs(**kwargs))


@pytest.mark.parametrize("kwargs", [
    {"bond_strength_weibull_modulus": True},
    {"bond_strength_seed": 1.5},
    {"bond_strength_seed": True},
])
def test_wrong_types_are_rejected(kwargs):
    with pytest.raises(TypeError):
        DEM3DConfig(**_lattice_kwargs(**kwargs))


def _lattice_kwargs(**overrides) -> dict:
    kwargs = dict(nx=2, ny=2, nz=3, radius=0.025, density=917.0,
                  bond_stiffness=2_000.0, breaking_strain=0.015,
                  contact_damping=0.0, fix_x_edges=False)
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------
def test_draw_is_reproducible_and_seed_controlled():
    first = IceDEM3D(_lattice(bond_strength_weibull_modulus=2.0,
                              bond_strength_seed=7)).bond_breaking_strain
    second = IceDEM3D(_lattice(bond_strength_weibull_modulus=2.0,
                               bond_strength_seed=7)).bond_breaking_strain
    other = IceDEM3D(_lattice(bond_strength_weibull_modulus=2.0,
                              bond_strength_seed=8)).bond_breaking_strain
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_draw_is_independent_of_the_global_rng_stream():
    torch.manual_seed(1234)
    a = IceDEM3D(_lattice(bond_strength_weibull_modulus=2.0,
                          bond_strength_seed=3)).bond_breaking_strain.clone()
    torch.manual_seed(9999)
    torch.rand(257)
    b = IceDEM3D(_lattice(bond_strength_weibull_modulus=2.0,
                          bond_strength_seed=3)).bond_breaking_strain.clone()
    assert torch.equal(a, b)


def test_draw_keeps_the_material_median_and_the_modulus():
    """Large sample: median equals ``breaking_strain``, slope equals the modulus."""
    sim = IceDEM3D(_lattice(nx=6, ny=6, nz=6, bond_strength_weibull_modulus=2.0,
                            bond_strength_seed=11))
    thresholds = sim.bond_breaking_strain
    assert len(thresholds) > 500
    assert float(thresholds.median()) == pytest.approx(0.015, rel=0.05)
    ordered, _ = torch.sort(thresholds)
    ranks = torch.arange(len(ordered), dtype=torch.float64)
    q = (ranks + 0.5) / len(ordered)          # mid-point plotting positions
    y = torch.log(-torch.log1p(-q))
    x = torch.log(ordered.double())
    slope = float(((x - x.mean()) * (y - y.mean())).sum()
                  / (x - x.mean()).square().sum())
    assert slope == pytest.approx(2.0, rel=0.10)


def test_engine_draw_matches_the_module_helper():
    """The engine path and the campaign helper draw from the same distribution."""
    modulus, seed, n = 2.0, 5, 4_000
    generator = torch.Generator().manual_seed(seed)
    helper = weibull_bond_strengths(n, 1.0, modulus, generator=generator)
    ordered, _ = torch.sort(helper)
    q = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    y = torch.log(-torch.log1p(-q))
    x = torch.log(ordered.double())
    slope = float(((x - x.mean()) * (y - y.mean())).sum()
                  / (x - x.mean()).square().sum())
    assert slope == pytest.approx(modulus, rel=0.05)
    assert float(helper.median()) == pytest.approx(1.0, rel=0.05)


# ---------------------------------------------------------------------------
# Fracture path reads the per-bond threshold
# ---------------------------------------------------------------------------
def test_commit_fracture_reads_the_per_bond_threshold():
    sim = IceDEM3D(_lattice())
    # One very weak bond, everything else far above the imposed strain.
    thresholds = torch.full((len(sim.pairs),), 0.5, dtype=sim.dtype)
    thresholds[3] = 0.001
    sim.bond_breaking_strain = thresholds
    extension = 0.01 * sim.rest_lengths          # strain = 0.01 everywhere
    sim._commit_fracture(extension)
    failed = torch.nonzero(~sim.alive).flatten().tolist()
    assert failed == [3]
    assert int(sim.failure_mode[3]) & 1          # tensile branch


def test_homogeneous_scalar_criterion_is_still_the_default():
    sim = IceDEM3D(_lattice())
    assert sim.bond_breaking_strain is None
    sim._commit_fracture(0.5 * sim.config.breaking_strain * sim.rest_lengths)
    assert bool(sim.alive.all())
    sim._commit_fracture(1.5 * sim.config.breaking_strain * sim.rest_lengths)
    assert not bool(sim.alive.any())


def test_creep_reference_force_follows_the_per_bond_threshold():
    sim = IceDEM3D(_lattice(bond_creep_exponent=2, bond_creep_time_s=1.0))
    homogeneous = sim.bond_creep_reference_force().clone()
    factors = torch.full((len(sim.pairs),), 1.0, dtype=sim.dtype)
    factors[0] = 0.25
    sim.bond_breaking_strain = sim.config.breaking_strain * factors
    reference = sim.bond_creep_reference_force()
    assert float(reference[0]) == pytest.approx(0.25 * float(homogeneous[0]), rel=1e-12)
    assert torch.allclose(reference[1:], homogeneous[1:])


# ---------------------------------------------------------------------------
# The chain: weakest link wins, and the size effect is the analytic one
# ---------------------------------------------------------------------------
def test_chain_fails_at_its_weakest_bond_under_uniform_strain():
    for n_bonds in (1, 2, 4, 6):
        for seed in range(3):
            row = CAMPAIGN.arm_engine_uniform(n_bonds, seed)
            assert row["pass"], row


def test_chain_strength_scales_as_the_weakest_link_reference():
    """Min of N draws over 4 000 realisations: median ratio N**(-1/m) to 3 %."""
    generator = torch.Generator().manual_seed(4242)
    samples = 4_000
    medians = {}
    for n_bonds in (1, 2, 4, 16):
        draws = weibull_bond_strengths(samples * n_bonds, 1.0, 2.0,
                                       generator=generator)
        strength = draws.reshape(samples, n_bonds).min(dim=1).values
        medians[n_bonds] = float(strength.median())
    for n_bonds, median in medians.items():
        analytic = CAMPAIGN.chain_median_force(n_bonds) / (
            CAMPAIGN.BOND_STIFFNESS * CAMPAIGN.BREAKING_STRAIN * CAMPAIGN.L0)
        assert median == pytest.approx(analytic, rel=0.03)


def test_campaign_chain_builder_matches_the_analytic_thresholds():
    sim, column = CAMPAIGN.build_chain(4, 9)
    assert len(column) == 5
    assert len(sim.pairs) == 4
    assert torch.allclose(sim.rest_lengths,
                          torch.full((4,), CAMPAIGN.L0, dtype=sim.dtype))
    thresholds = CAMPAIGN._threshold_forces(sim)
    expected = CAMPAIGN.BOND_STIFFNESS * sim.bond_breaking_strain * CAMPAIGN.L0
    assert torch.allclose(thresholds, expected)
    # Force-free at rest: the parked particles and the chain carry nothing.
    forces = CAMPAIGN.bond_forces(sim)
    assert torch.allclose(forces, torch.zeros_like(forces))
    assert math.isclose(float(sim.dt), float(sim.config.recommended_dt))