"""Tests for the multi-disperse floe-field generator (IN22-N01).

Covers the FSD analytics, the coverage / overlap / moment acceptance numbers, the
monodisperse-limit regression lock against the H1 field, the RSA blocking
negative control, determinism, the DEM bridge and the input validation
(negative cases).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)

from tensordem.fsd_packing import (
    BimodalFSD,
    DeltaFSD,
    FloeSizeDistribution,
    LogNormalFSD,
    PowerLawFSD,
    generate_field,
    make_fsd,
    rsa_placement,
)
from tensordem.ice_channel import HullGeometry, IceChannelSimulation

DOMAIN = (90.0, 45.0)
FAMILIES = [
    PowerLawFSD(1.5),
    PowerLawFSD(2.5),
    PowerLawFSD(3.5),
    LogNormalFSD(0.3),
    LogNormalFSD(0.5),
    BimodalFSD(),
]


# --- FSD analytics ---------------------------------------------------------

@pytest.mark.parametrize("fsd", FAMILIES, ids=lambda f: f.kind)
def test_family_median_is_h1_reference(fsd):
    """Every family is anchored on the H1 reference median radius (4.6-5.4 m)."""
    assert 4.6 <= fsd.median <= 5.4
    assert abs(fsd.median - 5.0) < 1e-9


@pytest.mark.parametrize("fsd", [PowerLawFSD(1.5), PowerLawFSD(2.5),
                                 PowerLawFSD(3.5), LogNormalFSD(0.3),
                                 LogNormalFSD(0.5), BimodalFSD()],
                         ids=lambda f: f.kind)
def test_analytic_moments_match_quadrature(fsd):
    """The closed-form moments agree with Gauss-Legendre quadrature."""
    quad = FloeSizeDistribution.moments(fsd, n_quad=512)
    closed = fsd.moments()
    for key in ("m1", "m2"):
        assert closed[key] == pytest.approx(quad[key], rel=2e-4)


@pytest.mark.parametrize("fsd", FAMILIES, ids=lambda f: f.kind)
def test_quantile_and_cdf_are_inverse(fsd):
    u = torch.linspace(0.01, 0.99, 17, dtype=torch.float64)
    a = fsd.quantile(u)
    back = fsd.cdf(a)
    assert torch.allclose(back, u, atol=1e-8)


def test_power_law_support_ratio():
    fsd = PowerLawFSD(kappa=2.5, upper_ratio=3.0)
    lo, hi = fsd.support()
    assert hi / lo == pytest.approx(3.0, rel=1e-12)
    assert fsd.cdf(torch.tensor(lo, dtype=torch.float64)) == pytest.approx(0.0, abs=1e-12)
    assert fsd.cdf(torch.tensor(hi, dtype=torch.float64)) == pytest.approx(1.0, abs=1e-12)


def test_bimodal_solves_its_median():
    fsd = BimodalFSD(coarse_median=7.2, fine_median=2.2, median=5.0)
    assert 0.0 < fsd.fine_fraction < 1.0
    assert fsd.median == pytest.approx(5.0, abs=1e-4)
    # the coarse component dominates the ice area, the fine one the count
    mf = fsd.fine.moments()
    mc = fsd.coarse.moments()
    fine_area_share = fsd.fine_fraction * mf["m2"] / (
        fsd.fine_fraction * mf["m2"] + (1 - fsd.fine_fraction) * mc["m2"])
    assert fine_area_share < 0.35


def test_monodisperse_limit_is_degenerate():
    assert LogNormalFSD(0.0).is_degenerate()
    assert DeltaFSD(5.0).is_degenerate()
    assert not LogNormalFSD(0.01).is_degenerate()
    radii = LogNormalFSD(0.0).sample_radii(8, torch.Generator().manual_seed(1))
    assert torch.all(radii == 5.0)


# --- coverage / overlap / moments -----------------------------------------

@pytest.mark.parametrize("phi", [0.60, 0.75, 0.85])
def test_coverage_control_within_half_a_percent(phi):
    field = generate_field(LogNormalFSD(0.5), phi, domain_length=DOMAIN[0],
                           domain_width=DOMAIN[1], seed=7)
    assert field.coverage_deviation() < 5e-3
    assert field.floe_count >= 5


def test_initial_overlap_at_eight_tenths():
    """8/10 ice is inside the 0.06 m initial-overlap criterion (H1 convention)."""
    field = generate_field(PowerLawFSD(2.5), 0.80, domain_length=160.0,
                           domain_width=80.0, seed=11)
    assert field.max_overlap <= 0.06
    assert field.coordination_number > 0.0


def test_overlap_exceeds_criterion_above_random_close_packing():
    """Honest limit: 8.5/10 is above the 2-D disordered packing limit.

    The field is still generated at the exact coverage, but the disks cannot be
    placed without overlap, so the overlap criterion is *not* met -- the test
    locks the measured behaviour instead of hiding it.
    """
    field = generate_field(PowerLawFSD(2.5), 0.85, domain_length=DOMAIN[0],
                           domain_width=DOMAIN[1], seed=11)
    assert field.coverage_deviation() < 5e-3
    assert field.max_overlap > 0.06
    assert field.extra["converged"] is False


def test_m1_matches_analytic_fsd_moment():
    field = generate_field(PowerLawFSD(2.5), 0.80, domain_length=160.0,
                           domain_width=80.0, seed=3)
    assert field.floe_count > 60
    assert field.m1_deviation() < 0.01


def test_generation_is_deterministic_for_a_seed():
    kwargs = dict(domain_length=DOMAIN[0], domain_width=DOMAIN[1], seed=5)
    a = generate_field(LogNormalFSD(0.3), 0.70, **kwargs)
    b = generate_field(LogNormalFSD(0.3), 0.70, **kwargs)
    assert a.field_digest() == b.field_digest()
    c = generate_field(LogNormalFSD(0.3), 0.70, domain_length=DOMAIN[0],
                       domain_width=DOMAIN[1], seed=6)
    assert a.field_digest() != c.field_digest()


# --- monodisperse-limit regression lock -----------------------------------

def test_monodisperse_limit_reproduces_h1_field_bit_for_bit():
    coverage = 0.80
    generated = generate_field(LogNormalFSD(0.0), coverage, domain_length=DOMAIN[0],
                               domain_width=DOMAIN[1], seed=13)
    reference = IceChannelSimulation.from_field(
        coverage, domain_length=DOMAIN[0], domain_width=DOMAIN[1], seed=13)
    assert generated.method == "h1_lattice"
    assert torch.equal(generated.pos, reference.pos)
    assert torch.equal(generated.radius, reference.radius)
    assert torch.equal(generated.thickness, reference.thickness)
    assert generated.state_digest() == reference.state_digest()


def test_near_monodisperse_limit_stays_close_to_h1():
    field = generate_field(LogNormalFSD(0.01), 0.70, domain_length=DOMAIN[0],
                           domain_width=DOMAIN[1], seed=13)
    assert field.method == "lattice"
    assert field.coverage_deviation() < 5e-3
    assert field.fsd["degenerate"] is False


# --- RSA blocking negative control ----------------------------------------

def test_rsa_blocks_far_below_eight_and_a_half_tenths():
    field = rsa_placement(DeltaFSD(5.0), 0.85, domain_length=DOMAIN[0],
                          domain_width=DOMAIN[1], seed=3,
                          max_consecutive_failures=1500)
    assert field.extra["blocked"] is True
    assert field.realised_coverage < 0.65
    assert field.realised_coverage < 0.85 - 0.005


def test_compaction_generates_what_rsa_cannot():
    coverage = 0.85
    lattice = generate_field(PowerLawFSD(2.5), coverage, domain_length=DOMAIN[0],
                             domain_width=DOMAIN[1], seed=3, method="lattice")
    rsa = rsa_placement(PowerLawFSD(2.5), coverage, domain_length=DOMAIN[0],
                       domain_width=DOMAIN[1], seed=3, max_consecutive_failures=1500)
    assert lattice.coverage_deviation() < 5e-3
    assert rsa.realised_coverage < coverage - 0.05


# --- DEM bridge -----------------------------------------------------------

def test_field_can_be_run_in_the_h1_simulation():
    field = generate_field(LogNormalFSD(0.3), 0.70, domain_length=DOMAIN[0],
                           domain_width=DOMAIN[1], seed=2)
    hull = HullGeometry(length=40.0, beam=12.0, speed=2.0,
                        y_position=0.5 * DOMAIN[1], x_start=-20.0)
    sim = field.to_simulation(hull=hull, dt=1e-2)
    assert sim.n == field.floe_count
    sim.run(120, record_every=40)
    assert math.isfinite(float(sim.pos.abs().max()))
    width = sim.cleared_width()
    assert 0.0 <= width <= DOMAIN[1] + 1e-9


# --- negative cases -------------------------------------------------------

@pytest.mark.parametrize("kappa", [-1.0, 0.0, 1.0])
def test_invalid_power_law_exponent_raises(kappa):
    with pytest.raises(ValueError):
        PowerLawFSD(kappa=kappa)


def test_negative_cv_raises():
    with pytest.raises(ValueError):
        LogNormalFSD(cv=-0.2)


def test_bimodal_requires_coarse_above_fine():
    with pytest.raises(ValueError):
        BimodalFSD(coarse_median=2.0, fine_median=5.0)


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.2, float("nan")])
def test_invalid_coverage_raises(coverage):
    with pytest.raises(ValueError):
        generate_field(LogNormalFSD(0.3), coverage, domain_length=DOMAIN[0],
                       domain_width=DOMAIN[1], seed=1)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        generate_field(LogNormalFSD(0.3), 0.7, domain_length=DOMAIN[0],
                       domain_width=DOMAIN[1], seed=1, method="teleport")


def test_h1_lattice_method_rejects_multidisperse_fsd():
    with pytest.raises(ValueError):
        generate_field(LogNormalFSD(0.3), 0.7, domain_length=DOMAIN[0],
                       domain_width=DOMAIN[1], seed=1, method="h1_lattice")


def test_floe_too_large_for_the_domain_raises():
    with pytest.raises(ValueError):
        generate_field(PowerLawFSD(2.5), 0.5, domain_length=200.0,
                       domain_width=14.0, seed=1)


def test_unknown_fsd_kind_raises():
    with pytest.raises(ValueError):
        make_fsd({"kind": "weibull", "shape": 2.0})


def test_invalid_sample_mode_raises():
    with pytest.raises(ValueError):
        PowerLawFSD(2.5).sample_radii(4, torch.Generator().manual_seed(0),
                                      mode="magic")


def test_empty_floe_set_is_rejected():
    with pytest.raises(ValueError):
        DeltaFSD(5.0).sample_radii(0, torch.Generator().manual_seed(0))


# --- artifact -------------------------------------------------------------

def test_artifact_declares_every_criterion():
    path = Path(__file__).resolve().parent.parent / "artifacts" / "fsd_packing.json"
    if not path.exists():                      # pragma: no cover - artifact optional
        pytest.skip("production artifact not generated in this checkout")
    data = json.loads(path.read_text())
    criteria = data["criteria"]
    for key in ("a_coverage_deviation_lt_0p005", "b_max_overlap_le_0p06",
                "c_monodisperse_limit_bit_identical", "d_m1_deviation_lt_0p01",
                "e_seed_dispersion_lt_0p05", "f_0p85_generatable_vs_rsa",
                "g_generation_time_near_linear", "h_artifact_programmatic"):
        assert key in criteria
        assert "pass" in criteria[key]
    assert data["coverage_sweep"]
    assert data["transit"]
    assert data["monodisperse_lock"]["digest_identical"] is True