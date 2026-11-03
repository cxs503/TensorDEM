"""Tests for the finite-contact-arc Brazilian reference and its resolution sweep.

Two independent things are pinned here.

Analytic side (:mod:`tensordem.brazilian_reference`)
    The closed-form centre field for a load spread over a contact arc must
    (a) reduce to the classical Hondros/ASTM point-load field as ``alpha -> 0``,
    (b) agree with a brute-force summation of the same pair-tensor model, and
    (c) be load-independent in the principal ratio.  These are the checks that
    license using the closed form to judge a simulation, so they are asserted
    against the physics constants rather than against stored numbers.

Lattice side (:mod:`brazilian_resolution_sweep`)
    The refinement family must keep the material (Born constants) fixed while
    refining the lattice, and must reproduce the structural facts the sweep doc
    relies on: a particle on the crown for odd ``nx`` (point-like patch) and a
    wide flat crown for even ``nx``.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from brazilian_resolution_sweep import (  # noqa: E402
    BASE_DIAMETER_M,
    BASE_THICKNESS_M,
    _even_case,
    build_case,
    measure,
)
from tensordem.brazilian_reference import (  # noqa: E402
    contact_half_angle,
    finite_arc_centre_field,
    finite_arc_centre_field_discrete,
    point_load_centre_field,
    reference_comparison,
)

LOAD = 3.3723083637766242
DIAMETER = 0.30
THICKNESS = 0.06


# ---------------------------------------------------------------------------
# Analytic reference
# ---------------------------------------------------------------------------

def test_point_load_field_matches_hondros_constants():
    field = point_load_centre_field(LOAD, DIAMETER, THICKNESS)
    split = 2.0 * LOAD / (math.pi * DIAMETER * THICKNESS)
    assert field["sigma_xx_Pa"] == pytest.approx(split, rel=1e-15)
    assert field["sigma_zz_Pa"] == pytest.approx(-3.0 * split, rel=1e-15)
    assert field["principal_ratio"] == pytest.approx(-3.0)


def test_finite_arc_reduces_to_point_load_as_angle_vanishes():
    point = point_load_centre_field(LOAD, DIAMETER, THICKNESS)
    for alpha in (1e-3, 1e-4):
        field = finite_arc_centre_field(LOAD, DIAMETER, 0.0, THICKNESS, half_angle_rad=alpha)
        # Both corrections vanish as alpha^2.
        assert field["tension_correction_factor"] == pytest.approx(1.0, abs=2e-6)
        assert field["compression_correction_factor"] == pytest.approx(1.0, abs=2e-6)
        assert field["principal_ratio"] == pytest.approx(-3.0, abs=1e-5)
        assert abs(field["sigma_xx_Pa"] - point["sigma_xx_Pa"]) < 2e-6 * point["sigma_xx_Pa"]


def test_finite_arc_softens_the_point_load_baseline_monotonically():
    previous = None
    for ratio in (0.01, 0.05, 0.1, 0.2, 0.4):
        field = finite_arc_centre_field(LOAD, DIAMETER, ratio, THICKNESS)
        assert 0.0 < field["tension_correction_factor"] < 1.0
        assert field["principal_ratio"] < -3.0  # ratio magnitude grows with patch width
        if previous is not None:
            assert field["tension_correction_factor"] < previous
        previous = field["tension_correction_factor"]


def test_contact_half_angle_is_the_arcsine_of_the_measured_ratio():
    for ratio in (0.02, 0.0669, 0.25):
        alpha = contact_half_angle(ratio)
        assert math.sin(alpha) == pytest.approx(ratio, rel=1e-12)
        # Consistency: the closed form accepts the ratio and the angle alike.
        from_ratio = finite_arc_centre_field(LOAD, DIAMETER, ratio, THICKNESS)
        from_angle = finite_arc_centre_field(LOAD, DIAMETER, 0.0, THICKNESS, half_angle_rad=alpha)
        assert from_ratio["sigma_xx_Pa"] == pytest.approx(from_angle["sigma_xx_Pa"], rel=1e-12)


def test_discrete_pair_summation_reproduces_the_closed_form():
    ratio = 0.0669
    closed = finite_arc_centre_field(LOAD, DIAMETER, ratio, THICKNESS)
    coarse = finite_arc_centre_field_discrete(
        LOAD, DIAMETER, THICKNESS, closed["contact_half_angle_rad"], segments=9)
    fine = finite_arc_centre_field_discrete(
        LOAD, DIAMETER, THICKNESS, closed["contact_half_angle_rad"], segments=999)
    assert abs(fine["sigma_xx_Pa"] - closed["sigma_xx_Pa"]) < 1e-8 * abs(closed["sigma_xx_Pa"])
    assert abs(fine["sigma_zz_Pa"] - closed["sigma_zz_Pa"]) < 1e-8 * abs(closed["sigma_zz_Pa"])
    # Convergence, not a coincidence at one discretisation.
    assert abs(coarse["sigma_xx_Pa"] - closed["sigma_xx_Pa"]) > 10.0 * abs(
        fine["sigma_xx_Pa"] - closed["sigma_xx_Pa"])


def test_principal_ratio_is_load_independent_but_stresses_are_not():
    low = finite_arc_centre_field(1.0, DIAMETER, 0.0669, THICKNESS)
    high = finite_arc_centre_field(10.0, DIAMETER, 0.0669, THICKNESS)
    assert low["principal_ratio"] == pytest.approx(high["principal_ratio"], rel=1e-14)
    assert high["sigma_xx_Pa"] == pytest.approx(10.0 * low["sigma_xx_Pa"], rel=1e-12)


def test_reference_comparison_attributes_the_contact_width_share():
    measured = {"centre_sigma_xx_Pa": 155.59, "centre_principal_ratio": -2.015}
    comparison = reference_comparison(measured, load_N=LOAD, diameter_m=DIAMETER,
                                      thickness_m=THICKNESS, contact_arc_ratio=0.0669)
    # The measured lattice field misses the point-load baseline by ~30%, while the
    # finite contact patch alone can explain well under 1%.
    assert comparison["point_load"]["sigma_xx_rel_error"] == pytest.approx(0.3045, abs=5e-3)
    assert abs(comparison["attributable_to_contact_width"]["sigma_xx"]) < 0.01
    assert abs(comparison["unexplained_sigma_xx_rel_error"]
               - comparison["point_load"]["sigma_xx_rel_error"]) < 0.01


# ---------------------------------------------------------------------------
# Refinement family
# ---------------------------------------------------------------------------

def test_refinement_family_keeps_the_material_fixed():
    from tensordem.lattice_elasticity3d import periodic_elastic_constants
    constants = []
    for nx in (15, 25, 35):
        case = build_case(nx)
        config = case["config"]
        spacing = 2.0 * float(config["radius_m"])
        born = periodic_elastic_constants(float(config["bond_stiffness_N_m"]), spacing)
        constants.append((born["C11_Pa"], born["C12_Pa"], born["C44_Pa"]))
        # Geometry invariants: fixed D and (near) fixed t/D.
        radius = float(config["radius_m"])
        assert 2.0 * radius * config["nx"] == pytest.approx(BASE_DIAMETER_M, rel=1e-12)
        assert 2.0 * radius * config["ny"] == pytest.approx(BASE_THICKNESS_M, rel=1e-9)
        assert config["nx"] % 2 == 1
    for candidate in constants[1:]:
        for a, b in zip(constants[0], candidate):
            assert a == pytest.approx(b, rel=1e-12)


def test_even_resolution_has_no_particle_on_the_crown():
    """The documented trap: an even ``nx`` gives a flat, wide crown patch."""
    odd = build_case(15)
    even = _even_case(20)
    assert even["config"]["nx"] % 2 == 0
    # The spacing of the even case is comparable to the odd one, so the difference
    # in contact particles is a lattice-offset effect and not a resolution effect.
    odd_measure = measure(odd, 7.0913e-4, increments=1, tolerance_Pa=1e-4)
    even_measure = measure(even, 7.0913e-4, increments=1, tolerance_Pa=1e-4)
    assert odd_measure["contact_particles_top"] <= 2 * odd["config"]["ny"]
    assert even_measure["contact_particles_top"] > 3 * odd_measure["contact_particles_top"]
    assert even_measure["contact_arc_ratio"] > 0.2  # outside the manifest limit


def test_coarse_static_field_has_the_documented_sign_structure():
    """Fast plumbing check: the reference comparison of ``measure`` is wired up."""
    result = measure(build_case(15), 7.0913e-4, increments=2, tolerance_Pa=1e-4)
    assert result["centre_sigma_xx_Pa"] > 0.0 > result["centre_sigma_zz_Pa"]
    assert -3.5 < result["centre_principal_ratio"] < -1.0
    assert result["contact_arc_ratio"] > 0.0
    assert result["equilibrium_converged"] is True
    assert result["reference"]["finite_arc"]["sigma_xx_Pa"] < result["reference"]["point_load"]["sigma_xx_Pa"]
    assert result["reference"]["point_load"]["sigma_xx_rel_error"] > 0.2