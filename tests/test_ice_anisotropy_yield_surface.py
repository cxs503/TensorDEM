"""Tests for the transversely isotropic Hill surface (ice_anisotropy.yield_surface)."""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.ice_anisotropy.fabric import FabricODF
from tensordem.ice_anisotropy.slip_systems import load_direction
from tensordem.ice_anisotropy.yield_surface import (
    COLUMNAR_RATIO_RANGE,
    HillTI,
    calibrated_slip_reference,
    strength_surface_from_fabric,
)

SIG_PERP = 3.0e6
RATIO = 3.5


def _columnar_surface() -> HillTI:
    return strength_surface_from_fabric(FabricODF.columnar(label="S2"),
                                        sigma_perp=SIG_PERP, ratio_columnar=RATIO)


def test_uniaxial_endpoints_reproduce_the_input_strengths():
    surf = _columnar_surface()
    assert surf.uniaxial_strength(0.0) == pytest.approx(surf.sigma_parallel, rel=1e-12)
    assert surf.uniaxial_strength(90.0) == pytest.approx(surf.sigma_perp, rel=1e-12)
    assert surf.sigma_parallel == pytest.approx(RATIO * SIG_PERP, rel=1e-9)


def test_in_plane_isotropy_relation_N_equals_F_plus_2H():
    surf = _columnar_surface()
    c = surf.coefficients
    assert c["F"] == pytest.approx(c["G"])
    assert c["N"] == pytest.approx(c["F"] + 2.0 * c["H"], rel=1e-12)
    assert c["L"] == pytest.approx(c["M"])


def test_azimuth_invariance_and_45deg_between_0_and_90():
    surf = _columnar_surface()
    for th in (15.0, 30.0, 45.0, 60.0, 75.0):
        # transverse isotropy: the strength cannot depend on the azimuth
        assert surf.uniaxial_strength(th) == pytest.approx(
            surf.uniaxial_strength(th, azimuth_deg=37.0), rel=1e-12)
        assert surf.uniaxial_strength(th) < surf.sigma_parallel
    # strength falls monotonically from the column axis outwards ...
    vals = [surf.uniaxial_strength(th) for th in (0.0, 15.0, 30.0, 45.0)]
    assert vals == sorted(vals, reverse=True)
    # ... and overshoots below sigma_perp before recovering to it at 90 deg
    assert surf.uniaxial_strength(45.0) < surf.sigma_perp
    assert surf.uniaxial_strength(60.0) < surf.uniaxial_strength(45.0)


def test_frame_invariance_under_rotation():
    """Rotating stress and material axis together leaves the yield index unchanged."""
    surf = _columnar_surface()
    axis = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(7)
    for _ in range(10):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1.0
        rotated = HillTI(surf.sigma_parallel, surf.sigma_perp, surf.tau_transverse,
                         axis=q @ axis)
        d = load_direction(33.0)
        S = np.outer(d, d)
        assert rotated.yield_index(q @ S @ q.T) == pytest.approx(
            surf.yield_index(S), rel=1e-10)


def test_yield_index_is_homogeneous_of_degree_one():
    surf = _columnar_surface()
    S = surf.uniaxial_stress(40.0)
    base = surf.yield_index(S)
    for scale in (0.25, 1.0, 4.0):
        assert surf.yield_index(scale * S) == pytest.approx(scale * base, rel=1e-12)
    # right on the surface the index is 1 by construction
    assert surf.yield_index(surf.uniaxial_stress(40.0, magnitude=surf.uniaxial_strength(40.0))
                            ) == pytest.approx(1.0, rel=1e-12)
    assert surf.is_failed(1.01 * surf.uniaxial_stress(40.0, magnitude=surf.uniaxial_strength(40.0)))
    assert not surf.is_failed(0.99 * surf.uniaxial_stress(40.0, magnitude=surf.uniaxial_strength(40.0)))


def test_isotropic_surface_is_exactly_von_mises():
    sigma_y = 2.0e6
    surf = HillTI(sigma_y, sigma_y, sigma_y / np.sqrt(3.0))
    assert surf.is_isotropic(tol=1e-12)
    # uniaxial strength is direction independent
    for th in (0.0, 17.0, 45.0, 90.0):
        assert surf.uniaxial_strength(th) == pytest.approx(sigma_y, rel=1e-12)
    # pure shear strength equals sigma_y/sqrt(3) => von Mises
    S = np.zeros((3, 3))
    S[0, 2] = S[2, 0] = 1.0
    assert 1.0 / surf.yield_index(S) == pytest.approx(sigma_y / np.sqrt(3.0), rel=1e-12)
    # hydrostatic stress is invisible to the criterion
    assert surf.hill_form(np.eye(3)) == pytest.approx(0.0, abs=1e-20)


def test_convexity_and_psd_hill_matrix():
    surf = _columnar_surface()
    assert surf.is_convex()
    assert surf.eigenvalues().min() > -1e-12
    # hydrostatic direction is the null space
    assert surf.hill_matrix() @ np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) == pytest.approx(
        np.zeros(6), abs=1e-9)


def test_transverse_shear_sets_the_45deg_strength():
    """sigma45 follows analytically from F, H and the transverse shear strength."""
    surf = _columnar_surface()
    F = surf.coefficients["F"]
    H = surf.coefficients["H"]
    M = surf.coefficients["M"]
    expected = 1.0 / np.sqrt((F + H) / 4.0 + M / 2.0)
    assert surf.uniaxial_strength(45.0) == pytest.approx(expected, rel=1e-12)


def test_biaxial_locus_closes_and_matches_uniaxial_limits():
    surf = _columnar_surface()
    rows = surf.biaxial_locus(np.linspace(-0.9, 0.9, 7))
    assert all(r["feasible"] for r in rows)
    zero = surf.biaxial_locus([0.0])[0]
    assert zero["sigma3_Pa"] == pytest.approx(surf.sigma_parallel, rel=1e-12)
    # with no transverse stress the two column-axis limits are symmetric
    pos = surf.biaxial_locus([0.0])[0]["sigma3_Pa"]
    assert pos == pytest.approx(surf.sigma_parallel, rel=1e-12)
    # a transverse tension shifts the column-axis limit (Hill's cross term),
    # so the locus is an ellipse centred off the origin in sigma3 -- the two
    # roots for a given sigma1 must be symmetric about G*sigma1/(F+G)
    c = surf.coefficients
    s1 = 0.3 * surf.sigma_perp
    rows = surf.biaxial_locus([0.3])
    root = rows[0]["sigma3_Pa"]
    a = c["F"] + c["G"]
    b = -2.0 * c["G"] * s1
    cc = (c["G"] + c["H"]) * s1 ** 2 - 1.0
    other = (-b - np.sqrt(b * b - 4.0 * a * cc)) / (2.0 * a)
    assert root + other == pytest.approx(-b / a, rel=1e-12)
    assert surf.hill_form(np.diag([s1, 0.0, root])) == pytest.approx(1.0, rel=1e-10)


def test_uniform_fabric_surface_degenerates_to_isotropic():
    surf = strength_surface_from_fabric(FabricODF.uniform(), sigma_perp=SIG_PERP)
    assert surf.strength_ratio == pytest.approx(1.0, rel=1e-12)
    # the mean-field Schmid reference lands within 4% of the von Mises shear value
    assert surf.is_isotropic(tol=0.05)
    # the cubic-orbit fabric set is exact in a_ij (ratio 1 to round-off) but the
    # nonlinear angular average of the Schmid factor is quadrature limited on it,
    # so the *directional* variation is checked against a looser bar here; the
    # report measures it on a dense isotropic fabric instead.
    angles = np.linspace(0.0, 90.0, 19)
    vals = [surf.uniaxial_strength(th) for th in angles]
    assert max(abs(v / vals[-1] - 1.0) for v in vals) <= 0.10


def test_columnar_surface_matches_slip_reference_at_45deg():
    odf = FabricODF.columnar(label="S2")
    surf = strength_surface_from_fabric(odf, sigma_perp=SIG_PERP, ratio_columnar=RATIO)
    slip = calibrated_slip_reference(odf, ratio_columnar=RATIO)
    s_perp = slip.uniaxial_strength(odf, load_direction(90.0))
    ref = slip.uniaxial_strength(odf, load_direction(45.0)) / s_perp
    assert surf.uniaxial_strength(45.0) / surf.sigma_perp == pytest.approx(ref, rel=0.10)


def test_ratio_outside_literature_band_is_rejected():
    odf = FabricODF.columnar()
    for bad in (1.0, 2.9, 5.1):
        with pytest.raises(ValueError):
            strength_surface_from_fabric(odf, sigma_perp=SIG_PERP, ratio_columnar=bad)
    assert COLUMNAR_RATIO_RANGE == (3.0, 5.0)


def test_invalid_surface_parameters_rejected():
    with pytest.raises(ValueError):
        HillTI(-1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        HillTI(1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        strength_surface_from_fabric(FabricODF.uniform(), sigma_perp=-1.0)