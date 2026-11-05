"""Verification of the ridge-grounding Winkler subgrade modulus.

References: Vesic (1961) beam-on-elastic-half-space modulus (closed form) and a
self-contained 3-D Boussinesq half-space beam solver, whose own pieces are checked
against the Hetenyi (1946) closed form, brute-force quadrature and global
equilibrium.  This module verifies the *code* against those references; it is not a
physical validation.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.ridge_geotech import MohrCoulombSoil, vesic_subgrade_modulus

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts"
          / "verify_ridge_winkler_subgrade_modulus.py")
_spec = importlib.util.spec_from_file_location("ridge_winkler_verify", SCRIPT)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

SOIL = MohrCoulombSoil(young_modulus=10.0e6, poisson_ratio=0.30)
EI = 2.1e11 * ref.pipe_inertia(0.914, 0.0159)


# ------------------------------------------------------------------ kernel
def test_strip_kernel_matches_brute_force_quadrature():
    """The closed-form 1/r rectangle integral reproduces 2-D Gauss quadrature."""
    for u0, u1 in ((3.0, 4.0), (0.5, 2.5), (-4.0, -3.0)):
        exact = ref.strip_kernel(u0, u1, 0.457)
        brute = ref._brute_quadrature(u0, u1, 0.457)
        assert abs(exact - brute) / brute < 1e-9


def test_strip_kernel_folds_signed_interval():
    """Crossing the singularity must add both halves, not subtract them."""
    half_width = 0.457
    left = ref.strip_kernel(-1.0, 0.0, half_width)
    right = ref.strip_kernel(0.0, 1.0, half_width)
    assert ref.strip_kernel(-1.0, 1.0, half_width) == pytest.approx(left + right)
    assert left == pytest.approx(right)
    assert math.isfinite(ref.strip_kernel(-0.1, 0.1, half_width))


def test_compliance_matrix_is_symmetric_and_positive():
    C = ref.compliance_matrix(40, 0.4, 0.914, 10.0e6, 0.3)
    assert np.allclose(C, C.T, rtol=0.0, atol=0.0)
    assert np.all(np.linalg.eigvalsh(C) > 0.0)


def test_compliance_self_term_scales_with_softness():
    soft = ref.compliance_matrix(20, 0.4, 0.914, 5.0e6, 0.3)
    stiff = ref.compliance_matrix(20, 0.4, 0.914, 10.0e6, 0.3)
    assert soft[10, 10] == pytest.approx(2.0 * stiff[10, 10], rel=1e-12)


# ------------------------------------------------------------- beam solver
def test_beam_fe_reproduces_hetenyi_patch_closed_form():
    """Finite-element beam on a spring bed vs the exact patch-load solution."""
    k_eff = SOIL.subgrade_modulus(0.914) * 0.914
    q = 8.67e5 / 2.0
    solution = ref.solve_winkler(80.0, 400, EI, k_eff, q, 2.0)
    w0, m0 = ref.closed_form_patch(80.0 / 400, k_eff, EI, q, 2.0)
    assert solution["deflection_max_m"] == pytest.approx(w0, rel=0.005)
    assert solution["moment_max_Nm_per_m"] == pytest.approx(m0, rel=0.005)


# --------------------------------------------------- half-space reference
def test_halfspace_reference_is_in_equilibrium():
    solution = ref.solve_halfspace(80.0, 400, EI, 0.914, 10.0e6, 0.3, 8.67e5 / 2.0, 2.0)
    assert solution["equilibrium_rel_error"] < 1e-6
    assert solution["compatibility_residual_m"] < 1e-12
    assert solution["pressure_max_Pa"] > 0.0


def test_halfspace_moment_converges_with_mesh_and_length():
    q = 8.67e5 / 2.0
    coarse = ref.solve_halfspace(80.0, 400, EI, 0.914, 10.0e6, 0.3, q, 2.0)
    longer = ref.solve_halfspace(160.0, 400, EI, 0.914, 10.0e6, 0.3, q, 2.0)
    finer = ref.solve_halfspace(80.0, 800, EI, 0.914, 10.0e6, 0.3, q, 2.0)
    for other in (longer, finer):
        assert abs(other["moment_max_Nm_per_m"]
                   - coarse["moment_max_Nm_per_m"]) / coarse["moment_max_Nm_per_m"] < 0.02


# --------------------------------------------------------- Vesic modulus
def test_vesic_scaling_exponents():
    """k ~ E_s^(13/12), EI^(-1/12), B^(-2/3), 1/(1-nu^2)."""
    base = vesic_subgrade_modulus(10.0e6, 0.3, 0.914, EI)
    assert vesic_subgrade_modulus(20.0e6, 0.3, 0.914, EI) / base == pytest.approx(
        2.0 ** (13.0 / 12.0), rel=1e-12)
    assert vesic_subgrade_modulus(10.0e6, 0.3, 0.914, 2.0 * EI) / base == pytest.approx(
        2.0 ** (-1.0 / 12.0), rel=1e-12)
    assert vesic_subgrade_modulus(10.0e6, 0.3, 2.0 * 0.914, EI) / base == pytest.approx(
        2.0 ** (-2.0 / 3.0), rel=1e-12)


def test_vesic_reduces_to_merged_modulus_when_rigidity_is_unity():
    """With E_s B^4 / (E I) = 1 the rigidity factor is 1 and both formulae agree."""
    young = 12.0e6
    width = 0.75
    beam_ei = young * width ** 4
    assert vesic_subgrade_modulus(young, 0.3, width, beam_ei) == pytest.approx(
        MohrCoulombSoil(young_modulus=young, poisson_ratio=0.3).subgrade_modulus(width),
        rel=1e-12)


def test_merged_modulus_regression_default_path_unchanged():
    assert SOIL.subgrade_modulus(0.914) == pytest.approx(
        0.65 * 10.0e6 / (0.91 * 0.914), rel=1e-12)
    assert SOIL.initial_vertical_stiffness(0.914) == pytest.approx(
        SOIL.subgrade_modulus(0.914) * 0.914, rel=1e-12)
    assert SOIL.subgrade_modulus_model == "width_only"


def test_vesic_model_opt_in_path():
    soil = MohrCoulombSoil(young_modulus=10.0e6, poisson_ratio=0.30,
                           subgrade_modulus_model="vesic")
    with pytest.raises(ValueError):
        soil.subgrade_modulus(0.914)
    k = soil.subgrade_modulus(0.914, beam_ei=EI)
    assert k == pytest.approx(vesic_subgrade_modulus(10.0e6, 0.30, 0.914, EI), rel=1e-12)
    assert k == pytest.approx(SOIL.subgrade_modulus(0.914) * (10.0e6 * 0.914 ** 4 / EI)
                              ** (1.0 / 12.0), rel=1e-12)
    assert soil.initial_vertical_stiffness(0.914, beam_ei=EI) == pytest.approx(k * 0.914)


def test_invalid_subgrade_modulus_model_rejected():
    with pytest.raises(ValueError):
        MohrCoulombSoil(subgrade_modulus_model="biot")


# --------------------------------------------- the debt, quantified
def test_independent_reference_recovers_vesic_modulus():
    """Inverting the half-space reference for k reproduces Vesic (1961)."""
    q = 8.67e5 / 2.0
    half = ref.solve_halfspace(80.0, 400, EI, 0.914, 10.0e6, 0.3, q, 2.0)
    k_equiv = ref.equivalent_winkler_modulus(EI, half["moment_max_Nm_per_m"], q, 2.0,
                                             80.0, 400, 7.0e6)
    k_vesic = vesic_subgrade_modulus(10.0e6, 0.3, 0.914, EI) * 0.914
    assert abs(k_equiv / k_vesic - 1.0) < 0.05


def test_merged_modulus_moment_error_exceeds_three_percent():
    """Debt assertion: the merged modulus bends the design moment by >3 %."""
    q = 8.67e5 / 2.0
    half = ref.solve_halfspace(80.0, 400, EI, 0.914, 10.0e6, 0.3, q, 2.0)
    merged = ref.solve_winkler(80.0, 400, EI, SOIL.subgrade_modulus(0.914) * 0.914, q, 2.0)
    vesic = ref.solve_winkler(80.0, 400, EI,
                              vesic_subgrade_modulus(10.0e6, 0.3, 0.914, EI) * 0.914, q, 2.0)
    merged_error = 100.0 * (merged["moment_max_Nm_per_m"]
                            - half["moment_max_Nm_per_m"]) / half["moment_max_Nm_per_m"]
    vesic_error = 100.0 * (vesic["moment_max_Nm_per_m"]
                           - half["moment_max_Nm_per_m"]) / half["moment_max_Nm_per_m"]
    assert merged_error < -3.0          # merged modulus is too stiff -> moment too low
    assert abs(vesic_error) < 3.0       # Vesic reproduces the reference moment