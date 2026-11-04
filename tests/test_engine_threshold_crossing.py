"""Engine-level crossing of the explicit time-step stability boundary.

Companion of ``scripts/verify_engine_threshold_crossing.py`` and of
``docs/dem3d_engine_threshold_crossing.md``.  These are *executable* assertions
driving ``IceDEM3D.step`` at steps that the public constructor refuses
(``dt > DEM3DConfig.recommended_dt``); the step is installed by assigning
``model.dt`` after construction, which is the documented internal bypass used by
the verification harness only.

Reference values come from the assembled step matrix of the probed tangent
stiffness (dense eigen-solve), and from the closed form
``complexity``-free relation ``rho <= 1 <=> 0 <= g <= 2 and d^2 + 2 g <= 4``
derived from ``|tr| <= 1 + det``.  The module under test is used only for the
closed-form growth factor and the step limits.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem import stability
from scripts.verify_engine_threshold_crossing import (
    amplification_matrix,
    build_model,
    reconstruct_stiffness,
    run_case,
)


@pytest.fixture(scope="module")
def linear_lattice():
    model = build_model()
    stiffness = reconstruct_stiffness(model)
    stiffness = 0.5 * (stiffness + stiffness.T)
    eigenvalues = torch.linalg.eigvalsh(stiffness)
    omega = math.sqrt(float(eigenvalues[-1]) / model.config.mass)
    return model, stiffness, omega


def test_probe_recovers_a_symmetric_zero_force_operator(linear_lattice):
    model, stiffness, _ = linear_lattice
    force, _ = model.forces(update_fracture=False, update_history=False)
    assert float(force.abs().max()) == pytest.approx(0.0, abs=1e-10)
    scale = float(stiffness.abs().max())
    assert float((stiffness - stiffness.T).abs().max()) / scale < 1e-6


def test_closed_form_growth_factor_matches_the_assembled_step_matrix(linear_lattice):
    model, stiffness, omega = linear_lattice
    mass = model.config.mass
    for damping_number in (0.0, 0.5, 1.5, 1.95):
        for elastic_number in (0.4, 1.0, 1.5, 2.05):
            dt = elastic_number / omega
            damping_rate = damping_number / dt
            matrix = amplification_matrix(stiffness, mass, dt, damping_rate)
            dense = float(torch.linalg.eigvals(matrix).abs().max())
            closed = stability.symplectic_euler_growth_factor(dt, omega, damping_rate)
            # The closed form is evaluated at the *top* mode; every other mode has
            # d_i <= d_max and rho(d -> 0) = 1 (rigid-body modes), so the many-body
            # spectral radius is max(1, rho_top) up to eigen-solve tolerance.
            if closed > 1.0 + 1e-6:
                assert dense == pytest.approx(closed, rel=1e-6)
            else:
                assert dense <= 1.0 + 1e-6
                assert dense >= closed - 1e-6


def test_engine_growth_matches_the_step_matrix_above_the_threshold(linear_lattice):
    model, stiffness, omega = linear_lattice
    case = run_case(stiffness, model.config.mass, omega, elastic_number=2.1,
                    damping_number=0.0, steps=200, amplitude_cap=2e-5)
    assert case["regime"] == "exponential"
    assert case["rho_dense"] > 1.8
    assert case["rho_measured_engine"] == pytest.approx(case["rho_dense"], rel=1e-3)
    assert case["rho_measured_engine"] == pytest.approx(case["rho_closed_form"], rel=1e-3)


def test_rectangle_criterion_is_not_sufficient_and_the_engine_proves_it(linear_lattice):
    """``d = g = 1.5`` passes both axis tests but has ``rho = 2`` exactly."""
    model, stiffness, omega = linear_lattice
    case = run_case(stiffness, model.config.mass, omega, elastic_number=1.5,
                    damping_number=1.5, steps=60, amplitude_cap=2e-5)
    assert case["axis_rectangle_stable"] is True          # min(2/w, 2/c) would allow it
    assert case["coupled_region_stable"] is False         # 1.5^2 + 2*1.5 = 5.25 > 4
    assert case["rho_dense"] == pytest.approx(2.0, rel=1e-9)
    assert case["rho_measured_engine"] == pytest.approx(2.0, rel=2e-3)
    assert case["dt_s"] < stability.elastic_timestep_limit(omega) / 1.3


def test_coupled_limit_is_the_binding_one_for_an_active_damped_mode():
    coupled = stability.coupled_timestep_limit(4.0, 1.0)
    assert coupled < min(stability.elastic_timestep_limit(4.0),
                         stability.damping_timestep_limit(1.0))
    assert coupled == pytest.approx(4.0 / (math.sqrt(1.0 + 64.0) + 1.0), rel=1e-15)
    # The limit is exact at the fixed rate gamma = 1: just inside rho <= 1, just
    # outside rho > 1 (d^2 + 2 g crosses 4 at exactly this dt).
    inside = coupled * 0.999
    outside = coupled * 1.001
    assert (inside * 4.0) ** 2 + 2.0 * (inside * 1.0) < 4.0
    assert (outside * 4.0) ** 2 + 2.0 * (outside * 1.0) > 4.0
    assert stability.symplectic_euler_growth_factor(inside, 4.0, 1.0) <= 1.0 + 1e-9
    assert stability.symplectic_euler_growth_factor(outside, 4.0, 1.0) > 1.0 + 1e-4
