"""Verification of the Stefan (phase-change) module against Neumann's solution.

The enthalpy solver and the similarity solution share no code: the solver
inverts ``H(T)`` on a fixed grid, the reference is the transcendental equation
``lambda exp(lambda^2) erf(lambda) = Ste / sqrt(pi)``.  Two negative controls
pin the limits: no superheat means no melting, and a sub-melting surface
temperature must reduce to pure conduction.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from tensordem.interface_thermal.conduction import (
    semi_infinite_temperature_profile,
    thermal_diffusivity,
)
from tensordem.interface_thermal.stefan import (
    EnthalpyStefan1D,
    neumann_interface_position,
    neumann_interface_velocity,
    neumann_lambda,
    neumann_stefan_number,
    neumann_temperature,
)

WATER = dict(k_w_m_k=0.56, rho_kg_m3=1000.0, cp_j_kg_k=4200.0,
             cp_solid_j_kg_k=2100.0, latent_heat_j_kg=3.34e5)
ALPHA = thermal_diffusivity(WATER["k_w_m_k"], WATER["rho_kg_m3"], WATER["cp_j_kg_k"])
SUPERHEAT = 5.0
STE = neumann_stefan_number(SUPERHEAT, WATER["latent_heat_j_kg"], WATER["cp_j_kg_k"])
LAMBDA = neumann_lambda(STE)


def _solver(**overrides):
    settings = dict(length_m=0.01, cells=801, t_melt=0.0, t_initial=0.0,
                    dT_melt=0.01, theta=1.0, **WATER)
    settings.update(overrides)
    return EnthalpyStefan1D(**settings)


def test_neumann_lambda_satisfies_the_transcendental_equation():
    residual = LAMBDA * math.exp(LAMBDA ** 2) * erf(LAMBDA) - STE / math.sqrt(math.pi)
    assert abs(residual) < 1e-12


def test_neumann_lambda_small_stefan_asymptote():
    """``lambda -> sqrt(Ste/2)`` for Ste -> 0 (checked on three decades).

    Small ``lambda``: ``lambda e^{lambda^2} erf(lambda) ~ 2 lambda^2/sqrt(pi)``,
    so the transcendental equation collapses to ``lambda^2 = Ste/2``.
    """
    for ste in (1e-4, 1e-6, 1e-8):
        lam = neumann_lambda(ste)
        assert lam ** 2 == pytest.approx(ste / 2.0, rel=2e-2)


def test_neumann_interface_scaling_laws():
    """``X ~ sqrt(t)`` and ``dX/dt`` consistent with the time derivative."""
    t = np.array([1.0, 4.0, 9.0])
    x = neumann_interface_position(t, LAMBDA, ALPHA)
    assert x[1] / x[0] == pytest.approx(2.0, rel=1e-12)
    assert x[2] / x[0] == pytest.approx(3.0, rel=1e-12)
    dt = 1e-6
    numeric = (float(neumann_interface_position(2.0 + dt, LAMBDA, ALPHA))
               - float(neumann_interface_position(2.0 - dt, LAMBDA, ALPHA))) / (2 * dt)
    assert numeric == pytest.approx(float(neumann_interface_velocity(2.0, LAMBDA, ALPHA)),
                                    rel=1e-6)


def test_enthalpy_melt_front_matches_neumann_solution():
    """Melt thickness against ``X = 2 lambda sqrt(alpha t)`` (<5 %)."""
    solver = _solver()
    history = solver.run(5.0, 0.01, t_surface=SUPERHEAT,
                         record_times=[0.5, 1.0, 2.0, 5.0])
    for time, thickness in zip(history["time"], history["melt_thickness"]):
        exact = float(neumann_interface_position(time, LAMBDA, ALPHA))
        assert abs(thickness - exact) / exact < 0.05, (
            f"t={time}: {thickness*1e3:.4f} mm vs {exact*1e3:.4f} mm")


def test_enthalpy_liquid_profile_matches_neumann_solution():
    solver = _solver()
    solver.run(3.0, 0.01, t_surface=SUPERHEAT)
    exact = neumann_interface(self_time := 3.0)
    # compare inside the liquid layer only (the front smearing is grid limited)
    mask = solver.x < 0.6 * float(neumann_interface_position(self_time, LAMBDA, ALPHA))
    error = np.max(np.abs(solver.t[mask] - exact[mask])) / SUPERHEAT
    assert error < 0.05


def neumann_interface(t: float) -> np.ndarray:
    solver_x = _solver().x
    return neumann_temperature(solver_x, t, LAMBDA, ALPHA, SUPERHEAT, 0.0)


def test_phase_change_energy_ledger_closes():
    """``rho L X + sensible = int q dt`` (the Stefan budget)."""
    solver = _solver()
    initial = solver.capacity_scale * solver.enthalpy(solver.t)
    flux_in = 0.0
    dt = 0.01
    for _ in range(200):                      # 2 s
        solver.step(dt, t_surface=SUPERHEAT)
        # the strong Dirichlet boundary replaces the surface-cell balance, so
        # the conservative budget is the cells j >= 1 fed through the face x=dx/2
        flux_in += solver.conductance[0] * (solver.t[0] - solver.t[1]) * dt
    stored = float(np.sum((solver.capacity_scale * solver.enthalpy(solver.t))[1:]
                          - initial[1:]))
    assert abs(stored - flux_in) / flux_in < 1e-3
    assert solver.latent_energy() > 0.0


def test_no_superheat_means_no_melting():
    """Negative control: a surface at the melting point melts nothing."""
    solver = _solver()
    solver.run(2.0, 0.01, t_surface=0.0)
    assert solver.melt_thickness() == pytest.approx(0.0, abs=1e-15)
    assert np.all(solver.liquid_fraction(solver.t) == 0.0)


def test_sub_melting_surface_reduces_to_pure_conduction():
    """Negative control: below the melting point the solver is a conductor."""
    surface = -1.0
    solver = _solver(cells=801, dT_melt=0.01)
    solver.run(2.0, 0.01, t_surface=surface)
    # the whole body stays in the ice phase, so the diffusivity is the *ice*
    # one (k / rho c_solid) -- not the liquid value used by ALPHA
    alpha_ice = solver.k_w_m_k / (solver.rho_kg_m3 * solver.cp_solid_j_kg_k)
    exact = semi_infinite_temperature_profile(solver.x, 2.0, alpha_ice, 0.0, surface)
    error = np.max(np.abs(solver.t - exact))
    assert error < 0.02 * abs(surface)
    assert solver.melt_thickness() == pytest.approx(0.0, abs=1e-15)


def test_neumann_temperature_is_continuous_at_the_front():
    t = 2.0
    front = float(neumann_interface_position(t, LAMBDA, ALPHA))
    values = neumann_temperature(np.array([0.0, 0.5 * front, front, 1.5 * front]),
                                 t, LAMBDA, ALPHA, SUPERHEAT, 0.0)
    assert values[0] == pytest.approx(SUPERHEAT, rel=1e-12)
    assert values[1] > values[2]
    assert values[2] == pytest.approx(0.0, abs=1e-12)   # front at Tm
    assert values[3] == pytest.approx(0.0, abs=1e-12)   # solid stays at Tm


@pytest.mark.parametrize("kwargs", [
    dict(t_initial=1.0), dict(theta=0.0), dict(cells=3), dict(dT_melt=0.0),
])
def test_solver_rejects_unphysical_inputs(kwargs):
    with pytest.raises((ValueError, TypeError)):
        _solver(**kwargs)