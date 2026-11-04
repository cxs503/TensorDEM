"""Finite-amplitude (nonlinear) heave period: verification of the closed debt.

The small-amplitude anchor ``T = 2 pi sqrt(m/k)`` of
``benchmarks/ice_floe_physics.py`` (anchor T) was the only heave-period
reference in the repository, and ``docs/ice_floe_physics_validation.md``
§limitations recorded the finite-amplitude period as *unverified*.  These
tests verify the closure of that debt:

* the energy-integral machinery itself, against the closed-form Duffing period
  (elliptic integral), so the reference is not checked against itself;
* the linear limit, where the machinery must reproduce ``2 pi sqrt(m/K)``;
* the solver, whose measured period must match the energy integral at
  amplitudes where the linearised reference is measurably wrong;
* the negative controls: a pure linear-spring stub must return the linear
  period at *every* amplitude (no manufactured amplitude dependence) and the
  linearised reference must fail beyond 3 % at ``A = 0.3 R``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.special import ellipk

from tensordem.buoyancy_wet import WetCouplingConfig
from tensordem.floe_buoyancy import (
    RigidHeaveFloe,
    discrete_stiffness,
    energy_drop,
    equilibrium_center_height,
    finite_amplitude_heave_period,
    hex_lattice_floe,
    lattice_finite_amplitude_heave_period,
    lattice_heave_acceleration,
    mean_period,
    upward_crossing_periods,
)

RHO_ICE = 917.0
RHO_WATER = 1025.0
G = 9.81
RADIUS = 0.5
THICKNESS = 1.0


def _cfg(drag: float = 0.0) -> WetCouplingConfig:
    return WetCouplingConfig(enabled=True, water_density=RHO_WATER, gravity=G,
                             water_level=0.0, wet_drag_coeff=drag)


def _disk_case():
    cfg = _cfg()
    offsets = torch.zeros((1, 2), dtype=torch.float64)
    y_eq = equilibrium_center_height(RHO_ICE, RHO_WATER, RADIUS)
    stiffness = discrete_stiffness(offsets, RADIUS, THICKNESS, RHO_ICE, cfg, y_eq)
    return offsets, cfg, y_eq, stiffness


# ---------------------------------------------------------------------------
# 1. the machinery, against a closed-form nonlinear period
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amplitude", [0.01, 0.05, 0.1, 0.2])
def test_duffing_period_matches_elliptic_integral(amplitude: float) -> None:
    """``x'' = -(w0^2 x + beta x^3)`` has the exact period
    ``T = 4/sqrt(w0^2 + beta A^2) K(k)``, ``k^2 = beta A^2/(2(w0^2+beta A^2))``.

    Nothing in the implementation knows about elliptic integrals, so this is an
    independent check of the energy integral, the turning-point search and the
    ``u = sqrt(...)`` regularisation of the turning-point singularity.
    """
    w0, beta = 2.0, 50.0

    def accel(ys):
        ys = np.asarray(ys, dtype=np.float64)
        return -(w0 ** 2 * ys + beta * ys ** 3)

    est = finite_amplitude_heave_period(accel, 0.0, amplitude)
    k2 = beta * amplitude ** 2 / (2.0 * (w0 ** 2 + beta * amplitude ** 2))
    exact = 4.0 / math.sqrt(w0 ** 2 + beta * amplitude ** 2) * ellipk(k2)

    assert est.period == pytest.approx(exact, rel=1e-5)
    # the lower turning point of a symmetric potential is -A by symmetry
    assert est.y_min == pytest.approx(-amplitude, rel=1e-6, abs=1e-12)


def test_linear_stub_has_no_amplitude_dependence() -> None:
    """A pure linear spring must give the same period at every amplitude.

    Anti-tautology for the *machinery*: if the integration scheme manufactured
    an amplitude dependence, this test would see it.
    """
    omega = 3.1

    def accel(ys):
        return -omega ** 2 * np.asarray(ys, dtype=np.float64)

    periods = [finite_amplitude_heave_period(accel, 0.0, a).period
               for a in (0.01, 0.1, 0.5)]
    exact = 2.0 * math.pi / omega
    # amplitude independence must be exact to round-off ...
    assert max(periods) - min(periods) < 1e-9 * exact
    # ... and the common value must sit on the analytic one.  rel=1e-5 rather
    # than 1e-12 because the offset (~2.1e-6, constant in A) is the machinery's
    # own grid/interpolation floor, independently measured by the Duffing
    # closed-form check in the benchmark (max 3.3e-6).
    for t in periods:
        assert t == pytest.approx(exact, rel=1e-5)


def test_rejects_non_restoring_release_point() -> None:
    """Releasing below equilibrium (a > 0) must raise, not silently integrate."""
    with pytest.raises(ValueError):
        finite_amplitude_heave_period(lambda ys: np.asarray(ys, float), 0.0, 0.1)


# ---------------------------------------------------------------------------
# 2. linear limit of the shipped force kernel
# ---------------------------------------------------------------------------

def test_small_amplitude_recovers_linear_anchor() -> None:
    """``A = 2e-3 R`` (the amplitude of the existing anchor T) must reproduce
    ``2 pi sqrt(m/K)`` to better than 1e-3 relative."""
    offsets, cfg, y_eq, stiffness = _disk_case()
    est = lattice_finite_amplitude_heave_period(offsets, RADIUS, THICKNESS,
                                                RHO_ICE, cfg, y_eq, 2e-3 * RADIUS)
    assert est.period == pytest.approx(est.period_linear, rel=1e-3)
    assert est.period_linear == pytest.approx(2.0 * math.pi
                                              * math.sqrt(est_period_mass(offsets)
                                                          / stiffness), rel=1e-12)


def est_period_mass(offsets: torch.Tensor) -> float:
    return RHO_ICE * math.pi * RADIUS ** 2 * THICKNESS * int(offsets.shape[0])


# ---------------------------------------------------------------------------
# 3. the amplitude effect is real and the linear anchor fails
# ---------------------------------------------------------------------------

def test_linear_reference_is_wrong_at_finite_amplitude() -> None:
    """At ``A = 0.3 R`` the linearised anchor is off by more than 3 % while the
    energy integral still agrees with the solver -- i.e. the debt mattered."""
    offsets, cfg, y_eq, _k = _disk_case()
    est = lattice_finite_amplitude_heave_period(offsets, RADIUS, THICKNESS,
                                                RHO_ICE, cfg, y_eq, 0.3 * RADIUS)
    linear_error = abs(est.period / est.period_linear - 1.0)
    assert linear_error > 0.03
    assert est.period > est.period_linear  # softening: chord shrinks downward


def test_monotone_softening_for_single_disk() -> None:
    """For one disk the period grows monotonically with amplitude up to ``R/2``."""
    offsets, cfg, y_eq, _k = _disk_case()
    periods = [lattice_finite_amplitude_heave_period(offsets, RADIUS, THICKNESS,
                                                     RHO_ICE, cfg, y_eq, f * RADIUS).period
               for f in (0.05, 0.1, 0.2, 0.3, 0.4)]
    assert all(b > a for a, b in zip(periods, periods[1:]))


def test_energy_drop_is_zero_at_release_and_positive_below() -> None:
    """``ΔU`` bookkeeping: exactly zero at the release height, positive below."""
    offsets, cfg, y_eq, _k = _disk_case()
    accel = lattice_heave_acceleration(offsets, RADIUS, THICKNESS, RHO_ICE, cfg)
    y_end = y_eq + 0.2 * RADIUS
    assert energy_drop(accel, y_end, y_end) == pytest.approx(0.0, abs=1e-14)
    assert energy_drop(accel, y_eq, y_end) > 0.0


# ---------------------------------------------------------------------------
# 4. solver versus the energy-integral reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ratio", [0.1, 0.3])
def test_solver_matches_energy_integral(ratio: float) -> None:
    """``RigidHeaveFloe`` (velocity Verlet, production kernel) versus the
    drag-free energy integral.  Criterion: 0.5 % (the benchmark run at
    1500 steps/period reports <0.02 %; the test uses 800 to stay fast)."""
    offsets, cfg, y_eq, _k = _disk_case()
    amplitude = ratio * RADIUS
    est = lattice_finite_amplitude_heave_period(offsets, RADIUS, THICKNESS,
                                                RHO_ICE, cfg, y_eq, amplitude)
    dt = est.period / 800.0
    floe = RigidHeaveFloe(offsets, RADIUS, THICKNESS, RHO_ICE, cfg, dt=dt,
                          y0=y_eq + amplitude)
    traj = floe.run(int(3.2 * 800))
    measured = mean_period(upward_crossing_periods(traj.t, traj.y, y_eq), skip=1)
    assert measured == pytest.approx(est.period, rel=5e-3)


def test_lattice_floe_force_profile_has_kinks_but_period_holds() -> None:
    """A 19-particle hex lattice (particles enter/leave the water, so the
    aggregate force is only piecewise smooth) must still agree with the
    solver: the reference is not restricted to the smooth single-disk case."""
    offsets = hex_lattice_floe(0.1, 19)
    cfg = _cfg()
    accel = lattice_heave_acceleration(offsets, 0.1, 0.2, RHO_ICE, cfg)
    y_eq = equilibrium_center_height_lattice(accel, offsets, 0.1)
    est = lattice_finite_amplitude_heave_period(offsets, 0.1, 0.2, RHO_ICE, cfg,
                                                y_eq, 0.15 * 0.1)
    dt = est.period / 600.0
    floe = RigidHeaveFloe(offsets, 0.1, 0.2, RHO_ICE, cfg, dt=dt,
                          y0=y_eq + 0.15 * 0.1)
    traj = floe.run(int(3.0 * 600))
    measured = mean_period(upward_crossing_periods(traj.t, traj.y, y_eq), skip=1)
    assert measured == pytest.approx(est.period, rel=5e-3)


def equilibrium_center_height_lattice(accel, offsets: torch.Tensor, radius: float) -> float:
    lo, hi = -float(offsets[:, 1].abs().max()) - radius, float(offsets[:, 1].abs().max()) + radius
    a_lo = float(accel(np.array([lo]))[0])
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        a_mid = float(accel(np.array([mid]))[0])
        if a_lo * a_mid <= 0.0:
            hi = mid
        else:
            lo, a_lo = mid, a_mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)