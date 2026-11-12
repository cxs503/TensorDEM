"""H1 heave transient against its closed form (verifier seat, m16270).

The merged H1 suite asserts only the *endpoint* of the buoyancy case
(``rho_ice/rho_water``).  These tests assert the *transient*: for a prismatic
floe the heave satisfies ``m z'' = -rho_w g A (z - z_eq) - c z'`` with every
coefficient fixed before the solver runs, so the fitted damped frequency,
decay rate, restoring stiffness, ledger and the time-step convergence order are
all checkable against closed form rather than against a previous output.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from h1_heave_analytic_verification import (  # noqa: E402
    analyse,
    convergence_table,
    fit_oscillator,
    run_transient,
)

BASE = dict(footprint_area=0.36, thickness=0.12, n_sides=16,
            drag_fraction=0.7, dt=5.0e-4, n_steps=4000)


@pytest.fixture(scope="module")
def analysis():
    run = run_transient(**BASE)
    return run, analyse(run)


def test_restoring_stiffness_from_transient_matches_waterplane():
    """omega_0 recovered from the ring-down must equal sqrt(rho_w g A / m)."""
    run = run_transient(**BASE)
    result = analyse(run)
    assert result["errors_pct"]["stiffness"] < 3.0, result["errors_pct"]
    # the reference itself is the waterplane force, not a fitted constant
    assert run["k_analytic"] == pytest.approx(
        run["rho_water"] * 9.81 * run["waterplane_area"], rel=1e-12)


def test_decay_rate_matches_imposed_damping(analysis):
    """gamma_fit must equal c/(2m) with c = drag_fraction * 2 m omega_0."""
    run, result = analysis
    assert result["errors_pct"]["gamma"] < 3.0, result["errors_pct"]
    assert run["zeta"] == pytest.approx(0.7, rel=1e-12)


def test_equilibrium_matches_archimedes_draft(analysis):
    """Settled centroid equals h/2 - (rho_i/rho_w) h to <1 %."""
    _run, result = analysis
    assert result["errors_pct"]["equilibrium_z"] < 1.0, result["errors_pct"]


def test_drag_ledger_closes_against_integrated_drag_work(analysis):
    """sum(c v^2 dt) from the recorded velocities must equal the ledger."""
    _run, result = analysis
    assert result["ledger"]["drag_work_integrated_J"] > 0.0
    assert result["ledger"]["ledger_vs_integral_pct"] < 2.0, result["ledger"]
    assert result["ledger"]["energy_residual_dimensionless"] < 1.0e-2


def test_frequency_error_halves_with_time_step():
    """Leapfrog with a velocity-dependent force: observed order ~1, not 0."""
    rows = convergence_table(dict(BASE, n_steps=2000))
    errs = [r["omega0_error_pct"] for r in rows]
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(1.6 < r < 2.6 for r in ratios), (errs, ratios)
    order = math.log2(ratios[0])
    assert 0.6 < order < 1.5, order
    # equilibrium is exact to discretisation round-off at every dt (linear profile)
    assert max(r["equilibrium_error_pct"] for r in rows) < 1.0


def test_fit_recovers_a_synthetic_oscillator():
    """Negative control: the fitter must invert a known signal to round-off."""
    t = np.linspace(0.0, 2.0, 2001)
    true = dict(a=0.04, gam=3.1, om=7.3, phi=0.2, off=0.0)
    z = true["a"] * np.exp(-true["gam"] * t) * np.cos(true["om"] * t + true["phi"])
    fit = fit_oscillator(t, z, 0.0)
    assert fit["gamma_per_s"] == pytest.approx(true["gam"], rel=1e-5)
    assert fit["omega_d_rad_per_s"] == pytest.approx(true["om"], rel=1e-5)
    assert fit["amplitude_m"] == pytest.approx(true["a"], rel=1e-5)