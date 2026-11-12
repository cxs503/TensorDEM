"""Invariants of the 3-D zero-pressure jamming measurement.

The point of these tests is *not* to re-measure the RCP density (that is the
study in ``benchmarks/rcp_3d_jamming``), but to pin the properties that make the
measurement admissible:

* the reported density is **not** an input of the protocol (it must not equal
  any imposed level, and the deposition/grid schedule must not set it),
* the forces are the exact gradient of the potential that is minimised (a sign
  error here silently produces a "jammed" state that is an energy maximum),
* the virial pressure equals ``-dU/dV`` on the same potential,
* both minimisers reach the same energy minimum,
* an equation of state that does not rise in ``phi`` is rejected instead of
  being extrapolated, and
* a ``(n, seed)`` pair is bit-reproducible.

The protocol runs use small ``N`` and a coarse ``dphi`` so the file stays fast;
the numbers they produce are *not* quoted as physics results.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular3d import (
    PLACEMENT_PHI,
    ContactModel,
    SpherePacking,
    finite_size_scale,
    fire_minimize,
    fit_jamming_density,
    growth_protocol,
    make_packing,
    periodic_rsa,
    pg_minimize,
    pressure_resolution,
)


def _overlapping_packing(n=40, phi=0.55, seed=3):
    """A deliberately out-of-equilibrium packing used for gradient checks."""
    return make_packing(n, phi, seed=seed)


# ----------------------------------------------------------------- deposition
def test_rsa_deposition_has_no_overlap():
    box = 6.0
    radii = np.full(60, 0.5)
    pos = periodic_rsa(60, radii, box, seed=11)
    tree_pairs = []
    for i in range(59):
        d = pos[i + 1:] - pos[i]
        d -= box * np.round(d / box)
        tree_pairs.append(np.linalg.norm(d, axis=1) - (radii[i + 1:] + radii[i]))
    assert np.min(np.concatenate(tree_pairs)) >= -1e-12


def test_deposition_density_is_imposed_and_reported_as_such():
    # the *protocol* imposes PLACEMENT_PHI; this is not a measurement
    packing = make_packing(120, 0.40, seed=5)
    assert packing.volume_fraction() == pytest.approx(0.40, abs=1e-12)


# ------------------------------------------------------------- force/pressure
def test_forces_are_the_energy_gradient():
    packing = _overlapping_packing()
    forces = packing.forces()
    ev = packing.evaluate()
    assert ev["max_overlap_over_diameter"] > 1e-3  # genuinely out of equilibrium
    h = 1e-7
    for idx in (0, 7, 23):
        for axis in (0, 1, 2):
            saved = packing.pos[idx, axis]
            packing.pos[idx, axis] = saved + h
            u_plus = packing.energy()
            packing.pos[idx, axis] = saved - h
            u_minus = packing.energy()
            packing.pos[idx, axis] = saved
            grad = (u_plus - u_minus) / (2.0 * h)
            assert grad == pytest.approx(-forces[idx, axis], rel=2e-4, abs=2e-6)


def test_virial_pressure_equals_minus_dUdV():
    """``P = -dU/dV`` at fixed radii: the virial formula must match a finite
    difference of the same potential under an affine scaling of the cell."""
    packing = _overlapping_packing(n=40, phi=0.58, seed=4)
    pressure = packing.pressure()
    assert pressure > 0.0
    h = 1e-6

    def energy_at_scale(scale: float) -> float:
        shifted = SpherePacking(packing.pos * scale, packing.radii,
                                box=packing.box * scale, model=packing.model)
        return shifted.energy()

    u_plus = energy_at_scale((1.0 + h) ** (1.0 / 3.0))
    u_minus = energy_at_scale((1.0 - h) ** (1.0 / 3.0))
    # dV/dh = V for a cubic cell scaled by (1+h)^{1/3}
    p_fd = -(u_plus - u_minus) / (2.0 * h * packing.volume)
    assert p_fd == pytest.approx(pressure, rel=5e-3)


# ----------------------------------------------------------------- minimisers
def test_fire_is_monotone_and_converges():
    packing = _overlapping_packing()
    u0 = packing.energy()
    res = fire_minimize(packing, force_tol=1e-7, max_iter=60000)
    assert res.converged
    assert res.energy <= u0
    assert res.max_force <= 1e-7
    assert res.rejected_steps >= 0


def test_minimisers_reach_the_same_minimum():
    a = _overlapping_packing(seed=8)
    b = _overlapping_packing(seed=8)
    ra = fire_minimize(a, force_tol=1e-8, max_iter=80000)
    rb = pg_minimize(b, force_tol=1e-8, max_iter=80000)
    assert ra.converged and rb.converged
    assert ra.energy == pytest.approx(rb.energy, rel=1e-5, abs=1e-12)


# ------------------------------------------------------------ jamming estimate
@pytest.fixture(scope="module")
def protocol_runs():
    grid = np.arange(0.36, 0.741, 0.03)
    a = growth_protocol(n=120, seed=1, phi_grid=grid, force_tol=3e-6, max_iter=40000)
    b = growth_protocol(n=120, seed=1, phi_grid=np.arange(0.48, 0.741, 0.03),
                        force_tol=3e-6, max_iter=40000)
    return a, b


def _fit(run):
    phi = np.array([r["volume_fraction"] for r in run["rows"]])
    pressure = np.array([r["pressure"] for r in run["rows"]])
    floor = max(
        pressure_resolution(n=run["n"], volume=run["box"] ** 3,
                            coordination=r["coordination"], max_force=r["max_force"],
                            mean_radius=run["mean_radius"])
        for r in run["rows"]
    )
    return fit_jamming_density(phi, pressure, floor=floor)


def test_jamming_density_is_not_an_input(protocol_runs):
    run, _ = protocol_runs
    fit = _fit(run)
    phi = np.array([r["volume_fraction"] for r in run["rows"]])
    phi_j = fit["phi_j"]
    # a measurement, not a restatement of a grid level
    assert np.min(np.abs(phi - phi_j)) > 1e-6
    # and not the deposition density either
    assert abs(phi_j - PLACEMENT_PHI) > 0.05
    # physically plausible for a 3-D monodisperse packing
    assert 0.55 < phi_j < 0.70
    assert fit["r2_power_law"] > 0.9


def test_jamming_density_is_insensitive_to_the_grid_start(protocol_runs):
    a, b = protocol_runs
    fa, fb = _fit(a), _fit(b)
    assert fa["phi_j"] == pytest.approx(fb["phi_j"], abs=0.02)


def test_reproducible_for_a_fixed_seed():
    grid = np.arange(0.40, 0.62, 0.10)
    a = growth_protocol(n=80, seed=7, phi_grid=grid, force_tol=1e-4, max_iter=20000)
    b = growth_protocol(n=80, seed=7, phi_grid=grid, force_tol=1e-4, max_iter=20000)
    assert [r["pressure"] for r in a["rows"]] == [r["pressure"] for r in b["rows"]]


# ------------------------------------------------------------------- guardrails
def test_fit_rejects_non_increasing_eos():
    phi = np.array([0.62, 0.64, 0.66, 0.68])
    pressure = np.array([2e-3, 1e-3, 5e-4, 1e-4])
    with pytest.raises(ValueError):
        fit_jamming_density(phi, pressure, model="power_law")
    with pytest.raises(ValueError):
        fit_jamming_density(phi[:2], pressure[:2], floor=1e-6)


def test_pressure_resolution_documents_the_floor():
    kwargs = dict(n=500, volume=800.0, coordination=6.0, mean_radius=1.0)
    p1 = pressure_resolution(max_force=1e-6, **kwargs)
    p2 = pressure_resolution(max_force=1e-4, **kwargs)
    assert p2 == pytest.approx(100.0 * p1)
    assert p1 > 0.0


def test_finite_size_scale_recovers_a_known_law():
    sizes = np.array([200.0, 500.0, 1000.0, 2500.0])
    nu = 0.6
    phi = 0.6380 - 0.05 * sizes ** (-1.0 / (3.0 * nu))
    fit = finite_size_scale(sizes, phi, n_boot=20)
    assert fit["phi_j_inf"] == pytest.approx(0.6380, abs=2e-3)
    assert fit["nu"] == pytest.approx(nu, rel=0.1)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        SpherePacking(np.zeros((3, 2)), np.ones(3), box=1.0)
    with pytest.raises(ValueError):
        SpherePacking(np.zeros((3, 3)), -np.ones(3), box=1.0)
    with pytest.raises(ValueError):
        ContactModel(kn=-1.0)
    with pytest.raises(ValueError):
        growth_protocol(n=50, seed=1, phi_grid=np.array([0.5, 0.4]),
                        force_tol=1e-4, max_iter=100)
    with pytest.raises(ValueError):
        growth_protocol(n=50, seed=1, phi_grid=np.array([0.2]),
                        force_tol=1e-4, max_iter=100)