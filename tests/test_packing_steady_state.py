"""Analytic tests for the packing steady-state / isostaticity witnesses.

Every case here has a closed-form answer, so the tests assert *invariants*
(contact counts of a lattice, force balance of a static stack) rather than
merely executing the code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.packing_steady_state import (
    contact_network,
    coordination_stats,
    coordination_vs_tolerance,
    isostatic_coordination,
    mobility_stats,
    plateau_drift,
    plateau_onset,
    static_equilibrium,
)


# ---------------------------------------------------------------------------
# isostatic counting
# ---------------------------------------------------------------------------
def test_isostatic_coordination_counting_values():
    # frictionless spheres: N_c = d N  ->  z = 2d
    assert isostatic_coordination(3, frictionless=True) == pytest.approx(6.0)
    # frictional + rolling resistance: 3 constraints/contact, 2d N dof -> 4d/3
    assert isostatic_coordination(3, frictionless=False) == pytest.approx(4.0)
    assert isostatic_coordination(2, frictionless=True) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# contact network against lattices with known coordination
# ---------------------------------------------------------------------------
def _lattice(n_side: int, spacing: float, width: float) -> np.ndarray:
    """``n_side^3`` particles on a cubic lattice, laterally periodic cell."""
    g = np.arange(n_side) * spacing + 0.5 * spacing
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    pos = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    assert width >= n_side * spacing
    return pos


def test_simple_cubic_lattice_has_coordination_six():
    spacing = 0.999                     # slight overlap, so contacts are counted
    pos = _lattice(5, spacing, width=5.0 * spacing)
    net = contact_network(pos, diameter=1.0, width=5.0 * spacing)
    stats = coordination_stats(pos, 1.0, 5.0 * spacing,
                               band=(0.2, float(pos[:, 2].max()) - 0.2),
                               frictionless=True, net=net)
    assert stats["z_mean"] == pytest.approx(6.0)
    assert stats["rattler_fraction"] == 0.0


def test_face_centred_cubic_lattice_has_coordination_twelve():
    a = math.sqrt(2.0) * 0.999          # fcc nearest-neighbour distance = a/sqrt2
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]) * a
    n = 4
    pts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pts.append(basis + np.array([i, j, k]) * a)
    pos = np.vstack(pts) + 0.25 * a
    width = n * a                        # lateral periodicity: full 12 neighbours
    pos[:, 0] %= width
    pos[:, 1] %= width
    net = contact_network(pos, diameter=1.0, width=width)
    # interior particles of the fcc lattice have 12 equidistant neighbours; the
    # two z-boundary layers lose the ones outside the (open) vertical cell.
    assert int(np.max(net.n_contacts)) == 12
    assert np.median(net.n_contacts) == pytest.approx(12.0)


def test_periodic_boundary_pair_is_counted_once():
    """Two spheres straddling the periodic boundary are one contact, not zero."""
    width = 1.0
    d = 0.6
    pos = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]])   # min-image gap = 0.2 < d
    net = contact_network(pos, diameter=d, width=width)
    assert net.n_pairs == 1                              # deduplicated, not 2
    assert net.mean_coordination == pytest.approx(1.0)
    naive = net.pair_overlap[0]
    assert naive == pytest.approx(d - 0.2)

    # the same two spheres without periodicity are NOT neighbours
    far = np.array([[0.1, 0.5, 0.5], [0.9 + 2 * width, 0.5, 0.5]])
    assert contact_network(far, diameter=d, width=width).n_pairs == 0


def test_gap_tolerance_is_monotone():
    pos = _lattice(4, 1.02, width=4.0 * 1.02)            # gaps of 0.02 d, no overlap
    zs = [s["z_mean"] for s in coordination_vs_tolerance(
        pos, 1.0, 4.0 * 1.02, tolerances=(0.0, 0.03, 0.05))]
    assert zs[0] == pytest.approx(0.0)                   # hard threshold: no contacts
    assert zs[1] > zs[0]                                 # opening the tolerance adds them
    assert zs[2] >= zs[1]                                # and the count never drops


# ---------------------------------------------------------------------------
# static equilibrium against hand-computable stacks
# ---------------------------------------------------------------------------
def test_single_sphere_on_floor_is_in_equilibrium():
    diameter, stiffness, mass, gravity = 1.0, 4000.0, 0.5236, 1.0
    overlap = mass * gravity / stiffness                 # floor carries the weight
    pos = np.array([[0.5, 0.5, 0.5 * diameter - overlap]])
    eq = static_equilibrium(pos, diameter, 5.0, stiffness, mass, gravity,
                           band=(0.0, diameter))
    assert eq["ratio_median"] == pytest.approx(0.0, abs=1e-9)
    assert eq["n_band"] == 1


def test_two_sphere_stack_is_in_equilibrium():
    """Bottom sphere carries the floor reaction of two weights, top one of one."""
    diameter, stiffness, mass, gravity = 1.0, 4000.0, 0.5236, 1.0
    delta = mass * gravity / stiffness
    z_bottom = 0.5 * diameter - 2.0 * delta
    z_top = z_bottom + diameter - delta
    pos = np.array([[0.5, 0.5, z_bottom], [0.5, 0.5, z_top]])
    eq = static_equilibrium(pos, diameter, 5.0, stiffness, mass, gravity,
                            band=(0.0, diameter))
    assert eq["ratio_median"] == pytest.approx(0.0, abs=1e-9)
    assert eq["ratio_max"] == pytest.approx(0.0, abs=1e-9)


def test_unsupported_sphere_has_unbalanced_force_equal_to_weight():
    diameter, stiffness, mass, gravity = 1.0, 4000.0, 0.5236, 1.0
    pos = np.array([[0.5, 0.5, 5.0], [0.5, 0.5, 5.0 + 3.0]])   # no contact, no floor
    eq = static_equilibrium(pos, diameter, 50.0, stiffness, mass, gravity,
                            band=(0.0, 10.0))
    # |R| = m g for both -> ratio = 1
    assert eq["ratio_median"] == pytest.approx(1.0)
    assert eq["ratio_p95"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# steady state of a signal
# ---------------------------------------------------------------------------
def test_plateau_onset_locates_the_flat_part():
    t = np.linspace(0.0, 20.0, 201)
    phi = np.where(t < 10.0, 0.5 + 0.01 * t, 0.62 + 1e-4 * np.sin(3.0 * t))
    out = plateau_onset(t, phi, tol=0.002)
    assert 9.5 <= out["t_onset"] <= 10.5
    assert out["plateau_peak_to_peak"] < 0.002


def test_plateau_onset_on_linear_drift_is_only_the_last_samples():
    t = np.linspace(0.0, 20.0, 201)
    phi = 0.5 + 1e-3 * t                                  # never plateaus
    out = plateau_onset(t, phi, tol=0.002, min_samples=4)
    # the tolerated window can only span tol/slope = 2.0 time units, i.e. 10 % of
    # the record -- a drifting signal must not be reported as a plateau
    assert out["plateau_length"] <= 2.0 + 1e-9
    assert out["plateau_length"] < 0.2 * (t[-1] - t[0])


def test_plateau_drift_two_block_means():
    t = np.linspace(0.0, 20.0, 201)
    phi = np.r_[np.full(160, 0.60), np.full(41, 0.605)]   # step exactly at t = 16
    out = plateau_drift(t, phi, window=4.0)
    assert out["drift"] == pytest.approx(0.005, abs=1e-6)


# ---------------------------------------------------------------------------
# mobility
# ---------------------------------------------------------------------------
def test_mobility_separates_bulk_from_surface():
    diameter = 1.0
    pos = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 10.0]])       # floor + free particle
    vel = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])        # only the free one moves
    out = mobility_stats(pos, vel, diameter, mass=1.0, gravity=1.0,
                         band=(0.0, 2.5))
    assert out["mobile_fraction_bulk"] == pytest.approx(0.0)
    assert out["mobile_fraction_surface"] == pytest.approx(1.0)
    assert out["speed_max_over_vref"] == pytest.approx(0.5)   # v_ref = sqrt(g d) = 1


def test_mobility_is_deterministic():
    rng = np.random.default_rng(7)
    pos = np.c_[rng.uniform(0, 10, (50, 2)), rng.uniform(0, 20, 50)]
    vel = rng.normal(scale=0.01, size=(50, 3))
    a = mobility_stats(pos, vel, 1.0, 1.0, 1.0, band=(2.5, 15.0))
    b = mobility_stats(pos, vel, 1.0, 1.0, 1.0, band=(2.5, 15.0))
    assert a == b

# ---------------------------------------------------------------------------
# regression against the deposited artefacts (skipped when they are absent)
# ---------------------------------------------------------------------------
def test_coordination_matches_solver_self_report_on_stored_packing():
    """The independent contact network must reproduce the solver's own ``Z``.

    ``runs/*.npz`` stores the final positions and ``metrics.coordination_bulk``
    stores the solver's own count; agreement to ``1e-9`` is what makes the
    periodic-image contact enumeration trustworthy on the 10^4-particle case.
    """
    import json
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = os.path.join(root, "benchmarks", "case_rcp_deposition", "runs",
                       "deposition_n10000_mu0_kn4000_e0.1_seed1.json")
    if not os.path.exists(run):  # pragma: no cover - artefacts not fetched
        pytest.skip("deposition artefact not present")
    with open(run) as fh:
        d = json.load(fh)
    pos = np.load(run.replace(".json", ".npz"))["positions"]
    spec, metrics = d["spec"], d["metrics"]
    net = contact_network(pos, spec["diameter"], spec["box_width"])
    stats = coordination_stats(pos, spec["diameter"], spec["box_width"],
                               band=tuple(metrics["band"]),
                               frictionless=(spec["mu"] == 0.0), net=net)
    assert stats["z_mean"] == pytest.approx(metrics["coordination_bulk"], abs=1e-9)
