"""Invariant tests for the unbiased jamming measurement (m16308 audit).

The tests fall in two groups:

* **negative controls** that pin down the tautology of the merged
  CASE-DEM-RCP-PACKING-FRACTION gates — ``make_random_close_packing`` returns
  ``phi == 1 - target_porosity`` to machine precision, so a tolerance band on
  that number cannot measure the packing;
* **invariant tests** for :mod:`tensordem.bedcoupling.jamming`, whose output
  (residual overlap -> jamming solid fraction) is *not* imposed anywhere.
"""

import numpy as np
import pytest

from tensordem.bedcoupling.jamming import (
    ScanPoint,
    jamming_curve,
    jamming_from_scan,
    overlap_scan,
    overlap_statistics,
)
from tensordem.bedcoupling.packing import make_random_close_packing

RCP_REF_PHI = 0.6366


# ---------------------------------------------------------------------------
# negative controls: the merged gates are vacuous
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [0.30, 0.36, 0.45])
def test_achieved_solid_fraction_is_the_input(eps):
    """phi == 1 - target_porosity exactly: a gate on phi cannot fail."""
    bed = make_random_close_packing(200, 1e-3, eps, seed=42, final_relax_iterations=20)
    assert abs(bed.solid_fraction - (1.0 - eps)) < 1e-12


def test_claimed_agreement_with_rcp_is_set_by_the_target_porosity():
    """The '0.53 % vs Bernal & Mason' claim is the choice eps = 0.36."""
    bed = make_random_close_packing(200, 1e-3, 0.36, seed=42, final_relax_iterations=20)
    assert abs(abs(bed.solid_fraction - RCP_REF_PHI) - 0.0034) < 1e-3
    # ... and the same code "reproduces" any other literature value on demand
    bed2 = make_random_close_packing(200, 1e-3, 1.0 - 0.55, seed=42, final_relax_iterations=20)
    assert abs(bed2.solid_fraction - 0.55) < 1e-12


def test_n_independence_is_also_an_identity():
    phis = [
        make_random_close_packing(n, 1e-3, 0.36, seed=42, final_relax_iterations=20).solid_fraction
        for n in (150, 300)
    ]
    assert max(phis) - min(phis) < 1e-12


# ---------------------------------------------------------------------------
# invariants of the unbiased estimator
# ---------------------------------------------------------------------------
def test_overlap_statistics_on_a_lattice_is_clean():
    """A dilute lattice has no contacts and therefore no overlap."""
    bed = make_random_close_packing(150, 1e-3, 0.90, seed=1, final_relax_iterations=50)
    stats = overlap_statistics(bed)
    assert stats["max_overlap_over_diameter"] < 1e-9
    assert stats["n_contacts"] <= bed.n_particles  # no chain of overlaps in a dilute cell


def test_overlap_grows_with_imposed_solid_fraction():
    """Monotone response: imposing a denser cell must leave more overlap."""
    pts = overlap_scan(200, 1e-3, (0.55, 0.62, 0.66), seed=42, final_relax_iterations=1500)
    maxes = [p.max_overlap_over_diameter for p in pts]
    assert maxes[0] < 1e-6
    assert maxes[2] > 3.0 * maxes[1] > 0.0


def test_jamming_from_scan_recovers_a_known_crossing():
    """Forward model with an analytic crossing: phi_J must be exact."""
    pts = [
        ScanPoint(0.60, 1e-9, 0.0, 0.0, 100, 0.0),
        ScanPoint(0.62, 1e-6, 0.0, 0.0, 110, 0.0),
        ScanPoint(0.64, 1e-2, 0.0, 0.0, 120, 0.0),  # log-linear in phi below
        ScanPoint(0.66, 1e0, 0.0, 0.0, 130, 0.0),
    ]
    res = jamming_from_scan(pts, tolerance=1e-4)
    assert res["status"] == "interpolated"
    # log10(ovl) goes -6 -> -2 over phi 0.62 -> 0.64, so it crosses -4 at phi = 0.63
    assert res["phi_j"] == pytest.approx(0.63, abs=1e-6)


def test_jamming_from_scan_never_invents_a_value_outside_the_grid():
    clean = [ScanPoint(0.60, 1e-9, 0.0, 0.0, 100, 0.0), ScanPoint(0.62, 1e-9, 0.0, 0.0, 110, 0.0)]
    assert jamming_from_scan(clean, 1e-3)["status"] == "below_grid"
    dirty = [ScanPoint(0.60, 1e-1, 0.0, 0.0, 100, 0.0), ScanPoint(0.62, 1e0, 0.0, 0.0, 110, 0.0)]
    assert jamming_from_scan(dirty, 1e-3)["status"] == "above_grid"


def test_jamming_estimate_is_strictly_below_full_compression():
    """The measured jamming limit must be below the densest imposed cell: if it
    were not, the estimator would be reporting an input rather than a limit."""
    pts = overlap_scan(200, 1e-3, (0.60, 0.65), seed=42, final_relax_iterations=1500)
    curve = jamming_curve(pts, tolerances=(1e-3, 1e-2))
    phi_j = curve[f"{1e-3:.0e}"]["phi_j"]
    assert 0.55 < phi_j < 0.65
    assert phi_j < 0.65 - 1e-3


def test_tolerance_ordering_and_spread():
    """A looser tolerance can only accept a denser packing, and the spread is
    the honest uncertainty that a single number hides."""
    pts = overlap_scan(200, 1e-3, (0.60, 0.63, 0.66), seed=42, final_relax_iterations=1500)
    curve = jamming_curve(pts, tolerances=(1e-3, 1e-2))
    assert curve[f"{1e-3:.0e}"]["phi_j"] <= curve[f"{1e-2:.0e}"]["phi_j"] + 1e-9
    assert 0.0 <= curve["spread"] <= 0.05


def test_scan_is_deterministic_in_the_seed():
    a = overlap_scan(150, 1e-3, (0.62,), seed=42, final_relax_iterations=800)[0]
    b = overlap_scan(150, 1e-3, (0.62,), seed=42, final_relax_iterations=800)[0]
    assert a.max_overlap_over_diameter == b.max_overlap_over_diameter
    assert a.n_contacts == b.n_contacts