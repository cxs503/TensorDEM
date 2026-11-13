"""Tests for :mod:`tensordem.packing_measure` (analyst-independent phi).

The estimators are used to audit the RCP deposition solid fraction, so they are
themselves pinned to closed forms and to known synthetic packings:

* ``sphere_slab_volume`` vs adaptive quadrature of the exact cross-section;
* the Monte-Carlo union estimator vs closed-form union volumes (single sphere,
  two overlapping spheres via the lens volume, two spheres of different radii);
* the CIC band read-out vs the exact slab quadrature on a synthetic cloud;
* a regression test that the CIC reproduction used by the audit reproduces the
  stored ``phi_bulk`` of a merged run (skipped if the run bundle is absent).
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pytest
from scipy.integrate import quad

from tensordem.deposition_packing import packing_profile
from tensordem.packing_measure import (
    band_solid_fraction_exact,
    band_solid_fraction_mc,
    sphere_slab_volume,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "benchmarks", "case_rcp_deposition", "runs")


def _quad_slab(zc, r, a, b):
    lo = max(a - zc, -r)
    hi = min(b - zc, r)
    if hi <= lo:
        return 0.0
    val, _ = quad(lambda h: math.pi * (r * r - h * h), lo, hi, epsabs=1e-12)
    return val


@pytest.mark.parametrize("zc,r,a,b", [
    (0.5, 0.5, 0.0, 1.0),
    (0.5, 0.5, 0.2, 0.7),
    (1.0, 0.5, 1.1, 2.0),
    (-0.3, 0.5, -0.6, 0.1),
    (2.0, 0.25, 0.0, 10.0),
    (0.0, 1.0, -10.0, 10.0),
])
def test_slab_volume_matches_quadrature(zc, r, a, b):
    got = float(sphere_slab_volume(np.array([zc]), r, a, b)[0])
    assert got == pytest.approx(_quad_slab(zc, r, a, b), rel=1e-10, abs=1e-12)


def test_slab_volume_full_and_empty():
    r = 0.5
    full = float(sphere_slab_volume(np.array([0.0]), r, -1.0, 1.0)[0])
    assert full == pytest.approx(4.0 / 3.0 * math.pi * r ** 3, rel=1e-12)
    # slab entirely above the sphere
    assert float(sphere_slab_volume(np.array([0.0]), r, 0.6, 1.0)[0]) == 0.0
    # inverted slab never returns a negative volume
    assert float(sphere_slab_volume(np.array([0.0]), r, 0.4, 0.2)[0]) >= 0.0


def test_band_fraction_single_sphere_closed_form():
    """A lone sphere in a box: phi equals the exact cap volume over the slab."""
    r, box = 0.5, 4.0
    pos = np.array([[2.0, 2.0, 1.0]])
    z_lo, z_hi = 0.8, 1.3
    got = band_solid_fraction_exact(pos, r, box, z_lo, z_hi)
    want = _quad_slab(1.0, r, z_lo, z_hi) / (box * box * (z_hi - z_lo))
    assert got == pytest.approx(want, rel=1e-12)


def test_mc_single_sphere_matches_closed_form():
    r, box = 0.5, 3.0
    pos = np.array([[1.5, 1.5, 1.5]])
    z_lo, z_hi = 1.0, 2.0
    res = band_solid_fraction_mc(pos, box, z_lo, z_hi, r, n_points=200_000,
                                 seed=3)
    want = (4.0 / 3.0 * math.pi * r ** 3) / (box * box * (z_hi - z_lo))
    assert res["phi"] == pytest.approx(want, abs=4.0 * res["stderr"])


def test_mc_union_subtracts_the_lens_of_two_overlapping_spheres():
    """The union estimator must not double count a contact (unlike a volume sum)."""
    r, box = 0.5, 4.0
    delta = 0.1                       # overlap depth
    D = 2.0 * r - delta
    pos = np.array([[1.0, 2.0, 1.0], [1.0 + D, 2.0, 1.0]])
    z_lo, z_hi = 0.4, 1.6
    lens = math.pi * (4.0 * r + D) * (2.0 * r - D) ** 2 / 12.0
    union = 2.0 * 4.0 / 3.0 * math.pi * r ** 3 - lens
    want = union / (box * box * (z_hi - z_lo))
    res = band_solid_fraction_mc(pos, box, z_lo, z_hi, r, n_points=200_000,
                                 seed=5)
    summed = band_solid_fraction_exact(pos, r, box, z_lo, z_hi)
    assert res["phi"] == pytest.approx(want, abs=4.0 * res["stderr"])
    assert summed > res["phi"]          # the analytic sum double counts


def test_mc_per_particle_radii_union():
    """Per-particle radii: two disjoint spheres, union volume = sum."""
    box = 5.0
    pos = np.array([[1.0, 1.0, 2.5], [3.5, 3.5, 2.5]])
    radii = np.array([0.4, 0.7])
    z_lo, z_hi = 1.5, 3.5
    want = (4.0 / 3.0 * math.pi * (0.4 ** 3 + 0.7 ** 3)) / (box * box * (z_hi - z_lo))
    res = band_solid_fraction_mc(pos, box, z_lo, z_hi, 0.5, n_points=200_000,
                                 seed=7, radii=radii)
    assert res["phi"] == pytest.approx(want, abs=4.0 * res["stderr"])


def test_mc_empty_band_raises():
    with pytest.raises(ValueError):
        band_solid_fraction_mc(np.zeros((1, 3)), 2.0, 1.0, 1.0, 0.5,
                               n_points=100)


def test_mc_interior_margin_reports_fraction():
    rng = np.random.default_rng(11)
    box, r = 6.0, 0.5
    pos = rng.random((40, 3)) * box
    res = band_solid_fraction_mc(pos, box, 0.5, 3.0, r, n_points=120_000,
                                 seed=2, interior_margin=1.0)
    assert 0.0 <= res["phi_interior"] <= 1.0
    assert res["n_interior_points"] > 0
    assert res["n_interior_points"] < res["n_points"]


def test_mc_statistical_error_shrinks_with_more_points():
    rng = np.random.default_rng(13)
    box, r = 6.0, 0.5
    pos = rng.random((40, 3)) * box
    a = band_solid_fraction_mc(pos, box, 0.5, 3.0, r, n_points=50_000, seed=1)
    b = band_solid_fraction_mc(pos, box, 0.5, 3.0, r, n_points=200_000, seed=1)
    assert b["stderr"] < a["stderr"]
    assert b["phi"] == pytest.approx(a["phi"], abs=4.0 * a["stderr"])


def test_cic_band_mean_agrees_with_exact_slab_on_a_synthetic_cloud():
    """CIC smearing must not bias the band mean at 0.5 d resolution."""
    rng = np.random.default_rng(19)
    box, r = 10.0, 0.5
    n = 1100
    vol = 4.0 / 3.0 * math.pi * r ** 3
    pos = np.concatenate([rng.random((n, 2)) * box,
                          rng.random((n, 1)) * 8.4 + r], axis=1)
    surface = float(np.max(pos[:, 2]) + r)
    z_lo, z_hi = 1.5, surface - 1.5
    exact = band_solid_fraction_exact(pos, r, box, z_lo, z_hi)

    def cic(bin_width):
        prof = packing_profile(pos, np.full(n, vol), box ** 2, bin_width, surface)
        mask = (prof["z"] >= z_lo) & (prof["z"] <= z_hi)
        return float(np.mean(prof["phi"][mask]))

    fine = cic(0.5)
    assert fine == pytest.approx(exact, abs=5e-3)
    # the operator error is monotone in the bin width: that is the effect the
    # audit measures on the real runs (coarse bins smear the free surface)
    assert abs(cic(1.0) - exact) > abs(fine - exact)


def _stored_run_pair():
    tags = sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(os.path.join(RUNS, "*.npz")))
    tags = [t for t in tags if t.startswith("deposition_n10000_mu0_kn4000")]
    if not tags:
        pytest.skip("stored deposition runs not present")
    return tags[0]


def test_cic_reproduction_of_a_stored_production_run():
    """The audit's D1 must be exactly the merged read-out on the stored state."""
    tag = _stored_run_pair()
    meta = json.load(open(os.path.join(RUNS, tag + ".json")))
    pos = np.load(os.path.join(RUNS, tag + ".npz"))["positions"]
    spec = meta["spec"]
    r = float(spec["radius"])
    box = float(spec["box_width"])
    vol = 4.0 / 3.0 * math.pi * r ** 3
    surface = float(np.max(pos[:, 2]) + r)
    prof = packing_profile(pos, np.full(len(pos), vol), box ** 2, r, surface)
    mask = (prof["z"] >= 2.5 * 2 * r) & (prof["z"] <= surface - 2.5 * 2 * r)
    cic = float(np.mean(prof["phi"][mask]))
    assert cic == pytest.approx(float(meta["metrics"]["phi_bulk"]), abs=1e-9)
    # and the stored number is inside the acceptance band
    assert 0.62 <= cic <= 0.66