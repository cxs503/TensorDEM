"""Round 15 (m20791) regression tests: solid-fraction definitions.

These tests fix the *definitions* that rounds 6-14 conflated.  They are written
against the shipped engine functions (``tensordem.mu_i_rheology.phi_definitions``)
and against a device run, so a future change that silently re-mixes the
definitions fails here rather than in a ledger verdict.

No literature constant is involved: every assertion is either an analytic
packing with a known solid fraction or an internal-consistency requirement.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (REPO / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tensordem.mu_i_rheology import (  # noqa: E402
    ServoShearCase, phi_definitions, simulate_servo_shear)
from phi_estimator_audit import archived_slab_phi, homogeneous_packing  # noqa: E402

RADIUS = 0.005


@pytest.mark.parametrize("n,height", [(2860, 0.12), (1430, 0.06), (1073, 0.045)])
def test_container_and_extent_estimators_are_unbiased(n, height):
    """phi_bed and phi_extent must reproduce the analytic solid fraction."""
    pos, phi_true, bw, bd, h = homogeneous_packing(n=n, height=height)
    leg = archived_slab_phi(pos, box_width=bw, box_depth=bd, plate_z=h)
    d = phi_definitions(pos, RADIUS, bw, bd, h, n_slab=leg["n"],
                        slab_volume=leg["volume"])
    assert abs(d["phi_bed"] / phi_true - 1.0) < 0.02
    assert abs(d["phi_extent"] / phi_true - 1.0) < 0.03
    # the centre-counted slab is a *different* quantity: biased high, and the
    # bias grows as the gap narrows -- the mechanism that manufactured the
    # archived "phi rises with I" trend.
    assert d["phi_centres"] / phi_true - 1.0 > 0.10
    assert d["slab_over_bed"] > 1.05


def test_boundary_layer_factor_explains_the_definitional_gap():
    """The measured slab/bed gap is dominated by plate_z/(plate_z-2r)."""
    ratios = []
    for n, height in ((2860, 0.12), (1430, 0.06), (1073, 0.045)):
        pos, _, bw, bd, h = homogeneous_packing(n=n, height=height)
        leg = archived_slab_phi(pos, box_width=bw, box_depth=bd, plate_z=h)
        d = phi_definitions(pos, RADIUS, bw, bd, h, n_slab=leg["n"],
                            slab_volume=leg["volume"])
        ratios.append(d["slab_over_bed"] / d["boundary_factor"])
    ratios = np.array(ratios)
    # the z-boundary factor must capture 85-115 % of the measured gap: any
    # remaining part comes from the y-walls and finite-size heterogeneity.
    assert np.all(ratios > 0.85)
    assert np.all(ratios < 1.15)
    assert np.all(np.diff(ratios) < 0.05)     # gap grows as the gap narrows


def test_live_device_phi_definitions_are_consistent():
    """On a real (short) device run the two physical estimators must agree."""
    case = ServoShearCase(
        radius=RADIUS, density=2500.0, mu=0.5, mu_rolling=0.05, restitution=0.3,
        kn=6.25e4, kt_ratio=0.5, box_width=0.12, box_depth=0.05,
        fill_height=0.06, normal_stress=6.25e4, inertial_number=1e-2, seed=5279,
        settle_steps=400, min_shear_steps=300, shear_strain=0.05,
        sample_stride=50, bulk_stress=False, wall_mode="smooth")
    res = simulate_servo_shear(case, return_series=True)
    phi_bed = res["phi_container_final"]
    phi_ext = res["phi_extent_final"]
    phi_cen = res["phi_centres_slab"]
    assert 0.30 < phi_bed < 0.75                       # physical bound
    assert abs(phi_ext - phi_bed) / phi_bed < 0.06     # platen bookkeeping OK
    assert phi_cen > phi_bed * 1.05                    # definitions must differ
    series = res["series"]
    assert all(math.isfinite(s["slab_over_bed"]) for s in series[1:])
    # the archived key keeps its historical meaning (no silent redefinition)
    assert res["solid_fraction"] == pytest.approx(phi_cen, rel=1e-12)