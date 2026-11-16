"""Round 15 (m20791) regression tests: solid-fraction definitions.

These tests fix the *definitions* that rounds 6-14 conflated.  They are written
against the shipped engine functions (``tensordem.mu_i_rheology.phi_definitions``)
and against a device run, so a future change that silently re-mixes the
definitions fails here rather than in a ledger verdict.

No literature constant is involved: every assertion is either an analytic
packing with a known solid fraction or an internal-consistency requirement.
"""
from __future__ import annotations

import json
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
from phi_estimator_audit import (  # noqa: E402
    BOUNDARY_TOL_PCT, RUNS, archived_slab_phi, homogeneous_packing,
    synthetic_audit)

RADIUS = 0.005
BOUNDARY_TOL = BOUNDARY_TOL_PCT / 100.0


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
    # Round 15-c (m23285): on the device the z-wall shell must *bound* the
    # measured gap from above (the grafted bed sits denser against the y-walls
    # than a geometric shell allows), and the two-wall product must overshoot.
    last = series[-1]
    ratio = last["slab_over_bed"] / last["boundary_layer_factor"]
    assert 0.85 < ratio <= 1.05                    # predeclared interval
    assert last["slab_over_bed"] < last["boundary_layer_factor_2wall"]
    assert last["boundary_layer_factor_2wall"] == pytest.approx(
        last["boundary_layer_factor"] * last["boundary_layer_factor_y"])
    assert last["slab_over_z_bound"] == pytest.approx(ratio)
    # the archived key keeps its historical meaning (no silent redefinition)
    assert res["solid_fraction"] == pytest.approx(phi_cen, rel=1e-12)


# --------------------------------------------------------------------------
# round 15-c (m23285): the shell accounting has two walled pairs, not one
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n,height", [(2860, 0.12), (1430, 0.06), (1073, 0.045)])
def test_two_wall_boundary_factor_reproduces_the_measured_gap(n, height):
    """z x y wall shells must predict the measured slab/container gap to <5 %.

    This is the round-15-c correction: the round-15 model used the z factor
    alone and left a 9.7-12.6 % residual (its own artifact carried
    ``ok: false``); the product of both walled pairs closes it to <1.5 %.
    """
    pos, phi_true, bw, bd, h = homogeneous_packing(n=n, height=height)
    leg = archived_slab_phi(pos, box_width=bw, box_depth=bd, plate_z=h)
    d = phi_definitions(pos, RADIUS, bw, bd, h, n_slab=leg["n"],
                        slab_volume=leg["volume"])
    assert d["boundary_factor_y"] == pytest.approx(bd / (bd - 2 * RADIUS))
    assert d["boundary_factor_2wall"] == pytest.approx(
        d["boundary_factor_z"] * d["boundary_factor_y"])
    assert abs(d["slab_over_bed"] / d["boundary_factor_2wall"] - 1.0) \
        < BOUNDARY_TOL
    # and the incumbent z-only model is demonstrably insufficient here, so the
    # correction is a correction and not a tolerance widening
    assert d["slab_over_bed"] / d["boundary_factor_z"] - 1.0 > BOUNDARY_TOL


def test_synthetic_audit_artifact_passes_its_own_criterion():
    """The committed artifact must carry ok=True and be bit-reproducible.

    Round 15 shipped ``ok: false, verdict: "unexpected"`` in
    runs/phi_audit/synthetic.json without any test noticing; this pins it.
    """
    path = RUNS / "synthetic.json"
    before = json.loads(path.read_text()) if path.exists() else None
    fresh = synthetic_audit(verbose=False)
    assert fresh["ok"] is True, fresh["failed_checks"]
    assert all(fresh["checks"].values())
    assert "two_wall" in fresh["verdict"]
    if before is not None:
        assert [r["measured_slab_over_bed"] for r in before["rows"]] == \
            [r["measured_slab_over_bed"] for r in fresh["rows"]]
        assert before["ok"] is True


def test_archived_rederivation_is_reproducible():
    """Re-deriving the archived points must reproduce the committed numbers."""
    import subprocess
    here = Path(__file__).resolve().parent
    out = RUNS / "archived_rederived.json"
    committed = json.loads(out.read_text()) if out.exists() else None
    subprocess.run([sys.executable, str(here / "rederive_archived_phi.py")],
                   check=True, capture_output=True)
    rows = json.loads(out.read_text())
    assert len(rows) == 4
    for r in rows:
        # the archived estimator is biased high by the z-shell; the corrected
        # container definition is what the reference law is compared with
        assert r["slab_over_bed_measured"] > 1.05
        assert 0.85 < r["slab_over_z_only"] <= 1.05
        assert r["boundary_factor_2wall"] > r["slab_over_bed_measured"]
    if committed is not None:
        assert [r["phi_container"] for r in committed] == \
            [r["phi_container"] for r in rows]
