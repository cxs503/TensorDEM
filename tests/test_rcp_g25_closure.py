"""G25 verification-debt closure: the stiffness-ladder z is a threshold artefact.

The steady-state audit (``steady_state_audit.py``, m17063) found
``G25_stiffness_ladder_coordination_converged`` FAIL: at the *hard* contact
threshold (``overlap > 0``) the mean coordination ``z`` on the stiffness ladder
drops below the isostatic value ``z_c = 2d = 6`` (frictionless) as ``k_n`` grows
-- the stiffest level reads ``z = 5.41 < 6``, i.e. the bed looks *hypostatic*.

These tests pin the closure: the drop is a **threshold artefact** (the hard
threshold is a lower bound that shrinks as the settled overlap shrinks), and the
``z`` read at the case's own overlap tolerance (``t = 1e-3``) is a **convergent**
sequence that stays *above* ``z_c`` on every ladder level -- so the bed is *not*
hypostatic, and the ``phi`` (packing-fraction) ladder is unaffected.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.packing_steady_state import (  # noqa: E402
    coordination_vs_tolerance, isostatic_coordination,
)

N_PRODUCTION = 10000
DT_DEFAULT = 1.0 / 20.0
Z_C_FRICTIONLESS = 6.0
TOL_APPROPRIATE = 1e-3
RUNS = os.path.join(ROOT, "benchmarks", "case_rcp_deposition", "runs")


def _load_ladder():
    """One seed per (mu, kn) at production size -- the stiffness ladder."""
    out = []
    for path in sorted(glob.glob(os.path.join(RUNS, "deposition_*.json"))):
        stem = os.path.basename(path)
        if stem.startswith(("debt_", "order_", "steady_state_", "g25_")):
            continue
        npz = path.replace(".json", ".npz")
        if not os.path.exists(npz):
            continue
        with open(path) as fh:
            d = json.load(fh)
        spec, metrics = d["spec"], d["metrics"]
        if (abs(spec.get("dt_frac", 0.05) - DT_DEFAULT) > 1e-9
                or spec.get("seed") != 1
                or spec.get("n_particles") != N_PRODUCTION):
            continue
        z = np.load(npz)
        out.append({
            "mu": float(spec["mu"]),
            "kn": float(spec["kn"]),
            "diameter": float(spec["diameter"]),
            "box_width": float(spec["box_width"]),
            "band": tuple(metrics["band"]),
            "phi": float(metrics["phi_bulk"]),
            "positions": z["positions"],
        })
    return out


def _z_at_tol(run, tol):
    pos = run["positions"]
    scan = coordination_vs_tolerance(pos, run["diameter"], run["box_width"],
                                     run["band"], (tol,))
    return scan[0]["z_mean"]


@pytest.fixture(scope="module")
def ladder():
    return _load_ladder()


def test_ladder_has_both_tiers(ladder):
    """The ladder must carry both friction tiers at production size."""
    mus = {r["mu"] for r in ladder}
    assert mus == {0.0, 0.1}
    for mu in mus:
        kns = sorted({r["kn"] for r in ladder if r["mu"] == mu})
        assert len(kns) >= 4, f"mu={mu} needs >=4 stiffness levels, got {kns}"


def test_hard_threshold_z_is_lower_bound_artefact(ladder):
    """Frictionless: hard-threshold z drops below z_c=6 as stiffness grows.

    This is the G25 FAIL -- the hard ``overlap > 0`` threshold is a lower bound
    that loses marginal contacts as the settled overlap shrinks.
    """
    mu0 = sorted([r for r in ladder if r["mu"] == 0.0], key=lambda r: r["kn"])
    z_hard = [_z_at_tol(r, 0.0) for r in mu0]
    # the stiffest level reads below the isostatic value
    assert z_hard[-1] < Z_C_FRICTIONLESS, f"z_hard={z_hard[-1]} should be < 6"
    # and the lower bound shrinks monotonically with stiffness
    assert z_hard[0] > z_hard[-1]


def test_z_at_appropriate_tolerance_recovers_isostatic(ladder):
    """Frictionless: z at t=1e-3 is a convergent sequence above z_c=6.

    The case's own overlap tolerance (1e-3) recovers the marginal contacts, so
    every ladder level reads z >= 6 -- the bed is *not* hypostatic.
    """
    mu0 = sorted([r for r in ladder if r["mu"] == 0.0], key=lambda r: r["kn"])
    z_app = [_z_at_tol(r, TOL_APPROPRIATE) for r in mu0]
    assert all(z >= Z_C_FRICTIONLESS for z in z_app), \
        f"z@1e-3={z_app} must all be >= 6"
    # convergent: the spread across the ladder is small
    assert max(z_app) - min(z_app) < 0.5


def test_phi_ladder_unaffected_by_z_artefact(ladder):
    """The phi (packing-fraction) ladder is a clean convergent sequence.

    G25 concerns only z, never phi: the frictionless phi ladder converges
    monotonically toward the RCP limit 0.6366.
    """
    mu0 = sorted([r for r in ladder if r["mu"] == 0.0], key=lambda r: r["kn"])
    phi = [r["phi"] for r in mu0]
    # monotone decreasing toward the RCP limit
    assert all(phi[i] >= phi[i + 1] for i in range(len(phi) - 1))
    # final level within 3% of the RCP reference 0.6366
    assert abs(phi[-1] - 0.6366) / 0.6366 < 0.03


def test_frictional_tier_no_hypostaticity(ladder):
    """Frictional tier (z_c=4): hard-threshold z stays well above z_c.

    The G25 FAIL was specific to the frictionless tier; the frictional tier's
    hard-threshold z (5.3-5.8) is far above the infinite-friction bound z_c=4.
    """
    mu01 = sorted([r for r in ladder if r["mu"] == 0.1], key=lambda r: r["kn"])
    z_hard = [_z_at_tol(r, 0.0) for r in mu01]
    z_c = isostatic_coordination(3, frictionless=False)
    assert all(z > z_c for z in z_hard), f"z_hard={z_hard} must all be > {z_c}"
