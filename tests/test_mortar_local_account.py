"""Unit tests for the patch-local mortar account (p423w2-n07).

The physical statements under test:

* the 30 degree crease chart is isometric (``J == 1``) and Newton-free
  projection lands on it exactly, so a ``C^0`` chart is admissible;
* the per-facet (not just global) coverage identity holds on the crease at
  round-off, whereas the plane-projection construction fails it;
* the gap / penetration volumes close the identity, and dropping them leaves the
  defect visible (negative control);
* the time-domain interface power ``P_int`` -- the energy a coupled solver
  injects -- is zero for a locally conserved operator and ``> 1e-3`` for a
  deliberately broken one;
* the transpose-consistent adjoint transports a smooth field at second order.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensordem.interface_operators import mortar_local_account as M
from tensordem.interface_operators.curved_patch_mortar import (
    CurvedMortarOperator,
    GraphPatch,
    per_patch_balance,
)


def test_fold_chart_is_isometric_and_projection_is_exact():
    fold = M.FoldPatch(30.0)
    uv = np.array([[0.13, 0.2], [0.5, 0.4], [0.87, 0.71], [0.49, 0.9]])
    x = fold.x(uv)
    assert np.allclose(fold.jac(uv), 1.0)
    # J = 1 => the chart is an isometry: the patch area is the parameter area
    assert fold.area() == pytest.approx(1.0, abs=1e-12)
    uv_back, dist, _it, gnorm = fold.project(x)
    assert np.allclose(uv_back, uv, atol=1e-12)
    assert np.max(np.asarray(dist)) < 1e-12
    assert np.max(np.asarray(gnorm)) < 1e-12
    # the crease is continuous (C^0) even though x_u jumps
    assert np.allclose(fold.x([[0.5, 0.3]]), fold.x([[0.5000001, 0.3]]), atol=1e-6)
    assert fold.curvature_scale == pytest.approx(math.tan(math.radians(15.0)))


@pytest.mark.parametrize("patch", [M.FoldPatch(30.0), GraphPatch()])
def test_per_facet_gate_and_projection_control(patch):
    master, slave = M.mismatch_case(patch, 4, 2)
    bal = per_patch_balance(master, slave)
    assert bal["per_facet_max_rel"] <= 1e-12          # every facet, not the sum
    assert bal["n_uncovered"] == 0
    r = M.local_information_loss(patch, 4, 2)
    assert r["true_per_facet"] <= 1e-12
    assert r["plane_per_facet"] >= 1e-3               # the control must be visible


def test_gap_penetration_identity_and_dropped_term_control():
    acc = M.gap_penetration_account(M.FoldPatch(30.0), n=3)
    assert acc["max_identity_balance_rel"] <= 1e-12
    # the conforming case must not invent a gap / penetration volume out of sliver noise
    conf = [r for r in acc["rows"] if r["case"] == "conforming"][0]
    assert conf["gap_unit_volume_rel"] <= 1e-12
    assert conf["pen_unit_volume_rel"] <= 1e-12
    # the gap / penetration cases are *non-zero by construction* and must be booked
    assert [r for r in acc["rows"] if r["case"] == "gap"][0]["gap_unit_volume_rel"] >= 1e-2
    assert [r for r in acc["rows"] if r["case"] == "penetration"][0]["pen_unit_volume_rel"] >= 1e-2
    # negative control: without the gap / penetration volumes the identity fails
    assert acc["control_residual_gap_case"] >= 1e-3
    assert acc["control_residual_pen_case"] >= 1e-3


def test_energy_account_conserving_vs_broken():
    patch = M.FoldPatch(30.0)
    master, slave = M.mismatch_case(patch, 4, 2)
    op = CurvedMortarOperator(master, slave, order=2, measure="curved")
    good = M.interface_energy_account(op, dt=1e-3, nsteps=120)
    assert good["per_facet_defect"] <= 1e-12
    assert good["interface_power_rel_max"] <= 1e-10     # no artificial injection
    assert good["lam_max_uniform"] <= 1e-12
    assert good["ledger_residual_rel_max"] <= 1e-10
    broken = M.interface_energy_account(M.break_local_conservation(op), dt=1e-3, nsteps=120)
    assert broken["per_facet_defect"] >= 1e-3
    assert broken["interface_power_rel_max"] >= 1e-3    # the gate can fail
    # a broken measure (curvature ignored) also injects power at the interface
    gm, gs = M.mismatch_case(GraphPatch(), 4, 1)
    flat = M.interface_energy_account(
        CurvedMortarOperator(gm, gs, order=2, measure="flat"), dt=1e-3, nsteps=120)
    assert flat["lam_max_uniform"] >= 1e-3


def test_transpose_consistency_second_order():
    study = M.transpose_consistency_study(M.FoldPatch(30.0), meshes=(6, 8, 12, 16), ratio=1)
    assert study["duality_rel_max"] <= 1e-12            # discrete symmetry
    assert study["rate_transport_curved"] >= 1.8        # adjoint transport is 2nd order
    assert study["transport_rel_naive_finest"] > study["transport_rel_curved_finest"]


def test_determinism_and_provenance():
    a = M.local_information_loss(M.FoldPatch(30.0), 3, 2)
    b = M.local_information_loss(M.FoldPatch(30.0), 3, 2)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    prov = M.provenance()
    assert prov["module"] == "mortar_local_account.py"
    assert len(prov["module_sha256"]) == 64
    assert "git_head" in prov


def test_acceptance_smoke_all_criteria_present():
    acc = M.run_acceptance(meshes=(4,), energy_resolutions=(4,), energy_ratios=(1,),
                           e_meshes=(4, 6), with_regression=False)
    for key in ("A_per_facet_conservation", "B_true_intersection_vs_projection",
                "C_gap_penetration_terms", "D_time_domain_energy_account",
                "E_transpose_consistency"):
        assert key in acc["criteria"]
    assert acc["criteria"]["A_per_facet_conservation"]["pass"]
    assert acc["criteria"]["C_gap_penetration_terms"]["pass"]
    assert acc["criteria"]["D_time_domain_energy_account"]["pass"]