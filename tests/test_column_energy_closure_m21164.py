"""Regression tests for the column-collapse energy-ledger repair (m21164).

Anchor ``CASE-COLUMN-COLLAPSE``.  Pre-registration:
``docs/PREREG_COLUMN_ENERGY_CLOSURE_m21164.md``.

The defect: the axisymmetric / 3-D column integrators accumulated five of the six
dissipation channels -- ``cur_wall["dissip_rolling"]`` (rolling resistance at the
floor) was never added to the ledger, although the corresponding torque *was*
applied to the particles.  A single grain rolling on the floor therefore lost its
whole kinetic energy to a channel the ledger never saw (~-15 % of E0 imbalance).
The ``release_loss`` bookkeeping also omitted the rolling spring store.

These tests pin the repair from three sides:

1. ``test_*_canonical`` -- a hand-rolled one-grain integrator using the public
   kernel closes to <0.5 % of E0 *only* when the wall rolling channel is summed;
   dropping the channel reproduces the old >10 % imbalance (negative control).
2. ``test_forces_untouched`` -- the repair may not move any grain: the pre-repair
   trajectory fixture (``docs/artifacts/column_energy_closure_m21164/``) must be
   reproduced bit for bit.
3. ``test_release_loss_books_whole_store`` -- ``release_loss`` books the complete
   stored spring energy (normal + tangential + rolling) of a contact that leaves
   the neighbour list.
"""

from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import pytest

from tensordem.granular3d_core import Material3D
from tensordem.granular_column import axisym as ax
from tensordem.granular_column.kernel import (
    PairSpringState, WallSpringState, build_pairs,
    pair_contact_forces, wall_contact_forces,
)

ART = os.path.join(os.path.dirname(__file__), "..", "docs", "artifacts",
                   "column_energy_closure_m21164")


def _material(mu: float = 0.5, mu_rolling: float = 0.5) -> Material3D:
    return Material3D(radius=0.06, density=2500.0, mu=mu, mu_rolling=mu_rolling,
                      restitution=0.5, kn=6.0e4 * (0.06 / 0.03) ** 2,
                      kt_ratio=0.5, gravity=9.81)


def _one_grain_budget(mat: Material3D, *, vx: float = 0.5, spin: bool = True,
                      n_steps: int = 200) -> dict:
    """Minimal velocity-Verlet copy of the column loop, one grain on the floor."""
    r = mat.radius
    pos = np.array([[0.0, 0.0, r]])
    vel = np.array([[vx, 0.0, 0.0]])
    omega = np.array([[0.0, (vx / r if spin else 0.0), 0.0]])   # pure rolling
    mat_dt = mat.timestep
    gravity_vec = np.array([0.0, 0.0, -mat.gravity])
    pair_state, wall_state = PairSpringState(), WallSpringState(1)
    mass, inertia = mat.mass, mat.inertia

    U0 = float(mass * mat.gravity * pos[:, 2].sum())
    K0 = float(0.5 * mass * (vel ** 2).sum() + 0.5 * inertia * (omega ** 2).sum())
    diss_all = 0.0      # six channels (the repaired ledger)
    diss_legacy = 0.0   # five channels (the pre-repair ledger)
    for _ in range(n_steps):
        iu, ju = build_pairs(pos, r)
        pred = pair_contact_forces(pos, vel, omega, iu, ju, pair_state, mat, mat_dt,
                                   update_state=False)
        pw = wall_contact_forces(pos, vel, omega, mat, mat_dt, mu_wall=mat.mu,
                                 mu_rolling_wall=mat.mu_rolling, state=wall_state,
                                 update_state=False)
        acc = (pred["force"] + pw["force"]) / mass + gravity_vec
        alpha = (pred["torque"] + pw["torque"]) / inertia
        v_h = vel + 0.5 * mat_dt * acc
        w_h = omega + 0.5 * mat_dt * alpha
        pos = pos + v_h * mat_dt
        cur = pair_contact_forces(pos, v_h, w_h, iu, ju, pair_state, mat, mat_dt,
                                  update_state=True)
        cw = wall_contact_forces(pos, v_h, w_h, mat, mat_dt, mu_wall=mat.mu,
                                 mu_rolling_wall=mat.mu_rolling, state=wall_state,
                                 update_state=True)
        acc = (cur["force"] + cw["force"]) / mass + gravity_vec
        alpha = (cur["torque"] + cw["torque"]) / inertia
        vel = v_h + 0.5 * mat_dt * acc
        omega = w_h + 0.5 * mat_dt * alpha
        diss_all += (cur["dissip_collision"] + cur["dissip_friction"]
                     + cur["dissip_rolling"] + cw["dissip_collision"]
                     + cw["dissip_friction"] + cw["dissip_rolling"])
        diss_legacy += (cur["dissip_collision"] + cur["dissip_friction"]
                        + cur["dissip_rolling"] + cw["dissip_collision"]
                        + cw["dissip_friction"])
        store = (cur["energy_normal"] + cur["energy_tangential"]
                 + cur["energy_rolling"] + cw["energy_normal"]
                 + cw["energy_tangential"] + cw["energy_rolling"])
        cutoff = cur["release_loss"] + cw["release_loss"]
    U1 = float(mass * mat.gravity * pos[:, 2].sum())
    K1 = float(0.5 * mass * (vel ** 2).sum() + 0.5 * inertia * (omega ** 2).sum())
    e0 = U0 + K0
    common = (U0 - U1) - (K1 - K0) - store - cutoff
    return {
        "e0": e0,
        "residual_all": common - diss_all,
        "residual_legacy": common - diss_legacy,
        "wall_rolling": cw["dissip_rolling"],
        "kinetic_lost": K0 - K1,
    }


def test_canonical_closure_needs_the_wall_rolling_channel():
    """Pure rolling: closure <0.5 % of E0, and >10 % without the channel.

    The second assertion is the negative control: it reproduces the pre-repair
    imbalance by simply *not* summing the wall rolling channel.
    """
    b = _one_grain_budget(_material(), spin=True)
    rel_all = abs(b["residual_all"]) / b["e0"]
    rel_legacy = abs(b["residual_legacy"]) / b["e0"]
    assert b["wall_rolling"] > 0.0, "wall rolling resistance must dissipate"
    assert rel_all < 5.0e-3, f"repaired ledger does not close: {rel_all:.4%}"
    assert rel_legacy > 0.10, (
        f"negative control failed: dropping the wall rolling channel must reopen "
        f"the imbalance, got {rel_legacy:.4%}")


def test_canonical_closure_from_rest_spin():
    """Released without spin: friction spins the grain up, rolling resistance then
    acts -- the repaired ledger closes, the legacy one does not."""
    b = _one_grain_budget(_material(), spin=False)
    rel_all = abs(b["residual_all"]) / b["e0"]
    rel_legacy = abs(b["residual_legacy"]) / b["e0"]
    assert b["wall_rolling"] == pytest.approx(0.0, abs=1e-12)   # no initial spin
    assert rel_all < 5.0e-3, f"repaired ledger does not close: {rel_all:.4%}"
    assert rel_legacy > 5.0 * rel_all


def test_canonical_closure_without_rolling_resistance_is_also_closed():
    """``mu_rolling = 0`` must not have been 'fixed' by the missing channel."""
    b = _one_grain_budget(_material(mu_rolling=0.0), spin=False)
    assert b["wall_rolling"] == pytest.approx(0.0, abs=1e-15)
    assert abs(b["residual_all"]) / b["e0"] < 5.0e-3


def test_forces_untouched_by_the_ledger_repair():
    """The repair is accounting-only: reproduce the pre-repair trajectory exactly."""
    path = os.path.join(ART, "prefix_trajectory_a3_g4.json")
    with open(path) as fh:
        fix = json.load(fh)
    case = ax.AxisymCase(aspect_ratio=3.0, grains_per_radius=4.0,
                         radius=0.48 / (2.0 * 4.0), kn=6.0e4 * ((0.48 / 8.0) / 0.03) ** 2,
                         mu=0.5, mu_rolling=0.5, mu_wall=0.8, restitution=0.5,
                         kt_ratio=0.5, density=2500.0, gravity=9.81, packing_gap=0.02,
                         jitter=0, duration_factor=8.0, min_time=0.5, sample_stride=5,
                         rest_ke_tol=1.0e-4, rest_hold=200, contact_mode="reference")
    lib = ax.simulate_axisymmetric_column(case, max_steps=fix["n_steps"],
                                          return_positions=True)
    pos = np.array(lib["positions"])
    assert hashlib.sha256(pos.tobytes()).hexdigest() == fix["positions_sha256"]
    assert lib["runout_ratio_p995"] == pytest.approx(fix["runout_ratio_p995"],
                                                     rel=0, abs=0)


def test_ledger_exposes_the_wall_rolling_channel():
    """The column ledger must report the channel separately and inside the total."""
    case = ax.AxisymCase(aspect_ratio=1.0, grains_per_radius=3.0,
                         radius=0.48 / (2.0 * 3.0), kn=6.0e4 * ((0.48 / 6.0) / 0.03) ** 2,
                         mu=0.5, mu_rolling=0.5, mu_wall=0.8, restitution=0.5,
                         kt_ratio=0.5, density=2500.0, gravity=9.81, packing_gap=0.02,
                         jitter=0, duration_factor=8.0, min_time=0.5, sample_stride=5,
                         rest_ke_tol=1.0e-4, rest_hold=200, contact_mode="reference")
    out = ax.simulate_axisymmetric_column(case, max_steps=250, return_positions=False)
    en = out["energy"]
    assert "dissipated_wall_rolling" in en
    assert en["dissipated_wall_rolling"] > 0.0
    assert en["dissipated_friction"] >= en["dissipated_wall_rolling"]
    assert en["dissipated_total"] == pytest.approx(
        en["dissipated_collision"] + en["dissipated_friction"]
        + en["dissipated_cutoff"], rel=1e-12)
    # the repaired ledger closes on this (still short) run
    assert abs(en["residual_fraction_of_potential"]) < 0.05


def test_release_loss_books_the_whole_store_including_rolling():
    """A contact that leaves the list books normal + tangential + *rolling* store."""
    mat = _material(mu=0.0, mu_rolling=0.5)
    r, dt = mat.radius, 1.0e-6
    state = PairSpringState()
    # two grains in contact, sliding + spinning -> non-zero tang/roll springs
    pos = np.array([[0.0, 0.0, r], [2.0 * r - 0.01, 0.0, r]])
    vel = np.array([[0.0, 0.3, 0.0], [0.0, -0.3, 0.0]])
    omega = np.array([[0.0, 0.0, 40.0], [0.0, 0.0, -40.0]])
    iu, ju = build_pairs(pos, r)
    prev = pair_contact_forces(pos, vel, omega, iu, ju, state, mat, dt,
                               update_state=True)
    store = (prev["energy_normal"] + prev["energy_tangential"]
             + prev["energy_rolling"])
    assert prev["energy_rolling"] > 0.0
    # separate the pair beyond the neighbour cutoff -> its store must be booked
    far = np.array([[0.0, 0.0, r], [50.0 * r, 0.0, r]])
    iu2, ju2 = build_pairs(far, r)
    assert iu2.size == 0
    out = pair_contact_forces(far, vel, omega, iu2, ju2, state, mat, dt,
                              update_state=True)
    assert out["release_loss"] == pytest.approx(store, rel=1e-9)


def test_remaining_residual_contracts_with_timestep():
    """Pre-registered H2: after the repair what is left is O(dt), not a leak.

    Reads the archived closure runs (``closure_runs.json``, produced by
    ``scripts/audit_column_energy_closure_m21164.py``) and checks the
    pre-registered contraction band 0.35..0.65 per halving of dt.
    """
    path = os.path.join(ART, "closure_runs.json")
    if not os.path.exists(path):
        pytest.skip("closure_runs.json not archived in this checkout")
    with open(path) as fh:
        runs = json.load(fh)
    r1 = runs["C4_col_a3_g4"]["R"]
    r2 = runs["C5_col_a3_g4_dt2"]["R"]
    r3 = runs["C6_col_a3_g4_dt4"]["R"]
    assert r1 > r2 > r3, "residual must contract with dt"
    for a, b in ((r1, r2), (r2, r3)):
        ratio = b / a
        assert 0.35 <= ratio <= 0.65, f"contraction ratio {ratio:.3f} out of band"