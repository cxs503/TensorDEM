"""P82W5-N07 acceptance tests -- the multi-rate contact-integration gate.

The suite is deliberately split in two:

* **Numerical identity of the engineering change.**  ``_fast_step`` (the fused
  kernel) and ``_fast_pass`` (the reference composition of
  ``_pair_geometry``/``_pair_kinematics``/``_tangential_force``) must integrate
  the *same* contact laws: force, torque and the tangential/rolling internal
  variables have to agree to round-off on the same state.  The wall contacts
  must likewise be identical whether they are evaluated once per coarse step
  (``wall_rate="slow"``, the matured assignment, a wall spring is ``kn``) or in
  every sub-step (``wall_rate="fast"``, the legacy ablation).
* **The acceptance contract.**  The stability gate must reject
  ``dt/dt_c = 0.5``; the restitution ladder must stay inside 2 %; the
  undamped ledger must conserve linear momentum to round-off; and the *rate
  contract* (``fast`` = ``n_sub`` evaluations per coarse step, ``slow`` and
  ``wall`` = one) must be observable from the work counters.  The wall-clock
  criterion itself lives in ``benchmarks/mts_speedup_gate.py`` (it needs
  minutes, not milliseconds).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tensordem.mts import (
    TIGHT_RATIO,
    InadmissibleTimeStep,
    MTSConfig,
    MTSDisks,
    alpha_for_restitution,
    impact_restitution,
    stability_gate,
)
from tensordem.hertz_impact_time import hertz_contact_time

KN = 3.0e4


# --------------------------------------------------------------------- rigs
def _periodic_rig(*, kernel: str, mu: float = 0.5) -> MTSDisks:
    """Doubly periodic bidisperse cell with a 100:1 stiffness skeleton."""
    from tensordem.granular2d.jamming import bidisperse_radii

    radii = bidisperse_radii(48, 0.010, seed=1)
    side = math.sqrt(float(np.sum(math.pi * radii ** 2)) / 0.80)
    rng = np.random.default_rng(7)
    rmax = float(radii.max())
    pos = rmax + rng.random((len(radii), 2)) * (side - 2.0 * rmax)
    cfg = MTSConfig(kn=KN, mu=mu, mu_rolling=0.05, zeta=0.4, stiff_ratio=100.0,
                    stiff_fraction=0.15, fast_kernel=kernel)
    sys_ = MTSDisks(pos, radii, cfg, box=(side, side), periodic=True)
    sys_.vel = rng.normal(0.0, 0.05, size=sys_.vel.shape)
    sys_.omega = rng.normal(0.0, 5.0, size=sys_.omega.shape)
    sys_.setup_timestep(mode="mts")
    sys_.step(30)
    return sys_


def _walled_rig(*, kernel: str, wall_rate: str) -> MTSDisks:
    """Square-walled cell (the jamming geometry) with the same skeleton."""
    from tensordem.granular2d.jamming import bidisperse_radii

    radii = bidisperse_radii(48, 0.010, seed=2)
    side = math.sqrt(float(np.sum(math.pi * radii ** 2)) / 0.78)
    rng = np.random.default_rng(11)
    rmax = float(radii.max())
    pos = rmax + rng.random((len(radii), 2)) * (side - 2.0 * rmax)
    cfg = MTSConfig(kn=KN, mu=0.5, mu_rolling=0.05, zeta=0.4, stiff_ratio=100.0,
                    stiff_fraction=0.15, fast_kernel=kernel, wall_rate=wall_rate)
    sys_ = MTSDisks(pos, radii, cfg, box=(side, side))
    sys_.vel = rng.normal(0.0, 0.05, size=sys_.vel.shape)
    sys_.setup_timestep(mode="mts")
    sys_.step(20)
    return sys_


# ------------------------------------------------- numerical identity checks
def test_fused_kernel_reproduces_the_reference_contact_laws():
    """``_fast_step`` == ``_fast_pass`` for force, torque and internal variables."""
    lean = _periodic_rig(kernel="lean")
    lean._rebuild(force=True)
    lean._slow_pass(lean.dt)
    lean._prepare_fast()
    h = lean.dt / lean.n_sub
    assert lean._f_n > 0, "the rig must contain stiff contacts for this test"

    xi0 = lean._xi.copy()
    roll0 = lean._roll.copy()

    ref_force, ref_torque = lean._fast_pass(h)
    ref_xi = lean._xi.copy()
    ref_roll = lean._roll.copy()

    lean._xi = xi0.copy()
    lean._roll = roll0.copy()
    new_force, new_torque = lean._fast_step(h)

    scale_f = float(np.abs(ref_force).max())
    scale_t = float(np.abs(ref_torque).max())
    assert np.abs(new_force - ref_force).max() <= 1e-11 * max(scale_f, 1.0)
    assert np.abs(new_torque - ref_torque).max() <= 1e-11 * max(scale_t, 1.0)
    assert np.abs(lean._xi - ref_xi).max() <= 1e-13
    assert np.abs(lean._roll - ref_roll).max() <= 1e-13


def test_fused_kernel_is_not_silently_frictionless():
    """The tangential arm must actually move ``xi`` (a guard against a no-op)."""
    sys_ = _periodic_rig(kernel="lean")
    sys_._rebuild(force=True)
    sys_._slow_pass(sys_.dt)
    sys_._prepare_fast()
    before = sys_._xi.copy()
    sys_._fast_step(sys_.dt / sys_.n_sub)
    assert np.abs(sys_._xi - before).max() > 0.0


def _wall_total_force(sys_: MTSDisks, wall_rate: str) -> tuple[np.ndarray, np.ndarray]:
    """Total contact force of one state under a given wall rate assignment."""
    h = sys_.dt / sys_.n_sub
    wxi0 = sys_._wxi.copy()
    sys_.config = dataclasses.replace(sys_.config, wall_rate=wall_rate)
    sys_._wxi = wxi0.copy()
    sys_._rebuild(force=True)
    f_slow, t_slow = sys_._slow_pass(sys_.dt)
    f_fast, t_fast = sys_._fast_pass(h)
    sys_._wxi = wxi0.copy()
    return f_slow + f_fast, t_slow + t_fast


def test_wall_normal_force_is_rate_independent():
    """Frictionless walls: the rate assignment is pure accounting (exact)."""
    sys_ = _walled_rig(kernel="reference", wall_rate="slow")
    sys_.config = dataclasses.replace(sys_.config, mu=0.0, mu_rolling=0.0)
    f_slow, t_slow = _wall_total_force(sys_, "slow")
    f_fast, t_fast = _wall_total_force(sys_, "fast")
    scale_f = float(np.abs(f_fast).max())
    assert np.abs(f_slow - f_fast).max() <= 1e-10 * max(scale_f, 1.0)
    assert np.abs(t_slow - t_fast).max() <= 1e-10 * max(
        float(np.abs(t_fast).max()), 1.0)


def test_wall_friction_coarsening_stays_bounded():
    """Frictional walls: the coarse arm is an O(dt) approximation, not an identity.

    The Cundall-Strack tangential state is integrated once per coarse step when
    the walls sit on the slow arm, and ``n_sub`` times when they sit on the fast
    arm; the cone return map makes the two differ at first order in ``dt``.  The
    test pins the *size* of that effect (a few percent of the contact force on a
    saturated contact) so a future change cannot silently make it large; the
    physics criterion (``phi_J``) is decided in the benchmark, which integrates
    the whole ladder rather than one instant.
    """
    sys_ = _walled_rig(kernel="reference", wall_rate="slow")
    f_slow, _ = _wall_total_force(sys_, "slow")
    f_fast, _ = _wall_total_force(sys_, "fast")
    scale = float(np.abs(f_fast).max())
    assert np.abs(f_slow - f_fast).max() <= 0.05 * scale


def test_wall_contacts_are_not_dropped_between_coarse_steps():
    """Regression: a stale wall selection silently removed wall contacts.

    A disk is driven at a wall for a whole coarse step; the wall force has to be
    non-zero at the end of the step (with the pre-P82W5 caching it was computed
    once and never renewed).
    """
    radii = np.array([0.010])
    cfg = MTSConfig(kn=KN, mu=0.5, mu_rolling=0.05, zeta=0.4)
    sys_ = MTSDisks(np.array([[0.05, 0.05]]), radii, cfg, box=(0.1, 0.1))
    sys_.setup_timestep(mode="mts")
    sys_.vel = np.array([[0.20, 0.0]])          # drive into the x = 0.1 plane
    for _ in range(20000):
        sys_.step(1)
        if sys_.pos[0, 0] > 0.1 - sys_.radii[0]:   # wall overlap reached
            break
    assert sys_.pos[0, 0] > 0.1 - sys_.radii[0], "the disk never reached the wall"
    assert sys_._wall_sets, "the wall selection must be refreshed every coarse step"
    assert 0 in sys_._wall_sets[0][4]
    fw, _tw = sys_._wall_forces(0.0)
    assert np.abs(fw).max() > 0.0
    assert sys_.work["wall"] > 1


# --------------------------------------------------------- acceptance gate
def test_tightness_negative_control():
    """``dt/dt_c = 0.5`` is inadmissible *and* visibly wrong when forced."""
    tc = hertz_contact_time(0.05, 1.0e-3, 1.0e4)
    gate = stability_gate(TIGHT_RATIO * tc, tc, margin=0.05)
    assert gate.status == "fail"
    with pytest.raises(InadmissibleTimeStep):
        gate.check()

    alpha = alpha_for_restitution(e_target=0.5, v_impact=0.05, mass=1.0e-3,
                                  stiffness=1.0e4)
    ref = impact_restitution(mass=1.0e-3, stiffness=1.0e4, v_impact=0.05,
                             alpha=alpha, dt_ratio=5.0e-4)
    forced = impact_restitution(mass=1.0e-3, stiffness=1.0e4, v_impact=0.05,
                                alpha=alpha, dt_ratio=TIGHT_RATIO, subcycle=False)
    assert abs(forced.e_measured / ref.e_measured - 1.0) > 0.10


@pytest.mark.parametrize("e_target", [0.5, 0.9])
def test_restitution_ladder_within_two_percent(e_target):
    """Criterion (a): every rung of ``dt/dt_c in {0.02, 0.05, 0.1, 0.2}``."""
    alpha = alpha_for_restitution(e_target=e_target, v_impact=0.05, mass=1.0e-3,
                                  stiffness=1.0e4)
    ref = impact_restitution(mass=1.0e-3, stiffness=1.0e4, v_impact=0.05,
                             alpha=alpha, dt_ratio=5.0e-4)
    for ratio in (0.02, 0.05, 0.1, 0.2):
        res = impact_restitution(mass=1.0e-3, stiffness=1.0e4, v_impact=0.05,
                                 alpha=alpha, dt_ratio=ratio)
        err = abs(res.e_measured / ref.e_measured - 1.0)
        assert err <= 0.02, f"e={e_target} dt/t_c={ratio}: error {err:.3%}"


def test_momentum_is_conserved_to_roundoff():
    """Criterion (b), momentum half: equal-and-opposite pairs only."""
    sys_ = _periodic_rig(kernel="lean", mu=0.0)
    p0 = sys_.momentum()
    norm0 = float(np.linalg.norm(p0))
    sys_.step(200)
    p1 = sys_.momentum()
    assert float(np.linalg.norm(p1 - p0)) / max(norm0, 1e-30) <= 1e-12


def test_rate_contract_work_counters():
    """The arms are paid the number of times the contract claims."""
    sys_ = _periodic_rig(kernel="lean")
    n_sub = sys_.n_sub
    assert n_sub >= 2, "the rig must need sub-cycling"
    before = dict(sys_.work)
    sys_.step(10)
    assert sys_.work["fast"] - before["fast"] == 10 * n_sub
    assert sys_.work["slow"] - before["slow"] == 10
    assert sys_.work["single"] == before["single"]


def test_walls_are_paid_once_per_coarse_step():
    """``wall_rate="slow"`` evaluates the wall springs once, not ``n_sub`` times."""
    sys_ = _walled_rig(kernel="lean", wall_rate="slow")
    n_sub = sys_.n_sub
    before = dict(sys_.work)
    sys_.step(5)
    assert sys_.work["wall"] - before["wall"] == 5, "walls must be a slow arm"

    legacy = _walled_rig(kernel="reference", wall_rate="fast")
    before = dict(legacy.work)
    legacy.step(5)
    assert legacy.work["wall"] - before["wall"] == 5 * legacy.n_sub


def test_config_rejects_unknown_rate_assignments():
    with pytest.raises(ValueError):
        MTSConfig(wall_rate="everywhere")
    with pytest.raises(ValueError):
        MTSConfig(fast_kernel="turbo")


def test_coarse_step_alone_would_be_inadmissible_for_the_stiff_skeleton():
    """The gate, not the taste of the operator, decides ``n_sub``."""
    sys_ = _periodic_rig(kernel="lean")
    scales = sys_.critical_scales()
    assert sys_.dt / scales["dt_c_stiff"] > 0.05      # too coarse for the skeleton
    assert sys_.dt / scales["dt_c_soft"] <= 0.1 + 1e-12  # fine for the majority
    assert sys_.n_sub >= math.ceil(sys_.dt / scales["dt_c_stiff"] / 0.1 - 1e-12)