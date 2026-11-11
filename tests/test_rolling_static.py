"""Unit + engine-level invariants for the static / shape-aware rolling law.

Covers ``src/tensordem/contact/rolling_static.py`` (GAP-DEM-CLUMP round 2):

* R1 shape-aware moment arm: ``com`` lever == ``member`` lever bit for bit on a
  single sphere, and anisotropic (tip >> flank) on an elongated clump;
* R2 static branch: full Coulomb limit above the one-step stopping speed and
  exact arrest of the relative spin below it;
* R3 sign / bound invariants and bit-exact reproduction of the legacy viscous
  law (so round-1 numbers stay reproducible);
* R4 engine-level A/B: a sphere given an initial spin on a floor stops in one
  step with the static branch and lingers with the legacy viscous branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.contact import rolling_static as rs  # noqa: E402
from tensordem.contact.clump_contact import ContactParams  # noqa: E402
from tensordem.particle.clump import ellipsoid_clump, single_sphere  # noqa: E402
from tensordem.particle.clump_dem import ClumpDEM  # noqa: E402

DENSITY = 2650.0


# ---------------------------------------------------------------------------
# R1 — shape-aware moment arm
# ---------------------------------------------------------------------------
def test_com_lever_equals_member_lever_on_a_single_sphere():
    r = 1.2e-3
    com = np.zeros((1, 3))
    rng = np.random.default_rng(0)
    u = rng.normal(size=(64, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    pts = r * u
    arm_com = rs.rolling_lever_arm(pts, com, np.full(64, r), "com")
    arm_mem = rs.rolling_lever_arm(pts, com, np.full(64, r), "member")
    assert np.allclose(arm_com, arm_mem, rtol=0.0, atol=1e-15)
    # ... and both equal the sphere radius
    assert np.allclose(arm_mem, r, atol=1e-15)


def test_com_lever_is_anisotropic_for_an_elongated_clump():
    p = ellipsoid_clump(3.0 * 1.2e-3, 1.2e-3, n_spheres=7, density=DENSITY)
    assert p.n_members >= 3
    local = p.local_centers  # body frame, about the union centre of mass
    radii = p.radii
    tip = np.argmax(np.abs(local[:, 0]))
    flank = np.argmin(np.abs(local[:, 0]) + 10.0 * np.abs(local[:, 1]))
    pts_tip = np.asarray(local[tip] + radii[tip] * np.array([1.0, 0.0, 0.0])).reshape(1, 3)
    pts_flank = np.asarray(local[flank] + radii[flank] * np.array([0.0, 1.0, 0.0])).reshape(1, 3)
    com = np.zeros((1, 3))
    tip_arm = rs.rolling_lever_arm(pts_tip, com, np.array([radii[tip]]), "com")[0]
    flank_arm = rs.rolling_lever_arm(pts_flank, com, np.array([radii[flank]]), "com")[0]
    assert tip_arm > 2.0 * flank_arm, (tip_arm, flank_arm)
    # the legacy lever is blind to the position: a/b == 3 must give the *same* arm
    tip_mem = rs.rolling_lever_arm(pts_tip, com, np.array([radii[tip]]), "member")[0]
    flank_mem = rs.rolling_lever_arm(pts_flank, com, np.array([radii[flank]]), "member")[0]
    assert tip_mem == pytest.approx(radii[tip]) and flank_mem == pytest.approx(radii[flank])


def test_bad_model_and_lever_are_rejected():
    with pytest.raises(ValueError):
        rs.check_lever("bounding")
    with pytest.raises(ValueError):
        rs.check_model("jiang")


# ---------------------------------------------------------------------------
# R2 — static (coulomb) branch
# ---------------------------------------------------------------------------
def test_full_coulomb_limit_is_delivered_above_the_stopping_speed():
    dt = 5e-5
    lim = np.array([4.0e-5])
    inert = np.array([6.7e-9])
    w_eps = rs.stopping_speed(lim, inert, dt)[0]
    w = np.array([[100.0 * w_eps, 0.0, 0.0]])
    tau = rs.rolling_torque(w, lim, inert, dt, model="coulomb")
    assert np.linalg.norm(tau) == pytest.approx(lim[0], rel=1e-12)
    assert rs.rolling_saturation(w, lim, inert, dt, model="coulomb")[0] == pytest.approx(1.0)


def test_relative_spin_is_arrested_in_one_step_below_the_limit():
    dt = 5e-5
    lim = np.array([4.0e-5])
    inert = np.array([6.7e-9])
    w_eps = rs.stopping_speed(lim, inert, dt)[0]
    w = np.array([[0.0, 0.1 * w_eps, 0.0]])
    tau = rs.rolling_torque(w, lim, inert, dt, model="coulomb")
    omega_after = w + tau / inert[:, None] * dt
    assert np.allclose(omega_after, 0.0, atol=1e-18), omega_after
    assert rs.rolling_saturation(w, lim, inert, dt, model="coulomb")[0] < 0.11


def test_stopping_speed_shrinks_with_a_smaller_moment_of_inertia():
    lim = np.array([4.0e-5])
    a = rs.stopping_speed(lim, np.array([6.7e-9]), 5e-5)[0]
    b = rs.stopping_speed(lim, np.array([6.7e-11]), 5e-5)[0]
    assert b == pytest.approx(100.0 * a)


# ---------------------------------------------------------------------------
# R3 — sign / bound invariants, legacy equivalence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["viscous", "coulomb"])
def test_torque_opposes_the_relative_spin_and_respects_the_limit(model):
    rng = np.random.default_rng(1)
    w = rng.normal(size=(200, 3)) * 2.0
    lim = rng.uniform(1e-6, 1e-4, size=200)
    inert = rng.uniform(1e-10, 1e-8, size=200)
    tau = rs.rolling_torque(w, lim, inert, 5e-5, model=model)
    dot = np.sum(tau * w, axis=1)
    assert np.all(dot <= 0.0)                     # strictly dissipative
    assert np.all(np.linalg.norm(tau, axis=1) <= lim * (1.0 + 1e-12) + 1e-18)


def test_viscous_model_reproduces_the_legacy_law_bit_for_bit():
    rng = np.random.default_rng(2)
    w = rng.normal(size=(50, 3))
    lim = rng.uniform(1e-6, 1e-4, size=50)
    eps = 1.0
    legacy = (-lim / (np.linalg.norm(w, axis=1) + eps))[:, None] * w
    new = rs.rolling_torque(w, lim, np.ones(50), 5e-5, model="viscous", eps_w=eps)
    assert np.array_equal(legacy, new)


def test_zero_spin_gives_zero_torque_in_both_models():
    w = np.zeros((3, 3))
    for model in rs.ROLLING_MODELS:
        tau = rs.rolling_torque(w, np.ones(3), np.ones(3), 5e-5, model=model)
        assert np.all(tau == 0.0)


# ---------------------------------------------------------------------------
# R4 — engine-level A/B: spin arrest on a floor
# ---------------------------------------------------------------------------
def _spin_down(model: str, *, steps: int = 20, omega0: float = 0.5) -> dict:
    r, dens = 1.2e-3, DENSITY
    p = single_sphere(r, dens)
    sim = ClumpDEM(
        [p], [np.array([0.0, 0.0, 1.0 * r])],
        params=ContactParams(kn=150.0, mu=0.5, mu_roll=0.3,
                             rolling_model=model, rolling_lever="com"),
        dt=5e-5,
        obstacles=[__import__("tensordem.contact.clump_contact", fromlist=["BoxObstacle"]).BoxObstacle(
            center=np.array([0.0, 0.0, -0.01]), half=np.array([0.02, 0.02, 0.01]))],
    )
    for _ in range(200):                       # settle onto the floor
        sim.step()
    # spin about the *vertical* axis: the contact point does not slide, so the
    # test isolates the rolling branch (no sliding-friction contamination).
    sim.omega[0] = np.array([0.0, 0.0, omega0])
    w = []
    for _ in range(steps):
        sim.step()
        w.append(float(np.linalg.norm(sim.omega[0])))
    return {"omega_trace": w, "omega_final": w[-1], "omega0": omega0}


def test_static_branch_arrests_a_spinning_sphere_within_two_steps():
    stat = _spin_down("coulomb")
    visc = _spin_down("viscous")
    tr = stat["omega_trace"]
    # one step of plastic rolling (Coulomb limit), then full stick
    assert stat["omega0"] - tr[0] > 0.0
    assert tr[1] < 1e-15 * stat["omega0"] and stat["omega_final"] < 1e-15, tr[:3]
    # the legacy branch is rate proportional: exponential tail, never arrested
    assert visc["omega_final"] > 1e-4, visc["omega_final"]
    ratio = visc["omega_trace"][5] / visc["omega_trace"][4]
    assert 0.6 < ratio < 0.9, ratio
    assert visc["omega_final"] > 10.0 * max(stat["omega_final"], 1e-300)


def test_legacy_law_has_no_static_branch_and_creeps_on_a_slope():
    """Fixed-drive creep: the old law settles at a non-zero spin, the new one at 0.

    A body held on a slope of angle ``theta`` feels a constant driving torque
    ``m g sin(theta) R`` against the rolling resistance.  With ``M_drive < M_max``
    the physical answer is "no motion".  Integrating ``I dw/dt = M_drive - M_roll(w)``
    with each law exposes the defect: ``viscous`` converges to a non-zero creep
    spin, ``coulomb`` sticks.
    """
    dt, inert, lever = 5e-5, 1.105e-11, 1.2e-3
    m, g, mu_roll, theta = 1.918e-5, 9.81, 0.3, np.deg2rad(12.0)
    drive = m * g * np.sin(theta) * lever           # < mu_roll m g cos(theta) lever
    lim = np.array([mu_roll * lever * m * g * np.cos(theta)])
    assert drive < lim[0]

    out = {}
    for model in ("viscous", "coulomb"):
        w = 0.0
        for _ in range(4000):
            tau = float(rs.rolling_torque(np.array([[w, 0.0, 0.0]]), lim,
                                          np.array([inert]), dt, model=model,
                                          eps_w=1.0)[0, 0])
            w = w + (drive + tau) / inert * dt
            w = max(w, 0.0)
        out[model] = w
    # The stateless static branch leaves exactly one step of drive unopposed:
    # w_res = M_drive dt / I (resolution limit, not a model limit -- an
    # elastic-plastic rolling spring with history would arrest exactly).
    w_res = drive * dt / inert
    assert out["coulomb"] <= 1.05 * w_res, (out["coulomb"], w_res)
    # analytic steady state of the rate-proportional branch: drive = M_max w/(w+eps)
    w_ss = drive * 1.0 / (lim[0] - drive)
    assert out["viscous"] == pytest.approx(w_ss, rel=0.05), (out["viscous"], w_ss)
    # ... and it is an order of magnitude above the static residual
    assert out["viscous"] > 10.0 * out["coulomb"], out


def test_engine_energy_ledger_still_closes_with_the_static_branch():
    r = 1.2e-3
    from tensordem.contact.clump_contact import BoxObstacle

    p = single_sphere(r, DENSITY)
    sim = ClumpDEM(
        [p], [np.array([0.0, 0.0, 1.05 * r])],
        params=ContactParams(kn=150.0, mu=0.5, mu_roll=0.3,
                             rolling_model="coulomb", rolling_lever="com"),
        dt=2e-5,
        obstacles=[BoxObstacle(center=np.array([0.0, 0.0, -0.01]),
                               half=np.array([0.02, 0.02, 0.01]))],
    )
    sim.omega[0] = np.array([0.3, 0.0, 0.0])
    e0 = sim.total_energy() if hasattr(sim, "total_energy") else None
    for _ in range(400):
        sim.step()
    e1 = sim.total_energy() if hasattr(sim, "total_energy") else None
    if e0 is not None:
        # dissipative law: energy must not be created
        assert e1 <= e0 * (1.0 + 1e-9) + 1e-18, (e0, e1)
    assert np.isfinite(float(np.linalg.norm(sim.omega[0])))