"""Tests for the multi-rate contact integrator and the time-step gate (P82W2-N09).

The suite is deliberately cheap (a few seconds): it pins the *contract*, the
contact-law conventions, the momentum/energy bookkeeping and the agreement of
the two rate classes on a quasi-static compression.  The acceptance campaign
(1e5-step ledger, N=256 jamming, shard reproducibility) lives in
``benchmarks/mts_contact_integration.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.hertz_impact_time import hertz_contact_time
from tensordem.mts import (
    DEFAULT_MARGIN,
    TIGHT_RATIO,
    InadmissibleTimeStep,
    MTSConfig,
    MTSDisks,
    calibrate_alpha,
    cundall_gate,
    friction_gate,
    hertz_gate,
    hooke_gate,
    impact_restitution,
    stability_gate,
)


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------
def test_gate_status_and_substeps():
    gate = stability_gate(0.02, 1.0, margin=0.05)
    assert gate.status == "ok" and gate.substeps == 1
    gate = stability_gate(0.12, 1.0, margin=0.05)
    assert gate.status == "subcycle"
    # ceil(0.12 / 0.05) = 3 sub-steps, each inside the margin
    assert gate.substeps == 3
    assert gate.fine_step / gate.dt_c <= 0.05 + 1e-12
    assert math.isclose(gate.ratio, 0.12)


def test_gate_rejects_tight_ratio():
    gate = stability_gate(TIGHT_RATIO, 1.0, margin=DEFAULT_MARGIN)
    assert gate.status == "fail" and not gate.admissible
    with pytest.raises(InadmissibleTimeStep):
        gate.check()
    # the control path must still work when explicitly allowed
    gate.check(allow_inadmissible=True)
    # just below the tight ratio is admissible (but must be sub-cycled)
    soft = stability_gate(0.49, 1.0, margin=DEFAULT_MARGIN)
    assert soft.admissible and soft.status == "subcycle"


def test_gate_rejects_bad_arguments():
    with pytest.raises(ValueError):
        stability_gate(0.0, 1.0)
    with pytest.raises(ValueError):
        stability_gate(0.01, -1.0)
    with pytest.raises(ValueError):
        stability_gate(0.01, 1.0, margin=0.6)   # margin must be < tight_ratio
    with pytest.raises(ValueError):
        stability_gate(0.01, 1.0, kind="bogus")


def test_closed_form_gates_match_their_anchors():
    mass, stiffness, v = 1.0e-3, 1.0e4, 0.05
    gate = hertz_gate(1.0e-4, m_eff=mass, k_hertz=stiffness, v_impact=v)
    assert math.isclose(gate.dt_c, hertz_contact_time(v, mass, stiffness), rel_tol=1e-12)
    hooke = hooke_gate(1.0e-4, m_eff=mass, stiffness=stiffness)
    assert math.isclose(hooke.dt_c, math.pi * math.sqrt(mass / stiffness), rel_tol=1e-12)
    cundall = cundall_gate(1.0e-4, m_min=mass, k_max=stiffness)
    assert math.isclose(cundall.dt_c, 2.0 * math.sqrt(mass / stiffness), rel_tol=1e-12)
    fric = friction_gate(1.0e-4, m_eff=mass, k_t=0.5 * stiffness)
    assert math.isclose(fric.dt_c, math.pi * math.sqrt(mass / (0.5 * stiffness)),
                        rel_tol=1e-12)


# ---------------------------------------------------------------------------
# single-pair impact (acceptance a and d)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("e_target", [0.5, 0.9])
def test_restitution_ladder_inside_two_percent(e_target):
    cal = calibrate_alpha(e_target=e_target, v_impact=0.05, mass=1.0e-3,
                          stiffness=1.0e4, dt_ratio=5.0e-4)
    reference = impact_restitution(mass=1.0e-3, stiffness=1.0e4, v_impact=0.05,
                                   alpha=cal["alpha"], dt_ratio=5.0e-4)
    assert reference.restitution_error_vs(cal['e_measured']) < 1.0e-3
    for ratio in (0.02, 0.05, 0.1, 0.2):
        res = impact_restitution(mass=1.0e-3, stiffness=1.0e4, v_impact=0.05,
                                 alpha=cal["alpha"], dt_ratio=ratio)
        assert res.gate["status"] in ("ok", "subcycle")
        assert res.substeps >= 1
        err = res.restitution_error_vs(reference.e_measured)
        assert err <= 0.02, (ratio, err)


def test_tight_step_must_fail_loudly():
    cal = calibrate_alpha(e_target=0.5, v_impact=0.05, mass=1.0e-3,
                          stiffness=1.0e4, dt_ratio=5.0e-4)
    with pytest.raises(InadmissibleTimeStep):
        impact_restitution(mass=1.0e-3, stiffness=1.0e4, v_impact=0.05,
                           alpha=cal["alpha"], dt_ratio=TIGHT_RATIO)
    forced = impact_restitution(mass=1.0e-3, stiffness=1.0e4, v_impact=0.05,
                                alpha=cal["alpha"], dt_ratio=TIGHT_RATIO,
                                subcycle=False)
    reference = impact_restitution(mass=1.0e-3, stiffness=1.0e4, v_impact=0.05,
                                   alpha=cal["alpha"], dt_ratio=5.0e-4)
    error = forced.restitution_error_vs(reference.e_measured)
    assert error > 0.10, error
    # the failure is an energy-injecting one, not a silent damping
    assert forced.e_measured > reference.e_measured


def test_impact_rejects_impossible_input():
    with pytest.raises(ValueError):
        impact_restitution(mass=1.0e-3, stiffness=1.0e4, v_impact=0.05,
                           dt_ratio=-0.01, alpha=0.0)
    with pytest.raises(ValueError):
        impact_restitution(mass=1.0e-3, stiffness=1.0e4, v_impact=0.05,
                           dt_ratio=0.05)


# ---------------------------------------------------------------------------
# assembly: conventions, bookkeeping, rate agreement
# ---------------------------------------------------------------------------
def _pair(offset=0.019, *, stiff_ratio=1.0, stiff_fraction=0.0, mu=0.0):
    radii = np.array([0.010, 0.010])
    pos = np.array([[0.05, 0.05], [0.05 + offset, 0.05]])
    cfg = MTSConfig(kn=3.0e4, mu=mu, mu_rolling=0.0, zeta=0.0,
                    stiff_ratio=stiff_ratio, stiff_fraction=stiff_fraction)
    return MTSDisks(pos, radii, cfg, box=(0.2, 0.2)), cfg


def test_normal_force_sign_is_repulsive():
    """Two overlapping disks must be pushed apart (convention check)."""
    sys_, _ = _pair(offset=0.019)
    sys_.setup_timestep(mode="single")
    sys_.step(1)
    assert sys_.vel[0, 0] < 0.0 < sys_.vel[1, 0]


def test_stiffness_hierarchy_scales_the_pair_stiffness():
    sys_, cfg = _pair(offset=0.019, stiff_ratio=100.0, stiff_fraction=0.5)
    i, j = sys_._cache
    assert np.allclose(sys_._pair_stiffness(i, j), 100.0 * cfg.kn)
    assert np.all(sys_._pair_stiff)


def test_free_flight_conserves_momentum_and_energy():
    radii = np.full(4, 0.01)
    pos = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    cfg = MTSConfig(kn=3.0e4, mu=0.0, mu_rolling=0.0, zeta=0.0,
                    stiff_ratio=100.0, stiff_fraction=0.25)
    sys_ = MTSDisks(pos, radii, cfg, box=(0.3, 0.3), periodic=True)
    sys_.vel = np.array([[0.01, 0.0], [-0.01, 0.0], [0.0, 0.02], [0.0, -0.02]])
    sys_.setup_timestep(mode="mts")
    e0, p0 = sys_.energy(), sys_.momentum()
    sys_.step(200)
    assert np.allclose(sys_.momentum(), p0, atol=1e-18)
    assert abs(sys_.energy() - e0) / e0 < 1e-12


def test_multirate_matches_single_rate_on_a_quasistatic_compression():
    """The whole point: the coarse multi-rate step must not move the physics."""
    rng = np.random.default_rng(7)
    n = 48
    radii = np.full(n, 0.01)
    side = math.sqrt(n * math.pi * 0.01 ** 2 / 0.55)
    grid = int(math.ceil(math.sqrt(n)))
    spacing = side / grid
    pts = [[(c + 0.5) * spacing, (row + 0.5) * spacing]
           for row in range(grid) for c in range(grid)][:n]
    pos = np.array(pts) + rng.normal(0.0, 0.02 * spacing, size=(n, 2))
    cfg = MTSConfig(kn=3.0e4, mu=0.5, mu_rolling=0.05, zeta=0.4,
                    damping_time=1.0e-3, stiff_ratio=25.0, stiff_fraction=0.10)
    results = {}
    for mode in ("single", "mts"):
        sys_ = MTSDisks(pos.copy(), radii, cfg, box=(side, side))
        sys_.setup_timestep(mode=mode)
        n_steps = max(1, int(round(0.030 / sys_.dt)))
        sys_.servo_pressure(1.0e3, steps=n_steps, eta=150.0,
                            every=max(1, int(round(20 * 8.8e-6 / sys_.dt))))
        results[mode] = (sys_.area_fraction(), sys_.pressure(), sys_.coordination())
    phi_single, p_single, z_single = results["single"]
    phi_mts, p_mts, z_mts = results["mts"]
    phi_initial = n * math.pi * 0.01 ** 2 / side ** 2
    assert phi_single > 1.1 * phi_initial, (phi_initial, phi_single)
    # 2.5 %: the deeply overlapped end of a violent compression is the
    # worst case for the frozen-state coarse step (see the report's ablation)
    assert abs(phi_mts / phi_single - 1.0) < 0.025, (phi_single, phi_mts)
    assert abs(z_mts - z_single) < 0.25
    assert p_single > 0.0 and p_mts > 0.0


def test_split_slow_contacts_ablation_is_available_and_documented():
    """The frozen-soft-force ablation exists (and is off by default)."""
    cfg = MTSConfig()
    assert cfg.split_slow_contacts is False


def test_step_requires_a_timestep():
    radii = np.array([0.01])
    sys_ = MTSDisks(np.zeros((1, 2)), radii, MTSConfig(), box=(0.1, 0.1))
    with pytest.raises(RuntimeError):
        sys_.step(1)


def test_servo_pressure_reaches_its_target_and_is_deterministic():
    rng = np.random.default_rng(3)
    n = 24
    radii = np.full(n, 0.01)
    side = math.sqrt(n * math.pi * 0.01 ** 2 / 0.45)
    pos = 0.01 + rng.random((n, 2)) * (side - 0.02)
    cfg = MTSConfig(kn=3.0e4, mu=0.5, mu_rolling=0.05, zeta=0.4, damping_time=1.0e-3)

    def run():
        sys_ = MTSDisks(pos.copy(), radii, cfg, box=(side, side))
        sys_.setup_timestep(mode="single")
        out = sys_.servo_pressure(500.0, steps=400, eta=150.0, every=10)
        return out, sys_.pos.copy()

    first, pos_a = run()
    second, pos_b = run()
    assert math.isclose(first["phi"], second["phi"], rel_tol=0, abs_tol=0)
    assert np.array_equal(pos_a, pos_b)
    assert first["pressure"] > 100.0


def test_stiff_skeleton_substeps_are_required_by_the_gate():
    radii = np.full(16, 0.01)
    rng = np.random.default_rng(5)
    pos = rng.random((16, 2)) * 0.2
    cfg = MTSConfig(kn=3.0e4, mu=0.5, stiff_ratio=100.0, stiff_fraction=0.25)
    sys_ = MTSDisks(pos, radii, cfg, box=(0.25, 0.25))
    gate = sys_.setup_timestep(mode="mts")
    assert gate["n_sub"] >= 2
    assert gate["fine_gate"]["status"] == "ok"
    assert gate["outer_gate"]["status"] == "ok"
    assert gate["friction_gate"]["status"] == "ok"


def test_single_rate_step_is_the_stiff_limit():
    radii = np.full(16, 0.01)
    rng = np.random.default_rng(5)
    pos = rng.random((16, 2)) * 0.2
    cfg = MTSConfig(kn=3.0e4, mu=0.5, stiff_ratio=100.0, stiff_fraction=0.25)
    sys_ = MTSDisks(pos, radii, cfg, box=(0.25, 0.25))
    single = sys_.setup_timestep(mode="single")
    multirate = sys_.setup_timestep(mode="mts")
    assert multirate["dt_outer"] > 5.0 * single["dt_outer"]
    assert math.isclose(multirate["dt_outer"] / multirate["n_sub"],
                        single["dt_outer"], rel_tol=1e-9)