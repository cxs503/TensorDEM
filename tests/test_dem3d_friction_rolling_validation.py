"""Verification tests for the DEM3D Coulomb-friction and rolling-resistance branches.

These are *quantitative* checks against closed form, not smoke tests: the first
version of the friction suite only asserted that ``sim.step()`` ran, which is
why four defects in the branch went unnoticed (unclamped persistent slip,
missing re-projection of the stored slip onto the rotating contact plane,
friction memory surviving separation, and a rolling-resistance couple that was
orthogonal to the motion and therefore did no work). See
``docs/dem3d_friction_rolling_integration_validation.md``.
"""
import math

import pytest

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.friction_rolling_verification import (
    ContactPatch,
    PatchConfig,
    _stick_limit,
    scenario_normal_rotation,
    scenario_reversal,
    scenario_rolling,
    scenario_separation,
    scenario_sliding,
    scenario_sliding_work,
    scenario_stick,
    scenario_work_convergence,
)

CFG = PatchConfig()


# ---------------------------------------------------------------------------
# Elastic (sticking) regime
# ---------------------------------------------------------------------------
def test_stick_regime_matches_spring_law():
    result = scenario_stick(CFG, mu=0.3, speed=0.01, steps=20)
    assert result["stick_ratio"] < 1.0, "scenario must stay below the friction limit"
    assert abs(result["rel_error"]) < 1e-9
    assert abs(result["stored_slip_rel_error"]) < 1e-9


def test_no_friction_reproduces_pure_normal_contact():
    """Negative control: with mu = 0 the tangential branch must be inert."""
    patch = ContactPatch(CFG, mu=0.0)
    patch.set_velocity(tangent_axis=1, speed=1.0)
    force, _ = patch.sim.forces()
    comp = force[1].tolist()
    assert comp[1] == pytest.approx(0.0, abs=1e-12)
    assert comp[0] == pytest.approx(+CFG.normal_force, rel=1e-12)
    assert patch.n_states() == 0


# ---------------------------------------------------------------------------
# Saturated (sliding) regime
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mu", [0.05, 0.1, 0.2, 0.3])
def test_sliding_saturates_at_coulomb_limit(mu):
    result = scenario_sliding(CFG, mus=(mu,))
    row = result["rows"][0]
    assert abs(row["f_tangent_rel_error"]) < 1e-9, row
    # The persisted slip must be the clamped one, not the driven slip.
    assert row["stored_slip_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert row["unclamped_slip_m"] > 5 * row["stored_slip_measured_m"]


def test_sliding_force_scales_with_mu():
    rows = scenario_sliding(CFG, mus=(0.05, 0.1, 0.2, 0.3))["rows"]
    magnitudes = [-r["f_tangent_measured_N"] for r in rows]
    for mu, mag in zip((0.05, 0.1, 0.2, 0.3), magnitudes):
        assert mag == pytest.approx(mu * CFG.normal_force, rel=1e-9)
    assert all(b > a for a, b in zip(magnitudes, magnitudes[1:]))


def test_sliding_work_matches_ramp_plus_dissipation():
    result = scenario_sliding_work(CFG, mu=0.2)
    assert abs(result["rel_error"]) < 0.01
    assert abs(result["dissipation_rel_error"]) < 0.01


def test_sliding_work_converges_with_dt():
    result = scenario_work_convergence()
    assert result["monotone_decreasing"] is True
    assert abs(result["rows"][-1]["rel_error"]) < 2e-3
    # First order: quartering dt must cut the residual by roughly four.
    assert result["error_ratio_first_to_last"] > 3.0


# ---------------------------------------------------------------------------
# Reversal: friction memory must be bounded by the friction limit
# ---------------------------------------------------------------------------
def test_reversal_slip_is_bounded_by_stick_limit():
    result = scenario_reversal(CFG, mu=0.1)
    quantum = result["a_stick_m"] / 5.0
    for row in result["rows"]:
        slip = row["reversal_slip_m"]
        assert math.isfinite(slip), row
        # a_stick of reversed slip, plus at most one integration step.
        assert slip <= result["a_stick_m"] + 1.5 * quantum, row


def test_reversal_slip_is_independent_of_sliding_history():
    result = scenario_reversal(CFG, mu=0.1, slip_durations=(0.05, 0.1, 0.2))
    assert result["history_independent"] is True, result
    assert result["reversal_slip_spread_m"] == pytest.approx(0.0, abs=1e-12)
    # The unclamped variant would scale with the driven slip (0.05 -> 0.2 s).
    driven = [row["sliding_force_evaluations"] * row["slip_quantum_m"]
              for row in result["rows"]]
    assert max(driven) > 10 * result["a_stick_m"]


# ---------------------------------------------------------------------------
# Contact normal rotation / separation
# ---------------------------------------------------------------------------
def test_rotated_normal_has_no_friction_component():
    result = scenario_normal_rotation(CFG, mu=0.2)
    assert abs(result["spurious_normal_component_N"]) < 1e-9, result
    assert abs(result["rel_error"]) < 1e-9


def test_separation_clears_friction_memory():
    result = scenario_separation(CFG, mu=0.2)
    assert result["stored_slip_before_m"] > 0.0
    assert result["states_retained_after_separation"] == 0
    assert result["recontact_tangential_force_N"] == pytest.approx(0.0, abs=1e-12)


def test_friction_memory_retained_while_contact_persists():
    """Guard against over-pruning: an ongoing contact must keep its slip."""
    patch = ContactPatch(CFG, mu=0.2)          # a_stick = 1 mm at mu=0.2
    first = patch.drive(speed=0.01, steps=20)[-1]["stored_slip"]
    second = patch.drive(speed=0.01, steps=20)[-1]["stored_slip"]
    assert patch.n_states() == 1
    assert first == pytest.approx(0.01 * CFG.dt * 20, rel=1e-9)
    assert second == pytest.approx(2 * first, rel=1e-9)


# ---------------------------------------------------------------------------
# Rolling resistance
# ---------------------------------------------------------------------------
def test_rolling_resistance_opposes_motion_and_dissipates():
    result = scenario_rolling(CFG, mu_r=0.05)
    assert abs(result["rel_error"]) < 1e-9, result
    assert result["f_tangential_measured_N"] < 0.0, "force must oppose the motion"
    assert abs(result["f_lateral_measured_N"]) < 1e-12, "no lateral force"
    assert result["dissipation_rate_W"] < 0.0
    assert result["dissipation_rate_W"] == pytest.approx(
        result["dissipation_rate_reference_W"], rel=1e-9)


def test_rolling_zero_control():
    result = scenario_rolling(CFG, mu_r=0.05)
    assert result["control_mu_r0_f_tangential_N"] == pytest.approx(0.0, abs=1e-12)


def test_rolling_and_coulomb_add_up():
    """Both branches act tangentially, so their limit forces add."""
    mu, mu_r = 0.1, 0.05
    patch = ContactPatch(CFG, mu=mu, mu_r=mu_r)
    patch.set_velocity(tangent_axis=1, speed=1.0)
    with_rolling = float(patch.sim.forces()[0][1].tolist()[1])
    patch.drive(speed=1.0, steps=20)          # saturate Coulomb friction
    patch.set_velocity(tangent_axis=1, speed=1.0)
    saturated = float(patch.sim.forces()[0][1].tolist()[1])
    expected = -(mu + mu_r) * CFG.normal_force
    assert saturated == pytest.approx(expected, rel=1e-9)
    assert saturated < with_rolling


# ---------------------------------------------------------------------------
# Bonded contacts keep their own branch
# ---------------------------------------------------------------------------
def test_bonded_pair_does_not_receive_contact_friction():
    """A live bond suppresses the contact branch, so no friction acts."""
    patch = ContactPatch(CFG, mu=0.5)
    sim = patch.sim
    bond = (sim.pairs[:, 0] == 0) & (sim.pairs[:, 1] == 1)
    assert int(bond.sum()) == 1
    sim.alive[:] = False
    sim.alive[bond] = True                     # only this pair is bonded
    patch.set_velocity(tangent_axis=1, speed=1.0)
    # update_fracture=False so the bond topology is the one just prescribed
    force_bonded = float(sim.forces(update_fracture=False)[0][1].tolist()[1])
    sim.alive[:] = False                       # bond broken -> contact branch
    force_broken = float(sim.forces(update_fracture=False)[0][1].tolist()[1])
    assert abs(force_bonded) < 1e-12, "bonded contact must not carry friction"
    # One step of relative slip, so the spring force is k_t * |v| * dt.
    assert force_broken == pytest.approx(-CFG.contact_stiffness * CFG.dt, rel=1e-9)
    # (tangential sign: the friction force on particle 1 opposes its +y motion)
