import copy
import json
import math
import pytest
import torch
from tensordem import DEMConfig, IceDEM


def test_restart_bitwise_and_fail_closed():
    sim = IceDEM(DEMConfig(nx=3, ny=2, drag=0, fix_edges=False, tool_gap=1))
    load = torch.ones_like(sim.positions)
    for _ in range(12):
        sim.step(load, load_time=sim.time)
    saved = json.loads(json.dumps(sim.snapshot(), allow_nan=False))
    restarted = IceDEM.from_snapshot(saved)
    for _ in range(8):
        sim.step(load)
        restarted.step(load)
    assert sim.snapshot() == restarted.snapshot()
    for key, value in (("schema", "unknown"), ("positions_m", [[0., 0.]]), ("time_s", float('nan'))):
        broken = copy.deepcopy(saved)
        broken[key] = value
        with pytest.raises(ValueError):
            IceDEM.from_snapshot(broken)
    with pytest.raises(ValueError):
        sim.step(load, load_time=0)


def test_load_work_moment_and_momentum_impulse():
    sim = IceDEM(DEMConfig(nx=3, ny=2, drag=0, fix_edges=False, tool_gap=1))
    old = sim.positions.clone()
    loads = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    sim.step(loads)
    assert sim.last_step['external_work_J'] == pytest.approx(float((loads * (sim.positions-old)).sum()))
    assert sim.last_step['external_moment_Nm'] == pytest.approx(float((old[:,0]*loads[:,1]-old[:,1]*loads[:,0]).sum()))
    torch.testing.assert_close(sim.config.mass * sim.velocities.sum(0), sim.dt * loads.sum(0))
    assert sum(f['mass_kg'] for f in sim.fragments()) == pytest.approx(6 * sim.config.mass)


def test_support_impulse_and_release_not_double_counted():
    sim = IceDEM(DEMConfig(nx=3, ny=2, drag=0))
    loads = torch.ones_like(sim.positions)
    sim.step(loads)
    torch.testing.assert_close(torch.tensor(sim.last_step['support_force_N'], dtype=torch.float64), -loads[sim.fixed].sum(0))
    sim.positions[:,0] *= 1.2
    sim.forces()
    release = sim.accounting['fracture_release_J']
    assert release > 0
    sim.forces()
    sim.diagnostics()
    assert sim.accounting['fracture_release_J'] == release


def test_timestep_refinement_uniform_translation():
    cfg = DEMConfig(nx=3, ny=2, drag=0, fix_edges=False, tool_gap=1)
    duration = 20 * cfg.recommended_dt
    errors = []
    for divisor in (1, 2, 4):
        sim = IceDEM(DEMConfig(nx=3, ny=2, drag=0, fix_edges=False, tool_gap=1, dt=cfg.recommended_dt/divisor))
        loads = torch.zeros_like(sim.positions)
        loads[:,0] = sim.config.mass
        initial = sim.positions.clone()
        for _ in range(20 * divisor):
            sim.step(loads)
        exact = initial[:,0] + .5 * duration**2
        errors.append(float((sim.positions[:,0]-exact).abs().max()))
        torch.testing.assert_close(sim.velocities[:,0], torch.full_like(sim.velocities[:,0],duration), atol=1e-13, rtol=1e-12)
    assert errors[1] < .51 * errors[0]
    assert errors[2] < .51 * errors[1]


# ---------------------------------------------------------------------------
# Restart boundary matrix: every leaf of the restart record is compared
# exactly (no tolerance) at the interruption point and after one further step,
# so a single disagreeing field turns the test red.
# ---------------------------------------------------------------------------

BOUNDARY_STEPS = (0, 1, 2, 5, 11, 13)


def _state_diffs(reference, actual, path="state"):
    """Collect every leaf that differs between two restart records."""
    out = []
    if type(reference) is not type(actual):
        out.append(f"{path}: type {type(reference).__name__} != {type(actual).__name__}")
    elif isinstance(reference, dict):
        if set(reference) != set(actual):
            out.append(f"{path}: field set differs by {sorted(set(reference) ^ set(actual))}")
        for key in sorted(set(reference) & set(actual)):
            out.extend(_state_diffs(reference[key], actual[key], f"{path}.{key}"))
    elif isinstance(reference, list):
        if len(reference) != len(actual):
            out.append(f"{path}: length {len(reference)} != {len(actual)}")
        for index, (left, right) in enumerate(zip(reference, actual)):
            out.extend(_state_diffs(left, right, f"{path}[{index}]"))
    elif reference != actual:
        out.append(f"{path}: {reference!r} != {actual!r}")
    return out


def _assert_same_state(reference, actual, context):
    """Exact, field-by-field restart equality with a pinpointable failure."""
    diffs = _state_diffs(reference, actual)
    assert not diffs, f"{context}: {len(diffs)} field(s) disagree; " + "; ".join(diffs[:4])


def _loads(sim, step):
    """Step-dependent nodal loads so the branch is never load-history-free."""
    pattern = torch.linspace(-1.0, 1.0, sim.positions.numel(), dtype=torch.float64)
    return (0.5 + 0.1 * step) * pattern.reshape(sim.positions.shape)


def _roundtrip(snapshot):
    return json.loads(json.dumps(snapshot, allow_nan=False))


def _advance(sim, steps, *, stamp_load_time=False):
    for step in range(steps):
        sim.step(_loads(sim, step), load_time=sim.time if stamp_load_time else None)


@pytest.mark.parametrize("interrupt_step", BOUNDARY_STEPS)
@pytest.mark.parametrize("fix_edges", (False, True))
def test_restart_boundary_fieldwise_before_and_after_step(interrupt_step, fix_edges):
    sim = IceDEM(DEMConfig(nx=3, ny=2, drag=0, fix_edges=fix_edges, tool_gap=1, tool_speed=0.3))
    _advance(sim, interrupt_step, stamp_load_time=True)
    committed = sim.snapshot()
    # JSON round-trip of the checkpoint must not move a single leaf.
    _assert_same_state(committed, _roundtrip(committed), f"json round-trip at step {interrupt_step}")
    restarted = IceDEM.from_snapshot(_roundtrip(committed))
    # Interruption point: state immediately before the next step.
    _assert_same_state(committed, restarted.snapshot(), f"interruption state at step {interrupt_step}")
    assert restarted.time == sim.time and restarted.step_count == interrupt_step
    # One step either side of the interruption clock is rejected.
    for offset in (-1, 1):
        with pytest.raises(ValueError):
            restarted.step(_loads(restarted, interrupt_step), load_time=restarted.time + offset * restarted.dt)
    # One step after the interruption: both branches must stay identical.
    sim.step(_loads(sim, interrupt_step), load_time=sim.time)
    restarted.step(_loads(restarted, interrupt_step), load_time=restarted.time)
    _assert_same_state(sim.snapshot(), restarted.snapshot(), f"first step after interruption {interrupt_step}")
    # The restored branch keeps stepping bitwise after the boundary.
    for step in range(interrupt_step, interrupt_step + 3):
        sim.step(_loads(sim, step))
        restarted.step(_loads(restarted, step))
        _assert_same_state(sim.snapshot(), restarted.snapshot(), f"trajectory after interruption {step}")


def test_restart_at_first_failure_boundary_preserves_ledger():
    """Interrupt exactly at the step that writes the first failed bond."""
    sim = IceDEM(DEMConfig(nx=3, ny=2, drag=0, tool_gap=1))
    loads = torch.zeros_like(sim.positions)
    loads[:, 0] = 20.0
    for _ in range(100):
        sim.step(loads)
        if bool((sim.failure_mode > 0).any()):
            break
    assert bool((sim.failure_mode > 0).any()), "scenario must interrupt at a fracture event"
    assert bool((sim.failure_mode == 0).any()), "scenario must keep intact bonds alongside failed ones"
    assert sim.accounting["fracture_release_J"] > 0
    saved = _roundtrip(sim.snapshot())
    restarted = IceDEM.from_snapshot(saved)
    _assert_same_state(saved, restarted.snapshot(), "failure-boundary interruption state")
    # The failure ledger must survive exactly, including per-bond arrays.
    for key in ("failure_mode", "failure_time_s", "failure_extension_m",
                "failure_energy_J", "alive"):
        assert restarted.snapshot()[key] == saved[key], key
    # One step after: release energy is neither lost nor double counted.
    sim.step(loads)
    restarted.step(loads)
    _assert_same_state(sim.snapshot(), restarted.snapshot(), "first step after failure boundary")
    assert restarted.accounting["fracture_release_J"] == sim.accounting["fracture_release_J"]
    # A checkpoint shifted by one step must fail closed, not silently rewind.
    shifted = copy.deepcopy(saved)
    shifted["step_count"] += 1
    with pytest.raises(ValueError):
        IceDEM.from_snapshot(shifted)


def test_restart_boundary_with_hull_contact_active():
    profile = torch.tensor([[0.0, -0.1], [0.0, 0.2]], dtype=torch.float64)
    sim = IceDEM(DEMConfig(nx=5, ny=3, tool_speed=0.2, fix_edges=False, drag=0.0),
                 hull_profile=profile, hull_start=(0.01, 0.0), hull_velocity=(0.2, 0.0))
    loads = torch.zeros_like(sim.positions)
    loads[:, 0] = 1.0
    for _ in range(6):
        sim.step(loads)
    assert float(sim.last_step["tool_reaction_N"][0]) > 0.0, "hull contact must be active"
    saved = _roundtrip(sim.snapshot())
    restarted = IceDEM.from_snapshot(saved)
    _assert_same_state(saved, restarted.snapshot(), "hull-contact interruption state")
    assert restarted.snapshot()["hull_profile_m"] == profile.tolist()
    sim.step(loads)
    restarted.step(loads)
    _assert_same_state(sim.snapshot(), restarted.snapshot(), "first step after hull-contact interruption")


def test_state_comparator_flags_single_leaf_drift():
    """Negative control: the comparator itself must not be vacuous."""
    sim = IceDEM(DEMConfig(nx=3, ny=2, drag=0, fix_edges=False, tool_gap=1))
    sim.step(_loads(sim, 0))
    good = _roundtrip(sim.snapshot())
    assert not _state_diffs(good, copy.deepcopy(good))

    def bumped(value):
        return math.nextafter(value, math.inf)

    mutations = {
        "state.positions_m[0][0]": lambda s: s["positions_m"][0].__setitem__(0, bumped(s["positions_m"][0][0])),
        "state.velocities_m_s[1][1]": lambda s: s["velocities_m_s"][1].__setitem__(1, bumped(s["velocities_m_s"][1][1])),
        "state.accounting.external_impulse_Ns[0]": lambda s: s["accounting"]["external_impulse_Ns"].__setitem__(
            0, bumped(s["accounting"]["external_impulse_Ns"][0])),
        "state.last_step.external_work_J": lambda s: s["last_step"].__setitem__(
            "external_work_J", bumped(s["last_step"]["external_work_J"])),
        "state.failure_mode[7]": lambda s: s["failure_mode"].__setitem__(7, 1),
        "state.step_count": lambda s: s.__setitem__("step_count", s["step_count"] + 1),
    }
    for path, mutate in mutations.items():
        broken = copy.deepcopy(good)
        mutate(broken)
        diffs = _state_diffs(good, broken)
        assert diffs, f"comparator missed a mutation of {path}"
        assert any(diff.startswith(path) for diff in diffs), (path, diffs)
        with pytest.raises(AssertionError, match="field\\(s\\) disagree"):
            _assert_same_state(good, broken, "negative control")