import copy
import json
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
