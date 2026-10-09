"""Regenerate the independent constant-acceleration timestep audit."""
import json
from pathlib import Path
import torch
from tensordem import DEMConfig, IceDEM

base = DEMConfig(nx=3, ny=2, drag=0, fix_edges=False, tool_gap=1)
records = []
for divisor in (1, 2, 4):
    sim = IceDEM(DEMConfig(nx=3, ny=2, drag=0, fix_edges=False, tool_gap=1,
                          dt=base.recommended_dt / divisor))
    initial = sim.positions.clone()
    force = torch.zeros_like(initial)
    force[:, 0] = sim.config.mass
    for _ in range(20 * divisor):
        sim.step(force, load_time=sim.time)
    energy, accounting = sim.mechanical_energy(), sim.accounting
    records.append(dict(dt_s=sim.dt, time_s=sim.time,
        max_position_error_m=float((sim.positions[:, 0] - initial[:, 0] - .5 * sim.time**2).abs().max()),
        energy=energy, accounting=accounting,
        energy_residual_J=energy['mechanical_J'] + accounting['fracture_release_J']
            + accounting['damping_dissipation_J'] - accounting['external_work_J'] - accounting['tool_work_J']))
output = Path(__file__).resolve().parents[1] / 'docs/coupling/uniform_load_refinement.json'
output.write_text(json.dumps(dict(schema='tensordem.uniform-load-refinement/1',
    analytic_acceleration_m_s2=1, records=records), indent=2, allow_nan=False) + '\n')
