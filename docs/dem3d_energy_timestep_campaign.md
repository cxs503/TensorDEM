# DEM3D energy-ledger timestep campaign

## Purpose

This campaign measures how the existing discrete energy ledger changes when the same nominal physical duration is simulated with timestep factors 1, 1/2 and 1/4 of the conservative recommended step. It complements the fracture-objectivity tests by probing numerical bookkeeping over an evolving indenter/contact simulation.

## Run

```bash
python scripts/benchmark_dem3d_energy_timestep.py --output results-dem3d-energy-timestep
python -m unittest tests.test_dem3d_energy_timestep -v
```

Outputs:

- `dem3d_energy_timestep_report.json`: per-case acceptance and machine-readable results
- `dem3d_energy_timestep_summary.csv`: one row per timestep
- `dem3d_energy_timestep_history.csv`: sampled energy/work/dissipation histories

## Metrics and acceptance

For each timestep, the report includes timestep, steps, simulated duration, particle/bond counts, broken bonds, indenter work, drag/contact damping, fracture-release estimate, final mechanical energy, absolute and relative energy-balance residuals, and repeat signatures.

The current acceptance gate checks finite values and deterministic repeatability for each timestep. It deliberately does **not** set an arbitrary universal upper bound on energy residual: the current solver uses explicit stepping and discrete work/damping estimates, and a valid bound must be established from a documented convergence study for a particular loading protocol.

## Limits

This campaign is a numerical sensitivity study, not proof of exact energy conservation. A residual that changes with timestep is expected to be studied alongside force/displacement convergence and event timing. The bond-release ledger stores the elastic energy in a linear bond at failure; it is not a calibrated fracture toughness or fracture energy (G_c). No experimental ice data are bundled or fabricated.
