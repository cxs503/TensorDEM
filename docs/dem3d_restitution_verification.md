# DEM3D restitution ↔ contact-damping mapping: closed-loop verification

Status: verification note, 2026-11-03. Code: `src/tensordem/contact_restitution.py`,
`scripts/verify_dem3d_restitution.py`, `tests/test_dem3d_restitution_verification.py`.
Artifacts: `outputs/dem3d_restitution/`.

## 1. What was verified and why

`tensordem.contact_tangential.restitution_to_damping` converts a target coefficient of
restitution *e* into a normal damping coefficient

    gamma_n = 2 zeta sqrt(k_n m_eff),   zeta = |ln e| / sqrt(pi^2 + ln^2 e),

i.e. it inverts the **unclamped** linear spring-dashpot result
`e = exp(-pi zeta / sqrt(1 - zeta^2))`. The restitution anchor in
`benchmarks/ice_floe_physics.py` (anchor 2) only checks that mapping against itself
(round trip) and against a hand-written ODE, not against the solver that ships the
contact law.

The shipped solvers (`tensordem.dem`, `tensordem.dem3d`) evaluate a
**compressive-only** normal force

    F_n = max(k_n * overlap - gamma_n * v_n, 0),

so the damping term can switch the contact off while the spheres still overlap. That is
a different contact law from the one the closed form assumes, and it is the quantity that
sets the rebound of ice floes in wet/buoyant coupling cases: restitution controls how
much translational energy survives a floe-floe or floe-hull impact, and therefore the
post-impact kinetic energy ledger and the velocity statistics reported by
`dem3d_energy`/`buoyancy_wet` studies.

The debt closed here: **the mapping had never been validated in closed loop against the
actual solver**, and the size of the clamp-induced error was unknown.

## 2. Method

Three evidence levels, evaluated on the same two-sphere normal impact:

| level | what it is | how it is computed |
|---|---|---|
| `classic` | closed form used by `restitution_to_damping` | `e = exp(-pi zeta/sqrt(1-zeta^2))` |
| `reference` | the clamped 1-D impact model the solver is supposed to reproduce | RK4 on `m_eff δ̈ = -max(k δ + γ δ̇, 0)`, event-located at separation |
| `solver` | the shipped `IceDEM3D` | two-sphere impact carved out of a 2×2×2 box, all bonds severed, non-pair particles frozen, tool parked ≥1 m away |

Setup (`docs/m2254/outputs/dem3d_restitution/dem3d_restitution_verification.json`):

| quantity | value |
|---|---|
| stiffness `k_n` | 5000 N/m |
| particle radius / density | 0.025 m / 917 kg/m³ |
| particle mass | 0.0600175 kg |
| closing speed | 0.5 m/s |
| initial gap | 0.01 m |
| integrator | semi-implicit (symplectic) Euler, `dt = recommended_dt` and `dt/2` |
| measured `e` | `|v_rel|` at first overlap compared with `|v_rel|` at first separation |
| drag | 0 (isolated), then 0.5 and 2.0 (default) as a separate sweep |
| effective mass | `m` for a frozen partner (wall), `m/2` for two free spheres |

The reference integrator is resolved by `steps = 4000` per undamped contact period;
doubling to 8000 changes `e` by < 1e-6 relative (test
`test_reference_integration_is_step_independent`).

## 3. Results

### 3.1 Solver vs reference vs classic closed form

| gamma (N s/m) | zeta | e classic | e clamped ref | e solver (dt) | solver-vs-ref | solver-vs-classic |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 1.0000 | 1.0000 | 1.0001 | +0.01 % | +0.01 % |
| 1.0 | 0.029 | 0.9133 | 0.9148 | 0.9158 | +0.11 % | +0.28 % |
| 2.0 | 0.058 | 0.8339 | 0.8395 | 0.8411 | +0.19 % | +0.86 % |
| 3.0 | 0.087 | 0.7611 | 0.7726 | 0.7747 | +0.27 % | +1.79 % |
| 5.0 | 0.144 | 0.6324 | 0.6597 | 0.6583 | -0.21 % | +4.09 % |
| 8.0 | 0.231 | 0.4745 | 0.5300 | 0.5287 | -0.23 % | +11.44 % |
| 10.0 | 0.289 | 0.3879 | 0.4628 | 0.4632 | +0.10 % | +19.43 % |
| 15.0 | 0.433 | 0.2212 | 0.3400 | 0.3381 | -0.57 % | +52.86 % |

Two conclusions:

1. **The contact law is implemented as documented.** The solver tracks the clamped
   reference to within 0.57 % over the whole damping range (max at zeta = 0.43), i.e.
   the contact branch, the relative-velocity sign convention and the separation
   detection are all correct.
2. **The closed-form mapping is the only inaccurate piece, and only for larger
   damping.** It is inside 3 % up to zeta ≈ 0.087 (`e_classic >= 0.76`) and then
   diverges: +4.1 % at zeta = 0.144, +19.4 % at zeta = 0.289, +52.9 % at zeta = 0.433.
   Because the force clamp releases the contact while `overlap > 0` (reference
   `release_overlap_m` grows from 1.8e-4 m at zeta = 0.03 to 1.0e-3 m at zeta = 0.43),
   the elastic energy still in the "spring" is never spent and the rebound is faster
   than the unclamped oscillator predicts.

The deviation is **not** a discretisation artefact: halving `dt` changes the measured
`e` by at most 0.36 % and does not reduce the gap to the classic curve (test
`test_classic_mapping_is_within_3pct_only_for_small_damping`).

### 3.2 Closing the loop: solver-calibrated inverse mapping

`damping_for_target_restitution(e, k_n, m_eff)` bisects the damping ratio on the clamped
reference model, so it inverts the law the solver actually integrates.

| e target | gamma clamped map | gamma classic map | e measured (clamped map) | e measured (classic map) | clamped map err | classic map err |
|---|---|---|---|---|---|---|
| 0.90 | 1.187 | 1.161 | 0.9012 | 0.9032 | +0.13 % | +0.35 % |
| 0.80 | 2.576 | 2.455 | 0.8020 | 0.8100 | +0.25 % | +1.25 % |
| 0.70 | 4.234 | 3.908 | 0.7025 | 0.7207 | +0.36 % | +2.95 % |
| 0.60 | 6.266 | 5.560 | 0.5975 | 0.6329 | -0.42 % | +5.49 % |
| 0.50 | 8.844 | 7.465 | 0.4981 | 0.5482 | -0.38 % | +9.64 % |
| 0.40 | 12.282 | 9.701 | 0.4001 | 0.4723 | +0.02 % | +18.09 % |

The calibrated mapping reproduces the requested restitution within 0.42 % across
`e = 0.4 … 0.9`; the classic mapping degrades from 0.35 % to 18.1 % over the same range.

### 3.3 Reduced mass for two free spheres

| gamma | m_eff (kg) | zeta | e classic | e clamped ref | e solver |
|---|---|---|---|---|---|
| 2.0 | 0.0300 | 0.082 | 0.7731 | 0.7835 | 0.7866 |
| 5.0 | 0.0300 | 0.204 | 0.5195 | 0.5659 | 0.5632 |

An equal-mass pair behaves as `m_eff = m/2` (the classic curve computed with the
unreduced mass would be wrong by the same clamp bias plus a factor sqrt(2) in zeta), and
the solver matches the reduced-mass reference to +0.40 % and -0.48 %.

### 3.4 Confound: the global drag term

| gamma | drag (N s/m) | e measured |
|---|---|---|
| 2.0 | 0.0 | 0.8411 |
| 2.0 | 0.5 | 0.8033 |
| 2.0 | 2.0 | 0.7014 |
| 5.0 | 0.0 | 0.6583 |
| 5.0 | 0.5 | 0.6284 |
| 5.0 | 2.0 | 0.5476 |

`DEM3DConfig` applies `-drag * v` to **every** particle, including during contact. At the
default `drag = 2 N s/m` a nominal contact `e = 0.84` is measured as 0.70 (-16.6 %), i.e.
the drag channel dominates the observed rebound. Any restitution claim taken from a run
with the default drag is not attributable to `contact_damping` alone; the drag-free
measurement is the one comparable with a restitution coefficient.

## 4. Recommendations

1. Use `restitution_to_damping` only while `zeta <= 0.1` (≈ `e >= 0.75`). Beyond that use
   `damping_for_target_restitution`, which inverts the shipped contact law; the
   difference matters from `e ≈ 0.60` upwards (5 % error) and is large below `e = 0.5`
   (10–18 %).
2. When reporting a coefficient of restitution from a simulation, set `drag = 0` (or
   declare the drag value), otherwise the measured value mixes contact and drag
   dissipation.
3. For equal-mass pairs use `m_eff = m/2`; the pair reference is reproduced to < 0.5 %.
4. Keep `contact_damping` explicitly documented as a linear (not Hertzian) damping
   coefficient; the mapping above is derived for linear springs only.

## 5. Limitations

- One-dimensional normal impacts only: no tangential friction, no rotation, no rolling,
  no adhesion, no Hertzian contact. The verified law is the linear spring-dashpot branch
  that `dem3d` currently implements.
- The reference model is drag-free by construction; the drag sweep quantifies the
  confound but does not fold drag into a joint mapping.
- Stiffness, radius and closing speed were held at a single representative point
  (`k_n = 5000 N/m`, `m = 0.06 kg`, `v_in = 0.5 m/s`). The clamp bias depends on zeta and
  the residual-overlap ratio only through the same dimensionless grouping, but
  multi-parameter sweeps (`k_n`, `m`, `v_in`) that would confirm this are not run here.
- The impact is carved out of a bonded lattice with `bonded=False`; interactions with
  intact bonds (bonded neighbours inside the contact branch) are deliberately excluded,
  so this note says nothing about bonded-particle restitution.
- Contact damping in the solver is velocity-threshold free (no `clamp_min` on `v_n`
  before damping); the match with the reference confirms this is consistent, but it also
  means viscous damping is active for all overlaps.

## 6. Reproduction

    PYTHONPATH=src python scripts/verify_dem3d_restitution.py --output outputs/dem3d_restitution
    PYTHONPATH=src python -m pytest tests/test_dem3d_restitution_verification.py -q

Runtime is ~1 min for the full campaign and ~22 s for the test module on one CPU core.