# XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT · 3-D numeric anchor (round 2, m17960)

Anchor: `XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT`
Repo: `TensorDEM` · branch `agent/m17960`
Predecessors: m17806 (round 1, 2-D anchor + frozen declarative contract), m17871 (rework round 1)

## 1. What was missing (the real gap this round closes)

Round 1 froze the *surface* that cross-repo binders consume and anchored the
**2-D** entry (`IceDEM`) numerically: `benchmark_physics()` compares impulse-momentum,
the semi-implicit-Euler closed form, continuum convergence, Hertz kernel, the
native clock and the restart payload against analytic references.

The frozen contract has **two** entries — `IceDEM` (2-D) and `IceDEM3D` (3-D) —
but only the 2-D entry had a numeric anchor.  For the 3-D entry the guarantee was
structural: fingerprint (`FROZEN_FINGERPRINT`), introspection (`verify()`), and
behavioural probes (`probe_semantics`, the 3-D part covers impulse-momentum,
non-accumulated loads and the `(N,3)` shape guard).  What it did **not** have was
a row-by-row *ref vs got* table with a number next to every promise: 3-D
kinematics, the integrator-exact free-flight closed form, first-order continuum
convergence, energy drift of a conservative bonded lattice, restart via the 3-D
`state_dict`/`load_state_dict` names.

That is exactly the class of gap the anchor targets: "frozen, fingerprinted,
structurally gated — but no numeric anchor of its own."

## 2. What was added

| file | role |
|---|---|
| `src/tensordem/binding_contract.py` | new `benchmark_physics_3d()`, `free_flight_case_3d()`, `isolated_3d_model()`, `BENCHMARK_3D_ANCHORS` |
| `src/tensordem/cli.py` | `tensordem.cli binding-contract` now runs **both** benchmarks and fails on either `FAIL`; new `--benchmark-energy-steps-3d` |
| `scripts/run_binding_contract_3d_benchmark.py` | standalone entry point; writes JSON artifact + generated markdown table |
| `scripts/xval_library_binding_contract.py` | XVAL report gains a `benchmark_physics_3d` section and two 3-D verdict lines |
| `tests/test_binding_contract_3d_benchmark.py` | 9 tests pinning rows, tolerances, convergence order, restart exactness and the *teeth* of the reference |
| `artifacts/xval_library_binding_contract/benchmark_physics_3d.json` | the run artifact (single source of truth for every number below) |
| `artifacts/xval_library_binding_contract/benchmark_physics_3d_table.md` | generated table (zero hand transcription) |

The model used for every row is `isolated_3d_model()`: a 5×4×3 = 60-particle
lattice with no supports, no contact dashpot, no drag, indenter parked at
`tool_gap = 10 m`, friction off.  `test_reference_model_is_force_free` asserts
`forces()` returns exactly zero, i.e. the analytic references are legitimate for
this configuration — if some other force switched on, the reference would become
wrong and the test would catch it.

## 3. Rows and references

| row | analytic / bitwise reference |
|---|---|
| `impulse_momentum_3d` | `m·ΣΔv = n·dt·ΣF_ext` (internal bond/contact forces momentum-neutral, Newton's 2nd law) |
| `loads_not_accumulated_3d` | impulse projection ratio = 1 (accumulating the load would give `1+2+…+n`) |
| `free_flight_symplectic_3d` | semi-implicit Euler closed form `x_n − x_0 = a·dt²·n(n+1)/2` (Hairer/Lubich/Wanner 2006, §I.1) |
| `free_flight_continuum_3d_dt_x{1,0.5,0.25}` | continuum `0.5·a·t²`, owner acceptance 15 % |
| `free_flight_convergence_order_3d` | observed order `p ≈ 1` in `err ∝ dt^p` |
| `energy_drift_conservative_3d` | `E(t) − E(0) = 0` is a constant of motion (self-declared tolerance 2 %) |
| `hertz_{reduced_modulus,effective_radius,normal_force}_3d` | Hertz (1882): `E* = E/(2(1−ν²))`, `R* = R/2`, `F = (4/3)E*√R*·δ^1.5` |
| `clock_n_dt_3d` | native clock `t = n·dt` |
| `restart_roundtrip_3d`, `restart_continuation_3d` | `load_state_dict(state_dict())` reproduces the trajectory bitwise (0 m) |
| `shape_guard_3d` | `(N,2)` load must raise `ValueError` on the frozen `(N,3)` surface |

## 4. Results

The generated table is `artifacts/xval_library_binding_contract/benchmark_physics_3d_table.md`;
`artifacts/xval_library_binding_contract/benchmark_physics_3d.json` carries the same
numbers plus the environment (python/torch/platform, contract fingerprint).

Summary of that run (15/15 rows MET, worst relative error `8.642e-03`):

- identity/closed-form rows at machine precision: impulse-momentum `2.53e-16`,
  load non-accumulation `3.33e-16`, symplectic closed form `3.09e-13`, clock
  `6.57e-16`, Hertz kernel `0.0`;
- continuum error halves with `dt` (7.14e-03 → 3.57e-03 → 1.79e-03), observed
  orders `[1.0, 1.0]`;
- energy drift of the conservative bonded lattice `8.64e-03` over 1000 steps
  (self-declared line 2 %);
- restart bitwise exact (`0.0 m` round-trip and after a further step).

## 5. Error judgement vs the reference

The owner rule is ">15 % is unacceptable"; every row is therefore checked against
15 % *and* against its own tighter pre-registered tolerance (the tighter one is
what the row verdict uses).  Worst row is the energy-drift row at 0.86 %, i.e.
**6× inside** the 15 % line; every analytic row is at machine precision
(`≤ 3.1e-13`).  Answer to "is the key quantity < 3 %?": yes for all 15 rows —
the worst is 0.86 %.

Interpretation limitation: the analytic rows validate the *binding semantics*
(dt, clock, per-step load consumption, restart payload, kernel constants).  They
do not validate contact physics against experiment; that is the job of the case
benchmarks (`benchmarks/case_*`).

## 6. Notes, limitations, honest caveats

1. **Self-declared tolerances.** 15 % is the owner line; the identity rows use
   1e-12/1e-11 because they are exact identities, and the energy row uses 2 %.
   These are pre-registered in code (`benchmark_physics_3d`), not tuned after the
   numbers were seen.
2. **Energy drift is integrator drift, not error.**  `8.64e-03` over 1000 steps
   of a bonded, damping-free lattice is the semi-implicit-Euler energy behaviour
   of that stiff bonded system; it is bounded (the test suite pins it), but the
   row's 2 % line is a stability monitor, not an accuracy claim.
3. **The 3-D surface has no `load_time` stamp** (frozen property).  The clock-guard
   branch is therefore only exercised by the 2-D entry; the 3-D rows check the
   clock itself (`clock_n_dt_3d`) instead.
4. **Small lattices.** 60 particles, ~1000 energy steps, ~14 s wall on one core.
   The rows are semantic anchors, not a convergence study of the bonded contact
   model.
5. **Reference has teeth, verified.**  At `n = 4` steps the symplectic closed form
   and the continuum answer differ by `1/n = 25 %`; the benchmark matches the
   former to `3.1e-13` while the latter would be rejected at the 15 % line.
   `test_reference_has_teeth_against_the_naive_continuum_form` pins this, so the
   rows cannot be satisfied by a wrong reference.

## 7. Reproduction

```bash
# one line, in the repo checkout, no GPU (~14 s on one core)
python scripts/run_binding_contract_3d_benchmark.py

# tests that pin the rows
python -m pytest tests/test_binding_contract_3d_benchmark.py -q
# existing 2-D contract suite (regression)
python -m pytest tests/test_binding_contract.py -q
# CLI path used by binders / CI
python -m tensordem.cli binding-contract --benchmark-energy-steps-3d 1000 \
    --out artifacts/xval_library_binding_contract
```

## 8. Integration (who calls the frozen surface)

- `src/tensordem/dem3d.py:1808` — `load_state_dict()` gates the restart payload on
  `assert_entry_conforms(..., "IceDEM3D")` (production 3-D solver path).
- `src/tensordem/dem.py:8` — the same gate on the 2-D solver.
- `src/tensordem/cli.py:220` — `binding-contract` subcommand: `verify()` +
  `probe_semantics()` + `benchmark_physics()` + `benchmark_physics_3d()`; exits
  non-zero on any drift or failed row.
- `scripts/run_binding_contract_3d_benchmark.py`, `scripts/xval_library_binding_contract.py`
  — run entries that write the artifacts.

The 3-D benchmark is therefore not a detached script: the same function is the
CLI gate and the XVAL report section.