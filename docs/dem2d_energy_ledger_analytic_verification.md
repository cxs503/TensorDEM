# Closed-form verification of the 2-D `IceDEM` energy ledger

**Status:** the 2-D engine's cumulative energy channels are now scored against
independent closed forms, not only against their own balance identity.

## 1. The debt

`src/tensordem/dem.py` accumulates `tool_work_J`, `external_work_J`,
`damping_dissipation_J`, `fracture_release_J`, `pair_contact_acquired_J` and a
four-term `mechanical_energy()`. `tests/test_accounting_ledger_validation.py`
closes the identity

```
dE_mech = W_external + W_tool - Q_damping - E_fracture + E_pair_contact_acquired
```

on indentation runs — but an identity cannot grade the *numbers*: a channel that
is uniformly mis-scaled still closes the balance when the opposite channel
absorbs the error. The 3-D engine had a closed-form audit
(`docs/dem3d_energy_ledger_analytic_verification.md`); the 2-D engine — the one
the hull/ridge benchmarks use — had none. That gap was recorded as the next
verification debt in the 2-D restitution closure study (report 3800, §5), and it
is what this increment pays off.

## 2. Method

Two exactly solvable subsystems are driven through the *production* `step()`;
the subsystem state of a fresh solver is overwritten before the first step, so
no solver source is modified and the default numerical path is untouched.

### Case A — exponential viscous drag decay

One free particle, mass `m = ρ π r² t = 0.360105 kg`, drag `c = 2.0 N·s/m`, no
bonds, no contacts, tool parked at `1e-9 m/s`:

```
m dv/dt = -c v      ->  v(t) = v0 exp(-t/tau),   tau = m/c = 0.180053 s
D(t) = ∫ c v² dt'   ->  D(t) = 0.5 m v0² (1 - exp(-2t/tau))
```

Run to `t = 1.5 tau`, `v0 = 0.5 m/s`. The ledger samples the dissipated power at
the *pre-update* velocity, so each step over-records `0.5 c v² (dt c/m) dt`, i.e.
a fraction `0.5 (c/m) dt` of the dissipated energy — the residual has a closed
form, `R_pred(t) = 0.5 (c/m) dt D(t)`.

### Case B — bonded harmonic oscillator

Two free particles joined by one linear bond `k = 2000 N/m`, an amplitude of
0.5 % of the rest length (so the linear bond law is the exact model), fracture
thresholds lifted, released from rest:

```
xi'' = -omega² xi,  omega = sqrt(2k/m) = 333.33 rad/s,  xi(t) = A cos(omega t)
```

### Case C — tool-work channel under refinement

Tool indentation of a 7×3 bonded lattice, `tool_speed = 0.2 m/s`, fixed duration
0.12 s, at `dt`, `dt/2`, `dt/4`. The balance residual must shrink with `dt`; a
non-vanishing floor would mean a bookkeeping defect (as in the 3-D
`pair_contact_acquired` case).

## 3. Results

Relative errors at the production `dt` (`dt = 0.001014 s` for case A, 266 steps):

| case | quantity | engine | closed form | rel. error | <3 % ? |
|---|---|---|---|---|---|
| A | `v(t)` [m/s] | 0.111258 | 0.111730 | 0.423 % | PASS |
| A | kinetic energy [J] | 2.228742e-03 | 2.247708e-03 | 0.844 % | PASS |
| A | `Q_damping` [J] | 4.290524e-02 | 4.276542e-02 | 0.327 % | PASS |
| A | residual vs `R_pred` [J] | 1.208542e-04 | 1.204604e-04 | 0.33 % | PASS |
| B | ledger residual / `E0` | 1.701e-07 J | 0 (conservative) | 0.272 % | PASS |
| B | `xi(t)` vs `A cos(omega t)` (|err|/A) | — | — | **6.17 %** | **FAIL (see §4)** |
| C | balance residual / energy scale | 3.610e-04 J | 0 | 0.438 % | PASS |

Refinement (the discriminator between bookkeeping and discretisation):

| `dt/dt0` | A `v` | A `Q_damping` | B `xi` (% of A) | B residual | C residual |
|---|---|---|---|---|---|
| 1 | 0.423 % | 0.327 % | 6.172 % | 0.272 % | 0.438 % |
| 1/2 | 0.212 % | 0.163 % | 2.879 % | 0.114 % | 0.218 % |
| 1/4 | 0.106 % | 0.082 % | 1.388 % | 0.019 % | 0.109 % |

Every column halves with `dt` — first-order convergence consistent with
semi-implicit Euler plus a rectangle-rule power quadrature. Nothing sits on a
floor, so the residual is discretisation, not bookkeeping.

## 4. Attention points / limits

1. **Case B honestly misses the 3 % bar, and it is not a ledger defect.** The
   *trajectory* error (6.17 % of the amplitude at production `dt`) is the
   symplectic-Euler phase/amplitude error at `omega dt = 0.107`; halving `dt`
   gives 2.879 %, quartering 1.388 % (linear). The ledger residual for the same
   run is 0.272 %, i.e. the bookkeeping is right while the integrator is the
   limiter. This is reported as a FAIL for that anchor rather than re-tuned by
   shrinking the amplitude.
2. **Normalisation matters.** A trajectory error must be normalised by the
   amplitude, never by the instantaneous reference: at a zero crossing the
   relative error diverges while the physical error is `O(dt)`. The first
   implementation made this mistake and reported 649 %.
3. **Residual sign convention.** `residual > 0` means the ledger under-reports
   energy input. For case A the sign is set by pre-update power sampling, which
   is why the closed form (`+0.5 (c/m) dt D`) matches rather than its negative.
4. **Provenance trap (encountered).** `import tensordem` on this host resolved
   to a *sibling* worktree (`work/m991/src`) through a leaked `sys.path` entry,
   so the audit would have scored another agent's engine. The module exposes
   `ENGINE_PATH` and there is a provenance test; runs must use
   `PYTHONPATH=$PWD/src`.
5. **Scope.** Cases A and B are algebraic identities, so agreement proves
   *implementation* fidelity to the stated law; they do not validate the ice
   material model, nor that `bond_stiffness` corresponds to any measured
   modulus (see `docs/dem3d_lattice_elasticity.md`). `fracture_release_J` and
   `pair_contact_acquired_J` are still covered only by the identity test — a
   closed-form fracture-release anchor (released energy `0.5 k xi_break²`) is
   the natural next debt.
6. **Load budget.** All three cases are single-digit-thousand particle-steps on
   CPU, so this audit is safe to run during a saturated campaign (it was run at
   load15 ≈ 126 on 128 cores).

## 5. Pointers

| artefact | path |
|---|---|
| evidence generator | `scripts/verify_dem2d_energy_ledger.py` |
| tests (10) | `tests/test_dem2d_energy_ledger.py` |
| closure report | `docs/artifacts/dem2d_energy_ledger/ledger_closure.json` |
| per-case table | `docs/artifacts/dem2d_energy_ledger/ledger_cases.csv` |
| anchor table | `docs/artifacts/dem2d_energy_ledger/ledger_anchors.csv` |
| figure | `docs/artifacts/dem2d_energy_ledger/ledger_closure.png` |
| 3-D analogue | `docs/dem3d_energy_ledger_analytic_verification.md` |
| identity-only predecessor | `tests/test_accounting_ledger_validation.py` |

## 6. Reproduction

```bash
PYTHONPATH=$PWD/src python scripts/verify_dem2d_energy_ledger.py \
    --output docs/artifacts/dem2d_energy_ledger
PYTHONPATH=$PWD/src python -m pytest tests/test_dem2d_energy_ledger.py -q   # 10 passed
```