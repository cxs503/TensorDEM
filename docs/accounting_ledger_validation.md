# Impulse and energy ledger validation (2-D bonded disks)

This note closes a validation debt on the 2-D accounting ledger that ships with
`IceDEM.diagnostics()`, `snapshot()` and the `tensordem` CLI `accounting.json`.
Before this study the ledger was covered only by presence/finiteness assertions
(`tests/test_accounting_output.py`); no channel had ever been confronted with an
independently known budget. The verification below does that, and it exposes a
systematic bookkeeping defect that has now been fixed.

## 1. Case and closed-form references

The reference budgets are exact statements of classical mechanics for the
existing semi-implicit Euler integrator, so no experimental data is required to
adjudicate the ledger:

```text
momentum:  m * sum_i v_i(t) = I_external + I_support + I_drag + I_tool   (exact)
energy:    dE_mech          = W_external + W_tool - Q_damping - E_fracture + E_retained
```

`E_retained` is the term this study added. A bond that fails while its pair is
still overlapping does not disappear: the pair immediately acts as a repulsive
contact and keeps `0.5 * k_contact * overlap^2` of potential energy. Reporting
the full bond spring energy as "released" therefore double counts it.

Case: `nx=7`, `ny=3`, tool speed 0.2 m/s,
duration 0.12 s, default `DEMConfig` material data (ice,
`bond_stiffness = contact_stiffness = 2000` N/m, `contact_damping = 5` Ns/m,
`drag = 2` Ns/m), 2-D bonded disks in SI units. Reproduce with:

```bash
python scripts/validate_accounting_ledger.py        # writes outputs/accounting_ledger/*
python -m pytest tests/test_accounting_ledger_validation.py -q
```

## 2. Integral momentum balance

Summing Newton's second law over all particles makes internal pair forces cancel,
so the four ledger channels must reproduce the total momentum to roundoff. The
identity holds at every step of the indentation run:

| quantity | value |
| --- | --- |
| steps simulated | 345 |
| time step dt | 3.4793e-04 s |
| max relative momentum error | 8.416e-08 |
| mean relative momentum error | 1.752e-08 |

The residual has no trend and alternates sign, i.e. it is floating-point
accumulation, not a modelling error. This pins the sign conventions of all four
channels (external load, fixity clamping, drag, ice-on-tool reaction).

## 3. Integral energy balance

| dt [s] | broken bonds | dE_mech [J] | W_tool [J] | Q_damp [J] | E_release [J] | E_retained [J] | raw resid [J] | raw/scale | closed resid [J] | closed/scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.479e-04 | 21 | +8.2350e-02 | +1.0314e-01 | 1.2655e-02 | 1.3297e-02 | 4.7985e-03 | +5.1595e-03 | 2.604% | +3.6102e-04 | 0.182% |
| 1.740e-04 | 21 | +8.2442e-02 | +1.0334e-01 | 1.2628e-02 | 1.3254e-02 | 4.8077e-03 | +4.9873e-03 | 2.514% | +1.7959e-04 | 0.091% |
| 8.698e-05 | 21 | +8.2486e-02 | +1.0347e-01 | 1.2657e-02 | 1.3193e-02 | 4.7778e-03 | +4.8675e-03 | 2.451% | +8.9666e-05 | 0.045% |

Reading of the table:

- **Raw ledger deficit is time-step independent.** It sits near 2.5 % of the
  energy scale at every dt (ratios 1.03, 1.02); halving dt does not shrink it.
  A discretization artifact would halve. The documented "semi-implicit Euler
  explains the residual" reading is therefore wrong for failure-dominated runs.
- **The deficit is exactly the retained pair-contact energy.** At the finest
  step the raw residual (+5.1595e-03 J) and the credited term
  (4.7985e-03 J) agree to two digits. The credited
  term matches the analytic `0.5*k*overlap^2` of the broken pairs to
  8.7e-19 J (machine precision), so it is not a tuned fudge factor.
- **The retained share is large.** 36.1%, 36.3%, 36.2% of the reported fracture release
  is energy that never left the system, because shear failure of a lattice bond
  usually happens while the pair is still compressed.
- **After the credit the identity closes and converges.** Residual drops to
  0.182% / 0.091% / 0.045% for
  dt / dt2 / dt4 and the ratios are 2.01, 2.00, i.e. clean first-order
  convergence of the genuine discretization residue.

![Ledger validation](outputs/accounting_ledger/ledger_validation.png)

![Momentum balance](outputs/accounting_ledger/momentum_balance.png)

## 4. What changed in the solver

`src/tensordem/dem.py` (additive only, no force-law change):

- new scalar ledger channel `pair_contact_acquired_J`, accumulated at the moment
  a pair fails and exposed by `diagnostics()`;
- restart tolerates checkpoints that predate the channel (`ADDITIVE_ACCOUNTING_KEYS`)
  and restores it as zero, so old `tensordem.restart/1` files still load.

`fracture_release_J` semantics are unchanged: it remains the elastic spring
energy removed from the bond (still not a calibrated `Gc`). It is simply no
longer the only channel a closure calculation needs.

## 5. Error judgement against the reference

| key quantity | reference | achieved | threshold | verdict |
| --- | --- | --- | --- | --- |
| momentum balance residual | 0 (exact identity) | 8.416e-08 relative | < 1e-6 | pass |
| credited pair-contact energy | analytic `0.5 k overlap^2` | 8.7e-19 J absolute | < 1e-12 J | pass |
| energy closure, no failure | exact identity | < 5e-4 relative | < 1e-2 | pass |
| energy closure, with failure (corrected) | exact identity | 0.182% relative | < 5e-3 | pass |
| energy closure, with failure (raw) | exact identity | 2.604% relative | < 5e-3 | **FAIL before fix** |
| first-order convergence of the residue | order 1 | 2 per halving | 1.8-2.2 | pass |

No quantity here is judged against an experimental measurement: this is a
bookkeeping/consistency verification, not material validation. The 3 % criterion
used by material benchmarks is not applicable and is not claimed.

## 6. Limitations and follow-ups

- The 3-D audit subclass (`dem3d_energy.py`) computes
  `energy_balance_residual_J` with the same "release = dissipated" assumption and
  the same bond-to-contact conversion exists there. The 3-D ledger has **not**
  been re-verified or patched in this change; treat its residual as an upper
  bound on dissipation until it is.
- Only the bonded-disk (no rotation, no bending, no calibrated fracture energy)
  model is covered. `E_retained` is an elastic-potential correction; it says
  nothing about fracture toughness.
- The overhead of the correction is a single sum over newly broken pairs, so the
  measurement costs nothing measurable in runtime.
