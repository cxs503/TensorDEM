# DEM contact-parameter dimension ledger and geometric-scale invariance gate

Node **P482-N06** (TensorDEM).  Code: `src/tensordem/contact_param_ledger.py`
(ledger, M1/M4), `src/tensordem/contact_scale_invariance.py` (experiment,
M2/M3), driver `scripts/run_contact_scale_ledger.py`, pre-registration
`prereg/P482-N06_contact_scale_prereg.md`, raw records
`artifacts/contact_scale_ledger/`.

## 1. Problem

A DEM contact parameter is a bare number in almost every configuration file,
and the same symbol is used with different dimensions across the literature:

* rolling resistance: `M_r = mu_r |F_n|` with `mu_r` in **metres** (Ai et al.
  2011) versus `M_r = mu_r R_eff |F_n|` with `mu_r` **dimensionless** - a factor
  `R` apart;
* normal stiffness: linear `kn` [N/m] versus Hertz `kn` [N/m^(3/2)], and the
  often quoted "`kn ~ sqrt(R)`" law which is the *fixed-modulus* rule, not a
  similarity law;
* restitution silently clamped inside the dashpot helper, so `e = 1.05` runs
  instead of raising.

Coarsening a case (`R -> lambda R` with the particle count fixed) is therefore
a minefield: the case still runs, it just answers a different physics question.

## 2. What was built

### M1 - dimension ledger and fleet scan

`PARAMETERS` gives every contact/geometry parameter its SI base-dimension
vector, unit, convention and admissible range.  Required similarity exponents
are **derived**, never tabulated:

```
M_exp = density_exp + 3 * length_exp          (fixed grain density)
T_exp = (length_exp - gravity_exp) / 2        (fixed gravity, g = L/T^2)
exp(q) = a*M_exp + b*L_exp + c*T_exp          for q with dimension M^a L^b T^c
```

| parameter | unit | dimension | Froude exponent |
|---|---|---|---|
| `radius` | m | M^0 L^1 T^0 | 1 |
| `kn` | N/m | M^1 L^0 T^-2 | 2 |
| `kt` | N/m | M^1 L^0 T^-2 | 2 |
| `kr` | N*m | M^1 L^2 T^-2 | 4 |
| `mu_rolling` | - | M^0 L^0 T^0 | 0 |
| `rolling_lever` | m | M^0 L^1 T^0 | 1 |
| `mu` | - | M^0 L^0 T^0 | 0 |
| `restitution` | - | M^0 L^0 T^0 | 0 |
| `gamma` | N*s/m | M^1 L^0 T^-1 | 2.5 |
| `dt` | s | M^0 L^0 T^1 | 0.5 |
| `normal_stress` | Pa | M^1 L^-1 T^-2 | 1 |
| `shear_velocity` | m/s | M^0 L^1 T^-1 | 0.5 |
| `density` | kg/m^3 | M^1 L^-3 T^0 | 0 |
| `gravity` | m/s^2 | M^0 L^1 T^-2 | 0 |

The scan parses every `src/tensordem` module with `ast` (no imports, no torch),
declares a module *in scope* when it declares at least one registered contact
parameter, and then audits it **fail-closed**: a contact-like field that is not
in the registry is an error.  The same pass checks value ranges, detects the
rolling-convention mix-up (`mu_r` multiplied by a length while being declared
as a ratio) and reports silently clamped restitutions.

| class | count | meaning |
|---|---|---|
| contact-like fields extracted | 2966 | literal defaults in scope-bearing modules |
| labelled with a dimension | 1945 | registry + alias table + token rules |
| `CONVENTION_MISMATCH` | 17 | rolling coefficient consumed under the other convention |
| `DIMENSION_AMBIGUITY` | 18 | same symbol carries two dimensions in the fleet |
| `SILENT_CLAMP` | 33 | out-of-range input silently clamped inside a helper |

Top dimension ambiguities in the fleet (a label that cannot be attached is a
finding, so the gate reports instead of guessing):

| parameter | sites | value range | note |
|---|---|---|---|
| `punch_stiffness` | 1 module(s) | 5.000e+09 .. 5.000e+09 | linear [N/m] vs per-area [Pa] |
| `bond_stiffness` | 12 module(s) | 1200 .. 1.000e+09 | linear [N/m] vs per-area [Pa] |
| `contact_stiffness` | 10 module(s) | 2000 .. 1.000e+09 | linear [N/m] vs Hertz [N/m^(3/2)] |
| `stiffness` | 6 module(s) | 100 .. 1.000e+09 | linear [N/m] vs Hertz [N/m^(3/2)] vs modulus [Pa] |
| `contact_damping` | 11 module(s) | 0 .. 1.000e+05 | dashpot [N s/m] vs ratio [-] |
| `damping` | 9 module(s) | 0 .. 30000 | dashpot [N s/m] vs damping ratio [-] |
| `tangent_damping` | 1 module(s) | 30000 .. 30000 | dashpot [N s/m] vs ratio [-] |
| `bond_damping` | 1 module(s) | 20000 .. 20000 | dashpot [N s/m] vs ratio [-] |
| `viscous_damping` | 2 module(s) | 0 .. 2000 | dashpot [N s/m] vs ratio [-] |
| `bulk_damping` | 2 module(s) | 12 .. 40 | dashpot [N s/m] vs ratio [-] |

17 rolling-resistance multiplication site(s) consume the coefficient without a radius factor while the ledger labels it dimensionless (warning class, review list).

### M2 - geometric-scale invariance of three macroscopic probes

Same particle count (460), same lattice, `rho` and `g` fixed; `lambda = 2.0`:

| probe | protocol | decision metric |
|---|---|---|
| `repose` | dam-break column collapse on a frictional floor, free-surface flank fit | `theta_r` [deg] |
| `shear` | constant-stress direct shear, floor reaction as the transmitted shear | `tau_peak/sigma` [-] |
| `drag` | vertical blade pushed through a settled bed | `F/(rho g R^3)` [-] |
| `rebound` | single sphere dropped from 20 R (restitution anchor) | apex ratio [-] |

The transform is not hand-coded: `scale_dataclass` applies the ledger's own
exponents, so the experiment cannot drift from the published law.

| probe | metric | lambda=1 reference | lambda=2 correct transform | deviation | verdict |
|---|---|---|---|---|---|
| repose | `theta_r_deg` | 16.6632 | 16.6636 | 0.00 % | PASS |
| shear | `tau_over_sigma_peak` | 0.320054 | 0.320054 | 0.00 % | PASS |
| drag | `resistance_dimensionless` | 1080.46 | 1080.46 | 0.00 % | PASS |
| rebound | `rebound_height_ratio` | 0.226845 | 0.226845 | 0.00 % | PASS |

### M3 - negative controls

| negative control | repose | shear | drag | rebound | max | caught |
|---|---|---|---|---|---|---|
| `ctl_kn_constant` | 13.14 % | 41.35 % | 45.22 % | 4.07 % | 45.22 % | YES |
| `ctl_lever_constant` | 44.49 % | 6.40 % | 14.58 % | 0.00 % | 44.49 % | YES |
| `ctl_restitution_boost` | 5.13 % | 6.89 % | 1.76 % | 38.01 % | 38.01 % | YES |
| `ctl_restitution_out_of_range` | 21.18 % | 27.32 % | 2.34 % | 372.98 % | 372.98 % | YES |

Compliance sweep (does the "stiffness not scaled" control have a lever in the
stiff limit?  No - and that is *why* the dimensional gate, not the experiment,
carries the stiffness rule):

| kn [N/m] | repose | shear | drag | rebound |
|---|---|---|---|---|
| 150.0 | 48.57 % | 82.05 % | 52.98 % | 7.81 % |
| 600.0 | 13.14 % | 41.35 % | 45.22 % | 4.07 % |
| 2400.0 | 8.31 % | 26.41 % | 51.46 % | 2.65 % |

### M4 - CI gate

`gate_manifest.json` states the similarity law as data; `audit_manifest`
compares the observed exponent `log(variant/base)/log(lambda)` with the
required one.  Green for the correct transform, red for every mistake -
including the two literature stiffness laws:

| variant | expected by the manifest | gate verdict | deviating parameters |
|---|---|---|---|
| `lamN_similar` | OK | OK | - |
| `ctl_kn_constant` | violation | CAUGHT | `kn` |
| `ctl_lever_constant` | violation | CAUGHT | `rolling_lever` |
| `ctl_restitution_boost` | violation | CAUGHT | `restitution` |
| `ctl_restitution_out_of_range` | violation | CAUGHT | `restitution` |
| `diag_kn_sqrt` | violation | CAUGHT | `kn` |
| `diag_kn_hertz` | violation | CAUGHT | `kn` |

## 3. Verdicts

| criterion | pre-registered | measured | verdict |
|---|---|---|---|
| M2 invariance | `<= 2 %` | max 0.00 % | PASS |
| M3 controls | `> 10 %` each | min over controls 38.01 % | PASS |
| M4 gate | red/green contrast | OK (0 unexpected violation(s)) | PASS |
| M1 scan | 0 error-class violations | 0 errors in 221 modules | PASS |

## 4. Notes, limitations, caveats

* **Exact self-similarity is the result.** The transformed case reproduces the
  reference metric to machine precision (deviation 0.00 %) because a
  linear-spring DEM under Froude similarity is *exactly* self-similar once
  `kn ~ lambda^2`, `mu_r[m] ~ lambda`, `dt ~ lambda^0.5`.  The test therefore
  probes the *transform*, not the statistical noise.  Any non-zero deviation
  would be a defect in the transform or in a boundary condition.
* **The "Hertz sqrt(R)" recipe is not similarity preserving** (`diag_kn_sqrt`,
  caught) and neither is `kn ~ lambda^1.5` for a linear-spring kernel
  (`diag_kn_hertz`, also caught): for a linear spring, gravity similarity
  requires `kn ~ lambda^2`.  This is the falsified sub-hypothesis H3.
* **Restitution is a quasi-static invisible.** `e + 0.1` moves the repose angle
  by 5.13 %
  and the shear ratio by
  6.89 % -
  below the control threshold.  Restitution is only observable where it acts:
  the rebound probe gives
  38.01 %.
  A gate must therefore pair each parameter with an observable that has
  leverage on it; the ledger catches the mistake deterministically, the
  experiment supplies the physical consequence.
* **Restitution anchor bias.** Rebound apex at `e = 0.40` measures
  `e_eff = 0.476` - the
  clamped Kelvin-Voigt law (`f = max(k n delta - gamma v_n, 0)`) separates
  early and returns ~19 % more energy than the nominal `e`.  This is a
  *solver* property (visible only through an anchor), reported here rather than
  hidden: the gate compares variants, so the bias cancels; a calibration claim
  built on the nominal `e` would not.
* **Rolling resistance matters for spheres.** Without wall rolling resistance a
  sphere rolling on the floor coasts forever and the heap never comes to rest;
  the deposit stays a thin carpet (pilot observation).  The probe includes it,
  so the reference repose angle
  (16.66 deg for
  `mu_r/R = 0.3`) is rolling-dominated, not a calibrated material value.
* **Compliance regime.** The reference stiffness is
  `kn = 600 N/m` (pre-declared); in the stiff limit (`kn = 2400 N/m`) the
  stiffness control loses power
  (13.14 %
  on the repose angle).  The gate therefore does not rely on the experiment
  alone for the stiffness rule.
* Monodisperse spheres, no cohesion, smooth side walls, one realisation per
  cell (deterministic, so no confidence intervals are quoted), 460 particles.

## 5. Reproduction

```
python -m tensordem.contact_param_ledger --root .     --out artifacts/contact_scale_ledger/ledger.json --check
python scripts/run_contact_scale_ledger.py --shard i --shards 8     --kn-sweep 150,600,2400 --probes repose,shear,drag,rebound
python scripts/run_contact_scale_ledger.py --merge --kn-sweep 150,600,2400     --probes repose,shear,drag,rebound
python -m pytest tests/test_contact_param_ledger.py                  tests/test_contact_scale_invariance.py -q
python scripts/emit_contact_scale_report.py
```
