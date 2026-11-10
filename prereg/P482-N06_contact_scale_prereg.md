# P482-N06 pre-registration: DEM contact-parameter dimension ledger and scale-invariance gate

Frozen before the production run (only a 1-realisation smoke run of the three
probes at the reference stiffness was executed to check that the probes are
numerically stable and that `dt` resolves the contact).

## Decision the experiment must support
Adopt (or reject) a machine gate that blocks a coarsened DEM case whose contact
parameters do not follow the gravity/Froude similarity law.

## Alternatives
A1. No gate (status quo: contact parameters are unchecked numbers).
A2. Purely dimensional gate (ledger + exponent audit, no simulation).
A3. Dimensional gate **plus** a macroscopic invariance/negative-control
    experiment (this node).

## Hypothesis
H1 (invariance): under `L -> 2L` with `rho`, `g` fixed, `kn ~ L^2`,
`mu_r[m] ~ L`, `mu`, `e` fixed, the three macroscopic probes (repose angle,
direct-shear yield ratio, blade resistance) change by <= 2 %.
H2 (discriminating power): each of the three pre-registered mistakes changes at
least one probe by > 10 %: (a) `kn` held constant, (b) `mu_r[m]` held constant,
(c) `e + 0.1` (out of range).
H3 (falsifiable sub-hypothesis): the commonly quoted "Hertz sqrt(R)" stiffness
rule (`kn ~ L^0.5`) is *not* similarity preserving for a linear-spring kernel.

## Metrics (unit of analysis = one probe, one scale variant)
* `theta_r` [deg] - free-surface slope of a settled dam-break deposit.
* `tau_peak/sigma` [-] - mobilised peak shear ratio in direct shear.
* `F/(rho g R^3)` [-] - blade resistance in a settled bed.
Deviation = |x_variant - x_correctly_scaled| / |x_correctly_scaled|.

## Baseline / control
Baseline = the correctly transformed case at `lambda = 2` (all other variants
are compared to it). `lambda = 1` is carried as a resolution reference.

## Pre-declared thresholds and stopping rule
* acceptance: every probe `<= 2 %` deviation for the correct transform;
* discrimination: every control `> 10 %` deviation on at least one probe;
* stopping rule: one realisation per (probe, variant, stiffness level). No
  parameter is re-tuned after seeing the deviations; if a control stays below
  10 % the gate item is reported as honest FAIL and the compliance sweep is
  reported as the explanation.

## Variables and confounders
Controlled: particle count (identical lattice, 460 particles), lattice ratios,
grain density, gravity, seed, initial packing. Changed: `R`, `kn`, `mu_r`,
`e` according to the variant.
Known confounders: (i) the time step is tied to the contact period, so a
stiffness mistake also changes `dt`; the controls therefore inherit the
correctly scaled `dt` of the `lambda = 2` case; (ii) implicit DEM damping
depends on `kn`; (iii) a soft-contact reference regime is used because the
stiffness control has no lever in the rigid limit - this is quantified by a
compliance sweep (kn = 150 / 600 / 2400 N/m).

## Environment / reproducibility
`tensordem.contact_scale_invariance`, `granular3d_core` contact laws, numpy
2.4 + scipy 1.16, 460 particles, seed 20260810, deterministic (no RNG in the
run). Commands: `python scripts/run_contact_scale_ledger.py --shard i --shards 8`
then `--merge`. Raw records: `artifacts/contact_scale_ledger/shards/`.

## Analysis pre-declared
Deviation per (probe, variant); verdicts PASS / FAIL (invariance) and
CAUGHT / MISSED (negative control); gate = AND over both families. Uncertainty:
single realisation, so differences are deterministic numerical differences, not
statistical ones - no confidence interval is quoted, and the resolution limit is
documented instead.

## Interpretation limits
The probes are numerical invariance probes of this solver, not laboratory
validations of a material; monodisperse spheres, no cohesion, smooth side walls,
rolling resistance at walls included (without it a sphere coasts forever).

## Amendment 1 (recorded before the production campaign, after the pilot)

The pilot (one realisation per probe at the reference stiffness) showed that
(i) a quasi-static deposit at `e = 0.95` never settles, so a base restitution
high enough for `e + 0.1` to leave `[0, 1]` destroys the repose/shear/drag
probes, and (ii) restitution hardly influences a quasi-static probe at all.
Consequences, fixed before the campaign:

* base case keeps `e = 0.40`; the pre-registered `e + 0.1` control therefore
  tests *sensitivity* (0.40 -> 0.50, still admissible);
* the *out-of-range* branch is added as a separate control (`e = 1.05`) and is
  additionally covered by the deterministic ledger/gate audit;
* a fourth probe `rebound` (single sphere dropped from 20 R, apex ratio = e^2)
  is added because it is the observable that restitution actually controls.
  Its anchor value is reported as measured-vs-nominal restitution.
