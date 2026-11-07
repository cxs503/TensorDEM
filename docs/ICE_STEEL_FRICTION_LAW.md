# Ice-steel friction law `mu(v, T, p)`: calibration, DEM realisation and audit

Increment: campaign node **IN12-S2** (ice-resistance precision chain, last link).
Repository: `TensorDEM` · module `tensordem.ice_steel_friction` ·
branch `agent/m5072` · base `main`.

---

## 1. Why the constant coefficient had to go

Ice resistance is *sliding* resistance as much as crushing resistance: the
tangential traction a steel hull carries against a floe is Coulomb-saturated at

```
F_t = mu * N
```

and every part of the ice chain in this repository used a **constant** `mu`:
`ContactParameters.friction = 0.3` in the ice channel, `mu_p` as a scalar in the
platen contact, and the engineering default `mu = 0.1` for a steel hull. The
tribology of ice against steel is not constant in any of the three variables that
matter at a hull:

| mechanism | variable | sign | physical origin |
|---|---|---|---|
| frictional melting | sliding speed `v` | `mu` **decreases** | frictional power `p*v` melts a water film that separates the asperities |
| dryness | temperature `T` | `mu` **increases** as ice gets colder | less melt water, higher effective shear strength of the interface |
| pressure melting / area growth | contact pressure `p` | `mu` **decreases** | real contact area grows with `p`, pressure melting adds liquid water |

A single number therefore mis-states the resistance everywhere except at the one
condition it was fitted to — and it does so *silently*, because nothing in the
old chain carried a validity range.

## 2. The law

```
mu(v, T, p) = mu_ref
            + a * log10(v / v_ref)
            + b * (T - T_ref)
            + c * (T - T_ref)^2
            + d * log10(p / p_ref)
```

with the reference condition `v_ref = 0.01 m/s`, `T_ref = -10 degC`,
`p_ref = 0.2 MPa` placed inside the data range, so `mu_ref` is directly
interpretable ("coefficient at 1 cm/s, -10 degC, 0.2 MPa").

Two forms are implemented and both are fitted:

* `form="loglinear"` — four parameters, *exactly* the form requested by the
  campaign brief (`mu0 + a log v + b (T-T0) + c log p`).
* `form="loglinear+thermal"` — adds the `(T-T_ref)^2` curvature. **This is the
  recommended form**, and the reason is empirical, not aesthetic: the
  four-parameter law misses the near-melting validation row by 12.3 % (gate 10 %),
  the curvature term brings it to 6.7 % and halves the calibration residual
  (2.52 % → 1.06 % RMS).

Fitted coefficients (weighted least squares, weights `1/sigma` of the rows):

| form | `mu_ref` | `a` (per decade of v) | `b` (per degC) | `c` (per degC^2) | `d` (per decade of p) | fit RMS | fit max |
|---|---|---|---|---|---|---|---|
| `loglinear` | 0.098345 | -0.024000 | -0.002747 | — | -0.022341 | 2.52 % | 9.10 % |
| `loglinear+thermal` | 0.098911 | -0.024000 | -0.003177 | -3.262e-05 | -0.022663 | 1.06 % | 2.01 % |

Reading the numbers physically:

* `a = -0.024 per decade` — a factor of 10 in sliding speed changes the coefficient
  by 0.024, i.e. **40 % of the 0.06 coefficient at 0.1 m/s**; this is the dominant
  effect over a ship's speed range (0.1 → 3 m/s ⇒ `mu` falls by ~0.036).
* `b = -0.0032 per degC` — 20 degC of cooling adds ~0.064, i.e. **doubles** the
  near-melting coefficient. Cold operation is the friction-limited case.
* `d = -0.023 per decade` — contact pressure is the weakest of the three (0.02 over
  two decades), which is why the old constant was tolerable at *design* pressure
  and wrong everywhere else.

## 3. Reference set and its honesty label

17 calibration rows (3 independent source sweeps), 5 independent validation rows
(2 further sources), 2 deliberate extrapolation probes — all in
`CALIBRATION_ANCHORS` / `VALIDATION_ANCHORS` / `EXTRAPOLATION_PROBES` with a
per-row source string.

**The anchors are compiled representative values transcribed from the ranges
reported in the cited literature, not digitised measurements** (no on-line access
to the original figures at build time) — see `ANCHOR_PROVENANCE`, which is
propagated into the artifact. Consequently:

* `evidence.law_implementation = REAL` (closed-form fit, machine-checked trends),
* `evidence.anchor_set = LIT-RANGE` (compiled reference set),
* `evidence.independent_experimental_validation = WITHHELD` (no certified
  experimental set in the worktree).

The `<= 10 %` gate below is therefore a statement about the *shape consistency* of
the law with a reference set drawn from independent published sources; it is not an
independent experimental replication. This is stated in the report as well.

## 4. Validation protocol (and why three protocols)

| protocol | what it measures | result (`loglinear+thermal`) | result (`loglinear`) |
|---|---|---|---|
| independent holdout (5 rows, 2 sources) | prediction at conditions the fit never saw | **max 6.67 %**, RMS 5.11 %, 5/5 ≤ 10 % | max 12.31 % (**fails**) |
| k-fold CV, k=5, stratified by source | interpolation error, every variable present in every training set | **max 4.33 %** | max 9.18 % |
| leave-one-source-out | — | **degenerate** (all 3 folds `identifiable=False`) | degenerate |

The LOSO result is a *structural* finding, reported rather than hidden: each source
sweep varies exactly one variable, so removing a source removes that variable from
the training set, the design matrix loses rank and the corresponding coefficient is
unidentifiable. `leave_one_source_out` computes the rank of every training design
matrix and flags the fold `identifiable=False` instead of quoting a meaningless
error. That is why k-fold (source-stratified) is the protocol that actually measures
interpolation error, and it is the one carrying the accuracy claim.

Point-by-point holdout table (recommended form):

| condition | reference `mu` | predicted `mu` | error |
|---|---|---|---|
| 0.05 m/s, -5 degC, 0.3 MPa | 0.065 | 0.0614 | 5.5 % |
| 0.005 m/s, -20 degC, 0.5 MPa | 0.130 | 0.1256 | 3.4 % |
| 0.02 m/s, -3 degC, 0.1 MPa | 0.070 | 0.0747 | 6.7 % |
| 0.5 m/s, -15 degC, 1.0 MPa | 0.055 | 0.0574 | 4.3 % |
| 0.2 m/s, -8 degC, 0.5 MPa | 0.055 | 0.0522 | 5.1 % |

## 5. DEM realisation: the coefficient the solver actually uses

A fitted curve is not evidence until it drives the contact law. `dem_friction_probe`
pushes the calibrated coefficient through the existing Cundall-Strack platen
contact (`tensordem.platen_friction.platen_friction_case`, slip branch) and
recovers the *effective* coefficient from the measured slip acceleration,

```
m*a = F_h - mu_p*N   ⇒   mu_eff = (F_h - m*a_measured)/N
```

| condition | `mu` from the law | `mu_eff` measured in DEM | slope error | gate (3 %) |
|---|---|---|---|---|
| 0.01 m/s, -10 degC, 0.5 MPa | 0.0899 | 0.0889 | 1.90 % | pass |
| 0.1 m/s, -10 degC, 0.5 MPa | 0.0659 | 0.0651 | 1.98 % | pass |
| 1.0 m/s, -20 degC, 1.0 MPa | 0.0636 | 0.0629 | 1.69 % | pass |

Wiring into a running simulation is one call:
`attach_friction_field(contact, model, v=..., temperature_c=..., pressure_mpa=...)`
returns the (frozen) `ice_channel.ContactParameters` with `friction` replaced by
the law and every other field untouched. Calling it as the operating point changes
is what turns the constant into a field; the default coefficient of the channel
simulation is deliberately *not* changed here, so no existing baseline moves.

### 5.1 Audit finding on the existing defaults

The compiled ice-steel range of the calibration + validation rows is
**0.050 … 0.150** (slowest, coldest row 0.150; fastest, warmest row 0.050; the two
deliberate extrapolation probes assume 0.030 and 0.062). Two consequences for the
existing chain:

* `ContactParameters.friction = 0.3` in `tensordem.ice_channel` is **twice the
  largest compiled ice-steel coefficient** and is not an ice-steel value at all
  (it is a conservative place-holder that makes floe-floe and hull-floe tangential
  traction saturate at a level the interface cannot sustain).
* the engineering default `mu = 0.1` sits at the *cold, slow* corner of the range,
  i.e. at the **most friction-limited** operating point — using it at 1 m/s
  over-predicts the coefficient by a factor 2 (at -10 degC: 0.051) to 4.3 (at
  -2 degC: 0.023).

Neither default is changed here (they are baselines other tests rely on); the law
plus `attach_friction_field` is the replacement path.

## 6. Sensitivity: `mu` ±20 %

On the 80-point operating envelope (v = 0.01…2 m/s, T = -2…-30 degC,
p = 0.1…2 MPa) with the rule-style resistance decomposition

```
R = k_c * sigma_c(T) * A_c + mu(v,T,p) * p * A_c
```

* resistance is **strictly monotone increasing** in the `mu` scale factor at all
  80 points (44 of them inside the calibrated envelope) — the physical expectation;
* `dR/R` for `mu ± 20 %` spans **[0.35 %, 13.81 %]**, bounded above by the scale
  change itself: the friction share of the reference resistance is ~10 % (mean
  10.1 %), so a 20 % tribology error becomes a ~2–14 % resistance error depending on
  the crushing share at that condition. Sub-linear by construction, exactly as it
  must be for a linear-in-`mu` friction branch added to a `mu`-independent
  crushing branch.

## 7. Negative control: what the hard-coded `mu = 0.1` costs

Same envelope, constant `mu = 0.1` vs the law:

* inside the calibrated envelope (44 points): coefficient error up to **304 %**
  (at 0.1 m/s, -2 degC, 2 MPa the law gives 0.025 while the constant says 0.1; at
  1 m/s, -2 degC, 0.2 MPa it gives 0.023), total-resistance
  error up to **54.5 %**, RMS 12.8 %; the constant **over-predicts** at 31 of the
  44 in-envelope points (70 %) — i.e. the old default was systematically
  conservative in warm, fast, high-pressure conditions, and optimistic only in the
  cold, slow corner where the coefficient approaches 0.1.
* outside the envelope (36 points, 3 of them at the coefficient floor): the law
  itself saturates at `MU_FLOOR` and must not be used — an explicit applicability
  limit, not a result.

## 8. Figures

| file | content |
|---|---|
| `docs/ice-steel-friction/mu_vs_speed.png` | law slices at four temperatures + the speed-sweep anchors (log v axis) |
| `docs/ice-steel-friction/mu_vs_temperature.png` | law slices at four speeds + the temperature-sweep anchors |
| `docs/ice-steel-friction/mu_vs_pressure.png` | law at the reference speed + the pressure-sweep anchors |
| `docs/ice-steel-friction/parity.png` | prediction vs reference for calibration / validation / extrapolation sets with the ±10 % band |
| `docs/ice-steel-friction/sensitivity_negative_control.png` | resistance vs `mu` scale (monotone) and constant-`mu` friction error vs the calibrated coefficient |
| `docs/ice-steel-friction/dem_probe.png` | effective DEM coefficient vs prescribed coefficient with the ±3 % gate |

## 9. Limitations (all of them)

1. **Anchor provenance.** Compiled representative values, not digitised data; the
   ≤10 % gate is shape consistency against those rows (see §3). A true independent
   experimental validation is **WITHHELD**.
2. **`p*v` interaction is unconstrained.** No calibration row combines
   `v > 0.1 m/s` with `p > 1 MPa`, so the friction power density above
   ~2e5 W/m² is not measured. `IceSteelFrictionModel.coverage()` reports this, and
   the deliberate extrapolation probe at (0.2 m/s, -25 degC, 2 MPa) — where a
   heating-dominated coefficient ~0.062 is plausible — is missed by **37.6 %**.
   Inside every per-variable box, outside the power-density envelope: use the
   coverage flag, not the box, as the validity test.
3. **The separable form cannot represent heating saturation.** A saturating
   melt-film model (`mu = mu_dry(T) - h*(pv/(pv+p0))`) was fitted as an alternative
   and did *worse* on both the calibration set (RMS 2.8 % vs 1.1 %) and the
   holdout (max 45 % vs 6.7 % at the corner it was meant to fix), because the
   compiled rows do not identify the saturation constant. The separable law is
   kept and the saturation question is left open with its evidence.
4. **Resistance case is rule-style.** `k_c = 0.055` pressure-area reduction and a
   1.5 m² contact patch are documented reference values, not a class-society
   resistance prediction; only the decomposition and its monotonicity are used.
5. **No floe-scale DEM ice-resistance run.** The DEM evidence is contact-level
   (slip-branch realisation, 3 % gate). A channel transit with the field attached
   would need a runtime of the same order as the existing channel benchmarks and
   is *not* included here; `attach_friction_field` is the ready interface.
6. **Downstream ship-ice comparison not run.** `benchmarks/dem3d/ship_ice_validation`
   (MT Uikku level-ice cases) lists explicit hull-ice friction as a precondition for
   a fair resistance comparison; this law supplies the coefficient and the adapter
   supplies the wiring, but the floe-scale comparison itself is out of scope for
   this increment and is reported as not comparable rather than passed.
7. **Domain guards are hard.** `T >= 0 degC`, `v > 20 m/s`, `p > 30 MPa` raise
   instead of silently extrapolating; the floor `MU_FLOOR = 0.002` is physical
   bookkeeping (no negative friction), and its activation is reported by
   `mu_diagnostics`.

## 10. Reproduction

```bash
# full calibration + validation + audit + figures (≈1 min, 3 DEM probes)
PYTHONPATH=src python3 benchmarks/ice_steel_friction_calibration.py

# tests (27, ≈26 s including one 18 s DEM probe)
python3 -m pytest tests/test_ice_steel_friction.py -q
```

Artifacts: `artifacts/ice_steel_friction.json` (every table above, plus the
`evidence` and `acceptance` blocks), `runs/ice_steel_friction/summary.txt`.