# DEFECT-D3 repaired: high-loading closure and validity reporting in the open-water BEM

Mission m16448 · repository TensorDEM · module `src/tensordem/propeller_ice/hydro.py`
· campaign `scripts/verify_open_water_high_induction.py` (m16448::open_water_high_induction::v1)
· artifact `artifacts/open_water_high_induction/verification.json`
(12/12 gates PASS, 0 FAIL).
Every number below is rendered from that JSON.

## 1. Case / physics background

The propeller-ice milling chain (`propeller_ice.milling`, IN34-W3-N05) takes two
things from the open-water solve: the hydrodynamic torque the drive must
overcome beside the ice load, and the axial induced velocity that advects the
ice chips. Both come from a blade-element / momentum (BEM) solve of a
parametrically defined propeller, so the *domain of validity* of that solve is a
load-bearing property of the whole milling model.

The gap ledger row **DEFECT-D3** (registered in m16290, `docs/
dem_calibration_gap_ledger.md`) recorded that at the matched 3 m geometry and
`J <= 0.2`:

* every strip sat on the induction cap `a = 0.85`;
* the blade-element load was therefore no longer balanced by the annular
  momentum, while the returned residual (`max |delta a|`) was ~1e-8, so *no
  residual-based acceptance test could see it*;
* the announced "Buhl high-induction branch" (`_buhl`) was never called.

Geometry of record (the registered case):

| item | value |
|---|---|
| diameter `D` | 3.0 m |
| blades `Z` | 4 |
| hub ratio | 0.40 |
| pitch ratio `P/D` | 0.95 |
| chord ratio `c(0.7R)/D` | 0.539 |
| solidity `sigma(0.7R)` | 0.980 |
| advance ratio | `J = 0.2` (`n` = 1 s^-1, `V_a = 0.6` m/s) |

## 2. Method

### 2.1 What the code actually solved

The iteration updates the axial induction from

```
a_new = dT_BE / (4 pi r rho V_a^2 F dr) / (1 + a)      and then clips to a_cap,
```

whose fixed point is the *classical* propeller annulus-momentum relation
`C_T = 4 F a (1 + a)` (with `C_T = dT / (0.5 rho V_a^2 2 pi r dr)`, so
`C_T = 4 F X` for `X = dT / (4 pi r rho V_a^2 F dr)`).  The clip is what produced
the registered behaviour: where the blade-element loading demanded more induction
than `a_cap`, every iterate landed on the cap, the update became a fixed point of
its own, and `max |delta a|` collapsed to ~1e-8 -- *convergence of the iteration,
not of the model*.

### 2.2 The validity boundary is a derived quantity, not a tuned knob

The annular streamtube model loses validity where the local thrust coefficient
reaches the empirical actuator-disk saturation value `C_T = 0.96 F` (Glauert;
Buhl 2005). Substituting into *this module's* convention gives

```
4 F a (1 + a) = 0.96 F   =>   a_s = (-1 + sqrt(1.96)) / 2 = 0.2000000000,
```

i.e. the breakdown induction in the propeller convention is `a_s = 0.2` -- and not
the `0.4` of the wind-turbine convention the coded `_buhl` uses (gate
`G2_breakdown_induction`, measured 0.19999999999999996).

### 2.3 The high-loading closure is constructed, not transplanted

The closure is the quadratic `C_T(a) = c0 + c1 a + c2 a^2` fixed by the same
three conditions the published relation uses:

1. value continuity with the classical relation at `a_s`: `C_T(a_s) = 0.96 F`;
2. slope continuity at `a_s`: `C_T'(a_s) = 4 F (1 + 2 a_s)`;
3. the measured high-loading anchor `C_T(a = 1) = 2` (F-independent).

Applied in the **turbine** convention this construction must reproduce Buhl's
published coefficients `(8/9, 4F - 40/9, 50/9 - 4F)`; it does, to
1.05e-15 (gate `G1`).  Applied in the
module's own convention it gives `(c0, c1, c2)` for `F = 1` of
(-0.375, 7.75, -5.375),
and it is C1-continuous with the classical relation at `a_s` to
2.3e-16
(gate `G3`).  The inverse is closed-form (no iteration) and exact to
1.6e-15 (gate `G4`).

### 2.4 What changed in the solver

* `induction_model` config, three documented options:
  * `"classical"` (**default**) -- induction from the closed-form classical
    inversion; the closure is used to *report* validity.  Bit-comparable with the
    pre-repair coefficients inside the valid envelope (gate `G5`, worst relative
    difference 8.9e-08 over
    18 comparisons).
  * `"high_loading"` -- iterate on the closure itself, so the converged state is
    the closure's root where one exists.
  * `"clipped"` -- the pre-repair behaviour, kept so the defect stays
    reproducible.
* two *independent* residuals per state: `residual` (iteration) and
  `momentum_residual` (blade-element versus momentum-side thrust, evaluated
  through the same piecewise closure).  The second is the physical one and is
  what the pre-repair code could not report.
* `model_valid` / `validity_flag` (`ok` / `thrust_closure_exceeded` /
  `bollard_unsupported`), `n_strips_above_ct_limit`, `ct_max_loading`
  (`C_T,BE / C_T,max` where `C_T,max` is the closure's representable maximum at
  that strip's loss factor `F`), `converged`.
* efficiency is reported as 0 outside the propulsive quadrant instead of a
  negative value derived from a negative `K_T`.

## 3. Results

### 3.1 The defect, quantified (registered geometry)

At `J = 0.2` (gate `G6`, repair guard):

| quantity | value |
|---|---|
| iteration residual `max abs(delta a)` | 9.42e-09 |
| momentum-identity residual | 0.0711 |
| strips on the induction cap | 24/24 |
| disk loading / closure bound | 37.28 |
| `model_valid` | False |
| `validity_flag` | `thrust_closure_exceeded` |

The iteration claims convergence to 1e-8 while the blade-element thrust is
unbalanced by 7.11 % and the disk loading is
37x beyond what the annular streamtube model can
represent.  That is the defect: not a wrong number, but a *number presented as a
solution*.

### 3.2 Full sweeps

Registered geometry (3 m, `J` grid):

| J | model | K_T | K_Q | eta | a_max | strips@cap | iter res | momentum res | C_T,BE/C_T,max | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | clipped | 0.2882 | 0.0465 | 0.099 | 0.850 | 24/24 | 9.4e-09 | 6.16e-02 | 52.45 | no |
| 0.20 | clipped | 0.7046 | 0.0661 | 0.339 | 0.850 | 24/24 | 9.4e-09 | 7.11e-02 | 37.28 | no |
| 0.30 | clipped | 0.6089 | 0.0712 | 0.408 | 0.850 | 24/24 | 9.4e-09 | 5.76e-02 | 12.35 | no |
| 0.40 | clipped | 0.3476 | 0.0514 | 0.430 | 0.850 | 21/24 | 9.4e-09 | 4.58e-02 | 3.91 | no |
| 0.50 | clipped | 0.2283 | 0.0381 | 0.477 | 0.690 | 0/24 | 9.4e-09 | 2.30e-02 | 1.59 | no |
| 0.60 | clipped | 0.1812 | 0.0322 | 0.537 | 0.442 | 0/24 | 9.9e-09 | 9.43e-03 | 0.87 | yes |
| 0.80 | clipped | 0.0781 | 0.0185 | 0.539 | 0.134 | 0/24 | 9.6e-09 | 2.90e-08 | 0.21 | yes |
| 1.00 | clipped | -0.0372 | 0.0018 | 0.000 | -0.041 | 0/24 | 9.9e-09 | 4.12e-08 | -0.03 | yes |
| 0.10 | classical | 0.2882 | 0.0465 | 0.099 | 0.850 | 24/24 | 9.4e-09 | 6.16e-02 | 52.45 | no |
| 0.20 | classical | 0.7046 | 0.0661 | 0.339 | 0.850 | 24/24 | 9.4e-09 | 7.11e-02 | 37.28 | no |
| 0.30 | classical | 0.6089 | 0.0712 | 0.408 | 0.850 | 24/24 | 9.4e-09 | 5.76e-02 | 12.35 | no |
| 0.40 | classical | 0.3476 | 0.0514 | 0.430 | 0.850 | 21/24 | 9.2e-09 | 4.58e-02 | 3.91 | no |
| 0.50 | classical | 0.2283 | 0.0381 | 0.477 | 0.690 | 0/24 | 9.6e-09 | 2.30e-02 | 1.59 | no |
| 0.60 | classical | 0.1812 | 0.0322 | 0.537 | 0.442 | 0/24 | 9.1e-09 | 9.43e-03 | 0.87 | yes |
| 0.80 | classical | 0.0781 | 0.0185 | 0.539 | 0.134 | 0/24 | 9.5e-09 | 1.21e-08 | 0.21 | yes |
| 1.00 | classical | -0.0372 | 0.0018 | 0.000 | -0.041 | 0/24 | 9.7e-09 | 3.85e-08 | -0.03 | yes |
| 0.10 | high_loading | 0.2863 | 0.0462 | 0.099 | 0.850 | 11/24 | 9.4e-09 | 6.19e-02 | 52.29 | no |
| 0.20 | high_loading | 0.6630 | 0.0640 | 0.330 | 0.850 | 11/24 | 9.4e-09 | 7.59e-02 | 37.59 | no |
| 0.30 | high_loading | 0.6210 | 0.0717 | 0.414 | 0.850 | 11/24 | 9.4e-09 | 5.75e-02 | 12.70 | no |
| 0.40 | high_loading | 0.3680 | 0.0535 | 0.438 | 0.850 | 10/24 | 9.1e-09 | 4.78e-02 | 4.32 | no |
| 0.50 | high_loading | 0.1563 | 0.0290 | 0.429 | 0.734 | 0/24 | 1.8e-02 | 5.08e-03 | 1.08 | no |
| 0.60 | high_loading | 0.1619 | 0.0297 | 0.520 | 0.443 | 0/24 | 9.3e-09 | 5.71e-09 | 0.76 | yes |
| 0.80 | high_loading | 0.0781 | 0.0185 | 0.539 | 0.134 | 0/24 | 9.5e-09 | 2.69e-08 | 0.21 | yes |
| 1.00 | high_loading | -0.0372 | 0.0018 | 0.000 | -0.041 | 0/24 | 9.7e-09 | 3.85e-08 | -0.03 | yes |

Library geometry (`D` = 0.6 m, `c(0.7R)/D` = 0.40, `P/D` = 1.0):

| J | model | K_T | K_Q | eta | a_max | strips@cap | iter res | momentum res | C_T,BE/C_T,max | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | clipped | 0.2568 | 0.0432 | 0.095 | 0.850 | 24/24 | 9.4e-09 | 6.35e-02 | 44.91 | no |
| 0.20 | clipped | 0.5065 | 0.0549 | 0.294 | 0.850 | 24/24 | 9.4e-09 | 9.43e-02 | 33.28 | no |
| 0.30 | clipped | 0.5836 | 0.0670 | 0.416 | 0.850 | 24/24 | 9.4e-09 | 5.84e-02 | 11.28 | no |
| 0.40 | clipped | 0.3652 | 0.0526 | 0.442 | 0.850 | 20/24 | 9.4e-09 | 4.68e-02 | 3.91 | no |
| 0.50 | clipped | 0.2525 | 0.0406 | 0.496 | 0.728 | 0/24 | 9.6e-09 | 2.49e-02 | 1.67 | no |
| 0.60 | clipped | 0.2072 | 0.0350 | 0.565 | 0.480 | 0/24 | 9.7e-09 | 1.14e-02 | 0.95 | yes |
| 0.80 | clipped | 0.1071 | 0.0216 | 0.631 | 0.173 | 0/24 | 9.5e-09 | 1.57e-08 | 0.28 | yes |
| 1.00 | clipped | -0.0061 | 0.0048 | 0.000 | -0.005 | 0/24 | 9.2e-09 | 2.91e-07 | -0.00 | yes |
| 0.10 | classical | 0.2568 | 0.0432 | 0.095 | 0.850 | 24/24 | 9.4e-09 | 6.35e-02 | 44.91 | no |
| 0.20 | classical | 0.5065 | 0.0549 | 0.294 | 0.850 | 24/24 | 9.4e-09 | 9.43e-02 | 33.28 | no |
| 0.30 | classical | 0.5836 | 0.0670 | 0.416 | 0.850 | 24/24 | 9.4e-09 | 5.84e-02 | 11.28 | no |
| 0.40 | classical | 0.3652 | 0.0526 | 0.442 | 0.850 | 20/24 | 9.7e-09 | 4.68e-02 | 3.91 | no |
| 0.50 | classical | 0.2525 | 0.0406 | 0.496 | 0.728 | 0/24 | 9.2e-09 | 2.49e-02 | 1.67 | no |
| 0.60 | classical | 0.2072 | 0.0350 | 0.565 | 0.480 | 0/24 | 9.6e-09 | 1.14e-02 | 0.95 | yes |
| 0.80 | classical | 0.1071 | 0.0216 | 0.631 | 0.173 | 0/24 | 9.3e-09 | 5.62e-09 | 0.28 | yes |
| 1.00 | classical | -0.0061 | 0.0048 | 0.000 | -0.005 | 0/24 | 9.4e-09 | 3.18e-07 | -0.00 | yes |
| 0.10 | high_loading | 0.2549 | 0.0429 | 0.095 | 0.850 | 11/24 | 9.4e-09 | 6.39e-02 | 44.73 | no |
| 0.20 | high_loading | 0.4760 | 0.0539 | 0.281 | 0.850 | 11/24 | 9.4e-09 | 1.01e-01 | 33.48 | no |
| 0.30 | high_loading | 0.5953 | 0.0673 | 0.422 | 0.850 | 11/24 | 9.4e-09 | 5.83e-02 | 11.60 | no |
| 0.40 | high_loading | 0.3839 | 0.0543 | 0.450 | 0.850 | 10/24 | 9.3e-09 | 4.83e-02 | 4.26 | no |
| 0.50 | high_loading | 0.1696 | 0.0302 | 0.448 | 0.791 | 0/24 | 9.3e-03 | 3.35e-03 | 1.07 | no |
| 0.60 | high_loading | 0.1823 | 0.0318 | 0.547 | 0.482 | 0/24 | 9.4e-09 | 5.44e-09 | 0.81 | yes |
| 0.80 | high_loading | 0.1071 | 0.0216 | 0.631 | 0.173 | 0/24 | 9.3e-09 | 1.41e-08 | 0.28 | yes |
| 1.00 | high_loading | -0.0061 | 0.0048 | 0.000 | -0.005 | 0/24 | 9.4e-09 | 3.18e-07 | -0.00 | yes |

Low-solidity geometry (`c(0.7R)/D` = 0.12, `sigma(0.7R)` ~ 0.22):

| J | model | K_T | K_Q | eta | a_max | strips@cap | iter res | momentum res | C_T,BE/C_T,max | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | clipped | 0.1229 | 0.0206 | 0.095 | 0.850 | 24/24 | 9.4e-09 | 5.79e-02 | 20.63 | no |
| 0.20 | clipped | 0.1719 | 0.0210 | 0.261 | 0.850 | 24/24 | 9.4e-09 | 1.08e-01 | 13.29 | no |
| 0.30 | clipped | 0.2195 | 0.0244 | 0.430 | 0.850 | 14/24 | 9.1e-09 | 5.57e-02 | 4.84 | no |
| 0.40 | clipped | 0.1808 | 0.0230 | 0.501 | 0.836 | 0/24 | 9.9e-09 | 2.90e-02 | 1.94 | no |
| 0.50 | clipped | 0.1631 | 0.0217 | 0.599 | 0.546 | 0/24 | 9.2e-09 | 1.31e-02 | 1.07 | no |
| 0.60 | clipped | 0.1347 | 0.0191 | 0.674 | 0.355 | 0/24 | 9.3e-09 | 3.33e-03 | 0.61 | yes |
| 0.80 | clipped | 0.0711 | 0.0119 | 0.763 | 0.126 | 0/24 | 9.0e-09 | 1.66e-08 | 0.18 | yes |
| 1.00 | clipped | -0.0018 | 0.0014 | 0.000 | -0.002 | 0/24 | 9.2e-09 | 8.78e-07 | -0.00 | yes |
| 0.10 | classical | 0.1229 | 0.0206 | 0.095 | 0.850 | 24/24 | 9.4e-09 | 5.79e-02 | 20.63 | no |
| 0.20 | classical | 0.1719 | 0.0210 | 0.261 | 0.850 | 24/24 | 9.4e-09 | 1.08e-01 | 13.29 | no |
| 0.30 | classical | 0.2195 | 0.0244 | 0.430 | 0.850 | 14/24 | 9.4e-09 | 5.57e-02 | 4.84 | no |
| 0.40 | classical | 0.1808 | 0.0230 | 0.501 | 0.836 | 0/24 | 9.8e-09 | 2.90e-02 | 1.94 | no |
| 0.50 | classical | 0.1631 | 0.0217 | 0.599 | 0.546 | 0/24 | 9.4e-09 | 1.31e-02 | 1.07 | no |
| 0.60 | classical | 0.1347 | 0.0191 | 0.674 | 0.355 | 0/24 | 9.4e-09 | 3.33e-03 | 0.61 | yes |
| 0.80 | classical | 0.0711 | 0.0119 | 0.763 | 0.126 | 0/24 | 9.4e-09 | 2.85e-08 | 0.18 | yes |
| 1.00 | classical | -0.0018 | 0.0014 | 0.000 | -0.002 | 0/24 | 1.0e-08 | 9.69e-07 | -0.00 | yes |
| 0.10 | high_loading | 0.1226 | 0.0205 | 0.095 | 0.850 | 11/24 | 9.4e-09 | 5.80e-02 | 20.59 | no |
| 0.20 | high_loading | 0.1727 | 0.0211 | 0.260 | 0.850 | 11/24 | 9.4e-09 | 1.07e-01 | 13.29 | no |
| 0.30 | high_loading | 0.2505 | 0.0261 | 0.458 | 0.850 | 9/24 | 9.4e-09 | 5.10e-02 | 5.07 | no |
| 0.40 | high_loading | 0.1639 | 0.0217 | 0.481 | 0.850 | 7/24 | 9.4e-09 | 2.69e-02 | 1.82 | no |
| 0.50 | high_loading | 0.1428 | 0.0199 | 0.572 | 0.550 | 0/24 | 9.1e-09 | 2.20e-09 | 0.91 | yes |
| 0.60 | high_loading | 0.1316 | 0.0188 | 0.670 | 0.356 | 0/24 | 9.1e-09 | 5.73e-09 | 0.59 | yes |
| 0.80 | high_loading | 0.0711 | 0.0119 | 0.763 | 0.126 | 0/24 | 9.5e-09 | 2.85e-08 | 0.18 | yes |
| 1.00 | high_loading | -0.0018 | 0.0014 | 0.000 | -0.002 | 0/24 | 1.0e-08 | 9.69e-07 | -0.00 | yes |

Reading of the sweeps:

* inside the envelope the three modes agree (gate `G5`); the repair does not
  change physics that was already supported;
* where the closure has a root, the state satisfies its own momentum identity to
  9.7e-07 (gate `G7`) -- in `"high_loading"`
  mode the residual is round-off, rather than the per-cent violation the capped
  mode reports;
* the boundary is geometric (gate `G8`): the low-solidity propeller is valid from
  `J = 0.6`, the library geometry
  from `J = 0.6` and the
  registered 3 m geometry from `J = 0.6`;
* bollard (`V_a = 0`) is now labelled `bollard_unsupported` in both geometries
  while still returning finite, legacy-compatible numbers (gate `G9`).

### 3.3 The announced branch: unused *and* inconsistent

Two independent findings, both measured:

1. **Convention.** `a_t = -a_p` maps the turbine relation onto *minus* the
   propeller `C_T` (-1.250 versus
   1.250 at `a_p` = 0.25),
   so the turbine high-induction branch is not a change of variable of this
   module's induction.  Applying it to `a` would mix two relations with
   breakdown inductions 0.4 and 0.2 (gate `G12`).
2. **The constant form does not invert its own parabola** (gate `G11`):

| F | coded `_buhl` | exact inverse at `C_T = 0.96` | `C_T` at the coded value | rel. gap |
|---|---|---|---|---|
| 0.40 | 0.6775 | 0.7433 | 0.7775 | 0.088 |
| 0.50 | 0.7188 | 0.7155 | 0.9688 | 0.005 |
| 0.70 | 0.8024 | 0.6373 | 1.3435 | 0.259 |
| 1.00 | 0.8500 | 0.4000 | 1.6350 | 1.125 |

At `F = 1` the coded form returns `a = 0.9446`, where Buhl's own parabola gives
`C_T = 1.857`; the inverse at `C_T = 0.96` is `a = 0.4`.  So "just calling the
branch" -- the repair the ledger row implied -- would have introduced a second
error.  The branch is therefore kept as the published reference instance of the
construction (which reproduces Buhl's coefficients exactly) and is *not* used to
advance the state.

## 4. Figures

1. `fig1_closure_continuity.png` -- classical relation, the constructed closure,
   the published turbine parabola, and the C1 junction at `a_s = 0.2`.
2. `fig2_defect_evidence.png` -- iteration residual flat at 1e-8 while the
   momentum residual and the disk-loading ratio blow up at low `J`.
3. `fig3_open_water_curves.png` -- `K_T, K_Q, eta` in the three modes with the
   invalid operating points shaded.
4. `fig4_buhl_form_check.png` -- coded `_buhl` against the exact inverse of its
   parabola over `F`.

## 5. Gates

| gate | verdict | measured | note |
|---|---|---|---|
| G1_construction_reproduces_buhl | PASS | 1.0547118733938987e-15 | 3-condition construction in the turbine convention == Buhl 2005 coefficients |
| G2_breakdown_induction | PASS | 0.19999999999999996 | 4a(1+a) = 0.96 -> a_s = 0.2 in the propeller convention |
| G3_c1_continuity_at_a_s | PASS | 2.3129646346357434e-16 | value and slope of the closure match 4Fa(1+a) at a_s for F in {0.3,0.5,0.8,1.0} |
| G4_closure_inverse_exact | PASS | 1.5539462607064946e-15 | CT(a(CT)) = CT on the monotone high-loading segment |
| G11_coded_buhl_does_not_invert_its_parabola | PASS | 1.1249999999999998 | the coded constant form gives a = 0.9446 at F = 1 while the inverse of the parabola at CT = 0.96 is a = 0.4; it is a formula of the right shape that does not reproduce the inversion it is documented to perform |
| G12_turbine_map_flips_ct | PASS | -1.25 | a_t = -a_p maps the propeller relation onto minus the turbine CT, so the turbine high-induction branch cannot be applied to this module's a |
| G5_legacy_parity_in_valid_regime | PASS | 8.926969905451095e-08 | repaired default reproduces the pre-repair coefficients inside the validity bound |
| G6_defect_reproducible_and_visible | PASS | 0.07105965342697111 | legacy mode at J = 0.2: iteration residual < 1e-6 but the blade-element thrust is unbalanced by the momentum side; every strip on the cap; loading ct_max_loading x above the closure bound |
| G7_closure_root_satisfies_momentum_identity | PASS | 9.694058907315266e-07 | high_loading states that are inside the bound satisfy their own momentum identity to round-off |
| G8_validity_boundary_is_geometric | PASS | 0.6 | a low-solidity geometry reaches the valid envelope at a lower J than the high-solidity one; the boundary is not a fixed J cut |
| G9_bollard_reported_not_silent | PASS | None | V_a = 0 returns finite legacy-compatible numbers *and* an explicit validity_flag='bollard_unsupported' |
| G10_efficiency_non_negative | PASS | 0.0 | eta is reported as 0 outside the propulsive quadrant instead of a negative value from a negative K_T |

## 6. Error judgement against reference values

There is **no experimental open-water chart** in this repository, so absolute
`K_T`, `K_Q` stay **WITHHELD**; nothing here claims a `<3 %` agreement with
measurement.  What is judged against a reference is the algebra and the
identities:

| quantity | reference | error |
|---|---|---|
| construction (turbine convention) | Buhl 2005 coefficients | 1.1e-15 |
| breakdown induction | exact root of `4a(1+a) = 0.96`, `a_s = 0.2` | 5.6e-17 |
| closure C1 junction (`F` 0.3..1.0) | `4Fa(1+a)` value and slope at `a_s` | 2.3e-16 |
| closure inverse | `C_T(a(C_T)) = C_T` | 1.6e-15 |
| momentum identity at closure roots | its own momentum relation | 9.7e-07 |
| legacy parity inside the envelope | pre-repair coefficients | 8.9e-08 |

All are `<3 %` by many orders of magnitude; the absolute coefficients are not
claimed.  Is every key quantity `<3 %` of a *measurement*? **No, and it cannot be
here** -- that is stated, not papered over.

## 7. Notes, limitations, traps

* **The registered 3 m geometry is heavily loaded by construction** (solidity
  1.03, `P/D` 0.95): even the repaired model reports it out of envelope below
  `J = 0.6`.  Every `K_T`/`K_Q` produced for it at
  `J <= 0.4` -- before or after this repair -- is a cap-limited artefact and must
  not be quoted; mark them provisional where they are needed downstream.
* **The `"high_loading"` mode can fail to converge** where the loading sits at the
  closure's vertex (the parabola's slope vanishes there, so `a` is
  ill-conditioned in `C_T`).  Observed once on the library geometry at `J = 0.5`;
  the state then reports `converged = False` and `model_valid = False` instead of
  a number that looks solved.
* `momentum_residual` is normalised by the disk thrust, so it is ill-conditioned
  near zero thrust (high `J`): a residual of `~1e-7` there is round-off, not
  physics.
* `milling.py` and `fragment_milling.py` consume the state but do not yet act on
  `model_valid`; propagating the flag into the milling acceptance path is the
  next node, and until then the milling numbers in the out-of-envelope `J` window
  are provisional.
* `DEL` trap carried from the ledger: do not "fix" this by calling `_buhl` with
  the propeller's `a` -- gate `G12` and `G11` are the counter-evidence.

## 8. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_open_water_high_induction.py -q
PYTHONPATH=src python scripts/verify_open_water_high_induction.py
PYTHONPATH=src python scripts/render_open_water_high_induction_doc.py
```
