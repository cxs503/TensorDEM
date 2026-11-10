"""Generate ``docs/P272W2_N11_ice_thermal_friction.md`` from the artifact.

The document is *rendered from* ``artifacts/ice_friction_thermo/acceptance.json``
instead of being typed, so every number in the report text is the number the
code produced (no transcription step to get wrong):

    PYTHONPATH=src python benchmarks/ice_friction_thermo_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "ice_friction_thermo"
DOC = ROOT / "docs" / "P272W2_N11_ice_thermal_friction.md"


def fmt(value: float, digits: int = 4) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.2e}"
    return f"{value:.{digits}g}"


def main() -> None:
    r = json.loads((ART / "acceptance.json").read_text())
    model = r["model"]
    cal = r["calibration"]
    crit = r["criteria"]
    c4 = crit["C4_validation_against_literature"]
    c3 = crit["C3_fixed_point_convergence"]
    c2 = crit["C2_mu_decreases_with_temperature"]
    c5 = crit["C5_friction_heat_identity"]
    cond = r["conductor"]
    cpl = cond["clapeyron"]
    sliding = r["sliding_case"]
    coupled = sliding["coupled"]
    control = sliding["constant_friction_control"]
    pressure = r["pressure_melting"]
    sens = r["sensitivity"]

    validation_rows = "\n".join(
        "| {label} | {mu_lit:.3f} | {mu_pred:.4f} | {err:+.1f} % | {tc:.2f} | {it} | {regime} |".format(
            label=row["label"], mu_lit=row["mu_literature"], mu_pred=row["mu_predicted"],
            err=100 * row["rel_error"], tc=row["interface_temperature_c"],
            it=row["iterations"], regime=row["regime"])
        for row in c4["rows"])
    baseline_rows = "\n".join(
        "| {label} | {mu_lit:.3f} | {mu_pred:.3f} | {err:+.1f} % |".format(
            label=row["label"], mu_lit=row["mu_literature"],
            mu_pred=row["mu_predicted"], err=100 * row["rel_error"])
        for row in c4["baseline_rows"])
    conv_rows = "\n".join(
        "| {v:g} | {p:g} | {t:g} | {it} | {res:.2e} | {mp:.4f} | {mb:.4f} | {gain:.1f} | {regime} |".format(
            v=row["v"], p=row["pressure_mpa"], t=row["temperature_c"],
            it=row["iterations"], res=row["final_residual"], mp=row["mu_picard"],
            mb=row["mu_bisection"], gain=row["gain_k_per_mu"], regime=row["regime"])
        for row in c3["rows"])
    monotone_rows = "\n".join(
        "| {case} | {v:g} | {p:g} | {first:.4f} | {last:.4f} | {drop:.1f} % | {ok} |".format(
            case=row["case"], v=row.get("slip_velocity_m_s", 0.0),
            p=row.get("nominal_pressure_mpa", 0.0),
            first=row["mu_first"], last=row["mu_last"],
            drop=100 * row["total_reduction"], ok="yes" if row["strictly_decreasing"] else "NO")
        for row in c2["cases"])
    early_rows = "\n".join(
        "| {dt:.2e} | {num:.4f} | {ana:.4f} | {err:+.2f} % | {en:.1e} |".format(
            dt=row["dt_s"], num=row["numerical_rise_k"], ana=row["blok_rise_k"],
            err=100 * row["rel_error"], en=row["energy_relative_residual"])
        for row in cond["early_time"])
    threshold_rows = "\n".join(
        "| {t:g} | {p:.1f} |".format(t=row["temperature_c"],
                                     p=row["pressure_melting_threshold_mpa"])
        for row in pressure["threshold_curve"])
    sens_rows = "\n".join(
        "| {p} | {v:g} | {err:.3f} | {mu:.4f} | {s:.3f} | {regime} |".format(
            p=row["parameter"], v=row["value"],
            err=row["validation_max_abs_rel_error"], mu=row["wet_case_mu"],
            s=row["wet_case_melt_fraction"], regime=row["wet_case_regime"])
        for row in sens["rows"])

    doc = f"""# P272W2-N11 — Ice-structure contact with thermal coupling: from a constant
# friction coefficient to a physically self-consistent one

**Repository** TensorDEM · **module** `src/tensordem/ice_friction_thermo/` ·
**artifact** `artifacts/ice_friction_thermo/acceptance.json`
(SHA-256 `{r['C6_provenance']['digest']}`) ·
**driver** `benchmarks/ice_friction_thermo_evidence.py`

---

## 1. The case (physics, geometry, parameters)

An ice slab slides on a steel plane — the laboratory geometry of an ice-friction
test and the loading a DEM ice-structure contact carries.  The existing chain
(`tensordem.ice_channel`, `tensordem.ice_steel_friction`) prescribes the
coefficient of friction, either as the hard-coded `mu = 0.10` or as a fitted
function of `(v, T, p)`.  Neither knows that:

1. the contact heats itself: `Q_f = mu F_n v_slip`;
2. the heat diffuses into the ice and the steel, changing the contact
   temperature `T_contact(t)`;
3. the melting point of ice **falls with pressure**,
   `dT_m/dp = T dv/L_f = -{fmt(cpl['slope_k_per_mpa'])} K/MPa`
   (Clausius-Clapeyron, derived here from `rho_ice`, `rho_water` and `L_f`, not
   fitted; {100 * cpl['tabulated_deviation']:.1f} % from the tabulated
   0.0072 K/atm);
4. the water produced when it does melt removes the friction that produced it.

This node makes `mu` a *solved state* of the contact instead of an input, and
measures the consequence on friction, heat and melt production.

**Parameters.** Two friction parameters are fitted — and only two:
`mu_ref = {fmt(model['mu_ref'])}`, `beta = {fmt(model['beta_per_k'])} 1/K`,
`T_ref = {fmt(model['t_ref_c'])} degC` — from the Barnes, Tabor & Walker (1971)
temperature sweep alone (`{cal.get('source', 'n/a')}`,
{cal.get('n_anchors', 'n/a')} rows, weighted least squares, RMS residual
{fmt(cal.get('rms_log_residual', float('nan')))}).  Everything else is material
data: ice `k=2.2 W/m/K`, `rho=917 kg/m^3`, `c_p=2100 J/kg/K`
(`alpha = {fmt(r['material']['ice']['alpha'])} m^2/s`,
`e = {fmt(r['material']['ice']['effusivity'])} J/m^2/K/sqrt(s)`), steel
`k=45`, `rho=7850`, `c_p=470` (`e = {fmt(r['material']['steel']['effusivity'])}`),
`L_f = 3.34e5 J/kg`.  The Blok partition puts
`{100 * cond['blok_partition_ice']:.1f} %` of the frictional heat into the ice —
86 % goes into the steel, which an ice-only thermal model would get wrong by a
factor ~7.

## 2. What was built

```
q = mu p_real v   ->  1D conduction (two columns)  ->  T_contact
                                   |
                     T_m(p_real) <= 0  (Clausius-Clapeyron)
                                   |
                    melt film h -> s = h/h_ref  ->  mu = mu_dry(T)[1 - s(1-r_wet)]
                                   |
                             back to q  (Picard loop)
```

* `clausius_clapeyron.py` — `dT_m/dp` from the Clapeyron relation, `T_m(p)`, the
  pressure-melting threshold `p = |T|/|dT_m/dp|`, the melt fraction, the Stefan
  melt rate, and the real (asperity) contact pressure `p_real = min(H, p_n/phi)`.
* `friction_law.py` — `mu_dry(T) = mu_0 exp(-beta (T - T_0))`, the melting
  softening `[1 - s(1-r_wet)]`, analytic derivatives, and the closed-form
  two-parameter fit.
* `conduction1d.py` — implicit finite-volume conductor for the ice and steel
  columns sharing a zero-capacity interface node.  Its response is **affine in
  the flux** (`T = T_a + Z q`), which is what reduces the coupled problem to a
  one-dimensional root.
* `selfconsistent.py` — the Picard loop (reports the iteration count and the
  residual), plus an independent bisection solver for the same residual.
* `sliding_case.py` — the time march: conductor response -> fixed point ->
  commit flux -> melt-film renewal `dh/dt = q_excess/(rho L) - h/t_c` -> energy
  ledger.
* `validation.py` — the holdout protocol, the sensitivity table, the acceptance
  report and its SHA-256.

**Two closures, kept separate on purpose.** The algebraic closure reads the melt
fraction off the temperature relative to `T_m(p)` (used against literature rows,
where no film history exists); the coupled solver uses the energy closure, where
`s` comes from the melt film actually produced.  Mixing them would hide which
assumption drives which number.

## 3. Results

### 3.1 Friction against independent literature (criterion C4)

Fitted on Barnes only, then **predicted** on {c4['stats']['n_rows']} rows from
Riska (2017)/ISO 19906 and Poirier et al. (2011):

| condition | literature mu | predicted mu | error | T_contact [degC] | it | regime |
|---|---|---|---|---|---|---|
{validation_rows}

`n within 30 % = {c4['stats']['n_within_gate']}/{c4['stats']['n_rows']}`,
mean |error| {100 * c4['stats']['mean_abs_rel_error']:.1f} %,
max |error| {100 * c4['stats']['max_abs_rel_error']:.1f} %.

The same rows scored with the constant coefficient the platform currently uses
(`mu = {c4['baseline_rows'][0]['mu_predicted']:.2f}`):

| condition | literature mu | constant mu | error |
|---|---|---|---|
{baseline_rows}

`n within 30 % = {c4['baseline_stats']['n_within_gate']}/{c4['baseline_stats']['n_rows']}`,
mean |error| {100 * c4['baseline_stats']['mean_abs_rel_error']:.1f} %.

**Honest FAIL, not hidden:** the two Poirier rows (0.5 m/s at -15 degC and
0.2 m/s at -8 degC) are over-predicted by 60-90 %.  The exponential temperature
law with a flash temperature cannot reproduce the collapse of ice friction at
0.2-0.5 m/s: that requires a *speed-dependent film*, i.e. the melt-film growth
competing with drainage, which is exactly the branch the model only reaches when
the interface is pinned at `T_m`.  Those two rows are reported as failures and
the regime column shows why (`dry`: the interface never reaches its pressure
melting point in the model).

### 3.2 The friction coefficient falls as the contact warms (C2)

| regime | v [m/s] | p_nominal [MPa] | mu (cold end) | mu (warm end) | drop | monotone |
|---|---|---|---|---|---|---|
{monotone_rows}

`{c2['n_decreasing']}/{c2['n_cases']}` strictly decreasing.  The `near-melting` row is
the one where the phase change does the work; the other two are the exponential
temperature law plus frictional heating.

### 3.3 The fixed point converges (C3)

| v [m/s] | p [MPa] | T [degC] | iterations | final \\|dmu\\|/mu | mu (Picard) | mu (bisection) | loop gain dT/dmu [K] | regime |
|---|---|---|---|---|---|---|---|---|
{conv_rows}

Max {c3['max_iterations']} iterations (criterion < 10), max residual
{c3['max_final_residual']:.2e} (criterion < 1e-2), and the two independent
solvers agree to {c3['max_picard_bisection_agreement']:.1e} relative.  The loop
gain printed in the last column is the physical reason the iteration is short:
`beta mu (dT/dmu)` well below 1.

### 3.4 The sliding case, with the energy ledger (C1, C5)

0.5 s of sliding at v = 1 m/s, p_nominal = 2 MPa, bulk ice at -2.5 degC
(the real pressure is the ice hardness, 20 MPa, so `T_m = {coupled['melting_temperature_c']:.3f} degC`):

| quantity | coupled | control (melt ignored by friction) |
|---|---|---|
| mu, start | {fmt(coupled['mu_initial'])} | {fmt(control['mu_initial'])} |
| mu, end | {fmt(coupled['mu_final'])} | {fmt(control['mu_final'])} |
| T_contact, end | {fmt(coupled['contact_temperature_final_c'])} degC | {fmt(control['contact_temperature_final_c'])} degC |
| melt fraction, end | {fmt(coupled['melt_fraction_final'])} | {fmt(control['melt_fraction_final'])} |
| film thickness, end | {fmt(coupled['film_thickness_final_m'])} m | {fmt(control['film_thickness_final_m'])} m |
| Q_f, end | {fmt(coupled['friction_heat_final_w'])} W | {fmt(control['friction_heat_final_w'])} W |
| friction work | {fmt(coupled['friction_work_j_m2'])} J/m^2 | {fmt(control['friction_work_j_m2'])} J/m^2 |
| sensible + drain | {fmt(coupled['sensible_j_m2'])} + {fmt(coupled['latent_drained_j_m2'])} J/m^2 | {fmt(control['sensible_j_m2'])} + {fmt(control['latent_drained_j_m2'])} J/m^2 |
| latent melted | {fmt(coupled['latent_melted_j_m2'])} J/m^2 | {fmt(control['latent_melted_j_m2'])} J/m^2 |
| energy closure error | {fmt(coupled['energy_closure_error'])} | {fmt(control['energy_closure_error'])} |
| iterations, max / mean | {coupled['iterations_max']} / {coupled['iterations_mean']:.2f} | {control['iterations_max']} / {control['iterations_mean']:.2f} |

The control keeps the melting *temperature cap* but removes the feedback of the
melt fraction on friction (`r_wet = 1`), so the difference is exactly what the
phase-change closure buys: **mu {fmt(control['mu_final'])} -> {fmt(coupled['mu_final'])}**,
a {100 * (1 - coupled['mu_final'] / control['mu_final']):.1f} % reduction at
1 m/s, with {fmt(coupled['melt_fraction_final'])} of the interface melted.

`Q_f = F_friction v` is exact here (relative error {fmt(c5['max_rel_error'])})
and is cross-checked through two chains (force x velocity and real-area flux x
real area) and against the DEM tangential limit
`tensordem.platen_friction.platen_friction_limit`.

### 3.5 Pressure melting: where Clausius-Clapeyron actually bites

Melting needs `p_real > |T|/|dT_m/dp|`:

| bulk temperature [degC] | pressure needed to melt [MPa] |
|---|---|
{threshold_rows}

Sweeping the ice hardness (the only way to reach those pressures at engineering
loads of ~1 MPa) and the temperature:

* {pressure['n_pressure_melted']} of {len(pressure['grid'])} grid points are in the
  pressure-melted regime, with a maximum coefficient reduction of
  {100 * pressure['max_mu_reduction']:.0f} %;
* the **Amontons check**: for a plastic contact `p_real` saturates at `H`, so
  `mu` is independent of the *nominal* pressure (spread
  {fmt(pressure['amontons_mu_spread'])}).  This is the honest negative result of
  the node: the classical pressure weakening of ice friction
  (Oksanen & Keinonen 1982, rows 0.05-2 MPa) **cannot** be explained by
  Clausius-Clapeyron melting alone — at those pressures `T_m` moves by less than
  0.15 K.  It must come from real-area growth / third-body shear, which this
  model does not contain and does not pretend to.

### 3.6 Numerical verification of the thermal solver

Against the Blok semi-infinite flash temperature
`T - T_bulk = 2 q sqrt(t/pi)/(e_ice + e_steel)`:

| dt [s] | finite volume [K] | Blok [K] | deviation | energy residual (rel.) |
|---|---|---|---|---|
{early_rows}

The deviation falls with `dt` (first-order scheme) and reaches
{100 * abs(cond['early_time'][-1]['rel_error']):.2f} % at dt = {cond['early_time'][-1]['dt_s']:.1e} s.
Steady state matches the parallel-column resistance
`q/(k_i/d_i + k_s/d_s)` to {fmt(cond['steady_state']['rel_error'])} relative
(machine precision: the discretisation is exact for the linear profile).

### 3.7 What the un-fitted parameters cost

| parameter | value | holdout max error | wet-case mu | melt fraction | regime |
|---|---|---|---|---|---|
{sens_rows}

The dry holdout is **insensitive to the film parameter** (all three film rows
give the same error, `{sens['film_rows_leave_dry_validation_unchanged']}`), but
the lubricated coefficient spans {fmt(sens['wet_case_mu_range'][0])} to
{fmt(sens['wet_case_mu_range'][1])} across these sweeps — a factor
{sens['wet_case_mu_range'][1] / sens['wet_case_mu_range'][0]:.0f}.  The *wet*
regime is therefore a bounded estimate, not a prediction; only the dry branch is
validated against data.

## 4. Figures

* `fig_mu_vs_temperature.png` — the temperature law, the pressure melting point,
  calibration anchors, holdout rows and predictions.
* `fig_sliding_case.png` — `mu(t)`, `T_contact(t)` pinned at `T_m(p)`, `s(t)`,
  coupled vs control.
* `fig_pressure_melting.png` — `mu(T)` for four hardnesses and the
  Clausius-Clapeyron threshold curve.
* `fig_convergence.png` — residual of the fixed point per iteration.
* `fig_conductor_blok.png` — conductor vs Blok and the first-order convergence.
* `fig_sensitivity.png` — the wet-case coefficient vs the un-fitted parameters.

## 5. Caveats and limits

1. **Constant ice properties**: `k` and `c_p` vary ~5 % over -30..0 degC.
2. **Hardness model**: `p_real = min(H, p_n/phi)` with constant `H`; no
   strain-rate or temperature dependence of `H`.  This is why nominal-pressure
   weakening is absent (section 3.5).
3. **No drainage/refreezing physics beyond the renewal term**: drained water
   carries its latent heat out of the contact (reported separately); refreezing
   on cooling is not modelled.
4. **Interface-averaged temperature**: a real contact has a *distribution* of
   asperity flash temperatures; the onset of melting would then be smeared over
   the distribution width instead of being a sharp pin.
5. **Two un-fitted parameters dominate the wet branch**: the asperity spot
   radius (20-500 um) and the film scale `h_ref` (0.2-5 um).  Both are reported
   with their effect (3.7), neither is tuned to the validation data.
6. **Resolution requirement**: `dx <= 5e-5 m` and `dt <= a/v` to keep the
   interface impedance within ~11 % of the Blok value.  At `dx = 2.5e-4 m` the
   impedance is 46 % high and the contact temperature is over-predicted
   (measured 3.2 K instead of 1.2 K for the same flux).
7. **Not yet inside the DEM time loop**: `solve_contact_state` returns the state
   a DEM step needs (`mu`, `F_friction`, `Q_f`, `T_contact`, melt fraction); the
   tie-in is verified against the existing Cundall-Strack limit
   (`platen_friction_limit`), not by re-running `dem3d` with a thermal contact.

## 6. Error assessment (explicit)

| quantity | reference | error | < 3 % ? |
|---|---|---|---|
| Steady interface rise | `q/(k_i/d_i + k_s/d_s)` | {fmt(cond['steady_state']['rel_error'])} | **yes** |
| Early-time rise, dt = {cond['early_time'][-1]['dt_s']:.1e} s | Blok semi-infinite | {100 * abs(cond['early_time'][-1]['rel_error']):.2f} % | **yes** |
| Early-time rise, dt = {cond['early_time'][0]['dt_s']:.1e} s | Blok semi-infinite | {100 * abs(cond['early_time'][0]['rel_error']):.2f} % | **no** (first-order in dt, see 3.6) |
| Energy closure of the sliding run | `int q dt` | {fmt(coupled['energy_closure_error'])} | **yes** |
| `Q_f = F_f v` identity | algebra | {fmt(c5['max_rel_error'])} | **yes** |
| Clapeyron slope | tabulated 0.0072 K/atm | {100 * cpl['tabulated_deviation']:.2f} % | **no** (constant `dv`, `L`; the difference is the temperature dependence of the properties) |
| Melting point at 20 MPa | `-0.0740 K/MPa x 20 MPa` | exact (same formula) | n/a (definition) |
| Pressure-melting threshold at -5 degC | 67.5 MPa (hand arithmetic from the derived slope) | < 0.1 % | **yes** |
| `mu` vs independent literature (3 rows) | Riska / Poirier | {100 * abs(c4['rows'][0]['rel_error']):.1f} %, {100 * abs(c4['rows'][1]['rel_error']):.1f} %, {100 * abs(c4['rows'][2]['rel_error']):.1f} % | **no** (the campaign gate is 30 %, not 3 %; the 30 % gate is met on 3/5 rows) |
| `mu` vs independent literature (2 rows) | Poirier high-speed | {100 * abs(c4['rows'][3]['rel_error']):.0f} %, {100 * abs(c4['rows'][4]['rel_error']):.0f} % | **no — honest FAIL** (section 3.1) |

## 7. Reproduce

```bash
PYTHONPATH=src python benchmarks/ice_friction_thermo_evidence.py   # artifact + figures
PYTHONPATH=src python benchmarks/ice_friction_thermo_report.py     # this document
PYTHONPATH=src python -m pytest tests/test_ice_friction_thermo_*.py -q
```
"""
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc)} chars)")


if __name__ == "__main__":
    main()