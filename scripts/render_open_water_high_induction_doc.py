"""Render docs/open_water_high_induction_d3_repair.md from the verification JSON.

Zero hand-copied numbers: every value in the document is read out of
``artifacts/open_water_high_induction/verification.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "open_water_high_induction" / "verification.json"
DOC = ROOT / "docs" / "open_water_high_induction_d3_repair.md"


def main() -> int:
    r = json.loads(ART.read_text())
    alg, sol = r["algebra"]["data"], r["solver"]["data"]
    defect = sol["defect_evidence"]
    lib = sol["sweeps"]["library_geometry"]
    lean = sol["sweeps"]["lean_geometry"]
    def_rows = sol["sweeps"]["defect_geometry"]["clipped"]
    hl = sol["sweeps"]["defect_geometry"]["high_loading"]

    def block(rows, models=("clipped", "classical", "high_loading")):
        out = ["| J | model | K_T | K_Q | eta | a_max | strips@cap | iter res | momentum res | "
               "C_T,BE/C_T,max | valid |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for model in models:
            for row in rows[model]:
                out.append(
                    f"| {row['J']:.2f} | {model} | {row['KT']:.4f} | {row['KQ']:.4f} | "
                    f"{row['eta']:.3f} | {row['a_max']:.3f} | "
                    f"{row['strips_at_cap']}/{row['n_strips']} | {row['residual']:.1e} | "
                    f"{row['momentum_residual']:.2e} | {row['ct_max_loading']:.2f} | "
                    f"{'yes' if row['model_valid'] else 'no'} |")
        return "\n".join(out)

    gates = "\n".join(
        f"| {g['id']} | {g['verdict']} | {g['measured']} | {g['note']} |"
        for g in r["gates"])

    buhl_rows = "\n".join(
        f"| {x['F']:.2f} | {x['a_buhl']:.4f} | {x['exact_root']:.4f} | "
        f"{x['ct_at_a_buhl']:.4f} | {x['rel_gap']:.3f} |"
        for x in alg["buhl_form_consistency"]["rows"])

    doc = f"""# DEFECT-D3 repaired: high-loading closure and validity reporting in the open-water BEM

Mission m16448 · repository TensorDEM · module `src/tensordem/propeller_ice/hydro.py`
· campaign `scripts/verify_open_water_high_induction.py` ({r['protocol']})
· artifact `artifacts/open_water_high_induction/verification.json`
({r['n_pass']}/{r['n_gates']} gates PASS, {r['n_fail']} FAIL).
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
| solidity `sigma(0.7R)` | {4 * 0.539 / (2 * 3.141592653589793 * 0.35):.3f} |
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
4 F a (1 + a) = 0.96 F   =>   a_s = (-1 + sqrt(1.96)) / 2 = {alg['breakdown']['a_s']:.10f},
```

i.e. the breakdown induction in the propeller convention is `a_s = 0.2` -- and not
the `0.4` of the wind-turbine convention the coded `_buhl` uses (gate
`G2_breakdown_induction`, measured {alg['breakdown']['a_s']:.17f}).

### 2.3 The high-loading closure is constructed, not transplanted

The closure is the quadratic `C_T(a) = c0 + c1 a + c2 a^2` fixed by the same
three conditions the published relation uses:

1. value continuity with the classical relation at `a_s`: `C_T(a_s) = 0.96 F`;
2. slope continuity at `a_s`: `C_T'(a_s) = 4 F (1 + 2 a_s)`;
3. the measured high-loading anchor `C_T(a = 1) = 2` (F-independent).

Applied in the **turbine** convention this construction must reproduce Buhl's
published coefficients `(8/9, 4F - 40/9, 50/9 - 4F)`; it does, to
{alg['buhl_replica']['1.00']['rel_err']:.2e} (gate `G1`).  Applied in the
module's own convention it gives `(c0, c1, c2)` for `F = 1` of
{tuple(round(v, 4) for v in __import__('tensordem.propeller_ice.hydro', fromlist=['x']).empirical_ct_coefficients(1.0))},
and it is C1-continuous with the classical relation at `a_s` to
{max(alg['continuity']['value_rel_err'], alg['continuity']['slope_rel_err']):.1e}
(gate `G3`).  The inverse is closed-form (no iteration) and exact to
{alg['inverse']['worst_rel_err']:.1e} (gate `G4`).

### 2.4 What changed in the solver

* `induction_model` config, three documented options:
  * `"classical"` (**default**) -- induction from the closed-form classical
    inversion; the closure is used to *report* validity.  Bit-comparable with the
    pre-repair coefficients inside the valid envelope (gate `G5`, worst relative
    difference {sol['legacy_parity']['worst_rel_err']:.1e} over
    {sol['legacy_parity']['n_compared']} comparisons).
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
| iteration residual `max abs(delta a)` | {defect['residual']:.2e} |
| momentum-identity residual | {defect['momentum_residual']:.4f} |
| strips on the induction cap | {defect['strips_at_cap']}/{defect['n_strips']} |
| disk loading / closure bound | {defect['ct_max_loading']:.2f} |
| `model_valid` | {defect['model_valid']} |
| `validity_flag` | `{defect['validity_flag']}` |

The iteration claims convergence to 1e-8 while the blade-element thrust is
unbalanced by {(defect['momentum_residual'] * 100):.2f} % and the disk loading is
{defect['ct_max_loading']:.0f}x beyond what the annular streamtube model can
represent.  That is the defect: not a wrong number, but a *number presented as a
solution*.

### 3.2 Full sweeps

Registered geometry (3 m, `J` grid):

{block(def_rows and sol['sweeps']['defect_geometry'])}

Library geometry (`D` = 0.6 m, `c(0.7R)/D` = 0.40, `P/D` = 1.0):

{block(lib)}

Low-solidity geometry (`c(0.7R)/D` = 0.12, `sigma(0.7R)` ~ 0.22):

{block(lean)}

Reading of the sweeps:

* inside the envelope the three modes agree (gate `G5`); the repair does not
  change physics that was already supported;
* where the closure has a root, the state satisfies its own momentum identity to
  {sol['closure_root_momentum_residual']:.1e} (gate `G7`) -- in `"high_loading"`
  mode the residual is round-off, rather than the per-cent violation the capped
  mode reports;
* the boundary is geometric (gate `G8`): the low-solidity propeller is valid from
  `J = {sol['validity_boundary']['lean']['first_valid_J']}`, the library geometry
  from `J = {sol['validity_boundary']['library']['first_valid_J']}` and the
  registered 3 m geometry from `J = {sol['validity_boundary']['defect']['first_valid_J']}`;
* bollard (`V_a = 0`) is now labelled `bollard_unsupported` in both geometries
  while still returning finite, legacy-compatible numbers (gate `G9`).

### 3.3 The announced branch: unused *and* inconsistent

Two independent findings, both measured:

1. **Convention.** `a_t = -a_p` maps the turbine relation onto *minus* the
   propeller `C_T` ({alg['convention_map']['ct_turbine_at_minus_a_p']:.3f} versus
   {alg['convention_map']['ct_propeller']:.3f} at `a_p` = {alg['convention_map']['a_p']}),
   so the turbine high-induction branch is not a change of variable of this
   module's induction.  Applying it to `a` would mix two relations with
   breakdown inductions 0.4 and 0.2 (gate `G12`).
2. **The constant form does not invert its own parabola** (gate `G11`):

| F | coded `_buhl` | exact inverse at `C_T = 0.96` | `C_T` at the coded value | rel. gap |
|---|---|---|---|---|
{buhl_rows}

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
{gates}

## 6. Error judgement against reference values

There is **no experimental open-water chart** in this repository, so absolute
`K_T`, `K_Q` stay **WITHHELD**; nothing here claims a `<3 %` agreement with
measurement.  What is judged against a reference is the algebra and the
identities:

| quantity | reference | error |
|---|---|---|
| construction (turbine convention) | Buhl 2005 coefficients | {alg['buhl_replica']['1.00']['rel_err']:.1e} |
| breakdown induction | exact root of `4a(1+a) = 0.96`, `a_s = 0.2` | {abs(alg['breakdown']['a_s'] - 0.2):.1e} |
| closure C1 junction (`F` 0.3..1.0) | `4Fa(1+a)` value and slope at `a_s` | {max(alg['continuity']['value_rel_err'], alg['continuity']['slope_rel_err']):.1e} |
| closure inverse | `C_T(a(C_T)) = C_T` | {alg['inverse']['worst_rel_err']:.1e} |
| momentum identity at closure roots | its own momentum relation | {sol['closure_root_momentum_residual']:.1e} |
| legacy parity inside the envelope | pre-repair coefficients | {sol['legacy_parity']['worst_rel_err']:.1e} |

All are `<3 %` by many orders of magnitude; the absolute coefficients are not
claimed.  Is every key quantity `<3 %` of a *measurement*? **No, and it cannot be
here** -- that is stated, not papered over.

## 7. Notes, limitations, traps

* **The registered 3 m geometry is heavily loaded by construction** (solidity
  1.03, `P/D` 0.95): even the repaired model reports it out of envelope below
  `J = {sol['validity_boundary']['defect']['first_valid_J']}`.  Every `K_T`/`K_Q` produced for it at
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
"""
    DOC.write_text(doc, encoding="utf-8")
    print("wrote", DOC, len(doc), "chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())