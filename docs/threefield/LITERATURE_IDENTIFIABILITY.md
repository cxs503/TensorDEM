# Level-ice resistance baselines: identifiability against the published MT Uikku predictions

Protocol `tensordem-threefield-literature-identifiability-v1`. Artifacts: `artifacts/threefield/literature_identifiability.json`
(regenerate with the command in section 7). Figures:
`artifacts/threefield/figures/literature_as_shipped_errors.png`,
`artifacts/threefield/figures/literature_speed_signature.png`.

## 1. What question this answers

`docs/threefield/FORMULAS.md` grades literature agreement of the shipped
Lindqvist/Riska baselines as **WITHHELD**: the component structure follows the
published models, but the coefficient set is a convention written in this
repository, and the primary papers were not reachable from the build sandbox. The
two baselines are nevertheless used as the *reference gate* for the DEM ice
resistance work, so "unverified" is not a comfortable resting state: a reader
cannot tell whether the gate is literature-anchored or an arbitrary curve set.

The primary papers are still unavailable (checked again: `10.1016/j.ijnaoe.2016.01.004`
resolves through Crossref as CC BY-NC-ND, and both the Elsevier API and the
ScienceDirect full-text endpoints refuse anonymous access with HTTP 400 / 403).
What *is* available is better than the paper text for this purpose: the
repository already transcribes ten MT Uikku level-ice cases **together with the
published Lindqvist and Riska predictions in kN**
(`benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv`, from Hu and Zhou,
*Int. J. Naval Archit. Ocean Eng.* 8(2):169-176, 2016). Published numbers computed
from the real formulas are exactly what a paraphrase must be tested against, so
literature agreement becomes a computable question with three parts:

| # | question | why it is decisive |
|---|---|---|
| Q1 | Is the published Riska column structurally `R = C1 + C2 V`? | if yes, its speed slope is readable from the data without the hull, and `C2/C1` becomes a hull-insensitive signature |
| Q2 | Can the shipped chain reproduce the published columns at *any* admissible hull? | the hull enters only through `(L, B, alpha, mu, phi)`; if the best point of a physically bounded envelope still misses the 15 % gate, no "different ship" argument rescues the coefficients |
| Q3 | How far apart are the speed signatures? | `C2/C1` is nearly independent of `B`, so this comparison cannot be dismissed as a geometry choice |

Fixture: the ten level-ice cases of the source table (0.63-1.04 m thickness,
0.2-1.0 m/s drift speed, 540-1029 kPa flexural strength). Reported forces in kN;
all computations in SI.

## 2. Method

1. **Data gate** (`load_cases`) - the CSV is parsed strictly: ten cases, `ice_type`
   must be `level`, SI conversion applied at load time, missing source entries kept
   as missing rather than zero-filled.
2. **Linearity probe** (`published_riska_linearity`) - the published Riska column is
   fitted with `C1 + C2 V` inside each constant-thickness group; the residual is
   compared with the 1 kN rounding floor of the source table.
3. **As-shipped probe** (`default_hull_errors`) - the shipped chain is evaluated at
   the package default hull `L=120 m, B=20 m, alpha=22 deg, phi=30 deg, mu=0.05`
   (the configuration the existing 22 structural tests pin) and compared case by
   case with the published column.
4. **Bounded best-fit probe** (`best_feasible_fit`) - the objective is
   `max_i |error_i|`, *not* an aggregate: a gate that screens on MAPE can hide one
   bad case. The search is a seeded random scan followed by coordinate refinement
   over the declared envelope, so it is deterministic and re-runnable.
5. **Signature probe** (`shipped_speed_signature`, `max_feasible_speed_signature`) -
   the shipped Riska law is affine in `V` by construction, so its `C1, C2` are
   recovered exactly from two evaluations; the largest achievable `C2/C1` inside the
   envelope is then compared with the published value.

The envelope is the only assumption in the audit and is declared as
`CONVENTION`: L in [120, 260] m, B in [18, 46] m,
B/L in [0.09, 0.26], waterline entry angle in [10, 36] deg,
stem angle in [15, 48] deg, steel-ice friction in [0.02, 0.15],
draught 8 m. The bounds are wide enough to contain any icebreaking tanker of the
MT Uikku class, and narrow enough that a fit cannot hide in unphysical geometry.

## 3. Q1 - the published Riska column is exactly speed-affine

| sheet h [m] | tests | C1 [kN] | C2 [kN/(m/s)] | C2/C1 [s/m] | max abs residual [N] | flexural spread in sheet [kPa] |
|---|---:|---:|---:|---:|---:|---:|
| 0.63 | 401-403 | 454 | 93.8 | 0.206 | 81.6 | 221 |
| 1.04 | 301-303 | 859 | 166.2 | 0.194 | 81.6 | 129 |

The residual is **81.6 N**, i.e. 8 % of the 1 kN rounding floor of the source
table: the published Riska predictions are Riska's speed-linear model to
machine precision. This is `REAL` evidence, and it is the load-bearing fact for
Q3 - the ratio `C2/C1` of the published chain is 0.206 / 0.194 s/m.

**By-product (a data-provenance caveat).** Inside a sheet the flexural strength
spread across tests is 221 / 129 kPa (15-27 % of the mean), yet a single `C1`
reproduces all three published Riska values to the rounding floor. The published
formula columns are therefore computed per ice *sheet* with one property set, not
per test with the measured values. A DEM-versus-formula comparison that feeds the
per-test `sigma_f, sigma_c` columns is not comparing like with like; this is now
asserted by the `test_published_formula_columns_ignore_the_per_test_strength_spread` check in
`tests/test_threefield_resistance_literature_check.py`.

## 4. Q2 - as-shipped, and at the best admissible hull

Signed error, shipped chain minus published prediction (positive = over-prediction):

| test | h [m] | V [m/s] | published Lindqvist [kN] | shipped Lindqvist error | published Riska [kN] | shipped Riska error |
|---|---:|---:|---:|---:|---:|---:|
| 103 | 0.77 | 0.2 | 320 | +13.9 % | 610 | -3.6 % |
| 104 | 0.76 | 0.5 | 380 | +10.1 % | 630 | +8.2 % |
| 205 | 0.96 | 0.2 | 510 | +16.4 % | 800 | +19.3 % |
| 206 | 0.95 | 0.5 | 560 | +4.9 % | 840 | +13.8 % |
| 301 | 1.04 | 0.2 | 426 | +96.5 % | 892 | +52.0 % |
| 302 | 1.04 | 0.5 | 538 | +57.5 % | 942 | +46.3 % |
| 303 | 1.04 | 1.0 | 572 | +45.1 % | 1025 | +32.3 % |
| 401 | 0.63 | 0.2 | 251 | +80.8 % | 473 | +55.5 % |
| 402 | 0.63 | 0.5 | 322 | +80.6 % | 501 | +89.4 % |
| 403 | 0.63 | 1.0 | 351 | +46.4 % | 548 | +53.9 % |

| probe | Lindqvist max abs | Lindqvist MAPE | Lindqvist bias | Riska max abs | Riska MAPE | Riska bias |
|---|---:|---:|---:|---:|---:|---:|
| as shipped (default hull) | 96.5 % | 45.2 % | +45.2 % | 89.4 % | 37.4 % | +36.7 % |
| best hull inside the envelope | 27.6 % | 15.6 % | -6.6 % | 16.7 % | 8.7 % | +1.8 % |

Best-fit hulls: Lindqvist needs `L=143 m, B=22 m, alpha=10.9 deg, mu=0.033`;
Riska needs `L=250 m, B=42 m, alpha=10.5 deg, phi=18.9 deg, mu=0.040`.
Both are *inside* the declared envelope, and both are pushed to its edge -
small entry angle, extreme length/beam - which is the signature of a fit that is
compensating for a wrong coefficient rather than describing the ship.

Even there, the worst case stays at **27.6 %** (Lindqvist) and **16.7 %**
(Riska). The envelope cannot be widened honestly beyond those bounds for a
single-screw icebreaking tanker, so the conclusion is stated as it is:
**no admissible geometry reproduces the published columns within the 15 % gate.**

## 5. Q3 - the decisive, hull-insensitive finding

| quantity | value |
|---|---:|
| published relative speed dependence `C2/C1` (mean of two sheets) | 0.200 s/m |
| shipped chain at the default hull | 0.058 s/m |
| gap factor (published / maximum reachable) | **3.44 x** |

`C2` of the shipped chain is dominated by `B * h^{3/2}` terms while its `C1`
contains the beam-free crushing term, so the ratio depends on the hull only
weakly; maximizing it over the whole envelope still lands **3.44 x below the
published value**. The shipped Riska baseline is therefore not merely
mis-calibrated in magnitude - its *speed structure* is different from the
published model: the published column grows by 20 % of
its static part between V = 0 and V = 1 m/s, while the shipped chain grows by at
most 6 %.

## 6. Verdict, and what this does *not* say

| claim | evidence grade | evidence |
|---|---|---|
| published Riska column is `C1 + C2 V` | REAL | residual {:.1f} N vs 1 kN rounding floor; `test_published_riska_column_is_affine_in_speed` |
| shipped chain numbers in this document | REAL | generated by `run_audit`, no hand-typed digits |
| shipped chain vs published, as shipped | REAL | `96.5 %` / `89.4 %` worst case; two columns over-predicted on all ten cases |
| shipped chain vs published, best admissible hull | REAL | `27.6 %` / `16.7 %` worst case, gate missed |
| literature agreement of the shipped coefficient set | **REFUTED-IN-ENVELOPE** | replaces the previous `WITHHELD` grade with a quantified negative |
| speed-term structure | **REFUTED** | `3.44 x` shortfall, hull-insensitive |
| agreement with the *text* of Lindqvist (1989) / Riska (1989) | WITHHELD | primary papers not retrievable (Crossref resolves, full text 400/403) |

### Error judgement, item by item (the question asked explicitly)

| key quantity | reference | shipped error | < 3 % ? |
|---|---|---:|---|
| Lindqvist total, worst case, default hull | published Lindqvist column | 96.5 % | **NO** |
| Lindqvist total, mean bias, default hull | published Lindqvist column | +45.2 % | **NO** |
| Riska total, worst case, default hull | published Riska column | 89.4 % | **NO** |
| Lindqvist total, worst case, best hull in envelope | published Lindqvist column | 27.6 % | **NO** |
| Riska total, worst case, best hull in envelope | published Riska column | 16.7 % | **NO** |
| Riska speed slope `C2/C1` | published affine fit (`REAL`, residual < rounding floor) | 71 % of the reachable maximum remains below the published value | **NO** |
| published Riska linearity residual | `C1 + C2 V` | 81.6 N = 8 % of the 1 kN rounding floor | **YES** (measurement floor, not a model error) |

Consequences for downstream use, stated plainly:

* The 46-48 % band spread reported in `docs/threefield/FORMULAS.md` is **not** a
  literature uncertainty band. Both edges of the band come from the same
  convention set, and both sit 5-97 % above the published predictions at the
  default hull. A DEM result "inside the band" therefore carries no
  literature-level information.
* Any DEM-versus-baseline error must be quoted against the published predictions
  of the corresponding sheet, or the report must state that the comparison uses
  the in-repo convention chain and is a code-versus-code comparison rather than
  code-versus-literature.
* This audit does **not** say the shipped coefficients are wrong *as a convention*
  - the 22 structural tests in `tests/test_threefield_resistance.py` still pin
  their internal consistency. It says they are not a reproduction of the
  published Lindqvist/Riska chains, so the word "Lindqvist" in their names is a
  family name, not an attribution of coefficient provenance.

## 7. Reproducing

```bash
PYTHONPATH=src python -m pytest tests/test_threefield_resistance_literature_check.py -q   # 10 checks
PYTHONPATH=src python scripts/audit_threefield_resistance_literature.py                   # JSON + 2 figures
```

The audit is deterministic (seeded) and runs in about 2 s on one core, so it is a
unit-test-grade instrument rather than a campaign.

## 8. Limitations of the audit itself

* The envelope is an assumption (`CONVENTION`). A radically different vessel -
  a wide-beam drillship or a shallow-draught river icebreaker - is outside it,
  and the bounded search does not claim to have excluded every conceivable hull.
* The published predictions are rounded to 1 kN and are not the paper's raw force
  time series; only the resistance summary is compared.
* The published columns are sheet-averaged (section 3), so the per-test strength
  columns of the CSV are *not* the properties behind the published numbers; the
  audit uses the per-test values as the ice state of the shipped chain, which is
  the configuration the repository itself uses. This makes the comparison
  conservative rather than favourable, and it is why the by-product caveat is
  filed as a provenance finding instead of being smoothed away.
* No experimental validation is performed here; the earlier
  `benchmarks/dem3d/ship_ice_validation/README.md` protocol still governs that
  question and still marks the DEM-versus-Uikku target as not yet comparable.
