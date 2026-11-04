# Level-ice resistance baselines: derivation and verification ledger

Protocol id: `tensordem-threefield-resistance-audit-v1`

Scope: `src/tensordem/threefield/resistance.py`, the two engineering baselines
used as reference gates for the DEM coupling study
(`docs/threefield/README.md`). The module's own provenance caveat cites this
document and the guard suite `tests/test_threefield_resistance.py`; both were
missing until this audit, so the caveat previously pointed at nothing.

The audit answers exactly one question: **which parts of this module are
independently verified, and which are conventions that must not be cited as
physics?** It does not, and cannot, verify agreement with the published primary
papers, because those papers are not reachable from this offline sandbox.

## 1. What the module computes

Volumetric force decomposition, all terms in newtons, summed without any
per-metre convention:

```
R_lindqvist(V) = R_crush + R_bend + R_sub + 0.5 rho_w B h V^2
R_riska(V)     = C1 + C2 V
C1 = (1 + 0.021 phi)(R_crush + R_sub) + R_bend
C2 = (1 + 0.063 phi)[ 0.5 mu rho_w B h sqrt(g h) + sigma_f B h^2 / (3 L sqrt(g h)) ]

R_crush = 0.5 sigma_c h^2 (tan(alpha) + mu / cos(alpha))
R_bend  = 0.25 sigma_f B h^2 / L
R_sub   = rho_w g h^2 B (0.9 + 0.05 mu L / B)
```

Symbols: `h` ice thickness, `B` beam, `L` length, `alpha` waterline entry angle,
`phi` stem angle **in degrees**, `mu` hull-ice friction coefficient,
`sigma_c` / `sigma_f` compressive / flexural ice strength, `rho_w` water
density, `g` gravity.

### Units

Every component is a force. The earlier documentation labelled `R_crush` and
`R_bend` as `[N/m]` ("per unit width"); that label is wrong and has been
corrected, because the three components are added directly and the total is a
total resistance, not a per-metre load. Dimensional exponents as measured by the
guard suite:

| quantity | `sigma_c` | `sigma_f` | `h` | `B` | `L` | `rho_w` | `g` |
|---|---|---|---|---|---|---|---|
| `R_crush` | 1 | – | 2 | 0 | 0 | – | – |
| `R_bend` | – | 1 | 2 | 1 | -1 | – | – |
| `R_sub` (at `mu=0`) | – | – | 2 | 1 | 0 | 1 | 1 |
| total | – | – | – | – | – | – | – |

The exponents are measured numerically by two-point log-log scaling, so a
reintroduced per-width component fails the suite immediately.

## 2. Coefficient ledger

Each coefficient is graded by how it can be justified offline.

| constant | value | status | justification |
|---|---|---|---|
| `CRUSHING_EFFICIENCY` | 0.5 | **DERIVED** | Mean of a triangular contact-pressure profile: pressure rises linearly from zero at the upper ice surface to `sigma_c` at the lower surface, so `(1/h) int_0^h sigma_c (y/h) dy = sigma_c/2`. The suite recomputes this mean by quadrature, so `0.5` is not a free parameter. A uniform-pressure assumption would give 1.0 and double the crushing force. |
| crushing bracket | `tan(alpha) + mu/cos(alpha)` | **DERIVED** | Normal-plus-friction resolution on the inclined bow face: the friction contribution carries `1/cos(alpha)` (friction acting along the face projected onto the advance direction), not `mu cos(alpha)`. The two conventions differ by >1 % at a 22 deg entry angle, and the suite asserts the implemented one matches `mu/cos(alpha)` and is measurably different from the other. |
| `BENDING_EFFICIENCY` | 0.25 | **CONVENTION** | Read as a root bending moment `sigma_f B h^2 / 6` with an effective lever arm `L/1.5`, i.e. 1.5x the pure cantilever factor `1/6`. It is *not* derived here and not matched to any published coefficient. Treated as a pinned implementation constant. |
| `SUBMERSION_BUOYANCY_COEFFICIENT` | 0.9 | **CONVENTION** | Dimensionless channel-opening coefficient: the cleared channel floats a displaced ice volume `0.9 B h^2`. Physically plausible order one, no independent derivation offline. |
| `SUBMERSION_FRICTION_COEFFICIENT` | 0.05 | **CONVENTION** | Scales the friction of the displaced rubble sheet over the hull length, `0.05 mu L / B`. Pinned; the exact partial derivative `dR_sub/dL = 0.05 mu rho_w g h^2` is asserted, which fixes the coefficient's role even though its value is a choice. |
| `0.021`, `0.063` (stem correction) | – | **CONVENTION** | Linear stem-angle corrections of the ice-class type, applied to `(R_crush + R_sub)` and to `C2` respectively. The suite pins both: `dC1/dphi = 0.021 (R_crush + R_sub)` **per degree**, and the two `(1 + 0.0xx phi)` factors are recovered from the C2 identities below. |
| `0.5 rho_w B h V^2` (added drag) | – | **DERIVED** | Dynamic-pressure drag on the displaced sheet area `B h`, drag coefficient 0.5. Asserted to be purely quadratic in `V` with no linear part. |
| C2 friction term `0.5 mu rho_w B h sqrt(g h)` | – | **DERIVED (internally)** | At the shallow-water celerity `V = sqrt(g h)` it evaluates to `0.5 mu rho_w g B h^2`, i.e. half the hydrostatic friction load `mu rho_w g B h^2` of the opened channel. The suite asserts the ratio `= 0.5 (1 + 0.063 phi)`, so the stem factor and the 0.5 are both pinned. |
| C2 bending term `sigma_f B h^2/(3 L sqrt(g h))` | – | **DERIVED (internally)** | At `V = sqrt(g h)` it becomes `sigma_f B h^2/(3L) = (4/3) R_bend`. The suite asserts the ratio `= (4/3)(1 + 0.063 phi)`. |
| literature agreement of the whole set | – | **WITHHELD** | Lindqvist (1989) and Riska (1989) are not available offline. No statement in this repository claims the numbers reproduce the published predictions, and none should be inferred. The baselines are a *band*, not a calibration target. |

"DERIVED" grades a closed-form derivation from stated physical assumptions, not
agreement with published coefficients. "CONVENTION" means the value is a pinned
implementation choice. "WITHHELD" means the evidence needed to grade it is
unavailable.

## 3. Reference-speed reading of the Riska coefficients

The two C2 terms only make dimensional sense as *per-unit-speed* coefficients,
and their physical content becomes visible when the speed is expressed in units
of the shallow-water celerity `sqrt(g h)`, which is the natural speed scale of the
broken-ice channel:

```
C2_friction * sqrt(g h) = 0.5  (1 + 0.063 phi) * mu rho_w g B h^2
C2_bending  * sqrt(g h) = 4/3  (1 + 0.063 phi) * R_bend
```

So at one celerity the friction term is half the hydrostatic channel friction and
the bending term is 4/3 of the static bending resistance. Both identities are
asserted over `h = 0.30 … 1.00 m`. They are the strongest internal checks
available, and they are what makes the Riska baseline more than a curve fit: a
sign, factor or unit slip in C1/C2 breaks at least one of them.

## 4. Inert parameters

`HullGeometry.buttock_angle_deg` and `IceProperties.elastic_modulus_pa` are
validated by the dataclasses but enter **neither** baseline. `stem_angle_deg`
enters the Riska correction only. The suite asserts this inertness, so making any
of them active in future is a visible test failure rather than a silent change.

## 5. Speed dependence

`lindqvist_resistance` carries an explicit quadratic added-drag term, so the
"nearly speed independent" wording in its docstring must be read narrowly: at
`V = 1 m/s` the added drag is about 3 % of the static components and grows
quadratically (about 12 % at 2 m/s). The suite asserts that at 2 m/s the added
drag exceeds 5 % of the static part, i.e. that the speed term is not negligible
at operational ice speeds.

## 6. Reference gate width

Reproduction at `V = 1.0 m/s`, default hull (`L=120 m`, `B=20 m`,
`alpha=22 deg`, `phi=30 deg`, `mu=0.05`, `sigma_c=1.5 MPa`, `sigma_f=0.5 MPa`),
`rho_w=1025 kg/m^3`, `g=9.80665 m/s^2`. Values in newtons, reproduced
programmatically and pinned by the guard suite:

| h [m] | R_crush | R_bend | R_sub | added drag | R_lindqvist | R_riska | band spread |
|---|---|---|---|---|---|---|---|
| 0.30 | 30 911.8 | 1 875.0 | 16 555.3 | 3 075.0 | 52 417.2 | 84 220.9 | 46.6 % |
| 0.45 | 69 551.6 | 4 218.8 | 37 249.5 | 4 612.5 | 115 632.4 | 187 443.2 | 47.4 % |
| 0.60 | 123 647.3 | 7 500.0 | 66 221.4 | 6 150.0 | 203 518.7 | 331 055.7 | 47.7 % |

**Finding:** the two baselines differ by 46-48 % over the studied thickness
range, i.e. about three times the 15 % engineering tolerance used elsewhere in
this platform. Consequences, stated plainly:

* A DEM result compared against the *band centre* is meaningless: the centre is
  an artefact of two unverified coefficient sets.
* Errors must be quoted against each baseline separately, which is what the
  coupling report does.
* The band cannot certify the DEM model. It can only show that a DEM prediction
  is or is not inside the range that two engineering models span, and even that
  only if both models are structurally appropriate for the geometry.


> **Update (literature identifiability audit).** The `WITHHELD` grade below is *partly superseded*: the
> published Lindqvist and Riska predictions transcribed in
> `benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv` are now used as the reference.
> Result: no admissible hull inside a declared envelope reproduces either published column within the
> 15 % gate, and the published relative speed dependence `C2/C1` is several times larger than the shipped
> chain can reach. The coefficient set is therefore graded `REFUTED-IN-ENVELOPE`, not merely unverified.
> See `docs/threefield/LITERATURE_IDENTIFIABILITY.md` for the tables, grades and limitations. The
> statements below describe the audit that preceded this one and are kept unchanged.
## 7. What this audit does not establish

* No comparison with Lindqvist (1989), Riska (1989) or ice-class rule text: the
  coefficient set is unverified against them (`WITHHELD`).
* No physical validation against measured ship resistance in ice.
* No claim that `R_riska > R_lindqvist` at 1 m/s generalises: it holds for the
  default hull and the three studied thicknesses, and the ordering is asserted
  only there. At higher speed the two cross (the Lindqvist quadratic term grows
  faster), which is why the band spread narrows in the speed sweep figure of the
  mission report.
* The bending and submersion coefficients remain conventions; the audit pins them
  so they cannot drift silently, which is the most that can be done offline.

## 8. Reproducing

```bash
python -m pytest tests/test_threefield_resistance.py -q      # 22 checks
python scripts/audit_threefield_resistance.py                # table + figures
```

The script writes `artifacts/threefield/resistance_audit.json` and the two
figures `artifacts/threefield/figures/resistance_thickness_sweep.png` and
`resistance_speed_sweep.png`. The JSON is generated from the same calls the tests
pin, so the published table is never hand-transcribed.