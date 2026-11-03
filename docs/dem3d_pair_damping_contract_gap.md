# Pair-contact damping contract gap in `IceDEM3D` (root-caused, not yet fixed)

Module: `src/tensordem/dem3d.py` (`IceDEM3D.step`, particle–particle branch)
Probe: `scripts/audit_dem3d_pair_damping_gap.py` → `artifacts/dem3d_pair_damping_gap/audit.json`
Pinned by: `tests/test_dem3d_pair_damping_gap.py` (1 pass = control, 2 xfail = the debt)
Figure: `docs/figures/dem3d_pair_damping_gap.png`

## 1. Why this note exists

`tests/test_restitution.py::EngineClosureTests` has been red on `main` since the
`m2536` restoration and was explicitly left red there ("needs an owner decision,
outside the scope of the restoration"). Two red tests, no diagnosis, is exactly
the kind of debt that gets re-argued every mission. This note replaces the guess
("the engine ignores the dashpot") with a three-probe localisation: *which* knob
is dead, *which* branch still works, and how large the residual error is if the
dead knob is bypassed.

## 2. Physics and set-up

Two free spheres, radius `r = 0.025 m`, ice density `917 kg/m³`
(mass `6.00175e-2 kg`, effective pair mass `m/2 = 3.00088e-2 kg`), contact
stiffness `k_n = 5000 N/m`, head-on impact at `v = 0.5 m/s`, drag 0, all bonds
pre-broken without releasing bond energy, tool/platen out of reach. The
documented contact law is the **clipped** linear spring-dashpot

    F_n = max(0, k_n δ - c_n δ̇)          (δ = overlap > 0)

whose reference restitution is `tensordem.restitution.unilateral_restitution`
(RK4 with detach/exit event localisation) and whose textbook (bilateral)
counterpart is `classical_restitution`. Engine restitution is measured as
`|v_out| / |v_in|` of the relative normal velocity on the last free-flight step
before and the first free-flight step after the contact.

## 3. Probes and results

### (A) `DEM3DConfig.contact_damping` sweep, `restitution_coefficient = 1.0`

| `contact_damping` (N s/m) | engine `e` | documented law `e` | bilateral `e` |
| ---: | ---: | ---: | ---: |
| 0 | 0.99987 | 1.00000 | 1.00000 |
| 2 | 0.99987 | 0.78353 | 0.77311 |
| 5 | 1.00003 | 0.56594 | 0.51946 |
| 10 | 0.99996 | 0.35747 | 0.24544 |
| 20 | 1.00000 | 0.17540 | 0.01178 |
| 40 | 1.00000 | 0.06626 | 0.00000 |

`e` is **flat at 1.0 for a 40× change of `c_n`**. Inverting the reference law
gives an *implied* pair damping of `0.0 N s/m` at every sweep point: the
particle–particle branch never reads `contact_damping`.
Consequence: every document that states the pair contact as
`k_n δ - c_n δ̇` with `c_n = contact_damping`
(`docs/dem3d_contact_impact_analytic.md`, `docs/dem3d_mechanics_benchmark_protocol.md`,
`docs/ice_ridge_pipeline_grounding.md`, `docs/accounting_ledger_validation.md`)
describes a law the engine does not execute; the published closure table of
`docs/dem3d_contact_restitution.md` (`c_n = 2.994 → e = 0.69971`) is
unreproducible on `HEAD` (`e = 0.99987`, 42.8 % relative error).

### (B) The other knob: `restitution_coefficient`

| requested `e` | engine `e` | rel. error | <3 % ? |
| ---: | ---: | ---: | --- |
| 0.9 | 0.90211 | +0.24 % | yes |
| 0.7 | 0.71785 | +2.55 % | yes |
| 0.5 | 0.54919 | +9.84 % | **no** |
| 0.3 | 0.39481 | +31.60 % | **no** |

So the engine *has* dissipation, but the mapping it applies is the **bilateral**
`ζ = |ln e| / sqrt(π² + ln² e)`, `c_n = 2 ζ sqrt(k_n m_eff)`, whereas the
equation it integrates is the clipped one. The clipped law needs more damping
for the same `e`; using the bilateral value under-damps, and the error grows as
`e` falls, reaching +31.6 % at `e = 0.3` — the same direction and order as the
32.3 % worst case already published for the bilateral mapping in
`docs/dem3d_contact_restitution.md`.

### (C) Control: the platen branch

| `platen_damping` (N s/m) | engine `e` | documented law `e` |
| ---: | ---: | ---: |
| 0 | 1.00000 | 1.00000 |
| 2 | 0.83892 | 0.83948 |
| 5 | 0.65679 | 0.65972 |
| 10 | 0.46099 | 0.46276 |
| 20 | 0.25903 | 0.25901 |
| 40 | 0.11038 | 0.11113 |

The platen branch tracks the documented clipped law to ≤ 0.5 % across the whole
sweep (it uses `platen_stiffness`/`platen_damping` directly). The defect is
therefore **localised to the particle–particle branch**, not a global absence of
damping, and not a time-integration artefact.

## 4. Root cause (two independent defects, same branch)

* **D1 – dead parameter.** The pair branch computes `gamma_n` only from
  `cfg.restitution_coefficient`, and sets `gamma_n = 0` when it is 1.0 (the
  default). `cfg.contact_damping` survives only in the time-step stability
  bound, never in the force. Any model that tunes `contact_damping` (the
  default `5.0 N s/m`, the scale benchmarks, the shear-specimen protocol) is
  running undamped pair contacts.
* **D2 – wrong inverse.** When `restitution_coefficient < 1`, the damping uses
  the bilateral `ζ` formula instead of the inverse of the clipped law
  (`tensordem.restitution.damping_for_restitution`), i.e. it under-prescribes
  `c_n` by up to ~1.9×, giving +31.6 % error in `e`.

## 5. Error verdict against the platform bar

| quantity | baseline | measured | <3 % ? |
| --- | --- | --- | --- |
| pair `e`, `c_n = 2.994` via `contact_damping` | clipped RK4 `0.69971` | `0.99987` | **no** (+42.8 %) |
| pair `e`, `restitution_coefficient = 0.9` | requested `0.9` | `0.90211` | yes (+0.24 %) |
| pair `e`, `restitution_coefficient = 0.7` | requested `0.7` | `0.71785` | yes (+2.55 %) |
| pair `e`, `restitution_coefficient = 0.5` | requested `0.5` | `0.54919` | **no** (+9.84 %) |
| pair `e`, `restitution_coefficient = 0.3` | requested `0.3` | `0.39481` | **no** (+31.6 %) |
| platen `e`, `c_n = 5` | clipped RK4 `0.65972` | `0.65679` | yes (−0.44 %) |
| platen `e`, `c_n = 40` | clipped RK4 `0.11113` | `0.11038` | yes (−0.68 %) |

Capability grading after this audit: **pair-contact restitution closure =
WITHHELD** (two parameter paths published, neither reproduces its own contract;
`docs/dem3d_restitution_verification.md` numbers hold for the platen branch
only). Bonded-lattice contact stiffness, friction and rolling branches are not
touched by this note.

## 6. Why the fix is not in this commit (honest scope statement)

Both candidate repairs change solver forces, so they change every regression
artefact that uses non-zero `contact_damping` or `restitution_coefficient`:

* D2 alone is a two-line change
  (`gamma_n = damping_for_restitution(e, k, m_eff)` for the linear law,
  secant-stiffness analogue for Hertz) that fixes `e ≥ 0.3` closure, but it
  moves the restitution of every existing restitution-bearing case.
* D1 requires deciding the *precedence and default* of two damping knobs
  (`contact_damping = 5.0` vs `restitution_coefficient = 1.0` are mutually
  inconsistent as defaults: one asks for a damped contact, the other for an
  elastic one). Re-defaulting either silently changes the meaning of every
  published benchmark, so it needs an owner decision, exactly as `m2536`
  recorded.

What is delivered instead is the thing that was missing: a machine-checked
localisation (`scripts/audit_dem3d_pair_damping_gap.py`, `audit.json`), a
control that passes (`test_platen_restitution_matches_documented_clipped_law`),
and two debt pins (`expectedFailure`) that turn into hard failures the moment a
fix lands without updating them.

## 7. Reproduce

```bash
python3 scripts/audit_dem3d_pair_damping_gap.py     # ~25 s, writes artifacts/ + figure
python3 -m pytest tests/test_dem3d_pair_damping_gap.py -q   # 1 passed, 2 xfailed
python3 -m pytest tests/test_restitution.py -q -k Engine    # 2 failed (the debt, unchanged)
```