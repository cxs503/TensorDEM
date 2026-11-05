# Subgrade bearing-capacity model: reference verification (validation-debt closure)

Scope: the undrained / drained ultimate vertical capacity used by the ice-ridge
grounding workflow (`tensordem.ridge_geotech.MohrCoulombSoil`, consumed by
`SofteningVerticalSpring`, `ridge_grounding.capacity_envelope` and the DNV-OS-F101
utilisation chain).

Modules: `src/tensordem/ridge_geotech.py` (model),
`src/tensordem/ridge_geotech_verification.py` (reference ledger),
`scripts/verify_subgrade_capacity.py` (runner + figures).
Tests: `tests/test_ridge_geotech_capacity_verification.py`,
`tests/test_ridge_grounding.py`.
Artifacts: `artifacts/ridge_geotech_capacity/verification.json`,
`artifacts/ridge_geotech_capacity/figures/*.png`.

## 1. Why this verification existed as debt

`docs/ice_ridge_pipeline_grounding.md` states that *"capacities are the verified
quantities"* of the subsea model, yet the only test of the undrained branch
(`test_undrained_vertical_capacity_uses_skempton_embedment`) re-derived the
implementation:

```python
assert pressure == pytest.approx(30.0e3 * min(5.14 * 1.2, 9.0) * 1.2, rel=1e-9)
```

A test written that way can never fail: it compares the code with itself. The
same pattern hid a real defect, described in section 3.

## 2. What the model is checked against

| reference | quantity | source |
|---|---|---|
| Prandtl (1920) slip-line solution | `q_ult = (2 + pi) s_u = 5.1416 s_u` for a surface strip on phi = 0 clay | closed form |
| Skempton (1951) | `N_c = N_c0 (1 + 0.2 z/B)`, `N_c0 = 5.14` (strip) / `6.2` (square, circular), capped at `7.5` / `9.0` for `z/B >= 2.5` | table values (Das, *Principles of Foundation Engineering*, table 3.3) |
| Vesic (1973) | `N_c, N_q, N_gamma` at phi = 20, 25, 30, 35, 40 deg | table values (Das table 3.4) |
| Meyerhof / Hansen | `d_c = 1 + 0.4 k`, `d_q = 1 + 2 tan(phi) (1 - sin phi)^2 k`, `k = min(z/B, 1)` | closed form |
| physical invariants | linearity in `s_u`, monotone and capped lateral `N_p`, spring peak = 0.9 p_ult, residual = 0.8 peak | construction |

The lateral (horizontal) pipe-soil resistance `N_p = 2 + 1.5 z/D` capped at 9 is a
rule of practice whose primary text (DNV-RP-F114) is **not available in this
worktree**, so only its invariants (monotone, capped) are machine-checked and its
reference grade is recorded as `WITHHELD`, not `REAL`.

## 3. Defect found and corrected

The undrained branch read

```python
n_c = min(5.14 * (1.0 + 0.2 * depth_below_mudline / width), 9.0)
return self.shear_strength * n_c * (1.0 + 0.2 * k)      # k = min(z/B, 1)
```

Skempton's `N_c` **already contains** the embedment enhancement, so the capacity
was multiplied by an embedment factor twice, and the "capped at 9" cap was applied
*before* the second factor, so the cap did not hold either:

| `z/B` | pre-fix `N_c` | Skempton (1951) | over-prediction |
|---|---|---|---|
| 0.0 | 5.1400 | 5.1400 | +0.0 % |
| 0.5 | 6.2194 | 5.6540 | +10.0 % |
| 1.0 | 7.4016 | 6.1680 | +20.0 % |
| 2.0 | 8.6352 | 7.1960 | +20.0 % |
| 2.5 | 9.2520 | 7.5000 (cap) | +23.4 % |
| 3.0 | 9.8688 | 7.5000 (cap) | +31.6 % |
| 6.0 | 10.8000 | 7.5000 (cap) | +44.0 % |

Rows are `defect.rows` of `artifacts/ridge_geotech_capacity/verification.json`
(computed by `verify_subgrade_capacity_model`, never transcribed by hand).

Three consequences, all of them the kind the >15 % acceptance rule forbids:

1. a systematic 20 % over-prediction of undrained subgrade capacity in the range
   that the Beaufort case actually uses (`z/B = 1.55`);
2. the documented `N_c <= 9` cap was exceeded by up to 20 % (10.8 reached);
3. the strip-footing cap that belongs to a pipeline (`7.5`) was replaced by the
   square/circular cap (`9.0`).

**Correction** (`skempton_nc` in `ridge_geotech.py`, exported):

```
q_ult = s_u * N_c,   N_c = min(5.14 * (1 + 0.2 z/B), 7.5)      (strip, default)
                     N_c = min(6.2  * (1 + 0.2 z/B), 9.0)      (square / circular)
```

one embedment factor, cap applied last, and the drained branch is untouched.

## 4. Results of the 43 reference checks

`python scripts/verify_subgrade_capacity.py` (artifact
`artifacts/ridge_geotech_capacity/verification.json`):

| group | checks | worst relative error | verdict |
|---|---|---|---|
| Prandtl anchor (`N_c(z/B=0)`, surface pressure) | 2 | 3.07e-4 | pass |
| Skempton table, strip and square | 12 | 7.08e-4 | pass |
| Skempton caps (strip 7.5, square 9.0) | 2 | 0 | pass |
| Vesic `N_c, N_q, N_gamma` at 5 friction angles | 15 | 6.83e-4 | pass |
| Drained composite `q_ult` from table factors + Hansen depth factors | 1 | 1.31e-5 | pass |
| Embedment applied once (`z/B` = 0.5 … 6) | 5 | 0 | pass |
| Linear in `s_u`, spring peak/residual fractions | 3 | <= 1.4e-16 | pass |
| Lateral `N_p` invariants (reference `WITHHELD`) | 2 | 0 | pass |
| **total** | **43** | **7.08e-4 (0.071 %)** | **43 / 43 pass** |

Every error is table-rounding noise (the references are printed to 2 decimals);
the largest is 0.071 % on `Skempton N_c` at `z/B = 0.5`. Nothing is above 3 %.

Before the fix the same ledger failed at three families of checks: the two
`embedment_once_*` rows, the `skempton_cap_strip` row and the linearity of the
spring peak with `p_ult` (the spring inherits `p_ult`).

## 5. Engineering impact on the ice-ridge grounding campaign

Beaufort case (`D = 0.914 m`, cover `0.5 m`, `z/B = 1.547`, `s_u = 30 kPa`):

| quantity | pre-fix | corrected | change |
|---|---|---|---|
| ultimate capacity `p_ult` (kN/m) | 221.46 | 184.55 | -16.7 % |
| subgrade spring peak `0.9 p_ult` (kN/m) | 199.31 | 166.09 | -16.7 % |
| subgrade spring residual `0.8 peak` (kN/m) | 159.45 | 132.87 | -16.7 % |

The coupled DEM campaign is subgrade-governed, so this propagates: the seabed
capacity envelope and the pipe settlement at a given ice load both change by the
same 16.7 % factor, and `artifacts/ridge_grounding/*.json` were regenerated with
`scripts/run_ice_ridge_grounding.py` after the fix. Section 7 of
`docs/ice_ridge_pipeline_grounding.md` carries the regenerated numbers; the
qualitative verdicts of that campaign (load-displacement shape satisfied, pipe
stress against Winkler theory satisfied, seabed-strength scaling monotone) are
unchanged, and the ice-side capacity remains 5-7 times the seabed capacity, so the
classification "subgrade-governed" survives the 16.7 % correction.

## 6. Honest limitations of *this* verification

1. It verifies the **capacity formula**, not the geotechnical idealisation. The
   soil is still a single homogeneous Mohr-Coulomb layer with no pore pressure
   coupling, no softening of `s_u` with plasticity, no trench geometry and no
   gapping; a 0.07 % agreement with a textbook formula does not make the model a
   site-specific prediction.
2. Skempton's table is cited through a secondary source available offline
   (Das). The primary 1951 paper was not read here; the closed-form Prandtl anchor
   (`2 + pi`) is the primary-text-independent part.
3. The choice of the strip cap (`7.5`) over the square/circular cap (`9.0`) is a
   modelling decision justified by the pipe being a very long footing; it changes
   nothing below `z/B = 2.3` and everything above it.
4. The lateral `N_p` rule is graded `WITHHELD` (primary text absent). Its cap of 9
   is asserted as practice, not as a verified reference.
5. Relative errors are reported against 2-decimal table values; agreement better
   than ~0.5 % cannot be distinguished from the printing precision.
6. The 0.9 / 0.8 peak-and-residual fractions and the subgrade modulus
   `0.65 E_s/((1-nu^2) B)` remain chosen parameters (see
   `docs/ice_ridge_pipeline_grounding.md` section 4); they are consistent, not
   calibrated.

## 7. Reproducing

```bash
python scripts/verify_subgrade_capacity.py          # ledger + 2 figures, < 10 s
python -m pytest tests/test_ridge_geotech_capacity_verification.py -q
python -m pytest tests/test_ridge_grounding.py -q   # 24 tests, downstream behaviour
```