# TensorDEM capability: Mindlin–Deresiewicz partial-slip tangential law

**Status**: available (opt-in) in `tensordem.dem3d` as
`DEM3DConfig.tangential_model="mindlin_partial_slip"`, default unchanged
(`"cundall_strack"`).
**Validation case**: `CASE-DEM-MINDLIN-TANGENTIAL` — see
`benchmarks/case_mindlin_tangential/README.md` for the paper-style report.

## Why it exists

The legacy tangential branch is Cundall–Strack: `F_t = k_t δ` **clamped** at
`μF_n`. Two elastic spheres/plates under partial slip are not linear: Coulomb
saturation at the contact rim starts at the *first* tangential increment, so
the pre-slip branch is curved (Mindlin & Deresiewicz 1953):

```
F_t = μ F_n [1 − (1 − δ/δ_s)^{3/2}],   δ_s = 3 μ F_n / (16 G* a)
```

with the stick circle `c/a = (1 − F_t/(μF_n))^{1/3}` shrinking to zero at gross
slip. Consequence measured on the case contact: the linear law is off by
**19.24 %** of `μF_n` just past its own clamp point (10.36 % at `δ_s/2`). For
ice–structure friction, shear-band onset and any calibration of `μ` from a
force–displacement curve, that is well above the 5 % acceptance line.

## Implementation map (L2 / L3 / L4 evidence)

| requirement | artifact | evidence |
|---|---|---|
| **impl_files** (non-placeholder) | `src/tensordem/mindlin_partial_slip.py` (316 lines, no stubs): `hertz_contact_radius`, `mindlin_no_slip_stiffness`, `gross_slip_displacement`, `partial_slip_magnitude`, `partial_slip_force`, `stick_radius_ratio`, `inverse_displacement`, `reversal_branch`, `reversal_loop_area` | reduced to `8G*a`/`3μF_n/(2k_t)` identities in `tests/test_mindlin_partial_slip.py` (10 tests, 13.45 s) |
| **entry** (runnable entry point) | `benchmarks/case_mindlin_tangential/run.py` (+ `validate.py` gate) | `PYTHONPATH=src python benchmarks/case_mindlin_tangential/run.py --out artifacts/mindlin_tangential` → exit 0, `results.json` + 4 PNG + CSV + log |
| **run_artifacts** (real run) | `artifacts/mindlin_tangential/{results.json, curves.csv, run.log, checksums.json, figures/*.png}` | produced by the command above in 318.3 s on `Linux loongarch64`, torch 2.7.1; sha256 manifest in `checksums.json`, verified by `validate.py` (`sha256_match` PASS) |
| **integration_callers** (production solver) | `src/tensordem/dem3d.py`: config field + validation (`__post_init__`), particle–particle branch (`mindlin_partial_slip.partial_slip_force` in `forces()`), platen branch (`_plane_friction_force` → `mindlin_no_slip_stiffness` / `partial_slip_magnitude`) | `grep -n "mindlin_partial_slip" src/tensordem/dem3d.py` → 15 hits; the case runs through `IceDEM3D.step()`, not through the module directly, and `tests/test_mindlin_partial_slip.py::test_dem3d_mindlin_branch_reproduces_the_analytic_curve` asserts the solver path |
| **benchmark** (ref vs got, numeric) | `results.json → got_mindlin_partial_slip` vs `reference.chain` / closed form | micro-slip **3.70e-16**, transition **1.85e-16**, plateau **1.85e-16** of `μF_n` (gate 5 %); `δ_s` fit vs theory **4.39e-16**; slope **0.92 %** (gate 2 %) |
| **repro_cmd** | see below | `validate.py` → **9/9 ACCEPT** |

## Repro commands

```bash
# 1) full benchmark, writes artifacts/mindlin_tangential (≈ 5.3 min, single core)
PYTHONPATH=src python benchmarks/case_mindlin_tangential/run.py --out artifacts/mindlin_tangential

# 2) acceptance gate, artifacts only (no solver)
PYTHONPATH=src python benchmarks/case_mindlin_tangential/validate.py --out artifacts/mindlin_tangential

# 3) tests
PYTHONPATH=src python -m pytest tests/test_mindlin_partial_slip.py -q

# 4) solver regression of the legacy path (must stay green: 60 passed, 2 skipped)
PYTHONPATH=src python -m pytest tests/test_dem3d.py tests/test_contact_tangential.py \
    tests/test_tangential_friction_verification.py tests/test_dem3d_cell_list.py -q
```

Outputs are deterministic: the same command reproduces `results.json` leaf for
leaf (the seeds are analytic — there is no RNG in the case).

## Physics/verification summary

| quantity | theory | solver | rel diff |
|---|---|---|---|
| `k_t = 8G*a` | 4.65960e+05 N/m | 4.6169e+05 N/m (secant over δ ≤ 0.05 δ_s) | 9.15e-3 (estimator bias, gate 2 %) |
| `δ_s = 3μF_n/(2k_t)` | 9.65748e-07 m | fit 9.65748e-07 m | 4.39e-16 |
| plateau | 0.3 N | 0.3000000000 N | 1.85e-16 |
| `δ_s(μ)` exponent | 1 | 0.9999999999 | – |
| `δ_s(F_n)` exponent | 2/3 | 0.6666666666 | – |

## Limits

* The opt-in branch implements the **monotonic backbone** plus closed-form
  reversal-from-gross-slip primitives. Arbitrary multi-reversal paths inside the
  envelope are handled by the anchor state machine in
  `tensordem.contact_tangential.mindlin_tangential_step`, not by this flag; the
  docstring states this.
* Requires `contact_model="hertz"` and `friction_regularization="none"`
  (validated, `ValueError` otherwise), because `δ_s` is defined with the Hertz
  contact radius and a single `μ`.
* No adhesion (no JKR/DMT offset) and no rate dependence: the law is
  quasi-static; solver dashpots act outside it.
* Reference provenance: the 1953 original is paywalled and unreachable from the
  sandbox, so the comparison is against the closed form (Johnson 1985 §4.3)
  evaluated independently in `run.py`; no table digit is transcribed.