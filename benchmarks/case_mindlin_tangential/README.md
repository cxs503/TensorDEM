# CASE-DEM-MINDLIN-TANGENTIAL — tangential micro-slip → gross-slip (Mindlin & Deresiewicz 1953)

**Anchor**: `CASE-DEM-MINDLIN-TANGENTIAL` · repo `TensorDEM` · solver
`tensordem.dem3d.IceDEM3D` · `tangential_model="mindlin_partial_slip"`.

**Verdict (machine-checked)**: `validate.py` → **9/9 PASS, ACCEPT**.
Micro-slip max deviation **3.70e-16 μF_n**, gross-slip plateau **0.300000 N = μF_n**
(gate 5 %), initial slope within **0.92 %** of `8 G* a`, fitted `δ_s` within
**4.4e-16** of `3μF_n/(2k_t)`. Legacy linear Cundall–Strack deviates **19.24 %**
on the identical contact → the case discriminates.

---

## 1. Problem and reference

Two elastic bodies pressed together with normal force `F_n` and loaded
tangentially. Mindlin (1949) gives the no-slip tangential spring `k_t = 8 G* a`;
Mindlin & Deresiewicz (1953) add Coulomb friction: the traction cannot exceed
`μp` anywhere, so a **slip annulus** appears at the contact rim from the first
tangential increment while an inner **stick circle** of radius `c < a` carries
the load. The resulting law is

```
F_t = μ F_n [ 1 − (1 − δ/δ_s)^{3/2} ],      δ_s = 3 μ F_n / (16 G* a)
c/a = (1 − F_t/(μ F_n))^{1/3}
```

- `δ ≪ δ_s` → `F_t → k_t δ` (Mindlin no-slip spring, slope `8 G* a`);
- `δ → δ_s` → `F_t → μ F_n` (gross slip, friction saturated).

**Reference source / provenance (zero hand-copy).** Mindlin, R. D. &
Deresiewicz, H. (1953), *Elastic Spheres in Contact Under Varying Oblique
Forces*, **J. Appl. Mech. 20:327–344, DOI 10.1115/1.4010702**. The 1953 original
is paywalled and this sandbox has **no network access** (verified: the DOI
request times out), so **no number is transcribed from its tables**. Every
reference value in `results.json` is instead (a) the closed form above — the
same equation reproduced in Johnson, *Contact Mechanics* (1985) §4.3, eq. 4.32 —
**evaluated inside `run.py` from Hertz first principles** (independent of
`tensordem.mindlin_partial_slip`, which `run.py` never imports), and (b) two
self-consistency identities of the law itself (initial slope `= k_t`, asymptote
`= μF_n`). Two algebraically distinct routes for `δ_s`
(`3μF_n/(16G*a)` vs `3μF_n/(2k_t)`) are evaluated independently and agree to
`< 1e-12` (see `reference.delta_s_form_identity_rel`).

## 2. Method and implementation

| layer | file | what it does |
|---|---|---|
| constitutive law | `src/tensordem/mindlin_partial_slip.py` (new) | closed-form backbone, inverse, stick-circle radius, reversal branch, loop area |
| solver integration | `src/tensordem/dem3d.py` | opt-in `DEM3DConfig.tangential_model="mindlin_partial_slip"` used by the **particle–particle** and **platen** tangential branches |
| case driver | `benchmarks/case_mindlin_tangential/run.py` | runs the production solver, builds the independent reference, errors, convergence, sensitivity, figures |
| acceptance gate | `benchmarks/case_mindlin_tangential/validate.py` | re-reads artifacts only (no solver) and re-applies the thresholds |
| tests | `tests/test_mindlin_partial_slip.py` | closed-form anchors, cross-implementation agreement, solver integration, legacy-path invariance |

**Setup (one real contact).** A single sphere (R = 25 mm, ice-like E = 1 GPa,
ν = 0.3, ρ = 917 kg/m³) rests on the frictional bottom platen. Gravity is off,
so a constant external load `−F_n e_z = −1 N` fixes the contact normal force at
exactly `F_n = 1 N` at equilibrium. A soft lateral drive spring
(`k_drive = 2e4 N/m`, end speed `1e-3 m/s`) ramps the tangential force through
`δ ≈ 1.6 δ_s`. The recorded force is the solver's own platen friction force and
the recorded displacement is the solver's own accumulated tangential contact
state — no post-processing of the law. The other seven lattice particles are
parked 5 m away (no gravity ⇒ inert) and bonds are muted (`bond_stiffness=1e-6`)
so the run contains exactly one contact.

Note the platen-branch convention: its overlap is `bottom_z + R − z` with
`bottom_z` the *centre* reference, i.e. the effective contact surface is
`z = 0` for a sphere whose centre starts at `z = 0`. The Hertz equilibrium is
therefore `z = −δ_n`.

## 3. Parameters (dimensionless groups complete)

| symbol | value | unit | note |
|---|---|---|---|
| R | 2.50000e-02 | m | sphere radius |
| E, ν | 1.000e+09, 0.300 | Pa, – | ice-like elastic |
| ρ | 917 | kg/m³ | m = 6.00175e-02 kg |
| μ | 0.300 | – | Coulomb, plane and sphere |
| F_n | 1.000 | N | applied normal load |
| E* | 1.09890e+09 | Pa | `E/(1−ν²)` (rigid plane) |
| G* | 2.26244e+08 | Pa | `E/(2(2−ν)(1+ν))` |
| a | 2.57443e-04 | m | `a/R = 1.0298e-02` |
| δ_n | 2.65107e-06 | m | `δ_n/R = 1.0604e-04` |
| k_n | 3.77206e+05 | N/m | Hertz secant |
| k_t | 4.65960e+05 | N/m | `8G*a`, `k_t/k_n = 1.23529 = 3(1−ν)/(2−ν)` |
| δ_s | 9.65748e-07 | m | `δ_s/δ_n = 0.36429` |
| μF_n/(G*a²) | 1.17688e-02 | – | dimensionless load |
| Δt | 2.78783e-06 | s | `recommended_dt` (×0.5, ×2 swept) |

## 4. Reference vs got (every row machine-generated)

`F_t` normalised by `μF_n`, `δ` by `δ_s`; "err" = `|F_t − F_MD|/(μF_n)`.

| region | δ range | got (TensorDEM) | reference (Mindlin–Deresiewicz) | err | verdict |
|---|---|---|---|---|---|
| micro-slip | δ ≤ 0.9 δ_s (n=5766) | curved branch, `F_t → k_t δ` | `μF_n[1−(1−δ/δ_s)^{3/2}]` | **3.70e-16** | ✅ < 5 % |
| transition | \|δ−δ_s\| ≤ 0.1 δ_s | knee at δ/δ_s = 1 | same | **1.85e-16** | ✅ < 5 % |
| gross slip | δ ≥ 1.1 δ_s (n=672) | plateau 0.3000000 N | `μF_n` = 0.3 N | **1.85e-16** | ✅ < 1 % |
| initial slope | δ ≤ 0.05 δ_s | `k_t = 4.6169e+05` N/m | `8G*a = 4.6596e+05` N/m | 9.15e-3 (secant estimator bias) | ✅ < 2 % |
| δ_s (fit) | whole micro-slip branch | 9.657482e-07 m | 9.657482e-07 m | 4.39e-16 | ✅ < 5 % |
| **discrimination** | same contact, legacy `cundall_strack` | linear spring to the clamp | MD law | **19.24 %** max at δ/δ_s = 0.667 | ✅ > 5 % |

The analytic (solver-free) linear law shows the same signature: **19.23 %** max
deviation just past its clamp point and **10.36 %** at `δ = δ_s/2`, which is the
closed-form `3/4 − (1−2^{-1.5}) = 0.1036`. So the 5 % gate can only be met by a
law with the correct curvature — the case is not passable by the legacy model.

## 5. Convergence and sensitivity (≥ 2 axes)

| axis | levels | result |
|---|---|---|
| time step | Δt × {0.5, 1, 2} = {1.394e-6, 2.788e-6, 5.576e-6} s | micro-slip err {7.40e-16, 3.70e-16, 3.70e-16}; no rate dependence (the branch is algebraic, so Δt only changes sampling; fitted-order slope is meaningless here and is reported as such) |
| friction μ | {0.2, 0.3, 0.5} | fitted δ_s {6.43832e-07, 9.65748e-07, 1.60958e-06} m vs theory, rel err ≤ 4.6e-15; measured exponent **μ^1.000000** (theory 1) |
| normal load F_n | {0.5, 1, 2} N | fitted δ_s {6.08383e-07, 9.65748e-07, 1.53303e-06} m, rel err ≤ 2.8e-15; measured exponent **F_n^0.666667** (theory 2/3) |

The two exponents are the non-trivial part: `δ_s ∝ μ` and `δ_s ∝ F_n^{2/3}` follow
from `δ_s = 3μF_n/(2·8G*a)` with `a ∝ F_n^{1/3}`, and the solver reproduces both
to 1e-15, which means the normal-force → contact-radius → stiffness chain is
wired correctly, not just the curve shape.

## 6. Failure modes, limits, artifacts

* **Stated limit**: the closed forms cover monotonic loading and reversals *from
  gross slip*. A general multi-reversal path inside the envelope needs a moving
  kinematic anchor; the solver's opt-in branch is documented as
  monotonic-backbone accurate and the cyclic state machine lives in
  `tensordem.contact_tangential.mindlin_tangential_step`.
* **Opt-in only**: the flag requires `contact_model="hertz"` and
  `friction_regularization="none"` (validated with `ValueError`); the default
  `cundall_strack` path is unchanged and covered by a regression test.
* **No rate dependence / no tangential dashpot** in the law itself; solver
  damping terms are separate and do not enter `F_t(δ)`.
* **Reference provenance risk**: the 1953 table values were not consulted (no
  network). This is a closed-form comparison, not a digitisation of the paper's
  figures; if a scan of the original becomes available the same driver can
  compare against it by extending `reference_chain`.
* **Coarse statistics on the plateau**: only 672 samples lie past 1.1 δ_s
  because the drive keeps pulling after gross slip; the plateau is exact to
  machine precision so this is not a limitation of accuracy.

**Artifacts** (`artifacts/mindlin_tangential/`, sha256 in `checksums.json`):

| file | content |
|---|---|
| `results.json` | reference chain, all metrics, criteria, convergence/sensitivity, environment |
| `curves.csv` | solver curve (δ, F_t, c/a) + analytic curve on a 1200-point grid |
| `figures/fig1_ft_delta_curve.png` | `F_t/(μF_n)` vs `δ/δ_s`: analytic, solver MD branch, legacy linear |
| `figures/fig2_error_vs_delta.png` | point-wise deviation, MD branch vs legacy, gate line |
| `figures/fig3_dt_and_sensitivity.png` | Δt convergence + δ_s(μ) + δ_s(F_n) |
| `figures/fig4_stick_circle_map.png` | stick circle shrinking inside the slip annulus |
| `run.log` | full console log with environment |
| `checksums.json` | sha256 of every artifact |

## 7. Reproduce (one command each)

```bash
# full case (≈ 5.3 min CPU, 8 threads not needed, single core)
PYTHONPATH=src python benchmarks/case_mindlin_tangential/run.py \
    --out artifacts/mindlin_tangential

# acceptance gate (re-reads artifacts only)
PYTHONPATH=src python benchmarks/case_mindlin_tangential/validate.py \
    --out artifacts/mindlin_tangential

# unit/integration tests (10 tests, ≈ 13 s)
PYTHONPATH=src python -m pytest tests/test_mindlin_partial_slip.py -q
```

Measured on `Linux loongarch64`, python 3.11.6, torch 2.7.1, numpy 2.4.6:
`run.py` **318.3 s** (6438 recorded samples, 6551 solver steps in the main run),
`validate.py` **9/9 ACCEPT**, `pytest` **10 passed in 13.45 s**.