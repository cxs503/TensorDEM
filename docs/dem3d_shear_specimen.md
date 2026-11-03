# DEM3D mode-II (direct-shear) specimen protocol and shear-threshold calibration

## Why this page exists

Two repository documents flagged the same gap and neither closed it:

* `docs/dem3d_material_calibration_roadmap.md`: *"Still open: shear specimen
  protocol, objectivity under rigid-body rotation, fracture-energy calibration
  from measured crack growth."*
* `docs/dem3d_bond_fracture_energy.md`: *"The **shear** criterion
  (`shear_breaking_strain`) releases the elastic energy of a bond that may be
  barely stretched axially. It is a strain criterion without a mode-II energy
  release rate, so `G_f` above applies to tensile sheets only."*

`tensordem.dem3d` decides mode-II bond failure with a **deviatoric strain
invariant** (SIR-PC2|F05):

```
J2_i = sqrt(2/3 * dev(E_i) : dev(E_i))          # per particle, E from least-squares local F
bond shear strain = 0.5 * (J2_i + J2_j)
failure when bond shear strain > shear_breaking_strain (default 0.03)
```

Before this increment no document mapped `shear_breaking_strain` to a
measurable macroscopic strain, no test verified the invariant against a closed
form, and no case drove a specimen in shear. The only shear case in the suite
(`benchmarks/dem3d/mechanics_cases.json::single_bond_shear_failure`) applies a
fixed `gamma = 0.1` once and checks the failure-mode bit — it does not calibrate
anything.

This page adds `src/tensordem/shear_specimen.py`,
`scripts/benchmark_dem3d_shear_specimen.py` and
`tests/test_dem3d_shear_specimen.py`. **The solver is not modified.**

## 1. Case description (physics, geometry, parameters)

| item | value |
| --- | --- |
| lattice | simple-cubic, 26 neighbours (13 half-space bonds), central linear spring |
| bond stiffness `k` | 2000 N/m |
| particle radius / lattice spacing `a` | 0.025 m / 0.050 m |
| particle mass | 0.060018 kg (rho = 917 kg/m³) |
| tensile threshold `breaking_strain` | parked at 1.0e3 in all mode-II runs (mode isolation) |
| shear threshold `shear_breaking_strain` | 0.003 / 0.01 / 0.03 / 0.05 (mapping), 0.03 (specimen) |
| affine constant `C44 = 7k/(3a)` | 93 333.33 Pa |
| direct-shear specimen | nx×ny×nz = 9×5×5 (225 particles), height `h = 4a = 0.200 m`, `A = 9a·5a = 0.1125 m²` |
| bottom boundary | hard fixed particle layer (`fix_bottom`) |
| top boundary | rigid kinematic platen, lateral displacement `Δx = v t` |
| lateral faces | free (`fix_x_edges = False`) |
| ramp speed / time step | v = 0.1–0.4 m/s, `Δt = 3.2737e-5 s` (solver guard) |
| damping | `drag = 10 N·s/m`, `contact_damping = 5 N·s/m` |

The number of particles is small on purpose: this is a criterion/protocol audit
of the solver's own model, not a sea-ice material validation.

## 2. Method

1. **Closed forms.** For simple shear `u_a = gamma·x_b` the Green-Lagrange
   tensor has `E_ab = gamma/2`, `E_bb = gamma²/2`, hence
   `J2 = sqrt(gamma²/3 + gamma⁴/9)`; inverting `J2 = s` gives

   ```
   gamma_c(s) = sqrt(1.5 * (sqrt(1 + 4 s²) - 1))   =  sqrt(3)·s · (1 - s²/6 + ...)
   ```

   For uniaxial strain `F = diag(1+eps,1,1)` the same invariant is
   `(2/3)(eps + eps²/2)`, so the deviatoric criterion fires at
   `eps = sqrt(1+3s) - 1`.
2. **Solver-side verification (`affine_shear_probe`, `shear_failure_onset`,
   `uniaxial_failure_onset`).** A 3×3×3 specimen is deformed affinely; the
   solver's own measure `IceDEM3D._objective_shear_strain` is compared with the
   closed form, and the failure onset is found by bisection (80 halvings) on the
   solver's own `forces(update_fracture=True)` path. `{100}`-plane isotropy and
   rigid-rotation invariance are probed on the same specimen.
3. **Driven specimen (`run_direct_shear`).** The bottom layer is a hard support,
   the top layer is displaced laterally at constant speed (rigid platen), the
   lateral faces are free. The shear traction is the reaction `Σ F_x` on the fixed
   layer divided by `A`; the engineering strain is `Δx/h`. The apparent modulus
   is the least-squares slope of `tau(gamma)` on a late intact window (free
   intercept, because the load path from the platen into the block has its own
   transient).
4. **Failure run.** Same specimen driven to `gamma = 0.06`; the first bond event
   is located by step-by-step monitoring, and the local criterion value at that
   step is recorded.

Toolchain: pure `torch` (float64, `torch.set_num_threads(1)`), matplotlib for the
figures, JSON+CSV for the machine-readable record. Run:

```bash
python scripts/benchmark_dem3d_shear_specimen.py --output results-dem3d-shear-specimen
python -m pytest tests/test_dem3d_shear_specimen.py -q
```

## 3. Results

### 3.1 Affine closed-form check of the solver's criterion

| gamma | solver `J2` | closed form | relative error |
| --- | --- | --- | --- |
| 1.0e-3 | 5.773503654145e-04 | 5.773503654147e-04 | 3.5e-13 |
| 1.0e-2 | 5.773598916139e-03 | 5.773598916139e-03 | 4.4e-16 |
| 3.0e-2 | 1.732310595707e-02 | 1.732310595707e-02 | 9.9e-16 |
| 5.0e-2 | 2.887953908527e-02 | 2.887953908527e-02 | 1.8e-15 |
| −2.0e-2 | 1.154777515849e-02 | 1.154777515849e-02 | 8.0e-16 |

Particle-to-particle spread ≤ 1.3e-15, so every particle (including corners)
reconstructs the affine `F` exactly. `{100}`-plane isotropy at `gamma = 0.02`:
xy = 1.154777515849e-02, yz = 1.154777515849e-02, xz = 1.154777515849e-02,
absolute spread 1.2e-16. A rigid rotation + translation of the same specimen
registers `J2 = 3.8e-15` (rotation invariance of the measure).

### 3.2 Threshold → macroscopic strain mapping (bisection on the solver)

| `s` | measured onset `gamma` | closed form `gamma_c(s)` | first order `√3·s` | rel. error | local `J2` at onset | failure modes |
| --- | --- | --- | --- | --- | --- | --- |
| 0.003 | 5.196129040e-03 | 5.196129040e-03 | 5.196152423e-03 | 4.1e-12 | 0.0030000000000000243 | [2] |
| 0.010 | 1.731964220e-02 | 1.731964220e-02 | 1.732050808e-02 | 2.5e-13 | 0.010000000000000037 | [2] |
| 0.030 | 5.193817829e-02 | 5.193817829e-02 | 5.196152423e-02 | 9.4e-15 | 0.030000000000000010 | [2] |
| 0.050 | 8.649475804e-02 | 8.649475804e-02 | 8.660254038e-02 | 5.9e-15 | 0.050000000000000044 | [2] |

| `s` | measured uniaxial onset | closed form `√(1+3s)−1` | axial threshold needed to pre-empt |
| --- | --- | --- | --- |
| 0.003 | 4.489920308e-03 | 4.489920308e-03 | > 4.49e-3 |
| 0.010 | 1.488915651e-02 | 1.488915651e-02 | > 1.49e-2 |
| 0.030 | 4.403065089e-02 | 4.403065089e-02 | > 4.40e-2 |
| 0.050 | 7.238052948e-02 | 7.238052948e-02 | > 7.24e-2 |

Two consequences that were previously undocumented:

* `shear_breaking_strain` is a **deviatoric** threshold, not a mode-II one. Under
  uniaxial strain it fires at `eps = √(1+3s)−1 ≈ 1.47 s`; for the shipped default
  `s = 0.03` that is `0.0440`, far above the default `breaking_strain = 0.015`.
  The shear criterion therefore never governs uniaxial loading of the default
  set: it only matters for shear-dominated paths, or for `ε_b > 0.044`.
* The "effective shear failure strain" of an *affine* deformation is
  `√3·s = 0.0520` for the default, i.e. 73 % above `s` itself. Quoting `s` as a
  material strain is a factor-1.7 error.

### 3.3 Driven specimen: rate dependence (the honest negative result)

| v [m/s] | steps | τ(γ=0.025) [Pa] | G_app [Pa] | G_app/C44 | τ deviation vs next slower |
| --- | --- | --- | --- | --- | --- |
| 0.4 | 382 | 232.71 | 18 724 | 0.201 | — |
| 0.2 | 764 | 1098.03 | 87 217 | 0.934 | 3.72 (vs 0.4) |
| 0.1 | 1527 | 1320.04 | 60 246 | 0.645 | 0.202 (vs 0.2) |

The response is **not yet rate converged**: halving the ramp speed still changes
the traction by 20 %, and the apparent modulus by 31 %. The cause is structural,
not a bug: the imposed displacement must diffuse from the platen into the block.
With per-particle viscous drag the displacement diffusion time scales as
`t ~ drag·h²/(k·a²) = 10·0.04/(2000·0.0025) = 0.08 s`, while the ramp takes only
`gamma_max·h/v = 0.005/0.1 = 0.05 s` at v = 0.1. Reaching the quasi-static limit
needs roughly one decade slower ramps, i.e. ~1.5e4 steps per run at the solver's
conservative `Δt`.

**Therefore the driven-specimen apparent modulus is reported as WITHHELD**, not
as a validated quantity. The geometry series below is given for completeness but
carries the same ≥20 % rate bias, which is larger than the geometric trend.

| varied axis | nx×ny×nz | L_x/h | L_y/h | G_app [Pa] | G_app/C44 | fit residual |
| --- | --- | --- | --- | --- | --- | --- |
| nx | 5×5×5 | 1.25 | 1.25 | 57 536 | 0.616 | 2.4 % |
| nx | 9×5×5 | 2.25 | 1.25 | 86 406 | 0.926 | 0.7 % |
| nx | 13×5×5 | 3.25 | 1.25 | 97 892 | 1.049 | 0.6 % |
| ny | 9×7×5 | 2.25 | 1.75 | 91 141 | 0.977 | 0.6 % |

(The 1.049 value is *above* `C44`, which is physically admissible for a
non-affine block with a clamped base; it is one more reason not to read these
numbers as a modulus claim.)

### 3.4 Shear run to first bond failure (mode II isolated)

| quantity | value |
| --- | --- |
| first bond event at `gamma` | 0.03991 (`t = 0.03991 s`) |
| bonds broken at that event | 2 |
| failure-mode codes at that event | `{2}` (shear only) |
| local bond invariant at the event | 0.0300264 |
| local / threshold | 1.00088 |
| affine onset `gamma_c(0.03)` | 0.051938 |
| macro onset / affine onset | 0.768 |
| max alive-bond invariant before the event | 0.029604 |
| global `gamma/√3` at the same instant | 0.022681 |
| strain concentration (local max / global affine) | 1.305 |
| bonds broken by end of run (γ = 0.06) | 778, all mode code 2 |
| recorded release energy at end | 0.6855 J (mean 8.81e-4 J/bond) |
| bond elastic energy at end | 0.4075 J |

The macroscopic onset is 23 % *earlier* than the affine prediction, and the
deficit is fully accounted for by the local strain concentration:
`1/1.305 = 0.766` against the measured `0.768`. The criterion itself is honoured
exactly — the first event occurs when the local invariant reaches `s` to within
one sampling increment (0.09 %).

The energy ledger confirms the documented model limitation quantitatively: a
bond that fails in shear is barely stretched axially, so its recorded release
energy `½kδ²` is not a mode-II energy release rate. The whole run releases
0.686 J while 778 bonds break at a mean 8.81e-4 J each; there is no bond
tangential stiffness and no `G_IIc` to calibrate against. Mode-II fracture energy
in this model is **NOT_SUPPORTED** by construction.

## 4. Figures

`shear_specimen_response.png` (archived copy in `docs/dem3d-shear-specimen/`):

1. direct-shear `tau(gamma)` for `nx = 5, 9, 13`, the failure run, the first
   bond event and the affine onset;
2. `G_app/C44` versus `L_x/h` and `L_y/h` against the affine line;
3. local bond invariant, global `gamma/√3` and the threshold in one frame — the
   concentration factor is the vertical gap.

`shear_specimen_criterion.png`:

1. solver measure versus the closed form on a log-log plot;
2. the bisected onset mapping `gamma_c(s)` (and the uniaxial branch), with the
   first-order `√3·s` visible as the 0.04 % offset;
3. the energy ledger around the first event window.

## 5. Notes, pitfalls and limits

* The affine probes are **linear-kinematics statements about the failure
  criterion**. They say nothing about dynamics, crack propagation or sea ice.
* `J2` is computed from a least-squares local `F` using `pinv` of the bond
  covariance. On a *free surface* particle the neighbourhood is missing bonds,
  so the reconstructed `F` is an average over a smaller stencil — this is exactly
  what makes the driven specimen's local invariant exceed the global estimate
  (concentration factor 1.305 here).
* The rigid-platen displacement control imposes work on the specimen without a
  servo; the reaction therefore contains the elastic wave transient. A
  servo-controlled (constant-force) boundary or a much slower ramp would be the
  next step, and is the reason the modulus is withheld.
* The specimen area convention is `nx·a × ny·a` (population), not
  `(nx−1)a × (ny−1)a` (footprint); the two differ by 23 % in stress for `nx=9,
  ny=5`. Any stress quoted from a DEM block must state which convention it used.
* `run_direct_shear` uses the solver's private `_objective_shear_strain`
  deliberately: the audit must exercise the same code path that decides failure,
  not a reimplementation.

## 6. Error judgement against baselines

| quantity | value | baseline | relative error | < 3 % ? |
| --- | --- | --- | --- | --- |
| affine shear invariant | see §3.1 | `sqrt(gamma²/3+gamma⁴/9)` | ≤ 3.5e-13 | **yes** |
| `{100}` plane isotropy | spread 1.2e-16 | zero spread | exact | **yes** |
| rigid-rotation invariance | 3.8e-15 | 0 | exact | **yes** |
| affine shear onset `gamma_c` | see §3.2 | `sqrt(1.5(√(1+4s²)−1))` | ≤ 4.1e-12 | **yes** |
| uniaxial deviatoric onset | see §3.2 | `√(1+3s)−1` | ≤ 7.4e-14 | **yes** |
| local invariant at first event (driven, non-affine) | 0.030026 | `s = 0.03` | +0.09 % | **yes** |
| macro onset vs affine onset | 0.03991 | 0.051938 | −23.2 % | **no** — explained by the local concentration factor 1.305 (predicted 1/1.305 = 0.766 vs measured 0.768, i.e. 0.3 %); the affine value is not the applicable baseline for a driven block |
| driven apparent modulus `G_app` | 0.65–1.05 `C44` | `C44 = 7k/3a` | up to ±35 % and rate dependent | **no** — WITHHELD: the ramp is 20 % from rate convergence; no quantitative claim made |
| mode-II fracture energy | not defined | — | — | **no baseline exists** (model has no tangential bond stiffness); reported as NOT_SUPPORTED |

Every quantity that has a closed-form baseline is reproduced to machine
precision. The two comparisons that are *not* inside 3 % are a boundary-value
statement (driven block, no closed form available) and a rate-convergence
statement; both are reported as withheld rather than passed.

## 7. What this closes, and what it does not

Closed or quantified:

* the mapping from `shear_breaking_strain` to a macroscopic strain, with closed
  forms and machine-precision verification (§3.2);
* the fact that the criterion is deviatoric, not mode-II specific, and that under
  the shipped defaults it never governs uniaxial loading (§3.2);
* a *shear specimen protocol* that drives a bonded lattice to mode-II failure,
  with the local-criterion check and the concentration factor (§3.4);
* the quantitative statement that mode-II fracture energy is not representable in
  this bond law (§3.4), upgrading a prose caveat into numbers.

Still open:

* quasi-static apparent modulus of a bonded DEM block (needs ~10× slower ramps,
  or a servo-controlled boundary), and hence any geometric convergence study;
* a mode-II energy release rate would require a tangential bond spring (and a
  corresponding energy term) — a model change, not a verification task;
* comparison against a measured ice shear specimen (no source-traceable
  shear-strength dataset has been adopted for this lattice).