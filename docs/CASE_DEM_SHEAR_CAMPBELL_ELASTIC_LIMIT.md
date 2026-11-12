# CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT — elastic-limit shear-stress scaling (TensorDEM)

**Status (round 1): honest FAIL — the physics gates reject every point; the
Coefficient leg is additionally WITHHELD because the reference tables are behind
the publisher paywall.** Mission m16136, case pool 2026-11-12.
Engine: `tensordem.elastic_limit_shear` (new module, plane Couette cell of soft
disks). Driver: `benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py`.
Artifacts: `artifacts/case_dem_shear_campbell_elastic_limit/` (`point_*.json`,
`scaling.json`, `reference_evidence.json`, `fig_tau_star_kappa.png`).
Tests: `tests/test_elastic_limit_shear.py` (6 passed).

---

## 1. Problem and reference

Dense granular shear flows in the **elastic limit** — persistent interparticle
contacts, particle elasticity a first-order rheological parameter. The measured
object is the dimensionless shear stress

    tau* = tau / (rho d^2 gdot^2),        kappa = k_n / (rho d^3 gdot^2)

so that the classic **Bagnold / inertial** behaviour is `tau* = const`, i.e. the
shear-rate exponent `n = d ln tau / d ln gdot = 2`, and the
**elastic-quasi-static** behaviour is `tau ~ k_n` (`tau* ~ kappa`), i.e. `n = 0`.
Because `n = 2 (1 - beta)` with `beta = d ln tau* / d ln kappa`, a stiffness
sweep at fixed shear rate samples exactly the same function as a shear-rate
sweep at fixed stiffness.

**Reference.** Campbell, C. S. (2002) *Granular shear flows at the elastic
limit*, J. Fluid Mech. **465**, 261–291, DOI 10.1017/S002211200200109X.

**Reference-access result (zero hand-copy discipline).** The DOI resolves and
the publisher landing page was fetched on 2026-11-12, but the full text is
CLOSED: Unpaywall returns `best_oa_location: null`, SemanticScholar returns
`openAccessPdf.status = CLOSED`, `cambridge.org/.../article` returns 404 to the
anon crawler, and no institutional copy is mounted. **The numerical tables and
figures of pp. 261–291 are therefore not readable in this sandbox.** Everything
quoted below is from the *publisher abstract* and from one *open-access
secondary source*; both texts, with their SHA-256, are stored in
`artifacts/.../reference_evidence.json`.

* Publisher abstract (landing page, quoted): "There are two regimes at this
  limit, one for which the stresses scale with both elastic and inertial
  properties (called the elastic–inertial regime), and a non-inertial
  quasi-static regime in which the stresses scale purely elastically
  (elastic–quasi-static). … Reducing the concentration slightly causes a
  transition from an elastic to a purely inertial behaviour. This transition
  occurs so abruptly that a 2 % concentration reduction can be accompanied by
  nearly three orders of magnitude of stress reduction. … there is no path
  between inertial (rapid) flow and quasi-static flow by varying the shear rate
  at a fixed concentration".
* Secondary (arXiv:1712.02693, open access, quoted): "These two dimensionless
  numbers I and κ are equivalent to φ and k/ρd³γ̇² from Campbell's work …
  Campbell presented his regime map based on a series of DEM simulations of
  homogeneous shear flow … the cases that has κ > 10⁴ … the µ(I) model has been
  proved to be valuable."

**A-priori criterion set (declared before any run was inspected).**

| id | quantity | reference value | tolerance | source |
|---|---|---|---|---|
| R1 | Bagnold exponent `n` in the inertial limit | 2.0 | 15 % rel. | Bagnold (1954) + Campbell abstract (inertial branch, `tau ~ rho d^2 gdot^2`) |
| R2 | regime boundary `kappa_c` (elastic-inertial → inertial) | 1e4 | order of magnitude (factor 3), **not** the 15 % rule | secondary OA source quoting Campbell's map |
| R3 | elastic-quasi-static exponent `n` (`tau* ~ kappa`) | 0.0 | absolute 0.3 | Campbell abstract (stresses scale purely elastically) |
| R4 | Bagnold coefficient `tau*` plateau | **WITHHELD** | — | Campbell pp. 261–291 not accessible; no number is hand-copied |

R1/R3 are exponent-level statements, which are the robust part of the reference;
R4 (the coefficient) is exactly the leg the paywall kills, and it is declared
WITHHELD rather than filled from memory.

## 2. Method and implementation

New module `src/tensordem/elastic_limit_shear.py`:

* **Cell** — two-dimensional bi-disperse disks (50/50 by number, diameter ratio
  1.4) in a plane Couette cell: rough walls (a row of fixed disks, 10 % overlap,
  a wall friction coefficient of the same kind as the bulk), bottom row moving at
  `-V`, top row at `+V`, periodic in `x`; the gap (hence the bulk solids
  fraction) is **constant** — a volume-controlled shear, the same control
  Campbell used.
* **Contact law** — linear spring-dashpot normal force with restitution-matched
  damping plus a Coulomb-capped tangential spring with per-pair history (the
  standard soft-sphere law used to explore the elastic limit). Sliding
  dissipation is accumulated.
* **Starting state** — jittered triangular lattice at the target solids fraction
  followed by a POCA-style overlap relaxation at fixed volume; the resulting
  configuration is overlap-free (`max_initial_overlap/d <= 0.05`, pinned by
  test), so no elastic prestress is injected.
* **Measurement** — wall momentum fluxes `tau = -F_x(top)/Lx`,
  `P = F_y(top)/Lx` (both walls cross-checked); no peculiar-velocity
  bookkeeping. `tau*`, `kappa`, `I = gdot d / sqrt(P/rho)`, coordination `Z`,
  granular temperature and the dissipation rate are reported per point.
* **Physics gates (part of the contract)** — (i) energy balance
  `|tau gdot A - dissipation| / |tau gdot A| <= 0.30`, (ii) `max_overlap/d <=
  0.60`, (iii) steady-state drift of `|tau|` over the averaging window `<=
  0.30`. A point that fails any gate is **not** admitted to the fit.

## 3. Parameters

| quantity | symbol | value |
|---|---|---|
| dimensionality | — | 2-D disks |
| bulk solid fraction (constant volume) | phi | 0.800 |
| number of disks | N | 100 (ladder), 200/400 (cross-check) |
| diameter, small / big | d_s, d_b | 1.00 / 1.40 (ratio 1.4, 50/50) |
| reference diameter (number average) | d | 1.20 |
| areal density | rho | 1.0 |
| wall velocity (each wall) | V | 4.0 |
| wall-to-wall gap | H | 16.0 d_s |
| shear rate | gdot = 2V/H | 0.50 |
| normal stiffness ladder | k_n | 1e2 … 1e6 |
| dimensionless stiffness | kappa = k_n/(rho d^3 gdot^2) | 2.31e2 … 2.31e6 |
| tangential/normal stiffness | k_t/k_n | 0.50 |
| interparticle + wall friction | mu | 0.50 |
| restitution | e | 0.70 |
| timestep | dt = 0.05 T_contact | 1.2e-3 (k_n = 1e4) |
| accumulated strain per point | gamma | 1.5 (ladder) |
| inertial number | I = gdot d sqrt(rho/P) | 1.0e-2 … 2.0e-1 (see table) |

## 4. Reference vs measured (per point, with pass marks)

| tag | kappa | tau | P | tau* | T | Z | max overlap/d | energy-balance err | steady drift | gate | R1 (n vs 2) | R3 (n vs 0) | R4 (coeff) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kn1e2 | 2.31e2 | 1.311 | 25.9 | 3.64 | 29.8 | 1.12 | 0.95 | 163 | n/a | FAIL | ✗ not identifiable | ✗ | WITHHELD |
| kn1e3 | 2.31e3 | −4.055 | 185 | 11.3 | 455 | 0.47 | 0.95 | 901 | 0.25 | FAIL | ✗ | ✗ | WITHHELD |
| kn1e4 | 2.31e4 | 0.449 | 565 | 1.25 | 4.8e3 | 0.09 | 0.68 | 3.8e4 | 0.20 | FAIL | ✗ | ✗ | WITHHELD |
| kn1e5 | 2.31e5 | −6.77 | 2.9e3 | 18.8 | 5.0e4 | 0.04 | 0.46 | 2.6e4 | 0.38 | FAIL | ✗ | ✗ | WITHHELD |
| kn1e6 | 2.31e6 | −132 | 2.0e4 | 367 | 5.3e5 | 0.02 | 0.39 | 2.1e4 | 0.06 | FAIL | ✗ | ✗ | WITHHELD |

* `tau*` is **not monotonic** in `kappa` and `tau` **changes sign** between
  rungs, so no exponent can be extracted: R1 and R3 are *not measurable* on this
  data (✗), and the scaling fit is refused by the driver (`scaling.json`
  contains `fewer than 3 points passed the physics gates`).
* R4 (coefficient) is WITHHELD for the independent reason that the reference
  tables are inaccessible.
* Error columns are therefore **not** reported as numbers: quoting a percentage
  against a value that is either undefined (R1/R3) or WITHHELD (R4) would be
  fabrication. **No量 is <15 %; nothing is <3 %; the case does not pass.**

Headline honest statement: **the round-1 kernel does not reproduce the
elastic-limit steady state at all**, so it cannot confirm or refute Campbell's
Bagnold/inertial exponent.

## 5. Convergence / sensitivity (and why they cannot be used yet)

Two declared sensitivity axes were exercised on the aggregation path:

1. **Stiffness ladder k_n = 1e2 … 1e6** at fixed `gdot`, phi (the main axis,
   section 4): `T` scales as `T ~ k_n` (29.8 → 5.3e5, ratio 1.8e4 for a
   stiffness ratio 1e4) — a signature of a *fixed geometrical penetration*
   `delta` being converted into kinetic energy (`v ~ delta sqrt(k_n/m)`), not of
   thermostatted shear flow.
2. **System size N = 100 / 200 / 400** and **gap/wall speed** were exercised in
   the exploratory runs (`artifacts/.../sweep_kn.json`, N = 200): the pathology
   is size independent (coordination `Z` collapses to 0.01–0.05 and the
   overlap/dissipation gates fail identically), which is why the ladder is run
   at N = 100 and the size axis is *not* reported as a convergence study.

A convergence study with respect to `dt`, `N` and strain window is **not
meaningful until the energy defect is fixed**: the gates reject the points, so
the numbers are noise, not a converged state.

## 6. Failure modes, root cause, and artifacts

Observed failure modes (all reproducible from the driver):

1. **Wall-layer penetration / crush.** `max_overlap/d` grows to 0.4–0.95 within a
   strain of O(1) even though the start is overlap-free — particles are driven
   into the wall disks and into each other. The first revision was worse (an
   affine-stretched lattice was wrapped onto itself, producing 1.2 d overlaps
   and an exponential blow-up; the overlap-free POCA start removed that, but not
   the driven crush).
2. **Energy-balance violation of 2–5 orders of magnitude.**
   `tau gdot A` and the accumulated contact dissipation differ by 10²–10⁴,
   i.e. the wall does work that is not dissipated by contacts: the closed
   momentum-flux measurement and the dissipation accounting disagree, which also
   means the sign/normalisation of `tau` cannot be trusted at this stage.
3. **Coordination collapse.** `Z` → 0.01–0.05 from an overlap-free dense start
   (φ = 0.80, `Z_init ≈ 1.2–1.4`), i.e. the cell is being evacuated toward the
   walls instead of sustaining a dense sheared layer — the opposite of the
   elastic limit.
4. **Sign flips / non-monotonic `tau*(kappa)`** — a direct consequence of 1–3.

Most probable root causes to attack in round 2 (declared, not yet verified):

* the **wall boundary condition** (kinematic rows of disks with a memoryless
  Coulomb friction) is too stiff/too smooth: the required fix is either a
  servo-controlled wall (constant normal load with a load-controlled platen, as
  in `tensordem.shear_cell`) or a **homogeneous SLLOD/Lees–Edwards** cell with a
  proper peculiar-velocity stress estimator;
* the **normal damping / tangential history** bookkeeping under deep
  penetration (an overlap of 0.4–0.9 d is outside any linear-spring validity
  window, so the restitution matching is meaningless there);
* the driving amplitude `V = 4` at `k_n <= 1e4` gives `kappa` in a range where
  the *contact duration* exceeds the mean free time, i.e. the runs are not in
  the regime the module claims to probe.

Reproduce everything (single command per point, then aggregation):

```
python benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py \
    --tag kn1e4 --kn 1e4 --n 100 --strain 1.5 --average-from 0.4
python benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py --aggregate
python -m pytest -q tests/test_elastic_limit_shear.py
```

Artifacts: `point_kn*.json` (one per rung, all diagnostics + gates),
`scaling.json` (fit refusal + local slopes when available),
`reference_evidence.json` (reference quotes + SHA-256),
`fig_tau_star_kappa.png` (gated points blue, gate-rejected red).
Runtime: 5-rung ladder ≈ 90 s wall clock on 128 cores (single point ≤ 25 s),
so no cluster submission was needed.

### Ecosystem friction log (as required)

1. **Reference paywall** — JFM 2002 is closed; the platform has no institutional
   access path and no local PDF mirror. A "zero hand-copy" rule plus a paywalled
   primary source makes the coefficient leg unachievable by construction; the
   case pool should record the access status of every reference DOI, otherwise
   every pool case with a paywalled source burns a round on discovery.
2. **Knowledge base offline** — `GET /api/v1/tensor/knowledge/search` returns
   `engine_offline` (`127.0.0.1:8021` refused), so the mandated 三查
   (knowledge_base capability / negative-knowledge / oss_solutions) could only be
   done partially (`negative-knowledge/search?q=TensorDEM` → 0 hits,
   `stats` reachable). Domain prior art had to be re-derived from the repo
   (`mu_i_rheology`, `nonlocal_granular`, `chute_flow`).
3. **Installed package shadowing** — `import tensordem` in the shared env
   resolves to `work/m14461/src/tensordem`, i.e. *another worktree*; every run
   here therefore sets `sys.path[0]` to this worktree's `src` explicitly
   (`run_case.py` does). Cross-worktree import shadowing is a live footgun for
   any benchmark that trusts the ambient environment.
4. **No existing elastic-limit capability to reuse** — `mu_i_rheology` is a
   constant-normal-stress μ(I) cell (it cannot sweep `kappa` at fixed φ) and
   `chute_flow` is gravity-driven; the closest reusable pieces were the contact
   law and the wall-servo of `shear_cell`, which is 3-D and load-controlled.
   Hence a new module, as the mission's "reuse-or-build" rule allows, but the
   round-2 fix should reuse `shear_cell`'s servo platen rather than re-derive
   one.

## 7. What would make round 2 pass

1. Replace the kinematic wall by a **load-controlled servo platen** (reuse
   `tensordem.shear_cell`'s `servo_mode="bedref"` actuator) or move to a
   **SLLOD/Lees–Edwards homogeneous cell** with a verified peculiar-velocity
   stress estimator (the earlier SLLOD attempt is documented in the module
   history and failed for the same energy reason, so the estimator must be
   gated by the wall-free energy balance).
2. Gate the *start*: require `Z_init >= 3.0` at φ = 0.80 (a genuinely dense
   packing) before any strain is applied — today's relaxed lattice gives 1.2.
3. Declare the wall slip as a first-class gate (`V_slip/V <= 0.1`): without it
   the imposed `gdot` is nominal, not actual.
4. Only then extract `n(kappa)` and compare R1 (`n = 2`, 15 %) / R3 (`n → 0`)
   and, if an accessible copy of Campbell's tables is ever found, R4.