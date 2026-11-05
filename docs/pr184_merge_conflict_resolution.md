# PR #184 (agent/m3129) — main-merge cure report

**PR**: #184 · **branch**: `agent/m3129` · **repo**: TensorDEM
**Symptom**: `mergeable=false` against `main`.
**Outcome**: semantic-union merge produced; core test suite at or above the `main`
baseline; no content from either side silently dropped.

| ref | commit | note |
|---|---|---|
| merge base | `f5de971` | last shared ancestor |
| `main` (ours) | `e12a2ab` | Merge #203 (m3317) via merge-governor |
| `agent/m3129` (theirs) | `134010a` | old SHA of the PR branch head |
| merged result (this work) | cure `HEAD` of `agent/m3973`/`agent/m3129` | merge commit, parents `6c6f98b`(⊃`e12a2ab`) + `134010a` |
| new `agent/m3129` | same cure `HEAD` | branch ref fast-forwarded to the merge commit |

Because the merged commit has `134010a` as a parent, merging this cure back into
`main` makes the PR branch head an ancestor of `main` — which is exactly the
condition pr-gate / merge-governor needs to record PR #184 as merged.

---

## 1. Conflict inventory (complete)

`git merge-tree --write-tree main agent/m3129` reported **one** conflicted path:

| # | file | kind | changed by main | changed by m3129 |
|---|---|---|---|---|
| 1 | `src/tensordem/contact_restitution.py` | content conflict | adds `contact_damping_model="absolute"` to `DEM3DConfig(...)` inside `measure_dem3d_impact()`, with a comment explaining the clipped pair law | adds the *same* kwarg with a different comment explaining that the `"restitution"` default rebuilds the damper from `restitution_coefficient` (1.0 → zero damping) and that the convention must be explicit, mirroring `contact_impact._build_pair` |

Non-conflicting m3129 addition (auto-merged, preserved byte-identically):

* `scripts/verify_restitution_closed_loop.py` (141 lines, new file)

Main-side-only changes auto-merged without conflict: 18 further paths
(`benchmarks/`, `docs/`, `scripts/`, `src/tensordem/fracture_gen.py`,
`src/tensordem/dem3d_wedge.py`, `artifacts/pr166/*`, …), i.e. 20 paths in total
differ between the merge result and the m3129 side, and all of them are *main's*
new content.

### Why this conflict is dangerous

Both sides added the **same functional line**, so git could have produced two
outcomes that look fine in review but are not:

* `--ours` / `--theirs` overlay: drops the other side's rationale comment (the
  comment is the only thing that documents *why* the absolute convention is
  mandatory); no diff-time signal.
* naive "keep both hunks": duplicates the keyword argument —
  `f(x=1, x=1)` is a **SyntaxError**, so the module no longer imports (verified as
  negative control C below).

## 2. Per-file resolution

### `src/tensordem/contact_restitution.py`

Semantic union: keep **one** kwarg, merge **both** rationale comments into one
coherent argument. The code line is identical on both sides, so no behaviour is
at stake; the comment carries both parties' reasoning:

```python
config = DEM3DConfig(
    nx=2, ny=2, nz=2,
    radius=radius, density=density,
    contact_stiffness=k_n, contact_damping=gamma_n,
    # DEM3DConfig defaults to the legacy "restitution" dashpot convention, in
    # which ``contact_damping`` does not enter the particle-particle pair law
    # at all: the damper is rebuilt from ``restitution_coefficient`` (default
    # 1.0 -> zero damping), which would silently make every measured impact
    # elastic.  This harness installs an *absolute* dashpot gamma_n [N s/m],
    # so the pair law must be the clipped F = max(k delta - gamma_n v_n, 0)
    # that the reference integration in this module solves, and the
    # convention has to be explicit -- mirroring ``contact_impact._build_pair``.
    contact_damping_model="absolute",
    drag=drag, fix_x_edges=False,
    tool_speed=1e-3, tool_gap=1.0,
)
```

* from **main**: the clipped-law argument (`F = max(k·δ − γ_n·v_n, 0)`, matched to
  the RK4 reference integration in the same module);
* from **m3129**: the failure mode of the default convention
  (`restitution_coefficient` default 1.0 → zero damping → every impact measured
  as elastic), the m3129 wording that the dashpot is *absolute* [N s/m], and the
  `contact_impact._build_pair` cross-reference.

### Union audit (mechanical, not by eye)

`scripts/verify_pr184_merge_union.py` re-checks the union without git history:

| check | result |
|---|---|
| conflict markers in module (`<<<`, `===`, `>>>`) | none |
| `contact_damping_model` keywords in `measure_dem3d_impact` | 1 (was 2 before resolution) |
| convention value | `"absolute"` |
| main rationale fingerprints present | 3/3 |
| m3129 rationale fingerprints present | 3/3 |
| `scripts/verify_restitution_closed_loop.py` present & parseable | yes |

Diff-level audit against both parents:

* `git diff main..HEAD --stat` → exactly 2 files: the m3129 script (141 lines
  added) and `contact_restitution.py` (13 lines: comment block only). So nothing
  from main was lost.
* `git diff agent/m3129..HEAD --stat` → the 20 main-side paths only; **no** m3129
  path is deleted or reverted in the merge result.
* Line-set audit: each side contributed exactly **one** new code line vs the merge
  base (`contact_damping_model="absolute",`); the merged file contains it exactly
  **once** — semantic union, no duplication, no loss.

## 3. Test evidence

Python 3.11, `torch==2.14.1`, CPU, `python3 -m pytest -q -p no:randomly`.
Baseline = an unmodified `main` tree extracted with
`git archive main | tar -x -C /tmp/base_main`; the merged tree is the worktree.

| # | suite (core cluster) | merged (cure `HEAD`) | baseline (`main`, `e12a2ab`) |
|---|---|---|---|
| A | `test_dem3d_restitution_verification`, `test_contact_impact_dashpot_convention`, `test_contact_tangential` | 31 passed (23.1 s) | 31 passed (24.8 s) |
| B | `test_dem3d_contact_impact`, `test_dem3d_contact_invariants`, `test_dem3d_hertz_contact` | 47 passed + 44 subtests (31.8 s) | 47 passed + 44 subtests (33.9 s) |
| C | `test_dem`, `test_dem3d` | 31 passed, 1 skipped + 8 subtests (18.8 s) | 31 passed, 1 skipped + 8 subtests (18.1 s) |
| D | `test_dem3d_pair_contact_damping`, `test_dem3d_restitution`, `test_restitution`, `test_friction_cone_verification`, `test_rolling_resistance`, `test_dem3d_stability` | 204 passed + 26 subtests (84.5 s) | 204 passed + 26 subtests (90.9 s) |
| E | `test_dem3d_platen_compression`, `test_dem3d_platen_contact_mask`, `test_dem3d_bond_fracture_energy`, `test_dem3d_integration_order`, `test_dem3d_lattice_elastic`, `test_dem3d_lattice_elasticity3d` | 58 passed (28.9 s) | 58 passed (57.2 s) |
| F | `test_hull_contact*`, `test_moving_plane*`, `test_verification` | 65 passed + 48 subtests (37.4 s) | 65 passed + 48 subtests (28.9 s) |
| G | **new** `tests/test_pr184_merge_union.py` (union pin) | 5 passed (6.9 s) | n/a (new file) |
| Σ | core cluster (A–F) | **436 passed, 1 skipped, 0 failed** | **436 passed, 1 skipped, 0 failed** |

Δ vs baseline: **0** — the merged tree is not below the `main` baseline on this
cluster, and no test that passes on `main` fails after the merge.

Scope honesty: the repo collects **1234** tests in 109 files. The full-suite run
was started on both trees but the pytest processes were killed by the harness
process-group reaping while the machine was simultaneously running ~8 other
agents' suites; it was replaced by the 436-test cluster above (contact /
restitution / friction / lattice / hull-contact / core 2-D–3-D solver — i.e. the
entire surface the conflicted file can influence). Files unrelated to normal
contact (`fracture_gen`, `wake_lbm`, buoyancy, wedge, ship-ice baselines) were
left to CI, which also runs on the merge.

### Negative controls for the new pin test (do the tests actually bite?)

| mutation of the merged file | expected | observed |
|---|---|---|
| C: flip value to `contact_damping_model="restitution"` | detect | 3 failed (`test_single_absolute_dashpot_convention`, `test_both_parents_rationale_survive`, `test_absolute_convention_reproduces_clamped_reference`) |
| B: strip m3129's clause from the merged comment | detect | 1 failed (`test_both_parents_rationale_survive`) |
| A: keep both hunks (duplicate kwarg) | detect | collection error — `SyntaxError: keyword argument repeated`, i.e. the module would not import at all |

## 4. Physics re-verification (merged module, real 3-D solver)

`scripts/verify_pr184_merge_union.py` (numbers in
`artifacts/pr184/union_audit.json`):

| quantity | value |
|---|---|
| target restitution `e_target` | 0.600000 |
| calibrated absolute dashpot `gamma_n` [N·s/m] | 6.266030 |
| damping ratio `zeta` | 0.180858 |
| solver-measured `e_measured` | 0.597486 |
| clamped RK4 reference `e_reference` | 0.600000 |
| **relative error** | **0.419 %** (gate 3 %) |

Wider sweep, from m3129's preserved evidence generator
(`scripts/verify_restitution_closed_loop.py` →
`artifacts/restitution_closed_loop_m3129/evidence.json`,
figure `restitution_closed_loop.png`):

| `gamma_n` | `zeta` | solver `e` | clamped reference | rel. err | classic closed form | classic rel. err |
|---|---|---|---|---|---|---|
| 1.0 | 0.0289 | 0.915808 | 0.914802 | 0.110 % | 0.913278 | 0.167 % |
| 3.0 | 0.0866 | 0.774698 | 0.772607 | 0.271 % | 0.761050 | 1.496 % |
| 5.0 | 0.1443 | 0.658328 | 0.659722 | 0.211 % | 0.632434 | 4.136 % |
| calibrated → e=0.6 | 0.1809 | 0.597486 | 0.600000 | 0.419 % | 0.632947 | 5.491 % |

Equal-mass pair (`m_eff = m/2`): solver 0.786562 vs reference 0.783533
(0.387 %). Timestep refinement: 0.463225 → 0.462447 (0.168 %, converging).

## 5. Notes / caveats

* The code semantics of the two sides were **identical** on the conflicted line;
  what the conflict destroyed was documentation, not behaviour. The union
  therefore costs no numerical risk, and the physics checks above confirm the
  merged module still obeys the clipped impact law (if the `"restitution"`
  default had been used by mistake, `e_measured` would be ≈ 1.0 and
  `test_absolute_convention_reproduces_clamped_reference` fails — control C).
* `docs/dem3d_contact_impact_dashpot_convention.md` (main side) documents the
  same convention from the engine's perspective; the merged comment now cites the
  `contact_impact._build_pair` mirror explicitly, so the two remain linked.
* No `.git` configuration, no push and no history rewrite was performed from
  inside this mission; the branch move `agent/m3129: 134010a → cure HEAD` is a
  fast-forward of a branch ref whose tree content is a strict superset of both
  parents.

## 6. Reproduce

```bash
git merge --no-ff agent/m3129            # 1 conflicted file
# resolve src/tensordem/contact_restitution.py as in §2
python3 scripts/verify_pr184_merge_union.py      # union + physics audit
python3 -m pytest tests/test_pr184_merge_union.py -q
python3 scripts/verify_restitution_closed_loop.py  # m3129 evidence + figure
python3 -m pytest tests -q                       # (long; cluster in §3 is the gate)
```