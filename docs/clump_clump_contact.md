# Clump–clump contact with geometric junction merging

Status: **implemented and verified** — campaign node **P16-N3** (mission m5186).
Evidence level: **REAL** (Hertz–Mindlin contact law, merged narrow phase,
third-law/angular-momentum closure, 1e5-step stability all machine-checked).
Module: `src/tensordem/clump_clump.py`; tests:
`tests/test_clump_clump_contact.py`; quantitative evidence:
`artifacts/clump_clump_contact/criteria.json` + figures, produced by
`benchmarks/clump_clump_contact_verification.py`.

This closes the `clump-clump contact` row that
[`docs/clump_multisphere_particles.md`](clump_multisphere_particles.md) listed as
`NOT_SUPPORTED`, and the corresponding gap in
[`docs/dem_calibration_gap_ledger.md`](dem_calibration_gap_ledger.md)
(GAP-DEM-G4, "friction between two clumps has no path").

## 1. What the problem is

A clump is a rigid union of overlapping spheres. Before this node, TensorDEM
could only touch a clump against a wall (`PlaneContact`) or an infinite-mass
sphere (`SphereObstacle`): both are one-sided, so the reaction force had no
second body to act on, no second inertia tensor and no second torque. Two clumps
require a two-sided interaction in which **one** force and **one** contact point
produce equal and opposite forces *and* torques about two different centres of
mass.

The second, less obvious problem is **duplicate sampling**. Because the members
of a clump overlap, several member pairs `(i in A, j in B)` can indent the *same*
physical junction — the two unions touch over one lens-shaped region that happens
to be sampled twice (or four times). Summing the per-pair forces then counts one
contact twice: for two symmetric duplicate pairs the repulsion is exactly `2x`
the physical value, which inflates the contact stiffness, the restitution
behaviour and the effective friction of the whole packing.

## 2. Method

### 2.1 Narrow phase: candidates, then junctions

1. `member_pair_candidates(body_a, body_b)` enumerates every member pair with
   strictly positive penetration
   `delta_ij = r_i + r_j - |c_j - c_i|` and returns, per candidate,
   the contact point `x = c_i + (r_i - delta/2) n` (mid-point of the overlap
   along the line of centres), the unit normal `n` pointing from A into B, the
   effective radius `R* = r_i r_j / (r_i + r_j)` and the Hertz patch radius
   `a = sqrt(R* delta)`.
2. `merge_candidates` clusters candidates with **single linkage** under the
   criterion

   ```text
   |x_p - x_q| <= merge_factor * (a_p + a_q)      and      n_p . n_q >= cos(merge_angle)
   ```

   i.e. two candidates are the same junction when their *contact patches*
   overlap and their normals agree. The default is `merge_factor = 1.0`,
   `merge_angle = 30 deg`. Single linkage makes the result independent of the
   pair enumeration order.
3. Each cluster becomes one `ContactJoint`. Its force magnitude uses the
   **deepest** candidate of the cluster (largest penetration) — adding the
   cluster's penetrations would put the double counting back inside the force
   law — while its point and normal are the penetration-weighted means, so a
   symmetric duplicate pair reproduces the symmetry axis exactly
   (`|n - e_y| <= 1e-15` in the tests).

The patch-based criterion is the clump analogue of the contact clustering used
for polyhedra (Bourrier et al. 2013): two indentation regions closer than their
own patch size are physically one contact.

### 2.2 Contact law (Hertz–Mindlin, shared with `tensordem.hertz_mindlin`)

```text
F_n^elastic = (4/3) E* sqrt(R*) delta^{3/2}        Hertz
k_n(delta)  = 2 E* sqrt(R* delta)                  tangent stiffness
c_n         = 2 zeta sqrt(m_eff k_n(delta))        dashpot, zeta from the log-decrement law
F_n         = max(F_n^elastic - c_n v_n, 0)
k_t(delta)  = 8 G* sqrt(R* delta)                  Mindlin no-slip
F_t         = -k_t xi,   |F_t| <= mu F_n           Cundall–Strack history, cone projection
```

with `1/m_eff = 1/m_A + 1/m_B + n.[(I_A^-1 (a_A x n)) x a_A] + n.[(I_B^-1 (a_B x n)) x a_B]`
(the same `effective_mass_along` helper the clump–wall path uses; `a_A`, `a_B`
are the contact-point arms). `E*` and `G*` are the reduced moduli of the two
materials; the formulas are identical to
`hertz_mindlin.hertz_reduced_shear_modulus` and reproduce
`hertz_mindlin.mindlin_stiffness_ratio` (`k_t / secant-k_n = 6 G*/E*`) to
`1e-12`. They are evaluated in NumPy here so the clump path keeps no torch
dependency; the equality with the torch module is asserted in
`test_single_member_clump_matches_hertz_mindlin_material_constants`.

Contact-point velocities come from `ClumpBody.point_velocity` (the arm is the
contact point, *not* the member centre — the defect class documented for the
clump–wall path). The implementation caches the rotation/inertia per evaluation
in a private `_Kinematics` record; `test_cached_kinematics_matches_clump_body_api`
pins bitwise equality with the public `ClumpBody` methods.

The tangential spring is stored per junction, in the world frame, keyed by the
**deepest member pair** of the junction; it is projected back onto the friction
cone after every slip increment and dropped when the junction closes.

### 2.3 Third law and angular momentum are structural, not approximate

A junction publishes one point `x` and one force `F`. Body B receives `+F` with
torque `(x - x_B) x F`; body A receives `-F` with torque `(x - x_A) x (-F)`. The
sum of the two contributions to the moment about the origin is

```text
x_A x F_A + tau_A + x_B x F_B + tau_B = x x F - x x F = 0
```

identically, because the arms differ by exactly the vector between the two
centres of mass. This is why the measured residuals are at round-off
(`1.7e-16` relative for the moment, `1.2e-14` for linear momentum) rather than at
truncation-error order.

Rolling/drilling resistance (implemented for clump–wall in
`src/tensordem/clump_rolling.py`) is deliberately **not** applied to clump–clump
contacts: a per-member internal couple has no third-law partner and would break
the identity above.

## 3. Using it

```python
from tensordem.clump import Clump, SphereMember
from tensordem.clump_dynamics import ClumpBody, PlaneContact
from tensordem.clump_clump import ClumpClumpConfig, ClumpClumpContact

cfg = ClumpClumpConfig(young_modulus=1e8, poisson_ratio=0.3,
                       restitution=0.8, friction=0.4)   # merging=True by default
contact = ClumpClumpContact(cfg)          # owns the tangential history

for step in range(n_steps):
    # one velocity-Verlet step for the pair; optional one-body contacts
    # (e.g. a floor) are folded into the same force evaluations
    contact.step(body_a, body_b, dt, extra_b=(floor,))

F_on_a, T_on_a = contact.last.force_a, contact.last.torque_a
energy = contact.stored_energy()          # Hertz + Mindlin elastic energy [J]
audit = duplicate_force_ratio(body_a, body_b, cfg)
```

Static queries: `member_pair_candidates`, `merge_candidates`,
`clump_clump_joints`, `hertz_normal_force`, `pair_material_constants`,
`total_linear_momentum`, `total_angular_momentum`.

Setting `merging=False` (or `merge_factor=0`) reproduces the naive member sum;
that switch *is* the negative control of the verification below.

## 4. Verification (campaign criteria C1–C6)

All numbers below are copied from
`artifacts/clump_clump_contact/criteria.json` (generated, never typed by hand).
Configuration: two-sphere clumps of radius 50 mm, density 900 kg/m³,
`E = 100 MPa`, `nu = 0.3`, `E* = 54.95 MPa`, `G* = 11.31 MPa`.

| id | criterion | measured | verdict |
|---|---|---|---|
| C1 | one-member clump == sphere–sphere, relative error <= 1e-12 | `0.0` (12 penetration/velocity combinations); pair integrator vs `ClumpBody.step` in free flight: `0.0` m | PASS |
| C2 | two-sphere clump contact within 5 % of an independent geometric estimate | two junctions: `5.4e-16`; **four overlapping member pairs**: `3.96e-3` (0.40 %) | PASS |
| C3 | merged force <= 1.05x the unique-junction force; naive member sum ~2x | merged/unique = `1.000000`; naive sum/merged = `1.9872` (peak over the compression sweep 1.99) | PASS |
| C4 | third law <= 1e-12, momentum <= 1e-12 | role-swapped force residual `0.0`; net moment `1.7e-16`; linear momentum `1.2e-14`; angular momentum (same collision) `7.2e-12` | PASS |
| C5 | angular momentum <= 1e-10, restitution deviation <= 0.05 | `1.14e-11` relative over an oblique frictional spinning collision; worst restitution deviation `+0.0236` (targets 0.5/0.7/0.9/0.95) | PASS |
| C6 | resting run over 1e5 steps: drift <= 1e-9 m, energy non-increasing | drift `0.0` m; relative energy change `0.0`; a 5 % deeper perturbed start settles back (offset `2.7e-6 -> 2.0e-7` m) with monotonically decreasing energy | PASS |
| C7 | m4950 regression `tests/test_clump_multisphere.py` | 36 passed, unchanged (no edits outside the new module/tests) | PASS |

Figures (`artifacts/clump_clump_contact/`):

| file | shows |
|---|---|
| `c1_hertz_force.png` | junction force on the analytic Hertz curve over five decades of penetration |
| `duplicate_overcount_vs_penetration.png` | merged force vs the naive member sum during quasi-static compression; the overcount ratio rises to ~1.99 |
| `c3_duplicate_control.png` | force / unique-junction force against `merge_factor` (acceptance band 1.05, naive sum 1.987) |
| `merge_geometry.png` | the two duplicate candidate points, their patch circles and the single merged junction |
| `c5_restitution.png` | measured vs requested restitution with the ±0.05 band |
| `c6_stability.png` | centre-of-mass drift over 1e5 steps and the total energy budget |

Why the C6 test uses a *loaded* stack: a purely repulsive contact cannot rest in
zero gravity — the pair simply flies apart — so the resting state has to be a
load path. The benchmark therefore rests clump A on clump B on a plane; the
plane overlap (`9.246e-7` m at `k_n = 1e7` N/m) and the Hertz pair overlap
(`5.421e-5` m) are solved in closed form so that the initial state is statically
admissible to round-off, and only then is drift a meaningful spurious-motion
detector.

## 5. Limits and known gaps

* **Merging replaces the multi-contact force by one force at the deepest
  indentation.** For a junction sampled by members with different penetrations
  the merged force is the single-junction value, which is ~1-3 % below an
  area-weighted multi-Hertz integral for heavily curved unions (the four-pair
  case above measures 0.40 % against the scalar member sum). This is deliberate
  — the alternative is the non-physical doubling.
* **The merge tolerance is heuristic.** `merge_factor = 1.0` (patch-sum length)
  was chosen so that (i) genuinely distinct junctions of a dumbbell (0.1 m
  apart, patch radius 1.6 mm) stay separate and (ii) nearly concentric duplicate
  pairs (1.3 mm apart, patch radius 17 mm) merge. For very elongated clumps with
  a deeply indented contact, two real junctions can be closer than their patch
  radii and would merge; `merge_factor` is exposed for such cases, and
  `test_over_merging_is_detectable` pins the failure mode.
* **Tangential history is keyed by the deepest member pair.** If the deepest
  member of a merged junction switches between steps, that junction's tangential
  spring restarts. The resulting slip is bounded by `mu F_n / k_t` per switch
  (sub-micrometre here), but the effect is not quantified for rapidly rotating
  contacts.
* **No rolling/drilling resistance, no adhesion/cohesion, no liquid bridge** on
  the clump–clump path (see §2.3 for why rolling resistance is excluded).
* **Two bodies per interaction.** There is no broad phase, neighbour list or
  many-body loop: the caller drives pairs. Cost is ~0.3-1 ms per step for a small
  two-clump pair on the LoongArch node used for verification (240 s for 1e5
  steps including a floor contact), which is Python-loop overhead, not physics.
* **Not validated against experiment.** All references here are closed-form
  (Hertz, Mindlin, momentum conservation). There is no laboratory
  clump-clump restitution/force measurement in this repository; the restitution
  band ±0.05 is a model-calibration property, not an experimental agreement
  claim.

## 6. Reproduction

```bash
# unit/integration tests (torch-free; the hertz_mindlin cross-check needs torch)
python -m pytest tests/test_clump_clump_contact.py tests/test_clump_multisphere.py -q

# full campaign evidence (writes artifacts/clump_clump_contact/)
python benchmarks/clump_clump_contact_verification.py --steps 100000

# regenerate only the figures from an existing criteria.json
python benchmarks/clump_clump_contact_verification.py \
    --reuse-json artifacts/clump_clump_contact/criteria.json
```

Environment note: the verification run recorded in `criteria.json` was executed
on the 702 cluster node `n315` (LoongArch, Python 3.11.6, NumPy 2.0.1). Those
nodes currently cannot import torch (`libflexiblas.so.3` missing), so
`tensordem/__init__.py` cannot be imported there either; the run used a small
package shim that registers `tensordem` with `__path__` pointing at `src/`, which
is legitimate because this module and its tests are NumPy-only. The torch
cross-check (`test_single_member_clump_matches_hertz_mindlin_material_constants`)
therefore passes locally and is skipped/failed by the environment on the cluster.