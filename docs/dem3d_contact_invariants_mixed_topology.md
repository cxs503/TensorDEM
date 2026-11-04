# DEM3D contact-kernel invariants on a MIXED bonded/unbonded topology (open finding)

## 1. Case / why this is a verification debt

`docs/dem3d_contact_invariants.md` §5 explicitly records the audit's "natural next
increment": the contact-kernel metamorphic audit (`tensordem.contact_invariants`)
covers only the two extremes — fully-bonded (intact) and fully-unbonded
(fractured, 326/326 bonds broken). A **mixed** topology (some bonds live, some
broken, on the same lattice) was never audited. That is the realistic case: a
partially-damaged ice lattice where the kernel must handle live bonds AND
unbonded contact pairs simultaneously, and where the bond/contact mutual
exclusion mask is exercised on a non-trivial subset.

## 2. What was added (minimal, default-preserving)

* `CaseSpec.fracture_fraction` (default `1.0` = original fully-fractured
  behaviour, bit-for-bit).
* `frozen_state()`: for `fracture_fraction < 1.0`, start from the intact bond
  set and kill a random fraction of the TOTAL bonds (private generator, seed+1000),
  guaranteeing a mixed topology regardless of how many the natural pass breaks.
* `CaseSpec("linear_mixed", ..., pre_fracture=True, fracture_fraction=0.5,
  perturbation=0.6)` added to `DEFAULT_CASES`.
* `tests/test_dem3d_contact_invariants.py`: two new tests —
  `test_mixed_topology_exercises_both_bond_and_contact` (asserts the mixed case
  carries BOTH live bonds and >10 unbonded contact pairs, and is NOT fully
  fractured) and `test_mixed_topology_all_invariants_pass`.

## 3. Result (honest)

`test_mixed_topology_exercises_both_bond_and_contact` **PASSES** — the mixed case
genuinely carries both live bonds and unbonded contacts (not vacuous).

But `test_mixed_topology_all_invariants_pass` **FAILS** on the mixed topology:
`brute_force`, `rotation`, `determinism` (and `translation`) do **not** satisfy
their declared tolerances on the mixed case, while they pass on the intact and
fully-fractured cases.

This is a **genuine open verification-debt finding**, not a claimed fix: the
contact kernel's bookkeeping invariants — which the audit certifies for the two
extreme topologies — **do not hold on a mixed bonded/unbonded topology**. The
`brute_force` failure means the engine's force vector disagrees with the
independent O(N^2) reference on a lattice that has both live bonds and unbonded
contacts, i.e. the bond/contact mutual-exclusion bookkeeping is inconsistent on
the mixed path. This is exactly the "silent, plausible, wrong" defect class the
audit exists to catch.

## 4. Error verdict / scope

The invariants are floating-point identities (tolerance 1e-12), not physical
predictions, so the 15% engineering band does not apply. The honest reading:
the mixed-topology increment is **NOT yet certified** — it exposes a real
bookkeeping inconsistency that must be root-caused (likely the bond/contact
exclusion mask or the `alive`-replay path in `IceDEM3D.forces`) before the
mixed case can be declared PASS. This is left as an open finding with a
reproducible red test, per the platform's honest-FAIL discipline.

## 5. Reproduce

```
PYTHONPATH=src python3 -m pytest tests/test_dem3d_contact_invariants.py -q
```
