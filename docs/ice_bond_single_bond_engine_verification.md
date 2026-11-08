# Single-bond engine driver: the emergent rate exponent at the engine level

**Status: PASS.** The single-bond engine driver reproduces the analytic ductile
exponent ``1/(m+1) = 1/3`` to **0.075 %** for the linear+creep arm, closing the
gap-ledger "next node" left open by the specimen-level campaign.

## Why this exists

The specimen-level campaign (`docs/ice_bond_specimen_rate_dependence.md`,
mission m6464) was an **honest FAIL**: at 0.25–4 s⁻¹ a 4×4×3 opposed-platen
specimen is inertia-dominated (measured strength 7–9× the rigid-plate bound,
negative control not flat), so its emergent strength exponent could not be
compared to the analytic envelope. The gap ledger's next node was to *isolate
the constitutive response* so the inertial term no longer dominates.

This driver does that at the single-bond level:

* A 2-particle / 1-bond assembly is built from the real `IceDEM3D` engine
  (a 2×2×2 grid reduced to one bond), so the **engine's** call sites
  (`_bond_force` history advance, `_commit_fracture`, `_commit_creep_damage`
  with the engine's `F_ref`) are the ones under test — not the raw law in
  isolation.
* The bond extension is prescribed kinematically at a constant strain rate
  `edot`: `d(t) = edot·t·L0`. Because the extension is prescribed (not
  integrated), there is **no inertia** in the constitutive response; the
  measured failure force `F_f` is the pure constitutive failure force.
* The bond fails by whichever mechanism triggers first: brittle cleavage
  (`strain > eps_b`, mode 1) or Kachanov creep damage (`D ≥ 1`, mode 4).

## Analytic reference (linear bond + creep, exact)

With a linear ramp `F(t) = k·edot·L0·t` and `F_ref = k·eps_b·L0`
(`fraction = 1`), the Kachanov clock gives `D(t) = (1/t_ref)·(k·edot·L0/F_ref)^m·t^(m+1)/(m+1)`,
so `D = 1` at

```
t_f    = [(m+1)·t_ref·(eps_b/edot)^m]^(1/(m+1)),
F_f(edot) = k·L0·eps_b^(m/(m+1))·((m+1)·t_ref)^(1/(m+1))·edot^(1/(m+1)),
```

i.e. the emergent envelope is `F_f ~ edot^(1/(m+1))`. The brittle criterion
fails the bond at `F_b = k·eps_b·L0` (rate independent). The measured strength
is `min(F_b, F_f(edot))`; the crossover rate is `edot_c = eps_b/((m+1)·t_ref)`.

## Constants (single source of truth: the script)

| quantity | value |
|---|---|
| radius `R` | 0.025 m |
| rest length `L0` | 0.05 m |
| unrelaxed stiffness `k` | 2000 N/m |
| breaking strain `eps_b` | 0.015 |
| creep exponent `m` | 2 |
| creep time `t_ref` | 5×10⁻³ s |
| relaxation times | (5×10⁻³, 5×10⁻²) s |
| equilibrium / delayed fractions | 0.35 / (0.45, 0.20) |
| `dt` | 1×10⁻⁴ s |
| creep rates | 0.03, 0.1, 0.3 s⁻¹ |
| brittle rates | 1.0, 3.0 s⁻¹ |

## Results (from `artifacts/ice_bond_single_bond/verification.json`)

| arm | edot (s⁻¹) | F_f (N) | mode | analytic ref (N) | rel err |
|---|---|---|---|---|---|
| linear | 0.03–3.0 | 1.5000 | 1 (brittle) | F_b = 1.5000 | ≤ 0.02 % |
| linear+creep | 0.03 | 0.4662 | 4 (creep) | 0.4661 | +0.02 % |
| linear+creep | 0.1 | 0.6960 | 4 (creep) | 0.6962 | −0.03 % |
| linear+creep | 0.3 | 1.0050 | 4 (creep) | 1.0041 | +0.08 % |
| linear+creep | 1.0, 3.0 | 1.5000 | 1 (brittle) | F_b = 1.5000 | 0.00 % |
| maxwell_wiechert | 0.03→3.0 | 0.5619→1.2372 | 1 (brittle) | rate stiffening | — |
| maxwell_wiechert+creep | 0.03 | 0.3308 | 4 (creep) | — | — |
| maxwell_wiechert+creep | 0.1 | 0.5105 | 4 (creep) | — | — |
| maxwell_wiechert+creep | 0.3 | 0.7720 | 4 (creep) | — | — |

**Fitted exponents (log-log slope of F_f vs edot in the creep window):**

| arm | fitted exponent | analytic 1/(m+1) | rel err |
|---|---|---|---|
| linear+creep | **0.3336** | 0.3333 | **0.075 %** |
| maxwell_wiechert+creep | 0.3678 | 0.3333 | +10.3 % |

## Interpretation

* **Arm A (linear)** — negative control: rate-independent strength
  (spread < 0.02 %), all brittle. Confirms the driver itself introduces no
  spurious rate dependence.
* **Arm B (linear+creep)** — the closed-form anchor: the measured exponent
  **0.3336** matches the analytic **1/3** to **0.075 %**, and every point
  sits on the closed-form creep force to < 0.1 %. The crossover between creep
  and brittle failure occurs between 0.3 and 1.0 s⁻¹, consistent with the
  analytic `edot_c = 1.0 s⁻¹`.
* **Arm C (maxwell_wiechert)** — the relaxation spectrum stiffens the
  instantaneous response: the brittle strength rises from 0.56 N (slow) to
  1.24 N (fast), and the slow strength is below the unrelaxed brittle bound.
  This is the rate *stiffening* mechanism, distinct from creep damage.
* **Arm D (maxwell_wiechert+creep)** — the physical model: the creep-window
  exponent is **0.3678**, i.e. **above** the pure-creep 1/3. The relaxation
  spectrum adds rate stiffening on top of the creep damage, so the effective
  exponent is higher. This is a secondary observation (the MW force is not a
  linear ramp, so there is no strict closed-form reference); it is reported,
  not claimed.

## What this closes / what stays open

* **Closed:** the engine applies the rate-dependent bond law correctly at the
  single-bond level, in a regime where inertia is absent. The constitutive
  exponent is verified at the engine level, not just the raw-law level.
* **Still open (specimen level):** a bonded *specimen* (many bonds, load
  redistribution, inertia) has not yet been shown to follow the envelope.
  That needs either a quasi-static servo at 2–3 decades slower rates, or a
  single-*chain* driver (series bonds, weakest-link statistics). Both are
  larger than a single-bond driver and are the next node.

## Reproduction

```bash
PYTHONPATH=src python scripts/verify_ice_bond_single_bond_engine.py
PYTHONPATH=src python -m pytest tests/test_ice_bond_single_bond_engine.py -q
```
