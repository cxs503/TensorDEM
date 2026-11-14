# Kerr three-parameter foundation for ice-ridge keel grounding: reduction, verification, granular cross-check

Status: **new capability** — before this page the package had a Winkler beam
(`ridge_geotech.solve_beam_on_winkler`, Hetenyi closed forms), a Winkler plate
(`plate_on_foundation`) and an *empirical* keel resistance law
(`ice_ridge_keel_freezebond.kerr_keel_resistance`, Kerr 1975), but no member that
represents a **layered** rubble keel: a loose, soft surface layer over a
compacted, shear-coupled core.  That is the three-parameter Kerr foundation.
Nothing here is a rule-based number: the two numerical routes, the footing closed
form and the bonded-particle cross-check are all reproduced by code and tested.

Everything below is produced by

```
PYTHONPATH=src python scripts/gen_w27_ice_ridge_keel_anchor.py
PYTHONPATH=src python scripts/run_w27_ridge_keel_grounding.py
PYTHONPATH=src python scripts/figures_w27_ridge_keel.py
PYTHONPATH=src python -m pytest tests/test_ridge_keel_grounding.py -q
```

Numbers live in `papers/artifacts/w27/ice-ridge-keel-grounding/ridge_keel_anchor.json`
and the ISO 19906 A.8 register in `benchmarks/anchors/ice-ridge-keel-grounding.json`;
this page states the *method* and the interpretation, and never restates a number
by hand.

## 1. Reduction

Upper Winkler layer `p = k_u w_u`, lower Pasternak layer
`p = k_l w_l - G w_l''`, series coupling `w = w_u + w_l`:

```
p - c p'' = K w - G2 w'',      K = k_u k_l/(k_u+k_l),   c = G/(k_u+k_l),   G2 = k_u G/(k_u+k_l)
```

* `k_u -> inf` ⇒ `c -> 0`, `K -> k_l`, `G2 -> G` (Pasternak, two parameters),
* `G -> 0` ⇒ `p = K w` (Winkler, one parameter; series springs give `K`).

With an Euler-Bernoulli beam (`EI w'''' + p = q`, `w` and `q` downward positive,
`p` upward positive) and `p = q - EI w''''` the governing equation is the sixth
order ODE

```
EI w'''' - c EI w'''''' - G2 w'' + K w = q - c q''
```

whose symbol is `P(lambda) = -c EI lambda^6 + EI lambda^4 - G2 lambda^2 + K`.

## 2. Two independent numerical routes

* **Closed form** (`kerr_beam_analytic`): symbol inversion.  For a uniform patch
  load the solution is `w = u - c u''` with `u = L^-1 q` obtained from the exact
  antiderivative of the Green function `G(x) = sum_j exp(lambda_j |x|)/P'(lambda_j)`
  over the three roots with `Re(lambda_j) < 0`.  No quadrature, no assembled
  matrix.
* **Mixed Galerkin FEM** (`kerr_beam_fem`): Hermite cubics for `w` plus
  piecewise-linear `p`, assembled from the *two weak forms*

  ```
  int EI w'' dN'' + int p N  = int q N
  int p L + int c p' L'      = int K w L + int G2 w' L'
  ```

  i.e. never from the elimination used by the closed form.  The natural
  (free-end) conditions follow from the weak form.

A third, fully external check is used as well: in the `k_u -> inf`, `G -> 0`
limit the FEM must reproduce the **Hetenyi (1946)** patch-load closed form that
already existed in `ridge_geotech` — it does, to `1e-8` relative.

## 3. Rigid footing on a Kerr layer

A rigid footing of width `b` settling by `w0`: `w = w0` under the footing, free
surface (`p = 0`) outside, and the weak-form jump `c p' = G2 w'` at the edges.
The result is a pressure that rises towards the edges,

```
p(x) = K w0 + (w0 sqrt(G2 K / c) / sinh(b/(2 sqrt(c)))) cosh(x/sqrt(c)),   |x| <= b/2
w(x) = w0 exp(-(|x| - b/2)/L0),   L0 = sqrt(G2/K)                          outside
```

and a total resistance

```
R = K w0 b + 2 w0 sqrt(G2 K)                                   [N/m]
```

The `sqrt(c)`-dependent amplitude cancels in the integral: **the footing test
cannot separate `k_u` and `k_l`** (only `K` and `G2` are identifiable), which is
why the reduction of the granular specimen fixes `k_u = k_l = 2K`.  The closed
form is verified inside `kerr_footing_selfcheck` against (a) the Kerr law inside,
(b) the free-surface equation outside, (c) the edge jump, (d) the integral.

## 4. Bonded-partite keel specimen (`keel_foundation_dem`)

Plane strain, unit thickness, jittered triangular bonded lattice at solid
fraction `phi = 1 - n`, one bond per neighbour pair within
`1.06 (r_i + r_j)`, bond axial stiffness `E_ice w_b / l0` and shear
`0.25` of it.  Quasi-static loading is dynamic relaxation (velocity Verlet,
implicitly damped, unit masses: equilibrium only, no inertia claim).  Lateral
boundaries are periodic so the specimen is a *patch* of an extended keel.

Because the tested strain is `~4e-3`, no contact is created or lost: the frozen
bond topology is the contact problem, not an approximation of it.  That is what
makes 25k / 50k / 100k particles affordable — one step is a handful of vector
operations on the bond arrays.

Three tests are extracted from one specimen:

| test | measured | use |
|---|---|---|
| `confined_compression` (wide lid) | `p(w)` curve, subgrade modulus `K` | series modulus of the Kerr model |
| `simple_shear` (lid sheared) | skeleton shear modulus `G_s` | diagnostic; `G_s` is stiff relative to the footing edge term and is *not* used for the prediction |
| `footing_test` (rigid footing, width `b`) | total resistance `R(b)` at settlement `w0` | identification (`b = 0.5 H_k`) and predeclared validation (`b = H_k`) |

## 5. What the comparison does and does not prove

The two-point identification (`K` from the confined test, `sqrt(G2 K)` from one
narrow footing) followed by an out-of-sample prediction at a second footing width
tests the *functional form* `R = A b + B` of the reduction, not the absolute
modulus of ice (the DEM modulus is a lattice modulus, and the Kerr `K` is taken
from the same specimen).  What it does catch is a wrong reduction: a one-parameter
subgrade predicts `R = K w0 b` with no intercept, a wrong layering predicts a
non-linear `R(b)`, and either would miss the predeclared 15 % band.

Limitations (also in the artifact): small-strain only, no bond breakage, no
contact creation, plane strain, periodic lateral boundaries, and the Kerr layer
cannot resolve the `k_u`/`k_l` split from a footing test.

## 6. ISO 19906:2019 A.8

The standard's clause text is **not available in this environment**; the anchor
`benchmarks/anchors/ice-ridge-keel-grounding.json` registers it `WITHHELD` with
the machine record of the acquisition probe (filesystem by name and by content,
extraction toolchain, publisher endpoints) and registers the in-tree secondary
citations by path + sha256 only, with their numbers deliberately not extracted.
Only the **operator layer** is machine checked
(`pressure_area_operator_checks`: anchor point, log-log slope `-n`, ratio and
composition identities, force-law exponent `1 - n`, inversion round trip,
`n -> 0` limit, linearity in `p0`), and every identity holds for arbitrary
`(p0, A0, n)`, so a future transcription can be re-checked with the same
function and no number needs to be invented now.  No computed quantity in this
capability depends on a standard value; A.8 enters only through its operators and
through the applicability statement (**continuous ice, not rubble**).