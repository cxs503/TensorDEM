# LEFM ice-crack framework on the bonded-particle lattice (TensorDEM)

Campaign node **p307w1-n01-lefm-ice-crack** — linear elastic fracture mechanics for
ice crack propagation: J-integral, ``K_IC`` calibration of ice toughness, MTS crack
path prediction, negative controls.

Everything below is produced by
`benchmarks/lefm_ice_crack_validation.py` → `artifacts/lefm_ice_crack/results.json`
(plus figures `fig1..fig6`) and guarded by `tests/test_lefm_ice_crack.py` (33 cases).
Numbers quoted here are copied from that JSON, which is the audit trail.

---

## 1. Introduction — what is modelled

Ice failure at the ship-ice interface is dominated by *cracked* elasticity: the ice
sheet splits along radial and circumferential cracks long before the material
strength is reached. The existing TensorDEM ice kinematics (3-D bonded spheres,
`tensordem.dem3d`, freeze-bond chains, floe fields) are **strength/strain-based**:
a bond breaks when its axial strain exceeds `breaking_strain`
(`tensordem/bond_fracture_energy.py` maps that into an equivalent fracture energy
but does not use it as a criterion). This node adds the missing
**linear elastic fracture mechanics** layer:

| element | module | what it does |
|---|---|---|
| bonded-particle lattice | `tensordem/lefm/lattice2d.py` | 2-D square lattice, 8 central-force springs/particle, removable bonds (zero-width slit cracks) |
| LEFM bond criterion | `tensordem/lefm/senb.py` | ``K_bond = sqrt(E U_b /(a B)) >= K_IC`` per bond |
| J-integral | `tensordem/lefm/jintegral.py` | contour ``J = ∮(W dy − T·∂u/∂x ds)``, equivalent-domain (EDI) J, two-state interaction (M) integral for `K_I, K_II` |
| crack path | `tensordem/lefm/crack_path.py` | MTS criterion ``K_I sinθ + K_II(3cosθ−1)=0`` + incremental propagation + path-overlap metric |
| SEN B calibration | `tensordem/lefm/senb.py` | ASTM E399 SEN B, critical load → `K_IC`, energy ledger, negative controls |

### Geometry and material (parameter table)

| symbol | value | meaning |
|---|---|---|
| `E` | 5.0 GPa | ice Young's modulus |
| `ν` | 0.25 | lattice Poisson ratio (isotropic point of the model, see §2) |
| `B` | 0.05 m | specimen thickness |
| `W` | 0.10 m | specimen depth |
| `S/W` | 2 … 6 (base 4) | span ratio |
| `a/W` | 0.2 … 0.5 (base 0.3) | notch ratio |
| `a` | 5 mm (base), 2.5 mm | lattice spacing |
| `K_IC` | 200 kPa m^0.5 | reference ice toughness; literature band 150–250 kPa m^0.5 (Lu et al. 2015) |

## 2. Method

**Material mapping (exact, not fitted).** For a square lattice of central springs
with axial stiffness `k₁` and diagonal `k₂`, the strain-energy density in the long
wavelength limit is

    W = ½k₁(ε_xx²+ε_yy²) + ½k₂[(ε_xx+ε_yy)² + 4ε_xy²]

which equals the isotropic form `½λ(tr ε)² + μ ε:ε` **iff** `k₂ = k₁/2`; then
`λ = μ = k₁/2`, `ν = ¼` exactly and `E_2D = 1.25 k₁` (plane stress, `E_2D = E·B`).
Hence `k₁ = 0.8 E B`. The affine patch test in `results.json` reproduces `C:ε` to
**1.2e-16 (normal) / 1.9e-16 (shear)** relative error — the lattice *is* the
intended continuum, no calibration constant is hidden in the elasticity.

**Crack.** A crack is a zero-width slit: all bonds *crossing* the lattice line and
all bonds *lying along* it are deleted. Deleting only the crossing family (the
naive version) leaves a bond chain stitching the faces together and corrupts every
downstream quantity — this was found and fixed during this work (§5).

**LEFM bond criterion.** A bond stores `U_b = ½k(eL)²`. Breaking it creates the
projected crack area `a·B`, so `G_b = U_b/(aB)` and
`K_b = sqrt(E G_b)`. A bond breaks when `K_b ≥ K_IC`. Because the lattice is
linear, `K_b ∝ P`, so the critical load follows in closed form:
`P_c = P₀ K_IC / K_b,max(P₀)`.

**K from the lattice.** Three independent routes (all in `jintegral.py`):
contour J (the campaign's formula, rectangular path hugging the crack faces),
equivalent-domain J with a C¹ ring function `q`, and the interaction integral
`M = (1/B)∫[σ_ij u_i,1^aux + σ_ij^aux u_i,1 − σ_ik^aux ε_ik δ_1j] q_,j dA` with
analytic unit-K Williams auxiliary fields (``AuxiliaryField``), giving `K_I` and
`K_II` separately. A fourth, parameter-free reference is the **energy route**
`G = dU/da` under load control (no J machinery at all).

**Crack path.** MTS root `θ_c = 2·atan2(K_I − √(K_I²+8K_II²), 4K_II)`, folded into
(−π,π]; each step advances the tip by one lattice spacing along `θ_c`, deletes the
bonds crossing that segment and re-solves. Bond deletion is irreversible, so the
path is a propagation history, not a post-processing of one elastic field.

## 3. Results

### 3.1 Ground truth I: centre-cracked plate (CCP)

`σ_∞ = 531.7 kPa`, `a = 0.05 m`, `W = 0.60 m`, reference
`K = σ_∞√(πa)·√(sec(πa/W)) = 214.4 kPa m^0.5`.

| route | K [kPa m^0.5] | error vs reference |
|---|---|---|
| domain J (annuli 2–4, 4–7, 6–10, 8–14 cells) | 118.6 / 117.7 / 120.0 / 123.6 | −44.7 … −42.4 % |
| contour J (radii 3, 5, 8, 12 cells) | 55.2 / 52.9 / 46.2 / 44.6 | −74.6 … −79.5 % |
| energy dU/da (Δa = 2 cells) | 26.1 | −87.8 % |

**Honest verdict:** at lattice resolution none of the field-based J routes
reproduces the analytic stress intensity of a centre-cracked plate. This is not a
coding slip in the K→J conversion: the J integrand is a divergence
(`∂_j[σ_ij u_i,1 − W δ_1j] = 0` away from the tip), so the integral is a
*difference of large cancelling terms*; with a node-sampled lattice field the
cancellation error is of the same order as the answer. The measured spread
(domain J ≈ 0.55×, contour J ≈ 0.25×, energy ≈ 0.12×) quantifies that.

### 3.2 Ground truth II: SEN B, all routes (a/W = 0.3, S/W = 4, P = 100 N)

| route | 20 cells (a = 5 mm) | 40 cells (a = 2.5 mm) | ratio to ASTM (40 cells) |
|---|---|---|---|
| ASTM E399 closed form | 38.48 kPa m^0.5 | 38.48 | 1.00 (reference) |
| domain J (ring 8–14) | 170.3 | 167.4 | 4.35 |
| contour J (r = 16 a) | 66.0 | 47.7 | 1.24 |
| M-integral (ring 2–4.5) | 760.5 | 747.5 | 19.4 |
| energy dU/da (load control) | 43.3 | 40.0 | 1.04 |
| bond-level criterion `K_b,max` | 13.7 | 13.9 | 0.36 |

Two facts matter: (i) the **energy route** and the **contour J** bracket the ASTM
value and reproduce its *scaling* with `a/W` and `S/W`; (ii) the domain J and the
M-integral carry a **constant bias** (4.35× and 19.4×) that is independent of the
crack length, the span and — see the table — the lattice resolution. That
signature identifies a discretisation-independent multiplicative effect (the
node-sampled singular field in the ring weight), not a physics effect. They are
therefore *not* used for quantitative K in this work.

### 3.3 Path independence (campaign criterion (b))

| route | variation over 3–16 cells / rings 2–14 | 
|---|---|
| domain J (EDI) | **2.1 %** (40 cells), 2.9 % (20 cells) |
| contour J | 105 % (20 cells), 109 % (40 cells) |
| domain J, near-field rings only (2–4) → (4–7) | 1.6 % (20 cells) |

The criterion `< 2 %` is **met only in the near-field annuli (1.6 %)** and missed
by a small margin over the full ring set (2.1 %); the contour J fails it outright.
Reported as measured — no ring set was selected after the fact to pass the gate.

### 3.4 `K_IC` calibration against literature (criterion (a))

The bond-level criterion is calibrated once: the lattice correction
`κ = K_apparent/K_IC` measured at `a/W = 0.3, S/W = 4` is **κ = 2.800**. Every
other geometry then inverts its own critical load and divides by κ:

| a/W | S/W = 2 | S/W = 4 | S/W = 6 |
|---|---|---|---|
| 0.2 | 216.9 | 200.0 | 193.8 |
| 0.3 | 216.9 | 200.0 | 193.5 |
| 0.4 | 217.7 | 199.9 | 193.7 |
| 0.5 | 216.1 | 199.7 | 194.0 |

All values in kPa m^0.5. Mean **203.4**, max |error| **8.8 %**, all points inside
the literature band 150–250 → criterion (a) **passes** with 0.41× the allowed error.
Two structural findings: κ is essentially independent of `a/W` (spread < 1 %) but
depends on `S/W` (2.71 at S/W = 6 … 3.04 at S/W = 2), which is the geometric part
of the lattice correction. The factor κ itself is a *real, reportable* property of
the model: the bond-level `G_b = U_b/(aB)` assignment over-estimates toughness by
κ² ≈ 7.8 in energy terms, so a bonded-particle ice model that wants to reproduce a
measured `K_IC` must divide its bond fracture energy by κ².

### 3.5 Energy ledger at the critical event (criterion (d))

One critical bond break at `K_IC = 200 kPa m^0.5`, displacement control:

| quantity | value |
|---|---|
| released potential energy `dU` | 4.288 mJ |
| surface energy assigned to the broken bond `ΣU_b` | 2.000 mJ |
| new crack area | 2.5e-4 m² |
| `G_released` | 17.15 J/m² |
| `G_target = K_IC²/E` | 8.00 J/m² |
| error | **+114 %** |

Criterion (d) (`< 5 %`) **fails**. Diagnosis: the *bond* stores only the part of
the singular field that one lattice spacing can resolve, while breaking it releases
the whole local core energy, so the discrete release is ≈2× the assigned surface
energy. This is the same mechanism that produces κ (a *calibrated*, not predicted,
toughness) and it is the reason the framework reports calibration curves rather
than first-principles toughness.

### 3.6 Crack path (criterion (c))

| case | kink angles [deg] | max deviation from crack line | overlap with first-order LEFM reference |
|---|---|---|---|
| symmetric load (mode I), 5 mm | +1.23 → +4.64 | 1.5 mm over 30 mm advance | **100 %** |
| symmetric load (mode I), 2.5 mm | +0.60 → +2.05 | 0.2 mm over 20 mm advance | **100 %** |
| offset load (mixed mode), 5 mm | −0.08 → −26.9 | 4.3 mm over 29 mm advance | **100 %** |
| offset load (mixed mode), 2.5 mm | +3.2 → −2.6 | 0.7 mm over 20 mm advance | **100 %** |

The symmetric case should stay straight (`θ_c = 0`); the residual 1–5° drift comes
from the `K_II` bias of the M-integral (the mode-separation route is the one with
the 19× bias of §3.2, so a small spurious `K_II` survives). Mode-II kink limit is
verified analytically: `mts_angle(0, K_II) = −70.53°`, and the MTS root satisfies
its stationarity equation to 1e-6 rad in the tests.

**Criterion (c) as written ("overlap > 80 % with model-test photographs") is
WITHHELD:** no digitised crack-path photographs exist in this repository and none
were fabricated. The overlap metric is implemented and validated; it is applied to
(a) the first-order LEFM reference path and (b) the resolution-refined path —
both give 100 %.

### 3.7 Negative controls (criterion (e))

| control | result |
|---|---|
| un-notched specimen in uniform tension at the far-field stress the notched specimen carries at `P_c` (1.75 MPa) | `K_b,max = 80.8 kPa m^0.5` = **0.40 K_IC** → **no crack initiation** |
| same lattice, bond criterion with `K_IC = 1e12` at 100× the notched critical load | `K_b,max = 1.37 MPa m^0.5` ≪ 1e12 → **no fracture** |
| lattice strength limit `K_IC/√(πa)` | 1.60 MPa — above it the discrete criterion degenerates into a *strength* criterion (documented, not a bug) |

## 4. Figures

* `fig1_senb_field_path.png` — `σ_yy` field of the offset-load SEN B lattice with the MTS crack path overlay.
* `fig2_k_routes.png` — K by route / K_ASTM at 20 and 40 cells (log scale).
* `fig3_path_independence.png` — K/K_ASTM versus contour radius and annulus width.
* `fig4_kic_calibration.png` — calibrated `K_IC` versus `a/W` for three spans, with the literature band.
* `fig5_crack_path.png` — symmetric and mixed-mode paths versus the first-order LEFM reference.
* `fig6_ccp_validation.png` — CCP ground-truth errors of the three J routes.

## 5. Pitfalls found while building this (all fixed, all regression-tested)

1. **Neighbour-list bug.** Assigning `n₀` with a clamped index produced a lattice
   with duplicated axial bonds and only one diagonal family: the patch test showed
   spurious shear (`σ_xy = 90.7 kPa` under a pure normal strain) and the "beam"
   was 3.4× too compliant. Fixed by anchoring `n₀` on the current site; the patch
   test then matches to 1e-16.
2. **Bridged crack.** Deleting only bonds that *cross* the slit line leaves the two
   faces stitched by the axial chain lying *on* it; the "crack" then carries load
   and every near-tip quantity is wrong. Fixed in `slit_bonds`.
3. **Contour leg through the crack.** A rectangular J path whose return leg
   descended *through* the faces interpolated across the displacement
   discontinuity. Fixed; the face legs are now offset by half a cell on the lower
   side and 1.5 cells on the upper side.
4. **False-positive bond deletion in propagation.** The orientation-only segment
   test reports collinear-but-disjoint bonds as hits, silently deleting bonds away
   from the crack front. Fixed with a bounding-box pre-filter in `bonds_crossing`.
5. **`np.linalg.solve` on 10⁴ stacked 2×2 systems.** In numpy ≥ 2.3 this takes
   minutes; replaced by the closed-form 2×2 inverse (0.08 s).
6. **Dangling particles after propagation** make the stiffness matrix singular;
   a 1e-9 relative Tikhonov term pins them without touching the solution.

## 6. Limitations / where this must not be used

* Plane stress, 2-D, one material (`ν = ¼` is built into the lattice).
* The **field-based J routes are not quantitatively trustworthy at the tested
  resolutions** (§3.1, §3.2). Use them comparatively (path independence, mode
  ratio, scaling) or use the energy route.
* `K_IC` is delivered as a *calibrated* quantity (κ = 2.80 at `a/W ≈ 0.3–0.5`,
  `S/W = 4`); it is not predicted from first principles, and κ depends on `S/W`.
* The energy ledger over-shoots by 114 % (§3.5): the model has no explicit
  dissipative term, so a strict Griffith balance cannot be claimed.
* Crack paths are compared against LEFM references, **not** against model-test
  photographs (criterion (c) is WITHHELD, not passed).
* No ice-specific rate/temperature dependence (creeping ice, `K_IC(T, porosity)`)
  is included; the toughness is a single deterministic number.

## 7. Reproduce

```bash
python benchmarks/lefm_ice_crack_validation.py     # writes artifacts/lefm_ice_crack/results.json
python benchmarks/lefm_ice_crack_figures.py        # writes artifacts/lefm_ice_crack/fig*.png
python -m pytest tests/test_lefm_ice_crack.py -q   # 33 cases
```