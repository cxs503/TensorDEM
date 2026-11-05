# Lateral periodic closure: the direct-shear deficit is an edge effect, and `C44 = 7k/(3a)` is exact

Status: implemented in `src/tensordem/shear_periodic.py`, tested by
`tests/test_dem3d_shear_periodic_closure.py` (13 tests, ~8 s) and reproducible with
`scripts/audit_dem3d_shear_periodic_closure.py` (~37 s, single thread); generated
numbers in `docs/dem3d-shear-periodic-closure/`.

## 1. The debt this closes

`docs/dem3d_shear_static_reference.md` §5 left the material shear modulus open and named
two ways to extract it from the direct-shear specimen: *either* a cross-section much
wider than the boundary layer *or* periodic boundaries. The first route became
`docs/dem3d_shear_cross_section_convergence.md` (18 sizes up to `61×61×3`,
`d(n) = 1.191/n + 1.846/n²`, `G_eff(∞)/C44 = 0.99899`). That page also recorded the
open item this one settles:

> **`nz = 3` is load-bearing.** ... a height-independent material modulus still needs
> periodic boundaries, which the 3-D solver does not have.

This page is the second route: a **static, laterally periodic** closure of the same bond
network. It is a verification object, not a new solver: nothing in `dem3d.py`,
`shear_static.py` or the fracture/contact model is touched, and the time-domain solver
still has no periodic option. What it buys is a *falsifiable* statement about the
numbers the direct-shear specimen reports.

## 2. Method

| item | value |
| --- | --- |
| lattice | simple-cubic + 26-neighbour bonds, `a = 2r = 0.05 m` |
| cross-section | square `n × n`, `n ∈ {3,4,5,7,9,11,13,15,17,21,25,29,33,41,49,61}` |
| height | `nz = 3` (`h/a = 2`) for the size sweep; `nz ∈ {3..9}` for the height series |
| stiffness / density | `k = 2000 N/m`, `ρ = 917 kg/m³` (`DEM3DConfig()` defaults) |
| strain | `γ = 1e-3`, linear elastic, fracture disabled |
| closed form | `C44 = 7k/(3a) = 93.333 kPa`, `A_eff` census identity of `shear_static.py` |
| solver | one `scipy.sparse` equilibrium solve per case; `wrap` is a pure topology flag |

Boundary conditions are the solver's (`bottom` layer fixed in all three components,
`top` layer with `u_x = γh` prescribed only, everything else free). "Periodic" means the
neighbour image is wrapped in that lateral direction while the bond keeps its
**unwrapped** offset, i.e. minimum image for the central-force law; `z` is never
periodic because the top and bottom layers are real surfaces. `G_eff` divides the bottom
reaction by the bond-census area `A_eff = Σ l² n_x² n_z² / (7a²/3)` over the bonds that
leave the fixed layer, exactly as in `shear_static.affine_effective_area`.

### The claim, and why it can fail

Under lateral periodic closure every node owns its full, centrally symmetric
26-neighbour set, so the affine field `u_x = γz` is an *exact* solution of
`K u = 0` on the free DOFs. The bottom reaction is then
`R = Σ_nodes k Σ_upward γ dz n_x² = n² · (7/3) k γ a`, and `A_eff = n² a²`, hence

    G_eff / C44 = 1   exactly, at every n, every nz, with no fit.

That could fail in three distinct ways, and the tests check all three: a wrong bond
enumeration, a wrong normal/sign convention, or a wrong census would all break it. The
guard against a *self-consistent* bug is the last link of the chain:
**with the wrap disabled the module must reproduce
`shear_static.static_direct_shear`** — the reference that is itself cross-checked against
`IceDEM3D.forces` — for the same block. It does, to `1e-10` relative
(`tests/test_dem3d_shear_periodic_closure.py::test_wrap_off_reproduces_the_solver_reference`),
so the periodic number comes from the same physics as the free-surface numbers.

## 3. Results

Full tables: `docs/dem3d-shear-periodic-closure/{periodic_series,bc_variants,height_series}.csv`,
`summary.md`, `report.json`.

**(a) Size sweep with both lateral faces periodic** — `G_eff/C44 = 1` at every size:

| n | particles | `A_eff/A_nom` | `G_eff/C44` | deficit | pre-solve affine residual / `kγa` |
| --- | --- | --- | --- | --- | --- |
| 3 | 27 | 1.000000 | 1.000000000000 | 0 | 2.8e-15 |
| 5 | 75 | 1.000000 | 1.000000000000 | 0 | 2.8e-15 |
| 9 | 243 | 1.000000 | 1.000000000000 | -4.4e-16 | 2.8e-15 |
| 17 | 867 | 1.000000 | 1.000000000000 | 0 | 2.8e-15 |
| 33 | 3267 | 1.000000 | 1.000000000000 | +1.1e-16 | 2.8e-15 |
| 61 | 11163 | 1.000000 | 1.000000000000 | +4.4e-16 | 2.8e-15 |

max `|deficit| = 4.4e-16` over the 16 sizes; max affine residual `2.8e-15 kγa`;
free-DOF residual after the solve `≤ 3.5e-14 kγa` (against `7e-3 … 2e-2` for the
free-surface solves — the periodic system is far better conditioned).

**(b) Boundary-condition variants** — the mechanism:

| variant | n = 5 | n = 9 | n = 15 | n = 25 | pre-solve affine residual (n=5) |
| --- | --- | --- | --- | --- | --- |
| `periodic_x + periodic_y` | 0 | 0 | 0 | 0 | 2.8e-15 |
| `periodic_x + free_y` | 0 | 0 | 0 | 0 | 1.7e-15 |
| `free_x + free_y` | 0.312786 | 0.154276 | 0.087672 | 0.050976 | 2.33 |

So **one pair of free lateral faces costs exactly nothing** and two pairs cost the whole
deficit. The reason is visible in the missing-bond first moment: for a node on a single
free face the missing neighbours come in `±` pairs in the sheared (`x`) direction, so
`Σ dz n_x²` over the surviving upward bonds still equals `7/3` and the affine field
remains an equilibrium; at an **edge where two free faces meet** the missing set is
one-sided (`Σ_missing dx ≠ 0`), the affine field is no longer an equilibrium, and the
relaxation starts there. The `nz = 3` deficit of the cross-section study is therefore an
*edge* effect that decays as `1/n` along the faces — it is not a face effect and not a
census bias (`A_eff` already removes the census exactly, per bond).

**(c) Height independence** (`n = 9`, both lateral faces periodic):

| nz | 3 | 4 | 5 | 6 | 7 | 9 |
| --- | --- | --- | --- | --- | --- | --- |
| h/a | 2 | 3 | 4 | 5 | 6 | 8 |
| `G_eff/C44` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

max `|deficit| = 2.7e-15`. Under periodic closure the reported modulus has **no `h/a`
bias at all**, which is what makes the `nz = 3` caveat of the cross-section page a
statement about free surfaces rather than about the bond law.

**(d) The field itself** (`n = 25`, `nz = 3`, mid layer): with free lateral faces the
equilibrium field departs from the affine field by `7.7 %` of the driven displacement
`γh` in the edge zone, and that departure *is* the fitted boundary layer; under periodic
closure the same measure is `1.0e-15 γh`, i.e. the affine field is the equilibrium field
pointwise, not just in the reaction. Checked in
`tests/test_dem3d_shear_periodic_closure.py::test_periodic_field_is_the_affine_field_pointwise`.

**Figures** `docs/dem3d-shear-periodic-closure/figures/periodic_closure.png`: free-surface
deficit (committed study) vs the exact affine closure; the three boundary variants; and
the height series. `docs/dem3d-shear-periodic-closure/figures/field_map.png`: `(u_x - γz)/(γh)` over the mid layer for the
free and the periodic case.

## 4. What this does and does not change

- **Does change:** the asymptotic bias of the fitted cross-section law is now *known to
  be zero*, not merely estimated. The `c0 = 1.01e-3` of
  `docs/dem3d_shear_cross_section_convergence.md` is a fit artefact — it sits inside that
  fit's own `|residual| ≤ 1.0e-3`, and the periodic closure shows the true asymptote is
  `0` to `1e-16`. The material constant `C44 = 7k/(3a)` and the `A_eff` census are
  mutually consistent exactly, which is the precondition for the
  `shear_breaking_strain → γ_c` calibration of `docs/dem3d_shear_specimen.md` to be
  attached to a correctly normalised stress.
- **Does not change:** the free-surface numbers themselves (`0.687 → 0.980` over
  `n = 5…61`), the `nz = 3` restriction on *those* numbers, and the conclusion that a
  practical measurement needs `n ≥ 45` for `< 3 %` with free faces.
- **Not provided:** periodic boundaries in the **time-domain** solver `IceDEM3D`. This
  module is a static verification object; adding wrap to `forces()`/neighbour lists would
  be a model change with its own contact-bookkeeping questions and is out of scope.

## 5. Caveats and limits of validity

- **Linear elastic, no fracture, no contact, no rate.** The whole chain is the
  `γ → 0` limit on an intact network; `shear_breaking_strain` calibration and mode-II
  fracture energy remain untouched (mode-II is still NOT_SUPPORTED: no tangential bond
  stiffness).
- **Cubic-symmetry lattice.** `C44` is the affine constant of this bond network, not of
  isotropic ice; no Zener/directional claim is made here.
- **The `1/N` exponent used by the cross-section fit is not re-derived here.** What is
  established is the *value of the asymptote* (zero), not the analytic form of the
  approach; the edge-localisation in (b) explains qualitatively why it decays slowly
  along the faces.
- **Exactness is floating point.** The residuals quoted are `1e-15…1e-14` relative to
  `kγa`, i.e. one `spsolve` of a well-conditioned symmetric system; they are not a
  measure of physical accuracy, only of algebraic consistency.

## 6. Reproduce

```bash
PYTHONPATH=src python scripts/audit_dem3d_shear_periodic_closure.py   # ~37 s, 1 thread
PYTHONPATH=src python -m pytest tests/test_dem3d_shear_periodic_closure.py -q   # 13 passed
```