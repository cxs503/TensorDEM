# Independent audit: point-particle CFD-DEM self-response vs the Hasimoto periodic-array baseline

**Repo / module**: TensorDEM · `tensordem.point_cfd_dem` (untouched public API;
no module code changed by this audit except one corrected docstring claim)
**Mission**: m16963 (self-start, proposer seat) · branch `agent/m16963`
**Anchor**: `NEW` — no `feature_ledger` row covers the point-particle
`point_cfd_dem` verification surface (ledger rows are the `CASE-DEM-*` /
`GAP-DEM-*` families; `GAP-DEM-G1` is recorded as "3-D coupling not
implemented", which is stale — see §5.1). Proposed new entry at the end.
**Artifacts**: `benchmarks/point_particle_cfd_dem/hasimoto_array.py`,
`benchmarks/point_particle_cfd_dem/hasimoto_array/{result.json,tables.md,figures/*.png}`,
tests `tests/test_point_cfd_dem_hasimoto_array.py`
**Reproduce**:
```
PYTHONPATH=src python benchmarks/point_particle_cfd_dem/hasimoto_array.py          # ~4 min
PYTHONPATH=src python benchmarks/point_particle_cfd_dem/hasimoto_array.py --quick  # operator sweep only, <5 s
PYTHONPATH=src python -m pytest tests/test_point_cfd_dem_hasimoto_array.py -q
```
Environment: loongarch64, numpy 2.x, single core (no cluster job — all runs are
< 5 min single-core, load15 of the host was 92/128 so no parallel sweep was
submitted).

---

## 1. Case description (physics background / geometry / parameters)

`tensordem.point_cfd_dem` is the unresolved (point-particle) two-way coupling of
TensorDEM: an Eulerian pseudo-spectral incompressible solver plus Lagrangian
point particles whose drag is an analytical closure. Its first verification gate
(`verification.stokes_terminal_velocity_case`) settles one sphere in a triply
periodic box and compares the container-frame velocity with the **unbounded**
Stokes terminal velocity

  `v_t = (rho_p - rho_f) g d^2 / (18 mu)`  (d = 1, rho_p/rho_f = 2, mu = 0.5, g = 0.5 → v_t = 0.0556).

The gate passes at 1.984 % because a "self-response" correction
`C_eff = C / (1 - C R)` cancels the velocity the particle induces at its own
position (`SpectralPeriodicFluid.self_response`). The gate docstring justifies
this with the claim that the *physical* finite-domain effect is
"`3 pi d / L ~ 0.3 %`" at `L = 32 d` and therefore negligible.

That claim is testable against an independent literature anchor. A sphere in a
cube of side `L` with periodic boundary conditions is equivalent to a **simple
cubic array** of equal spheres of solid fraction `phi = (pi/6)(d/L)^3`, and the
dilute lattice sum (Hasimoto 1959; coefficients cross-checked by Sangani &
Acrivos 1982) gives a drag enhancement

  `F / (6 pi mu a U) = 1 + 1.7601 phi^(1/3) + phi - 1.5593 phi^2 + O(phi^(8/3))`

so the container-frame settling velocity of the discretised system is

  `U_array = v_t / (1 + C_H)`,  `C_H = 1.7601 phi^(1/3) + phi - 1.5593 phi^2`.

Over the audited range (`phi <= 8.2e-3`) the neglected `3.979 phi^(8/3)` term is
below `1e-6` and the retained `phi^2` term below `1.1e-4`, so the series is
exact to better than 0.1 % of `C_H` (asserted in the test file).
At `L = 32 d` this gives `C_H = 0.0443` — i.e. **4.43 % below `v_t`**, 15x larger
than the 0.3 % the module claims.

| quantity | value |
|---|---|
| particle diameter `d` | 1 |
| density ratio `rho_p/rho_f` | 2 |
| fluid viscosity `mu` (operator sweep) | 0.5 |
| fluid viscosity `mu` (settling runs) | 5 (chosen so that `L^2/nu` is affordable single-core; `Re_t = 7e-4`, Stokes regime preserved) |
| gravity `g` | 0.5 |
| grid spacing `h` | `2 d` (gate-1 regime) and `d` (beyond the correction's stability limit) |
| box `L/d` | 4, 6, 8, 12, 16, 24, 32 |
| particle sub-grid position | cell centre `(0.5,0.5,0.5)` and off-centre `(0.23,0.37,0.11)` |
| time step (settling) | 0.005 |

## 2. How it was done (methods / settings / tool chain)

Three independent measurements, no module behaviour modified:

1. **Operator sweep (Q1, no dynamics).** The module's steady self-response `R`
   (`fluid.self_response(pos, dt, elapsed=None)`) is evaluated over
   `h ∈ {2d, d}` x `L/d ∈ {4…32}` x two sub-grid positions and expressed in drag
   units, `C R` with `C = 3 pi mu d`, then compared with `C_H(phi)` at the same
   solid fraction. `C R` is exactly the fractional drag amplification the
   implicit closure applies (`C_eff = C/(1-CR)`), and the stiffness guard in
   `particles.evaluate_forces` (raises when `C R >= 0.95`) fixes where the
   correction is even usable.
2. **Long-time settling (Q2/Q3).** The existing gate stops at `t = 2` while the
   box viscous time is `L^2/nu = 2048` at `L = 32 d`: the periodic back-flow is
   only 0.1 % developed there, so the gate cannot see any finite-domain effect.
   The audit instead integrates `6 L^2/nu` (99.75 % developed), `dt = 0.005`,
   and takes the last 20 % time window. Three modes:
   `none` (no self-correction, `self_disturbance=False`), `spectral` (module
   default), `array_subtracted` (`R_eff = max(R - C_H/C, 0)`, i.e. remove only
   the numerical self-field and leave the physical array back-flow in the drag).
3. **Tool chain.** numpy/matplotlib/pytest, `PointParticleCFDDEM` with
   `zero_net_flux=True` (container frame), `exchange_terms=("drag",)`,
   `drag_filter_width=None` — identical to the gate configuration except
   `h/L/mu/t_end`.

## 3. Results (all numbers programmatically emitted to `result.json` / `tables.md`)

### 3.1 Q1 — the self-response is a grid-scale field, not the physical lattice sum

| L/d | h/d | cells | frac | phi | C R | C_H | C R / C_H |
|---|---|---|---|---|---|---|---|
| 8.0 | 2.0 | 4 | centre | 1.0227e-03 | 0.4147 | 0.1784 | 2.3 |
| 12.0 | 2.0 | 6 | centre | 3.0301e-04 | 0.4829 | 0.1185 | 4.1 |
| 16.0 | 2.0 | 8 | centre | 1.2783e-04 | 0.5158 | 0.0888 | 5.8 |
| 24.0 | 2.0 | 12 | centre | 3.7876e-05 | 0.5477 | 0.0591 | 9.3 |
| 32.0 | 2.0 | 16 | centre | 1.5979e-05 | 0.5633 | 0.0443 | **12.7** |
| 32.0 | 2.0 | 16 | off-centre | 1.5979e-05 | 0.3048 | 0.0443 | 6.9 |
| 32.0 | 1.0 | 32 | centre | 1.5979e-05 | 1.1609 | 0.0443 | 26.2 |
| 8.0 | 1.0 | 8 | centre | 1.0227e-03 | 1.0199 | 0.1784 | 5.7 |

Three properties separate `C R` from the physical array enhancement `C_H`:

* **Magnitude**: at the gate's own grid (`h = 2d`, `L = 32 d`) `C R = 0.5633`,
  i.e. **12.7x** `C_H = 0.0443`; the closure therefore doubles the drag
  coefficient (`C/(1-CR) = 2.29 C`) to cancel an effect that is 4.4 % in reality.
* **Grid scaling**: halving `h` at fixed `L = 32 d` multiplies `C R` by
  `1.1609/0.5633 = 2.06` (`~1/h`). The physical lattice sum has no `h`
  dependence at all.
* **Sub-grid position dependence**: at `L = 32 d`, `h = 2d` the value changes
  from 0.5633 (cell centre) to 0.3048 (off-centre), a factor 1.85; the physical
  array back-flow is uniform over the cell by construction.

Additionally the stiffness guard puts the correction outside its own stability
window as soon as `h = d`: `C R = 0.954 / 1.020 / 1.161` at `L/d = 6 / 8 / 32`
(raises `ValueError`), so the correction is confined to `h >= 2 d`, where it is
purely grid-scale.

### 3.2 Q2/Q3 — fully developed settling, `L = 8 d` (`phi = 1.0227e-3`, `C_H = 0.1784`, Hasimoto `U/v_t = 0.8486`)

| L/d | mode | U/v_t measured | Stokes 1.000 | Hasimoto 0.8486 | error vs Stokes | error vs Hasimoto | steps | wall |
|---|---|---|---|---|---|---|---|---|
| 8.0 | none | **1.2421** | 1.0000 | 0.8486 | +24.21 % | **+46.36 %** | 15360 | 116 s |
| 8.0 | spectral (module) | **0.9114** | 1.0000 | 0.8486 | −8.86 % | **+7.39 %** | 15360 | 121 s |

* The **uncorrected** model over-settles by 24 % against Stokes and 46 % against
  the array solution: the numerical self-field of the smeared point force is a
  large, *positive* contribution of the same order as the physical array effect.
* The module's **spectral** correction pulls the result back towards Stokes but
  does not reproduce either anchor: 8.86 % below Stokes, 7.39 % above Hasimoto.
  The correction is not exact at long time — it is exact only in the gate's
  short-time window (`t = 2 = 0.001 L^2/nu`) where the back-flow is absent.
* `fig2_settling_transients.png` shows both transients against the two anchors.

## 4. Figures

* `figures/fig1_self_response_vs_hasimoto.png` — `C R` vs `C_H` over `phi`
  (log-log), `h = 2d` and `h = d`: the module's curve is flat/growing while the
  literature array correction decays as `phi^(1/3)`.
* `figures/fig2_settling_transients.png` — `|u_p|/v_t` history for the audited
  modes with Stokes and Hasimoto levels at `L = 8 d`.
* `figures/fig3_settling_vs_phi.png` — steady `U/v_t` vs `phi` (single audited
  box; marked as such — see §5.4).

## 5. Points of attention (traps, limitations, scope)

1. **`GAP-DEM-G1` in `docs/dem_calibration_gap_ledger.md` is stale.** It records
   "3-D coupling not implemented" open; in fact `tensordem.point_cfd_dem`
   (triply periodic spectral, adjoint particle-grid transfer, 3-D) and
   `tensordem.lbmdem` (resolved IBM) both exist. The real remaining gap is not
   "3-D" but *walls* and *Model A* (see the module's own §4 limitations) and the
   array-consistency question audited here. The ledger row should be re-scoped,
   not closed.
2. **The gate-1 "<3 %" verdict is convention-dependent.** It is a
   *self-consistency* statement about recovering the unbounded Stokes velocity at
   `t << L^2/nu`. As a physical statement about the discretised periodic domain
   the literature-anchored error is 4.43 % at `L = 32 d` (already > 3 %) and
   7–46 % at boxes where the back-flow is resolved. The verification docstring's
   "`3 pi d / L ~ 0.3 %`" is quantitatively wrong by ~15x and was corrected to
   the Hasimoto value in this mission.
3. **`self_response` cannot be interpreted as "removing the finite-domain
   effect".** It removes a numerical self-field that is 12.7x the physical
   effect and grid/position dependent. Both interpretations cannot hold at once;
   the audit keeps the residual of the *physical* array anchor explicit instead.
4. **Not run (honest list).** `L = 12 d` (3 modes) and `L = 16 d` settling runs,
   and the `array_subtracted` mode, were queued but not completed within this
   mission's compute budget (host load15 = 92/128, single-core runs of 2–8 min
   each); only `L = 8 d` / `none` / `spectral` are reported. Consequently
   `fig3` contains a single box and must not be read as a `phi` trend. The
   operator sweep (Q1) is complete over 7 boxes x 2 grids x 2 positions.
5. **`array_subtracted` is a hypothesis, not a result.** With `C R = 0.415` at
   `L = 8 d` and `C_H = 0.178` the proposed `R_eff` gives `C R_eff = 0.237`;
   whether that lands on `0.8486` is *measured by the script*, not asserted here.
   The mode is left in the benchmark, not in the library, until it is verified.
6. **Stokes regime only.** `Re_t = 7e-4` in the settling runs and `0.11` in the
   gate; nothing here says anything about the `Re > 1` closure errors.

## 6. Error assessment vs the benchmark (explicit <3 % answers)

| key quantity | reference | measured / derived | error | < 3 % ? |
|---|---|---|---|---|
| Hasimoto array drag enhancement (series truncation) | first neglected term `3.979 phi^(8/3)` | `< 1.1e-4` of `C_H` over audited `phi` | < 0.1 % | **YES** |
| two-term (repo `lbmdem.drag.array_correction`) vs three-term series | `1.5593 phi^2` | `<= 1.6e-5` over audited `phi` | < 0.1 % | **YES** |
| module self-response `C R` vs `C_H`, gate-1 grid | 0.0443 | 0.5633 | +1171 % (12.7x) | **NO** |
| uncorrected settling vs Hasimoto (`L = 8 d`) | 0.8486 | 1.2421 | +46.36 % | **NO** |
| module `spectral` settling vs Hasimoto (`L = 8 d`) | 0.8486 | 0.9114 | +7.39 % | **NO** |
| module `spectral` settling vs Stokes (`L = 8 d`) | 1.0000 | 0.9114 | −8.86 % | **NO** |
| gate-1 published error (`L = 32 d`, `t = 2`) vs Stokes | 1.0000 | 0.9802 | 1.98 % | YES (short time, unbounded-surrogate convention) |
| gate-1 configuration vs Hasimoto array (`L = 32 d`) | 0.9576 | 0.9802 | +2.36 %... see note | NO (>3 % only when the run is long enough to develop the back-flow: 7.39 % at `L = 8 d`) |

**Explicit answer**: the audit finds **no** forward physical quantity inside 3 %
except the literature-series truncation controls (which are algebra, not
physics). Every comparison of the module's settling velocity against a
*physical* finite-domain reference is outside 3 % (7.4 % at best), so the
honest verdict for the gate-1 row is:

`stokes_terminal_velocity: PASS as unbounded-surrogate self-consistency gate,
NOT_SUPPORTED as a physical finite-domain accuracy claim`.

## 7. Disposition (adversarial-review convention)

* **Approve with notes** for the module code: no defect was found in the
  solver, the adjoint transfer operators, or the momentum ledger; the mechanism
  works as implemented.
* **Request changes** (documentation/verification claims, done in this mission
  for the docstring; the report caveat is added to
  `docs/point-particle-cfd-dem.md`):
  * the "0.3 % finite-domain effect" claim (wrong by ~15x, corrected);
  * the gate-1 "<3 %" row must carry the unbounded-surrogate qualifier;
  * the "exact fixed point" wording for `self_response` should be qualified to
    "exact for the steady source in a frozen configuration, short-time window",
    since the fully developed run is 8.9 % off Stokes.
* **Escalate (ledger)**: propose `FEAT-DEM-POINT-CFD-ARRAY` (see below).

## 8. Proposed ledger entry (this mission is `NEW`)

* `slug`: `AUDIT-DEM-POINT-CFD-ARRAY`
* `title`: point-particle CFD-DEM 自响应 vs Hasimoto 阵列基准独立复核
* `acceptance`: (i) operator sweep `C R` vs `C_H` over >= 6 boxes x 2 grids;
  (ii) fully developed settling (`>= 6 L^2/nu`) for `none` / `spectral` /
  `array_subtracted` at `>= 2` boxes; (iii) a mode within 3 % of
  `v_t/(1+C_H)` — currently **unmet** (best 7.39 %), so a future mission must
  either produce such a mode or record `NOT_SUPPORTED` for the periodic-array
  convention.
* domain: TensorDEM solid–fluid coupling verification.