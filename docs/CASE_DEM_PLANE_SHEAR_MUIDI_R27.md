# CASE-DEM-PLANE-SHEAR-MUIDI — round 27 (m22662): the ordinate is ambiguous, and the window is a transient

**Ledger:** `CASE-DEM-PLANE-SHEAR-MUIDI` — 算例池·平面剪切 mu(I) 流变律标定 (status `open`).
**Seat:** verifier. This round adds **no new device and no new fit**; it puts the numbers
that rounds 24–26 judged the ledger on, on independent instruments, and reports what those
instruments say. Two of the findings are blocking, one is a self-inflicted correction.

**Round 26 left open:** "drain `cj4227` and read the 9-point polydispersity sweep".
`cj4227` was still `running` at the start of this round and is **not** the subject of this
document — round 26's own artifact is unchanged and its `mu(I)`/`phi(I)` lines remain
`WITHHELD`. This round answers a different question that had to be settled before any
sweep can be judged:

> Rounds 24–26 judged `mu_wall = tau_wall / p_wall` averaged over a **fixed 1-strain
> window**. Is that number a steady-state friction coefficient, and is the pressure that
> normalises it unique?

## ① Case introduction

Plane shear of a 2-D frictional disk bed at imposed pressure, the geometry of the
reference law (da Cruz et al. 2005, PRE 72:021309; GDR MiDi 2004, EPJE 14:341):

| quantity | value |
|---|---|
| free disks | 458, `d = 10 mm` (probe cell) |
| cell | `W = 0.30 m`, wall row centre-to-centre `0.147 m`, `gap_eff = 0.137 m` |
| confining pressure | `P = 1.0e4 Pa`, constant-pressure servo on the top row |
| contact friction | `mu_p = 0.4`, `mu_rolling = 0` |
| protocol | 40 `d/sqrt(P/rho)` compaction, 0.5 strain transient, 1.0 strain measure |

The archived artifacts audited here are the round-25 probes
(`runs/plane_shear_2d/probe_I*.json`, `structure_random_I*.json`), each carrying its full
per-sample time series.

## ② What was done

1. **`benchmarks/plane_shear_mu_i/audit_r27_protocol.py`** — reads the archived series and
   reports, per run: the integrated autocorrelation time `tau = 1 + 2*sum_k rho_k` of
   `mu_wall` (Sokal window), the honest standard error `sd/sqrt(n_eff)` with
   `n_eff = n/tau`, a **two-block stationarity test** with that corrected error, quartile
   means in strain, and an exponential saturation fit whose asymptote is only reported as
   *measured* when the window covers >= 3 decay lengths (otherwise *extrapolated*, and if
   the fit pins a bound, *unidentifiable*).
2. **Ordinate matrix.** Four `(tau, p)` pairings and three `I` definitions are tabulated;
   the spread is compared with the ledger's 10 % tolerance.
3. **`benchmarks/plane_shear_mu_i/virial_profile_r27.py`** — the pressure mismatch is
   attacked directly: the discrete virial identity `sum_c l x f = sum_p x_p x F_p^ext`
   fixes the *only* volume that makes the virial stress equal the wall traction, and the
   candidate normalisations are compared on one short probe run, with a layer-resolved
   `sigma_yy(y)` profile to separate a prefactor error from a boundary-layer effect.
4. **`tests/test_plane_shear_2d_r27_protocol_audit.py`** (17 cases, **17 passed**) pins the
   statistics against their analytic values (white noise `tau = 1`; AR(1)
   `tau = (1+phi)/(1-phi)`; exponential recovery; a random walk must *not* be reported as a
   drift), binds the audited `mu` to the stored artifact number bit-for-bit, and pins the
   verdict wiring (a series 3x below the whole reference family may not be rescuable).

## ③ Results

### ③a The reported error bars were 1.3–5.6x too small

| run | I | n samples | `tau` | `n_eff` | `mu_wall` ± se(corrected) | se naive | inflation |
|---|---|---|---|---|---|---|---|
| `probe_I0.02` | 0.0261 | 813 | 28.9 | 28.1 | 0.0963 ± 0.0153 | 0.0028 | 5.4x |
| `probe_I0.05` | 0.0631 | 327 | 31.5 | 10.4 | 0.1003 ± 0.0127 | 0.0023 | 5.6x |
| `probe_I0.1` | 0.1393 | 164 | 3.7 | 44.8 | 0.1238 ± 0.0108 | 0.0056 | 1.9x |
| `probe_I0.2` | 0.2464 | 83 | 4.0 | 20.8 | 0.1294 ± 0.0140 | 0.0070 | 2.0x |
| `structure_random_I0.05` | 0.0669 | 326 | 19.6 | 16.6 | 0.0593 ± 0.0108 | 0.0024 | 4.4x |
| `structure_random_I0.2` | 0.2383 | 83 | 1.6 | 52.8 | 0.1156 ± 0.0087 | 0.0070 | 1.3x |

The samples are 40-step blocks of the same sheared state; at `I = 0.05` the run holds only
**10 independent states**, not 327. Round 25's stationarity z-scores (`-6.9 … -13.2`) were
computed with `sd/sqrt(n)` and are therefore not on the scale they appear to be on.

### ③b Where the series is going — the window is not a steady state everywhere

| run | `dmu/dgamma` (descriptive) | half-split z (corrected) | decay length | window mean vs `mu_inf` | fit status |
|---|---|---|---|---|---|
| `probe_I0.02` | −0.052 | −1.0 | — | — | unidentifiable |
| `probe_I0.05` | −0.073 | **−8.9** | 0.84 (1.2 covered) | **+168 %** | extrapolated |
| `probe_I0.1` | −0.059 | −1.1 | — | — | unidentifiable |
| `probe_I0.2` | +0.075 | +1.8 | — | — | unidentifiable |
| `structure_random_I0.05` | −0.059 | −1.8 | 0.36 (2.8 covered) | +70 % | extrapolated |
| `structure_random_I0.2` | +0.019 | +1.5 | 0.26 (3.8 covered) | −6 % | measured |

Two statements survive the corrected errors:

* at `I = 0.05` (monodisperse) `mu` is **still falling at −8.9 σ** when the window closes,
  and the exponential fit puts the asymptote **168 % below** the reported window mean,
  i.e. `mu_inf ~ 0.04`. That is the *same* direction as round 25's crystallisation finding:
  the longer this bed is sheared, the lower its macroscopic friction — basal slip of a
  re-crystallising packing, not a critical state;
* at the other points the asymptote is **unidentifiable** inside a 1-strain window (the fit
  pins a bound). "Saturated" and "20 strains away from saturation" cannot be distinguished
  with this data — which means the fixed-window number carries an *unquantified* protocol
  bias. From the quartile drift alone (e.g. `probe_I0.05`: 0.1268 → 0.1155 → 0.0839 →
  0.0747) that bias is **> 40 %**, i.e. four times the acceptance tolerance.

### ③c The ordinate is not unique: which pressure?

| run | `tau_w/p_w` | `tau_w/p_b` | `tau_b/p_w` | `tau_b/p_b` | spread | `p_bulk/p_wall` | `sigma_yy/P` (wall) |
|---|---|---|---|---|---|---|---|
| `probe_I0.02` | 0.0963 | 0.1223 | 0.0811 | 0.1022 | +51 % | 0.775 | 1.039 |
| `probe_I0.05` | 0.1003 | 0.1333 | 0.0882 | 0.1157 | +51 % | 0.763 | 1.066 |
| `probe_I0.1` | 0.1238 | 0.1679 | 0.0859 | 0.1110 | +95 % | 0.755 | 0.973 |
| `probe_I0.2` | 0.1294 | 0.1621 | 0.0925 | 0.1098 | +75 % | 0.818 | 1.082 |
| `structure_random_I0.05` | 0.0593 | 0.0828 | 0.0296 | 0.0410 | +180 % | 0.720 | 1.032 |
| `structure_random_I0.2` | 0.1156 | 0.1503 | 0.0765 | 0.0958 | +97 % | 0.794 | 1.214 |

The two pressure estimators disagree by **20–28 %** in *every* run and in the same
direction: the wall traction reproduces the imposed `P` (`sigma_yy/P = 0.97…1.08`) while the
interior virial sits at `0.72…0.82 P`. Consequences that are independent of which one is
"right":

* `mu_wall` uses `p_wall` but `I_bulk` uses `p_bulk`, so the ledger's abscissa and ordinate
  are built on two different pressures; `I ∝ p^{-1/2}` makes the abscissa 12–18 % high
  relative to a wall-pressure definition;
* the spread of `mu` over the four pairings is **51–180 %**, i.e. 5–18× the 10 % tolerance.
  **No verdict on this ledger can be tighter than that ambiguity until it is resolved.**
* the I definition table (`I_definitions` in the JSON) shows the same series at
  `I = 0.0631` (interior virial) vs `0.0723` (wall pressure) — a 15 % shift in the abscissa,
  which alone moves the reference `mu` by ~0.02, i.e. 20 % of a 0.10 `mu`.

`virial_profile_r27.py` (job `cj4296`, submitted this round) decides *why*: the discrete
virial identity fixes the volume that makes virial and wall traction agree; if the candidate
normalisation `W*(y_top-y_bottom)*th` closes the 24 % gap, the defect is a one-line
normalisation in `measure()` (it divides by `gap_eff = y_top-y_bottom-2r`), not physics.

### ③d The distance to the reference family is structural, not bibliographic

Best case over the *whole* admissible parameter envelope (`reference_provenance`,
`mu_s ∈ [0.34,0.40]`, `mu_2 ∈ [0.55,0.95]`, `I_0 ∈ [0.22,0.38]`):

| run | I | `mu` judged | best-case worst-point error | provenance can rescue? |
|---|---|---|---|---|
| `probe_I0.02` | 0.0261 | 0.0963 | **+267 %** | no |
| `probe_I0.05` | 0.0631 | 0.1003 | **+269 %** | no |
| `probe_I0.1` | 0.1393 | 0.1238 | **+220 %** | no |
| `probe_I0.2` | 0.2464 | 0.1294 | **+227 %** | no |
| `structure_random_I0.05` | 0.0669 | 0.0593 | **+526 %** | no |

The measured `mu` is a **factor ~3** below the lowest reading of the reference law. Neither
the provenance dispute (which spans ±20 %) nor the window protocol (bias < 40 % on the
points where it is measurable) nor the pressure convention (which *raises* `mu` by at most
35 % if `p_bulk` is used) can close a factor of 3. The discrepancy is a property of the
device, exactly as rounds 25/26 diagnosed (crystal + free rolling at `mu_p = 0.4`).

## ④ Figures

`docs/figures/r27_protocol.png` — left: the judged `mu` of every archived run against the
two candidate laws, error bars autocorrelation-corrected; right: the corrected second-half
minus first-half difference per run (the stationarity instrument), so the one significant
drift (`probe_I0.05`) is visible against the noise of the others.

## ⑤ Notes, limitations, applicability

* **Not done here:** `cj4227` (round 26's 9-point polydispersity sweep) was still running
  when this document was written; its `mu`/`phi` lines stay `WITHHELD` in round 26's file.
* The saturation fit is a *diagnostic of the window*, not a rheological measurement: on the
  three `unidentifiable` points it says only "1 strain is too short to know", which is the
  point being made.
* The half-split test has the right scale for an integrated series (a random walk is tested
  and does not fire), but it is a two-block test: it detects a monotone trend, not a slow
  oscillation. With `n_eff = 10` its power is limited — the honest reading of the other
  five rows is **"no drift detected"**, not "stationary".
* `p_bulk`'s 24 % deficit is measured, not yet explained; the explanation candidates are
  enumerated in ③c and one of them (normalisation volume) is under test in `cj4296`.

## ⑥ Error verdict (honest)

| quantity | criterion | this round | verdict |
|---|---|---|---|
| audited `mu` equals the stored artifact | bit-level | equal to 1e-12 (test) | **PASS** |
| statistics pinned on analytic values | `tau`, AR(1), exponential, random walk | 17/17 tests | **PASS** |
| window is a steady state (`I = 0.05`, monodisperse) | no drift | z = −8.9, asymptote 168 % below | **FAIL** |
| window is a steady state (other 5 runs) | no drift | not detected, `n_eff` 10–53 | **WITHHELD** |
| `mu` unique given the archived data | spread < 10 % | spread 51–180 % | **FAIL** |
| pressure conventions agree | < 10 % | 20–28 % apart | **FAIL** |
| `mu(I)` within 10 % of the law | < 10 % | factor ~3 low; best-case envelope error 220–526 % | **FAIL (pre-existing, independent of provenance)** |
| `phi(I)` within 10 % of the law | < 10 % | not judged here (round 26 owns it) | **WITHHELD** |

**Is any key transport quantity within 3 % of its reference? No.** No quantity measured in
this round has an external reference at all: the audited numbers are device diagnostics
whose criteria are internal (identity, analytic statistics, self-consistency). The one
external comparison that can be made — `mu` vs the reference law — fails by a factor of 3
and is *not* rescuable by the provenance envelope.

## ⑦ What this changes for the next round

1. **Judge against a declared pressure convention.** The ledger needs one: `mu` and `I` must
   be formed from the same `P`. The wall traction currently reproduces `P` to 3–8 %, the
   interior virial does not; adopt wall pressure for both, and fix (or justify) the virial
   normalisation before `mu_bulk` is quoted again.
2. **Report a stationarity decision with an autocorrelation-corrected error**, or report the
   run as `WITHHELD`. `n_eff`, not `n`, is what the window contains.
3. **A window of 1 strain cannot certify saturation** at `I <= 0.05`: either extend it
   (round 27's `cj4286` runs 1 + 8 strains at `w = 0.5` for exactly this) or report the
   drift explicitly with the window mean.
4. The factor-3 offset is the *device's* problem (crystal / free rolling), not the
   reference's: polydispersity (round 26) is the right lever and remains the open question.
---

## ⑧ Amendment — `cj4227` drained during this round: the polydispersity axis is measured

Sections ①-⑦ were written while `cj4227` was still `running`. It reached `succeeded`
(node n315) inside this round; the nine artifacts were fetched and committed to
`benchmarks/plane_shear_mu_i/runs/plane_shear_2d/poly_w*_I*.json`, and the audit was re-run
over them (`outputs/r27/protocol_audit.*`, 15 series). Round 26's open `mu`/`phi` lines are
now measured. The table is the audit's, not a transcription:

| `w` | `I` measured | `mu_wall` (± corrected se) | `phi_gap` | `psi6` | `p_wall/P` |
|---|---|---|---|---|---|
| 0.00 | 0.0259 | 0.0448 ± 0.0044 | 0.875 | 0.937 | 1.00 |
| 0.00 | 0.0669 | 0.0593 ± 0.0108 | 0.875 | 0.914 | 0.97 |
| 0.00 | 0.2383 | 0.1156 ± 0.0087 | 0.873 | 0.957 | 1.14 |
| 0.25 | 0.0208 | 0.1379 ± 0.0104 | 0.816 | 0.448 | 0.95 |
| 0.25 | 0.0502 | 0.1160 ± 0.0115 | 0.809 | 0.436 | 1.03 |
| 0.25 | 0.2092 | 0.1639 ± 0.0142 | 0.801 | 0.465 | 0.93 |
| 0.50 | 0.0199 | 0.1278 ± 0.0095 | 0.815 | 0.397 | 0.94 |
| 0.50 | 0.0474 | 0.1543 ± 0.0114 | 0.812 | 0.401 | 1.06 |
| 0.50 | 0.1730 | 0.1616 ± 0.0099 | 0.808 | 0.376 | 1.36 |

**What this establishes (and it is a real gain):**

1. **The crystal hypothesis of round 25 is confirmed quantitatively.** The monodisperse bed
   keeps `psi6 = 0.91…0.96` (a crystal) and `phi_gap = 0.875` (above 2-D random close
   packing 0.84, i.e. crystallised); the polydisperse seeds hold `psi6 = 0.38…0.47` and
   `phi_gap = 0.81`. Breaking the crystal **raises `mu` by a factor 2.4–3.1 at low `I`**
   (0.045 → 0.128 at `I ~ 0.02`; 0.059 → 0.154 at `I ~ 0.05`) and moves `phi` from the
   crystal branch onto a physically sensible dense branch. At matched `I`, `w = 0.25` and
   `w = 0.5` agree within their error bars, so this is a *disorder* effect, not a
   size-spread effect.
2. **`mu(I)` now has the right slope as well as the right sign**: 0.128 → 0.154 → 0.162 over
   `I = 0.02 → 0.17` at `w = 0.5`, monotonically increasing, as the law requires.
3. **The acceptance still fails, and now not because of the seed.** Best case over the whole
   admissible reference envelope: worst-point error **174 %** at `w = 0.5` (214 % at
   `w = 0.25`, 689 % at `w = 0`). The measured `mu = 0.128…0.162` sits a factor ~2 below the
   law's `0.28…0.40` over the same `I` range. So the ledger verdict is:

   | quantity | criterion | `w = 0.5` (crystal-free) | verdict |
   |---|---|---|---|
   | `mu(I)` vs law, all points | < 10 % | best-case envelope error 174 % | **FAIL** |
   | `mu(I)` trend | increasing in `I` | monotone increasing | **PASS** |
   | seed disorder | `psi6 < 0.5` | 0.38–0.47 | **PASS** |
   | `phi(I)` | < 10 % | `phi` flat at 0.808–0.815 while the reference falls 0.81 → 0.76 over this `I` range: at `I = 0.17` the error is ~+6 %, at `I = 0.02` ~0 %: **inside** tolerance on the archived `phi = 0.81 - 0.30 I` reading, but that reading is itself provenance-limited (round 13) | **PASS on the archived reading, WITHHELD on the disputed triple** |

4. **The remaining factor ~2 is still the device.** With the crystal gone, the residual
   candidates are the ones ③c already isolated and ⑦ lists: `mu_p = 0.4` with **zero rolling
   resistance** in a 2-D disk packing (disks roll instead of sliding), the pressure-convention
   ambiguity (③c: up to +35 % if `p_bulk` is the ordinate, still not ×2), and the transient
   (③b). A rolling-resistance ladder (`mu_rolling > 0`) at `w = 0.5` is now the cheapest
   decisive experiment on this ledger — it is a single parameter, changes no geometry, and is
   the one thing the current matrix has never varied.

Rounding note: `cj4296` (the virial-normalisation test that decides the ③c pressure question)
and `cj4286` (1 + 8 strain saturation runs at `w = 0.5`) were still queued when this
document closed. Their artifacts, when they land, attach to this round without changing any
conclusion above: ③c and ③b already state the criterion each of them decides.
