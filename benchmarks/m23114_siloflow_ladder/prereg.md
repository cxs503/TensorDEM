# Pre-registration — numerical convergence audit of the `silo_flow` Beverloo kernel (m23114)

Written **before** any run of this ladder was executed. The estimator that turns
the artefacts into verdicts lives in `analyse.py` and is pinned on synthetic
ground truth by `tests/test_m23114_siloflow_ladder.py` (recovery of a known
limit and order, all three verdict branches, the "not decidable" branch, and the
overlap geometry factor). Ledger anchor: `CASE-DEM-SILO-BEVERLOO`
(checked at checkout: `status = open`, and **not** already decided by this
mission — see §0).

## 0. What is already in the repo, and what is therefore NOT being redone

`CASE-DEM-SILO-BEVERLOO` has been worked ~12 times (`m6296 m6457 m15259 m15381
m20055 m20585 m20707 m21033 m21119 m21274 m22124 m22628 m22657 m22853 m22885`).
Two *different* DEM kernels carry those numbers:

| kernel | module | campaigns | dt-convergence audited? |
|---|---|---|---|
| **A** `silo_discharge` + `granular3d_core` | `src/tensordem/silo_discharge.py` | m6296 m15259 m20055 | **yes** — `docs/AUDIT_M20055_DT_CONVERGENCE.md` (levels dt, dt/2, dt/4, dt/8 → extrapolated +4.33 %) |
| **B** `silo_flow` (`SiloFlow`, rough walls + energy ledger) | `src/tensordem/silo_flow/kernel.py` | m20707 m21033 m21119 m21274 m22124 m22628 m22657 m22853 m22885 | **no** — grep finds no dt/stiffness ladder anywhere for this kernel |

Every *recent* verdict, including the filed `+25.0 %` FAIL and the twelve-seed
"powered ensemble" of m22885, is produced by kernel **B** at its *default*
resolution `dt = T_c/20` (`Material.dt = 0.05·contact_time`). That resolution has
never been tested. So the campaign's ledger verdict rests on an unverified
numerical setting — this is the verification debt this mission pays. The
m22853 status document additionally *asserts* a root cause ("low stiffness ⇒
large overlaps ⇒ ±20…90 % error") that has never been measured; §4 tests it.

Nothing from kernel A is re-run, no orifice scan is repeated (m22628 owns it),
no seed ensemble is repeated (m22885 owns it), and no coefficient is refitted.

## 1. Questions (each separately falsifiable)

* **N1** Is the discharge rate `W` at the campaign's default resolution
  `dt = T_c/20` converged? Quantity: the discretisation bias
  `E_dt = (W_∞ − W_nom)/W_nom`, with `W_∞` from a Richardson extrapolation over
  three resolutions (20/40/80 steps per contact period) **with the observed
  order**, not an assumed one.
* **N2** Does the *soft contact* hypothesis of m22853 hold? Quantity:
  `E_kn = (W(kn=8e4) − W(kn=5e3))/W(kn=5e3)` at fixed 20 steps/contact, compared
  with the analytic prediction of the overlap mechanism (§5) and with zero.
* **N3** With numerically converged `W_∞`, does the anchor's own acceptance
  (`|W/W_ref − 1| < 10 %`, literature `C = 0.58`, `k = 1.4`, never refitted)
  hold?

## 2. Frozen configuration (identical to the campaign's primary case, m22885 §2)

| parameter | value |
|---|---|
| particle diameter `d` | 3.0 mm, density 2500 kg/m³ |
| orifice `D` | 15.0 mm (`D/d = 5.0`) |
| silo radius | 24.0 mm (`silo_D/D = 3.2`) |
| fill height `H` | 36.0 mm (`H/D = 2.4`) |
| `mu` / `mu_rolling` / restitution | 0.50 / 0.05 / 0.30 |
| wall friction | 0.0 (flat bottom plate) |
| settle (plug closed) / discard | 0.06 s / 0.12 s |
| measurement | 8 windows × 0.05 s; `W` = mean of the last 50 % |
| `kn` | 5.0e3 N/m (nominal), ladder 2.0e4, 8.0e4 |
| seeds | 1 and 2 (a *numerical* ladder needs two seeds only to show the level changes are not seed noise) |

Ladder axes (everything else bit-identical — same seed ⇒ same packing, because
`build_bed` builds the bed at the same `dt`… see §6 note):

```
dt axis : steps/contact = 20 (nominal), 40, 80      [dt = T_c(N)/…, T_c = 2π√(m_eff/k_n)]
kn axis : kn = 5e3, 2e4, 8e4 N/m   at steps/contact = 20
```

## 3. Estimator (frozen, implemented in `analyse.py`)

Per level, `W` is read from the run record — never re-estimated by hand.
Discretisation bias from three resolutions `W1 (20), W2 (40), W3 (80)`:

```
p_obs = log(|W1 − W2| / |W2 − W3|) / log 2        (observed order, must be > 0)
W_∞   = W3 + (W3 − W2) / (2^p_obs − 1)            (Richardson)
E_dt  = (W_∞ − W1) / W1
```

**Decision rule (frozen before the runs).** With `E_phys = (W_nom − W_ref)/W_ref`
the campaign's own discrepancy relative to the literature reference:

| condition | verdict for N1 |
|---|---|
| `|E_dt| < 0.2·|E_phys|` | **NUMERICS-NEGLIGIBLE** — the campaign's error is physics, not the time step |
| `0.2·|E_phys| ≤ |E_dt| < 0.5·|E_phys|` | **NUMERICS-MIXED** — part of the error is discretisation; earlier verdicts must be restated as `W_∞` |
| `|E_dt| ≥ 0.5·|E_phys|` | **NUMERICS-DOMINATED** — every kernel-B verdict at `dt = T_c/20` is withdrawn |
| `p_obs ≤ 0` or not finite, or the two seeds disagree by more than `|E_dt|` | **NOT_DECIDABLE** — say so, claim nothing |

`E_dt` is additionally cross-checked against the *seed-to-seed* spread at the
finest level; a bias smaller than the seed noise is reported as such.

## 4. Falsification and honest-FAIL policy

* If `p_obs ≤ 0`, the ladder is noise-limited: the mission reports
  NOT_DECIDABLE and does **not** quote a limit.
* If `E_dt` is large, the honest conclusion is that the campaign's numbers were
  never converged — the old FAIL is *withdrawn*, not laundered: N3 is then judged
  on `W_∞`, and if `W_∞` also misses 10 % the anchor keeps its FAIL.
* The `kn` ladder is only allowed to *support* the soft-contact story if the
  measured `E_kn` and the analytic overlap prediction of §5 agree in sign and
  magnitude within a factor 3. Otherwise the m22853 claim is recorded as
  **not substantiated**.

## 5. Analytic prediction for the overlap mechanism (N2)

A soft contact under a squeezing force `F` overlaps by `δ = F/k_n`. Two
estimates bound the effect at the nominal setting:

* static (own weight): `δ_s = m g / k_n` (a single contact carries one particle);
* dynamic (orifice velocity): `δ_v ≈ v_o √(m_eff/k_n)` with `v_o = √(2 g H)`
  the free-fall velocity at the plate.

An overlap shrinks the particle's effective radius in the aperture, i.e. it acts
like `d → d − 2δ` (both particles), so through
`W ∝ (D − k·d)^{5/2}`

```
ΔW/W ≈ 2.5 · k · 2δ / (D − k·d)
```

This is the number the m22853 "large overlap" claim implies; `analyse.py`
computes it from `d, k_n, m, D, H` for both estimates and compares with `E_kn`.

## 6. Known protocol caveat (declared up front)

`build_bed(..., dt=…)` builds the packing at the ladder's `dt`, so the settling
phase is not bit-identical across levels (a different `dt` integrates the
deposition differently). This is *intended*: a numerical ladder must vary the
integration of the whole protocol, not only the measurement window. The
consequence — level differences mix discretisation and bed-history scatter — is
why N1 needs two seeds and why the seed spread is quoted next to `E_dt`.