# Pre-registration — CASE-SILO-JANSSEN (m15265)

Written **before** any DEM result of this mission was inspected (the single local
timing probe launched at the start was killed by the shell timeout and produced
no output; the cluster probe `cj1029` was still queued when this file was
written). Frozen at authoring time; deviations are reported in the final report.

## Decision

Does the TensorDEM `silo_flow` kernel reproduce Janssen's classical static
wall-pressure distribution of a deep filled silo within the benchmark tolerance
of **15 %**? Two alternatives:

* **H1** — the DEM bed is in Janssen equilibrium: the saturating amplitude
  `P_inf`, the characteristic depth `z0` and the base pressure agree with
  Janssen's slab solution built from the *same-bed measured* lateral stress
  ratio `K_meas`, and the parameter-free identity `P_inf/z0 = rho_b g R` holds.
* **H0** — at least one of these quantities deviates by ≥15 %; then either the
  DEM bed is not in a Janssen state (finite-stiffness / packing artefacts) or
  the measurement reduction is biased.

A second, stricter alternative is reported separately and does **not** enter the
pass/fail verdict: (H2) the absolute wall-pressure level is predicted a priori by
Janssen with the Jaky estimate `K_J = 1 - sin(atan(mu))`.

## Reference (ref) definition

`sigma_h(z) = K sigma_v(z)` with Janssen's slab solution

    sigma_v(z) = P_inf (1 - exp(-z/z0)),  P_inf = rho_b g R/(2 mu_w K),  z0 = R/(2 mu_w K)

Source of the closed form: H. A. Janssen, *Versuche über Getreidedruck in
Silozellen*, Zeitschrift des Vereines deutscher Ingenieure **39** (1895)
1045–1049; English translation with commentary: M. Sperl, *Experiments on corn
pressure in silo cells — translation and comment of Janssen's paper from 1895*,
Granular Matter **8** (2006) 59–65, DOI 10.1007/s10035-005-0224-z.

All reference numbers are **evaluated by code** from this closed form at the
declared case parameters and the measured bulk density `rho_b` (no hand-typed
numbers). Janssen's original measured pressure table could not be retrieved in
this sandbox (Sperl 2006 full text is paywalled) — this is recorded as a
WITHHELD provenance gap, not replaced by a fabricated figure.

## Metrics and pre-declared thresholds

Primary (self-consistent route, per case), tolerance **15 %** each:

| # | metric | definition |
|---|---|---|
| M1 | `p_inf_rel_err` | \|P_inf(DEM fit) - P_inf(Janssen, K_meas)\| / P_inf(Janssen) |
| M2 | `z0_rel_err` | \|z0(DEM fit) - z0(Janssen, K_meas)\| / z0(Janssen) |
| M3 | `identity_rel_err` | \|P_inf/z0 (DEM) - rho_b g R\| / (rho_b g R) — parameter free |
| M4 | `base_pressure_rel_err` | \|p_base(DEM plate force) - P_inf(Janssen, K_meas)\| / P_inf |

A case **passes** iff M1–M4 all < 15 %. The anchor case is `base`; the mission
verdict is the anchor case, and the other cases are sensitivity axes.

Secondary (reported, not gating):
`profile.rel_err_*` — slab-by-slab error of `sigma_w(z)` with an **absolute
floor** of `0.05 * P_inf` in the denominator so that near-surface slabs, whose
predicted stress is ≈0, cannot dominate the relative error. The un-floored worst
slab is reported as well (the earlier bulk-cargo campaign, m7751, reported 28.7 %
worst slab on a different geometry — expected to recur here since it is a
small-stress artefact; it is *not* allowed to be dropped from the report).

## Variables / confounders

* controlled: geometry (R, H, outlet radius), particle diameter, stiffness `kn`,
  contact model, settle protocol, post-processing layer count.
* varied (sensitivity axes, one at a time): wall friction `mu_w` (0.25/0.40/0.60),
  fill height (0.30/0.45 m → `H/z0`), particle diameter (4.5/6.0 mm → `R/d`),
  interparticle friction `mu` (0.40/0.25 → `K`), random seed (42/7).
* confounders monitored: bulk density (reported per case, enters P_inf linearly),
  `K_meas` (measured on the same run, so M1–M2 test shape/balance only),
  finite-stiffness overlap (`kn` is soft, 1e4 N/m → nonzero overlap may bias
  the base plate force), polydispersity = 0 (monodisperse → ordering artefacts).

## Repetition rationale

One seed per sensitivity case, plus one extra seed (`seed7`) for the anchor
configuration to estimate run-to-run scatter. Rationale: a full resolution
sweep is unaffordable (N ≈ 7.6e4 particles, ~1e3 s per run); the seed pair gives
the scatter of the metric, which is the quantity needed to judge whether a 15 %
band is meaningful.

## Pre-declared stopping rule

Stop the campaign when (a) the anchor case plus at least three sensitivity cases
have completed, and (b) the seed pair has completed. If a case has not finished
after 90 min of wall clock on the cluster it is abandoned and reported as
NOT_RUN (no cherry-picking of cases).

## Environment / provenance

* frontend expectation: failure to reach cluster = honest report, no local
  over-subscription of the control node.
* artefacts: `runs/m15265/sim/<case>.json` (raw DEM result + analysis),
  `runs/m15265/comparison.{json,md}`, `runs/m15265/figures/*.png`.
* repro: `python3 benchmarks/silo_janssen_wall_pressure.py --case <case> --out ...`
  then `--analyze runs/m15265`.

## Interpretation limits

Correlation with the Janssen *shape* is not a validation of the absolute
pressure level (K is fitted). The a priori route (H2) is the only absolute test
and is expected to be the harder one; a H2 miss with a H1 pass means "DEM is in
Janssen equilibrium but the Jaky K estimate is not the DEM's K", not "Janssen is
wrong".