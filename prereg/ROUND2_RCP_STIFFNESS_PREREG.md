# Round-2 pre-registration — CASE-DEM-RCP-PACKING-FRACTION (TensorDEM, mission 18143)

Written **before** the round-2 runs were submitted (2026-11-13). Frozen design: the
thresholds below are not to be relaxed after the numbers are in; any deviation is
reported as a deviation in the report.

## 0 Decision this design must support

Does a *poured* (gravity-deposited) DEM packing of 10 000 monodisperse spheres reach the
ledger volume fraction `phi in [0.62, 0.66]` in the **rigid-grain (hard-sphere) limit**, in
both friction tiers (frictionless, low friction), and does the measured number agree with the
literature RCP value `phi_ref = 0.6366` (Scott 1960 Nature 188:908; Bernal & Mason 1960
Nature 188:910) within the < 3 % accuracy standard?

## 1 Why round 2 exists (state of the artefact before this round)

The previous wave (N = 10000, 3 seeds/tier) was run at soft-sphere stiffness `k_n = 4000`.
Its own gates said the number is **not yet a rigid-grain number**:

| gate | criterion | got (kn=4000) | verdict |
|---|---|---|---|
| G5 | `|phi(k_n=prod) - phi(k_n->inf)| < 0.005` (mu=0) | 0.0059 | FAIL |
| G6 | same, mu=0.1 | 0.0092 | FAIL |
| G9 | `max overlap/d < 0.01` settled | 0.0115 | FAIL |
| G13 | `|phi(k_n->inf)/phi_ref - 1| < 3 %` (mu=0.1) | 0.6171 (-3.1 %) | FAIL |

Residual overlaps of 1.15 % mean the bed is *compressed*; a soft bed reads denser than a rigid
one, so the in-band `phi(mu=0.1) = 0.6263` cannot be quoted as a packing fraction until the
stiffness systematic is removed. Round 2 therefore **escalates the production stiffness** and
tests the claim in the hard-sphere limit, instead of reducing the gate.

## 2 Competing alternatives considered

| alternative | why not chosen |
|---|---|
| Keep `k_n = 4000`, report the Richardson limit only | limit rides on a 3-point fit of a first-order-in-`1/k_n` series with observed order 0.5 (frictional tier); the same tree's `order_audit.py` warns this is weakly identified |
| Geometric compression / Lubachevsky–Stillinger to RCP | that is what m16179 did (phi = 0.6400): it *imposes* the answer (jam the volume until contacts lock) and cannot produce a friction tier — the anchor asks for a *poured* packing |
| Physical experiment only | no apparatus; the anchor is a code capability |
| **Escalate `k_n` to 64000 and check 4 stiffness levels + timestep halving** (chosen) | overlap becomes O(1e-3), the hard-sphere limit is bracketed by a 4th level that the 3-level fit must predict (leave-one-out), and `dt` halving separates stiffness from integrator error |

## 3 Pre-declared run set (all N = `10000`, `e = 0.1`, `dt = t_c/20` unless stated)

| arm | mu | k_n | seeds | purpose |
|---|---|---|---|---|
| production | 0.0 | 64000 | 1, 2, 3 | frictionless tier |
| production | 0.1 | 64000 | 1, 2, 3 | low-friction tier |
| stiffness ladder | 0.0 / 0.1 | 4000, 16000, 64000, 256000 | 1 (256000 only) | 4-level `phi(k_n)`, leave-one-out |
| timestep probe | 0.0 / 0.1 | 64000 | 1, `dt = t_c/40` | temporal convergence at production stiffness |
| friction sensitivity | 0.05 / 0.2 | 64000 | 1 | shape of `phi(mu)`, not a tier |

Seeds 1/2/3 use independent random sequential-addition placements (same kernel sha256).
Cost estimate: `k_n = 64000` ~16 min/run, `dt/2` ~32 min, `k_n = 256000` ~70 min, submitted
as independent single-core cluster jobs.

## 4 Metrics (unit of analysis = one settled configuration)

Primary: `phi_bulk` — slab-averaged solid fraction over the bulk band `2.5 d` above the floor
to `2.5 d` below the free surface (excludes both wall layers); `phi_bulk_overlap_corrected`
(per-sphere deflation) reported alongside as the geometric-consistency witness.
Secondary: `Z` (bulk coordination), `max overlap/d`, `dt`, settled flag, wall clock.

## 5 Pre-declared gates (unchanged thresholds, re-evaluated on the new artefact)

| id | criterion | rationale |
|---|---|---|
| G1/G3 | `|phi(mu=0) - 0.6366| / 0.6366 < 3 %` and `phi in [0.62, 0.66]` | ledger + accuracy standard |
| G2 | `0.62 <= phi(mu=0.1) <= 0.66` | ledger, **low-friction tier — the honest risk of this round** |
| G5/G6 | `|phi(k_n=64000) - phi_inf| < 0.005` per tier | rigidity |
| G9 | `max overlap/d < 0.01` | geometric consistency |
| G12/G13 | `|phi_inf/phi_ref - 1| < 3 %` per tier | hard-sphere limit vs literature |
| G14 | `|phi(dt/2) - phi(dt)| < 0.002` at `k_n = 64000` | integrator error |
| G17 | 3-level (4000/16000/64000) prediction of `phi(256000)` within `0.002` | the extrapolation is not self-fitted |
| G18 | `phi(mu=0.1) <= phi(mu=0) + 0.005` | friction must not densify |

**Stopping rule / no p-hacking**: the production stiffness is fixed at 64000 *a priori*. If G2
fails at the rigid-grain limit, the finding is reported as a FAIL (`phi(mu=0.1) < 0.62`) with the
friction scan as the explanation — the band is **not** re-opened by choosing a different `mu`,
a different measurement band, or a different seed after the fact. The `mu = 0.05` arm is declared
here as a sensitivity point before it is run and cannot be promoted to the tier.

## 6 Interpretation limits (stated in advance)

1. `phi(mu)` is known to be a decreasing function of friction (RLP ~0.60 at `mu >= 0.3`);
   a low-friction tier slightly below the band is *expected physics*, not automatically a bug,
   and must be separated from a softness/measurement artefact by the stiffness ladder.
2. Linear spring-dashpot (not Hertz) — `phi` is compared in the `k_n -> inf` limit only.
3. Finite size: `N = 10^4` in a periodic-footprint box; the tap/pour protocol is a single
   release from a dilute RSA cloud, so it samples the *poured* branch (RLP side), not the
   vibrated/compressed branch of Scott (1960).
4. `e = 0.1` throughout; restitution dependence is out of scope for this round.

## 7 Analysis (fixed in advance)

`benchmarks/case_rcp_deposition/aggregate.py` reads the run JSONs (no hand-copied numbers),
selects the production stiffness from `PROD_KN`, fits `phi(k_n) = phi_inf + A k_n^-p` by
Richardson with the observed order, reports the leave-one-out prediction, and emits the gates
above. `debt_audit.py` re-checks production size / seed coverage. Deviations from this plan are
listed in the report's §5.