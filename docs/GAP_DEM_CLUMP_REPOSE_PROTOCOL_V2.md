# GAP-DEM-CLUMP round 2 — repose protocol validity (measurement + rolling-law audit)

Status: **partial / honest** — the three protocol defects are diagnosed and
quantified below from executed runs; the full mass-matched family scan
(4 shapes x 2 seeds x mu_roll sweep) was launched
(`benchmarks/gap_dem_clump/repose_protocol_v2.py --out artifacts/GAP-DEM-CLUMP-ROUND2`)
but **had not finished when this deliverable was sealed**, so its numbers are
`not_run` here and must not be quoted from this document.

## 1 What was wrong with the round-1 C3 leg (m14681)

| defect | measured evidence (this session, `python3` + `PROTOCOL` of round 1) |
|---|---|
| (a) fill is packed by the clump extent, so shape and particle count are entangled | sphere a/b=1: **648** particles, V_s=**4.690e-6 m^3**; a/b=1.5: 260 / 2.825e-6; a/b=2: 243 / 3.509e-6; a/b=3: **104** / **2.270e-6** -> N x6.2, V_s x2.07 for the same box |
| (b) the heap is measured after a **fixed** 1200-step tail (0.06 s) | at the measurement instant the *same* sphere heap still has mean body speed **0.107 m/s** (max 0.568 m/s): in 0.06 s a body travels ~6.7 mm = 2.8 particle diameters, i.e. the pile is still flowing |
| (c) the Cundall rolling branch is rate proportional | `tau = -mu_r r |Fn| w/(|w| + eps_w)`, `eps_w = 1.0 rad/s` (default, `ContactParams`), so the delivered torque is the Coulomb limit times `s = |w|/(|w|+eps_w)`; measured `s` median **0.32** at rest, **0.35** mean over bodies |

(b) is the dominant artefact. Same run protocol, same fill, only the tail
changed from a fixed 1200 steps to the new quiescence gate
(`quiescence_mean_speed = 0.01 m/s`, probe every 300-500 steps):

| measurement | value |
|---|---|
| fixed 1200-step tail (round-1 protocol) | **25.573 deg** (mean speed 0.107 m/s) |
| quiescence-gated tail, settled at 2500 extra steps | **18.754 deg** (mean speed 1.08e-3 m/s) |

=> a **6.82 deg (26.7 %)** difference produced purely by measuring a moving pile.
The round-1 sphere baseline (25.82 deg, reported "0.8 deg above the glass-bead
band") is therefore not a measurement of the repose angle at all.

## 2 What was implemented

* `src/tensordem/particle/clump_dem.py`
  * `heap_case(..., spacing=None)` — forwards the lattice spacing so a caller can
    match the fill between shapes (mass matching without changing the box).
  * `lifted_box_repose(..., min_tail_steps, max_tail_steps, quiescence_mean_speed,
    probe_every)` — optional quiescence gate; **default path unchanged** (fixed
    tail, `quiesced=None`) so round-1 numbers stay reproducible. Returns
    `tail_steps_used`, `quiesced`, `flow_spread`, `flow_rolling_saturation`,
    `rolling_saturation`, `solid_volume`, `mass`.
  * `ClumpDEM.rolling_torque_saturation(eps_w=None)` — fraction of the Coulomb
    rolling limit actually delivered; `velocity_spread()` now also reports
    `mean_omega`, `median_omega`.
* `benchmarks/gap_dem_clump/repose_protocol_v2.py` — mass-matched fill
  (bisection of the fill height to equal total solid volume at the
  non-overlapping spacing `1.02*ext`), quiescence-gated repose, and legs
  C4 (protocol validity), C5 (rolling-law static limit), C6 (shape trend).
* `benchmarks/gap_dem_clump/run_gap_dem_clump.py` — **reproducibility defect
  fixed**: the whole payload/figure/`results.json` block of `main()` was dead
  code placed after `return out` *inside* `derive_failure_reason`, so the round-1
  campaign entry point printed nothing and wrote no artefact. `main()` now emits
  `results.json`, `criteria.json`, the figure and exits non-zero on an honest
  FAIL. Guarded by `tests/test_repose_protocol_v2.py::test_round1_campaign_main_writes_outputs`.
* `tests/test_repose_protocol_v2.py` — 9 cases (fill confound baseline,
  mass matching, non-overlapping matching, quiescence gate reach/fail/fixed-tail
  compatibility, saturation law + `eps_w` scaling, campaign output regression).

## 3 Rolling resistance: the model has no static branch

`tau(w) = mu_r r |Fn| * w/(w + eps_w)` with the default `eps_w = 1.0 rad/s`:

* at `|w| = eps_w` only **50 %** of the Coulomb limit is delivered, at
  `|w| = 0.1 rad/s` only **9 %**;
* `|w| -> 0` gives `tau -> 0`: the branch is **viscous** below `eps_w` and cannot
  provide static slope support. Sliding friction plus geometry alone hold a
  settled heap, which is why a clump can creep on a slope regardless of `mu_r`.

Consequence for the shape-effect question: `mu_r` does not act as a
rate-independent friction coefficient in this regime, so a repose-angle *trend*
cannot be attributed to rolling resistance as the literature comparison assumes.
A static (elastic-plastic) rolling spring with a Coulomb limit (Jiang-type) is the
missing capability; it is **not** implemented here (out of scope, honest gap).

## 4 Honest status

* **done**: protocol defects diagnosed and quantified; production hooks added
  without changing defaults; round-1 campaign reproducibility defect fixed; tests.
* **not_run / withheld**: the mass-matched family scan and the `mu_roll` sweep
  (`repose_protocol_v2.py`); therefore the round-1 verdict "shape trend does not
  match the literature" is **neither confirmed nor refuted** — it is replaced by
  the statement that the round-1 measurement was invalid for a 26.7 % artefact.
* GAP-DEM-CLUMP stays **open**; the anchor's C3 sub-question is re-scoped.

## 5 Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_repose_protocol_v2.py -q
python benchmarks/gap_dem_clump/repose_protocol_v2.py --out artifacts/GAP-DEM-CLUMP-ROUND2 --fast
```
