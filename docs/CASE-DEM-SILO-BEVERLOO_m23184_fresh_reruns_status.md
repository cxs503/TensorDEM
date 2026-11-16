# m23184 — fresh re-runs of the frozen Beverloo case: status

`run_compute` job **cj4672** (5 shards = seeds 1–5 of the frozen primary case,
`benchmarks/m23184_repro/run_case.py`, 2 cores/shard, `stage_dir` = this worktree)
was submitted at 08:27 and was **still queued** (queue position ~120, 125 jobs in
flight on the gate) when this mission's window closed. Per the honest-FAIL rule its
output is reported as **NOT_RUN**, not as agreement.

What is *not* waiting on that job, and is therefore decided here:

1. **The archive is not self-consistent.** Two records of the frozen case with the
   *same* seed (1) are stored in the repository with W = 0.030750 kg/s
   (`m22885/runs/m22885_beverloo_powered/fill36_seed1.json`) and
   W = 0.003358 kg/s (`m23114/runs/m23114_siloflow_ladder/dt_nom20_s1.json`).
   Their spread, 9.2 × W_ref ≈ 92 % of the reference rate, is one order of
   magnitude larger than the counting-noise floor of either record (7.6 % / 22.9 %),
   and one of the two is an **arch**, not a measurement (5 of 8 windows empty).
2. **The kernel is bit-reproducible in-process and across processes**
   (`benchmarks/m23184_repro/probe_det.py`: two runs with `PYTHONHASHSEED=0` and
   `PYTHONHASHSEED=987654321` return the same `pos`/`vel` sha256 after the same
   settle + 400 steps), so the disagreement cannot be blamed on hash-order or
   threading non-determinism inside `build_bed` / `SiloFlow`. The remaining
   explanations are (a) a different code state in the worktree that produced the
   `m22885` record, or (b) genuine bistability of the bed (arch / no arch) under
   an initial-state difference far below the stored precision. The frozen record
   does not carry enough information (no bed checksum) to separate (a) from (b) —
   that is itself a defect fixed for future runs (`run_case.py` now stores the
   git sha, host and window history; `discharge_status` stores the diagnosis).
3. **A plugged record must never be averaged in.** `discharge_status` (this
   mission) fires on the m23114 record via two independent indicators
   (62 % empty windows, 11.5 % of the reference rate) and stays silent on the four
   m22885 records (+5.5 %, −9.1 %, +4.7 %, −1.5 %).

When `cj4672` lands, `python3 benchmarks/m23184_repro/analyse.py --figures`
re-ingests `runs/m23184_repro/seed*.json` automatically (the collector filters on
the frozen signature) and regenerates `runs/m23184_repro/summary.json`,
`docs/CASE-DEM-SILO-BEVERLOO_m23184_reproducibility_audit.md` and the two figures —
no further coding is needed.

Reproduction of everything that is decided here:

```bash
PYTHONPATH=src python benchmarks/m23184_repro/probe_det.py           # bit-identical bed
PYTHONPATH=src python benchmarks/m23184_repro/analyse.py --figures   # archive audit + figures
PYTHONPATH=src python -m pytest tests/test_m23184_beverloo_arrest.py -q   # 13 passed
```

Consequence for the anchor: the `±10 %` acceptance can only be quoted as a
**ensemble** statement with the **arrest fraction** beside it. With the four
flowing archived records the mean is −1.3 % and the sd 6.4 % — i.e. the anchor is
*not* refuted by the flowing records, but a single record (the campaign's own
practice so far) cannot decide it, and at least one archived record
(m23114's ladder point, which feeds a Richardson extrapolation) is invalid.