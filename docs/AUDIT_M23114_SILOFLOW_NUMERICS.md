# `silo_flow` Beverloo kernel — numerical-convergence audit (m23114)

Anchor: `CASE-DEM-SILO-BEVERLOO` · kernel under audit:
`tensordem.silo_flow (kernel B)` · reference:
Beverloo, Leniger & van de Velde (1961), *Chem. Eng. Sci.* **15**, 260–269,
Eq. (4), `W = C ρ_b √g (D − k d)^(5/2)`, `C = 0.58`, `k = 1.4`
(literature, never refitted).

Ladder status: the ladder has DEM records — see `analyse.py` output for the verdict.

```text
not queried
```

## 1. Why this audit, and what is deliberately not redone

Twelve rounds of the anchor (`m6296 m6457 m15259 m15381 m20055 m20585 m20707
m21033 m21119 m21274 m22124 m22628 m22657 m22853 m22885`) ended in FAIL
(`+25.0 %` at `D/d = 4`) or `NOT_RUN`. Two different DEM kernels produced those
numbers:

* **kernel A** `silo_discharge` + `granular3d_core` — time step audited in
  `docs/AUDIT_M20055_DT_CONVERGENCE.md` (dt, dt/2, dt/4, dt/8 → extrapolated
  `+4.33 %`);
* **kernel B** `silo_flow` (`SiloFlow`, rough walls, exact energy ledger) —
  **every recent verdict**, and its resolution `dt = T_c/20`
  (`Material.dt = 0.05·contact_time`) had **never** been tested: a grep of
  `docs/`, `benchmarks/`, `tests/` finds no dt or stiffness ladder for it.

So the recent verdicts rest on an unverified numerical setting. This mission
measures that setting. It does **not** re-run kernel A, does **not** repeat the
orifice scan (m22628 owns S1–S4), does **not** repeat the seed ensemble (m22885
owns it), and never refits `C` or `k`.

## 2. Frozen design (prereg.md)

Case identical to the campaign's primary: `d = 3 mm`, `D/d = 5`, `H = 36 mm`,
`μ = 0.5`, smooth plate, 8 × 0.05 s windows, `W` = mean of the last 50 %.
Axes: `dt` = `T_c/20, T_c/40, T_c/80` (10/20/40… steps per contact period),
`k_n = 5e3, 2e4, 8e4 N/m`; two seeds each ⇒ 10 points, submitted as 10 shards
(`run_ladder.py --shard i --nshards 10`).

Verdict rule, fixed **before** any run (prereg.md §3):

```text
p_obs = log2((W20−W40)/(W40−W80));  W∞ = W80 + (W80−W40)/(2^p_obs − 1)
|E_dt| / |E_phys| < 0.2  -> NUMERICS-NEGLIGIBLE
                   < 0.5  -> NUMERICS-MIXED
                   >=0.5  -> NUMERICS-DOMINATED   (earlier dt-vs-W verdicts void)
p_obs <= 0, or seed scatter > |E_dt|  -> NOT_DECIDABLE
```

The estimator that computes it (`analyse.py`) is pinned by 22 tests
(`tests/test_m23114_siloflow_ladder.py`): exact recovery of a known limit and
order, all four verdict branches, the `NOT_DECIDABLE` branch, the bulk-density
convention switch, and the overlap geometry below.

## 3. Analytic result that needs no DEM — the "soft contact" story, quantified

The m22853 status document asserts as *root cause* that "the linear-spring DEM
kernel with low stiffness (`k_n ~ 5×10³–10⁵ N/m`) produces large overlaps and
unrealistic contact mechanics, leading to systematic underprediction (−82 %) or
overprediction (+20…38 %)". That is a falsifiable arithmetic claim. An overlap
`δ` shrinks the effective particle diameter by `2δ` and
`W ∝ (D − k d)^2.5`, so

```text
ΔW/W ≈ 2.5 · k · 2δ / (D − k d),   δ_static = mg/k_n,   δ_dyn = v_o√(m_eff/k_n)
```

with `v_o = √(2gH) = 0.8404 m/s` at this case:

| k_n [N/m] | δ_static [m] | δ_dynamic [m] | bias (static) | bias (dynamic) |
|---|---|---|---|---|
| 5e+03 | 6.934e-08 | 4.996e-05 | 4.494e-05 | 3.238e-02 |
| 2e+04 | 1.734e-08 | 2.498e-05 | 1.124e-05 | 1.619e-02 |
| 8e+04 | 4.334e-09 | 1.249e-05 | 2.809e-06 | 8.096e-03 |
| 2e+05 | 1.734e-09 | 7.900e-06 | 1.124e-06 | 5.120e-03 |
| 1e+06 | 3.467e-10 | 3.533e-06 | 2.247e-07 | 2.290e-03 |

At the campaign's own setting the **upper bound** of the overlap mechanism is
`3.24 %` (`δ_dyn = 4.996e-05 m`,
i.e. `1.67 %` of a particle diameter) and the
static overlap is `6.934e-08 m` — utterly negligible. Going all
the way from `k_n = 5e3` to `8e4 N/m` may therefore change `W` by at most
`2.43 %`.

**Consequence (verified numerics, no DEM):** overlap can account for at most
`3.24` of the filed `25 %`
gap — a shortfall of a factor `7.7`. The m22853 root-cause
attribution is **not substantiated**; the `+25 %` must have a different origin
(wall-friction-free plate, packing preparation, estimator/protocol, or the
reference's own `k`-band). The `k_n` ladder in §2 is the DEM counterpart of this
table and will confirm or refute it empirically.

## 4. What is genuinely unfinished

* The 10 ladder points (`cj4608`, `run_ladder.py --shard i --nshards 10`) were
  **submitted and queued, then not dispatched** inside this mission's window
  (116 jobs in flight on the gate; 12 earlier rounds show the same pattern).
  Per the honest-FAIL rule they are reported as `NOT_RUN`, **not** as agreement.
* Therefore N1 (is `T_c/20` converged?), N2-DEM and N3 (ledger verdict on `W∞`)
  remain **open**; `analyse.py` turns them into a verdict the moment
  `runs/m23114_siloflow_ladder/*.json` exists — no further coding needed.
* Scope: single frozen configuration (`D/d = 5`, monodisperse, smooth plate).
  Nothing here transfers to conical hoppers, wall friction, or polydisperse beds.

## 5. Reproduction

```bash
PYTHONPATH=src python benchmarks/m23114_siloflow_ladder/run_ladder.py --list
PYTHONPATH=src python benchmarks/m23114_siloflow_ladder/run_ladder.py --shard 1 --nshards 10
python3 benchmarks/m23114_siloflow_ladder/analyse.py --figures      # verdict + doc + 2 figures
python3 benchmarks/m23114_siloflow_ladder/render_status.py           # this document
python3 -m pytest tests/test_m23114_siloflow_ladder.py -q            # 22 passed
```

Artifacts: `benchmarks/m23114_siloflow_ladder/{prereg.md,run_level.py,run_ladder.py,analyse.py,render_status.py}`,
`tests/test_m23114_siloflow_ladder.py`, `docs/figures/m23114_overlap_bias.png`,
`runs/m23114_siloflow_ladder/` (populated when `cj4608` lands).
