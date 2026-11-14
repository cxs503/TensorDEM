# Board audit — m20585 / CASE-DEM-SILO-BEVERLOO (TensorDEM)

**Mission audited**: m20585 `case-dem-silo-beverloo`, status `succeeded` (finished 2026-11-14),
worktree `/root/tp-agent/work/m20585`, branch `agent/m20585`, base `fb96bd97` (Merge #1536).
**Audit mission**: m20620 (this worktree, `/root/tp-agent/work/m20620`, branch `agent/m20620`).
**Type**: cross-check / 交差核验 (not a redo). **Verdict**: **L0**, benchmark anchor **N/A (no result)**.

---

## 0. What was reviewed (identities)

All mission deliverables are **uncommitted** in the mission worktree
(`git status`: 4 untracked paths); no commit id binds the work. Every artefact was reviewed by direct file read:

| reviewed file (mission) | sha256 | lines |
|---|---|---|
| `benchmarks/m20585_beverloo/analyse.py` | `5ff82fd0d438e084...` | 211 |
| `benchmarks/m20585_beverloo_kn_sensitivity/run_kn_sweep.py` | `2c05e86575cf1d0b...` | 168 |
| `docs/CASE-DEM-SILO-BEVERLOO_m20585.md` | `cd37d6230f8450de...` | 166 |
| `docs/oss_solutions_CASE-DEM-SILO-BEVERLOO.json` | `51d1d6581c75975f...` | 92 |

## 1. Scope and checks performed

| # | check | result |
|---|---|---|
| C1 | Do run artifacts exist (JSON results, figures, summary)? | **NO** — zero artifacts from m20585's own computation |
| C2 | Does the report contain original computational results? | **NO** — all numbers cited from m20055 and m15259 |
| C3 | Are the analysis scripts executable? | **YES** — syntactically valid Python, but never executed |
| C4 | Does the report have all six sections? | **PRESENT BUT HOLLOW** — ①-⑥ headings exist, ③/④ cite other missions, ⑤ has theoretical expectations only |
| C5 | Is the repro_cmd executable? | **NO** — scripts exist but produced no output; no artifacts to verify |
| C6 | Is there a registered referee row for the anchor? | **NO** — 0 rows in tensor_benchmark_cases for CASE-DEM-SILO-BEVERLOO |
| C7 | Are the scripts real code (input→compute→output)? | **YES** — run_kn_sweep.py calls SiloCase + simulate_silo; analyse.py reads JSON and produces figures |
| C8 | Is the oss_solutions document well-structured? | **YES** — correct transfer vocabulary, candidates, migration plan |

## 2. Findings — blocking

### B1 (blocking) — NO run artifacts
The mission's own report (line ~100) explicitly states:
> "本单集群计算（cj2811, cj2812）因 SSH 超时/排队未产出完整结果"

Cluster jobs cj2798, cj2800, cj2811, cj2812 all failed or were queued. No JSON results, no figures,
no summary.json from m20585's own computation exist anywhere in the worktree.

### B2 (blocking) — NO benchmark results from this mission
The report's section ④ (对照表) contains only one row: D4 from m20055 (kn=1e5, +25.0% FAIL).
This is explicitly marked as "诚实引用而非手抄" (honest citation, not hand-copied). m20585 produced
no got values of its own.

### B3 (blocking) — NO reproducibility
The scripts `run_kn_sweep.py` and `analyse.py` exist but were never executed. There are no output
artifacts to verify that the scripts work correctly or produce meaningful results.

## 3. Findings — medium

### M1 — All deliverables uncommitted
`git status` shows 4 untracked paths. No commit binds the work. The repo head is `fb96bd97`
(Merge #1536), which does not include any m20585 deliverables.

### M2 — Diagnostic hypothesis is sound but unverified
The report correctly identifies:
- m20055's +25% error at kn=1e5 (soft particles)
- m15259's -0.4% at kn=1e6 (stiffer particles)
- The hypothesis: kn sensitivity is the root cause

This reasoning is consistent with the evidence from other missions, but m20585 itself never
ran the kn sensitivity sweep to verify it.

### M3 — Report sections ③/④/⑤ contain no original data
- ③ (参数表): cites m20055's D4 parameters
- ④ (对照表): one row from m20055
- ⑤ (收敛性与参数敏感性): theoretical expectations, no actual data

## 4. Findings — credit

* **Honest reporting**: The mission explicitly admits its cluster jobs failed and does not fabricate results.
* **Correct diagnostic reasoning**: The kn sensitivity hypothesis is consistent with m20055 (+25% at kn=1e5)
  and m15259 (-0.4% at kn=1e6) evidence.
* **Well-structured scripts**: `run_kn_sweep.py` correctly implements the kn sensitivity study design
  (kn=1e5, 1e6, 1e7) with proper parameter handling. `analyse.py` would produce meaningful figures if run.
* **oss_solutions document**: Correctly identifies transfer vocabulary (adapt/reuse/reference),
  candidates, migration plan, and external anchors.

## 5. Level determination

| Level | Criterion | Met? |
|---|---|---|
| L0 | 文档成果 (document output) | **YES** — diagnostic report + oss_solutions JSON |
| L1 | 数学/单元验证 (math/unit verification) | **NO** — analysis is mathematical but never verified by computation |
| L2 | 独立功能实现 (input→compute→output) | **NO** — scripts exist but were never executed |
| L3 | 系统集成 (integration) | **NO** |
| L4 | 基准验证 (benchmark verification) | **NO** — no results |
| L5 | 科研工程成果 | **NO** |

**Verdict: L0** — document output only.

## 6. Disposition

**Approve at L0.** The mission produced a diagnostic report and analysis scripts but no computational
results. The honest-FAIL admission is creditable.

**Remediation order**: Execute the kn sensitivity sweep (`run_kn_sweep.py`) on the compute gate,
produce results, and re-submit for L2+ evaluation.

---

## Appendix: Evidence pointers

| evidence key | pointer |
|---|---|
| impl_files | `benchmarks/m20585_beverloo/analyse.py:1-211`, `benchmarks/m20585_beverloo_kn_sensitivity/run_kn_sweep.py:1-168` |
| entry | None executed |
| run_artifacts | NONE |
| tests | NONE |
| integration_callers | NONE |
| benchmark | N/A (no result from this mission) |
| repro_cmd | Scripts exist but never executed |
| external_anchor | Beverloo 1961, Nedderman 1992, m20055, m15259 |
