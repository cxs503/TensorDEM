# Cell-list pair search: a vectorised replacement is slower on this stack (honest FAIL)

**Mission** m16007 · **repository** TensorDEM · **gap** GAP-DEM-G6
**Target** `src/tensordem/dem3d.py::IceDEM3D._contact_candidate_pairs`
**Tests** `tests/test_dem3d_cell_list.py` (11 cases) · **Bench** `scripts/m16007_cell_list_bench.py`
**Artifacts** `artifacts/cell_list/` (`baseline.json`, `vectorized.json`, `negative_result.json`)

## 1. Why this was attempted

The gap ledger calls the CPU-side cell hashing of `IceDEM3D` a "prototype": the
search walks a Python `dict` of cells with four nested loops, and it was measured
at 46 % of a solver step at 104 k particles (`artifacts/cell_list/baseline.json`:
3.28 s of a 7.15 s step; 1.42 s of 3.88 s at 50 653). Removing that term looked
like the cheapest large win for ice-DEM runs, so a fully vectorised search was
implemented: pack the three integer cell coordinates into one `int64` key, one
`torch.unique` + stable `argsort` for the member ranges, 27 `torch.searchsorted`
probes, `repeat_interleave`/`cumsum` expansion, and a final stable sort on
`(cell insertion rank, offset)` to keep the historical pair order.

## 2. What was measured

Same process, same machine window, the frozen shipped implementation
(`_legacy_candidate_pairs`, verbatim copy) against the vectorised candidate, on
the same positions:

| N | candidate pairs | legacy [ms] | vectorised [ms] | ratio | pairs identical |
|---|---|---|---|---|---|
| 4913 | 56368 | 80.3 | 1545.4 | 19.2× | yes |
| 50653 | 622188 | 1424.7 | 6523.6 | 4.6× | yes |
| 1331 | 14230 | 22.5 | 448.6 | 19.9× | yes |
| 10648 | 125748 | 214.2 | 1377.5 | 6.4× | yes |

The solver-level bench (`repeat 3`) agrees: 22.5 → 448.6 ms at N = 1 331 and
214.2 → 1 377.5 ms at N = 10 648 (co-tenanted control plane, load15 ≈ 60 / 128
cores, so the *ratio*, not the absolute value, is the signal — and the ratio is
against the legacy code in the same window).

## 3. Result

* **Correctness: exact.** The vectorised search returned the *bit-for-bit
  identical* pair tensor — same rows, same order, SHA-256 equal at every size —
  and matched an independent O(N²) definition on 11 configurations × 3 radii, with
  no missed overlapping pair and no duplicate rows. The equivalence argument is
  sound.
* **Performance: a regression, 4.6×–20×.** Interpretation: on this CPU/torch
  (2.7.1, many-core, heavily co-tenanted) the 27 `searchsorted` + expansion
  passes over `int64` tensors cost far more than the Python loop they replace,
  whose inner work is a handful of `list.append`s on a few thousand cells. The
  cost is in per-call tensor overhead and allocation churn, not in asymptotics —
  i.e. my hypothesis "the Python loop is the constant that matters at these N"
  was wrong for N ≤ 100 k on CPU.

## 4. Disposition

**The shipped implementation was left unchanged** (the branch contains no edit
to `src/tensordem/dem3d.py`). What is kept is the measurement and invariant
surface that did not exist before:

* `tests/test_dem3d_cell_list.py` — 11 regression cases: an independent O(N²)
  candidate-pair definition, the overlap-completeness invariant (`|d| < 2R` must
  never be missed — the property that would break silently if the cell width
  were wrong), no-duplicate/`i<j` structure, determinism, immutability of
  positions, and a solver-trajectory digest check against a stored baseline. All
  pass on the shipped code; they had no equivalent before this mission.
* `scripts/m16007_cell_list_bench.py` — the before/after harness (pair search,
  full step, ordered/canonical pair digests, trajectory digest), reusable for the
  next attempt.

**Error judgement** (the key quantity vs its benchmark): pair set deviation
**0** (exact, < 3 % trivially); the quantity that failed is wall time, which is
**+4.6× to +19.9× the
baseline** — i.e. this is an honest FAIL on the performance objective, not a
tolerance issue.

## 5. What would be tried next

* Vectorise **inside the force kernel**, not in the search: build the neighbour
  list once per force call as a *flat* array and consume it directly, so the
  `(i, j)` tensor materialisation (1.29 M rows at 104 k) disappears.
* Or change the data layout: keep particles sorted by cell key across steps
  (Verlet skin + rebuild every k steps) so the same sort is amortised, which is
  the standard fix and needs no per-call re-sort.
* Or leave the CPU path alone and take the win on GPU, where the dict loop is
  genuinely fatal (GAP-DEM-GPU-100K already exists as a ledger row).

## 6. Reproduce

```bash
PYTHONPATH=src python -m pytest tests/test_dem3d_cell_list.py -q
PYTHONPATH=src python scripts/m16007_cell_list_bench.py --out artifacts/cell_list/baseline.json
```
