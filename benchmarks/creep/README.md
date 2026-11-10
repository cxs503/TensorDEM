# `benchmarks/creep/` — ice creep acceptance harness (mission P46W3-N02)

Executable acceptance gates for the ice-creep node: Glen exponent, Andrade
primary creep, long-time contact-force relaxation and the negative controls,
plus literature anchors with explicit provenance grading.

```
PYTHONPATH=src:benchmarks python benchmarks/creep/run_creep_benchmarks.py
PYTHONPATH=src:benchmarks python benchmarks/creep/make_report_doc.py
```

The first command writes `artifacts/ice_creep_p46w3n02/results.json` and six
figures; the second regenerates `docs/P46W3_N02_ice_creep_andrade_fractional.md`
from that JSON (no hand-copied numbers anywhere in the chain).

## Files

| file | role |
|---|---|
| `gates.py` | the four gates (a)-(d) and their predeclared thresholds (`THRESHOLDS`) |
| `literature.py` | literature loader, band/relative-error checks, digitized-curve injection hook |
| `run_creep_benchmarks.py` | driver: gates + literature + JSON + figures |
| `make_report_doc.py` | Markdown report generator (numbers injected from the JSON) |
| `data/gates_p46w3n02.md` | the predeclared thresholds, fixed before the run |
| `data/literature_ice_creep.json` | literature constants **with provenance grades** |
| `data/digitized/` | drop-in location for digitized lab curves (empty by design) |

## Gate summary (latest run)

| gate | measurement | verdict |
|---|---|---|
| (a) Glen exponent | fitted n = 2.9134, rel. err 2.89 % (mission limit 10 %) | mission criterion PASS, predeclared statistical-CI item **FAIL** (see report §5.1) |
| (b) Andrade curve | max rel. err 5.795e-03 vs `eps_0 + eps_1 t^(1/3)` (limit 5 %) | PASS |
| (c) relaxation | tau ∝ A^-1 (Norton slope -1.0000, r = -1.000), tau ∝ eta^(1/alpha) (slope 3.000000, r = 1.000), many-body flat punch slope -1.0063 (r = -0.999996) | PASS |
| (d) negative control | A → 0: strain change 5.6e-08; eta → ∞: creep/elastic 1.0e-05 | PASS |

## Rules of the harness

* Thresholds are predeclared in `data/gates_p46w3n02.md` **and** in
  `creep.gates.THRESHOLDS`; `tests/test_ice_creep_p46w3n02.py` pins both, so a
  later edit cannot quietly relax a gate.
* A failing gate stays failing in `results.json`. There is no "re-banding after
  the fact" path; explanations go in the report, not into the threshold.
* Literature comparison is at the level of constants and bands with a stated
  provenance grade. Missing digitized curves are reported as `WITHHELD`, never
  as a pass.