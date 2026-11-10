# `data/` — literature anchors and the digitized-curve injection format

## `literature_ice_creep.json`

Single source of truth for every literature number used by the creep gates.
Each entry carries a **provenance grade**:

| grade | meaning |
|---|---|
| `standard-constant` | textbook / standard value, tight (e.g. Glen exponent n = 3) |
| `literature-summary-band` | value quoted in review literature with a stated uncertainty band; not digitized from the primary figure |
| `derived-consistency-band` | band implied by another entry (e.g. creep rate from `A sigma^n`) |
| `withheld_digitizations` | source registered, **data not bundled** |

Rules enforced by `creep.literature.load_literature()`:

* every entry must carry `provenance` (the loader raises otherwise);
* an entry of `kind = "band"` must **not** carry a point `value` — a band is
  never turned into an invented number.

## Injecting a digitized laboratory curve

To enable the curve-level overlay against Cole (1998) / Schulson (2001), drop a
pair of files into `data/digitized/`:

```
data/digitized/<name>.csv     header:  time_s,strain
data/digitized/<name>.json    {"source": "...", "temperature_C": -10.0,
                               "stress_MPa": 1.0, "ice_type": "granular",
                               "note": "digitized from Fig. X"}
```

`creep.literature.load_digitized_curves()` validates the pair:

* a CSV without a sidecar JSON is rejected (`ValueError`);
* a sidecar without `source` / `temperature_C` / `stress_MPa` is rejected;
* the CSV header must start with `time_s,strain`.

Nothing is bundled by default, so the gate reports

```json
"curve_overlay": {"status": "WITHHELD", "pass": null, "reason": "..."}
```

That is deliberate: a missing dataset must not be able to look like a green
check. Fabricating digitized points to satisfy the mission text would be
fabrication and is the one thing this harness refuses to do.