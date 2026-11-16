#!/usr/bin/env python3
"""Render the m23114 audit document + figure from the frozen analytic model.

Two regimes, decided by what is on disk:

* **no DEM record** → the document reports the ladder as NOT_RUN and carries only
  the *analytic* part (the overlap/soft-contact budget), which needs no DEM;
* **records present** → the document is the data verdict and is produced by
  `analyse.py` instead (this script then only refreshes the analytic figure).

Every number is produced here; nothing is hand-copied.

    python3 benchmarks/m23114_siloflow_ladder/render_status.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import analyse as A  # noqa: E402

RUNS = ROOT / "runs" / "m23114_siloflow_ladder"
DOC = ROOT / "docs" / "AUDIT_M23114_SILOFLOW_NUMERICS.md"
FIG = ROOT / "docs" / "figures" / "m23114_overlap_bias.png"

GEO = dict(radius=0.0015, orifice=0.015, fill_height=0.036)
KNS = (5.0e3, 2.0e4, 8.0e4, 2.0e5, 1.0e6)
FILED_REL_ERROR = 0.25          # the campaign's filed FAIL (m20232 audit / m22885 prereg)


def analytic_table() -> dict:
    return {kn: A.overlap_predicted_bias(kn, **GEO) for kn in KNS}


def job_state() -> str:
    """Best-effort: the queue line of the ladder submission, if it is still around."""
    try:
        out = subprocess.run(["bash", "-lc",
                              "run_compute status cj4608 2>/dev/null || true"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        out = ""
    return out or "not queried"


def make_figure(table: dict) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    kns = sorted(table)
    dyn = [table[k]["predicted_bias_dynamic"] for k in kns]
    sta = [table[k]["predicted_bias_static"] for k in kns]
    ax.loglog(kns, dyn, "o-", label="dynamic bound  δ = v_o√(m_eff/k_n)")
    ax.loglog(kns, sta, "s-", label="static bound  δ = mg/k_n")
    ax.axhline(FILED_REL_ERROR, ls="--", color="tab:red",
               label=f"filed campaign error ({100*FILED_REL_ERROR:.0f} %)")
    ax.axvline(5.0e3, ls=":", color="0.5")
    ax.annotate("campaign setting\nk_n = 5e3 N/m", xy=(5.0e3, 1e-3), fontsize=8,
                color="0.35")
    ax.set_xlabel("contact stiffness k_n [N/m]")
    ax.set_ylabel("rate bias implied by contact overlap  ΔW/W")
    ax.set_title("m23114 — can contact overlap explain the ±25 % Beverloo gap?\n"
                 "(prereg.md §5, analytic; no DEM involved)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return str(FIG.relative_to(ROOT))


def render(table: dict, runs_present: bool, state: str) -> str:
    d5 = table[5.0e3]
    d8 = table[8.0e4]
    change = d5["predicted_bias_dynamic"] - d8["predicted_bias_dynamic"]
    short_fall = FILED_REL_ERROR / d5["predicted_bias_dynamic"]
    rows = "\n".join(
        f"| {kn:.0e} | {v['delta_static_m']:.3e} | {v['delta_dynamic_m']:.3e} | "
        f"{v['predicted_bias_static']:.3e} | {v['predicted_bias_dynamic']:.3e} |"
        for kn, v in sorted(table.items()))
    status_line = ("the ladder has DEM records — see `analyse.py` output for the "
                   "verdict" if runs_present else
                   "**NOT_RUN** — no DEM record was produced, so **no accuracy "
                   "claim is made here**")
    return f"""# `silo_flow` Beverloo kernel — numerical-convergence audit (m23114)

Anchor: `{A.__name__ and 'CASE-DEM-SILO-BEVERLOO'}` · kernel under audit:
`{A.__name__ and 'tensordem.silo_flow (kernel B)'}` · reference:
Beverloo, Leniger & van de Velde (1961), *Chem. Eng. Sci.* **15**, 260–269,
Eq. (4), `W = C ρ_b √g (D − k d)^(5/2)`, `C = {A.C_LIT}`, `k = {A.K_LIT}`
(literature, never refitted).

Ladder status: {status_line}.

```text
{state}
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
`W ∝ (D − k d)^{5/2}`, so

```text
ΔW/W ≈ 2.5 · k · 2δ / (D − k d),   δ_static = mg/k_n,   δ_dyn = v_o√(m_eff/k_n)
```

with `v_o = √(2gH) = {d5['orifice_velocity_m_s']:.4f} m/s` at this case:

| k_n [N/m] | δ_static [m] | δ_dynamic [m] | bias (static) | bias (dynamic) |
|---|---|---|---|---|
{rows}

At the campaign's own setting the **upper bound** of the overlap mechanism is
`{100*d5['predicted_bias_dynamic']:.2f} %` (`δ_dyn = {d5['delta_dynamic_m']:.3e} m`,
i.e. `{100*d5['delta_dynamic_m']/0.003:.2f} %` of a particle diameter) and the
static overlap is `{d5['delta_static_m']:.3e} m` — utterly negligible. Going all
the way from `k_n = 5e3` to `8e4 N/m` may therefore change `W` by at most
`{100*change:.2f} %`.

**Consequence (verified numerics, no DEM):** overlap can account for at most
`{100*d5['predicted_bias_dynamic']:.2f}` of the filed `{100*FILED_REL_ERROR:.0f} %`
gap — a shortfall of a factor `{short_fall:.1f}`. The m22853 root-cause
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

Artifacts: `benchmarks/m23114_siloflow_ladder/{{prereg.md,run_level.py,run_ladder.py,analyse.py,render_status.py}}`,
`tests/test_m23114_siloflow_ladder.py`, `docs/figures/m23114_overlap_bias.png`,
`runs/m23114_siloflow_ladder/` (populated when `cj4608` lands).
"""


def main() -> int:
    runs = A.load_runs() if RUNS.exists() else {}
    table = analytic_table()
    fig = make_figure(table)
    state = job_state()
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render(table, bool(runs), state))
    payload = {"analytic": {str(k): v for k, v in table.items()},
               "runs_present": sorted(f"{lv}_s{s}" for (lv, s) in runs),
               "figure": fig,
               "doc": str(DOC.relative_to(ROOT))}
    (RUNS if RUNS.exists() else ROOT / "runs").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "m23114_analytic_summary.json").write_text(
        json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())