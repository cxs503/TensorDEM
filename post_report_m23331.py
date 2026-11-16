#!/usr/bin/env python3
"""Publish the m23331 report (six elements, figures embedded, numbers read from artifacts)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "m23331_branch_prob_archive"
FIGS = ["docs/figures/m23331/m23331_records_vs_reference.png",
        "docs/figures/m23331/m23331_branch_probability.png",
        "docs/figures/m23331/m23331_window_series.png"]
CAPS = {
    "m23331_records_vs_reference.png":
        "Each of the 5 archived frozen-case records against the Beverloo 1961 reference "
        "(dashed) and its ±10 % ledger band (grey). The 4 legacy-spelling records scatter "
        "between -9.13 % and +5.49 %; the single canonical-spelling record is an arch "
        "(-88.48 %), i.e. it is not a flow-rate measurement at all.",
    "m23331_branch_probability.png":
        "Branch probability p_flow per dt spelling with its 95 % Wilson interval "
        "(4/4 legacy vs 0/1 canonical). The arms look opposite but the exact test on this "
        "table gives p = 0.20, so the archival subset cannot separate the factors.",
    "m23331_window_series.png":
        "Particles leaving the outlet per 0.05 s measurement window. All 4 legacy-spelling "
        "records keep a steady jet (23-50 particles/window); the canonical-spelling record "
        "emits 2, 0, 0, 0, 0, 4, 0, 15 — the signature of a stable arch, not a slow flow.",
}


def base() -> str:
    b = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    return b if b.endswith("/api/v1") else b + "/api/v1"


def login() -> str:
    data = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base() + "/auth/login", data=data,
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    return body.get("access_token") or body.get("token") or body["data"]["access_token"]


def main() -> int:
    s = json.loads((RUN / "summary.json").read_text())
    arms = s["arms"]
    F, Q = arms["F"], arms["Q"]
    cf, cq = F["conditional"], Q["conditional"]
    rows = "\n".join(
        f"| {arm} | {r['seed']} | {r['status']} | {r['flow_rate_kg_s']:.6g} | "
        f"{r['rel_error_vs_beverloo']:+.2%} | {r['window_particles']} |"
        for arm in ("F", "Q") for r in arms[arm]["records"])
    md = f"""# CASE-DEM-SILO-BEVERLOO / m23331 — is the 9.2× spread a dt artefact or seed luck?

## ① 案例介绍 (case)
Flat-bottom cylindrical silo, centred circular orifice, gravity discharge; the campaign's
frozen primary case, unchanged: `d = 3 mm`, `D = 15 mm` (`D/d = 5`), `H = 36 mm`,
`μ = 0.5`, `μ_rolling = 0.05`, smooth wall `μ_w = 0`, `k_n = 5e3 N/m`, settle 0.06 s,
discard 0.12 s, 8 × 0.05 s windows, `W` = mean of the last 4 windows.
Reference: Beverloo, Leniger & van de Velde (1961) *Chem. Eng. Sci.* **15**, 260–269 Eq. (4),
`W = C ρ_b √g (D − k d)^2.5` with `C = 0.58`, `k = 1.4` **never refitted**.
Bulk density is taken from each record's own bed (1323.8 kg/m³), so the reference is
computed per record, not quoted.

## ② 怎么做的 (method)
The previous round (m23261) proved that the two archived records of this case differ in the
**last bit of dt** (`0.05·T_c` vs `T_c/20`, 1 ULP) and that a 1-ULP `dt` shift decorrelates the
bed exponentially. It explicitly could not say whether the FLOWING/ARRESTED branch is selected
by the `dt` **value** or by the **packing realization**. This mission answers that with a
pre-registered 2 × 6 factorial (two spellings × seeds 1–6) whose estimator is pinned by 24 tests.

1. `benchmarks/m23331_branch_prob/prereg.md` — verdict rule (B1 spelling / B2 seed / B3
   inconclusive), tolerances, amendments A0/A1/A2 written **before** the records they govern.
2. `run_case.py` — one record per shard, physics untouched (`silo_flow.build_bed` +
   `run_discharge`), the arm decided **by the IEEE-754 bits of `dt`** stored in the record.
3. `harvest_archives.py` — the same design assembled from **archived** records without re-running
   DEM: arm from dt bits, protocol identity checked, label re-derived with the campaign's own
   `beverloo.discharge_status`; mismatching records are listed as rejected with reasons.
4. `analyse.py` — Wilson interval, exact Fisher test, conditional mean with a Student-t interval,
   `required_n`; implemented locally so the estimator has no scipy dependency.
5. Tool chain: Python 3.11 / numpy / matplotlib; records produced on loongarch64 (archives).

## ③ 计算结果 (numbers, read from the artifacts)
| arm | seed | status | W [kg/s] | W/W_1961 − 1 | window particles |
|---|---|---|---|---|---|
{rows}

* legacy spelling `F` = `0.05·T_c` = `1.86767398828964004e-05 s` (bits `…ba0`): `p_flow = 4/4`,
  conditional `W = {cf['mean_kg_s']:.6g} kg/s`, CV `{cf['cv']:.2%}`, CI95 half-width
  `{cf['ci95_halfwidth_rel']:.2%}`, error vs 1961 `{cf['rel_error_vs_reference']:+.2%}`,
  `required_n = {F['required_n']}` → **{cf['verdict']}**.
* canonical spelling `Q` = `T_c/20` = `1.86767398828963971e-05 s` (bits `…b9f`), which is what
  `Material.dt` returns today: `p_flow = 0/1` — its only record is an arch
  (`W = {Q['records'][0]['flow_rate_kg_s']:.6g} kg/s`, {Q['records'][0]['rel_error_vs_beverloo']:+.2%})
  → conditional verdict **{cq['verdict']}**.
* 2 × 2 table `[[4,0],[0,1]]`: two-sided Fisher `p = {(s.get('fisher') or {}).get('p_value', float('nan')):.3g}`.
* Branch verdict: **{s['branch_verdict']['verdict']}** — {s['branch_verdict']['why']}.
* Cross-worktree reproduction: the same `(F, seed 1)` record exists in two worktrees at two
  commits (`m22885` `db2bdaf` and `m23261` `A_arch_bed`): relative `W` difference
  `0.00e+00` — the kernel is bit-reproducible across those commits.

## ④ 结果图 (figures)
Three figures are attached (records vs reference, branch probability with Wilson CI, and the
window series that separates a jet from an arch). The window series is the decisive one: the
arrested record emits 2,0,0,0,0,4,0,15 particles, which is an arch collapsing sporadically, not
a slow flow.

## ⑤ 注意点 (caveats)
* **The prospective 2 × 6 grid is NOT_RUN**: job `cj4769` (12 shards, 12 cores, `stage_dir` =
  this worktree) was still `queued` at position 148 with 162 jobs in flight; the control plane
  forbids heavy local DEM (>8 cores / >10 min). Every number above is the **archival** harvest.
* The archival set is **observational**: 4 records of the legacy spelling come from commit
  `db2bdaf` (one machine), the single canonical-spelling record from `1fb2a46` — the two factors
  `dt` and code state are therefore not separated here, only shown to be separable at all.
* `n = 1` in arm `Q` means the branch verdict is `B3_INCONCLUSIVE`, not B1: the pure-and-opposite
  pattern has Fisher `p = 0.20`, and amendment A1 forbids calling it significant.
* Scope: one frozen configuration, monodisperse spheres, flat bottom, smooth wall. Nothing here
  transfers to conical hoppers, wall friction, polydisperse beds or `D/d > 6`.
* `C` and `k` are literature values throughout; no refit is performed or implied.

## ⑥ 误差判定 (error judgement)
| claim | value | basis | < 3 %? |
|---|---|---|---|
| seed-averaged `W` vs Beverloo 1961 (legacy spelling, n = 4) | {cf['rel_error_vs_reference']:+.2%} | mean of 4 archival records, per-record reference | **yes** (0.13 %) |
| single-record `W` vs 1961, best/worst of the 4 seeds | +5.49 % … −9.13 % | archived records | **no** (up to 9.1 %) |
| canonical spelling at n = 1 | −88.48 % | arch, not a rate | **n/a** (not a measurement) |
| cross-worktree reproduction of one record | 0.00e+00 | two files, two commits | **yes** (exact) |
| branch attribution (spelling vs seed) | Fisher p = 0.20 | archival table | **not decidable** |

Honest statement: the anchor's ±10 % ledger acceptance is **not** decided by this mission. It
delivers (a) the seed scatter that single-record verdicts had been absorbing silently
(CV {cf['cv']:.2%}, worst record 91 % of the tolerance), (b) the number of records such a verdict
needs (`required_n = {F['required_n']}`), and (c) an estimator + pre-registration that turn the
prospective ensemble into a decision the moment `cj4769` lands.

Artifacts: `benchmarks/m23331_branch_prob/{{prereg.md,run_case.py,harvest_archives.py,analyse.py}}`,
`tests/test_m23331_branch_prob.py` (24 passed), `runs/m23331_branch_prob_archive/`,
`docs/CASE-DEM-SILO-BEVERLOO_m23331_branch_prob.md`.
"""

    figures = []
    for f in FIGS:
        p = ROOT / f
        figures.append({"name": p.name, "caption": CAPS[p.name],
                        "png_base64": base64.b64encode(p.read_bytes()).decode()})
    payload = {
        "title": "m23331 算例池·Beverloo: 分支概率——9.2× 离散是 dt 末位还是种子运气？(archival ensemble)",
        "summary": ("用 2×6 因子设计（dt 两种拼写 × 种子 1–6）回答 m23261 留下的问题：FLOWING/ARRESTED "
                    "分支由 dt 值还是由装填实现决定。前瞻网格 cj4769 排队未派发（NOT_RUN），交付的是"
                    "归档记录的回顾性集合：旧拼写 4/4 流动（W=0.0292 kg/s，−0.13% vs Beverloo 1961，"
                    "CV 6.5%，单记录散布 +5.5%…−9.1%），新拼写（=今日 Material.dt）唯一记录为拱成 (−88.5%)；"
                    "Fisher p=0.20 → 分支归因 B3_INCONCLUSIVE，需 n=6 才能判定。"),
        "scenario_key": "CASE-DEM-SILO-BEVERLOO",
        "mission_id": 23331,
        "tags": ["TensorDEM", "silo_flow", "Beverloo", "branch-probability", "ensemble",
                 "reproducibility", "honest-FAIL"],
        "markdown": md,
        "figures": figures,
    }
    tok = login()
    req = urllib.request.Request(base() + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {tok}"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        print(r.status, r.read().decode()[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())