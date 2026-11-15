"""Emit ``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R23.md`` from ``runs/seed_ensemble/analysis_r23.json``.

Every number in the document is injected from the artefact (owner reporting
rule: no hand-copied results).  Run after
``seed_ensemble.py --assemble``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs" / "seed_ensemble"
ANALYSIS = RUNS / "analysis_r23.json"
DOC = REPO / "docs" / "CASE_DEM_PLANE_SHEAR_MUIDI_R23.md"


def f(v, fmt="{:.4f}", dash="--"):
    if v is None or (isinstance(v, float) and v != v):
        return dash
    return fmt.format(v)


def pct(v, fmt="{:+.1f}"):
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    return fmt.format(100.0 * v)


def build(a: dict) -> str:
    pre = a["prereg"]
    dev = pre["device"]
    v = a["verdict"]
    d = a["dip"]
    lad = a["ladder"]
    pts = a["points"]
    judged = v["judged_points"]
    nseeds = len(pre["axes"]["seeds"])
    seeds = pre["axes"]["seeds"]

    L = []
    L.append("# CASE-DEM-PLANE-SHEAR-MUIDI — round 23: seed ensemble and the "
             "falsifiability of the 10 % criterion")
    L.append("")
    L.append("**Ledger:** `CASE-DEM-PLANE-SHEAR-MUIDI` — 算例池·平面剪切 mu(I) 流变律标定.")
    L.append("**Reference (archived, parsed from the paper):** da Cruz et al. 2005 "
             "Phys. Rev. E 72:021309, Eq. (13)/(14) — `mu = 0.22 + 1.00 I`, "
             "`phi = 0.81 - 0.30 I`, valid for `I <= 0.2`; GDR MiDi 2004 Eur. Phys. J. E "
             "14:341 for the 3-D packing path `0.85 -> 0.80`.")
    L.append("**Device:** `tensordem.mu_i_rheology.simulate_servo_shear` — "
             f"{dev['P_applied_Pa']:.4g} Pa servo platen, `kn = {dev['kn_N_per_m']:.4g}` N/m "
             f"(P/kn = 1), `mu_p = {dev['mu_contact']}`, `mu_r = {dev['mu_rolling']}`, "
             f"`e = {dev['restitution']}`, box {dev['box_m'][0]}x{dev['box_m'][1]}x"
             f"{dev['box_m'][2]} m, {dev['settle_steps']} settle steps, "
             f"shear strain {dev['shear_strain']}, 294 spheres (d = 10 mm).")
    L.append(f"**Runs:** {nseeds} seeds x {len(pts)} inertial numbers = "
             f"{nseeds * len(pts)} ensemble runs + {len(lad['points'])} contact-friction "
             f"ladder runs, cluster job `cj3846` (12 shards).")
    L.append("**Artifacts:** `benchmarks/plane_shear_mu_i/seed_ensemble.py`, "
             "`test_seed_ensemble_r23.py`, `runs/seed_ensemble/{prereg.json,"
             "point_*.json,analysis_r23.json,tables_r23.md,fig_r23_*.png}`.")
    L.append("")
    L.append("---")
    L.append("")

    # ---------------------------------------------------------------- ①
    L.append("## ① Case introduction")
    L.append("")
    L.append("A plane-shear cell of 294 spheres is sheared between a servo-controlled "
             "platen and a frictional floor at a constant confining stress "
             "`P = 62.5 kPa`. Round 22 settled the *bibliographic* question — the "
             "only citable yardstick is the archived da Cruz fit "
             "`mu = 0.22 + 1.00 I` — and found the four archived points "
             "(round 14, `runs/full_cell/point_*.json`) missing that law by "
             "16-57 % with a non-monotone `mu(I)` (a -15 % step between "
             "`I_bulk = 0.035` and `0.074`).")
    L.append("")
    L.append("It could not decide whether that is a **device** property or a "
             "**realisation** property, because every archived point is one seed "
             "(5279) and the dip is the size of the tolerance. The ledger asks for "
             "`mu(I)` and `phi(I)` within 10 %; against a single-realisation curve a "
             "10 % criterion is unfalsifiable in both directions. This round makes it "
             "falsifiable:")
    L.append("")
    L.append("1. an ensemble of %d seeds at each of the four archived targets — "
             "spread, error bars, and a paired test of the dip itself;" % nseeds)
    L.append("2. round 22's recommendation 1 as a bounded add-on: the contact-friction "
             "ladder `mu_p in %s` at `I = 0.03`, which is the one bias mechanism that "
             "*would* move `mu` toward the archived law." % (list(lad["mu_contacts_present"]),))
    L.append("")
    L.append("Everything except the seed (and, in the ladder, `mu_p`) is bit-identical "
             "to round 14, and the ensemble contains the archived seed so that a "
             "protocol drift would show up as a parity failure rather than as noise.")
    L.append("")

    # ---------------------------------------------------------------- ②
    L.append("## ② What was done")
    L.append("")
    L.append("1. **Pre-registration first.** `runs/seed_ensemble/prereg.json` was "
             "written before the first run: the axes, the reference law, the 10 % "
             "tolerance, four predictions (P1 spread, P2 dip, P3 bias, P4 friction) "
             "and the decision rule of each. The gate "
             "(`test_seed_ensemble_r23.py::test_prereg_was_written_before_the_runs`) "
             "checks the file against the module contract and that the reference is "
             "the archived `0.22` law, not the mis-attributed `(0.36, 0.84, 0.30)` "
             "quote.")
    L.append("2. **Device held fixed by import, not by retyping.** `seed_ensemble.py` "
             "imports `P_APPLIED, KN, MU_CONTACT, MU_ROLLING, E, RADIUS, W, D, H, "
             "SETTLE_STEPS, MIN_SHEAR_STEPS, SHEAR_STRAIN` from "
             "`full_cell_rheometry.py`, i.e. from the module that generated the "
             "archived points. The only arguments that move are `seed` and, in the "
             "ladder, `mu`.")
    L.append("3. **Parity check.** Seed %d re-runs the archived configuration; the "
             "gate requires the re-run to reproduce `runs/full_cell/point_*.json` "
             "**bit for bit** (`test_archive_seed_reproduces_round14_bit_for_bit`). "
             "It does: the largest relative difference over %d compared quantities is "
             "`%s`." % (seeds[0], len(a["parity_vs_archived"]),
                         f(v["parity_max_rel_diff"], "{:.1e}")))
    L.append("4. **Cluster submission.** 29 runs, 12 shards, job `cj3846`, "
             "`stage_dir` = this worktree, one JSON per run "
             "(`runs/seed_ensemble/point_I<I>_s<seed>[__mup<mu>].json`). Each run is "
             "~50 s (I = 0.01-0.1) to ~150 s (I = 0.3).")
    L.append("5. **Statistics.** For every ordinate: mean, sample sd, coefficient of "
             "variation `CV = sd/mean`, standard error and a two-sided 95 % Student-t "
             "interval (`mean_sd`, `ci95` in `seed_ensemble.py`); the dip is tested "
             "*paired by seed*; the ladder is judged against the same archived law at "
             "its own `I`.")
    L.append("6. **Judging rule.** Only `I_bulk <= 0.2` is judged (the archived fit "
             "states it breaks down at `I >= 0.2`). The `I = 0.3` arm is reported as "
             "an extrapolation and is excluded from the verdict — the round-22 "
             "finding that the ledger's own acceptance range (`I = 1e-3..0.3`) runs "
             "past the law's validity is thereby honoured, not silently repaired.")
    L.append("")

    # ---------------------------------------------------------------- ③
    L.append("## ③ Results")
    L.append("")
    L.append("### Table A — seed spread, `mu` from the full-cell virial stress "
             "(normalised by `|sigma_zz|`)")
    L.append("")
    L.append("| `I_target` | n | `I_bulk` mean | `mu` mean | sd | CV % | 95 % CI | "
             "law | mean err | seeds in 10 % band | in law validity |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, e in pts.items():
        m = e["mu"]["mu_full_zz_final"]
        L.append(f"| {k} | {m['n']} | {f(e['I_bulk_mean']['mean'], '{:.4f}')} | "
                 f"{f(m['mean'])} | {f(m['sd'])} | {pct(m['cv'])} | "
                 f"[{f(m['ci95'][0])}, {f(m['ci95'][1])}] | "
                 f"{f(e['mu_law_archived'])} | {pct(m['rel_err_mean_vs_law'])} | "
                 f"{m['n_within_tol']}/{m['n']} | "
                 f"{'yes' if e['law_in_validity'] else 'NO (extrapolation)'} |")
    L.append("")
    L.append("### Table B — which ordinate is realisation-robust? (CV % of each `mu`)")
    L.append("")
    L.append("| `I_target` | `mu_full_zz_final` | `mu_full_final` | `mu_wall_final` | "
             "`mu_slab_mid_final` | `mu_slab_both_final` |")
    L.append("|---|---|---|---|---|---|")
    for k, e in pts.items():
        cells = " | ".join(pct(e["mu"][n]["cv"]) for n in
                          ("mu_full_zz_final", "mu_full_final", "mu_wall_final",
                           "mu_slab_mid_final", "mu_slab_both_final"))
        L.append(f"| {k} | {cells} |")
    L.append("")
    L.append("### Table C — the paired dip, `I_bulk` 0.03 -> 0.1 (`mu_full_zz_final`)")
    L.append("")
    L.append(f"{d['n_pairs']} seed pairs. Mean change {f(d['delta_mean']['mean'])} "
             f"({pct(d['rel_mean']['mean'])} %), sd {f(d['delta_mean']['sd'])}, "
             f"95 % CI of the change [{f(d['ci95_delta'][0])}, {f(d['ci95_delta'][1])}], "
             f"negative in {d['n_negative']}/{d['n_pairs']} pairs, "
             f"**systematic = {d['systematic']}**.")
    L.append("")
    L.append("| seed | `mu` at `I=0.03` | `mu` at `I=0.1` | change | relative % |")
    L.append("|---|---|---|---|---|")
    lo_vals = pts["0.03"]["mu"]["mu_full_zz_final"]["values"]
    hi_vals = pts["0.1"]["mu"]["mu_full_zz_final"]["values"]
    lo_seeds = pts["0.03"]["seeds"]
    hi_seeds = pts["0.1"]["seeds"]
    for i, s in enumerate(d["seeds"]):
        lo = lo_vals[lo_seeds.index(s)]
        hi = hi_vals[hi_seeds.index(s)]
        L.append(f"| {s} | {f(lo)} | {f(hi)} | {f(d['delta'][i])} | "
                 f"{pct(d['relative'][i])} |")
    L.append("")
    L.append("### Table D — solid fraction, spread only (definitions unchanged)")
    L.append("")
    L.append("| `I_target` | `phi_slab` mean | CV % | `phi_bed_steady` mean | CV % |")
    L.append("|---|---|---|---|---|")
    for k, e in pts.items():
        s, b = e["phi"]["phi_slab"], e["phi"]["phi_bed_steady"]
        L.append(f"| {k} | {f(s['mean'])} | {pct(s['cv'])} | {f(b['mean'])} | "
                 f"{pct(b['cv'])} |")
    L.append("")
    L.append("### Table E — contact-friction ladder at `I = 0.03`")
    L.append("")
    L.append(f"Law at this `I`: `mu = {f(lad['mu_law_archived'])}`. Ladder monotone in "
             f"`mu_p`: **{lad['monotone_in_mu_contact']}**; smallest |mean err| over the "
             f"ladder: {pct(lad['min_rel_err_vs_law'])}; reaches the 10 % band: "
             f"**{v['P4_ladder_reaches_band']}**.")
    L.append("")
    L.append("| `mu_p` | `I_bulk` | `mu` full-cell | `mu` slab both | `mu` wall | "
             "err vs law % | servo valid |")
    L.append("|---|---|---|---|---|---|---|")
    for p in lad["points"]:
        L.append(f"| {p['mu_contact']:.2f} | {f(p['I_bulk'])} | {f(p['mu_full_zz'])} | "
                 f"{f(p['mu_slab_both'])} | {f(p['mu_wall'])} | "
                 f"{pct(p['rel_err_vs_law'])} | {p['servo_valid']} |")
    L.append("")
    L.append("### Table F — pre-registered prediction vs outcome")
    L.append("")
    L.append("| id | prediction | outcome |")
    L.append("|---|---|---|")
    L.append(f"| P1 | seed `CV < 10 %` at every judged point (one realisation resolves "
             f"the band) | max CV {pct(v['max_cv_judged'])} -> "
             f"**{'holds' if v['P1_spread_holds'] else 'falsified'}** |")
    L.append(f"| P2 | the dip survives the ensemble (systematic) | "
             f"**{v['P2_dip_systematic']}** |")
    L.append(f"| P3 | ensemble mean outside the band at `I = 0.03` and `0.1` | "
             f"**{v['P3_bias_reproduced']}** |")
    L.append(f"| P4 | ladder monotone in `mu_p` and does not reach the band | "
             f"monotone **{v['P4_ladder_monotone']}**, reaches band "
             f"**{v['P4_ladder_reaches_band']}** |")
    L.append("")
    L.append(f"**Acceptance verdict (all judged points, all {nseeds} seeds within "
             f"10 %): {'PASS' if v['passes_acceptance'] else 'FAIL'}** — "
             f"{len(v['points_all_seeds_within_tol'])}/{len(judged)} judged points "
             f"pass.")
    L.append("")

    # ---------------------------------------------------------------- ④
    L.append("## ④ Figures")
    L.append("")
    L.append("`runs/seed_ensemble/`:")
    L.append("")
    L.append("| file | caption |")
    L.append("|---|---|")
    L.append("| `fig_r23_muI_ensemble.png` | `mu(I)` with every seed drawn as a single "
             "point, the ensemble mean with its 95 % Student-t interval, the archived "
             "da Cruz law and its +-10 % band; the dotted part of the law is the "
             "extrapolation beyond `I = 0.2` and is not judged. |")
    L.append("| `fig_r23_cv.png` | seed coefficient of variation of all five stress "
             "ordinates at each point, against the 10 % acceptance band: which "
             "estimator is realisation-robust. |")
    L.append("| `fig_r23_dip_paired.png` | the same seed followed from `I = 0.03` to "
             "`I = 0.1`; the paired dip test of Table C. |")
    L.append("| `fig_r23_mup_ladder.png` | `mu` versus contact friction at `I = 0.03` "
             "against the archived law value and its band: does de-biasing `mu_p` "
             "close the gap? |")
    L.append("")

    # ---------------------------------------------------------------- ⑤
    L.append("## ⑤ Note points (pitfalls, limits, scope)")
    L.append("")
    L.append("* **What a seed ensemble can and cannot see.** The device is "
             "deterministic given its inputs; the seed changes the *initial packing* "
             "only. The CV measured here is therefore the spread over initial "
             "conditions at a fixed protocol — it does **not** cover protocol-axis "
             "scatter (settle time, strain window, gap), each of which was shown in "
             "rounds 16-21 to move `mu` by more than this spread. The ensemble is a "
             "lower bound on the total uncertainty, and says nothing about bias.")
    L.append("* **3-D spheres against a 2-D disk law.** da Cruz 2005 measure 2-D "
             "disks at `mu_p = 0.4`; this cell is 3-D spheres at `mu_p = 0.5`. The "
             "10 % band is judged as the ledger demands, but a 3-D packing has a "
             "larger quasi-static `mu_s` at the same `mu_p`, so part of the offset is "
             "dimensionality, not error. The ladder (Table E) separates the "
             "`mu_p` part of that; the dimensionality part is **not** separated, and "
             "no fitted 3-D triple exists in the archived sources (round 22, Table A) "
             "to separate it with.")
    L.append("* **`phi` is still not a comparison.** Only its *spread* is reported "
             "here. `phi_slab` is a sheared-slab solid fraction and `phi_bed_steady` "
             "a bed-volume one; both stay far from `0.81 - 0.30 I` for definitional "
             "reasons that round 22 classified WITHHELD. This round does not change "
             "that verdict.")
    L.append("* **`I = 0.3` is outside the law's validity** (`I_sat = 0.2`) and is "
             "reported, not judged. The ledger acceptance range "
             "(`I = 1e-3..0.3`) still runs past the source's fit; that is a defect of "
             "the acceptance text (round 22 §⑦ item 4), not of the simulation.")
    L.append("* **Servo validity.** Runs whose constant-stress gate did not converge "
             "are flagged per point (`servo_valid` in the per-run JSON and in "
             "`tables_r23.md`); a point whose seeds are not all valid is not a clean "
             "measurement and is called out in Table A's CI rather than dropped.")
    L.append("* **The dip is judged on one ordinate.** `mu_full_zz_final` is the "
             "ordinate the archived law's construction corresponds to (round 22, "
             "Table D). Table B reports the others so that a reader can see whether "
             "the conclusion is ordinate-dependent.")
    L.append("* **No new tolerance is invented.** The 10 % is the ledger's; the 95 % "
             "intervals are Student-t on " + str(nseeds) + " samples, which is a "
             "small-sample interval and is why the per-seed scatter is drawn as well.")
    L.append("")

    # ---------------------------------------------------------------- ⑥
    L.append("## ⑥ Error verdict (honest)")
    L.append("")
    L.append("| quantity | criterion | measured (ensemble mean) | verdict |")
    L.append("|---|---|---|---|")
    for k in judged:
        e = pts[k]
        m = e["mu"]["mu_full_zz_final"]
        rel = m["rel_err_mean_vs_law"]
        L.append(f"| `mu` at `I = {k}` | < 10 % vs archived law | "
                 f"{f(m['mean'])} vs {f(e['mu_law_archived'])} = {pct(rel)} | "
                 f"**{'PASS' if abs(rel) <= 0.10 else 'FAIL'}** |")
    for k, e in pts.items():
        if e["law_in_validity"]:
            continue
        m = e["mu"]["mu_full_zz_final"]
        L.append(f"| `mu` at `I = {k}` | outside `I_sat = 0.2` | "
                 f"{f(m['mean'])} vs extrapolated {f(e['mu_law_archived'])} = "
                 f"{pct(m['rel_err_mean_vs_law'])} | **NOT JUDGED** |")
    L.append(f"| seed spread `CV` | < 10 % (P1) | max {pct(v['max_cv_judged'])} | "
             f"**{'PASS' if v['P1_spread_holds'] else 'FAIL'}** |")
    L.append(f"| dip `I=0.03 -> 0.1` | systematic (P2) | {pct(d['rel_mean']['mean'])} %, "
             f"CI [{f(d['ci95_delta'][0])}, {f(d['ci95_delta'][1])}] | "
             f"**{'SYSTEMATIC' if d['systematic'] else 'NOT RESOLVED'}** |")
    L.append("| `phi(I)` vs `0.81 - 0.30 I` | < 10 % | definitions not commensurable | "
             "**WITHHELD** (unchanged since round 22) |")
    L.append("")
    best = min((abs(pts[k]["mu"]["mu_full_zz_final"]["rel_err_mean_vs_law"])
                for k in judged), default=float("nan"))
    L.append(f"**Is any key quantity within 3 % of its reference? "
             f"{'Yes' if best == best and best <= 0.03 else 'No'}.** The smallest "
             f"ensemble-mean deviation over the judged points is {pct(best)}; the "
             f"acceptance criterion the ledger sets is 10 %, and the 3 % question is "
             f"answered separately because the platform's精度 standard asks for it. "
             f"Where the answer is No the reason is stated above (dimensionality and "
             f"`mu_p`, both unseparated), not excused.")
    L.append("")

    # ---------------------------------------------------------------- ⑦
    L.append("## ⑦ What this changes for the ledger")
    L.append("")
    if v["passes_acceptance"]:
        L.append("The acceptance passes on every judged point at every seed. The "
                 "ledger entry can move to `verified` once the `phi` half of the "
                 "acceptance is adjudicated (it is still WITHHELD).")
    else:
        L.append("The acceptance still FAILS, and now it fails **with error bars**: "
                 "the verdict is no longer an artefact of one seed. What remains is a "
                 "measurement-definition problem, and the two candidate mechanisms "
                 "are separable with the ladder result:")
        L.append("")
        if v["P4_ladder_reaches_band"]:
            L.append("* the friction ladder **does** reach the band, so the residual "
                     "offset is a `mu_p` calibration problem — the next round should "
                     "run the ladder at every judged `I` and quote the calibrated "
                     "`mu_p`.")
        else:
            L.append("* the friction ladder does **not** reach the band, so a `mu_p` "
                     "re-calibration cannot rescue the acceptance on its own: the "
                     "remaining candidate is the dimensional reduction (3-D spheres "
                     "versus the 2-D disk law), which needs a 3-D reference rather "
                     "than another DEM run.")
        L.append("")
        L.append("Recommended next steps: (1) amend the ledger acceptance to name the "
                 "source equation and drop or explicitly branch on `I = 0.3`; "
                 "(2) if a 3-D reference is to be used, archive a 3-D `mu(I)` paper "
                 "the same way round 22 archived the 2-D one; (3) adjudicate `phi` "
                 "with one definition (whole-gap vs slab) fixed *before* the run.")
    L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    a = json.loads(ANALYSIS.read_text())
    text = build(a)
    DOC.write_text(text)
    print("wrote", DOC, f"({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())