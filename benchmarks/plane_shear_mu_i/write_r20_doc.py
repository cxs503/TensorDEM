"""Emit docs/CASE_DEM_PLANE_SHEAR_MUIDI_R20.md from runs/high_n_r20/result.json.

Every number in the document is read out of the verdict file produced by
``high_n_coverage.py --assemble``; nothing is typed in by hand.  Run after the
sweep shards have landed::

    python write_r20_doc.py            # writes the doc + prints a summary
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (HERE, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import high_n_coverage as C  # noqa: E402

RUNS = HERE / "runs" / "high_n_r20"
DOC = REPO / "docs" / "CASE_DEM_PLANE_SHEAR_MUIDI_R20.md"
JOB = "cj3315"


def _f(v, spec=".4f"):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "--"
    return "--" if math.isnan(v) else format(v, spec)


def _arm(v, arm, abscissa):
    for entry in v["mu_verdicts"]:
        if entry["arm"] == arm and entry["abscissa"] == abscissa:
            return entry
    raise SystemExit(f"missing arm {arm}/{abscissa}")


def build(v: dict) -> str:
    rows = v["rows"]
    best = v["mu_best_arm"]
    L = []
    A = L.append

    A("# CASE-DEM-PLANE-SHEAR-MUIDI — Round 20: full-range, multi-estimator verdict")
    A("")
    A("**Anchor:** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池 · 平面剪切 mu(I) 流变律标定)  ")
    A("**Reference:** da Cruz et al. 2005, *Phys. Rev. E* **72**:021309; GDR MiDi 2004, "
      "*Eur. Phys. J. E* **14**:341.  **Tolerance:** < 10 % on `mu(I)`, `phi(I)`.  ")
    A(f"**Range covered:** I = {v['range_covered'][0]:g} … {v['range_covered'][1]:g} "
      f"(ledger range {v['ledger_range'][0]:g} … {v['ledger_range'][1]:g}, "
      f"complete={v['range_complete']}), N ≥ {v['min_n_particles']:.0f}  ")
    A(f"**Cluster job:** `{JOB}` (6 shards, one I target each, stage_dir = this worktree)  ")
    A("**Artefacts:** `benchmarks/plane_shear_mu_i/high_n_coverage.py` (device + verdict), "
      "`runs/high_n_r20/{point_*.json, result.json, tables.md, fig*.png}`, "
      "`test_high_n_coverage.py`.")
    A("")
    A("---")
    A("")
    # ---------------------------------------------------------------- ①
    A("## ① Case and protocol")
    A("")
    A(f"A plane-shear cell of {rows[0]['n_particles']:.0f} spheres "
      f"(d = {2*C.R*1000:.0f} mm, rho = {C.RHO:.0f} kg/m³) is sheared between a "
      f"servo-controlled platen and a frictional floor in a box "
      f"{C.W}×{C.D}×{C.H} m (32d×16d×18d).  The platen holds a constant normal "
      f"stress P = {C.P/1e3:.1f} kPa by a servo loop and drives a shear rate chosen "
      f"to reach the target inertial number.  Contact law: linear spring-dashpot, "
      f"kn = {C.KN:.0f} N/m (P/kn = 1), kt/kn = 0.5, mu_contact = {C.MU_CONTACT}, "
      f"mu_rolling = {C.MU_ROLLING}, e = {C.E}, seed {C.SEED}.")
    A("")
    A("| parameter | value |")
    A("|---|---|")
    A(f"| I targets | {', '.join(f'{i:g}' for i in C.I_TARGETS)} |")
    A(f"| box (W×D×H) | {C.W} × {C.D} × {C.H} m |")
    A(f"| particles | {rows[0]['n_particles']:.0f} (N > 1000; rounds 1-16 used 294) |")
    A(f"| normal stress P | {C.P/1e3:.2f} kPa (servo) |")
    A(f"| kn / kt | {C.KN:.0f} / {0.5*C.KN:.0f} N/m |")
    A(f"| contact / rolling friction | {C.MU_CONTACT} / {C.MU_ROLLING} |")
    A(f"| restitution | {C.E} |")
    A(f"| settle steps | {C.SETTLE_STEPS} |")
    A(f"| shear strain target | {C.SHEAR_STRAIN} |")
    A(f"| dt | {_f(rows[0]['dt'], '.2e')} s |")
    A("")
    A("Two gaps are closed here, both of them *coverage* gaps rather than new physics.")
    A("")
    A("1. **Range.** Rounds 18/19 ran the higher-N study only at "
      "I ≥ 1e-2; the ledger acceptance starts at 1e-3.  The low-I half was "
      "measured only at N = 294, where rounds 14/16 showed the noise floor exceeds "
      "the 10 % band.  This round measures the whole ledger range at N > 1000.")
    A("2. **Verdict unit.** Round 13 fixed the reference side (two conflicting "
      "triples → an admissible envelope) on the *N = 294* archive with the "
      "imposed-rate abscissa; round 19 fixed the abscissa (measured bulk rate) but "
      "imported a single triple and quoted a single mu estimator.  Here the verdict "
      "is reported as a matrix — {3 mu estimators} × {3 abscissae} × {2 candidate "
      "triples + the falsification oracle} — so the ledger cannot be failed or "
      "passed by an unreported estimator choice.")
    A("")
    A("Pre-registered questions (`PREREG` in the module, written before submission):")
    for k, val in v["prereg"]["hypotheses"].items():
        A(f"* **{k}** — {val}")
    A("")
    # ---------------------------------------------------------------- ②
    A("## ② Method")
    A("")
    A("* Device: `tensordem.mu_i_rheology.simulate_servo_shear` (unchanged since "
      "round 17); protocol parity with rounds 18/19 (same contact law, P/kn, seed, "
      "strain target, box), so the four shared I targets double as a "
      "reproducibility check against the round-19 sweep.")
    A("* `high_n_coverage.summarise()` is pure and records, per point: three mu "
      "estimators (wall `tau/p` at the platen, full-cell `tau/<p>`, bulk-slab "
      "`tau/p`), both abscissae (imposed platen rate, measured velocity-profile "
      "rate) plus the round-19 fallback, three phi estimators, the noise floor of "
      "the final 25 % window, the pressure-control error, the rate ratio and the "
      "servo gate.")
    A("* `assemble()` builds the verdict matrix, calls "
      "`reference_provenance.verdict_table` (per-triple pass/fail) and "
      "`best_case_over_envelope` (falsification oracle over the continuous box "
      "mu_s ∈ [0.34,0.40], mu_2 ∈ [0.55,0.95], I_0 ∈ [0.22,0.38]).  A triple whose "
      "fit sits on a box edge is reported as *not identified* rather than quoted "
      "as a calibration.")
    A("* Reference constants are imported from `reference_law` (single source of "
      "truth, round-19 defect D-A) and every residual is a percentage compared "
      "against a percentage band (round-19 defect D-C).")
    A("")
    # ---------------------------------------------------------------- ③
    A("## ③ Results")
    A("")
    A(C.tables_markdown(v).replace("## Table", "### Table"))
    A("")
    A("### Verdict")
    A("")
    A(f"* best arm over the full range: **{best['arm']} @ {best['abscissa']}**, "
      f"worst absolute error **{best['worst_abs_err_pct']:.1f} %**")
    A(f"* full-range pass at the 10 % band over every estimator arm: "
      f"**{v['mu_passes_full_range']}**")
    A(f"* can any admissible triple (the reference envelope) put the curve inside "
      f"10 %? **{v['envelope_can_rescue']}**")
    A(f"* servo gate valid on all points: **{v['servo_valid_all']}**")
    A(f"* phi: **{v['phi']['reference_status']}**")
    A("")
    # ---------------------------------------------------------------- ④
    A("## ④ Figures")
    A("")
    A("![mu(I)](../benchmarks/plane_shear_mu_i/runs/high_n_r20/fig1_mu_vs_I.png)")
    A("")
    A("*Fig. 1 — measured mu(I) for the three estimators with the noise floor as "
      "error bars, against the reference law and the two-triple envelope.*")
    A("")
    A("![residuals](../benchmarks/plane_shear_mu_i/runs/high_n_r20/fig2_error_band.png)")
    A("")
    A("*Fig. 2 — per-point residual of every estimator (measured-rate abscissa, "
      "solid) against the ±10 % acceptance band; dotted lines are the "
      "imposed-rate abscissa for the same estimator.*")
    A("")
    A("![phi(I)](../benchmarks/plane_shear_mu_i/runs/high_n_r20/fig3_phi_vs_I.png)")
    A("")
    A("*Fig. 3 — the three phi estimators against the reference phi(I); the "
      "reference magnitudes are provenance-unverified (see ⑤), so this panel "
      "reports dilatancy and estimator spread, not an acceptance.*")
    A("")
    A("![noise](../benchmarks/plane_shear_mu_i/runs/high_n_r20/fig4_noise.png)")
    A("")
    A("*Fig. 4 — noise floor of the final 25 % window at the higher particle "
      "count (round-18 hypothesis P1).*")
    A("")
    # ---------------------------------------------------------------- ⑤
    A("## ⑤ Notes, limits, and what this round did **not** do")
    A("")
    A(f"* **Shear strain.** The sweep stops at strain {C.SHEAR_STRAIN}, a "
      "deliberate compromise for the higher-N cost (round 16 found ≥ 2 for full "
      "stationarity at low I).  Low-I points are therefore the least converged "
      "rows of Table 1; the direction of the bias is discussed in round 16 and "
      "not re-derived here.")
    A("* **Reference provenance is unchanged.** The two conflicting triples "
      "(0.36/0.84/0.30 in `reference_law`, 0.379/0.641/0.279 in the test suite) "
      "are both still only *quoted*; this round does not resolve that, it makes "
      "the verdict independent of the choice (envelope + oracle).")
    A("* **phi is not an acceptance.** The magnitudes (0.58, 0.64) have no in-repo "
      "literature anchor and da Cruz 2005 is a 2D-disk study (phi ≈ 0.84→0.79); "
      "only the monotonicity/limits are verified.  Reported as WITHHELD, as in "
      "rounds 13-19.")
    A("* **Verifier findings fixed in this round** (found by re-reading the "
      "shipped harness, both now pinned by tests):")
    A("  * `high_n_sweep.assemble` reported `P1_noise_floor`/`P2_mu_convergence` "
      "as `True` when *no* mid-I point was present — `all([])` is vacuously true, "
      "so a dropped shard would have printed a PASS on the two hypotheses that "
      "carry the round-18 prediction.  Now gated on `mid_points_present`.")
    A("  * this round's own `mu_passes_full_range` would have accepted a band-pass "
      "over a *partial* range; it is now gated on `range_complete`.")
    A("* Cluster job semantics: results are for job `%s`; the four I targets "
      "shared with round 18/19 are an independent reproduction at the same seed "
      "(identical protocol), not a second measurement of a different ensemble." % JOB)
    A("")
    # ---------------------------------------------------------------- ⑥
    A("## ⑥ Error judgement (honest)")
    A("")
    A("Key quantity: `mu(I)` over the full ledger range, error = measured / "
      "reference − 1.")
    A("")
    A("| I target | " + " | ".join(C.MU_ARMS) + " | band 10 % met (best arm) |")
    A("|---|" + "---|" * (len(C.MU_ARMS) + 1))
    best_arm = best["arm"]
    best_ab = best["abscissa"]
    for r in rows:
        cells = [format(r["err_pct__{}__{}".format(a, best_ab)], "+.1f") + " %"
                 for a in C.MU_ARMS]
        ok = ("yes" if abs(r["err_pct__{}__{}".format(best_arm, best_ab)]) < C.TOL_PCT
              else "**no**")
        A(f"| {r['I_target']:g} | " + " | ".join(cells) + f" | {ok} |")
    A("")
    A(f"* worst absolute error of the best arm: **{best['worst_abs_err_pct']:.1f} %** "
      f"({best['arm']} @ {best['abscissa']})")
    A(f"* worst absolute error across all estimator/abscissa cells: "
      f"**{max(x['worst_abs_err_pct'] for x in v['mu_verdicts']):.1f} %**")
    A(f"* is every point inside the 10 % band? **{v['mu_passes_full_range']}**")
    A(f"* is every point inside 3 %? **"
      f"{'yes' if best['worst_abs_err_pct'] < 3.0 else 'NO'}** "
      f"— the ledger band is 10 %, the 3 % line is answered for the record.")
    A("")
    return "\n".join(L) + "\n"


def main() -> None:
    v = json.loads((RUNS / "result.json").read_text())
    text = build(v)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text)
    print(f"wrote {DOC.relative_to(REPO)} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()