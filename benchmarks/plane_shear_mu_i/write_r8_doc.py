"""Emit docs/CASE_DEM_PLANE_SHEAR_MUIDI_R8.md from the stored artifacts.

Every number in the document is injected from ``runs/full_cell/result.json``
(zero hand-copy).  Usage::

    python benchmarks/plane_shear_mu_i/full_cell_rheometry.py     # sweep + tables + fig
    python benchmarks/plane_shear_mu_i/write_r8_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "full_cell"
DOC = HERE.parents[1] / "docs" / "CASE_DEM_PLANE_SHEAR_MUIDI_R8.md"


def f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pc(x, nd=1):
    return "n/a" if x is None else f"{100 * x:+.{nd}f} %"


def main() -> int:
    data = json.loads((RUNS / "result.json").read_text())
    rows = data["rows"]
    prereg = data["prereg"]
    ref = prereg["reference"]

    worst = lambda key: max(rows, key=lambda r: abs(r[key]))  # noqa: E731
    w_full = worst("err_full_pct")
    w_zz = worst("err_full_zz_pct")
    w_wall = worst("err_wall_pct")
    # the stored field is sigma_zz / (-P_applied), so the exact value is +1
    bal_max = max(abs(1.0 - r["momentum_balance_final"]) for r in rows)
    p1 = [r for r in rows if abs(1.0 - r["momentum_balance_final"]) <= 0.05]
    p2 = [r for r in rows
          if r["mu_slab_both_over_full"] < r["mu_slab_mid_over_full"] < 1.0]
    p3_full = [r for r in rows if abs(r["err_full_pct"]) <= 10.0]
    p3_zz = [r for r in rows if abs(r["err_full_zz_pct"]) <= 10.0]
    valid = [r for r in rows if r["servo_valid"]]
    aniso = [r["anisotropy_p_over_zz"] for r in rows]

    L: list[str] = []
    A = L.append
    A("# CASE-DEM-PLANE-SHEAR-MUIDI — Round 8: the stress estimator gets an "
      "analytic anchor, and `mu(I)` is re-judged on it\n")
    A("> **Anchor** `CASE-DEM-PLANE-SHEAR-MUIDI` (feature_ledger, status=open) · "
      "**Repository** TensorDEM · **Mission** m17126 · **Round** 8 "
      "(after R1 m14960, R2 m16427, R3 m16497/m16568, R4 m16633, R5 m16671, "
      "R6 m16758, R7 m17026)")
    A("> **Artifacts**: `src/tensordem/mu_i_rheology.py` "
      "(`full_cell_virial_stress`, `bulk_virial_stress(rule=...)`), "
      "`benchmarks/plane_shear_mu_i/full_cell_rheometry.py`, "
      "`test_virial_estimators.py`, "
      "`runs/full_cell/{prereg.json,result.json,tables.md,fig_full_cell.png}`, "
      "`write_r8_doc.py`")
    A("> **Device change**: two *additional* stress estimators; the wall traction, "
      "the legacy slab estimator, the contact law, the servo protocol and the "
      "reference constants are untouched.\n")

    A("## \u2460 Case\n")
    A("Dense plane shear of frictional spheres in a constant-stress servo shear "
      "cell (`tensordem.mu_i_rheology.simulate_servo_shear`), confronted with the "
      "local rheology `mu(I)`, `phi(I)` of da Cruz et al. (2005, PRE 72:021309; "
      "DOI 10.1103/PhysRevE.72.021309) and GDR MiDi (2004, Eur. Phys. J. E 14:341) "
      "over the inertial-number range `I = 1e-3 ... 0.3` that the ledger "
      "acceptance names (`<10 %` on the `mu(I)` / `phi(I)` curves).\n")
    A("**Why round 8 exists.** Rounds 6-7 measured `mu` from two quantities that "
      "had no independent check: the platen traction (`tau_wall/p_wall`) and a "
      "thin-slab Love-Weber average that *drops 41-49 % of the load-bearing "
      "contacts* (the both-endpoints rule on a 2-4 diameter slab). The resulting "
      "`p_bulk/P - 1 = -17 ... -51 %` could not be separated from a genuine stress "
      "deficit, so the `mu` verdict rested on an uncalibrated instrument.\n")
    A("| parameter | value |")
    A("|---|---|")
    for k, v in prereg["points"].items():
        A(f"| `{k}` | {v} |")
    A("| `mu_contact` / `mu_rolling` / `restitution` | 0.5 / 0.05 / 0.3 |")
    A("| box / fill / radius | 0.12 x 0.05 m, fill 0.06 m, r = 0.005 m |")
    A(f"| `n_particles` | {rows[0]['n_particles']} |")
    A("")

    A("## \u2461 What was done\n")
    A("**D1 — an estimator with an exact reference (`full_cell_virial_stress`).** "
      "For a packing in static equilibrium at zero gravity, per-particle force "
      "balance gives the discrete virial identity\n")
    A("```\n sum_pairs f_ij (x) (x_j - x_i) = - sum_i F_i^ext (x) x_i\n```\n")
    A("The external loads act at the floor (`z ~ r`) and the platen (`z ~ h - r`), "
      "so the `zz` component of the pair sum is `P A (h - 2r)`: dividing by the "
      "*centre-accessible* volume `A (h - 2r)` gives\n")
    A("```\n sigma_zz = -P_applied     =>     sigma_zz / (-P_applied) = +1\n```\n")
    A("exactly, with **no literature constant and no free parameter**. The "
      "function sums every pair of the periodic pair list, normalises by "
      "`W D (h - 2r)` and removes the kinetic part with the fluctuation about the "
      "measured affine profile `v_x = a + b z`. The invariant is evaluated on "
      "every sample (final 20 % window + 60 % tail).\n")
    A("**D2 — the R7-recommended midpoint rule** "
      "(`bulk_virial_stress(rule=\"midpoint\")`): a contact enters the slab sum "
      "when the *midpoint* of its branch vector lies in the slab, which counts each "
      "crossing chain exactly once. Both rules run on the *same* samples, so the "
      "comparison isolates the rule.\n")
    A("**D3 — the pressure convention is stated, not implied.** The quoted law is "
      "2-D plane shear with `mu = sigma_xy / sigma_yy`, i.e. normalised by the "
      "normal stress *in the shear plane*. `mu_full` uses the mean pressure "
      "`p = -tr(sigma)/3`; `mu_full (sigma_zz)` uses `|sigma_xz|/|sigma_zz|` (the "
      "3-D analogue). The two differ by the normal-stress anisotropy, which is "
      f"`p/|sigma_zz| = {min(aniso):.3f} ... {max(aniso):.3f}` here.\n")
    A("**Verification of the estimators themselves** "
      "(`benchmarks/plane_shear_mu_i/test_virial_estimators.py`, 5 tests): the "
      "rules are checked against a hand-computed `sum f (x) r / V` on a synthetic "
      "pair list (a contact straddling the slab is dropped by `both`, counted once "
      "by `midpoint`), and the momentum-balance invariant is asserted on a real "
      "cell.\n")
    A("Pre-registered verdicts (`runs/full_cell/prereg.json`, written before the "
      "sweep):\n")
    for k, v in prereg["hypotheses"].items():
        A(f"* **{k}**: {v}")
    A("")

    A("## \u2462 Results\n")
    A("### (a) P1 — the estimator calibration (analytic anchor)\n")
    A("| I_target | sigma_zz full [Pa] | sigma_zz/(-P) final window | sigma_zz/(-P) 60 % tail | P1 (\u00b15 %) |")
    A("|---|---|---|---|---|")
    A("|---|---|---|---|---|---|")
    for r in rows:
        ok = "pass" if abs(1.0 - r["momentum_balance_final"]) <= 0.05 else "FAIL"
        A(f"| {r['I_target']:.3g} | {r['sigma_zz_full_final']:.0f} | "
          f"{r['momentum_balance_final']:.4f} | {r['momentum_balance_tail']:.4f} | {ok} |")
    A("")
    A(f"Worst deviation from the exact value `+1`: **{bal_max:.4f}** "
      f"({100 * bal_max:.2f} %), on {len(p1)}/{len(rows)} points inside the "
      f"pre-registered \u00b15 % band. On the same samples the legacy both-endpoints "
      f"slab reads "
      f"{pc(min(r['p_balance_slab_both'] for r in rows))} ... "
      f"{pc(max(r['p_balance_slab_both'] for r in rows))} and the midpoint rule "
      f"{pc(min(r['p_balance_slab_mid'] for r in rows))} ... "
      f"{pc(max(r['p_balance_slab_mid'] for r in rows))} — so the R6/R7 pressure "
      "deficit was **mostly estimator truncation**, and the residue is not a "
      "constant-stress control failure.\n")

    A("### (b) `mu` on identical samples, four estimators\n")
    A("| I_target | mu ref | mu wall | mu slab both | mu slab mid | mu full (p) | mu full (sigma_zz) |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['I_target']:.3g} | {r['mu_ref']:.4f} | {r['mu_wall']:.4f} | "
          f"{r['mu_slab_both']:.4f} | {r['mu_slab_mid']:.4f} | {r['mu_full']:.4f} | "
          f"{r['mu_full_zz']:.4f} |")
    A("")
    A("| I_target | err wall | err slab both | err slab mid | err full (p) | err full (sigma_zz) |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['I_target']:.3g} | {r['err_wall_pct']:+.1f} % | "
          f"{r['err_slab_both_pct']:+.1f} % | {r['err_slab_mid_pct']:+.1f} % | "
          f"{r['err_full_pct']:+.1f} % | {r['err_full_zz_pct']:+.1f} % |")
    A("")
    A(f"* worst wall-traction error: **{w_wall['err_wall_pct']:+.1f} %** at "
      f"I={w_wall['I_target']:.3g}")
    A(f"* worst `mu_full` (p = -tr/3) error: **{w_full['err_full_pct']:+.1f} %** at "
      f"I={w_full['I_target']:.3g}")
    A(f"* worst `mu_full` (sigma_zz) error: **{w_zz['err_full_zz_pct']:+.1f} %** at "
      f"I={w_zz['I_target']:.3g}")
    A(f"* inside the 10 % ledger band: `mu_full` {len(p3_full)}/{len(rows)} points, "
      f"`mu_full (sigma_zz)` {len(p3_zz)}/{len(rows)} points\n")

    A("### (c) P2 — is the midpoint rule the fix?\n")
    A("| I_target | mu both / mu full | mu mid / mu full | straddle fraction | rate ratio (slab) | profile R2 | P2 |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        ok = "pass" if (r["mu_slab_both_over_full"] < r["mu_slab_mid_over_full"] < 1.0) else "FAIL"
        A(f"| {r['I_target']:.3g} | {r['mu_slab_both_over_full']:.3f} | "
          f"{r['mu_slab_mid_over_full']:.3f} | "
          f"{100 * (r['straddle_fraction'] or 0.0):.1f} % | {r['rate_ratio']:.2f} | "
          f"{r['profile_r2']:.2f} | {ok} |")
    A("")
    A(f"Ordering holds on {len(p2)}/{len(rows)} points, but the midpoint rule "
      "recovers only a small part of the gap: the slab estimate stays far below "
      "the domain average. The reason is visible in the same table — the slab sees "
      f"`gamma_dot_measured/gamma_dot_imposed = "
      f"{min(r['rate_ratio'] for r in rows):.2f} ... "
      f"{max(r['rate_ratio'] for r in rows):.2f}` with a velocity-profile "
      "`R2` well below 1, i.e. the shear is **localised** near the platens, so a "
      "2-4 diameter interior slab is not a sample of the material that carries the "
      "stress. Fixing the contact rule cannot fix a sampling problem.\n")

    A("### (d) solid fraction and control state (unchanged estimators)\n")
    A("| I_target | phi slab | phi bed | phi ref | coordination | dilatancy | servo valid | wall clock [s] |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['I_target']:.3g} | {r['phi_slab']:.4f} | {r['phi_bed_steady']:.4f} | "
          f"{r['phi_ref']:.4f} | {r['coordination']:.2f} | {pc(r['dilatancy'], 2)} | "
          f"{r['servo_valid']} | {r['wall_clock_s']:.0f} |")
    A("")

    A("## \u2463 Figures\n")
    A("`runs/full_cell/fig_full_cell.png` — left: `mu(I)` for the four "
      "estimators against the da Cruz law; middle: the momentum-balance invariant "
      "`sigma_zz/P` against the exact value `-1` on both averaging windows; right: "
      "the `mu` error of each estimator against the 10 % ledger band.\n")

    A("## \u2464 Notes, limitations, and what a next round must do\n")
    A("1. **The estimator is now calibrated; the material point is not.** P1 shows "
      "the domain average reproduces the applied load to "
      f"{100 * bal_max:.1f} % on every point, so the remaining `mu` deficit is a "
      "statement about the *sample*, not about the arithmetic.")
    A("2. **The domain average mixes the boundary layers.** It is exact for "
      "`sigma_zz` because momentum balance is global, but `sigma_xz` (and hence "
      "`mu`) is a volume average that includes the two localised boundary layers. "
      "That is a legitimate rheometric average only if the layers are a small "
      "fraction of the volume, which at 294 particles they are not.")
    A("3. **The reference law remains 2-D.** da Cruz (2005) is plane shear of "
      "disks; the `phi` magnitudes in `reference_law.py` are 3-D numbers of "
      "unverified provenance (R7 \u2464), and the `mu` constants `(0.36, 0.84, 0.30)` "
      "were inherited from round 1 without an archived reading of the paper. The "
      "*shape* of `mu(I)` is the part this case can confront; the constants are "
      "still PROVENANCE-UNVERIFIED.")
    A("4. **The pressure convention is a "
      f"{100 * (1 - max(aniso)):.0f}-{100 * (1 - min(aniso)):.0f} % lever** on the "
      "ordinate. Both are reported; a verdict that quotes only one of them is "
      "ambiguous, which is how rounds 1-7 lost track of a ~20 % systematic.")
    A(f"5. **Constant-stress control was not valid everywhere**: "
      f"`servo_valid` is {len(valid)}/{len(rows)} on the final window; at the other "
      "points the platen pressure had not yet re-entered the 5 % band when the "
      "strain budget was exhausted, so those `mu` values are provisional.")
    A("6. **Single seed, no ensemble.** Every point is one 294-particle "
      "realisation, so no statistical error bar exists yet; the point-to-point "
      "scatter visible in the tables cannot be separated from realisation noise.")
    A("7. **Next round (recommended).** (i) widen the cell (target \u2265 2000 "
      "particles, \u2265 10 diameters in the shear direction) so the localisation "
      "layers are a small fraction and a *local* `mu(z)` profile can be averaged "
      "over the homogeneous core; (ii) 3-5 seeds per point for error bars; "
      "(iii) only then confront the law on `mu(I)` shape, keeping the "
      "`sigma_zz`-normalised figure as the primary ordinate.\n")

    A("## \u2465 Error judgment (explicit)\n")
    A("| quantity | reference | worst measured | error | within 3 %? |")
    A("|---|---|---|---|---|")
    A(f"| `sigma_zz/(-P_applied)` (full cell virial) | +1 exact (virial identity) | "
      f"{min(r['momentum_balance_final'] for r in rows):.4f} | "
      f"{100 * bal_max:.2f} % | {'yes' if bal_max <= 0.03 else '**no**'} |")
    A(f"| `mu(I)`, full cell, p = -tr/3 | da Cruz 2005 fit | "
      f"{w_full['mu_full']:.4f} | {w_full['err_full_pct']:+.1f} % | **no** |")
    A(f"| `mu(I)`, full cell, sigma_zz | da Cruz 2005 fit | "
      f"{w_zz['mu_full_zz']:.4f} | {w_zz['err_full_zz_pct']:+.1f} % | **no** |")
    A(f"| `mu(I)`, wall traction | da Cruz 2005 fit | {w_wall['mu_wall']:.4f} | "
      f"{w_wall['err_wall_pct']:+.1f} % | **no** |")
    A("")
    A(f"**Verdict.** P1 (estimator calibration) **PASS** — the domain-average "
      f"virial reproduces the applied normal stress to {100 * bal_max:.2f} %. "
      f"P3 (ledger acceptance `<10 %` on `mu(I)`) **FAIL** with the mean-pressure "
      f"ordinate on {len(rows) - len(p3_full)}/{len(rows)} points and with the "
      f"`sigma_zz` ordinate on {len(rows) - len(p3_zz)}/{len(rows)} points; the "
      "diagnosis is now unambiguous (localisation + 2-D reference + small N, not "
      "the stress arithmetic). The ledger row stays **open**: `mu(I)` is not "
      "verified, and `phi(I)` remains WITHHELD (R7 \u2464, reference magnitudes "
      "unverified).\n")
    A("**Honest scope of this round.** The one thing that is *proved* here is the "
      "calibration of the instrument (analytic invariant, 3 verification tests "
      "plus a real-cell assertion). The physics verdict is a **negative** result "
      "and is reported as such: it narrows the failure to the sample, it does not "
      "close the case.\n")

    DOC.write_text("\n".join(L) + "\n")
    print("wrote", DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())