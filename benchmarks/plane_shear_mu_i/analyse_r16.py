"""Round 16 (m20878) -- strain-convergence analysis of the R15 cluster sweep.

Why this round exists
---------------------
Round 15 (m20791) fixed the phi *definition* (container-averaged, not
centre-counted) and launched a strain-convergence sweep -- six points,
``I = 0.001 ... 0.3``, smooth walls, seed 5279, sheared to ``strain >= 2``
(cluster jobs ``cj2950``-``cj2955``) -- to settle whether the low-``I``
deficits recorded in rounds 8/14 were strain transients.  The jobs returned
after the R15 session closed; this round is the analysis R15 explicitly
deferred ("that is the next round's first measurement").

Zero new simulation.  Every number below is read from the six archived
point files in ``runs/strain_sweep_r16/``; the reference law comes from
``reference_law.py`` (single source of truth, provenance status carried
through).

What is measured
----------------

1. **mu(I) at converged strain.**  The archived full-cell points (rounds 8/14)
   reached only ``gamma ~ 0.4`` at low ``I``; these points reached
   ``gamma >= 2`` (``gamma ~ 99`` at ``I = 0.3``).  If the low-``I`` mu
   deficit were a transient, it must shrink here.
2. **phi(I) at converged strain.**  Does the bed stop compacting once the
   strain budget is large enough?
3. **The abscissa.**  ``I_bulk_final`` (measured rate x bulk pressure) is
   reported next to ``I_target`` (imposed platen rate x applied pressure),
   per the round-10 discipline: a volume-averaged ordinate must be judged
   against the measured abscissa, and the two are both shown.
4. **The noise floor.**  The ledger band is 10 %.  For each point the
   standard deviation of ``full_mu`` over the final 25 % of samples is
   computed; if it exceeds the band, the curve is *unresolvable* at this
   particle count regardless of protocol.

Usage::

    python benchmarks/plane_shear_mu_i/analyse_r16.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
RUNS = HERE / "runs" / "strain_sweep_r16"
RUNS.mkdir(parents=True, exist_ok=True)

LEDGER_BAND_PCT = 10.0


def _load_rows() -> list[dict]:
    rows = []
    for f in sorted(RUNS.glob("I*_smooth_*.json")):
        d = json.loads(f.read_text())
        rows.append(d)
    rows.sort(key=lambda r: r["I_target"])
    return rows


def _tail(arr: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(frac * len(arr)))
    return arr[-k:]


def analyse(rows: list[dict]) -> dict:
    from reference_law import ref_mu, ref_phi, reference

    out_rows = []
    for r in rows:
        s = r["series"]
        mu = np.asarray(s["full_mu"], dtype=float)
        phi = np.asarray(s["phi_bed_now"], dtype=float)
        strain = np.asarray(s["strain"], dtype=float)

        mu_tail = _tail(mu, 0.25)
        phi_tail = _tail(phi, 0.25)
        mu_mean = float(mu_tail.mean())
        mu_std = float(mu_tail.std(ddof=1))
        phi_mean = float(phi_tail.mean())
        phi_std = float(phi_tail.std(ddof=1))

        I_t = float(r["I_target"])
        I_b = float(r["I_bulk_final"])
        mu_ref_t = float(ref_mu(I_t))
        mu_ref_b = float(ref_mu(I_b))
        phi_ref_t = float(ref_phi(I_t))

        out_rows.append({
            "I_target": I_t,
            "I_bulk_final": I_b,
            "abscissa_shift_pct": 100.0 * (I_b / I_t - 1.0),
            "strain_achieved": float(r["strain_achieved"]),
            "n_particles": int(r["n_particles"]),
            "n_samples": int(len(mu)),
            "wall_clock_s": float(r["wall_clock_s"]),
            # 20 % tail (the archived validity-gate window)
            "mu_full_20pct": float(r["mu_full_final"]),
            "phi_container_20pct": float(r["phi_container_final"]),
            # 25 % tail means (this round's headline estimates)
            "mu_full_25pct": mu_mean,
            "phi_container_25pct": phi_mean,
            # noise floor
            "mu_std_25pct": mu_std,
            "mu_std_rel_pct": 100.0 * mu_std / mu_mean,
            "mu_range_25pct": [float(mu_tail.min()), float(mu_tail.max())],
            "phi_std_25pct": phi_std,
            "phi_range_25pct": [float(phi_tail.min()), float(phi_tail.max())],
            # errors, both abscissas, both windows
            "err_mu_20pct_at_target_pct": 100.0 * (r["mu_full_final"] / mu_ref_t - 1.0),
            "err_mu_20pct_at_bulk_pct": 100.0 * (r["mu_full_final"] / mu_ref_b - 1.0),
            "err_mu_25pct_at_target_pct": 100.0 * (mu_mean / mu_ref_t - 1.0),
            "err_mu_25pct_at_bulk_pct": 100.0 * (mu_mean / mu_ref_b - 1.0),
            "err_phi_20pct_pct": 100.0 * (r["phi_container_final"] / phi_ref_t - 1.0),
            "err_phi_25pct_pct": 100.0 * (phi_mean / phi_ref_t - 1.0),
            "mu_ref_at_target": mu_ref_t,
            "mu_ref_at_bulk": mu_ref_b,
            "phi_ref": phi_ref_t,
            "steady_phi": bool(r["steady_phi"]),
            "steady_mu": bool(r["steady_mu"]),
            "drift_mu_pct": float(r["drift_rel_pct"]["full_mu"]),
            "drift_phi_pct": float(r["drift_rel_pct"]["phi_bed_now"]),
            "pressure_control_error_final": float(r["pressure_control_error_final"]),
            "servo_valid": bool(r["servo_valid"]),
        })

    mu_pass_20 = all(abs(x["err_mu_20pct_at_target_pct"]) < LEDGER_BAND_PCT
                     for x in out_rows)
    mu_pass_25 = all(abs(x["err_mu_25pct_at_target_pct"]) < LEDGER_BAND_PCT
                     for x in out_rows)
    phi_pass = all(abs(x["err_phi_25pct_pct"]) < LEDGER_BAND_PCT
                   for x in out_rows)
    noise_exceeds_band = all(x["mu_std_rel_pct"] > LEDGER_BAND_PCT
                             for x in out_rows)
    verdict = {
        "mu_curve_within_10pct": bool(mu_pass_20 and mu_pass_25),
        "phi_curve_within_10pct": bool(phi_pass),
        "mu_noise_exceeds_band_everywhere": bool(noise_exceeds_band),
        "overall": "PASS" if (mu_pass_20 and mu_pass_25 and phi_pass)
                   else "FAIL",
    }
    return {"rows": out_rows, "verdict": verdict,
            "reference": reference(),
            "ledger_band_pct": LEDGER_BAND_PCT}


def _md_table(res: dict) -> str:
    L = ["# Round 16 -- strain-converged mu(I)/phi(I) vs the JOP law",
         "",
         "Source: cluster jobs cj2950-cj2955 (R15 submission), smooth walls,",
         "seed 5279, N = 294, sheared to strain >= 2 (gamma ~ 99 at I = 0.3).",
         "Errors are vs `reference_law.py` (provenance status in result.json).",
         "",
         "| I_target | I_bulk | dI/I [%] | gamma | mu(20%) | mu(25%) | mu_ref(t) | mu_ref(b) "
         "| err mu 20%@t [%] | err mu 25%@t [%] | err mu 25%@b [%] "
         "| phi(25%) | phi_ref | err phi [%] | mu std(25%) [%] | steady phi/mu |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in res["rows"]:
        L.append(
            f"| {x['I_target']:g} | {x['I_bulk_final']:.5f} "
            f"| {x['abscissa_shift_pct']:+.0f} | {x['strain_achieved']:.2f} "
            f"| {x['mu_full_20pct']:.4f} | {x['mu_full_25pct']:.4f} "
            f"| {x['mu_ref_at_target']:.4f} | {x['mu_ref_at_bulk']:.4f} "
            f"| {x['err_mu_20pct_at_target_pct']:+.1f} "
            f"| {x['err_mu_25pct_at_target_pct']:+.1f} "
            f"| {x['err_mu_25pct_at_bulk_pct']:+.1f} "
            f"| {x['phi_container_25pct']:.4f} | {x['phi_ref']:.4f} "
            f"| {x['err_phi_25pct_pct']:+.1f} "
            f"| {x['mu_std_rel_pct']:.1f} "
            f"| {'Y' if x['steady_phi'] else 'N'}/"
            f"{'Y' if x['steady_mu'] else 'N'} |")
    v = res["verdict"]
    L += ["",
          "## Verdict",
          "",
          f"- mu(I) within {LEDGER_BAND_PCT:.0f} % (20 % window, I_target): "
          f"**{'PASS' if v['mu_curve_within_10pct'] else 'FAIL'}**",
          f"- mu(I) within {LEDGER_BAND_PCT:.0f} % (25 % window, I_target): "
          f"**{'PASS' if v['mu_curve_within_10pct'] else 'FAIL'}**",
          f"- phi(I) within {LEDGER_BAND_PCT:.0f} % (25 % window): "
          f"**{'PASS' if v['phi_curve_within_10pct'] else 'FAIL'}**",
          f"- mu noise (std of final 25 % window) exceeds the 10 % band at "
          f"every point: **{'yes' if v['mu_noise_exceeds_band_everywhere'] else 'no'}**",
          f"- **Overall: {v['overall']}**",
          ""]
    return "\n".join(L)


def figures(rows: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reference_law import ref_mu, ref_phi

    Is = np.array([x["I_target"] for x in rows])
    Ib = np.array([x["I_bulk_final"] for x in rows])
    grid = np.logspace(math.log10(max(Is.min() / 3, 1e-4)),
                       math.log10(Is.max() * 3), 80)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].plot(grid, ref_mu(grid), "k-", lw=1.6,
               label="JOP law (0.36, 0.84, 0.30)")
    ax[0].plot(Is, [x["mu_full_20pct"] for x in rows], "o-",
               label="this work, 20 % tail (I_target)")
    ax[0].plot(Ib, [x["mu_full_25pct"] for x in rows], "s--",
               label="25 % tail (I_bulk)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("inertial number I")
    ax[0].set_ylabel(r"$\mu = \tau/p$")
    ax[0].legend(fontsize=8)
    ax[0].set_title(r"$\mu(I)$, strain-converged (R16)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(grid, ref_phi(grid), "k-", lw=1.6,
               label="reference band (provenance-unverified)")
    ax[1].plot(Is, [x["phi_container_25pct"] for x in rows], "s-",
                label="container-averaged $\\phi$ (25 % tail)")
    ax[1].axhline(0.7405, color="grey", lw=0.8, ls="-.", label="HCP limit")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("inertial number I")
    ax[1].set_ylabel(r"$\phi$ (container-averaged)")
    ax[1].legend(fontsize=8)
    ax[1].set_title(r"$\phi(I)$, strain-converged (R16)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p1 = RUNS / "fig_mu_i_strain_r16.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    # panel 2: mu vs strain for the two ends of the range
    fig2, ax2 = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for x, axx in zip((rows[0], rows[-1]), ax2):
        s = None
        for f in sorted(RUNS.glob("I*_smooth_*.json")):
            d = json.loads(f.read_text())
            if abs(d["I_target"] - x["I_target"]) < 1e-12:
                s = d["series"]
                break
        st = np.asarray(s["strain"])
        m = np.asarray(s["full_mu"], dtype=float)
        axx.plot(st, m, lw=0.5, alpha=0.5)
        w = max(1, int(0.1 * len(m)))
        kern = np.ones(w) / w
        ma = np.convolve(m, kern, mode="same")
        axx.plot(st, ma, "r-", lw=1.6, label="10 % moving mean")
        axx.set_xscale("log")
        axx.set_xlabel("strain $\\gamma$")
        axx.set_ylabel(r"$\mu$ (full-cell virial)")
        axx.set_title(f"I = {x['I_target']:g}")
        axx.legend(fontsize=8)
        axx.grid(alpha=0.3)
    fig2.tight_layout()
    p2 = RUNS / "fig_mu_vs_strain_r16.png"
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)
    return [str(p1), str(p2)]


def main() -> None:
    rows = _load_rows()
    if not rows:
        raise SystemExit(f"no point files in {RUNS}")
    res = analyse(rows)
    (RUNS / "result.json").write_text(json.dumps(res, indent=2))
    (RUNS / "tables.md").write_text(_md_table(res))
    figs = figures(res["rows"])
    print(_md_table(res))
    print("figures:", *figs, sep="\n  ")


if __name__ == "__main__":
    main()
