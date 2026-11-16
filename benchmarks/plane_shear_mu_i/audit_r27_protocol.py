"""Round 27 (m22662) — protocol audit: is the judged ``mu`` a steady-state value?

What this file is for
---------------------
Rounds 24-26 judged the ledger ``CASE-DEM-PLANE-SHEAR-MUIDI`` on a single
ordinate, ``mu_wall = tau_wall / p_wall``, averaged over a **fixed** window of
``strain_measure = 1.0`` after ``0.5`` of transient.  Round 25's own artifacts
show the ordinate is *not* stationary inside that window (``I = 0.05``:
``mu_wall_first_half = 0.1215``, ``mu_wall_second_half = 0.0792``), and round 25
reported the resulting z-statistic as a stationarity test.

Two things are missing, and this audit supplies them without re-running the
device (it reads the archived time series):

1. **An unbiased uncertainty.**  Successive samples are not independent: they are
   consecutive 40-step blocks of the same sheared state.  ``sd/sqrt(n)`` (what
   the reported z used) therefore overstates the precision by
   ``sqrt(1 + 2*tau_int)``.  This audit estimates the integrated autocorrelation
   time of every series and reports ``n_eff``, the honest error bar, and hence
   whether the reported stationarity failures are significant at all.
2. **A strain-resolved view, and a saturation estimate.**  Instead of asking
   "mean over [0.5, 1.5] strain", it asks "where is the series going": quartile
   means in gamma, an OLS slope with its (autocorrelation-corrected) error, and
   an exponential saturation fit ``mu(g) = mu_inf + (mu_0 - mu_inf) e^{-(g-g0)/lam}``.
   If the window closes while the fit still has a long way to go, the honest
   statement is "``mu_inf`` is not measured", and the size of the resulting bias
   is quantified rather than guessed.

It also tabulates two ordinate ambiguities that the acceptance tolerance
(10 %) is smaller than:

3. **Which pressure?**  ``mu_wall`` uses the wall normal stress, ``mu_bulk`` the
   interior virial, and ``I_bulk`` is formed with the interior pressure while the
   wall pressure is what the servo holds at ``P``.  The four (tau, p) pairings and
   the three I definitions are all printed with their spread.
4. **Stress uniformity.**  With negligible particle inertia the normal stress
   must be uniform across the gap; ``p_bulk/p_wall`` and ``sigma_yy/P`` say how
   uniform it actually is in this device at this softness.

Usage
-----
    python benchmarks/plane_shear_mu_i/audit_r27_protocol.py
writes ``outputs/r27/protocol_audit.json``, ``outputs/r27/protocol_audit.md`` and
``docs/figures/r27_protocol.png``.  Nothing is hard-coded: every number in the
markdown comes from the JSON, which comes from the archived runs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks" / "plane_shear_mu_i"
RUNS = BENCH / "runs" / "plane_shear_2d"
OUT = REPO / "outputs" / "r27"
FIG = REPO / "docs" / "figures"
sys.path.insert(0, str(BENCH))

import reference_law as rl            # noqa: E402
import reference_provenance as rp     # noqa: E402

#: Series examined.  ``kind`` is used only for grouping in the report: the audit
#: itself is identical for every run.
SERIES = {
    "probe_I0.02": "monodisperse probe (round 25)",
    "probe_I0.05": "monodisperse probe (round 25)",
    "probe_I0.1": "monodisperse probe (round 25)",
    "probe_I0.2": "monodisperse probe (round 25)",
    "structure_random_I0.05": "disordered seed (round 25)",
    "structure_random_I0.2": "disordered seed (round 25)",
}


# --------------------------------------------------------------------------- #
# Statistics: honest error bars for correlated samples
# --------------------------------------------------------------------------- #
def integrated_autocorr(x: np.ndarray) -> float:
    """Integrated autocorrelation time ``tau = 1 + 2*sum_k rho_k`` (in samples).

    ``tau = 1`` for independent samples, and ``n_eff = n / tau``.  Sokal's
    automatic window: accumulate until ``W >= c*tau(W)`` with ``c = 5``, which
    keeps the truncation bias below the statistical error of the sum.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return float("nan")
    d = x - x.mean()
    var = float(np.dot(d, d) / n)
    if var <= 0.0:
        return 0.0
    # FFT autocovariance (unbiased normalisation done at the end)
    nfft = 1 << int(2 * n - 1).bit_length()
    f = np.fft.rfft(d, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n]
    acov /= (n - np.arange(n))
    rho = acov / var
    tau = 1.0
    for w in range(1, n // 2):
        tau += 2.0 * rho[w]
        if w >= 5.0 * tau:
            break
    return float(max(1.0, tau))


def mean_se(x: np.ndarray) -> dict:
    """Mean with autocorrelation-corrected standard error."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return {"mean": float("nan"), "sd": float("nan"), "n": int(n),
                "tau_int": float("nan"), "n_eff": float("nan"),
                "se": float("nan"), "se_naive": float("nan"),
                "inflation": float("nan")}
    sd = float(x.std(ddof=1))
    tau = integrated_autocorr(x)
    n_eff = n / tau
    se_naive = sd / math.sqrt(n)
    se = sd / math.sqrt(max(n_eff, 1.0))
    return {"mean": float(x.mean()), "sd": sd, "n": int(n), "tau_int": tau,
            "n_eff": float(n_eff), "se": se, "se_naive": se_naive,
            "inflation": se / se_naive if se_naive > 0 else float("nan")}


def saturation_fit(gamma: np.ndarray, mu: np.ndarray) -> dict:
    """Fit ``mu = mu_inf + (mu_0 - mu_inf) exp(-(g - g0)/lam)``.

    ``mu_inf`` is the saturation level the window is heading for, ``lam`` the
    decay length in strain.  The fit is only *reported as a measurement* when
    the window covers at least 3 decay lengths (``decades_covered >= 3``);
    otherwise the result is an extrapolation and is flagged as such.
    """
    from scipy.optimize import curve_fit

    g = np.asarray(gamma, dtype=float)
    m = np.asarray(mu, dtype=float)
    ok = np.isfinite(g) & np.isfinite(m)
    g, m = g[ok], m[ok]
    if g.size < 20:
        return {"status": "not_run", "reason": "fewer than 20 samples"}
    g0 = float(g.min())

    def model(x, mu_inf, amp, lam):
        return mu_inf + amp * np.exp(-(x - g0) / lam)

    p0 = [float(m[-max(1, m.size // 10):].mean()),
          float(m[:max(1, m.size // 10)].mean() - m[-max(1, m.size // 10):].mean()),
          max(0.2, float(g.max() - g.min()) / 2.0)]
    try:
        popt, pcov = curve_fit(model, g, m, p0=p0, maxfev=40000,
                               bounds=([-1.0, -3.0, 0.02], [3.0, 3.0, 50.0]))
    except Exception as exc:                      # pragma: no cover - numeric
        return {"status": "fit_failed", "reason": repr(exc)}
    mu_inf, amp, lam = (float(v) for v in popt)
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [float("nan")] * 3
    covered = float((g.max() - g0) / lam) if lam > 0 else float("inf")
    # A fit pinned to a bound is not a measurement of the asymptote: inside a
    # window shorter than its own relaxation time the data cannot tell "already
    # saturated" from "still 20 strains away from it".  Say so instead of
    # publishing the bound value as mu_inf.
    at_bound = [nm for nm, v, lo, hi in
                (("mu_inf", mu_inf, -1.0, 3.0), ("amplitude", amp, -3.0, 3.0),
                 ("lam", lam, 0.02, 50.0))
                if abs(v - lo) < 1e-6 or abs(v - hi) < 1e-6]
    status = ("unidentifiable" if at_bound
              else "measured" if covered >= 3.0 else "extrapolated")
    return {
        "status": status,
        "at_bound": at_bound,
        "mu_inf": mu_inf, "mu_inf_se": float(perr[0]),
        "amplitude": amp, "amplitude_se": float(perr[1]),
        "decay_length_strain": lam, "decay_length_se": float(perr[2]),
        "decay_lengths_covered": covered,
        "window_mean": float(m.mean()),
        "bias_of_window_mean": (float(m.mean()) / mu_inf - 1.0)
                               if (status != "unidentifiable" and mu_inf) else float("nan"),
        "gamma_span": [g0, float(g.max())],
    }


def quartile_means(gamma: np.ndarray, mu: np.ndarray, n: int = 4) -> list[dict]:
    g = np.asarray(gamma, dtype=float)
    m = np.asarray(mu, dtype=float)
    ok = np.isfinite(g) & np.isfinite(m)
    g, m = g[ok], m[ok]
    if g.size < n * 4:
        return []
    edges = np.linspace(g.min(), g.max(), n + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (g >= lo) & (g < hi) if hi < edges[-1] else (g >= lo) & (g <= hi)
        if sel.sum() < 4:
            continue
        st = mean_se(m[sel])
        out.append({"gamma": [float(lo), float(hi)], "mu": st["mean"],
                    "se": st["se"], "n": st["n"]})
    return out


def ols_slope(x: np.ndarray, y: np.ndarray) -> dict:
    """Descriptive ``dy/dx`` from a least-squares line.

    **The slope's nominal error is not reported as a significance.**  An OLS
    trend on a serially correlated (or integrated) series is a textbook
    spurious regression: the residual autocorrelation inflates the *coefficient*
    variance far beyond the white-noise formula, and no post-hoc inflation
    repairs it.  The slope is kept because its *sign and magnitude* describe the
    window; the significance decision is :func:`half_split`, which compares two
    block means whose uncertainty is estimated with :func:`mean_se`.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 8:
        return {"slope": float("nan"), "n": int(n)}
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"slope": float(coef[0]), "n": int(n),
            "note": "descriptive; significance from half_split"}


def half_split(gamma: np.ndarray, mu: np.ndarray) -> dict:
    """Two-block stationarity test with autocorrelation-corrected errors.

    Splits the window in half and asks whether the two half-means differ by more
    than their (correlation-corrected) uncertainties allow.  Unlike an OLS t on
    the raw series, this test has the right *scale*: if the series is a random
    walk, the low-frequency variance that makes the halves differ also makes
    each half's error bar large, so the statistic does not fire spuriously.
    """
    g = np.asarray(gamma, float)
    m = np.asarray(mu, float)
    ok = np.isfinite(g) & np.isfinite(m)
    g, m = g[ok], m[ok]
    n = g.size
    if n < 16:
        return {"status": "not_run", "n": int(n)}
    mid = g.min() + 0.5 * (g.max() - g.min())
    a = mean_se(m[g < mid])
    b = mean_se(m[g >= mid])
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    return {"status": "measured", "first_half": a["mean"], "second_half": b["mean"],
            "delta": b["mean"] - a["mean"], "delta_se": se,
            "z": (b["mean"] - a["mean"]) / se if se > 0 else float("nan"),
            "n_first": a["n"], "n_second": b["n"],
            "tau_first": a["tau_int"], "tau_second": b["tau_int"]}


# --------------------------------------------------------------------------- #
# Per-run audit
# --------------------------------------------------------------------------- #
def audit_run(name: str, path: Path) -> dict:
    d = json.loads(path.read_text())
    samples = d.get("samples") or []
    win = [s for s in samples if s.get("phase") == "measure"]
    if len(win) < 10:
        return {"status": "not_run", "reason": "no measurement window in artifact"}

    def col(k):
        return np.array([s.get(k, float("nan")) for s in win], float)

    g = col("strain")
    tw, pw = col("tau_wall"), col("p_wall")
    tb, pb = col("tau_bulk"), col("p_bulk")
    mu_wall = tw / pw
    rows = {
        "status": "measured",
        "artifact": str(path.relative_to(REPO)),
        "kind": SERIES.get(name, "unclassified"),
        "I_target": d.get("I_target"),
        "polydispersity": d.get("polydispersity"),
        "n_samples": int(len(win)),
        "gamma_window": [float(np.nanmin(g)), float(np.nanmax(g))],
        "psi6_mean": d.get("psi6_mean"),
        "phi_gap": mean_se(col("phi_gap")),
        "mu_wall": mean_se(mu_wall),
        "mu_bulk": mean_se(col("mu_bulk")),
        "I_bulk": mean_se(col("I_bulk")),
        "tau_wall": mean_se(tw),
        "p_wall": mean_se(pw),
        "p_bulk_over_p_wall": mean_se(pb / pw),
        "sigma_yy_over_P": mean_se(col("sigma_yy_over_P")),
        "slope_mu_wall_per_strain": ols_slope(g, mu_wall),
        "stationarity_mu_wall": half_split(g, mu_wall),
        "stationarity_phi_gap": half_split(g, col("phi_gap")),
        "saturation": saturation_fit(g, mu_wall),
        "quartiles": quartile_means(g, mu_wall),
    }
    # ordinate ambiguity: which (tau, p) pairing, and which I definition
    variants = {}
    for tname, tau in (("tau_wall", tw), ("tau_bulk", tb)):
        for pname, p in (("p_wall", pw), ("p_bulk", pb)):
            variants[f"{tname}/{pname}"] = mean_se(tau / p)
    rows["mu_variants"] = {k: {"mean": v["mean"], "se": v["se"], "n_eff": v["n_eff"]}
                           for k, v in variants.items()}
    vals = [v["mean"] for v in variants.values() if math.isfinite(v["mean"])]
    rows["mu_variant_spread_rel"] = ((max(vals) - min(vals)) / min(vals)
                                    if len(vals) > 1 and min(vals) > 0 else float("nan"))
    # I: stored value uses the interior pressure; rebuild it from the wall pressure
    gamma_rate = 2.0 * float(d.get("case", {}).get("normal_stress", 0.0)) ** 0 * 0.0
    I_stored = rows["I_bulk"]["mean"]
    p_ratio = rows["p_bulk_over_p_wall"]["mean"]
    rows["I_definitions"] = {
        "as_stored_(interior_virial_P)": I_stored,
        "wall_pressure_P": I_stored / math.sqrt(p_ratio) if p_ratio and p_ratio > 0 else float("nan"),
        "nominal_target": d.get("I_target"),
    }
    # ledger verdict, on the judged ordinate and on the saturation estimate
    Ij = rows["I_bulk"]["mean"]
    sat = rows["saturation"]
    mu_judged = rows["mu_wall"]["mean"]
    rows["ledger_verdict_judged"] = rp.verdict_table([Ij], [mu_judged])
    rows["ledger_best_case_judged"] = rp.best_case_over_envelope([Ij], [mu_judged])
    if isinstance(sat, dict) and sat.get("status") in ("measured", "extrapolated") \
            and math.isfinite(sat.get("mu_inf", float("nan"))):
        mu_sat = sat["mu_inf"]
        rows["ledger_verdict_saturated"] = rp.verdict_table([Ij], [mu_sat])
        rows["ledger_best_case_saturated"] = rp.best_case_over_envelope([Ij], [mu_sat])
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    found = {k: RUNS / f"{k}.json" for k in SERIES if (RUNS / f"{k}.json").exists()}
    # round-27 saturation runs, when they have drained
    for pat, kind in (("r27sat_*.json", "strain-saturation (round 27)"),
                      ("poly_w*_I*.json", "polydispersity sweep cj4227 (round 26)")):
        for p in sorted(RUNS.glob(pat)):
            found[p.stem] = p
            SERIES.setdefault(p.stem, kind)
    audit = {
        "what": "protocol/stationarity audit of the plane-shear mu ordinate",
        "reference_law_used_for_verdicts": "reference_provenance envelope",
        "runs": {},
        "notes": {
            "tau_int": "integrated autocorrelation time in sample steps (40 steps/sample)",
            "se": "sd/sqrt(n_eff), n_eff = n/(1+2 tau_int)",
            "saturation": "mu_inf of an exponential fit; 'measured' only if >= 3 decay lengths",
        },
    }
    for name, path in found.items():
        try:
            audit["runs"][name] = audit_run(name, path)
        except Exception as exc:                     # pragma: no cover - defensive
            audit["runs"][name] = {"status": "error", "reason": repr(exc)}
    (OUT / "protocol_audit.json").write_text(json.dumps(audit, indent=1, default=float))
    (OUT / "protocol_audit.md").write_text(render_markdown(audit))
    make_figure(audit)
    print(f"wrote {OUT/'protocol_audit.json'}")
    print(f"wrote {OUT/'protocol_audit.md'}")
    print(f"wrote {FIG/'r27_protocol.png'}")
    return 0


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_markdown(audit: dict) -> str:
    L = ["# Round 27 — protocol / stationarity audit of the plane-shear `mu` ordinate",
         "",
         "Generated by `benchmarks/plane_shear_mu_i/audit_r27_protocol.py` from archived "
         "run artifacts; no number here is typed by hand.", ""]
    L += ["## 1. Is the judged window a steady state?", "",
          "| run | I (stored) | n | tau_int | n_eff | mu_wall ± se(corr) | se naive | inflation |",
          "|---|---|---|---|---|---|---|---|"]
    for name, r in audit["runs"].items():
        if r.get("status") != "measured":
            L.append(f"| {name} | — | — | — | — | {r.get('status')} | — | — |")
            continue
        m, s = r["mu_wall"], r["slope_mu_wall_per_strain"]
        L.append(f"| `{name}` | {r['I_bulk']['mean']:.4f} | {m['n']} | {m['tau_int']:.1f} | "
                 f"{m['n_eff']:.1f} | {m['mean']:.4f} ± {m['se']:.4f} | "
                 f"{m['se_naive']:.4f} | {m['inflation']:.1f}x |")
    L += ["", "## 2. Where is the series going?", "",
          "| run | dmu/dgamma (descriptive) | half-split z (corrected) | decay length | window mean vs mu_inf | fit status |",
          "|---|---|---|---|---|---|"]
    for name, r in audit["runs"].items():
        if r.get("status") != "measured":
            continue
        s = r["slope_mu_wall_per_strain"]
        sat = r["saturation"]
        st = r.get("stationarity_mu_wall", {})
        sig = (f"z={st['z']:+.1f}" if st.get("status") == "measured" else "—")
        if sat.get("status") in ("measured", "extrapolated"):
            L.append(f"| `{name}` | {s['slope']:+.4f} | {sig} | {sat['decay_length_strain']:.2f} "
                     f"({sat['decay_lengths_covered']:.1f} covered) | "
                     f"{100*sat['bias_of_window_mean']:+.1f} % | {sat['status']} |")
        else:
            L.append(f"| `{name}` | {s.get('slope', float('nan')):+.4f} | {sig} | — | — | "
                     f"{sat.get('status')} |")
    L += ["", "## 3. Ordinate ambiguity — which pressure?", "",
          "| run | tau_w/p_w | tau_w/p_b | tau_b/p_w | tau_b/p_b | spread | p_bulk/p_wall | sigma_yy/P |",
          "|---|---|---|---|---|---|---|---|"]
    for name, r in audit["runs"].items():
        if r.get("status") != "measured":
            continue
        v = r["mu_variants"]
        L.append("| `%s` | %s | %s | %s | %s | %+.0f %% | %.3f | %.3f |" % (
            name, v["tau_wall/p_wall"]["mean"], v["tau_wall/p_bulk"]["mean"],
            v["tau_bulk/p_wall"]["mean"], v["tau_bulk/p_bulk"]["mean"],
            100 * r["mu_variant_spread_rel"], r["p_bulk_over_p_wall"]["mean"],
            r["sigma_yy_over_P"]["mean"]))
    L += ["", "## 4. Ledger verdict, judged ordinate vs saturation estimate", "",
          "| run | I | mu judged | best-case envelope err (judged) | mu_inf | "
          "best-case envelope err (saturated) | verdict flips? |",
          "|---|---|---|---|---|---|---|"]
    for name, r in audit["runs"].items():
        if r.get("status") != "measured":
            continue
        bc = r["ledger_best_case_judged"]
        bcs = r.get("ledger_best_case_saturated")
        mu_sat = (r["saturation"].get("mu_inf")
                  if isinstance(r["saturation"], dict) else float("nan"))
        flip = ("yes" if bcs and (bcs["min_max_abs_rel_err"] <= 0.10)
                != (bc["min_max_abs_rel_err"] <= 0.10) else "no")
        L.append("| `%s` | %.4f | %.4f | %+.1f %% | %s | %s | %s |" % (
            name, r["I_bulk"]["mean"], r["mu_wall"]["mean"],
            100 * bc["min_max_abs_rel_err"], f"{mu_sat:.4f}" if math.isfinite(mu_sat) else "—",
            f"{100*bcs['min_max_abs_rel_err']:+.1f} %" if bcs else "—", flip))
    L += ["", "## 5. Quartile drift inside each window", ""]
    for name, r in audit["runs"].items():
        if r.get("status") != "measured" or not r["quartiles"]:
            continue
        cells = " → ".join(f"{q['mu']:.4f}" for q in r["quartiles"])
        L.append(f"* `{name}`: {cells}  (strain quartiles of "
                 f"{r['gamma_window'][0]:.2f}…{r['gamma_window'][1]:.2f})")
    return "\n".join(L) + "\n"


def make_figure(audit: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {k: v for k, v in audit["runs"].items() if v.get("status") == "measured"}
    if not runs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    Igrid = np.logspace(-3, math.log10(0.3), 200)
    for cand, style in ((rp.PRIMARY_REPO, "-"), (rp.TEST_GDR_MIDI, "--")):
        ax[0].plot(Igrid, [cand.mu(I) for I in Igrid], style, color="0.35", lw=1.4,
                   label=f"{cand.name} {cand.triple}")
    for name, r in runs.items():
        I = r["I_bulk"]["mean"]
        ax[0].errorbar([I], [r["mu_wall"]["mean"]], yerr=[r["mu_wall"]["se"]],
                       fmt="o", ms=5, capsize=3, label=name)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("inertial number I")
    ax[0].set_ylabel(r"$\mu$")
    ax[0].set_title("judged ordinate vs reference family\n(error bars: autocorr-corrected)")
    ax[0].legend(fontsize=6, loc="upper left")
    ax[0].grid(alpha=0.3)

    labels, slopes, sigs = [], [], []
    for name, r in runs.items():
        st = r.get("stationarity_mu_wall", {})
        labels.append(name.replace("structure_", "").replace("probe_", ""))
        slopes.append(st.get("delta", float("nan")))
        sigs.append(st.get("delta_se", float("nan")))
    y = np.arange(len(labels))
    ax[1].barh(y, slopes, xerr=sigs, color="#4878a8", capsize=3)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(labels, fontsize=7)
    ax[1].axvline(0.0, color="k", lw=1)
    ax[1].set_xlabel(r"$\mu$ second half $-$ first half (error: autocorr-corrected)")
    ax[1].set_title("is the judged window stationary?\n(error bars include autocorrelation)")
    ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "r27_protocol.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())