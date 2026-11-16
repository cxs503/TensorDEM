#!/usr/bin/env python3
"""Round 29 (m22733) analysis: is the judged ``mu`` a steady state?

Reads the ``long_r29_*`` campaign files (8-strain measurement window, 2-D probe
cell, ``w = 0.5`` polydispersity + the ``w = 0`` crystal control) and produces,
per run:

* the **native-sampling autocorrelation time** ``tau = 1 + 2 sum_k rho_k``
  (Sokal window) of ``mu_wall`` and the honest standard error
  ``sd/sqrt(n_eff)``, ``n_eff = n/tau`` - the 1-strain windows of rounds 24-26
  reported ``sd/sqrt(n)``, which is 2-10x too small;
* **1-strain block means** of ``mu_wall``, ``mu_bulk``, ``phi_gap``, so the
  strain evolution is visible instead of being averaged away;
* a **weighted linear trend test** on the block means (slope +/- error, z) and
  the resulting stationarity decision: the last two strains must agree within
  2 sigma and the trend over the second half must be insignificant;
* an **exponential saturation fit** ``mu(gamma) = mu_inf + a exp(-gamma/gamma0)``
  (``gamma0`` by grid search, ``(mu_inf, a)`` by weighted least squares) as a
  diagnostic of how far the window mean still is from its own asymptote;
* the ledger verdict per point: ``mu`` and ``phi`` against the archived law
  (``reference_law``) and against its full provenance envelope
  (``reference_provenance.best_case_over_envelope``), plus the honest 2-D/3-D
  caveat that the archived law is a 3-D glass-bead fit.

Outputs: ``outputs/r29/r29_summary.json``, ``outputs/r29/r29_tables.md`` and
the figures ``docs/figures/r29_*.png``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "src"))

import reference_law as rl  # noqa: E402
import reference_provenance as rp  # noqa: E402

RUNS = BENCH / "runs" / "plane_shear_2d"
OUT = REPO / "outputs" / "r29"
FIG = REPO / "docs" / "figures"
TOL = 0.10


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def integrated_autocorr_time(x: np.ndarray, max_lag: int | None = None) -> dict:
    """``tau_int = 1 + 2 sum_k rho_k`` with Sokal's automatic window.

    ``n_eff = n / tau_int`` is what the window really contains; a white-noise
    series gives ``tau -> 1`` (``n_eff -> n``) and an AR(1) with coefficient
    ``phi`` gives ``tau -> (1+phi)/(1-phi)``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return {"tau": float("nan"), "n_eff": float("nan"), "n": n,
                "window": 0, "sd": float("nan"), "sem": float("nan")}
    xc = x - x.mean()
    var = float(np.dot(xc, xc) / n)
    if var <= 0.0:
        return {"tau": 1.0, "n_eff": float(n), "n": n, "window": 0,
                "sd": 0.0, "sem": 0.0}
    nmax = n // 2 if max_lag is None else min(max_lag, n // 2)
    rho = np.empty(nmax)
    for k in range(1, nmax + 1):
        rho[k - 1] = float(np.dot(xc[:-k], xc[k:]) / (n * var))
    tau = 1.0
    window = 0
    for k, r in enumerate(rho, start=1):
        tau += 2.0 * r
        window = k
        if k >= 5.0 * tau:          # Sokal: stop once the window exceeds 5 tau
            break
    tau = max(tau, 1.0)
    sd = math.sqrt(var)
    return {"tau": float(tau), "n_eff": float(n / tau), "n": int(n),
            "window": int(window), "sd": float(sd),
            "sem": float(sd / math.sqrt(n / tau))}


def block_means(rows: list[dict], key: str, edges: np.ndarray) -> list[dict]:
    """Mean/sem of ``key`` in each strain block ``[edges[i], edges[i+1])``."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["strain"] < hi and math.isfinite(r.get(key, float("nan")))]
        vals = np.array([r[key] for r in sel], dtype=float)
        if vals.size == 0:
            out.append({"lo": float(lo), "hi": float(hi), "n": 0,
                        "mean": float("nan"), "sem": float("nan")})
            continue
        if vals.size < 8:
            sem = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else float("nan")
        else:
            sem = integrated_autocorr_time(vals)["sem"]
        out.append({"lo": float(lo), "hi": float(hi), "n": int(vals.size),
                    "mean": float(vals.mean()), "sem": float(sem)})
    return out


def weighted_trend(blocks: list[dict]) -> dict:
    """Least-squares slope of block mean vs block index with sem weights."""
    idx, val, sig = [], [], []
    for i, b in enumerate(blocks):
        if b["n"] and math.isfinite(b["mean"]) and math.isfinite(b["sem"]) and b["sem"] > 0:
            idx.append(i)
            val.append(b["mean"])
            sig.append(b["sem"])
    if len(idx) < 3:
        return {"slope": float("nan"), "slope_err": float("nan"),
                "z": float("nan"), "per_strain": float("nan")}
    idx = np.array(idx, float)
    val = np.array(val, float)
    sig = np.array(sig, float)
    w = 1.0 / sig ** 2
    A = np.vstack([idx, np.ones_like(idx)]).T
    cov = np.linalg.inv((A * w[:, None]).T @ A)
    beta = cov @ ((A * w[:, None]).T @ val)
    slope, slope_err = float(beta[0]), float(math.sqrt(cov[0, 0]))
    return {"slope": slope, "slope_err": slope_err,
            "z": slope / slope_err if slope_err > 0 else float("nan"),
            "per_strain": slope}


def saturation_fit(strain: np.ndarray, mu: np.ndarray, sem: np.ndarray) -> dict:
    """``mu = mu_inf + a exp(-gamma/gamma0)``; ``gamma0`` on a log grid."""
    ok = np.isfinite(strain) & np.isfinite(mu)
    if ok.sum() < 20:
        return {"mu_inf": float("nan"), "gamma0": float("nan"), "a": float("nan"),
                "rms": float("nan"), "n": int(ok.sum())}
    s, m = strain[ok], mu[ok]
    w = 1.0 / np.where(ok, np.nan_to_num(sem, nan=1.0), 1.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    best = None
    for gamma0 in np.geomspace(0.05, 50.0, 240):
        basis = np.exp(-s / gamma0)
        A = np.vstack([np.ones_like(s), basis]).T
        Aw = A * w[:, None]
        try:
            beta = np.linalg.lstsq(Aw, m * w, rcond=None)[0]
        except np.linalg.LinAlgError:              # pragma: no cover
            continue
        resid = (m - A @ beta) * w
        sse = float(np.sum(resid ** 2))
        if best is None or sse < best[0]:
            best = (sse, gamma0, beta)
    sse, gamma0, beta = best
    rms = math.sqrt(sse / max(1, s.size))
    return {"mu_inf": float(beta[0]), "a": float(beta[1]),
            "gamma0": float(gamma0), "rms": rms, "n": int(s.size)}


# --------------------------------------------------------------------------- #
# per-run reduction
# --------------------------------------------------------------------------- #
def load_runs(pattern: str = "long_r29_w*.json") -> list[dict]:
    runs = []
    for path in sorted(RUNS.glob(pattern)):
        if "index" in path.name:
            continue
        d = json.loads(path.read_text())
        d["_file"] = path.name
        runs.append(d)
    return runs


def reduce_run(d: dict) -> dict:
    rows = [s for s in d["samples"] if s.get("phase") == "measure"]
    if len(rows) < 16:
        return {"file": d["_file"], "skipped": "too few measure samples"}
    strain = np.array([r["strain"] for r in rows], float)
    gmax = float(np.ceil(strain.max()))
    edges = np.arange(0.0, gmax + 1.0, 1.0)
    key_stats = {k: integrated_autocorr_time(np.array([r[k] for r in rows], float))
                 for k in ("mu_wall", "mu_bulk", "phi_gap")}
    blocks = {k: block_means(rows, k, edges)
              for k in ("mu_wall", "mu_bulk", "phi_gap", "I_bulk")}
    half = len(blocks["mu_wall"]) // 2
    second_half = blocks["mu_wall"][half:]
    trend_all = weighted_trend(blocks["mu_wall"])
    trend_second = weighted_trend(second_half)
    last_two = [b for b in blocks["mu_wall"] if b["n"]][-2:]
    stationary = False
    last_two_consistent = False
    if len(last_two) == 2:
        a, b = last_two
        denom = math.hypot(a["sem"], b["sem"]) or float("nan")
        diff_z = (b["mean"] - a["mean"]) / denom if denom == denom and denom > 0 else float("nan")
        last_two_consistent = bool(abs(diff_z) < 2.0)
        stationary = bool(last_two_consistent and abs(trend_second["z"]) < 2.0)
    else:
        diff_z = float("nan")
    mu_native = np.array([r["mu_wall"] for r in rows], float)
    sem_native = np.full(mu_native.shape, key_stats["mu_wall"]["sem"])
    fit = saturation_fit(strain, mu_native, sem_native)
    w = float(d["argv"]["w"]) if "argv" in d else float(d.get("polydispersity", 0.0))
    return {
        "file": d["_file"],
        "w": w,
        "I_target": float(d["argv"]["I_target"]) if "argv" in d else float(d["I_target"]),
        "seed": int(d["argv"]["seed"]) if "argv" in d else int(d["case"]["seed"]),
        "strain_measure": float(d["argv"]["strain_measure"]) if "argv" in d else float(d["case"]["strain_measure"]),
        "strain_max": float(strain.max()),
        "n_samples": int(strain.size),
        "duration_s": float(d.get("duration_s", float("nan"))),
        "stats": key_stats,
        "blocks": blocks,
        "trend_all": trend_all,
        "trend_second": trend_second,
        "last_block_pair_z": diff_z,
        "last_two_consistent": last_two_consistent,
        "stationary": stationary,
        "saturation": fit,
        "mu_wall_mean": float(np.mean([r["mu_wall"] for r in rows])),
        "mu_wall_last_half": float(np.mean([r["mu_wall"] for r in rows if r["strain"] >= 0.5 * strain.max()])),
        "mu_bulk_last_half": float(np.mean([r["mu_bulk"] for r in rows if r["strain"] >= 0.5 * strain.max()])),
        "phi_gap_last_half": float(np.mean([r["phi_gap"] for r in rows if r["strain"] >= 0.5 * strain.max()])),
        "I_bulk_last_half": float(np.mean([r["I_bulk"] for r in rows if r["strain"] >= 0.5 * strain.max()])),
        "psi6_mean": d.get("psi6_mean"), "z_mean": d.get("z_mean"),
        "p_bulk_over_p_wall": (d.get("p_bulk", float("nan")) / d.get("p_wall", float("nan"))
                               if d.get("p_wall") else float("nan")),
        "series": {"strain": strain.tolist(),
                   "mu_wall": [r["mu_wall"] for r in rows],
                   "mu_bulk": [r["mu_bulk"] for r in rows],
                   "phi_gap": [r["phi_gap"] for r in rows],
                   "I_bulk": [r["I_bulk"] for r in rows]},
    }


def archived_baseline() -> dict:
    """The 1-strain window of round 26, same cell, for the drift comparison."""
    out = {}
    for tag, name in (("w0.5_I0.05", "poly_w0.5_I0.05.json"),
                      ("w0.5_I0.2", "poly_w0.5_I0.2.json"),
                      ("w0.5_I0.02", "poly_w0.5_I0.02.json"),
                      ("w0.0_I0.05", "poly_w0.0_I0.05.json")):
        p = RUNS / name
        if p.exists():
            d = json.loads(p.read_text())
            out[tag] = {"mu_wall": d.get("mu_wall"), "mu_bulk": d.get("mu_bulk"),
                        "phi_gap": d.get("phi_gap"), "I_bulk": d.get("I_bulk"),
                        "mu_wall_sd": d.get("mu_wall_sd"),
                        "drift_z_mu_wall": d.get("drift_z_mu_wall"),
                        "strain_measure": d.get("case", {}).get("strain_measure")}
    return out


# --------------------------------------------------------------------------- #
# verdicts and tables
# --------------------------------------------------------------------------- #
def verdict_point(mu: float, I: float, phi: float | None = None) -> dict:
    ref_mu = float(rl.ref_mu(I))
    err = 100.0 * (mu - ref_mu) / ref_mu
    best = rp.best_case_over_envelope([I], [mu], tol=TOL)
    out = {"I": I, "mu": mu, "mu_ref": ref_mu, "err_pct": err,
           "within_tol": bool(abs(err) <= 100.0 * TOL),
           "envelope_best_case_err_pct": 100.0 * best["min_max_abs_rel_err"],
           "envelope_can_rescue": best["provenance_can_rescue"]}
    if phi is not None:
        ref_phi = float(rl.ref_phi(I))
        out["phi"] = phi
        out["phi_ref"] = ref_phi
        out["phi_err_pct"] = 100.0 * (phi - ref_phi) / ref_phi
        out["phi_within_tol"] = bool(abs(out["phi_err_pct"]) <= 100.0 * TOL)
    return out


def curve_envelope(I_values, mu_values) -> dict:
    """Curve-level falsification oracle: can *any* admissible law fit the series?

    A single point is nearly always inside the wide provenance envelope, so the
    envelope question can only be asked of a curve (``best_case_over_envelope``
    scans ``mu_s in [0.34, 0.40]``, ``mu_2 in [0.55, 0.95]``, ``I_0 in
    [0.22, 0.38]`` and returns the smallest possible worst-point error).
    """
    return rp.best_case_over_envelope(I_values, mu_values, tol=TOL)


def seed_ensemble(reduced: list[dict]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict]] = {}
    for r in reduced:
        if "skipped" in r:
            continue
        groups.setdefault((r["w"], r["I_target"]), []).append(r)
    rows = []
    for (w, I), runs in sorted(groups.items()):
        mus = np.array([r["mu_wall_last_half"] for r in runs])
        phis = np.array([r["phi_gap_last_half"] for r in runs])
        Is = np.array([r["I_bulk_last_half"] for r in runs])
        mu = float(mus.mean())
        sem = float(mus.std(ddof=1) / math.sqrt(mus.size)) if mus.size > 1 else float("nan")
        Ibar = float(Is.mean())
        row = {"w": w, "I_target": I, "n_seeds": int(mus.size),
               "I_measured": Ibar, "mu": mu, "mu_seed_sem": sem,
               "phi": float(phis.mean()),
               "phi_seed_sem": (float(phis.std(ddof=1) / math.sqrt(phis.size))
                                if phis.size > 1 else float("nan")),
               "all_stationary": bool(all(r["stationary"] for r in runs)),
               "any_stationary": bool(any(r["stationary"] for r in runs)),
               "files": [r["file"] for r in runs]}
        row.update(verdict_point(mu, Ibar, row["phi"]))
        rows.append(row)
    return rows


def md_tables(reduced: list[dict], ensemble: list[dict], baseline: dict,
              curves: dict, smoke: bool = False) -> str:
    L: list[str] = []
    L.append("# R29 — long-window stationarity campaign (tables)\n")
    L.append(f"campaign: `benchmarks/plane_shear_mu_i/run_long_window_r29.py`, "
             f"strain window 8 (archived rounds 24-26: 1)\n")
    if smoke:
        L.append("**SMOKE DATA** — not the campaign.\n")
    L.append("\n## A. Per-run stationarity\n")
    L.append("| file | w | I_t | seed | γ_max | mu_wall | sem | tau | n_eff | "
             "trend z (all) | trend z (2nd half) | last-2 z | stationary | mu_inf fit | γ0 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in reduced:
        if "skipped" in r:
            L.append(f"| {r['file']} | – | – | – | – | skipped: {r['skipped']} |" + " |" * 12)
            continue
        s = r["stats"]["mu_wall"]
        f = r["saturation"]
        L.append("| %s | %.2f | %.3f | %d | %.1f | %.4f | %.4f | %.1f | %.0f | "
                 "%+.1f | %+.1f | %+.2f | %s | %.4f | %.2f |"
                 % (r["file"], r["w"], r["I_target"], r["seed"], r["strain_max"],
                    r["mu_wall_last_half"], s["sem"], s["tau"], s["n_eff"],
                    r["trend_all"]["z"], r["trend_second"]["z"], r["last_block_pair_z"],
                    "YES" if r["stationary"] else "no", f["mu_inf"], f["gamma0"]))
    L.append("\n## B. Seed ensemble, last half of the window\n")
    L.append("| w | I_target | seeds | I_measured | mu | seed sem | phi | phi sem | "
             "mu_ref(I) | err % | <10%? | envelope best-case err % | phi_ref | phi err % |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in ensemble:
        L.append("| %.2f | %.3f | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | "
                 "%+.1f | %s | %s | %.4f | %+.1f |"
                 % (e["w"], e["I_target"], e["n_seeds"], e["I_measured"], e["mu"],
                    e["mu_seed_sem"], e["phi"], e["phi_seed_sem"], e["mu_ref"],
                    e["err_pct"], "YES" if e["within_tol"] else "NO",
                    "n/a" if e["envelope_best_case_err_pct"] is None
                    else "%.0f" % e["envelope_best_case_err_pct"],
                    e["phi_ref"], e["phi_err_pct"]))
    L.append("\n## B2. Curve-level falsification oracle (can *any* admissible "
             "reading of the citation fit this curve?)\n")
    if curves:
        L.append("| branch | points | min possible worst-point error % | "
                 "can any admissible law fit? |")
        L.append("|---|---|---|---|")
        for name, c in curves.items():
            L.append("| %s | %d | %.0f | %s |"
                     % (name, len(c["per_point_rel_err"]),
                        100.0 * c["min_max_abs_rel_err"],
                        "yes" if c["provenance_can_rescue"] else "NO"))
    else:
        L.append("_fewer than two I points available - curve oracle not applicable_")
    L.append("\n## C. Drift of the judged mu against the archived 1-strain window\n")
    L.append("| point | archived mu (1 strain) | R29 mu (last half of 8 strains) | change % |")
    L.append("|---|---|---|---|")
    by_key = {(e["w"], round(e["I_target"], 4)): e for e in ensemble}
    for tag, w, I in (("w0.5_I0.02", 0.5, 0.02), ("w0.5_I0.05", 0.5, 0.05),
                      ("w0.5_I0.2", 0.5, 0.2), ("w0.0_I0.05", 0.0, 0.05)):
        if tag not in baseline:
            continue
        b = baseline[tag]
        e = by_key.get((w, I))
        if e is None:
            L.append(f"| {tag} | {b['mu_wall']:.4f} | missing | – |")
            continue
        change = 100.0 * (e["mu"] - b["mu_wall"]) / b["mu_wall"]
        L.append(f"| {tag} | {b['mu_wall']:.4f} | {e['mu']:.4f} | {change:+.1f} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(reduced: list[dict], ensemble: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    written = []
    good = [r for r in reduced if "skipped" not in r]
    if not good:
        return written

    # 1. mu vs strain, one panel per I (w = 0.5), all seeds
    Is = sorted({r["I_target"] for r in good if r["w"] > 0})
    fig, axes = plt.subplots(1, max(1, len(Is)), figsize=(4.6 * max(1, len(Is)), 3.6),
                            sharey=True, squeeze=False)
    for ax, I in zip(axes[0], Is):
        for r in good:
            if r["w"] > 0 and r["I_target"] == I:
                s = np.array(r["series"]["strain"])
                m = np.array(r["series"]["mu_wall"])
                ax.plot(s, m, lw=0.6, alpha=0.45, label=f"seed {r['seed']}")
                blk = r["blocks"]["mu_wall"]
                c = [0.5 * (b["lo"] + b["hi"]) for b in blk if b["n"]]
                v = [b["mean"] for b in blk if b["n"]]
                e = [b["sem"] for b in blk if b["n"]]
                ax.errorbar(c, v, yerr=e, fmt="o-", ms=3, lw=1.2, color="k", alpha=0.85)
        ax.axhline(float(rl.ref_mu(I)), ls="--", c="crimson", lw=1.2,
                   label="da Cruz/GDR MiDi law")
        ax.set_title(f"w = 0.5, I_target = {I:g}")
        ax.set_xlabel("shear strain $\\gamma$")
        ax.set_ylabel("$\\mu_{wall}$")
        ax.legend(fontsize=6)
    fig.suptitle("R29: mu_wall over an 8-strain window (thin: samples, thick: 1-strain blocks)")
    fig.tight_layout()
    p = FIG / "r29_mu_vs_strain.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p.relative_to(REPO)))

    # 2. phi vs strain
    fig, axes = plt.subplots(1, max(1, len(Is)), figsize=(4.6 * max(1, len(Is)), 3.4),
                             sharey=True, squeeze=False)
    for ax, I in zip(axes[0], Is):
        for r in good:
            if r["w"] > 0 and r["I_target"] == I:
                ax.plot(r["series"]["strain"], r["series"]["phi_gap"], lw=0.6, alpha=0.6)
                blk = r["blocks"]["phi_gap"]
                c = [0.5 * (b["lo"] + b["hi"]) for b in blk if b["n"]]
                v = [b["mean"] for b in blk if b["n"]]
                ax.plot(c, v, "ko-", ms=3, lw=1.2)
        ax.set_title(f"w = 0.5, I_target = {I:g}")
        ax.set_xlabel("shear strain $\\gamma$")
        ax.set_ylabel("$\\phi_{gap}$")
    fig.suptitle("R29: solid fraction over the 8-strain window")
    fig.tight_layout()
    p = FIG / "r29_phi_vs_strain.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p.relative_to(REPO)))

    # 3. mu / phi vs I with the reference law and envelope
    if ensemble:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
        Igrid = np.geomspace(1e-3, 0.4, 200)
        ax1.plot(Igrid, rl.ref_mu(Igrid), "r--", label="da Cruz/GDR MiDi law (3-D)")
        lo, hi = [], []
        for I in Igrid:
            b = rp.envelope_bounds(I)
            lo.append(b[0])
            hi.append(b[1])
        ax1.fill_between(Igrid, lo, hi, color="r", alpha=0.12, label="provenance envelope")
        for w, marker, lab in ((0.5, "o", "w = 0.5 (R29, 8 strains)"),
                               (0.0, "s", "w = 0 (crystal control)")):
            pts = [e for e in ensemble if e["w"] == w]
            if not pts:
                continue
            x = [e["I_measured"] for e in pts]
            y = [e["mu"] for e in pts]
            yerr = [e["mu_seed_sem"] if e["mu_seed_sem"] == e["mu_seed_sem"] else 0.0
                    for e in pts]
            ax1.errorbar(x, y, yerr=yerr, fmt=marker + "-", ms=5, capsize=3, label=lab)
        ax1.set_xscale("log")
        ax1.set_xlabel("inertial number $I$")
        ax1.set_ylabel("$\\mu$")
        ax1.legend(fontsize=7)
        ax1.set_title("$\\mu(I)$: measured vs law")
        ax1.grid(alpha=0.3, which="both")

        ax2.plot(Igrid, rl.ref_phi(Igrid), "r--", label="archived phi(I) law (3-D)")
        for w, marker, lab in ((0.5, "o", "w = 0.5"), (0.0, "s", "w = 0")):
            pts = [e for e in ensemble if e["w"] == w]
            if not pts:
                continue
            ax2.errorbar([e["I_measured"] for e in pts], [e["phi"] for e in pts],
                         fmt=marker + "-", ms=5, capsize=3, label=lab)
        ax2.axhline(0.8404, ls=":", c="k", lw=1,
                    label="2-D random close packing 0.8404")
        ax2.set_xscale("log")
        ax2.set_xlabel("inertial number $I$")
        ax2.set_ylabel("$\\phi$")
        ax2.set_title("$\\phi(I)$: measured vs archived law")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = FIG / "r29_mu_phi_vs_I.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p.relative_to(REPO)))

    # 4. autocorrelation of mu_wall (one representative run)
    rep = max(good, key=lambda r: r["strain_max"])
    x = np.array(rep["series"]["mu_wall"])
    xc = x - x.mean()
    n = xc.size
    var = float(np.dot(xc, xc) / n)
    lags = np.arange(0, min(400, n // 2))
    rho = [1.0] + [float(np.dot(xc[:-k], xc[k:]) / (n * var)) for k in lags[1:]]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(lags, rho, lw=1.0)
    ax.axhline(0.0, c="k", lw=0.6)
    ax.set_xlabel("lag (samples)")
    ax.set_ylabel("$\\rho_k$ of $\\mu_{wall}$")
    ax.set_title("R29 autocorrelation, %s (tau = %.1f samples)"
                 % (rep["file"], rep["stats"]["mu_wall"]["tau"]))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / "r29_autocorr.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p.relative_to(REPO)))
    return written


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    smoke = "--smoke" in argv
    pattern = "long_r29_w*.json" if not smoke else "long_r29_w*.json"
    runs = load_runs(pattern)
    if smoke:
        runs = [r for r in runs if r["_file"] in
                {p.name for p in RUNS.glob("long_r29_w*.json")}
                and json.loads((RUNS / r["_file"]).read_text())["argv"].get("smoke")]
    if not runs:
        print("no long_r29 runs found", file=sys.stderr)
        return 2
    reduced = [reduce_run(d) for d in runs]
    ensemble = seed_ensemble(reduced)
    curves = {}
    for w in sorted({e["w"] for e in ensemble}):
        pts = [e for e in ensemble if e["w"] == w]
        if len(pts) >= 2:
            curves[f"w={w:g}"] = curve_envelope([e["I_measured"] for e in pts],
                                               [e["mu"] for e in pts])
    baseline = archived_baseline()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"round": 29, "mission": 22733, "tolerance": TOL,
               "reference": {"mu_s": rl.MU_S, "mu_2": rl.MU_2, "I_0": rl.I_0,
                             "status": rl.MU_REF_STATUS},
               "runs": [{k: v for k, v in r.items() if k != "series"} for r in reduced],
               "ensemble": ensemble,
               "curve_envelope": curves,
               "archived_1_strain": baseline}
    (OUT / "r29_summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "r29_series.json").write_text(
        json.dumps({r["file"]: r["series"] for r in reduced if "series" in r}))
    (OUT / "r29_tables.md").write_text(md_tables(reduced, ensemble, baseline, curves, smoke))
    figs = make_figures(reduced, ensemble)
    print((OUT / "r29_tables.md").read_text())
    print("figures:", *figs, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())