#!/usr/bin/env python
"""Figures for the LW3-N06 bulk-cargo verification surface (from results.json only).

Every number plotted here is read back from ``artifacts/bulk_cargo/results.json``
(which in turn holds the raw case outputs), so no figure can drift from the data
and nothing is hand-copied.  Writes PNGs into ``docs/bulk_cargo/``.

    python scripts/m8875_bulk_cargo_figures.py [--results artifacts/bulk_cargo/results.json]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.join("docs", "bulk_cargo")


def _fetch(raw, cid):
    for r in raw:
        if r["id"] == cid:
            return r
    raise KeyError(cid)


def fig_repose(raw, outdir):
    r1 = _fetch(raw, "R1_beads")
    r4 = _fetch(raw, "R4_frictionless")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for rec, label, color in ((r1, "poured heap (mu=0.5, mu_r=0.05)", "tab:blue"),
                              (r4, "frictionless control (mu=mu_r=0)", "tab:red")):
        rep = rec["repose"]
        r = np.array(rep["surface"]["r"])
        z = np.array(rep["surface"]["z"])
        ax.plot(r * 1e3, z * 1e3, "o-", ms=3, lw=1.0, color=color, label=label)
        lo, hi = rep["fit_range_m"]
        m = (r >= lo) & (r <= hi)
        if m.sum() >= 2:
            fit = z[m][0] - rep["tan_theta"] * (r[m] - r[m][0])
            ax.plot(r[m] * 1e3, fit * 1e3, "--", lw=1.6, color=color,
                    label=f"cone fit: theta = {rep['theta_deg']:.1f} deg "
                          f"(r2={rep['r2']:.2f})")
    ax.set_xlabel("radial distance from heap centre [mm]")
    ax.set_ylabel("free-surface height [mm]")
    ax.set_title("(a) poured-heap free surface and conical fit")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    pours = [r for r in raw if r["kind"] == "pour"]
    pours.sort(key=lambda r: r["mu_rolling"])
    th = [r["repose"]["theta_deg"] for r in pours]
    lab = [f"mu_r={r['mu_rolling']:.2f}\nmu={r['mu']:.2f}" for r in pours]
    ax.bar(range(len(pours)), th, color="tab:blue", alpha=0.8)
    ax.set_xticks(range(len(pours)))
    ax.set_xticklabels(lab, fontsize=7)
    ax.axhspan(25.0, 35.0, color="tab:green", alpha=0.12,
               label="literature band 25-35 deg (sub-rounded cargo)")
    ax.set_ylabel("measured angle of repose [deg]")
    ax.set_title("(b) repose angle vs rolling resistance")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_repose.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_beverloo(raw, outdir, crit):
    c2 = crit["C2_beverloo_curve"]
    pts = c2["points"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    D = np.array([p["D_m"] for p in pts]) * 1e3
    W = np.array([p["W_measured_kg_s"] for p in pts])
    Wt = np.array([p["W_theory_kg_s"] for p in pts])
    Ws = np.array([p["W_std_kg_s"] for p in pts])
    ax.errorbar(D, W, yerr=Ws, fmt="o", color="tab:blue", capsize=3,
                label="DEM mass balance")
    ax.plot(D, Wt, "s--", color="tab:red",
            label="Beverloo  W = C rho_b sqrt(g) (D - k d)^(5/2)")
    for p, x, y in zip(pts, D, W):
        ax.annotate(f"{p['rel_dev'] * 100:.1f}%", (x, y), textcoords="offset points",
                    xytext=(4, 6), fontsize=7)
    ax.set_xlabel("outlet diameter D [mm]")
    ax.set_ylabel("mass flow rate W [kg/s]")
    ax.set_title("(a) discharge-rate curve, labels = deviation from Beverloo")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    d = pts[0]["D_m"] / pts[0]["D_over_d"]
    x = np.array([p["D_m"] - 1.4 * d for p in pts])
    ax.loglog(x, W, "o", color="tab:blue", label="measured")
    if c2["fit_exponent"] == c2["fit_exponent"]:
        xs = np.linspace(x.min(), x.max(), 50)
        ax.loglog(xs, c2["fit_prefactor"] * xs ** c2["fit_exponent"], "-",
                  color="tab:green",
                  label=f"fit exponent p = {c2['fit_exponent']:.2f} "
                        f"(Beverloo: 2.50, r2={c2['fit_r2']:.3f})")
    ax.set_xlabel("effective aperture D - k d [m]")
    ax.set_ylabel("W [kg/s]")
    ax.set_title("(b) power-law exponent of the measured curve")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_beverloo.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_janssen(raw, outdir, crit):
    j = _fetch(raw, "J_mu040")
    j2 = _fetch(raw, "J_mu020")
    prof = j["profile"]
    depth = np.array([j["bed_top_m"] - row["z_mid"] for row in prof])
    sig_h = np.array([row["sigma_h_Pa"] for row in prof])
    sig_v = np.array([row["sigma_v_Pa"] for row in prof])
    hydro = np.array(j["hydrostatic_reference_Pa"])
    fit = j["janssen_fit_sigma_v"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(sig_h, depth * 1e3, "o-", color="tab:orange", ms=4,
            label="measured wall stress sigma_h (normal force / wall area)")
    ax.plot(sig_v, depth * 1e3, "s-", color="tab:blue", ms=4,
            label="measured vertical stress sigma_v (slice force balance)")
    ax.plot(hydro, depth * 1e3, ":", color="tab:gray",
            label="hydrostatic rho_b g z")
    zfit = np.linspace(depth.min(), depth.max(), 100)
    ax.plot(fit["p_inf"] * (1 - np.exp(-zfit / fit["lambda"])), zfit * 1e3, "--",
            color="tab:red",
            label=f"Janssen fit: p_inf={fit['p_inf']:.0f} Pa, "
                  f"lambda={fit['lambda'] * 1e3:.1f} mm")
    ax.set_xlabel("stress [Pa]")
    ax.set_ylabel("depth below free surface [mm]")
    ax.set_title("(a) hold wall/base pressure profile (plugged hold)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for rec, color in ((j, "tab:blue"), (j2, "tab:purple")):
        ident = rec["identity_p_inf_over_lambda_Pa_per_m"]
        theory = rec["identity_theory_Pa_per_m"]
        ax.bar(["mu_w=%.2f\nmeasured" % rec["wall_mu"],
                "mu_w=%.2f\ntheory rho_b g" % rec["wall_mu"]],
               [ident, theory], color=[color, "tab:gray"], alpha=0.85)
        ax.annotate(f"{abs(ident - theory) / theory * 100:.1f}%",
                    (0.0, ident), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    ax.set_ylabel("p_inf / lambda  [Pa/m]")
    ax.set_title("(b) parameter-free Janssen identity p_inf/lambda = rho_b g")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_janssen.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_controls(raw, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    ax = axes[0]
    blocked = _fetch(raw, "W_D008_blocked")
    flowing = [r for r in raw if r["role"] == "beverloo"]
    labels = [f"{r['id']}\nD/d={r['outlet_over_d']:.1f}" for r in flowing]
    labels.append(f"{blocked['id']}\nD/d={blocked['outlet_over_d']:.1f}")
    vals = [r["flow_rate_kg_s"] for r in flowing] + [blocked["flow_rate_kg_s"]]
    colors = ["tab:blue"] * len(flowing) + ["tab:red"]
    ax.bar(range(len(vals)), vals, color=colors, alpha=0.85)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=6, rotation=30)
    ax.set_ylabel("measured W [kg/s]")
    ax.set_title("(a) negative control: D < 2d arches over (red)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for r, color in (("W_D040", "tab:blue"), ("W_D056", "tab:green")):
        rec = _fetch(raw, r)
        w = np.array(rec["flow_rate_windows_kg_s"])
        win = rec["params"]["discharge_time"] / rec["params"]["n_windows"]
        t = (np.arange(1, len(w) + 1)) * win
        ax.plot(t, w, "o-", color=color, label=f"{r} (D={rec['outlet_diameter'] * 1e3:.0f} mm)")
    ax.set_xlabel("time after opening [s]")
    ax.set_ylabel("window flow rate [kg/s]")
    ax.set_title("(b) steadiness of the measured flow (per-window mass balance)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "fig_controls.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join("artifacts", "bulk_cargo", "results.json"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    with open(args.results) as fh:
        doc = json.load(fh)
    raw = doc["raw"]
    os.makedirs(args.out, exist_ok=True)
    made = [
        fig_repose(raw, args.out),
        fig_beverloo(raw, args.out, {"C2_beverloo_curve": doc["criteria"]["C2_beverloo_curve"]}),
        fig_janssen(raw, args.out, doc["criteria"]),
        fig_controls(raw, args.out),
    ]
    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())