#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO (m22628): figures for the orifice scan.

Reads ``runs/m22628_beverloo_scan/summary.json`` and writes PNGs to
``docs/figures/``.  Every number on every axis comes from the run artefacts.

    PYTHONPATH=src python benchmarks/m22628_beverloo_scan/render_figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "m22628_beverloo_scan"
SUMMARY = RUN_DIR / "summary.json"
FIG_DIR = ROOT / "docs" / "figures"
G = 9.81


def _out(name: str) -> Path:
    """Figure path; the directory is created on demand so each figure function
    is callable on its own (not only through ``main``)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR / name


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    return plt


def fig_loglog(s: dict, plt) -> Path:
    d = s["particle_diameter_m"]
    rows = s["model_A_literature"]
    D = np.array([r["D_over_d"] * d for r in rows])
    W = np.array([r["W_measured_kg_s"] for r in rows])
    Wp = np.array([r["W_pred_kg_s"] for r in rows])
    std = np.array([r["window_rel_std"] for r in rows])
    fitD = s["fits"]["D_free_Ckp"]
    fitC = s["fits"]["C_free_Ck"]
    rho = float(np.mean([r["rho_bulk"] for r in rows]))

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    eff14 = D - 1.4 * d
    ax.errorbar(np.log(eff14), np.log(W), yerr=std, fmt="o", capsize=3,
                color="C0", label="DEM (mass balance, ±1σ of windows)")
    xx = np.linspace(eff14.min() * 0.92, eff14.max() * 1.08, 60)
    ax.plot(np.log(xx), np.log(0.58 * rho * math.sqrt(G)) + 2.5 * np.log(xx),
            "--", color="0.4", label="Beverloo 1961: C=0.58, k=1.4, p=5/2")
    k, p, C = fitD["k"], fitD["p"], fitD["C"]
    if np.all(xx - k * d > 0):
        ax.plot(np.log(xx), np.log(C * rho * math.sqrt(G)) + p * np.log(xx - k * d),
                "-", color="C3",
                label=f"3-param fit: C={C:.3f}, k={k:.2f}, p={p:.3f}")
    k2 = fitC["k"]
    if np.all(xx - k2 * d > 0):
        ax.plot(np.log(xx), np.log(fitC["C"] * rho * math.sqrt(G))
                + 2.5 * np.log(xx - k2 * d), "-", color="C2",
                label=f"2-param fit (p=5/2): C={fitC['C']:.3f}, k={k2:.2f}")
    ax.set_xlabel("ln(D − 1.4·d)   [m]")
    ax.set_ylabel("ln W   [kg/s]")
    ax.set_title("CASE-DEM-SILO-BEVERLOO (m22628): orifice scan\n"
                 "slope = exponent of the discharge law", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    out = _out("m22628_beverloo_loglog.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_residuals(s: dict, plt) -> Path:
    rows = s["model_A_literature"]
    x = np.array([r["D_over_d"] for r in rows])
    lit = np.array([r["rel_err"] for r in rows]) * 100.0
    fitC = s["fits"]["C_free_Ck"]
    d = s["particle_diameter_m"]
    D = x * d
    rho = np.array([r["rho_bulk"] for r in rows])
    W = np.array([r["W_measured_kg_s"] for r in rows])
    pred = fitC["C"] * rho * math.sqrt(G) * (D - fitC["k"] * d) ** 2.5
    refit = (W / pred - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    wbar = 0.36
    ax.bar(x - wbar / 2, lit, wbar, color="0.55", label="literature C=0.58, k=1.4")
    ax.bar(x + wbar / 2, refit, wbar, color="C2",
           label=f"same data, fitted k={fitC['k']:.2f}")
    ax.axhspan(-10, 10, color="C0", alpha=0.10, label="±10 % acceptance band")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("D/d")
    ax.set_ylabel("relative error of W  [%]")
    ax.set_title("A wrong k(d) offset shows up as a D-dependent amplitude error", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    out = _out("m22628_beverloo_residuals.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_kprofile(s: dict, plt) -> Path:
    prof = s["k_profile"]
    k = np.array([r["k"] for r in prof])
    sse = np.array([r["sse_log"] for r in prof])
    fitC = s["fits"]["C_free_Ck"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(k, sse, "-", color="C0")
    ax.axvline(fitC["k"], color="C3", ls="--", label=f"fitted k = {fitC['k']:.2f}")
    ax.axvspan(1.2, 1.6, color="C2", alpha=0.15, label="literature k ∈ [1.2, 1.6]")
    ax.set_xlabel("k   (offset  D_eff = D − k·d)")
    ax.set_ylabel("profiled SSE in ln W  (C minimised out)")
    ax.set_title("Profile of the finite-size offset k at p = 5/2", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    out = _out("m22628_beverloo_kprofile.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_fill_height(s: dict, plt) -> Path:
    fh = s["fill_height_test"]
    rows = fh["rows"]
    hd = np.array([r["H_over_D"] for r in rows])
    W = np.array([r["W_kg_s"] for r in rows])
    std = np.array([r["W_std_kg_s"] for r in rows])
    base = float(W[-1])
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.errorbar(hd, W / base, yerr=std / base, fmt="o-", capsize=3, color="C0")
    ax.axhspan(0.9, 1.1, color="C0", alpha=0.10, label="±10 % band")
    ax.axhline(1.0, color="0.4", ls="--", label="Beverloo: H-independent")
    for r in rows:
        ax.annotate(f"n={r['n_bed']}", (r["H_over_D"], r["W_kg_s"] / base),
                    textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax.set_xlabel("filling height  H/D")
    ax.set_ylabel("W(H) / W(H = 45 mm)")
    ax.set_title(f"Filling-height independence of the discharge rate\n"
                 f"spread = {fh['relative_spread'] * 100:.1f} %", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    out = _out("m22628_beverloo_fill_height.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_windows(s: dict, plt) -> Path:
    """Per-window rate series: is the flow steady before it is averaged?"""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    any_series = False
    for p in sorted(RUN_DIR.glob("*.json")):
        if p.name == "summary.json":
            continue
        run = json.loads(p.read_text())
        if not isinstance(run, dict) or "flow_rate_windows_kg_s" not in run.get("result", {}):
            continue
        series = run["result"]["flow_rate_windows_kg_s"]
        if not series:
            continue
        any_series = True
        dt = run["case"]["window_time"]
        ax.plot(dt * np.arange(1, len(series) + 1), series, "o-", ms=3,
                label=run["case_id"])
    if not any_series:
        plt.close(fig)
        return _out("m22628_beverloo_windows.png")
    ax.set_xlabel("time after the plug is removed  [s]")
    ax.set_ylabel("window-averaged W  [kg/s]")
    ax.set_title("Discharge windows: steadiness before averaging", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    out = _out("m22628_beverloo_windows.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    s = json.loads(SUMMARY.read_text())
    plt = _plt()
    for fn in (fig_loglog, fig_residuals, fig_kprofile, fig_fill_height, fig_windows):
        print(fn.__name__, "->", fn(s, plt).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())