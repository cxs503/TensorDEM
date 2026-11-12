#!/usr/bin/env python
"""Figures for CASE-DEM-ROLLING-SINGLE-PARTICLE (reads result.json, no recompute).

    PYTHONPATH=src python benchmarks/case_rolling_single_particle/figures.py \
        --result artifacts/case_rolling_single_particle/result.json \
        --outdir artifacts/case_rolling_single_particle

Produces ``case_rolling_single_particle.png`` (four panels: velocity/spin fit,
instantaneous-acceleration detail, time-step convergence, reference-vs-measured).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="artifacts/case_rolling_single_particle/result.json")
    ap.add_argument("--outdir", default="artifacts/case_rolling_single_particle")
    ap.add_argument("--name", default="case_rolling_single_particle.png")
    args = ap.parse_args()

    with open(args.result) as fh:
        case = json.load(fh)
    base = case["points"]["A_base"]
    ref = base["reference"]
    got = base["got"]
    trace = base["trace"]
    detail = base["trace_detail"]
    conv = case["convergence_dt"]
    stiff = case["stiffness_invariance"]
    tol = case["summary"]["tolerance"]

    R = base["inputs"]["radius_m"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    # (a) v(t) and R*omega(t): the no-slip closure ------------------------
    ax = axes[0][0]
    t = trace["time"]
    v = trace["v"]
    ax.plot(t, v, color="#1f77b4", lw=1.4, label=r"$v(t)$ (DEM)")
    ax.plot(t, [-R * w for w in trace["omega"]], color="#d62728", lw=1.0, ls="--",
            label=r"$-R\,\omega(t)$ (DEM)")
    ax.plot(t, [ref["a_ref"] * ti for ti in t], color="k", lw=0.9, ls=":",
            label=rf"$a_{{ref}}t$, $a_{{ref}}$={ref['a_ref']:.5f} m/s$^2$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$v$ [m/s], $-R\omega$ [m/s]")
    ax.set_title(r"(a) no-slip closure $v=-R\omega$, $\theta=15^\circ$, $\mu_r=0$")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # (b) instantaneous acceleration detail -------------------------------
    ax = axes[0][1]
    a_inst = detail["a_inst"]
    dt = detail["dt"]
    tt = [i * dt for i in range(len(a_inst))]
    ax.plot(tt, a_inst, color="#2ca02c", lw=1.0, label=r"$a(t)$ instantaneous")
    ax.axhline(ref["a_ref"], color="k", lw=1.2, ls="--",
               label=rf"analytic mean $a_{{ref}}$={ref['a_ref']:.5f}")
    ax.axhline(got["a_instantaneous_max"], color="#9467bd", lw=0.8, ls=":")
    ax.axhline(got["a_instantaneous_min"], color="#9467bd", lw=0.8, ls=":",
               label="measured min/max")
    ax.set_xlabel("t [s] (3 tangential-oscillation periods)")
    ax.set_ylabel(r"$a$ [m/s$^2$]")
    ax.set_title("(b) undamped tangential spring: $a(t)$ oscillates about $a_{ref}$")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # (c) time-step convergence -------------------------------------------
    ax = axes[1][0]
    dts = [c["dt_s"] for c in conv]
    errs = [100 * c["a_rel_error"] for c in conv]
    errs_a = [100 * c["alpha_rel_error"] for c in conv]
    ax.loglog(dts, errs, "o-", color="#1f77b4", label=r"$a_t$")
    ax.loglog(dts, errs_a, "s--", color="#d62728", label=r"$\alpha$")
    ax.axhline(100 * tol, color="k", lw=1.0, ls="--", label="2 % gate")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel("relative error [%]")
    ax.set_title("(c) time-step convergence (semi-implicit Euler)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (d) reference vs measured -------------------------------------------
    ax = axes[1][1]
    pB = case["points"]["B_rolling"]
    labels = [r"$a$, $\mu_r$=0", r"$\alpha$, $\mu_r$=0", r"$a$, $\mu_r$=0.1", r"$\alpha$, $\mu_r$=0.1"]
    refs = [ref["a_ref"], ref["alpha_ref"], pB["reference"]["a_ref"], pB["reference"]["alpha_ref"]]
    gots = [got["a_t_m_per_s2"], got["alpha_rad_per_s2"],
            pB["got"]["a_t_m_per_s2"], pB["got"]["alpha_rad_per_s2"]]
    xs = list(range(len(labels)))
    # each quantity normalised by its own reference -> all bars at 100 % if exact
    ax.plot([-0.5, len(labels) - 0.5], [100, 100], color="k", lw=1.0, ls="--", label="reference = 100 %")
    ax.plot(xs, [100 * g / r for g, r in zip(gots, refs)], "o", ms=9, color="#1f77b4",
            label="TensorDEM")
    for x, g, r in zip(xs, gots, refs):
        ax.annotate(f"{100*(g-r)/r:+.4f} %", (x, 100 * g / r), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(99.9, 100.1)
    ax.set_ylabel("value normalised to reference [%]")
    ax.set_title(r"(d) reference vs measured ($\theta=15^\circ$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "CASE-DEM-ROLLING-SINGLE-PARTICLE — single sphere rolling down an incline "
        f"(TensorDEM; a_ref={ref['a_ref']:.5f} m/s$^2$, err a={100*case['summary']['a_rel_error']:.3f} %, "
        f"err alpha={100*case['summary']['alpha_rel_error']:.3f} %)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, args.name)
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())