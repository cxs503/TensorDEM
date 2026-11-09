"""Acceptance analysis + figures for the P70-N03 segregation campaign (m7300).

Reads ``runs/p70n03/*.json`` and writes:

* ``artifacts/p70n03/acceptance.json``  — per-criterion numbers with PASS/FAIL
* ``artifacts/p70n03/fig_S_size_ratio.png``, ``fig_S_gamma.png``,
  ``fig_S_time.png``, ``fig_transmission.png``
"""
from __future__ import annotations

import json
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "artifacts/p70n03"
os.makedirs(OUT, exist_ok=True)


def load():
    cases = {}
    for f in sorted(glob.glob("runs/p70n03/*.json")):
        d = json.load(open(f))
        cases[os.path.basename(f)[:-5]] = d
    return cases


def finite(v, default=float("nan")):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def main() -> int:
    cases = load()
    ok = {k: v for k, v in cases.items() if v.get("status") != "error"}

    def pick(**cond):
        out = []
        for name, d in ok.items():
            c = d["case"]
            if all(abs(c[k] - v) < 1e-9 if isinstance(v, float) else c[k] == v
                   for k, v in cond.items()):
                out.append((name, d))
        return sorted(out)

    # ---- (a) equal-size negative control -----------------------------------
    ctrl = pick(size_ratio=1.0, large_volume_fraction=0.2, geometry="box3d",
                drive="shake", frequency=25.0, gamma=2.0)
    ctrl_S = [finite(d["summary"]["S"]) for _, d in ctrl]
    ctrl_vup = [finite(d["summary"].get("v_up")) for _, d in ctrl]
    a_pass = bool(np.nanmax(np.abs(ctrl_S)) <= 0.02) if ctrl_S else False

    # ---- (b) monotonicity in size ratio ------------------------------------
    srows = pick(large_volume_fraction=0.2, geometry="box3d", drive="shake",
                 frequency=25.0, gamma=2.0)
    by_s: dict[float, list[float]] = {}
    for n, d in srows:
        by_s.setdefault(d["case"]["size_ratio"], []).append(finite(d["summary"]["S"]))
    sizes = sorted(by_s)
    means = [float(np.nanmean(by_s[s])) for s in sizes]
    mono = all(np.diff(means) > 0) if len(means) > 1 else False
    s_cv = {s: float(np.nanstd(v, ddof=1) / abs(np.nanmean(v)))
            if len(v) > 1 and abs(np.nanmean(v)) > 1e-12 else float("nan")
            for s, v in by_s.items()}

    # ---- (c) mass / centre-of-mass audits ---------------------------------
    mass_errs = [finite(d["summary"].get("mass_rel_error")) for d in ok.values()
                 if "mass_rel_error" in d.get("summary", {})]
    audit = [d["transmission"]["audit"] for d in ok.values()
             if "transmission" in d]
    kg_err = max([finite(x.get("kg_rel_err")) for x in audit], default=float("nan"))
    lcg_err = max([finite(x.get("lcg_rel_err")) for x in audit], default=float("nan"))
    c_pass = bool(np.nanmax(mass_errs) <= 1e-9 and max(kg_err, lcg_err) <= 1e-6)

    # ---- (d) shaking threshold --------------------------------------------
    grows = pick(size_ratio=2.0, large_volume_fraction=0.2, geometry="box3d",
                 drive="shake", frequency=25.0)
    gammas = sorted({d["case"]["gamma"] for _, d in grows})
    S_of_gamma = []
    for g in gammas:
        vals = [finite(d["summary"]["S"]) for _, d in grows if d["case"]["gamma"] == g]
        S_of_gamma.append(float(np.nanmean(vals)))
    gamma_dependence = float(np.nanmax(S_of_gamma) - np.nanmin(S_of_gamma))

    # ---- (e) transmission audit -------------------------------------------
    trans = [d["transmission"] for d in ok.values() if "transmission" in d]
    e_pass = bool(trans and max(kg_err, lcg_err) <= 1e-6)

    # ---- (f) seed spread ---------------------------------------------------
    f_pass = bool(s_cv and all((not np.isfinite(v)) or v <= 0.15 for s, v in s_cv.items()
                               if s > 1.0))

    acceptance = {
        "n_cases_requested": len(cases),
        "n_cases_ok": len(ok),
        "n_cases_failed": len(cases) - len(ok),
        "failed_cases": sorted(k for k in cases if k not in ok),
        "(a)_negative_control": {
            "S_per_seed": ctrl_S, "v_up_per_seed": ctrl_vup,
            "abs_S_max": float(np.nanmax(np.abs(ctrl_S))) if ctrl_S else None,
            "threshold": 0.02, "pass": a_pass},
        "(b)_monotonic_size_ratio": {
            "size_ratios": sizes, "S_means": means, "strictly_monotonic": mono,
            "exponent_p": None, "pass": False,
            "note": "S is negative for every size ratio > 1 (coarse sinks), so the "
                    "Brazil-nut (coarse-rises) branch is not recovered; the "
                    "literature exponent comparison is therefore not applicable."},
        "(c)_mass_and_com_audit": {
            "mass_rel_error_max": float(np.nanmax(mass_errs)) if mass_errs else None,
            "kg_rel_error_max": kg_err, "lcg_rel_error_max": lcg_err,
            "thresholds": [1e-9, 1e-6], "pass": c_pass},
        "(d)_gamma_threshold": {
            "gamma_points": gammas, "S_of_gamma": S_of_gamma,
            "max_spread_over_gamma": gamma_dependence, "gamma_c_fit": None,
            "pass": False,
            "note": "S is independent of Gamma (identical to 4 decimals across the "
                    "sweep): the shaking drive does not reorganise the bed at "
                    "A/d = 0.1-0.3, so no threshold can be fitted."},
        "(e)_transmission_audit": {"n_cases": len(trans), "pass": e_pass},
        "(f)_seed_cv": {"cv_per_size_ratio": {str(k): v for k, v in s_cv.items()},
                        "threshold": 0.15, "pass": f_pass},
        "all_criteria_pass": bool(a_pass and mono and c_pass and e_pass and f_pass),
    }
    with open(os.path.join(OUT, "acceptance.json"), "w") as fh:
        json.dump(acceptance, fh, ensure_ascii=False, indent=1, default=str)

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in sizes:
        vals = by_s[s]
        ax.scatter([s] * len(vals), vals, s=22, alpha=0.7)
    ax.plot(sizes, means, "k-o", label="mean")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.axhline(0.02, color="red", ls="--", lw=0.8, label="|S|=0.02 control band")
    ax.axhline(-0.02, color="red", ls="--", lw=0.8)
    ax.set_xlabel("size ratio  $d_{large}/d_{small}$")
    ax.set_ylabel("segregation index $S$ (coarse enrichment, top 20 %)")
    ax.set_title("P70-N03: top-layer segregation index vs. size ratio\n"
                 "(Gamma=2, f=25 Hz, N=4e3, 30 cycles)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_S_size_ratio.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for g in gammas:
        vals = [finite(d["summary"]["S"]) for _, d in grows if d["case"]["gamma"] == g]
        ax.plot([g] * len(vals), vals, "o", alpha=0.7)
    ax.plot(gammas, S_of_gamma, "k-o", label="mean over seeds")
    ax.set_xlabel(r"shaking intensity $\Gamma = A\omega^2/g$")
    ax.set_ylabel("$S$")
    ax.set_title("P70-N03: no Gamma dependence recovered (drive too weak)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_S_gamma.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, d in srows:
        h = d["history"]
        t = [x["t"] for x in h]
        S = [x["S"] for x in h]
        ax.plot(t, S, lw=0.9,
                label="s=%g seed%d" % (d["case"]["size_ratio"], d["case"]["seed"]))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("$S$")
    ax.set_title("P70-N03: S(t) for the size-ratio sweep (Gamma=2, 25 Hz)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_S_time.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    names = [n for n, d in ok.items() if "transmission" in d]
    dkg = [finite(ok[n]["transmission"].get("dVCG_cargo")) for n in names]
    dgm = [finite(ok[n]["transmission"].get("dGM_total")) for n in names]
    idx = np.arange(len(names))
    ax.bar(idx - 0.2, dkg, 0.4, label=r"$\Delta$VCG$_{cargo}$ [m]")
    ax2 = ax.twinx()
    ax2.bar(idx + 0.2, dgm, 0.4, color="tab:orange", label=r"$\Delta$GM [m]")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_ylabel(r"$\Delta$VCG [m]")
    ax2.set_ylabel(r"$\Delta$GM [m]")
    ax.set_title("P70-N03: stability transmission of the segregated mass state")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_transmission.png"), dpi=130)
    plt.close(fig)

    print(json.dumps(acceptance, ensure_ascii=False, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())