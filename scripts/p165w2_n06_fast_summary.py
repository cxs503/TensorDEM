"""Merge the P165-W2-N06 fast column-collapse series into an artefact + figures.

Reads every ``runs/p165w2_n06/fast/<case>.json`` produced by
``scripts/p165w2_n06_fast_series.py`` and writes

* ``artifacts/dem/p165w2_n06/fast_column_collapse.json`` -- every raw number plus
  the criterion (f) verdict (particle-number convergence <= 5 %, dt convergence),
  with MISSING cases reported as ``withheld`` (never silently skipped), and
* ``artifacts/dem/p165w2_n06/figures/fig7..fig10`` -- the convergence figures.

Numbers in the artefact are computed here, never typed by hand.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNDIR = os.path.join(REPO, "runs", "p165w2_n06", "fast")
ART = os.path.join(REPO, "artifacts", "dem", "p165w2_n06")
FIG = os.path.join(ART, "figures")

N_SERIES = ["n2k", "n4k", "n8k", "n16k", "n32k"]
DT_SERIES = ["dt05", "n8k", "dt2"]
MU_SERIES = ["mu30", "n8k", "mu60"]


def load(name: str):
    path = os.path.join(RUNDIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def spread(runs) -> dict:
    """Max-min of the plateau angles, relative to their mean."""
    ang = [r["repose_angle_deg"] for r in runs]
    mean = float(np.mean(ang))
    rng = float(np.max(ang) - np.min(ang))
    return {"angles_deg": [float(a) for a in ang],
            "n": [int(r["n_actual"]) for r in runs],
            "mean_deg": mean, "range_deg": rng,
            "relative_range": rng / abs(mean) if abs(mean) > 1e-9 else float("inf")}


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    cases = {os.path.basename(p)[:-5]: json.load(open(p))
             for p in sorted(glob.glob(os.path.join(RUNDIR, "*.json")))}
    art = {"campaign_node": "P165-W2-N06", "repo": "TensorDEM",
           "engine": "tensordem.sphere_pack_dem.SpherePack",
           "case_definition": {
               "geometry": "loose cubic packing, aspect H/L = 2, monodisperse spheres r = 20 mm",
               "bed": "2 layers of frozen spheres (rough base, no-slip)",
               "material": "rho = 2500 kg/m3, E* = 1e7 Pa, mu = 0.45, mu_rolling = 0.10, e = 0.4",
               "gates": "4 vertical half-spaces released after 400 hold steps",
               "integrator": "velocity-Verlet (KDK) with contact-force half-step correction",
               "measurement": "radial-bin free-surface fit, 0.90 quantile, 12 bins",
           },
           "cases": {k: {kk: vv for kk, vv in v.items()
                         if kk not in ("surface_profile", "repose_angle_history")}
                     for k, v in cases.items()},
           "criterion_f": {}}

    # -- particle-number convergence --------------------------------------
    runs = [cases[n] for n in N_SERIES if n in cases]
    missing = [n for n in N_SERIES if n not in cases]
    if len(runs) >= 2:
        s = spread(runs)
        s["cases"] = [n for n in N_SERIES if n in cases]
        s["missing"] = missing
        s["criterion"] = "max-min over the settled N-series <= 5 % of the mean"
        s["verdict"] = "PASS" if (not missing and s["relative_range"] <= 0.05) else \
            ("FAIL" if not missing else "WITHHELD")
        art["criterion_f"]["particle_number"] = s
    else:
        art["criterion_f"]["particle_number"] = {"verdict": "WITHHELD",
                                                 "missing": missing}

    # -- time-step convergence --------------------------------------------
    runs = [cases[n] for n in DT_SERIES if n in cases]
    if len(runs) == len(DT_SERIES):
        s = spread(runs)
        s["cases"] = DT_SERIES
        s["dt"] = [float(r["dt"]) for r in runs]
        s["verdict"] = "PASS" if s["relative_range"] <= 0.05 else "FAIL"
        art["criterion_f"]["time_step"] = s
    else:
        art["criterion_f"]["time_step"] = {
            "verdict": "WITHHELD",
            "missing": [n for n in DT_SERIES if n not in cases]}

    # -- material scan and the smooth-floor control ------------------------
    runs = [cases[n] for n in MU_SERIES if n in cases]
    if runs:
        art["criterion_f"]["friction_scan"] = spread(runs)
    if "smooth" in cases:
        art["criterion_f"]["smooth_floor_control"] = {
            "repose_angle_deg": cases["smooth"]["repose_angle_deg"],
            "note": ("parent wall path has no rolling resistance: the grains roll "
                     "outwards and the measured angle collapses to ~0 -> the "
                     "parent column test does not measure a repose angle"),
            "verdict": "negative control exercised"}

    # -- figures -----------------------------------------------------------
    def _fig_series(names, xkey, xlabel, title, fname):
        pts = [(cases[n][xkey], cases[n]["repose_angle_deg"], cases[n]["label"])
               for n in names if n in cases]
        if not pts:
            return None
        pts.sort()
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(x, y, "o-", lw=1.4, ms=6)
        mean = float(np.mean(y))
        ax.fill_between([min(x) * 0.8, max(x) * 1.2],
                        mean * 0.95, mean * 1.05, color="0.8", alpha=0.6,
                        label="+-5 % band around the mean")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("repose angle [deg]")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(FIG, fname)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return out

    figs = []
    p = _fig_series(N_SERIES, "n_actual", "particles N", "repose angle vs particle number",
                    "fig7_repose_vs_N.png")
    if p:
        figs.append({"name": "fig7_repose_vs_N.png",
                     "caption": "Repose angle vs particle number (criterion f, <= 5 % band)"})
    p = _fig_series(DT_SERIES, "dt", "timestep dt [s]", "repose angle vs timestep",
                    "fig8_repose_vs_dt.png")
    if p:
        figs.append({"name": "fig8_repose_vs_dt.png",
                     "caption": "Repose angle vs timestep (criterion f, time-step convergence)"})
    p = _fig_series(MU_SERIES, "mu", "particle friction mu", "repose angle vs friction",
                    "fig9_repose_vs_mu.png")
    if p:
        figs.append({"name": "fig9_repose_vs_mu.png",
                     "caption": "Repose angle vs inter-particle friction (material sensitivity)"})

    # height profiles of the settled heaps
    prof_names = [n for n in N_SERIES if n in cases]
    if prof_names:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for n in prof_names:
            prof = cases[n]["surface_profile"]
            if not prof:
                continue
            r = [q[0] for q in prof]
            z = [q[1] for q in prof]
            ax.plot(r, z, "o-", ms=3, lw=1.2, label=f"N={cases[n]['n_actual']}")
        ax.set_xlabel("radius from the column axis [m]")
        ax.set_ylabel("0.90-quantile height [m]")
        ax.set_title("settled free surface")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig10_surface_profiles.png"), dpi=140)
        plt.close(fig)
        figs.append({"name": "fig10_surface_profiles.png",
                     "caption": "Settled free-surface profiles for the N-series"})

    art["figures"] = figs
    out = os.path.join(ART, "fast_column_collapse.json")
    with open(out, "w") as fh:
        json.dump(art, fh, indent=1)
    print(json.dumps({"cases": sorted(cases), "artefact": out,
                      "criterion_f": art["criterion_f"]}, indent=1))


if __name__ == "__main__":
    main()