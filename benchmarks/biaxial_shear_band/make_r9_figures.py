"""Figures for R9 of CASE-BIAXIAL-SHEAR-BAND (no simulation, JSON in, PNG out).

Reads ``artifacts/biaxial_shear_band/r9_law_test/law_test.json`` and writes

* ``figures/r9_law_comparison.png`` — every published band angle against the
  three law predictions at the specimen's own friction angle (stress-dilatancy
  dilation at the declared midpoint), with the borrowed 55-60 deg band and the
  60 deg anchor drawn;
* ``figures/r9_phi_cv_sensitivity.png`` — mean absolute relative error of each
  law against the declared critical-state friction range, with the 3 % precision
  bar and the 15 % tolerance bar;
* ``figures/r9_dilation_saturation.png`` — the reported whole-specimen dilation
  angle per specimen, the ``asin(0.95)`` clip that saturates it, and the
  stress-dilatancy replacement.

Every number is read from the JSON; nothing is retyped.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts", "biaxial_shear_band", "r9_law_test")
SOURCE = os.path.join(ART, "law_test.json")
FIGS = os.path.join(ART, "figures")
LAW_ORDER = ("mohr_coulomb", "roscoe", "arthur")
COLOURS = {"mohr_coulomb": "#1f77b4", "roscoe": "#ff7f0e", "arthur": "#2ca02c"}
LABELS = {"mohr_coulomb": "Mohr-Coulomb 45+phi/2",
          "roscoe": "Roscoe 45+psi/2",
          "arthur": "Arthur 45+(phi+psi)/4"}


def _load():
    with open(SOURCE) as handle:
        return json.load(handle)


def figure_law_comparison(payload: dict, midpoint: str = "stress_dilatancy[32]"):
    specimens = payload["specimens"]
    names = [s["name"].replace("anchor_s", "s") for s in specimens]
    theta = np.array([s["theta_deg"] for s in specimens], dtype=float)
    block = payload["ensemble"]["arms"][midpoint]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    x = np.arange(len(specimens))
    ax.axhspan(55.0, 60.0, color="#cccccc", alpha=0.5,
               label="borrowed band 55-60 deg")
    ax.axhline(60.0, color="k", lw=0.9, ls="--", label="anchor 60 deg")
    ax.plot(x, theta, "o", color="k", ms=8, label="measured Radon (R7 evidence)")
    for law in LAW_ORDER:
        values = [s["stress_dilatancy_arm"]["32.0"]["errors"][law]["predicted_deg"]
                  for s in specimens]
        ax.plot(x, values, "s--", color=COLOURS[law], ms=5, lw=1.2,
                label=LABELS[law])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.set_ylabel("band inclination from sigma_3 axis (deg)")
    ax.set_title(f"same field, three laws\n(dilation from stress-dilatancy, phi_cv = 32 deg; "
                 f"mean |err| Arthur = {block['arthur']['abs_relative_error']['mean'] * 100:.2f} %)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    arms = [(name, arms) for name, arms in payload["ensemble"]["arms"].items()]
    index = np.arange(len(arms))
    width = 0.26
    for i, law in enumerate(LAW_ORDER):
        values = [arms[law]["abs_relative_error"]["mean"] * 100 for _, arms in arms]
        ax.bar(index + (i - 1) * width, values, width, color=COLOURS[law],
               label=LABELS[law])
    ax.axhline(3.0, color="g", ls=":", lw=1.2, label="precision bar 3 %")
    ax.axhline(15.0, color="r", ls=":", lw=1.2, label="tolerance bar 15 %")
    ax.set_xticks(index)
    ax.set_xticklabels([name.replace("stress_dilatancy", "sd").replace("measured_psi", "meas. psi")
                        .replace("zero_psi", "psi=0") for name, _ in arms],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean |error| vs measured angle (%)")
    ax.set_title("which dilation input makes which law work")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("R9 CASE-BIAXIAL-SHEAR-BAND: the withheld psi-based laws, tested on the "
                 "published eight specimens", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGS, "r9_law_comparison.png")
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_phi_cv_sensitivity(payload: dict):
    ensemble = payload["ensemble"]
    phi_cv = [float(k) for k in payload["protocol"]["phi_cv_range_deg"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for law in LAW_ORDER:
        values = [ensemble["arms"][f"stress_dilatancy[{cv:g}]"][law]
                  ["abs_relative_error"]["mean"] * 100 for cv in phi_cv]
        ax.plot(phi_cv, values, "o-", color=COLOURS[law], label=LABELS[law])
    zero = [ensemble["arms"]["zero_psi"][law]["abs_relative_error"]["mean"] * 100
            for law in LAW_ORDER]
    for law, value in zip(LAW_ORDER, zero):
        ax.axhline(value, color=COLOURS[law], ls=":", lw=0.9)
    ax.axhline(3.0, color="g", ls=":", lw=1.2, label="precision bar 3 %")
    ax.axhline(15.0, color="r", ls=":", lw=1.2, label="tolerance bar 15 %")
    ax.set_xlabel("declared critical-state friction angle phi_cv (deg)")
    ax.set_ylabel("mean |error| over 8 specimens (%)")
    ax.set_title("sensitivity to the declared phi_cv\n(dotted: same law with psi = 0, i.e. "
                 "45+phi/4 vs 45+phi/2 only)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "r9_phi_cv_sensitivity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_dilation_saturation(payload: dict):
    specimens = payload["specimens"]
    names = [s["name"].replace("anchor_s", "s") for s in specimens]
    measured = [s["state_primary"]["psi_deg"] for s in specimens]
    sd = [s["stress_dilatancy_arm"]["32.0"]["psi_deg"] for s in specimens]
    clip = payload["ensemble"]["dilation_saturation"]["clip_deg"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(names))
    ax.bar(x - 0.2, measured, 0.4, color="#9467bd",
           label="reported whole-specimen dilation angle (R7)")
    ax.bar(x + 0.2, sd, 0.4, color="#2ca02c",
           label="stress-dilatancy replacement (phi_cv = 32 deg)")
    ax.axhline(clip, color="r", ls="--", lw=1.2,
               label=f"asin(0.95) clip = {clip:.2f} deg (saturated)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.set_ylabel("dilation angle (deg)")
    ax.set_title("why the psi-based laws were withheld: the whole-specimen dilation\n"
                 f"angle saturates the fit clip in "
                 f"{payload['ensemble']['dilation_saturation']['n_saturated']} of "
                 f"{payload['ensemble']['dilation_saturation']['n']} specimens")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "r9_dilation_saturation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    payload = _load()
    for path in (figure_law_comparison(payload),
                 figure_phi_cv_sensitivity(payload),
                 figure_dilation_saturation(payload)):
        print("wrote", os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())