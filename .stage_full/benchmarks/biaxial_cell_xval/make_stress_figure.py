#!/usr/bin/env python3
"""Figure for the biaxial-cell stress-definition audit.

Three panels, all drawn straight from the recorded ``.npz`` dumps (no hand-typed
numbers):

1. axial stress vs axial strain for the rough arm: the production reading
   (bottom platen), the driven platen, their mean, and the internal virial with
   and without the wall term.  The two platens separate as soon as the rough side
   walls take vertical shear;
2. the additive decomposition of the old production gap into
   ``part_a`` (missing driven platen) + ``part_b`` (missing wall virial) +
   ``residual`` (closure, must vanish) for both friction arms;
3. platen-to-platen drop, measured vs the value predicted from the measured
   side-wall shear traction (vertical equilibrium), i.e. how much of the
   asymmetry the friction load path explains.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "biaxial_cell_xval"


def main() -> int:
    arms = {}
    for arm in ("rough", "smooth"):
        dump = ARTIFACTS / f"case_p100_{arm}.npz"
        analysis = ARTIFACTS / f"case_p100_{arm}_analysis.json"
        data = np.load(dump)
        arms[arm] = (data, json.loads(analysis.read_text()))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))

    data, analysis = arms["rough"]
    strains = data["strains"]
    axes[0].plot(strains, data["sigma_v"] / 1e3, label="bottom platen (production)",
                 color="#1f77b4", lw=1.4)
    axes[0].plot(strains, data["sigma_v_top"] / 1e3, label="driven (top) platen",
                 color="#d62728", lw=1.4)
    axes[0].plot(strains, data["sigma_v_wall_mean"] / 1e3, label="mean of both platens",
                 color="#2ca02c", lw=1.2, ls="--")
    axes[0].plot(strains, data["sigma_v_internal"] / 1e3, label="internal virial",
                 color="#9467bd", lw=1.0, ls=":")
    axes[0].plot(strains, data["sigma_v_full"] / 1e3, label="virial + wall term",
                 color="#ff7f0e", lw=1.0, ls="-.")
    peak = analysis["stress_definition"]["peak_index"]
    axes[0].axvline(float(strains[peak]), color="grey", lw=0.8, ls=":")
    axes[0].set_xlabel("axial strain $\\varepsilon_a$ [-]")
    axes[0].set_ylabel("$\\sigma_v$ [kPa]")
    axes[0].set_title("(a) axial stress, rough walls $\\mu=0.55$")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].grid(alpha=0.3)

    labels = ["old gap\n(internal - bottom)", "part A\n(missing top platen)",
              "part B\n(missing wall virial)", "residual\n(closure)"]
    width = 0.36
    for offset, arm, colour in ((-width / 2, "rough", "#1f77b4"),
                                (width / 2, "smooth", "#ff7f0e")):
        axial = arms[arm][1]["stress_definition"]["axial"]
        values = [axial["gap_old_internal_vs_bottom"], axial["part_a_missing_platen"],
                  axial["part_b_missing_wall_virial"], axial["residual_full_vs_mean"]]
        positions = np.arange(len(values)) + offset
        axes[1].bar(positions, np.array(values) * 100.0, width=width,
                    label=f"{arm} walls", color=colour)
        for x, v in zip(positions, values):
            axes[1].annotate(f"{v * 100:.1f}", (x, v * 100.0), ha="center",
                             va="bottom" if v >= 0 else "top", fontsize=7)
    axes[1].axhline(15.0, color="red", lw=0.8, ls="--", label="15 % acceptance")
    axes[1].axhline(-15.0, color="red", lw=0.8, ls="--")
    axes[1].axhline(0.0, color="k", lw=0.6)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("relative to the bottom-platen reading [%]")
    axes[1].set_title("(b) gap decomposition at peak")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, axis="y")

    for arm, colour in (("rough", "#1f77b4"), ("smooth", "#ff7f0e")):
        axial = arms[arm][1]["stress_definition"]["axial"]
        axes[2].bar([arm], [axial["drop_measured"] / 1e3], color=colour, alpha=0.55,
                    label="measured $\\sigma_v^{top}-\\sigma_v^{bottom}$")
        axes[2].bar([arm], [axial["drop_predicted"] / 1e3], width=0.4,
                    color="#2ca02c", alpha=0.85, label="predicted from $2\\tau H/W$")
    axes[2].set_ylabel("platen-to-platen drop [kPa]")
    axes[2].set_title("(c) load path: platen asymmetry vs side-wall shear")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = ARTIFACTS / "stress_definition.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())