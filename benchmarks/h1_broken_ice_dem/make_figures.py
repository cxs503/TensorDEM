"""Figures for the H1-BROKEN-ICE-DEM round-2 report (from results.json only)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "artifacts", "h1_broken_ice_dem")


def main() -> None:
    data = json.load(open(os.path.join(OUT, "results.json")))
    cases = data["cases"]

    if "buoyancy" in cases:
        b = cases["buoyancy"]
        hist = b["history"]
        t = [h["t"] for h in hist]
        frac = [h["submerged_fraction"] for h in hist]
        target = b["ref"]["submerged_fraction_archimedes"]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(t, frac, lw=1.4, label="DEM, relaxed floe")
        ax.axhline(target, ls="--", color="k", lw=1.0,
                   label=f"Archimedes rho_i/rho_w = {target:.4f}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("submerged volume fraction [-]")
        ax.set_title("H1-C1 floating equilibrium: draft relaxation")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig1_buoyancy.png"), dpi=150)
        plt.close(fig)

    if "repose" in cases:
        r = cases["repose"]
        mus = [run["friction"] for run in r["runs"]]
        ang = [run["repose_angle_deg"] for run in r["runs"]]
        lo, hi = r["ref"]["band_deg"]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.axhspan(lo, hi, color="tab:green", alpha=0.15,
                   label=f"ice-rubble literature band {lo:.0f}-{hi:.0f} deg")
        ax.plot(mus, ang, "o-", color="tab:red", label="DEM rubble heap (polyhedra)")
        ax.plot(mus, [x for x in ang], "o", color="tab:red")
        ax.set_xlabel("ice-ice friction coefficient mu [-]")
        ax.set_ylabel("angle of repose [deg]")
        ax.set_title("H1-C2 rubble repose angle vs friction")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig2_repose.png"), dpi=150)
        plt.close(fig)

    if "channel" in cases:
        c = cases["channel"]
        s = c["series"]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(s["time_s"], s["concentration"], lw=1.4, color="tab:blue",
                label="corridor ice concentration")
        ax.axhline(c["summary"]["concentration_final"], ls=":", color="k", lw=1.0,
                   label=f"final {c['summary']['concentration_final']:.3f}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("ice concentration [-]")
        ax.set_title("H1-C3 channel clearing: concentration evolution")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(s["time_s"], s["resistance_N"], lw=0.9, color="tab:orange", alpha=0.7,
                 label="hull resistance")
        ax2.set_ylabel("hull resistance [N]")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig3_channel.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
