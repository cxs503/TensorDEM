"""Figures for the committed C1/C3 evidence (anchor H1-BROKEN-ICE-DEM, round 8).

Both panels are drawn from ``runs/h1_broken_ice_dem/results_c1c3.json`` -- the
file the test suite re-checks -- so the figure cannot disagree with the numbers
(it has no other input).  No hand-typed data.

    PYTHONPATH=src python benchmarks/h1_broken_ice_dem/make_figures_round8.py
    # -> docs/figures/h1_round8_c1_buoyancy.png
    # -> docs/figures/h1_round8_c3_channel.png
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "runs" / "h1_broken_ice_dem" / "results_c1c3.json"
OUT_DIR = ROOT / "docs" / "figures"


def _c1(ax, case):
    hist = case["history"]
    t = [h["t"] for h in hist]
    frac = [h["submerged_fraction"] for h in hist]
    ref = case["ref"]["submerged_fraction_archimedes"]
    ax.plot(t, frac, lw=1.2, label="DEM submerged volume fraction")
    ax.axhline(ref, color="crimson", ls="--", lw=1.0,
               label=f"Archimedes $\\rho_i/\\rho_w$ = {ref:.6f}")
    ax.set_xlim(0, t[-1] if t else 1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("submerged volume fraction [-]")
    ax.set_title("C1 floating equilibrium: relaxes onto the analytic draft")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    got, ref_d = case["got"], case["ref"]
    ax.text(0.02, 0.04,
            "measured %.6f vs analytic %.6f\nrel. error %.4f %%  |  draft %.5f m vs %.5f m"
            % (got["submerged_fraction"], ref_d["submerged_fraction_archimedes"],
               100.0 * case["error"]["rel_err_submerged_fraction"],
               got["draft_m"], ref_d["draft_archimedes_m"]),
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(fc="white", alpha=0.8, ec="0.7"))


def _c3(ax, case):
    series = case["series"]
    t = series["time_s"]
    ax.plot(t, series["lane_concentration"], lw=1.4,
            label="lane: exact clipped solid volume")
    if "lane_concentration_centroid_proxy" in series:
        ax.plot(t, series["lane_concentration_centroid_proxy"], lw=1.0, ls=":",
                label="lane: centroid-count proxy (old metric)")
    if "concentration" in series:
        ax.plot(t, series["concentration"], lw=1.0, alpha=0.8,
                label="whole-field concentration (1 - porosity)")
    got = case["got"]
    ax.axhline(0.5 * got["lane_concentration_initial"], color="crimson", ls="--",
               lw=1.0, label="half-of-initial criterion")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ice concentration [-]")
    ax.set_title("C3 channel clearing: the swept lane empties")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.04,
            "lane %.4f -> %.4f (ratio %.3f), hull advance %.2f m\nmass residual %.1e, "
            "energy residual %.3f"
            % (got["lane_concentration_initial"], got["lane_concentration_final"],
               got["lane_concentration_ratio"], got["hull_advance_m"],
               got["mass_residual"], got["energy_residual"]),
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(fc="white", alpha=0.8, ec="0.7"))


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    produced = []
    for name, drawer in (("c1_buoyancy", _c1), ("c3_channel", _c3)):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        drawer(ax, evidence["cases"][name.split("_", 1)[1]])
        fig.tight_layout()
        path = OUT_DIR / f"h1_round8_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        produced.append(str(path.relative_to(ROOT)))
    print(json.dumps({"figures": produced,
                      "passed": evidence["passed"],
                      "maturity_passed": evidence["maturity"]["passed"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())