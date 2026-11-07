"""Figures for the IN34-W3-N01 broken-ice microstructure study.

Every figure is regenerated from the stage JSON written by
``scripts/broken_ice_micro_study.py`` -- nothing is transcribed by hand.

    python scripts/broken_ice_micro_figures.py --out-dir artifacts/broken_ice_micro_study
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_parts(out_dir: Path) -> dict:
    parts = out_dir / "parts"
    data: dict = {}
    for name in ("fsd", "janssen", "chain"):
        path = parts / f"{name}.json"
        if path.exists():
            data[name] = json.loads(path.read_text())
    arch_files = sorted(parts.glob("arch_*.json"))
    if arch_files:
        data["arch"] = [json.loads(p.read_text()) for p in arch_files]
    return data


def make_figures(out_dir) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = _load_parts(out_dir)
    written = []

    # --- (1) FSD log-log with the fitted power law -------------------------
    fsd = data.get("fsd", {}).get("cases", [])
    if fsd:
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        colours = plt.cm.viridis([0.15, 0.4, 0.65, 0.9])
        for colour, case in zip(colours, fsd):
            block = case.get("fragmented_only") or case.get("all")
            if not block or not block.get("binned_x"):
                continue
            ax.loglog(block["binned_x"], block["binned_y"], "o", ms=4, color=colour,
                      label=f"{case['label']}: alpha={block['alpha']:.2f}")
            lo, hi = block["x_min_m"], block["x_max_m"]
            xs = [lo, hi]
            ys = [block["binned_y"][0] * (x / lo) ** (-block["alpha"]) for x in xs]
            ax.loglog(xs, ys, "-", color=colour, lw=1.0)
        ax.set_xlabel("fragment equivalent diameter D [m]")
        ax.set_ylabel(r"number density $n(D)$ [m$^{-1}$]")
        ax.set_title("Level-ice fragmentation: floe-size distribution\n"
                     "(markers: log-binned census, lines: MLE power law)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        path = fig_dir / "fig1_fsd_power_law.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # --- (2) clamping / clogging curve -------------------------------------
    arch_runs = []
    for part in data.get("arch", []):
        arch_runs.extend(part.get("runs", []))
    if arch_runs:
        ratios = sorted({round(r["width_ratio"], 3) for r in arch_runs})
        prob, counts = [], []
        for ratio in ratios:
            sel = [r for r in arch_runs if round(r["width_ratio"], 3) == ratio]
            prob.append(sum(1 for r in sel if r["clogged"]) / len(sel))
            counts.append(len(sel))
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(ratios, prob, "o-", color="#1f4e79", label="DEM clogging probability")
        ax.axhline(0.5, ls="--", lw=0.8, color="#888888")
        ax.axvspan(3.0, 6.0, color="#d9ead3", alpha=0.6, label="campaign window [3, 6]")
        for x, y, n in zip(ratios, prob, counts):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7)
        ax.set_xlabel("channel outlet width ratio W/h")
        ax.set_ylabel(r"$P_\mathrm{clog}$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Broken-ice channel discharge: arching/clogging criterion")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        path = fig_dir / "fig2_clogging_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # --- (3) Janssen wall-stress profile -----------------------------------
    cases = data.get("janssen", {}).get("cases", [])
    if cases:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        for case in cases:
            fit = case["fit"]
            depths = fit["depths_m"]
            ax.plot(depths, fit["measured_Pa"], "o", ms=3.5,
                    label=f"DEM {case['label']}")
            ax.plot(depths, fit["model_Pa"], "-", lw=1.2,
                    label=(f"Janssen fit {case['label']}: "
                           f"err$_{{max}}$={100 * fit['max_relative_error']:.0f}%"))
        ax.set_xlabel("depth below free surface z [m]")
        ax.set_ylabel(r"cumulative wall normal stress $\Sigma_w$ [Pa m]")
        ax.set_title("Janssen wall-load profile of the confined broken-ice column")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
        path = fig_dir / "fig3_janssen_profile.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # --- (4) contact-force distribution ------------------------------------
    chain_cases = data.get("chain", {}).get("cases", [])
    pdf_cases = []
    for case in chain_cases:
        chain_fit = (case.get("contact_network") or {}).get("force_chain")
        if chain_fit and chain_fit.get("binned_x"):
            pdf_cases.append((case["label"], chain_fit))
    for case in cases:
        net = case.get("network") or {}
        chain_fit = net.get("force_chain")
        if chain_fit and chain_fit.get("binned_x"):
            pdf_cases.append((f"2D column {case['label']}", chain_fit))
    if pdf_cases:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        for label, chain_fit in pdf_cases[:5]:
            ax.semilogy(chain_fit["binned_x"], chain_fit["binned_p"], "o-", ms=3,
                        label=f"{label}: tail rate={chain_fit['tail_rate']:.2f}, "
                              f"$R^2$={chain_fit['tail_r2']:.2f}")
        xs = [0.05, 4.0]
        ax.semilogy(xs, [math.exp(-0.05), math.exp(-4.0)], "k--", lw=0.8,
                    label=r"unit-mean exponential $e^{-f}$")
        ax.set_xlabel(r"normalised contact force $f = F/\langle F\rangle$")
        ax.set_ylabel(r"$P(f)$")
        ax.set_title("Contact-force distribution of the compacted broken-ice medium")
        ax.set_ylim(1e-3, 1e1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
        path = fig_dir / "fig4_force_chain_pdf.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts/broken_ice_micro_study")
    args = ap.parse_args(argv)
    for path in make_figures(args.out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())