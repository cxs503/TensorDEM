"""Analyse the plane-shear mu(I) sweep: tables, fit, figures (zero hand-copy).

Reads ``runs/sweep.json`` (written by ``run_sweep.py``) and emits
``runs/analysis.json`` + ``figures/*.png``.  Every number in the report is
injected from these artifacts.

Round-3 fix (m16568): use ``phi_bed_steady`` (steady-state plate height during
shear) as the primary solid-fraction measure for the phi(I) curve.  The old
``phi_bed`` used the *settled* plate height (before shear), which is constant
across all I and cannot capture shear dilatancy.  When ``phi_bed_steady`` is
absent (stale sweep.json from before the field was added), fall back to
``phi_bed`` with a warning.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def _phi_for_row(r: dict) -> tuple[float, str]:
    """Return the best available solid-fraction measure and its provenance.

    Priority: ``phi_bed_steady`` (steady-state during shear, captures dilatancy)
    > ``phi_bed`` (settled height, constant across I, legacy) > ``phi_device``
    (bulk-region slab, layering artifact).
    """
    if "phi_bed_steady" in r and r["phi_bed_steady"] is not None:
        return float(r["phi_bed_steady"]), "phi_bed_steady"
    if "phi_bed" in r and r["phi_bed"] is not None:
        return float(r["phi_bed"]), "phi_bed"
    if "phi_device" in r and r["phi_device"] is not None:
        return float(r["phi_device"]), "phi_device"
    return float("nan"), "none"


def main() -> None:
    data = json.loads((HERE / "runs" / "sweep.json").read_text())
    rows = data["rows"]
    I = np.array([r["I_measured"] for r in rows])
    mu = np.array([r["mu_measured"] for r in rows])
    phi_dev = np.array([r["phi_device"] for r in rows])
    ref_mu = np.array([r["ref_mu"] for r in rows])
    ref_phi = np.array([r["ref_phi"] for r in rows])

    # Use the best available phi measure per row (round-3 fix)
    phi_vals = []
    phi_provenance = []
    for r in rows:
        val, prov = _phi_for_row(r)
        phi_vals.append(val)
        phi_provenance.append(prov)
    phi_bed = np.array(phi_vals)

    # Warn if any row fell back to legacy measure
    legacy = [p for p in phi_provenance if p != "phi_bed_steady"]
    if legacy:
        print(f"WARNING: {len(legacy)}/{len(rows)} rows use legacy phi measure "
              f"({', '.join(set(legacy))}); re-run sweep for phi_bed_steady.",
              file=sys.stderr)

    mu_err = 100.0 * (mu / ref_mu - 1.0)
    phi_err = 100.0 * (phi_bed / ref_phi - 1.0)

    # --- fit the law to the measured points (repo fitter) ------------------
    from tensordem.granular_column.scaling import fit_jop_law
    fit = fit_jop_law(list(I), list(mu))
    i_grid = np.logspace(-3.3, 0.5, 200)
    mu_fit = fit["mu_s"] + (fit["mu_2"] - fit["mu_s"]) / (fit["i_0"] / i_grid + 1.0)
    mu_ref_grid = (data["reference"]["mu_s"]
                   + (data["reference"]["mu_2"] - data["reference"]["mu_s"])
                   / (data["reference"]["I_0"] / i_grid + 1.0))
    phi_ref_grid = (data["reference"]["phi_c"]
                    + (data["reference"]["phi_j"] - data["reference"]["phi_c"])
                    / (data["reference"]["I_0"] / i_grid + 1.0))

    # --- figure 1: mu(I) ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    ax.semilogx(i_grid, mu_ref_grid, "-", color="0.35", lw=2,
                label="ref: da Cruz 2005 / GDR MiDi 2004")
    ax.semilogx(i_grid, mu_fit, "--", color="tab:red", lw=1.5,
                label=f"fit: $\\mu_s$={fit['mu_s']:.3f}, "
                      f"$\\mu_2$={fit['mu_2']:.3f}, $I_0$={fit['i_0']:.3f}")
    ax.semilogx(I, mu, "o", color="tab:blue", ms=7,
                label="TensorDEM servo shear cell")
    for r, e in zip(rows, mu_err):
        ax.annotate(f"{e:+.0f}%", (r["I_measured"], r["mu_measured"]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8)
    ax.set_xlabel("inertial number $I = \\dot\\gamma d / \\sqrt{P/\\rho_p}$")
    ax.set_ylabel("effective friction $\\mu = \\tau/P$")
    ax.set_title("Plane shear: $\\mu(I)$ vs glass-bead reference")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "mu_of_I.png")
    plt.close(fig)

    # --- figure 2: phi(I) ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    ax.semilogx(i_grid, phi_ref_grid, "-", color="0.35", lw=2,
                label="ref: $\\phi_c$=0.58, $\\phi_j$=0.64")
    ax.semilogx(I, phi_bed, "s", color="tab:green", ms=7,
               label=f"TensorDEM bed solid fraction ({phi_provenance[0]})")
    ax.semilogx(I, phi_dev, "^", color="tab:orange", ms=6,
                label="device slab solid fraction (layering artifact)")
    for r, e in zip(rows, phi_err):
        ax.annotate(f"{e:+.0f}%", (r["I_measured"], r.get("phi_bed_steady", r["phi_bed"])),
                   textcoords="offset points", xytext=(0, 9),
                   ha="center", fontsize=8)
    ax.set_xlabel("inertial number $I$")
    ax.set_ylabel("solid fraction $\\phi$")
    ax.set_title("Plane shear: $\\phi(I)$ vs glass-bead reference")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "phi_of_I.png")
    plt.close(fig)

    # --- figure 3: relative error vs I ---------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(10, color="r", ls=":", lw=1)
    ax.axhline(-10, color="r", ls=":", lw=1)
    ax.semilogx(I, mu_err, "o-", color="tab:blue", label="mu error")
    ax.semilogx(I, phi_err, "s-", color="tab:green", label="phi error")
    ax.set_xlabel("inertial number $I$")
    ax.set_ylabel("relative error vs reference [%]")
    ax.set_title("Calibration error (band = $\\pm10\\%$ gate)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "error_vs_I.png")
    plt.close(fig)

    # --- figure 4: dilatancy (only if phi_bed_steady available) --------------
    has_dilatancy = all("dilatancy" in r and r["dilatancy"] is not None for r in rows)
    if has_dilatancy:
        dilatancy = np.array([r["dilatancy"] for r in rows])
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        ax.axhline(0, color="k", lw=0.8)
        ax.semilogx(I, dilatancy, "d-", color="tab:purple", ms=7,
                    label="dilatancy = (h_steady - h_settled) / h_settled")
        ax.set_xlabel("inertial number $I$")
        ax.set_ylabel("dilatancy")
        ax.set_title("Shear dilatancy vs inertial number")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "dilatancy_vs_I.png")
        plt.close(fig)

    out = {
        "points": [
            {"I": float(r["I_measured"]), "mu": float(r["mu_measured"]),
             "ref_mu": float(r["ref_mu"]), "mu_err_pct": float(e),
             "phi_bed": float(r.get("phi_bed_steady", r["phi_bed"])),
             "phi_provenance": prov,
             "ref_phi": float(r["ref_phi"]),
             "phi_err_pct": float(pe),
             "coordination": float(r["coordination"]),
             "n_particles": int(r["n_particles"]),
             "wall_clock_s": float(r["wall_clock_s"]),
             "dilatancy": r.get("dilatancy")}
            for r, e, pe, prov in zip(rows, mu_err, phi_err, phi_provenance)
        ],
        "fit": {k: (float(v) if isinstance(v, (int, float)) else v)
                   for k, v in fit.items()},
        "fit_vs_literature_pct": {
            "mu_s": 100.0 * (fit["mu_s"] / data["reference"]["mu_s"] - 1.0),
            "mu_2": 100.0 * (fit["mu_2"] / data["reference"]["mu_2"] - 1.0),
            "i_0": 100.0 * (fit["i_0"] / data["reference"]["I_0"] - 1.0),
        },
        "worst_mu_err_pct": float(np.max(np.abs(mu_err))),
        "worst_phi_err_pct": float(np.max(np.abs(phi_err))),
        "mu_within_10pct": bool(np.all(np.abs(mu_err) < 10.0)),
        "phi_within_10pct": bool(np.all(np.abs(phi_err) < 10.0)),
        "phi_provenance_note": (
            "phi_bed_steady (steady-state plate height during shear) is the "
            "correct measure for phi(I); phi_bed (settled height) is constant "
            "and cannot capture dilatancy. Re-run sweep to populate "
            "phi_bed_steady." if legacy
            else "phi_bed_steady used (steady-state plate height during shear)"
        ),
    }
    (HERE / "runs" / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
