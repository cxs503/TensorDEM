"""Round 15 (m20791): re-derive the archived phi(I)/mu(I) table under the

corrected solid-fraction definition, and draw the definitional-audit figure.

Inputs are the archived round-8 full-cell points (runs/full_cell/point_*.json,
untouched) plus the synthetic-packing audit.  Nothing is re-simulated here, so
the table is a pure re-expression of already-committed evidence.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (REPO / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from reference_law import ref_mu, ref_phi          # noqa: E402

FULL = HERE / "runs" / "full_cell"
OUT = HERE / "runs" / "phi_audit"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    rows = []
    for f in sorted(FULL.glob("point_*.json")):
        d = json.loads(f.read_text())
        I = float(d["I_target"])
        r = float(d.get("radius", 0.005))
        # plate_height_steady is not archived in the round-8 point files; it is
        # recovered exactly from the archived phi_bed_steady (= N Vp/(W D h))
        n_p = float(d.get("n_particles", 294))
        vp = 4.0 / 3.0 * math.pi * r ** 3
        plate = n_p * vp / (0.12 * 0.05 * float(d["phi_bed_steady"]))
        boundary = plate / max(plate - 2.0 * r, 1e-12)
        rows.append({
            "I_target": I,
            "I_bulk": d.get("I_bulk"),
            "mu_full": d.get("mu_full_final"),
            "mu_ref": ref_mu(I),
            "err_mu_pct": 100.0 * (d["mu_full_final"] / ref_mu(I) - 1.0),
            "phi_centres_archived": d.get("phi_slab"),
            "phi_container": d.get("phi_bed_steady"),
            "phi_ref": ref_phi(I),
            "err_phi_pct": 100.0 * (d["phi_bed_steady"] / ref_phi(I) - 1.0),
            "slab_over_bed_measured": (d.get("phi_slab") / d["phi_bed_steady"]
                                       if d.get("phi_slab") else None),
            "boundary_layer_factor": boundary,
            "pressure_control_error": d.get("pressure_control_error_final"),
            "servo_valid": d.get("servo_valid"),
            "rate_ratio": d.get("rate_ratio"),
        })
    rows.sort(key=lambda r: r["I_target"])

    head = ["# Round 15 - archived phi(I)/mu(I) re-expressed under the",
            "# container-averaged solid-fraction definition",
            "",
            "`phi_centres_archived` is the value rounds 6-14 compared with the",
            "reference; `phi_container` is the same run re-expressed in the",
            "definition the reference law uses (N V_p / V_gap).",
            "",
            "| I | mu_full | mu_ref | err_mu [%] | phi_centres (archived) "
            "| phi_container | phi_ref | err_phi [%] | slab/bed measured "
            "| plate/(plate-2r) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        head.append(
            f"| {r['I_target']:g} | {r['mu_full']:.4f} | {r['mu_ref']:.4f} "
            f"| {r['err_mu_pct']:+.1f} | {r['phi_centres_archived']:.4f} "
            f"| {r['phi_container']:.4f} | {r['phi_ref']:.4f} "
            f"| {r['err_phi_pct']:+.1f} | {r['slab_over_bed_measured']:.3f} "
            f"| {r['boundary_layer_factor']:.3f} |")
    txt = "\n".join(head) + "\n"
    (OUT / "archived_rederived_tables.md").write_text(txt)
    (OUT / "archived_rederived.json").write_text(json.dumps(rows, indent=2))
    print(txt)

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    synth = json.loads((OUT / "synthetic.json").read_text()) if \
        (OUT / "synthetic.json").exists() else None
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))
    Is = np.array([r["I_target"] for r in rows])
    grid = np.logspace(math.log10(Is.min() / 3), math.log10(Is.max() * 2), 60)

    ax[0].plot(Is, [r["phi_centres_archived"] for r in rows], "o--",
               label=r"centre-counted slab (archived $\phi$)")
    ax[0].plot(Is, [r["phi_container"] for r in rows], "s-",
               label=r"container-averaged $\phi$ (literature definition)")
    ax[0].plot(grid, ref_phi(grid), "k:", label="reference band (provenance-unverified)")
    ax[0].axhline(0.7405, color="grey", lw=0.8, ls="-.", label="HCP limit 0.7405")
    ax[0].set_xscale("log"); ax[0].set_xlabel("I"); ax[0].set_ylabel(r"$\phi$")
    ax[0].legend(fontsize=6.5); ax[0].set_title("archived runs, two definitions")
    ax[0].set_ylim(0.45, 0.78)

    ax[1].plot(Is, [r["slab_over_bed_measured"] for r in rows], "o-",
               label="measured slab/container")
    ax[1].plot(Is, [r["boundary_layer_factor"] for r in rows], "k--",
               label=r"$plate_z/(plate_z-2r)$ (z boundary layer)")
    if synth:
        gaps = [0.12, 0.06, 0.045]
        ax[1].plot(gaps, [r["measured_slab_over_bed"] for r in synth["rows"]],
                   "^:", color="tab:green", label="synthetic ghosts (gap [m])")
    ax[1].set_xscale("log"); ax[1].set_xlabel("I  (or gap height [m] for ghosts)")
    ax[1].set_ylabel("slab / container phi")
    ax[1].legend(fontsize=7)
    ax[1].set_title("definitional gap, measured on the state")

    ax[2].plot(Is, [r["mu_full"] for r in rows], "o-", label="full-cell virial")
    ax[2].plot(grid, ref_mu(grid), "k:", label=r"$\mu(I)$ reference")
    for r in rows:
        ax[2].annotate(f"{r['err_mu_pct']:+.0f}%", (r["I_target"], r["mu_full"]),
                       fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[2].set_xscale("log"); ax[2].set_xlabel("I"); ax[2].set_ylabel(r"$\mu$")
    ax[2].legend(fontsize=8); ax[2].set_title("friction coefficient (unchanged)")
    fig.suptitle("CASE-DEM-PLANE-SHEAR-MUIDI  R15: phi definition audit "
                 "(archived runs, no re-simulation)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_phi_definition_audit.png", dpi=140)
    print(f"figure -> {OUT / 'fig_phi_definition_audit.png'}")
    return {"rows": rows}


if __name__ == "__main__":
    main()