"""CASE-DEM-PLANE-SHEAR-MUIDI round 7: phi(I) verdict under corrected definitions.

Two definitional defects were found in the mu(I)/phi(I) case machinery (m17026):

D1 (device, ``tensordem.mu_i_rheology``) — the per-sample field exposed as
   ``solid_fraction`` was computed as ``1 - n_bulk*V_p/V_slab``, i.e. the slab
   **porosity**.  Every artifact of rounds 1-2 therefore carries
   ``phi_device`` = porosity (~0.36-0.39) while it is tabulated as a solid
   fraction.  The fix keeps the same slab estimator and returns
   ``n_bulk*V_p/V_slab`` (plus an explicit ``porosity`` field and the audit
   fields ``slab_n_particles`` / ``slab_volume``).

D2 (reference law, ``benchmarks/plane_shear_mu_i``) — ``phi(I)`` was written
   **rising** with ``I`` (0.58 -> 0.64).  Dilatancy is the opposite; the
   corrected single-source law (``reference_law.py``) is
   ``phi = 0.64 - 0.06/(I0/I+1)``, decreasing, same two magnitudes.

This script re-reads the *stored* round-1/2 sweep rows (no re-simulation), inverts
the documented legacy formula (an exact algebraic inversion, verifiable against a
fresh run), and re-states the phi(I) verdict.  Deliverables:
``runs/phi_reference_audit.json``, ``runs/phi_reference_audit_tables.md``,
``runs/fig_phi_reference_audit.png``.

Honest caveat that does not go away: the *magnitudes* (0.58, 0.64) still have no
in-repo literature anchor, so the phi acceptance of the ledger stays WITHHELD
even when the numbers fall inside the 10 % band.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from reference_law import PHI_MAX, PHI_MIN, ref_mu, ref_phi, reference  # noqa: E402

RUNS = HERE / "runs"
TOL = 0.10
LEGACY_SWEEPS = {"round1": "sweep.json",
                 "round2_baseline": "sweep_baseline_round2.json",
                 "round2_stiff": "sweep_stiff.json",
                 "round2_thick": "sweep_thick.json"}


def legacy_solid_fraction(row: dict) -> float:
    """Recover the slab solid fraction from a legacy ``phi_device`` row.

    The pre-m17026 device returned ``1 - n*V_p/V_slab``; the inversion is exact
    and is checked field-by-field against a fresh run in
    ``test_phi_estimator_and_reference.py``.
    """
    return 1.0 - float(row["phi_device"])


def audit() -> dict:
    arms = {}
    for arm, fname in LEGACY_SWEEPS.items():
        path = RUNS / fname
        if not path.exists():
            continue
        rows = json.loads(path.read_text())["rows"]
        out = []
        for r in rows:
            I = float(r["I_measured"])
            phi_legacy = float(r["phi_device"])          # porosity (defect)
            phi_solid = legacy_solid_fraction(r)
            mu = float(r["mu_measured"])
            out.append({
                "I": I,
                "mu": mu,
                "mu_ref": ref_mu(I),
                "mu_err_pct": 100.0 * (mu / ref_mu(I) - 1.0),
                "phi_legacy_porosity": phi_legacy,
                "phi_solid": phi_solid,
                "phi_ref": ref_phi(I),
                "phi_err_pct": 100.0 * (phi_solid / ref_phi(I) - 1.0),
                "phi_err_pct_legacy_reading": 100.0 * (phi_legacy / ref_phi(I) - 1.0),
                "normal_stress_pa": r.get("normal_stress_pa"),
                "phi_bed": r.get("phi_bed"),
            })
        arms[arm] = out
    summary = {}
    for arm, rows in arms.items():
        worst_phi = max(abs(x["phi_err_pct"]) for x in rows)
        worst_phi_legacy = max(abs(x["phi_err_pct_legacy_reading"]) for x in rows)
        worst_mu = max(abs(x["mu_err_pct"]) for x in rows)
        summary[arm] = {
            "worst_phi_err_pct": worst_phi,
            "worst_phi_err_pct_if_mislabeled": worst_phi_legacy,
            "phi_within_10pct": worst_phi <= 10.0,
            "worst_mu_err_pct": worst_mu,
            "mu_within_10pct": worst_mu <= 10.0,
            "phi_monotone_decreasing": all(
                rows[i + 1]["phi_solid"] <= rows[i]["phi_solid"] + 1e-3
                for i in range(len(rows) - 1)),
        }
    return {"case": "CASE-DEM-PLANE-SHEAR-MUIDI", "round": 7, "mission": 17026,
            "defects": {
                "D1_solid_fraction_was_porosity":
                    "tensordem.mu_i_rheology.simulate_servo_shear sample field "
                    "'solid_fraction' = 1 - n*V_p/V_slab (porosity); fixed to "
                    "n*V_p/V_slab, porosity exposed separately",
                "D2_phi_reference_sign":
                    "phi(I) reference was rising in I (0.58->0.64); corrected to "
                    "decreasing 0.64->0.58 (dilatancy), magnitudes unchanged, "
                    "provenance still unverified",
            },
            "reference": reference(),
            "arms": arms, "summary": summary,
            "phi_acceptance_verdict": "WITHHELD (reference magnitudes unverified)",
            }


def tables(a: dict) -> str:
    L = ["# R7 phi(I) verdict under corrected definitions (programmatic)", "",
         "> generated by `benchmarks/plane_shear_mu_i/phi_reference_audit.py`; "
         "no number is hand-copied. Legacy rows are re-read, not re-simulated.",
         "",
         "## Reference laws used", "",
         f"* mu(I)  = {a['reference']['mu']['law']}  ({a['reference']['mu']['status']})",
         f"* phi(I) = {a['reference']['phi']['law']}", "",
         f"  status: {a['reference']['phi']['status']}", ""]
    for arm, rows in a["arms"].items():
        L += [f"## arm `{arm}`", "",
              "| I | mu | mu ref | mu err % | phi_device (legacy, = porosity) | phi solid ( = 1-phi_device) | phi ref | phi err % (corrected) | phi err % (mislabeled reading) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['I']:.4g} | {r['mu']:.4f} | {r['mu_ref']:.4f} | "
                     f"{r['mu_err_pct']:+.1f} | {r['phi_legacy_porosity']:.4f} | "
                     f"{r['phi_solid']:.4f} | {r['phi_ref']:.4f} | "
                     f"{r['phi_err_pct']:+.1f} | "
                     f"{r['phi_err_pct_legacy_reading']:+.1f} |")
        s = a["summary"][arm]
        L += ["", f"* worst |phi err| corrected = {s['worst_phi_err_pct']:.1f} %; "
                  f"as mislabeled = {s['worst_phi_err_pct_if_mislabeled']:.1f} %",
              f"* phi monotone decreasing in I: {s['phi_monotone_decreasing']}",
              f"* phi within 10 % of reference: {s['phi_within_10pct']} "
              f"(acceptance still {a['phi_acceptance_verdict']})",
              f"* worst |mu err| = {s['worst_mu_err_pct']:.1f} % "
              f"(within 10 %: {s['mu_within_10pct']})", ""]
    return "\n".join(L) + "\n"


def figure(a: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.logspace(-3.2, math.log10(0.6), 200)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.semilogx(grid, ref_phi(grid), "k-", lw=2,
                label=r"corrected ref: $0.64-0.06/(I_0/I+1)$")
    rising = PHI_MIN + (PHI_MAX - PHI_MIN) / (0.30 / grid + 1.0)
    ax.semilogx(grid, rising, "r--", lw=1.4,
                label=r"legacy ref (rising, sign error)")
    for arm, style in (("round1", "o"), ("round2_baseline", "s"),
                       ("round2_thick", "^")):
        if arm not in a["arms"]:
            continue
        rows = a["arms"][arm]
        ax.semilogx([r["I"] for r in rows], [r["phi_solid"] for r in rows],
                    style + "-", ms=5, label=f"{arm}: phi solid (corrected)")
        ax.semilogx([r["I"] for r in rows],
                    [r["phi_legacy_porosity"] for r in rows], style + ":",
                    ms=3, alpha=.45, label=f"{arm}: phi_device (legacy = porosity)")
    ax.set_xlabel("I"); ax.set_ylabel(r"$\phi$"); ax.grid(alpha=.3)
    ax.legend(fontsize=6.5); ax.set_title(r"$\phi(I)$: definition and reference sign")
    ax = axes[1]
    for arm, style in (("round1", "o"), ("round2_baseline", "s"),
                       ("round2_thick", "^")):
        if arm not in a["arms"]:
            continue
        rows = a["arms"][arm]
        ax.semilogx([r["I"] for r in rows],
                    [r["phi_err_pct"] for r in rows], style + "-", ms=5,
                    label=f"{arm} corrected")
    ax.axhline(10.0, color="r", lw=.8, ls="--")
    ax.axhline(-10.0, color="r", lw=.8, ls="--")
    ax.axhline(0.0, color="k", lw=.8)
    ax.set_xlabel("I"); ax.set_ylabel(r"$\phi$ error vs reference [%]")
    ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title("phi error after the two fixes")
    fig.tight_layout()
    fig.savefig(RUNS / "fig_phi_reference_audit.png", dpi=140)
    plt.close(fig)


def main() -> int:
    a = audit()
    (RUNS / "phi_reference_audit.json").write_text(json.dumps(a, indent=2))
    (RUNS / "phi_reference_audit_tables.md").write_text(tables(a))
    figure(a)
    for arm, s in a["summary"].items():
        print(f"{arm}: worst_phi_err={s['worst_phi_err_pct']:.1f}% "
              f"(mislabeled {s['worst_phi_err_pct_if_mislabeled']:.1f}%) "
              f"worst_mu_err={s['worst_mu_err_pct']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())