"""Render the m17503 report artefacts (figure + markdown tables) from results.json.

No number in the report is typed by hand: the tables are printed from the JSON the
solver wrote and the figure is drawn from the same source.
"""
from __future__ import annotations

import base64
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/xval_periodic_sllod_m17503"


def main() -> int:
    d = json.loads((OUT / "results.json").read_text())
    cases = d["cases"]
    gammas = sorted({c["gamma_dot"] for c in cases})
    by = {(c["gamma_dot"], c["arm"]): c for c in cases}

    # ---- markdown tables ---------------------------------------------------
    lines = ["| gamma_dot [1/s] | arm | P [Pa] | tau [Pa] | tau/P | tau/P -/+ se | coord | I | strain_end |",
             "|---|---|---|---|---|---|---|---|---|"]
    for g in gammas:
        for arm in ("legacy", "lees_edwards"):
            c = by[(g, arm)]
            se = (c["shear_stderr"] / c["pressure"]) if c["pressure"] else float("nan")
            lines.append(f"| {g:.0f} | {arm} | {c['pressure']:.3e} | {c['shear']:+.3e} | "
                         f"{c['tau_over_p']:+.4f} | {se:.4f} | {c['coord']:.2f} | "
                         f"{c['I']:.3f} | {c['strain_end']:.3f} |")
    cross = ["| gamma_dot [1/s] | tau/P legacy | tau/P LE | rel. diff vs LE | verdict (<15 %) |",
             "|---|---|---|---|---|"]
    for g in gammas:
        a = by[(g, "legacy")]; b = by[(g, "lees_edwards")]
        rel = abs(a["tau_over_p"] - b["tau_over_p"]) / abs(b["tau_over_p"]) if b["tau_over_p"] else float("nan")
        cross.append(f"| {g:.0f} | {a['tau_over_p']:+.4f} | {b['tau_over_p']:+.4f} | "
                     f"{100 * rel:.1f} % | {'PASS' if rel < 0.15 else 'FAIL'} |")
    inv = d["invariants"]
    inv_tab = ["| arm | contacts | straddling y contacts | straddle fraction | frame-invariance residual | un-corrected rate error |",
               "|---|---|---|---|---|---|"]
    for arm in ("legacy", "lees_edwards"):
        r = inv[arm]
        inv_tab.append(f"| {arm} | {r['n_contacts']} | {r['n_straddle']} | "
                       f"{100 * r['straddle_fraction']:.1f} % | {r['frame_invariance_rel']:.3e} | "
                       f"{100 * r['rate_error_legacy_exact']:.1f} % |")
    tables = {"cases": "\n".join(lines), "cross": "\n".join(cross), "invariants": "\n".join(inv_tab)}
    (OUT / "tables.json").write_text(json.dumps(tables, indent=2))
    print(tables["cases"]); print(); print(tables["cross"]); print(); print(tables["invariants"]); print()

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for arm, style in (("legacy", dict(marker="o", ls="--", color="C3")),
                       ("lees_edwards", dict(marker="s", ls="-", color="C0"))):
        p = [by[(g, arm)]["tau_over_p"] for g in gammas]
        e = [by[(g, arm)]["shear_stderr"] / max(by[(g, arm)]["pressure"], 1e-30) for g in gammas]
        ax[0].errorbar(gammas, p, yerr=e, capsize=3, label=arm, **style)
        ax[1].semilogy(gammas, [by[(g, arm)]["pressure"] for g in gammas], label=arm, **style)
    ax[0].axhspan(0.30, 0.42, color="0.85", label="dense-flow mu_s 0.30-0.42")
    ax[0].set_xlabel(r"$\dot\gamma$ [1/s]"); ax[0].set_ylabel(r"$\tau/P$")
    ax[0].set_title("macroscopic friction, strain-matched"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel(r"$\dot\gamma$ [1/s]"); ax[1].set_ylabel("P [Pa]")
    ax[1].set_title("pressure at identical packing fraction")
    names = ["frame\ninvariance", "straddling-contact\nrate error"]
    x = [0, 1]
    for k, (arm, col) in enumerate((("legacy", "C3"), ("lees_edwards", "C0"))):
        r = inv[arm]
        vals = [max(r["frame_invariance_rel"], 1e-16), max(r["rate_error_legacy_exact"], 1e-16)]
        ax[2].bar([xx + 0.35 * (k - 0.5) for xx in x], vals, width=0.35,
                  label=arm, color=col)
    ax[2].set_yscale("log"); ax[2].set_xticks(x); ax[2].set_xticklabels(names)
    ax[2].set_ylabel("relative error"); ax[2].axhline(0.15, color="k", ls=":", label="15 % criterion")
    ax[2].set_title("pair-level invariants"); ax[2].legend(fontsize=7)
    fig.suptitle("m17503 XVAL: periodic SLLOD (no tilt) vs Lees-Edwards cell, phi=0.80, mu=0.5, N=64")
    fig.tight_layout()
    png = OUT / "xval_periodic_sllod.png"
    fig.savefig(png, dpi=140)
    print("wrote", png, "and", OUT / "tables.json")
    print("png_base64_len", len(base64.b64encode(png.read_bytes())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())