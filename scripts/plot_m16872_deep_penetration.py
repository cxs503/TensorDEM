"""Figure for the m16872 deep-penetration audit (reads the two sweep JSONs)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs/m16872")


def series(rows, az=0.0):
    sel = [r for r in rows if r["az_deg"] == az]
    return ([r["depth_m"] * 1e3 for r in sel],
            [r["force_N"] / 1e3 for r in sel],
            [r["rel_err"] for r in sel],
            [r["expected_N"] / 1e3 for r in sel])


def main() -> int:
    before = json.loads((RUNS / "sweep_before.json").read_text())
    after = json.loads((RUNS / "sweep_after.json").read_text())
    tb, fb, eb, xb = series(before["cylinder_sweep"])
    ta, fa, ea, xa = series(after["cylinder_sweep"])
    box = after["box_sweep"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(ta, xa, "k-", lw=1.2, label=r"$k(r_c+t)$ / $k(r_c-t)$ (analytic)")
    ax[0].plot(tb, fb, "x--", color="0.45", ms=6, label="pre-fix ($closed\\_surface$=True)")
    ax[0].plot(ta, fa, "o", ms=4, color="C0", label="patched")
    ax[0].axvline(50.0, color="0.7", ls=":", lw=1.0)
    ax[0].text(51, 10, r"$t=r_c$", fontsize=8, color="0.4")
    ax[0].set_xlabel("signed depth $t$ [mm]   ($t>0$ inside)")
    ax[0].set_ylabel("penalty force magnitude [kN]")
    ax[0].set_title("capped cylinder hull, $R_{hull}$=0.5 m, $r_c$=50 mm")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].semilogy(tb, [max(e, 1e-16) for e in eb], "x--", color="0.45", ms=6,
                   label="pre-fix, cylinder")
    ax[1].semilogy(ta, [max(e, 1e-16) for e in ea], "o", ms=4, label="patched, cylinder")
    ax[1].semilogy([r["depth_m"] * 1e3 for r in box],
                   [max(r["rel_err"], 1e-16) for r in box], "s-", ms=3,
                   color="C2", label="patched, box (exact mesh)")
    ax[1].axhline(0.03, color="r", ls="--", lw=0.9, label="3 % criterion")
    ax[1].axhline(0.15, color="0.4", ls="--", lw=0.9, label="15 % criterion")
    ax[1].set_xlabel("signed depth $t$ [mm]")
    ax[1].set_ylabel("|F - analytic| / analytic")
    ax[1].set_ylim(1e-16, 3)
    ax[1].set_title("force error vs analytic (log)")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].grid(alpha=0.3, which="both")

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "fig_deep_penetration_compare.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())