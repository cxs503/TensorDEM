"""Figures for the P40W4-N06 campaign report (reads ``runs/`` only)."""
from __future__ import annotations

import json
import os

import numpy as np

OUT = os.path.join("artifacts", "p40w4_n06")


def _load():
    summary = json.load(open(os.path.join("runs", "p40w4_n06", "summary.json")))
    cases = {}
    cdir = os.path.join("runs", "p40w4_n06", "cases")
    for name in os.listdir(cdir):
        if name.endswith(".json"):
            data = json.load(open(os.path.join(cdir, name)))
            cases[data["id"]] = data
    return summary, cases


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    summary, cases = _load()
    made = []

    # ---- 1 runout scaling -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for state, colour in (("dense", "C0"), ("loose", "C1")):
        d = summary["runout_series"][state]
        a = np.asarray(d["aspect"]); r = np.asarray(d["runout"])
        axes[0].loglog(a, r, "o-", color=colour, label=f"{state} DEM")
        axes[1].plot(a, r, "o-", color=colour, label=f"{state} DEM")
    if summary["scaling"].get("large_a"):
        lf = summary["scaling"]["large_a"]
        a_ref = np.linspace(0.5, 12, 100)
        axes[0].loglog(a_ref, lf["prefactor"] * a_ref ** (2.0 / 3.0), "k--",
                       label=r"Lube 2004 slope $a^{2/3}$")
        axes[0].loglog(a_ref, lf["prefactor"] * a_ref ** lf["exponent"], "r:",
                       label=f"fit slope {lf['exponent']:.3f}")
    if summary["scaling"].get("small_a"):
        sf = summary["scaling"]["small_a"]
        axes[1].plot(np.asarray(np.linspace(0.4, 1.2, 20)),
                     sf["prefactor"] * np.linspace(0.4, 1.2, 20) ** sf["exponent"],
                     "k--", label=f"small-a slope {sf['exponent']:.3f}")
    for ax in axes:
        ax.set_xlabel("initial aspect ratio a = H0/L0")
        ax.set_ylabel(r"runout $\Delta L / L_0$")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("runout scaling (log-log)")
    axes[1].set_title("runout vs aspect ratio")
    fig.tight_layout()
    path = os.path.join(OUT, "runout_scaling.png")
    fig.savefig(path, dpi=140)
    made.append(path)

    # ---- 2 energy ledger --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ledger = summary.get("energy_ledger", {})
    names = list(ledger)
    if names:
        dts = np.array([ledger[n]["dt"] for n in names])
        res = np.array([abs(ledger[n]["residual"]) for n in names])
        res_phys = np.array([abs(ledger[n]["residual_physical_only"]) for n in names])
        axes[0].loglog(dts, res, "o-", label="ledger residual (incl. cutoff channel)")
        axes[0].loglog(dts, res_phys, "s--", label="physical channels only")
        axes[0].axhline(1e-3, color="k", ls=":", label="criterion 1e-3")
        axes[0].set_xlabel("time step dt [s]")
        axes[0].set_ylabel(r"$|residual| / E_{p0}$")
        axes[0].grid(alpha=0.3, which="both")
        axes[0].legend(fontsize=8)
        axes[0].set_title("energy ledger closure vs dt")
    case = cases.get("col_dense_a2") or cases.get("ledger_a2_dt0.5")
    if case:
        hist = case["history"]
        t = np.asarray(hist["time"])
        e0 = case["energy"]["potential_initial"]
        axes[1].plot(t, np.asarray(hist["potential"]) / e0, label="PE/PE0")
        axes[1].plot(t, np.asarray(hist["kinetic"]) / e0, label="KE/PE0")
        axes[1].plot(t, np.asarray(hist["dissipation"]) / e0, label="dissipated/PE0")
        axes[1].set_xlabel("t [s]")
        axes[1].set_ylabel("energy / initial PE")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)
        axes[1].set_title(f"energy budget, a={case['aspect_measured']:.2f}")
    fig.tight_layout()
    path = os.path.join(OUT, "energy_ledger.png")
    fig.savefig(path, dpi=140)
    made.append(path)

    # ---- 3 mu(I) ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    mu = summary.get("mu_i")
    if mu:
        pts = mu["points"]
        i_m = np.array([p["I_measured"] for p in pts])
        mu_m = np.array([p["mu_measured"] for p in pts])
        ax.semilogx(i_m, mu_m, "o", label="DEM servo shear")
        grid = np.logspace(-4, 0.5, 200)
        f = mu["fit"]
        ax.semilogx(grid, f["mu_s"] + (f["mu_2"] - f["mu_s"]) / (1 + f["i_0"] / grid),
                    "-", label=f"fit $\\mu_s$={f['mu_s']:.2f}, $\\mu_2$={f['mu_2']:.2f}, $I_0$={f['i_0']:.2f}")
        ax.semilogx(grid, 0.36 + (0.84 - 0.36) / (1 + 0.30 / grid), "k--",
                    label="Jop 2006 (0.36/0.84/0.30)")
        fn = mu["fit_negative"]
        ax.semilogx(grid, fn["mu_s"] + (0.0 - fn["mu_s"]) / (1 + fn["i_0"] / grid), ":",
                    color="C3", label="negative control $\\mu_2=0$")
    ax.set_xlabel("inertial number I")
    ax.set_ylabel(r"$\mu = \tau/P$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax.set_title(r"$\mu(I)$ calibration")
    fig.tight_layout()
    path = os.path.join(OUT, "mu_i_rheology.png")
    fig.savefig(path, dpi=140)
    made.append(path)

    # ---- 4 front kinematics ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    d = summary["runout_series"]["dense"]
    h = np.asarray(d["height"]); v = np.asarray(d["peak_speed"])
    axes[0].loglog(h, v, "o-", label="peak front speed")
    if summary.get("front_speed_scaling"):
        fs = summary["front_speed_scaling"]
        hh = np.linspace(h.min(), h.max(), 50)
        axes[0].loglog(hh, fs["prefactor"] * hh ** fs["exponent"], "k--",
                       label=f"fit slope {fs['exponent']:.3f}")
    axes[0].set_xlabel("H0 [m]"); axes[0].set_ylabel("peak front speed [m/s]")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=8)
    axes[0].set_title("front speed vs column height (sqrt(gH0) = slope 0.5)")
    for cid, colour in (("col_dense_a0.5", "C0"), ("col_dense_a2", "C1"), ("col_dense_a10", "C2")):
        if cid in cases:
            hist = cases[cid]["history"]
            t = np.asarray(hist["time"]); fr = np.asarray(hist["front"]) - cases[cid]["front_initial"]
            axes[1].plot(t / np.sqrt(cases[cid]["height_measured"] / 9.81),
                         fr / cases[cid]["length_measured"], color=colour,
                         label=f"a={cases[cid]['aspect_measured']:.1f}")
    axes[1].set_xlabel(r"$t / \sqrt{H_0/g}$"); axes[1].set_ylabel(r"$\Delta L/L_0$")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    axes[1].set_title("rescaled front kinematics")
    fig.tight_layout()
    path = os.path.join(OUT, "kinematics.png")
    fig.savefig(path, dpi=140)
    made.append(path)

    # ---- 5 deposit snapshots ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, cid in zip(axes, ("col_dense_a1", "col_dense_a10")):
        case = cases.get(cid)
        if not case:
            continue
        pos = np.asarray(case.get("positions", []))
        if pos.size:
            ax.plot(pos[:, 0] * 1e2, pos[:, 2] * 1e2, ".", ms=2)
        ax.set_xlabel("x [cm]"); ax.set_ylabel("z [cm]")
        ax.set_title(f"final deposit, a={case['aspect_measured']:.1f}, N={case['n_particles']}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "deposit_snapshots.png")
    fig.savefig(path, dpi=140)
    made.append(path)

    print("\n".join(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
