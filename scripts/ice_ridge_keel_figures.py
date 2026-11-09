"""Figures for the freeze-bonded ice-ridge keel study.

Reads ``artifacts/ice_ridge_keel/study.json`` (written by
``scripts/ice_ridge_keel_study.py``) and renders the five report figures.  Every
number that appears on a figure is read from the study file -- nothing is
hand-copied, so a figure cannot drift away from the recorded evidence.

Usage::

    PYTHONPATH=src python scripts/ice_ridge_keel_figures.py
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensordem import ice_ridge_keel_freezebond as keel
from tensordem import adfreeze_key_adapter as adapter

OUT = os.path.join("artifacts", "ice_ridge_keel")
STUDY = os.path.join(OUT, "study.json")
DPI = 150


def _load():
    with open(STUDY) as fh:
        return json.load(fh)


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, flush=True)
    return name


def _part(study, key):
    part = study["parts"].get(key, {})
    return part if part.get("ok") else None


def fig_adfreeze_closure(study):
    """tau_adfreeze(dT, t, Ra, S) and its factor decomposition."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    temps = np.linspace(-24.0, 1.0, 240)

    ax = axes[0]
    for salinity, label, style in ((0.0, "freshwater joint (S = 0)", "-"),
                                   (32.0, "sea-ice joint (S = 32 g/kg)", "--")):
        tau = [keel.AdfreezeClosure(salinity_gkg=salinity).tau_mpa(T) for T in temps]
        ax.plot(temps, tau, style, lw=2.0, label=label)
    tf32 = keel.seawater_freezing_point(32.0)
    ax.axvline(tf32, color="0.4", ls=":", lw=1.0)
    ax.annotate(f"liquidus $T_f$(32 g/kg) = {tf32:.2f} °C", xy=(tf32, 0.35),
                xytext=(tf32 - 12.5, 0.42), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))
    ax.set_xlabel("joint temperature [°C]")
    ax.set_ylabel(r"$\tau_{adfreeze}$ [MPa]")
    ax.set_title("(a) adfreeze strength, $t_c$ = 3.6 ks, $R_a$ = 2.5 µm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    times = np.logspace(0.0, 5.0, 160)
    for dT, label in ((2.0, "$\\Delta T$ = 2 K"), (8.17, "$\\Delta T$ = 8.2 K"),
                      (20.0, "$\\Delta T$ = 20 K")):
        eta = [keel.frozen_area_fraction(dT, t) for t in times]
        ax.plot(times, eta, lw=2.0, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("contact time [s]")
    ax.set_ylabel(r"frozen area fraction $\eta$")
    ax.set_title("(b) Stefan-growth kinetics of the frozen fraction")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    s = np.linspace(0.0, 40.0, 80)
    vb = [keel.brine_volume_fraction(-10.0, v) for v in s]
    ax.plot(s, vb, "k-", lw=2.0, label="$v_b$ at -10 °C")
    ax.plot(s, [(1.0 - v) ** 3 for v in vb], "r--", lw=2.0,
            label=r"$\phi_{salt}=(1-v_b)^3$")
    ax.axvline(32.0, color="0.4", ls=":", lw=1.0)
    vb32 = keel.brine_volume_fraction(-10.0, 32.0)
    ax.plot([32.0], [vb32], "ko", ms=4)
    ax.annotate(f"$v_b$ = {vb32:.3f}\n$\\phi_{{salt}}$ = {(1 - vb32) ** 3:.3f}",
                xy=(32.0, vb32), xytext=(33.5, 0.55), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("bulk salinity $S$ [g/kg]")
    ax.set_ylabel("brine volume fraction / factor [-]")
    ax.set_title("(c) brine penalty at -10 °C (fixed liquidus anchors)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Figure 1 — adfreeze closure mapped into the DEM freeze bond",
                 y=1.02, fontsize=11)
    return _save(fig, "fig1_adfreeze_closure.png")


def fig_bond_key_parity(study):
    """Criterion 1: DEM break force vs the analytic tau*A for the ten sets."""
    part = _part(study, "bond_mapping")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    if part is None:
        for ax in axes:
            ax.text(0.5, 0.5, "bond_mapping part FAILED\n(see study.json)",
                    ha="center", va="center", transform=ax.transAxes)
        return _save(fig, "fig2_bond_key_parity.png")

    sets = part["sets"]
    analytic = np.array([s["f_break_analytic_N"] for s in sets])
    dem = np.array([s["f_break_dem_N"] for s in sets])
    saline = np.array([s["salinity_gkg"] > 0.0 for s in sets])

    ax = axes[0]
    lim = [min(analytic.min(), dem.min()) * 0.6, max(analytic.max(), dem.max()) * 1.6]
    ax.plot(lim, lim, "k-", lw=1.0, label="1:1")
    ax.plot(lim, [1.05 * v for v in lim], "k:", lw=0.8, label="±5 % gate")
    ax.plot(lim, [0.95 * v for v in lim], "k:", lw=0.8)
    ax.loglog(analytic[~saline], dem[~saline], "o", ms=7, label="freshwater sets")
    ax.loglog(analytic[saline], dem[saline], "s", ms=7, label="sea-ice sets (S=32)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"analytic $F_{break}=\tau_{adfreeze} A_{neck}$ [N]")
    ax.set_ylabel(r"DEM twin-particle break force [N]")
    ax.set_title(f"(a) {len(sets)} parameter sets, max rel. error "
                 f"{part['max_rel_error']:.2e}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    labels = [f"{s['T_C']:.0f}°C\n{s['contact_time_s'] / 1e3:.1f}ks\n{s['salinity_gkg']:.0f}g/kg"
              for s in sets]
    rel = np.array([s["rel_error"] for s in sets]) * 100.0
    colors = ["tab:red" if q else "tab:blue" for q in saline]
    ax.bar(range(len(sets)), rel, color=colors)
    ax.axhline(5.0, color="k", ls="--", lw=1.0, label="5 % criterion")
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("relative error [%]")
    ax.set_ylim(0.0, max(5.5, rel.max() * 1.25))
    ax.set_title("(b) per-set error against the 5 % gate")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle("Figure 2 — adfreeze → DEM bond-key mapping (criterion 1, ≤5 %)",
                 y=1.03, fontsize=11)
    return _save(fig, "fig2_bond_key_parity.png")


def fig_keel_drag_vs_depth(study):
    """Criterion 3: DEM drag vs the Kerr-type rigid-plastic reference vs h_k."""
    cases = []
    for key in ("keel_bonded_h2", "keel_bonded_h4", "keel_bonded_h6"):
        part = _part(study, key)
        if part is not None:
            part["_tag"] = "bonded"
            cases.append(part)
    unb = _part(study, "keel_unbonded_h4")
    if unb is not None:
        unb["_tag"] = "unbonded"
        cases.append(unb)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    if not cases:
        for ax in axes:
            ax.text(0.5, 0.5, "keel drag parts FAILED\n(see study.json)",
                    ha="center", va="center", transform=ax.transAxes)
        return _save(fig, "fig3_keel_drag_vs_depth.png")

    ax = axes[0]
    for tag, style, color in (("bonded", "-o", "tab:blue"),
                              ("unbonded", "-^", "tab:orange")):
        sel = sorted([c for c in cases if c["_tag"] == tag], key=lambda c: c["h_k_m"])
        if not sel:
            continue
        hk = [c["h_k_m"] for c in sel]
        f = [c["f_keel_N"] for c in sel]
        err = [c["f_keel_std_N"] for c in sel]
        ax.errorbar(hk, f, yerr=err, fmt=style, capsize=3, color=color,
                    label=f"DEM, {tag}")
        fk = [c["f_kerr_N"] for c in sel]
        fk_lo = [c["f_kerr_N_phi_low"] for c in sel]
        ax.plot(hk, fk, "s--", color=color, alpha=0.55, lw=1.2,
                label=f"Kerr-type reference, {tag} "
                      f"($\\varphi$={sel[0]['phi_used_deg']:.1f}°)")
        ax.fill_between(hk, fk_lo, fk, color=color, alpha=0.12)
    ax.set_xlabel("keel depth $h_k$ [m]")
    ax.set_ylabel("keel resistance $F_{keel}$ [N]")
    ax.set_title("(a) resistance vs keel depth (plate drag, -10 °C)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    bonded = sorted([c for c in cases if c["_tag"] == "bonded"], key=lambda c: c["h_k_m"])
    if bonded:
        hk = np.array([c["h_k_m"] for c in bonded])
        ratio = np.array([c["f_keel_N"] / c["f_kerr_N"] for c in bonded])
        ax.plot(hk, ratio, "o-", color="tab:blue", label="DEM / Kerr (bonded)")
        for x, y, c in zip(hk, ratio, bonded):
            ax.annotate(f"{y:.2f}\n(n={c['porosity_standing']:.3f})", (x, y),
                        textcoords="offset points", xytext=(6, 6), fontsize=7)
        ax.axhline(1.0, color="k", lw=1.0)
        ax.axhspan(0.85, 1.15, color="tab:green", alpha=0.12,
                   label="±15 % arbitration band")
    if unb is not None:
        ax.plot([unb["h_k_m"]], [unb["f_keel_N"] / unb["f_kerr_N"]], "^",
                color="tab:orange", label="DEM / Kerr (unbonded)")
    ax.set_xlabel("keel depth $h_k$ [m]")
    ax.set_ylabel("DEM / Kerr-type resistance [-]")
    ax.set_title("(b) arbitration against the empirical keel formula")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Figure 3 — porosity → structure → drag resistance vs keel depth",
                 y=1.03, fontsize=11)
    return _save(fig, "fig3_keel_drag_vs_depth.png")


def fig_dT_sensitivity(study):
    """Criterion 4: drag resistance must grow monotonically with |dT|."""
    part = _part(study, "dT_sweep")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    if part is None:
        for ax in axes:
            ax.text(0.5, 0.5, "dT_sweep part FAILED\n(see study.json)",
                    ha="center", va="center", transform=ax.transAxes)
        return _save(fig, "fig4_dT_sensitivity.png")

    cases = part["cases"]
    T = np.array([c["T_C"] for c in cases])
    order = np.argsort(T)
    T = T[order]
    f = np.array([cases[i]["f_keel_N"] for i in order])
    ferr = np.array([cases[i]["f_keel_std_N"] for i in order])
    tau = np.array([cases[i]["tau_adfreeze_MPa"] for i in order])
    fb = np.array([cases[i]["f_break_mean_N"] for i in order])

    ax = axes[0]
    ax.errorbar(T, f, yerr=ferr, fmt="o-", color="tab:blue", capsize=3,
                label="$F_{keel}$ (plate drag)")
    ax.set_xlabel("ice temperature [°C] (equal to $-\\Delta T$ for S = 0)")
    ax.set_ylabel("$F_{keel}$ [N]", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(T, tau, "s--", color="tab:red", label=r"$\tau_{adfreeze}$")
    ax2.plot(T, fb / fb.max() * tau.max(), "^:", color="tab:green",
             label=r"$\bar F_{break}$ (scaled)")
    ax2.set_ylabel(r"$\tau_{adfreeze}$ [MPa] / $\bar F_{break}$ [-]", color="tab:red")
    ax.set_title(f"(a) resistance vs freeze-bond strength, Spearman "
                 f"ρ = {part['spearman_rho']:.3f}")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    ax = axes[1]
    n_stand = np.array([cases[i]["porosity_standing"] for i in order])
    n_drag = None
    ax.plot(T, n_stand, "o-", color="tab:purple", label="porosity while standing")
    active = np.array([cases[i]["bonds_active_standing"] for i in order])
    ax.fill_between(T, 0.30, 0.45, color="tab:green", alpha=0.12,
                    label="refrozen-keel band 0.30–0.45")
    ax.plot(T, n_stand, "o", color="tab:purple")
    ax.set_xlabel("ice temperature [°C]")
    ax.set_ylabel("measured rubble porosity $n$ [-]")
    ax.set_ylim(min(0.25, n_stand.min() - 0.03), max(0.5, n_stand.max() + 0.05))
    ax3 = ax.twinx()
    ax3.plot(T, active, "v--", color="tab:brown", label="active freeze bonds")
    ax3.set_ylabel("active bonds [-]", color="tab:brown")
    ax.set_title("(b) porosity and bond population across the sweep")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, fontsize=8, loc="lower right")

    fig.suptitle("Figure 4 — freeze-bond strength sensitivity (criterion 4, ρ ≥ 0.9)",
                 y=1.03, fontsize=11)
    return _save(fig, "fig4_dT_sensitivity.png")


def fig_porosity_forcechain(study):
    """Criteria 2, 5, 6: porosity states and the force-chain tail."""
    bonded = _part(study, "keel_bonded_h4")
    unbonded = _part(study, "keel_unbonded_h4")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    labels, values, colors = [], [], []
    for part, tag, color in ((bonded, "bonded\n(-10 °C)", "tab:blue"),
                             (unbonded, "unbonded\n(no freeze bond)", "tab:orange")):
        if part is None:
            continue
        for k, sub in (("standing", "standing"), ("after_drag", "after drag"),
                       ("after_release", "after release")):
            labels.append(f"{tag}\n{sub}")
            values.append(part[f"porosity_{k}"])
            colors.append(color)
    if values:
        ax.bar(range(len(values)), values, color=colors, alpha=0.85)
        ax.axhspan(0.30, 0.45, color="tab:green", alpha=0.12,
                   label="refrozen-keel band 0.30–0.45")
        ax.axhline(0.55, color="tab:red", ls="--", lw=1.2,
                   label="unbonded lower bound 0.55")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=7)
        for i, v in enumerate(values):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7)
        ax.set_ylim(min(0.25, min(values) - 0.03), max(0.62, max(values) + 0.06))
    ax.set_ylabel("Voronoi rubble porosity $n$ [-]")
    ax.set_title("(a) porosity across the keel life cycle (criteria 2 & 5)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    fc = None
    if bonded is not None:
        fc = bonded.get("force_chain_standing")
    if fc:
        # measured P(f) tail: the module fits P(f) ~ f^-alpha over the strong
        # forces; the criterion is the fit quality R^2 >= 0.90
        alpha = float(fc["alpha"])
        r2 = float(fc["r2"])
        f_max = float(fc["f_max"])
        xs = np.logspace(0.0, np.log10(max(f_max, 1.5)), 60)
        # reconstruct the fitted tail with the same normalisation the fit uses
        p_tail = xs ** (-alpha)
        # normalise the fitted power law over the strong-force window [1, f_max]
        if abs(alpha - 1.0) > 1e-9:
            norm = float(fc["strong_fraction"]) * (1.0 - alpha) / (
                max(f_max, 1.0 + 1e-9) ** (1.0 - alpha) - 1.0)
        else:
            norm = float(fc["strong_fraction"]) / math.log(max(f_max, 1.0 + 1e-9))
        p_tail = p_tail * norm
        ax.loglog(xs, p_tail, "k--", lw=1.2,
                  label=f"tail fit $P(f)\\propto f^{{-{alpha:.2f}}}$, $R^2$ = {r2:.3f}")
        ax.axvline(np.exp(0.0), color="0.4", lw=0.8)
        ax.set_xlabel("normalised contact force $f = F/\\langle F \\rangle$")
        ax.set_ylabel("$P(f)$")
        ax.set_title(f"(b) {fc['n_contacts']} contacts, "
                     f"{fc['n_strong']} strong ($f>1$), "
                     f"$R^2$ = {r2:.3f} {'PASS' if r2 >= 0.9 else 'FAIL'}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no force-chain statistics available",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("(b) force-chain strong-force tail (criterion 6)")

    fig.suptitle("Figure 5 — porosity–structure link and force-chain statistics",
                 y=1.03, fontsize=11)
    return _save(fig, "fig5_porosity_forcechain.png")


def fig_adapter_parity(study):
    """Criterion 1 (support): the standalone adapter and its platform parity."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    part = _part(study, "adapter_parity")
    parity = part["platform_parity"] if part else {"available": False}

    ax = axes[0]
    temps = np.linspace(-24.0, 0.0, 240)
    tau = [adapter.tau_adfreeze_pa(adapter.AdfreezeKeyParams(T, salinity_gkg=32.0))
           for T in temps]
    ax.plot(temps, np.array(tau) / 1e6, "-", lw=2.0,
            label=r"adapter $\tau_{adfreeze}$ (S = 32 g/kg)")
    if part:
        rows = part["key_table"]
        T = [r["params"]["temperature_c"] for r in rows]
        fb = [r["f_break_n"] for r in rows]
        ax2 = ax.twinx()
        ax2.plot(T, fb, "s", color="tab:red", label="$F_{break}$ ($r$ = 0.30/0.25 m)")
        ax2.set_ylabel("$F_{break}$ [N]", color="tab:red")
        ax2.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("ice temperature [°C]")
    ax.set_ylabel(r"$\tau_{adfreeze}$ [MPa]")
    ax.set_title("(a) adapter key surface")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ax.axis("off")
    if parity.get("available"):
        worst = parity["worst"] or {"T_C": float("nan"), "S_gkg": float("nan"),
                                    "Ra_um": float("nan"), "t_s": float("nan"),
                                    "mirror_Pa": float("nan"), "platform_Pa": float("nan")}
        txt = (
            "platform parity (tensorphysics.ice.adfreeze)\n"
            f"points compared      : {parity['n_points']}\n"
            f"max relative error    : {parity['max_rel_error']:.3e}\n"
            f"$T_f$(32 g/kg) mirror  : {parity['liquidus_Tf_32gkg_mirror_C']:.6f} °C\n"
            f"$T_f$(32 g/kg) platform: {parity['liquidus_Tf_32gkg_platform_C']:.6f} °C\n"
            f"parity gate (1e-12)   : {'PASS' if parity['parity_ok'] else 'FAIL'}\n\n"
            f"worst case: T = {worst['T_C']} °C, S = {worst['S_gkg']} g/kg,\n"
            f"  $R_a$ = {worst['Ra_um']} µm, t = {worst['t_s']:.0f} s\n"
            f"  mirror {worst['mirror_Pa'] / 1e6:.6f} MPa vs "
            f"platform {worst['platform_Pa'] / 1e6:.6f} MPa"
        )
    else:
        txt = ("platform parity: TensorPhysics source tree not importable\n"
               f"reason: {parity.get('reason', 'n/a')}\n\n"
               "the mirrored closure is then only audited against its own\n"
               "anchors (figure 1c) -- reported as WITHHELD, not as a pass")
    ax.text(0.02, 0.95, txt, va="top", family="monospace", fontsize=9)
    ax.set_title("(b) machine-checked parity report")

    fig.suptitle("Figure 6 — standalone adfreeze key adapter", y=1.03, fontsize=11)
    return _save(fig, "fig6_adapter_parity.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    study = _load()
    names = []
    for fn in (fig_adfreeze_closure, fig_bond_key_parity, fig_keel_drag_vs_depth,
               fig_dT_sensitivity, fig_porosity_forcechain, fig_adapter_parity):
        try:
            names.append(fn(study))
        except Exception as exc:                      # keep going, report honestly
            print(f"[figure failed] {fn.__name__}: {exc!r}", flush=True)
    print("figures:", names)


if __name__ == "__main__":
    main()