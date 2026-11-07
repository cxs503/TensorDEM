"""Figures for the ice-contact study (IN27-S2).

Every figure is drawn from the study dictionary produced by
:func:`tensordem.ice_contact.study.run_study`; no number is typed by hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["render_all", "render_sobol"]


def _first(study: dict, speed: float = None, thickness: float = None) -> dict:
    for c in study["conditions"]:
        if speed is not None and abs(c["speed"] - speed) > 1e-9:
            continue
        if thickness is not None and abs(c["thickness"] - thickness) > 1e-9:
            continue
        return c
    return study["conditions"][0]


def render_all(study: dict, out: Path) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.append(_pressure_maps(study, out))
    paths.append(_correlation_and_peak(study, out))
    paths.append(_proxy_bias(study, out))
    return [p for p in paths if p is not None]


def render_sobol(study: dict, out: Path) -> Path | None:
    """Sobol-indices figure; separated because the GSA runs after the study."""
    return _sobol(study, Path(out))


def _pressure_maps(study: dict, out: Path) -> Path | None:
    cond = _first(study, thickness=1.0)
    if cond.get("arrays") is None:
        return None
    poly = cond["arrays"]["poly_mean_field"]
    sphere = cond["arrays"]["sphere_mean_field"]
    sigma_c = cond["polyhedron"]["sim"]["sigma_c"]
    dy = cond["polyhedron"]["sim"]["grid"][0]
    dz = cond["polyhedron"]["sim"]["grid"][1]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    vmax = max(poly.max(), sphere.max()) / sigma_c
    for ax, field, title in (
        (axes[0], poly, "polyhedron (truth)"),
        (axes[1], sphere, "sphere + $\\mu_r$ = %0.2f" % cond["sphere"]["biases"][2]["mu_r"]),
    ):
        im = ax.imshow(
            field / sigma_c, origin="lower", aspect="auto",
            extent=[0, field.shape[1] * dy, 0, field.shape[0] * dz],
            cmap="magma", vmin=0.0, vmax=vmax,
        )
        ax.set_xlabel("y along structure [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="$\\bar p/\\sigma_c$")
    ax = axes[2]
    for label, field, color in (("poly", poly, "C0"), ("sphere", sphere, "C1")):
        profile = field.mean(axis=1) / sigma_c
        ax.plot(np.arange(len(profile)) * dz, profile, "o-", ms=3, color=color, label=label)
    ax.axhline(1.0, color="0.4", ls=":", lw=1, label="$\\sigma_c$")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("$\\bar p(z)/\\sigma_c$ (height average)")
    ax.set_title("vertical profile")
    ax.legend(fontsize=7)
    fig.suptitle(
        "Contact pressure field on the structure, v = %.1f m/s, $h_{ice}$ = %.1f m"
        % (cond["speed"], cond["thickness"]), fontsize=10,
    )
    path = out / "fig1_pressure_field.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _correlation_and_peak(study: dict, out: Path) -> Path | None:
    conds = study["conditions"]
    if conds[0].get("arrays") is None:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    ax = axes[0]
    for cond in conds:
        radii = np.asarray(cond["arrays"]["radii"])
        profile = np.asarray(cond["arrays"]["radial_profile"])
        if radii.size == 0:
            continue
        ax.plot(radii, profile, lw=1.0,
                label="v=%.0f h=%.1f" % (cond["speed"], cond["thickness"]))
    ax.axhline(np.exp(-1.0), color="k", ls="--", lw=1)
    ax.text(0.02, np.exp(-1.0) + 0.02, "$1/e$", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xlabel("separation $r$ [m]")
    ax.set_ylabel("$R(r)$")
    ax.set_title("spatial autocorrelation of $\\bar p(y,z)$")
    ax.legend(fontsize=6, ncol=2)

    speeds = sorted({c["speed"] for c in conds})
    thicknesses = sorted({c["thickness"] for c in conds})
    ax = axes[1]
    for h in thicknesses:
        sel = [c for c in conds if c["thickness"] == h]
        lc = [c["polyhedron"]["statistics"]["L_c"] for c in sel]
        ax.plot([c["speed"] for c in sel], lc, "o-", label="h=%.1f m" % h)
    ax.axhspan(0.5, 2.0, color="C2", alpha=0.12, label="acceptance band")
    ax.set_xlabel("indentation speed v [m/s]")
    ax.set_ylabel("$L_c$ [m]")
    ax.set_title("correlation length")
    ax.legend(fontsize=7)

    ax = axes[2]
    for h in thicknesses:
        sel = [c for c in conds if c["thickness"] == h]
        ax.plot([c["speed"] for c in sel],
                [c["polyhedron"]["statistics"]["gamma_spatial"] for c in sel],
                "o-", label="h=%.1f $\\gamma_{spatial}$" % h)
        ax.plot([c["speed"] for c in sel],
                [c["polyhedron"]["statistics"]["gamma_instant"] for c in sel],
                "s--", mfc="none", label="h=%.1f $\\gamma_{instant}$" % h)
    ax.axhspan(2.0, 5.0, color="C2", alpha=0.12, label="acceptance band")
    ax.set_xlabel("indentation speed v [m/s]")
    ax.set_ylabel("peak factor $\\gamma$")
    ax.set_title("peak factor")
    ax.legend(fontsize=6, ncol=2)
    path = out / "fig2_correlation_peakfactor.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _proxy_bias(study: dict, out: Path) -> Path | None:
    conds = study["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    ax = axes[0]
    cond = _first(study, thickness=1.0)
    if cond.get("arrays") is not None:
        time = np.asarray(cond["arrays"]["time"])
        poly = np.asarray(cond["arrays"]["poly_force"]) / 1e6
        ax.plot(time, poly, color="C0", lw=1.2, label="polyhedron")
        for key in list(cond["arrays"]["sphere_force"])[:3]:
            force = np.asarray(cond["arrays"]["sphere_force"][key]) / 1e6
            ax.plot(time, force, lw=0.9, ls="--", label="$\\mu_r$ = " + key)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("normal force on structure [MN]")
    ax.set_title("force history, v=%.0f m/s, h=%.1f m" % (cond["speed"], cond["thickness"]))
    ax.legend(fontsize=7)

    ax = axes[1]
    for cond in conds:
        mu = [b["mu_r"] for b in cond["sphere"]["biases"]]
        delta = [100.0 * b["delta_mean"] for b in cond["sphere"]["biases"]]
        ax.plot(mu, delta, "o-", lw=1.0,
                label="v=%.0f h=%.1f" % (cond["speed"], cond["thickness"]))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-10.0, 10.0, color="C2", alpha=0.12, label="$\\pm$10 % gate")
    geo = np.nanmean([c["mu_r_geometric"]["value"] for c in conds])
    if np.isfinite(geo):
        ax.axvline(geo, color="0.3", ls=":", label="mean $\\mu_{geo}$ = %.2f" % geo)
    ax.set_xlabel("rolling-friction coefficient $\\mu_r$")
    ax.set_ylabel("$\\delta = (F_{poly}-F_{sphere})/F_{poly}$ [%]")
    ax.set_title("proxy bias of the sphere + $\\mu_r$ model")
    ax.legend(fontsize=6, ncol=2)
    path = out / "fig3_rolling_proxy_bias.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _sobol(study: dict, out: Path) -> Path | None:
    gsa = study.get("gsa")
    if not gsa:
        return None
    names = gsa["names"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.bar(x - 0.2, gsa["S1"], width=0.4, label="$S_1$ (first order)")
    ax.bar(x + 0.2, gsa["ST"], width=0.4, label="$S_T$ (total)")
    ax.set_xticks(x)
    ax.set_xticklabels(["$\\mu_r$", "v", "$h_{ice}$", "angularity", "$\\mu_{pp}$"])
    ax.set_ylabel("Sobol index of $|\\delta|$")
    ax.set_title(
        "global sensitivity of the proxy bias (Saltelli, N=%d, %d runs)"
        % (gsa["n_base"], gsa["evaluations"]), fontsize=9
    )
    ax.legend()
    path = out / "fig4_sobol.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path