"""Figures of the mixed FY/MY campaign (matplotlib, deterministic).

Four figures, each drawn from the artifact only:

``fig1_my_fraction_curve.png``  R(p_MY) for both anchors + strength band, and
                                the relative change at three speeds.
``fig2_fsd_sensitivity.png``    tornado of the one-at-a-time spans, the joint
                                Latin-hypercube envelope, and the FY/MY floe
                                thickness histogram of the p = 0.4 field.
``fig3_ageing.png``             brine volume, strengths, modulus, thickness and
                                toughness along the ageing trajectory.
``fig4_dem_factorial.png``      DEM versus anchor strength exponents and the
                                MY/FY class ratio.

No number is typed here: everything is read from the campaign artifact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["make_all", "fig_my_fraction", "fig_fsd", "fig_ageing", "fig_dem"]

_COLORS = {"lindqvist": "tab:blue", "riska": "tab:orange", "dem": "tab:red"}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def fig_my_fraction(artifact: dict[str, Any], path: Path) -> Path:
    sweep = artifact["my_fraction_sweep"]
    primary = artifact["primary_speed_m_s"]
    band = artifact["my_strength_band"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    ax = axes[0]
    for model in ("lindqvist", "riska"):
        rows = [r for r in sweep["curves"][f"{model}_v{primary:g}"]["rows"]
                if r["my_strength_multiplier"] == 1.0]
        p = [r["my_fraction"] * 100.0 for r in rows]
        r = [r["resistance_mean_N"] / 1e6 for r in rows]
        ax.plot(p, r, "o-", color=_COLORS[model], label=f"{model.capitalize()}")
    # strength-parameterisation band (Lindqvist only: the two anchors would
    # otherwise overlap and the band is a property of the ice mechanics)
    band_rows = {}
    for row in sweep["curves"][f"lindqvist_v{primary:g}"]["rows"]:
        band_rows.setdefault(row["my_fraction"], []).append(row["resistance_mean_N"] / 1e6)
    p = [k * 100.0 for k in band_rows]
    lo = [min(v) for v in band_rows.values()]
    hi = [max(v) for v in band_rows.values()]
    ax.fill_between(p, lo, hi, color=_COLORS["lindqvist"], alpha=0.15,
                    label="Lindqvist, MY strength x%.1f-x%.1f" % (min(band), max(band)))
    ax.set_xlabel("multi-year ice area fraction  p_MY  [%]")
    ax.set_ylabel("ice resistance  R  [MN]")
    ax.set_title(f"R(p_MY), icebreaker, v = {primary:g} m/s")
    ax.legend(loc="upper left")

    ax = axes[1]
    for model in ("lindqvist", "riska"):
        for speed in artifact["speeds_m_s"]:
            rows = [r for r in sweep["curves"][f"{model}_v{speed:g}"]["rows"]
                    if r["my_strength_multiplier"] == 1.0]
            p = [r["my_fraction"] * 100.0 for r in rows]
            d = [100.0 * r["relative_increase"] for r in rows]
            ax.plot(p, d, "o-" if speed == primary else "s--",
                    alpha=1.0 if model == "lindqvist" else 0.55,
                    color=_COLORS[model],
                    label=f"{model.capitalize()}, v = {speed:g} m/s")
    ax.set_xlabel("multi-year ice area fraction  p_MY  [%]")
    ax.set_ylabel("R(p_MY) / R(0) - 1  [%]")
    ax.set_title("relative change of the field resistance")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_fsd(artifact: dict[str, Any], path: Path) -> Path:
    sens = artifact["sensitivity"]
    env = artifact["joint_envelope"]
    hist = artifact["field_histogram_p04"]
    base = sens["base_resistance_N"] / 1e6
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))

    ax = axes[0]
    names = sens["rank_by_span"]
    labels = []
    spans = []
    for name in names:
        row = sens["parameters"][name]
        levels = sorted(v["level"] for v in row["levels"])
        spans.append(row["span_N"] / 1e6)
        labels.append(f"{name}\n[{levels[0]:g}, {levels[-1]:g}] ({row['group']})")
    y = np.arange(len(names))
    ax.barh(y, spans, color=["tab:green" if sens["parameters"][n]["group"] == "fsd"
                             else "tab:purple" for n in names])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("resistance span over the declared range  [MN]")
    ax.set_title(f"one-at-a-time sensitivity, p_MY = {sens['base_my_fraction']:g}, "
                 f"R = {base:.2f} MN")

    ax = axes[1]
    for label, colour in (("fsd", "tab:green"), ("material", "tab:purple")):
        vals = np.array([s["resistance_N"] for s in env[label]["samples"]]) / 1e6
        ax.hist(vals, bins=18, alpha=0.55, color=colour,
                label=f"{label}: p5-p95 = {env[label]['p05_N']/1e6:.2f}-"
                      f"{env[label]['p95_N']/1e6:.2f} MN")
        ax.axvline(env[label]["p05_N"] / 1e6, color=colour, ls=":", lw=1)
        ax.axvline(env[label]["p95_N"] / 1e6, color=colour, ls=":", lw=1)
    ax.set_xlabel("R  [MN]")
    ax.set_ylabel("count")
    ax.set_title("joint Latin-hypercube envelope")
    ax.legend(fontsize=7)

    ax = axes[2]
    edges = np.array(hist["bin_edges_m"])
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    ax.bar(centres, hist["FY"], width=width, alpha=0.6, color="tab:blue", label="FY floes")
    ax.bar(centres, hist["MY"], width=width, alpha=0.6, color="tab:red", label="MY floes")
    ax.set_xlabel("floe thickness  h  [m]")
    ax.set_ylabel("floe count (20 000-floe sample)")
    ax.set_title("bimodal FSD of the p_MY = 0.4 field")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_ageing(artifact: dict[str, Any], path: Path) -> Path:
    table = artifact["ageing_table"]
    age = np.array([row["age_years"] for row in table])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    ax = axes[0]
    for key, label, colour in (
        ("flexural_strength_pa", r"$\sigma_f$ [MPa]", "tab:blue"),
        ("compressive_strength_pa", r"$\sigma_c$ [MPa]", "tab:red"),
    ):
        ax.plot(age, [row[key] / 1e6 for row in table], "o-", color=colour, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("ice age  [a]")
    ax.set_ylabel("strength [MPa]")
    ax.set_title("strength gain with age")
    ax.legend(loc="center right")

    ax2 = ax.twinx()
    ax2.plot(age, [row["elastic_modulus_pa"] / 1e9 for row in table], "s--",
             color="tab:green", label=r"$E$ [GPa]")
    ax2.plot(age, [row["brine_volume_fraction"] * 100.0 for row in table], "^:",
             color="tab:gray", label=r"$v_b$ [%]")
    ax2.grid(False)
    ax2.set_ylabel("E [GPa] / $v_b$ [%]")
    ax2.legend(loc="center left", fontsize=7)

    ax = axes[1]
    ax.plot(age, [row["thickness_m"] for row in table], "o-", color="tab:brown",
            label="class-mean thickness [m]")
    ax.set_xscale("log")
    ax.set_xlabel("ice age  [a]")
    ax.set_ylabel("thickness  [m]")
    ax.set_title("thickness growth and toughness")
    ax2 = ax.twinx()
    ax2.plot(age, [row["fracture_toughness_pa_sqrt_m"] / 1e6 for row in table], "s--",
             color="tab:purple", label=r"$K_{Ic}$ [MPa m$^{1/2}$]")
    ax2.grid(False)
    ax2.set_ylabel(r"$K_{Ic}$ [MPa m$^{1/2}$]")
    ax2.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_dem(artifact: dict[str, Any], path: Path) -> Path:
    dem = artifact["dem"]
    fit = dem["dem_fit"]["exponents"]
    anchor = dem["anchor_fit"]["lindqvist"]["exponents"]
    order = ["compressive_strength_pa", "flexural_strength_pa", "elastic_modulus_pa"]
    labels = [r"$\sigma_c$", r"$\sigma_f$", r"$E$"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9))

    ax = axes[0]
    x = np.arange(len(order))
    ax.bar(x - 0.2, [anchor[k] for k in order], 0.4, label="Lindqvist anchor",
           color="tab:blue")
    ax.bar(x + 0.2, [fit[k] for k in order], 0.4, label="DEM 2$^3$ factorial",
           color=_COLORS["dem"])
    for i, k in enumerate(order):
        ax.annotate(f"{anchor[k]:.3f}", (i - 0.2, anchor[k]), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{fit[k]:.3f}", (i + 0.2, fit[k]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\partial \\ln R / \\partial \\ln {l.strip('$')}$" for l in labels])
    ax.set_ylabel("log-log sensitivity exponent")
    ax.set_title("strength / modulus sensitivity: DEM vs anchor")
    ax.legend()
    ax.axhline(0.0, color="k", lw=0.8)

    ax = axes[1]
    ratio = dem["class_ratio"]
    names = ["DEM", "Lindqvist", "Riska"]
    values = [ratio["dem_full_high_over_low"], ratio["lindqvist_full_high_over_low"],
              ratio["riska_full_high_over_low"]]
    colours = [_COLORS["dem"], _COLORS["lindqvist"], _COLORS["riska"]]
    ax.bar(names, values, color=colours)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.3f}\n(+{100*(v-1):.1f} %)", (i, v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(r"$R_{MY} / R_{FY}$ at h = 1 m")
    ax.set_title("class ratio: bonded-particle section vs engineering anchors")
    ax.set_ylim(0.0, max(values) * 1.22)

    ax = axes[2]
    sweep = dem.get("my_fraction_sweep")
    if sweep:
        p = [100.0 * row["my_fraction"] for row in sweep["rows"]]
        ax.plot(p, [100.0 * row["dem_relative_increase"] for row in sweep["rows"]],
                "o-", color=_COLORS["dem"], label="DEM section (h = 1 m)")
        ax.plot(p, [100.0 * row["lindqvist_relative_increase"] for row in sweep["rows"]],
                "s--", color=_COLORS["lindqvist"], label="Lindqvist (h = 1 m)")
        ax.plot(p, [100.0 * row["riska_relative_increase"] for row in sweep["rows"]],
                "^:", color=_COLORS["riska"], label="Riska (h = 1 m)")
        ax.set_xlabel("multi-year ice area fraction  p_MY  [%]")
        ax.set_ylabel("relative change at fixed h  [%]")
        ax.set_title("DEM-domain MY-fraction sweep (material switching only)")
        ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(artifact: dict[str, Any], outdir: str | Path) -> list[Path]:
    """Render every figure and return the paths."""
    _style()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    made = [
        fig_my_fraction(artifact, out / "fig1_my_fraction_curve.png"),
        fig_fsd(artifact, out / "fig2_fsd_sensitivity.png"),
        fig_ageing(artifact, out / "fig3_ageing.png"),
    ]
    if "skipped" not in artifact["dem"]:
        made.append(fig_dem(artifact, out / "fig4_dem_factorial.png"))
    return made