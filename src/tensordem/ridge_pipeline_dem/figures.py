"""Figures of the p54w3-n06 campaign report.

Every figure is generated from ``artifacts/p54w3_n06/campaign.json`` - the same
file the report's numbers come from - so a figure can never drift away from the
adjudicated numbers.  ``matplotlib`` is an optional dependency (``pip install
.[viz]``); the module raises a clear error if it is missing.
"""
from __future__ import annotations

import math
import os

try:  # pragma: no cover - optional dependency
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("figures need matplotlib: pip install '.[viz]'") from exc

import json

__all__ = ["make_all"]

VERTICAL_CASE = "v_cover0.50_su30"


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_displacement(campaign: dict, path: str) -> str:
    """DEM load-displacement curves of the vertical cases vs ISO capacity lines."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    payloads = campaign["case_payloads"]
    for name, payload in payloads.items():
        if not name.startswith("v_"):
            continue
        curve = payload["curves"]
        ax.plot(curve["keel_penetration"], [v / 1.0e3 for v in curve["pipe_load"]],
                lw=1.0, label=f"DEM {name}")
        iso = payload["comparison"]["vertical"]["iso_n_per_m"] / 1.0e3
        ax.axhline(iso, ls="--", lw=0.9, color=ax.lines[-1].get_color(),
                   label=f"ISO form {name}: {iso:.1f} kN/m")
    ax.set_xlabel("keel penetration (m)")
    ax.set_ylabel("ice load on pipe (kN/m)")
    ax.set_title("Ridge-keel grounding: DEM load-displacement vs ISO 19906-form capacity")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def dem_vs_iso(campaign: dict, path: str) -> str:
    """Grouped bars: DEM load and theoretical capacity per case."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    vertical = campaign["results"]["vertical"]["rows"]
    lateral = campaign["results"]["lateral"]["rows"]
    ax = axes[0]
    names = [row["case"] for row in vertical]
    dem = [row["dem_n_per_m"] / 1.0e3 for row in vertical]
    iso = [row["iso_n_per_m"] / 1.0e3 for row in vertical]
    index = range(len(names))
    ax.bar([i - 0.2 for i in index], dem, width=0.4, label="DEM peak")
    ax.bar([i + 0.2 for i in index], iso, width=0.4, label="ISO 19906 form")
    for i, row in enumerate(vertical):
        ax.annotate(f"{100 * row['deviation']:+.1f}%", (i, max(dem[i], iso[i]) * 1.02),
                    ha="center", fontsize=7)
    ax.set_xticks(list(index))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("vertical load (kN/m)")
    ax.set_title("(a) vertical: DEM vs ISO form")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    names = [row["case"] for row in lateral]
    dem = [row["dem_sustained_n_per_m"] / 1.0e3 for row in lateral]
    iso = [row["iso_n_per_m"] / 1.0e3 for row in lateral]
    index = range(len(names))
    ax.bar([i - 0.2 for i in index], dem, width=0.4, label="DEM sustained")
    ax.bar([i + 0.2 for i in index], iso, width=0.4, label="$N_p s_u D$")
    for i, row in enumerate(lateral):
        ax.annotate(f"{100 * row['deviation_sustained']:+.1f}%",
                    (i, max(dem[i], iso[i]) * 1.02), ha="center", fontsize=7)
    ax.set_xticks(list(index))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("lateral load (kN/m)")
    ax.set_title("(b) lateral break-out: DEM vs $N_p$ form")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def energy_closure(campaign: dict, path: str) -> str:
    """Closure error vs time step, whole record and loading phase."""
    rows = campaign["results"]["energy"]["rows"]
    dt = [row["dt"] for row in rows]
    whole = [row["whole_record"]["closure_error"] for row in rows]
    loading = [row["loading_phase"]["closure_error"] for row in rows]
    legacy = [row["legacy_whole_record"]["closure_error"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    ax.loglog(dt, whole, "o-", label="whole record (corrected partition)")
    ax.loglog(dt, legacy, "s--", label="whole record (legacy partition)")
    ax.loglog(dt, loading, "^-", label="loading phase (corrected partition)")
    ax.axhline(campaign["thresholds"]["energy_closure"], color="k", ls=":",
               label="5 % acceptance line")
    ax.set_xlabel("time step dt (s)")
    ax.set_ylabel("closure error $|W_{in}-W_{out}|/\\max$")
    ax.set_title("(a) ledger closure vs dt")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    finest = min(rows, key=lambda row: row["dt"])
    sinks = finest["loading_phase"]["sinks"]
    keys = ["dissipated_dem", "absorbed_pipe", "absorbed_soil", "absorbed_plane",
            "bond_storage", "contact_storage", "kinetic"]
    values = [sinks[key] / 1.0e3 for key in keys]
    total = max(sinks["total"] / 1.0e3, 1e-9)
    ax.barh(keys, [100.0 * v / total for v in values], label="corrected")
    ax.barh(keys, [100.0 * sinks["soil_storage_double_count"] / total],
            left=[100.0 * v / total for v in values], color="grey",
            label="legacy extra term")
    ax.set_xlabel("share of the sink budget (%)")
    ax.set_title(f"(b) dt={finest['dt']:.2e} s sinks")
    ax.legend(fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def grid_refinement(campaign: dict, path: str) -> str:
    """Peak load vs particle count under refinement, with the 5 % band."""
    rows = campaign["results"]["grid"]["rows"]
    particles = [row["particles"] for row in rows]
    peak = [row["peak_n_per_m"] / 1.0e3 for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogx(particles, peak, "o-", label="DEM peak load")
    finest = peak[0]
    ax.axhline(finest, ls="--", lw=0.8, color="k")
    ax.fill_between([min(particles) * 0.8, max(particles) * 1.2],
                    finest * 0.95, finest * 1.05, color="k", alpha=0.12,
                    label="5 % acceptance band")
    for row in rows:
        ax.annotate(f"r={row['particle_radius']:.2f} m", (row["particles"],
                    row["peak_n_per_m"] / 1.0e3), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("particles in the keel section")
    ax.set_ylabel("peak load (kN/m)")
    ax.set_title("Grid refinement: peak load vs particle count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return _save(fig, path)


def cover_envelope(campaign: dict, path: str) -> str:
    """Pipe settlement vs cover depth with observed and predicted instability."""
    rows = campaign["results"]["cover"]["rows"]
    limits = campaign["results"]["cover"]["limits"]
    covers = [row["cover_depth_m"] for row in rows]
    ratio = [row["vertical_displacement_over_limit"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(covers, ratio, "o-", label="DEM settlement / (D/10)")
    ax.axhline(1.0, color="k", ls=":", label="instability limit")
    for row in rows:
        marker = "x" if row["dem_unstable"] else "o"
        marker_pred = row["iso_predicted_unstable"]
        ax.plot(row["cover_depth_m"], row["vertical_displacement_over_limit"],
                marker=marker, ms=11, mfc="none",
                color="crimson" if marker_pred else "seagreen")
        ax.annotate(f"DEM {'unstable' if row['dem_unstable'] else 'stable'} / "
                    f"ISO {'unstable' if marker_pred else 'stable'}",
                    (row["cover_depth_m"], row["vertical_displacement_over_limit"]),
                    fontsize=6, xytext=(6, -10), textcoords="offset points")
    ax.set_xlabel("cover depth (m)")
    ax.set_ylabel("settlement / limit")
    ax.set_title("Cover-instability consistency (marker colour = theoretical verdict)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def field_snapshot(campaign: dict, path: str) -> str:
    """DEM field cloud: contact-force magnitude on the disks and the pipe."""
    payload = campaign["case_payloads"].get(VERTICAL_CASE)
    if not payload or not payload.get("snapshot"):
        return ""
    snapshot = payload["snapshot"]
    positions = snapshot["positions"]
    magnitude = snapshot["force_magnitude"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    scatter = ax.scatter([p[0] for p in positions], [p[1] for p in positions],
                         c=[m / 1.0e3 for m in magnitude], s=48, cmap="inferno",
                         edgecolor="k", linewidth=0.3)
    fig.colorbar(scatter, ax=ax, label="contact-force magnitude (kN)")
    cover = payload["case"]["cover_depth"]
    radius = 0.5 * 0.914
    circle = plt.Circle((0.0, -(cover + radius)), radius, fill=False,
                        color="deepskyblue", lw=1.6, label="pipeline")
    ax.add_patch(circle)
    ax.axhline(0.0, color="tan", lw=1.2, ls="--", label="mudline")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"DEM field cloud at t={snapshot['time']:.2f} s "
                 f"({payload['case']['name']})")
    ax.legend(fontsize=8, loc="upper right")
    return _save(fig, path)


def make_all(campaign: dict, directory: str) -> dict:
    """Write every figure of the report; return name -> path."""
    os.makedirs(directory, exist_ok=True)
    paths = {
        "load_displacement": load_displacement(
            campaign, os.path.join(directory, "fig1_load_displacement.png")),
        "dem_vs_iso": dem_vs_iso(campaign,
                                 os.path.join(directory, "fig2_dem_vs_iso.png")),
        "energy_closure": energy_closure(
            campaign, os.path.join(directory, "fig3_energy_closure.png")),
        "grid_refinement": grid_refinement(
            campaign, os.path.join(directory, "fig4_grid_refinement.png")),
        "cover_envelope": cover_envelope(
            campaign, os.path.join(directory, "fig5_cover_envelope.png")),
        "field_snapshot": field_snapshot(
            campaign, os.path.join(directory, "fig6_field_snapshot.png")),
    }
    print("[campaign] figures: " + json.dumps(paths), flush=True)
    return paths