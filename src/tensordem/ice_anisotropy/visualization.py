"""Figures for the columnar-ice anisotropy study (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "plot_pole_figures",
    "plot_strength_vs_angle",
    "plot_yield_loci",
    "plot_dem_saturation",
]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _equal_area(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stereographic equal-area projection of c-axes onto the x-y plane."""
    d = np.asarray(directions, dtype=float)
    lower = d[:, 2] < 0.0
    d = d.copy()
    d[lower] *= -1.0
    # Lambert azimuthal equal-area: r = sqrt(2 (1 - cos(colatitude)) )
    cos_colat = np.clip(d[:, 2], -1.0, 1.0)
    r = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - cos_colat)) / 2.0)
    phi = np.arctan2(d[:, 1], d[:, 0])
    return r * np.cos(phi), r * np.sin(phi), lower


def plot_pole_figures(odfs: Mapping[str, Any], path: str | Path,
                      title: str = "c-axis fabric (equal-area poles)") -> Path:
    """Pole figures of the recorded ODFs, one panel each.

    ``odfs`` maps a label to a :class:`tensordem.ice_anisotropy.fabric.FabricODF`.
    """
    plt = _plt()
    n = len(odfs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (label, odf) in zip(axes, odfs.items()):
        x, y, _ = _equal_area(odf.directions)
        ax.scatter(x, y, s=4.0, alpha=0.55, c="#1f4e79", edgecolors="none")
        circ = plt.Circle((0.0, 0.0), 1.0, fill=False, color="0.4", lw=0.9)
        ax.add_patch(circ)
        ax.axhline(0.0, color="0.85", lw=0.6)
        ax.axvline(0.0, color="0.85", lw=0.6)
        vals = odf.eigenpairs()[0]
        ax.set_title(f"{label}\ng = {odf.anisotropy_index():.3f}, "
                     f"a_max = {vals[0]:.3f}", fontsize=9)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def plot_strength_vs_angle(curves: Mapping[str, Sequence[Mapping[str, float]]],
                           path: str | Path, *, highlight_deg: float = 45.0,
                           title: str = "Uniaxial strength vs load angle from the column axis") -> Path:
    """Normalised uniaxial strength ``sigma(theta)/sigma(90 deg)`` for each model."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    styles = ["o-", "s--", "^:", "d-.", "v-"]
    for (label, rows), style in zip(curves.items(), styles):
        th = [r["theta_deg"] for r in rows]
        base = rows[-1]["strength_Pa"] if rows[-1]["theta_deg"] == 90 else None
        if base is None:
            base = next(r["strength_Pa"] for r in rows if abs(r["theta_deg"] - 90.0) < 1e-9)
        val = [r["strength_Pa"] / base for r in rows]
        ax.plot(th, val, style, ms=4, lw=1.4, label=label)
    ax.axvline(highlight_deg, color="0.7", lw=0.8, ls=":")
    ax.annotate(f"{highlight_deg:.0f} deg", (highlight_deg, ax.get_ylim()[1]),
                xytext=(3, -12), textcoords="offset points", fontsize=8, color="0.4")
    ax.set_xlabel("load angle from the column axis (deg)")
    ax.set_ylabel(r"$\sigma(\theta)\,/\,\sigma_\perp$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def plot_yield_loci(surfaces: Mapping[str, Any], path: str | Path,
                    *, title: str = "Plane-stress yield loci (sigma_2 = 0)") -> Path:
    """Yield loci in the (transverse stress, column-axis stress) plane."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    styles = ["-", "--", ":", "-."]
    for (label, surf), style in zip(surfaces.items(), styles):
        s1 = np.linspace(-1.0, 1.0, 161)
        rows = surf.biaxial_locus(s1)
        x = [r["sigma1_over_perp"] for r in rows if r["feasible"]]
        y_up = [r["sigma3_over_parallel"] for r in rows if r["feasible"]]
        y_dn = [-v for v in y_up]
        ax.plot(x, y_up, style, color="C0", lw=1.4, label=f"{label} (column tension)")
        ax.plot(x, y_dn, style, color="C3", lw=1.4, label=f"{label} (column compression)")
    ax.axhline(0.0, color="0.7", lw=0.7)
    ax.axvline(0.0, color="0.7", lw=0.7)
    ax.set_xlabel(r"$\sigma_\perp$ (normalised by $\sigma_\perp$)")
    ax.set_ylabel(r"$\sigma_\parallel$ (normalised by $\sigma_\parallel$)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def plot_dem_saturation(table: Sequence[Mapping[str, Any]], path: str | Path,
                        title: str = "DEM lattice: micro bond-strength contrast vs macro anisotropy") -> Path:
    """Macro anisotropy of the bonded specimen vs the applied micro contrast."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    beta = [r["micro_contrast"] for r in table]
    ratio = [r["macro_anisotropy"] for r in table]
    ax.plot(beta, ratio, "o-", color="#1f4e79", lw=1.4)
    for b, r in zip(beta, ratio):
        ax.annotate(f"{r:.3f}", (b, r), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
    ax.axhspan(3.0, 5.0, color="C2", alpha=0.10)
    ax.annotate("literature band 3-5", (min(beta), 4.0), fontsize=8, color="#2e7d32")
    ax.set_xlabel(r"micro contrast $\beta$ in $\varepsilon_b(n)=\varepsilon_{b0}(1+\beta(\hat n\cdot a)^2)$")
    ax.set_ylabel(r"macro anisotropy $\sigma(0°)/\sigma(90°)$")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p