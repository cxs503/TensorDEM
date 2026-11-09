"""Figures of the floe-population campaign (matplotlib, optional dependency)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["make_all"]


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for the campaign figures") from error
    return plt


def collision_kernel_figure(summary, path: Path) -> Path:
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(5.2, 4.6))
    measured, predicted, labels = [], [], []
    for table in summary["collision_tables"]:
        diameters = table["diameters"]
        for a in range(len(diameters)):
            for b in range(a, len(diameters)):
                km = table["k_measured"][a][b]
                kp = table["k_predicted"][a][b]
                if km > 0.0 and kp > 0.0:
                    measured.append(km)
                    predicted.append(kp)
                    labels.append(f"{diameters[a]:.1f}/{diameters[b]:.1f}")
    measured = np.asarray(measured)
    predicted = np.asarray(predicted)
    axes.scatter(measured, predicted, s=26, color="#1f77b4", label="DEM class pairs (3 seeds)")
    lo = float(min(measured.min(), predicted.min())) * 0.7
    hi = float(max(measured.max(), predicted.max())) * 1.4
    grid = np.array([lo, hi])
    axes.plot(grid, grid, "k-", lw=1.0, label="K_pred = K_meas")
    axes.fill_between(grid, grid * 0.85, grid * 1.15, color="0.8", alpha=0.5,
                      label="±15 % band")
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("measured K(D1,D2)  [D³ / time]")
    axes.set_ylabel("Enskog prediction πσ²⟨v_rel⟩g")
    axes.set_title(
        "C1 collision kernel: rate-weighted error "
        f"{summary['ledger']['C1']['value'] * 100:.1f} %"
    )
    axes.legend(fontsize=7, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def breakage_kernel_figure(summary, path: Path) -> Path:
    plt = _pyplot()
    kernel = summary["breakage_kernel"]
    figure, axes = plt.subplots(figsize=(5.4, 4.6))
    impacts = summary["impacts"]
    held = {(round(p["diameter"], 4), round(p["energy"], 4)) for p in kernel["validation_points"]}
    colors = {2.0: "#1f77b4", 3.0: "#2ca02c", 4.0: "#d62728", 5.0: "#9467bd"}
    energies = np.geomspace(1.0, 400.0, 60)
    for diameter in sorted({row["diameter"] for row in impacts}):
        rows = [row for row in impacts if abs(row["diameter"] - diameter) < 1e-9]
        fit_rows = [r for r in rows if (round(r["diameter"], 4), round(r["energy"], 4)) not in held]
        val_rows = [r for r in rows if (round(r["diameter"], 4), round(r["energy"], 4)) in held]
        axes.loglog([r["energy"] for r in fit_rows], [r["fragments"] for r in fit_rows], "o",
                    color=colors.get(diameter, "0.3"), label=f"DEM D={diameter:.0f} (fit)")
        if val_rows:
            axes.loglog([r["energy"] for r in val_rows], [r["fragments"] for r in val_rows], "s",
                        mfc="none", color=colors.get(diameter, "0.3"),
                        label=f"DEM D={diameter:.0f} (held out)")
        axes.loglog(energies, kernel["k_break"] * diameter ** kernel["alpha"] * energies ** kernel["beta"],
                    "-", lw=1.0, color=colors.get(diameter, "0.3"), alpha=0.8)
    axes.set_xlabel("impact energy E  [model units]")
    axes.set_ylabel("fragments per event B")
    axes.set_title(
        "C2 breakage kernel B = k D^α E^β  "
        f"(α={kernel['alpha']:.2f}, β={kernel['beta']:.2f}, held-out error "
        f"{summary['ledger']['C2']['value'] * 100:.0f} %)"
    )
    axes.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def forecast_figure(summary, path: Path) -> Path:
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(5.6, 4.4))
    for index, record in enumerate(summary["field"]):
        forecast = record["forecast"]
        comparison = forecast["comparison"]
        times = comparison["times"]
        label = f"seed {record['seed']}"
        axes.plot(record["times"], record["mean_size"], "o", ms=3.5,
                  color=f"C{index}", label=f"DEM {label}")
        axes.plot(times, comparison["mean_model"], "-", lw=1.3, color=f"C{index}",
                  label=f"population balance {label}")
        axes.axvline(forecast["start_time"], color="0.4", ls=":", lw=1.0)
    axes.set_xlabel("time  [D_min / v_rms]")
    axes.set_ylabel("mean floe diameter  [D_0]")
    axes.set_title(
        "C3 held-out forecast of the mean floe size "
        f"({summary['ledger']['C3']['value'] * 100:.1f} % mean relative error; dotted = forecast start)"
    )
    axes.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def control_figure(summary, path: Path) -> Path:
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(5.4, 4.2))
    for index, record in enumerate(summary["control"]):
        axes.plot(record["times"], record["mean_size"], "-", color=f"C{index}",
                  label=f"breakage off, seed {record['seed']}")
    for index, record in enumerate(summary["field"]):
        axes.plot(record["times"], record["mean_size"], "--", color=f"C{index}", alpha=0.7,
                  label=f"breakage on, seed {record['seed']}")
    axes.set_xlabel("time  [D_min / v_rms]")
    axes.set_ylabel("mean floe diameter  [D_0]")
    axes.set_title(
        "C5 negative control: no breakage kernel, no size drift "
        f"(std/mean = {summary['ledger']['C5']['value']:.3f})"
    )
    axes.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def fragment_census_figure(summary, path: Path) -> Path:
    plt = _pyplot()
    diagnostic = summary["ledger"]["C4"]["substitutes"]["fragment_exponent"]
    figure, axes = plt.subplots(figsize=(5.2, 4.2))
    if not diagnostic.get("available", False):
        pooled = summary.get("pooled_fragment_diameters") or []
        if pooled:
            axes.hist(np.asarray(pooled, dtype=float), bins=40, color="#1f77b4")
            axes.set_xlabel("fragment diameter  [D_0]")
            axes.set_ylabel("count")
        axes.set_title(
            "fragment census: quantised by the sub-sphere lattice, so no power law is "
            "fitted"
        )
    else:
        x = np.asarray(diagnostic["binned_x"])
        y = np.asarray(diagnostic["binned_y"])
        axes.loglog(x, y, "o", color="#1f77b4", label="pooled DEM fragment census")
        grid = np.geomspace(x.min(), x.max(), 40)
        axes.loglog(grid, y[0] * (grid / x[0]) ** (-diagnostic["alpha"]), "k--", lw=1.0,
                    label=f"power law α = {diagnostic['alpha']:.2f}")
    axes.set_xlabel("fragment diameter  [D_0]")
    axes.set_ylabel("n(D)  [count per unit diameter]")
    axes.set_title("fragment census and the broken-ice exponent window 2.5 – 3.5")
    axes.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def make_all(summary, artifact_dir: Path) -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return {
        "figure_collision": collision_kernel_figure(summary, artifact_dir / "c1_collision_kernel.png"),
        "figure_breakage": breakage_kernel_figure(summary, artifact_dir / "c2_breakage_kernel.png"),
        "figure_forecast": forecast_figure(summary, artifact_dir / "c3_forecast.png"),
        "figure_control": control_figure(summary, artifact_dir / "c5_negative_control.png"),
        "figure_census": fragment_census_figure(summary, artifact_dir / "c4_fragment_census.png"),
    }