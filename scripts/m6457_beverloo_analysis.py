"""m6457 — analysis of the Beverloo measurement sweep (summary.json + figures)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs/m6457_beverloo")
OUT = Path("artifacts/m6457")
DOC_FIG = Path("docs/figures")

ORDER = ["A_base_D5", "F_base_D4", "B_jitter_D5", "C_lowmu_D5", "E_lowmu_D5_tall", "D_lowmu_D4"]


def cv_txt(cv: float) -> str:
    return "n/a" if not math.isfinite(cv) else f"{cv:.3f}"


def reference_C(rec: dict) -> float:
    """C recovered using the literature reference bulk density 0.64 rho_p."""
    d_eff = rec["params"]["orifice_diameter"] - 1.5 * 0.015
    if d_eff <= 0:
        return float("nan")
    rho_ref = 0.64 * 2500.0
    return rec["W_measured_kg_s"] / (rho_ref * math.sqrt(9.81) * d_eff ** 2.5)


def load() -> dict[str, dict]:
    data = {}
    for p in sorted(RUNS.glob("*.json")):
        if p.stem == "quick":
            continue
        data[p.stem] = json.loads(p.read_text())
    return data


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    DOC_FIG.mkdir(parents=True, exist_ok=True)
    data = load()
    names = [n for n in ORDER if n in data]
    for n in names:
        data[n].setdefault("recovered_C_at_reference_rho", reference_C(data[n]))
        data[n].setdefault("initial_max_overlap_over_radius", None)

    # ---- md table ------------------------------------------------------
    cols = ("case D/d n0 discharged phi_b W_meas W_std W_lit rel err r2 nonzero windows CV wall_s")
    rows = []
    for n in names:
        r = data[n]
        rows.append(
            f"| {n} | {r['params']['orifice_diameter']/0.015:.2f} | {r['n_initial']} | "
            f"{r['n_discharged']} | {r['packing_fraction']:.3f} | "
            f"{r['W_measured_kg_s']:.4f} | {r['W_std_kg_s']:.4f} | {r['W_lit_kg_s']:.4f} | "
            f"{r['rel_err_vs_beverloo']:+.1%} | {r['steady_r2']:.3f} | "
            f"{r['window_nonzero_fraction']:.2f} | "
            f"{cv_txt(r['tail_window_cv'])} | {r['wall_clock_s']:.0f} |")
    header = ("| case | D/d | n0 | discharged | phi (measured) | W_meas (kg/s) | +/-1 sigma | "
              "W_lit (kg/s) | rel. err | steady r2 | windows>0 | CV(windows 2-5) | wall (s) |")
    table = "\n".join([header, "|" + "---|" * 13] + rows)

    summary = {
        "cases": {n: {k: data[n][k] for k in (
            "n_initial", "n_discharged", "packing_fraction", "W_measured_kg_s",
            "W_std_kg_s", "W_lit_kg_s", "rel_err_vs_beverloo", "steady_r2",
            "recovered_C", "recovered_C_at_reference_rho", "window_nonzero_fraction",
            "tail_window_cv", "initial_max_overlap_over_radius", "wall_clock_s",
            "W_windows_kg_s", "params")} for n in names},
        "markdown_table": table,
        "interpretation": {
            "ordered_lattice_controls": ["A_base_D5", "F_base_D4"],
            "disordered_and_flowing": [n for n in names if n not in ("A_base_D5", "F_base_D4")],
            "ordered_mean_abs_rel_err": sum(abs(data[n]["rel_err_vs_beverloo"]) for n in
                                            ("A_base_D5", "F_base_D4") if n in data) / 2,
            "flowing_mean_abs_rel_err": sum(abs(data[n]["rel_err_vs_beverloo"]) for n in names
                                            if n not in ("A_base_D5", "F_base_D4")) / max(1, len(names) - 2),
        },
    }
    for n in names:
        data[n].setdefault("recovered_C_at_reference_rho", reference_C(data[n]))
        data[n].setdefault("initial_max_overlap_over_radius", None)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "table.md").write_text(table + "\n")

    # ---- figure 1: measured vs literature ------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = range(len(names))
    meas = [data[n]["W_measured_kg_s"] for n in names]
    err = [data[n]["W_std_kg_s"] for n in names]
    lit = [data[n]["W_lit_kg_s"] for n in names]
    ax.errorbar(list(xs), meas, yerr=err, fmt="o", color="C0", capsize=4,
                label="measured (DEM mass balance, ±1σ)")
    ax.plot(list(xs), lit, "s", color="C3", label="Beverloo C=0.58, k=1.5, ρ_b=0.64ρ_p")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("flow rate W (kg/s)")
    ax.set_title("m6457: measured vs predicted orifice flow rate (D/d = 4–5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "W_measured_vs_literature.png", dpi=150)
    fig.savefig(DOC_FIG / "m6457_W_measured_vs_literature.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: per-window series -----------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for n in names:
        w = data[n]["W_windows_kg_s"]
        ax.plot(range(1, len(w) + 1), w, "o-", lw=1.2,
                label=f"{n} (r2={data[n]['steady_r2']:.2f})")
    ax.set_xlabel("flow-rate window index")
    ax.set_ylabel("window flow rate (kg/s)")
    ax.set_title("m6457: start-up burst and arching vs steady discharge")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "window_series.png", dpi=150)
    fig.savefig(DOC_FIG / "m6457_window_series.png", dpi=150)
    plt.close(fig)

    # ---- figure 3: recovered C vs measured packing fraction ------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for n in names:
        r = data[n]
        ax.plot(r["packing_fraction"], r["recovered_C"], "o")
        ax.annotate(n, (r["packing_fraction"], r["recovered_C"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.axhline(0.58, color="C3", ls="--", lw=1, label="literature C = 0.58")
    ax.set_xlabel("measured bulk packing fraction φ")
    ax.set_ylabel("C recovered with the measured ρ_b")
    ax.set_title("m6457: recovered Beverloo constant vs packing fraction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "recovered_C_vs_packing.png", dpi=150)
    fig.savefig(DOC_FIG / "m6457_recovered_C_vs_packing.png", dpi=150)
    plt.close(fig)

    print(table)
    print(json.dumps(summary["interpretation"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
