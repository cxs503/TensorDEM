"""m14840 figures + programmatic tables from artifacts/shear_cell_m14840/*.json."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path("artifacts/shear_cell_m14840")
FIG = Path("docs/figures")
FIG.mkdir(parents=True, exist_ok=True)

DATA = {t: json.loads((ART / f"{t}.json").read_text()) for t in ("drag", "bedref", "long")}


def rows(tag):
    return DATA[tag]["rows"]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # --- panel A: normal-stress holding error, drag vs bed-referenced dashpot
    ax = axes[0]
    sig = [r["sigma_target"] for r in rows("drag")]
    drag = [100 * r["sigma_rel"] for r in rows("drag")]
    bed = [100 * r["sigma_rel"] for r in rows("bedref")]
    x = np.arange(len(sig))
    ax.axhspan(-3, 3, color="0.85", zorder=0, label="+/-3 % band")
    ax.bar(x - 0.19, drag, 0.38, color="tab:red", label='drag servo (m14721)')
    ax.bar(x + 0.19, bed, 0.38, color="tab:green", label='bed-referenced (m14840)')
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{int(s)} Pa" for s in sig])
    ax.set_ylabel(r"$\sigma_n$ error vs target  [%]")
    ax.set_title("A. Load holding of the platen servo")
    ax.legend(fontsize=8, loc="lower left")

    # --- panel B: is the traction steady inside the averaging window?
    ax = axes[1]
    for tag, style, label in (
        ("drag", dict(color="tab:red", ls="--", marker="o"),
         r"drag servo, $\gamma$=0.30 (4000 Pa)"),
        ("bedref", dict(color="tab:green", ls="-", marker="s"),
         r"bed-referenced, $\gamma$=0.30 (4000 Pa)"),
        ("long", dict(color="tab:blue", ls="-", marker="^"),
         r"bed-referenced, $\gamma$=1.00 (4000 Pa)"),
    ):
        row = next(r for r in rows(tag) if abs(r["sigma_target"] - 4000.0) < 1e-6)
        q = np.asarray(row["tau_window_quarters"], dtype=float)
        ax.plot(np.arange(1, 5), q / q.mean(), **style, label=label + "  drift %+.2f" % row["drift"])
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("quarter of the shear window")
    ax.set_ylabel(r"$\tau$ / window mean")
    ax.set_title(r"B. Traction is not stationary (78 beads, 3 layers)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = FIG / "shear_cell_m14840_servo.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)

    # --- programmatic tables (no hand-typed numbers) -----------------------
    lines = []
    lines.append("### 应力保持（伺服对比）\n")
    lines.append("| sigma_n 目标 [Pa] | drag sigma_n [Pa] | drag 相对误差 | bedref sigma_n [Pa] | bedref 相对误差 | drag mu | bedref mu |")
    lines.append("|---|---|---|---|---|---|---|")
    for rd, rb in zip(rows("drag"), rows("bedref")):
        lines.append("| %.0f | %.2f | %+.2f%% | %.2f | %+.2f%% | %.4f | %.4f |" % (
            rd["sigma_target"], rd["sigma"], 100 * rd["sigma_rel"], rb["sigma"], 100 * rb["sigma_rel"],
            rd["mu"], rb["mu"]))
    lines.append("\n### 拖曳伺服载荷偏置的机理检验（drag 模式）\n")
    lines.append("稳态解 dN = M_p v_z (1-e^{-dt/tau})/dt\n")
    lines.append("| sigma_n [Pa] | v_z [m/s] | dN 实测 [N] | dN 拖曳律预测 [N] | 相对差 |")
    lines.append("|---|---|---|---|---|")
    for rd in rows("drag"):
        rel = (rd["dN_drag_law_N"] - rd["dN_data_N"]) / rd["dN_data_N"] if rd["dN_data_N"] else float("nan")
        lines.append("| %.0f | %.3e | %+.4f | %+.4f | %+.1f%% |" % (
            rd["sigma_target"], rd["vz_platen"], rd["dN_data_N"], rd["dN_drag_law_N"], 100 * rel))
    lines.append("\n### 剪切率/惯性数记账（fill_height 名义 vs 实测床高）\n")
    lines.append("| sigma_n [Pa] | 床高 [m] | 名义 gamma_dot [1/s] | 实际 gamma_dot [1/s] | I (名义率) | I (实测率, 修正) | tau 四分之一均值 [Pa] | 窗口相对漂移 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for tag, note in (("bedref", ""), ("long", " (gamma=1.0)")):
        for rb in rows(tag):
            i_act = rb["shear_rate_actual"] * 0.01 / math.sqrt(rb["sigma"] / 2500.0)
            lines.append("| %.0f%s | %.4f | %.2f | %.2f | %.4f | %.4f | %s | %+.2f |" % (
                rb["sigma_target"], note, rb["bed_height"], rb["shear_rate_nominal"],
                rb["shear_rate_actual"], rb["inertial_number"], i_act,
                ", ".join("%.0f" % q for q in rb["tau_window_quarters"]), rb["drift"]))
    tbl = Path("docs/SHEAR_CELL_SERVO_AUDIT_M14840_tables.md")
    tbl.write_text("\n".join(lines) + "\n")
    print("wrote", tbl)


if __name__ == "__main__":
    main()