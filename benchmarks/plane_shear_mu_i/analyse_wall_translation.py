"""Round 33: tables and figures for the ``wall_translate`` ablation.

Reads every ``runs/wall_translation_r33/*.json`` produced by
``wall_translation_ablation.py`` / ``run_wall_translation_campaign.py`` and emits

* ``runs/wall_translation_r33/tables_r33.md`` -- markdown tables, numbers
  interpolated from the JSON (no hand-typed value),
* ``figures/wall_translation_r33.png`` -- left: the measured velocity profile
  closure, right: mu/phi against the da Cruz 2-D ruler.

Usage::

    python analyse_wall_translation.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parents[1]
RUNS = BENCH_DIR / "runs" / "wall_translation_r33"
FIGS = BENCH_DIR / "figures"
sys.path.insert(0, str(BENCH_DIR))

from reference_2d import mu_2d as mu_ref, nu_2d as phi_ref  # noqa: E402


def load() -> list[dict]:
    rows = []
    for p in sorted(RUNS.glob("*.json")):
        d = json.loads(p.read_text())
        d["_file"] = p.name
        rows.append(d)
    return rows


def _f(v, fmt="{:.3f}"):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return fmt.format(v) if math.isfinite(v) else "n/a"


def table_ablation(rows: list[dict]) -> str:
    head = ("| mode | I_target | n_free | p_wall/P | φ | mu_bulk | "
            "γ̇_meas/γ̇_nom | slip_bot/h | slip_top/h | r² | source |\n")
    head += "|" + "---|" * 11 + "\n"
    out = head
    for r in rows:
        out += ("| {mode} | {I} | {n} | {p} | {phi} | {mu} | {ratio} | {sb} | {st} | "
                "{r2} | `{f}` |\n").format(
            mode=("advected" if r.get("wall_translate") else "frozen"),
            I=_f(r.get("I_target"), "{:g}"), n=r.get("n_free"),
            p=_f(r.get("p_wall_over_P")), phi=_f(r.get("phi_gap"), "{:.4f}"),
            mu=_f(r.get("mu_bulk")), ratio=_f(r.get("gamma_dot_ratio")),
            sb=_f(r.get("slip_length_bottom_over_h"), "{:+.2f}"),
            st=_f(r.get("slip_length_top_over_h"), "{:+.2f}"),
            r2=_f(r.get("profile_r2")), f=r["_file"])
    return out


def table_ruler(rows: list[dict]) -> str:
    """Each point judged against the archived 2-D law at its *own* measured I."""
    out = ("| mode | I_measured | mu | mu_ref(I_meas) | err % | φ | φ_ref(I_meas) | "
           "err % | within 10 % both? |\n")
    out += "|" + "---|" * 9 + "\n"
    for r in rows:
        I = float(r.get("I_measured", float("nan")))
        if not math.isfinite(I) or I <= 0:
            continue
        mu, phi = float(r["mu_bulk"]), float(r["phi_gap"])
        mr, pr = mu_ref(I), phi_ref(I)
        em, ep = 100.0 * (mu - mr) / mr, 100.0 * (phi - pr) / pr
        out += ("| {m} | {I:.4f} | {mu:.3f} | {mr:.3f} | {em:+.1f} | {phi:.4f} | "
                "{pr:.4f} | {ep:+.1f} | {ok} |\n").format(
            m=("advected" if r.get("wall_translate") else "frozen"), I=I, mu=mu,
            mr=mr, em=em, phi=phi, pr=pr, ep=ep,
            ok="yes" if abs(em) <= 10 and abs(ep) <= 10 else "no")
    return out


def figure(rows: list[dict]) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax = axes[0]
    labels = [("frozen", "frozen wall (archive, defect)", "tab:red", "o"),
              ("advected", "advected wall (R33 fix)", "tab:blue", "s")]
    for key, lab, col, mk in labels:
        sel = [r for r in rows if (r.get("wall_translate") is (key == "advected"))]
        if not sel:
            continue
        x = [float(r["I_target"]) for r in sel]
        y = [float(r["gamma_dot_ratio"]) for r in sel]
        ax.semilogx(x, y, mk + "-", color=col, label=lab, lw=1.4, ms=6)
    ax.axhline(1.0, color="k", ls=":", lw=1.0, label="closure = 1")
    ax.axhspan(0.9, 1.1, color="0.85", zorder=0, label="±10 % band")
    ax.set_xlabel("nominal inertial number $I$")
    ax.set_ylabel(r"$\dot\gamma_{meas}/\dot\gamma_{nom}$")
    ax.set_title("velocity-profile closure: does the wall impose its rate?")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    I = np.logspace(-3, np.log10(0.3), 120)
    ax.plot(I, [mu_ref(i) for i in I], "k-", lw=1.6,
            label=r"da Cruz 2005 2-D: $\mu=0.22+1.0\,I$")
    for key, lab, col, mk in labels:
        sel = [r for r in rows if (r.get("wall_translate") is (key == "advected"))]
        if not sel:
            continue
        ax.loglog([float(r["I_target"]) for r in sel],
                  [float(r["mu_bulk"]) for r in sel], mk, color=col, ms=7, label=lab)
    ax.set_xlabel("$I$ (nominal)")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(r"friction against the archived ruler")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIGS / "wall_translation_r33.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    rows = load()
    if not rows:
        print("no run artifacts yet")
        return 1
    md = ["# R33 — `wall_translate` ablation (generated, do not hand-edit)\n",
          f"artifacts: {len(rows)} JSON in `runs/wall_translation_r33/`\n",
          "## Table A — point-by-point closure diagnostics\n",
          table_ablation(rows), "\n## Table B — judged against the archived 2-D ruler\n",
          table_ruler(rows)]
    (RUNS / "tables_r33.md").write_text("\n".join(md))
    fig = figure(rows)
    print(f"wrote {RUNS / 'tables_r33.md'} and {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())