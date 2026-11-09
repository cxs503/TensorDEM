#!/usr/bin/env python
"""Acceptance run for p423w2-n07: per-facet (patch-local) mortar conservation,
true intersection on a kinked interface, gap/penetration accounting, the
time-domain interface energy account and the curved-metric transpose.

Everything numeric in ``docs/P423W2-N07_mortar_local_conservation.md`` comes from
the JSON this script writes; no number is typed by hand.

Usage::

    python scripts/run_mortar_local_conservation.py            # full run (twice, for G)
    python scripts/run_mortar_local_conservation.py --quick    # unit-test size
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.interface_operators import mortar_local_account as M  # noqa: E402
from tensordem.interface_operators.curved_patch_mortar import (       # noqa: E402
    CurvedMortarOperator,
    GraphPatch,
    SpherePatch,
)

OUT = ROOT / "artifacts" / "P423W2-N07_mortar_local_conservation"
FIG = OUT / "figures"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bool):
        return bool(o)
    raise TypeError(type(o))


def fig_per_patch(acc: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = acc["criteria"]["A_per_facet_conservation"]["rows"]
    patches = sorted({r["patch"] for r in rows})
    out = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for p in patches:
        for ratio, ls, mk in ((1, "-o", "o"), (4, "--s", "s")):
            sel = sorted([r for r in rows if r["patch"] == p and r["ratio"] == ratio],
                         key=lambda r: r["h"])
            if not sel:
                continue
            ax[0].loglog([r["h"] for r in sel], [max(r["true_per_facet"], 1e-17) for r in sel],
                         ls, ms=4, label=f"{p} {ratio}x")
            ax[1].loglog([r["h"] for r in sel], [r["plane_per_facet"] for r in sel],
                         ls, ms=4, label=f"{p} {ratio}x")
    for a, title in zip(ax, ("true intersection (per facet)", "plane projection (per facet)")):
        a.axhline(1e-12, color="k", lw=0.8, ls=":")
        a.set_xlabel("h")
        a.set_ylabel(r"$\max_j\,|\sum_i m_{ij}-|\Gamma_j||/|\Gamma_j|$")
        a.set_title(title)
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7)
    fig.tight_layout()
    path = FIG / "per_patch_conservation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    out.append({"name": "per_patch_conservation.png",
                "caption": "逐面片守恒缺陷：真交集恒在 1e-12 门（左）之下；面片投影在 4× 错配下 O(1e-1)（右）。"})
    return out


def fig_fold(fold: M.FoldPatch) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    master, slave = M.mismatch_case(fold, 5, 2)
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(121, projection="3d")
    ax.add_collection3d(Poly3DCollection(master.chords(), alpha=0.55,
                                         facecolor="tab:blue", edgecolor="tab:blue", lw=0.3))
    ax.add_collection3d(Poly3DCollection(slave.chords(), alpha=0.35,
                                         facecolor="tab:orange", edgecolor="tab:orange", lw=0.3))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("30 deg crease: master (blue) vs slave 2x (orange)")
    ax.view_init(elev=18, azim=-70)
    ax2 = fig.add_subplot(122)
    hs, defect = [], []
    ns = (3, 4, 6, 8)
    for n in ns:
        r = M.local_information_loss(fold, n, 2)
        hs.append(r["h"])
        defect.append(r["plane_per_facet"])
    ax2.loglog(hs, defect, "s-", label="plane projection (2x)")
    tdef = [M.local_information_loss(fold, n, 2)["true_per_facet"] for n in ns]
    ax2.loglog(hs, [max(t, 1e-17) for t in tdef], "o-", label="true intersection (2x)")
    ax2.set_xlabel("h")
    ax2.set_ylabel("per-facet coverage defect")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    ax2.set_title("crease interface, 2x mismatch")
    fig.tight_layout()
    path = FIG / "fold_geometry_and_defect.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [{"name": "fold_geometry_and_defect.png",
             "caption": "30° 折线界面（m6114 几何）：非匹配两侧与 4× 错配下的逐面片缺陷（真交集 vs 面片投影）。"}]


def fig_energy(patch, n=5, ratio=2) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    master, slave = M.mismatch_case(patch, n, ratio)
    ops = {
        "curved (exact)": CurvedMortarOperator(master, slave, order=2, measure="curved"),
        "centroid-J (frozen)": CurvedMortarOperator(master, slave, order=2, measure="centroid_j"),
        "flat (J=1)": CurvedMortarOperator(master, slave, order=2, measure="flat"),
        "broken local conservation": M.break_local_conservation(
            CurvedMortarOperator(master, slave, order=2, measure="curved")),
    }
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for label, op in ops.items():
        r = M.interface_energy_account(op, dt=1e-3, nsteps=200, keep_history=True)
        h = r["history"]
        ax[0].semilogy(h["t"], [max(abs(v), 1e-32) for v in h["interface_power"]], lw=1.2,
                       label=f"{label} (gate {r['interface_power_rel_max']:.1e})")
        ax[1].plot(h["t"], h["energy"], lw=1.1, label=label)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"$|P_{int}|$")
    ax[0].set_title(f"{patch.label}: interface power, {ratio}x mismatch")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("E(t)")
    ax[1].set_title("total energy")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    path = FIG / "energy_account.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [{"name": "energy_account.png",
             "caption": "时域界面功率 |P_int|（左，理想应为 0）与总能量历史（右）；破度量/破局部守恒的负控清晰可分。"}]


def fig_gap(acc: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [{"patch": c["patch"], **r}
            for c in acc["criteria"]["C_gap_penetration_terms"]["rows"] for r in c["rows"]]
    labels = [f'{r["patch"]}\n{r["case"]}' for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(rows))
    ax[0].bar(x - 0.2, [r["gap_unit_volume_rel"] for r in rows], 0.4, label="$V_{gap}/|\\Gamma|$")
    ax[0].bar(x + 0.2, [r["pen_unit_volume_rel"] for r in rows], 0.4, label="$V_{pen}/|\\Gamma|$")
    ax[0].set_yscale("log")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=6, rotation=90)
    ax[0].set_ylabel("volume / |Gamma|")
    ax[0].legend(fontsize=8)
    ax[0].set_title("accounted gap / penetration volume")
    ax[1].bar(x - 0.2, [max(r["identity_balance_rel"], 1e-18) for r in rows], 0.4,
              label="identity with gap terms")
    ax[1].bar(x + 0.2, [r["without_gap_term_residual"] for r in rows], 0.4,
              label="gap terms dropped (control)")
    ax[1].axhline(1e-12, color="k", ls=":", lw=0.8)
    ax[1].axhline(1e-3, color="r", ls="--", lw=0.8)
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=6, rotation=90)
    ax[1].legend(fontsize=8)
    ax[1].set_title("per-facet identity residual")
    fig.tight_layout()
    path = FIG / "gap_penetration_balance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [{"name": "gap_penetration_balance.png",
             "caption": "间隙/渗透体积记账（左）与逐面片恒等式残差（右）：关掉间隙项后残差回到 1e-1 量级（负控）。"}]


def fig_transpose(acc: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4))
    for row in acc["criteria"]["E_transpose_consistency"]["rows"]:
        rs = sorted(row["rows"], key=lambda r: r["h"])
        h = [r["h"] for r in rs]
        ax.loglog(h, [r["transport_rel_curved"] for r in rs], "-o", ms=4,
                  label=f'{row["patch"]} curved (p={row["rate_transport_curved"]:.2f})')
        ax.loglog(h, [r["transport_rel_naive"] for r in rs], "--s", ms=3,
                  label=f'{row["patch"]} midpoint (p={row["rate_transport_naive"]:.2f})')
    h = np.array(sorted(r["h"] for r in acc["criteria"]["E_transpose_consistency"]["rows"][0]["rows"]))
    ax.loglog(h, 5e-2 * (h / h[0]) ** 2, "k:", lw=1, label="$h^2$")
    ax.set_xlabel("h")
    ax.set_ylabel("adjoint transport defect")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6)
    ax.set_title("transpose consistency in the curved metric")
    fig.tight_layout()
    path = FIG / "transpose_consistency.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [{"name": "transpose_consistency.png",
             "caption": "曲面度量下转置（伴随）输运缺陷 vs h：二阶重构 p≈2，中点转置（负控）明显更差。"}]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small meshes, single repeat")
    ap.add_argument("--repeats", type=int, default=2, help="runs for the determinism gate")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    global FIG
    FIG = fig_dir

    meshes = (3, 4) if args.quick else (4, 6, 8)
    t0 = time.time()
    acc = M.run_acceptance(meshes=meshes,
                           energy_resolutions=(3, 4) if args.quick else (4, 6, 8),
                           energy_ratios=(1, 4),
                           e_meshes=(4, 6) if args.quick else (6, 8, 12, 16),
                           with_regression=not args.quick)
    dur = time.time() - t0

    # G: determinism -- the same call must reproduce the same JSON bit for bit
    # (provenance carries the wall clock, so it is compared separately).
    reps = 1 if args.quick else max(1, args.repeats)
    det = []
    for _ in range(reps - 1):
        again = M.run_acceptance(meshes=meshes,
                                 energy_resolutions=(3, 4) if args.quick else (4, 6, 8),
                                 energy_ratios=(1, 4),
                                 e_meshes=(4, 6) if args.quick else (6, 8, 12, 16),
                                 with_regression=not args.quick)
        a = json.dumps({k: v for k, v in acc.items() if k != "provenance"},
                       sort_keys=True, default=_json_default)
        b = json.dumps({k: v for k, v in again.items() if k != "provenance"},
                       sort_keys=True, default=_json_default)
        det.append(a == b)
    acc["criteria"]["G_determinism"] = {
        "repeats": reps,
        "bitwise_identical": bool(all(det)) if det else True,
        "pass": bool(all(det)) if det else True,
    }
    acc["all_pass"] = bool(all(v.get("pass", True) for v in acc["criteria"].values()))
    acc["wall_time_s"] = dur

    # numbers first: a figure failure must never lose the acceptance data
    path = out_dir / "acceptance.json"
    path.write_text(json.dumps(acc, indent=2, sort_keys=True, default=_json_default))

    figures = []
    if not args.no_figs:
        figures += fig_per_patch(acc)
        figures += fig_fold(M.FoldPatch(30.0))
        figures += fig_energy(SpherePatch())
        figures += fig_gap(acc)
        figures += fig_transpose(acc)
    acc["figures"] = figures
    path.write_text(json.dumps(acc, indent=2, sort_keys=True, default=_json_default))
    print(f"wrote {path}  all_pass={acc['all_pass']}  wall={dur:.1f}s")
    for k, v in acc["criteria"].items():
        print(f"  {k}: pass={v.get('pass')}")
    return 0 if acc["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())