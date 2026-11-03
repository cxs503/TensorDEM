"""Run the tangential/normal contact verification campaign and write artifacts.

Writes into ``outputs/tangential_friction_verification/``:

* ``results.json``   - full campaign output (numbers, convergence series, loops)
* ``tables.md``      - markdown tables rendered from ``results.json``
* ``fig_restitution.png``        - measured vs target e, and of the rebound ratio
* ``fig_stick_slip_loops.png``   - measured hysteresis loops vs closed-form polygon
* ``fig_stick_slip_energy.png``  - dissipated work per cycle vs amplitude (+ negative control)

Usage:
    PYTHONPATH=src python examples/tangential_friction_verification.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.tangential_verification import run_campaign  # noqa: E402

DEFAULT_OUT = pathlib.Path("outputs/tangential_friction_verification")


def render_tables(res: dict) -> str:
    rr = res["anchor_restitution"]
    tt = res["anchor_stick_slip"]
    lines = []
    lines.append("### Table R1 - restitution mapping round trip (e -> gamma_n -> e)\n")
    lines.append("| e target | k_n [N/m] | m_eff [kg] | gamma_n [N s/m] | e round trip | abs error |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in rr["mapping_roundtrip"]:
        lines.append("| {e:g} | {k_n:g} | {m_eff:g} | {gamma_n:.6f} | {e_roundtrip:.12f} | {abs_error:.2e} |".format(**row))

    lines.append("\n### Table R2 - time-stepped rebound vs target e (finest dt)\n")
    lines.append("| e target | e measured | rel. error [%] | zeta [-] | gamma_n [N s/m] | dt [s] |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in rr["sweep"]:
        lines.append("| {e_target:g} | {e_measured:.6f} | {rel:.4f} | {zeta:.6f} | {gamma_n:.6f} | {dt:.3e} |".format(
            rel=100.0 * abs(row["rel_error"]), **row))

    lines.append("\n### Table R3 - dt refinement of the rebound measurement (e target = 0.5)\n")
    lines.append("| steps per contact period | dt [s] | e measured | rel. error [%] |")
    lines.append("|---:|---:|---:|---:|")
    for row in rr["convergence"]:
        lines.append("| {steps_per_period:d} | {dt:.3e} | {e_measured:.6f} | {rel:.4f} |".format(
            rel=100.0 * abs(row["rel_error"]), **row))

    lines.append("\n### Table T1 - stick-slip loop area vs closed form (F_n = 100 N, k_t = 1000 N/m, mu = 0.3)\n")
    lines.append("| amplitude a [m] | a_stick [m] | sliding fraction | W analytic [J] | W measured [J] | rel. error [%] | W unclamped (negative control) [J] |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for pos, neg in zip(tt["sweep"], tt["negative_control_unclamped"]):
        lines.append("| {a:g} | {a_stick:g} | {sf:.3f} | {wa:.6f} | {wm:.6f} | {rel:.4f} | {neg:.3e} |".format(
            a=pos["amplitude"], a_stick=pos["a_stick"], sf=pos["sliding_fraction"],
            wa=pos["work_analytic"], wm=pos["work_measured"],
            rel=100.0 * abs(pos["rel_error"]), neg=neg["work_measured"]))

    lines.append("\n### Table T2 - steps-per-cycle refinement of the loop area (a = 0.2 m)\n")
    lines.append("| steps per cycle | W analytic [J] | W measured [J] | rel. error [%] |")
    lines.append("|---:|---:|---:|---:|")
    for row in tt["convergence"]:
        lines.append("| {steps_per_cycle:d} | {work_analytic:.6f} | {work_measured:.6f} | {rel:.5f} |".format(
            rel=100.0 * abs(row["rel_error"]), **row))
    return "\n".join(lines) + "\n"


def plot_restitution(res: dict, out: pathlib.Path) -> None:
    rr = res["anchor_restitution"]
    e_t = [r["e_target"] for r in rr["sweep"]]
    e_m = [r["e_measured"] for r in rr["sweep"]]
    err = [100.0 * abs(r["rel_error"]) for r in rr["sweep"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="identity (exact mapping)")
    ax1.plot(e_t, e_m, "o-", ms=4, label="time-stepped rebound")
    ax1.set_xlabel("target coefficient of restitution e [-]")
    ax1.set_ylabel("measured rebound ratio [-]")
    ax1.set_title("Anchor R: e mapping verified by impact integration")
    ax1.legend()
    ax1.grid(alpha=0.3)

    spp = [r["steps_per_period"] for r in rr["convergence"]]
    cerr = [100.0 * abs(r["rel_error"]) for r in rr["convergence"]]
    ax2.loglog(spp, cerr, "s-", label="rel. error vs dt")
    ax2.loglog(spp, [cerr[0] * spp[0] / s for s in spp], "k:", lw=1, label="1st-order slope reference")
    ax2.set_xlabel("integration steps per contact period")
    ax2.set_ylabel("|e_measured - e_target| / e_target  [%]")
    ax2.set_title("Anchor R: discretisation convergence (e = 0.5)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_restitution.png", dpi=130)
    plt.close(fig)
    print(f"[fig] worst rel. error over e sweep: {max(err):.4f} %")


def plot_loops(res: dict, out: pathlib.Path) -> None:
    loops = res["anchor_stick_slip"]["loops"]
    mu, f_n, k_t = 0.3, 100.0, 1000.0
    fig, axes = plt.subplots(1, len(loops), figsize=(4.1 * len(loops), 4.0), squeeze=False)
    for ax, (key, loop) in zip(axes[0], sorted(loops.items(), key=lambda kv: float(kv[0]))):
        a = float(key)
        lim = mu * f_n
        a_stick = lim / k_t
        # closed-form path: stick branches + friction plateaus
        d_analytic = [a, a_stick, -a_stick, -a, -a_stick, a_stick, a]
        f_analytic = [lim, lim, -lim, -lim, -lim, lim, lim]
        ax.plot(d_analytic, f_analytic, "k--", lw=1.2, label="closed form (Coulomb loop)")
        ax.plot(loop["delta"], loop["force"], "-", lw=2, alpha=0.75,
                label=f"stepped model, W={loop['work_measured']:.4f} J")
        ax.axhline(lim, color="C3", lw=0.8, ls=":")
        ax.axhline(-lim, color="C3", lw=0.8, ls=":")
        ax.set_title(f"a = {a:g} m  ($W_{{analytic}}$ = {loop['work_analytic']:.4f} J)")
        ax.set_xlabel("tangential displacement [m]")
        ax.set_ylabel("tangential force [N]")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Anchor T: Cundall-Strack stick-slip loops, model vs closed form "
                 "(dotted lines = +/- mu F_n)")
    fig.tight_layout()
    fig.savefig(out / "fig_stick_slip_loops.png", dpi=130)
    plt.close(fig)


def plot_energy(res: dict, out: pathlib.Path) -> None:
    tt = res["anchor_stick_slip"]
    a = [r["amplitude"] for r in tt["sweep"]]
    w_an = [r["work_analytic"] for r in tt["sweep"]]
    w_me = [r["work_measured"] for r in tt["sweep"]]
    w_neg = [r["work_measured"] for r in tt["negative_control_unclamped"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.plot(a, w_an, "k--", lw=1.2, label="4 mu F_n (a - mu F_n / k_t)")
    ax1.plot(a, w_me, "o", ms=5, label="stepped model (clamped state)")
    ax1.plot(a, w_neg, "x", ms=6, label="negative control (unclamped state)")
    ax1.set_xlabel("tangential amplitude a [m]")
    ax1.set_ylabel("dissipated work per cycle [J]")
    ax1.set_title("Anchor T: per-cycle dissipation vs amplitude")
    ax1.grid(alpha=0.3)
    ax1.legend()

    spc = [r["steps_per_cycle"] for r in tt["convergence"]]
    cerr = [100.0 * abs(r["rel_error"]) for r in tt["convergence"]]
    ax2.loglog(spc, cerr, "s-")
    ax2.set_xlabel("integration steps per cycle")
    ax2.set_ylabel("|W_measured - W_analytic| / W_analytic [%]")
    ax2.set_title("Anchor T: loop-area convergence (a = 0.2 m)")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig_stick_slip_energy.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--quick", action="store_true", help="coarser step counts")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.quick:
        res = run_campaign(steps_per_period=(2000,), steps_per_cycle=(1000,))
    else:
        res = run_campaign()

    (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "tables.md").write_text(render_tables(res))
    plot_restitution(res, out)
    plot_loops(res, out)
    plot_energy(res, out)
    print(f"[out] artifacts written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())