"""Round 16 (m20855) -- strain-convergence audit of the mu(I)/phi(I) case.

Round 15 (``phi_estimator_audit.py``) closed with one explicit open question:

    "The archived protocol cannot separate 'loose initial fill still
     compacting' from a genuine steady state, because the shear budget is
     ``max(min_shear_steps, gamma_target/(gamma_dot dt))`` ...  A
     strain-convergence sweep ... was launched ... it had not returned within
     this session, so no claim about the steady-state phi(I) is made here.
     That is the next round's first measurement."

This module makes that measurement.  It reads the six archived runs
(``runs/phi_audit/I*_smooth_s5279.json``, cluster jobs ``cj2950 .. cj2955``)
and answers three questions:

Q1  Was the requested strain actually delivered?  (``--strain 2.0``)
Q2  Is ``phi`` converged at that strain, i.e. can a steady-state ``phi(I)`` be
    quoted at all?
Q3  With the corrected container definition, where does the measured
    ``phi(I)`` / ``mu(I)`` sit against the reference law?

The answer to Q1 is *no* at high ``I``, and the mechanism is arithmetic, not
statistical -- see :func:`strain_budget`.  That is itself the round's main
finding: the round-15 sweep cannot be read as a strain-convergence study above
``I = 0.01``, and any future round that quotes a high-``I`` point from it is
quoting a state sheared up to 50x past its budget.

Run
---
    python benchmarks/plane_shear_mu_i/strain_convergence_audit.py
    python benchmarks/plane_shear_mu_i/strain_convergence_audit.py --figures
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, "runs", "phi_audit")

# device constants of the shipped servo-shear cell (phi_estimator_audit.py)
PRESSURE = 6.25e4
DENSITY = 2500.0
DIAMETER = 0.01                       # RADIUS = 0.005
STRAIN_TOL = 1.05                     # >5 % over the request == not respected
SETTLE_GUARD = 100                    # the +100 in phi_estimator_audit.live_run


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------
def load_runs(run_dir: str = RUN_DIR) -> list:
    """All archived strain-sweep points, ordered by ``I_target``."""
    paths = sorted(glob.glob(os.path.join(run_dir, "I*_smooth_*.json")))
    runs = []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        if d.get("provenance", {}).get("series_downsample_factor") is None:
            continue                  # synthetic / unrelated artifacts
        d["_path"] = os.path.relpath(p, HERE)
        runs.append(d)
    runs.sort(key=lambda d: d["I_target"])
    return runs


def gamma_dot_of(I: float) -> float:
    """Shear rate the device applies for ``inertial_number = I``.

    ``mu_i_rheology`` resolves ``gamma_dot = I sqrt(p/rho) / d`` from the
    settled pressure; the servo then drives ``plate_vx = gamma_dot * plate_z``,
    so the true strain rate equals that value.
    """
    return I * math.sqrt(PRESSURE / DENSITY) / DIAMETER


# --------------------------------------------------------------------------
# Q1 -- was the requested strain delivered?
# --------------------------------------------------------------------------
def strain_budget(run: dict) -> dict:
    """Reconstruct the shear-step budget and attribute the overrun.

    The device shears for
        shear_steps = max(min_shear_steps, ceil(strain/(gamma_dot dt)))
    and then, for constant-stress validity, keeps shearing *while the servo has
    not re-established the applied pressure*, up to ``max_recover_steps``
    (default 20 000) more steps.  ``shear_steps`` is therefore a floor, never a
    ceiling: nothing in the loop bounds the strain itself.
    """
    I = run["I_target"]
    dt = run["dt"]
    gd = gamma_dot_of(I)
    req = run["strain_requested"]
    executed = int(run["shear_steps_executed"])
    achieved = float(run["strain_achieved"])

    planned = int(math.ceil(req / (gd * dt))) + SETTLE_GUARD
    extra = executed - planned
    # strain the device *would* have run for the planned floor
    strain_floor = planned * gd * dt
    return {
        "I_target": I,
        "gamma_dot": gd,
        "strain_requested": req,
        "planned_floor_steps": planned,
        "strain_at_planned_floor": strain_floor,
        "steps_executed": executed,
        "recover_steps_spent": extra,
        "strain_achieved": achieved,
        "overrun_factor": achieved / req,
        "respects_budget": achieved <= req * STRAIN_TOL,
        # cross-check: the device's own strain bookkeeping uses the same gd
        "strain_reconstructed": executed * gd * dt,
        "bookkeeping_error": abs(executed * gd * dt - achieved),
    }


def budget_table(runs: list) -> list:
    return [strain_budget(r) for r in runs]


# --------------------------------------------------------------------------
# Q2 -- is phi converged at the achieved strain?
# --------------------------------------------------------------------------
def block_means(series: list, n: int = 4) -> list:
    """Mean of each of ``n`` equal blocks of a series (NaN-safe)."""
    vals = [float(v) for v in series if v == v]
    if len(vals) < n:
        return []
    k = len(vals) // n
    out = []
    for i in range(n):
        blk = vals[i * k:(i + 1) * k] if i < n - 1 else vals[i * k:]
        out.append(sum(blk) / len(blk))
    return out


def convergence(run: dict, key: str = "phi_extent", n: int = 4) -> dict:
    """Is ``key`` stationary?  Report block means and a normalised drift.

    ``drift`` is ``(last - first) / first`` over the block means, i.e. how far
    the quantity moved between the start and the end of the *shear* series.  A
    converged quantity drifts by well under a percent; a bed still compacting
    drifts by more.
    """
    series = (run.get("series") or {}).get(key) or []
    bm = block_means(series, n)
    if not bm or bm[0] == 0:
        return {"key": key, "blocks": bm, "drift_pct": float("nan"),
                "terminal_drift_pct": float("nan"), "converged": False}
    drift = 100.0 * (bm[-1] - bm[0]) / bm[0]
    # the *terminal* drift -- last block vs second-to-last -- is the steady
    # state test: a bed that already travelled from a loose fill to its steady
    # density can still show a large whole-series drift while being settled by
    # the end, and a bed still compacting shows a terminal drift of the same
    # sign and comparable size.
    terminal = float("nan")
    if len(bm) >= 2 and bm[-2] != 0:
        terminal = 100.0 * (bm[-1] - bm[-2]) / bm[-2]
    # quote a quantity only if it has stopped moving at the end (< 0.5 %)
    return {"key": key, "blocks": bm, "drift_pct": drift,
            "terminal_drift_pct": terminal,
            "converged": abs(terminal) <= 0.5}


# --------------------------------------------------------------------------
# Q3 -- position against the reference law
# --------------------------------------------------------------------------
def reference_compare(runs: list, abscissa: str = "I_bulk_final") -> list:
    from reference_law import ref_mu, ref_phi

    rows = []
    for r in runs:
        Ix = float(r[abscissa])
        mu, phi = float(r["mu_full_final"]), float(r["phi_container_final"])
        mr, pr = float(ref_mu(Ix)), float(ref_phi(Ix))
        rows.append({
            "I_target": r["I_target"],
            "I_used": Ix,
            "mu": mu, "mu_ref": mr, "mu_err_pct": 100.0 * (mu - mr) / mr,
            "phi": phi, "phi_ref": pr,
            "phi_err_pct": 100.0 * (phi - pr) / pr,
            "phi_extent": float(r["phi_extent_final"]),
            "strain_achieved": float(r["strain_achieved"]),
        })
    return rows


def verdict(runs: list) -> dict:
    """Separate the points that may be quoted from those that may not."""
    budget = budget_table(runs)
    respected = [b for b in budget if b["respects_budget"]]
    violated = [b for b in budget if not b["respects_budget"]]

    by_i = {r["I_target"]: r for r in runs}
    conv = {b["I_target"]: convergence(by_i[b["I_target"]]) for b in respected}
    conv_mu = {b["I_target"]: convergence(by_i[b["I_target"]], key="full_mu")
               for b in respected}

    return {
        "n_points": len(runs),
        "n_strain_respected": len(respected),
        "n_strain_violated": len(violated),
        "respected_I": [b["I_target"] for b in respected],
        "violated_I": [(b["I_target"], b["overrun_factor"]) for b in violated],
        "phi_converged_I": [I for I, c in conv.items() if c["converged"]],
        "phi_not_converged_I": [I for I, c in conv.items() if not c["converged"]],
        "worst_overrun_factor": max((b["overrun_factor"] for b in budget),
                                    default=float("nan")),
        # a run may be quotable for phi and simultaneously useless for mu:
        # phi settles within a couple of strain units, the virial stress needs
        # an order of magnitude more.  Never adjudicate mu from these runs.
        "mu_converged_I": [I for I, c in conv_mu.items() if c["converged"]],
        "mu_usable_for_acceptance": False,
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def markdown_table(budget: list, ref: list) -> str:
    lines = [
        "| I_target | gamma_dot | planned floor | executed | recover steps "
        "| strain req | strain ach | overrun | budget respected |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for b in budget:
        lines.append(
            f"| {b['I_target']:g} | {b['gamma_dot']:.1f} | "
            f"{b['planned_floor_steps']} | {b['steps_executed']} | "
            f"{b['recover_steps_spent']} | {b['strain_requested']:.2f} | "
            f"{b['strain_achieved']:.2f} | {b['overrun_factor']:.2f}x | "
            f"{'yes' if b['respects_budget'] else '**NO**'} |")

    lines += [
        "",
        "| I_target | I_bulk | mu | mu_ref | mu err % | phi | phi_ref "
        "| phi err % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ref:
        lines.append(
            f"| {r['I_target']:g} | {r['I_used']:.5f} | {r['mu']:.4f} | "
            f"{r['mu_ref']:.4f} | {r['mu_err_pct']:+.1f} | {r['phi']:.4f} | "
            f"{r['phi_ref']:.4f} | {r['phi_err_pct']:+.1f} |")
    return "\n".join(lines)


def figures(runs: list, budget: list, ref: list, out_dir: str) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    made = []

    # --- Fig 1: strain budget accounting -----------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    Is = [b["I_target"] for b in budget]
    ax[0].loglog(Is, [b["planned_floor_steps"] for b in budget], "o-",
                 label="planned floor (requested strain)")
    ax[0].loglog(Is, [b["steps_executed"] for b in budget], "s--",
                 label="executed")
    ax[0].set_xlabel("inertial number I")
    ax[0].set_ylabel("shear steps")
    ax[0].set_title("(a) shear-step budget vs actual")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[1].axhline(STRAIN_TOL, color="r", lw=0.8, ls="--",
                  label=f"tolerance {STRAIN_TOL}x")
    ax[1].semilogx(Is, [b["overrun_factor"] for b in budget], "o-", color="C3")
    ax[1].set_xlabel("inertial number I")
    ax[1].set_ylabel(r"strain achieved / strain requested")
    ax[1].set_title("(b) strain-budget overrun")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=8)
    for x, b in zip(Is, budget):
        ax[1].annotate(f"{b['overrun_factor']:.2f}x", (x, b["overrun_factor"]),
                       textcoords="offset points", xytext=(4, 6), fontsize=7)
    fig.suptitle("Round 16 - the requested strain is not delivered above I=0.01")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_strain_budget.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    # --- Fig 2: phi(I) and mu(I) vs reference ------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    import numpy as np
    grid = np.logspace(-3.1, -0.4, 200)
    from reference_law import ref_mu, ref_phi
    for a, fn, col, lab in ((ax[0], ref_phi, "phi", "phi"), (ax[1], ref_mu, "mu", "mu")):
        a.plot(grid, fn(grid), "k-", lw=1.6, label="reference law (unverified)")
        a.set_xscale("log")
        a.set_xlabel("inertial number I (bulk)")
        a.set_ylabel(lab)
        a.grid(True, which="both", alpha=0.3)

    for r, b in zip(ref, budget):
        ok = b["respects_budget"]
        sty = dict(marker="o" if ok else "s", ls="none",
                   mfc="C0" if ok else "none", mec="C0" if ok else "C3",
                   label="strain respected" if ok else "strain OVERRUN")
        ax[0].plot(r["I_used"], r["phi"], **sty)
        ax[1].plot(r["I_used"], r["mu"], **sty)
    ax[0].set_title("(a) phi(I), container definition")
    ax[1].set_title("(b) mu(I), full-cell virial")
    for a in ax:
        h, l = a.get_legend_handles_labels()
        seen = {}
        for hh, ll in zip(h, l):
            seen.setdefault(ll, hh)
        a.legend(seen.values(), seen.keys(), fontsize=8)
    fig.suptitle("Round 16 - measured vs reference (open/filled = budget violated/respected)")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_phi_mu_vs_reference.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    # --- Fig 3: convergence of phi_extent along the shear series ------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in runs:
        s = r.get("series") or {}
        st, ph = s.get("strain"), s.get("phi_extent")
        if not st or not ph:
            continue
        b = strain_budget(r)
        ax.plot(st, ph, lw=1.2,
                label=f"I={r['I_target']:g} (strain {b['strain_achieved']:.1f})",
                ls="-" if b["respects_budget"] else ":")
    ax.set_xscale("log")
    ax.set_xlabel("accumulated shear strain")
    ax.set_ylabel("phi_extent")
    ax.set_title("Round 16 - does phi settle? (dotted = budget violated)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_phi_convergence.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=RUN_DIR)
    ap.add_argument("--out-dir", default=os.path.join(RUN_DIR, "..", ".."))
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--json-out", default=os.path.join(
        RUN_DIR, "strain_convergence_audit.json"))
    args = ap.parse_args(argv)

    runs = load_runs(args.run_dir)
    if not runs:
        raise SystemExit(f"no archived strain-sweep runs in {args.run_dir}")

    budget = budget_table(runs)
    ref = reference_compare(runs)
    v = verdict(runs)

    print("== Q1: strain budget ==")
    for b in budget:
        flag = "OK " if b["respects_budget"] else "OVER"
        print(f"  I={b['I_target']:6.3f} {flag} planned={b['planned_floor_steps']:6d}"
              f" executed={b['steps_executed']:6d} recover={b['recover_steps_spent']:6d}"
              f" strain {b['strain_requested']:.2f}->{b['strain_achieved']:7.2f}"
              f" ({b['overrun_factor']:.2f}x) bookkeeping_err={b['bookkeeping_error']:.2e}")
    print("\n== Q2/Q3: convergence and reference comparison (I_bulk) ==")
    for r, b in zip(ref, budget):
        run = next(x for x in runs if x["I_target"] == r["I_target"])
        con = convergence(run)
        con_mu = convergence(run, key="full_mu")
        print(f"  I={r['I_target']:6.3f} phi={r['phi']:.4f}"
              f" ({r['phi_err_pct']:+.1f}%) mu={r['mu']:.4f}"
              f" ({r['mu_err_pct']:+.1f}%) phi_drift={con['drift_pct']:+.2f}%"
              f"/term {con['terminal_drift_pct']:+.2f}%"
              f" | mu_term {con_mu['terminal_drift_pct']:+.2f}%"
              f" {'phi-quotable' if b['respects_budget'] and con['converged'] else 'NOT quotable'}")
    print(f"\nverdict: {ascii(v)}")

    report = {
        "round": 16,
        "mission": 20855,
        "source": "cluster jobs cj2950..cj2955 (round 15 strain-convergence sweep)",
        "budget": budget, "reference": ref, "verdict": v,
        "markdown_table": markdown_table(budget, ref),
    }
    with open(args.json_out, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print(f"\nwrote {args.json_out}")

    if args.figures:
        for p in figures(runs, budget, ref, os.path.join(HERE, "runs", "round16")):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())