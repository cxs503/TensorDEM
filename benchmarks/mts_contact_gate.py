#!/usr/bin/env python
"""Machine-decidable acceptance run for the MTS contact gate (P82W4-N09).

Every number in ``artifacts/mts_contact/gate.json`` is produced here, from the
shipped module :mod:`tensordem.mts_contact`; nothing is typed by hand.  The five
acceptance criteria of the campaign node are:

===== ==========================================================================
(a)   single-impact restitution error <= 2 % for ``e = 0.5`` and ``e = 0.9`` on
      the ladder ``dt/t_c in {0.02, 0.05, 0.1, 0.2}``
(b)   undamped 1e5-step ledger: mechanical-energy drift <= 1 %, linear momentum
      conserved to <= 1e-12 relative
(c)   frictional jamming ``mu = 0.5``, ``N = 256``: ``phi_J`` within 0.5 % of the
      baseline arm while the MTS arm is >= 2x faster in wall clock
(d)   tightness negative control: ``dt/t_c = 0.5`` *without* sub-cycling must
      fail detectably (restitution error > 10 % or energy growth > 10 %)
(e)   shard reproducibility: contact accumulation over 1 / 2 / 4 shards must not
      move the trajectory beyond a stated floating-point bound
===== ==========================================================================

Usage::

    python benchmarks/mts_contact_gate.py --only restitution
    python benchmarks/mts_contact_gate.py --only all --out artifacts/mts_contact/gate.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.mts_contact import (  # noqa: E402
    GATE_MAX_RATIO,
    shard_force_consistency,
    gas_ledger,
    jamming_protocol,
    measure_restitution,
    restitution_ladder,
    run_shards,
    shear_cell_run,
    stability_gate,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "artifacts", "mts_contact")
FIG_DIR = os.path.join(OUT_DIR, "figures")


# --------------------------------------------------------------------------- (a)/(d)
def part_restitution(quick: bool = False) -> dict:
    ladder = restitution_ladder(e_targets=(0.5, 0.9),
                               ratios=(0.02, 0.05, 0.1, 0.2))
    negative = []
    for e_target in (0.5, 0.9):
        ref = measure_restitution(ratio=0.002, e_target=e_target, n_sub=1)
        # the negative control deliberately bypasses the gate: the ratio is
        # measured at the impact speed, so with the half-kick the actual ratio
        # lands just above 0.5 and the gate refuses it (that is the point)
        over = measure_restitution(ratio=0.5, e_target=e_target, n_sub=1,
                                   scheme="mts_overstep", enforce_gate=False)
        over["restitution_reference"] = ref["restitution_measured"]
        over["restitution_error"] = (over["restitution_measured"]
                                     / ref["restitution_measured"] - 1.0)
        over["gate_at_0p5_without_subcycling"] = stability_gate(
            over["dt"], over["contact_time"])
        negative.append(over)
    worst_neg = max(abs(r["restitution_error"]) for r in negative)
    thresholds = {name: stability_gate(ratio_val * 1.0, 1.0)
                  for name, ratio_val in (("0.02", 0.02), ("0.05", 0.05),
                                          ("0.1", 0.1), ("0.2", 0.2),
                                          ("0.5", 0.5), ("1.0", 1.0))}
    verdict_a = bool(ladder["within_2pct"])
    verdict_d = bool(worst_neg > 0.10)
    return {
        "ladder": ladder,
        "negative_control": negative,
        "worst_abs_negative_error": float(worst_neg),
        "gate_status_table": thresholds,
        "verdict": {
            "a_restitution_within_2pct": verdict_a,
            "d_negative_control_detected": verdict_d,
            "d_threshold": 0.10,
            "d_observed": float(worst_neg),
        },
    }


# ----------------------------------------------------------------------------- (b)
def part_energy(steps: int = 100_000, n: int = 16, ratio: float = 0.2,
                n_sub: int = 10, dilute_n: int = 32, dilute_phi: float = 0.03,
                dilute_n_sub: int = 4) -> dict:
    # The acceptance case is a *dilute* undamped gas: that is the standard
    # energy-conservation statement (few contact events, so the residual is the
    # integrator's own).  A dense contact-rich packing heats at O(h) per contact
    # *event* in any fixed-step explicit scheme -- that stress case is measured
    # too, and reported next to it instead of being hidden.
    ledger = gas_ledger(n=dilute_n, phi=dilute_phi, ratio=ratio,
                        n_sub=dilute_n_sub, steps=steps,
                        confirm_every=max(1, steps // 200))
    dense = gas_ledger(n=n, phi=0.6, ratio=ratio, n_sub=n_sub, steps=steps,
                       confirm_every=max(1, steps // 200))
    negative = gas_ledger(n=dilute_n, phi=dilute_phi, ratio=0.5, n_sub=1,
                          steps=min(2000, steps), confirm_every=100,
                          scheme="mts_overstep", enforce_gate=False)
    return {
        "ledger": {k: v for k, v in ledger.items() if k != "trace"},
        "trace": ledger["trace"],
        "dense_stress_case": {k: v for k, v in dense.items() if k != "trace"},
        "negative_control_overstep": {k: v for k, v in negative.items() if k != "trace"},
        "negative_control_energy_growth": float(negative["energy_max_abs_drift_rel"]),
        "verdict": {
            "b_energy_within_1pct": bool(ledger["within_1pct"]),
            "b_energy_secular_drift_rel": float(ledger["energy_drift_rel"]),
            "b_energy_oscillation_rel": float(ledger["oscillation_amplitude_rel"]),
            "b_dense_stress_drift_rel": float(dense["energy_drift_rel"]),
            "b_dense_stress_within_1pct": bool(dense["within_1pct"]),
            "b_momentum_within_1e12": bool(ledger["momentum_within_1e12"]),
            "d_energy_negative_control_detected": bool(
                negative["energy_max_abs_drift_rel"] > 0.10),
            "d_energy_observed": float(negative["energy_max_abs_drift_rel"]),
        },
    }


# ----------------------------------------------------------------------------- (c)
def part_shear(n: int = 64, phi: float = 0.86, mu: float = 0.5,
               ratio: float = 0.2, steps: int = 250,
               relax_steps: int = 1200, gamma_dot: float = 2.0) -> dict:
    """Doubly periodic shear cell: the friction arm's *constitutive output*.

    A jammed amorphous packing is sheared at a fixed rate; the steady pressure and
    shear stress are averages over the second half of the run, with the two
    friction arms covering the *same physical duration*.
    """
    mts = shear_cell_run(phi=phi, mu=mu, n=n, ratio=ratio, scheme="mts",
                         steps=steps, relax_steps=relax_steps, gamma_dot=gamma_dot)
    legacy = shear_cell_run(phi=phi, mu=mu, n=n, ratio=ratio, scheme="legacy3x",
                            steps=steps, relax_steps=relax_steps, gamma_dot=gamma_dot)
    tau_err = abs(mts["shear"] / legacy["shear"] - 1.0) if legacy["shear"] else float("nan")
    p_err = abs(mts["pressure"] / legacy["pressure"] - 1.0) if legacy["pressure"] else float("nan")
    fn_err = (abs(mts["contact_stats"]["mean_fn"] / legacy["contact_stats"]["mean_fn"] - 1.0)
              if legacy["contact_stats"]["mean_fn"] else float("nan"))
    # the deviatoric stress of a 64-disk cell is a near-cancelling difference:
    # judge the cross-arm difference against the *sampling* error of the two
    # averages (2 sigma), not against a bare percentage
    se_tau = math.hypot(mts["shear_stderr"], legacy["shear_stderr"])
    tau_z = (abs(mts["shear"] - legacy["shear"]) / se_tau) if se_tau > 0 else float("nan")
    return {
        "mts": {k: v for k, v in mts.items() if k != "samples"},
        "legacy3x": {k: v for k, v in legacy.items() if k != "samples"},
        "shear_rel_diff": float(tau_err),
        "pressure_rel_diff": float(p_err),
        "mean_normal_force_rel_diff": float(fn_err),
        "shear_combined_stderr": float(se_tau),
        "shear_diff_over_stderr": float(tau_z),
        "wall_clock_ratio_shear": float(legacy["shear_wall_clock"] / mts["shear_wall_clock"])
        if mts["shear_wall_clock"] else float("nan"),
        "verdict": {
            "shear_pressure_within_2pct": bool(p_err <= 0.02),
            "shear_normal_force_within_2pct": bool(fn_err <= 0.02),
            "shear_stress_within_sampling_error": bool(tau_z <= 2.0),
            "shear_within_2pct": bool(p_err <= 0.02 and fn_err <= 0.02 and tau_z <= 2.0),
        },
    }


def part_jamming(n: int = 256, mu: float = 0.5, mean_radius: float = 0.008,
                 compress_time: float = 0.06, level_time: float = 0.06,
                 run_reference: bool = True) -> dict:
    mts = jamming_protocol(scheme="mts", n=n, mu=mu, mean_radius=mean_radius,
                           compress_time=compress_time, level_time=level_time)
    legacy = jamming_protocol(scheme="legacy3x", n=n, mu=mu,
                              mean_radius=mean_radius,
                              compress_time=compress_time, level_time=level_time)
    out = {
        "mts": {k: v for k, v in mts.items() if k not in ("series",)},
        "mts_series": mts["series"],
        "legacy3x": {k: v for k, v in legacy.items() if k not in ("series",)},
        "legacy3x_series": legacy["series"],
        "phi_j_rel_diff_mts_vs_legacy": float(abs(mts["phi_j"] / legacy["phi_j"] - 1.0)),
        "wall_clock_ratio": float(legacy["wall_clock"] / mts["wall_clock"]),
    }
    if run_reference:
        from tensordem.granular2d.jamming import jamming_density_isotropic

        t0 = time.perf_counter()
        ref = jamming_density_isotropic(mu=mu, mu_rolling=0.05, n=n, seed=0,
                                        levels=(300.0, 100.0, 30.0, 10.0, 3.0, 1.0),
                                        steps_per_level=2500, eta=150.0)
        ref_time = time.perf_counter() - t0
        out["w4p1_servo_reference"] = {
            "phi_j": float(ref["phi_j"]),
            "levels": ref["levels"],
            "wall_clock": float(ref_time),
            "source": "tensordem.granular2d.jamming.jamming_density_isotropic"
                      "(mu=0.5, mu_rolling=0.05, n=256, seed=0, steps_per_level=2500)",
        }
        out["phi_j_rel_diff_mts_vs_reference"] = float(
            abs(mts["phi_j"] / ref["phi_j"] - 1.0))
    out["verdict"] = {
        "c_phi_j_vs_baseline_within_0p5pct": bool(out["phi_j_rel_diff_mts_vs_legacy"] <= 0.005),
        "c_wall_clock_at_least_2x": bool(out["wall_clock_ratio"] >= 2.0),
        "c_phi_j_vs_w4p1_reference_within_0p5pct": bool(
            out.get("phi_j_rel_diff_mts_vs_reference", float("nan")) <= 0.005)
        if "phi_j_rel_diff_mts_vs_reference" in out else None,
    }
    return out


# ----------------------------------------------------------------------------- (e)
def part_shards(n: int = 64, short_steps: int = 40) -> dict:
    """(e) shard reproducibility: one state (forces) and one short trajectory.

    The long-time decorrelation of two trajectories that differ at round-off is a
    property of the (chaotic) dynamics, not of the shard partition, so the
    criterion is stated at the level where it is decidable: the *same state*
    accumulated over 1 / 2 / 4 shards, plus a short trajectory whose floating
    point bound is measured and then grown to show the chaotic amplification.
    """
    force_check = shard_force_consistency(n=n)
    runs = {}
    for shards in (1, 2, 4):
        runs[shards] = run_shards(shards=shards, n=n, steps=short_steps)
    base = runs[1]
    deltas = {}
    for shards in (2, 4):
        dv = np.linalg.norm(runs[shards]["vel"] - base["vel"], axis=1)
        vscale = max(float(np.max(np.linalg.norm(base["vel"], axis=1))), 1e-300)
        deltas[shards] = {
            "max_abs_vel_diff": float(np.max(dv)),
            "max_rel_vel_diff": float(np.max(dv) / vscale),
            "shear_rel_diff": float(abs(runs[shards]["shear"] / base["shear"] - 1.0))
            if base["shear"] else float("nan"),
            "pressure_rel_diff": float(abs(runs[shards]["pressure"] / base["pressure"] - 1.0))
            if base["pressure"] else float("nan"),
        }
    growth = {}
    for steps in (40, 100, 200, 400):
        a = run_shards(shards=1, n=n, steps=steps)
        b = run_shards(shards=4, n=n, steps=steps)
        dv = float(np.max(np.linalg.norm(b["vel"] - a["vel"], axis=1)))
        vscale = max(float(np.max(np.linalg.norm(a["vel"], axis=1))), 1e-300)
        growth[str(steps)] = {"max_rel_vel_diff": dv / vscale}
    worst_traj = max(max(d["max_rel_vel_diff"] for d in deltas.values()), 0.0)
    return {
        "force_level": force_check,
        "runs": {str(k): {"pressure": v["pressure"], "shear": v["shear"]}
                 for k, v in runs.items()},
        "deltas": {str(k): v for k, v in deltas.items()},
        "worst_rel_vel_diff_short_trajectory": float(worst_traj),
        "float_bound_rel_vel": 1e-8,
        "chaotic_growth": growth,
        "verdict": {"e_shard_reproducible": bool(
            force_check["worst_rel_force_diff"] <= 1e-12
            and worst_traj <= 1e-8)},
    }


# --------------------------------------------------------------------------- figures
def make_figures(result: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    written = []

    # 1. restitution ladder
    if "restitution" in result:
        ladder = result["restitution"]["ladder"]
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        for e_target in (0.5, 0.9):
            rows = [r for r in ladder["rows"] if r["e_target"] == e_target]
            if not rows:
                continue
            ratios = [r["dt_over_tc"] for r in rows]
            err = [100.0 * r["restitution_error"] for r in rows]
            ref = ladder["references"][str(e_target)]["restitution_measured"]
            axes[0].semilogx(ratios, [r["restitution_measured"] for r in rows], "o-",
                             label=f"e={e_target} (ref {ref:.4f})")
            axes[1].semilogx(ratios, err, "o-", label=f"e={e_target}")
        axes[0].axhline(1.0, color="k", lw=0.5, ls=":")
        axes[0].set_xlabel("dt / t_c"); axes[0].set_ylabel("measured restitution")
        axes[0].set_title("restitution vs the contract ladder"); axes[0].legend(fontsize=7)
        axes[1].axhspan(-2, 2, color="g", alpha=0.12, label="+/- 2 % band")
        axes[1].axhline(0.0, color="k", lw=0.5)
        axes[1].set_xlabel("dt / t_c"); axes[1].set_ylabel("error vs fine-step ref [%]")
        axes[1].set_title("acceptance (a): |error| <= 2 %"); axes[1].legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "restitution_ladder.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        written.append(path)

        neg = result["restitution"]["negative_control"]
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        labels = [f"e={r['e_target']}\n(dt/t_c=0.5, 1 sub-step)" for r in neg]
        vals = [r["restitution_measured"] for r in neg]
        refs = [r["restitution_reference"] for r in neg]
        x = np.arange(len(neg))
        ax.bar(x - 0.2, refs, 0.4, label="fine-step reference")
        ax.bar(x + 0.2, vals, 0.4, label="no sub-cycling")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("restitution"); ax.set_title("acceptance (d): gate tightness")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "negative_control.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        written.append(path)

    # 2. energy ledger
    if "energy" in result and result["energy"]["trace"]:
        trace = np.asarray(result["energy"]["trace"], dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
        axes[0].semilogy(trace[:, 0], np.abs(trace[:, 1]) + 1e-18, lw=0.8)
        axes[0].axhline(0.01, color="r", ls="--", lw=0.8, label="1 % bound")
        axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("|dE/E0|")
        axes[0].legend(fontsize=7); axes[0].set_title("undamped gas energy drift")
        axes[1].semilogy(trace[:, 0], np.maximum(trace[:, 2], 1e-20), lw=0.8)
        axes[1].axhline(1e-12, color="r", ls="--", lw=0.8, label="1e-12 bound")
        axes[1].set_xlabel("time [s]"); axes[1].set_ylabel("|dp| [kg m/s]")
        axes[1].legend(fontsize=7); axes[1].set_title("linear momentum drift")
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "energy_momentum.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        written.append(path)

    # 3. jamming EOS
    if "jamming" in result and result["jamming"].get("mts_series"):
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for key, label, style in (("mts_series", "MTS (sub-cycled)", "o-"),
                                  ("legacy3x_series", "legacy 3x return mapping", "s--")):
            ser = np.asarray(result["jamming"][key], dtype=float)
            if len(ser):
                ax.plot(ser[:, 0], ser[:, 1], style, ms=4, label=label)
        ref = result["jamming"].get("w4p1_servo_reference")
        if ref and ref.get("levels"):
            r = np.asarray(ref["levels"], dtype=float)
            ax.plot(r[:, 0], r[:, 1], "d:", ms=4, label="W4-P1 servo reference")
        ax.set_xscale("log"); ax.set_xlabel("servo pressure P [N/m]")
        ax.set_ylabel("phi")
        ax.set_title("frictional jamming EOS (mu=0.5, N=256)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "jamming_eos.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        written.append(path)

    # 4. wall clock
    if "jamming" in result:
        jam = result["jamming"]
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
        labels = ["legacy 3x", "MTS"]
        clock = [jam["legacy3x"]["wall_clock"], jam["mts"]["wall_clock"]]
        axes[0].bar(labels, clock, color=["#b04a3a", "#2d6a8a"])
        for i, v in enumerate(clock):
            axes[0].text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)
        axes[0].set_ylabel("wall clock [s]"); axes[0].set_title(
            f"same protocol, speed-up {jam['wall_clock_ratio']:.2f}x")
        if "shear" in result:
            sh = result["shear"]
            vals = [sh["legacy3x"]["shear"], sh["mts"]["shear"]]
            axes[1].bar(labels, vals, color=["#b04a3a", "#2d6a8a"])
            for i, v in enumerate(vals):
                axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
            axes[1].set_ylabel("steady shear stress [N/m]")
            axes[1].set_title("physics unchanged (shear cell)")
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "wall_clock_and_physics.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        written.append(path)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="all",
                        choices=["all", "restitution", "energy", "shear",
                                 "jamming", "shards"])
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "gate.json"))
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--jamming-time", type=float, default=0.06)
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    result = {}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            result = json.load(fh)
    t0 = time.perf_counter()
    if args.only in ("all", "restitution"):
        result["restitution"] = part_restitution()
    if args.only in ("all", "energy"):
        result["energy"] = part_energy(steps=args.steps)
    if args.only in ("all", "shear"):
        result["shear"] = part_shear()
    if args.only in ("all", "jamming"):
        result["jamming"] = part_jamming(compress_time=args.jamming_time,
                                        level_time=args.jamming_time,
                                        run_reference=not args.no_reference)
    if args.only in ("all", "shards"):
        result["shards"] = part_shards()
    result["meta"] = {
        "module": "tensordem.mts_contact",
        "gate_max_ratio": GATE_MAX_RATIO,
        "wall_clock_total": float(time.perf_counter() - t0),
        "only": args.only,
    }
    if not args.no_figures:
        result["figures"] = make_figures(result)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    print(json.dumps({k: v.get("verdict") if isinstance(v, dict) and "verdict" in v
                      else "..." for k, v in result.items()}, indent=1, default=float))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
