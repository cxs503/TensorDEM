"""Run the analytic verification campaign of the three-field LBM core.

Writes ``artifacts/threefield_lbm_verification/results.json`` plus the figures in
``artifacts/threefield_lbm_verification/figures/``.  Every number in the campaign
report is read back from that JSON, nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python scripts/verify_threefield_lbm.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensordem.threefield import lbm_verification as V  # noqa: E402
from tensordem.threefield.wake_lbm import LBMConfig, WakeLBM  # noqa: E402

OUT = pathlib.Path("artifacts/threefield_lbm_verification")
TAU_SWEEP = (0.6, 0.8, 1.0, 1.5, 1.9)
CS_SWEEP = (0.05, 0.10, 0.16, 0.25)
STRAIN_SWEEP = (0.005, 0.01, 0.02, 0.04)


def viscosity_campaign(ny: int, mode: int, steps_cap: int) -> list[dict]:
    rows = []
    for tau0 in TAU_SWEEP:
        nu = (tau0 - 0.5) / 3.0
        k = V.shear_mode_number(ny, mode)
        steps = int(min(steps_cap, max(240, round(3.2 / (nu * k * k)))))
        res = V.run_shear_decay(
            tau0=tau0, ny=ny, mode=mode, steps=steps, sample_every=max(10, steps // 40)
        )
        res.pop("final_profile")
        rows.append(res)
        print(
            f"  tau0={tau0:.3f} nu_th={nu:.5f} nu_meas={res['nu_measured']:.5f} "
            f"rel={100 * res['nu_relative_error']:+.3f}% steps={steps}"
        )
    return rows


def closure_campaign() -> list[dict]:
    rows = []
    for cs in CS_SWEEP:
        solver = V.shear_solver(smagorinsky_cs=cs)
        for strain in STRAIN_SWEEP:
            out = V.chapman_enskog_field(solver, strain)
            out.pop("field")
            out["nu_eddy_relative_error"] = (
                (out["nu_eddy_measured"] - out["nu_eddy_theory"]) / out["nu_eddy_theory"]
                if out["nu_eddy_theory"]
                else 0.0
            )
            out["nu_eddy_over_cs2_norm"] = (
                out["nu_eddy_measured"] / (cs * cs * out["strain_norm"]) if cs else 0.0
            )
            rows.append(out)
    return rows


def dynamic_campaign(ny: int, mode: int, steps: int) -> dict:
    """LBM with an active Smagorinsky closure versus the 1-D closure model."""
    cs = 0.16
    res = V.run_shear_decay(
        tau0=0.8, smagorinsky_cs=cs, ny=ny, mode=mode, steps=steps, sample_every=10
    )
    res.pop("final_profile")
    reference = V.shear_decay_reference_1d(
        ny=ny, tau0=0.8, smagorinsky_cs=cs, mode=mode, steps=steps, substeps=40
    )
    lbm = res["amplitudes"]
    n = min(len(lbm), len(reference))
    tail = slice(max(0, n // 2), n)
    lbm_tail = lbm[tail]
    ref_tail = reference[tail]
    denom = max(abs(a) for a in ref_tail)
    res["reference_amplitudes"] = reference
    res["tail_max_deviation_rel"] = max(
        abs(a - b) for a, b in zip(lbm_tail, ref_tail)
    ) / denom
    res["lbm_final_over_initial"] = lbm[n - 1] / lbm[0]
    res["reference_final_over_initial"] = reference[n - 1] / reference[0]
    res["decay_ratio_lbm_over_reference"] = (
        res["lbm_final_over_initial"] / res["reference_final_over_initial"]
    )
    # Molecular control: the same run with the closure switched off.
    control = V.run_shear_decay(
        tau0=0.8, smagorinsky_cs=0.0, ny=ny, mode=mode, steps=steps, sample_every=10
    )
    control.pop("final_profile")
    res["control_amplitudes"] = control["amplitudes"]
    res["control_final_over_initial"] = control["amplitudes"][-1] / control["amplitudes"][0]
    return res


def profile_shape(args) -> dict:
    """Resolved profile versus ``U cos(k y)`` at the last sample."""
    solver = V.shear_solver(ny=args.ny, tau0=0.8)
    k = V.initialise_shear_wave(solver, 0.02, args.mode)
    zero = torch.zeros(3, dtype=solver.dtype)
    solver.step(200, body_force=zero)
    state = V.shear_profiles(solver, args.mode)
    y = torch.arange(args.ny, dtype=solver.dtype)
    analytic = state["amplitude"] * torch.cos(k * y)
    return {
        "y": y.tolist(),
        "profile": state["profile"].tolist(),
        "analytic": analytic.tolist(),
        "amplitude": state["amplitude"],
        "max_deviation_rel_initial_amplitude": float(
            (state["profile"] - analytic).abs().max().item()
        )
        / 0.02,
    }


def figures(results: dict, outdir: pathlib.Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    made = []

    # fig1: decay histories
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for row in results["viscosity"]:
        nu = row["nu_theory"]
        k = row["wavenumber"]
        t = [v for v in row["times"]]
        a = [abs(v) for v in row["amplitudes"]]
        ax.semilogy([ti / 500.0 for ti in t], a, "o", ms=3, label=f"solver tau0={row['tau0']:.2f}")
        ax.semilogy(
            [ti / 500.0 for ti in t],
            [row["amplitude"] * math.exp(-nu * k * k * ti) for ti in t],
            "-",
            lw=1.0,
            color=ax.lines[-1].get_color(),
            label=f"exp(-nu k^2 t) nu={nu:.4f}",
        )
    ax.set_xlabel("t / 500 (lattice steps / 500)")
    ax.set_ylabel("|u_x| cosine amplitude")
    ax.set_title("Decaying shear wave: solver vs exact NS solution")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    made.append(_save(fig, outdir, "fig1_shear_wave_decay.png",
                      "Decaying shear wave histories with the exact exponential."))

    # fig2: nu measured vs theory
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    th = [r["nu_theory"] for r in results["viscosity"]]
    me = [r["nu_measured"] for r in results["viscosity"]]
    ax.loglog(th, me, "o", label="measured")
    lim = [min(th) * 0.7, max(th) * 1.4]
    ax.loglog(lim, lim, "--", color="k", lw=1, label="nu = (tau0-0.5)/3")
    for x, y, r in zip(th, me, results["viscosity"]):
        ax.annotate(f"{r['tau0']:.2f}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7)
    ax.set_xlabel("theory nu = (tau0 - 0.5)/3")
    ax.set_ylabel("measured nu = rate / k^2")
    ax.set_title("Viscosity realised by the BGK kernel")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    made.append(_save(fig, outdir, "fig2_viscosity_recovery.png",
                      "Measured versus theoretical kinematic viscosity."))

    # fig3: Smagorinsky closure
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for cs in sorted({r["smagorinsky_cs"] for r in results["closure"]}):
        rows = [r for r in results["closure"] if r["smagorinsky_cs"] == cs]
        x = [r["strain_norm"] for r in rows]
        ax.plot(x, [r["nu_eddy_measured"] for r in rows], "o", label=f"kernel Cs={cs}")
        xs = [0.001 * i for i in range(1, 61)]
        ax.plot(xs, [V.analytic_eddy_viscosity(0.8, cs, s) for s in xs], "-", lw=1,
                color=ax.lines[-1].get_color())
    ax.set_xlabel("|S| = sqrt(2 S_ab S_ab)")
    ax.set_ylabel("eddy viscosity nu_t")
    ax.set_title("Smagorinsky closure: shipped kernel vs analytic value")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    made.append(_save(fig, outdir, "fig3_smagorinsky_closure.png",
                      "Eddy viscosity from the shipped closure versus the analytic value."))

    # fig4: impulse audit
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    taus = [0.5006, 0.7, 1.0, 1.5, 2.0]
    ratio = [r["ratio_measured_over_expected"] for r in results["impulse_pre_fix"]]
    ax.plot(taus, ratio, "s-", label="pre-fix (measured)")
    ax.plot(taus, [1.0 for _ in taus], "o-", label="post-fix (measured)")
    ax.plot(taus, [1.0 + 0.5 / t for t in taus], "--", lw=1, color="k",
            label="unguarded prediction 1 + 1/(2 tau)")
    ax.set_xlabel("tau0")
    ax.set_ylabel("injected momentum / sum(F)")
    ax.set_title("Guo forcing impulse identity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    made.append(_save(fig, outdir, "fig4_forcing_impulse.png",
                      "Momentum injected per unit booked impulse before and after the fix."))

    # fig5: dynamic closure versus 1-D model
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    dyn = results["dynamic"]
    t = [i * dyn["sample_every"] for i in range(len(dyn["amplitudes"]))]
    ax.plot(t, dyn["amplitudes"], "o", ms=3, label="solver, Cs=0.16")
    ax.plot(t[: len(dyn["reference_amplitudes"])],
            dyn["reference_amplitudes"][: len(t)], "-", lw=1.2,
            label="1-D Smagorinsky model")
    ax.plot(t, dyn["control_amplitudes"], "^", ms=3, label="solver, Cs=0 (molecular)")
    ax.set_xlabel("lattice steps")
    ax.set_ylabel("cosine amplitude")
    ax.set_title("Dynamic Smagorinsky decay vs the 1-D closure model")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    made.append(_save(fig, outdir, "fig5_dynamic_closure.png",
                      "Closure-active decay against the 1-D nonlinear-diffusion reference."))

    # fig6: profile shape
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ps = results["profile_shape"]
    ax.plot(ps["y"], ps["profile"], "o", ms=4, label="solver u_x(y)")
    ax.plot(ps["y"], ps["analytic"], "-", lw=1.2, label="U cos(k y)")
    ax.set_xlabel("y (lattice node)")
    ax.set_ylabel("u_x")
    ax.set_title("Shear-wave shape after 200 steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    made.append(_save(fig, outdir, "fig6_shear_profile.png",
                      "Resolved shear-wave profile against the cosine mode."))
    return made


def _save(fig, outdir: pathlib.Path, name: str, caption: str) -> dict:
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return {"name": name, "path": str(path), "caption": caption}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short records for smoke runs")
    args = ap.parse_args()
    args.ny = 48
    args.mode = 2
    steps_cap = 500 if args.quick else 1600
    OUT.mkdir(parents=True, exist_ok=True)

    print("viscosity recovery")
    viscosity = viscosity_campaign(args.ny, args.mode, steps_cap)
    print("Smagorinsky closure")
    closure = closure_campaign()
    print("forcing impulse audit")
    impulse_post = [V.momentum_impulse_audit(tau0=t) for t in (0.5006, 0.7, 1.0, 1.5, 2.0)]
    # The pre-fix numbers are the analytic prediction 1 + 1/(2 tau0), reproduced
    # numerically by the unguarded forcing term in docs/threefield/lbm_core_verification.md.
    impulse_pre = [
        {"tau0": r["tau0"], "ratio_measured_over_expected": 1.0 + 0.5 / r["tau0"]}
        for r in impulse_post
    ]
    print("boundary audit")
    bc = V.boundary_audit()
    print("dynamic closure")
    dynamic = dynamic_campaign(args.ny, 1, 400 if args.quick else 800)
    print("profile shape")
    shape = profile_shape(args)

    results = {
        "module": "tensordem.threefield.wake_lbm",
        "protocol": "tensordem-threefield-lbm-analytic-verification-v1",
        "lattice": {"model": "D3Q19", "cs2": 1.0 / 3.0, "dtype": "float64"},
        "viscosity": viscosity,
        "closure": closure,
        "impulse_post_fix": impulse_post,
        "impulse_pre_fix": impulse_pre,
        "boundary_audit": bc,
        "dynamic": dynamic,
        "profile_shape": shape,
    }
    results["figures"] = figures(results, OUT / "figures")
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    worst_nu = max(abs(r["nu_relative_error"]) for r in viscosity)
    worst_shape = max(r["shape_error_rel_amplitude"] for r in viscosity)
    worst_drift = max(max(r["mass_drift_rel"], r["momentum_drift_rel"]) for r in viscosity)
    lines = ["| tau0 | nu theory | nu measured | rel. error | profile err | mass/momentum drift | R^2 |",
             "|---|---|---|---|---|---|---|"]
    for r in viscosity:
        lines.append(
            f"| {r['tau0']:.3f} | {r['nu_theory']:.6f} | {r['nu_measured']:.6f} | "
            f"{100*r['nu_relative_error']:+.3f}% | {100*r['shape_error_rel_amplitude']:.2e}% | "
            f"{max(r['mass_drift_rel'], r['momentum_drift_rel']):.1e} | {r['decay_fit']['r2']:.6f} |"
        )
    lines += ["", "| Cs | strain |S| | nu_t kernel | nu_t analytic | rel. error | nu_t/(Cs^2|S|) |",
              "|---|---|---|---|---|---|"]
    for r in closure:
        lines.append(
            f"| {r['smagorinsky_cs']:.2f} | {r['strain_norm']:.3f} | "
            f"{r['nu_eddy_measured']:.9f} | {r['nu_eddy_theory']:.9f} | "
            f"{r['nu_eddy_relative_error']:+.2e} | {r['nu_eddy_over_cs2_norm']:.6f} |"
        )
    lines += ["", "| tau0 | injected/booked | booked/expected | unguarded prediction 1+1/(2tau) |",
              "|---|---|---|---|"]
    for r in impulse_post:
        lines.append(
            f"| {r['tau0']:.4f} | {r['ratio_measured_over_booked']:.12f} | "
            f"{r['ratio_booked_over_expected']:.12f} | {r['unguarded_prefactor']:.6f} |"
        )
    lines += ["", f"boundary audit: max population deviation {bc['max_population_deviation']:.3e} "
              f"(relative {bc['max_population_deviation_rel']:.3e}), uniform-stream streaming "
              f"deviation {bc['streaming_max_deviation']:.3e}; "
              f"dynamic closure tail deviation vs 1-D model {100*dynamic['tail_max_deviation_rel']:.2f}%, "
              f"decay ratio LBM/reference {dynamic['decay_ratio_lbm_over_reference']:.4f}."]
    (OUT / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsummary: worst viscosity error {100 * worst_nu:.4f}%  "
          f"worst profile error {100 * worst_shape:.4f}%  "
          f"worst mass/momentum drift {worst_drift:.2e}")
    print(f"          impulse ratio post-fix {min(r['ratio_measured_over_expected'] for r in impulse_post):.12f}"
          f"..{max(r['ratio_measured_over_expected'] for r in impulse_post):.12f}")
    print(f"          written to {OUT/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())