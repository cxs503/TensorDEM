"""Driver for the P295W1-N03 study: NGF calibration on DEM data + NSCD tiers.

Usage
-----
    python -m tensordem.nonlocal_rheology.study runs   --out artifacts/nonlocal_rheology [--shard i --shards n] [--quick]
    python -m tensordem.nonlocal_rheology.study aggregate --out artifacts/nonlocal_rheology [--skip-nscd]

``runs`` writes one JSON per Couette case (shardable across cores/nodes),
``aggregate`` performs the two-scale calibration, the pooled scoring, the NSCD
verification tiers and the figures.  Everything numeric ends up in
``summary.json`` so the report can cite machine-produced numbers only.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .calibration import (RheologySample, fit_amplitude, fit_local_law,
                          ngf_predict, rate_floor, sample_from_profile, score)
from .couette import CouetteCase, simulate_couette
from .fluidity import JopLaw, NGFParams, RadialGrid, cg_jacobi, solve_ngf
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# experimental design (declared before any run)
# ---------------------------------------------------------------------------
GAPS = {"thin": 0.022, "wide": 0.045}     # outer radius; inner 0.010 -> 6 d / 17.5 d
OMEGAS = (15.0, 50.0, 150.0)
SEEDS = (1, 2)


def case_table(quick: bool = False):
    table = []
    for gap_name, r_out in GAPS.items():
        for omega in OMEGAS:
            for seed in SEEDS:
                name = f"{gap_name}_w{int(omega):03d}_s{seed}"
                kw = dict(r_inner=0.010, r_outer=r_out, omega=omega, seed=seed,
                          settle_strain=3.0, shear_strain=12.0, average_strain=5.0)
                if quick:
                    kw.update(settle_strain=1.5, shear_strain=4.0, average_strain=2.0,
                              n_bins=10)
                if gap_name == "wide" and not quick:
                    kw.update(settle_strain=2.0, shear_strain=6.0, average_strain=3.0)
                table.append((name, gap_name, CouetteCase(**kw)))
    return table


def run_cases(out: Path, shard: int = 0, shards: int = 1, quick: bool = False) -> dict:
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    table = case_table(quick)
    mine = [c for i, c in enumerate(table) if i % shards == shard]
    done = []
    for name, gap_name, case in mine:
        target = runs_dir / f"{name}.json"
        if target.exists():
            done.append(name)
            continue
        t0 = time.time()
        profile = simulate_couette(case)
        payload = {"name": name, "gap": gap_name, "seed": case.seed,
                   "case": {k: v for k, v in asdict(case).items()},
                   "elapsed_s": time.time() - t0}
        for key, value in profile.items():
            if key in ("case", "traces"):
                continue
            payload[key] = (np.asarray(value).tolist()
                            if isinstance(value, (np.ndarray, list, tuple)) else value)
        target.write_text(json.dumps(payload, indent=1))
        done.append(name)
        print(f"[run] {name} N={profile['n_particles']} steps={profile['steps']} "
              f"P={profile['pressure_mean']:.0f} ({time.time() - t0:.0f} s)", flush=True)
    return {"completed": done, "shard": shard, "shards": shards}


def load_samples(out: Path):
    samples = []
    for path in sorted((out / "runs").glob("*.json")):
        payload = json.loads(path.read_text())
        case_info = payload["case"]
        case = CouetteCase(**{k: v for k, v in case_info.items() if k != "dt"})
        profile = {k: np.asarray(v) for k, v in payload.items()
                   if isinstance(v, list)}
        profile["case"] = case
        profile["d50"] = payload["d50"]
        samples.append(sample_from_profile(profile, run=payload["name"],
                                           seed=int(payload["seed"])))
    if not samples:
        raise SystemExit("no runs found -- execute the 'runs' stage first")
    return samples


# ---------------------------------------------------------------------------
# solver evidence (CG convergence + manufactured solution)
# ---------------------------------------------------------------------------
def solver_evidence() -> dict:
    """CG/Jacobi convergence, dense-solve agreement and discretisation order."""
    law = JopLaw()
    grid = RadialGrid(0.010, 0.045, 60)
    mu = np.linspace(0.75, 0.36, 60)
    pressure = np.full(60, 500.0)
    params = NGFParams(law=law, amplitude=0.4)
    res = solve_ngf(mu, pressure, 0.002, 2500.0, params, grid)
    xi2 = params.xi(mu, 0.002) ** 2
    A = (sp.diags(grid.measure) + grid.operator(xi2)).toarray()
    dense = np.linalg.solve(A, grid.measure * res.g_loc)
    # manufactured solution on a 2D Cartesian grid (Neumann-free, Dirichlet rim)
    from .fluidity import cartesian_2d_operator
    orders = []
    for n in (21, 41, 81):
        h = 1.0 / (n - 1)
        K, measure, index = cartesian_2d_operator(n, n, h)
        X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
        exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        xi = 0.05
        src = (1.0 + 2.0 * np.pi ** 2 * xi ** 2) * exact
        A2 = (sp.diags(measure) + xi ** 2 * K).tocsr()
        active = np.ones((n, n), dtype=bool)
        active[0, :] = active[-1, :] = active[:, 0] = active[:, -1] = False
        free = index[active]
        Af = A2[free][:, free]
        fixed = np.ones(n * n, dtype=bool)
        fixed[free] = False
        rhs = measure[free] * src.ravel()[free] - A2[free][:, fixed] @ exact.ravel()[fixed]
        sol, rep = cg_jacobi(Af, rhs, rtol=1e-13)
        err = float(np.max(np.abs(sol - exact.ravel()[free])))
        orders.append({"n": n, "h": h, "linf": err, "cg_iterations": rep.iterations,
                       "cg_residual": rep.residual})
    for a, b in zip(orders[:-1], orders[1:]):
        a["ratio"] = a["linf"] / max(b["linf"], 1e-300)
    return {
        "annular_cg_iterations": res.iterations,
        "annular_cg_residual": res.residual,
        "annular_cg_converged": bool(res.converged),
        "annular_dense_agreement": float(np.max(np.abs(dense - res.g))
                                         / max(np.max(np.abs(dense)), 1e-300)),
        "manufactured": orders,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(out: Path, samples, fit, amp, scored, nscd) -> list:
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []

    def save(fig, name, caption):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        figs.append({"name": name, "caption": caption,
                     "png_base64": base64.b64encode(buf.getvalue()).decode()})

    law = fit["law"]
    params = NGFParams(law=law, amplitude=amp["amplitude"],
                       length_law=amp["length_law"])
    picks = []
    for s in samples:
        if len(picks) < 6 and (not picks or s.gap_d not in [p.gap_d for p in picks]
                               or s.omega not in [p.omega for p in picks]):
            picks.append(s)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for s in picks:
        pred = ngf_predict(s, law, params)
        m = s.valid()
        axes[0].semilogy(s.r[m] / s.d50, s.gamma_dot[m], "o", ms=3,
                         label=f"{s.gap_d:.0f}d w{s.omega:.0f}")
        axes[0].semilogy(s.r[m] / s.d50, np.maximum(pred["rate_loc"][m], 1e-6), ":", lw=1)
        axes[0].semilogy(s.r[m] / s.d50, np.maximum(pred["rate_nl"][m], 1e-6), "-", lw=1)
    axes[0].set_xlabel(r"$r/d$"); axes[0].set_ylabel(r"$\dot\gamma$ [1/s]")
    axes[0].set_title("DEM (o), local ':' , NGF '-'"); axes[0].legend(fontsize=6)
    xs, ys = [], []
    for s in samples:
        m = s.valid()
        xs.append(s.gamma_dot[m] * s.d50 / np.sqrt(s.pressure[m] / s.rho))
        ys.append(s.mu[m])
    axes[1].loglog(np.concatenate(xs), np.concatenate(ys), ".", ms=3)
    ii = np.logspace(-4, 0, 100)
    axes[1].loglog(ii, law.mu(ii), "k-", lw=1.2, label=r"$\mu(I)$ wide-band fit")
    axes[1].set_xlabel(r"$I$"); axes[1].set_ylabel(r"$\mu$"); axes[1].legend(fontsize=6)
    axes[1].set_title(r"pooled $\mu$ vs $I$")
    mu_r = np.linspace(law.mu_s + 1e-3, law.mu_2 - 0.02, 200)
    axes[2].plot(mu_r, params.xi(mu_r, samples[0].d50) / samples[0].d50, "-")
    axes[2].set_xlabel(r"$\mu$"); axes[2].set_ylabel(r"$\xi/d$")
    axes[2].set_title(f"calibrated A = {amp['amplitude']:.3f}")
    fig.tight_layout()
    save(fig, "rheology_profiles.png",
         "Radial strain-rate profiles: DEM (circles), local mu(I) prediction (dotted), "
         "NGF prediction (solid); pooled mu vs I with the wide-band local law; the "
         "calibrated cooperativity length.")

    if nscd:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        rows = nscd["tier1"]["nscd"]
        dts = sorted({r["dt"] for r in rows})
        for e in sorted({r["restitution_target"] for r in rows}):
            sel = [r for r in rows if r["restitution_target"] == e]
            axes[0].loglog([r["dt"] for r in sel], [max(r["restitution_error"], 1e-17) for r in sel],
                           "o-", label=f"e={e}")
            axes[2].loglog([r["dt"] for r in sel],
                           [max(r["bounce_gap_over_d"], 1e-9) for r in sel], "o-", label=f"e={e}")
        axes[0].set_xlabel("dt [s]"); axes[0].set_ylabel("|e_meas - e|")
        axes[0].set_title("restitution error"); axes[0].legend(fontsize=6)
        axes[2].set_xlabel("dt [s]"); axes[2].set_ylabel("bounce offset / d")
        axes[2].set_title("anticipated-contact offset (O(dt))")
        pen = nscd["tier1"]["penalty"]
        axes[1].loglog([max(r["max_penetration"], 1e-17) for r in rows],
                       [max(r["dt"], 1e-9) for r in rows], "o", label="NSCD")
        axes[1].loglog([max(p["max_penetration"], 1e-17) for p in pen],
                       [max(p["dt"], 1e-9) for p in pen], "s", label="penalty")
        axes[1].set_xlabel("max penetration [m]"); axes[1].set_ylabel("dt [s]")
        axes[1].set_title("penetration: NSCD vs penalty"); axes[1].legend(fontsize=6)
        fig.tight_layout()
        save(fig, "nscd_verification.png",
             "NSCD verification: restitution error vs dt, penetration comparison with "
             "the soft-sphere reference, and the O(dt) anticipated-contact offset.")
    return figs


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def aggregate(out: Path, skip_nscd: bool = False) -> dict:
    samples = load_samples(out)
    wide = [s for s in samples if s.gap_d >= 10.0]
    thin = [s for s in samples if s.gap_d < 10.0]
    fit = fit_local_law(samples)
    floor = rate_floor(samples)
    amp = fit_amplitude(thin, fit["law"], floor)
    params = NGFParams(law=fit["law"], amplitude=amp["amplitude"],
                       length_law=amp["length_law"])
    scored = score(samples, fit["law"], params, floor=floor)
    nscd = None
    if not skip_nscd:
        from .validation import tier1_refinement, tier2_chain, tier3_packing
        nscd = {}
        for key, fn in (("tier1", tier1_refinement), ("tier2", tier2_chain),
                        ("tier3", tier3_packing)):
            try:
                nscd[key] = fn()
            except Exception as exc:               # noqa: BLE001 - reported
                nscd[key] = {"error": f"{type(exc).__name__}: {exc}"}
    evidence = solver_evidence()
    summary = {
        "node": "P295W1-N03",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "runs": [{"run": s.run, "gap_d": s.gap_d, "omega": s.omega, "seed": s.seed,
                  "n_valid_bins": int(s.valid().sum())} for s in samples],
        "local_law": {"mu_s": fit["law"].mu_s, "mu_2": fit["law"].mu_2,
                      "i_0": fit["law"].i_0, "r2": fit["r2"],
                      "n_points": fit["n_points"], "i_cut": fit["i_cut"],
                      "rmse": fit["rmse"]},
        "amplitude": {"A_xi_over_d": amp["amplitude"], "length_law": amp["length_law"],
                      "rmse_log_rate": amp["rmse_log"],
                      "scan_amplitude": amp["scan"]["amplitude"].tolist(),
                      "scan_rmse": amp["scan"]["rmse_log"].tolist()},
        "scoring": {k: v for k, v in scored.items() if k != "rows"},
        "nscd": nscd,
        "solver_evidence": evidence,
    }
    figs = make_figures(out, samples, fit, amp, scored, nscd)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for f in figs:
        import base64
        (out / "figures" / f["name"]).write_bytes(base64.b64decode(f["png_base64"]))
    summary["figures"] = [{"name": f["name"], "caption": f["caption"],
                           "path": f"figures/{f['name']}"} for f in figs]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["runs", "aggregate", "all"])
    parser.add_argument("--out", default=str(REPO / "artifacts" / "nonlocal_rheology"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-nscd", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("runs", "all"):
        print(json.dumps(run_cases(out, args.shard, args.shards, args.quick)))
    if args.stage in ("aggregate", "all"):
        summary = aggregate(out, skip_nscd=args.skip_nscd)
        print(json.dumps({k: summary[k] for k in
                          ("local_law", "amplitude")}, indent=1))
        print(json.dumps({k: v for k, v in summary["scoring"].items()
                          if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())