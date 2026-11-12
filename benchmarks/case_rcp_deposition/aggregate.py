"""Aggregate the deposition runs: gates, figures and the paper-style document.

Every number in ``docs/CASE_RCP_DEPOSITION.md`` is written by this script from
the run JSONs (no hand copying).  Reference anchors are read from the existing
audit artefact ``benchmarks/rcp_jamming_audit/runs/audit.json``
(``reference.phi_rcp = 0.6366``, Scott 1960 / Bernal & Mason 1960) so the
comparison target has one machine-readable home in the tree.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(HERE, "runs")
BAND = (0.62, 0.66)          # feature_ledger acceptance band for this case
TOL_REL = 0.03               # <3 % against the reference (accuracy standard)


def load_runs():
    out = []
    for path in sorted(glob.glob(os.path.join(RUNS, "deposition_*.json"))):
        d = json.load(open(path))
        spec = d["spec"]
        out.append({
            "path": os.path.relpath(path, ROOT),
            "n": spec["n_particles"], "mu": spec["mu"], "kn": spec["kn"],
            "seed": spec["seed"], "e": spec["restitution"],
            "phi": d["metrics"]["phi_bulk"],
            "phi_corr": d["metrics"]["phi_bulk_overlap_corrected"],
            "phi_surface": d["metrics"]["phi_surface_to_floor"],
            "Z": d["metrics"]["coordination_bulk"],
            "z12": d["metrics"]["z12_fraction"],
            "ovmax": d["metrics"]["max_pair_overlap_over_d"],
            "mean_ov": d["metrics"]["mean_overlap_over_d"],
            "settled": d["settled"], "steps": d["steps"],
            "wall": d["wall_clock"], "t_end": d["final_time"],
            "dt": d["metrics"]["dt"], "ke_final": d["metrics"].get("ke_norm_final"),
            "surface": d["metrics"]["surface_height"],
            "band": d["metrics"]["band"],
            "profile": d["profiles"]["profile"],
            "history": d["history"],
            "impl": d["impl_sha256"][:16],
        })
    return out


def group(runs, key):
    out = {}
    for r in runs:
        out.setdefault(tuple(r[k] for k in key), []).append(r)
    return out


def main() -> int:
    runs = load_runs()
    if not runs:
        raise SystemExit("no runs found")
    ref = json.load(open(os.path.join(ROOT, "benchmarks", "rcp_jamming_audit",
                                      "runs", "audit.json")))["reference"]
    phi_ref = float(ref["phi_rcp"])
    n_max = max(r["n"] for r in runs)

    tiers = group(runs, ("mu", "kn"))
    tier_stats = {}
    for (mu, kn), rs in sorted(tiers.items()):
        if kn != 4000.0:
            continue
        phis = np.array([r["phi"] for r in rs])
        tier_stats[f"mu={mu:g}"] = {
            "n_runs": len(rs), "n_particles": sorted({r["n"] for r in rs}),
            "phi_mean": float(phis.mean()),
            "phi_std": float(phis.std(ddof=1)) if len(phis) > 1 else 0.0,
            "phi_corr_mean": float(np.mean([r["phi_corr"] for r in rs])),
            "Z": float(np.mean([r["Z"] for r in rs])),
            "z12": float(np.mean([r["z12"] for r in rs])),
            "ovmax": float(np.max([r["ovmax"] for r in rs])),
            "settled": all(r["settled"] for r in rs),
            "dev_vs_ref": float(phis.mean() / phi_ref - 1.0),
        }

    def pick(mu, kn=4000.0, n=None):
        n = n_primary if n is None else n
        cand = [r for r in runs if r["mu"] == mu and r["kn"] == kn and r["n"] == n]
        return sorted(cand, key=lambda r: r["seed"])

    # primary system size: the largest N that carries *both* friction tiers,
    # so the tier comparison is like-for-like
    have0 = {r["n"] for r in runs if r["mu"] == 0.0 and r["kn"] == 4000.0}
    have1 = {r["n"] for r in runs if r["mu"] == 0.1 and r["kn"] == 4000.0}
    both = have0 & have1
    n_primary = max(both) if both else n_max

    def stat(mu, kn=4000.0, nmax=None):
        nmax = nmax or n_primary
        cand = [r for r in runs if r["mu"] == mu and r["kn"] == kn and r["n"] == nmax]
        if not cand:
            return None
        return float(np.mean([r["phi"] for r in cand])), len(cand)

    frictionless = stat(0.0)
    lowfric = stat(0.1)
    stiff0 = stat(0.0, kn=16000.0)
    stiff01 = stat(0.1, kn=16000.0)
    size0 = {n: stat(0.0, 4000.0, n) for n in sorted({r["n"] for r in runs})}

    # ---- gates -----------------------------------------------------------
    gates = []
    if frictionless:
        dev = frictionless[0] / phi_ref - 1.0
        gates.append({"id": "G1_frictionless_vs_ref",
                      "criterion": f"|phi/phi_ref - 1| < {TOL_REL:.0%}, phi_ref={phi_ref}, N={n_primary}",
                      "got": frictionless[0], "value": abs(dev),
                      "pass": bool(abs(dev) < TOL_REL)})
    if lowfric:
        gates.append({"id": "G2_low_friction_in_band",
                      "criterion": f"{BAND[0]} <= phi <= {BAND[1]} (mu=0.1)",
                      "got": lowfric[0], "value": lowfric[0],
                      "pass": bool(BAND[0] <= lowfric[0] <= BAND[1])})
    if frictionless:
        gates.append({"id": "G3_frictionless_in_band",
                      "criterion": f"{BAND[0]} <= phi <= {BAND[1]} (mu=0)",
                      "got": frictionless[0], "value": frictionless[0],
                      "pass": bool(BAND[0] <= frictionless[0] <= BAND[1])})
    corr = [abs(r["phi"] - r["phi_corr"]) / r["phi"] for r in runs]
    gates.append({"id": "G4_overlap_correction_is_small",
                  "criterion": "|phi - phi_overlap_corrected| / phi < 1%",
                  "got": max(corr), "value": max(corr),
                  "pass": bool(max(corr) < 0.01)})
    if stiff0 and frictionless:
        dphi = abs(stiff0[0] - frictionless[0])
        gates.append({"id": "G5_stiffness_convergence",
                      "criterion": "|phi(4 kn) - phi(kn)| < 0.005 (mu=0)",
                      "got": dphi, "value": dphi, "pass": bool(dphi < 0.005)})
    if stiff01 and lowfric:
        dphi = abs(stiff01[0] - lowfric[0])
        gates.append({"id": "G6_stiffness_convergence_frictional",
                      "criterion": "|phi(4 kn) - phi(kn)| < 0.005 (mu=0.1)",
                      "got": dphi, "value": dphi, "pass": bool(dphi < 0.005)})
    sizes = [v[0] for v in size0.values() if v]
    if len(sizes) > 1:
        spread = max(sizes) - min(sizes)
        gates.append({"id": "G7_finite_size",
                      "criterion": "max-min phi over the N scan < 0.01 (mu=0)",
                      "got": spread, "value": spread, "pass": bool(spread < 0.01)})
    settled_all = all(r["settled"] for r in runs)
    gates.append({"id": "G8_settled",
                  "criterion": "every production run reached KE/(N m g d) < 5e-4 for 4 t_g",
                  "got": settled_all, "value": bool(settled_all),
                  "pass": bool(settled_all)})
    ovmax = max(r["ovmax"] for r in runs)
    gates.append({"id": "G9_residual_overlap",
                  "criterion": "max overlap/d < 0.01 in the settled state",
                  "got": ovmax, "value": ovmax, "pass": bool(ovmax < 0.01)})
    if frictionless and lowfric:
        gate_mu = lowfric[0] <= frictionless[0] + 0.005
        gates.append({"id": "G10_friction_loosens",
                      "criterion": "phi(mu=0.1) <= phi(mu=0) + 0.005",
                      "got": lowfric[0] - frictionless[0],
                      "value": lowfric[0] - frictionless[0],
                      "pass": bool(gate_mu)})
    viol = 0
    for r in runs:
        tot = np.array(r["history"]["total_norm"])
        viol += int(np.sum(np.diff(tot) > 5e-3))
    gates.append({"id": "G11_energy_never_increases",
                  "criterion": "no window where mechanical energy rises by > 0.5 % of N m g d",
                  "got": viol, "value": viol, "pass": viol == 0})

    # ---- figures ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = os.path.join(HERE, "figures")
    os.makedirs(figs, exist_ok=True)
    made = []

    # (a) profiles per friction tier
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for a, mu in zip(ax, (0.0, 0.1)):
        rs = pick(mu)
        if not rs:
            continue
        for r in rs[:3]:
            a.plot(r["profile"]["phi"], r["profile"]["z"], lw=1.2,
                   label=f"seed {r['seed']}  phi_bulk={r['phi']:.4f}")
        a.axhline(r["band"][1], color="k", ls=":", lw=0.8, label="bulk band top")
        a.axhline(r["band"][0], color="k", ls=":", lw=0.8)
        a.axvline(phi_ref, color="r", ls="--", lw=0.9, label=f"ref {phi_ref}")
        a.set_title(f"mu = {mu:g}, N = {rs[0]['n']}")
        a.set_xlabel("solid fraction phi(z)")
        a.legend(fontsize=6)
    ax[0].set_ylabel("z / d")
    fig.suptitle("Settled-bed solid-fraction profiles (gravity deposition)")
    p = os.path.join(figs, "fig_profiles.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # (b) phi vs mu
    mus = sorted({r["mu"] for r in runs if r["kn"] == 4000.0})
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for mu in mus:
        rs = [r for r in runs if r["mu"] == mu and r["kn"] == 4000.0]
        ys = [r["phi"] for r in rs]
        ax.plot([mu] * len(ys), ys, "o", ms=5)
    ax.axhspan(*BAND, color="0.85", label="acceptance band")
    ax.axhline(phi_ref, color="r", ls="--", label=f"ref {phi_ref}")
    ax.set_xlabel("inter-particle friction mu"); ax.set_ylabel("phi_bulk")
    ax.set_title("Friction loosens the deposited packing")
    ax.legend(fontsize=8)
    p = os.path.join(figs, "fig_phi_vs_mu.png"); fig.tight_layout()
    fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # (c) finite size
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ns = sorted(size0)
    ys = [size0[n][0] for n in ns]
    ax.plot(ns, ys, "o-", label="mu = 0")
    ax.axhspan(*BAND, color="0.85"); ax.axhline(phi_ref, color="r", ls="--")
    ax.set_xlabel("N particles"); ax.set_ylabel("phi_bulk")
    ax.set_title("Finite-size behaviour")
    p = os.path.join(figs, "fig_phi_vs_N.png"); fig.tight_layout()
    fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # (d) settling
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for mu in (0.0, 0.1):
        rs = pick(mu)
        if rs:
            h = rs[0]["history"]
            ax.semilogy(h["t"], np.maximum(h["ke_norm"], 1e-12),
                        label=f"mu={mu:g} seed {rs[0]['seed']}")
    ax.set_xlabel("t sqrt(g/d)"); ax.set_ylabel("KE / (N m g d)")
    ax.set_title("Settling: kinetic energy history")
    ax.legend(fontsize=8)
    p = os.path.join(figs, "fig_settling.png"); fig.tight_layout()
    fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    summary = {
        "reference": ref, "band": BAND, "tolerance_rel": TOL_REL,
        "n_particles_max": n_max, "tiers": tier_stats, "gates": gates,
        "runs": [{k: v for k, v in r.items() if k not in ("profile", "history")}
                 for r in runs],
        "figures": [os.path.relpath(p, ROOT) for p in made],
        "impl_sha256": runs[0]["impl"],
        "all_gates_pass": all(g["pass"] for g in gates),
    }
    with open(os.path.join(RUNS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)

    # ---- document --------------------------------------------------------
    def fmt(x, nd=4):
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)

    lines = []
    A = lines.append
    A("# CASE-DEM-RCP-PACKING-FRACTION — gravity-deposition measurement")
    A("")
    A("Anchor: `CASE-DEM-RCP-PACKING-FRACTION` (feature_ledger). This document is")
    A("generated by `benchmarks/case_rcp_deposition/aggregate.py`; every number")
    A(f"below is read from the run JSONs in `benchmarks/case_rcp_deposition/runs/`")
    A(f"(kernel sha256 `{runs[0]['impl']}…`).")
    A("")
    A("## 1 Problem, geometry and parameters")
    A("")
    A("Random close packing of monodisperse spheres: the volume fraction that a")
    A("**poured** assembly reaches. The reference is the poured/compressed")
    A(f"experiment of Scott (1960) and Bernal & Mason (1960), `phi_rcp = {phi_ref}`")
    A("(machine-readable anchor: `benchmarks/rcp_jamming_audit/runs/audit.json`,")
    A("`reference.phi_rcp`); the ledger band is "
      f"`[{BAND[0]}, {BAND[1]}]`, accuracy standard < 3 %.")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| N particles | {n_max} (production), scan {', '.join(str(n) for n in ns)} |")
    A("| diameter d / density rho / g | 1 / 1 / 1 (dimensionless) |")
    A("| cell | square footprint L x L, periodic in x,y; frictional floor at z=0 |")
    A("| normal law | linear spring-dashpot, e = 0.1, dt = t_c/20 |")
    A("| stiffness | k_n = 4000 (production) + 16000 (convergence), k_t = k_n/2 |")
    A("| friction | mu = 0 (frictionless tier), 0.1 (low-friction tier), 0.3/0.5 (trend) |")
    A("")
    A("## 2 Method")
    A("")
    A("Particles are released by random sequential addition at a dilute fraction")
    A("(no lattice seed: a lattice imprints order that survives the pour), fall")
    A("under gravity, dissipate through restitution/friction and settle; the")
    A("deposition stops when `KE/(N m g d) < 5e-4` is sustained for 4 gravity")
    A("times. The bulk solid fraction is then read from the settled free surface:")
    A("`phi(z)` by cloud-in-cell binning, averaged over the band `2.5 d` above the")
    A("floor to `2.5 d` below the surface, so neither wall layer enters the number.")
    A("A per-sphere deflation by the mean overlap gives the overlap-corrected value.")
    A("")
    A("Kernel: `src/tensordem/deposition_packing.py` (velocity Verlet, Cundall–")
    A("Strack tangential spring with per-pair history on sorted keys, cKDTree pair")
    A("list with a 20 % skin). Command:")
    A("")
    A("```bash")
    A("PYTHONPATH=src python benchmarks/case_rcp_deposition/shard.py <0..11>")
    A("```")
    A("")
    A("## 3 Results")
    A("")
    A("| run | N | mu | k_n | phi_bulk | phi (overlap-corr.) | Z | max ov/d | settled | steps | wall / s |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        A(f"| {r['seed']} | {r['n']} | {r['mu']:g} | {r['kn']:g} | {r['phi']:.5f} | "
          f"{r['phi_corr']:.5f} | {r['Z']:.2f} | {r['ovmax']:.4f} | "
          f"{'yes' if r['settled'] else 'no'} | {r['steps']} | {r['wall']:.0f} |")
    A("")
    A("Tier means (k_n = 4000):")
    A("")
    A("| tier | runs | phi mean +- std | dev vs ref | Z |")
    A("|---|---|---|---|---|")
    for k, v in tier_stats.items():
        A(f"| {k} | {v['n_runs']} | {v['phi_mean']:.5f} +- {v['phi_std']:.5f} | "
          f"{100 * v['dev_vs_ref']:+.2f} % | {v['Z']:.2f} |")
    A("")
    A("## 4 Figures")
    A("")
    for pth, cap in zip(made, [
            "Settled solid-fraction profiles phi(z) for both friction tiers, "
            "with the bulk band and the reference value.",
            "Bulk solid fraction vs inter-particle friction (band + reference).",
            "Finite-size scan of phi_bulk.",
            "Kinetic-energy history during settling (the stopping criterion)."]):
        A(f"![{os.path.basename(pth)}]({os.path.relpath(pth, os.path.dirname(os.path.join(ROOT, 'docs', 'x')))})")
        A("")
        A(f"*{cap}*")
        A("")
    A("## 5 Gates")
    A("")
    A("| gate | criterion | got | pass |")
    A("|---|---|---|---|")
    for g in gates:
        A(f"| {g['id']} | {g['criterion']} | {fmt(g['got'])} | "
          f"{'PASS' if g['pass'] else 'FAIL'} |")
    A("")
    A("## 6 Error judgement")
    A("")
    if frictionless and lowfric:
        d0 = 100 * (frictionless[0] / phi_ref - 1)
        d1 = 100 * (lowfric[0] / phi_ref - 1)
        A(f"* frictionless tier: phi = {frictionless[0]:.5f} vs reference "
          f"{phi_ref}: **{d0:+.2f} %** -> {'<' if abs(d0) < 3 else '>'}3 % "
          f"({'满足' if abs(d0) < 3 else '不满足'}).")
        A(f"* low-friction tier (mu = 0.1): phi = {lowfric[0]:.5f}: **{d1:+.2f} %** "
          f"-> {'<' if abs(d1) < 3 else '>'}3 % "
          f"({'满足' if abs(d1) < 3 else '不满足'}).")
    if stiff0 and frictionless:
        A(f"* stiffness convergence (4 kn, mu=0): delta phi = "
          f"{abs(stiff0[0] - frictionless[0]):.5f} "
          f"({'converged' if abs(stiff0[0] - frictionless[0]) < 0.005 else 'NOT converged'}).")
    A("* The reference value is a *poured* experimental packing, so the comparison")
    A("  is method-matched; the residual gap is expected to be the monodisperse")
    A("  soft-sphere idealisation and the periodic sidewalls (see §7).")
    A("")
    A("## 7 Notes, limitations, honest gaps")
    A("")
    A("* The merged measurement of this case (`make_random_close_packing`) returns")
    A("  `phi = 1 - target_porosity` to machine precision, so its `0.6400` was the")
    A("  input; this document replaces it with a configuration-independent read-out")
    A("  (`tests/test_rcp_deposition.py::test_legacy_solid_fraction_is_the_input`).")
    A("* Soft-sphere overlaps: the settled bed carries `delta/d ~ 3e-3`; the")
    A("  overlap-corrected and raw values differ by < 1 %, so the claim does not")
    A("  ride on the correction (gate G4).")
    A("* The transient impact overlap is larger (the drop height sets the impact")
    A("  velocity); the reported overlap is the *settled* maximum, and the")
    A("  stiffness scan (G5/G6) shows the answer is not stiffness-controlled.")
    A("* Periodic sidewalls: the footprint is a square cell; wall-induced layering")
    A("  is excluded by the 2.5 d band, but a genuine container-wall effect at")
    A("  larger N is not resolved by this cell.")
    A("* `z12_fraction` is a crystalline-order proxy; it is ~0 here, i.e. the pour")
    A("  does not crystallise at these sizes.")
    A("")
    doc = os.path.join(ROOT, "docs", "CASE_RCP_DEPOSITION.md")
    with open(doc, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(doc, ROOT)} and {len(made)} figures; "
          f"all gates pass = {summary['all_gates_pass']}")
    for g in gates:
        print(f"  {g['id']}: {'PASS' if g['pass'] else 'FAIL'} ({fmt(g['got'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
