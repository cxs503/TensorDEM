"""Campaign driver: freeze-bond constitutive -> multi-bond ledger -> ice arch.

Produces ``artifacts/freeze_bond_arch/`` (results JSON, CSV tables, PNG figures)
and prints the acceptance numbers.  Every number in the JSON is computed here,
never transcribed: the report generator reads the JSON.

Run::

    PYTHONPATH=src python scripts/run_freeze_bond_arch.py [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.freeze_bond import (  # noqa: E402
    FREEZE_BOND_LAW_REFERENCE,
    SpecimenConfig,
    build_box_specimen,
    build_channel_specimen,
    load_steps,
)
from tensordem.freeze_bond.cohesive import BondEnvironment, CohesiveBondSet  # noqa: E402
from tensordem.freeze_bond.experiments import (  # noqa: E402
    CampaignConfig,
    channel_arch_run,
    energy_ledger_case,
    kappa_sweep,
    self_sustain,
    single_bond_self_consistency,
)
from tensordem.freeze_bond.lindqvist import band_over_speeds  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "artifacts" / "freeze_bond_arch"
FIG = OUT / "figures"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------- single bond
def single_bond_figure(path: Path) -> dict:
    law = FREEZE_BOND_LAW_REFERENCE
    k0 = 1.5e9
    sigma = 3.0e4
    cs = CohesiveBondSet(law, BondEnvironment.uniform(1), stiffness_pa_per_m=k0)
    d0 = sigma / k0
    df = 2.0 * law.fracture_energy_at(sigma, k0) / sigma
    deltas = np.linspace(0.0, 1.15 * df, 600)
    tractions, work, diss = [], [], []
    for delta in deltas:
        state = cs.trial(np.array([delta]), np.array([0.0]))
        tractions.append(float(state["traction"][0]))
        cs.commit(state)
        work.append(float(cs.work_J_m2[0]))
        diss.append(float(cs.dissipated_J_m2[0]))
    tractions = np.array(tractions)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(deltas * 1e6, tractions / 1e3, label="bond element")
    ax[0].axhline(sigma / 1e3, ls="--", c="k", label=r"analytic $\sigma_{fb}$")
    ax[0].axvline(d0 * 1e6, ls=":", c="gray", label=r"$\delta_0=\sigma/k_0$")
    ax[0].set_xlabel(r"effective separation $\delta_e$ [$\mu$m]")
    ax[0].set_ylabel(r"traction $t_e$ [kPa]")
    ax[0].set_title("freeze-bond cohesive law (self-consistency)")
    ax[0].legend(fontsize=8)
    ax[1].plot(deltas * 1e6, work, label=r"work $W=\int t\,d\delta$")
    ax[1].plot(deltas * 1e6, diss, label=r"dissipated $W_{diss}$")
    ax[1].axhline(law.fracture_energy_at(sigma, k0), ls="--", c="k",
                  label=r"$G_f=3.0$ J/m$^2$")
    ax[1].set_xlabel(r"effective separation $\delta_e$ [$\mu$m]")
    ax[1].set_ylabel(r"energy per unit bond area [J/m$^2$]")
    ax[1].set_title("fracture energy is spent exactly")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {
        "peak_traction_kPa": float(tractions.max() / 1e3),
        "sigma_analytic_kPa": sigma / 1e3,
        "G_f_used_J_m2": float(work[-1]),
        "G_f_declared_J_m2": float(law.fracture_energy_at(sigma, k0)),
        "dissipated_J_m2": float(diss[-1]),
        "delta_0_m": float(d0),
        "delta_f_m": float(df),
    }


# ------------------------------------------------------------------- ledger
def ledger_study(quick: bool) -> dict:
    steps = (20, 40) if quick else (20, 40, 80)
    rows = []
    for n in steps:
        t0 = time.time()
        case = energy_ledger_case(kappa=0.02, n_steps=n, cv=0.0)
        led = case["ledger"]
        rows.append({
            "n_steps": n,
            "kappa": 0.02,
            "external_work_J": led["external_work_J"],
            "stored_energy_change_J": led["stored_energy_change_J"],
            "bond_dissipated_J": led["bond_dissipated_J"],
            "slip_dissipated_J": led["slip_dissipated_J"],
            "residual_J": led["residual_J"],
            "residual_fraction": led["residual_fraction"],
            "broken_bonds": case["final"]["broken_bonds"],
            "mean_damage": case["final"]["mean_damage"],
            "wall_time_s": time.time() - t0,
        })
    return {"rows": rows,
            "max_abs_residual_fraction": max(abs(r["residual_fraction"]) for r in rows)}


# ---------------------------------------------------------------- biaxial
def biaxial_study(kappas, campaign: CampaignConfig) -> dict:
    rows = []
    for kappa in kappas:
        cfg = campaign.specimen_config()
        spec = build_box_specimen(FREEZE_BOND_LAW_REFERENCE, config=cfg)
        from tensordem.freeze_bond.experiments import _environment
        env = _environment(spec.topology.n_bonds,
                           campaign.strength_kappa(kappa), campaign)
        spec = build_box_specimen(FREEZE_BOND_LAW_REFERENCE, config=cfg, environment=env)
        res = load_steps(spec, total_displacement_m=campaign.total_displacement_m,
                         n_steps=campaign.n_steps)
        a = res.arrays()
        height = spec.height_m
        width = spec.width_m
        sigma_axial = a["platen_force_N"][-1] / width
        sigma_lateral = a["lateral_force_N"][-1] / height
        rows.append({
            "kappa": float(kappa),
            "axial_stress_Pa": float(sigma_axial),
            "lateral_stress_Pa": float(sigma_lateral),
            "lateral_over_axial": float(sigma_lateral / sigma_axial) if sigma_axial else None,
            "broken_bonds": int(a["broken_bonds"][-1]),
            "mean_damage": float(a["mean_damage"][-1]),
            "residual_fraction": res.ledger["residual_fraction"],
        })
    axial = np.array([r["axial_stress_Pa"] for r in rows])
    ratio = axial / axial[0] if axial[0] else np.ones_like(axial)
    return {"rows": rows,
            "axial_strengthening_ratio": ratio.tolist(),
            "note": "confined biaxial cell; sigma_axial = F_platen / width, "
                    "sigma_lateral = F_sidewalls / height"}


# ----------------------------------------------------------------- figures
def sweep_figure(sweep: dict, path: Path) -> None:
    k = np.asarray(sweep["kappas"])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    ax[0].plot(k, sweep["arch_count"], "o-", label="cycle rank (arch count)")
    ax[0].plot(k, sweep["spanning_loops"], "s--", label="spanning loops")
    ax[0].set_xlabel(r"$\kappa=\sigma_{fb}/\sigma_{bulk}$")
    ax[0].set_ylabel("count")
    ax[0].set_title("strong-force network loops")
    ax[0].legend(fontsize=8)
    ax2 = ax[0].twinx()
    ax2.plot(k, sweep["broken_bond_fraction"], "^:", c="tab:red",
             label="broken bond fraction")
    ax2.set_ylabel("broken bond fraction", color="tab:red")
    ax2.tick_params(axis="y", colors="tab:red")
    ax2.legend(fontsize=8, loc="center right")

    ax[1].plot(k, sweep["transmission"], "o-", c="tab:green",
               label=r"$F_{load\,cell}/F_{platen}$")
    ax[1].plot(k, sweep["mean_damage"], "s-", c="tab:purple", label="mean bond damage")
    ax[1].set_xlabel(r"$\kappa=\sigma_{fb}/\sigma_{bulk}$")
    ax[1].set_title("load transmission and damage")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def forcerun_figure(runs: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for run in runs:
        rows = run["rows"]
        u = np.array([r["step"] for r in rows]) * 1.0
        fp = np.array([r["platen_force_N"] for r in rows]) / 1e3
        fl = np.array([r["load_cell_force_N"] for r in rows]) / 1e3
        ax.plot(u, fp, "-", label=rf"platen $\kappa$={run['kappa']:.3f}")
        ax.plot(u, fl, "--", label=rf"load cell $\kappa$={run['kappa']:.3f}")
    ax.set_xlabel("load step")
    ax.set_ylabel("force [kN]")
    ax.set_title("channel push: platen vs downstream load cell")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def network_figure(run: dict, path: Path) -> None:
    data = run["result"].steps[-1].contact_forces
    pos = data["positions"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, key, title in ((ax[0], "bond_magnitudes", "freeze-bond forces [N]"),
                          (ax[1], "contact_magnitudes", "ice-ice contact forces [N]")
                          ):
        pairs = data["bond_pairs"] if key == "bond_magnitudes" else data["contact_pairs"]
        mag = data[key]
        if pairs.size:
            seg = pos[pairs]
            lc = np.repeat(mag, 2)
            from matplotlib.collections import LineCollection
            col = LineCollection(seg, array=lc, cmap="inferno", linewidths=0.9)
            a.add_collection(col)
            fig.colorbar(col, ax=a, fraction=0.046)
        a.scatter(pos[:, 0], pos[:, 1], s=4, c="tab:blue")
        a.set_aspect("equal")
        a.set_title(title)
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
    fig.suptitle(rf"force network at $\kappa$={run['kappa']:.3f} "
                 f"(arch count {run['final']['arch_count']})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def band_figure(band: dict, path: Path) -> None:
    speeds = [b["speed_m_s"] for b in band["bands"]]
    lower = [b["lower_total_N"] / 1e3 for b in band["bands"]]
    upper = [b["upper_total_N"] / 1e3 for b in band["bands"]]
    base = [b["unbonded_total_N"] / 1e3 for b in band["bands"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(speeds, base, "k--", marker="o", label="Lindqvist (unbonded anchor)")
    ax.fill_between(speeds, lower, upper, alpha=0.3, color="tab:orange",
                    label="freeze-bond difference interval")
    ax.set_xlabel("ice drift speed [m/s]")
    ax.set_ylabel("resistance [kN]")
    ax.set_title(rf"freeze-bond correction interval at $\kappa$={band['kappa']:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    campaign = CampaignConfig(
        n_x=10 if args.quick else 14,
        n_y=6,
        n_steps=16 if args.quick else 30,
        total_displacement_m=1.0e-3 if args.quick else 1.5e-3,
    )
    kappas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    if args.quick:
        kappas = [0.0, 0.005, 0.02, 0.1]

    results: dict = {
        "campaign": {
            "n_x": campaign.n_x, "n_y": campaign.n_y,
            "n_steps": campaign.n_steps,
            "total_displacement_m": campaign.total_displacement_m,
            "bulk_strength_pa": campaign.bulk_strength_pa,
            "bond_fraction": campaign.bond_fraction,
            "bond_strength_cv": campaign.bond_strength_cv,
            "kappas": kappas,
        },
        "law": FREEZE_BOND_LAW_REFERENCE.as_dict(),
    }

    print("[1/6] single-bond self-consistency")
    results["single_bond"] = single_bond_self_consistency()
    results["single_bond_figure"] = single_bond_figure(FIG / "fig1_single_bond.png")

    print("[2/6] energy ledger")
    results["energy_ledger"] = ledger_study(args.quick)

    print("[3/6] kappa sweep (arch)")
    sweep = kappa_sweep(kappas, campaign=campaign)
    sweep_figure(sweep, FIG / "fig3_kappa_sweep.png")
    results["kappa_sweep"] = {k: v for k, v in sweep.items() if k != "runs"}
    results["kappa_sweep_runs"] = [
        {"kappa": r["kappa"], "rows": r["rows"], "ledger": r["ledger"]}
        for r in sweep["runs"]
    ]

    print("[4/6] force histories and network snapshot")
    runs_for_fig = [sweep["runs"][0], sweep["runs"][len(sweep["runs"]) // 2],
                    sweep["runs"][-1]]
    forcerun_figure(runs_for_fig, FIG / "fig4_forceruns.png")
    network_figure(sweep["runs"][-1], FIG / "fig5_network.png")

    print("[5/6] self-sustain protocol")
    results["self_sustain"] = self_sustain(
        kappas[-1], campaign=campaign,
        refreeze_time_s=86400.0)

    print("[6/6] biaxial cell and Lindqvist band")
    results["biaxial"] = biaxial_study([0.0, 0.01, 0.05], campaign)
    ratio = float(np.mean([sweep["runs"][-1]["final"]["platen_force_N"] /
                           max(sweep["runs"][0]["final"]["platen_force_N"], 1e-12)]))
    results["lindqvist_band"] = band_over_speeds(kappas[-1], ratio)
    band_figure(results["lindqvist_band"], FIG / "fig6_lindqvist_band.png")

    results["wall_time_s"] = time.time() - t_start
    results["figures"] = sorted(p.name for p in FIG.glob("*.png"))

    # CSV tables (numbers only, straight from memory)
    csv = OUT / "kappa_sweep.csv"
    with csv.open("w") as fh:
        fh.write("kappa,arch_count,spanning_loops,transmission,platen_force_N,"
                 "lateral_force_N,broken_bonds,mean_damage\n")
        for i, kappa in enumerate(sweep["kappas"]):
            run = sweep["runs"][i]
            fh.write(f"{kappa},{sweep['arch_count'][i]},{sweep['spanning_loops'][i]},"
                     f"{sweep['transmission'][i]:.6f},{run['final']['platen_force_N']:.4f},"
                     f"{run['final']['lateral_force_N']:.4f},{run['final']['broken_bonds']},"
                     f"{run['final']['mean_damage']:.6f}\n")

    jpath = OUT / "results.json"
    with jpath.open("w") as fh:
        json.dump(results, fh, indent=2, default=float)
    results["results_sha256"] = sha256_file(jpath)
    results["artifacts"] = {
        "results_json": "artifacts/freeze_bond_arch/results.json",
        "kappa_sweep_csv": "artifacts/freeze_bond_arch/kappa_sweep.csv",
        "figures": [f"artifacts/freeze_bond_arch/figures/{p.name}"
                    for p in sorted(FIG.glob("*.png"))],
    }
    with jpath.open("w") as fh:
        json.dump(results, fh, indent=2, default=float)

    print(json.dumps({
        "single_bond_max_rel_error": results["single_bond"]["max_rel_error"],
        "ledger_max_abs_residual_fraction": results["energy_ledger"]["max_abs_residual_fraction"],
        "kappa_c_arch": results["kappa_sweep"]["kappa_c_arch"],
        "kappa_c_transmission": results["kappa_sweep"]["kappa_c_transmission"],
        "kappa_c_damage": results["kappa_sweep"]["kappa_c_damage"],
        "arch_count": results["kappa_sweep"]["arch_count"],
        "transmission": results["kappa_sweep"]["transmission"],
        "self_sustain": results["self_sustain"]["verdict"],
        "lindqvist_interval_kN": [
            (b["lower_total_N"] / 1e3, b["upper_total_N"] / 1e3)
            for b in results["lindqvist_band"]["bands"]],
        "wall_time_s": results["wall_time_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())