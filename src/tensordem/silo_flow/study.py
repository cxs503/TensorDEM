"""Driver for the silo-flow verification chain (Beverloo -> Janssen -> regime).

Each stage writes one JSON artefact so the runs can be sharded across nodes:

    python -m tensordem.silo_flow.study beverloo --ratio 5 --out runs/.../b5.json
    python -m tensordem.silo_flow.study janssen --out runs/.../janssen.json
    python -m tensordem.silo_flow.study hopper --angle 40 --wall-mu 0.4 --out ...
    python -m tensordem.silo_flow.study energy --out runs/.../energy.json
    python -m tensordem.silo_flow.study analyze --dir runs/...          # tables + figures

Every number in the report is read back from these JSON files (no hand copying).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict

import numpy as np

from .beverloo import (
    BEVERLOO_C, BEVERLOO_K, DischargeCase, beverloo_prediction, fit_law, run_discharge,
)
from .flow_pattern import (
    JENIKE_TABLE_UNCERTAINTY_DEG, HopperCase, find_critical_angle,
    jenike_mass_flow_limit_deg, run_hopper,
)
from .janssen import JanssenCase, run_janssen
from .kernel import Material, SiloGeometry
from .packing import build_bed

__all__ = ["main"]


def _meta() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    return {
        "git_sha": sha,
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _dump(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload["meta"] = _meta()
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
    print(f"wrote {path}", flush=True)


# --------------------------------------------------------------------- stage 1
def stage_beverloo(args) -> dict:
    case = DischargeCase(
        orifice_diameter=args.ratio * args.particle_diameter,
        particle_diameter=args.particle_diameter,
        silo_radius=args.silo_radius,
        fill_height=args.fill_height,
        wall_mu=args.wall_mu,
        seed=args.seed,
        settle_time=args.settle_time,
        discard_time=args.discard_time,
        window_time=args.window_time,
        n_windows=args.n_windows,
        kn=args.kn,
        mu=args.mu,
    )
    res = run_discharge(case)
    res["ratio"] = args.ratio
    res["beverloo_prediction_kg_s"] = beverloo_prediction(
        case.orifice_diameter, case.particle_diameter, res["rho_bulk"])
    return res


# --------------------------------------------------------------------- stage 2
def stage_janssen(args) -> dict:
    case = JanssenCase(
        silo_radius=args.silo_radius,
        fill_height=args.fill_height,
        outlet_radius=args.outlet_radius,
        wall_mu=args.wall_mu,
        particle_diameter=args.particle_diameter,
        n_layers=args.n_layers,
        settle_time=args.settle_time,
        seed=args.seed,
        kn=args.kn,
        mu=args.mu,
    )
    return run_janssen(case)


# --------------------------------------------------------------------- stage 3
def stage_hopper(args) -> dict:
    case = HopperCase(
        half_angle_deg=args.angle,
        cylinder_radius=args.silo_radius,
        outlet_radius=args.outlet_radius,
        cylinder_height=args.cylinder_height,
        wall_mu=args.wall_mu,
        particle_diameter=args.particle_diameter,
        seed=args.seed,
        settle_time=args.settle_time,
        measure_time=args.measure_time,
        startup_time=args.startup_time,
        kn=args.kn,
        mu=args.mu,
    )
    res = run_hopper(case)
    delta = math.degrees(math.atan(args.mu)) + 5.0   # DEM interlocking adds ~5 deg
    res["delta_deg_estimate"] = delta
    res["jenike_theta_c_deg"] = jenike_mass_flow_limit_deg(delta, res["phi_wall_deg"])
    return res


# --------------------------------------------------------------------- stage 4
def stage_energy(args) -> dict:
    """Per-timestep energy-ledger audit on (a) a settled bed and (b) a discharge."""
    out = {"dt_fractions": [], "cases": []}
    material = Material(radius=0.5 * args.particle_diameter, kn=args.kn, mu=args.mu)
    geometry = SiloGeometry(
        cylinder_radius=args.silo_radius,
        cylinder_height=args.fill_height + 0.03,
        outlet_radius=0.5 * args.orifice_ratio * args.particle_diameter,
    )
    base_dt = material.dt
    for frac in (0.05, 0.025):
        dt = frac * 2.0 * math.pi * math.sqrt(material.mass_eff / material.kn)
        flow = build_bed(geometry, material, args.fill_height, wall_mu=args.wall_mu,
                         seed=args.seed, settle_time=args.settle_time, dt=dt,
                         plug_outlet=True)
        flow.residual_series.clear()
        flow.residual_max = 0.0
        for _ in range(int(round(0.02 / dt))):
            flow.step()
        static = flow.balance()
        static_resid = flow.residual_max
        flow.plugged = False
        flow.residual_series.clear()
        flow.residual_max = 0.0
        for _ in range(int(round(0.12 / dt))):
            flow.step()
        discharge = flow.balance()
        out["cases"].append({
            "dt": dt,
            "dt_over_contact_period": frac,
            "n_particles": flow.n,
            "static_residual_max": static_resid,
            "static_closure_rel": static["closure_rel"],
            "discharge_residual_max": flow.residual_max,
            "discharge_closure_rel": discharge["closure_rel"],
            "discharge_energy": discharge,
            "particles_exited": flow.n_exited,
        })
        out["dt_fractions"].append(frac)
    out["default_dt"] = base_dt
    out["acceptance"] = {
        "residual_limit": 0.01,
        "best_residual": min(c["discharge_residual_max"] for c in out["cases"]),
        "pass": bool(min(c["discharge_residual_max"] for c in out["cases"]) <= 0.01),
    }
    return out


# ------------------------------------------------------------------- analysis
def stage_analyze(args) -> dict:
    """Aggregate the stage JSONs: fits, comparisons, verdicts, figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = args.dir
    out_dir = args.out_dir or os.path.join(d, "figures")
    os.makedirs(out_dir, exist_ok=True)
    summary: dict = {"dir": d, "files": sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.json")))}

    # ---------------- stage 1: Beverloo
    bev = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "beverloo_*.json")))]
    if bev:
        bev.sort(key=lambda r: r["case"]["orifice_over_particle"])
        ratios = np.array([r["case"]["orifice_over_particle"] for r in bev])
        D = np.array([r["case"]["orifice_diameter"] for r in bev])
        W = np.array([r["flow_rate_kg_s"] for r in bev])
        Wsd = np.array([r["flow_rate_std_kg_s"] for r in bev])
        dpart = bev[0]["case"]["particle_diameter"]
        rho = float(np.mean([r["rho_bulk"] for r in bev]))
        fits = fit_law(D, dpart, W, rho)
        pred = np.array([beverloo_prediction(Dr, dpart, rho) for Dr in D])
        summary["beverloo"] = {
            "ratios": ratios.tolist(),
            "flow_rate_kg_s": W.tolist(),
            "flow_rate_std_kg_s": Wsd.tolist(),
            "n_particles": [r["n_bed"] for r in bev],
            "rho_bulk": [r["rho_bulk"] for r in bev],
            "rho_bulk_mean": rho,
            "window_rates": [r["flow_rate_windows_kg_s"] for r in bev],
            "prediction_kg_s": pred.tolist(),
            "rel_error": ((W - pred) / pred).tolist(),
            "fit": fits,
            "scaling_check": {
                "exponent_fixed_k": fits["fixed_k"]["exponent"],
                "exponent_free_k": fits["free_k"]["exponent"],
                "r2_fixed_k": fits["fixed_k"]["r2"],
                "r2_free_k": fits["free_k"]["r2"],
                "k_free": fits["free_k"]["k"],
                "c_fixed_k": fits["fixed_k"]["c"],
                "target_exponent": 2.5,
                "exponent_tolerance": 0.2,
                "target_r2": 0.95,
                "pass": bool(abs(fits["fixed_k"]["exponent"] - 2.5) <= 0.2 and fits["fixed_k"]["r2"] >= 0.95),
            },
        }
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        eff = D - BEVERLOO_K * dpart
        ax.errorbar(np.log(eff), np.log(W), yerr=Wsd / np.maximum(W, 1e-12), fmt="o",
                    capsize=3, label="DEM (measured mass balance)")
        xx = np.linspace(eff.min() * 0.9, eff.max() * 1.1, 50)
        ax.plot(np.log(xx), np.log(fits["fixed_k"]["amplitude"]) +
                fits["fixed_k"]["exponent"] * np.log(xx),
                "-", label=f"fit p={fits['fixed_k']['exponent']:.2f}, R2={fits['fixed_k']['r2']:.3f}")
        ax.plot(np.log(xx), np.log(fits["fixed_k"]["amplitude"]) + 2.5 * np.log(xx),
                "--", label="Beverloo p=5/2")
        ax.set_xlabel("ln(D - 1.4 d)  [m]")
        ax.set_ylabel("ln W  [kg/s]")
        ax.set_title("Stage 1: Beverloo scaling")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "stage1_beverloo.png"), dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.errorbar(ratios, W, yerr=Wsd, fmt="o", capsize=3, label="DEM")
        ax.plot(ratios, pred, "s--", label=f"Beverloo C={BEVERLOO_C}, k={BEVERLOO_K}")
        ax.set_xlabel("D / d")
        ax.set_ylabel("W  [kg/s]")
        ax.set_title("Stage 1: discharge rate vs orifice")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "stage1_rate_vs_orifice.png"), dpi=140)
        plt.close(fig)

    # ---------------- stage 2: Janssen
    jf = sorted(glob.glob(os.path.join(d, "janssen_*.json")))
    if jf:
        j = json.load(open(jf[0]))
        wp = j["wall_profile"]
        vp = j["vertical_profile"]
        summary["janssen"] = {
            "rho_bulk": j["rho_bulk"],
            "rho_g_R": j["rho_g_R"],
            "base_pressure_pa": j["base_pressure_pa"],
            "base_pressure_janssen_pa": j.get("base_pressure_janssen_pa"),
            "k_measured": j.get("k_measured"),
            "wall_fit": j["wall_fit"],
            "vertical_fit": j["vertical_fit"],
            "identity_ratio": j.get("identity_ratio"),
            "identity_error": j.get("identity_error"),
            "p_inf_rel_error": j.get("p_inf_rel_error"),
            "z0_rel_error": j.get("z0_rel_error"),
            "n_particles": j["n_particles"],
            "free_surface": j["free_surface"],
        }
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        dep = np.array(wp["depth"])
        sig = np.array(wp["sigma_w"])
        ax.plot(dep, sig, "o", label="DEM wall stress $\\sigma_h$")
        if j["wall_fit"].get("ok"):
            zz = np.linspace(0.0, max(dep) * 1.05, 200)
            from .janssen import janssen_profile
            ax.plot(zz, janssen_profile(zz, j["wall_fit"]["p_inf"], j["wall_fit"]["z0"]), "-",
                    label=f"Janssen fit $P_\\infty$={j['wall_fit']['p_inf']:.0f} Pa, "
                          f"$z_0$={j['wall_fit']['z0']:.3f} m")
            ax.plot(zz, janssen_profile(zz, j["rho_g_R"] * 0.0 + j["wall_fit"]["p_inf"], 1.0), ":")
        dv = np.array(vp["depth"])
        sv = np.array(vp["sigma_v"])
        ax.plot(dv, sv, "s", ms=3, label="DEM vertical stress (force balance)")
        ax.set_xlabel("depth below free surface [m]")
        ax.set_ylabel("stress [Pa]")
        ax.set_title("Stage 2: Janssen wall-pressure saturation")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "stage2_janssen.png"), dpi=140)
        plt.close(fig)

    # ---------------- stage 3: hopper regimes
    hp = []
    for p in sorted(glob.glob(os.path.join(d, "hopper_*.json"))):
        hp.append(json.load(open(p)))
    if hp:
        by_wall: dict[str, list] = {}
        for r in hp:
            by_wall.setdefault(f"{r['case']['wall_mu']:.2f}", []).append(r)
        summary["hopper"] = {}
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        for mu, rows in sorted(by_wall.items()):
            rows.sort(key=lambda r: r["case"]["half_angle_deg"])
            ang = np.array([r["case"]["half_angle_deg"] for r in rows])
            wall_ratio = np.array([r["classification"]["wall_motion_ratio"] for r in rows])
            stag = np.array([r["classification"]["stagnant_fraction"] for r in rows])
            crit = find_critical_angle(rows)
            phi_w = rows[0]["phi_wall_deg"]
            delta = rows[0]["delta_deg_estimate"]
            th_jenike = jenike_mass_flow_limit_deg(delta, phi_w)
            err = (abs(crit["theta_c"] - th_jenike) / th_jenike) if crit.get("ok") else float("nan")
            summary["hopper"][mu] = {
                "angles": ang.tolist(),
                "wall_motion_ratio": wall_ratio.tolist(),
                "stagnant_fraction": stag.tolist(),
                "regimes": [r["classification"]["regime"] for r in rows],
                "flow_rate_kg_s": [r["flow_rate_kg_s"] for r in rows],
                "critical": crit,
                "phi_wall_deg": phi_w,
                "delta_deg_estimate": delta,
                "jenike_theta_c_deg": th_jenike,
                "jenike_uncertainty_deg": JENIKE_TABLE_UNCERTAINTY_DEG,
                "rel_error": err,
                "pass": bool(crit.get("ok") and err <= 0.15),
            }
            ax.plot(ang, wall_ratio, "o-", label=f"wall motion ratio, $\\mu_w$={mu}")
            ax.plot(ang, stag, "s--", label=f"stagnant fraction, $\\mu_w$={mu}")
            if crit.get("ok"):
                ax.axvline(crit["theta_c"], ls=":", lw=1)
            ax.axvline(th_jenike, ls="-.", lw=0.8)
        ax.set_xlabel("hopper half angle [deg]")
        ax.set_ylabel("metric [-]")
        ax.set_title("Stage 3: mass/funnel flow transition")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "stage3_regimes.png"), dpi=140)
        plt.close(fig)

    # ---------------- stage 4: energy
    ef = sorted(glob.glob(os.path.join(d, "energy_*.json")))
    if ef:
        e = json.load(open(ef[0]))
        summary["energy"] = {
            "cases": e["cases"],
            "acceptance": e["acceptance"],
            "default_dt": e["default_dt"],
        }
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        fracs = [c["dt_over_contact_period"] for c in e["cases"]]
        res_static = [c["static_residual_max"] for c in e["cases"]]
        res_disch = [c["discharge_residual_max"] for c in e["cases"]]
        ax.loglog(fracs, res_static, "o-", label="settled bed")
        ax.loglog(fracs, res_disch, "s-", label="discharging silo")
        ax.axhline(0.01, color="k", ls="--", label="1 % acceptance")
        ax.set_xlabel("dt / contact period")
        ax.set_ylabel("max |per-step energy residual| / E_ref")
        ax.set_title("Stage 4: energy-ledger residual")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "stage4_energy.png"), dpi=140)
        plt.close(fig)

    verdicts = {
        "stage1_beverloo": summary.get("beverloo", {}).get("scaling_check", {}).get("pass"),
        "stage2_janssen": bool(
            summary.get("janssen", {}).get("wall_fit", {}).get("r2", 0.0) >= 0.95
            and summary.get("janssen", {}).get("identity_error", 9.9) <= 0.10),
        "stage3_regime": all(v["pass"] for v in summary.get("hopper", {}).values()) if summary.get("hopper") else None,
        "stage4_energy": summary.get("energy", {}).get("acceptance", {}).get("pass"),
    }
    summary["verdicts"] = verdicts
    summary["_figures"] = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    _dump(os.path.join(d, "summary.json"), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="silo-flow verification chain")
    sub = p.add_subparsers(dest="stage", required=True)

    b = sub.add_parser("beverloo")
    b.add_argument("--ratio", type=float, required=True)
    b.add_argument("--particle-diameter", type=float, default=0.0045)
    b.add_argument("--silo-radius", type=float, default=0.045)
    b.add_argument("--fill-height", type=float, default=0.10)
    b.add_argument("--wall-mu", type=float, default=0.30)
    b.add_argument("--mu", type=float, default=0.40)
    b.add_argument("--kn", type=float, default=1.0e4)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--settle-time", type=float, default=0.08)
    b.add_argument("--discard-time", type=float, default=0.15)
    b.add_argument("--window-time", type=float, default=0.20)
    b.add_argument("--n-windows", type=int, default=6)
    b.add_argument("--out", required=True)

    j = sub.add_parser("janssen")
    j.add_argument("--silo-radius", type=float, default=0.025)
    j.add_argument("--fill-height", type=float, default=0.30)
    j.add_argument("--outlet-radius", type=float, default=0.008)
    j.add_argument("--wall-mu", type=float, default=0.40)
    j.add_argument("--particle-diameter", type=float, default=0.0045)
    j.add_argument("--n-layers", type=int, default=20)
    j.add_argument("--settle-time", type=float, default=0.10)
    j.add_argument("--mu", type=float, default=0.40)
    j.add_argument("--kn", type=float, default=1.0e4)
    j.add_argument("--seed", type=int, default=11)
    j.add_argument("--out", required=True)

    h = sub.add_parser("hopper")
    h.add_argument("--angle", type=float, required=True)
    h.add_argument("--silo-radius", type=float, default=0.035)
    h.add_argument("--outlet-radius", type=float, default=0.010)
    h.add_argument("--cylinder-height", type=float, default=0.045)
    h.add_argument("--wall-mu", type=float, default=0.40)
    h.add_argument("--particle-diameter", type=float, default=0.0045)
    h.add_argument("--mu", type=float, default=0.40)
    h.add_argument("--kn", type=float, default=1.0e4)
    h.add_argument("--seed", type=int, default=21)
    h.add_argument("--settle-time", type=float, default=0.06)
    h.add_argument("--measure-time", type=float, default=0.10)
    h.add_argument("--startup-time", type=float, default=0.08)
    h.add_argument("--out", required=True)

    e = sub.add_parser("energy")
    e.add_argument("--particle-diameter", type=float, default=0.0045)
    e.add_argument("--silo-radius", type=float, default=0.035)
    e.add_argument("--fill-height", type=float, default=0.08)
    e.add_argument("--orifice-ratio", type=float, default=5.0)
    e.add_argument("--wall-mu", type=float, default=0.30)
    e.add_argument("--mu", type=float, default=0.40)
    e.add_argument("--kn", type=float, default=1.0e4)
    e.add_argument("--seed", type=int, default=31)
    e.add_argument("--settle-time", type=float, default=0.05)
    e.add_argument("--out", required=True)

    a = sub.add_parser("analyze")
    a.add_argument("--dir", required=True)
    a.add_argument("--out-dir", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "beverloo":
        payload = stage_beverloo(args)
    elif args.stage == "janssen":
        payload = stage_janssen(args)
    elif args.stage == "hopper":
        payload = stage_hopper(args)
    elif args.stage == "energy":
        payload = stage_energy(args)
    elif args.stage == "analyze":
        stage_analyze(args)
        return 0
    else:
        raise SystemExit(f"unknown stage {args.stage}")
    _dump(args.out, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())