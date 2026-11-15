#!/usr/bin/env python
"""CASE-DEM-BIAXIAL-FRICTION round 8 (m22106): mu-ladder critical-state analysis.

Reads the five arms written by ``scripts/m22106_mu_ladder_probe.py``
(``artifacts/CASE-DEM-BIAXIAL-FRICTION/mu_ladder/iso_*.json``) and answers the
four preregistered questions of ``prereg/m22106_mu_ladder_prereg.md``:

* per-arm critical state through the round-3 four-gate locator
  (:func:`tensordem.biaxial_critical_state.critical_state_from_stress_path`,
  logarithmic strain, isochoric boundary) -- a failing arm is WITHHELD;
* **H1** is ``phi_cv`` controlled by ``mu`` (monotone, end-to-end >= 10 deg)?
* **H2** is it insensitive to ``mu`` (ladder spread < 2 deg => engine defect)?
* **H3** is ``phi_cv(mu = 1.5)`` below the Rowe D=0 limit 28.81 deg
  (=> no rotationless 2-D disc lattice can reach the anchor reference)?
* **H4** energy cross-check: ``q * eps_dot * V`` (boundary power) vs the
  contact-layer dissipated power ``P_fric + P_damp`` read from the friction
  meter in the same window (|residual| <= 15 %).

Outputs (numbers injected programmatically, nothing typed by hand):

    benchmarks/case_dem_biaxial_friction/results_m22106/mu_ladder.json
    docs/figures/CASE-DEM-BIAXIAL-FRICTION_R8_mu_ladder.png

Usage:
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_mu_ladder.py \
        [--glob 'artifacts/CASE-DEM-BIAXIAL-FRICTION/mu_ladder/iso_*.json']
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensordem.biaxial_critical_state import (  # noqa: E402
    critical_state_from_stress_path,
    rowe_stress_dilatancy,
)

OUT_JSON = os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction",
                        "results_m22106", "mu_ladder.json")
FIG_PATH = os.path.join(REPO, "docs", "figures",
                        "CASE-DEM-BIAXIAL-FRICTION_R8_mu_ladder.png")

# --- preregistered thresholds (prereg/m22106_mu_ladder_prereg.md) -----------
H1_MIN_SPREAD_DEG = 10.0
H2_INSENSITIVE_DEG = 2.0
H3_CEILING_DEG = math.degrees(math.atan(0.55))   # 28.81 deg (Rowe 1962, D=0)
H4_MAX_RESIDUAL = 0.15

# reference angles the case pool compares against
REFS = {
    "rowe_1962_D0_atan_mu055": H3_CEILING_DEG,
    "bolton_1986_quartz_crit": 33.0,
    "bolton_1986_plane_strain_peak_dense": 38.0,
}


def _eng_to_log(eps_eng: float) -> float:
    """Uniaxial compression: ``dln(H)`` from the engineering strain."""
    return -math.log(max(1.0 - eps_eng, 1e-12))


def _mob_window(mobilisation: list, lo: float, hi: float) -> dict:
    """Friction-meter window means over the *same* window as the stress path.

    The meter samples carry the **engineering** axial strain while the
    critical-state locator works with the **logarithmic** measure (the only one
    for which an isochoric cell reports ``eps_v = 0``), so the sample strains
    are converted before the window test.  Comparing them directly silently
    selects nothing -- a defect this analysis hit on its first real run.
    """
    keys_power = ("sum_friction_power", "sum_damping_power", "calls", "contacts",
                  "sum_fn", "sum_fn_abs_tanh", "saturated", "sum_vt_abs")
    sel = [s for s in mobilisation
           if lo <= _eng_to_log(float(s.get("axial_strain", 0.0))) <= hi]
    if not sel:
        return {"samples": 0}
    tot = {k: float(sum(float(s.get(k, 0.0)) for s in sel)) for k in keys_power}
    calls = max(tot["calls"], 1e-30)
    contacts = max(tot["contacts"], 1e-30)
    return {
        "samples": len(sel),
        "calls": tot["calls"],
        "mean_contacts_per_call": tot["contacts"] / calls,
        "fn_weighted_mean_abs_tanh": (tot["sum_fn_abs_tanh"] / tot["sum_fn"]
                                      if tot["sum_fn"] else float("nan")),
        "saturated_fraction": tot["saturated"] / contacts,
        "mean_abs_vt": tot["sum_vt_abs"] / contacts,
        "friction_power_W": tot["sum_friction_power"] / calls,
        "damping_power_W": tot["sum_damping_power"] / calls,
    }


def analyse_arm(path: str) -> dict:
    with open(path) as fh:
        raw = json.load(fh)
    cfg = dict(raw["config"])
    sp = {k: np.asarray(v, dtype=float) for k, v in raw["stress_path"].items()
          if isinstance(v, (list, tuple))}
    mu = float(cfg["mu"])
    p0 = float(cfg["confining_stress"])
    spec = raw.get("specimen", {})
    arm = {
        "tag": raw.get("tag", os.path.basename(path)),
        "path": os.path.relpath(path, REPO),
        "mu": mu,
        "atan_mu_deg": math.degrees(math.atan(mu)),
        "specimen": {k: spec.get(k) for k in
                     ("columns", "rows", "particles", "radius", "dt")},
        "config": {k: cfg.get(k) for k in
                   ("confining_stress", "strain_rate", "strain_target",
                    "consolidation_steps", "lateral_control")},
        "elapsed_s": raw.get("elapsed_s"),
        "peak": raw.get("peak"),
        "inertial_number": raw.get("loading", {}).get("inertial_number"),
        "gate_state": "UNKNOWN",
        "phi_cv_deg": float("nan"),
    }
    try:
        cs = critical_state_from_stress_path(
            sp, sigma3_target=p0, strain_measure="log",
            boundary_control="isochoric")
    except Exception as exc:                     # short/smoke arm
        arm["gate_state"] = "WITHHELD"
        arm["gate_error"] = str(exc)
        return arm
    lo, hi = cs.window
    sens = cs.window_sensitivity
    eps_eng = float(np.asarray(sp["axial_strain"], dtype=float)[-1])
    eps_log = float(np.asarray(sp["axial_strain_log"], dtype=float)[-1])
    target = float(cfg.get("strain_target") or float("nan"))
    arm["strain_budget"] = {
        "declared_target": target,
        "delivered_engineering": eps_eng,
        "delivered_log": eps_log,
        "delivered_relative_to_target_log": ((eps_log - target) / target
                                             if np.isfinite(target) and target else float("nan")),
        "note": ("strain_target is a logarithmic-strain budget after the m22106 "
                 "break-condition fix; engineering strain is 1-exp(-log)"),
    }
    arm.update({
        "gate_state": "PASS" if cs.converged else "WITHHELD",
        "flags": dict(cs.flags),
        "thresholds": dict(cs.thresholds),
        "window": [float(lo), float(hi)],
        "phi_cv_deg": float(cs.phi_cv_deg),
        "phi_cv_window_spread_deg": float(cs.window_spread_deg),
        "phi_last10pct_deg": (float(sens["last_10pct"]["phi_deg"])
                              if sens.get("last_10pct", {}).get("valid") else float("nan")),
        "phi_last25pct_deg": (float(sens.get("last_25pct", {}).get("phi_deg", float("nan")))
                              if sens.get("last_25pct", {}).get("valid") else float("nan")),
        "q_cv_MPa": float(cs.q_cv_Pa) / 1e6,
        "sigma_h_cv_kPa": float(cs.sigma_h_cv_Pa) / 1e3,
        "d_eps_v_d_eps_a": float(cs.d_eps_v_d_eps_a),
        "q_drift": float(cs.q_drift),
        "peak_phi_deg": float(cs.peak_phi_deg),
    })
    try:
        arm["rowe"] = rowe_stress_dilatancy(sp, mu=mu, min_strain=max(0.15, lo))
    except Exception as exc:                     # short arm: cross-check only
        arm["rowe"] = {"error": str(exc)}
    arm["relative_error_vs_rowe_atan_mu"] = (
        (arm["phi_cv_deg"] - arm["atan_mu_deg"]) / arm["atan_mu_deg"])
    arm["relative_error_vs_anchor_ref"] = (
        (arm["phi_cv_deg"] - REFS["rowe_1962_D0_atan_mu055"])
        / REFS["rowe_1962_D0_atan_mu055"])

    # --- H4: energy cross-check (same window, same contacts) ----------------
    mob = _mob_window(raw.get("mobilisation") or [], lo, hi)
    arm["mobilisation"] = mob
    width0 = float(spec.get("width0") or 0.0)
    height0 = float(spec.get("height0") or 0.0)
    rate = float(cfg["strain_rate"])
    volume = width0 * height0 * 1.0                # thickness = 1 in the driver
    q_win = float(cs.q_cv_Pa)
    p_boundary = q_win * rate * volume             # W: q*eps_dot*V (isochoric)
    p_contact = mob.get("friction_power_W", float("nan")) + \
        mob.get("damping_power_W", float("nan"))
    arm["energy_balance"] = {
        "volume_m3": volume,
        "q_window_Pa": q_win,
        "strain_rate": rate,
        "boundary_power_W": p_boundary,
        "contact_friction_power_W": mob.get("friction_power_W", float("nan")),
        "contact_damping_power_W": mob.get("damping_power_W", float("nan")),
        "contact_total_power_W": p_contact,
        "residual_fraction": ((p_boundary - p_contact) / p_boundary
                              if p_boundary else float("nan")),
    }
    return arm


def judge(arms: list[dict]) -> dict:
    ok = [a for a in arms if a.get("gate_state") == "PASS"
          and np.isfinite(a.get("phi_cv_deg", float("nan")))]
    out = {
        "arms_total": len(arms),
        "arms_with_critical_state": len(ok),
        "thresholds": {"H1_min_spread_deg": H1_MIN_SPREAD_DEG,
                       "H2_insensitive_deg": H2_INSENSITIVE_DEG,
                       "H3_ceiling_deg": H3_CEILING_DEG,
                       "H4_max_residual": H4_MAX_RESIDUAL},
    }
    if len(ok) < 2:
        out["verdict"] = "WITHHELD"
        out["reason"] = ("fewer than two arms reached a gated critical state; "
                         "no ladder can be judged")
        return out
    ok.sort(key=lambda a: a["mu"])
    phis = [a["phi_cv_deg"] for a in ok]
    mus = [a["mu"] for a in ok]
    spread = max(phis) - min(phis)
    monotone = all(b >= a - 0.25 for a, b in zip(phis, phis[1:]))
    longest = ok[-1]
    out.update({
        "phi_cv_by_mu": {f"{a['mu']:.2f}": round(a["phi_cv_deg"], 3) for a in ok},
        "ladder_spread_deg": spread,
        "ladder_end_to_end_deg": phis[-1] - phis[0],
        "monotone_nondecreasing": monotone,
        "H1_mu_is_controlling_parameter": bool(monotone
                                               and (phis[-1] - phis[0]) >= H1_MIN_SPREAD_DEG),
        "H2_insensitive_to_mu_engine_defect": bool(spread < H2_INSENSITIVE_DEG),
        "H3_model_class_ceiling_below_rowe": bool(
            longest["mu"] >= 1.0 and longest["phi_cv_deg"] < H3_CEILING_DEG),
        "H3_note": ("H3 needs a high-mu arm (mu >= 1.0); "
                    f"highest mu run = {longest['mu']:.2f}"),
    })
    resid = [(a["mu"], a["energy_balance"]["residual_fraction"]) for a in ok
             if np.isfinite(a["energy_balance"]["residual_fraction"])]
    if resid:
        worst = max(resid, key=lambda t: abs(t[1]))
        out["H4_energy_crosscheck_pass"] = bool(abs(worst[1]) <= H4_MAX_RESIDUAL)
        out["H4_worst_residual"] = {"mu": worst[0], "residual_fraction": worst[1]}
    else:
        out["H4_energy_crosscheck_pass"] = None
        out["H4_worst_residual"] = None
    out["verdict"] = ("H1_coulomb_controlled" if out["H1_mu_is_controlling_parameter"]
                      else "H2_mu_insensitive" if out["H2_insensitive_to_mu_engine_defect"]
                      else "PARTIAL")
    return out


def render(arms: list[dict], verdict: dict, path: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                  # pragma: no cover
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ok = sorted([a for a in arms if np.isfinite(a.get("phi_cv_deg", float("nan")))],
                key=lambda a: a["mu"])
    ax = axes[0]
    if ok:
        mu = np.array([a["mu"] for a in ok])
        phi = np.array([a["phi_cv_deg"] for a in ok])
        err = np.array([a.get("phi_cv_window_spread_deg", np.nan) or np.nan for a in ok])
        ax.errorbar(mu, phi, yerr=err, marker="o", lw=1.6, capsize=3,
                    label=r"measured $\phi_{cv}$ (gated CS window)")
        mm = np.linspace(min(mu) * 0.9, max(mu) * 1.05, 100)
        ax.plot(mm, np.degrees(np.arctan(mm)), "--", color="k",
                label=r"Rowe(1962) $D=0$: $\arctan\mu$")
        for name, val, style in (("Bolton86 quartz $\\phi_{crit}$", REFS["bolton_1986_quartz_crit"], ":"),
                                 ("Bolton86 dense PS peak", REFS["bolton_1986_plane_strain_peak_dense"], "-.")):
            ax.axhline(val, ls=style, color="C3", lw=1.0)
            ax.annotate(name, (min(mu), val), fontsize=7, color="C3",
                        xytext=(1, 2), textcoords="offset points")
    ax.set_xlabel(r"interparticle friction $\mu$ [-]")
    ax.set_ylabel(r"friction angle [deg]")
    ax.set_title("(a) $\\phi_{cv}$ vs $\\mu$ (prereg H1/H2/H3)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    for a in ok:
        ax.errorbar(a["mu"], a["mobilisation"].get("fn_weighted_mean_abs_tanh", np.nan),
                    marker="s", color="C0")
    ax.set_xlabel(r"$\mu$ [-]")
    ax.set_ylabel(r"$\langle|\tanh(v_t/v_{reg})|\rangle_{F_n}$")
    ax.set_title("(b) contact-level mobilisation of the Coulomb capacity")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    ax = axes[2]
    if ok:
        pb = [a["energy_balance"]["boundary_power_W"] for a in ok]
        pc = [a["energy_balance"]["contact_total_power_W"] for a in ok]
        pf = [a["energy_balance"]["contact_friction_power_W"] for a in ok]
        x = np.arange(len(ok))
        ax.bar(x - 0.2, pb, width=0.4, label=r"boundary $q\dot\varepsilon_a V$")
        ax.bar(x + 0.2, pc, width=0.4, label=r"contact $P_{fric}+P_{damp}$")
        ax.plot(x, pf, "k.", ms=4, label=r"$P_{fric}$ only")
        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\mu$={a['mu']:.2f}" for a in ok], fontsize=8)
        ax.set_ylabel("power [W]")
        ax.set_title("(c) energy cross-check (prereg H4)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"m22106 mu-ladder: verdict {verdict.get('verdict')}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(
        REPO, "artifacts", "CASE-DEM-BIAXIAL-FRICTION", "mu_ladder", "iso_*.json"))
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"no arm files matched {args.glob}")
        return 2
    arms = [analyse_arm(p) for p in paths]
    verdict = judge(arms)
    fig = render(arms, verdict, FIG_PATH)
    payload = {"module": "CASE-DEM-BIAXIAL-FRICTION", "round": "R8(m22106)",
               "prereg": "prereg/m22106_mu_ladder_prereg.md",
               "references": REFS, "arms": arms, "verdict": verdict,
               "figure": os.path.relpath(fig, REPO) if fig else None}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1, default=lambda o: o.tolist()
                  if hasattr(o, "tolist") else str(o))
    print(f"wrote {args.out}")
    if fig:
        print(f"wrote {fig}")
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("thresholds",)}, indent=1, default=str))
    for a in arms:
        print(f"  mu={a['mu']:.2f} gates={a.get('gate_state')} "
              f"phi_cv={a.get('phi_cv_deg')} atan_mu={a['atan_mu_deg']:.2f} "
              f"mob={a.get('mobilisation', {}).get('fn_weighted_mean_abs_tanh')} "
              f"resid={a.get('energy_balance', {}).get('residual_fraction')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())