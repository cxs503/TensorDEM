#!/usr/bin/env python
"""CASE-DEM-BIAXIAL-FRICTION · micro-mechanical validity ladder analysis (m22376).

Consumes the per-arm JSON written by ``scripts/m22376_fabric_validity_probe.py``
(``artifacts/CASE-DEM-BIAXIAL-FRICTION/fabric_validity/arm_*.json``) and answers
one question that nine previous rounds never asked:

    **Is the specimen itself in a numerically valid contact regime, and does the
    measured friction angle respond to the things it must respond to?**

The ladder is judged on three pre-declared criteria (fixed in
``docs/CASE-DEM-BIAXIAL-FRICTION_R10_MICRO_VALIDITY.md`` before the stiffness
arms returned):

* **V1 (geometric validity)** -- ``phi_cv`` must rise monotonically with
  ``kn / (p d)`` and the top stiffness arm must come within 8 % of the Rowe
  (1962) reference ``arctan(mu)``.  If it does, the low angle is a soft-particle
  artefact (overlap ``delta / d`` of a few percent) and the anchor geometry is
  numerically invalid.
* **V2 (friction must matter)** -- ``phi_cv(mu = 1.5) - phi_cv(mu = 0.55) >= 5 deg``
  and ``phi_cv(mu = 0) <= 1 deg``.  The latter is a *control*: a frictionless
  packing must report a vanishing friction angle.  If it does not, the stress
  measurement itself is biased.
* **V3 (quasi-static)** -- ``phi_cv`` at ``eps_dot / 2`` must stay within 2 deg of
  the baseline, i.e. the strength is rate independent (a rate-dependent value
  would be a viscous flow stress, not a Mohr-Coulomb strength).

Outputs
-------
``benchmarks/case_dem_biaxial_friction/results_m22376/fabric_validity.json``
``docs/figures/CASE-DEM-BIAXIAL-FRICTION_R10_micro_validity.png``

Usage: ``PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_fabric_validity.py``
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

ARM_GLOB = os.path.join(REPO, "artifacts", "CASE-DEM-BIAXIAL-FRICTION",
                        "fabric_validity", "arm_*.json")
OUT_DIR = os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction",
                       "results_m22376")
OUT_JSON = os.path.join(OUT_DIR, "fabric_validity.json")
FIG_PATH = os.path.join(REPO, "docs", "figures",
                        "CASE-DEM-BIAXIAL-FRICTION_R10_micro_validity.png")

ROWE_MU = 0.55
ROWE_DEG = math.degrees(math.atan(ROWE_MU))          # 28.81 deg
BOLTON_CRIT_DEG = 33.0                              # angular quartz sand
TOLERANCE = 0.08                                    # case-pool acceptance
V1_MIN_RISE_DEG = 3.0                               # monotone-rise floor
V2_MIN_GAP_DEG = 5.0
V2_MU0_MAX_DEG = 1.0
V3_MAX_DRIFT_DEG = 2.0


def load_arms() -> dict[str, dict]:
    arms = {}
    for path in sorted(glob.glob(ARM_GLOB)):
        with open(path) as fh:
            rec = json.load(fh)
        rec["_path"] = os.path.relpath(path, REPO)
        arms[rec["tag"]] = rec
    return arms


def _micro(rec: dict) -> dict:
    micro = rec.get("micro") or [{}]
    return micro[-1] if micro else {}


def _cfg(rec: dict) -> dict:
    return rec.get("config", {})


def summarise(arms: dict[str, dict]) -> dict:
    rows = []
    for tag, rec in arms.items():
        cs = rec.get("critical_state", {})
        mic = _micro(rec)
        cfg = _cfg(rec)
        rows.append({
            "tag": tag,
            "kn_N_per_m": cfg.get("kn"),
            "mu": cfg.get("mu"),
            "rate_1_per_s": cfg.get("rate"),
            "vreg": cfg.get("vreg"),
            "strain_target_log": cfg.get("strain_target"),
            "phi_cv_deg": cs.get("phi_cv_deg"),
            "peak_phi_deg": cs.get("peak_phi_deg"),
            "q_cv_Pa": cs.get("q_cv_Pa"),
            "sigma_h_cv_Pa": cs.get("sigma_h_cv_Pa"),
            "q_drift": cs.get("q_drift"),
            "converged": cs.get("converged"),
            "flags": cs.get("flags"),
            "coordination_number": mic.get("coordination_number"),
            "overlap_ratio_mean": mic.get("overlap_ratio_mean"),
            "overlap_ratio_max": mic.get("overlap_ratio_max"),
            "dimensionless_stiffness": mic.get("dimensionless_stiffness_kn_over_pd"),
            "fabric_anisotropy_a_c": mic.get("fabric_anisotropy_a_c"),
            "normal_force_anisotropy_a_f": mic.get("normal_force_anisotropy_a_f"),
            "phi_stress_force_fabric_deg": mic.get("phi_stress_force_fabric_deg"),
            "phi_from_measured_stress_deg": mic.get("phi_from_measured_stress_deg"),
            "mobilisation": rec.get("mobilisation_critical_window") or {},
            "elapsed_s": rec.get("elapsed_s"),
            "source": rec.get("_path"),
        })
    rows.sort(key=lambda r: (r["kn_N_per_m"] or 0.0, r["mu"] or 0.0,
                             r["rate_1_per_s"] or 0.0))
    return {"rows": rows}


def judge(rows: list[dict]) -> dict:
    by_tag = {r["tag"]: r for r in rows}

    def phi(tag):
        r = by_tag.get(tag)
        return None if r is None else r["phi_cv_deg"]

    # ---- V1: stiffness ladder -------------------------------------------
    ladder = sorted([r for r in rows
                     if (r["mu"] or 0.0) == ROWE_MU
                     and abs((r["rate_1_per_s"] or 0.0) - 0.05) < 1e-12
                     and (r["vreg"] in (None, 0.01))
                     and r["tag"].startswith("kn")
                     and not r["tag"].endswith("_mob")],
                    key=lambda r: r["kn_N_per_m"] or 0.0)
    v1 = {"arms": [{"tag": r["tag"], "kn": r["kn_N_per_m"],
                    "dimensionless_stiffness": r["dimensionless_stiffness"],
                    "phi_cv_deg": r["phi_cv_deg"],
                    "overlap_ratio_mean": r["overlap_ratio_mean"]} for r in ladder]}
    phis = [r["phi_cv_deg"] for r in ladder if r["phi_cv_deg"] is not None]
    if len(phis) >= 2:
        v1["monotone_rise_deg"] = phis[-1] - phis[0]
        v1["monotone"] = all(b >= a - 1e-9 for a, b in zip(phis, phis[1:]))
        v1["min_rise_required_deg"] = V1_MIN_RISE_DEG
        v1["rise_ok"] = bool(v1["monotone"] and v1["monotone_rise_deg"]
                             >= V1_MIN_RISE_DEG)
        top = ladder[-1]["phi_cv_deg"]
        v1["top_arm"] = ladder[-1]["tag"]
        v1["top_arm_phi_cv_deg"] = top
        v1["top_arm_error_vs_rowe"] = (abs(top - ROWE_DEG) / ROWE_DEG
                                       if top is not None else None)
        v1["top_arm_within_8pct"] = bool(
            v1["top_arm_error_vs_rowe"] is not None
            and v1["top_arm_error_vs_rowe"] <= TOLERANCE)
        v1["verdict"] = ("SUPPORTED (soft-particle artefact)"
                         if v1["rise_ok"] and v1["top_arm_within_8pct"]
                         else "FALSIFIED (stiffness does not recover the angle)"
                         if v1["rise_ok"] or v1["top_arm_within_8pct"]
                         else "FALSIFIED (no response to stiffness at all)")
    else:
        v1["verdict"] = "WITHHELD (ladder arms missing)"

    # ---- V2: friction must matter ---------------------------------------
    v2 = {"phi_mu0_deg": phi("mu000"),
          "phi_mu055_deg": phi("kn1e7") if phi("kn1e7") is not None else phi("kn1e7_mob"),
          "phi_mu15_deg": phi("mu150")}
    if v2["phi_mu0_deg"] is not None:
        v2["mu0_ok"] = bool(abs(v2["phi_mu0_deg"]) <= V2_MU0_MAX_DEG)
    else:
        v2["mu0_ok"] = None
    if v2["phi_mu15_deg"] is not None and v2["phi_mu055_deg"] is not None:
        v2["mu_gap_deg"] = v2["phi_mu15_deg"] - v2["phi_mu055_deg"]
        v2["gap_ok"] = bool(v2["mu_gap_deg"] >= V2_MIN_GAP_DEG)
    else:
        v2["mu_gap_deg"] = None
        v2["gap_ok"] = None
    if v2["mu0_ok"] is False:
        v2["verdict"] = "FALSIFIED (frictionless packing reports a finite angle)"
    elif v2["mu0_ok"] and v2["gap_ok"]:
        v2["verdict"] = "SUPPORTED (control clean, friction transfers to macro)"
    elif v2["mu0_ok"] and v2["gap_ok"] is False:
        v2["verdict"] = "FALSIFIED (mu insensitive)"
    else:
        v2["verdict"] = "WITHHELD (control arms missing)"

    # ---- V3: rate independence ------------------------------------------
    v3 = {"phi_rate_half_deg": phi("rate_half"),
          "phi_baseline_deg": v2["phi_mu055_deg"]}
    if v3["phi_rate_half_deg"] is not None and v3["phi_baseline_deg"] is not None:
        v3["drift_deg"] = v3["phi_rate_half_deg"] - v3["phi_baseline_deg"]
        v3["rate_independent"] = bool(abs(v3["drift_deg"]) <= V3_MAX_DRIFT_DEG)
        v3["verdict"] = ("SUPPORTED (quasi-static, rate independent)"
                         if v3["rate_independent"] else
                         "FALSIFIED (rate dependent -> viscous flow stress)")
    else:
        v3["drift_deg"] = None
        v3["verdict"] = "WITHHELD (rate arm missing)"

    # ---- anchor judgement, unchanged by this study -----------------------
    baseline = v2["phi_mu055_deg"]
    anchor = {
        "reference_deg": ROWE_DEG,
        "reference_source": "Rowe (1962) D=0 limit arctan(mu=0.55)",
        "secondary_reference_deg": BOLTON_CRIT_DEG,
        "measured_deg": baseline,
        "relative_error": (abs(baseline - ROWE_DEG) / ROWE_DEG
                           if baseline is not None else None),
        "tolerance": TOLERANCE,
        "pass": bool(baseline is not None
                     and abs(baseline - ROWE_DEG) / ROWE_DEG <= TOLERANCE),
        "note": ("10x20 specimen, not the 24x48 anchor geometry: this row is a "
                 "diagnostic, the anchor verdict is the 24x48 record"),
    }
    return {"V1_stiffness": v1, "V2_friction": v2, "V3_rate": v3,
            "anchor_diagnostic": anchor}


def figure(payload: dict) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"matplotlib unavailable: {exc}")
        return None
    rows = payload["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # panel 1: stiffness ladder
    ax = axes[0]
    ladder = [r for r in rows if (r["mu"] or 0) == ROWE_MU
              and abs((r["rate_1_per_s"] or 0) - 0.05) < 1e-12
              and (r["vreg"] in (None, 0.01))
              and r["tag"].startswith("kn")
              and not r["tag"].endswith("_mob")
              and r["phi_cv_deg"] is not None]
    if ladder:
        x = [r["dimensionless_stiffness"] for r in ladder]
        y = [r["phi_cv_deg"] for r in ladder]
        ax.semilogx(x, y, "o-", color="#1f77b4", label=r"$\phi_{cv}$ (measured)")
        ax.errorbar([], [], fmt="none")
    ax.axhline(ROWE_DEG, color="#d62728", ls="--",
               label=f"Rowe arctan(mu) = {ROWE_DEG:.2f} deg")
    ax.axhline(BOLTON_CRIT_DEG, color="#7f7f7f", ls=":",
               label=f"Bolton critical = {BOLTON_CRIT_DEG:.0f} deg")
    ax.axvspan(1e3, 1e6, color="#2ca02c", alpha=0.10)
    ax.text(2e3, 2.0, "DEM practice\n$k_n/(p d)\\gtrsim10^3$", fontsize=8,
            color="#2ca02c")
    ax.set_xlabel(r"dimensionless stiffness $k_n/(p\,d)$")
    ax.set_ylabel(r"$\phi_{cv}$ [deg]")
    ax.set_title("(a) stiffness / overlap validity ladder")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # panel 2: micro validity
    ax = axes[1]
    if ladder:
        x = [r["dimensionless_stiffness"] for r in ladder]
        ov = [100.0 * (r["overlap_ratio_mean"] or 0.0) for r in ladder]
        z = [r["coordination_number"] for r in ladder]
        ax.semilogx(x, ov, "s-", color="#ff7f0e", label="overlap $\\delta/d$ [%]")
        ax.set_xlabel(r"$k_n/(p\,d)$")
        ax.set_ylabel("mean overlap [%]", color="#ff7f0e")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.semilogx(x, z, "^--", color="#9467bd", label="Z")
        ax2.axhline(3.0, color="#8c564b", ls=":", lw=1.0)
        ax2.set_ylabel("coordination number Z", color="#9467bd")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8,
                  loc="center right")
    ax.set_title("(b) contact-regime validity")

    # panel 3: friction and rate response
    ax = axes[2]
    mu_pts = [(r["mu"], r["phi_cv_deg"]) for r in rows
              if r["phi_cv_deg"] is not None
              and abs((r["rate_1_per_s"] or 0) - 0.05) < 1e-12
              and (r["kn_N_per_m"] or 0) == 1.0e7
              and (r["vreg"] in (None, 0.01))
              and not r["tag"].endswith("_mob")]
    mu_pts.sort()
    if mu_pts:
        ax.plot([m for m, _ in mu_pts], [p for _, p in mu_pts], "o-",
                color="#2ca02c", label=r"$\phi_{cv}$ vs $\mu$")
        mus = [m for m, _ in mu_pts if m > 0]
        if mus:
            ax.plot([0, max(mus)], [0, math.degrees(math.atan(max(mus)))],
                    "k:", lw=1.0, label="Rowe arctan($\\mu$)")
    rate_pts = [(r["rate_1_per_s"], r["phi_cv_deg"]) for r in rows
                if r["phi_cv_deg"] is not None
                and (r["kn_N_per_m"] or 0) == 1.0e7
                and (r["mu"] or 0) == ROWE_MU]
    rate_pts.sort()
    if len(rate_pts) > 1:
        ax.plot([x for x, _ in rate_pts], [y for _, y in rate_pts], "v--",
                color="#d62728", label=r"$\phi_{cv}$ vs $\dot\varepsilon_a$")
    ax.set_xlabel(r"interparticle $\mu$  /  $\dot\varepsilon_a$ [1/s]")
    ax.set_ylabel(r"$\phi_{cv}$ [deg]")
    ax.set_title("(c) friction + rate response ($k_n=10^7$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("CASE-DEM-BIAXIAL-FRICTION R10 · micro-mechanical validity of the "
                 "biaxial specimen (m22376)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    return os.path.relpath(FIG_PATH, REPO)


def markdown_table(rows: list[dict]) -> str:
    head = ("| arm | kn [N/m] | mu | rate [1/s] | phi_cv [deg] | peak [deg] | "
            "overlap/d | Z | a_c | a_f | q_drift | converged |")
    sep = "|" + "---|" * 12
    lines = [head, sep]
    for r in rows:
        lines.append("| {tag} | {kn:.0e} | {mu:.2f} | {rate:.4f} | {phi} | {peak} | "
                     "{ov} | {z} | {ac} | {af} | {qd} | {conv} |".format(
                         tag=r["tag"], kn=r["kn_N_per_m"] or 0.0,
                         mu=r["mu"] or 0.0, rate=r["rate_1_per_s"] or 0.0,
                         phi=_fmt(r["phi_cv_deg"], 2),
                         peak=_fmt(r["peak_phi_deg"], 2),
                         ov=_fmt(r["overlap_ratio_mean"], 4),
                         z=_fmt(r["coordination_number"], 2),
                         ac=_fmt(r["fabric_anisotropy_a_c"], 3),
                         af=_fmt(r["normal_force_anisotropy_a_f"], 3),
                         qd=_fmt(r["q_drift"], 3),
                         conv=r["converged"]))
    return "\n".join(lines)


def _fmt(v, nd):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main() -> int:
    arms = load_arms()
    if not arms:
        print(f"no arms found under {ARM_GLOB}")
        return 1
    payload = summarise(arms)
    payload["judgement"] = judge(payload["rows"])
    payload["figure"] = figure(payload)
    payload["markdown_table"] = markdown_table(payload["rows"])
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(payload["markdown_table"])
    for key in ("V1_stiffness", "V2_friction", "V3_rate"):
        print(f"\n[{key}] {json.dumps(payload['judgement'][key], default=str)}")
    print(f"\nanchor diagnostic: "
          f"{json.dumps(payload['judgement']['anchor_diagnostic'], default=str)}")
    print(f"\nwrote {os.path.relpath(OUT_JSON, REPO)}")
    if payload["figure"]:
        print(f"wrote {payload['figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())