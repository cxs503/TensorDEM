#!/usr/bin/env python3
"""Frozen estimator for the m23114 `silo_flow` numerical ladder (see prereg.md).

Pure arithmetic on the per-level JSON records, so every verdict can be
re-derived from the artefacts with one command:

    python3 benchmarks/m23114_siloflow_ladder/analyse.py --figures

`tests/test_m23114_siloflow_ladder.py` pins the estimator branches on synthetic
ground truth (known limit + order, all three verdict classes, the not-decidable
branch, the overlap geometry factor and the sign conventions).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "m23114_siloflow_ladder"
SUMMARY = RUNS / "ladder_summary.json"
DOC = ROOT / "docs" / "AUDIT_M23114_SILOFLOW_NUMERICS.md"
FIG_DIR = ROOT / "docs" / "figures"

C_LIT = 0.58
K_LIT = 1.4
G = 9.81
TOL_LEDGER = 0.10        # feature_ledger CASE-DEM-SILO-BEVERLOO acceptance
TOL_PLATFORM = 0.15      # platform precision standard

#: dt-axis levels in order of decreasing dt (steps per contact period)
DT_AXIS = (("nom20", 20), ("nom40", 40), ("nom80", 80))
#: kn-axis levels, nominal (= nom20) first
KN_AXIS = (("nom20", 5.0e3), ("kn2e4", 2.0e4), ("kn8e4", 8.0e4))

# frozen thresholds of the decision rule (prereg.md §3)
NEAR_ZERO_SHARE = 0.2
DOMINATED_SHARE = 0.5
MAX_ORDER = 6.0
KN_AGREEMENT_FACTOR = 3.0


# --------------------------------------------------------------------- helpers
def beverloo_ref(orifice: float, particle: float, rho_b: float,
                 c: float = C_LIT, k: float = K_LIT, gravity: float = G) -> float:
    """Literature-coefficient reference rate (never refitted)."""
    eff = orifice - k * particle
    if eff <= 0.0:
        return 0.0
    return c * rho_b * math.sqrt(gravity) * eff ** 2.5


def rel_error(w_meas: float, w_ref: float) -> float:
    if not (w_ref > 0.0) or not math.isfinite(w_ref):
        raise ValueError("reference rate must be finite and positive")
    if not math.isfinite(w_meas):
        raise ValueError("measured rate must be finite")
    return w_meas / w_ref - 1.0


def observed_order(w_coarse: float, w_mid: float, w_fine: float) -> float:
    """``p = log2((W1-W2)/(W2-W3))`` for a halving ladder; NaN if it cannot exist."""
    d12 = w_coarse - w_mid
    d23 = w_mid - w_fine
    if d23 == 0.0:
        return float("nan")
    ratio = d12 / d23
    if not math.isfinite(ratio) or ratio <= 0.0:
        return float("nan")
    return math.log(ratio) / math.log(2.0)


def richardson(w_coarse: float, w_mid: float, w_fine: float, order: float) -> float:
    """Limit estimate from three halving levels at the *observed* order."""
    if not math.isfinite(order) or order <= 0.0:
        return float("nan")
    return w_fine + (w_fine - w_mid) / (2.0 ** order - 1.0)


def classify_numerics(e_dt: float | None, e_phys: float | None) -> str:
    """Frozen decision rule of prereg.md §3."""
    if e_dt is None or e_phys is None:
        return "NOT_DECIDABLE"
    if not (math.isfinite(e_dt) and math.isfinite(e_phys)):
        return "NOT_DECIDABLE"
    share = abs(e_dt) / abs(e_phys) if e_phys != 0.0 else float("inf")
    if share < NEAR_ZERO_SHARE:
        return "NUMERICS-NEGLIGIBLE"
    if share < DOMINATED_SHARE:
        return "NUMERICS-MIXED"
    return "NUMERICS-DOMINATED"


def overlap_predicted_bias(kn: float, *, radius: float, orifice: float,
                           fill_height: float, density: float = 2500.0,
                           k: float = K_LIT, gravity: float = G) -> dict:
    """prereg.md §5 — rate bias implied by a soft-contact overlap ``δ``.

    ``W ∝ (D − k d)^{5/2}`` and an overlap shrinks the effective particle
    diameter by ``2δ`` (both partners), so ``ΔW/W ≈ 2.5·k·2δ/(D − k d)``.
    """
    mass = density * 4.0 / 3.0 * math.pi * radius ** 3
    m_eff = 0.5 * mass
    d = 2.0 * radius
    eff = orifice - k * d
    delta_static = mass * gravity / kn
    v_o = math.sqrt(2.0 * gravity * fill_height)
    delta_dyn = v_o * math.sqrt(m_eff / kn)
    scale = 2.5 * k * 2.0 / eff
    return {
        "delta_static_m": delta_static,
        "delta_dynamic_m": delta_dyn,
        "orifice_velocity_m_s": v_o,
        "predicted_bias_static": scale * delta_static,
        "predicted_bias_dynamic": scale * delta_dyn,
    }


def kn_story_agrees(e_kn: float, predicted: float) -> bool:
    """m22853's soft-contact claim survives only if sign and size agree."""
    if not (math.isfinite(e_kn) and math.isfinite(predicted)):
        return False
    if e_kn == 0.0 or predicted == 0.0:
        return False
    if (e_kn > 0.0) != (predicted > 0.0):
        return False
    ratio = max(abs(e_kn), abs(predicted)) / min(abs(e_kn), abs(predicted))
    return ratio <= KN_AGREEMENT_FACTOR


# ----------------------------------------------------------------- load record
def load_runs(pattern: str = "*.json") -> dict:
    """``{(level, seed): record}`` from the per-level artefacts."""
    out = {}
    for path in sorted(glob.glob(str(RUNS / pattern))):
        rec = json.loads(Path(path).read_text())
        tag = rec.get("m23114") or {}
        if "level" not in tag:
            continue
        out[(tag["level"], int(tag["seed"]))] = rec
    return out


def measured_rate(rec: dict, rho_convention: str = "measured") -> float:
    return float(rec["flow_rate_kg_s"])


def bulk_density(rec: dict, convention: str = "measured") -> float:
    if convention == "measured":
        return float(rec["rho_bulk_measured"])
    if convention == "nominal":
        return float(rec["rho_bulk"])
    raise ValueError("convention must be 'measured' or 'nominal'")


def case_geometry(rec: dict) -> dict:
    c = rec["case"]
    return {
        "orifice": float(c["orifice_diameter"]),
        "particle": float(c["particle_diameter"]),
        "silo_radius": float(c["silo_radius"]),
        "fill_height": float(c["fill_height"]),
        "kn": float(c["kn"]),
    }


# ------------------------------------------------------------------- analysis
def analyse(runs: dict, *, rho_convention: str = "measured") -> dict:
    seeds = sorted({s for (_, s) in runs})
    dt_levels = [lv for lv, _ in DT_AXIS if any((lv, s) in runs for s in seeds)]
    kn_levels = [lv for lv, _ in KN_AXIS if any((lv, s) in runs for s in seeds)]

    per_seed = {}
    for seed in seeds:
        row = {}
        for level, steps in DT_AXIS:
            rec = runs.get((level, seed))
            if rec is None:
                continue
            geo = case_geometry(rec)
            w = measured_rate(rec)
            rho = bulk_density(rec, rho_convention)
            row[level] = {
                "steps_per_contact": steps,
                "dt_s": float(rec["dt"]),
                "W_kg_s": w,
                "rho_b": rho,
                "W_ref_kg_s": beverloo_ref(geo["orifice"], geo["particle"], rho),
                "rel_vs_beverloo": rel_error(
                    w, beverloo_ref(geo["orifice"], geo["particle"], rho)),
                "windows": rec["flow_rate_windows_kg_s"],
                "window_particles": rec.get("window_particles"),
                "steps": rec.get("steps"),
                "wall_clock_s": rec.get("wall_clock_s"),
                "kn": geo["kn"],
            }
        per_seed[seed] = row

    # ---- N1: dt convergence
    n1 = {"levels_present": dt_levels, "per_seed": {}, "seed_mean": {}}
    for seed, row in per_seed.items():
        if all(lv in row for lv, _ in DT_AXIS):
            w1 = row["nom20"]["W_kg_s"]
            w2 = row["nom40"]["W_kg_s"]
            w3 = row["nom80"]["W_kg_s"]
            p = observed_order(w1, w2, w3)
            w_inf = richardson(w1, w2, w3, p)
            n1["per_seed"][seed] = {
                "W_20": w1, "W_40": w2, "W_80": w3,
                "order_obs": p,
                "W_inf": w_inf,
                "E_dt": (w_inf - w1) / w1 if math.isfinite(w_inf) else None,
                "rel_finest": row["nom80"]["rel_vs_beverloo"],
            }
    if len(n1["per_seed"]) == len(seeds) and seeds:
        e_list = [v["E_dt"] for v in n1["per_seed"].values() if v["E_dt"] is not None]
        p_list = [v["order_obs"] for v in n1["per_seed"].values()
                  if math.isfinite(v["order_obs"])]
        w_nom = [per_seed[s]["nom20"]["W_kg_s"] for s in seeds if "nom20" in per_seed[s]]
        seed_spread = (max(w_nom) - min(w_nom)) / (sum(w_nom) / len(w_nom)) if len(w_nom) > 1 else 0.0
        e_phys = per_seed[seeds[0]]["nom20"]["rel_vs_beverloo"]
        e_dt = sum(e_list) / len(e_list) if e_list else None
        p_mean = sum(p_list) / len(p_list) if p_list else float("nan")
        verdict = classify_numerics(e_dt, e_phys)
        if not p_list or p_mean <= 0.0 or p_mean > MAX_ORDER:
            verdict = "NOT_DECIDABLE"
        if e_dt is not None and seed_spread > abs(e_dt) and verdict == "NUMERICS-DOMINATED":
            verdict = "NOT_DECIDABLE"
        n1["seed_mean"] = {
            "W_nom": sum(w_nom) / len(w_nom) if w_nom else None,
            "E_dt_mean": e_dt,
            "order_obs_mean": p_mean,
            "E_phys": e_phys,
            "share_numeric_over_physics": (abs(e_dt) / abs(e_phys)
                                           if e_dt is not None and e_phys else None),
            "seed_spread_nominal": seed_spread,
            "verdict": verdict,
        }

    # ---- N2: stiffness ladder
    n2 = {"levels_present": kn_levels, "per_seed": {}}
    for seed in seeds:
        base_rec = runs.get(("nom20", seed))
        if base_rec is None:
            continue
        base = float(base_rec["flow_rate_kg_s"])
        g0 = case_geometry(base_rec)
        geo = {"orifice": g0["orifice"], "radius": 0.5 * g0["particle"],
               "fill_height": g0["fill_height"]}
        entry = {"W_nom": base, "kn_base": g0["kn"]}
        for level, kn in KN_AXIS:
            if level == "nom20":          # nom20 *is* the kn = 5e3 run
                continue
            rec = runs.get((level, seed))
            if rec is None:
                continue
            entry[f"kn_{level}"] = kn
            entry[f"W_{level}"] = float(rec["flow_rate_kg_s"])
            entry[f"E_{level}"] = (float(rec["flow_rate_kg_s"]) - base) / base
        if "E_kn8e4" in entry:
            pred = overlap_predicted_bias(entry["kn_kn8e4"], **geo)
            entry["predicted_bias"] = pred
            entry["kn_story_agrees"] = kn_story_agrees(
                entry["E_kn8e4"], pred["predicted_bias_dynamic"])
            entry["measured_over_predicted"] = (
                entry["E_kn8e4"] / pred["predicted_bias_dynamic"]
                if pred["predicted_bias_dynamic"] else None)
        n2["per_seed"][seed] = entry
    claims = [v.get("kn_story_agrees") for v in n2["per_seed"].values()
              if "kn_story_agrees" in v]
    n2["soft_contact_claim"] = ("SUBSTANTIATED" if claims and all(claims)
                                else "NOT_SUBSTANTIATED" if claims else "NOT_RUN")

    # ---- N3: ledger acceptance on the converged rate
    n3 = {"rho_convention": rho_convention, "per_seed": {}}
    for seed, row in per_seed.items():
        if "nom80" not in row:
            continue
        rec = runs[("nom80", seed)]
        geo = case_geometry(rec)
        rho = bulk_density(rec, rho_convention)
        w_ref = beverloo_ref(geo["orifice"], geo["particle"], rho)
        w_inf = n1["per_seed"].get(seed, {}).get("W_inf")
        finest = row["nom80"]["W_kg_s"]
        # both bulk-density conventions, so the amplitude verdict can not be
        # confused with a convention choice (m20232 found a 14 % gap between them)
        rho_nom = bulk_density(rec, "nominal")
        w_ref_nom = beverloo_ref(geo["orifice"], geo["particle"], rho_nom)
        n3["per_seed"][seed] = {
            "rel_finest_nominal_convention": rel_error(finest, w_ref_nom),
            "rho_b_nominal_kg_m3": rho_nom,
            "W_ref_nominal_kg_s": w_ref_nom,
            "W_ref_kg_s": w_ref,
            "rho_b_kg_m3": rho,
            "W_finest_kg_s": finest,
            "rel_finest": rel_error(finest, w_ref),
            "W_inf_kg_s": w_inf,
            "rel_inf": rel_error(w_inf, w_ref) if w_inf and math.isfinite(w_inf) else None,
        }
    rels = [v["rel_inf"] for v in n3["per_seed"].values() if v["rel_inf"] is not None]
    n3["rel_inf_mean"] = sum(rels) / len(rels) if rels else None
    n3["pass_ledger_10pct"] = (abs(n3["rel_inf_mean"]) < TOL_LEDGER
                               if n3["rel_inf_mean"] is not None else None)
    n3["pass_platform_15pct"] = (abs(n3["rel_inf_mean"]) < TOL_PLATFORM
                                 if n3["rel_inf_mean"] is not None else None)

    return {
        "anchor": "CASE-DEM-SILO-BEVERLOO",
        "kernel": "tensordem.silo_flow (kernel B; dt-convergence never audited before)",
        "coefficients": {"C": C_LIT, "k": K_LIT, "source": "Beverloo et al. 1961"},
        "rho_convention": rho_convention,
        "seeds": seeds,
        "N1_dt_convergence": n1,
        "N2_stiffness": n2,
        "N3_ledger": n3,
        "runs_present": sorted(f"{lv}_s{s}" for (lv, s) in runs),
    }


def render_doc(summary: dict) -> str:
    n1 = summary["N1_dt_convergence"]
    n2 = summary["N2_stiffness"]
    n3 = summary["N3_ledger"]
    sm = n1.get("seed_mean") or {}

    def pct(v, digits=2):
        return "n/a" if v is None or not isinstance(v, (int, float)) else f"{100.0 * v:+.{digits}f} %"

    rows = []
    for seed, v in n1["per_seed"].items():
        rows.append(f"| {seed} | {v['W_20']:.6f} | {v['W_40']:.6f} | {v['W_80']:.6f} | "
                    f"{v['order_obs']:.3f} | {v['W_inf']:.6f} | {pct(v['E_dt'])} |")
    rows_kn = []
    for seed, v in n2["per_seed"].items():
        rows_kn.append(
            f"| {seed} | {v['W_nom']:.6f} | "
            f"{v.get('W_kn2e4', float('nan')):.6f} | {v.get('W_kn8e4', float('nan')):.6f} | "
            f"{pct(v.get('E_kn2e4'))} | {pct(v.get('E_kn8e4'))} | "
            f"{pct((v.get('predicted_bias') or {}).get('predicted_bias_dynamic'))} | "
            f"{v.get('measured_over_predicted', float('nan')):.2f} |")
    rows_n3 = []
    for seed, v in n3["per_seed"].items():
        rows_n3.append(f"| {seed} | {v['rho_b_kg_m3']:.1f} | {v['W_ref_kg_s']:.6f} | "
                       f"{v['W_finest_kg_s']:.6f} | {pct(v['rel_finest'])} | "
                       f"{v['W_inf_kg_s']:.6f} | {pct(v['rel_inf'])} |")

    return f"""# Numerical convergence audit of the `silo_flow` Beverloo kernel (m23114)

Anchor: `{summary['anchor']}` · kernel: `{summary['kernel']}` ·
reference: Beverloo, Leniger & van de Velde (1961),
*Chem. Eng. Sci.* **15**, 260–269, Eq. (4),
`W = C ρ_b √g (D − k d)^(5/2)`, `C = {summary['coefficients']['C']}`,
`k = {summary['coefficients']['k']}` (literature, **not** refitted).
Bulk-density convention for the reference: **{summary['rho_convention']}**
(settled bed surface, `rho_bulk_measured`).

*This document is rendered from `runs/m23114_siloflow_ladder/ladder_summary.json`
— no number in it is hand-copied.*

## 1. Why this audit exists

Twelve rounds of `CASE-DEM-SILO-BEVERLOO` produced a `+25 %` FAIL and a dozen
`NOT_RUN`s, all with the `silo_flow` kernel at its **default** resolution
`dt = T_c/20` (`Material.dt = 0.05·contact_time`). Kernel A (`silo_discharge`)
had its time step audited (`docs/AUDIT_M20055_DT_CONVERGENCE.md`); kernel **B**,
which carries every recent verdict, never did. This audit measures the
discretisation bias of the campaign's own setting and thereby decides whether
the filed errors are physics or numerics.

## 2. Method

Same frozen case as the campaign (`d = 3 mm`, `D/d = 5`, `H = 36 mm`,
`μ = 0.5`, smooth plate, 8 × 0.05 s windows, `W` = mean of the last 50 %),
only the integration resolution (20 / 40 / 80 steps per contact period) and the
contact stiffness (`k_n = 5e3 / 2e4 / 8e4 N/m`) vary. Limit estimate at the
*observed* order:

```
p_obs = log2((W20 − W40)/(W40 − W80));  W∞ = W80 + (W80 − W40)/(2^p_obs − 1)
```

## 3. N1 — is the default time step converged?

| seed | W(20) | W(40) | W(80) | p_obs | W∞ | E_dt = (W∞ − W20)/W20 |
|---|---|---|---|---|---|---|
{os.linesep.join(rows) if rows else '| — | NOT_RUN | | | | | |'}

* mean observed order `p = {sm.get('order_obs_mean', float('nan')):.3f}`
* `E_dt = {pct(sm.get('E_dt_mean'))}` versus the campaign's own discrepancy
  `E_phys = {pct(sm.get('E_phys'))}` → numeric share
  `{sm.get('share_numeric_over_physics', float('nan')):.3f}`
* seed scatter of the nominal level `{pct(sm.get('seed_spread_nominal'))}`

**Verdict N1: {sm.get('verdict', 'NOT_RUN')}** (frozen rule: < 0.2 →
NUMERICS-NEGLIGIBLE, < 0.5 → MIXED, ≥ 0.5 → DOMINATED; `p_obs ≤ 0` or seed
scatter dominating → NOT_DECIDABLE).

## 4. N2 — does the "soft contact" explanation hold?

`E_kn` is the measured rate change relative to `k_n = 5e3 N/m`; the predicted
column is the overlap mechanism of `prereg.md` §5, `ΔW/W ≈ 2.5·k·2δ_v/(D − k d)`
with `δ_v = v_o√(m_eff/k_n)` at `k_n = 8e4 N/m`.

| seed | W(5e3) | W(2e4) | W(8e4) | E_kn(2e4) | E_kn(8e4) | predicted (8e4) | measured/predicted |
|---|---|---|---|---|---|---|---|
{os.linesep.join(rows_kn) if rows_kn else '| — | NOT_RUN | | | | | |'}

**Soft-contact claim (m22853 §1): {n2.get('soft_contact_claim', 'NOT_RUN')}**
(survives only if sign and magnitude agree within a factor
{KN_AGREEMENT_FACTOR:g}).

## 5. N3 — the ledger acceptance on the converged rate

| seed | ρ_b [kg/m³] | W_ref | W(80) | rel(80) | W∞ | rel(∞) |
|---|---|---|---|---|---|---|
{os.linesep.join(rows_n3) if rows_n3 else '| — | NOT_RUN | | | | | |'}

* mean `rel(∞) = {pct(n3.get('rel_inf_mean'))}`
* ledger acceptance `< 10 %`: **{n3.get('pass_ledger_10pct')}**
* platform standard `< 15 %`: **{n3.get('pass_platform_15pct')}**

## 6. Reproduction

```bash
PYTHONPATH=src python benchmarks/m23114_siloflow_ladder/run_ladder.py --list
PYTHONPATH=src python benchmarks/m23114_siloflow_ladder/run_ladder.py --shard 1 --nshards 10
python3 benchmarks/m23114_siloflow_ladder/analyse.py --figures
python3 -m pytest tests/test_m23114_siloflow_ladder.py -q
```
"""


def make_figures(summary: dict, runs: dict, rho_convention: str) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    n1 = summary["N1_dt_convergence"]

    # --- figure 1: W(dt)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    steps = [s for _, s in DT_AXIS]
    for seed, v in n1["per_seed"].items():
        ys = [v["W_20"], v["W_40"], v["W_80"]]
        ax.plot(steps, ys, "o-", label=f"seed {seed}")
        ax.annotate("", xy=(steps[-1] * 1.15, v["W_inf"]),
                    xytext=(steps[-1] * 1.15, ys[-1]),
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=1))
    finite = [v for v in n1["per_seed"].values() if math.isfinite(v["W_inf"])]
    if finite:
        w_inf = sum(v["W_inf"] for v in finite) / len(finite)
        ax.axhline(w_inf, ls="--", color="k", lw=1, label=f"W∞ (Richardson) = {w_inf:.4f}")
        w_ref = None
        for (lv, s), rec in runs.items():
            if lv == "nom20":
                geo = case_geometry(rec)
                w_ref = beverloo_ref(geo["orifice"], geo["particle"],
                                     bulk_density(rec, rho_convention))
                break
        if w_ref:
            ax.axhline(w_ref, ls=":", color="tab:red", lw=1.2,
                       label=f"Beverloo C={C_LIT}, k={K_LIT} = {w_ref:.4f}")
            ax.fill_between([min(steps) * 0.9, max(steps) * 1.25],
                            w_ref * 0.9, w_ref * 1.1, color="tab:red", alpha=0.10,
                            label="±10 % ledger band")
            ax.set_xlim(min(steps) * 0.9, max(steps) * 1.25)
    ax.set_xlabel("steps per contact period  (dt = T_c / n)")
    ax.set_ylabel("measured discharge rate W [kg/s]")
    ax.set_title("m23114 — `silo_flow` kernel: W versus time-step resolution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = FIG_DIR / "m23114_siloflow_dt_ladder.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))

    # --- figure 2: W(kn) with the overlap prediction
    n2 = summary["N2_stiffness"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    kns = [k for _, k in KN_AXIS]
    for seed, v in n2["per_seed"].items():
        ys = [v["W_nom"], v.get("W_kn2e4", float("nan")), v.get("W_kn8e4", float("nan"))]
        ax.plot(kns, ys, "s-", label=f"seed {seed}")
    first = next(iter(n2["per_seed"].values()), None)
    if first:
        base_kn = 5.0e3
        geo = {"orifice": 0.015, "particle": 0.003, "fill_height": 0.036, "radius": 0.0015}
        pred = [first["W_nom"] * (1.0 + overlap_predicted_bias(
            kn, **geo)["predicted_bias_dynamic"]) for kn in kns]
        ax.plot(kns, pred, "k--", lw=1, label="overlap mechanism (prereg §5)")
    ax.set_xscale("log")
    ax.set_xlabel("contact stiffness k_n [N/m]  (20 steps per contact period)")
    ax.set_ylabel("measured discharge rate W [kg/s]")
    ax.set_title("m23114 — soft-contact hypothesis test")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    path = FIG_DIR / "m23114_siloflow_kn_ladder.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(str(path.relative_to(ROOT)))
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(RUNS))
    ap.add_argument("--rho", default="measured", choices=("measured", "nominal"))
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--no-doc", action="store_true")
    args = ap.parse_args(argv)

    globals()["RUNS"] = Path(args.dir)
    runs = load_runs()
    if not runs:
        raise SystemExit(f"no run records under {RUNS}")
    summary = analyse(runs, rho_convention=args.rho)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "ladder_summary.json").write_text(json.dumps(summary, indent=1))
    if args.figures:
        summary["figures"] = make_figures(summary, runs, args.rho)
        (RUNS / "ladder_summary.json").write_text(json.dumps(summary, indent=1))
    if not args.no_doc:
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(render_doc(summary))
        print(f"wrote {DOC}")
    print(json.dumps({k: summary[k] for k in ("N1_dt_convergence",)})[:1200])
    print("N1 verdict:", summary["N1_dt_convergence"].get("seed_mean", {}).get("verdict"))
    print("N2 claim  :", summary["N2_stiffness"].get("soft_contact_claim"))
    print("N3 rel_inf:", summary["N3_ledger"].get("rel_inf_mean"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())