"""CASE-DEM-PLANE-SHEAR-MUIDI round 23: seed ensemble + contact-friction ladder.

Why this round
--------------
Round 22 (``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R22.md`` §⑦) closed the *citation*
question: the yardstick is da Cruz 2005 Eq. (13)/(14) as archived and parsed
from the paper, ``mu(I) = 0.22 + 1.00 I`` and ``phi(I) = 0.81 - 0.30 I`` for
``I <= 0.2``.  Against that law the archived four points (round 14,
``runs/full_cell/point_*.json``) miss by 16-57 % and ``mu(I)`` is
non-monotone (a -15 % step between ``I_bulk = 0.035`` and ``0.074``).

Round 22 could not say whether that is a property of the *device* or of the
*realisation*: every archived point is **one seed** (5279).  A -15 % dip is
the same size as the 10 % acceptance band, so until the realisation spread is
measured the ledger verdict is unfalsifiable in either direction.  This round
measures it, and re-runs round 22's recommendation 1 (the contact-friction
ladder) as a bounded add-on, because a friction bias is the one defect that
*would* move ``mu`` toward the archived law.

Protocol
--------
* **Ensemble**: the four archived inertial-number targets
  ``I = 0.01, 0.03, 0.1, 0.3`` x seeds ``(5279, 6113, 7919, 104729, 1299709,
  2000003)`` = 24 runs.  Seed 5279 is the archived seed, so the ensemble
  contains a **parity point** against ``runs/full_cell/point_*.json``: the
  device is deterministic, so a mismatch there is protocol drift, not noise.
* **Ladder**: ``mu_contact = 0.2, 0.3, 0.4, 0.5, 0.7`` at ``I = 0.03``,
  seed 5279, everything else identical = 5 runs.
* No contact law, protocol, geometry or reference constant is changed, and no
  estimator is redefined; the axes are *realisation* and *micro-friction*.

Judging
-------
Ordinates are judged against the **archived** law (``reference_archive``), and
only on ``I_bulk <= I_sat = 0.2`` -- an ``I`` beyond the law's validity is
reported as an extrapolation and is not judged (round 22, Table D).

Pre-registered predictions (written to ``runs/seed_ensemble/prereg.json``
*before* any run, and never edited afterwards):

* P1 (spread): at every judged point the seed coefficient of variation
  ``CV = sd/mean`` of ``mu_full_zz_final`` is reported.  Prediction:
  ``CV < 10 %``, i.e. the acceptance band is resolvable by one realisation.
* P2 (dip): the ``I = 0.03 -> 0.1`` step in ``mu`` is tested paired over seeds
  (one-sided, the measured curve is expected to *decrease*).  Prediction: the
  dip survives the ensemble (median paired change < 0 and the two 95 %
  confidence intervals disjoint), i.e. it is systematic, not noise.
* P3 (bias): the ensemble mean ``mu`` stays outside the 10 % band against the
  archived law at ``I = 0.03`` and ``I = 0.1``.  Prediction: outside (FAIL
  reproduced with error bars).
* P4 (friction): lowering ``mu_contact`` toward da Cruz's ``mu_p = 0.4``
  lowers ``mu``, and at ``I = 0.03`` the ladder does **not** land inside 10 %
  of ``0.22 + 1.00 I = 0.25``.  Prediction: monotone increase with
  ``mu_contact``, ladder minimum still above the band.

Usage
-----
    python benchmarks/plane_shear_mu_i/seed_ensemble.py --shard 0 --nshards 12
    python benchmarks/plane_shear_mu_i/seed_ensemble.py --parity   # 1 run, archive check
    python benchmarks/plane_shear_mu_i/seed_ensemble.py --assemble # stats + figures
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

RUNS = HERE / "runs" / "seed_ensemble"
RUNS.mkdir(parents=True, exist_ok=True)
ARCHIVE = HERE / "runs" / "full_cell"

#: the archived device configuration (``full_cell_rheometry.run_point``).  The
#: numbers are *imported*, never re-typed, so a device change cannot silently
#: separate this round from the archived points.
from full_cell_rheometry import (  # noqa: E402
    D, E, H, KN, MIN_SHEAR_STEPS, MU_CONTACT, MU_ROLLING, P_APPLIED, RADIUS,
    SEED, SETTLE_STEPS, SHEAR_STRAIN, W,
)

I_TARGETS = (1e-2, 3e-2, 1e-1, 3e-1)
SEEDS = (5279, 6113, 7919, 104729, 1299709, 2000003)
LADDER_I = 3e-2
LADDER_MU = (0.2, 0.3, 0.4, 0.5, 0.7)
TOL = 0.10

#: ordinates judged (all from the same state sample; names as archived)
MU_ORDINATES = ("mu_full_zz_final", "mu_full_final", "mu_wall_final",
                "mu_slab_mid_final", "mu_slab_both_final")
PHI_ORDINATES = ("phi_slab", "phi_bed_steady")

PREREG = {
    "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "round": 23,
    "question": "is the mu(I) dip and the 16-57 % offset of round 22 a "
                "realisation artefact or a device property?",
    "device": {"simulator": "tensordem.mu_i_rheology.simulate_servo_shear",
               "P_applied_Pa": P_APPLIED, "kn_N_per_m": KN,
               "mu_contact": MU_CONTACT, "mu_rolling": MU_ROLLING,
               "restitution": E, "radius_m": RADIUS,
               "box_m": [W, D, H], "settle_steps": SETTLE_STEPS,
               "min_shear_steps": MIN_SHEAR_STEPS,
               "shear_strain": SHEAR_STRAIN},
    "axes": {"I_targets": list(I_TARGETS), "seeds": list(SEEDS),
             "ladder_I": LADDER_I, "ladder_mu_contact": list(LADDER_MU)},
    "reference": "archived da Cruz 2005 Eq. (13)/(14): mu = 0.22 + 1.00 I, "
                 "phi = 0.81 - 0.30 I, valid for I <= 0.2",
    "tolerance": TOL,
    "predictions": {
        "P1_spread": "CV(mu_full_zz_final) < 10 % at every judged point",
        "P2_dip": "the I=0.03 -> 0.1 decrease survives the ensemble "
                  "(median paired delta < 0 and disjoint 95 % CIs)",
        "P3_bias": "ensemble mean mu outside the 10 % band at I = 0.03 and 0.1",
        "P4_friction": "mu increases with mu_contact; at I = 0.03 the ladder "
                       "does not reach the 10 % band around 0.25",
    },
    "decision_rules": {
        "P1": "CV <= 0.10 -> one realisation resolves the band",
        "P2": "dip is systematic if the paired differences are all negative "
              "and the mean change is more than 2 standard errors from zero",
        "P3": "outside the band if |mean/law - 1| > 10 % and the 95 % CI of "
              "the mean excludes the law value",
    },
}


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def servo_case(I: float, seed: int, mu_contact: float = MU_CONTACT):
    from tensordem.mu_i_rheology import ServoShearCase

    return ServoShearCase(
        radius=RADIUS, density=2500.0, mu=mu_contact, mu_rolling=MU_ROLLING,
        restitution=E, kn=KN, kt_ratio=0.5, box_width=W, box_depth=D,
        fill_height=H, normal_stress=P_APPLIED, inertial_number=I, seed=int(seed),
        settle_steps=SETTLE_STEPS, min_shear_steps=MIN_SHEAR_STEPS,
        shear_strain=SHEAR_STRAIN, bulk_stress=True)


def row_from_result(res: dict, I: float, seed: int, mu_contact: float) -> dict:
    """Same field names as ``runs/full_cell/point_*.json`` where they exist."""
    row = {
        "I_target": I, "seed": int(seed), "mu_contact": mu_contact,
        "I_measured": res["inertial_number"],
        "I_bulk": res["inertial_number_bulk"],
        "mu_wall": res["mu_effective"], "mu_wall_final": res["mu_final"],
        "mu_slab_both": res["bulk_mu"], "mu_slab_both_final": res["bulk_mu_final"],
        "mu_slab_mid": res["bulk_mid_mu"], "mu_slab_mid_final": res["bulk_mid_mu_final"],
        "mu_full": res["full_mu"], "mu_full_final": res["full_mu_final"],
        "mu_full_zz": res["full_mu_zz"], "mu_full_zz_final": res["full_mu_zz_final"],
        "anisotropy_p_over_zz": res["full_anisotropy"],
        "p_full": res["full_pressure_pa"],
        "sigma_zz_full_final": res["full_sigma_zz_final"],
        "momentum_balance_final": res["full_momentum_balance_final"],
        "phi_slab": res["solid_fraction"],
        "phi_bed_steady": res["phi_bed_steady"],
        "dilatancy": res["dilatancy"],
        "coordination": res["coordination_number"],
        "n_particles": res["n_particles"],
        "shear_rate_imposed": res["shear_rate_imposed"],
        "shear_rate_measured": res["shear_rate_measured"],
        "profile_r2": res["profile_r2"],
        "pressure_control_error_final": res["pressure_control_error_final"],
        "servo_valid": res["servo_valid"],
        "recover_steps": res["recover_steps"],
        "wall_clock_s": res["wall_clock_s"],
    }
    return row


def run_one(I: float, seed: int, mu_contact: float = MU_CONTACT) -> dict:
    from tensordem.mu_i_rheology import simulate_servo_shear

    res = simulate_servo_shear(servo_case(I, seed, mu_contact))
    return row_from_result(res, I, seed, mu_contact)


def _point_path(I: float, seed: int, mu_contact: float) -> Path:
    tag = "" if mu_contact == MU_CONTACT else f"__mup{mu_contact:.2f}"
    return RUNS / f"point_I{I:.3g}_s{seed}{tag}.json"


def _jobs() -> list:
    jobs = [(I, s, MU_CONTACT) for I in I_TARGETS for s in SEEDS]
    jobs += [(LADDER_I, SEED, m) for m in LADDER_MU]
    return jobs


def run_shard(shard: int, nshards: int) -> int:
    jobs = _jobs()
    mine = [j for k, j in enumerate(jobs) if k % nshards == shard]
    for I, seed, mup in mine:
        path = _point_path(I, seed, mup)
        if path.exists():
            print(f"skip {path.name}", flush=True)
            continue
        row = run_one(I, seed, mup)
        path.write_text(json.dumps(row, indent=2))
        print(f"{path.name}: I_bulk={row['I_bulk']:.4f} "
              f"mu_full_zz={row['mu_full_zz_final']:.4f} "
              f"mu_slab_both={row['mu_slab_both_final']:.4f} "
              f"phi_slab={row['phi_slab']:.4f} "
              f"({row['wall_clock_s']:.0f}s)", flush=True)
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def mean_sd(values) -> dict:
    v = [float(x) for x in values]
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "cv": float("nan"), "sem": float("nan")}
    m = sum(v) / n
    if n == 1:
        return {"n": 1, "mean": m, "sd": 0.0, "cv": 0.0, "sem": 0.0}
    var = sum((x - m) ** 2 for x in v) / (n - 1)
    sd = math.sqrt(var)
    return {"n": n, "mean": m, "sd": sd,
            "cv": (sd / abs(m) if m else float("nan")),
            "sem": sd / math.sqrt(n)}


#: two-sided 95 % Student t quantiles by degrees of freedom (n - 1)
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


def ci95(values) -> tuple:
    st = mean_sd(values)
    n = st["n"]
    if n < 2:
        return (st["mean"], st["mean"])
    t = _T975.get(n - 1, 1.96)
    half = t * st["sem"]
    return (st["mean"] - half, st["mean"] + half)


def archived_law():
    from reference_archive import archived_law as _law

    return _law(load_manifest())


def load_manifest():
    from reference_archive import load_manifest as _lm

    return _lm()


def law_mu(I: float):
    """``(mu_law, in_validity)`` from the archived da Cruz fit."""
    law = archived_law()
    v = law.mu(I)
    if v is None:
        return law.mu_extrapolated(I), False
    return v, True


def rows(kind: str = "ensemble") -> list:
    out = []
    for I in I_TARGETS:
        for s in SEEDS:
            p = _point_path(I, s, MU_CONTACT)
            if p.exists():
                d = json.loads(p.read_text())
                d["_file"] = str(p.relative_to(HERE))
                out.append(d)
    if kind == "ladder":
        out = []
        for m in LADDER_MU:
            p = _point_path(LADDER_I, SEED, m)
            if p.exists():
                d = json.loads(p.read_text())
                d["_file"] = str(p.relative_to(HERE))
                out.append(d)
    return out


def analyse() -> dict:
    ens = rows()
    ladder = rows("ladder")
    by_I = {}
    for I in I_TARGETS:
        pts = [r for r in ens if abs(r["I_target"] - I) < 1e-12]
        pts.sort(key=lambda r: r["seed"])
        mu_law, in_validity = law_mu(I)
        entry = {"I_target": I, "n_seeds": len(pts),
                 "I_bulk_mean": mean_sd([r["I_bulk"] for r in pts]),
                 "mu_law_archived": mu_law, "law_in_validity": in_validity,
                 "seeds": [r["seed"] for r in pts],
                 "I_bulk_values": [r["I_bulk"] for r in pts],
                 "mu": {}, "phi": {}}
        for name in MU_ORDINATES:
            vals = [r[name] for r in pts]
            st = mean_sd(vals)
            lo, hi = ci95(vals)
            st.update({"ci95": [lo, hi], "values": vals,
                       "rel_err_mean_vs_law": (st["mean"] / mu_law - 1.0),
                       "rel_err_per_seed": [v / mu_law - 1.0 for v in vals],
                       "n_within_tol": sum(1 for v in vals
                                           if abs(v / mu_law - 1.0) <= TOL),
                       "ci_excludes_law": bool(lo > mu_law or hi < mu_law)})
            entry["mu"][name] = st
        for name in PHI_ORDINATES:
            st = mean_sd([r[name] for r in pts])
            lo, hi = ci95([r[name] for r in pts])
            st.update({"ci95": [lo, hi]})
            entry["phi"][name] = st
        entry["servo_valid"] = [bool(r["servo_valid"]) for r in pts]
        by_I[f"{I:.3g}"] = entry

    # -- P2: paired dip between I = 0.03 and I = 0.1, paired by seed --------
    lo_I, hi_I = 3e-2, 1e-1
    low = {r["seed"]: r["mu_full_zz_final"] for r in ens
           if abs(r["I_target"] - lo_I) < 1e-12}
    high = {r["seed"]: r["mu_full_zz_final"] for r in ens
            if abs(r["I_target"] - hi_I) < 1e-12}
    common = sorted(set(low) & set(high))
    deltas = [high[s] - low[s] for s in common]
    rel = [(high[s] - low[s]) / low[s] for s in common]
    dip = {"I_low": lo_I, "I_high": hi_I, "ordinate": "mu_full_zz_final",
           "seeds": common, "delta": deltas, "relative": rel,
           "n_pairs": len(common),
           "delta_mean": mean_sd(deltas),
           "rel_mean": mean_sd(rel),
           "n_negative": sum(1 for d in deltas if d < 0),
           "ci95_delta": list(ci95(deltas))}
    if len(deltas) >= 2:
        st = mean_sd(deltas)
        dip["systematic"] = bool(st["mean"] < 0 and
                                 abs(st["mean"]) > 2.0 * st["sem"] and
                                 dip["n_negative"] >= len(deltas) - 1)
    else:
        dip["systematic"] = None

    # -- P4: contact-friction ladder ---------------------------------------
    lad = []
    mu_law_l, in_validity_l = law_mu(LADDER_I)
    for m in LADDER_MU:
        hit = [r for r in ladder if abs(r["mu_contact"] - m) < 1e-12]
        if not hit:
            continue
        r = hit[0]
        lad.append({"mu_contact": m, "I_bulk": r["I_bulk"],
                    "mu_full_zz": r["mu_full_zz_final"],
                    "mu_slab_both": r["mu_slab_both_final"],
                    "mu_wall": r["mu_wall_final"],
                    "phi_slab": r["phi_slab"],
                    "rel_err_vs_law": r["mu_full_zz_final"] / mu_law_l - 1.0,
                    "servo_valid": bool(r["servo_valid"]),
                    "wall_clock_s": r["wall_clock_s"]})
    ladder_ok = [p["mu_contact"] for p in lad]
    monotone = all(lad[i + 1]["mu_full_zz"] >= lad[i]["mu_full_zz"]
                   for i in range(len(lad) - 1)) if len(lad) > 1 else None
    ladder_min = min((p["rel_err_vs_law"] for p in lad), default=float("nan"))
    ladder_summary = {"I": LADDER_I, "mu_law_archived": mu_law_l,
                      "law_in_validity": in_validity_l,
                      "mu_contacts_present": ladder_ok,
                      "monotone_in_mu_contact": monotone,
                      "min_rel_err_vs_law": ladder_min,
                      "points": lad}

    # -- parity with the archived single-seed points -----------------------
    parity = []
    for I in I_TARGETS:
        p = ARCHIVE / f"point_{I:.3g}.json"
        hit = [r for r in ens if abs(r["I_target"] - I) < 1e-12
               and r["seed"] == SEED]
        if not (p.exists() and hit):
            continue
        old, new = json.loads(p.read_text()), hit[0]
        for key in ("mu_full_zz_final", "mu_slab_both_final", "mu_full_final",
                    "phi_slab", "phi_bed_steady", "I_bulk"):
            a, b = old.get(key), new.get(key)
            if a is None or b is None:
                continue
            parity.append({"I": I, "key": key, "archived": a, "rerun": b,
                           "rel_diff": (b - a) / a if a else float("nan"),
                           "bitwise_equal": a == b})

    # -- verdict ----------------------------------------------------------
    judged = [k for k, v in by_I.items() if v["law_in_validity"]]
    inside = [k for k in judged
              if by_I[k]["mu"]["mu_full_zz_final"]["n_within_tol"]
              == by_I[k]["n_seeds"]]
    cv_all = {k: by_I[k]["mu"]["mu_full_zz_final"]["cv"] for k in judged}
    out = {
        "prereg": PREREG,
        "points": by_I,
        "dip": dip,
        "ladder": ladder_summary,
        "parity_vs_archived": parity,
        "verdict": {
            "judged_points": judged,
            "points_all_seeds_within_tol": inside,
            "passes_acceptance": len(inside) == len(judged) and len(judged) > 0,
            "max_cv_judged": max(cv_all.values()) if cv_all else float("nan"),
            "P1_spread_holds": (max(cv_all.values()) < TOL if cv_all else None),
            "P2_dip_systematic": dip["systematic"],
            "P3_bias_reproduced": all(
                abs(by_I[k]["mu"]["mu_full_zz_final"]["rel_err_mean_vs_law"]) > TOL
                for k in judged if k in ("0.03", "0.1")),
            "P4_ladder_monotone": monotone,
            "P4_ladder_reaches_band": (abs(ladder_min) <= TOL
                                       if ladder_min == ladder_min else None),
            "parity_max_rel_diff": max((abs(p["rel_diff"]) for p in parity),
                                       default=float("nan")),
        },
    }
    return out


def write_tables(a: dict) -> str:
    L = ["# R23 seed ensemble — programmatic tables", "",
         "> generated by `benchmarks/plane_shear_mu_i/seed_ensemble.py --assemble`; "
         "no number below is hand-copied.", "",
         f"Reference: archived da Cruz 2005 fit, `mu = {PREREG_LAW}` (valid `I <= 0.2`).",
         "", "## (1) seed spread per judged point (`mu_full_zz_final`)", "",
         "| I_target | n | mean | sd | CV % | 95 % CI | law | mean err % | seeds in band |"]
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, v in a["points"].items():
        m = v["mu"]["mu_full_zz_final"]
        L.append(f"| {k} | {m['n']} | {m['mean']:.4f} | {m['sd']:.4f} | "
                 f"{100*m['cv']:+.1f} | [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}] | "
                 f"{v['mu_law_archived']:.4f} | {100*m['rel_err_mean_vs_law']:+.1f} | "
                 f"{m['n_within_tol']}/{m['n']} |")
    L += ["", "## (2) all five stress ordinates, ensemble mean (CV %)", "",
          "| I_target | " + " | ".join(MU_ORDINATES) + " |",
          "|---|" + "---|" * len(MU_ORDINATES)]
    for k, v in a["points"].items():
        cell = " | ".join(f"{v['mu'][n]['mean']:.4f} ({100*v['mu'][n]['cv']:+.0f})"
                          for n in MU_ORDINATES)
        L.append(f"| {k} | {cell} |")
    L += ["", "## (3) solid fraction (definitions unchanged, spread only)", "",
          "| I_target | phi_slab mean | CV % | phi_bed_steady mean | CV % |",
          "|---|---|---|---|---|"]
    for k, v in a["points"].items():
        s, b = v["phi"]["phi_slab"], v["phi"]["phi_bed_steady"]
        L.append(f"| {k} | {s['mean']:.4f} | {100*s['cv']:+.1f} | "
                 f"{b['mean']:.4f} | {100*b['cv']:+.1f} |")
    d = a["dip"]
    L += ["", "## (4) paired dip test, `I_bulk` 0.03 -> 0.1 (`mu_full_zz_final`)", "",
          f"{d['n_pairs']} seed pairs; mean change {d['delta_mean']['mean']:+.4f} "
          f"({100*d['rel_mean']['mean']:+.1f} %), sd {d['delta_mean']['sd']:.4f}, "
          f"95 % CI [{d['ci95_delta'][0]:+.4f}, {d['ci95_delta'][1]:+.4f}], "
          f"negative in {d['n_negative']}/{d['n_pairs']} pairs, "
          f"systematic = **{d['systematic']}**.", "",
          "| seed | low | high | delta | relative % |", "|---|---|---|---|---|"]
    for i, s in enumerate(d["seeds"]):
        L.append(f"| {s} | — | — | {d['delta'][i]:+.4f} | {100*d['relative'][i]:+.1f} |")
    lad = a["ladder"]
    L += ["", f"## (5) contact-friction ladder at `I = {lad['I']:.3g}` (seed {SEED})", "",
          "| mu_contact | I_bulk | mu_full_zz | mu_slab_both | mu_wall | err vs law % | servo valid |",
          "|---|---|---|---|---|---|---|"]
    for p in lad["points"]:
        L.append(f"| {p['mu_contact']:.2f} | {p['I_bulk']:.4f} | "
                 f"{p['mu_full_zz']:.4f} | {p['mu_slab_both']:.4f} | "
                 f"{p['mu_wall']:.4f} | {100*p['rel_err_vs_law']:+.1f} | "
                 f"{p['servo_valid']} |")
    L += ["", "## (6) parity of the ensemble's archive seed against round 14", "",
          "| I | key | archived | re-run | rel diff | bitwise |", "|---|---|---|---|---|---|"]
    for p in a["parity_vs_archived"]:
        L.append(f"| {p['I']:.3g} | {p['key']} | {p['archived']:.6g} | "
                 f"{p['rerun']:.6g} | {p['rel_diff']:+.2e} | {p['bitwise_equal']} |")
    v = a["verdict"]
    L += ["", "## (7) pre-registered predictions vs outcome", "",
          "| id | prediction | outcome |", "|---|---|---|",
          f"| P1 | CV < 10 % at every judged point | max CV = "
          f"{100*v['max_cv_judged']:.1f} % -> **{'holds' if v['P1_spread_holds'] else 'falsified'}** |",
          f"| P2 | the dip is systematic | **{v['P2_dip_systematic']}** |",
          f"| P3 | ensemble mean outside the band at I = 0.03, 0.1 | "
          f"**{v['P3_bias_reproduced']}** |",
          f"| P4 | ladder monotone, and does not reach the band | monotone "
          f"**{v['P4_ladder_monotone']}**, reaches band "
          f"**{v['P4_ladder_reaches_band']}** |",
          "", f"Acceptance (all judged points, all seeds within 10 %): "
          f"**{'PASS' if v['passes_acceptance'] else 'FAIL'}** "
          f"({len(v['points_all_seeds_within_tol'])}/{len(v['judged_points'])} points)."]
    text = "\n".join(L) + "\n"
    (RUNS / "tables_r23.md").write_text(text)
    return text


PREREG_LAW = "0.22 + 1.00 I"


def make_figures(a: dict) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    made = []
    # ---- fig 1: mu(I) with per-seed scatter and CI band -------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    Igrid = np.logspace(-3, math.log10(0.35), 200)
    law = np.array([law_mu(float(i))[0] for i in Igrid])
    valid = np.array([law_mu(float(i))[1] for i in Igrid])
    ax.plot(Igrid[valid], law[valid] * 1.0, "-", color="k", lw=1.6,
            label="da Cruz 2005 (archived): $\\mu=0.22+1.00I$")
    ax.fill_between(Igrid[valid], law[valid] * 0.9, law[valid] * 1.1,
                    color="k", alpha=0.12, label="±10 % band")
    ax.plot(Igrid[~valid], law[~valid], ":", color="k", lw=1.2,
            label="extrapolated (not judged)")
    xs, means, los, his = [], [], [], []
    for k, v in a["points"].items():
        m = v["mu"]["mu_full_zz_final"]
        ax.scatter(v["I_bulk_values"], m["values"], s=16, facecolors="none",
                   edgecolors="C3", label="single seeds" if not xs else None)
        xs.append(v["I_bulk_mean"]["mean"])
        means.append(m["mean"])
        los.append(m["ci95"][0])
        his.append(m["ci95"][1])
    ax.errorbar(xs, means, yerr=[np.array(means) - np.array(los),
                                 np.array(his) - np.array(means)],
                fmt="s-", color="C3", ms=6, lw=1.4, capsize=3,
                label="ensemble mean ±95 % CI")
    ax.set_xscale("log")
    ax.set_xlabel("inertial number $I_{\\rm bulk}$")
    ax.set_ylabel("$\\mu = |\\sigma_{xz}|/|\\sigma_{zz}|$ (full-cell virial)")
    ax.set_title("R23: $\\mu(I)$, %d seeds per point vs the archived law" % len(SEEDS))
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = RUNS / "fig_r23_muI_ensemble.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p.name)

    # ---- fig 2: CV per ordinate ------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    width = 0.8 / len(MU_ORDINATES)
    keys = list(a["points"])
    idx = np.arange(len(keys))
    for j, name in enumerate(MU_ORDINATES):
        cv = [100 * a["points"][k]["mu"][name]["cv"] for k in keys]
        ax.bar(idx + j * width, cv, width=width, label=name)
    ax.axhline(100 * TOL, color="r", ls="--", lw=1.2, label="10 % band")
    ax.set_xticks(idx + 0.4 - width / 2)
    ax.set_xticklabels([f"I={k}" for k in keys])
    ax.set_ylabel("seed CV of $\\mu$ [%]")
    ax.set_title("R23: which ordinate is realisation-robust?")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = RUNS / "fig_r23_cv.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p.name)

    # ---- fig 3: paired dip ------------------------------------------------
    d = a["dip"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    lows = [a["points"]["0.03"]["mu"]["mu_full_zz_final"]["values"][d["seeds"].index(s)]
            for s in d["seeds"]]
    highs = [a["points"]["0.1"]["mu"]["mu_full_zz_final"]["values"][d["seeds"].index(s)]
             for s in d["seeds"]]
    for i, s in enumerate(d["seeds"]):
        ax.plot([0, 1], [lows[i], highs[i]], "o-", lw=1.0, ms=4,
                label=f"seed {s}" if i < 6 else None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$I=0.03$", "$I=0.1$"])
    ax.set_ylabel("$\\mu$ (full-cell virial, $\\sigma_{zz}$)")
    ax.set_title("R23: paired seed dip, mean change %+.1f %%" % (100 * d["rel_mean"]["mean"]))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = RUNS / "fig_r23_dip_paired.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p.name)

    # ---- fig 4: contact-friction ladder ----------------------------------
    lad = a["ladder"]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    mup = [p["mu_contact"] for p in lad["points"]]
    mz = [p["mu_full_zz"] for p in lad["points"]]
    ax.plot(mup, mz, "o-", color="C0", label="$\\mu$ full-cell virial")
    ax.axhline(lad["mu_law_archived"], color="k", lw=1.4,
               label="da Cruz law at $I=0.03$: %.3f" % lad["mu_law_archived"])
    ax.fill_between([min(mup) - 0.05, max(mup) + 0.05],
                    lad["mu_law_archived"] * 0.9, lad["mu_law_archived"] * 1.1,
                    color="k", alpha=0.12, label="±10 % band")
    ax.set_xlabel("contact friction $\\mu_p$")
    ax.set_ylabel("$\\mu$ at $I=0.03$")
    ax.set_title("R23: does lowering $\\mu_p$ close the 16-57 % gap?")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = RUNS / "fig_r23_mup_ladder.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(p.name)
    return made


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ns = ap.parse_args(argv)
    prereg_path = RUNS / "prereg.json"
    if not prereg_path.exists():
        prereg_path.write_text(json.dumps(PREREG, indent=2))
    if ns.assemble:
        a = analyse()
        (RUNS / "analysis_r23.json").write_text(json.dumps(a, indent=2, default=float))
        write_tables(a)
        figs = make_figures(a)
        print("wrote analysis_r23.json, tables_r23.md,", ", ".join(figs))
        print("verdict:", json.dumps(a["verdict"], default=float))
        return 0
    if ns.parity:
        row = run_one(1e-2, SEED)
        print(json.dumps({k: row[k] for k in ("I_bulk", "mu_full_zz_final",
                                              "mu_slab_both_final", "phi_slab",
                                              "phi_bed_steady")}, indent=2))
        return 0
    return run_shard(ns.shard or 0, ns.nshards)


if __name__ == "__main__":
    raise SystemExit(main())