"""CASE-DEM-PLANE-SHEAR-MUIDI: provenance / consistency adjudication.

Round 12 left the ledger row blocked on a bibliographic question ("which Jop
triple is the paper's?") and deferred the verdict until a provenance round could
be run.  This driver refuses to defer: it takes the *archived* round-9/10 point
files (``runs/full_cell``, ``runs/bulk_rheometry``) - zero new simulation - and
asks whether the answer depends on the bibliography at all.

Three questions, in increasing order of severity:

Q1 (consistency)  Is the archived ``(mu, I)`` pair internally consistent?  The
                  law requires the *same* pressure in both ordinate and
                  abscissa; the archived ``I_measured`` is built from the
                  **imposed platen rate** while ``mu_full`` is normalised by
                  ``p = -tr(sigma)/3`` and ``ratio``/``rate_ratio`` record a
                  bulk rate as low as 38 % of the imposed one.  Every admissible
                  pairing is enumerated instead of assumed.

Q2 (provenance)   Does any corner of the disputed reference envelope bring the
                  curve inside the 10 % band?

Q3 (falsification) Does *any* Jop law at all - over a box that strictly contains
                  the disputed envelope - fit the measured curve within 10 % at
                  every point?  If not, the provenance question is moot for the
                  acceptance and the real defect is in the device/abscissa.

Run:  python adjudicate_provenance.py            (writes runs/provenance_adjudication.*)

The driver is import-only-safe: everything of interest is a pure function of the
archived JSON, so the pytest module can re-derive every published number.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import reference_provenance as rp

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# device constants (identical to runs/abscissa_audit/prereg.json)
D_PARTICLE = 0.01        # m
RHO_GRAIN = 2500.0       # kg/m^3
P_TARGET = 6.25e4        # Pa, servo target
TOL = 0.10               # ledger acceptance band


# --------------------------------------------------------------------------- #
# Archive access
# --------------------------------------------------------------------------- #
def load_points() -> list[dict]:
    """Merged view of the archived round-9 (full_cell) and round-10
    (bulk_rheometry) points.  ``full_cell`` carries the richer estimator set,
    so it is the base and ``bulk_rheometry`` only fills ``*_bulk`` gaps."""
    pts: dict[float, dict] = {}
    for f in sorted((RUNS / "full_cell").glob("point_*.json")):
        d = json.loads(f.read_text())
        d["_source"] = str(f.relative_to(HERE))
        pts[float(d["I_target"])] = d
    for f in sorted((RUNS / "bulk_rheometry").glob("point_*.json")):
        d = json.loads(f.read_text())
        I = float(d["I_target"])
        base = pts.setdefault(I, {})
        for k, v in d.items():
            base.setdefault(k, v)
    return [pts[I] for I in sorted(pts)]


def inertial_number(rate: float, pressure: float) -> float:
    """``I = gamma_dot d sqrt(rho/p)``."""
    return rate * D_PARTICLE * math.sqrt(RHO_GRAIN / pressure)


# --------------------------------------------------------------------------- #
# Arm construction (Q1)
# --------------------------------------------------------------------------- #
#: ``(mu_key, rate_key, pressure_key, label)``.  ``mu_key`` picks the ordinate,
#: ``rate_key`` the strain rate in the abscissa and ``pressure_key`` the
#: pressure in the abscissa; consistency demands ``pressure_key`` match the
#: normaliser implicit in ``mu_key`` (``*_zz`` <-> ``sigma_zz``, ``*_full`` <->
#: ``p_full``).
ARMS = (
    ("mu_full_zz_final", "shear_rate_imposed", "sigma_zz_full_final",
     "zz-consistent, nominal (imposed) rate  [archived convention]"),
    ("mu_full_zz_final", "shear_rate_measured", "sigma_zz_full_final",
     "zz-consistent, measured bulk rate"),
    ("mu_full_final", "shear_rate_imposed", "p_full",
     "p-consistent, nominal (imposed) rate"),
    ("mu_full_final", "shear_rate_measured", "p_full",
     "p-consistent, measured bulk rate  [law-consistent]"),
    ("mu_wall_final", "shear_rate_measured", "p_wall",
     "wall traction, measured bulk rate"),
)


def build_arms(points: list[dict]) -> list[dict]:
    """Build every admissible ``(mu, I)`` series from the archived points."""
    out = []
    for mu_key, rate_key, p_key, label in ARMS:
        I_vals, mu_vals, used = [], [], []
        ok = True
        for p in points:
            if mu_key not in p or rate_key not in p or p_key not in p:
                ok = False
                break
            Pr = abs(float(p[p_key]))
            if Pr <= 0.0:
                ok = False
                break
            I_vals.append(inertial_number(float(p[rate_key]), Pr))
            mu_vals.append(float(p[mu_key]))
            used.append(float(p["I_target"]))
        if not ok:
            continue
        out.append({
            "label": label,
            "ordinate": mu_key,
            "rate": rate_key,
            "pressure": p_key,
            "I_nominal": used,
            "I": I_vals,
            "mu": mu_vals,
            "verdicts": rp.verdict_table(I_vals, mu_vals, TOL),
            "best_case": rp.best_case_over_envelope(I_vals, mu_vals, tol=TOL),
            "reverse_fit": rp.fit_jop_law(I_vals, mu_vals),
            "envelope_corners": envelope_corner_matrix(I_vals, mu_vals),
        })
    return out


def envelope_corner_matrix(I_vals, mu_vals, tol: float = TOL) -> dict:
    """Pass/fail of the measured series against **every corner** of the box
    spanned by the two disputed triples.

    This is the honest form of the round-12 question.  ``best_case_over_envelope``
    can formally reach the band, but only by leaving the box (see
    ``boundary_hit``); the corners are the set of readings the repository
    actually claims to be the paper's.  If no corner passes, the citation
    dispute cannot be the reason the ledger row is open.
    """
    corners = []
    for c in rp.ENVELOPE:
        rel = [float(mu / c.mu(I) - 1.0) for I, mu in zip(I_vals, mu_vals)]
        corners.append({
            "corner": c.name,
            "triple": list(c.triple),
            "rel_err": rel,
            "n_within_tol": int(sum(1 for r in rel if abs(r) <= tol)),
            "max_abs_rel_err": max((abs(r) for r in rel), default=float("nan")),
            "passes_all": all(abs(r) <= tol for r in rel),
        })
    return {
        "corners": corners,
        "n_corners": len(corners),
        "n_corners_passing_all": sum(1 for c in corners if c["passes_all"]),
        "best_corner_n_within_tol": max(c["n_within_tol"] for c in corners),
        "n_points": len(I_vals),
    }


def consistency_audit(points: list[dict]) -> list[dict]:
    """Record, per point, how far the archived pair is from law-consistency."""
    rows = []
    for p in points:
        Pr = abs(float(p["sigma_zz_full_final"]))
        pfull = float(p["p_full"])
        rec = {
            "I_target": float(p["I_target"]),
            "I_measured_archived": float(p["I_measured"]),
            "I_imposed_sigma_zz": inertial_number(float(p["shear_rate_imposed"]), P_TARGET),
            "I_measured_rate_sigma_zz": inertial_number(float(p["shear_rate_measured"]), Pr),
            "I_measured_rate_p_full": inertial_number(float(p["shear_rate_measured"]), pfull),
            "rate_ratio": float(p["rate_ratio"]),
            "profile_r2": float(p["profile_r2"]),
            "servo_valid": bool(p["servo_valid"]),
            "pressure_control_error": float(p.get("pressure_control_error_final", float("nan"))),
            "anisotropy_p_over_zz": pfull / Pr,
            "coordination": float(p.get("coordination", float("nan"))),
            # the spurious ordinate inflation from mixing normalisers
            "mixing_inflation": Pr / pfull,
            "kinematically_admissible": bool(
                float(p["rate_ratio"]) >= 0.90
                and float(p["profile_r2"]) >= 0.90
                and bool(p["servo_valid"])
            ),
        }
        rec["I_archived_vs_imposed_rel"] = abs(
            rec["I_measured_archived"] / rec["I_imposed_sigma_zz"] - 1.0)
        rows.append(rec)
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> dict:
    points = load_points()
    arms = build_arms(points)
    cons = consistency_audit(points)

    # envelope disagreement, on the ledger's own abscissa grid
    I_grid = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    envelope = []
    for I in I_grid:
        lo, hi = rp.envelope_bounds(I)
        envelope.append({
            "I": I,
            "mu_min": lo, "mu_max": hi,
            "rel_spread": rp.envelope_spread(I),
            "PRIMARY_REPO": rp.PRIMARY_REPO.mu(I),
            "TEST_GDR_MIDI": rp.TEST_GDR_MIDI.mu(I),
        })

    report = {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 13,
        "kind": "archival re-adjudication (zero new simulation)",
        "tol": TOL,
        "constants": {"d_particle_m": D_PARTICLE, "rho_grain": RHO_GRAIN,
                      "P_target_pa": P_TARGET},
        "sources": [p["_source"] for p in points],
        "provenance_registry": [
            {"name": c.name, "triple": list(c.triple),
             "status": c.status, "source": c.source}
            for c in rp.CANDIDATES
        ],
        "envelope": envelope,
        "consistency": cons,
        "arms": arms,
    }

    # ---- headline verdict ----
    law = next(a for a in arms if a["label"].startswith("p-consistent, measured"))
    report["headline"] = {
        "law_consistent_arm": law["label"],
        "I": law["I"], "mu": law["mu"],
        "n_within_tol_primary": law["verdicts"][0]["n_within_tol"],
        "max_abs_rel_err_primary": law["verdicts"][0]["max_abs_rel_err"],
        "min_max_abs_rel_err_over_all_laws":
            law["best_case"]["min_max_abs_rel_err"],
        "oracle_argmin": law["best_case"]["argmin"],
        "oracle_boundary_hit": law["best_case"]["boundary_hit"],
        "oracle_provenance_can_rescue": law["best_case"]["provenance_can_rescue"],
        "n_envelope_corners_passing_all": law["envelope_corners"]["n_corners_passing_all"],
        "n_envelope_corners": law["envelope_corners"]["n_corners"],
        "best_corner_n_within_tol": law["envelope_corners"]["best_corner_n_within_tol"],
        "n_kinematically_admissible": sum(1 for c in cons if c["kinematically_admissible"]),
        "n_points": len(cons),
        "max_anisotropy_spurious_ordinate_pct":
            100.0 * max(c["mixing_inflation"] - 1.0 for c in cons),
    }

    (RUNS / "provenance_adjudication.json").write_text(
        json.dumps(report, indent=2, sort_keys=True))
    (RUNS / "provenance_adjudication_tables.md").write_text(render_tables(report))
    return report


def _f(x, nd=4):
    try:
        if x != x:                      # NaN
            return "n/a"
        return f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def render_tables(rep: dict) -> str:
    L = []
    A = L.append
    A("# Round 13 - provenance / consistency adjudication")
    A("")
    A("Archival only: no new simulation. Sources:")
    for s in rep["sources"]:
        A(f"* `{s}`")
    A("")
    A("## Table A - the disputed reference envelope")
    A("")
    A("| I | PRIMARY_REPO (0.36/0.84/0.30) | TEST_GDR_MIDI (0.379/0.641/0.279)"
      " | envelope min | envelope max | rel. spread |")
    A("|---|---|---|---|---|---|")
    for e in rep["envelope"]:
        A(f"| {e['I']:g} | {_f(e['PRIMARY_REPO'])} | {_f(e['TEST_GDR_MIDI'])} "
          f"| {_f(e['mu_min'])} | {_f(e['mu_max'])} | {100*e['rel_spread']:.1f}% |")
    A("")
    A("## Table B - is the archived (mu, I) pair law-consistent?")
    A("")
    A("| I_target | rate_ratio | profile R^2 | servo | I archived | I(imposed, sigma_zz) "
      "| I(meas rate, p) | p/sigma_zz | spurious mu inflation | admissible |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for c in rep["consistency"]:
        A(f"| {c['I_target']:g} | {_f(c['rate_ratio'],3)} | {_f(c['profile_r2'],3)} "
          f"| {'PASS' if c['servo_valid'] else 'FAIL'} "
          f"| {_f(c['I_measured_archived'])} | {_f(c['I_imposed_sigma_zz'])} "
          f"| {_f(c['I_measured_rate_p_full'])} "
          f"| {_f(c['anisotropy_p_over_zz'],3)} | +{100*(c['mixing_inflation']-1):.1f}% "
          f"| {'yes' if c['kinematically_admissible'] else 'no'} |")
    A("")
    A("Errors are **measured / reference - 1** (the sign convention of rounds 8-12).")
    A("")
    A("## Table C - pass matrix (band 10 %), per arm x per candidate law")
    A("")
    A("| arm | candidate | " + " | ".join(f"I={i:g}" for i in rep["arms"][0]["I_nominal"])
      + " | n pass | max |err| |")
    A("|---|---|" + "---|" * len(rep["arms"][0]["I_nominal"]) + "---|---|")
    for a in rep["arms"]:
        for v in a["verdicts"]:
            cells = " | ".join(f"{100*r:+.1f}%" for r in v["rel_err"])
            A(f"| {a['label']} | {v['candidate']} | {cells} "
              f"| {v['n_within_tol']}/{v['n_points']} | {100*v['max_abs_rel_err']:.1f}% |")
    A("")
    A("## Table D - can ANY Jop law fit the measured curve? (falsification oracle)")
    A("")
    A("| arm | min worst-point err over the whole law family | argmin (mu_s, mu_2, I_0) "
      "| argmin on box edge? | #param sets inside 10 % |")
    A("|---|---|---|---|---|")
    for a in rep["arms"]:
        b = a["best_case"]
        g = b["argmin"]
        edge = ", ".join(b["boundary_hit"]) if b["boundary_hit"] else "-"
        A(f"| {a['label']} | **{100*b['min_max_abs_rel_err']:.1f}%** "
          f"| ({g['mu_s']:.3f}, {g['mu_2']:.3f}, {g['I_0']:.3f}) "
          f"| {edge} | {b['n_param_sets_within_tol']} |")
    A("")
    A("An argmin sitting on the search-box edge means the optimum is **not "
      "identified**: the measurement is fitted only by pushing the parameters "
      "outside anything either candidate paper claims.  Read Table D together "
      "with Table D2.")
    A("")
    A("## Table D2 - within the DISPUTED envelope only (8 corners)")
    A("")
    A("| arm | #corners passing all points | best #points any corner passes "
      "| smallest worst-point err over the 8 corners |")
    A("|---|---|---|---|")
    for a in rep["arms"]:
        ec = a["envelope_corners"]
        A(f"| {a['label']} | {ec['n_corners_passing_all']}/{ec['n_corners']} "
          f"| {ec['best_corner_n_within_tol']}/{ec['n_points']} "
          f"| {100*min(c['max_abs_rel_err'] for c in ec['corners']):.1f}% |")
    A("")
    A("## Table E - reverse calibration (what law DOES the device produce?)")
    A("")
    A("| arm | mu_s | mu_2 | I_0 | rms | R^2 |")
    A("|---|---|---|---|---|---|")
    for a in rep["arms"]:
        f = a["reverse_fit"]
        if not f.get("identified"):
            A(f"| {a['label']} | not identified ({f.get('reason')}) | | | | |")
            continue
        A(f"| {a['label']} | {f['mu_s']:.3f} | {f['mu_2']:.3f} | {f['I_0']:.3f} "
          f"| {f['rms']:.4f} | {f['r2']:+.2f} |")
    A("")
    h = rep["headline"]
    A("## Headline")
    A("")
    A(f"* law-consistent arm: `{h['law_consistent_arm']}`")
    A(f"* points inside the 10 % band vs the primary candidate: "
      f"{h['n_within_tol_primary']}/{h['n_points']} (max |err| "
      f"{100*h['max_abs_rel_err_primary']:.1f} %)")
    A(f"* corners of the disputed envelope passing every point: "
      f"{h['n_envelope_corners_passing_all']}/{h['n_envelope_corners']} "
      f"(best corner reaches {h['best_corner_n_within_tol']}/{h['n_points']} points)")
    A(f"* best possible worst-point error over the **entire** Jop family: "
      f"{100*h['min_max_abs_rel_err_over_all_laws']:.1f} % at "
      f"({h['oracle_argmin']['mu_s']:.3f}, {h['oracle_argmin']['mu_2']:.3f}, "
      f"{h['oracle_argmin']['I_0']:.3f}), argmin on box edge: "
      f"{', '.join(h['oracle_boundary_hit']) or 'no'}")
    A(f"* provenance can rescue the acceptance (identified fit): "
      f"{'YES' if h['oracle_provenance_can_rescue'] else '**NO**'}")
    A(f"* kinematically admissible points (rate_ratio>=0.9, R^2>=0.9, servo pass): "
      f"{h['n_kinematically_admissible']}/{h['n_points']}")
    A(f"* worst spurious ordinate inflation from mixing pressure normalisers: "
      f"+{h['max_anisotropy_spurious_ordinate_pct']:.1f} %")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    rep = main()
    h = rep["headline"]
    print(json.dumps(h, indent=2))
    print("\nwrote runs/provenance_adjudication.json / .tables.md")