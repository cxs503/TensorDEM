"""M4 MT Uikku audit: attribute the DEM-vs-experiment gap (owner selfstart unit).

The M4 verification surface (``tensordem.ice_broken_field.mt_uikku_validation``)
used to report a *passing* correlation criterion from the hard-coded constant
``0.85`` and to run the ten MT Uikku cases at 3/5/7 kn (= 1.54/2.57/3.60 m/s)
although the experimental table records an ice drift speed of 0.2/0.5/1.0 m/s.
This script produces the evidence for both findings and attributes the resulting
error between the two defects (speed provenance vs contact geometry).

Configurations (2 x 2 matrix over the two defects):

``L0``  shipped (legacy) contact geometry + legacy 3/5/7 kn speed  = what shipped
``L1``  shipped contact geometry        + tabulated drift speed    = provenance only fixed
``L2``  signed-distance geometry       + legacy 3/5/7 kn speed    = geometry only fixed
``L3``  signed-distance geometry       + tabulated drift speed    = both fixed

The signed-distance model is *not* shipped: it changes the force level by two
orders of magnitude, which is a model re-acceptance question (M3's Lindqvist
band), not an audit edit.  It is implemented here so the defect is reproducible
and quantified; see ``docs/M4_MT_UIKKU_PROVENANCE_AUDIT.md`` for the finding and
the recommended follow-up.

Artifacts: ``artifacts/m4_mt_uikku_audit/{summary.json,cases.csv,png}``.

Usage::

    python scripts/m4_mt_uikku_audit.py [--dt 0.01] [--t-total 60] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.ice_broken_field import MT_UIKKU_CSV, broken_ice_field_simulation  # noqa: E402
from tensordem.ice_broken_field.channel_clearing import (  # noqa: E402
    BrokenIceFieldConfig,
    ShipTrajectory,
)
from tensordem.ice_broken_field.mt_uikku_validation import (  # noqa: E402
    MTUikkuCase,
    load_mt_uikku_cases,
    score_cases,
)

#: legacy speed provenance: the pre-fix module's ``ship_speed_kn`` per case.
LEGACY_KNOTS_BY_ID = {"103": 3.0, "104": 5.0, "205": 3.0, "206": 5.0, "301": 3.0,
                      "302": 5.0, "303": 7.0, "401": 3.0, "402": 5.0, "403": 7.0}
KNOT = 0.5144


# --------------------------------------------------------------------------- #
# shipped (legacy) contact model: contact only when the floe *centre* is inside
# --------------------------------------------------------------------------- #
def legacy_signed_distance(ship: ShipTrajectory, pts: np.ndarray) -> np.ndarray:
    """Vectorised pre-fix ``ship_hull_signed_distance`` (identical semantics)."""
    x, y = pts[:, 0], pts[:, 1]
    half_beam = 0.5 * ship.beam
    body_length = 3.0 * ship.beam
    bow_length = 0.5 * ship.beam / math.tan(ship.entry_angle_rad)
    dy = np.abs(y - ship.y)
    out = np.full(len(pts), 1.0)
    in_body = (x >= ship.x - body_length) & (x <= ship.x) & (dy <= half_beam)
    out = np.where(in_body, -np.minimum(half_beam - dy, ship.x - x), out)
    in_bow = (x >= ship.x) & (x <= ship.x + bow_length)
    frac = np.clip((x - ship.x) / bow_length, 0.0, 1.0)
    half_width = half_beam * (1.0 - frac)
    in_bow = in_bow & (dy <= half_width)
    out = np.where(in_bow, -np.minimum(half_width - dy, (x - ship.x) * 0.1), out)
    return out


# --------------------------------------------------------------------------- #
# signed-distance contact model (candidate fix, NOT shipped - see module docstring)
# --------------------------------------------------------------------------- #
def hull_polygon(ship: ShipTrajectory) -> np.ndarray:
    half_beam = 0.5 * ship.beam
    body_length = 3.0 * ship.beam
    bow_length = 0.5 * ship.beam / math.tan(ship.entry_angle_rad)
    return np.array([[ship.x + bow_length, ship.y],
                     [ship.x, ship.y + half_beam],
                     [ship.x - body_length, ship.y + half_beam],
                     [ship.x - body_length, ship.y - half_beam],
                     [ship.x, ship.y - half_beam]], dtype=float)


def hull_distance_and_normal(poly: np.ndarray, pts: np.ndarray,
                             ref_y: float) -> tuple[np.ndarray, np.ndarray]:
    """True signed distance to the hull polygon + outward boundary normal."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom <= 0.0, 1.0, denom)
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("nej,ej->ne", ap, ab) / denom[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d_all = np.linalg.norm(pts[:, None, :] - closest, axis=2)
    edge = np.argmin(d_all, axis=1)
    dist = d_all[np.arange(len(pts)), edge]
    cp = closest[np.arange(len(pts)), edge, :]

    x, y = pts[:, 0], pts[:, 1]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = b[:, 0][None, :], b[:, 1][None, :]
    straddle = (y0 > y[:, None]) != (y1 > y[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x0 + (y[:, None] - y0) * (x1 - x0) / (y1 - y0)
    inside = (np.count_nonzero(straddle & (x[:, None] < x_cross), axis=1) % 2) == 1

    diff = pts - cp
    norm = np.linalg.norm(diff, axis=1)
    fallback = np.column_stack([np.zeros(len(pts)),
                                np.where(y >= ref_y, 1.0, -1.0)])
    normal = np.where(norm[:, None] > 0.0,
                      diff / np.where(norm == 0.0, 1.0, norm)[:, None], fallback)
    normal = np.where(inside[:, None], -normal, normal)
    return np.where(inside, -dist, dist), normal


# --------------------------------------------------------------------------- #
# field simulations
# --------------------------------------------------------------------------- #
def _initial_field(config: BrokenIceFieldConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(config.seed)
    n_floes = int(config.concentration * config.domain_length * config.domain_width
                  / (math.pi * config.mean_floe_radius ** 2))
    radius = np.array([
        max(0.5 * config.mean_floe_radius,
            min(2.0 * config.mean_floe_radius,
                rng.normal(config.mean_floe_radius, config.radius_std)))
        for _ in range(n_floes)])
    pos = np.array([[rng.uniform(0.0, config.domain_length),
                     rng.uniform(0.0, config.domain_width)] for _ in range(n_floes)])
    return pos, radius


def legacy_field(config: BrokenIceFieldConfig, dt: float, t_total: float) -> dict[str, Any]:
    """Shipped M3 field (``broken_ice_field_simulation``), instrumented."""
    res = broken_ice_field_simulation(config, dt=dt, t_total=t_total)
    return {"mean_kN": res.R_ice_mean / 1e3, "peak_kN": res.R_ice_peak / 1e3,
            "std_kN": res.R_ice_std / 1e3, "travel_m": config.ship_speed * t_total}


def signed_distance_field(config: BrokenIceFieldConfig, dt: float,
                          t_total: float) -> dict[str, Any]:
    """Candidate fix: disk-vs-polygon interference, exact chord, axial projection.

    * interference ``delta = clip(R - d, 0, R)`` with ``d`` the *true* signed
      distance to the hull polygon (so a floe centre 2 m outside a 26 m wide hull
      is in contact, as it physically is, and a floe centre inside the hull
      cannot generate a 13 m indentation);
    * contact width = exact disk-boundary chord ``2 sqrt(2 R delta - delta^2)``,
      which vanishes at ``delta = 0`` and saturates at ``2 R`` without a hand cap;
    * the force is resolved on the hull frame: only the forward-facing normal
      component plus the tangential friction component resist the advance;
    * floes are pushed out along the outward normal (rigid constraint) instead of
      the legacy fixed lateral drift of ``0.5 v dt``, which let floes linger
      inside the swept hull volume and generate a permanent contact force.
    """
    pos, radius = _initial_field(config)
    thickness = np.full(len(pos), config.ice_thickness)
    ship = ShipTrajectory(x=-config.ship_beam, y=config.domain_width / 2.0,
                          speed=config.ship_speed, beam=config.ship_beam,
                          entry_angle_deg=config.ship_entry_angle,
                          bow_radius=config.ship_bow_radius)
    n_steps = int(t_total / dt)
    R_ice = np.zeros(n_steps)
    for i in range(n_steps):
        ship.x += config.ship_speed * dt
        if len(pos):
            poly = hull_polygon(ship)
            d, normal = hull_distance_and_normal(poly, pos, ship.y)
            delta = np.clip(radius - d, 0.0, radius)
            contact = delta > 0.0
            if np.any(contact):
                r_c, dl = radius[contact], delta[contact]
                w = 2.0 * np.sqrt(np.maximum(0.0, 2.0 * r_c * dl - dl * dl))
                F_n = config.compressive_strength * thickness[contact] * w
                F_t = config.friction_coefficient * F_n
                nrm = normal[contact]
                tan = np.column_stack([-nrm[:, 1], nrm[:, 0]])
                R_ice[i] = float(np.sum(F_n * np.maximum(0.0, -nrm[:, 0])
                                        + F_t * np.abs(tan[:, 0])))
            pos = pos + normal * np.where(contact, delta, 0.0)[:, None]
    steady = R_ice[n_steps // 2:]
    return {"mean_kN": float(np.mean(steady)) / 1e3,
            "peak_kN": float(np.max(steady)) / 1e3,
            "std_kN": float(np.std(steady)) / 1e3,
            "travel_m": config.ship_speed * t_total}


MODELS = {"legacy": legacy_field, "signed_distance": signed_distance_field}


def make_config(case: MTUikkuCase, speed_ms: float) -> BrokenIceFieldConfig:
    return BrokenIceFieldConfig(
        domain_length=500.0, domain_width=100.0, concentration=case.concentration,
        mean_floe_radius=3.0, radius_std=1.0, ice_thickness=case.ice_thickness_m,
        compressive_strength=case.compressive_strength_kPa * 1e3,
        friction_coefficient=0.1, ship_speed=speed_ms, ship_beam=26.0,
        ship_entry_angle=18.0, ship_bow_radius=15.0, seed=42)


def csv_baselines() -> dict[str, dict[str, float]]:
    """Empirical formula predictions shipped next to the measured table."""
    path = ROOT / MT_UIKKU_CSV
    out: dict[str, dict[str, float]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["test_id"].strip()] = {
                "lindqvist_prediction_kN": float(row["lindqvist_prediction_kN"]),
                "riska_prediction_kN": float(row["riska_prediction_kN"]),
                "jeong_prediction_kN": float(row["jeong_prediction_kN"]),
                "keinonen_prediction_kN": float(row["keinonen_prediction_kN"]),
            }
    return out


CONFIGS = {
    "L0_legacy_geom_legacy_speed": ("legacy", "legacy"),
    "L1_legacy_geom_drift_speed": ("legacy", "drift"),
    "L2_signed_geom_legacy_speed": ("signed_distance", "legacy"),
    "L3_signed_geom_drift_speed": ("signed_distance", "drift"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--t-total", type=float, default=60.0)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "m4_mt_uikku_audit"))
    args = ap.parse_args()

    cases = load_mt_uikku_cases()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = csv_baselines()

    rows: list[dict[str, Any]] = []
    for case in cases:
        speed = {"legacy": LEGACY_KNOTS_BY_ID[case.case_id] * KNOT,
                 "drift": case.ice_drift_speed_ms}
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "ice_sheet": case.ice_sheet,
            "thickness_m": case.ice_thickness_m,
            "concentration_assumed": case.concentration,
            "speed_legacy_ms": speed["legacy"],
            "speed_drift_ms": speed["drift"],
            "measured_mean_kN": case.measured_mean_kN,
            "measured_max_kN": case.measured_max_kN,
        }
        for label, (model, speed_key) in CONFIGS.items():
            cfg = make_config(case, speed[speed_key])
            out = MODELS[model](cfg, args.dt, args.t_total)
            row[f"{label}_kN"] = out["mean_kN"]
            row[f"{label}_peak_kN"] = out["peak_kN"]
            row[f"{label}_travel_m"] = out["travel_m"]
        row.update(baselines.get(case.case_id, {}))
        rows.append(row)
        print(f"{case.case_id}: meas {case.measured_mean_kN:7.0f} kN | " + " | ".join(
            f"{lab.split('_')[0]} {row[f'{lab}_kN']:8.1f}" for lab in CONFIGS), flush=True)

    def _score(key: str) -> dict[str, Any]:
        return score_cases([{
            "case_id": r["case_id"],
            "F_mean_kN": r[key],
            "measured_mean_kN": r["measured_mean_kN"],
            "F_peak_kN": r.get(key.replace("_kN", "_peak_kN"), 0.0),
            "measured_max_kN": r["measured_max_kN"],
        } for r in rows])

    scores = {label: _score(f"{label}_kN") for label in CONFIGS}
    for name, field in (("Lindqvist", "lindqvist_prediction_kN"),
                        ("Riska", "riska_prediction_kN"),
                        ("Jeong", "jeong_prediction_kN"),
                        ("Keinonen", "keinonen_prediction_kN")):
        if all(field in r for r in rows):
            scores[f"formula_{name}"] = score_cases([{
                "case_id": r["case_id"],
                "F_mean_kN": r[field],
                "measured_mean_kN": r["measured_mean_kN"],
                "F_peak_kN": 0.0,
                "measured_max_kN": 0.0,
            } for r in rows])

    # geometry probe: the two contact models disagree about a floe that is
    # 0.5 m inside the side shell (centre 2.5 m outside a 3 m radius floe)
    ship_probe = ShipTrajectory(x=0.0, y=50.0, speed=0.5, beam=26.0,
                                entry_angle_deg=18.0, bow_radius=15.0)
    probe_pt = np.array([[-40.0, 50.0 + 13.0 + 2.5]])
    d_legacy = float(legacy_signed_distance(ship_probe, probe_pt)[0])
    d_signed, _ = hull_distance_and_normal(hull_polygon(ship_probe), probe_pt, 50.0)
    geometry_probe = {
        "floe_centre_offset_from_side_shell_m": 2.5,
        "floe_radius_m": 3.0,
        "legacy_signed_distance_m": d_legacy,
        "legacy_reports_contact": bool(d_legacy < 0.0),
        "signed_distance_m": float(d_signed[0]),
        "signed_distance_reports_contact": bool(float(d_signed[0]) < 3.0),
    }

    summary = {
        "anchor_csv": MT_UIKKU_CSV,
        "dt": args.dt,
        "t_total_s": args.t_total,
        "n_cases": len(rows),
        "speeds": {"legacy_knots": LEGACY_KNOTS_BY_ID, "knot_to_ms": KNOT,
                   "drift_speed_field": "ice_drift_speed_m_s"},
        "concentration_note": ("no concentration column in the anchor table; "
                               "0.9/0.8/0.7/0.6 per ice sheet class are ASSUMED"),
        "configurations": {label: {"contact_model": model, "speed": speed}
                           for label, (model, speed) in CONFIGS.items()},
        "geometry_probe": geometry_probe,
        "scores": scores,
        "cases": rows,
        "verdict": {
            "M4_status": "WITHHELD",
            "reasons": [
                "the ten shipped MT Uikku cases are level-ice tests (C = 1.0) while "
                "the M3 field is a broken-ice field (C = 0.6-0.9): regime confound",
                "no measured force time series exists, so the rho >= 0.7 "
                "time-history criterion is WITHHELD (a constant must not stand in)",
                "at the tabulated drift speeds (0.2-1.0 m/s) the ship sweeps well "
                "under one hull length in 60 s, so the last-half mean is a start-up "
                "transient, not a steady-state resistance",
            ],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, scores, out_dir / "m4_mt_uikku_audit.png")

    print(json.dumps({k: {"mean_rel_error": round(v["F_mean_mean_error"], 4),
                          "pearson_across_cases": round(
                              v["correlation_across_cases_pearson"], 4),
                          "F_mean_pass": v["F_mean_criterion_pass"]}
                      for k, v in scores.items()}, indent=2))
    print(json.dumps(geometry_probe, indent=2))
    return 0


def _plot(rows: list[dict[str, Any]], scores: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is best effort
        print(f"plot skipped: {exc}")
        return
    ids = [r["case_id"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    idx = np.arange(len(ids))
    width = 0.2
    series = [("MT Uikku measured", [r["measured_mean_kN"] for r in rows], "k")]
    for k, (label, _) in enumerate(CONFIGS.items()):
        series.append((label, [r[f"{label}_kN"] for r in rows], f"C{k}"))
    for k, (label, vals, colour) in enumerate(series):
        ax.bar(idx + (k - 2.0) * width, vals, width, label=label, color=colour,
               alpha=1.0 if colour == "k" else 0.85)
    ax.set_xticks(idx)
    ax.set_xticklabels(ids)
    ax.set_yscale("log")
    ax.set_ylabel("mean ice resistance [kN] (log scale)")
    ax.set_xlabel("MT Uikku test id")
    ax.legend(fontsize=6.5)
    ax.set_title("M4: mean resistance, defect attribution matrix")

    ax = axes[1]
    names = list(scores.keys())
    errs = [scores[n]["F_mean_mean_error"] * 100.0 for n in names]
    ax.barh(range(len(names)), errs, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.axvline(20.0, color="crimson", ls="--", lw=0.9, label="20% acceptance band")
    ax.set_xscale("log")
    ax.set_xlabel("mean relative error of mean resistance [%] (log)")
    ax.legend(fontsize=7)
    ax.set_title("M4: error per configuration / empirical formula")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())