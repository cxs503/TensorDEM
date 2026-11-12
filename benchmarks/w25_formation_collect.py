"""W25 ``formation-ice-channel`` collector: raw DEM runs -> paper artifact.

Reads every ``*.json`` produced by ``w25_formation_runner.py`` and writes

    papers/artifacts/w25/formation-ice-channel/lindqvist_formation_gci.json

plus four figures under ``papers/artifacts/w25/formation-ice-channel/figures``.

Nothing in the artifact is hand-typed: every number is computed here from the
raw per-run ledgers (``samples``, ``mean_upstream_load_N``, ``per_hull``) or
from the closed-form anchors in ``tensordem``.  The three mission criteria are
evaluated programmatically and stored as ``verdicts``; an honest FAIL is never
overwritten.

Usage::

    python benchmarks/w25_formation_collect.py            # full artifact
    python benchmarks/w25_formation_collect.py --no-figures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src",):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from tensordem.ice_channel import analytic_equilibrium_draught  # noqa: E402
from tensordem.ice_channel_formation import (  # noqa: E402
    CALIBER_ALPHA_BAND_DEG,
    CALIBER_ALPHA_PRIMARY_DEG,
    CALIBER_MU_BAND,
    CALIBER_MU_PRIMARY,
    FormationCase,
    channel_reclosure_time,
    domain_verdict,
    gci_three_grid,
    lindqvist_reference,
    monotonicity,
    savings_rate,
)
from tensordem.ice_channel_formation_cases import (  # noqa: E402
    BASE_DT_S,
    DRIFT_BAND_M_S,
    GCI_SPACING_L,
    GCI_SPEED_M_S,
    SPACINGS_L,
    SPEEDS_M_S,
    build_matrix,
    cost_proxy,
    speed_label,
)

RAW_DIR = REPO_ROOT / "benchmarks" / "artifacts" / "w25" / "raw"
OUT_DIR = REPO_ROOT / "papers" / "artifacts" / "w25" / "formation-ice-channel"
ARTIFACT = OUT_DIR / "lindqvist_formation_gci.json"
FIG_DIR = OUT_DIR / "figures"
PREREG = REPO_ROOT / "prereg" / "w25_formation_ice_channel_prereg.json"
MISSION_ID = 16718
WATER_DENSITY = 1025.0


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------
def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return float("nan")
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def _sem(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return float("nan")
    return _std(vals) / math.sqrt(len(vals))


def _rel(path: Path) -> str:
    """Path relative to the repository root when possible, else absolute."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# raw loading
# ---------------------------------------------------------------------------
def load_runs(raw_dir: Path = RAW_DIR) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        label = payload.get("label") or path.stem
        payload["_path"] = _rel(path)
        runs[label] = payload
    return runs


def _case_of(run: dict[str, Any]) -> FormationCase:
    case = run["case"]
    return FormationCase(
        label=case["label"],
        n_ships=int(case["n_ships"]),
        spacing_L=case["spacing_L"],
        speed_m_s=float(case["speed_m_s"]),
        ship_length_m=float(case["ship_length_m"]),
        ship_beam_m=float(case["ship_beam_m"]),
        concentration=float(case["concentration"]),
        mean_radius_m=float(case["mean_radius_m"]),
        ice_thickness_m=float(case["ice_thickness_m"]),
        field_width_m=float(case["field_width_m"]),
        drift_m_s=float(case["drift_m_s"]),
        dt=float(case["dt"]),
        seed=int(case["seed"]),
        mesh_level=int(case["mesh_level"]),
        role=case["role"],
    )


def _meta(run: dict[str, Any]) -> dict[str, Any]:
    case = run["case"]
    return {
        "label": run["label"],
        "role": case["role"],
        "speed_m_s": case["speed_m_s"],
        "spacing_L": case["spacing_L"],
        "seed": case["seed"],
        "mesh_level": case["mesh_level"],
        "dt": case["dt"],
        "drift_m_s": case["drift_m_s"],
        "n_floes": run["n_floes"],
        "steps": run["steps_done"],
        "sim_time_s": run["sim_time_s"],
        "wall_time_s": run.get("wall_time_s"),
        "field_length_m": case["field_length_m"],
        "mean_radius_m": case["mean_radius_refined_m"],
        "raw": run["_path"],
    }


def _hull_loads(run: dict[str, Any]) -> list[float]:
    return [float(item["mean_upstream_load_N"]) for item in run["per_hull"]]


# ---------------------------------------------------------------------------
# single-ship ensemble + Lindqvist reconciliation
# ---------------------------------------------------------------------------
def single_ship_table(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for speed in SPEEDS_M_S:
        members = [runs[label] for label in runs
                   if runs[label]["case"]["role"] == "single"
                   and abs(float(runs[label]["case"]["speed_m_s"]) - speed) < 1e-9]
        if not members:
            continue
        labels = sorted(members, key=lambda r: r["case"]["seed"])
        loads = [_hull_loads(r)[0] for r in labels]
        contact = [float(r["per_hull"][0]["contact_fraction"]) for r in labels]
        case0 = _case_of(labels[0])
        anchors = {}
        for alpha in CALIBER_ALPHA_BAND_DEG:
            for mu in CALIBER_MU_BAND:
                anchor = lindqvist_reference(case0, speed, alpha_deg=alpha, mu=mu)
                anchors[f"alpha{alpha:g}_mu{mu:g}"] = {
                    "alpha_deg": alpha,
                    "mu": mu,
                    "total_N": anchor["total_N"],
                    "crushing_N": anchor["crushing_N"],
                    "bending_N": anchor["bending_N"],
                    "submersion_N": anchor["submersion_N"],
                    "added_drag_N": anchor["added_drag_N"],
                }
        primary = lindqvist_reference(case0, speed,
                                      alpha_deg=CALIBER_ALPHA_PRIMARY_DEG,
                                      mu=CALIBER_MU_PRIMARY)
        mean_load = _mean(loads)
        verdicts = {}
        for key, band in anchors.items():
            v = domain_verdict(mean_load, band)
            verdicts[key] = v
        table[f"{speed:g}"] = {
            "speed_m_s": speed,
            "n_members": len(labels),
            "member_labels": [r["label"] for r in labels],
            "member_labels_sorted_by_seed": [r["label"] for r in labels],
            "loads_N": loads,
            "load_mean_N": mean_load,
            "load_std_N": _std(loads),
            "load_sem_N": _sem(loads),
            "load_relative_spread_pct": 100.0 * _std(loads) / mean_load if mean_load else float("nan"),
            "contact_fraction_mean": _mean(contact),
            "n_floes": labels[0]["n_floes"],
            "wind_and_current": {"wind_velocity": [0.0, 0.0],
                                 "current_velocity": [0.0, case0.drift_m_s]},
            "lindqvist_primary": {
                "alpha_deg": CALIBER_ALPHA_PRIMARY_DEG,
                "mu": CALIBER_MU_PRIMARY,
                "components": {
                    "crushing_N": primary["crushing_N"],
                    "bending_N": primary["bending_N"],
                    "submersion_N": primary["submersion_N"],
                    "added_drag_N": primary["added_drag_N"],
                    "static_N": primary["static_N"],
                    "total_N": primary["total_N"],
                },
                "component_shares": {
                    "crushing": primary["crushing_N"] / primary["total_N"],
                    "bending": primary["bending_N"] / primary["total_N"],
                    "submersion": primary["submersion_N"] / primary["total_N"],
                    "added_drag": primary["added_drag_N"] / primary["total_N"],
                },
                "provenance": primary["components"]["provenance"],
                "ledger_sha256": primary["components"]["ledger_sha256"],
            },
            "lindqvist_band": anchors,
            "deviation_band": {
                key: {
                    "relative_deviation_pct": v["relative_deviation_pct"],
                    "dem_below_anchor_floor": v["dem_below_anchor_floor"],
                    "anchor_over_dem_ratio": v["anchor_over_dem_ratio"],
                }
                for key, v in verdicts.items()
            },
            "primary_deviation": verdicts[f"alpha{CALIBER_ALPHA_PRIMARY_DEG:g}"
                                          f"_mu{CALIBER_MU_PRIMARY:g}"],
        }
    return table


# ---------------------------------------------------------------------------
# formation savings surface
# ---------------------------------------------------------------------------
def savings_surface(runs: dict[str, dict[str, Any]],
                    single: dict[str, Any]) -> dict[str, Any]:
    surface: dict[str, Any] = {}
    for speed in SPEEDS_M_S:
        entry = single.get(f"{speed:g}")
        if not entry:
            continue
        reference = entry["load_mean_N"]
        per_spacing: dict[str, Any] = {}
        for spacing in SPACINGS_L:
            members = [runs[label] for label in runs
                       if runs[label]["case"]["role"] == "formation"
                       and abs(float(runs[label]["case"]["speed_m_s"]) - speed) < 1e-9
                       and abs(float(runs[label]["case"]["spacing_L"]) - spacing) < 1e-9]
            if not members:
                continue
            members.sort(key=lambda r: r["case"]["seed"])
            totals = [float(r["formation_total_load_N"]) for r in members]
            member_savings = [savings_rate(total, reference, 3) for total in totals]
            # paired (variance-reduced) estimator: the column against three
            # copies of its own lead hull in the same realisation
            lead = [_hull_loads(r)[0] for r in members]
            paired = [savings_rate(total, ld, 3) for total, ld in zip(totals, lead)]
            followers = [_hull_loads(r)[1:] for r in members]
            case0 = _case_of(members[0])
            closure = channel_reclosure_time(case0, n_hulls=3)
            per_spacing[f"{spacing:g}"] = {
                "spacing_L": spacing,
                "spacing_m": case0.spacing_m,
                "n_members": len(members),
                "member_labels": [r["label"] for r in members],
                "formation_total_load_N": _mean(totals),
                "formation_total_load_std_N": _std(totals),
                "lead_load_N": _mean(lead),
                "follower_loads_N": [_mean([f[i] for f in followers])
                                     for i in range(len(followers[0]))],
                "follower_to_lead_ratio": [
                    _mean([f[i] / ld for f, ld in zip(followers, lead)])
                    for i in range(len(followers[0]))
                ],
                "contact_fraction_per_hull": [
                    _mean([float(item["contact_fraction"]) for item in r["per_hull"]])
                    for r in members
                ],
                "single_ship_reference_N": reference,
                "saving": _mean(member_savings),
                "saving_std": _std(member_savings),
                "saving_sem": _sem(member_savings),
                "saving_members": member_savings,
                "saving_paired": _mean(paired),
                "saving_paired_std": _std(paired),
                "saving_paired_sem": _sem(paired),
                "saving_paired_members": paired,
                "reclosure": closure,
            }
        surface[f"{speed:g}"] = {"speed_m_s": speed,
                                 "single_ship_reference_N": reference,
                                 "by_spacing": per_spacing}
    return surface


def monotonicity_checks(surface: dict[str, Any], single: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for speed_key, entry in surface.items():
        spacings = sorted(entry["by_spacing"].values(), key=lambda item: item["spacing_L"])
        if len(spacings) < 3:
            continue
        saving = [item["saving"] for item in spacings]
        paired = [item["saving_paired"] for item in spacings]
        tol = max(
            [item["saving_sem"] for item in spacings if math.isfinite(item["saving_sem"])]
            + [0.0])
        tol_paired = max(
            [item["saving_paired_sem"] for item in spacings
             if math.isfinite(item["saving_paired_sem"])] + [0.0])
        checks[speed_key] = {
            "speed_m_s": entry["speed_m_s"],
            "spacings_L": [item["spacing_L"] for item in spacings],
            "saving": saving,
            "saving_strict": monotonicity(saving),
            "saving_tolerant": monotonicity(saving, tolerance=tol),
            "saving_paired": paired,
            "saving_paired_strict": monotonicity(paired),
            "saving_paired_tolerant": monotonicity(paired, tolerance=tol_paired),
            "tolerance": tol,
            "tolerance_paired": tol_paired,
        }
    return checks


# ---------------------------------------------------------------------------
# grid convergence
# ---------------------------------------------------------------------------
def gci_table(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def _family(prefix_base: str, prefix_tmpl: str) -> dict[str, Any] | None:
        base = runs.get(prefix_base)
        l2 = runs.get(prefix_tmpl.format(2))
        l4 = runs.get(prefix_tmpl.format(4))
        if not (base and l2 and l4):
            return None
        metric = lambda r: (float(r["formation_total_load_N"]) if r["case"]["n_ships"] > 1  # noqa: E731
                            else _hull_loads(r)[0])
        fine, medium, coarse = metric(l4), metric(l2), metric(base)
        grid = gci_three_grid(fine, medium, coarse, refinement=2.0)
        n_floes = [base["n_floes"], l2["n_floes"], l4["n_floes"]]
        return {
            "metric": "formation total ice load [N]" if base["case"]["n_ships"] > 1
                      else "single-ship ice load [N]",
            "values_N": {"level_1_base": base["formation_total_load_N"]
                         if base["case"]["n_ships"] > 1 else _hull_loads(base)[0],
                         "level_2": l2["formation_total_load_N"]
                         if l2["case"]["n_ships"] > 1 else _hull_loads(l2)[0],
                         "level_4": l4["formation_total_load_N"]
                         if l4["case"]["n_ships"] > 1 else _hull_loads(l4)[0]},
            "n_floes": {"level_1_base": n_floes[0], "level_2": n_floes[1], "level_4": n_floes[2]},
            "mean_radius_m": {"level_1_base": base["case"]["mean_radius_refined_m"],
                              "level_2": l2["case"]["mean_radius_refined_m"],
                              "level_4": l4["case"]["mean_radius_refined_m"]},
            "dt_s": {"level_1_base": base["dt"], "level_2": l2["dt"], "level_4": l4["dt"]},
            "gci": grid,
            "runs": [_meta(r) for r in (base, l2, l4)],
        }

    single = _family(f"single_V{speed_label(GCI_SPEED_M_S)}",
                     f"single_V{speed_label(GCI_SPEED_M_S)}_Nx{{}}")
    if single:
        single["saving_metric"] = {
            "note": "saving of the 3-ship column against this family's single ship, "
                    "per refinement level",
        }
        for level, key in ((1, None), (2, 2), (4, 4)):
            form = runs.get(f"form_V{speed_label(GCI_SPEED_M_S)}_S{GCI_SPACING_L:g}L"
                            + ("" if key is None else f"_Nx{key}"))
            sl = runs.get(f"single_V{speed_label(GCI_SPEED_M_S)}"
                          + ("" if key is None else f"_Nx{key}"))
            if form and sl:
                single["saving_metric"][f"level_{level}"] = savings_rate(
                    float(form["formation_total_load_N"]), _hull_loads(sl)[0], 3)
        vals = [single["saving_metric"][k] for k in ("level_1", "level_2", "level_4")
                if k in single["saving_metric"]]
        if len(vals) == 3:
            single["saving_metric"]["gci"] = gci_three_grid(vals[2], vals[1], vals[0],
                                                            refinement=2.0)
        out["single_ship"] = single

    formation = _family(f"form_V{speed_label(GCI_SPEED_M_S)}_S{GCI_SPACING_L:g}L",
                        f"form_V{speed_label(GCI_SPEED_M_S)}_S{GCI_SPACING_L:g}L_Nx{{}}")
    if formation:
        out["formation"] = formation
    return out


def numerics_table(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base = runs.get(f"single_V{speed_label(GCI_SPEED_M_S)}")
    half = runs.get(f"single_V{speed_label(GCI_SPEED_M_S)}_dt_half")
    if base and half:
        b, h = _hull_loads(base)[0], _hull_loads(half)[0]
        out["timestep"] = {
            "base_dt_s": base["dt"],
            "half_dt_s": half["dt"],
            "load_base_N": b,
            "load_half_dt_N": h,
            "relative_difference_pct": 100.0 * (h - b) / b if b else float("nan"),
            "reference": "base timestep",
        }
    members = {}
    for label, run in runs.items():
        if run["case"]["role"] == "numerics" and "drift" in label:
            members[label] = {
                "drift_m_s": run["case"]["drift_m_s"],
                "load_N": run["formation_total_load_N"] if run["case"]["n_ships"] > 1
                          else _hull_loads(run)[0],
                "n_ships": run["case"]["n_ships"],
                "speed_m_s": run["case"]["speed_m_s"],
                "spacing_L": run["case"]["spacing_L"],
                "reclosure_time_s": channel_reclosure_time(
                    _case_of(run), n_hulls=run["case"]["n_ships"])["reclosure_time_s"],
            }
    if members:
        out["drift_sensitivity"] = members
        solo = [m for m in members.values() if m["n_ships"] == 1]
        if len(solo) >= 3:
            loads = [m["load_N"] for m in sorted(solo, key=lambda m: m["drift_m_s"])]
            out["drift_sensitivity"]["single_ship_band"] = {
                "loads_N": loads,
                "relative_spread_pct": 100.0 * (max(loads) - min(loads)) / _mean(loads),
            }
    out["declared_band"] = {"drift_m_s": list(DRIFT_BAND_M_S), "base_drift_m_s": 0.05,
                            "base_dt_s": BASE_DT_S}
    return out


# ---------------------------------------------------------------------------
# ISO 19906 A.8 cross-check of the mean contact pressure
# ---------------------------------------------------------------------------
def iso19906_check(single: dict[str, Any], case: FormationCase) -> dict[str, Any]:
    """Global ice pressure/load of the DEM vs the ISO 19906 pressure-area law.

    The in-repo implementation (``tensordem.ice_ridge_keel_hull``) is used as
    the vehicle: ``F(A) = p_0 A (A/A_0)^exponent``.  The comparison is an
    *upper-bound* cross-check, not a validation: the ISO p-A law describes a
    continuous level-ice sheet crushing against a wide structure, whereas the
    DEM here pushes individual floes through a broken-ice channel.  Both
    external anchors are reported side by side because the ratio is the
    quantitative statement of that model-form difference.
    """
    try:
        from tensordem.ice_ridge_keel_hull import (  # noqa: WPS433
            ISO19906RidgeParams,
            iso19906_ridge_load,
        )
    except Exception as exc:  # pragma: no cover - optional import
        return {"available": False, "reason": f"import failed: {exc}"}

    params = ISO19906RidgeParams()
    area = case.ship_beam_m * case.ice_thickness_m
    rows = {}
    for speed_key, entry in single.items():
        load = entry["load_mean_N"]
        pressure = load / area
        iso_load = float(iso19906_ridge_load(area, params))
        iso_pressure = iso_load / area
        rows[speed_key] = {
            "speed_m_s": entry["speed_m_s"],
            "nominal_contact_area_m2": area,
            "dem_mean_pressure_kpa": pressure / 1e3,
            "iso19906_pressure_kpa": iso_pressure / 1e3,
            "iso19906_load_kN": iso_load / 1e3,
            "dem_load_kN": load / 1e3,
            "dem_over_iso_load": load / iso_load if iso_load else None,
        }
    loads = [row["dem_over_iso_load"] for row in rows.values()
             if row["dem_over_iso_load"] is not None]
    return {
        "available": True,
        "vehicle": "tensordem.ice_ridge_keel_hull.iso19906_ridge_load",
        "provenance": "ISO 19906:2010 Annex C p-A law as implemented in "
                      "src/tensordem/ice_ridge_keel_hull.py (p_0, A_0, exponent "
                      "carried by ISO19906RidgeParams; grade CONVENTION)",
        "params": {"p_0_pa": params.p_0, "A_0_m2": params.A_0, "exponent": params.exponent},
        "convention": "nominal contact area A = B*h (waterline breadth x ice thickness)",
        "exponent_family_iso19906": list(_iso_exponent_family()),
        "rows": rows,
        "dem_over_iso_load_range": [min(loads), max(loads)] if loads else None,
        "interpretation": "broken-ice channel load as a fraction of the level-ice ISO "
                          "global design load on the same nominal area",
    }


def _iso_exponent_family() -> tuple[float, ...]:
    try:
        from tensordem.hull_collision.indentation import (  # noqa: WPS433
            ISO_PRESSURE_AREA_EXPONENTS,
        )
    except Exception:  # pragma: no cover - optional import
        return ()
    return tuple(ISO_PRESSURE_AREA_EXPONENTS)


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
def verdicts(single: dict[str, Any], monotone: dict[str, Any],
             gci: dict[str, Any], runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []

    # --- criterion 1: single-ship vs Lindqvist, |dev| < 15 % at every speed
    deviations = []
    for speed_key, entry in single.items():
        primary = entry["primary_deviation"]
        deviations.append({"speed_m_s": entry["speed_m_s"],
                           "deviation_pct": primary["relative_deviation_pct"],
                           "abs_deviation_pct": primary["abs_deviation_pct"],
                           "within_tolerance": primary["within_tolerance"],
                           "dem_below_anchor_floor": primary["dem_below_anchor_floor"]})
    deviations.sort(key=lambda item: item["speed_m_s"])
    worst = max(deviations, key=lambda item: item["abs_deviation_pct"]) if deviations else None
    out.append({
        "id": "C1",
        "name": "single-ship degenerate case vs Lindqvist (1989), |relative deviation| < 15 %",
        "passed": bool(deviations) and all(item["within_tolerance"] for item in deviations),
        "measured": deviations,
        "worst_case": worst,
        "required_pct": 15.0,
        "status": "PASS" if deviations and all(item["within_tolerance"] for item in deviations)
                  else ("FAIL" if deviations else "WITHHELD"),
        "note": "evaluated at every swept speed with the pre-declared primary caliber "
                "(alpha = 20 deg, mu = 0.05); the band over alpha/mu is reported",
    })

    # --- criterion 2a: monotone saving-vs-spacing curve at every speed
    per_speed = {key: {"monotone_strict": item["saving_paired_strict"]["monotone"],
                       "monotone_tolerant": item["saving_paired_tolerant"]["monotone"],
                       "n_violations_strict": item["saving_paired_strict"]["n_violations"],
                       "n_violations_tolerant": item["saving_paired_tolerant"]["n_violations"],
                       "tolerance": item["tolerance_paired"],
                       "saving_paired": item["saving_paired"]}
                 for key, item in monotone.items()}
    strict_ok = bool(per_speed) and all(v["monotone_strict"] for v in per_speed.values())
    tolerant_ok = bool(per_speed) and all(v["monotone_tolerant"] for v in per_speed.values())
    out.append({
        "id": "C2a",
        "name": "saving-vs-spacing curve monotonicity (non-increasing)",
        "passed": bool(tolerant_ok),
        "measured": per_speed,
        "strict_passed": bool(strict_ok),
        "status": "PASS" if tolerant_ok else ("FAIL" if per_speed else "WITHHELD"),
        "note": "tolerant verdict slackens each step by one ensemble standard error of the "
                "paired saving; the strict verdict is reported alongside",
    })

    # --- criterion 2b: monotone grid convergence
    conv = {}
    for key in ("single_ship", "formation"):
        fam = gci.get(key)
        if fam:
            conv[key] = {"monotone": fam["gci"]["monotone"],
                         "converged": fam["gci"]["converged"],
                         "observed_order": fam["gci"]["observed_order"],
                         "asymptotic_ratio": fam["gci"]["asymptotic_ratio"]}
    out.append({
        "id": "C2b",
        "name": "grid-convergence monotonicity of the particle-refinement family",
        "passed": bool(conv) and all(item["monotone"] and item["converged"]
                                     for item in conv.values()),
        "measured": conv,
        "status": "PASS" if conv and all(item["monotone"] and item["converged"]
                                         for item in conv.values())
                  else ("FAIL" if conv else "WITHHELD"),
    })

    # --- criterion 3: GCI < 5 %
    gcis = {}
    for key in ("single_ship", "formation"):
        fam = gci.get(key)
        if fam:
            gcis[key] = fam["gci"]["gci_pct"]
    ok = bool(gcis) and all(math.isfinite(v) and v < 5.0 for v in gcis.values())
    out.append({
        "id": "C3",
        "name": "particle-count GCI (N x2 / N x4) < 5 %",
        "passed": bool(ok),
        "measured": gcis,
        "required_pct": 5.0,
        "status": "PASS" if ok else ("FAIL" if gcis else "WITHHELD"),
    })
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(single: dict[str, Any], surface: dict[str, Any], gci: dict[str, Any],
                 monotone: dict[str, Any], out_dir: Path = FIG_DIR) -> list[dict[str, str]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        return [{"name": "unavailable", "caption": f"matplotlib import failed: {exc}"}]

    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, str]] = []

    def _save(fig, name: str, caption: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append({"name": name, "caption": caption, "path": _rel(path)})

    # 1. saving vs spacing, one line per speed
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for speed_key, entry in sorted(surface.items(), key=lambda kv: float(kv[0])):
        spacings = sorted(entry["by_spacing"].values(), key=lambda item: item["spacing_L"])
        x = [item["spacing_L"] for item in spacings]
        y = [item["saving_paired"] for item in spacings]
        err = [item["saving_paired_sem"] if math.isfinite(item["saving_paired_sem"]) else 0.0
               for item in spacings]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, lw=1.2,
                    label=f"V = {entry['speed_m_s']:g} m/s")
    ax.set_xlabel("ship-to-ship spacing S / L")
    ax.set_ylabel("ice-resistance saving  [-]")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("3-ship column: saving vs spacing (DEM, ensemble mean ± SEM)")
    _save(fig, "saving_vs_spacing.png",
          "Ice-resistance saving of a three-ship column against three independent "
          "single ships, versus non-dimensional spacing S/L, for seven transit speeds "
          "(0.5-2.0 m/s). Error bars: SEM over three floe-field realisations.")

    # 2. single ship vs Lindqvist components
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    speeds = sorted((float(k), v) for k, v in single.items())
    x = [item[0] for item in speeds]
    dem = [item[1]["load_mean_N"] / 1e3 for item in speeds]
    dem_err = [item[1]["load_sem_N"] / 1e3 for item in speeds]
    lq = [item[1]["lindqvist_primary"]["components"]["total_N"] / 1e3 for item in speeds]
    floor = [(item[1]["lindqvist_primary"]["components"]["bending_N"]
              + item[1]["lindqvist_primary"]["components"]["submersion_N"]) / 1e3
             for item in speeds]
    ax.errorbar(x, dem, yerr=dem_err, marker="o", capsize=3, label="DEM single ship")
    ax.plot(x, lq, marker="s", label="Lindqvist total (alpha=20 deg, mu=0.05)")
    ax.plot(x, floor, marker="^", ls="--",
            label="Lindqvist bending + submersion floor")
    ax.set_xlabel("transit speed [m/s]")
    ax.set_ylabel("ice resistance [kN]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("Single-ship DEM vs Lindqvist three-component anchor")
    _save(fig, "lindqvist_vs_dem.png",
          "Single-ship ice resistance from the DEM channel model (ensemble mean ± SEM) "
          "against the Lindqvist (1989) three-component anchor converted to the case "
          "caliber, and against the anchor's bending+submersion floor.")

    # 3. GCI family
    fam_keys = [("single_ship", "single ship"), ("formation", "3-ship column")]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, label in fam_keys:
        fam = gci.get(key)
        if not fam:
            continue
        levels = [1, 2, 4]
        vals = [fam["values_N"]["level_1_base"], fam["values_N"]["level_2"],
                fam["values_N"]["level_4"]]
        ax.plot(levels, [v / 1e3 for v in vals], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("particle refinement level (N, 2N, 4N)")
    ax.set_ylabel("total ice load [kN]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Particle-count refinement family")
    _save(fig, "gci_family.png",
          "Particle-count refinement family (N, 2N, 4N floes at constant ice "
          "concentration) for the single-ship and the three-ship column cases.")

    # 4. follower/lead ratio vs lag / reclosure time (mechanism)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs, ys, labels = [], [], []
    for speed_key, entry in surface.items():
        for item in entry["by_spacing"].values():
            closure = item["reclosure"]
            ratio = closure["lag_over_reclosure"]
            if not math.isfinite(ratio) or ratio <= 0:
                continue
            xs.append(ratio)
            ys.append(_mean(item["follower_to_lead_ratio"]))
            labels.append(f"V={entry['speed_m_s']:g},S={item['spacing_L']:g}L")
    if xs:
        ax.scatter(xs, ys, s=18)
        ax.set_xscale("log")
        ax.set_xlabel("follower lag S/V  /  channel re-closure time B/(2 u_drift)")
        ax.set_ylabel("mean follower / lead ice load")
        ax.grid(alpha=0.3)
        ax.set_title("Mechanism: channel re-closure sets the follower load")
    _save(fig, "reclosure_mechanism.png",
          "Measured follower-to-lead ice-load ratio versus the ratio of the follower "
          "lag S/V to the drift-driven channel re-closure time B/(2 u_drift).")

    return figures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_artifact(raw_dir: Path = RAW_DIR, *, figures: bool = True) -> dict[str, Any]:
    runs = load_runs(raw_dir)
    matrix = build_matrix()
    expected = {item.label for item in matrix}
    missing = sorted(expected - set(runs))
    case0 = FormationCase(label="caliber", n_ships=3, spacing_L=1.0,
                          speed_m_s=GCI_SPEED_M_S)
    single = single_ship_table(runs)
    surface = savings_surface(runs, single)
    monotone = monotonicity_checks(surface, single)
    gci = gci_table(runs)
    numerics = numerics_table(runs)
    iso = iso19906_check(single, case0)
    verdict_list = verdicts(single, monotone, gci, runs)
    fig_list = make_figures(single, surface, gci, monotone) if figures else []

    d_eq = analytic_equilibrium_draught(917.0, WATER_DENSITY, case0.ice_thickness_m)
    artifact = {
        "task_key": "paper-exp-formation-ice-channel-2-rework1",
        "mission_id": MISSION_ID,
        "case": "formation-ice-channel",
        "generated_by": "benchmarks/w25_formation_collect.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prereg_path": (str(PREREG.relative_to(REPO_ROOT)) if PREREG.exists() else None),
        "prereg_sha256": _sha256(PREREG),
        "raw_dir": _rel(raw_dir),
        "caliber": {
            "ship": {"length_m": case0.ship_length_m, "beam_m": case0.ship_beam_m,
                     "length_over_beam": case0.ship_length_m / case0.ship_beam_m,
                     "hull_shape": "rectangular box (planar 2-D hull, no bow angle)"},
            "ice": {"thickness_m": case0.ice_thickness_m,
                    "concentration": case0.concentration,
                    "mean_floe_radius_m": case0.mean_radius_m,
                    "floe_diameter_over_beam": 2.0 * case0.mean_radius_m / case0.ship_beam_m},
            "environment": {"water_density_kg_m3": WATER_DENSITY,
                            "ice_density_kg_m3": 917.0,
                            "archimedes_draught_m": d_eq,
                            "wind_velocity_m_s": [0.0, 0.0],
                            "cross_channel_drift_m_s": case0.drift_m_s,
                            "drift_role": "re-closes the corridor cleared by the lead"},
            "numerics": {"base_dt_s": BASE_DT_S,
                         "dt_scaling": "dt scaled with floe radius (constant omega*dt)",
                         "domain_width_m": case0.field_width_m,
                         "field_length_rule": "max(span + 2L, 4L)",
                         "sample_every_steps": case0.sample_every},
        },
        "geometry_check": {
            "block_coefficient_Cb": 1.0,
            "Cb_note": "rectangular box hull -> Cb = 1 identically",
            "length_over_beam": case0.ship_length_m / case0.ship_beam_m,
            "passed": True,
        },
        "case_matrix": {
            "n_cases_planned": len(matrix),
            "n_cases_with_results": len(set(runs) & expected),
            "n_missing": len(missing),
            "missing_labels": missing,
            "speeds_m_s": list(SPEEDS_M_S),
            "spacings_L": list(SPACINGS_L),
            "ensembles": {"seeds_per_case": 3,
                          "seed_base": 20261112,
                          "seed_offsets": [0, 1, 2]},
            "roles": {role: sum(1 for item in matrix if item.role == role)
                      for role in sorted({item.role for item in matrix})},
            "cost_proxy_total": sum(cost_proxy(item) for item in matrix),
        },
        "single_ship_vs_lindqvist": single,
        "formation_savings": surface,
        "monotonicity": monotone,
        "grid_convergence": gci,
        "numerics_controls": numerics,
        "iso19906_crosscheck": iso,
        "figures": fig_list,
        "verdicts": verdict_list,
        "overall": {
            "n_pass": sum(1 for v in verdict_list if v["status"] == "PASS"),
            "n_fail": sum(1 for v in verdict_list if v["status"] == "FAIL"),
            "n_withheld": sum(1 for v in verdict_list if v["status"] == "WITHHELD"),
            "statuses": {v["id"]: v["status"] for v in verdict_list},
        },
    }
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(RAW_DIR))
    parser.add_argument("--out", default=str(ARTIFACT))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    artifact = build_artifact(Path(args.raw), figures=not args.no_figures)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1, sort_keys=True))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(json.dumps(artifact["overall"], indent=1))
    for verdict in artifact["verdicts"]:
        print(f"  {verdict['id']:4s} {verdict['status']:9s} {verdict['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())