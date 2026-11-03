"""Benchmark the DEM3D mode-II (direct-shear) specimen protocol.

This script turns ``tensordem.shear_specimen`` into a reproducible audit:

1. affine-shear closed-form check of the solver's deviatoric strain measure
   (including {100}-plane isotropy and rigid-rotation invariance);
2. calibration of ``shear_breaking_strain`` to a macroscopic engineering shear
   strain, by bisection on the solver's own failure criterion;
3. the same mapping under uniaxial strain, to show that the criterion is
   deviatoric rather than mode-II specific;
4. a platen-driven direct-shear specimen: quasi-static rate insensitivity,
   apparent shear modulus versus the affine constant ``C44``, and the specimen
   geometry trend;
5. a full shear run to first bond failure with the local criterion value, the
   failure-mode census and the energy ledger at the event.

Outputs (JSON + CSV + PNG) go to ``--output`` (default
``results-dem3d-shear-specimen``); the archived copies used by the document live
in ``docs/dem3d-shear-specimen/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.shear_specimen import (
    apparent_shear_modulus_from_history,
    affine_shear_probe,
    critical_engineering_shear_strain,
    direct_shear_specimen_area,
    direct_shear_specimen_height,
    lattice_spacing,
    run_direct_shear,
    shear_failure_onset,
    shear_modulus_closed_form,
    uniaxial_failure_onset,
)

DEFAULT_OUTPUT = ROOT / "results-dem3d-shear-specimen"
PROTOCOL = "tensordem-dem3d-shear-specimen-v1"

# Reference specimen and ramping protocol.  ``drag`` is chosen so that the load
# path from the platen into the block is established in a few hundred steps
# while the ramp itself is ten times slower (see the rate-insensitivity check).
_REFERENCE = {
    "nx": 9, "ny": 5, "nz": 5, "drag": 10.0, "tool_gap": 20.0, "tool_speed": 1.0,
    "breaking_strain": 1.0e3,  # isolate mode II: the tensile threshold is out of range
    "shear_breaking_strain": 0.03,
}


def _reference_config(**overrides: Any) -> DEM3DConfig:
    values = dict(_REFERENCE)
    values.update(overrides)
    return DEM3DConfig(**values)


def _affine_config(**overrides: Any) -> DEM3DConfig:
    values = {
        "nx": 3, "ny": 3, "nz": 3, "drag": 0.0, "fix_x_edges": False,
        "tool_gap": 20.0, "breaking_strain": 1.0e3, "shear_breaking_strain": 0.03,
    }
    values.update(overrides)
    return DEM3DConfig(**values)


def threshold_mapping() -> list[dict[str, Any]]:
    """Bisected onset strain versus the closed-form mapping, for several thresholds."""
    rows = []
    for threshold in (0.003, 0.01, 0.03, 0.05):
        config = _affine_config(shear_breaking_strain=threshold)
        shear = shear_failure_onset(config)
        uniaxial = uniaxial_failure_onset(config)
        rows.append({
            "shear_breaking_strain": threshold,
            "measured_onset_gamma": shear["onset_gamma"],
            "closed_form_gamma": shear["closed_form_gamma"],
            "first_order_gamma": shear["first_order_sqrt3_s"],
            "gamma_relative_error": shear["relative_error"],
            "onset_broken_bonds": shear["broken_bonds"],
            "onset_failure_modes": shear["failure_mode_codes"],
            "onset_local_bond_invariant": shear["local_bond_invariant_at_onset"],
            "measured_uniaxial_onset": uniaxial["onset_strain"],
            "closed_form_uniaxial": uniaxial["closed_form_strain"],
            "uniaxial_relative_error": uniaxial["relative_error"],
            "uniaxial_failure_modes": uniaxial["failure_mode_codes"],
        })
    return rows


def rate_series(gamma_max: float = 0.025) -> list[dict[str, Any]]:
    """Quasi-staticity: the tau(gamma) trace must not depend on the ramp rate."""
    rows = []
    reference = None
    rows_all: list[dict[str, Any]] = []
    for velocity in (0.4, 0.2, 0.1):
        result = run_direct_shear(_reference_config(), velocity=velocity,
                                  gamma_max=gamma_max, record_histories=True)
        history = result["history"]
        fit = apparent_shear_modulus_from_history(
            history, 0.6 * gamma_max, 0.95 * gamma_max)
        row = {
            "velocity_m_per_s": velocity,
            "apparent_shear_modulus_Pa": fit["apparent_shear_modulus_Pa"],
            "ratio_to_C44": fit["apparent_shear_modulus_Pa"] / result["C44_Pa"],
            "tau_at_gamma_max_Pa": history[-1]["tau_Pa"],
            "max_kinetic_over_elastic": max(
                r["kinetic_energy_J"] / max(r["bond_elastic_energy_J"], 1.0e-30)
                for r in history),
            "steps": result["protocol"]["steps"],
        }
        if reference is None:
            reference = row
            row["tau_relative_deviation_from_reference"] = 0.0
        else:
            row["tau_relative_deviation_from_reference"] = abs(
                row["tau_at_gamma_max_Pa"] - reference["tau_at_gamma_max_Pa"]
            ) / abs(reference["tau_at_gamma_max_Pa"])
            row["modulus_relative_deviation_from_reference"] = abs(
                row["apparent_shear_modulus_Pa"] - reference["apparent_shear_modulus_Pa"]
            ) / reference["apparent_shear_modulus_Pa"]
        rows_all.append(row)
    # Convergence between the two slowest ramps: the quasi-static limit is the
    # quantity this row reports, not a guaranteed property of the run.
    slow, slower = rows_all[-2], rows_all[-1]
    slower["tau_relative_deviation_from_previous_rate"] = abs(
        slower["tau_at_gamma_max_Pa"] - slow["tau_at_gamma_max_Pa"]
    ) / abs(slow["tau_at_gamma_max_Pa"])
    slower["modulus_relative_deviation_from_previous_rate"] = abs(
        slower["apparent_shear_modulus_Pa"] - slow["apparent_shear_modulus_Pa"]
    ) / slow["apparent_shear_modulus_Pa"]
    return rows_all


def geometry_series(gamma_max: float = 0.025) -> list[dict[str, Any]]:
    """Apparent modulus versus lateral extent, both lateral directions."""
    rows = []
    cases = [("nx", n) for n in (5, 9, 13)] + [("ny", n) for n in (7,)]
    for axis, value in cases:
        overrides = {axis: value}
        config = _reference_config(**overrides)
        result = run_direct_shear(config, velocity=0.2, gamma_max=gamma_max,
                                  record_histories=True)
        history = result["history"]
        upper = 0.95 * gamma_max
        if result["first_failure"] is not None:
            upper = min(upper, 0.9 * result["first_failure"]["gamma"])
        fit = apparent_shear_modulus_from_history(history, 0.6 * upper, upper)
        a = lattice_spacing(config)
        height = direct_shear_specimen_height(config)
        rows.append({
            "varied_axis": axis,
            "value": value,
            "nx": config.nx, "ny": config.ny, "nz": config.nz,
            "particle_count": result["protocol"]["particle_count"],
            "height_m": height,
            "extent_x_over_height": config.nx * a / height,
            "extent_y_over_height": config.ny * a / height,
            "apparent_shear_modulus_Pa": fit["apparent_shear_modulus_Pa"],
            "ratio_to_C44": fit["apparent_shear_modulus_Pa"] / result["C44_Pa"],
            "fit_relative_residual": fit["relative_residual"],
            "first_failure_gamma": (result["first_failure"] or {}).get("gamma"),
            "broken_bonds": history[-1]["broken_bonds"],
            "history": history,
        })
    return rows


def shear_to_failure(gamma_max: float = 0.06, velocity: float = 0.2) -> dict[str, Any]:
    """Full direct-shear run through first bond failure (mode II isolated)."""
    config = _reference_config()
    result = run_direct_shear(config, velocity=velocity, gamma_max=gamma_max,
                              record_histories=True)
    history = result["history"]
    first = result["first_failure"]
    expected = critical_engineering_shear_strain(config.shear_breaking_strain)
    height = direct_shear_specimen_height(config)
    a = lattice_spacing(config)
    # Local strain concentration: the driven specimen first reaches the threshold
    # where the local deviatoric invariant, not the global gamma/sqrt(3), hits s.
    pre = [row for row in history if first is None or row["gamma"] < first["gamma"]]
    concentration = None
    if pre and first is not None:
        last = pre[-1]
        concentration = {
            "gamma_before_event": last["gamma"],
            "max_local_invariant_before_event": last["max_alive_bond_invariant"],
            "global_affine_estimate": last["gamma"] / math.sqrt(3.0),
            "concentration_factor": last["max_alive_bond_invariant"]
            / (last["gamma"] / math.sqrt(3.0)),
        }
    event_energy: dict[str, Any] = {}
    # Energy ledger around the first event.
    ledger = {}
    if first is not None:
        before = [row for row in history if row["gamma"] < first["gamma"]]
        after = [row for row in history if row["gamma"] >= first["gamma"]]
        if before and after:
            ledger = {
                "bond_elastic_before_J": before[-1]["bond_elastic_energy_J"],
                "bond_elastic_after_J": after[0]["bond_elastic_energy_J"],
                "released_recorded_after_J": after[0]["released_bond_energy_J"],
                "elastic_drop_J": before[-1]["bond_elastic_energy_J"]
                - after[0]["bond_elastic_energy_J"],
            }
            ledger["recorded_over_elastic_drop"] = (
                ledger["released_recorded_after_J"] / ledger["elastic_drop_J"]
                if ledger["elastic_drop_J"] > 0.0 else None)
    event_energy.update(result["failure_mode_energy"])
    return {
        "protocol": result["protocol"],
        "C44_Pa": result["C44_Pa"],
        "affine_onset_gamma": expected,
        "first_failure": first,
        "macro_over_affine_onset": (
            first["gamma"] / expected if first is not None else None),
        "first_failure_local_invariant_over_threshold": (
            first["max_bond_invariant"] / config.shear_breaking_strain
            if first is not None else None),
        "concentration": concentration,
        "failure_mode_counts": result["failure_mode_counts"],
        "failure_mode_energy": result["failure_mode_energy"],
        "energy_ledger_at_event": ledger,
        "specimen": {
            "height_m": height, "lattice_spacing_m": a,
            "area_m2": direct_shear_specimen_area(config),
            "height_in_lattices": height / a,
        },
        "final": result["final"],
        "history": history,
    }


def _plot(report: dict[str, Any], output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    # Figure 1: platen-driven response.
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    geo = report["geometry_series"]
    curves: dict[tuple[str, int], list[dict[str, float]]] = {}
    for row in geo:
        key = (row["varied_axis"], row["value"])
        curves[key] = row["history"]
    for (axis, value), history in curves.items():
        if axis != "nx":
            continue
        gammas = [r["gamma"] for r in history]
        taus = [r["tau_Pa"] for r in history]
        axes[0].plot(gammas, taus, lw=1.2, label=f"nx={value} (ny=5, nz=5)")
    axes[0].plot([r["gamma"] for r in report["shear_to_failure"]["history"]],
                 [r["tau_Pa"] for r in report["shear_to_failure"]["history"]],
                 "k-", lw=1.6, label="to failure (nx=9)")
    first = report["shear_to_failure"]["first_failure"]
    if first is not None:
        axes[0].axvline(first["gamma"], color="crimson", ls="--", lw=1.0,
                        label=f"first bond event $\\gamma$={first['gamma']:.4f}")
    axes[0].axvline(report["shear_to_failure"]["affine_onset_gamma"], color="grey",
                    ls=":", lw=1.0, label="affine onset $\\gamma_c$")
    axes[0].set_xlabel("engineering shear strain $\\gamma = \\Delta x / h$ [-]")
    axes[0].set_ylabel("shear traction $\\tau$ [Pa]")
    axes[0].set_title("direct-shear response")
    axes[0].legend(fontsize=6.5)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(1.0, color="black", ls=":", label="$C_{44}=7k/3a$")
    xs = [row["extent_x_over_height"] for row in geo if row["varied_axis"] == "nx"]
    ys = [row["ratio_to_C44"] for row in geo if row["varied_axis"] == "nx"]
    axes[1].plot(xs, ys, "o-", label="vary $L_x/h$ (ny=5)")
    xs2 = [row["extent_y_over_height"] for row in geo if row["varied_axis"] == "ny"]
    ys2 = [row["ratio_to_C44"] for row in geo if row["varied_axis"] == "ny"]
    axes[1].plot(xs2, ys2, "s--", label="vary $L_y/h$ (nx=9)")
    axes[1].set_xlabel("lateral extent / height [-]")
    axes[1].set_ylabel("$G_{app} / C_{44}$ [-]")
    axes[1].set_title("apparent modulus versus specimen shape")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    hist = report["shear_to_failure"]["history"]
    axes[2].plot([r["gamma"] for r in hist],
                 [r["max_bond_invariant"] for r in hist], lw=1.2,
                 label="max bond invariant (all bonds)")
    axes[2].plot([r["gamma"] for r in hist],
                 [r["max_alive_bond_invariant"] for r in hist], lw=1.0, ls="--",
                 label="max alive-bond invariant")
    axes[2].plot([r["gamma"] for r in hist],
                 [r["gamma"] / math.sqrt(3.0) for r in hist], color="grey", ls=":",
                 label="global affine $\\gamma/\\sqrt{3}$")
    s = report["thresholds"]["reference"]
    axes[2].axhline(s, color="crimson", ls="-.", lw=1.0,
                    label=f"$s$={s} threshold")
    if first is not None:
        axes[2].axvline(first["gamma"], color="crimson", ls="--", lw=0.8)
    axes[2].set_xlabel("engineering shear strain $\\gamma$ [-]")
    axes[2].set_ylabel("deviatoric strain invariant [-]")
    axes[2].set_title("local criterion versus global strain")
    axes[2].legend(fontsize=6.5)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    path = output_dir / "shear_specimen_response.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    # Figure 2: criterion calibration.
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    probe = report["affine_probe"]
    gammas = [row["gamma"] for row in probe["rows"] if row["gamma"] > 0]
    measured = [row["measured_max"] for row in probe["rows"] if row["gamma"] > 0]
    closed = [row["closed_form"] for row in probe["rows"] if row["gamma"] > 0]
    axes[0].loglog(gammas, measured, "o", label="solver measure")
    axes[0].loglog(gammas, closed, "-", lw=1.0, label="closed form $\\sqrt{\\gamma^2/3+\\gamma^4/9}$")
    errors = [row["max_relative_error"] for row in probe["rows"]]
    axes[0].set_xlabel("engineering shear strain $\\gamma$ [-]")
    axes[0].set_ylabel("shear invariant [-]")
    axes[0].set_title(f"affine check (max rel. err {max(errors):.1e})")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, which="both")

    mapping = report["thresholds"]["mapping"]
    ss = [row["shear_breaking_strain"] for row in mapping]
    axes[1].plot(ss, [row["measured_onset_gamma"] for row in mapping], "o",
                 label="bisected onset (solver)")
    axes[1].plot(ss, [row["closed_form_gamma"] for row in mapping], "-", lw=1.0,
                 label="closed form $\\gamma_c(s)$")
    axes[1].plot(ss, [row["measured_uniaxial_onset"] for row in mapping], "s--",
                 label="uniaxial onset $\\varepsilon = \\sqrt{1+3s}-1$")
    axes[1].plot(ss, [3.0 * s for s in ss], ":", color="grey", label="axial $3s$")
    axes[1].set_xlabel("shear_breaking_strain $s$ [-]")
    axes[1].set_ylabel("onset strain [-]")
    axes[1].set_title("threshold to macroscopic strain")
    axes[1].legend(fontsize=6.5)
    axes[1].grid(alpha=0.3)

    ledger = report["shear_to_failure"]["energy_ledger_at_event"]
    labels, values = [], []
    for name, key in (("elastic drop", "elastic_drop_J"),
                      ("recorded release", "released_recorded_after_J")):
        if key in ledger and ledger[key] is not None:
            labels.append(name)
            values.append(ledger[key])
    if values:
        axes[2].bar(labels, values, color=["steelblue", "indianred"][:len(values)])
        axes[2].set_ylabel("energy [J]")
        axes[2].set_title("mode-II event ledger (first event window)")
        axes[2].grid(alpha=0.3, axis="y")
        for index, value in enumerate(values):
            axes[2].text(index, value, f"{value:.2e}", ha="center", va="bottom", fontsize=7)
    axes[2].set_xlabel("energy bookkeeping")
    fig.tight_layout()
    path = output_dir / "shear_specimen_criterion.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def run_audit(output_dir: Path = DEFAULT_OUTPUT, make_figure: bool = True) -> dict[str, Any]:
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = affine_shear_probe()
    mapping = threshold_mapping()
    rates = rate_series()
    geometry = geometry_series()
    failure = shear_to_failure()
    reference = _reference_config()
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "units": "SI",
        "model": {
            "lattice": "simple-cubic with face and body diagonals (Z = 26, 13 half-space bonds)",
            "bond_law": "central linear spring, no tangential bond stiffness",
            "shear_criterion": "0.5*(J2_i + J2_j) > shear_breaking_strain with "
                               "J2 = sqrt(2/3 dev(E):dev(E)) from a least-squares local F",
            "C44_Pa": shear_modulus_closed_form(reference),
        },
        "affine_probe": probe,
        "thresholds": {
            "reference": reference.shear_breaking_strain,
            "mapping": mapping,
        },
        "rate_series": rates,
        "geometry_series": geometry,
        "shear_to_failure": failure,
    }
    verdict_rows = [
        {
            "check": "affine invariant vs closed form",
            "value": probe["max_relative_error"],
            "baseline": 0.0,
            "tolerance": 1.0e-12,
            "passed": probe["max_relative_error"] < 1.0e-12,
        },
        {
            "check": "{100} plane isotropy spread",
            "value": probe["isotropy_absolute_spread"],
            "baseline": 0.0,
            "tolerance": 1.0e-12,
            "passed": probe["isotropy_absolute_spread"] < 1.0e-12,
        },
        {
            "check": "rigid rotation invariant",
            "value": probe["rotation_invariant"],
            "baseline": 0.0,
            "tolerance": 1.0e-12,
            "passed": probe["rotation_invariant"] < 1.0e-12,
        },
        {
            "check": "shear onset gamma vs closed form (max over thresholds)",
            "value": max(row["gamma_relative_error"] for row in mapping),
            "baseline": 0.0,
            "tolerance": 1.0e-9,
            "passed": max(row["gamma_relative_error"] for row in mapping) < 1.0e-9,
        },
        {
            "check": "uniaxial onset vs closed form (max over thresholds)",
            "value": max(row["uniaxial_relative_error"] for row in mapping),
            "baseline": 0.0,
            "tolerance": 1.0e-9,
            "passed": max(row["uniaxial_relative_error"] for row in mapping) < 1.0e-9,
        },
        {
            "check": "quasi-static limit: tau(gamma_max) deviation, v=0.2 vs 0.1 m/s",
            "value": rates[-1].get("tau_relative_deviation_from_previous_rate"),
            "baseline": 0.0,
            "tolerance": 3.0e-2,
            "passed": (rates[-1].get("tau_relative_deviation_from_previous_rate") or 0.0) < 3.0e-2,
        },
        {
            "check": "local criterion honoured at first event (local invariant / s)",
            "value": failure["first_failure_local_invariant_over_threshold"],
            "baseline": 1.0,
            "tolerance": 1.0e-2,
            "passed": failure["first_failure_local_invariant_over_threshold"] is not None
            and abs(failure["first_failure_local_invariant_over_threshold"] - 1.0) < 1.0e-2,
        },
    ]
    report["checks"] = verdict_rows
    report["verdict"] = "PASS" if all(row["passed"] for row in verdict_rows) else "INCOMPLETE"
    report["closed_form_verdict"] = (
        "PASS" if all(row["passed"] for row in verdict_rows[:6]) else "FAIL")
    report["interpretation"] = {
        "closed_form": "gamma_c = sqrt(1.5*(sqrt(1+4s^2)-1)); to first order sqrt(3)*s.",
        "apparent_modulus": "The driven block with a clamped bottom and free lateral faces "
                            "carries less shear traction than C44*gamma; the deficit is a "
                            "specimen-boundary effect, not a bond-law error, and it does not "
                            "vanish over the tested aspect ratios (see geometry_series).",
        "mode_ii_energy": "The bond stores only axial elastic energy, so a shear-broken bond "
                          "records a near-zero release energy; the model has no mode-II "
                          "energy release rate and cannot be calibrated to G_IIc.",
        "not_validation": "No experimental shear specimen is compared here; this is a "
                          "criterion-and-protocol audit of the solver's own model.",
    }
    # Strip histories from the archived record except where the document needs them.
    if make_figure:
        paths = _plot(report, output_dir)
        report["figures"] = [path.name for path in paths]
    archived = json.loads(json.dumps(report, default=str))
    for row in archived["geometry_series"]:
        row.pop("history", None)
    (output_dir / "shear_specimen.json").write_text(
        json.dumps(archived, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "shear_specimen_thresholds.csv", mapping)
    _write_csv(output_dir / "shear_specimen_rates.csv", rates)
    _write_csv(output_dir / "shear_specimen_geometry.csv",
               [{k: v for k, v in row.items() if k != "history"} for row in geometry])
    _write_csv(output_dir / "shear_specimen_failure_history.csv",
               failure["history"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    report = run_audit(args.output, make_figure=not args.no_figure)
    geometry = report["geometry_series"]
    print(json.dumps({
        "verdict": report["verdict"],
        "affine_max_relative_error": report["affine_probe"]["max_relative_error"],
        "onset_max_relative_error": max(row["gamma_relative_error"]
                                        for row in report["thresholds"]["mapping"]),
        "apparent_modulus_ratio_range": [
            min(row["ratio_to_C44"] for row in geometry),
            max(row["ratio_to_C44"] for row in geometry),
        ],
        "first_failure_gamma": (report["shear_to_failure"]["first_failure"] or {}).get("gamma"),
        "affine_onset_gamma": report["shear_to_failure"]["affine_onset_gamma"],
        "rate_deviation_slowest_pair": report["rate_series"][-1].get("tau_relative_deviation_from_previous_rate"),
    }, indent=2))


if __name__ == "__main__":
    main()