"""Ice-beam three-point-bending anchor for the DEM3D bonded-particle model.

Reproduces the Timoshenko/Euler-Bernoulli closed-form solution for
``L = 1 m, b = 0.2 m, h = 0.1 m``, ``d = 5 mm`` bonded particles, ice modulus
``E = 9 GPa`` and bond (flexural) strength ``sigma_b = 1.5 MPa``:

* elastic anchor     -- DEM equilibrium vs beam theory (deflection + stress)
* fracture cascade   -- first-bond event, ultimate load, crack path
* strength scan      -- ``sigma_b`` +/- 20 %
* negative control   -- the same beam with every bond removed
* resolution table   -- discretisation bias of the lattice bending stiffness

Usage::

    python scripts/benchmark_ice_beam_bending.py --output artifacts/ice-beam-bending
    python scripts/benchmark_ice_beam_bending.py --quick   # coarse, for CI

Every reported number is written to ``report.json`` and re-used (never re-typed)
by :func:`tensordem.ice_beam_bending.evaluate_criteria`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem import ice_beam_bending as ib  # noqa: E402

PROBE_FRACTION = 0.10
BASE_LOAD_N = 200.0


def _rel(a: float, b: float) -> float:
    return float(abs(a - b) / abs(b)) if b else float("nan")


def _deflection_comparison(anchor: dict, theory: ib.BeamTheory,
                           load_N: float) -> dict[str, float]:
    """Score a stored DEM deflection profile against one beam-theory variant."""
    x = np.asarray(anchor["deflection"]["x_over_L"], dtype=float) * theory.spec.length_m
    w_dem = np.asarray(anchor["deflection"]["w_dem_m"], dtype=float)
    h = theory.spec.thickness_m
    L = theory.spec.length_m
    window = (x >= 0.10 * L) & (x <= 0.90 * L) & (np.abs(x - L / 2.0) > h)
    mid = len(x) // 2
    w_ts = theory.deflection(x, load_N, "timoshenko")
    w_eb = theory.deflection(x, load_N, "euler_bernoulli")
    return {
        "strain_state": theory.strain_state,
        "young_modulus_Pa": theory.young_modulus_Pa,
        "delete_": None,
        "midspan_dem_m": float(w_dem[mid]),
        "midspan_timoshenko_m": float(w_ts[mid]),
        "midspan_euler_bernoulli_m": float(w_eb[mid]),
        "rel_error_midspan_vs_timoshenko": _rel(w_dem[mid], w_ts[mid]),
        "rel_error_midspan_vs_euler_bernoulli": _rel(w_dem[mid], w_eb[mid]),
        "rel_error_profile_max_vs_timoshenko": float(np.max(
            np.abs(w_dem[window] - w_ts[window]) / np.abs(w_ts[window]))),
        "rel_error_profile_max_vs_euler_bernoulli": float(np.max(
            np.abs(w_dem[window] - w_eb[window]) / np.abs(w_eb[window]))),
        "effective_flexural_rigidity_Nm2": float(
            theory.flexural_rigidity_Nm2 * w_ts[mid] / w_dem[mid]),
        "flexural_rigidity_ratio_dem_over_theory": float(w_ts[mid] / w_dem[mid]),
    }


def run_case(spec: ib.IceBeamSpec, *, do_axial: bool = True,
             do_cascade: bool = True, cascade_events: int = 40,
             events_per_step: int = 3, do_anchor: bool = True) -> dict:
    """Full anchor package for one discretisation."""
    t0 = time.time()
    model = ib.beam_model(spec)
    properties = model.properties
    theory_ps = ib.beam_theory(spec, properties)
    theory_pn = ib.beam_theory(spec, properties, plane_strain=True)
    out: dict = {
        "spec": spec.as_dict(),
        "model": model.as_dict(),
        "properties": {k: float(v) for k, v in properties.items()},
        "theory_plane_stress": theory_ps.as_dict(),
        "theory_plane_strain": theory_pn.as_dict(),
        "build_seconds": round(time.time() - t0, 1),
    }
    if do_axial:
        t0 = time.time()
        out["axial_modulus"] = ib.axial_modulus_check(model)
        out["axial_seconds"] = round(time.time() - t0, 1)
    t0 = time.time()
    anchor = ib.elastic_anchor(model, theory_ps, probe_fraction=PROBE_FRACTION) \
        if do_anchor else None
    out["anchor_seconds"] = round(time.time() - t0, 1)
    if anchor is None:
        return out
    anchor["deflection_plane_stress"] = _deflection_comparison(
        anchor, theory_ps, anchor["probe_load_N"])
    anchor["deflection_plane_strain"] = _deflection_comparison(
        anchor, theory_pn, anchor["probe_load_N"])
    out["anchor"] = anchor
    if do_cascade:
        t0 = time.time()
        cascade = ib.fracture_cascade(
            model, theory_ps, start_load_N=anchor["first_failure"][
                "first_break_load_from_elastic_probe_N"],
            max_events=cascade_events, events_per_step=events_per_step)
        out["cascade_seconds"] = round(time.time() - t0, 1)
        out["cascade"] = {k: v for k, v in cascade.items()
                          if k not in ("events", "history")}
        out["cascade_events"] = cascade["events"]
        out["cascade_history"] = cascade["history"]
        out["crack"] = ib.crack_report(cascade, spec)
    return out


def resolution_table(diameters, *, do_cascade: bool = False) -> list[dict]:
    rows = []
    for d in diameters:
        spec = ib.IceBeamSpec().with_particle_diameter(d)
        case = run_case(spec, do_axial=(d >= 0.0125), do_cascade=do_cascade,
                        cascade_events=30)
        anchor = case["anchor"]
        d_ps = anchor["deflection_plane_stress"]
        d_pn = anchor["deflection_plane_strain"]
        rows.append({
            "particle_diameter_m": d,
            "depth_particles": spec.counts()[2] - 1,
            "particle_count": case["model"]["particle_count"],
            "bond_count": case["model"]["bond_count"],
            "rel_error_midspan_vs_timoshenko_plane_stress":
                d_ps["rel_error_midspan_vs_timoshenko"],
            "rel_error_midspan_vs_timoshenko_plane_strain":
                d_pn["rel_error_midspan_vs_timoshenko"],
            "rel_error_profile_max_vs_timoshenko_plane_stress":
                d_ps["rel_error_profile_max_vs_timoshenko"],
            "flexural_rigidity_ratio": d_ps["flexural_rigidity_ratio_dem_over_theory"],
            "rel_error_sigma_max_fit": anchor["stress"]["rel_error_sigma_max_fit"],
            "rel_error_first_break_load":
                anchor["first_failure"]["rel_error_first_break_load"],
            "rel_error_section_moment": anchor["stress"]["rel_error_section_moment"],
            "pcg_iterations": anchor["solution"]["iterations"],
            "solver_seconds": case["anchor_seconds"],
        })
    return rows


def load_deflection_curve(loads, theory: ib.BeamTheory, anchor: dict,
                          model: ib.BeamModel) -> list[dict]:
    """Elastic load-deflection ramp (linear response, one solve per point)."""
    rows = []
    for load in loads:
        solution = ib.static_solution(model, load, verify=False)
        w = ib.deflection_profile(model, solution.displacement)[
            "w_neutral_m"][model.midspan_index]
        rows.append({
            "load_N": float(load),
            "midspan_deflection_m": float(w),
            "timoshenko_m": theory.deflection_midspan(load),
            "euler_bernoulli_m": theory.deflection_midspan(load, "euler_bernoulli"),
        })
    return rows


def make_figures(output: Path, report: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    anchor = report["primary"]["anchor"]
    deflect = anchor["deflection"]
    x = np.asarray(deflect["x_over_L"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, np.asarray(deflect["w_dem_m"]) * 1e3, "o", ms=3.0,
            label="DEM (rate-free lattice equilibrium)")
    ax.plot(x, np.asarray(deflect["w_timoshenko_m"]) * 1e3, "-",
            label="Timoshenko (E, G from lattice)")
    ax.plot(x, np.asarray(deflect["w_euler_bernoulli_m"]) * 1e3, "--",
            label="Euler-Bernoulli")
    ax.set_xlabel("x / L")
    ax.set_ylabel("mid-plane deflection w [mm]")
    ax.set_title(f"Ice beam L/h=10, P={anchor['probe_load_N']:.0f} N "
                 f"({report['primary']['spec']['particle_diameter_m']*1e3:.1f} mm particles)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = output / "deflection_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    figures.append({"name": "deflection_profile", "path": path,
                    "caption": "DEM mid-plane deflection profile vs Timoshenko and "
                               "Euler-Bernoulli beam theory at 10 % of the flexural "
                               "failure load; error window excludes one thickness at "
                               "the supports and at the load line."})

    layers = anchor["stress"]["layers"]
    z = np.asarray([row["z_over_h"] for row in layers])
    sd = np.asarray([row["sigma_xx_dem_Pa"] for row in layers])
    st = np.asarray([row["sigma_xx_theory_Pa"] for row in layers])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(sd / 1e3, z, "o-", ms=4, label="DEM virial $\\sigma_{xx}$ (midspan)")
    ax.plot(st / 1e3, z, "--", label="beam theory $M(z_{na}-z)/I$")
    ax.axvline(0.0, color="k", lw=0.6)
    ax.set_ylabel("z / h")
    ax.set_xlabel("$\\sigma_{xx}$ [kPa]")
    ax.set_title("Bending stress through the depth at midspan")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = output / "stress_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    figures.append({"name": "stress_profile", "path": path,
                    "caption": "Midspan bending stress through the depth: DEM virial "
                               "stress per lattice layer versus the linear beam-theory "
                               "profile."})

    events = report["primary"].get("cascade_events", [])
    if events:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ex = [e["x_over_L"] for e in events]
        ez = [e["z_over_h"] for e in events]
        order = [e["event"] for e in events]
        scatter = ax.scatter(ex, ez, c=order, cmap="viridis", s=18)
        fig.colorbar(scatter, label="bond-break event index")
        ax.set_xlabel("x / L")
        ax.set_ylabel("z / h")
        ax.set_title(f"Quasi-static crack path at P = "
                     f"{report['primary']['cascade']['peak_load_N']:.0f} N")
        ax.grid(alpha=0.3)
        path = output / "crack_path.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        figures.append({"name": "crack_path", "path": path,
                        "caption": "Location of every broken bond in the quasi-static "
                                   "damage cascade; colour is the event index, showing "
                                   "initiation at the midspan bottom fibre and upward "
                                   "propagation."})

    ramp = report["primary"].get("load_deflection", [])
    if ramp:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ld = [r["load_N"] for r in ramp]
        ax.plot([r["midspan_deflection_m"] * 1e3 for r in ramp], ld, "o-",
                label="DEM equilibrium")
        ax.plot([r["timoshenko_m"] * 1e3 for r in ramp], ld, "--",
                label="Timoshenko")
        peak = report["primary"]["cascade"]["peak_load_N"]
        ax.axhline(peak, color="r", lw=0.8,
                   label=f"first-bond failure {peak:.0f} N")
        ax.set_xlabel("midspan deflection [mm]")
        ax.set_ylabel("central load P [N]")
        ax.set_title("Load-deflection response and predicted ultimate load")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        path = output / "load_deflection.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        figures.append({"name": "load_deflection", "path": path,
                        "caption": "Load-deflection response: DEM lattice equilibrium "
                                   "against Timoshenko, with the first-bond failure "
                                   "load from the elastic anchor."})

    table = report["resolution_table"]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ratio = np.asarray([row["depth_particles"] for row in table], dtype=float)
    err = np.asarray([row["rel_error_midspan_vs_timoshenko_plane_stress"] for row in table])
    err_pn = np.asarray([row["rel_error_midspan_vs_timoshenko_plane_strain"]
                         for row in table])
    ax.plot(1.0 / ratio, err * 100, "o-", label="vs plane-stress $E$")
    ax.plot(1.0 / ratio, err_pn * 100, "s--", label="vs plate modulus $E/(1-\\nu^2)$")
    ax.set_xlabel("particle diameter / depth  (a / h)")
    ax.set_ylabel("midspan deflection error [%]")
    ax.set_title("Discretisation bias of the lattice bending stiffness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = output / "resolution_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    figures.append({"name": "resolution_convergence", "path": path,
                    "caption": "Midspan deflection error versus particle size over "
                               "depth for both beam-theory references: the lattice "
                               "bending bias decreases steadily with refinement."})

    control = report.get("negative_control")
    if control:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for key, label in (("bonded", "bonded beam"), ("unbonded", "no bonds (friction only)")):
            samples = control[key]["samples"]
            ax.semilogy([s["step"] for s in samples],
                        [abs(s["midspan_deflection_m"]) for s in samples], "o-",
                        ms=3, label=label)
        ax.set_xlabel("explicit step")
        ax.set_ylabel("|midspan deflection| [m]")
        ax.set_title(f"Negative control at P = {control['load_N']:.0f} N")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        path = output / "negative_control.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        figures.append({"name": "negative_control", "path": path,
                        "caption": "Paired negative control: with the identical load "
                                   "and supports, the bonded beam stays at its elastic "
                                   "deflection while the unbonded packing collapses."})
    return figures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "ice-beam-bending")
    parser.add_argument("--particle-diameter-mm", type=float, default=5.0,
                        help="primary discretisation (task specification: 5 mm)")
    parser.add_argument("--resolution-table-mm", type=str,
                        default="25,12.5,10,5",
                        help="comma separated particle diameters for the refinement study")
    parser.add_argument("--crack-diameter-mm", type=float, default=12.5,
                        help="discretisation used for the damage cascade "
                             "(coarser than the anchor: each bond event needs a "
                             "fresh factorisation of the damaged network)")
    parser.add_argument("--quick", action="store_true",
                        help="coarse CI-grade run (12.5 mm primary, no 5 mm case)")
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if args.quick:
        primary_d = 0.0125
        diameters = [0.025, 0.0125]
    else:
        primary_d = args.particle_diameter_mm / 1000.0
        diameters = sorted({float(v) / 1000.0 for v in
                            args.resolution_table_mm.split(",") if v.strip()},
                           reverse=True)

    spec = ib.IceBeamSpec().with_particle_diameter(primary_d)
    theory = ib.beam_theory(spec)
    coefficients = ib.lattice_elastic_properties(spec)

    # ---- primary case ---------------------------------------------------
    primary = run_case(spec, do_axial=primary_d >= 0.0125, do_cascade=False)

    # ---- crack path at a coarser discretisation -------------------------
    crack_d = args.crack_diameter_mm / 1000.0
    crack_spec = ib.IceBeamSpec().with_particle_diameter(crack_d)
    crack_case = run_case(crack_spec, do_axial=False, do_cascade=True,
                          cascade_events=10, events_per_step=3, do_anchor=True)
    crack_model = ib.beam_model(crack_spec)
    crack_theory = ib.beam_theory(crack_spec)
    ramp_loads = np.linspace(0.0, 0.9 * crack_case["cascade"]["peak_load_N"], 11)[1:]
    primary["load_deflection"] = load_deflection_curve(ramp_loads, crack_theory,
                                                       crack_case["anchor"],
                                                       crack_model)
    primary["crack_case_diameter_m"] = crack_d
    primary["cascade"] = crack_case["cascade"]
    primary["cascade_events"] = crack_case["cascade_events"]
    primary["cascade_history"] = crack_case["cascade_history"]
    primary["crack"] = crack_case["crack"]
    primary["crack_anchor"] = crack_case["anchor"]
    del crack_model

    # ---- strength scan (elastic probe; identical stiffness, different eps_b) ----
    sensitivity = ib.stress_sensitivity(spec, scales=(0.8, 1.0, 1.2))

    # ---- explicit-integrator controls at a coarse discretisation ---------
    control_spec = ib.IceBeamSpec().with_particle_diameter(0.025)
    control_model = ib.beam_model(control_spec)
    control_theory = ib.beam_theory(control_spec)
    control_load = 0.5 * control_theory.failure_load()
    drift = ib.explicit_equilibrium_drift(
        control_model, control_load,
        displacement=ib.static_solution(control_model, control_load,
                                        verify=False).displacement,
        steps=100, sample_every=20)
    negative = ib.unbonded_collapse(control_model, control_load, steps=200,
                                    sample_every=20)
    negative["particle_diameter_m"] = control_spec.particle_diameter_m
    negative["bonded_static_reference_m"] = control_theory.deflection_midspan(control_load)
    del control_model

    # ---- refinement study ----------------------------------------------
    table = resolution_table(diameters, do_cascade=False)
    spec_row = [row for row in table
                if abs(row["particle_diameter_m"] - primary_d) < 1e-12]
    coarse = [row for row in table if row["particle_diameter_m"] > primary_d]
    convergence = None
    if coarse:
        finest_coarse = min(coarse, key=lambda r: r["particle_diameter_m"])
        change = abs(finest_coarse["rel_error_midspan_vs_timoshenko_plane_stress"]
                     - primary["anchor"]["deflection"][
                         "rel_error_midspan_vs_timoshenko"]) if spec_row else None
        convergence = {
            "reference_particle_diameter_m": finest_coarse["particle_diameter_m"],
            "reference_error": finest_coarse[
                "rel_error_midspan_vs_timoshenko_plane_stress"],
            "fine_error": primary["anchor"]["deflection"][
                "rel_error_midspan_vs_timoshenko"],
            "deflection_rel_change": float(change) if change is not None else None,
            "flexural_rigidity_ratio_fine":
                primary["anchor"]["deflection_plane_stress"][
                    "flexural_rigidity_ratio_dem_over_theory"],
            "flexural_rigidity_ratio_coarse": finest_coarse["flexural_rigidity_ratio"],
        }
        # bias law: ratio - 1 ~ c * a / h
        ratios = [(1.0 / row["depth_particles"],
                   row["flexural_rigidity_ratio"] - 1.0) for row in table
                  if row["depth_particles"] >= 3]
        if len(ratios) >= 2:
            slope, intercept = np.polyfit([r[0] for r in ratios],
                                         [r[1] for r in ratios], 1)
            convergence["bias_law_slope_times_a_over_h"] = float(slope)
            convergence["bias_law_intercept"] = float(intercept)

    report = {
        "protocol": ib.ICE_BEAM_PROTOCOL,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary": primary,
        "sensitivity": sensitivity,
        "equilibrium_drift": drift,
        "negative_control": negative,
        "resolution_table": table,
        "resolution_convergence": convergence,
        "harness": {
            "probe_fraction_of_failure_load": PROBE_FRACTION,
            "primary_particle_diameter_m": primary_d,
            "resolution_diameters_m": diameters,
            "theory_constants": {k: float(v) for k, v in coefficients.items()},
            "wall_seconds": None,
        },
    }
    criteria = ib.evaluate_criteria(report)
    report["criteria"] = criteria
    report["verdict"] = "PASS" if criteria["all_pass"] else "FAIL"
    report["harness"]["wall_seconds"] = round(time.time() - t_start, 1)

    (output / "ice_beam_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (output / "resolution_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    if primary.get("cascade_events"):
        with (output / "crack_events.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(primary["cascade_events"][0]))
            writer.writeheader()
            writer.writerows(primary["cascade_events"])
    if primary.get("load_deflection"):
        with (output / "load_deflection.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(primary["load_deflection"][0]))
            writer.writeheader()
            writer.writerows(primary["load_deflection"])

    figures = make_figures(output, report)
    report["figures"] = [{"name": f["name"], "path": str(f["path"].relative_to(ROOT)),
                         "caption": f["caption"]} for f in figures]
    (output / "ice_beam_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("verdict", "criteria", "resolution_table",
                               "resolution_convergence")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())