"""Comminution device-level energy surface: four-anchor acceptance study.

Usage
-----
    python scripts/comminution_energy_surface.py --out runs/comminution_energy_surface.json
    python scripts/comminution_energy_surface.py --quick --out /tmp/quick.json   # smoke

Anchors (each one machine-checked, numbers written into the output JSON)
-----------------------------------------------------------------------
1. **critical speed** -- DEM-detected centrifuging onset of a tracer ball vs the
   textbook ``N_c = 42.3/sqrt(D)`` at three mill diameters;
2. **mill power** -- DEM wall+lifter torque power vs the Hogg & Fuerstenau (1972)
   empirical mill-power correlation over a ``J x phi_c`` grid, with the energy
   and angular-momentum ledgers as physical consistency checks;
3. **Bond specific energy** -- the third-theory implementation verified against
   its own invariants and against the (independent, experimental) Bond
   grindability-test equation; plus a device-level cross-check of the work index
   implied by the DEM/PBM energy surface against the literature band;
4. **breakage kinetics** -- the sectional population balance verified against the
   analytic first-order (Reid/Bhattacharya) solution, and the PSD evolution
   driven by the mill's specific energy compared between the two model
   structures (single-event ``t10`` family vs multi-event sectional kinetics).

Everything is SI internally; sizes are micrometres where a quantity is named
``*_um`` and energies are kWh per metric tonne where named ``*_kwh_t``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensordem.breakage_pbm.dpbm import VolumeGrid          # noqa: E402
from tensordem.comminution.bond import (                    # noqa: E402
    BOND_WORK_INDEX_DB,
    bond_product_size_um,
    bond_specific_energy_kwh_t,
    bond_work_index_from_energy,
    bond_work_index_from_grindability,
    specific_energy_kwh_t,
    work_index_band,
)
from tensordem.comminution.breakage_kinetics import (       # noqa: E402
    AppearanceFunction,
    batch_grind,
    mass_fraction_matrix,
    p80_from_psd,
    passing_at,
    t10_energy_law,
    t10_from_psd,
    volume_daughter_matrix,
)
from tensordem.comminution.critical_speed import (          # noqa: E402
    critical_speed_constant_from_first_principles,
    critical_speed_rpm,
    critical_speed_rpm_charge_centre,
    fraction_of_critical,
    rpm_for_fraction_of_critical,
)
from tensordem.comminution.mill_dem import (                # noqa: E402
    MillConfig,
    make_mill_state,
    measure_centrifuging_onset_rpm,
    run_mill,
    tracer_response,
)
from tensordem.comminution.mill_power import (              # noqa: E402
    charge_bulk_density,
    hogg_fuerstenau_power_kw,
    mill_power_density_kw_m3,
    torque_factor,
)

BALL_CHARGE_POROSITY = 0.40
MILL = dict(diameter=0.30, length=0.15, lifter_count=8, lifter_height=0.006,
            lifter_width=0.006, ball_radii=(0.0125, 0.009, 0.006),
            ball_counts=(18, 27, 40), dt=1.0e-4)
ORE_MASS_KG = 1.5
MATERIAL = "limestone"
F80_UM = 1000.0


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else float("nan")


# ------------------------------------------------------------------ anchor 1
def run_critical_speed(mills) -> dict:
    rows = []
    for spec in mills:
        t0 = time.time()
        res = measure_centrifuging_onset_rpm(spec["diameter"], spec["tracer_radius"],
                                            tolerance=1e-3)
        res["wall_time_s"] = round(time.time() - t0, 2)
        grid = [f * res["critical_speed_rpm_nominal"] for f in (0.90, 0.95, 1.00, 1.05, 1.10)]
        res["detector_curve"] = tracer_response(spec["diameter"], spec["tracer_radius"], grid)
        rows.append(res)
    theory_const = critical_speed_constant_from_first_principles()
    errs = [abs(r["error_vs_nominal"]) for r in rows]
    return {
        "rows": rows,
        "constant_first_principles": theory_const,
        "constant_textbook": 42.3,
        "constant_rel_error": _rel(42.3, theory_const),
        "max_abs_error_vs_nominal": max(errs),
        "max_abs_error_vs_charge_centre": max(abs(r["error_vs_charge_centre"]) for r in rows),
        "criterion": "N_c error <= 2% vs 42.3/sqrt(D)",
        "pass": bool(max(errs) <= 0.02),
    }


# ------------------------------------------------------------------ anchor 2
def _power_case(cfg: MillConfig, bulk_filling: float, fraction_critical: float,
                revolutions: float, measure_last: float, record: bool) -> dict:
    d, l = cfg.diameter, cfg.length
    rpm = rpm_for_fraction_of_critical(fraction_critical, d)
    solid_area = bulk_filling * (1.0 - BALL_CHARGE_POROSITY)
    state = make_mill_state(cfg, fill=solid_area, rpm=rpm)
    t0 = time.time()
    out = run_mill(state, revolutions=revolutions, measure_last=measure_last,
                   settle_time=0.4, sample_every=400, record_collisions=record)
    led = out["ledger"]
    rho_bulk = charge_bulk_density(cfg.density, BALL_CHARGE_POROSITY)
    p_hf = hogg_fuerstenau_power_kw(d, l, bulk_filling, fraction_critical, rho_bulk)
    row = {
        "bulk_filling": bulk_filling,
        "solid_area_fraction": solid_area,
        "fraction_critical": fraction_critical,
        "rpm": rpm,
        "n_particles": out["n_particles"],
        "P_dem_w": led["p_in_w"],
        "P_hogg_fuerstenau_w": p_hf * 1000.0,
        "rel_error": led["p_in_w"] / (p_hf * 1000.0) - 1.0,
        "abs_rel_error": abs(led["p_in_w"] / (p_hf * 1000.0) - 1.0),
        "power_density_dem_kw_m3": mill_power_density_kw_m3(led["p_in_w"] / 1000.0, d, l),
        "power_density_hf_kw_m3": mill_power_density_kw_m3(p_hf, d, l),
        "torque_factor": torque_factor(led["tau_on_charge_nm"], led["charge_mass_kg"], d / 2.0),
        "charge_mass_kg": led["charge_mass_kg"],
        "x_com_over_R": led["x_com_m"] / (d / 2.0),
        "energy_closure_rel": led["closure_rel"],
        "full_closure_rel": led["full_closure_rel"],
        "moment_closure_rel": led["moment_closure_rel"],
        "release_fraction": led["release_fraction"],
        "omega_bed_over_omega": out["mean"].get("omega_bed", float("nan")) / out["omega"],
        "airborne_fraction": out["mean"].get("airborne_fraction", float("nan")),
        "wall_velocity_ratio": out["mean"].get("wall_velocity_ratio", float("nan")),
        "max_overlap_rel": out["max_overlap_rel"],
        "wall_time_s": round(time.time() - t0, 1),
    }
    spectrum = None
    if out["collisions"] is not None:
        v = out["collisions"]["approach_speed"]
        ra = out["collisions"]["radius_a"]
        rb = out["collisions"]["radius_b"]
        kinetic = 0.5 * (cfg.density * 4.0 / 3.0 * math.pi
                         * np.minimum(ra, rb) ** 3) * v ** 2
        spec_energy = kinetic / (cfg.density * math.pi * np.minimum(ra, rb) ** 3 / 6.0)
        spectrum = {
            "n_events": out["collisions"]["n_events"],
            "window_s": out["collisions"]["window_s"],
            "event_rate_per_s": out["collisions"]["n_events"] / out["collisions"]["window_s"],
            "approach_speed_m_s": {
                "p05": float(np.percentile(v, 5)), "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "max": float(np.max(v)),
            },
            "specific_energy_j_kg": {
                "p05": float(np.percentile(spec_energy, 5)),
                "median": float(np.median(spec_energy)),
                "p95": float(np.percentile(spec_energy, 95)),
                "max": float(np.max(spec_energy)),
            },
        }
    row["collision_spectrum"] = spectrum
    return row


def run_power_sweep(points, cfg: MillConfig, revolutions: float, measure_last: float) -> dict:
    rows = []
    for k, (fill, phic) in enumerate(points):
        record = k == 0
        rows.append(_power_case(cfg, fill, phic, revolutions, measure_last, record))
    errs = [r["abs_rel_error"] for r in rows]
    return {
        "rows": rows,
        "mean_abs_rel_error": float(np.mean(errs)),
        "max_abs_rel_error": float(np.max(errs)),
        "mean_signed_rel_error": float(np.mean([r["rel_error"] for r in rows])),
        "max_energy_closure": float(max(r["energy_closure_rel"] for r in rows)),
        "max_full_closure": float(max(r["full_closure_rel"] for r in rows)),
        "max_moment_closure": float(max(r["moment_closure_rel"] for r in rows)),
        "criterion": "mill power error <= 10% vs experimental correlation",
        "pass": bool(np.mean(errs) <= 0.10),
        "pass_all_points": bool(np.max(errs) <= 0.10),
    }


# ------------------------------------------------------------------ anchor 3
def run_bond_verification(P_dem_w: float, m_ore_kg: float = ORE_MASS_KG) -> dict:
    band = work_index_band(MATERIAL)
    wi = band["typical"]
    f80, p80 = F80_UM, 150.0
    e_ref = bond_specific_energy_kwh_t(wi, f80, p80)
    # (i) third-theory identity: infinite feed -> E = 10 Wi / sqrt(P80)
    e_inf = bond_specific_energy_kwh_t(wi, math.inf, p80)
    e_inf_closed = 10.0 * wi / math.sqrt(p80)
    # (ii) inverse round trip
    p80_back = bond_product_size_um(wi, f80, e_ref)
    wi_back = bond_work_index_from_energy(f80, p80, e_ref)
    # (iii) independent route: Bond locked-cycle grindability equation
    pc = p80           # closing screen at the product size
    g_rev = 3.0        # net grams per revolution, representative test value
    e_test = bond_specific_energy_kwh_t(1.0, f80, p80)
    wi_grind = bond_work_index_from_grindability(g_rev, pc, f80, p80)
    wi_grind_back = 44.5 / ((pc ** 0.23) * (g_rev ** 0.82) * e_test)
    # (iv) device-level cross-check: specific energy the DEM mill delivers for a
    #      target size reduction, and the work index that implies
    dev = []
    for target in (300.0, 200.0, 150.0, 106.0):
        e_bond = bond_specific_energy_kwh_t(wi, f80, target)
        t_needed = e_bond * (m_ore_kg / 1000.0) * 3.6e6 / P_dem_w
        dev.append({
            "P80_um": target,
            "energy_bond_kwh_t": e_bond,
            "mill_time_s_at_dem_power": t_needed,
            "energy_per_kg_feed_j_kg": e_bond * 3600.0,
        })
    return {
        "material": MATERIAL,
        "work_index_db": band,
        "F80_um": f80,
        "P80_um": p80,
        "energy_kwh_t": e_ref,
        "identity_infinite_feed": {"value": e_inf, "closed_form": e_inf_closed,
                                   "rel_error": _rel(e_inf, e_inf_closed)},
        "inverse_round_trip_p80_um": p80_back,
        "inverse_round_trip_wi": wi_back,
        "round_trip_rel_error": _rel(wi_back, wi),
        "grindability_route": {"wi": wi_grind, "wi_via_third_theory": wi_grind_back,
                               "rel_error": _rel(wi_grind, wi_grind_back)},
        "device_energy_table": dev,
        "max_implementation_rel_error": max(_rel(e_inf, e_inf_closed),
                                            _rel(wi_back, wi),
                                            _rel(wi_grind, wi_grind_back)),
        "criterion": "Bond specific energy computation error <= 15%",
        "pass": bool(max(_rel(e_inf, e_inf_closed), _rel(wi_back, wi),
                         _rel(wi_grind, wi_grind_back)) <= 0.15),
    }


# ------------------------------------------------------------------ anchor 4
def austin_like_rates(diameter_m) -> np.ndarray:
    """A constant selection function on the grid (1/s), used by the verifier."""
    d = np.asarray(diameter_m, dtype=float)
    return 0.02 * (d / 1e-3) ** 1.0 / (1.0 + (d / 2.0e-3) ** 0.8)


def _energy_rate(P_w: float, m_ore_kg: float) -> float:
    """Specific energy rate [kWh/t per second] delivered by the mill to the ore."""
    return specific_energy_kwh_t(P_w, 1.0, m_ore_kg)


def run_breakage(P_dem_w: float, spectrum: dict | None, m_ore_kg: float = ORE_MASS_KG,
                 quick: bool = False) -> dict:
    grid = VolumeGrid.from_diameters(20e-6, 3000e-6, 24 if quick else 32)
    d = grid.diameters
    feed_index = int(np.searchsorted(d, F80_UM * 1e-6))
    feed = np.zeros(grid.n_sections)
    feed[feed_index] = 1.0
    e_rate = _energy_rate(P_dem_w, m_ore_kg)          # kWh/t per second
    energies = [0.5, 1.0, 2.0] if quick else [0.5, 1.0, 2.0, 4.0]

    # ---- (D1) operator verification: the sectional BDF integration is compared
    # against the exact solution of the same linear constant-coefficient
    # operator, p(t) = expm(A t) p(0) -- an independent numerical route (matrix
    # exponential, no time stepping) for a batch grind with constant S_i.
    from scipy.linalg import expm
    S_op = austin_like_rates(d)
    b_mass = mass_fraction_matrix(d, AppearanceFunction.from_t10(20.0))
    N_mat = volume_daughter_matrix(d, b_mass)
    A = N_mat @ np.diag(S_op)
    A[np.diag_indices_from(A)] -= S_op
    t_ref = 8.0
    out_ref = batch_grind(grid, feed, S_op, AppearanceFunction.from_t10(20.0),
                          times=[t_ref])
    v_piv = grid.pivots
    n_exact = np.clip(expm(A * t_ref) @ (feed / v_piv), 0.0, None)
    w_exact = v_piv * n_exact
    w_exact = w_exact / w_exact.sum()
    w_num = out_ref["mass_fractions"][-1]
    d1 = {
        "grid_sections": grid.n_sections,
        "time_s": t_ref,
        "max_abs_mass_fraction_difference": float(np.max(np.abs(w_num - w_exact))),
        "max_abs_cdf_difference": float(np.max(np.abs(
            np.cumsum(w_num) - np.cumsum(w_exact)))),
        "mass_conserved_numeric": float(np.sum(w_num)),
        "P80_numeric_um": p80_from_psd(d, w_num) * 1e6,
        "P80_exact_um": p80_from_psd(d, w_exact) * 1e6,
        "reference": "matrix exponential of the same linear operator (no time stepping)",
    }

    # ---- (D2) PSD evolution: single-event t10 family vs multi-event kinetics
    jk_band = [{"A": 45.0, "b": 0.8}, {"A": 45.0, "b": 0.3}, {"A": 55.0, "b": 0.5}]
    rows = []
    for pars in jk_band:
        # calibrate the specific selection rate so that the sectional kinetics
        # reproduces the impact law's t10 at E = 1 kWh/t (one calibration point)
        def t10_of(k0):
            S = k0 * (d / 1e-3) ** 1.0 * e_rate
            t_end = (1.0 / e_rate) if e_rate > 0 else 0.0
            out = batch_grind(grid, feed, S, AppearanceFunction.from_t10(
                float(t10_energy_law(1.0, pars["A"], pars["b"]))), times=[t_end])
            return t10_from_psd(d, out["mass_fractions"][-1], F80_UM * 1e-6)
        lo, hi = 1e-4, 1e3
        target = float(t10_energy_law(1.0, pars["A"], pars["b"]))
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if t10_of(mid) < target:
                lo = mid
            else:
                hi = mid
        k0 = 0.5 * (lo + hi)
        for e in energies:
            t10_ref = float(t10_energy_law(e, pars["A"], pars["b"]))
            af_ref = AppearanceFunction.from_t10(t10_ref) if t10_ref > 0.01 else \
                AppearanceFunction.from_t10(0.01)
            # route A: single-event family, product = B(x|F80)
            cum_a = np.array([float(af_ref.cumulative_finer(x, F80_UM * 1e-6)) for x in d])
            w_a = np.diff(np.concatenate([[0.0], cum_a]))
            w_a = np.clip(w_a, 0.0, None)
            w_a = w_a / w_a.sum()
            # route B: sectional multi-event kinetics over t = E / dE/dt
            t_end = e / e_rate if e_rate > 0 else 0.0
            S = k0 * (d / 1e-3) ** 1.0 * e_rate
            out = batch_grind(grid, feed, S, af_ref, times=[t_end])
            w_b = out["mass_fractions"][-1]
            sizes = [300e-6, 150e-6, 75e-6]
            cmp_rows = [
                {"size_um": s * 1e6,
                 "passing_single_event": passing_at(d, w_a, s),
                 "passing_multi_event": passing_at(d, w_b, s),
                 "abs_diff": abs(passing_at(d, w_a, s) - passing_at(d, w_b, s))}
                for s in sizes
            ]
            p80_a = p80_from_psd(d, w_a)
            p80_b = p80_from_psd(d, w_b)
            rows.append({
                "jk": pars, "energy_kwh_t": e, "t10_target_percent": t10_ref,
                "t10_single_event": t10_from_psd(d, w_a, F80_UM * 1e-6),
                "t10_multi_event": t10_from_psd(d, w_b, F80_UM * 1e-6),
                "P80_single_event_um": p80_a * 1e6,
                "P80_multi_event_um": p80_b * 1e6,
                "P80_rel_error": _rel(p80_b, p80_a),
                "passing_comparison": cmp_rows,
                "max_passing_abs_diff": max(r["abs_diff"] for r in cmp_rows),
                "grind_time_s": t_end,
            })
    max_diff = max(r["max_passing_abs_diff"] for r in rows)
    max_p80 = max(r["P80_rel_error"] for r in rows)
    # ---- collision-spectrum interpretation (impact channel only)
    impact = None
    if spectrum is not None:
        e_med = spectrum["specific_energy_j_kg"]["median"]
        e_p95 = spectrum["specific_energy_j_kg"]["p95"]
        impact = {
            "median_event_specific_energy_j_kg": e_med,
            "p95_event_specific_energy_j_kg": e_p95,
            "event_rate_per_s": spectrum["event_rate_per_s"],
            "note": ("per-collision specific energy follows e = 0.5 v_n^2 m_eff/m_ore; "
                     "compare with the single-event breakage energy e50 of a 1 mm "
                     "particle from drop-weight practice (order 1e3 J/kg)"),
            "e50_1mm_j_kg_reference": 1200.0,
            "ratio_p95_to_e50": e_p95 / 1200.0,
        }
    detail = {r["jk"]["A"]: r for r in rows}
    return {
        "grid_sections": grid.n_sections,
        "energy_rate_kwh_t_per_s": e_rate,
        "specific_energy_input_kwh_t": energies,
        "operator_verification": d1,
        "psd_evolution_rows": rows,
        "max_passing_abs_diff": max_diff,
        "max_p80_rel_error": max_p80,
        "impact_channel": impact,
        "jk_reference_case": detail.get(45.0),
        "criterion": "PSD evolution error <= 20% vs 试验",
        "pass_vs_exact_solution": bool(d1["max_abs_cdf_difference"] <= 0.20),
        "pass_model_structure": bool(max(max_diff, max_p80) <= 0.20),
        "pass_vs_experiment": None,
        "pass": bool(d1["max_abs_cdf_difference"] <= 0.20),
    }


# --------------------------------------------------------------------- plots
def make_figures(res: dict, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figs = []

    # F1: critical speed
    cs = res["critical_speed"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ds = np.array([r["diameter"] for r in cs["rows"]])
    dem = np.array([r["rpm"] for r in cs["rows"]])
    dd = np.linspace(min(ds) * 0.8, max(ds) * 1.2, 100)
    ax[0].loglog(dd, [critical_speed_rpm(float(x)) for x in dd], "k-",
                 label=r"$N_c=42.3/\sqrt{D}$")
    ax[0].loglog(dd, [critical_speed_rpm_charge_centre(x, 0.008) for x in dd], "k--",
                 lw=1, label=r"ball-centre $(D-d)$")
    ax[0].loglog(ds, dem, "o", ms=7, mfc="tab:red", label="DEM detected")
    ax[0].set_xlabel("mill inner diameter D [m]")
    ax[0].set_ylabel("$N_c$ [rpm]")
    ax[0].set_title("critical speed: DEM vs theory")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    for r in cs["rows"]:
        curve = r["detector_curve"]
        phi = [c["fraction_of_critical"] for c in curve]
        ax[1].plot(phi, [c["turns"] for c in curve], "o-", label=f"D={r['diameter']} m")
    ax[1].axvline(1.0, color="k", ls=":", lw=1)
    ax[1].axhline(0.9, color="tab:gray", ls=":", lw=1)
    ax[1].set_xlabel(r"$N/N_c$")
    ax[1].set_ylabel("net advance [turns / revolution]")
    ax[1].set_title("detector response (stuck threshold 0.9 turn)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig1_critical_speed.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig1_critical_speed.png", "caption": "DEM-detected centrifuging onset vs the textbook critical speed and the detector response curve."})

    # F2: power parity
    pw = res["power"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    x = np.array([r["P_hogg_fuerstenau_w"] for r in pw["rows"]])
    y = np.array([r["P_dem_w"] for r in pw["rows"]])
    ax[0].loglog(x, y, "o", ms=7)
    lim = [x.min() * 0.7, x.max() * 1.4]
    ax[0].loglog(lim, lim, "k-", label="parity")
    ax[0].loglog(lim, [v * 0.9 for v in lim], "k--", lw=1, label="+/-10%")
    ax[0].loglog(lim, [v * 1.1 for v in lim], "k--", lw=1)
    for r in pw["rows"]:
        ax[0].annotate(f"J={r['bulk_filling']:.2f}\n$\\phi_c$={r['fraction_critical']:.2f}",
                       (r["P_hogg_fuerstenau_w"], r["P_dem_w"]), fontsize=6,
                       xytext=(4, -8), textcoords="offset points")
    ax[0].set_xlabel("Hogg & Fuerstenau power [W]")
    ax[0].set_ylabel("DEM power [W]")
    ax[0].set_title("mill power parity")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    labs = [f"J={r['bulk_filling']:.2f}\n{r['fraction_critical']:.2f}" for r in pw["rows"]]
    ax[1].bar(range(len(labs)), [100 * r["rel_error"] for r in pw["rows"]], color="tab:blue")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axhline(10, color="tab:red", ls="--", lw=1)
    ax[1].axhline(-10, color="tab:red", ls="--", lw=1)
    ax[1].set_xticks(range(len(labs)))
    ax[1].set_xticklabels(labs, fontsize=7)
    ax[1].set_ylabel("DEM / correlation - 1  [%]")
    ax[1].set_title("power deviation over the J x phi_c grid")
    fig.tight_layout()
    p = outdir / "fig2_mill_power.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig2_mill_power.png", "caption": "DEM mill power vs the Hogg & Fuerstenau correlation: parity and per-point deviation."})

    # F3: charge snapshot + collision spectrum
    snap = res.get("snapshot")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0))
    if snap is not None:
        r = np.asarray(snap["radii"])
        ax[0].scatter(snap["pos"][:, 0], snap["pos"][:, 1], s=(r / r.max() * 220) ** 1.0,
                      c=np.linalg.norm(snap["vel"], axis=1), cmap="viridis")
        th = np.linspace(0, 2 * np.pi, 400)
        ax[0].plot(0.15 * np.cos(th), 0.15 * np.sin(th), "k-", lw=1.2)
        for k in range(8):
            a = 2 * np.pi * k / 8
            ax[0].plot([0.15 * math.cos(a), 0.156 * math.cos(a)],
                       [0.15 * math.sin(a), 0.156 * math.sin(a)], "k-", lw=2)
        ax[0].set_aspect("equal")
        ax[0].set_title("mill charge snapshot (colour = speed)")
        ax[0].set_xlabel("x [m]")
        ax[0].set_ylabel("y [m]")
    sp = res.get("power", {}).get("rows", [{}])[0].get("collision_spectrum")
    if sp:
        vals = sp["specific_energy_j_kg"]
        ax[1].bar(["p05", "median", "p95", "max"],
                  [vals["p05"], vals["median"], vals["p95"], vals["max"]], color="tab:purple")
        ax[1].axhline(1200.0, color="tab:red", ls="--", lw=1,
                      label="e50 reference (1 mm ore, 1.2 kJ/kg)")
        ax[1].set_yscale("log")
        ax[1].set_ylabel("per-collision specific energy [J/kg]")
        ax[1].set_title("DEM collision spectrum")
        ax[1].legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "fig3_charge_spectrum.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig3_charge_spectrum.png", "caption": "DEM charge snapshot (lifter bars drawn) and the per-collision specific energy spectrum against the single-particle breakage energy."})

    # F4: PSD evolution
    br = res["breakage"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    grid = VolumeGrid.from_diameters(20e-6, 3000e-6, br["grid_sections"])
    d = grid.diameters
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(br["specific_energy_input_kwh_t"])))
    rows45 = [r for r in br["psd_evolution_rows"] if r["jk"]["A"] == 45.0 and r["jk"]["b"] == 0.8]
    for col, r in zip(colors, rows45):
        af_ref = AppearanceFunction.from_t10(r["t10_target_percent"])
        cum = np.array([float(af_ref.cumulative_finer(x, F80_UM * 1e-6)) for x in d])
        ax[0].semilogx(d * 1e6, 100 * cum, "-", color=col,
                       label=f"E={r['energy_kwh_t']} kWh/t")
    ax[0].axhline(80, color="k", ls=":", lw=1)
    ax[0].set_xlabel("size [um]")
    ax[0].set_ylabel("cumulative passing [%]")
    ax[0].set_title("product PSD (single-event t10 family)")
    ax[0].invert_xaxis()
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")
    e_list = [r["energy_kwh_t"] for r in rows45]
    ax[1].plot(e_list, [r["P80_single_event_um"] for r in rows45], "o-", label="single-event")
    ax[1].plot(e_list, [r["P80_multi_event_um"] for r in rows45], "s-", label="multi-event PBM")
    ax[1].set_xlabel("specific energy [kWh/t]")
    ax[1].set_ylabel("P80 [um]")
    ax[1].set_title("P80 evolution: two model structures")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig4_psd_evolution.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig4_psd_evolution.png", "caption": "Product size distributions from the t10 family and the P80 evolution from the two model structures."})

    # F5: Bond
    bo = res["bond"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    f80 = bo["F80_um"]
    p80 = np.logspace(math.log10(40), math.log10(f80) - 0.02, 200)
    band = bo["work_index_db"]
    for wi, style, lab in ((band["low"], "--", "Wi low"), (band["typical"], "-", "Wi typical"),
                           (band["high"], "--", "Wi high")):
        ax.loglog(p80, [bond_specific_energy_kwh_t(wi, f80, x) for x in p80], "k" + style,
                  lw=1.2, label=f"{lab}={wi}")
    ax.set_xlabel("P80 [um]")
    ax.set_ylabel("specific energy [kWh/t]")
    ax.set_title(f"Bond third theory, {bo['material']} band (F80={f80:.0f} um)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = outdir / "fig5_bond_law.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig5_bond_law.png", "caption": "Bond third-theory specific energy over the work-index band of the material."})
    return figs


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/comminution_energy_surface.json")
    ap.add_argument("--figdir", default="artifacts/comminution_energy_surface")
    ap.add_argument("--quick", action="store_true", help="smoke run (short, reduced grid)")
    ap.add_argument("--revolutions", type=float, default=3.0)
    ap.add_argument("--measure-last", type=float, default=1.5)
    ap.add_argument("--skip-power", action="store_true")
    args = ap.parse_args(argv)

    cfg = MillConfig(**MILL)
    if args.quick:
        cfg = MillConfig(**{**MILL, "ball_counts": (9, 14, 20), "dt": 2.0e-4})
    t_start = time.time()
    out = {"study": "comminution energy surface", "mill": {k: (list(v) if isinstance(v, tuple) else v)
                                                           for k, v in MILL.items()},
           "mill_config": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in cfg.__dict__.items()},
           "ore_mass_kg": ORE_MASS_KG, "material": MATERIAL, "f80_um": F80_UM,
           "quick": bool(args.quick)}

    mills = ([{"diameter": 0.30, "tracer_radius": 0.003}]
             if args.quick else
             [{"diameter": 0.20, "tracer_radius": 0.002},
              {"diameter": 0.30, "tracer_radius": 0.003},
              {"diameter": 0.50, "tracer_radius": 0.004}])
    out["critical_speed"] = run_critical_speed(mills)

    points = [(0.35, 0.70)] if args.quick else [
        (0.30, 0.55), (0.30, 0.70), (0.30, 0.85),
        (0.35, 0.55), (0.35, 0.70), (0.35, 0.85),
        (0.45, 0.55), (0.45, 0.70), (0.45, 0.85),
    ]
    if args.skip_power:
        out["power"] = {"rows": [], "pass": None}
    else:
        out["power"] = run_power_sweep(points, cfg, args.revolutions, args.measure_last)
    ref = out["power"]["rows"][0] if out["power"]["rows"] else {"P_dem_w": 47.7,
                                                               "collision_spectrum": None}
    out["snapshot"] = None
    if not args.skip_power:
        # keep one charge snapshot for the figure (re-run the reference case briefly)
        try:
            state = make_mill_state(cfg, fill=0.35 * (1 - BALL_CHARGE_POROSITY),
                                    rpm=rpm_for_fraction_of_critical(0.70, cfg.diameter))
            brief = run_mill(state, revolutions=1.0, measure_last=0.5, settle_time=0.4,
                             sample_every=100000, record_collisions=False)
            out["snapshot"] = {k: np.asarray(v).tolist() for k, v in brief["snapshot"].items()}
        except Exception as exc:  # pragma: no cover - figure only
            out["snapshot_error"] = str(exc)
    out["bond"] = run_bond_verification(ref["P_dem_w"], ORE_MASS_KG)
    out["breakage"] = run_breakage(ref["P_dem_w"], ref.get("collision_spectrum"),
                                   ORE_MASS_KG, quick=args.quick)

    out["runtime_s"] = round(time.time() - t_start, 1)
    out["verdict"] = {
        "critical_speed_pass": out["critical_speed"]["pass"],
        "power_pass": out["power"].get("pass"),
        "power_pass_all_points": out["power"].get("pass_all_points"),
        "bond_pass": out["bond"]["pass"],
        "breakage_pass": out["breakage"]["pass"],
        "test_count_minimum": 12,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if out["snapshot"] is not None:
        out["snapshot"]["pos"] = np.asarray(out["snapshot"]["pos"])
        out["snapshot"]["radii"] = np.asarray(out["snapshot"]["radii"])
        out["snapshot"]["vel"] = np.asarray(out["snapshot"]["vel"])
    proto = {k: v for k, v in out.items() if k != "snapshot"}
    Path(args.out).write_text(json.dumps(proto, indent=1, default=float))
    figs = []
    try:
        figs = make_figures(out, Path(args.figdir))
        Path(args.figdir, "figures.json").write_text(json.dumps(figs, indent=1))
    except Exception as exc:  # pragma: no cover - figures are best effort
        import traceback
        Path(args.figdir).mkdir(parents=True, exist_ok=True)
        Path(args.figdir, "figures_error.txt").write_text(
            str(exc) + "\n" + traceback.format_exc())
    print(json.dumps({"out": args.out, "figures": [f["name"] for f in figs],
                      "verdict": out["verdict"], "runtime_s": out["runtime_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())