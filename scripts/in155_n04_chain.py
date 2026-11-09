"""IN155-N04: DEM contact-force history -> rainflow -> S-N -> Miner chain.

Runs the whole node in one pass and writes ``artifacts/in155_n04/``:

1. the F(t) export contract on the three verification cases (single impact,
   two-ball collision, reduced-scale floe field vs bow), with the energy budget
   closure study;
2. the m5405 statistics-fidelity check of the export (round trip + comparison
   with the tabulated m5405 numbers);
3. the analytical anchors (constant amplitude Miner, single-pulse Duhamel);
4. the sampling-compression error envelope (1/5/20/100 Hz x naive/peak) on the
   m5405 record and on the floe-field record;
5. the statistics-compression error (RMS-Rayleigh, peak block);
6. the minimum-sampling-rate law ``fs_min = a m^b / t_p^c``.

Usage::

    python scripts/in155_n04_chain.py            # full node
    python scripts/in155_n04_chain.py --quick    # smaller DEM case, no figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.fatigue_chain import (  # noqa: E402
    ModalChannel,
    chain_damage,
    damage_from_stress,
    sampling_envelope,
    statistics_compression,
)
from tensordem.fatigue_chain.damage import PowerLawSNCurve, stress_history  # noqa: E402
from tensordem.fatigue_chain.envelope import random_pulse_train  # noqa: E402
from tensordem.fatigue_chain import cases as fc_cases  # noqa: E402
from tensordem.fatigue_chain.contract import (  # noqa: E402
    HULL,
    ContactForceHistory,
)
from tensordem.fatigue_chain.envelope import fit_min_rate_law  # noqa: E402
from tensordem.fatigue_chain.resample import compress, reconstruct  # noqa: E402
from tensordem.fatigue_chain.response import (  # noqa: E402
    ModalChannel as MC,
    half_sine_closed_form,
    sdof_response,
)
from tensordem.whipping import load_calibration as lc  # noqa: E402
from tensordem.whipping.fatigue import DNV_CURVES  # noqa: E402
from tensordem.whipping.rainflow import rainflow  # noqa: E402

OUT = ROOT / "artifacts" / "in155_n04"
FIGS = OUT / "figures"

GAIN_PA_PER_N = 6.2          # 6.2 MPa per MN, docs/ice_load_statistics_calibration.md
MODE_SPLIT = ((0.69, 0.015, 0.62), (1.83, 0.015, 0.19),
              (3.57, 0.015, 0.12), (5.85, 0.015, 0.07))
M5405_CONTROLS = dict(crushing_scale=1.681, contact_damping=0.173,
                      floe_geometry=1.092, froude=0.0179)
M5405_DOC = {"peak_N": 15.0e6, "mean_N": 5.94e6, "R_mean_over_peak": 0.396,
             "rainflow_cycles": 1953, "stress_std_MPa": 18.5, "stress_peak_MPa": 93.0}


def channels():
    return [ModalChannel(f, z, GAIN_PA_PER_N * w) for f, z, w in MODE_SPLIT]


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serialisable: {type(obj)}")


def dump(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------------------
# 1. contract cases
# ---------------------------------------------------------------------------

def run_contract_cases(quick: bool) -> dict:
    print("[1] contract cases")
    out = {}
    t0 = time.time()
    c1 = fc_cases.single_ball_impact()
    r1 = c1.history.validate()
    ode = fc_cases.integrate_impact_ode(mass=0.25, velocity=0.5, stiffness=2.0e5,
                                       damping_ratio=0.05)
    out["single_ball_impact"] = {
        "contract": r1,
        "analytic": c1.analytic,
        "ode": {k: ode[k] for k in ("contact_duration_s", "peak_force_N", "restitution",
                                    "analytic_restitution", "analytic_contact_duration_s")},
        "errors": {
            "peak_force_rel": abs(c1.analytic["peak_force_N"] - ode["peak_force_N"]) / ode["peak_force_N"],
            "contact_duration_rel": abs(c1.analytic["contact_duration_s"] - ode["contact_duration_s"]) / ode["contact_duration_s"],
            "restitution_rel": abs(c1.analytic["restitution"] - ode["restitution"]) / ode["restitution"],
        },
    }
    c1.history.save(OUT / "contact_force_single_ball_impact.json")

    c2 = fc_cases.two_ball_collision()
    r2 = c2.history.validate()
    out["two_ball_collision"] = {
        "contract": r2,
        "analytic": c2.analytic,
        "errors": {
            "momentum_rel": abs(c2.analytic["momentum_final_kg_m_s"]
                                - c2.analytic["momentum_initial_kg_m_s"])
            / abs(c2.analytic["momentum_initial_kg_m_s"]),
        },
    }
    c2.history.save(OUT / "contact_force_two_ball_collision.json")

    kwargs = dict(duration_s=0.45 if quick else 1.5, dt_s=4.0e-5,
                  n_floes=26 if quick else 45)
    c3 = fc_cases.floe_field_bow(**kwargs)
    r3 = c3.history.validate(tol_energy=0.10)
    chan_t, chan_f = c3.history.to_uniform(1000.0)
    (OUT / "floe_field_channel.csv").write_text(
        "t_s,F_bow_N\n" + "\n".join(f"{tt:.6f},{ff:.6f}" for tt, ff in zip(chan_t, chan_f)),
        encoding="utf-8")
    durs = np.array([(c3.history.contact_series(cid)["t"][-1]
                      - c3.history.contact_series(cid)["t"][0]) for cid in c3.history.contact_ids()])
    out["floe_field_bow"] = {
        "contract": r3, "analytic": c3.analytic,
        "channel_statistics": c3.history.statistics(1000.0),
        "contact_duration_ms": {"median": float(np.median(durs) * 1e3),
                                "p10": float(np.percentile(durs, 10) * 1e3),
                                "p90": float(np.percentile(durs, 90) * 1e3)},
        "n_samples": int(c3.history.n_samples),
    }

    # energy budget closure vs record interval (first-order convergence)
    study = []
    for dt in (8.0e-5, 4.0e-5, 2.0e-5):
        run = fc_cases.floe_field_bow(duration_s=0.30, dt_s=dt,
                                      n_floes=26 if quick else 30)
        rep = run.history.validate(tol_energy=0.5)
        study.append({"dt_s": dt, "rate_hz": 1.0 / dt,
                      "budget_rel_error": rep["energy"]["max_rel_budget_error"],
                      "state_function_rel_error": rep["energy"]["max_rel_state_function_error"],
                      "n_samples": run.history.n_samples})
        print(f"    dt={dt:g} s -> budget residual {study[-1]['budget_rel_error']:.3e}")
    out["energy_budget_resolution_study"] = study
    out["elapsed_s"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# 2. m5405 fidelity of the export
# ---------------------------------------------------------------------------

def run_m5405_fidelity() -> dict:
    print("[2] m5405 statistics fidelity")
    controls = lc.IceLoadControls(**M5405_CONTROLS)
    t, f, meta = lc.synthesise(controls)
    history = ContactForceHistory(
        t=t, contact_id=np.zeros(t.size, dtype=int),
        floe_i=np.zeros(t.size, dtype=int), floe_j=np.full(t.size, HULL),
        F_n=f, F_t=np.zeros(t.size), patch_area=f / 1.5e6,
        meta={"case": "m5405_calibrated_bow_load", "sampling_rate_hz": 1000.0,
              "mode": "statistics_only", "title": "IN29-N03 calibrated record"},
    )
    report = history.validate()
    # the full 60 000-sample table is 9.5 MB; the contract round trip is tested
    # on the full table in memory, the artifact keeps a 3 000-sample preview
    n_prev = 3000
    preview = ContactForceHistory(
        t=t[:n_prev], contact_id=np.zeros(n_prev, dtype=int),
        floe_i=np.zeros(n_prev, dtype=int), floe_j=np.full(n_prev, HULL),
        F_n=f[:n_prev], F_t=np.zeros(n_prev), patch_area=f[:n_prev] / 1.5e6,
        meta={**history.meta, "preview_samples": n_prev},
    )
    preview.save(OUT / "contact_force_m5405_preview.json")
    reloaded = ContactForceHistory.from_dict(history.to_dict())
    stats_raw = history.statistics(1000.0)
    stats_rt = reloaded.statistics(1000.0)
    rt_errors = {key: abs(stats_rt[key] - stats_raw[key]) / max(abs(stats_raw[key]), 1e-30)
                 for key in ("peak_N", "rms_N", "p95_N", "mean_N")}
    doc_errors = {
        "peak_N": abs(stats_raw["peak_N"] - M5405_DOC["peak_N"]) / M5405_DOC["peak_N"],
        "mean_N": abs(stats_raw["mean_N"] - M5405_DOC["mean_N"]) / M5405_DOC["mean_N"],
        "R_mean_over_peak": abs(stats_raw["mean_N"] / stats_raw["peak_N"]
                                - M5405_DOC["R_mean_over_peak"]) / M5405_DOC["R_mean_over_peak"],
    }
    return {"contract": report, "statistics": stats_raw, "round_trip_error": rt_errors,
            "m5405_doc_error": doc_errors, "m5405_doc": M5405_DOC,
            "calibration_meta": meta.get("controls", {})}


# ---------------------------------------------------------------------------
# 3. analytical anchors
# ---------------------------------------------------------------------------

def run_anchors() -> dict:
    print("[3] analytical anchors")
    curve = DNV_CURVES["D_air"]
    out = {}

    # (a) constant amplitude: D = n / N(dS) exactly.  The stress record is
    # sampled at its turning points (4 samples/period) so the range is exactly
    # 2A in floating point; no modal filter is involved (that half of the chain
    # is anchored in (b)).
    rows = []
    for amp_MPa, n_periods in ((60.0, 40), (25.0, 40), (150.0, 25), (90.0, 12)):
        T = 1.0 / 2.0
        t = np.arange(0.0, n_periods * T + T / 4.0, T / 4.0)
        sigma = amp_MPa * 1e6 * np.cos(2.0 * math.pi * t / T)
        chain = damage_from_stress(sigma, curve)
        analytic = n_periods / curve.cycles_to_failure(np.array([2.0 * amp_MPa * 1e6]))[0]
        rows.append({"amplitude_MPa": amp_MPa, "periods": n_periods,
                     "damage_chain": chain.damage, "damage_analytic": float(analytic),
                     "rel_error": abs(chain.damage - analytic) / analytic,
                     "rainflow_cycles": chain.n_cycles,
                     "range_Pa": chain.max_range_Pa})
    out["constant_amplitude"] = rows

    # (b) single pulse: exact SDOF recurrence vs the undamped Duhamel closed form
    F0, td, fn = 1.0e6, 0.05, 2.0
    t = np.arange(0.0, 0.30, 1.0 / 4000.0)
    force = np.where(t <= td, F0 * np.sin(math.pi * t / td), 0.0)
    q_num = sdof_response(t, force, fn, 0.0)
    q_cf = half_sine_closed_form(t, F0, td, fn)
    q_damped = sdof_response(t, force, fn, 0.05)
    q_damp_cf = half_sine_closed_form(t, F0, td, fn, 0.0)
    out["single_pulse"] = {
        "peak_numeric_N": float(q_num.max()),
        "peak_closed_form_N": float(q_cf.max()),
        "peak_rel_error": float(abs(q_num.max() - q_cf.max()) / q_cf.max()),
        "max_abs_dev_rel_peak": float(np.max(np.abs(q_num - q_cf)) / q_cf.max()),
        "damped_peak_N": float(q_damped.max()),
        "damped_vs_undamped_rel": float(abs(q_damped.max() - q_damp_cf.max()) / q_damp_cf.max()),
    }

    # (c) ASTM E1049-85 example + three-level block (item by item)
    def cycle_list(signal):
        return sorted([(round(c.range, 10), round(c.count, 10), round(c.mean, 10))
                       for c in rainflow(np.asarray(signal, dtype=float))])

    astm = [-2, 1, -3, 5, -1, 3, -4, 4, -2]
    block = [0, 3, -1, 5, -3, 2, 0]
    out["astm_e1049_85"] = {
        "sequence": astm,
        "cycles": cycle_list(astm),
        "three_level_block": block,
        "three_level_block_cycles": cycle_list(block),
        "three_level_block_x2_cycles": cycle_list(block + block),
    }
    return out


# ---------------------------------------------------------------------------
# 4/5. compression envelopes and statistics compression
# ---------------------------------------------------------------------------

def _floe_field_full_scale(quick: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    kwargs = dict(duration_s=0.45 if quick else 1.5, dt_s=4.0e-5,
                  n_floes=26 if quick else 45)
    case = fc_cases.floe_field_bow(**kwargs)
    lam = case.history.meta["config"]["lambda_L"]
    lam_t = math.sqrt(lam)
    lam_f = lam ** 3
    t, f = case.history.to_uniform(1000.0)
    # Froude scale: model -> full scale
    t_full = t * lam_t
    f_full = f * lam_f
    grid = np.arange(0.0, t_full[-1], 1.0 / 1000.0)
    f_grid = np.interp(grid, t_full, f_full)
    return grid, f_grid, {"lambda_L": lam, "lambda_t": lam_t, "lambda_F": lam_f,
                          "peak_model_N": float(f.max()),
                          "peak_full_scale_N": float(f_grid.max())}


def run_envelopes(quick: bool) -> dict:
    print("[4] sampling envelopes + statistics compression")
    curve = DNV_CURVES["D_air"]
    ch = channels()
    out = {}

    controls = lc.IceLoadControls(**M5405_CONTROLS)
    t, f, _ = lc.synthesise(controls)
    env = sampling_envelope(t, f, ch, curve, label="m5405_bow_load",
                            methods=("naive", "raw", "peak"))
    out["m5405"] = env.as_dict()
    out["m5405"]["statistics_compression"] = statistics_compression(curve, t, f, ch)

    tg, fg, scale_meta = _floe_field_full_scale(quick)
    # calibrate the DEM-derived record to the same hot-spot peak as the hull
    # reference (93 MPa at the 15 MN bow load), so that the two instances
    # differ in load *shape* only, not in S-N branch occupancy
    probe = stress_history(tg, fg, ch)
    factor = (M5405_DOC["stress_peak_MPa"] * 1e6) / float(np.max(np.abs(probe)))
    ch3 = [ModalChannel(c.freq_hz, c.damping, c.static_gain_Pa_per_N * factor) for c in ch]
    scale_meta["stress_normalisation_factor"] = factor
    scale_meta["peak_stress_Pa"] = M5405_DOC["stress_peak_MPa"] * 1e6
    env3 = sampling_envelope(tg, fg, ch3, curve, label="floe_field_bow_full_scale",
                             methods=("naive", "raw", "peak"))
    out["floe_field"] = env3.as_dict()
    out["floe_field"]["statistics_compression"] = statistics_compression(curve, tg, fg, ch3)
    out["floe_field"]["froude_scaling"] = scale_meta
    return out


# ---------------------------------------------------------------------------
# 6. minimum sampling rate law
# ---------------------------------------------------------------------------

def _first_crossing(points, threshold: float):
    """Smallest rate at which |dD/D| <= threshold (log-log interpolation)."""
    points = sorted(points)
    for i, (rate, err) in enumerate(points):
        if abs(err) <= threshold:
            if i == 0:
                return rate
            r0, e0 = points[i - 1]
            e0, e1 = abs(e0), abs(err)
            if e0 <= e1 or e0 <= 0.0:
                return rate
            frac = (math.log(e0) - math.log(threshold)) / (math.log(e0) - math.log(e1))
            return float(math.exp(math.log(r0) + frac * (math.log(rate) - math.log(r0))))
    return None


def run_min_rate_law(quick: bool) -> dict:
    print("[6] minimum sampling rate law")
    pulse_widths = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32] if not quick else [0.02, 0.08]
    slopes = (3, 5)
    n_pulses = 40
    samples = []
    rows_out = []
    for m in slopes:
        curve = PowerLawSNCurve(m)
        for tp in pulse_widths:
            period = 6.0 * tp
            duration = n_pulses * period
            base_rate = 20.0 / tp
            t, f = random_pulse_train(duration, base_rate, tp, mean_interval_s=period,
                                      peak=1.0e6, seed=17)
            ch = [MC(0.35 / tp, 0.015, GAIN_PA_PER_N)]
            base = chain_damage(t, f, ch, curve)
            rates = np.geomspace(0.5 / tp, 30.0 / tp, 18)
            row = {"m": m, "pulse_width_s": tp, "period_s": period,
                   "base_rate_hz": base_rate, "baseline_damage": base.damage,
                   "mode_hz": 0.35 / tp, "scan": []}
            for method in ("naive", "peak"):
                best = None
                points = []
                for rate in rates:
                    tc, fc = compress(t, f, float(rate), method=method)
                    fr = reconstruct(t, tc, fc)
                    d = chain_damage(t, fr, ch, curve).damage
                    err = float(d / base.damage - 1.0)
                    points.append((float(rate), err))
                    row["scan"].append({"method": method, "rate_hz": float(rate),
                                        "dD_over_D": err})
                # first crossing of the 10 % band, interpolated in log-log
                best = _first_crossing(points, 0.10)
                row[f"fs_min_{method}_hz"] = best
                if best is not None:
                    samples.append({"m": float(m), "pulse_width_s": tp, "fs_min_hz": best,
                                    "method": method})
            print(f"    m={m} t_p={tp*1e3:.0f} ms: fs_min naive={row['fs_min_naive_hz']}, "
                  f"peak={row['fs_min_peak_hz']}")
            rows_out.append(row)
    fits = {}
    for method in ("naive", "peak"):
        pts = [s for s in samples if "fs_min_hz" in s and s.get("method") == method]
        if len(pts) >= 4:
            fits[method] = fit_min_rate_law(pts)
    return {"pulse_widths_s": pulse_widths,
            "rows": [{k: v for k, v in r.items() if k != "scan"}
                     | {"scan_kept": len(r["scan"])} for r in rows_out],
            "fits": fits}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)

    # 1. the three contact-force histories
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    c1 = fc_cases.single_ball_impact()
    s = c1.history.contact_series(0)
    ax[0, 0].plot(s["t"] * 1e3, s["F_n"], lw=1.2)
    ax[0, 0].set_title("case 1: single ball impact (Kelvin-Voigt)")
    ax[0, 0].set_xlabel("t [ms]"); ax[0, 0].set_ylabel("$F_n$ [N]")
    c2 = fc_cases.two_ball_collision()
    s2 = c2.history.contact_series(0)
    ax[0, 1].plot(s2["t"] * 1e3, s2["F_n"], lw=1.2, color="C1")
    ax[0, 1].set_title("case 2: two-ball collision"); ax[0, 1].set_xlabel("t [ms]")
    tg, fg, _ = _floe_field_full_scale(False)
    ax[1, 0].plot(tg, fg / 1e3, lw=0.6, color="C2")
    ax[1, 0].set_title("case 3: floe field vs bow (full scale, Froude)")
    ax[1, 0].set_xlabel("t [s]"); ax[1, 0].set_ylabel("$F$ [kN]")
    m = (tg > tg[0]) & (tg < tg[0] + 0.35)
    ax[1, 1].plot(tg[m], fg[m] / 1e3, lw=0.9, color="C3")
    ax[1, 1].set_title("case 3: zoom (0.35 s window)"); ax[1, 1].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(FIGS / "fig1_contact_forces.png", dpi=130); plt.close(fig)

    # 2. energy budget closure
    study = results["contract"]["energy_budget_resolution_study"]
    dts = np.array([s["dt_s"] for s in study])
    err = np.array([s["budget_rel_error"] for s in study])
    st = np.array([s["state_function_rel_error"] for s in study])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(dts, err, "o-", label="energy budget residual")
    ax.loglog(dts, st, "s--", label="spring state-function residual")
    ax.loglog(dts, err[0] * (dts / dts[0]), ":", color="gray", label="slope 1 reference")
    ax.set_xlabel("record interval $\\Delta t$ [s]"); ax.set_ylabel("relative residual")
    ax.set_title("contract energy closure vs record resolution"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig2_energy_budget.png", dpi=130); plt.close(fig)

    # 3. anchors
    F0, td, fn = 1.0e6, 0.05, 2.0
    t = np.arange(0.0, 0.30, 1.0 / 4000.0)
    force = np.where(t <= td, F0 * np.sin(math.pi * t / td), 0.0)
    qn = sdof_response(t, force, fn, 0.0) / F0
    qc = half_sine_closed_form(t, F0, td, fn) / F0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    ax[0].plot(t, qc, lw=2.2, label="Duhamel closed form")
    ax[0].plot(t, qn, "--", lw=1.2, label="exact SDOF recurrence")
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("q / F$_0$")
    ax[0].set_title("single half-sine pulse, $f_n$ = 2 Hz"); ax[0].legend()
    rows = results["anchors"]["constant_amplitude"]
    ax[1].semilogy([r["amplitude_MPa"] for r in rows],
                   [max(r["rel_error"], 1e-18) for r in rows], "o-")
    ax[1].axhline(1e-12, ls=":", color="gray", label="judge a) 1e-12")
    ax[1].set_xlabel("amplitude [MPa]"); ax[1].set_ylabel("|D$_{chain}$ - D$_{analytic}$| / D")
    ax[1].set_title("constant-amplitude Miner anchor"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig3_anchors.png", dpi=130); plt.close(fig)

    # 4. envelopes
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for k, (key, title) in enumerate((("m5405", "m5405 calibrated bow load"),
                                      ("floe_field", "floe-field DEM, full scale"))):
        block = results["envelopes"][key]
        for method, style in (("naive", "o-"), ("peak", "s--")):
            rows = [r for r in block["rows"] if r["method"] == method]
            rows.sort(key=lambda r: r["rate_hz"])
            ax[k].loglog([r["rate_hz"] for r in rows],
                         [max(abs(r["dD_over_D"]), 1e-6) for r in rows], style, label=method)
        ax[k].axhline(0.10, color="gray", ls=":", label="10 % criterion")
        ax[k].set_xlabel("sampling rate [Hz]"); ax[k].set_title(title)
        ax[k].legend()
    ax[0].set_ylabel("|dD/D|")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_envelopes.png", dpi=130); plt.close(fig)

    # 5. min-rate law
    fits = results["min_rate_law"]["fits"]
    rows = [r for r in results["min_rate_law"]["rows"] if "fs_min_naive_hz" in r]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method, mk in (("naive", "o"), ("peak", "s")):
        pts = [r for r in rows if r.get(f"fs_min_{method}_hz")]
        if pts:
            ax.loglog([1.0 / r["pulse_width_s"] for r in pts],
                      [r[f"fs_min_{method}_hz"] for r in pts], mk, label=method)
    if fits:
        tp_lo = min(r["pulse_width_s"] for r in rows) / 2.0
        tp_hi = max(r["pulse_width_s"] for r in rows) * 2.0
        tp_grid = np.geomspace(tp_lo, tp_hi, 40)
        for method, style in (("naive", "k:"), ("peak", "k-.")):
            if method not in fits:
                continue
            f = fits[method]
            for mval in sorted({r["m"] for r in rows}):
                ax.loglog(1.0 / tp_grid, f["a"] * mval ** f["b"] / tp_grid ** f["c"],
                          style, lw=0.9,
                          label=(f"{method} fit: {f['a']:.2f} m$^{f['b']:.2f}$"
                                 f"/t$_p^{{{f['c']:.2f}}}$ (m={mval:g})"))
    ax.set_xlabel("1 / pulse width $t_p$ [1/s]"); ax.set_ylabel("minimum $f_s$ for |dD/D|<10% [Hz]")
    ax.set_title("minimum sampling rate law"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "fig5_min_rate_law.png", dpi=130); plt.close(fig)
    print(f"  wrote figures to {(FIGS).relative_to(ROOT)}")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smaller DEM case; writes to artifacts/in155_n04_quick/ "
                         "so the node artifacts are never overwritten")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    global OUT, FIGS
    if args.quick:
        OUT = OUT.with_name(OUT.name + "_quick")
        FIGS = OUT / "figures"
    t0 = time.time()
    results = {}
    results["contract"] = run_contract_cases(args.quick)
    dump("contract_cases.json", results["contract"])
    results["m5405"] = run_m5405_fidelity()
    dump("m5405_fidelity.json", results["m5405"])
    results["anchors"] = run_anchors()
    dump("anchors.json", results["anchors"])
    results["envelopes"] = run_envelopes(args.quick)
    dump("envelopes.json", results["envelopes"])
    results["min_rate_law"] = run_min_rate_law(args.quick)
    dump("min_rate_law.json", results["min_rate_law"])
    if not args.no_figures:
        make_figures(results)
    results["elapsed_s"] = time.time() - t0
    dump("run_summary.json", results)

    print("\n=== headline numbers ===")
    print("energy budget residual (case 3):",
          results["contract"]["floe_field_bow"]["contract"]["energy"]["max_rel_budget_error"],
          "at dt =", results["contract"]["floe_field_bow"]["contract"]["t_span_s"])
    print("m5405 doc errors:", results["m5405"]["m5405_doc_error"])
    print("single pulse peak error:", results["anchors"]["single_pulse"]["peak_rel_error"])
    print("constant amplitude rel errors:",
          [r["rel_error"] for r in results["anchors"]["constant_amplitude"]])
    for key in ("m5405", "floe_field"):
        block = results["envelopes"][key]
        print(f"{key}: dD/D " + ", ".join(
            f"{r['rate_hz']:g}Hz/{r['method']}={r['dD_over_D']:+.3f}" for r in block["rows"]))
        print("   statistics compression: " + ", ".join(
            f"{k}={v['dD_over_D']:+.2f}" for k, v in block["statistics_compression"].items()
            if k != "exact_rainflow"))
    print("min-rate fits:", {k: {"a": round(v["a"], 3), "b": round(v["b"], 3),
                                 "c": round(v["c"], 3), "r2": round(v["r2_log"], 4)}
                             for k, v in results["min_rate_law"]["fits"].items()})
    print(f"total elapsed {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())