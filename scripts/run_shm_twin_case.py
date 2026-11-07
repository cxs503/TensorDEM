#!/usr/bin/env python3
"""Reproduce every acceptance number of the ice-SHM load-inversion twin.

The script writes, under ``artifacts/shm_twin/``:

* ``case_results.json``  — all numbers, programmatically assembled (no hand copy)
* ``dem_events.npz``     — the raw 1 kHz DEM ice-load records and their metadata
* ``fig1_modes.png`` ... ``fig6_dem.png`` — the figures of the report

Run:  python scripts/run_shm_twin_case.py [--skip-dem] [--quick]
The DEM records are cached; a second run reuses them unless ``--refresh-dem``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.shm.dem_compare import compare_events, correction_trigger
from tensordem.shm.fatigue_update import FatigueAccumulator, StreamingRainflow
from tensordem.shm.load_inversion import bandlimit, load_error_metrics
from tensordem.shm.model import HullGirderROM, SHMConfig
from tensordem.shm.modal_filter import ModalFilter, harmonic_modal_estimate
from tensordem.shm.separation import separation_criteria, split_rigid_elastic
from tensordem.shm.twin import AcquisitionConfig, IceLoadTwin

OUT = ROOT / "artifacts" / "shm_twin"
OUT.mkdir(parents=True, exist_ok=True)

ACQ = AcquisitionConfig()          # 100 Hz, 1 s windows, SG 51, 0.5 Hz split


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def triangle_load(t, rate, amp, rise_fraction=0.8):
    """Ice crushing cycle: slow build-up, fast brittle-failure release."""
    ph = np.mod(t * rate, 1.0)
    return amp * np.where(ph < rise_fraction, ph / rise_fraction,
                          (1.0 - ph) / (1.0 - rise_fraction))


def batch_damage(stress, gate_pa=1.0e6):
    acc = FatigueAccumulator(gate=gate_pa)
    acc.update(stress)
    return acc.total_damage(include_residue=True), acc.summary()


def spectral(x, dt):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    freq = np.fft.rfftfreq(x.size, dt)
    power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    return freq, power


def relative(a, b):
    if b == 0.0:
        return 0.0 if a == 0.0 else float("inf")
    return abs(a - b) / abs(b)


def event_loads(events, dt):
    """Resample cached DEM records onto the twin's 100 Hz grid, scaled."""
    out = []
    for ev in events:
        t = ev["time"]
        f = ev["force"]
        f = f / f.max() * ev["scale_peak_N"]
        grid = np.arange(0.0, t[-1], dt)
        out.append((ev["label"], grid, np.interp(grid, t, f), ev))
    return out


DEM_CASES = [
    dict(label="E1_speed1.5", hull_speed=1.5, breaking_strain=0.012, duration=4.5, seed=1),
    dict(label="E2_speed0.5", hull_speed=0.5, breaking_strain=0.012, duration=4.5, seed=2),
    dict(label="E3_speed0.8_bs0.008", hull_speed=0.8, breaking_strain=0.008, duration=4.5, seed=3),
    dict(label="E4_speed1.0_bs0.02", hull_speed=1.0, breaking_strain=0.02, duration=4.5, seed=4),
    dict(label="E5_speed0.3", hull_speed=0.3, breaking_strain=0.012, duration=4.5, seed=5),
]


def generate_dem_events(refresh=False):
    """Run the reduced DEM indentation case for every ice event (cached)."""
    from tensordem.whipping.dem_load import DEMLoadConfig, dem_contact_force

    cache = OUT / "dem_events.npz"
    meta_path = OUT / "dem_events.json"
    if cache.exists() and meta_path.exists() and not refresh:
        data = np.load(cache)
        metas = json.loads(meta_path.read_text())
        events = []
        for i, m in enumerate(metas):
            events.append(dict(m, time=data[f"t{i}"], force=data[f"f{i}"]))
        return events, [m for m in metas]
    events, metas = [], []
    for case in DEM_CASES:
        cfg = DEMLoadConfig(duration=case["duration"], nx=8, ny=3,
                            hull_speed=case["hull_speed"],
                            breaking_strain=case["breaking_strain"])
        t0 = time.time()
        t, f, info = dem_contact_force(cfg, seed=case["seed"])
        wall = time.time() - t0
        # the DEM case is a reduced model-scale indentation; the record is used
        # as a *shape* and scaled by one force factor to the documented
        # full-scale bow load of the structural case (see report, limits)
        scale_peak = 1.0e7
        meta = dict(case,
                    wall_s=wall, peak_N=float(f.max()), mean_N=float(f.mean()),
                    n_samples=int(t.size), dem_dt_s=float(info["dem_dt_s"]),
                    particle_count=int(info["particle_count"]),
                    broken_bonds=int(info["broken_bonds"]),
                    scale_peak_N=scale_peak,
                    scale_factor=scale_peak / float(f.max()))
        events.append(dict(meta, time=t, force=f))
        metas.append({k: v for k, v in meta.items()})
        print(f"[dem] {case['label']}: {t.size} samples, peak {f.max():.1f} N, "
              f"{wall:.1f} s", flush=True)
    np.savez_compressed(cache, **{f"t{i}": ev["time"] for i, ev in enumerate(events)},
                        **{f"f{i}": ev["force"] for i, ev in enumerate(events)})
    meta_path.write_text(json.dumps(metas, indent=2))
    return events, metas


# --------------------------------------------------------------------------- #
# case A: structural ROM
# --------------------------------------------------------------------------- #
def case_rom(rom):
    from tensordem.whipping import modes as wm

    analytic = wm.free_free_analytic_frequencies(
        rom.case.youngs_modulus * rom.case.inertia, rom.case.mass_per_length,
        rom.case.length, rom.n_modes)
    return {
        "case": rom.case.as_dict(),
        "rom": rom.as_dict(),
        "analytic_dry_frequencies_Hz": [float(v) for v in analytic],
        "dry_frequency_error_vs_analytic": [
            relative(float(f), float(a)) for f, a in zip(rom.dry_freqs, analytic)],
        "dry_frequency_max_error": float(np.max([relative(float(f), float(a))
                                                 for f, a in zip(rom.dry_freqs, analytic)])),
        "strain_mode_matrix": rom.H.tolist(),
        "sensor_condition_number": rom.condition_number(),
        "first_wet_frequency_Hz": float(rom.freqs[0]),
        "separation_cutoff_MHz_ratio": float(rom.freqs[0] / ACQ.separation_cutoff_hz),
    }


# --------------------------------------------------------------------------- #
# case B: analytic anchor of the modal filter
# --------------------------------------------------------------------------- #
def case_anchor(rom, snr_list=(10.0, 15.0, 20.0, 25.0, 30.0), mode=0,
                duration_s=240.0, trim_s=100.0):
    """Single-mode harmonic excitation: truth q_n(t) from the modal integrator."""
    from tensordem.whipping.response import modal_response

    dt = 1.0 / ACQ.sample_rate_hz
    n_modes = rom.n_modes
    t = np.arange(0.0, duration_s, dt)
    amplitude = 2.0e2          # modal force amplitude [N] (~300 micro strain)
    p = np.zeros((n_modes, t.size))
    p[mode] = amplitude * np.sin(rom.omega[mode] * t)
    solution = modal_response(t, p, rom.omega, rom.zeta)
    truth = solution.q
    clean = rom.strain_history(truth)
    rms = float(np.sqrt(np.mean(clean**2)))
    filt = ModalFilter(rom.H)
    half = int(trim_s / dt)
    sl = slice(half, t.size - half)
    window_s = 5.0

    rows = []
    for snr in snr_list:
        rng = np.random.default_rng(2024)
        sigma = rms / (10.0 ** (snr / 20.0))
        eps = clean + rng.normal(0.0, sigma, clean.shape)
        q_inst = filt.apply(eps).q
        q_harm = harmonic_modal_estimate(eps, dt, rom.omega, window_s=window_s,
                                         estimator=filt)
        # evaluate on the steady-state part only
        # relative modal-coordinate error, i.e. rms(est - truth) / rms(truth)
        ref_rms = float(np.sqrt(np.mean(truth[mode, sl] ** 2)))
        inst_err = float(np.sqrt(np.mean((q_inst[mode, sl] - truth[mode, sl]) ** 2)) / ref_rms)
        harm_err = float(np.sqrt(np.mean((q_harm[mode, sl] - truth[mode, sl]) ** 2)) / ref_rms)
        rows.append({"snr_db": snr, "sigma_noise_strain": sigma,
                     "instantaneous_relative_error": inst_err,
                     "harmonic_relative_error": harm_err})
    return {
        "mode": mode + 1, "mode_frequency_Hz": float(rom.freqs[mode]),
        "duration_s": duration_s, "trim_s": trim_s, "harmonic_window_s": window_s,
        "strain_rms": rms, "noise_gain": filt.noise_gain,
        "rows": rows,
        "anchor_error_at_20dB": rows[2]["harmonic_relative_error"],
        "anchor_pass": bool(rows[2]["harmonic_relative_error"] < 0.01),
        "instantaneous_error_at_20dB": rows[2]["instantaneous_relative_error"],
    }


# --------------------------------------------------------------------------- #
# case C: load inversion on synthetic and DEM ice-load histories
# --------------------------------------------------------------------------- #
def case_inversion(twin, rom):
    dt = twin.dt
    synthetic = {}
    for label, rate, amp, kind in (("sawtooth_0.2Hz", 0.2, 1.0e7, "saw"),
                                   ("sawtooth_0.5Hz", 0.5, 8.0e6, "saw"),
                                   ("crushing_0.5Hz", 0.5, 8.0e6, "triangle"),
                                   ("crushing_0.2Hz", 0.2, 1.2e7, "triangle")):
        t = np.arange(0.0, 200.0, dt)
        load = (amp * np.mod(t * rate, 1.0) if kind == "saw"
                else triangle_load(t, rate, amp))
        sim = twin.simulate(load, snr_db=20.0, seed=11)
        res = twin.process(sim["strain"], reference_load=load, q_truth=sim["q"])
        band = res.metrics
        raw = load_error_metrics(res.point_force, load, dt, cut_edges=0.05)
        synthetic[label] = {
            "peak_error": band["peak_error"], "energy_error": band["energy_error"],
            "correlation": band["correlation"], "rms_error": band["rms_error"],
            "peak_error_raw": raw["peak_error"], "correlation_raw": raw["correlation"],
            "bandlimit_hz": band["bandlimit_hz"],
            "cutoff_hz": res.config["inversion_cutoff_hz"],
            "modal_q_error": res.as_dict()["modal_q_error"],
            "strain_rms": float(np.sqrt(np.mean(sim["strain"] ** 2))),
            "snr_db": 20.0,
        }
    return synthetic


# --------------------------------------------------------------------------- #
# case D: rigid / elastic separation
# --------------------------------------------------------------------------- #
def pulse_load(t, rate, amp, width):
    ph = np.mod(t * rate, 1.0)
    return amp * np.where(ph < width, np.sin(np.pi * ph / width) ** 2, 0.0)


def case_separation(twin):
    dt = twin.dt
    t = np.arange(0.0, 200.0, dt)
    smooth_load = 6.0e6 * (0.5 - 0.5 * np.cos(2 * np.pi * 0.15 * t)) ** 2
    fast_load = 6.0e6 * np.mod(t * 1.0, 1.0)
    whipping_load = pulse_load(t, 0.3, 1.0e7, 0.05)
    out = {}
    for label, load in (("quasi_static", smooth_load), ("vibratory", fast_load),
                        ("whipping_pulses", whipping_load)):
        sim = twin.simulate(load, snr_db=20.0, seed=13)
        res = twin.process(sim["strain"], reference_load=load, q_truth=sim["q"])
        strain_split = split_rigid_elastic(sim["strain"][2], dt,
                                           cutoff_hz=ACQ.separation_cutoff_hz)
        load_split = split_rigid_elastic(load, dt, cutoff_hz=ACQ.separation_cutoff_hz)
        inv_split = split_rigid_elastic(res.point_force, dt,
                                       cutoff_hz=ACQ.separation_cutoff_hz)
        out[label] = {
            "strain_split": strain_split.as_dict(),
            "load_split": load_split.as_dict(),
            "inverted_load_split": inv_split.as_dict(),
            "criteria_strain": separation_criteria(strain_split.rigid_fraction,
                                                   strain_split.elastic_fraction),
            "criteria_load": separation_criteria(load_split.rigid_fraction,
                                                 load_split.elastic_fraction),
            "reconstruction_error": float(np.max(np.abs(
                strain_split.rigid + strain_split.elastic - sim["strain"][2]))),
        }
    out["pass_rigid"] = bool(out["quasi_static"]["strain_split"]["rigid_fraction"] > 0.80)
    out["pass_elastic"] = bool(
        out["vibratory"]["strain_split"]["elastic_fraction"] > 0.70 and
        out["whipping_pulses"]["strain_split"]["elastic_fraction"] > 0.70)
    return out


# --------------------------------------------------------------------------- #
# case E: fatigue accumulation over 1000 ice load cycles
# --------------------------------------------------------------------------- #
def case_fatigue(twin, n_cycles=1000, rate=0.5):
    dt = twin.dt
    duration = n_cycles / rate
    t = np.arange(0.0, duration, dt)
    load = triangle_load(t, rate, 8.0e6)
    sim = twin.simulate(load, snr_db=20.0, seed=3)
    res = twin.process(sim["strain"], reference_load=load, q_truth=sim["q"])
    gate = ACQ.fatigue_gate_MPa * 1.0e6
    d_truth, s_truth = batch_damage(sim["sigma_deck"], gate)
    d_inv, s_inv = batch_damage(res.sigma_deck, gate)
    online = res.fatigue["damage"]
    # streaming equivalence on the identical stress record (gate 0)
    counter = StreamingRainflow(gate=0.0)
    counter.update(res.sigma_deck)
    stream_closed_free = counter.all_cycles()
    from tensordem.whipping.rainflow import rainflow as batch_rainflow

    batch = batch_rainflow(res.sigma_deck)
    pair_ok = (len(stream_closed_free) == len(batch) and
               all(math.isclose(a.range, b.range, rel_tol=1e-12, abs_tol=1e-9)
                   for a, b in zip(stream_closed_free, batch)))
    # discontinuous sawtooth: documented limit case
    saw = 8.0e6 * np.mod(t * rate, 1.0)
    sim_saw = twin.simulate(saw, snr_db=20.0, seed=3)
    res_saw = twin.process(sim_saw["strain"], reference_load=saw)
    d_saw_truth, _ = batch_damage(sim_saw["sigma_deck"], gate)
    hist = np.array(res.fatigue_history)
    return {
        "n_cycles": n_cycles, "cycle_rate_Hz": rate, "duration_s": duration,
        "snr_db": 20.0, "gate_MPa": ACQ.fatigue_gate_MPa,
        "online_damage": online,
        "online_damage_closed": res.fatigue["damage_closed"],
        "batch_damage_inverted_stress": d_inv,
        "batch_damage_truth_stress": d_truth,
        "error_online_vs_batch_inverted": relative(online, d_inv),
        "error_online_vs_truth": relative(online, d_truth),
        "cycles_online": res.fatigue["cycles"],
        "cycles_truth": s_truth["cycles"],
        "equivalent_range_MPa": res.fatigue["equivalent_range_MPa"],
        "max_range_MPa": res.fatigue["max_range_MPa"],
        "batch_rainflow_equivalence_gate0": bool(pair_ok),
        "streaming_vs_batch_cycles": [len(stream_closed_free), len(batch)],
        "limit_case_sawtooth": {
            "online_damage": res_saw.fatigue["damage"],
            "truth_damage": d_saw_truth,
            "error": relative(res_saw.fatigue["damage"], d_saw_truth),
        },
        "damage_history_final": float(hist[-1, 1]) if hist.size else 0.0,
        "latency": res.latency_summary(),
        "latency_raw_ms": [float(v) for v in res.latency_ms],
        "_history": hist,
    }


# --------------------------------------------------------------------------- #
# case F: DEM online comparison over the ice-event set
# --------------------------------------------------------------------------- #
def case_dem(twin, events):
    dt = twin.dt
    records, per_event = [], []
    for label, t, load, meta in event_loads(events, dt):
        sim = twin.simulate(load, snr_db=20.0, seed=17)
        res = twin.process(sim["strain"], reference_load=load)
        # edge trim: 3/(2 f_c) = 0.75 s at each end at the 2 Hz comparison band;
        # 0.20 of a 4.5 s record leaves 2.7 s of steady-state comparison window
        m = load_error_metrics(res.point_force, load, dt, cut_edges=0.20,
                               bandlimit_hz=ACQ.bandlimit_hz)
        raw = load_error_metrics(res.point_force, load, dt, cut_edges=0.20)
        corr = correction_trigger(m)
        per_event.append({
            "label": label, "n_samples": int(load.size),
            "dem_peak_N": float(np.max(np.abs(load))),
            "dem_mean_N": float(load.mean()),
            "peak_error": m["peak_error"], "energy_error": m["energy_error"],
            "correlation": m["correlation"], "rms_error": m["rms_error"],
            "peak_error_raw": raw["peak_error"],
            "correlation_raw": raw["correlation"],
            "correction": corr,
            "bandlimit_hz": ACQ.bandlimit_hz,
            "strain_rms": float(np.sqrt(np.mean(sim["strain"] ** 2))),
        })
        records.append({"estimate": res.point_force, "reference": load, "dt": dt,
                        "label": label, "bandlimit_hz": ACQ.bandlimit_hz,
                        "cut_edges": 0.20})
        per_event[-1]["_series"] = (t, load, res.point_force)
    summary = compare_events(records, min_events=5)
    summary.pop("events", None)
    return per_event, summary


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(rom, anchor, synthetic, separation, fatigue, history, dem_events,
                 dem_per_event, dem_summary, dem_metas):
    plt.rcParams.update({"figure.dpi": 130, "font.size": 8})

    # --- fig1: modes, spectra, H -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    x = np.linspace(0, rom.length, 301)
    for n in range(rom.n_modes):
        phi = rom.shape(x, n)
        axes[0].plot(x, phi / np.max(np.abs(phi)),
                     label=f"$\\phi_{n+1}$  {rom.freqs[n]:.2f} Hz (wet)"
                           f" / {rom.dry_freqs[n]:.2f} Hz (dry)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("normalised mode shape")
    axes[0].legend(fontsize=6)
    axes[0].set_title("hull-girder dry/wet modes")
    im = axes[1].imshow(rom.H, aspect="auto", cmap="RdBu_r")
    axes[1].set_xlabel("mode n")
    axes[1].set_ylabel("sensor k")
    axes[1].set_title(f"strain-mode matrix H (cond={rom.condition_number():.1f})")
    fig.colorbar(im, ax=axes[1], label="-$y_k\\phi_n''(x_k)$ [1/m]")
    snr = [r["snr_db"] for r in anchor["rows"]]
    axes[2].semilogy(snr, [100 * r["instantaneous_relative_error"] for r in anchor["rows"]],
                     "o-", label="instantaneous LS")
    axes[2].semilogy(snr, [100 * r["harmonic_relative_error"] for r in anchor["rows"]],
                     "s-", label="harmonic (lock-in)")
    axes[2].axhline(1.0, color="k", ls="--", lw=0.8, label="1 % anchor")
    axes[2].set_xlabel("sensor SNR [dB]")
    axes[2].set_ylabel("$q_1$ relative error [%]")
    axes[2].legend(fontsize=6)
    axes[2].set_title("modal filter error vs SNR")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_modes.png")
    plt.close(fig)

    # --- fig2: anchor time series + spectrum --------------------------------
    t = np.arange(0, 40, 0.01)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    ph = np.mod(t * 0.5, 1.0)
    load = 8.0e6 * np.where(ph < 0.8, ph / 0.8, (1 - ph) / 0.2)
    axes[0].plot(t, load / 1e6, lw=0.8, label="crushing cycle (truth)")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("ice load [MN]")
    axes[0].set_title("ice crushing cycle: slow build-up, brittle release")
    axes[0].legend(fontsize=7)
    for label, color in (("crushing_0.5Hz", "C0"), ("sawtooth_0.5Hz", "C1")):
        dt = 0.01
        tt = np.arange(0, 200, dt)
        ref = (triangle_load(tt, 0.5, 8e6) if label.startswith("crushing")
               else 8e6 * np.mod(tt * 0.5, 1.0))
        f, P = spectral(ref, dt)
        axes[1].semilogy(f, P / P.max(), lw=0.8, color=color, label=label + " (truth)")
    axes[1].set_xlim(0, 6)
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("normalised PSD")
    axes[1].axvline(ACQ.separation_cutoff_hz, color="k", ls="--", lw=0.8)
    axes[1].axvline(0.44 / (0.5 * ACQ.sg_window * 0.01), color="r", ls=":", lw=0.8,
                    label="inversion bandwidth")
    axes[1].legend(fontsize=6)
    axes[1].set_title("load spectrum: rigid (<0.5 Hz) vs elastic band")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_load_spectrum.png")
    plt.close(fig)

    # --- fig3: inversion time series for the best DEM event -----------------
    best = min(dem_per_event, key=lambda e: e["peak_error"])
    t, ref, est = best["_series"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot(t, ref / 1e6, lw=1.0, label="DEM reference")
    axes[0].plot(t, est / 1e6, lw=1.0, ls="--", label="strain inversion")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("ice load [MN]")
    axes[0].set_title(f"{best['label']}: peak err {100*best['peak_error']:.1f}%, "
                      f"corr {best['correlation']:.3f}")
    axes[0].legend(fontsize=7)
    for sig, lab in ((ref, "DEM reference"), (est, "inversion")):
        f, P = spectral(sig, 0.01)
        axes[1].semilogy(f, P / P.max(), lw=0.9, label=lab)
    axes[1].set_xlim(0, 8)
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("normalised PSD")
    axes[1].legend(fontsize=7)
    axes[1].set_title("spectral match")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_inversion.png")
    plt.close(fig)

    # --- fig4: per-event comparison scatter --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    corr = [e["correlation"] for e in dem_per_event]
    peak = [100 * e["peak_error"] for e in dem_per_event]
    energy = [100 * e["energy_error"] for e in dem_per_event]
    labels = [e["label"] for e in dem_per_event]
    axes[0].scatter(corr, peak, c="C0", label="peak error")
    axes[0].scatter(corr, energy, c="C3", marker="s", label="energy error")
    for c, p, lab in zip(corr, peak, labels):
        axes[0].annotate(lab.split("_")[0], (c, p), fontsize=6)
    axes[0].axvline(0.70, color="k", ls="--", lw=0.8, label="corr floor 0.7")
    axes[0].axhline(15.0, color="C0", ls=":", lw=0.8)
    axes[0].axhline(10.0, color="C3", ls=":", lw=0.8)
    axes[0].set_xlabel("correlation")
    axes[0].set_ylabel("error [%]")
    axes[0].legend(fontsize=6)
    axes[0].set_title("DEM online comparison, 5 events")
    axes[1].plot(history[:, 0], history[:, 1], lw=1.0, color="C2")
    axes[1].set_xlabel("accumulated damage $D\\times10^{0}$ (final "
                       f"D={fatigue['online_damage']:.2e})")
    axes[1].set_ylabel("windows")
    axes[1].set_title("online Miner damage growth (1000 cycles)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_dem_compare.png")
    plt.close(fig)

    # --- fig5: separation + latency ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    dt = 0.01
    tt = np.arange(0, 200, dt)
    for load_label, color, load in (
            ("quasi_static", "C0", 6e6 * (0.5 - 0.5 * np.cos(2 * np.pi * 0.15 * tt)) ** 2),
            ("vibratory (1 Hz)", "C1", 6e6 * np.mod(tt * 1.0, 1.0)),
            ("whipping pulses", "C3", pulse_load(tt, 0.3, 1e7, 0.05))):
        f, P = spectral(load, dt)
        axes[0].semilogy(f, P / P.max(), lw=0.8, color=color, label=load_label)
    axes[0].axvline(0.5, color="k", ls="--", lw=0.8, label="0.5 Hz split")
    axes[0].set_xlim(0, 4)
    axes[0].set_xlabel("frequency [Hz]")
    axes[0].set_ylabel("normalised PSD")
    axes[0].legend(fontsize=6)
    axes[0].set_title("rigid vs elastic excitation")
    fr = [separation["quasi_static"]["strain_split"]["rigid_fraction"],
          separation["vibratory"]["strain_split"]["elastic_fraction"],
          separation["whipping_pulses"]["strain_split"]["elastic_fraction"]]
    axes[1].bar(["quasi-static\nrigid fraction", "1 Hz crushing\nelastic fraction",
                 "whipping pulses\nelastic fraction"], fr,
                color=["C0", "C1", "C3"])
    axes[1].axhline(0.80, color="k", ls=":", lw=0.8)
    axes[1].axhline(0.70, color="k", ls="--", lw=0.8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("energy fraction")
    axes[1].set_title("separation criteria against 0.80 / 0.70")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_separation.png")
    plt.close(fig)

    # --- fig6: latency ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    lat = np.asarray(fatigue["latency_raw_ms"])
    axes[0].plot(lat, lw=0.6)
    axes[0].axhline(50.0, color="r", ls="--", lw=0.8, label="50 ms budget")
    axes[0].set_xlabel("window index (1 s windows)")
    axes[0].set_ylabel("per-window latency [ms]")
    axes[0].legend(fontsize=7)
    axes[0].set_title(f"inversion latency, median {np.median(lat):.1f} ms")
    f, P = spectral(dem_per_event[0]["_series"][1], 0.01)
    axes[1].loglog(f[1:], P[1:], lw=0.8)
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("PSD (DEM reference)")
    axes[1].set_title("DEM record spectral content")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_latency.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-dem", action="store_true")
    ap.add_argument("--skip-dem", action="store_true")
    ap.add_argument("--quick", action="store_true", help="200 instead of 1000 cycles")
    args = ap.parse_args()

    t_start = time.time()
    rom = HullGirderROM()
    twin = IceLoadTwin(acquisition=ACQ)
    results = {"scenario": "IN30-N08 ice SHM / load-inversion twin",
               "repository": "TensorDEM", "module": "tensordem.shm"}

    results["A_structural_model"] = case_rom(rom)
    print("[A] ROM built", flush=True)

    results["B_analytic_anchor"] = case_anchor(rom)
    print("[B] anchor done", flush=True)

    results["C_load_inversion_synthetic"] = case_inversion(twin, rom)
    print("[C] synthetic inversion done", flush=True)

    results["D_rigid_elastic"] = case_separation(twin)
    print("[D] separation done", flush=True)

    fatigue = case_fatigue(twin, n_cycles=200 if args.quick else 1000)
    hist = fatigue.pop("_history")
    results["E_fatigue"] = fatigue
    print("[E] fatigue done", flush=True)

    if args.skip_dem:
        events, metas, per_event, summary = [], [], [], {}
    else:
        events, metas = generate_dem_events(refresh=args.refresh_dem)
        per_event, summary = case_dem(twin, events)
        print("[F] DEM comparison done", flush=True)
    results["F_dem_events_meta"] = [
        {k: v for k, v in m.items() if k not in ("time", "force")} for m in metas]
    results["F_dem_comparison"] = summary
    results["F_dem_per_event"] = [
        {k: v for k, v in e.items() if not k.startswith("_")} for e in per_event]
    results["G_latency"] = fatigue.get("latency", {})

    # ---------------------------------------------------------------- verdicts
    verdicts = {}
    b = results["B_analytic_anchor"]
    verdicts["a_modal_filter_anchor"] = {
        "value": b["anchor_error_at_20dB"], "limit": 0.01,
        "pass": bool(b["anchor_error_at_20dB"] < 0.01),
        "note": "single-mode sine, harmonic estimator at 20 dB sensor SNR"}
    syn = results["C_load_inversion_synthetic"]
    worst_peak = max(v["peak_error"] for v in syn.values())
    worst_energy = max(v["energy_error"] for v in syn.values())
    verdicts["b_load_inversion"] = {
        "worst_peak_error": worst_peak,
        "worst_energy_error": worst_energy,
        "worst_correlation": min(v["correlation"] for v in syn.values()),
        "pass": bool(worst_peak < 0.15 and worst_energy < 0.10),
        "note": ("worst case over the four analytic ice-load histories, "
                 "band-limited (2 Hz) comparison; raw peak errors reported separately")}
    d = results["D_rigid_elastic"]
    verdicts["c_rigid_elastic_separation"] = {
        "rigid_fraction": d["quasi_static"]["strain_split"]["rigid_fraction"],
        "elastic_fraction": d["vibratory"]["strain_split"]["elastic_fraction"],
        "pass": bool(d["pass_rigid"] and d["pass_elastic"]),
        "limits": {"rigid": 0.80, "elastic": 0.70}}
    e = results["E_fatigue"]
    verdicts["d_fatigue_accumulation"] = {
        "error_vs_batch_same_stress": e["error_online_vs_batch_inverted"],
        "error_vs_truth_stress": e["error_online_vs_truth"],
        "pass": bool(e["error_online_vs_batch_inverted"] < 0.10 and
                     e["error_online_vs_truth"] < 0.10),
        "note": "1000 crushing cycles, 20 dB SNR, 1 MPa rainflow gate"}
    lat = results["G_latency"]
    verdicts["e_realtime_latency"] = {
        "median_ms": lat.get("median_ms"), "p95_ms": lat.get("p95_ms"),
        "p99_ms": lat.get("p99_ms"), "max_ms": lat.get("max_ms"),
        "limit_ms": 50.0, "n_windows_over_limit": lat.get("n_windows_over_limit"),
        "n_windows": lat.get("n_windows"),
        "pass": bool(lat.get("within_limit", False)),
        "note": ("verdict on p99 < 25 ms with a <=0.1 % allowance of windows over the "
                 "50 ms budget; the raw maximum is reported as well")}
    if summary:
        verdicts["f_dem_comparison"] = {
            "n_events": summary["n_events"],
            "correlation_mean": summary["correlation_mean"],
            "correlation_min": summary["correlation_min"],
            "pass": bool(summary["n_events"] >= 5 and summary["correlation_min"] > 0.70),
            "n_corrections_triggered": summary["n_corrections_triggered"]}
    results["verdicts"] = verdicts
    results["all_pass"] = bool(all(v.get("pass", False) for v in verdicts.values()))
    results["wall_clock_s"] = time.time() - t_start
    results["environment"] = {"python": sys.version.split()[0],
                              "numpy": np.__version__,
                              "sample_rate_hz": ACQ.sample_rate_hz,
                              "chunk_s": ACQ.chunk_s, "sg_window": ACQ.sg_window,
                              "bandlimit_hz": ACQ.bandlimit_hz}

    make_figures(rom, b, syn, d, e, hist, events, per_event, summary, metas)
    (OUT / "case_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(verdicts, indent=2, default=float), flush=True)
    print(f"[done] {results['wall_clock_s']:.1f} s -> {OUT/'case_results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())