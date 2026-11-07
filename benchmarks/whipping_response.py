"""End-to-end whipping benchmark: DEM ice pulses -> wet modes -> fatigue.

Produces ``artifacts/whipping.json`` (all acceptance numbers, programmatically
injected -- nothing is transcribed by hand), ``artifacts/whipping/*.png``
figures and a cached 1 kHz DEM contact-force record
(``artifacts/whipping/contact_force_1kHz.csv``) so the structural part can be
re-run without repeating the DEM integration.

Usage
-----
``python benchmarks/whipping_response.py``             # full run (DEM + chain)
``python benchmarks/whipping_response.py --reuse``     # reuse the cached record
``python benchmarks/whipping_response.py --fast``      # coarse, for smoke tests
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from tensordem.whipping import (  # noqa: E402
    ShipCase,
    bandpass,
    dem_contact_force,
    detect_pulses,
    lowpass,
    rainflow_cycles,
    resample_uniform,
)
from tensordem.whipping import pipeline, pulses  # noqa: E402
from tensordem.whipping.dem_load import DEMLoadConfig  # noqa: E402
from tensordem.whipping.rainflow import range_histogram  # noqa: E402

ART = ROOT / "artifacts" / "whipping"
CSV = ART / "contact_force_1kHz.csv"


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover - only for the report metadata
        return "unknown"


def _load_or_run_dem(reuse: bool, fast: bool):
    if reuse and CSV.exists():
        data = np.loadtxt(CSV, delimiter=",", skiprows=1)
        info_path = ART / "dem_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        return data[:, 0], data[:, 1], info, "cached"
    cfg = DEMLoadConfig(duration=0.4 if fast else 2.4, nx=10 if not fast else 8,
                        ny=4 if not fast else 3)
    t, f, info = dem_contact_force(cfg)
    ART.mkdir(parents=True, exist_ok=True)
    np.savetxt(CSV, np.column_stack([t, f]), delimiter=",",
               header="time_s,force_N", comments="")
    (ART / "dem_info.json").write_text(json.dumps(info, indent=1))
    return t, f, info, "fresh"


def _figure_force_pulses(t, f, plist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(t, f / 1e3, lw=0.9, color="#22456b")
    for p in plist:
        ax.plot([p.t_onset, p.t_peak, p.t_end], [0, p.peak / 1e3, 0],
                color="#c0392b", lw=1.2, alpha=0.8)
        ax.annotate(f"$t_r$={p.rise_time*1e3:.0f} ms", (p.t_peak, p.peak / 1e3),
                    textcoords="offset points", xytext=(3, 4), fontsize=7,
                    color="#c0392b")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("hull reaction |F| [kN] (DEM)")
    ax.set_title("DEM ice-hull contact force and detected pulse events")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _figure_pulse_fit(t, f, pparams, path, window=0.7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    if pparams:
        p0 = pparams[0].features
        m = (t >= p0.t_onset - 0.02) & (t <= p0.t_onset + window)
        ax.plot(t[m], f[m] / 1e3, color="#22456b", lw=1.0, label="DEM")
        tt = np.linspace(t[m][0], t[m][-1], 2000)
        ax.plot(tt, pparams[0](tt) / 1e3, "--", color="#c0392b",
                label=f"parametric ($p$={pparams[0].exponent:.2f})")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("force [kN]")
        ax.legend()
    ax.set_title("Single pulse: four-feature parametric reconstruction")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _figure_modes(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case = ShipCase()
    from tensordem.whipping import modes as modemod

    ei = case.youngs_modulus * case.inertia
    dry = modemod.dry_modes(ei, case.mass_per_length, case.length,
                            case.n_elements, case.n_modes)
    mass = pipeline.hull_added_mass_per_length(case)
    wet = modemod.wet_modes(dry, mass["m_a_of_x"])
    x = np.linspace(0, case.length, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for i in range(min(3, case.n_modes)):
        axes[0].plot(x, dry.shape(x, i), lw=1.2, label=f"mode {i+1} dry {dry.freqs[i]:.2f} Hz")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("mode shape (mass normalised)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Dry hull-girder modes (beam FEM)")
    axes[0].grid(alpha=0.3)
    idx = np.arange(1, case.n_modes + 1)
    width = 0.38
    axes[1].bar(idx - width / 2, dry.freqs, width, label="dry", color="#7f8c8d")
    axes[1].bar(idx + width / 2, wet.freqs, width, label="wet", color="#22456b")
    for i in idx:
        axes[1].annotate(f"-{100*wet.reduction[i-1]:.1f}%",
                         (i + width / 2, wet.freqs[i - 1]),
                         textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=7)
    axes[1].set_xticks(idx)
    axes[1].set_xlabel("mode")
    axes[1].set_ylabel("frequency [Hz]")
    axes[1].legend()
    axes[1].set_title("Dry vs wet frequencies (2-D BEM added mass)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _figure_stress(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = result["series"]
    t = s["time"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(t, s["stress"] / 1e6, lw=0.8, color="#22456b", label=r"$\sigma$ total")
    ax.plot(t, s["stress_qs"] / 1e6, lw=1.4, color="#27ae60", label="quasi-static (<1 Hz)")
    ax.plot(t, s["stress_wh"] / 1e6, lw=1.1, color="#c0392b", label="whipping (1-50 Hz)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("deck stress [MPa]")
    ax.legend(fontsize=8)
    ax.set_title("Deck stress split: quasi-static vs whipping")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _figure_rainflow(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for name, sig, colour in (("total", result["series"]["stress"], "#22456b"),
                              ("whipping", result["series"]["stress_wh"], "#c0392b"),
                              ("quasi-static", result["series"]["stress_qs"], "#27ae60")):
        hist = range_histogram(rainflow_cycles(sig))
        centres = 0.5 * (np.array(hist["bin_edges"][:-1]) + np.array(hist["bin_edges"][1:])) / 1e6
        axes[0].step(centres, hist["counts"], where="mid", lw=1.2, label=name, color=colour)
    axes[0].set_xlabel("stress range $\\Delta\\sigma$ [MPa]")
    axes[0].set_ylabel("rainflow cycles")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Rainflow range spectrum")
    axes[0].grid(alpha=0.3)
    labels = ["total", "quasi-static", "whipping"]
    vals = [result["response"]["damage_total"],
            result["response"]["damage_quasi_static"],
            result["response"]["damage_whipping"]]
    axes[1].bar(labels, vals, color=["#22456b", "#27ae60", "#c0392b"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Miner damage D (-)")
    axes[1].set_title("DNV-RP-C203 curve D damage")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _relief_sensitivity(t, f, case, reliefs=(1.0, 0.85, 0.7, 0.5)):
    out = []
    for r in reliefs:
        res = pipeline.assess(t, f, ShipCase(relief=r), integration_rate_hz=2000.0)
        out.append({
            "relief": r,
            "dry_freqs_Hz": res["modes"]["dry_freqs_Hz"],
            "wet_freqs_Hz": res["modes"]["wet_freqs_Hz"],
            "reduction": res["modes"]["reduction"],
            "added_mass_to_hull_mass": res["modes"]["added_mass_to_hull_mass"],
            "whipping_stress_share": res["criteria"]["c_whipping_stress_share"],
            "whipping_damage_share": res["criteria"]["d_whipping_damage_share"],
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse the cached DEM record")
    ap.add_argument("--fast", action="store_true", help="coarse run for smoke tests")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    ART.mkdir(parents=True, exist_ok=True)
    t, f, dem_info, source = _load_or_run_dem(args.reuse, args.fast)
    case = ShipCase()
    result = pipeline.assess(t, f, case, integration_rate_hz=2000.0)
    t2, f2 = resample_uniform(t, f, 2000.0)
    plist = detect_pulses(t2, f2)

    payload = {
        "scenario": "ice_induced_whipping_demo",
        "repository": "TensorDEM",
        "module": "tensordem.whipping",
        "git_rev": _git_rev(),
        "python": platform.python_version(),
        "dem_source": source,
        "dem_info": dem_info,
        "sampling": {
            "record_rate_hz": float(1.0 / (t[1] - t[0])),
            "integration_rate_hz": 2000.0,
            "duration_s": float(t[-1]),
        },
        "ship": result["ship"],
        "load": result["load"],
        "modes": result["modes"],
        "response": result["response"],
        "energy": result["energy"],
        "per_pulse": result["per_pulse"],
        "criteria": result["criteria"],
        "relief_sensitivity": (_relief_sensitivity(t, f, case) if not args.fast else []),
    }
    (ART / "whipping.json").write_text(json.dumps(payload, indent=1, default=float))
    (ROOT / "artifacts" / "whipping.json").write_text(json.dumps(payload, indent=1, default=float))

    if not args.no_figures:
        _figure_force_pulses(t2, f2, plist, ART / "fig1_contact_force_pulses.png")
        _figure_pulse_fit(t2, f2, [pulses.ParametricPulse.from_features(p)
                                   for p in plist if p.impulse > 0][:1],
                          ART / "fig2_pulse_parametrisation.png")
        _figure_modes(result, ART / "fig3_modes_dry_wet.png")
        _figure_stress(result, ART / "fig4_stress_split.png")
        _figure_rainflow(result, ART / "fig5_rainflow_damage.png")

    crit = result["criteria"]
    print(json.dumps(crit, indent=1))
    for key, label in (
        ("a_in_band_10_30_pct", "wet frequency drop 10-30 %"),
        ("b_in_band_10_50_ms", "rise time 10-50 ms"),
        ("c_in_band_20_40_pct", "whipping stress share 20-40 %"),
        ("d_in_band_15_35_pct", "whipping damage share 15-35 %"),
        ("e_residual_below_1e_6", "energy ledger residual < 1e-6"),
        ("f_negative_control_red", "negative control red (<5 %)"),
        ("g_all_below_20_pct", "pulse dispersion < 20 %"),
    ):
        print(f"  [{'PASS' if crit[key] else 'FAIL'}] {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())