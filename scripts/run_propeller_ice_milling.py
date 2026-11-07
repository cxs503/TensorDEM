"""Run the propeller-ice milling case matrix and write artifacts + figures.

Usage::

    PYTHONPATH=src python scripts/run_propeller_ice_milling.py [--out artifacts/propeller_ice]

Produces ``milling_cases.json`` (all numbers, no hand-transcribed values) and the
figures used by ``docs/PROPELLER_ICE_MILLING_TRANSIENT.md``:

``fig_torque_timeseries.png``   Q(t), T(t), F_c(t) of the four matrix cases
``fig_spectrum.png``            torque PSD with the BPF and harmonics marked
``fig_debris.png``              chip trajectories in the (r, z) meridional plane
``fig_skew_sweep.png``          torque pulsation and harmonic content versus skew
``fig_open_water.png``          K_T, K_Q, eta versus J of the BEM model
``fig_thin_ice.png``            bore-through of a thin ice block
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.propeller_ice import (  # noqa: E402
    MillingResult,
    IceBlockGeometry,
    MillingConfig,
    PropellerGeometry,
    analyse_torque_spectrum,
    blade_passing_frequency,
    harmonic_amplitudes,
    periodogram,
    simulate_milling,
    solve_open_water,
)

RPM_CASES = (60.0, 120.0)
THICKNESS_CASES = (0.30, 0.80)
ADVANCE_RATIO = 0.60
DURATION = 4.0
SPIN_UP = 1.0


def _case_config(rpm: float, thickness: float, **overrides) -> MillingConfig:
    return MillingConfig(rpm=rpm, advance_ratio=ADVANCE_RATIO, duration=DURATION,
                         spin_up=SPIN_UP,
                         ice=IceBlockGeometry(thickness=thickness), **overrides)


def _case_record(name: str, result) -> dict:
    spectrum = analyse_torque_spectrum(result.time, result.torque_ice, rpm=result.config.rpm,
                                       n_blades=result.config.propeller.n_blades,
                                       window=result.steady_mask())
    mask = result.steady_mask()
    harmonics = harmonic_amplitudes(result.time[mask], result.torque_ice[mask],
                                    blade_passing_frequency(result.config.rpm,
                                                            result.config.propeller.n_blades))
    record = {
        "case": name,
        "torque_pulsation": result.torque_pulsation(),
        "ejection": result.ejection_statistics(),
        "spectrum": spectrum.as_dict(),
        "harmonic_fit": {
            "amplitudes": {str(k): float(v) for k, v in harmonics["amplitudes"].items()},
            "ratios_to_fundamental": {str(k): float(v)
                                      for k, v in harmonics["ratios_to_fundamental"].items()},
            "r2": float(harmonics["harmonic_fit_r2"]),
        },
        "summary": result.as_dict(),
    }
    return record


def _fig_timeseries(results: list, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True)
    for result, label in results:
        t = result.time
        axes[0].plot(t, result.torque_ice, lw=0.9, label=label)
        axes[1].plot(t, result.thrust, lw=0.9, label=label)
        axes[2].plot(t, result.normal_force, lw=0.9, label=label)
    axes[0].set_ylabel("ice torque Q [N m]")
    axes[1].set_ylabel("thrust T [N]")
    axes[2].set_ylabel("contact force $F_c$ [N]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Propeller-ice milling transients: n = 60/120 rpm, $h_{ice}$ = 0.30/0.80 m")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_spectrum(results: list, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    for result, label in results:
        mask = result.steady_mask()
        freqs, psd = periodogram(result.time[mask], result.torque_ice[mask])
        axes[0].semilogy(freqs[1:], psd[1:], lw=0.8, label=label)
        f_bpf = blade_passing_frequency(result.config.rpm, result.config.propeller.n_blades)
        for k in (1, 2, 3, 4):
            axes[0].axvline(k * f_bpf, color="0.7", ls=":", lw=0.6)
        axes[1].plot(result.time, result.torque_ice, lw=0.8, label=label)
    axes[0].set_ylabel("PSD [N$^2$m$^2$/Hz]")
    axes[0].set_xlabel("frequency [Hz]  (dotted: BPF and harmonics)")
    axes[1].set_ylabel("Q [N m]")
    axes[1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Torque spectrum: dominant line at the blade-passing frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_debris(result, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    radius = result.config.propeller.radius
    for path_pts in result.chip_paths:
        if path_pts.shape[0] < 2:
            continue
        r = np.hypot(path_pts[:, 0], path_pts[:, 1])
        ax.plot(r / radius, path_pts[:, 2], lw=0.6, color="tab:blue", alpha=0.5)
    ax.scatter(np.hypot(result.chip_release_pos[:, 0], result.chip_release_pos[:, 1]) / radius,
               np.zeros(result.chip_release_pos.shape[0]), s=4, color="tab:red",
               label="release points")
    ax.axhline(result.config.exit_distance_resolved, color="k", ls="--", lw=0.8,
               label=f"ejection stand-off {result.config.exit_distance_resolved*1e3:.0f} mm")
    ax.set_xlabel("r / R")
    ax.set_ylabel("axial distance downstream z [m]")
    ax.set_title(f"Ice-debris trajectories ({result.config.rpm:.0f} rpm), "
                 f"mean ejection speed {result.ejection_statistics()['ejection_speed_mean']:.2f} m/s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_skew(sweep: list[dict], path: Path) -> None:
    skew = [row["skew_tip_deg"] for row in sweep]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].plot(skew, [row["delta_q_over_mean_p95"] for row in sweep], "o-")
    axes[0].axhspan(0.10, 0.30, color="tab:green", alpha=0.12, label="acceptance band")
    axes[0].set_xlabel("tip skew [deg]")
    axes[0].set_ylabel(r"$\Delta Q / \bar Q$ (p95 range)")
    axes[0].legend(fontsize=8)
    axes[1].plot(skew, [row["p2_over_p1"] for row in sweep], "s-", label="$A_2/A_1$")
    axes[1].plot(skew, [row["p3_over_p1"] for row in sweep], "^-", label="$A_3/A_1$")
    axes[1].set_xlabel("tip skew [deg]")
    axes[1].set_ylabel("harmonic amplitude ratio")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Skew smears the sector-entry load step: pulsation and harmonic content")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_open_water(geometry: PropellerGeometry, path: Path) -> list[dict]:
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax2 = ax.twinx()
    js = np.round(np.arange(0.0, 1.15, 0.1), 2)
    kt, kq, eta = [], [], []
    for j in js:
        va = j * (60.0 / 60.0) * geometry.diameter
        state = solve_open_water(geometry, 1.0, va)
        kt.append(state.thrust_coefficient)
        kq.append(state.torque_coefficient)
        eta.append(state.efficiency)
        rows.append({"J": float(j), **{k: float(v) for k, v in state.as_dict().items()}})
    ax.plot(js, kt, "o-", label="$K_T$")
    ax.plot(js, kq, "s-", label="$K_Q$")
    ax2.plot(js, eta, "^--", color="tab:green", label=r"$\eta$")
    ax.set_xlabel("advance ratio J")
    ax.set_ylabel("$K_T$ / $K_Q$")
    ax2.set_ylabel(r"$\eta$")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)
    ax.set_title("BEM open-water diagram of the baseline propeller")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rows


def _fig_thin_ice(result, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    mask = np.ones_like(result.time, dtype=bool)
    ax.plot(result.time[mask], result.torque_ice[mask], lw=0.8, color="tab:red")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ice torque Q [N m]")
    ax.set_title(f"Thin-ice bore-through (h={result.config.ice.thickness:.2f} m, "
                 f"$v_f$={result.config.feed_velocity*1e3:.2f} mm/s)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "propeller_ice"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    plotted = []
    for rpm in RPM_CASES:
        for thickness in THICKNESS_CASES:
            cfg = _case_config(rpm, thickness)
            result = simulate_milling(cfg)
            name = f"n{rpm:.0f}_h{thickness:.2f}"
            record = _case_record(name, result)
            records.append(record)
            plotted.append((result, f"n={rpm:.0f} rpm, h={thickness:.2f} m"))
            print(f"[case] {name}: dQ/Q={record['torque_pulsation']['delta_q_over_mean_p95']:.3f} "
                  f"f_peak={record['spectrum']['f_peak_Hz']:.2f} Hz "
                  f"(BPF {record['spectrum']['f_bpf_Hz']:.2f}) "
                  f"eject={record['ejection']['ejection_speed_mean']:.3f} m/s")

    _fig_timeseries(plotted, out / "fig_torque_timeseries.png")
    _fig_spectrum(plotted, out / "fig_spectrum.png")
    _fig_debris(plotted[0][0], out / "fig_debris.png")

    # ---- skew parametric study ------------------------------------------
    skew_sweep = []
    for skew in (0.0, 10.0, 25.0, 40.0):
        geometry = PropellerGeometry(skew_tip_deg=skew)
        cfg = MillingConfig(propeller=geometry, rpm=60.0, advance_ratio=ADVANCE_RATIO,
                            duration=DURATION, spin_up=SPIN_UP)
        result = simulate_milling(cfg)
        pulse = result.torque_pulsation()
        mask = result.steady_mask()
        harmonics = harmonic_amplitudes(result.time[mask], result.torque_ice[mask],
                                        blade_passing_frequency(60.0, 4))
        skew_sweep.append({
            "skew_tip_deg": skew,
            "delta_q_over_mean_p95": pulse["delta_q_over_mean_p95"],
            "delta_q_over_mean_rms": pulse["delta_q_over_mean_rms"],
            "p2_over_p1": float(harmonics["ratios_to_fundamental"][2]),
            "p3_over_p1": float(harmonics["ratios_to_fundamental"][3]),
            "harmonic_fit_r2": float(harmonics["harmonic_fit_r2"]),
            "Q_mean_Nm": pulse["Q_mean_Nm"],
        })
        print(f"[skew] {skew:5.1f} deg: dQ/Q={pulse['delta_q_over_mean_p95']:.3f} "
              f"A2/A1={harmonics['ratios_to_fundamental'][2]:.3f}")
    _fig_skew(skew_sweep, out / "fig_skew_sweep.png")

    # ---- open-water diagram (supporting) --------------------------------
    open_water = _fig_open_water(PropellerGeometry(), out / "fig_open_water.png")

    # ---- thin-ice bore-through -----------------------------------------
    thin = simulate_milling(MillingConfig(rpm=60.0, advance_ratio=ADVANCE_RATIO,
                                          duration=22.0, spin_up=0.5, dt=1.0e-3,
                                          ice=IceBlockGeometry(thickness=0.05)))
    _fig_thin_ice(thin, out / "fig_thin_ice.png")
    mask = thin.steady_mask()
    dry_out = thin.time[thin.torque_ice < 0.02 * np.max(thin.torque_ice)]
    bore_record = {
        "thickness_m": thin.config.ice.thickness,
        "feed_velocity_m_s": thin.config.feed_velocity,
        "expected_bore_through_s": thin.config.ice.thickness / thin.config.feed_velocity,
        "measured_bore_through_s": float(dry_out[0]) if dry_out.size else None,
        "mean_torque_before_Nm": float(np.mean(thin.torque_ice[(thin.time > 1.0) & (thin.time < 10.0)])),
        "mean_torque_after_Nm": float(np.mean(thin.torque_ice[thin.time > 18.0])),
    }
    print(f"[bore-through] {bore_record}")

    # ---- timestep sensitivity ------------------------------------------
    base = simulate_milling(_case_config(60.0, 0.30))
    half = simulate_milling(_case_config(60.0, 0.30, dt=DURATION / 16000.0))
    timestep = {
        "dt_base_s": base.config.dt,
        "dt_half_s": half.config.dt,
        "delta_q_base": base.torque_pulsation()["delta_q_over_mean_p95"],
        "delta_q_half": half.torque_pulsation()["delta_q_over_mean_p95"],
        "delta_q_rel_change": abs(half.torque_pulsation()["delta_q_over_mean_p95"]
                                  - base.torque_pulsation()["delta_q_over_mean_p95"])
        / base.torque_pulsation()["delta_q_over_mean_p95"],
        "Q_mean_rel_change": abs(half.torque_pulsation()["Q_mean_Nm"]
                                 - base.torque_pulsation()["Q_mean_Nm"])
        / base.torque_pulsation()["Q_mean_Nm"],
        "eject_base": base.ejection_statistics()["ejection_speed_mean"],
        "eject_half": half.ejection_statistics()["ejection_speed_mean"],
    }
    print(f"[timestep] {timestep}")

    # ---- strip-count convergence ---------------------------------------
    strip_rows = []
    for n_strips in (24, 32, 48):
        cfg = MillingConfig(rpm=60.0, advance_ratio=ADVANCE_RATIO, duration=2.5,
                            spin_up=0.5, dt=1.0e-3, n_strips=n_strips,
                            ice=IceBlockGeometry(thickness=0.30))
        result = simulate_milling(cfg)
        pulse = result.torque_pulsation()
        strip_rows.append({
            "n_strips": n_strips,
            "Q_mean_Nm": pulse["Q_mean_Nm"],
            "delta_q_over_mean_p95": pulse["delta_q_over_mean_p95"],
            "ejection_speed_mean": result.ejection_statistics()["ejection_speed_mean"],
        })
    print(f"[strips] {strip_rows}")

    # ---- debris integrator vs closed form ------------------------------
    import numpy as _np

    from tensordem.propeller_ice.milling import drag_relax_analytic, drag_relax_step
    v0 = _np.array([1.4, 0.2, 0.1])
    u = _np.array([0.0, 0.0, 0.58])
    k_drag, dt_drag, steps = 27.9, 5.0e-4, 400
    v = v0.copy()
    for _ in range(steps):
        v = drag_relax_step(v - u, k_drag, dt_drag) + u
    closed = drag_relax_analytic(v0, u, k_drag, steps * dt_drag)
    drag_check = {
        "relative_error": float(_np.max(_np.abs(v - closed) / _np.abs(closed))),
        "steps": steps,
        "dt_s": dt_drag,
    }
    print(f"[drag] {drag_check}")

    payload = {
        "tool": "tensordem.propeller_ice",
        "campaign_node": "IN34-W3-N05",
        "case_matrix": {"rpm": list(RPM_CASES), "h_ice_m": list(THICKNESS_CASES),
                        "advance_ratio": ADVANCE_RATIO, "duration_s": DURATION,
                        "spin_up_s": SPIN_UP},
        "records": records,
        "skew_sweep": skew_sweep,
        "open_water_curve": open_water,
        "bore_through": bore_record,
        "timestep_sensitivity": timestep,
        "strip_convergence": strip_rows,
        "drag_closed_form_check": drag_check,
        "acceptance": {
            "delta_q_over_mean_band": [0.10, 0.30],
            "delta_q_values": [r["torque_pulsation"]["delta_q_over_mean_p95"] for r in records],
            "delta_q_pass": [0.10 <= r["torque_pulsation"]["delta_q_over_mean_p95"] <= 0.30
                             for r in records],
            "bpf_peak_pass": [bool(r["spectrum"]["peak_matches_bpf"]) for r in records],
            "harmonic_ratio_p2": [r["harmonic_fit"]["ratios_to_fundamental"]["2"] for r in records],
            "ejection_speed_values": [r["ejection"]["ejection_speed_mean"] for r in records],
            "ejection_band_m_s": [0.5, 2.0],
            "ejection_pass": [0.5 <= r["ejection"]["ejection_speed_mean"] <= 2.0 for r in records],
        },
    }
    (out / "milling_cases.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out / 'milling_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())