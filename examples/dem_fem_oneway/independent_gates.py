"""Independent verification gates for the DEM-FEM one-way coupling example.

Runs the shipped coupling (``run.py``) and re-checks its physics against
*external* anchors instead of the shipped self-comparisons:

===========================================  ===========================  ================
gate                                         reference                    independence
===========================================  ===========================  ================
G1 contact duration                          ``π sqrt(m/k)``              analytic
G2 peak contact force                        ``v sqrt(m k)``              analytic
G3 impulse                                   ``(1+e) m v``                analytic
G4 impulse vs particle momentum change       ``m (v_f - v_i)``            cross_side
G5 beam fundamental frequency (dt-refined)   ``(π/L)² sqrt(EI/ρA)``       analytic
G6 peak centre deflection (dynamic)          continuum modal superposition independent impl
G7 deflection history (windowed RMS)         continuum modal superposition independent impl
G8 beam energy at end of the forced phase    continuum modal superposition independent impl
===========================================  ===========================  ================

The shipped ``impulse ledger`` and ``work-energy`` gates are reported with
``independence="self_comparison"``: they are consistency checks, not evidence.

Usage
-----
    python examples/dem_fem_oneway/independent_gates.py

Outputs
-------
- ``artifacts/dem_fem_coupling/independent_gates.json``
- ``docs/figures/dem_fem_independent_gates.png``
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from tensordem.dem_fem_gates import (  # noqa: E402
    beam_natural_frequencies,
    evaluate_gates,
    hookean_impact,
    modal_response,
    modal_static_deflection,
    summarise,
)


def _load_run_module():
    spec = importlib.util.spec_from_file_location("dem_fem_run", _HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def _fundamental_from_history(signal, dt: float) -> dict:
    """Dominant frequency of a free-vibration record (Hann FFT + parabolic peak)."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if n < 16:
        raise ValueError("need >= 16 samples to resolve a frequency")
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(x * window))
    freqs = np.fft.rfftfreq(n, d=dt)
    k = int(np.argmax(spectrum[1:]) + 1)
    if 0 < k < spectrum.size - 1:
        y0, y1, y2 = spectrum[k - 1], spectrum[k], spectrum[k + 1]
        denom = (y0 - 2.0 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
    else:
        delta = 0.0
    df = freqs[1] - freqs[0]
    return {"frequency_hz": float(freqs[k] + delta * df),
            "bin_frequency_hz": float(freqs[k]),
            "peak_bin": k,
            "n_samples": n,
            "dt": dt,
            "samples_per_period": float(1.0 / (freqs[k] * dt)) if freqs[k] > 0 else float("nan")}


def _fem_at_refinement(run, config, force, times, factor: int, t1: float) -> dict:
    """Drive the FEM with the same force history sampled ``factor`` times finer."""
    if factor == 1:
        t_fine, f_fine = times, force
    else:
        dt = times[1] - times[0]
        dt_fine = dt / factor
        t_fine = np.arange(times[0], times[-1] + 0.5 * dt_fine, dt_fine)
        f_fine = np.interp(t_fine, times, force)
    dt_use = float(t_fine[1] - t_fine[0])
    ringdown_steps = int(math.ceil(5.0 * t1 / dt_use))
    fem = run.run_fem(f_fine, t_fine, config, ringdown_steps=ringdown_steps)
    ring = np.asarray(fem["ring_down_deflection"], dtype=np.float64)
    fit = _fundamental_from_history(ring, dt_use)
    fit.update({
        "refinement": factor,
        "registered_dt": dt_use,
        "n_forced_samples": int(t_fine.size),
        "ringdown_samples": ringdown_steps,
        "peak_deflection_m": fem["peak_deflection_m"],
        "impulse_fem_Ns": float(fem["impulse_fem_Ns"]),
        "work_done_J": float(fem["work_done_J"]),
        "energy_end_of_forced_J": float(fem["energy_history"][-1]["total"]),
        "max_energy_J": float(max(e["total"] for e in fem["energy_history"])),
        "deflection": np.asarray(fem["deflection_history"], dtype=np.float64),
        "ring_down": ring,
        "t": t_fine,
    })
    return fit


def main() -> int:
    run = _load_run_module()
    config = run.CouplingConfig()

    # ---------------------------------------------------------------- DEM --
    dem = run.run_dem(config)
    times = np.asarray(dem["time_history"], dtype=np.float64)
    force = np.asarray(dem["force_history"], dtype=np.float64)

    beam = run.BeamConfig(
        length=config.beam_length, width=config.beam_width,
        height=config.beam_height, young_modulus=config.beam_young_modulus,
        density=config.beam_density, n_elements=config.beam_n_elements,
        bc=config.beam_bc, damping_ratio=config.beam_damping,
    )
    ei = config.beam_young_modulus * beam.inertia
    rho_a = beam.mass_per_length
    omega1 = float(beam_natural_frequencies(ei, rho_a, config.beam_length)[0])
    f1 = omega1 / (2.0 * math.pi)
    t1 = 1.0 / f1

    # ------------------------------------------- FEM at the coupling dt, ... --
    coarse = _fem_at_refinement(run, config, force, times, 1, t1)
    convergence = [coarse]
    for factor in (2, 4):
        convergence.append(_fem_at_refinement(run, config, force, times, factor, t1))
    finest = convergence[-1]
    for entry in convergence:
        entry["f1_rel_error"] = entry["frequency_hz"] / f1 - 1.0

    # ------------------------------------------------------------ anchors --
    m = float(dem["particle_mass_kg"])
    k_eff = float(config.platen_stiffness)      # flat-platen reaction spring
    v0 = float(config.initial_velocity)         # sign carries the direction
    impact = hookean_impact(m, k_eff, abs(v0), restitution=config.restitution)
    impulse_ref = math.copysign(impact["impulse_Ns"], v0)

    # Extend the drive with zeros over the free ring-down: both sides then share
    # one measurement window (forced phase + ring-down at the coupling dt).
    dt = float(times[1] - times[0])
    ringdown_steps = int(math.ceil(5.0 * t1 / dt))
    t_ext = np.concatenate([times, times[-1] + dt * np.arange(1, ringdown_steps + 1)])
    f_ext = np.concatenate([force, np.zeros(ringdown_steps)])

    modal = modal_response(
        f_ext, t_ext, ei=ei, mass_per_length=rho_a, length=config.beam_length,
        x_load=config.beam_length / 2.0, x_probe=config.beam_length / 2.0,
        n_modes=9, bc=config.beam_bc,
    )

    fem_hist = np.concatenate([coarse["deflection"], coarse["ring_down"]])
    n_forced = coarse["deflection"].size
    modal_forced = modal["deflection"][:n_forced]

    static_ref = modal_static_deflection(float(dem["peak_force_N"]),
                                         config.beam_length, ei, bc=config.beam_bc)
    daf_fem = float(np.max(np.abs(fem_hist))) / static_ref
    daf_modal = modal["peak_deflection"] / static_ref

    fem_energy_end = float(coarse["energy_end_of_forced_J"])
    modal_energy_end = float(modal["energy"][n_forced - 1])

    # --------------------------------------------------------------- gates --
    gates = evaluate_gates([
        ("G1 contact duration", "s", impact["contact_time_s"],
         dem["contact_duration_s"], 0.05, "analytic"),
        ("G2 peak contact force", "N", impact["peak_force_N"],
         dem["peak_force_N"], 0.03, "analytic"),
        ("G3 impulse", "N s", impulse_ref, dem["impulse_dem_Ns"], 0.01, "analytic"),
        ("G4 momentum vs recorded impulse", "N s",
         m * (dem["v_final_ms"] - dem["v_initial_ms"]),
         -dem["impulse_dem_Ns"], 0.01, "cross_side"),
        ("G5 beam f1 (dt/4-refined FEM)", "Hz", f1,
         finest["frequency_hz"], 0.02, "analytic"),
        ("G6 peak centre deflection", "m", modal["peak_deflection"],
         coarse["peak_deflection_m"], 0.05, "independent_implementation"),
        ("G7 deflection history RMS", "m",
         float(np.sqrt(np.mean(modal["deflection"] ** 2))),
         float(np.sqrt(np.mean(fem_hist ** 2))), 0.10,
         "independent_implementation"),
        ("G8 peak beam mechanical energy", "J",
         modal["max_energy"], coarse["max_energy_J"], 0.05,
         "independent_implementation"),
        # shipped gates, demoted: they cannot fail for a physical reason
        ("S1 shipped impulse ledger", "N s", coarse["impulse_fem_Ns"],
         dem["impulse_dem_Ns"], 1e-10, "self_comparison"),
        ("S2 shipped work-energy", "J", coarse["work_done_J"],
         fem_energy_end, 1e-8, "self_comparison"),
    ])
    summary = summarise(gates)

    result = {
        "scenario": "dem_fem_oneway_independent_gates",
        "config": {
            "particle_radius_m": config.particle_radius,
            "particle_mass_kg": m,
            "contact_stiffness_N_m": config.contact_stiffness,
            "platen_stiffness_N_m": config.platen_stiffness,
            "initial_velocity_ms": v0,
            "restitution": config.restitution,
            "beam_length_m": config.beam_length,
            "beam_width_m": config.beam_width,
            "beam_height_m": config.beam_height,
            "beam_ei_N_m2": ei,
            "beam_mass_per_length_kg_m": rho_a,
            "beam_n_elements": config.beam_n_elements,
            "beam_bc": config.beam_bc,
        },
        "bandwidth": {
            "f1_hz_analytic": f1,
            "t1_s": t1,
            "contact_time_s_analytic": impact["contact_time_s"],
            "contact_time_over_t1": impact["contact_time_s"] / t1,
            "shipped_docstring_claim_tc_over_t1": 11.0,
            "shipped_docstring_claim_daf": 1.035,
        },
        "dem": {
            "n_steps": dem["n_steps"],
            "dt": dt,
            "peak_force_N": dem["peak_force_N"],
            "peak_force_time_s": dem["peak_force_time_s"],
            "contact_duration_s": dem["contact_duration_s"],
            "impulse_Ns": dem["impulse_dem_Ns"],
            "v_initial_ms": dem["v_initial_ms"],
            "v_final_ms": dem["v_final_ms"],
        },
        "fem_dt_convergence": [
            {k: v for k, v in entry.items()
             if k not in ("deflection", "ring_down", "t")}
            for entry in convergence
        ],
        "modal": {
            "n_modes": modal["n_modes"],
            "peak_deflection_m": modal["peak_deflection"],
            "peak_deflection_time_s": modal["peak_deflection_time"],
            "energy_end_of_forced_J": modal_energy_end,
        },
        "energy_conditioning": {
            "peak_energy_modal_J": modal["max_energy"],
            "peak_energy_fem_J": coarse["max_energy_J"],
            "end_of_forced_modal_J": modal_energy_end,
            "end_of_forced_fem_J": fem_energy_end,
            "end_of_forced_residual_over_peak": abs(fem_energy_end - modal_energy_end)
            / modal["max_energy"],
            "note": ("the end-of-forced-phase energy is a difference of quantities of order "
                     "1e-7 J that cancel to 1e-9 J (residual ~0.2% of the peak), so it is "
                     "reported as a diagnostic, not as a gate; the well-conditioned energy "
                     "gate is G8 (peak mechanical energy)"),
        },
        "dynamic_amplification": {
            "static_reference_m": static_ref,
            "daf_fem": daf_fem,
            "daf_modal": daf_modal,
        },
        "summary": summary,
    }

    out_dir = _ROOT / "artifacts" / "dem_fem_coupling"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "independent_gates.json"
    out_path.write_text(json.dumps(result, indent=2))

    fig_path = _ROOT / "docs" / "figures" / "dem_fem_independent_gates.png"
    _plot(fig_path, times, force, t_ext, fem_hist, modal["deflection"],
          coarse["ring_down"], coarse, f1, summary)

    print("DEM-FEM independent gates")
    print("=" * 88)
    for gate in summary["gates"]:
        print(f"  {gate['name']:<38} {gate['independence']:<26} "
              f"rel={gate['rel_error']:+.3e} {'PASS' if gate['pass'] else 'FAIL'}")
    print(f"\n  external gates: {summary['external_pass']}/{summary['n_external']} pass, "
          f"worst |rel|={summary['worst_external_rel_error']:.3e}")
    print("  beam f1 convergence (analytic "
          f"{f1:.2f} Hz): " + ", ".join(
              f"dt/{e['refinement']}: {e['frequency_hz']:.2f} Hz "
              f"({e['f1_rel_error']:+.2%})" for e in convergence))
    print(f"  t_c/T1 = {impact['contact_time_s'] / t1:.3f} "
          f"(shipped docstring claims 11)")
    print(f"  DAF: FEM {daf_fem:.4f} / modal {daf_modal:.4f}")
    print(f"\n  artifact: {out_path}")
    print(f"  figure:   {fig_path}")
    return 0 if summary["all_external_pass"] else 1


def _plot(path: Path, times, force, t_ext, fem_hist, modal_hist, ring,
          coarse, f1_analytic, summary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    ax = axes[0]
    ax.plot(times * 1e3, force, lw=1.4, color="#1f4e79")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("contact force [N]")
    ax.set_title("(a) DEM contact force (FEM input)")

    ax = axes[1]
    ax.plot(t_ext * 1e3, np.asarray(fem_hist) * 1e9, lw=1.4, color="#c00000",
            label="FEM (2-node Hermite)")
    ax.plot(t_ext * 1e3, np.asarray(modal_hist) * 1e9, lw=1.2, ls="--",
            color="#2e7d32", label="modal superposition (independent)")
    ax.axvline(times[-1] * 1e3, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("mid-span deflection [nm]")
    ax.set_title("(b) dynamic response: FEM vs continuum modes")
    ax.legend(fontsize=8)

    ax = axes[2]
    ring = np.asarray(ring, dtype=np.float64)
    dt = float(coarse["registered_dt"])
    ax.plot(np.arange(ring.size) * dt * 1e3, (ring - ring.mean()) * 1e9,
            lw=1.2, color="#1f4e79", label="FEM ring-down")
    ax.set_xlabel("t after impact [ms]")
    ax.set_ylabel("centred deflection [nm]")
    ax.set_title("(c) free ring-down + spectrum")
    ax2 = ax.twinx()
    spec = np.abs(np.fft.rfft((ring - ring.mean()) * np.hanning(ring.size)))
    f = np.fft.rfftfreq(ring.size, d=dt)
    ax2.plot(f, spec, lw=1.0, color="#c00000", alpha=0.7)
    ax2.axvline(f1_analytic, color="#2e7d32", ls="--", lw=1.0)
    ax2.set_ylabel("|FFT|")
    ax2.set_xlim(0, 4.0 * f1_analytic)
    ax.text(f1_analytic * 1.05, spec.max() * 0.8,
            f"analytic f1={f1_analytic:.1f} Hz\nmeasured "
            f"{coarse['frequency_hz']:.1f} Hz (dt/T1={dt * f1_analytic:.2f})",
            fontsize=8, color="#2e7d32")
    ok = summary["all_external_pass"]
    fig.suptitle(
        f"DEM-FEM one-way coupling: independent gates "
        f"{summary['external_pass']}/{summary['n_external']} pass"
        f" ({'PASS' if ok else 'FAIL'}); "
        f"shipped self-comparisons demoted: {summary['n_self_comparison']}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())