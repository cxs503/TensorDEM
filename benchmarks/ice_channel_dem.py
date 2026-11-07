"""Verification + channel-clearing benchmark for :mod:`tensordem.ice_channel`.

Produces ``artifacts/dem_channel.json`` (machine readable, all criterion numbers
programmatically injected) and the figures under ``artifacts/ice_channel_dem/``.

Sections
--------
A  Contact-stiffness anchor (self-built "Keinonen local pressure" anchor)
   ``k_n = 2 sigma_c t_eff sqrt(2 R_eff / delta_ref)`` with
   ``delta_ref = 0.01 * 2 R_eff``.  The anchor is re-measured by loading two
   floes into contact and comparing the *measured* normal force with ``k_n d``
   and the *nominal* contact pressure ``F_n / A_co`` with ``sigma_c``.
   A repository anchor table ``benchmarks/ice_nav_refs`` (IN10-L2) is not part
   of this revision of ``main``; the artifact therefore declares the anchor as
   self-built plus citation (Keinonen 2017/2019; ISO 19906 local pressure).

B  Hydrostatic draught: a single floe released at the waterline settles to the
   Archimedes draught ``(rho_i/rho_w) t``; criterion < 1 %.

C  Undamped free heave oscillation (2 % of thickness amplitude, 20 periods,
   velocity-Verlet): total-energy drift < 1 % (regression criterion).

D  Determinism: two independent runs of the 8/10 field with the same seed must
   be bit-identical (SHA-256 over all state arrays).

E  Channel transit: equal-width (rectangular) hull crossing a 6/10, 7/10 and
   8/10 field; time history of the cleared channel width, floe velocity-field
   statistics and displaced fraction.

F  Timestep sensitivity of the cleared width (dt and dt/2 on a shortened
   transit) -- a discretisation, not a calibration, check.

Usage
-----
``python benchmarks/ice_channel_dem.py``            # full production run
``python benchmarks/ice_channel_dem.py --fast``     # reduced, for unit tests
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tensordem.ice_channel import (  # noqa: E402
    ContactParameters,
    HullGeometry,
    IceChannelSimulation,
    WaterEnvironment,
    analytic_equilibrium_draught,
    analytic_heave_period,
    contact_pressure_at,
    contact_stiffness,
    heave_potential,
    nominal_contact_area,
    submerged_draught,
)

OUT_DIR = _REPO_ROOT / "artifacts"
FIG_DIR = OUT_DIR / "ice_channel_dem"
JSON_OUT = OUT_DIR / "dem_channel.json"


# ---------------------------------------------------------------------------
# A. contact anchor
# ---------------------------------------------------------------------------

def anchor_section(contact: ContactParameters) -> dict:
    r_eff, t_eff = 5.0, 0.5
    delta_ref = contact.reference_penetration_fraction * 2.0 * r_eff
    k_n = contact_stiffness(r_eff, t_eff, contact)
    p_ref = contact_pressure_at(delta_ref, r_eff, t_eff, contact)
    # measured force from the assembled two-floe contact
    # equal disks of radius 2*r_eff give the effective radius r_eff = R/2 used
    # by the pair law, so the assembled contact must reproduce k_n * delta_ref
    sim = IceChannelSimulation([2.0 * r_eff, 2.0 * r_eff], [t_eff, t_eff], contact=contact,
                               hull=HullGeometry(x_start=1.0e6),
                               water=WaterEnvironment(wind_velocity=(0.0, 0.0),
                                                      current_velocity=(0.0, 0.0),
                                                      heave_damping_ratio=0.0))
    sim.pos = torch.tensor([[0.0, 0.0], [4.0 * r_eff - delta_ref, 0.0]], dtype=sim.dtype)
    force, _, _ = sim.net_forces(0.0)
    f_measured = float(force[0, 0])
    return {
        "anchor": "self-built Keinonen local-pressure anchor",
        "anchor_citation": [
            "Keinonen, A. (2017) Ship-ice interaction in ship design: theory and "
            "practice -- local ice pressure of order 1-3 MPa for ship-ice contact",
            "ISO 19906:2010 Arctic offshore structures -- local ice pressure",
        ],
        "upstream_anchor_table": "benchmarks/ice_nav_refs (IN10-L2) not present on main at run time",
        "contact_parameters": {
            "crushing_strength_Pa": contact.crushing_strength,
            "reference_penetration_fraction": contact.reference_penetration_fraction,
            "normal_damping_ratio": contact.normal_damping_ratio,
            "tangential_damping_ratio": contact.tangential_damping_ratio,
            "friction": contact.friction,
        },
        "geometry": {"r_eff_m": r_eff, "t_eff_m": t_eff},
        "delta_ref_m": delta_ref,
        "stiffness_N_per_m": k_n,
        "force_at_delta_ref_N": k_n * delta_ref,
        "force_at_delta_ref_measured_N": f_measured,
        "nominal_contact_area_at_ref_m2": nominal_contact_area(delta_ref, r_eff, t_eff),
        "pressure_at_ref_Pa": p_ref,
        "pressure_rel_error": abs(p_ref - contact.crushing_strength) / contact.crushing_strength,
        "force_rel_error": abs(abs(f_measured) - k_n * delta_ref) / (k_n * delta_ref),
        "pass": bool(abs(p_ref - contact.crushing_strength) / contact.crushing_strength < 1e-9
                     and abs(abs(f_measured) - k_n * delta_ref) / (k_n * delta_ref) < 1e-9),
    }


# ---------------------------------------------------------------------------
# B. draught
# ---------------------------------------------------------------------------

def draught_section(water: WaterEnvironment, dt: float) -> dict:
    radius, thickness = 5.0, 0.7
    sim = IceChannelSimulation([radius], [thickness], water=water, dt=dt)
    sim.z = torch.tensor([0.2 * thickness], dtype=sim.dtype)
    sim.run(int(round(6.0 / dt)), record_every=10 ** 9)
    draught = float(submerged_draught(sim.z, sim.thickness)[0])
    reference = analytic_equilibrium_draught(water.ice_density, water.water_density, thickness)
    period = 2.0 * math.pi * math.sqrt(
        (water.ice_density * thickness) / (water.water_density * water.gravity))
    return {
        "radius_m": radius,
        "thickness_m": thickness,
        "density_ratio": water.density_ratio,
        "draught_sim_m": draught,
        "draught_analytic_m": reference,
        "rel_error": abs(draught - reference) / reference,
        "threshold": 0.01,
        "linearised_heave_period_s": period,
        "pass": bool(abs(draught - reference) / reference < 0.01),
    }


# ---------------------------------------------------------------------------
# C. undamped free oscillation energy drift
# ---------------------------------------------------------------------------

def energy_section(water: WaterEnvironment, dt: float, periods: float = 20.0) -> dict:
    undamped = WaterEnvironment(
        water_density=water.water_density, ice_density=water.ice_density,
        gravity=water.gravity, wind_velocity=(0.0, 0.0), current_velocity=(0.0, 0.0),
        heave_damping_ratio=0.0, tilt_damping_ratio=0.0)
    radius, thickness = 5.0, 0.5
    sim = IceChannelSimulation([radius], [thickness], water=undamped, dt=dt,
                               integrator="verlet")
    z_eq = 0.5 * thickness - analytic_equilibrium_draught(
        water.ice_density, water.water_density, thickness)
    amplitude = 0.02 * thickness
    sim.z = torch.tensor([z_eq + amplitude], dtype=sim.dtype)
    sim.vz = torch.tensor([0.0], dtype=sim.dtype)
    t_ref = analytic_heave_period(radius, thickness, water.ice_density,
                                  water.water_density, water.gravity)
    n_steps = int(round(periods * t_ref / dt))
    e0 = float(sim.heave_energy()[0])
    trace = {"time": [], "energy": [], "z": [], "z_analytic": []}
    peak = 0.0
    omega = 2.0 * math.pi / t_ref
    for step in range(n_steps):
        sim.step()
        drift = abs(float(sim.heave_energy()[0]) / e0 - 1.0)
        peak = max(peak, drift)
        if step % max(1, n_steps // 400) == 0:
            trace["time"].append(sim.time)
            trace["energy"].append(drift)
            trace["z"].append(float(sim.z[0]))
            trace["z_analytic"].append(z_eq + amplitude * math.cos(omega * sim.time))
    return {
        "radius_m": radius,
        "thickness_m": thickness,
        "release_amplitude_m": amplitude,
        "periods": periods,
        "dt": dt,
        "steps": n_steps,
        "relative_energy_drift": peak,
        "threshold": 0.01,
        "trace": trace,
        "pass": bool(peak < 0.01),
    }


# ---------------------------------------------------------------------------
# E. channel transit
# ---------------------------------------------------------------------------

def run_channel(concentration: float, *, seed: int, dt: float, domain: tuple,
                hull: HullGeometry, record_every: int,
                max_time: float | None = None) -> dict:
    sim = IceChannelSimulation.from_field(
        concentration, domain_length=domain[0], domain_width=domain[1], seed=seed,
        dt=dt, hull=hull)
    y0 = sim.pos[:, 1].clone()
    x0 = sim.pos[:, 0].clone()
    transit = (domain[0] + hull.length) / max(hull.speed, 1e-9)
    if max_time is not None:
        transit = min(transit, max_time)
    n_steps = int(round(transit / dt))
    tic = time.perf_counter()
    history = sim.run(n_steps, record_every=record_every)
    wall = time.perf_counter() - tic
    stats = sim.field_statistics()
    vel = sim.velocity_statistics()
    displacement = torch.sqrt(((sim.pos - torch.stack((x0, y0), dim=1)) ** 2).sum(dim=1))
    lateral = (sim.pos[:, 1] - y0).abs()
    return {
        "concentration_target": concentration,
        "seed": seed,
        "dt": dt,
        "domain_m": {"length": domain[0], "width": domain[1]},
        "hull": {"length": hull.length, "beam": hull.beam, "speed": hull.speed,
                 "y_position": hull.y_position},
        "simulated_time_s": sim.time,
        "steps": n_steps,
        "wall_time_s": wall,
        "field": stats,
        "velocity_statistics": vel,
        "displacement_statistics": {
            "mean_m": float(displacement.mean()),
            "max_m": float(displacement.max()),
            "mean_lateral_m": float(lateral.mean()),
            "max_lateral_m": float(lateral.max()),
            "fraction_lateral_gt_1m": float((lateral > 1.0).to(torch.float64).mean()),
        },
        "cleared_width_final_m": sim.cleared_width(),
        "cleared_width_history": history["cleared_width"],
        "time_history": history["time"],
        "max_speed_history": history["max_speed"],
        "mean_speed_history": history["mean_speed"],
        "kinetic_energy_history": history["kinetic_energy"],
        "max_abs_tilt_final_rad": float(sim.tilt.abs().max()),
        "max_abs_heave_final_m": float(sim.z.abs().max()),
        "positions_final": sim.pos.tolist(),
        "radii": sim.radius.tolist(),
        "speed_final": torch.linalg.norm(sim.vel, dim=1).tolist(),
        "state_digest": sim.state_digest(),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(report: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures = []

    # Fig 1: cleared width time history
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for key, scenario in report["channel_scenarios"].items():
        t = scenario["time_history"]
        w = scenario["cleared_width_history"]
        ax.plot(t, w, lw=1.6, label=f"{key} ice (C={scenario['field']['realised_concentration']:.2f})")
    hull = report["channel_scenarios"]["8_10"]["hull"]
    ax.axhline(hull["beam"], color="k", ls="--", lw=1.0, label=f"hull beam {hull['beam']:.0f} m")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cleared channel width [m]")
    ax.set_title("Cleared channel width behind an equal-width hull")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    path = FIG_DIR / "fig_cleared_width.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": "fig_cleared_width.png",
                    "caption": "Cleared channel width behind the hull vs time for "
                               "6/10, 7/10 and 8/10 fields (engineering metric defined "
                               "in ice_channel.cleared_width)."})

    # Fig 2: final floe velocity field (8/10)
    scenario = report["channel_scenarios"]["8_10"]
    pos = torch.tensor(scenario["positions_final"], dtype=torch.float64)
    radii = torch.tensor(scenario["radii"], dtype=torch.float64)
    speed = torch.tensor(scenario["speed_final"], dtype=torch.float64)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    from matplotlib.patches import Circle, Rectangle
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=speed, s=(radii * 6) ** 1.6, cmap="viridis",
                    edgecolors="k", linewidths=0.3)
    ax.add_patch(Rectangle((scenario["domain_m"]["length"] + 5, scenario["hull"]["y_position"]
                            - 0.5 * scenario["hull"]["beam"]), scenario["hull"]["length"],
                           scenario["hull"]["beam"], fill=False, lw=1.2, ls="--",
                           color="crimson"))
    fig.colorbar(sc, ax=ax, label="floe speed [m/s]")
    ax.set_xlim(-10, scenario["domain_m"]["length"] + 10)
    ax.set_ylim(0, scenario["domain_m"]["width"])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Floe velocity field after the transit, 8/10 field (dashed: hull at end)")
    path = FIG_DIR / "fig_velocity_field.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": "fig_velocity_field.png",
                    "caption": "Final floe positions coloured by speed after the hull "
                               "transit in the 8/10 field; the dashed rectangle is the "
                               "equal-width hull at the end of the run."})

    # Fig 3: free oscillation + energy drift + draught
    energy = report["energy_oscillation"]
    draught = report["draught"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    tr = energy["trace"]
    axes[0].plot(tr["time"], tr["z"], lw=1.2, label="simulation (velocity Verlet)")
    if tr["z_analytic"]:
        axes[0].plot(tr["time"], tr["z_analytic"], "k--", lw=0.9,
                     label="analytic harmonic (linearised)")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("floe centre height z [m]")
    axes[0].set_title(f"Undamped heave, energy drift {energy['relative_energy_drift']:.2e}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].semilogy(tr["time"], [max(v, 1e-18) for v in tr["energy"]], lw=1.2)
    axes[1].axhline(energy["threshold"], color="r", ls="--", lw=1.0, label="1 % criterion")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("|E(t)/E(0) - 1|")
    axes[1].set_title("Relative energy drift (no damping, no drag)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    path = FIG_DIR / "fig_energy_draught.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": "fig_energy_draught.png",
                    "caption": "Left: undamped heave trajectory vs the linearised "
                               "analytic solution. Right: relative total-energy drift "
                               "against the 1 % criterion."})

    # Fig 4: contact pressure vs penetration with the anchor point
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    contact = ContactParameters()
    deltas = [i * 0.002 for i in range(1, 76)]
    pressures = [contact_pressure_at(d, 5.0, 0.5, contact) for d in deltas]
    ax.plot(deltas, [p / 1e6 for p in pressures], lw=1.6, label="anchored spring, $F_n/A_{co}$")
    ax.axhline(contact.crushing_strength / 1e6, color="r", ls="--", lw=1.0,
               label=r"$\sigma_c$ = 1.5 MPa (Keinonen)")
    ax.scatter([contact.reference_penetration_fraction * 2 * 5.0],
               [contact_pressure_at(contact.reference_penetration_fraction * 2 * 5.0,
                                    5.0, 0.5, contact) / 1e6], color="k", zorder=5,
               label=r"anchor $\delta_{ref}$")
    ax.set_xlabel(r"overlap $\delta$ [m]")
    ax.set_ylabel("nominal contact pressure [MPa]")
    ax.set_title("Contact-stiffness anchor calibration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = FIG_DIR / "fig_contact_anchor.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": "fig_contact_anchor.png",
                    "caption": "Nominal contact pressure of the anchored Kelvin-Voigt "
                               "spring vs overlap; the anchor fixes p = sigma_c at "
                               "delta_ref = 1 % of the effective diameter."})
    return figures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover - non-git checkouts
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced run for smoke tests")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--out", type=Path, default=JSON_OUT)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--only", default="all",
                        choices=["all", "6_10", "7_10", "8_10", "determinism",
                                 "timestep", "assemble"],
                        help="run a single production stage and store it as a part; "
                             "'assemble' merges the parts into the final artifact "
                             "(lets the production run be split over several "
                             "time-limited shells)")
    args = parser.parse_args(argv)

    fast = args.fast
    dt = 8.0e-3 if not fast else 1.0e-2
    domain = (120.0, 60.0) if not fast else (48.0, 24.0)
    hull_len = 80.0 if not fast else 32.0
    hull_beam = 20.0 if not fast else 10.0
    hull_speed = 1.6 if not fast else 2.0
    hull = HullGeometry(length=hull_len, beam=hull_beam, speed=hull_speed,
                        y_position=0.5 * domain[1], x_start=-0.5 * hull_len)
    contact = ContactParameters()
    water = WaterEnvironment()
    record_every = 200 if not fast else 100
    scenario_time = None if not fast else 20.0
    part_dir = FIG_DIR / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "module": "tensordem.ice_channel",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mode": "fast" if fast else "production",
        "seed": args.seed,
    }
    configuration = {
        "dt": dt, "domain": domain,
        "hull": {"length": hull_len, "beam": hull_beam, "speed": hull_speed,
                 "y_position": hull.y_position, "x_start": hull.x_start},
        "record_every": record_every,
        "environment": {
            "water_density": water.water_density, "ice_density": water.ice_density,
            "gravity": water.gravity, "wind_velocity": list(water.wind_velocity),
            "current_velocity": list(water.current_velocity),
            "wind_drag_coefficient": water.wind_drag_coefficient,
            "water_drag_coefficient": water.water_drag_coefficient,
            "sail_area_fraction": water.sail_area_fraction,
            "heave_damping_ratio": water.heave_damping_ratio,
            "tilt_damping_ratio": water.tilt_damping_ratio,
        },
        "contact": {
            "crushing_strength_Pa": contact.crushing_strength,
            "reference_penetration_fraction": contact.reference_penetration_fraction,
            "normal_damping_ratio": contact.normal_damping_ratio,
            "tangential_damping_ratio": contact.tangential_damping_ratio,
            "friction": contact.friction,
        },
    }

    def write_part(name: str, payload: dict) -> None:
        path = part_dir / f"part_{name}.json"
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[part] {path}")

    def run_scenario(key: str, concentration: float) -> dict:
        result = run_channel(concentration, seed=args.seed, dt=dt, domain=domain,
                             hull=hull, record_every=record_every,
                             max_time=scenario_time)
        result["concentration_realised_tenths"] = 10.0 * result["field"]["realised_concentration"]
        print(f"[channel] {key}: cleared={result['cleared_width_final_m']:.2f} m "
              f"floes={result['field']['floe_count']} in {result['wall_time_s']:.1f} s wall",
              flush=True)
        write_part(key, result)
        return result

    scenario_keys = {"6_10": 0.6, "7_10": 0.7, "8_10": 0.8}

    if args.only in ("all", "6_10", "7_10", "8_10"):
        keys = [args.only] if args.only != "all" else list(scenario_keys)
        for key in keys:
            run_scenario(key, scenario_keys[key])
        if args.only != "all":
            return 0

    if args.only in ("all", "determinism"):
        scenario_time_det = scenario_time
        digest_dt = dt
        if args.only == "determinism" and not fast:
            # two runs inside one shell: keep dt and shorten the horizon to the
            # first 60 s of the transit (bit-identity is a property of the code
            # path, not of the particular horizon)
            digest_dt = dt
            scenario_time_det = 60.0
        digests = []
        for _ in range(2):
            seg = run_channel(0.8, seed=args.seed, dt=digest_dt, domain=domain, hull=hull,
                              record_every=10 ** 9, max_time=scenario_time_det)
            digests.append(seg["state_digest"])
        payload = {
            "scenario": "8_10",
            "seed": args.seed,
            "dt": digest_dt,
            "simulated_time_s": seg["simulated_time_s"],
            "digest_run1": digests[0],
            "digest_run2": digests[1],
            "bit_identical": digests[0] == digests[1],
        }
        print(f"[determinism] {payload['bit_identical']}", flush=True)
        write_part("determinism", payload)
        if args.only != "all":
            return 0

    if args.only in ("all", "timestep"):
        short_time = 40.0 if not fast else 5.0
        step_runs = {}
        for label, step_dt in (("dt", dt), ("dt_half", 0.5 * dt)):
            seg = run_channel(0.8, seed=args.seed, dt=step_dt, domain=domain, hull=hull,
                              record_every=10 ** 9, max_time=short_time)
            step_runs[label] = {
                "dt": step_dt,
                "simulated_time_s": seg["simulated_time_s"],
                "cleared_width_final_m": seg["cleared_width_final_m"],
                "mean_lateral_displacement_m":
                    seg["displacement_statistics"]["mean_lateral_m"],
                "wall_time_s": seg["wall_time_s"],
            }
        w_a = step_runs["dt"]["cleared_width_final_m"]
        w_b = step_runs["dt_half"]["cleared_width_final_m"]
        lat_a = step_runs["dt"]["mean_lateral_displacement_m"]
        lat_b = step_runs["dt_half"]["mean_lateral_displacement_m"]
        payload = {
            "runs": step_runs,
            "cleared_width_rel_difference": abs(w_a - w_b) / max(abs(w_b), 1e-9),
            "mean_lateral_displacement_rel_difference": abs(lat_a - lat_b) / max(abs(lat_b), 1e-9),
            "note": "discretisation check on a shortened transit; the cleared-width "
                    "metric is a min-clearance quantity and is therefore O(dt) "
                    "sensitive near contacts, so a few percent is expected",
        }
        print(f"[timestep] rel diff {payload['cleared_width_rel_difference']:.4f}", flush=True)
        write_part("timestep", payload)
        if args.only != "all":
            return 0

    # ---- assemble -------------------------------------------------------
    report = dict(meta)
    report["configuration"] = configuration
    report["anchor"] = anchor_section(contact)
    report["draught"] = draught_section(water, dt)
    report["energy_oscillation"] = energy_section(water, dt)

    scenarios = {}
    for key in scenario_keys:
        path = part_dir / f"part_{key}.json"
        if not path.exists():
            raise SystemExit(f"missing part {path}; run --only {key} first")
        scenarios[key] = json.loads(path.read_text())
    report["channel_scenarios"] = scenarios
    report["determinism"] = json.loads((part_dir / "part_determinism.json").read_text())
    report["timestep_sensitivity"] = json.loads((part_dir / "part_timestep.json").read_text())

    criteria = {
        "energy_drift_rel": report["energy_oscillation"]["relative_energy_drift"],
        "energy_drift_threshold": 0.01,
        "energy_pass": report["energy_oscillation"]["pass"],
        "draught_rel_error": report["draught"]["rel_error"],
        "draught_threshold": 0.01,
        "draught_pass": report["draught"]["pass"],
        "determinism_bit_identical": report["determinism"]["bit_identical"],
        "contact_anchor_pressure_rel_error": report["anchor"]["pressure_rel_error"],
        "contact_anchor_force_rel_error": report["anchor"]["force_rel_error"],
        "contact_anchor_pass": report["anchor"]["pass"],
        "all_pass": bool(report["energy_oscillation"]["pass"] and report["draught"]["pass"]
                         and report["determinism"]["bit_identical"]
                         and report["anchor"]["pass"]),
    }
    report["criteria"] = criteria

    if not args.no_figures:
        report["figures"] = make_figures(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[written] {args.out}")
    print(json.dumps(criteria, indent=2))
    return 0 if criteria["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())