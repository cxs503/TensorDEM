#!/usr/bin/env python3
"""Dense broken-ice force-chain / jamming campaign (IN22-N03).

Two modes:

``--shard i/n``
    Run the deterministic subset ``tasks[i::n]`` and write
    ``runs/force_chain_jamming/shard_<i>.json``.  Safe to interrupt: the file is
    rewritten after every finished task.

``--merge``
    Combine the shard files into ``artifacts/force_chain_jamming.json``, fit the
    piecewise-linear / BIC breakpoints, and render the figures into
    ``artifacts/figures``.

Task types
----------
``anchor``   jamming area fraction from the dense (isotropic decompression)
             branch, for frictionless and frictional disks (external hard anchor).
``channel``  one channel transit at a given ice concentration / clearance / seed.
``checks``   the machine-checkable identities: Virial balance, stress symmetry,
             hydrostatic pressure, momentum conservation, digest reproducibility,
             random-packing fabric anisotropy, percolation seed sensitivity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.granular2d import (  # noqa: E402
    DiskConfig,
    DiskSystem,
    analyze_force_network,
    bidisperse_radii,
    box_walls,
    grow_to_fraction,
    jamming_density_isotropic,
    make_channel_system,
    piecewise_bic_breakpoint,
    random_packing_positions,
    run_channel,
)
from tensordem.granular2d.force_chain import fabric_tensor  # noqa: E402
from tensordem.granular2d.stress import (  # noqa: E402
    hydrostatic_check,
    lattice_stress,
    stress_balance,
    stress_invariants,
    virial_tensor,
)

PHI_LEVELS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
SEEDS = (0, 1, 2, 3, 4)
CLEARANCES = (0.10, 0.07, 0.05)  # (width - beam)/2 for width = 0.40
CHANNEL = dict(n_particles=200, mean_radius=0.013, width=0.40, length=0.62,
               ship_length=0.24, ship_bow=0.07, relax_steps=150, dphi=0.02,
               drift_speed=0.20, drag_time=0.010)


# --------------------------------------------------------------------- tasks
def build_tasks() -> list[dict]:
    tasks: list[dict] = []
    for mu, tag in ((0.0, "frictionless"), (0.5, "frictional")):
        tasks.append({"kind": "anchor", "mu": mu, "tag": tag, "seed": 0})
    for phi in PHI_LEVELS:
        for seed in SEEDS:
            tasks.append({"kind": "channel", "phi": phi, "seed": seed, "beam": 0.26,
                          "clearance": 0.07})
    for clearance in CLEARANCES:
        tasks.append({"kind": "channel", "phi": 0.78, "seed": 0,
                      "beam": 0.40 - 2.0 * clearance, "clearance": clearance})
    tasks.append({"kind": "checks", "seed": 0})
    return tasks


def task_channel(task: dict, duration: float) -> dict:
    cfg = dict(CHANNEL)
    drift = cfg.pop("drift_speed")
    drag = cfg.pop("drag_time")
    system, info = make_channel_system(phi=task["phi"], seed=task["seed"],
                                       ship_beam=task["beam"], drag_time=drag,
                                       drift_speed=drift, **cfg)
    out = run_channel(system, drift_speed=drift, duration=duration,
                      area=info["free_area"], sample_every=600, seed=task["seed"])
    keep = {k: v for k, v in info.items() if k not in ("phi_history", "pressure_history")}
    out.update({"kind": "channel", "beam": task["beam"], "clearance": task["clearance"],
                "setup": keep, "digest": system.digest()})
    out.pop("fx_series", None)
    out.pop("speed_series", None)
    return out


def task_anchor(task: dict) -> dict:
    res = jamming_density_isotropic(mu=task["mu"],
                                    mu_rolling=0.0 if task["mu"] == 0.0 else 0.05,
                                    n=400, mean_radius=0.008, seed=task["seed"],
                                    levels=(300.0, 100.0, 30.0, 10.0, 3.0, 1.0),
                                    steps_per_level=2200, eta=150.0)
    res["kind"] = "anchor"
    res["tag"] = task["tag"]
    res["reference_window"] = [0.80, 0.84] if task["mu"] > 0 else [0.83, 0.85]
    return res


def _isotropic_bulk_packing(phi: float, seed: int = 0, mu: float = 0.0, n: int = 180):
    """Relaxed random packing at ``phi`` (no gravity, isotropic preparation).

    The configuration starts from uniformly random (overlapping) positions, so it
    carries no memory of a growth/RSA history; the over-damped relaxation then
    reaches a force-balanced random packing whose bulk fabric is isotropic.
    """
    radii = bidisperse_radii(n, 0.010, seed=seed)
    side = math.sqrt(float(np.sum(math.pi * radii**2)) / phi)
    rng = np.random.default_rng(seed + 3)
    rmax = float(radii.max())
    pos = rmax + rng.random((n, 2)) * (side - 2.0 * rmax)
    cfg = DiskConfig(mu=mu, mu_rolling=0.0, damping_time=3.0e-4, skin=0.003)
    system = DiskSystem(pos, radii, cfg, walls=box_walls(side, side))
    system.step(8000)
    system.zero_velocities()
    system.step(200)
    return system, side


def _bulk_fabric(system, side: float, margin: float, samples: int = 40,
                 stride: int = 100) -> dict:
    """Fabric tensor of the bulk, averaged over ``samples`` snapshots.

    A single snapshot with ~200 contacts has a sampling noise of order
    1/sqrt(M) ~ 0.07 in the anisotropy, so the average is what the
    ``|F_aniso| < 0.02`` criterion must be applied to.
    """
    accumulated = np.zeros((2, 2))
    count = 0
    for _ in range(samples):
        system.step(stride)
        contacts = system.contact_forces()
        kind = np.asarray(contacts["kind"])
        disk = kind == 0
        if not disk.any():
            continue
        i = np.asarray(contacts["i"])[disk]
        j = np.asarray(contacts["j"])[disk]
        normals = np.asarray(contacts["n"])[disk]
        mid = 0.5 * (system.pos[i] + system.pos[j])
        inside = ((mid > margin) & (mid < side - margin)).all(axis=1)
        nrm = normals[inside]
        if len(nrm):
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-30)[:, None]
            accumulated += np.einsum("ma,mb->ab", nrm, nrm)
            count += len(nrm)
    if count == 0:
        return {"tensor": np.zeros((2, 2)), "anisotropy": 0.0, "eigenvalues": np.zeros(2),
                "n_contacts": 0}
    tensor = accumulated / count
    eig = np.sort(np.maximum(np.linalg.eigvalsh(0.5 * (tensor + tensor.T)), 0.0))
    aniso = float((eig[-1] - eig[0]) / max(float(eig.sum()), 1e-30))
    return {"tensor": tensor, "anisotropy": aniso, "eigenvalues": eig,
            "n_contacts": int(count)}


def task_checks() -> dict:
    out: dict[str, object] = {"kind": "checks"}

    # (a)(b) Virial identity + stress symmetry on a settled gravity packing
    n = 180
    radii = bidisperse_radii(n, 0.011, seed=3)
    side = math.sqrt(float(np.sum(math.pi * radii**2)) / 0.40)
    pos = random_packing_positions(n, radii, side, side, seed=4)
    # frictionless walls and contacts: the vertical stress then measures the
    # weight per unit width exactly (no side-wall shear transfer)
    cfg = DiskConfig(mu=0.0, mu_rolling=0.0, gravity=(0.0, -9.81), skin=0.003)
    system = DiskSystem(pos, radii, cfg, walls=box_walls(side, side))
    system.step(12000)                                  # free settling under gravity
    # repeated quenches drive the bed to a force-balanced (static) state
    system.config = replace(system.config, damping_time=2.0e-5)
    for _ in range(10):
        system.step(1200)
        system.zero_velocities()
    system.step(400)
    system.contact_forces()
    balance = stress_balance(system)
    out["virial_identity_rel"] = balance["virial_identity_rel"]
    out["virial_identity_static_rel"] = balance["virial_identity_static_rel"]
    out["stress_symmetry_rel"] = balance["stress_symmetry_rel"]
    out["antisymmetric_fraction"] = balance["antisymmetric_fraction"]

    # negative control: double-count one side of every contact and check that the
    # symmetry/balance diagnostics really do detect the corruption
    raw = virial_tensor(system, symmetrize=False)
    table = system.contacts
    kind = np.asarray(table["kind"]) if len(table.get("i", [])) else np.zeros(0)
    if len(kind):
        disk = kind == 0
        i = np.asarray(table["i"])[disk]
        j = np.asarray(table["j"])[disk]
        f = np.asarray(table["f_on_i"])[disk]
        # BUG CLASS: count both particle sides of a contact (r_i + r_j) instead of
        # the branch vector (r_j - r_i).  The Virial route residual must explode.
        w_bad = np.einsum("ma,mb->ab", system.pos[i] + system.pos[j], f)
        w_ext = np.asarray(balance["external_virial"]).reshape(2, 2)
        out["route_negative_control"] = float(
            np.linalg.norm(w_bad - w_ext) / max(np.linalg.norm(w_ext), 1e-30))
        anti = np.array([[0.0, 1.0], [-1.0, 0.0]]) * 0.01 * max(np.linalg.norm(raw), 1e-30)
        corrupted = raw + anti
        out["symmetry_negative_control"] = float(
            np.linalg.norm(corrupted - corrupted.T) / max(np.linalg.norm(corrupted), 1e-30))
    else:
        out["route_negative_control"] = 1.0
        out["symmetry_negative_control"] = 1.0

    # lattice conservation: sum(cell sigma * area) equals the total virial
    field = lattice_stress(system, nx=6, ny=6)
    area = float(np.prod(field["bounds"][2:] - field["bounds"][:2]))
    total = float(np.sum(field["sigma"] * field["cell_area"]))
    out["lattice_sum_rel"] = float(abs(total - float(np.sum(raw) * area)) /
                                  max(abs(float(np.sum(raw) * area)), 1e-30))
    out["lattice_cells_filled"] = int(np.sum(field["counts"] > 0))

    # (c) hydrostatic pressure versus weight per unit width
    hydro = hydrostatic_check(system, n_bands=10, band_index=2)
    out["hydrostatic"] = {k: float(v) for k, v in hydro.items()}
    out["hydrostatic_band_rel_error"] = float(hydro["relative_error"])
    out["hydrostatic_traction_rel_error"] = float(hydro["traction_relative_error"])

    # (g) momentum conservation without external forces
    n2 = 120
    r2 = bidisperse_radii(n2, 0.010, seed=7)
    side2 = math.sqrt(float(np.sum(math.pi * r2**2)) / 0.42)
    pos2 = random_packing_positions(n2, r2, side2, side2, seed=8)
    free = DiskSystem(pos2, r2, DiskConfig(mu=0.3, mu_rolling=0.02, skin=0.003))
    free.config = replace(free.config, gravity=(0.0, 0.0))
    rng = np.random.default_rng(11)
    free.vel = 0.15 * rng.standard_normal((n2, 2))
    p0 = free.momentum()
    free.step(2500)
    p1 = free.momentum()
    scale = float(np.sum(free.mass * np.linalg.norm(free.vel, axis=1))) + 1e-30
    out["momentum_drift_rel"] = float(np.linalg.norm(p1 - p0) / scale)
    out["momentum_drift_abs"] = float(np.linalg.norm(p1 - p0))

    # (h) digest reproducibility
    def digest_run(seed: int) -> str:
        rr = bidisperse_radii(90, 0.010, seed=seed)
        ss = math.sqrt(float(np.sum(math.pi * rr**2)) / 0.40)
        pp = random_packing_positions(90, rr, ss, ss, seed=seed + 1)
        cfg2 = DiskConfig(mu=0.4, mu_rolling=0.03, gravity=(0.0, -9.81), skin=0.003)
        sysd = DiskSystem(pp, rr, cfg2, walls=box_walls(ss, ss))
        sysd.step(1200)
        return sysd.digest()
    d1, d2 = digest_run(21), digest_run(21)
    out["digest"] = d1
    out["digest_repeat"] = d2
    out["digest_identical"] = bool(d1 == d2)
    out["digest_seed_sensitive"] = bool(d1 != digest_run(22))

    # (e) fabric anisotropy of an isotropic random packing (bulk, no gravity)
    iso_system, iso_side = _isotropic_bulk_packing(0.82, seed=0)
    bulk = _bulk_fabric(iso_system, iso_side, margin=4.0 * iso_system.mean_radius)
    out["fabric_anisotropy_bulk"] = bulk["anisotropy"]
    out["fabric_anisotropy_bulk_contacts"] = bulk["n_contacts"]

    # force-chain statistics of the isotropic packing
    net = analyze_force_network(iso_system, bounds=(0.0, iso_side))
    out["isotropic_z"] = net["z_all"]
    out["isotropic_load_bearing_ratio"] = net["load_bearing_ratio"]
    out["isotropic_chain_stats"] = {
        "n_chains": net["chain"]["n_chains"],
        "mean_length": net["chain"]["mean_length"],
        "max_length": net["chain"]["max_length"],
    }
    # force magnitude statistics (heterogeneity of the contact network)
    mags = np.asarray(net["magnitudes"])
    if len(mags):
        out["force_stats"] = {
            "mean": float(mags.mean()),
            "std": float(mags.std()),
            "max": float(mags.max()),
            "p90": float(np.percentile(mags, 90)),
            "ratio_max_mean": float(mags.max() / mags.mean()),
        }
    out["wall_clock_s"] = None
    return out


# ---------------------------------------------------------------- merge side
def _percolation_threshold(phi: np.ndarray, p: np.ndarray) -> float:
    """phi at which the spanning probability crosses 0.5 (linear interpolation)."""
    order = np.argsort(phi)
    x, y = phi[order], p[order]
    for k in range(len(x) - 1):
        if y[k] < 0.5 <= y[k + 1]:
            if y[k + 1] == y[k]:
                return float(x[k])
            t = (0.5 - y[k]) / (y[k + 1] - y[k])
            return float(x[k] + t * (x[k + 1] - x[k]))
    return float(x[-1]) if len(x) else float("nan")


def merge(shard_files: list[str], out_json: str, fig_dir: str) -> dict:
    tasks: list[dict] = []
    for path in shard_files:
        with open(path) as handle:
            tasks.extend(json.load(handle)["results"])
    anchors = [t for t in tasks if t["kind"] == "anchor"]
    channels = [t for t in tasks if t["kind"] == "channel"]
    checks = [t for t in tasks if t["kind"] == "checks"]

    scan = [t for t in channels if abs(t.get("clearance", 0.07) - 0.07) < 1e-9
            and abs(t.get("beam", 0.26) - 0.26) < 1e-9]
    # clearance comparison at phi = 0.78: the 0.07 m rows come from the main scan
    clear = [t for t in channels if abs(t.get("phi", 0.0) - 0.78) < 1e-9]

    phi_levels = np.array(sorted({round(t["phi"], 4) for t in scan}))
    tau, tau_std, z, span, speed, jam = [], [], [], [], [], []
    for phi in phi_levels:
        rows = [t for t in scan if abs(t["phi"] - phi) < 5e-4]
        tau.append(float(np.mean([r["resistance_stress_pa"] for r in rows])))
        tau_std.append(float(np.std([r["resistance_stress_pa"] for r in rows])))
        z.append(float(np.mean([r["z_all"] for r in rows])))
        span.append(float(np.mean([1.0 if r["jammed"] or r["spanning_fraction"] > 0.5 else 0.0
                                   for r in rows])))
        speed.append(float(np.mean([r["speed_ratio"] for r in rows])))
        jam.append(float(np.mean([1.0 if r["jammed"] else 0.0 for r in rows])))
    tau = np.array(tau); tau_std = np.array(tau_std); z = np.array(z)
    span = np.array(span); speed = np.array(speed)
    jam = np.array(jam)

    fit_tau = piecewise_bic_breakpoint(phi_levels, tau)
    fit_z = piecewise_bic_breakpoint(phi_levels, z)
    fit_span = piecewise_bic_breakpoint(phi_levels, span) if phi_levels.size >= 7 else None

    # (f) percolation threshold sensitivity: first three seeds versus all five
    def threshold_for(seeds):
        rows = [t for t in scan if t["seed"] in seeds]
        xs = np.array(sorted({round(t["phi"], 4) for t in rows}))
        ys = []
        for x in xs:
            group = [t for t in rows if abs(t["phi"] - x) < 5e-4]
            ys.append(np.mean([1.0 if (t["jammed"] or t["spanning_fraction"] > 0.5) else 0.0
                               for t in group]))
        return _percolation_threshold(xs, np.array(ys))
    phi_c3 = threshold_for({0, 1, 2})
    phi_c5 = threshold_for({0, 1, 2, 3, 4})
    sensitivity = abs(phi_c5 - phi_c3) / max(abs(phi_c5), 1e-9)

    anchor_states = {}
    for a in anchors:
        anchor_states[a["tag"]] = {"phi_j": a["phi_j"], "levels": a["levels"],
                                   "reference_window": a["reference_window"],
                                   "phi_at_lowest_pressure": a["phi_at_lowest_pressure"]}

    checks_out = checks[0] if checks else {}
    criteria = {
        "a_virial_identity_rel": float(checks_out.get("virial_identity_rel", float("nan"))),
        "a_pass": bool(checks_out.get("virial_identity_rel", 1.0) < 1e-10),
        "b_stress_symmetry_rel": float(checks_out.get("stress_symmetry_rel", float("nan"))),
        "b_pass": bool(checks_out.get("stress_symmetry_rel", 1.0) < 1e-12),
        "c_hydrostatic_rel_error": float(checks_out.get("hydrostatic", {}).get("traction_relative_error", float("nan"))),
        "c_hydrostatic_band_rel_error": float(checks_out.get("hydrostatic", {}).get("relative_error", float("nan"))),
        "c_pass": bool(checks_out.get("hydrostatic", {}).get("traction_relative_error", 1.0) < 0.01),
        "d_phi_j_frictionless": float(anchor_states.get("frictionless", {}).get("phi_j", float("nan"))),
        "d_phi_j_frictional": float(anchor_states.get("frictional", {}).get("phi_j", float("nan"))),
        "d_pass": bool(abs(anchor_states.get("frictionless", {}).get("phi_j", 0.0) - 0.84) <= 0.01
                       and 0.80 <= anchor_states.get("frictional", {}).get("phi_j", 0.0) <= 0.84),
        "e_fabric_anisotropy": float(checks_out.get("fabric_anisotropy_bulk", float("nan"))),
        "e_pass": bool(abs(checks_out.get("fabric_anisotropy_bulk", 1.0)) < 0.02),
        "f_phi_c_3seeds": float(phi_c3),
        "f_phi_c_5seeds": float(phi_c5),
        "f_sensitivity": float(sensitivity),
        "f_pass": bool(sensitivity < 0.03),
        "g_momentum_drift_rel": float(checks_out.get("momentum_drift_rel", float("nan"))),
        "g_pass": bool(checks_out.get("momentum_drift_rel", 1.0) < 1e-10),
        "h_digest_identical": bool(checks_out.get("digest_identical", False)),
        "h_digest": checks_out.get("digest"),
        "h_pass": bool(checks_out.get("digest_identical", False)),
    }

    artifact = {
        "campaign": "IN22-N03",
        "title": "Force-chain percolation and jamming phase boundary of dense broken ice",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "contact": "Hertz-Mindlin normal/tangential + rolling resistance (Coulomb caps)",
            "discretisation": "2D rigid disks, semi-implicit Euler, float64",
            "drift": "linear drag m(V_d - v)/tau_d in the ship frame",
            "channel": CHANNEL,
            "phi_definition": "disk area / free channel area (channel minus hull)",
        },
        "phi_levels": phi_levels.tolist(),
        "curves": {
            "tau_pa": tau.tolist(), "tau_std_pa": tau_std.tolist(),
            "z": z.tolist(), "spanning_probability": span.tolist(),
            "jamming_probability": jam.tolist(), "mean_speed_ratio": speed.tolist(),
        },
        "breakpoints": {
            "tau": fit_tau.as_dict(),
            "z": fit_z.as_dict(),
            "spanning": fit_span.as_dict() if fit_span is not None else None,
        },
        "anchors": anchor_states,
        "checks": checks_out,
        "criteria": criteria,
        "percolation": {
            "phi_c_3seeds": float(phi_c3), "phi_c_5seeds": float(phi_c5),
            "sensitivity": float(sensitivity),
        },
        "clearance_sweep": [
            {"clearance": c,
             "phi": float(np.mean([r["phi"] for r in rows])),
             "beam": float(np.mean([r["beam"] for r in rows])),
             "seeds": len(rows),
             "resistance_stress_pa": float(np.mean([r["resistance_stress_pa"] for r in rows])),
             "speed_ratio": float(np.mean([r["speed_ratio"] for r in rows])),
             "spanning_fraction": float(np.mean([r["spanning_fraction"] for r in rows]))}
            for c in sorted({round(r["clearance"], 4) for r in clear})
            for rows in [[r for r in clear if abs(r["clearance"] - c) < 5e-4]]
        ],
        "task_counts": {"anchor": len(anchors), "channel": len(channels), "checks": len(checks)},
        "raw_channels": [
            {k: v for k, v in t.items() if k in
             ("phi", "seed", "beam", "clearance", "resistance_n", "resistance_stress_pa",
              "resistance_std_n", "speed_ratio", "z_all", "fabric_anisotropy",
              "load_bearing_ratio", "spanning_fraction", "strong_spanning_fraction",
              "jammed", "largest_chain", "mean_n_chains", "digest")}
            for t in channels
        ],
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as handle:
        json.dump(artifact, handle, indent=2)
    _figures(artifact, fig_dir)
    return artifact


def _figures(artifact: dict, fig_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    phi = np.array(artifact["phi_levels"])
    curve = artifact["curves"]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    tau = np.array(curve["tau_pa"]); err = np.array(curve["tau_std_pa"])
    ax.errorbar(phi, tau, yerr=err, marker="o", lw=1.6, capsize=3, label=r"$\tau(\varphi)$")
    brk = artifact["breakpoints"]["tau"]
    ax.axvline(brk["x_break"], ls="--", color="crimson",
               label=r"BIC breakpoint $\varphi^*=%.3f$" % brk["x_break"])
    ax.set_xlabel(r"ice area fraction $\varphi$ (free area)")
    ax.set_ylabel(r"hull resistance stress $\tau$ [Pa]")
    ax.set_title(r"Resistance rheology and jamming breakpoint ($\Delta BIC=%.1f$)" % brk["delta_bic"])
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "tau_phi.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(phi, curve["z"], marker="s", color="#1f77b4", label=r"$Z(\varphi)$ coordination number")
    ax.set_xlabel(r"$\varphi$"); ax.set_ylabel(r"$Z$", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(phi, curve["spanning_probability"], marker="^", color="#d62728",
             label=r"$P_{span}(\varphi)$")
    ax2.plot(phi, curve["jamming_probability"], marker="v", color="#2ca02c", ls="--",
             label=r"$P_{jam}(\varphi)$")
    ax2.set_ylabel("probability"); ax2.set_ylim(-0.05, 1.05)
    zbrk = artifact["breakpoints"]["z"]
    ax.axvline(zbrk["x_break"], ls=":", color="grey")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
    ax.set_title(r"Contact network: $Z$, spanning and jamming probability")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "z_pjam_phi.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for tag, color in (("frictionless", "#1f77b4"), ("frictional", "#d62728")):
        data = artifact["anchors"].get(tag)
        if not data:
            continue
        levels = np.array(data["levels"])
        ax.plot(levels[:, 2], levels[:, 1], marker="o", color=color,
                label=r"%s: $\varphi_J=%.3f$" % (tag, data["phi_j"]))
    ax.set_xlabel(r"pressure $P$ [N/m]"); ax.set_ylabel(r"area fraction $\varphi$")
    ax.set_title("Isotropic decompression anchor: jamming density extrapolated to $P\\to0$")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "anchor_phi_j.png"), dpi=140); plt.close(fig)

    clear = artifact.get("clearance_sweep") or []
    if clear:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        cs = [c["clearance"] for c in clear]
        rs = [c["resistance_stress_pa"] for c in clear]
        ax.plot(cs, rs, marker="o")
        ax.set_xlabel("side clearance [m]"); ax.set_ylabel(r"$\tau$ [Pa]")
        ax.set_title(r"Clearance width plateau at $\varphi=0.78$")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "clearance.png"), dpi=140); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--duration", type=float, default=1.2)
    ap.add_argument("--outdir", default="runs/force_chain_jamming")
    ap.add_argument("--artifacts", default="artifacts/force_chain_jamming.json")
    ap.add_argument("--figures", default="artifacts/figures")
    args = ap.parse_args()

    if args.merge:
        files = sorted(os.path.join(args.outdir, f) for f in os.listdir(args.outdir)
                       if f.startswith("shard_") and f.endswith(".json"))
        if not files:
            print("no shard files", file=sys.stderr)
            return 2
        art = merge(files, args.artifacts, args.figures)
        print(json.dumps(art["criteria"], indent=2))
        return 0

    tasks = build_tasks()
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        tasks = tasks[i::n]
        path = os.path.join(args.outdir, f"shard_{i}.json")
    else:
        path = os.path.join(args.outdir, "shard_0.json")
    os.makedirs(args.outdir, exist_ok=True)
    results = []
    for task in tasks:
        t0 = time.time()
        try:
            if task["kind"] == "channel":
                res = task_channel(task, args.duration)
            elif task["kind"] == "anchor":
                res = task_anchor(task)
            else:
                res = task_checks()
                res["wall_clock_s"] = None
        except Exception as exc:  # keep the shard usable
            res = {"kind": task["kind"], "error": repr(exc), **task}
        res["wall_clock_s"] = round(time.time() - t0, 1)
        res["task"] = task
        results.append(res)
        with open(path, "w") as handle:
            json.dump({"results": results}, handle, indent=2)
        print(f"[{task['kind']}] {task} -> {res.get('wall_clock_s')} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())