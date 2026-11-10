#!/usr/bin/env python3
"""Clear-ice floe-field DEM campaign (P103W2-N03 / m6997).

Runs the study in :mod:`tensordem.clear_ice.study`, writes

* ``artifacts/clear_ice_dem/result.json`` -- the contract-shaped result
  (``floe_kinematics``, ``force_chain_stats``, ``porosity``,
  ``convergence_report``, ``uncertainty_quantification``, ``provenance``);
* ``artifacts/clear_ice_dem/figures/*.png`` -- the report figures;
* ``artifacts/clear_ice_dem/figures.json`` -- the same figures as base64 (what
  the platform report embeds).

Usage::

    python benchmarks/clear_ice_dem.py --out artifacts/clear_ice_dem
    python benchmarks/clear_ice_dem.py --quick          # smoke / CI
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.clear_ice.field import FieldConfig  # noqa: E402
from tensordem.clear_ice.study import run_study  # noqa: E402


def _figures(payload: dict, raw: dict, outdir: str) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    os.makedirs(outdir, exist_ok=True)

    # 1. force-chain distribution with the exponential fit
    dist = raw["statistics"]["force_distribution"]
    f = np.asarray(dist["f_over_mean"])
    p = np.asarray(dist["P_f"])
    good = p > 0
    chains = payload["force_chain_stats"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.semilogy(f[good], p[good], "o", ms=4, label="DEM contacts")
    f0 = chains["f0_exponential"]
    if np.isfinite(f0):
        ff = np.linspace(1.0, 4.0, 50)
        # anchored at the first tail bin so the line shows the fitted slope
        idx = np.argmin(np.abs(f - 1.0))
        p0 = p[idx]
        ax.semilogy(ff, p0 * np.exp(-(ff - f[idx]) / f0), "-",
                    label=f"exp fit tail: f0={f0:.2f}, R$^2$={chains['r2_exponential_tail']:.3f}")
    ax.axvline(1.0, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel(r"$f = F_n/\langle F_n\rangle$")
    ax.set_ylabel(r"$P(f)$")
    ax.set_title("Force-chain distribution (clear-ice floe field)")
    ax.set_xlim(0, 5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    figs.append({"name": "force_chain_distribution.png",
                 "caption": "Normalised normal-force distribution P(f) of the "
                            "steady clear-ice floe field; exponential tail fit "
                            f"R^2={chains['r2_exponential_tail']:.3f}, "
                            f"f0={f0:.2f}.",
                 "fig": fig})

    # 2. contact-order convergence
    traj = payload["convergence_report"]["trajectory_order"]
    dts = np.asarray(traj["dt_levels_s"])
    errs = np.asarray(traj["l2_error_m"])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    keep = errs > 0
    ax.loglog(dts[keep], errs[keep], "o-", label="L2 position error")
    ref = errs[keep][-1] * (dts[keep] / dts[keep][-1]) ** 1.0
    ax.loglog(dts[keep], ref, "k--", lw=1, label="order 1 reference")
    ref2 = errs[keep][-1] * (dts[keep] / dts[keep][-1]) ** 2.0
    ax.loglog(dts[keep], ref2, "k:", lw=1, label="order 2 reference")
    ax.set_xlabel(r"$\Delta t$ (s)")
    ax.set_ylabel("L2 centroid-position error (m)")
    ax.set_title(f"Step-size convergence of a floe collision "
                 f"(observed order {traj['observed_order_fit']:.2f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    figs.append({"name": "timestep_convergence.png",
                 "caption": "Deterministic two-floe Hertz-Mindlin collision: "
                            "L2 centroid-position error against the finest step, "
                            f"observed order {traj['observed_order_fit']:.2f}.",
                 "fig": fig})

    # 3. porosity by region
    por = payload["porosity"]
    names, vals, errs_bar = [], [], []
    for key, label in (("domain_mean", "domain mean"), ("feed", "feed"),
                       ("jam", "jam"), ("wake", "wake")):
        if key in por and isinstance(por[key], dict) and "areal_mean" in por[key]:
            names.append(label)
            vals.append(por[key]["areal_mean"])
            errs_bar.append(por[key].get("areal_std", 0.0))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(names, vals, yerr=errs_bar, capsize=4, color="#4c72b0")
    ax.axhspan(0.30, 0.45, color="#55a868", alpha=0.18, label="criterion band [0.30, 0.45]")
    ax.set_ylabel("areal porosity n")
    ax.set_title("Porosity of the floe field by region")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    figs.append({"name": "porosity_regions.png",
                 "caption": "Areal porosity by region (domain mean, upstream feed, "
                            "gate jam, cleared wake) against the criterion band.",
                 "fig": fig})

    # 4. kinematics time series + negative control
    out = raw["run_output"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(out.time, out.mean_speed, lw=1.0, label="mean floe speed (resolved)")
    ax.axvline(out.config.warmup, color="0.7", ls="--", lw=0.8,
               label="statistics window start")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$\langle |\mathbf{v}| \rangle$ (m/s)")
    ax.set_title("Floe kinematics: drift then compression on the hull face")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    figs.append({"name": "kinematics_timeseries.png",
                 "caption": "Mean floe speed: the field drifts with the ice and is "
                            "brought to rest by the hull face; the record is smooth "
                            "(no grid-frequency oscillation).",
                 "fig": fig})

    # 5. field map
    pos = out.positions[-1]
    vel = np.linalg.norm(out.velocities[-1], axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=(out.radii * 300) ** 1.0, c=vel,
                    cmap="viridis", edgecolors="k", linewidths=0.4)
    ax.axvline(out.config.gate_centre[0], color="crimson", lw=2.0, label="hull face")
    fig.colorbar(sc, ax=ax, label=r"$|\mathbf{v}|$ (m/s)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Clear-ice floe field at the end of the run")
    ax.legend(fontsize=8)
    figs.append({"name": "floe_field_map.png",
                 "caption": "Floe positions at the end of the run (marker area "
                            "proportional to floe area, colour = speed); the field "
                            "is compressed against the hull face at the right.",
                 "fig": fig})

    paths = []
    for item in figs:
        path = os.path.join(outdir, item["name"])
        item["fig"].savefig(path, dpi=130, bbox_inches="tight")
        plt.close(item["fig"])
        paths.append({"name": item["name"], "caption": item["caption"], "path": path})
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/clear_ice_dem")
    ap.add_argument("--quick", action="store_true", help="reduced study for CI")
    ap.add_argument("--n-floes", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()

    cfg = FieldConfig()
    overrides = {}
    if args.n_floes:
        overrides["n_floes"] = args.n_floes
    if args.duration:
        overrides["duration"] = args.duration
        overrides["warmup"] = 0.5 * args.duration
        overrides["kinematics_window"] = (0.1 * args.duration, 0.35 * args.duration)
    if overrides:
        cfg = FieldConfig(**{**cfg.__dict__, **overrides})

    t0 = time.time()
    payload, raw = run_study(cfg, quick=args.quick)
    wall = time.time() - t0

    os.makedirs(args.out, exist_ok=True)
    payload["provenance"]["wall_time_s"] = wall
    payload["provenance"]["artifact"] = os.path.join(args.out, "result.json")
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)

    figs = _figures(payload, raw, os.path.join(args.out, "figures"))
    encoded = []
    for item in figs:
        with open(item["path"], "rb") as fh:
            encoded.append({"name": item["name"], "caption": item["caption"],
                            "png_base64": base64.b64encode(fh.read()).decode()})
    with open(os.path.join(args.out, "figures.json"), "w") as fh:
        json.dump(encoded, fh)

    print(json.dumps({
        "wall_time_s": wall,
        "checks": payload["convergence_report"]["checks"],
        "C1": payload["floe_kinematics"]["check"],
        "C2": payload["force_chain_stats"]["check"],
        "C3": payload["porosity"]["check"],
        "artifact": os.path.join(args.out, "result.json"),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())