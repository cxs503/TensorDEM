"""Audit the *near-level* righting moment of ``tensordem.ice_rideup`` against
exact large-angle hydrostatics.

The ride-up kernel (``RideUpSimulation._hydrostatic_forces``) restores a floe
with the small-angle law

    M_near(phi) = rho_w g V_sub GM sin(phi),   GM = R^2 / (4 d),
    V_sub = pi R^2 d,  d = (rho_ice/rho_w) t,

i.e. the waterplane second moment of a *level* disk with the keel-to-centre-of-
gravity term (``BG``) dropped.  A floe carried up a wedge bow is heeled to the
bow angle for a large part of its transit, so the question this script answers
is quantitative: over the heel range that actually occurs in the ride-up case,
how far is that law from the exact heave-relaxed hydrostatics already verified
in :mod:`tensordem.floe_6dof` (IN22-N02, ``submerged_hydrostatics`` /
``capsize_curve``)?

The reference model is not re-derived here: it is the merged, separately
verified quadrature kernel.  Every number written to
``artifacts/ice_rideup_righting/metrics.json`` is produced by this script (no
hand transcription); the figures are drawn from the same arrays.

Run:  PYTHONPATH=src python scripts/audit_ice_rideup_righting.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)

from tensordem.floe_6dof import (capsize_angle_from_hessian,
                                 capsize_angle_from_moment, capsize_curve)
from tensordem.ice_channel import WaterEnvironment
from tensordem.ice_rideup import GRAVITY, RHO_ICE, RHO_WATER

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ice_rideup_righting"
DTYPE = torch.float64

# (radius [m], thickness [m]) -- aspect ratio R/t from thin first-year ice to a
# thick, almost-blocky floe.  (4.0, 0.5) is the ride-up benchmark default
# (``RideUpSimulation(radius_mean=4.0, thickness=0.5)``).
GEOMETRIES = [(2.5, 0.5), (4.0, 0.5), (5.0, 0.5), (8.0, 0.5)]

# heel angles reached in the ride-up case: the bow slope angle (BowSlope
# default) plus dynamic overshoot.  The error is quoted at these angles.
RIDEUP_HEELS_DEG = (10.0, 20.0, 25.0, 30.0, 45.0)

GATES = (0.03, 0.15)


def near_level_moment(tilt: torch.Tensor, radius: float, thickness: float, *,
                      rho_water: float = RHO_WATER, rho_ice: float = RHO_ICE,
                      gravity: float = GRAVITY) -> torch.Tensor:
    """The law used by ``RideUpSimulation._hydrostatic_forces`` (verbatim)."""
    d_sub = (rho_ice / rho_water) * thickness
    gm = radius * radius / (4.0 * d_sub)              # BM for a disk, BG neglected
    v_sub = math.pi * radius * radius * d_sub
    return rho_water * gravity * v_sub * gm * torch.sin(tilt)


def near_level_stiffness(radius: float, thickness: float) -> float:
    """Small-angle roll stiffness implied by the near-level law, N m / rad."""
    d = (RHO_ICE / RHO_WATER) * thickness
    return RHO_WATER * GRAVITY * (math.pi * radius ** 2 * d) * (radius ** 2 / (4.0 * d))


def crossing(tilts: torch.Tensor, err: torch.Tensor, gate: float) -> float | None:
    """First heel angle (deg) at which ``err`` exceeds ``gate`` (linear interp)."""
    above = torch.nonzero(err > gate).flatten()
    if above.numel() == 0:
        return None
    j = int(above[0])
    if j == 0:
        return float(torch.rad2deg(tilts[0]))
    x0, x1 = float(tilts[j - 1]), float(tilts[j])
    e0, e1 = float(err[j - 1]), float(err[j])
    if e1 == e0:
        return float(torch.rad2deg(tilts[j]))
    x = x0 + (gate - e0) * (x1 - x0) / (e1 - e0)
    return math.degrees(x)


def brute_force_moment(radius: float, thickness: float, tilt_deg: float,
                       *, samples: int = 4_000_000, seed: int = 20261106) -> dict:
    """Independent route to V_sub and the righting moment: Monte-Carlo integration.

    Uniform samples are drawn in the body cylinder and kept when the *world*
    elevation is below the free surface.  This shares no code with the disk
    quadrature of :mod:`tensordem.floe_6dof`; agreement to the statistical
    resolution (~1/sqrt(N) on the volume, a few 1e-3 relative) is the
    independent check of the reference used by the audit.
    """
    g = torch.Generator().manual_seed(int(seed))
    n = int(samples)
    phi = math.radians(float(tilt_deg))
    # body-frame cylinder: disk in (x, y), axis z; then roll about x
    rho = radius * torch.sqrt(torch.rand(n, generator=g, dtype=DTYPE))
    th = 2.0 * math.pi * torch.rand(n, generator=g, dtype=DTYPE)
    zeta = thickness * (torch.rand(n, generator=g, dtype=DTYPE) - 0.5)
    x = rho * torch.cos(th)
    y = rho * torch.sin(th)
    # world elevation of a body point under a roll phi about the x axis
    z_world = y * math.sin(phi) + zeta * math.cos(phi)
    # heave: place the centre so the displaced volume matches the floe mass
    d_eq = (RHO_ICE / RHO_WATER) * thickness
    volume_exact = math.pi * radius ** 2 * d_eq
    # bisection on the centre height so that MC volume == analytic displacement
    lo, hi = -radius - thickness, radius + thickness
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v = float(((z_world + mid) < 0.0).to(DTYPE).mean()) * (math.pi * radius ** 2 * thickness)
        if v > volume_exact:
            lo = mid
        else:
            hi = mid
    centre_z = 0.5 * (lo + hi)
    submerged = (z_world + centre_z) < 0.0
    frac = float(submerged.to(DTYPE).mean())
    cell = math.pi * radius ** 2 * thickness
    v_mc = frac * cell
    # horizontal moment arm of the displaced volume: the *world* y component of
    # the centre of buoyancy (a body point (0, y, 0) maps to y cos(phi) in the
    # world horizontal after the roll; forgetting the cos(phi) is a 6 % error at
    # 20 deg heel)
    y_arm = float((y[submerged].mean()) * math.cos(phi))
    f_buoy = RHO_WATER * GRAVITY * v_mc
    return {"tilt_deg": float(tilt_deg),
            "volume_m3": v_mc,
            "volume_analytic_m3": volume_exact,
            "volume_rel_error": abs(v_mc - volume_exact) / volume_exact,
            "centre_height_m": centre_z,
            "moment_nm": -f_buoy * y_arm,
            "samples": n,
            "volume_stat_error": float((frac * (1.0 - frac) / n) ** 0.5 / max(frac, 1e-12))}


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    water = WaterEnvironment(water_density=RHO_WATER, ice_density=RHO_ICE,
                             gravity=GRAVITY)
    rows = []
    curves = {}
    brute_force = []
    for radius, thickness in GEOMETRIES:
        aspect = radius / thickness
        curve = capsize_curve(radius, thickness, water, tilt_count=181,
                              tilt_max=0.5 * math.pi)
        tilts = curve.tilt
        m_exact = curve.moment
        m_near = near_level_moment(tilts, radius, thickness)
        scale = float(m_exact.abs().max())
        err = (m_near - m_exact).abs() / scale

        k_near = near_level_stiffness(radius, thickness)
        k_exact = curve.analytic_stiffness            # rho_w g V GM with BG
        # the merged kernel's own small-angle limit, from the moment curve
        eps = float(tilts[1])
        k_measured = float(m_exact[1] / eps)

        fold_moment = float(torch.rad2deg(tilts[int(torch.argmax(m_exact))]))
        fold_hess = math.degrees(float(capsize_angle_from_hessian(radius, thickness, water)["capsize_angle"]))
        fold_force = math.degrees(float(capsize_angle_from_moment(radius, thickness, water)["capsize_angle"]))

        if (radius, thickness) == GEOMETRIES[1]:
            bf_rows = []
            for deg in (2.0, 5.0, 10.0, 20.0):
                bf = brute_force_moment(radius, thickness, deg)
                idx = min(range(len(tilts)), key=lambda i: abs(float(tilts[i]) - math.radians(deg)))
                bf["moment_kernel_nm"] = float(m_exact[idx])
                bf["moment_rel_error"] = abs(bf["moment_nm"] - float(m_exact[idx])) / scale
                bf["height_kernel_m"] = float(curve.height[idx])
                bf["height_abs_error_m"] = abs(bf["centre_height_m"] - float(curve.height[idx]))
                bf_rows.append(bf)
            brute_force = bf_rows

        heels = {}
        for deg in RIDEUP_HEELS_DEG:
            phi = math.radians(deg)
            # evaluate the exact moment on the same 0.5-deg grid (interpolated)
            idx = min(range(len(tilts)), key=lambda i: abs(float(tilts[i]) - phi))
            me = float(m_exact[idx])
            mn = float(near_level_moment(torch.tensor(phi, dtype=DTYPE), radius, thickness))
            heaps = abs(mn - me) / scale
            heaps_signed = (mn - me) / scale
            heels[str(deg)] = {
                "tilt_deg_grid": float(torch.rad2deg(tilts[idx])),
                "moment_exact_nm": me,
                "moment_near_level_nm": mn,
                "abs_error_over_peak_moment": heaps,
                "signed_error_over_peak_moment": heaps_signed,
            }

        rows.append({
            "radius_m": radius,
            "thickness_m": thickness,
            "aspect_ratio_R_over_t": aspect,
            "draught_m": (RHO_ICE / RHO_WATER) * thickness,
            "near_level_GM_m": radius ** 2 / (4.0 * (RHO_ICE / RHO_WATER) * thickness),
            "analytic_GM_with_BG_m": curve.metacentric_height,
            "GM_rel_error_no_BG": abs(radius ** 2 / (4.0 * (RHO_ICE / RHO_WATER) * thickness)
                                      - curve.metacentric_height) / abs(curve.metacentric_height),
            "stiffness_near_level_nm_per_rad": k_near,
            "stiffness_analytic_nm_per_rad": k_exact,
            "stiffness_measured_nm_per_rad": k_measured,
            "stiffness_near_vs_analytic_rel_error": abs(k_near - k_exact) / abs(k_exact),
            "stiffness_kernel_vs_analytic_rel_error": abs(k_measured - k_exact) / abs(k_exact),
            "peak_exact_moment_nm": scale,
            "peak_exact_at_deg": fold_moment,
            "near_level_at_90deg_nm": float(near_level_moment(
                torch.tensor(0.5 * math.pi, dtype=DTYPE), radius, thickness)),
            "near_over_peak_at_90deg": float(near_level_moment(
                torch.tensor(0.5 * math.pi, dtype=DTYPE), radius, thickness)) / scale,
            "capsize_angle_deg_moment_route": fold_force,
            "capsize_angle_deg_hessian_route": fold_hess,
            "capsize_route_spread_deg": abs(fold_force - fold_hess),
            "capsize_angle_deg_implied_by_near_level": None,   # law never folds
            "first_heel_deg_err_gt_3pct": crossing(tilts, err, GATES[0]),
            "first_heel_deg_err_gt_15pct": crossing(tilts, err, GATES[1]),
            "max_abs_error_over_peak_moment": float(err.max()),
            "max_abs_error_deg": float(torch.rad2deg(tilts[int(err.argmax())])),
            "errors_at_rideup_heels": heels,
        })
        curves[aspect] = {
            "tilt_deg": [float(v) for v in torch.rad2deg(tilts)],
            "moment_exact_nm": [float(v) for v in m_exact],
            "moment_near_level_nm": [float(v) for v in m_near],
            "energy_exact_j": [float(v) for v in curve.energy],
            "error_over_peak": [float(v) for v in err],
            "stiffness_near_level": k_near,
            "energy_near_level_j": [float(k_near * (1.0 - math.cos(float(p))))
                                    for p in tilts],
        }

    metrics = {
        "water": {"rho_water": RHO_WATER, "rho_ice": RHO_ICE, "gravity": GRAVITY},
        "convention": ("moment is the heave-relaxed righting moment about the "
                       "centre of mass, positive = restoring; errors are "
                       "normalised by the peak exact moment of the same floe"),
        "tilt_grid": "0 .. 90 deg, 181 points (0.5 deg)",
        "reference": "tensordem.floe_6dof.capsize_curve / submerged_hydrostatics "
                     "(24x32 disk quadrature, heave relaxed)",
        "models": {
            "ice_rideup_near_level":
                "M = rho_w g (pi R^2 d) (R^2/(4d)) sin(phi), d = (rho_ice/rho_w) t",
            "exact": "centre-of-buoyancy moment of the tilted cylinder, heave relaxed",
        },
        "geometries": rows,
        "reference_cross_check_monte_carlo": brute_force,
        "curves": {str(k): v for k, v in curves.items()},
        "rideup_heel_band_deg": list(RIDEUP_HEELS_DEG),
        "artifacts": {},
    }

    metrics["artifacts"] = {"metrics.json": "every number in this report"}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- figures ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for aspect, c in curves.items():
        d = [abs(v) for v in c["moment_exact_nm"]]
        ax.plot(c["tilt_deg"], d, lw=1.8, label=f"exact, R/t={aspect:g}")
        ax.plot(c["tilt_deg"], c["moment_near_level_nm"], lw=1.2, ls="--",
                label=f"near-level, R/t={aspect:g}")
    ax.axvspan(0.0, 35.0, color="0.85", zorder=0)
    ax.text(17.0, ax.get_ylim()[1] * 0.94, "ride-up heel band", ha="center", fontsize=8)
    ax.set_xlabel("heel angle phi [deg]")
    ax.set_ylabel("|righting moment| [N m]")
    ax.set_title("Righting moment: exact large-angle hydrostatics vs the\n"
                 "near-level law used by tensordem.ice_rideup")
    ax.set_xlim(0.0, 90.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "righting_moment_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for aspect, c in curves.items():
        ax.semilogy(c["tilt_deg"], [max(v, 1e-8) for v in c["error_over_peak"]],
                    lw=1.6, label=f"R/t={aspect:g}")
    for gate, colour, name in ((0.03, "tab:green", "3 % gate"),
                               (0.15, "tab:red", "15 % gate")):
        ax.axhline(gate, color=colour, ls=":", lw=1.2, label=name)
    ax.axvspan(0.0, 35.0, color="0.85", zorder=0)
    ax.set_xlabel("heel angle phi [deg]")
    ax.set_ylabel("|M_near - M_exact| / max M_exact")
    ax.set_title("Near-level righting-moment error vs heel angle")
    ax.set_xlim(0.0, 90.0)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_heel.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for aspect, c in curves.items():
        e = c["energy_exact_j"]
        e0 = e[0]
        ax.plot(c["tilt_deg"], [v - e0 for v in e], lw=1.8,
                label=f"exact, R/t={aspect:g}")
        ax.plot(c["tilt_deg"], c["energy_near_level_j"], lw=1.2, ls="--",
                label=f"near-level k(1-cos phi), R/t={aspect:g}")
    ax.set_xlabel("heel angle phi [deg]")
    ax.set_ylabel("attitude potential minus upright value [J]")
    ax.set_title("Heave-relaxed attitude energy: the exact curve saturates at the\n"
                 "capsize fold, the near-level law keeps climbing")
    ax.set_xlim(0.0, 90.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "attitude_energy.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    aspects = [r["aspect_ratio_R_over_t"] for r in rows]
    axes[0].plot(aspects, [r["capsize_angle_deg_moment_route"] for r in rows],
                 "o-", label="exact fold (moment route)")
    axes[0].plot(aspects, [r["capsize_angle_deg_hessian_route"] for r in rows],
                 "x--", label="exact fold (energy route)")
    axes[0].axhline(90.0, color="tab:red", ls=":",
                    label="near-level law: no fold (max at 90 deg)")
    axes[0].set_xlabel("aspect ratio R/t")
    axes[0].set_ylabel("capsize angle [deg]")
    axes[0].set_title("Capsize fold angle")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    for deg in (20.0, 25.0, 30.0):
        axes[1].plot(aspects,
                     [100.0 * r["errors_at_rideup_heels"][str(float(deg))][
                         "abs_error_over_peak_moment"] for r in rows],
                     "o-", label=f"phi = {deg:g} deg")
    axes[1].axhline(3.0, color="tab:green", ls=":", label="3 % gate")
    axes[1].axhline(15.0, color="tab:red", ls=":", label="15 % gate")
    axes[1].set_xlabel("aspect ratio R/t")
    axes[1].set_ylabel("|error| / peak exact moment [%]")
    axes[1].set_title("Near-level error inside the ride-up heel band")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "capsize_and_band_error.png", dpi=150)
    plt.close(fig)

    metrics["artifacts"] = {
        "righting_moment_curves.png": "exact vs near-level M(phi) for four R/t",
        "error_vs_heel.png": "normalised error vs heel with the 3 %/15 % gates",
        "attitude_energy.png": "heave-relaxed attitude energy, exact vs k(1-cos phi)",
        "capsize_and_band_error.png": "capsize fold angle and band error vs R/t",
        "metrics.json": "every number in this report",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = main()
    print(json.dumps({"geometries": [
        {k: v for k, v in row.items() if k != "errors_at_rideup_heels"}
        for row in m["geometries"]]}, indent=2))