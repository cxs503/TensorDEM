"""Audit the 3-D spherical-particle buoyancy kernel against closed-form anchors.

Produces, under ``docs/dem3d-buoyancy-float/``:
  * metrics.json  — every number quoted in the report, machine written
  * *.png         — figures (geometry kernel, force curve, relaxation, 2-D misuse)
  * REPORT.md     — generated results section (tables injected from metrics.json)

Run:  python scripts/audit_dem3d_buoyancy_float.py
All computations are analytic or one-dimensional ODE integrations: the script is
lightweight and needs no cluster.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensordem.buoyancy_wet import _submerged_fraction_2d  # noqa: E402
from tensordem.buoyancy_wet3d import (  # noqa: E402
    equilibrium_centre_height,
    hydrostatic_stiffness_sphere,
    submerged_fraction_sphere,
    submerged_fraction_sphere_factored,
    submerged_volume_sphere,
    waterplane_area_sphere,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "dem3d-buoyancy-float")
R = 0.05                 # sphere radius, m
RHO_W = 1025.0           # sea water, kg/m^3
RHO_ICE = 917.0          # ice, kg/m^3
G = 9.81


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0.0 else abs(a)


# ---------------------------------------------------------------------------
# 1) Geometry kernel checks
# ---------------------------------------------------------------------------
def geometry_checks():
    hs = np.linspace(-1.0, 1.0, 401) * R
    # independent quadrature in centre-relative coordinates q = z - h:
    # the free surface is q = -h, so the submerged cap is q in [-R, -h].
    def simpson_volume(h):
        n = 20000
        z = np.linspace(-R, -h, n + 1)
        f = np.pi * (R ** 2 - z ** 2)
        w = np.ones(n + 1)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        return float(np.sum(w * f) * (z[1] - z[0]) / 3.0)

    quad_err, form_err, deriv_err, mono_ok = [], [], [], True
    prev = None
    for h in hs:
        v = submerged_volume_sphere(float(h), R)
        if prev is not None and v > prev + 1e-18:
            mono_ok = False
        prev = v
        vq = simpson_volume(float(h))
        quad_err.append(rel(v, vq) if vq > 0 else abs(v))
        form_err.append(abs(submerged_fraction_sphere(float(h), R)
                            - submerged_fraction_sphere_factored(float(h), R)))
        # -dV/dh = A_wp(h) via a central difference on the analytic volume
        dh = 1e-6 * R
        dV = -(submerged_volume_sphere(float(h) + dh, R)
               - submerged_volume_sphere(float(h) - dh, R)) / (2 * dh)
        aw = waterplane_area_sphere(float(h), R)
        deriv_err.append(rel(dV, aw) if aw > 1e-12 else abs(dV))

    # vectorised tensor route must match the scalar route
    ys = torch.tensor(hs, dtype=torch.float64)
    from tensordem.buoyancy_wet3d import submerged_fraction_sphere_tensor
    f_t = submerged_fraction_sphere_tensor(ys, R).numpy()
    f_s = np.array([submerged_fraction_sphere(float(h), R) for h in hs])
    return dict(
        n_samples=int(hs.size),
        max_rel_err_vs_simpson=float(np.max(quad_err)),
        max_abs_diff_two_factorisations=float(np.max(form_err)),
        max_rel_err_minus_dVdh_vs_waterplane=float(np.max(deriv_err)),
        monotone_decreasing=bool(mono_ok),
        max_abs_diff_scalar_vs_tensor=float(np.max(np.abs(f_t - f_s))),
        f_at_h_0=submerged_fraction_sphere(0.0, R),
        f_at_h_R=submerged_fraction_sphere(R, R),
        f_at_h_minus_R=submerged_fraction_sphere(-R, R),
    )


# ---------------------------------------------------------------------------
# 2) Archimedes equilibrium: bisection vs polynomial root, force residual
# ---------------------------------------------------------------------------
def equilibrium_checks():
    rows = []
    for rho_p in (330.0, 500.0, 700.0, 850.0, RHO_ICE, 1000.0):
        r = rho_p / RHO_W
        h_eq = equilibrium_centre_height(R, rho_p, RHO_W)
        # independent route: roots of u^3 - 3u + 2 - 4r = 0
        roots = np.roots([1.0, 0.0, -3.0, 2.0 - 4.0 * r])
        real = [float(np.real(x)) for x in roots if abs(np.imag(x)) < 1e-9
                and -1.0 - 1e-9 <= np.real(x) <= 1.0 + 1e-9]
        u_poly = min(real, key=lambda u: abs(u * R - h_eq))
        f_res = submerged_fraction_sphere(h_eq, R) - r
        vol = submerged_volume_sphere(h_eq, R)
        f_b = RHO_W * G * vol
        w = rho_p * G * (4.0 / 3.0 * math.pi * R ** 3)
        rows.append(dict(
            particle_density=rho_p,
            density_ratio=r,
            h_eq_m=h_eq,
            u_eq=h_eq / R,
            draft_m=R - h_eq,
            submerged_fraction=submerged_fraction_sphere(h_eq, R),
            fraction_residual=f_res,
            u_from_polynomial_root=u_poly,
            abs_diff_u_vs_polynomial=abs(u_poly - h_eq / R),
            buoyancy_N=f_b,
            weight_N=w,
            archimedes_force_rel_error=rel(f_b, w),
            hydrostatic_stiffness_N_per_m=hydrostatic_stiffness_sphere(h_eq, R, RHO_W, G),
        ))
    return rows


# ---------------------------------------------------------------------------
# 3) 1-DOF vertical dynamics: relaxation to h_eq and oscillation period
# ---------------------------------------------------------------------------
def dynamics(mass_density=RHO_ICE, drag_coeff=2.0, dt=2e-4, t_end=6.0):
    m = mass_density * 4.0 / 3.0 * math.pi * R ** 3
    h_eq = equilibrium_centre_height(R, mass_density, RHO_W)
    k = RHO_W * G * 4.0 / 3.0 * math.pi * R ** 3   # scale so f maps to volume

    def accel(h, v):
        # a = (F_b(h) + F_wet(v) - m g) / m  — positive upwards, h = elevation
        f_b = k * submerged_fraction_sphere(h, R)
        f_wet = -drag_coeff * submerged_fraction_sphere(h, R) * v
        return (f_b + f_wet - m * G) / m

    # (a) damped relaxation from fully submerged rest
    h, v, n = -R, 0.0, int(round(t_end / dt))
    traj = [(0.0, h)]
    for i in range(n):
        v += accel(h, v) * dt
        h += v * dt
        if (i + 1) % 50 == 0:
            traj.append(((i + 1) * dt, h))
    h_final = traj[-1][1]

    # (b) undamped small oscillation about h_eq -> period vs analytic
    h_eq2 = equilibrium_centre_height(R, mass_density, RHO_W)
    K = hydrostatic_stiffness_sphere(h_eq2, R, RHO_W, G)
    w0 = math.sqrt(K / m)
    amp = 0.01 * R
    h0 = h_eq2 + amp
    hu, vu = h0, 0.0
    t_prev, crossings = 0.0, []
    prev_h = hu
    dt2 = 2e-5
    n2 = int(round(2.5 * 2 * math.pi / w0 / dt2))

    def accel_undamped(hh):
        return (k * submerged_fraction_sphere(hh, R) - m * G) / m

    for i in range(n2):
        vu += accel_undamped(hu) * dt2
        hu += vu * dt2
        if prev_h > h_eq2 >= hu:  # downward crossing
            crossings.append((i + 1) * dt2)
        prev_h = hu
    period = ((crossings[-1] - crossings[-2]) if len(crossings) >= 2 else float("nan"))
    return dict(
        mass_kg=m,
        h_eq_m=h_eq,
        damped_final_h_m=h_final,
        damped_error_over_R=abs(h_final - h_eq) / R,
        drag_coeff_N_s_per_m=drag_coeff,
        dt_s=dt,
        t_end_s=t_end,
        analytic_period_s=2.0 * math.pi / w0,
        measured_period_s=period,
        period_rel_error=rel(period, 2.0 * math.pi / w0) if period == period else float("nan"),
        amplitude_over_R=amp / R,
        trajectory=[(round(t, 6), h) for t, h in traj],
    )


# ---------------------------------------------------------------------------
# 4) 2-D disk kernel misapplied to a sphere: quantified error
# ---------------------------------------------------------------------------
def cross_dimension_error():
    rows = []
    for frac in (-0.9, -0.5, 0.0, 0.25, 0.5, 0.9):
        h = frac * R
        f2 = _submerged_fraction_2d(h, R)
        f3 = submerged_fraction_sphere(h, R)
        rows.append(dict(h_over_R=frac, f_2d_disk=f2, f_3d_sphere=f3,
                         rel_error_vs_3d=rel(f2, f3)))
    h_eq = equilibrium_centre_height(R, RHO_ICE, RHO_W)
    f2 = _submerged_fraction_2d(h_eq, R)
    f3 = submerged_fraction_sphere(h_eq, R)
    rows.append(dict(h_over_R=h_eq / R, f_2d_disk=f2, f_3d_sphere=f3,
                     rel_error_vs_3d=rel(f2, f3), context="ice equilibrium r=0.8946"))
    return rows


def cross_dimension_draft():
    """Equilibrium draft if the 2-D disk kernel is (wrongly) used for a sphere."""
    r = RHO_ICE / RHO_W
    lo, hi = -R, R
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _submerged_fraction_2d(mid, R) > r:
            lo = mid
        else:
            hi = mid
    h2d = 0.5 * (lo + hi)
    h3d = equilibrium_centre_height(R, RHO_ICE, RHO_W)
    return dict(density_ratio=r, h_eq_2d_kernel_m=h2d, h_eq_3d_kernel_m=h3d,
                draft_2d_kernel_m=R - h2d, draft_3d_kernel_m=R - h3d,
                draft_rel_error=rel(R - h2d, R - h3d))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figures(metrics):
    os.makedirs(OUT, exist_ok=True)
    hs = np.linspace(-1.0, 1.0, 401)
    f3 = np.array([submerged_fraction_sphere(h * R, R) for h in hs])
    f2 = np.array([_submerged_fraction_2d(h * R, R) for h in hs])
    awp = np.array([waterplane_area_sphere(h * R, R) for h in hs]) / (math.pi * R ** 2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(hs, f3, "C0-", lw=2, label="3-D sphere: volume fraction")
    ax[0].plot(hs, f2, "C1--", lw=1.6, label="2-D disk: area fraction (SI-D module)")
    ax[0].axhline(RHO_ICE / RHO_W, color="k", ls=":", lw=1,
                  label=r"$\rho_{ice}/\rho_{w}=%.3f$" % (RHO_ICE / RHO_W))
    ax[0].set_xlabel(r"centre elevation $h/R$")
    ax[0].set_ylabel("submerged fraction")
    ax[0].set_title("Geometry kernels differ in 3-D")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(hs, awp, "C2-", lw=2, label=r"$A_{wp}(h)/\pi R^2$")
    ax[1].plot(hs, -np.gradient(f3, hs), "C3--", lw=1.4,
               label=r"$-dV_f/dh$ (finite difference)")
    ax[1].set_xlabel(r"$h/R$")
    ax[1].set_ylabel("normalised")
    ax[1].set_title(r"Waterplane identity: $-dV_f/dh = A_{wp}/\pi R^2$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "geometry_kernel.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for rho_p, colour in ((RHO_ICE, "C0"), (700.0, "C4"), (330.0, "C2")):
        hh = np.linspace(-R, R, 400)
        fb = np.array([RHO_W * G * submerged_volume_sphere(float(x), R) for x in hh])
        w = rho_p * G * 4.0 / 3.0 * math.pi * R ** 3
        ax.plot(hh / R, fb, colour + "-", lw=1.8,
                label=r"$F_b$ (bulk curve, rho_p=%.0f)" % rho_p)
        ax.axhline(w, color="grey", ls=":", lw=1)
        heq = equilibrium_centre_height(R, rho_p, RHO_W)
        ax.plot(heq / R, w, "o", ms=6, color=colour)
    ax.set_xlabel(r"centre elevation $h/R$")
    ax.set_ylabel(r"$F_b$ [N]")
    ax.set_title("Archimedes equilibrium points (o) on the buoyancy curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "buoyancy_equilibrium.png"), dpi=140)
    plt.close(fig)

    dyn = metrics["dynamics"]
    traj = np.array(dyn["trajectory"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(traj[:, 0], traj[:, 1] / R, "C0-", lw=1.6, label="1-DOF relaxation")
    ax.axhline(dyn["h_eq_m"] / R, color="C3", ls="--", lw=1.2,
               label=r"analytic $h_{eq}/R$")
    ax.axhline(-1.0, color="grey", ls=":", lw=1, label="fully submerged start")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$h/R$")
    ax.set_title("Floating sphere settles on the Archimedes draft")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "relaxation.png"), dpi=140)
    plt.close(fig)

    rows = metrics["cross_dimension"]
    labels = ["h/R=%.2f\n%s" % (x["h_over_R"], x.get("context", "")) for x in rows]
    vals = [100.0 * x["rel_error_vs_3d"] for x in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(range(len(vals)), vals, color="C1")
    ax.axhline(15.0, color="C3", ls="--", lw=1.4, label="15 % accuracy gate")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("relative error of 2-D kernel used for a sphere [%]")
    ax.set_title("Cross-dimension misuse error")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cross_dimension_error.png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generated report section
# ---------------------------------------------------------------------------
def write_report(metrics):
    geo, eq, dyn, cross = (metrics["geometry"], metrics["equilibrium"],
                           metrics["dynamics"], metrics["cross_dimension"])
    lines = ["<!-- generated by scripts/audit_dem3d_buoyancy_float.py; do not hand-edit -->",
             "", "## Generated verification metrics", "",
             f"Sphere radius R = {R} m, rho_w = {RHO_W} kg/m3, g = {G} m/s2.", "",
             "### Geometry kernel", "",
             "| check | value |", "|---|---:|",
             f"| samples | {geo['n_samples']} |",
             f"| max rel. error vs 20 001-point Simpson quadrature | {geo['max_rel_err_vs_simpson']:.3e} |",
             f"| max abs. difference between two algebraic factorisations | {geo['max_abs_diff_two_factorisations']:.3e} |",
             f"| max rel. error of -dV/dh vs waterplane area pi(R^2-h^2) | {geo['max_rel_err_minus_dVdh_vs_waterplane']:.3e} |",
             f"| max abs. difference scalar vs tensor route | {geo['max_abs_diff_scalar_vs_tensor']:.3e} |",
             f"| monotone decreasing in h | {geo['monotone_decreasing']} |",
             f"| f(h=0) / f(h=R) / f(h=-R) | {geo['f_at_h_0']:.6f} / {geo['f_at_h_R']:.6f} / {geo['f_at_h_minus_R']:.6f} |",
             "", "### Archimedes equilibrium", "",
             "| rho_p [kg/m3] | r | h_eq/R | draft [m] | f(h_eq) | residual f-r | u (bisection) | u (poly root) | |u diff| | F_b/F_g rel. err | K [N/m] |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in eq:
        lines.append(
            "| {particle_density:.0f} | {density_ratio:.4f} | {u_eq:+.6f} | {draft_m:.6f} "
            "| {submerged_fraction:.9f} | {fraction_residual:+.2e} | {u_eq:+.9f} "
            "| {u_from_polynomial_root:+.9f} | {abs_diff_u_vs_polynomial:.2e} "
            "| {archimedes_force_rel_error:.2e} | {hydrostatic_stiffness_N_per_m:.4f} |".format(**row))
    lines += ["", "### Vertical dynamics (1-DOF)", "",
              "| quantity | value |", "|---|---:|",
              f"| particle mass [kg] | {dyn['mass_kg']:.6f} |",
              f"| analytic h_eq [m] | {dyn['h_eq_m']:.6f} |",
              f"| damped relaxation final h [m] | {dyn['damped_final_h_m']:.6f} |",
              f"| |h_final - h_eq| / R | {dyn['damped_error_over_R']:.3e} |",
              f"| drag coefficient [N s/m] | {dyn['drag_coeff_N_s_per_m']} |",
              f"| dt [s] / t_end [s] | {dyn['dt_s']} / {dyn['t_end_s']} |",
              f"| analytic period [s] | {dyn['analytic_period_s']:.6f} |",
              f"| measured period [s] | {dyn['measured_period_s']:.6f} |",
              f"| period rel. error | {dyn['period_rel_error']:.3e} |",
              f"| initial amplitude / R | {dyn['amplitude_over_R']} |",
              "", "### 2-D disk kernel misapplied to a sphere", "",
              "| h/R | f_2D | f_3D | rel. error | context |", "|---:|---:|---:|---:|---|"]
    for row in cross:
        lines.append("| {h_over_R:+.2f} | {f_2d_disk:.6f} | {f_3d_sphere:.6f} | "
                     "{rel_error_vs_3d:.2%} | {ctx} |".format(ctx=row.get("context", ""), **row))
    dr = metrics["cross_dimension_draft"]
    lines += ["", "### 2-D kernel substituted for a sphere at ice equilibrium", "",
              "| quantity | value |", "|---|---:|",
              f"| density ratio r | {dr['density_ratio']:.4f} |",
              f"| h_eq (2-D kernel) [m] | {dr['h_eq_2d_kernel_m']:.6f} |",
              f"| h_eq (3-D kernel) [m] | {dr['h_eq_3d_kernel_m']:.6f} |",
              f"| draft (2-D kernel) [m] | {dr['draft_2d_kernel_m']:.6f} |",
              f"| draft (3-D kernel) [m] | {dr['draft_3d_kernel_m']:.6f} |",
              f"| draft rel. error | {dr['draft_rel_error']:.4%} |"]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "REPORT.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    metrics = dict(geometry=geometry_checks(), equilibrium=equilibrium_checks(),
                   dynamics=dynamics(), cross_dimension=cross_dimension_error(),
                   cross_dimension_draft=cross_dimension_draft())
    metrics["case"] = dict(radius_m=R, water_density=RHO_W, gravity=G)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    figures(metrics)
    write_report(metrics)
    geo, dyn = metrics["geometry"], metrics["dynamics"]
    print(json.dumps({
        "max_rel_err_vs_simpson": geo["max_rel_err_vs_simpson"],
        "max_abs_diff_two_factorisations": geo["max_abs_diff_two_factorisations"],
        "max_rel_err_minus_dVdh": geo["max_rel_err_minus_dVdh_vs_waterplane"],
        "archimedes_force_rel_error_max": max(r["archimedes_force_rel_error"]
                                              for r in metrics["equilibrium"]),
        "equilibrium_u_diff_max": max(r["abs_diff_u_vs_polynomial"]
                                      for r in metrics["equilibrium"]),
        "damped_error_over_R": dyn["damped_error_over_R"],
        "period_rel_error": dyn["period_rel_error"],
        "draft_rel_error_2d_kernel": metrics["cross_dimension_draft"]["draft_rel_error"],
    }, indent=2))


if __name__ == "__main__":
    main()