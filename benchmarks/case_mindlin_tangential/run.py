#!/usr/bin/env python3
"""CASE-DEM-MINDLIN-TANGENTIAL -- Mindlin & Deresiewicz (1953) partial slip.

Benchmark driver for the tangential micro-slip -> gross-slip transition of a
single elastic sphere on a rigid frictional plane, solved with the *production*
DEM solver (``tensordem.dem3d.IceDEM3D``) using the opt-in
``tangential_model="mindlin_partial_slip"`` branch.

Reference
---------
Mindlin, R. D. & Deresiewicz, H. (1953), "Elastic Spheres in Contact Under
Varying Oblique Forces", *J. Appl. Mech.* 20, 327-344, DOI 10.1115/1.4010702.
The closed form used here,

    F_t = mu F_n [1 - (1 - delta/delta_s)^{3/2}],
    delta_s = 3 mu F_n / (16 G* a)             (a = sqrt(R* delta_n) here;
                                                equal spheres: 3 mu F_n (2-nu)/(8 G a))

is the classical partial-slip solution reproduced in Johnson, *Contact
Mechanics* (Cambridge, 1985), Sec. 4.3, eq. 4.32.  **The 1953 original is
paywalled and the sandbox has no network access, so no value is transcribed
from its tables**; every reference number below is (a) the closed form above
evaluated in this file with an *independent* parameter chain, and (b) two
self-consistency identities of the same law (initial slope = k_t, full
saturation = mu F_n).

Nothing in this file imports ``tensordem.mindlin_partial_slip``: the reference
is recomputed from Hertz first principles so that the comparison exercises the
solver's own constant chain (E*, G*, R*, a, k_t, delta_s) end to end.

Usage
-----
    PYTHONPATH=src python benchmarks/case_mindlin_tangential/run.py \
        --out artifacts/mindlin_tangential

Outputs (all relative to ``--out``): ``results.json``, ``curves.csv``,
``figures/*.png``, ``run.log`` (also echoed to stdout).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
import time
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensordem.dem3d import DEM3DConfig, IceDEM3D  # noqa: E402  (production solver)

# --------------------------------------------------------------------------
# Case definition (single contact, SI units)
# --------------------------------------------------------------------------
RADIUS = 0.025          # m, sphere radius
YOUNG = 1.0e9           # Pa, ice-like Young's modulus
POISSON = 0.3           # -
DENSITY = 917.0         # kg/m^3
MU = 0.3                # Coulomb friction coefficient (plane and sphere)
NORMAL_LOAD = 1.0       # N, applied normal load (gravity is switched off)
PLATEN_DAMPING = 2.0    # N s/m, normal dashpot of the plane contact
DRIVE_K = 2.0e4         # N/m, lateral drive spring (force ramp generator)
DRIVE_V = 1.0e-3        # m/s, drive spring end speed

GATE = 0.05             # acceptance: < 5 % of mu F_n


# --------------------------------------------------------------------------
# Independent reference chain (Hertz first principles, no tensordem imports)
# --------------------------------------------------------------------------
def reference_chain(mu=MU, normal_load=NORMAL_LOAD, young=YOUNG, nu=POISSON,
                    radius=RADIUS):
    """Sphere-on-rigid-plane Hertz/Mindlin constants, derived here from scratch.

    Returns a dict with the contact radius ``a``, the no-slip stiffness
    ``k_t = 8 G* a``, the gross-slip displacement in the *classical* form
    ``delta_s = 3 mu F_n (2 - nu) / (8 G a)`` and the normal overlap.
    """
    e_star = young / (1.0 - nu ** 2)                     # rigid counter-surface
    g = young / (2.0 * (1.0 + nu))                       # shear modulus
    g_star = young / (2.0 * (2.0 - nu) * (1.0 + nu))     # 1/G* = (2-nu)/G
    r_eff = radius
    delta_n = (3.0 * normal_load / (4.0 * e_star * math.sqrt(r_eff))) ** (2.0 / 3.0)
    a = math.sqrt(r_eff * delta_n)                       # Hertz: a^2 = R* delta
    k_n = (4.0 / 3.0) * e_star * math.sqrt(r_eff * delta_n)
    k_t = 8.0 * g_star * a
    # General partial-slip form delta_s = 3 mu F_n / (16 G* a); for two equal
    # spheres (G* = G/(2(2-nu))) this is the classical 3 mu F_n (2-nu)/(8 G a).
    delta_s = 3.0 * mu * normal_load / (16.0 * g_star * a)
    delta_s_alt = 3.0 * mu * normal_load / (2.0 * k_t)              # k_t form
    return {
        "E_star_Pa": e_star, "G_Pa": g, "G_star_Pa": g_star,
        "R_eff_m": r_eff, "a_m": a, "k_n_N_per_m": k_n,
        "k_t_N_per_m": k_t, "k_t_over_k_n": k_t / k_n,
        "delta_n_m": delta_n, "delta_s_m": delta_s,
        "delta_s_alt_form_m": delta_s_alt,
        "delta_s_form_identity_rel": abs(delta_s - delta_s_alt) / delta_s,
        "delta_s_over_delta_n": delta_s / delta_n,
        "a_over_R": a / radius, "delta_n_over_R": delta_n / radius,
        "mu_Fn_over_G_a2": mu * normal_load / (g * a * a),
        "mu": mu, "normal_load_N": normal_load, "young_modulus_Pa": young,
        "poisson_ratio": nu, "radius_m": radius,
        "mass_kg": DENSITY * 4.0 / 3.0 * math.pi * radius ** 3,
    }


def reference_force(delta, mu=MU, normal_load=NORMAL_LOAD, delta_s=None, **kw):
    """``F_t = mu F_n [1 - (1 - delta/delta_s)^{3/2}]`` (independent evaluator)."""
    if delta_s is None:
        delta_s = reference_chain(mu, normal_load, **kw)["delta_s_m"]
    x = np.clip(np.asarray(delta, dtype=float) / delta_s, 0.0, 1.0)
    return mu * normal_load * (1.0 - (1.0 - x) ** 1.5)


def reference_stick_radius_ratio(f_t, mu=MU, normal_load=NORMAL_LOAD):
    """``c/a = (1 - F_t/(mu F_n))^{1/3}`` -- inner stick-circle radius."""
    x = np.clip(1.0 - np.asarray(f_t, dtype=float) / (mu * normal_load), 0.0, 1.0)
    return x ** (1.0 / 3.0)


def reference_linear_cundall_strack(delta, k_t, mu=MU, normal_load=NORMAL_LOAD):
    """Linear-spring Coulomb clamp (the legacy model) for discrimination."""
    return np.minimum(k_t * np.asarray(delta, dtype=float), mu * normal_load)


# --------------------------------------------------------------------------
# Production-solver run: one sphere on the frictional bottom platen
# --------------------------------------------------------------------------
def run_case(*, model="mindlin_partial_slip", mu=MU, normal_load=NORMAL_LOAD,
             dt_factor=1.0, delta_stop=1.6, max_steps=60000, sample_every=1,
             verbose=False):
    """Ramp a lateral force on a sphere resting on the frictional plane.

    The sphere is pressed by a constant external load ``-N e_z`` onto the
    bottom platen (gravity off, so the contact normal force is exactly ``N`` in
    equilibrium) and pulled sideways by a soft drive spring whose end moves at
    constant speed.  The recorded tangential force is the solver's own platen
    friction force and ``delta`` is the solver's own accumulated tangential
    contact displacement, i.e. the state the constitutive law sees.

    Returns a dict of numpy arrays plus the solver's time step.
    """
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY,
        # a single physical contact: bonds muted, no lateral pinning, no gravity
        bond_stiffness=1e-6, fix_x_edges=False, fix_bottom=False,
        contact_model="hertz", young_modulus=YOUNG, poisson_ratio=POISSON,
        friction_coefficient=mu, platen_friction_coefficient=mu,
        bottom_platen_enabled=True, bottom_platen_gap=0.0,
        tangential_model=model,
        # The legacy comparator is given the *same initial slope* 8 G* a as the
        # Mindlin law (the platen branch otherwise defaults to the penalty
        # stiffness platen_stiffness), so the comparison isolates the branch
        # shape -- curvature and saturation -- and not a stiffness mismatch.
        platen_friction_stiffness=None if model == "mindlin_partial_slip"
        else reference_chain(mu, normal_load)["k_t_N_per_m"],

        restitution_coefficient=1.0, platen_damping=PLATEN_DAMPING,
        dt=None,
    )
    ref = reference_chain(mu, normal_load)
    sim = IceDEM3D(cfg)
    if dt_factor != 1.0:
        sim.dt = cfg.recommended_dt * dt_factor
    # Park the seven lattice neighbours far away (no gravity -> they are inert);
    # particle 0 is the single contact of interest.
    for idx in range(1, sim.positions.shape[0]):
        sim.positions[idx, 0] += 5.0 * idx
        sim.positions[idx, 1] += 5.0 * idx
    sim.positions[0, 0] = 0.0
    sim.positions[0, 1] = 0.0
    # The platen contact surface of the legacy platen branch is z = 0 (its
    # overlap is ``bottom_z + R - z`` with bottom_z the *centre* reference), so
    # a centre height of ``-delta_n`` is the Hertz equilibrium overlap.
    sim.positions[0, 2] = -ref["delta_n_m"]
    sim.velocities[:] = 0.0

    ext = torch.zeros_like(sim.positions)
    ext[0, 2] = -normal_load
    dt = sim.dt
    delta_stop_m = delta_stop * ref["delta_s_m"]
    rows = []
    t = 0.0
    k = 0
    for k in range(max_steps):
        x = float(sim.positions[0, 0])
        ext[0, 0] = DRIVE_K * (DRIVE_V * t - x)
        sim.step(ext)
        t += dt
        friction = sim._bottom_platen_tang_disp  # (n,3) accumulated tangential state
        delta = float(friction[0, 0])            # = particle travel on the fixed plane
        f_t = -float(sim.last_bottom_platen_friction[0, 0])   # reaction on the sphere
        if k % sample_every == 0:
            rows.append((t, delta, f_t, float(ext[0, 0])))
        if abs(delta) > delta_stop_m and k > 10:
            break
        if not math.isfinite(delta) or not math.isfinite(f_t):
            raise RuntimeError(f"run diverged at step {k} (delta={delta}, F_t={f_t})")
    if verbose:
        print(f"  run_case model={model} mu={mu} N={normal_load} dt_factor={dt_factor}: "
              f"{k + 1} steps, dt={dt:.4e} s, delta_max={rows[-1][1]:.4e} m")
    arr = np.asarray(rows, dtype=float)
    return {
        "t": arr[:, 0], "delta": np.abs(arr[:, 1]), "f_t": np.abs(arr[:, 2]),
        "f_ext": arr[:, 3], "dt": dt, "steps": k + 1,
        "wall_s": None,  # filled by the caller
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def fit_delta_s(delta, f_t, mu=MU, normal_load=NORMAL_LOAD):
    """Least-squares ``delta_s`` for the measured curve (single-parameter fit).

    Solves ``min_{delta_s} sum (F_meas - mu N [1 - (1 - delta/delta_s)^{3/2}])^2``
    on the micro-slip part (``delta < 1.2 delta_s_theory``) by a golden-section
    search on ``log delta_s``.  Used for the mu / N sensitivity scaling.
    """
    theory = reference_chain(mu, normal_load)["delta_s_m"]
    mask = delta < 1.2 * theory
    d = delta[mask]
    f = f_t[mask]
    if d.size < 5:
        return float("nan")

    def sse(log_ds):
        ref = reference_force(d, mu, normal_load, delta_s=math.exp(log_ds))
        return float(np.sum((f - ref) ** 2))

    lo, hi = math.log(theory / 20.0), math.log(theory * 20.0)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):
        c = hi - phi * (hi - lo)
        dd = lo + phi * (hi - lo)
        if sse(c) < sse(dd):
            hi = dd
        else:
            lo = c
    return math.exp(0.5 * (lo + hi))


def curve_metrics(delta, f_t, ref):
    """Error breakdown against the analytic partial-slip law."""
    mu_n = ref["mu"] * ref["normal_load_N"]
    d_s = ref["delta_s_m"]
    f_ref = reference_force(delta, ref["mu"], ref["normal_load_N"], delta_s=d_s)
    err = np.abs(f_t - f_ref) / mu_n
    micro = delta <= 0.9 * d_s
    trans = np.abs(delta - d_s) <= 0.1 * d_s
    gross = delta >= 1.1 * d_s
    # initial slope: secant estimate over the first 5 % of delta_s
    early = delta <= 0.05 * d_s
    k_t_meas = float(np.sum(delta[early] * f_t[early]) / np.sum(delta[early] ** 2)) \
        if early.sum() > 2 else float("nan")
    return {
        "f_ref": f_ref, "err_rel_muN": err,
        "max_micro_slip_err": float(err[micro].max()) if micro.any() else float("nan"),
        "max_transition_err": float(err[trans].max()) if trans.any() else float("nan"),
        "max_gross_slip_err": float(err[gross].max()) if gross.any() else float("nan"),
        "mean_gross_slip_err": float(err[gross].mean()) if gross.any() else float("nan"),
        "plateau_force_N": float(f_t[gross].mean()) if gross.any() else float("nan"),
        "k_t_secant_N_per_m": k_t_meas,
        "k_t_rel_err": abs(k_t_meas - ref["k_t_N_per_m"]) / ref["k_t_N_per_m"],
        "n_micro": int(micro.sum()), "n_gross": int(gross.sum()),
        "delta_s_fit_m": fit_delta_s(delta, f_t, ref["mu"], ref["normal_load_N"]),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(outdir, main, cs, ref, conv, sens_mu, sens_n, quick=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    made = []
    d_s = ref["delta_s_m"]
    mu_n = ref["mu"] * ref["normal_load_N"]

    # ---- fig 1: F_t vs delta (the case's central curve) ----
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=130)
    grid = np.linspace(0.0, main["delta"].max() * 1.02, 400)
    ax.plot(grid / d_s, reference_force(grid, ref["mu"], ref["normal_load_N"],
                                       delta_s=d_s) / mu_n,
            "k-", lw=2.0, label="Mindlin & Deresiewicz 1953 (analytic)")
    ax.plot(main["delta"] / d_s, main["f_t"] / mu_n, "o", ms=2.6, mfc="none",
            mec="tab:blue", label="TensorDEM dem3d, mindlin_partial_slip")
    ax.plot(cs["delta"] / d_s, cs["f_t"] / mu_n, "-", color="tab:red", lw=1.6,
            label="TensorDEM dem3d, cundall_strack (legacy)")
    ax.axvline(1.0, color="0.6", ls="--", lw=1.0)
    ax.text(1.02, 0.08, r"$\delta/\delta_s = 1$ (gross slip)", color="0.35", fontsize=9)
    ax.set_xlabel(r"normalised tangential displacement  $\delta/\delta_s$ [-]")
    ax.set_ylabel(r"normalised tangential force  $F_t/(\mu F_n)$ [-]")
    ax.set_title("CASE-DEM-MINDLIN-TANGENTIAL: micro-slip $\\rightarrow$ gross-slip")
    ax.set_xlim(0, main["delta"].max() / d_s)
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    p = os.path.join(figdir, "fig1_ft_delta_curve.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    made.append(p)

    # ---- fig 2: point-wise error vs delta/delta_s ----
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=130)
    m = main["metrics"]
    ax.semilogy(main["delta"] / d_s, np.maximum(m["err_rel_muN"], 1e-18), "-",
                color="tab:blue", lw=1.2, label="mindlin_partial_slip vs analytic")
    dev_cs = np.abs(cs["f_t"] - reference_force(cs["delta"], ref["mu"],
                                                ref["normal_load_N"], delta_s=d_s)) / mu_n
    ax.semilogy(cs["delta"] / d_s, np.maximum(dev_cs, 1e-18), "-", color="tab:red",
                lw=1.2, label="cundall_strack vs analytic")
    ax.axhline(GATE, color="k", ls="--", lw=1.0, label="gate 5 % of $\\mu F_n$")
    ax.set_xlabel(r"$\delta/\delta_s$ [-]")
    ax.set_ylabel(r"$|F_t - F_t^{ref}|/(\mu F_n)$ [-]")
    ax.set_title("Point-wise deviation from the Mindlin-Deresiewicz law")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)
    p = os.path.join(figdir, "fig2_error_vs_delta.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    made.append(p)

    # ---- fig 3: convergence and sensitivity ----
    n_panel = 3 if not quick else 2
    fig, axes = plt.subplots(1, n_panel, figsize=(4.0 * n_panel, 3.6), dpi=130)
    ax = axes[0]
    dts = [c["dt"] for c in conv]
    errs = [c["max_micro_slip_err"] for c in conv]
    ax.loglog(dts, np.maximum(errs, 1e-18), "o-", color="tab:green")
    ax.axhline(GATE, color="k", ls="--", lw=1.0)
    ax.set_xlabel("time step $\\Delta t$ [s]")
    ax.set_ylabel("max micro-slip error / $(\\mu F_n)$")
    ax.set_title("time-step convergence")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    mus = [s["mu"] for s in sens_mu]
    fitted = [s["delta_s_fit_m"] for s in sens_mu]
    theory = [reference_chain(s["mu"], NORMAL_LOAD)["delta_s_m"] for s in sens_mu]
    ax.plot(mus, np.array(fitted) * 1e6, "o", color="tab:purple", label="fit (solver)")
    ax.plot(mus, np.array(theory) * 1e6, "-", color="k", label=r"$3\mu F_n/(2k_t)$")
    ax.set_xlabel("friction coefficient $\\mu$ [-]")
    ax.set_ylabel("$\\delta_s$ [$\\mu$m]")
    ax.set_title("$\\delta_s$ vs $\\mu$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    if n_panel == 3:
        ax = axes[2]
        ns = [s["normal_load_N"] for s in sens_n]
        fitted = [s["delta_s_fit_m"] for s in sens_n]
        theory = [reference_chain(MU, n)["delta_s_m"] for n in ns]
        ax.loglog(ns, np.array(fitted) * 1e6, "o", color="tab:orange", label="fit (solver)")
        ax.loglog(ns, np.array(theory) * 1e6, "-", color="k", label=r"theory ($\propto F_n^{2/3}$)")
        ax.set_xlabel("normal load $F_n$ [N]")
        ax.set_ylabel("$\\delta_s$ [$\\mu$m]")
        ax.set_title("$\\delta_s$ vs $F_n$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    p = os.path.join(figdir, "fig3_dt_and_sensitivity.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    made.append(p)

    # ---- fig 4: contact-patch state map (stick circle c/a at the sampled points) ----
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=130)
    sel = np.linspace(0, len(main["delta"]) - 1, 9).astype(int)
    for i, s_i in enumerate(sel[1:]):
        d = main["delta"][s_i]
        c_over_a = float(reference_stick_radius_ratio(
            reference_force(d, ref["mu"], ref["normal_load_N"], delta_s=d_s),
            ref["mu"], ref["normal_load_N"]))
        a = float(ref["a_m"])
        # contact patch annulus: stick circle (white) inside the slip annulus (shaded)
        ax.add_patch(plt.Circle((i, 0.0), a * 1e3, color="tab:blue", alpha=0.30))
        ax.add_patch(plt.Circle((i, 0.0), a * c_over_a * 1e3, color="white"))
        ax.add_patch(plt.Circle((i, 0.0), a * c_over_a * 1e3, fill=False, ec="k", lw=0.7))
        ax.text(i, -a * 1.35e3, f"{d / d_s:.2f}", ha="center", fontsize=8)
    ax.set_xlim(-0.7, len(sel) - 0.3)
    ax.set_ylim(-1.35 * float(ref["a_m"]) * 1e3 * 1.25, 1.25 * float(ref["a_m"]) * 1e3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_xlabel(r"snapshot at $\delta/\delta_s$ (labels below)")
    ax.set_ylabel("contact patch, radius [mm]")
    ax.set_title("stick circle shrinking inside the slip annulus\n"
                 "(blue = slip annulus, white = no-slip stick circle, $c/a=(1-F_t/\\mu F_n)^{1/3}$)",
                 fontsize=9)
    p = os.path.join(figdir, "fig4_stick_circle_map.png")
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    made.append(p)
    return made


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/mindlin_tangential")
    ap.add_argument("--quick", action="store_true",
                    help="skip the uncertainty sweep (dt / mu / F_n)")
    ap.add_argument("--refresh-checksums", action="store_true",
                    help="recompute checksums.json for an existing --out dir "
                         "without re-running the solver")
    args = ap.parse_args(argv)
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "run.log")
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        print(msg, flush=True)

    if args.refresh_checksums:
        res_path = os.path.join(outdir, "results.json")
        res = json.load(open(res_path))
        res.pop("sha256", None)
        with open(res_path, "w") as fh:
            json.dump(res, fh, indent=1)
        files = [res_path, os.path.join(outdir, "curves.csv")]
        files += [os.path.join(outdir, rel) for rel in res.get("figures", [])]
        checks = {os.path.relpath(p, outdir): sha256(p) for p in files
                  if os.path.isfile(p)}
        with open(os.path.join(outdir, "checksums.json"), "w") as fh:
            json.dump({"note": "sha256 of every artifact, written after results.json",
                       "sha256": checks}, fh, indent=1)
        print(json.dumps(checks, indent=1))
        return 0

    t_start = time.time()
    log(f"[{_dt.datetime.now().isoformat(timespec='seconds')}] CASE-DEM-MINDLIN-TANGENTIAL")
    log(f"python {platform.python_version()}  torch {torch.__version__}  "
        f"numpy {np.__version__}  platform {platform.platform()}")

    ref = reference_chain()
    log("reference chain (independent Hertz): "
        + json.dumps({k: v for k, v in ref.items()
                      if isinstance(v, (int, float))}, indent=None))
    if ref["delta_s_form_identity_rel"] > 1e-12:
        raise RuntimeError("classical and k_t forms of delta_s disagree")

    # ---- production solver: Mindlin partial slip ----
    t0 = time.time()
    main_run = run_case(model="mindlin_partial_slip", verbose=True)
    main_wall = time.time() - t0
    main_run["metrics"] = curve_metrics(main_run["delta"], main_run["f_t"], ref)
    mm = main_run["metrics"]
    log(f"main run: {main_run['steps']} steps, {main_wall:.1f} s, "
        f"delta_max={main_run['delta'].max():.4e} m")
    log(f"  micro-slip  max err = {mm['max_micro_slip_err']:.3e} of mu*Fn "
        f"({'PASS' if mm['max_micro_slip_err'] < GATE else 'FAIL'}, gate {GATE})")
    log(f"  transition  max err = {mm['max_transition_err']:.3e}")
    log(f"  gross slip  max err = {mm['max_gross_slip_err']:.3e} "
        f"(plateau {mm['plateau_force_N']:.6f} N vs mu*Fn={MU * NORMAL_LOAD:.6f} N)")
    log(f"  initial slope k_t = {mm['k_t_secant_N_per_m']:.4e} N/m vs "
        f"8G*a = {ref['k_t_N_per_m']:.4e} N/m (rel {mm['k_t_rel_err']:.3e})")
    log(f"  fitted delta_s = {mm['delta_s_fit_m']:.4e} m vs "
        f"theory {ref['delta_s_m']:.4e} m "
        f"(rel {abs(mm['delta_s_fit_m'] - ref['delta_s_m']) / ref['delta_s_m']:.3e})")

    # ---- legacy linear model on the identical case (discrimination) ----
    t0 = time.time()
    cs_run = run_case(model="cundall_strack", verbose=True)
    cs_wall = time.time() - t0
    f_cs_ref = reference_force(cs_run["delta"], MU, NORMAL_LOAD, delta_s=ref["delta_s_m"])
    cs_dev = np.abs(cs_run["f_t"] - f_cs_ref) / (MU * NORMAL_LOAD)
    cs_max = float(cs_dev.max())
    log(f"legacy cundall_strack: max deviation {cs_max:.3%} of mu*Fn "
        f"({'case discriminates >5%' if cs_max > GATE else 'case does NOT discriminate'})")

    # the analytic linear model, evaluated independently of the solver
    grid = np.linspace(0.0, 0.9 * ref["delta_s_m"], 2001)
    lin_dev = np.abs(reference_linear_cundall_strack(grid, ref["k_t_N_per_m"])
                     - reference_force(grid, MU, NORMAL_LOAD,
                                       delta_s=ref["delta_s_m"])) / (MU * NORMAL_LOAD)
    lin_dev_max = float(lin_dev.max())
    lin_dev_at_half = float(np.abs(
        reference_linear_cundall_strack(0.5 * ref["delta_s_m"], ref["k_t_N_per_m"])
        - reference_force(0.5 * ref["delta_s_m"], MU, NORMAL_LOAD,
                          delta_s=ref["delta_s_m"])) / (MU * NORMAL_LOAD))
    log(f"analytic linear law vs MD: max {lin_dev_max:.3%} of mu*Fn, "
        f"at delta=0.5 delta_s {lin_dev_at_half:.3%} "
        f"(1-2^-1.5 = {1 - 0.5 ** 1.5:.4%})")

    conv, sens_mu, sens_n = [], [], []
    if not args.quick:
        for factor in (0.5, 1.0, 2.0):
            t0 = time.time()
            run = run_case(model="mindlin_partial_slip", dt_factor=factor,
                           delta_stop=1.2, verbose=True)
            met = curve_metrics(run["delta"], run["f_t"], ref)
            conv.append({"dt": run["dt"], "dt_factor": factor,
                         "steps": run["steps"], "wall_s": time.time() - t0,
                         "max_micro_slip_err": met["max_micro_slip_err"],
                         "max_gross_slip_err": met["max_gross_slip_err"]})
            log(f"  dt={run['dt']:.4e} s (x{factor}): micro err "
                f"{met['max_micro_slip_err']:.3e}, gross err "
                f"{met['max_gross_slip_err']:.3e}")
        for mu in (0.2, 0.3, 0.5):
            run = run_case(model="mindlin_partial_slip", mu=mu, delta_stop=1.2,
                           verbose=False)
            met = curve_metrics(run["delta"], run["f_t"],
                                reference_chain(mu, NORMAL_LOAD))
            s_mu = met["delta_s_fit_m"]
            s_th = reference_chain(mu, NORMAL_LOAD)["delta_s_m"]
            sens_mu.append({"mu": mu, "delta_s_fit_m": s_mu,
                            "delta_s_theory_m": s_th,
                            "rel_err": abs(s_mu - s_th) / s_th,
                            "max_micro_slip_err": met["max_micro_slip_err"]})
            log(f"  mu={mu}: delta_s fit {s_mu:.4e} vs theory {s_th:.4e} "
                f"(rel {abs(s_mu - s_th) / s_th:.2e})")
        for n_load in (0.5, 1.0, 2.0):
            run = run_case(model="mindlin_partial_slip", normal_load=n_load,
                           delta_stop=1.2, verbose=False)
            r_n = reference_chain(MU, n_load)
            met = curve_metrics(run["delta"], run["f_t"], r_n)
            s_n = met["delta_s_fit_m"]
            sens_n.append({"normal_load_N": n_load, "delta_s_fit_m": s_n,
                           "delta_s_theory_m": r_n["delta_s_m"],
                           "rel_err": abs(s_n - r_n["delta_s_m"]) / r_n["delta_s_m"],
                           "max_micro_slip_err": met["max_micro_slip_err"]})
            log(f"  Fn={n_load} N: delta_s fit {s_n:.4e} vs theory "
                f"{r_n['delta_s_m']:.4e} (rel {abs(s_n - r_n['delta_s_m']) / r_n['delta_s_m']:.2e})")
    else:
        log("quick mode: uncertainty sweep skipped")

    # scaling exponents of the two sensitivity sweeps
    def exponent(xs, ys):
        x = np.log(np.asarray(xs, dtype=float))
        y = np.log(np.asarray(ys, dtype=float))
        return float(np.polyfit(x, y, 1)[0])

    exp_mu = exponent([s["mu"] for s in sens_mu], [s["delta_s_fit_m"] for s in sens_mu]) \
        if sens_mu else float("nan")
    exp_n = exponent([s["normal_load_N"] for s in sens_n],
                     [s["delta_s_fit_m"] for s in sens_n]) if sens_n else float("nan")
    if sens_mu:
        log(f"  scaling: delta_s ~ mu^{exp_mu:.4f} (theory 1)   "
            f"delta_s ~ Fn^{exp_n:.4f} (theory 2/3 = 0.6667)")
        conv_slope = (np.polyfit(np.log([c["dt"] for c in conv]),
                                 np.log([max(c["max_micro_slip_err"], 1e-18) for c in conv]),
                                 1)[0] if conv else float("nan"))
    else:
        conv_slope = float("nan")

    figures = make_figures(outdir, {"delta": main_run["delta"], "f_t": main_run["f_t"],
                                    "metrics": mm}, cs_run, ref, conv, sens_mu,
                           sens_n, quick=args.quick)

    # ---- curves.csv ----
    csv_path = os.path.join(outdir, "curves.csv")
    grid_full = np.linspace(0.0, main_run["delta"].max(), 1200)
    ref_full = reference_force(grid_full, MU, NORMAL_LOAD, delta_s=ref["delta_s_m"])
    with open(csv_path, "w") as fh:
        fh.write("# CASE-DEM-MINDLIN-TANGENTIAL  curves\n")
        fh.write(f"# delta_s = {ref['delta_s_m']:.12e} m ; mu*Fn = {MU * NORMAL_LOAD:.12e} N\n")
        fh.write("delta_solver_m,Ft_solver_N,delta_normalised,Ft_normalised,"
                 "delta_analytic_m,Ft_analytic_N,c_over_a_analytic\n")
        for d, f in zip(main_run["delta"], main_run["f_t"]):
            c = float(reference_stick_radius_ratio(f, MU, NORMAL_LOAD))
            fh.write(f"{d:.12e},{f:.12e},{d / ref['delta_s_m']:.12e},"
                     f"{f / (MU * NORMAL_LOAD):.12e},,,{c:.12e}\n")
        for d, f in zip(grid_full, ref_full):
            c = float(reference_stick_radius_ratio(f, MU, NORMAL_LOAD))
            fh.write(f",,,,{d:.12e},{f:.12e},{c:.12e}\n")

    criteria = {
        "micro_slip_max_err_lt_5pct": mm["max_micro_slip_err"] < GATE,
        "transition_max_err_lt_5pct": mm["max_transition_err"] < GATE,
        "gross_slip_plateau_within_1pct": mm["max_gross_slip_err"] < 0.01,
        "initial_slope_within_2pct": mm["k_t_rel_err"] < 0.02,
        "fitted_delta_s_within_5pct": (abs(mm["delta_s_fit_m"] - ref["delta_s_m"])
                                       / ref["delta_s_m"]) < 0.05,
        "case_discriminates_linear_gt_5pct": cs_max > GATE,
    }
    total_wall = time.time() - t_start
    results = {
        "case": "CASE-DEM-MINDLIN-TANGENTIAL",
        "reference": {
            "citation": "Mindlin, R. D. & Deresiewicz, H. (1953) J. Appl. Mech. "
                        "20:327-344, DOI 10.1115/1.4010702",
            "closed_form": "F_t = mu F_n [1 - (1 - delta/delta_s)^{3/2}], "
                           "delta_s = 3 mu F_n / (16 G* a); equal spheres: "
                           "3 mu F_n (2-nu)/(8 G a)",
            "secondary_source": "Johnson, Contact Mechanics (1985) Sec. 4.3 eq. 4.32",
            "provenance": "closed form evaluated in run.py from Hertz first "
                          "principles; the 1953 original is paywalled and no "
                          "network access exists in the sandbox, so no table "
                          "value is transcribed",
            "chain": ref,
        },
        "solver": {
            "class": "tensordem.dem3d.IceDEM3D",
            "tangential_model": "mindlin_partial_slip",
            "dt_s": main_run["dt"], "steps": main_run["steps"],
            "wall_s": main_wall, "drive_k_N_per_m": DRIVE_K,
            "drive_v_m_per_s": DRIVE_V, "platen_damping_Ns_per_m": PLATEN_DAMPING,
        },
        "got_mindlin_partial_slip": {
            "n_samples": int(main_run["delta"].size),
            "delta_max_m": float(main_run["delta"].max()),
            "max_micro_slip_err": mm["max_micro_slip_err"],
            "max_transition_err": mm["max_transition_err"],
            "max_gross_slip_err": mm["max_gross_slip_err"],
            "mean_gross_slip_err": mm["mean_gross_slip_err"],
            "plateau_force_N": mm["plateau_force_N"],
            "mu_Fn_N": MU * NORMAL_LOAD,
            "k_t_secant_N_per_m": mm["k_t_secant_N_per_m"],
            "k_t_theory_N_per_m": ref["k_t_N_per_m"],
            "k_t_rel_err": mm["k_t_rel_err"],
            "delta_s_fit_m": mm["delta_s_fit_m"],
            "delta_s_theory_m": ref["delta_s_m"],
        },
        "comparison_legacy_cundall_strack": {
            "max_dev_vs_analytic_muN": cs_max,
            "analytic_linear_max_dev_muN": lin_dev_max,
            "analytic_linear_dev_at_half_delta_s": lin_dev_at_half,
            "note": "1 - 2^-1.5 = 0.6464 vs 0.5 -> the linear spring is off by "
                    "14.6 % of mu F_n at delta = delta_s/2",
        },
        "convergence": conv, "convergence_order_fit": conv_slope,
        "sensitivity_mu": sens_mu, "sensitivity_Fn": sens_n,
        "scaling_exponent_delta_s_mu": exp_mu,
        "scaling_exponent_delta_s_Fn": exp_n,
        "criteria": criteria,
        "all_criteria_pass": all(criteria.values()),
        "gate": GATE, "wall_s": total_wall,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
        },
        "figures": [os.path.relpath(p, outdir) for p in figures],
    }
    results_path = os.path.join(outdir, "results.json")
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=1)

    # checksums for the artifact bundle (external manifest: results.json cannot
    # hash itself, and run.py must be re-runnable without changing them)
    checks = {os.path.relpath(p, outdir): sha256(p)
              for p in [results_path, csv_path] + figures}
    with open(os.path.join(outdir, "checksums.json"), "w") as fh:
        json.dump({"note": "sha256 of every artifact, written after results.json",
                   "sha256": checks}, fh, indent=1)

    log("sha256: " + json.dumps(checks, indent=1))
    log(f"criteria: {sum(criteria.values())}/{len(criteria)} pass  -> "
        + json.dumps(criteria))
    log(f"total wall {total_wall:.1f} s")
    with open(log_path, "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(json.dumps({"out": outdir, "all_criteria_pass": all(criteria.values()),
                      "max_micro_slip_err": mm["max_micro_slip_err"],
                      "cundall_strack_max_dev": cs_max,
                      "wall_s": total_wall}, indent=1))
    return 0 if all(criteria.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())