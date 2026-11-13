"""XVAL: TensorDEM static-Coulomb + COM-lever rolling law vs its OSS reference.

Anchor ``XVAL-DEM-ROUND2-STATIC-COULOMB-PLUS-COM-LEVER`` (mission m17674).

Reference = the OSS artifact ``TensorDEM/self``: the做法 is read from
``src/tensordem/contact/rolling_static.py`` (module docstring + pure functions)
and ``docs/GAP_DEM_CLUMP_ROUND2.md`` §2.1, and the published numbers from
``docs/GAP_DEM_CLUMP_ROUND2.md`` §3 / ``docs/AUDIT_m15187_GAP_DEM_CLUMP.md``.
The cross-validation is done against an **independent** re-implementation
(``oracle.py``, written from the equations only, no tensordem import) plus the
analytic closed forms, and against the published round-2 repose angles.

Legs
----
A   spec conformance on a 3x2x2x2x3x7 grid (mu_r, R, Fn, I_eff, dt, w/w_eps)
A2  lever geometry: sphere com==member, clump tip/flank ratio vs exact geometry
A3  engine coupling: single-sphere spin-down steps vs ``ceil(w0/w_eps)``
B   reproduction of the published round-2 repose angles (``--repose``)

Usage
-----
    python benchmarks/xval_round2_static_coulomb/run_xval.py --legs a,a2,a3
    python benchmarks/xval_round2_static_coulomb/run_xval.py --repose
    python benchmarks/xval_round2_static_coulomb/run_xval.py --aggregate
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "XVAL-DEM-R2-COULOMB-COM"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "gap_dem_clump"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle  # noqa: E402
import xval_spec as S  # noqa: E402
from tensordem.contact import rolling_static as rs  # noqa: E402
from tensordem.contact.clump_contact import ContactParams, BoxObstacle  # noqa: E402
from tensordem.particle.clump import ellipsoid_clump, single_sphere  # noqa: E402
from tensordem.particle.clump_dem import ClumpDEM  # noqa: E402


def _rel(a, b, floor=1e-300):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(np.abs(b), floor)))


# ---------------------------------------------------------------------------
# Leg A — spec conformance (pure functions vs independent oracle)
# ---------------------------------------------------------------------------
def leg_a(seed: int = 0) -> dict:
    """One panel per time step: the implementation takes a scalar ``dt``."""
    rng = np.random.default_rng(seed)
    rows = []
    err = {"torque": 0.0, "sat": 0.0, "res": 0.0, "eps": 0.0, "analytic": 0.0}
    n_cells = 0
    for dt in S.DT_GRID:
        cells = list(itertools.product(S.MU_ROLL_GRID, S.LEVER_GRID, S.FN_GRID,
                                       S.INERTIA_GRID, S.RATIO_GRID))
        mu = np.array([c[0] for c in cells])
        arm = np.array([c[1] for c in cells])
        fn = np.array([c[2] for c in cells])
        inert = np.array([c[3] for c in cells])
        ratio = np.array([c[4] for c in cells])

        limit = oracle.coulomb_limit(mu, arm, fn)
        w_eps = oracle.arrest_speed(limit, inert, dt)
        u = rng.normal(size=(len(cells), 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        w = (ratio * w_eps)[:, None] * u

        tau_impl = rs.rolling_torque(w, limit, inert, dt, model="coulomb")
        tau_ref = oracle.torque(w, limit, inert, dt)
        sat_impl = rs.rolling_saturation(w, limit, inert, dt, model="coulomb")
        sat_ref = oracle.saturation(w, limit, inert, dt)
        w_after = w + tau_impl * dt / inert[:, None]
        res_impl = np.linalg.norm(w_after, axis=1)
        res_ref = oracle.stop_residual(w, limit, inert, dt)
        eps_impl = rs.stopping_speed(limit, inert, dt)

        err["torque"] = max(err["torque"], _rel(tau_impl, tau_ref))
        err["sat"] = max(err["sat"], float(np.max(np.abs(sat_impl - sat_ref))))
        # scale by w_eps: the reference residual is *exactly* 0 below w_eps, so a
        # relative error there is undefined; w_eps is the law's own scale.
        err["res"] = max(err["res"], float(np.max(np.abs(res_impl - res_ref)
                                                 / np.maximum(w_eps, 1e-300))))
        # same difference expressed against the spin scale |w| (immune to the
        # catastrophic cancellation of w - w_eps at w/w_eps = 1e4)
        err["res_w"] = max(err.get("res_w", 0.0), float(np.max(
            np.abs(res_impl - res_ref) / np.maximum(ratio * w_eps, 1e-300))))
        err["eps"] = max(err["eps"], _rel(eps_impl, w_eps))
        err["analytic"] = max(err["analytic"], float(np.max(np.abs(
            sat_impl - np.minimum(1.0, ratio)))))
        n_cells += len(cells)
        for i, c in enumerate(cells):
            rows.append({
                "mu_roll": c[0], "lever_arm_m": c[1], "fn_N": c[2],
                "inertia_kgm2": c[3], "dt_s": dt, "w_over_w_eps": c[4],
                "w_eps_rad_s": float(w_eps[i]),
                "sat_impl": float(sat_impl[i]),
                "residual_over_w_eps": float(res_impl[i] / max(w_eps[i], 1e-300)),
                "tau_over_limit_impl": float(np.linalg.norm(tau_impl[i]) / max(limit[i], 1e-300)),
            })

    out = {
        "leg": "A_spec_conformance",
        "n_cells": n_cells,
        "max_rel_err_torque": err["torque"],
        "max_abs_err_saturation": err["sat"],
        "max_rel_err_residual_vs_w_eps": err["res"],
        "max_rel_err_residual_vs_w": err["res_w"],
        "max_rel_err_w_eps": err["eps"],
        "max_rel_err_saturation_vs_analytic_min": err["analytic"],
        "gate": {},
        "rows": rows,
    }
    T = S.TOL
    out["gate"] = {
        "P1": S.verdict(err["torque"], T["P1_torque_rel_err"]),
        "P2": S.verdict(err["sat"], T["P2_saturation_abs_err"]),
        # P3 restated: |Δresidual| / |w| (the residual itself vanishes below w_eps,
        # so a residual-relative metric is undefined there and ill-conditioned at
        # w/w_eps = 1e4 by cancellation: eps*ratio ~ 2e-12).
        "P3": S.verdict(err["res_w"], T["P3_residual_rel_err"]),
        "P2_analytic": S.verdict(err["analytic"], T["P2_saturation_abs_err"]),
    }
    return out


# ---------------------------------------------------------------------------
# Leg A2 — lever geometry
# ---------------------------------------------------------------------------
def _make_shape(ar: float):
    """Same factory as benchmarks/gap_dem_clump/run_gap_dem_clump.py:96."""
    if abs(ar - 1.0) < 1e-12:
        return lambda: single_sphere(S.BASE_RADIUS, S.DENSITY, name="sphere")
    n = max(3, int(round(2.0 * ar)))
    return lambda: ellipsoid_clump(ar * S.BASE_RADIUS, S.BASE_RADIUS, n_spheres=n,
                                   density=S.DENSITY, name=f"clump-AR{ar:.1f}")


def leg_a2(seed: int = 0) -> dict:
    """Lever geometry: sphere degeneracy, union-extent identity, anisotropy.

    The pre-registered closed form ``(a+r_mem)/(b+r_mem)`` assumed ``a`` is the
    members' centre line; ``ellipsoid_clump`` fixes the **union** long semi-axis
    to ``a`` and volume-matches the radii, so the well-posed statements are
    ``arm_tip == a`` (exact by construction) and ``flank lever == max_m
    hypot(c_mx, r_m)``.  The superseded pre-registered number is reported, not
    hidden (prereg deviation log in the report).
    """
    rng = np.random.default_rng(seed)
    r = S.BASE_RADIUS

    # (i) single sphere: com lever == member lever == r for arbitrary contacts
    com = np.zeros((1, 3))
    u = rng.normal(size=(64, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    pts = r * u
    arm_com = rs.rolling_lever_arm(pts, com, np.full(64, r), "com")
    arm_mem = rs.rolling_lever_arm(pts, com, np.full(64, r), "member")
    sphere_err = float(np.max(np.abs(arm_com - arm_mem)))
    sphere_vs_r = float(np.max(np.abs(arm_mem - r)))

    shapes, worst_tip, worst_def = [], 0.0, 0.0
    for ar in S.AR_LIST:
        p = _make_shape(ar)()
        local = np.asarray(p.local_centers, float)
        radii = np.asarray(p.radii, float)
        a_union = float(ar * S.BASE_RADIUS)
        tip = int(np.argmax(np.abs(local[:, 0])))
        d = np.zeros(3)
        d[0] = np.sign(local[tip, 0]) or 1.0
        pt_tip = (local[tip] + radii[tip] * d).reshape(1, 3)
        arm_tip = float(rs.rolling_lever_arm(pt_tip, com, radii[tip:tip + 1], "com")[0])
        # every member's largest arm about the long axis, from its own COM distance
        flank_arms = np.hypot(local[:, 0], radii)
        m = int(np.argmax(flank_arms))
        pt_flank = (local[m] + radii[m] * np.array([0.0, 1.0, 0.0])).reshape(1, 3)
        arm_flank = float(rs.rolling_lever_arm(pt_flank, com, radii[m:m + 1], "com")[0])
        def_err = max(abs(arm_tip - float(np.linalg.norm(pt_tip))),
                      abs(arm_flank - float(np.linalg.norm(pt_flank))))
        worst_def = max(worst_def, def_err)
        tip_err = abs(arm_tip - a_union) / a_union
        worst_tip = max(worst_tip, tip_err)
        b_ach = float(radii.max())                      # union transverse half-extent
        achieved_ar = a_union / b_ach
        # legacy member lever is blind to the position: ratio = radius ratio only
        arm_tip_mem = float(rs.rolling_lever_arm(pt_tip, com, radii[tip:tip + 1], "member")[0])
        arm_flank_mem = float(rs.rolling_lever_arm(pt_flank, com, radii[m:m + 1], "member")[0])
        shapes.append({
            "ar_param": ar,
            "n_members": int(p.n_members),
            "a_union_m": a_union,
            "b_union_m": b_ach,
            "achieved_ar": achieved_ar,
            "achieved_ar_over_nominal": achieved_ar / ar,
            "member_radius_min_m": float(radii.min()),
            "member_radius_max_m": float(radii.max()),
            "arm_tip_m": arm_tip,
            "arm_flank_max_m": arm_flank,
            "rel_err_tip_vs_union_semiaxis": tip_err,
            "ratio_com": arm_tip / arm_flank,
            "ratio_vs_achieved_ar_rel_err": abs(arm_tip / arm_flank / achieved_ar - 1.0),
            "ratio_member": arm_tip_mem / arm_flank_mem,
            "member_ratio_vs_achieved_ar_rel_err": abs(arm_tip_mem / arm_flank_mem / achieved_ar - 1.0),
            "rel_err_definitional": def_err,
            "flank_member_index": m,
            "member_at_midplane": bool(abs(local[m, 0]) <= 1e-12 * a_union),
            "min_surface_distance_to_com_m": float(np.min(np.linalg.norm(local, axis=1) - radii)),
            "clip_binds": bool(np.min(np.linalg.norm(local, axis=1) - radii) < radii.min()),
            "prereg_ratio_a_plus_r_superseded": float(
                (a_union + radii[tip]) / (S.BASE_RADIUS + radii[m])),
        })
    out = {
        "leg": "A2_lever_geometry",
        "sphere_max_abs_err_com_vs_member": sphere_err,
        "sphere_max_abs_err_member_vs_r": sphere_vs_r,
        "clump_max_abs_err_definitional": worst_def,
        "clump_worst_rel_err_tip_vs_union_semiaxis": worst_tip,
        "prereg_closed_form_status": "superseded: a is the union extent, not the member "
                                     "centre line; gate restated on arm_tip == a",
        "shapes": shapes,
        "gate": {
            "P4": S.verdict(sphere_err, S.TOL["P4_sphere_lever_abs"]),
            # P5a = pre-registered P5 restated as the only well-posed identity
            "P5a_definitional": S.verdict(worst_def, S.TOL["P5_lever_ratio_rel"]),
            # P5b = ADDED after pre-registration: the builder's union extent claim,
            # tolerance 1e-6 because the clump COM comes from a grid=72 voxel mass
            # property (accuracy ~1e-8 relative), not from a closed form.
            "P5b_tip_vs_union_semiaxis": S.verdict(worst_tip, 1e-6),
        },
    }
    return out


# ---------------------------------------------------------------------------
# Leg A3 — engine coupling: spin arrest on a floor
# ---------------------------------------------------------------------------
def _spin_down(model: str, dt: float, omega0: float, steps: int = 40,
               settle: int = 200) -> dict:
    r = S.BASE_RADIUS
    p = single_sphere(r, S.DENSITY)
    sim = ClumpDEM(
        [p], [np.array([0.0, 0.0, r])],
        params=ContactParams(kn=150.0, mu=0.5, mu_roll=S.MU_ROLL,
                             rolling_model=model, rolling_lever="com"),
        dt=dt,
        obstacles=[BoxObstacle(center=np.array([0.0, 0.0, -0.01]),
                               half=np.array([0.02, 0.02, 0.01]))],
    )
    for _ in range(settle):
        sim.step()
    sim.omega[0] = np.array([0.0, 0.0, omega0])   # spin about the contact normal
    inertia = float(sim.mean_inertia[0])
    ke0 = 0.5 * inertia * omega0 * omega0
    trace = []
    ke = []
    for _ in range(steps):
        sim.step()
        w = float(np.linalg.norm(sim.omega[0]))
        trace.append(w)
        ke.append(0.5 * inertia * w * w)
    arrested = next((i + 1 for i, w in enumerate(trace) if w <= 1e-12 * omega0), None)
    return {
        "model": model, "dt_s": dt, "omega0_rad_s": omega0,
        "inertia_kgm2": inertia, "mass_kg": float(sim.mass[0]),
        "omega_trace": trace,
        "ke_trace": ke,
        "steps_to_arrest": arrested,
        "ke_max_ratio": float(max(ke) / ke0),           # 1.0 == energy never grew
        "w_eps_closed_form": oracle.sphere_eps_omega(S.MU_ROLL, S.G, dt, r),
        "w_eps_engine": float(rs.stopping_speed(
            np.array([S.MU_ROLL * r * float(sim.mass[0]) * S.G]),
            np.array([inertia]), dt)[0]),
    }


def leg_a3() -> dict:
    cells = []
    for model in ("coulomb", "viscous"):
        for dt in (2e-5, 5e-5, 2e-4):
            w_eps = oracle.sphere_eps_omega(S.MU_ROLL, S.G, dt, S.BASE_RADIUS)
            for omega0 in (0.05, 0.4 * w_eps, 0.5, 5.0):
                c = _spin_down(model, dt, omega0, steps=60 if omega0 > 1.0 else 20)
                n_ref = float(np.maximum(1.0, np.ceil(omega0 / w_eps)))
                c["steps_closed_form"] = n_ref
                c["step_slack"] = (abs(c["steps_to_arrest"] - n_ref)
                                   if c["steps_to_arrest"] is not None else None)
                cells.append(c)
    stat = [c for c in cells if c["model"] == "coulomb"]
    visc = [c for c in cells if c["model"] == "viscous"]
    worst_slack = max((c["step_slack"] for c in stat), default=None)
    out = {
        "leg": "A3_engine_spin_down",
        "cells": cells,
        "worst_step_slack_static": worst_slack,
        "max_ke_ratio_static": max(c["ke_max_ratio"] for c in stat),
        "legacy_final_omega_max": max(c["omega_trace"][-1] for c in visc),
        "static_final_omega_max": max(c["omega_trace"][-1] for c in stat),
        "gate": {
            "P6_steps": S.verdict(worst_slack, S.TOL["P6_engine_step_slack"]),
            # one-sided: the dissipative law must never *add* kinetic energy
            "P6_energy": S.verdict(max(0.0, max(c["ke_max_ratio"] for c in stat) - 1.0),
                                   S.TOL["P6_energy_rel"]),
        },
    }
    return out


# ---------------------------------------------------------------------------
# Leg A4 — inertia model: the law's isotropic mean inertia vs the body tensor
# ---------------------------------------------------------------------------
def leg_a4() -> dict:
    """``mean_inertia = trace(I)/3`` (law) vs the per-axis inertia (integration).

    The engine integrates ``omega_dot = invI_world . torque`` with the *full*
    tensor but the law builds its arresting speed from the isotropic mean, so
    ``|omega| = w_eps`` stops the spin in one step only when the rolling axis
    has ``I_axis == mean`` (exact for a sphere).  Quantified here on the very
    clumps the round-2 scan uses.
    """
    dt = 5e-5
    mu = S.MU_ROLL
    fn = 1.0e-3
    rows = []
    for ar in S.AR_LIST:
        p = _make_shape(ar)()
        inertia = np.asarray(p.inertia, float)
        diag = np.diag(inertia)
        mean = float(np.trace(inertia) / 3.0)
        inv = np.linalg.inv(inertia)
        per_axis = []
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1.0
            limit = float(oracle.coulomb_limit(mu, p.bounding_radius, fn))
            w_eps = float(oracle.arrest_speed(limit, mean, dt))
            w0 = w_eps * e
            tau = rs.rolling_torque(w0.reshape(1, 3), np.array([limit]),
                                    np.array([mean]), dt, model="coulomb")[0]
            w_after = w0 + inv @ tau * dt
            per_axis.append({
                "axis": k, "inertia_kgm2": float(diag[k]),
                "mean_over_axis": float(mean / diag[k]),
                "residual_over_w_eps": float(np.linalg.norm(w_after) / w_eps),
            })
        rows.append({
            "ar_param": ar,
            "n_members": int(p.n_members),
            "inertia_diag": [float(v) for v in diag],
            "offdiag_max": float(np.max(np.abs(inertia - np.diag(diag)))),
            "mean_inertia": mean,
            "mean_over_long_axis": float(mean / diag[0]),
            "mean_over_transverse_axis": float(mean / diag[1]),
            "per_axis": per_axis,
            "max_residual_over_w_eps": max(x["residual_over_w_eps"] for x in per_axis),
        })
    sphere = rows[0]
    return {
        "leg": "A4_inertia_model",
        "rows": rows,
        "sphere_max_residual_over_w_eps": sphere["max_residual_over_w_eps"],
        "clump_worst_residual_over_w_eps": max(r["max_residual_over_w_eps"] for r in rows[1:]),
        "gate": {
            # exact one-step arrest is only claimed for an isotropic body
            "P8_sphere_exact": S.verdict(sphere["max_residual_over_w_eps"], 1e-12),
        },
    }


# ---------------------------------------------------------------------------
# Leg B — reproduction of the published round-2 repose angles
# ---------------------------------------------------------------------------
def leg_b(seeds=(0,)) -> dict:
    import run_round2_scan as r2  # benchmarks/gap_dem_clump (same protocol)

    fills = r2.fill_heights()
    rows = []
    for cfg in ("viscous/member", "coulomb/member", "coulomb/com"):
        for seed in seeds:
            t0 = time.time()
            row = r2.run_case(1.0, seed, cfg, fills)
            row["wall_seconds"] = round(time.time() - t0, 1)
            ref = S.REFERENCE_REPOSE_DEG[cfg]
            got = float(row["angle_profile_deg"])
            row["ref_deg"] = ref
            row["got_deg"] = got
            row["err_pct"] = 100.0 * (got - ref) / ref
            row["within_15pct"] = abs(row["err_pct"]) < S.TOL["P7_repose_pct"]
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("config", "seed", "ref_deg", "got_deg",
                                                  "err_pct", "within_15pct",
                                                  "quiesced", "wall_seconds")}))
            (OUT / "legs").mkdir(parents=True, exist_ok=True)
            (OUT / "legs" / f"B_case_{cfg.replace('/', '-')}_seed{seed}.json").write_text(
                json.dumps(row, indent=1))
    bit = [r for r in rows if r["config"] == S.BITWISE_REPOSE["config"]]
    out = {
        "leg": "B_repose_reproduction",
        "rows": rows,
        "max_abs_err_pct": max(abs(r["err_pct"]) for r in rows),
        "all_within_15pct": all(r["within_15pct"] for r in rows),
        "bitwise_abs_err_deg": (abs(bit[0]["got_deg"] - S.BITWISE_REPOSE["angle_profile_deg"])
                                if bit else None),
        "gate": {
            "P7_15pct": all(r["within_15pct"] for r in rows),
            "P7_bitwise": (S.verdict(abs(bit[0]["got_deg"]
                                        - S.BITWISE_REPOSE["angle_profile_deg"]),
                                     S.TOL["P7_bitwise_deg"]) if bit else None),
        },
    }
    return out


# ---------------------------------------------------------------------------
# aggregation + figures
# ---------------------------------------------------------------------------
def _env() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        text=True).strip()
    except Exception:                                     # pragma: no cover
        commit = "unknown"
    return {"python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "commit": commit,
            "branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                     cwd=ROOT, capture_output=True, text=True).stdout.strip()}


def make_figures(results: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figs = []

    # (1) saturation curve: plastic -> sticking, dt-invariant in w/w_eps
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = results.get("A")
    if a:
        rows = a["rows"]
        for dt in sorted({r["dt_s"] for r in rows}):
            sel = [r for r in rows if r["dt_s"] == dt and r["mu_roll"] == 0.3]
            xs = [r["w_over_w_eps"] for r in sel]
            ys = [r["sat_impl"] for r in sel]
            ax[0].plot(xs, ys, "o", ms=4, label=f"impl dt={dt:g} s")
        xs = np.logspace(-3, 4, 200)
        ax[0].plot(xs, np.minimum(1.0, xs), "-", color="k", lw=1.2,
                   label="min(1, w/w_eps)")
        ax[0].set_xscale("log")
        ax[0].set_xlabel(r"$\omega/\omega_\epsilon$")
        ax[0].set_ylabel(r"$|\tau|\,/\,(\mu_r R |F_n|)$")
        ax[0].set_title("plastic→sticking transition (dt collapse)")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3)
        xs = [r["w_over_w_eps"] for r in rows]
        ys = [max(r["residual_over_w_eps"], 1e-18) for r in rows]
        ax[1].loglog(xs, ys, ".", ms=3)
        grid = np.logspace(-3, 4, 200)
        ax[1].loglog(grid, np.maximum(grid - 1.0, 1e-18), "k-", lw=1.0,
                     label=r"$\max(0,\omega-\omega_\epsilon)/\omega_\epsilon$")
        ax[1].set_xlabel(r"$\omega/\omega_\epsilon$")
        ax[1].set_ylabel("one-step residual spin / $\\omega_\\epsilon$")
        ax[1].set_title("arrest identity (closed form vs implementation)")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = figdir / "xval_legA_saturation_residual.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"path": p, "caption": "Leg A: delivered Coulomb fraction and one-step arrest "
                                      "residual vs closed form, three time steps."})

    # (2) lever geometry
    a2 = results.get("A2")
    if a2:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        s = a2["shapes"]
        ar = np.array([x["ar_param"] for x in s])
        ax[0].plot(ar, [x["ratio_com"] for x in s], "o-", label="com lever (impl)")
        ax[0].plot(ar, [x["achieved_ar"] for x in s], "x--",
                   label="achieved aspect ratio a/(s b)")
        ax[0].plot(ar, [x["ratio_member"] for x in s], "s-", label="member lever (legacy)")
        ax[0].set_xlabel("nominal aspect ratio a/b")
        ax[0].set_ylabel("tip / flank moment arm")
        ax[0].set_title("shape-aware lever anisotropy")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        a4 = results.get("A4")
        if a4:
            ax[1].plot([r["ar_param"] for r in a4["rows"]],
                       [r["mean_over_long_axis"] for r in a4["rows"]], "o-",
                       label="mean / I_long")
            ax[1].plot([r["ar_param"] for r in a4["rows"]],
                       [r["mean_over_transverse_axis"] for r in a4["rows"]], "s-",
                       label="mean / I_transverse")
            ax[1].axhline(1.0, color="k", ls="--", lw=1)
        ax[1].set_xlabel("aspect ratio a/b")
        ax[1].set_ylabel("mean inertia / axis inertia")
        ax[1].set_title("isotropic mean inertia vs the body tensor")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        p = figdir / "xval_legA2_lever_inertia.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"path": p, "caption": "Leg A2/A4: COM-lever tip/flank anisotropy vs the achieved "
                                          "aspect ratio, and the law's isotropic mean inertia vs "
                                          "the body tensor."})

    # (3) engine spin-down
    a3 = results.get("A3")
    if a3:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for c in a3["cells"]:
            if c["dt_s"] != 5e-5:
                continue
            lbl = f"{c['model']} w0={c['omega0_rad_s']:.3g}"
            ax[0].plot(np.arange(1, len(c["omega_trace"]) + 1), c["omega_trace"],
                       label=lbl, lw=1.2)
        ax[0].set_yscale("log")
        ax[0].set_xlabel("step")
        ax[0].set_ylabel("|omega| [rad/s]")
        ax[0].set_title("spin-down on a floor (engine, dt=5e-5 s)")
        ax[0].legend(fontsize=6)
        ax[0].grid(alpha=0.3)
        stat = [c for c in a3["cells"] if c["model"] == "coulomb"]
        vv = [c["steps_to_arrest"] or np.nan for c in stat]
        rr = [c["steps_closed_form"] for c in stat]
        ax[1].plot(rr, vv, "o")
        lim = [1, max(rr) * 1.1]
        ax[1].plot(lim, lim, "k--", lw=1)
        ax[1].set_xlabel("closed form ceil(w0/w_eps)")
        ax[1].set_ylabel("engine steps to arrest")
        ax[1].set_title("engine vs closed form (static branch)")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        p = figdir / "xval_legA3_engine.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"path": p, "caption": "Leg A3: engine spin-down traces (static vs legacy) "
                                          "and engine arrest steps vs the closed form."})

    # (4) repose reproduction
    b = results.get("B")
    if b:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        cfgs = [r["config"] for r in b["rows"]]
        ref = [r["ref_deg"] for r in b["rows"]]
        got = [r["got_deg"] for r in b["rows"]]
        x = np.arange(len(cfgs))
        ax.bar(x - 0.18, ref, 0.36, label="published (OSS TensorDEM/self)")
        ax.bar(x + 0.18, got, 0.36, label="re-run in this worktree")
        for i, r in enumerate(b["rows"]):
            ax.annotate(f"{r['err_pct']:+.3f}%", (x[i], max(ref[i], got[i]) + 0.3),
                        ha="center", fontsize=8)
        ax.axhspan(22, 25, color="0.85", label="literature dry-sand band 22–25°")
        ax.set_xticks(x, cfgs, fontsize=8)
        ax.set_ylabel("repose angle [deg]")
        ax.set_title("round-2 repose reproduction (ar=1.0, seed=0)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = figdir / "xval_legB_repose.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"path": p, "caption": "Leg B: published round-2 repose angles vs the re-run "
                                          "in this worktree, with the literature band."})

    return [{"name": f["path"].name, "path": str(f["path"]), "caption": f["caption"]}
            for f in figs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="a,a2,a3,a4")
    ap.add_argument("--repose", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args(argv)

    (OUT / "legs").mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    results.setdefault("mission", 17674)
    results.setdefault("anchor", "XVAL-DEM-ROUND2-STATIC-COULOMB-PLUS-COM-LEVER")
    results["env"] = _env()

    if "a" in args.legs.split(","):
        results["A"] = leg_a()
        print("legA", json.dumps({k: v for k, v in results["A"].items() if k != "rows"}))
    if "a2" in args.legs.split(","):
        results["A2"] = leg_a2()
        print("legA2", json.dumps({k: v for k, v in results["A2"].items() if k != "shapes"}))
    if "a4" in args.legs.split(","):
        results["A4"] = leg_a4()
        print("legA4", json.dumps({k: v for k, v in results["A4"].items() if k != "rows"}))
    if "a3" in args.legs.split(","):
        results["A3"] = leg_a3()
        print("legA3", json.dumps({k: v for k, v in results["A3"].items() if k != "cells"}))
    if args.repose:
        results["B"] = leg_b(tuple(int(s) for s in args.seeds.split(",")))
        print("legB", json.dumps({k: v for k, v in results["B"].items() if k != "rows"}))

    (OUT / "legs").mkdir(parents=True, exist_ok=True)
    for k in ("A", "A2", "A3", "A4", "B"):
        if k in results:
            (OUT / "legs" / f"leg{k}.json").write_text(json.dumps(results[k], indent=1))

    if args.aggregate:
        results["figures"] = make_figures(results)
        gates = {k: results[k]["gate"] for k in ("A", "A2", "A3", "A4", "B") if k in results}
        results["gates"] = gates
        flat = [v for g in gates.values() for v in g.values() if v is not None]
        results["verdict"] = "PASS" if all(flat) else "FAIL"
        res_path.write_text(json.dumps(results, indent=1))
        print("verdict", results["verdict"], json.dumps(gates))

    res_path.write_text(json.dumps(results, indent=1))
    print("wrote", res_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())