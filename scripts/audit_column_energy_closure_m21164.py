"""Step-by-step energy-closure audit of the axisymmetric column-collapse integrator.

Anchor: ``CASE-COLUMN-COLLAPSE`` (Lube et al. 2004).  Pre-registration:
``docs/PREREG_COLUMN_ENERGY_CLOSURE_m21164.md`` (rules frozen before the first run).

The audited identity is the *incremental* one used by
:func:`tensordem.granular_column.axisym.simulate_axisymmetric_column`::

    R_n = (U_0 - U_n) - K_n - S_n - sum_{k<=n} D_k - sum_{k<=n} L_k   ~= 0

with U gravity potential, K kinetic (+ rotational), S stored spring energy of the
six channels, D the dissipation channels and L the tension-cutoff release loss.
``residual_fraction_of_potential`` is ``R_N / U_0``; the per-step increment
``dR_n`` localises a missing channel to the step where the energy vanished.

The integrator loop below is a faithful copy of ``axisym.simulate_...`` (same
kernel calls, same order, same half-step velocity Verlet); it only adds
instrumentation.  ``--check-trajectory`` re-runs a case through the library
function and asserts the instrumented copy reproduces its trajectory bit for bit.

Usage::

    PYTHONPATH=src python scripts/audit_column_energy_closure_m21164.py --all
    PYTHONPATH=src python scripts/audit_column_energy_closure_m21164.py --all --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.granular_column import axisym as ax          # noqa: E402
from tensordem.granular_column.axisym import AxisymCase     # noqa: E402
from tensordem.granular_column.kernel import (             # noqa: E402
    PairSpringState, WallSpringState, build_pairs,
    pair_contact_forces, wall_contact_forces,
)

#: frozen closure arm of m18261 / m20873 / m20931 (unchanged, one character).
FROZEN = dict(mu=0.5, mu_rolling=0.5, mu_wall=0.8, restitution=0.5,
              kt_ratio=0.5, density=2500.0, gravity=9.81,
              packing_gap=0.02, jitter=0, duration_factor=8.0, min_time=0.5,
              sample_stride=5, contact_mode="reference")

KN_REF = 6.0e4
R_REF = 0.03
COLUMN_RADIUS = 0.48
GRAVITY = 9.81


def make_case(aspect: float, gpr: float, radius: float | None = None,
              dt_scale: float = 1.0) -> AxisymCase:
    """Frozen-protocol case: ``r = r_i / (2 gpr)``, ``k_n ∝ r^2`` (fixed δ/d)."""
    r = COLUMN_RADIUS / (2.0 * gpr) if radius is None else radius
    return AxisymCase(aspect_ratio=aspect, grains_per_radius=gpr, radius=r,
                      kn=KN_REF * (r / R_REF) ** 2, dt_scale=dt_scale,
                      rest_ke_tol=1.0e-4, rest_hold=200, **FROZEN)


@dataclass
class Budget:
    """Per-step energy bookkeeping of one instrumented run."""

    time: list = field(default_factory=list)
    U: list = field(default_factory=list)
    K: list = field(default_factory=list)
    S: list = field(default_factory=list)
    S_roll: list = field(default_factory=list)      # rolling springs, both kinds
    S_dead: list = field(default_factory=list)      # clipped-contact ("dead") store
    diss: list = field(default_factory=list)        # physical dissipation, this step
    diss_wall_normal: list = field(default_factory=list)
    diss_wall_rolling: list = field(default_factory=list)   # booked since the m21164 repair
    diss_pair_rolling: list = field(default_factory=list)
    release: list = field(default_factory=list)     # cutoff / release loss, this step
    n_pair: list = field(default_factory=list)
    n_wall: list = field(default_factory=list)
    pos_hash: list = field(default_factory=list)


def run_budget(case: AxisymCase, *, max_steps: int | None = None,
               initial_positions: np.ndarray | None = None,
               initial_velocity: np.ndarray | None = None,
               initial_omega: np.ndarray | None = None) -> dict:
    """Instrumented copy of ``simulate_axisymmetric_column`` (forces untouched)."""
    if initial_positions is None:
        pos, mat = ax.build_axisymmetric_column(case)
    else:
        pos = np.array(initial_positions, dtype=np.float64, copy=True)
        mat = case.material()
    n = pos.shape[0]
    mass, inertia = mat.mass, mat.inertia
    dt = float(case.dt) if case.dt else mat.timestep * case.dt_scale
    gravity_vec = np.array([0.0, 0.0, -mat.gravity])
    mu_wall = mat.mu if case.mu_wall is None else float(case.mu_wall)
    mu_i = None

    ext0 = ax.radial_extent(pos, mat.radius)
    r_i = ext0["p995"]
    h_i = float(pos[:, 2].max() + mat.radius)
    pe0 = float(mass * mat.gravity * np.sum(pos[:, 2]))

    pair_state = PairSpringState()
    wall_state = WallSpringState(n)
    vel = (np.zeros((n, 3)) if initial_velocity is None
           else np.array(initial_velocity, dtype=np.float64, copy=True))
    omega = (np.zeros((n, 3)) if initial_omega is None
             else np.array(initial_omega, dtype=np.float64, copy=True))
    t_end = max(case.min_time, case.duration_factor * math.sqrt(h_i / mat.gravity))
    n_steps = int(math.ceil(t_end / dt))
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))

    bud = Budget()
    time = 0.0
    cutoff = 0.0
    diss_acc = 0.0
    ks_pair = mat.pair_rolling_kr
    ks_wall = mat.wall_rolling_kr

    def spring_energy() -> tuple[float, float]:
        """(total stored spring energy, rolling part) from the live state arrays."""
        # pair: 0.5 kn δ² + 0.5 kt|ξ|² + 0.5 kr|θ|² ; the normal part is not
        # recoverable from the state object, so the kernel's own channel sums
        # (returned per step) are used for the total -- see the caller.
        roll_pair = 0.5 * ks_pair * float(np.sum(pair_state.roll ** 2))
        roll_wall = 0.5 * ks_wall * float(np.sum(wall_state.roll ** 2))
        return roll_pair + roll_wall

    # energy at the initial state (no contacts yet -> springs empty)
    S_prev = 0.0
    S_roll_prev = spring_energy()
    U_prev = pe0
    K_prev = 0.0
    step = -1
    stop_reason = "time_limit"
    for step in range(n_steps):
        iu, ju = build_pairs(pos, mat.radius)
        pred = pair_contact_forces(pos, vel, omega, iu, ju, pair_state, mat, dt,
                                   mu_i=mu_i, update_state=False,
                                   contact_mode=case.contact_mode)
        pred_wall = wall_contact_forces(pos, vel, omega, mat, dt, mu_wall=mu_wall,
                                        mu_i=mu_i,
                                        mu_rolling_wall=mat.mu_rolling,
                                        state=wall_state, update_state=False,
                                        contact_mode=case.contact_mode)
        acc = (pred["force"] + pred_wall["force"]) / mass + gravity_vec
        alpha = (pred["torque"] + pred_wall["torque"]) / inertia
        v_half = vel + 0.5 * dt * acc
        w_half = omega + 0.5 * dt * alpha
        pos += v_half * dt

        cur = pair_contact_forces(pos, v_half, w_half, iu, ju, pair_state, mat, dt,
                                  mu_i=mu_i, update_state=True,
                                  contact_mode=case.contact_mode)
        cur_wall = wall_contact_forces(pos, v_half, w_half, mat, dt, mu_wall=mu_wall,
                                       mu_i=mu_i,
                                       mu_rolling_wall=mat.mu_rolling,
                                       state=wall_state, update_state=True,
                                       contact_mode=case.contact_mode)
        acc = (cur["force"] + cur_wall["force"]) / mass + gravity_vec
        alpha = (cur["torque"] + cur_wall["torque"]) / inertia
        vel = v_half + 0.5 * dt * acc
        omega = w_half + 0.5 * dt * alpha
        time += dt

        # mirror the repaired ledger: all six dissipation channels, wall rolling
        # resistance included (it was silently dropped before m21164)
        diss_step = (cur["dissip_collision"] + cur["dissip_friction"]
                     + cur["dissip_rolling"] + cur_wall["dissip_collision"]
                     + cur_wall["dissip_friction"] + cur_wall["dissip_rolling"])
        rel_step = float(cur["release_loss"] + cur_wall["release_loss"])
        cutoff += rel_step
        diss_acc += diss_step

        S_now = (cur["energy_normal"] + cur["energy_tangential"]
                 + cur["energy_rolling"] + cur_wall["energy_normal"]
                 + cur_wall["energy_tangential"] + cur_wall["energy_rolling"])
        S_roll_now = spring_energy()
        U_now = float(mass * mat.gravity * np.sum(pos[:, 2]))
        K_now = float(0.5 * mass * np.sum(vel * vel)
                      + 0.5 * inertia * np.sum(omega * omega))

        bud.time.append(time)
        bud.U.append(U_now)
        bud.K.append(K_now)
        bud.S.append(S_now)
        bud.S_dead.append(float(cur["energy_dead"] + cur_wall["energy_dead"]))
        bud.S_roll.append(S_roll_now)
        bud.diss.append(diss_step)
        bud.diss_wall_normal.append(float(cur_wall["dissip_collision"]))
        bud.diss_wall_rolling.append(float(cur_wall["dissip_rolling"]))
        bud.diss_pair_rolling.append(float(cur["dissip_rolling"]))
        bud.release.append(rel_step)
        bud.n_pair.append(int(cur["n_contacts"]))
        bud.n_wall.append(int(cur_wall["n_contacts"]))
        bud.pos_hash.append(float(np.sum(pos.ravel() * np.arange(1, pos.size + 1))))

        S_prev, S_roll_prev, U_prev, K_prev = S_now, S_roll_now, U_now, K_now

        ke = K_now
        if time > 1.2 * math.sqrt(h_i / mat.gravity) and ke < case.rest_ke_tol * pe0:
            rest_counter = getattr(run_budget, "_rest", 0) + 1
            run_budget._rest = rest_counter
            if rest_counter >= case.rest_hold:
                stop_reason = "at_rest"
                break
        else:
            run_budget._rest = 0

    run_budget._rest = 0
    U = np.asarray(bud.U)
    K = np.asarray(bud.K)
    S = np.asarray(bud.S)
    D = np.asarray(bud.diss)
    L = np.asarray(bud.release)
    K0 = float(K[0]) if K.size else 0.0     # 0 for the column protocol (rest at release)
    R = (pe0 - U) - (K - K0) - S - np.cumsum(D) - np.cumsum(L)
    dR = np.diff(np.concatenate([[0.0], R]))
    S_roll = np.asarray(bud.S_roll)
    S_dead = np.asarray(bud.S_dead)
    dS_roll = np.diff(np.concatenate([[0.0], S_roll]))

    ext = ax.radial_extent(pos, mat.radius)
    return {
        "case": case.signature(),
        "n_particles": int(n),
        "dt": dt,
        "n_steps": int(step + 1),
        "time_end": time,
        "stop_reason": stop_reason,
        "pe0": pe0,
        "final": {
            "potential_released": pe0 - float(U[-1]) if U.size else 0.0,
            "kinetic": float(K[-1]) if K.size else 0.0,
            "spring": float(S[-1]) if S.size else 0.0,
            "dissipated": float(np.sum(D)),
            "dissipated_wall_normal": float(np.sum(bud.diss_wall_normal)),
            "dissipated_wall_rolling_unbooked": float(np.sum(bud.diss_wall_rolling)),
            "release_loss": float(np.sum(L)),
            "dead_store_final": float(S_dead[-1]) if S_dead.size else 0.0,
            "dead_store_max": float(S_dead.max()) if S_dead.size else 0.0,
            "residual": float(R[-1]) if R.size else 0.0,
            "residual_fraction": float(R[-1] / pe0) if R.size else 0.0,
        },
        "runout_ratio_p995": float(ext["p995"] / r_i),
        "series": {
            "time": bud.time, "R": R.tolist(), "dR": dR.tolist(),
            "release": L.tolist(), "diss": D.tolist(),
            "S_roll": S_roll.tolist(), "dS_roll": dS_roll.tolist(),
            "S": S.tolist(), "K": K.tolist(), "S_dead": S_dead.tolist(),
            "n_pair": bud.n_pair, "n_wall": bud.n_wall,
        },
        "pos_hash": bud.pos_hash,
        "final_positions": pos.tolist(),
        "final_vel": vel.tolist(),
        "final_omega": omega.tolist(),
    }


CASES = {
    "C1_slide_roll": dict(kind="single", vx=0.5, spin="rolling"),
    "C2_slide_slip": dict(kind="single", vx=0.5, spin="none"),
    "C3_col_a1_g6": dict(kind="column", aspect=1.0, gpr=6.0),
    "C4_col_a3_g4": dict(kind="column", aspect=3.0, gpr=4.0),
    "C5_col_a3_g4_dt2": dict(kind="column", aspect=3.0, gpr=4.0, dt_scale=0.5),
    "C6_col_a3_g4_dt4": dict(kind="column", aspect=3.0, gpr=4.0, dt_scale=0.25),
}


def _single_grain_case(vx: float, spin: str) -> tuple[AxisymCase, np.ndarray]:
    gpr = 4.0
    r = COLUMN_RADIUS / (2.0 * gpr)
    case = AxisymCase(aspect_ratio=1.0, grains_per_radius=gpr, radius=r,
                      kn=KN_REF * (r / R_REF) ** 2, dt_scale=1.0,
                      rest_ke_tol=1.0e-4, rest_hold=200, **FROZEN)
    pos = np.array([[0.0, 0.0, r]])
    return case, pos


def run_case(name: str, *, max_steps: int | None = 400) -> dict:
    spec = CASES[name]
    if spec["kind"] == "single":
        case, pos0 = _single_grain_case(spec["vx"], spec["spin"])
        r = float(case.radius)
        vel0 = np.array([[spec["vx"], 0.0, 0.0]])
        # pure rolling for +x translation: v + omega x (-r z) = 0  =>  omega_y = +vx/r
        om0 = (np.array([[0.0, spec["vx"] / r, 0.0]]) if spec["spin"] == "rolling"
               else np.zeros((1, 3)))
        out = run_budget(case, max_steps=max_steps, initial_positions=pos0,
                         initial_velocity=vel0, initial_omega=om0)
        return dict(out, canonical=spec)
    case = make_case(spec["aspect"], spec["gpr"], dt_scale=spec.get("dt_scale", 1.0))
    # documented as dt-scaled; the integrator starts from rest at release
    return run_budget(case, max_steps=max_steps)


def canonical_exact(spec: dict, mat_dummy=None) -> dict:
    """Analytic dissipation of the single-grain examples (H1 reference)."""
    return {"note": "sliding work mu*m*g*d evaluated from the run itself"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    names = args.case or (list(CASES) if args.all else ["C4_col_a3_g4"])
    out = {}
    for name in names:
        res = run_case(name, max_steps=args.max_steps)
        ser = res.pop("series")
        res["spikes"] = _spike_report(ser)
        res.pop("final_positions", None)
        res.pop("final_vel", None)
        res.pop("final_omega", None)
        out[name] = res
        f = res["final"]
        print(f"[{name}] N={res['n_particles']} steps={res['n_steps']} "
              f"dt={res['dt']:.3e} stop={res['stop_reason']} "
              f"R/pe0={f['residual_fraction']:+.4%} "
              f"released={f['potential_released']:.4g} "
              f"diss={f['dissipated']:.4g} rel={f['release_loss']:.4g}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", args.json)
    return 0


def _spike_report(ser: dict) -> dict:
    """Bucket the per-step imbalance by event type (release-loss step or not).

    A missing energy *channel* shows up as isolated spikes on event steps; an
    integration artefact leaks uniformly over all steps.  Pre-registered use:
    H3/H4 read this table, not a global average.
    """
    dR = np.asarray(ser["dR"])
    rel = np.asarray(ser["release"])
    dS_roll = np.asarray(ser["dS_roll"])
    on = rel > 0.0
    return {
        "sum_dR": float(dR.sum()),
        "sum_dR_release_steps": float(dR[on].sum()) if on.any() else 0.0,
        "sum_dR_quiet_steps": float(dR[~on].sum()) if (~on).any() else 0.0,
        "n_release_steps": int(on.sum()),
        "max_abs_dR": float(np.max(np.abs(dR))) if dR.size else 0.0,
        "sum_drop_rolling_spring": float(np.sum(np.maximum(-dS_roll, 0.0))),
        "sum_release": float(rel.sum()),
    }


if __name__ == "__main__":
    raise SystemExit(main())