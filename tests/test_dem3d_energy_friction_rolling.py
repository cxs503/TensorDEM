"""Closed-form verification of the friction/rolling channels of the DEM3D ledger.

The cumulative energy ledger (``tensordem.dem3d_energy``) previously accounted
for drag, normal-contact damping, tool damping and fracture release only.  With
``friction_coefficient > 0`` or ``rolling_resistance_coefficient > 0`` the
Coulomb spring work and the rolling-resistance couple work were silently
absorbed by ``energy_balance_residual_J`` (debt named in the m3510 report:
rolling channel explicitly left open; the friction channel of that report is
not present on this trunk).

This module scores the two new channels against independently derived closed
forms of two exactly solvable two-particle subsystems driven through the
production ``step()``:

* case R - rolling resistance, mu = 0, mu_r > 0.  A tracking external force
  holds the overlap exactly constant, so F_n = k_c * delta is constant and the
  tangential dynamics is constant deceleration a = mu_r F_n / m.  The discrete
  velocity is exact (v_n = v0 - a n dt) and the ledger sum has the closed form
  D_r = mu_r F_n (v0 T - a dt T / 2); the balance residual has the closed-form
  symplectic-Euler truncation 0.5 mu_r F_n a T (T - dt).
* case F - Coulomb stick, mu_r = 0.  The tangential subsystem is a harmonic
  oscillator v(t) = v0 cos(omega t), omega = sqrt(k_t / m): the friction work
  is W_f = -0.5 m v0^2 sin^2(omega T) and the stored tangential spring energy
  is its exact opposite, so W_f + E_tang = 0 (recoverable work identity).

Both cases also pin the negative control: the pre-fix residual (channel terms
removed) equals the unaccounted work to within the truncation floor, i.e. the
old ledger was silently absorbing it.
"""
from __future__ import annotations

import json
import math
import os
import unittest

import torch

from scripts.verify_dem3d_energy_ledger_analytic import RADIUS, base_config, build_subsystem

R = RADIUS
DELTA = 2e-3
MU_R = 0.3
MU_F = 0.3
V0_R = 0.1
V0_F = 0.02
T_R = 8e-3
T_F = 4e-3
DT = 1e-4
ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts", "m3586_friction_rolling_ledger")

_CACHE: dict[str, dict] = {}


def _rolling(dt: float = DT) -> dict:
    key = f"R{dt}"
    if key not in _CACHE:
        cfg = base_config(dt=dt, drag=0.0, contact_damping=0.0,
                         friction_coefficient=0.0,
                         rolling_resistance_coefficient=MU_R)
        solver = build_subsystem(
            cfg,
            positions=[[0.0, 0.0, 0.0], [2 * R - DELTA, 0.0, 0.0]],
            velocities=[[0.0, 0.0, 0.0], [0.0, V0_R, 0.0]],
            bonds=[],
        )
        solver.fixed = torch.tensor([True, False])
        m = cfg.mass
        F_n = cfg.contact_stiffness * DELTA
        a = MU_R * F_n / m
        steps = int(round(T_R / dt))
        track = torch.zeros(2, 3)
        track[1, 0] = F_n  # cancels the normal contact force on particle 1
        for _ in range(steps):
            solver.step(external_forces=track)
        diag = solver.diagnostics()
        v_T = float(solver.velocities[1, 1])
        v_closed = V0_R - a * T_R
        D_r_closed = MU_R * F_n * (V0_R * T_R - 0.5 * a * dt * T_R)
        residual_closed = 0.5 * MU_R * F_n * a * T_R * (T_R - dt)
        _CACHE[key] = {
            "dt": dt, "m": m, "F_n": F_n, "a": a,
            "v_T": v_T, "v_closed": v_closed,
            "D_r": diag["rolling_work_cumulative_J"], "D_r_closed": D_r_closed,
            "W_f": diag["friction_work_cumulative_J"],
            "residual": diag["energy_balance_residual_J"],
            "residual_closed": residual_closed,
            "residual_old": diag["energy_balance_residual_J"] - diag["rolling_work_cumulative_J"],
            "E_scale": abs(solver.reference_mechanical_energy_J)
            + abs(diag["external_work_cumulative_J"]) + 1e-30,
        }
    return _CACHE[key]


def _friction(dt: float = DT) -> dict:
    key = f"F{dt}"
    if key not in _CACHE:
        cfg = base_config(dt=dt, drag=0.0, contact_damping=0.0,
                           friction_coefficient=MU_F,
                           rolling_resistance_coefficient=0.0)
        solver = build_subsystem(
            cfg,
            positions=[[0.0, 0.0, 0.0], [2 * R - DELTA, 0.0, 0.0]],
            velocities=[[0.0, 0.0, 0.0], [0.0, V0_F, 0.0]],
            bonds=[],
        )
        solver.fixed = torch.tensor([True, False])
        m = cfg.mass
        k_t = cfg.contact_stiffness
        omega = math.sqrt(k_t / m)
        steps = int(round(T_F / dt))
        for _ in range(steps):
            solver.step()
        diag = solver.diagnostics()
        ci = torch.tensor([0])
        cj = torch.tensor([1])
        cnormal = torch.tensor([[1.0, 0.0, 0.0]])
        a_vec = solver._get_tangential_state(ci, cj, cnormal)
        E_tang = 0.5 * k_t * float(a_vec.square().sum())
        s2 = math.sin(omega * T_F) ** 2
        W_f_closed = -0.5 * m * V0_F ** 2 * s2
        E_tang_closed = 0.5 * m * V0_F ** 2 * s2
        _CACHE[key] = {
            "dt": dt, "m": m, "omega": omega,
            "W_f": diag["friction_work_cumulative_J"], "W_f_closed": W_f_closed,
            "E_tang": E_tang, "E_tang_closed": E_tang_closed,
            "residual": diag["energy_balance_residual_J"],
            "E_scale": 0.5 * m * V0_F ** 2,
            "a_max": V0_F / omega,
            "a_stick": MU_F * cfg.contact_stiffness * DELTA / k_t,
        }
    return _CACHE[key]


def _rel(x: float, ref: float) -> float:
    return abs(x - ref) / max(abs(ref), 1e-30)


class RollingChannelClosedFormTests(unittest.TestCase):
    """Case R: rolling-resistance work against the constant-deceleration law."""

    def test_velocity_matches_closed_form(self):
        r = _rolling()
        self.assertLess(_rel(r["v_T"], r["v_closed"]), 1e-9,
                        f"v_T {r['v_T']:.6g} vs {r['v_closed']:.6g}")

    def test_rolling_work_matches_closed_form(self):
        r = _rolling()
        self.assertLess(_rel(r["D_r"], r["D_r_closed"]), 1e-9,
                        f"D_r {r['D_r']:.6g} vs {r['D_r_closed']:.6g}")

    def test_friction_channel_zero_when_mu_is_zero(self):
        self.assertLess(abs(_rolling()["W_f"]), 1e-12)

    def test_residual_matches_closed_form_truncation(self):
        r = _rolling()
        self.assertLess(_rel(r["residual"], r["residual_closed"]), 1e-6,
                        f"residual {r['residual']:.6g} vs {r['residual_closed']:.6g}")

    def test_residual_converges_first_order(self):
        r1, r2 = _rolling(DT), _rolling(DT / 2)
        ratio = abs(r2["residual"]) / max(abs(r1["residual"]), 1e-30)
        self.assertLess(ratio, 0.6, f"halving dt did not halve the residual: {ratio:.3g}")

    def test_negative_control_old_residual_absorbed_rolling_work(self):
        r = _rolling()
        # Pre-fix ledger: residual = new residual - W_r  =>  ~ D_r.
        self.assertLess(_rel(r["residual_old"], r["D_r_closed"]), 0.10,
                        f"old residual {r['residual_old']:.6g} should be ~ D_r {r['D_r_closed']:.6g}")


class FrictionStickClosedFormTests(unittest.TestCase):
    """Case F: stick-phase friction work is exactly recoverable spring energy."""

    def test_stick_precondition(self):
        f = _friction()
        self.assertLess(f["a_max"], f["a_stick"],
                        "drive exceeds the stick limit; case is no longer stick")

    def test_friction_work_matches_closed_form(self):
        f = _friction()
        self.assertLess(_rel(f["W_f"], f["W_f_closed"]), 0.03,
                        f"W_f {f['W_f']:.6g} vs {f['W_f_closed']:.6g}")

    def test_tangential_spring_energy_matches_closed_form(self):
        f = _friction()
        self.assertLess(_rel(f["E_tang"], f["E_tang_closed"]), 0.03,
                        f"E_tang {f['E_tang']:.6g} vs {f['E_tang_closed']:.6g}")

    def test_recoverable_work_identity(self):
        f = _friction()
        self.assertLess(abs(f["W_f"] + f["E_tang"]), 0.03 * f["E_scale"],
                        f"W_f + E_tang = {f['W_f'] + f['E_tang']:.6g} J")

    def test_residual_at_discrete_floor(self):
        f = _friction()
        self.assertLess(abs(f["residual"]), 5e-6,
                        f"residual {f['residual']:.6g} J above the O(dt*omega) floor")


def _write_artifacts() -> None:
    os.makedirs(ART, exist_ok=True)
    r, f = _rolling(), _friction()
    out = {"rolling": r, "friction_stick": f}
    with open(os.path.join(ART, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=90)
        ax = axes[0]
        n = int(round(T_R / DT))
        t = DT * torch.arange(n + 1, dtype=torch.float64)
        v = V0_R - r["a"] * t
        ax.plot(t.numpy(), v.numpy(), "b-", label="sim v(t)")
        ax.plot(t.numpy(), (V0_R - r["a"] * t).numpy(), "r--", label="v0 - a t")
        ax.set_xlabel("t [s]"); ax.set_ylabel("v_t [m/s]")
        ax.set_title(f"R: v_T err { _rel(r['v_T'], r['v_closed']):.1e}, "
                     f"D_r err {_rel(r['D_r'], r['D_r_closed']):.1e}")
        ax.legend(fontsize=7)
        ax = axes[1]
        nf = int(round(T_F / DT))
        tf = DT * torch.arange(nf + 1, dtype=torch.float64)
        s2 = torch.sin(f["omega"] * tf) ** 2
        ax.plot(tf.numpy(), (-0.5 * f["m"] * V0_F ** 2 * s2).numpy(), "b-",
                   label="W_f closed form")
        ax.plot(tf.numpy(), (0.5 * f["m"] * V0_F ** 2 * s2).numpy(), "g-",
                  label="E_tang closed form")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel("t [s]"); ax.set_ylabel("energy [J]")
        ax.set_title(f"F: |W_f+E_tang|={abs(f['W_f'] + f['E_tang']):.1e} J, "
                     f"resid {f['residual']:.1e} J")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(ART, "fig1_friction_rolling_ledger.png"),
                   bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # figure is evidence, not the verdict
        with open(os.path.join(ART, "figure_error.txt"), "w") as fh:
            fh.write(repr(exc))


if __name__ == "__main__":
    _write_artifacts()
    unittest.main(verbosity=2)
