"""Closed-form verification of the friction/rolling channels of the DEM3D ledger.

The cumulative energy ledger (``tensordem.dem3d_energy``) accounted for drag,
normal-contact damping, tool damping and fracture release only; with
``friction_coefficient > 0`` or ``rolling_resistance_coefficient > 0`` the
Coulomb spring work and the rolling-resistance couple work used to be silently
absorbed by ``energy_balance_residual_J``.

This module scores both channels against independently derived closed forms of
two two-particle subsystems driven through the production ``step()``:

* case R - rolling resistance, ``mu = 0, mu_r > 0``.  A tracking external force
  holds the overlap constant, so ``F_n = k_c delta`` is constant and the
  tangential deceleration is ``a = mu_r F_n / m``.  Because the rolling couple
  can only *oppose* sliding, the sliding stops at ``t_stop = v0 / a`` and the
  travelled distance is ``s = v0^2 / (2 a)``; the work absorbed by the rolling
  channel is therefore ``-mu_r F_n s``, not the constant-deceleration value
  ``-mu_r F_n (v0 T - a dt T / 2)`` that a free-running closed form predicts
  (that expression would require the couple to *accelerate* the particle
  backwards after ``t_stop``, which the constitutive law does not do).

* case F - Coulomb stick, ``mu_r = 0``.  The tangential subsystem is a harmonic
  oscillator, ``v_t(t) = v0 cos(omega t)`` with ``omega = sqrt(k_t / m)``: the
  friction work is ``W_f = -0.5 m v0^2 sin^2(omega T)`` and the stored
  tangential spring energy is its exact opposite, so ``W_f + E_tang = 0``
  (recoverable work identity).  The normal channel of this subsystem is open
  (the pair is not held against the normal contact force), so the *absolute*
  ledger residual is dominated by the normal-spring exchange and is scored
  against the mechanical energy scale instead of an absolute joule floor.

Tolerances are stated as fractions of the energy scale of the case, which is the
level at which symplectic Euler resolves the channel (first order in ``dt``).
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


def _rolling(dt: float = DT, mu_r: float = MU_R) -> dict:
    key = f"R{dt}_{mu_r}"
    if key not in _CACHE:
        cfg = base_config(dt=dt, drag=0.0, contact_damping=0.0,
                          friction_coefficient=0.0,
                          rolling_resistance_coefficient=mu_r)
        solver = build_subsystem(
            cfg,
            positions=[[0.0, 0.0, 0.0], [2 * R - DELTA, 0.0, 0.0]],
            velocities=[[0.0, 0.0, 0.0], [0.0, V0_R, 0.0]],
            bonds=[],
        )
        solver.fixed = torch.tensor([True, False])
        m = cfg.mass
        F_n = cfg.contact_stiffness * DELTA
        a = mu_r * F_n / m
        steps = int(round(T_R / dt))
        track = torch.zeros(2, 3)
        track[1, 0] = F_n  # cancels the normal contact force on particle 1
        for _ in range(steps):
            solver.step(external_forces=track)
        diag = solver.diagnostics()
        v_T = float(solver.velocities[1, 1])
        t_stop = V0_R / a if a > 0 else math.inf
        s_stop = V0_R ** 2 / (2.0 * a) if a > 0 else 0.0
        _CACHE[key] = {
            "dt": dt, "m": m, "F_n": F_n, "a": a, "mu_r": mu_r,
            "v_T": v_T, "t_stop": t_stop, "s_stop": s_stop,
            "slide_distance": float(solver.positions[1, 1]),
            "D_r": diag["rolling_work_cumulative_J"],
            "D_r_closed": -mu_r * F_n * s_stop,
            "W_f": diag["friction_work_cumulative_J"],
            "residual": diag["energy_balance_residual_J"],
            "E_scale": abs(solver.reference_mechanical_energy_J)
            + abs(diag["external_work_cumulative_J"]) + abs(diag["mechanical_energy_J"]) + 1e-30,
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
            "M_scale": abs(solver.reference_mechanical_energy_J)
            + abs(diag["mechanical_energy_J"]) + 1e-30,
            "a_max": V0_F / omega,
            "a_stick": MU_F * cfg.contact_stiffness * DELTA / k_t,
        }
    return _CACHE[key]


def _rel(x: float, ref: float) -> float:
    return abs(x - ref) / max(abs(ref), 1e-30)


class RollingChannelClosedFormTests(unittest.TestCase):
    """Case R: rolling-resistance work against the stopping-distance law."""

    def test_velocity_is_arrested_by_rolling_resistance(self):
        """The couple can only oppose sliding: v0 = 0.1 m/s is brought to rest."""
        r = _rolling()
        # t_stop = 2.0e-3 s is well inside the 8e-3 s window, so the tangential
        # velocity must have collapsed; it settles at a few 1e-3 m/s (the
        # discrete couple removes mu_r F_n dt / m per step, so the last step
        # can only approach zero) -- 2% of v0 is the observable statement.
        self.assertLess(abs(r["v_T"]), 0.02 * V0_R,
                        f"v_T {r['v_T']:.6g} m/s: rolling resistance did not arrest the slide")

    def test_rolling_work_matches_stopping_distance(self):
        r = _rolling()
        self.assertLess(_rel(r["D_r"], r["D_r_closed"]), 0.05,
                        f"D_r {r['D_r']:.6g} vs -mu_r F_n v0^2/(2a) = {r['D_r_closed']:.6g}")

    def test_friction_channel_zero_when_mu_is_zero(self):
        self.assertLess(abs(_rolling()["W_f"]), 1e-12)

    def test_zero_rolling_coefficient_leaves_the_rolling_channel_empty(self):
        """Negative control: with mu_r = 0 the channel must not accumulate work."""
        free = _rolling(mu_r=0.0)
        self.assertEqual(free["D_r"], 0.0)
        # ... and the particle keeps sliding (no spurious tangential force).
        self.assertGreater(abs(free["v_T"]), 0.9 * V0_R)

    def test_rolling_work_error_shrinks_with_dt(self):
        """The channel work is first order in dt, so refining dt must not hurt."""
        coarse, fine = _rolling(DT), _rolling(DT / 2)
        err_coarse = _rel(coarse["D_r"], coarse["D_r_closed"])
        err_fine = _rel(fine["D_r"], fine["D_r_closed"])
        self.assertLess(err_fine, err_coarse, f"err(dt/2)={err_fine:.3g} >= err(dt)={err_coarse:.3g}")
        self.assertLess(err_fine, 0.03, f"refined error {err_fine:.3g} above 3%")

    def test_residual_is_small_on_the_case_energy_scale(self):
        """With both channels accounted the residual is a truncation-level term."""
        r = _rolling()
        self.assertLess(abs(r["residual"]), 0.01 * r["E_scale"],
                        f"residual {r['residual']:.6g} J above 1% of {r['E_scale']:.6g} J")


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
        self.assertLess(abs(f["W_f"] + f["E_tang"]), 0.05 * f["E_scale"],
                        f"W_f + E_tang = {f['W_f'] + f['E_tang']:.6g} J")

    def test_residual_is_small_relative_to_the_mechanical_energy(self):
        """Scored against the mechanical scale: the normal channel of this
        two-particle subsystem is deliberately open (no tracking force), so the
        absolute residual carries the normal-spring exchange and cannot be
        pinned to a fixed joule floor."""
        f = _friction()
        self.assertLess(abs(f["residual"]), 0.02 * f["M_scale"],
                        f"residual {f['residual']:.6g} J above 2% of {f['M_scale']:.6g} J")


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
        v = torch.clamp(V0_R - r["a"] * t, min=0.0)
        ax.plot(t.numpy(), v.numpy(), "r--", label="max(0, v0 - a t)")
        ax.axhline(r["v_T"], color="b", lw=0.8, label=f"sim v_T = {r['v_T']:.2e}")
        ax.set_xlabel("t [s]"); ax.set_ylabel("v_t [m/s]")
        ax.set_title(f"R: D_r err {_rel(r['D_r'], r['D_r_closed']):.1e}")
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
        ax.set_title(f"F: |W_f+E_tang|={abs(f['W_f'] + f['E_tang']):.1e} J")
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