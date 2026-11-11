"""True-triaxial bonded-particle DEM protocol for the 3-D ice engine.

This module drives :class:`tensordem.dem3d.IceDEM3D` through a two-phase
true-triaxial protocol:

Phase 1 (confinement)
    The four lateral (x, y) servo platens are driven by a force-feedback
    controller until the lateral pressures settle at the prescribed
    ``(sigma_x, sigma_y)``.  The axial (z) platens are held.

Phase 2 (axial loading)
    The top platen translates at constant velocity (bottom platen fixed for
    the ``triaxial`` protocol, or both platens oppose for the ``ucs_opposed``
    protocol used by the uniaxial benchmark).  The run stops at failure,
    defined as the axial stress falling below ``post_peak_fraction`` of the
    running peak, or at ``max_axial_steps``.

The Lode parameter is an *output*: the two lateral pressures are prescribed,
the axial stress is measured at failure, and

    b = (sigma_2 - sigma_3) / (sigma_1 - sigma_3)

with ``sigma_1`` the major (axial) principal stress, ``sigma_2`` the
intermediate (x) and ``sigma_3`` the minor (y) principal stress.  For
``sigma_x >= sigma_y`` this is the compression branch, ``b in [0, 1]``.

The module is deliberately independent of the 2-D implementation and adds no
domain-specific constants: every material and geometric parameter is an input.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Any

import torch

from .dem3d import DEM3DConfig, IceDEM3D
from .dem3d_energy import EnergyAuditedIceDEM3D
from .dem3d_energy_quadrature import relative_error


def servo_velocity(error: float, gain: float, v_max: float) -> float:
    """Clamped velocity-proportional servo law (shared pressure controller).

    ``v = gain * error`` saturated to ``+-v_max``.  ``error`` is a force or a
    force-proxy (pressure times area) residual; the sign convention is that a
    positive error drives the platen *inward* (compressing the specimen).
    """
    return max(-v_max, min(v_max, gain * error))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class TriaxialConfig:
    """Parameters for a true-triaxial bonded-particle DEM test (SI units)."""
    # --- specimen geometry (simple-cubic bonded lattice) ---
    nx: int = 4
    ny: int = 4
    nz: int = 3
    radius: float = 0.025
    # --- material / contact ---
    density: float = 917.0
    bond_stiffness: float = 2_000.0
    contact_stiffness: float = 5_000.0
    breaking_strain: float = 0.015
    shear_breaking_strain: float = 0.03
    platen_stiffness: float = 5_000.0
    platen_damping: float = 0.5
    contact_damping: float = 0.5
    drag: float = 0.1
    friction_coefficient: float = 0.0
    # --- loading ---
    loading_speed: float = 0.05          # axial platen speed (m/s)
    protocol: str = "triaxial"           # "triaxial" | "ucs_opposed"
    # --- prescribed lateral pressures (Pa) ---
    sigma_x: float = 0.0                 # sigma_2 (intermediate, x)
    sigma_y: float = 0.0                 # sigma_3 (minor, y)
    # --- servo controller ---
    servo_gain: float = 1.0e-4           # (m/s) per N of force error
    servo_v_max: float = 0.02            # clamp on servo speed (m/s)
    servo_tol: float = 0.02              # |F-F_target|/F_target settle tolerance
    confine_steps: int = 400             # max steps to establish confinement
    confine_settle: int = 40             # consecutive settled steps required
    # --- axial phase ---
    # ``False`` (legacy): the lateral platens are frozen once confinement is
    # over, so the axial phase is strain-controlled laterally and the *measured*
    # lateral stresses drift away from the prescribed ones.  ``True`` keeps the
    # pressure servo running, which is what a constant-confining-pressure
    # triaxial test means physically; ``_summary`` reports the measured values
    # either way.
    servo_during_axial: bool = False
    max_axial_steps: int = 1200
    post_peak_fraction: float = 0.55     # failure when stress < frac * peak
    peak_min_stress: float = 1.0         # Pa; ignore peaks below this
    # --- opt-in bond constitutive model ---
    # See docs/ice_bond_specimen_rate_dependence.md.
    bond_model: str = "linear"            # "linear" | "maxwell_wiechert"
    bond_equilibrium_fraction: float = 0.35
    bond_delayed_elastic_fraction: tuple[float, ...] = (0.45, 0.20)
    bond_relaxation_time_s: tuple[float, ...] = (0.5, 30.0)
    bond_creep_exponent: int = 0          # m; 0 = brittle strain criterion only
    bond_creep_time_s: float = 1.0        # t_ref
    bond_creep_force_fraction: float = 1.0
    # --- numerics ---
    dt_factor: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.protocol not in ("triaxial", "ucs_opposed"):
            raise ValueError("protocol must be 'triaxial' or 'ucs_opposed'")
        for name in ("nx", "ny", "nz"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be >= 2")
        if self.radius <= 0 or self.density <= 0:
            raise ValueError("radius and density must be positive")
        if self.sigma_x < 0 or self.sigma_y < 0:
            raise ValueError("confining pressures must be nonnegative")
        if self.protocol == "ucs_opposed" and (self.sigma_x or self.sigma_y):
            raise ValueError("ucs_opposed protocol requires zero lateral pressure")


def _face_areas(cfg: TriaxialConfig) -> tuple[float, float]:
    """(A_xz, A_yz): lateral face areas in m^2 (gross, particle-box based)."""
    a_xz = (cfg.ny * 2.0 * cfg.radius) * (cfg.nz * 2.0 * cfg.radius)
    a_yz = (cfg.nx * 2.0 * cfg.radius) * (cfg.nz * 2.0 * cfg.radius)
    return a_xz, a_yz


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
class TriaxialRunner:
    """Drive one true-triaxial test and return a machine-readable record."""

    def __init__(self, cfg: TriaxialConfig) -> None:
        self.cfg = cfg
        torch.set_num_threads(1)
        a_xz, a_yz = _face_areas(cfg)
        self.area_x = a_xz
        self.area_y = a_yz
        # Target lateral forces (total over the two faces of each pair).
        self.fx_target = cfg.sigma_x * a_xz
        self.fy_target = cfg.sigma_y * a_yz
        base = DEM3DConfig(
            nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
            radius=cfg.radius,
            density=cfg.density,
            bond_stiffness=cfg.bond_stiffness,
            contact_stiffness=cfg.contact_stiffness,
            breaking_strain=cfg.breaking_strain,
            shear_breaking_strain=cfg.shear_breaking_strain,
            platen_stiffness=cfg.platen_stiffness,
            platen_damping=cfg.platen_damping,
            contact_damping=cfg.contact_damping,
            drag=cfg.drag,
            friction_coefficient=cfg.friction_coefficient,
            top_platen_enabled=True,
            bottom_platen_enabled=True,
            top_platen_velocity=-cfg.loading_speed,
            bottom_platen_velocity=(cfg.loading_speed if cfg.protocol == "ucs_opposed" else 0.0),
            fix_x_edges=False,
            fix_bottom=False,
            lateral_platens=(cfg.sigma_x > 0.0 or cfg.sigma_y > 0.0),
            bond_model=cfg.bond_model,
            bond_equilibrium_fraction=cfg.bond_equilibrium_fraction,
            bond_delayed_elastic_fraction=cfg.bond_delayed_elastic_fraction,
            bond_relaxation_time_s=cfg.bond_relaxation_time_s,
            bond_creep_exponent=cfg.bond_creep_exponent,
            bond_creep_time_s=cfg.bond_creep_time_s,
            bond_creep_force_fraction=cfg.bond_creep_force_fraction,
        )
        self.sim = EnergyAuditedIceDEM3D(DEM3DConfig(
            **{**base.__dict__, "dt": base.recommended_dt * cfg.dt_factor}
        ))
        # Isolate the platen protocol: disable the spherical indenter.
        self.sim.tool_start[:] = torch.tensor([1.0e6, 1.0e6, 1.0e6], dtype=self.sim.dtype)
        self.sim.tool_velocity.zero_()
        self.initial_height = cfg.nz * 2.0 * cfg.radius
        self.gross_area = (cfg.nx * 2.0 * cfg.radius) * (cfg.ny * 2.0 * cfg.radius)
        self.rows: list[dict[str, float]] = []
        self.phase = "init"

    # -- helpers -----------------------------------------------------------
    def _reactions(self) -> tuple[float, float, float, float, float, float]:
        """(top_z, bottom_z, x_min, x_max, y_min, y_max) reactions, N.

        Read straight from the step-owner's last force evaluation, so no extra
        force pass is needed (``diagnostics()`` would re-evaluate every force).
        """
        sim = self.sim
        return (
            float(sim.last_top_platen_reaction_z),
            float(sim.last_bottom_platen_reaction_z),
            float(sim.last_lateral_platen_reaction[0]),
            float(sim.last_lateral_platen_reaction[1]),
            float(sim.last_lateral_platen_reaction[2]),
            float(sim.last_lateral_platen_reaction[3]),
        )

    def _axial_load(self, top: float, bottom: float) -> float:
        return 0.5 * (abs(top) + abs(bottom))

    def _lateral_forces(self, rxmin: float, rxmax: float,
                        rymin: float, rymax: float) -> tuple[float, float]:
        """Total compressive force in x and y (N), from per-plane reactions."""
        fx = -rxmin + rxmax   # x_min reaction is negative, x_max positive
        fy = -rymin + rymax
        return fx, fy

    def _servo(self, fx: float, fy: float) -> None:
        """Set lateral platen velocities to drive (fx, fy) to their targets."""
        sim = self.sim
        if not self.cfg.sigma_x and not self.cfg.sigma_y:
            sim.lateral_platen_velocity.zero_()
            return
        ex = self.fx_target - fx
        ey = self.fy_target - fy
        g = self.cfg.servo_gain
        vmax = self.cfg.servo_v_max
        # x_min moves +x into the specimen to raise x pressure; x_max moves -x.
        vx_min = servo_velocity(ex, g, vmax)
        vx_max = servo_velocity(-ex, g, vmax)
        vy_min = servo_velocity(ey, g, vmax)
        vy_max = servo_velocity(-ey, g, vmax)
        sim.lateral_platen_velocity[0] = vx_min
        sim.lateral_platen_velocity[1] = vx_max
        sim.lateral_platen_velocity[2] = vy_min
        sim.lateral_platen_velocity[3] = vy_max

    def _axial_closure(self) -> float:
        t = self.sim.time
        if self.cfg.protocol == "ucs_opposed":
            return 2.0 * self.cfg.loading_speed * t
        return self.cfg.loading_speed * t

    def _record(self, phase: str, load: float, fx: float, fy: float) -> None:
        strain = self._axial_closure() / self.initial_height
        self.rows.append({
            "step": int(self.sim.step_count),
            "time_s": float(self.sim.time),
            "phase": phase,
            "axial_engineering_strain": strain,
            "axial_load_N": load,
            "axial_stress_Pa": load / self.gross_area,
            "lateral_force_x_N": fx,
            "lateral_force_y_N": fy,
            "pressure_x_Pa": fx / self.area_x,
            "pressure_y_Pa": fy / self.area_y,
            "broken_bonds": int(self.sim.broken_bonds),
        })

    # -- phases ------------------------------------------------------------
    def confine(self) -> None:
        cfg = self.cfg
        self.phase = "confine"
        settled = 0
        for _ in range(cfg.confine_steps):
            self.sim.step()
            top, bot, rxmin, rxmax, rymin, rymax = self._reactions()
            load = self._axial_load(top, bot)
            fx, fy = self._lateral_forces(rxmin, rxmax, rymin, rymax)
            self._record("confine", load, fx, fy)
            self._servo(fx, fy)
            # settle check (only meaningful when a target is nonzero)
            if cfg.sigma_x > 0 or cfg.sigma_y > 0:
                tol = cfg.servo_tol
                ok = True
                if cfg.sigma_x > 0:
                    ok = ok and abs(fx - self.fx_target) <= tol * max(self.fx_target, 1e-12)
                if cfg.sigma_y > 0:
                    ok = ok and abs(fy - self.fy_target) <= tol * max(self.fy_target, 1e-12)
                settled = settled + 1 if ok else 0
                if settled >= cfg.confine_settle:
                    break
        # freeze the lateral platens for the axial phase
        self.sim.lateral_platen_velocity.zero_()
        self.phase = "axial"

    def load_axial(self) -> None:
        cfg = self.cfg
        peak = 0.0
        steps_since_peak = 0
        for _ in range(cfg.max_axial_steps):
            self.sim.step()
            top, bot, rxmin, rxmax, rymin, rymax = self._reactions()
            load = self._axial_load(top, bot)
            stress = load / self.gross_area
            fx, fy = self._lateral_forces(rxmin, rxmax, rymin, rymax)
            if cfg.servo_during_axial:
                self._servo(fx, fy)
            self._record("axial", load, fx, fy)
            if stress > peak:
                peak = stress
                steps_since_peak = 0
            else:
                steps_since_peak += 1
            if (peak > cfg.peak_min_stress
                    and stress < cfg.post_peak_fraction * peak
                    and steps_since_peak > 5):
                break
        self.phase = "done"

    def run(self) -> dict[str, Any]:
        if self.cfg.sigma_x > 0 or self.cfg.sigma_y > 0:
            self.confine()
        else:
            self.phase = "axial"
        self.load_axial()
        return self._summary()

    # -- summary -----------------------------------------------------------
    def _summary(self) -> dict[str, Any]:
        rows = self.rows
        axial = [r for r in rows if r["phase"] == "axial"]
        if not axial:
            axial = rows
        peak_row = max(axial, key=lambda r: r["axial_stress_Pa"])
        sigma1 = float(peak_row["axial_stress_Pa"])
        # principal stresses at peak: sigma1 = axial, sigma2 = sigma_x, sigma3 = sigma_y
        s2, s3 = self.cfg.sigma_x, self.cfg.sigma_y
        denom = sigma1 - s3
        b_lode = (s2 - s3) / denom if abs(denom) > 1e-12 else 0.0
        # The prescribed lateral pressures are inputs; the platen reactions are
        # the only measurement of the stress state that was actually reached.
        # ``b_lode`` above is therefore an *input* Lode parameter.  Report the
        # measured state and the Lode parameter it implies, so a reader can see
        # whether the run realised the intended true-triaxial condition.
        px_peak = float(peak_row["pressure_x_Pa"])
        py_peak = float(peak_row["pressure_y_Pa"])
        denom_meas = sigma1 - py_peak
        b_lode_measured = (px_peak - py_peak) / denom_meas if abs(denom_meas) > 1e-12 else 0.0
        confine_rows = [r for r in rows if r["phase"] == "confine"]
        conf = confine_rows[-1] if confine_rows else {
            "pressure_x_Pa": 0.0, "pressure_y_Pa": 0.0, "step": 0}
        tol = self.cfg.servo_tol

        def _rel(meas: float, target: float) -> float:
            return (meas - target) / target if target else 0.0

        conf_x = _rel(float(conf["pressure_x_Pa"]), self.cfg.sigma_x)
        conf_y = _rel(float(conf["pressure_y_Pa"]), self.cfg.sigma_y)
        converged = all(
            abs(v) <= tol for v, t in ((conf_x, self.cfg.sigma_x), (conf_y, self.cfg.sigma_y))
            if t > 0)
        peak_x = _rel(px_peak, self.cfg.sigma_x)
        peak_y = _rel(py_peak, self.cfg.sigma_y)
        stress_state_verified = all(
            abs(v) <= tol for v, t in ((peak_x, self.cfg.sigma_x), (peak_y, self.cfg.sigma_y))
            if t > 0)
        # energy ledger from a single final diagnostics() snapshot
        d = self.sim.diagnostics()
        e0 = float(getattr(self.sim, "reference_mechanical_energy_J", 0.0))
        e_t = float(self.sim._mechanical_energy())
        w_in = (float(self.sim.cumulative_tool_work_J)
                + float(self.sim.cumulative_external_work_J)
                + float(self.sim.cumulative_platen_work_J))
        diss = (float(self.sim.cumulative_drag_dissipation_J)
                + float(self.sim.cumulative_particle_damping_J)
                + float(self.sim.cumulative_tool_damping_J)
                + float(self.sim.cumulative_friction_work_J)
                + float(self.sim.cumulative_rolling_work_J)
                + float(self.sim.cumulative_platen_damping_J)
                + float(self.sim.cumulative_bond_dissipation_J))
        frac = float(self.sim.cumulative_fracture_release_J)
        resid = e_t - e0 - w_in + diss + frac
        scale = max(abs(e0) + abs(w_in) + frac, 1e-30)
        # Exact-quadrature arm: the boundary work is the potential increment of
        # the penalty springs at fixed particle positions instead of
        # ``reaction * v * dt`` (see docs/dem3d_energy_quadrature_closure.md).
        w_in_exact = (float(self.sim.cumulative_tool_work_exact_J)
                      + float(self.sim.cumulative_external_work_exact_J)
                      + float(self.sim.cumulative_platen_work_exact_J))
        resid_exact = e_t - e0 - w_in_exact + diss + frac
        scale_exact = max(abs(e0) + abs(w_in_exact) + frac, 1e-30)
        return {
            "energy_ledger": {
                "e_mech_0_J": e0,
                "e_mech_t_J": e_t,
                "input_work_J": w_in,
                "dissipation_J": diss,
                "fracture_release_J": frac,
                "closure_residual_J": float(resid),
                "closure_residual_relative": float(resid / scale),
                "gate": "closure_residual_relative <= 1e-9",
                "pass": abs(resid / scale) <= 1e-9,
                "input_work_exact_J": w_in_exact,
                "closure_residual_exact_J": float(resid_exact),
                "closure_residual_relative_exact": float(resid_exact / scale_exact),
                "pass_exact": abs(resid_exact / scale_exact) <= 1e-9,
                "quadrature": "left-endpoint (legacy) vs exact potential increment",
                # Closed-form cross-check of the quadrature defect: the exact
                # arm must exceed the legacy arm by sum(1/2 k n dz^2), which
                # dem3d_energy_quadrature predicts step by step.
                "legacy_deficit_predicted_J": float(
                    self.sim.cumulative_platen_deficit_predicted_J),
                "legacy_deficit_measured_J": float(
                    self.sim.cumulative_platen_work_exact_J
                    - self.sim.cumulative_platen_work_J),
                "legacy_deficit_relative_error": float(relative_error(
                    self.sim.cumulative_platen_work_exact_J
                    - self.sim.cumulative_platen_work_J,
                    self.sim.cumulative_platen_deficit_predicted_J)),
                "legacy_deficit_gate": "legacy_deficit_relative_error <= 0.05",
                # The closed form 1/2 k n dz^2 covers the *spring* part of the
                # platen force only.  A non-zero platen dashpot adds a work term
                # to the legacy arm that the penalty potential does not contain,
                # so the comparison is declared inapplicable (None) rather than
                # reported as a spurious failure.
                "legacy_deficit_scope": "spring-only; applies when platen_damping == 0",
                "legacy_deficit_applicable": bool(self.cfg.platen_damping == 0.0),
                "pass_legacy_deficit": (
                    bool(relative_error(
                        self.sim.cumulative_platen_work_exact_J
                        - self.sim.cumulative_platen_work_J,
                        self.sim.cumulative_platen_deficit_predicted_J) <= 0.05)
                    if self.cfg.platen_damping == 0.0 else None),
            },
            "config": asdict(self.cfg),
            "area_x_m2": self.area_x,
            "area_y_m2": self.area_y,
            "gross_area_m2": self.gross_area,
            "initial_height_m": self.initial_height,
            "sigma1_Pa": sigma1,
            "sigma2_Pa": s2,
            "sigma3_Pa": s3,
            "b_lode": float(b_lode),
            "b_lode_measured": float(b_lode_measured),
            "confinement": {
                "target_pressure_x_Pa": self.cfg.sigma_x,
                "target_pressure_y_Pa": self.cfg.sigma_y,
                "measured_pressure_x_Pa": float(conf["pressure_x_Pa"]),
                "measured_pressure_y_Pa": float(conf["pressure_y_Pa"]),
                "relative_error_x": float(conf_x),
                "relative_error_y": float(conf_y),
                "steps_used": int(conf["step"]),
                "tolerance": tol,
                "converged": bool(converged),
            },
            "stress_state_at_peak": {
                "prescribed_sigma2_Pa": s2,
                "prescribed_sigma3_Pa": s3,
                "measured_pressure_x_Pa": px_peak,
                "measured_pressure_y_Pa": py_peak,
                "lateral_relative_error_x": float(peak_x),
                "lateral_relative_error_y": float(peak_y),
                "tolerance": tol,
                "verified": bool(stress_state_verified),
                "servo_during_axial": bool(self.cfg.servo_during_axial),
            },
            "strain_at_peak": float(peak_row["axial_engineering_strain"]),
            "peak_load_N": float(peak_row["axial_load_N"]),
            "final_broken_bonds": int(self.sim.broken_bonds),
            "final_kinetic_energy_J": float(d["kinetic_energy_J"]),
            "final_mechanical_energy_J": float(d["mechanical_energy_J"]),
            "released_bond_energy_J": float(d["released_bond_energy_J"]),
            "final_lateral_reaction": [float(v) for v in d["lateral_platen_reaction"]],
            "n_steps": int(self.sim.step_count),
            "rows": rows,
        }


def run_triaxial(cfg: TriaxialConfig) -> dict[str, Any]:
    """Run a single true-triaxial test and return its record."""
    return TriaxialRunner(cfg).run()
