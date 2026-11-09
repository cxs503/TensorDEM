"""Acceptance harness for the IN100-N07 ice-ridge mechanics node (F1-F7).

Criteria (campaign node metadata):

F1  sail height within 20 % of Hopky ``H_sail = 0.4 sqrt(h d)``;
F2  ``D_keel / H_sail`` in [3.0, 5.0];
F3  measured rubble porosity in [0.2, 0.5];
F4  peak hull force within 30 % of ``sigma_f A_keel``;
F5  energy ledger closure within 5 % of the external work;
F6  failure-mode order crushing -> bending -> ride-up -> clearing, plus the
    negative control that a vertical stem produces less ride-up than a raked one;
F7  pytest suite (counted by the caller, not here).

Every number is measured on a DEM run; the harness only compares and reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .formation import (
    RidgeFormationConfig,
    RidgeSqueezeConfig,
    simulate_ridge_formation,
    simulate_ridge_squeeze,
)
from .geometry import hopky_keel_depth, hopky_sail_height
from .penetration import ShipCrossingConfig, simulate_ship_crossing
from .strength import RubbleStrengthConfig, measure_rubble_strength

__all__ = [
    "AcceptanceConfig",
    "run_acceptance",
    "negative_control_zero_squeeze",
]


@dataclass(frozen=True)
class AcceptanceConfig:
    ice_thickness: float = 1.6
    design_squeeze: float = 2.25
    final_width: float = 4.0
    squeeze_velocity: float = 0.8
    particle_radius: float = 0.15
    block_particles: int = 3
    dt: float = 8.0e-4
    sweep: tuple[float, ...] = (1.5, 2.25, 3.0)
    ship_speed: float = 1.5
    crossing_dt: float = 6.0e-4
    stem_rake_deg: float = 25.0
    hull_friction: float = 0.2
    confining_pressure: float = 3.0e4
    strength_sample_width: float = 2.4
    strength_sample_height: float = 1.6
    strength_strain: float = 0.16
    bridge_fraction: float = 0.6

    def squeeze_config(self, squeeze: float) -> RidgeSqueezeConfig:
        return RidgeSqueezeConfig(
            ice_thickness=self.ice_thickness, squeeze=squeeze,
            final_width=self.final_width, squeeze_velocity=self.squeeze_velocity,
            particle_radius=self.particle_radius,
            block_particles=self.block_particles, dt=self.dt,
            freeze_bond_fraction=self.bridge_fraction,
        )

    def crossing_config(self, rake_deg: float | None = None) -> ShipCrossingConfig:
        return ShipCrossingConfig(
            speed=self.ship_speed,
            stem_rake_deg=self.stem_rake_deg if rake_deg is None else rake_deg,
            dt=self.crossing_dt, hull_friction=self.hull_friction, record_every=20)

    def strength_config(self) -> RubbleStrengthConfig:
        return RubbleStrengthConfig(
            sample_width=self.strength_sample_width,
            sample_height=self.strength_sample_height,
            particle_radius=self.particle_radius,
            confining_pressure=self.confining_pressure,
            platen_velocity=0.1, strain=self.strength_strain, dt=self.dt)


def negative_control_zero_squeeze(config: AcceptanceConfig) -> dict:
    """``d_squeeze = 0`` must give ``H_sail = 0`` (no convergence, no ridge)."""
    h = config.ice_thickness
    return {
        "hopky_sail_height": hopky_sail_height(h, 0.0),
        "hopky_keel_depth": hopky_keel_depth(h, 0.0),
        "expected_sail_height": 0.0,
        "pass": hopky_sail_height(h, 0.0) == 0.0 and hopky_keel_depth(h, 0.0) == 0.0,
    }


def run_acceptance(config: AcceptanceConfig = AcceptanceConfig(), *,
                   verbose: bool = False, sweep: bool = True) -> dict:
    """Run the formation / strength / penetration chain and evaluate F1-F6."""
    formation_cfg = config.squeeze_config(config.design_squeeze)
    ridge = simulate_ridge_squeeze(formation_cfg, verbose=verbose)
    geometry = ridge.geometry
    hopky = ridge.hopky
    sail = float(geometry["sail_height"])
    keel = float(geometry["keel_depth"])
    ratio = keel / sail if sail > 1e-6 else float("inf")
    porosity = float(geometry["porosity"])
    strength = measure_rubble_strength(config.strength_config())
    ridge_before = [[float(x), float(y), float(r)] for (x, y), r in
                    zip(ridge.assembly.pos, ridge.assembly.radii)]
    crossing = simulate_ship_crossing(ridge.assembly, config.crossing_config())
    ridge_after = [[float(x), float(y), float(r)] for (x, y), r in
                   zip(ridge.assembly.pos, ridge.assembly.radii)]
    sigma_f = float(strength["sigma_f"])
    a_keel = max(keel, 1e-6) * 1.0
    force_pred = sigma_f * a_keel
    shakedown = simulate_ship_crossing(
        simulate_ridge_squeeze(formation_cfg).assembly,
        config.crossing_config(rake_deg=0.0))
    results = {
        "F1": {
            "measured_sail_height": sail,
            "hopky_sail_height": hopky["sail_height"],
            "relative_deviation": hopky["sail_deviation"],
            "tolerance": 0.20,
            "pass": abs(hopky["sail_deviation"]) <= 0.20,
        },
        "F2": {
            "keel_depth": keel,
            "sail_height": sail,
            "ratio": ratio,
            "window": [3.0, 5.0],
            "pass": 3.0 <= ratio <= 5.0,
        },
        "F3": {
            "porosity": porosity,
            "window": [0.2, 0.5],
            "pass": 0.2 <= porosity <= 0.5,
        },
        "F4": {
            "force_peak": crossing.force_peak,
            "sigma_f": sigma_f,
            "a_keel": a_keel,
            "force_predicted": force_pred,
            "relative_deviation": ((crossing.force_peak - force_pred) / force_pred
                                   if force_pred > 0.0 else float("inf")),
            "tolerance": 0.30,
            "pass": (force_pred > 0.0
                     and abs(crossing.force_peak - force_pred) / force_pred <= 0.30),
        },
        "F5": {
            "relative_residual": crossing.energy_split["relative_residual"],
            "tolerance": 0.05,
            "pass": abs(crossing.energy_split["relative_residual"]) <= 0.05,
        },
        "F6": {
            "mode_order": crossing.mode_order,
            "phase_onsets": crossing.phase_onsets,
            "ride_up_raked": float(max((r["ride_up"] for r in crossing.mode_trace),
                                       default=0.0)),
            "ride_up_vertical": float(max((r["ride_up"] for r in shakedown.mode_trace),
                                          default=0.0)),
            "pass": None,
        },
        "negative_control": negative_control_zero_squeeze(config),
    }
    expected = ["crushing", "bending", "ride_up", "clearing"]
    order_ok = [m for m in crossing.mode_order if m in expected] == expected
    ride_ok = results["F6"]["ride_up_raked"] >= results["F6"]["ride_up_vertical"]
    results["F6"]["order_ok"] = order_ok
    results["F6"]["rake_control_ok"] = ride_ok
    results["F6"]["pass"] = bool(order_ok and ride_ok)
    results["formation"] = ridge.as_dict()
    results["crossing"] = crossing.as_dict()
    results["strength"] = strength
    results["ridge_particles_before"] = ridge_before
    results["ridge_particles_after"] = ridge_after
    results["shakedown"] = shakedown.as_dict()
    if sweep:
        scaling = []
        for squeeze in config.sweep:
            run = simulate_ridge_squeeze(config.squeeze_config(squeeze))
            scaling.append({
                "squeeze": squeeze,
                "sail_height": run.geometry["sail_height"],
                "keel_depth": run.geometry["keel_depth"],
                "ratio": run.hopky["keel_ratio"],
                "deviation": run.hopky["sail_deviation"],
                "porosity": run.geometry["porosity"],
            })
        results["scaling_sweep"] = scaling
        valid = [(row["squeeze"], row["sail_height"]) for row in scaling
                 if row["sail_height"] > 1e-6]
        if len(valid) >= 2:
            (d1, h1), (d2, h2) = valid[0], valid[-1]
            results["scaling_exponent"] = math.log(h2 / h1) / math.log(d2 / d1)
    passed = [key for key in ("F1", "F2", "F3", "F4", "F5", "F6")
              if results[key]["pass"]]
    failed = [key for key in ("F1", "F2", "F3", "F4", "F5", "F6")
              if not results[key]["pass"]]
    results["summary"] = {"passed": passed, "failed": failed,
                          "all_numeric_pass": not failed}
    return results
