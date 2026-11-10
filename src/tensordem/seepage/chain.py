"""Closed-chain runner for seepage–piping verification (campaign p299w4-n02).

Runs all four verification links and returns the full evidence bundle with
pass/fail verdicts for every acceptance gate.

Acceptance criteria
-------------------
C1  Pore water pressure profile vs analytic hydrostatic: error < 10 %
C2  Critical hydraulic gradient i_c vs Terzaghi formula: error < 15 %
C3  Permeability k vs Kozeny–Carman formula: error < 20 %
C4  Fine-particle loss rate vs Skempton–Brogan trend: trend consistent
C5  Mass conservation (solid + liquid): error < 2 %
C6  pytest >= 12 all green (enforced by test suite, not here)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .effective_stress import (
    SaturatedColumn,
    effective_stress_profile,
    hydrostatic_pore_pressure,
)
from .critical_gradient import (
    PipingAnalysis,
    terzaghi_critical_gradient,
)
from .kozeny_carman import (
    KozenyCarmanModel,
    kozeny_carman_conductivity,
    kozeny_carman_porosity_sweep,
)
from .suffusion import (
    SkemptonBroganEnvelope,
    SuffusionAnalysis,
    skempton_brogan_critical_gradient,
    suffusion_erosion_rate,
)

__all__ = [
    "SeepagePrototype",
    "run_seepage_chain",
    "mass_conservation_audit",
]

#: Unit weight of water (N/m^3).
WATER_UNIT_WEIGHT = 9810.0
#: Water density (kg/m^3).
WATER_DENSITY = 1000.0
#: Gravity (m/s^2).
GRAVITY = 9.81


@dataclass(frozen=True)
class SeepagePrototype:
    """Physical set-up of the seepage verification column (SI).

    This is the reference geometry and material that all four verification
    links share.
    """

    height: float = 0.10              # column height (m)
    diameter: float = 2.0e-4          # particle diameter (m)
    rho_s: float = 2650.0             # solid density (kg/m^3)
    porosity: float = 0.40            # porosity
    fine_fraction: float = 0.20       # fine fraction for suffusion
    rho_w: float = WATER_DENSITY
    g: float = GRAVITY
    gamma_w: float = WATER_UNIT_WEIGHT
    mu_w: float = 1.0e-3              # water viscosity (Pa s)

    @property
    def gamma_sat(self) -> float:
        return (1.0 - self.porosity) * self.rho_s * self.g + \
               self.porosity * self.rho_w * self.g

    @property
    def gamma_prime(self) -> float:
        return self.gamma_sat - self.gamma_w

    @property
    def void_ratio(self) -> float:
        return self.porosity / (1.0 - self.porosity)

    @property
    def specific_gravity(self) -> float:
        return self.rho_s / self.rho_w

    @property
    def terzaghi_i_c(self) -> float:
        return terzaghi_critical_gradient(
            self.rho_s, self.porosity, self.rho_w, self.g,
        )


def mass_conservation_audit(
    *,
    initial_solid_mass: float,
    initial_fluid_mass: float,
    final_solid_mass: float,
    final_fluid_mass: float,
    fine_mass_lost: float = 0.0,
) -> dict:
    """Mass conservation audit for the seepage system (C5).

    Checks that the total mass change is within tolerance, accounting for
    fine particle loss through suffusion.

    Parameters
    ----------
    initial_solid_mass : float
        Initial solid mass (kg).
    initial_fluid_mass : float
        Initial fluid mass (kg).
    final_solid_mass : float
        Final solid mass (kg).
    final_fluid_mass : float
        Final fluid mass (kg).
    fine_mass_lost : float
        Mass of fine particles lost through suffusion (kg).

    Returns
    -------
    dict
        Mass conservation audit with pass/fail verdict.
    """
    initial_total = initial_solid_mass + initial_fluid_mass
    final_total = final_solid_mass + final_fluid_mass + fine_mass_lost
    if initial_total > 0.0:
        error = abs(final_total - initial_total) / initial_total
    else:
        error = 0.0
    return {
        "initial_solid_mass": initial_solid_mass,
        "initial_fluid_mass": initial_fluid_mass,
        "initial_total": initial_total,
        "final_solid_mass": final_solid_mass,
        "final_fluid_mass": final_fluid_mass,
        "fine_mass_lost": fine_mass_lost,
        "final_total_accounted": final_total,
        "mass_error_fraction": error,
        "mass_error_pct": error * 100.0,
        "threshold_pct": 2.0,
        "pass": bool(error <= 0.02),
    }


def run_seepage_chain(
    *,
    prototype: SeepagePrototype | None = None,
    measured_pore_pressures: dict | None = None,
    measured_critical_gradient: float | None = None,
    measured_permeabilities: dict | None = None,
    measured_erosion_rates: dict | None = None,
    mass_audit: dict | None = None,
    tolerance: dict | None = None,
) -> dict:
    """Run all four seepage verification gates and return the evidence bundle.

    Parameters
    ----------
    prototype : SeepagePrototype, optional
        Physical set-up.  Defaults to a standard quartz sand column.
    measured_pore_pressures : dict, optional
        {"z": [...], "u": [...]} from DEM or experiment.
    measured_critical_gradient : float, optional
        DEM-simulated critical gradient.
    measured_permeabilities : dict, optional
        {"porosities": [...], "k_measured": [...]} from DEM.
    measured_erosion_rates : dict, optional
        {"gradients": [...], "rates": [...]} from DEM.
    mass_audit : dict, optional
        Mass conservation data for C5.
    tolerance : dict, optional
        Override default tolerance thresholds.

    Returns
    -------
    dict
        Full evidence bundle with pass/fail for each gate.
    """
    proto = prototype or SeepagePrototype()
    tol = {
        "C1_pct": 10.0,
        "C2_pct": 15.0,
        "C3_pct": 20.0,
        "C5_pct": 2.0,
    }
    if tolerance:
        tol.update(tolerance)

    # -- C1: pore pressure profile vs analytic hydrostatic -------------------
    column = SaturatedColumn(
        height=proto.height,
        porosity=proto.porosity,
        rho_s=proto.rho_s,
        rho_w=proto.rho_w,
        g=proto.g,
    )
    c1 = {"pass": True, "max_deviation_pct": 0.0, "points": []}
    if measured_pore_pressures is not None:
        z_meas = np.asarray(measured_pore_pressures["z"], dtype=np.float64)
        u_meas = np.asarray(measured_pore_pressures["u"], dtype=np.float64)
        u_analytic = column.pore_pressure(z_meas)
        devs = []
        for zi, um, ua in zip(z_meas, u_meas, u_analytic):
            if ua > 0.0:
                dev = abs(um - ua) / ua * 100.0
            else:
                dev = abs(um - ua) * 100.0  # absolute error when ua = 0
            devs.append({
                "z": float(zi),
                "u_measured": float(um),
                "u_analytic": float(ua),
                "deviation_pct": float(dev),
            })
        max_dev = max(d["deviation_pct"] for d in devs) if devs else 0.0
        c1 = {
            "points": devs,
            "max_deviation_pct": max_dev,
            "threshold_pct": tol["C1_pct"],
            "pass": bool(max_dev <= tol["C1_pct"]),
        }

    # -- C2: critical hydraulic gradient vs Terzaghi -------------------------
    c2 = {"pass": True, "deviation_pct": 0.0}
    if measured_critical_gradient is not None:
        i_c_analytic = proto.terzaghi_i_c
        dev = abs(measured_critical_gradient - i_c_analytic) / i_c_analytic * 100.0
        c2 = {
            "i_c_measured": measured_critical_gradient,
            "i_c_terzaghi": i_c_analytic,
            "deviation_pct": dev,
            "threshold_pct": tol["C2_pct"],
            "pass": bool(dev <= tol["C2_pct"]),
        }

    # -- C3: permeability vs Kozeny–Carman -----------------------------------
    c3 = {"pass": True, "max_deviation_pct": 0.0, "points": []}
    if measured_permeabilities is not None:
        phis = measured_permeabilities["porosities"]
        k_meas = measured_permeabilities["k_measured"]
        kc_model = KozenyCarmanModel(
            diameter=proto.diameter,
            gamma_w=proto.gamma_w,
            mu_w=proto.mu_w,
        )
        devs = []
        for phi, km in zip(phis, k_meas):
            kp = kc_model.conductivity(phi)
            if km > 0.0:
                dev = abs(kp - km) / km * 100.0
            else:
                dev = float("inf")
            devs.append({
                "porosity": phi,
                "k_measured": km,
                "k_kozeny_carman": kp,
                "deviation_pct": dev,
            })
        max_dev = max(d["deviation_pct"] for d in devs) if devs else 0.0
        c3 = {
            "points": devs,
            "max_deviation_pct": max_dev,
            "threshold_pct": tol["C3_pct"],
            "pass": bool(max_dev <= tol["C3_pct"]),
        }

    # -- C4: suffusion trend vs Skempton–Brogan ------------------------------
    c4 = {"pass": True, "trend_consistent": True, "points": []}
    if measured_erosion_rates is not None:
        grads = measured_erosion_rates["gradients"]
        rates = measured_erosion_rates["rates"]
        env = SkemptonBroganEnvelope()
        i_c_suff = env.critical_gradient(proto.terzaghi_i_c, proto.fine_fraction)

        # Check trend: erosion rate should be ~0 below i_c_suff and increase above
        below_threshold = [(g, r) for g, r in zip(grads, rates) if g <= i_c_suff]
        above_threshold = [(g, r) for g, r in zip(grads, rates) if g > i_c_suff]

        below_ok = all(r < 1e-10 for _, r in below_threshold) if below_threshold else True
        above_ok = all(r > 0.0 for _, r in above_threshold) if above_threshold else True
        increasing = True
        if len(above_threshold) >= 2:
            above_grads = [g for g, _ in above_threshold]
            above_rates = [r for _, r in above_threshold]
            increasing = all(
                above_rates[i] <= above_rates[i + 1]
                for i in range(len(above_rates) - 1)
            )

        trend_ok = below_ok and above_ok and increasing
        points = []
        for g, r in zip(grads, rates):
            points.append({
                "gradient": g,
                "erosion_rate": r,
                "above_threshold": g > i_c_suff,
                "predicted_rate": suffusion_erosion_rate(g, i_c_suff),
            })
        c4 = {
            "points": points,
            "i_c_suffusion": i_c_suff,
            "below_threshold_zero": below_ok,
            "above_threshold_positive": above_ok,
            "monotonically_increasing": increasing,
            "trend_consistent": trend_ok,
            "pass": bool(trend_ok),
        }

    # -- C5: mass conservation -----------------------------------------------
    c5 = {"pass": True, "mass_error_pct": 0.0}
    if mass_audit is not None:
        c5 = mass_conservation_audit(**mass_audit)

    # -- Overall verdict -----------------------------------------------------
    all_pass = c1["pass"] and c2["pass"] and c3["pass"] and c4["pass"] and c5["pass"]

    return {
        "prototype": {
            "height": proto.height,
            "diameter": proto.diameter,
            "rho_s": proto.rho_s,
            "porosity": proto.porosity,
            "fine_fraction": proto.fine_fraction,
            "terzaghi_i_c": proto.terzaghi_i_c,
        },
        "C1_pore_pressure": c1,
        "C2_critical_gradient": c2,
        "C3_permeability": c3,
        "C4_suffusion_trend": c4,
        "C5_mass_conservation": c5,
        "overall_pass": all_pass,
    }
