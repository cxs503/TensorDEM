"""Unit tests for the pipe/pump/deposition faces (A2, A3, A4, A5, A7)."""

import math

import numpy as np
import pytest

from tensordem.slurry_transport.critical_velocity import (
    CLEANAWAY_TABLE, DURAND_FL_TABLE, cleanaway_deposition_velocity,
    deposition_envelope_deviation, deposition_velocity_pair,
    deposition_velocity_scaling_exponent, dimensional_audit,
    durand_deposition_velocity, durand_fl, set_durand_fl_table)
from tensordem.slurry_transport.pressure_drop import (
    colebrook_friction, darcy_head_loss, einstein_viscosity,
    excess_gradient_suspension, fit_durand_form, friction_factor,
    friction_factor_laminar, haaland_friction, mixture_gradient,
    moody_reference_points, pressure_drop, swamee_jain_friction, water_gradient)
from tensordem.slurry_transport.pump_pipe import (
    PipeSpec, PumpSpec, SlurrySpec, npsh_available, npsh_margin, npsh_required,
    operating_point, production_rate, pump_efficiency, pump_head, shaft_power,
    system_head)
from tensordem.slurry_transport.verification import (DESIGN_CASE, face_a2,
                                                     face_a3, face_a4, face_a5,
                                                     face_a7)


# --------------------------------------------------------------------- A2
def test_colebrook_solves_its_own_equation():
    for Re, rr in ((5e3, 1e-4), (1e5, 1e-3), (3e6, 5e-4)):
        out = colebrook_friction(Re, rr)
        assert out["residual"] < 1e-12
        f, x = out["f"], out["x"]
        assert x == pytest.approx(1.0 / math.sqrt(f))
        # the Colebrook equation itself
        assert x == pytest.approx(-2.0 * math.log10(rr / 3.7 + 2.51 * x / Re),
                                  abs=1e-12)


def test_colebrook_agrees_with_explicit_approximations():
    rows = moody_reference_points()
    for r in rows:
        assert r["err_swamee_jain"] < 0.01
        assert r["err_haaland"] < 0.01
    a2 = face_a2()
    assert a2["pass_moody"] and a2["pass_laminar"]


def test_laminar_limit_is_exact():
    for Re in (10.0, 500.0, 1999.0):
        assert friction_factor_laminar(Re) == pytest.approx(64.0 / Re, rel=1e-15)
        assert friction_factor(Re, 1e-3) == pytest.approx(64.0 / Re, rel=1e-15)


def test_friction_factor_monotone_in_roughness_and_Re():
    # smooth-wall limit decreases with Re, roughness increases f at fixed Re
    fs = colebrook_friction(1e5, 1e-6)["f"]
    fr = colebrook_friction(1e5, 1e-2)["f"]
    assert fr > fs
    assert colebrook_friction(1e4, 1e-4)["f"] > colebrook_friction(1e6, 1e-4)["f"]


def test_darcy_head_loss_and_water_gradient_consistent():
    V, D, L = 3.0, 0.4, 1000.0
    w = water_gradient(V, D)
    f = w["f"]
    h = darcy_head_loss(f, L, D, V)
    assert h == pytest.approx(w["i_w"] * L, rel=1e-12)
    assert w["Re"] == pytest.approx(998.2 * V * D / 1.002e-3)


# --------------------------------------------------------------------- A3
def test_durand_scaling_and_dimensional_audit():
    a3 = face_a3()
    assert a3["pass_dimensional"]
    assert abs(a3["slope_durand"]["slope"] - 0.5) <= 0.05
    assert abs(a3["slope_cleanaway"]["slope"] - 0.5) <= 0.05
    assert a3["pass_sqrt_D"]
    # exact homogeneity: quadrupling D doubles V_c
    v1 = float(durand_deposition_velocity(0.1, 2.65, 0.1, 0.5e-3))
    v4 = float(durand_deposition_velocity(0.4, 2.65, 0.1, 0.5e-3))
    assert v4 == pytest.approx(2.0 * v1, rel=1e-12)


def test_positivity_and_limits_of_deposition_velocity():
    # Ss = 1: no settling, no deposition velocity
    with pytest.raises(ValueError):
        durand_deposition_velocity(0.3, 1.0, 0.1, 0.5e-3)
    # denser solids deposit faster at fixed everything else
    v_light = float(durand_deposition_velocity(0.3, 1.5, 0.1, 0.5e-3))
    v_heavy = float(durand_deposition_velocity(0.3, 3.0, 0.1, 0.5e-3))
    assert v_heavy > v_light


def test_mutual_adjudication_within_20_percent():
    a3 = face_a3()
    assert a3["envelope_deviation"]["max_relative_deviation"] <= 0.20
    assert a3["pass_adjudication"]
    p = deposition_velocity_pair(0.4, 2.585, 0.15, 1e-3)
    assert p["relative_deviation"] < 0.20


def test_durand_table_is_overridable():
    base = float(durand_fl(0.5e-3, 0.1))
    try:
        set_durand_fl_table(fl=[2.0 * v for v in DURAND_FL_TABLE["fl"]],
                            provenance="test override")
        assert float(durand_fl(0.5e-3, 0.1)) == pytest.approx(2.0 * base, rel=1e-12)
    finally:
        set_durand_fl_table(d50_mm=DURAND_FL_TABLE["d50_mm"],
                            fl=DURAND_FL_TABLE["fl"],
                            cv_ref=DURAND_FL_TABLE["cv_ref"],
                            cv_exponent=DURAND_FL_TABLE["cv_exponent"],
                            provenance=DURAND_FL_TABLE["provenance"])
    assert float(durand_fl(0.5e-3, 0.1)) == pytest.approx(base, rel=1e-12)


def test_cleanaway_table_shipped_with_provenance():
    assert CLEANAWAY_TABLE["provenance"]
    assert DURAND_FL_TABLE["provenance"]
    assert len(CLEANAWAY_TABLE["d50_mm"]) == len(CLEANAWAY_TABLE["kc"])


# --------------------------------------------------------------------- A4
def test_pressure_drop_linear_in_concentration():
    a4 = face_a4()
    assert a4["R2_dense"] >= 0.98 and a4["R2_tabulated"] >= 0.98
    assert a4["pass"]
    # the slope must be positive and the intercept the water gradient
    assert a4["slope_Pa_per_Cv"] > 0.0
    assert a4["intercept_Pa"] > 0.0


def test_excess_gradient_derivation_and_kappa():
    V, cv, Ss, ws = 3.0, 0.1, 2.585, 0.2
    ie = excess_gradient_suspension(V, cv, Ss, ws)
    assert ie == pytest.approx(cv * (Ss - 1.0) * ws / V)
    assert excess_gradient_suspension(V, cv, Ss, ws, kappa=2.0) == pytest.approx(2 * ie)


def test_mixture_gradient_matches_parts():
    mg = mixture_gradient(4.0, 0.4, 0.15, 2.585, 0.2)
    assert mg["i_m"] == pytest.approx(mg["i_w"] + mg["i_e"], rel=1e-15)
    assert mg["mu_m"] == pytest.approx(einstein_viscosity(0.15))
    # Einstein viscosity increases the carrier friction
    assert mg["i_w"] > water_gradient(4.0, 0.4)["i_w"]


def test_durand_form_fit_quality():
    fit = fit_durand_form(V_list=[1.5, 2.0, 2.6, 3.5, 5.0], D=0.1,
                          cv_list=[0.05, 0.1, 0.2, 0.3], Ss=2.65, d50=4e-3)
    assert fit["R2"] > 0.99
    # the derived closure reduces to Phi ~ V^-3 at fixed D and d50
    assert 2.8 < fit["m"] < 3.2
    assert fit["K"] > 0.0


# --------------------------------------------------------------------- A5
def _objects(case=None):
    c = case or DESIGN_CASE
    pipe = PipeSpec(c["D"], c["L"], eps_D=c["eps_D"], zeta_minor=c["zeta_minor"],
                    H_static=c["H_static"], L_suction=c["L_suction"],
                    h_suction=c["h_suction"])
    pump = PumpSpec(**c["pump"])
    slurry = SlurrySpec(c["Ss"], c["cv"], c["d50"], rho_f=c["rho_f"],
                        mu=c["mu"], n_void=c["n_void"])
    return pipe, pump, slurry


def test_operating_point_closes_the_head_balance():
    pipe, pump, slurry = _objects()
    op = operating_point(pipe, pump, slurry)
    assert op["residual"] < 1e-6
    assert op["Q"] > 0.0 and op["H"] > 0.0
    assert pump_head(op["Q"], pump) == pytest.approx(op["H"], abs=1e-6)
    assert system_head(op["Q"], pipe, slurry) == pytest.approx(op["H"], abs=1e-6)
    # two intersections exist (U-shaped demand); the stable one is selected
    op_low = operating_point(pipe, pump, slurry, branch="first")
    assert op_low["Q"] < op["Q"]
    assert op["n_crossings"] >= 2


def test_power_identity():
    pipe, pump, slurry = _objects()
    op = operating_point(pipe, pump, slurry)
    assert op["P_shaft"] * op["eta_p"] == pytest.approx(op["P_hydraulic"], rel=1e-3)
    assert shaft_power(1.0, 50.0, 1300.0, 0.8) == pytest.approx(
        1300.0 * 9.80665 * 1.0 * 50.0 / 0.8)


def test_npsh_margin_and_velocity_above_deposition():
    pipe, pump, slurry = _objects()
    op = operating_point(pipe, pump, slurry)
    assert op["npsh_margin"] >= 0.5
    assert op["npsh_margin"] == pytest.approx(op["npsh_a"] - op["npsh_r"], rel=1e-12)
    # the design point must not silt: V > V_c
    assert op["V_over_Vc"] > 1.0
    a5 = face_a5()
    assert a5["pass_residual"] and a5["pass_power"] and a5["pass_npsh"]


def test_pump_speed_affinity():
    pipe, pump, slurry = _objects()
    lo = operating_point(pipe, pump, slurry, n=0.9)
    hi = operating_point(pipe, pump, slurry, n=1.0)
    assert lo["Q"] < hi["Q"] and lo["H"] < hi["H"]
    # affinity: shut-off head scales with n^2
    assert pump_head(0.0, pump, n=0.9) == pytest.approx(0.81 * pump_head(0.0, pump))


def test_production_rate_conversion():
    pipe, pump, slurry = _objects()
    op = operating_point(pipe, pump, slurry)
    pr = op["production"]
    Q = op["Q"]
    assert pr["Q_solid_m3h"] == pytest.approx(3600.0 * Q * slurry.cv)
    assert pr["Q_bank_m3h"] == pytest.approx(pr["Q_solid_m3h"] / (1 - slurry.n_void))
    assert pr["solids_tph"] == pytest.approx(pr["Q_solid_m3h"] * slurry.rho_s / 1000.0)


def test_efficiency_parabola_peaks_at_the_design_flow():
    pump = PumpSpec(H0=60, a=1, b=20, eta_p=0.7, Q_opt=0.5, eta_max=0.8)
    assert pump_efficiency(0.5, pump) == pytest.approx(0.8)
    assert pump_efficiency(0.1, pump) < 0.8
    assert pump_efficiency(0.9, pump) < 0.8


# --------------------------------------------------------------------- A7
def test_negative_control_neutral_density():
    out = face_a7()
    assert out["w_s"] == 0.0
    assert out["relative_excess"] < 0.01
    assert out["pass_analytic"]
    # and with a finite (non-zero) particle density the excess is non-zero
    mg = mixture_gradient(2.5, 0.1, 0.1, 2.65, 0.44)
    assert mg["i_e"] > 0.0