"""XVAL: Jiang-2005 quadratic rolling-stiffness rule against closed forms.

Anchor ``XVAL-DEM-ELASTIC-PLASTIC-ROLLING-SPRING-KR-225-MUR2-KN-R2``: the
library must expose the Jiang, Yu & Harris (2005) rule

    k_r = 2.25 * mu_r**2 * k_n * R**2

(and not silently substitute the linear ``2.25 * mu_r * k_n * R**2`` that the
older in-repo note attributed to the same paper).  These tests pin:

* the rule itself (coefficient identity, ``mu_r``-ratio to the linear rule,
  quadratic scaling, input validation);
* the *consequences* of the rule, each against an independent closed form:
  - elastic rolling-mode frequency ``sqrt(k_r / (m R^2 (1 + I/mR^2)))`` and its
    ``sqrt(mu_r)`` ratio between the two rules, with a dt-refinement check that
    the residual is discretisation error and not an offset;
  - plastic branch: yield time ``t_y = mu_r|F_n|r_eff/(k_r omega)``, the clamp
    ``|M_r| <= mu_r|F_n|r_eff``, and the dissipated-work identity
    ``W = mu_r|F_n|r_eff (theta - theta_y)``;
  - single-sphere incline harness: steady downhill acceleration vs
    ``g (sin t - mu_r cos t) / (1 + I/mR^2)`` under the Jiang rule;
  - static branch: no creep below the critical angle.

Reference: Jiang, M.J., Yu, H.-S., Harris, D. (2005). A novel discrete model for
granular material incorporating rolling resistance. Comput. Geotech. 32, 340-357.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.rolling_resistance import (  # noqa: E402
    JIANG2005_ANCHOR_STATE,
    RollingResistanceConfig,
    SphereInclineConfig,
    critical_incline_angle_deg,
    fit_acceleration,
    incline_rolling_acceleration,
    pure_rolling_feasible,
    rolling_mode_frequency,
    rolling_resistance_torque,
    rolling_stiffness,
    rolling_stiffness_table,
    rolling_yield_angle,
    rolling_yield_time,
    simulate_sphere_on_incline,
)

KN_ANCHOR = 1.0e6
R_ANCHOR = 0.05
DENSITY = 900.0


def _mass(radius: float = R_ANCHOR, density: float = DENSITY) -> float:
    return 4.0 / 3.0 * math.pi * radius ** 3 * density


def _oscillator(mu_r: float, rule: str, *, steps_per_period: float = 500.0,
                periods: float = 4.0) -> dict:
    """Integrate ``I_eff theta_ddot = M_r`` with the library EPSD torque.

    ``I_eff = m R^2 (1 + I/(mR^2))`` is the no-slip rolling inertia, so the
    analytic reference is exactly :func:`rolling_mode_frequency`.  ``dt`` is
    deliberately *not* commensurate with the analytic period, and the crossings
    are linearly interpolated, so the measured frequency cannot coincide with
    the analytic one by step quantisation.
    """
    radius = R_ANCHOR
    mass = _mass(radius)
    inertia_factor = 0.4
    i_eff = mass * radius ** 2 * (1.0 + inertia_factor)
    k_r = rolling_stiffness(KN_ANCHOR, mu_r, radius, rule=rule)
    f_n = mass * 9.81
    limit = mu_r * f_n * radius
    theta_y = limit / k_r
    theta0 = 0.5 * theta_y  # stay inside the elastic branch

    f_ana = rolling_mode_frequency(k_r, mass, radius, inertia_factor=inertia_factor)
    t_ana = 2.0 * math.pi / f_ana
    dt = t_ana / steps_per_period
    steps = int(round(periods * t_ana / dt))

    cfg = RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r)
    state = torch.zeros(1, 3, dtype=torch.float64)
    state[0, 2] = -k_r * theta0      # stress-free state is defined at theta = 0
    omega = torch.zeros(1, 3, dtype=torch.float64)
    theta = torch.tensor([theta0], dtype=torch.float64)
    f_n_t = torch.tensor([f_n], dtype=torch.float64)
    r_t = torch.tensor([radius], dtype=torch.float64)

    hist = torch.empty(steps, dtype=torch.float64)
    for i in range(steps):
        torque, state = rolling_resistance_torque(
            omega, f_n_t, r_t, cfg, dt=dt, torque_state=state
        )
        omega[:, 2] = omega[:, 2] + (torque[0, 2] / i_eff) * dt
        theta = theta + omega[:, 2] * dt
        hist[i] = theta[0]

    crossings = []
    for i in range(1, steps):
        y0, y1 = float(hist[i - 1]), float(hist[i])
        if y0 <= 0.0 < y1:
            crossings.append((i - 1 + (-y0) / (y1 - y0)) * dt)
    assert len(crossings) >= 3, "not enough zero crossings to measure the period"
    seg = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
    period = sum(seg) / len(seg)
    measured = 2.0 * math.pi / period
    return {
        "k_r": k_r, "theta_y": theta_y, "limit": limit, "dt": dt,
        "freq_measured": measured, "freq_analytic": f_ana,
        "rel_error": abs(measured - f_ana) / f_ana,
        "max_torque_over_limit": float(hist.abs().max()) / k_r / theta_y * theta0 / theta0,
    }


# ---------------------------------------------------------------------------
# A. The rule itself
# ---------------------------------------------------------------------------

def test_anchor_rule_is_quadratic_and_reproduces_225():
    """k_r = 2.25 mu_r^2 k_n R^2; the anchor state gives k_r = 225 N*m/rad."""
    state = JIANG2005_ANCHOR_STATE
    k_r = rolling_stiffness(state["k_n"], state["mu_r"], state["radius"])
    assert k_r == pytest.approx(2.25 * state["mu_r"] ** 2 * state["k_n"] * state["radius"] ** 2,
                                rel=1e-15)
    assert k_r == pytest.approx(225.0, rel=1e-12), (
        "anchor slug KR-225 / MUR2-KN-R2 must reproduce k_r = 225 N*m/rad at the "
        "documented reference state"
    )


@pytest.mark.parametrize("mu_r", [0.05, 0.1, 0.2, 0.3, 0.5, 0.9])
def test_jiang_rule_is_mu_r_times_the_linear_rule(mu_r):
    """The two conventions differ by exactly a factor mu_r (not mu_r^-1)."""
    jia = rolling_stiffness(KN_ANCHOR, mu_r, R_ANCHOR, rule="jiang2005")
    lin = rolling_stiffness(KN_ANCHOR, mu_r, R_ANCHOR, rule="linear")
    assert jia / lin == pytest.approx(mu_r, rel=1e-14)


def test_table_ratio_column_matches_mu_r():
    rows = rolling_stiffness_table([0.05, 0.2, 0.5], k_n=2.0e4, radius=0.01)
    for r in rows:
        assert r["ratio_jiang_over_linear"] == pytest.approx(r["mu_r"], rel=1e-14)


def test_quadratic_scaling_in_mu_r_and_geometry():
    mu = 0.25
    assert (rolling_stiffness(KN_ANCHOR, 2 * mu, R_ANCHOR)
            / rolling_stiffness(KN_ANCHOR, mu, R_ANCHOR)) == pytest.approx(4.0, rel=1e-14)
    assert (rolling_stiffness(2 * KN_ANCHOR, mu, 2 * R_ANCHOR)
            / rolling_stiffness(KN_ANCHOR, mu, R_ANCHOR)) == pytest.approx(8.0, rel=1e-14)
    assert rolling_stiffness(KN_ANCHOR, 0.0, R_ANCHOR) == 0.0


@pytest.mark.parametrize("kwargs", [
    {"rule": "quadratic"},
    {"mu_r": -0.1},
    {"k_n": 0.0},
    {"radius": -1.0},
    {"coefficient": 0.0},
])
def test_rule_rejects_invalid_parameters(kwargs):
    args = {"k_n": KN_ANCHOR, "mu_r": 0.2, "radius": R_ANCHOR}
    args.update(kwargs)
    with pytest.raises(ValueError):
        rolling_stiffness(**args)


def test_reference_functions_reject_invalid_inputs():
    with pytest.raises(ValueError):
        rolling_mode_frequency(0.0, 1.0, 0.05)
    with pytest.raises(ValueError):
        rolling_yield_angle(0.2, 1.0, 0.05, 0.0)
    with pytest.raises(ValueError):
        rolling_yield_time(0.2, 1.0, 0.05, 10.0, 0.0)


# ---------------------------------------------------------------------------
# B. Elastic rolling mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ["linear", "jiang2005"])
@pytest.mark.parametrize("mu_r", [0.1, 0.2, 0.3, 0.5])
def test_mode_frequency_matches_closed_form(rule, mu_r):
    """Discrete EPSD torque reproduces sqrt(k_r/(m R^2 (1+I/mR^2))) to < 1e-4."""
    res = _oscillator(mu_r, rule, steps_per_period=500.0)
    assert res["rel_error"] < 1.0e-4, res


def test_frequency_ratio_between_rules_is_sqrt_mu():
    for mu_r in (0.2, 0.5):
        lin = _oscillator(mu_r, "linear", steps_per_period=500.0)
        jia = _oscillator(mu_r, "jiang2005", steps_per_period=500.0)
        ratio = jia["freq_measured"] / lin["freq_measured"]
        assert ratio == pytest.approx(math.sqrt(mu_r), rel=1e-6), (
            f"mu_r={mu_r}: measured ratio {ratio} != sqrt(mu_r) {math.sqrt(mu_r)}"
        )


def test_mode_frequency_residual_is_discretisation_error():
    """Halving dt must cut the residual by ~4 (second-order symplectic Euler).

    Guards against the residual being a hidden model offset: an offset would not
    shrink with dt.
    """
    coarse = _oscillator(0.2, "jiang2005", steps_per_period=500.0, periods=3.0)
    fine = _oscillator(0.2, "jiang2005", steps_per_period=1000.0, periods=3.0)
    assert coarse["dt"] / fine["dt"] == pytest.approx(2.0, rel=1e-9)
    ratio = coarse["rel_error"] / fine["rel_error"]
    assert 3.5 < ratio < 4.5, (coarse["rel_error"], fine["rel_error"], ratio)


# ---------------------------------------------------------------------------
# C. Plastic branch
# ---------------------------------------------------------------------------

def _plastic_ramp(mu_r: float, rule: str, *, omega_roll: float = 1.0) -> dict:
    """Constant-spin ramp: |M_r| grows at k_r*omega until the plastic limit."""
    radius = R_ANCHOR
    mass = _mass(radius)
    f_n = mass * 9.81
    k_r = rolling_stiffness(KN_ANCHOR, mu_r, radius, rule=rule)
    limit = mu_r * f_n * radius
    theta_y = rolling_yield_angle(mu_r, f_n, radius, k_r)
    t_y = rolling_yield_time(mu_r, f_n, radius, k_r, omega_roll)
    period = 2.0 * math.pi / rolling_mode_frequency(k_r, mass, radius)
    # dt tied to the elastic period only -> the measured saturation step cannot
    # coincide with the analytic yield time by construction
    dt = min(period / 1997.3, t_y / 397.3)
    steps = int(round(4.0 * t_y / dt))

    cfg = RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r)
    omega = torch.tensor([[0.0, 0.0, -omega_roll]], dtype=torch.float64)
    f_n_t = torch.tensor([f_n], dtype=torch.float64)
    r_t = torch.tensor([radius], dtype=torch.float64)
    state = torch.zeros(1, 3, dtype=torch.float64)

    mag_max = 0.0
    prev = 0.0
    t_sat = None
    plastic_work = 0.0
    theta = 0.0
    for i in range(steps):
        torque, state = rolling_resistance_torque(
            omega, f_n_t, r_t, cfg, dt=dt, torque_state=state
        )
        mag = float(torque.abs().max())
        mag_max = max(mag_max, mag)
        if t_sat is None and mag >= limit * (1.0 - 1e-12):
            frac = (limit - prev) / (mag - prev) if mag > prev else 0.0
            t_sat = (i + frac) * dt
        else:
            plastic_work += limit * max(abs(omega_roll) * dt - (mag - prev) / k_r, 0.0)
        prev = mag
        theta += abs(omega_roll) * dt
    return {
        "k_r": k_r, "limit": limit, "theta_y": theta_y, "t_y": t_y, "t_sat": t_sat,
        "yield_rel_error": abs(t_sat - t_y) / t_y,
        "max_over_limit": mag_max / limit,
        "work_measured": plastic_work,
        "work_analytic": limit * max(theta - theta_y, 0.0),
    }


@pytest.mark.parametrize("rule", ["linear", "jiang2005"])
def test_plastic_yield_time_and_clamp(rule):
    res = _plastic_ramp(0.2, rule)
    assert res["max_over_limit"] <= 1.0 + 1e-12, "EPSD clamp must never overshoot"
    assert res["t_sat"] is not None and res["t_sat"] > 0.0
    assert res["yield_rel_error"] < 5e-3, res


@pytest.mark.parametrize("rule", ["linear", "jiang2005"])
def test_plastic_dissipation_identity(rule):
    res = _plastic_ramp(0.2, rule)
    assert res["work_measured"] > 0.0
    rel = abs(res["work_measured"] - res["work_analytic"]) / res["work_analytic"]
    assert rel < 1e-2, (res["work_measured"], res["work_analytic"], rel)


def test_jiang_rule_yields_mu_r_times_later():
    """Softer spring => the yield angle grows by 1/mu_r at fixed mu_r."""
    lin = _plastic_ramp(0.3, "linear")
    jia = _plastic_ramp(0.3, "jiang2005")
    assert jia["theta_y"] / lin["theta_y"] == pytest.approx(1.0 / 0.3, rel=1e-12)
    assert jia["t_y"] / lin["t_y"] == pytest.approx(1.0 / 0.3, rel=1e-12)


# ---------------------------------------------------------------------------
# D. Incline harness under the Jiang rule
# ---------------------------------------------------------------------------

def test_incline_acceleration_under_jiang_rule():
    """Steady rolling acceleration is k_r-independent; must match to < 3 %."""
    mu_r, k_n, radius, dt = 0.2, 2.0e4, 0.01, 5.0e-5
    k_r = rolling_stiffness(k_n, mu_r, radius, rule="jiang2005")
    theta_c = critical_incline_angle_deg(mu_r)
    angles = [1.25 * theta_c, 2.0 * theta_c, 12.0]
    angles = [a for a in angles if pure_rolling_feasible(a, mu_r, 0.9)]
    res = simulate_sphere_on_incline(
        angles, RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r),
        config=SphereInclineConfig(radius=radius, k_n=k_n, dt=dt),
        duration=0.12, settle_time=0.04,
    )
    acc, _ = fit_acceleration(res)
    for j, ang in enumerate(angles):
        ref = incline_rolling_acceleration(ang, mu_r)
        err = abs(float(acc[j]) - ref) / abs(ref)
        assert err < 3e-2, f"angle {ang:.2f} deg: sim {float(acc[j]):.6f} vs ref {ref:.6f} ({err:.2%})"


def test_static_branch_holds_with_jiang_rule():
    """Below arctan(mu_r) the (softer) Jiang spring must still arrest the sphere."""
    mu_r, k_n, radius, dt = 0.2, 2.0e4, 0.01, 5.0e-5
    k_r = rolling_stiffness(k_n, mu_r, radius, rule="jiang2005")
    angle = 0.8 * critical_incline_angle_deg(mu_r)
    res = simulate_sphere_on_incline(
        [angle], RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r),
        config=SphereInclineConfig(radius=radius, k_n=k_n, dt=dt),
        duration=0.12, settle_time=0.04,
    )
    v_tail = res["v"][-1000:, 0]
    free_roll_accel = 9.81 * math.sin(math.radians(angle))
    assert float(v_tail.abs().mean()) < 1e-3, float(v_tail.abs().mean())
    # displacement must stay negligible against a free roll of the same duration
    assert abs(float(res["x"][-1, 0])) < 1e-3 * 0.5 * free_roll_accel * 0.12 ** 2 + 1e-4

# ---------------------------------------------------------------------------
# E. Integration caller: the anchor case must be *computed*, not hardcoded
# ---------------------------------------------------------------------------

def test_integration_caller_harness_path_resolves():
    """The caller must point at ``<repo>/scripts/validate_rolling_resistance.py``.

    It used ``parents[1]`` (= ``src/``), so the harness subprocess never ran and
    every case silently reported a default value while still claiming L4.
    """
    from tensordem import integration_callers as ic

    root = Path(ic.__file__).resolve().parents[2]
    assert (root / "scripts" / "validate_rolling_resistance.py").exists()
    assert (root / "src" / "tensordem" / "__init__.py").exists()


def test_integration_caller_kr_case_fails_when_rule_regresses(monkeypatch, tmp_path):
    """Sabotaging the rule to the *linear* formula must fail the anchor case.

    A hardcoded ``"passed": True`` (as shipped before this test) cannot detect
    such a regression, so this is the guard for the anchor itself.
    """
    import tensordem.rolling_resistance as rr
    from tensordem import integration_callers as ic

    def linear_rule(k_n, mu_r, radius, *, rule="jiang2005", coefficient=2.25):
        return coefficient * mu_r * k_n * radius ** 2  # the OLD/wrong exponent

    monkeypatch.setattr(rr, "rolling_stiffness", linear_rule)
    res = ic.run_rolling_resistance_xval(output_dir=str(tmp_path / "caller"))
    case = res["cases"]["kr_rule_225_mur2_kn_r2"]
    assert case["passed"] is False
    assert case["got"]["k_r_N_m_per_rad"] == pytest.approx(1125.0, rel=1e-12)
    assert case["relative_error"] == pytest.approx(4.0, rel=1e-12)
    assert res["overall"] == "FAIL"
