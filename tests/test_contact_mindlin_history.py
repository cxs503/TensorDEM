"""Machine checks for the Mindlin tangential history model.

This is the file ``src/tensordem/contact_tangential.py`` has always pointed at
("Properties that are machine-checked (tests/test_contact_mindlin_history.py)")
but which did not exist -- the reference was dangling until m16810.

Two blocks:

* **documented properties** of :func:`tensordem.contact_tangential.mindlin_tangential_step`
  (linear backbone reproduces Cundall--Strack, virgin loading follows the
  closed-form backbone, elastic unloading, dashpot, inactive contacts);
* **the m16810 audit checks**: the engine, driven through a steady stick-slip
  cycle, is compared pointwise with the exact Mindlin--Deresiewicz reversal
  branch and its dissipated energy is compared with the exact closed-loop area
  ``1.2 (mu F_n)^2 / k_t``.  The default linear unloading branch is measured
  against the same references so the (large) deviation it carries is pinned by
  a test, not just by a document.

References: Mindlin & Deresiewicz (1953) J. Appl. Mech. 20:327,
doi:10.1115/1.4010702; Johnson, *Contact Mechanics* (1985) Sec. 4.3-4.4.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.contact_tangential import (
    MindlinTangentialConfig,
    mindlin_backbone,
    mindlin_tangential_step,
)
from tensordem.mindlin_partial_slip import (
    gross_slip_displacement,
    partial_slip_magnitude,
    reversal_loop_area,
    steady_state_cycle_area,
)

FN = 10.0
MU = 0.3
E = 9.0e9
NU = 0.33
R = 0.05
_ESTAR = E / (1.0 - NU * NU)          # sphere on a rigid plane
_GSTAR = (E / (2.0 * (1.0 + NU))) / (2.0 - NU)
_A = (3.0 * FN * R / (4.0 * _ESTAR)) ** (1.0 / 3.0)
KT0 = 8.0 * _GSTAR * _A
DS = float(gross_slip_displacement(FN, MU, KT0))
CAP = MU * FN


def _state(n=1):
    z = torch.zeros(n, 1, dtype=torch.float64)
    fn = torch.full((n,), FN, dtype=torch.float64)
    return z.clone(), z.clone(), z.clone(), z.clone(), fn


def _step(state, v, cfg, dt=1.0):
    F, delta, F_r, d_r, fn = state
    v_t = torch.full((1, 1), float(v), dtype=torch.float64)
    out = mindlin_tangential_step(F, delta, F_r, d_r, v_t, fn, cfg, dt)
    return out


def _cycle(unloading: str, n_steps: int = 400):
    """Virgin load to delta_s, reverse by 2 delta_s, reload by 2 delta_s."""
    cfg = MindlinTangentialConfig(k_t0=KT0, mu_s=MU, backbone="mindlin",
                                  gamma_t=0.0, unloading=unloading)
    F, delta, F_r, d_r, fn = _state()
    trace = []
    legs = (("load", DS / n_steps), ("reverse", -2.0 * DS / n_steps),
            ("reload", 2.0 * DS / n_steps))
    for name, v in legs:
        s = 0.0
        for _ in range(n_steps):
            out, F, delta, F_r, d_r, sliding = _step((F, delta, F_r, d_r, fn), v, cfg)
            s += abs(v)
            trace.append(dict(segment=name, delta=float(delta[0, 0]),
                              force=float(out[0, 0]), excursion=s,
                              sliding=bool(sliding[0])))
    return trace


def _branch(s: float) -> float:
    x = min(max(s / (2.0 * DS), 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 1.5


def _exact_force(p) -> float:
    if p["segment"] == "load":
        return -float(partial_slip_magnitude(
            torch.tensor(abs(p["delta"])), FN, MU, KT0))
    if p["segment"] == "reverse":       # starts at -cap, ends at +cap
        return -CAP + 2.0 * CAP * _branch(p["excursion"])
    return CAP - 2.0 * CAP * _branch(p["excursion"])   # reload: +cap -> -cap


def _loop_area(trace) -> float:
    last_load = [p for p in trace if p["segment"] == "load"][-1]
    pts = [(last_load["delta"], last_load["force"])]
    pts += [(p["delta"], p["force"]) for p in trace if p["segment"] != "load"]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return -area


# ---------------------------------------------------------------------------
# 1. documented properties of the history model
# ---------------------------------------------------------------------------

def test_virgin_loading_follows_closed_form_backbone():
    """Monotonic loading from a virgin contact reproduces the M&D backbone."""
    tr = _cycle("linear")
    err = max(abs(p["force"] - _exact_force(p)) / CAP
              for p in tr if p["segment"] == "load")
    assert err < 1e-6, f"virgin backbone deviation {err:.3e} of mu F_n"


def test_linear_backbone_equals_cundall_strack_spring():
    """``backbone="linear"`` is a plain Cundall--Strack spring up to the cap."""
    cfg = MindlinTangentialConfig(k_t0=KT0, mu_s=MU, backbone="linear",
                                  gamma_t=0.0, unloading="linear")
    F, delta, F_r, d_r, fn = _state()
    v = 4e-8
    for k in range(1, 6):
        out, F, delta, F_r, d_r, _ = _step((F, delta, F_r, d_r, fn), v, cfg)
        assert float(out[0, 0]) == pytest.approx(-KT0 * v * k, rel=1e-12)


def test_inactive_contacts_forget_history():
    cfg = MindlinTangentialConfig(k_t0=KT0, mu_s=MU, gamma_t=0.0, unloading="mindlin")
    F, delta, F_r, d_r, _ = _state()
    F[0, 0] = -CAP
    delta[0, 0] = 2.0 * DS
    F_r[0, 0] = -CAP
    d_r[0, 0] = 2.0 * DS
    fn = torch.zeros(1, dtype=torch.float64)
    out, F2, delta2, F_r2, d_r2, sliding = mindlin_tangential_step(
        F, delta, F_r, d_r, torch.zeros(1, 1, dtype=torch.float64), fn, cfg, 1.0)
    assert float(out.abs().max()) == 0.0
    assert float(F_r2.abs().max()) == 0.0 and float(d_r2.abs().max()) == 0.0
    assert not bool(sliding.any())


def test_dashpot_subtracts_gamma_v():
    cfg = MindlinTangentialConfig(k_t0=KT0, mu_s=MU, gamma_t=30.0,
                                  unloading="linear")
    F, delta, F_r, d_r, fn = _state()
    v = 4e-8
    out, F_el, *_ = mindlin_tangential_step(
        F, delta, F_r, d_r, torch.tensor([[v]], dtype=torch.float64), fn, cfg, 1.0)
    assert float(out[0, 0]) == pytest.approx(float(F_el[0, 0]) - 30.0 * v, rel=1e-12)


def test_config_validation():
    with pytest.raises(ValueError):
        MindlinTangentialConfig(k_t0=KT0, mu_s=MU, unloading="hooke")
    with pytest.raises(ValueError):
        MindlinTangentialConfig(k_t0=KT0, mu_s=MU, unloading="mindlin",
                                unloading_ratio=1.5)


def test_backbone_envelope_endpoints():
    env0 = mindlin_backbone(torch.tensor([0.0], dtype=torch.float64),
                            torch.tensor([FN], dtype=torch.float64),
                            MindlinTangentialConfig(k_t0=KT0, mu_s=MU))
    env1 = mindlin_backbone(torch.tensor([DS], dtype=torch.float64),
                            torch.tensor([FN], dtype=torch.float64),
                            MindlinTangentialConfig(k_t0=KT0, mu_s=MU))
    assert float(env0[0]) == pytest.approx(0.0, abs=1e-15)
    assert float(env1[0]) == pytest.approx(CAP, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. m16810 audit checks: engine vs exact Mindlin--Deresiewicz
# ---------------------------------------------------------------------------

def test_mindlin_unloading_matches_exact_reversal_branch():
    """Opt-in M&D branch: pointwise error < 1e-3 of ``mu F_n`` over the cycle."""
    tr = _cycle("mindlin")
    err = max(abs(p["force"] - _exact_force(p)) / CAP for p in tr)
    assert err < 1e-3, f"engine vs exact M&D branch: {err:.3e} of mu F_n"


def test_linear_unloading_deviation_is_pinned():
    """The legacy linear branch deviates strongly from M&D (pinned, not hidden)."""
    tr = _cycle("linear")
    err = max(abs(p["force"] - _exact_force(p)) / CAP for p in tr)
    assert 0.2 < err < 0.4, f"unexpected legacy deviation {err:.3e}"


def test_steady_state_cycle_area_engine_vs_closed_form():
    """Engine dissipates the exact closed-loop area within 1 %."""
    exact = float(steady_state_cycle_area(FN, MU, KT0, n_quad=20001))
    for unloading, tol in (("mindlin", 1e-2), ("linear", 0.5)):
        area = _loop_area(_cycle(unloading))
        assert area == pytest.approx(exact, rel=tol), (
            f"unloading={unloading}: engine {area:.6e} vs exact {exact:.6e}")


def test_steady_state_area_closed_form_and_legacy_factor():
    """``W = 1.2 (mu F_n)^2 / k_t``; the pre-m16810 value was 5x that."""
    for mu, fn in ((0.3, 10.0), (0.5, 2.0), (0.1, 0.5)):
        k_t = float(8.0 * _GSTAR * (3.0 * fn * R / (4.0 * _ESTAR)) ** (1.0 / 3.0))
        area = float(steady_state_cycle_area(fn, mu, k_t, n_quad=200001))
        closed = 1.2 * (mu * fn) ** 2 / k_t
        assert area == pytest.approx(closed, rel=1e-6)
        d_s = float(gross_slip_displacement(fn, mu, k_t))
        assert area == pytest.approx(0.8 * mu * fn * d_s, rel=1e-6)
        legacy = float(reversal_loop_area(fn, mu, k_t, n_quad=200001))
        assert legacy == pytest.approx(area, rel=1e-6)          # now delegated
        assert legacy * 5.0 == pytest.approx(
            6.0 * (mu * fn) ** 2 / k_t, rel=1e-6)   # the old (wrong) formula


def test_mindlin_branch_is_energy_positive():
    """The opt-in cycle absorbs work (thermodynamic consistency)."""
    assert _loop_area(_cycle("mindlin")) > 0.0