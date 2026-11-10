"""Transit runner: steady broken-ice resistance of one case, with an audit.

The ship is prescribed (kinematic), the floe field is discrete and every force
on a floe is either a hull/floe contact (crushing-limited law) or a water drag.
The ice resistance is the *reaction* the floes exert on the hull along the track:

    R_ice = - sum_i F_i->hull  . e_x      [N]

Averaged over the steady window (ship fully embedded in the field: stem abaft
``L`` and forward of the field end) the running mean is reported with its
standard deviation and standard error.

Audit
-----
For a steady transit the hull power must be dissipated by water drag on the
entrained floes and by hull-ice friction:

    R_ice * V = P_drag + P_friction

``friction_power_share = 1 - P_drag / (R_ice * V)`` therefore has to lie in
``[0, 1]``; a value outside that band means the measured reaction and the
dissipation are inconsistent (a bookkeeping error).  The share is reported, not
tuned -- both dissipations are computed from the *same* state.

All quantities SI; the returned resistance is in kN.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ..ice_channel import submerged_draught
from .contact import BrokenIceChannel
from .ledger import CaseSpec, ledger_sha256


@dataclass
class CaseResult:
    """Converged (or honest-failed) result of one transit case."""

    spec: dict[str, Any]
    hull: dict[str, Any]
    mean_kN: float
    std_kN: float
    sem_kN: float
    peak_kN: float
    min_kN: float
    contact_count_mean: float
    floe_speed_mean_m_s: float
    drag_power_kW: float
    hull_power_kW: float
    friction_power_share: float
    kinematic_energy_change_kW: float
    stationarity_drift: float
    floe_count: int
    realised_concentration: float
    steady_window_s: tuple[float, float]
    simulated_time_s: float
    steps: int
    wall_time_s: float
    series: list[dict[str, float]] = field(default_factory=list)
    ledger_sha256: str = ""
    state_digest: str = ""

    # -- derived ------------------------------------------------------------
    @property
    def mean_N(self) -> float:
        return 1e3 * self.mean_kN

    def as_dict(self, *, include_series: bool = True) -> dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items() if k != "series"}
        if include_series:
            payload["series"] = self.series
        return payload


def _drag_dissipation_W(sim: BrokenIceChannel) -> float:
    """Water-drag dissipation rate of the floe field [W] (always >= 0)."""
    speed = torch.linalg.norm(sim.vel, dim=1)
    draught = torch.clamp(submerged_draught(sim.z, sim.thickness), min=0.0)
    area = 2.0 * sim.radius * draught
    coefficient = 0.5 * sim.water.water_density * sim.water.water_drag_coefficient
    return float((coefficient * area * speed ** 3).sum())


def _kinetic_energy_J(sim: BrokenIceChannel) -> float:
    return float((0.5 * sim.mass * (sim.vel ** 2).sum(dim=1)).sum())


def run_case(spec: CaseSpec, *, dtype=torch.float64,
             max_steps: int = 200_000) -> CaseResult:
    """Run one direct-navigation transit and return the steady resistance."""
    torch.set_num_threads(1)
    started = time.time()
    sim = BrokenIceChannel.from_case(spec, dtype=dtype)
    hull = sim.hull

    # transit: from the initial stem abscissa until the stem reaches the field end
    x_start = hull.x_centre(0.0)
    x_stop = spec.domain_length_m - 5.0
    travel = max(x_stop - x_start, 10.0)
    steps = int(round(travel / spec.speed_m_s / spec.time_step_s))
    steps = min(steps, int(max_steps))

    records: list[dict[str, float]] = []
    ke0 = _kinetic_energy_J(sim)
    for step in range(steps):
        sim.step()
        if step % int(spec.record_every) == 0:
            ledger = sim.hull_contact(sim.time)[0]
            reaction = ledger["force_on_hull_N"]
            drag_power = _drag_dissipation_W(sim)
            records.append({
                "time_s": float(sim.time),
                "resistance_kN": -float(reaction[0]) / 1e3,
                "contact_count": float(ledger["contact_count"]),
                "floe_speed_mean_m_s": float(
                    torch.linalg.norm(sim.vel, dim=1).mean()),
                "drag_power_kW": drag_power / 1e3,
                "kinetic_energy_kJ": _kinetic_energy_J(sim) / 1e3,
            })

    ke1 = _kinetic_energy_J(sim)
    total_time = float(sim.time)

    # steady window: ship fully embedded in the field
    stem = hull.x_centre(0.0) + spec.speed_m_s * torch.tensor(
        [r["time_s"] for r in records], dtype=torch.float64)
    window = (stem >= hull.length) & (stem <= spec.domain_length_m)
    if bool(window.any()):
        window &= torch.arange(len(records)) >= int(spec.steady_fraction * len(records))
    if not bool(window.any()):  # short transit: fall back to the second half
        window = torch.arange(len(records)) >= int(0.5 * len(records))

    force = torch.tensor([r["resistance_kN"] for r in records], dtype=torch.float64)[window]
    if force.numel() == 0:  # pragma: no cover - defensive
        raise RuntimeError("empty steady window")
    half = max(force.numel() // 2, 1)
    first, second = force[:half], force[half:]
    length_s = float(records[-1]["time_s"] - records[0]["time_s"]) or 1.0

    drag_power = float(sum(r["drag_power_kW"] for r in records) / len(records))
    # time-average over the window
    idx = torch.nonzero(window).squeeze(1).tolist()
    drag_window = sum(records[i]["drag_power_kW"] for i in idx) / len(idx)
    hull_power = float(force.mean()) * spec.speed_m_s
    friction_share = 1.0 - drag_window / hull_power if hull_power > 0 else float("nan")
    ke_rate = (ke1 - ke0) / total_time / 1e3  # kW (mean over the whole run)

    return CaseResult(
        spec=spec.as_dict(),
        hull=hull.as_dict(),
        mean_kN=float(force.mean()),
        std_kN=float(force.std(unbiased=True)) if force.numel() > 1 else 0.0,
        sem_kN=(float(force.std(unbiased=True)) / math.sqrt(force.numel())
                if force.numel() > 1 else 0.0),
        peak_kN=float(force.max()),
        min_kN=float(force.min()),
        contact_count_mean=sum(records[i]["contact_count"] for i in idx) / len(idx),
        floe_speed_mean_m_s=sum(records[i]["floe_speed_mean_m_s"] for i in idx) / len(idx),
        drag_power_kW=drag_window,
        hull_power_kW=hull_power,
        friction_power_share=friction_share,
        kinematic_energy_change_kW=ke_rate,
        stationarity_drift=float(second.mean() - first.mean()) / float(force.mean())
        if float(force.mean()) != 0.0 else float("nan"),
        floe_count=int(sim.n),
        realised_concentration=float(sim.field_statistics()["realised_concentration"]),
        steady_window_s=(float(records[idx[0]]["time_s"]),
                         float(records[idx[-1]]["time_s"])),
        simulated_time_s=total_time,
        steps=steps,
        wall_time_s=time.time() - started,
        series=records,
        ledger_sha256=ledger_sha256(),
        state_digest=(sim.state_digest() if hasattr(sim, "state_digest") else ""),
    )