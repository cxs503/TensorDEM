"""Independent verification anchors for the DEM-FEM one-way coupling example.

Why this module exists
----------------------
``examples/dem_fem_oneway`` ships four green acceptance gates, but two of them
cannot fail for a physical reason:

* **impulse ledger** -- ``Σ∫F dt`` is evaluated on the *same* force array on
  both sides of the coupling, so the "error" is identically zero (measured
  ``0.000e+00``); it is a consistency check on the trapezoid rule, not evidence
  that the momentum exchange is right;
* **work-energy** -- ``work_done = ∫F·dv_center`` and the beam energy both come
  from the same FEM object, so ``5.2e-10`` only says the integrator is
  self-consistent (PB-005 style tautology);
* **separability** -- true by construction of a one-way transfer.

Only the deflection gate (peak deflection vs ``P L³/48EI``) is external, and it
uses a *static* formula for a *dynamic* peak, which is only legitimate when the
contact time ``t_c`` is long compared with the beam period ``T₁``.  The shipped
docstring asserts ``t_c/T₁ ≈ 11`` for the default configuration; with the beam
section actually used (``L=0.3``, ``h=0.08``, ``E=2.1e11``, ``ρ=7850``) the
first period is ``T₁ ≈ 3.01 ms`` and ``t_c/T₁ ≈ 1.8`` -- so the static anchor
needs a dynamic replacement, not a footnote.

This module supplies three *independent* anchors, each computed by a route that
does not touch the code under test:

===============  ===========================================================
``hookean_impact`` closed-form linear-spring collision (peak force, contact
                 time, impulse, max overlap) from ``m``, ``k``, ``v`` alone
``hertz_impact``   closed-form Hertzian collision (same four quantities)
``beam_natural_frequencies`` / ``beam_mode_shape``
                 Euler-Bernoulli modal data from the continuum formula
``modal_response`` continuum modal superposition (Duhamel) driven by a
                 *measured* force history -- an implementation that shares no
                 line with ``fem_euler_bernoulli``
===============  ===========================================================

Every gate produced here is tagged with its :data:`INDEPENDENCE` class so a
reader can tell an external anchor from a self-comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "INDEPENDENCE",
    "Gate",
    "beam_mode_shape",
    "beam_natural_frequencies",
    "evaluate_gates",
    "hertz_impact",
    "hookean_impact",
    "modal_response",
    "modal_static_deflection",
]

#: Vocabulary for how independent a gate is from the code it verifies.
INDEPENDENCE = {
    "analytic": "closed-form theory; the code under test is not referenced",
    "independent_implementation": "second implementation sharing no code path",
    "cross_side": "different physical side of the coupling (DEM Kinematics vs FEM)",
    "self_comparison": "same data through a different arithmetic path (consistency only)",
}


@dataclass(frozen=True)
class Gate:
    """One machine-checkable acceptance entry."""

    name: str
    quantity: str
    reference: float
    measured: float
    tol_rel: float
    independence: str

    @property
    def rel_error(self) -> float:
        if self.reference == 0.0:
            return float("nan")
        return (self.measured - self.reference) / abs(self.reference)

    @property
    def passed(self) -> bool:
        return abs(self.rel_error) <= self.tol_rel

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "quantity": self.quantity,
            "reference": self.reference,
            "measured": self.measured,
            "rel_error": self.rel_error,
            "tol_rel": self.tol_rel,
            "independence": self.independence,
            "pass": self.passed,
        }


# --------------------------------------------------------------- collisions --
def hookean_impact(mass: float, stiffness: float, v_impact: float,
                   restitution: float = 1.0, *, target_mass: float = math.inf) -> dict:
    """Closed-form linear-spring (Hookean) collision of a sphere with a wall.

    The contact force is ``F = k δ`` with ``k`` the *effective* (series)
    stiffness and ``m_eff`` the reduced mass.  The contact is a half sine of
    duration ``t_c = π/ω``, ``ω = sqrt(k/m_eff)``:

    * peak overlap ``δ_max = v ω⁻¹``
    * peak force  ``F_max = k δ_max = v sqrt(k m_eff)``
    * impulse     ``J = (1 + e) m_eff v``  (``e = 1``: elastic)

    Parameters are the *nominal* case definition, so the result never looks at
    the DEM integrator, the recorder or the recorder's time stamping.
    """
    for name, value in (("mass", mass), ("stiffness", stiffness)):
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if not (isinstance(v_impact, (int, float)) and math.isfinite(v_impact)):
        raise ValueError("v_impact must be finite")
    if not (0.0 <= restitution <= 1.0):
        raise ValueError("restitution must lie in [0, 1]")

    m_eff = mass * target_mass / (mass + target_mass) if math.isfinite(target_mass) else mass
    omega = math.sqrt(stiffness / m_eff)
    t_c = math.pi / omega
    delta_max = abs(v_impact) / omega
    return {
        "m_eff": m_eff,
        "omega": omega,
        "contact_time_s": t_c,
        "max_overlap_m": delta_max,
        "peak_force_N": stiffness * delta_max,
        "impulse_Ns": (1.0 + restitution) * m_eff * abs(v_impact),
        "restitution": restitution,
        "model": "hookean",
    }


def hertz_impact(mass: float, stiffness: float, v_impact: float,
                 *, target_mass: float = math.inf) -> dict:
    """Closed-form Hertzian (``F = k δ^{3/2}``) collision.

    ``k`` is the Hertz stiffness coefficient (units ``N m^{-3/2}``).  The
    standard results (Johnson, *Contact Mechanics*, ch. 4; used elsewhere in
    this package as :func:`tensordem.mts_contact.hertz_contact_time`):

    * ``δ_max = (5 m_eff v² / (4 k))^{2/5}``
    * ``F_max = k δ_max^{3/2}``
    * ``t_c  = 2.943275 (5 m_eff / (4 k))^{2/5} v^{-1/5}``
    * elastic impulse ``J = 2 m_eff v``
    """
    for name, value in (("mass", mass), ("stiffness", stiffness)):
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if not (isinstance(v_impact, (int, float)) and math.isfinite(v_impact)) or v_impact == 0.0:
        raise ValueError("v_impact must be finite and non-zero")

    m_eff = mass * target_mass / (mass + target_mass) if math.isfinite(target_mass) else mass
    v = abs(v_impact)
    delta_max = (5.0 * m_eff * v * v / (4.0 * stiffness)) ** 0.4
    t_c = 2.943275 * (5.0 * m_eff / (4.0 * stiffness)) ** 0.4 * v ** -0.2
    return {
        "m_eff": m_eff,
        "contact_time_s": t_c,
        "max_overlap_m": delta_max,
        "peak_force_N": stiffness * delta_max ** 1.5,
        "impulse_Ns": 2.0 * m_eff * v,
        "restitution": 1.0,
        "model": "hertz",
    }


# -------------------------------------------------------------------- beam --
def _bc_check(bc: str) -> None:
    if bc != "simply_supported":
        raise NotImplementedError(
            f"modal anchors are implemented for bc='simply_supported' only "
            f"(got {bc!r}); clamped/cantilever continuum modes are not needed by "
            f"the verified case and are deliberately not faked")


def beam_mode_shape(n: int, x: float, length: float, *, bc: str = "simply_supported") -> float:
    """Euler-Bernoulli mode shape, normalised so that ``∫φ_n² dx = L/2``."""
    _bc_check(bc)
    if n < 1:
        raise ValueError("mode index n must be >= 1")
    return math.sin(n * math.pi * x / length)


def beam_natural_frequencies(ei: float, mass_per_length: float, length: float,
                             n_modes: int = 10, *, bc: str = "simply_supported") -> np.ndarray:
    """Circular natural frequencies ``ω_n = (nπ/L)² sqrt(EI/(ρA))`` in rad/s."""
    _bc_check(bc)
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    for name, value in (("ei", ei), ("mass_per_length", mass_per_length), ("length", length)):
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and positive")
    n = np.arange(1, n_modes + 1, dtype=np.float64)
    base = math.sqrt(ei / mass_per_length) * (math.pi / length) ** 2
    return base * n ** 2


def modal_static_deflection(load: float, length: float, ei: float,
                            *, bc: str = "simply_supported") -> float:
    """Mid-span static deflection of a centrally loaded beam (closed form)."""
    _bc_check(bc)
    return load * length ** 3 / (48.0 * ei)


def modal_response(force: Sequence[float], times: Sequence[float], *,
                   ei: float, mass_per_length: float, length: float,
                   x_load: float, x_probe: float,
                   n_modes: int = 9, bc: str = "simply_supported") -> dict:
    """Continuum modal superposition of a beam driven by a tabulated force.

    Solves the modal SDOF ``q̈_n + ω_n² q_n = F(t) φ_n(x_load)/M_n`` with
    ``M_n = ρA L/2`` **exactly** for piecewise-linear ``F(t)`` (Duhamel
    recursion, unconditionally stable -- no time-step tuning), and returns

    * ``deflection`` -- ``w(x_probe, t) = Σ q_n φ_n(x_probe)``,
    * ``energy``     -- ``Σ ½ (M_n q̇_n² + K_n q_n²)`` (beam mechanical energy),
    * ``modal``      -- per-mode ``q_n`` histories for diagnostics.

    Nothing here touches ``fem_euler_bernoulli``: the beam is represented by its
    continuum modes, so agreement with the FEM is a genuine cross-check of the
    assembled mass/stiffness matrices and the time integrator.  Modes with an
    anti-node at ``x_load`` (even ``n`` for a mid-span load) are not excited and
    stay identically zero, so ``n_modes = 9`` already truncates below 1e-5 of
    the peak (``q_n ∝ n⁻⁴``).
    """
    times = np.asarray(times, dtype=np.float64)
    force = np.asarray(force, dtype=np.float64)
    if times.ndim != 1 or force.ndim != 1 or times.size != force.size or times.size < 2:
        raise ValueError("times and force must be 1-D arrays of equal length >= 2")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")

    omega = beam_natural_frequencies(ei, mass_per_length, length, n_modes, bc=bc)
    mass_modal = mass_per_length * length / 2.0          # ∫φ_n² dx = L/2
    stiff_modal = mass_modal * omega ** 2
    phi_load = np.array([beam_mode_shape(n, x_load, length, bc=bc)
                         for n in range(1, n_modes + 1)])
    phi_probe = np.array([beam_mode_shape(n, x_probe, length, bc=bc)
                          for n in range(1, n_modes + 1)])

    q = np.zeros(n_modes)
    qd = np.zeros(n_modes)
    w = np.zeros(times.size)
    energy = np.zeros(times.size)
    q_hist = np.zeros((times.size, n_modes))
    q_hist[0] = q

    for k in range(times.size - 1):
        dt = times[k + 1] - times[k]
        g0 = force[k] * phi_load / mass_modal              # modal acceleration
        slope = (force[k + 1] - force[k]) / dt
        g1 = slope * phi_load / mass_modal                 # ramp rate of g
        theta = omega * dt
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        q_new = (q * cos_t + (qd / omega) * sin_t
                 + (g0 / omega ** 2) * (1.0 - cos_t)
                 + (g1 / omega ** 3) * (theta - sin_t))
        qd_new = (-q * omega * sin_t + qd * cos_t
                  + (g0 / omega) * sin_t
                  + (g1 / omega ** 2) * (1.0 - cos_t))
        q, qd = q_new, qd_new
        w[k + 1] = float(q @ phi_probe)
        energy[k + 1] = float(0.5 * (mass_modal * qd ** 2 + stiff_modal * q ** 2).sum())
        q_hist[k + 1] = q

    return {
        "deflection": w,
        "energy": energy,
        "modal": q_hist,
        "omega": omega,
        "phi_probe": phi_probe,
        "peak_deflection": float(np.max(np.abs(w))),
        "peak_deflection_time": float(times[int(np.argmax(np.abs(w)))]),
        "final_energy": float(energy[-1]),
        "max_energy": float(energy.max()),
        "n_modes": n_modes,
    }


# ------------------------------------------------------------------- gates --
def evaluate_gates(entries: Iterable[tuple]) -> list[Gate]:
    """Build :class:`Gate` objects from ``(name, quantity, reference, measured, tol, independence)``."""
    gates = []
    for name, quantity, reference, measured, tol, independence in entries:
        if independence not in INDEPENDENCE:
            raise ValueError(f"unknown independence class {independence!r}")
        gates.append(Gate(name=name, quantity=quantity, reference=float(reference),
                          measured=float(measured), tol_rel=float(tol),
                          independence=independence))
    return gates


def summarise(gates: Sequence[Gate]) -> dict:
    """Reduce a gate list to the machine verdict used by the report/CI."""
    external = [g for g in gates if g.independence != "self_comparison"]
    self_checks = [g for g in gates if g.independence == "self_comparison"]
    return {
        "n_gates": len(gates),
        "n_external": len(external),
        "n_self_comparison": len(self_checks),
        "external_pass": sum(1 for g in external if g.passed),
        "all_external_pass": all(g.passed for g in external) if external else True,
        "worst_external_rel_error": max((abs(g.rel_error) for g in external), default=float("nan")),
        "gates": [g.as_dict() for g in gates],
    }