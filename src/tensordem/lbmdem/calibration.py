"""Case drivers and gate evaluation for the LBM--DEM drag-closure calibration.

Three drivers:

``run_fixed_sphere``
    A body-force-driven periodic box with one or two *fixed* spheres.  The force
    density is known exactly from the momentum balance (``g * V_fluid``), and the
    immersed-boundary force is measured independently, so the two can be
    cross-checked.  Covers the Stokes, Schiller--Naumann, IBM-scheme, resolution,
    ``Cs`` and ``L/d`` gates.

``run_settling``
    A free sphere settling under an effective weight in a quiescent periodic box
    with a moving direct-forcing immersed boundary.  Covers the terminal-velocity
    gate, the "no negative-drag spike" gate and the global momentum audit.

All results are plain dictionaries (JSON-friendly) so the benchmark runner can
write ``result.json`` without any post-hoc hand editing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from .drag import (
    array_correction,
    drag_from_cd,
    reynolds,
    schiller_naumann_cd,
    stokes_drag,
    tandem_pair_ratio,
    terminal_velocity_schiller_naumann,
    terminal_velocity_stokes,
)
from .ibm import DirectForcingIBM, MomentumExchangeIBM, Sphere
from .lattice import Lattice, LatticeShape

__all__ = [
    "FixedSphereSpec",
    "TowedSphereSpec",
    "SettlingSpec",
    "run_fixed_sphere",
    "run_towed_sphere",
    "run_settling",
    "evaluate_gates",
]


@dataclass(frozen=True)
class FixedSphereSpec:
    """Configuration of a body-force-driven fixed-sphere case."""

    label: str = "case"
    diameter: float = 8.0
    domain_ratio: float = 8.0
    re_target: float = 0.05
    scheme: str = "mem"  # "mem" | "direct"
    smagorinsky_cs: float = 0.0
    tau0: float = 0.8
    marker_density: float = 1.4
    separation_radius: float | None = None  # centre separation / radius
    flow_control: bool = True  # constant-flow-rate pump instead of a frozen body force
    controller_gain: float = 0.02
    controller_window: int = 10
    max_steps: int = 6000
    min_steps: int = 400
    sample_every: int = 100
    rtol: float = 1.0e-6
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.scheme not in {"mem", "direct"}:
            raise ValueError("scheme must be 'mem' or 'direct'")
        if self.re_target <= 0.0:
            raise ValueError("re_target must be positive")
        if self.separation_radius is not None and self.separation_radius <= 1.0:
            raise ValueError("separation_radius must exceed 1 (non-overlapping spheres)")

    @property
    def n_spheres(self) -> int:
        return 1 if self.separation_radius is None else 2


def _solid_fraction_of_box(spec: FixedSphereSpec, length: float) -> float:
    d = spec.diameter
    volume = spec.n_spheres * math.pi * d**3 / 6.0
    return volume / length**3


def _sphere_centres(spec: FixedSphereSpec, length: float) -> list[tuple[float, float, float]]:
    mid = length / 2.0
    if spec.separation_radius is None:
        return [(mid, mid, mid)]
    half = 0.5 * spec.separation_radius * spec.diameter / 2.0
    return [(mid - half, mid, mid), (mid + half, mid, mid)]


def run_fixed_sphere(spec: FixedSphereSpec = FixedSphereSpec()) -> dict[str, Any]:
    """Drive a fixed sphere (or a tandem pair) and measure the drag force."""
    length = int(round(spec.domain_ratio * spec.diameter))
    if length % 2:
        raise ValueError("domain length must be even for a centred sphere")
    nu = (spec.tau0 - 0.5) / 3.0
    diameter = spec.diameter
    u_target = spec.re_target * nu / diameter
    solid_fraction = _solid_fraction_of_box(spec, length)
    correction = array_correction(solid_fraction)
    if spec.re_target < 1.0:
        force_estimate = stokes_drag(nu, diameter, u_target) * correction
    else:
        force_estimate = (
            drag_from_cd(schiller_naumann_cd(spec.re_target), 1.0, u_target, diameter) * correction
        )
    body = spec.n_spheres * force_estimate / length**3

    lattice = Lattice(
        LatticeShape(length, length, length),
        tau0=spec.tau0,
        smagorinsky_cs=spec.smagorinsky_cs,
        body_force=(body, 0.0, 0.0),
        dtype=spec.dtype,
    )
    # A body-force-driven periodic box needs a viscous diffusion time L^2/nu
    # (thousands of steps) to build the pressure field that balances the force,
    # which is unaffordable and adds nothing physical.  The pump controller below
    # instead holds the *flow rate* at its target by rescaling the body force, so
    # the flow starts (and stays) close to the steady state and only the local
    # sphere boundary layer has to adjust.
    lattice.initialize_velocity((u_target, 0.0, 0.0))
    spheres = [Sphere(centre, diameter / 2.0) for centre in _sphere_centres(spec, length)]

    if spec.scheme == "mem":
        ibms = [MomentumExchangeIBM(lattice, sphere) for sphere in spheres]
        n_interface = [ibm.n_interface for ibm in ibms]

        def hook(f_post: torch.Tensor) -> torch.Tensor:
            for ibm in ibms:
                f_post = ibm.apply(f_post)
            return f_post

    else:
        n_interface = []
        ibms = [
            DirectForcingIBM(lattice, sphere, marker_density=spec.marker_density)
            for sphere in spheres
        ]

    def measure() -> dict[str, Any]:
        rho, u = lattice.macroscopic(lattice.f)
        forces = torch.stack([ibm.force_on_particle for ibm in ibms])
        momentum = lattice.fluid_momentum()
        return {
            "force": forces,  # (n_spheres, 3)
            "u_mean": float(momentum[0].item() / lattice.shape.volume),
            "mass": float(rho.mean().item()),
            "max_mach": float(u.abs().max().item() * math.sqrt(3.0)),
        }

    history: list[dict[str, float]] = []
    converged = False
    sample = measure()
    impulse_injected = 0.0
    u_window: list[float] = []
    for step in range(1, spec.max_steps + 1):
        if spec.flow_control:
            _, u_now = lattice.macroscopic(lattice.f)
            u_window.append(float(u_now[..., 0].mean().item()))
            if len(u_window) > spec.controller_window:
                u_window.pop(0)
            u_control = sum(u_window) / len(u_window)
            gain = spec.controller_gain
            factor = 1.0 + gain * (u_target - u_control) / u_target
            factor = min(max(factor, 0.5), 1.5)
            body = min(max(body * factor, 0.05 * spec.n_spheres * force_estimate / length**3),
                       20.0 * spec.n_spheres * force_estimate / length**3)
            lattice.body_force = torch.tensor([body, 0.0, 0.0], dtype=lattice.dtype)
        impulse_injected += body * lattice.shape.volume
        if spec.scheme == "mem":
            lattice.step(post_collision_hook=hook)
        else:
            _, u = lattice.macroscopic(lattice.f)
            density = torch.zeros((length, length, length, 3), dtype=lattice.dtype)
            for ibm in ibms:
                density = density + ibm.apply(u)
            lattice.step(force_density=density)
        if step % spec.sample_every == 0:
            new = measure()
            total_old = float(sample["force"][:, 0].sum().item())
            total_new = float(new["force"][:, 0].sum().item())
            history.append(
                {
                    "step": float(step),
                    "force": total_new,
                    "u_mean": new["u_mean"],
                    "max_mach": new["max_mach"],
                }
            )
            sample = new
            if (
                step >= spec.min_steps
                and abs(total_new - total_old) <= spec.rtol * max(abs(total_new), 1e-30)
            ):
                converged = True
                break

    final = measure()
    forces = final["force"]
    total_force = float(forces[:, 0].sum().item())
    u_mean = final["u_mean"]
    re = reynolds(diameter, u_mean, nu)
    area = math.pi * diameter**2 / 4.0
    cd_measured = total_force / (0.5 * u_mean**2 * area) if u_mean > 0 else float("nan")
    cd_reference = schiller_naumann_cd(re) if re > 1e-8 else float("nan")
    stokes_ratio = total_force / (
        spec.n_spheres * stokes_drag(nu, diameter, u_mean)
    )
    momentum_balance = total_force / (body * length**3)
    pump_force = body * length**3
    return {
        "label": spec.label,
        "kind": "fixed_sphere",
        "scheme": spec.scheme,
        "diameter": diameter,
        "domain_length": length,
        "domain_ratio": spec.domain_ratio,
        "n_spheres": spec.n_spheres,
        "separation_radius": spec.separation_radius,
        "solid_fraction": solid_fraction,
        "array_correction": correction,
        "smagorinsky_cs": spec.smagorinsky_cs,
        "tau0": spec.tau0,
        "nu": nu,
        "re_target": spec.re_target,
        "re_measured": re,
        "reynolds": re,
        "u_mean": u_mean,
        "max_mach": final["max_mach"],
        "force_total": total_force,
        "force_per_sphere": [float(v) for v in forces[:, 0].tolist()],
        "force_estimated": spec.n_spheres * force_estimate,
        "pump_force": pump_force,
        "body_force": body,
        "u_target": u_target,
        "flow_control": spec.flow_control,
        "momentum_balance_ratio": momentum_balance,
        "cd_measured": cd_measured,
        "cd_schiller_naumann": cd_reference,
        "cd_rel_error": (cd_measured / cd_reference - 1.0) if cd_reference == cd_reference else float("nan"),
        "stokes_ratio": stokes_ratio,
        "stokes_ratio_corrected": stokes_ratio / correction,
        "n_interface": n_interface,
        "n_markers": [ibm.n_markers for ibm in ibms] if spec.scheme == "direct" else [],
        "steps": int(lattice.steps),
        "converged": converged,
        "history": history,
    }


@dataclass(frozen=True)
class SettlingSpec:
    """Configuration of a free-settling sphere case."""

    label: str = "settling"
    diameter: float = 8.0
    domain_ratio: float = 8.0
    density_ratio: float = 2.0
    tau0: float = 0.8
    re_target: float = 1.0
    scheme: str = "mem"  # "mem" | "direct"
    marker_density: float = 1.4
    steps: int = 3000
    sample_every: int = 5
    window: int = 200
    dtype: str = "float64"


def run_settling(spec: SettlingSpec = SettlingSpec()) -> dict[str, Any]:
    """Settle a sphere under an effective weight and audit coupling closure."""
    length = int(round(spec.domain_ratio * spec.diameter))
    if length % 2:
        raise ValueError("domain length must be even for a centred sphere")
    nu = (spec.tau0 - 0.5) / 3.0
    diameter = spec.diameter
    radius = diameter / 2.0
    volume = math.pi * diameter**3 / 6.0
    mass = spec.density_ratio * volume
    u_guess = spec.re_target * nu / diameter
    # Effective weight that would give the target terminal velocity in the
    # Stokes limit; the case then relaxes to the true Schiller-Naumann balance.
    gravity = 18.0 * nu * u_guess / ((spec.density_ratio - 1.0) * diameter**2)
    drive = (mass - volume) * gravity

    lattice = Lattice(LatticeShape(length, length, length), tau0=spec.tau0, dtype=spec.dtype)
    position = torch.tensor([length / 2.0, length / 2.0, length / 2.0], dtype=lattice.dtype)
    # Warm start at the Stokes terminal velocity: starting from rest would give
    # an impulsive-start transient whose added-mass peak is not the physics this
    # case is gated on.
    velocity = torch.zeros(3, dtype=lattice.dtype)
    velocity[0] = terminal_velocity_stokes(diameter, spec.density_ratio, 1.0, nu, gravity=gravity)
    momentum_initial = mass * velocity + lattice.fluid_momentum()

    history: list[dict[str, float]] = []
    max_drag = -float("inf")
    min_drag = float("inf")
    worst_jump = 0.0
    previous_drag = None
    momentum_residual = 0.0
    drag_series: list[float] = []

    for step in range(1, spec.steps + 1):
        centre = tuple(float((position[i] % length).item()) for i in range(3))
        sphere = Sphere(centre, radius)
        if spec.scheme == "mem":
            # The solid fraction is rebuilt every step because the sphere moves;
            # the cost is a 4^3 sub-sampled bounding box, negligible next to one
            # lattice step.
            ibm = MomentumExchangeIBM(
                lattice, sphere, wall_velocity=(float(velocity[0].item()), 0.0, 0.0)
            )
            lattice.step(post_collision_hook=ibm.apply)
        else:
            ibm = DirectForcingIBM(
                lattice,
                sphere,
                marker_density=spec.marker_density,
                velocity=(float(velocity[0].item()), 0.0, 0.0),
            )
            _, u = lattice.macroscopic(lattice.f)
            density = ibm.apply(u)
            lattice.step(force_density=density)
        force_on_particle = ibm.force_on_particle
        drag = -float(force_on_particle[0].item())  # +x is the settling direction
        velocity = velocity.clone()
        velocity[0] = velocity[0] + (drive + float(force_on_particle[0].item())) / mass
        position = position + velocity
        fluid_momentum = lattice.fluid_momentum()
        momentum_now = mass * velocity[0] + fluid_momentum[0]
        residual = abs(float(momentum_now.item()) - float(momentum_initial[0].item()) - drive * step)
        momentum_residual = max(momentum_residual, residual / max(abs(drive) * step, 1e-30))
        drag_series.append(drag)
        max_drag = max(max_drag, drag)
        min_drag = min(min_drag, drag)
        if previous_drag is not None:
            worst_jump = max(worst_jump, abs(drag - previous_drag) / max(abs(drag), 1e-30))
        previous_drag = drag
        if step % spec.sample_every == 0:
            history.append(
                {
                    "step": float(step),
                    "velocity": float(velocity[0].item()),
                    "relative_velocity": float(velocity[0].item() - fluid_momentum[0].item() / length**3),
                    "drag": drag,
                    "position": float(position[0].item()),
                    "momentum_total": float(momentum_now.item()),
                }
            )

    window = min(spec.window, len(history))
    tail = history[-window:]
    u_terminal = sum(row["velocity"] for row in tail) / window
    u_relative = sum(row["relative_velocity"] for row in tail) / window
    drag_terminal = sum(row["drag"] for row in tail) / window
    re_terminal = reynolds(diameter, u_relative, nu)
    solid_fraction = volume / length**3
    correction = array_correction(solid_fraction)
    u_analytic_sn = terminal_velocity_schiller_naumann(
        diameter, spec.density_ratio, 1.0, nu, gravity=gravity
    )
    u_analytic_stokes = terminal_velocity_stokes(
        diameter, spec.density_ratio, 1.0, nu, gravity=gravity
    )
    cd_measured = drag_terminal / (0.5 * u_relative**2 * math.pi * diameter**2 / 4.0)
    return {
        "label": spec.label,
        "kind": "settling",
        "scheme": spec.scheme,
        "diameter": diameter,
        "domain_length": length,
        "domain_ratio": spec.domain_ratio,
        "density_ratio": spec.density_ratio,
        "tau0": spec.tau0,
        "nu": nu,
        "gravity": gravity,
        "drive_force": drive,
        "mass": mass,
        "solid_fraction": solid_fraction,
        "array_correction": correction,
        "u_terminal": u_terminal,
        "u_relative_terminal": u_relative,
        "u_relative_terminal_corrected": u_relative * correction,
        "u_analytic_stokes": u_analytic_stokes,
        "u_analytic_schiller_naumann": u_analytic_sn,
        "u_rel_error_stokes_corrected": u_relative * correction / u_analytic_stokes - 1.0,
        "u_rel_error_sn_corrected": u_relative * correction / u_analytic_sn - 1.0,
        "re_terminal": re_terminal,
        "drag_terminal": drag_terminal,
        "drag_balance_residual": abs(drag_terminal - drive) / drive,
        "cd_measured": cd_measured,
        "cd_schiller_naumann": schiller_naumann_cd(re_terminal),
        "drag_max": max_drag,
        "drag_min": min_drag,
        "drag_never_negative": min_drag > 0.0,
        "max_step_jump": worst_jump,
        "momentum_residual_max": momentum_residual,
        "steps": spec.steps,
        "history": history,
    }


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def _gate(name: str, criterion: str, measured: float, threshold: float, evidence: str,
          *, relu: bool = True) -> dict[str, Any]:
    if relu:
        passed = abs(measured) <= threshold
    else:
        passed = measured <= threshold
    return {
        "gate": name,
        "criterion": criterion,
        "measured": measured,
        "threshold": threshold,
        "passed": bool(passed),
        "evidence": evidence,
    }


def evaluate_gates(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Turn measured case results into machine-checkable gates.

    Each gate carries its criterion text, the measured number, the threshold it
    is compared with and the case labels it was computed from, so that a reader
    can re-derive every pass/fail without re-running anything by hand.
    """
    gates: list[dict[str, Any]] = []
    fixed = [case for case in cases if case["kind"] == "fixed_sphere"]

    # -- Stokes anchor ---------------------------------------------------
    stokes = [
        case for case in fixed
        if case["n_spheres"] == 1 and case["scheme"] == "mem" and case["re_target"] < 1.0
    ]
    reference = next((c for c in stokes if c["label"].startswith("stokes_d8")), stokes[0] if stokes else None)
    if reference is not None:
        gates.append(
            _gate(
                "stokes_drag_analytic",
                "F/(3 pi mu d U) after periodic-array correction vs 1 (Stokes), <= 2 %",
                reference["stokes_ratio_corrected"] - 1.0,
                0.02,
                f"{reference['label']} (d={reference['diameter']:g}, L/d={reference['domain_ratio']:g}, "
                f"Re={reference['re_measured']:.3g}, correction={reference['array_correction']:.4f})",
            )
        )

    # -- Cs insensitivity ------------------------------------------------
    # The Cs scan is judged at a *fixed* geometry: every Stokes case that shares
    # the diameter and box of the first Cs-scan case, including the Cs = 0
    # baseline, so the spread is not measured against a different discretisation.
    scan = [c for c in stokes if "cs" in c["label"]]
    cs_cases = []
    if scan:
        d0, l0 = scan[0]["diameter"], scan[0]["domain_ratio"]
        cs_cases = [c for c in stokes
                    if c["diameter"] == d0 and c["domain_ratio"] == l0 and c["n_spheres"] == 1]
    if len(cs_cases) >= 2:
        ratios = [c["stokes_ratio_corrected"] for c in cs_cases]
        mean = sum(ratios) / len(ratios)
        gates.append(
            _gate(
                "stokes_cs_insensitivity",
                "spread (max-min)/mean of the corrected Stokes drag over Cs in {0.5,1,2}, <= 2 %",
                (max(ratios) - min(ratios)) / mean,
                0.02,
                ", ".join(c["label"] for c in cs_cases),
            )
        )

    # -- domain insensitivity --------------------------------------------
    ld_cases = [c for c in stokes if "ld" in c["label"]]
    if len(ld_cases) >= 2:
        ratios = [c["stokes_ratio_corrected"] for c in ld_cases]
        mean = sum(ratios) / len(ratios)
        gates.append(
            _gate(
                "stokes_domain_insensitivity",
                "spread (max-min)/mean of the corrected Stokes drag over L/d in {8,16}, <= 1 %",
                (max(ratios) - min(ratios)) / mean,
                0.01,
                ", ".join(c["label"] for c in ld_cases),
            )
        )

    # -- MEM vs direct forcing -------------------------------------------
    grouped: dict[tuple, dict[str, dict[str, Any]]] = {}
    for case in fixed:
        if case["n_spheres"] != 1:
            continue
        key = (case["diameter"], case["domain_length"], round(case["re_target"], 9))
        grouped.setdefault(key, {})[case["scheme"]] = case
    for key, pair in sorted(grouped.items()):
        if "mem" in pair and "direct" in pair:
            mem, direct = pair["mem"], pair["direct"]
            gates.append(
                _gate(
                    f"ibm_scheme_agreement_d{mem['diameter']:g}_re{mem['re_target']:.3g}",
                    "|F_MEM - F_direct| / F_MEM for the same case, <= 2 %",
                    abs(direct["force_total"] - mem["force_total"]) / abs(mem["force_total"]),
                    0.02,
                    f"{mem['label']} vs {direct['label']}",
                )
            )

    # -- Schiller-Naumann -------------------------------------------------
    for case in fixed:
        if case["n_spheres"] != 1 or case["re_target"] < 1.0:
            continue
        gates.append(
            _gate(
                f"schiller_naumann_{case['label']}",
                "C_D measured vs Schiller-Naumann at the measured Re, <= 5 %",
                case["cd_rel_error"],
                0.05,
                f"{case['label']} (Re={case['re_measured']:.3g}, d={case['diameter']:g}, "
                f"L/d={case['domain_ratio']:g})",
            )
        )

    # -- two-sphere tandem ------------------------------------------------
    pairs = [c for c in fixed if c["n_spheres"] == 2]
    singles = [c for c in fixed if c["n_spheres"] == 1 and c["scheme"] == "mem" and c["re_target"] < 1.0]
    if pairs and singles:
        pair = pairs[0]
        reference = min(
            singles,
            key=lambda case: abs(case["domain_length"] - pair["domain_length"])
            + abs(case["diameter"] - pair["diameter"]),
        )
        xi = 1.0 / pair["separation_radius"]
        corrected = (
            pair["force_total"] / (2.0 * reference["force_total"])
            * reference["array_correction"] / pair["array_correction"]
        )
        analytic = tandem_pair_ratio(xi)
        gates.append(
            _gate(
                "two_sphere_tandem_ratio",
                "F_pair/(2 F_single) vs 1 - 3 xi/2 (Stimson-Jeffery leading term), <= 5 %",
                corrected / analytic - 1.0,
                0.05,
                f"{pair['label']} vs {reference['label']} (xi=a/r={xi:.3g}, "
                f"series truncation O(xi^3)={xi ** 3:.3g})",
            )
        )

    # -- settling ---------------------------------------------------------
    for case in [c for c in cases if c["kind"] == "settling"]:
        gates.append(
            _gate(
                f"terminal_velocity_{case['label']}",
                "corrected relative terminal velocity vs the Schiller-Naumann drag "
                "balance, <= 3 %",
                case["u_rel_error_sn_corrected"],
                0.03,
                f"{case['label']} (Re_t={case['re_terminal']:.3g})",
            )
        )
        gates.append(
            _gate(
                f"drag_balance_{case['label']}",
                "|F_drag(terminal) - effective weight| / weight, <= 3 %",
                case["drag_balance_residual"],
                0.03,
                case["label"],
            )
        )
        gates.append(
            {
                "gate": f"drag_no_negative_artifact_{case['label']}",
                "criterion": "drag time series never negative and no step-to-step jump > 5 %",
                "measured": {
                    "min_drag": case["drag_min"],
                    "max_step_jump": case["max_step_jump"],
                },
                "threshold": {"min_drag": 0.0, "max_step_jump": 0.05},
                "passed": bool(case["drag_never_negative"] and case["max_step_jump"] <= 0.05),
                "evidence": case["label"],
            }
        )
        gates.append(
            {
                "gate": f"momentum_conservation_{case['label']}",
                "criterion": "max relative residual of d(m v)/dt + dP_fluid/dt - F_external, <= 1e-10",
                "measured": case["momentum_residual_max"],
                "threshold": 1e-10,
                "passed": bool(case["momentum_residual_max"] <= 1e-10),
                "evidence": case["label"],
            }
        )

    passed = all(gate["passed"] for gate in gates)
    return {
        "n_gates": len(gates),
        "n_passed": sum(1 for gate in gates if gate["passed"]),
        "all_recomputed_pass": bool(passed),
        "gates": gates,
    }

@dataclass(frozen=True)
class TowedSphereSpec:
    """Configuration of a sphere towed at constant velocity through a periodic box.

    This is the canonical arrangement for the Hasimoto / Sangani--Acrivos array
    correction: the sphere and all of its periodic images translate at ``u_tow``,
    so the measured drag divided by ``3 pi mu d (u_tow - u_backflow)`` must equal
    the array correction.  Unlike a body-force-driven box it has *no* global
    pressure transient: the disturbance is local and reaches a steady state in a
    few ``d^2 / nu`` steps.

    Limitation (measured during P55-N08): because the constraint keeps injecting
    the drag force into a periodic box, the fluid momentum drifts and the
    *relative* velocity decays over thousands of steps, so this driver is kept
    for local-transient studies only.  The campaign matrix uses the constant
    flow-rate pump (:func:`run_fixed_sphere`) for the fixed-geometry gates.
    """

    label: str = "towed"
    diameter: float = 6.0
    domain_ratio: float = 8.0
    re_target: float = 0.05
    scheme: str = "mem"  # "mem" | "direct"
    smagorinsky_cs: float = 0.0
    tau0: float = 0.8
    marker_density: float = 1.4
    separation_radius: float | None = None
    steps: int = 800
    min_steps: int = 200
    sample_every: int = 25
    rtol: float = 1.0e-4
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.scheme not in {"mem", "direct"}:
            raise ValueError("scheme must be 'mem' or 'direct'")
        if self.re_target <= 0.0:
            raise ValueError("re_target must be positive")

    @property
    def n_spheres(self) -> int:
        return 1 if self.separation_radius is None else 2


def run_towed_sphere(spec: TowedSphereSpec = TowedSphereSpec()) -> dict[str, Any]:
    """Tow a sphere (or a tandem pair) at constant velocity and measure the drag."""
    length = int(round(spec.domain_ratio * spec.diameter))
    if length % 2:
        raise ValueError("domain length must be even for a centred sphere")
    diameter = spec.diameter
    radius = diameter / 2.0
    volume_sphere = math.pi * diameter**3 / 6.0
    solid_fraction = spec.n_spheres * volume_sphere / length**3
    correction = array_correction(solid_fraction)
    nu = (spec.tau0 - 0.5) / 3.0
    u_tow = spec.re_target * nu / diameter

    lattice = Lattice(
        LatticeShape(length, length, length),
        tau0=spec.tau0,
        smagorinsky_cs=spec.smagorinsky_cs,
        dtype=spec.dtype,
    )
    offsets = [0.0] if spec.separation_radius is None else [
        -0.5 * spec.separation_radius * radius, 0.5 * spec.separation_radius * radius
    ]
    mid = length / 2.0
    travel = torch.zeros(1, dtype=lattice.dtype)
    if spec.scheme == "mem":
        ibms = [
            MomentumExchangeIBM(lattice, Sphere((mid + dx, mid, mid), radius),
                                wall_velocity=(u_tow, 0.0, 0.0))
            for dx in offsets
        ]

        def hook(f_post: torch.Tensor) -> torch.Tensor:
            for ibm in ibms:
                f_post = ibm.apply(f_post)
            return f_post
    else:
        ibms = [None] * len(offsets)

    history: list[dict[str, float]] = []
    previous = None
    converged = False
    drag_series: list[float] = []
    for step in range(1, spec.steps + 1):
        travel += u_tow
        if spec.scheme == "mem":
            lattice.step(post_collision_hook=hook)
        else:
            current = []
            for index, dx in enumerate(offsets):
                centre = ((mid + dx + float(travel.item())) % length, mid, mid)
                current.append(
                    DirectForcingIBM(lattice, Sphere(centre, radius),
                                     marker_density=spec.marker_density,
                                     velocity=(u_tow, 0.0, 0.0))
                )
            ibms = current
            _, u = lattice.macroscopic(lattice.f)
            density = torch.zeros((length, length, length, 3), dtype=lattice.dtype)
            for ibm in ibms:
                density = density + ibm.apply(u)
            lattice.step(force_density=density)
        forces = torch.stack([ibm.force_on_particle for ibm in ibms])
        total_drag = -float(forces[:, 0].sum().item())
        drag_series.append(total_drag)
        if step % spec.sample_every == 0:
            momentum = lattice.fluid_momentum()
            u_mean = float(momentum[0].item() / lattice.shape.volume)
            history.append(
                {
                    "step": float(step),
                    "drag": total_drag,
                    "u_mean": u_mean,
                    "relative_velocity": u_tow - u_mean,
                }
            )
            if previous is not None and step >= spec.min_steps:
                if abs(total_drag - previous) <= spec.rtol * abs(total_drag):
                    converged = True
                    break
            previous = total_drag

    forces = torch.stack([ibm.force_on_particle for ibm in ibms])
    total_force = -float(forces[:, 0].sum().item())
    momentum = lattice.fluid_momentum()
    u_mean = float(momentum[0].item() / lattice.shape.volume)
    u_rel = u_tow - u_mean
    re = reynolds(diameter, u_rel, nu)
    area = math.pi * diameter**2 / 4.0
    cd_measured = total_force / (0.5 * u_rel**2 * area) if u_rel > 0 else float("nan")
    cd_reference = schiller_naumann_cd(re) if re > 1e-8 else float("nan")
    stokes_ratio = total_force / (spec.n_spheres * stokes_drag(nu, diameter, u_rel))
    return {
        "label": spec.label,
        "kind": "fixed_sphere",
        "drive": "towed",
        "scheme": spec.scheme,
        "diameter": diameter,
        "domain_length": length,
        "domain_ratio": spec.domain_ratio,
        "n_spheres": spec.n_spheres,
        "separation_radius": spec.separation_radius,
        "solid_fraction": solid_fraction,
        "array_correction": correction,
        "smagorinsky_cs": spec.smagorinsky_cs,
        "tau0": spec.tau0,
        "nu": nu,
        "re_target": spec.re_target,
        "re_measured": re,
        "reynolds": re,
        "u_tow": u_tow,
        "u_mean": u_mean,
        "u_relative": u_rel,
        "max_mach": u_tow * math.sqrt(3.0),
        "force_total": total_force,
        "force_per_sphere": [-float(v) for v in forces[:, 0].tolist()],
        "cd_measured": cd_measured,
        "cd_schiller_naumann": cd_reference,
        "cd_rel_error": (cd_measured / cd_reference - 1.0) if cd_reference == cd_reference else float("nan"),
        "stokes_ratio": stokes_ratio,
        "stokes_ratio_corrected": stokes_ratio / correction,
        "n_interface": [int(getattr(ibm, "n_interface", 0)) for ibm in ibms],
        "n_markers": [int(getattr(ibm, "n_markers", 0)) for ibm in ibms],
        "drag_min": min(drag_series),
        "drag_max": max(drag_series),
        "drag_never_negative": min(drag_series) > 0.0,
        "steps": step,
        "converged": converged,
        "history": history,
    }
