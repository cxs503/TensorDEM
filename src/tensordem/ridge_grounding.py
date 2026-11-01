"""Ice-ridge grounding on a buried subsea pipeline: coupled DEM workflow.

The workflow couples three levels, each of which is separately verifiable:

1. **DEM** (:mod:`tensordem.ridge_dem`) resolves the ridge keel - pipeline
   contact and the keel - seabed gouging, and produces the vertical (and, in
   the lateral loading mode, horizontal) load-displacement curve per metre of
   ridge.
2. **Subgrade** (:class:`tensordem.ridge_geotech.MohrCoulombSoil`) turns that
   load into the pipeline settlement through a hyperbolic Mohr-Coulomb
   reaction curve, and provides the ultimate bearing capacity that limits the
   pipe-soil system.
3. **Structure** (:func:`tensordem.ridge_geotech.solve_beam_on_winkler`) carries
   the longitudinally distributed load into bending moment, stress and the
   DNV-OS-F101 combined-loading utilisation.

The result of one run is a load-displacement record plus a set of scalar
metrics: peak load (ultimate capacity of the ice-pipe contact), penetration at
the peak, post-peak softening ratio (the ISO 19906-style peak-then-softening
signature), and the residual load.

Honest scope: the DEM is two-dimensional in plane strain with unit thickness,
disks carry no rotation, and the reported loads are per metre of ridge length.
Numbers are DEM predictions for the documented micro-parameter set, not
validated field measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .ridge_dem import (Plane, RidgeAssembly, RidgeKeel, RidgeMaterial,
                        generate_keel_assembly)
from .ridge_geotech import (GRAVITY, MohrCoulombSoil, PipeSection,
                            SofteningVerticalSpring, dnv_os_f101_utilization,
                            hetenyi_infinite_beam_patch_load,
                            hetenyi_infinite_beam_point_load, solve_beam_on_winkler)

__all__ = [
    "GroundingScenario",
    "GroundingResult",
    "run_grounding",
    "load_displacement_metrics",
    "axial_footprint_loads",
    "pipe_stress_profile",
    "continuum_limit_capacity",
    "capacity_envelope",
]

GRAVITY_ACCELERATION = GRAVITY


@dataclass(frozen=True)
class GroundingScenario:
    """Full input deck for one ridge-grounding case."""

    keel: RidgeKeel
    material: RidgeMaterial
    soil: MohrCoulombSoil
    section: PipeSection
    cover_depth: float = 0.8
    tip_clearance: float = 1.0
    stroke: float | None = None
    platen_velocity: float = 0.25
    loading: str = "vertical"
    pipe_response: str = "compliant"
    contact_length: float = 30.0
    dt: float | None = None
    steps_factor: float = 1.0
    sample_every: int = 20
    plastic_rate: float = 0.05
    pipe_line_mass: float = 400.0
    pipe_damping_ratio: float = 0.7

    def __post_init__(self) -> None:
        for name in ("cover_depth", "tip_clearance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.stroke is None:
            object.__setattr__(self, "stroke",
                               self.tip_clearance + self.cover_depth + 0.3)
        if self.stroke <= 0 or not math.isfinite(self.stroke):
            raise ValueError("stroke must be finite and positive")
        if self.platen_velocity <= 0 or not math.isfinite(self.platen_velocity):
            raise ValueError("platen_velocity must be finite and positive")
        if self.loading not in ("vertical", "lateral"):
            raise ValueError("loading must be 'vertical' or 'lateral'")
        if self.pipe_response not in ("compliant", "rigid"):
            raise ValueError("pipe_response must be 'compliant' or 'rigid'")
        if self.contact_length <= 0:
            raise ValueError("contact_length must be positive")
        if not isinstance(self.sample_every, int) or self.sample_every < 1:
            raise ValueError("sample_every must be an integer >= 1")
        if not isinstance(self.steps_factor, (int, float)) or self.steps_factor <= 0:
            raise ValueError("steps_factor must be positive")
        if self.plastic_rate < 0 or not math.isfinite(self.plastic_rate):
            raise ValueError("plastic_rate must be finite and nonnegative")
        if self.pipe_line_mass <= 0 or not math.isfinite(self.pipe_line_mass):
            raise ValueError("pipe_line_mass must be finite and positive")
        if not 0.0 <= self.pipe_damping_ratio < 1.0:
            raise ValueError("pipe_damping_ratio must lie in [0, 1)")

    @property
    def pipe_centre_height(self) -> float:
        """Pipe centre elevation relative to the mudline (negative = buried)."""
        return -(self.cover_depth + 0.5 * self.section.outer_diameter)

    @property
    def dt_used(self) -> float:
        base = self.material.recommended_dt
        return base if self.dt is None else self.dt


@dataclass
class GroundingResult:
    """Container for one grounding run."""

    scenario: GroundingScenario
    history: dict
    info: dict
    capacity: float
    dt: float
    steps: int
    final_pipe_displacement: float

    def summary(self) -> dict:
        """Compact, JSON-friendly view of the run."""
        return {
            "capacity_per_length": self.capacity,
            "dt": self.dt,
            "steps": self.steps,
            "final_pipe_displacement": self.final_pipe_displacement,
            "info": self.info,
        }


def _build_assembly(scenario: GroundingScenario) -> tuple[RidgeAssembly, dict]:
    generated = generate_keel_assembly(scenario.keel, scenario.material,
                                      tip_height=scenario.tip_clearance)
    assembly = generated["assembly"]
    assembly.soil = scenario.soil
    assembly.mudline = 0.0
    centre = torch.tensor((0.0, scenario.pipe_centre_height), dtype=torch.float64)
    cylinder = {"centre": centre, "radius": 0.5 * scenario.section.outer_diameter,
                "velocity": torch.zeros(2, dtype=torch.float64)}
    assembly.cylinders = [cylinder]
    if scenario.loading == "vertical":
        platen_velocity = torch.tensor((0.0, -scenario.platen_velocity),
                                       dtype=torch.float64)
    else:
        # lateral loading keeps the ridge at mouldline level and pushes it along
        # +x into the pipeline
        platen_velocity = torch.tensor((scenario.platen_velocity, 0.0),
                                       dtype=torch.float64)
        assembly.driven = assembly.driven & (assembly.pos[:, 1] >= 0.0)
    assembly.set_platen_velocity(platen_velocity)
    return assembly, generated["info"] | {"cylinder": cylinder}


def _moving_average(values: torch.Tensor, window: int) -> torch.Tensor:
    """Centred moving average that preserves the record length."""
    if window <= 1 or values.numel() < 3:
        return values.clone()
    kernel = torch.ones(window, dtype=torch.float64) / window
    left = window // 2
    right = window - 1 - left
    padded = torch.nn.functional.pad(values.reshape(1, 1, -1), (left, right),
                                     mode="replicate")
    return torch.nn.functional.conv1d(padded, kernel.reshape(1, 1, -1)).reshape(-1)


def run_grounding(scenario: GroundingScenario, *, max_steps: int = 40000) -> GroundingResult:
    """Integrate one grounding run and return the load-displacement record."""
    if not isinstance(scenario, GroundingScenario):
        raise TypeError("scenario must be a GroundingScenario")
    assembly, info = _build_assembly(scenario)
    dt = scenario.dt_used
    steps = int(round(scenario.stroke / (scenario.platen_velocity * dt)
                      * scenario.steps_factor))
    steps = max(1, min(steps, max_steps))
    cylinder = assembly.cylinders[0]
    spring = (SofteningVerticalSpring(scenario.soil, scenario.section.outer_diameter,
                                      scenario.cover_depth)
              if scenario.pipe_response == "compliant" else None)

    history = {
        "time": [], "stroke": [], "pipe_load": [], "pipe_load_horizontal": [],
        "pipe_displacement": [], "drive_force": [], "soil_force": [],
        "keel_penetration": [], "broken_bonds": [], "contacts": [],
        "kinetic_energy": [], "work": [], "driven_drop": [], "pipe_load_soil": [],
    }
    initial_top = float(assembly.pos[assembly.driven, 1].mean())
    pipe_displacement = 0.0
    pipe_velocity = 0.0
    smoothed_load = 0.0
    tip_start = float(assembly.pos[:, 1].min())

    for step in range(steps):
        out = assembly.step(dt)
        load = -float(out["pipe_force"][1])
        load_h = float(out["pipe_force"][0])
        smoothed_load += 0.1 * (load - smoothed_load)

        if scenario.pipe_response == "compliant":
            stiffness = scenario.soil.initial_vertical_stiffness(
                scenario.section.outer_diameter)
            reaction = spring.reaction(pipe_displacement)
            damping = (2.0 * scenario.pipe_damping_ratio
                       * math.sqrt(stiffness * scenario.pipe_line_mass))
            acceleration = (load - reaction - damping * pipe_velocity) \
                / scenario.pipe_line_mass
            pipe_velocity += acceleration * dt
            pipe_displacement = max(pipe_displacement + pipe_velocity * dt, 0.0)
            if pipe_displacement <= 0.0:
                pipe_velocity = max(pipe_velocity, 0.0)
            centre = torch.tensor((0.0, scenario.pipe_centre_height - pipe_displacement),
                                  dtype=torch.float64)
            cylinder["centre"] = centre
            assembly.cylinders[0] = cylinder

        if step % scenario.sample_every == 0 or step == steps - 1:
            driven_drop = initial_top - float(assembly.pos[assembly.driven, 1].mean())
            history["time"].append(assembly.time)
            history["stroke"].append(driven_drop)
            history["pipe_load"].append(load)
            history["pipe_load_horizontal"].append(load_h)
            history["pipe_displacement"].append(pipe_displacement)
            history["drive_force"].append(-float(out["platen_force"][1])
                                          if scenario.loading == "vertical"
                                          else float(out["platen_force"][0]))
            history["soil_force"].append(float(out["soil_force"][1]))
            history["keel_penetration"].append(tip_start - float(assembly.pos[:, 1].min()))
            history["broken_bonds"].append(assembly.broken_bonds)
            history["contacts"].append(out["contacts"] + out["pipe_contacts"])
            history["kinetic_energy"].append(assembly.kinetic_energy())
            history["pipe_load_soil"].append(spring.reaction(pipe_displacement)
                                            if spring is not None else 0.0)
            history["work"].append(float(abs(out["platen_force"][1])
                                         * scenario.platen_velocity * assembly.time))
            history["driven_drop"].append(driven_drop)

    loads = torch.tensor(history["pipe_load"], dtype=torch.float64)
    window = min(25, max(1, loads.numel() // 8))
    smooth = _moving_average(loads, window)
    history["pipe_load_smoothed"] = smooth.tolist()
    capacity = float(smooth.max()) if smooth.numel() else 0.0
    info["particles"] = assembly.count
    info["bonds_final"] = int(assembly.bond_alive.sum())
    info["broken_bonds"] = assembly.broken_bonds
    info["pipe_load_peak_raw"] = float(loads.max()) if loads.numel() else 0.0
    info["smoothing_window"] = window
    if spring is not None:
        info["soil_spring"] = spring.describe()
    return GroundingResult(scenario=scenario, history=history, info=info,
                           capacity=capacity, dt=dt, steps=steps,
                           final_pipe_displacement=pipe_displacement)


def load_displacement_metrics(result: GroundingResult, *,
                              displacement_key: str = "keel_penetration",
                              softening_threshold: float = 0.10,
                              peak_window: float = 0.90) -> dict:
    """Scalar metrics and peak-then-softening checks of a load-displacement curve.

    The ISO 19906-style signature of an ice load-displacement curve is a rise to
    a peak followed by softening (not brittle collapse to zero, not monotone
    hardening).  The machine checks here are:

    * ``peak_load > 0``;
    * the peak is reached strictly before ``peak_window`` of the record (so
      there *is* a post-peak branch to measure);
    * post-peak softening ``1 - mean(last 20%)/peak`` exceeds
      ``softening_threshold``;
    * the pre-peak branch is predominantly rising (>= 60% of the increments).
    """
    loads = torch.tensor(result.history["pipe_load_smoothed"], dtype=torch.float64)
    displacement = torch.tensor(result.history[displacement_key], dtype=torch.float64)
    count = int(loads.numel())
    if count < 5:
        raise ValueError("record too short for load-displacement metrics")
    peak_index = int(torch.argmax(loads))
    peak = float(loads[peak_index])
    tail_start = max(int(0.8 * count), peak_index + 1)
    tail = loads[tail_start:]
    residual = float(tail.mean()) if tail.numel() else 0.0
    softening = 1.0 - residual / peak if peak > 0 else 0.0
    increments = torch.diff(loads[:peak_index + 1])
    rising_fraction = float((increments > 0).to(torch.float64).mean()) if increments.numel() else 0.0
    displacement_peak = float(displacement[peak_index])
    displacement_end = float(displacement[-1])
    checks = {
        "peak_positive": peak > 0.0,
        "peak_before_end": peak_index < int(peak_window * count),
        "post_peak_softening": softening >= softening_threshold,
        "rising_branch": rising_fraction >= 0.6,
        "finite_and_nonnegative": bool(torch.isfinite(loads).all()) and peak >= 0.0,
    }
    response = float(torch.tensor(result.history["pipe_displacement"]).max()) \
        if result.history["pipe_displacement"] else 0.0
    return {
        "peak_load_per_length": peak,
        "peak_index": peak_index,
        "samples": count,
        "displacement_at_peak": displacement_peak,
        "displacement_final": displacement_end,
        "residual_load_per_length": residual,
        "softening_ratio": softening,
        "rising_increment_fraction": rising_fraction,
        "initial_secant_stiffness": peak / displacement_peak if displacement_peak > 0 else math.inf,
        "pipe_displacement_max": response,
        "checks": checks,
        "passed": all(checks.values()),
    }


def axial_footprint_loads(total_load: float, *, contact_length: float,
                          node_count: int = 241, length: float | None = None,
                          shape: str = "uniform") -> tuple[torch.Tensor, torch.Tensor]:
    """Longitudinal line load (N/m) that carries ``total_load`` over a footprint.

    The DEM gives the load per metre of ridge; the pipeline samples that load
    over the axial length where the keel is deeper than the pipe crown.  The
    axial distribution is an explicit model choice: ``"uniform"`` (conservative
    for bending moment) or ``"cosine"`` (a smoothed chord of a keel section).
    ``total_load`` is in N per metre of ridge, so the integrated footprint load
    equals ``total_load`` times the ridge length, i.e. one metre of ridge
    footprint -- the returned line load is normalised to ``total_load``.
    """
    if not math.isfinite(total_load) or total_load < 0.0:
        raise ValueError("total_load must be finite and nonnegative")
    if not math.isfinite(contact_length) or contact_length <= 0.0:
        raise ValueError("contact_length must be finite and positive")
    if not isinstance(node_count, int) or node_count < 11:
        raise ValueError("node_count must be an integer >= 11")
    if shape not in ("uniform", "cosine"):
        raise ValueError("shape must be 'uniform' or 'cosine'")
    if length is None:
        length = 6.0 * contact_length
    x = torch.linspace(0.0, length, node_count, dtype=torch.float64)
    centre = 0.5 * length
    inside = (x - centre).abs() <= 0.5 * contact_length
    if shape == "uniform":
        weights = inside.to(torch.float64)
    else:
        phase = (x - centre) / (0.5 * contact_length)
        weights = torch.where(inside, torch.cos(0.5 * math.pi * phase) ** 2,
                              torch.zeros_like(x))
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise ValueError("footprint does not overlap the beam")
    spacing = float(x[1] - x[0])
    # q(z) is scaled so that the integral of the line load over the footprint
    # equals the DEM load per metre of ridge
    loads = weights * (total_load / (total_weight * spacing))
    return x, loads


def pipe_stress_profile(total_load: float, scenario: GroundingScenario, *,
                        node_count: int = 241, length: float | None = None,
                        shape: str = "uniform", beam=None) -> dict:
    """Beam-on-Winkler response of the pipe to a DEM-derived footprint load.

    Returns the finite-element deflection, bending moment, bending stress,
    DNV-OS-F101 utilisation and the comparison against the closed-form Hetenyi
    infinite-beam solution for the same uniform patch -- the independent check
    behind criterion (c) of the campaign.
    """
    section = scenario.section
    x, loads = axial_footprint_loads(total_load, contact_length=scenario.contact_length,
                                     node_count=node_count, length=length, shape=shape)
    subgrade = scenario.soil.subgrade_modulus(section.outer_diameter) * section.outer_diameter
    solution = beam if beam is not None else solve_beam_on_winkler(
        loads, inertia=section.inertia, young_modulus=section.young_modulus,
        length=float(x[-1]), node_count=node_count, foundation_modulus=subgrade,
        end_condition="infinite")
    element_length = float(x[-1]) / (node_count - 1)
    patch_load = float(loads.max())
    beta = (subgrade / (4.0 * section.young_modulus * section.inertia)) ** 0.25
    reference_deflection, reference_moment = hetenyi_infinite_beam_patch_load(
        solution["x"] - 0.5 * float(x[-1]), load_per_length=patch_load,
        half_length=0.5 * scenario.contact_length, beta=beta, k=subgrade)
    moment = solution["moment"]
    stress = moment / section.elastic_section_modulus
    peak_moment = float(moment.max())
    peak_moment_reference = float(reference_moment.max())
    return {
        "x": solution["x"].tolist(),
        "load": loads.tolist(),
        "deflection": solution["deflection"].tolist(),
        "moment": moment.tolist(),
        "stress": stress.tolist(),
        "reaction": solution["reaction"].tolist(),
        "peak_deflection": float(solution["deflection"].max()),
        "peak_deflection_reference": float(reference_deflection.max()),
        "peak_moment": peak_moment,
        "peak_moment_reference": peak_moment_reference,
        "moment_relative_error": abs(peak_moment - peak_moment_reference)
        / abs(peak_moment_reference) if peak_moment_reference else math.inf,
        "deflection_relative_error": abs(float(solution["deflection"].max())
                                        - float(reference_deflection.max()))
        / abs(float(reference_deflection.max())) if float(reference_deflection.max()) else math.inf,
        "peak_stress": float(stress.max()),
        "peak_utilization": dnv_os_f101_utilization(
            peak_moment, float(scenario.section.plastic_axial_force) * 0.0, section),
        "beta": beta,
        "subgrade_stiffness": subgrade,
        "element_length": element_length,
        "iterations": solution["iterations"],
        "converged": solution["converged"],
    }


def continuum_limit_capacity(cohesion: float, friction_angle_deg: float, *,
                             surcharge: float, diameter: float) -> float:
    """Plastic limit load per unit length of a cylinder pressed into the rubble.

    Plane-strain bearing-capacity limit equilibrium of a rough cylinder of
    diameter ``diameter`` in a Mohr-Coulomb medium (Vesic factors) driven by a
    confining/surcharge pressure ``surcharge``::

        q_ult = ( c' N_c(phi) + surcharge N_q(phi) ) * diameter

    ``surcharge`` is the pressure the driving ice exerts on the keel, which is
    what confines the rubble around the pipe; without confinement a purely
    frictional assembly has no vertical resistance.  ``(c', phi)`` come from
    :func:`tensordem.ridge_dem.calibrate_rubble_strength`, so the comparison in
    the campaign report is DEM (particle level) against an independent
    continuum limit analysis using *measured* micro-to-macro strength.
    """
    from .ridge_geotech import bearing_capacity_factors

    if not math.isfinite(cohesion) or cohesion < 0.0:
        raise ValueError("cohesion must be finite and nonnegative")
    if not 0.0 <= friction_angle_deg < 90.0:
        raise ValueError("friction_angle_deg must lie in [0, 90)")
    if not math.isfinite(surcharge) or surcharge < 0.0:
        raise ValueError("surcharge must be finite and nonnegative")
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("diameter must be finite and positive")
    n_c, n_q, _n_gamma = bearing_capacity_factors(math.radians(friction_angle_deg))
    return (cohesion * n_c + surcharge * n_q) * diameter


def capacity_envelope(keel_depths, ice_capacities, *, soil: MohrCoulombSoil,
                      section: PipeSection, cover_depth: float,
                      contact_length: float) -> dict:
    """Ultimate-capacity envelope: ice-limited vs soil-limited vs pipe strength.

    ``ice_capacities`` are the DEM peak loads per metre of ridge at each
    ``keel_depths`` entry.  The envelope takes the smaller of the ice-limited
    and soil-limited (subgrade bearing) capacities, expresses it as a total load
    over the axial contact length, and reports the DNV-OS-F101 utilisation of
    the resulting pipe bending moment.
    """
    if len(keel_depths) != len(ice_capacities):
        raise ValueError("keel_depths and ice_capacities must have equal length")
    soil_capacity = soil.vertical_capacity_per_length(section.outer_diameter, cover_depth)
    rows = []
    for depth, ice in zip(keel_depths, ice_capacities):
        governing = min(float(ice), soil_capacity)
        total = governing * contact_length
        profile = pipe_stress_profile(total, GroundingScenario(
            keel=RidgeKeel(float(depth), max(3.0 * float(depth), 6.0)),
            material=_default_material(), soil=soil, section=section,
            cover_depth=cover_depth, contact_length=contact_length))
        rows.append({
            "keel_depth": float(depth),
            "ice_capacity_per_length": float(ice),
            "soil_capacity_per_length": soil_capacity,
            "governing_capacity_per_length": governing,
            "governing_mechanism": "ice" if float(ice) <= soil_capacity else "soil",
            "total_load": total,
            "peak_moment": profile["peak_moment"],
            "peak_stress": profile["peak_stress"],
            "peak_utilization": profile["peak_utilization"],
            "acceptable": profile["peak_utilization"] <= 1.0,
        })
    return {"soil_capacity_per_length": soil_capacity, "rows": rows}


def _default_material() -> RidgeMaterial:
    return RidgeMaterial()
