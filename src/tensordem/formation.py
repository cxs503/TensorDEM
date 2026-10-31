"""Multi-ship formation ice-channel mission framework.

Couples the three physics layers into one engineering workflow:

1. :mod:`tensordem.formation_thermal` — refreezing of the channel cleared by the
   lead ship (one-phase Stefan), which sets the ice thickness a follower meets;
2. :mod:`tensordem.formation_resistance` — ship resistance over a transverse ice
   thickness profile (channel centre / ridge edge / parent sheet), plus the
   ITTC-57 open-water resistance of the same hull;
3. :mod:`tensordem.formation_hydro` — hydrodynamic interference coefficients
   ``C_ij(d, theta)`` of the formation in a channel.

For every pass of the mission and every ship the framework reports the
encountered ice state, the resistance decomposition, the interference row, and
the energy spent.  A mission optimiser sweeps spacing / lateral offset / speed
under an operational minimum-spacing constraint and a mission-duration limit.

Modelling statements that matter for interpretation
--------------------------------------------------
* Hydrodynamic interference multiplies only the *open-water* part of the
  resistance.  Breaking ice is a contact/fracture process that the neighbour ships
  do not change, so ``C_ij`` is applied to ``R_ow`` and never to ``R_ice``.  In
  level ice this makes the hydrodynamic interference a small correction and the
  channel effect dominant — that split is reported explicitly for every case.
* The channel contains refrozen brash, not clean open water: its resistance is
  represented by an equivalent thickness
  ``h_frozen + brash_equivalent_fraction * h_parent`` with a reduced flexural
  strength (:class:`~tensordem.formation_resistance.IceSheet.brash_strength_factor`).
* Weather / current / manoeuvring transients are outside the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .formation_hydro import (FormationFlow, FormationFlowConfig, HullWaterline,
                              froude_number)
from .formation_resistance import (GRAVITY, IcebreakerShip, IceResistanceModel,
                                   IceSheet, level_ice_resistance, strip_resistance)
from .formation_thermal import (ChannelEncounterConfig, ChannelRefreezing,
                                ChannelThermalConfig, channel_thickness_profile)

KINEMATIC_VISCOSITY = 1.83e-6  # m^2/s, sea water at 0 C
DEFAULT_MIN_SPACING_FACTOR = 1.5


def _number(value: float, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class WakeInterferenceConfig:
    """Empirical wake-deficit ("drafting") interference term.

    A turbulent far wake decays as ``x^(-2/3)`` and is Gaussian in the transverse
    direction, so the centreline deficit of ship ``j`` at distance ``dx`` downstream
    and ``dy`` off the track is modelled as

        ``w(dx, dy) = w0 (1 + dx / x0)^(-2/3) exp(-dy^2 / (2 bw^2))``

    with ``w0`` the near-field deficit, ``x0 = L`` (one ship length downstream) and
    ``bw = 0.5 B``.  The contribution to the interference coefficient is
    ``-beta * w`` (slower inflow, less resistance).  These constants are
    engineering values, not calibrated against formation trials; the campaign
    therefore reports a sensitivity range for ``w0``.
    """

    deficit: float = 0.0
    decay_length_factor: float = 1.0
    half_width_factor: float = 0.5
    resistance_exponent: float = 2.0

    def __post_init__(self) -> None:
        for name in ("deficit", "decay_length_factor", "half_width_factor"):
            _number(getattr(self, name), name, nonnegative=True)
        _number(self.resistance_exponent, "resistance_exponent", positive=True)


@dataclass(frozen=True)
class FormationScenario:
    """Complete description of a formation mission through a level-ice field."""

    ships: Sequence[IcebreakerShip]
    speed: float
    parent_thickness: float
    ice: IceSheet = IceSheet()
    resistance_model: IceResistanceModel = IceResistanceModel()
    spacing_factor: float = 2.0
    lateral_factor: float = 0.0
    layout: str = "column"
    channel_width_factor: float = 1.15
    passes: int = 1
    pass_interval_s: float = 3.6e3
    reuse_previous_channel: bool = False
    initial_channel_age_s: float = 0.0
    transit_length: float = 2.0e4
    thermal: ChannelThermalConfig = ChannelThermalConfig()
    encounter: ChannelEncounterConfig = ChannelEncounterConfig()
    wake: WakeInterferenceConfig = WakeInterferenceConfig()
    channel_half_width_factor: float = 1.6
    include_confinement: bool = True
    form_factor: float = 1.15
    correlation_allowance: float = 4.0e-4
    time_samples: int = 25

    def __post_init__(self) -> None:
        ships = tuple(self.ships)
        if not ships:
            raise ValueError("a formation needs at least one ship")
        object.__setattr__(self, "ships", ships)
        _number(self.speed, "speed", positive=True)
        _number(self.parent_thickness, "parent_thickness", positive=True)
        for name in ("spacing_factor", "channel_width_factor", "channel_half_width_factor"):
            _number(getattr(self, name), name, positive=True)
        _number(self.lateral_factor, "lateral_factor", nonnegative=True)
        if not isinstance(self.include_confinement, bool):
            raise ValueError("include_confinement must be a boolean")
        if self.layout not in ("column", "staggered"):
            raise ValueError("layout must be 'column' or 'staggered'")
        if isinstance(self.passes, bool) or not isinstance(self.passes, int) or self.passes < 1:
            raise ValueError("passes must be a positive integer")
        _number(self.pass_interval_s, "pass_interval_s", nonnegative=True)
        _number(self.initial_channel_age_s, "initial_channel_age_s", nonnegative=True)
        _number(self.transit_length, "transit_length", positive=True)
        _number(self.form_factor, "form_factor", positive=True)
        _number(self.correlation_allowance, "correlation_allowance", nonnegative=True)
        if isinstance(self.time_samples, bool) or not isinstance(self.time_samples, int) \
                or self.time_samples < 2:
            raise ValueError("time_samples must be an integer >= 2")
        for name in ("passes", "pass_interval_s", "time_samples"):
            pass
        if self.passes > 1 and self.pass_interval_s <= 0.0:
            raise ValueError("a multi-pass mission needs a positive pass_interval_s")

    # -- geometry --------------------------------------------------------------
    @property
    def leader(self) -> IcebreakerShip:
        return self.ships[0]

    def offsets(self) -> list[tuple[float, float]]:
        """Longitudinal/transverse offsets of every ship from the leader (m)."""
        offsets: list[tuple[float, float]] = []
        for index, ship in enumerate(self.ships):
            dx = index * self.spacing_factor * self.leader.length
            sign = 1.0 if index % 2 == 0 else -1.0
            dy = 0.0 if self.layout == "column" else sign * self.lateral_factor * self.leader.beam
            offsets.append((dx, dy))
        return offsets

    def spacings(self) -> list[float]:
        """Centre-to-centre separation between consecutive ships (m)."""
        offsets = self.offsets()
        gaps = []
        for index in range(1, len(offsets)):
            dx = offsets[index][0] - offsets[index - 1][0]
            dy = offsets[index][1] - offsets[index - 1][1]
            gaps.append(math.hypot(dx, dy))
        return gaps

    @property
    def cleared_channel_width(self) -> float:
        return self.channel_width_factor * self.leader.beam

    @property
    def channel_half_width(self) -> float:
        return self.channel_half_width_factor * self.leader.beam

    # -- component builders ----------------------------------------------------
    def flow(self) -> FormationFlow:
        waterlines = [HullWaterline(length=ship.length, beam=ship.beam) for ship in self.ships]
        config = FormationFlowConfig(
            speed=self.speed,
            channel_half_width=(self.channel_half_width if self.include_confinement else None))
        return FormationFlow(waterlines, config)

    def thermal_model(self) -> ChannelRefreezing:
        return ChannelRefreezing(self.thermal)

    def validate(self) -> dict[str, Any]:
        """Pre-flight checks: hull geometry, spacing and channel fit."""
        reports = [ship.geometry_report() for ship in self.ships]
        spacings = self.spacings()
        minimum = min(spacings) if spacings else math.inf
        return {
            "ship_count": len(self.ships),
            "ships": reports,
            "spacings_m": spacings,
            "minimum_spacing_over_length": (minimum / self.leader.length) if spacings else None,
            "cleared_channel_width_m": self.cleared_channel_width,
            "channel_fits_beam": self.cleared_channel_width >= self.leader.beam,
            "froude_number": froude_number(self.speed, self.leader.length),
            "wave_interference_negligible": froude_number(self.speed, self.leader.length) <= 0.15,
        }


# ---------------------------------------------------------------------------
# Resistance building blocks
# ---------------------------------------------------------------------------
def open_water_resistance(ship: IcebreakerShip, speed: float, *, form_factor: float = 1.15,
                          correlation_allowance: float = 4.0e-4,
                          water_density: float = 1025.0) -> dict[str, float]:
    """ITTC-57 calm-water resistance with a Holtrop-type wetted-surface estimate."""
    speed = _number(speed, "speed", positive=True)
    reynolds = speed * ship.length / KINEMATIC_VISCOSITY
    cf = 0.075 / (math.log10(reynolds) - 2.0) ** 2
    wetted = (ship.length * (2.0 * ship.draft + ship.beam)
              * math.sqrt(ship.block_coefficient))
    dynamic = 0.5 * water_density * speed * speed * wetted
    viscous = dynamic * cf * form_factor
    allowance = dynamic * correlation_allowance
    return {
        "wetted_surface_m2": wetted,
        "reynolds": reynolds,
        "friction_coefficient": cf,
        "viscous_resistance_N": viscous,
        "correlation_allowance_N": allowance,
        "total_resistance_N": viscous + allowance,
    }


def wake_interference(offsets: Sequence[tuple[float, float]], ships: Sequence[IcebreakerShip],
                      config: WakeInterferenceConfig) -> list[list[float]]:
    """Empirical wake-deficit interference matrix (see :class:`WakeInterferenceConfig`)."""
    count = len(offsets)
    matrix = [[0.0] * count for _ in range(count)]
    if config.deficit == 0.0:
        return matrix
    for target in range(count):
        for source in range(count):
            if source == target:
                continue
            dx = offsets[target][0] - offsets[source][0]
            dy = offsets[target][1] - offsets[source][1]
            if dx <= 0.0:
                continue
            length = ships[source].length
            x0 = config.decay_length_factor * length
            bw = config.half_width_factor * ships[source].beam
            deficit = config.deficit * (1.0 + dx / x0) ** (-2.0 / 3.0)
            deficit *= math.exp(-0.5 * (dy / bw) ** 2)
            matrix[target][source] = -config.resistance_exponent * deficit
    return matrix


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _channel_state(scenario: FormationScenario, model: ChannelRefreezing,
                   channel_age_s: float) -> dict[str, float]:
    refrozen = model.thickness(channel_age_s)
    width = model.channel_width(channel_age_s, scenario.cleared_channel_width)
    return {
        "channel_age_s": channel_age_s,
        "refrozen_thickness_m": refrozen,
        "open_width_m": width,
        "lateral_growth_m": model.lateral_growth(channel_age_s),
        "stefan_reference_m": model.stefan_analytic(channel_age_s),
        "neumann_reference_m": model.neumann_analytic(channel_age_s),
    }


def _parent_state(scenario: FormationScenario) -> dict[str, float]:
    """Ice state of the undisturbed parent sheet (the lead breaker sees this)."""
    return {
        "channel_age_s": 0.0,
        "refrozen_thickness_m": scenario.parent_thickness,
        "open_width_m": 0.0,
        "lateral_growth_m": 0.0,
        "stefan_reference_m": 0.0,
        "neumann_reference_m": 0.0,
    }


def _ship_encounter(scenario: FormationScenario, ship: IcebreakerShip, lateral_offset: float,
                    state: dict[str, float], *, breaking_parent: bool = False) -> dict[str, Any]:
    positions = [lateral_offset + centre for centre in ship.strip_centres()]
    if breaking_parent:
        thicknesses = [scenario.parent_thickness] * ship.strips
        brash = [False] * ship.strips
        resistance = strip_resistance(ship, scenario.resistance_model, thicknesses,
                                      brash_flags=brash)
        return {
            "strip_positions_m": positions,
            "strip_thicknesses_m": thicknesses,
            "strip_brash": brash,
            "resistance": resistance,
            "channel_strip_fraction": 0.0,
        }
    thicknesses = channel_thickness_profile(
        positions,
        parent_thickness=scenario.parent_thickness,
        channel_width=state["open_width_m"],
        refrozen_thickness=state["refrozen_thickness_m"],
        beam=ship.beam,
        config=scenario.encounter,
    )
    half = 0.5 * state["open_width_m"]
    brash = [abs(position) <= half for position in positions]
    resistance = strip_resistance(ship, scenario.resistance_model, thicknesses,
                                  brash_flags=brash)
    return {
        "strip_positions_m": positions,
        "strip_thicknesses_m": thicknesses,
        "strip_brash": brash,
        "resistance": resistance,
        "channel_strip_fraction": sum(brash) / len(brash),
    }


def simulate_formation(scenario: FormationScenario) -> dict[str, Any]:
    """Run the formation mission and return all reported quantities."""
    validation = scenario.validate()
    offsets = scenario.offsets()
    model = scenario.thermal_model()
    flow = scenario.flow()
    hydrodynamic = flow.interference(offsets)
    potential = hydrodynamic["coefficients"]
    wake = wake_interference(offsets, scenario.ships, scenario.wake)
    count = len(scenario.ships)

    pass_rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float]]] = {ship.name: [] for ship in scenario.ships}
    total_energy = 0.0
    for pass_index in range(scenario.passes):
        start_time = pass_index * scenario.pass_interval_s
        reuse = (scenario.reuse_previous_channel
                 and (pass_index > 0 or scenario.initial_channel_age_s > 0.0))
        ships_rows: list[dict[str, Any]] = []
        pass_energy = 0.0
        for index, ship in enumerate(scenario.ships):
            dx, dy = offsets[index]
            if reuse:
                channel_age = (scenario.initial_channel_age_s
                               + pass_index * scenario.pass_interval_s + dx / scenario.speed)
                channel_origin = "previous_pass"
            elif index == 0:
                channel_age = 0.0
                channel_origin = "lead_breaking"
            else:
                # The follower reaches each point dx / U after the leader cleared it.
                channel_age = dx / scenario.speed
                channel_origin = "lead_track"
            breaking_parent = (index == 0 and not reuse)
            state = (_parent_state(scenario) if breaking_parent
                     else _channel_state(scenario, model, channel_age))
            encounter = _ship_encounter(scenario, ship, dy, state,
                                        breaking_parent=breaking_parent)
            ice = encounter["resistance"]
            ow = open_water_resistance(ship, scenario.speed,
                                       form_factor=scenario.form_factor,
                                       correlation_allowance=scenario.correlation_allowance)
            interference_sum = float(sum(potential[index, j].item() for j in range(count) if j != index)
                                    + sum(wake[index][j] for j in range(count) if j != index))
            confinement = float(hydrodynamic["confinement"][index].item())
            ow_factor = 1.0 + interference_sum + confinement
            ow_effective = ow["total_resistance_N"] * (ow_factor if scenario.include_confinement
                                                       else 1.0 + interference_sum)
            total_resistance = ice["total_resistance_N"] + ow_effective
            transit_time = scenario.transit_length / scenario.speed
            entry_time = start_time + dx / scenario.speed
            energy = total_resistance * scenario.transit_length
            pass_energy += energy
            ships_rows.append({
                "name": ship.name,
                "index": index,
                "dx_m": dx,
                "dy_m": dy,
                "channel_origin": channel_origin,
                "channel_state": state,
                "strip_positions_m": encounter["strip_positions_m"],
                "strip_thicknesses_m": encounter["strip_thicknesses_m"],
                "strip_brash": encounter["strip_brash"],
                "channel_strip_fraction": encounter["channel_strip_fraction"],
                "mean_encountered_thickness_m": ice["mean_thickness_m"],
                "ice_resistance_N": ice["total_resistance_N"],
                "ice_breaking_N": ice["breaking_resistance_N"],
                "ice_submergence_N": ice["submergence_resistance_N"],
                "ice_friction_N": ice["friction_resistance_N"],
                "open_water_resistance_N": ow["total_resistance_N"],
                "interference_sum": interference_sum,
                "confinement_coefficient": confinement,
                "open_water_factor": ow_factor,
                "open_water_effective_N": ow_effective,
                "total_resistance_N": total_resistance,
                "entry_time_s": entry_time,
                "transit_time_s": transit_time,
                "energy_J": energy,
                "froude_number": froude_number(scenario.speed, ship.length),
            })
            # Time history of this ship's passage (position-resolved ice state).
            samples = scenario.time_samples
            for step in range(samples):
                fraction = step / (samples - 1)
                time_s = start_time + dx / scenario.speed + fraction * transit_time
                histories[ship.name].append({
                    "time_s": time_s,
                    "pass_index": float(pass_index),
                    "position_fraction": fraction,
                    "channel_age_at_position_s": channel_age,
                    "resistance_N": total_resistance,
                })
        total_energy += pass_energy
        pass_rows.append({
            "pass_index": pass_index,
            "start_time_s": start_time,
            "channel_reused": reuse,
            "ships": ships_rows,
            "energy_J": pass_energy,
        })

    # Interference report (potential + empirical wake term).
    c_total = [[float(potential[i, j]) + wake[i][j] for j in range(count)] for i in range(count)]
    interference_report = {
        "potential": [[float(potential[i, j]) for j in range(count)] for i in range(count)],
        "wake": wake,
        "total": c_total,
        "confinement": [float(value) for value in hydrodynamic["confinement"]],
        "clamped": bool(hydrodynamic["clamped"]),
        "neumann_residual": float(hydrodynamic["neumann_residual"]),
        "hull_panel_count": int(hydrodynamic["hull_panel_count"]),
        "wall_panel_count": int(hydrodynamic["wall_panel_count"]),
        "row_sums": [sum(row[j] for j in range(count) if j != i) for i, row in enumerate(c_total)],
    }

    leader_level_ice = level_ice_resistance(scenario.leader, scenario.resistance_model,
                                            scenario.parent_thickness)
    summary = _summarise(scenario, pass_rows, total_energy, leader_level_ice)
    return {
        "scenario": _scenario_report(scenario),
        "validation": validation,
        "interference": interference_report,
        "passes": pass_rows,
        "time_histories": histories,
        "summary": summary,
        "single_ship_baseline": {
            "parent_thickness_m": scenario.parent_thickness,
            "resistance_N": leader_level_ice["total_resistance_N"],
            "breaking_N": leader_level_ice["breaking_resistance_N"],
            "submergence_N": leader_level_ice["submergence_resistance_N"],
            "friction_N": leader_level_ice["friction_resistance_N"],
        },
    }


def _summarise(scenario: FormationScenario, passes: Sequence[dict[str, Any]],
               total_energy: float, leader_level_ice: dict[str, Any]) -> dict[str, Any]:
    first = passes[0]["ships"]
    leader = first[0]
    followers = first[1:]
    mean_follower = (sum(row["total_resistance_N"] for row in followers) / len(followers)
                     if followers else leader["total_resistance_N"])
    follower_reduction = 1.0 - mean_follower / leader_level_ice["total_resistance_N"]
    baseline_total = leader_level_ice["total_resistance_N"]
    channel_gain = baseline_total - mean_follower if followers else 0.0
    transit = scenario.transit_length / scenario.speed
    mission_end = (scenario.passes - 1) * scenario.pass_interval_s + transit \
        + (scenario.offsets()[-1][0] / scenario.speed)
    hydrodynamic_share = 0.0
    if followers:
        differences = [abs(row["open_water_effective_N"] - row["open_water_resistance_N"])
                       for row in followers]
        hydrodynamic_share = max(differences) / max(mean_follower, 1.0)
    return {
        "ships": len(scenario.ships),
        "speed_m_s": scenario.speed,
        "parent_thickness_m": scenario.parent_thickness,
        "spacings_m": scenario.spacings(),
        "total_energy_J": total_energy,
        "total_fuel_energy_GJ": total_energy / 1.0e9,
        "mission_duration_s": mission_end,
        "mission_duration_h": mission_end / 3600.0,
        "leader_ice_resistance_N": leader["ice_resistance_N"],
        "mean_follower_resistance_N": mean_follower,
        "follower_over_leader_ratio": mean_follower / max(leader["ice_resistance_N"], 1.0),
        "follower_saving_vs_level_ice": follower_reduction,
        "leader_vs_level_ice_error_percent": (leader["ice_resistance_N"]
                                              / baseline_total - 1.0) * 100.0,
        "channel_gain_N": channel_gain,
        "follower_refreezing_spread_N": (
            max(row["total_resistance_N"] for row in followers)
            - min(row["total_resistance_N"] for row in followers)) if followers else 0.0,
        "hydrodynamic_interference_share_of_follower": hydrodynamic_share,
        "leader_level_ice_resistance_N": leader_level_ice["total_resistance_N"],
    }


def _scenario_report(scenario: FormationScenario) -> dict[str, Any]:
    return {
        "speed_m_s": scenario.speed,
        "parent_thickness_m": scenario.parent_thickness,
        "ice": {
            "flexural_strength_Pa": scenario.ice.flexural_strength,
            "elastic_modulus_Pa": scenario.ice.elastic_modulus,
            "friction": scenario.ice.friction,
            "brash_strength_factor": scenario.ice.brash_strength_factor,
        },
        "resistance_scale_factor": scenario.resistance_model.scale_factor,
        "layout": scenario.layout,
        "spacing_factor": scenario.spacing_factor,
        "lateral_factor": scenario.lateral_factor,
        "passes": scenario.passes,
        "pass_interval_s": scenario.pass_interval_s,
        "reuse_previous_channel": scenario.reuse_previous_channel,
        "initial_channel_age_s": scenario.initial_channel_age_s,
        "transit_length_m": scenario.transit_length,
        "cleared_channel_width_m": scenario.cleared_channel_width,
        "channel_half_width_m": scenario.channel_half_width,
        "thermal": {
            "air_temperature_C": scenario.thermal.air_temperature,
            "water_temperature_C": scenario.thermal.water_temperature,
            "ice_conductivity_W_mK": scenario.thermal.ice_conductivity,
            "latent_heat_J_kg": scenario.thermal.latent_heat,
            "nodes": scenario.thermal.nodes,
        },
        "encounter": {
            "brash_equivalent_fraction": scenario.encounter.brash_equivalent_fraction,
            "edge_zone_fraction": scenario.encounter.edge_zone_fraction,
            "edge_thickness_amplification": scenario.encounter.edge_thickness_amplification,
        },
        "wake": {
            "deficit": scenario.wake.deficit,
            "decay_length_factor": scenario.wake.decay_length_factor,
            "half_width_factor": scenario.wake.half_width_factor,
        },
    }


# ---------------------------------------------------------------------------
# Mission optimisation
# ---------------------------------------------------------------------------
def evaluate_mission(scenario: FormationScenario) -> dict[str, float]:
    """Objective and constraint values for one scenario (optimiser building block)."""
    result = simulate_formation(scenario)
    summary = result["summary"]
    return {
        "total_energy_GJ": summary["total_fuel_energy_GJ"],
        "mission_duration_h": summary["mission_duration_h"],
        "follower_saving_vs_level_ice": summary["follower_saving_vs_level_ice"],
        "spacings_m": summary["spacings_m"],
    }


def optimize_formation(scenario: FormationScenario,
                       spacing_candidates: Sequence[float],
                       speed_candidates: Sequence[float],
                       lateral_candidates: Sequence[float] = (0.0,),
                       *, max_duration_h: float | None = None,
                       min_spacing_factor: float = DEFAULT_MIN_SPACING_FACTOR,
                       layout: str = "column") -> dict[str, Any]:
    """Grid-search the formation spacing / lateral offset / speed.

    ``spacing_candidates`` are in ship lengths, ``lateral_candidates`` in beams.
    The operational constraint ``spacing >= min_spacing_factor * L`` is enforced
    before evaluation; ``max_duration_h`` optionally prunes missions that cannot be
    completed in time.  All evaluated points are returned so the optimum can be
    audited rather than trusted.
    """
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for spacing in spacing_candidates:
        spacing = _number(spacing, "spacing", positive=True)
        if spacing < min_spacing_factor:
            rows.append({"spacing_factor": spacing, "rejected": "below minimum spacing",
                         "speed_m_s": None, "lateral_factor": None,
                         "total_energy_GJ": None, "mission_duration_h": None,
                         "feasible": False})
            continue
        for lateral in lateral_candidates:
            lateral = _number(lateral, "lateral", nonnegative=True)
            if layout == "column" and lateral > 0.0:
                raise ValueError("lateral offsets require layout='staggered'")
            for speed in speed_candidates:
                speed = _number(speed, "speed", positive=True)
                candidate = replace(scenario, speed=speed, spacing_factor=spacing,
                                    lateral_factor=lateral, layout=layout)
                result = evaluate_mission(candidate)
                feasible = (max_duration_h is None
                            or result["mission_duration_h"] <= max_duration_h)
                row = {
                    "spacing_factor": spacing,
                    "lateral_factor": lateral,
                    "speed_m_s": speed,
                    "total_energy_GJ": result["total_energy_GJ"],
                    "mission_duration_h": result["mission_duration_h"],
                    "follower_saving_vs_level_ice": result["follower_saving_vs_level_ice"],
                    "feasible": feasible,
                }
                rows.append(row)
                if feasible and (best is None or row["total_energy_GJ"] < best["total_energy_GJ"]):
                    best = row
    if best is None:
        raise RuntimeError("no feasible formation was found in the search grid")
    improvement = 0.0
    energies = [row["total_energy_GJ"] for row in rows if row.get("feasible")]
    if energies and max(energies) > 0:
        improvement = 1.0 - best["total_energy_GJ"] / max(energies)
    return {
        "best": best,
        "candidates": rows,
        "energy_reduction_vs_worst_feasible": improvement,
        "min_spacing_factor": min_spacing_factor,
        "max_duration_h": max_duration_h,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def write_result(result: dict[str, Any], path: str | Path) -> Path:
    """Write a result bundle as JSON (lists/tensors already plain Python)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return target