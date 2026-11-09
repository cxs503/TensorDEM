"""Integrated verification of the DEM contact-physics module (P121W5-N11).

The node matures three campaign deliverables into one module:

* ``P121W5-N02`` temperature-dependent freeze bond,
* ``P121W5-N05`` ice-ridge morphology / contact-network statistics,
* ``P121W5-N08`` melt-lubrication <-> contact two-way coupling.

Acceptance criteria pinned here (see the campaign table metadata):

(a) **Integration** -- an integrated run of
    :class:`tensordem.contact_physics.ContactPhysicsMarch` reproduces the three
    modules run standalone in the same工况 to ``<= 1e-10`` relative error, and
    the coupling invariants hold.
(b) **API stability** -- the public signatures and file hashes of the three
    source modules are unchanged against the committed pre-integration baseline
    (zero signature changes), and every new public API has a docstring.
(c) **Performance** -- the integration layer costs ``< 5 %`` of wall time and of
    Python-side peak memory relative to the three standalone drivers.
(d) **Regression** -- the node test suites of N02 / N05 / N08 still pass (the
    gate is executed by ``scripts/contact_physics_regression_gate.py``; the
    committed artifact is asserted here).
(e) **Documentation** -- ``docs/contact_physics_module.md`` documents every
    public API name and ``examples/contact_physics_demo.py`` runs standalone.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import pathlib
import subprocess
import sys
import time
import tracemalloc

import numpy as np
import pytest
import torch

from tensordem.contact_physics import (
    ContactPhysicsMarch,
    OperatingPoint,
    RidgeContactStatistics,
    RidgeMorphologyCase,
    TemperatureDependentBond,
    LubricationContactCoupling,
)
from tensordem.contact_physics import bond as bond_module
from tensordem.ice_bond_viscoelastic import advance_branches, damage_increment, stored_energy
from tensordem.ice_rubble_microstructure import force_chain_pdf
from tensordem.interface_thermal.coupling import CoupledInterface, IceSteelContact
from tensordem.interface_thermal.lubrication import MeltLubricationModel
from tensordem.ridge_dem import RidgeKeel, RidgeMaterial

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "contact_physics"

TOLERANCE = 1e-10


# ---------------------------------------------------------------------------
#工况 set (one operating point per physical regime)
# ---------------------------------------------------------------------------
def _cold_contact() -> IceSteelContact:
    """Slow sliding contact of sea ice at -20 degC.

    ``v = 0.02 m/s`` keeps the interface inside the fitted band of the
    Timco-Weeks strength relation (``T <= -2 degC``) for the whole window, so
    the bond damage there is pure Kachanov creep -- not the near-melting
    collapse of the brine-volume factor.
    """
    return IceSteelContact(cells=61, slip_velocity_m_s=0.02,
                           initial_temperature_c=-20.0)


def _cold_slow() -> OperatingPoint:
    """Cold sliding contact, bonds stay intact (creep damage << 1)."""
    return OperatingPoint(
        contact=_cold_contact(), strain_rate_1_s=1.0e-5, dt_s=2.0e-4,
        statistics_stride=10)


def _cold_rupture() -> OperatingPoint:
    """Cold sliding contact at a strain rate that ruptures the bond bundle."""
    point = _cold_slow()
    return OperatingPoint(
        contact=point.contact, bond=point.bond, ridge=point.ridge,
        strain_rate_1_s=0.06, n_bonds=point.n_bonds,
        gauge_length_range_m=point.gauge_length_range_m, dt_s=point.dt_s,
        statistics_stride=point.statistics_stride, seed=point.seed)


def _melting() -> OperatingPoint:
    """Interface pinned at the melting point: Stefan film grows, mu -> mu_wet."""
    contact = IceSteelContact(
        cells=61,
        lubrication=MeltLubricationModel(mu_base=0.10, t_onset=-3.0, mu_wet=0.02))
    point = _cold_slow()
    return OperatingPoint(
        contact=contact, bond=point.bond, ridge=point.ridge,
        strain_rate_1_s=point.strain_rate_1_s, n_bonds=point.n_bonds,
        gauge_length_range_m=point.gauge_length_range_m, dt_s=point.dt_s,
        statistics_stride=point.statistics_stride, seed=point.seed)


CASES = {
    "cold_slow": (_cold_slow, 200),
    "cold_rupture": (_cold_rupture, 200),
    "melting": (_melting, 300),
}

COMPARED_KEYS = (
    "time", "interface_temperature_c", "mu", "melt_fraction",
    "load_bearing_fraction", "friction_flux_w_m2", "film_thickness_m",
    "film_normal_stiffness_n_m", "bond_force_mean_n", "bond_force_max_n",
    "bond_stiffness_ratio", "bond_strain", "damage_increment_max", "damage_max",
    "broken_bonds", "work_j", "dissipation_j", "stored_energy_j",
    "coupled_iterations",
)


# ---------------------------------------------------------------------------
# standalone reference: the three modules driven with raw library calls only
# ---------------------------------------------------------------------------
def standalone_reference(point: OperatingPoint, n_steps: int,
                         statistics: "RidgeContactStatistics | None" = None) -> dict:
    """Drive N08 / N02 / N05 with direct library calls (no adapter objects).

    This is the reference the integrated march must reproduce.  It is written
    against the library functions themselves, so it fails if the adapter
    silently changes an input, an order or a formula.
    """
    dt = float(point.dt_s)
    coupled = CoupledInterface(point.contact, max_iterations=40,
                               tolerance=1e-6, allow_melting=True)
    law = point.bond.law
    gauge = torch.as_tensor(point.gauge_lengths(), dtype=torch.float64)
    offsets = torch.zeros((int(point.n_bonds), law.n_branches), dtype=torch.float64)
    damage = torch.zeros(int(point.n_bonds), dtype=torch.float64)
    elongation = torch.zeros(int(point.n_bonds), dtype=torch.float64)
    strain = 0.0
    cumulative = {"work_j": 0.0, "dissipation_j": 0.0, "stored_energy_j": 0.0}
    history: dict[str, list] = {}
    samples: list[float] = []

    lubrication_records = []
    for _ in range(n_steps):
        lubrication_records.append(coupled.step(dt))

    for index, lubricated in enumerate(lubrication_records):
        temperature = float(lubricated["interface_temperature_c"])
        strain_new = strain + float(point.strain_rate_1_s) * dt
        elongation_new = torch.as_tensor(strain_new, dtype=torch.float64) * gauge
        stored_old = float(stored_energy(law, offsets, elongation).sum())
        offsets, force_raw, work, dissipation = advance_branches(
            law, offsets, elongation, elongation_new, dt)
        stored_new = float(stored_energy(law, offsets, elongation_new).sum())
        increment = damage_increment(
            force_raw,
            torch.as_tensor(point.bond.reference_force(temperature), dtype=torch.float64),
            dt, point.bond.damage_time_s, point.bond.damage_exponent)
        damage = torch.clamp(damage + increment, max=1.0)
        broken = damage >= 1.0
        force = force_raw.masked_fill(broken, 0.0)
        cumulative["work_j"] += float(work.sum())
        cumulative["dissipation_j"] += float(dissipation.sum())
        cumulative["stored_energy_j"] += stored_new - stored_old
        record = {
            "step": index,
            "time": float(lubricated["time"]),
            "interface_temperature_c": temperature,
            "mu": float(lubricated["mu"]),
            "melt_fraction": point.contact.lubrication.melt_fraction(temperature),
            "load_bearing_fraction": max(
                0.0, 1.0 - point.contact.lubrication.melt_fraction(temperature)),
            "friction_flux_w_m2": point.contact.lubrication.mu(
                temperature, mu_base=point.contact.mu_cold)
            * point.contact.mean_pressure_pa * point.contact.slip_velocity_m_s,
            "film_thickness_m": float(lubricated["film_thickness_m"]),
            "film_normal_stiffness_n_m": (
                float(point.contact.contact_stiffness_n_m)
                if lubricated["film_thickness_m"] <= 0.0
                else float(point.contact.lubrication.effective_normal_stiffness(
                    point.contact.contact_stiffness_n_m,
                    point.contact.contact_area_m2,
                    lubricated["film_thickness_m"]))),
            "bond_force_mean_n": float(force.mean()),
            "broken_bonds": int(broken.sum()),
            "bond_force_max_n": float(force.max()),
            "bond_stiffness_ratio": (float(force.mean() / elongation_new.mean())
                                     if float(elongation_new.mean()) > 0.0 else 0.0),
            "bond_strain": float(strain_new),
            "damage_increment_max": float(increment.max()),
            "damage_max": float(damage.max()),
            "coupled_step_mode": str(lubricated["mode"]),
            "coupled_iterations": int(lubricated["iterations"]),
            "work_j": float(work.sum()),
            "dissipation_j": float(dissipation.sum()),
            "stored_energy_j": stored_new,
        }
        for key, value in record.items():
            history.setdefault(key, []).append(value)
        if index % int(point.statistics_stride) == 0:
            samples.extend(float(x) for x in force.abs().tolist())
        strain = strain_new
        elongation = elongation_new

    positive = [value for value in samples if value > 0.0]
    sampler = statistics if statistics is not None else RidgeContactStatistics(
        RidgeMorphologyCase(keel=point.ridge.keel, material=point.ridge.material,
                            tip_height=point.ridge.tip_height))
    statistics = None
    if len(positive) >= 4:
        statistics = {
            "morphology": sampler.morphology(),
            "force_network": {
                key: float(force_chain_pdf(positive, n_bins=20)[key])
                for key in ("n_contacts", "mean_force", "coefficient_of_variation",
                            "strong_fraction", "strong_load_fraction")},
        }
    work = cumulative["work_j"]
    return {
        "histories": {key: np.asarray(values) for key, values in history.items()},
        "cumulative": cumulative,
        "ledger_error": (work - cumulative["stored_energy_j"]
                         - cumulative["dissipation_j"]) / max(abs(work), 1.0),
        "statistics": statistics,
    }


def relative_error(actual, expected) -> float:
    """``max |a - e| / max(|e|, 1e-30)`` over the series."""
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if a.size == 0:
        return 0.0
    scale = np.maximum(np.abs(e), 1e-30)
    return float(np.max(np.abs(a - e) / scale))


# ---------------------------------------------------------------------------
# (a) integration: integrated run == standalone runs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case_name", sorted(CASES))
def test_integrated_matches_standalone(case_name):
    builder, n_steps = CASES[case_name]
    point = builder()
    integrated = ContactPhysicsMarch(point).run(n_steps)
    reference = standalone_reference(point, n_steps)

    worst = {}
    for key in COMPARED_KEYS:
        error = relative_error(integrated["histories"][key],
                               reference["histories"][key])
        worst[key] = error
        assert error <= TOLERANCE, f"{case_name}/{key}: rel. error {error:.3e}"
    assert max(worst.values()) <= TOLERANCE
    # the per-step record layout itself is identical (fair overhead comparison)
    assert (sorted(integrated["final"]) == sorted(reference["histories"]))
    assert (integrated["histories"]["coupled_step_mode"]
            == list(reference["histories"]["coupled_step_mode"]))
    # energy ledger of the integrated run closes at round-off
    assert abs(integrated["energy"]["ledger_error"]) <= 1e-12
    # the statistics agree (N05 sampling of the same window)
    assert reference["statistics"] is not None
    assert integrated["statistics"] is not None
    assert (integrated["statistics"]["morphology"]
            == reference["statistics"]["morphology"])
    for key, value in reference["statistics"]["force_network"].items():
        assert integrated["statistics"]["force_network"][key] == pytest.approx(
            value, rel=TOLERANCE, abs=0.0)


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_coupling_invariants(case_name):
    """Self-consistency of the three modules in one工况."""
    builder, n_steps = CASES[case_name]
    point = builder()
    summary = ContactPhysicsMarch(point).run(n_steps)
    invariants = summary["invariants"]
    history = summary["histories"]
    contact = point.contact

    # N08: interface temperature never leaves [initial - 1 K, T_melt]
    assert invariants["interface_temperature_within_bounds"]
    assert min(history["interface_temperature_c"]) >= contact.initial_temperature_c - 1e-12
    assert max(history["interface_temperature_c"]) <= contact.t_melt + 1e-12
    # N08: mu is the melt-fraction law at the *reported* interface temperature
    for temperature, mu in zip(history["interface_temperature_c"], history["mu"]):
        assert mu == pytest.approx(
            contact.lubrication.mu(temperature, mu_base=contact.mu_cold),
            rel=1e-15, abs=1e-18)
        assert 0.0 <= mu <= contact.mu_cold
    # N08: the film never softens the contact below the elastic stiffness
    for stiffness in history["film_normal_stiffness_n_m"]:
        assert 0.0 < stiffness <= contact.contact_stiffness_n_m * (1.0 + 1e-12)
    assert invariants["film_monotone"]
    # N02: damage is monotone, truncated at rupture, and never exceeds 1
    assert invariants["damage_monotone"]
    assert max(history["damage_max"]) <= 1.0
    assert min(history["damage_max"]) >= 0.0
    # N02 -> N05: broken bonds carry no force
    final_force = history["bond_force_mean_n"][-1]
    if history["broken_bonds"][-1] == point.n_bonds:
        assert final_force == 0.0
    # energy ledger of the bond law (N02) closes at round-off
    assert abs(summary["energy"]["ledger_error"]) <= 1e-12
    assert summary["energy"]["dissipation_j"] >= 0.0


def test_melting_regime_grows_a_film_and_kills_the_bond():
    """Stefan film growth and the ``F_ref -> 0`` melting limit, tied together."""
    point = _melting()
    summary = ContactPhysicsMarch(point).run(120)
    history = summary["histories"]
    melting_steps = [mode for mode in history["coupled_step_mode"] if mode == "melting"]
    assert melting_steps, "the melting branch of N08 was never entered"
    assert history["film_thickness_m"][-1] > 0.0
    # mu collapses to the residual wet value at the melting point
    assert history["mu"][-1] == pytest.approx(point.contact.lubrication.mu_wet, rel=1e-12)
    assert history["melt_fraction"][-1] == pytest.approx(1.0, abs=1e-15)
    # a liquid interface carries no freeze bond: strength factor -> 0, so the
    # Kachanov clock saturates within one step (documented limit)
    assert point.bond.strength_factor(0.0) == 0.0
    assert point.bond.reference_force(0.0) == 0.0
    first_melting = next(index for index, mode in
                         enumerate(history["coupled_step_mode"]) if mode == "melting")
    # the melting point is reached at step ``first_melting``; F_ref is then 0,
    # so the clock saturates within the next increment
    assert first_melting >= 1
    assert history["interface_temperature_c"][first_melting] == pytest.approx(
        point.contact.t_melt, abs=1e-12)
    assert history["damage_max"][first_melting + 1] == pytest.approx(1.0, abs=1e-12)
    assert history["broken_bonds"][first_melting + 1] == point.n_bonds


def test_creep_rupture_happens_inside_the_fitted_band():
    """The rupture case is creep-driven, not a brine-volume collapse."""
    point = _cold_rupture()
    summary = ContactPhysicsMarch(point).run(200)
    history = summary["histories"]
    first = next((index for index, value in enumerate(history["damage_max"])
                  if value >= 1.0), None)
    assert first is not None, "no bond ruptured in the window"
    assert point.bond.within_strength_fit(history["interface_temperature_c"][first])
    assert 0 < history["broken_bonds"][-1] <= point.n_bonds
    assert max(history["film_thickness_m"]) == 0.0


def test_cold_regime_keeps_the_bonds_and_the_film_at_zero():
    """No latent heat below the onset band: film stays 0, k_eff = k_contact."""
    point = _cold_slow()
    summary = ContactPhysicsMarch(point).run(120)
    history = summary["histories"]
    assert all(mode == "sensible" for mode in history["coupled_step_mode"])
    assert max(history["film_thickness_m"]) == 0.0
    assert history["film_normal_stiffness_n_m"][-1] == pytest.approx(
        point.contact.contact_stiffness_n_m, rel=1e-15)
    assert history["broken_bonds"][-1] == 0
    assert history["damage_max"][-1] < 1e-6
    # the whole window stays inside the fitted band of the strength relation
    assert max(history["interface_temperature_c"]) <= -2.0
    assert all(point.bond.within_strength_fit(t)
               for t in history["interface_temperature_c"])
    assert history["load_bearing_fraction"][-1] == pytest.approx(
        1.0 - point.contact.lubrication.melt_fraction(
            history["interface_temperature_c"][-1]), rel=1e-15)


def test_temperature_enters_the_damage_clock_in_the_documented_direction():
    """Colder ice is stronger: ``F_ref`` increases as ``T`` decreases."""
    bond = _cold_slow().bond
    temperatures = (-2.0, -5.0, -10.0, -20.0)
    forces = [bond.reference_force(t) for t in temperatures]
    assert forces == sorted(forces)
    assert bond.strength_factor(bond.reference_temperature_c) == pytest.approx(1.0, rel=1e-15)
    for temperature, force in zip(temperatures, forces):
        assert force == pytest.approx(
            bond.reference_force_n
            * bond_module.timco_weeks_strength_factor(
                bond_module.timco_weeks_brine_volume(bond.salinity_ppt, temperature))
            / bond_module.timco_weeks_strength_factor(
                bond_module.timco_weeks_brine_volume(
                    bond.salinity_ppt, bond.reference_temperature_c)),
            rel=1e-15)
    # validity window of the strength relation is reported, not hidden
    assert bond.within_strength_fit(-5.0)
    assert not bond.within_strength_fit(-1.0)
    assert not bond.within_strength_fit(bond_module.STRENGTH_FIT_CEILING_C + 0.5)
    # at 5 ppt the factor has collapsed well before the melting point: 1.1 % of
    # the -10 degC value at -1 degC, and exactly 0 once the brine volume passes
    # the 0.29 pore limit (T >= -0.72 degC at this salinity)
    assert bond.strength_factor(-1.0) < 0.02
    assert bond.strength_factor(-0.7) == 0.0
    assert bond.within_strength_fit(-1.0) is False
    with pytest.raises(ValueError):
        bond.brine_volume(0.0)
    with pytest.raises(ValueError):
        bond.brine_volume(bond_module.BRINE_RELATION_FLOOR_C - 1.0)


def test_bond_bundle_reproduces_the_raw_law_exactly():
    """One adapter step == one raw library step (no hidden rescaling)."""
    point = _cold_slow()
    bond = point.bond
    gauge = torch.as_tensor(point.gauge_lengths(), dtype=torch.float64)
    offsets = torch.zeros((point.n_bonds, bond.law.n_branches), dtype=torch.float64)
    elongation = torch.zeros(point.n_bonds, dtype=torch.float64)
    elongation_new = torch.full_like(elongation, 1.0e-7)
    adapted = bond.advance(offsets, elongation, elongation_new, 1.0e-3)
    raw_offsets, raw_force, raw_work, raw_dissipation = advance_branches(
        bond.law, offsets, elongation, elongation_new, 1.0e-3)
    assert torch.equal(adapted["branch_offsets"], raw_offsets)
    assert torch.equal(adapted["force"], raw_force)
    assert torch.equal(adapted["work"], raw_work)
    assert torch.equal(adapted["dissipation"], raw_dissipation)
    assert torch.equal(adapted["stored_energy"],
                       stored_energy(bond.law, raw_offsets, elongation_new))
    assert gauge.shape == (point.n_bonds,)


def test_lubrication_adapter_delegates_the_coupled_solver():
    """``LubricationContactCoupling.step`` == ``CoupledInterface.step``."""
    point = _cold_slow()
    adapter = LubricationContactCoupling(point.contact)
    raw = CoupledInterface(point.contact, max_iterations=40, tolerance=1e-6,
                           allow_melting=True)
    for _ in range(3):
        adapted_record = adapter.step(point.dt_s)
        raw_record = raw.step(point.dt_s)
        for key in ("time", "interface_temperature_c", "mu", "film_thickness_m",
                    "iterations", "residual", "converged", "mode"):
            assert adapted_record[key] == raw_record[key]
        assert adapter.normal_stiffness() == pytest.approx(
            raw.normal_force_at_approach(1.0) / 1.0, rel=1e-12)


def test_statistics_sampler_is_deterministic_and_complete():
    """The ridge realisation and its statistics are seed-reproducible."""
    case = RidgeMorphologyCase(
        keel=RidgeKeel(keel_depth=6.0, keel_width=5.0, porosity=0.30),
        material=RidgeMaterial(particle_radius=0.30))
    first = RidgeContactStatistics(case).sample(np.linspace(1.0, 10.0, 40))
    second = RidgeContactStatistics(case).sample(np.linspace(1.0, 10.0, 40))
    assert first == second
    morphology = first["morphology"]
    assert 0.0 < morphology["realised_porosity"] < 0.6
    assert morphology["n_bonds"] > 0
    assert morphology["mean_coordination"] > 0.0
    assert morphology["keel_angle_deg"] == pytest.approx(
        math.degrees(math.atan(0.5 * 5.0 / 6.0)), rel=1e-15)
    network = first["force_network"]
    assert network["strong_fraction"] == pytest.approx(0.5, abs=0.2)
    assert 0.0 < network["strong_load_fraction"] <= 1.0
    assert network["coefficient_of_variation"] > 0.0
    # the cutoff sweep is monotone: a higher force threshold keeps fewer contacts
    sweep = RidgeContactStatistics(case).coordination_sweep()
    assert [row["n_contacts"] for row in sweep] == sorted(
        (row["n_contacts"] for row in sweep), reverse=True)


# ---------------------------------------------------------------------------
# (b) API stability against the frozen pre-integration baseline
# ---------------------------------------------------------------------------
def _baseline() -> dict:
    path = ARTIFACTS / "api_baseline.json"
    assert path.exists(), f"missing API baseline {path}"
    return json.loads(path.read_text())


def test_source_modules_are_byte_identical_to_the_baseline():
    baseline = _baseline()
    for module_name, record in baseline["modules"].items():
        path = ROOT / record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"], (
            f"{module_name} changed since the pre-integration baseline "
            f"({baseline['generated_from_git_rev']})")


def _public_names(module) -> list[str]:
    """Mirror of ``scripts/freeze_contact_physics_api.py::public_names``."""
    declared = getattr(module, "__all__", None)
    if declared:
        return sorted(declared)
    names = []
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(value) or inspect.isclass(value):
            if getattr(value, "__module__", None) == module.__name__:
                names.append(name)
    return sorted(names)


def test_public_signatures_are_unchanged():
    baseline = _baseline()
    for module_name, record in baseline["modules"].items():
        module = __import__(module_name, fromlist=["__all__"])
        assert _public_names(module) == sorted(record["signatures"])
        for name, expected in record["signatures"].items():
            target = getattr(module, name)
            try:
                live = str(inspect.signature(target))
            except (TypeError, ValueError):
                live = "<no signature>"  # module-level dataclass *instance*
            assert live == expected, f"{module_name}.{name} signature changed"


def test_every_new_public_api_has_a_docstring():
    import tensordem.contact_physics as package

    missing = [name for name in package.__all__
               if not (getattr(package, name).__doc__ or "").strip()]
    assert not missing, f"public names without a docstring: {missing}"
    for name in package.__all__:
        target = getattr(package, name)
        assert len((target.__doc__ or "").strip()) >= 40, name
        for attribute in dir(target):
            if attribute.startswith("_"):
                continue
            member = getattr(target, attribute)
            if inspect.isfunction(member) or inspect.ismethod(member):
                assert (member.__doc__ or "").strip(), f"{name}.{attribute}"


# ---------------------------------------------------------------------------
# (c) performance: overhead of the integration layer
# ---------------------------------------------------------------------------
def _measure(function, repeats: int) -> tuple[list[float], list[int]]:
    times, peaks = [], []
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    return times, peaks


def _paired_overhead(n_steps: int = 60, repeats: int = 9) -> dict:
    """Min-of-N interleaved timing of the integrated vs the standalone paths.

    The measurement conventions (all of them matter on a shared 128-core box):

    * torch is pinned to one thread -- the tensors here are 64 elements wide, so
      threading only adds queueing noise;
    * both paths pre-assemble the ridge realisation once, outside the timed
      region (it is state independent and identical in both paths);
    * the two implementations are timed alternately, and the **minimum** of
      ``repeats`` is the estimator (``timeit`` convention: the minimum converges
      to the unloaded cost, the median on a contended box does not).  The full
      sample lists are written to the artifact so the estimate can be re-judged.
    """
    point = _cold_slow()
    sampler_integrated = RidgeContactStatistics(point.ridge)
    sampler_integrated.assemble()
    sampler_standalone = RidgeContactStatistics(point.ridge)
    sampler_standalone.assemble()
    integrated_fn = lambda: ContactPhysicsMarch(
        point, statistics=sampler_integrated).run(n_steps)
    standalone_fn = lambda: standalone_reference(
        point, n_steps, statistics=sampler_standalone)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        integrated_fn()
        standalone_fn()
        integrated_times, integrated_peaks = [], []
        standalone_times, standalone_peaks = [], []
        for _ in range(repeats):
            times, peaks = _measure(integrated_fn, 1)
            integrated_times += times
            integrated_peaks += peaks
            times, peaks = _measure(standalone_fn, 1)
            standalone_times += times
            standalone_peaks += peaks
    finally:
        torch.set_num_threads(previous_threads)

    integrated_time = float(np.min(integrated_times))
    standalone_time = float(np.min(standalone_times))
    integrated_memory = int(np.min(integrated_peaks))
    standalone_memory = int(np.min(standalone_peaks))
    return {
        "n_steps": n_steps,
        "repeats": repeats,
        "threads": 1,
        "estimator": "min",
        "integrated_samples_s": integrated_times,
        "standalone_samples_s": standalone_times,
        "integrated_wall_s": integrated_time,
        "standalone_wall_s": standalone_time,
        "integrated_wall_median_s": float(np.median(integrated_times)),
        "standalone_wall_median_s": float(np.median(standalone_times)),
        "time_overhead_percent": 100.0 * (integrated_time - standalone_time) / standalone_time,
        "time_overhead_median_percent":
            100.0 * (np.median(integrated_times) - np.median(standalone_times))
            / np.median(standalone_times),
        "integrated_peak_py_bytes": integrated_memory,
        "standalone_peak_py_bytes": standalone_memory,
        "memory_overhead_percent":
            100.0 * (integrated_memory - standalone_memory) / standalone_memory,
        "time_target_percent": 5.0,
        "memory_target_percent": 5.0,
    }


def test_integration_overhead_is_below_five_percent():
    record = _paired_overhead()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "performance_pytest.json").write_text(json.dumps(record, indent=2))
    assert record["time_overhead_percent"] < 5.0, record
    assert record["memory_overhead_percent"] < 5.0, record


# ---------------------------------------------------------------------------
# (d) regression gate over the upstream node suites
# ---------------------------------------------------------------------------
NODE_TESTS = (
    "tests/test_ice_bond_viscoelastic.py",          # N02
    "tests/test_ice_bond_engine.py",                # N02
    "tests/test_ice_bond_single_bond_engine.py",    # N02
    "tests/test_interface_thermal_conduction.py",   # N08
    "tests/test_interface_thermal_coupling.py",     # N08
    "tests/test_interface_thermal_friction.py",     # N08
    "tests/test_interface_thermal_stefan.py",       # N08
    "tests/test_ridge_dem_core_physics.py",         # N05
    "tests/test_ice_rubble_microstructure.py",      # N05
)


def test_regression_gate_artifact_is_green():
    path = ARTIFACTS / "regression_gate.json"
    assert path.exists(), (
        "missing regression gate artifact; run "
        "python3 scripts/contact_physics_regression_gate.py")
    gate = json.loads(path.read_text())
    assert sorted(gate["node_tests"]) == sorted(NODE_TESTS)
    assert gate["returncode"] == 0
    assert gate["failed"] == 0
    assert gate["passed"] > 0
    assert gate["errors"] == 0
    # the gate must postdate the module (it is not a stale copy)
    module_mtime = max(
        (ROOT / "src" / "tensordem" / "contact_physics").glob("*.py"),
        key=lambda p: p.stat().st_mtime).stat().st_mtime
    assert gate["finished_unix"] >= module_mtime - 1.0


# ---------------------------------------------------------------------------
# (e) documentation coverage and a runnable example
# ---------------------------------------------------------------------------
def test_documentation_covers_every_public_api():
    doc = (ROOT / "docs" / "contact_physics_module.md").read_text()
    import tensordem.contact_physics as package

    for name in package.__all__:
        assert name in doc, f"{name} not documented in docs/contact_physics_module.md"
    for module in ("bond", "lubrication", "statistics", "march"):
        assert f"contact_physics/{module}.py" in doc or f"contact_physics.{module}" in doc
    for node in ("P121W5-N02", "P121W5-N05", "P121W5-N08"):
        assert node in doc


def test_example_runs_standalone():
    script = ROOT / "examples" / "contact_physics_demo.py"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, str(script)], cwd=str(ROOT), capture_output=True,
        text=True, timeout=600, env=environment)
    assert completed.returncode == 0, completed.stderr[-4000:]
    out = completed.stdout
    assert "interface_temperature_c" in out
    assert "ledger_error" in out
    assert "realised_porosity" in out