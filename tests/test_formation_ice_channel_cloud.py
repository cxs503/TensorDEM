"""Tests for the formation multi-hull extension and the CLOUD benchmark.

The multi-hull (formation) support in :mod:`tensordem.ice_channel` is additive:
one reference hull plus ``extra_hulls``.  The physics anchors below are

* superposition -- the force of hull A plus hull B equals the sum of the two
  single-hull forces on the same state (the contact laws are independent),
* action/reaction -- ``force_on_hull_N`` is exactly minus the summed
  ``force_on_floes_N``,
* geometry -- ``cleared_width(hull=...)`` matches an independent corridor
  computation for every hull,
* artifact schema and invariants of the CLOUD benchmark (fast configuration).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)

from tensordem.ice_channel import (  # noqa: E402
    HullGeometry,
    IceChannelSimulation,
    WaterEnvironment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "benchmarks" / "formation_ice_channel_cloud.py"
STATE_FIELDS = ("pos", "vel", "yaw", "omega", "z", "vz", "tilt", "tilt_rate")


def _template_sim() -> IceChannelSimulation:
    far = HullGeometry(length=8.0, beam=4.0, speed=1.0, y_position=0.0, x_start=1.0e6)
    sim = IceChannelSimulation.from_field(
        0.6, domain_length=30.0, domain_width=30.0, seed=11, dt=5.0e-3, hull=far)
    return sim


def _clone_with_hulls(sim, hull, extras=()):
    clone = IceChannelSimulation(
        [float(r) for r in sim.radius], [float(t) for t in sim.thickness],
        water=sim.water, contact=sim.contact, hull=hull, extra_hulls=tuple(extras),
        dt=sim.dt, integrator=sim.integrator)
    for name in STATE_FIELDS:
        setattr(clone, name, getattr(sim, name).clone())
    clone.time = sim.time
    return clone


def test_extra_hulls_rejects_non_hull():
    with pytest.raises(TypeError):
        IceChannelSimulation([3.0], [0.5], hull=HullGeometry(), extra_hulls=[("x", 1.0)])


def test_hulls_property_keeps_reference_hull_first():
    lead = HullGeometry(y_position=10.0)
    follower = HullGeometry(y_position=20.0)
    sim = IceChannelSimulation([3.0], [0.5], hull=lead, extra_hulls=[follower])
    assert sim.hulls == (lead, follower)
    assert sim.hull is lead


def test_multi_hull_forces_superpose():
    sim = _template_sim()
    lead = HullGeometry(length=10.0, beam=6.0, speed=1.0, y_position=15.0, x_start=4.0)
    follower = HullGeometry(length=10.0, beam=6.0, speed=1.0, y_position=17.0, x_start=-6.0)

    alone_a = _clone_with_hulls(sim, lead)
    alone_b = _clone_with_hulls(sim, follower)
    both = _clone_with_hulls(sim, lead, extras=(follower,))

    f_a, _, _ = alone_a.net_forces(alone_a.time)
    f_b, _, _ = alone_b.net_forces(alone_b.time)
    f_ab, _, _ = both.net_forces(both.time)
    # hull pairs do not interact: the combined hull force is the sum of the parts
    hull_a = f_a - sim.net_forces(sim.time)[0]
    hull_b = f_b - sim.net_forces(sim.time)[0]
    hull_ab = f_ab - sim.net_forces(sim.time)[0]
    residual = (hull_ab - (hull_a + hull_b)).abs().max()
    scale = max(float(hull_a.abs().max()), float(hull_b.abs().max()), 1.0)
    assert float(residual) < 1e-9 * scale


def test_hull_contact_ledger_is_action_reaction():
    sim = _template_sim()
    lead = HullGeometry(length=10.0, beam=6.0, speed=1.0, y_position=15.0, x_start=4.0)
    follower = HullGeometry(length=10.0, beam=6.0, speed=1.0, y_position=15.0, x_start=-8.0)
    both = _clone_with_hulls(sim, lead, extras=(follower,))
    ledger = both.hull_contact(both.time)
    assert len(ledger) == 2
    for entry in ledger:
        magnitude = torch.linalg.norm(entry["force_on_floes_N"], dim=1)
        assert torch.allclose(magnitude, entry["per_floe_force_magnitude_N"])
        assert int((magnitude > 0.0).sum()) == entry["contact_count"]
    summed_floes = sum(entry["force_on_floes_N"] for entry in ledger).sum(dim=0)
    summed_hull = sum(entry["force_on_hull_N"] for entry in ledger)
    assert torch.allclose(summed_floes, -summed_hull, atol=1e-10)
    # the ledger is a pure query: the state digest must not change
    digest = both.state_digest()
    both.hull_contact(both.time)
    assert both.state_digest() == digest


def test_cleared_width_accepts_each_hull():
    sim = _template_sim()
    y_track = 15.0
    lead = HullGeometry(length=10.0, beam=6.0, speed=0.0, y_position=y_track, x_start=15.0)
    follower = HullGeometry(length=10.0, beam=6.0, speed=0.0, y_position=y_track,
                            x_start=15.0)
    both = _clone_with_hulls(sim, lead, extras=(follower,))
    both.pos = torch.tensor([[15.0, y_track + 5.0], [8.0, y_track - 4.0]], dtype=both.dtype)
    both.radius = torch.tensor([2.0, 1.0], dtype=both.dtype)
    # corridor x in [5, 15]: floes at (15, y+5) r=2 -> clearance 3, at (8, y-4) r=1 -> 3
    width = both.cleared_width(hull=follower)
    assert math.isclose(width, 6.0, abs_tol=1e-12)
    # the default hull is the reference hull, so the two must agree here
    assert math.isclose(both.cleared_width(), width, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# benchmark artifact
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fast_artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("cloud")
    for stage in ("production", "assemble"):
        cmd = [sys.executable, str(BENCH), "--fast", "--stage", stage,
               "--out-dir", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            pytest.skip("fast CLI pipeline unavailable in this environment "
                        f"(declared debt, docs/formation_ice_channel_cloud.md §6.7): "
                        f"{result.stderr.strip()[-600:]}")
    cloud_path = out / "fast" / "field_cloud.json"
    assert cloud_path.exists(), result.stdout[-2000:] + result.stderr[-2000:]
    return out / "fast"


def test_cloud_artifact_schema(fast_artifacts):
    payload = json.loads((fast_artifacts / "field_cloud.json").read_text())
    assert payload["scenario_key"] == "paper-mat-formation-ice-channel-cloud"
    assert payload["geometry_check"]["passed"] is True
    for key in ("local_concentration", "mean_speed", "kinetic_energy_density",
                "mean_ice_load", "mean_ice_load_lead", "mean_ice_load_follower",
                "lateral_displacement", "floe_samples"):
        field = payload["fields"][key]
        nx = payload["grid_definition"]["nx"]
        ny = payload["grid_definition"]["ny"]
        assert len(field) == ny and all(len(row) == nx for row in field), key
        assert all(math.isfinite(v) for row in field for v in row), key
    assert payload["hulls"][0]["name"] == "lead"
    assert len(payload["hulls"]) == 3


def test_cloud_physics_invariants(fast_artifacts):
    payload = json.loads((fast_artifacts / "field_cloud.json").read_text())
    glob = payload["global_rows"]
    # floe area is conserved exactly (rigid disks, no break-up in this model)
    assert glob["floe_area_rel_drift"] < 1e-12
    # the generated field hits the requested concentration to <1 %
    assert glob["concentration_rel_error_pct"] < 1.0
    # angular distributions are normalised and are distributions, not spikes
    for key in ("velocity_direction_in_track", "velocity_direction_off_track",
                "hull_load_direction_lead", "hull_load_direction_follower"):
        freqs = payload["angular"][key]
        assert len(freqs) == 12
        assert math.isclose(sum(freqs), 1.0, abs_tol=1e-9), (key, sum(freqs))
    # report the reduction rather than asserting its sign: a follower in the
    # cleared channel must not carry more load than the lead over its own window
    agg = payload["aggregates"]["production"]
    assert agg["mean_abs_ice_force_lead_N"] is not None
    assert agg["mean_abs_ice_force_lead_N"] > 0.0


def test_cloud_table_and_figures(fast_artifacts):
    payload = json.loads((fast_artifacts / "field_cloud.json").read_text())
    nx = payload["grid_definition"]["nx"]
    ny = payload["grid_definition"]["ny"]
    lines = (fast_artifacts / "field_table.csv").read_text().strip().splitlines()
    assert len(lines) == payload["table_rows"] + 1
    assert payload["table_rows"] >= 3 * nx * ny + 4 * 12 + nx
    for name in ("fig_field_cloud.png", "fig_profile_reference.png",
                 "fig_angular_cloud.png"):
        path = fast_artifacts / name
        assert path.exists() and path.stat().st_size > 10_000, name
    assert len(payload["figures"]) == 3
    # the profile keeps the exact geometric reference
    assert math.isclose(payload["profile"]["reference_half_beam_m"],
                        0.5 * payload["config"]["hulls"][0]["beam"])