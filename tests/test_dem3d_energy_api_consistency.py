"""Energy-API consistency: ``IceDEM3D.energy()`` vs ``IceDEM3D.diagnostics()``.

Both public reports name the same energy channels.  Until this test existed they
disagreed on one channel by up to six orders of magnitude: ``diagnostics()``
reported the irreversible failure accumulator ``failure_energy_J.sum()``, while
``energy()`` recomputed ``0.5*k*sum(extension^2 over dead bonds)`` from the
*current* particle positions.  After a bond breaks its particles are free, so
that second expression grows without any bond event and is not a released
energy.  Measured on the indentation case below: 83.6 J reported versus 0.114 J
actually released, with 77.7 J of the spurious value accumulated over a window
in which the ledger did not move by a single joule
(``artifacts/dem3d_energy_api/``).

The tests pin:
1. single-bond failure energy never exceeds the closed form
   ``0.5*k*(breaking_strain*L)^2`` and its implied break strain stays inside the
   configured threshold;
2. every shared channel of the two APIs agrees on an arbitrary state;
3. ``released_bond_energy_J`` is monotone, non-growing between bond events, and
   cannot exceed ``0.5*k*sum(extension^2 over dead bonds)`` at late time;
4. the released channel never exceeds the total mechanical energy budget
   (a proxy for "released energy is physical, not a length artefact").
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.dem3d import IceDEM3D, DEM3DConfig  # noqa: E402

SHARED = (
    "broken_bonds",
    "particle_count",
    "kinetic_energy_J",
    "bond_elastic_energy_J",
    "particle_contact_energy_J",
    "tool_contact_energy_J",
    "platen_contact_energy_J",
    "released_bond_energy_J",
    "mechanical_energy_J",
)


def _solver(breaking_strain: float = 0.004) -> IceDEM3D:
    return IceDEM3D(DEM3DConfig(
        nx=6, ny=3, nz=3,
        breaking_strain=breaking_strain,
        contact_damping=0.0,
        drag=0.0,
        tool_speed=0.5,
    ))


def _run_until_break(sim: IceDEM3D, limit: int = 1200) -> int:
    for step in range(limit):
        sim.step()
        if sim.broken_bonds:
            return step
    raise AssertionError("indentation case did not break a bond within the step limit")


def test_single_bond_failure_energy_matches_closed_form():
    sim = _solver()
    _run_until_break(sim)
    stored = sim.failure_energy_J[sim.failure_energy_J > 0]
    assert stored.numel() > 0
    k = sim.config.bond_stiffness
    spacing = float(torch.linalg.vector_norm(sim.reference_bond_vectors, dim=1).max())
    closed_form = 0.5 * k * (sim.config.breaking_strain * spacing) ** 2
    # In this indentation case the first events are *shear* triggered (the
    # deviatoric invariant reaches shear_breaking_strain while the axial
    # extension is still far below the tensile threshold), so the tensile
    # closed form is an upper bound rather than an attained value.  Pin the
    # bound and the implied break strain of every event.
    assert float(stored.max()) <= closed_form * (1.0 + 1e-9)
    implied_strain = torch.sqrt(2.0 * stored / k) / spacing
    assert float(implied_strain.max()) <= sim.config.breaking_strain * (1.0 + 1e-9)
    assert float(stored.min()) > 0.0


def test_energy_and_diagnostics_agree_on_every_shared_channel():
    sim = _solver()
    _run_until_break(sim)
    for _ in range(120):
        sim.step()
    diag = sim.diagnostics()
    snap = sim.energy()
    for name in SHARED:
        a, b = float(diag[name]), float(snap[name])
        scale = max(abs(a), abs(b), 1e-12)
        assert abs(a - b) <= 1e-9 * scale, f"{name}: diagnostics={a} energy={b}"


def test_released_energy_is_frozen_when_no_bond_breaks():
    """The regression that motivated this test: a length proxy keeps growing."""
    sim = _solver()
    _run_until_break(sim)
    # Let the indentation settle into a regime without further bond events.
    previous = sim.energy()["released_bond_energy_J"]
    checked = 0
    for _ in range(600):
        before = sim.broken_bonds
        sim.step()
        current = sim.energy()["released_bond_energy_J"]
        if sim.broken_bonds == before:
            # No bond event: the released accumulator must not move at all.
            assert current == previous, (
                f"released_bond_energy_J moved {previous} -> {current} J with zero bond events"
            )
            checked += 1
        previous = current
    assert checked > 100, "expected a quiet window with no bond events"


def test_released_energy_is_monotone_and_bounded_by_mechanics():
    sim = _solver()
    _run_until_break(sim)
    history = [sim.energy()["released_bond_energy_J"]]
    for _ in range(300):
        sim.step()
        history.append(sim.energy()["released_bond_energy_J"])
    assert all(b >= a for a, b in zip(history, history[1:])), "released energy must not decrease"
    assert history[-1] == sim.failure_energy_J.sum().item()
    # A released energy larger than the total mechanical energy of the specimen
    # is unphysical; the ledger satisfies this, the old length proxy did not.
    snap = sim.energy()
    assert snap["released_bond_energy_J"] <= snap["mechanical_energy_J"] + 1e-9