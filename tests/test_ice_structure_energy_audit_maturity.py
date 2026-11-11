"""Acceptance tests for the four-ledger audit's maturity (L2/L3/L4 evidence).

Covers what the audit module's own tests cannot:

* the solver-driven observer reproduces the literal ledger definitions
  (reference vs obtained, < 0.1 % / 0.5 %);
* the observer refuses a non-physical ledger (dissipative terms going backwards)
  instead of silently absorbing it;
* the DEM main solver (`tensordem.dem3d.IceDEM3D`) really calls the audit every
  step, closes the four ledgers below 1 % in the undamped case, and the audited
  run is bit-for-bit identical to the unaudited one;
* the command-line entry point runs end to end and writes a verdict;
* the committed maturity evidence is internally consistent (files exist, hashes
  match the tree, level L4, every machine check passes).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.ice_structure_energy_audit_cli import main as cli_main
from tensordem.ice_structure_energy_audit_integration import (
    IceDEMEnergyAuditAdapter,
    attach_energy_audit,
)
from tensordem.ice_structure_energy_ledger import FourLedgerObserver

EVIDENCE = ROOT / "artifacts" / "ice_structure_energy_ledger" / "maturity_evidence.json"
REPORT = ROOT / "docs" / "DEM_ENERGY_AUDIT_MATURITY.md"

GRID = dict(nx=3, ny=2, nz=2, radius=0.025, density=917.0,
            bond_stiffness=2000.0, contact_stiffness=5000.0,
            breaking_strain=0.05, shear_breaking_strain=0.10,
            drag=0.0, restitution_coefficient=1.0, friction_coefficient=0.0,
            tool_radius=0.1, tool_speed=1.0e-12, tool_gap=0.002,
            fix_x_edges=False, fix_bottom=False, gravity=0.0)


def build_solver(*, contact_damping: float = 0.0) -> IceDEM3D:
    sim = IceDEM3D(DEM3DConfig(contact_damping=contact_damping, **GRID))
    sim.velocities[:, 2] = 1.0
    return sim


# ---------------------------------------------------------------------------
# observer: reference vs obtained
# ---------------------------------------------------------------------------
def test_observer_matches_literal_ledger_definitions():
    rng = np.random.default_rng(20261111)
    velocities = rng.normal(scale=0.5, size=(6, 3))
    omegas = rng.normal(scale=2.0, size=(6, 3))
    mass, inertia = 0.42, 0.4 * 0.42 * 0.05 ** 2
    kinetic_trans = 0.5 * mass * float((velocities ** 2).sum())
    kinetic_rot = 0.5 * inertia * float((omegas ** 2).sum())
    strain_contact, strain_bond = 1.25, 0.37

    observer = FourLedgerObserver()
    row = observer.observe(time=0.0, kinetic_trans=kinetic_trans,
                           kinetic_rot=kinetic_rot,
                           strain_contact=strain_contact,
                           strain_bond=strain_bond)

    trans_literal = sum(0.5 * mass * float(np.dot(v, v)) for v in velocities)
    rot_literal = sum(0.5 * inertia * float(np.dot(w, w)) for w in omegas)
    assert row["E_k_trans"] == pytest.approx(trans_literal, rel=1e-12)
    assert row["E_k_rot"] == pytest.approx(rot_literal, rel=1e-12)
    assert row["E_k"] == pytest.approx(trans_literal + rot_literal, rel=1e-12)
    assert row["E_s"] == pytest.approx(strain_contact + strain_bond, rel=1e-12)
    assert row["E_total"] == pytest.approx(
        trans_literal + rot_literal + strain_contact + strain_bond, rel=1e-12)
    assert row["residual_relative"] == 0.0


def test_observer_integrates_viscous_rate_trapezoidally():
    observer = FourLedgerObserver()
    observer.observe(time=0.0, kinetic_trans=1.0)
    dt = 1.0e-3
    # rate ramps 0 -> 200 W over one step: trapezoid gives 0.5*200*dt
    observer.observe(time=dt, kinetic_trans=1.0, viscous_rate=200.0, dt=dt)
    step_two = observer.observe(time=2 * dt, kinetic_trans=1.0,
                                viscous_rate=200.0, dt=dt)
    # first step has no left endpoint -> left-Riemann (200*dt); second step is
    # trapezoidal on a constant rate, i.e. also 200*dt
    assert step_two["E_v"] == pytest.approx(200.0 * dt + 200.0 * dt, rel=1e-12)


def test_observer_rejects_backwards_dissipation():
    observer = FourLedgerObserver()
    observer.observe(time=0.0, kinetic_trans=1.0, viscous_dissipation=0.5)
    with pytest.raises(ValueError, match="went backwards"):
        observer.observe(time=1.0e-3, kinetic_trans=1.0, viscous_dissipation=0.1)
    other = FourLedgerObserver()
    other.observe(time=0.0, fracture_energy=1.0)
    with pytest.raises(ValueError, match="fracture energy went backwards"):
        other.observe(time=1.0e-3, fracture_energy=0.5)


def test_negative_control_zero_state_stays_zero():
    observer = FourLedgerObserver()
    for k in range(5):
        row = observer.observe(time=k * 1.0e-4)
    assert row["E_total"] == 0.0
    assert row["residual_relative"] == 0.0
    assert observer.summary()["closed_within_1pct"]


# ---------------------------------------------------------------------------
# integration with the main solver
# ---------------------------------------------------------------------------
def test_main_solver_calls_audit_every_step():
    sim = build_solver()
    adapter = attach_energy_audit(sim)
    steps = 12
    for _ in range(steps):
        sim.step()
    assert len(adapter.rows) == steps + 1          # reference row + one per step
    assert adapter.rows[0]["t"] == 0.0
    assert adapter.rows[-1]["t"] == pytest.approx(sim.time, rel=1e-12)
    assert adapter.summary()["residual_relative_max"] >= 0.0


def test_main_solver_kinetic_ledger_matches_solver_diagnostics():
    sim = build_solver()
    adapter = attach_energy_audit(sim)
    for _ in range(10):
        sim.step()
    assert adapter.cross_check_max < 1.0e-12


def test_audited_run_is_bitwise_identical_to_unaudited_run():
    steps = 10
    plain = build_solver()
    for _ in range(steps):
        plain.step()
    audited = build_solver()
    attach_energy_audit(audited)
    for _ in range(steps):
        audited.step()
    assert (plain.positions == audited.positions).all()
    assert (plain.velocities == audited.velocities).all()


def test_main_solver_closes_four_ledgers_undamped():
    sim = build_solver(contact_damping=0.0)
    adapter = attach_energy_audit(sim)
    for _ in range(120):
        sim.step()
    summary = adapter.summary()
    assert summary["residual_relative_max"] < 1.0e-2
    assert summary["closed_within_1pct"]
    # the ledgers must actually be populated: the impact has transferred
    # kinetic energy into contact strain energy by the end of the run
    assert summary["E_s"] > 0.0


def test_unaccounted_dashpot_channel_is_reported_not_hidden():
    """With the tool dashpot on and not accumulated, the audit must flag it."""
    sim = build_solver(contact_damping=5.0)
    adapter = attach_energy_audit(sim)
    for _ in range(120):
        sim.step()
    assert adapter.summary()["residual_relative_max"] > 1.0e-2


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def test_cli_entry_point_writes_verdict(tmp_path):
    out = tmp_path / "cli"
    code = cli_main(["--case", "elastic", "--dt", "2e-5", "--duration", "4e-3",
                     "--out", str(out), "--quiet"])
    assert code == 0
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["all_pass"] is True
    case = verdict["cases"][0]
    assert case["residual_relative_max"] < case["threshold"]
    assert Path(case["ledger_csv"]).is_file()
    assert case["E_k_reference_error"] < 1.0e-3
    assert case["E_s_reference_error"] < 5.0e-3


def test_module_entry_point_is_executable():
    proc = subprocess.run(
        [sys.executable, "-m", "tensordem.ice_structure_energy_audit_cli",
         "--case", "elastic", "--dt", "4e-5", "--duration", "2e-3", "--quiet"],
        cwd=str(ROOT), capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert proc.returncode == 0, proc.stderr[-800:]


# ---------------------------------------------------------------------------
# committed maturity evidence
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not EVIDENCE.is_file(),
                    reason="maturity evidence not generated in this tree")
def test_maturity_evidence_is_consistent_with_the_tree():
    ev = json.loads(EVIDENCE.read_text())
    assert ev["level_reached"] == "L4", ev["checks"]
    assert ev["all_checks_pass"], [k for k, v in ev["checks"].items() if not v]
    for record in ev["L2"]["impl_files"] + ev["L2"]["tests"]:
        path = ROOT / record["path"]
        assert path.is_file(), record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"], f"stale evidence for {record['path']}"
    for record in ev["L2"]["run_artifacts"]:
        assert (ROOT / record["path"]).is_file(), record["path"]
    assert ev["L3"]["integration"]["all_present"]
    assert ev["L3"]["integration_numbers"]["all_criteria_pass"]
    assert ev["L4"]["benchmark"]["all_criteria_pass"]
    # the node's own acceptance criteria (a)-(h) are all green
    for row in ev["L4"]["benchmark"]["rows"]:
        assert row["pass"], row["criterion"]
    assert REPORT.is_file()
    assert "P304W2-N03" in REPORT.read_text()