"""Integration caller + entry-point tests for the DEM erosion face (p176w2-n11).

These tests guard the *coupling* obligations that the standalone verification
face does not cover:

* the recorder books exactly what the solver sees and its ledger closes;
* the erosion observer cannot move the flow (bitwise identical trajectory);
* the main solver (slurry pipe DEM-CFD) really drives the recorder;
* the CLI entry points work and emit computable JSON;
* the reference replay decomposition is an algebraic identity.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from tensordem.erosion.materials import get_material
from tensordem.erosion.models import dnv_angle_function, dnv_ductile_erosion_ratio
from tensordem.erosion_dem.cli import main as cli_main
from tensordem.erosion_dem.crater import OkaAnchoredLaw
from tensordem.erosion_dem.recorder import ImpactErosionRecorder
from tensordem.slurry_transport.pipe_dem import PipeSectionDEMCFD

WALL = get_material("carbon_steel")
DNV_K = 2.0e-9
DNV_N = 2.6
V_REF = 30.0
D_M = 4.0e-3


def _law(diameter_m: float = D_M) -> OkaAnchoredLaw:
    anchor = dnv_ductile_erosion_ratio(V_REF, 90.0, WALL, k=DNV_K, n=DNV_N)
    return OkaAnchoredLaw.for_material(
        "carbon_steel", anchor, v_ref_m_s=V_REF, alpha_ref_deg=90.0,
        d_ref_um=diameter_m * 1e6,
    )


def _checksum(pipe: PipeSectionDEMCFD) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(pipe.pos, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(pipe.vel, dtype=np.float64).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# recorder (the sink of the coupling)
# ---------------------------------------------------------------------------
def test_zero_speed_books_nothing():
    rec = ImpactErosionRecorder(material=WALL, law=_law())
    assert rec.record(0.0, 90.0, D_M, 2650.0) is None
    assert rec.n_impacts == 0
    assert rec.erosion_ratio_kg_kg == 0.0


def test_zero_impacts_gives_exact_zero_ratio():
    rec = ImpactErosionRecorder(material=WALL, law=_law())
    assert rec.state()["erosion_ratio_kg_kg"] == 0.0
    assert rec.ledger_residual == 0.0


def test_ledger_closes_and_masses_stay_non_negative():
    rec = ImpactErosionRecorder(material=WALL, law=_law())
    for v, a in [(1.0, 5.0), (2.0, 15.0), (3.0, 45.0), (5.0, 90.0)]:
        rec.record(v, a, D_M, 2650.0)
    st = rec.state()
    assert st["n_impacts"] == 4
    assert st["ledger_residual"] < 1e-12
    assert st["masses_non_negative"] is True
    assert st["removed_mass_kg"] < st["contact_mass_kg"]
    assert st["removed_volume_m3"] > 0.0
    assert st["max_crater_depth_m"] <= 0.5 * D_M


def test_removed_mass_increases_with_speed():
    removed = []
    for v in (1.0, 2.0, 4.0, 8.0):
        rec = ImpactErosionRecorder(material=WALL, law=_law())
        rec.record(v, 30.0, D_M, 2650.0)
        removed.append(rec.removed_mass_kg)
    assert all(b > a for a, b in zip(removed, removed[1:]))


def test_wall_loss_rate_is_the_dnv_conversion():
    rec = ImpactErosionRecorder(material=WALL, law=_law())
    for _ in range(10):
        rec.record(2.0, 10.0, D_M, 2650.0)
    mdot, area = 0.5, 0.01
    got = rec.wall_loss_mm_per_year(mdot, area_m2=area)
    rate = rec.erosion_ratio_kg_kg * mdot / (WALL.density_kg_m3 * area)
    want = rate * 365.25 * 24.0 * 3600.0 * 1000.0
    assert got == pytest.approx(want, rel=1e-12)


def test_replay_matches_direct_dnv_evaluation():
    rec = ImpactErosionRecorder(material=WALL, law=_law())
    for v, a in [(1.0, 5.0), (2.0, 20.0), (3.0, 60.0)]:
        rec.record(v, a, D_M, 2650.0)
    out = rec.replay(lambda v, a: dnv_ductile_erosion_ratio(v, a, WALL,
                                                            k=DNV_K, n=DNV_N))
    assert out["n_replayed"] == 3
    assert out["reference_erosion_ratio_kg_kg"] > 0.0
    assert 0.0 <= out["per_impact_mean_relative_deviation"] < 1.0


def test_replay_decomposition_is_an_identity():
    """model/dnv = (v/v_ref)**(k-n) * [g(a)/g(a_ref)] / [F(a)/F(a_ref)]."""
    law = _law()
    rec = ImpactErosionRecorder(material=WALL, law=law)
    for v, a in [(1.0, 5.0), (1.5, 12.0), (2.5, 40.0), (4.0, 90.0)]:
        rec.record(v, a, D_M, 2650.0)
    k = law.velocity_exponent
    for r in rec.records:
        model = float(law.ratio(r["speed_m_s"], r["alpha_deg"], D_M * 1e6))
        dnv = float(dnv_ductile_erosion_ratio(r["speed_m_s"], r["alpha_deg"], WALL,
                                              k=DNV_K, n=DNV_N))
        speed = (r["speed_m_s"] / V_REF) ** (k - DNV_N)
        shape = float(law.angle_factor(r["alpha_deg"])) / float(
            dnv_angle_function(r["alpha_deg"], DNV_N)
        )
        assert speed * shape == pytest.approx(model / dnv, rel=1e-12)


def test_recorder_rejects_perforating_impacts():
    """A crater deeper than the particle radius is refused, not extrapolated."""
    rec = ImpactErosionRecorder(material=WALL, law=_law())
    # absurd speed -> removed volume does not fit inside the particle radius
    assert rec.record(1.0e5, 90.0, D_M, 2650.0) is None
    assert rec.n_rejected == 1
    assert rec.n_impacts == 0
    assert rec.removed_mass_kg == 0.0


# ---------------------------------------------------------------------------
# main solver coupling (the integration caller)
# ---------------------------------------------------------------------------
def _run_pipe(erosion, u_mix=2.0, steps=120, seed=0):
    pipe = PipeSectionDEMCFD(D=0.10, d=D_M, erosion=erosion, seed=seed)
    pipe.initialize(u_mix)
    for _ in range(steps):
        pipe.step(2e-4)
    return pipe


def test_coupling_is_an_observer_not_a_force():
    """Trajectory must be bitwise identical with and without the recorder."""
    plain = _run_pipe(None)
    rec = ImpactErosionRecorder(material=WALL, law=_law())
    coupled = _run_pipe(rec)
    assert rec.n_impacts > 0, "the pipe run must actually produce wall impacts"
    assert _checksum(plain) == _checksum(coupled)


def test_pipe_books_impacts_and_closes_the_ledger():
    rec = ImpactErosionRecorder(material=WALL, law=_law())
    pipe = _run_pipe(rec)
    st = rec.state()
    assert st["n_impacts"] > 100
    assert st["removed_mass_kg"] > 0.0
    assert st["ledger_residual"] < 1e-9
    assert 0.0 < st["mean_impact_angle_deg"] <= 90.0
    assert st["mean_impact_speed_m_s"] > 0.0
    # impacts are booked on transitions into contact, not on every step
    assert st["n_impacts"] < 100 * pipe.pos.shape[0]


def test_pipe_erosion_grows_with_mixture_velocity():
    removed = []
    for u in (1.5, 2.5):
        rec = ImpactErosionRecorder(material=WALL, law=_law())
        _run_pipe(rec, u_mix=u, steps=120)
        removed.append(rec.removed_mass_kg)
    assert removed[1] > removed[0]


def test_uncoupled_pipe_results_unchanged():
    """The default (erosion=None) path must stay numerically inert."""
    a = _run_pipe(None)
    b = _run_pipe(None)
    assert _checksum(a) == _checksum(b)


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def test_cli_impact_emits_json(capsys):
    rc = cli_main(["impact", "--speed", "30", "--angle", "90"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "impact"
    assert payload["physically_admissible"] is True
    assert payload["removed_volume_m3"] > 0.0


def test_cli_stream_emits_json(capsys):
    rc = cli_main(["stream", "--angle", "20", "--n-particles", "40"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "stream"
    assert payload["p_delivered"] if "p_delivered" in payload else True
    assert payload["impacts"] > 0
    assert payload["erosion_ratio_kg_kg"] > 0.0


def test_cli_pipe_emits_json(capsys):
    rc = cli_main(["pipe", "--speed", "2.0", "--seconds", "0.01"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "pipe"
    assert payload["law"] == "OkaAnchoredLaw"
    assert payload["n_impacts"] >= 0


def test_module_entry_point_runs():
    """``python -m tensordem.erosion_dem`` must exist and print JSON."""
    import os

    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    out = subprocess.run(
        [sys.executable, "-m", "tensordem.erosion_dem", "impact",
         "--speed", "30", "--angle", "90"],
        capture_output=True, text=True, env=env, timeout=180,
    )
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload["crater_depth_m"] > 0.0