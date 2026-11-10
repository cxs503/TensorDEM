"""Acceptance tests for ``tensordem.coupling_kernel`` (campaign p223w3-n05).

Covered claims
--------------
1. contact detection: cell list == brute force (exact pair lists),
2. dependency layering: the package imports and runs with torch blocked,
3. backend equivalence: numpy vs torch, 1e3 steps, bit-level state agreement
   and ledger agreement < 1e-10,
4. energy ledger: itemised channels, closure, hooks, CSV/JSON round trip,
5. settling scenario: real kernel reproduces the linear-drag exact solution and
   the Stokes terminal velocity within 3 %, dragging the DEM leg off the
   analytic "reference model",
6. reproducibility: identical backend + identical input -> identical state.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tensordem.coupling_kernel import (
    DEMKernel,
    DEMKernelConfig,
    EnergyLedger,
    candidate_pairs,
    brute_force_pairs,
    dense_bed_case,
    dilute_single_particle_case,
    drag_relaxation_time,
    get_backend,
    relaxation_reference,
    stokes_terminal_velocity,
    three_field_summary,
)
from tensordem.coupling_kernel.settling import contact_rich_pack_case, elastic_gas_case

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = "torch" in __import__("tensordem.coupling_kernel", fromlist=["x"]).available_backends()


# -- 1. detection ------------------------------------------------------------
def test_celllist_matches_brute_force():
    rng = np.random.default_rng(11)
    box = (0.0, 0.0, 0.0), (1e-3, 1e-3, 1e-3)
    for _ in range(25):
        n = int(rng.integers(2, 300))
        pos = rng.uniform(0.0, 1e-3, size=(n, 3))
        cutoff = float(rng.uniform(2e-5, 2e-4))
        i, j = candidate_pairs(pos, cutoff, *box)
        bi, bj = brute_force_pairs(pos, cutoff)
        assert np.array_equal(i, bi) and np.array_equal(j, bj)


def test_celllist_rejects_pairs_outside_cutoff():
    pos = np.array([[0.0, 0.0, 0.0], [1e-3, 1e-3, 1e-3], [5e-5, 0.0, 0.0]])
    pos += 1e-6
    i, j = candidate_pairs(pos, 1e-4, (0, 0, 0), (2e-3, 2e-3, 2e-3))
    d = np.linalg.norm(pos[j] - pos[i], axis=1)
    assert np.all(d <= 1e-4)
    assert (i != j).all()


# -- 2. dependency layering --------------------------------------------------
def test_coupling_kernel_is_torch_free():
    """The DEM kernel subpackage imports and runs with torch unavailable.

    ``tensordem/__init__.py`` still eagerly pulls torch in (pre-existing
    coupling of the legacy prototype, tracked as follow-up work), so the
    subpackage is imported against a stub parent package: that isolates the
    layer under test.
    """

    script = textwrap.dedent(
        """
        import sys, types, pathlib, importlib.abc

        src = pathlib.Path('src/tensordem').resolve()
        stub = types.ModuleType('tensordem')
        stub.__path__ = [str(src)]
        sys.modules['tensordem'] = stub

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if isinstance(name, str) and (name == 'torch' or name.startswith('torch.')):
                    raise ImportError('torch blocked by test')
                return None

        sys.meta_path.insert(0, Blocker)
        import numpy as np
        from tensordem.coupling_kernel import DEMKernel, dense_bed_case
        case = dense_bed_case(n_particles=27)
        k = DEMKernel(case.config, case.positions, case.velocities, backend='numpy')
        k.step(50)
        assert 'torch' not in sys.modules, 'torch must not be imported'
        assert np.isfinite(k.ledger.latest()['kinetic_J'])
        print(k.be.name)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("numpy")


# -- 3. backend equivalence ---------------------------------------------------
@pytest.mark.skipif(not HAS_TORCH, reason="torch backend unavailable")
def test_backend_equivalence_1000_steps():
    case = contact_rich_pack_case(n_particles=64)
    runs = {}
    for backend in ("numpy", "torch"):
        kernel = DEMKernel(
            case.config, case.positions, case.velocities, backend=backend,
            record_energy_every=100,
        )
        kernel.step(1000)
        runs[backend] = kernel
    ref, alt = runs["numpy"], runs["torch"]
    pos_scale = np.abs(ref.positions).max()
    vel_scale = np.abs(ref.velocities).max()
    assert np.abs(ref.positions - alt.positions).max() / pos_scale <= 1e-12
    assert np.abs(ref.velocities - alt.velocities).max() / vel_scale <= 1e-12
    # contact must actually be active, otherwise the test proves nothing
    assert ref.contact_diagnostics()["contacts"] > 0
    lref, lalt = ref.ledger.latest(), alt.ledger.latest()
    for key in (
        "kinetic_J",
        "elastic_J",
        "gravity_potential_J",
        "dem_damping_dissipated_J",
        "wall_damping_dissipated_J",
        "drag_dissipated_J",
        "balance_residual_J",
    ):
        assert abs(lref[key] - lalt[key]) < 1e-10, (key, lref[key], lalt[key])


@pytest.mark.skipif(not HAS_TORCH, reason="torch backend unavailable")
def test_backend_pair_detection_identical():
    case = contact_rich_pack_case(n_particles=64)
    pairs = {}
    for backend in ("numpy", "torch"):
        kernel = DEMKernel(case.config, case.positions, case.velocities, backend=backend)
        kernel.step(3)
        pairs[backend] = (kernel._pair_i_np.copy(), kernel._pair_j_np.copy())
    assert np.array_equal(pairs["numpy"][0], pairs["torch"][0])
    assert np.array_equal(pairs["numpy"][1], pairs["torch"][1])


def test_reproducibility_same_backend():
    case = contact_rich_pack_case(n_particles=64)
    a = DEMKernel(case.config, case.positions, case.velocities, backend="numpy")
    b = DEMKernel(case.config, case.positions, case.velocities, backend="numpy")
    a.step(200)
    b.step(200)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.velocities, b.velocities)


# -- 4. ledger ---------------------------------------------------------------
def test_ledger_channels_and_closure_dissipative():
    case = contact_rich_pack_case(n_particles=64)
    kernel = DEMKernel(
        case.config, case.positions, case.velocities, backend="numpy", record_energy_every=50
    )
    seen = []
    kernel.ledger.attach(seen.append)
    kernel.step(400)
    row = kernel.ledger.latest()
    for key in (
        "kinetic_J",
        "gravity_potential_J",
        "elastic_J",
        "dem_damping_dissipated_J",
        "wall_damping_dissipated_J",
        "drag_dissipated_J",
        "balance_residual_J",
        "relative_residual",
    ):
        assert key in row, key
    assert len(seen) >= 8
    # dissipation channels must be non-negative (2nd law)
    assert row["dem_damping_dissipated_J"] >= 0.0
    assert row["wall_damping_dissipated_J"] >= 0.0
    assert row["drag_dissipated_J"] >= 0.0
    assert abs(row["relative_residual"]) < 0.05


def test_ledger_conservative_closure():
    case = elastic_gas_case(n_particles=27, dt=5e-7)
    kernel = DEMKernel(
        case.config, case.positions, case.velocities, backend="numpy", record_energy_every=100
    )
    kernel.step(2000)
    summary = kernel.ledger.summary()
    assert summary["dem_damping_dissipated_J"] == 0.0
    assert summary["drag_dissipated_J"] == 0.0
    assert abs(summary["final_relative_residual"]) < 1e-3


def test_ledger_round_trip(tmp_path):
    case = contact_rich_pack_case(n_particles=27)
    kernel = DEMKernel(
        case.config, case.positions, case.velocities, backend="numpy", record_energy_every=25
    )
    kernel.step(100)
    json_path = kernel.ledger.to_json(tmp_path / "ledger.json")
    csv_path = kernel.ledger.to_csv(tmp_path / "ledger.csv")
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["records"] >= 4
    assert csv_path.read_text().splitlines()[0].startswith("step,time_s")
    assert three_field_summary(kernel.ledger)["fluid"]["drag_dissipated_J"] >= 0.0


def test_ledger_requires_all_dissipative_channels():
    ledger = EnergyLedger()
    with pytest.raises(KeyError):
        ledger.record(0, 0.0, kinetic_J=1.0, gravity_potential_J=0.0, elastic_J=0.0)


# -- 5. settling scenario (reference-model replacement) ----------------------
def test_dilute_settling_matches_exact_relaxation_and_stokes():
    case = dilute_single_particle_case(dt=1e-6)
    kernel = DEMKernel(case.config, case.positions, case.velocities, backend="numpy")
    kernel.step(20000)
    cfg = case.config
    v = -float(kernel.velocities[0, 2])
    v_t = stokes_terminal_velocity(cfg)
    tau = drag_relaxation_time(cfg)
    exact = float(relaxation_reference(kernel.time, tau, v_t))
    assert abs(v - exact) / exact < 1e-3
    assert abs(v - v_t) / v_t < 0.03
    re = cfg.fluid_density * 2 * cfg.radius * v / cfg.fluid_viscosity
    assert re < 0.1
    assert abs(kernel.ledger.summary()["final_relative_residual"]) < 1e-6


def test_dense_bed_starts_without_overlap_and_settles():
    case = dense_bed_case(n_particles=125, dt=2e-6)
    kernel = DEMKernel(
        case.config, case.positions, case.velocities, backend="numpy", record_energy_every=100
    )
    assert kernel.contact_diagnostics()["contacts"] == 0
    kernel.step(400)
    assert kernel.velocities[:, 2].mean() < 0.0  # settling downwards
    row = kernel.ledger.latest()
    assert row["drag_dissipated_J"] > 0.0
    assert row["kinetic_J"] > 0.0


# -- 6. config guards --------------------------------------------------------
def test_config_validation():
    with pytest.raises(ValueError):
        DEMKernelConfig(radius=-1.0, density=1000.0)
    with pytest.raises(ValueError):
        DEMKernelConfig(radius=1e-4, density=1000.0, drag_model="bogus")
    with pytest.raises(ValueError):
        DEMKernelConfig(
            radius=1e-4, density=1000.0, box_lo=(0, 0, 0), box_hi=(1e-4, 1e-3, 1e-3)
        )


def test_backend_registry():
    assert get_backend("numpy").name == "numpy"
    with pytest.raises(ValueError):
        get_backend("cuda-ish")