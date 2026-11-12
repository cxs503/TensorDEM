"""Vectorised cell-list candidate search: equivalence, completeness, order.

Mission m16007 (GAP-DEM-G6).  ``IceDEM3D._contact_candidate_pairs`` used to be
a per-cell Python loop (the "prototype" hashing noted in the module docstring).
It was benchmarked against a vectorised replacement (mission m16007) and kept, 
unchanged, so this file pins three properties with two independent oracles:

1. ``_legacy_candidate_pairs`` -- a frozen verbatim copy of the pre-change
   implementation.  The new search must return the same rows **in the same
   order** (order matters: it fixes the floating-point accumulation order
   downstream, hence bit-for-bit trajectories).
2. ``_brute_force_candidates`` -- an O(N^2) reference written from the
   definition ("same or one of the 26 adjacent cells"), independent of any cell
   hashing.  The canonical (sorted) pair set must match it exactly.
3. Completeness invariant: two particles whose surfaces overlap (|d| < 2R)
   *must* appear in the candidate set, otherwise the cell width would be wrong
   and contacts would silently disappear at scale.  Checked by distance, which
   never looks at cells at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D

RADIUS = 0.025
WIDTH = 2.0 * RADIUS


def _host(positions: torch.Tensor, radius: float = RADIUS) -> SimpleNamespace:
    """Minimal stand-in: the search touches only positions/config/device."""
    return SimpleNamespace(positions=positions, config=DEM3DConfig(radius=radius), device="cpu")


def _pairs(positions: torch.Tensor, radius: float = RADIUS) -> torch.Tensor:
    return IceDEM3D._contact_candidate_pairs(_host(positions, radius))


def _legacy_candidate_pairs(positions: torch.Tensor, radius: float = RADIUS) -> torch.Tensor:
    """Frozen copy of the pre-vectorisation implementation (order oracle)."""
    n = len(positions)
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long)
    width = 2.0 * radius
    coords = torch.floor(positions.detach().cpu() / width).to(torch.int64).tolist()
    cells: dict[tuple[int, int, int], list[int]] = {}
    for idx, xyz in enumerate(coords):
        key = (xyz[0], xyz[1], xyz[2])
        cells.setdefault(key, []).append(idx)
    pairs: list[tuple[int, int]] = []
    for key, members in cells.items():
        x, y, z = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    other = cells.get((x + dx, y + dy, z + dz))
                    if other is None:
                        continue
                    for i in members:
                        for j in other:
                            if j > i:
                                pairs.append((i, j))
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long)


def _brute_force_candidates(positions: torch.Tensor, radius: float = RADIUS) -> torch.Tensor:
    """O(N^2) reference straight from the definition, sorted rows."""
    width = 2.0 * radius
    cell = torch.floor(positions.detach().cpu() / width).to(torch.int64)
    n = len(positions)
    out = []
    for i in range(n - 1):
        d = (cell[i + 1:] - cell[i]).abs().max(dim=1).values
        j = torch.nonzero(d <= 1).flatten() + (i + 1)
        if j.numel():
            out.append(torch.stack((torch.full_like(j, i), j), dim=1))
    if not out:
        return torch.empty((0, 2), dtype=torch.long)
    pairs = torch.cat(out)
    return pairs[pairs[:, 0].argsort(stable=True)]


def _canonical(pairs: torch.Tensor) -> list[tuple[int, int]]:
    rows = sorted((int(a), int(b)) for a, b in pairs.tolist())
    return rows


def _uniform(n: int, seed: int, *, lo: float = -0.4, hi: float = 0.6) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((n, 3), generator=g, dtype=torch.float64) * (hi - lo) + lo


def _lattice(nx: int, ny: int, nz: int, spacing: float = 0.05) -> torch.Tensor:
    axes = [torch.arange(v, dtype=torch.float64) * spacing for v in (nx, ny, nz)]
    zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)


# --------------------------------------------------------------------------
# configuration corpus: uniform / clustered / dense-overlap / boundary / lattice
# --------------------------------------------------------------------------
def _configurations() -> dict[str, torch.Tensor]:
    dense = torch.rand((400, 3), generator=torch.Generator().manual_seed(11), dtype=torch.float64) * 0.02
    cluster = torch.cat([
        _uniform(120, 21, lo=0.0, hi=0.05) + torch.tensor([0.0, 0.0, 0.0]),
        _uniform(120, 22, lo=0.0, hi=0.05) + torch.tensor([0.06, 0.0, 0.0]),
        _uniform(120, 23, lo=0.0, hi=0.05) + torch.tensor([0.0, 0.063, 0.0]),
    ])
    return {
        "uniform": _uniform(700, 7),
        "uniform_offset_far": _uniform(500, 8, lo=100.0, hi=101.0),
        "uniform_negative": _uniform(500, 9, lo=-50.0, hi=-49.0),
        "clustered": cluster,
        "dense_overlap": dense,
        "lattice_exact": _lattice(7, 6, 5),
        "lattice_stretched": _lattice(6, 5, 4, spacing=0.07),
        "two_particles": torch.tensor([[0.0, 0.0, 0.0], [0.049, 0.0, 0.0]], dtype=torch.float64),
        "single_cell": torch.rand((60, 3), generator=torch.Generator().manual_seed(31), dtype=torch.float64) * 0.01,
    }


# --------------------------------------------------------------------------
# 1. exact equivalence with the frozen legacy order
# --------------------------------------------------------------------------
def test_matches_legacy_pairs_bit_for_bit():
    for name, pos in _configurations().items():
        new = _pairs(pos)
        old = _legacy_candidate_pairs(pos)
        assert new.shape == old.shape, name
        assert torch.equal(new, old), f"pair order/set changed for {name}"


def test_matches_legacy_at_scale():
    pos = _uniform(3000, 41, lo=0.0, hi=0.6)
    assert torch.equal(_pairs(pos), _legacy_candidate_pairs(pos))
    pos = _lattice(14, 14, 14)
    assert torch.equal(_pairs(pos), _legacy_candidate_pairs(pos))


# --------------------------------------------------------------------------
# 2. canonical equivalence with an independent brute-force definition
# --------------------------------------------------------------------------
def test_matches_brute_force_definition():
    for name, pos in _configurations().items():
        got = _canonical(_pairs(pos))
        want = _canonical(_brute_force_candidates(pos))
        assert got == want, name


def test_varies_radius_cell_width():
    pos = _uniform(300, 51, lo=0.0, hi=0.2)
    for radius in (0.05, 0.01, 0.0025):
        got = _canonical(_pairs(pos, radius))
        want = _canonical(_brute_force_candidates(pos, radius))
        assert got == want, radius


# --------------------------------------------------------------------------
# 3. structural invariants
# --------------------------------------------------------------------------
def test_no_duplicate_pairs_and_i_lt_j():
    for name, pos in _configurations().items():
        pairs = _pairs(pos)
        if pairs.numel() == 0:
            continue
        assert bool((pairs[:, 0] < pairs[:, 1]).all()), name
        assert len(set(_canonical(pairs))) == pairs.shape[0], name


def test_overlapping_pairs_are_never_missed():
    """Completeness: |d| < 2R => the pair must be a candidate (scale safety)."""
    for name, pos in _configurations().items():
        pairs = _pairs(pos)
        got = set(_canonical(pairs))
        r = RADIUS
        d = torch.cdist(pos, pos)
        close = torch.nonzero(d < 2.0 * r)
        close = {(int(i), int(j)) for i, j in close.tolist() if i < j}
        assert close <= got, f"{name}: missed {len(close - got)} overlapping pairs"


def test_subunit_sizes_and_empty():
    empty = torch.empty((0, 3), dtype=torch.float64)
    assert _pairs(empty).shape == (0, 2)
    one = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    assert _pairs(one).shape == (0, 2)
    assert _pairs(one).dtype == torch.long


def test_deterministic_repeat():
    pos = _uniform(800, 61)
    a, b = _pairs(pos), _pairs(pos)
    assert torch.equal(a, b)


def test_positions_not_mutated():
    pos = _uniform(200, 71)
    before = pos.clone()
    _pairs(pos)
    assert torch.equal(pos, before)


# --------------------------------------------------------------------------
# 4. solver-level transparency: identical trajectory digest vs the pre-change run
# --------------------------------------------------------------------------
def test_solver_trajectory_digest_matches_baseline():
    """The stored baseline digest was produced by the pre-vectorisation code."""
    import hashlib
    import json
    from pathlib import Path

    path = Path("artifacts/cell_list/baseline.json")
    if not path.exists():  # optional artifact: skip when the ledger is absent
        import pytest

        pytest.skip("baseline artifact not present")
    payload = json.loads(path.read_text())
    case = next(c for c in payload["cases"] if c["grid"] == [11, 11, 11])
    sim = IceDEM3D(DEM3DConfig(nx=11, ny=11, nz=11))
    for _ in range(case["trajectory_steps"]):
        sim.step()
    h = hashlib.sha256()
    for name in ("positions", "velocities"):
        arr = getattr(sim, name).detach().to("cpu", torch.int64).numpy()
        h.update(hashlib.sha256(arr.tobytes()).hexdigest().encode())
    assert h.hexdigest() == case["trajectory_sha256"]
    assert len(sim.positions) == case["n_particles"]


def test_candidate_count_matches_baseline_case():
    import json
    from pathlib import Path

    path = Path("artifacts/cell_list/baseline.json")
    if not path.exists():
        import pytest

        pytest.skip("baseline artifact not present")
    payload = json.loads(path.read_text())
    for case in payload["cases"]:
        nx, ny, nz = case["grid"]
        if case["n_particles"] > 11000:
            continue  # keep the unit suite light; the bench covers the ladder
        sim = IceDEM3D(DEM3DConfig(nx=nx, ny=ny, nz=nz))
        pairs = sim._contact_candidate_pairs()
        assert pairs.shape[0] == case["n_candidate_pairs"], case["grid"]