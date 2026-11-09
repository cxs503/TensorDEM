"""Force-network topology and the channel-arch campaign observables.

The arch observables are read off the strong-force graph, so the graph statistics
themselves are unit-tested against hand-computable topologies (a tree has no
loops, a triangle has one, a ``2x2`` square grid has one, a ``3x3`` grid has
four), and the campaign drivers are checked for structural consistency and for
honest behaviour when a threshold is *not* detectable.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.freeze_bond.experiments import (
    CampaignConfig,
    bilinear_breakpoint,
    channel_arch_run,
    kappa_sweep,
    single_bond_self_consistency,
)
from tensordem.freeze_bond.force_network import analyze_network, strong_force_subgraph


def _square_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    pos = np.stack([xs.ravel(), ys.ravel()], axis=1)
    edges = []
    for j in range(n):
        for i in range(n):
            k = j * n + i
            if i + 1 < n:
                edges.append((k, k + 1))
            if j + 1 < n:
                edges.append((k, k + n))
    return pos, np.array(edges, dtype=np.int64)


def test_cycle_rank_of_known_graphs():
    pos, edges = _square_grid(2)          # 4 nodes, 4 edges -> 1 loop
    net = analyze_network(pos, edges, np.ones(len(edges)), k_strong=0.0,
                          width=1.0, span_fraction=0.5)
    assert net.cycle_rank == 1
    pos, edges = _square_grid(3)          # 9 nodes, 12 edges -> 4 loops
    net = analyze_network(pos, edges, np.ones(len(edges)), k_strong=0.0,
                          width=2.0, span_fraction=0.5)
    assert net.cycle_rank == 4
    # a tree: no loops
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    net = analyze_network(pos, np.array([[0, 1], [1, 2]]), np.ones(2),
                          k_strong=0.0, width=1.0)
    assert net.cycle_rank == 0
    assert net.n_spanning_loops == 0


def test_strong_force_threshold_selects_links():
    magnitudes = np.array([10.0, 1.0, 1.0])
    mask, ref = strong_force_subgraph(magnitudes, k_strong=1.0)
    assert ref == pytest.approx(4.0)
    assert mask.tolist() == [True, False, False]


def test_spanning_detected_only_when_walls_are_connected():
    # 3x3 grid: a strong path from the bottom row to the top row exists, so the
    # two transverse walls are connected in the strong graph.
    pos, edges = _square_grid(3)
    mags = np.ones(len(edges))
    tag = np.zeros(9, dtype=int)
    tag[[0, 1, 2]] = -1
    tag[[6, 7, 8]] = 1
    net = analyze_network(pos, edges, mags, k_strong=0.0, width=2.0,
                          wall_tag=tag)
    assert net.spanning
    # cut the middle row out of the strong network: no spanning path any more
    mags_cut = mags.copy()
    for idx, (a, b) in enumerate(edges):
        if 3 in (a, b) or 4 in (a, b) or 5 in (a, b):
            mags_cut[idx] = 0.0
    net_cut = analyze_network(pos, edges, mags_cut, k_strong=0.5, width=2.0,
                              wall_tag=tag)
    assert not net_cut.spanning


def test_bilinear_breakpoint_recovers_a_planted_threshold():
    x = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    y = np.where(x < 0.05, 5.0 + 0.0 * x, 7.0 + 0.0 * x)
    fit = bilinear_breakpoint(x, y)
    assert fit["threshold_detected"]
    assert 0.02 <= fit["breakpoint"] <= 0.1
    # a straight line has no threshold: the fit must say so instead of inventing one
    flat = bilinear_breakpoint(x, 1.0 + 2.0 * x)
    assert flat["breakpoint"] is None
    # degenerate input
    assert bilinear_breakpoint(np.array([1.0, 2.0]), np.array([1.0, 2.0]))["breakpoint"] is None


def test_single_bond_self_consistency_is_machine_precision():
    result = single_bond_self_consistency(
        delta_T_grid=(2.0, 5.0), salinity_grid=(0.0, 10.0), time_grid=(600.0,))
    assert result["n_points"] == 4
    assert result["max_rel_error"] < 1e-12
    assert result["verdict"] == "self-consistent"


def test_channel_arch_run_structure():
    campaign = CampaignConfig(n_x=8, n_y=5, n_steps=8, total_displacement_m=6.0e-4)
    run = channel_arch_run(0.02, campaign=campaign)
    assert run["rows"][0]["step"] == 1
    assert len(run["rows"]) == 8
    last = run["final"]
    assert last["platen_force_N"] > 0.0
    assert 0.0 <= last["transmission"] <= 1.5        # load cell / platen
    assert last["arch_count"] >= 0
    assert 0.0 <= last["max_span_ratio"] <= 5.0
    assert "residual_fraction" in run["ledger"]


def test_kappa_sweep_reports_two_judgements_and_never_fabricates():
    campaign = CampaignConfig(n_x=8, n_y=5, n_steps=6, total_displacement_m=4.0e-4)
    sweep = kappa_sweep([0.0, 0.02], campaign=campaign)
    assert len(sweep["kappas"]) == 2
    # with two abscissae no breakpoint may be claimed
    assert sweep["kappa_c_arch"] is None
    assert sweep["kappa_c_transmission"] is None
    assert "two independent judgements" in sweep["policy"]
    assert sweep["fit_arch"]["reason"] or sweep["fit_arch"]["breakpoint"] is None