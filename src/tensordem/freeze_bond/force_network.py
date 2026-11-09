"""Force-chain topology of a bonded assembly: loops, arch count, load paths.

The arch observables of the campaign are read off a *force network* -- the graph
whose nodes are disks and whose edges are the mechanical links that actually
carry load.  Edges are bonds (magnitude ``traction * area``) plus active disk
contacts (normal force with the Cundall-Strack tangential component).  A load
carrying link is *strong* when its magnitude exceeds ``k_strong`` times the mean
link magnitude (the usual Radjai-style separation of the strong network; ``k = 1``
is the default and is not tuned per run).

On the strong subgraph :func:`analyze_network` reports

* ``cycle_rank`` -- the cyclomatic number ``mu = E - V + C`` (independent loops of
  the strong network).  A tree-like, purely vertical force chain has ``mu = 0``;
  arching requires loops, so ``mu`` is the "number of arches" candidate
  (``arch_count``);
* ``n_spanning_loops`` -- loops whose transverse extent reaches
  ``span_fraction * width``, i.e. loops that actually bridge the channel; this is
  the geometric statement "the arch spans the channel" and needs no threshold on
  the force magnitude beyond ``k_strong``;
* ``max_span_ratio`` -- the largest transverse extent of a loop divided by the
  channel width;
* ``spanning`` -- whether any strong path connects the two transverse walls
  (wall-to-wall load transfer), computed as connectivity in the strong graph
  between particles touching the two walls.

The loop decomposition uses a spanning forest plus one fundamental cycle per
non-tree edge, so the counted loops form a cycle *basis*: the number is exact
(``mu``) and the geometry of each basis loop is well defined, but a basis is not
unique -- the loop *count* is, the individual loop shapes are not.  That is
stated here because the arch threshold is judged on the count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ForceNetwork", "strong_force_subgraph", "analyze_network"]


@dataclass
class ForceNetwork:
    """Strong-force subgraph and its loop/arch statistics."""

    n_nodes: int
    n_strong_edges: int
    cycle_rank: int
    arch_count: int
    n_spanning_loops: int
    max_span_ratio: float
    spanning: bool
    k_strong: float
    mean_link_force_N: float
    total_link_force_N: float
    loops: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_strong_edges": self.n_strong_edges,
            "cycle_rank": self.cycle_rank,
            "arch_count": self.arch_count,
            "n_spanning_loops": self.n_spanning_loops,
            "max_span_ratio": self.max_span_ratio,
            "spanning": self.spanning,
            "k_strong": self.k_strong,
            "mean_link_force_N": self.mean_link_force_N,
            "total_link_force_N": self.total_link_force_N,
        }


def strong_force_subgraph(magnitudes, *, k_strong: float = 1.0,
                          reference: str = "mean"
                          ) -> tuple[np.ndarray, float]:
    """Boolean mask of links carrying more than ``k_strong`` times a reference."""
    f = np.asarray(magnitudes, dtype=np.float64)
    if f.size == 0:
        return np.zeros(0, dtype=bool), 0.0
    if reference == "mean":
        ref = float(np.mean(f))
    elif reference == "median":
        ref = float(np.median(f))
    else:
        raise ValueError("reference must be 'mean' or 'median'")
    return f >= k_strong * ref, ref


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _fundamental_loops(n_nodes: int, edges: np.ndarray) -> tuple[int, int, list[list[int]]]:
    """Spanning forest + one fundamental cycle per non-tree edge."""
    uf = _UnionFind(n_nodes)
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    tree_edges: list[tuple[int, int]] = []
    extra: list[tuple[int, int]] = []
    for a, b in edges:
        if uf.union(int(a), int(b)):
            tree_edges.append((int(a), int(b)))
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
        else:
            extra.append((int(a), int(b)))
    n_components = len({uf.find(i) for i in range(n_nodes)}) if n_nodes else 0
    cycle_rank = len(extra)
    loops: list[list[int]] = []
    if cycle_rank:
        parent = -np.ones(n_nodes, dtype=np.int64)
        depth = np.zeros(n_nodes, dtype=np.int64)
        for root in range(n_nodes):
            if parent[root] != -1:
                continue
            parent[root] = root
            stack = [root]
            while stack:
                node = stack.pop()
                for nb in adjacency[node]:
                    if parent[nb] == -1:
                        parent[nb] = node
                        depth[nb] = depth[node] + 1
                        stack.append(nb)
        for a, b in extra:
            left, right = [int(a)], [int(b)]
            x, y = int(a), int(b)
            while x != y:
                if depth[x] >= depth[y]:
                    x = int(parent[x])
                    left.append(x)
                else:
                    y = int(parent[y])
                    right.append(y)
            loops.append(left + right[-2::-1] if len(right) > 1 else left)
    return cycle_rank, n_components, loops


def analyze_network(
    positions: np.ndarray,
    edges: np.ndarray,
    magnitudes: np.ndarray,
    *,
    k_strong: float = 1.0,
    width: float | None = None,
    transverse_axis: int = 1,
    span_fraction: float = 0.5,
    wall_tag: np.ndarray | None = None,
) -> ForceNetwork:
    """Loop and arch statistics of the strong-force network.

    ``edges`` is ``(m, 2)`` node indices, ``magnitudes`` the link force [N].
    ``width`` is the channel extent along ``transverse_axis``; when given, loops
    whose extent covers ``span_fraction * width`` count as spanning loops.
    ``wall_tag`` (``n_nodes``, values ``-1/0/+1`` for the low wall / bulk / high
    wall) enables the wall-to-wall connectivity test.
    """
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    mask, ref = strong_force_subgraph(magnitudes, k_strong=k_strong)
    strong = edges[mask]
    n_nodes = int(positions.shape[0])
    if strong.size == 0:
        return ForceNetwork(n_nodes=n_nodes, n_strong_edges=0, cycle_rank=0,
                            arch_count=0, n_spanning_loops=0, max_span_ratio=0.0,
                            spanning=False, k_strong=k_strong,
                            mean_link_force_N=ref,
                            total_link_force_N=float(np.sum(magnitudes)))
    nodes = np.unique(strong.reshape(-1))
    remap = -np.ones(n_nodes, dtype=np.int64)
    remap[nodes] = np.arange(nodes.size)
    local_edges = remap[strong]
    cycle_rank, n_components, loops = _fundamental_loops(nodes.size, local_edges)

    coord = positions[nodes][:, transverse_axis]
    span = float(coord.max() - coord.min()) if nodes.size else 0.0
    scale = float(width) if width else (span if span > 0 else 1.0)
    loop_info: list[dict[str, Any]] = []
    n_spanning = 0
    max_ratio = 0.0
    for loop in loops:
        if not loop:
            continue
        y = coord[np.asarray(loop, dtype=np.int64)]
        extent = float(y.max() - y.min())
        ratio = extent / scale if scale > 0 else 0.0
        max_ratio = max(max_ratio, ratio)
        spans = ratio >= span_fraction
        n_spanning += int(spans)
        loop_info.append({"n_nodes": len(loop), "extent": extent, "ratio": ratio,
                          "spanning": bool(spans)})

    spanning = False
    if wall_tag is not None:
        tag = np.asarray(wall_tag)[nodes]
        low = set(np.nonzero(tag < 0)[0].tolist())
        high = set(np.nonzero(tag > 0)[0].tolist())
        if low and high:
            uf = _UnionFind(nodes.size)
            for a, b in local_edges:
                uf.union(int(a), int(b))
            low_roots = {uf.find(i) for i in low}
            high_roots = {uf.find(i) for i in high}
            spanning = bool(low_roots & high_roots)

    return ForceNetwork(n_nodes=n_nodes, n_strong_edges=int(strong.shape[0]),
                        cycle_rank=int(cycle_rank), arch_count=int(cycle_rank),
                        n_spanning_loops=int(n_spanning), max_span_ratio=float(max_ratio),
                        spanning=spanning, k_strong=k_strong,
                        mean_link_force_N=ref,
                        total_link_force_N=float(np.sum(magnitudes)),
                        loops=loop_info)