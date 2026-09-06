"""
Charikar's greedy peeling algorithm for the Densest Subgraph Problem.

Reference: M. Charikar, "Greedy Approximation Algorithms for Finding Dense
Components in a Graph", APPROX 2000. Provides a 1/2-approximation to the
maximum-density subgraph in O((n + m) log n) time: repeatedly remove the
vertex of lowest (weighted) degree, tracking the density of the remaining
induced subgraph after each removal; return the densest subgraph observed
over the whole peeling sequence.
"""
from __future__ import annotations

import networkx as nx

from .common import peel_once, reconstruct_remaining_set, weighted_degree


def charikar_peeling(G: nx.Graph):
    """Run Charikar's greedy peeling algorithm.

    Returns a dict with:
        best_density: float, the highest density observed during peeling
        best_nodes: frozenset, the node set achieving best_density
        removal_order: list of nodes in peeling (removal) order
        density_trace: list of (total_weight, density), one per peeling step,
                   in shrinking order (density_trace[0] = full graph) --
                   O(1) per entry; use `all_nodes`/`removal_order` with
                   `common.reconstruct_snapshot_tail` to materialize actual
                   node-set snapshots for a suffix of steps (e.g. for
                   visualization), rather than storing them all up front.
        best_index: index into `density_trace` of the best-density step
        all_nodes: frozenset of every vertex in G, for snapshot reconstruction
    """
    all_nodes = frozenset(G.nodes())
    removal_order, density_trace, _degree_at_removal = peel_once(
        G, priority_fn=lambda v: weighted_degree(G, v)
    )
    best_index = max(range(len(density_trace)), key=lambda i: density_trace[i][1])
    best_density = density_trace[best_index][1]
    best_nodes = reconstruct_remaining_set(all_nodes, removal_order, best_index)
    return {
        "best_density": best_density,
        "best_nodes": best_nodes,
        "removal_order": removal_order,
        "density_trace": density_trace,
        "best_index": best_index,
        "all_nodes": all_nodes,
    }
