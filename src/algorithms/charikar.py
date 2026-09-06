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

from .common import peel_once, weighted_degree


def charikar_peeling(G: nx.Graph):
    """Run Charikar's greedy peeling algorithm.

    Returns a dict with:
        best_density: float, the highest density observed during peeling
        best_nodes: frozenset, the node set achieving best_density
        removal_order: list of nodes in peeling (removal) order
        snapshots: list of (node_set, total_weight, density), one per peeling
                   step, in shrinking order (snapshots[0] = full graph)
        best_index: index into `snapshots` of the best-density snapshot
    """
    removal_order, snapshots, _degree_at_removal = peel_once(
        G, priority_fn=lambda v: weighted_degree(G, v)
    )
    return _finalize(snapshots, removal_order)


def _finalize(snapshots, removal_order):
    best_index = max(range(len(snapshots)), key=lambda i: snapshots[i][2])
    best_nodes, _, best_density = snapshots[best_index]
    return {
        "best_density": best_density,
        "best_nodes": best_nodes,
        "removal_order": removal_order,
        "snapshots": snapshots,
        "best_index": best_index,
    }
