"""
Greedy++: an iterative refinement of Charikar's peeling algorithm.

Reference: D. Boob, Y. Gao, R. Peng, S. Sawlani, C. Tsourakakis, D. Wang,
J. Wang, "Flowless: Extracting Densest Subgraphs Without Flow Computations"
/ "Greedy++" (WWW 2020). Each round re-peels the *entire* graph, but instead
of ranking vertices by plain degree, it ranks them by degree plus an
accumulated "load" carried over from every previous round (the induced
degree each vertex had at the moment it was peeled in earlier rounds). This
is a discretized multiplicative-weights / Frank-Wolfe style subgradient
scheme that provably converges towards the exact LP-optimal density as the
number of rounds grows, closing much of the 2x approximation gap that a
single Charikar pass leaves open, without ever running a max-flow.
"""
from __future__ import annotations

import networkx as nx

from .common import peel_once, reconstruct_remaining_set, weighted_degree


def greedy_plus_plus(G: nx.Graph, max_rounds: int = 40, tol: float = 1e-6, patience: int = 5):
    """Run Greedy++ for up to `max_rounds` rounds.

    Returns a dict with:
        best_density: float, the highest density observed across all rounds
        best_nodes: frozenset, the node set achieving best_density
        best_round: int, which round produced the global best
        round_best_density: list[float], best density seen in each round (for
                             a convergence plot)
        round_removal_order: list[list[node]], each round's peeling order
                          (round_removal_order[best_round] is what the
                          visualizer replays, via `reconstruct_snapshot_tail`)
        round_density_trace: list[list[(total_weight, density)]], each
                          round's O(1)-per-step scalar trace (NOT full
                          node-set snapshots -- see `common.peel_once` for
                          why that would be a fatal O(n^2 * rounds) blow-up
                          on real datasets)
        round_best_index: list[int], index into each round's own trace of
                           that round's best-density step
        all_nodes: frozenset of every vertex in G, for snapshot reconstruction
    """
    all_nodes = frozenset(G.nodes())
    load = {v: 0.0 for v in G.nodes()}

    round_best_density = []
    round_best_index = []
    round_removal_order = []
    round_density_trace = []

    global_best_density = -1.0
    global_best_round = 0

    stall_count = 0
    prev_best = -1.0

    for t in range(max_rounds):
        removal_order, density_trace, degree_at_removal = peel_once(
            G, priority_fn=lambda v, _load=load: _load[v] + weighted_degree(G, v)
        )
        this_round_best_idx = max(range(len(density_trace)), key=lambda i: density_trace[i][1])
        this_round_best_density = density_trace[this_round_best_idx][1]

        round_best_density.append(this_round_best_density)
        round_best_index.append(this_round_best_idx)
        round_removal_order.append(removal_order)
        round_density_trace.append(density_trace)

        if this_round_best_density > global_best_density:
            global_best_density = this_round_best_density
            global_best_round = t

        # update persistent load with this round's induced degree-at-removal
        for v, d in degree_at_removal.items():
            load[v] += d

        if this_round_best_density <= prev_best + tol:
            stall_count += 1
        else:
            stall_count = 0
        prev_best = this_round_best_density

        if stall_count >= patience:
            break

    best_index = round_best_index[global_best_round]
    best_removal_order = round_removal_order[global_best_round]
    best_nodes = reconstruct_remaining_set(all_nodes, best_removal_order, best_index)

    return {
        "best_density": global_best_density,
        "best_nodes": best_nodes,
        "best_round": global_best_round,
        "round_best_density": round_best_density,
        "round_removal_order": round_removal_order,
        "round_density_trace": round_density_trace,
        "round_best_index": round_best_index,
        "all_nodes": all_nodes,
    }
