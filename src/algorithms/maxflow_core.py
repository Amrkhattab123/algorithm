"""
Shared parametric-binary-search-over-min-cut driver, used by both Goldberg's
exact edge-density algorithm and the exact triangle-density algorithm. Both
reduce "does a subgraph of density >= g exist?" to a single s-t min-cut
computation on a graph-dependent flow network; this module only implements
the outer bisection search and the (documented) nested-cuts sweep used for
visualization, leaving network construction to each caller.
"""
from __future__ import annotations

from typing import Callable

import networkx as nx


def min_cut_source_side(flow_network: nx.DiGraph) -> set:
    """Run min s-t cut on `flow_network` (must have nodes 's','t' and a
    'capacity' edge attribute) and return the set of ordinary graph vertices
    (i.e. excluding 's' itself, and excluding any auxiliary non-vertex nodes
    the caller may filter out afterwards) on the source side of the cut.
    """
    _cut_value, (source_side, _sink_side) = nx.minimum_cut(
        flow_network, "s", "t", capacity="capacity"
    )
    return source_side - {"s"}


def parametric_binary_search(
    build_network_fn: Callable[[float], nx.DiGraph],
    lo: float,
    hi: float,
    eps: float = 1e-9,
    max_iter: int = 200,
):
    """Binary search the largest feasible density `g` in [lo, hi].

    `build_network_fn(g)` must return a flow network encoding "is there a
    subgraph with density >= g?" as "is the min-cut's source side, excluding
    s, non-trivial?" (this holds for both the Goldberg and triangle-density
    reductions used in this project).

    Returns (best_g, best_witness_set, n_iterations):
        best_g: the largest g found feasible (a lower bound on / approximation
                of the true optimal density, exact once hi-lo < eps for
                integer-weighted inputs per Goldberg's classic analysis)
        best_witness_set: the vertex set achieving that feasibility
        n_iterations: number of min-cut calls performed (for timing/reporting)
    """
    best_witness: set = set()
    n_iter = 0

    while hi - lo > eps and n_iter < max_iter:
        g = (lo + hi) / 2.0
        network = build_network_fn(g)
        source_side = min_cut_source_side(network)
        n_iter += 1

        if len(source_side) > 0:
            lo = g
            best_witness = source_side
        else:
            hi = g

    return lo, best_witness, n_iter


def nested_sweep(
    build_network_fn: Callable[[float], nx.DiGraph],
    g_star: float,
    g_hi: float,
    n_points: int = 20,
):
    """Sweep `n_points` density thresholds from `g_hi` down to `g_star`,
    exploiting the nested-cuts property of parametric max-flow (Gallo,
    Grigoriadis & Tarjan 1989): for g1 < g2, the feasible source-side witness
    S(g1) is a superset of S(g2). Reading the sweep in decreasing-g order
    therefore yields a monotonically GROWING sequence of vertex sets, ending
    exactly at S(g_star) -- used purely for the incremental "growth" video,
    not for the correctness search itself (whose time is measured separately,
    by `parametric_binary_search` above).

    Returns a list of (g, witness_set), ordered from the smallest witness
    (highest g) to the largest witness (g_star).
    """
    if n_points < 2:
        n_points = 2
    step = (g_hi - g_star) / (n_points - 1)
    g_values = [g_hi - i * step for i in range(n_points)]  # descending, ends at g_star
    g_values[-1] = g_star  # avoid float drift

    frames = []
    for g in g_values:
        network = build_network_fn(g)
        source_side = min_cut_source_side(network)
        frames.append((g, source_side))
    return frames
