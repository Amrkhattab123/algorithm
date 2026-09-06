"""
Goldberg's algorithm: exact solution of the (edge-)Densest Subgraph Problem
via a reduction to parametric max-flow/min-cut.

Reference: A. V. Goldberg, "Finding a Maximum Density Subgraph", UC Berkeley
Tech Report UCB/CSD-84-171, 1984. For a candidate density g, build:
    s -> v         capacity m                         (for every vertex v)
    v -> t         capacity m + 2g - weighted_degree(v) (for every vertex v)
    u <-> v        capacity w(u,v), both directions     (for every graph edge)
where m = total edge weight. A subgraph of density >= g exists iff the
min s-t cut's source side (excluding s) is non-empty; binary searching g
over [0, Delta/2] (Delta = max weighted degree, the provable upper bound on
any subgraph's density) converges to the *exact* maximum density and its
witness vertex set.
"""
from __future__ import annotations

import time

import networkx as nx

from .common import density, total_weight, weighted_degree
from .maxflow_core import nested_sweep, parametric_binary_search


def _build_network(G: nx.Graph, m: float, g: float) -> nx.DiGraph:
    F = nx.DiGraph()
    for v in G.nodes():
        deg_v = weighted_degree(G, v)
        F.add_edge("s", v, capacity=m)
        F.add_edge(v, "t", capacity=max(0.0, m + 2.0 * g - deg_v))
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        F.add_edge(u, v, capacity=w)
        F.add_edge(v, u, capacity=w)
    return F


def _integer_weighted(G: nx.Graph) -> bool:
    return all(abs(d.get("weight", 1.0) - round(d.get("weight", 1.0))) < 1e-9 for _, _, d in G.edges(data=True))


def goldberg_densest_subgraph(G: nx.Graph, n_video_frames: int = 20):
    """Run Goldberg's exact densest-subgraph algorithm on G.

    Returns a dict with:
        best_density: float, the exact maximum density (recomputed directly
                      from the witness set for precision, not just the raw
                      bisection value)
        best_nodes: frozenset, the optimal densest-subgraph vertex set
        search_time: float, seconds spent in the timed binary search only
                     (excludes the visualization sweep below)
        n_iterations: int, number of min-cut calls in the binary search
        growth_frames: list[(g, node_set)], a monotonically GROWING sequence
                       (nested-cuts sweep from g_hi down to best_density) for
                       the incremental-construction video -- NOT included in
                       search_time
    """
    n = G.number_of_nodes()
    m = total_weight(G)

    if n == 0 or m == 0:
        return {
            "best_density": 0.0, "best_nodes": frozenset(), "search_time": 0.0,
            "n_iterations": 0, "growth_frames": [],
        }

    delta = max(weighted_degree(G, v) for v in G.nodes())
    hi = delta / 2.0 + 1e-9  # tiny buffer against float boundary issues
    lo = 0.0

    if _integer_weighted(G) and n > 1:
        eps = 1.0 / (n * (n - 1))
    else:
        eps = 1e-9

    build_fn = lambda g: _build_network(G, m, g)

    t0 = time.perf_counter()
    g_star, witness, n_iter = parametric_binary_search(build_fn, lo, hi, eps=eps)
    search_time = time.perf_counter() - t0

    best_nodes = frozenset(witness) if witness else frozenset()
    best_density = density(G, best_nodes) if best_nodes else 0.0

    growth_frames = nested_sweep(build_fn, g_star=g_star, g_hi=hi, n_points=n_video_frames)
    # keep only the (g, node_set) pairs, discarding duplicate consecutive sets
    # that a fine-grained sweep can produce once it has already reached S*
    cleaned = []
    seen = None
    for g, nodes in growth_frames:
        if nodes != seen:
            cleaned.append((g, nodes))
            seen = nodes

    return {
        "best_density": best_density,
        "best_nodes": best_nodes,
        "search_time": search_time,
        "n_iterations": n_iter,
        "growth_frames": cleaned,
    }
