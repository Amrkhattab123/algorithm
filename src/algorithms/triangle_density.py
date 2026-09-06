"""
Exact triangle-density algorithm: solves max_S triangles(S)/|S| exactly via a
flow-network reduction that generalizes Goldberg's edge-density construction
to a 3-uniform hypergraph term.

References: S. Khuller & B. Saha, "On Finding Dense Subgraphs" (ICALP 2009)
generalize densest-subgraph flow reductions to r-uniform hypergraphs; C. E.
Tsourakakis, "The K-clique Densest Subgraph Problem" (WWW 2015) and
Tsourakakis et al., "Denser than the Densest Subgraph: Extracting Optimal
Quasi-Cliques with Quality Guarantees" (KDD 2013) motivate triangle-density as
a less "clique-fooled" alternative to plain edge-density.

For a candidate triangle-density g, build:
    s -> Delta         capacity 1                      (for every triangle Delta)
    Delta -> u, v, w   capacity BIG (>= total triangles) (Delta's 3 endpoints)
    v -> t             capacity g                        (for every vertex v)
A subgraph with triangle-density >= g exists iff the min-cut's source side
(restricted to graph-vertex nodes, discarding triangle-node artifacts) is
non-empty. The BIG capacity on Delta -> {u,v,w} ensures those edges are never
the cheapest way to cut -- a triangle can only be "cut off" via its s->Delta
edge, which costs 1, unless all 3 of its vertices are already on the sink
side (that path being cheaper only if all 3 v->t edges are cheaper than 1
combined, which the binary search on g balances against the count of
included triangles).
"""
from __future__ import annotations

import time

import networkx as nx

from .common import enumerate_triangles, induced_triangle_count, triangle_density as triangle_density_of
from .maxflow_core import nested_sweep, parametric_binary_search

BIG_CAPACITY_MARGIN = 1.0


def _is_triangle_node(x) -> bool:
    return isinstance(x, tuple) and len(x) == 2 and x[0] == "tri"


def _build_network(G: nx.Graph, triangles: list, big: float, g: float) -> nx.DiGraph:
    F = nx.DiGraph()
    for v in G.nodes():
        F.add_edge(v, "t", capacity=g)
    for i, tri in enumerate(triangles):
        tri_node = ("tri", i)
        F.add_edge("s", tri_node, capacity=1.0)
        for v in tri:
            F.add_edge(tri_node, v, capacity=big)
    return F


def exact_triangle_densest_subgraph(G: nx.Graph, n_video_frames: int = 20):
    """Run the exact triangle-density algorithm on G.

    Returns a dict with:
        best_density: float, the exact maximum triangle-density
                      (triangles(S)/|S|), recomputed directly from the
                      witness set
        best_nodes: frozenset, the optimal vertex set
        n_triangles_total: int, total triangles enumerated in G (a cheap
                            up-front size/feasibility check -- callers should
                            inspect this before running on a large graph)
        search_time: float, seconds spent in triangle enumeration + the timed
                     binary search (excludes the visualization sweep)
        n_iterations: int, number of min-cut calls in the binary search
        growth_frames: list[(g, node_set)], monotonically GROWING nested-cuts
                       sweep for the incremental-construction video
    """
    t0 = time.perf_counter()
    triangles = enumerate_triangles(G)
    n_triangles_total = len(triangles)

    if n_triangles_total == 0:
        return {
            "best_density": 0.0, "best_nodes": frozenset(), "n_triangles_total": 0,
            "search_time": time.perf_counter() - t0, "n_iterations": 0, "growth_frames": [],
        }

    big = n_triangles_total + BIG_CAPACITY_MARGIN
    lo, hi = 0.0, float(n_triangles_total)  # safe, generous upper bound
    eps = 1e-6

    build_fn = lambda g: _build_network(G, triangles, big, g)

    g_star, raw_witness, n_iter = parametric_binary_search(build_fn, lo, hi, eps=eps)
    search_time = time.perf_counter() - t0

    best_nodes = frozenset(x for x in raw_witness if not _is_triangle_node(x))
    best_density = (
        triangle_density_of(G, best_nodes, triangles) if best_nodes else 0.0
    )

    raw_frames = nested_sweep(build_fn, g_star=g_star, g_hi=hi, n_points=n_video_frames)
    cleaned = []
    seen = None
    for g, nodes in raw_frames:
        vertex_nodes = frozenset(x for x in nodes if not _is_triangle_node(x))
        if vertex_nodes != seen:
            cleaned.append((g, vertex_nodes))
            seen = vertex_nodes

    return {
        "best_density": best_density,
        "best_nodes": best_nodes,
        "n_triangles_total": n_triangles_total,
        "search_time": search_time,
        "n_iterations": n_iter,
        "growth_frames": cleaned,
    }
