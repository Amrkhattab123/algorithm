"""
Shared utilities for all four densest-subgraph algorithms: density/degree
helpers, triangle enumeration, brute-force ground truth (tiny graphs only),
the reusable greedy-peeling core, and small synthetic fixture graphs used by
the test suite.
"""
from __future__ import annotations

import heapq
import itertools
from typing import Callable, Iterable

import networkx as nx


# ---------------------------------------------------------------------------
# Density / degree helpers
# ---------------------------------------------------------------------------

def weighted_degree(G: nx.Graph, v) -> float:
    return sum(d.get("weight", 1.0) for _, _, d in G.edges(v, data=True))


def total_weight(G: nx.Graph) -> float:
    return sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True))


def induced_edge_weight_sum(G: nx.Graph, nodes: Iterable) -> float:
    """Sum of edge weights with both endpoints in `nodes`; O(sum of degrees in nodes)."""
    node_set = nodes if isinstance(nodes, (set, frozenset)) else set(nodes)
    seen_pairs: set = set()
    total = 0.0
    for u in node_set:
        if u not in G:
            continue
        for v, d in G[u].items():
            if v in node_set:
                pair = frozenset((u, v))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                total += d.get("weight", 1.0)
    return total


def density(G: nx.Graph, nodes: Iterable) -> float:
    """Edge density of the subgraph induced by `nodes`: sum(weights) / |nodes|."""
    node_set = nodes if isinstance(nodes, (set, frozenset)) else set(nodes)
    if len(node_set) == 0:
        return 0.0
    return induced_edge_weight_sum(G, node_set) / len(node_set)


# ---------------------------------------------------------------------------
# Triangle enumeration (used by triangle_density.py and the brute-force
# triangle-objective check)
# ---------------------------------------------------------------------------

def enumerate_triangles(G: nx.Graph) -> list[tuple]:
    """Node-iterator ("forward") triangle listing: O(m^1.5) worst case.

    Returns a list of frozenset({u, v, w}) triples, each counted once.
    """
    # Order vertices by degree ascending and orient each edge from lower to
    # higher degree (ties broken by id) -- the classic trick that bounds this
    # to O(m^1.5) instead of the naive O(n^3) or O(m*n).
    order = {v: i for i, v in enumerate(sorted(G.nodes(), key=lambda v: (G.degree(v), str(v))))}

    directed_adj: dict = {v: set() for v in G.nodes()}
    for u, v in G.edges():
        if order[u] < order[v]:
            directed_adj[u].add(v)
        else:
            directed_adj[v].add(u)

    triangles = []
    for u in G.nodes():
        neighbors = directed_adj[u]
        neighbor_list = list(neighbors)
        for i in range(len(neighbor_list)):
            v = neighbor_list[i]
            for j in range(i + 1, len(neighbor_list)):
                w = neighbor_list[j]
                if w in directed_adj[v] or v in directed_adj[w]:
                    # v-w edge exists in the original undirected graph iff one
                    # direction was recorded during orientation.
                    if G.has_edge(v, w):
                        triangles.append(frozenset((u, v, w)))
    return triangles


def induced_triangle_count(triangles: list, nodes: Iterable) -> int:
    node_set = nodes if isinstance(nodes, (set, frozenset)) else set(nodes)
    return sum(1 for t in triangles if t <= node_set)


def triangle_density(G: nx.Graph, nodes: Iterable, triangles: list | None = None) -> float:
    node_set = nodes if isinstance(nodes, (set, frozenset)) else set(nodes)
    if len(node_set) == 0:
        return 0.0
    if triangles is None:
        triangles = enumerate_triangles(G)
    return induced_triangle_count(triangles, node_set) / len(node_set)


# ---------------------------------------------------------------------------
# Brute force (tiny graphs only -- ground truth for tests)
# ---------------------------------------------------------------------------

def brute_force_densest_subgraph(G: nx.Graph, objective: str = "edge"):
    """Exhaustively check all non-empty node subsets; returns (best_density, best_set).

    objective: "edge" (default) or "triangle". Only feasible for very small
    graphs (n <= ~18); used exclusively as test ground truth, never on real
    project datasets.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n > 20:
        raise ValueError(f"brute_force_densest_subgraph is only for tiny graphs (n={n} too large)")

    triangles = enumerate_triangles(G) if objective == "triangle" else None

    best_density = 0.0
    best_set = frozenset()
    for r in range(1, n + 1):
        for combo in itertools.combinations(nodes, r):
            s = frozenset(combo)
            if objective == "edge":
                d = density(G, s)
            elif objective == "triangle":
                d = triangle_density(G, s, triangles)
            else:
                raise ValueError(f"unknown objective '{objective}'")
            if d > best_density:
                best_density = d
                best_set = s
    return best_density, best_set


# ---------------------------------------------------------------------------
# Shared greedy-peeling core (used by both Charikar's algorithm and Greedy++)
# ---------------------------------------------------------------------------

def peel_once(G: nx.Graph, priority_fn: Callable[[object], float]):
    """Run one full greedy-peeling pass, repeatedly removing the vertex with the
    lowest `priority_fn(v)` value among those still "remaining", recomputing
    priorities of neighbors as they lose the removed vertex.

    `priority_fn` is evaluated ONCE per vertex up front (its return value is
    then decremented by the weight of each incident edge as neighbors are
    peeled) -- this lets callers fold in extra additive state (e.g. Greedy++'s
    per-vertex "load") on top of plain weighted degree, while reusing the same
    heap-based peeling mechanics.

    Returns:
        removal_order: list of nodes in the order they were peeled off
        snapshots: list of (node_set: frozenset, total_weight: float, density: float),
                   one entry recorded *before* each removal (so snapshots[0] is
                   the full graph, snapshots[-1] is the final single-vertex set)
        degree_at_removal: dict {node: induced weighted degree at the moment
                   it was removed} -- needed by Greedy++ to update its load vector
    """
    remaining = set(G.nodes())
    current_score = {v: float(priority_fn(v)) for v in G.nodes()}
    current_degree = {v: weighted_degree(G, v) for v in G.nodes()}
    total_w = total_weight(G)

    heap = [(current_score[v], v) for v in G.nodes()]
    heapq.heapify(heap)

    removal_order = []
    snapshots = []
    degree_at_removal = {}

    while remaining:
        n_remaining = len(remaining)
        snapshots.append((frozenset(remaining), total_w, total_w / n_remaining))

        # pop the lowest-score live, up-to-date vertex (lazy deletion)
        v = None
        while heap:
            score, cand = heapq.heappop(heap)
            if cand in remaining and abs(score - current_score[cand]) < 1e-12:
                v = cand
                break
        if v is None:
            # Heap exhausted due to floating point staleness edge case;
            # fall back to a linear scan (only happens on tiny/degenerate graphs).
            v = min(remaining, key=lambda x: current_score[x])

        degree_at_removal[v] = current_degree[v]
        remaining.discard(v)
        removal_order.append(v)

        for u, d in G[v].items():
            if u in remaining:
                w = d.get("weight", 1.0)
                total_w -= w
                current_degree[u] -= w
                current_score[u] -= w
                heapq.heappush(heap, (current_score[u], u))

    return removal_order, snapshots, degree_at_removal


# ---------------------------------------------------------------------------
# Synthetic fixture graphs (used by tests/conftest.py)
# ---------------------------------------------------------------------------

def clique_graph(n: int = 5) -> nx.Graph:
    G = nx.complete_graph(n)
    nx.set_edge_attributes(G, 1.0, "weight")
    return G


def bowtie_graph() -> nx.Graph:
    """Two triangles sharing a single vertex."""
    G = nx.Graph()
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]
    G.add_edges_from(edges, weight=1.0)
    return G


def k4_bridge_chain_graph() -> nx.Graph:
    """A K4 (dense core) attached via a single bridge edge to a sparse path.

    The whole graph's average density is much lower than K4's internal
    density (which is 4*3/2 / 4 = 1.5), so any correct densest-subgraph
    algorithm must isolate the K4, not default to the whole graph.
    """
    G = nx.Graph()
    G.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], weight=1.0
    )  # K4 on {0,1,2,3}
    G.add_edge(3, 4, weight=1.0)  # bridge
    G.add_edges_from([(4, 5), (5, 6), (6, 7), (7, 8), (8, 9)], weight=1.0)  # sparse chain
    return G


def weighted_variant_graph() -> nx.Graph:
    """Same topology as k4_bridge_chain_graph but with distinct integer weights."""
    G = k4_bridge_chain_graph()
    weights = {
        (0, 1): 3, (0, 2): 2, (0, 3): 4, (1, 2): 5, (1, 3): 1, (2, 3): 2,
        (3, 4): 1,
        (4, 5): 1, (5, 6): 1, (6, 7): 1, (7, 8): 1, (8, 9): 1,
    }
    for (u, v), w in weights.items():
        G[u][v]["weight"] = float(w)
    return G


def two_triangles_bridge_graph() -> nx.Graph:
    """Two disjoint triangles joined by a bridge path -- for triangle-density checks.

    Triangle {0,1,2} and triangle {5,6,7}, connected via a path 2-3-4-5 that
    contributes zero triangles. Edge-density and triangle-density of any
    single triangle: edge-density = 3/3 = 1.0, triangle-density = 1/3 (one
    triangle over 3 nodes).
    """
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)], weight=1.0)  # triangle A
    G.add_edges_from([(2, 3), (3, 4), (4, 5)], weight=1.0)  # bridge path
    G.add_edges_from([(5, 6), (6, 7), (7, 5)], weight=1.0)  # triangle B
    return G


def k4_graph() -> nx.Graph:
    """A lone K4: edge-density 1.5, triangle-density 4/4 = 1.0 (4 triangles, 4 nodes)."""
    G = nx.complete_graph(4)
    nx.set_edge_attributes(G, 1.0, "weight")
    return G
