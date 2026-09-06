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

    IMPORTANT (performance): this does NOT materialize a full node-set
    "snapshot" at every one of the n peeling steps -- doing so would cost
    O(n) space and time PER STEP (a frozenset copy of up to n elements),
    i.e. O(n^2) overall, which is fine for tiny test graphs but is a fatal
    blow-up on real datasets (e.g. n=317,080 for ca-dblp-2012 would try to
    copy multi-hundred-thousand-element sets ~300K times -- observed to
    exhaust >11GB of RAM and never finish). Instead only the O(1)-per-step
    scalar trace is recorded here; call `reconstruct_snapshot_tail` below to
    materialize actual node-set snapshots for just the (small) suffix of
    steps actually needed for visualization.

    Returns:
        removal_order: list of nodes in the order they were peeled off (length n)
        density_trace: list of (total_weight: float, density: float), recorded
                   *before* each removal (so density_trace[0] describes the
                   full graph, density_trace[-1] the final single-vertex set)
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
    density_trace = []
    degree_at_removal = {}

    while remaining:
        n_remaining = len(remaining)
        density_trace.append((total_w, total_w / n_remaining))

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

    return removal_order, density_trace, degree_at_removal


def reconstruct_remaining_set(all_nodes: frozenset, removal_order: list, upto_step: int) -> frozenset:
    """The exact node set remaining just before `removal_order[upto_step]` is
    removed (i.e. the snapshot at peeling step `upto_step`). O(n) -- call
    this only a small, constant number of times per peeling run (e.g. once,
    for the single best-density node set), not once per step.
    """
    return all_nodes - frozenset(removal_order[:upto_step])


def reconstruct_snapshot_tail(all_nodes: frozenset, removal_order: list, density_trace: list, start_step: int):
    """Materialize actual (node_set, total_weight, density) snapshots for
    peeling steps [start_step, len(removal_order)) only -- the small suffix
    a visualizer actually replays (typically bounded by the discovered
    densest subgraph's size, not the whole graph's). One O(n) set-difference
    to seed the starting `remaining` set, then O(k) discards for a suffix of
    length k = len(removal_order) - start_step, with each of the k `frozenset`
    snapshots costing O(current remaining size) <= O(k) -- overall O(n + k^2),
    versus O(n^2) if this were done for the full history.
    """
    remaining = set(all_nodes) - set(removal_order[:start_step])
    tail = []
    for i in range(start_step, len(removal_order)):
        total_w, dens = density_trace[i]
        tail.append((frozenset(remaining), total_w, dens))
        remaining.discard(removal_order[i])
    return tail


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
