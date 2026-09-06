import math

from algorithms.common import (
    brute_force_densest_subgraph,
    density,
    enumerate_triangles,
    induced_triangle_count,
    triangle_density,
    weighted_degree,
)


def test_weighted_degree_unweighted(clique5):
    # complete graph on 5 nodes: each vertex has degree 4
    for v in clique5.nodes():
        assert weighted_degree(clique5, v) == 4.0


def test_weighted_degree_weighted(weighted_variant):
    # node 0 has edges to 1 (w=3), 2 (w=2), 3 (w=4)
    assert weighted_degree(weighted_variant, 0) == 3.0 + 2.0 + 4.0


def test_density_clique(clique5):
    # K5 has 10 edges over 5 nodes -> density 2.0
    assert math.isclose(density(clique5, clique5.nodes()), 10 / 5)


def test_density_empty_and_singleton(clique5):
    assert density(clique5, []) == 0.0
    assert density(clique5, [0]) == 0.0


def test_density_isolates_dense_core(k4_bridge_chain):
    # K4 subgraph {0,1,2,3}: 6 edges / 4 nodes = 1.5
    assert math.isclose(density(k4_bridge_chain, {0, 1, 2, 3}), 1.5)
    # whole graph (10 nodes, 11 edges) has much lower density
    whole = density(k4_bridge_chain, k4_bridge_chain.nodes())
    assert whole < 1.5


def test_enumerate_triangles_k4(k4):
    triangles = enumerate_triangles(k4)
    assert len(triangles) == 4  # C(4,3) = 4 triangles in K4


def test_enumerate_triangles_two_triangles_bridge(two_triangles_bridge):
    triangles = enumerate_triangles(two_triangles_bridge)
    assert len(triangles) == 2
    assert frozenset((0, 1, 2)) in triangles
    assert frozenset((5, 6, 7)) in triangles


def test_triangle_density_single_triangle(two_triangles_bridge):
    triangles = enumerate_triangles(two_triangles_bridge)
    d = triangle_density(two_triangles_bridge, {0, 1, 2}, triangles)
    assert math.isclose(d, 1 / 3)


def test_triangle_density_k4(k4):
    d = triangle_density(k4, k4.nodes())
    assert math.isclose(d, 4 / 4)  # 4 triangles / 4 nodes = 1.0


def test_brute_force_edge_density_clique(clique5):
    best_density, best_set = brute_force_densest_subgraph(clique5, objective="edge")
    assert math.isclose(best_density, 2.0)
    assert set(best_set) == set(clique5.nodes())


def test_brute_force_edge_density_isolates_k4(k4_bridge_chain):
    best_density, best_set = brute_force_densest_subgraph(k4_bridge_chain, objective="edge")
    assert math.isclose(best_density, 1.5)
    assert set(best_set) == {0, 1, 2, 3}


def test_brute_force_triangle_density_prefers_k4_over_lone_triangle():
    from algorithms.common import k4_graph, two_triangles_bridge_graph
    import networkx as nx

    # Combine a K4 (triangle-density 1.0) with a disjoint lone triangle
    # (triangle-density 1/3) via a bridge -- brute force must pick the K4.
    G = nx.disjoint_union(k4_graph(), two_triangles_bridge_graph())
    best_density, best_set = brute_force_densest_subgraph(G, objective="triangle")
    assert math.isclose(best_density, 1.0)
    assert len(best_set) == 4
