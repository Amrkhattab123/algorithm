import math

from algorithms.common import brute_force_densest_subgraph, triangle_density
from algorithms.triangle_density import exact_triangle_densest_subgraph


def test_triangle_density_k4(k4):
    result = exact_triangle_densest_subgraph(k4)
    assert math.isclose(result["best_density"], 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert set(result["best_nodes"]) == {0, 1, 2, 3}
    assert result["n_triangles_total"] == 4


def test_triangle_density_lone_triangle_component(two_triangles_bridge):
    # both triangles have identical triangle-density (1/3); the algorithm
    # must find *a* triangle-density-optimal set, and its density must match
    # brute force exactly.
    opt_density, _ = brute_force_densest_subgraph(two_triangles_bridge, objective="triangle")
    result = exact_triangle_densest_subgraph(two_triangles_bridge)
    assert math.isclose(result["best_density"], opt_density, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(opt_density, 1 / 3)


def test_triangle_density_prefers_k4_over_lone_triangle():
    import networkx as nx
    from algorithms.common import k4_graph, two_triangles_bridge_graph

    G = nx.disjoint_union(k4_graph(), two_triangles_bridge_graph())
    opt_density, opt_set = brute_force_densest_subgraph(G, objective="triangle")
    result = exact_triangle_densest_subgraph(G)
    assert math.isclose(result["best_density"], opt_density, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(opt_density, 1.0)
    assert len(result["best_nodes"]) == 4  # the K4 component, not a lone triangle


def test_triangle_density_no_triangles():
    import networkx as nx
    G = nx.path_graph(5)  # a tree/path has zero triangles
    result = exact_triangle_densest_subgraph(G)
    assert result["best_density"] == 0.0
    assert result["n_triangles_total"] == 0


def test_triangle_density_growth_frames_nested_and_consistent(k4):
    result = exact_triangle_densest_subgraph(k4)
    frames = result["growth_frames"]
    assert len(frames) >= 1
    sizes = [len(nodes) for _, nodes in frames]
    assert sizes == sorted(sizes)
    for i in range(len(frames) - 1):
        assert frames[i][1] <= frames[i + 1][1]
    assert set(frames[-1][1]) == set(result["best_nodes"])
