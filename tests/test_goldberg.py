import math

from algorithms.common import brute_force_densest_subgraph, density
from algorithms.goldberg_maxflow import goldberg_densest_subgraph


def test_goldberg_exact_match_all_fixtures(all_fixtures):
    for name, G in all_fixtures.items():
        opt_density, opt_set = brute_force_densest_subgraph(G, objective="edge")
        result = goldberg_densest_subgraph(G)
        assert math.isclose(result["best_density"], opt_density, rel_tol=1e-6, abs_tol=1e-6), (
            name, result["best_density"], opt_density
        )
        # tie-aware witness check: the returned set must itself actually
        # achieve the optimal density (there can be multiple optimal sets)
        assert math.isclose(density(G, result["best_nodes"]), opt_density, rel_tol=1e-6, abs_tol=1e-6), name


def test_goldberg_isolates_k4(k4_bridge_chain):
    result = goldberg_densest_subgraph(k4_bridge_chain)
    assert math.isclose(result["best_density"], 1.5)
    assert set(result["best_nodes"]) == {0, 1, 2, 3}


def test_goldberg_growth_frames_are_nested_and_end_at_optimum(k4_bridge_chain):
    result = goldberg_densest_subgraph(k4_bridge_chain)
    frames = result["growth_frames"]
    assert len(frames) >= 1
    # sizes must be non-decreasing as g decreases (frames are already ordered
    # from high-g/small-set to low-g/large-set)
    sizes = [len(nodes) for _, nodes in frames]
    assert sizes == sorted(sizes)
    # nested: each frame's set must be a subset of the next frame's set
    for i in range(len(frames) - 1):
        assert frames[i][1] <= frames[i + 1][1]
    assert set(frames[-1][1]) == set(result["best_nodes"])


def test_goldberg_weighted(weighted_variant):
    opt_density, _ = brute_force_densest_subgraph(weighted_variant, objective="edge")
    result = goldberg_densest_subgraph(weighted_variant)
    assert math.isclose(result["best_density"], opt_density, rel_tol=1e-6, abs_tol=1e-6)


def test_goldberg_search_time_recorded(clique5):
    result = goldberg_densest_subgraph(clique5)
    assert result["search_time"] >= 0.0
    assert result["n_iterations"] > 0
