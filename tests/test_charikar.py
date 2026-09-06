import math

from algorithms.charikar import charikar_peeling
from algorithms.common import brute_force_densest_subgraph, density


def _check_2_approx(G):
    opt_density, _ = brute_force_densest_subgraph(G, objective="edge")
    result = charikar_peeling(G)
    assert result["best_density"] >= 0.5 * opt_density - 1e-9
    # the density recorded in the result must match recomputing density() on best_nodes
    assert math.isclose(result["best_density"], density(G, result["best_nodes"]), rel_tol=1e-9)


def test_charikar_2_approximation_all_fixtures(all_fixtures):
    for name, G in all_fixtures.items():
        _check_2_approx(G)


def test_charikar_exact_on_clique(clique5):
    # peeling a clique is provably optimal: removing any vertex only ever
    # decreases density, so the full clique itself is the best snapshot.
    result = charikar_peeling(clique5)
    assert math.isclose(result["best_density"], 2.0)
    assert set(result["best_nodes"]) == set(clique5.nodes())


def test_charikar_isolates_k4(k4_bridge_chain):
    result = charikar_peeling(k4_bridge_chain)
    assert math.isclose(result["best_density"], 1.5)
    assert set(result["best_nodes"]) == {0, 1, 2, 3}


def test_charikar_snapshots_shrink_monotonically(k4_bridge_chain):
    result = charikar_peeling(k4_bridge_chain)
    sizes = [len(s[0]) for s in result["snapshots"]]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == k4_bridge_chain.number_of_nodes()
    assert sizes[-1] == 1
    assert len(result["removal_order"]) == k4_bridge_chain.number_of_nodes()


def test_charikar_weighted(weighted_variant):
    _check_2_approx(weighted_variant)
