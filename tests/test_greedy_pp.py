import math

from algorithms.charikar import charikar_peeling
from algorithms.common import brute_force_densest_subgraph, density
from algorithms.greedy_pp import greedy_plus_plus


def test_greedy_pp_never_worse_than_charikar(all_fixtures):
    for name, G in all_fixtures.items():
        charikar_result = charikar_peeling(G)
        gpp_result = greedy_plus_plus(G, max_rounds=40)
        assert gpp_result["best_density"] >= charikar_result["best_density"] - 1e-9, name


def test_greedy_pp_converges_to_brute_force_optimum(all_fixtures):
    for name, G in all_fixtures.items():
        opt_density, _ = brute_force_densest_subgraph(G, objective="edge")
        gpp_result = greedy_plus_plus(G, max_rounds=60, tol=1e-9, patience=10)
        assert math.isclose(gpp_result["best_density"], opt_density, rel_tol=1e-6, abs_tol=1e-6), (
            name, gpp_result["best_density"], opt_density
        )


def test_greedy_pp_best_nodes_density_is_consistent(k4_bridge_chain):
    result = greedy_plus_plus(k4_bridge_chain)
    assert math.isclose(
        result["best_density"], density(k4_bridge_chain, result["best_nodes"]), rel_tol=1e-9
    )


def test_greedy_pp_round_bookkeeping_shapes(k4_bridge_chain):
    result = greedy_plus_plus(k4_bridge_chain, max_rounds=10, patience=100)
    n_rounds = len(result["round_best_density"])
    assert n_rounds == 10  # patience disabled -> runs the full max_rounds
    assert len(result["round_snapshots"]) == n_rounds
    assert len(result["round_best_index"]) == n_rounds
    assert 0 <= result["best_round"] < n_rounds
    # every round's snapshot sequence shrinks from the full graph to 1 node
    for snaps in result["round_snapshots"]:
        assert len(snaps[0][0]) == k4_bridge_chain.number_of_nodes()
        assert len(snaps[-1][0]) == 1
