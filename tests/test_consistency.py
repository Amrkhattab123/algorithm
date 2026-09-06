"""
Cross-algorithm sanity ties. Goldberg's algorithm is exact, so it must never
be beaten (up to floating tolerance) by either heuristic peeling algorithm on
the same graph -- this is the single most valuable regression check tying
all three edge-density algorithms together.
"""
from algorithms.charikar import charikar_peeling
from algorithms.goldberg_maxflow import goldberg_densest_subgraph
from algorithms.greedy_pp import greedy_plus_plus

EPS = 1e-6


def test_goldberg_dominates_charikar(all_fixtures):
    for name, G in all_fixtures.items():
        goldberg_result = goldberg_densest_subgraph(G)
        charikar_result = charikar_peeling(G)
        assert goldberg_result["best_density"] >= charikar_result["best_density"] - EPS, name


def test_goldberg_dominates_greedy_pp(all_fixtures):
    for name, G in all_fixtures.items():
        goldberg_result = goldberg_densest_subgraph(G)
        gpp_result = greedy_plus_plus(G, max_rounds=40)
        assert goldberg_result["best_density"] >= gpp_result["best_density"] - EPS, name


def test_greedy_pp_at_least_as_good_as_charikar(all_fixtures):
    for name, G in all_fixtures.items():
        charikar_result = charikar_peeling(G)
        gpp_result = greedy_plus_plus(G, max_rounds=40)
        assert gpp_result["best_density"] >= charikar_result["best_density"] - EPS, name
