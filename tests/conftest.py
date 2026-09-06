import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from algorithms.common import (
    bowtie_graph,
    clique_graph,
    k4_bridge_chain_graph,
    k4_graph,
    two_triangles_bridge_graph,
    weighted_variant_graph,
)


@pytest.fixture
def clique5():
    return clique_graph(5)


@pytest.fixture
def bowtie():
    return bowtie_graph()


@pytest.fixture
def k4_bridge_chain():
    return k4_bridge_chain_graph()


@pytest.fixture
def weighted_variant():
    return weighted_variant_graph()


@pytest.fixture
def two_triangles_bridge():
    return two_triangles_bridge_graph()


@pytest.fixture
def k4():
    return k4_graph()


@pytest.fixture
def all_fixtures(clique5, bowtie, k4_bridge_chain, weighted_variant, two_triangles_bridge, k4):
    return {
        "clique5": clique5,
        "bowtie": bowtie,
        "k4_bridge_chain": k4_bridge_chain,
        "weighted_variant": weighted_variant,
        "two_triangles_bridge": two_triangles_bridge,
        "k4": k4,
    }
