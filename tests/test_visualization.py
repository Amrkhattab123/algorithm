import os
import tempfile

import networkx as nx

from algorithms.charikar import charikar_peeling
from algorithms.common import enumerate_triangles
from algorithms.goldberg_maxflow import goldberg_densest_subgraph
from algorithms.greedy_pp import greedy_plus_plus
from algorithms.triangle_density import exact_triangle_densest_subgraph
from visualization import (
    export_growth_snapshots,
    growth_frames_from_goldberg,
    growth_frames_from_greedy_pp,
    growth_frames_from_peeling,
    growth_frames_from_triangle_density,
)


def _assert_valid_growth_frames(frames, expected_final_nodes):
    assert len(frames) >= 1
    sizes = [len(f["nodes"]) for f in frames]
    assert sizes == sorted(sizes)  # strictly non-decreasing
    for i in range(len(frames) - 1):
        assert frames[i]["nodes"] <= frames[i + 1]["nodes"]  # nested/monotonic growth
    assert set(frames[-1]["nodes"]) == set(expected_final_nodes)
    assert [f["step"] for f in frames] == list(range(len(frames)))


def test_growth_frames_from_peeling(k4_bridge_chain):
    result = charikar_peeling(k4_bridge_chain)
    frames = growth_frames_from_peeling(result)
    _assert_valid_growth_frames(frames, result["best_nodes"])


def test_growth_frames_from_greedy_pp(k4_bridge_chain):
    result = greedy_plus_plus(k4_bridge_chain)
    frames = growth_frames_from_greedy_pp(result)
    _assert_valid_growth_frames(frames, result["best_nodes"])


def test_growth_frames_from_goldberg(k4_bridge_chain):
    result = goldberg_densest_subgraph(k4_bridge_chain)
    frames = growth_frames_from_goldberg(result, k4_bridge_chain)
    _assert_valid_growth_frames(frames, result["best_nodes"])


def test_growth_frames_from_triangle_density(k4):
    result = exact_triangle_densest_subgraph(k4)
    triangles = enumerate_triangles(k4)
    frames = growth_frames_from_triangle_density(result, k4, triangles)
    _assert_valid_growth_frames(frames, result["best_nodes"])


def test_graphml_export_roundtrip_preserves_attributes(k4_bridge_chain):
    result = charikar_peeling(k4_bridge_chain)
    frames = growth_frames_from_peeling(result)

    with tempfile.TemporaryDirectory() as tmp:
        export_growth_snapshots(k4_bridge_chain, frames, tmp)
        files = sorted(f for f in os.listdir(tmp) if f.endswith(".graphml"))
        assert len(files) == len(frames)

        first = nx.read_graphml(os.path.join(tmp, files[0]))
        last = nx.read_graphml(os.path.join(tmp, files[-1]))

        # node/edge counts of the exported subgraph match the final frame's
        # induced subgraph (attributes preserved through the GraphML round-trip)
        assert first.number_of_nodes() == len(frames[-1]["nodes"])
        assert last.number_of_nodes() == len(frames[-1]["nodes"])

        last_in_subgraph = {n for n, d in last.nodes(data=True) if str(d.get("in_subgraph")).lower() == "true"}
        assert last_in_subgraph == {str(n) for n in frames[-1]["nodes"]}
