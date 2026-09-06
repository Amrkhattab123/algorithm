"""
Timing harness across the full (dataset x algorithm) matrix, plus the
combined runtime-growth plots (vs. nodes and vs. edges, all four algorithms
on one legend) the course PDF asks for.

Sampling policy (documented, reproducible, applied only where empirically
necessary -- never silently, always recorded in the results table):

- Charikar and Greedy++ are near-linear and run at FULL SIZE on every
  dataset, including the very-large ca-dblp-2012.
- Goldberg's exact max-flow algorithm runs at full size on
  ca-netscience/ca-CSphd/ca-GrQc/ca-HepTh (empirically verified tractable,
  ca-HepTh's full 11,204-node/117,619-edge network completes in ~2 minutes);
  on ca-dblp-2012 it uses a BFS/snowball sample (seed=42, ~10,000 nodes).
- The exact triangle-density algorithm's flow network scales with the
  TRIANGLE COUNT, not the node count, so it needs a much more aggressive,
  per-dataset sample: ca-netscience and ca-CSphd run at full size (both have
  a tiny handful of triangles), but ca-GrQc (47,779 triangles at full size)
  and ca-HepTh (3.36 MILLION triangles at full size -- empirically confirmed
  intractable within minutes) and ca-dblp-2012 are each BFS-sampled down to a
  size chosen so the sampled subgraph's triangle count stays in the
  low-thousands (verified empirically below), keeping the flow network small
  enough to solve in seconds.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

import os

from algorithms.charikar import charikar_peeling
from algorithms.common import enumerate_triangles
from algorithms.goldberg_maxflow import goldberg_densest_subgraph
from algorithms.greedy_pp import greedy_plus_plus
from algorithms.triangle_density import exact_triangle_densest_subgraph
from data_io import DATASET_REGISTRY, load_dataset, sample_subgraph
from visualization import (
    export_growth_snapshots,
    growth_frames_from_goldberg,
    growth_frames_from_greedy_pp,
    growth_frames_from_peeling,
    growth_frames_from_triangle_density,
    render_growth_video,
)

# Per-dataset sample size for Goldberg, applied only to datasets listed here.
GOLDBERG_SAMPLE_TARGET_N = {
    "ca-dblp-2012": 10000,
}

# Per-dataset sample size for the triangle-density algorithm, applied only to
# datasets listed here (chosen empirically so triangle count stays tractable).
TRIANGLE_SAMPLE_TARGET_N = {
    "ca-GrQc": 500,
    "ca-HepTh": 250,
    "ca-dblp-2012": 1000,
}

SAMPLE_SEED = 42

ALGORITHMS = ["Charikar", "Greedy++", "Goldberg", "TriangleDensity"]


def _prepare_variant(G: nx.Graph, name: str, sample_map: dict):
    if name in sample_map:
        return sample_subgraph(G, seed=SAMPLE_SEED, target_n=sample_map[name]), True
    return G, False


def _maybe_render(name, algo, G_variant, frames, video_dir, outputs_dir, generate_visuals):
    if not generate_visuals or not frames:
        return
    video_path = os.path.join(video_dir, f"{name}_{algo}.mp4")
    render_growth_video(G_variant, frames, video_path, title=f"{name} — {algo}")
    export_growth_snapshots(G_variant, frames, os.path.join(outputs_dir, name, algo))


def run_one(
    name: str,
    G: nx.Graph,
    generate_visuals: bool = False,
    video_dir: str = "videos",
    outputs_dir: str = "outputs",
    max_frames: int = 20,
) -> tuple[list[dict], dict]:
    """Run all four algorithms on dataset `name` (graph already loaded as `G`),
    once each -- timing recorded from that single run, and (if
    `generate_visuals`) the same run's growth frames used for the video and
    GraphML export, so nothing is computed twice.

    Returns (rows, raw_results) where `raw_results` maps algorithm name to
    its full result dict (e.g. for a Greedy++ convergence plot)."""
    rows = []
    raw_results = {}

    t0 = time.perf_counter()
    charikar_result = charikar_peeling(G)
    charikar_time = time.perf_counter() - t0
    rows.append({
        "dataset": name, "algorithm": "Charikar", "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(), "runtime": charikar_time,
        "density": charikar_result["best_density"], "subgraph_size": len(charikar_result["best_nodes"]),
        "sampled": False,
    })
    raw_results["Charikar"] = charikar_result
    _maybe_render(name, "Charikar", G, growth_frames_from_peeling(charikar_result, max_frames),
                  video_dir, outputs_dir, generate_visuals)

    t0 = time.perf_counter()
    gpp_result = greedy_plus_plus(G)
    gpp_time = time.perf_counter() - t0
    rows.append({
        "dataset": name, "algorithm": "Greedy++", "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(), "runtime": gpp_time,
        "density": gpp_result["best_density"], "subgraph_size": len(gpp_result["best_nodes"]),
        "sampled": False,
    })
    raw_results["Greedy++"] = gpp_result
    _maybe_render(name, "GreedyPP", G, growth_frames_from_greedy_pp(gpp_result, max_frames),
                  video_dir, outputs_dir, generate_visuals)

    G_gb, gb_sampled = _prepare_variant(G, name, GOLDBERG_SAMPLE_TARGET_N)
    goldberg_result = goldberg_densest_subgraph(G_gb)
    rows.append({
        "dataset": name, "algorithm": "Goldberg", "nodes": G_gb.number_of_nodes(),
        "edges": G_gb.number_of_edges(), "runtime": goldberg_result["search_time"],
        "density": goldberg_result["best_density"], "subgraph_size": len(goldberg_result["best_nodes"]),
        "sampled": gb_sampled,
    })
    raw_results["Goldberg"] = goldberg_result
    _maybe_render(name, "Goldberg", G_gb, growth_frames_from_goldberg(goldberg_result, G_gb, max_frames),
                  video_dir, outputs_dir, generate_visuals)

    G_td, td_sampled = _prepare_variant(G, name, TRIANGLE_SAMPLE_TARGET_N)
    triangle_result = exact_triangle_densest_subgraph(G_td)
    rows.append({
        "dataset": name, "algorithm": "TriangleDensity", "nodes": G_td.number_of_nodes(),
        "edges": G_td.number_of_edges(), "runtime": triangle_result["search_time"],
        "density": triangle_result["best_density"], "subgraph_size": len(triangle_result["best_nodes"]),
        "sampled": td_sampled,
    })
    raw_results["TriangleDensity"] = triangle_result
    if generate_visuals and triangle_result["n_triangles_total"] > 0:
        triangles_td = enumerate_triangles(G_td)
        frames = growth_frames_from_triangle_density(triangle_result, G_td, triangles_td, max_frames)
        _maybe_render(name, "TriangleDensity", G_td, frames, video_dir, outputs_dir, generate_visuals)

    return rows, raw_results


def run_full_benchmark(
    dataset_names: list[str] | None = None,
    verbose: bool = True,
    generate_visuals: bool = False,
    video_dir: str = "videos",
    outputs_dir: str = "outputs",
) -> tuple[pd.DataFrame, dict]:
    """Run the full (dataset x algorithm) matrix once each. If
    `generate_visuals` is True, also renders each run's incremental-
    construction video and GraphML snapshots as a side effect of that same
    run (no algorithm is ever executed twice).

    Returns (results_df, all_raw_results) where all_raw_results maps
    (dataset, algorithm) -> full result dict."""
    if dataset_names is None:
        dataset_names = list(DATASET_REGISTRY.keys())

    all_rows = []
    all_raw_results = {}
    for name in dataset_names:
        if verbose:
            print(f"=== {name} ===")
        G = load_dataset(name)
        rows, raw_results = run_one(
            name, G, generate_visuals=generate_visuals, video_dir=video_dir, outputs_dir=outputs_dir
        )
        all_rows.extend(rows)
        for algo, result in raw_results.items():
            all_raw_results[(name, algo)] = result
        if verbose:
            for row in rows:
                flag = " (sampled)" if row["sampled"] else ""
                print(f"  {row['algorithm']:16s} n={row['nodes']:>7} m={row['edges']:>8} "
                      f"runtime={row['runtime']:8.3f}s density={row['density']:.4f}{flag}")

    return pd.DataFrame(all_rows), all_raw_results


def plot_runtime_growth(results_df: pd.DataFrame, out_path_nodes: str, out_path_edges: str):
    """One combined log-log plot per x-axis (nodes, edges), all four
    algorithms on one legend, as the course PDF requires."""
    for x_col, out_path, x_label in [("nodes", out_path_nodes, "Nodes"), ("edges", out_path_edges, "Edges")]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for algo in ALGORITHMS:
            subset = results_df[results_df["algorithm"] == algo].sort_values(x_col)
            if subset.empty:
                continue
            ax.plot(subset[x_col], subset["runtime"], marker="o", label=algo)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{x_label} (log scale)")
        ax.set_ylabel("Runtime (s, log scale)")
        ax.set_title(f"Densest-Subgraph Algorithm Runtime vs. {x_label}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_greedy_pp_convergence(round_best_density: list[float], out_path: str, dataset_name: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(round_best_density)), round_best_density, marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel("Best density seen so far in round")
    ax.set_title(f"Greedy++ Convergence — {dataset_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
