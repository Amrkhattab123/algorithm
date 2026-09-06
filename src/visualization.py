"""
Turns each algorithm's raw output into a common "growth frame" sequence,
exports that sequence to GraphML (one file per frame, mirroring the
reference notebook's per-snapshot GraphML export pattern), and renders it to
an MP4 video via matplotlib + imageio.

Growth-frame convention used everywhere in this module: a list of dicts
    {"step": int, "nodes": frozenset, "density": float}
where `nodes` is monotonically non-decreasing from frame to frame, starting
small and ending at exactly the algorithm's discovered densest-subgraph
vertex set -- the "grown till completed" sequence the course PDF asks for,
for every one of the four algorithms, however differently each one produces
it internally (reverse-replayed peeling order vs. a nested max-flow sweep).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import imageio.v2 as imageio

from algorithms.common import density as edge_density
from algorithms.common import triangle_density as tri_density


# ---------------------------------------------------------------------------
# Adapters: raw algorithm output -> common growth-frame format
# ---------------------------------------------------------------------------

def _stride(seq, max_frames):
    """Keep at most `max_frames` items from `seq`, always including the first and last."""
    n = len(seq)
    if n <= max_frames:
        return list(range(n))
    step = (n - 1) / (max_frames - 1)
    return sorted({round(i * step) for i in range(max_frames)})


def growth_frames_from_peeling(result: dict, max_frames: int = 20) -> list[dict]:
    """Charikar / Greedy++: reverse-replay the shrink sequence's tail from the
    best-density snapshot through to the final singleton, played backwards.
    """
    snapshots = result["snapshots"]
    best_index = result["best_index"]
    tail = list(reversed(snapshots[best_index:]))  # growing: singleton -> S*
    indices = _stride(tail, max_frames)
    return [
        {"step": step, "nodes": frozenset(tail[i][0]), "density": tail[i][2]}
        for step, i in enumerate(indices)
    ]


def growth_frames_from_greedy_pp(result: dict, max_frames: int = 20) -> list[dict]:
    """Same reverse-replay idea, applied to the winning round's snapshot sequence."""
    best_round = result["best_round"]
    snapshots = result["round_snapshots"][best_round]
    best_index = result["round_best_index"][best_round]
    tail = list(reversed(snapshots[best_index:]))
    indices = _stride(tail, max_frames)
    return [
        {"step": step, "nodes": frozenset(tail[i][0]), "density": tail[i][2]}
        for step, i in enumerate(indices)
    ]


def growth_frames_from_goldberg(result: dict, G: nx.Graph, max_frames: int = 20) -> list[dict]:
    """Goldberg: the nested-cuts sweep is already a growing sequence; just
    attach the actual recomputed edge-density of each frame's node set."""
    frames = result["growth_frames"]
    indices = _stride(frames, max_frames)
    return [
        {"step": step, "nodes": frozenset(frames[i][1]), "density": edge_density(G, frames[i][1])}
        for step, i in enumerate(indices)
    ]


def growth_frames_from_triangle_density(result: dict, G: nx.Graph, triangles: list, max_frames: int = 20) -> list[dict]:
    frames = result["growth_frames"]
    indices = _stride(frames, max_frames)
    return [
        {"step": step, "nodes": frozenset(frames[i][1]), "density": tri_density(G, frames[i][1], triangles)}
        for step, i in enumerate(indices)
    ]


# ---------------------------------------------------------------------------
# GraphML export (per-frame, induced subgraph only -- keeps file size small
# even when the parent graph is huge, since growth-frame node sets are the
# discovered densest subgraph, not the whole input graph)
# ---------------------------------------------------------------------------

def export_growth_snapshots(G: nx.Graph, frames: list[dict], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    final_nodes = frames[-1]["nodes"]
    H_full = G.subgraph(final_nodes)

    for frame in frames:
        H = nx.Graph()
        H.add_nodes_from(H_full.nodes(data=True))
        H.add_edges_from(H_full.edges(data=True))
        for n in H.nodes():
            H.nodes[n]["in_subgraph"] = bool(n in frame["nodes"])
        H.graph["step"] = frame["step"]
        H.graph["density"] = frame["density"]
        path = os.path.join(out_dir, f"snapshot_{frame['step']:04d}.graphml")
        nx.write_graphml(H, path)
    return out_dir


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------

def render_growth_video(
    G: nx.Graph,
    frames: list[dict],
    out_path: str,
    title: str,
    fps: int = 2,
    seed: int = 42,
    node_size: int = 60,
    figsize: tuple = (8, 6),
) -> str:
    """Render `frames` (common growth-frame format) into an MP4 at `out_path`.

    Layout is computed ONCE on the induced subgraph of the final frame's node
    set (i.e. the discovered densest subgraph) and reused for every frame, so
    nodes don't jump around between frames -- only their color/edge
    highlighting changes as the set grows.
    """
    final_nodes = frames[-1]["nodes"]
    H_final = G.subgraph(final_nodes)
    pos = nx.spring_layout(H_final, seed=seed)
    all_edges = list(H_final.edges())

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    images = []

    for frame in frames:
        current = frame["nodes"]
        fig, ax = plt.subplots(figsize=figsize)

        nx.draw_networkx_edges(H_final, pos, edgelist=all_edges, alpha=0.15, width=1, ax=ax)
        current_edges = [(u, v) for u, v in all_edges if u in current and v in current]
        nx.draw_networkx_edges(H_final, pos, edgelist=current_edges, edge_color="crimson", width=1.5, ax=ax)

        others = [n for n in H_final.nodes() if n not in current]
        nx.draw_networkx_nodes(H_final, pos, nodelist=others, node_color="lightgray", node_size=node_size, ax=ax)
        nx.draw_networkx_nodes(H_final, pos, nodelist=list(current), node_color="seagreen", node_size=node_size, ax=ax)

        ax.set_title(f"{title}\nstep {frame['step']}: |S|={len(current)}, density={frame['density']:.3f}")
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        image = _fig_to_array(fig)
        images.append(image)
        plt.close(fig)

    # hold the final frame a little longer so viewers can read the result
    images += [images[-1]] * (fps * 1)

    imageio.mimsave(out_path, images, fps=fps)
    return out_path


def _fig_to_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    import numpy as np
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return arr[:, :, :3]
