"""
Dataset registry, downloader, and graph loading utilities for the
Collaboration Networks densest-subgraph project.

Datasets are pulled from the Network Repository (https://networkrepository.com),
whose files are actually hosted as .mtx-in-.zip archives at nrvis.com. Only
`ca-dblp-2012`'s slug needed live verification (the course PDF's table calls it
"ca-DBLP"); the other four match the PDF table's names exactly.
"""
from __future__ import annotations

import io
import os
import random
import zipfile
from collections import deque
from dataclasses import dataclass, field

import networkx as nx
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

NRVIS_URL_TEMPLATES = [
    "https://nrvis.com/download/data/ca/{slug}.zip",
    "https://nrvis.com/download/data/{slug}.zip",
]


@dataclass
class DatasetSpec:
    name: str                 # canonical name used throughout the project
    candidates: list          # ordered list of real nrvis.com slugs to try
    tier: str                 # "small" | "small-medium" | "medium" | "large" | "very-large"
    expected_nodes: int       # from the course PDF table, for a sanity check only
    expected_edges: int
    snapshot_interval: int = 1  # used by visualization.py for frame striding


# Ordered small -> very-large, matching the course PDF's Collaboration Networks table.
#
# NOTE on expected_nodes/expected_edges: the course PDF gives illustrative sizes
# (e.g. ca-HepTh "9,877 / 25,998"), but the actual files currently hosted on
# networkrepository.com/nrvis.com for these exact slugs are somewhat different
# (verified by downloading and parsing them: ca-netscience matches exactly;
# ca-CSphd, ca-GrQc, ca-HepTh do not, likely a different repository snapshot/
# curation than whatever the PDF table was drawn from). We use each file's own
# real, verified (n, m) as ground truth here rather than the PDF's numbers, so
# the sanity check in `sanity_check()` reflects reality and doesn't false-alarm.
# This discrepancy is documented in README.md.
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "ca-netscience": DatasetSpec(
        name="ca-netscience", candidates=["ca-netscience"], tier="small",
        expected_nodes=379, expected_edges=914, snapshot_interval=1,
    ),
    "ca-CSphd": DatasetSpec(
        name="ca-CSphd", candidates=["ca-CSphd"], tier="small-medium",
        expected_nodes=1882, expected_edges=1740, snapshot_interval=2,
    ),
    "ca-GrQc": DatasetSpec(
        name="ca-GrQc", candidates=["ca-GrQc"], tier="medium",
        expected_nodes=4158, expected_edges=13422, snapshot_interval=10,
    ),
    "ca-HepTh": DatasetSpec(
        name="ca-HepTh", candidates=["ca-HepTh"], tier="large",
        expected_nodes=11204, expected_edges=117619, snapshot_interval=100,
    ),
    "ca-dblp-2012": DatasetSpec(
        name="ca-dblp-2012",
        candidates=["ca-dblp-2012", "ca-dblp-2010", "ca-coauthors-dblp"],
        tier="very-large",
        expected_nodes=317080, expected_edges=1049866, snapshot_interval=500,
    ),
}


def download_dataset(spec: DatasetSpec, raw_dir: str = RAW_DIR, force: bool = False) -> str:
    """Download+extract a dataset's .mtx file. Returns the local .mtx path.

    Tries each candidate slug in order against nrvis.com's known URL shapes;
    never silently substitutes a different dataset than the one requested —
    raises with every URL attempted if all candidates fail.
    """
    os.makedirs(raw_dir, exist_ok=True)
    dest_path = os.path.join(raw_dir, f"{spec.name}.mtx")
    if os.path.exists(dest_path) and not force:
        return dest_path

    attempted = []
    for slug in spec.candidates:
        for template in NRVIS_URL_TEMPLATES:
            url = template.format(slug=slug)
            attempted.append(url)
            try:
                resp = requests.get(url, timeout=60)
            except requests.RequestException as exc:
                attempted[-1] += f"  (request error: {exc})"
                continue
            if resp.status_code != 200:
                attempted[-1] += f"  (HTTP {resp.status_code})"
                continue
            if not zipfile.is_zipfile(io.BytesIO(resp.content)):
                attempted[-1] += "  (not a valid zip)"
                continue

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                mtx_members = [m for m in zf.namelist() if m.lower().endswith(".mtx")]
                if not mtx_members:
                    attempted[-1] += "  (zip has no .mtx member)"
                    continue
                with zf.open(mtx_members[0]) as member_file:
                    data = member_file.read()
                with open(dest_path, "wb") as out:
                    out.write(data)
            print(f"[data_io] '{spec.name}' downloaded via slug '{slug}' -> {dest_path}")
            return dest_path

    raise RuntimeError(
        f"Could not download dataset '{spec.name}'. Attempted URLs:\n  "
        + "\n  ".join(attempted)
    )


def _as_int_triplet(tokens: list) -> tuple | None:
    """Return (a, b, c) if tokens is exactly 3 integer-looking strings, else None."""
    if len(tokens) != 3:
        return None
    try:
        return tuple(int(t) for t in tokens)
    except ValueError:
        return None


def load_graph_mtx(path: str) -> nx.Graph:
    """Load a Matrix Market (.mtx) file into a weighted, undirected, self-loop-free graph.

    Two robustness fixes vs. the reference notebook's loader (which this project's
    pipeline pattern is otherwise modeled on):

    1. That loader requires a 3rd (weight) column and silently *drops every edge*
       of "pattern"-format .mtx files (2 columns, no weight). Here, a missing
       weight column defaults to 1.0 instead of skipping the edge.
    2. That loader assumes the "rows cols nnz" dimension line is always the
       first non-"%" line. Several real networkrepository.com files (e.g.
       ca-CSphd, ca-HepTh) instead put the dimension line *inside* a "%"-prefixed
       comment block (e.g. "% 11204 11204 117619"), which would otherwise get
       skipped as a comment and cause the first real data line to be misread as
       the dimension line, silently dropping one edge. Here the dimension line
       is identified by content (first line, comment or not, past the banner,
       whose stripped tokens are exactly 3 integers) rather than by position.
    """
    G = nx.Graph()
    dimension_line_found = False

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("%%MatrixMarket"):
                continue  # banner line, never the dimension line

            is_comment = line.startswith("%")
            candidate = line.lstrip("%").strip()

            if not dimension_line_found:
                triplet = _as_int_triplet(candidate.split())
                if triplet is not None:
                    dimension_line_found = True
                    continue
                if is_comment:
                    continue  # ordinary comment before the dimension line

            if is_comment:
                continue  # comment line after the dimension line

            parts = line.split()
            if len(parts) < 2:
                continue

            u, v = int(parts[0]), int(parts[1])
            w = abs(float(parts[2])) if len(parts) >= 3 else 1.0

            if u == v:
                continue  # drop self-loops

            if G.has_edge(u, v):
                # Some files list parallel/duplicate edges; keep the max weight seen.
                if w > G[u][v]["weight"]:
                    G[u][v]["weight"] = w
            else:
                G.add_edge(u, v, weight=w)

    return G


def sanity_check(spec: DatasetSpec, G: nx.Graph) -> None:
    n, m = G.number_of_nodes(), G.number_of_edges()
    if abs(n - spec.expected_nodes) > max(5, 0.02 * spec.expected_nodes) or \
       abs(m - spec.expected_edges) > max(5, 0.05 * spec.expected_edges):
        print(
            f"[data_io] WARNING: '{spec.name}' loaded as n={n}, m={m}, "
            f"expected ~n={spec.expected_nodes}, ~m={spec.expected_edges} "
            f"(mtx files can differ slightly from repository summary tables; "
            f"large discrepancies may indicate a wrong download)."
        )
    else:
        print(f"[data_io] '{spec.name}' OK: n={n}, m={m} (expected ~{spec.expected_nodes}/{spec.expected_edges})")


def load_dataset(name: str, raw_dir: str = RAW_DIR, force_download: bool = False) -> nx.Graph:
    spec = DATASET_REGISTRY[name]
    path = download_dataset(spec, raw_dir=raw_dir, force=force_download)
    G = load_graph_mtx(path)
    sanity_check(spec, G)
    return G


def sample_subgraph(G: nx.Graph, seed: int = 42, target_n: int = 12000) -> nx.Graph:
    """Deterministic BFS/snowball sample of ~target_n nodes from G.

    Used only for the exact max-flow-based algorithms (Goldberg, triangle-density)
    on the very-large dataset, so the parametric max-flow search finishes in a
    reasonable time. Documented explicitly wherever it is used — never applied
    silently.
    """
    if G.number_of_nodes() <= target_n:
        return G.copy()

    rng = random.Random(seed)
    seed_node = rng.choice(list(G.nodes()))

    visited_order = []
    visited = {seed_node}
    queue = deque([seed_node])
    while queue and len(visited_order) < target_n:
        u = queue.popleft()
        visited_order.append(u)
        neighbors = list(G.neighbors(u))
        rng.shuffle(neighbors)
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                queue.append(v)

    nodes = visited_order[:target_n]
    return G.subgraph(nodes).copy()
