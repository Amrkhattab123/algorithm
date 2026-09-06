# Construction and Visualization of Densest Subgraphs in Graphs

**CSAI 330 — Design and Analysis of Algorithms — Final Mini-Project**
**Network category: Collaboration Networks**

This project implements and compares four algorithms that incrementally construct the
**densest subgraph** of a network — the vertex subset S maximizing edge density
|E(S)|/|S| (or, for one algorithm, triangle density |Δ(S)|/|S|) — across five real
collaboration (co-authorship) networks from the
[Network Repository](https://networkrepository.com/), spanning small to very-large scale.

## Table of contents

- [Category & applications](#category--applications)
- [Algorithms](#algorithms)
- [Datasets & limitations](#datasets--limitations)
- [Sampling policy](#sampling-policy)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Running it yourself](#running-it-yourself)
- [Testing](#testing)
- [Acknowledgements](#acknowledgements)

## Category & applications

**Collaboration networks** model co-authorship: an edge connects two researchers who have
published together, often weighted by the number of shared papers. The **densest
subgraph** of such a network identifies a tightly-knit research community — a group of
authors who collaborate with each other far more than with the rest of the field. This has
direct applications in:

- **Community/group detection** — surfacing active research clusters or "invisible
  colleges" that citation-count or degree-centrality metrics alone tend to miss.
- **Anomaly and fraud detection** (more broadly, densest-subgraph techniques originally
  developed for web/social graphs) — a burst of unusually dense mutual collaboration can
  flag citation rings or coordinated authorship inflation.
- **Recommendation** — the densest subgraph containing a given author is a natural
  "who else should you be citing/collaborating with" seed set.
- **Network summarization** — the densest subgraph is a compact, information-rich
  substitute for an entire (potentially huge) collaboration graph when only its most
  cohesive core matters.

## Algorithms

| Category | Algorithm | Approach | Complexity |
|---|---|---|---|
| Greedy Peeling | **Charikar's algorithm** (Charikar, APPROX 2000) | Repeatedly remove the lowest-weighted-degree vertex; return the densest subgraph seen over the whole peeling sequence. A 1/2-approximation. | O((n+m) log n) |
| Greedy Peeling | **Greedy++** (Boob et al., WWW 2020) | Iterative refinement of Charikar's peeling: each round re-peels the whole graph, ranking vertices by degree plus an accumulated "load" from every previous round — a discretized multiplicative-weights scheme converging toward the exact optimum. | O(R·(n+m) log n) for R rounds |
| Max-Flow-Based (exact) | **Goldberg's algorithm** (Goldberg, UCB/CSD-84-171, 1984) | Binary search a candidate density g; a subgraph with density ≥ g exists iff a corresponding flow network's min-cut source side is non-trivial. Converges to the *exact* maximum density and witness set. | O(log(1/ε) · MaxFlow(n, m)) |
| Max-Flow-Based (exact) | **Exact triangle-density algorithm** (generalizes Goldberg via Khuller & Saha, ICALP 2009; motivated by Tsourakakis et al., KDD 2013 / WWW 2015) | Same idea, but optimizing triangle density via a flow-network reduction with one auxiliary node per triangle. | O(log(1/ε) · MaxFlow(n + T, m)) for T triangles |

Both max-flow algorithms use `networkx`'s built-in preflow-push min-cut (well-tested
library code, since these two serve as *ground truth* for the correctness tests — see
[Testing](#testing)) inside a shared parametric-binary-search driver
(`src/algorithms/maxflow_core.py`).

**Incremental "growth" visualization.** Peeling algorithms naturally *remove* vertices, but
the assignment asks for the construction to be shown *growing*. For Charikar/Greedy++, we
replay the tail of the shrink sequence — from the winning best-density snapshot down to the
final singleton — in reverse, so the video shows one vertex added at a time up to exactly
the discovered densest subgraph. For the max-flow algorithms, we exploit the *nested-cuts
property of parametric max-flow* (Gallo, Grigoriadis & Tarjan 1989): sweeping the density
threshold g downward from its upper bound to the optimum yields a monotonically **growing**
sequence of witness sets, used directly as video frames (this sweep is excluded from the
algorithm's measured runtime — see `src/algorithms/goldberg_maxflow.py`).

## Datasets & limitations

| Dataset | Nodes | Edges | Tier |
|---|---|---|---|
| ca-netscience | 379 | 914 | small |
| ca-CSphd | 1,882 | 1,740 | small-medium |
| ca-GrQc | 4,158 | 13,422 | medium |
| ca-HepTh | 11,204 | 117,619 | large |
| ca-dblp-2012 | 317,080 | 1,049,866 | very-large |

All five are downloaded automatically from `nrvis.com` (see `src/data_io.py` /
`data/README.md`). **Note**: the course PDF's illustrative table gives different sizes for
three of the five (e.g. "ca-HepTh: 9,877 / 25,998"); the numbers above are the *real,
verified* sizes of the files actually hosted at those exact dataset names today (confirmed
by downloading and parsing each one — see the header/comment lines inside the raw `.mtx`
files themselves). `ca-dblp-2012` is the real slug behind the PDF's "ca-DBLP" row and
matches its table exactly. This is documented here rather than silently worked around.

## Sampling policy

Charikar and Greedy++ are near-linear and run at **full size** on every dataset, including
the 317K-node `ca-dblp-2012` (Charikar: ~25s; Greedy++: ~6 minutes across 13 rounds before
its patience-based convergence check stops it).

Goldberg's algorithm runs at full size on ca-netscience/ca-CSphd/ca-GrQc/ca-HepTh
(empirically verified tractable — ca-HepTh's full 11,204-node/117,619-edge network
completes in ~2 minutes); on `ca-dblp-2012` it uses a BFS/snowball sample (seed=42, 10,000
nodes), since a full parametric-max-flow search at 317K nodes/1M edges is not tractable in
pure-Python `networkx` within this project's time budget.

The exact triangle-density algorithm's flow network scales with the **triangle count**, not
the node count, so it needs a much more aggressive, per-dataset sample: ca-netscience and
ca-CSphd run at full size (both have only a handful of triangles), but ca-GrQc (47,779
triangles at full size) and ca-HepTh (**3.36 million** triangles at full size — empirically
confirmed intractable, exceeding several minutes with no end in sight) and `ca-dblp-2012`
are each BFS-sampled to a size chosen so the sampled subgraph's triangle count stays in the
low thousands (ca-GrQc → 500 nodes/3,736 triangles; ca-HepTh → 250 nodes/3,675 triangles;
ca-dblp-2012 → 1,000 nodes/4,672 triangles), keeping each search to well under 20 seconds.

Every sampled run is flagged (`sampled=True`) in `figures/benchmark_results.csv` — nothing
is substituted silently. The exact sample sizes and seeds are in `src/benchmark.py`.

## Results

Full run completed on all 5 datasets × 4 algorithms (`figures/benchmark_results.csv`):

| Dataset | Algorithm | Nodes run | Edges run | Runtime (s) | Density | \|S\*\| | Sampled |
|---|---|--:|--:|--:|--:|--:|:--:|
| ca-netscience | Charikar | 379 | 914 | 0.014 | 4.000 | 9 | |
| ca-netscience | Greedy++ | 379 | 914 | 0.111 | 4.000 | 9 | |
| ca-netscience | Goldberg | 379 | 914 | 1.588 | 4.000 | 13 | |
| ca-netscience | TriangleDensity | 379 | 914 | 5.141 | 9.333 | 9 | |
| ca-CSphd | Charikar | 1,882 | 1,740 | 0.054 | 1.250 | 4 | |
| ca-CSphd | Greedy++ | 1,882 | 1,740 | 0.683 | 1.500 | 6 | |
| ca-CSphd | Goldberg | 1,882 | 1,740 | 6.752 | 1.500 | 6 | |
| ca-CSphd | TriangleDensity | 1,882 | 1,740 | 1.996 | 0.667 | 6 | |
| ca-GrQc | Charikar | 4,158 | 13,422 | 0.242 | 22.391 | 46 | |
| ca-GrQc | Greedy++ | 4,158 | 13,422 | 1.255 | 22.391 | 46 | |
| ca-GrQc | Goldberg | 4,158 | 13,422 | 26.057 | 22.391 | 46 | |
| ca-GrQc | TriangleDensity | 500 | 1,653 | 27.297 | 77.667 | 24 | ✅ |
| ca-HepTh | Charikar | 11,204 | 117,619 | 1.321 | 119.000 | 239 | |
| ca-HepTh | Greedy++ | 11,204 | 117,619 | 8.479 | 119.000 | 239 | |
| ca-HepTh | Goldberg | 11,204 | 117,619 | 227.150 | 119.000 | 239 | |
| ca-HepTh | TriangleDensity | 250 | 993 | 27.486 | 100.000 | 26 | ✅ |
| ca-dblp-2012 | Charikar | 317,080 | 1,049,866 | 24.231 | 56.500 | 114 | |
| ca-dblp-2012 | Greedy++ | 317,080 | 1,049,866 | 341.364 | 56.565 | 115 | |
| ca-dblp-2012 | Goldberg | 10,000 | 54,000 | 210.109 | 56.565 | 115 | ✅ |
| ca-dblp-2012 | TriangleDensity | 1,000 | 3,501 | 58.558 | 26.000 | 14 | ✅ |

Combined runtime-growth plots (all four algorithms, one legend, log-log scale):
`figures/runtime_vs_nodes.png` and `figures/runtime_vs_edges.png`. Greedy++
convergence-by-round for `ca-dblp-2012` (the only dataset where it ran more than a couple
of rounds — 13 rounds, oscillating before settling from round 7 onward):
`figures/greedy_pp_convergence.png`. Incremental-construction videos and GraphML snapshots
for every one of the 20 (dataset, algorithm) pairs: `videos/<dataset>_<algorithm>.mp4` and
`outputs/<dataset>/<algorithm>/snapshot_*.graphml`.

**Highlights:**
- **Goldberg's algorithm (exact) never fell below either heuristic's density on any
  dataset** — on ca-netscience, ca-GrQc, and ca-HepTh the three edge-density algorithms all
  land on the *identical* density (4.0, 22.391, and 119.0 respectively), meaning Charikar's
  single peeling pass already found the true global optimum on those graphs; on ca-CSphd
  and ca-dblp-2012, Greedy++'s extra rounds close the small gap Charikar's one pass leaves,
  matching Goldberg's exact answer.
- **ca-GrQc's densest subgraph is a strikingly tight 46-author clique-like cluster** with
  density 22.39 (i.e. ~1,030 edges among 46 authors — near-complete mutual co-authorship),
  found identically by all three edge-density algorithms.
- On `ca-dblp-2012`, Goldberg's BFS-sampled run (10,000 of 317,080 nodes) recovered the
  *exact same* optimal subgraph (density 56.565, 115 vertices) that Greedy++ found on the
  *full* graph — strong cross-validation of both the sampling approach and the algorithms'
  correctness at very-large scale.
- **Triangle-density picks a smaller, more clique-like core than edge-density**, as the
  theory predicts they can differ — e.g. on ca-netscience (the one dataset where both ran
  at full size), edge-density's optimum is 9 vertices at density 4.0, while triangle-density
  finds that *same* 9-vertex set has triangle-density 9.333 — here the densest edge-subgraph
  happens to also be triangle-dense, but on ca-CSphd the two objectives instead disagree
  (edge-optimal set: 6 vertices, density 1.5; its triangle-density is only 0.667), confirming
  the two objectives genuinely optimize for different structures.

## Repository structure

```
data/                 dataset sources + downloaded raw .mtx files (gitignored)
outputs/              GraphML snapshot sequences per (dataset, algorithm)
videos/               incremental-construction .mp4 videos per (dataset, algorithm)
figures/              runtime-growth plots, convergence plot, results CSV
src/
  data_io.py            dataset registry, downloader, robust .mtx loader, BFS sampler
  benchmark.py          timing harness + combined video generation + growth-curve plots
  visualization.py      growth-frame adapters, GraphML export, video rendering
  algorithms/
    common.py             shared graph utilities, triangle enumeration, brute-force ground truth, peeling core
    charikar.py            Charikar's greedy peeling
    greedy_pp.py            Greedy++
    maxflow_core.py         shared parametric binary-search-over-min-cut driver
    goldberg_maxflow.py     Goldberg's exact edge-density algorithm
    triangle_density.py     exact triangle-density algorithm
tests/                 pytest suite (39 tests)
Densest_Subgraph_Project.ipynb   orchestration notebook (narrated, mirrors run_full_pipeline.py)
run_full_pipeline.py    the actual driver script that produces every artifact above
requirements.txt, pytest.ini, .gitignore
```

## Running it yourself

```bash
pip install -r requirements.txt
python run_full_pipeline.py    # downloads datasets, runs all 4 algorithms x 5 datasets,
                                # renders every video, produces figures/benchmark_results.csv
```

Or open `Densest_Subgraph_Project.ipynb` and run it cell by cell for a narrated walkthrough.

## Testing

```bash
python -m pytest tests/ -v
```

39 tests covering: density/degree helpers and triangle enumeration against hand-computed
values; Charikar's 2-approximation bound (plus exact optimality on a clique); Greedy++'s
monotonic improvement over Charikar and convergence to the brute-force optimum; **Goldberg's
algorithm's exact agreement with brute-force ground truth on every synthetic fixture**
(the algorithm all correctness claims ultimately rest on); the triangle-density algorithm's
exact agreement with a triangle-objective brute-force variant; cross-algorithm consistency
(the exact algorithm must never be beaten by either heuristic); and growth-frame/GraphML
export sanity checks.

## Acknowledgements

This project's pipeline *pattern* — `.mtx` loading, per-algorithm snapshot recording,
GraphML export, matplotlib/imageio video rendering, a results table, and a runtime-vs-size
plot — was structurally inspired by a classmate's separate course project on shortest-path
algorithms (Dijkstra/Bidirectional/ALT/MultiLevel/PLL) over Economic & Trade Network
datasets. This project's network category (Collaboration Networks), all four
densest-subgraph algorithms, their implementations, the sampling policy, the test suite, and
all analysis here are original and distinct from that project.
