# Data sources

All datasets are Collaboration Networks from the
[Network Repository](https://networkrepository.com/), downloaded automatically by
`src/data_io.py` from their direct `nrvis.com` mirrors. Raw `.mtx` files are
gitignored (re-download with `python -c "from src.data_io import load_dataset;
load_dataset('<name>')"`); this file lists the verified, real sizes actually
downloaded (which differ slightly from the course PDF's illustrative table for
three of the five — the PDF table appears to reference a different snapshot/
curation of these networks than what's currently hosted; see `README.md` at the
project root, "Datasets & limitations" section, for the full explanation).

| name | nodes | edges | tier | source page |
|---|---|---|---|---|
| ca-netscience | 379 | 914 | small | https://networkrepository.com/ca-netscience.php |
| ca-CSphd | 1,882 | 1,740 | small-medium | https://networkrepository.com/ca-CSphd.php |
| ca-GrQc | 4,158 | 13,422 | medium | https://networkrepository.com/ca-GrQc.php |
| ca-HepTh | 11,204 | 117,619 | large | https://networkrepository.com/ca-HepTh.php |
| ca-dblp-2012 | 317,080 | 1,049,866 | very-large | https://networkrepository.com/ca-dblp-2012.php |

All five are weighted (pattern-format files default to weight 1.0), undirected,
connected-ish co-authorship graphs (see `load_graph_mtx` in `src/data_io.py` for
loader details, including two robustness fixes over the reference notebook's
original `.mtx` loader).
