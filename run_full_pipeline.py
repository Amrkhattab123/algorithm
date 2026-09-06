"""
Driver script: runs the full (dataset x algorithm) matrix once each, saving
timing results, runtime-growth plots, a Greedy++ convergence plot, and
per-(dataset, algorithm) incremental-construction videos + GraphML
snapshots. This is the actual computation behind
Densest_Subgraph_Project.ipynb -- run directly (not through a notebook
kernel, which isn't available in this environment) so the real deliverable
artifacts (figures/, videos/, outputs/) get produced; the notebook mirrors
this same code for a reader to re-run interactively.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from benchmark import run_full_benchmark, plot_runtime_growth, plot_greedy_pp_convergence

os.makedirs("figures", exist_ok=True)

print(">>> Running full benchmark (timing + video generation, one pass per algorithm)...")
results_df, raw_results = run_full_benchmark(generate_visuals=True)

results_df.to_csv("figures/benchmark_results.csv", index=False)
print("\n>>> Results table:")
print(results_df.to_string())

plot_runtime_growth(results_df, "figures/runtime_vs_nodes.png", "figures/runtime_vs_edges.png")
print("\n>>> Saved figures/runtime_vs_nodes.png and figures/runtime_vs_edges.png")

# Greedy++ convergence plot for the very-large dataset (the only one where
# Greedy++ ran more than a couple of rounds before stalling/converging)
gpp_result = raw_results[("ca-dblp-2012", "Greedy++")]
plot_greedy_pp_convergence(
    gpp_result["round_best_density"], "figures/greedy_pp_convergence.png", "ca-dblp-2012"
)
print(">>> Saved figures/greedy_pp_convergence.png")

print("\n>>> DONE")
