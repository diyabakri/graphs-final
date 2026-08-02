import os
import pandas as pd
import logging
from src.temporal_analysis import (
    split_graph_by_eras, 
    compute_era_metrics, 
    plot_degree_distributions
)

PROCESSED_CSV_PATH = "data/processed/israeli_actors_edges.csv"
METRICS_OUTPUT_CSV = "data/processed/temporal_metrics_summary.csv"
METRICS_LATEX_PATH = "data/processed/temporal_metrics_table.tex"
DISTRIBUTION_PLOT_PATH = "data/processed/degree_distributions.png"

def main():
    if not os.path.exists(PROCESSED_CSV_PATH):
        raise FileNotFoundError(f"Missing input dataset: {PROCESSED_CSV_PATH}. Run Step 1 first.")

    logging.info("Loading processed edges dataset...")
    df_edges = pd.read_csv(PROCESSED_CSV_PATH)

    # 1. Segment graph into eras
    subgraphs = split_graph_by_eras(df_edges)

    # 2. Compute metrics for each era
    metrics_list = []
    for era_name, G_sub in subgraphs.items():
        metrics = compute_era_metrics(G_sub, era_name)
        metrics_list.append(metrics)

    df_summary = pd.DataFrame(metrics_list)

    # 3. Export Metric Tables
    df_summary.to_csv(METRICS_OUTPUT_CSV, index=False)
    logging.info(f"Exported metrics CSV to {METRICS_OUTPUT_CSV}")

    # Generate LaTeX table snippet for direct inclusion in assignment PDF report
    with open(METRICS_LATEX_PATH, "w") as f:
        f.write(df_summary.to_latex(index=False, caption="Temporal Metrics Comparison Across Eras", label="tab:temporal_metrics"))
    logging.info(f"Exported LaTeX table to {METRICS_LATEX_PATH}")

    # 4. Plot & Save Degree Distributions
    plot_degree_distributions(subgraphs, DISTRIBUTION_PLOT_PATH)

    print("\n" + "="*80)
    print("TEMPORAL GRAPH ANALYSIS SUMMARY")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()