import os
import pandas as pd
import logging
import networkx as nx
from src.temporal_analysis import split_graph_by_eras
from src.community_analysis import run_community_detection_era_c, profile_communities
from src.centrality_analysis import compute_all_centralities, get_top_k_centrality_tables

PROCESSED_CSV_PATH = "data/processed/israeli_actors_edges.csv"
COMMUNITY_PROFILES_CSV = "data/processed/era_c_community_profiles.csv"
CENTRALITY_SUMMARY_CSV = "data/processed/actor_centralities.csv"

def main():
    if not os.path.exists(PROCESSED_CSV_PATH):
        raise FileNotFoundError(f"Missing input dataset: {PROCESSED_CSV_PATH}. Run Step 1 first.")

    logging.info("Loading processed graph dataset...")
    df_edges = pd.read_csv(PROCESSED_CSV_PATH)
    
    # Reconstruct full graph
    G_full = nx.Graph()
    for _, row in df_edges.iterrows():
        G_full.add_edge(row["Source"], row["Target"], weight=row["Weight"], year=row["First_CoStar_Year"])

    # -------------------------------------------------------------
    # STEP 3: Community Detection (Era C)
    # -------------------------------------------------------------
    logging.info("--- Executing Step 3: Community Detection (Era C) ---")
    subgraphs = split_graph_by_eras(df_edges)
    G_era_c = subgraphs["Era_C_1990_Present"]

    df_partitions, algo_results = run_community_detection_era_c(G_era_c)
    df_profiles = profile_communities(G_era_c, df_partitions)

    df_profiles.to_csv(COMMUNITY_PROFILES_CSV, index=False, encoding="utf-8-sig")
    logging.info(f"Saved Community Profiles to {COMMUNITY_PROFILES_CSV}")

    print("\n" + "="*80)
    print("STEP 3: COMMUNITY DETECTION ALGORITHM COMPARISON (ERA C)")
    print("="*80)
    print(pd.DataFrame(algo_results).T.to_string())
    print("\nTOP ERA C COMMUNITIES PROFILES:")
    print(df_profiles.head(5).to_string(index=False))
    print("="*80 + "\n")

    # -------------------------------------------------------------
    # STEP 4: Centrality Analysis (Full Network)
    # -------------------------------------------------------------
    logging.info("--- Executing Step 4: Centrality Metrics Calculation ---")
    df_centrality = compute_all_centralities(G_full)
    df_centrality.to_csv(CENTRALITY_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    top_tables = get_top_k_centrality_tables(df_centrality, top_k=10)

    print("\n" + "="*80)
    print("STEP 4: TOP 10 ACTORS BY CENTRALITY METRICS")
    print("="*80)
    for metric_name, df_top in top_tables.items():
        print(f"\n--- Top 10: {metric_name} ---")
        print(df_top.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()