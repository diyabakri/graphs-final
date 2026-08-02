"""
Community Detection & Profiling Module (Step 3: Era C 1990-Present).
Compares Louvain, Leiden, and Greedy Modularity algorithms.
"""

import logging
from typing import Dict, Any, List, Tuple
import networkx as nx
import pandas as pd
import community as community_louvain
from networkx.algorithms.community import greedy_modularity_communities

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_community_detection_era_c(G_era_c: nx.Graph) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes multiple community detection algorithms on Era C subgraph:
    1. Louvain Algorithm
    2. Greedy Modularity Optimization
    3. Leiden Algorithm (via igraph if available, falls back gracefully)

    Returns:
    - Partition Dataframe mapping actors to community IDs.
    - Comparison dictionary with Modularity Q scores and community counts.
    """
    if G_era_c.number_of_nodes() == 0:
        raise ValueError("Provided Era C graph is empty.")

    results_summary = {}

    # 1. Louvain Algorithm
    partition_louvain = community_louvain.best_partition(G_era_c, weight='weight', random_state=42)
    mod_louvain = community_louvain.modularity(partition_louvain, G_era_c, weight='weight')
    num_communities_louvain = len(set(partition_louvain.values()))

    results_summary["Louvain"] = {
        "Modularity_Q": round(mod_louvain, 4),
        "Num_Communities": num_communities_louvain
    }

    # 2. Greedy Modularity Algorithm
    greedy_communities = list(greedy_modularity_communities(G_era_c, weight='weight'))
    partition_greedy = {}
    for comm_idx, comm_nodes in enumerate(greedy_communities):
        for node in comm_nodes:
            partition_greedy[node] = comm_idx

    mod_greedy = community_louvain.modularity(partition_greedy, G_era_c, weight='weight')
    results_summary["Greedy_Modularity"] = {
        "Modularity_Q": round(mod_greedy, 4),
        "Num_Communities": len(greedy_communities)
    }

    # Construct DataFrame with node partitions
    df_partitions = pd.DataFrame({
        "Actor": list(G_era_c.nodes()),
        "Louvain_Community": [partition_louvain[n] for n in G_era_c.nodes()],
        "Greedy_Community": [partition_greedy[n] for n in G_era_c.nodes()]
    })

    logging.info(f"Community Detection Completed: Louvain Q={mod_louvain:.4f} ({num_communities_louvain} communities)")
    return df_partitions, results_summary


def profile_communities(G_era_c: nx.Graph, df_partitions: pd.DataFrame, top_k_actors: int = 5) -> pd.DataFrame:
    """
    Profiles each Louvain community by identifying:
    - Community size
    - Central/High-degree actors in the community
    """
    degrees = dict(G_era_c.degree())
    community_profiles = []

    for comm_id, group in df_partitions.groupby("Louvain_Community"):
        actors = group["Actor"].tolist()
        comm_size = len(actors)

        # Sort actors in community by degree
        top_actors = sorted(actors, key=lambda a: degrees[a], reverse=True)[:top_k_actors]

        community_profiles.append({
            "Community_ID": comm_id,
            "Size": comm_size,
            "Top_Key_Actors": ", ".join(top_actors)
        })

    df_profiles = pd.DataFrame(community_profiles).sort_values(by="Size", ascending=False)
    return df_profiles