"""
Centrality Analysis Module (Step 4).
Computes Degree, Betweenness, Closeness, and Eigenvector centralities.
"""

import logging
from typing import Dict, Any
import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_all_centralities(G: nx.Graph) -> pd.DataFrame:
    """
    Computes 4 primary centrality metrics for all nodes in graph G:
    1. Degree Centrality
    2. Betweenness Centrality
    3. Closeness Centrality
    4. Eigenvector Centrality
    """
    logging.info("Calculating Degree Centrality...")
    deg_cent = nx.degree_centrality(G)

    logging.info("Calculating Betweenness Centrality...")
    between_cent = nx.betweenness_centrality(G, weight='weight')

    logging.info("Calculating Closeness Centrality...")
    close_cent = nx.closeness_centrality(G)

    logging.info("Calculating Eigenvector Centrality...")
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
    except nx.PowerIterationFailedConvergence:
        logging.warning("Eigenvector centrality power iteration failed to converge. Falling back to numpy solver.")
        eigen_cent = nx.eigenvector_centrality_numpy(G, weight='weight')

    # Construct compiled metric DataFrame
    df_centrality = pd.DataFrame({
        "Actor": list(G.nodes()),
        "Degree": [G.degree(n) for n in G.nodes()],
        "Degree_Centrality": [deg_cent[n] for n in G.nodes()],
        "Betweenness_Centrality": [between_cent[n] for n in G.nodes()],
        "Closeness_Centrality": [close_cent[n] for n in G.nodes()],
        "Eigenvector_Centrality": [eigen_cent[n] for n in G.nodes()]
    })

    return df_centrality


def get_top_k_centrality_tables(df_centrality: pd.DataFrame, top_k: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Extracts top K actors for each centrality metric.
    """
    metrics = [
        "Degree_Centrality", 
        "Betweenness_Centrality", 
        "Closeness_Centrality", 
        "Eigenvector_Centrality"
    ]

    top_tables = {}
    for metric in metrics:
        df_sorted = df_centrality.sort_values(by=metric, ascending=False).head(top_k)
        top_tables[metric] = df_sorted[["Actor", metric, "Degree"]]

    return top_tables