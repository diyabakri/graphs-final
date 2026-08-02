"""
Temporal analysis module for Israeli Cinema Co-Stardom Graph.
Splits data into Eras A, B, and C, and computes global, structural, 
and distributional metrics per temporal subgraph.
"""

import os
import logging
from typing import Dict, Any, Tuple
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def split_graph_by_eras(df_edges: pd.DataFrame) -> Dict[str, nx.Graph]:
    """
    Splits the full edge dataframe into three temporal subgraphs:
    - Era A: Edges formed up to 1970
    - Era B: Edges formed between 1971 and 1990
    - Era C: Edges formed from 1991 to present
    """
    eras = {
        "Era_A_Pre1970": df_edges[df_edges["First_CoStar_Year"] <= 1970],
        "Era_B_1970_1990": df_edges[(df_edges["First_CoStar_Year"] > 1970) & (df_edges["First_CoStar_Year"] <= 1990)],
        "Era_C_1990_Present": df_edges[df_edges["First_CoStar_Year"] > 1990]
    }

    subgraphs = {}
    for era_name, df_sub in eras.items():
        G_sub = nx.Graph()
        for _, row in df_sub.iterrows():
            G_sub.add_edge(
                row["Source"], 
                row["Target"], 
                weight=row["Weight"],
                year=row["First_CoStar_Year"]
            )
        subgraphs[era_name] = G_sub
        logging.info(f"Sub-graph {era_name}: {G_sub.number_of_nodes()} Nodes, {G_sub.number_of_edges()} Edges")

    return subgraphs


def compute_era_metrics(G: nx.Graph, era_name: str) -> Dict[str, Any]:
    """
    Computes topology, connectivity, and degree distribution metrics for a given graph.
    """
    if G.number_of_nodes() == 0:
        return {"Era": era_name}

    # 1. Basic Metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)

    # 2. Connectivity Metrics
    connected_components = list(nx.connected_components(G))
    num_cc = len(connected_components)
    gcc_nodes = max(connected_components, key=len)
    gcc_size = len(gcc_nodes)
    G_gcc = G.subgraph(gcc_nodes)

    # Radius and Diameter are strictly defined on connected components
    radius = nx.radius(G_gcc) if gcc_size > 1 else 0
    diameter = nx.diameter(G_gcc) if gcc_size > 1 else 0

    # 3. Distribution Metrics
    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees))
    max_degree = int(np.max(degrees))

    return {
        "Era": era_name,
        "Nodes (|V|)": num_nodes,
        "Edges (|E|)": num_edges,
        "Density": round(density, 5),
        "Avg Clustering (C)": round(avg_clustering, 4),
        "Num CC": num_cc,
        "GCC Size": gcc_size,
        "GCC Ratio": round(gcc_size / num_nodes, 4),
        "GCC Radius": radius,
        "GCC Diameter": diameter,
        "Avg Degree <k>": round(avg_degree, 2),
        "Max Degree": max_degree
    }


def plot_degree_distributions(subgraphs: Dict[str, nx.Graph], output_path: str) -> None:
    """
    Generates comparison plots of degree distributions across eras.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (era_name, G_sub) in enumerate(subgraphs.items()):
        degrees = [d for _, d in G_sub.degree()]
        sns.histplot(
            degrees, 
            kde=True, 
            ax=axes[idx], 
            color=colors[idx], 
            bins=20, 
            stat="density"
        )
        axes[idx].set_title(f"Degree Distribution: {era_name}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Degree (k)")
        axes[idx].set_ylabel("Density")
        axes[idx].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Exported degree distribution plot to {output_path}")