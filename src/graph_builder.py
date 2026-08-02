"""
Graph construction and export module.
"""

import itertools
import logging
from typing import Dict, List, Any
import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_co_star_graph(movie_records: List[Dict[str, Any]]) -> nx.Graph:
    """
    Constructs an undirected NetworkX graph where:
    - Nodes represent actors.
    - Edges represent co-starring in a movie.
    - Edge attributes store list of shared movies and years.
    """
    G = nx.Graph()

    for record in movie_records:
        title = record["title"]
        year = record["year"]
        actors = record["actors"]

        if year is None or len(actors) < 2:
            continue

        # Add node attributes
        for actor in actors:
            if not G.has_node(actor):
                G.add_node(actor, label=actor)

        # Generate all pairwise co-stardom combinations
        for actor1, actor2 in itertools.combinations(actors, 2):
            if G.has_edge(actor1, actor2):
                # Update existing edge
                G[actor1][actor2]["movies"].append(title)
                G[actor1][actor2]["years"].append(year)
                G[actor1][actor2]["weight"] += 1
                # Keep earliest year as baseline or store list
                G[actor1][actor2]["min_year"] = min(G[actor1][actor2]["min_year"], year)
            else:
                # Add new edge
                G.add_edge(
                    actor1, 
                    actor2, 
                    movies=[title], 
                    years=[year], 
                    min_year=year, 
                    weight=1
                )

    logging.info(f"Graph Construction Complete: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges.")
    return G

def export_graph_data(G: nx.Graph, edge_csv_path: str, gexf_path: str) -> None:
    """
    Exports the graph to CSV for tabular processing and GEXF for Gephi visualization.
    """
    # Export Edges CSV
    edge_rows = []
    for u, v, data in G.edges(data=True):
        edge_rows.append({
            "Source": u,
            "Target": v,
            "Weight": data["weight"],
            "First_CoStar_Year": data["min_year"],
            "Movies": "; ".join(data["movies"]),
            "Years": "; ".join(map(str, data["years"]))
        })
    
    df_edges = pd.DataFrame(edge_rows)
    df_edges.to_csv(edge_csv_path, index=False, encoding="utf-8-sig")
    logging.info(f"Exported edges CSV to {edge_csv_path}")

    # Export GEXF for Gephi (convert lists to strings for XML compatibility)
    G_export = G.copy()
    for u, v, data in G_export.edges(data=True):
        data["movies"] = ", ".join(data["movies"])
        data["years"] = ", ".join(map(str, data["years"]))

    nx.write_gexf(G_export, gexf_path)
    logging.info(f"Exported Gephi GEXF file to {gexf_path}")