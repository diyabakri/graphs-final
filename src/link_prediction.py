"""
Link Prediction Machine Learning Engine (Step 5).
Features: Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment,
Shortest Path, Centrality Products, and SVD Adjacency Matrix Embeddings.
"""

import logging
import random
from typing import Dict, Any, List, Tuple
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_temporal_splits(
    df_edges: pd.DataFrame, 
    train_cutoff_year: int = 2020
) -> Tuple[nx.Graph, List[Tuple[str, str]]]:
    """
    Splits graph temporally into:
    - Base Training Graph G_train (edges up to current_cutoff)
    - Ground Truth Future Edges E_test_pos (new edges formed after current_cutoff)
    
    If requested cutoff year yields fewer than 5 future edges between existing actors,
    it steps back dynamically to earlier years until a valid test split is guaranteed.
    """
    df_filtered = df_edges[df_edges["First_CoStar_Year"].notnull()].copy()

    current_cutoff = train_cutoff_year
    min_year = int(df_filtered["First_CoStar_Year"].min())

    while current_cutoff > min_year:
        df_train_edges = df_filtered[df_filtered["First_CoStar_Year"] <= current_cutoff]
        df_future_edges = df_filtered[df_filtered["First_CoStar_Year"] > current_cutoff]

        # Build Training Graph
        G_train = nx.Graph()
        for _, row in df_train_edges.iterrows():
            G_train.add_edge(row["Source"], row["Target"], weight=row["Weight"])

        train_nodes = set(G_train.nodes())

        # Identify Positive Future Edges (both endpoints must exist in G_train)
        E_test_pos = []
        for _, row in df_future_edges.iterrows():
            u, v = row["Source"], row["Target"]
            if u in train_nodes and v in train_nodes and not G_train.has_edge(u, v):
                E_test_pos.append(tuple(sorted((u, v))))

        E_test_pos = list(set(E_test_pos))

        if len(E_test_pos) >= 5:
            logging.info(
                f"Using Temporal Cutoff Year: {current_cutoff} | "
                f"G_train Edges: {G_train.number_of_edges()} | "
                f"Ground-Truth Test Edges: {len(E_test_pos)}"
            )
            return G_train, E_test_pos

        # Step back cutoff year if future test edges are sparse
        current_cutoff -= 2

    # Fallback to Random Edge Masking Split if temporal split is too sparse
    logging.warning(
        "Could not find a valid temporal cutoff year with sufficient future co-stardom edges. "
        "Falling back to random edge-masking split."
    )

    G_full = nx.Graph()
    for _, row in df_edges.iterrows():
        G_full.add_edge(row["Source"], row["Target"], weight=row.get("Weight", 1))

    all_edges = list(G_full.edges())
    random.seed(42)

    # Prefer edges where both endpoints have degree > 1 so nodes remain in G_train
    candidate_edges = [
        (u, v) for u, v in all_edges 
        if G_full.degree(u) > 1 and G_full.degree(v) > 1
    ]

    test_count = max(5, int(len(all_edges) * 0.2))
    if len(candidate_edges) >= test_count:
        E_test_pos = random.sample(candidate_edges, test_count)
    else:
        E_test_pos = random.sample(all_edges, min(test_count, len(all_edges)))

    E_test_pos = [tuple(sorted((u, v))) for u, v in E_test_pos]

    G_train = G_full.copy()
    G_train.remove_edges_from(E_test_pos)

    logging.info(
        f"Using Random Edge Split | "
        f"G_train Edges: {G_train.number_of_edges()} | "
        f"Ground-Truth Test Edges: {len(E_test_pos)}"
    )

    return G_train, E_test_pos



def compute_node_embeddings_svd(G: nx.Graph, dimensions: int = 16) -> Dict[str, np.ndarray]:
    """
    Computes node embeddings via Matrix Factorization (Truncated SVD on Adjacency Matrix).
    """
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return {}

    adj_matrix = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float)

    n_components = min(dimensions, max(1, len(nodes) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings_matrix = svd.fit_transform(adj_matrix)

    embeddings = {nodes[i]: embeddings_matrix[i] for i in range(len(nodes))}
    return embeddings


def extract_pair_features(
    G: nx.Graph, 
    node_pairs: List[Tuple[str, str]], 
    embeddings: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Extracts topological and embedding-based features for node pairs (u, v):
    - Common Neighbors
    - Jaccard Coefficient
    - Adamic-Adar Index
    - Preferential Attachment
    - Shortest Path Length
    - Centrality Products
    - Embedding Similarity
    """
    feature_cols = [
        "u", "v", "common_neighbors", "jaccard", "adamic_adar",
        "preferential_attachment", "shortest_path", "deg_centrality_prod",
        "bet_centrality_prod", "embedding_similarity"
    ]

    if not node_pairs:
        return pd.DataFrame(columns=feature_cols)

    deg_centrality = nx.degree_centrality(G)
    bet_centrality = nx.betweenness_centrality(G)
    feature_rows = []

    for u, v in node_pairs:
        cn = len(list(nx.common_neighbors(G, u, v)))
        
        jaccard = list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]
        adamic_adar = list(nx.adamic_adar_index(G, [(u, v)]))[0][2]
        pref_attach = list(nx.preferential_attachment(G, [(u, v)]))[0][2]

        try:
            shortest_path = nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            shortest_path = 99

        deg_prod = deg_centrality[u] * deg_centrality[v]
        bet_prod = bet_centrality[u] * bet_centrality[v]

        emb_dim = next(iter(embeddings.values())).shape[0] if embeddings else 16
        vec_u = embeddings.get(u, np.zeros(emb_dim))
        vec_v = embeddings.get(v, np.zeros(emb_dim))
        norm_u, norm_v = np.linalg.norm(vec_u), np.linalg.norm(vec_v)
        emb_sim = np.dot(vec_u, vec_v) / (norm_u * norm_v) if norm_u > 0 and norm_v > 0 else 0.0

        feature_rows.append({
            "u": u,
            "v": v,
            "common_neighbors": cn,
            "jaccard": jaccard,
            "adamic_adar": adamic_adar,
            "preferential_attachment": pref_attach,
            "shortest_path": shortest_path,
            "deg_centrality_prod": deg_prod,
            "bet_centrality_prod": bet_prod,
            "embedding_similarity": emb_sim
        })

    return pd.DataFrame(feature_rows)


def prepare_dataset(
    G_train: nx.Graph, 
    E_pos: List[Tuple[str, str]], 
    neg_ratio: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Constructs a balanced dataset with positive links and sampled negative non-links.
    """
    if len(E_pos) == 0:
        raise ValueError("Ground truth positive edges list (E_pos) is empty. Check data split.")

    non_edges = list(nx.non_edges(G_train))
    num_negatives = max(int(len(E_pos) * neg_ratio), 10)

    random.seed(42)
    E_neg = random.sample(non_edges, min(num_negatives, len(non_edges)))

    # Embeddings
    embeddings = compute_node_embeddings_svd(G_train, dimensions=16)

    # Feature extraction
    df_pos_feats = extract_pair_features(G_train, E_pos, embeddings)
    df_pos_feats["label"] = 1

    df_neg_feats = extract_pair_features(G_train, E_neg, embeddings)
    df_neg_feats["label"] = 0

    df_dataset = pd.concat([df_pos_feats, df_neg_feats], ignore_index=True)
    df_dataset = df_dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Feature matrix X safely excluding u, v, label
    X = df_dataset.drop(columns=["u", "v", "label"], errors="ignore")
    y = df_dataset["label"].values

    return df_dataset, X, y


def train_and_evaluate_model(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    X_test: pd.DataFrame, 
    y_test: np.ndarray
) -> Tuple[Any, Dict[str, float], np.ndarray]:
    """
    Trains a Random Forest Classifier and computes performance metrics.
    """
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1_Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_test, y_proba), 4) if len(set(y_test)) > 1 else 0.5
    }

    return clf, metrics, y_proba


def perform_error_analysis(
    df_test: pd.DataFrame, 
    y_proba: np.ndarray, 
    top_k: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Performs qualitative error analysis identifying True Positives, False Positives, and False Negatives.
    """
    df_analysis = df_test.copy()
    df_analysis["prediction_score"] = y_proba

    tp_df = df_analysis[(df_analysis["label"] == 1)].sort_values(
        by="prediction_score", ascending=False
    ).head(top_k)

    fp_df = df_analysis[(df_analysis["label"] == 0)].sort_values(
        by="prediction_score", ascending=False
    ).head(top_k)

    fn_df = df_analysis[(df_analysis["label"] == 1)].sort_values(
        by="prediction_score", ascending=True
    ).head(top_k)

    return {
        "Top_True_Positives": tp_df[["u", "v", "prediction_score", "common_neighbors", "jaccard"]],
        "Top_False_Positives": fp_df[["u", "v", "prediction_score", "common_neighbors", "jaccard"]],
        "Top_False_Negatives": fn_df[["u", "v", "prediction_score", "common_neighbors", "jaccard"]]
    }