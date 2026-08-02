"""
Execution script for Step 5: Link Prediction Machine Learning Pipeline.
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.link_prediction import (
    create_temporal_splits, 
    prepare_dataset, 
    train_and_evaluate_model, 
    perform_error_analysis
)

PROCESSED_CSV_PATH = "data/processed/israeli_actors_edges.csv"
MODEL_METRICS_PATH = "data/processed/link_prediction_metrics.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    if not os.path.exists(PROCESSED_CSV_PATH):
        raise FileNotFoundError(f"Missing input dataset: {PROCESSED_CSV_PATH}. Run Step 1 first.")

    logging.info("--- Executing Step 5: Link Prediction Machine Learning Engine ---")
    df_edges = pd.read_csv(PROCESSED_CSV_PATH)

    # 1. Temporal Data Split (G_train <= 2020, Future edges 2021-2025; falls back dynamically if sparse)
    G_train, E_test_pos = create_temporal_splits(df_edges, train_cutoff_year=2020)

    # 2. Build Dataset & Extract Features
    logging.info("Extracting topological and embedding features...")
    df_dataset, X, y = prepare_dataset(G_train, E_test_pos, neg_ratio=1.0)

    if len(X) == 0:
        raise ValueError("Extracted feature matrix X is empty. Verify input edges dataset.")

    # 3. Train/Test Split (handles small datasets safely)
    test_size = 0.3 if len(X) >= 20 else 0.2
    
    # Stratify only if both classes have enough samples
    stratify_target = y if len(set(y)) > 1 and min(pd.Series(y).value_counts()) >= 2 else None

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_dataset.index, test_size=test_size, random_state=42, stratify=stratify_target
    )

    df_test = df_dataset.loc[idx_test].copy()

    # 4. Train Model & Evaluate
    logging.info("Training Random Forest link prediction classifier...")
    clf, metrics, y_proba = train_and_evaluate_model(X_train, y_train, X_test, y_test)

    # Export Metrics
    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame([metrics]).to_csv(MODEL_METRICS_PATH, index=False)
    logging.info(f"Saved evaluation metrics to {MODEL_METRICS_PATH}")

    # 5. Perform Qualitative Error Analysis
    error_analysis_results = perform_error_analysis(df_test, y_proba, top_k=5)

    print("\n" + "="*80)
    print("STEP 5: LINK PREDICTION EVALUATION METRICS")
    print("="*80)
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")
    print("="*80)

    print("\n" + "="*80)
    print("QUALITATIVE ERROR ANALYSIS (TOP INSTANCES)")
    print("="*80)

    print("\n--- Top Realized Predictions (True Positives) ---")
    print(error_analysis_results["Top_True_Positives"].to_string(index=False))

    print("\n--- Strong Predictions That Didn't Materialize (False Positives) ---")
    print(error_analysis_results["Top_False_Positives"].to_string(index=False))

    print("\n--- Missed Realized Links (False Negatives) ---")
    print(error_analysis_results["Top_False_Negatives"].to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()