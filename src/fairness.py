# file: src/fairness.py

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import mlflow
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
import argparse
import os

def eval_performance_metrics(y_true, y_pred, y_prob):
    """Calculates and returns a dictionary of standard performance metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob)
    }

def main(args):
    """
    Main function to run the fairness and interpretability audit.
    """
    print("--- Running Fairness and Interpretability Audit ---")

    # --- 1. Data Preparation ---
    print("\nStep 1: Preparing data...")
    df = pd.read_csv(args.input_file)
    rng = np.random.default_rng(args.seed)
    sensitive_col = 'location'
    df[sensitive_col] = rng.integers(0, 2, size=len(df))
    print(f"Added sensitive attribute '{sensitive_col}'.")

    # --- 2. Data Splitting ---
    print("\nStep 2: Splitting data for training and testing...")
    feature_cols = [c for c in df.columns if c not in {args.target_col, sensitive_col}]
    X = df[feature_cols]
    y = df[args.target_col]
    sensitive_features = df[sensitive_col]
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive_features, test_size=0.3, stratify=y, random_state=args.seed
    )
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # --- 3. Model Training ---
    print("\nStep 3: Training Decision Tree model...")
    model = DecisionTreeClassifier(random_state=args.seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Model training complete.")

    # --- 4. MLflow Experiment Logging ---
    mlflow.set_experiment("Fairness_and_Interpretability_Audit")
    with mlflow.start_run(run_name="DecisionTree_with_Location"):
        print("\nStep 4: Logging results to MLflow...")
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("sensitive_attribute", sensitive_col)

        performance_metrics = eval_performance_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(performance_metrics)
        print(f"Logged performance metrics.")

        # --- 5. SHAP Explainability ---
        print("\nStep 5: Generating and logging SHAP plots...")
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_test)
        
        # Generate and log beeswarm plot for the positive class (class 1)
        plt.figure()
        shap.plots.beeswarm(explanation[:, :, 1], show=False)
        summary_plot_path = "shap_beeswarm_plot.png"
        plt.savefig(summary_plot_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(summary_plot_path, "shap_plots")
        print("Logged SHAP beeswarm plot.")

        # --- NEW: Generate and log the interactive stacked force plot ---
        print("Generating interactive force plot for the first 500 samples...")
        num_samples_for_plot = min(30, len(X_test))
        # Pass a slice of the explanation object for multiple samples and the positive class
        interactive_force_plot = shap.plots.force(explanation[:num_samples_for_plot, :, 1])
        interactive_plot_path = "interactive_force_plot.html"
        shap.save_html(interactive_plot_path, interactive_force_plot)
        mlflow.log_artifact(interactive_plot_path, "shap_plots")
        print("Logged interactive force plot as HTML.")


        # --- 6. Fairlearn Auditing ---
        print("\nStep 6: Evaluating and logging fairness metrics...")
        metric_frame = MetricFrame(
            metrics={'selection_rate': selection_rate, 'accuracy': accuracy_score},
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_test
        )

        for metric, values in metric_frame.by_group.items():
            for group, value in values.items():
                mlflow.log_metric(f"{metric}_location_{group}", value)
        print("Logged fairness metrics by group.")

        disparity_metrics = {
            "demographic_parity_difference": demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test),
            "equalized_odds_difference": equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_test)
        }
        mlflow.log_metrics(disparity_metrics)
        print(f"Logged disparity metrics.")

    print("\n--- Audit complete. Check the 'Fairness_and_Interpretability_Audit' experiment in MLflow. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a fairness and interpretability audit for a model.")
    parser.add_argument('--input-file', type=str, default='data/processed/version_1.csv', help='Path to the input dataset.')
    parser.add_argument('--target-col', type=str, default='Class', help='Name of the target column.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    args = parser.parse_args()
    main(args)

