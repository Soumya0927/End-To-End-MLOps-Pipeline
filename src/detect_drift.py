# file: src/detect_drift.py

import pandas as pd
import joblib
import mlflow
import argparse
import os

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


def detect_drift(reference_path, current_path, model_path=None, target_col=None):
    print("\n--- Starting Data Drift Analysis ---\n")

    # 1. Load data (model optional)
    print("Loading reference & current datasets...")
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    if model_path and os.path.exists(model_path):
        joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        print("No model loaded (model_path optional).\n")

    # 2. Run Evidently Report
    print("Running Evidently Report...")
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    dashboard = report.run(reference_data=reference_df, current_data=current_df)

    # 3. Save reports
    html_path = "evidently_report.html"
    json_path = "evidently_report.json"
    dashboard.save_html(html_path)
    dashboard.save_json(json_path)
    print(f"Saved report: {html_path}, {json_path}")

    # 4. Extract dictionary or JSON for metrics
    report_dict = dashboard.dict()
    report_json = dashboard.json()
    print("Extracted report via .dict() and .json() successfully.")

    # Optional: extract a metric (e.g., share of drifted features)
    drift_value = None
    for item in report_dict.get("metrics", []):
        if item.get("metric") == "DataDriftPreset":
            drift_value = item["result"].get("dataset_drift", {}).get("share_drifted_features")
            break
    if drift_value is not None:
        print(f"Share of drifted features: {drift_value:.4f}")

    # 5. Log to MLflow
    print("\nLogging reports to MLflow...")
    mlflow.set_experiment("Model Drift Monitoring")
    with mlflow.start_run(run_name="Evidently Drift Report"):
        mlflow.log_param("reference_data", os.path.basename(reference_path))
        mlflow.log_param("current_data", os.path.basename(current_path))
        if drift_value is not None:
            mlflow.log_metric("share_drifted_features", drift_value)
        mlflow.log_artifact(html_path)
        mlflow.log_artifact(json_path)
        print("Reports and metrics logged to MLflow.")

    print("\n--- Drift analysis complete. ---")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Run data drift detection using Evidently and log dictionary results.") 
    parser.add_argument('--reference-data', type=str, default='data/processed/version_1.csv', help='Path to the reference dataset.') 
    parser.add_argument('--current-data', type=str, default='data/processed/version_2.csv', help='Path to the current dataset.') 
    parser.add_argument('--model-path', type=str, default='dt_model.joblib', help='Path to the trained model file.') 
    parser.add_argument('--target-col', type=str, default='Class', help='Name of the target column.') 
    args = parser.parse_args() 
    detect_drift(args.reference_data, args.current_data, args.model_path, args.target_col)
