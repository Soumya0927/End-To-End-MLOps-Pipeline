import pandas as pd
import numpy as np
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import joblib

def main(train_path, val_path, mlflow_uri):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("fraud_detection_training")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    target = "Class"
    features = [col for col in train_df.columns if col != target]
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    
    with mlflow.start_run(run_name="dt_baseline_v1") as run:
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:,1]
        
        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)
        
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        })
        
        # Confusion matrix plot
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")
        
        # Training report
        report = f"""
        ## Training Report (Decision Tree)
        
        Accuracy: {accuracy:.4f}
        F1 Score: {f1:.4f}
        Precision: {precision:.4f}
        Recall: {recall:.4f}
        AUC: {auc:.4f}
        Confusion Matrix:
        {cm}
        """
        report_path = "training_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="reports")
        
        # Save model with joblib
        local_model_path = "dt_model.joblib"
        joblib.dump(clf, local_model_path)
        print(f"Model saved locally as {local_model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(clf, "dt_model")

        print("Run completed. Run ID:", run.info.run_id)
        print("Training report generated:", report_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--mlflow_uri", type=str, required=True)
    args = parser.parse_args()
    
    main(args.train, args.val, args.mlflow_uri)

