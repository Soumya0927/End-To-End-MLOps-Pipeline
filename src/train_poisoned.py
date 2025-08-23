# file: src/train_poisoned.py

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import argparse
import os

def eval_metrics(y_true, y_pred, y_prob):
    """Calculate and return a dictionary of metrics."""
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

def main(data_dir, poison_levels, target_col, seed, model_type):
    """
    Train models on poisoned datasets and log each as a separate run
    within the same experiment for easy comparison.
    """
    # All runs will be logged to this single experiment
    experiment_name = "Data Poisoning Robustness"
    mlflow.set_experiment(experiment_name)
    
    print(f"Logging runs to experiment: '{experiment_name}'")

    # Loop through each poisoning level
    for level in poison_levels:
        poison_pct = int(level * 100)
        
        # Start a new MLflow run for this specific poison level
        with mlflow.start_run(run_name=f"{model_type}_poisoned_{poison_pct}pct"):
            print(f"\n--- Starting run for {poison_pct}% poison level ---")
            
            # Log parameters for this run
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("poison_level_pct", poison_pct)
            
            # Load the specific poisoned dataset
            file_path = os.path.join(data_dir, f"poisoned_{poison_pct}pct.csv")
            df = pd.read_csv(file_path)
            
            # Split data
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            
            # Initialize and train the specified model
            if model_type == "DecisionTree":
                model = DecisionTreeClassifier(random_state=seed)
            else:
                raise ValueError("Unsupported model type specified.")
                
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = eval_metrics(y_test, y_pred, y_prob)
            
            # Log all metrics to MLflow for this run
            mlflow.log_metrics(metrics)
            print(f"Logged metrics for {poison_pct}% poison: {metrics}")
            
            # Log the model itself
            mlflow.sklearn.log_model(model, "model")

    print(f"\nAll runs have been logged to the '{experiment_name}' experiment.")
    print("You can now compare them in the MLflow UI.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on poisoned data for comparison.")
    parser.add_argument("--data-dir", type=str, default="data/poisoned", help="Directory containing the poisoned datasets.")
    parser.add_argument("--target-col", type=str, default="Class", help="Name of the target label column.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--model-type", type=str, default="DecisionTree", choices=["DecisionTree"], help="Type of model to train.")
    args = parser.parse_args()
    
    poison_levels = [0.02, 0.08, 0.15, 0.30]
    main(args.data_dir, poison_levels, args.target_col, args.seed, args.model_type)

