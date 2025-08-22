import unittest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class TestModelMetrics(unittest.TestCase):
    def test_model_metrics_on_sample_data(self):
        # Load Decision Tree model
        model = joblib.load("dt_model.joblib")

        # Load the sample data
        df = pd.read_csv("data/samples.csv")

        # Features and true labels (same as in train.py)
        X = df.drop(columns=["Class"])
        y_true = df["Class"]

        # Predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba)

        # Print all metrics so they appear in CML report
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")

        # Assertions (basic sanity checks)
        self.assertGreater(auc, 0.5, f"AUC too low: {auc:.4f}")
        self.assertGreaterEqual(accuracy, 0.5, f"Accuracy too low: {accuracy:.4f}")

if __name__ == "__main__":
    unittest.main()

