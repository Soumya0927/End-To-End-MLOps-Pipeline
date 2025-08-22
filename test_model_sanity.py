import unittest
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

class TestModelAUC(unittest.TestCase):
    def test_model_auc_on_sample_data(self):
        # Load model
        model = joblib.load("rf_model.pkl")

        # Load the sample data
        df = pd.read_csv("data/samples.csv")

        # Features and true labels (same as in train.py)
        X = df.drop(columns=["Class"])
        y_true = df["Class"]

        # Predict probabilities
        y_proba = model.predict_proba(X)[:, 1]

        # Calculate AUC
        auc = roc_auc_score(y_true, y_proba)
        print(f"AUC Score (sanity): {auc:.4f}")

        # Assert AUC sanity check
        self.assertGreater(auc, 0.5, f"Sanity check failed: AUC={auc:.2f}, expected > 0.5")

if __name__ == "__main__":
    unittest.main()

