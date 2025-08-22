import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Define the data folder path relative to project root
def test_model_auc():
    # Load a small test set and trained model (mock or actual)

# Get project root folder from the file location
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Adjust number of `.parent` levels as needed

# Define the data folder path relative to project root
    DATA_DIR = PROJECT_ROOT / "data"
    
    sample_path = os.path.join(DATA_DIR, "samples.csv")
    X = pd.read_csv(sample_path)
    y = pd.Series([0, 1, 0, 1])

    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    y_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_proba)
    assert auc > 0.5, f"Sanity check failed: AUC={auc:.2f}, expected >0.5"
    print(f"AUC Score (sanity): {auc:.4f}")
