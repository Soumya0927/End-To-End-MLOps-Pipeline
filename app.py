from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import logging
import os
from typing import Dict, Any
from pathlib import Path

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ FastAPI app ------------------
app = FastAPI(
    title="Fraud Detection API 💳",
    description="API for fraud transaction detection using RandomForestClassifier",
    version="1.0.0"
)

model = None
feature_names = None  # keep feature list consistent with training

# ------------------ Model loader ------------------
def get_model_path():
    """Get model path from local dir or env variable"""
    current_dir = Path(__file__).parent
    local_path = current_dir / "dt_model.joblib"
    if local_path.exists():
        return str(local_path)

    env_path = os.getenv("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    return "dt_model.joblib"

@app.on_event("startup")
async def load_model():
    """Load model & detect features"""
    global model, feature_names
    try:
        model_path = get_model_path()
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")

        # Extract number of features expected
        if hasattr(model, "n_features_in_"):
            logger.info(f"Model expects {model.n_features_in_} features")
    except Exception as e:
        logger.error(f"Model load failed: {str(e)}")
        raise e

# ------------------ Request schema ------------------
class TransactionInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 50554.0,
                "V1": -0.798671774485497,
                "V2": 1.18509290816488,
                "V3": 0.904547185437142,
                "V4": 0.694583749252136,
                "V5": 0.219040575098198,
                "V6": -0.319295358148838,
                "V7": 0.495236343903129,
                "V8": 0.139268600382104,
                "V9": -0.760213608856806,
                "V10": 0.17054681233401,
                "V11": 0.821998442387764,
                "V12": 0.46832176679532,
                "V13": -0.0575502861155432,
                "V14": 0.573006314979332,
                "V15": 0.358687757777865,
                "V16": -0.0116327063359799,
                "V17": -0.50456991587577,
                "V18": 0.722749576258471,
                "V19": 0.861540918769592,
                "V20": -0.081298289388587,
                "V21": 0.202287281499889,
                "V22": 0.578699296051212,
                "V23": -0.0922450213734442,
                "V24": 0.0137228520784373,
                "V25": -0.246466055551771,
                "V26": -0.38005716679397,
                "V27": -0.396030194077437,
                "V28": -0.112900666069163,
                "Amount": 4.18
            }
        }
    )

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# ------------------ Endpoints ------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Fraud Detection API 💳"}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict/")
def predict_fraud(data: TransactionInput) -> Dict[str, Any]:
    """Predict if transaction is fraud (1) or not (0)"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available")

        input_df = pd.DataFrame([data.model_dump()])

        prediction = int(model.predict(input_df)[0])
        proba = float(model.predict_proba(input_df)[0][1])

        return {
            "fraud_prediction": prediction,
            "fraud_probability": proba,
            "input_features": data.model_dump()
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ------------------ Entrypoint ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)

