# file: app.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import logging
import os
from typing import Dict, Any
from pathlib import Path
import time

# --- OpenTelemetry Setup ---
# This requires you to have the opentelemetry packages installed:
# pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
# In a real production setup, you would use an OTLP exporter to send to a collector (e.g., Cloud Trace)
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter 

# For simplicity in this example, we'll print traces to the console.
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

# Initialize Tracer Provider
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter())) # Prints to console
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ FastAPI app ------------------
app = FastAPI(
    title="Fraud Detection API 💳",
    description="API for fraud transaction detection using a machine learning model",
    version="1.0.0"
)

# Instrument the FastAPI app automatically
FastAPIInstrumentor.instrument_app(app)

model = None

# ------------------ Model loader ------------------
def get_model_path():
    """Get model path from local dir or env variable"""
    model_filename = os.getenv("MODEL_NAME", "dt_model.joblib")
    return Path(__file__).parent / model_filename

@app.on_event("startup")
async def load_model():
    """Load model at startup"""
    global model
    try:
        model_path = get_model_path()
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model load failed: {str(e)}")
        # Keep the app running but log the error; readiness probe will fail.
        model = None

# ------------------ Request schema ------------------
class TransactionInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 50554.0, "V1": -0.798, "V2": 1.185, "V3": 0.904, "V4": 0.694,
                "V5": 0.219, "V6": -0.319, "V7": 0.495, "V8": 0.139, "V9": -0.760,
                "V10": 0.170, "V11": 0.821, "V12": 0.468, "V13": -0.057, "V14": 0.573,
                "V15": 0.358, "V16": -0.011, "V17": -0.504, "V18": 0.722, "V19": 0.861,
                "V20": -0.081, "V21": 0.202, "V22": 0.578, "V23": -0.092, "V24": 0.013,
                "V25": -0.246, "V26": -0.380, "V27": -0.396, "V28": -0.112, "Amount": 4.18
            }
        }
    )
    Time: float; V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float

# ------------------ Health Check Endpoints ------------------
@app.get("/isLive", tags=["Health Checks"])
def liveness_check():
    """
    Liveness probe: Checks if the application process is running.
    Should always return 200 OK if the server is up.
    """
    return {"status": "alive"}

@app.get("/isReady", tags=["Health Checks"])
def readiness_check():
    """
    Readiness probe: Checks if the application is ready to serve traffic.
    Fails if the model is not loaded.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable")
    return {"status": "ready", "model_loaded": True}

# ------------------ Prediction Endpoint ------------------
@app.post("/predict/", tags=["Predictions"])
def predict_fraud(data: TransactionInput) -> Dict[str, Any]:
    """Predict if a transaction is fraud (1) or not (0)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    # --- Start of Custom Span for Prediction Logic ---
    with tracer.start_as_current_span("prediction_logic") as span:
        start_time = time.perf_counter()
        
        try:
            # 1. Data Preparation Span
            with tracer.start_as_current_span("data_preparation"):
                input_df = pd.DataFrame([data.model_dump()])
                span.set_attribute("num_features", len(input_df.columns))

            # 2. Model Inference Span
            with tracer.start_as_current_span("model_inference"):
                prediction = int(model.predict(input_df)[0])
                proba = float(model.predict_proba(input_df)[0][1])
                span.set_attribute("prediction_result", prediction)
                span.set_attribute("prediction_probability", proba)

            end_time = time.perf_counter()
            span.set_attribute("prediction_duration_ms", (end_time - start_time) * 1000)
            
            return {
                "fraud_prediction": prediction,
                "fraud_probability": proba,
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, "Prediction failed"))
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
