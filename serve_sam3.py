"""
SAM3 Inference Service
Based on HuggingFace MLflow serving cookbook pattern
"""

import logging
import os
from typing import List

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAM3 Inference Service", version="1.0.0")

# Global model variable
model = None
model_version = None


class PredictionRequest(BaseModel):
    image: str  # base64 encoded
    boxes: List[List[float]]
    threshold: float = 0.5
    mask_threshold: float = 0.5


@app.on_event("startup")
async def load_model():
    """Load model once at startup - following HF cookbook pattern"""
    global model, model_version

    # Setup MLflow connection
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    try:
        model_name = "sam3-inference"
        logger.info(f"Loading model '{model_name}'...")

        # Get latest version
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            raise ValueError(f"Model '{model_name}' not found in registry!")

        model_version = versions[0].version
        run_id = versions[0].run_id

        logger.info(f"Found version {model_version} (run: {run_id})")

        # Load model directly using mlflow.pyfunc.load_model()
        # This bypasses all the pyenv/conda issues!
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

        logger.info(f"✓ Model loaded successfully (version {model_version})")

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "SAM3 Inference API",
        "model": "sam3-inference",
        "version": model_version,
        "status": "running",
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_version": model_version,
    }


@app.post("/invocations")
async def predict(request: PredictionRequest):
    """
    Prediction endpoint - MLflow compatible

    Compatible with your existing SAM3InferenceClient!
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Call model.predict() directly (like HF cookbook example)
        result = model.predict(request.dict())
        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
