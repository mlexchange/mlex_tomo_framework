"""
Script to register SAM3 model to MLflow Model Registry
FIXED: Added conda_env=None to prevent pyenv requirement
"""

import os
import sys
from pathlib import Path

import mlflow

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI_OUTSIDE", "http://localhost:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
SAM_MODEL_NAME = os.getenv("SAM_MODEL_NAME", "facebook/sam3")
HF_TOKEN = os.getenv("HF_TOKEN")

# Set up MLflow authentication
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

# Skip model loading during registration
os.environ["MLFLOW_DISABLE_MODEL_LOADING"] = "true"

print("=" * 80)
print("SAM3 Model Registration to MLflow (NO CONDA)")
print("=" * 80)
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"SAM Model Name: {SAM_MODEL_NAME}")
print(f"HF Token configured: {'Yes' if HF_TOKEN else 'No'}")
print("=" * 80)

# Log the model
with mlflow.start_run(run_name="sam3-inference-service") as run:
    print("\n[1/4] Creating SAM3 model instance...")
    from utils.sam3_mlflow_model import SAM3MLflowModel

    sam3_model = SAM3MLflowModel()
    print("✓ Model instance created")

    print("\n[2/4] Defining environment...")
    # Define dependencies - MUST match Dockerfile.sam3-inference exactly
    pip_requirements = [
        "mlflow==2.22.0",
        "requests",
        "torch==2.2.2",
        "torchvision==0.17.2",
        "pillow==12.0.0",
        "huggingface-hub",
        "numpy<2.0.0",
        "transformers @ git+https://github.com/huggingface/transformers.git",
    ]
    print("✓ Environment defined")

    print("\n[3/4] Logging model to MLflow...")

    # CRITICAL FIX: Add conda_env=None to prevent conda.yaml creation
    mlflow.pyfunc.log_model(
        artifact_path="sam3_model",
        python_model=sam3_model,
        pip_requirements=pip_requirements,
        conda_env=None,  # ← THIS IS THE KEY FIX - prevents pyenv requirement
        registered_model_name="sam3-inference",
        code_paths=["utils"],
        signature=None,
    )
    print("✓ Model logged WITHOUT conda environment")

    print("\n[4/4] Registration complete!")
    print("=" * 80)
    print(f"Run ID: {run.info.run_id}")
    print(f"Model URI: runs:/{run.info.run_id}/sam3_model")
    print(f"Registered Model: sam3-inference")
    print("=" * 80)

    print("\nTo serve this model:")
    print(
        f"mlflow models serve -m 'models:/sam3-inference/latest' -p 5001 --host 0.0.0.0 --env-manager local --no-conda"
    )
