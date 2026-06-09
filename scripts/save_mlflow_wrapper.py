"""
save_mlflow_wrapper.py
======================
Reads config_register.yaml, builds a list of models to register based on
the register switches, and registers each one to the MLflow Model Registry.

Usage:
    python save_mlflow_wrapper.py
    python save_mlflow_wrapper.py --config config_register.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv

from mlex_lightly.wrapper import LightlySegWrapper

load_dotenv(dotenv_path="../.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config_register.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI_OUTSIDE"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

    cache = cfg["cache"]
    os.environ["LIGHTLY_TRAIN_CACHE_DIR"]       = cache["lightly_train_cache_dir"]
    os.environ["LIGHTLY_TRAIN_MODEL_CACHE_DIR"] = cache["lightly_train_model_cache_dir"]
    os.environ["TORCH_HOME"]                    = cache["torch_home"]

    return cfg


# ---------------------------------------------------------------------------
# Build model list from register switches
# ---------------------------------------------------------------------------

def build_model_list(cfg: dict) -> list[dict]:
    """
    Builds a list of model dicts to register based on register switches.

    Each dict has:
        model_name : MLflow registry name (no /)
        type       : "pth", "lightly_train", or "custom"
        base_model : lightly_train model name (for logging)
        pt_path    : local file path (for pth/custom types)
    """
    reg = cfg["register"]
    vit = reg["vit"]
    cache_dir = Path(cfg["cache"]["lightly_train_model_cache_dir"])
    models = []

    if reg.get("eomt"):
        # backbone .pth file — named dinov3_<vit>_lvd1689m.pth in lightly_cache
        pth_path = cache_dir / f"dinov3_{vit}_lvd1689m.pth"
        models.append({
            "model_name": f"dinov3_{vit}_eomt",
            "type":       "pth",
            "base_model": f"dinov3/{vit}-eomt",
            "pt_path":    str(pth_path),
        })

    if reg.get("eomt_coco"):
        models.append({
            "model_name": f"dinov3_{vit}_eomt_coco",
            "type":       "lightly_train",
            "base_model": f"dinov3/{vit}-eomt-coco",
            "pt_path":    None,
        })

    if reg.get("eomt_cityscapes"):
        models.append({
            "model_name": f"dinov3_{vit}_eomt_cityscapes",
            "type":       "lightly_train",
            "base_model": f"dinov3/{vit}-eomt-cityscapes",
            "pt_path":    None,
        })

    if reg.get("eomt_cityscapes_petiole"):
        pt_path = cfg["custom"]["pt_path"]
        models.append({
            "model_name": f"dinov3_{vit}_eomt_cityscapes_petiole",
            "type":       "custom",
            "base_model": f"dinov3_{vit}_eomt_cityscapes_petiole",
            "pt_path":    pt_path,
        })

    return models


# ---------------------------------------------------------------------------
# Resolve checkpoint path
# ---------------------------------------------------------------------------

def resolve_checkpoint(model_entry: dict, cfg: dict) -> Path:
    ckpt_type = model_entry["type"]

    if ckpt_type == "pth":
        # Backbone .pth file directly from lightly_cache
        ckpt_path = Path(model_entry["pt_path"])
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Backbone .pth not found: {ckpt_path}\n"
                "Please add the file to lightly_cache manually."
            )
        logger.info(f"Using backbone .pth: {ckpt_path}")
        return ckpt_path

    if ckpt_type == "custom":
        pt_path = model_entry.get("pt_path")
        if not pt_path:
            raise ValueError(
                f"pt_path must be set for custom model '{model_entry['model_name']}'"
            )
        ckpt_path = Path(pt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Custom checkpoint not found: {ckpt_path}")
        if ckpt_path.suffix not in (".pt", ".ckpt"):
            raise ValueError(
                f"Unsupported format '{ckpt_path.suffix}' — must be .pt or .ckpt"
            )
        logger.info(f"Using custom checkpoint ({ckpt_path.suffix}): {ckpt_path}")
        return ckpt_path

    # type == "lightly_train"
    from lightly_train._task_models import task_model_helpers

    base_model = model_entry["base_model"]
    cache_dir  = Path(cfg["cache"]["lightly_train_model_cache_dir"])
    logger.info(f"Cache dir  : {cache_dir}")

    existing = list(cache_dir.glob("*.pt"))
    if existing:
        logger.info(f"Found {len(existing)} cached .pt file(s)")
        for f in existing:
            logger.info(f"  {f.name}")
    else:
        logger.info(f"No cached model found — downloading: {base_model}")

    ckpt_path = task_model_helpers.download_checkpoint(checkpoint=base_model)
    logger.info(f"Checkpoint ready at: {ckpt_path}")
    return Path(ckpt_path)


# ---------------------------------------------------------------------------
# Register one model
# ---------------------------------------------------------------------------

def register_one(model_entry: dict, cfg: dict) -> tuple[str | None, str | None]:
    tracking_uri     = cfg["mlflow"]["tracking_uri"]
    experiment_name  = cfg["mlflow"]["experiment_name"]
    model_name       = model_entry["model_name"]
    pip_requirements = cfg["pip_requirements"]

    checkpoint_path = resolve_checkpoint(model_entry, cfg)

    ckpt_size_mb = checkpoint_path.stat().st_size / 1024 / 1024
    logger.info(f"Checkpoint: {checkpoint_path} ({ckpt_size_mb:.1f} MB)")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = f"register_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        try:
            mlflow.log_params({
                "model_name":      model_name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_mb":   round(ckpt_size_mb, 2),
                "task":            "semantic_segmentation",
                "base_model":      model_entry["base_model"],
                "type":            model_entry["type"],
            })
            mlflow.set_tags({
                "task":      "semantic_segmentation",
                "framework": "lightly_train",
            })

            t0 = time.time()
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=LightlySegWrapper(),
                artifacts={"checkpoint": str(checkpoint_path)},
                registered_model_name=model_name,
                pip_requirements=pip_requirements,
            )
            mlflow.log_metric("registration_time_s", time.time() - t0)

            logger.info(f"✅ Registered '{model_name}' (run={run.info.run_id})")
            return model_name, run.info.run_id

        except Exception:
            logger.error(f"Registration failed for '{model_name}':")
            traceback.print_exc()
            return None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_register.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    models = build_model_list(cfg)

    if not models:
        print("No models selected — check register switches in config.")
    else:
        print(f"\nModels to register ({len(models)}):")
        for m in models:
            print(f"  [{m['type']:13}] {m['model_name']}")
        print()

    results = []
    for m in models:
        logger.info("=" * 60)
        logger.info(f"Registering: {m['model_name']}")
        logger.info("=" * 60)
        name, run_id = register_one(m, cfg)
        results.append((m["model_name"], name is not None))

    print("\n---------- SUMMARY ----------")
    for model_name, ok in results:
        print(f"  {'✅' if ok else '❌'} {model_name}")