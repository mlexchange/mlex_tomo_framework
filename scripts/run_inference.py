"""
run_inference.py
================
1. Register any missing models to MLflow (based on config_register.yaml switches)
2. Load registered models from MLflow (based on config_inference.yaml switches)
3. Connect to Tiled, randomly select one paired image + mask
4. Run inference with each enabled model
5. Save results to /results

Usage:
    docker compose run --rm lightly_infer python run_inference.py
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import mlflow
import numpy as np
from mlex_utils.mlflow_utils.mlflow_model_client import MLflowModelClient
import yaml
from dotenv import load_dotenv
from PIL import Image
from tiled.client import from_uri

load_dotenv(dotenv_path="../.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR      = Path("/results")
INFERENCE_CONFIG = Path("config_inference.yaml")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.dtype == np.uint8:
        return arr
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    palette = np.array([
        [0,   0,   0  ],
        [31,  119, 180],
        [255, 127, 14 ],
        [44,  160, 44 ],
        [214, 39,  40 ],
        [148, 103, 189],
        [140, 86,  75 ],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34 ],
    ], dtype=np.uint8)

    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id in np.unique(mask):
        if cls_id == 255:
            rgb[mask == cls_id] = [200, 200, 200]
            continue
        rgb[mask == cls_id] = palette[cls_id % len(palette)]
    return rgb


def _inf_model_names(inf_cfg: dict) -> list[str]:
    inf = inf_cfg["infer"]
    vit = inf["vit"]
    names = []
    if inf.get("eomt"):
        names.append(f"dinov3_{vit}_eomt")
    if inf.get("eomt_coco"):
        names.append(f"dinov3_{vit}_eomt_coco")
    if inf.get("eomt_cityscapes"):
        names.append(f"dinov3_{vit}_eomt_cityscapes")
    if inf.get("eomt_cityscapes_petiole"):
        names.append(f"dinov3_{vit}_eomt_cityscapes_petiole")
    for name in inf.get("finetuned") or []:
        names.append(name)
    return names


def load_model(inf_cfg: dict, model_name: str):
    tracking_uri  = os.environ["MLFLOW_TRACKING_URI_OUTSIDE"]
    model_version = inf_cfg["mlflow"]["model_version"]

    client = MLflowModelClient(
        tracking_uri=tracking_uri,
        cache_dir=os.getenv("MLFLOW_CACHE_DIR"),
    )

    # Resolve "latest" to an actual version number
    version = None if str(model_version).lower() == "latest" else str(model_version)

    logger.info(f"Loading model '{model_name}' version={version or 'latest'} ...")
    model = client.load_model(model_name, version=version)
    if model is None:
        raise ValueError(f"Failed to load model '{model_name}' from MLflow.")
    logger.info("✅ Model loaded")
    return model


def image_to_mask_key(img_key: str) -> str:
    return img_key


def load_sample(inf_cfg: dict) -> tuple[np.ndarray, np.ndarray, str]:
    api_key      = os.environ["TILED_API_KEY"]
    images_uri   = os.environ["DATA_TILED_URI_IMAGES"]
    masks_uri    = os.environ["DATA_TILED_URI_MASKS"]

    tiled_images = from_uri(images_uri, api_key=api_key)
    tiled_masks  = from_uri(masks_uri,  api_key=api_key)

    image_keys = list(tiled_images)
    mask_keys  = set(tiled_masks)

    logger.info(f"Image keys (first 3): {image_keys[:3]}")
    logger.info(f"Mask  keys (first 3): {list(mask_keys)[:3]}")

    paired_keys = [k for k in image_keys if image_to_mask_key(k) in mask_keys]

    if paired_keys:
        img_key = random.choice(paired_keys)
        msk_key = image_to_mask_key(img_key)
        logger.info(f"Paired by name — {len(paired_keys)} pairs found")
    else:
        logger.warning("Could not pair by name — falling back to index-based pairing.")
        idx = random.randint(0, min(len(image_keys), len(list(mask_keys))) - 1)
        img_key = image_keys[idx]
        msk_key = list(mask_keys)[idx]

    logger.info(f"Selected image key: {img_key}")
    logger.info(f"Selected mask  key: {msk_key}")

    raw_image = np.array(tiled_images[img_key])
    raw_mask  = np.array(tiled_masks[msk_key])

    # Minimal fix: Tiled stores containers as (n, 2560, 2560).
    # Select the same slice index for image and mask.
    slice_idx = None

    if raw_image.ndim == 3 and raw_image.shape[0] < raw_image.shape[1]:
        slice_idx = raw_image.shape[0] // 2
        logger.info(f"Image stack — using slice {slice_idx}/{raw_image.shape[0]}")
        raw_image = raw_image[slice_idx]

    if raw_mask.ndim == 3 and raw_mask.shape[0] < raw_mask.shape[1]:
        if slice_idx is None:
            slice_idx = raw_mask.shape[0] // 2
        logger.info(f"Mask stack — using slice {slice_idx}/{raw_mask.shape[0]}")
        raw_mask = raw_mask[slice_idx]

    img_uint8 = to_uint8(raw_image)
    if img_uint8.ndim == 2:
        img_uint8 = np.stack([img_uint8] * 3, axis=-1)
    elif img_uint8.shape[-1] == 1:
        img_uint8 = np.concatenate([img_uint8] * 3, axis=-1)

    gt_mask = np.squeeze(raw_mask).astype(np.int32)

    logger.info(f"Image shape : {img_uint8.shape}")
    logger.info(f"Mask shape  : {gt_mask.shape}")
    logger.info(f"Mask unique : {np.unique(gt_mask)}")

    return img_uint8, gt_mask, img_key


def run_inference(model, img_uint8: np.ndarray) -> np.ndarray:
    logger.info("Running inference ...")
    pred_mask = model.predict(img_uint8)
    logger.info(f"Predicted classes: {np.unique(pred_mask)}")
    return pred_mask


def save_results(
    img_uint8: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    img_key: str,
    inf_cfg: dict,
    model_name: str = "",
) -> None:
    tag = img_key.strip("_").replace("/", "_")
    model_tag = f"_{model_name}" if model_name else ""

    # Save input image
    Image.fromarray(img_uint8).save(
        RESULTS_DIR / f"{tag}_image.png"
    )

    # Save ground-truth mask
    Image.fromarray(mask_to_rgb(gt_mask)).save(
        RESULTS_DIR / f"{tag}_gt_mask.png"
    )

    # Save predicted mask
    Image.fromarray(mask_to_rgb(pred_mask)).save(
        RESULTS_DIR / f"{tag}{model_tag}_pred_mask.png"
    )

    logger.info(f"✅ Saved PNG results for {model_name} to {RESULTS_DIR}/")


if __name__ == "__main__":
    inf_cfg = load_yaml(INFERENCE_CONFIG)

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

    img_uint8, gt_mask, img_key = load_sample(inf_cfg)

    model_names = _inf_model_names(inf_cfg)
    if not model_names:
        logger.warning("No models selected — check infer switches in config_inference.yaml")

    for model_name in model_names:
        logger.info("=" * 60)
        logger.info(f"Inference: {model_name}")
        logger.info("=" * 60)

        try:
            model = load_model(inf_cfg, model_name)
            if model is None:
                logger.warning(
                    f"Skipping '{model_name}' — not found in MLflow registry. "
                    "Register it first (pretrained: save_mlflow_wrapper.py, "
                    "finetuned: finetune.py)."
                )
                continue
            pred_mask = run_inference(model, img_uint8)
            save_results(img_uint8, gt_mask, pred_mask, img_key, inf_cfg, model_name)

        except Exception as e:
            logger.exception(f"Inference failed for '{model_name}': {e}")