"""
MLflow-compatible SAM3 model wrapper for inference serving
Uses huggingface_hub.login() for authentication (same as your working code)
"""

import base64
import logging
import os
from io import BytesIO

import mlflow
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("seg.sam3_mlflow")


class SAM3MLflowModel(mlflow.pyfunc.PythonModel):
    """MLflow-compatible SAM3 model wrapper"""

    def load_context(self, context):
        """Load model on server startup"""
        # Skip loading during registration/serialization
        if os.getenv("MLFLOW_DISABLE_MODEL_LOADING", "false").lower() == "true":
            logger.info("Skipping model loading (registration mode)")
            return

        # Authenticate with Hugging Face FIRST (like your working code)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import login

                login(token=hf_token)
                logger.info("Successfully authenticated with Hugging Face")
            except Exception as e:
                logger.warning(f"Failed to authenticate with Hugging Face: {e}")
        else:
            logger.warning("HF_TOKEN not found - may fail for gated models")

        # Now import and load (like your working code)
        from transformers import Sam3Model, Sam3Processor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = os.getenv("SAM_MODEL_NAME", "facebook/sam3")

        logger.info(f"Loading SAM3 model '{model_name}' on {device}...")

        # Don't pass token here - we already logged in above
        self.model = Sam3Model.from_pretrained(model_name).to(device)
        self.processor = Sam3Processor.from_pretrained(model_name)
        self.device = device
        logger.info("SAM3 model loaded successfully")

    def predict(self, context, model_input):
        """
        Inference endpoint

        Input format:
        {
            "image": base64_encoded_image_string,
            "boxes": [[x1, y1, x2, y2], ...],
            "threshold": 0.5,
            "mask_threshold": 0.5
        }

        Returns:
        {
            "masks": [base64_encoded_mask_1, ...],
            "scores": [score_1, ...],
            "num_masks": int,
            "success": bool,
            "error": str (if failed)
        }
        """
        try:
            # Parse input
            if isinstance(model_input, dict):
                input_data = model_input
            else:
                # Handle pandas DataFrame input from MLflow
                input_data = model_input.to_dict("records")[0]

            # Decode image
            image_bytes = base64.b64decode(input_data["image"])
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            boxes = input_data["boxes"]
            threshold = input_data.get("threshold", 0.5)
            mask_threshold = input_data.get("mask_threshold", 0.5)

            logger.info(f"Processing {len(boxes)} boxes on image {image.size}")

            # Prepare inputs
            box_labels = [1] * len(boxes)
            inputs = self.processor(
                images=image,
                input_boxes=[boxes],
                input_boxes_labels=[box_labels],
                return_tensors="pt",
            ).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            # Encode masks as base64 for transmission
            encoded_masks = []
            scores = []

            if "masks" in results:
                for mask in results["masks"]:
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                    mask_img = Image.fromarray(mask_np, mode="L")
                    buffer = BytesIO()
                    mask_img.save(buffer, format="PNG")
                    encoded_mask = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    encoded_masks.append(encoded_mask)

                if "scores" in results:
                    scores = results["scores"].cpu().tolist()

            return {
                "masks": encoded_masks,
                "scores": scores,
                "num_masks": len(encoded_masks),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {
                "masks": [],
                "scores": [],
                "num_masks": 0,
                "success": False,
                "error": str(e),
            }
