import json
import logging
from pathlib import Path
from typing import Dict

import tensorflow as tf

from app.core.config import MODEL_DIR, METADATA_PATH
from app.schemas.metadata import MetadataSchema

# Logggin
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Path validation
if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"metadata.json not found at: {METADATA_PATH}"
    )

if not MODEL_DIR.exists() or not MODEL_DIR.is_dir():
    raise FileNotFoundError(
        f"saved_models directory not found at: {MODEL_DIR}"
    )


# validating metadata.json
try:
    with open(METADATA_PATH, "r") as f:
        raw_metadata = json.load(f)

    metadata = MetadataSchema.model_validate(raw_metadata)

except Exception:
    logger.exception("Failed to load or validate metadata.json")
    raise


# loading model
MODELS: Dict[str, Dict] = {}

for crop, classes in metadata.root.items():
    try:
        # Find model file for this crop
        model_files = list(MODEL_DIR.glob(f"*{crop}*.h5"))

        if not model_files:
            raise FileNotFoundError(
                f"No .h5 model found for crop '{crop}' in {MODEL_DIR}"
            )

        if len(model_files) > 1:
            logger.warning(
                f"Multiple model files found for '{crop}'. "
                f"Using: {model_files[0].name}"
            )

        model_path: Path = model_files[0]

        logger.info(f"Loading model for crop '{crop}' from {model_path.name}")

        model = tf.keras.models.load_model(model_path)

        MODELS[crop] = {
            "model": model,
            "classes": classes
        }

        logger.info(
            f"Model loaded for '{crop}' | "
            f"Classes: {len(classes)}"
        )

    except Exception:
        logger.exception(f"Failed to load model for crop '{crop}'")
        raise


# if model not found
if not MODELS:
    raise RuntimeError("No models were loaded. Backend cannot start.")

logger.info(f"✅ Total models loaded successfully: {len(MODELS)}")
