import json
import logging
from pathlib import Path
from typing import Dict

import requests
import tensorflow as tf

from app.core.config import MODEL_DIR, METADATA_PATH, HF_BASE_URL
from app.schemas.metadata import MetadataSchema

# Logggin
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# helper function to download model from hugginface
def download_file(filename: str) -> Path:
    target_path = MODEL_DIR / filename

    # use cached file if already downloaded
    if target_path.exists():
        logger.info(f"Using cached file: {filename}")
        return target_path

    url = f"{HF_BASE_URL}/{filename}"
    logger.info(f"Downloading {filename} from Hugging Face")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return target_path


# Path validation
if not MODEL_DIR.exists() or not MODEL_DIR.is_dir():
    raise FileNotFoundError(
        f"saved_models directory not found at: {MODEL_DIR}"
    )


# validating metadata.json
try:
    # download metadata.json from Hugging Face
    metadata_path = download_file("metadata.json")

    with open(metadata_path, "r") as f:
        raw_metadata = json.load(f)

    metadata = MetadataSchema.model_validate(raw_metadata)

except Exception:
    logger.exception("Failed to load or validate metadata.json")
    raise


# loading model
MODELS: Dict[str, Dict] = {}

# explicit crop → model filename mapping
MODEL_FILES = {
    "apple": "best_apple_model.h5",
    "grapes": "best_grapes_model.h5",
    "tomato": "best_tomato_cnn_model.h5",
    "corn": "corn_best_model.h5",
    "potato": "potato_best_model.h5",
}

for crop, classes in metadata.root.items():
    try:
        if crop not in MODEL_FILES:
            raise FileNotFoundError(
                f"No model mapping found for crop '{crop}'"
            )

        model_filename = MODEL_FILES[crop]

        # download model file from Hugging Face
        model_path: Path = download_file(model_filename)

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
