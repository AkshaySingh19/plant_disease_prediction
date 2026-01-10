import logging
import numpy as np

from app.model_loading.model_loader import MODELS

logger = logging.getLogger(__name__)


def predict(crop: str, image_array: np.ndarray) -> dict:
    """
    Run inference for a given crop and preprocessed image.
    """

    entry = MODELS[crop]
    model = entry["model"]
    classes = entry["classes"]

    # model prediction
    preds = model.predict(image_array)

    # safety check
    if preds.ndim != 2 or preds.shape[0] != 1:
        raise ValueError("Invalid prediction output shape from model")

    confidence = float(np.max(preds))
    class_index = int(np.argmax(preds))

    # label safety
    if class_index >= len(classes):
        raise IndexError("Predicted class index out of range")

    result = {
        "crop": crop,
        "disease": classes[class_index],
        "confidence": round(confidence, 4)
    }

    logger.info(
        f"Prediction done | crop={crop} | "
        f"disease={result['disease']} | "
        f"confidence={result['confidence']}"
    )

    return result
