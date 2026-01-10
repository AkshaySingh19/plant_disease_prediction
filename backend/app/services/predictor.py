import logging
import numpy as np
import tensorflow as tf

from app.model_loading.model_loader import get_model

logger = logging.getLogger(__name__)


def predict(crop: str, image_array: np.ndarray) -> dict:
    """
    Run TFLite inference for a given crop and preprocessed image.
    image_array shape: (1, H, W, C)
    """

    # fetch model on-demand
    model_data = get_model(crop)

    interpreter: tf.lite.Interpreter = model_data["interpreter"]
    classes = model_data["classes"]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # dtype safety (VERY IMPORTANT)
    image_array = image_array.astype(input_details[0]["dtype"])

    # set input tensor
    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    # run inference
    interpreter.invoke()

    # get output
    preds = interpreter.get_tensor(
        output_details[0]["index"]
    )

    # safety checks
    if preds.ndim != 2 or preds.shape[0] != 1:
        raise ValueError(
            f"Invalid prediction output shape: {preds.shape}"
        )

    confidence = float(np.max(preds))
    class_index = int(np.argmax(preds))

    if class_index >= len(classes):
        raise IndexError(
            "Predicted class index out of range"
        )

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

