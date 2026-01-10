import json
import tensorflow as tf
from app.core.config import MODEL_DIR, METADATA_PATH

# Load metadata.json from backend root
with open(METADATA_PATH, "r") as f:
    CLASS_METADATA = json.load(f)

MODELS = {}

for crop, classes in CLASS_METADATA.items():
    # load corresponding model
    model_path = next(MODEL_DIR.glob(f"*{crop}*.h5"))
    model = tf.keras.models.load_model(model_path)

    MODELS[crop] = {
        "model": model,
        "classes": classes
    }
