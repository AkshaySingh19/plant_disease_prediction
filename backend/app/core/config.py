from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

HF_BASE_URL = ("https://huggingface.co/shikhar0718/plant_disease_prediction/resolve/main")

MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = MODEL_DIR / "metadata.json"
CONFIDENCE_THRESHOLD = 0.6
