from pathlib import Path

BASE_DIR= Path(__file__).resolve().parent.parent.parent

MODEL_DIR= BASE_DIR / "saved_models"
METADATA_PATH= BASE_DIR / "metadata.json"

CONFIDENCE_THRESHOLD= 0.6
