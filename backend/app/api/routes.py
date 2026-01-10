from typing import List, Union

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from app.model_loading.model_loader import MODELS
from app.services.predictor import predict
from app.utils.image_utils import preprocess_image
from app.core.config import CONFIDENCE_THRESHOLD
from app.schemas.response import (
    PredictionResponse,
    LowConfidenceResponse,
    ErrorResponse,
)

router = APIRouter()


# --------------------------------------------------
# List supported crops
# --------------------------------------------------
@router.get(
    "/crops",
    response_model=List[str],
)
def get_supported_crops():
    return list(MODELS.keys())


# --------------------------------------------------
# Predict disease for selected crop
# --------------------------------------------------
@router.post(
    "/predict/{crop}",
    response_model=Union[PredictionResponse, LowConfidenceResponse],
    responses={
        200: {
            "model": Union[PredictionResponse, LowConfidenceResponse],
            "description": "Successful prediction",
        },
        400: {"model": ErrorResponse},
    },
)
async def predict_crop(
    crop: str,
    file: UploadFile = File(...),
):
    crop = crop.lower()

    if crop not in MODELS:
        raise HTTPException(
            status_code=400,
            detail="Invalid crop selected",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image",
        )

    try:
        image = Image.open(file.file)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file",
        )

    image_array = preprocess_image(image)
    result = predict(crop, image_array)

    if result["confidence"] < CONFIDENCE_THRESHOLD:
        return LowConfidenceResponse(
            message="Image unclear or may not belong to selected crop",
            confidence=result["confidence"],
        )

    return PredictionResponse(**result)
