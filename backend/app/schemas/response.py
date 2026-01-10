from typing import Optional
from pydantic import BaseModel,Field


class PredictionResponse(BaseModel):
    crop: str = Field(
        ...,
        description="Crop selected for disease prediction",
        examples=["tomato"],
    )

    disease: str = Field(
        ...,
        description="Predicted disease class",
        examples=["Tomato___Leaf_Mold"],
    )

    confidence: float = Field(
        ...,
        description="Prediction confidence score",
        examples=[0.92],
    )


class LowConfidenceResponse(BaseModel):
    message: str = Field(
        ...,
        examples=["Image unclear or may not belong to selected crop"],
    )
    confidence: float = Field(
        ...,
        examples=[0.42],
    )


class ErrorResponse(BaseModel):
    detail: str = Field(
        ...,
        examples=["Invalid crop selected"],
    )
