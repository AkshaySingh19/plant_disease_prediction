import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


app=FastAPI(
    title="Crop Disease Prediction",
    description="""
    A scalable and modular backend service for multi-crop plant disease detection
    powered by deep learning.

    This backend is designed with production-grade architecture and focuses on
    clean separation of concerns, reliability, and future scalability.

    Key Features:
        - Multi-crop support using dedicated deep learning models
        - Dynamic model selection through RESTful API routes
        - Robust input validation for crops and images
        - Confidence-based safeguards to handle uncertain predictions
        - Metadata-driven class mapping to avoid hard-coded labels
        - Clean, modular FastAPI project structure

    Created by: Shikhar Srivastava
    """,
    version="1.0.0"

)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://plantdiseaseprediction.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    )


# Registering API routes
app.include_router(router)

@app.get("/")
def home():
    return {
        "message": "Crop Disease Detection Backend is running",
        "status": "OK"
    }