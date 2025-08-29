# app/routers/predict.py
from fastapi import APIRouter
from app.schemas.payloads import (
    PredictIn, PredictOut, PredictBatchIn, PredictBatchOut
)
from app.services.predictor import predictor, DEVICE
from app.core.config import settings

router = APIRouter(prefix="/api", tags=["predict"])

@router.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_dir": settings.model_dir}

@router.post("/predict", response_model=PredictOut)
def predict(req: PredictIn):
    exp, labels, probs = predictor.predict_batch([req.text], return_probs=req.return_probs)
    return PredictOut(expanded=exp[0], label=labels[0], probs=(probs[0] if probs else None))

@router.post("/predict_batch", response_model=PredictBatchOut)
def predict_batch(req: PredictBatchIn):
    exp, labels, probs = predictor.predict_batch(req.texts, return_probs=req.return_probs)
    return PredictBatchOut(expanded=exp, labels=labels, probs=probs)
