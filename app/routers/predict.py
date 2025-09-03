# app/routers/predict.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict
import csv
import io

import torch
from app.schemas.payloads import (
    PredictIn, PredictOut, PredictBatchIn, PredictBatchOut
)
from app.services.model_manager import model_manager
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from app.core.config import settings

router = APIRouter(prefix="/api", tags=["predict"])

@router.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@router.post("/predict", response_model=PredictOut)
def predict(req: PredictIn, model_name: str = "rnn"):
    model = model_manager.get_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    exps, label, probs = model.predict([req.text], return_probs=req.return_probs)
    return PredictOut(expanded=exps[0], label=label[0], probs=(probs[0] if probs else None))

@router.post("/predict_batch")
def predict_batch(req: PredictBatchIn, model_name: str = "lstm"):
    model = model_manager.get_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    exps, labels, probs = model.predict(req.texts, return_probs=True)

    predictions: List[Dict[str, object]] = []
    for i in range(len(labels)):
        row_prob = probs[i] if probs and i < len(probs) else None
        label = labels[i]
        score = None
        if isinstance(row_prob, dict) and row_prob:
            score = float(row_prob.get(label) or row_prob.get(str(label).lower()) or max(row_prob.values()))
        predictions.append({
            "text": exps[i] if exps and i < len(exps) else req.texts[i],
            "sentiment": label,
            "score": score
        })

    return PredictBatchOut(expanded=exps, labels=labels, probs=probs)


@router.post("/predict_file")
def predict_file(
    file: UploadFile = File(...),
    model_name: str = Form("lstm"),
    column: str = Form("text"),
    delimiter: str = Form(","),
):
    model = model_manager.get_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        raw_bytes = file.file.read()
        text = raw_bytes.decode("utf-8-sig")
    finally:
        file.file.close()

    texts: List[str] = []
    if (file.filename or "").lower().endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text), delimiter=(delimiter[:1] if delimiter else ','))
        if not reader.fieldnames or column not in reader.fieldnames:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found in CSV. Available: {reader.fieldnames}")
        for row in reader:
            value = row.get(column)
            if value is not None:
                value_str = str(value).strip()
                if value_str:
                    texts.append(value_str)
    else:
        for line in io.StringIO(text):
            ln = line.strip()
            if ln:
                texts.append(ln)

    if not texts:
        raise HTTPException(status_code=400, detail="No texts found in file")

    exps, labels, probs = model.predict(texts, return_probs=True)

    predictions: List[Dict[str, object]] = []
    for i in range(len(labels)):
        row_prob = probs[i] if probs and i < len(probs) else None
        label = labels[i]
        score = None
        if isinstance(row_prob, dict) and row_prob:
            score = float(row_prob.get(label) or row_prob.get(str(label).lower()) or max(row_prob.values()))
        predictions.append({
            "text": exps[i] if exps and i < len(exps) else texts[i],
            "sentiment": label,
            "score": score
        })

    return PredictBatchOut(expanded=exps, labels=labels, probs=probs)
