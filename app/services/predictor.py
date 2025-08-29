# app/services/predictor.py
import os
from typing import List, Tuple, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.core.config import settings
from app.utils.text import expand_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("medium")

def load_labels(path: str) -> List[str]:
    p = os.path.join(path, "labels.txt")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return ["Negative", "Neutral", "Positive"]

class Predictor:
    def __init__(self):
        self.model_dir = settings.model_dir
        self.max_len = int(settings.max_len)
        self.use_half = bool(settings.use_half) and DEVICE == "cuda"

        self.labels = load_labels(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(DEVICE).eval()

    def _enc(self, texts: List[str]) -> dict:
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

    def predict_batch(
        self, texts: List[str], return_probs: bool = True
    ) -> Tuple[List[str], List[str], Optional[List[Dict[str, float]]]]:
        if not texts:
            return [], [], [] if return_probs else None

        expanded = [expand_text(t) for t in texts]
        batch = self._enc(expanded)
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

        with torch.no_grad():
            if self.use_half:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**batch).logits
            else:
                logits = self.model(**batch).logits

            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            labels = [self.labels[i] for i in pred_ids]

            if not return_probs:
                return expanded, labels, None

            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        probs_list: List[Dict[str, float]] = []
        for row in probs:
            probs_list.append({self.labels[i]: float(round(float(row[i]), 4))
                               for i in range(len(self.labels))})

        return expanded, labels, probs_list

predictor = Predictor()
