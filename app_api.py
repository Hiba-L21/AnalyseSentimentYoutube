import os, re, joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List

MODEL_PATH = os.path.join("models", "sentiment_model.joblib")
LABELS     = {-1: "Négatif", 0: "Neutre", 1: "Positif"}

_pipeline = None

def load_model():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class PredictRequest(BaseModel):
    comments: List[str]

    @field_validator("comments")
    @classmethod
    def validate(cls, v):
        if not v:
            raise ValueError("Liste de commentaires vide.")
        if len(v) > 500:
            raise ValueError("Maximum 500 commentaires par requête.")
        return [c.strip() for c in v if c.strip()]

app = FastAPI(
    title="YouTube Sentiment API",
    description="Analyse de sentiment — Négatif / Neutre / Positif",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    load_model()
    print("Modele charge.")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}

@app.post("/predict_batch")
def predict_batch(request: PredictRequest):
    try:
        model   = load_model()
        cleaned = [clean_text(c) for c in request.comments]
        labels  = model.predict(cleaned)
        probas  = model.predict_proba(cleaned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    counts  = {"Négatif": 0, "Neutre": 0, "Positif": 0}
    for i, comment in enumerate(request.comments):
        label     = int(labels[i])
        sentiment = LABELS[label]
        conf      = round(float(np.max(probas[i])), 4)
        results.append({"comment": comment, "label": label,
                         "sentiment": sentiment, "confidence": conf})
        counts[sentiment] += 1

    total = len(results)
    stats = {k: {"count": v, "pct": round(v / total * 100, 1)}
             for k, v in counts.items()}

    return {"total": total, "results": results, "stats": stats}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app_api:app", host="0.0.0.0", port=port, reload=False)