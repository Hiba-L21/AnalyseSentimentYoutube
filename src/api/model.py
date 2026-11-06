import os
import joblib
import numpy as np

MODEL_PATH = os.path.join("models", "sentiment_model.joblib")
LABELS     = {-1: "Négatif", 0: "Neutre", 1: "Positif"}

_pipeline = None
def load_model():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline

def is_loaded() -> bool:
    return _pipeline is not None

def predict(comments: list[str]) -> list[dict]:
    model = load_model()
    labels     = model.predict(comments)
    proba      = model.predict_proba(comments)
    classes    = list(model.classes_)

    results = []
    for i, comment in enumerate(comments):
        label      = int(labels[i])
        confidence = float(np.max(proba[i]))
        results.append({
            "comment":    comment,
            "label":      label,
            "sentiment":  LABELS[label],
            "confidence": round(confidence, 4),
        })
    return results

