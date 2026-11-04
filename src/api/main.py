from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictRequest, PredictResponse, HealthResponse
from src.api import model as ml


app = FastAPI(
    title="YouTube Sentiment API",
    description="Analyse de sentiment de commentaires YouTube (Négatif / Neutre / Positif)",
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
    try:
        ml.load_model()
        print("Modele charge avec succes.")
    except FileNotFoundError as e:
        print(f"AVERTISSEMENT : {e}")

@app.get("/health", response_model=HealthResponse)
def health():
    """Vérifie l'état de l'API et si le modèle est chargé."""
    return HealthResponse(
        status="ok",
        model_loaded=ml.is_loaded(),
        model_path=ml.MODEL_PATH,
    )

@app.post("/predict_batch", response_model=PredictResponse)
def predict_batch(request: PredictRequest):
    """
    Analyse un batch de commentaires.
    Retourne le sentiment, le label et la confiance pour chaque commentaire,
    ainsi que les statistiques de distribution.
    """
    try:
        results = ml.predict(request.comments)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {e}")

    counts = {"Négatif": 0, "Neutre": 0, "Positif": 0}
    for r in results:
        counts[r["sentiment"]] += 1
    total = len(results)
    stats = {k: {"count": v, "pct": round(v / total * 100, 1)}
             for k, v in counts.items()}

    return PredictResponse(total=total, results=results, stats=stats)