import time
import pytest
from fastapi.testclient import TestClient
from app_api import app


@pytest.fixture(scope="module")
def client():
    """Client avec startup event déclenché → modèle chargé."""
    with TestClient(app) as c:
        yield c


# ── /health ────────────────────────────────────────────────

def test_health_status(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_health_model_loaded(client):
    res = client.get("/health")
    assert res.json()["model_loaded"] is True


# ── /predict_batch ─────────────────────────────────────────

def test_predict_batch_basique(client):
    res = client.post("/predict_batch", json={"comments": ["I love this video!", "This is terrible."]})
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2


def test_predict_batch_labels_valides(client):
    res = client.post("/predict_batch", json={"comments": ["Great content"]})
    result = res.json()["results"][0]
    assert result["label"] in [-1, 0, 1]
    assert result["sentiment"] in ["Positif", "Neutre", "Negatif"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_batch_stats_presentes(client):
    res = client.post("/predict_batch", json={"comments": ["Good", "Bad", "Neutral"]})
    stats = res.json()["stats"]
    assert any(k in stats for k in ["Positif", "Neutre", "Negatif"])


def test_predict_batch_grand(client):
    comments = [f"Comment number {i}" for i in range(100)]
    res = client.post("/predict_batch", json={"comments": comments})
    assert res.status_code == 200
    assert res.json()["total"] == 100


# ── Cas d'erreur ───────────────────────────────────────────

def test_predict_batch_vide(client):
    res = client.post("/predict_batch", json={"comments": []})
    assert res.status_code == 422


def test_predict_batch_trop_grand(client):
    comments = [f"Comment {i}" for i in range(501)]
    res = client.post("/predict_batch", json={"comments": comments})
    assert res.status_code == 422


def test_predict_batch_champ_manquant(client):
    res = client.post("/predict_batch", json={})
    assert res.status_code == 422


def test_predict_batch_mauvais_type(client):
    res = client.post("/predict_batch", json={"comments": "pas une liste"})
    assert res.status_code == 422


# ── Temps de réponse ───────────────────────────────────────

def test_temps_reponse_health(client):
    start = time.time()
    client.get("/health")
    assert (time.time() - start) < 0.2


def test_temps_reponse_predict_50(client):
    comments = [f"Comment {i}" for i in range(50)]
    start = time.time()
    client.post("/predict_batch", json={"comments": comments})
    assert (time.time() - start) < 0.5