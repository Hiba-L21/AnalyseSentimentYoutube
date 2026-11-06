import os, time, pytest, joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "sentiment_model.joblib")
TEST_PATH  = os.path.join("data", "processed", "test.csv")


@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), f"Modèle introuvable : {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_data():
    df = pd.read_csv(TEST_PATH)
    return df["text"].tolist(), df["label"].tolist()


# ── Performance ───────────────────────────────────────────────────────────────

def test_accuracy(model, test_data):
    from sklearn.metrics import accuracy_score
    X, y = test_data
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc >= 0.80, f"Accuracy trop faible : {acc:.4f} (seuil 0.80)"


def test_f1_per_class(model, test_data):
    from sklearn.metrics import f1_score
    X, y = test_data
    preds  = model.predict(X)
    scores = f1_score(y, preds, average=None, labels=[-1, 0, 1])
    labels = ["Négatif", "Neutre", "Positif"]
    for i, (name, score) in enumerate(zip(labels, scores)):
        assert score >= 0.75, f"F1 {name} trop faible : {score:.4f} (seuil 0.75)"


# ── Temps d'inférence ─────────────────────────────────────────────────────────

def test_inference_time(model):
    batch = ["This video is amazing!"] * 50
    t0 = time.time()
    model.predict(batch)
    ms = (time.time() - t0) * 1000
    assert ms < 100, f"Inférence trop lente : {ms:.1f}ms (seuil 100ms)"


# ── Cas limites ───────────────────────────────────────────────────────────────

def test_texte_vide(model):
    result = model.predict([""])
    assert result[0] in [-1, 0, 1]


def test_texte_tres_court(model):
    result = model.predict(["ok"])
    assert result[0] in [-1, 0, 1]


def test_texte_tres_long(model):
    long_text = "great " * 300
    result = model.predict([long_text])
    assert result[0] in [-1, 0, 1]


def test_emojis(model):
    result = model.predict(["I love this so much"])
    assert result[0] in [-1, 0, 1]


def test_langues_melangees(model):
    result = model.predict(["This video is super bien vraiment good"])
    assert result[0] in [-1, 0, 1]


def test_urls_et_mentions(model):
    result = model.predict(["Check http://youtube.com @user great content #wow"])
    assert result[0] in [-1, 0, 1]


def test_predict_proba(model):
    proba = model.predict_proba(["I love this"])
    assert proba.shape[1] == 3
    assert abs(proba[0].sum() - 1.0) < 1e-6