# 🎬 YouTube Sentiment Analyzer — MLOps Pipeline

> Système MLOps complet d'analyse de sentiment des commentaires YouTube  

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange?logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![Tests](https://img.shields.io/badge/Tests-22%2F22%20passed-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-89.66%25-success)

---

## Description

Extension Chrome + API REST qui analyse en temps réel le sentiment des commentaires d'une vidéo YouTube. Le modèle classifie chaque commentaire en **Positif**, **Neutre** ou **Négatif** avec un score de confiance.

**Pipeline MLOps complet :**
```
Données Reddit → Nettoyage → EDA → Entraînement → API FastAPI → Docker → Extension Chrome
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     UTILISATEUR                             │
│                  (Extension Chrome)                         │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP POST /predict_batch
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              API REST FastAPI (Docker)                      │
│         GitHub Codespaces — Port 7860 public                │
│                                                             │
│   /health ──── Statut + modèle chargé                      │
│   /predict_batch ── Prédictions batch (max 500)            │
│   /docs ──── Documentation Swagger auto                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           Modèle ML — Pipeline scikit-learn                 │
│   TF-IDF (bigrams, 50k features) + Logistic Regression     │
│   Accuracy: 89.66% · F1-macro: 88.74%                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Structure du projet

```
AnalyseDeSentimentYoutube/
├── data/
│   ├── raw/                    # Dataset Reddit original
│   └── processed/              # Données nettoyées + splits
├── models/
│   └── sentiment_model.joblib  # Pipeline TF-IDF + LR entraîné
├── src/
│   ├── data/
│   │   ├── preprocess.py       # Nettoyage du texte
│   │   ├── eda.py              # Analyse exploratoire
│   │   └── split.py            # Split train/test (80/20)
│   ├── models/
│   │   ├── train.py            # Entraînement + RandomizedSearchCV
│   │   └── compare_models.py   # Comparaison algorithmes
│   └── api/
│       ├── main.py             # Application FastAPI
│       ├── model.py            # Singleton chargement modèle
│       └── schemas.py          # Schémas Pydantic
├── chrome-extension/
│   ├── manifest.json           # Manifest V3
│   ├── popup.html              # Interface utilisateur
│   ├── popup.js                # Logique + appel API
│   └── content.js              # Extraction commentaires YouTube
├── tests/
│   ├── test_model.py           # 10 tests unitaires modèle
│   └── test_api.py             # 12 tests API (endpoints + erreurs)
├── app_api.py                  # API standalone pour Docker
├── Dockerfile                  # Image Docker production
├── requirements_prod.txt       # Dépendances production
├── requirements.txt            # Dépendances développement
└── conftest.py                 # Configuration pytest
```

---

## Installation et utilisation

### Prérequis
- Python 3.10+
- Google Chrome
- Docker (pour le déploiement)

### 1. Cloner le repository

```bash
git clone https://github.com/<votre-username>/AnalyseDeSentimentYoutube.git
cd AnalyseDeSentimentYoutube
```

### 2. Environnement virtuel

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Pipeline ML complet

```bash
# Nettoyage des données
python -m src.data.preprocess

# Analyse exploratoire
python -m src.data.eda

# Split train/test
python -m src.data.split

# Entraînement du modèle
python -m src.models.train
# → models/sentiment_model.joblib créé
```

### 4. Lancer l'API localement

```bash
uvicorn app_api:app --host 0.0.0.0 --port 8000
# Documentation : http://localhost:8000/docs
```

### 5. Installer l'extension Chrome

```
1. Ouvrir Chrome → chrome://extensions/
2. Activer "Mode développeur"
3. Cliquer "Charger l'extension non empaquetée"
4. Sélectionner le dossier : chrome-extension/
```

---

## Déploiement Docker

```bash
# Build de l'image
docker build -t youtube-sentiment-api .

# Lancer le container
docker run -d -p 7860:7860 youtube-sentiment-api

# Vérifier
curl http://localhost:7860/health
```

**Sur GitHub Codespaces :**
1. Lancer `docker run -d -p 7860:7860 youtube-sentiment-api`
2. Onglet "Ports" → Port 7860 → Visibility = **Public**
3. Copier l'URL publique → mettre à jour `API_URL` dans `popup.js`

---

## Performances du modèle

| Métrique | Valeur |
|----------|--------|
| Accuracy | **89.66%** |
| F1-macro | **88.74%** |
| F1 Positif | 91% |
| F1 Neutre | 87% |
| F1 Négatif | 88% |
| Temps d'inférence (50 commentaires) | < 100ms |

**Dataset :** Reddit Sentiment Analysis — 36 934 commentaires après nettoyage  
**Distribution :** Positif 42.9% · Neutre 34.7% · Négatif 22.4%

**Meilleurs hyperparamètres (RandomizedSearchCV, 20 itérations, CV=3) :**
```
tfidf__max_features: 50000
tfidf__ngram_range: (1, 2)
tfidf__min_df: 2
clf__C: 5.0
clf__solver: saga
class_weight: balanced
```

---

## Tests

```bash
# Tests du modèle (accuracy, F1, temps d'inférence, cas limites)
pytest tests/test_model.py -v
# → 10/10 PASSED

# Tests de l'API (endpoints, erreurs, temps de réponse)
pytest tests/test_api.py -v
# → 12/12 PASSED
```

---

## API — Endpoints

### GET `/health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/sentiment_model.joblib"
}


### POST `/predict_batch`
**Request :**
```json
{
  "comments": ["This video is amazing!", "Not great honestly."]
}
```
**Response :**
```json
{
  "total": 2,
  "results": [
    {"comment": "This video is amazing!", "label": 1, "sentiment": "Positif", "confidence": 0.94},
    {"comment": "Not great honestly.", "label": -1, "sentiment": "Negatif", "confidence": 0.81}
  ],
  "stats": {"Positif": 1, "Neutre": 0, "Negatif": 1}
}
```

**Contraintes :** max 500 commentaires par requête · liste non vide

---

## Fonctionnalités de l'extension

- Extraction automatique des commentaires YouTube
- Analyse de sentiment en temps réel via l'API
- Visualisation : stats globales + barre de répartition
- Filtres par sentiment (Tous / Positif / Neutre / Négatif)
- Dark mode avec persistance (localStorage)
- Copie des résultats en un clic

