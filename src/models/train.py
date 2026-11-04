import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH  = os.path.join("data", "processed", "test.csv")
MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
LABELS = {-1: "Négatif", 0: "Neutre", 1: "Positif"}

def train():
    # Chargement
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    X_train, y_train = train["text"], train["label"]
    X_test,  y_test  = test["text"],  test["label"]

    # Pipeline TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,   # taille du vocabulaire
            ngram_range=(1, 2),    # unigrammes + bigrammes
            min_df=2,              # ignore les tokens rares (< 2 docs)
            sublinear_tf=True,     # log(tf) pour réduire l'effet des fréquences élevées
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    # Entraînement
    print("Entraînement en cours...")
    pipeline.fit(X_train, y_train)

    # Évaluation
    y_pred = pipeline.predict(X_test)
    print("\n" + classification_report(
        y_test, y_pred,
        target_names=[LABELS[i] for i in sorted(LABELS)]
    ))

    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modèle sauvegardé → {MODEL_PATH}")


if __name__ == "__main__":
    train()