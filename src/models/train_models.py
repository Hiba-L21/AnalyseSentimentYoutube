import os, time, joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

TRAIN_PATH  = os.path.join("data", "processed", "train.csv")
TEST_PATH   = os.path.join("data", "processed", "test.csv")
MODEL_PATH  = os.path.join("models", "sentiment_model.joblib")
LABELS      = {-1: "Negatif", 0: "Neutre", 1: "Positif"}
LABEL_NAMES = [LABELS[i] for i in sorted(LABELS)]

def train():
    tr = pd.read_csv(TRAIN_PATH)
    te = pd.read_csv(TEST_PATH)
    X_train, y_train = tr["text"], tr["label"]
    X_test,  y_test  = te["text"], te["label"]
    print(f"Train : {len(X_train):,}  |  Test : {len(X_test):,}\n")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf",   LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    param_dist = {
        "tfidf__max_features": [30_000, 50_000, 80_000],
        "tfidf__ngram_range":  [(1, 1), (1, 2)],
        "tfidf__min_df":       [1, 2, 3],
        "clf__C":              [0.1, 0.5, 1.0, 5.0, 10.0],
        "clf__solver":         ["lbfgs", "saga"],
    }
    search = RandomizedSearchCV(pipeline, param_dist, n_iter=20, cv=3,
                                scoring="f1_macro", n_jobs=-1,
                                random_state=42, verbose=1)
    search.fit(X_train, y_train)
    model = search.best_estimator_

    print(f"\nMeilleurs parametres :")
    for k, v in search.best_params_.items():
        print(f"  {k:<30} : {v}")
    print(f"  F1-macro CV                    : {search.best_score_:.4f}")

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_cls = f1_score(y_test, y_pred, average=None, labels=sorted(LABELS))

    print(f"\n{'='*50}")
    print(f"  Accuracy : {acc:.4f}  ({'OK' if acc >= 0.80 else 'FAIL'} seuil 80%)")
    print(f"  F1-macro : {f1_mac:.4f}")
    print(f"\n  F1 par classe :")
    for i, val in enumerate(sorted(LABELS)):
        ok = "OK" if f1_cls[i] >= 0.75 else "FAIL"
        print(f"    {LABELS[val]:>8} ({val:+d}) : {f1_cls[i]:.4f}  [{ok}]")

    print(f"\n{classification_report(y_test, y_pred, target_names=LABEL_NAMES)}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=sorted(LABELS))
    print("  Matrice de confusion :")
    print(f"{'':>12}" + "".join(f"{n:>10}" for n in LABEL_NAMES))
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:>10}" + "".join(f"{cm[i][j]:>10}" for j in range(3)))

    # Temps d inference
    batch = X_test.iloc[:50].tolist()
    t0 = time.time()
    model.predict(batch)
    ms = (time.time() - t0) * 1000
    print(f"\n  Inference 50 commentaires : {ms:.1f}ms  [{'OK' if ms < 100 else 'FAIL'}]")
    print(f"{'='*50}")

    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n  Modele sauvegarde : {MODEL_PATH}\n")

if __name__ == "__main__":
    train()