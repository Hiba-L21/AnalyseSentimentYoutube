import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import time

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH  = os.path.join("data", "processed", "test.csv")

TFIDF = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                        min_df=2, sublinear_tf=True)

MODELS = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "LinearSVC":           LinearSVC(class_weight="balanced", max_iter=2000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced",random_state=42, n_jobs=-1),
    "Naive Bayes":         MultinomialNB(),
}

def compare():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    X_train, y_train = train["text"], train["label"]
    X_test,  y_test  = test["text"],  test["label"]

    print(f"\n{'─'*55}")
    print(f"  {'Modèle':<22} {'F1-macro':>9} {'Temps':>8}")
    print(f"{'─'*55}")

    for name, clf in MODELS.items():
        pipeline = Pipeline([("tfidf", TFIDF), ("clf", clf)])
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        elapsed = time.time() - t0
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"  {name:<22} {f1:>9.4f} {elapsed:>7.1f}s")

    print(f"{'─'*55}\n")


if __name__ == "__main__":
    compare()