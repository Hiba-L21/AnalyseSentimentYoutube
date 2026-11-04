import os
import urllib.request
import pandas as pd

URL = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
RAW_PATH = os.path.join("data", "raw", "reddit.csv")

LABELS = {-1: "Négatif", 0: "Neutre", 1: "Positif"}


def download():
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

    print(f"Téléchargement depuis :\n  {URL}\n")
    urllib.request.urlretrieve(URL, RAW_PATH)
    print(f"Fichier sauvegardé → {RAW_PATH}\n")

def rename_columns():
    df = pd.read_csv(RAW_PATH)
    df = df.rename(columns={"clean_comment": "text", "category": "label"})
    df.to_csv(RAW_PATH, index=False)
    print("Colonnes renommées → 'text' et 'label'\n")

def show_stats():
    df = pd.read_csv(RAW_PATH)

    print("=" * 45)
    print("  STATISTIQUES DU DATASET")
    print("=" * 45)
    print(f"  Nombre total de commentaires : {len(df)}")
    print(f"  Colonnes                     : {list(df.columns)}")
    print(f"  Valeurs manquantes           : {df.isnull().sum().sum()}")
    print()

    print("  Distribution des labels :")
    counts = df["label"].value_counts().sort_index()
    for val, count in counts.items():
        label_name = LABELS.get(int(val), str(val))
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label_name:>8} ({val:+d}) : {count:>5} ({pct:5.1f}%)  {bar}")

    print("=" * 45)
    print()
    print("  Aperçu des 3 premiers commentaires :")
    for i, row in df.head(3).iterrows():
        sentiment = LABELS.get(int(row["label"]), "?")
        print(f"  [{sentiment}] {str(row['text'])[:80]}...")
    print("=" * 45)


if __name__ == "__main__":
    download()
    rename_columns()
    show_stats()