import pandas as pd

df = pd.read_csv("data/processed/reddit_clean.csv")

# distribution des classe
print("Distribution des classes :\n", df["label"].value_counts())

# longueur des textes
df["longueur"] = df["text"].str.split().str.len()
print("Longueur des textes (exemples):\n", df["longueur"].head())
print("Longueur des textes (infos sur tout le dataset):\n", df["longueur"].describe())