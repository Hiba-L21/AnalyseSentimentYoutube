import os
import re
import pandas as pd
RAW_PATH       = os.path.join("data", "raw", "reddit.csv")
PROCESSED_PATH = os.path.join("data", "processed", "reddit_clean.csv")

def remove_urls(text: str) -> str:
    """Supprime les URLs (http, https, www)"""
    return re.sub(r"http\S+|www\.\S+", "", text)

def remove_mentions(text: str) -> str:
    """Supprime les mentions @utilisateur"""
    return re.sub(r"@\w+", "", text)

def remove_hashtags(text: str) -> str:
    """Supprime les hashtags #mot"""
    return re.sub(r"#\w+", "", text)

def remove_special_chars(text: str) -> str:
    """Supprime les caractères spéciaux — garde lettres, chiffres, espaces"""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)

def normalize_whitespace(text: str) -> str:
    """Remplace les espaces multiples par un seul espace"""
    return re.sub(r"\s+", " ", text).strip()

def to_lowercase(text: str) -> str:
    return text.lower()

def clean_text(text: str) -> str:
    """Pipeline complet de nettoyage appliqué dans l'ordre"""
    if not isinstance(text, str):
        return ""
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)
    return text

def preprocess():
    df = pd.read_csv(RAW_PATH)
    # Suppression des valeurs manquantes
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    print(f"Valeurs manquantes supprimées : {before - len(df)} lignes")
    # Nettoyage du texte
    print("Nettoyage du texte en cours...")
    df["text"] = df["text"].apply(clean_text)
    # Suppression des textes vides après nettoyage
    before = len(df)
    df = df[df["text"].str.len() > 0]
    print(f"Textes vides supprimés        : {before - len(df)} lignes")
    # Conversion du label en entier
    df["label"] = df["label"].astype(int)

    # Sauvegarde
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

if __name__ == "__main__":
    preprocess()