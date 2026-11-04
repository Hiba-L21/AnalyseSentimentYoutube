import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_PATH = os.path.join("data", "processed", "reddit_clean.csv")
TRAIN_PATH     = os.path.join("data", "processed", "train.csv")
TEST_PATH      = os.path.join("data", "processed", "test.csv")

def split():
    df = pd.read_csv(PROCESSED_PATH)

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]   # préserve la distribution des classes
    )

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH,  index=False)

    print(f"Train : {len(train):,}")
    print(f"Test  : {len(test):,}")
    print(f"\nDistribution dans le train :")
    print(train["label"].value_counts().sort_index())

if __name__ == "__main__":
    split()