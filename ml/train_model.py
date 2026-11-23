"""
train_model.py
---------------
Trains the Random Forest trend-classification model on the processed
dataset using an 80:20 chronological split (so future data is never used
to predict the past), and saves the trained model + feature list to disk.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.preprocessing import PROCESSED_PATH, FEATURE_COLUMNS

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "trend_model.pkl")


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Split each stock's history 80:20 along time, then recombine."""
    train_parts, test_parts = [], []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("date")
        split_idx = int(len(group) * train_frac)
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])
    return pd.concat(train_parts), pd.concat(test_parts)


def train():
    print("=" * 70)
    print("STOCKSENSE TREND MODEL - TRAINING")
    print("=" * 70)

    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    train_df, test_df = chronological_split(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["trend"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["trend"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=15,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print("\nTraining completed successfully")
    print("Total records   :", len(df))
    print("Training records:", len(train_df))
    print("Testing records :", len(test_df))
    print(f"Train accuracy  : {train_acc:.4f}")
    print(f"Test accuracy   : {test_acc:.4f}")

    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nFeature importances:")
    for name, score in importances:
        print(f"  {name:<15} {score:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, MODEL_PATH)
    print("\nModel saved to:", MODEL_PATH)
    print("=" * 70)


if __name__ == "__main__":
    train()
