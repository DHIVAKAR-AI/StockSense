"""
evaluate.py
-----------
Loads the trained model and reports classification metrics (accuracy,
precision, recall, F1-score, confusion matrix) on the held-out test split,
and saves a few visualizations (confusion matrix heatmap, feature
importance bar chart, class distribution) to model/*.png.
"""

import json
import os
import joblib
import matplotlib
matplotlib.use("Agg")  # no display available / needed — just save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ml.preprocessing import PROCESSED_PATH, FEATURE_COLUMNS
from ml.train_model import MODEL_PATH, chronological_split

LABELS = ["Bearish", "Neutral", "Bullish"]
LABEL_COLORS = {"Bearish": "#ff5c6a", "Neutral": "#f2b544", "Bullish": "#35d68f"}
METRICS_PATH = os.path.join("model", "metrics.json")
CONFUSION_PLOT_PATH = os.path.join("model", "confusion_matrix.png")
IMPORTANCE_PLOT_PATH = os.path.join("model", "feature_importance.png")
DISTRIBUTION_PLOT_PATH = os.path.join("model", "trend_distribution.png")

plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#0d1117",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#e6edf3",
        "text.color": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "font.size": 10,
    }
)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str]):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Trend Classification", pad=12)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i, j]
            color = "white" if value > cm.max() / 2 else "#0d1117"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_PLOT_PATH, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, features: list[str]):
    importances = sorted(zip(features, model.feature_importances_), key=lambda x: x[1])
    names = [n.replace("_", " ") for n, _ in importances]
    values = [v for _, v in importances]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(names, values, color="#4fa3ff")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance — Random Forest", pad=12)
    fig.tight_layout()
    fig.savefig(IMPORTANCE_PLOT_PATH, dpi=150)
    plt.close(fig)


def plot_trend_distribution(df: pd.DataFrame):
    counts = df["trend"].value_counts().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(counts.index, counts.values, color=[LABEL_COLORS[l] for l in counts.index])
    ax.set_ylabel("Number of trading days")
    ax.set_title("Trend Label Distribution — Full Dataset", pad=12)
    fig.tight_layout()
    fig.savefig(DISTRIBUTION_PLOT_PATH, dpi=150)
    plt.close(fig)


def evaluate():
    print("=" * 70)
    print("STOCKSENSE TREND MODEL - EVALUATION")
    print("=" * 70)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    _, test_df = chronological_split(df)

    X_test, y_true = test_df[FEATURE_COLUMNS], test_df["trend"]
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print(f"\nOverall Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix (rows=actual, cols=predicted) — order:", LABELS)
    print(cm)

    os.makedirs("model", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump({"accuracy": acc, "report": report, "labels": LABELS, "confusion_matrix": cm.tolist()}, f, indent=2)

    print("\nMetrics saved to:", METRICS_PATH)

    plot_confusion_matrix(cm, LABELS)
    plot_feature_importance(model, FEATURE_COLUMNS)
    plot_trend_distribution(df)

    print("Visualizations saved to:")
    print(" ", CONFUSION_PLOT_PATH)
    print(" ", IMPORTANCE_PLOT_PATH)
    print(" ", DISTRIBUTION_PLOT_PATH)

    print("\nEvaluation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
