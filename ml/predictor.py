"""
predictor.py
------------
Loads the trained model once, and exposes predict_trend(symbol) which
returns the predicted trend, confidence, feature importances, and the
raw feature values used — everything the generative-AI layer needs to
build an explanation.
"""

import os
import joblib
import pandas as pd

from ml.preprocessing import PROCESSED_PATH, FEATURE_COLUMNS
from ml.train_model import MODEL_PATH

_bundle = None
_data = None


def _load():
    global _bundle, _data
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Trained model not found. Run `python -m ml.train_model` first."
            )
        _bundle = joblib.load(MODEL_PATH)
    if _data is None:
        if not os.path.exists(PROCESSED_PATH):
            raise FileNotFoundError(
                "Processed dataset not found. Run `python -m ml.preprocessing` first."
            )
        _data = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    return _bundle, _data


def list_symbols() -> list[dict]:
    _, data = _load()
    latest = data.sort_values("date").groupby("symbol").tail(1)
    return latest[["symbol", "company", "sector", "close", "date"]].to_dict(orient="records")


def get_latest_row(symbol: str) -> pd.Series:
    _, data = _load()
    rows = data[data["symbol"].str.upper() == symbol.upper()].sort_values("date")
    if rows.empty:
        raise KeyError(f"Stock '{symbol}' not found")
    return rows.iloc[-1]


def predict_trend(symbol: str, horizon_days: int = 5) -> dict:
    """
    Predicts the trend for `symbol` using its most recent feature row.
    horizon_days is informational only in this simple offline model — a
    production version could retrain per horizon or use a sequence model.
    """
    bundle, _ = _load()
    model = bundle["model"]
    features = bundle["features"]

    row = get_latest_row(symbol)
    X = row[features].to_frame().T

    probs = model.predict_proba(X)[0]
    classes = list(model.classes_)
    pred_idx = probs.argmax()

    importances = dict(zip(features, model.feature_importances_.tolist()))

    return {
        "symbol": row["symbol"],
        "company": row["company"],
        "sector": row["sector"],
        "as_of_date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
        "horizon_days": horizon_days,
        "trend": classes[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "class_probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "feature_importances": importances,
        "feature_values": {f: float(row[f]) for f in features},
        "avg_volatility": float(row.get("avg_volatility", row["volatility10"])),
    }


if __name__ == "__main__":
    import json

    for s in ["RELIANCE", "TCS", "HDFCBANK"]:
        print(json.dumps(predict_trend(s), indent=2))
