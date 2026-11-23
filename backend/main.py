"""
backend/main.py
----------------
FastAPI application tying the ML model and the generative-AI layer
together behind a small REST API, and serving the static dashboard.

Run with:
    uvicorn backend.main:app --reload
Then open:
    http://127.0.0.1:8000
"""

import os
from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ai.insight_generator import build_explanation, build_insight_summary, build_verdict, parse_query
from backend.schemas import AssistantQueryRequest
from ml.predictor import list_symbols, predict_trend
from ml.preprocessing import PROCESSED_PATH

app = FastAPI(
    title="StockSense API",
    description="AI-Powered Intelligent Stock Trend Prediction and Investment Insight System",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/stocks")
def get_stocks():
    """List all tracked stocks with their latest close price."""
    return list_symbols()


@app.get("/predict/{symbol}")
def get_prediction(symbol: str, horizon_days: int = 5):
    """Raw model output: trend, confidence, feature importances/values."""
    try:
        return predict_trend(symbol, horizon_days=horizon_days)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/insight/{symbol}")
def get_insight(symbol: str, horizon_days: int = 5):
    """Prediction plus a plain-language explanation, narrative summary,
    and an actionable 'is this a good time to invest' verdict."""
    try:
        prediction = predict_trend(symbol, horizon_days=horizon_days)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "prediction": prediction,
        "explanation": build_explanation(prediction),
        "summary": build_insight_summary(prediction),
        "verdict": build_verdict(prediction),
    }


@app.get("/history/{symbol}")
def get_history(
    symbol: str,
    start: date | None = Query(None, description="YYYY-MM-DD, inclusive"),
    end: date | None = Query(None, description="YYYY-MM-DD, inclusive"),
):
    """
    Historical OHLCV + indicator series for charting, optionally filtered
    to a date range. Powers the dashboard's price/volume charts and the
    'view stocks within a certain time period' feature.
    """
    if not os.path.exists(PROCESSED_PATH):
        raise HTTPException(status_code=503, detail="Processed dataset not found. Run the ML pipeline first.")

    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    rows = df[df["symbol"].str.upper() == symbol.upper()].sort_values("date")

    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not found")

    if start is not None:
        rows = rows[rows["date"] >= pd.Timestamp(start)]
    if end is not None:
        rows = rows[rows["date"] <= pd.Timestamp(end)]

    if rows.empty:
        raise HTTPException(status_code=404, detail="No data in the requested date range")

    columns = ["date", "open", "high", "low", "close", "volume", "sma5", "sma10", "rsi14", "trend"]
    rows = rows[columns].copy()
    rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")

    full_range = df[df["symbol"].str.upper() == symbol.upper()]["date"]

    return {
        "symbol": symbol.upper(),
        "available_start": full_range.min().strftime("%Y-%m-%d"),
        "available_end": full_range.max().strftime("%Y-%m-%d"),
        "points": rows.to_dict(orient="records"),
    }


@app.post("/assistant/ask")
def ask_assistant(req: AssistantQueryRequest):
    """Conversational endpoint: free-text query in, natural-language reply out."""
    parsed = parse_query(req.message)
    try:
        prediction = predict_trend(parsed.symbol, horizon_days=parsed.horizon_days)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    reply = build_explanation(prediction, intent=parsed.intent)
    verdict = build_verdict(prediction)

    return {
        "symbol": parsed.symbol,
        "intent": parsed.intent,
        "horizon_days": parsed.horizon_days,
        "prediction": prediction,
        "reply": reply,
        "verdict": verdict,
    }


# Serve the static dashboard at "/"
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
