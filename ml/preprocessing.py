"""
preprocessing.py
-----------------
Loads the raw OHLCV dataset, cleans it, engineers technical-indicator
features, creates the Bullish/Bearish/Neutral trend label, and saves the
processed dataset used by every later stage (training, evaluation,
prediction, and the generative-AI layer).
"""

import os
import numpy as np
import pandas as pd

RAW_PATH = os.path.join("data", "stock_data.csv")
PROCESSED_PATH = os.path.join("data", "processed_stock_data.csv")

FEATURE_COLUMNS = ["sma5", "sma10", "rsi14", "daily_return", "volatility10"]


def add_indicators(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()

    group["daily_return"] = group["close"].pct_change()
    group["sma5"] = group["close"].rolling(5).mean()
    group["sma10"] = group["close"].rolling(10).mean()

    delta = group["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    group["rsi14"] = 100 - (100 / (1 + rs))
    group["rsi14"] = group["rsi14"].fillna(50)  # neutral RSI while warming up

    group["volatility10"] = group["daily_return"].rolling(10).std()
    group["avg_volatility"] = group["daily_return"].expanding().std()

    # forward-looking label input: tomorrow's return
    group["next_return"] = group["close"].pct_change().shift(-1)
    return group


def label_trend(next_return: float, up_th: float = 0.01, down_th: float = -0.01) -> str:
    if pd.isna(next_return):
        return np.nan
    if next_return > up_th:
        return "Bullish"
    if next_return < down_th:
        return "Bearish"
    return "Neutral"


def run():
    print("=" * 60)
    print("STOCKSENSE STOCK DATA PREPROCESSING")
    print("=" * 60)

    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    print("\nOriginal Dataset Shape:")
    print(df.shape)
    print("\nMissing Values Before Preprocessing:")
    print(df.isnull().sum())

    before = len(df)
    df = df.drop_duplicates(subset=["symbol", "date"])
    after = len(df)
    print("\nDuplicate Records Removed:", before - after)

    df = df.sort_values(["symbol", "date"])
    price_cols = ["open", "high", "low", "close", "volume"]
    df[price_cols] = df.groupby("symbol")[price_cols].ffill()

    # NOTE: iterate manually (rather than groupby(...).apply(...)) so the
    # "symbol" column is preserved in every group regardless of pandas
    # version (recent pandas excludes the grouping column inside .apply).
    df = pd.concat(
        [add_indicators(group) for _, group in df.groupby("symbol")],
        ignore_index=True,
    )

    df["trend"] = df["next_return"].apply(label_trend)

    required = FEATURE_COLUMNS + ["next_return", "trend"]
    df = df.dropna(subset=required)

    print("\nProcessed Dataset Shape:")
    print(df.shape)
    print("\nMissing Values After Preprocessing:")
    print(df[required].isnull().sum())
    print("\nTrend Label Distribution:")
    print(df["trend"].value_counts())

    os.makedirs("data", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("\nProcessed dataset saved successfully!")
    print("File:", PROCESSED_PATH)
    print("=" * 60)


if __name__ == "__main__":
    run()
