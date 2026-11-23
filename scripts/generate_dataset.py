"""
generate_dataset.py
--------------------
Creates a synthetic (but realistic-looking) daily OHLCV dataset for a small
basket of NSE-style stocks and saves it to data/stock_data.csv.

This is a stand-in for a real historical data feed (e.g. Yahoo Finance /
NSE bhavcopy). Swap this script for a real data-fetching script when you
have API access; every downstream script only depends on the column
layout, not on where the data came from.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

STOCKS = [
    {"symbol": "RELIANCE", "company": "Reliance Industries", "sector": "Energy", "start_price": 2700},
    {"symbol": "TCS", "company": "Tata Consultancy Services", "sector": "IT", "start_price": 3800},
    {"symbol": "INFY", "company": "Infosys", "sector": "IT", "start_price": 1700},
    {"symbol": "HDFCBANK", "company": "HDFC Bank", "sector": "Banking", "start_price": 1600},
    {"symbol": "ICICIBANK", "company": "ICICI Bank", "sector": "Banking", "start_price": 1150},
    {"symbol": "SBIN", "company": "State Bank of India", "sector": "Banking", "start_price": 780},
    {"symbol": "ITC", "company": "ITC Limited", "sector": "FMCG", "start_price": 430},
    {"symbol": "WIPRO", "company": "Wipro", "sector": "IT", "start_price": 290},
    {"symbol": "BHARTIARTL", "company": "Bharti Airtel", "sector": "Telecom", "start_price": 1450},
    {"symbol": "MARUTI", "company": "Maruti Suzuki", "sector": "Automobile", "start_price": 11800},
]

N_DAYS = 300  # ~ one trading year


def generate_price_path(start_price: float, n_days: int, drift: float, vol: float) -> np.ndarray:
    """
    Price path with a slow-moving, mean-reverting 'momentum regime' layered
    on top of a random walk. A pure random walk has no learnable structure
    (tomorrow's move is independent of today's), so a classifier can never
    beat chance on it. Real markets do show short-term momentum/mean
    reversion, which is exactly what indicators like RSI and moving
    averages are designed to pick up — so we simulate a light version of
    that here via an AR(1) regime process, giving the model genuine (if
    modest) signal to learn from.
    """
    regime = 0.0
    regime_persistence = 0.85       # how "sticky" the current momentum is
    regime_vol = vol * 0.22         # how much the regime itself drifts

    daily_log_returns = np.zeros(n_days)
    for t in range(n_days):
        regime = regime_persistence * regime + np.random.normal(0, regime_vol)
        noise = np.random.normal(0, vol)
        daily_log_returns[t] = drift + regime + noise

    log_prices = np.log(start_price) + np.cumsum(daily_log_returns)
    prices = np.exp(log_prices)

    # safety net: keep the year's move within a realistic band even on an
    # unlucky random draw, so no stock accidentally 100x's or goes to zero
    prices = np.clip(prices, start_price * 0.25, start_price * 4.0)
    return prices


def build_ohlcv(symbol_info: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    drift = np.random.uniform(-0.0004, 0.0009)   # slight per-stock bias
    vol = np.random.uniform(0.012, 0.028)        # per-stock volatility

    close = generate_price_path(symbol_info["start_price"], len(dates), drift, vol)

    # derive open/high/low around close with small intraday noise
    intraday_noise = np.random.normal(0, vol * 0.6, size=len(dates))
    open_ = close * (1 + intraday_noise)
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, vol * 0.4, size=len(dates))))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, vol * 0.4, size=len(dates))))

    base_volume = np.random.uniform(2_00_000, 9_00_000)
    volume = np.random.normal(base_volume, base_volume * 0.25, size=len(dates)).clip(min=10_000)

    df = pd.DataFrame(
        {
            "symbol": symbol_info["symbol"],
            "company": symbol_info["company"],
            "sector": symbol_info["sector"],
            "date": dates,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": volume.round(0).astype(int),
        }
    )
    return df


def main():
    end_date = pd.Timestamp.today().normalize()
    all_dates = pd.bdate_range(end=end_date, periods=N_DAYS)  # business days only

    frames = [build_ohlcv(s, all_dates) for s in STOCKS]
    df = pd.concat(frames, ignore_index=True)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "stock_data.csv")
    df.to_csv(out_path, index=False)

    print("=" * 60)
    print("STOCKSENSE SYNTHETIC DATASET GENERATION")
    print("=" * 60)
    print(f"Stocks generated : {len(STOCKS)}")
    print(f"Trading days     : {N_DAYS}")
    print(f"Total rows       : {len(df)}")
    print(f"Saved to         : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
