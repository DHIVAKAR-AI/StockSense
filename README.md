# StockSense

AI-Powered Intelligent Stock Trend Prediction and Investment Insight System.

Historical OHLCV data → technical-indicator feature engineering → a Random
Forest classifier that predicts **Bullish / Bearish / Neutral** trends →
an offline generative-AI layer that turns the model's output into
plain-language investment insights and a **clear invest / wait / avoid
verdict** → a small FastAPI backend and an interactive dashboard with
**price/volume charts and a date-range picker**.

This matches the two project reports:
- **SSA-1** — the core ML pipeline (`ml/`)
- **Active Learning-1** — the Generative AI layer (`ai/`)

> The included dataset is **synthetically generated** (a random-walk price
> simulation), so the project runs fully offline with no API keys. Swap
> `scripts/generate_dataset.py` for a real data source (Yahoo Finance, an
> NSE/BSE feed, a CSV export, etc.) when you have one — every downstream
> script only depends on the column layout (`symbol, company, sector,
> date, open, high, low, close, volume`), not on where the data came from.

## Project structure

```
stocksense/
├── scripts/
│   └── generate_dataset.py     # synthetic OHLCV data generator
├── ml/
│   ├── preprocessing.py        # cleaning + technical indicators + trend label
│   ├── train_model.py          # trains & saves the Random Forest model
│   ├── evaluate.py             # accuracy / precision / recall / F1 / confusion matrix
│   └── predictor.py            # loads the model, predicts trend for a symbol
├── ai/
│   ├── insight_generator.py    # query parsing, explanations, insight summaries
│   └── evaluate_generation.py  # keyword coverage + lexical diversity checks
├── backend/
│   ├── main.py                 # FastAPI app (serves the API + the dashboard)
│   └── schemas.py
├── frontend/
│   └── index.html              # dashboard (ticker tape, gauge, chat assistant)
├── data/                       # generated CSVs land here
├── model/                      # trained model + metrics land here
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline (in order)

```bash
# 1. Generate the synthetic historical dataset
python scripts/generate_dataset.py

# 2. Clean the data and engineer technical-indicator features + trend labels
python -m ml.preprocessing

# 3. Train the Random Forest trend classifier
python -m ml.train_model

# 4. Evaluate it (accuracy, precision, recall, F1, confusion matrix)
#    also saves confusion_matrix.png, feature_importance.png, and
#    trend_distribution.png to model/
python -m ml.evaluate

# 5. (optional) Evaluate the generative-AI text outputs
python -m ai.evaluate_generation
```

## Run the app

```bash
uvicorn backend.main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser — it serves the
dashboard directly. You can click any stock to see:

- A **price chart** (close, SMA5, SMA10) and a **volume chart**, with a
  date-range picker (1M / 3M / 6M / All presets, or pick exact dates) to
  inspect any period of the stock's history.
- A **confidence gauge** showing the Bearish / Neutral / Bullish
  probability split.
- A prominent **investment verdict banner** (🟢 good time to invest /
  🟡 watch or wait / 🔴 avoid for now) — a plain-language recommendation
  built from the model's trend, confidence, and current volatility.
- A plain-language **explanation** of what drove the signal, an
  **insight summary**, and a **chat box** to ask the assistant free-text
  questions like *"Should I invest in TCS right now?"*.

### Static evaluation visualizations

Running `python -m ml.evaluate` saves three PNGs to `model/`:

- `confusion_matrix.png` — actual vs predicted trend, per class
- `feature_importance.png` — which indicators the model relies on most
- `trend_distribution.png` — class balance across the full dataset

### API reference

| Method | Endpoint              | Description                                   |
|--------|------------------------|------------------------------------------------|
| GET    | `/stocks`              | List all tracked stocks + latest close price   |
| GET    | `/predict/{symbol}`    | Raw model output (trend, confidence, features) |
| GET    | `/insight/{symbol}`    | Prediction + explanation + summary + verdict   |
| GET    | `/history/{symbol}`    | OHLCV + indicator series, optional `?start=&end=` date range |
| POST   | `/assistant/ask`       | `{"message": "..."}` → conversational reply + verdict |

## Notes

- The classifier uses 5 engineered features: `sma5`, `sma10`, `rsi14`,
  `daily_return`, `volatility10`. Extend `FEATURE_COLUMNS` in
  `ml/preprocessing.py` to add more (e.g. MACD, Bollinger Bands, sector
  momentum).
- The investment verdict (`ai/insight_generator.py::build_verdict`) is a
  simple, transparent rule layer on top of the model's own probabilities
  and current volatility — not a second model — so it stays easy to audit
  and explain. Tune `CONFIDENT_THRESHOLD` / `HIGH_RISK_MULTIPLIER` there.
- The generative layer (`ai/insight_generator.py`) is template-based and
  fully offline. It's deliberately written with the same call shape a
  real LLM integration would use — set `USE_LLM = True` and implement
  `_call_llm()` / `_call_llm_summary()` to swap in OpenAI or a local
  Ollama model without touching `backend/main.py`.
- This is a college project built on synthetic data. Predictions are
  illustrative only and are **not financial advice**.
